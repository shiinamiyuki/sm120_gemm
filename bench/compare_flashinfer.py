# FlashInfer's dense 4-bit GEMM on the shapes our W4A16 kernel targets.
#
# Run our side with the SAME timing mode or the numbers are not comparable:
#
#     GEMM_BENCH_GRAPH=1 ./build/bench_w4a16 \
#         --cache build/autotune_cache_w4a16.txt --jit-cache build/jit_cache
#     python bench/compare_flashinfer.py
#
# Two things are compared here, and only one of them is like-for-like.
#
#   Against W4A4 (--shape on compute-bound sizes): exact. FlashInfer's mm_fp4
#   is NVFP4 x NVFP4, the same numerical contract our W4A4GemmMMA implements.
#
#   Against W4A16 (the default decode shapes): approximate. mm_fp4 quantizes
#   the activations too, so it moves marginally fewer bytes (0.5% at those
#   shapes) and has a weaker contract. FlashInfer's actual W4A16 is MoE-only
#   (b12x_fused_moe, and MXFP4 rather than NVFP4); there is no dense W4A16
#   entry point, so it is the closest available baseline rather than a peer.
#
# Two things have to match our C++ harness or the comparison is meaningless:
#
#  1. CUDA graph timing. Every op is captured into a graph of ITERS calls, so
#     no Python, dispatch or host launch cost lands in the number. (Our harness
#     does the same under GEMM_BENCH_GRAPH=1.)
#
#  2. Buffer rotation. The weights here are 33-66 MB and this card's L2 is
#     96 MB, so replaying one weight tensor back-to-back measures L2, not DRAM
#     -- it reads ~1700 GB/s and flatters everything. We allocate enough
#     independent weight sets to exceed 2x L2 and cycle through them inside the
#     graph, which is what our harness does.
import argparse, torch, flashinfer as fi

DEV = "cuda"
ITERS = 48
REPS = 12

def graph_ms(make_call, nbufs):
    calls = [make_call(i % nbufs) for i in range(ITERS)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:5]:
            c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(REPS):
        e0.record(); g.replay(); e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / ITERS)
    return best

def run_shape(M, N, K, wall, backends, l2_bytes):
    torch.manual_seed(0)
    one = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    alpha = torch.tensor([1.0], device=DEV, dtype=torch.float32)

    w_bytes = N * K // 2 + N * K // 16
    nbufs = max(1, (2 * l2_bytes + w_bytes - 1) // w_bytes)

    X = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) / 8
    Xq, Xsf = fi.nvfp4_quantize(X, one, sf_vec_size=16)
    Ws, Wqs, Wsfs = [], [], []
    for _ in range(nbufs):
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) / 8
        wq, wsf = fi.nvfp4_quantize(W, one, sf_vec_size=16)
        Ws.append(W); Wqs.append(wq.T); Wsfs.append(wsf.T)
    ref = (X.float() @ Ws[0].float().T)
    refmag = ref.abs().mean().item()

    x4 = M * K // 2 + M * K // 16
    nb = w_bytes + x4 + M * N * 2
    flops = 2.0 * M * N * K
    print(f"\n=== M={M} N={N} K={K} ===  weights {w_bytes/2**20:.1f} MB x {nbufs} sets "
          f"({nbufs*w_bytes/2**20:.0f} MB vs {l2_bytes/2**20:.0f} MB L2)")
    for be in backends:
        try:
            out = fi.mm_fp4(Xq, Wqs[0], Xsf, Wsfs[0], alpha=alpha, out_dtype=torch.bfloat16,
                            block_size=16, backend=be)
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().mean().item() / refmag
            def mk(i, be=be):
                return lambda: fi.mm_fp4(Xq, Wqs[i], Xsf, Wsfs[i], alpha=alpha,
                                         out_dtype=torch.bfloat16, block_size=16, backend=be)
            ms = graph_ms(mk, nbufs)
            bw = nb / (ms * 1e-3) / 1e9
            tf = flops / (ms * 1e-3) / 1e12
            print(f"  mm_fp4 [{be:8s}] {ms:8.4f} ms  {tf:8.1f} TFLOPS  {bw:8.1f} GB/s"
                  f"  {100*bw/wall:4.0f}% of wall   rel_err={err:.4f}")
        except Exception as e:
            print(f"  mm_fp4 [{be:8s}] FAILED  {type(e).__name__}: {str(e)[:90]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=float, default=1707.0)
    ap.add_argument("--backends", default="b12x,cutlass,cudnn")
    ap.add_argument("--shape", action="append", default=None,
                    help="M,N,K; repeatable. Defaults to the W4A16 decode set.")
    a = ap.parse_args()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    print(torch.cuda.get_device_name(0), "| flashinfer", fi.__version__, "| torch", torch.__version__)
    print(f"L2 = {l2/2**20:.0f} MB, bandwidth wall = {a.wall:.0f} GB/s")
    print(f"timing: CUDA graph of {ITERS} calls rotating over >2x L2 of weights, best of {REPS}")
    shapes = ([tuple(int(v) for v in s.split(",")) for s in a.shape] if a.shape
              else [(1,4096,14336),(1,28672,4096),(4,4096,14336),
                    (4,28672,4096),(8,4096,14336),(8,28672,4096)])
    for (M, N, K) in shapes:
        run_shape(M, N, K, a.wall, a.backends.split(","), l2)
