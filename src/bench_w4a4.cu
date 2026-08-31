// W4A4: NVFP4 activations x NVFP4 weights (e2m1 data + one ue4m3 per 16
// elements along K + an fp32 global scale on each side).
//
// Unlike W4A16, this one has a native instruction: sm_120a can feed packed
// e2m1 straight into the tensor core with per-16 block scales, so there is no
// dequantize step anywhere in the mainloop. What has to be established before
// a kernel can be written is exactly where that instruction expects its
// operands and its scales to sit in the warp's registers -- which is what
// --probe-mma does. See src/nvfp4_mma.cuh for the conclusions.
//
// --selfcheck covers the other half of the groundwork: that the problem setup
// quantizes both operands correctly and that the swizzled scale tensors a
// kernel will be handed decode to the same values as the row-major ones the
// fp32 reference uses. Both probes are deliberately falsifiable -- perturbing
// the scale lane mapping or shifting a swizzled read by one k-block makes them
// fail loudly, which is the only reason to trust them when they pass.
#include "nvfp4_mma.cuh"
#include "w4a4_harness.h" // W4A4Problem, CUDABuffer, BenchOptions
#include "w4a4_gemm.cuh"
#include "kernel_jit.h"

#include <optional>

#include <array>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Probe 1: the scale fragment ────────────────────────────────────────
//
// A is all 1.0 and B is 1.0 in exactly one k-block and 0 elsewhere, so
// D[m][n] = 16 * sfa(m, kb) * sfb(n, kb) -- one 16-element block's worth.
// Every scale byte in the warp is 1.0 except one, set to 2.0; the rows (or
// columns) that come out 32 instead of 16 are the ones that byte feeds, and
// they only do so when its byte index matches the live k-block. Sweeping
// lane x byte x k-block x side maps the fragment completely, including what a
// byte *means*, which a uniform-B probe cannot see.
__global__ void probe_scale_kernel(float *D, const uint32_t *sfa, const uint32_t *sfb,
                                   int kblock, int tid_sel) {
    const int lane = threadIdx.x, q = lane & 3;
    // b[0] covers k = q*8 + 0..7 -> k-block (q >= 2); b[1] the same at k + 32.
    const int hi = (q >= 2) ? 1 : 0;
    uint32_t a[4] = {kE2M1_Ones, kE2M1_Ones, kE2M1_Ones, kE2M1_Ones};
    uint32_t b[2] = {0, 0};
    if (kblock < 2) { if (hi == (kblock & 1)) b[0] = kE2M1_Ones; }
    else            { if (hi == (kblock & 1)) b[1] = kE2M1_Ones; }

    float d[4] = {0, 0, 0, 0};
    if (tid_sel == 0) mma_m16n8k64_e2m1_block_scaled<0>(d, a, b, sfa[lane], sfb[lane]);
    else              mma_m16n8k64_e2m1_block_scaled<1>(d, a, b, sfa[lane], sfb[lane]);

    const int g = lane >> 2, l = lane & 3;
    D[g * 8 + l * 2]           = d[0];
    D[g * 8 + l * 2 + 1]       = d[1];
    D[(g + 8) * 8 + l * 2]     = d[2];
    D[(g + 8) * 8 + l * 2 + 1] = d[3];
}

static int probe_scale_fragment(cudaStream_t stream, int tid_sel) {
    CUDABuffer<float> dD(16 * 8);
    CUDABuffer<uint32_t> dsa(32), dsb(32);
    const std::vector<uint32_t> ones(32, kUE4M3_Ones);
    std::vector<uint32_t> sa(32), sb(32);
    std::vector<float> hD(16 * 8);
    int bad = 0;

    for (int side = 0; side < 2; side++)       // 0 = A scale (rows), 1 = B (cols)
        for (int kb = 0; kb < 4; kb++)
            for (int lane = 0; lane < 32; lane++)
                for (int byte = 0; byte < 4; byte++) {
                    sa = ones;
                    sb = ones;
                    std::vector<uint32_t> &probed = side ? sb : sa;
                    // 0x40 is ue4m3 2.0, 0x38 is 1.0.
                    probed[lane] = (ones[lane] & ~(0xffu << (8 * byte))) | (0x40u << (8 * byte));
                    dsa.copy_from_host(sa.data(), stream);
                    dsb.copy_from_host(sb.data(), stream);
                    CHECK_CUDA(cudaMemsetAsync(dD.data, 0, 16 * 8 * sizeof(float), stream));
                    probe_scale_kernel<<<1, 32, 0, stream>>>(dD.data, dsa.data, dsb.data,
                                                             kb, tid_sel);
                    CHECK_CUDA(cudaGetLastError());
                    dD.copy_to_host(hD.data(), stream);
                    CHECK_CUDA(cudaStreamSynchronize(stream));

                    const int fed = side ? nvfp4_sfb_col_for_lane(lane, tid_sel)
                                         : nvfp4_sfa_row_for_lane(lane, tid_sel);
                    for (int m = 0; m < 16; m++)
                        for (int n = 0; n < 8; n++) {
                            const int idx = side ? n : m;
                            const bool doubled = (fed == idx) && (byte == kb);
                            const float expect = doubled ? 32.0f : 16.0f;
                            if (hD[m * 8 + n] != expect) {
                                if (bad < 8)
                                    printf("  %s-scale tid=%d kb=%d lane=%d byte=%d: "
                                           "D[%d][%d]=%g want %g\n",
                                           side ? "B" : "A", tid_sel, kb, lane, byte,
                                           m, n, hD[m * 8 + n], expect);
                                bad++;
                            }
                        }
                }
    printf("  scale fragment (thread-id %d): %s (%d checks wrong)\n",
           tid_sel, bad ? "WRONG" : "CONFIRMED", bad);
    return bad;
}

// ── Probe 2: the operand fragments ─────────────────────────────────────
//
// Random A[16][64], B[64][8] and per-16 scales drawn from the exact e2m1 /
// ue4m3 grids, packed per the layout in nvfp4_mma.cuh, against an fp64
// reference. This is what validates the a[]/b[] nibble order and the
// accumulator mapping; the scale probe above cannot see either.
__global__ void probe_operand_kernel(const uint32_t *A, const uint32_t *B,
                                     const uint32_t *SFA, const uint32_t *SFB,
                                     float *D, int tid_sel) {
    const int l = threadIdx.x;
    uint32_t a[4] = {A[l * 4 + 0], A[l * 4 + 1], A[l * 4 + 2], A[l * 4 + 3]};
    uint32_t b[2] = {B[l * 2 + 0], B[l * 2 + 1]};
    float d[4] = {0, 0, 0, 0};
    if (tid_sel == 0) mma_m16n8k64_e2m1_block_scaled<0>(d, a, b, SFA[l], SFB[l]);
    else              mma_m16n8k64_e2m1_block_scaled<1>(d, a, b, SFA[l], SFB[l]);
    for (int i = 0; i < 4; i++) D[l * 4 + i] = d[i];
}

static int probe_operand_fragment(cudaStream_t stream, int tid_sel) {
    srand(1234 + tid_sel);
    int Ac[16][64], Bc[64][8];
    unsigned SAc[16][4], SBc[8][4];
    for (int m = 0; m < 16; m++) for (int k = 0; k < 64; k++) Ac[m][k] = rand() % 16;
    for (int k = 0; k < 64; k++) for (int n = 0; n < 8; n++) Bc[k][n] = rand() % 16;
    // 0x38/0x40/0x48 are ue4m3 1, 2, 4 -- exact, so the reference is exact.
    for (int m = 0; m < 16; m++) for (int j = 0; j < 4; j++) SAc[m][j] = 0x38 + ((rand() % 3) << 3);
    for (int n = 0; n < 8; n++)  for (int j = 0; j < 4; j++) SBc[n][j] = 0x38 + ((rand() % 3) << 3);

    std::vector<uint32_t> hA(32 * 4, 0), hB(32 * 2, 0), hSA(32, 0), hSB(32, 0);
    for (int l = 0; l < 32; l++) {
        const int g = l >> 2, q = l & 3;
        for (int r = 0; r < 2; r++)
            for (int kg = 0; kg < 2; kg++) {
                const int row = g + r * 8, kbase = kg * 32 + q * 8;
                uint32_t w = 0;
                for (int i = 0; i < 8; i++) w |= (uint32_t)(Ac[row][kbase + i] & 0xF) << (4 * i);
                hA[l * 4 + kg * 2 + r] = w;
            }
        for (int kg = 0; kg < 2; kg++) {
            const int kbase = kg * 32 + q * 8;
            uint32_t w = 0;
            for (int i = 0; i < 8; i++) w |= (uint32_t)(Bc[kbase + i][g] & 0xF) << (4 * i);
            hB[l * 2 + kg] = w;
        }
    }
    for (int m = 0; m < 16; m++) {
        uint32_t w = 0;
        for (int j = 0; j < 4; j++) w |= SAc[m][j] << (8 * j);
        hSA[nvfp4_sfa_lane_for_row(m) + 2 * tid_sel] = w;
    }
    for (int n = 0; n < 8; n++) {
        uint32_t w = 0;
        for (int j = 0; j < 4; j++) w |= SBc[n][j] << (8 * j);
        hSB[nvfp4_sfb_lane_for_col(n) + tid_sel] = w;
    }

    CUDABuffer<uint32_t> dA(32 * 4), dB(32 * 2), dSA(32), dSB(32);
    CUDABuffer<float> dD(32 * 4);
    dA.copy_from_host(hA.data(), stream);
    dB.copy_from_host(hB.data(), stream);
    dSA.copy_from_host(hSA.data(), stream);
    dSB.copy_from_host(hSB.data(), stream);
    probe_operand_kernel<<<1, 32, 0, stream>>>(dA.data, dB.data, dSA.data, dSB.data,
                                               dD.data, tid_sel);
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> hD(128);
    dD.copy_to_host(hD.data(), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    double ref[16][8];
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            double s = 0;
            for (int k = 0; k < 64; k++)
                s += (double)e2m1_to_float(Ac[m][k]) * ue4m3_to_float(SAc[m][k / 16]) *
                     (double)e2m1_to_float(Bc[k][n]) * ue4m3_to_float(SBc[n][k / 16]);
            ref[m][n] = s;
        }

    int bad = 0;
    double worst = 0;
    for (int l = 0; l < 32; l++)
        for (int h = 0; h < 2; h++)
            for (int e = 0; e < 2; e++) {
                const int m = (l >> 2) + 8 * h, n = (l & 3) * 2 + e;
                const double got = hD[l * 4 + h * 2 + e], want = ref[m][n];
                const double err = fabs(got - want) / fmax(1.0, fabs(want));
                if (err > 1e-5) {
                    if (bad < 8)
                        printf("  operand tid=%d: D[%d][%d]=%g want %g\n", tid_sel, m, n, got, want);
                    bad++;
                    worst = fmax(worst, err);
                }
            }
    printf("  operand fragment (thread-id %d): %s (%d/128 wrong, worst rel %.3g)\n",
           tid_sel, bad ? "WRONG" : "CONFIRMED", bad, worst);
    return bad;
}

// ── Harness self-check ─────────────────────────────────────────────────
//
// No kernel exists yet, so what can be validated is the problem setup: that
// quantizing both operands to NVFP4 and reading them back costs what the
// recipe says, and -- the part a kernel actually depends on -- that the
// swizzled scale tensors it will be handed decode to the same values as the
// row-major ones the fp32 reference used. Those two paths read identical
// bytes from different addresses, so the difference must be exactly zero.
using Shape = std::array<int, 3>;
static const std::vector<Shape> kSelfcheckShapes = {
    {16, 256, 256}, {8, 4096, 2048}, {128, 512, 1024}, {256, 256, 4096},
};

static int cmd_selfcheck(cudaStream_t stream) {
    cublasHandle_t cub;
    CHECK_CUBLAS(cublasCreate(&cub));
    CHECK_CUBLAS(cublasSetStream(cub, stream));
    BenchOptions opt;
    int bad = 0;
    printf("%-18s %10s %10s %12s %12s\n", "M,N,K", "X rms", "W rms", "swizzle max", "bf16 base");
    for (auto &sh : kSelfcheckShapes) {
        W4A4Problem p(sh[0], sh[1], sh[2], stream);
        const double worst = p.selfcheck_swizzled_scales();
        const double base_ms = p.time_bf16_baseline(cub, opt);
        char name[32];
        snprintf(name, sizeof(name), "%d,%d,%d", sh[0], sh[1], sh[2]);
        printf("%-18s %9.3f%% %9.3f%% %12.3g %9.4f ms%s\n", name,
               100.0 * p.x_quant_error().rms_rel, 100.0 * p.w_quant_error().rms_rel,
               worst, base_ms, worst == 0.0 ? "" : "   <-- SWIZZLE MISMATCH");
        if (worst != 0.0) bad++;
    }
    cublasDestroy(cub);
    printf("%s\n", bad ? "HARNESS SELF-CHECK FAILED"
                        : "harness self-check passed (swizzled scales agree exactly)");
    return bad ? 1 : 0;
}

// ── Direct kernel test ─────────────────────────────────────────────────
//
// A handful of configs instantiated straight into this TU, before any JIT or
// autotune plumbing exists. The point is correctness: does the mainloop agree
// with the fp32 reference over the same quantized operands?
template <int BM, int BN, int NS, int CWG, int WM, int WN, int YS = 1>
static int run_one(int M, int N, int K, cudaStream_t stream, cublasHandle_t cub) {
    using G = W4A4GemmMMA<BM, BN, 256, NS, CWG, WM, WN, YS>;
    if (M % BM || N % BN || K % 256) return 0; // not this config's shape
    int max_smem = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    char name[96];
    snprintf(name, sizeof(name), "w4a4_%dx%dx256_s%d_cwg%d_w%dx%d_ys%d", BM, BN, NS, CWG, WM, WN, YS);
    if (G::SMEM_SIZE > max_smem) {
        printf("  %-40s smem %d KB > %d KB, skipped\n", name, G::SMEM_SIZE >> 10, max_smem >> 10);
        return 0;
    }
    W4A4Problem p(M, N, K, stream);
    BenchOptions opt;
    const float out_scale = p.x_global() * p.w_global();
    auto call = [&](const unsigned char *X, const unsigned char *W, bf16 *Y, cudaStream_t s) {
        G::run(M, N, K, X, W, Y, (const unsigned char *)p.x_sf(),
               (const unsigned char *)p.w_sf(), out_scale, s);
    };
    CheckResult r = p.check(call, 0.06f);
    if (!r.ok) {
        printf("  %-40s FAIL  %s\n", name, r.reason.c_str());
        p.print_mismatches(0.06f, 4);
        return 1;
    }
    const double ms = p.time(call, opt);
    const double base = p.time_bf16_baseline(cub, opt);
    printf("  %-40s ok  rel_err %.4f  %8.4f ms  %7.1f TFLOPS  (%.2fx bf16)\n",
           name, r.vs_fp32.rel_err, ms, p.tflops(ms), base / ms);
    return 0;
}

static std::vector<Shape> g_test_shapes = {{256, 256, 512}, {512, 512, 1024}, {1024, 1024, 2048}};

static int cmd_test(cudaStream_t stream) {
    cublasHandle_t cub;
    CHECK_CUBLAS(cublasCreate(&cub));
    CHECK_CUBLAS(cublasSetStream(cub, stream));
    int bad = 0;
    for (auto &sh : g_test_shapes) {
        printf("=== M=%d N=%d K=%d ===\n", sh[0], sh[1], sh[2]);
        bad += run_one<64, 64, 2, 2, 32, 16>(sh[0], sh[1], sh[2], stream, cub);
        bad += run_one<128, 64, 2, 2, 32, 32>(sh[0], sh[1], sh[2], stream, cub);
        bad += run_one<128, 128, 2, 2, 64, 32>(sh[0], sh[1], sh[2], stream, cub);
        bad += run_one<128, 128, 2, 2, 64, 32, 2>(sh[0], sh[1], sh[2], stream, cub);
        bad += run_one<128, 128, 2, 2, 64, 32, 4>(sh[0], sh[1], sh[2], stream, cub);
    }
    cublasDestroy(cub);
    printf("%s\n", bad ? "W4A4 KERNEL TEST FAILED" : "w4a4 kernel matches the fp32 reference");
    return bad ? 1 : 0;
}

// ── JIT-backed bench / autotune ────────────────────────────────────────

static auto bind(const W4A4CompiledKernel &kern, W4A4Problem &p, float out_scale) {
    return [&, out_scale](const unsigned char *X, const unsigned char *W, bf16 *Y,
                          cudaStream_t s) {
        kern.fn(p.M(), p.N(), p.K(), X, W, Y, p.x_sf(), p.w_sf(), out_scale,
                p.workspace(), s);
    };
}

static std::vector<GemmConfig> enumerate_configs(size_t max_smem) {
    return GemmConfig::enumerate(max_smem, ElemType::W4A4);
}

struct ShapeRun {
    W4A4Problem p;
    cublasHandle_t cublas;
    double bf16_ms = 0.0;
    float out_scale;

    ShapeRun(int M, int N, int K, cudaStream_t stream, const BenchOptions &opt)
        : p(M, N, K, stream), out_scale(p.x_global() * p.w_global()) {
        CHECK_CUBLAS(cublasCreate(&cublas));
        CHECK_CUBLAS(cublasSetStream(cublas, stream));
        const AccuracyStats x = p.x_quant_error(), w = p.w_quant_error();
        printf("  quant err   X nvfp4: rms_rel=%.2f%%   W nvfp4: rms_rel=%.2f%%\n",
               100.0 * x.rms_rel, 100.0 * w.rms_rel);
        printf("  bytes       w4a4 %.1f MB vs bf16 %.1f MB  (%.2fx less)\n",
               p.bytes() / 1e6, p.bf16_bytes() / 1e6, p.bf16_bytes() / p.bytes());
        bf16_ms = p.time_bf16_baseline(cublas, opt);
        printf("  %-38s %8.4f ms  %9.2f TFLOPS\n", "bf16 cuBLAS (unquantized)",
               bf16_ms, p.tflops(bf16_ms));
    }
    ~ShapeRun() { cublasDestroy(cublas); }
};

static double run_one(ShapeRun &s, const W4A4CompiledKernel &kern, const BenchOptions &opt) {
    auto fn = bind(kern, s.p, s.out_scale);
    const std::string name = kern.config.name();
    if (opt.check) {
        CheckResult r = s.p.check(fn, opt.tol);
        if (!r.ok) {
            printf("  %-38s FAIL  %s\n", name.c_str(), r.reason.c_str());
            s.p.print_mismatches(opt.tol, 4);
            return 0.0;
        }
    }
    const double ms = s.p.time(fn, opt);
    printf("  %-38s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.2fx bf16)\n",
           name.c_str(), ms, s.p.tflops(ms), s.p.gbps(ms), s.bf16_ms / ms);
    return ms;
}

static W4A4KernelJit make_jit(const std::string &cache_dir, int jobs) {
    JitOptions o = w4a4_jit_options();
    o.nvcc = GEMM_NVCC;
    o.src_dir = GEMM_SRC_DIR;
    o.cache_dir = cache_dir;
    o.cuda_stub_dir = GEMM_CUDA_STUB_DIR;
    o.arch = GEMM_CUDA_ARCH;
    o.jobs = jobs;
    return W4A4KernelJit(std::move(o));
}

static int cmd_bench(cudaStream_t stream, const std::string &cache_path,
                     const std::string &jit_dir, int jobs, const BenchOptions &opt,
                     const std::optional<GemmConfig> &forced) {
    AutotuneCache cache;
    if (!forced && !cache.load(cache_path)) {
        printf("no autotune cache at %s\nhint: bench_w4a4 --autotune\n", cache_path.c_str());
        return 1;
    }
    auto jit = make_jit(jit_dir, jobs);
    int bad = 0;
    for (auto &sh : g_test_shapes) {
        GemmConfig cfg;
        if (forced) cfg = *forced;
        else if (auto e = cache.lookup(sh[0], sh[1], sh[2])) cfg = e->config;
        else { printf("=== M=%d N=%d K=%d ===  no cached config, skipped\n",
                      sh[0], sh[1], sh[2]); continue; }
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", sh[0], sh[1], sh[2],
               cfg.name().c_str());
        if (std::string err = cfg.validate(); !err.empty()) {
            printf("  invalid config: %s\n", err.c_str());
            bad++;
            continue;
        }
        const W4A4CompiledKernel *kern = jit.get(cfg);
        if (!kern) { printf("  compile failed\n"); bad++; continue; }
        ShapeRun s(sh[0], sh[1], sh[2], stream, opt);
        if (run_one(s, *kern, opt) == 0.0) bad++;
    }
    printf("\n%s\n", bad ? "SOME SHAPES FAILED" : "all shapes passed");
    return bad ? 1 : 0;
}

static int cmd_autotune(cudaStream_t stream, const std::string &cache_path,
                        const std::string &jit_dir, int jobs, const BenchOptions &opt,
                        size_t max_smem) {
    AutotuneCache cache;
    cache.load(cache_path);
    auto jit = make_jit(jit_dir, jobs);
    const auto all = enumerate_configs(max_smem);
    for (auto &sh : g_test_shapes) {
        printf("\n=== Autotune M=%d N=%d K=%d ===\n", sh[0], sh[1], sh[2]);
        std::vector<GemmConfig> cands;
        for (auto &c : all)
            if (c.fits_shape(sh[0], sh[1], sh[2])) cands.push_back(c);
        if (cands.empty()) { printf("  no config fits this shape\n"); continue; }
        auto kernels = jit.get_many(cands);
        ShapeRun s(sh[0], sh[1], sh[2], stream, opt);
        const W4A4CompiledKernel *best = nullptr;
        double best_ms = 1e30;
        for (auto *k : kernels) {
            const double ms = run_one(s, *k, opt);
            if (ms > 0.0 && ms < best_ms) { best_ms = ms; best = k; }
        }
        if (!best) { printf("  nothing passed\n"); continue; }
        printf("  BEST: %s  %.4f ms  %.2f TFLOPS  (%.2fx bf16)\n",
               best->config.name().c_str(), best_ms, s.p.tflops(best_ms),
               s.bf16_ms / best_ms);
        cache.store(sh[0], sh[1], sh[2], best->config, best_ms);
        cache.save(cache_path);
    }
    printf("\nAutotune complete -> %s\n", cache_path.c_str());
    return 0;
}

static int cmd_probe_mma(cudaStream_t stream) {
    printf("NVFP4 block-scaled MMA "
           "(mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X\n"
           "                       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3)\n");
    int bad = 0;
    for (int tid = 0; tid < 2; tid++) bad += probe_operand_fragment(stream, tid);
    for (int tid = 0; tid < 2; tid++) bad += probe_scale_fragment(stream, tid);
    printf("%s\n", bad ? "MMA LAYOUT PROBE FAILED" : "MMA layout matches src/nvfp4_mma.cuh");
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    bool probe_mma = false, selfcheck = false, test = false, shape_set = false;
    bool bench = false, autotune = false, list_configs = false;
    std::string cache_path = "autotune_cache_w4a4.txt", jit_dir = GEMM_JIT_CACHE_DIR;
    int jobs = 0;
    std::optional<GemmConfig> forced;
    BenchOptions opt;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printf(R"(usage: bench_w4a4 [options]

NVFP4 activations x NVFP4 weights. Unlike W4A16 this has a native instruction
on sm_120a -- packed e2m1 goes straight into the tensor core with per-16 ue4m3
block scales -- so there is no dequantize step in the mainloop.

Only the MMA layout probe exists so far; the kernel is not written yet.

options:
  --probe-mma   re-derive the block-scaled MMA operand and scale fragment
                layouts against the hardware, and check them against the
                mapping src/nvfp4_mma.cuh documents
  --selfcheck   quantize both operands over a few shapes, report what that
                costs, and verify the swizzled scale tensors decode identically
                to the row-major ones the fp32 reference uses
  --test        instantiate a few tensor-core W4A4 configs directly and check
                them against the fp32 reference
  --shape M,N,K use this shape instead of the built-in list; repeatable.
                Selects the shapes only -- the mode is --bench (the default),
                --test or --autotune.
  --bench       bench every shape with its cached config (the default mode)
  --autotune    sweep the configuration space and cache the winner
  --config CFG  use exactly this configuration, e.g.
                w4a4_128x128x256_s2_cwg2_w64x32_ys2_sk1
  --list-configs  print the configuration space
  --cache PATH  autotune cache file (default autotune_cache_w4a4.txt)
  --jit-cache DIR  compiled-kernel cache directory
  --jobs N      parallel nvcc jobs
  --no-check    skip the fp32 correctness check (its reference is O(M*N*K) on
                CUDA cores and dominates a large-shape sweep)
  -h, --help    this message
)");
            return 0;
        } else if (arg == "--probe-mma") {
            probe_mma = true;
        } else if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--test") {
            test = true;
        } else if (arg == "--bench") {
            bench = true;
        } else if (arg == "--autotune") {
            autotune = true;
        } else if (arg == "--list-configs") {
            list_configs = true;
        } else if (arg == "--config" && i + 1 < argc) {
            std::string err;
            forced = GemmConfig::parse(argv[++i], &err);
            if (!forced) { fprintf(stderr, "bad --config: %s\n", err.c_str()); return 2; }
            forced->elem = ElemType::W4A4;
            forced->family = KernelFamily::Mma;
            if (forced->warp_m == 0 && !forced->derive_warp_tile()) {
                fprintf(stderr, "no warp tile covers %dx%d with %d warps\n",
                        forced->bm, forced->bn, forced->cwg * 4);
                return 2;
            }
            bench = true;
        } else if (arg == "--cache" && i + 1 < argc) {
            cache_path = argv[++i];
        } else if (arg == "--jit-cache" && i + 1 < argc) {
            jit_dir = argv[++i];
        } else if (arg == "--jobs" && i + 1 < argc) {
            jobs = atoi(argv[++i]);
        } else if (arg == "--no-check") {
            opt.check = false;
        } else if (arg == "--shape" && i + 1 < argc) {
            int m, n, k;
            if (sscanf(argv[++i], "%d,%d,%d", &m, &n, &k) != 3) {
                fprintf(stderr, "bad --shape (want M,N,K)\n");
                return 2;
            }
            if (!shape_set) { g_test_shapes.clear(); shape_set = true; }
            g_test_shapes.push_back({m, n, k});
        } else {
            fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]);
            return 2;
        }
    }

    int dev = 0, major = 0, minor = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    printf("device: sm_%d%d\n", major, minor);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    int rc = 0;
    if (probe_mma) rc |= cmd_probe_mma(stream);
    if (selfcheck) rc |= cmd_selfcheck(stream);
    if (test) rc |= cmd_test(stream);
    size_t max_smem = 0;
    {
        int v = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        max_smem = (size_t)v;
    }
    if (list_configs) {
        const auto cfgs = enumerate_configs(max_smem);
        for (auto &c : cfgs)
            printf("%-46s smem %6.1f KB\n", c.name().c_str(), c.smem_bytes() / 1024.0);
        printf("%zu configs (max smem %.0f KB)\n", cfgs.size(), max_smem / 1024.0);
    }
    if (autotune) rc |= cmd_autotune(stream, cache_path, jit_dir, jobs, opt, max_smem);
    else if (bench) rc |= cmd_bench(stream, cache_path, jit_dir, jobs, opt, forced);
    else if (!probe_mma && !selfcheck && !test && !list_configs)
        rc |= cmd_bench(stream, cache_path, jit_dir, jobs, opt, forced);
    CHECK_CUDA(cudaStreamDestroy(stream));
    return rc;
}
