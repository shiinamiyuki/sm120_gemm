// FP8 (e4m3) GEMM with per-tensor scaling: harness, reference, and the
// cuBLASLt baseline. No hand-written FP8 kernel yet — this establishes the
// ground truth and the number to beat.
//
//   Y = (x_scale * X) @ (w_scale * W)^T
//   X: M x K row-major e4m3, W: N x K row-major e4m3, Y: M x N row-major bf16
#include "fp8_harness.h"
#include "fp8_gemm.cuh"

#include <array>

using Shape = std::array<int, 3>;

// Our hand-written FP8 kernel, as a table of instantiations. BK is pinned to
// 128 (a 128-byte smem row for the 128B swizzle), so a stage costs the same
// shared memory as the bf16 kernel's BK=64 while covering twice the k.
template <int BM, int BN, int BK, int S, int CWG, int WM, int WN>
static void fp8_entry(int M, int N, int K, const fp8e4m3 *X, const fp8e4m3 *W,
                      bf16 *Y, float scale, cudaStream_t stream) {
    FP8GemmMMA<BM, BN, BK, S, CWG, WM, WN>::run(M, N, K, X, W, Y, scale, stream);
}

struct Variant {
    const char *name;
    void (*fn)(int, int, int, const fp8e4m3 *, const fp8e4m3 *, bf16 *, float, cudaStream_t);
    int bm, bn, bk;
    size_t smem;
};

#define VARIANT(BM, BN, BK, S, CWG, WM, WN)                                        \
    Variant {                                                                      \
        #BM "x" #BN "x" #BK "_s" #S "_cwg" #CWG "_w" #WM "x" #WN,                   \
        &fp8_entry<BM, BN, BK, S, CWG, WM, WN>, BM, BN, BK,                        \
        (size_t)FP8GemmMMA<BM, BN, BK, S, CWG, WM, WN>::SMEM_SIZE                  \
    }

static const std::vector<Variant> kVariants = {
    VARIANT(128, 128, 128, 2, 2, 64, 32),
    VARIANT(128, 128, 128, 2, 2, 32, 64),
    VARIANT(128, 64, 128, 3, 2, 32, 32),
    VARIANT(64, 128, 128, 3, 2, 16, 64),
    VARIANT(64, 64, 128, 4, 2, 16, 32),
    VARIANT(64, 64, 128, 4, 2, 32, 16),
};


static const std::vector<Shape> kShapes = {
    {256, 256, 256},         // small: easy to eyeball if something is wrong
    {1024, 1024, 1024},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {4096, 14336 * 2, 4096}, // llama3-8b upgate
    {4096, 4096, 14336},     // llama3-8b downproj
    {128, 14336 * 2, 4096},  // upgate, batch 128
    {128, 4096, 14336},      // downproj, batch 128
};

// Pins down the m16n8k32 e4m3 fragment layout in isolation: one MMA, one
// warp, fragments packed straight from global memory by the mapping the
// kernel assumes, checked against a CPU dot product. Small integers keep the
// e4m3 codes and the fp32 sums exact, so any mismatch is a layout error and
// nothing else. This is what the layout documented in fp8_gemm.cuh rests on.
__device__ __forceinline__ uint32_t probe_pack4(const fp8e4m3 *src, int row, int k0, int stride) {
    uint32_t r = 0;
    for (int j = 0; j < 4; j++)
        r |= (uint32_t) * reinterpret_cast<const unsigned char *>(&src[row * stride + k0 + j]) << (8 * j);
    return r;
}

__global__ void probe_layout_kernel(const fp8e4m3 *A, const fp8e4m3 *B, float *D) {
    int lane = threadIdx.x, group = lane >> 2, lig = lane & 3;
    uint32_t a[4], b[2];
    a[0] = probe_pack4(A, group,     lig * 4,      32);
    a[1] = probe_pack4(A, group + 8, lig * 4,      32);
    a[2] = probe_pack4(A, group,     lig * 4 + 16, 32);
    a[3] = probe_pack4(A, group + 8, lig * 4 + 16, 32);
    b[0] = probe_pack4(B, group, lig * 4,      32);
    b[1] = probe_pack4(B, group, lig * 4 + 16, 32);

    float d[4] = {0, 0, 0, 0};
    mma_m16n8k32_e4m3(d, a, b);

    D[group * 8 + lig * 2]           = d[0];
    D[group * 8 + lig * 2 + 1]       = d[1];
    D[(group + 8) * 8 + lig * 2]     = d[2];
    D[(group + 8) * 8 + lig * 2 + 1] = d[3];
}

static int probe_layout(cudaStream_t stream) {
    const int M = 16, N = 8, K = 32;
    std::vector<float> hA(M * K), hW(N * K);
    srand(7);
    for (auto &v : hA) v = (float)(rand() % 7) - 3.0f;
    for (auto &v : hW) v = (float)(rand() % 7) - 3.0f;
    std::vector<fp8e4m3> qA(hA.size()), qW(hW.size());
    for (size_t i = 0; i < hA.size(); i++) qA[i] = fp8e4m3(hA[i]);
    for (size_t i = 0; i < hW.size(); i++) qW[i] = fp8e4m3(hW[i]);

    CUDABuffer<fp8e4m3> dA(qA.size()), dW(qW.size());
    CUDABuffer<float> dD((size_t)M * N);
    dA.copy_from_host(qA.data(), stream);
    dW.copy_from_host(qW.data(), stream);
    probe_layout_kernel<<<1, 32, 0, stream>>>(dA.data, dW.data, dD.data);
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> hD((size_t)M * N);
    dD.copy_to_host(hD.data(), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float ref = 0;
            for (int k = 0; k < K; k++) ref += hA[m * K + k] * hW[n * K + k];
            if (hD[m * N + n] != ref) {
                if (bad < 4) printf("  (m=%d,n=%d) got %g want %g\n", m, n, hD[m * N + n], ref);
                bad++;
            }
        }
    printf("m16n8k32 e4m3 fragment layout: %s (%d/%d entries wrong)\n",
           bad ? "WRONG" : "CONFIRMED", bad, M * N);
    return bad;
}

static void usage() {
    printf(R"(usage: bench_fp8 [options]

Benchmarks per-tensor-scaled FP8 (e4m3) GEMM via cuBLASLt against a naive
FP32 reference over the same quantized inputs.

options:
  --shape M,N,K     problem shape; repeatable (default: built-in list)
  --quant-error     also report what per-tensor fp8 costs at the output,
                    by running a full-precision reference (allocates fp32
                    copies of both operands)
  --normal          draw inputs from a unit Gaussian instead of uniform[-1,1]
  --outlier         Gaussian with 0.1%% of elements scaled by 1000x, which is
                    what actually defeats a single per-tensor scale
  --fast-accum      enable CUBLASLT_MATMUL_DESC_FAST_ACCUM
  --probe-layout    verify the m16n8k32 e4m3 fragment layout and exit
  --tol F           correctness tolerance (default 0.02)
  --warmup N        timed-loop warmup iterations (default 5)
  --repeat N        timed-loop iterations (default 20)
  -h, --help        this message
)");
}

int main(int argc, char **argv) {
    std::vector<Shape> shapes;
    BenchOptions opt;
    opt.tol = 0.02f; // bf16 output rounding alone is ~0.4%
    bool quant_error = false, fast_accum = false, probe_only = false;
    Dist dist = Dist::Uniform;

    for (int i = 1; i < argc; i++) {
        std::string_view a = argv[i];
        auto need = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires an argument\n", argv[i]);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--shape") {
            Shape s;
            if (sscanf(need(), "%d,%d,%d", &s[0], &s[1], &s[2]) != 3) {
                fprintf(stderr, "bad --shape (want M,N,K)\n");
                return 2;
            }
            shapes.push_back(s);
        }
        else if (a == "--quant-error") quant_error = true;
        else if (a == "--normal") dist = Dist::Normal;
        else if (a == "--outlier") dist = Dist::Outlier;
        else if (a == "--fast-accum") fast_accum = true;
        else if (a == "--probe-layout") probe_only = true;
        else if (a == "--tol") opt.tol = atof(need());
        else if (a == "--warmup") opt.warmup = atoi(need());
        else if (a == "--repeat") opt.repeat = atoi(need());
        else { fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]); return 2; }
    }
    if (shapes.empty()) shapes = kShapes;

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    int device, major = 0, minor = 0, max_smem = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    printf("device: sm_%d%d   fp8 = e4m3 (max %.0f), per-tensor scaling, %s inputs%s\n",
           major, minor, kE4M3Max,
           dist == Dist::Normal ? "gaussian" : dist == Dist::Outlier ? "gaussian+outlier" : "uniform",
           fast_accum ? ", fast-accum" : "");

    if (probe_only) return probe_layout(stream);

    int failures = probe_layout(stream);
    for (auto [M, N, K] : shapes) {
        printf("\n=== M=%d N=%d K=%d ===\n", M, N, K);
        Fp8Problem p(M, N, K, stream, dist);
        AccuracyStats q = p.input_quant_error();
        printf("  scales    x=%.4g  w=%.4g   input quant err: max_abs=%.3g rms_rel=%.2f%%\n",
               p.x_scale(), p.w_scale(), q.max_abs, 100.0 * q.rms_rel);

        Fp8GemmLt lt(M, N, K, p.w_scale_dev(), p.x_scale_dev(), fast_accum);
        if (!lt.supported()) {
            printf("  cuBLASLt: no FP8 algorithm for this shape on sm_%d%d\n", major, minor);
            failures++;
            continue;
        }
        auto run = [&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t s) {
            lt.run(X, W, Y, s);
        };

        CheckResult r = p.check(run, opt.tol);
        printf("  check     vs fp32 reference: abs=%.4g rel=%.4g -> %s\n",
               r.vs_fp32.abs_err, r.vs_fp32.rel_err, r.ok ? "PASS" : "FAIL");
        if (!r.ok) {
            p.print_mismatches(opt.tol);
            failures++;
        }

        if (quant_error) {
            AccuracyStats c = p.quantization_cost_at_output();
            printf("  fp8 cost  exact fp8 result vs fp32 inputs: max_abs=%.4g rms_rel=%.2f%%\n",
                   c.max_abs, 100.0 * c.rms_rel);
        }

        double ms = p.time(run, opt);
        printf("  %-28s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n", "cuBLASLt",
               ms, p.tflops(ms), p.gbps(ms));

        const float scale = p.x_scale() * p.w_scale();
        for (auto &v : kVariants) {
            if (M % v.bm || N % v.bn || K % v.bk) continue;
            if (v.smem > (size_t)max_smem) {
                printf("  %-28s SKIP (needs %.1f KB smem)\n", v.name, v.smem / 1024.0);
                continue;
            }
            auto ours = [&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t s) {
                v.fn(M, N, K, X, W, Y, scale, s);
            };
            CheckResult rv = p.check(ours, opt.tol);
            if (!rv.ok) {
                printf("  %-28s FAIL  %s\n", v.name, rv.reason.c_str());
                p.print_mismatches(opt.tol, 4);
                failures++;
                continue;
            }
            double vms = p.time(ours, opt);
            printf("  %-28s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.2fx cuBLASLt, rel=%.4g)\n",
                   v.name, vms, p.tflops(vms), p.gbps(vms), ms / vms, rv.vs_fp32.rel_err);
        }
    }

    cudaStreamDestroy(stream);
    printf("\n%s\n", failures ? "SOME SHAPES FAILED" : "all shapes passed");
    return failures ? 1 : 0;
}
