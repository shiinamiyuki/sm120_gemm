// Prototype driver for the tiny-M CUDA-core kernel (src/bf16_gemm_tinym.cuh).
//
// Not wired into the autotuner yet: this compiles a fixed set of variants
// directly and runs each through the normal bench harness (cuBLAS + naive
// fp32 references, L2-flushing timing) so correctness and bandwidth can be
// judged before committing to a configuration space.
#include "bench_harness.h"
#include "bf16_gemm_tinym.cuh"

#include <array>

template <int BM, int BN, int BK, int STAGES, int CWG, int SPLIT_K>
static void tinym_entry(int M, int N, int K, const void *X, const void *W, void *Y,
                        float *ws, cudaStream_t stream)
{
    BF16GemmTinyM<BM, BN, BK, STAGES, CWG, SPLIT_K>::run(
        M, N, K, static_cast<const bf16 *>(X), static_cast<const bf16 *>(W),
        static_cast<bf16 *>(Y), ws, stream);
}

struct Variant {
    const char *name;
    GemmKernelFn fn;
    int bm, bn, bk, split_k;
    size_t smem;      // the template's real SMEM_SIZE
    GemmConfig cfg;   // what the autotuner thinks this config is
};

#define VARIANT(BM, BN, BK, S, CWG, SK)                                          \
    Variant                                                                      \
    {                                                                            \
        "tinym_" #BM "x" #BN "x" #BK "_s" #S "_cwg" #CWG "_sk" #SK,              \
            &tinym_entry<BM, BN, BK, S, CWG, SK>, BM, BN, BK, SK,                \
            (size_t)BF16GemmTinyM<BM, BN, BK, S, CWG, SK>::SMEM_SIZE,            \
            GemmConfig{BM, BN, BK, S, CWG, 0, 0, SK, KernelFamily::TinyM}        \
    }


// BM=8 costs exactly the same shared memory as BM=4: X_STAGE_ROWS rounds both
// up to one 8-row swizzle block. It does double the per-thread X fragment
// (xf[BM][8] floats) and the FMA count per k chunk, which is what the M=5..8
// sweep is here to measure.
static const std::vector<Variant> kVariants = {
    // BM  BN   BK  stages cwg split-k
    VARIANT(4, 128, 64, 4, 1, 1),
    VARIANT(4, 128, 64, 4, 1, 4),
    VARIANT(4, 128, 64, 4, 1, 8),
    VARIANT(4, 128, 64, 4, 1, 16),
    VARIANT(4, 128, 64, 3, 1, 8),
    VARIANT(4, 128, 64, 2, 1, 8),
    VARIANT(4, 256, 64, 2, 2, 8),
    VARIANT(4, 256, 64, 2, 1, 8),
    VARIANT(4, 256, 64, 2, 1, 16),
    VARIANT(8, 128, 64, 4, 1, 1),
    VARIANT(8, 128, 64, 4, 1, 4),
    VARIANT(8, 128, 64, 4, 1, 8),
    VARIANT(8, 128, 64, 4, 1, 16),
    VARIANT(8, 128, 64, 3, 1, 8),
    VARIANT(8, 128, 64, 2, 1, 8),
    VARIANT(8, 256, 64, 2, 2, 8),
    VARIANT(8, 256, 64, 2, 1, 8),
    VARIANT(8, 256, 64, 2, 1, 16),
};

// GemmConfig::smem_bytes() gates every launch against the device limit, and
// GemmConfig::name() is the JIT cache key — both are hand-derived host-side.
// Check them against the actual templates so a layout change cannot silently
// desynchronise the autotuner from the kernel.
static int check_config_agrees_with_template()
{
    int bad = 0;
    for (auto &v : kVariants)
    {
        if (v.cfg.smem_bytes() != v.smem)
        {
            printf("MISMATCH %s: smem_bytes()=%zu but SMEM_SIZE=%zu\n",
                   v.name, v.cfg.smem_bytes(), v.smem);
            bad++;
        }
        if (v.cfg.name() != v.name)
        {
            printf("MISMATCH name: config says '%s', variant says '%s'\n",
                   v.cfg.name().c_str(), v.name);
            bad++;
        }
        if (auto why = v.cfg.validate(); !why.empty())
        {
            printf("MISMATCH %s: validate() rejects a working config: %s\n", v.name, why.c_str());
            bad++;
        }
    }
    printf("config/template agreement: %s (%zu variants)\n",
           bad ? "FAILED" : "ok", kVariants.size());
    return bad;
}

using Shape = std::array<int, 3>;
static const std::vector<Shape> kShapes = {
    {1, 128, 64},      // smallest possible: one tile, one k-step
    {4, 128, 128},     // 2 k-tiles: regression guard for the swizzle phase bug
    // M sweep on the llama3-8b downproj shape: 1..4 use BM=4, 5..8 use BM=8.
    {1, 4096, 14336},
    {2, 4096, 14336},
    {4, 4096, 14336},
    {5, 4096, 14336},
    {6, 4096, 14336},
    {7, 4096, 14336},
    {8, 4096, 14336},
    {1, 28672, 4096},  // llama3-8b upgate, batch 1
    {4, 28672, 4096},
    {8, 28672, 4096},
    {4, 4096, 4096},
    {8, 4096, 4096},
};

// Synthetic probes. Each fills X and W from closed-form functions of (row, k)
// so the exact expected output is known, isolating one failure mode at a time:
//   probe A  X=1, W=k  -> is W's k-mapping right? (blind to X)
//   probe B  X=k, W=1  -> is X's k-mapping right? (blind to W)
//   probe C  X=k, W=k  -> do X and W agree on *which* k? sum(k^2) only comes
//                         out right if the two are aligned element-for-element
static void run_probe(const char *label, cudaStream_t stream,
                      float (*fx)(int m, int k), float (*fw)(int n, int k),
                      double expected)
{
    constexpr int M = 4, N = 128, K = 128;
    std::vector<bf16> hX((size_t)M * K), hW((size_t)N * K);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            hX[(size_t)m * K + k] = __float2bfloat16(fx(m, k));
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            hW[(size_t)n * K + k] = __float2bfloat16(fw(n, k));

    CUDABuffer<bf16> dX(hX.size()), dW(hW.size()), dY((size_t)M * N);
    dX.copy_from_host(hX.data(), stream);
    dW.copy_from_host(hW.data(), stream);
    CHECK_CUDA(cudaMemsetAsync(dY.data, 0, (size_t)M * N * sizeof(bf16), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    BF16GemmTinyM<4, 128, 64, 4, 1, 1>::run(M, N, K, dX.data, dW.data, dY.data, nullptr, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<bf16> hY((size_t)M * N);
    dY.copy_to_host(hY.data(), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("  %-22s expected %10.0f   got", label, expected);
    for (int m = 0; m < M; m++)
        printf(" %.0f", __bfloat162float(hY[(size_t)m * N + 0]));
    printf("   (m=0..3, n=0)\n");
}

// X is one-hot at k=j and W[n][k]=k, so Y reports exactly which W element the
// kernel paired with X's element j — i.e. it reads out the permutation
// directly.
static void probe_permutation(cudaStream_t stream)
{
    constexpr int M = 4, N = 128, K = 128;
    std::vector<bf16> hX((size_t)M * K), hW((size_t)N * K);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            hW[(size_t)n * K + k] = __float2bfloat16((float)k);

    CUDABuffer<bf16> dX(hX.size()), dW(hW.size()), dY((size_t)M * N);
    dW.copy_from_host(hW.data(), stream);

    printf("  k -> paired-with (only mismatches shown):\n   ");
    int bad = 0;
    for (int j = 0; j < K; j++)
    {
        std::fill(hX.begin(), hX.end(), __float2bfloat16(0.0f));
        for (int m = 0; m < M; m++) hX[(size_t)m * K + j] = __float2bfloat16(1.0f);
        dX.copy_from_host(hX.data(), stream);
        CHECK_CUDA(cudaMemsetAsync(dY.data, 0, (size_t)M * N * sizeof(bf16), stream));
        BF16GemmTinyM<4, 128, 64, 4, 1, 1>::run(M, N, K, dX.data, dW.data, dY.data, nullptr, stream);
        std::vector<bf16> hY((size_t)M * N);
        dY.copy_to_host(hY.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        int got = (int)__bfloat162float(hY[0]);
        if (got != j)
        {
            printf(" %d->%d", j, got);
            if (++bad % 8 == 0) printf("\n   ");
        }
    }
    printf("\n  %d of %d k-indices mispaired\n", bad, K);
}

static void diagnose(cudaStream_t stream)
{
    constexpr int K = 128, BK = 64;
    double sum_k = 0, sum_k2 = 0, sum_k_lo = 0;
    for (int k = 0; k < K; k++) { sum_k += k; sum_k2 += (double)k * k; }
    for (int k = 0; k < BK; k++) sum_k_lo += k;

    printf("\n=== probes (M=4 N=128 K=128, BM=4 BN=128 BK=64 -> 2 k-tiles) ===\n");
    printf("  reference: sum(k)=%.0f  sum(k^2)=%.0f  sum(k<%d)=%.0f\n",
           sum_k, sum_k2, BK, sum_k_lo);
    run_probe("A: X=1, W=k", stream, [](int, int) { return 1.0f; },
              [](int, int k) { return (float)k; }, sum_k);
    run_probe("B: X=k, W=1", stream, [](int, int k) { return (float)k; },
              [](int, int) { return 1.0f; }, sum_k);
    run_probe("C: X=k, W=k", stream, [](int, int k) { return (float)k; },
              [](int, int k) { return (float)k; }, sum_k2);
    probe_permutation(stream);
}

int main(int argc, char **argv)
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    int max_smem;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int l2_bytes, num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
    printf("device: %d SMs, %.1f KB max dynamic smem, %.1f MB L2\n",
           num_sm, max_smem / 1024.0, l2_bytes / 1048576.0);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    BenchOptions opt;
    if (argc > 1 && strcmp(argv[1], "--no-check") == 0) opt.check = false;
    if (argc > 1 && strcmp(argv[1], "--diag") == 0)
    {
        diagnose(stream);
        return 0;
    }

    int failures = check_config_agrees_with_template();

    int max_split_k = 1;
    for (auto &v : kVariants) max_split_k = std::max(max_split_k, v.split_k);

    for (auto [M, N, K] : kShapes)
    {
        printf("\n=== M=%d N=%d K=%d ===\n", M, N, K);
        Problem p(M, N, K, handle, stream, max_split_k);

        // Streaming a skinny GEMM is bandwidth work, so report GB/s too.
        double bytes = ((double)M * K + (double)N * K + (double)M * N) * sizeof(bf16);
        auto gbs = [&](double ms) { return bytes / (ms * 1e-3) / 1e9; };

        double cublas_ms = p.time_cublas(opt);
        printf("  %-34s %8.4f ms  %8.1f GB/s\n", "cuBLAS", cublas_ms, gbs(cublas_ms));

        for (auto &v : kVariants)
        {
            if (M > v.bm || N % v.bn || K % (v.bk * v.split_k)) continue;
            if (v.smem > (size_t)max_smem)
            {
                printf("  %-34s SKIP (needs %.1f KB smem)\n", v.name, v.smem / 1024.0);
                continue;
            }

            CompiledKernel kern{};
            kern.fn = v.fn;

            if (opt.check)
            {
                CheckResult r = p.check(kern, opt.tol);
                if (!r.ok)
                {
                    printf("  %-34s FAIL  %s\n", v.name, r.reason.c_str());
                    p.print_mismatches(opt.tol, 8);
                    failures++;
                    continue;
                }
            }
            double ms = p.time(kern, opt);
            printf("  %-34s %8.4f ms  %8.1f GB/s  (%.2fx cuBLAS)\n",
                   v.name, ms, gbs(ms), cublas_ms / ms);
        }
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    printf("\n%s\n", failures ? "SOME VARIANTS FAILED" : "all variants passed");
    return failures ? 1 : 0;
}
