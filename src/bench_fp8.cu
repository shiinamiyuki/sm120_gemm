// FP8 (e4m3) GEMM with per-tensor scaling: harness, reference, cuBLASLt
// baseline and our own JIT-compiled kernels.
//
//   Y = (x_scale * X) @ (w_scale * W)^T
//   X: M x K row-major e4m3, W: N x K row-major e4m3, Y: M x N row-major bf16
//
// Same structure as bench.cu: configs are compiled on demand by KernelJit and
// the winner per shape is cached in an autotune file.
#include "fp8_harness.h"
#include "fp8_gemm.cuh" // probe_layout uses the raw MMA wrapper directly
#include "kernel_jit.h"

#include <array>
#include <optional>

using Shape = std::array<int, 3>;

// Square matrices plus LLaMA 3 8B projection shapes, swept from prefill-sized
// batches down to the decode-sized ones (M <= 8) the tiny-M family exists for.
static const std::vector<Shape> kDefaultShapes = {
    {256, 256, 256}, // small: easy to eyeball if something is wrong
    {1024, 1024, 1024},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {4096, 14336 * 2, 4096}, // llama3-8b upgate
    {4096, 4096, 14336},     // llama3-8b downproj
    {128, 14336 * 2, 4096},  // upgate, batch 128
    {128, 4096, 14336},      // downproj, batch 128
    {8, 14336 * 2, 4096},    // upgate, batch 8
    {8, 4096, 14336},        // downproj, batch 8
    {4, 14336 * 2, 4096},    // upgate, batch 4
    {4, 4096, 14336},        // downproj, batch 4
    {1, 14336 * 2, 4096},    // upgate, batch 1 (decode)
    {1, 4096, 14336},        // downproj, batch 1 (decode)
};

struct Args {
    std::vector<Shape> shapes;
    std::optional<GemmConfig> config;
    bool autotune = false;
    bool list_configs = false;
    bool probe_only = false;
    bool quant_error = false;
    bool fast_accum = false;
    Dist dist = Dist::Uniform;
    std::string cache_path = "autotune_cache_fp8.txt";
    std::string jit_cache;
    BenchOptions bench;
    int jobs = 0;
    int max_smem_kb = 0; // 0 = query the device
    bool force = false;
    bool verbose = false;
};

static void usage() {
    printf(R"(usage: bench_fp8 [options]

Per-tensor-scaled FP8 (e4m3) GEMM, checked against an exact FP32 reference over
the same quantized inputs and timed against cuBLASLt.

modes:
  bench_fp8                          bench every shape in the autotune cache
  bench_fp8 --autotune               autotune the built-in shape list
  bench_fp8 --shape M,N,K            bench one shape with its cached config
  bench_fp8 --shape M,N,K --autotune autotune one shape
  bench_fp8 --shape M,N,K --config C compile, check and bench one config
  bench_fp8 --list-configs           print the configuration space

config strings (tokens may appear in any order, defaults fill the rest):
  fp8_128x128x128_s2_cwg2_w64x32_sk1   BMxBNxBK, stages, consumer warp groups,
                                       WARP_MxWARP_N, split-k
  fp8_tinym_4x256x128_s2_cwg1_sk16     tiny-M (CUDA cores, M <= BM)
  fp8_128x128                          bk=128 s=3 cwg=2 sk=1, warp tile derived
The leading fp8 token is implied here and may be omitted.

options:
  --shape M,N,K     problem shape; repeatable
  --config CFG      use exactly this configuration
  --autotune        sweep the configuration space and cache the winner
  --cache PATH      autotune cache file (default autotune_cache_fp8.txt)
  --jit-cache DIR   compiled-kernel cache directory
  --jobs N          parallel nvcc jobs (default: hardware concurrency)
  --max-smem-kb N   shared-memory budget when enumerating configs
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
  --force           re-tune and re-compile even when cached
  -v, --verbose     verbose JIT output
  -h, --help        this message
)");
}

static bool parse_args(int argc, char **argv, Args &a) {
    auto need = [&](int &i, const char *what) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "%s requires an argument\n", what);
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            exit(0);
        } else if (arg == "--shape") {
            const char *v = need(i, "--shape");
            if (!v) return false;
            Shape s;
            if (sscanf(v, "%d,%d,%d", &s[0], &s[1], &s[2]) != 3 ||
                s[0] <= 0 || s[1] <= 0 || s[2] <= 0) {
                fprintf(stderr, "bad --shape '%s' (want M,N,K)\n", v);
                return false;
            }
            a.shapes.push_back(s);
        } else if (arg == "--config") {
            const char *v = need(i, "--config");
            if (!v) return false;
            std::string err;
            a.config = GemmConfig::parse(v, &err);
            if (!a.config) {
                fprintf(stderr, "bad --config: %s\n", err.c_str());
                return false;
            }
            a.config->elem = ElemType::Fp8; // this binary only builds fp8 kernels
        } else if (arg == "--autotune") {
            a.autotune = true;
        } else if (arg == "--list-configs") {
            a.list_configs = true;
        } else if (arg == "--cache") {
            const char *v = need(i, "--cache");
            if (!v) return false;
            a.cache_path = v;
        } else if (arg == "--jit-cache") {
            const char *v = need(i, "--jit-cache");
            if (!v) return false;
            a.jit_cache = v;
        } else if (arg == "--jobs") {
            const char *v = need(i, "--jobs");
            if (!v) return false;
            a.jobs = atoi(v);
        } else if (arg == "--max-smem-kb") {
            const char *v = need(i, "--max-smem-kb");
            if (!v) return false;
            a.max_smem_kb = atoi(v);
        } else if (arg == "--warmup") {
            const char *v = need(i, "--warmup");
            if (!v) return false;
            a.bench.warmup = atoi(v);
        } else if (arg == "--repeat") {
            const char *v = need(i, "--repeat");
            if (!v) return false;
            a.bench.repeat = atoi(v);
        } else if (arg == "--tol") {
            const char *v = need(i, "--tol");
            if (!v) return false;
            a.bench.tol = atof(v);
        } else if (arg == "--quant-error") {
            a.quant_error = true;
        } else if (arg == "--normal") {
            a.dist = Dist::Normal;
        } else if (arg == "--outlier") {
            a.dist = Dist::Outlier;
        } else if (arg == "--fast-accum") {
            a.fast_accum = true;
        } else if (arg == "--probe-layout") {
            a.probe_only = true;
        } else if (arg == "--force") {
            a.force = true;
        } else if (arg == "-v" || arg == "--verbose") {
            a.verbose = true;
        } else {
            fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]);
            return false;
        }
    }
    return true;
}

// ── m16n8k32 fragment-layout probe ─────────────────────────────────────
//
// Pins down the e4m3 fragment layout in isolation: one MMA, one warp,
// fragments packed straight from global memory by the mapping the kernel
// assumes, checked against a CPU dot product. Small integers keep the e4m3
// codes and the fp32 sums exact, so any mismatch is a layout error and nothing
// else. This is what the layout documented in fp8_gemm.cuh rests on.
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

// ── Per-shape plumbing ─────────────────────────────────────────────────

// Bind one compiled kernel to a problem, as the (X, W, Y, stream) callable
// Fp8Problem::check/time expect.
static auto bind(const Fp8CompiledKernel &kern, Fp8Problem &p) {
    return [&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t s) {
        kern.fn(p.M(), p.N(), p.K(), X, W, Y, p.x_scale(), p.w_scale(), p.workspace(), s);
    };
}

// The kernel aborts the process inside cudaFuncSetAttribute if it asks for more
// shared memory than the device allows, so screen it here instead.
static bool smem_fits(const GemmConfig &cfg, size_t device_smem) {
    if (cfg.smem_bytes() <= device_smem) return true;
    fprintf(stderr, "config %s needs %.1f KB shared memory, device allows %.1f KB\n",
            cfg.name().c_str(), cfg.smem_bytes() / 1024.0, device_smem / 1024.0);
    return false;
}

// Everything a shape needs before any kernel runs: quantized inputs, the FP32
// reference, and the cuBLASLt baseline timing.
struct ShapeRun {
    Fp8Problem p;
    Fp8GemmLt lt;
    double cublas_ms = 0.0;
    bool ok = true;

    ShapeRun(int M, int N, int K, const Args &a, cudaStream_t stream, int max_split_k)
        : p(M, N, K, stream, a.dist, max_split_k),
          lt(M, N, K, p.w_scale_dev(), p.x_scale_dev(), a.fast_accum) {
        AccuracyStats q = p.input_quant_error();
        printf("  scales    x=%.4g  w=%.4g   input quant err: max_abs=%.3g rms_rel=%.2f%%\n",
               p.x_scale(), p.w_scale(), q.max_abs, 100.0 * q.rms_rel);

        if (!lt.supported()) {
            printf("  cuBLASLt: no FP8 algorithm for this shape\n");
            ok = false;
            return;
        }
        auto run = [&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t s) {
            lt.run(X, W, Y, s);
        };
        CheckResult r = p.check(run, a.bench.tol);
        printf("  check     cuBLASLt vs fp32 reference: abs=%.4g rel=%.4g -> %s\n",
               r.vs_fp32.abs_err, r.vs_fp32.rel_err, r.ok ? "PASS" : "FAIL");
        if (!r.ok) {
            p.print_mismatches(a.bench.tol);
            ok = false;
        }
        if (a.quant_error) {
            AccuracyStats c = p.quantization_cost_at_output();
            printf("  fp8 cost  exact fp8 result vs fp32 inputs: max_abs=%.4g rms_rel=%.2f%%\n",
                   c.max_abs, 100.0 * c.rms_rel);
        }
        cublas_ms = p.time(run, a.bench);
        printf("  %-32s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n", "cuBLASLt",
               cublas_ms, p.tflops(cublas_ms), p.gbps(cublas_ms));
    }
};

// Check + time one kernel, printing its line. Returns the time, or 0 on failure.
static double run_one(ShapeRun &s, const Fp8CompiledKernel &kern, const Args &a) {
    auto fn = bind(kern, s.p);
    const std::string name = kern.config.name();
    if (a.bench.check) {
        CheckResult r = s.p.check(fn, a.bench.tol);
        if (!r.ok) {
            printf("  %-32s FAIL  %s\n", name.c_str(), r.reason.c_str());
            s.p.print_mismatches(a.bench.tol, 4);
            return 0.0;
        }
    }
    double ms = s.p.time(fn, a.bench);
    printf("  %-32s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.2fx cuBLASLt)\n",
           name.c_str(), ms, s.p.tflops(ms), s.p.gbps(ms), s.cublas_ms / ms);
    return ms;
}

// ── Modes ──────────────────────────────────────────────────────────────

static int cmd_list_configs(size_t max_smem) {
    auto configs = GemmConfig::enumerate(max_smem, ElemType::Fp8);
    for (auto &c : configs)
        printf("%-36s smem %6.1f KB\n", c.name().c_str(), c.smem_bytes() / 1024.0);
    printf("%zu configs (max smem %.0f KB)\n", configs.size(), max_smem / 1024.0);
    return 0;
}

static int cmd_config(const Args &a, Fp8KernelJit &jit, size_t device_smem,
                      cudaStream_t stream, const std::vector<Shape> &shapes) {
    const GemmConfig &cfg = *a.config;
    if (auto why = cfg.validate(); !why.empty()) {
        fprintf(stderr, "invalid config %s: %s\n", cfg.name().c_str(), why.c_str());
        return 1;
    }
    if (!smem_fits(cfg, device_smem)) return 1;
    const Fp8CompiledKernel *kern = jit.get(cfg);
    if (!kern) {
        fprintf(stderr, "failed to build %s\n", cfg.name().c_str());
        return 1;
    }

    int rc = 0;
    for (auto [M, N, K] : shapes) {
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!cfg.fits_shape(M, N, K)) {
            printf("  SKIP: config does not tile this shape\n");
            continue;
        }
        ShapeRun s(M, N, K, a, stream, cfg.split_k);
        if (!s.ok) {
            rc = 1;
            continue;
        }
        if (run_one(s, *kern, a) == 0.0) rc = 1;
    }
    return rc;
}

static int cmd_autotune(const Args &a, Fp8KernelJit &jit, AutotuneCache &cache, size_t max_smem,
                        cudaStream_t stream, const std::vector<Shape> &shapes) {
    auto all_configs = GemmConfig::enumerate(max_smem, ElemType::Fp8);

    for (auto [M, N, K] : shapes) {
        printf("\n=== Autotune M=%d N=%d K=%d ===\n", M, N, K);
        if (!a.force) {
            if (auto e = cache.lookup(M, N, K)) {
                printf("  cached: %s  %.4f ms\n", e->config.name().c_str(), e->time_ms);
                continue;
            }
        }

        std::vector<GemmConfig> candidates;
        int max_split_k = 1;
        for (auto &c : all_configs)
            if (c.fits_shape(M, N, K)) {
                candidates.push_back(c);
                max_split_k = std::max(max_split_k, c.split_k);
            }
        if (candidates.empty()) {
            printf("  SKIP: no config tiles this shape\n");
            continue;
        }
        printf("  %zu candidate configs\n", candidates.size());

        auto kernels = jit.get_many(candidates);
        if (kernels.empty()) {
            printf("  SKIP: nothing compiled\n");
            continue;
        }

        ShapeRun s(M, N, K, a, stream, max_split_k);
        if (!s.ok) continue;

        const Fp8CompiledKernel *best = nullptr;
        double best_ms = 1e30;
        for (auto *kern : kernels) {
            double ms = run_one(s, *kern, a);
            if (ms > 0.0 && ms < best_ms) {
                best_ms = ms;
                best = kern;
            }
        }

        if (!best) {
            printf("  NO WORKING CONFIG\n");
            continue;
        }
        printf("  BEST: %s  %.4f ms  %.2f TFLOPS  %.1f GB/s  (%.2fx cuBLASLt)\n",
               best->config.name().c_str(), best_ms, s.p.tflops(best_ms), s.p.gbps(best_ms),
               s.cublas_ms / best_ms);
        cache.store(M, N, K, best->config, best_ms);
        cache.save(a.cache_path);
    }
    printf("\nAutotune complete -> %s\n", a.cache_path.c_str());
    return 0;
}

static int cmd_bench_cached(const Args &a, Fp8KernelJit &jit, const AutotuneCache &cache,
                            size_t device_smem, cudaStream_t stream,
                            const std::vector<Shape> &shapes) {
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        auto entry = cache.lookup(M, N, K);
        if (!entry) {
            fprintf(stderr, "no cached config for M=%d N=%d K=%d\n", M, N, K);
            fprintf(stderr, "hint: bench_fp8 --shape %d,%d,%d --autotune\n", M, N, K);
            return 1;
        }
        const GemmConfig &cfg = entry->config;
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!smem_fits(cfg, device_smem)) {
            rc = 1;
            continue;
        }
        const Fp8CompiledKernel *kern = jit.get(cfg);
        if (!kern) {
            fprintf(stderr, "  failed to build %s\n", cfg.name().c_str());
            rc = 1;
            continue;
        }
        ShapeRun s(M, N, K, a, stream, cfg.split_k);
        if (!s.ok) {
            rc = 1;
            continue;
        }
        if (run_one(s, *kern, a) == 0.0) rc = 1;
    }
    return rc;
}

int main(int argc, char **argv) {
    Args a;
    a.bench.tol = 0.02f; // bf16 output rounding alone is ~0.4%
    if (!parse_args(argc, argv, a)) return 2;

    int device, major = 0, minor = 0, optin = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    size_t device_smem = (size_t)optin;
    size_t max_smem = a.max_smem_kb > 0 ? (size_t)a.max_smem_kb * 1024 : device_smem;

    if (a.list_configs) return cmd_list_configs(max_smem);

    printf("device: sm_%d%d   fp8 = e4m3 (max %.0f), per-tensor scaling, %s inputs%s\n",
           major, minor, kE4M3Max,
           a.dist == Dist::Normal ? "gaussian" : a.dist == Dist::Outlier ? "gaussian+outlier" : "uniform",
           a.fast_accum ? ", fast-accum" : "");

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    if (a.probe_only) {
        int bad = probe_layout(stream);
        cudaStreamDestroy(stream);
        return bad ? 1 : 0;
    }
    int rc = probe_layout(stream) ? 1 : 0;

    JitOptions jopts = fp8_jit_options();
    if (!a.jit_cache.empty()) jopts.cache_dir = a.jit_cache;
    jopts.jobs = a.jobs;
    jopts.force = a.force;
    jopts.verbose = a.verbose;
    Fp8KernelJit jit(jopts);

    if (a.config) {
        rc |= cmd_config(a, jit, device_smem, stream,
                         a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else if (a.autotune) {
        AutotuneCache cache;
        cache.load(a.cache_path); // fine if missing
        rc |= cmd_autotune(a, jit, cache, max_smem, stream,
                           a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else {
        AutotuneCache cache;
        if (!cache.load(a.cache_path)) {
            fprintf(stderr, "no autotune cache at %s\nhint: bench_fp8 --autotune\n",
                    a.cache_path.c_str());
            rc = 1;
        } else {
            std::vector<Shape> shapes = a.shapes;
            if (shapes.empty())
                for (auto &[key, entry] : cache.entries)
                    shapes.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key)});
            rc |= cmd_bench_cached(a, jit, cache, device_smem, stream, shapes);
        }
    }

    cudaStreamDestroy(stream);
    printf("\n%s\n", rc ? "SOME SHAPES FAILED" : "all shapes passed");
    return rc ? 1 : 0;
}
