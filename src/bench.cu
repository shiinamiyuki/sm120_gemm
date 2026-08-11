#include "bench_harness.h"
#include "kernel_jit.h"

#include <array>
#include <optional>

using Shape = std::array<int, 3>; // M, N, K

// Square matrices plus LLaMA 3 8B projection shapes, swept from prefill-sized
// batches down to the decode-sized ones (M <= 8) that the tiny-M family exists
// for. M=1/4/8 spans both tiny-M tile heights: BM=4 covers M<=4, BM=8 covers 5..8.
static const std::vector<Shape> kDefaultShapes = {
    {512, 512, 14336},
    {1024, 1024, 14336},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {4096, 14336 * 2, 4096}, // upgate
    {4096, 4096, 14336},     // downproj
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
    std::string cache_path = "autotune_cache.txt";
    std::string jit_cache;
    BenchOptions bench;
    int jobs = 0;
    int max_smem_kb = 0; // 0 = query the device
    bool force = false;
    bool verbose = false;
};

static void usage() {
    printf(R"(usage: bench [options]

modes:
  bench                              bench every shape in the autotune cache
  bench --autotune                   autotune the built-in shape list
  bench --shape M,N,K                bench one shape with its cached config
  bench --shape M,N,K --autotune     autotune one shape
  bench --shape M,N,K --config CFG   compile, check and bench one config
  bench --list-configs               print the configuration space

config strings (tokens may appear in any order, defaults fill the rest):
  128x128x64_s3_cwg2_w32x32_sk1      BMxBNxBK, stages, consumer warp groups,
                                     WARP_MxWARP_N, split-k
  128x128                            bk=64 s=3 cwg=2 sk=1, warp tile derived
  128x64_sk2

options:
  --shape M,N,K       problem shape; repeatable
  --config CFG        use exactly this configuration
  --autotune          sweep the configuration space and cache the winner
  --cache PATH        autotune cache file (default autotune_cache.txt)
  --jit-cache DIR     compiled-kernel cache directory
  --jobs N            parallel nvcc jobs (default: hardware concurrency)
  --max-smem-kb N     shared-memory budget when enumerating configs
  --warmup N          timed-loop warmup iterations (default 5)
  --repeat N          timed-loop iterations (default 20)
  --tol F             correctness tolerance (default 0.1)
  --no-check          skip correctness verification
  --force             re-tune and re-compile even when cached
  -v, --verbose       verbose JIT output
  -h, --help          this message
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
        } else if (arg == "--no-check") {
            a.bench.check = false;
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

// ── Shared per-shape reporting ─────────────────────────────────────────

// Time one kernel against cuBLAS on this shape and print both lines.
static void report(Problem &p, const CompiledKernel &kern, const BenchOptions &opt,
                   const char *label) {
    double cublas_ms = p.time_cublas(opt);
    double ms = p.time(kern, opt);
    printf("  cuBLAS  %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n",
           cublas_ms, p.tflops(cublas_ms), p.gbps(cublas_ms));
    printf("  %-6s  %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.1f%% of cuBLAS)\n",
           label, ms, p.tflops(ms), p.gbps(ms), 100.0 * cublas_ms / ms);
}

// The kernel aborts the process inside cudaFuncSetAttribute if it asks for
// more shared memory than the device allows, so screen it here instead.
static bool smem_fits(const GemmConfig &cfg, size_t device_smem) {
    if (cfg.smem_bytes() <= device_smem) return true;
    fprintf(stderr, "config %s needs %.1f KB shared memory, device allows %.1f KB\n",
            cfg.name().c_str(), cfg.smem_bytes() / 1024.0, device_smem / 1024.0);
    return false;
}

// Returns false if the kernel produced a wrong result.
static bool verify(Problem &p, const CompiledKernel &kern, const BenchOptions &opt, bool verbose) {
    if (!opt.check) return true;
    CheckResult r = p.check(kern, opt.tol);
    if (!r.launched) {
        printf("  check   %s\n", r.reason.c_str());
        return false;
    }
    if (verbose || !r.ok) {
        printf("  check   vs cuBLAS abs=%g rel=%g | vs fp32 abs=%g rel=%g | "
               "cuBLAS vs fp32 rel=%g -> %s\n",
               r.vs_cublas.abs_err, r.vs_cublas.rel_err,
               r.vs_fp32.abs_err, r.vs_fp32.rel_err,
               p.cublas_vs_fp32().rel_err, r.ok ? "PASS" : "FAIL");
    }
    if (!r.ok) p.print_mismatches(opt.tol);
    return r.ok;
}

// ── Modes ──────────────────────────────────────────────────────────────

static int cmd_list_configs(size_t max_smem) {
    auto configs = GemmConfig::enumerate(max_smem);
    for (auto &c : configs)
        printf("%-32s smem %6.1f KB\n", c.name().c_str(), c.smem_bytes() / 1024.0);
    printf("%zu configs (max smem %.0f KB)\n", configs.size(), max_smem / 1024.0);
    return 0;
}

static int cmd_config(const Args &a, KernelJit &jit, size_t device_smem, cublasHandle_t handle,
                      cudaStream_t stream, const std::vector<Shape> &shapes) {
    const GemmConfig &cfg = *a.config;
    if (auto why = cfg.validate(); !why.empty()) {
        fprintf(stderr, "invalid config %s: %s\n", cfg.name().c_str(), why.c_str());
        return 1;
    }
    if (!smem_fits(cfg, device_smem)) return 1;
    const CompiledKernel *kern = jit.get(cfg);
    if (!kern) {
        fprintf(stderr, "failed to build %s\n", cfg.name().c_str());
        return 1;
    }

    bool all_ok = true;
    for (auto [M, N, K] : shapes) {
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!cfg.fits_shape(M, N, K)) {
            printf("  SKIP: config does not tile this shape\n");
            continue;
        }
        Problem p(M, N, K, handle, stream, cfg.split_k);
        if (!verify(p, *kern, a.bench, /*verbose=*/true)) all_ok = false;
        report(p, *kern, a.bench, "ours");
    }
    return all_ok ? 0 : 1;
}

static int cmd_autotune(const Args &a, KernelJit &jit, AutotuneCache &cache, size_t max_smem,
                        cublasHandle_t handle, cudaStream_t stream,
                        const std::vector<Shape> &shapes) {
    auto all_configs = GemmConfig::enumerate(max_smem);

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

        Problem p(M, N, K, handle, stream, max_split_k);
        double cublas_ms = p.time_cublas(a.bench);
        printf("  %-32s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n", "cuBLAS",
               cublas_ms, p.tflops(cublas_ms), p.gbps(cublas_ms));

        const CompiledKernel *best = nullptr;
        double best_ms = 1e30;
        for (auto *kern : kernels) {
            if (a.bench.check) {
                CheckResult r = p.check(*kern, a.bench.tol);
                if (!r.ok) {
                    printf("  %-32s FAIL  %s\n", kern->config.name().c_str(), r.reason.c_str());
                    continue;
                }
            }
            double ms = p.time(*kern, a.bench);
            printf("  %-32s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.1f%%)\n",
                   kern->config.name().c_str(), ms, p.tflops(ms), p.gbps(ms),
                   100.0 * cublas_ms / ms);
            if (ms < best_ms) {
                best_ms = ms;
                best = kern;
            }
        }

        if (!best) {
            printf("  NO WORKING CONFIG\n");
            continue;
        }
        printf("  BEST: %s  %.4f ms  %.2f TFLOPS  %.1f GB/s  (%.1f%% of cuBLAS)\n",
               best->config.name().c_str(), best_ms, p.tflops(best_ms), p.gbps(best_ms),
               100.0 * cublas_ms / best_ms);
        cache.store(M, N, K, best->config, best_ms);
        cache.save(a.cache_path);
    }
    printf("\nAutotune complete -> %s\n", a.cache_path.c_str());
    return 0;
}

static int cmd_bench_cached(const Args &a, KernelJit &jit, const AutotuneCache &cache,
                            size_t device_smem, cublasHandle_t handle, cudaStream_t stream,
                            const std::vector<Shape> &shapes) {
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        auto entry = cache.lookup(M, N, K);
        if (!entry) {
            fprintf(stderr, "no cached config for M=%d N=%d K=%d\n", M, N, K);
            fprintf(stderr, "hint: bench --shape %d,%d,%d --autotune\n", M, N, K);
            return 1;
        }
        const GemmConfig &cfg = entry->config;
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!smem_fits(cfg, device_smem)) {
            rc = 1;
            continue;
        }

        const CompiledKernel *kern = jit.get(cfg);
        if (!kern) {
            fprintf(stderr, "  failed to build %s\n", cfg.name().c_str());
            rc = 1;
            continue;
        }
        Problem p(M, N, K, handle, stream, cfg.split_k);
        if (!verify(p, *kern, a.bench, /*verbose=*/false)) rc = 1;
        report(p, *kern, a.bench, "ours");
    }
    return rc;
}

int main(int argc, char **argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    int device, optin;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    size_t device_smem = (size_t)optin;
    // Budget used when enumerating the config space; the device limit still
    // applies at launch time.
    size_t max_smem = a.max_smem_kb > 0 ? (size_t)a.max_smem_kb * 1024 : device_smem;

    if (a.list_configs) return cmd_list_configs(max_smem);

    JitOptions jopts;
    if (!a.jit_cache.empty()) jopts.cache_dir = a.jit_cache;
    jopts.jobs = a.jobs;
    jopts.force = a.force;
    jopts.verbose = a.verbose;
    KernelJit jit(jopts);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    int rc;
    if (a.config) {
        rc = cmd_config(a, jit, device_smem, handle, stream,
                        a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else if (a.autotune) {
        AutotuneCache cache;
        cache.load(a.cache_path); // fine if missing
        rc = cmd_autotune(a, jit, cache, max_smem, handle, stream,
                          a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else {
        AutotuneCache cache;
        if (!cache.load(a.cache_path)) {
            fprintf(stderr, "no autotune cache at %s\nhint: bench --autotune\n",
                    a.cache_path.c_str());
            rc = 1;
        } else {
            std::vector<Shape> shapes = a.shapes;
            if (shapes.empty())
                for (auto &[key, entry] : cache.entries)
                    shapes.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key)});
            rc = cmd_bench_cached(a, jit, cache, device_smem, handle, stream, shapes);
        }
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return rc;
}
