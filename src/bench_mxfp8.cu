// MXFP8 (OCP microscaling, e4m3 data + ue8m0 scale per 32 elements along K):
// cuBLASLt baseline and the scale-factor-layout probe it rests on.
//
// This is the baseline half of the MXFP8 work. Our own kernels are not wired
// in yet; what this establishes is (a) that cuBLASLt has an MXFP8 algorithm on
// this device, (b) the exact scale-factor tensor layout it expects, and
// (c) the time and accuracy to beat.
#include "mxfp8_harness.h"
#include "mxfp8_mma.cuh"
#include "kernel_jit.h"

#include <array>
#include <optional>

using Shape = std::array<int, 3>;

static const std::vector<Shape> kDefaultShapes = {
    {256, 256, 256},
    {1024, 1024, 1024},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {4096, 14336 * 2, 4096}, // llama3-8b upgate
    {4096, 4096, 14336},     // llama3-8b downproj
    {128, 14336 * 2, 4096},  // upgate, batch 128
    {128, 4096, 14336},      // downproj, batch 128
    {8, 14336 * 2, 4096},    // upgate, batch 8
    {8, 4096, 14336},        // downproj, batch 8
    {1, 14336 * 2, 4096},    // upgate, batch 1 (decode)
    {1, 4096, 14336},        // downproj, batch 1 (decode)
};

struct Args {
    std::vector<Shape> shapes;
    std::optional<GemmConfig> config;
    bool autotune = false;
    bool baseline_only = false;
    bool list_configs = false;
    bool probe_only = false;
    bool probe_mma_only = false;
    bool skip_probe = false;
    bool fast_accum = false;
    bool compare_per_tensor = true;
    bool spec_scale_rule = false;
    Dist dist = Dist::Uniform;
    std::string cache_path = "autotune_cache_mxfp8.txt";
    std::string jit_cache;
    BenchOptions bench;
    int jobs = 0;
    int max_smem_kb = 0; // 0 = query the device
    bool force = false;
    bool verbose = false;
};

static void usage() {
    printf(R"(usage: bench_mxfp8 [options]

MXFP8 GEMM (e4m3 data + one ue8m0 scale per 32 elements along K), checked
against an exact FP32 reference over the same block-quantized inputs and timed
against cuBLASLt's VEC32_UE8M0 path.

modes:
  bench_mxfp8                          bench every shape in the autotune cache
  bench_mxfp8 --autotune               autotune the built-in shape list
  bench_mxfp8 --shape M,N,K            bench one shape with its cached config
  bench_mxfp8 --shape M,N,K --autotune autotune one shape
  bench_mxfp8 --shape M,N,K --config C compile, check and bench one config
  bench_mxfp8 --list-configs           print the configuration space
  bench_mxfp8 --baseline               cuBLASLt only, no kernels of our own

config strings (tokens may appear in any order, defaults fill the rest):
  mxfp8_128x128x128_s2_cwg2_w64x32_sk1 BMxBNxBK, stages, consumer warp groups,
                                       WARP_MxWARP_N, split-k
  mxfp8_tinym_4x256x128_s2_cwg1_sk16   tiny-M (CUDA cores, M <= BM)
The leading mxfp8 token is implied here and may be omitted.

options:
  --shape M,N,K     problem shape; repeatable (K must be a multiple of 128)
  --config CFG      use exactly this configuration
  --autotune        sweep the configuration space and cache the winner
  --baseline        only run cuBLASLt, skipping our kernels entirely
  --cache PATH      autotune cache file (default autotune_cache_mxfp8.txt)
  --jit-cache DIR   compiled-kernel cache directory
  --jobs N          parallel nvcc jobs (default: hardware concurrency)
  --max-smem-kb N   shared-memory budget when enumerating configs
  --list-configs    print the configuration space and exit
  --probe-layout    re-derive the scale-factor tensor layout and exit
  --probe-mma       re-derive the block-scaled MMA scale-fragment layout and exit
  --no-probe        skip the layout self-checks
  --no-per-tensor   do not also time the per-tensor fp8 path
  --mx-spec-scale   use the OCP-literal floor() block scale rule, which clips
  --normal          unit Gaussian inputs instead of uniform[-1,1]
  --outlier         Gaussian with 0.1%% of elements scaled by 1000x. Note this
                    defeats the correctness metric rather than any kernel:
                    products of ~1e4 summing to ~1 means fp32 accumulation
                    order alone moves the result by ~1e-2, and compare()
                    floors its relative denominator at 1. cuBLASLt fails the
                    default tolerance here too; raise --tol to ~0.05.
  --fast-accum      enable CUBLASLT_MATMUL_DESC_FAST_ACCUM
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
        if (i + 1 >= argc) { fprintf(stderr, "%s requires an argument\n", what); return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(); exit(0); }
        else if (arg == "--shape") {
            const char *v = need(i, "--shape");
            if (!v) return false;
            Shape s;
            if (sscanf(v, "%d,%d,%d", &s[0], &s[1], &s[2]) != 3 || s[0] <= 0 || s[1] <= 0 || s[2] <= 0) {
                fprintf(stderr, "bad --shape '%s' (want M,N,K)\n", v);
                return false;
            }
            a.shapes.push_back(s);
        }
        else if (arg == "--config") {
            const char *v = need(i, "--config");
            if (!v) return false;
            std::string err;
            a.config = GemmConfig::parse(v, &err);
            if (!a.config) {
                fprintf(stderr, "bad --config: %s\n", err.c_str());
                return false;
            }
            a.config->elem = ElemType::MxFp8; // this binary only builds mxfp8 kernels
        }
        else if (arg == "--autotune") a.autotune = true;
        else if (arg == "--baseline") a.baseline_only = true;
        else if (arg == "--list-configs") a.list_configs = true;
        else if (arg == "--cache") { const char *v = need(i, "--cache"); if (!v) return false; a.cache_path = v; }
        else if (arg == "--jit-cache") { const char *v = need(i, "--jit-cache"); if (!v) return false; a.jit_cache = v; }
        else if (arg == "--jobs") { const char *v = need(i, "--jobs"); if (!v) return false; a.jobs = atoi(v); }
        else if (arg == "--max-smem-kb") { const char *v = need(i, "--max-smem-kb"); if (!v) return false; a.max_smem_kb = atoi(v); }
        else if (arg == "--force") a.force = true;
        else if (arg == "-v" || arg == "--verbose") a.verbose = true;
        else if (arg == "--probe-layout") a.probe_only = true;
        else if (arg == "--probe-mma") a.probe_mma_only = true;
        else if (arg == "--no-probe") a.skip_probe = true;
        else if (arg == "--no-per-tensor") a.compare_per_tensor = false;
        else if (arg == "--mx-spec-scale") a.spec_scale_rule = true;
        else if (arg == "--normal") a.dist = Dist::Normal;
        else if (arg == "--outlier") a.dist = Dist::Outlier;
        else if (arg == "--fast-accum") a.fast_accum = true;
        else if (arg == "--tol") { const char *v = need(i, "--tol"); if (!v) return false; a.bench.tol = atof(v); }
        else if (arg == "--warmup") { const char *v = need(i, "--warmup"); if (!v) return false; a.bench.warmup = atoi(v); }
        else if (arg == "--repeat") { const char *v = need(i, "--repeat"); if (!v) return false; a.bench.repeat = atoi(v); }
        else { fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]); return false; }
    }
    return true;
}

// ── Scale-factor layout probe ──────────────────────────────────────────
//
// MxScaleLayout::offset() was reverse-engineered, not documented, so it is
// worth re-deriving whenever the toolkit changes. The trick: rig the data so
// that Y[m][n] reads out one scale byte directly, then recover the buffer
// index that byte came from one bit at a time.
//
//   X[m][k] = 1 everywhere;  W[n][k] = 1 only inside k-block kb0, else 0
//   => Y[m][n] = 32 * sB(m, kb0) * sA(n, kb0)
//
// With one operand's scales all 1.0, sweep the other's buffer setting byte i
// to 2.0 iff bit j of i is set: Y/32 is then 2 when the scale cuBLASLt used
// for (row, kb0) lives at a buffer index with bit j set, and 1 otherwise.
// Thirteen such matmuls spell out the index. The scale buffers are
// over-allocated 4x so that an index outside the predicted extent shows up as
// a mismatch rather than reading out of bounds.
static int probe_scale_layout(cudaStream_t stream) {
    const int M = 256, N = 256, K = 256, sk = K / 32;
    MxScaleLayout LX(M, K), LW(N, K);
    const size_t alloc = std::max(LX.bytes(), LW.bytes()) * 4;
    int nbits = 0;
    while ((size_t)1 << nbits < alloc) nbits++;

    CUDABuffer<fp8e4m3> dX((size_t)M * K), dW((size_t)N * K);
    CUDABuffer<bf16> dY((size_t)M * N);
    CUDABuffer<unsigned char> dsX(alloc), dsW(alloc);
    MxFp8GemmLt lt(M, N, K, dsW.data, dsX.data);
    if (!lt.supported()) {
        printf("MXFP8 scale layout: cuBLASLt has no algorithm, cannot probe\n");
        return 1;
    }

    std::vector<fp8e4m3> hX((size_t)M * K, fp8e4m3(1.0f)), hW((size_t)N * K);
    dX.copy_from_host(hX.data(), stream);
    std::vector<bf16> hY((size_t)M * N);
    std::vector<unsigned char> ones(alloc, kUe8m0One), pat(alloc);

    auto matmul = [&] {
        lt.run(dX.data, dW.data, dY.data, stream);
        dY.copy_to_host(hY.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    int bad = 0;
    for (int which = 0; which < 2; which++) { // 0 = B scale (X), 1 = A scale (W)
        const MxScaleLayout &L = which ? LW : LX;
        CUDABuffer<unsigned char> &probed = which ? dsW : dsX;
        CUDABuffer<unsigned char> &other = which ? dsX : dsW;
        for (int kb0 = 0; kb0 < sk; kb0++) {
            std::fill(hW.begin(), hW.end(), fp8e4m3(0.0f));
            for (int n = 0; n < N; n++)
                for (int j = 0; j < 32; j++) hW[(size_t)n * K + kb0 * 32 + j] = fp8e4m3(1.0f);
            dW.copy_from_host(hW.data(), stream);
            other.copy_from_host(ones.data(), stream);

            std::vector<long long> idx(L.rows, 0);
            bool clean = true;
            for (int bit = 0; bit < nbits; bit++) {
                for (size_t i = 0; i < alloc; i++) pat[i] = ((i >> bit) & 1) ? kUe8m0One + 1 : kUe8m0One;
                probed.copy_from_host(pat.data(), stream);
                matmul();
                for (int r = 0; r < L.rows; r++) {
                    float y = which ? __bfloat162float(hY[(size_t)0 * N + r])
                                    : __bfloat162float(hY[(size_t)r * N + 0]);
                    if (y == 64.0f) idx[r] |= 1LL << bit;
                    else if (y != 32.0f) clean = false;
                }
            }
            if (!clean) {
                printf("  %s-scale kb=%d: ambiguous readout (data rigging failed)\n",
                       which ? "A" : "B", kb0);
                bad++;
                continue;
            }
            for (int r = 0; r < L.rows; r++)
                if ((size_t)idx[r] != L.offset(r, kb0)) {
                    if (bad < 8)
                        printf("  %s-scale (row=%d, kb=%d): cuBLASLt reads byte %lld, "
                               "MxScaleLayout predicts %zu\n",
                               which ? "A" : "B", r, kb0, idx[r], L.offset(r, kb0));
                    bad++;
                }
        }
    }
    printf("MXFP8 scale-factor layout: %s (%d entries wrong)\n", bad ? "WRONG" : "CONFIRMED", bad);
    return bad;
}


// ── Block-scaled MMA scale-fragment probe ──────────────────────────────
//
// A and B fragments are all 1.0, so D[m][n] = 32 * sfa(m) * sfb(n). Every
// scale byte in the warp is 127 (=1.0) except one, set to 128 (=2.0); the
// rows (or columns) that come out 64 instead of 32 are exactly the ones that
// byte feeds. Sweeping all 32 lanes x 4 bytes maps the whole fragment.
__global__ void probe_mma_kernel(float *D, const uint32_t *sfa, const uint32_t *sfb, int byte) {
    int lane = threadIdx.x;
    const uint32_t ones4 = 0x38383838u; // four e4m3 1.0 bytes
    uint32_t a[4] = {ones4, ones4, ones4, ones4};
    uint32_t b[2] = {ones4, ones4};
    uint32_t sf_a = sfa[lane], sf_b = sfb[lane];
    float d[4] = {0, 0, 0, 0};
    mma_m16n8k32_e4m3_block_scaled(byte, d, a, b, sf_a, sf_b);
    int g = lane >> 2, l = lane & 3;
    D[g * 8 + l * 2]           = d[0];
    D[g * 8 + l * 2 + 1]       = d[1];
    D[(g + 8) * 8 + l * 2]     = d[2];
    D[(g + 8) * 8 + l * 2 + 1] = d[3];
}

static int probe_mma_layout(cudaStream_t stream) {
    CUDABuffer<float> dD(16 * 8);
    CUDABuffer<uint32_t> dsa(32), dsb(32);
    std::vector<uint32_t> ones(32, 0x7f7f7f7fu), sa(32), sb(32);
    std::vector<float> hD(16 * 8);
    int bad = 0;

    for (int side = 0; side < 2; side++) // 0 = A scale (rows), 1 = B scale (cols)
        for (int lane = 0; lane < 32; lane++)
            for (int byte = 0; byte < 4; byte++) {
                sa = ones;
                sb = ones;
                std::vector<uint32_t> &probed = side ? sb : sa;
                probed[lane] = (ones[lane] & ~(0xffu << (8 * byte))) | (0x80u << (8 * byte));
                dsa.copy_from_host(sa.data(), stream);
                dsb.copy_from_host(sb.data(), stream);
                CHECK_CUDA(cudaMemsetAsync(dD.data, 0, 16 * 8 * sizeof(float), stream));
                probe_mma_kernel<<<1, 32, 0, stream>>>(dD.data, dsa.data, dsb.data, byte);
                CHECK_CUDA(cudaGetLastError());
                dD.copy_to_host(hD.data(), stream);
                CHECK_CUDA(cudaStreamSynchronize(stream));

                // The MMA reads byte `byte` of every lane, so a perturbation
                // at this (lane, byte) should land iff the byte matches and
                // the lane carries a scale.
                int want = side ? mx_sfb_col_for_lane(lane) : mx_sfa_row_for_lane(lane);
                for (int m = 0; m < 16; m++)
                    for (int n = 0; n < 8; n++) {
                        int idx = side ? n : m;
                        float expect = (want == idx) ? 64.0f : 32.0f;
                        if (hD[m * 8 + n] != expect) {
                            if (bad < 8)
                                printf("  %s-scale lane=%d byte=%d: D[%d][%d]=%g want %g\n",
                                       side ? "B" : "A", lane, byte, m, n, hD[m * 8 + n], expect);
                            bad++;
                        }
                    }
            }
    printf("block-scaled MMA scale fragment: %s (%d checks wrong)\n",
           bad ? "WRONG" : "CONFIRMED", bad);
    return bad;
}

// ── Per-shape plumbing ─────────────────────────────────────────────────

// Bind one compiled kernel to a problem, as the (X, W, Y, stream) callable
// MxFp8Problem::check/time expect.
static auto bind(const MxFp8CompiledKernel &kern, MxFp8Problem &p) {
    return [&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t s) {
        kern.fn(p.M(), p.N(), p.K(), X, W, Y, p.x_sf(), p.w_sf(), p.workspace(), s);
    };
}

static bool smem_fits(const GemmConfig &cfg, size_t device_smem) {
    if (cfg.smem_bytes() <= device_smem) return true;
    fprintf(stderr, "config %s needs %.1f KB shared memory, device allows %.1f KB\n",
            cfg.name().c_str(), cfg.smem_bytes() / 1024.0, device_smem / 1024.0);
    return false;
}

// Everything a shape needs before any kernel of ours runs: block-quantized
// inputs, the FP32 reference, and the cuBLASLt baseline timing.
struct ShapeRun {
    MxFp8Problem p;
    MxFp8GemmLt lt;
    double cublas_ms = 0.0;
    bool ok = true;

    ShapeRun(int M, int N, int K, const Args &a, cudaStream_t stream, int max_split_k)
        : p(M, N, K, stream, a.dist, a.spec_scale_rule, max_split_k),
          lt(M, N, K, p.w_sf(), p.x_sf(), a.fast_accum) {
        AccuracyStats mx = p.mx_input_error(), pt = p.per_tensor_input_error();
        printf("  input quant err   mxfp8: max_abs=%.3g rms_rel=%.2f%%   "
               "per-tensor: max_abs=%.3g rms_rel=%.2f%%\n",
               mx.max_abs, 100.0 * mx.rms_rel, pt.max_abs, 100.0 * pt.rms_rel);
        printf("  scale tensors     X %zu B  W %zu B  (%.2f%% of operand bytes)\n",
               p.x_layout().bytes(), p.w_layout().bytes(),
               100.0 * (p.x_layout().bytes() + p.w_layout().bytes()) /
                   ((double)M * K + (double)N * K));

        if (!lt.supported()) {
            printf("  cuBLASLt: no MXFP8 algorithm for this shape\n");
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
        cublas_ms = p.time(run, a.bench);
        printf("  %-34s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n", "cuBLASLt mxfp8",
               cublas_ms, p.tflops(cublas_ms), p.gbps(cublas_ms));

        if (a.compare_per_tensor) {
            // Same operand bytes under per-tensor scaling: what cuBLASLt's
            // block scaling costs it, and the number our per-tensor tiny-M
            // kernel already competes with.
            CUDABuffer<float> s1(1);
            float one = 1.0f;
            s1.copy_from_host(&one, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            Fp8GemmLt pt_lt(M, N, K, s1.data, s1.data, a.fast_accum);
            if (pt_lt.supported()) {
                BenchOptions o = a.bench;
                o.check = false;
                double pt_ms = p.time([&](const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y,
                                          cudaStream_t s) { pt_lt.run(X, W, Y, s); }, o);
                printf("  %-34s %8.4f ms  %9.2f TFLOPS  (mxfp8 is %.2fx)\n",
                       "cuBLASLt per-tensor fp8", pt_ms, p.tflops(pt_ms), pt_ms / cublas_ms);
            }
        }
    }
};

// Check + time one kernel, printing its line. Returns the time, or 0 on failure.
static double run_one(ShapeRun &s, const MxFp8CompiledKernel &kern, const Args &a) {
    auto fn = bind(kern, s.p);
    const std::string name = kern.config.name();
    if (a.bench.check) {
        CheckResult r = s.p.check(fn, a.bench.tol);
        if (!r.ok) {
            printf("  %-34s FAIL  %s\n", name.c_str(), r.reason.c_str());
            s.p.print_mismatches(a.bench.tol, 4);
            return 0.0;
        }
    }
    double ms = s.p.time(fn, a.bench);
    printf("  %-34s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.2fx cuBLASLt)\n",
           name.c_str(), ms, s.p.tflops(ms), s.p.gbps(ms), s.cublas_ms / ms);
    return ms;
}

// ── Modes ──────────────────────────────────────────────────────────────

static std::vector<GemmConfig> enumerate_configs(size_t max_smem) {
    return GemmConfig::enumerate(max_smem, ElemType::MxFp8);
}

static int cmd_list_configs(size_t max_smem) {
    auto configs = enumerate_configs(max_smem);
    for (auto &c : configs)
        printf("%-40s smem %6.1f KB\n", c.name().c_str(), c.smem_bytes() / 1024.0);
    printf("%zu configs (max smem %.0f KB)\n", configs.size(), max_smem / 1024.0);
    return 0;
}

static int cmd_baseline(const Args &a, cudaStream_t stream, const std::vector<Shape> &shapes) {
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        printf("\n=== M=%d N=%d K=%d ===\n", M, N, K);
        if (K % 32) {
            printf("  SKIP: K must be a multiple of 32\n");
            continue;
        }
        ShapeRun s(M, N, K, a, stream, 1);
        if (!s.ok) rc = 1;
    }
    return rc;
}

static int cmd_config(const Args &a, MxFp8KernelJit &jit, size_t device_smem,
                      cudaStream_t stream, const std::vector<Shape> &shapes) {
    const GemmConfig &cfg = *a.config;
    if (auto why = cfg.validate(); !why.empty()) {
        fprintf(stderr, "invalid config %s: %s\n", cfg.name().c_str(), why.c_str());
        return 1;
    }
    if (!smem_fits(cfg, device_smem)) return 1;
    const MxFp8CompiledKernel *kern = jit.get(cfg);
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

static int cmd_autotune(const Args &a, MxFp8KernelJit &jit, AutotuneCache &cache,
                        size_t max_smem, cudaStream_t stream,
                        const std::vector<Shape> &shapes) {
    auto all_configs = enumerate_configs(max_smem);

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

        const MxFp8CompiledKernel *best = nullptr;
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

static int cmd_bench_cached(const Args &a, MxFp8KernelJit &jit, const AutotuneCache &cache,
                            size_t device_smem, cudaStream_t stream,
                            const std::vector<Shape> &shapes) {
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        auto entry = cache.lookup(M, N, K);
        if (!entry) {
            fprintf(stderr, "no cached config for M=%d N=%d K=%d\n", M, N, K);
            fprintf(stderr, "hint: bench_mxfp8 --shape %d,%d,%d --autotune\n", M, N, K);
            return 1;
        }
        const GemmConfig &cfg = entry->config;
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!smem_fits(cfg, device_smem)) {
            rc = 1;
            continue;
        }
        const MxFp8CompiledKernel *kern = jit.get(cfg);
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

    printf("device: sm_%d%d   mxfp8 = e4m3 data + ue8m0 scale per 32 along K, %s inputs%s\n",
           major, minor,
           a.dist == Dist::Normal ? "gaussian" : a.dist == Dist::Outlier ? "gaussian+outlier" : "uniform",
           a.fast_accum ? ", fast-accum" : "");

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    int rc = 0;
    if (a.probe_mma_only) {
        rc = probe_mma_layout(stream) ? 1 : 0;
        cudaStreamDestroy(stream);
        return rc;
    }
    if (a.probe_only) {
        rc = probe_scale_layout(stream) ? 1 : 0;
        cudaStreamDestroy(stream);
        return rc;
    }
    if (!a.skip_probe) {
        rc |= probe_mma_layout(stream) ? 1 : 0;
        rc |= probe_scale_layout(stream) ? 1 : 0;
    }

    JitOptions jopts = mxfp8_jit_options();
    if (!a.jit_cache.empty()) jopts.cache_dir = a.jit_cache;
    jopts.jobs = a.jobs;
    jopts.force = a.force;
    jopts.verbose = a.verbose;
    MxFp8KernelJit jit(jopts);

    if (a.baseline_only) {
        rc |= cmd_baseline(a, stream, a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else if (a.config) {
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
            fprintf(stderr, "no autotune cache at %s\nhint: bench_mxfp8 --autotune\n",
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
