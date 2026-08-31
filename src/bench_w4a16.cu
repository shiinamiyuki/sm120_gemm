// W4A16: bf16 activations x NVFP4 weights (packed e2m1, one ue4m3 scale per
// 16 elements of K, one fp32 global scale), checked against an exact FP32
// reference and timed against the bf16 GEMM it replaces.
//
// There is no vendor baseline to beat here: cuBLASLt has no mixed-width
// matmul at all (--probe-support) and the tensor cores have no mixed 4-bit x
// 16-bit MMA. The bar is bf16 cuBLAS -- the unquantized GEMM -- and the win
// is the weight bandwidth NVFP4 saves.
#include "w4a16_harness.h"
#include "kernel_jit.h"

#include <array>
#include <optional>

using Shape = std::array<int, 3>;

// Decode-sized batches on LLaMA 3 8B projections: what W4A16 is actually for.
// K must be a multiple of BK = 256; 14336 and 4096 both are.
static const std::vector<Shape> kDefaultShapes = {
    {1, 14336 * 2, 4096},
    {1, 4096, 14336},
    {4, 14336 * 2, 4096},
    {4, 4096, 14336},
    {8, 14336 * 2, 4096},
    {8, 4096, 14336},
};

struct Args {
    std::vector<Shape> shapes;
    std::optional<GemmConfig> config;
    bool autotune = false, list_configs = false;
    bool probe_layout = false, probe_support = false, skip_probe = false;
    Dist dist = Dist::Uniform;
    std::string cache_path = "autotune_cache_w4a16.txt";
    std::string jit_cache;
    BenchOptions bench;
    int jobs = 0, max_smem_kb = 0;
    bool force = false, verbose = false;
};

static void usage() {
    printf(R"(usage: bench_w4a16 [options]

bf16 activations x NVFP4 weights (e2m1 + ue4m3 per 16 + fp32 global), tiny-M
(M <= BM, CUDA cores). Baseline is bf16 cuBLAS: neither cuBLASLt nor the
tensor cores offer a mixed 4-bit x 16-bit path.

modes:
  bench_w4a16                          bench every shape in the autotune cache
  bench_w4a16 --autotune               autotune the built-in shape list
  bench_w4a16 --shape M,N,K --config C compile, check and bench one config
  bench_w4a16 --probe-support          sweep cuBLASLt type combinations and exit
  bench_w4a16 --probe-layout           re-derive the NVFP4 scale layout and exit

config strings:
  w4a16_tinym_1x128x256_s3_cwg1_sk4    BMxBNxBK, stages, consumer warp groups,
                                       split-k. BK is always 256.

options:
  --shape M,N,K     problem shape; repeatable (K must be a multiple of 256)
  --config CFG      use exactly this configuration
  --autotune        sweep the configuration space and cache the winner
  --cache PATH      autotune cache file (default autotune_cache_w4a16.txt)
  --jit-cache DIR   compiled-kernel cache directory
  --jobs N          parallel nvcc jobs
  --max-smem-kb N   shared-memory budget when enumerating configs
  --list-configs    print the configuration space and exit
  --no-probe        skip the layout self-check
  --normal          unit Gaussian inputs instead of uniform[-1,1]
  --outlier         Gaussian with 0.1%% of elements scaled by 1000x
  --tol F           correctness tolerance (default 0.03)
  --warmup N        timed-loop warmup iterations (default 5)
  --repeat N        timed-loop iterations (default 20)
  --force           re-tune and re-compile even when cached
  -v, --verbose     verbose JIT output
  -h, --help        this message
)");
}

static bool parse_args(int argc, char **argv, Args &a) {
    auto need = [&](int &i, const char *w) -> const char * {
        if (i + 1 >= argc) { fprintf(stderr, "%s requires an argument\n", w); return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(); exit(0); }
        else if (arg == "--shape") {
            const char *v = need(i, "--shape"); if (!v) return false;
            Shape s;
            if (sscanf(v, "%d,%d,%d", &s[0], &s[1], &s[2]) != 3 || s[0] <= 0 || s[1] <= 0 || s[2] <= 0) {
                fprintf(stderr, "bad --shape '%s'\n", v); return false;
            }
            a.shapes.push_back(s);
        }
        else if (arg == "--config") {
            const char *v = need(i, "--config"); if (!v) return false;
            std::string err;
            a.config = GemmConfig::parse(v, &err);
            if (!a.config) { fprintf(stderr, "bad --config: %s\n", err.c_str()); return false; }
            a.config->elem = ElemType::W4A16;
            // W4A16 has no dense-M family, so a config that named none defaults
            // to CUDA-core tiny-M; an explicit "tinymtc" is preserved.
            if (!a.config->is_skinny()) a.config->family = KernelFamily::TinyM;
        }
        else if (arg == "--autotune") a.autotune = true;
        else if (arg == "--list-configs") a.list_configs = true;
        else if (arg == "--probe-layout") a.probe_layout = true;
        else if (arg == "--probe-support") a.probe_support = true;
        else if (arg == "--no-probe") a.skip_probe = true;
        else if (arg == "--normal") a.dist = Dist::Normal;
        else if (arg == "--outlier") a.dist = Dist::Outlier;
        else if (arg == "--cache") { const char *v = need(i, "--cache"); if (!v) return false; a.cache_path = v; }
        else if (arg == "--jit-cache") { const char *v = need(i, "--jit-cache"); if (!v) return false; a.jit_cache = v; }
        else if (arg == "--jobs") { const char *v = need(i, "--jobs"); if (!v) return false; a.jobs = atoi(v); }
        else if (arg == "--max-smem-kb") { const char *v = need(i, "--max-smem-kb"); if (!v) return false; a.max_smem_kb = atoi(v); }
        else if (arg == "--tol") { const char *v = need(i, "--tol"); if (!v) return false; a.bench.tol = atof(v); }
        else if (arg == "--warmup") { const char *v = need(i, "--warmup"); if (!v) return false; a.bench.warmup = atoi(v); }
        else if (arg == "--repeat") { const char *v = need(i, "--repeat"); if (!v) return false; a.bench.repeat = atoi(v); }
        else if (arg == "--force") a.force = true;
        else if (arg == "-v" || arg == "--verbose") a.verbose = true;
        else { fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]); return false; }
    }
    return true;
}

// ── Is there any cuBLASLt path for mixed-width operands? ───────────────
//
// cublasLtMatmulAlgoGetHeuristic is the authority: it returns zero algorithms
// for an unsupported combination rather than an error. Sweeping the whole
// space is what turns "we could not find one" into "there is none".
static int probe_support() {
    cublasLtHandle_t lt;
    CHECK_CUBLASLT(cublasLtCreate(&lt));
    auto one = [&](cudaDataType ta, cudaDataType tb, cudaDataType td,
                   cublasComputeType_t comp, cudaDataType st,
                   cublasOperation_t tA, cublasOperation_t tB,
                   int M, int N, int K, int a_mode, int b_mode = -1) {
        cublasLtMatmulDesc_t op;
        if (cublasLtMatmulDescCreate(&op, comp, st) != CUBLAS_STATUS_SUCCESS) return 0;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        if (a_mode >= 0) {
            int32_t m = a_mode; void *p = (void *)0x1000;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof(m));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p, sizeof(p));
        }
        if (b_mode >= 0) {
            int32_t m = b_mode; void *p = (void *)0x2000;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof(m));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p, sizeof(p));
        }
        int ar = (tA == CUBLAS_OP_T) ? K : N, ac = (tA == CUBLAS_OP_T) ? N : K;
        int br = (tB == CUBLAS_OP_T) ? M : K, bc = (tB == CUBLAS_OP_T) ? K : M;
        cublasLtMatrixLayout_t la, lb, lc, ld;
        if (cublasLtMatrixLayoutCreate(&la, ta, ar, ac, ar) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulDescDestroy(op); return 0;
        }
        cublasLtMatrixLayoutCreate(&lb, tb, br, bc, br);
        cublasLtMatrixLayoutCreate(&lc, td, N, M, N);
        cublasLtMatrixLayoutCreate(&ld, td, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = 32u << 20;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        cublasLtMatmulHeuristicResult_t h[4]{};
        int got = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, ld, pref, 4, h, &got);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(ld); cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(la);
        cublasLtMatmulDescDestroy(op);
        return got;
    };

    cublasOperation_t ops[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    cublasComputeType_t comps[4] = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16BF,
                                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_16F};
    cudaDataType scal[4] = {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_16F};
    cudaDataType wts[3] = {CUDA_R_4F_E2M1, CUDA_R_4I, CUDA_R_4U};
    cudaDataType acts[2] = {CUDA_R_16BF, CUDA_R_16F};
    int shapes[4][3] = {{4096,4096,4096},{1,4096,14336},{16,4096,4096},{128,128,128}};
    int modes[4] = {-1, 0, 1, 2};

    long total = 0, hits = 0;
    for (auto w : wts) for (auto a : acts) for (int c = 0; c < 4; c++)
      for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
        for (auto &sh : shapes) for (int sm : modes)
          for (cudaDataType td : {a, CUDA_R_32F}) {
              total++;
              if (one(w, a, td, comps[c], scal[c], ops[i], ops[j], sh[0], sh[1], sh[2], sm) > 0) hits++;
          }
    printf("cuBLASLt mixed-width (4-bit weights x 16-bit activations):\n");
    printf("  %ld combinations of {weight type, activation type, compute type,\n"
           "  op, shape, scale mode, output type} -> %ld supported\n", total, hits);
    // For contrast, the same-width 4-bit path cuBLASLt does have.
    int nv = one(CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                 CUBLAS_OP_T, CUBLAS_OP_N, 4096, 4096, 4096,
                 CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
                 CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
    printf("  for contrast, NVFP4 W4A4 (e2m1 x e2m1, vec16_ue4m3): %d algorithms\n", nv);
    cublasLtDestroy(lt);
    return hits > 0 ? 0 : 0; // informational either way
}

// ── NVFP4 scale-layout self-check ──────────────────────────────────────
//
// Our kernel reads the same swizzled ue4m3 tensor cuBLASLt consumes, so it is
// worth re-deriving that layout against the hardware. Rigged so that
// Y[m][n] = 16 * sB(m,kb) * sA(n,kb), then the buffer index feeding each row
// is read back one bit per matmul. Also checks the e2m1 nibble order: with W
// non-zero only in the low nibble, exactly half of each 16-block survives.
static int probe_layout(cudaStream_t stream) {
    const int M = 256, N = 256, K = 256, VEC = 16, sk = K / VEC;
    BlockScaleLayout LX(M, K, VEC), LW(N, K, VEC);
    const size_t alloc = std::max(LX.bytes(), LW.bytes()) * 4;
    int nbits = 0;
    while ((size_t)1 << nbits < alloc) nbits++;

    CUDABuffer<unsigned char> dX((size_t)M * K / 2), dW((size_t)N * K / 2);
    CUDABuffer<unsigned char> dsA(alloc), dsB(alloc);
    CUDABuffer<bf16> dY((size_t)M * N);

    cublasLtHandle_t lt;
    CHECK_CUBLASLT(cublasLtCreate(&lt));
    void *ws;
    size_t wsb = 32u << 20;
    CHECK_CUDA(cudaMalloc(&ws, wsb));
    cublasLtMatmulDesc_t op;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    void *pa = dsA.data, *pb = dsB.data;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb));
    cublasLtMatrixLayout_t la, lb, lc, ld;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    cublasLtMatmulHeuristicResult_t h{};
    int got = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, ld, pref, 1, &h, &got);
    if (!got) { printf("NVFP4 scale layout: cuBLASLt has no NVFP4 algorithm, cannot probe\n"); return 1; }

    std::vector<bf16> hY((size_t)M * N);
    auto matmul = [&] {
        const float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLASLT(cublasLtMatmul(lt, op, &alpha, dW.data, la, dX.data, lb, &beta,
                                      dY.data, lc, dY.data, ld, &h.algo, ws, wsb, stream));
        dY.copy_to_host(hY.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    const unsigned char one4 = float_to_e2m1(1.0f);
    std::vector<unsigned char> hX((size_t)M * K / 2, (unsigned char)(one4 | (one4 << 4)));
    std::vector<unsigned char> hW((size_t)N * K / 2), ones(alloc, kUe4m3One), pat(alloc);
    dX.copy_from_host(hX.data(), stream);

    // (a) nibble order.
    std::fill(hW.begin(), hW.end(), 0);
    for (int n = 0; n < N; n++)
        for (int j = 0; j < 8; j++) hW[((size_t)n * K) / 2 + j] = one4; // low nibbles only
    dW.copy_from_host(hW.data(), stream);
    dsA.copy_from_host(ones.data(), stream);
    dsB.copy_from_host(ones.data(), stream);
    matmul();
    float nib = __bfloat162float(hY[0]);
    int bad = (nib == 8.0f) ? 0 : 1;
    printf("e2m1 nibble order: low nibble is the %s element (Y=%g, expect 8)\n",
           nib == 8.0f ? "even/first" : "UNEXPECTED", nib);

    // (b) scale-factor layout.
    for (int which = 0; which < 2; which++) {
        const BlockScaleLayout &L = which ? LW : LX;
        CUDABuffer<unsigned char> &probed = which ? dsA : dsB;
        CUDABuffer<unsigned char> &other = which ? dsB : dsA;
        for (int kb0 = 0; kb0 < sk; kb0++) {
            std::fill(hW.begin(), hW.end(), 0);
            for (int n = 0; n < N; n++)
                for (int j = 0; j < 8; j++)
                    hW[((size_t)n * K + kb0 * 16) / 2 + j] = (unsigned char)(one4 | (one4 << 4));
            dW.copy_from_host(hW.data(), stream);
            other.copy_from_host(ones.data(), stream);
            probed.copy_from_host(ones.data(), stream);
            matmul();
            if (__bfloat162float(hY[0]) != 16.0f) {
                printf("  kb=%d: baseline Y[0]=%g (expect 16)\n", kb0, __bfloat162float(hY[0]));
                bad++;
                continue;
            }
            std::vector<long long> idx(L.rows, 0);
            bool clean = true;
            for (int bit = 0; bit < nbits; bit++) {
                for (size_t i = 0; i < alloc; i++) pat[i] = ((i >> bit) & 1) ? 0x40 : kUe4m3One;
                probed.copy_from_host(pat.data(), stream);
                other.copy_from_host(ones.data(), stream);
                matmul();
                for (int r = 0; r < L.rows; r++) {
                    float y = which ? __bfloat162float(hY[(size_t)r])
                                    : __bfloat162float(hY[(size_t)r * N]);
                    if (y == 32.0f) idx[r] |= 1LL << bit;
                    else if (y != 16.0f) clean = false;
                }
            }
            if (!clean) { printf("  %s-scale kb=%d: ambiguous readout\n", which ? "A" : "B", kb0); bad++; continue; }
            for (int r = 0; r < L.rows; r++)
                if ((size_t)idx[r] != L.offset(r, kb0)) {
                    if (bad < 8)
                        printf("  %s-scale (row=%d,kb=%d): cuBLASLt reads byte %lld, predicted %zu\n",
                               which ? "A" : "B", r, kb0, idx[r], L.offset(r, kb0));
                    bad++;
                }
        }
    }
    printf("NVFP4 (VEC16_UE4M3) scale layout: %s (%d wrong)\n",
           bad ? "WRONG" : "CONFIRMED, identical to the ue8m0 tile", bad);
    cudaFree(ws);
    cublasLtDestroy(lt);
    return bad;
}

// ── Per-shape plumbing ─────────────────────────────────────────────────

static auto bind(const W4A16CompiledKernel &kern, W4A16Problem &p) {
    return [&](const bf16 *X, const unsigned char *W, bf16 *Y, cudaStream_t s) {
        kern.fn(p.M(), p.N(), p.K(), X, W, Y, p.w_sf(), p.w_global(), p.workspace(), s);
    };
}

static bool smem_fits(const GemmConfig &cfg, size_t device_smem) {
    if (cfg.smem_bytes() <= device_smem) return true;
    fprintf(stderr, "config %s needs %.1f KB shared memory, device allows %.1f KB\n",
            cfg.name().c_str(), cfg.smem_bytes() / 1024.0, device_smem / 1024.0);
    return false;
}

struct ShapeRun {
    W4A16Problem p;
    cublasHandle_t cublas;
    double bf16_ms = 0.0;
    bool ok = true;

    ShapeRun(int M, int N, int K, const Args &a, cudaStream_t stream, int max_split_k)
        : p(M, N, K, stream, a.dist, max_split_k) {
        CHECK_CUBLAS(cublasCreate(&cublas));
        CHECK_CUBLAS(cublasSetStream(cublas, stream));
        AccuracyStats w = p.w_quant_error(), x = p.x_quant_error();
        printf("  quant err   W nvfp4: max_abs=%.3g rms_rel=%.2f%%   "
               "X bf16: rms_rel=%.2f%%   w_global=%.4g\n",
               w.max_abs, 100.0 * w.rms_rel, 100.0 * x.rms_rel, p.w_global());
        printf("  bytes       w4a16 %.1f MB vs bf16 %.1f MB  (%.2fx less)\n",
               p.bytes() / 1e6, p.bf16_bytes() / 1e6, p.bf16_bytes() / p.bytes());
        bf16_ms = p.time_bf16_baseline(cublas, a.bench);
        printf("  %-34s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s\n", "bf16 cuBLAS (unquantized)",
               bf16_ms, p.tflops(bf16_ms), p.bf16_bytes() / (bf16_ms * 1e-3) / 1e9);
    }
    ~ShapeRun() { cublasDestroy(cublas); }
};

static double run_one(ShapeRun &s, const W4A16CompiledKernel &kern, const Args &a) {
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
    printf("  %-34s %8.4f ms  %9.2f TFLOPS  %8.1f GB/s  (%.2fx bf16)\n",
           name.c_str(), ms, s.p.tflops(ms), s.p.gbps(ms), s.bf16_ms / ms);
    return ms;
}

// ── Modes ──────────────────────────────────────────────────────────────

static std::vector<GemmConfig> enumerate_configs(size_t max_smem) {
    std::vector<GemmConfig> out;
    for (auto &c : GemmConfig::enumerate(max_smem, ElemType::W4A16))
        if (c.is_skinny()) out.push_back(c);
    return out;
}

static int cmd_list_configs(size_t max_smem) {
    auto configs = enumerate_configs(max_smem);
    for (auto &c : configs)
        printf("%-40s smem %6.1f KB\n", c.name().c_str(), c.smem_bytes() / 1024.0);
    printf("%zu configs (max smem %.0f KB)\n", configs.size(), max_smem / 1024.0);
    return 0;
}

static int cmd_config(const Args &a, W4A16KernelJit &jit, size_t device_smem,
                      cudaStream_t stream, const std::vector<Shape> &shapes) {
    const GemmConfig &cfg = *a.config;
    if (auto why = cfg.validate(); !why.empty()) {
        fprintf(stderr, "invalid config %s: %s\n", cfg.name().c_str(), why.c_str());
        return 1;
    }
    if (!smem_fits(cfg, device_smem)) return 1;
    const W4A16CompiledKernel *kern = jit.get(cfg);
    if (!kern) { fprintf(stderr, "failed to build %s\n", cfg.name().c_str()); return 1; }
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!cfg.fits_shape(M, N, K)) { printf("  SKIP: config does not tile this shape\n"); continue; }
        ShapeRun s(M, N, K, a, stream, cfg.split_k);
        if (run_one(s, *kern, a) == 0.0) rc = 1;
    }
    return rc;
}

static int cmd_autotune(const Args &a, W4A16KernelJit &jit, AutotuneCache &cache,
                        size_t max_smem, cudaStream_t stream, const std::vector<Shape> &shapes) {
    auto all = enumerate_configs(max_smem);
    for (auto [M, N, K] : shapes) {
        printf("\n=== Autotune M=%d N=%d K=%d ===\n", M, N, K);
        if (!a.force)
            if (auto e = cache.lookup(M, N, K)) {
                printf("  cached: %s  %.4f ms\n", e->config.name().c_str(), e->time_ms);
                continue;
            }
        std::vector<GemmConfig> cands;
        int max_split_k = 1;
        for (auto &c : all)
            if (c.fits_shape(M, N, K)) { cands.push_back(c); max_split_k = std::max(max_split_k, c.split_k); }
        if (cands.empty()) { printf("  SKIP: no config tiles this shape\n"); continue; }
        printf("  %zu candidate configs\n", cands.size());
        auto kernels = jit.get_many(cands);
        if (kernels.empty()) { printf("  SKIP: nothing compiled\n"); continue; }
        ShapeRun s(M, N, K, a, stream, max_split_k);
        const W4A16CompiledKernel *best = nullptr;
        double best_ms = 1e30;
        for (auto *k : kernels) {
            double ms = run_one(s, *k, a);
            if (ms > 0.0 && ms < best_ms) { best_ms = ms; best = k; }
        }
        if (!best) { printf("  NO WORKING CONFIG\n"); continue; }
        printf("  BEST: %s  %.4f ms  %.1f GB/s  (%.2fx bf16 cuBLAS)\n",
               best->config.name().c_str(), best_ms, s.p.gbps(best_ms), s.bf16_ms / best_ms);
        cache.store(M, N, K, best->config, best_ms);
        cache.save(a.cache_path);
    }
    printf("\nAutotune complete -> %s\n", a.cache_path.c_str());
    return 0;
}

static int cmd_bench_cached(const Args &a, W4A16KernelJit &jit, const AutotuneCache &cache,
                            size_t device_smem, cudaStream_t stream, const std::vector<Shape> &shapes) {
    int rc = 0;
    for (auto [M, N, K] : shapes) {
        auto entry = cache.lookup(M, N, K);
        if (!entry) {
            fprintf(stderr, "no cached config for M=%d N=%d K=%d\n", M, N, K);
            fprintf(stderr, "hint: bench_w4a16 --shape %d,%d,%d --autotune\n", M, N, K);
            return 1;
        }
        const GemmConfig &cfg = entry->config;
        printf("\n=== M=%d N=%d K=%d  config=%s ===\n", M, N, K, cfg.name().c_str());
        if (!smem_fits(cfg, device_smem)) { rc = 1; continue; }
        const W4A16CompiledKernel *kern = jit.get(cfg);
        if (!kern) { fprintf(stderr, "  failed to build %s\n", cfg.name().c_str()); rc = 1; continue; }
        ShapeRun s(M, N, K, a, stream, cfg.split_k);
        if (run_one(s, *kern, a) == 0.0) rc = 1;
    }
    return rc;
}

int main(int argc, char **argv) {
    Args a;
    a.bench.tol = 0.03f; // nvfp4 weights land further from the reference than fp8
    if (!parse_args(argc, argv, a)) return 2;

    int device, major = 0, minor = 0, optin = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    size_t device_smem = (size_t)optin;
    size_t max_smem = a.max_smem_kb > 0 ? (size_t)a.max_smem_kb * 1024 : device_smem;

    if (a.list_configs) return cmd_list_configs(max_smem);
    if (a.probe_support) return probe_support();

    printf("device: sm_%d%d   W4A16 = bf16 activations x NVFP4 weights "
           "(e2m1 + ue4m3/16 + fp32), %s inputs\n", major, minor,
           a.dist == Dist::Normal ? "gaussian" : a.dist == Dist::Outlier ? "gaussian+outlier" : "uniform");

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    int rc = 0;
    if (a.probe_layout) {
        rc = probe_layout(stream) ? 1 : 0;
        cudaStreamDestroy(stream);
        return rc;
    }
    if (!a.skip_probe) rc |= probe_layout(stream) ? 1 : 0;

    JitOptions jopts = w4a16_jit_options();
    if (!a.jit_cache.empty()) jopts.cache_dir = a.jit_cache;
    jopts.jobs = a.jobs;
    jopts.force = a.force;
    jopts.verbose = a.verbose;
    W4A16KernelJit jit(jopts);

    if (a.config) {
        rc |= cmd_config(a, jit, device_smem, stream, a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else if (a.autotune) {
        AutotuneCache cache;
        cache.load(a.cache_path);
        rc |= cmd_autotune(a, jit, cache, max_smem, stream, a.shapes.empty() ? kDefaultShapes : a.shapes);
    } else {
        AutotuneCache cache;
        if (!cache.load(a.cache_path)) {
            fprintf(stderr, "no autotune cache at %s\nhint: bench_w4a16 --autotune\n", a.cache_path.c_str());
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
