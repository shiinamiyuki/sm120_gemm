#pragma once
// MXFP8 problem setup: block quantization, the exact FP32 reference over the
// quantized inputs, and timing. Reuses the per-tensor fp8 harness for the
// input distributions and the accuracy metrics.
#include "fp8_harness.h"
#include "mxfp8_cublaslt.h"

#include <cmath>
#include <vector>

// Exact FP32 reference for MXFP8:
//   Y[m][n] = sum_b 2^(ex[m][b]-127) * 2^(ew[n][b]-127) * sum_{j<32} X[m][32b+j]*W[n][32b+j]
// Scale codes arrive row-major here (sk per row); the swizzled layout is only
// cuBLASLt's business. fp8 -> float widening is exact and the scales are exact
// powers of two, so every difference from this is the GEMM's own error.
__global__ void mxfp8_naive_gemm_f32_kernel(int M, int N, int K, int sk,
                                            const fp8e4m3 *__restrict__ X,
                                            const fp8e4m3 *__restrict__ W,
                                            const unsigned char *__restrict__ ex,
                                            const unsigned char *__restrict__ ew,
                                            float *__restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int b = 0; b < sk; b++) {
        float p = 0.0f;
        for (int j = 0; j < 32; j++) {
            int k = b * 32 + j;
            p += float(X[(size_t)row * K + k]) * float(W[(size_t)col * K + k]);
        }
        acc += ue8m0_to_float(ex[(size_t)row * sk + b]) *
               ue8m0_to_float(ew[(size_t)col * sk + b]) * p;
    }
    Y[(size_t)row * N + col] = acc;
}

// ── MX quantization of one 32-element block ────────────────────────────
//
// Two rules, both power-of-two so the scale is exactly a ue8m0 code:
//
//   spec  (OCP MX, "floor"):  2^e,  e = floor(log2(amax)) - 8
//   noclip ("ceil"):          2^e,  e = ceil(log2(amax / 448))
//
// The spec rule puts amax/2^e in [256, 512), so every block whose amax lands
// in the top eighth of its binade has its largest element clipped to 448 --
// an error of up to 12.5% on precisely the element carrying the most energy.
// That costs more than it buys: e4m3 is a *floating point* format with ~3
// mantissa bits of relative precision at every binade, so pushing values up
// against the top of the range gains nothing. The ceil rule guarantees
// amax/2^e is in [224, 448] and never clips, which is what NVIDIA's
// Transformer Engine and the other production MXFP8 stacks do.
//
// The difference is worth ~2x in input RMS error, so noclip is the default
// here; --mx-spec-scale switches back for comparison.
inline unsigned char mx_block_scale_code(const float *v, int n, bool spec_rule = false) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) amax = std::max(amax, std::fabs(v[i]));
    if (!(amax > 0.0f)) return kUe8m0One; // all-zero block: scale is arbitrary
    int E;
    float m = std::frexp(amax, &E); // amax = m * 2^E, m in [0.5, 1)
    // floor(log2 amax) = E - 1; 448 = 0.875 * 2^9, so
    // ceil(log2(amax/448)) = E - 9 + (m > 0.875).
    int e = spec_rule ? (E - 1) - 8 : (E - 9) + (m > 0.875f ? 1 : 0);
    return (unsigned char)std::min(254, std::max(0, 127 + e));
}

// ── One MXFP8 problem ──────────────────────────────────────────────────
class MxFp8Problem {
public:
    MxFp8Problem(int M, int N, int K, cudaStream_t stream, Dist dist = Dist::Uniform,
                 bool spec_scale_rule = false, int max_split_k = 1)
        : M_(M), N_(N), K_(K), sk_(K / 32), stream_(stream), dist_(dist), spec_(spec_scale_rule),
          lx_(M, K), lw_(N, K) {
        if (K % 32 != 0) {
            fprintf(stderr, "MXFP8 needs K %% 32 == 0, got K=%d\n", K);
            exit(EXIT_FAILURE);
        }
        std::vector<float> hX, hW;
        generate_inputs(hX, hW);

        std::vector<fp8e4m3> qX(hX.size()), qW(hW.size());
        cX_.resize((size_t)M * sk_);
        cW_.resize((size_t)N * sk_);
        quantize(hX, M, qX, cX_);
        quantize(hW, N, qW, cW_);

        mx_err_ = dequant_error(hX, qX, cX_, M) ;
        {
            AccuracyStats w = dequant_error(hW, qW, cW_, N);
            mx_err_.max_abs = std::max(mx_err_.max_abs, w.max_abs);
            mx_err_.rms_rel = std::max(mx_err_.rms_rel, w.rms_rel);
        }
        pt_err_ = per_tensor_error(hX);
        {
            AccuracyStats w = per_tensor_error(hW);
            pt_err_.max_abs = std::max(pt_err_.max_abs, w.max_abs);
            pt_err_.rms_rel = std::max(pt_err_.rms_rel, w.rms_rel);
        }

        // Rotate over enough input sets to overflow L2 between timed iterations.
        int device, l2_bytes;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
        num_bufs_ = std::max<size_t>(1, (2 * (size_t)l2_bytes + bytes_per_set() - 1) / bytes_per_set());
        for (size_t b = 0; b < num_bufs_; b++) {
            X_.emplace_back(qX.size());
            W_.emplace_back(qW.size());
            Y_.emplace_back((size_t)M * N);
            X_.back().copy_from_host(qX.data(), stream_);
            W_.back().copy_from_host(qW.data(), stream_);
        }

        // Scale tensors: swizzled for cuBLASLt, row-major for the reference.
        upload_swizzled(cX_, lx_, sfX_);
        upload_swizzled(cW_, lw_, sfW_);
        rowX_ = CUDABuffer<unsigned char>(cX_.size());
        rowW_ = CUDABuffer<unsigned char>(cW_.size());
        rowX_.copy_from_host(cX_.data(), stream_);
        rowW_.copy_from_host(cW_.data(), stream_);

        // Split-K partials, sized from the largest SPLIT_K in the candidate
        // set so no allocation ever lands inside a timed loop.
        workspace_ = CUDABuffer<float>((size_t)std::max(1, max_split_k) * M * N);
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        build_reference();
    }

    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }
    int sk() const { return sk_; }
    const BlockScaleLayout &x_layout() const { return lx_; }
    const BlockScaleLayout &w_layout() const { return lw_; }
    float *workspace() const { return workspace_.data; }
    const void *x_sf() const { return sfX_.data; }
    const void *w_sf() const { return sfW_.data; }
    // What block quantization costs at the inputs, versus a single per-tensor scale.
    AccuracyStats mx_input_error() const { return mx_err_; }
    AccuracyStats per_tensor_input_error() const { return pt_err_; }

    double tflops(double ms) const { return 2.0 * M_ * N_ * K_ / (ms * 1e-3) / 1e12; }
    // fp8 operands plus their ue8m0 scales in, bf16 out.
    double bytes() const {
        return (double)M_ * K_ + (double)N_ * K_
               + (double)lx_.bytes() + (double)lw_.bytes()
               + (double)M_ * N_ * sizeof(bf16);
    }
    double gbps(double ms) const { return bytes() / (ms * 1e-3) / 1e9; }

    template <class F>
    CheckResult check(F &&fn, float tol) {
        size_t n = (size_t)M_ * N_;
        CheckResult r;
        CHECK_CUDA(cudaMemsetAsync(Y_[0].data, 0, n * sizeof(bf16), stream_));
        fn(X_[0].data, W_[0].data, Y_[0].data, stream_);
        if (cudaError_t err = cudaStreamSynchronize(stream_); err != cudaSuccess) {
            cudaGetLastError();
            r.launched = false;
            r.reason = std::string("launch failed: ") + cudaGetErrorString(err);
            return r;
        }
        last_output_.resize(n);
        Y_[0].copy_to_host(last_output_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        r.vs_fp32 = compare(last_output_.data(), h_Y_ref_.data(), n);
        r.vs_cublas = r.vs_fp32;
        r.ok = r.vs_fp32.rel_err < tol;
        if (!r.ok) r.reason = "rel err " + std::to_string(r.vs_fp32.rel_err) + " >= tol";
        return r;
    }

    template <class F>
    double time(F &&fn, const BenchOptions &opt) {
        return bench_ms([&] {
            size_t b = buf_idx_++ % num_bufs_;
            fn(X_[b].data, W_[b].data, Y_[b].data, stream_);
        }, stream_, opt.warmup, opt.repeat);
    }

    void print_mismatches(float tol, int limit = 8) const {
        size_t n = std::min(last_output_.size(), h_Y_ref_.size());
        for (size_t i = 0, shown = 0; i < n && shown < (size_t)limit; i++) {
            float y = __bfloat162float(last_output_[i]), ref = h_Y_ref_[i];
            if (fabsf(y - ref) / fmaxf(fmaxf(fabsf(y), fabsf(ref)), 1.0f) > tol) {
                printf("    [%zu] ours=%f ref=%f\n", i, y, ref);
                shown++;
            }
        }
    }

private:
    void generate_inputs(std::vector<float> &hX, std::vector<float> &hW) const {
        hX.resize((size_t)M_ * K_);
        hW.resize((size_t)N_ * K_);
        srand(42);
        auto sample = [&] {
            if (dist_ == Dist::Uniform) return (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = (float)rand() / RAND_MAX;
            float g = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
            if (dist_ == Dist::Outlier && rand() < RAND_MAX / 1000) g *= 1000.0f;
            return g;
        };
        for (auto &v : hX) v = sample();
        for (auto &v : hW) v = sample();
    }

    // Row-blocked MX quantization: one ue8m0 scale per 32 elements along K.
    void quantize(const std::vector<float> &src, int rows,
                  std::vector<fp8e4m3> &q, std::vector<unsigned char> &codes) const {
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < sk_; b++) {
                const float *v = &src[(size_t)r * K_ + b * 32];
                unsigned char c = mx_block_scale_code(v, 32, spec_);
                codes[(size_t)r * sk_ + b] = c;
                float inv = ldexpf(1.0f, 127 - (int)c);
                for (int j = 0; j < 32; j++)
                    q[(size_t)r * K_ + b * 32 + j] = fp8e4m3(v[j] * inv);
            }
    }

    AccuracyStats dequant_error(const std::vector<float> &src, const std::vector<fp8e4m3> &q,
                                const std::vector<unsigned char> &codes, int rows) const {
        std::vector<float> deq(src.size());
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < sk_; b++) {
                float s = ue8m0_to_float(codes[(size_t)r * sk_ + b]);
                for (int j = 0; j < 32; j++) {
                    size_t i = (size_t)r * K_ + b * 32 + j;
                    deq[i] = float(q[i]) * s;
                }
            }
        return accuracy(deq.data(), src.data(), src.size());
    }

    // The same tensor under one per-tensor scale, for contrast.
    AccuracyStats per_tensor_error(const std::vector<float> &src) const {
        float amax = 0.0f;
        for (float v : src) amax = std::max(amax, std::fabs(v));
        float s = amax / kE4M3Max;
        std::vector<float> deq(src.size());
        for (size_t i = 0; i < src.size(); i++) deq[i] = float(fp8e4m3(src[i] / s)) * s;
        return accuracy(deq.data(), src.data(), src.size());
    }

    void upload_swizzled(const std::vector<unsigned char> &codes, const BlockScaleLayout &L,
                         CUDABuffer<unsigned char> &dst) {
        std::vector<unsigned char> sw(L.bytes(), kUe8m0One); // padding decodes to 1.0
        for (int r = 0; r < L.rows; r++)
            for (int b = 0; b < L.sk; b++)
                sw[L.offset(r, b)] = codes[(size_t)r * L.sk + b];
        dst = CUDABuffer<unsigned char>(sw.size());
        dst.copy_from_host(sw.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    size_t bytes_per_set() const {
        return (size_t)M_ * K_ + (size_t)N_ * K_ + (size_t)M_ * N_ * sizeof(bf16);
    }

    void build_reference() {
        size_t n = (size_t)M_ * N_;
        h_Y_ref_.resize(n);
        CUDABuffer<float> dY(n);
        mxfp8_naive_gemm_f32_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
            M_, N_, K_, sk_, X_[0].data, W_[0].data, rowX_.data, rowW_.data, dY.data);
        CHECK_CUDA(cudaGetLastError());
        dY.copy_to_host(h_Y_ref_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    int M_, N_, K_, sk_;
    cudaStream_t stream_;
    Dist dist_;
    bool spec_ = false;
    BlockScaleLayout lx_, lw_;

    size_t num_bufs_ = 1, buf_idx_ = 0;
    std::vector<CUDABuffer<fp8e4m3>> X_, W_;
    std::vector<CUDABuffer<bf16>> Y_;
    CUDABuffer<unsigned char> sfX_, sfW_; // swizzled, for cuBLASLt
    CUDABuffer<unsigned char> rowX_, rowW_; // row-major, for the reference
    CUDABuffer<float> workspace_;
    std::vector<unsigned char> cX_, cW_;

    AccuracyStats mx_err_, pt_err_;
    std::vector<float> h_Y_ref_;
    std::vector<bf16> last_output_;
};
