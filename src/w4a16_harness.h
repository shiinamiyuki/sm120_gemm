#pragma once
// W4A16 problem setup: NVFP4-quantized weights against bf16 activations, the
// exact FP32 reference over them, and the bf16 cuBLAS baseline.
//
// cuBLASLt has no W4A16 path -- see bench_w4a16 --probe-support, which walks
// 3072 (Atype, Btype, Dtype, compute, op, shape, scale-mode) combinations and
// finds zero. Neither does the hardware: `mma` rejects e2m1 x bf16. So the
// baseline is bf16 cuBLAS, i.e. the unquantized GEMM that W4A16 replaces, and
// the bar is the weight bandwidth it saves.
#include "fp8_harness.h" // CUDABuffer/compare/BenchOptions, plus Dist, AccuracyStats, accuracy, naive_grid
#include "block_scale.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cmath>
#include <vector>

// ── NVFP4 weight quantization ──────────────────────────────────────────
//
// Two levels, as the OCP/NVIDIA recipe defines them:
//
//   w_global  = amax(W) / (6 * 448)          one fp32 for the whole tensor
//   s[n][b]   = ue4m3( blockamax / 6 / w_global )   one per 16 elements of K
//   code      = e2m1( w / (s * w_global) )
//   w        ~= code * s * w_global
//
// 6 is e2m1's largest magnitude and 448 ue4m3's, so w_global is exactly the
// factor that puts the largest block scale at the top of ue4m3's range.
// Unlike MX's ue8m0 the block scale has 3 mantissa bits, so it tracks the
// block amax closely and there is no power-of-two rounding loss.

__host__ __device__ __forceinline__ float e2m1_code_to_float(unsigned int nib) {
    // +-{0, .5, 1, 1.5, 2, 3, 4, 6}: 1 sign, 2 exponent (bias 1), 1 mantissa.
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = mag[nib & 7];
    return (nib & 8) ? -v : v;
}

inline unsigned char float_to_e2m1(float v) {
    return (unsigned char)__nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
}
// ue4m3 is e4m3 with the sign bit clear.
inline unsigned char float_to_ue4m3(float v) {
    if (!(v > 0.0f)) return 0;
    return (unsigned char)(__nv_cvt_float_to_fp8(std::min(v, kUe4m3Max), __NV_SATFINITE,
                                                 __NV_E4M3) & 0x7f);
}

// Exact FP32 reference: bf16 X times the dequantized NVFP4 W, accumulated in
// FP32. bf16 -> float and fp4 -> float are both exact, so any difference from
// this is the GEMM's own error, not the quantizer's.
__global__ void w4a16_naive_gemm_f32_kernel(int M, int N, int K, int sk,
                                            const bf16 *__restrict__ X,
                                            const unsigned char *__restrict__ Wp,
                                            const unsigned char *__restrict__ sfw,
                                            float w_global,
                                            float *__restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int b = 0; b < sk; b++) {
        float p = 0.0f;
        for (int j = 0; j < 16; j++) {
            int k = b * 16 + j;
            size_t idx = (size_t)col * K + k;
            unsigned char byte = Wp[idx >> 1];
            unsigned int nib = (k & 1) ? (byte >> 4) : (byte & 0xf);
            p += __bfloat162float(X[(size_t)row * K + k]) * e2m1_code_to_float(nib);
        }
        acc += ue4m3_to_float(sfw[(size_t)col * sk + b]) * p;
    }
    Y[(size_t)row * N + col] = acc * w_global;
}

// ── One W4A16 problem ──────────────────────────────────────────────────
class W4A16Problem {
public:
    W4A16Problem(int M, int N, int K, cudaStream_t stream, Dist dist = Dist::Uniform,
                 int max_split_k = 1)
        : M_(M), N_(N), K_(K), sk_(K / 16), stream_(stream), dist_(dist), lw_(N, K, 16) {
        if (K % 16 != 0) {
            fprintf(stderr, "W4A16 needs K %% 16 == 0, got K=%d\n", K);
            exit(EXIT_FAILURE);
        }
        std::vector<float> hX, hW;
        generate_inputs(hX, hW);

        // Activations: bf16, no scaling.
        std::vector<bf16> qX(hX.size());
        for (size_t i = 0; i < hX.size(); i++) qX[i] = __float2bfloat16(hX[i]);

        // Weights: NVFP4.
        std::vector<unsigned char> packed((size_t)N * K / 2, 0);
        cW_.resize((size_t)N * sk_);
        quantize_w(hW, packed);

        // What each side costs, measured against the fp32 originals.
        {
            std::vector<float> deq(hX.size());
            for (size_t i = 0; i < hX.size(); i++) deq[i] = __bfloat162float(qX[i]);
            x_err_ = accuracy(deq.data(), hX.data(), hX.size());
            deq.assign(hW.size(), 0.0f);
            for (int n = 0; n < N; n++)
                for (int b = 0; b < sk_; b++) {
                    float s = ue4m3_to_float(cW_[(size_t)n * sk_ + b]) * w_global_;
                    for (int j = 0; j < 16; j++) {
                        size_t i = (size_t)n * K + b * 16 + j;
                        unsigned char byte = packed[i >> 1];
                        unsigned int nib = ((b * 16 + j) & 1) ? (byte >> 4) : (byte & 0xf);
                        deq[i] = e2m1_code_to_float(nib) * s;
                    }
                }
            w_err_ = accuracy(deq.data(), hW.data(), hW.size());
            // The bf16 weights the baseline reads, for a like-for-like timing.
            hWb_.resize(hW.size());
            for (size_t i = 0; i < hW.size(); i++) hWb_[i] = __float2bfloat16(hW[i]);
        }

        int device, l2_bytes;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
        num_bufs_ = std::max<size_t>(1, (2 * (size_t)l2_bytes + bytes_per_set() - 1) / bytes_per_set());
        for (size_t b = 0; b < num_bufs_; b++) {
            X_.emplace_back(qX.size());
            W_.emplace_back(packed.size());
            Y_.emplace_back((size_t)M * N);
            X_.back().copy_from_host(qX.data(), stream_);
            W_.back().copy_from_host(packed.data(), stream_);
        }
        // Single bf16 copy of W: the baseline is bandwidth-bound on it, and
        // one copy is already 4x the fp4 set, so it does not rotate.
        Wb_ = CUDABuffer<bf16>(hWb_.size());
        Wb_.copy_from_host(hWb_.data(), stream_);

        upload_swizzled();
        row_sf_ = CUDABuffer<unsigned char>(cW_.size());
        row_sf_.copy_from_host(cW_.data(), stream_);
        workspace_ = CUDABuffer<float>((size_t)std::max(1, max_split_k) * M * N);
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        build_reference();
    }

    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }
    const BlockScaleLayout &w_layout() const { return lw_; }
    const void *w_sf() const { return sf_.data; }
    float w_global() const { return w_global_; }
    float *workspace() const { return workspace_.data; }
    AccuracyStats w_quant_error() const { return w_err_; }
    AccuracyStats x_quant_error() const { return x_err_; }

    double tflops(double ms) const { return 2.0 * M_ * N_ * K_ / (ms * 1e-3) / 1e12; }
    // bf16 activations in, packed fp4 weights plus their ue4m3 scales, bf16 out.
    double bytes() const {
        return (double)M_ * K_ * sizeof(bf16) + (double)N_ * K_ / 2
               + (double)lw_.bytes() + (double)M_ * N_ * sizeof(bf16);
    }
    double gbps(double ms) const { return bytes() / (ms * 1e-3) / 1e9; }
    // What the bf16 baseline has to move, for the ratio that actually matters.
    double bf16_bytes() const {
        return ((double)M_ * K_ + (double)N_ * K_ + (double)M_ * N_) * sizeof(bf16);
    }

    // fn(X, W_packed, Y, stream)
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

    // The unquantized GEMM this is meant to replace: bf16 X times bf16 W.
    double time_bf16_baseline(cublasHandle_t h, const BenchOptions &opt) {
        return bench_ms([&] {
            size_t b = buf_idx_++ % num_bufs_;
            cublas_gemm(h, M_, N_, K_, X_[b].data, Wb_.data, Y_[b].data);
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

    void quantize_w(const std::vector<float> &hW, std::vector<unsigned char> &packed) {
        float amax = 0.0f;
        for (float v : hW) amax = std::max(amax, std::fabs(v));
        w_global_ = amax > 0 ? amax / (kE2m1Max * kUe4m3Max) : 1.0f;

        for (int n = 0; n < N_; n++)
            for (int b = 0; b < sk_; b++) {
                const float *v = &hW[(size_t)n * K_ + b * 16];
                float bmax = 0.0f;
                for (int j = 0; j < 16; j++) bmax = std::max(bmax, std::fabs(v[j]));
                unsigned char c = float_to_ue4m3(bmax / kE2m1Max / w_global_);
                cW_[(size_t)n * sk_ + b] = c;
                float s = ue4m3_to_float(c) * w_global_;
                float inv = s > 0 ? 1.0f / s : 0.0f;
                for (int j = 0; j < 16; j++) {
                    int k = b * 16 + j;
                    size_t idx = (size_t)n * K_ + k;
                    unsigned char nib = float_to_e2m1(v[j] * inv) & 0xf;
                    // Low nibble is the even element -- probed against cuBLASLt.
                    if (k & 1) packed[idx >> 1] |= (unsigned char)(nib << 4);
                    else packed[idx >> 1] |= nib;
                }
            }
    }

    void upload_swizzled() {
        std::vector<unsigned char> sw(lw_.bytes(), kUe4m3One); // padding decodes to 1.0
        for (int r = 0; r < lw_.rows; r++)
            for (int b = 0; b < lw_.sk; b++)
                sw[lw_.offset(r, b)] = cW_[(size_t)r * lw_.sk + b];
        sf_ = CUDABuffer<unsigned char>(sw.size());
        sf_.copy_from_host(sw.data(), stream_);
    }

    size_t bytes_per_set() const {
        return (size_t)M_ * K_ * sizeof(bf16) + (size_t)N_ * K_ / 2
               + (size_t)M_ * N_ * sizeof(bf16);
    }

    void build_reference() {
        size_t n = (size_t)M_ * N_;
        h_Y_ref_.resize(n);
        CUDABuffer<float> dY(n);
        w4a16_naive_gemm_f32_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
            M_, N_, K_, sk_, X_[0].data, W_[0].data, row_sf_.data, w_global_, dY.data);
        CHECK_CUDA(cudaGetLastError());
        dY.copy_to_host(h_Y_ref_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    int M_, N_, K_, sk_;
    cudaStream_t stream_;
    Dist dist_;
    BlockScaleLayout lw_;

    size_t num_bufs_ = 1, buf_idx_ = 0;
    std::vector<CUDABuffer<bf16>> X_;
    std::vector<CUDABuffer<unsigned char>> W_; // packed e2m1
    std::vector<CUDABuffer<bf16>> Y_;
    CUDABuffer<bf16> Wb_;                      // dequantized, for the bf16 baseline
    CUDABuffer<unsigned char> sf_, row_sf_;    // swizzled / row-major ue4m3
    CUDABuffer<float> workspace_;

    std::vector<unsigned char> cW_;
    std::vector<bf16> hWb_;
    float w_global_ = 1.0f;
    AccuracyStats w_err_, x_err_;
    std::vector<float> h_Y_ref_;
    std::vector<bf16> last_output_;
};
