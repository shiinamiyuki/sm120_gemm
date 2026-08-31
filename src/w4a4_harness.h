#pragma once
// W4A4 problem setup: both operands NVFP4-quantized, the exact FP32 reference
// over them, and the bf16 cuBLAS baseline.
//
// W4A16 (w4a16_harness.h) quantizes only the weights, because no mixed
// 4-bit x 16-bit MMA exists. W4A4 has a native instruction -- see
// nvfp4_mma.cuh -- so the activations are quantized too, and everything the
// weight side already needed (a global scale, a per-16 ue4m3 block scale, a
// swizzled scale tensor) now exists twice.
//
// The asymmetry worth remembering: the weight scales are blocked along K for a
// tensor laid out [N][K], the activation scales along K for [M][K]. Both are
// therefore BlockScaleLayout(rows, K, 16), just with different row counts, and
// the MMA wants one ue4m3 per 16 k for each of A's 16 rows and B's 8 columns.
#include "fp8_harness.h" // CUDABuffer/compare/BenchOptions, Dist, AccuracyStats, accuracy, naive_grid
#include "block_scale.h"
#include "nvfp4_mma.cuh" // e2m1_to_float / ue4m3_to_float

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cmath>
#include <vector>

// ── NVFP4 quantization, the OCP/NVIDIA two-level recipe ────────────────
//
//   g       = amax(T) / (6 * 448)                one fp32 for the tensor
//   s[r][b] = ue4m3( blockamax / 6 / g )         one per 16 elements of K
//   code    = e2m1( t / (s * g) )
//   t      ~= code * s * g
//
// 6 is e2m1's largest magnitude and 448 ue4m3's, so g is exactly the factor
// that puts the largest block scale at the top of ue4m3's range.

inline unsigned char w4a4_float_to_e2m1(float v) {
    return (unsigned char)__nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
}
// ue4m3 is e4m3 with the sign bit clear.
inline unsigned char w4a4_float_to_ue4m3(float v) {
    if (!(v > 0.0f)) return 0;
    return (unsigned char)(__nv_cvt_float_to_fp8(std::min(v, kUe4m3Max), __NV_SATFINITE,
                                                 __NV_E4M3) & 0x7f);
}

// One NVFP4 tensor: packed codes, row-major ue4m3 scales, and the global fp32.
struct NVFP4Tensor {
    std::vector<unsigned char> packed; // rows * K / 2
    std::vector<unsigned char> sf;     // rows * (K/16), row-major
    float global = 1.0f;
};

// Quantize a rows x K fp32 tensor. Blocks run along K, matching both the
// storage order and the MMA's k-extent.
inline NVFP4Tensor w4a4_quantize(const std::vector<float> &src, int rows, int K) {
    const int sk = K / 16;
    NVFP4Tensor t;
    t.packed.assign((size_t)rows * K / 2, 0);
    t.sf.assign((size_t)rows * sk, 0);

    float amax = 0.0f;
    for (float v : src) amax = std::max(amax, std::fabs(v));
    t.global = amax > 0 ? amax / (kE2m1Max * kUe4m3Max) : 1.0f;

    for (int r = 0; r < rows; r++)
        for (int b = 0; b < sk; b++) {
            const float *v = &src[(size_t)r * K + b * 16];
            float bmax = 0.0f;
            for (int j = 0; j < 16; j++) bmax = std::max(bmax, std::fabs(v[j]));
            const unsigned char c = w4a4_float_to_ue4m3(bmax / kE2m1Max / t.global);
            t.sf[(size_t)r * sk + b] = c;
            const float s = ue4m3_to_float(c) * t.global;
            const float inv = s > 0 ? 1.0f / s : 0.0f;
            for (int j = 0; j < 16; j++) {
                const int k = b * 16 + j;
                const size_t idx = (size_t)r * K + k;
                const unsigned char nib = w4a4_float_to_e2m1(v[j] * inv) & 0xf;
                // Low nibble is the even element.
                if (k & 1) t.packed[idx >> 1] |= (unsigned char)(nib << 4);
                else       t.packed[idx >> 1] |= nib;
            }
        }
    return t;
}

// Dequantize back to fp32, for measuring what the quantizer cost.
inline std::vector<float> w4a4_dequantize(const NVFP4Tensor &t, int rows, int K) {
    const int sk = K / 16;
    std::vector<float> out((size_t)rows * K);
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < sk; b++) {
            const float s = ue4m3_to_float(t.sf[(size_t)r * sk + b]) * t.global;
            for (int j = 0; j < 16; j++) {
                const int k = b * 16 + j;
                const size_t idx = (size_t)r * K + k;
                const unsigned char byte = t.packed[idx >> 1];
                const unsigned int nib = (k & 1) ? (byte >> 4) : (byte & 0xf);
                out[idx] = e2m1_to_float(nib) * s;
            }
        }
    return out;
}

// Exact FP32 reference over the quantized operands. Every fp4 -> float and
// ue4m3 -> float step is exact, so any difference from this is the GEMM's own
// error and not the quantizer's. Scales are read row-major here; the swizzled
// copies exist for kernels, and --selfcheck is what proves the two agree.
__global__ void w4a4_naive_gemm_f32_kernel(int M, int N, int K, int sk,
                                           const unsigned char *__restrict__ Xp,
                                           const unsigned char *__restrict__ Wp,
                                           const unsigned char *__restrict__ sfx,
                                           const unsigned char *__restrict__ sfw,
                                           float x_global, float w_global,
                                           float *__restrict__ Y) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int b = 0; b < sk; b++) {
        float p = 0.0f;
        for (int j = 0; j < 16; j++) {
            const int k = b * 16 + j;
            const size_t xi = (size_t)row * K + k, wi = (size_t)col * K + k;
            const unsigned int xn = (k & 1) ? (Xp[xi >> 1] >> 4) : (Xp[xi >> 1] & 0xf);
            const unsigned int wn = (k & 1) ? (Wp[wi >> 1] >> 4) : (Wp[wi >> 1] & 0xf);
            p += e2m1_to_float(xn) * e2m1_to_float(wn);
        }
        acc += ue4m3_to_float(sfx[(size_t)row * sk + b]) *
               ue4m3_to_float(sfw[(size_t)col * sk + b]) * p;
    }
    Y[(size_t)row * N + col] = acc * x_global * w_global;
}

// Same maths, but reading the *swizzled* scale tensors a kernel is given.
// If this disagrees with the row-major reference the layout plumbing is wrong,
// which is the failure a kernel would otherwise discover the hard way.
__global__ void w4a4_naive_gemm_swizzled_kernel(int M, int N, int K, int sk,
                                                const unsigned char *__restrict__ Xp,
                                                const unsigned char *__restrict__ Wp,
                                                const unsigned char *__restrict__ sfx,
                                                const unsigned char *__restrict__ sfw,
                                                BlockScaleLayout lx, BlockScaleLayout lw,
                                                float x_global, float w_global,
                                                float *__restrict__ Y) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int b = 0; b < sk; b++) {
        float p = 0.0f;
        for (int j = 0; j < 16; j++) {
            const int k = b * 16 + j;
            const size_t xi = (size_t)row * K + k, wi = (size_t)col * K + k;
            const unsigned int xn = (k & 1) ? (Xp[xi >> 1] >> 4) : (Xp[xi >> 1] & 0xf);
            const unsigned int wn = (k & 1) ? (Wp[wi >> 1] >> 4) : (Wp[wi >> 1] & 0xf);
            p += e2m1_to_float(xn) * e2m1_to_float(wn);
        }
        acc += ue4m3_to_float(sfx[lx.offset(row, b)]) *
               ue4m3_to_float(sfw[lw.offset(col, b)]) * p;
    }
    Y[(size_t)row * N + col] = acc * x_global * w_global;
}

// ── One W4A4 problem ───────────────────────────────────────────────────
class W4A4Problem {
public:
    W4A4Problem(int M, int N, int K, cudaStream_t stream, Dist dist = Dist::Uniform,
                int max_split_k = 1)
        : M_(M), N_(N), K_(K), sk_(K / 16), stream_(stream), dist_(dist),
          lx_(M, K, 16), lw_(N, K, 16) {
        if (K % 16 != 0) {
            fprintf(stderr, "W4A4 needs K %% 16 == 0, got K=%d\n", K);
            exit(EXIT_FAILURE);
        }
        std::vector<float> hX, hW;
        generate_inputs(hX, hW);

        qx_ = w4a4_quantize(hX, M_, K_);
        qw_ = w4a4_quantize(hW, N_, K_);
        x_err_ = accuracy(w4a4_dequantize(qx_, M_, K_).data(), hX.data(), hX.size());
        w_err_ = accuracy(w4a4_dequantize(qw_, N_, K_).data(), hW.data(), hW.size());

        // bf16 copies for the unquantized baseline this is meant to replace.
        std::vector<bf16> hXb(hX.size()), hWb(hW.size());
        for (size_t i = 0; i < hX.size(); i++) hXb[i] = __float2bfloat16(hX[i]);
        for (size_t i = 0; i < hW.size(); i++) hWb[i] = __float2bfloat16(hW[i]);

        int device, l2_bytes;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
        num_bufs_ = std::max<size_t>(1, (2 * (size_t)l2_bytes + bytes_per_set() - 1) / bytes_per_set());
        for (size_t b = 0; b < num_bufs_; b++) {
            X_.emplace_back(qx_.packed.size());
            W_.emplace_back(qw_.packed.size());
            Y_.emplace_back((size_t)M * N);
            X_.back().copy_from_host(qx_.packed.data(), stream_);
            W_.back().copy_from_host(qw_.packed.data(), stream_);
        }
        Xb_ = CUDABuffer<bf16>(hXb.size());
        Wb_ = CUDABuffer<bf16>(hWb.size());
        Xb_.copy_from_host(hXb.data(), stream_);
        Wb_.copy_from_host(hWb.data(), stream_);

        row_sfx_ = CUDABuffer<unsigned char>(qx_.sf.size());
        row_sfw_ = CUDABuffer<unsigned char>(qw_.sf.size());
        row_sfx_.copy_from_host(qx_.sf.data(), stream_);
        row_sfw_.copy_from_host(qw_.sf.data(), stream_);
        sfx_ = upload_swizzled(lx_, qx_.sf);
        sfw_ = upload_swizzled(lw_, qw_.sf);

        workspace_ = CUDABuffer<float>((size_t)std::max(1, max_split_k) * M * N);
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        build_reference();
    }

    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }
    const BlockScaleLayout &x_layout() const { return lx_; }
    const BlockScaleLayout &w_layout() const { return lw_; }
    const void *x_sf() const { return sfx_.data; }
    const void *w_sf() const { return sfw_.data; }
    float x_global() const { return qx_.global; }
    float w_global() const { return qw_.global; }
    float *workspace() const { return workspace_.data; }
    AccuracyStats x_quant_error() const { return x_err_; }
    AccuracyStats w_quant_error() const { return w_err_; }

    double tflops(double ms) const { return 2.0 * M_ * N_ * K_ / (ms * 1e-3) / 1e12; }
    // Both operands packed fp4, both scale tensors, bf16 out.
    double bytes() const {
        return (double)M_ * K_ / 2 + (double)N_ * K_ / 2 +
               (double)lx_.bytes() + (double)lw_.bytes() +
               (double)M_ * N_ * sizeof(bf16);
    }
    double gbps(double ms) const { return bytes() / (ms * 1e-3) / 1e9; }
    double bf16_bytes() const {
        return ((double)M_ * K_ + (double)N_ * K_ + (double)M_ * N_) * sizeof(bf16);
    }

    // fn(X_packed, W_packed, Y, stream)
    template <class F>
    CheckResult check(F &&fn, float tol) {
        const size_t n = (size_t)M_ * N_;
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
            const size_t b = buf_idx_++ % num_bufs_;
            fn(X_[b].data, W_[b].data, Y_[b].data, stream_);
        }, stream_, opt.warmup, opt.repeat);
    }

    double time_bf16_baseline(cublasHandle_t h, const BenchOptions &opt) {
        return bench_ms([&] {
            const size_t b = buf_idx_++ % num_bufs_;
            cublas_gemm(h, M_, N_, K_, Xb_.data, Wb_.data, Y_[b].data);
        }, stream_, opt.warmup, opt.repeat);
    }

    // Does the swizzled scale tensor a kernel is handed decode to the same
    // thing as the row-major one the reference used? Returns the max relative
    // difference, which must be 0 -- both paths read identical bytes, only
    // from different addresses.
    double selfcheck_swizzled_scales() {
        const size_t n = (size_t)M_ * N_;
        CUDABuffer<float> dY(n);
        w4a4_naive_gemm_swizzled_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
            M_, N_, K_, sk_, X_[0].data, W_[0].data,
            (const unsigned char *)sfx_.data, (const unsigned char *)sfw_.data,
            lx_, lw_, qx_.global, qw_.global, dY.data);
        CHECK_CUDA(cudaGetLastError());
        std::vector<float> got(n);
        dY.copy_to_host(got.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        double worst = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double d = std::fabs((double)got[i] - (double)h_Y_ref_[i]);
            worst = std::max(worst, d / std::max(1e-6, std::fabs((double)h_Y_ref_[i])));
        }
        return worst;
    }

    void print_mismatches(float tol, int limit = 8) const {
        const size_t n = std::min(last_output_.size(), h_Y_ref_.size());
        for (size_t i = 0, shown = 0; i < n && shown < (size_t)limit; i++) {
            const float y = __bfloat162float(last_output_[i]), ref = h_Y_ref_[i];
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

    CUDABuffer<unsigned char> upload_swizzled(const BlockScaleLayout &l,
                                              const std::vector<unsigned char> &row_major) {
        std::vector<unsigned char> sw(l.bytes(), kUe4m3One); // padding decodes to 1.0
        for (int r = 0; r < l.rows; r++)
            for (int b = 0; b < l.sk; b++)
                sw[l.offset(r, b)] = row_major[(size_t)r * l.sk + b];
        CUDABuffer<unsigned char> buf(sw.size());
        buf.copy_from_host(sw.data(), stream_);
        return buf;
    }

    size_t bytes_per_set() const {
        return (size_t)M_ * K_ / 2 + (size_t)N_ * K_ / 2 + (size_t)M_ * N_ * sizeof(bf16);
    }

    void build_reference() {
        const size_t n = (size_t)M_ * N_;
        h_Y_ref_.resize(n);
        CUDABuffer<float> dY(n);
        w4a4_naive_gemm_f32_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
            M_, N_, K_, sk_, X_[0].data, W_[0].data, row_sfx_.data, row_sfw_.data,
            qx_.global, qw_.global, dY.data);
        CHECK_CUDA(cudaGetLastError());
        dY.copy_to_host(h_Y_ref_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    int M_, N_, K_, sk_;
    cudaStream_t stream_;
    Dist dist_;
    BlockScaleLayout lx_, lw_;

    size_t num_bufs_ = 1, buf_idx_ = 0;
    std::vector<CUDABuffer<unsigned char>> X_, W_; // packed e2m1
    std::vector<CUDABuffer<bf16>> Y_;
    CUDABuffer<bf16> Xb_, Wb_;                     // dequantized, for the bf16 baseline
    CUDABuffer<unsigned char> sfx_, sfw_;          // swizzled ue4m3
    CUDABuffer<unsigned char> row_sfx_, row_sfw_;  // row-major ue4m3
    CUDABuffer<float> workspace_;

    NVFP4Tensor qx_, qw_;
    AccuracyStats x_err_, w_err_;
    std::vector<float> h_Y_ref_;
    std::vector<bf16> last_output_;
};
