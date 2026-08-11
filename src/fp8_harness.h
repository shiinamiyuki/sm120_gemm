#pragma once
#include "bench_harness.h" // CUDABuffer, compare, ErrorPair, CheckResult, BenchOptions
#include "fp8_cublaslt.h"

#include <cmath>
#include <vector>

// Largest finite e4m3 magnitude in the OCP/NVIDIA encoding. e4m3 has no
// infinities, so the conversion saturates here rather than overflowing.
static constexpr float kE4M3Max = 448.0f;

// ── Per-tensor quantization model ──────────────────────────────────────
//
//   scale = amax(real) / 448          (the *dequantization* scale)
//   code  = e4m3(real / scale)        saturating, round-to-nearest-even
//   real ~= scale * float(code)
//
// so a GEMM over the codes must be scaled by x_scale * w_scale to land back
// in real units. That single product is what cuBLASLt applies through its A
// and B scale pointers.

// Reference: Y = scale * (X @ W^T) with X row-major MxK and W row-major NxK,
// both e4m3, accumulated in FP32. fp8 -> float widening is exact, so this is
// the exact value the FP8 GEMM is supposed to produce; any difference is the
// GEMM's own error (accumulation order, output rounding), not quantization.
__global__ void fp8_naive_gemm_f32_kernel(int M, int N, int K,
                                          const fp8e4m3 *__restrict__ X,
                                          const fp8e4m3 *__restrict__ W,
                                          float *__restrict__ Y,
                                          float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += float(X[(size_t)row * K + k]) * float(W[(size_t)col * K + k]);
    Y[(size_t)row * N + col] = acc * scale;
}

// Same GEMM over the un-quantized fp32 inputs, for measuring what per-tensor
// fp8 costs in absolute accuracy.
__global__ void f32_naive_gemm_kernel(int M, int N, int K,
                                      const float *__restrict__ X,
                                      const float *__restrict__ W,
                                      float *__restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += X[(size_t)row * K + k] * W[(size_t)col * K + k];
    Y[(size_t)row * N + col] = acc;
}

inline dim3 naive_grid(int M, int N) { return dim3((N + 15) / 16, (M + 15) / 16); }

// Quantization accuracy is reported as an RMS-relative norm,
//     ||got - ref||_2 / ||ref||_2
// not as a max elementwise relative error. Both the inputs (uniform around 0)
// and the outputs (sums of random-sign products) contain values arbitrarily
// close to zero, and a max-relative metric is entirely determined by those:
// it reads ~1.0 no matter how good the quantizer is, and says nothing.
struct AccuracyStats {
    double max_abs = 0.0;
    double rms_rel = 0.0;
};

template <typename A, typename B>
inline AccuracyStats accuracy(const A *got, const B *ref, size_t n) {
    auto to_f = [](auto v) {
        if constexpr (std::is_same_v<decltype(v), bf16>) return __bfloat162float(v);
        else if constexpr (std::is_same_v<decltype(v), fp8e4m3>) return float(v);
        else return (float)v;
    };
    double se = 0.0, sr = 0.0;
    AccuracyStats s;
    for (size_t i = 0; i < n; i++) {
        double g = to_f(got[i]), r = to_f(ref[i]);
        double d = g - r;
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        se += d * d;
        sr += r * r;
    }
    s.rms_rel = sr > 0 ? std::sqrt(se / sr) : 0.0;
    return s;
}

// Input distribution, for probing when a single per-tensor scale stops being
// enough.
//
// Note e4m3 is a *floating point* format: it keeps ~3 mantissa bits of
// relative precision across its whole normal range, 2^-6 up to 448, i.e.
// about 4.5 orders of magnitude. So unlike per-tensor int8, pushing amax up
// costs nothing until the bulk of the values fall through into subnormals.
// That is why Normal barely differs from Uniform below, and why breaking
// per-tensor scaling takes Outlier-scale dynamic range.
enum class Dist {
    Uniform, // uniform[-1, 1]
    Normal,  // unit Gaussian; amax lands at ~4-5 sigma
    Outlier, // Gaussian with 0.1% of elements scaled by 1000x, mimicking the
             // activation outliers that per-channel/per-block scaling targets
};

// ── One FP8 problem: quantized inputs, references, timing ──────────────
class Fp8Problem {
public:
    Fp8Problem(int M, int N, int K, cudaStream_t stream, Dist dist = Dist::Uniform)
        : M_(M), N_(N), K_(K), stream_(stream), dist_(dist) {
        std::vector<float> hX, hW;
        generate_inputs(hX, hW);

        x_scale_ = amax(hX) / kE4M3Max;
        w_scale_ = amax(hW) / kE4M3Max;

        // Quantize on the host: the e4m3 constructor is __host__ __device__
        // and saturates, so this matches what a device-side quantizer does.
        std::vector<fp8e4m3> qX(hX.size()), qW(hW.size());
        for (size_t i = 0; i < hX.size(); i++) qX[i] = fp8e4m3(hX[i] / x_scale_);
        for (size_t i = 0; i < hW.size(); i++) qW[i] = fp8e4m3(hW[i] / w_scale_);
        {
            // Dequantize back and measure against the originals.
            std::vector<float> deq(std::max(hX.size(), hW.size()));
            for (size_t i = 0; i < hX.size(); i++) deq[i] = float(qX[i]) * x_scale_;
            AccuracyStats ex = accuracy(deq.data(), hX.data(), hX.size());
            for (size_t i = 0; i < hW.size(); i++) deq[i] = float(qW[i]) * w_scale_;
            AccuracyStats ew = accuracy(deq.data(), hW.data(), hW.size());
            input_err_.max_abs = std::max(ex.max_abs, ew.max_abs);
            input_err_.rms_rel = std::max(ex.rms_rel, ew.rms_rel);
        }

        // Rotate over enough input sets to overflow L2 between timed iterations.
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        int l2_bytes;
        CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
        num_bufs_ = std::max<size_t>(1, (2 * (size_t)l2_bytes + bytes_per_set() - 1) / bytes_per_set());

        for (size_t b = 0; b < num_bufs_; b++) {
            X_.emplace_back(qX.size());
            W_.emplace_back(qW.size());
            Y_.emplace_back((size_t)M * N);
            X_.back().copy_from_host(qX.data(), stream_);
            W_.back().copy_from_host(qW.data(), stream_);
        }

        // cuBLASLt wants the scales as device scalars.
        d_x_scale_ = CUDABuffer<float>(1);
        d_w_scale_ = CUDABuffer<float>(1);
        d_x_scale_.copy_from_host(&x_scale_, stream_);
        d_w_scale_.copy_from_host(&w_scale_, stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        build_fp8_reference();
    }

    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }
    float x_scale() const { return x_scale_; }
    float w_scale() const { return w_scale_; }
    const float *x_scale_dev() const { return d_x_scale_.data; }
    const float *w_scale_dev() const { return d_w_scale_.data; }
    // Error introduced by quantizing the inputs themselves.
    AccuracyStats input_quant_error() const { return input_err_; }

    double tflops(double ms) const { return 2.0 * M_ * N_ * K_ / (ms * 1e-3) / 1e12; }
    // fp8 operands in, bf16 result out.
    double bytes() const {
        return (double)M_ * K_ + (double)N_ * K_ + (double)M_ * N_ * sizeof(bf16);
    }
    double gbps(double ms) const { return bytes() / (ms * 1e-3) / 1e9; }

    // fn(X, W, Y, stream) writes Y = dequantized X @ W^T as bf16.
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
        r.vs_cublas = r.vs_fp32; // only one reference here
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

    // How far the exact FP8 result sits from the full-precision one — the cost
    // of per-tensor quantization itself, independent of any GEMM kernel.
    // Allocates fp32 copies of both operands, so it is opt-in.
    AccuracyStats quantization_cost_at_output() {
        std::vector<float> hX, hW;
        generate_inputs(hX, hW);
        std::vector<float> h_hp((size_t)M_ * N_);
        {
            CUDABuffer<float> dX(hX.size()), dW(hW.size()), dY((size_t)M_ * N_);
            dX.copy_from_host(hX.data(), stream_);
            dW.copy_from_host(hW.data(), stream_);
            f32_naive_gemm_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
                M_, N_, K_, dX.data, dW.data, dY.data);
            CHECK_CUDA(cudaGetLastError());
            dY.copy_to_host(h_hp.data(), stream_);
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
        return accuracy(h_Y_ref_.data(), h_hp.data(), h_hp.size());
    }

private:
    // Deterministic so quantization_cost_at_output() can regenerate the same
    // inputs instead of keeping fp32 copies alive for the whole run.
    void generate_inputs(std::vector<float> &hX, std::vector<float> &hW) const {
        hX.resize((size_t)M_ * K_);
        hW.resize((size_t)N_ * K_);
        srand(42);
        auto sample = [&] {
            if (dist_ == Dist::Uniform) return (float)rand() / RAND_MAX * 2.0f - 1.0f;
            // Box-Muller, unit variance.
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = (float)rand() / RAND_MAX;
            float g = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
            if (dist_ == Dist::Outlier && rand() < RAND_MAX / 1000) g *= 1000.0f;
            return g;
        };
        for (auto &v : hX) v = sample();
        for (auto &v : hW) v = sample();
    }

    static float amax(const std::vector<float> &v) {
        float m = 0.0f;
        for (float x : v) m = std::max(m, std::fabs(x));
        return m;
    }

    size_t bytes_per_set() const {
        return (size_t)M_ * K_ + (size_t)N_ * K_ + (size_t)M_ * N_ * sizeof(bf16);
    }

    void build_fp8_reference() {
        size_t n = (size_t)M_ * N_;
        h_Y_ref_.resize(n);
        CUDABuffer<float> dY(n);
        fp8_naive_gemm_f32_kernel<<<naive_grid(M_, N_), dim3(16, 16), 0, stream_>>>(
            M_, N_, K_, X_[0].data, W_[0].data, dY.data, x_scale_ * w_scale_);
        CHECK_CUDA(cudaGetLastError());
        dY.copy_to_host(h_Y_ref_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    int M_, N_, K_;
    cudaStream_t stream_;
    size_t num_bufs_ = 1;
    size_t buf_idx_ = 0;

    std::vector<CUDABuffer<fp8e4m3>> X_, W_;
    std::vector<CUDABuffer<bf16>> Y_;
    CUDABuffer<float> d_x_scale_, d_w_scale_;

    float x_scale_ = 1.0f, w_scale_ = 1.0f;
    AccuracyStats input_err_;
    Dist dist_ = Dist::Uniform;
    std::vector<float> h_Y_ref_;   // exact fp32 result over the quantized inputs
    std::vector<bf16> last_output_;
};
