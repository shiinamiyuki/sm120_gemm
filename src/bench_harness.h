#pragma once
#include "common.h"
#include "kernel_jit.h"

#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t err = (call);                                          \
        if (err != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error in %s at line %d: %d\n",            \
                    __FILE__, __LINE__, (int)err);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

template <typename T>
struct CUDABuffer {
    T *data = nullptr;
    size_t size = 0;

    CUDABuffer() = default;
    explicit CUDABuffer(size_t n) : size(n) {
        if (n) CHECK_CUDA(cudaMalloc(&data, n * sizeof(T)));
    }
    CUDABuffer(const CUDABuffer &) = delete;
    CUDABuffer &operator=(const CUDABuffer &) = delete;
    CUDABuffer(CUDABuffer &&o) noexcept : data(o.data), size(o.size) {
        o.data = nullptr;
        o.size = 0;
    }
    CUDABuffer &operator=(CUDABuffer &&o) noexcept {
        std::swap(data, o.data);
        std::swap(size, o.size);
        return *this;
    }
    ~CUDABuffer() { cudaFree(data); }

    void copy_from_host(const T *host, cudaStream_t stream = nullptr) {
        CHECK_CUDA(cudaMemcpyAsync(data, host, size * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void copy_to_host(T *host, cudaStream_t stream = nullptr) const {
        CHECK_CUDA(cudaMemcpyAsync(host, data, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
};

inline void rand_bf16(bf16 *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        data[i] = __float2bfloat16(val);
    }
}

// X: MxK row-major, W: KxN col-major, Y: MxN row-major.
// cuBLAS is column-major, so we compute Y^T = W^T * X^T.
inline void cublas_gemm(cublasHandle_t handle, int M, int N, int K,
                        const bf16 *X, const bf16 *W, bf16 *Y) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K, // W is KxN col-major; W^T is NxK
        X, CUDA_R_16BF, K, // X row-major MxK = X^T col-major KxM
        &beta,
        Y, CUDA_R_16BF, N, // Y row-major MxN = Y^T col-major NxM
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Naive fp32 reference: Y = X * W, X row-major MxK, W col-major KxN.
__global__ void naive_gemm_f32_kernel(int M, int N, int K,
                                             const bf16 *__restrict__ X,
                                             const bf16 *__restrict__ W,
                                             float *__restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += __bfloat162float(X[row * K + k]) * __bfloat162float(W[k + col * K]);
    Y[row * N + col] = acc;
}

inline void naive_gemm_f32(int M, int N, int K, const bf16 *X, const bf16 *W,
                           float *Y, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_gemm_f32_kernel<<<grid, block, 0, stream>>>(M, N, K, X, W, Y);
    CHECK_CUDA(cudaGetLastError());
}

struct ErrorPair {
    float abs_err = 0.0f;
    float rel_err = 0.0f;
};

// Elementwise max absolute / relative error, with the relative denominator
// floored at 1 so near-zero outputs do not dominate.
template <typename A, typename B>
inline ErrorPair compare(const A *a, const B *b, size_t n) {
    auto to_f = [](auto v) {
        if constexpr (std::is_same_v<decltype(v), bf16>) return __bfloat162float(v);
        else return (float)v;
    };
    ErrorPair e;
    for (size_t i = 0; i < n; ++i) {
        float x = to_f(a[i]), y = to_f(b[i]);
        float abs_e = fabsf(x - y);
        e.abs_err = std::max(e.abs_err, abs_e);
        e.rel_err = std::max(e.rel_err, abs_e / fmaxf(fmaxf(fabsf(x), fabsf(y)), 1.0f));
    }
    return e;
}

struct CheckResult {
    bool ok = false;
    bool launched = true; // false if the kernel itself failed to run
    std::string reason;   // populated when !ok
    ErrorPair vs_cublas;
    ErrorPair vs_fp32;
};

struct BenchOptions {
    uint32_t warmup = 5;
    uint32_t repeat = 20;
    float tol = 0.1f;
    bool check = true;
};

// ── One problem shape: inputs, references, and L2-flushing timing ──────
//
// Owns everything the three old benchmark loops each set up by hand. Buffer
// set 0 doubles as the correctness scratch, so no extra M*K/K*N allocation.
class Problem {
public:
    Problem(int M, int N, int K, cublasHandle_t handle, cudaStream_t stream, int max_split_k)
        : M_(M), N_(N), K_(K), handle_(handle), stream_(stream) {
        std::vector<bf16> h_X((size_t)M * K), h_W((size_t)K * N);
        srand(42);
        rand_bf16(h_X.data(), h_X.size());
        rand_bf16(h_W.data(), h_W.size());

        // Rotate over enough input sets to overflow L2 between timed iterations.
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        int l2_bytes;
        CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
        size_t per_set = ((size_t)M * K + (size_t)K * N + (size_t)M * N) * sizeof(bf16);
        num_bufs_ = std::max<size_t>(1, (2 * (size_t)l2_bytes + per_set - 1) / per_set);

        for (size_t b = 0; b < num_bufs_; b++) {
            X_.emplace_back((size_t)M * K);
            W_.emplace_back((size_t)K * N);
            Y_.emplace_back((size_t)M * N);
            X_.back().copy_from_host(h_X.data(), stream_);
            W_.back().copy_from_host(h_W.data(), stream_);
        }
        workspace_ = CUDABuffer<float>(std::max(1, max_split_k) * (size_t)M * N);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    int M() const { return M_; }
    int N() const { return N_; }
    int K() const { return K_; }
    size_t num_bufs() const { return num_bufs_; }
    double tflops(double ms) const { return 2.0 * M_ * N_ * K_ / (ms * 1e-3) / 1e12; }

    // Compulsory traffic for the operation: both operands in, result out.
    // Split-K workspace round-trips are deliberately not counted, so their
    // cost shows up as reduced effective bandwidth rather than an inflated
    // figure. For skinny shapes this, not TFLOPS, is the number that matters:
    // arithmetic intensity is ~M flop/byte, so small M is purely bandwidth work.
    double bytes() const {
        return ((double)M_ * K_ + (double)N_ * K_ + (double)M_ * N_) * sizeof(bf16);
    }
    double gbps(double ms) const { return bytes() / (ms * 1e-3) / 1e9; }

    // cuBLAS and naive-fp32 references, computed at most once per shape.
    // The device-side reference buffers are released again straight away;
    // only the host copies are kept.
    void ensure_references() {
        if (have_refs_) return;
        size_t n = (size_t)M_ * N_;
        h_Y_ref_.resize(n);
        h_Y_f32_.resize(n);
        {
            CUDABuffer<bf16> d_ref(n);
            CHECK_CUDA(cudaMemsetAsync(d_ref.data, 0, n * sizeof(bf16), stream_));
            cublas_gemm(handle_, M_, N_, K_, X_[0].data, W_[0].data, d_ref.data);
            d_ref.copy_to_host(h_Y_ref_.data(), stream_);
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
        {
            CUDABuffer<float> d_f32(n);
            naive_gemm_f32(M_, N_, K_, X_[0].data, W_[0].data, d_f32.data, stream_);
            d_f32.copy_to_host(h_Y_f32_.data(), stream_);
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
        cublas_vs_fp32_ = compare(h_Y_ref_.data(), h_Y_f32_.data(), n);
        have_refs_ = true;
    }

    ErrorPair cublas_vs_fp32() const { return cublas_vs_fp32_; }

    // Run once and compare against the fp32 reference. A launch failure is
    // reported as a failed check rather than aborting, so an autotune sweep
    // can keep going.
    CheckResult check(const CompiledKernel &kern, float tol) {
        ensure_references();
        size_t n = (size_t)M_ * N_;
        CheckResult r;

        CHECK_CUDA(cudaMemsetAsync(Y_[0].data, 0, n * sizeof(bf16), stream_));
        CHECK_CUDA(cudaMemsetAsync(workspace_.data, 0, workspace_.size * sizeof(float), stream_));
        kern.fn(M_, N_, K_, X_[0].data, W_[0].data, Y_[0].data, workspace_.data, stream_);
        if (cudaError_t err = cudaStreamSynchronize(stream_); err != cudaSuccess) {
            cudaGetLastError(); // clear sticky error
            r.launched = false;
            r.reason = std::string("launch failed: ") + cudaGetErrorString(err);
            return r;
        }

        last_output_.resize(n);
        Y_[0].copy_to_host(last_output_.data(), stream_);
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        r.vs_cublas = compare(last_output_.data(), h_Y_ref_.data(), n);
        r.vs_fp32 = compare(last_output_.data(), h_Y_f32_.data(), n);
        r.ok = r.vs_fp32.rel_err < tol;
        if (!r.ok) r.reason = "rel err " + std::to_string(r.vs_fp32.rel_err) + " >= tol";
        return r;
    }

    // Report up to `limit` elements of the last check() that differ by more
    // than tol.
    void print_mismatches(float tol, int limit = 10) const {
        size_t n = std::min(last_output_.size(), h_Y_f32_.size());
        for (size_t i = 0, shown = 0; i < n && shown < (size_t)limit; i++) {
            float y = __bfloat162float(last_output_[i]), ref = h_Y_f32_[i];
            if (fabsf(y - ref) / fmaxf(fmaxf(fabsf(y), fabsf(ref)), 1.0f) > tol) {
                printf("    [%zu] ours=%f ref=%f\n", i, y, ref);
                shown++;
            }
        }
    }

    double time(const CompiledKernel &kern, const BenchOptions &opt) {
        return bench_ms([&] {
            size_t b = buf_idx_++ % num_bufs_;
            kern.fn(M_, N_, K_, X_[b].data, W_[b].data, Y_[b].data, workspace_.data, stream_);
        }, stream_, opt.warmup, opt.repeat);
    }

    double time_cublas(const BenchOptions &opt) {
        return bench_ms([&] {
            size_t b = buf_idx_++ % num_bufs_;
            cublas_gemm(handle_, M_, N_, K_, X_[b].data, W_[b].data, Y_[b].data);
        }, stream_, opt.warmup, opt.repeat);
    }

private:
    int M_, N_, K_;
    cublasHandle_t handle_;
    cudaStream_t stream_;
    size_t num_bufs_ = 1;
    size_t buf_idx_ = 0;

    std::vector<CUDABuffer<bf16>> X_, W_, Y_;
    CUDABuffer<float> workspace_;

    std::vector<bf16> h_Y_ref_;
    std::vector<float> h_Y_f32_;
    std::vector<bf16> last_output_; // result of the most recent check()
    ErrorPair cublas_vs_fp32_;
    bool have_refs_ = false;
};
