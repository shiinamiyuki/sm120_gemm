#pragma once
#include "common.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <string>

using fp8e4m3 = __nv_fp8_e4m3;

#define CHECK_CUBLASLT(call)                                                     \
    do {                                                                         \
        cublasStatus_t st = (call);                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLASLt error in %s at line %d: %s (%s)\n",        \
                    __FILE__, __LINE__, cublasLtGetStatusName(st),               \
                    cublasLtGetStatusString(st));                                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ── Per-tensor-scaled FP8 GEMM through cuBLASLt ────────────────────────
//
// Computes   Y = (x_scale * X) @ (w_scale * W)^T   accumulating in FP32,
// with X (M x K) and W (N x K) row-major e4m3, and Y (M x N) row-major bf16.
// x_scale / w_scale are the per-tensor *dequantization* scales, i.e. the
// factors that turn the stored fp8 codes back into real values. cuBLASLt
// applies them to A and B before the multiply, so alpha stays 1.
//
// cuBLASLt only supports FP8 matmuls in "TN" form: op(A) = T, op(B) = N.
// Our layout already satisfies that, in exactly the same way the bf16 path
// does — pass W as A and X as B and compute Y^T = W^T * X^T in column-major
// terms:
//
//   A = W   stored K x N col-major (== W row-major N x K), op T -> N x K
//   B = X   stored K x M col-major (== X row-major M x K), op N -> K x M
//   D = Y   stored N x M col-major (== Y row-major M x N)
//
// so m = N, n = M, k = K. Because A is W, A's scale is w_scale.
class Fp8GemmLt {
public:
    Fp8GemmLt(int M, int N, int K,
              const float *w_scale_dev, const float *x_scale_dev,
              bool fast_accum = false, size_t workspace_bytes = 32u << 20)
        : M_(M), N_(N), K_(K), ws_bytes_(workspace_bytes) {
        CHECK_CUBLASLT(cublasLtCreate(&lt_));
        CHECK_CUDA(cudaMalloc(&ws_, ws_bytes_));

        CHECK_CUBLASLT(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        set(CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        set(CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale_dev, sizeof(w_scale_dev));
        set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_scale_dev, sizeof(x_scale_dev));
        if (fast_accum) {
            int8_t on = 1;
            set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &on, sizeof(on));
        }

        // rows, cols, ld — all column-major as described above.
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_, CUDA_R_8F_E4M3, K, N, K));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_, CUDA_R_8F_E4M3, K, M, K));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_, CUDA_R_16BF, N, M, N));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_, CUDA_R_16BF, N, M, N));

        cublasLtMatmulPreference_t pref;
        CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof(ws_bytes_)));

        cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        // Not CHECK'd: "no algorithm found" is a supportability answer, not a
        // failure, and the caller reports it.
        cublasLtMatmulAlgoGetHeuristic(lt_, op_, a_, b_, c_, d_, pref, 1, &heur, &returned);
        cublasLtMatmulPreferenceDestroy(pref);
        if (returned > 0) {
            algo_ = heur.algo;
            have_algo_ = true;
        }
    }

    ~Fp8GemmLt() {
        cublasLtMatrixLayoutDestroy(d_);
        cublasLtMatrixLayoutDestroy(c_);
        cublasLtMatrixLayoutDestroy(b_);
        cublasLtMatrixLayoutDestroy(a_);
        cublasLtMatmulDescDestroy(op_);
        cudaFree(ws_);
        cublasLtDestroy(lt_);
    }

    Fp8GemmLt(const Fp8GemmLt &) = delete;
    Fp8GemmLt &operator=(const Fp8GemmLt &) = delete;

    // False when cuBLASLt has no FP8 algorithm for this shape on this device.
    bool supported() const { return have_algo_; }

    void run(const fp8e4m3 *X, const fp8e4m3 *W, bf16 *Y, cudaStream_t stream) const {
        const float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLASLT(cublasLtMatmul(lt_, op_, &alpha,
                                      W, a_,
                                      X, b_,
                                      &beta,
                                      Y, c_, // unused at beta = 0, but must be valid
                                      Y, d_,
                                      &algo_, ws_, ws_bytes_, stream));
    }

private:
    void set(cublasLtMatmulDescAttributes_t attr, const void *buf, size_t bytes) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(op_, attr, buf, bytes));
    }

    int M_, N_, K_;
    size_t ws_bytes_;
    void *ws_ = nullptr;
    cublasLtHandle_t lt_{};
    cublasLtMatmulDesc_t op_{};
    cublasLtMatrixLayout_t a_{}, b_{}, c_{}, d_{};
    cublasLtMatmulAlgo_t algo_{};
    bool have_algo_ = false;
};
