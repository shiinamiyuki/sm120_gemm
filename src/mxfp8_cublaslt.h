#pragma once
#include "common.h"
#include "mxfp8_scale.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using fp8e4m3 = __nv_fp8_e4m3;

#ifndef CHECK_CUBLASLT
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
#endif

// ════════════════════════════════════════════════════════════════════════
// MXFP8 (OCP microscaling) through cuBLASLt
// ════════════════════════════════════════════════════════════════════════
//
// MXFP8 replaces the single per-tensor scale with one ue8m0 (power-of-two)
// scale per 32 consecutive elements along K:
//
//   Y[m][n] = sum over k-blocks b of
//               2^(ex[m][b]-127) * 2^(ew[n][b]-127) * sum_{j<32} X[m][32b+j] * W[n][32b+j]
//
// so the dynamic range is per-32-element rather than per-tensor. Because the
// scales are exact powers of two they fold into the tensor core's own
// block-scale operands instead of costing a multiply.
//
// ── The cuBLASLt entry point ──────────────────────────────────────────
//
// Same cublasLtMatmul as the per-tensor path; three attributes change it
// from "scalar" to "microscaled":
//
//   CUBLASLT_MATMUL_DESC_A_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
//   CUBLASLT_MATMUL_DESC_B_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
//   A/B_SCALE_POINTER now point at a *tensor* of ue8m0 bytes, not a float
//
// (cublasLtMatmulMatrixScale_t also has VEC16_UE4M3 for NVFP4, VEC128_32F and
// BLK128x128_32F for DeepSeek-style scaling, and OUTER_VEC_32F for per-row /
// per-column. VEC32_UE8M0 is the MX one.)
//
// The operand roles are unchanged from Fp8GemmLt: A = W, B = X, and we
// compute Y^T = W^T * X^T in column-major terms, so m = N, n = M, k = K.
// A's scale tensor therefore has N rows and B's has M rows; both are scaled
// along K, which is the innermost dimension of both operands, as
// VEC32_UE8M0 requires.
//
// ── Scale-factor tensor layout ────────────────────────────────────────
//
// Not row-major. cuBLASLt wants the swizzled layout the Blackwell tensor
// cores consume directly (the same one CUTLASS calls the SFA/SFB layout).
// Reverse-engineered by bit-plane probing; the formula lives in
// mxfp8_scale.h, and `bench_mxfp8 --probe-layout` re-derives it against the
// hardware on a new toolkit.

// ── The baseline ──────────────────────────────────────────────────────
class MxFp8GemmLt {
public:
    // w_sf / x_sf are device pointers to swizzled scale tensors laid out by
    // MxScaleLayout(N, K) and MxScaleLayout(M, K) respectively.
    MxFp8GemmLt(int M, int N, int K, const void *w_sf, const void *x_sf,
                bool fast_accum = false, size_t workspace_bytes = 32u << 20)
        : M_(M), N_(N), K_(K), ws_bytes_(workspace_bytes) {
        CHECK_CUBLASLT(cublasLtCreate(&lt_));
        CHECK_CUDA(cudaMalloc(&ws_, ws_bytes_));

        CHECK_CUBLASLT(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        set(CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        set(CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

        int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
        set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
        set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sf, sizeof(w_sf));
        set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_sf, sizeof(x_sf));
        if (fast_accum) {
            int8_t on = 1;
            set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &on, sizeof(on));
        }

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
        // Not CHECK'd: "no algorithm" is a supportability answer, not a failure.
        cublasLtMatmulAlgoGetHeuristic(lt_, op_, a_, b_, c_, d_, pref, 1, &heur, &returned);
        cublasLtMatmulPreferenceDestroy(pref);
        if (returned > 0) {
            algo_ = heur.algo;
            have_algo_ = true;
        }
    }

    ~MxFp8GemmLt() {
        cublasLtMatrixLayoutDestroy(d_);
        cublasLtMatrixLayoutDestroy(c_);
        cublasLtMatrixLayoutDestroy(b_);
        cublasLtMatrixLayoutDestroy(a_);
        cublasLtMatmulDescDestroy(op_);
        cudaFree(ws_);
        cublasLtDestroy(lt_);
    }

    MxFp8GemmLt(const MxFp8GemmLt &) = delete;
    MxFp8GemmLt &operator=(const MxFp8GemmLt &) = delete;

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
