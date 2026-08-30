// Translation unit compiled on the fly by KernelJit, the MXFP8 counterpart of
// fp8_kernel_entry.cu. Tile parameters arrive as -D defines.
//
#include "mxfp8_gemm_tinym.cuh" // also pulls in mxfp8_gemm.cuh's dependencies
#include "mxfp8_gemm.cuh"

#if !defined(GEMM_BM) || !defined(GEMM_BN) || !defined(GEMM_BK) ||      \
    !defined(GEMM_STAGES) || !defined(GEMM_CWG) || !defined(GEMM_WM) || \
    !defined(GEMM_WN) || !defined(GEMM_SPLIT_K)
#error "mxfp8_kernel_entry.cu requires -DGEMM_{BM,BN,BK,STAGES,CWG,WM,WN,SPLIT_K}"
#endif

// Kernel family selector: 0 = MXFP8GemmMMA (tensor cores), 1 = MXFP8GemmTinyM
// (CUDA cores, M <= BM). GEMM_WM/GEMM_WN are unused when tiny-M is selected;
// GEMM_SPLIT_K is unused by the MMA family, which has no split-k epilogue.
#ifndef GEMM_TINYM
#define GEMM_TINYM 0
#endif

static constexpr int BM = GEMM_BM;
static constexpr int BN = GEMM_BN;
static constexpr int BK = GEMM_BK;
static constexpr int NUM_STAGES = GEMM_STAGES;
static constexpr int CWG = GEMM_CWG;
static constexpr int WARP_M = GEMM_WM;
static constexpr int WARP_N = GEMM_WN;
static constexpr int SPLIT_K = GEMM_SPLIT_K;

extern "C" void gemm_run(int M, int N, int K,
                         const void *X, const void *W, void *Y,
                         const void *x_sf, const void *w_sf,
                         float *workspace, cudaStream_t stream) {
    auto *x = static_cast<const fp8e4m3 *>(X);
    auto *w = static_cast<const fp8e4m3 *>(W);
    auto *y = static_cast<bf16 *>(Y);
    auto *xs = static_cast<const unsigned char *>(x_sf);
    auto *ws = static_cast<const unsigned char *>(w_sf);
#if GEMM_TINYM
    MXFP8GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::run(
        M, N, K, x, w, y, xs, ws, workspace, stream);
#else
    MXFP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::run(
        M, N, K, x, w, y, xs, ws, stream);
#endif
}
