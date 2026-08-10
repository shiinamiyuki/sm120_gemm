// Translation unit compiled on the fly by KernelJit (see kernel_jit.h).
// One .so per configuration; the tile parameters arrive as -D defines.
#include "bf16_gemm.cuh"

#if !defined(GEMM_BM) || !defined(GEMM_BN) || !defined(GEMM_BK) ||     \
    !defined(GEMM_STAGES) || !defined(GEMM_CWG) || !defined(GEMM_WM) || \
    !defined(GEMM_WN) || !defined(GEMM_SPLIT_K)
#error "kernel_entry.cu requires -DGEMM_{BM,BN,BK,STAGES,CWG,WM,WN,SPLIT_K}"
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
                         float *workspace, cudaStream_t stream) {
    auto *x = static_cast<const bf16 *>(X);
    auto *w = static_cast<const bf16 *>(W);
    auto *y = static_cast<bf16 *>(Y);
    BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>::run(
        M, N, K, x, w, y, workspace, stream);
}
