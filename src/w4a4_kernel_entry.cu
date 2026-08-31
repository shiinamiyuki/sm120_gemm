// Translation unit compiled on the fly by KernelJit. Tile parameters arrive as
// -D defines. W4A4 has only the tensor-core family: sm_120a's block-scaled
// e2m1 MMA takes both operands packed, so there is no CUDA-core variant worth
// having and no dequantize step to trade against.
#include "w4a4_gemm.cuh"

#if !defined(GEMM_BM) || !defined(GEMM_BN) || !defined(GEMM_BK) ||      \
    !defined(GEMM_STAGES) || !defined(GEMM_CWG) || !defined(GEMM_WM) || \
    !defined(GEMM_WN) || !defined(GEMM_YS)
#error "w4a4_kernel_entry.cu requires -DGEMM_{BM,BN,BK,STAGES,CWG,WM,WN,YS}"
#endif

static constexpr int BM = GEMM_BM;
static constexpr int BN = GEMM_BN;
static constexpr int BK = GEMM_BK;
static constexpr int NUM_STAGES = GEMM_STAGES;
static constexpr int CWG = GEMM_CWG;
static constexpr int WARP_M = GEMM_WM;
static constexpr int WARP_N = GEMM_WN;
static constexpr int Y_SLICES = GEMM_YS;

extern "C" void gemm_run(int M, int N, int K,
                         const void *X, const void *W, void *Y,
                         const void *x_sf, const void *w_sf, float out_scale,
                         float *workspace, cudaStream_t stream) {
    (void)workspace; // no split-k path
    W4A4GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, Y_SLICES>::run(
        M, N, K,
        static_cast<const unsigned char *>(X), static_cast<const unsigned char *>(W),
        static_cast<bf16 *>(Y),
        static_cast<const unsigned char *>(x_sf), static_cast<const unsigned char *>(w_sf),
        out_scale, stream);
}
