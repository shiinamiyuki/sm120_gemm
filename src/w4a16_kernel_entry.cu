// Translation unit compiled on the fly by KernelJit. Tile parameters arrive as
// -D defines. Two families: tiny-M on CUDA cores (GEMM_TINYM) and tiny-M on
// tensor cores (GEMM_TINYM_TC). There is no mixed 4-bit x 16-bit MMA, so the
// tensor-core family dequantizes W to bf16 fragments in registers first -- see
// w4a16_gemm_tinym_tc.cuh for why that is cheaper than it sounds at M >= 4.
#include "w4a16_gemm_tinym.cuh"
#include "w4a16_gemm_tinym_tc.cuh"

#if !defined(GEMM_BM) || !defined(GEMM_BN) || !defined(GEMM_BK) ||     \
    !defined(GEMM_STAGES) || !defined(GEMM_CWG) || !defined(GEMM_SPLIT_K)
#error "w4a16_kernel_entry.cu requires -DGEMM_{BM,BN,BK,STAGES,CWG,SPLIT_K}"
#endif

#ifndef GEMM_TINYM
#define GEMM_TINYM 0
#endif
#ifndef GEMM_TINYM_TC
#define GEMM_TINYM_TC 0
#endif
#if !GEMM_TINYM && !GEMM_TINYM_TC
#error "W4A16 builds only the skinny families (GEMM_TINYM=1 or GEMM_TINYM_TC=1)"
#endif

static constexpr int BM = GEMM_BM;
static constexpr int BN = GEMM_BN;
static constexpr int BK = GEMM_BK;
static constexpr int NUM_STAGES = GEMM_STAGES;
static constexpr int CWG = GEMM_CWG;
static constexpr int SPLIT_K = GEMM_SPLIT_K;

extern "C" void gemm_run(int M, int N, int K,
                         const void *X, const void *W, void *Y,
                         const void *w_sf, float w_global,
                         float *workspace, cudaStream_t stream) {
#if GEMM_TINYM_TC
    W4A16GemmTinyMTC<BN, NUM_STAGES, CWG, SPLIT_K>::run(
#else
    W4A16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::run(
#endif
        M, N, K,
        static_cast<const bf16 *>(X), static_cast<const unsigned char *>(W),
        static_cast<bf16 *>(Y), static_cast<const unsigned char *>(w_sf),
        w_global, workspace, stream);
}
