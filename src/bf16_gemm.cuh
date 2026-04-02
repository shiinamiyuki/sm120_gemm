#pragma once
#include "common.h"
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;


struct TMADescriptor {

};

/* PSEUDOCODE for BF16Gemm. DO NOT DELETE
 void gemm(X, W, Y) {
    __shared__ bf16 shared_X[NUM_STAGES][BM * BK];
    __shared__ bf16 shared_W[NUM_STAGES][BK * BN];
    __shared__ mbarrier full_barrier[NUM_STAGES];
    __shared__ mbarrier empty_barrier[NUM_STAGES];
    __shared__ y_output_tile[BM * BN];
    int phase = 0;
    if (is_producer_wg) {
        if (warp_id_in_wg == 0) {
            int stage = 0;
            for (each tile) {
                wait(empty_barrier[stage], phase);
                tma_load(shared_X[stage], X_tile);
                tma_load(shared_W[stage], W_tile);
                signal(full_barrier[stage]);
                stage++;
                if (stage == NUM_STAGES) {
                    stage = 0;
                    phase ^= 1; // be careful! check nv docs
                }
            }
        } else {
            // other warps are idle for now
        }
    } else {
        int stage = 0;
        for (each tile) {
            wait(full_barrier[stage], phase);
            gemm(shared_X[stage], shared_W[stage], Y_tile);
            signal(empty_barrier[stage]);
            stage++;
            if (stage == NUM_STAGES) {
                stage = 0;
                phase ^= 1; // be careful! check nv docs
            }
        }
        store(y_output_tile, Y_tile);
        tma_store(Y, y_output_tile);
    }
 }

 */
template <int BM, int BN, int BK, int NUM_STAGES, int CWG>
struct BF16Gemm
{
    constexpr static int SHARED_MEM_SIZE = (BM * BK + BK * BN) * sizeof(bf16) * NUM_STAGES;

    __global__ void run(
        int M, int N, int K,
        const bf16 *__restrict__ X, // activation, row-majhor
        const bf16 *__restrict__ W, // weight, column-major
        bf16 *__restrict__ Y        // output
    )
    {
    }
};

template <int BM, int BN, int BK, int NUM_STAGES, int CWG>
void bf16_gemm(
    int M, int N, int K,
    const bf16 *__restrict__ X, // activation
    const bf16 *__restrict__ W, // weight
    bf16 *__restrict__ Y        // output
)
{
    // check M % BM == 0, N % BN == 0, K % BK == 0
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");
    }
}