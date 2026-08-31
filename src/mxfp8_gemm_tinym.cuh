#pragma once
#include "fp8_gemm_tinym.cuh" // fp8e4m3, TMA/mbarrier helpers, load_fp8x16_f32
#include "block_scale.h"

// ════════════════════════════════════════════════════════════════════════
// MXFP8GemmTinyM — skinny MXFP8 GEMM (M <= BM, typically M <= 8) on CUDA cores
// ════════════════════════════════════════════════════════════════════════
//
// FP8GemmTinyM with per-tensor scaling replaced by MX block scaling: one
// ue8m0 scale per 32 consecutive elements along K, per row of X and per row
// of W. The pipeline, the tiling and the epilogue are untouched; what changes
// is where the scale enters the arithmetic.
//
// ── Why this family first ─────────────────────────────────────────────
//
// cuBLASLt's MXFP8 kernels collapse at decode shapes -- 475 GB/s at
// M=1,N=4096,K=14336 on a part with ~1.7 TB/s, and 0.41x its own per-tensor
// fp8 time. These shapes are pure bandwidth: the whole job is to stream W
// once. MX adds 3.1% more bytes (one scale byte per 32 operand bytes) and a
// little arithmetic, so a good MXFP8 kernel should land within a few percent
// of a good per-tensor one, not at 2.4x its time.
//
// ── Where the scale enters ────────────────────────────────────────────
//
// The per-tensor kernel multiplies once in the epilogue. That is not
// available here: the scale changes every 32 elements of k, inside the
// accumulation. Two placements, and the cheap one is not the obvious one.
//
//   (a) Scale the operands as they are widened. Costs BM*32 multiplies for X
//       and NPT*32 for W per 32-element k-block, against BM*NPT*32 FMAs --
//       i.e. (BM + NPT) / (BM * NPT), which for the shapes that matter
//       (BM=1, NPT=1) is +200%.
//
//   (b) Accumulate one k-block into a partial, then fold the scale product in
//       with a single FMA per (n-column, m-row). Costs 2*BM*NPT against
//       BM*NPT*32, i.e. +6% regardless of tile shape.
//
// So (b): `pacc` below holds one k-block's worth of products, and the only
// price is BM*NPT extra live registers. VEC is 16 and an MX block is 32, so a
// block is exactly two of the existing vector loads.
//
// Measured, at BM=8 BN=256 CWG=1 (the worst case for (b), since BM*NPT is
// largest there) on M=8,N=4096,K=14336:
//
//   (b) partial accumulator      0.0595 ms   225 registers
//   (a) scale operands at load   0.0710 ms   231 registers
//
// (a) does not even buy back the registers it was supposed to: the widened
// operands stay live just as long, and the extra multiplies land in the
// middle of the FMA chain. At BM=1 the two are indistinguishable (0.0510 vs
// 0.0512 ms) because nothing there is ALU-bound.
//
// Forcing ptxas to use fewer registers does not help either: `#pragma unroll 1`
// on the k-block loop brings (b) down to 142 registers and 0.0612 ms. Both
// counts sit above the 128 that a second CTA per SM would need, so the
// occupancy is 1 either way and the only effect is less scheduling freedom.
//
// ── Where the scales come from ────────────────────────────────────────
//
// Straight from the same swizzled tensor cuBLASLt consumes -- no repacking,
// so the kernel is drop-in against a framework that already produces MXFP8
// for cuBLAS. That layout turns out to fit this kernel exactly: BK is 128,
// which is four 32-element k-blocks, and those four scales for one row are
// four contiguous bytes (mx_scale_pack). One 32-bit load per stage per row of
// X, and one per stage per n-column a thread owns, is the entire cost.
//
// The loads are issued before the mbarrier wait, so their latency overlaps
// the TMA they do not depend on.

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K = 1>
struct MXFP8GemmTinyM
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;
    static constexpr int CONSUMER_THREADS = CWG * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(fp8e4m3);

    static constexpr int VEC = 16;             // 16-byte smem vector loads
    static constexpr int NPT = BN / CONSUMER_THREADS;
    static constexpr int MX_VEC = 32;          // elements per ue8m0 scale
    static constexpr int KB_PER_STAGE = BK / MX_VEC;

    static constexpr bool DIRECT_OUTPUT = (SPLIT_K == 1);

    static_assert(BN % CONSUMER_THREADS == 0,
                  "BN must be a multiple of the consumer thread count");
    static_assert(NPT >= 1, "BN must be >= CONSUMER_THREADS (raise BN or lower CWG)");
    static_assert(BK % VEC == 0, "BK must be a multiple of 16");
    static_assert(BK == 128,
                  "the scale pack assumes a stage covers exactly four 32-element "
                  "k-blocks, i.e. BK = 128");
    static_assert(BM <= 128, "one scale-tensor row block covers 128 rows");
    static_assert(BM <= 256 && BN <= 256 && BK <= 256, "TMA box dims are capped at 256");

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    static constexpr int SWIZZLE_ROW_BLOCK = 8;
    static constexpr int X_STAGE_ROWS =
        ((BM + SWIZZLE_ROW_BLOCK - 1) / SWIZZLE_ROW_BLOCK) * SWIZZLE_ROW_BLOCK;
    static_assert(BN % SWIZZLE_ROW_BLOCK == 0, "BN must be a multiple of 8 rows");

    struct SMemStorage
    {
        // X must stay FIRST: smem_raw is __align__(1024), so being at offset 0
        // is what puts stage 0 at swizzle phase 0. The scale tensors are read
        // straight from global and never staged here.
        fp8e4m3 X[NUM_STAGES][X_STAGE_ROWS * BK];
        fp8e4m3 W[NUM_STAGES][BK * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    // x_sf / w_sf are swizzled ue8m0 tensors laid out by BlockScaleLayout(M, K)
    // and BlockScaleLayout(N, K); workspace must hold >= SPLIT_K * M * N floats.
    static void run(
        int M, int N, int K,
        const fp8e4m3 *__restrict__ X,
        const fp8e4m3 *__restrict__ W,
        bf16 *__restrict__ Y,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf,
        float *__restrict__ workspace = nullptr,
        cudaStream_t stream = nullptr);
};

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K>
__global__ void __launch_bounds__(MXFP8GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::TOTAL_THREADS, 1, 1)
    mxfp8_gemm_tinym_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int num_k_per_split,
        int sf_k_tiles, // BlockScaleLayout::k_tiles(), same for X and W
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf,
        bf16 *__restrict__ Y,
        float *__restrict__ workspace)
{
    using P = MXFP8GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(1024) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int num_blocks = gridDim.x;
    const int total_vtiles = total_tiles * SPLIT_K;

    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    fence_proxy_async_shared();

    // ── Producer warp group (unchanged: scales bypass the pipeline) ─
    if (wg_id == 0)
    {
        if (warp_in_wg == 0 && lane_id == 0)
        {
            int stage = 0;
            int phase = 0;
            int total_k = 0;

            for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
            {
                int tile_id = vtid / SPLIT_K;
                int split_idx = vtid % SPLIT_K;
                int bm, bn;
                rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n,
                                        P::SWIZZLE_WIDTH, bm, bn);

                const int k_start = split_idx * num_k_per_split;
                const int k_end = k_start + num_k_per_split;

                for (int k = k_start; k < k_end; k++)
                {
                    if (total_k >= NUM_STAGES)
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);

                    mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), P::TX_BYTES);
                    tma_X.load_2d(k * BK, bm * BM, smem_u32(smem.X[stage]),
                                  smem_u32(&smem.full_barrier[stage]));
                    tma_W.load_2d(k * BK, bn * BN, smem_u32(smem.W[stage]),
                                  smem_u32(&smem.full_barrier[stage]));

                    stage++;
                    if (stage == NUM_STAGES)
                    {
                        stage = 0;
                        phase ^= 1;
                    }
                    total_k++;
                }
            }
        }
    }
    // ── Consumer warp groups ───────────────────────────────────────
    else
    {
        const int tid_c = tid - P::THREADS_PER_WG;

        int stage = 0;
        int phase = 0;

        for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
        {
            int tile_id = vtid / SPLIT_K;
            int split_idx = vtid % SPLIT_K;
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n,
                                    P::SWIZZLE_WIDTH, bm, bn);

            const int k_start = split_idx * num_k_per_split;
            const int k_end = k_start + num_k_per_split;

            // Scale-tensor rows this thread needs, fixed for the whole tile.
            const int n_tile_base = bn * BN;
            int sf_w_row[P::NPT];
#pragma unroll
            for (int p = 0; p < P::NPT; p++)
                sf_w_row[p] = n_tile_base + p * P::CONSUMER_THREADS + tid_c;

            float acc[P::NPT][BM]{};

            for (int k = k_start; k < k_end; k++)
            {
                // Issued before the barrier wait: these depend on k alone, not
                // on the TMA, so the latency hides behind the stage that is
                // still in flight. Each is one aligned 32-bit load holding all
                // four of this stage's k-blocks.
                unsigned int xpack[BM], wpack[P::NPT];
#pragma unroll
                for (int m = 0; m < BM; m++)
                    xpack[m] = mx_scale_pack(x_sf, bm * BM + m, k, sf_k_tiles);
#pragma unroll
                for (int p = 0; p < P::NPT; p++)
                    wpack[p] = mx_scale_pack(w_sf, sf_w_row[p], k, sf_k_tiles);

                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const fp8e4m3 *sX = smem.X[stage];
                const fp8e4m3 *sW = smem.W[stage];

                // One MX k-block (32 elements) per iteration = two VEC loads.
#pragma unroll
                for (int kb = 0; kb < P::KB_PER_STAGE; kb++)
                {
                    float sx[BM], sw[P::NPT];
#pragma unroll
                    for (int m = 0; m < BM; m++)
                        sx[m] = ue8m0_to_float_fast((xpack[m] >> (8 * kb)) & 0xffu);
#pragma unroll
                    for (int p = 0; p < P::NPT; p++)
                        sw[p] = ue8m0_to_float_fast((wpack[p] >> (8 * kb)) & 0xffu);

                    // Unscaled products for this k-block only.
                    float pacc[P::NPT][BM]{};

#pragma unroll
                    for (int half = 0; half < P::MX_VEC / P::VEC; half++)
                    {
                        const int kk = kb * P::MX_VEC + half * P::VEC;

                        // Broadcast read: every thread wants the same X.
                        float xf[BM][P::VEC];
#pragma unroll
                        for (int m = 0; m < BM; m++)
                            load_fp8x16_f32(
                                &sX[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(m, kk, BK)], xf[m]);

#pragma unroll
                        for (int p = 0; p < P::NPT; p++)
                        {
                            const int n = p * P::CONSUMER_THREADS + tid_c;
                            float wf[P::VEC];
                            load_fp8x16_f32(
                                &sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(n, kk, BK)], wf);

#pragma unroll
                            for (int e = 0; e < P::VEC; e++)
#pragma unroll
                                for (int m = 0; m < BM; m++)
                                    pacc[p][m] = fmaf(xf[m][e], wf[e], pacc[p][m]);
                        }
                    }

                    // The whole cost of MX scaling: one multiply and one FMA
                    // per output element per 32 elements of k.
#pragma unroll
                    for (int p = 0; p < P::NPT; p++)
#pragma unroll
                        for (int m = 0; m < BM; m++)
                            acc[p][m] = fmaf(pacc[p][m], sx[m] * sw[p], acc[p][m]);
                }

                __syncwarp();
                if (lane_id == 0)
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                __syncwarp();

                stage++;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase ^= 1;
                }
            }

            // ── Epilogue ───────────────────────────────────────────
            // No scale left to apply: the MMA-less path has already folded
            // every ue8m0 in. Rows m >= M are dropped here, which is also what
            // makes their scale bytes irrelevant — TMA zero-fills their X data,
            // so whatever the padding rows of the scale tensor hold multiplies
            // an accumulator that is never stored.
#pragma unroll
            for (int p = 0; p < P::NPT; p++)
            {
                const int n = n_tile_base + p * P::CONSUMER_THREADS + tid_c;
#pragma unroll
                for (int m = 0; m < BM; m++)
                {
                    if (m >= M) break;
                    if constexpr (P::DIRECT_OUTPUT)
                        Y[(size_t)m * N + n] = __float2bfloat16(acc[p][m]);
                    else
                        workspace[(size_t)split_idx * M * N + (size_t)m * N + n] = acc[p][m];
                }
            }
        }
    }

    __syncthreads();
    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_inval(&smem.full_barrier[s]);
            mbarrier_inval(&smem.empty_barrier[s]);
        }
    }
}

// ── Launch ─────────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K>
void MXFP8GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ X,
    const fp8e4m3 *__restrict__ W,
    bf16 *__restrict__ Y,
    const unsigned char *__restrict__ x_sf,
    const unsigned char *__restrict__ w_sf,
    float *__restrict__ workspace,
    cudaStream_t stream)
{
    if (M > BM || N % BN != 0 || K % (BK * SPLIT_K) != 0)
        throw std::runtime_error(
            "MXFP8 TinyM: need M <= BM, N divisible by BN, K divisible by BK*SPLIT_K.");

    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d_raw(W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    int num_tiles_m = 1;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;
    int num_k_per_split = (K / BK) / SPLIT_K;
    // BK = 128 = 4 * 32, so a stage index is exactly a scale-tile index.
    int sf_k_tiles = BlockScaleLayout(M, K).k_tiles();

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles * SPLIT_K);

    auto kernel = mxfp8_gemm_tinym_kernel<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<dim3(num_blocks), dim3(TOTAL_THREADS), SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, num_k_per_split, sf_k_tiles,
        tma_X, tma_W, x_sf, w_sf, Y, workspace);
    CHECK_CUDA(cudaGetLastError());

    if constexpr (SPLIT_K > 1)
    {
        constexpr int REDUCE_THREADS = 512;
        auto MN = (size_t)M * (size_t)N;
        auto reduce_blocks = (MN + REDUCE_THREADS - 1) / REDUCE_THREADS;
        splitk_reduce_kernel<SPLIT_K><<<reduce_blocks, REDUCE_THREADS, 0, stream>>>(
            workspace, Y, MN);
        CHECK_CUDA(cudaGetLastError());
    }
}
