#pragma once
#include "fp8_gemm.cuh" // TMA / mbarrier / ldmatrix / stmatrix / swizzle helpers
#include "mxfp8_mma.cuh"
#include "block_scale.h"

// ════════════════════════════════════════════════════════════════════════
// MXFP8GemmMMA — MXFP8 GEMM on tensor cores
// ════════════════════════════════════════════════════════════════════════
//
// FP8GemmMMA with real block scales. The pipeline, the warp tiling, the
// fragment loads and the epilogue are untouched; the MMA is the same
// instruction it was already issuing. What changes is that its two ue8m0
// operands stop being four copies of one per-tensor exponent and start
// carrying a different scale per 32 elements of k.
//
// This family is *cheaper* than the per-tensor one, not dearer: the
// per-tensor kernel had to leave a mantissa residual for the epilogue
// (an arbitrary scale is not a power of two), and MX scales are powers of two
// by construction, so the epilogue loses a multiply and gains nothing.
//
// ── Getting the scales into the right lanes ───────────────────────────
//
// The block-scaled MMA does not read the scale from whichever lane happens to
// hold the matching accumulator row. It reads row m of A from lane
// 4*(m%8) + (m/8) and column n of B from lane 4*n, at a byte selected by the
// instruction's byte-id immediate (see mxfp8_mma.cuh). So each lane has to end
// up holding the scales for rows/columns that are *not* the ones it computes.
//
// The obvious worry is that this needs a shuffle. It does not, and TMA is the
// wrong tool for it either way:
//
//   * TMA can reorder in exactly two ways -- permuting tensor dimensions via
//     the descriptor (which still requires a globally contiguous axis to be
//     innermost), and the fixed XOR bank swizzle (a function of the smem row
//     index, designed for ldmatrix). The mapping wanted here, lane =
//     4*(m%8) + (m/8), is a bit interleave of m's low and high halves: it puts
//     16 values into 32 lanes with gaps, and TMA only writes dense boxes.
//     Even the best dimension permutation lands the packs 16 bytes apart,
//     which is where they already are.
//
//   * It does not matter, because the mapping is free. It is a per-lane
//     *address*, not a data movement: lane l simply loads the pack for row
//     (l/4) + 8*(l%2) instead of the pack for its own row.
//
// And the scale-factor layout hands us the fragment ready-made. A row's
// scales for four consecutive k-blocks are four contiguous bytes, and one
// stage is BK = 128 = four k-blocks, so MMA_K = 4 -- exactly the range of the
// byte-id immediate. One aligned 32-bit load per lane per m-tile per stage
// covers every MMA that stage will issue, with byte-id stepping 0,1,2,3.
//
// ── Why the packs go through the pipeline ─────────────────────────────
//
// The tiny-M kernel reads its packs straight from global and pays nothing for
// it. Here that costs 33%: 0.3604 ms against 0.2715 ms with the loads removed
// (M=N=K=4096, 128x128x128_s2_cwg2_w64x32). The block-scaled MMA itself is
// free -- the isolated variant beats the per-tensor kernel's 0.2738 ms,
// because MX scales are powers of two and the epilogue loses its residual
// multiply.
//
// The problem is sector amplification. A warp's 16 useful lanes read 4 bytes
// each at a 16-byte stride, so one load instruction touches eight 32-byte
// sectors to collect 64 useful bytes. At 8 loads per warp per stage and 8
// warps per CTA, a 1 KB stage of scales turns into ~16 KB of L1 sector
// traffic. Tiny-M escapes this because it is bandwidth-bound with idle LSU
// slots; the MMA kernel is not.
//
// So the producer brings the tiles in instead. A 512-byte tile is a plain
// contiguous byte range, which is what the 1D `cp.async.bulk` form copies --
// no tensor descriptor, and it completes on the same mbarrier as the two TMA
// tensor loads, so it costs one more expect_tx and nothing else. Consumers
// then read 4 bytes from shared memory, where the 16-byte stride is at worst
// a two-way bank conflict on eight loads per stage.
//
// Note what TMA is and is not doing here: it is pure transport. The lane
// mapping is still done by address arithmetic at the smem load, because that
// is free and TMA could not have done it anyway.

template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M = 64, int WARP_N = 64>
struct MXFP8GemmMMA
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES_OPERANDS = (BM * BK + BK * BN) * sizeof(fp8e4m3);

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 32;
    // 32-bit scale packs per row per stage: one per four k-blocks.
    static constexpr int SF_PACKS = BK / 128;
    // 512-byte scale tiles a CTA tile spans. BM and BN are 64 or 128 here, so
    // both are 1; the general form keeps larger tiles honest.
    static constexpr int SF_X_TILES = (BM + 127) / 128;
    static constexpr int SF_W_TILES = (BN + 127) / 128;
    static constexpr int SF_X_BYTES = SF_PACKS * SF_X_TILES * 512;
    static constexpr int SF_W_BYTES = SF_PACKS * SF_W_TILES * 512;

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM×BN exactly");
    static_assert(BK % 128 == 0,
                  "a scale pack covers four 32-element k-blocks, so BK must be a "
                  "multiple of 128");

    // Operands plus the stage's scale tiles: all of it lands on one mbarrier.
    static constexpr int TX_BYTES = TX_BYTES_OPERANDS + SF_X_BYTES + SF_W_BYTES;

    static constexpr int ACC_REGS = MMA_M * MMA_N * 4;
    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    // X stays first so stage 0 sits at swizzle phase 0. The scale tiles are
    // 512-byte multiples, so they need no padding of their own; they are the
    // only addition to the per-tensor kernel's storage, and
    // GemmConfig::smem_bytes() accounts for them.
    struct SMemStorage
    {
        fp8e4m3 X[NUM_STAGES][BM * BK];
        fp8e4m3 W[NUM_STAGES][BK * BN];
        unsigned char SF_X[NUM_STAGES][SF_X_BYTES];
        unsigned char SF_W[NUM_STAGES][SF_W_BYTES];
        bf16 Y_out[BM * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const fp8e4m3 *__restrict__ X,
        const fp8e4m3 *__restrict__ W,
        bf16 *__restrict__ Y,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf,
        cudaStream_t stream = nullptr);
};

// ── MXFP8 MMA kernel ───────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
__global__ void __launch_bounds__(MXFP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::TOTAL_THREADS, 1, 1)
    mxfp8_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int sf_k_tiles, // BlockScaleLayout::k_tiles(), same for X and W
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        __grid_constant__ const TMADescriptor tma_Y,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf)
{
    using P = MXFP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(128) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int num_k_tiles = K / BK;
    const int num_blocks = gridDim.x;

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
            int stage = 0, phase = 0, total_k = 0;
            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
            {
                int bm, bn;
                rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n, P::SWIZZLE_WIDTH, bm, bn);
                const int sf_x_rb = (bm * BM) / 128;
                const int sf_w_rb = (bn * BN) / 128;
                for (int k = 0; k < num_k_tiles; k++)
                {
                    if (total_k >= NUM_STAGES)
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);

                    const uint32_t mbar = smem_u32(&smem.full_barrier[stage]);
                    mbarrier_expect_tx(mbar, P::TX_BYTES);
                    tma_X.load_2d(k * BK, bm * BM, smem_u32(smem.X[stage]), mbar);
                    tma_W.load_2d(k * BK, bn * BN, smem_u32(smem.W[stage]), mbar);

                    // Scale tiles: contiguous 512-byte ranges, so the 1D bulk
                    // form does it without a tensor descriptor.
#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                    {
#pragma unroll
                        for (int t = 0; t < P::SF_X_TILES; t++)
                            cp_async_bulk_g2s(
                                smem_u32(&smem.SF_X[stage][(j * P::SF_X_TILES + t) * 512]),
                                x_sf + mx_scale_tile_bytes(sf_x_rb + t, k * P::SF_PACKS + j,
                                                           sf_k_tiles),
                                512, mbar);
#pragma unroll
                        for (int t = 0; t < P::SF_W_TILES; t++)
                            cp_async_bulk_g2s(
                                smem_u32(&smem.SF_W[stage][(j * P::SF_W_TILES + t) * 512]),
                                w_sf + mx_scale_tile_bytes(sf_w_rb + t, k * P::SF_PACKS + j,
                                                           sf_k_tiles),
                                512, mbar);
                    }

                    stage++;
                    if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                    total_k++;
                }
            }
        }
    }
    // ── Consumer warp groups ───────────────────────────────────────
    else
    {
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;

        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        // Which row of each 16-row A tile / 8-column B tile this lane must
        // supply the scale for. Inverses of the mappings probed in
        // mxfp8_mma.cuh, at thread-id 0. Both stay inside their tile for every
        // lane, so lanes the MMA ignores still address in bounds.
        const int a_sf_row = (lane_id / 4) + 8 * (lane_id % 2); // 0..15
        const int b_sf_col = lane_id / 4;                       // 0..7

        int stage = 0, phase = 0;
        bool has_tma_store_in_flight = false;

        for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
        {
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n, P::SWIZZLE_WIDTH, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};

            // Row of the 128-row scale block this CTA tile starts at.
            const int sf_x_row0 = (bm * BM) % 128;
            const int sf_w_row0 = (bn * BN) % 128;

            for (int k = 0; k < num_k_tiles; k++)
            {
                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                // One 4-byte shared load per m-tile / n-tile per stage; it
                // feeds all MMA_K MMAs, stepping the instruction's byte-id.
                uint32_t sfa[P::MMA_M][P::SF_PACKS], sfb[P::MMA_N][P::SF_PACKS];
#pragma unroll
                for (int mi = 0; mi < P::MMA_M; mi++)
                {
                    const int r = sf_x_row0 + m_warp_base + mi * 16 + a_sf_row;
#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                        sfa[mi][j] = *reinterpret_cast<const uint32_t *>(
                            &smem.SF_X[stage][(j * P::SF_X_TILES + (r >> 7)) * 512 +
                                              mx_scale_tile_offset(r)]);
                }
#pragma unroll
                for (int ni = 0; ni < P::MMA_N; ni++)
                {
                    const int r = sf_w_row0 + n_warp_base + ni * 8 + b_sf_col;
#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                        sfb[ni][j] = *reinterpret_cast<const uint32_t *>(
                            &smem.SF_W[stage][(j * P::SF_W_TILES + (r >> 7)) * 512 +
                                              mx_scale_tile_offset(r)]);
                }

                const fp8e4m3 *sX = smem.X[stage];
                const fp8e4m3 *sW = smem.W[stage];
#pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 32;

                    uint32_t b_frag[P::MMA_N][2];
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;
                        int b_n = n_base + (lane_id & 7);
                        int b_k = k_base + (((lane_id >> 3) & 1) * 16);
                        ldmatrix_x2(b_frag[ni],
                                    smem_u32(&sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(b_n, b_k, BK)]));
                    }
#pragma unroll
                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        const int m_base = m_warp_base + mi * 16;

                        uint32_t a[4];
                        {
                            int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                            int a_col = k_base + (lane_id >> 4) * 16;
                            ldmatrix_x4(a,
                                        smem_u32(&sX[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(a_row, a_col, BK)]));
                        }
#pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                            mma_m16n8k32_e4m3_block_scaled(
                                ki, acc[mi][ni], a, b_frag[ni],
                                sfa[mi][ki / 4], sfb[ni][ki / 4]);
                    }
                }

                __syncwarp();
                if (lane_id == 0)
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                __syncwarp();

                stage++;
                if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
            }

            // ── Epilogue ───────────────────────────────────────────
            // Nothing to apply: every ue8m0 is a power of two and the MMA has
            // already taken all of them. The per-tensor kernel's residual
            // multiply is gone.
            if (has_tma_store_in_flight)
            {
                if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                    tma_store_wait();
                named_barrier_sync(P::TOTAL_WGS, CWG * P::THREADS_PER_WG);
            }

            bf16 *sY = smem.Y_out;
#pragma unroll
            for (int mi = 0; mi < P::MMA_M; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < P::MMA_N; ni++)
                {
                    int m_base = m_warp_base + mi * 16;
                    int n_base = n_warp_base + ni * 8;
                    uint32_t c0 = f32x2_to_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
                    uint32_t c1 = f32x2_to_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
                    int st_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                    stmatrix_x2(smem_u32(&sY[st_row * BN + n_base]), c0, c1);
                }
            }

            named_barrier_sync(P::TOTAL_WGS, CWG * P::THREADS_PER_WG);

            if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
            {
                fence_proxy_async_shared();
                tma_store_2d(reinterpret_cast<const uint64_t *>(tma_Y.raw),
                             bn * BN, bm * BM, smem_u32(sY));
                tma_store_arrive();
            }
            has_tma_store_in_flight = true;
        }

        if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0 && has_tma_store_in_flight)
            tma_store_wait();
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
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
void MXFP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ X,
    const fp8e4m3 *__restrict__ W,
    bf16 *__restrict__ Y,
    const unsigned char *__restrict__ x_sf,
    const unsigned char *__restrict__ w_sf,
    cudaStream_t stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");

    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d_raw(W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_Y = create_tma_desc_2d(Y, N, M, BN, BM);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;
    int sf_k_tiles = BlockScaleLayout(M, K).k_tiles();

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles);

    auto kernel = mxfp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<dim3(num_blocks), dim3(TOTAL_THREADS), SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, sf_k_tiles,
        tma_X, tma_W, tma_Y, x_sf, w_sf);
    CHECK_CUDA(cudaGetLastError());
}
