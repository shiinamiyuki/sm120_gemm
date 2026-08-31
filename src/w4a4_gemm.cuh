#pragma once
#include "fp8_gemm.cuh"  // TMA / mbarrier / ldmatrix / stmatrix / swizzle helpers
#include "nvfp4_mma.cuh" // the block-scaled m16n8k64 e2m1 MMA and its layouts
#include "block_scale.h"

// ════════════════════════════════════════════════════════════════════════
// W4A4GemmMMA — NVFP4 x NVFP4 on tensor cores
// ════════════════════════════════════════════════════════════════════════
//
// This is the easy one. W4A16 had to widen the weights to bf16 in registers
// because no mixed 4-bit x 16-bit MMA exists; here both operands are e2m1 and
// sm_120a feeds them to the tensor core directly, with the per-16 ue4m3 scales
// applied by the instruction itself. There is no dequantize step anywhere in
// the mainloop -- the consumer is ldmatrix, ldmatrix, mma, and nothing else.
//
// The structure is MXFP8GemmMMA's, with three things changed by the format.
//
// ── BK is pinned to 256 ───────────────────────────────────────────────
//
// Two independent constraints, and 256 is the smallest value meeting both.
// A packed-e2m1 row of BK elements is BK/2 bytes, and the 128B swizzle that
// makes the ldmatrix reads conflict-free needs 128 of them, so BK >= 256. And
// mma.m16n8k64 covers 64 k, so BK must be a multiple of 64. (W4A16 pins BK to
// 256 for the first reason alone.)
//
// ── One scale pack per MMA, not four ──────────────────────────────────
//
// MXFP8's scale_vec::1X gives one ue8m0 per row per MMA, covering all 32 k, so
// a b32 of four scale bytes fed four consecutive MMAs and the byte-id
// immediate stepped 0,1,2,3 through it. NVFP4's scale_vec::4X gives *four*
// ue4m3 per row per MMA, one per 16 k, which is exactly a b32 -- so each MMA
// consumes a whole pack and byte-id is pinned to 0. SF_PACKS therefore equals
// MMA_K here, and pack ki belongs to MMA ki. Both work out to BK/64.
//
// ── The epilogue still has a multiply ─────────────────────────────────
//
// MXFP8 has none: every ue8m0 is a power of two and the MMA takes all of them.
// NVFP4 is two-level, so after the MMA has applied both block scales what is
// left is the product of the two per-tensor fp32 globals -- one multiply per
// output element, folded into the bf16 conversion.
//
// ── The tail, and why a bigger GEMM can be slower ─────────────────────
//
// Work is handed out one output tile per CTA, with num_blocks capped at the SM
// count, so the kernel finishes when the CTA holding the most tiles finishes.
// With T tiles on C = 170 SMs that is ceil(T/C) rounds regardless of how
// unevenly the last round is filled, and the cost is a sawtooth in the shape.
// Holding 128x128_s2_cwg1_w64x64_ys4 fixed at K=4096 and sweeping M=N:
//
//   M=N    tiles  rounds   TFLOPS
//   1024      64       1      481   <- 64 tiles cannot even fill 170 SMs
//   1280     100       1      735
//   1536     144       1      984
//   1792     196       2      726   <- bigger problem, 26% slower
//   2048     256       2      906
//   2560     400       3      971
//   3072     576       4     1113
//   4096    1024       7     1192
//
// 1792 against 1536 is the whole story in one line: 196 tiles need two rounds
// where 144 needed one, so half the machine idles through the second. This is
// also the entire gap against FlashInfer at 2048 (we are 25% behind there but
// only 8% behind at 4096) -- not a slower mainloop, a worse schedule.
//
// Tile tuning cannot fix it. Every tile shape that keeps the mainloop fast
// lands on the same tile count at 2048 (128x128 -> 256, 128x64 -> 512 which is
// 3.01 rounds and quantizes to 4, 256x128 -> 128 which is one round using 128
// of 170 SMs), and the shapes that quantize well (64x64) give up too much
// mainloop throughput to be worth it. The fix is a stream-k schedule: split the
// total k-work evenly across exactly C CTAs and reduce the partial tiles,
// so occupancy stops depending on how T divides C.
//
// ── What is unchanged, and why it matters ─────────────────────────────
//
// The scale tiles still come in through the producer as 1D bulk copies rather
// than being read from global by the consumers. The reason is the same sector
// amplification MXFP8 hit: 16 useful lanes reading 4 bytes at a 16-byte stride
// turn a 1 KB stage of scales into ~16 KB of L1 sector traffic. A 512-byte
// scale tile is a contiguous byte range, so cp.async.bulk moves it with no
// tensor descriptor and it completes on the same mbarrier as the two operand
// loads -- one more expect_tx and nothing else.
template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M = 64, int WARP_N = 64, int Y_SLICES = 1>
struct W4A4GemmMMA
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    // Packed e2m1: half a byte per element, on both sides.
    static constexpr int X_ROW_BYTES = BK / 2;
    static constexpr int W_ROW_BYTES = BK / 2;
    static constexpr int TX_BYTES_OPERANDS = BM * X_ROW_BYTES + BN * W_ROW_BYTES;

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 64;   // mma.m16n8k64
    static constexpr int SF_VEC = 16;       // ue4m3 per 16 elements of k
    // One b32 of four ue4m3 per MMA, so packs and MMAs are in step.
    static constexpr int SF_PACKS = BK / 64;
    static_assert(SF_PACKS == MMA_K, "scale_vec::4X gives exactly one pack per MMA");

    static constexpr int SF_X_TILES = (BM + 127) / 128;
    static constexpr int SF_W_TILES = (BN + 127) / 128;
    static constexpr int SF_X_BYTES = SF_PACKS * SF_X_TILES * 512;
    static constexpr int SF_W_BYTES = SF_PACKS * SF_W_TILES * 512;

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM x BN exactly");
    static_assert(BK == 256,
                  "packed e2m1 needs BK/2 == 128 bytes per row for the 128B swizzle, "
                  "and BK must be a multiple of the MMA's k=64");
    static_assert(WARP_M % 16 == 0 && WARP_N % 8 == 0, "warp tile must fit m16n8");

    static constexpr int TX_BYTES = TX_BYTES_OPERANDS + SF_X_BYTES + SF_W_BYTES;

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    // The output staging buffer is the largest single thing in shared memory
    // once the operands are 4-bit: at BM=BN=128 the packed X and W stages come
    // to 64 KB together while a full BM x BN bf16 Y_out is 32 KB on its own,
    // which is what puts 128x128 over the 99 KB limit. It cannot simply alias
    // the stage buffers, because the producer is already filling the next
    // tile's stages while this tile is in its epilogue.
    //
    // So the epilogue stores in Y_SLICES column slices instead, staging only
    // BM x (BN/Y_SLICES) at a time and issuing one TMA store per slice. Each
    // slice waits for the previous slice's store to have *read* smem
    // (cp.async.bulk.wait_group.read), which is cheap next to a compute-bound
    // mainloop and is what buys the tile size back.
    //
    // Worth it. TFLOPS, cwg2 w64x32 throughout:
    //
    //                      1024^2 x2048   2048^2 x4096   4096^3
    //   128x64  ys1              399            633        725
    //   128x128 ys1        does not fit -- 104 KB against a 99 KB limit
    //   128x128 ys2              336            803        899
    //   128x128 ys4              344            798        906
    //
    // 1.25x at the compute-bound sizes, purely from being allowed to use the
    // bigger tile. ys2 and ys4 are the same to within noise, so the extra
    // stores cost nothing measurable -- take the fewest slices that fit.
    //
    // The 1024 column is the caveat: there the 128x128 config is *slower* than
    // 128x64, because 8x8 tiles cannot cover 170 SMs. Tile size is not a free
    // win, it is a trade against how many CTAs the shape can produce, which is
    // what the autotuner is for.
    static constexpr int N_SLICE = BN / Y_SLICES;
    static_assert(BN % Y_SLICES == 0, "Y_SLICES must divide BN");
    static_assert(N_SLICE % 8 == 0, "an output slice must be a whole number of n8 tiles");
    static_assert(N_SLICE <= 256, "TMA box dims are capped at 256");

    // X first so stage 0 sits at swizzle phase 0; a stage is BM*128 bytes, a
    // multiple of 1024, so every stage keeps that phase.
    struct SMemStorage
    {
        unsigned char X[NUM_STAGES][BM * X_ROW_BYTES];
        unsigned char W[NUM_STAGES][BN * W_ROW_BYTES];
        unsigned char SF_X[NUM_STAGES][SF_X_BYTES];
        unsigned char SF_W[NUM_STAGES][SF_W_BYTES];
        bf16 Y_out[BM * N_SLICE];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const unsigned char *__restrict__ X,
        const unsigned char *__restrict__ W,
        bf16 *__restrict__ Y,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf,
        float out_scale,
        cudaStream_t stream);
};

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, int Y_SLICES>
__global__ void __launch_bounds__(W4A4GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, Y_SLICES>::TOTAL_THREADS, 1, 1)
    w4a4_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int num_k_tiles, int sf_x_k_tiles, int sf_w_k_tiles,
        float out_scale,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        __grid_constant__ const TMADescriptor tma_Y,
        const unsigned char *__restrict__ x_sf,
        const unsigned char *__restrict__ w_sf)
{
    using P = W4A4GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, Y_SLICES>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(1024) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;
    const int num_blocks = gridDim.x;

    if (tid == 0)
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    __syncthreads();
    fence_proxy_async_shared();

    // ── Producer ───────────────────────────────────────────────────
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
                    // Both operands are packed, so the descriptors' inner
                    // extent is K/2 bytes and the k offset halves.
                    tma_X.load_2d(k * P::X_ROW_BYTES, bm * BM, smem_u32(smem.X[stage]), mbar);
                    tma_W.load_2d(k * P::W_ROW_BYTES, bn * BN, smem_u32(smem.W[stage]), mbar);

#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                    {
#pragma unroll
                        for (int t = 0; t < P::SF_X_TILES; t++)
                            cp_async_bulk_g2s(
                                smem_u32(&smem.SF_X[stage][(j * P::SF_X_TILES + t) * 512]),
                                x_sf + mx_scale_tile_bytes(sf_x_rb + t, k * P::SF_PACKS + j,
                                                           sf_x_k_tiles),
                                512, mbar);
#pragma unroll
                        for (int t = 0; t < P::SF_W_TILES; t++)
                            cp_async_bulk_g2s(
                                smem_u32(&smem.SF_W[stage][(j * P::SF_W_TILES + t) * 512]),
                                w_sf + mx_scale_tile_bytes(sf_w_rb + t, k * P::SF_PACKS + j,
                                                           sf_w_k_tiles),
                                512, mbar);
                    }

                    stage++;
                    if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                    total_k++;
                }
            }
        }
    }
    // ── Consumers ──────────────────────────────────────────────────
    else
    {
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;
        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        // Which row of each 16-row A tile / 8-column B tile this lane supplies
        // the scale for, at thread-id 0. Inverses of the mapping probed in
        // nvfp4_mma.cuh -- and note it is *not* the accumulator's row mapping,
        // which is why the scale is fetched by its own address arithmetic
        // rather than falling out of the A fragment load.
        const int a_sf_row = (lane_id / 4) + 8 * (lane_id % 2); // 0..15
        const int b_sf_col = lane_id / 4;                       // 0..7

        int stage = 0, phase = 0;
        bool has_tma_store_in_flight = false;

        for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
        {
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n, P::SWIZZLE_WIDTH, bm, bn);
            const int sf_x_row0 = (bm * BM) % 128;
            const int sf_w_row0 = (bn * BN) % 128;

            float acc[P::MMA_M][P::MMA_N][4]{};

            for (int k = 0; k < num_k_tiles; k++)
            {
                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

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

                const unsigned char *sX = smem.X[stage];
                const unsigned char *sW = smem.W[stage];
#pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    // One MMA spans 64 elements of k, i.e. 32 packed bytes.
                    const int k_byte = ki * 32;

                    uint32_t b_frag[P::MMA_N][2];
#pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        // B is 64(k) x 8(n) = two 8x8 b16 matrices, i.e. 8 rows
                        // of 16 bytes. Lanes 0-7 address matrix 0 (k 0..31),
                        // lanes 8-15 matrix 1 (k 32..63); +16 bytes is +32 k.
                        const int b_n = n_warp_base + ni * 8 + (lane_id & 7);
                        const int b_kb = k_byte + ((lane_id >> 3) & 1) * 16;
                        ldmatrix_x2(b_frag[ni],
                                    smem_u32(&sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(
                                        b_n, b_kb, P::W_ROW_BYTES)]));
                    }
#pragma unroll
                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        // A is 16(m) x 64(k) = four 8x8 b16 matrices: rows
                        // 0-7 / 8-15 crossed with k 0..31 / 32..63.
                        uint32_t a[4];
                        {
                            const int a_row = m_warp_base + mi * 16 + (lane_id & 7) +
                                              ((lane_id >> 3) & 1) * 8;
                            const int a_kb = k_byte + (lane_id >> 4) * 16;
                            ldmatrix_x4(a, smem_u32(&sX[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(
                                             a_row, a_kb, P::X_ROW_BYTES)]));
                        }
#pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                            mma_m16n8k64_e2m1_block_scaled<0>(acc[mi][ni], a, b_frag[ni],
                                                              sfa[mi][ki], sfb[ni][ki]);
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
            // The MMA has taken both per-16 ue4m3 scales; what is left is the
            // product of the two per-tensor fp32 globals.
            bf16 *sY = smem.Y_out;
            const bool store_leader = (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0);
#pragma unroll
            for (int sl = 0; sl < Y_SLICES; sl++)
            {
                // Wait for whatever store still owns sY -- the previous slice,
                // or the last slice of the previous tile.
                if (sl > 0 || has_tma_store_in_flight)
                {
                    if (store_leader) tma_store_wait();
                    named_barrier_sync(P::TOTAL_WGS, CWG * P::THREADS_PER_WG);
                }

                const int n_lo = sl * P::N_SLICE;
#pragma unroll
                for (int mi = 0; mi < P::MMA_M; mi++)
#pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;
                        if (n_base < n_lo || n_base >= n_lo + P::N_SLICE) continue;
                        const int m_base = m_warp_base + mi * 16;
                        const uint32_t c0 = f32x2_to_bf16x2(acc[mi][ni][0] * out_scale,
                                                            acc[mi][ni][1] * out_scale);
                        const uint32_t c1 = f32x2_to_bf16x2(acc[mi][ni][2] * out_scale,
                                                            acc[mi][ni][3] * out_scale);
                        const int st_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                        stmatrix_x2(smem_u32(&sY[st_row * P::N_SLICE + (n_base - n_lo)]), c0, c1);
                    }

                named_barrier_sync(P::TOTAL_WGS, CWG * P::THREADS_PER_WG);

                if (store_leader)
                {
                    fence_proxy_async_shared();
                    tma_store_2d(reinterpret_cast<const uint64_t *>(tma_Y.raw),
                                 bn * BN + n_lo, bm * BM, smem_u32(sY));
                    tma_store_arrive();
                }
                has_tma_store_in_flight = true;
            }
        }

        if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0 && has_tma_store_in_flight)
            tma_store_wait();
    }

    __syncthreads();
    if (tid == 0)
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_inval(&smem.full_barrier[s]);
            mbarrier_inval(&smem.empty_barrier[s]);
        }
}

// ── Launch ─────────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, int Y_SLICES>
void W4A4GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, Y_SLICES>::run(
    int M, int N, int K,
    const unsigned char *__restrict__ X,
    const unsigned char *__restrict__ W,
    bf16 *__restrict__ Y,
    const unsigned char *__restrict__ x_sf,
    const unsigned char *__restrict__ w_sf,
    float out_scale,
    cudaStream_t stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        throw std::runtime_error("W4A4: M, N, K must be divisible by BM, BN, BK.");

    // Packed e2m1 seen as bytes, so the global inner extent is K/2.
    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K / 2, M, X_ROW_BYTES, BM,
                                                 CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d_raw(W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K / 2, N, W_ROW_BYTES, BN,
                                                 CU_TENSOR_MAP_SWIZZLE_128B);
    // One TMA store per output slice, so the box is BM x N_SLICE.
    TMADescriptor tma_Y = create_tma_desc_2d(Y, N, M, N_SLICE, BM);

    const int num_tiles_m = M / BM;
    const int num_tiles_n = N / BN;
    const int total_tiles = num_tiles_m * num_tiles_n;
    const int num_k_tiles = K / BK;
    // Both scale tensors are blocked by 16 along K, but over different row
    // counts, so they can have different tile strides.
    const int sf_x_k_tiles = BlockScaleLayout(M, K, SF_VEC).k_tiles();
    const int sf_w_k_tiles = BlockScaleLayout(N, K, SF_VEC).k_tiles();

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    const int num_blocks = min(num_sm, total_tiles);

    auto kernel = w4a4_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, Y_SLICES>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<dim3(num_blocks), dim3(TOTAL_THREADS), SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, num_k_tiles,
        sf_x_k_tiles, sf_w_k_tiles, out_scale, tma_X, tma_W, tma_Y, x_sf, w_sf);
    CHECK_CUDA(cudaGetLastError());
}
