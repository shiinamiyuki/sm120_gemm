#pragma once
#include "bf16_gemm.cuh" // TMA / mbarrier / swizzle helpers, splitk_reduce_kernel

// ════════════════════════════════════════════════════════════════════════
// BF16GemmTinyM — skinny GEMM (M <= BM, typically M <= 4) on CUDA cores
// ════════════════════════════════════════════════════════════════════════
//
// Same skeleton as BF16GemmMMA: persistent CTAs, a producer warp group
// issuing TMA loads into a multi-stage smem pipeline, consumer warp groups
// draining it through mbarriers. Only the consumer math differs.
//
// Why not tensor cores: mma.m16n8k16 fixes M at 16, so an M<=4 problem wastes
// >=75% of every MMA. At that shape the kernel is nowhere near compute bound
// anyway — arithmetic intensity is ~M flop/byte (each W element is used M
// times), so at M=4 saturating ~1.5 TB/s needs only ~6 TFLOPS, well inside
// what the FP32 CUDA cores deliver. The job is purely to stream W.
//
// ── Work decomposition (no MMA — plain fp32 FMA on CUDA cores) ────────
//
//   CTA     BM x BN outputs, over a 1/SPLIT_K slice of k.
//   warp    BM x (32 * NPT), as NPT runs of 32 consecutive n. Run p of
//           warp w covers n = p*CONSUMER_THREADS + w*32 + (0..31), so the
//           runs are strided by CONSUMER_THREADS rather than contiguous.
//   thread  BM x NPT — the *entire* M extent, NPT columns of n.
//           acc[NPT][BM] floats stay in registers for the whole k-range,
//           so there is never a cross-thread reduction.
//   k step  each thread issues BM + NPT vector loads (16 B = 8 bf16 each)
//           per 8-wide k chunk, and BM*NPT*8 scalar FMAs.
//
// So the "tile shape per thread" is BM x NPT x 8 FMAs per chunk, versus the
// tensor-core kernel where one lane owns 4 scattered accumulators of a 16x8
// tile. Two consequences of dropping mma.m16n8k16:
//   - M is not tiled at all. The MMA shape forces a 16-row granularity in M,
//     which an M<=4 problem cannot fill; here BM >= M and lanes spread purely
//     along N.
//   - No fragment shuffling. Each lane reads the operands it needs directly,
//     so there is no ldmatrix/stmatrix and no per-lane layout to match.
//
// Consecutive threads take consecutive n, which makes both the smem reads and
// the epilogue stores conflict-free/coalesced (see SMEM ACCESS on the k-loop).
//
// SPLIT_K is not optional here in practice: with no M-parallelism, an
// N=4096 problem yields only N/BN tiles, far fewer than the SM count, so K
// must be partitioned to fill the machine.

// ── bf16 -> fp32 widening for CUDA-core math ───────────────────────────
__device__ __forceinline__ float2 bf16x2_to_f32x2(uint32_t packed)
{
    bf16_2 v;
    memcpy(&v, &packed, 4);
    return __bfloat1622float2(v);
}

// One 16-byte shared-memory vector = 8 contiguous bf16, widened to fp32.
__device__ __forceinline__ void load_bf16x8_f32(const bf16 *src, float (&out)[8])
{
    uint4 raw = *reinterpret_cast<const uint4 *>(src);
    const uint32_t packed[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 f = bf16x2_to_f32x2(packed[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K = 1>
struct BF16GemmTinyM
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;
    static constexpr int CONSUMER_THREADS = CWG * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(bf16);

    // 16-byte vector loads: 8 bf16 per thread per smem access.
    static constexpr int VEC = 8;
    // n-columns owned by each consumer thread.
    static constexpr int NPT = BN / CONSUMER_THREADS;

    static constexpr bool DIRECT_OUTPUT = (SPLIT_K == 1);

    static_assert(BN % CONSUMER_THREADS == 0,
                  "BN must be a multiple of the consumer thread count");
    static_assert(NPT >= 1, "BN must be >= CONSUMER_THREADS (raise BN or lower CWG)");
    static_assert(BK % VEC == 0, "BK must be a multiple of 8");
    static_assert(BK * (int)sizeof(bf16) >= 128, "128B swizzle needs BK >= 64");
    static_assert(BM <= 256 && BN <= 256 && BK <= 256, "TMA box dims are capped at 256");

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    // The 128B TMA swizzle takes its XOR from the *absolute* smem row index
    // within a 1024-byte (8 x 128B) block, not from a box-local row counter.
    // A stage stride that is not a whole number of 8-row blocks therefore
    // gives each stage a different swizzle phase, and swizzle_smem_offset()
    // (which assumes phase 0) silently reads the wrong elements. BM is tiny
    // here — BM=4 is only 4 rows — so pad the X stage stride up to a block.
    // W needs no padding: BN is always a multiple of 8.
    static constexpr int SWIZZLE_ROW_BLOCK = 8;
    static constexpr int X_STAGE_ROWS =
        ((BM + SWIZZLE_ROW_BLOCK - 1) / SWIZZLE_ROW_BLOCK) * SWIZZLE_ROW_BLOCK;
    static_assert(BN % SWIZZLE_ROW_BLOCK == 0, "BN must be a multiple of 8 rows");

    struct SMemStorage
    {
        // X must stay the FIRST member: smem_raw is declared __align__(1024),
        // so being at offset 0 is what puts stage 0 at swizzle phase 0. The
        // alignment is deliberately not repeated here — an __align__(1024) on
        // the member would round sizeof() up to a multiple of 1024 and make
        // SMEM_SIZE disagree with GemmConfig::smem_bytes().
        bf16 X[NUM_STAGES][X_STAGE_ROWS * BK];
        bf16 W[NUM_STAGES][BK * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    // workspace must hold >= SPLIT_K * M * N floats; unused when SPLIT_K == 1.
    static void run(
        int M, int N, int K,
        const bf16 *__restrict__ X,
        const bf16 *__restrict__ W,
        bf16 *__restrict__ Y,
        float *__restrict__ workspace = nullptr,
        cudaStream_t stream = nullptr);
};

// ── Tiny-M CUDA-core kernel ────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K>
__global__ void __launch_bounds__(BF16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::TOTAL_THREADS, 1, 1)
    bf16_gemm_tinym_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int num_k_per_split,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        bf16 *__restrict__ Y,
        float *__restrict__ workspace)
{
    using P = BF16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
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

    // ── Initialize barriers (once) ─────────────────────────────────
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

    // ── Producer warp group (wg_id == 0) ───────────────────────────
    // Identical to the tensor-core kernel: the consumer swap does not change
    // what needs to be staged.
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
                    {
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);
                    }

                    mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), P::TX_BYTES);

                    tma_X.load_2d(
                        k * BK, bm * BM,
                        smem_u32(smem.X[stage]),
                        smem_u32(&smem.full_barrier[stage]));

                    tma_W.load_2d(
                        k * BK, bn * BN,
                        smem_u32(smem.W[stage]),
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
    // ── Consumer warp groups (wg_id >= 1) ──────────────────────────
    else
    {
        // Consumer-thread index in [0, CONSUMER_THREADS).
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

            // This thread's output tile: all BM rows x its NPT columns of n.
            // Lives in registers for the whole k-range: no reduction needed.
            float acc[P::NPT][BM]{};

            for (int k = k_start; k < k_end; k++)
            {
                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const bf16 *sX = smem.X[stage];
                const bf16 *sW = smem.W[stage];

                // SMEM ACCESS: with the 128B swizzle and BK=64 (one 128-byte
                // row per n), element (n, kk) sits at byte
                //   n*128 + ((kk/8) ^ (n&7))*16 + (kk%8)*2
                // so eight consecutive threads (eight consecutive n) land on
                // eight distinct 16-byte banks and together cover all 32
                // banks exactly once — the W loads are conflict-free. Every
                // thread reads the same X address, which is a broadcast.
#pragma unroll
                for (int kk = 0; kk < BK; kk += P::VEC)
                {
                    // The full M extent of this 8-wide k chunk, shared by
                    // every thread in the warp (a broadcast read).
                    float xf[BM][P::VEC];
#pragma unroll
                    for (int m = 0; m < BM; m++)
                        load_bf16x8_f32(
                            &sX[swizzle_smem_offset<P::SWIZZLE_BYTES>(m, kk, BK)], xf[m]);

#pragma unroll
                    // One column of n per iteration; the 32 lanes of a warp
                    // cover 32 consecutive n for a given p.
                    for (int p = 0; p < P::NPT; p++)
                    {
                        const int n = p * P::CONSUMER_THREADS + tid_c;
                        float wf[P::VEC];
                        load_bf16x8_f32(
                            &sW[swizzle_smem_offset<P::SWIZZLE_BYTES>(n, kk, BK)], wf);

#pragma unroll
                        for (int e = 0; e < P::VEC; e++)
#pragma unroll
                            for (int m = 0; m < BM; m++)
                                acc[p][m] = fmaf(xf[m][e], wf[e], acc[p][m]);
                    }
                }

                __syncwarp();
                if (lane_id == 0)
                {
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                }
                __syncwarp();

                stage++;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase ^= 1;
                }
            }

            // ── Epilogue ───────────────────────────────────────────
            // Consecutive threads hold consecutive n, so both stores are
            // fully coalesced and need no smem staging.
            const int n_tile_base = bn * BN;
#pragma unroll
            for (int p = 0; p < P::NPT; p++)
            {
                const int n = n_tile_base + p * P::CONSUMER_THREADS + tid_c;
#pragma unroll
                for (int m = 0; m < BM; m++)
                {
                    if (m >= M) break; // BM may over-hang a shorter M
                    if constexpr (P::DIRECT_OUTPUT)
                        Y[(size_t)m * N + n] = __float2bfloat16(acc[p][m]);
                    else
                        workspace[(size_t)split_idx * M * N + (size_t)m * N + n] = acc[p][m];
                }
            }
        } // end consumer tile loop
    }

    // Cleanup barriers
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

// ── Tiny-M Launch ──────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K>
void BF16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::run(
    int M, int N, int K,
    const bf16 *__restrict__ X,
    const bf16 *__restrict__ W,
    bf16 *__restrict__ Y,
    float *__restrict__ workspace,
    cudaStream_t stream)
{
    if (M > BM || N % BN != 0 || K % (BK * SPLIT_K) != 0)
    {
        throw std::runtime_error(
            "TinyM: need M <= BM, N divisible by BN, K divisible by BK*SPLIT_K.");
    }

    // globalDim1 = M but the TMA box is BM rows tall: rows >= M are out of
    // bounds and the TMA unit zero-fills them, so the over-hanging
    // accumulators stay harmless and the epilogue simply drops them.
    TMADescriptor tma_X = create_tma_desc_2d(X, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d(W, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    int num_tiles_m = 1; // M <= BM, so there is exactly one row-tile
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;
    int num_k_per_split = (K / BK) / SPLIT_K;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles * SPLIT_K);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    auto kernel = bf16_gemm_tinym_kernel<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<grid, block, SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, num_k_per_split,
        tma_X, tma_W, Y, workspace);
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
