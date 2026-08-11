#pragma once
#include "bf16_gemm.cuh" // TMA / mbarrier / swizzle / ldmatrix helpers

#include <cuda_fp8.h>

using fp8e4m3 = __nv_fp8_e4m3;

// ════════════════════════════════════════════════════════════════════════
// FP8GemmMMA — e4m3 GEMM with per-tensor scaling
// ════════════════════════════════════════════════════════════════════════
//
// Structurally a copy of BF16GemmMMA: same persistent CTAs, same producer
// warp group driving a multi-stage TMA/mbarrier pipeline, same warp tiling,
// same epilogue. The consumer swaps mma.m16n8k16.bf16 for
// mma.m16n8k32.e4m3, and everything else follows from that one change.
//
// ── What actually differs from the bf16 kernel ────────────────────────
//
// 1. k per MMA doubles, 16 -> 32, so MMA_K = BK / 32.
//
// 2. BK doubles, 64 -> 128. The 128B swizzle needs each smem row to be
//    exactly 128 bytes, and an fp8 element is 1 byte, so BK = 128. A stage
//    therefore costs the same shared memory as the bf16 kernel's BK=64 while
//    covering twice the k.
//
// 3. The fragment *addressing is unchanged in bytes*. An A/B register holds
//    4 bytes either way — 2 bf16 or 4 fp8 — so the same ldmatrix.b16
//    instructions work on fp8 data reinterpreted as b16 pairs, and the only
//    edit is that per-lane k offsets count elements rather than bytes:
//        bf16   a_col = k_base + (lane>>4)*8       (8 bf16 = 16 B)
//        fp8    a_col = k_base + (lane>>4)*16      (16 fp8 = 16 B)
//
// 4. The accumulator layout is *identical* (4 fp32 per lane, m16n8), so the
//    epilogue is unchanged apart from applying the dequantization scale.
//
// ── Per-lane m16n8k32 fragment layout (verified against a CPU reference) ──
//
//   D[16x8] = A[16x32] * B[32x8] + C[16x8],  A row-major, B col-major.
//   With group = lane >> 2 (0..7) and lig = lane & 3 (0..3), each A/B
//   register packs FOUR fp8 along k, byte j holding k offset j:
//
//     A   a[0]  m = group      k = lig*4 + 0..3
//         a[1]  m = group + 8  k = lig*4 + 0..3
//         a[2]  m = group      k = lig*4 + 16..19
//         a[3]  m = group + 8  k = lig*4 + 16..19
//     B   b[0]  n = group      k = lig*4 + 0..3
//         b[1]  n = group      k = lig*4 + 16..19
//     C   c[0] = (group, lig*2)      c[1] = (group, lig*2 + 1)
//         c[2] = (group + 8, lig*2)  c[3] = (group + 8, lig*2 + 1)
//
// Reading an 8x8 b16 tile through ldmatrix hands lane l exactly
// (row = l>>2, bytes (l&3)*4 .. +3), which is precisely a[0]/b[0] above —
// that is why the bf16 loaders carry over untouched.
//
// ── Scaling ───────────────────────────────────────────────────────────
//
// Per-tensor: the caller passes the single product x_scale * w_scale and the
// epilogue applies it to the fp32 accumulator before narrowing to bf16.
// (A production kernel would take a device pointer so the scale can be
// produced on-device; a host float is enough for the prototype.)

// D = A*B + D over one m16n8k32 e4m3 tile, accumulating in fp32.
__device__ __forceinline__ void mma_m16n8k32_e4m3(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M = 64, int WARP_N = 64>
struct FP8GemmMMA
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(fp8e4m3);

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 32; // k=32 per fp8 MMA, vs 16 for bf16

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM×BN exactly");
    static_assert(BK % 32 == 0, "BK must be a multiple of the MMA's k=32");
    static_assert(BK * (int)sizeof(fp8e4m3) >= 128,
                  "128B swizzle needs a 128-byte smem row, so BK >= 128 for fp8");

    static constexpr int ACC_REGS = MMA_M * MMA_N * 4;
    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    struct SMemStorage
    {
        fp8e4m3 X[NUM_STAGES][BM * BK];
        fp8e4m3 W[NUM_STAGES][BK * BN];
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
        float scale, // x_scale * w_scale
        cudaStream_t stream = nullptr);
};

// ── FP8 MMA kernel ─────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
__global__ void __launch_bounds__(FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::TOTAL_THREADS, 1, 1)
    fp8_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        float scale,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        __grid_constant__ const TMADescriptor tma_Y)
{
    using P = FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>;
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

    // ── Producer warp group (wg_id == 0) ───────────────────────────
    // Unchanged from the bf16 kernel: only the element size differs, and
    // that is baked into the TMA descriptors and TX_BYTES.
    if (wg_id == 0)
    {
        if (warp_in_wg == 0 && lane_id == 0)
        {
            int stage = 0, phase = 0, total_k = 0;
            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
            {
                int bm, bn;
                rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n, P::SWIZZLE_WIDTH, bm, bn);

                for (int k = 0; k < num_k_tiles; k++)
                {
                    if (total_k >= NUM_STAGES)
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);

                    mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), P::TX_BYTES);

                    tma_X.load_2d(k * BK, bm * BM,
                                  smem_u32(smem.X[stage]),
                                  smem_u32(&smem.full_barrier[stage]));
                    tma_W.load_2d(k * BK, bn * BN,
                                  smem_u32(smem.W[stage]),
                                  smem_u32(&smem.full_barrier[stage]));

                    stage++;
                    if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                    total_k++;
                }
            }
        }
    }
    // ── Consumer warp groups (wg_id >= 1) ──────────────────────────
    else
    {
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;

        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        int stage = 0, phase = 0;
        bool has_tma_store_in_flight = false;

        for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
        {
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n, P::SWIZZLE_WIDTH, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};

            for (int k = 0; k < num_k_tiles; k++)
            {
                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const fp8e4m3 *sX = smem.X[stage];
                const fp8e4m3 *sW = smem.W[stage];
#pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 32; // 32 fp8 per MMA

                    // B fragment: two 8x8 b16 tiles = 8(n) x 32(k) of fp8.
                    // Same 16-byte lane stride as bf16; only the element count
                    // doubles (16 fp8 == 8 bf16 == 16 bytes).
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

                        // A fragment: four 8x8 b16 tiles = 16(m) x 32(k) of fp8.
                        uint32_t a[4];
                        {
                            int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                            int a_col = k_base + (lane_id >> 4) * 16;
                            ldmatrix_x4(a,
                                        smem_u32(&sX[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(a_row, a_col, BK)]));
                        }
#pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                            mma_m16n8k32_e4m3(acc[mi][ni], a, b_frag[ni]);
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
            // Identical to the bf16 kernel: the m16n8 accumulator layout is
            // the same, so only the dequantization scale is new.
            if (has_tma_store_in_flight)
            {
                if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                    tma_store_wait();
                named_barrier_sync(P::TOTAL_WGS, CWG * P::THREADS_PER_WG);
            }

            bf16 *sY = smem.Y_out;
            for (int mi = 0; mi < P::MMA_M; mi++)
            {
                for (int ni = 0; ni < P::MMA_N; ni++)
                {
                    int m_base = m_warp_base + mi * 16;
                    int n_base = n_warp_base + ni * 8;

                    uint32_t c0 = f32x2_to_bf16x2(acc[mi][ni][0] * scale, acc[mi][ni][1] * scale);
                    uint32_t c1 = f32x2_to_bf16x2(acc[mi][ni][2] * scale, acc[mi][ni][3] * scale);
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

// ── FP8 MMA Launch ─────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
void FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ X,
    const fp8e4m3 *__restrict__ W,
    bf16 *__restrict__ Y,
    float scale,
    cudaStream_t stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");

    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d_raw(W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    // Y stays bf16, so this descriptor is unchanged from the bf16 kernel.
    TMADescriptor tma_Y = create_tma_desc_2d(Y, N, M, BN, BM);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles);

    auto kernel = fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<dim3(num_blocks), dim3(TOTAL_THREADS), SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, scale, tma_X, tma_W, tma_Y);
    CHECK_CUDA(cudaGetLastError());
}
