#pragma once
#include "common.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>
#include <cuda.h>
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

__device__ __forceinline__ uint32_t bf16x2_as_u32(bf16 lo, bf16 hi)
{
    bf16_2 v = make_bfloat162(lo, hi);
    uint32_t r;
    memcpy(&r, &v, 4);
    return r;
}

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b)
{
    bf16_2 v = __floats2bfloat162_rn(a, b);
    uint32_t r;
    memcpy(&r, &v, 4);
    return r;
}

// ── TMA descriptor helper ──────────────────────────────────────────────
// Wraps a CUtensorMap that lives on the host and is passed by-value to the kernel.
// The descriptor is 128 bytes; we store it in a uint64_t[16] so it can be trivially
// copied to __constant__ or kernel args.
struct TMADescriptor
{
    alignas(64) uint64_t raw[16]; // 128 bytes = CUtensorMap

    // cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    __device__ __forceinline__ void load_2d(uint32_t tile_coord0, uint32_t tile_coord1,
                                            uint64_t smem_addr, uint64_t mbar_addr) const
    {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "l"(smem_addr), "l"((const uint64_t *)raw), "r"(tile_coord0), "r"(tile_coord1),
              "r"((uint32_t)mbar_addr)
            : "memory");
    }
};

// cp.async.bulk.tensor.2d.global.shared::cta (TMA store)
__device__ __forceinline__ void tma_store_2d(
    const uint64_t *tma_desc,
    uint32_t tile_coord0, uint32_t tile_coord1,
    uint64_t smem_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];" ::
            "l"(tma_desc),
        "r"(tile_coord0), "r"(tile_coord1),
        "l"(smem_addr)
        : "memory");
}

__device__ __forceinline__ void tma_store_arrive()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait()
{
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Host-side: create a 2D TMA descriptor for a row-major or column-major bf16 matrix
// globalDim0 = inner-most (contiguous) dimension, globalDim1 = outer dimension
// boxDim0 / boxDim1 = tile sizes along each dimension
static inline TMADescriptor create_tma_desc_2d(
    const bf16 *gmem_ptr,
    uint32_t globalDim0, uint32_t globalDim1,
    uint32_t boxDim0, uint32_t boxDim1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE)
{
    TMADescriptor desc{};
    CUtensorMap *map = reinterpret_cast<CUtensorMap *>(&desc.raw);

    uint64_t globalDims[2] = {globalDim0, globalDim1};
    uint64_t globalStrides[1] = {globalDim0 * sizeof(bf16)}; // stride of dim-1 in bytes
    uint32_t boxDims[2] = {boxDim0, boxDim1};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // rank
        (void *)gmem_ptr,
        globalDims,
        globalStrides,
        boxDims,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS)
    {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errStr ? errStr : "unknown");
        exit(EXIT_FAILURE);
    }
    return desc;
}

// ── mbarrier helpers (PTX) ─────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(mbar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint64_t *mbar)
{
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"l"(mbar) : "memory");
}

// Expected transaction bytes for TMA loads arriving at this mbarrier
__device__ __forceinline__ void mbarrier_expect_tx(uint64_t smem_mbar_addr, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((uint32_t)smem_mbar_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t smem_mbar_addr)
{
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)smem_mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t smem_mbar_addr, uint32_t phase)
{
    // spin-wait
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP;\n"
        "}\n" ::"r"((uint32_t)smem_mbar_addr),
        "r"(phase) : "memory");
}

// ── smem address helper ────────────────────────────────────────────────
__device__ __forceinline__ uint32_t smem_u32(const void *smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

// Compute swizzled shared memory offset for TMA swizzle modes.
// SWIZZLE_BYTES: 0=none, 32/64/128 for corresponding mode.
// row/col: logical position in tile; row_elems: elements per row.
template <int SWIZZLE_BYTES, int ELEM_BYTES = 2>
__device__ __forceinline__ int swizzle_smem_offset(int row, int col, int row_elems)
{
    if constexpr (SWIZZLE_BYTES == 0)
    {
        return row * row_elems + col;
    }
    else
    {
        constexpr int BANKS = SWIZZLE_BYTES / 16;
        int col_bytes = col * ELEM_BYTES;
        int seg       = col_bytes / SWIZZLE_BYTES;
        int in_seg    = col_bytes % SWIZZLE_BYTES;
        int bank      = in_seg >> 4;
        int off       = in_seg & 0xF;
        int sw_bank   = bank ^ (row & (BANKS - 1));
        int sw_col    = (seg * SWIZZLE_BYTES + sw_bank * 16 + off) / ELEM_BYTES;
        return row * row_elems + sw_col;
    }
}

// ════════════════════════════════════════════════════════════════════════
// BF16GemmMMA — uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// ════════════════════════════════════════════════════════════════════════
//
// Warp tiling: each consumer warp handles a WARP_M × WARP_N region of the
// output tile. The warp iterates over (WARP_M/16) × (WARP_N/8) MMA tiles
// and (BK/16) k-inner steps per pipeline stage.
//
// WARP_M, WARP_N are template parameters so the user can tune them.
// Constraint: num_consumer_warps * WARP_M * WARP_N == BM * BN
//   where num_consumer_warps = CWG * 4.
// Default arrangement: warps are laid out in a 2D grid over the BM×BN tile.

template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M = 64, int WARP_N = 64>
struct BF16GemmMMA
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(bf16);

    // Warp tiling
    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 16;

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM×BN exactly");

    static constexpr int ACC_REGS = MMA_M * MMA_N * 4;

    // Shared memory swizzle size in bytes (0=none, 32/64/128)
    static constexpr int SWIZZLE_BYTES = 128;

    // Swizzle width for tile rasterization (number of N-tiles per super-column)
    static constexpr int SWIZZLE_WIDTH = 4;

    // Rasterize tile_id to (bm, bn) with swizzled ordering for L2 locality
    __device__ static void rasterize_tile(int tile_id, int num_tiles_m, int num_tiles_n, int &bm, int &bn)
    {
        // Clamp swizzle width to actual N tiles
        int sw = SWIZZLE_WIDTH < num_tiles_n ? SWIZZLE_WIDTH : num_tiles_n;
        int tiles_per_super_col = num_tiles_m * sw;
        int super_col = tile_id / tiles_per_super_col;
        int within = tile_id % tiles_per_super_col;
        int bn_base = super_col * sw;
        // Handle last partial super-column
        int actual_sw = (bn_base + sw <= num_tiles_n) ? sw : (num_tiles_n - bn_base);
        bm = within / actual_sw;
        bn = bn_base + within % actual_sw;
    }

    struct SMemStorage
    {
        bf16 X[NUM_STAGES][BM * BK];
        bf16 W[NUM_STAGES][BK * BN];
        bf16 Y_out[BM * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const bf16 *__restrict__ X,
        const bf16 *__restrict__ W,
        bf16 *__restrict__ Y,
        cudaStream_t stream = nullptr);
};

// ── MMA kernel ─────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
__global__ void __launch_bounds__(BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::TOTAL_THREADS, 1, 1)
    bf16_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        __grid_constant__ const TMADescriptor tma_Y,
        bf16 *__restrict__ Y)
{
    using P = BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>;
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

    // ── Initialize barriers (once) ─────────────────────────────────
    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            // full_barrier: only the producer's expect_tx arrives (count=1)
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ── Producer warp group (wg_id == 0) ───────────────────────────
    if (wg_id == 0)
    {

        if (warp_in_wg == 0 && lane_id == 0)
        {
            int stage = 0;
            int phase = 0;
            int total_k = 0;

            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
            {
                int bm, bn;
                P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

                for (int k = 0; k < num_k_tiles; k++)
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
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;

        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        int stage = 0;
        int phase = 0;
        bool has_tma_store_in_flight = false;

        for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
        {
            int bm, bn;
            P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};

            for (int k = 0; k < num_k_tiles; k++)
            {
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const bf16 *sX = smem.X[stage];
                const bf16 *sW = smem.W[stage];
#pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 16;

                    // Preload all B fragments for this k-step via ldmatrix
                    uint32_t b_frag[P::MMA_N][2];
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;
                        int b_n = n_base + (lane_id & 7);
                        int b_k = k_base + (((lane_id >> 3) & 1) * 8);
                        uint32_t b_addr = smem_u32(&sW[swizzle_smem_offset<P::SWIZZLE_BYTES>(b_n, b_k, BK)]);
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                            : "r"(b_addr));
                    }
#pragma unroll
                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        const int m_base = m_warp_base + mi * 16;

                        // Load A fragment via ldmatrix (only depends on mi, ki)
                        uint32_t a[4];
                        {
                            int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                            int a_col = k_base + (lane_id >> 4) * 8;
                            uint32_t a_addr = smem_u32(&sX[swizzle_smem_offset<P::SWIZZLE_BYTES>(a_row, a_col, BK)]);
                            asm volatile(
                                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                : "r"(a_addr));
                        }
#pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                        {
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%10, %11, %12, %13};\n"
                                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                                  "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                  "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                                  "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]),
                                  "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3]));
                        }
                    }
                }

                if (lane_id == 0)
                {
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                }

                stage++;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase ^= 1;
                }
            }

            // ── Store output via smem + TMA store ──────────────────
            // Wait for any previous TMA store before writing to smem.Y_out
            if (has_tma_store_in_flight)
            {
                if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                {
                    tma_store_wait();
                }
                // Sync ALL consumer WGs so everyone sees the wait complete
                asm volatile("bar.sync %0, %1;" :: "r"(P::TOTAL_WGS), "r"(CWG * P::THREADS_PER_WG) : "memory");
            }

            bf16 *sY = smem.Y_out;

            for (int mi = 0; mi < P::MMA_M; mi++)
            {
                for (int ni = 0; ni < P::MMA_N; ni++)
                {
                    int m_base = m_warp_base + mi * 16;
                    int n_base = n_warp_base + ni * 8;

                    uint32_t c0 = f32x2_to_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
                    uint32_t c1 = f32x2_to_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
                    int st_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                    uint32_t st_addr = smem_u32(&sY[st_row * BN + n_base]);
                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                        :: "r"(st_addr), "r"(c0), "r"(c1));
                }
            }

            // Sync all consumers before TMA store
            asm volatile("bar.sync %0, %1;" :: "r"(P::TOTAL_WGS), "r"(CWG * P::THREADS_PER_WG) : "memory");

            // TMA store: one thread issues the store
            if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
            {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                tma_store_2d(
                    reinterpret_cast<const uint64_t *>(tma_Y.raw),
                    bn * BN, bm * BM,
                    smem_u32(sY));
                tma_store_arrive();
            }
            has_tma_store_in_flight = true;
        } // end consumer tile loop

        // Wait for last TMA store before exit
        if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
        {
            if (has_tma_store_in_flight)
            {
                tma_store_wait();
            }
        }
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

// ── MMA Launch ─────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N>
void BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>::run(
    int M, int N, int K,
    const bf16 *__restrict__ X,
    const bf16 *__restrict__ W,
    bf16 *__restrict__ Y,
    cudaStream_t stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");
    }

    TMADescriptor tma_X = create_tma_desc_2d(X, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d(W, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    // Y is row-major M×N, so dim0 (contiguous) = N, dim1 = M, box = BN×BM
    // boxDim0=BN=128 → 256 bytes > 128B swizzle limit, so use NONE for Y
    TMADescriptor tma_Y = create_tma_desc_2d(Y, N, M, BN, BM);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    // Launch exactly num_sm blocks for persistent kernel
    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    CHECK_CUDA(cudaFuncSetAttribute(bf16_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    bf16_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N><<<grid, block, SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, tma_X, tma_W, tma_Y, Y);
    CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════
// BF16GemmMMASplitK — MMA kernel with split-K for better SM utilization
//   on small matrices. Two kernels: MMA partial sums → reduction.
// ════════════════════════════════════════════════════════════════════════

template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M, int WARP_N, int SPLIT_K>
struct BF16GemmMMASplitK
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(bf16);

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 16;

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM×BN exactly");

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    __device__ static void rasterize_tile(int tile_id, int num_tiles_m, int num_tiles_n, int &bm, int &bn)
    {
        int sw = SWIZZLE_WIDTH < num_tiles_n ? SWIZZLE_WIDTH : num_tiles_n;
        int tiles_per_super_col = num_tiles_m * sw;
        int super_col = tile_id / tiles_per_super_col;
        int within = tile_id % tiles_per_super_col;
        int bn_base = super_col * sw;
        int actual_sw = (bn_base + sw <= num_tiles_n) ? sw : (num_tiles_n - bn_base);
        bm = within / actual_sw;
        bn = bn_base + within % actual_sw;
    }

    struct SMemStorage
    {
        bf16 X[NUM_STAGES][BM * BK];
        bf16 W[NUM_STAGES][BK * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const bf16 *__restrict__ X,
        const bf16 *__restrict__ W,
        bf16 *__restrict__ Y,
        float *__restrict__ workspace, // size >= SPLIT_K * M * N
        cudaStream_t stream = nullptr);
};

// ── Split-K MMA kernel ────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, int SPLIT_K>
__global__ void __launch_bounds__(BF16GemmMMASplitK<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>::TOTAL_THREADS, 1, 1)
    bf16_gemm_mma_splitk_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int num_k_per_split,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        float *__restrict__ workspace)
{
    using P = BF16GemmMMASplitK<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(128) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int num_blocks = gridDim.x;
    const int total_vtiles = total_tiles * SPLIT_K;

    // ── Initialize barriers ────────────────────────────────────────
    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ── Producer warp group (wg_id == 0) ───────────────────────────
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
                P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

                int k_start = split_idx * num_k_per_split;
                int k_end = k_start + num_k_per_split;

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
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;

        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        int stage = 0;
        int phase = 0;

        for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
        {
            int tile_id = vtid / SPLIT_K;
            int split_idx = vtid % SPLIT_K;
            int bm, bn;
            P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

            int k_start = split_idx * num_k_per_split;
            int k_end = k_start + num_k_per_split;

            float acc[P::MMA_M][P::MMA_N][4]{};
            for (int k = k_start; k < k_end; k++)
            {
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const bf16 *sX = smem.X[stage];
                const bf16 *sW = smem.W[stage];

                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 16;

                    // Preload all B fragments for this k-step via ldmatrix
                    uint32_t b_frag[P::MMA_N][2];
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;
                        int b_n = n_base + (lane_id & 7);
                        int b_k = k_base + (((lane_id >> 3) & 1) * 8);
                        uint32_t b_addr = smem_u32(&sW[swizzle_smem_offset<P::SWIZZLE_BYTES>(b_n, b_k, BK)]);
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                            : "r"(b_addr));
                    }

                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        const int m_base = m_warp_base + mi * 16;

                        // Load A fragment via ldmatrix (only depends on mi, ki)
                        uint32_t a[4];
                        {
                            int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                            int a_col = k_base + (lane_id >> 4) * 8;
                            uint32_t a_addr = smem_u32(&sX[swizzle_smem_offset<P::SWIZZLE_BYTES>(a_row, a_col, BK)]);
                            asm volatile(
                                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                : "r"(a_addr));
                        }

                        for (int ni = 0; ni < P::MMA_N; ni++)
                        {
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%10, %11, %12, %13};\n"
                                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                                  "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                  "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                                  "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]),
                                  "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3]));
                        }
                    }
                }

                if (lane_id == 0)
                {
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                }

                stage++;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase ^= 1;
                }
            }
            // mysterious sync..
            // otherwise it triggers a mysterious bug in split k
            asm volatile("bar.sync %0, %1;" :: "r"(P::TOTAL_WGS), "r"(CWG * P::THREADS_PER_WG) : "memory");
            // ── Store f32 partial results directly to global workspace ──
            {
                float *ws = workspace + (size_t)split_idx * M * N;
                const int gm_base = bm * BM;
                const int gn_base = bn * BN;
#pragma unroll
                for (int mi = 0; mi < P::MMA_M; mi++)
                {
#pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        int m_base = m_warp_base + mi * 16;
                        int n_base = n_warp_base + ni * 8;

                        int row0 = gm_base + m_base + (lane_id >> 2);
                        int col0 = gn_base + n_base + (lane_id & 3) * 2;
                        ws[row0 * N + col0]     = acc[mi][ni][0];
                        ws[row0 * N + col0 + 1] = acc[mi][ni][1];
                        int row1 = row0 + 8;
                        ws[row1 * N + col0]     = acc[mi][ni][2];
                        ws[row1 * N + col0 + 1] = acc[mi][ni][3];
                    }
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

// ── Split-K reduction kernel ──────────────────────────────────────────
// Sums SPLIT_K partial f32 results and converts to bf16.
// Coalesced: consecutive threads access consecutive elements.
template <int SPLIT_K>
__global__ void splitk_reduce_kernel(
    const float *__restrict__ workspace,
    bf16 *__restrict__ Y,
    size_t MN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    float sum = workspace[idx];
    #pragma unroll
    for (int s = 1; s < SPLIT_K; s++)
        sum += workspace[(size_t)s * MN + idx];
    Y[idx] = bf16(sum);
}

// ── Split-K MMA Launch ────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, int SPLIT_K>
void BF16GemmMMASplitK<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>::run(
    int M, int N, int K,
    const bf16 *__restrict__ X,
    const bf16 *__restrict__ W,
    bf16 *__restrict__ Y,
    float *__restrict__ workspace,
    cudaStream_t stream)
{
    if (M % BM != 0 || N % BN != 0 || K % (BK * SPLIT_K) != 0)
    {
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK*SPLIT_K respectively.");
    }

    TMADescriptor tma_X = create_tma_desc_2d(X, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_W = create_tma_desc_2d(W, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;
    int num_k_tiles = K / BK;
    int num_k_per_split = num_k_tiles / SPLIT_K;
    int total_vtiles = total_tiles * SPLIT_K;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_vtiles);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    CHECK_CUDA(cudaFuncSetAttribute(
        bf16_gemm_mma_splitk_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    bf16_gemm_mma_splitk_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, SPLIT_K>
        <<<grid, block, SMEM_SIZE, stream>>>(
            M, N, K, num_tiles_m, num_tiles_n, total_tiles, num_k_per_split,
            tma_X, tma_W, workspace);
    CHECK_CUDA(cudaGetLastError());

    // Reduction: sum SPLIT_K partials → bf16 output
    constexpr int REDUCE_THREADS = 512;
    auto MN = (size_t)M * (size_t)N;
    auto reduce_blocks = (MN + REDUCE_THREADS - 1) / REDUCE_THREADS;
    splitk_reduce_kernel<SPLIT_K><<<reduce_blocks, REDUCE_THREADS, 0, stream>>>(
        workspace, Y, MN);
    CHECK_CUDA(cudaGetLastError());
}