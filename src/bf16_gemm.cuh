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

// ── TMA descriptor helper ──────────────────────────────────────────────
// Wraps a CUtensorMap that lives on the host and is passed by-value to the kernel.
// The descriptor is 128 bytes; we store it in a uint64_t[16] so it can be trivially
// copied to __constant__ or kernel args.
struct TMADescriptor
{
    alignas(64) uint64_t raw[16]; // 128 bytes = CUtensorMap

    // cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    __device__ __forceinline__ void load_2d(uint32_t tile_coord0, uint32_t tile_coord1,
                                            uint64_t smem_addr, uint64_t mbar_addr)
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

/* PSEUDOCODE for BF16GemmSIMT. DO NOT DELETE
 void gemm(X, W, Y) {
    __shared__ bf16 shared_X[NUM_STAGES][BM * BK];
    __shared__ bf16 shared_W[NUM_STAGES][BK * BN];
    __shared__ mbarrier full_barrier[NUM_STAGES];
    __shared__ mbarrier empty_barrier[NUM_STAGES];
    __shared__ y_output_tile[BM * BN];
    int phase = 0;
    if (is_producer_wg) {
        decrease_registers();
        if (warp_id_in_wg == 0) {
            int stage = 0;
            for (each tile) {
                wait(empty_barrier[stage], phase ^ 1); // previous phase
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
        increase_registers();
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
        // synchronized store first
        store(Y, Y_tile);

        // use TMA store later
        // store(y_output_tile, Y_tile);
        // tma_store(Y, y_output_tile);
    }
 }

 */

// ── BF16GemmSIMT struct ────────────────────────────────────────────────────
// CWG = number of consumer warp groups. Total WGs = CWG + 1 (1 producer).
// Each warp group = 4 warps = 128 threads.
template <int BM, int BN, int BK, int NUM_STAGES, int CWG>
struct BF16GemmSIMT
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP; // 128
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(bf16);

    static constexpr int ROWS_PER_CWG = BM / CWG;
    static constexpr int ELEMS_PER_CWG = ROWS_PER_CWG * BN;
    static constexpr int ELEMS_PER_THREAD = ELEMS_PER_CWG / THREADS_PER_WG;

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
        cudaStream_t stream = nullptr);
};

// ── Kernel (free function — CUDA requires __global__ to be non-member) ─
template <int BM, int BN, int BK, int NUM_STAGES, int CWG>
__global__ void __launch_bounds__(BF16GemmSIMT<BM, BN, BK, NUM_STAGES, CWG>::TOTAL_THREADS, 1, 1)
    bf16_gemm_kernel(
        int M, int N, int K,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        bf16 *__restrict__ Y)
{
    using P = BF16GemmSIMT<BM, BN, BK, NUM_STAGES, CWG>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(128) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int num_k_tiles = K / BK;

    // ── Initialize barriers ────────────────────────────────────────
    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], CWG);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ── Producer warp group (wg_id == 0) ───────────────────────────
    if (wg_id == 0)
    {
        // asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" :::);

        if (warp_in_wg == 0 && lane_id == 0)
        {
            int stage = 0;
            int phase = 0;

            for (int k = 0; k < num_k_tiles; k++)
            {
                if (k >= NUM_STAGES)
                {
                    mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);
                }

                mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), P::TX_BYTES);

                const_cast<TMADescriptor &>(tma_X).load_2d(
                    k * BK, bm * BM,
                    smem_u32(smem.X[stage]),
                    smem_u32(&smem.full_barrier[stage]));

                const_cast<TMADescriptor &>(tma_W).load_2d(
                    k * BK, bn * BN,
                    smem_u32(smem.W[stage]),
                    smem_u32(&smem.full_barrier[stage]));

                stage++;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase ^= 1;
                }
            }
        }
    }
    // ── Consumer warp groups (wg_id >= 1) ──────────────────────────
    else
    {
        // asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" :::);

        const int cwg_id = wg_id - 1;
        const int row_start = cwg_id * P::ROWS_PER_CWG;
        const int tid_in_wg = warp_in_wg * P::THREADS_PER_WARP + lane_id;

        float acc[P::ELEMS_PER_THREAD];
        for (int i = 0; i < P::ELEMS_PER_THREAD; i++)
            acc[i] = 0.0f;

        int stage = 0;
        int phase = 0;

        for (int k = 0; k < num_k_tiles; k++)
        {
            mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

            const bf16 *sX = smem.X[stage];
            const bf16 *sW = smem.W[stage];

            for (int i = 0; i < P::ELEMS_PER_THREAD; i++)
            {
                int linear = tid_in_wg + i * P::THREADS_PER_WG;
                int m_local = linear / BN;
                int n = linear % BN;
                int m = row_start + m_local;

                float sum = 0.0f;
                for (int kk = 0; kk < BK; kk++)
                {
                    auto x_val = float(sX[m * BK + kk]);
                    auto w_val = float(sW[n * BK + kk]);
                    sum += x_val * w_val;
                }
                acc[i] += sum;
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

        // ── Store output (synchronized store) ──────────────────────
        __syncthreads();

        for (int i = 0; i < P::ELEMS_PER_THREAD; i++)
        {
            int linear = tid_in_wg + i * P::THREADS_PER_WG;
            int m_local = linear / BN;
            int n = linear % BN;
            int m = (bm * BM) + row_start + m_local;
            int gn = bn * BN + n;
            Y[m * N + gn] = bf16(acc[i]);
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

// ── Launch ─────────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG>
void BF16GemmSIMT<BM, BN, BK, NUM_STAGES, CWG>::run(
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

    TMADescriptor tma_X = create_tma_desc_2d(X, K, M, BK, BM);
    TMADescriptor tma_W = create_tma_desc_2d(W, K, N, BK, BN);

    dim3 grid(M / BM, N / BN, 1);
    dim3 block(TOTAL_THREADS);

    cudaFuncSetAttribute(bf16_gemm_kernel<BM, BN, BK, NUM_STAGES, CWG>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    bf16_gemm_kernel<BM, BN, BK, NUM_STAGES, CWG><<<grid, block, SMEM_SIZE, stream>>>(
        M, N, K, tma_X, tma_W, Y);
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
            mbarrier_init(&smem.full_barrier[s], CWG);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    const int groupID = lane_id >> 2;
    const int threadID_in_group = lane_id & 3;

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

                    const_cast<TMADescriptor &>(tma_X).load_2d(
                        k * BK, bm * BM,
                        smem_u32(smem.X[stage]),
                        smem_u32(&smem.full_barrier[stage]));

                    const_cast<TMADescriptor &>(tma_W).load_2d(
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

            float acc[P::MMA_M][P::MMA_N][4];
            for (int mi = 0; mi < P::MMA_M; mi++)
                for (int ni = 0; ni < P::MMA_N; ni++)
                    for (int r = 0; r < 4; r++)
                        acc[mi][ni][r] = 0.0f;

            for (int k = 0; k < num_k_tiles; k++)
            {
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const bf16 *sX = smem.X[stage];
                const bf16 *sW = smem.W[stage];

                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 16;

                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        const int m_base = m_warp_base + mi * 16;

                        for (int ni = 0; ni < P::MMA_N; ni++)
                        {
                            const int n_base = n_warp_base + ni * 8;

                            uint32_t a[4];
                            {
                                int r0 = m_base + groupID;
                                int c0 = k_base + threadID_in_group * 2;
                                int i0 = swizzle_smem_offset<P::SWIZZLE_BYTES>(r0, c0, BK);
                                a[0] = bf16x2_as_u32(sX[i0], sX[i0 + 1]);

                                int r1 = m_base + groupID + 8;
                                int i1 = swizzle_smem_offset<P::SWIZZLE_BYTES>(r1, c0, BK);
                                a[1] = bf16x2_as_u32(sX[i1], sX[i1 + 1]);

                                int c1 = k_base + threadID_in_group * 2 + 8;
                                int i2 = swizzle_smem_offset<P::SWIZZLE_BYTES>(r0, c1, BK);
                                a[2] = bf16x2_as_u32(sX[i2], sX[i2 + 1]);
                                int i3 = swizzle_smem_offset<P::SWIZZLE_BYTES>(r1, c1, BK);
                                a[3] = bf16x2_as_u32(sX[i3], sX[i3 + 1]);
                            }

                            uint32_t b[2];
                            {
                                int bk0 = k_base + threadID_in_group * 2;
                                int bn0 = n_base + groupID;
                                int j0 = swizzle_smem_offset<P::SWIZZLE_BYTES>(bn0, bk0, BK);
                                b[0] = bf16x2_as_u32(sW[j0], sW[j0 + 1]);

                                int bk1 = k_base + threadID_in_group * 2 + 8;
                                int j1 = swizzle_smem_offset<P::SWIZZLE_BYTES>(bn0, bk1, BK);
                                b[1] = bf16x2_as_u32(sW[j1], sW[j1 + 1]);
                            }

                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%10, %11, %12, %13};\n"
                                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                                  "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                  "r"(b[0]), "r"(b[1]),
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
                if (warp_in_wg == 0 && lane_id == 0)
                {
                    tma_store_wait();
                }
                asm volatile("bar.sync %0, %1;" :: "r"(wg_id), "r"(P::THREADS_PER_WG) : "memory");
            }

            bf16 *sY = smem.Y_out;

            for (int mi = 0; mi < P::MMA_M; mi++)
            {
                for (int ni = 0; ni < P::MMA_N; ni++)
                {
                    int m_base = m_warp_base + mi * 16;
                    int n_base = n_warp_base + ni * 8;

                    int sm0 = m_base + groupID;
                    int sn0 = n_base + threadID_in_group * 2;
                    sY[sm0 * BN + sn0]     = bf16(acc[mi][ni][0]);
                    sY[sm0 * BN + sn0 + 1] = bf16(acc[mi][ni][1]);
                    int sm1 = m_base + groupID + 8;
                    sY[sm1 * BN + sn0]     = bf16(acc[mi][ni][2]);
                    sY[sm1 * BN + sn0 + 1] = bf16(acc[mi][ni][3]);
                }
            }

            // Warpgroup sync before TMA store
            asm volatile("bar.sync %0, %1;" :: "r"(wg_id), "r"(P::THREADS_PER_WG) : "memory");

            // TMA store: one thread per consumer warp group issues the store
            if (warp_in_wg == 0 && lane_id == 0)
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
        if (warp_in_wg == 0 && lane_id == 0)
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