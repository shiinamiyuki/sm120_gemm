#pragma once
#include "bf16_gemm.cuh" // TMA / mbarrier / ldmatrix / mma / splitk_reduce_kernel
#include "block_scale.h"
#include "w4a16_gemm_tinym.cuh" // ue4m3_to_float_fast, load_fp4x32_f32

#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ════════════════════════════════════════════════════════════════════════
// W4A16GemmTinyMTC — NVFP4 weights x bf16 activations on TENSOR cores
// ════════════════════════════════════════════════════════════════════════
//
// The CUDA-core sibling in w4a16_gemm_tinym.cuh spends BM FMAs per weight
// element, so its cost grows linearly with M: measured 0.0206 ms at M=1 but
// 0.0348 ms at M=8 on N=4096 K=14336, i.e. 57% of the bandwidth wall. cuDNN's
// dense NVFP4 GEMM does the same shapes in a flat 0.026 ms for M=1, 4 and 8 --
// M-independent, because a tensor core swallows the whole m16 tile whether one
// row is live or sixteen. That flatness is the thing worth copying.
//
// The old header note claimed a tensor-core W4A16 "would have to dequantize the
// weights to bf16 first, which throws away the whole point at decode shapes".
// That is wrong at M >= 4: the dequant is ~1.5 instructions per weight element
// against BM FMAs saved, and the weight stream is the same either way.
//
// ── Why no bf16 staging buffer ────────────────────────────────────────
//
// mma.m16n8k16 wants its B operand as 16(k) x 8(n) bf16, and ldmatrix cannot
// dequantize, so the obvious plan is to widen a W tile into smem and ldmatrix
// from there. That costs a second BN x BK buffer and a barrier per k-chunk.
//
// It is unnecessary. The B fragment ldmatrix *would* have produced is, per
// lane l, n = n_base + (l >> 2) and k = k_base + (l & 3) * 2 + {0,1} for the
// low 8 k, plus the same shifted by 8 -- which is exactly two bytes of packed
// e2m1, both inside one 16-element scale block. So each lane reads its own two
// bytes straight out of the packed tile and builds the fragment in registers.
//
// Those reads are conflict-free, and it is the existing 128B TMA swizzle that
// makes them so. Lane l reads byte column k_base/2 + (l & 3) of row
// n_base + (l >> 2); swizzle_smem_offset XORs the 16-byte group by (row & 7),
// so the eight rows a warp touches land in eight different bank groups, while
// the four lanes sharing a row read four bytes of one 4-byte word -- one bank
// access between them. Eight accesses, eight banks.
//
// ── Where the ue4m3 scale goes ────────────────────────────────────────
//
// mma.m16n8k16 covers exactly 16 k, which is exactly one NVFP4 scale block, so
// the scale could equally be folded into the weight before the MMA or applied
// to the accumulator after it. Folding is cheaper (4 multiplies per lane per
// MMA against 4 FFMA plus a second scale lookup) and it is exact: e2m1 has one
// mantissa bit and ue4m3 has three, so the product needs four and bf16 has
// eight.
//
// ── Where the time actually goes ──────────────────────────────────────
//
// At M=8 N=4096 K=14336 this runs at 1359 GB/s, 79% of the 1707 GB/s measured
// read wall, and it is tempting to read that as an inefficient mainloop. It is
// not. Hold the config and grow K:
//
//     K =  14336   0.0245 ms   1359 GB/s
//     K =  28672   0.0428 ms   1556 GB/s
//     K =  57344   0.0902 ms   1476 GB/s
//     K = 114688   0.1729 ms   1539 GB/s
//
// The steady state is ~1500-1550 GB/s, i.e. ~91% of the wall, and the decode
// shape gives up ~13% of that to costs that do not scale with K: the TMA
// pipeline fill and drain, and the split-k reduce launch that follows every
// call. A 25-microsecond kernel simply cannot amortise them.
//
// Three things were tried against the 79% and none of them moved it, which is
// how the above was arrived at:
//   * split-k values that divide the 56 k-tiles (7, 14, 28, 56) instead of
//     only powers of two -- sk8 is already best, and sk14 is 24% *worse*
//     because 224 vtiles over 170 SMs leaves 54 of them doing two.
//   * smaller BN to cut the split-k workspace -- every variant is slower.
//     BN=256 is worth its 1 MB workspace because halving the CTA count halves
//     how many times X is re-read.
//   * a cheaper dequantise (the fp16 scale path above) -- ~1%.
//
// So the remaining lever is the fixed cost, not the mainloop: folding the
// split-k reduction into the GEMM would remove a whole kernel launch from
// every call.
template <int BN, int NUM_STAGES, int CWG, int SPLIT_K = 1>
struct W4A16GemmTinyMTC
{
    static constexpr int BM = 16;  // the m extent of mma.m16n8k16; M <= BM
    static constexpr int BK = 256; // packed e2m1 row = 128B, the swizzle unit

    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;
    static constexpr int CONSUMER_THREADS = CWG * THREADS_PER_WG;
    static constexpr int CONSUMER_WARPS = CWG * WARPS_PER_WG;

    static constexpr int W_ROW_BYTES = BK / 2;
    static constexpr int TX_BYTES = BM * BK * (int)sizeof(bf16) + BN * W_ROW_BYTES;

    static constexpr int N_PER_WARP = BN / CONSUMER_WARPS;
    static constexpr int MMA_N = N_PER_WARP / 8;  // n-tiles of 8 per warp
    static constexpr int MMA_K = BK / 16;         // k-steps per stage
    static constexpr int SF_VEC = 16;
    static constexpr int SF_PACKS = (BK / SF_VEC) / 4; // 4 scale blocks per u32

    static constexpr bool DIRECT_OUTPUT = (SPLIT_K == 1);

    static_assert(BN % CONSUMER_WARPS == 0, "BN must split over the consumer warps");
    static_assert(N_PER_WARP % 8 == 0, "each warp needs a whole number of n8 tiles");
    static_assert(MMA_N >= 1, "BN too small for this many consumer warps");
    static_assert(BN <= 256, "TMA box dims are capped at 256");

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    // W first, for the same 1024-byte swizzle-phase reason as the CUDA-core
    // kernel: a W stage is BN * 128 bytes, a multiple of 1024, so every stage
    // starts on swizzle phase 0.
    struct SMemStorage
    {
        unsigned char W[NUM_STAGES][BN * W_ROW_BYTES];
        bf16 X[NUM_STAGES][BM * BK];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const bf16 *__restrict__ X,
        const unsigned char *__restrict__ W,
        bf16 *__restrict__ Y,
        const unsigned char *__restrict__ w_sf,
        float w_global,
        float *__restrict__ workspace = nullptr,
        cudaStream_t stream = nullptr);
};

// ue4m3 -> half. The sign bit of an NVFP4 scale is always 0.
__device__ __forceinline__ __half ue4m3_to_half(unsigned int b)
{
    return __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b & 0x7fu), __NV_E4M3));
}

// Two packed e2m1 -> one bf16x2 register, with the block scale folded in.
//
// The scale multiply happens in fp16, not fp32. That is safe rather than lucky:
// e2m1 tops out at 6 and ue4m3 at 448, so the product cannot exceed 2688,
// against fp16's 65504 -- and it needs four mantissa bits (one from e2m1, three
// from ue4m3) against fp16's ten, so it is exact. One hmul2 replaces the pair
// of fp32 multiplies, and the whole widen-scale-narrow chain is
// cvt / hmul2 / 2x cvt.f32.f16 / prmt.
//
// Worth being straight about the payoff: this shaved ~1% (0.0247 -> 0.0245 ms
// at M=8 N=4096 K=14336, and nothing measurable at N=28672). The mainloop is
// not instruction-bound -- see the note on where the time actually goes below.
// It is kept because it is the cheaper formulation and provably exact, not
// because it bought anything.
//
// The final pack truncates rather than rounds to nearest. bf16 is the top half
// of fp32, so prmt does it in one instruction; the half-ulp that costs is far
// below the 10% rms error the 4-bit quantisation already carries.
__device__ __forceinline__ uint32_t fp4x2_scaled_bf16x2(unsigned int byte, __half2 s)
{
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(byte & 0xffu), __NV_E2M1);
    float2 f = __half22float2(__hmul2(__half2(h), s));
    uint32_t lo = __float_as_uint(f.x), hi = __float_as_uint(f.y), r;
    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(r) : "r"(lo), "r"(hi));
    return r;
}

template <int BN, int NUM_STAGES, int CWG, int SPLIT_K>
__global__ void __launch_bounds__(W4A16GemmTinyMTC<BN, NUM_STAGES, CWG, SPLIT_K>::TOTAL_THREADS, 1, 1)
    w4a16_gemm_tinym_tc_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        int num_k_per_split,
        int sf_k_tiles,
        float w_global,
        __grid_constant__ const TMADescriptor tma_X,
        __grid_constant__ const TMADescriptor tma_W,
        const unsigned char *__restrict__ w_sf,
        bf16 *__restrict__ Y,
        float *__restrict__ workspace)
{
    using P = W4A16GemmTinyMTC<BN, NUM_STAGES, CWG, SPLIT_K>;
    using SmemStorage = typename P::SMemStorage;
    constexpr int BM = P::BM, BK = P::BK;

    extern __shared__ __align__(1024) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;

    const int num_blocks = gridDim.x;
    const int total_vtiles = total_tiles * SPLIT_K;

    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], P::CONSUMER_WARPS);
        }
    }
    __syncthreads();
    fence_proxy_async_shared();

    // ── Producer ───────────────────────────────────────────────────
    if (wg_id == 0)
    {
        if (warp_id == 0 && lane_id == 0)
        {
            int stage = 0, phase = 0, total_k = 0;
            for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
            {
                int tile_id = vtid / SPLIT_K, split_idx = vtid % SPLIT_K;
                int bm, bn;
                rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n,
                                        P::SWIZZLE_WIDTH, bm, bn);
                const int k_start = split_idx * num_k_per_split;
                const int k_end = k_start + num_k_per_split;
                for (int k = k_start; k < k_end; k++)
                {
                    if (total_k >= NUM_STAGES)
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);

                    const uint32_t mbar = smem_u32(&smem.full_barrier[stage]);
                    mbarrier_expect_tx(mbar, P::TX_BYTES);
                    tma_X.load_2d(k * BK, bm * BM, smem_u32(smem.X[stage]), mbar);
                    tma_W.load_2d(k * P::W_ROW_BYTES, bn * BN, smem_u32(smem.W[stage]), mbar);

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
        const int cwarp = warp_id - P::WARPS_PER_WG;   // 0 .. CONSUMER_WARPS-1
        const int warp_n = cwarp * P::N_PER_WARP;      // n offset inside the BN tile
        // B-fragment lane mapping: lane l owns n = base + (l>>2),
        // k = k_base + (l&3)*2 + {0,1} and the same +8.
        const int b_n = lane_id >> 2;
        const int b_kbyte = lane_id & 3;
        int stage = 0, phase = 0;

        for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
        {
            int tile_id = vtid / SPLIT_K, split_idx = vtid % SPLIT_K;
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n,
                                    P::SWIZZLE_WIDTH, bm, bn);
            const int k_start = split_idx * num_k_per_split;
            const int k_end = k_start + num_k_per_split;
            const int n_tile_base = bn * BN + warp_n;

            float acc[P::MMA_N][4]{};

            for (int k = k_start; k < k_end; k++)
            {
                // Scale packs depend on k alone, so issuing them before the
                // wait overlaps the stage still in flight.
                uint32_t wpack[P::MMA_N][P::SF_PACKS];
#pragma unroll
                for (int ni = 0; ni < P::MMA_N; ni++)
#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                        wpack[ni][j] = mx_scale_pack(w_sf, n_tile_base + ni * 8 + b_n,
                                                     k * P::SF_PACKS + j, sf_k_tiles);

                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const bf16 *sX = smem.X[stage];
                const unsigned char *sW = smem.W[stage];

#pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 16;

                    // A fragment: X is 16(m) x 16(k), four 8x8 b16 matrices.
                    uint32_t a[4];
                    {
                        int a_row = (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                        int a_col = k_base + (lane_id >> 4) * 8;
                        ldmatrix_x4(a, smem_u32(&sX[a_row * BK + a_col]));
                    }

#pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        // One MMA covers 16 k == one scale block, so one ue4m3
                        // serves both halves of this lane's fragment.
                        const __half2 s = __half2half2(ue4m3_to_half(
                            (wpack[ni][ki / 4] >> (8 * (ki % 4))) & 0xffu));

                        const int n_row = ni * 8 + b_n + warp_n;
                        const int c0 = k_base / 2 + b_kbyte;
                        uint32_t b[2];
                        b[0] = fp4x2_scaled_bf16x2(
                            sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(n_row, c0, P::W_ROW_BYTES)], s);
                        b[1] = fp4x2_scaled_bf16x2(
                            sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(n_row, c0 + 4, P::W_ROW_BYTES)], s);

                        mma_m16n8k16_bf16(acc[ni], a, b);
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
            // C layout of m16n8k16: lane l holds (m = l>>2, n = (l&3)*2+{0,1})
            // and the same at m + 8. Only NVFP4's fp32 global scale is left.
            const int c_m = lane_id >> 2;
            const int c_n = (lane_id & 3) * 2;
#pragma unroll
            for (int ni = 0; ni < P::MMA_N; ni++)
            {
                const int n = n_tile_base + ni * 8 + c_n;
#pragma unroll
                for (int h = 0; h < 2; h++)
                {
                    const int m = c_m + h * 8;
                    if (m >= M) continue;
#pragma unroll
                    for (int e = 0; e < 2; e++)
                    {
                        const float v = acc[ni][h * 2 + e] * w_global;
                        if constexpr (P::DIRECT_OUTPUT)
                            Y[(size_t)m * N + n + e] = __float2bfloat16(v);
                        else
                            workspace[(size_t)split_idx * M * N + (size_t)m * N + n + e] = v;
                    }
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
template <int BN, int NUM_STAGES, int CWG, int SPLIT_K>
void W4A16GemmTinyMTC<BN, NUM_STAGES, CWG, SPLIT_K>::run(
    int M, int N, int K,
    const bf16 *__restrict__ X,
    const unsigned char *__restrict__ W,
    bf16 *__restrict__ Y,
    const unsigned char *__restrict__ w_sf,
    float w_global,
    float *__restrict__ workspace,
    cudaStream_t stream)
{
    if (M > BM || N % BN != 0 || K % (BK * SPLIT_K) != 0)
        throw std::runtime_error(
            "W4A16 TinyM-TC: need M <= 16, N divisible by BN, K divisible by BK*SPLIT_K.");

    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2,
                                                 K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_NONE);
    TMADescriptor tma_W = create_tma_desc_2d_raw(W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                                 K / 2, N, W_ROW_BYTES, BN,
                                                 CU_TENSOR_MAP_SWIZZLE_128B);

    int num_tiles_m = 1;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;
    int num_k_per_split = (K / BK) / SPLIT_K;
    int sf_k_tiles = BlockScaleLayout(N, K, SF_VEC).k_tiles();

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = min(num_sm, total_tiles * SPLIT_K);

    auto kernel = w4a16_gemm_tinym_tc_kernel<BN, NUM_STAGES, CWG, SPLIT_K>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    kernel<<<dim3(num_blocks), dim3(TOTAL_THREADS), SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n, total_tiles, num_k_per_split, sf_k_tiles,
        w_global, tma_X, tma_W, w_sf, Y, workspace);
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
