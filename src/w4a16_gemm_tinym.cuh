#pragma once
#include "bf16_gemm.cuh" // TMA / mbarrier / swizzle helpers, splitk_reduce_kernel
#include "block_scale.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ════════════════════════════════════════════════════════════════════════
// W4A16GemmTinyM — NVFP4 weights x bf16 activations, M <= BM, on CUDA cores
// ════════════════════════════════════════════════════════════════════════
//
// Neither cuBLASLt nor the tensor cores do W4A16: `mma` rejects e2m1 x bf16,
// and a sweep of 3072 cuBLASLt type combinations finds no mixed-width matmul
// at all (bench_w4a16 --probe-support). On tensor cores W4A16 therefore means
// dequantizing weights to bf16 in registers first. On CUDA cores that cost
// disappears: this kernel already widens both operands to fp32 to feed FMAs,
// so a 4-bit weight is just a different widening instruction.
//
// That argument holds only at M = 1. This kernel spends BM FMAs per weight
// element, so its cost grows with M while the weight stream does not -- and
// past M = 4 the dequantize-and-MMA kernel in w4a16_gemm_tinym_tc.cuh wins,
// because a tensor core eats the whole m16 tile whether one row is live or
// sixteen. Use this family at M <= 2 and that one above it; the autotuner
// enumerates both and picks per shape.
//
// Decode is where W4A16 earns its keep, and it is pure bandwidth there: the
// job is to stream W once. NVFP4 makes W a quarter of bf16 plus 1/16th of a
// byte per element of ue4m3 scale, so the ceiling is ~3.7x the bf16 GEMM.
//
// ── Layout choices ────────────────────────────────────────────────────
//
// BK is pinned to 256, and that is forced by W. Packed e2m1 is half a byte,
// so a BK-wide row of W is BK/2 bytes, and the 128B TMA swizzle -- the thing
// that makes the per-n smem reads conflict-free -- needs a 128-byte row.
// BK = 256 is the smallest BK that gives one.
//
// X does not need a swizzle at all. Every thread reads the same X (it is the
// M <= 8 side), so those reads are broadcasts and cannot conflict whatever
// the layout. Loading X with SWIZZLE_NONE keeps its smem addressing to plain
// row-major and sidesteps the question of what the 128B swizzle does to a
// 512-byte row, which nothing else in this repo exercises.
//
// ── Why the inner loop is chunked at 32, not 16 ───────────────────────
//
// The scale block is 16 elements, which is the natural unit. But 16 packed
// e2m1 is only 8 bytes, and an 8-byte-per-thread smem read is serviced in two
// phases of 16 threads; with the 128B swizzle those 16 threads cover only 8
// distinct bank pairs, so it is a 2-way conflict. 32 elements is a full
// 16-byte read, which the hardware splits into four phases of 8 threads, and
// 8 threads x 16 bytes covers all 32 banks exactly once -- conflict-free, the
// same property the fp8 kernels rely on.
//
// So W is read 32 elements at a time and the two scale blocks inside are
// handled separately.
//
// ── Why X is widened once per CTA, not per thread ─────────────────────
//
// Every consumer thread reads the same X -- it is the M <= 8 side -- so
// widening bf16 -> fp32 inside the inner loop had every thread doing the same
// BM*BK conversions. At NPT == 1 (BN=256, CWG=2) that cost about as many
// instructions as the FMAs they fed, and the float xf[BM][SF_VEC] fragment it
// needed was 128 registers at BM=8, enough to push the wide configs into
// spilling. X is now widened cooperatively into smem once per stage --
// BM*BK/CONSUMER_THREADS elements per thread -- behind two named barriers.
// Worth 1.21x at M=8 and it is what makes CWG=1 (NPT=2) competitive at all
// there: 606 -> 892 GB/s.

// ── Widening ───────────────────────────────────────────────────────────

// 16 bytes of shared memory = 32 packed e2m1, widened to fp32. The low nibble
// of each byte is the even element (probed against cuBLASLt).
__device__ __forceinline__ void load_fp4x32_f32(const unsigned char *src, float (&out)[32])
{
    uint4 raw = *reinterpret_cast<const uint4 *>(src);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(
                (__nv_fp4x2_storage_t)((w[i] >> (8 * j)) & 0xffu), __NV_E2M1);
            float2 f = __half22float2(__half2(h));
            out[8 * i + 2 * j] = f.x;
            out[8 * i + 2 * j + 1] = f.y;
        }
}

// ue4m3 -> fp32 via the e4m3 hardware conversion (the sign bit is always 0).
__device__ __forceinline__ float ue4m3_to_float_fast(unsigned int b)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b & 0x7fu), __NV_E4M3);
    return __half2float(__half(h));
}

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K = 1>
struct W4A16GemmTinyM
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;
    static constexpr int CONSUMER_THREADS = CWG * THREADS_PER_WG;

    static constexpr int W_ROW_BYTES = BK / 2;                 // packed e2m1
    static constexpr int TX_BYTES = BM * BK * (int)sizeof(bf16) + BN * W_ROW_BYTES;

    static constexpr int NPT = BN / CONSUMER_THREADS;
    static constexpr int SF_VEC = 16;                          // elements per ue4m3
    static constexpr int KB_PER_STAGE = BK / SF_VEC;           // scale blocks per stage
    static constexpr int SF_PACKS = KB_PER_STAGE / 4;          // 4 blocks per 32-bit pack
    static constexpr int KCHUNK = 32;                          // elements per W read

    static constexpr bool DIRECT_OUTPUT = (SPLIT_K == 1);

    static_assert(BN % CONSUMER_THREADS == 0,
                  "BN must be a multiple of the consumer thread count");
    static_assert(NPT >= 1, "BN must be >= CONSUMER_THREADS (raise BN or lower CWG)");
    static_assert(BK == 256,
                  "packed e2m1 needs BK/2 == 128 bytes per row for the 128B swizzle");
    static_assert(BN % 8 == 0, "BN must be a multiple of 8 rows");
    static_assert(BM <= 128, "one scale-tensor row block covers 128 rows");
    static_assert(BK / 2 <= 256 && BN <= 256, "TMA box dims are capped at 256");

    static constexpr int SWIZZLE_BYTES = 128;
    static constexpr int SWIZZLE_WIDTH = 4;

    // W must come FIRST. smem_raw is __align__(1024) and the 128B TMA swizzle
    // takes its XOR from the absolute smem row index within a 1024-byte block,
    // so a W array that does not start on a 1024-byte boundary gives every
    // stage a different swizzle phase and swizzle_smem_offset() (which assumes
    // phase 0) silently reads the wrong bytes. A W stage is BN * 128 bytes,
    // itself a multiple of 1024, so the stages stay in phase with each other.
    // X is unswizzled and only needs 16-byte alignment, which it gets for free
    // sitting after W.
    struct SMemStorage
    {
        unsigned char W[NUM_STAGES][BN * W_ROW_BYTES];
        bf16 X[NUM_STAGES][BM * BK];
        // X widened to fp32, once per stage for the whole CTA. Only one buffer
        // is needed rather than one per stage: every consumer warp waits on the
        // same full_barrier[stage], so they are all inside the same stage at
        // once, and the two named barriers around its use keep the next stage's
        // conversion from overtaking a reader.
        //
        // Kept in [m][k], matching X. Transposing to [k][m] to make the BM
        // values for one k contiguous was tried and is 5% slower: the FMA loop
        // is fully unrolled, so ptxas already vectorises the [m][k] reads along
        // e for each m, and the transpose only buys conflicted strided reads in
        // the conversion above.
        float Xf[BM * BK];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    // w_sf is the swizzled ue4m3 tensor laid out by BlockScaleLayout(N, K, 16);
    // w_global is NVFP4's second-level fp32 scale.
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

template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int SPLIT_K>
__global__ void __launch_bounds__(W4A16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::TOTAL_THREADS, 1, 1)
    w4a16_gemm_tinym_kernel(
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
    using P = W4A16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
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

    // ── Producer ───────────────────────────────────────────────────
    if (wg_id == 0)
    {
        if (warp_in_wg == 0 && lane_id == 0)
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
                    // W's global inner extent is K/2 bytes, so the k offset halves.
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
        const int tid_c = tid - P::THREADS_PER_WG;
        int stage = 0, phase = 0;

        for (int vtid = blockIdx.x; vtid < total_vtiles; vtid += num_blocks)
        {
            int tile_id = vtid / SPLIT_K, split_idx = vtid % SPLIT_K;
            int bm, bn;
            rasterize_tile_swizzled(tile_id, num_tiles_m, num_tiles_n,
                                    P::SWIZZLE_WIDTH, bm, bn);
            const int k_start = split_idx * num_k_per_split;
            const int k_end = k_start + num_k_per_split;

            const int n_tile_base = bn * BN;
            int sf_row[P::NPT];
#pragma unroll
            for (int p = 0; p < P::NPT; p++)
                sf_row[p] = n_tile_base + p * P::CONSUMER_THREADS + tid_c;

            float acc[P::NPT][BM]{};

            for (int k = k_start; k < k_end; k++)
            {
                // Depends on k alone, so issuing it before the wait overlaps
                // the stage still in flight.
                uint32_t wpack[P::NPT][P::SF_PACKS];
#pragma unroll
                for (int p = 0; p < P::NPT; p++)
#pragma unroll
                    for (int j = 0; j < P::SF_PACKS; j++)
                        wpack[p][j] = mx_scale_pack(w_sf, sf_row[p],
                                                    k * P::SF_PACKS + j, sf_k_tiles);

                __syncwarp();
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                // Widen X once for the whole CTA. Every consumer thread reads
                // the same X -- it is the M <= 8 side -- so widening it per
                // thread was pure redundancy, and at NPT == 1 it cost about as
                // many instructions as the FMAs it fed (BM*BK of them per
                // thread per stage, against BM*NPT*BK FMAs). Done once, it is
                // BM*BK/CONSUMER_THREADS per thread instead. It also retires
                // the float xf[BM][SF_VEC] register array, 128 registers at
                // BM=8, which is what pushed the wide configs into spilling.
                {
                    const bf16 *sXb = smem.X[stage];
#pragma unroll 4
                    for (int i = tid_c; i < BM * BK; i += P::CONSUMER_THREADS)
                        smem.Xf[i] = __bfloat162float(sXb[i]);
                }
                named_barrier_sync(1, P::CONSUMER_THREADS);

                const unsigned char *sW = smem.W[stage];

#pragma unroll
                for (int chunk = 0; chunk < BK / P::KCHUNK; chunk++)
                {
                    const int kk = chunk * P::KCHUNK;

                    // One 16-byte read per n-column: 32 packed e2m1,
                    // conflict-free (see the header note on chunking at 32).
                    float wf[P::NPT][P::KCHUNK];
#pragma unroll
                    for (int p = 0; p < P::NPT; p++)
                    {
                        const int n = p * P::CONSUMER_THREADS + tid_c;
                        load_fp4x32_f32(
                            &sW[swizzle_smem_offset<P::SWIZZLE_BYTES, 1>(n, kk / 2, P::W_ROW_BYTES)],
                            wf[p]);
                    }

#pragma unroll
                    for (int half = 0; half < P::KCHUNK / P::SF_VEC; half++)
                    {
                        // X is already fp32 in smem, so this is a plain read,
                        // uniform across the warp (broadcast) and free of the
                        // per-thread widening that used to sit here.
                        const float *xk = &smem.Xf[kk + half * P::SF_VEC];

#pragma unroll
                        for (int p = 0; p < P::NPT; p++)
                        {
                            const int kb = chunk * (P::KCHUNK / P::SF_VEC) + half;
                            const float s = ue4m3_to_float_fast(
                                (wpack[p][kb / 4] >> (8 * (kb % 4))) & 0xffu);

                            // Unscaled products for this one scale block, then
                            // one FMA per row to fold the ue4m3 in: BM extra
                            // operations per 16 elements of k, not 16.
                            float pacc[BM]{};
#pragma unroll
                            for (int e = 0; e < P::SF_VEC; e++)
#pragma unroll
                                for (int m = 0; m < BM; m++)
                                    pacc[m] = fmaf(xk[m * BK + e],
                                                   wf[p][half * P::SF_VEC + e], pacc[m]);
#pragma unroll
                            for (int m = 0; m < BM; m++)
                                acc[p][m] = fmaf(pacc[m], s, acc[p][m]);
                        }
                    }
                }

                // Everyone is done reading Xf, so the next stage may overwrite it.
                named_barrier_sync(1, P::CONSUMER_THREADS);
                if (lane_id == 0)
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                __syncwarp();

                stage++;
                if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
            }

            // ── Epilogue ───────────────────────────────────────────
            // Only NVFP4's second-level fp32 scale is left; the per-16 ue4m3
            // went in above, and bf16 activations carry no scale at all.
#pragma unroll
            for (int p = 0; p < P::NPT; p++)
            {
                const int n = n_tile_base + p * P::CONSUMER_THREADS + tid_c;
#pragma unroll
                for (int m = 0; m < BM; m++)
                {
                    if (m >= M) break;
                    if constexpr (P::DIRECT_OUTPUT)
                        Y[(size_t)m * N + n] = __float2bfloat16(acc[p][m] * w_global);
                    else
                        workspace[(size_t)split_idx * M * N + (size_t)m * N + n] =
                            acc[p][m] * w_global;
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
void W4A16GemmTinyM<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>::run(
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
            "W4A16 TinyM: need M <= BM, N divisible by BN, K divisible by BK*SPLIT_K.");

    // X: bf16, unswizzled. Rows >= M are out of bounds and TMA zero-fills them.
    TMADescriptor tma_X = create_tma_desc_2d_raw(X, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2,
                                                 K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_NONE);
    // W: packed e2m1 seen as bytes, so the inner extent is K/2.
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

    auto kernel = w4a16_gemm_tinym_kernel<BM, BN, BK, NUM_STAGES, CWG, SPLIT_K>;
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
