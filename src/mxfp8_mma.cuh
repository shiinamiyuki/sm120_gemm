#pragma once
// The block-scaled m16n8k32 MMA, and where it expects the ue8m0 scales to
// live in the warp's registers.
//
// fp8_gemm.cuh already issues this instruction, but under per-tensor scaling
// every scale byte holds the same value, so it never had to know *which*
// byte of *which* thread feeds which row. MXFP8 does. This header pins that
// down; bench_mxfp8 --probe-mma re-derives it against the hardware.
#include "common.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ── The instruction ───────────────────────────────────────────────────
//
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
//       .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
//       d, a, b, c, sf_a, {byte-id-a, thread-id-a}, sf_b, {byte-id-b, thread-id-b}
//
// D[16x8] = (SFA * A[16x32]) x (SFB * B[32x8]) + C, accumulating in fp32 and
// running at the full tensor-core rate (SASS QMMA.SF.16832.F32.E4M3.E4M3.E8),
// twice the rate of the unscaled mma.m16n8k32 on sm_120.
//
// scale_vec::1X means one scale per row of A and per column of B, covering all
// K=32 -- which is exactly the MXFP8 block size, so one MMA consumes exactly
// one scale block per row/col. That is the whole reason MXFP8 costs nothing
// here: the 32-element block boundary and the MMA's k-extent coincide.
//
// ── Where the scales come from (probed, not documented) ────────────────
//
// sf_a and sf_b are ordinary .b32 registers held by all 32 lanes, but the
// hardware reads only a few of them. The immediate selectors narrow it down:
// byte-id in [0,3] picks the byte within the register, thread-id in [0,1]
// picks which half of each quad supplies it (ptxas rejects thread-id > 1).
//
// With (byte-id, thread-id) = (bid, tid):
//
//   A: row m (0..15)  <- lane 4*(m % 8) + 2*tid + (m / 8),  byte bid
//   B: col n (0..7)   <- lane 4*n       +   tid          ,  byte bid
//
// So for tid = 0 the A scales live in the lanes with (lane % 4) < 2: lane l
// holds the scale for row (l / 4) + 8 * (l % 2). The B scales live in the
// lanes with lane % 4 == 0, holding column l / 4. Every other lane's sf
// register is ignored.
//
// Note this is *not* the accumulator's row mapping. The accumulator gives
// lane l rows l>>2 and (l>>2)+8; the A scale for row m instead sits in a lane
// determined by m%8 and m/8. A warp that has just loaded a 16-row A fragment
// therefore has to shuffle or re-read to get each scale into the right lane --
// see the loader in the kernel, not here.
//
// Only 16 of 32 lanes carry A scales and only 8 carry B scales, and byte-id
// gives four independent slots per register, so one b32 per lane can hold the
// scales for four consecutive MMAs along k (BK = 128 = 4 * 32). That is the
// natural packing: load four k-blocks' worth of scales once, then issue four
// MMAs stepping byte-id 0,1,2,3.

#define MXFP8_MMA_BLOCK_SCALED(BID_A, TID_A, BID_B, TID_B)                       \
    asm volatile(                                                                \
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"              \
        ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "                             \
        "{%0, %1, %2, %3}, "                                                     \
        "{%4, %5, %6, %7}, "                                                     \
        "{%8, %9}, "                                                             \
        "{%0, %1, %2, %3}, "                                                     \
        "%10, {" #BID_A ", " #TID_A "}, %11, {" #BID_B ", " #TID_B "};\n"        \
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                         \
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                            \
          "r"(b[0]), "r"(b[1]), "r"(sf_a), "r"(sf_b))

// One MMA reading scale byte BYTE of every lane's sf register (thread-id 0).
// BYTE selects which of the four packed k-blocks this MMA belongs to.
template <int BYTE>
__device__ __forceinline__ void mma_m16n8k32_e4m3_block_scaled(float (&d)[4],
                                                               const uint32_t (&a)[4],
                                                               const uint32_t (&b)[2],
                                                               uint32_t sf_a, uint32_t sf_b);

#define MXFP8_DEFINE_MMA(BYTE)                                                   \
    template <>                                                                  \
    __device__ __forceinline__ void mma_m16n8k32_e4m3_block_scaled<BYTE>(        \
        float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],           \
        uint32_t sf_a, uint32_t sf_b) {                                          \
        MXFP8_MMA_BLOCK_SCALED(BYTE, 0, BYTE, 0);                                \
    }
MXFP8_DEFINE_MMA(0)
MXFP8_DEFINE_MMA(1)
MXFP8_DEFINE_MMA(2)
MXFP8_DEFINE_MMA(3)
#undef MXFP8_DEFINE_MMA

// Dispatch on a byte-id carried as a loop index. When the k-loop is unrolled
// the index is a constant and the switch folds away entirely.
__device__ __forceinline__ void mma_m16n8k32_e4m3_block_scaled(int byte, float (&d)[4],
                                                               const uint32_t (&a)[4],
                                                               const uint32_t (&b)[2],
                                                               uint32_t sf_a, uint32_t sf_b) {
    switch (byte & 3) {
        case 0: mma_m16n8k32_e4m3_block_scaled<0>(d, a, b, sf_a, sf_b); break;
        case 1: mma_m16n8k32_e4m3_block_scaled<1>(d, a, b, sf_a, sf_b); break;
        case 2: mma_m16n8k32_e4m3_block_scaled<2>(d, a, b, sf_a, sf_b); break;
        default: mma_m16n8k32_e4m3_block_scaled<3>(d, a, b, sf_a, sf_b); break;
    }
}

// Which lane must hold the scale, at thread-id 0. Inverses of each other.
__host__ __device__ __forceinline__ int mx_sfa_lane_for_row(int m) { return 4 * (m % 8) + (m / 8); }
__host__ __device__ __forceinline__ int mx_sfb_lane_for_col(int n) { return 4 * n; }
// -1 when this lane carries no scale.
__host__ __device__ __forceinline__ int mx_sfa_row_for_lane(int lane) {
    return (lane % 4) < 2 ? (lane / 4) + 8 * (lane % 2) : -1;
}
__host__ __device__ __forceinline__ int mx_sfb_col_for_lane(int lane) {
    return (lane % 4) == 0 ? lane / 4 : -1;
}
