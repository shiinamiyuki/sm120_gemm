#pragma once
// The NVFP4 block-scaled m16n8k64 MMA, and where it expects its operands and
// its ue4m3 scales to live in the warp's registers.
//
// This is the W4A4 counterpart of mxfp8_mma.cuh. Everything below was probed
// against the hardware, not read out of a manual -- see the note at the end.
#include "common.h"
#include "block_scale.h" // ue4m3_to_float, kE2m1Max / kUe4m3Max

#include <cuda_fp8.h>
#include <cstdint>

// ── The instruction ───────────────────────────────────────────────────
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//       d, a, b, c, sf_a, {0, thread-id-a}, sf_b, {0, thread-id-b}
//
// D[16x8] = (SFA * A[16x64]) x (SFB * B[64x8]) + C, fp32 accumulate, with the
// tensor core consuming packed e2m1 directly. For W4A4 there is no dequantise
// step at all -- which is the whole reason this is a different kernel from
// w4a16_gemm_tinym_tc.cuh, where the weights had to be widened to bf16 because
// no mixed 4-bit x 16-bit MMA exists.
//
// It is available only on sm_120a, and only if nvcc is told properly:
// `-arch=sm_120a` silently forwards `-arch=compute_120` to ptxas and the
// instruction comes back "not supported on .target 'sm_120'". Use
// `-gencode arch=compute_120a,code=sm_120a`, which is what kernel_jit.h does.
//
// scale_vec::4X means four scales per row of A and per column of B, one per 16
// elements of k -- exactly NVFP4's block size, and exactly four of them across
// this instruction's k=64. So one MMA consumes one whole b32 of scales per
// operand, and byte-id is therefore pinned to 0 (ptxas rejects 1..3). That is
// the opposite of the MXFP8 1X form, where one b32 fed four consecutive MMAs.
//
// ── Operand layout (probed) ───────────────────────────────────────────
//
// A is 16x64 packed e2m1 = 512 bytes = 4 x b32 per lane; B is 64x8 = 2 x b32.
// Within a byte the low nibble is the earlier (even) k, matching the packing
// bench_w4a16 --probe-layout confirms for the weight tensor.
//
// With g = lane >> 2 and q = lane & 3:
//
//   a[0] = A[g    ][q*8 + 0..7]      b[0] = B[q*8 + 0..7 ][g]
//   a[1] = A[g + 8][q*8 + 0..7]      b[1] = B[32 + q*8 + 0..7][g]
//   a[2] = A[g    ][32 + q*8 + 0..7]
//   a[3] = A[g + 8][32 + q*8 + 0..7]
//
// i.e. the m16n8k32 e4m3 layout with the k-extent doubled, since a nibble is
// half a byte. The accumulator is the usual one: lane l holds (m = l>>2,
// n = (l&3)*2 + {0,1}) and the same at m + 8.
//
// ── Where the scales come from (probed) ───────────────────────────────
//
// sf_a and sf_b are .b32 registers held by all 32 lanes; the hardware reads
// only some of them, selected by the thread-id immediate (0 or 1):
//
//   A: row m (0..15) <- lane 4*(m % 8) + 2*tid + (m / 8)
//   B: col n (0..7)  <- lane 4*n       +   tid
//
// which is the same mapping as the MXFP8 1X form. Byte j of the register is
// the scale for k-block j, i.e. k in [16j, 16j+16).
//
// Note this is *not* the accumulator's row mapping: the accumulator gives lane
// l rows l>>2 and (l>>2)+8, while row m's scale sits in a lane chosen by m%8
// and m/8. A warp holding a 16-row A fragment must shuffle or re-read to get
// each scale into the right lane.
//
// ── How this was established ──────────────────────────────────────────
//
// `bench_w4a4 --probe-mma`, in two halves.
//
// The operand half fills a reference A[16][64], B[64][8] and per-16 scale grid
// with values drawn from the exact e2m1 / ue4m3 grids, packs them into
// registers per the layout above, issues one MMA, and compares all 128
// accumulator elements against an fp64 reference. That is what pins the nibble
// order and the accumulator mapping.
//
// The scale half sweeps lane x byte x k-block x side: A is all 1.0 and B is
// 1.0 in exactly one k-block, so D = 16 everywhere, and one scale byte in the
// warp is raised to 2.0. The entries that come out 32 name the row (or column)
// that byte feeds, and they only do so when the byte index matches the live
// k-block -- which is how the byte -> k-block meaning is established. A probe
// with uniform B cannot see that, because every k-block contributes equally.
//
// Both halves pass for both thread-id settings: 0/128 and 0 wrong.

// One MMA. TID selects which half of each lane quad supplies the scales.
template <int TID>
__device__ __forceinline__ void mma_m16n8k64_e2m1_block_scaled(float (&d)[4],
                                                               const uint32_t (&a)[4],
                                                               const uint32_t (&b)[2],
                                                               uint32_t sf_a, uint32_t sf_b);

#define NVFP4_DEFINE_MMA(TID)                                                    \
    template <>                                                                  \
    __device__ __forceinline__ void mma_m16n8k64_e2m1_block_scaled<TID>(         \
        float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],           \
        uint32_t sf_a, uint32_t sf_b) {                                          \
        asm volatile(                                                            \
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"          \
            ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                         \
            "{%0, %1, %2, %3}, "                                                 \
            "{%4, %5, %6, %7}, "                                                 \
            "{%8, %9}, "                                                         \
            "{%0, %1, %2, %3}, "                                                 \
            "%10, {0, " #TID "}, %11, {0, " #TID "};\n"                          \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                     \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                        \
              "r"(b[0]), "r"(b[1]), "r"(sf_a), "r"(sf_b));                       \
    }
NVFP4_DEFINE_MMA(0)
NVFP4_DEFINE_MMA(1)
#undef NVFP4_DEFINE_MMA

// Which lane must hold the scale register, at thread-id 0, and the inverses.
// Identical to the MXFP8 1X mapping; only what a byte *means* differs (there,
// four consecutive MMAs; here, four k-blocks of one MMA).
__host__ __device__ __forceinline__ int nvfp4_sfa_lane_for_row(int m) { return 4 * (m % 8) + (m / 8); }
__host__ __device__ __forceinline__ int nvfp4_sfb_lane_for_col(int n) { return 4 * n; }
// The inverses. TID must match the immediate given to the MMA; -1 means this
// lane carries no scale for that operand. Written as a search rather than
// closed form on purpose: the closed form has to be evaluated at lane - 2*TID,
// which goes negative for the low lanes at TID=1, and C++ modulo on a negative
// silently yields a plausible-looking wrong row.
__host__ __device__ __forceinline__ int nvfp4_sfa_row_for_lane(int lane, int tid = 0) {
    for (int m = 0; m < 16; m++)
        if (nvfp4_sfa_lane_for_row(m) + 2 * tid == lane) return m;
    return -1;
}
__host__ __device__ __forceinline__ int nvfp4_sfb_col_for_lane(int lane, int tid = 0) {
    for (int n = 0; n < 8; n++)
        if (nvfp4_sfb_lane_for_col(n) + tid == lane) return n;
    return -1;
}

// ── Host-side value grid, for probes and quantisation ─────────────────
// e2m1: 1 sign, 2 exponent, 1 mantissa bit. (ue4m3_to_float already lives in
// block_scale.h; do not add a second overload of it here -- one taking
// `unsigned` alongside one taking `unsigned char` makes every int-typed call
// ambiguous.)
__host__ __device__ __forceinline__ float e2m1_to_float(int code) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = mag[code & 7];
    return (code & 8) ? -v : v;
}
// Handy constants: a b32 of four packed e2m1 1.0s, and of four ue4m3 1.0s.
static constexpr uint32_t kE2M1_Ones = 0x22222222u;
static constexpr uint32_t kUE4M3_Ones = 0x38383838u;
