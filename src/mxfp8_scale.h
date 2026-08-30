#pragma once
// MXFP8 scale-factor format: the ue8m0 encoding and the swizzled tensor
// layout, shared by the cuBLASLt baseline, the harness and the kernels.
// Deliberately free of any cuBLASLt dependency so JIT-compiled kernels can
// include it.
#include <cmath>
#include <cstddef>

// ue8m0: byte b encodes 2^(b-127). b = 0 is 2^-127, b = 255 is NaN.
static constexpr unsigned char kUe8m0One = 127;

__host__ __device__ __forceinline__ float ue8m0_to_float(unsigned char b) {
    return ldexpf(1.0f, (int)b - 127);
}

// Device-side decode, one shift instead of a MUFU: an fp32 with exponent
// field b and zero mantissa *is* 2^(b-127) for b in [1, 254]. b = 0 would be
// 2^-127, which is subnormal in fp32, and comes out as 0 here; that is eight
// orders of magnitude below anything e4m3 data can contribute, and our
// quantizer never emits it.
__device__ __forceinline__ float ue8m0_to_float_fast(unsigned int b) {
    return __int_as_float((int)(b << 23));
}

// ── Swizzled ue8m0 scale-factor tensor for one operand ─────────────────
//
//   rows  = the operand's non-K extent (N for A = W, M for B = X)
//   sk    = K / 32, the number of scale blocks along K
//
// Rows are padded to 128 and scale-columns to 4. The unit is a 128 x 4 tile
// stored as 512 contiguous bytes; tiles run k-block-fastest within a row
// block, then by row block:
//
//   offset(row, kb) = ((row/128) * ceil(sk/4) + kb/4) * 512
//                   + (row % 32) * 16          // lane within a 32-row group
//                   + ((row % 128) / 32) * 4   // which of the 4 row groups
//                   + (kb % 4)                 // which of the 4 k-blocks
//
// Reverse-engineered from cuBLASLt by bit-plane probing; re-derive it with
// `bench_mxfp8 --probe-layout`.
//
// Two properties make this layout pleasant for a kernel rather than merely
// tolerable:
//
//   1. The four scales for k-blocks 4t..4t+3 of one row are four *contiguous*
//      bytes, so a thread picks up a whole BK=128 stage's worth of scaling
//      for its row with one aligned 32-bit load (see mx_scale_pack).
//   2. Those loads across 128 consecutive rows cover one 512-byte tile
//      exactly once, so a consumer warp group reads full cache lines with no
//      waste, even though consecutive threads are 16 bytes apart.
struct MxScaleLayout {
    static constexpr int kVec = 32; // elements per scale block

    int rows = 0, sk = 0;

    MxScaleLayout() = default;
    MxScaleLayout(int rows_, int K) : rows(rows_), sk(K / kVec) {}

    static __host__ __device__ __forceinline__ int ceil_div(int a, int b) {
        return (a + b - 1) / b;
    }

    __host__ __device__ __forceinline__ int row_tiles() const { return ceil_div(rows, 128); }
    __host__ __device__ __forceinline__ int k_tiles() const { return ceil_div(sk, 4); }

    __host__ __device__ __forceinline__ size_t offset(int row, int kb) const {
        return ((size_t)(row / 128) * k_tiles() + (kb / 4)) * 512
               + (size_t)(row % 32) * 16
               + ((row % 128) / 32) * 4
               + (kb % 4);
    }

    __host__ __device__ __forceinline__ size_t bytes() const {
        return (size_t)row_tiles() * k_tiles() * 512;
    }
};

// Byte offset of a row's 4-byte pack inside the 512-byte tile that holds it.
__host__ __device__ __forceinline__ int mx_scale_tile_offset(int row) {
    return (row & 31) * 16 + ((row & 127) >> 5) * 4;
}

// Byte offset of the 512-byte tile holding rows [128*rb, 128*rb+128) for
// k-blocks 4*ktile .. 4*ktile+3. k_tiles is MxScaleLayout::k_tiles().
__host__ __device__ __forceinline__ size_t mx_scale_tile_bytes(int rb, int ktile, int k_tiles) {
    return ((size_t)rb * k_tiles + ktile) * 512;
}

// The four ue8m0 codes for `row` covering k-blocks 4*ktile .. 4*ktile+3,
// packed little-endian: byte b is k-block 4*ktile + b. One aligned 32-bit
// load, by property (1) above.
__device__ __forceinline__ unsigned int mx_scale_pack(const unsigned char *sf, int row,
                                                      int ktile, int k_tiles) {
    return *reinterpret_cast<const unsigned int *>(
        sf + mx_scale_tile_bytes(row >> 7, ktile, k_tiles) + mx_scale_tile_offset(row));
}

// 1D bulk copy global -> shared, completing on an mbarrier's transaction
// count. Unlike cp.async.bulk.tensor this takes a plain contiguous byte range,
// which is exactly what a 512-byte scale tile is. `bytes` must be a multiple
// of 16 and both addresses 16-byte aligned.
__device__ __forceinline__ void cp_async_bulk_g2s(unsigned int smem_addr, const void *gmem,
                                                  unsigned int bytes, unsigned int mbar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"(smem_addr), "l"(gmem), "r"(bytes), "r"(mbar)
        : "memory");
}
