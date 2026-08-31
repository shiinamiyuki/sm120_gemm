#pragma once
#include <string>
#include <string_view>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// Function pointer types exported by each JIT-compiled kernel .so, one per
// element type. Both use void* for the matrix pointers to avoid C++ mangling
// issues in extern "C"; the fp8 form additionally carries the per-tensor
// dequantization scales.
using GemmKernelFn = void (*)(int M, int N, int K,
                              const void *X, const void *W, void *Y,
                              float *workspace, cudaStream_t stream);
using Fp8GemmKernelFn = void (*)(int M, int N, int K,
                                 const void *X, const void *W, void *Y,
                                 float x_scale, float w_scale,
                                 float *workspace, cudaStream_t stream);
// MXFP8 carries block-scale *tensors* instead of two scalars; they are the
// swizzled ue8m0 layout described in block_scale.h.
using MxFp8GemmKernelFn = void (*)(int M, int N, int K,
                                   const void *X, const void *W, void *Y,
                                   const void *x_sf, const void *w_sf,
                                   float *workspace, cudaStream_t stream);
// W4A16: bf16 activations, NVFP4 weights (packed e2m1 + swizzled ue4m3 per 16
// elements + one fp32 global scale). Only the weights carry a scale.
using W4A16GemmKernelFn = void (*)(int M, int N, int K,
                                   const void *X, const void *W, void *Y,
                                   const void *w_sf, float w_global,
                                   float *workspace, cudaStream_t stream);
// W4A4: both operands NVFP4, so two swizzled ue4m3 tensors, and the two fp32
// globals arrive pre-multiplied as one out_scale (the MMA has already taken
// both per-16 block scales).
using W4A4GemmKernelFn = void (*)(int M, int N, int K,
                                  const void *X, const void *W, void *Y,
                                  const void *x_sf, const void *w_sf, float out_scale,
                                  float *workspace, cudaStream_t stream);

// Which kernel template a config instantiates. Both share the producer /
// TMA / mbarrier pipeline and differ only in the consumer.
enum class KernelFamily {
    Mma,     // {BF16,FP8}GemmMMA   — tensor cores, requires M % BM == 0
    TinyM,   // {BF16,FP8}GemmTinyM — CUDA cores, for M <= BM (skinny GEMM)
    TinyMTC, // W4A16GemmTinyMTC    — tensor cores at decode M, BM fixed at 16.
             // Same skinny-GEMM shape as TinyM, but the mainloop is
             // mma.m16n8k16 rather than per-row FMAs, so its cost stops
             // growing with M. W4A16 only.
};

// Operand element type. The two share every kernel template; what changes is
// the element size (and everything derived from it: the smallest BK that fills
// a 128-byte swizzled smem row, the MMA's k extent, the vector-load width).
enum class ElemType {
    Bf16,
    Fp8,   // e4m3, one scale for the whole tensor
    MxFp8, // e4m3, one ue8m0 scale per 32 elements along K (OCP microscaling)
    W4A16, // bf16 activations x NVFP4 weights (e2m1 + ue4m3 per 16 + fp32)
    W4A4,  // NVFP4 on both sides; sm_120a has a native block-scaled e2m1 MMA
};

// ── GEMM configuration ─────────────────────────────────────────────────
//
// Canonical string form (also used as the .so filename):
//
//     128x128x64_s3_cwg2_w32x32_sk1
//      |          |   |    |      |
//      |          |   |    |      +-- SPLIT_K
//      |          |   |    +--------- WARP_M x WARP_N
//      |          |   +-------------- consumer warp groups (4 warps each)
//      |          +------------------ NUM_STAGES
//      +----------------------------- BM x BN x BK
//
// The tiny-M family carries a leading "tinym" token and no warp tile, since
// its consumer has no MMA fragments to lay out:
//
//     tinym_8x256x64_s2_cwg1_sk8
//
// "tinymtc" is the tensor-core skinny family (W4A16 only). Same shape, but the
// mainloop is mma.m16n8k16, so BM is pinned at 16 and the warp tile is again
// implicit -- each consumer warp takes BN/(CWG*4) columns:
//
//     w4a16_tinymtc_16x128x256_s3_cwg1_sk8
//
// FP8 configs carry a leading "fp8" token, before the family token:
//
//     fp8_128x128x128_s2_cwg2_w64x32_sk1
//     fp8_tinym_4x256x128_s2_cwg1_sk16
//
// parse() accepts the tokens in any order and fills in defaults, so partial
// specs work on the command line: "128x128", "128x128x64_sk2", "s4_cwg1_128x64",
// "tinym_8x256", "fp8_128x128x128".
struct GemmConfig {
    int bm = 0, bn = 0, bk = 64;
    int stages = 3, cwg = 2;
    int warp_m = 0, warp_n = 0; // 0 = derive from bm/bn/cwg (Mma only)
    int split_k = 1;
    // Last members so the positional aggregate initialisers below keep working
    // -- enumerate_mma builds a GemmConfig as {bm, bn, bk, stages, cwg, wm, wn,
    // sk, family, elem}, so anything inserted above this line silently shifts
    // what those braces mean. Add new fields at the end.
    KernelFamily family = KernelFamily::Mma;
    ElemType elem = ElemType::Bf16;
    // W4A4 only: how many column slices the epilogue stores in. Packed 4-bit
    // operands make a full BM x BN bf16 Y_out the largest thing in shared
    // memory, so slicing it is what lets BM=BN=128 fit at all.
    int y_slices = 1;

    bool is_tiny_m() const { return family == KernelFamily::TinyM; }
    bool is_tiny_m_tc() const { return family == KernelFamily::TinyMTC; }
    // Both skinny families take M <= BM and carry no warp tile in their name.
    bool is_skinny() const { return is_tiny_m() || is_tiny_m_tc(); }
    // Both fp8 flavours store e4m3, so everything the element size implies —
    // BK, the MMA k-extent, the vector width — is shared. Only where the
    // scale comes from differs.
    bool is_fp8() const { return elem == ElemType::Fp8 || elem == ElemType::MxFp8; }
    bool is_mx() const { return elem == ElemType::MxFp8; }
    // The two operands have different widths here, so elem_bytes() below is
    // the activation side; anything that needs the weight side asks directly.
    bool is_w4a16() const { return elem == ElemType::W4A16; }
    bool is_w4a4() const { return elem == ElemType::W4A4; }
    // Anything storing packed e2m1 has half-byte elements, so a BK-wide row is
    // BK/2 bytes rather than BK*elem_bytes().
    bool is_packed4() const { return is_w4a16() || is_w4a4(); }
    int elem_bytes() const { return is_fp8() ? 1 : 2; }
    // k covered by one MMA instruction: m16n8k16 for bf16, m16n8k32 for e4m3.
    int mma_k() const { return is_fp8() ? 32 : 16; }
    // Elements in one 16-byte smem vector load (tiny-M).
    int vec() const { return 16 / elem_bytes(); }

    std::string name() const {
        char buf[128];
        const char *dt = is_w4a4()    ? "w4a4_"
                         : is_w4a16() ? "w4a16_"
                         : is_mx()    ? "mxfp8_"
                         : is_fp8()   ? "fp8_" : "";
        if (is_w4a4())
            snprintf(buf, sizeof(buf), "%s%dx%dx%d_s%d_cwg%d_w%dx%d_ys%d_sk%d",
                     dt, bm, bn, bk, stages, cwg, warp_m, warp_n, y_slices, split_k);
        else if (is_skinny())
            snprintf(buf, sizeof(buf), "%s%s_%dx%dx%d_s%d_cwg%d_sk%d",
                     dt, is_tiny_m_tc() ? "tinymtc" : "tinym",
                     bm, bn, bk, stages, cwg, split_k);
        else
            snprintf(buf, sizeof(buf), "%s%dx%dx%d_s%d_cwg%d_w%dx%d_sk%d",
                     dt, bm, bn, bk, stages, cwg, warp_m, warp_n, split_k);
        return buf;
    }

    // Consumer threads (the producer warp group is excluded).
    int consumer_threads() const { return cwg * 128; }
    // Tiny-M only: n-columns per consumer thread.
    int npt() const { return bn / consumer_threads(); }

    // Constraints the kernel template asserts on. Returns "" if the config is
    // buildable, otherwise the reason — so a bad --config fails instantly
    // instead of after a multi-second nvcc round trip.
    std::string validate() const {
        char buf[256];
        auto fail = [&](const char *fmt, auto... args) {
            snprintf(buf, sizeof(buf), fmt, args...);
            return std::string(buf);
        };
        if (bm <= 0 || bn <= 0 || bk <= 0)
            return fail("tile dims must be positive (got %dx%dx%d)", bm, bn, bk);
        if (stages < 2) return fail("NUM_STAGES=%d must be >= 2 to pipeline", stages);
        if (cwg < 1) return fail("CWG=%d must be >= 1", cwg);
        if (split_k < 1) return fail("SPLIT_K=%d must be >= 1", split_k);
        if (is_packed4()) {
            // A BK-wide row of packed e2m1 is BK/2 bytes, and the 128B swizzle
            // that makes the per-n reads conflict-free needs 128 of them. W4A4
            // additionally needs BK to be a multiple of the MMA's k=64, which
            // 256 already satisfies.
            if (bk != 256)
                return fail("%s requires BK=256 (got %d)", is_w4a4() ? "W4A4" : "W4A16", bk);
        } else if (bk * elem_bytes() < 128) {
            return fail("BK=%d gives a %d-byte K row; 128B swizzle needs >= 128",
                        bk, bk * elem_bytes());
        }

        if (is_tiny_m_tc()) {
            // Mirrors the static_asserts in W4A16GemmTinyMTC.
            if (!is_w4a16())
                return fail("the tinymtc family exists only for W4A16 (elem=%d)", (int)elem);
            if (bm != 16)
                return fail("tinymtc fixes BM at 16 (mma.m16), got %d", bm);
            if (bn > 256) return fail("TMA box dims are capped at 256 (got BN=%d)", bn);
            const int cwarps = cwg * 4;
            if (bn % cwarps)
                return fail("BN=%d must split over the %d consumer warps", bn, cwarps);
            if ((bn / cwarps) % 8)
                return fail("BN/%d = %d must be a multiple of 8 (mma.n8)",
                            cwarps, bn / cwarps);
            return {};
        }

        if (is_tiny_m()) {
            // Mirrors the static_asserts in {BF16,FP8}GemmTinyM.
            if (bm > 256 || bn > 256 || bk > 256)
                return fail("TMA box dims are capped at 256 (got %dx%dx%d)", bm, bn, bk);
            if (bk % vec())
                return fail("BK=%d must be a multiple of %d (16B vector loads)", bk, vec());
            if (bn % 8) return fail("BN=%d must be a multiple of 8 rows", bn);
            if (bn % consumer_threads())
                return fail("BN=%d must be a multiple of the %d consumer threads",
                            bn, consumer_threads());
            if (npt() < 1)
                return fail("BN=%d < %d consumer threads; raise BN or lower CWG",
                            bn, consumer_threads());
            return {};
        }

        if (is_w4a4()) {
            // Mirrors the static_asserts in W4A4GemmMMA.
            if (family != KernelFamily::Mma)
                return fail("W4A4 has only the tensor-core family (no tiny-M)");
            if (split_k != 1)
                return fail("the W4A4 kernel has no split-k path (got sk%d)", split_k);
            if (y_slices < 1) return fail("ys%d must be >= 1", y_slices);
            if (bn % y_slices) return fail("BN=%d not divisible by ys%d", bn, y_slices);
            if ((bn / y_slices) % 8)
                return fail("output slice BN/ys = %d must be a multiple of 8",
                            bn / y_slices);
            if (bn / y_slices > 256)
                return fail("output slice BN/ys = %d exceeds the 256 TMA box cap",
                            bn / y_slices);
        } else if (y_slices != 1) {
            return fail("ys%d is a W4A4-only knob", y_slices);
        }

        // The fp8 MMA kernel has no split-k epilogue; tiny-M covers the skinny
        // shapes that would want one.
        if (is_fp8() && split_k != 1)
            return fail("the fp8 MMA kernel has no split-k path (got sk%d)", split_k);
        if (bm % 16) return fail("BM=%d must be a multiple of 16 (mma.m16)", bm);
        if (bn % 8) return fail("BN=%d must be a multiple of 8 (mma.n8)", bn);
        if (bk % mma_k())
            return fail("BK=%d must be a multiple of %d (mma.k%d)", bk, mma_k(), mma_k());
        if (warp_m <= 0 || warp_n <= 0)
            return fail("warp tile is unset (%dx%d)", warp_m, warp_n);
        if (warp_m % 16) return fail("WARP_M=%d must be a multiple of 16", warp_m);
        if (warp_n % 8) return fail("WARP_N=%d must be a multiple of 8", warp_n);
        if (bm % warp_m) return fail("BM=%d not divisible by WARP_M=%d", bm, warp_m);
        if (bn % warp_n) return fail("BN=%d not divisible by WARP_N=%d", bn, warp_n);
        int warps = (bm / warp_m) * (bn / warp_n);
        if (warps != cwg * 4)
            return fail("(BM/WARP_M)*(BN/WARP_N) = %d but CWG=%d needs %d consumer warps",
                        warps, cwg, cwg * 4);
        return {};
    }

    // Can this config run this problem shape at all?
    bool fits_shape(int M, int N, int K) const {
        if (N % bn || K % (bk * split_k)) return false;
        // Tiny-M does not tile M: one row-tile covers it, with TMA zero-filling
        // rows >= M. That also keeps these configs out of the running for any
        // shape they were not meant for.
        if (is_skinny()) return M <= bm;
        return M % bm == 0;
    }

    // Must agree exactly with sizeof(SMemStorage) in the matching kernel: it
    // is what gates a launch against the device shared-memory limit.
    size_t smem_bytes() const {
        size_t barriers = 2 * (size_t)stages * sizeof(uint64_t);
        size_t esz = (size_t)elem_bytes();
        if (is_w4a4()) {
            // Packed operands (BK/2 bytes per row), both scale tensors staged
            // as 512-byte tiles, and a Y_out of only BM x (BN/ys).
            const size_t row = (size_t)bk / 2;
            size_t tile = (size_t)stages * ((size_t)bm * row + (size_t)bn * row);
            const size_t packs = (size_t)bk / 64;
            const size_t x_tiles = ((size_t)bm + 127) / 128, w_tiles = ((size_t)bn + 127) / 128;
            tile += (size_t)stages * packs * (x_tiles + w_tiles) * 512;
            const size_t y_out = (size_t)bm * (bn / y_slices) * sizeof(bf16);
            return tile + y_out + barriers;
        }
        if (is_skinny() && is_w4a16()) {
            // bf16 X (unswizzled, so no row padding) plus packed e2m1 W.
            size_t tile = (size_t)stages * ((size_t)bm * bk * sizeof(bf16) +
                                            (size_t)bn * (bk / 2));
            return tile + barriers;
        }
        if (is_tiny_m()) {
            // The X stage stride is padded up to a whole 8-row (1024B) block
            // so every stage lands on the same 128B-swizzle phase.
            size_t x_rows = ((size_t)(bm + 7) / 8) * 8;
            size_t tile = (size_t)stages * (x_rows * bk + (size_t)bk * bn) * esz;
            return tile + barriers; // no smem Y staging: the epilogue goes direct
        }
        size_t tile = (size_t)stages * (bm * bk + bk * bn) * esz;
        // MXFP8 also stages the ue8m0 scale tiles: SF_PACKS = bk/128 per stage,
        // and one 512-byte tile per 128 rows of each operand.
        if (is_mx()) {
            size_t packs = (size_t)bk / 128;
            size_t x_tiles = ((size_t)bm + 127) / 128, w_tiles = ((size_t)bn + 127) / 128;
            tile += (size_t)stages * packs * (x_tiles + w_tiles) * 512;
        }
        // Y is bf16 whatever the operands are.
        size_t y_out = (split_k == 1) ? (size_t)bm * bn * sizeof(bf16) : 0;
        return tile + y_out + barriers;
    }

    // Pick the most square warp tile that exactly covers BM x BN with CWG*4
    // warps. Square warp tiles maximise operand reuse per accumulator register.
    bool derive_warp_tile() {
        int want = cwg * 4;
        double best_score = 1e30;
        int best_m = 0, best_n = 0;
        for (int wm = 16; wm <= bm; wm += 16) {
            if (bm % wm) continue;
            for (int wn = 8; wn <= bn; wn += 8) {
                if (bn % wn) continue;
                if ((bm / wm) * (bn / wn) != want) continue;
                double score = std::fabs(std::log2((double)wm / wn));
                if (score < best_score || (score == best_score && wm > best_m)) {
                    best_score = score;
                    best_m = wm;
                    best_n = wn;
                }
            }
        }
        if (!best_m) return false;
        warp_m = best_m;
        warp_n = best_n;
        return true;
    }

    static std::optional<GemmConfig> parse(std::string_view s, std::string *err = nullptr) {
        auto set_err = [&](std::string m) { if (err) *err = std::move(m); return std::nullopt; };

        if (s.rfind("bm", 0) == 0) {
            if (auto legacy = parse_legacy(s)) return legacy;
            return set_err("cannot parse legacy config '" + std::string(s) + "'");
        }

        GemmConfig c{};
        bool have_tile = false, have_bk = false;
        for (size_t start = 0; start <= s.size();) {
            size_t end = s.find('_', start);
            if (end == std::string_view::npos) end = s.size();
            std::string_view tok = s.substr(start, end - start);
            start = end + 1;
            if (tok.empty()) continue;

            int d[3];
            auto bad = [&](const char *want) {
                return set_err("bad token '" + std::string(tok) + "' (want " + want + ")");
            };
            if (tok == "tinymtc") {
                c.family = KernelFamily::TinyMTC;
            } else if (tok == "tinym") {
                c.family = KernelFamily::TinyM;
            } else if (tok == "mma") {
                c.family = KernelFamily::Mma;
            } else if (tok == "fp8") {
                c.elem = ElemType::Fp8;
            } else if (tok == "mxfp8") {
                c.elem = ElemType::MxFp8;
            } else if (tok == "w4a16") {
                c.elem = ElemType::W4A16;
            } else if (tok == "w4a4") {
                c.elem = ElemType::W4A4;
            } else if (tok.rfind("ys", 0) == 0) {
                if (!scan_tail(tok, 2, c.y_slices)) return bad("ys<n>");
            } else if (tok == "bf16") {
                c.elem = ElemType::Bf16;
            } else if (tok.rfind("sk", 0) == 0) {
                if (!scan_tail(tok, 2, c.split_k)) return bad("sk<n>");
            } else if (tok.rfind("cwg", 0) == 0) {
                if (!scan_tail(tok, 3, c.cwg)) return bad("cwg<n>");
            } else if (tok[0] == 's') {
                if (!scan_tail(tok, 1, c.stages)) return bad("s<n>");
            } else if (tok[0] == 'w') {
                if (scan_dims(tok, 1, d, 2) != 2) return bad("w<M>x<N>");
                c.warp_m = d[0];
                c.warp_n = d[1];
            } else if (tok[0] >= '0' && tok[0] <= '9') {
                int n = scan_dims(tok, 0, d, 3);
                if (n < 2) return bad("<BM>x<BN>[x<BK>]");
                c.bm = d[0];
                c.bn = d[1];
                if (n == 3) {
                    c.bk = d[2];
                    have_bk = true;
                }
                have_tile = true;
            } else {
                return set_err("unknown token '" + std::string(tok) + "' in config '" +
                               std::string(s) + "'");
            }
        }

        if (!have_tile)
            return set_err("config '" + std::string(s) + "' has no BMxBN tile");
        // The default BK is whatever fills a 128-byte swizzled smem row, which
        // depends on the element size — so it can only be applied once the whole
        // token list has been seen (tokens may appear in any order).
        if (!have_bk) c.bk = 128 / c.elem_bytes();
        if (c.is_skinny()) {
            if (c.warp_m || c.warp_n)
                return set_err("config '" + std::string(s) +
                               "' sets a warp tile, which the tiny-M family has no use for");
            return c;
        }
        if (c.warp_m == 0 && !c.derive_warp_tile())
            return set_err("no warp tile covers " + std::to_string(c.bm) + "x" +
                           std::to_string(c.bn) + " with " + std::to_string(c.cwg * 4) +
                           " warps; specify wMxN explicitly");
        return c;
    }

    // Full sweep of the configuration space, filtered by shared-memory budget.
    // Both families are enumerated; fits_shape() then keeps tiny-M configs out
    // of the running for anything but M <= BM, so a large-M autotune never
    // pays to compile them.
    static std::vector<GemmConfig> enumerate(size_t max_smem, ElemType elem = ElemType::Bf16) {
        std::vector<GemmConfig> out = enumerate_mma(max_smem, elem);
        for (auto &c : enumerate_tiny_m(max_smem, elem)) out.push_back(c);
        for (auto &c : enumerate_tiny_m_tc(max_smem, elem)) out.push_back(c);
        for (auto &c : enumerate_w4a4(max_smem, elem)) out.push_back(c);
        return out;
    }

    // Tiny-M leans on split-K far harder than the tensor-core family: with no
    // M-parallelism, N/BN tiles alone leave most SMs idle. BM below 4 only ever
    // matches M=1 or 2, but at those shapes the padding rows are pure waste —
    // measurably so, see the ncu notes in README.
    static std::vector<GemmConfig> enumerate_tiny_m(size_t max_smem,
                                                    ElemType elem = ElemType::Bf16) {
        const int bk0 = elem == ElemType::Bf16    ? 64
                        : elem == ElemType::W4A16 ? 256
                                                  : 128;
        std::vector<GemmConfig> out;
        for (int bm : {1, 2, 4, 8})
            for (int bn : {128, 256})
                for (int bk : {bk0})
                    for (int cwg : {1, 2})
                        for (int sk : {1, 2, 4, 8, 16})
                            for (int stages : {5, 4, 3, 2}) {
                                GemmConfig c{};
                                c.family = KernelFamily::TinyM;
                                c.elem = elem;
                                c.bm = bm;
                                c.bn = bn;
                                c.bk = bk;
                                c.stages = stages;
                                c.cwg = cwg;
                                c.split_k = sk;
                                if (!c.validate().empty()) continue;
                                if (c.smem_bytes() > max_smem) continue;
                                out.push_back(c);
                            }
        return out;
    }

    // Tensor-core skinny family. BM is fixed at 16 by mma.m16, so unlike
    // tiny-M there is no BM to sweep: one config serves every M <= 16, and the
    // padding rows cost nothing because the MMA computes them regardless.
    static std::vector<GemmConfig> enumerate_tiny_m_tc(size_t max_smem,
                                                       ElemType elem = ElemType::Bf16) {
        std::vector<GemmConfig> out;
        if (elem != ElemType::W4A16) return out;
        for (int bn : {64, 128, 256})
            for (int cwg : {1, 2})
                for (int sk : {1, 2, 4, 8, 16})
                    for (int stages : {5, 4, 3, 2}) {
                        GemmConfig c{};
                        c.family = KernelFamily::TinyMTC;
                        c.elem = elem;
                        c.bm = 16;
                        c.bn = bn;
                        c.bk = 256;
                        c.stages = stages;
                        c.cwg = cwg;
                        c.split_k = sk;
                        if (!c.validate().empty()) continue;
                        if (c.smem_bytes() > max_smem) continue;
                        out.push_back(c);
                    }
        return out;
    }

    // W4A4's tensor-core family. Only the output-slice count is new; ys is
    // swept because it trades shared memory against epilogue stores, and which
    // way that trade goes depends on whether the shape can fill the machine at
    // the larger tile it buys.
    static std::vector<GemmConfig> enumerate_w4a4(size_t max_smem,
                                                  ElemType elem = ElemType::Bf16) {
        std::vector<GemmConfig> out;
        if (elem != ElemType::W4A4) return out;
        for (int bm : {64, 128})
            for (int bn : {64, 128})
                for (int cwg : {1, 2})
                    for (int ys : {1, 2, 4})
                        for (int stages : {4, 3, 2}) {
                            GemmConfig c{};
                            c.family = KernelFamily::Mma;
                            c.elem = elem;
                            c.bm = bm;
                            c.bn = bn;
                            c.bk = 256;
                            c.stages = stages;
                            c.cwg = cwg;
                            c.y_slices = ys;
                            c.split_k = 1;
                            if (!c.derive_warp_tile()) continue;
                            if (!c.validate().empty()) continue;
                            if (c.smem_bytes() > max_smem) continue;
                            out.push_back(c);
                        }
        return out;
    }

    static std::vector<GemmConfig> enumerate_mma(size_t max_smem,
                                                 ElemType elem = ElemType::Bf16) {
        const bool fp8 = elem != ElemType::Bf16;
        const int bk0 = fp8 ? 128 : 64; // BK*elem_bytes must be >= SWIZZLE_128B
        std::vector<GemmConfig> out;
        for (int bm : {64, 128})
            for (int bn : {64, 128})
                for (int bk : {bk0})
                    for (int cwg : {1, 2})
                        // The fp8 MMA kernel has no split-k epilogue.
                        for (int sk : (fp8 ? std::vector<int>{1} : std::vector<int>{1, 2}))
                            for (int stages : {5, 4, 3, 2})
                                for (int wm = 16; wm <= bm; wm += 16) {
                                    if (bm % wm) continue;
                                    for (int wn = 8; wn <= bn; wn += 8) {
                                        if (bn % wn) continue;
                                        GemmConfig c{bm, bn, bk, stages, cwg, wm, wn, sk,
                                                     KernelFamily::Mma, elem};
                                        if (!c.validate().empty()) continue;
                                        if (c.smem_bytes() > max_smem) continue;
                                        out.push_back(c);
                                    }
                                }
        return out;
    }

private:
    static bool scan_uint(std::string_view s, size_t &i, int &out) {
        size_t start = i;
        int v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        if (i == start) return false;
        out = v;
        return true;
    }

    // "key<n>": one integer after a fixed-length prefix, consuming the token.
    static bool scan_tail(std::string_view tok, size_t offset, int &out) {
        return scan_uint(tok, offset, out) && offset == tok.size();
    }

    // "<a>x<b>[x<c>]" starting at offset. Returns how many numbers were read,
    // or 0 if the token is malformed or has more than max_dims of them.
    static int scan_dims(std::string_view tok, size_t i, int *out, int max_dims) {
        for (int n = 0; n < max_dims;) {
            if (!scan_uint(tok, i, out[n])) return 0;
            n++;
            if (i == tok.size()) return n;
            if (tok[i] != 'x') return 0;
            i++;
        }
        return 0;
    }

    // Accepts the pre-refactor name so existing autotune_cache.txt files load.
    // They are rewritten in the canonical format on the next save().
    static std::optional<GemmConfig> parse_legacy(std::string_view s) {
        GemmConfig c{};
        std::string str(s);
        if (sscanf(str.c_str(), "bm%d_bn%d_bk%d_s%d_cwg%d_wm%d_wn%d_sk%d",
                   &c.bm, &c.bn, &c.bk, &c.stages, &c.cwg,
                   &c.warp_m, &c.warp_n, &c.split_k) == 8)
            return c;
        return std::nullopt;
    }
};

// ── Autotune cache: one line per shape, "M,N,K config_name time_ms" ────
struct AutotuneCache {
    using Key = std::tuple<int, int, int>;
    struct Entry {
        GemmConfig config;
        double time_ms;
    };
    std::map<Key, Entry> entries;

    bool load(const std::string &path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            int M, N, K;
            char name_buf[256];
            double ms;
            if (sscanf(line.c_str(), "%d,%d,%d %255s %lf", &M, &N, &K, name_buf, &ms) == 5)
                if (auto cfg = GemmConfig::parse(name_buf))
                    entries[{M, N, K}] = {*cfg, ms};
        }
        printf("[cache] loaded %zu entries from %s\n", entries.size(), path.c_str());
        return true;
    }

    void save(const std::string &path) const {
        std::ofstream f(path);
        f << "# Autotune cache - M,N,K config_name time_ms\n";
        for (auto &[key, entry] : entries) {
            auto &[M, N, K] = key;
            f << M << "," << N << "," << K << " "
              << entry.config.name() << " " << entry.time_ms << "\n";
        }
    }

    std::optional<Entry> lookup(int M, int N, int K) const {
        auto it = entries.find({M, N, K});
        if (it != entries.end()) return it->second;
        return std::nullopt;
    }

    void store(int M, int N, int K, const GemmConfig &config, double time_ms) {
        entries[{M, N, K}] = {config, time_ms};
    }
};
