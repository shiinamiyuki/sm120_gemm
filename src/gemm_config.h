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

// Function pointer type exported by each JIT-compiled kernel .so.
// Uses void* for bf16 pointers to avoid C++ mangling issues in extern "C".
using GemmKernelFn = void (*)(int M, int N, int K,
                              const void *X, const void *W, void *Y,
                              float *workspace, cudaStream_t stream);

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
// parse() accepts the tokens in any order and fills in defaults, so partial
// specs work on the command line: "128x128", "128x128x64_sk2", "s4_cwg1_128x64".
struct GemmConfig {
    int bm = 0, bn = 0, bk = 64;
    int stages = 3, cwg = 2;
    int warp_m = 0, warp_n = 0; // 0 = derive from bm/bn/cwg
    int split_k = 1;

    std::string name() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "%dx%dx%d_s%d_cwg%d_w%dx%d_sk%d",
                 bm, bn, bk, stages, cwg, warp_m, warp_n, split_k);
        return buf;
    }

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
        if (bm % 16) return fail("BM=%d must be a multiple of 16 (mma.m16)", bm);
        if (bn % 8) return fail("BN=%d must be a multiple of 8 (mma.n8)", bn);
        if (bk % 16) return fail("BK=%d must be a multiple of 16 (mma.k16)", bk);
        if (bk * (int)sizeof(bf16) < 128)
            return fail("BK=%d gives a %zu-byte K row; 128B swizzle needs >= 128",
                        bk, bk * sizeof(bf16));
        if (stages < 2) return fail("NUM_STAGES=%d must be >= 2 to pipeline", stages);
        if (cwg < 1) return fail("CWG=%d must be >= 1", cwg);
        if (split_k < 1) return fail("SPLIT_K=%d must be >= 1", split_k);
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
        if (M % bm || N % bn || K % bk) return false;
        if (split_k > 1 && K % (bk * split_k)) return false;
        return true;
    }

    size_t smem_bytes() const {
        size_t tile = (size_t)stages * (bm * bk + bk * bn) * sizeof(bf16);
        size_t barriers = 2 * (size_t)stages * sizeof(uint64_t);
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
        bool have_tile = false;
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
            if (tok.rfind("sk", 0) == 0) {
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
                if (n == 3) c.bk = d[2];
                have_tile = true;
            } else {
                return set_err("unknown token '" + std::string(tok) + "' in config '" +
                               std::string(s) + "'");
            }
        }

        if (!have_tile)
            return set_err("config '" + std::string(s) + "' has no BMxBN tile");
        if (c.warp_m == 0 && !c.derive_warp_tile())
            return set_err("no warp tile covers " + std::to_string(c.bm) + "x" +
                           std::to_string(c.bn) + " with " + std::to_string(c.cwg * 4) +
                           " warps; specify wMxN explicitly");
        return c;
    }

    // Full sweep of the configuration space, filtered by shared-memory budget.
    static std::vector<GemmConfig> enumerate(size_t max_smem) {
        std::vector<GemmConfig> out;
        for (int bm : {64, 128})
            for (int bn : {64, 128})
                for (int bk : {64}) // BK*sizeof(bf16) must be >= SWIZZLE_128B
                    for (int cwg : {1, 2})
                        for (int sk : {1, 2})
                            for (int stages : {4, 3, 2})
                                for (int wm = 16; wm <= bm; wm += 16) {
                                    if (bm % wm) continue;
                                    for (int wn = 8; wn <= bn; wn += 8) {
                                        if (bn % wn) continue;
                                        GemmConfig c{bm, bn, bk, stages, cwg, wm, wn, sk};
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
