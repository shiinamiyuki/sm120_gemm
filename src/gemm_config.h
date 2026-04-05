#pragma once
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// Function pointer type exported by each kernel .so
// Uses void* for bf16 pointers to avoid C++ mangling issues in extern "C"
using GemmKernelFn = void (*)(int M, int N, int K,
                              const void *X, const void *W, void *Y,
                              float *workspace, cudaStream_t stream);

struct GemmConfig {
    int bm, bn, bk, num_stages, cwg, warp_m, warp_n, split_k;

    std::string name() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "bm%d_bn%d_bk%d_s%d_cwg%d_wm%d_wn%d_sk%d",
                 bm, bn, bk, num_stages, cwg, warp_m, warp_n, split_k);
        return buf;
    }

    bool is_valid_for(int M, int N, int K) const {
        if (M % bm != 0 || N % bn != 0 || K % bk != 0) return false;
        if (split_k > 1 && K % (bk * split_k) != 0) return false;
        return true;
    }

    size_t smem_bytes() const {
        size_t tile = (size_t)num_stages * (bm * bk + bk * bn) * 2;
        size_t barriers = 2 * num_stages * 8;
        size_t y_out = (split_k == 1) ? (size_t)bm * bn * 2 : 0;
        return tile + y_out + barriers;
    }

    static std::optional<GemmConfig> parse(const std::string &s) {
        GemmConfig c{};
        if (sscanf(s.c_str(), "bm%d_bn%d_bk%d_s%d_cwg%d_wm%d_wn%d_sk%d",
                   &c.bm, &c.bn, &c.bk, &c.num_stages, &c.cwg,
                   &c.warp_m, &c.warp_n, &c.split_k) == 8)
            return c;
        return std::nullopt;
    }
};

// ── A loaded kernel from a shared library ──────────────────────────────
struct LoadedKernel {
    GemmConfig config;
    GemmKernelFn fn;
    void *dl_handle;
};

// ── Registry: scans a directory for kernel .so files ───────────────────
struct KernelRegistry {
    std::vector<LoadedKernel> kernels;

    bool load_dir(const std::string &dir) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir)) {
            fprintf(stderr, "KernelRegistry: not a directory: %s\n", dir.c_str());
            return false;
        }
        for (auto &entry : fs::directory_iterator(dir)) {
            auto path = entry.path();
            if (path.extension() != ".so") continue;
            auto stem = path.stem().string(); // e.g. "libgemm_bm128_..."
            // strip "libgemm_" prefix
            const char *prefix = "libgemm_";
            if (stem.rfind(prefix, 0) != 0) continue;
            auto cfg_str = stem.substr(strlen(prefix));
            auto cfg = GemmConfig::parse(cfg_str);
            if (!cfg) {
                fprintf(stderr, "KernelRegistry: cannot parse config from %s\n", stem.c_str());
                continue;
            }
            void *handle = dlopen(path.c_str(), RTLD_LAZY);
            if (!handle) {
                fprintf(stderr, "KernelRegistry: dlopen failed for %s: %s\n",
                        path.c_str(), dlerror());
                continue;
            }
            auto fn = (GemmKernelFn)dlsym(handle, "gemm_run");
            if (!fn) {
                fprintf(stderr, "KernelRegistry: dlsym(gemm_run) failed for %s: %s\n",
                        path.c_str(), dlerror());
                dlclose(handle);
                continue;
            }
            kernels.push_back({*cfg, fn, handle});
        }
        printf("KernelRegistry: loaded %zu kernels from %s\n", kernels.size(), dir.c_str());
        return !kernels.empty();
    }

    std::vector<const LoadedKernel *> get_valid(int M, int N, int K, size_t max_smem = 0) const {
        std::vector<const LoadedKernel *> result;
        for (auto &k : kernels) {
            if (!k.config.is_valid_for(M, N, K)) continue;
            if (max_smem > 0 && k.config.smem_bytes() > max_smem) continue;
            result.push_back(&k);
        }
        return result;
    }

    ~KernelRegistry() {
        for (auto &k : kernels)
            if (k.dl_handle) dlclose(k.dl_handle);
    }
};

// ── Autotune cache: simple text file ───────────────────────────────────
// Format per line: M,N,K config_name time_ms
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
            if (sscanf(line.c_str(), "%d,%d,%d %255s %lf", &M, &N, &K, name_buf, &ms) == 5) {
                auto cfg = GemmConfig::parse(name_buf);
                if (cfg) entries[{M, N, K}] = {*cfg, ms};
            }
        }
        printf("AutotuneCache: loaded %zu entries from %s\n", entries.size(), path.c_str());
        return true;
    }

    void save(const std::string &path) const {
        std::ofstream f(path);
        f << "# Autotune cache — M,N,K config_name time_ms\n";
        for (auto &[key, entry] : entries) {
            auto &[M, N, K] = key;
            f << M << "," << N << "," << K << " "
              << entry.config.name() << " " << entry.time_ms << "\n";
        }
        printf("AutotuneCache: saved %zu entries to %s\n", entries.size(), path.c_str());
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

// ── Dispatch: run a kernel given a config ──────────────────────────────
inline void dispatch(const LoadedKernel &kernel,
                     int M, int N, int K,
                     const bf16 *X, const bf16 *W, bf16 *Y,
                     float *workspace, cudaStream_t stream) {
    kernel.fn(M, N, K, X, W, Y, workspace, stream);
}
