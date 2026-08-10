#pragma once
#include "gemm_config.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <dlfcn.h>

// ── On-the-fly kernel compilation ──────────────────────────────────────
//
// Each GemmConfig is turned into a .so by invoking nvcc on kernel_entry.cu
// with the tile parameters as -D defines, then dlopen'd. Results are cached on
// disk under a key that includes a fingerprint of the kernel sources, so
// editing bf16_gemm.cuh invalidates every previously built kernel instead of
// silently reusing a stale one.

struct JitOptions {
    std::string nvcc = GEMM_NVCC;
    std::string src_dir = GEMM_SRC_DIR;
    std::string cache_dir = GEMM_JIT_CACHE_DIR;
    std::string cuda_stub_dir = GEMM_CUDA_STUB_DIR;
    std::string arch = GEMM_CUDA_ARCH;
    int jobs = 0;        // 0 = hardware concurrency
    bool force = false;  // rebuild even if cached / previously failed
    bool verbose = false;
};

struct CompiledKernel {
    GemmConfig config;
    GemmKernelFn fn = nullptr;
    void *dl_handle = nullptr;
};

class KernelJit {
public:
    explicit KernelJit(JitOptions opts) : opts_(std::move(opts)) {
        if (opts_.jobs <= 0) {
            unsigned hw = std::thread::hardware_concurrency();
            opts_.jobs = hw ? (int)hw : 4;
        }
        std::filesystem::create_directories(opts_.cache_dir);
        fingerprint_ = compute_fingerprint();
        if (opts_.verbose)
            printf("[jit] cache=%s fingerprint=%016lx jobs=%d\n",
                   opts_.cache_dir.c_str(), fingerprint_, opts_.jobs);
    }

    ~KernelJit() {
        for (auto &[name, k] : loaded_)
            if (k.dl_handle) dlclose(k.dl_handle);
    }

    KernelJit(const KernelJit &) = delete;
    KernelJit &operator=(const KernelJit &) = delete;

    // Compile (if needed) and load one kernel. Returns nullptr on failure.
    const CompiledKernel *get(const GemmConfig &cfg) {
        auto r = get_many({cfg});
        return r.empty() ? nullptr : r.front();
    }

    // Compile a batch in parallel, then load. Failed configs are dropped, so
    // the result may be shorter than the input.
    std::vector<const CompiledKernel *> get_many(const std::vector<GemmConfig> &cfgs) {
        std::vector<GemmConfig> todo;
        for (auto &c : cfgs)
            if (!loaded_.count(c.name()) && !failed_.count(c.name()))
                todo.push_back(c);

        if (!todo.empty()) build_all(todo);

        std::vector<const CompiledKernel *> out;
        for (auto &c : cfgs) {
            auto it = loaded_.find(c.name());
            if (it != loaded_.end()) out.push_back(&it->second);
        }
        return out;
    }

private:
    JitOptions opts_;
    uint64_t fingerprint_ = 0;
    std::map<std::string, CompiledKernel> loaded_; // pointers must stay stable
    std::map<std::string, std::string> failed_;    // name -> reason

    std::string so_path(const GemmConfig &cfg) const {
        char buf[64];
        snprintf(buf, sizeof(buf), ".%016lx.so", fingerprint_);
        return opts_.cache_dir + "/" + cfg.name() + buf;
    }

    // Compile everything in `todo` across a thread pool, then dlopen serially.
    void build_all(const std::vector<GemmConfig> &todo) {
        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0}, cached{0}, built{0}, failed{0};
        std::vector<std::string> errors(todo.size());
        std::mutex io;

        int nthreads = std::min<int>(opts_.jobs, (int)todo.size());
        auto worker = [&]() {
            for (;;) {
                size_t i = next++;
                if (i >= todo.size()) return;
                Outcome oc = compile(todo[i], errors[i]);
                (oc == Outcome::Cached ? cached : oc == Outcome::Built ? built : failed)++;
                size_t d = ++done;
                std::lock_guard<std::mutex> lk(io);
                printf("\r[jit] %zu/%zu  cached %zu  built %zu  failed %zu   ",
                       d, todo.size(), cached.load(), built.load(), failed.load());
                fflush(stdout);
            }
        };

        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; t++) pool.emplace_back(worker);
        for (auto &t : pool) t.join();
        printf("\n");

        for (size_t i = 0; i < todo.size(); i++) {
            auto name = todo[i].name();
            if (!errors[i].empty()) {
                failed_[name] = errors[i];
                printf("[jit] %s FAILED: %s\n", name.c_str(), errors[i].c_str());
                continue;
            }
            std::string path = so_path(todo[i]);
            void *h = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h) {
                failed_[name] = std::string("dlopen: ") + dlerror();
                printf("[jit] %s FAILED: %s\n", name.c_str(), failed_[name].c_str());
                continue;
            }
            auto fn = (GemmKernelFn)dlsym(h, "gemm_run");
            if (!fn) {
                failed_[name] = std::string("dlsym(gemm_run): ") + dlerror();
                printf("[jit] %s FAILED: %s\n", name.c_str(), failed_[name].c_str());
                dlclose(h);
                continue;
            }
            loaded_[name] = CompiledKernel{todo[i], fn, h};
        }
    }

    enum class Outcome { Cached, Built, Failed };

    Outcome compile(const GemmConfig &cfg, std::string &error) {
        namespace fs = std::filesystem;
        std::string out = so_path(cfg);
        std::string log = out + ".log";
        std::string fail = out + ".fail";

        if (!opts_.force) {
            if (fs::exists(out)) return Outcome::Cached;
            if (fs::exists(fail)) {
                error = "nvcc failed previously, see " + log;
                return Outcome::Failed;
            }
        }

        std::string tmp = out + ".tmp." + std::to_string(getpid()) + "." +
                          std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));

        std::ostringstream cmd;
        cmd << '"' << opts_.nvcc << '"'
            << " -std=c++20 -O3 --use_fast_math"
            << " -arch=sm_" << opts_.arch
            << " -shared -Xcompiler -fPIC --cudart shared"
            << " -I\"" << opts_.src_dir << '"'
            << " -DGEMM_BM=" << cfg.bm
            << " -DGEMM_BN=" << cfg.bn
            << " -DGEMM_BK=" << cfg.bk
            << " -DGEMM_STAGES=" << cfg.stages
            << " -DGEMM_CWG=" << cfg.cwg
            << " -DGEMM_WM=" << cfg.warp_m
            << " -DGEMM_WN=" << cfg.warp_n
            << " -DGEMM_SPLIT_K=" << cfg.split_k;
        if (opts_.verbose) cmd << " --ptxas-options=-v,-warn-spills";
        cmd << " \"" << opts_.src_dir << "/kernel_entry.cu\""
            << " -o \"" << tmp << '"'
            << " -L\"" << opts_.cuda_stub_dir << "\" -lcuda"
            << " > \"" << log << "\" 2>&1";

        int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            std::error_code ec;
            fs::remove(tmp, ec);
            std::ofstream(fail) << "nvcc exit " << rc << "\n";
            error = "nvcc exit " + std::to_string(rc) + ", see " + log;
            return Outcome::Failed;
        }

        std::error_code ec;
        fs::rename(tmp, out, ec); // atomic within the cache dir
        if (ec) {
            fs::remove(tmp, ec);
            error = "rename to " + out + ": " + ec.message();
            return Outcome::Failed;
        }
        return Outcome::Built;
    }

    // FNV-1a over the kernel sources plus the toolchain/flags that affect
    // codegen. Any change here invalidates the whole on-disk cache.
    uint64_t compute_fingerprint() const {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](std::string_view s) {
            for (unsigned char c : s) {
                h ^= c;
                h *= 1099511628211ull;
            }
        };
        for (const char *f : {"kernel_entry.cu", "bf16_gemm.cuh", "common.h"})
            mix(read_file(opts_.src_dir + "/" + f));
        mix(opts_.nvcc);
        mix(opts_.arch);
        mix(run_capture('"' + opts_.nvcc + "\" --version 2>/dev/null"));
        return h;
    }

    static std::string read_file(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }

    static std::string run_capture(const std::string &cmd) {
        std::string out;
        FILE *p = popen(cmd.c_str(), "r");
        if (!p) return out;
        char buf[256];
        while (fgets(buf, sizeof(buf), p)) out += buf;
        pclose(p);
        return out;
    }
};
