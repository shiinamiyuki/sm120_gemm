// Device memory bandwidth: what this part actually sustains, and by which
// path. Every "GB/s" and "% of peak" claim about the GEMM kernels needs a
// denominator, and this is where it comes from.
//
// Measured on an RTX 5090 (170 SMs, 512-bit GDDR7 at 14.001 GHz), with the
// card at its real clocks (~2870 MHz SM under load, memory 14001):
//
//   theoretical (clock x2 x bus/8)      1792 GB/s
//   TMA read     (cp.async.bulk g2s)    1707 GB/s   95% of theoretical
//   TMA read     (warp-specialised)     1702 GB/s   95%
//   SIMT read    (float4, 4-way ILP)    1703 GB/s   95%
//   SIMT write                          1678 GB/s   94%
//   SIMT copy    (read+write, total)    1501 GB/s   84%
//   cudaMemcpy D2D (read+write, total)  1531 GB/s   85%
//
// Two things worth knowing from that table:
//
//   * TMA and SIMT vector loads land in the same place. Blackwell's bulk-copy
//     path is what the GEMM kernels use, so it is the one that has to be
//     measured, but for a pure stream it is not faster -- both sit at ~1705,
//     and so does the warp-specialised producer/consumer version. Stage count,
//     chunk size, block and thread counts, plain vs descriptor TMA, and 128B
//     swizzle vs none all move it by under 1%.
//
//   * A one-directional stream is the *best* case: read alone (1707) beats
//     mixed read+write (1501) by 12%, which is the usual DRAM bus-turnaround
//     penalty. Since the decode-shaped GEMMs here are almost entirely weight
//     reads, ~1700 GB/s is the right ceiling to judge them against.
//
// ---------------------------------------------------------------------------
// READ THIS BEFORE TRUSTING ANY NUMBER OUT OF THIS FILE.
//
// These figures are only meaningful if the GPU is actually at its clocks. This
// card was once found stuck at base clock (2010 MHz SM, 13801 MHz memory,
// never boosting, no throttle reason asserted, 243 W of a 575 W limit, 55 C).
// In that state *every* path here measured ~1360 GB/s -- a self-consistent,
// entirely plausible-looking 76% of theoretical, reproducible across TMA,
// warp-specialised TMA, descriptor TMA, LDGSTS and plain LDG, flat across
// every stage/chunk/block/thread combination, and stable under in-kernel
// repetition. Nothing about the shape of the result hinted that the machine,
// not the memory system, was the limit. A reboot restored boost and the same
// binary jumped to 1707.
//
// So the benchmark now measures the SM clock it actually ran at (clock64
// against wall time) and prints it against the card's nominal clock, which a
// healthy card boosts past. If that line looks low, fix the machine before
// believing anything below it:
//   nvidia-smi -q -d PERFORMANCE, then `sudo nvidia-smi -rgc` / reboot.
// ---------------------------------------------------------------------------
//
// Controls that were checked and did not matter: buffer contents (a constant
// byte, which any memory compression would exploit, measures the same as
// pseudorandom words to within 0.1%), and working-set size anywhere past L2.
#include "bf16_gemm.cuh"  // mbarrier helpers, smem_u32, fence_proxy_async_shared
#include "block_scale.h"  // cp_async_bulk_g2s -- the same instruction the kernels issue

#include <algorithm>
#include <string>
#include <vector>

// ── Kernels ────────────────────────────────────────────────────────────

// Pseudorandom fill, so that memory compression (which would make a constant
// buffer look faster than it is) cannot flatter the result.
__global__ void bw_fill_random(uint32_t *p, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t x = (uint32_t)i * 2654435761u;
        x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
        p[i] = x;
    }
}

// TMA read: one thread per CTA drives a NUM_STAGES-deep bulk-copy pipeline
// into shared memory. Nothing consumes the data, so this isolates the
// global -> shared path. The pipeline is the same shape as the GEMM
// producers': issue NUM_STAGES copies, then for each completion issue one
// more into the stage that just freed.
template <int NUM_STAGES, int CHUNK>
__global__ __launch_bounds__(32) void bw_tma_read(const char *__restrict__ src, size_t nchunks) {
    extern __shared__ __align__(1024) char smem[];
    uint64_t *bar = reinterpret_cast<uint64_t *>(smem + (size_t)NUM_STAGES * CHUNK);

    if (threadIdx.x == 0)
        for (int s = 0; s < NUM_STAGES; s++) mbarrier_init(&bar[s], 1);
    __syncthreads();
    fence_proxy_async_shared();

    if (threadIdx.x == 0) {
        // This CTA takes chunks blockIdx.x, +gridDim.x, +2*gridDim.x, ...
        long count = (long)(blockIdx.x < nchunks
                                ? (nchunks - blockIdx.x + gridDim.x - 1) / gridDim.x
                                : 0);

        auto issue = [&](long n, int stage) {
            size_t i = (size_t)blockIdx.x + (size_t)n * gridDim.x;
            uint32_t mb = smem_u32(&bar[stage]);
            mbarrier_expect_tx(mb, CHUNK);
            cp_async_bulk_g2s(smem_u32(smem + (size_t)stage * CHUNK),
                              src + i * (size_t)CHUNK, CHUNK, mb);
        };

        long issued = 0;
        while (issued < NUM_STAGES && issued < count) {
            issue(issued, (int)(issued % NUM_STAGES));
            issued++;
        }
        int stage = 0;
        uint32_t phase = 0;
        for (long done = 0; done < count; done++) {
            mbarrier_wait(smem_u32(&bar[stage]), phase);
            if (issued < count) { issue(issued, stage); issued++; }
            stage++;
            if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
        }
        for (int s = 0; s < NUM_STAGES; s++) mbarrier_inval(&bar[s]);
    }
    __syncthreads();
}

// The SM clock actually in effect during a memory-bound loop, in cycles. A
// card stuck in a low clock state produces a self-consistent, plausible-
// looking bandwidth figure that is simply wrong (see the header), and nothing
// else in this file can tell you that happened. This can.
__global__ void bw_clock_probe(const float4 *__restrict__ p, size_t n4, float *out,
                               long long *cyc) {
    long long t0 = clock64();
    float a = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = __ldg(&p[i]);
        a += v.x + v.y + v.z + v.w;
    }
    if (a == 1e30f) *out = a;
    if (threadIdx.x == 0) cyc[blockIdx.x] = clock64() - t0;
}

// Word-sum over the whole buffer, the reference the warp-specialised path
// checks itself against.
__global__ void bw_ref_sum(const uint32_t *__restrict__ p, size_t n, unsigned long long *out) {
    unsigned long long s = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) s += p[i];
    atomicAdd(out, s);
}

// Warp-specialised TMA read, one CTA per SM: two warps, exactly the shape a
// GEMM mainloop runs. Warp 0 produces, warp 1 consumes; both walk the same
// chunk sequence and index the ring as stage = n % NUM_STAGES, so the stage
// count is a free parameter -- decoupled from the warp count, bounded only by
// shared memory. (bw_tma_read above discards the data and so times only the
// global -> shared leg; here the producer cannot reuse a stage until the
// consumer has actually read it, which is the dependency a mainloop pays.)
//
// cp.async.bulk is issued by ONE elected thread -- lane 0 of the producer
// warp. Pipeline depth comes from NUM_STAGES, never from spreading the issue
// across lanes: 32 lanes each issuing their own copy would only make the warp
// replay the instruction serially, and their mbarrier spin-waits would diverge
// inside the one warp and fight each other.
//
//   full[s]   producer's arrive.expect_tx + the copy's tx completion -> consumer
//   empty[s]  the consumer warp's 32 arrivals                        -> producer
//
// Round k = n / NUM_STAGES is the ring lap, and drives the barrier parities:
// on lap k stage s has been filled k+1 times and emptied k times.
template <int NUM_STAGES, int CHUNK>
__global__ __launch_bounds__(64)
void bw_tma_ws(const char *__restrict__ src, size_t nchunks,
               unsigned long long *sum, unsigned long long *chunks) {
    extern __shared__ __align__(1024) char smem[];
    uint64_t *full = reinterpret_cast<uint64_t *>(smem + (size_t)NUM_STAGES * CHUNK);
    uint64_t *empty = full + NUM_STAGES;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // NUM_STAGES may exceed blockDim.x, so stride the init.
    for (int s = threadIdx.x; s < NUM_STAGES; s += blockDim.x) {
        mbarrier_init(&full[s], 1);
        mbarrier_init(&empty[s], 32);
    }
    __syncthreads();
    fence_proxy_async_shared();

    // This CTA takes chunks blockIdx.x, +gridDim.x, +2*gridDim.x, ...
    const long count = (long)(blockIdx.x < nchunks
                                  ? (nchunks - blockIdx.x + gridDim.x - 1) / gridDim.x
                                  : 0);

    if (warp == 0) {
        if (lane == 0) {
            int s = 0;
            long k = 0;
            for (long n = 0; n < count; n++) {
                // Lap 0 fills a virgin ring; after that, wait for the consumer
                // to have released this stage on the previous lap.
                if (k) mbarrier_wait(smem_u32(&empty[s]), (uint32_t)((k - 1) & 1));
                size_t g = (size_t)blockIdx.x + (size_t)n * gridDim.x;
                uint32_t fb = smem_u32(&full[s]);
                mbarrier_expect_tx(fb, CHUNK);
                cp_async_bulk_g2s(smem_u32(smem + (size_t)s * CHUNK),
                                  src + g * (size_t)CHUNK, CHUNK, fb);
                if (++s == NUM_STAGES) { s = 0; k++; }
            }
        }
    } else if (warp == 1) {
        unsigned long long acc = 0;
        int s = 0;
        long k = 0;
        for (long n = 0; n < count; n++) {
            mbarrier_wait(smem_u32(&full[s]), (uint32_t)(k & 1));
            const uint32_t *w = reinterpret_cast<const uint32_t *>(smem + (size_t)s * CHUNK);
#pragma unroll 4
            for (int i = lane; i < CHUNK / 4; i += 32) acc += w[i];
            __syncwarp();
            mbarrier_arrive(smem_u32(&empty[s]));
            if (++s == NUM_STAGES) { s = 0; k++; }
        }
        // Reduce in-warp first; 32 atomics onto one address would contend
        // enough to show up in the timing.
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) atomicAdd(sum, acc);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(chunks, (unsigned long long)count);
    for (int s = threadIdx.x; s < NUM_STAGES; s += blockDim.x) {
        mbarrier_inval(&full[s]);
        mbarrier_inval(&empty[s]);
    }
}

// SIMT read: four independent accumulators and four loads in flight, so
// neither a dependency chain nor a lack of memory-level parallelism can be
// what limits it. The sum is never stored (the guard is unreachable), but the
// compiler cannot prove that, so the loads survive.
__global__ void bw_simt_read(const float4 *__restrict__ p, size_t n4, float *out) {
    float4 a0{}, a1{}, a2{}, a3{};
    size_t lane = blockDim.x;
    size_t stride = (size_t)gridDim.x * blockDim.x * 4;
    for (size_t i = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x;
         i + 3 * lane < n4; i += stride) {
        float4 v0 = __ldg(&p[i]),            v1 = __ldg(&p[i + lane]);
        float4 v2 = __ldg(&p[i + 2 * lane]), v3 = __ldg(&p[i + 3 * lane]);
        a0.x += v0.x; a0.y += v0.y; a0.z += v0.z; a0.w += v0.w;
        a1.x += v1.x; a1.y += v1.y; a1.z += v1.z; a1.w += v1.w;
        a2.x += v2.x; a2.y += v2.y; a2.z += v2.z; a2.w += v2.w;
        a3.x += v3.x; a3.y += v3.y; a3.z += v3.z; a3.w += v3.w;
    }
    if (a0.x == 1e30f) *out = a0.x + a1.x + a2.x + a3.x;
}

__global__ void bw_simt_write(float4 *__restrict__ dst, size_t n4) {
    float4 v = make_float4(1, 2, 3, 4);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = v;
}

__global__ void bw_simt_copy(const float4 *__restrict__ src, float4 *__restrict__ dst, size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = __ldg(&src[i]);
}

// ── Harness ────────────────────────────────────────────────────────────

// Best-of, not mean: this is a ceiling measurement, and the thing being
// estimated is what the hardware can do when nothing else interferes. (The
// GEMM harness uses a mean over L2-flushed iterations instead, because there
// the question is throughput under realistic cache pressure.)
template <class F>
static double best_ms(F &&f, int reps) {
    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) f();
    CHECK_CUDA(cudaDeviceSynchronize());
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        CHECK_CUDA(cudaEventRecord(a));
        f();
        CHECK_CUDA(cudaEventRecord(b));
        CHECK_CUDA(cudaEventSynchronize(b));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

struct Ctx {
    char *src = nullptr, *dst = nullptr;
    float *sink = nullptr;
    unsigned long long *ver = nullptr;  // [0] word-sum, [1] chunk count, per launch
    unsigned long long ref = 0;         // word-sum over the whole buffer
    size_t bytes = 0;
    int sms = 0, reps = 15;
    double theoretical = 0.0;
};

static void report(const Ctx &c, const char *what, double ms, double moved) {
    printf("  %-36s %8.3f ms  %8.1f GB/s   %4.0f%% of theoretical\n",
           what, ms, moved / (ms * 1e-3) / 1e9,
           100.0 * (moved / (ms * 1e-3) / 1e9) / c.theoretical);
}

// One TMA configuration. Returns GB/s, or 0 if it does not fit / launch.
template <int NUM_STAGES, int CHUNK>
static double tma_variant(Ctx &c, int blocks, bool quiet) {
    size_t smem = (size_t)NUM_STAGES * CHUNK + NUM_STAGES * sizeof(uint64_t);
    auto k = bw_tma_read<NUM_STAGES, CHUNK>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)
        != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    size_t nchunks = c.bytes / CHUNK;
    double ms = best_ms([&] { k<<<blocks, 32, smem>>>(c.src, nchunks); }, c.reps);
    if (cudaGetLastError() != cudaSuccess) return 0.0;
    double gbps = (double)(nchunks * (size_t)CHUNK) / (ms * 1e-3) / 1e9;
    if (!quiet)
        printf("    stages=%-3d chunk=%6dB blocks=%5d smem=%3zuKB   %7.3f ms  %8.1f GB/s\n",
               NUM_STAGES, CHUNK, blocks, smem / 1024, ms, gbps);
    return gbps;
}

// One warp-specialised configuration, launched one CTA per SM. Verified before
// it is timed: every chunk must have been issued, and the sum of every word
// that actually landed in shared memory must equal the whole-buffer reference.
// A configuration that quietly reads less than the buffer therefore cannot
// report a flattering number -- it reports nothing. Returns GB/s, or 0.
template <int NUM_STAGES, int CHUNK>
static double tma_ws_variant(Ctx &c, bool quiet) {
    constexpr int THREADS = 64;  // one producer warp, one consumer warp
    size_t smem = (size_t)NUM_STAGES * CHUNK + 2 * NUM_STAGES * sizeof(uint64_t);
    auto k = bw_tma_ws<NUM_STAGES, CHUNK>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)
        != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    size_t nchunks = c.bytes / CHUNK;
    size_t covered = nchunks * (size_t)CHUNK;

    CHECK_CUDA(cudaMemset(c.ver, 0, 2 * sizeof(unsigned long long)));
    k<<<c.sms, THREADS, smem>>>(c.src, nchunks, c.ver, c.ver + 1);
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return 0.0; }
    unsigned long long h[2] = {0, 0};
    CHECK_CUDA(cudaMemcpy(h, c.ver, sizeof(h), cudaMemcpyDeviceToHost));
    // The word-sum matches the reference only when the chunk size divides the
    // buffer; otherwise the tail is legitimately outside the covered range.
    bool ok = h[1] == (unsigned long long)nchunks && (covered != c.bytes || h[0] == c.ref);
    if (!ok) {
        fprintf(stderr, "    stages=%-3d chunk=%6dB  COVERAGE FAILED "
                        "(chunks %llu/%zu, sum %s)\n",
                NUM_STAGES, CHUNK, h[1], nchunks,
                h[0] == c.ref ? "ok" : "mismatch");
        return 0.0;
    }

    double ms = best_ms([&] { k<<<c.sms, THREADS, smem>>>(c.src, nchunks, c.ver, c.ver + 1); },
                        c.reps);
    if (cudaGetLastError() != cudaSuccess) return 0.0;
    double gbps = (double)covered / (ms * 1e-3) / 1e9;
    if (!quiet)
        printf("    stages=%-3d chunk=%6dB thr=%3d smem=%3zuKB   %7.3f ms  %8.1f GB/s\n",
               NUM_STAGES, CHUNK, THREADS, smem / 1024, ms, gbps);
    return gbps;
}

int main(int argc, char **argv) {
    size_t mib = 4096;
    int reps = 15;
    bool constant_fill = false, sweep = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            printf(R"(usage: bench_bandwidth [options]

Sustained device memory bandwidth, by access path. The buffer must exceed L2
or the result is a cache measurement; the default 4 GiB is far past it.

options:
  --mib N       buffer size in MiB (default 4096)
  --reps N      timed iterations, best-of (default 15)
  --constant    fill the buffer with one repeated byte instead of pseudorandom
                words, as a control for memory compression
  --sweep       print every TMA stage/chunk/block combination, not just the best
  -h, --help    this message
)");
            return 0;
        } else if (a == "--mib" && i + 1 < argc) mib = strtoull(argv[++i], nullptr, 10);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--constant") constant_fill = true;
        else if (a == "--sweep") sweep = true;
        else { fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]); return 2; }
    }

    Ctx c;
    c.reps = reps;
    c.bytes = mib << 20;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int mem_khz = 0, bus = 0, l2 = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&c.sms, cudaDevAttrMultiProcessorCount, 0));
    CHECK_CUDA(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    CHECK_CUDA(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0));
    CHECK_CUDA(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    // GDDR6/7 transfer twice per reported clock; this is the number on the
    // spec sheet, and it is an upper bound no real access pattern reaches.
    c.theoretical = 2.0 * (double)mem_khz * 1e3 * (bus / 8.0) / 1e9;

    printf("%s: %d SMs, %d-bit bus @ %.3f GHz, L2 %.0f MB\n",
           prop.name, c.sms, bus, mem_khz / 1e6, l2 / 1048576.0);
    printf("theoretical peak: %.0f GB/s\n", c.theoretical);
    printf("buffer: %zu MiB of %s (L2 is %.0f MB, so nothing caches)\n\n",
           mib, constant_fill ? "one repeated byte" : "pseudorandom words", l2 / 1048576.0);

    if (c.bytes <= (size_t)l2 * 2)
        fprintf(stderr, "WARNING: buffer is not comfortably larger than L2; "
                        "raise --mib or this measures cache\n");

    CHECK_CUDA(cudaMalloc(&c.src, c.bytes));
    CHECK_CUDA(cudaMalloc(&c.dst, c.bytes));
    CHECK_CUDA(cudaMalloc(&c.sink, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c.ver, 2 * sizeof(unsigned long long)));
    if (constant_fill) {
        CHECK_CUDA(cudaMemset(c.src, 1, c.bytes));
    } else {
        bw_fill_random<<<1024, 256>>>((uint32_t *)c.src, c.bytes / 4);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemset(c.ver, 0, sizeof(unsigned long long)));
    bw_ref_sum<<<1024, 256>>>((const uint32_t *)c.src, c.bytes / 4, c.ver);
    CHECK_CUDA(cudaMemcpy(&c.ref, c.ver, sizeof(c.ref), cudaMemcpyDeviceToHost));

    // ── Clock sanity: is this card actually at its clocks? ─────────
    {
        int max_khz = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&max_khz, cudaDevAttrClockRate, 0));
        int blocks = c.sms * 4;
        long long *d_cyc = nullptr;
        CHECK_CUDA(cudaMalloc(&d_cyc, blocks * sizeof(long long)));
        std::vector<long long> h_cyc(blocks);
        size_t probe4 = c.bytes / sizeof(float4);
        double pms = best_ms([&] {
            bw_clock_probe<<<blocks, 256>>>((const float4 *)c.src, probe4, c.sink, d_cyc);
        }, 3);
        CHECK_CUDA(cudaMemcpy(h_cyc.data(), d_cyc, blocks * sizeof(long long),
                              cudaMemcpyDeviceToHost));
        long long mx = *std::max_element(h_cyc.begin(), h_cyc.end());
        // cudaDevAttrClockRate is the *nominal* clock, not the boost ceiling --
        // a healthy card boosts past it, so this is a floor to check against,
        // not a percentage to maximise.
        double mhz = mx / (pms * 1e-3) / 1e6, max_mhz = max_khz / 1e3;
        printf("SM clock under load: %.0f MHz (nominal %.0f MHz)%s\n\n", mhz, max_mhz,
               mhz < 0.9 * max_mhz ? "   <-- LOW, see warning" : "");
        if (mhz < 0.9 * max_mhz)
            fprintf(stderr,
                    "WARNING: the SM clock is far below this card's maximum. Every number\n"
                    "         below will be low, self-consistent and wrong. Check\n"
                    "         `nvidia-smi -q -d PERFORMANCE`; try `sudo nvidia-smi -rgc`,\n"
                    "         `sudo nvidia-smi -pm 1`, or a reboot, before trusting this run.\n\n");
        cudaFree(d_cyc);
    }

    // ── TMA read: the path the GEMM producers use ──────────────────
    printf("TMA read (cp.async.bulk global -> shared)\n");
    double best_tma = 0.0;
    for (int mult : {1, 2, 4}) {
        int blocks = c.sms * mult;
        best_tma = std::max(best_tma, tma_variant<3, 32768>(c, blocks, !sweep));
        best_tma = std::max(best_tma, tma_variant<6, 16384>(c, blocks, !sweep));
        best_tma = std::max(best_tma, tma_variant<8, 8192>(c, blocks, !sweep));
        best_tma = std::max(best_tma, tma_variant<16, 4096>(c, blocks, !sweep));
    }
    printf("  %-36s %8s   %8.1f GB/s   %4.0f%% of theoretical\n",
           "best over stages/chunk/blocks", "", best_tma, 100.0 * best_tma / c.theoretical);

    // ── Warp-specialised TMA: producer warps, consumer warps, 1 CTA/SM ──
    // Two warps, one CTA per SM, stage count free. Every configuration is
    // coverage-checked against the whole-buffer word-sum before it is timed.
    printf("\nWarp-specialised TMA read (2 warps, 1 CTA/SM, verified)\n");
    double best_ws = 0.0;
    best_ws = std::max(best_ws, tma_ws_variant<4, 16384>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<6, 16384>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<8, 8192>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<12, 8192>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<16, 4096>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<24, 4096>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<32, 2048>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<48, 2048>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<64, 1024>(c, !sweep));
    best_ws = std::max(best_ws, tma_ws_variant<96, 1024>(c, !sweep));
    printf("  %-36s %8s   %8.1f GB/s   %4.0f%% of theoretical\n",
           "best over stages/chunk", "", best_ws, 100.0 * best_ws / c.theoretical);

    // ── SIMT paths, for comparison ─────────────────────────────────
    printf("\nSIMT and copy-engine paths\n");
    size_t n4 = c.bytes / sizeof(float4);
    double ms = best_ms([&] {
        bw_simt_read<<<c.sms * 4, 256>>>((const float4 *)c.src, n4, c.sink);
    }, c.reps);
    report(c, "SIMT read (float4, 4-way ILP)", ms, (double)c.bytes);

    ms = best_ms([&] { bw_simt_write<<<c.sms * 4, 256>>>((float4 *)c.dst, n4); }, c.reps);
    report(c, "SIMT write", ms, (double)c.bytes);

    ms = best_ms([&] {
        bw_simt_copy<<<c.sms * 4, 256>>>((const float4 *)c.src, (float4 *)c.dst, n4);
    }, c.reps);
    report(c, "SIMT copy (read+write counted)", ms, 2.0 * c.bytes);

    ms = best_ms([&] {
        CHECK_CUDA(cudaMemcpyAsync(c.dst, c.src, c.bytes, cudaMemcpyDeviceToDevice));
    }, c.reps);
    report(c, "cudaMemcpy D2D (read+write counted)", ms, 2.0 * c.bytes);

    CHECK_CUDA(cudaGetLastError());
    printf("\nUse the TMA/SIMT *read* figure as the ceiling for decode-shaped GEMMs:\n"
           "they are almost entirely weight streaming, and a one-directional stream\n"
           "is the best case -- mixing writes in costs ~12%% to bus turnaround.\n"
           "If the SM clock line at the top was low, none of this is valid.\n");
    cudaFree(c.ver);
    cudaFree(c.sink);
    cudaFree(c.dst);
    cudaFree(c.src);
    return 0;
}
