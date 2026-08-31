#pragma once
#include <utility>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#define CHECK_CUDA(call)                                                                                       \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t err = call;                                                                                \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)

__global__ void sleep_kernel(uint32_t ms)
{
    __nanosleep(ms * 1000000);
}

// Back-to-back timing inside a CUDA graph, best-of. This is what external
// benchmarks (FlashInfer, vLLM, anything driven from Python) report, because a
// graph is the only way to get host launch cost out of a 20-microsecond kernel.
// It is a different question from the default bench_ms below -- no idle gap
// before each call, so caches and clocks are warm, and best-of rather than mean
// -- and the two can differ by more than the gap between two kernels. Compare
// like with like: set GEMM_BENCH_GRAPH=1 on both sides or neither.
template <class F>
double bench_ms_graph(F &&f, cudaStream_t stream, uint32_t iters = 50, uint32_t reps = 12)
{
    for (uint32_t i = 0; i < 5; i++) f();
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (uint32_t i = 0; i < iters; i++) f();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    double best = 1e30;
    for (uint32_t r = 0; r < reps; r++)
    {
        CHECK_CUDA(cudaEventRecord(a, stream));
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
        CHECK_CUDA(cudaEventRecord(b, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms / iters);
    }
    CHECK_CUDA(cudaEventDestroy(a));
    CHECK_CUDA(cudaEventDestroy(b));
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    return best;
}

inline bool bench_use_graph()
{
    const char *e = getenv("GEMM_BENCH_GRAPH");
    return e && *e && *e != '0';
}

template <class F>
double bench_ms(F &&f, cudaStream_t stream, uint32_t warmup = 5, uint32_t repeat = 20)
{
    if (bench_use_graph())
        return bench_ms_graph(std::forward<F>(f), stream);
    // warmup
    for (uint32_t i = 0; i < warmup; i++)
    {
        f();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events(repeat);
    auto total_time = 0.0f;
    for (uint32_t i = 0; i < repeat; i++)
    {
        CHECK_CUDA(cudaEventCreate(&events[i].first));
        CHECK_CUDA(cudaEventCreate(&events[i].second));
        CHECK_CUDA(cudaEventRecord(events[i].first, stream));
        f();
        CHECK_CUDA(cudaEventRecord(events[i].second, stream));
        sleep_kernel<<<1, 1, 0, stream>>>(100);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (uint32_t i = 0; i < repeat; i++)
    {
        float time;
        CHECK_CUDA(cudaEventElapsedTime(&time, events[i].first, events[i].second));
        total_time += time;
        CHECK_CUDA(cudaEventDestroy(events[i].first));
        CHECK_CUDA(cudaEventDestroy(events[i].second));
    }
    return total_time / repeat;
}