#pragma once
#include <cuda.h>
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
template <class F>
double bench_ms(F &&f, cudaStream_t stream, uint32_t warmup = 5, uint32_t repeat = 20)
{
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