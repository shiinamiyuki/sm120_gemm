cmake_minimum_required(VERSION 3.18)
project(sm120_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(GEMM_CUDA_ARCH "120a")
set(CMAKE_CUDA_ARCHITECTURES ${GEMM_CUDA_ARCH})

# Kernels are not built here — `bench` compiles them on demand by invoking
# nvcc on src/kernel_entry.cu (see src/kernel_jit.h) and caches the resulting
# .so files under KERNEL_JIT_CACHE_DIR.
set(KERNEL_JIT_CACHE_DIR "${CMAKE_BINARY_DIR}/jit_cache")

add_executable(bench src/bench.cu)

target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_compile_definitions(bench PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(bench PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    dl
)

# Prototype driver for the tiny-M CUDA-core kernel. Unlike the tensor-core
# kernels this one is compiled straight into the binary, not JIT'd.
add_executable(proto_tinym src/proto_tinym.cu)

target_compile_options(proto_tinym PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-v,-warn-spills>
)

target_compile_definitions(proto_tinym PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(proto_tinym PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    dl
)

# FP8 (e4m3) per-tensor-scaled GEMM: harness, FP32 reference, cuBLASLt baseline.
add_executable(bench_fp8 src/bench_fp8.cu)

target_compile_options(bench_fp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(bench_fp8 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    dl
)
