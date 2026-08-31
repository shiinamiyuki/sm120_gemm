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
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-warn-spills>
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
# Like `bench`, the kernels themselves are JIT-compiled from
# src/fp8_kernel_entry.cu rather than built here.
add_executable(bench_fp8 src/bench_fp8.cu)

target_compile_options(bench_fp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-warn-spills>
)

target_compile_definitions(bench_fp8 PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(bench_fp8 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    dl
)

# MXFP8 (OCP microscaling) baseline: cuBLASLt with VEC32_UE8M0 block scaling,
# plus the probe that pins down the scale-factor tensor layout it expects.
add_executable(bench_mxfp8 src/bench_mxfp8.cu)

target_compile_options(bench_mxfp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-warn-spills>
)

target_compile_definitions(bench_mxfp8 PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(bench_mxfp8 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    dl
)

# W4A16: bf16 activations x NVFP4 weights. No cuBLASLt path exists for mixed
# widths, so the baseline here is the bf16 GEMM it replaces.
add_executable(bench_w4a16 src/bench_w4a16.cu)

target_compile_options(bench_w4a16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-warn-spills>
)

target_compile_definitions(bench_w4a16 PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(bench_w4a16 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    dl
)

# Device memory bandwidth, by access path. Provides the denominator for every
# "% of peak" claim about the GEMM kernels -- the spec-sheet number is not it.
add_executable(bench_bandwidth src/bench_bandwidth.cu)

target_compile_options(bench_bandwidth PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(bench_bandwidth PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

# W4A4: NVFP4 activations x NVFP4 weights. sm_120a has a native block-scaled
# e2m1 MMA, so this family needs no dequantize step -- but the operand and
# scale fragment layouts have to be probed before a kernel can use it.
add_executable(bench_w4a4 src/bench_w4a4.cu)

target_compile_options(bench_w4a4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-warn-spills>
)

target_compile_definitions(bench_w4a4 PRIVATE
    GEMM_NVCC="${CMAKE_CUDA_COMPILER}"
    GEMM_SRC_DIR="${CMAKE_SOURCE_DIR}/src"
    GEMM_JIT_CACHE_DIR="${KERNEL_JIT_CACHE_DIR}"
    GEMM_CUDA_STUB_DIR="${CUDAToolkit_LIBRARY_DIR}/stubs"
    GEMM_CUDA_ARCH="${GEMM_CUDA_ARCH}"
)

target_link_libraries(bench_w4a4 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    dl
)
