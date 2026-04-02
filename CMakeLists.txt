cmake_minimum_required(VERSION 3.18)
project(sm120_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 120a)

add_executable(bench src/bench.cu)

target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v,-warn-spills
    >
)

target_link_libraries(bench PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)
