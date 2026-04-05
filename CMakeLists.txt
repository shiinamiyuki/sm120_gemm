cmake_minimum_required(VERSION 3.18)
project(sm120_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 120a)

# ── Generate kernel configs ─────────────────────────────────────────────
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
execute_process(
    COMMAND python3 "${CMAKE_SOURCE_DIR}/scripts/gen_configs.py"
        --output "${GENERATED_DIR}/configs.cmake"
        --max-smem-kb 128
    RESULT_VARIABLE GEN_RESULT
)
if(NOT GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "gen_configs.py failed with code ${GEN_RESULT}")
endif()
include("${GENERATED_DIR}/configs.cmake")

# ── Kernel shared libraries directory ───────────────────────────────────
set(KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/kernels")
file(MAKE_DIRECTORY "${KERNEL_OUTPUT_DIR}")

# Common CUDA flags for kernel .so's
set(KERNEL_CUDA_FLAGS --use_fast_math --ptxas-options=-v,-warn-spills)

foreach(cfg IN LISTS GEMM_CONFIGS)
    # Parse: "BM|BN|BK|NUM_STAGES|CWG|WARP_M|WARP_N|SPLIT_K|name"
    string(REPLACE "|" ";" cfg_list "${cfg}")
    list(GET cfg_list 0 BM)
    list(GET cfg_list 1 BN)
    list(GET cfg_list 2 BK)
    list(GET cfg_list 3 NUM_STAGES)
    list(GET cfg_list 4 CWG)
    list(GET cfg_list 5 WARP_M)
    list(GET cfg_list 6 WARP_N)
    list(GET cfg_list 7 SPLIT_K)
    list(GET cfg_list 8 CFG_NAME)

    set(KERNEL_SRC "${GENERATED_DIR}/kernels/gemm_${CFG_NAME}.cu")
    configure_file("${CMAKE_SOURCE_DIR}/src/kernel_entry.cu.in" "${KERNEL_SRC}" @ONLY)

    set(LIB_NAME "gemm_${CFG_NAME}")
    add_library(${LIB_NAME} SHARED "${KERNEL_SRC}")
    target_include_directories(${LIB_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_options(${LIB_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${KERNEL_CUDA_FLAGS}>
    )
    target_link_libraries(${LIB_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(${LIB_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${KERNEL_OUTPUT_DIR}"
        PREFIX "lib"
    )
endforeach()

# ── bench executable ────────────────────────────────────────────────────
add_executable(bench src/bench.cu)

target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v,-warn-spills
    >
)

target_compile_definitions(bench PRIVATE
    KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
)

target_link_libraries(bench PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    dl
)
