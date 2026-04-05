# A Zero-dependency GEMM library for RTX 50 series (SM120). 100% cuBLAS performance.

A high-performance BF16 GEMM (General Matrix Multiply) implementation targeting NVIDIA RTX Blackwell (SM120) GPUs, written entirely in CUDA C++ with inline PTX. Achieves **100%+ of cuBLAS performance** on large matrix sizes, with no dependencies beyond the CUDA toolkit.

Computes $Y = X \cdot W^T$ where $X$ is $(M \times K)$ BF16, $W$ is $(N \times K)$ BF16, and $Y$ is $(M \times N)$ BF16, with accumulation in FP32.

## Key Optimizations
- **TMA (Tensor Memory Accelerator)**: Uses `cp.async.bulk.tensor.2d` for both global→shared loads and shared→global stores, offloading data movement to a dedicated hardware unit.
- **Warp Specialization**: Dedicated producer warp group issues TMA loads while consumer warp groups execute MMA instructions, overlapping memory and compute.
- **Multi-stage Software Pipeline**: Configurable 2–4 stage pipeline with `mbarrier`-based synchronization between producer and consumer warp groups.
- **Persistent Kernel**: Each kernel is launched with exactly #SM blocks and loops over tiles, avoiding repeated kernel launch overhead and reducing pipeline bubbles.
- **Tensor Core via `mma.sync.aligned.m16n8k16`**: BF16→FP32 matrix-multiply-accumulate using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
- **Split-K**: For small-M problems, the K dimension is partitioned across multiple CTAs, with a separate reduction kernel to sum FP32 partial results.
- **128B Shared Memory Swizzling**: TMA descriptors use `CU_TENSOR_MAP_SWIZZLE_128B` and shared memory access uses matching swizzle indexing to eliminate bank conflicts.
- **Swizzled Tile Rasterization**: Output tiles are visited in a swizzled order (configurable `SWIZZLE_WIDTH`) to improve L2 cache locality.
- **`ldmatrix` / `stmatrix`**: Warp-cooperative shared memory loads (`ldmatrix.sync.aligned.m8n8.x4`) and stores (`stmatrix.sync.aligned.m8n8.x2`) for efficient MMA fragment movement.
- **L2-aware Benchmarking**: Buffer rotation flushes L2 between iterations for accurate throughput measurement.
- **Autotuning**: A lightweight autotuner sweeps the kernel configuration space (tile sizes, pipeline stages, warp grouping, split-K) for each problem size, caching results to avoid repeated tuning.

## Project Structure

```
src/
  bf16_gemm.cuh       # Core GEMM kernel: TMA, mbarrier pipeline,
                       # MMA loop, split-K, and launch logic
  common.h             # CUDA error checking, benchmarking utility
  gemm_config.h        # Runtime config parsing, kernel registry (.so loader),
                       # autotune cache, and dispatch
  kernel_entry.cu.in   # CMake template — instantiates one BF16GemmMMA or
                       # BF16GemmMMASplitK per configuration
  bench.cu             # Benchmarking harness: correctness checks vs cuBLAS
                       # and FP32 reference, autotuning, quick bench
scripts/
  gen_configs.py       # Generates all valid (BM, BN, BK, stages, CWG,
                       # WARP_M, WARP_N, SPLIT_K) configs as a CMake include
```

## Requirements

- NVIDIA GPU with SM120 (RTX 5090 / RTX Blackwell)
- CUDA Toolkit (12.8+ recommended, must support `sm_120a`)
- CMake 3.18+
- Python 3 (for config generation)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This builds:
- **135 kernel shared libraries** (`build/kernels/libgemm_*.so`), one per tile/warp configuration
- **`bench`** executable for benchmarking and autotuning

## Usage

### Quick Bench (uses cached autotune results)

```bash
cd build
./bench
```

Runs each test case with the previously autotuned best kernel and compares against cuBLAS.

### Autotune

```bash
./bench --autotune
```

Iterates all valid kernel configs for each problem size, checks correctness (vs cuBLAS and an FP32 reference), benchmarks each, and saves the best to `autotune_cache.txt`.

### Debug Mode

```bash
./bench --debug
```

Runs a fixed kernel config with detailed correctness output (max absolute/relative error vs cuBLAS and FP32 reference) for every test case.

## Kernel Configuration Space

Each kernel is parameterized by:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `BM` | 64, 128 | Tile size along M |
| `BN` | 64, 128 | Tile size along N |
| `BK` | 64 | Tile size along K |
| `NUM_STAGES` | 2, 3, 4 | Pipeline depth (bounded by 128 KB shared memory) |
| `CWG` | 1, 2 | Consumer warp groups (4 warps each) |
| `WARP_M` | 16–128 | Per-warp tile along M (multiples of 16) |
| `WARP_N` | 8–128 | Per-warp tile along N (multiples of 8) |
| `SPLIT_K` | 1, 2 | K-dimension parallelism factor |

Constraint: `(BM / WARP_M) * (BN / WARP_N) == CWG * 4` (total consumer warps must tile the output block exactly).

## Results (RTX 5090)

```
=== M=128 N=4096 K=14336  config=bm64_bn64_bk64_s4_cwg2_wm32_wn16_sk1 ===
  cuBLAS:  0.1087 ms  138.3443 TFLOPS
  Ours:    0.1073 ms  140.0707 TFLOPS  (101.2% of cuBLAS)

=== M=128 N=28672 K=4096  config=bm128_bn64_bk64_s3_cwg2_wm32_wn32_sk1 ===
  cuBLAS:  0.1927 ms  156.0399 TFLOPS
  Ours:    0.1935 ms  155.3831 TFLOPS  (99.6% of cuBLAS)

=== M=512 N=512 K=14336  config=bm64_bn64_bk64_s4_cwg2_wm16_wn32_sk2 ===
  cuBLAS:  0.0482 ms  156.0516 TFLOPS
  Ours:    0.0549 ms  136.9449 TFLOPS  (87.8% of cuBLAS)

=== M=1024 N=1024 K=1024  config=bm64_bn128_bk64_s3_cwg2_wm16_wn64_sk1 ===
  cuBLAS:  0.0186 ms  115.6651 TFLOPS
  Ours:    0.0185 ms  115.9749 TFLOPS  (100.3% of cuBLAS)

=== M=1024 N=1024 K=14336  config=bm128_bn64_bk64_s2_cwg2_wm32_wn32_sk1 ===
  cuBLAS:  0.1808 ms  166.2683 TFLOPS
  Ours:    0.1947 ms  154.4279 TFLOPS  (92.9% of cuBLAS)

=== M=2048 N=2048 K=2048  config=bm64_bn64_bk64_s4_cwg2_wm16_wn32_sk1 ===
  cuBLAS:  0.1131 ms  151.9224 TFLOPS
  Ours:    0.1101 ms  156.0240 TFLOPS  (102.7% of cuBLAS)

=== M=4096 N=4096 K=4096  config=bm128_bn64_bk64_s3_cwg2_wm32_wn32_sk1 ===
  cuBLAS:  0.7043 ms  195.1504 TFLOPS
  Ours:    0.6961 ms  197.4462 TFLOPS  (101.2% of cuBLAS)

=== M=4096 N=4096 K=14336  config=bm64_bn128_bk64_s2_cwg2_wm64_wn16_sk1 ===
  cuBLAS:  2.3613 ms  203.7162 TFLOPS
  Ours:    2.3593 ms  203.8907 TFLOPS  (100.1% of cuBLAS)

=== M=4096 N=28672 K=4096  config=bm128_bn64_bk64_s3_cwg2_wm16_wn64_sk1 ===
  cuBLAS:  4.4774 ms  214.8751 TFLOPS
  Ours:    4.4283 ms  217.2546 TFLOPS  (101.1% of cuBLAS)

=== M=8192 N=8192 K=8192  config=bm64_bn128_bk64_s2_cwg2_wm32_wn32_sk1 ===
  cuBLAS:  5.1118 ms  215.0943 TFLOPS
  Ours:    5.0508 ms  217.6914 TFLOPS  (101.2% of cuBLAS)
```

Test cases include both square matrices and **LLaMA 3 8B** shapes (upgate projection 4096×28672×4096, downproj 4096×4096×14336, and their batch-128 variants).
