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
- **JIT Kernel Compilation**: Kernels are compiled on demand by invoking `nvcc` at runtime and cached on disk, so only the configurations actually used are ever built.
- **Autotuning**: A lightweight autotuner sweeps the kernel configuration space (tile sizes, pipeline stages, warp grouping, split-K) for each problem size, caching results to avoid repeated tuning.

## Project Structure

```
src/
  bf16_gemm.cuh        # Core GEMM kernel: TMA, mbarrier pipeline,
                       # MMA loop, split-K, and launch logic
  common.h             # CUDA error checking, benchmarking utility
  gemm_config.h        # Config parsing/validation, config-space enumeration,
                       # autotune cache
  kernel_jit.h         # On-the-fly nvcc compilation, content-hashed .so cache,
                       # dlopen of compiled kernels
  kernel_entry.cu      # JIT translation unit — instantiates one BF16GemmMMA or
                       # BF16GemmMMASplitK from -D defines
  bench_harness.h      # CUDA buffers, cuBLAS + FP32 references, correctness
                       # checking, L2-flushing timing
  bench.cu             # Command line and the bench/autotune modes
```

## Requirements

- NVIDIA GPU with SM120 (RTX 5090 / RTX Blackwell)
- CUDA Toolkit (12.8+ recommended, must support `sm_120a`)
- CMake 3.18+

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds a single `bench` executable in a few seconds. Kernels are **not** built here — `bench` compiles them on demand with `nvcc` and caches the resulting `.so` files under `build/jit_cache/`. The cache key includes a hash of the kernel sources, so editing `bf16_gemm.cuh` transparently invalidates every previously compiled kernel.

## Usage

```bash
cd build

./bench                              # bench every shape in the autotune cache
./bench --autotune                   # autotune the built-in shape list
./bench --shape M,N,K                # bench one shape with its cached config
./bench --shape M,N,K --autotune     # autotune one shape
./bench --shape M,N,K --config CFG   # compile, check and bench one config
./bench --list-configs               # print the configuration space
```

`--shape` is repeatable. Benching a shape that is not in the cache exits with a hint to autotune it first:

```
$ ./bench --shape 4096,4096,4096
no cached config for M=4096 N=4096 K=4096
hint: bench --shape 4096,4096,4096 --autotune
```

Autotuning iterates every valid config for the shape, checks correctness (against cuBLAS and a naive FP32 reference), benchmarks each, and saves the winner to `autotune_cache.txt`.

Other options: `--cache PATH`, `--jit-cache DIR`, `--jobs N` (parallel `nvcc` jobs), `--max-smem-kb N`, `--warmup N`, `--repeat N`, `--tol F`, `--no-check`, `--force`, `-v`. See `./bench --help`.

## Kernel Configuration Space

A configuration is written as a single string, which is also the name used in the autotune cache and the JIT cache:

```
128x128x64_s3_cwg2_w32x32_sk1
 |          |   |    |      |
 |          |   |    |      +-- SPLIT_K
 |          |   |    +--------- WARP_M x WARP_N
 |          |   +-------------- consumer warp groups (4 warps each)
 |          +------------------ NUM_STAGES
 +----------------------------- BM x BN x BK
```

Tokens may be given in any order and omitted tokens take a default, so `--config` accepts partial specifications:

```bash
./bench --shape 4096,4096,4096 --config 128x64            # bk=64 s=3 cwg=2 sk=1, warp tile derived
./bench --shape 4096,4096,4096 --config 128x64x64_sk2
./bench --shape 4096,4096,4096 --config sk2_cwg1_128x64
```

Defaults are `BK=64`, `NUM_STAGES=3`, `CWG=2`, `SPLIT_K=1`. If the warp tile is omitted, the most square `WARP_M x WARP_N` that exactly covers `BM x BN` with `CWG * 4` warps is chosen.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `BM` | 64, 128 | Tile size along M |
| `BN` | 64, 128 | Tile size along N |
| `BK` | 64 | Tile size along K |
| `NUM_STAGES` | 2, 3, 4 | Pipeline depth (bounded by shared memory) |
| `CWG` | 1, 2 | Consumer warp groups (4 warps each) |
| `WARP_M` | 16–128 | Per-warp tile along M (multiples of 16) |
| `WARP_N` | 8–128 | Per-warp tile along N (multiples of 8) |
| `SPLIT_K` | 1, 2 | K-dimension parallelism factor |

Constraint: `(BM / WARP_M) * (BN / WARP_N) == CWG * 4` (total consumer warps must tile the output block exactly). The sweep enumerates 135 configurations within a 128 KB shared-memory budget; on a device with a 99 KB per-block limit, 122 of them are runnable.

## Results (RTX 5090)

```
=== M=128 N=4096 K=14336  config=64x64x64_s4_cwg2_w32x16_sk1 ===
  cuBLAS    0.1082 ms  138.9397 TFLOPS
  ours      0.1064 ms  141.2754 TFLOPS  (101.7% of cuBLAS)

=== M=128 N=28672 K=4096  config=128x64x64_s3_cwg2_w32x32_sk1 ===
  cuBLAS    0.1906 ms  157.7666 TFLOPS
  ours      0.1909 ms  157.5220 TFLOPS  (99.8% of cuBLAS)

=== M=512 N=512 K=14336  config=64x64x64_s4_cwg2_w16x32_sk2 ===
  cuBLAS    0.0462 ms  162.7333 TFLOPS
  ours      0.0534 ms  140.6263 TFLOPS  (86.4% of cuBLAS)

=== M=1024 N=1024 K=1024  config=64x128x64_s3_cwg2_w16x64_sk1 ===
  cuBLAS    0.0186 ms  115.3965 TFLOPS
  ours      0.0188 ms  114.3544 TFLOPS  (99.1% of cuBLAS)

=== M=1024 N=1024 K=14336  config=128x64x64_s2_cwg2_w32x32_sk1 ===
  cuBLAS    0.1797 ms  167.3434 TFLOPS
  ours      0.1939 ms  155.0498 TFLOPS  (92.7% of cuBLAS)

=== M=2048 N=2048 K=2048  config=64x64x64_s4_cwg2_w16x32_sk1 ===
  cuBLAS    0.1127 ms  152.4357 TFLOPS
  ours      0.1082 ms  158.8000 TFLOPS  (104.2% of cuBLAS)

=== M=4096 N=4096 K=4096  config=128x64x64_s3_cwg2_w32x32_sk1 ===
  cuBLAS    0.7027 ms  195.5783 TFLOPS
  ours      0.6917 ms  198.6907 TFLOPS  (101.6% of cuBLAS)

=== M=4096 N=4096 K=14336  config=64x128x64_s2_cwg2_w64x16_sk1 ===
  cuBLAS    2.3678 ms  203.1534 TFLOPS
  ours      2.3718 ms  202.8156 TFLOPS  (99.8% of cuBLAS)

=== M=4096 N=28672 K=4096  config=128x64x64_s3_cwg2_w16x64_sk1 ===
  cuBLAS    4.4580 ms  215.8105 TFLOPS
  ours      4.4231 ms  217.5125 TFLOPS  (100.8% of cuBLAS)

=== M=8192 N=8192 K=8192  config=64x128x64_s2_cwg2_w32x32_sk1 ===
  cuBLAS    5.0894 ms  216.0408 TFLOPS
  ours      5.0695 ms  216.8886 TFLOPS  (100.4% of cuBLAS)
```

Test cases include both square matrices and **LLaMA 3 8B** shapes (upgate projection 4096×28672×4096, downproj 4096×4096×14336, and their batch-128 variants).
