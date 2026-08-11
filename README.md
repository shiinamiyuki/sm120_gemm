# A Zero-dependency GEMM library for RTX 50 series (SM120). 100% cuBLAS performance.

A high-performance BF16 GEMM (General Matrix Multiply) implementation targeting NVIDIA RTX Blackwell (SM120) GPUs, written entirely in CUDA C++ with inline PTX. Achieves **100%+ of cuBLAS performance** on large matrix sizes, with no dependencies beyond the CUDA toolkit.

Computes $Y = X \cdot W^T$ where $X$ is $(M \times K)$ BF16, $W$ is $(N \times K)$ BF16, and $Y$ is $(M \times N)$ BF16, with accumulation in FP32.

## Key Optimizations
- **TMA (Tensor Memory Accelerator)**: Uses `cp.async.bulk.tensor.2d` for both global→shared loads and shared→global stores, offloading data movement to a dedicated hardware unit.
- **Warp Specialization**: Dedicated producer warp group issues TMA loads while consumer warp groups execute MMA instructions, overlapping memory and compute.
- **Multi-stage Software Pipeline**: Configurable 2–4 stage pipeline with `mbarrier`-based synchronization between producer and consumer warp groups.
- **Persistent Kernel**: Each kernel is launched with exactly #SM blocks and loops over tiles, avoiding repeated kernel launch overhead and reducing pipeline bubbles.
- **Tensor Core via `mma.sync.aligned.m16n8k16`**: BF16→FP32 matrix-multiply-accumulate using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
- **Split-K**: For small-M problems, the K dimension is partitioned across multiple CTAs, with a separate reduction kernel to sum FP32 partial results. Split-K is a compile-time parameter of the same kernel and changes only the epilogue: `SPLIT_K == 1` stages BF16 through shared memory and TMA-stores it, while `SPLIT_K > 1` writes FP32 partials straight to a workspace.
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
  bf16_gemm_tinym.cuh  # Tiny-M variant (M <= 8): same pipeline, CUDA-core
                       # consumer instead of tensor cores
  common.h             # CUDA error checking, benchmarking utility
  gemm_config.h        # Config parsing/validation, config-space enumeration,
                       # autotune cache
  kernel_jit.h         # On-the-fly nvcc compilation, content-hashed .so cache,
                       # dlopen of compiled kernels
  kernel_entry.cu      # JIT translation unit — instantiates one BF16GemmMMA or
                       # BF16GemmTinyM from -D defines
  bench_harness.h      # CUDA buffers, cuBLAS + FP32 references, correctness
                       # checking, L2-flushing timing
  bench.cu             # Command line and the bench/autotune modes
  proto_tinym.cu       # Standalone driver for the tiny-M kernel: synthetic
                       # correctness probes and a fixed variant sweep
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

### Kernel families

Configs come in two families, both sharing the producer / TMA / mbarrier pipeline and differing only in the consumer. A leading `tinym` token selects the CUDA-core kernel, which has no warp tile:

```
128x128x64_s3_cwg2_w32x32_sk1      tensor cores (BF16GemmMMA),   needs M % BM == 0
tinym_8x256x64_s2_cwg1_sk8         CUDA cores  (BF16GemmTinyM),  for M <= BM
```

The tiny-M family targets skinny GEMMs (M ≤ 8), where `mma.m16n8k16` would waste ≥ 50% of every instruction and the problem is bandwidth-bound anyway. Each thread owns all `BM` rows and `BN / (CWG*128)` columns of n, accumulating in registers with plain FP32 FMAs. Split-K matters far more here: with no M-parallelism, `N/BN` tiles alone leave most SMs idle.

Both families are enumerated into the same search space, and `--autotune` picks between them automatically — `fits_shape` keeps tiny-M configs out of the running for anything but `M <= BM`, so a large-M sweep never pays to compile them. Kernels that register-spill are rejected at compile time from the ptxas report and never benchmarked, since they have already lost.

| Parameter | Tensor core | Tiny-M | Description |
|-----------|-------------|--------|-------------|
| `BM` | 64, 128 | 4, 8 | Tile size along M |
| `BN` | 64, 128 | 128, 256 | Tile size along N |
| `BK` | 64 | 64 | Tile size along K |
| `NUM_STAGES` | 2, 3, 4 | 2, 3, 4 | Pipeline depth (bounded by shared memory) |
| `CWG` | 1, 2 | 1, 2 | Consumer warp groups (4 warps each) |
| `WARP_M` | 16–128 | — | Per-warp tile along M (multiples of 16) |
| `WARP_N` | 8–128 | — | Per-warp tile along N (multiples of 8) |
| `SPLIT_K` | 1, 2 | 1, 2, 4, 8, 16 | K-dimension parallelism factor |

Constraint: `(BM / WARP_M) * (BN / WARP_N) == CWG * 4` (total consumer warps must tile the output block exactly). The tensor-core sweep enumerates 135 configurations within a 128 KB shared-memory budget; on a device with a 99 KB per-block limit, 122 of them are runnable, plus 50 tiny-M configurations (`BM` ∈ {4, 8}, `BN` ∈ {128, 256}, `SPLIT_K` up to 16).

## Results (RTX 5090)

Sorted by M. The decode-sized rows (M <= 8) run the tiny-M CUDA-core kernel; everything from M=128 up runs the tensor-core kernel. `--autotune` chooses between the families on its own. Note that TFLOPS is the wrong yardstick for the small-M rows: they are bandwidth-bound at roughly 1.5 TB/s, so the percentage of cuBLAS is what matters.

```
=== M=1 N=4096 K=14336  config=tinym_4x256x64_s2_cwg1_sk8 ===
  cuBLAS    0.0764 ms    1.5377 TFLOPS
  ours      0.0760 ms    1.5461 TFLOPS  (100.5% of cuBLAS)

=== M=1 N=28672 K=4096  config=tinym_4x256x64_s2_cwg1_sk4 ===
  cuBLAS    0.1416 ms    1.6593 TFLOPS
  ours      0.1452 ms    1.6176 TFLOPS  (97.5% of cuBLAS)

=== M=4 N=4096 K=14336  config=tinym_4x256x64_s2_cwg1_sk8 ===
  cuBLAS    0.0787 ms    5.9677 TFLOPS
  ours      0.0772 ms    6.0839 TFLOPS  (101.9% of cuBLAS)

=== M=4 N=28672 K=4096  config=tinym_4x256x64_s2_cwg1_sk4 ===
  cuBLAS    0.1565 ms    6.0042 TFLOPS
  ours      0.1460 ms    6.4346 TFLOPS  (107.2% of cuBLAS)

=== M=8 N=4096 K=14336  config=tinym_8x256x64_s2_cwg1_sk8 ===
  cuBLAS    0.0919 ms   10.2197 TFLOPS
  ours      0.0783 ms   12.0048 TFLOPS  (117.5% of cuBLAS)

=== M=8 N=28672 K=4096  config=tinym_8x256x64_s2_cwg1_sk1 ===
  cuBLAS    0.1566 ms   11.9966 TFLOPS
  ours      0.1493 ms   12.5884 TFLOPS  (104.9% of cuBLAS)

=== M=128 N=4096 K=14336  config=64x128x64_s4_cwg2_w32x32_sk2 ===
  cuBLAS    0.1068 ms  140.7970 TFLOPS
  ours      0.1047 ms  143.6427 TFLOPS  (102.0% of cuBLAS)

=== M=128 N=28672 K=4096  config=128x64x64_s3_cwg2_w32x32_sk1 ===
  cuBLAS    0.1905 ms  157.7878 TFLOPS
  ours      0.1907 ms  157.6859 TFLOPS  (99.9% of cuBLAS)

=== M=512 N=512 K=14336  config=64x64x64_s3_cwg2_w16x32_sk2 ===
  cuBLAS    0.0460 ms  163.3160 TFLOPS
  ours      0.0532 ms  141.2733 TFLOPS  (86.5% of cuBLAS)

=== M=1024 N=1024 K=1024  config=128x64x64_s2_cwg2_w32x32_sk1 ===
  cuBLAS    0.0185 ms  116.1656 TFLOPS
  ours      0.0185 ms  116.0952 TFLOPS  (99.9% of cuBLAS)

=== M=1024 N=1024 K=14336  config=128x128x64_s3_cwg2_w64x32_sk2 ===
  cuBLAS    0.1796 ms  167.4448 TFLOPS
  ours      0.1909 ms  157.5273 TFLOPS  (94.1% of cuBLAS)

=== M=2048 N=2048 K=2048  config=64x64x64_s4_cwg2_w32x16_sk1 ===
  cuBLAS    0.1127 ms  152.4833 TFLOPS
  ours      0.1087 ms  158.0822 TFLOPS  (103.7% of cuBLAS)

=== M=4096 N=4096 K=4096  config=64x64x64_s4_cwg2_w16x32_sk1 ===
  cuBLAS    0.7027 ms  195.5997 TFLOPS
  ours      0.6905 ms  199.0350 TFLOPS  (101.8% of cuBLAS)

=== M=4096 N=4096 K=14336  config=64x128x64_s3_cwg2_w32x32_sk2 ===
  cuBLAS    2.2388 ms  214.8659 TFLOPS
  ours      2.3702 ms  202.9486 TFLOPS  (94.5% of cuBLAS)

=== M=4096 N=28672 K=4096  config=128x64x64_s2_cwg2_w64x16_sk1 ===
  cuBLAS    4.4766 ms  214.9109 TFLOPS
  ours      4.4260 ms  217.3669 TFLOPS  (101.1% of cuBLAS)

=== M=8192 N=8192 K=8192  config=64x128x64_s2_cwg2_w64x16_sk1 ===
  cuBLAS    5.0885 ms  216.0768 TFLOPS
  ours      5.0765 ms  216.5878 TFLOPS  (100.2% of cuBLAS)
```

Test cases include both square matrices and **LLaMA 3 8B** shapes — upgate projection (N=28672, K=4096) and downproj (N=4096, K=14336) — swept across batch sizes from 4096 down to the decode-time M=1.
