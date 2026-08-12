# A Zero-dependency GEMM library for RTX 50 series (SM120). 100% cuBLAS performance.

A high-performance GEMM (General Matrix Multiply) implementation targeting NVIDIA RTX Blackwell (SM120) GPUs, written entirely in CUDA C++ with inline PTX. Achieves **100%+ of cuBLAS performance** on large matrix sizes, with no dependencies beyond the CUDA toolkit.

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
  fp8_gemm.cuh         # FP8 (e4m3) tensor-core kernel: bf16 kernel structure
                       # with mma.m16n8k32.e4m3 and per-tensor scaling
  fp8_gemm_tinym.cuh   # FP8 tiny-M variant (M <= 8), CUDA-core consumer
  fp8_kernel_entry.cu  # JIT translation unit for the FP8 kernels
  fp8_cublaslt.h       # Per-tensor-scaled FP8 (e4m3) GEMM via cuBLASLt
  fp8_harness.h        # FP8 quantization model, FP32 reference kernels,
                       # accuracy metrics, Fp8Problem
  bench_fp8.cu         # FP8 command line and the bench/autotune modes
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

This builds the `bench` (bf16) and `bench_fp8` (e4m3) executables in a few seconds. Kernels are **not** built here — both drivers compile them on demand with `nvcc` and cache the resulting `.so` files under `build/jit_cache/`. The cache key includes a hash of the kernel sources, so editing `bf16_gemm.cuh` transparently invalidates every previously compiled kernel.

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

Configs come in two families, both sharing the producer / TMA / mbarrier pipeline and differing only in the consumer. A leading `tinym` token selects the CUDA-core kernel, which has no warp tile. A leading `fp8` token selects the e4m3 element type — implied by `bench_fp8`, which only builds fp8 kernels, so it may be omitted there:

```
128x128x64_s3_cwg2_w32x32_sk1      tensor cores (BF16GemmMMA),   needs M % BM == 0
tinym_8x256x64_s2_cwg1_sk8         CUDA cores  (BF16GemmTinyM),  for M <= BM
fp8_128x128x128_s2_cwg2_w64x32     tensor cores (FP8GemmMMA)
fp8_tinym_4x256x128_s2_cwg1_sk16   CUDA cores  (FP8GemmTinyM)
```

The element type changes everything derived from the element size: the smallest `BK` that fills a 128-byte swizzled smem row (64 for bf16, 128 for fp8), the MMA's k extent (16 vs 32), and the tiny-M vector-load width (8 vs 16 elements — the *byte* width stays 16). The default `BK` follows from it, so `--config 128x128` means `BK=64` under `bench` and `BK=128` under `bench_fp8`.

The tiny-M family targets skinny GEMMs (M ≤ 8), where `mma.m16n8k16` would waste ≥ 50% of every instruction and the problem is bandwidth-bound anyway. Each thread owns all `BM` rows and `BN / (CWG*128)` columns of n, accumulating in registers with plain FP32 FMAs. Split-K matters far more here: with no M-parallelism, `N/BN` tiles alone leave most SMs idle.

Both families are enumerated into the same search space, and `--autotune` picks between them automatically — `fits_shape` keeps tiny-M configs out of the running for anything but `M <= BM`, so a large-M sweep never pays to compile them. Kernels that register-spill are rejected at compile time from the ptxas report and never benchmarked, since they have already lost.

| Parameter | Tensor core | Tiny-M | Description |
|-----------|-------------|--------|-------------|
| `BM` | 64, 128 | 1, 2, 4, 8 | Tile size along M |
| `BN` | 64, 128 | 128, 256 | Tile size along N |
| `BK` | 64 (bf16) / 128 (fp8) | 64 (bf16) / 128 (fp8) | Tile size along K |
| `NUM_STAGES` | 2–5 | 2–5 | Pipeline depth (bounded by shared memory) |
| `CWG` | 1, 2 | 1, 2 | Consumer warp groups (4 warps each) |
| `WARP_M` | 16–128 | — | Per-warp tile along M (multiples of 16) |
| `WARP_N` | 8–128 | — | Per-warp tile along N (multiples of 8) |
| `SPLIT_K` | 1, 2 (bf16); 1 (fp8) | 1, 2, 4, 8, 16 | K-dimension parallelism factor |

Constraint: `(BM / WARP_M) * (BN / WARP_N) == CWG * 4` (total consumer warps must tile the output block exactly). `BM` below 4 in the tiny-M family only ever matches `M` of 1 or 2, but at those shapes the padding rows are pure waste — measurably so, see the ncu notes below. `FP8GemmMMA` has no split-K epilogue, so fp8 tensor-core configs are pinned to `SPLIT_K=1`; tiny-M covers the skinny shapes that would want one. On a device with a 99 KB per-block limit the search space is 254 bf16 configurations and 177 fp8 ones.

## FP8 (e4m3), per-tensor scaling

A harness, an FP32 reference, the cuBLASLt baseline, and two hand-written
kernels: `FP8GemmMMA` (tensor cores) and `FP8GemmTinyM` (CUDA cores, M ≤ 8).

`bench_fp8` has the same modes as `bench`, over the same JIT and autotune
machinery — the only difference is the element type, which it fixes to e4m3:

```bash
./bench_fp8                          # bench every shape in the autotune cache
./bench_fp8 --autotune                # autotune the built-in shape list
./bench_fp8 --shape M,N,K --autotune  # autotune one shape
./bench_fp8 --shape M,N,K --config C  # compile, check and bench one config
./bench_fp8 --list-configs            # print the configuration space
```

plus the fp8-specific diagnostics:

```bash
./bench_fp8 --quant-error    # also report what fp8 quantization costs
./bench_fp8 --normal         # gaussian inputs (--outlier adds 1000x outliers)
./bench_fp8 --probe-layout   # verify the m16n8k32 fragment layout, then exit
```

The winner per shape is cached in `autotune_cache_fp8.txt`, separately from the
bf16 cache.

Quantization model, per tensor:

```
scale = amax(real) / 448        # 448 = largest finite e4m3
code  = e4m3(real / scale)      # saturating, round-to-nearest-even
real ~= scale * float(code)
```

so the GEMM over codes is scaled by `x_scale * w_scale` — the single product
cuBLASLt applies through its A/B scale pointers, leaving `alpha = 1`. cuBLASLt
requires FP8 matmuls in "TN" form, which the existing layout already satisfies:
W is passed as A with `op = T` and X as B with `op = N`, exactly as the bf16
path does.

The reference is a naive FP32 GEMM **over the quantized codes**, so it is the
exact value the FP8 GEMM should produce — this isolates GEMM error from
quantization error. cuBLASLt matches it to **0.39%**, which is precisely bf16
output rounding (2⁻⁸) and is constant across every shape and distribution.
`--quant-error` separately reports quantization error against a full-precision
reference.

Two things worth knowing before designing a kernel:

- **Per-tensor e4m3 is robust to dynamic range.** Output RMS-relative error is
  3.61% for uniform inputs, 3.75% for gaussian, and only 3.81% with 0.1% of
  elements scaled 1000×. e4m3 is a *floating point* format holding ~3 mantissa
  bits across 2⁻⁶…448, so unlike per-tensor int8, raising amax costs nothing
  until values fall through into subnormals. Per-block scaling has to be
  motivated by something other than plain outliers.
- **Accuracy is reported as RMS-relative**, `||got-ref||₂ / ||ref||₂`, not max
  elementwise relative error. Inputs and outputs both contain values arbitrarily
  close to zero, and a max-relative metric is entirely determined by those — it
  reads ~1.0 however good the quantizer is.

### The FP8 kernel

`FP8GemmMMA` is the bf16 tensor-core kernel with the consumer swapped to an
fp8 MMA. Everything else — persistent CTAs, producer warp group, TMA/mbarrier
pipeline, warp tiling, epilogue — is unchanged.

| | fp8 `m16n8k32` | bf16 `m16n8k16` |
|---|---|---|
| A fragment | 4 regs x **4 fp8** along k | 4 regs x 2 bf16 |
| B fragment | 2 regs x **4 fp8** along k | 2 regs x 2 bf16 |
| C/D fragment | 4 x fp32 — **identical** | 4 x fp32 |
| k per instruction | **32** | 16 |

Because a register holds 4 bytes either way, fragment **addressing is
unchanged in bytes** — the same `ldmatrix.b16` loaders work on fp8
reinterpreted as b16 pairs; only per-lane k offsets count elements instead.
`BK` is pinned to 128 so an smem row is 128 bytes for the swizzle. The exact
per-lane layout is documented in `fp8_gemm.cuh` and checked by
`--probe-layout`.

**The MMA is block-scaled.** The obvious instruction,
`mma.m16n8k32...f32.e4m3.e4m3.f32`, runs at *half* the tensor-core issue rate
on sm_120 — 350 TFLOPS against a 700 TFLOPS ceiling, and visible in ncu as a
tensor pipe 97% busy while the HMMA subpipe sits at 48%. The block-scaled
form runs at the full rate **and still accumulates in fp32**:

| instruction | TFLOPS |
|---|---|
| `mma.m16n8k32` e4m3 -> f32 | 350 |
| **`QMMA.SF` block-scaled -> f32** (what cuBLAS issues) | **700** |
| `mma.m16n8k32` e4m3 -> f16 | 700 |

So there is no reason to trade precision for speed here: block-scaling gets
the same 2x that f16 accumulation would, with fp32 accumulate intact. **The
entire 2x comes from the instruction itself, not from the scaling** — cuBLAS
uses it exactly this way, feeding the MMA a neutral 2^0 scale (its SASS
literally does `MOV R5, 0x7f7f7f7f`) and applying the real scale afterwards.

The `ue8m0` scale operands only express powers of two, so an arbitrary
per-tensor scale is split as `scale = m * 2^e`: the `2^e` rides along in the
MMA for free and only the mantissa residual `m` reaches the epilogue as one
multiply. Any positive scale is accepted. That multiply is free in practice —
removing it entirely (by forcing a power-of-two scale) measured no faster,
because 32 FMULs per output tile are nothing against the k-loop.

Because all four scale bytes carry the same value under per-tensor scaling,
the scale-fragment layout does not matter yet — real per-block scaling will
have to pin it down.

> **Build gotcha:** this instruction is architecture-specific.
> `nvcc -arch=sm_120a` silently forwards `-arch=compute_120` to ptxas,
> dropping the `a`, and the instruction is rejected. Both the CMake build and
> `kernel_jit.h` therefore emit `-gencode arch=compute_120a,code=sm_120a`.

### The FP8 tiny-M kernel

`FP8GemmTinyM` is `BF16GemmTinyM` with the operand type swapped. The work
decomposition, pipeline and epilogue are identical; three things move:

- **`BK` 64 → 128**, so an smem row is still 128 bytes for the swizzle.
- **`VEC` 8 → 16 elements.** What has to stay fixed is the *byte* width: 16-byte
  accesses put eight consecutive `n` on eight distinct banks, covering all 32
  exactly once. Sixteen fp8 is sixteen bytes, so only the element count changes.
- **The widening is no longer free.** bf16 → f32 is a shift; e4m3 → f32 needs
  `cvt.rn.f16x2.e4m3x2` then f16 → f32.

That last point is the one that shows up in the numbers. Per k-chunk a thread
converts `BM*VEC` X elements against `BM*NPT*VEC` FMAs, so the (fully redundant
— every lane widens the same broadcast X) conversion cost is `1/NPT` of the FMA
cost. At `NPT=1` it is roughly 1.5× the FMA count, and `BN=256/CWG=1` (`NPT=2`)
measures **27% faster** than `BN=128/CWG=1` at K=14336.

An ncu profile at M=1 (`--set full`, N=28672 K=4096) says the rest is simply
memory:

| | duration | DRAM | SM | L2 hit | warp cyc/inst |
|---|---|---|---|---|---|
| cuBLASLt `nvjet…64x32x64…bz_TNNN` | 106.3 µs | 65.0% | 10.1% | 50.1% | 25.2 |
| `tinym_4x128x128_s4_cwg1_sk16` | 94.8 µs | **72.6%** | 38.0% | 3.4% | 5.0 |

cuBLAS picks a 64×32×64 tile with split-k 4, so at M=1 **63 of every 64 A rows
are padding** — hence its 50% L2 hit rate (re-reading the same padded tile) and
stalls dominated by `long_scoreboard` 39.6% + `mio_throttle` 19.8%. It throttles
its own memory pipes on work that produces nothing. Our 3.4% hit rate means W is
fetched once and never touched again, and 117.4 MB in 94.8 µs is 1.24 TB/s of
irreducible traffic.

Our own padding waste is real but small — at M=1, `BM` 8 → 4 → 2 → 1 measures
1144 → 1186 → 1217 → 1228 GB/s, with SM throughput falling from 60% to 38%.
That is why the tiny-M sweep enumerates `BM` down to 1. Two remaining overheads
the profile exposes: `splitk_reduce_kernel` costs **3.7% of total** (it writes
1.83 MB of fp32 partials to DRAM and reads them back to produce 57 KB of
output — an atomic epilogue would delete it), and `barrier` is 38% of stall
cycles because 3 of 8 warps are the producer group's idle warps, parked at
`__syncthreads()` while only lane 0 of warp 0 issues TMA.

### FP8 results (RTX 5090, autotuned)

Ours vs cuBLASLt, same run so clocks match. TFLOPS for the compute-bound
shapes:

| shape | cuBLASLt | ours | | config |
|---|---|---|---|---|
| 256^3 | 5.41 | 5.38 | 0.99x | `64x64x128_s3_cwg2_w64x8` |
| 1024^3 | 168.9 | 184.0 | 1.09x | `64x128x128_s3_cwg2_w32x32` |
| 4096^3 | 489.2 | 497.9 | 1.02x | `128x128x128_s2_cwg2_w64x32` |
| 8192^3 | 454.5 | **624.9** | **1.37x** | `128x128x128_s2_cwg2_w32x64` |
| 4096x28672x4096 | 435.4 | **614.8** | **1.41x** | `128x128x128_s2_cwg2_w32x64` |
| 4096x4096x14336 | 427.3 | 540.6 | 1.27x | `128x128x128_s2_cwg2_w32x64` |
| 128x28672x4096 | 217.0 | 294.3 | 1.36x | `128x64x128_s3_cwg1_w32x64` |
| 128x4096x14336 | 212.5 | 261.6 | 1.23x | `64x64x128_s5_cwg2_w16x32` |

624.9 is 89% of the 700 TFLOPS instruction ceiling.

The decode-sized rows are bandwidth work, so GB/s is the yardstick:

| shape | cuBLASLt | ours | | config |
|---|---|---|---|---|
| 8x28672x4096 | 1237 | 1117 | 0.90x | `tinym_8x256x128_s2_cwg1_sk16` |
| 8x4096x14336 | 1092 | 1064 | 0.97x | `tinym_8x256x128_s2_cwg1_sk8` |
| 4x28672x4096 | 1116 | 1223 | 1.10x | `tinym_4x128x128_s2_cwg1_sk2` |
| 4x4096x14336 | 1116 | 1113 | 1.00x | `tinym_4x128x128_s3_cwg1_sk4` |
| 1x28672x4096 | 1110 | **1248** | **1.12x** | `tinym_1x128x128_s3_cwg1_sk2` |
| 1x4096x14336 | 1121 | 1148 | 1.02x | `tinym_1x128x128_s2_cwg1_sk8` |

The two M=8 rows are the only shapes where cuBLASLt still wins. Note how little
the winning split-K resembles a rule of thumb — `sk2` for M=4 at N=28672 but
`sk8` for M=1 at N=4096 — which is the argument for autotuning rather than
hand-picking.

Known remaining inefficiency: our epilogue stages the tile through shared
memory (`stmatrix` into `Y_out`, then a TMA store), costing ~1.9M
shared-memory store bank conflicts per 4096^3 GEMM. cuBLAS has zero, but not
because it swizzles `Y_out` better — its SASS shows 768 `STG` and zero
`STS`/`STSM`, i.e. it skips shared memory in the epilogue entirely and stores
straight to global. Dropping the smem staging would also free the `Y_out`
buffer, which is what currently caps `128x128` at 2 pipeline stages; cuBLAS
fits 6 stages of `128x128x64` in 96 KB precisely because it has no `Y_out`.

### cuBLASLt FP8 baseline (RTX 5090)

| shape | TFLOPS | GB/s | vs bf16 cuBLAS |
|---|---|---|---|
| 1024³ | 204 | 399 | 1.8× |
| 4096³ | 542 | 265 | 2.8× |
| 8192³ | 579 | 141 | 2.7× |
| 4096×28672×4096 (upgate) | 529 | 203 | 2.5× |
| 4096×4096×14336 (downproj) | 520 | 163 | 2.6× |
| 128×28672×4096 | 250 | 1043 | 1.6× |
| 128×4096×14336 | 264 | 1082 | 1.9× |

The M=128 rows are bandwidth-bound (~1.05 TB/s), not compute-bound.

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
