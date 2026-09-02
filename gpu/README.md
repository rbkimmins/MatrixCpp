# GPU package — placeholder

Nothing here yet. This directory exists so the GPU work has a home alongside
`basic/`, the way MATLAB and NumPy ship GPU support as a separate library
rather than folding it into the core.

## What the machine already has

Measured, not assumed:

- **RTX 5060 Ti**, 16 GB, sm_120 (Blackwell), 36 SMs, driver 580.178.04
- **CUDA 13.2 fully installed** — `nvcc`, cuBLAS, cuSOLVER, cuSPARSE, cuFFT,
  cuRAND, all with dev headers. Nothing needs installing; it is only missing
  from `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## The number that shapes the design

|                          | fp64 (double)      | fp32 (float)          |
|--------------------------|--------------------|-----------------------|
| RTX 5060 Ti (cuBLAS)     | 318–341 GFLOP/s    | 13 900–16 600 GFLOP/s |
| this CPU (`mstore::gemm`)| 180–270 GFLOP/s    | —                     |

GeForce cards throttle FP64 to 1/64 of FP32 — a market decision, not a
technical one. **In double, which is what `basic/` uses everywhere, this GPU is
only 1.3–1.9× the CPU.** Add PCIe (28.7 GB/s pinned, 12.1 GB/s pageable, and a
plain `std::vector` gets the pageable path) and a one-shot `A*B` in double is
roughly break-even.

So the interesting work is not "move GEMM to the GPU". It is one of:

1. **A float / mixed-precision path.** Where the 45× lives. `Matrix<float>`
   already exists but every factorisation promotes to `work_t = double`
   internally, so this is real work and it moves every residual from ~1e-15 to
   ~1e-7.
2. **Device-resident matrices.** A `GpuMatrix` whose results stay on the GPU, so
   PCIe is paid once per *sequence* rather than per operation.
3. **cuSOLVER for the routines that are actually slow.** `schur` runs at 0.9
   GFLOP/s and `svd` at 1.4 — 200× off our own ceiling, against 1.5× for GEMM.
   `Dgesvd` / `Dsyevd` / `Dgeqrf` are already written and tuned, so even
   fp64-throttled hardware should win by a lot here.

(3) is the best first target: the largest real gap, the code already exists, and
it needs no rethink of precision or ownership.

## Constraint to keep

`basic/` is header-only, dependency-free and compiles anywhere. Whatever lands
here must be *optional* — a separate include, so a CPU-only build never sees
CUDA.
