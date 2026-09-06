# The GPU Matrix Package

The device-resident companion to `basic/`, in its own directory for the same
reason MATLAB and NumPy ship GPU support as separate packages: `basic/` stays
header-only, dependency-free and compilable anywhere, and nothing in it ever
includes a CUDA header.

```cpp
#include "basic/MatrixCpp.hpp"
#include "gpu/MatrixGpu.hpp"
using namespace mcpu;                        // bare Matrix<> is the CPU one

Matrix<double> A(4096, 4096);  A.set_Ran_values(0.0, 1.0);

mgpu::Matrix<double> dA = mgpu::upload(A);   // host -> device, once
auto dC = (dA * dA).tanh();                  // stays on the device
Matrix<double> C = dC.cpu();                 // device -> host, once
```

`mcpu::Matrix` and `mgpu::Matrix` are the same name one namespace apart — the
C++ rendering of numpy and cupy, and `namespace np = mcpu; namespace cp = mgpu;`
completes the analogy. Do not `using namespace` both at once: they both export
`Matrix`, so the bare name would be ambiguous.

---

## Why cuBLAS + cuSOLVER, and not CUTLASS

The question that shaped the package was whether to write kernels or front an
existing library, and if fronting, which one. The answer is: front, and front
cuBLAS/cuSOLVER.

**CUTLASS only solves GEMM.** It is a template library for matrix-multiply and
GEMM-shaped work — conv, grouped GEMM, fused epilogues. It has no LU, QR,
Cholesky, SVD or eigenproblem. The measured gap in `basic/` was never GEMM;
GEMM is its *strongest* routine at 180–270 GFLOP/s. It was `schur` at
0.9 GFLOP/s and `svd` at 1.4. CUTLASS covers the one thing already fast and
none of the slow ones.

**For a bare `C = A*B`, cuBLAS is the tuned kernel.** CUTLASS exists for what
cuBLAS will not do: a fused epilogue, an exotic dtype, an unusual shape. For a
plain product its stated goal is to *match* cuBLAS. Choosing it to get a
multiply means paying heavy template-instantiation compile times for parity.

**On this card its advantage is unreachable in double.** CUTLASS's wins live on
Tensor Cores, and fp64 tensor cores (DMMA) exist only on datacenter parts —
A100/H100/B200. sm_120 consumer Blackwell has none, and throttles fp64 to 1/64
of fp32 besides.

**cuBLAS and cuSOLVER are already installed.** CUTLASS is not.

CUTLASS's honest slot is later and optional: a fused *float* path, where
folding a scale/add/activation into the GEMM epilogue saves a memory round
trip. A real win, but step 5, not step 1.

### What each library actually contributed

Worth being precise, because only two of them changed a line of code.

| library | what it gave us |
|---|---|
| **Bandicoot** | the *shape*: a separate GPU package beside a header-only CPU one, same spelling, device-resident matrices, vendor kernels behind a thin backend |
| **CuPy** | two concrete fixes, both found by *measuring against it* rather than reading it — the memory pool, and then kernel fusion |
| **MatX** | the idea we initially dismissed. Its NumPy-style syntax is cosmetic; its real feature is lazy evaluation with fusion, which turned out to be our one genuine deficit |
| **cuBLAS / cuSOLVER** | the kernels. Not an influence — a dependency |
| **CUTLASS** | rejected, for the reasons above |

The division of labour that resulted:

| | |
|---|---|
| cuBLAS | GEMM, transpose, scaled sums |
| cuSOLVER | LU, Cholesky, QR, SVD, symmetric eigenproblem |
| cuRAND | uniform and normal streams |
| **us** | element-wise ops, reductions, shape utilities — ~250 lines |

We write the fourth row only because no vendor ships it. Every one of those
kernels is memory-bound and structurally trivial: one thread per element,
grid-stride loop. There is nothing to tune and nothing subtle to get wrong,
which is exactly why it is the only part worth hand-writing.

---

## The design rule

**Every operation takes device memory and returns device memory.** Nothing
crosses the bus except the explicit `gpu()` / `.cpu()` calls and the scalar
reductions, which have nowhere else to put their answer.

That is not style, it is the entire performance argument, and it is measurable:

```
8-step chain, n = 2048, f64

  device-resident            9.27 ms   (7.16x vs CPU)
  copying every step        60.82 ms   (1.09x vs CPU)  <- the trap
  CPU (basic/)              66.38 ms
```

Same arithmetic, same kernels, same card. A per-operation copy turns a 7x win
into a rounding error, because in double this GPU is only ~1.1–2.4x the tuned
CPU GEMM while PCIe moves 9–22 GB/s.

---

## Fused expressions

The eager API allocates a temporary per step. `(A%B).exp().sqrt()` is three
kernels and seven passes over memory where three would do. Adding `.lazy()`
collapses the chain into one kernel:

```cpp
mgpu::Matrix<double> C = (dA.lazy() % dB).exp().sqrt();
```

Once one operand is lazy the rest of the chain follows, so only the first needs
saying. Non-element-wise operations — matmul, transpose, reductions,
factorisations — terminate a chain; call `.eval()` and carry on.

**How, without shipping a compiler.** MatX and `cupy.fuse` both use NVRTC:
build the source for the exact expression, compile it at run time, cache it.
That would break this package's promise that user code needs no CUDA toolchain.
So the expression templates compile the chain — at C++ compile time — into a
tiny postfix program, and one precompiled kernel interprets it.

```
n = 4096, f64                    eager     fused
(A%B).exp().sqrt()               3.24 ms   2.40 ms   1.35x
6-op chain, 3 inputs             7.58 ms   4.95 ms   1.53x
4 cheap ops, 3 inputs            4.04 ms   1.35 ms   3.00x   <- 399 GB/s of 448
```

The spread across those rows is the interesting part. Fusion removes *memory
traffic*, so it only pays when memory is the limit. The first two chains use
fp64 `exp` and `sqrt`, and on a card that throttles fp64 to 1/64 those
transcendentals — not the memory — are the bottleneck; 1.35x is all that was
available. The cheap-op control has nothing to hide memory behind and gets
3.00x, landing at **89% of the card's theoretical bandwidth**.

Against CuPy's own fused path, which is NVRTC-compiled straight-line code:

```
  op                         n dt        ours      CuPy  ours/CuPy
  fused_chain3            1024 f64        0.16      0.15      1.02x
  fused_chain3            4096 f64        2.40      2.35      1.02x
  fused_cheap             4096 f64        1.35      1.33      1.01x
```

Parity with runtime compilation, from an interpreter. That is not a given — the
first version was **1.58x behind** on the memory-bound chain, because the
interpreter's stack is indexed by a runtime value and therefore lives in local
memory, which is off-chip. Almost every real expression is left-linear and
never needs more than two stack slots, and two slots fit in two registers, with
"which slot" becoming a predicated select rather than an address. Shallow
programs now take a register-only kernel; deeper trees fall back to the stack
machine. That one change took the cheap chain from 256 GB/s to 399.

---

## Views, masks and the rest of the CPU surface

Rows, columns and sub-blocks are addressable the way the CPU proxies are:

```cpp
dA(all, 1) = b;                     // overwrite a column
dA(3, all).set_Ran_values(0, 1);    // refill a row in place
dA.view(4, 3, 3, 2) = patch;        // write a sub-block
Matrix<double> c = dA(all, 2);      // read one out
```

A view is **not** a strided view every kernel understands. It materialises when
read and writes back through `setBlock` when assigned — exactly what the CPU
proxies do. Making every operation stride-aware would touch every kernel in the
package to save a copy that is O(size of the view), not of the matrix.

**Masks are numeric, not boolean.** `lt/le/gt/ge/eq/ne` return 1.0/0.0 in the
matrix's own type, so they compose with everything else: `A % A.gt(0)` zeroes
the negatives, `mask.sum()` counts the hits. A separate bool type would need its
own parallel vocabulary before it was good for anything, and it buys indexing we
do not have. Plus `any`, `all`, `nnz`, `allclose`.

Also added: `var`, `stddev`, `argmax`, `argmin`; `norm(One/Inf/Two/Fro)`;
`trace`, `cond`, `rank`, `pinv`; `reshape` (free — row-major contiguous data
with new dimensions *is* the reshape), `repmat`, `kron`, `fliplr`, `flipud`,
`rot90`, `circshift`; and device-side `linspace`, `logspace`, `range`.

### A disagreement between the two references

`linspace(a, b, 1)` returns **b** in MATLAB and in `basic/`, but **a** in NumPy.
This package follows `basic/`, since one-to-one correspondence with the CPU
library is the whole point. It is the kind of thing only a test catches.

---

## Against CuPy, across sizes

The GPU counterpart of the NumPy comparison in `benchmarks/`, built the same
way: one program times ours, one times CuPy, a third draws the figures — and
every figure is produced by our own C++ plotting package.

```bash
make -C gpu run-bench-cupy          # times both sides, then plots
```

Writes `gpu/bench/data/{gpu,cupy}_<dtype>_<op>.csv` and **one figure per
operation** into `gpu/bench/plots/` — 14 of them, each with two panels:

- **left** — the two absolute time curves, ours and CuPy's, log-log. This is
  the panel that says whether an operation costs a microsecond or a minute,
  which is what decides where to run it at all.
- **right** — their ratio on a log axis, so 10^0 is parity and "twice as fast"
  and "half as fast" sit the same distance from it. Plotted linearly, faster
  would look bigger than slower, which it is not. The region between the curve
  and parity is shaded **green where we are faster and red where CuPy is**, so
  the answer is legible before reading a single number.

  The shading changes colour at the exact crossing, not at the next measured
  size: the curve is split by interpolating in log10 on both axes, which is
  where the drawn line between two points actually sits.

One operation per figure rather than a family per figure: eight curves crowded
onto shared axes hide exactly the divergences the comparison exists to find.
The x axis is ticked at the sizes actually measured and labelled with them,
since a log axis otherwise prints 10^2.107 where 128 belongs.

**What this measures is front-end overhead, not kernels.** Both sides end in
the same cuBLAS, cuSOLVER, cuFFT and CUB routines, so a curve far from parity
means one of us is doing something structurally different — an extra copy, a
layout conversion, a worse launch count. That is the only thing a comparison
like this can honestly detect, and it is worth detecting.

```
op                 n=128    n=512   n=2048   n=4096      (CuPy time / ours)
gemm f64            1.08     1.00     0.96     1.00
gemm f32            1.36     1.08     0.98     0.95
solve               1.06     1.05     1.00     1.00
lu                  1.23     1.03     0.99     0.99
qr                  1.00     0.96     1.00        -
cholesky            0.80     0.76     1.55     1.26
svd                 0.95     0.97        -        -
eigh                1.00     0.99        -        -
eigvals             1.06     0.99        -        -

op                 n=4096   n=65536  n=1048576  n=16777216
add                  1.31      1.31       1.12        1.01
chain                1.67      1.16       0.82        1.00
sum                  1.04      1.42       1.21        1.04
sort                 1.07      1.09       0.82        0.93
fft                  1.12      1.02       0.94        0.84
```

Everything sits between 0.65x and 1.67x, which is the result to want: the C++
front end costs nothing measurable against a mature one. The one systematic
deviation is `cholesky` — slower below n=512, faster above — because our fixed
two-transpose layout cost is a constant only the large sizes amortise.

CuPy has to be installed for the Python half; see the note at the end of this
file.

---

## Sorting

`sort`, `sortrows`, `unique`, `median` and `mode`, with `mcpu`'s signatures and
conventions — stable `sortrows`, ties in `mode` going to the smallest value,
`median` averaging the two middle entries on an even count.

**Correcting something said earlier in this file's history:** Thrust and CUB
are NOT missing from CUDA 13. They moved to `include/cccl/`, which `nvcc` adds
to the include path by itself, so `<thrust/sort.h>` and `<cub/cub.cuh>` need no
extra flags. No hand-written sort network was necessary.

```
                          CPU          GPU
sort 1048576         61.45 ms     0.29 ms    210x
sort 16777216      1222.39 ms     7.29 ms    168x
sort rows 4096²     624.92 ms     6.72 ms     93x
sort cols 4096²    1087.92 ms     8.19 ms    133x
median 4194304       43.32 ms     1.68 ms     26x
```

These are the largest speedups in the package — comparison sorting is
bandwidth-bound and branch-heavy, which is the CPU's worst case and the GPU's
best.

Per-row sorting uses CUB's segmented sort: one launch for the whole matrix
rather than one per row. A **column** sort transposes in and back out, because
a CUB segment has to be contiguous — and it is still *faster* than the row
case relative to the CPU, since two `geam` passes cost far less than the
strided access the CPU pays.

---

## Complex

`mgpu::Matrix<std::complex<double>>` and `<float>` are full citizens — the same
one-to-one correspondence with `mcpu::Matrix` that the namespace split is for.

```cpp
mcpu::Matrix<std::complex<double>> A = /* ... */;
auto dA = mgpu::upload(A);

auto [Q, R] = dA.qr();          // Q is UNITARY: Q^H Q = I
auto [w, V] = dA.eigSym();      // Hermitian: w is REAL, V complex
auto Z = mgpu::fft(dA);         // complex in, complex out
```

**Transfers stay a single memcpy.** `std::complex<T>` is required by the
standard to have the same object representation as `T[2]`, and
`cuDoubleComplex` is a `double2`. So the host matrix, the device buffer, and
what cuBLAS/cuSOLVER expect are all the same bytes — nothing is ever repacked,
and the round-trip test is what proves it.

`T()` is the plain transpose and **`H()` is the conjugate transpose**. For
complex data `H()` is nearly always the one meant — it is what makes `Q^H Q = I`
and `A = U S V^H` true — and for a real matrix `H()` *is* `T()`, so generic code
can always say `H()` and be right. The layout conversion feeding cuSOLVER uses
the plain transpose deliberately: it is a storage change, not a mathematical
one, and conjugating there would silently corrupt every factorisation.

Results that are necessarily real come back as `real_type`: `abs()`, `norm()`,
`svdvals()`, and the Hermitian eigenvalues. `real()`, `imag()`, `arg()` drop to
a real matrix; `conj()` and `H()` stay complex. All of them exist for real
matrices too — as the identity or the plain transpose — so generic code does
not have to branch.

Ordering-based operations are **absent, not faked**: `max`, `min`, `floor`,
`ceil`, `round`, `sign`, `emax`, `emin`. The complex numbers are a field but not
an ordered one, and there is no defensible answer for `floor(1+2i)`. Calling one
on a complex matrix is a compile error naming the alternative.

```
                     CPU          GPU
zgemm 2048       442.95 ms   202.96 ms     2.18x   (338.6 GFLOP/s)
zpotrf 1024       71.32 ms     5.22 ms    13.66x
zheevd 512      3561.74 ms    28.82 ms   123.58x
zfft 1048576      19.59 ms    0.650 ms    30.14x
```

`zgemm` lands at the same ~340 GFLOP/s as the real fp64 GEMM, which is the
expected answer: a complex multiply is 4 real multiplies and 2 adds, so it does
4x the work at the same fp64-throttled rate.

Complex data also takes the **cheaper FFT path**: cuFFT wants interleaved
complex, which is already the storage, so it skips the interleave-and-split that
a real input needs. `Spectrum` remains for real pipelines that would rather keep
the parts separate, and `.toComplex()` bridges the two.

Fused `.lazy()` chains work for complex too, register-only fast path included.

---

## Mixed precision: the answer to the fp64 throttle

This card runs fp64 at 1/64 of fp32. `solveMixed()` factors in fp32 and refines
the solution back with fp64 residual corrections — MAGMA's signature technique,
via `cusolverDnDSgesv` — so the O(n³) work happens where the hardware is fast
and only the O(n²) correction happens where it is slow.

```
              solve()    solveMixed()            residual
n = 1024      5.58 ms        2.50 ms   2.23x     3.3e-15
n = 2048     25.19 ms        5.82 ms   4.33x     4.2e-15
n = 4096    154.63 ms       17.02 ms   9.08x     9.0e-15
```

Two refinement steps in every case, and the residuals are full fp64. This is
not a precision trade — it is the same answer, ~9x faster.

It is also safe by construction: if refinement fails to converge cuSOLVER
redoes the factorisation in fp64 on its own and reports it with a **negative**
iteration count, so the low-precision path can never quietly return a worse
answer. `Factor::Half` and `Factor::BFloat16` factor faster still and converge
on fewer matrices.

**CuPy exposes no mixed-precision solver**, so this is the one place the
package is not merely at parity with it:

```
  solve   n=4096   ours 17.02 ms    CuPy 156.44 ms    9.2x faster
```

## FFT, and factor-once

`signal.hpp` had no GPU counterpart at all; cuFFT supplies it.

```
                          CPU        GPU
fft 1-D  1048576     16.40 ms    0.694 ms    23.6x
fft 1-D    65536      0.82 ms    0.049 ms    16.8x
fft batched 2048²    12.18 ms    1.817 ms     6.7x
conv 200000*20000    61.78 ms    1.754 ms    35.2x
fft 1-D     4096      0.04 ms    0.052 ms     0.79x   <- launch-bound
```

Complex data travels as a `Spectrum` — two real matrices — because
`mgpu::Matrix` is real-only and split arrays are what the rest of the package
can actually operate on. `.cpu()` hands back an
`mcpu::Matrix<std::complex<T>>`, the same type `basic/`'s `fft()` returns.

`factorize()` caches the LU so repeated solves skip the factorisation, matching
`Matrix::factorize()` on the CPU side:

```
10 right-hand sides, n=2048    solve() x10  251.14 ms    factorize + 10  30.63 ms    8.2x
```

## A prediction that did not survive measurement

`tMul()` / `mulT()` / `gram()` pass a transpose flag to cuBLAS instead of
building the transpose. I expected that to matter for tall, skinny operands,
where the O(mn) transpose sits against a product with a small k. It does not:

```
2048x2048 A^T*B     materialise 52.60 ms    flag 52.47 ms    1.00x
2000000x8 gram      materialise 12.21 ms    flag 11.60 ms    1.05x
```

Even at 250,000:1 it is 5%. `geam` is fast enough, and cuBLAS is inefficient
enough at that shape, that the transpose never dominates. The methods stay —
they are clearer at the call site and never slower — but the performance
argument for them was wrong, and it took a measurement to find out.

---

## Measured results

RTX 5060 Ti (sm_120, 36 SMs, 16 GB, 448 GB/s peak), Ryzen 9 7950X, CUDA 13.2,
cuBLAS 13.4.1, cuSOLVER 12.2.0. Both sides `-O3 -march=native`, min of N with a
device sync inside the timed region.

### Where the GPU wins big

| op | n | CPU (`basic/`) | GPU | speedup |
|---|---|---|---|---|
| `eigSym` | 1024 | 10978 ms | 46.5 ms | **236x** |
| `eigSym` | 512 | 1312 ms | 14.4 ms | **91x** |
| `gemm` f32 | 4096 | 439 ms | 8.1 ms | **54x** |
| `gemm` f32 | 2048 | 49.5 ms | 1.1 ms | **45x** |
| element-wise chain | 4096 | 39.6 ms | 3.2 ms | **12.2x** |
| `lu` | 1024 | 40.2 ms | 5.4 ms | **7.5x** |
| `cholesky` | 2048 | 67.2 ms | 10.3 ms | **6.5x** |
| `solve` | 2048 | 94.5 ms | 24.8 ms | 3.8x |
| `qr` | 2048 | 293 ms | 120 ms | 2.4x |

### Where it barely wins, or loses

| op | n | CPU | GPU | speedup |
|---|---|---|---|---|
| `gemm` f64 | 1024 | 7.2 ms | 6.4 ms | 1.12x |
| `gemm` f64 | 4096 | 957 ms | 400 ms | 2.39x |
| `svd` | 1024 | 796 ms | 447 ms | 1.8x |
| `svd` | 256 | 23.8 ms | 28.3 ms | **0.84x** |

Two things drive that column. **fp64 is throttled to 1/64** on a GeForce card,
so double-precision GEMM tops out near 343 GFLOP/s against the CPU's 180–270 —
a market decision, not a technical one. And **small problems are all latency**:
at n=256 the kernel launches and the two layout conversions cost more than the
arithmetic.

Peak rates reached: **343 GFLOP/s fp64**, **17.1 TFLOP/s fp32**. The 50x gap
between those two numbers is the whole story of this hardware.

### Against CuPy — the fairness check

CuPy is NumPy's API over the same cuBLAS/cuSOLVER on the same card, so the
comparison is not "GPU vs CPU" but "does our C++ front-end cost anything
against a mature one". Both end in identical NVIDIA kernels; a gap either way
means someone is marshalling badly.

```
  op                         n dt        ours      CuPy  ours/CuPy
  gemm                    4096 f64      400.47    400.24      1.00x
  gemm                    4096 f32        8.06      8.10      0.99x
  lu                      2048 f64       25.01     25.02      1.00x
  qr                      2048 f64      120.04    119.73      1.00x
  svd                     2048 f64     2681.77   2668.84      1.00x
  eigSym                  1024 f64       46.45     46.83      0.99x
  solve                   2048 f64       24.77     24.91      0.99x
  cholesky                2048 f64       10.32     15.88      0.65x
  (A%B).exp().sqrt()      1024 f64        0.20      0.16      1.22x
```

Parity across the board (0.97–1.02x). Two deviations worth naming honestly:

- **`cholesky` at 2048 we are 1.5x faster**, and at 512 we are 1.3x slower.
  The crossover is our fixed two-transpose layout cost, which is a constant the
  large case amortises and the small case does not.
- **eager element-wise at n=1024 we are 1.22x slower** — about 40 µs, which is
  a few kernel launches, since the eager chain is three launches with three
  temporaries. This is what prompted the fused path above; `.lazy()` closes it
  (1.02x) and beats the eager path by up to 3x.

### PCIe

| | | |
|---|---|---|
| upload | 18.5–21.8 GB/s | pageable `std::vector` path |
| download | 9.4–13.6 GB/s | |

Nothing here is free. A single `A*B` in double that has to cross both ways is
roughly break-even against the CPU.

---

## What is not here, and why

**A correction, and it was ours.** An earlier version of this file reported
that `cusolverDnXgeev` was broken on this build, having "tested it three ways".
It is not broken. For a real matrix the eigenvector arrays take **real** data
types, not complex, because cuSOLVER returns them in LAPACK's packed form —
only `dataTypeW` is complex. Passing complex for `dataTypeVL`/`dataTypeVR`
gives `CUSOLVER_STATUS_INVALID_VALUE`, which we read as a library fault rather
than a usage error; the three tests were three variations of one wrong
assumption. Comparing against CuPy — which calls the same symbol in the same
`libcusolver.so.12.2.0.11` successfully — is what surfaced it.

`eig()` and `eigvals()` are available, with `mcpu`'s contracts: `eigvals()`
always works and returns complex; `eig()` returns the real pair and refuses a
complex spectrum rather than dropping the imaginary part.

**Check the size before reaching for it:**

```
n        eigvals cpu    eigvals gpu    ratio     eigSym gpu
128           10.5 ms        18.5 ms    0.57x        3.1 ms
256          109.5 ms        37.6 ms    2.91x        5.7 ms
512         1319.3 ms        97.5 ms   13.52x       11.8 ms
1024       11453.1 ms       297.5 ms   38.49x       31.6 ms
```

Below n≈200 the CPU wins outright, and if the matrix is symmetric `eigSym()` is
9x faster again at n=1024. The QR sweep behind a non-symmetric eigensolver is
sequential and shift-dependent — close to the worst shape for this hardware —
while the symmetric problem has a divide-and-conquer algorithm that maps onto
it properly.

**Still CPU-only:** `schur`, `hess`, `funm`, `qz`, `rref`, `null`, `orth`.

**No integral `mgpu::Matrix`.** `float`, `double`, `complex<float>` and
`complex<double>` instantiate; integers stay on the CPU.

**No pivoted QR.** cuSOLVER exposes no pivoted `geqrf`, so unlike
`Matrix::QR()` there is no permutation to return.

---

## Two bugs worth not repeating

**A hand-rolled memory pool is a trap.** Pooling was added because the
element-wise chain measured 2.1x slower than CuPy for kernels doing identical
work — `cudaMalloc` talks to the driver and costs tens of microseconds
regardless of size, so at n=1024 three temporaries cost more than three
kernels. The first version was a hand-written free list. It produced the right
speedup and then failed `cusolverDnDgesvd` with an internal error, because:

> **`cudaFree` implicitly synchronises the device. A pool does not.**

Launches are asynchronous, so when a scratch buffer's destructor runs its
kernel may still be executing. `cudaFree` waits, by accident. A naive pool
hands the block straight to the next allocation and two live kernels write the
same memory. `cudaMallocAsync`/`cudaFreeAsync` are stream-ordered and solve it
properly, in about a hundred lines less code.

**Stream-ordered means ordered against *a* stream.** The replacement used
`cudaStreamPerThread`, which looks more modern and is wrong here: cuBLAS,
cuSOLVER and `cudaMemcpy` all run on the *legacy* default stream because no
stream is ever set on the handles. Ordering allocations against a stream no
work runs on is the same as no ordering at all, and it showed up as
uninitialised `info` values out of cuSOLVER. Stream `0` throughout.

---

## Building

CUDA 13.2 is fully installed on this machine — `nvcc`, cuBLAS, cuSOLVER,
cuSPARSE, cuFFT, cuRAND, all with dev headers. It is only missing from `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

```bash
make -C gpu              # libmatrixcpp_gpu.a
make -C gpu run-test     # 306 checks against basic/
make -C gpu run-bench    # timings, writes gpu/test/results/cpp.csv
```

**Your own code needs no CUDA compiler.** Every public header here is ordinary
C++17; only `src/backend.cu` sees `nvcc`, and only when the library is built.

```bash
g++ -std=c++17 -O3 -march=native -fopenmp yours.cpp \
    -Lgpu -lmatrixcpp_gpu -L/usr/local/cuda/lib64 \
    -lcudart -lcublas -lcusolver -lcurand -lcufft -o yours
```

### CuPy, for the comparison column

System Python is externally managed and `python3-venv` is not installed, so
CuPy lives in its own directory rather than in site-packages:

```bash
pip3 install --target=~/.local/lib/matrixcpp-cupy cupy-cuda13x
PYTHONPATH=~/.local/lib/matrixcpp-cupy python3 gpu/test/cupy_timings.py
```

Setting `PYTHONPATH` is what opts in. Without it, `benchmarks/numpy_timings.py`
keeps seeing numpy 1.26.4 + scipy 1.11.4 exactly as before — CuPy pulls numpy
2.5.2, which would otherwise break the existing NumPy validation. CuPy prints a
harmless numpy-1.x/2.x warning at import when it probes for scipy.

---

## Layout

```
MatrixGpu.hpp        umbrella
device.hpp           which GPU, how much memory, is there one at all
detail/backend.hpp   declarations of everything CUDA implements — pure C++17
src/backend.cu       the only file nvcc sees
gpu_matrix.hpp       mgpu::Matrix<T> and mgpu::Expr<T> (the fused chains)
test/correctness.cpp 306 checks, all against basic/
test/benchmark.cpp   CPU vs GPU timings
test/cupy_timings.py CuPy timings + the three-way table
Makefile
```

## Verification

- **306 / 306** correctness checks against `basic/`, which is itself validated
  against NumPy/SciPy — so agreement here is transitively agreement with them.
- **0 errors** under `compute-sanitizer --tool memcheck`. The 8 reported
  "leaks" are the cuBLAS/cuSOLVER/cuRAND handles, deliberately never destroyed:
  a static destructor racing CUDA's teardown at exit is a known source of
  phantom crashes, and the driver reclaims everything on process exit anyway.
- Factorisations are checked by **residual** (`Q*R == A`, `L*L^T == A`,
  `A*V == V*diag(w)`) rather than against the CPU's factors, because factors
  are only unique up to signs and column order while residuals are what callers
  actually depend on.
