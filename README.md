# MatrixCpp

A header-only matrix library in C++17, with no third-party dependencies —
everything is built on the standard library alone. Currently expanding towards
ML and quantum circuits, and anything else that turns out to be interesting or
useful.

```cpp
#include "Matrix1.0.hpp"

Matrix<double> A(3, 3);
A = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};

Matrix<double> b(3, 1);
b = {{8}, {-11}, {-3}};

auto x = A.solve(b);        // prefer this over A.inverse() * b
std::cout << x << '\n';
```

## Building

The library is a single header — just `#include "Matrix1.0.hpp"`. It requires
**C++17** (it uses `if constexpr`, structured bindings and inline variables).

```
g++ -std=c++17 -O3 -march=native -ffast-math -fopenmp -o Running_time Running_time.cpp
```

`-fopenmp` is optional; the header guards its OpenMP use with `#ifdef _OPENMP`
and works without it.

> **`-ffast-math` and complex numbers.** `-ffast-math` implies
> `-fcx-limited-range`, which switches complex multiply and divide to the naive
> textbook formulas with no range reduction. Complex division can then overflow
> or underflow on operands well inside `double`'s range. Drop `-ffast-math`, or
> add `-fno-cx-limited-range`, before trusting complex results or benchmarks.
>
> `-ffast-math` also implies `-ffinite-math-only`, which lets the compiler assume
> infinities never occur — `std::isinf()` folds to a constant `false`. `cond()`
> still *returns* infinity for a singular matrix, but you cannot detect it with
> `isinf` under this flag; compare against a threshold instead.

## Operators, and how they map to MATLAB

The library follows MATLAB where C++ allows it. `A / B` is **matrix right
division** — the `X` solving `X * B = A` — not element-wise division, and it is
computed as a solve rather than by forming `inv(B)`.

| MATLAB | here | |
|---|---|---|
| `A * B` | `A * B` | matrix product |
| `A .* B` | `A % B`, `A.emul(B)`, `A *dot* B` | element-wise product |
| `A / B` | `A / B` | right division, `X*B = A` |
| `A ./ B` | `A.ediv(B)`, `A /dot/ B` | element-wise division |
| `A \ B` | `A.solve(B)` | left division — C++ has no `operator\` |
| `A ^ n` | `pow(A, n)` | matrix power — `^` has the wrong precedence in C++ |
| `A .^ n` | `A.pow(n)` | element-wise power |
| `A'` | `A.H()` | conjugate transpose |
| `A.'` | `A.T()` | plain transpose |

C++ cannot define an operator with a leading dot, so the element-wise family is
named (`emul`, `ediv`) with `%` and the `*dot*` / `/dot/` spellings as sugar.
`A /dot/ B` parses as `(A / dot) / B` — `*` and `/` share a precedence level and
associate left to right — so it composes correctly with surrounding `+` and `-`.

## The naming convention

This is the single most important thing to know about the API. Whether a
function is a **member** or a **free function** decides what it computes:

| Spelling | Meaning | Example |
|---|---|---|
| `A.f()` | **element-wise** — applies `f` to each entry independently | `A.exp()`, `A.sin()`, `A.pow(2)` |
| `f(A)`  | **matrix-wise** — the true matrix function | `exp(A)`, `sin(A)`, `pow(A, 2)` |

`A.exp()` and `exp(A)` are completely different matrices. The first
exponentiates each entry; the second sums `I + A + A²/2! + …`. They agree only
when `A` is diagonal.

Operators follow the same idea and are **element-wise**, with one exception:

- `A * B` is **matrix multiplication** (Strassen-Winograd accelerated).
- `A % B` is the Hadamard (element-wise) product, `A / B` element-wise division.
- `A | B` concatenates horizontally; use parentheses, as `|` binds loosely.
- `A(i, j)`, `A(i, all)`, `A({r1,r2}, {c1,c2})` are indexing and slicing, never
  arithmetic.

## What's implemented

**Decompositions** — `LU`, `QR` (Householder with column pivoting), `cholesky`,
`schurDecomp`, `svd` (one-sided Jacobi), `eig`, `eigvals`.

**Solving** — `solve` (square via LU; over-determined via least-squares QR),
`inverse`, `pinv`, `adjugate`.

**Scalars** — `det`, `tr`, `norm` (Frobenius / 1 / ∞ / 2), `rank`, `cond`.

**Reductions** — `sum`, `min`, `max`, `mean`, `var`, `stddev`, `argmin`,
`argmax`; each with a scalar form and a per-row/per-column form.

**Structural** — `T`, `H`, `conj`, `real`, `imag`, `diag`, `triu`, `tril`,
`reshape`, `concat`, `tensor` (Kronecker).

**Matrix functions** — `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
`sqrt`, `pow`, `log`.

### Controlling the Taylor series

`exp`, `sin`, `cos`, `sinh`, `cosh` (and `tan`/`tanh`, built on them) evaluate a
Taylor series, and every one accepts an optional trailing `TaylorOpts`:

```cpp
exp(A);                     // library defaults
exp(A, 25);                 // at most 25 terms
exp(A, {25, 1e-12});        // ... or stop early once terms fall below 1e-12
exp(A, {25, 1e-12, false}); // ... and disable scaling-and-squaring
```

By default the argument is scaled until its norm is small, the series runs until
the terms stop contributing, and the result is recovered by repeated squaring
(`exp`) or double-angle identities (trig). Disabling scaling is a study aid, not
a faster path — it is what keeps the series accurate for large `‖A‖`.

## Tests

Two suites, checking different things.

**`validate.cpp`** — tests the library against *itself*: identities, invariants
and hand-computed values. Exits non-zero on any failure.

```
g++ -std=c++17 -O2 -fopenmp -o validate validate.cpp && ./validate
```

**`numpy_validate.cpp` + `numpy_validate.py`** — tests it against an
*independent implementation*. The C++ half runs every operator on fixed inputs
and dumps inputs and results to `validation/*.dat`; the Python half recomputes
each one in NumPy/SciPy and compares. This catches the class of error where a
convention is self-consistent but not what the rest of the world means by that
operation.

```
g++ -std=c++17 -O2 -fopenmp -o numpy_validate numpy_validate.cpp
./numpy_validate && python3 numpy_validate.py
```

Where a result is unique (add, multiply, det, inverse, solve, `expm`, …) the
comparison is element by element. Where it is not — QR, LU, SVD and eig all
admit sign flips, column reorderings and different pivoting — it instead checks
what *is* well defined: that the factors reconstruct `A` under a NumPy product,
that they have the structure they claim, and that the spectrum matches NumPy's
once sorted.

`test.cpp` is the older demonstration program — it prints results for a human to
read rather than checking them.

## Benchmarks

Three pieces, sharing one data contract in `bench/`:

```
g++ -std=c++17 -O3 -march=native -fopenmp -o Running_time Running_time.cpp
./Running_time                       # -> bench/cpp_<dtype>_<op>.csv
python3 NumpyRunningtime.py          # -> bench/numpy_<dtype>_<op>.csv, then plots/
julia --project=. plot_running_time.jl   # -> plots/julia/
```

Every operation is timed for **both** `Matrix<double>` and
`Matrix<complex<double>>`. The Python script does not choose its own sizes — it
reads them back out of the C++ CSVs, so both implementations are measured at
exactly the same `n`, with no interpolation and no mismatched ranges.

Output:

| file | what it shows |
|---|---|
| `plots/summary_<dtype>.png` | every operation on one bar chart, sorted by speedup |
| `plots/ops/<dtype>_<op>.png` | times (log-log) beside the speedup ratio |
| `plots/julia/scaling_<dtype>.png` | measured exponent *k* in time ∝ n^k, against the textbook value |

Speedup is always **NumPy time ÷ MatrixCpp time**: above 1 means MatrixCpp is
faster (green), below means NumPy is (red).

Individual benchmarks are toggled by the `#define BENCH_*` lines at the top of
`Running_time.cpp`. Complex covers the container layer only — `det`, `inverse`,
`LU`, `QR`, `eig`, `pow` and `log` are still real-only.

### Where it stands

Measured on a 16-core / 64 MiB-L3 machine. At each operation's **largest** size,
50 of 55 are faster than NumPy (median 2.3x); averaged over the **upper half of
the size range**, 54 of 55 (median 3.1x). Both numbers come from the CSVs the
scripts above write, so you can recompute either.

| operation | vs NumPy | what changed |
|---|---|---|
| `transpose` (real) | 22x | blocked + threaded, thread count capped to physical cores |
| `elem_pow` (complex) | 18x | NumPy's ufuncs are single-threaded; ours are not |
| `elem_ln` (complex) | 15x | same |
| `qr` | 8.7x | column-major working array, threaded reflector application |
| `multiply` (real) | 8x | blocked, `__restrict`, OpenMP |
| `mat_pow_int` | 5.2x | binary exponentiation over the fast multiply |
| `inverse` | 4.6x | right-hand sides solved in parallel |
| `svd` | 3.0x | column-major Jacobi, cached norms, Brent-Luk parallel ordering |
| `hadamard` (real) | 2.5x | threaded, and no longer zero-filling a buffer it overwrites |
| `sum(axis)` | 1.4x | NumPy's own pairwise summation |
| `cholesky` | 1.4x | raw-pointer inner loops (was four integer divisions deep) |

Still behind at the largest size: `eig` (0.81x) and `pow(A, real)` (0.85x), which
want LAPACK's blocked multishift QR (`dlaqr0`); complex `add`/`subtract`
(~0.92x) and `reshape` (0.95x), which are at the DRAM roofline where there is
nothing left to win. `Matrix1.0.hpp` documents each one and what it would take.

### Reading the results

* **Run it on a quiet machine.** These are microbenchmarks, and a contended run
  can be off by orders of magnitude at small `n` — enough to make the curves
  meaningless. If a curve looks flat where it should be rising, re-run. The
  harness defends against this with three warm-ups and two independent
  measurement bursts per point, keeping the better; without that, a single
  scheduler stall could publish a figure 5-20x too high.
* **Mind the cache cliff at the largest size.** A 2000x2000 `double` matrix is
  32 MB, so a one-input/one-output real operation at `n = 2000` has a 64 MB
  working set — exactly this machine's L3. The complex version, at 128 MB, does
  not fit. Scalar multiply measures 771 GB/s at `n = 2000` and 34 GB/s at
  `n = 3000`. That cliff, not anything about complex arithmetic, is why the real
  element-wise speedups look so much larger than the complex ones. Check where
  your own L3 boundary falls before reading too much into the last point.
* **Strassen-Winograd is compiled out by default.** Measured against the same
  blocked naive multiply, it was slower at every size tested (0.43x at n=128
  down to 0.11x at n=1024): the recursion is serial while `naiveMul` is
  OpenMP-parallel across every core, and each level heap-allocates about twenty
  temporaries. Build with `-DMATRIXCPP_ENABLE_STRASSEN` to opt back in; the code
  and its tests are still there. See the note in `Matrix1.0.hpp`.
* Timing is best-of-k with an adaptive repeat count, matched on both sides. The
  minimum is used rather than the mean: the true cost is a floor, and noise only
  ever pushes a sample above it.
