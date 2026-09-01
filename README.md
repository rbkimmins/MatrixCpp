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

**The member dot is the element-wise marker.** `A.sin()` is element-wise,
`sin(A)` is the matrix function. `A.pow(n)` is element-wise, `pow(A, n)` is the
matrix power. That is MATLAB's leading dot, moved to where C++ can hold it — and
it is why the element-wise members are `mul` and `div` rather than `emul` and
`ediv`: being a member has already said "element-wise", so the `e` would say it
twice. It also keeps them the same three-letter shape as `sin`, `cos`, `exp`,
`abs` and `pow`.

`A / B` is **matrix right division** — the `X` solving `X * B = A` — not
element-wise division, and it's computed as a solve rather than by forming
`inv(B)`.

| MATLAB | here | |
|---|---|---|
| `A * B` | `A * B` | matrix product |
| `A .* B` | `A.mul(B)`, `A % B`, `A *dot* B` | element-wise product |
| `A / B` | `A / B` | right division, `X*B = A` |
| `A ./ B` | `A.div(B)`, `A /dot/ B` | element-wise division |
| `A \ B` | `A.solve(B)` | left division — C++ has no `operator\` |
| `A ^ n` | `pow(A, n)` | matrix power — `^` has the wrong precedence in C++ |
| `A .^ n` | `A.pow(n)` | element-wise power |
| `sin(A)` | `sin(A)` | matrix function (free) |
| element-wise `sin` | `A.sin()` | element-wise (member) |
| `A'` | `A.H()` | conjugate transpose |
| `A.'` | `A.T()` | plain transpose |
| `kron(A,B)` | `A.kron(B)`, `kron(A,B)` | Kronecker — no operator, deliberately |

`kron` keeps a name rather than getting an operator (`%` was considered) because
operators should go to frequent, cheap operations. Hadamard is O(n²) and
everywhere; Kronecker is O(n⁴) and rare — two 1000×1000 matrices produce 10¹²
elements, 8 TB. That cost should be visible at the call site. It's also why the
benchmark takes `kron` to n = 48 while everything else runs to 512 or 2000.

`T()` and `H()` are members that aren't element-wise, which bends the rule — but
harmlessly, since there's no such thing as an element-wise transpose for them to
be confused with.

For the operator spellings, `A /dot/ B` parses as `(A / dot) / B` — `*` and `/`
share a precedence level and associate left to right — so it composes correctly
with surrounding `+` and `-`.

## FFT

```cpp
fft(x)        fft(x, n)        fft(A, n, addcol)     // MATLAB semantics
ifft(X)       fftshift(X)      ifftshift(X)
```

A vector transforms along its own length, a matrix column by column, `n` pads or
truncates, and the whole `1/n` sits on the inverse — all as MATLAB does, which is
what makes the results checkable against `numpy.fft` rather than merely
self-consistent.

**Correct for every length.** Radix-2 where `n` is a power of two, Bluestein's
chirp-z otherwise. The tempting shortcut — zero-padding up to the next power of
two — computes the transform of a *different, longer* signal: right for a
convolution, silently wrong for a spectrum. So it isn't done.

| | vs NumPy |
|---|---|
| one 1-D transform, n = 2048 … 2²⁰ | 0.43× – 0.89× |
| **1024 × 64 columns** | **1.98×** |
| **1024 × 512 columns** | **2.59×** |

NumPy uses pocketfft — mixed-radix with hand-written codelets for radices
2/3/4/5/7/11. A single-radix-2 kernel won't beat that on one transform. But
pocketfft doesn't *thread*, and transforming the columns of a matrix is perfect
parallelism, so that case is simply not available to it.

## Sequences, shape and test matrices

```cpp
linspace(0, 1, 101)   logspace(-1, 2, 7)   range(0, 9, 2)     // a:step:b
hilb(5)  pascal(4)  wilkinson(7)  magic(4)  toeplitz(c)  vander(v)
randn(m, n, -seed)    randi(1, 6, m, n, -seed)   randperm(n, -seed)

A.numel()  A.repmat(2,3)  A.fliplr()  A.flipud()  A.rot90(-1)
A.circshift(1, 0)  A.blkdiag(B)
```

Sequences go through **`std::iota`**, the standard library's own sequence
generator, so the index run is exact and the only floating point in a `linspace`
is the single multiply that scales it — accumulating `v += step` would drift, and
drift further the longer the vector. `linspace` also **sets** its endpoint rather
than computing it, because `a + i*(b-a)/(n-1)` doesn't reliably land on `b`.

Every random constructor uses **`ran2` from `random.hpp`**, never `std::rand`.
Seeds stay negative, matching `set_Ran_values`. `randn` is Box–Muller, `randperm`
is Fisher–Yates. ⚠ `ran2` keeps static state, so these are deliberately **serial**
— the one place the OpenMP treatment the rest of the header gets would be wrong.

## General matrix functions and generalized eigenvalues

```cpp
funm(A, [](std::complex<double> z){ return 1.0/(1.0 + z); })   // any callable
eig(A, B)        // symmetric-definite:  A x = lambda B x
eigvals(A, B)    // general pencil, guarded
polyeig({A0, A1, A2})                                          // matrix polynomial
```

`funm` is **numeric, not symbolic** — `f` is a lambda called on complex numbers;
nothing differentiates or expands it. That boundary has a price and it's stated
rather than hidden: the robust Schur–Parlett algorithm needs `f'`, `f''` to
handle clustered eigenvalues, so this one **detects** that case and throws,
naming the two eigenvalues. `exp`, `log`, `sqrt`, `sin`, `cos`, `sinh`, `cosh`,
`tanh` and `pow` have dedicated implementations that don't go through Parlett and
have no separation requirement at all — the error message says so.

`eig(A, B)` returns eigenvalues ascending and eigenvectors that are
**B-orthonormal** (`XᵀBX = I`), which is the right normalisation for this problem
and what LAPACK's `dsygv` gives.

`eigvals(A, B)` reduces via `B⁻¹A` and is **guarded by a condition estimate** — it
refuses a near-singular `B` outright rather than returning plausible numbers, and
names QZ as what would be needed. QZ itself isn't implemented; that's the one
remaining gap in this tier.

## Factor once, solve many times

`A.solve(b)` factors `A` from scratch every call — right for a one-off, wrong in
a loop. `factorize()` returns a reusable object:

```cpp
auto dA = A.factorize();          // or decomposition(A), MATLAB's spelling
Matrix<double> x1 = dA.solve(b1);
Matrix<double> x2 = dA.solve(b2);
double d = dA.det();              // free from the factors already held
```

| 100 right-hand sides | `A.solve()` each time | factor once | |
|---|---|---|---|
| n = 256 | 104 ms | **3.1 ms** | 33× |
| n = 512 | 810 ms | **16.3 ms** | 50× |

It picks the factorisation from the **structure**, not from a flag you have to
get right — symmetric positive definite → Cholesky (half the flops, and the
attempt *is* the definiteness test), square otherwise → LU, rectangular →
column-pivoted QR. `kindName()` reports which. The object owns its factors, so it
outlives the matrix it came from.

```cpp
rcond(A)      condest(A)          // Hager–Higham 1-norm estimate
lsqminnorm(A, b)                  // minimum-norm least squares
```

`rcond`/`condest` estimate the condition number from those same factors in a
handful of solves rather than forming an inverse — 9.6 ms against 302 ms for a
full `cond(Two)` at n = 512. It's a **lower** bound, never pessimistic; on a 5×5
Hilbert matrix it lands on the true value exactly.

## Reductions and scans

```cpp
A.sum()   A.prod()   A.mean()   A.median()   A.mode()   A.min()   A.max()
A.var()   A.stddev()  A.argmin()  A.argmax()  A.nnz()
A.sum(false)      // per column        A.sum(true)      // per row
A.cumsum(false)   A.cumprod(true)      // scans — same shape as the input
A.diff(false)     // (rows-1 x cols)   A.diff(true)     // (rows x cols-1)
A.sort(false)     A.sort(true, /*descending=*/true)
A.sortrows(0)     A.unique()
```

One convention throughout: **`false` works down columns, `true` along rows** — so
`prod(false)` pairs with `sum(false)`, and `cumsum(false)` accumulates down the
axis `sum(false)` totals. Scans keep the input shape; `diff` shrinks the scanned
axis by one; `median` returns `double` because an even count averages the middle
two. `prod`/`cumsum`/`cumprod` work for complex; anything that has to *order*
elements refuses to compile for it, the same as `min`/`max`.

## Logical masks

Comparisons produce a `Matrix<bool>`, which is an ordinary matrix — it gets
shape, printing, `T()` and the rest for free.

```cpp
Matrix<double> A(2, 3);  A = {{-1, 2, -3}, {4, -5, 6}};

A > 0.0            A <= B           A.eq(B)      A.ne(0.0)     // masks
m1 && m2           m1 || m2         m1 ^ m2      !m1           // combine
m1.land(m2)        m1.lor(m2)       m1.lxor(m2)  m1.lnot()     // same, named
A.any()   A.all()   A.nnz()   A.find()                          // reduce
A.any(false)                                                    // per column

Matrix<double> pos = A(A > 0.0);   // read  -> column vector, row-major
A(A < 0.0) = 0.0;                  // write through the mask
A(A > 0.0) = repl;                 // one value per selected element
```

**`<` `>` `<=` `>=` are element-wise; `==` and `!=` are not.** That's not an
oversight. The rule is that the member dot marks element-wise *where both
meanings exist* — and for the orderings only one meaning exists, since matrices
have no ordering for `A < B` to be confused with. For `==` the other meaning very
much does exist, and `if (A == B)` is the idiom every C++ programmer reaches for,
so the operator keeps whole-matrix equality and `.eq()` is the element-wise form.
This is the one place the comparison family diverges from MATLAB.

**Logic is spelled in C: `&&`, `||`, `!`** — not NumPy's and MATLAB's `&` and
`|`, because `|` isn't available: `A | B` is the augmented-matrix operator here.
Taking `&` for *and* while *or* had to be a named function would have left the
pair lopsided, so the whole triple goes to C syntax instead. ⚠ The trap is still
there for anyone typing from NumPy habit — `(A > 0) | (B > 0)` compiles and
quietly returns a mask of twice the width. Use `||`.

`^` is C's exclusive-or, so that one *is* the language's own spelling. Its low
precedence is an argument against ever using `^` for a **power** — `A * B ^ 2`
would silently group as `(A*B) ^ 2`, which is why the matrix power here is
`pow(A, n)` and never an operator. For xor the precedence is harmless and in fact
convenient: relational operators bind tighter, so `A > 0 ^ A > 3` groups as
`(A > 0) ^ (A > 3)`. GCC still suggests parentheses there, so add them — the
grouping is already right, but a quiet build is worth two characters.

Both spellings exist on purpose: `&& || ^ !` for people arriving from C, and
`.land() .lor() .lxor() .lnot()` for people arriving from Python or MATLAB. Same
call either way.

Overloading `&&`/`||` costs short-circuit evaluation, which an element-wise *or*
never had — it must look at every element of both operands regardless. And since
the result is a `Matrix<bool>`, which has no conversion to `bool`, `if (m1 || m2)`
**doesn't compile**: you have to say `.any()` or `.all()`. Scalar conditions like
`if (A.any() || B.any())` are plain bools and short-circuit normally. Both
behaviours are pinned by tests.

**Count with `nnz()`, not `sum()`** — `sum()` returns `datatype`, and on a
`Matrix<bool>` that saturates at `true` rather than counting.

## Tensors

`Tensor.hpp` adds N-dimensional arrays. Include it *instead of* `Matrix1.0.hpp` —
it pulls the matrix header in.

```cpp
#include "Tensor.hpp"

Tensor<double> A(2, 3, 4);           // variadic shape, zero-filled
A(1, 2, 3) = 5.0;                    // variadic indexing

auto R = A.reshape(6, 4);            // O(1) — metadata only
auto P = A.permute({2, 0, 1});       // O(1) — strides only
auto S = A.slice(1, 2);              // O(1) — drops axis 1

A + B      A.mul(B)  A.div(B)        // element-wise, same spelling as Matrix
A * B                                // contracts A's last axis with B's first —
                                     // for rank 2 that IS the matrix product
contract(A, B, 2)                    // tensordot over two axes
contract(A, B, {2}, {0})             // over explicitly named axes
contractInto(A, B, 1, out)           // into storage you already own
```

One flat buffer plus shape and strides, the layout NumPy, PyTorch and TensorFlow
all use. The alternatives were measured before this was written, on a
(32, 32, 64, 64) tensor — 4.19M doubles, 33.6 MB:

| | `Matrix<Matrix<double>>` | `vector<Matrix<double>>` | flat (what's built) |
|---|---|---|---|
| element-wise add | 14.22 ms | 13.29 ms | **3.74 ms** |
| sum all | 0.83 ms | 0.79 ms | **0.37 ms** |
| allocations | 1024 | 1024 | **1** |
| usable as a GEMM | no | no | **122 GFLOP/s** |

Nesting `Matrix` inside `Matrix` compiles, and addition even works — but
multiplication *throws*, because the accumulator starts as `datatype(0)`, which
for a nested element is a 0×0 matrix. That's structural, not a bug to fix: a
generic algorithm needs a zero of the right *shape*.

`reshape`, `permute` and `contiguous` measure **30–50 nanoseconds** — they only
rewrite metadata. That matters because every fast tensor contraction is
`reshape → permute → GEMM → permute back`, and the GEMM is `mstore::gemm`,
the same kernel `Matrix::operator*` uses, unmodified.

**Copy semantics differ from `Matrix` in one way worth knowing.** Ordinary copies
deep-copy, as `Matrix` does. But `reshape`/`permute`/`swapAxes`/`T`/`slice` return
**views** that share storage, so writing through one writes through to the
original. Copying a view materialises it, so the aliasing never outlives a
variable you explicitly made with a view method. `clone()` forces a deep copy.

Validated by 95 assertions plus 14 operations cross-checked against NumPy
(`validate_tensor.cpp`, `tensor_numpy_validate.{cpp,py}`).

## Constants and literals

Math constants use the `std::numbers` names, but work from **C++17**:

```cpp
mconst::pi, mconst::e, mconst::sqrt2, mconst::phi, ...   // double
mconst::pi_v<float>, mconst::pi_v<long double>           // any floating type
```

`<numbers>` is C++20 — at `-std=c++17` it includes but `std::numbers::pi` is
"not declared" — so depending on it would silently force every user of this
header to C++20. `M_PI` was the other option and is worse: a POSIX/MSVC
extension rather than ISO C++, a macro (so it can't be scoped or templated), and
needs `_USE_MATH_DEFINES` on MSVC. So the constants are defined here with the
standard's own names and simply **alias to `std::numbers` when it exists**.
Nothing in user code changes on a move to C++20. All 13 constants are verified
bit-identical to `std::numbers` in `float`, `double` and `long double`.

The imaginary unit comes from the standard:

```cpp
using namespace matrix_literals;         // re-exports std::complex_literals

auto z = 3.0 + 4.0i;
Matrix<std::complex<double>> A(2, 2);
A = {{1.0 + 2.0i, 3.0 + 0.0i}, {0.0 + 0.0i, 1.0i}};
```

A literal suffix was chosen over a global `inline constexpr complex<double> i`
because nearly every loop in matrix code uses `i` as a counter, and a local
declaration shadows a global one — quietly making the imaginary unit unusable
inside the very functions you'd want it in. A suffix can't be shadowed.

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

**`validate_tensor.cpp`** — 95 assertions over `Tensor.hpp`: shapes, indexing,
view aliasing, element-wise arithmetic through strided views, reductions,
contraction against hand-rolled loops, `Matrix` interop, complex tensors, and a
5-qubit gate-application round trip.

**`tensor_numpy_validate.{cpp,py}`** — the same idea as `numpy_validate` but for
tensors. NumPy is the reference implementation of this layout, so agreeing with
it on `permute`/`reshape`/`tensordot` is the strongest available statement that
the stride arithmetic is right.

```bash
g++ -std=c++17 -O2 -fopenmp -o validate_tensor validate_tensor.cpp && ./validate_tensor
g++ -std=c++17 -O2 -fopenmp -o tensor_numpy_validate tensor_numpy_validate.cpp
./tensor_numpy_validate > tensor_cases.txt && python3 tensor_numpy_validate.py
```

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
