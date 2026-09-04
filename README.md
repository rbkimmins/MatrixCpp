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

## Layout

```
basic/                     the Basic Matrix Package — one include for all of it
  MatrixCpp.hpp            <-- include this
  mstore.hpp               raw storage, huge pages, the GEMM kernel
  constants.hpp            mconst::pi and the _i / _deg literals
  traits.hpp               is_complex / work_t, ROW and COL, the enums
  io.hpp                   output formats and file writing
  matrix.hpp               the Matrix class and its factorisations
  matrixfunctions.hpp      exp(A), log(A), sqrt(A) — matrix, not element-wise
  decomposition.hpp        factorize(): factor once, solve many times
  builders.hpp             linspace, magic, vander, polynomials
  eigen.hpp                funm, eig(A,B), QZ
  signal.hpp               fft, conv, filter, interp1, gradient
  identity.hpp             the global `I`
  tensor.hpp               the Tensor class
  random.hpp               the ran2 generator both classes use
plotting/                  Plots.jl from C++ — no julia syntax, no build flags
  MatrixPlot.hpp           <-- include this
  julia_script.hpp         script generation and the julia subprocess
  demo.cpp                 eight figures, doubles as a smoke test
gpu/                       placeholder; see gpu/README.md for the measurements
docs/
  DESIGN_NOTES.md          why the code is the way it is
```

```cpp
#include "basic/MatrixCpp.hpp"
using namespace mcpu;

Matrix<double> A(3, 3);
Tensor<double> T({2, 3, 4});
```

Each header stands on its own and chains its own dependencies, so including
just `basic/matrix.hpp` pulls exactly what it needs. `Matrix1.0.hpp` and
`Tensor.hpp` remain at the top level as shims that also hoist `mcpu` to global
scope, so existing code compiles unchanged.

## Namespaces

The package lives in `namespace mcpu`, and its GPU companion in `mgpu`, so the
two spell the same one namespace apart:

```cpp
namespace np = mcpu;                        // the C++ spelling of
namespace cp = mgpu;                        //   import numpy as np

np::Matrix<double> A(4096, 4096);
cp::Matrix<double> dA = cp::upload(A);      // one crossing of the bus
np::Matrix<double> C  = (dA * dA).cpu();    // and one back
```

`using namespace mcpu;` gives back the unqualified `Matrix<double>` for
CPU-only code, and that is what every demo here does. The one combination to
avoid is `using namespace mcpu;` **and** `using namespace mgpu;` in the same
file — both export `Matrix`, so the bare name becomes ambiguous. That is the
point of the split rather than a flaw in it: qualify one of them.

Anything that includes `Matrix1.0.hpp` or `Tensor.hpp` needs no change at all;
the shims hoist `mcpu` for you.

The ~1250-line comment block that used to live inside the header — every design
decision, measurement and negative result — is now `docs/DESIGN_NOTES.md`. That
was most of what made the header unreadable.

**Build with `-O3 -march=native`.**

## Choosing an axis

Reductions, scans and orderings that work along one axis take `ROW` or `COL`,
naming the axis you get one result *per*:

```cpp
A.sum(ROW)      // one sum per row     -> m x 1
A.sum(COL)      // one sum per column  -> 1 x n
A.cumsum(COL)   // accumulate down each column
A.sort(ROW)     // sort each row
```

They're plain `constexpr bool`, so the older `sum(true)` / `sum(false)` spelling
still compiles — but `sort(ROW)` says what `sort(true)` never did.

A `Tensor` has as many axes as dimensions, so it takes the axis *index*
instead — `t.sum(0)`, `t.cumsum(2)`.

## Building

The library is a single header — just `#include "Matrix1.0.hpp"`, which also
hoists `mcpu` to global scope so the samples above work verbatim. It requires
**C++17** (it uses `if constexpr`, structured bindings and inline variables).

```
g++ -std=c++17 -O3 -march=native -fopenmp -I. benchmarks/running_time.cpp -o running_time
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

## Signal, calculus and interpolation

```cpp
conv(a, b)      deconv(y, a)     poly(r)      poly(A)     // polynomials
y.trapz()       trapz(x, y)      y.cumtrapz(true)         // integrate
y.gradient(true)                 y.gradient(true, h)      // differentiate
interp1(x, y, xq)                filter(b, a, x)
```

**Axis operations are members, signal operations are free.** `trapz`, `cumtrapz`
and `gradient` take the same `addcol` flag as `sum` and `cumsum`, because that's
what they are; `conv`, `filter` and `interp1` treat a whole vector as one signal.

`conv` switches to the FFT above 16384 multiply-adds — a *measured* threshold; my
first guess was 10× too high and skipped the FFT across a range where it's twice
as fast. This is also the one place where zero-padding an FFT is correct, since
the longer signal *is* the answer wanted.

`gradient` keeps the length where `diff` shortens it (centred inside, one-sided
at the ends) — that's why both exist.

`interp1` returns its fill value outside the range, defaulting to NaN as MATLAB
does (NumPy clamps instead). ⚠ **That NaN default doesn't survive `-ffast-math`**
— the flag implies `-ffinite-math-only`, so `isnan()` folds to `false` and the
sentinel becomes undetectable. Pass an explicit fill (`interp1(x, y, xq, 0.0)`)
when that matters; it's MATLAB's own escape hatch and works in every build.

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

## Complex support

Routines that used to refuse complex now follow the input type through
`work_t<T>` — `double` for real, `complex<double>` for complex:

```cpp
Matrix<std::complex<double>> A(4, 4);
auto [U, S, V] = A.svd();      // U, V UNITARY; S stays real
A.cholesky();                  // A = L*L^H, input must be Hermitian
A.solve(b);  A.inverse();  A.det();  A.QR();  A.LU();  A.pinv();
exp(A);  sin(A);  cos(A);  tan(A);  sinh(A);  cosh(A);  tanh(A);
A.norm(NormType::Two);  A.cond();  A.rank();
```

Three details worth knowing:

- **`cholesky` needs Hermitian, not symmetric.** `A == A^H` is what forces the
  diagonal real and the pivots positive; a complex *symmetric* matrix is refused.
  That distinction doesn't exist in the real case and is the whole game here.
- **`var` stays real** — it's `E|x−μ|²`, a sum of squared magnitudes. `mean`
  follows the input and is complex. NumPy does the same.
- **`S` from the SVD stays real** — singular values are magnitudes.

**Still real-only, and it's one missing algorithm:** `eig`, `eigvals`, `hess`,
`schur`, `funm`, `log(A)`, `sqrt(A)` and `pow(A, real)` all route through the
**real** Schur form, which a complex matrix doesn't have. That needs a complex
Hessenberg reduction plus a complex QR iteration — a project of its own.

## Scalars, singular matrices, and what still refuses

A **1×1 matrix is a scalar** — it converts implicitly, in both `Matrix` and
`Tensor`:

```cpp
double energy = v.T() * A * v;      // a quadratic form is 1x1
double e      = contract(x, x, 1);  // so is a full contraction
```

Any other shape throws — the size is a runtime property, so it can't be a
compile-time check; NumPy makes the same trade with `float(arr)`. **Masks are
deliberately excluded**: an implicit `operator bool` on a `Matrix<bool>` would
turn `if (mask)` from a compile error into a runtime throw, so a `static_assert`
keeps it an error and tells you to use `.any()` / `.all()` / `.nnz()`.

`A + 3` adds 3 to every element, as in MATLAB (`3 - A` negates first). This had
to land with the conversion: without it, `A + 1.0` on a 5×5 would have fallen
through to the scalar conversion and thrown at runtime.

**`det()` of a singular matrix returns 0**, and `LU()` still factors it — `U`
just carries a zero on its diagonal and `P*A == L*U` holds. Both match MATLAB and
NumPy. What still refuses is `solve()`, `inverse()` and `factorize()`, because a
singular system has no unique solution to return — a different question from
whether the determinant or the factorisation exist.

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

## Least squares, with Q left implicit

LAPACK doesn't have *a* QR routine — it has three, and the split is the point:
`dgeqrf` factors and leaves Q implicit, `dormqr` applies Q without forming it,
`dorgqr` forms it only if you want the matrix. This header had only the third,
so everything went through a full m×m Q.

| least squares | before | after | NumPy `lstsq` |
|---|---|---|---|
| 2000×100 | 80.3 ms | **3.08 ms** | 10.58 ms |
| 4000×200 | 725.9 ms | **12.29 ms** | 87.55 ms |
| 8000×100 | 2399.2 ms | **11.35 ms** | 40.78 ms |

```cpp
auto [Q, R, P] = A.QR();                    // Complete: Q is m x m  (MATLAB's qr(A))
auto [Q, R, P] = A.QR(QRMode::Reduced);     // economy:  Q is m x k  (MATLAB's qr(A,0))
```

| `QR()` on a tall matrix | Complete | Reduced |
|---|---|---|
| 2000×100 | 82.1 ms, Q = 32 MB | **4.28 ms**, Q = 1.6 MB |
| 4000×200 | 677.1 ms, Q = 128 MB | **19.6 ms**, Q = 6.4 MB |
| 8000×100 | 2396.6 ms, Q = 512 MB | **15.8 ms**, Q = 6.4 MB |

The reduced Q is *exactly* the first k columns of the complete one — bit for bit.
Complete stays the default, because that's what MATLAB's `qr(A)` gives. (NumPy
defaults to reduced; its own `mode='complete'` costs 387.8 ms where `mode='r'` is
8.4 ms on the same 2000×100.)

An 8000×100 solve was building a **512 MB** `Q` to produce a 100-element answer.
Forming Q is O(m²·r); applying the reflectors to one right-hand side is O(m·r) —
6.4×10⁹ operations against 8×10⁵. `rank()` got the same treatment, since it only
ever read R's diagonal. `QR()` itself still returns an explicit Q, because that's
what a caller asking for the matrix wants.

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

```
g++ -std=c++17 -O3 -march=native -fopenmp -I. benchmarks/running_time.cpp -o running_time
./running_time
```

Times 57 operations across both `Matrix<double>` and `Matrix<complex<double>>`
and **draws the results itself** into `benchmarks/plots/` — one figure per
family, on log-log axes. About 45 s. No Julia, no matplotlib, no intermediate
file: the plots come out of the C++ program that took the measurements.

Comparing against NumPy is the one place data is written to disk, because the
two sides are measured by two different languages and have to meet somewhere:

```
python3 benchmarks/numpy_timings.py     # -> bench/numpy_*.csv + a table
g++ -std=c++17 -O2 -fopenmp -I. benchmarks/plot_comparison.cpp -o plot_comparison
./plot_comparison                       # -> benchmarks/plots/speedup_*.png
```

The Python side does not choose its own sizes — it reads them back out of the
C++ CSVs, so both implementations are measured at exactly the same `n`. Speedup
is always **NumPy time ÷ MatrixCpp time**, plotted against a parity line at 1.

**Read those speedups with care.** NumPy here links the *reference* BLAS
(`libblas.so.3`), not OpenBLAS or MKL: its matmul runs at ~4.8 GFLOP/s
single-threaded against ~200 for ours. See "QR: why it is level 2" below for
what a fair comparison would look like.

Individual benchmarks are toggled by the `#define BENCH_*` lines at the top of
`benchmarks/running_time.cpp`. See `benchmarks/README.md`.

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

## Complex support

Everything in the header works on `std::complex` except the operations that
genuinely have no complex meaning — `min`/`max`/`sort`/`median`/`mode`, the
ordering comparisons, and `floor`/`ceil`/`round`/`mod`/`atan2`, which need an
ordering `std::complex` does not have.

The last group to land was the Schur family. `eig`, `eigvals`, `hess`, `schur`,
`funm`, `sqrt(A)`, `log(A)`, `pow(A, p)` and `eig(A, B)` all used to go through
the **real** Schur form, which a complex matrix does not have: it parks a
conjugate pair in a 2×2 block, and that only works because a real matrix's
complex eigenvalues come in pairs. Complex matrices get the genuine complex
Schur decomposition, where `T` is fully triangular — so the complex paths are
*simpler* than the real ones, with no blocks to special-case.

```cpp
Matrix<std::complex<double>> A(n, n);
auto [T, Q] = A.schur();          // T upper triangular, Q unitary
auto [val, vec] = A.eig();        // A*v == lambda*v, no real-eigenvalue restriction
auto R = sqrt(A);                 // R*R == A
auto E = funm(A, [](std::complex<double> z){ return std::exp(z); });
```

Cross-checked against `scipy.linalg`'s `schur`, `hessenberg`, `sqrtm`, `logm`
and `expm`, and `funm(exp)` against an independent Taylor series as well.

`factorize()` is complex too — all three paths (LU, Cholesky, QR least squares),
plus `det()` and the `rcond` estimator. Cholesky dispatches on **Hermitian**,
not symmetric: a complex symmetric matrix is not Cholesky-able.

### Two real-matrix bugs this uncovered

Both were the same mistake — trusting a real Schur form to say something it
cannot — and both were silent:

- `eig()` returned the **Schur vectors** as eigenvectors. Only the first column
  of `Q` is ever an eigenvector. Eigenvalues right, vectors wrong, no error. It
  was invisible for symmetric input, where the Schur form is diagonal and the
  two coincide — which is why the existing tests all passed.
- `sqrt(A)`, `log(A)` and `pow(A, real)` read a 2×2 block as its real part
  twice. That part is usually positive, so the positive-eigenvalue check passed
  and the answer came back wrong: `||R*R - A|| = 6.7e-01` on a 4×4 with
  eigenvalues 6.084 ± 0.403i. They now route through `funm`.

Both are pinned in `validate.cpp` with the matrices that exposed them.

## QZ — the generalized Schur decomposition

```cpp
auto r = qz(A, B);              // Q^H A Z = S,  Q^H B Z = T,  both triangular
r.alpha(); r.beta();            // eigenvalue i is alpha[i]/beta[i]
r.infinite();                   // where beta == 0: B is singular there
```

`A == Q S Z^H` and `B == Q T Z^H`, with `Q` and `Z` unitary. **Nothing ever
forms `B⁻¹A`** — which is the entire point. `eigvals(A, B)` used to refuse
outright below `rcond(B) = 1e-10`, because forming that product spends the
available precision before the eigensolver starts. It now routes through QZ and
agrees with `scipy.linalg.eig(A,B)` to 2.3e-13 at exactly that conditioning.

Eigenvalues come back as the **pair** `(alpha, beta)`, the way LAPACK reports
them, because `beta == 0` is a legitimate *infinite* eigenvalue of a singular
pencil and a ratio cannot express it.

A real pencil works too. The Hessenberg–triangular reduction stays in real
arithmetic (that's the O(n³) stage), and only the single-shift sweep is complex
— which it has to be, since a real pencil can have complex eigenvalues.

### Eigenvectors, and the degenerate cases

```cpp
auto X = r.eigenvectors();      // one column per eigenvalue
r.undefined();                  // where alpha AND beta are 0: no eigenvalue exists
```

Eigenvectors come from back-substitution on the triangular pair in the
**homogeneous** form `(βA − αB)x = 0`. Dividing to get λ first would give NaN
for exactly the eigenvalues QZ exists to handle; the homogeneous form stays
finite, and an infinite eigenvalue comes back with a genuine vector satisfying
`Bx = 0` (measured at 6.6e-16).

A **singular pencil** — A and B sharing a null space, so `det(A − λB)` vanishes
identically and *no* eigenvalue is determined — is reported by `undefined()`
rather than passed off as an answer. Those pairs are 0/0, and the values that
come back are rounding noise that looks like ordinary numbers. On a 20×20 with a
shared 12-dimensional null space: 12 undefined pairs here and 12 from SciPy,
with the remaining 8 agreeing to 1.1e-15.

Interior zeros on `T`'s diagonal need no special handling — the ordinary sweep
drives them to an end. An earlier version chased them explicitly and was wrong: a
rotation zeroing `T(k+1,k+1)` leaves `T(k,k)` zero too, so the zero *spreads*
along the diagonal instead of moving, reporting n−1 infinite eigenvalues for a
pencil with one. Deleting that code fixed it.

Ill-scaled pencils are normalized first. With `‖A‖ ~ 1e9` against `‖B‖ ~ 1e-9`
the shift walks through eighteen orders of magnitude and the tolerances stop
meaning anything — 70 of 400 stress pencils failed to converge before this. The
scaling is undone on the factors, so `A == Q S Z^H` still holds to the bit.
After the fix: **400/400, worst relative residual 2.9e-15**.

## The matrix-multiply kernel

`mstore::gemm` tiled its output over **M only**, so any product with `M < 64` had
one tile and ran single-threaded — 6.5 GFLOP/s on 32 cores against 8.0 on one.
That's the shape every panel algorithm produces, and it was the real reason
blocked QR lost.

It now tiles over both dimensions, sizes the dynamic chunk from the work in a
tile, and shrinks the M block when N is narrower than one. Measured interleaved
against the old kernel, min of 7:

| shape | before | after | |
|---|---|---|---|
| square 1024 | 46.3 | 60.8 GFLOP/s | 1.31× |
| square 2048 | 68.7 | 67.4 | 0.98× |
| `C -= V W` (K=48) | 20.1 | 25.2 | 1.26× |
| `V^H C` (M=48) | 4.2 | 31.6 | **7.56×** |
| `V^H C` (M=48) large | 5.0 | 50.6 | **10.10×** |
| `W = Cᵀ V` (N=48) | 39.8 | 74.9 | 1.88× |
| very skinny (M=16) | 4.9 | 48.2 | **9.90×** |

Bit-identical output on a 137×137 — deliberately not a multiple of 64. This
lifts every level-3 path, not just QR: Tensor contractions, the Taylor matrix
functions, plain multiplication.

**Build with `-O3 -march=native`.** On a Zen 4 box, n=1024 square double:

| flags | 1 thread | 32 threads |
|---|---|---|
| `-O2` | 5.2 | 71 GFLOP/s |
| `-O3 -march=native` | 13.5 | **194** |

That's 2.7× for free. Any GFLOP/s figure quoted without its flags is meaningless.
A packed register-blocked AVX-512 microkernel prototype reaches 75 (1 thread) and
438 (32 threads) — 5.6× and 2.2× respectively — which is the ceiling for a
rewrite, and not enough to make blocked QR win.

## The real Schur decomposition

Two bugs, one hiding the other.

**The shift could not converge on a complex pair.** With a complex trailing 2×2
it fell back to `sigma = d`, a *real* shift — and no real shift converges to a
conjugate pair. So `schurDecomp` threw on ordinary matrices, in no pattern a size
threshold would catch: a random 16×16 failed while a 64×64 succeeded. `eig`,
`schur`, `funm`, `sqrt(A)`, `log(A)` and `pow(A, real)` all went down with it.
Replaced with the **Francis double shift**.

**Then `split2x2` turned out to be wrong too** — visible only once the Francis
step began reaching 2×2 blocks the old iteration never did. Its discriminant was
`((a+d)/2)² − (ad − bc)`, which cancels catastrophically when a ≈ d: for
`a = d = 1e3, b = c = 1e-3` the true `1e-6` is the difference of two numbers near
`1e6`. The rotation built from it didn't zero the subdiagonal, and the routine
then **forced it to zero anyway**, silently breaking the similarity. Replaced
with LAPACK's `dlanv2` formulation.

500 real matrices (random, near-identity, symmetric, triangular, 1e6-scaled):

| | clean | bad | threw | worst rel. residual |
|---|---|---|---|---|
| before | 460 | 2 | **38** | 7.9e-04 |
| after | **500** | 0 | 0 | **4.3e-15** |

The check that found the second bug: symmetric input must stay symmetric under an
orthogonal similarity, and symmetric plus zero subdiagonal means *diagonal*. H was
coming back triangular with a nonzero upper triangle.

## Tensor parity with Matrix

`Tensor` now carries the same element-wise family as `Matrix` — `abs`, `sqrt`,
`exp`, `ln`, `lg`, `log10`, `sin`/`cos`/`tan`, the hyperbolics and their
inverses, `pow`, `log(base)`, `sign`, `conj`, and `floor`/`ceil`/`round`/`fix`/
`expm1`/`log1p` (which refuse complex, as Matrix's do) — plus `prod`, `norm`,
`var`, `stddev`, axis reductions `prod`/`min`/`max`/`mean`, and the axis scans
`cumsum`, `cumprod` and `diff`.

The scans keep the axis; `diff` shortens it by one, as MATLAB's does. Reductions
drop it.

These are checked against `Matrix` on a 2-D tensor and agree **exactly**
(`0.0e+00` for sum, cumsum, diff, abs and sin) — the two reached them by
different routes, so an exact match is evidence rather than a shared assumption.

## Printing

`print()` goes to the terminal; every other overload takes a `std::ostream`,
which is how the rest of C++ spells this. A file, a `std::ostringstream` and a
socket are all the same thing to it.

```cpp
A.print();                          // aligned, to the terminal
A.print(4);                         // ... to 4 decimal places
A.print(matio::Fmt::CSV);           // comma separated, to the terminal
A.print(file, matio::Fmt::CSV);     // ... to a file
A.save("data.csv");                 // format taken from the extension
std::string s = A.str(matio::Fmt::Markdown);
std::cout << A;                     // operator<< still works
```

Eight formats, the same for `Matrix` and `Tensor`:

| | |
|---|---|
| `Pretty` | aligned and bracketed — the default, for a terminal |
| `Plain` | whitespace separated, nothing else |
| `CSV` / `TSV` | comma / tab separated, optional header row |
| `Markdown` | pastes straight into a document |
| `MATLAB` | `[1, 2; 3, 4]`, assignable with `Opts::name` |
| `NumPy` | `np.array([[1, 2], [3, 4]])` |
| `JSON` | `[[1, 2], [3, 4]]` |

`Opts` carries `precision`, `scientific`, `header` and `name`. `save()` picks
the format from the extension (`.csv`, `.tsv`, `.md`, `.json`, `.m`, `.py`,
`.txt`) and **throws** if the file can't be opened — a save that quietly did
nothing is the worst outcome.

**Complex formats as `3+4i`, not `(3,4)`.** The default `std::complex` spelling
contains a comma, which would silently add a column to every CSV row it appeared
in.

A rank > 2 `Tensor` has no 2-D layout, so `CSV`/`TSV`/`Plain` flatten it to
(leading axes) × (last axis) and write a `# shape (2, 2, 3)` line first —
without it the flattening couldn't be undone. `JSON` and `NumPy` nest to match
the rank instead, so the structure survives on its own.

Reading these back in is deliberately not here yet; the formats were chosen so
that `Plain`, `CSV` and `TSV` are trivially parseable when it lands.

## QR: why it is level 2

`QR` uses an unblocked Householder factorisation. That is a measured decision,
not an omission — **both** level-3 variants were implemented and both lost:

| | 1024² | 1500² | 2048² |
|---|---|---|---|
| blocked (compact-WY, best of 6 block sizes) | 0.63× | 0.49× | 0.53× |
| recursive (Elmroth–Gustavson, best of 5 leaf sizes) | 0.54× | 0.45× | 0.51× |

The argument for recursion was sound and it partly held: its root GEMMs run at
74.5 GFLOP/s, indistinguishable from a full square product. It still loses,
because the recursion does ~2.3× the arithmetic (every level pays for a `T`),
the deeper GEMMs run at a third of the root's rate, and the level-2 kernel it
has to beat is already streaming contiguous columns in parallel.

Two implementation traps are worth knowing, because the first version was **27×
slower** than unblocked and it would have been easy to stop there and blame the
algorithm. Both were hand-written triangular loops standing in for a GEMM. The
costly one — `W^T ← W^T conj(T)` inside the block update — is 4.5% of a blocked
QR at block size 48 and **84.6% of the whole factorisation** at block size n/2.
Fixing both took 2048² from 11.6 s to 0.73 s. A level-3 algorithm has no room
for a level-2 helper hiding inside it, and a phase profile is how you find one.

## Factorisation performance

Measured against **our own GEMM**, not against NumPy — NumPy here links the
reference BLAS, so beating it proves nothing. At n=1024:

| | before | after | |
|---|---|---|---|
| LU (`det`) | 52.2 ms | 24.7 ms | **2.1×** |
| `solve` | 52.1 ms | 25.0 ms | **2.1×** |
| `inverse` | 81.8 ms | 48.4 ms | **1.7×** |
| `cholesky` | 53.8 ms | 17.3 ms | **3.1×** |
| `qz` | 91.9 s | 21.9 s | **4.2×** |

**LU and Cholesky are blocked** (LAPACK's `dgetrf`/`dpotrf`), so the trailing
update is a GEMM. Unlike QR this costs nothing extra — QR's compact-WY form
needs a `T` matrix worth ~40% more arithmetic, which is why blocking lost there
twice; LU's blocked update is the same arithmetic regrouped.

**LU is also column-major internally.** Every operation in its panel runs *down*
a column, and row-major storage walks a fresh cache line per element — 21% of
the factorisation went on column scaling alone. The pivot sequence is unchanged,
so `det()` keeps its sign.

**QZ's 4.2× was entirely memory, not arithmetic.** Its rotations were applied to
full rows and columns when only a band can be nonzero, and `Q`/`Z` were stored so
that every update to them strided. `schurDecomp` had always done both correctly.
QZ is now 2.4–4.4× Schur rather than 12–14×, which is about what carrying four
matrices instead of two predicts.

`schur` and `svd` remain at ~1 GFLOP/s. They are iterative eigenvalue algorithms
built on Givens rotations — inherently level 1/2, and the real fix is LAPACK's
multishift blocked bulge-chasing, which is a project rather than a tuning pass.

## Plotting

```cpp
#include "plotting/MatrixPlot.hpp"

Matrix<double> x = linspace(0.0, 10.0, 200);
plt::plot(x, x.sin(), "sin");
plt::plot(x, x.cos(), "cos");
plt::title("trig");  plt::xlabel("x");  plt::legend();
plt::save("trig.png");
```

```
g++ -std=c++17 -O2 -fopenmp -I. demo.cpp -o demo
```

No Julia syntax, and no build flags beyond `-I.` — it needs `julia` on `PATH`
with Plots.jl, and nothing else. The model is matplotlib's: a current figure,
drawing calls add to it, `save()` finishes it.

`plot` `scatter` `bar` `stairs` `semilogx` `semilogy` `loglog` `heatmap`
`surface` `contour` `spy` `hist`, plus the usual labels, limits and `set()` for
any other Plots attribute. `plot(A)` on a multi-column matrix draws one series
per column.

See `plotting/README.md` — including why this shells out to `julia` rather than
embedding the runtime, which was tried first and crashes inside Julia's package
loader in a way that depends on the calling binary's size.

## GPU

`gpu/` is a placeholder. CUDA 13.2 and an RTX 5060 Ti are already installed on
this machine — nothing to install, only `PATH` to set. But measured, cuBLAS
`DGEMM` is 318–341 GFLOP/s against 180–270 for this CPU: GeForce cards throttle
FP64 to 1/64 of FP32, so **in double, which is what `basic/` uses everywhere,
the GPU is only 1.3–1.9× the CPU.** The numbers and what they imply for the
design are in `gpu/README.md`.
