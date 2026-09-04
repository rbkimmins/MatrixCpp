# MatrixCpp — design notes and roadmap

> **Namespaces.** Everything below is written as bare `Matrix<double>`, which
> is how it reads under `using namespace mcpu;`. The package lives in `mcpu`
> and its GPU companion in `mgpu`, so the fully qualified spellings are
> `mcpu::Matrix` and `mgpu::Matrix`. See the root README.


These were a ~1250-line comment block inside `Matrix1.0.hpp`. They are the
reasoning behind the code, not the code, and they were most of what made the
header unreadable. Nothing was deleted — this is that block, verbatim.

---

## Roadmap and design log

```text

─────────────────────────────────────────────────────────────────────────────
ROADMAP — where the project is at

Naming convention already in use (keep it):
  A.f()   member function  → ELEMENT-WISE   (A.pow(2), A.ln(), A.exp())
  f(A)    free function    → MATRIX-WISE    (pow(A,2), log(A,base))
  Operators are element-wise EXCEPT operator*(Matrix), which is matmul.
  operator() is indexing/slicing, not arithmetic.

Everything claimed below is checked by validate.cpp — build and run it before
trusting any of it:
    g++ -std=c++17 -O2 -fopenmp -o validate validate.cpp && ./validate

DONE
  [x] solve, norm, rank            — solve covers square (LU) and
  over-determined
                                      (least squares via column-pivoted QR);
                                      inverse() is now solve(I), one copy of
                                      the substitution code rather than two
  [x] cholesky                     — SPD, with the failed-pivot case doubling
  as
                                      the positive-definiteness test
  [x] svd, pinv, cond              — svd is one-sided Jacobi, chosen over
                                      bidiagonalise-then-QR for its relative
                                      accuracy on small singular values,
                                      which is what cond() and pinv()
                                      actually depend on
  [x] diag, triu, tril, reshape    — plus the free diag(v) that goes the
  other way [x] min/max/mean/var/stddev/argmin/argmax  — scalar and axis
  forms, mirroring sum(bool) [x] ==, !=, allclose, unary -    — and
  operator+=; operator+/-/| are now const,
                                      so they work on const operands
  [x] adjugate                     — det*inverse when non-singular, cofactor
                                      expansion when not
  [x] exp(A) matrix exponential    — scaling-and-squaring around a Taylor
  series [x] sin/cos/tan/sinh/cosh/tanh   — element-wise members and
  matrix-wise free fns.
                                      tan/tanh are SOLVES (cos(A)·X =
                                      sin(A)), not the element-wise division
                                      they look like
  [x] conj(), real(), imag(), H()  — see COMPLEX below
  [x] Caller-supplied series limits on every Taylor-based function, via
  TaylorOpts

THE 2x2 SCHUR BLOCK — resolved, and it was worse than previously recorded.
  The old note said eig() *read* complex-conjugate pairs wrongly. It did, but
  schurDecomp() also never CONVERGED on them: a conjugate pair's sub-diagonal
  entry does not go to zero (that is the definition of a real Schur form), so
  the iteration ran to its step limit and threw. A plain rotation matrix was
  enough. Fixed in schurDecomp() by deflating an isolated trailing 2x2 as a
  block, splitting it with a Givens rotation when its roots turn out to be
  real, and adding an exceptional shift for stalled blocks. Consequences:
    - eigvals() returns every eigenvalue including complex ones, block-aware.
    - eig() stays real-valued but now THROWS on a conjugate pair instead of
      silently returning the real part twice.
    - The matrix trig functions never needed this: they sum a Taylor series
    and
      so avoid the Schur form entirely. pow()/log() still go through it and
      still require positive eigenvalues.

COMPLEX NUMBER SUPPORT — Matrix<std::complex<double>>, via <complex> only.
Status as measured, not guessed (see the probe in validate.cpp):
  WORKING: constructors, =, + - * / %, matmul incl. Strassen, T(), H(),
  conj(),
    real(), imag(), tr(), sum(), IsDiagonal(), slicing/proxies,
    set_Ran_values(), concat, tensor, reshape/triu/tril/diag, ==/!=/allclose,
    unary -, norm() (all of Fro/One/Inf — std::abs already gives the complex
    modulus), and operator<< now formats properly.
  REFUSED AT COMPILE TIME, deliberately: min/max/argmin/argmax (complex has
  no
    ordering) and mean/var/stddev, svd, cholesky, and the Taylor matrix
    functions. Each of those would otherwise compile by taking std::real() of
    every entry and quietly discarding the imaginary part — a wrong answer
    dressed as a working one. They static_assert with an explanation instead.
  STILL TO DO: det, inverse, solve, LU, QR, eig, schurDecomp, pow, log. All
  of
    them fail on the same single construct, double(grid[k]).

  [ ] 1. Replace the double(...) casts using the work_t idea.
  real_of/is_complex
         already exist at the top of this header; what is missing is
           work_t = std::complex<double> when datatype is complex, else
           double
         plus turning the std::vector<double> scratch buffers into
         std::vector<work_t> and widening the return types to Matrix<work_t>.
         Real matrices are unaffected because work_t collapses to double.
  [x] 2. conj(), real(), imag(), H() — done. H() is the one that matters: for
         complex matrices it, not T(), is the adjoint. NOTE the corollary is
         still outstanding — Q^T in QR, Q*T*Q^T in schurDecomp and every
         symmetry check in this header still call .T(), and each must become
         .H() when item 1 lands. Every one of those is a SILENT wrong answer
         if missed; it compiles perfectly, it is just not the right matrix.
  [ ] 3. Householder reflectors need their complex form. The real code picks
         alpha = -sign(x_1)*||x||; the complex version is
         alpha = -exp(i*arg(x_1))*||x||, and tau becomes complex. This is a
         genuine algorithm change, not a type substitution. Same for the
         Givens rotations and the Wilkinson shift inside schurDecomp.
  [ ] 4. Pivoting is fine as written — std::abs() on a complex returns the
         real magnitude, so the LU pivot search keeps working and keeps
         meaning the right thing. Explicit sign tests like (W(k,k) >= 0.0) do
         NOT survive; complex has no ordering. Those are the lines to hunt,
         and split2x2()'s (half >= 0.0) is now one of them.
  [x] 5. Printing — toLines() now tests is_float_like<datatype>, which is
  true
         for complex<double> where std::is_floating_point is false.
  [x] 6. Complex eig() — eigvals() delivers this; see THE 2x2 SCHUR BLOCK
  above.

[x] The imaginary unit — DECIDED and implemented; see the Literals block near
    the top of the file. std::complex_literals won over a global
    `inline constexpr std::complex<double> i(0,1)`, for the reason weighed
    here originally: nearly every loop in this header uses `i` as a counter,
    and a local declaration shadows a global one. That still compiles (the
    loop wins) but leaves the imaginary unit unusable inside almost every
    function you would want to write matrix code in. A literal suffix cannot
    be shadowed by a variable, so it has no such failure mode — validate.cpp
    pins that down with a test that declares `int i = 7` and then uses 2.0i.
    Reached through `using namespace matrix_literals;`, which re-exports
    std::complex_literals so one using-directive covers future additions too.

BUILD FLAG WARNING — this one is specific to the current compile line.
-ffast-math enables -fcx-limited-range (verified with -Q --help=optimizers on
this toolchain: disabled at -O3, enabled once -ffast-math is added). That
switches complex multiply and divide to the naive textbook formulas with no
range reduction, so complex division can overflow or underflow spuriously on
operands that are individually well within double's range. Drop -ffast-math,
or add -fno-cx-limited-range, before trusting any complex benchmark numbers.

Also outstanding (not code):
  [x] .gitignore — added; the committed binaries still need `git rm --cached`
  [x] rt_mat_pow_real.txt was empty — root cause found and fixed. It was not
  a
      benchmark-harness problem at all: Strassen-Winograd was computing wrong
      products (see below), so B.T()*B came back non-symmetric with negative
      eigenvalues, pow(A,0.5) threw on the first size that used the padded
      Strassen path (n=86), and the program aborted before flushing the file.
  [x] README now documents the build line and the member/free convention
  [ ] The committed rt_*.txt timings predate the Strassen fix. The fix does
  not
      change the operation count, so the timings should still stand, but they
      were measured against a path that returned wrong answers — worth a
      rerun before they are quoted anywhere.

STRASSEN-WINOGRAD — was silently wrong, now correct but compiled OUT.
  Two errors in the combination table (T4 had its operands reversed, and U7
  used the wrong pair of intermediates) meant operator* returned incorrect
  products for EVERY size that actually entered the recursion. It went
  unnoticed because n=64 hits the base case and falls straight back to
  naiveMul, so the recursion first ran for real at n=128. validate.cpp now
  checks operator* against a reference product at every size class.
  Once correct, it was benchmarked against the same blocked naive multiply
  and lost at every size (0.43x at n=128, 0.11x at n=1024) — the recursion is
  serial where naiveMul is OpenMP-parallel, and it allocates ~20 temporaries
  per level. It is therefore behind MATRIXCPP_ENABLE_STRASSEN and off by
  default; see the note at operator*(Matrix) for what would make it pay.

PERFORMANCE WORK ALREADY DONE (all measured, all still green in validate.cpp
and numpy_validate.py):
  [x] NRVO — returning a local declared inside a try block suppresses the
      named return value optimisation, adding a full allocate-zero-copy of
      the result. 9x on Hadamard and element-wise division. The rule this
      leaves behind: keep the try around the CHECKS, not around the result.
  [x] Element-wise ops build into an uninitialised buffer in one pass instead
      of copy-then-modify. 1.6-2.1x.
  [x] schurDecomp QR step uses Givens rotations, exploiting the Hessenberg
      structure the previous full-length Householders ignored: O(n^4) ->
      O(n^3), and Q accumulates transposed so its updates are contiguous.
      107x on eig, ~90x on pow(A,real) and log(A).
  [x] QR accumulates Q transposed for the same reason. 1.5x.
  [x] concat/kron index directly instead of through the wrapping operator(),
      which ran an integer division per coordinate. Up to 5x.
  [x] Frobenius norm uses |x|^2 directly rather than squaring std::abs, which
      for a complex matrix was computing a square root only to undo it. 55x.
  [x] sum() and the norms use four accumulators so the loop runs at add
      throughput rather than add latency. 3.7x.

IDEAS TAKEN FROM Eigen / Armadillo / uBLAS / MTL4 (all measured):
  [x] Rule of FIVE. The class had a destructor, copy constructor and copy
      assignment but no MOVE pair, so every `C = A + B;` deep-copied the
      temporary operator+ had just built. Adding them, noexcept so that
      std::vector<Matrix> actually moves on reallocation, took a 4-term
      expression chain from 28.6ms to 7.2ms.
  [x] Rvalue-qualified arithmetic. In A + B + C the left operand of the
  second
      + is the temporary the first + produced, so the && overloads write into
      that buffer instead of allocating another. A chain of k operations now
      allocates once, not k times. This is the cheap half of what expression
      templates (Eigen, uBLAS, MTL4) do; the full version fuses the chain
      into one pass, but it changes what `A + B` RETURNS, which breaks
      template argument deduction in ordinary user code like f(A + B). Not
      worth it here. CAVEAT: only catches temporaries on the left. A + (B +
      C) still allocates.
  [x] Small-buffer storage, the runtime cousin of Eigen's fixed-size types.
      Matrices up to 4x4 live inside the object. A 2x2 A+B was 19ns of which
      19ns was new/delete — the allocator WAS the operation.
  [x] __restrict on the kernels. Biggest single win of the group: naiveMul
      roughly doubled at n=2048, because without it the compiler
      must assume the result aliases the operands and refuses to vectorise.
      Element-wise ops did NOT improve — at n=2000 they move 96MB and are
      already at the memory roofline, so there is nothing for vectorisation
      to recover.
  [ ] Aligned allocation (Eigen aligns to 16/32/64). new[] gives 16 here and
      never 32, so AVX loads are unaligned. Untested; likely small next to
      the bandwidth limit above.
  [ ] Register-blocked GEMM micro-kernel (Eigen/BLIS/Goto). The inner loop is
      an axpy doing one FMA per two memory ops. Computing a 4x4 tile of C in
      registers would reuse each loaded value four times. This is the single
      biggest remaining item for multiply.
  [ ] Sparse storage (PETSc/Trilinos). Out of scope for a dense library, but
      it is what those two are actually for.

A NOTE ON REF-QUALIFYING MEMBER OPERATORS, learned the hard way:
  once ANY overload of an operator name is ref-qualified, every sibling
  overload must be too. operator*(scalar) was qualified while
  operator*(Matrix) was not, and for an rvalue left operand the &&-qualified
  scalar template then beat the unqualified matrix one — silently routing
  `Q.T() * B` into scalar multiplication. It compiled. validate.cpp caught
  it.

IDEAS TAKEN FROM WHAT NumPy AND LAPACK ACTUALLY DO (all measured).
NumPy's speed comes from three places, and it turned out that copying them
was mostly about STORAGE and MEMORY, not about cleverer arithmetic:

  [x] madvise(MADV_HUGEPAGE) on large buffers. Lifted straight from NumPy's
      PyDataMem_NEW (numpy/_core/src/multiarray/alloc.c), same 4 MB
      threshold. Proved by toggling NumPy's own switch,
      _set_madvise_hugepage:
          np.hstack of two 2000x2000 doubles, hugepages ON   6.5 ms
                                              hugepages OFF 26.8 ms
          our concat, before the change                     24.8 ms
      That is, our copy loop was ALREADY as good as NumPy's and the entire
      4x gap was page-fault traffic. See adviseHuge().
  [x] Constructor-free storage (also Eigen/Armadillo). `new T[n]` runs T's
      default constructor, which for double is nothing but for
      std::complex<double> writes a zero to every element — faulting the
      whole buffer in before adviseHuge can apply, then paying a second full
      pass when the caller overwrites it. Complex concat: 49.8 -> 8.6 ms.
  [x] Pairwise summation, NumPy's reduction algorithm. Eight independent
      accumulator chains instead of one latency-bound one, and O(log n · eps)
      error growth instead of O(n · eps). sum(axis=1): 2.06 -> 0.36 ms.
      Kept OFF for complex, where it measured 3x slower — see pairwiseSum.
  [x] Column-major working arrays inside the factorisations. This is the
      single most valuable thing LAPACK does that a row-major library gets
      wrong for free. Every step of a Householder QR and every rotation of a
      one-sided Jacobi SVD walks a COLUMN; in row-major storage that strides
      by a whole row, one cache line per element, and no vectorisation.
      Fortran has contiguous columns by construction. Transposing the working
      copy (not the input) buys the same thing:
          svd, n=256:  644 -> 60.6 ms       qr, n=512:  838 -> 72.3 ms
  [x] Cached column norms in the Jacobi sweep, as dgesvj's sva[] array. Only
      the cross term p·q has to be recomputed per pair; the two squared norms
      update exactly through the rotation as alpha - t·gamma, beta + t·gamma.
      Three dot products per pair become one.
  [x] Brent-Luk round-robin pair ordering, so a sweep splits into rounds of
      column-disjoint pairs that run in parallel. Deterministic regardless of
      thread count. svd, n=256: 60.6 -> 19.7 ms, and 12.3 ms after the thread
      cap below. Total 52x, and it now beats NumPy's LAPACK dgesdd.
  [x] Parallelism where NumPy structurally cannot use it. NumPy's ufuncs are
      SIMD but strictly SINGLE-THREADED, so every element-wise map and every
      element-wise binary op is one core there. Threading them (mapElems,
      forEachIndex) is a gap that is simply not available to it:
          elem_ln 11.6 -> 0.62 ms      hadamard 2.55 -> 0.80 ms
      Also the per-right-hand-side loop in solve(), which is what inverse()
      is built on: 83 -> 10.7 ms at n=512.
  [x] Index arithmetic. tr(), IsDiagonal() and cholesky() recovered (i,j)
  with
      / and %, or went through the wrapping operator() — up to four integer
      divisions in the innermost loop of an O(n³) algorithm. cholesky was
      held to 0.78 GFLOP/s by this alone: 26.8 -> 6.0 ms.
  [x] Thread counts matched to the work. A memory-bound loop saturates with
      about half the reported threads (physical cores, no SMT) and gets
      SLOWER past that; a Jacobi sweep enters ~2000 parallel regions per
      factorisation and wants only as many threads as leave each one ~8k
      element-updates. See memoryThreads() and the table at svd's
      sweepThreads.

A CAVEAT ON THE BENCHMARK'S LARGEST SIZE, worth knowing before quoting it:
  this machine has 64 MiB of L3, and a 2000x2000 double matrix is 32 MB — so
  a one-input, one-output real op at n=2000 has a 64 MB working set that
  fits ENTIRELY in L3, while the complex version at 128 MB does not.
      A * scalar:  n=1000  794 GB/s | n=2000  771 GB/s | n=3000  34 GB/s
  That cliff is why the real element-wise numbers look so much better than
  the complex ones (2.3-23x against ~1.0x): above L3 both libraries are
  pinned to the same DRAM roofline and the only remaining lever is thread
  count.

STILL SLOWER THAN LAPACK/NumPy, in rough order of how much is on the table:
  [ ] eig (0.81x) and pow(A, real) (0.85x). schurDecomp is O(n^3) but
      memory-bound at n >= 512. LAPACK's answer is the blocked multishift QR
      of dlaqr0, which chases several bulges per pass over memory. The single
      biggest algorithmic item left.
  [ ] complex elem_div. libstdc++ implements complex division with Smith's
      algorithm — branches and a range reduction that will not vectorise.
      NumPy uses the naive formula. This is a correctness/speed trade, not an
      oversight: -ffast-math would take our path too (via -fcx-limited-range)
      and quietly change the answers, which is why the build flag warning
      above exists.
  [ ] reshape (0.95x) is a pure copy racing memcpy; there is nothing here.
  [x] QR IS NOW SPLIT THE WAY LAPACK SPLITS IT. LAPACK does not have "a QR
      routine"; it has three, and the split is the point:
          dgeqrf/dgeqp3   factor, leaving Q implicit as a list of reflectors
          dormqr          APPLY Q or Q^H, without ever forming it
          dorgqr          form Q explicitly, only if the matrix is wanted
      This header had only the third, so everything went through a full m×m Q.
      For a tall thin least-squares that was a catastrophe: an 8000x100 solve
      built a 512 MB Q to produce a 100-element answer. Forming Q is O(m²·r);
      applying the reflectors to one right-hand side is O(m·r) — 6.4e9
      operations against 8e5.
          least squares      before      after     NumPy lstsq
          2000x100           80.3 ms    3.08 ms      10.58 ms
          4000x200          725.9 ms   12.29 ms      87.55 ms
          8000x100         2399.2 ms   11.35 ms      40.78 ms
      211x at the worst case, and now 3-7x faster than NumPy. rank() got the
      same treatment — it only ever read R's diagonal. QR() itself still
      returns an explicit Q, because that is what a caller asking for the
      matrix wants.
      The column permutation now comes from the pivot list rather than from
      multiplying an n×n permutation matrix: O(n) to move n numbers instead
      of O(n²).
  [x] A REDUCED (economy) MODE, which is what NumPy defaults to and MATLAB
      spells qr(A,0). Q becomes m x k and R k x n with k = min(m,n); the rows
      of R below k are zero by construction, and the reduced Q is EXACTLY the
      first k columns of the complete one — bit for bit, which validate.cpp
      asserts rather than approximates.
      This header always built the full m x m Q, and on a tall matrix that is
      essentially the entire cost — measured, the complete Q was 98% of QR():
          2000x100    82.1 ms -> 4.28 ms    (Q 32 MB -> 1.6 MB)    25x
          4000x200   677.1 ms -> 19.6 ms    (Q 128 MB -> 6.4 MB)   29x
          8000x100  2396.6 ms -> 15.8 ms    (Q 512 MB -> 6.4 MB)  127x
      NumPy's own numbers show the same shape: on 2000x100 its mode='r' is
      8.4 ms, mode='reduced' 18.6 ms, mode='complete' 387.8 ms.
      Complete stays the DEFAULT, because that is what MATLAB's qr(A) gives
      and changing it would silently alter every existing caller.
  [x] THE REFLECTORS ARE ONE FLAT BLOCK, not a vector of vectors. Reflector k
      has length m-k and lives at hoff[k] in a single allocation sized to the
      exact total. The whole factorisation is now SEVEN heap allocations
      whatever the size, where the reflectors alone previously cost 2r+1 —
      201 for a 2000x100, 401 for a 4000x200. (Counted with an instrumented
      operator new, because wall-clock on a loaded machine could not
      distinguish it from noise.)
      LAPACK and Eigen both go one step further and pack the reflectors into
      the LOWER TRIANGLE of the same array that holds R, normalising v[0] to
      1 so it need not be stored. That is worth doing if the blocked update
      below ever lands, since it wants the whole factorisation contiguous;
      until then this gets the allocation win without the implicit-unit
      convention.
  [x] AN UNPIVOTED MODE, QRPivot::Off — MATLAB's two-output [Q,R] = qr(A),
      LAPACK's dgeqrf. P comes back as the identity, so A == Q*R exactly with
      no permutation to undo. Measured at 1.00-1.08x the pivoted path: the
      pivot search and norm downdating are O(n) per step against O(m*n) for
      the trailing update, so they were never the cost. This is an API and
      correctness feature, NOT a speed one, and is documented as such.

  [x] BLOCKING WAS TRIED AND REJECTED — a measured negative result, kept here
      so it is not attempted a third time.
      The compact-WY blocked update (Eigen's scheme, block size 48) was
      implemented in full: panel factorisation, the T recurrence built from a
      Gram matrix, and the trailing update as C -= V(T^H(V^H C)) through
      mstore::gemm. It was SLOWER AT EVERY SIZE AND EVERY BLOCK SIZE tried:
          NB           48     64    128    192    256
          1024x1024  0.63x  0.52x  0.35x  0.31x  0.17x
          1500x1500  0.49x  0.43x  0.32x  0.30x  0.14x
      Two independent reasons, both specific to this header:
        1. THE TRAILING MATRIX NEVER LEAVES CACHE. Blocking exists to cut DRAM
           traffic, and with 64 MB of L3 a double matrix stays resident up to
           n ~ 2900. The level-2 update is not memory bound here, so there is
           nothing for level 3 to recover. Confirmed by walking the cliff —
           the deficit narrows exactly where it should and nowhere else:
             n=2000 (31 MB) 0.56x | 2600 (52 MB) 0.51x
             n=3000 (69 MB) 0.55x | 3600 (99 MB) 0.78x
           It closes, but never reaches parity at any size worth optimising.
        2. THE UNBLOCKED UPDATE IS ALREADY PERFECT FOR OUR LAYOUT. Because the
           factorisation transposes into column-major, every trailing column a
           reflector touches is CONTIGUOUS — two streaming passes, vectorised,
           parallel over columns with every thread busy. Blocking replaces that
           with packing, a Gram matrix, a triangular solve and two gemms in a
           panel shape (N = 48) that mstore::gemm is not tuned for; it tiles in
           blocks of 64 and parallelises over M, so a product with M = 48 runs
           SINGLE-THREADED. That one detail alone cost 5x and is invisible in
           the flop count.
      The lesson generalises past QR: level-3 beats level-2 only when the data
      does not fit in cache AND the level-2 access pattern is poor. Neither
      holds here. LAPACK blocks because it targets machines where the matrix
      is far larger than cache; Eigen blocks its unpivoted QR for the same
      reason. Copying the structure without checking the premise loses 2x.

  [ ] STILL UNBLOCKED IN THE OTHER SENSE.
      into a WY block so the trailing update is a matrix multiply. Column
      pivoting rules that out (the norms must be downdated before the next
      pivot is chosen), so matching dgeqrf would mean offering an unpivoted
      path as well. We are 8.7x ahead of NumPy anyway, because NumPy's qr
      returns the full m x m Q.

─────────────────────────────────────────────────────────────────────────────
 GAP AGAINST BASE MATLAB  (no toolboxes, plotting ignored)
─────────────────────────────────────────────────────────────────────────────

A NOTE ON WHAT DOES *NOT* GET AN OPERATOR.
% is the element-wise (Hadamard) product and stays that way. Moving it to the
Kronecker product was considered — kron() is the only element-wise-adjacent
operation without an operator, and % would have freed the name `tensor` — and
rejected for three reasons:
  * Operators should go to FREQUENT, CHEAP operations. Hadamard is O(n²) and
    everywhere; Kronecker is O(n⁴) and rare. Two 1000x1000 matrices kron to
    10^12 elements, 8 TB. That belongs behind a name you have to type, not
    two characters. Our own benchmark is the evidence: every other operation
    runs to n = 512 or 2000, and this one stops at 48.
  * % meaning Hadamard is the established C++ convention (Armadillo). Giving
    it a different meaning here would mislead rather than merely surprise.
  * MATLAB has no operator for it either — it is kron(A, B) there too — so an
    operator would buy nothing in the MATLAB fidelity this header is aiming
    at.
The name collision that prompted the question was fixed at its source
instead: tensor() became kron(), so an operation no longer shares a name with
a type.

OPERATOR MAPPING — settled, and the one deliberate divergence is documented.
The organising rule is that THE MEMBER DOT MEANS ELEMENT-WISE: A.sin() is
element-wise and sin(A) is the matrix function, A.pow(n) is element-wise and
pow(A, n) is the matrix power. MATLAB's leading dot, put where C++ can hold
it.
    MATLAB    here                       note
    A * B     A * B
    A .* B    A.mul(B) / A % B / A*dot*B
    kron(A,B) A.kron(B) / kron(A,B)      no operator, deliberately — see
    above A / B     A / B                      right division; CHANGED to
    match A ./ B    A.div(B) / A /dot/ B       C++ cannot spell a leading dot
    A \ B     A.solve(B)                 no operator\ in C++
    A ^ n     pow(A, n)                  ^ has the wrong precedence in C++
    A .^ n    A.pow(n)
    sin(A)    sin(A)                     matrix function (free)
    sin(A) elementwise   A.sin()         element-wise (member)
    A'        A.H()                      conjugate transpose — a member that
    A.'       A.T()                      is NOT element-wise, but harmlessly
                                         so: there is no element-wise T

[x] TIER 1 — THE LOGICAL / MASKING LAYER. DONE. See the "Logical masks"
    section in the class for the full API and the reasoning; in brief:
      - a mask is a Matrix<bool>, so it is an ordinary matrix and inherits
        shape, printing, T(), slicing and the rest for nothing;
      - <  >  <=  >=  are ELEMENT-WISE operators, because no matrix-level
        ordering exists for them to be confused with — the same licence that
        lets A.T() be a non-element-wise member;
      - ==  != stay WHOLE-MATRIX and return bool, because that meaning does
        exist and `if (A == B)` is the idiom every C++ programmer reaches
        for. .eq() / .ne() are the element-wise forms. This is a deliberate
        divergence from MATLAB and the only one in the comparison family;
      - logic in C syntax: &&, || and !, plus named land / lor / lxor / lnot.
        NOT NumPy's & and | — `A | B` is already the augmented-matrix
        operator, and taking & for `and` while `or` needed a named function
        would have been lopsided, so the whole triple went to C instead. The
        trap for NumPy habits survives — `(A>0) | (B>0)` concatenates — and
        is written down at the definition rather than left to be discovered.
        Overloading && and || costs short-circuiting, which an element-wise
        or never had; and since Matrix<bool> has no conversion to bool, `if
        (m1 || m2)` does not compile, which is a safety win. Both checked;
      - any / all (both with the sum(bool) axis forms) / nnz / find;
      - logical indexing both ways: A(mask) reads a column vector, and
        A(mask) = scalar-or-vector writes through a MaskProxy.
    Counting a mask goes through nnz(), not sum(): sum() returns datatype,
    and for Matrix<bool> that saturates at true instead of counting. 46
    assertions in validate.cpp, 9 more cross-checked against NumPy. [x]
    Tensor has the same layer, same spellings — 29 more assertions. The
        two deliberate differences: Tensor::find() returns FLAT row-major
        indices (a tensor's positions are rank-long, so Matrix's (row, col)
        pairs do not generalise; unravel() converts one back), and Tensor
        uses
        || even though it has no augmented-tensor operator to avoid —
        matching Matrix matters more than claiming the free slot, since |
        meaning `or` on a Tensor and `concatenate` on a Matrix would be a
        worse trap.

[x] TIER 2 — REDUCTIONS. DONE for Matrix. prod, cumsum, cumprod, diff, sort,
    sortrows, median, mode and unique are all in, each following the same
    axis convention as sum(bool): 0 works DOWN columns, 1 ALONG rows, so
    prod(false) pairs with sum(false) and cumsum(false) accumulates down the
    same axis sum(false) totals. Scans (cumsum/cumprod/sort) keep the input
    shape; diff shrinks the scanned axis by one; median returns double
    because an even count averages the middle two. prod/cumsum/cumprod work
    for complex, and everything that has to ORDER elements static_asserts
    against it for the reason min()/max() already give. 35 assertions in
    validate.cpp, 17 cross-checked against NumPy. STILL OPEN: Tensor has none
    of these yet — it should get the same set, and unlike the mask layer the
    axis handling is genuinely different there (an arbitrary axis rather than
    a bool), so it is not a copy-paste.

[x] TIER 3 — ELEMENT-WISE MATH. DONE. sign, floor, ceil, round, fix, mod,
    rem, atan2, hypot, angle/arg, asinh/acosh/atanh, expm1 and log1p are all
    in, all members (so all element-wise, by the rule), all through mapElems
    so all inheriting its threading and restrict-qualified loop.
    Three things worth knowing:
      - mod and rem are NOT the same function. mod follows the DIVISOR's
        sign, rem the DIVIDEND's: mod(-1,3) is 2, rem(-1,3) is -1. rem is
        C's fmod; mod is the one you want for wrapping an index or an angle.
      - floor and fix differ on negatives: floor(-2.5) is -3, fix(-2.5) is
      -2.
      - angle()/arg() closes the complex gap this list called out — real(),
        imag() and conj() were all here but there was no way to get a phase.
        sign() is defined for complex too, as z/|z|, so sign(z)*abs(z) == z
        holds in both cases. The rounding family static_asserts against
        complex, since there is no "largest integer below" a complex number.
    29 assertions in validate.cpp, 13 cross-checked against NumPy.

[x] TIER 4 — LINEAR ALGEBRA. DONE except QZ, which is called out below.
    [x] null, orth      column subsets of V / U from svd(). Bases are not
                        unique, so the NumPy cross-check compares the
                        PROJECTORS they define, which are.
    [x] roots           companion matrix + eigvals(), which is how MATLAB and
                        NumPy both do it — the QR iteration is backward
                        stable where deflation is not. Leading zeros are
                        stripped and trailing zeros become exact roots at 0.
    [x] hess            A = Q H Qᵀ. Written out rather than lifted from
                        schurDecomp: that would have meant surgery on the
                        routine eig, pow and log all depend on, to save 30
                        lines. The duplication is deliberate.
    [x] schur           public wrapper over schurDecomp, returning matrices.
    [x] polyfit, polyval  polyfit goes through solve()'s pivoted QR rather
                        than the normal equations, which would square the
                        condition number. polyval is Horner.
    [x] dot, cross      dot conjugates the LEFT operand, so A.dot(A) is
                        ||A||_F² for complex as well as real.
    [x] rref            Gauss-Jordan with partial pivoting. Documented as a
                        teaching tool: on floating-point data the
                        pivot-is-zero decision is a guess. Use rank(), null()
                        and solve() for anything numerical.
    [x] IsSymmetric, IsHermitian, IsUpper, IsLower, IsBanded, bandwidth.
                        Named to match the IsDiagonal() that was already here
                        rather than MATLAB's lowercase — a predicate family
                        that agrees with itself beats one that is half and
                        half. All take a tolerance relative to the entries.
    [x] normest         power iteration on AᵀA, for when norm(A, Two)'s full
                        SVD is a great deal of work to throw away.
    [x] rcond, condest  the Hager-Higham 1-norm estimator (LAPACK's dlacn2 /
                        dgecon), driven from the factors Decomposition
                        already holds. It finds a vector that nearly
                        maximises
                        ||A⁻¹x||₁/||x||₁ in a handful of solves, instead of
                        forming the inverse. Measured at n=512: 9.6 ms
                        against 302 ms for a full cond(Two) — 31x. It is a
                        LOWER bound on the true condition number, never
                        pessimistic; on a 5x5 Hilbert matrix it lands on it
                        exactly.
    [x] lsqminnorm      minimum-norm least squares, through the SVD, so it is
                        defined for a rank-deficient A too — which is when it
                        is actually wanted. solve() still refuses an
                        under-determined system, on the grounds that
                        "infinitely many solutions" is usually a mistake
                        worth being told about; this is the escape hatch.
    [x] decomposition   DONE, and it was the highest value-per-line item as
                        predicted. A.factorize() picks the factorisation from
                        the STRUCTURE, the way MATLAB's decomposition does:
                        symmetric positive definite -> Cholesky (half the
                        flops, and the attempt is itself the definiteness
                        test), square otherwise -> LU, rectangular -> pivoted
                        QR. It owns its factors, so it outlives the matrix it
                        came from. Measured, 100 right-hand sides:
                            n=256   104 ms -> 3.1 ms   (33x)
                            n=512   810 ms -> 16.3 ms  (50x)
                        solve() and Decomposition share ONE substitution
                        routine (luSubstitute), so the parallel-over-RHS path
                        exists in exactly one place.
    [x] funm            DONE, and NUMERIC not symbolic: f is any callable
                        complex -> complex, and nothing here differentiates or
                        expands it. The 2x2 block problem this entry used to
                        describe is solved by converting the real Schur form
                        to a genuinely complex triangular one first, which
                        makes the Parlett recurrence scalar throughout and
                        removes the need for a Sylvester solve per block.
                        The price of staying numeric is stated rather than
                        hidden: Higham's robust algorithm reorders into
                        clusters and Taylor-expands each block, needing f',
                        f''. Without those, close eigenvalues make the Parlett
                        division meaningless — so funm DETECTS that and
                        throws, naming the two eigenvalues and pointing at
                        exp/log/sqrt/sin/cos/sinh/cosh/tanh/pow, none of which
                        go through Parlett and none of which have any
                        eigenvalue-separation requirement.
    [x] eig(A,B)        symmetric-definite, by Cholesky reduction — the
                        backward-stable route, and what LAPACK's dsygv does.
                        Eigenvalues ascending, eigenvectors B-ORTHONORMAL
                        (X^T B X = I) rather than Euclidean-normalised, which
                        is the right normalisation here and falls out of the
                        reduction for free.
    [x] eigvals(A,B)    the general pencil via B^-1 A, GUARDED by the
                        Hager-Higham condition estimate: it refuses outright
                        when B is near-singular rather than returning
                        plausible numbers, and names QZ as what is needed.
    [x] polyeig         matrix polynomial eigenvalues by companion
                        linearisation to a d*n pencil, then eigvals(A,B), so
                        it inherits that same guard.
    [ ] qz              STILL OPEN, and the one real gap left in this tier.
                        A general pencil with a singular or ill-conditioned B
                        needs the generalized Schur decomposition, with its
                        own Hessenberg-triangular reduction and shifted
                        sweeps. Everything above refuses that case loudly
                        instead of guessing at it.

[x] TIER 5 — CONSTRUCTION AND SHAPE. DONE. numel, repmat, fliplr, flipud,
    rot90, circshift, blkdiag as members; linspace, logspace, range, randn,
    randi, randperm and the structured matrices as free functions.
    Four things worth knowing:
      - SEQUENCES GO THROUGH std::iota, the standard library's own sequence
        generator. The index run is exact (0,1,2,... in long), so the only
        floating point in a linspace is the single multiply that scales it;
        accumulating v += step instead would drift, and drift further the
        longer the vector.
      - linspace SETS its endpoint rather than computing it. a+i*(b-a)/(n-1)
        does not reliably land on b, and "does linspace(0,1,101) contain
        exactly 1.0" is a question people write loops around. NumPy and
        MATLAB both special-case it. linspace(a,b,1) returns b, as MATLAB
        does.
      - EVERY RANDOM CONSTRUCTOR USES ran2() FROM random.hpp, never
      std::rand,
        whose low bits are poor and whose period can be 32767. Seeds stay
        negative, matching set_Ran_values. randn is Box-Muller (two draws per
        pair, so a seed stays reproducible), randperm is Fisher-Yates (the
        only shuffle uniform over all n! orderings). NOT THREAD SAFE: ran2
        keeps static state, so every filler here is deliberately serial — the
        one place the OpenMP treatment the rest of the header gets would be
        actively wrong.
      - magic() implements all three of MATLAB's cases (odd by the Siamese
        method, doubly even by a complement pattern, singly even by LUX), and
        the tests check rows, columns, BOTH diagonals, and that 1..n² each
        appear once, for n = 3, 4, 5, 6 and 8.
    hilb and wilkinson now earn their keep in validate.cpp as the
    ill-conditioning and eigenvalue stress cases they were wanted for.
    67 assertions in validate.cpp, 15 cross-checked against NumPy/SciPy.

[x] TIER 6 — IN BASE MATLAB, OUTSIDE LINEAR ALGEBRA. DONE.
    polyval, polyfit and roots landed with tier 4; fft/ifft/fftshift and then
    conv, deconv, poly, trapz, cumtrapz, gradient, interp1 and filter here.
    The decisions worth remembering:
      - AXIS OPERATIONS ARE MEMBERS, SIGNAL OPERATIONS ARE FREE. trapz,
        cumtrapz and gradient take the same axis flag as sum() and cumsum()
        because that is what they are; conv, filter, interp1 and the rest
        treat a whole vector as one signal and stay free. MATLAB spells the
        first group free too, but this header already chose otherwise for
        cumsum and diff, and agreeing with itself matters more.
      - conv SWITCHES TO THE FFT above 16384 multiply-adds, and that number
        was measured: the first guess of 100000 was ten times too high and
        would have skipped the FFT across a range where it is twice as fast.
        The table is at CONV_FFT_MIN_WORK. This is also the one place where
        zero-padding an FFT is CORRECT — the longer signal is the answer
        wanted — which is exactly why fft() itself refuses to pad silently.
      - gradient KEEPS THE LENGTH where diff() shortens it, by using a centred
        difference inside and a one-sided one at each end. That is the whole
        reason both exist.
      - interp1 RETURNS ITS FILL VALUE outside the range, defaulting to NaN
        as MATLAB does; NumPy's np.interp clamps instead. ⚠ The NaN default
        does NOT survive -ffast-math, which implies -ffinite-math-only and
        folds isnan() to false — the third time that flag has bitten this
        header, and the first where a FEATURE rather than a test broke. The
        explicit fill argument is MATLAB's own escape hatch and works in
        every build.
      - filter IS A RECURRENCE, so alone in this tier it cannot be
        parallelised over the output — y(n) depends on y(n-1).
    33 assertions in validate.cpp, 12 more cross-checked against
    NumPy/SciPy (np.convolve, np.polydiv, np.poly, np.gradient, np.interp,
    scipy.signal.lfilter, scipy.integrate.cumulative_trapezoid).
    Further out and still not this library's job: ode45, fzero, fminsearch,
    integral.

[x] COMPLEX SUPPORT — the real-only list is now short, and what remains is
    one algorithm rather than a scatter of static_asserts.
    The mechanism is work_t<T> at the top of this file: double for a real
    element type, complex<double> for a complex one. Every routine that used
    to convert to Matrix<double> — and therefore had to REFUSE a complex
    input rather than silently drop its imaginary part — now converts to
    work_t and returns Matrix<work_t> instead.

    [x] mean   follows the input (the mean of complex numbers is complex);
        var    stays REAL, because it is E|x-mu|^2 and cannot be complex.
               NumPy's np.var agrees.
    [x] cholesky   A = L*L^H for complex, and the input must be HERMITIAN
                   rather than merely symmetric — it is A == A^H that forces
                   the diagonal real and the pivots positive. A complex
                   symmetric matrix is refused, which is a distinction that
                   does not exist in the real case and is the whole game here.
    [x] svd        U and V come back UNITARY; S stays real, since singular
                   values are magnitudes. A complex inner product cannot be
                   zeroed by a real Jacobi rotation, so its PHASE is rotated
                   out first (scaling the column by conj(g/|g|), a unitary
                   step V absorbs) and the ordinary real rotation then
                   applies unchanged. The real path keeps the signed value so
                   its sign convention is untouched.
    [x] QR         complex Householder: alpha points away from x[0] in PHASE
                   rather than sign, and every inner product is conj(v)·w.
                   Q is unitary. This unblocked rank, orth, null and the
                   least-squares branch of solve.
    [x] LU, det, solve, inverse, luPacked   pivot on |a|, which is the right
                   reading of partial pivoting for complex too.
    [x] exp, sin, cos, tan, sinh, cosh, tanh   the Taylor machinery is now
                   templated on work_t, so the SAME series serves both — only
                   the arithmetic differs. exp(iA) == cos(A) + i*sin(A) is
                   asserted, which only means anything for complex.
    [x] norm(Two), cond, rank, pinv   all fell out of svd and QR.
    35 assertions in validate.cpp, 8 more cross-checked against NumPy/SciPy.

    [x] THE LAST REAL-ONLY GROUP IS DONE. eig, eigvals, hess, schur, funm,
        log(A), sqrt(A), pow(A, real) and eig(A,B) all used to route through
        schurDecomp(), the REAL Schur form, which a complex matrix does not
        have. schurDecompComplex() supplies the genuine one — complex
        Hessenberg reduction plus a single-shift QR iteration — and every one
        of those now dispatches on is_complex.
        The complex form is TRIANGULAR rather than quasi-triangular, so the
        complex paths came out simpler than the real ones: no 2x2 blocks to
        detect, no complexify() step in funm, no real-eigenvalue restriction
        on eig(). Cross-checked against scipy.linalg schur/hessenberg/sqrtm/
        logm/expm, and funm(exp) additionally against an independent Taylor
        series, since a bug shared between our Schur and our funm would
        otherwise cancel out.

        TWO SILENT REAL BUGS FELL OUT OF DOING THIS, both the same mistake in
        different places — trusting a real Schur form to say something it
        cannot:
          1. eig() returned the SCHUR VECTORS as eigenvectors. Only the first
             column of Q is ever an eigenvector. Right eigenvalues, wrong
             vectors, no error — and invisible for symmetric input, where the
             Schur form is diagonal and the two coincide, which is why every
             existing test passed. Fixed with the back-substitution in
             eigenvectorsFromSchur().
          2. sqrt(A), log(A) and pow(A, real) read a 2x2 block as its real
             part TWICE. That part is normally positive, so the
             positive-eigenvalue check passed and the answer came back
             silently wrong: ||R*R - A|| = 6.7e-01 on a 4x4 whose eigenvalues
             were 6.084 +- 0.403i. They now detect the block and route through
             funm, which complexifies it properly.
        Both are pinned in validate.cpp with the exact matrices that exposed
        them.

    [x] Decomposition / factorize() IS COMPLEX TOO. It stored Matrix<double>
        and began solve() by taking std::real() of the right-hand side, which
        silently discarded the imaginary part of every B. All three paths now
        run in work_t: LU, Cholesky (dispatching on IsHermitian, since a
        complex SYMMETRIC matrix is not Cholesky-able) and QR least squares,
        plus det() and the Hager-Higham rcond estimator — whose sign vector
        becomes a PHASE vector y/|y| for complex, exactly as LAPACK's zlacn2
        does, reducing to +-1 when the imaginary part is zero.

    [x] QZ — the generalized Schur decomposition — IS IN, as qz(A,B):
            Q^H A Z = S      Q^H B Z = T      both upper triangular
        Eigenvalues are the RATIOS S(i,i)/T(i,i), and nothing ever forms
        B^-1*A. That was the whole point: eigvals(A,B) used to refuse
        outright below rcond(B) = 1e-10 because forming that product spends
        the precision before the eigensolver starts. It now goes through QZ
        and agrees with scipy.linalg.eig(A,B) to 2.3e-13 at exactly that
        conditioning.
        Structure: Hessenberg-triangular reduction (an unpivoted QR of B,
        then paired rotations — every left rotation that tidies A spoils B,
        and the right rotation repairing B is what keeps the pair moving
        together), then an implicit SINGLE-shift bulge chase. Single shift
        because it runs in complex arithmetic; a real QZ needs a double shift
        and leaves S quasi-triangular, the same trade schurDecomp makes.
        The reduction stays in the pencil's own type, so a real pencil pays
        real cost for that O(n^3) stage; only the sweep is complex.
        Verified to machine precision for n up to 40, real and complex, and
        against scipy.linalg.eig(A,B).

    [x] QZ IS COMPLETE. Infinite eigenvalues, eigenvectors and singular
        pencils are all handled:
        - INFINITE eigenvalues (beta == 0, which is what a singular B gives
          and B^-1*A cannot express at all) deflate at either END of the
          active block, where the entries of T the rotation would touch are
          provably already zero. An INTERIOR zero needs no special handling —
          the ordinary sweep drives it to an end on its own. That was checked
          rather than assumed: singular pencils of rank deficiency 1, 2 and 3
          agree with scipy.linalg.eig(A,B) to 1e-15 as (alpha,beta) pairs on
          the Riemann sphere.
          An earlier version chased interior zeros explicitly and was WRONG:
          a rotation zeroing T(k+1,k+1) leaves T(k,k) zero too, because both
          entries it draws on are already zero, so the zero SPREADS along the
          diagonal instead of moving — n-1 infinite eigenvalues reported for a
          pencil with one. Deleting that code fixed it.
        - EIGENVECTORS, by back-substitution on the triangular pair in the
          HOMOGENEOUS form (beta*A - alpha*B) x = 0. Dividing to get lambda
          first would produce NaN for exactly the eigenvalues QZ exists to
          handle; the homogeneous form stays finite, and an infinite
          eigenvalue comes back with a genuine vector satisfying B x = 0
          (measured at 6.6e-16).
        - A SINGULAR PENCIL — A and B sharing a null space, so
          det(A - lambda B) vanishes identically and NO eigenvalue is
          determined — is reported by undefined() rather than passed off as
          an answer. The pairs are 0/0 and the values that come back are
          rounding noise that looks like ordinary numbers. On a 20x20 with a
          shared 12-dimensional null space: 12 undefined pairs here and 12
          from scipy, with the remaining 8 agreeing to 1.1e-15.
        - ILL-SCALED pencils. The shift is a ratio of products of S and T
          entries, so ||A|| ~ 1e9 against ||B|| ~ 1e-9 walks it through
          eighteen orders of magnitude and the tolerances stop meaning
          anything: 70 of 400 stress pencils failed to converge before qz()
          normalised the pair. Scaling is undone on the FACTORS afterwards,
          so A == Q S Z^H still holds to the bit. After the fix: 400/400, no
          failures, worst relative residual 2.9e-15.

    [x] THE GEMM PARALLELISES OVER BOTH DIMENSIONS NOW, and that was the
        real blocker behind the blocked QR. It tiled C over M only, so a
        product with M < 64 had exactly ONE tile and ran SINGLE-THREADED —
        6.5 GFLOP/s on 32 cores against 8.0 on one, the OpenMP entry costing
        more than the parallelism it failed to provide. That is not a corner
        case; it is the shape every panel algorithm produces.
        Three changes, each measured interleaved against the old kernel so
        both saw identical machine load:
          1. The unit of work is a tile of C indexed over M AND N. Tiles are
             disjoint in C so both dimensions parallelise safely, and the K
             loop moved inside the tile, which also keeps one 32 KB tile of C
             hot across the whole K sweep instead of re-reading each row-block
             K/64 times.
          2. The dynamic CHUNK is sized from the work in a tile (mB*BLOCK*K)
             rather than left at 1. At K=48 a tile is tiny and per-grab cost
             dominates — 26.9 GFLOP/s at chunk 1 against 68.9 coarser — while
             at K=1024 the coarse grain loses to imbalance. Scaling it gets
             both.
          3. The M block SHRINKS when N is smaller than a block, because then
             there is only one column of tiles and all the parallelism must
             come from M: 16 tasks for 32 threads at M=1024, N=48.
        Measured, interleaved, min of 7:
            square 1024        46.3 ->  60.8 GFLOP/s   1.31x
            square 2048        68.7 ->  67.4            0.98x (no-op, as intended)
            C -= V W  (K=48)   20.1 ->  25.2            1.26x
            V^H C     (M=48)    4.2 ->  31.6            7.56x
            V^H C  (M=48) big   5.0 ->  50.6           10.10x
            W = C^T V (N=48)   39.8 ->  74.9            1.88x
            very skinny (M=16)  4.9 ->  48.2            9.90x
        Bit-identical output on a 137x137 (deliberately not a multiple of 64).
        This lifts every level-3 path in the header, not just QR: Tensor
        contractions, the Taylor matrix functions, naiveMul.

    [x] THE REAL SCHUR IS FIXED, and it was TWO bugs, one hiding the other.
        FIRST, the shift could not converge on a complex pair:
            sigma = (disc >= 0.0) ? <Wilkinson root> : d;
        With a complex trailing 2x2 it fell back to d, a REAL shift, and no
        real shift converges to a conjugate pair. Replaced with the Francis
        DOUBLE shift (LAPACK's dlahqr): form only the first column of
        (H - l1 I)(H - l2 I) = H^2 - sH + tI, which Hessenberg structure
        leaves with three nonzero entries, then chase the bulge with 3x3
        reflectors. Both roots are applied at once and the arithmetic stays
        real.
        SECOND — and this only became visible once the first was fixed, since
        the Francis step started reaching 2x2 blocks the old iteration never
        did — split2x2() computed its discriminant as
            ((a+d)/2)^2 - (a*d - b*c)
        which CANCELS CATASTROPHICALLY when a and d are close: for
        a = d = 1e3, b = c = 1e-3 the true 1e-6 is the difference of two
        numbers near 1e6 and keeps four digits. The rotation built from it did
        not zero the sub-diagonal, and the routine then FORCED h(k+1,k) = 0
        anyway — breaking the similarity in silence. Replaced with LAPACK's
        dlanv2: the discriminant is p^2 + b*c with p = (a-d)/2 (no
        cancellation), everything is scaled, the second eigenvalue comes from
        the PRODUCT rather than the other root, and the standardised block is
        written out rather than hoped for.
        Finding the second one took an invariant check inside the loop —
        ||Q^T A Q - H|| after every reflector — which held at 2e-16 through
        the whole Francis sweep and jumped to 4.8e-10 across one split2x2
        call. Guessing at ranges had already cost three wrong attempts.
        500 real matrices (random, near-identity, symmetric, triangular,
        1e6-scaled), before and after:
            before:  clean 460   bad 2    THREW 38   worst rel-resid 7.9e-04
            after:   clean 500   bad 0    threw  0   worst rel-resid 4.3e-15
        The strongest single check is symmetric input: an orthogonal
        similarity must keep it symmetric, and symmetric plus zero
        sub-diagonal means DIAGONAL. That is what exposed the second bug,
        where H came back triangular with a nonzero upper triangle.

    [x] QR IS FINISHED. Both level-3 variants are now implemented, measured
        and rejected, and the unblocked level-2 path stands.

        RECURSIVE QR (Elmroth-Gustavson) was the last untried option, and the
        argument for it was real: fixed blocking always produces skinny N=48
        panels, the shape the GEMM is worst at, while a recursion halves the
        block width so the TOP level is one large square-ish product. That part
        held up — measured standalone, the root GEMMs run at 74.5 GFLOP/s,
        indistinguishable from a full 1024^3 square (77.2). It still lost:

            size        unblocked   recursive   ratio
            512x512        24.3        30.5      0.80x
            1024x1024      62.3       116.4      0.54x
            1500x1500     147.1       324.0      0.45x
            2048x2048     375.7       730.5      0.51x
            4000x200       20.1        54.4      0.37x
            2000x500       61.7       158.0      0.39x

        Leaf sizes 32, 64, 128, 256 and 512 were all tried; 32 is best and
        larger is monotonically worse, so "too many small GEMMs at depth" is
        NOT the explanation — raising the leaf only trades them for level-2
        panel work. Across the L3 cliff it behaves exactly as the blocked
        version did, 0.52x at 32 MB narrowing to 0.72x at 99 MB, and never
        crosses.

        WHY IT LOSES: the recursion does roughly 2.3x the arithmetic of the
        unblocked path, because every level pays for a T — a Gram matrix
        V1^H V2 plus two triangular multiplies. The root GEMMs are at full
        speed but the deeper ones are not (33.6 GFLOP/s at level 1, 22.6 at
        level 3), and 2.3x the work at a mixed rate does not beat 1x at the
        level-2 kernel's 22 GFLOP/s.

        TWO IMPLEMENTATION TRAPS, both the same mistake, both worth recording
        because the first version was 27x SLOWER than unblocked and it would
        have been easy to stop there and call the algorithm bad:
          1. The T-combine's two triangular multiplies were hand-written triple
             loops. O(b^3) each, and b is n/2 at the root.
          2. Worse, `W^T <- W^T conj(T)` inside the block update was also a
             hand-written triangular loop — O(bs^2 * trailing), serial. At the
             fixed block size of 48 that is 4.5% of the blocked version and
             invisible; at bs = n/2 it was 84.6% OF THE ENTIRE FACTORISATION.
             Profiling found it; three rounds of guessing had not.
          Fixing both took 2048x2048 from 11.6 s to 0.73 s. The lesson is that
          a level-3 algorithm has NO room for a level-2 helper hiding inside
          it, and that the phase profile is the way to find one.

        So: the level-2 unblocked path is the QR, for this library, on this
        hardware. What would change that is a faster GEMM KERNEL, not another
        QR algorithm — and that has its own measured ceiling of 2.2x threaded,
        which is not enough. See the GEMM notes.
        - The fixed-block BLOCKED QR IS REJECTED A SECOND TIME, now that the
          GEMM parallelises properly. This was the obvious follow-up — the
          panel GEMM that ran single-threaded was named as the headline
          reason blocking lost — so it was re-implemented and re-measured
          rather than assumed. It is barely different:
              size        before GEMM fix   after GEMM fix
              2048x2048        0.56x             0.53x
              2600x2600        0.51x             0.52x
              3000x3000 (>L3)  0.55x             0.58x
              3600x3600 (>L3)  0.78x             0.83x
          A 7-10x faster panel GEMM moved it by a few percent, which says
          plainly that the GEMM was NOT blocked QR's bottleneck. Profiling the
          phases says where the time actually goes (n=1024, NB=64):
              panel (level-2)        5.8%
              build V                1.2%
              Gram + T recurrence   26.3%
              pack C                 4.0%
              gemm W = C^T V        28.7%
              apply T to W           4.5%
              gemm C -= W V^T       29.1%
              unpack C               0.5%
          Two things follow. First, roughly 40% is PURE OVERHEAD the unblocked
          path never pays — the Gram matrix, the T recurrence, packing and
          unpacking. Second, and decisively, THE TWO GEMMS ALONE (58%, 158 ms)
          COST MORE THAN THE ENTIRE UNBLOCKED FACTORISATION (85 ms). No amount
          of tuning around the edges fixes that; only a faster GEMM would.
          Block sizes 32, 48, 64, 96, 128 and 192 were all tried. NB=64 is
          best and still loses.
        - WOULD A FASTER GEMM MAKE LEVEL-3 QR WIN? MEASURED: NO. A packed
          AVX-512 microkernel was prototyped and it is 2.2x on 32 threads
          (5.6x on one core, but the threaded case hits the memory wall
          first). The GEMMs are about half of blocked QR's time, so 2.2x on
          that half takes the blocked/unblocked ratio from 0.53x to roughly
          0.73x — still a loss. The ~40% that is Gram matrix, T recurrence
          and packing does not move at all.
          An earlier version of this note claimed a faster GEMM would flip
          it. That was arithmetic done before the prototype existed, and it
          was wrong. A faster GEMM is worth having for multiplication,
          Tensor contractions and the Taylor matrix functions — not as a way
          in to level-3 QR.

─────────────────────────────────────────────────────────────────────────────
 C++26 std::linalg (P1673) — what is worth taking, measured
─────────────────────────────────────────────────────────────────────────────

C++26 adds <linalg>: free functions over std::mdspan giving BLAS 1/2/3 —
matrix_product, matrix_vector_product, triangular_matrix_matrix_solve, dot,
vector_norm2, matrix_frob_norm, the rank-k updates, and so on.

FIRST, THE LIMIT: it is a BLAS, NOT a LAPACK. There is no LU, QR, SVD, eig or
Cholesky in it. Every factorisation in this header stays ours; what
std::linalg could ever replace is naiveMul and the norms. So it is a source
of DESIGN ideas here, not a dependency to plan around.

Availability checked on this toolchain (GCC 13.3): <mdspan> and <linalg> do
not exist at any -std setting. mdspan lands in libstdc++ 15. So none of the
below can be adopted directly yet — but two of the three ideas can be copied
without the header, and one turns out not to be worth copying at all.

[ ] IDEA 1 — LAZY transposed() / conjugated() / scaled() VIEWS.
    std::linalg never materialises a transpose: matrix_product(transposed(A),
    B, C) reads A in place with swapped indices. Obviously appealing, since
    A.T() * B here builds a whole temporary first.
    MEASURED, AND IT DOES NOT PAY:
        n=512    A*B 5.03 ms | A.T()*B 7.65 ms | direct Aᵀ·B kernel 8.02 ms
        n=1024   A*B 13.9 ms | A.T()*B 17.1 ms | direct Aᵀ·B kernel 16.8 ms
    i.e. 0.95x and 1.02x — a wash. The transpose KERNEL is only ~0.7% of
    A.T()*B (T() alone is 0.07 ms of 7.65 ms); what makes the product slower
    is the Aᵀ access pattern inside the GEMM, and a lazy view has exactly the
    same access pattern. Materialising actually converts the bad pattern into
    a good one and pays for itself. Verdict: do NOT build a lazy-view layer.
    For the record, the three transposes inside operator/ cost 1.4% at n=512
    and 0.5% at n=1024 — also not worth removing.

[x] IDEA 2 — OUT-PARAMETER AND "UPDATING" FORMS. This is the one that pays.
    std::linalg writes into a caller-supplied output: matrix_product(A,B,C)
    sets C = A*B, and matrix_product(A,B,E,C) sets C = E + A*B in one pass.
    Nothing allocates, and the accumulate is fused.
    Every iterative routine here allocates a fresh temporary per step
    instead. Measured on the shape of a Taylor / scaling-and-squaring loop (T
    = T*A/k ; S = S + T, 18 terms):
        n=128   loop 3.71 ms   vs 18 raw products 2.35 ms   -> 37% overhead
        n=256   loop 33.5 ms   vs 18 raw products 10.6 ms   -> 68% overhead
    At n=256 the loop costs 3.2x the arithmetic it actually performs. An
    internal multiplyInto(A, B, C) plus a fused C += A*B would take most of
    that back, and exp/sin/cos/sinh/cosh/log/pow all go through this shape.
    Highest-value item to come out of reading P1673.

[ ] IDEA 3 — mdspan's LAYOUT VOCABULARY for the tensor work below.
    layout_right / layout_left / layout_stride is exactly the
    shape-and-stride model the N-dimensional survey lands on, and layout_left
    is precisely the column-major working array that made svd and qr 10-50x
    faster. Matching the naming and the stride semantics costs nothing now
    and means a Tensor<T> can later hand out a real std::mdspan — and
    interoperate with std::linalg — without changing its storage.

[ ] Also noted, not needed: std::linalg's ExecutionPolicy overloads
    (matrix_product(std::execution::par, ...)) are the standard's version of
    what the OpenMP paths here already do by hand.

─────────────────────────────────────────────────────────────────────────────
 GOING N-DIMENSIONAL — a measured survey
─────────────────────────────────────────────────────────────────────────────

MATLAB arrays are N-dimensional; Matrix is strictly rank 2. That is a real
ceiling for the ML and quantum directions — a 5-qubit state is naturally
2x2x2x2x2, and a batch of images is (batch, channel, height, width). Four
layouts were considered and three of them measured, on a
(32, 32, 64, 64) tensor = 4.19M doubles = 33.6 MB:

    operation                nested   vector<Matrix>     flat
    allocate                11.77 ms      ~same        1.91 ms
    element-wise add        14.22 ms     13.29 ms      3.83 ms
    sum all                  0.83 ms      0.79 ms      0.43 ms
    allocations                 1024         1024            1
    contraction to a GEMM        no           no    3.10 ms @ 173 GFLOP/s

STATUS: BUILT. Option C below is implemented in Tensor.hpp — include that
instead of this header to get it. namespace mstore near the top of this file
exists so that Tensor shares the storage, the huge-page advice, the threaded
element-wise driver and the GEMM with Matrix rather than growing a second
copy of each. Measured after building it, on the same tensor as the survey:
    element-wise add  3.74 ms (survey predicted 3.83, nesting was 14.22)
    sum all           0.37 ms (survey predicted 0.43, nesting was 0.83)
    reshape / permute / contiguous   30-50 NANOseconds — metadata only
    contract (1024x4096)*(4096x64)   4.39 ms @ 122 GFLOP/s
    contractInto, same product       3.82 ms @ 141 GFLOP/s  (1.15x, no alloc)
The out-parameter form also beats the same product through Matrix (3.82 ms
against 5.46 ms) purely by not allocating — idea 2 from P1673, paying again.
Cross-checked against NumPy: 14/14 operations agree (tensor_numpy_validate).

Still open on Tensor, in rough order of value:
  [ ] Broadcasting. NumPy's shape rules are not implemented; shapes must
  match
      exactly. This is the biggest missing convenience.
  [ ] A strided element-wise path. Non-contiguous operands are materialised
      first, which is one extra pass; a stride-aware loop would avoid it.
  [ ] Single-qubit gates should not permute at all. Applying H to one qubit
  of
      a 22-qubit state costs 28 ms on the fast axis and 52 ms on a middle
      one, and almost all of it is the permute-and-clone, not the 2x2
      arithmetic. A strided kernel that walks the target axis directly would
      remove it.
  [ ] No small-buffer optimisation, so a 2x2 Tensor allocates where a 2x2
      Matrix does not. Use Matrix for gate-sized objects, or add SBO.
  [ ] Matrix is still a separate class rather than the rank-2 case of Tensor.
      Sharing mstore was the safe 90% of that; unifying the classes would be
      a rewrite of a 5000-line file that currently passes 236 checks.

[ ] OPTION A — Matrix<Matrix<double>>, i.e. nesting the existing template.
    It COMPILES, and element-wise addition even works, which makes it more
    tempting than it should be. Three reasons it is the wrong answer:
      1. operator* THROWS. naiveMul accumulates into datatype(0), and for a
         nested element that is a 0x0 matrix, so the first `0x0 += 64x64`
         fails. This is not a bug to fix but a structural problem: a generic
         algorithm needs a zero of the right SHAPE, and a nested element type
         cannot supply one without knowing the block dimensions. Every
         algorithm here that starts from an accumulator has the same issue.
      2. 1024 separate allocations instead of one, and 6x the allocation
      cost.
         Nothing is contiguous across the outer dimensions, so the huge-page
         work, the __restrict work and the GEMM blocking all stop applying at
         the block boundary.
      3. 3.7x slower on element-wise work, for a layout holding the same
      bytes.
    Verdict: viable only for genuine BLOCK matrices where the blocks are the
    mathematical objects (block-diagonal preconditioners, and so on) — not as
    a general tensor.

[ ] OPTION B — std::vector<Matrix<double>>, a list of rank-2 slices.
    Measurably the same as option A (13.29 ms vs 14.22 ms): still one
    allocation per slice, still no contiguity across the outer index, still
    no way to hand the whole thing to a GEMM. It buys ordinary container
    semantics and nothing else. Useful as a CONTAINER of matrices — a batch
    of independent problems — but not as a tensor.

[x] OPTION C — ONE FLAT BUFFER PLUS SHAPE AND STRIDE METADATA. This is what
    NumPy, PyTorch and TensorFlow all do, and the measurements say the same
    thing: 3.7x faster element-wise, 6x faster to allocate, one allocation.
    The decisive argument is not those numbers though, it is this:

      Every fast tensor contraction in every library is implemented as
      reshape -> permute -> 2-D GEMM -> permute back.

    With a flat buffer, reshape is FREE — it only rewrites the shape
    metadata, no data moves — and the GEMM is the naiveMul that already runs
    at 173 GFLOP/s. The (1024x4096)*(4096x64) contraction above IS that GEMM,
    unmodified. With options A or B the reshape is impossible without first
    copying everything into a flat buffer, at which point you have built
    option C the slow way.
    For quantum circuits this is exactly the operation that matters: applying
    a k-qubit gate to an n-qubit state is reshape, permute the k target axes
    to the front, multiply by the 2^k x 2^k gate, permute back.

[ ] OPTION D — compile-time rank, Tensor<T, N> (Eigen's unsupported Tensor
    module, xtensor). Same flat storage as C, but the rank is a template
    parameter, so index arithmetic unrolls and shape checks happen at compile
    time. Faster still for small ranks, at the cost of rank-generic code
    being hard to write and error messages getting much worse. Worth
    considering LATER as a typed layer over C's storage, not instead of it.

EVEN ON A FLAT BUFFER THE AXIS MATTERS — measured on the same tensor:
    reduce over the last axis   (contiguous)        0.98 ms
    reduce over the first axis  (contiguous passes) 0.77 ms
    reduce over a middle axis   (strided gather)    5.17 ms
A 5.3x spread, which is the whole reason the permute-then-GEMM strategy earns
its keep rather than reducing along whatever axis the caller asked for.

WHAT THIS WOULD MEAN HERE, concretely:
  - Add Tensor<T> holding { std::vector<long> shape, strides; T* data; },
    using the SAME allocator path as Matrix so it inherits adviseHuge() and
    the constructor-free storage.
  - Make Matrix a rank-2 special case over that storage rather than a
  separate
    class, so every optimisation already measured carries over unchanged.
  - reshape() becomes O(1) metadata for Tensor. NOTE it is a COPY today,
  which
    is the right behaviour for a value-semantics Matrix but would be a
    serious performance bug in a tensor.
  - permute()/transpose() sets strides; a materialise() forces contiguity
  when
    a GEMM needs it.
  - Contraction = reshape + permute + naiveMul. No new kernel.
  - Non-contiguous strides mean the element-wise fast paths need a
  "contiguous?"
    test before using the flat restrict loops, with a strided fallback. That
    branch is the main new cost, and it is per-operation, not per-element.
─────────────────────────────────────────────────────────────────────────────

Slice sentinel — used in place of all for row/column extraction.
A dedicated type prevents ambiguity with operator()(int, int).
Usage: A(i, all)  or  A(all, j)

```

## Tier 6 — signal, calculus and interpolation

```text

═══════════════════════════════════════════════════════════════════════════
 TIER 6 — SIGNAL, CALCULUS AND INTERPOLATION            *** WORK IN PROGRESS
═══════════════════════════════════════════════════════════════════════════

Everything below this line is SCAFFOLDING: contracts, conventions and design
decisions written down, implementations deliberately not. Fill them in here.

Already landed from this tier, in the Polynomials block above:
    polyval, polyfit, roots
Still to write, in a sensible order (each is useful on its own, and the later
ones get easier once the earlier ones exist):
    conv, deconv, poly           polynomial arithmetic — no new machinery
    trapz, cumtrapz, gradient    calculus on samples — no new machinery
    interp1                      needs a sorted-lookup helper
    filter                       a recurrence; the one with real edge cases
    fft, ifft                    the big one; see the design notes at the end

── CONVENTIONS THIS HEADER ALREADY COMMITTED TO ──────────────────────────
Worth having in front of you before writing any of these, because breaking
one of them is the kind of thing that only shows up much later:

  * THE MEMBER DOT MARKS ELEMENT-WISE. A.sin() is element-wise, sin(A) is the
    matrix function. So anything here that acts on a whole vector as a signal
    — conv, filter, fft, trapz — is a FREE function, not a member. cumtrapz
    and gradient are the interesting case: they act along an axis, like
    cumsum, so they are arguably members taking the same `axis` flag. Pick
    one and say why in the comment; do not leave it implied.
  * axis = false works DOWN columns, true works ALONG rows. Every reduction
    and scan in the header uses that flag with that meaning.
  * Polynomial coefficients are DESCENDING, matching polyval/polyfit/roots.
    conv and poly must agree, or roots(conv(a,b)) will silently be wrong.
  * Anything random goes through ran2() from random.hpp, never std::rand, and
    stays serial — ran2 keeps static state.
  * A free function cannot use uninit_t; it is private, deliberately. Use the
    ordinary constructor and accept the extra pass, as the tier 5
    constructors do.
  * Throw std::invalid_argument with the ACTUAL sizes in the message. Every
    error path in this header names the numbers it saw.

── THE CONTRACTS ─────────────────────────────────────────────────────────
Signatures are suggestions, but the SEMANTICS are not: these are what MATLAB
and NumPy do, and agreeing with them is what lets numpy_validate.py check the
results rather than just the shapes.

  Matrix<double> conv(const Matrix<A>& a, const Matrix<B>& b)
      Discrete convolution, equivalently polynomial multiplication. Result
      length is na + nb - 1. MATLAB conv, NumPy np.convolve.
      The direct O(na*nb) double loop is the right implementation up to
      roughly a thousand terms; above that the FFT route wins, which is a
      reason to write fft first and come back with a threshold — exactly the
      shape of the STRASSEN_THRESHOLD decision in operator*.

  std::pair<Matrix<double>, Matrix<double>> deconv(const Matrix<A>& y,
                                                   const Matrix<B>& a)
      Polynomial long division: returns {quotient, remainder} with
      y == conv(a, quotient) + remainder. MATLAB returns [q, r].
      Edge case worth deciding: a(0) == 0. MATLAB errors. Leading zeros in
      `a` are the same problem roots() already strips — reuse that reasoning.

  Matrix<double> poly(const Matrix<A>& r)
      The monic polynomial whose roots are r, in descending order, so that
      roots(poly(r)) returns r back up to ordering and rounding. MATLAB poly.
      NOTE MATLAB overloads this: poly(A) for a SQUARE MATRIX gives the
      characteristic polynomial, which is poly(eigvals(A)). Decide whether to
      support that too; if so it needs a separate overload, since a square
      matrix is also a valid list of numbers and the two would be ambiguous
      for a 1x1.

  double trapz(const Matrix<A>& y)                        uniform spacing 1
  double trapz(const Matrix<A>& x, const Matrix<A>& y)    given sample points
      Trapezoidal integral. Sum of (x[i+1]-x[i]) * (y[i]+y[i+1]) / 2.
      With n < 2 samples the integral is 0, not an error — MATLAB agrees.

  cumtrapz, gradient
      Cumulative trapezoid and the central-difference derivative. gradient
      uses a CENTRED difference in the interior and a one-sided difference at
      each end, so the result is the same length as the input — that is the
      whole difference between gradient and diff, and it is worth a comment.

  Matrix<double> interp1(const Matrix<A>& x, const Matrix<A>& y,
                         const Matrix<A>& xi)
      Linear interpolation of the samples (x, y) at the points xi.
      Decisions to make and document:
        - x must be sorted ascending. Check it, or sort internally? Checking
          is cheaper and catches a real class of caller bug.
        - Out-of-range xi: MATLAB returns NaN, NumPy's np.interp clamps to
        the
          end values. They disagree, so pick one and say which.
        - std::lower_bound is the right lookup, and it is already included
        via
          <algorithm>.

  Matrix<double> filter(const Matrix<A>& b, const Matrix<A>& a,
                        const Matrix<A>& x)
      The rational-transfer-function difference equation, MATLAB filter:
          a(0)*y(n) = b(0)*x(n) + b(1)*x(n-1) + ... - a(1)*y(n-1) - ...
      Normalise by a(0); error if it is zero. This is a RECURRENCE, so unlike
      everything else in this tier it cannot be parallelised over the output
      — worth stating in the comment so nobody tries later.

── fft / ifft: the decisions to make before writing a line ───────────────
This is the one with real design freedom, and the choices interact. Directly
relevant to the quantum-circuit goal, since the QFT is exactly this.

  1. WHERE DOES IT LIVE, AND ON WHAT?
     A signal is a vector. Matrix<complex<double>> is the obvious carrier and
     keeps everything in one type. Deciding fft(Matrix) -> Matrix now avoids
     a painful move later.
     MATLAB's fft(A) on a MATRIX transforms each COLUMN. That falls out of
     the axis convention if you want it, and is a natural second overload.

  2. WHAT ABOUT LENGTHS THAT ARE NOT A POWER OF TWO?
     Radix-2 Cooley-Tukey is thirty lines and only handles 2^k. The honest
     options, in increasing order of work:
       (a) radix-2 only, and THROW for other lengths. Clean, and enough for
           quantum work where everything is 2^n by construction.
       (b) radix-2, and zero-pad to the next power of two. Fast and easy, but
           it changes the answer — padding computes the transform of a
           different, longer signal. Only correct for convolution, never for
           spectra. If you take this route, do it INSIDE conv() and not
           inside fft(), or the result will be quietly wrong.
       (c) Bluestein's chirp-z for arbitrary n, which turns any length into a
           power-of-two convolution. Correct for every n, and about a hundred
           more lines.
     (a) now with a clear error message, (c) later, is a defensible path.
     What is NOT defensible is (b) hidden inside fft().

  3. NORMALISATION. MATLAB and NumPy both put the whole 1/n on the INVERSE
  and
     none on the forward transform. Match them — a different convention here
     would make every cross-check against numpy_validate.py fail for a reason
     that has nothing to do with correctness.

  4. THE PROPERTY TO TEST FIRST is ifft(fft(x)) == x. It catches
  normalisation,
     bit-reversal and twiddle-sign errors all at once, and it needs no
     reference implementation. After that, check a known pair by hand — the
     transform of a constant vector is a spike at index 0 — and only then
     compare against numpy.fft in numpy_validate.py.

  5. std::complex<double> already has everything needed: std::polar for the
     twiddle factors, and the arithmetic operators. is_complex<T> and
     real_t<T> at the top of this header are there for writing the guards.

── WHERE THE TESTS GO ────────────────────────────────────────────────────
  validate.cpp        — section("Signal, calculus and interpolation (tier
  6)")
                        is already there, empty, waiting.
  numpy_validate.cpp  — the "tier 6" block is stubbed with the Case() calls
                        commented out; uncomment them as each lands.
  numpy_validate.py   — matching NumPy references are stubbed the same way.
═══════════════════════════════════════════════════════════════════════════

```

---

## Optimising the factorisations (LU, Cholesky, QZ)

Measured against OUR OWN GEMM rather than against NumPy, because NumPy here
links the reference BLAS and beating it proves nothing. At n=1024 the multiply
kernel does ~180-270 GFLOP/s; that is the ceiling everything else is judged by.

    operation      before      after     speedup
    LU (det)      52.2 ms    24.7 ms      2.1x
    solve         52.1 ms    25.0 ms      2.1x
    inverse       81.8 ms    48.4 ms      1.7x
    cholesky      53.8 ms    17.3 ms      3.1x
    qz (n=1024)   91.9 s     21.9 s       4.2x

### LU — blocked AND column-major

Two changes, and the second was the larger.

BLOCKED (LAPACK's dgetrf): factor a narrow panel, one small triangular solve,
then take the whole trailing update as a single GEMM. Unlike QR this costs
NOTHING EXTRA — QR's compact-WY form needs a T matrix, about 40% more
arithmetic, which is why blocking lost there twice; LU's blocked update is the
same arithmetic regrouped.

COLUMN-MAJOR internally. Every operation in an LU panel runs DOWN a column —
the pivot search, the scaling, the rank-1 update — and in row-major storage each
one walks a fresh cache line per element. Profiling the row-major blocked
version put 21% of the factorisation in the column scaling alone and 29% in
packing. Held column-major, all five inner loops are contiguous and only the row
swaps stride, which is O(n^2) against O(n^3) of work. The trailing update is
formed TRANSPOSED, A22^T -= U12^T L21^T, so both GEMM operands pack as plain
copies. Interleaved A/B: column-major wins 1.26-1.51x over row-major blocked.

THE PIVOT SEQUENCE IS UNCHANGED, so det() keeps its sign and every existing
test passes untouched.

Two smaller traps, both worth naming because neither is arithmetic:
  - std::vector::assign VALUE-INITIALISES, and mstore::gemm then zeroes its
    output again on entry. Re-assigning the scratch buffers per block was two
    redundant passes over ~32 MB, and 29% of the factorisation. They are now
    allocated once, and the GEMM output uses mstore::RawBuf, which skips the
    zero-fill entirely.
  - The two transposes at the boundaries were naive and strided. Tiled at 32x32
    they went from 94 ms to ~41 ms of a 152 ms det() at n=2048.

### Cholesky — blocked

Same shape (LAPACK's dpotrf), and it was the worst offender at 4% of peak. The
trailing update is mathematically a SYRK, A22 -= L21 L21^H, which needs only the
lower triangle. It is done here as a full GEMM with only the lower half
subtracted: twice the arithmetic at more than twenty times the rate.

### QZ — 4.2x, and none of it was arithmetic

The QZ was 12-14x slower than schurDecomp for an algorithm that should be about
2x. Two causes, both about MEMORY rather than flops:

  1. THE ROTATION RANGES WERE NOT RESTRICTED. An earlier version applied every
     rotation to the full row and column, reasoning that rotating a pair of
     zeros is harmless. It is harmless and it was expensive: a column rotation
     touched all n rows where only the first p+2 can be nonzero. schurDecomp had
     always restricted its equivalent. Fixing it made the sweeps 3.3-4.8x faster.
  2. Q AND Z WERE STORED THE WRONG WAY ROUND. Every update to them rotates a
     COLUMN pair, which strides a cache line per element; held transposed the
     same update is two contiguous runs. schurDecomp already did this for its Q.
     Applied to both the sweeps and the Hessenberg-triangular reduction, which
     had become 63% of the total once the sweeps were fixed, and which halved.

qz/schur is now 2.4-4.4x, which is about what the structure predicts: QZ carries
four matrices where Schur carries two.

### Still slow, and why

    schur   0.9 GFLOP/s      svd   1.4 GFLOP/s
These are iterative eigenvalue algorithms built on Givens rotations and small
reflectors — inherently level 1 and 2. The real fix is LAPACK's multishift,
blocked bulge-chasing (dlaqr0) and a blocked Hessenberg reduction (dgehrd),
which accumulate many rotations and apply them through GEMM. That is a project
of its own, not a tuning pass.
