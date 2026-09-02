#pragma once

// ==========================================================================
//  The Matrix class
// ==========================================================================
//
// Storage, element access, arithmetic, reductions, scans, masks, norms and
// every factorisation that is a member: LU, QR, SVD, Cholesky, Schur, eig.
// By far the largest file here, because it is one class and a class cannot be
// split across headers.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "io.hpp"

template <typename datatype>
class Decomposition;

template <typename datatype>
class Matrix {
  private:
    // Guard shared by every scalar overload in this class: it stops a template
    // taking `const Scalar&` from swallowing a Matrix argument, so A + B picks
    // the matrix overload and A + 1.0 the scalar one. Declared here, ahead of
    // all of them, because a member TYPE must be declared before any member
    // declaration names it — unlike a member function body, which is compiled
    // after the class is complete.
    template <typename Scalar>
    using mask_scalar_t =
        std::enable_if_t<!std::is_base_of<Matrix, std::decay_t<Scalar>>::value>;

  public:
    // --- Constructors ---

    // Default: creates an empty 0x0 matrix
    Matrix() {
        rowSize = 0;
        colSize = 0;
        grid = nullptr;
    }
    // Creates an i x j matrix, zero-initialised
    Matrix(long i, long j) {
        if (i < 0 || j < 0)
            throw std::invalid_argument("Matrix: dimensions must be non-negative, got " +
                                        std::to_string(i) + "x" + std::to_string(j));
        // Indexing operators use signed int, so both dimensions and their product
        // must fit within INT_MAX to guarantee every element is reachable.
        static constexpr long MAX_IDX = std::numeric_limits<int>::max();
        if (i > MAX_IDX || j > MAX_IDX || i * j > MAX_IDX)
            throw std::invalid_argument("Matrix: dimensions " + std::to_string(i) + "x" +
                                        std::to_string(j) + " exceed the maximum indexable size (" +
                                        std::to_string(MAX_IDX) + ")");
        rowSize = i;
        colSize = j;
        allocZero(rowSize * colSize);
    }
    // Move constructor: takes over M's buffer instead of copying it.
    //
    // Without this the class stopped at the rule of THREE, so every Matrix
    // built from a temporary — which is every arithmetic result — paid a
    // full allocate + zero-fill + element-by-element copy for data that was
    // about to be destroyed anyway. Eigen and Armadillo both rely on this;
    // it is the cheapest structural win available to a value-semantics
    // matrix type.
    //
    // noexcept matters as much as the move itself: std::vector<Matrix> only
    // moves its elements when reallocating if the move cannot throw,
    // otherwise it silently falls back to copying them.
    Matrix(Matrix&& M) noexcept {
        rowSize = M.rowSize;
        colSize = M.colSize;
        if (M.isInline()) {
            // The source's data lives inside the source object, so there is
            // no pointer to steal — copy it across. It is at most
            // SBO_CAPACITY elements, which is cheaper than an allocation.
            const long total = rowSize * colSize;
            grid = sbo;
            for (long i = 0; i < total; i++)
                grid[i] = M.grid[i];
        } else {
            grid = M.grid;
            M.grid = nullptr;  // release() tolerates null
        }
        M.rowSize = 0;
        M.colSize = 0;
    }

    // Copy constructor: deep copies M
    Matrix(const Matrix& M) {
        rowSize = M.rowSize;
        colSize = M.colSize;
        const long total = rowSize * colSize;
        allocRaw(total);  // every element is written below
        for (long index = 0; index < total; index++)
            grid[index] = M.grid[index];
    }

    // --- Assignment operators ---

    // Move assignment: steals M's buffer. This is the one that matters most
    // in practice, because `C = A + B;` on an already-existing C used to deep
    // copy the temporary that operator+ had just built.
    Matrix& operator=(Matrix&& M) noexcept {
        if (this == &M)
            return *this;
        release();
        rowSize = M.rowSize;
        colSize = M.colSize;
        if (M.isInline()) {  // nothing to steal — see the move ctor
            const long total = rowSize * colSize;
            grid = sbo;
            for (long i = 0; i < total; i++)
                grid[i] = M.grid[i];
        } else {
            grid = M.grid;
            M.grid = nullptr;
        }
        M.rowSize = 0;
        M.colSize = 0;
        return *this;
    }

    // Assigns from another Matrix (deep copy)
    Matrix& operator=(const Matrix& M) {
        if (this == &M)
            return *this;  // self-assignment guard

        release();  // empty the Matrix
        rowSize = M.rowSize;
        colSize = M.colSize;
        const long total = rowSize * colSize;
        allocRaw(total);
        for (long index = 0; index < total; index++)
            grid[index] = M.grid[index];
        return *this;
    }
    // Assigns from a 2D initializer list, e.g. A = {{1,2},{3,4}}
    Matrix& operator=(const std::initializer_list<std::initializer_list<datatype>>& M) {
        try {
            if (M.size() == 0)
                throw std::invalid_argument("Cannot assign empty initializer list to Matrix");

            release();

            rowSize = M.size();
            auto itr = M.begin();
            colSize = itr->size();

            allocZero(rowSize * colSize);  // deep copy

            int i = 0, j = 0, index = 0;
            for (auto row : M) {
                for (auto element : row) {
                    index = i * colSize + (j++ % colSize);
                    grid[index] = element;
                }
                i++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Matrix assignment error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // --- Arithmetic operators ---

    // Element-wise addition. Requires identical dimensions. Returns a new Matrix.
    // const so that it works on const operands — matrix functions such as
    // exp(const Matrix&) accumulate their series with it.
    Matrix operator+(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize)
                throw std::invalid_argument(
                    "Column size mismatch in operator+: " + std::to_string(colSize) +
                    " != " + std::to_string(M.colSize));
            if (this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Row size mismatch in operator+: " + std::to_string(rowSize) +
                    " != " + std::to_string(M.rowSize));
        } catch (const std::exception& e) {
            std::cerr << "Matrix addition error: " << e.what() << std::endl;
            throw;
        }
        // One pass into an uninitialised buffer. Copy-then-accumulate would
        // zero the result, copy into it, then read it back to add — three
        // passes over the data where one will do.
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] + b[i]; });
        return ans;
    }

    // Scalar multiplication: multiplies every element by num. Returns a new
    // Matrix.
    template <typename scalar>
    Matrix operator*(const scalar& num) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] * num; });
        return ans;
    }
    template <typename scalar>
    Matrix operator*(const scalar& num) && {
        *this *= num;
        return std::move(*this);
    }

    // In-place scalar multiplication. Written directly rather than as
    // `*this = *this * num`, which allocated a whole second matrix and
    // assigned it back over the first.
    template <typename scalar>
    Matrix& operator*=(const scalar& num) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] *= num; });
        return *this;
    }

    // Matrix multiplication (dot product). Requires this->cols == M.rows.
    // Uses Strassen-Winograd for square matrices >= STRASSEN_THRESHOLD,
    // falling back to naive O(n³) for smaller or rectangular matrices.
    // Ref-qualified purely for consistency with the scalar operator* below.
    // Once ANY overload of a name is ref-qualified, an unqualified sibling
    // stops competing on equal terms: for an rvalue left operand the
    // &&-qualified one wins outright, which silently routed `Q.T() * B` into
    // the scalar path. Matrix multiply cannot reuse either operand's buffer
    // (the result has different dimensions), so this simply forwards.
    Matrix operator*(const Matrix& M) && { return static_cast<const Matrix&>(*this) * M; }
    Matrix operator*(const Matrix& M) const& {
        try {
            if (this->colSize != M.rowSize)
                throw std::invalid_argument(
                    "Inner dimensions must match for operator*: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") * (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");

                // ── Strassen-Winograd: OFF by default ──────────────────────
                //
                // Measured against the same blocked naive multiply this file
                // already contains, Strassen is SLOWER at every size tested:
                //
                //     n      naive / strassen
                //     100        0.23x
                //     128        0.43x
                //     256        0.42x
                //     512        0.25x
                //    1024        0.11x
                //
                // The asymptotics are real, but two implementation facts swamp
                // them here. First, naiveMul is OpenMP-parallel across every
                // core, while the Strassen recursion is serial — only its
                // base-case calls enter a parallel region, each on a block small
                // enough that the thread overhead dominates. Second, every
                // recursion level heap-allocates about twenty temporaries
                // (eight sub-blocks, eight sums, seven products), and that
                // allocation and copy traffic costs more than the one saved
                // multiply out of eight returns.
                //
                // Turning it back on is worth doing once the recursion itself is
                // parallelised (an OpenMP task per independent product) and the
                // temporaries come from one preallocated arena instead of the
                // heap. Until then it is a pessimisation, so it is compiled out
                // rather than silently costing 2-9x on every product.
                //
                // Define MATRIXCPP_ENABLE_STRASSEN to opt back in.
#ifdef MATRIXCPP_ENABLE_STRASSEN
            if (rowSize == colSize && M.rowSize == M.colSize && rowSize == M.rowSize &&
                rowSize >= STRASSEN_THRESHOLD) {
                long sz = nextPow2(rowSize);
                if (sz == rowSize)
                    return strassenWinograd(*this, M);
                // Only use Strassen if padding stays within 41% overhead (i.e. the
                // padded size is at most sqrt(2)*n ≈ 1.41n, keeping work < 2×).
                // Otherwise fall through to naive — the plateau jump is not worth it.
                if (sz <= rowSize * 3 / 2) {
                    Matrix Ap(sz, sz), Bp(sz, sz);
                    long total = rowSize * colSize;
                    for (long k = 0; k < total; k++) {
                        Ap.grid[k / colSize * sz + k % colSize] = grid[k];
                        Bp.grid[k / M.colSize * sz + k % M.colSize] = M.grid[k];
                    }
                    Matrix Cp = strassenWinograd(Ap, Bp);
                    Matrix ans(rowSize, rowSize);
                    long ansTotal = rowSize * rowSize;
                    for (long k = 0; k < ansTotal; k++)
                        ans.grid[k] = Cp.grid[k / rowSize * sz + k % rowSize];
                    return ans;
                }
            }
#endif
            return naiveMul(*this, M);

        } catch (const std::exception& e) {
            std::cerr << "Matrix multiplication error: " << e.what() << std::endl;
            throw;
        }
    }

    // In-place matrix multiplication
    Matrix& operator*=(const Matrix& M) {
        (*this) = (*this) * M;
        return (*this);
    }

    // Hadamard (element-wise) product. Requires identical dimensions. Returns a
    // new Matrix.
    Matrix operator%(const Matrix& M) const& {
        // The try covers only the check that can throw. Building `ans` and
        // returning it from inside a try would suppress NRVO and cost a full
        // extra allocate-zero-copy of the result — 9x on a 2000x2000 product.
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in Hadamard product: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Hadamard product error: " << e.what() << std::endl;
            throw;
        }
        // uninit_t, and restrict-qualified locals — the same two points as
        // operator+ and operator-, both of which this loop was missing.
        // Zero-filling a result whose every element is written on the next
        // line costs a whole extra pass over the output: the product moves
        // 3n² elements, so the wasted n² was a third of the operation. That
        // was the entire gap against NumPy here.
        Matrix<datatype> ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] * b[i]; });
        return ans;
    }
    Matrix operator%(const Matrix& M) && {
        *this %= M;
        return std::move(*this);
    }

    // Integer modulo: applies modulo to every element. Returns a new Matrix.
    // e.g. A % 3 gives a matrix where each element is a_ij % 3.
    // Note: n % A has no defined meaning and is not supported.
    Matrix operator%(const int& modulo) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        for (long i = 0; i < total; i++)
            r[i] = a[i] % modulo;
        return ans;
    }
    // Qualified to match operator%(const Matrix&); reuses the temporary.
    Matrix operator%(const int& modulo) && {
        *this %= modulo;
        return std::move(*this);
    }

    // In-place integer modulo
    Matrix& operator%=(const int& modulo) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        for (long i = 0; i < total; i++)
            r[i] %= modulo;
        return *this;
    }

    // In-place Hadamard product (Element-wise Matrix multiplication)
    Matrix& operator%=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in Hadamard product: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Hadamard product error: " << e.what() << std::endl;
            throw;
        }
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] *= b[i]; });
        return *this;
    }

    // ── Matrix right division, MATLAB's mrdivide. A / B is the X solving
    //    X * B = A, i.e. A * inv(B) — WITHOUT ever forming inv(B).
    //
    // This used to be element-wise division. It was changed to match MATLAB,
    // because `A / B` meaning two entirely different things in two systems
    // that otherwise line up is the kind of difference that produces a wrong
    // answer rather than an error. Element-wise division is now div() — see
    // below, and the operator table in the header comment.
    //
    // Implemented as a solve, not as A * B.inverse(): fewer flops, and better
    // conditioned. Right division is left division on the transposes,
    //     X * B = A   <=>   Bᵀ * Xᵀ = Aᵀ   <=>   X = (Bᵀ \ Aᵀ)ᵀ
    // so it inherits everything solve() already does — LU with partial
    // pivoting when B is square, least squares via column-pivoted QR when it
    // is not, which is also what MATLAB's / does.
    //
    // Returns Matrix<double> because a solve does; see solve().
    Matrix<double> operator/(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize)
                throw std::invalid_argument(
                    "Dimension mismatch in matrix right division X*B=A: A is (" +
                    std::to_string(rowSize) + "x" + std::to_string(colSize) + ") and B is (" +
                    std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) +
                    ") — they must agree in COLUMNS. "
                    "For element-wise division use A.div(B) or A /dot/ B");
            return M.T().solve(this->T()).T();
        } catch (const std::exception& e) {
            std::cerr << "Matrix right division error: " << e.what() << '\n';
            throw;
        }
    }

    // No buffer to reuse — the result of a solve is a fresh matrix of a
    // different shape in general — so the rvalue form just forwards. It still
    // has to exist: once one overload of an operator name is ref-qualified,
    // every sibling must be, or overload resolution silently picks the wrong
    // one for rvalue operands. See the note in the header comment.
    Matrix<double> operator/(const Matrix& M) && { return static_cast<const Matrix&>(*this) / M; }

    // A /= B is A = A / B, so it inherits the right-division meaning. Only
    // instantiable for Matrix<double>, since a solve produces doubles and
    // there is no narrowing conversion back.
    Matrix& operator/=(const Matrix& M) {
        static_assert(std::is_same<datatype, double>::value,
                      "A /= B is matrix RIGHT DIVISION and produces double results, so it\n"
                      "only applies to Matrix<double>. Write B = A / C for other element\n"
                      "types, or A = A.div(B) if you meant element-wise division.");
        *this = (*this) / M;
        return *this;
    }

    // ── Element-wise ("dot") operations ─────────────────────────────────
    // THE RULE THIS HEADER FOLLOWS, and the reason these are named the way
    // they are: the MEMBER DOT IS THE ELEMENT-WISE MARKER. A.sin() is
    // element-wise, sin(A) is the matrix function; A.pow(n) is element-wise,
    // pow(A, n) is the matrix power. That is the whole of MATLAB's leading
    // dot, moved to where C++ can actually put it.
    //
    //     MATLAB      here
    //     A .* B      A.mul(B)   or  A % B   or  A *dot* B
    //     A ./ B      A.div(B)                or  A /dot/ B
    //     A .^ n      A.pow(n)
    //     sin(A) elementwise      A.sin()
    //
    // So the names are div and mul, NOT ediv and emul: the dot has already
    // said element-wise, and an `e` prefix says it a second time. It also
    // keeps them the same shape as every other element-wise member — sin,
    // cos, exp, abs, pow are all three letters, and so are these.
    //
    // T() and H() are members that are NOT element-wise, which does bend the
    // rule. It cannot cause an ambiguity though: there is no such thing as an
    // element-wise transpose for A.T() to be mistaken for.
    //
    // The named forms are the primitives; % and the dot-sugar are spellings
    // of them. Both are ref-qualified so that a temporary on the left is
    // reused instead of reallocated, exactly like operator+ and operator-.

    // Element-wise division, MATLAB's ./
    Matrix div(const Matrix& M) const& {
        requireSameShape(M, "div");
        // uninit_t and restrict, for the reasons given on operator%.
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] / b[i]; });
        return ans;
    }

    // Rvalue form: divides in place and hands the same buffer back.
    Matrix div(const Matrix& M) && {
        requireSameShape(M, "div");
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] /= b[i]; });
        return std::move(*this);
    }

    // Element-wise multiplication, MATLAB's .* — a named spelling of operator%.
    Matrix mul(const Matrix& M) const& { return *this % M; }
    Matrix mul(const Matrix& M) && { return std::move(*this) % M; }
    // Scalar division: divides every element by n. Preserves datatype.
    // Note: n / A has no defined meaning and is not supported.
    template <typename scalar>
    Matrix operator/(const scalar& n) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] / n; });
        return ans;
    }

    template <typename scalar>
    Matrix operator/(const scalar& n) && {
        *this /= n;
        return std::move(*this);
    }

    // In-place scalar division
    template <typename scalar>
    Matrix& operator/=(const scalar& n) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] /= n; });
        return *this;
    }

    // ── Rvalue-qualified arithmetic ─────────────────────────────────
    // In a chain like A + B + C, the left operand of the second + is the
    // temporary that the first + just produced. These overloads recognise
    // that and write into that temporary's buffer instead of allocating
    // another one, so a chain of k operations allocates once rather than k
    // times. Expression templates (Eigen, uBLAS, MTL4) solve the same
    // problem more completely — they fuse the whole chain into a single
    // pass — but they change what `A + B` returns, which breaks template
    // argument deduction in ordinary user code like f(A + B). These keep
    // every type exactly as it was.
    //
    // Note this only catches temporaries on the LEFT. A + (B + C) still
    // allocates for the inner sum, because there the temporary is the
    // argument, not the object. Left-to-right is how chains normally parse.
    Matrix operator+(const Matrix& M) && {
        *this += M;
        return std::move(*this);
    }

    // ── Matrix + scalar, MATLAB's broadcast ────────────────────────────────
    // A + 3 adds 3 to EVERY element, and A - 3 subtracts it. MATLAB does this,
    // NumPy does this, and this header did not — which left `A + 1.0` as a
    // compile error for every shape.
    //
    // It also closes a hole the 1x1-to-scalar conversion would otherwise open.
    // Without these, `A + 1.0` on a 5x5 would fall through to that conversion,
    // compile, and throw at runtime. With them it means what MATLAB means and
    // never reaches the conversion at all. The two features had to land together
    // for either to be safe.
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix operator+(const Scalar& v) const& {
        return offsetBy(datatype(v));
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix operator+(const Scalar& v) && {
        *this += v;
        return std::move(*this);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix operator-(const Scalar& v) const& {
        return offsetBy(datatype(0) - datatype(v));
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix operator-(const Scalar& v) && {
        *this -= v;
        return std::move(*this);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix& operator+=(const Scalar& v) {
        const datatype d = datatype(v);
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(rowSize * colSize, [=](long i) { r[i] += d; });
        return *this;
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix& operator-=(const Scalar& v) {
        const datatype d = datatype(v);
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(rowSize * colSize, [=](long i) { r[i] -= d; });
        return *this;
    }

    // In-place element-wise addition. Requires identical dimensions.
    Matrix& operator+=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator+=: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
            for (long index = 0; index < colSize * rowSize; index++)
                grid[index] += M.grid[index];
        } catch (const std::exception& e) {
            std::cerr << "Matrix addition error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // Element-wise subtraction. Requires identical dimensions. Returns a new
    // Matrix. Subtracts directly rather than going via *this + M*-1: one pass
    // instead of two, no intermediate matrix, and it stays correct for unsigned
    // datatypes.
    Matrix operator-(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator-: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Matrix subtraction error: " << e.what() << std::endl;
            throw;
        }
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] - b[i]; });
        return ans;
    }

    Matrix operator-(const Matrix& M) && {
        *this -= M;
        return std::move(*this);
    }

    // In-place element-wise subtraction. Requires identical dimensions.
    Matrix& operator-=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator-=: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
            for (long index = 0; index < colSize * rowSize; index++)
                grid[index] -= M.grid[index];
        } catch (const std::exception& e) {
            std::cerr << "Matrix subtraction error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // Agrumented Matrix operator, allows for similar math notation.
    // Note, to preserve predence use with (), EX (A|B)
    Matrix operator|(const Matrix& M) const { return (*this).concat(M, 1); }

    // Unary negation: returns a new Matrix with every element negated.
    Matrix operator-() const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        for (long i = 0; i < total; i++)
            ans.grid[i] = -grid[i];
        return ans;
    }

    Matrix operator-() && {
        const long total = rowSize * colSize;
        for (long i = 0; i < total; i++)
            grid[i] = -grid[i];
        return std::move(*this);
    }

    // --- Comparison operators ---

    // Exact equality: same dimensions and every element compares equal.
    // For floating-point types prefer allclose() — exact == is rarely what you
    // want, because two mathematically equal results computed different ways
    // almost never agree bit-for-bit.
    bool operator==(const Matrix& M) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            return false;
        for (long index = 0; index < rowSize * colSize; index++)
            if (!(grid[index] == M.grid[index]))
                return false;
        return true;
    }

    bool operator!=(const Matrix& M) const { return !(*this == M); }

    // Approximate equality, mirroring numpy.allclose:
    //   |a_ij - b_ij| <= atol + rtol * |b_ij|  for every element.
    // Mismatched dimensions compare false rather than throwing, so it is safe
    // to use directly as a test assertion. This is what the test suite uses.
    bool allclose(const Matrix& M, double rtol = 1e-5, double atol = 1e-8) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            return false;
        for (long index = 0; index < rowSize * colSize; index++) {
            double diff = magnitude(grid[index] - M.grid[index]);
            if (!(diff <= atol + rtol * magnitude(M.grid[index])))
                return false;
        }
        return true;
    }

    // Write-through handle for logical indexing: A(mask) = x.
    //
    // The mask is stored BY VALUE, not by reference: `A(A > 0) = 0.0` builds
    // a temporary mask, and although that temporary does survive to the end
    // of the full expression, a proxy held any longer would dangle. One byte
    // per element against the datatype-sized assignment it is about to drive
    // makes the safety close to free.
    //
    // It is flattened into a vector<char> rather than held as a Matrix<bool>
    // because MaskProxy is a member of Matrix<datatype>, so a Matrix<bool>
    // member would make Matrix<bool> contain a MaskProxy containing a
    // Matrix<bool> — an infinitely recursive type. char, not bool, to keep
    // the plain-array indexing that vector<bool>'s bit-packing would take
    // away.
    class MaskProxy {
        Matrix& mat;
        std::vector<char> mask;

      public:
        MaskProxy(Matrix& m, const Matrix<bool>& k) : mat(m) {
            m.requireMaskShape(k);
            const long total = k.rows() * k.cols();
            mask.resize((std::size_t)total);
            for (long i = 0; i < total; i++)
                mask[(std::size_t)i] = k[int(i)] ? 1 : 0;
        }
        long selected() const {
            long n = 0;
            for (char c : mask)
                if (c)
                    n++;
            return n;
        }
        // A(mask) = scalar — sets every selected element.
        template <
            typename Scalar,
            typename = std::enable_if_t<!std::is_base_of<Matrix, std::decay_t<Scalar>>::value>>
        MaskProxy& operator=(const Scalar& v) {
            const long total = mat.rows() * mat.cols();
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    mat[int(i)] = datatype(v);
            return *this;
        }
        // A(mask) = column vector — one value per selected element, in the
        // same row-major order the read form produces.
        MaskProxy& operator=(const Matrix<datatype>& src) {
            const long n = selected();
            if (src.rows() * src.cols() != n)
                throw std::invalid_argument("A(mask) = src: the mask selects " + std::to_string(n) +
                                            " elements but src has " +
                                            std::to_string(src.rows() * src.cols()));
            const long total = mat.rows() * mat.cols();
            long k = 0;
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    mat[int(i)] = src[int(k++)];
            return *this;
        }
        // Read: A(mask) used in an expression.
        operator Matrix<datatype>() const {
            const long total = mat.rows() * mat.cols();
            Matrix<datatype> out(selected(), 1, uninit_t{});
            long k = 0;
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    out[int(k++)] = mat[int(i)];
            return out;
        }
        friend std::ostream& operator<<(std::ostream& os, const MaskProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

  private:
    // Declared here rather than with the other private helpers below: a
    // member TYPE must be declared before any member declaration names it,
    // and the comparison templates just below use mask_scalar_t in their
    // template parameter lists. Member function BODIES are compiled after
    // the class is complete, which is why CmpOp can be used in them freely.
    enum class CmpOp { LT, GT, LE, GE, EQ, NE };
    enum class LogOp { AND, OR, XOR };

    // (mask_scalar_t is declared at the top of the class — the scalar arithmetic
    // operators need it too, and a member type must precede every use of it.)

  public:
    // ═══════════════════════════════════════════════════════════════════
    //  Logical masks — MATLAB's logical arrays
    // ═══════════════════════════════════════════════════════════════════
    //
    // A mask is a Matrix<bool>, so it is an ordinary matrix and gets shape,
    // printing, T(), slicing and everything else for free.
    //
    // WHICH COMPARISONS ARE OPERATORS, AND WHY:
    //   <  >  <=  >=   are operators and are ELEMENT-WISE.
    //   ==  !=         are operators and are WHOLE-MATRIX, returning bool.
    //   .eq() .ne()    are the element-wise forms of those two.
    //
    // That looks inconsistent for a moment and is not. The rule this header
    // follows is that the member dot marks element-wise where BOTH meanings
    // exist. For < > <= >= only one meaning exists — there is no ordering of
    // matrices for `A < B` to be mistaken for — so the operator is free to
    // take it, exactly as A.T() is free to be a non-element-wise member
    // because no element-wise transpose exists to collide with.
    //
    // For == the other meaning very much does exist, and `if (A == B)` is the
    // idiom every C++ programmer reaches for, so the operator keeps whole
    // matrix equality and the element-wise version is the member, .eq().
    // MATLAB's == is element-wise; this is a deliberate divergence, and it is
    // the one place in the comparison family where the two differ.
    //
    //     A > 0        A <= B       A.eq(B)      A.ne(0)
    //     m1 && m2     m1 || m2     m1 ^ m2      !m1
    //     A.any()      A.all()      A.nnz()      A.find()
    //     A(A > 0)                  read the selected elements
    //     A(A < 0) = 0.0            write through the mask

    // Element-wise comparisons against another matrix.
    Matrix<bool> lt(const Matrix& M) const { return compare(M, "lt", CmpOp::LT); }
    Matrix<bool> gt(const Matrix& M) const { return compare(M, "gt", CmpOp::GT); }
    Matrix<bool> le(const Matrix& M) const { return compare(M, "le", CmpOp::LE); }
    Matrix<bool> ge(const Matrix& M) const { return compare(M, "ge", CmpOp::GE); }
    Matrix<bool> eq(const Matrix& M) const { return compare(M, "eq", CmpOp::EQ); }
    Matrix<bool> ne(const Matrix& M) const { return compare(M, "ne", CmpOp::NE); }

    // Element-wise comparisons against a scalar.
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> lt(const Scalar& v) const {
        return compareScalar(v, CmpOp::LT);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> gt(const Scalar& v) const {
        return compareScalar(v, CmpOp::GT);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> le(const Scalar& v) const {
        return compareScalar(v, CmpOp::LE);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> ge(const Scalar& v) const {
        return compareScalar(v, CmpOp::GE);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> eq(const Scalar& v) const {
        return compareScalar(v, CmpOp::EQ);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> ne(const Scalar& v) const {
        return compareScalar(v, CmpOp::NE);
    }

    // The ordering operators. No matrix-level meaning exists, so these are
    // element-wise with nothing to be confused with.
    Matrix<bool> operator<(const Matrix& M) const { return lt(M); }
    Matrix<bool> operator>(const Matrix& M) const { return gt(M); }
    Matrix<bool> operator<=(const Matrix& M) const { return le(M); }
    Matrix<bool> operator>=(const Matrix& M) const { return ge(M); }

    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator<(const Scalar& v) const {
        return lt(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator>(const Scalar& v) const {
        return gt(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator<=(const Scalar& v) const {
        return le(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator>=(const Scalar& v) const {
        return ge(v);
    }

    // Logical combinators, spelled in C: &&, || and !. Non-zero counts as
    // true, as it does in MATLAB, so these work on a numeric matrix and not
    // only on a mask.
    //
    // WHY C's && AND || RATHER THAN NumPy's AND MATLAB's & AND |:
    // because | is not available. `A | B` in this header is the
    // AUGMENTED-MATRIX operator, (A|B), which horizontally concatenates —
    // it predates the mask layer and is worth keeping. Taking & for `and`
    // while `or` had to be a named function would have left the pair
    // lopsided, and worse, would have left `(A > 0) | (B > 0)` compiling and
    // quietly returning a mask of twice the width. Going to the full C
    // triple instead gives both halves an operator, makes neither collide,
    // and reads the way logic reads in C.
    //
    // ⚠ THAT TRAP STILL EXISTS FOR ANYONE TYPING FROM NumPy HABIT: `|` is
    // concatenation here, not or. Use ||.
    //
    // OVERLOADING && AND || COSTS SHORT-CIRCUIT EVALUATION — for a built-in
    // ||, the right operand is skipped when the left already decides the
    // answer; for an overloaded one both are always evaluated, and the
    // sequencing guarantee goes with it. That loss is real and here it is
    // free: an element-wise or has to look at every element of both operands
    // regardless, so there was never anything to skip. Note also that the
    // result is a Matrix<bool>, which has no conversion to bool, so
    // `if (m1 || m2)` does not compile — you have to say which you meant,
    // .any() or .all(). Scalar conditions like `if (A.any() || B.any())` are
    // plain bools and short-circuit normally.
    Matrix<bool> operator&&(const Matrix& M) const { return land(M); }
    Matrix<bool> operator||(const Matrix& M) const { return lor(M); }

    // ^ IS exclusive-or in C, so this is the language's own spelling, not a
    // borrowed one. Its famously low precedence is an argument against ever
    // using ^ for a POWER — `A * B ^ 2` would silently group as `(A*B) ^ 2`
    // — which is exactly why the matrix power here is the function pow(A, n)
    // and never an operator. For exclusive-or the precedence is harmless and
    // in fact convenient: the relational operators bind TIGHTER than ^, so
    //     A > 0 ^ A > 3     groups as     (A > 0) ^ (A > 3)
    // which is what anyone writing it would mean. Checked in validate.cpp.
    // GCC nevertheless emits -Wparentheses for the unparenthesised form,
    // because a bare comparison beside ^ is a classic bug in bitwise code —
    // so parenthesise in real code even though the grouping is already
    // right. The suggestion is about legibility, not correctness.
    Matrix<bool> operator^(const Matrix& M) const { return lxor(M); }

    // Named forms, uniform across the whole family, and deliberately kept
    // alongside the operators rather than instead of them: && || ^ ! read
    // naturally to anyone arriving from C or C++, and land/lor/lxor/lnot
    // read naturally to anyone arriving from Python, MATLAB or Fortran.
    // Both spellings are the same call.
    Matrix<bool> land(const Matrix& M) const { return logical(M, "land", LogOp::AND); }
    Matrix<bool> lor(const Matrix& M) const { return logical(M, "lor", LogOp::OR); }
    Matrix<bool> lxor(const Matrix& M) const { return logical(M, "lxor", LogOp::XOR); }
    Matrix<bool> lnot() const { return !(*this); }

    // Element-wise negation: true where the element is zero. MATLAB's ~A.
    Matrix<bool> operator!() const {
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype zero = datatype(0);
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = (a[i] == zero); });
        return out;
    }

    // ── Mask reductions ────────────────────────────────────────────────
    // Non-zero counts as true throughout, so any()/all()/nnz() read the same
    // on a mask and on the numeric matrix it came from.

    bool any() const {
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (!(grid[i] == zero))
                return true;
        return false;
    }
    bool all() const {
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (grid[i] == zero)
                return false;
        return true;
    }
    // Count of non-zero elements. This — not sum() — is how you count a mask:
    // sum() returns datatype, and for Matrix<bool> that would saturate at
    // true rather than counting.
    long nnz() const {
        long n = 0;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (!(grid[i] == zero))
                n++;
        return n;
    }

    // Axis forms, mirroring sum(bool): axis=0 gives a (1 x cols) row of
    // per-column results, axis=1 a (rows x 1) column of per-row results.
    Matrix<bool> any(const bool& axis) const { return reduceLogical(axis, true); }
    Matrix<bool> all(const bool& axis) const { return reduceLogical(axis, false); }

    // Positions of the non-zero elements, as (row, column) pairs in
    // row-major order. Pairs rather than MATLAB's linear indices on purpose:
    // MATLAB's are COLUMN-major, and silently handing back numbers that mean
    // something different in the two systems is exactly the sort of thing
    // this header tries not to do.
    std::vector<std::pair<long, long>> find() const {
        std::vector<std::pair<long, long>> out;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (!(grid[i * colSize + j] == zero))
                    out.emplace_back(i, j);
        return out;
    }

    // ── Logical indexing ───────────────────────────────────────────────

    // Read: A(mask) gives a column vector of the selected elements, in
    // row-major order. (MATLAB's is column-major — same caveat as find().)
    Matrix operator()(const Matrix<bool>& mask) const {
        requireMaskShape(mask);
        const long n = mask.nnz();
        Matrix out(n, 1, uninit_t{});
        long k = 0;
        for (long i = 0; i < rowSize * colSize; i++)
            if (mask.grid[i])
                out.grid[k++] = grid[i];
        return out;
    }

    // Write: A(mask) = scalar, or A(mask) = column vector of nnz elements.
    MaskProxy operator()(const Matrix<bool>& mask) { return MaskProxy(*this, mask); }

    // --- Proxy classes for slice assignment ---
    // Returned by non-const slice operators. Holds a reference back to the
    // parent Matrix so that A(i, all) = B writes through to A.
    // Implicit Matrix<datatype> conversion lets them be used in read contexts
    // too.

    class RowProxy {
        Matrix& mat;
        int row;

      public:
        RowProxy(Matrix& m, int r) : mat(m), row(r) {}
        // Write: A(i, all) = src  — src must be a (1 x cols) row vector
        RowProxy& operator=(const Matrix<datatype>& src) {
            if (src.rows() != 1 || src.cols() != mat.cols())
                throw std::invalid_argument(
                    "RowProxy: source must be (1 x " + std::to_string(mat.cols()) + "), got (" +
                    std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
            for (long j = 0; j < mat.cols(); j++)
                mat(row, j) = src(0, j);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(1, mat.cols());
            for (long j = 0; j < mat.cols(); j++)
                ans[j] = mat(row, j);
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const RowProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    class ColProxy {
        Matrix& mat;
        int col;

      public:
        ColProxy(Matrix& m, int c) : mat(m), col(c) {}
        // Write: A(all, j) = src  — src must be a (rows x 1) column vector
        ColProxy& operator=(const Matrix<datatype>& src) {
            if (src.cols() != 1 || src.rows() != mat.rows())
                throw std::invalid_argument(
                    "ColProxy: source must be (" + std::to_string(mat.rows()) + " x 1), got (" +
                    std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
            for (long i = 0; i < mat.rows(); i++)
                mat(i, col) = src(i, 0);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(mat.rows(), 1);
            for (long i = 0; i < mat.rows(); i++)
                ans[i] = mat(i, col);
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const ColProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    class SubProxy {
        Matrix& mat;
        int r1, c1, rStep, cStep, numRows, numCols;

      public:
        SubProxy(Matrix& m, int r1, int c1, int rStep, int cStep, int numRows, int numCols)
            : mat(m),
              r1(r1),
              c1(c1),
              rStep(rStep),
              cStep(cStep),
              numRows(numRows),
              numCols(numCols) {}
        // Write: A({r1,r2},{c1,c2}) = src  — src must match the slice dimensions
        SubProxy& operator=(const Matrix<datatype>& src) {
            if (src.rows() != numRows || src.cols() != numCols)
                throw std::invalid_argument("SubProxy: source is (" + std::to_string(src.rows()) +
                                            "x" + std::to_string(src.cols()) + ") but slice is (" +
                                            std::to_string(numRows) + "x" +
                                            std::to_string(numCols) + ")");
            for (int i = 0; i < numRows; i++)
                for (int j = 0; j < numCols; j++)
                    mat(r1 + i * rStep, c1 + j * cStep) = src(i, j);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(numRows, numCols);
            for (int idx = 0; idx < numRows * numCols; idx++) {
                int i = idx / numCols, j = idx % numCols;
                ans[idx] = mat(r1 + i * rStep, c1 + j * cStep);
            }
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const SubProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    // --- Indexing operators ---
    // Note: negative indices wrap backwards (e.g. -1 gives last element).
    // Matrix is indexed as A(i, j) where i = row, j = column (0-based).

    // Returns a reference to element (i, j) — supports A(i,j) = x
    datatype& operator()(const int& i, const int& j) {
        return grid[(i % rowSize) * colSize + (j % colSize)];
    }
    // Const element access
    const datatype& operator()(const int& i, const int& j) const {
        return grid[(i % rowSize) * colSize + (j % colSize)];
    }
    // ── A 1x1 matrix IS a scalar ───────────────────────────────────────────
    //
    //     Matrix<double> q = v.T() * A * v;    // a quadratic form: 1x1
    //     double energy = q;                   // and now just a number
    //
    // Any operation can land on a 1x1 — a quadratic form, a full contraction, a
    // logical index that selects one element, a reduction of a vector — and
    // having to write .at(0,0) on the result of all of them is friction with no
    // purpose. This makes the conversion implicit, so a 1x1 reads as either a
    // matrix or the number it holds, whichever the surrounding code wants.
    //
    // THROWS for any other shape. The size is a runtime property, so this cannot
    // be a compile-time check; NumPy makes the same trade with float(arr), which
    // raises unless the array has exactly one element.
    //
    // ── WHY THIS IS A TEMPLATE AND NOT JUST operator datatype() ───────────
    // Because a plain conversion would give Matrix<bool> an implicit
    // operator bool, and that is the classic C++ trap. It was measured, not
    // guessed: adding the unconstrained version compiled all 555 checks without
    // a single error and SILENTLY BROKE the mask layer's safety property —
    // `if (m1 || m2)` started compiling, turning a deliberate compile-time error
    // into a runtime throw. Strictly worse.
    //
    // Deducing U from the target type and requiring U == datatype fixes both
    // halves at once:
    //   * Matrix<bool> gets no conversion at all, so `if (mask)` stays a
    //     compile error and .any()/.all() stays the only way to ask;
    //   * Matrix<double> converts to double but NOT to bool, so `if (A)` is a
    //     compile error there too — which it should be, since "is this matrix
    //     true" has no meaning.
    operator datatype() const {
        static_assert(!std::is_same<datatype, bool>::value,
            "Matrix<bool> is a MASK, and does not convert to bool. `if (mask)` has no "
            "single answer — ask .any() or .all(); count with .nnz().");
        if (rowSize * colSize != 1)
            throw std::logic_error(
                "a Matrix converts to a scalar only when it holds exactly one element, "
                "but this one is " + std::to_string(rowSize) + "x" + std::to_string(colSize) +
                " (" + std::to_string(rowSize * colSize) + " elements). Index it, or "
                "reduce it first — sum(), det(), dot() and the other reductions already "
                "return scalars.");
        return grid[0];
    }

    // Flat index access into the underlying row-major array — supports A[i] = x
    datatype& operator[](const int& i) { return grid[i % (rowSize * colSize)]; }
    // Const flat index access, so A[i] reads from a const Matrix too.
    const datatype& operator[](const int& i) const { return grid[i % (rowSize * colSize)]; }

    // Row extraction — non-const returns RowProxy: supports A(i, all) = B
    RowProxy operator()(const int& i, all_t) {
        int r = ((i % (int)rowSize) + (int)rowSize) % (int)rowSize;
        return RowProxy(*this, r);
    }
    // Row extraction — const returns Matrix by value for reading
    Matrix operator()(const int& i, all_t) const {
        Matrix<datatype> ans(1, this->colSize);
        for (long j = 0; j < this->colSize; j++)
            ans[j] = this->grid[(i % rowSize) * colSize + (j % colSize)];
        return ans;
    }

    // Column extraction — non-const returns ColProxy: supports A(all, j) = B
    ColProxy operator()(all_t, const int& i) {
        int c = ((i % (int)colSize) + (int)colSize) % (int)colSize;
        return ColProxy(*this, c);
    }
    // Column extraction — const returns Matrix by value for reading
    Matrix operator()(all_t, const int& i) const {
        Matrix<datatype> ans(rowSize, 1);
        for (long j = 0; j < rowSize; j++)
            ans[j] = grid[(j % rowSize) * colSize + (i % colSize)];
        return ans;
    }

    // Submatrix — non-const returns SubProxy: supports A({r1,r2},{c1,c2}) = B
    // Negative indices wrap; reversed ranges (e.g. {9,7}) return elements in
    // reverse order.
    SubProxy operator()(std::pair<int, int> rowRange, std::pair<int, int> colRange) {
        int rN = (int)rowSize, cN = (int)colSize;
        int r1 = ((rowRange.first % rN) + rN) % rN;
        int r2 = ((rowRange.second % rN) + rN) % rN;
        int c1 = ((colRange.first % cN) + cN) % cN;
        int c2 = ((colRange.second % cN) + cN) % cN;
        int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
        return SubProxy(*this, r1, c1, (r2 >= r1) ? 1 : -1, (c2 >= c1) ? 1 : -1, numRows, numCols);
    }
    // Submatrix — const returns Matrix by value for reading
    Matrix operator()(std::pair<int, int> rowRange, std::pair<int, int> colRange) const {
        int rN = (int)rowSize, cN = (int)colSize;
        int r1 = ((rowRange.first % rN) + rN) % rN;
        int r2 = ((rowRange.second % rN) + rN) % rN;
        int c1 = ((colRange.first % cN) + cN) % cN;
        int c2 = ((colRange.second % cN) + cN) % cN;
        int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
        int rStep = (r2 >= r1) ? 1 : -1, cStep = (c2 >= c1) ? 1 : -1;
        Matrix<datatype> ans(numRows, numCols);
        for (int idx = 0; idx < numRows * numCols; idx++) {
            int i = idx / numCols, j = idx % numCols;
            ans[idx] = (*this)(r1 + i * rStep, c1 + j * cStep);
        }
        return ans;
    }
    // --- Inspection ---

    // Returns true if the matrix has no elements (0x0 or any zero dimension)
    inline bool empty() const { return rowSize * colSize == 0; }
    // Returns the number of rows
    long rows() const { return rowSize; }
    // Returns the number of columns
    long cols() const { return colSize; }

    // --- Printing and string conversion ---

    // Returns a vector of formatted row strings, one per row.
    // Used internally by toString() and printSideBySide().
    // precision: decimal places for floating-point types (ignored for integral
    // types).
    std::vector<std::string> toLines(int precision = 6) const {
        // Pre-pass: format every element to find the widest string
        std::vector<std::string> cells(rowSize * colSize);
        size_t colWidth = 0;
        for (long i = 0; i < rowSize * colSize; i++) {
            std::ostringstream oss;
            // is_float_like, not std::is_floating_point: the latter is false for
            // std::complex<double>, which would silently drop the formatting.
            if (is_float_like<datatype>::value)
                oss << std::fixed << std::setprecision(precision);
            oss << grid[i];
            cells[i] = oss.str();
            if (cells[i].size() > colWidth)
                colWidth = cells[i].size();
        }

        std::vector<std::string> lines(rowSize);
        for (long i = 0; i < rowSize; i++) {
            std::ostringstream row;
            row << "[ ";
            for (long j = 0; j < colSize; j++) {
                row << std::setw((int)colWidth) << cells[i * colSize + j];
                if (j < colSize - 1)
                    row << "  ";
            }
            row << " ]";
            lines[i] = row.str();
        }
        return lines;
    }

    // Returns the full matrix as a formatted multi-line string.
    // precision: decimal places (default 6, matching NumPy's default).
    std::string toString(int precision = 6) const {
        auto lines = toLines(precision);
        std::string result;
        for (size_t i = 0; i < lines.size(); i++) {
            result += lines[i];
            if (i + 1 < lines.size())
                result += '\n';
        }
        return result;
    }

    // Explicit cast to std::string using default precision (6dp).
    // Use toString(n) directly when a specific precision is needed.
    explicit operator std::string() const { return toString(); }

    // ── Printing ────────────────────────────────────────────────────────
    //
    // print() goes to std::cout; every other overload takes a stream, which is
    // how the rest of C++ spells this. A file, a std::ostringstream and a
    // socket are all the same thing to it.
    //
    //     A.print();                        aligned, to the terminal
    //     A.print(4);                       ... to 4 decimal places
    //     A.print(matio::Fmt::CSV);         comma separated, to the terminal
    //     A.print(file);                    aligned, to a file
    //     A.print(file, matio::Fmt::CSV);   comma separated, to a file
    //     A.save("data.csv");               format taken from the extension
    //
    // The formats are in matio::Fmt: Pretty, Plain, CSV, TSV, Markdown,
    // MATLAB, NumPy, JSON.
    void print(std::ostream& os, const matio::Opts& o) const {
        matio::table(os, rowSize, colSize, [&](long i, long j) { return (*this)(int(i), int(j)); }, o);
        os << '\n';
    }
    void print(std::ostream& os) const { print(os, matio::Opts{}); }
    void print(const matio::Opts& o) const { print(std::cout, o); }
    void print() const { print(std::cout, matio::Opts{}); }
    // Legacy spelling: a bare int has always meant decimal places.
    void print(int precision) const {
        matio::Opts o;
        o.precision = precision;
        print(std::cout, o);
    }

    // The same thing as a string, for when it is going somewhere other than a
    // stream.
    std::string str(const matio::Opts& o = {}) const {
        std::ostringstream ss;
        matio::table(ss, rowSize, colSize, [&](long i, long j) { return (*this)(int(i), int(j)); }, o);
        return ss.str();
    }

    // Write to a file. With no format given the EXTENSION decides: .csv, .tsv,
    // .md, .json, .m, .py, .txt. Throws if the file cannot be opened or the
    // write fails — a save that quietly did nothing is the worst outcome here.
    void save(const std::string& path, matio::Opts o = {}) const {
        matio::saveWith(path, o, [&](std::ostream& f, const matio::Opts& r) {
            matio::table(f, rowSize, colSize,
                         [&](long i, long j) { return (*this)(int(i), int(j)); }, r);
        });
    }

    // --- Linear algebra ---

    // Returns the transpose of this matrix as a new (cols x rows) Matrix.
    // NOTE: for complex datatypes this is the PLAIN transpose, which is
    // usually not the operation you want — see H() below.
    Matrix T() const {
        Matrix<datatype> ans(colSize, rowSize, uninit_t{});
        // Blocked, because a transpose is inherently cache-hostile: one of the
        // two sides is always striding by a whole row. Walking the matrix in
        // tiles keeps both the source and destination tile resident in L1 for
        // the duration of the tile, instead of evicting a cache line per
        // element. The flat-index version this replaced also paid a division
        // and a modulo on every single element.
        constexpr long BLOCK = 32;
        const datatype* MATRIXCPP_RESTRICT src = grid;
        datatype* MATRIXCPP_RESTRICT dst = ans.grid;
        const long R = rowSize, C = colSize;
        // Each ii-strip of tiles writes a disjoint set of destination columns,
        // so the outer loop is independent and parallelises directly. A
        // transpose is pure memory traffic, and one core cannot saturate the
        // memory system on its own.
        auto strip = [=](long ii) {
            const long iMax = std::min(ii + BLOCK, R);
            for (long jj = 0; jj < C; jj += BLOCK) {
                const long jMax = std::min(jj + BLOCK, C);
                for (long i = ii; i < iMax; i++)
                    for (long j = jj; j < jMax; j++)
                        dst[j * R + i] = src[i * C + j];
            }
        };
#ifdef _OPENMP
        if (R * C >= PARALLEL_MIN_WORK) {
    #pragma omp parallel for schedule(static) num_threads(memoryThreads())
            for (long ii = 0; ii < R; ii += BLOCK)
                strip(ii);
            return ans;
        }
#endif
        for (long ii = 0; ii < R; ii += BLOCK)
            strip(ii);
        return ans;
    }

    // --- Complex support (see the COMPLEX NUMBER SUPPORT block at the top) ---
    // These four are the ones that make Matrix<std::complex<double>> usable.
    // For real datatypes they degrade gracefully: conj() and real() are the
    // identity, imag() is all zeros, and H() is exactly T().

    // Conjugate transpose (Hermitian adjoint), A^H = conj(A)^T.
    // THE important one. For complex matrices this — not T() — is what plays
    // the role the transpose plays in the real theory: it is the adjoint that
    // makes <Ax,y> = <x,A^H y>, it is what "orthogonal" becomes ("unitary",
    // Q^H Q = I), and it is what "symmetric" becomes ("Hermitian", A = A^H).
    // Every .T() inside QR, schurDecomp and the symmetry checks must become
    // .H() once complex is supported. Missing one is a silent wrong answer,
    // never a compile error.
    Matrix H() const {
        Matrix<datatype> ans(colSize, rowSize, uninit_t{});
        constexpr long BLOCK = 32;  // blocked for the same reason as T()
        for (long ii = 0; ii < rowSize; ii += BLOCK) {
            const long iMax = std::min(ii + BLOCK, rowSize);
            for (long jj = 0; jj < colSize; jj += BLOCK) {
                const long jMax = std::min(jj + BLOCK, colSize);
                for (long i = ii; i < iMax; i++)
                    for (long j = jj; j < jMax; j++) {
                        const datatype& v = grid[i * colSize + j];
                        if constexpr (is_complex<datatype>::value)
                            ans.grid[j * rowSize + i] = std::conj(v);
                        else
                            ans.grid[j * rowSize + i] = v;
                    }
            }
        }
        return ans;
    }

    // Element-wise complex conjugate. Identity for real datatypes.
    Matrix conj() const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        if constexpr (is_complex<datatype>::value)
            for (long i = 0; i < total; i++)
                ans.grid[i] = std::conj(grid[i]);
        else
            for (long i = 0; i < total; i++)
                ans.grid[i] = grid[i];
        return ans;
    }

    // Real parts of every element. The real_of trait keeps the element type
    // honest: Matrix<complex<float>> gives back Matrix<float>, and for a real
    // datatype this is the identity, returning the same type it started with.
    Matrix<real_t<datatype>> real() const {
        Matrix<real_t<datatype>> ans(rowSize, colSize);
        for (long i = 0; i < rowSize * colSize; i++) {
            if constexpr (is_complex<datatype>::value)
                ans[int(i)] = grid[i].real();
            else
                ans[int(i)] = grid[i];
        }
        return ans;
    }

    // Imaginary parts of every element. All zeros for real datatypes.
    Matrix<real_t<datatype>> imag() const {
        Matrix<real_t<datatype>> ans(rowSize, colSize);
        if constexpr (is_complex<datatype>::value)
            for (long i = 0; i < rowSize * colSize; i++)
                ans[int(i)] = grid[i].imag();
        return ans;
    }

    // element-wise power
    template <typename scalar>
    Matrix pow(scalar num) const {
        return mapElems([=](const datatype& x) { return std::pow(x, num); });
    }

    // element-wise exp
    Matrix exp() const {
        return mapElems([=](const datatype& x) { return std::exp(x); });
    }

    // element-wise log base 10
    Matrix log10() const {
        return mapElems([=](const datatype& x) { return std::log10(x); });
    }

    // element-wise log base 2, using computer science notation.
    Matrix lg() const {
        return mapElems([=](const datatype& x) { return std::log2(x); });
    }

    // element-wise natural log
    Matrix ln() const {
        return mapElems([=](const datatype& x) { return std::log(x); });
    }

    // element-wise log with arbitrary base
    template <typename scalar>
    Matrix log(scalar base) const {
        const auto invLogBase = 1.0 / std::log2(base);  // loop-invariant
        return mapElems([=](const datatype& x) { return std::log2(x) * invLogBase; });
    }

    // --- Element-wise trigonometry ---
    // Each applies std::<fn> to every element independently, exactly like
    // exp()/ln() above. These are NOT the matrix trig functions — for those
    // see the free functions sin(A)/cos(A)/... near the bottom of this header.
    // All follow the same shape: copy *this, map every element, return.

    // element-wise sine
    Matrix sin() const {
        return mapElems([=](const datatype& x) { return std::sin(x); });
    }

    // element-wise cosine
    Matrix cos() const {
        return mapElems([=](const datatype& x) { return std::cos(x); });
    }

    // element-wise tangent
    Matrix tan() const {
        return mapElems([=](const datatype& x) { return std::tan(x); });
    }

    // element-wise arcsine
    Matrix asin() const {
        return mapElems([=](const datatype& x) { return std::asin(x); });
    }

    // element-wise arccosine
    Matrix acos() const {
        return mapElems([=](const datatype& x) { return std::acos(x); });
    }

    // element-wise arctangent
    Matrix atan() const {
        return mapElems([=](const datatype& x) { return std::atan(x); });
    }

    // element-wise hyperbolic sine
    Matrix sinh() const {
        return mapElems([=](const datatype& x) { return std::sinh(x); });
    }

    // element-wise hyperbolic cosine
    Matrix cosh() const {
        return mapElems([=](const datatype& x) { return std::cosh(x); });
    }

    // element-wise hyperbolic tangent
    Matrix tanh() const {
        return mapElems([=](const datatype& x) { return std::tanh(x); });
    }

    // element-wise square root
    Matrix sqrt() const {
        return mapElems([=](const datatype& x) { return std::sqrt(x); });
    }

    // element-wise absolute value
    //  ═══════════════════════════════════════════════════════════════════
    //   Rounding, sign, phase and the two-argument functions  (tier 3)
    //  ═══════════════════════════════════════════════════════════════════
    //  All members, so all element-wise — the rule the header follows
    //  throughout. Each goes through mapElems, which means each inherits the
    //  threading and the restrict-qualified loop the transcendentals use.

    // -1, 0 or +1 for a real element. For a COMPLEX element this is z/|z|,
    // the unit vector in z's direction, which is what MATLAB's sign gives
    // and what makes sign(z)*abs(z) == z hold in both cases.
    Matrix sign() const {
        return mapElems([](const datatype& x) -> datatype {
            if constexpr (is_complex<datatype>::value) {
                const auto m = std::abs(x);
                return m == 0 ? datatype(0) : datatype(x / m);
            } else {
                return datatype(x > datatype(0) ? 1 : (x < datatype(0) ? -1 : 0));
            }
        });
    }

    // Largest integer <= x.
    Matrix floor() const {
        requireRoundable("floor");
        return mapElems([](const datatype& x) { return std::floor(x); });
    }
    // Smallest integer >= x.
    Matrix ceil() const {
        requireRoundable("ceil");
        return mapElems([](const datatype& x) { return std::ceil(x); });
    }
    // Nearest integer, halves away from zero — C's round, and MATLAB's.
    Matrix round() const {
        requireRoundable("round");
        return mapElems([](const datatype& x) { return std::round(x); });
    }
    // Truncation TOWARDS ZERO. Differs from floor for negatives:
    // floor(-2.5) is -3, fix(-2.5) is -2. MATLAB's fix, C's trunc.
    Matrix fix() const {
        requireRoundable("fix");
        return mapElems([](const datatype& x) { return std::trunc(x); });
    }

    // ── mod and rem are NOT the same function ──────────────────────────
    // They differ whenever the operands have opposite signs, and the
    // difference is which one the result follows:
    //     mod(-1, 3) ==  2     the sign of the DIVISOR
    //     rem(-1, 3) == -1     the sign of the DIVIDEND
    // rem is C's fmod. mod is the one you almost always want for wrapping
    // an index or an angle into a range. Both match MATLAB.
    template <typename Scalar>
    Matrix mod(const Scalar& y) const {
        requireRoundable("mod");
        const datatype d = datatype(y);
        return mapElems([=](const datatype& x) -> datatype {
            if (d == datatype(0))
                return x;  // MATLAB: mod(x,0) is x
            const datatype r = std::fmod(x, d);
            return (r != datatype(0) && ((r < datatype(0)) != (d < datatype(0)))) ? datatype(r + d)
                                                                                  : r;
        });
    }
    template <typename Scalar>
    Matrix rem(const Scalar& y) const {
        requireRoundable("rem");
        const datatype d = datatype(y);
        return mapElems([=](const datatype& x) -> datatype {
            return d == datatype(0) ? x : datatype(std::fmod(x, d));
        });
    }

    // Two-argument arctangent, element-wise: Y.atan2(X) is atan2(y, x),
    // the angle of the point (x, y) with the quadrant resolved — which is
    // exactly what plain atan(y/x) cannot do.
    Matrix atan2(const Matrix& X) const {
        requireRoundable("atan2");
        requireSameShape(X, "atan2");
        return zipElems(X, [](const datatype& y, const datatype& x) { return std::atan2(y, x); });
    }

    // sqrt(a² + b²) without the overflow that squaring would cause.
    Matrix hypot(const Matrix& B) const {
        requireRoundable("hypot");
        requireSameShape(B, "hypot");
        return zipElems(B, [](const datatype& a, const datatype& b) { return std::hypot(a, b); });
    }

    // Phase angle, in radians. THE missing piece for complex work: real(),
    // imag() and conj() were all here but there was no way to get an
    // argument out. Returns the real type, like real() and imag() do.
    // For a real matrix this is 0 where the element is >= 0 and pi where it
    // is negative, which is what MATLAB's angle gives.
    Matrix<real_t<datatype>> angle() const {
        // grid of another instantiation is reachable because Matrix
        // befriends every Matrix — see the friend declaration at the bottom.
        Matrix<real_t<datatype>> out(
            rowSize, colSize, typename Matrix<real_t<datatype>>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        real_t<datatype>* MATRIXCPP_RESTRICT r = out.grid;
        forEachIndex(rowSize * colSize, [=](long i) {
            if constexpr (is_complex<datatype>::value)
                r[i] = std::arg(a[i]);
            else
                r[i] = a[i] < datatype(0) ? real_t<datatype>(mconst::pi) : real_t<datatype>(0);
        });
        return out;
    }
    // arg() is the same function under the name the maths uses.
    Matrix<real_t<datatype>> arg() const { return angle(); }

    // Inverse hyperbolics, completing the set beside sinh/cosh/tanh.
    Matrix asinh() const {
        return mapElems([](const datatype& x) { return std::asinh(x); });
    }
    Matrix acosh() const {
        return mapElems([](const datatype& x) { return std::acosh(x); });
    }
    Matrix atanh() const {
        return mapElems([](const datatype& x) { return std::atanh(x); });
    }

    // exp(x) - 1 and log(1 + x), accurate for small x where the obvious
    // spelling loses every significant digit to cancellation.
    Matrix expm1() const {
        requireRoundable("expm1");
        return mapElems([](const datatype& x) { return std::expm1(x); });
    }
    Matrix log1p() const {
        requireRoundable("log1p");
        return mapElems([](const datatype& x) { return std::log1p(x); });
    }

    Matrix abs() const {
        return mapElems([=](const datatype& x) { return std::abs(x); });
    }

    // Returns whether a matrix is diagonal: all off-diagonal elements are zero.
    // Works for non-square matrices. Single flat loop — no allocations, early
    // exit.
    bool IsDiagonal() const {
        // The flat version this replaced recovered (i,j) from k with
        // k / colSize and k % colSize — two integer divisions for every
        // element of the matrix, which dominated the loop completely: the
        // scan ran at 6.5 GB/s against a machine that streams at ~45.
        // Walking rows explicitly makes the row/column comparison free and
        // splits each row into two runs the compiler can vectorise, since
        // neither contains the diagonal and so neither needs a test on j.
        const datatype* MATRIXCPP_RESTRICT g = grid;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize; i++) {
            const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
            const long d = (i < colSize) ? i : colSize;  // diagonal, or past the end
            for (long j = 0; j < d; j++)
                if (row[j] != zero)
                    return false;
            for (long j = d + 1; j < colSize; j++)
                if (row[j] != zero)
                    return false;
        }
        return true;
    }

    // Returns the sum of the main diagonal elements. Requires a square matrix.
    datatype tr() const {
        try {
            if (!(rowSize == colSize && rowSize > 0))
                throw std::invalid_argument("tr() requires a square non-empty matrix, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            // Indexed straight into grid rather than through operator(),
            // which wraps negative indices and so runs a modulo on BOTH
            // coordinates — two integer divisions (~20-40 cycles each) per
            // element, for indices the loop bounds already prove in range.
            // The stride is colSize+1, so this is a diagonal walk with one
            // cache miss per element and nothing else to do; four
            // accumulators keep those misses in flight concurrently instead
            // of serialising on the add.
            const datatype* MATRIXCPP_RESTRICT g = grid;
            const long step = colSize + 1;
            datatype s0 = datatype(0), s1 = datatype(0), s2 = datatype(0), s3 = datatype(0);
            long i = 0;
            for (; i + 3 < rowSize; i += 4) {
                s0 += g[i * step];
                s1 += g[(i + 1) * step];
                s2 += g[(i + 2) * step];
                s3 += g[(i + 3) * step];
            }
            for (; i < rowSize; i++)
                s0 += g[i * step];
            return (s0 + s1) + (s2 + s3);
        } catch (const std::exception& e) {
            std::cerr << "Matrix trace error: " << e.what() << '\n';
            throw;
        }
    }
    // Returns the sum of all elements in the matrix.
    datatype sum() const {
        // Defers to pairwiseSum — NumPy's reduction algorithm — which is both
        // faster (eight independent accumulator chains instead of one
        // latency-bound one) and more accurate (O(log n · eps) error growth
        // instead of O(n · eps)). See its definition near the top of the file.
        return pairwiseSum(grid, rowSize * colSize);
    }
    // Dimensional sum. axis=0: returns a (1 x cols) row matrix of column sums.
    //                  axis=1: returns a (rows x 1) column matrix of row sums.
    Matrix sum(const bool& axis) const {
        // Accumulates straight out of grid. The slice-based version this
        // replaced built a whole temporary Matrix per row/column, so a
        // (n x n) sum cost n allocations and n copies on top of the arithmetic.
        const datatype* MATRIXCPP_RESTRICT g = grid;
        if (!axis) {
            // Column sums. The accumulator is the whole output row, so this
            // is already a vector operation — but only if the compiler can
            // prove the output does not alias the input. It cannot: both are
            // datatype* from the same allocator, so without __restrict every
            // store to out[j] has to be re-loaded before the next row. The
            // restrict qualifiers are what let this run at memory bandwidth.
            Matrix<datatype> total_vec(1, colSize);
            datatype* MATRIXCPP_RESTRICT out = total_vec.grid;
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
                for (long j = 0; j < colSize; j++)
                    out[j] += row[j];
            }
            return total_vec;
        }
        // Row sums. Each row is a contiguous run, so this is exactly the
        // reduction pairwiseSum exists for: the single-accumulator loop this
        // replaced was latency-bound and read 32 MB at 15 GB/s on a machine
        // that streams at 45.
        Matrix<datatype> total_vec(rowSize, 1);
        datatype* MATRIXCPP_RESTRICT out = total_vec.grid;
        for (long i = 0; i < rowSize; i++)
            out[i] = pairwiseSum(g + i * colSize, colSize);
        return total_vec;
    }

    // --- Reductions ---
    // Every reduction comes in two forms, mirroring sum() / sum(bool) above:
    //   f()      → scalar over all elements
    //   f(bool)  → axis=0: (1 x cols) row matrix of per-column results
    //              axis=1: (rows x 1) column matrix of per-row results
    // The axis versions can reuse the (*this)(all, i) / (*this)(i, all) slice
    // pattern that sum(bool) already uses.

    // min/max and the arg- variants order their elements with <, which
    // std::complex deliberately does not provide. Instantiating them on a
    // complex Matrix is a compile-time error rather than a silent choice of
    // some arbitrary ordering; every other reduction here works for complex.

    // Smallest element in the matrix.
    datatype min() const {
        static_assert(!is_complex<datatype>::value,
                      "min() needs an ordering; std::complex has none. Use A.abs().min().");
        requireNonEmpty("min");
        datatype best = grid[0];
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[i] < best)
                best = grid[i];
        return best;
    }

    // Per-column (axis=0) or per-row (axis=1) minima.
    Matrix min(const bool& axis) const {
        static_assert(!is_complex<datatype>::value,
                      "min() needs an ordering; std::complex has none. Use A.abs().min().");
        requireNonEmpty("min");
        if (!axis) {
            Matrix<datatype> out(1, colSize);
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype& v = grid[i * colSize + j];
                    if (v < out.grid[j])
                        out.grid[j] = v;
                }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype best = grid[i * colSize];
            for (long j = 1; j < colSize; j++)
                if (grid[i * colSize + j] < best)
                    best = grid[i * colSize + j];
            out.grid[i] = best;
        }
        return out;
    }

    // Largest element in the matrix.
    datatype max() const {
        static_assert(!is_complex<datatype>::value,
                      "max() needs an ordering; std::complex has none. Use A.abs().max().");
        requireNonEmpty("max");
        datatype best = grid[0];
        for (long i = 1; i < rowSize * colSize; i++)
            if (best < grid[i])
                best = grid[i];
        return best;
    }

    // Per-column (axis=0) or per-row (axis=1) maxima.
    Matrix max(const bool& axis) const {
        static_assert(!is_complex<datatype>::value,
                      "max() needs an ordering; std::complex has none. Use A.abs().max().");
        requireNonEmpty("max");
        if (!axis) {
            Matrix<datatype> out(1, colSize);
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype& v = grid[i * colSize + j];
                    if (out.grid[j] < v)
                        out.grid[j] = v;
                }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype best = grid[i * colSize];
            for (long j = 1; j < colSize; j++)
                if (best < grid[i * colSize + j])
                    best = grid[i * colSize + j];
            out.grid[i] = best;
        }
        return out;
    }

    // Arithmetic mean of all elements. Returns double so integral matrices
    // do not truncate — note this differs from sum(), which preserves datatype.
    // Returns double for a real matrix and complex<double> for a complex one —
    // the mean of complex numbers IS complex, and the old signature could not
    // say so, which is why this used to refuse them outright.
    mean_t<datatype> mean() const {
        requireNonEmpty("mean");
        mean_t<datatype> acc = mean_t<datatype>(0);
        for (long i = 0; i < rowSize * colSize; i++) acc += mean_t<datatype>(grid[i]);
        return acc / double(rowSize * colSize);
    }

    // Per-column (axis=0) or per-row (axis=1) means. Element type follows
    // the input, as the scalar mean() does.
    Matrix<mean_t<datatype>> mean(const bool& axis) const {
        requireNonEmpty("mean");
        using M = mean_t<datatype>;
        if (!axis) {
            Matrix<M> out(1, colSize);
            for (long i = 0; i < rowSize; i++)
                for (long j = 0; j < colSize; j++)
                    out[int(j)] += M(grid[i * colSize + j]);
            for (long j = 0; j < colSize; j++) out[int(j)] /= double(rowSize);
            return out;
        }
        Matrix<M> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            M acc = M(0);
            for (long j = 0; j < colSize; j++) acc += M(grid[i * colSize + j]);
            out[int(i)] = acc / double(colSize);
        }
        return out;
    }

    // Variance of all elements. sample=false divides by N (population variance),
    // sample=true divides by N-1 (Bessel-corrected sample variance).
    // Two-pass: the mean first, then the squared deviations from it. The
    // one-pass E[x²]-E[x]² shortcut loses most of its significant digits when
    // the mean is large relative to the spread, so it is not used here.
    // Variance stays REAL even for a complex matrix, because it is defined as
    // E|x - mu|^2 — a sum of squared magnitudes, which cannot be complex. NumPy's
    // np.var does the same. So the return type is double whatever went in, and
    // only the subtraction and the magnitude need to know about complex.
    double var(bool sample = false) const {
        requireNonEmpty("var");
        long N = rowSize * colSize;
        if (sample && N < 2)
            throw std::invalid_argument("var(sample=true) needs at least 2 elements, got " +
                                        std::to_string(N));
        mean_t<datatype> mu = mean();
        double acc = 0.0;
        for (long i = 0; i < N; i++)
            acc += magnitudeSq(mean_t<datatype>(grid[i]) - mu);
        return acc / double(sample ? N - 1 : N);
    }

    // Standard deviation of all elements — sqrt of var(sample).
    double stddev(bool sample = false) const { return std::sqrt(var(sample)); }

    // Position {row, col} of the smallest element. Ties resolve to the first
    // encountered in row-major order.
    std::pair<long, long> argmin() const {
        static_assert(!is_complex<datatype>::value,
                      "argmin() needs an ordering; std::complex has none.");
        requireNonEmpty("argmin");
        long best = 0;
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[i] < grid[best])
                best = i;
        return {best / colSize, best % colSize};
    }

    // Position {row, col} of the largest element. Ties resolve to the first
    // encountered in row-major order.
    std::pair<long, long> argmax() const {
        static_assert(!is_complex<datatype>::value,
                      "argmax() needs an ordering; std::complex has none.");
        requireNonEmpty("argmax");
        long best = 0;
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[best] < grid[i])
                best = i;
        return {best / colSize, best % colSize};
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Hessenberg and Schur, exposed  (tier 4)
    // ═══════════════════════════════════════════════════════════════════

    // Upper Hessenberg form: A = Q * H * Qᵀ with H zero below the first
    // subdiagonal and Q orthogonal. This is the first half of every dense
    // eigenvalue algorithm — reducing to Hessenberg costs O(n³) once, after
    // which each QR sweep is O(n²) instead of O(n³).
    //
    // Written out here rather than lifted out of schurDecomp: that function
    // computes the same reduction, but pulling it apart would mean surgery
    // on the routine that eig, pow(A,real) and log(A) all depend on, for a
    // 30-line saving. The duplication is deliberate and small.
    //
    // Usage: auto [H, Q] = A.hess();
    std::pair<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>> hess() const {
        using W = work_t<datatype>;
        auto cj = [](const W& z) {
            if constexpr (is_complex<datatype>::value) return std::conj(z);
            else return z;
        };
        if (rowSize != colSize)
            throw std::invalid_argument("hess: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        const long n = rowSize;
        Matrix<W> H(n, n), Q(n, n);
        for (long i = 0; i < n * n; i++)
            H[int(i)] = W(grid[i]);
        for (long i = 0; i < n; i++)
            Q(int(i), int(i)) = W(1);

        std::vector<W> v((std::size_t)n);
        for (long k = 0; k + 2 < n; k++) {
            // Householder reflector zeroing H(k+2..n-1, k).
            double nrm = 0.0;
            for (long i = k + 1; i < n; i++)
                nrm += magnitudeSq(H(int(i), int(k)));
            nrm = std::sqrt(nrm);
            if (nrm == 0.0)
                continue;
            W alpha;
            if constexpr (is_complex<datatype>::value) {
                // Opposite in PHASE. The real sign rule does not generalise —
                // picking -nrm outright loses the cancellation it exists to avoid.
                const W hk = H(int(k + 1), int(k));
                const double a0 = std::abs(hk);
                alpha = -((a0 == 0.0) ? W(1) : hk / a0) * nrm;
            } else {
                alpha = W(H(int(k + 1), int(k)) >= W(0) ? -nrm : nrm);
            }
            for (long i = k + 1; i < n; i++)
                v[(std::size_t)i] = H(int(i), int(k));
            v[(std::size_t)(k + 1)] -= alpha;
            double vtv = 0.0;
            for (long i = k + 1; i < n; i++)
                vtv += magnitudeSq(v[(std::size_t)i]);
            if (vtv == 0.0)
                continue;
            const double tau = 2.0 / vtv;
            // H := (I - tau v v^H) H (I - tau v v^H), applied from both sides.
            // For a real matrix v^H is v^T and cj() is the identity, so this is
            // the same arithmetic the real-only version always did.
            for (long j = 0; j < n; j++) {
                W d = W(0);
                for (long i = k + 1; i < n; i++)
                    d += cj(v[(std::size_t)i]) * H(int(i), int(j));
                d *= tau;
                for (long i = k + 1; i < n; i++)
                    H(int(i), int(j)) -= d * v[(std::size_t)i];
            }
            for (long i = 0; i < n; i++) {
                W d = W(0);
                for (long j = k + 1; j < n; j++)
                    d += H(int(i), int(j)) * v[(std::size_t)j];
                d *= tau;
                for (long j = k + 1; j < n; j++)
                    H(int(i), int(j)) -= d * cj(v[(std::size_t)j]);
            }
            // Accumulate Q the same way, so that A == Q H Q^H afterwards.
            for (long i = 0; i < n; i++) {
                W d = W(0);
                for (long j = k + 1; j < n; j++)
                    d += Q(int(i), int(j)) * v[(std::size_t)j];
                d *= tau;
                for (long j = k + 1; j < n; j++)
                    Q(int(i), int(j)) -= d * cj(v[(std::size_t)j]);
            }
        }
        // Clean the numerical dust below the subdiagonal so the result is
        // exactly Hessenberg rather than Hessenberg to within rounding.
        for (long i = 2; i < n; i++)
            for (long j = 0; j + 2 <= i; j++)
                H(int(i), int(j)) = W(0);
        return {H, Q};
    }

    // Real Schur form: A = Q * T * Qᵀ with Q orthogonal and T quasi-upper
    // triangular — 1x1 blocks for real eigenvalues, 2x2 blocks for each
    // complex-conjugate pair, which is why it is "quasi".
    //
    // schurDecomp() below computes this already but hands back flat
    // std::vectors because it is the internal engine for eig, pow and log.
    // This is the public spelling.
    //
    // Factor once, solve many times. See the Decomposition class below for
    // which factorisation is chosen and why. Declared here and defined after
    // that class, which cannot be complete before Matrix is.
    //
    // Usage: auto dA = A.factorize();  then  dA.solve(b1), dA.solve(b2), ...
    Decomposition<datatype> factorize() const;

    // Usage: auto [T, Q] = A.schur();
    std::pair<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>> schur() const {
        using W = work_t<datatype>;
        if (rowSize != colSize)
            throw std::invalid_argument("schur: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        // For a complex matrix T is fully triangular rather than quasi-: there
        // is no such thing as a real 2x2 block when the arithmetic is complex.
        std::vector<W> Tv, Qv;
        if constexpr (is_complex<datatype>::value) {
            auto pr = schurDecompComplex();
            Tv = std::move(pr.first);
            Qv = std::move(pr.second);
        } else {
            auto pr = schurDecomp();
            Tv = std::move(pr.first);
            Qv = std::move(pr.second);
        }
        const long n = rowSize;
        Matrix<W> T(n, n, typename Matrix<W>::uninit_t{});
        Matrix<W> Q(n, n, typename Matrix<W>::uninit_t{});
        for (long i = 0; i < n * n; i++) {
            T[int(i)] = Tv[(std::size_t)i];
            Q[int(i)] = Qv[(std::size_t)i];
        }
        return {T, Q};
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Structure predicates  (tier 4)
    // ═══════════════════════════════════════════════════════════════════
    // Named to match the IsDiagonal() that was already here. That capital I
    // is not this header's usual style, but a predicate family that agrees
    // with itself beats one where half the members match MATLAB's lowercase
    // and half match the neighbour they sit next to.
    //
    // Every one takes a tolerance relative to the size of the entries, so
    // they answer "is this matrix symmetric" rather than "did these floats
    // come out bit-identical", which is almost never the useful question.

    bool IsSymmetric(double tol = -1.0) const {
        if (rowSize != colSize)
            return false;
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < i; j++)
                if (magnitude(grid[i * colSize + j] - grid[j * colSize + i]) > t)
                    return false;
        return true;
    }

    // A == conj(A)^T. For a real matrix this is the same question as
    // IsSymmetric; for a complex one it is the one that actually matters,
    // since it is Hermitian — not symmetric — that gives real eigenvalues.
    bool IsHermitian(double tol = -1.0) const {
        if (rowSize != colSize)
            return false;
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j <= i; j++) {
                datatype conjugated;
                if constexpr (is_complex<datatype>::value)
                    conjugated = std::conj(grid[j * colSize + i]);
                else
                    conjugated = grid[j * colSize + i];
                if (magnitude(grid[i * colSize + j] - conjugated) > t)
                    return false;
            }
        return true;
    }

    // Upper triangular: everything below the k-th diagonal is zero.
    // k > 0 tests above the main diagonal, k < 0 below — same convention as
    // triu()/tril(), so IsUpper(k) is exactly "A == A.triu(k)".
    bool IsUpper(int k = 0, double tol = -1.0) const {
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (j - i < k && magnitude(grid[i * colSize + j]) > t)
                    return false;
        return true;
    }
    bool IsLower(int k = 0, double tol = -1.0) const {
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (j - i > k && magnitude(grid[i * colSize + j]) > t)
                    return false;
        return true;
    }

    // How far the non-zeros reach below and above the main diagonal.
    // {0, 0} is diagonal, {rows-1, cols-1} is dense. MATLAB's bandwidth.
    std::pair<long, long> bandwidth(double tol = -1.0) const {
        const double t = structureTol(tol);
        long below = 0, above = 0;
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (magnitude(grid[i * colSize + j]) > t) {
                    if (i > j && i - j > below)
                        below = i - j;
                    if (j > i && j - i > above)
                        above = j - i;
                }
        return {below, above};
    }

    bool IsBanded(long lower, long upper, double tol = -1.0) const {
        const std::pair<long, long> b = bandwidth(tol);
        return b.first <= lower && b.second <= upper;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Subspaces, elimination and vector products  (tier 4)
    // ═══════════════════════════════════════════════════════════════════

    // Orthonormal basis for the null space: the columns of V belonging to
    // singular values that are numerically zero. A * null(A) == 0.
    // Empty (n x 0) when A has full column rank.
    Matrix<double> null(double tol = -1.0) const {
        auto [U, S, V] = svd();
        (void)U;
        const long n = colSize;
        const double t = svdTol(S, tol);
        std::vector<long> keep;
        for (long j = 0; j < n; j++)
            if (j >= std::min(rowSize, colSize) || S(int(j), int(j)) <= t)
                keep.push_back(j);
        Matrix<double> out(n, (long)keep.size(), Matrix<double>::uninit_t{});
        for (std::size_t c = 0; c < keep.size(); c++)
            for (long i = 0; i < n; i++)
                out(int(i), int(c)) = V(int(i), int(keep[c]));
        return out;
    }

    // Orthonormal basis for the range (column space): the columns of U
    // belonging to the non-zero singular values. m x rank(A).
    Matrix<double> orth(double tol = -1.0) const {
        auto [U, S, V] = svd();
        (void)V;
        const double t = svdTol(S, tol);
        std::vector<long> keep;
        for (long j = 0; j < std::min(rowSize, colSize); j++)
            if (S(int(j), int(j)) > t)
                keep.push_back(j);
        Matrix<double> out(rowSize, (long)keep.size(), Matrix<double>::uninit_t{});
        for (std::size_t c = 0; c < keep.size(); c++)
            for (long i = 0; i < rowSize; i++)
                out(int(i), int(c)) = U(int(i), int(keep[c]));
        return out;
    }

    // Reduced row echelon form, by Gauss-Jordan with partial pivoting.
    //
    // A NOTE ON WHAT THIS IS FOR: rref is a teaching and exact-arithmetic
    // tool. On floating-point data the pivot-is-zero decision is a guess,
    // and a slightly different tolerance can change the reported rank. Use
    // rank(), null() and solve(), which go through the SVD or a pivoted QR,
    // for anything numerical. MATLAB's documentation says the same thing
    // about its rref, and it is worth repeating here.
    Matrix<double> rref(double tol = -1.0) const {
        Matrix<double> R(rowSize, colSize);
        for (long i = 0; i < rowSize * colSize; i++)
            R[int(i)] = double(std::real(grid[i]));
        const double t = tol >= 0.0 ? tol
                                    : std::numeric_limits<double>::epsilon() *
                                          double(std::max(rowSize, colSize)) *
                                          std::max(1.0, norm(NormType::Inf));
        long row = 0;
        for (long col = 0; col < colSize && row < rowSize; col++) {
            // Partial pivoting: the largest remaining entry in this column.
            long piv = row;
            for (long i = row + 1; i < rowSize; i++)
                if (std::abs(R(int(i), int(col))) > std::abs(R(int(piv), int(col))))
                    piv = i;
            if (std::abs(R(int(piv), int(col))) <= t) {
                for (long i = row; i < rowSize; i++)
                    R(int(i), int(col)) = 0.0;
                continue;  // no pivot in this column
            }
            if (piv != row)
                for (long j = 0; j < colSize; j++)
                    std::swap(R(int(row), int(j)), R(int(piv), int(j)));
            const double d = R(int(row), int(col));
            for (long j = 0; j < colSize; j++)
                R(int(row), int(j)) /= d;
            for (long i = 0; i < rowSize; i++) {
                if (i == row)
                    continue;
                const double f = R(int(i), int(col));
                if (f == 0.0)
                    continue;
                for (long j = 0; j < colSize; j++)
                    R(int(i), int(j)) -= f * R(int(row), int(j));
            }
            row++;
        }
        return R;
    }

    // Inner product of two equally shaped operands, sum(conj(a) * b) — the
    // conjugate goes on the LEFT, which is the convention that makes
    // A.dot(A) equal ||A||_F² for complex as well as real.
    datatype dot(const Matrix& B) const {
        requireSameShape(B, "dot");
        datatype acc = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++) {
            if constexpr (is_complex<datatype>::value)
                acc += std::conj(grid[i]) * B.grid[i];
            else
                acc += grid[i] * B.grid[i];
        }
        return acc;
    }

    // Cross product. Three elements only — it is the one dimension where a
    // vector product of two vectors is again a vector.
    Matrix cross(const Matrix& B) const {
        if (rowSize * colSize != 3 || B.rows() * B.cols() != 3)
            throw std::invalid_argument("cross: both operands must have exactly 3 elements, got " +
                                        std::to_string(rowSize * colSize) + " and " +
                                        std::to_string(B.rows() * B.cols()));
        Matrix out(rowSize, colSize, uninit_t{});
        const datatype* a = grid;
        const datatype* b = B.grid;
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
        return out;
    }

    // Estimate of the 2-norm by power iteration on AᵀA, which is what
    // MATLAB's normest is for: norm(A, Two) computes a full SVD, and when
    // all you want is the largest singular value to a few digits that is a
    // great deal of work to throw away.
    double normest(double tol = 1e-6, int maxIter = 100) const {
        if (rowSize * colSize == 0)
            return 0.0;
        const long n = colSize;
        std::vector<double> x((std::size_t)n, 1.0 / std::sqrt(double(n)));
        std::vector<double> Ax((std::size_t)rowSize), y((std::size_t)n);
        double est = 0.0;
        for (int it = 0; it < maxIter; it++) {
            for (long i = 0; i < rowSize; i++) {
                double acc = 0.0;
                for (long j = 0; j < n; j++)
                    acc += double(std::real(grid[i * n + j])) * x[(std::size_t)j];
                Ax[(std::size_t)i] = acc;
            }
            for (long j = 0; j < n; j++) {
                double acc = 0.0;
                for (long i = 0; i < rowSize; i++)
                    acc += double(std::real(grid[i * n + j])) * Ax[(std::size_t)i];
                y[(std::size_t)j] = acc;
            }
            double nrm = 0.0;
            for (long j = 0; j < n; j++)
                nrm += y[(std::size_t)j] * y[(std::size_t)j];
            nrm = std::sqrt(nrm);
            if (nrm == 0.0)
                return 0.0;
            const double next = std::sqrt(nrm);
            for (long j = 0; j < n; j++)
                x[(std::size_t)j] = y[(std::size_t)j] / nrm;
            if (it && std::abs(next - est) <= tol * next)
                return next;
            est = next;
        }
        return est;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Scans, orderings and multiset reductions  (roadmap tier 2)
    // ═══════════════════════════════════════════════════════════════════
    //
    // Every one of these takes the same `axis` flag the existing reductions
    // use, and means the same thing by it — the axis you get ONE RESULT PER:
    //     COL   work DOWN each column  (result is 1 x n, one per column)
    //     ROW   work ALONG each row    (result is m x 1, one per row)
    // so prod(COL) pairs with sum(COL), cumsum(COL) accumulates down columns,
    // sort(COL) sorts each column, and so on. They are plain bools, so the
    // older sum(false)/sum(true) spelling still compiles.
    //
    // prod / cumsum / cumprod are defined for complex. Anything that has to
    // ORDER elements — sort, median, mode, unique — is not, for the reason
    // min() and max() already give: std::complex deliberately has no <, and
    // inventing one silently is worse than refusing to compile.

    // Product of every element.
    datatype prod() const {
        requireNonEmpty("prod");
        datatype p = datatype(1);
        for (long i = 0; i < rowSize * colSize; i++)
            p *= grid[i];
        return p;
    }

    // Per-column (axis=0) or per-row (axis=1) products.
    Matrix prod(const bool& axis) const {
        const datatype* MATRIXCPP_RESTRICT g = grid;
        if (!axis) {
            Matrix<datatype> out(1, colSize);
            datatype* MATRIXCPP_RESTRICT o = out.grid;
            for (long j = 0; j < colSize; j++)
                o[j] = datatype(1);
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
                for (long j = 0; j < colSize; j++)
                    o[j] *= row[j];
            }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype p = datatype(1);
            const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
            for (long j = 0; j < colSize; j++)
                p *= row[j];
            out.grid[i] = p;
        }
        return out;
    }

    // Cumulative sum. Same shape as the input — a scan, not a reduction.
    Matrix cumsum(const bool& axis) const { return scan(axis, true); }
    // Cumulative product, likewise.
    Matrix cumprod(const bool& axis) const { return scan(axis, false); }

    // Successive differences along an axis. The scanned dimension shrinks by
    // one, so diff(false) on (m x n) gives (m-1 x n) — MATLAB's diff.
    // ── Calculus on samples  (tier 6) ──────────────────────────────────────
    // These are MEMBERS taking the same axis flag as sum(), cumsum() and
    // diff(), because that is what they are: operations along an axis of array
    // data. MATLAB spells them as free functions, but this header already made
    // that call for cumsum and diff and agreeing with itself matters more.
    // conv, filter and the rest below stay free, because those treat a whole
    // vector as one signal rather than working along an axis.

    // Trapezoidal integral over every element, unit spacing.
    double trapz() const {
        const long n = rowSize * colSize;
        if (n < 2) return 0.0;   // one sample spans no interval; MATLAB agrees
        double acc = 0.0;
        for (long i = 0; i + 1 < n; i++)
            acc += 0.5 * (double(std::real(grid[i])) + double(std::real(grid[i + 1])));
        return acc;
    }

    // Per-column (axis=0) or per-row (axis=1) trapezoidal integrals.
    Matrix<double> trapz(const bool& axis) const {
        const long outer = axis ? rowSize : colSize;
        const long inner = axis ? colSize : rowSize;
        Matrix<double> out(axis ? rowSize : 1, axis ? 1 : colSize);
        if (inner < 2) return out;
        for (long k = 0; k < outer; k++) {
            double acc = 0.0;
            for (long t = 0; t + 1 < inner; t++) {
                const double a = axis ? double(std::real(grid[k * colSize + t]))
                                        : double(std::real(grid[t * colSize + k]));
                const double b = axis ? double(std::real(grid[k * colSize + t + 1]))
                                        : double(std::real(grid[(t + 1) * colSize + k]));
                acc += 0.5 * (a + b);
            }
            out[int(k)] = acc;
        }
        return out;
    }

    // Cumulative trapezoidal integral — same shape as the input, starting at 0,
    // so that cumtrapz(...)'s last entry equals trapz(...) along the same axis.
    Matrix<double> cumtrapz(const bool& axis) const {
        Matrix<double> out(rowSize, colSize);
        const long outer = axis ? rowSize : colSize;
        const long inner = axis ? colSize : rowSize;
        for (long k = 0; k < outer; k++) {
            double acc = 0.0;
            for (long t = 1; t < inner; t++) {
                const double a = axis ? double(std::real(grid[k * colSize + t - 1]))
                                        : double(std::real(grid[(t - 1) * colSize + k]));
                const double b = axis ? double(std::real(grid[k * colSize + t]))
                                        : double(std::real(grid[t * colSize + k]));
                acc += 0.5 * (a + b);
                if (axis) out(int(k), int(t)) = acc;
                else        out(int(t), int(k)) = acc;
            }
        }
        return out;
    }

    // Numerical derivative, SAME LENGTH as the input — that is the whole
    // difference between gradient and diff, and the reason both exist.
    // diff(false) on an m-row matrix gives m-1 rows; gradient(false) gives m,
    // by using a CENTRED difference in the interior and a one-sided difference
    // at each end. h is the sample spacing.
    Matrix<double> gradient(const bool& axis, double h = 1.0) const {
        if (h == 0.0) throw std::invalid_argument("gradient: spacing h must be non-zero");
        Matrix<double> out(rowSize, colSize);
        const long outer = axis ? rowSize : colSize;
        const long inner = axis ? colSize : rowSize;
        if (inner < 2) return out;
        auto at = [&](long k, long t) {
            return axis ? double(std::real(grid[k * colSize + t]))
                          : double(std::real(grid[t * colSize + k]));
        };
        auto put = [&](long k, long t, double v) {
            if (axis) out(int(k), int(t)) = v;
            else        out(int(t), int(k)) = v;
        };
        for (long k = 0; k < outer; k++) {
            put(k, 0, (at(k, 1) - at(k, 0)) / h);                      // forward
            for (long t = 1; t + 1 < inner; t++)
                put(k, t, (at(k, t + 1) - at(k, t - 1)) / (2.0 * h));  // centred
            put(k, inner - 1, (at(k, inner - 1) - at(k, inner - 2)) / h);  // backward
        }
        return out;
    }

    Matrix diff(const bool& axis) const {
        if (!axis) {
            if (rowSize < 2)
                return Matrix(0, colSize);
            Matrix out(rowSize - 1, colSize, uninit_t{});
            for (long i = 0; i + 1 < rowSize; i++)
                for (long j = 0; j < colSize; j++)
                    out.grid[i * colSize + j] = grid[(i + 1) * colSize + j] - grid[i * colSize + j];
            return out;
        }
        if (colSize < 2)
            return Matrix(rowSize, 0);
        Matrix out(rowSize, colSize - 1, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j + 1 < colSize; j++)
                out.grid[i * (colSize - 1) + j] = grid[i * colSize + j + 1] - grid[i * colSize + j];
        return out;
    }

    // Sorts each column (axis=0) or each row (axis=1). Same shape as the
    // input, like MATLAB's sort — this rearranges, it does not reduce.
    Matrix sort(const bool& axis, bool descending = false) const {
        static_assert(!is_complex<datatype>::value,
                      "sort() needs an ordering; std::complex has none. Sort a component "
                      "or a magnitude instead — for example A.abs().sort(0).");
        Matrix out = *this;
        std::vector<datatype> buf((std::size_t)(axis ? colSize : rowSize));
        const long outer = axis ? rowSize : colSize;
        const long inner = axis ? colSize : rowSize;
        for (long k = 0; k < outer; k++) {
            for (long t = 0; t < inner; t++)
                buf[(std::size_t)t] = axis ? grid[k * colSize + t] : grid[t * colSize + k];
            if (descending)
                std::sort(buf.begin(), buf.end(), std::greater<datatype>());
            else
                std::sort(buf.begin(), buf.end());
            for (long t = 0; t < inner; t++) {
                if (axis)
                    out.grid[k * colSize + t] = buf[(std::size_t)t];
                else
                    out.grid[t * colSize + k] = buf[(std::size_t)t];
            }
        }
        return out;
    }

    // Sorts whole ROWS, ordered by column `key`, carrying every other column
    // along — MATLAB's sortrows. The rows keep their contents; only their
    // order changes.
    Matrix sortrows(long key = 0, bool descending = false) const {
        static_assert(!is_complex<datatype>::value,
                      "sortrows() needs an ordering; std::complex has none.");
        if (rowSize == 0 || colSize == 0)
            return *this;
        if (key < 0 || key >= colSize)
            throw std::out_of_range("sortrows: key column " + std::to_string(key) +
                                    " is out of range for " + std::to_string(colSize) + " columns");
        std::vector<long> order((std::size_t)rowSize);
        std::iota(order.begin(), order.end(), 0L);
        const datatype* g = grid;
        const long cs = colSize;
        // stable_sort so that rows tying on the key keep their input order,
        // which is what makes repeated sortrows calls compose predictably.
        std::stable_sort(order.begin(), order.end(), [=](long a, long b) {
            return descending ? (g[b * cs + key] < g[a * cs + key])
                              : (g[a * cs + key] < g[b * cs + key]);
        });
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                out.grid[i * colSize + j] = grid[order[(std::size_t)i] * colSize + j];
        return out;
    }

    // Middle value; the mean of the two middle values when the count is even,
    // which is why this returns double rather than datatype.
    double median() const {
        static_assert(!is_complex<datatype>::value,
                      "median() needs an ordering; std::complex has none.");
        requireNonEmpty("median");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        return medianOf(buf);
    }

    // Per-column (axis=0) or per-row (axis=1) medians.
    Matrix<double> median(const bool& axis) const {
        static_assert(!is_complex<datatype>::value,
                      "median() needs an ordering; std::complex has none.");
        requireNonEmpty("median");
        const long outer = axis ? rowSize : colSize;
        const long inner = axis ? colSize : rowSize;
        Matrix<double> out(axis ? rowSize : 1, axis ? 1 : colSize);
        std::vector<datatype> buf((std::size_t)inner);
        for (long k = 0; k < outer; k++) {
            for (long t = 0; t < inner; t++)
                buf[(std::size_t)t] = axis ? grid[k * colSize + t] : grid[t * colSize + k];
            out[int(k)] = medianOf(buf);
        }
        return out;
    }

    // Most frequently occurring value. Ties go to the SMALLEST such value,
    // which is what MATLAB's mode does.
    datatype mode() const {
        static_assert(!is_complex<datatype>::value,
                      "mode() needs an ordering; std::complex has none.");
        requireNonEmpty("mode");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        std::sort(buf.begin(), buf.end());
        datatype best = buf[0];
        long bestRun = 0, run = 0;
        for (std::size_t i = 0; i < buf.size(); i++) {
            run = (i && buf[i] == buf[i - 1]) ? run + 1 : 1;
            if (run > bestRun) {
                bestRun = run;
                best = buf[i];
            }
        }
        return best;
    }

    // The distinct values, ascending, as a column vector — MATLAB's unique.
    Matrix unique() const {
        static_assert(!is_complex<datatype>::value,
                      "unique() needs an ordering; std::complex has none.");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        std::sort(buf.begin(), buf.end());
        buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
        Matrix out((long)buf.size(), 1, uninit_t{});
        for (std::size_t i = 0; i < buf.size(); i++)
            out.grid[i] = buf[i];
        return out;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Shape manipulation  (tier 5)
    // ═══════════════════════════════════════════════════════════════════

    // Total element count. rows()*cols(), but spelled the way MATLAB and
    // NumPy both spell it, and without the chance of writing rows*rows.
    long numel() const { return rowSize * colSize; }

    // Tiles this matrix m times down and n times across.
    Matrix repmat(long m, long n) const {
        if (m < 0 || n < 0)
            throw std::invalid_argument("repmat: counts must be non-negative, got " +
                                        std::to_string(m) + " and " + std::to_string(n));
        Matrix out(rowSize * m, colSize * n, uninit_t{});
        for (long bi = 0; bi < m; bi++)
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT src = grid + i * colSize;
                datatype* dst = out.grid + (bi * rowSize + i) * colSize * n;
                for (long bj = 0; bj < n; bj++)
                    std::copy(src, src + colSize, dst + bj * colSize);
            }
        return out;
    }

    // Reverses the column order — a mirror about the vertical axis.
    Matrix fliplr() const {
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                out.grid[i * colSize + j] = grid[i * colSize + (colSize - 1 - j)];
        return out;
    }

    // Reverses the row order — a mirror about the horizontal axis. Rows are
    // contiguous, so this is a run of whole-row copies.
    Matrix flipud() const {
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            std::copy(grid + (rowSize - 1 - i) * colSize,
                      grid + (rowSize - i) * colSize,
                      out.grid + i * colSize);
        return out;
    }

    // Rotates by k quarter-turns COUNTERCLOCKWISE, as MATLAB's rot90 does.
    // Negative k turns the other way; k is taken modulo 4, so rot90(5) is
    // rot90(1) and rot90(-1) is rot90(3).
    Matrix rot90(int k = 1) const {
        int t = k % 4;
        if (t < 0)
            t += 4;
        switch (t) {
            case 0:
                return *this;
            case 1:
                return T().flipud();  // counterclockwise
            case 2:
                return fliplr().flipud();  // half turn
            default:
                return T().fliplr();  // clockwise
        }
    }

    // Circularly shifts elements by k places. dim=0 shifts rows (down for
    // positive k), dim=1 shifts columns (right). Nothing is lost: what falls
    // off one end reappears at the other.
    Matrix circshift(long k, int dim = 0) const {
        if (dim != 0 && dim != 1)
            throw std::invalid_argument("circshift: dim must be 0 (rows) or 1 (columns), got " +
                                        std::to_string(dim));
        Matrix out(rowSize, colSize, uninit_t{});
        const long n = (dim == 0) ? rowSize : colSize;
        if (n == 0)
            return out;
        // C's % keeps the sign of the dividend, so a negative shift needs
        // wrapping back into [0, n) before it can be used as an offset.
        const long shift = ((k % n) + n) % n;
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++) {
                const long si = (dim == 0) ? ((i - shift) % rowSize + rowSize) % rowSize : i;
                const long sj = (dim == 1) ? ((j - shift) % colSize + colSize) % colSize : j;
                out.grid[i * colSize + j] = grid[si * colSize + sj];
            }
        return out;
    }

    // Block-diagonal concatenation: this and M placed on the diagonal of a
    // larger matrix, everything else zero. Shapes need not match, and need
    // not even be square.
    Matrix blkdiag(const Matrix& M) const {
        Matrix out(rowSize + M.rowSize, colSize + M.colSize);
        for (long i = 0; i < rowSize; i++)
            std::copy(grid + i * colSize, grid + (i + 1) * colSize, out.grid + i * out.colSize);
        for (long i = 0; i < M.rowSize; i++)
            std::copy(M.grid + i * M.colSize,
                      M.grid + (i + 1) * M.colSize,
                      out.grid + (rowSize + i) * out.colSize + colSize);
        return out;
    }

    // Concatenates M to this matrix. concatCol=0: vertical (stack rows, cols must
    // match).
    //                               concatCol=1: horizontal (stack cols, rows
    //                               must match).
    // If this matrix is empty, returns M directly.
    Matrix concat(const Matrix& M, const bool& concatCol) const {
        if (this->empty())
            return M;
        try {
            if (!concatCol && this->colSize != M.colSize)
                throw std::invalid_argument(
                    "concat: column size mismatch: " + std::to_string(colSize) +
                    " != " + std::to_string(M.colSize));
            if (concatCol && this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "concat: row size mismatch: " + std::to_string(rowSize) +
                    " != " + std::to_string(M.rowSize));
        } catch (const std::exception& e) {
            std::cerr << "Matrix concat error: " << e.what() << std::endl;
            throw;
        }
        // Both branches used to go through operator(), which wraps negative
        // indices and therefore runs a modulo on BOTH coordinates — four
        // integer divisions per element copied, for indices already known to
        // be in range. These are plain contiguous copies instead.
        // std::copy rather than a hand-written loop: for a trivially copyable
        // element type it lowers to memmove, which the C library implements
        // with wide vector loads and non-temporal stores. concat is purely
        // memory-bound, so that is the whole cost of the operation.
        if (!concatCol) {
            // Vertical: rows stack and the row length is unchanged, so the
            // two source blocks are already contiguous runs.
            Matrix<datatype> out(rowSize + M.rowSize, colSize, uninit_t{});
            const long a = rowSize * colSize;
            std::copy(grid, grid + a, out.grid);
            std::copy(M.grid, M.grid + M.rowSize * colSize, out.grid + a);
            return out;
        }
        // Horizontal: rows interleave, so copy one row segment at a time.
        const long outCols = colSize + M.colSize;
        Matrix<datatype> out(rowSize, outCols, uninit_t{});
        for (long i = 0; i < rowSize; i++) {
            datatype* dst = out.grid + i * outCols;
            std::copy(grid + i * colSize, grid + (i + 1) * colSize, dst);
            std::copy(M.grid + i * M.colSize, M.grid + (i + 1) * M.colSize, dst + colSize);
        }
        return out;
    }

    // Kronecker product: replaces every element A_ij with the block A_ij * M.
    //
    // Named kron(), not tensor(), for two reasons. It is what MATLAB, NumPy
    // and SciPy all call it, and — since Tensor.hpp added a TYPE called
    // Tensor — `A.tensor(B)` would have been an operation sharing its name
    // with a data type while returning the other one (a Matrix). That is the
    // kind of collision that reads fine to whoever wrote it and confuses
    // everyone else.
    //
    // It stays a named function rather than getting an operator. The result
    // is (m*p) x (n*q), so it is the one operation here that explodes: two
    // 1000x1000 matrices give 10^12 elements, 8 TB. That cost should be
    // visible at the call site, not two characters away. It is also why the
    // benchmark only takes this one to n = 48 while everything else runs to
    // 512 or 2000.
    // Returns a (rows*M.rows x cols*M.cols) Matrix.
    Matrix kron(const Matrix& M) const {
        const long rM = M.rowSize, cM = M.colSize;
        const long cOut = colSize * cM;
        Matrix<datatype> ans(rowSize * rM, cOut, uninit_t{});
        // Iterating by output block rather than by flat index. The flat-index
        // version cost six integer divisions per element — two to split idx
        // into (r, c), two more to split those into block and offset, and two
        // more inside operator() — for a loop whose structure already knows
        // every one of those values. Nested loops carry them for free, and
        // the A element is loaded once per block instead of per element.
        for (long i = 0; i < rowSize; i++) {
            for (long j = 0; j < colSize; j++) {
                const datatype a = grid[i * colSize + j];
                for (long p = 0; p < rM; p++) {
                    datatype* dst = ans.grid + (i * rM + p) * cOut + j * cM;
                    const datatype* src = M.grid + p * cM;
                    for (long q = 0; q < cM; q++)
                        dst[q] = a * src[q];
                }
            }
        }
        return ans;
    }

    // --- Structural extraction and reshaping ---

    // Extracts the main diagonal as a (min(rows,cols) x 1) column vector.
    // Works for non-square matrices. To go the other way — build a diagonal
    // matrix FROM a vector — use the free function diag(v) near the bottom.
    Matrix diag() const {
        long d = std::min(rowSize, colSize);
        Matrix<datatype> out(d, 1);
        for (long i = 0; i < d; i++)
            out.grid[i] = grid[i * colSize + i];
        return out;
    }

    // Upper triangle: copies elements on and above the k-th diagonal, zeros the
    // rest. k=0 is the main diagonal, k>0 moves above it, k<0 below. Mirrors
    // numpy.triu.
    Matrix triu(int k = 0) const {
        Matrix<datatype> out(rowSize, colSize);
        for (long i = 0; i < rowSize; i++)
            for (long j = std::max(0L, i + k); j < colSize; j++)
                out.grid[i * colSize + j] = grid[i * colSize + j];
        return out;
    }

    // Lower triangle: copies elements on and below the k-th diagonal, zeros the
    // rest. k=0 is the main diagonal, k>0 moves above it, k<0 below. Mirrors
    // numpy.tril.
    Matrix tril(int k = 0) const {
        Matrix<datatype> out(rowSize, colSize);
        for (long i = 0; i < rowSize; i++) {
            long hi = std::min(colSize - 1, i + k);
            for (long j = 0; j <= hi; j++)
                out.grid[i * colSize + j] = grid[i * colSize + j];
        }
        return out;
    }

    // Reinterprets the elements as a (newRows x newCols) matrix in row-major
    // order. Requires newRows * newCols == rows * cols. Returns a new Matrix; the
    // underlying data is copied, not aliased.
    Matrix reshape(long newRows, long newCols) const {
        if (newRows < 0 || newCols < 0)
            throw std::invalid_argument("reshape: dimensions must be non-negative, got " +
                                        std::to_string(newRows) + "x" + std::to_string(newCols));
        if (newRows * newCols != rowSize * colSize)
            throw std::invalid_argument("reshape: cannot reshape " + std::to_string(rowSize) + "x" +
                                        std::to_string(colSize) + " (" +
                                        std::to_string(rowSize * colSize) + " elements) into " +
                                        std::to_string(newRows) + "x" + std::to_string(newCols) +
                                        " (" + std::to_string(newRows * newCols) + ")");
        // uninit_t, not the zeroing constructor: every element is about to
        // be overwritten, so zero-filling first writes the whole buffer twice.
        // std::copy rather than an element loop, so a trivially copyable type
        // lowers to memmove and its wide vector stores.
        Matrix<datatype> out(newRows, newCols, uninit_t{});
        std::copy(grid, grid + rowSize * colSize, out.grid);
        return out;
    }

    // --- Random initialisation ---

    // Fills every element with a random value in [lowBound, highBound).
    // Seed is derived from the current time (unique within a 24-hour window).
    // Integral types are rounded to the nearest integer.
    // Returns *this to allow chaining.
    Matrix& set_Ran_values(double lowBound, double highBound) {
        try {
            if (lowBound >= highBound)
                throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) +
                    " > high: " + std::to_string(highBound));
            double range = highBound - lowBound;
            long seed;
            setRan(seed);
            for (long i = 0; i < rowSize * colSize; i++) {
                double val = range * ran2(&seed) + lowBound;
                grid[i] =
                    std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
            }
        } catch (const std::exception& e) {
            std::cerr << "Bounds error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }
    // Fills every element with a random value in [lowBound, highBound) using a
    // custom seed. customSeed MUST be negative (required by ran2 to trigger
    // initialisation). Integral types are rounded to the nearest integer. Returns
    // *this to allow chaining.
    Matrix& set_Ran_values(double lowBound, double highBound, long customSeed) {
        try {
            if (customSeed >= 0)
                throw std::invalid_argument("Seed, " + std::to_string(customSeed) + " is >= 0");
            if (lowBound >= highBound)
                throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) +
                    " > high: " + std::to_string(highBound));
            double range = highBound - lowBound;
            long seed = customSeed;
            for (long i = 0; i < rowSize * colSize; i++) {
                double val = range * ran2(&seed) + lowBound;
                grid[i] =
                    std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }
    //

    // QR factorisation with column pivoting using Householder reflections.
    // Mirrors LAPACK's DGEQP3: at each step the column with the largest remaining
    // norm is pivoted to the front, then a Householder reflector eliminates the
    // sub-diagonal entries of that column. Column norms are maintained via the
    // Bischof-Pan rank-1 downdate, avoiding a full norm recomputation each step.
    //
    // For m×n matrix A, computes  A * P = Q * R  where:
    //   Q — m×m orthogonal (product of Householder reflectors)
    //   R — m×n upper triangular
    //   P — n×n permutation matrix (column pivoting for stability)
    //
    // Works for any m×n, including m < n.
    // Usage: auto [Q, R, P] = A.QR();
    // Q and R follow the input type: ORTHOGONAL for a real matrix, UNITARY for
    // a complex one, with A*P == Q*R either way. P is a permutation and stays in
    // the input type.
    // ═══════════════════════════════════════════════════════════════════
    //  QR, split the way LAPACK splits it
    // ═══════════════════════════════════════════════════════════════════
    // LAPACK does not have "a QR routine". It has THREE, and the split is the
    // whole point:
    //     dgeqrf/dgeqp3  factor, leaving Q implicit as a list of reflectors
    //     dormqr         APPLY Q (or Q^H) to something, without ever forming it
    //     dorgqr         form Q explicitly, only if you actually want the matrix
    //
    // This header had only the third. Everything went through a full m×m Q, and
    // for a tall thin least-squares that is a catastrophe: an 8000x100 solve
    // built a 512 MB Q to produce a 100-element answer, and took 2.4 seconds.
    // Forming Q is O(m²·r); applying the reflectors to one right-hand side is
    // O(m·r). At 8000x100 that is 6.4e9 operations against 8e5.
    //
    // QRFactored is the "Q left implicit" form. QR() below still returns an
    // explicit Q because that is the interface a caller asked for, but solve()
    // and rank() no longer pay for one they were going to discard.
    struct QRFactored {
        std::vector<work_t<datatype>> wt;   // column-major; R lives in the upper triangle
        std::vector<double> taus;
        // The reflectors, in ONE flat block rather than r separate vectors.
        // Reflector k has length m-k and starts at hoff[k]. LAPACK and Eigen go
        // one step further and pack them into the lower triangle of wt itself,
        // normalising v[0] to 1 so it need not be stored — that is worth doing
        // if the blocked update ever lands, since it wants the whole
        // factorisation in one contiguous array. This gets the allocation win
        // without the implicit-unit convention.
        std::vector<work_t<datatype>> hstore;
        std::vector<long> hoff;
        std::vector<int> pivots;
        int m = 0, n = 0, r = 0;
    };

    // Householder QR with column pivoting, stopping before Q is formed.
    QRFactored qrFactor() const {
        using W = work_t<datatype>;
        auto cj = [](const W& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        QRFactored F;

            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("QR: matrix must be non-empty");

            int m = (int)rowSize, n = (int)colSize;
            int r = std::min(m, n);

            // Working copy in double, stored COLUMN BY COLUMN — Wt[j*m + i]
            // is A(i,j). Every step of a Householder QR works down columns: the
            // reflector is built from a column, applied to each trailing column,
            // and the pivot search swaps whole columns. Row-major storage makes
            // all of that stride by n, one cache line per element, which is why
            // the factorisation ran at ~0.3 GFLOP/s. LAPACK's dgeqp3 never has
            // this problem because Fortran arrays are column-major; storing the
            // transpose gets the same contiguity here, and the reflector
            // application becomes a plain dot product followed by an axpy.
            // Q is already accumulated transposed for exactly this reason —
            // see the Q block below.
            std::vector<W> Wt((size_t)n * m);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) Wt[(size_t)j * m + i] = W(grid[(size_t)i * n + j]);
            W* MATRIXCPP_RESTRICT wt = Wt.data();

            // Column pivot tracking — pivots[k] = original column index at position k
            std::vector<int> pivots(n);
            std::iota(pivots.begin(), pivots.end(), 0);

            // Squared column norms for Bischof-Pan pivot selection
            std::vector<double> sqNorms(n, 0.0);
            for (int j = 0; j < n; j++)
                sqNorms[j] = sumSq(wt + (size_t)j * m, m);

            // Householder taus + vectors stored for Q accumulation
            std::vector<double> taus(r, 0.0);
            // sum over k of (m-k) — the exact total, no over-allocation.
            std::vector<long> hoff((std::size_t)r + 1, 0);
            for (int k = 0; k < r; k++) hoff[(std::size_t)(k + 1)] = hoff[(std::size_t)k] + (m - k);
            std::vector<W> hstore((std::size_t)hoff[(std::size_t)r], W(0));

            for (int k = 0; k < r; k++) {
                // ── Pivot: bring the largest-norm remaining column to position k ──
                int jmax = k;
                for (int j = k + 1; j < n; j++)
                    if (sqNorms[j] > sqNorms[jmax])
                        jmax = j;
                if (jmax != k) {
                    // A column is a contiguous run now, so the swap is one
                    // memory-to-memory exchange instead of m strided ones.
                    std::swap_ranges(
                        wt + (size_t)k * m, wt + (size_t)k * m + m, wt + (size_t)jmax * m);
                    std::swap(pivots[k], pivots[jmax]);
                    std::swap(sqNorms[k], sqNorms[jmax]);
                }

                W* MATRIXCPP_RESTRICT colk = wt + (size_t)k * m;
                const int sz = m - k;

                // ── Householder reflector for column k, rows k:m-1 ──
                // alpha is chosen to point AWAY from x[0] so the subtraction
                // below never cancels. For a real x that is just the opposite
                // sign; for a complex one it is the opposite PHASE, which is the
                // same statement — sign is the real case of phase.
                double xnorm = std::sqrt(sumSq(colk + k, sz));

                if (xnorm == 0.0) continue;   // hstore is already zero here

                W alpha;
                if constexpr (is_complex<datatype>::value) {
                    const double a0 = magnitude(colk[k]);
                    const W phase = (a0 == 0.0) ? W(1) : colk[k] / a0;
                    alpha = -phase * xnorm;
                } else {
                    alpha = W((colk[k] >= W(0) ? -1.0 : 1.0) * xnorm);
                }
                W* MATRIXCPP_RESTRICT v = hstore.data() + hoff[(std::size_t)k];
                std::copy(colk + k, colk + m, v);
                v[0] -= alpha;  // v = x - alpha*e_1

                // v^H v, which is real because it is a sum of squared magnitudes.
                double vTv = 0.0;
                for (int i = 0; i < sz; i++) vTv += magnitudeSq(v[i]);
                if (vTv == 0.0) {
                    std::fill(v, v + sz, W(0));
                    continue;
                }
                double tau = 2.0 / vTv;
                taus[k] = tau;

                // Apply H_k = I - tau*v*v^T to trailing block W(k:m-1, k:n-1).
                // Every column j is updated independently of every other, so
                // this — the O(mn²) bulk of the factorisation — is embarrassingly
                // parallel. It is also the loop that column pivoting forces to
                // stay at BLAS level 2: the norms have to be downdated before
                // the next pivot can be chosen, so unlike LAPACK's unpivoted
                // blocked dgeqrf there is no way to batch several reflectors
                // into one matrix-matrix product.
                const W* MATRIXCPP_RESTRICT vp = v;
                auto applyCol = [=](int j) {
                    W* MATRIXCPP_RESTRICT wj = wt + (size_t)j * m + k;
                    // conj(v)·w, the Hermitian inner product — with the plain
                    // one the reflector would not be unitary for complex.
                    W d0 = W(0), d1 = W(0);
                    int i = 0;
                    for (; i + 1 < sz; i += 2) {
                        d0 += cj(vp[i]) * wj[i];
                        d1 += cj(vp[i + 1]) * wj[i + 1];
                    }
                    for (; i < sz; i++) d0 += cj(vp[i]) * wj[i];
                    const W f = tau * (d0 + d1);
                    for (i = 0; i < sz; i++)
                        wj[i] -= f * vp[i];
                };
                const long applyWork = (long)(n - k) * sz;
                (void)applyWork;  // only read on the OpenMP path
#ifdef _OPENMP
                if (applyWork >= 32768) {
                    const int th = (int)std::min<long>(std::max<long>(applyWork / 8192, 1),
                                                       omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(th)
                    for (int j = k; j < n; j++)
                        applyCol(j);
                } else
#endif
                {
                    for (int j = k; j < n; j++)
                        applyCol(j);
                }

                // Bischof-Pan downdate: H_k is orthogonal so column norms are
                // preserved; the squared norm below row k shrinks by W(k,j)^2.
                for (int j = k + 1; j < n; j++) {
                    // |W(k,j)|^2 — real, so the downdate stays real for complex.
                    sqNorms[j] -= magnitudeSq(wt[(size_t)j * m + k]);
                    if (sqNorms[j] < 0.0)
                        sqNorms[j] = 0.0;
                }
            }
        F.wt = std::move(Wt);
        F.taus = std::move(taus);
        F.hstore = std::move(hstore);
        F.hoff = std::move(hoff);
        F.pivots = std::move(pivots);
        F.m = m; F.n = n; F.r = r;
        return F;
    }

    // Unpivoted Householder QR — LAPACK's dgeqrf, MATLAB's two-output qr(A).
    //
    // Identical to qrFactor() minus the pivot search and the column-norm
    // downdating, which is pure overhead when the caller does not need the
    // rank-revealing property. P comes back as the identity.
    //
    // NOT BLOCKED, and that is a measured decision rather than an omission —
    // see the QR notes in the roadmap at the foot of this file.
    QRFactored qrFactorUnpivoted() const {
        using W = work_t<datatype>;
        auto cj = [](const W& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        const int m = (int)rowSize, n = (int)colSize;
        const int r = std::min(m, n);

        QRFactored F;
        F.m = m; F.n = n; F.r = r;
        // Transposed into column-major, so that every trailing column a
        // reflector touches is contiguous. That is what makes the level-2
        // update here beat a blocked level-3 one.
        F.wt.assign((std::size_t)n * m, W(0));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) F.wt[(std::size_t)j * m + i] = W(grid[(std::size_t)i * n + j]);
        F.taus.assign((std::size_t)r, 0.0);
        F.pivots.resize((std::size_t)n);
        std::iota(F.pivots.begin(), F.pivots.end(), 0);   // no pivoting: identity
        F.hoff.assign((std::size_t)r + 1, 0);
        for (int k = 0; k < r; k++) F.hoff[(std::size_t)(k + 1)] = F.hoff[(std::size_t)k] + (m - k);
        F.hstore.assign((std::size_t)F.hoff[(std::size_t)r], W(0));

        W* MATRIXCPP_RESTRICT wt = F.wt.data();
        for (int k = 0; k < r; k++) {
            W* MATRIXCPP_RESTRICT colk = wt + (std::size_t)k * m;
            const int sz = m - k;
            const double xnorm = std::sqrt(sumSq(colk + k, sz));
            if (xnorm == 0.0) continue;
            W alpha;
            if constexpr (is_complex<datatype>::value) {
                // Opposite in PHASE, not merely in sign — the real rule does
                // not generalise, and picking the wrong one loses cancellation.
                const double a0 = magnitude(colk[k]);
                alpha = -((a0 == 0.0) ? W(1) : colk[k] / a0) * xnorm;
            } else {
                alpha = W((colk[k] >= W(0) ? -1.0 : 1.0) * xnorm);
            }
            W* MATRIXCPP_RESTRICT v = F.hstore.data() + F.hoff[(std::size_t)k];
            std::copy(colk + k, colk + m, v);
            v[0] -= alpha;
            double vTv = 0.0;
            for (int i = 0; i < sz; i++) vTv += magnitudeSq(v[i]);
            if (vTv == 0.0) { std::fill(v, v + sz, W(0)); continue; }
            const double tau = 2.0 / vTv;
            F.taus[(std::size_t)k] = tau;
            colk[k] = alpha;
            std::fill(colk + k + 1, colk + m, W(0));

#ifdef _OPENMP
            const long work = (long)(n - k - 1) * sz;
            #pragma omp parallel for schedule(static) if (work >= mstore::PARALLEL_MIN_WORK)
#endif
            for (int c = k + 1; c < n; c++) {
                W* MATRIXCPP_RESTRICT cc = wt + (std::size_t)c * m + k;
                W d = W(0);
                for (int i = 0; i < sz; i++) d += cj(v[i]) * cc[i];
                const W f = tau * d;
                for (int i = 0; i < sz; i++) cc[i] -= f * v[i];
            }
        }
        return F;
    }

    // Q^H * B, WITHOUT forming Q — LAPACK's dormqr. Q = H_0 H_1 … H_{r-1} and
    // each reflector is Hermitian (H = I - tau v v^H with tau real, so H^H = H),
    // hence Q^H = H_{r-1} … H_0 and the reflectors apply in FORWARD order.
    // B is overwritten.
    static void applyQH(const QRFactored& F, Matrix<work_t<datatype>>& B) {
        using W = work_t<datatype>;
        auto cj = [](const W& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        const int nrhs = (int)B.cols();
        for (int k = 0; k < F.r; k++) {
            if (F.taus[(std::size_t)k] == 0.0) continue;
            const W* v = F.hstore.data() + F.hoff[(std::size_t)k];
            const int sz = F.m - k;
            const double tau = F.taus[(std::size_t)k];
            for (int c = 0; c < nrhs; c++) {
                W d = W(0);
                for (int i = 0; i < sz; i++) d += cj(v[i]) * B(k + i, c);
                const W f = tau * d;
                for (int i = 0; i < sz; i++) B(k + i, c) -= f * v[i];
            }
        }
    }

    // A*P == Q*R with Q unitary (orthogonal for a real matrix). This forms Q
    // explicitly — LAPACK's dorgqr step — which is what a caller asking for the
    // matrix wants. If you only need Q^H*b or only R, use qrFactor() and
    // applyQH() instead and skip an O(m²·r) step entirely.
    //
    //   QR()                      pivoted, Q is m×m   — MATLAB's [Q,R,P] = qr(A)
    //   QR(QRMode::Reduced)       pivoted, Q is m×k   — MATLAB's qr(A,0)
    //   QR(Complete, Pivot::Off)  blocked, Q is m×m   — MATLAB's [Q,R] = qr(A)
    //
    // With pivoting off P comes back as the identity and the blocked compact-WY
    // path runs; see QRPivot.
    std::tuple<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>, Matrix<datatype>> QR(
        QRMode mode = QRMode::Complete, QRPivot pivot = QRPivot::On) const {
        using W = work_t<datatype>;
        auto cj = [](const W& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        try {
            QRFactored F = (pivot == QRPivot::On) ? qrFactor() : qrFactorUnpivoted();
            const int m = F.m, n = F.n, r = F.r;
            const W* MATRIXCPP_RESTRICT wt = F.wt.data();
            const std::vector<double>& taus = F.taus;
            const std::vector<W>& hstore = F.hstore;
            const std::vector<long>& hoff = F.hoff;
            const std::vector<int>& pivots = F.pivots;

            // ── Materialise R (upper triangle of the worked array) ──
            // k columns of Q, and k rows of R. For Complete that is m; for
            // Reduced it is min(m,n), and the rows of R below it are all zero
            // by construction, so dropping them loses nothing.
            const int qcols = (mode == QRMode::Reduced) ? std::min(m, n) : m;
            Matrix<W> R(qcols, n);
            for (int i = 0; i < qcols; i++)
                for (int j = i; j < n; j++) R(i, j) = wt[(size_t)j * m + i];

            // ── Accumulate Q = H_0 * H_1 * … * H_{r-1} ──
            // Apply reflectors in reverse order to the m×m identity.
            // At descending step k, columns 0:k-1 of Q are zero in rows k:m-1,
            // so only columns k:m-1 need updating.
            //
            // Accumulated TRANSPOSED. Each reflector touches rows k..m-1 of a
            // fixed column, which in row-major storage strides by a whole row
            // — a cache miss per element, and this loop is the bulk of the
            // factorisation. Working on Qt makes the same update contiguous;
            // it is transposed back once at the end, which is O(m²) against
            // the O(m³) it saves. Same reasoning as the Q accumulation in
            // schurDecomp().
            std::vector<W> Qt((size_t)qcols * m, W(0));
            for (int i = 0; i < qcols; i++) Qt[(size_t)i * m + i] = W(1);
            W* MATRIXCPP_RESTRICT qt = Qt.data();
            for (int k = r - 1; k >= 0; k--) {
                if (taus[k] == 0.0)
                    continue;
                const W* MATRIXCPP_RESTRICT v = hstore.data() + hoff[(std::size_t)k];
                const int sz = m - k;
                const double tau = taus[k];
                // Independent per column j, exactly like the panel update, so
                // it parallelises the same way.
                auto applyQ = [=](int j) {
                    W* MATRIXCPP_RESTRICT qj = qt + (size_t)j * m + k;  // Q(k.., j)
                    // conj(v)·q again — the same Hermitian inner product the
                    // panel update uses, and for the same reason.
                    W d0 = W(0), d1 = W(0);
                    int i = 0;
                    for (; i + 1 < sz; i += 2) {
                        d0 += cj(v[i]) * qj[i];
                        d1 += cj(v[i + 1]) * qj[i + 1];
                    }
                    for (; i < sz; i++) d0 += cj(v[i]) * qj[i];
                    const W f = tau * (d0 + d1);
                    for (i = 0; i < sz; i++)
                        qj[i] -= f * v[i];
                };
                // qcols, not m: with Reduced there are fewer columns to update,
                // so the old (m - k) would have over-reported the work and
                // asked for threads there is nothing for.
                const long qWork = (long)(qcols - k) * sz;
                (void)qWork;  // only read on the OpenMP path
#ifdef _OPENMP
                if (qWork >= 32768) {
                    const int th =
                        (int)std::min<long>(std::max<long>(qWork / 8192, 1), omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(th)
                    for (int j = k; j < qcols; j++) applyQ(j);
                } else
#endif
                {
                    for (int j = k; j < qcols; j++) applyQ(j);
                }
            }
            Matrix<W> Q(m, qcols);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < qcols; j++) Q(i, j) = Qt[(size_t)j * m + i];

            // ── Materialise P: A*P = Q*R, so P[pivots[k], k] = 1 ──
            Matrix<datatype> P(n, n);
            for (int k = 0; k < n; k++)
                P(pivots[k], k) = datatype(1);

            return std::make_tuple(Q, R, P);

        } catch (const std::exception& e) {
            std::cerr << "QR factorization error: " << e.what() << '\n';
            throw;
        }
    }

    // Matrix norm. Works for any m×n.
    //   NormType::Fro — sqrt(sum of squares of every element)
    //   NormType::One — max absolute column sum
    //   NormType::Inf — max absolute row sum
    //   NormType::Two — largest singular value; needs svd(), so implement that
    //   first
    // Usage: double e = (A*x - b).norm();
    double norm(NormType type = NormType::Fro) const {
        if (rowSize * colSize == 0)
            return 0.0;
        switch (type) {
            case NormType::One: {
                double best = 0.0;
                for (long j = 0; j < colSize; j++) {
                    double acc = 0.0;
                    for (long i = 0; i < rowSize; i++)
                        acc += magnitude(grid[i * colSize + j]);
                    if (acc > best)
                        best = acc;
                }
                return best;
            }
            case NormType::Inf: {
                double best = 0.0;
                for (long i = 0; i < rowSize; i++) {
                    double acc = 0.0;
                    for (long j = 0; j < colSize; j++)
                        acc += magnitude(grid[i * colSize + j]);
                    if (acc > best)
                        best = acc;
                }
                return best;
            }
            case NormType::Two: {
                // The largest singular value. Works for complex now that svd()
                // does — the singular values are real either way, so this
                // returns a plain double whatever went in.
                auto [U, S, V] = svd();
                (void)U;
                (void)V;
                return S.rows() && S.cols() ? S(0, 0) : 0.0;  // sorted descending
            }
            case NormType::Fro:
            default: {
                // Fast path: one pass, four accumulators, tracking the largest
                // magnitude alongside the sum so the safety of the result can
                // be decided by arithmetic rather than by inspecting it.
                //
                // Deliberately NOT std::isfinite(acc): -ffast-math implies
                // -ffinite-math-only, under which that folds to a constant
                // true and the guard silently disappears. Comparing the
                // tracked maximum against a bound no optimisation flag can
                // reason away keeps this correct under every build line.
                const long total = rowSize * colSize;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, mx = 0.0;
                long i = 0;
                // Squared magnitudes throughout — the overflow guard compares
                // against the squared bound, so no square root is needed here
                // either. For a complex matrix that removes one sqrt per element.
                // The accumulators may overflow to infinity here; that is
                // harmless, because whether to trust them is decided by mx,
                // which tracks the largest COMPONENT and so cannot itself
                // overflow. Testing the accumulated square would be too late.
                for (; i + 3 < total; i += 4) {
                    a0 += magnitudeSq(grid[i]);
                    a1 += magnitudeSq(grid[i + 1]);
                    a2 += magnitudeSq(grid[i + 2]);
                    a3 += magnitudeSq(grid[i + 3]);
                    const double c0 = maxComponent(grid[i]), c1 = maxComponent(grid[i + 1]);
                    const double c2 = maxComponent(grid[i + 2]), c3 = maxComponent(grid[i + 3]);
                    const double p0 = c0 > c1 ? c0 : c1, p1 = c2 > c3 ? c2 : c3;
                    const double p = p0 > p1 ? p0 : p1;
                    if (p > mx)
                        mx = p;
                }
                for (; i < total; i++) {
                    a0 += magnitudeSq(grid[i]);
                    const double c = maxComponent(grid[i]);
                    if (c > mx)
                        mx = c;
                }
                // Safe when total * mx^2 cannot overflow, and when the squares
                // are still above the subnormal floor. Outside that window the
                // scaled pass below is the only way to get the right answer.
                // Safe when total * 2 * mx^2 stays inside double's range (the
                // 2 covers re^2 + im^2 for a complex element), and when the
                // squares stay above the subnormal floor.
                const double hiBound = std::sqrt(std::numeric_limits<double>::max() /
                                                 (2.0 * double(total > 0 ? total : 1)));
                const double loBound = std::sqrt(std::numeric_limits<double>::min()) * 1e3;
                if (mx == 0.0)
                    return 0.0;
                if (mx < hiBound && mx > loBound)
                    return std::sqrt((a0 + a1) + (a2 + a3));

                // Slow path: rescale by the largest magnitude so the squares
                // stay representable. Reached only for matrices whose entries
                // sit near the top or bottom of double's range.
                double sc = 0.0;
                for (long k = 0; k < total; k++) {
                    double t = magnitude(grid[k]) / mx;
                    sc += t * t;
                }
                return mx * std::sqrt(sc);
            }
        }
    }

    // Numerical rank: the number of linearly independent columns.
    // The column-pivoted QR above is already the rank-revealing tool — count the
    // diagonal entries of R whose magnitude exceeds tol. Passing tol < 0 selects
    // the LAPACK-style default, max(m,n) * eps * |R(0,0)|.
    long rank(double tol = -1.0) const {
        if (rowSize * colSize == 0)
            return 0;
        // rank() reads only the diagonal of R, so forming Q would be pure
        // waste — LAPACK's dgeqp3 leaves it implicit for exactly this reason.
        // At 8000x100 that is an O(m²·r) step and a 512 MB allocation avoided.
        QRFactored F = qrFactor();
        Matrix<work_t<datatype>> R(F.m, F.n);
        for (int i = 0; i < F.m; i++)
            for (int j = i; j < F.n; j++) R(i, j) = F.wt[(std::size_t)j * F.m + i];
        long d = std::min(rowSize, colSize);
        // Column pivoting orders |R(i,i)| non-increasingly, so the first entry
        // is the largest and the count can stop at the first one below tol.
        if (tol < 0.0)
            tol = double(std::max(rowSize, colSize)) * std::numeric_limits<double>::epsilon() *
                  std::abs(R(0, 0));
        long r = 0;
        for (long i = 0; i < d; i++) {
            if (std::abs(R(int(i), int(i))) <= tol)
                break;
            r++;
        }
        return r;
    }

    // Condition number ||A|| * ||A^-1|| in the given norm — how much a relative
    // error in b is amplified when solving A*x = b. Large means ill-conditioned.
    // NormType::Two is the usual choice and equals sigma_max / sigma_min from
    // svd().
    //
    // A singular matrix returns infinity. CAUTION under this project's compile
    // line: -ffast-math implies -ffinite-math-only, which lets the compiler
    // assume infinities never occur, and std::isinf() then folds to false. The
    // value returned is still infinity; it is the TEST that stops working. Use
    // a magnitude threshold (c > 1e15) rather than std::isinf() if you build
    // with -ffast-math, or drop the flag.
    double cond(NormType type = NormType::Two) const {
        const double inf = std::numeric_limits<double>::infinity();
        if (rowSize * colSize == 0)
            return 0.0;
        if (type == NormType::Two) {
            auto [U, S, V] = svd();
            (void)U;
            (void)V;
            long d = std::min(S.rows(), S.cols());
            double smax = S(0, 0), smin = S(int(d - 1), int(d - 1));
            return smin == 0.0 ? inf : smax / smin;  // singular, infinitely ill-conditioned
        }
        if (rowSize != colSize)
            throw std::invalid_argument(
                "cond: the One/Inf/Fro condition number needs a square matrix, got " +
                std::to_string(rowSize) + "x" + std::to_string(colSize) +
                " — use NormType::Two, which is defined for any shape");
        try {
            return norm(type) * inverse().norm(type);
        } catch (const std::exception&) {
            return inf;  // inverse() throws on a singular matrix
        }
    }

    // Performs LU factorization with partial pivoting (Doolittle's method).
    // Requires a square matrix of at least 2x2.
    // Returns std::tuple<L, U, P> where PA = LU.
    //
    // A SINGULAR MATRIX STILL HAS ONE: U simply carries a zero on its diagonal,
    // and P*A == L*U holds regardless. MATLAB's lu() does not error either. It
    // is solve() and inverse() that have to refuse, because there the answer
    // does not exist — not the factorisation, which does.
    //
    // Usage: auto [L, U, P] = A.LU();
    std::tuple<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>, Matrix<datatype>>
    LU() const {
        try {
            auto [packed, pivotVec] = luPacked(/*throwIfSingular=*/false);
            int n = (int)rowSize;
            auto pat = [&](int i, int j) { return packed[i * n + j]; };

            Matrix<work_t<datatype>> L(n, n), U(n, n);
            for (int i = 0; i < n; i++) {
                L(i, i) = work_t<datatype>(1);
                for (int j = 0; j < i; j++)
                    L(i, j) = pat(i, j);
                for (int j = i; j < n; j++)
                    U(i, j) = pat(i, j);
            }

            Matrix<datatype> P(n, n);
            for (int i = 0; i < n; i++)
                P(i, i) = datatype(1);
            for (int k = 0; k < n; k++) {
                if (pivotVec[k] != k)
                    for (int j = 0; j < n; j++)
                        std::swap(P(k, j), P(pivotVec[k], j));
            }

            return std::make_tuple(L, U, P);
        } catch (const std::exception& e) {
            std::cerr << "LU factorization error: " << e.what() << '\n';
            throw;
        }
    }

    // Cholesky factorisation A = L * L^T for symmetric positive-definite A.
    // Roughly half the work of LU since it exploits symmetry — the reason it is
    // the default for covariance matrices, normal equations, Kalman filters and
    // GP kernels. Returns the lower-triangular L (upper triangle zeroed).
    // Throws if A is not square, not symmetric, or not positive definite —
    // a failed Cholesky is in fact the standard *test* for positive definiteness.
    // Usage: auto L = A.cholesky();
    // Returns Matrix<double> for a real matrix and Matrix<complex<double>> for a
    // complex one — see work_t. For complex the factorisation is A = L * L^H,
    // the HERMITIAN one, and the input must be Hermitian rather than merely
    // symmetric: it is A == A^H that forces the diagonal real and the pivots
    // positive, which is what makes the whole thing well defined.
    Matrix<work_t<datatype>> cholesky() const {
        using W = work_t<datatype>;
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("cholesky: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("cholesky: matrix must be non-empty");
            int n = (int)rowSize;

            // Hermitian check, relative to the size of the entries involved. For a
            // real matrix conj() is the identity so this is the ordinary
            // symmetry test; for a complex one it is the test that matters,
            // since a complex SYMMETRIC matrix need not have real eigenvalues
            // and has no Cholesky factorisation in general.
            double scale = norm(NormType::Inf);
            double symTol =
                std::numeric_limits<double>::epsilon() * 100.0 * (scale > 0.0 ? scale : 1.0);
            auto conjOf = [](const datatype& v) {
                if constexpr (is_complex<datatype>::value) return std::conj(v);
                else return v;
            };
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    if (magnitude(grid[i * n + j] - conjOf(grid[j * n + i])) > symTol)
                        throw std::domain_error(
                            std::string(is_complex<datatype>::value ? "cholesky: matrix is not "
                                                                     "Hermitian — A("
                                                                   : "cholesky: matrix is not "
                                                                     "symmetric — A(") +
                            std::to_string(i) + "," + std::to_string(j) + ") != conj(A(" +
                            std::to_string(j) + "," + std::to_string(i) + "))");

            // Right-looking Cholesky in the shape of LAPACK's unblocked dpotf2:
            // row i of L is built from dot products of two ALREADY-COMPLETED
            // rows of L, which are contiguous runs in row-major order. Two
            // things were costing far more than the arithmetic:
            //
            //   * L(i,k) and L(j,k) went through operator(), which wraps
            //     negative indices and therefore runs a modulo on both
            //     coordinates — four integer divisions in the innermost loop
            //     of an O(n³/6) algorithm. This alone held the factorisation
            //     to 0.78 GFLOP/s.
            //   * one accumulator makes each multiply-add wait for the
            //     previous one, so the dot product ran at the latency of an
            //     FMA rather than its throughput. Four independent chains fix
            //     that, exactly as sum() does.
            //
            // The i==j case is split out of the j loop rather than tested
            // inside it: the diagonal needs L(i,k)² and a square root, the
            // off-diagonal needs a division, and branching on that in the
            // hot loop blocks vectorisation for both.
            Matrix<W> L(n, n);
            W* MATRIXCPP_RESTRICT Lg = L.grid;
            // conj() on the SECOND factor is what makes this the Hermitian
            // factorisation A = L*L^H. For a real datatype it is the identity,
            // so the same loops serve both and there is no second copy to keep
            // in step.
            auto cj = [](const W& v) {
                if constexpr (is_complex<datatype>::value) return std::conj(v);
                else return v;
            };
            for (long i = 0; i < n; i++) {
                W* MATRIXCPP_RESTRICT Li = Lg + i * n;
                for (long j = 0; j < i; j++) {
                    const W* MATRIXCPP_RESTRICT Lj = Lg + j * n;
                    W a0 = W(0), a1 = W(0), a2 = W(0), a3 = W(0);
                    long k = 0;
                    for (; k + 3 < j; k += 4) {
                        a0 += Li[k] * cj(Lj[k]);
                        a1 += Li[k + 1] * cj(Lj[k + 1]);
                        a2 += Li[k + 2] * cj(Lj[k + 2]);
                        a3 += Li[k + 3] * cj(Lj[k + 3]);
                    }
                    for (; k < j; k++) a0 += Li[k] * cj(Lj[k]);
                    const W acc = W(grid[i * n + j]) - ((a0 + a1) + (a2 + a3));
                    // L(j,j) is real and positive, so conj(L(j,j)) == L(j,j) and
                    // this division needs no special case.
                    Li[j] = acc / Lj[j];
                }
                // The diagonal accumulates |L(i,k)|^2, which is REAL however
                // complex the entries are — that is exactly why a Hermitian
                // matrix has real pivots and a merely symmetric complex one
                // does not.
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
                long k = 0;
                for (; k + 3 < i; k += 4) {
                    d0 += magnitudeSq(Li[k]);
                    d1 += magnitudeSq(Li[k + 1]);
                    d2 += magnitudeSq(Li[k + 2]);
                    d3 += magnitudeSq(Li[k + 3]);
                }
                for (; k < i; k++) d0 += magnitudeSq(Li[k]);
                const double piv =
                    double(std::real(grid[i * n + i])) - ((d0 + d1) + (d2 + d3));
                // A non-positive pivot IS the proof that A is not positive
                // definite — this failure is the standard PD test.
                if (piv <= 0.0)
                    throw std::domain_error(
                        "cholesky: matrix is not positive definite — non-positive pivot " +
                        std::to_string(piv) + " at index " + std::to_string(i));
                Li[i] = W(std::sqrt(piv));
            }
            return L;
        } catch (const std::exception& e) {
            std::cerr << "cholesky() error: " << e.what() << '\n';
            throw;
        }
    }

    // Returns the determinant via LU factorisation.
    // Integer types are rounded to avoid floating-point drift (e.g. 2.9999 → 3).
    datatype det() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("det() requires a square matrix, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            // A singular matrix has a determinant — it is zero — so this must not
            // throw. MATLAB and NumPy both return 0 here, and code that asks
            // "is this matrix singular" by testing det(A) == 0 is entitled to
            // an answer rather than an exception.
            auto [packed, pivotVec] = luPacked(/*throwIfSingular=*/false);
            int n = (int)rowSize;
            int sign = 1;
            for (int k = 0; k < n; k++)
                if (pivotVec[k] != k)
                    sign = -sign;
            // Accumulate in the working type: the determinant of a complex
            // matrix is complex, and datatype already says so.
            using W = work_t<datatype>;
            W d = W(sign);
            for (int i = 0; i < n; i++) d *= packed[(std::size_t)(i * n + i)];
            // An odd number of row swaps times a zero pivot gives NEGATIVE zero,
            // which compares equal to 0 but prints as "-0" and reads as a bug.
            // The comparison is true for -0.0, and the assignment replaces it
            // with +0.0.
            if (d == W(0)) d = W(0);
            if constexpr (std::is_integral<datatype>::value)
                return datatype(std::round(double(std::real(d))));
            else
                return datatype(d);
        } catch (const std::exception& e) {
            std::cerr << "det() error: " << e.what() << '\n';
            throw;
        }
    }

    // Eigendecomposition via the implicit-shift QR algorithm.
    // Usage: auto [eigenvalues, Q] = A.eig();
    //   eigenvalues — n×1 column vector (diagonal of Schur form)
    //   Q           — n×n orthogonal matrix (eigenvectors for symmetric A,
    //                 Schur vectors for general A)
    //
    // REAL eigenvalues only. schurDecomp() returns a REAL Schur form, in which
    // a complex-conjugate pair a±bi occupies a 2x2 diagonal block rather than
    // a single entry; reading the bare diagonal there yields `a` twice and
    // discards ±bi. This used to happen silently — a plain 2D rotation matrix
    // is enough to trigger it — so eig() now detects such a block and THROWS
    // rather than returning a plausible-looking wrong answer.
    //
    // If the matrix may have complex eigenvalues, call eigvals() below, which
    // is block-aware and returns Matrix<std::complex<double>>. Symmetric
    // matrices are always real-eigenvalued, so eig() is safe for those by
    // construction. schurDecomp() remains available for the raw factors.
    // Eigenvectors from a TRIANGULAR Schur form, by back-substitution.
    //
    // The Schur vectors in Q are NOT eigenvectors. A = Q T Q^H with T upper
    // triangular makes only the FIRST column of Q an eigenvector; the rest span
    // the invariant subspaces without being eigenvectors of anything. Returning
    // Q wholesale is silently wrong for every non-symmetric matrix — right
    // eigenvalues, wrong vectors — and it is only correct for symmetric input
    // because there T comes out diagonal.
    //
    // A(Qy) = Q(Ty), so an eigenvector of T maps to one of A. For lambda_k =
    // T(k,k), solve (T - lambda_k I) y = 0 with y_k = 1 and y_j = 0 for j > k,
    // back-substituting upward.
    template <class W>
    static Matrix<W> eigenvectorsFromSchur(const std::vector<W>& T,
                                           const std::vector<W>& Qv,
                                           int n) {
        double scale = 0.0;
        for (int i = 0; i < n * n; i++) scale = std::max(scale, magnitude(T[(std::size_t)i]));
        if (scale == 0.0) scale = 1.0;
        const double tiny = std::numeric_limits<double>::epsilon() * scale;

        Matrix<W> vecs(n, n);
        std::vector<W> y((std::size_t)n);
        for (int k = 0; k < n; k++) {
            const W lam = T[(std::size_t)k * n + k];
            std::fill(y.begin(), y.end(), W(0));
            y[(std::size_t)k] = W(1);
            for (int j = k - 1; j >= 0; j--) {
                W acc = W(0);
                for (int m = j + 1; m <= k; m++)
                    acc += T[(std::size_t)j * n + m] * y[(std::size_t)m];
                W den = T[(std::size_t)j * n + j] - lam;
                // A repeated eigenvalue makes the pivot zero and the eigenvector
                // non-unique. Perturbing rather than dividing by zero is what
                // LAPACK's dtrevc does, and it keeps a defective matrix from
                // returning NaN.
                if (magnitude(den) < tiny) den = W(tiny);
                y[(std::size_t)j] = -acc / den;
            }
            for (int i = 0; i < n; i++) {           // x = Q y
                W acc = W(0);
                for (int m = 0; m <= k; m++)
                    acc += Qv[(std::size_t)i * n + m] * y[(std::size_t)m];
                vecs(i, k) = acc;
            }
            double nrm = 0.0;
            for (int i = 0; i < n; i++) nrm += magnitudeSq(vecs(i, k));
            nrm = std::sqrt(nrm);
            if (nrm > 0.0)
                for (int i = 0; i < n; i++) vecs(i, k) /= W(nrm);
        }
        return vecs;
    }

    std::pair<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>> eig() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("eig: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("eig: matrix must be non-empty");
            using W = work_t<datatype>;
            int n = (int)rowSize;
            // A complex matrix has a genuinely triangular Schur form, so there
            // are no 2x2 blocks to reject and no real-eigenvalue restriction —
            // eig() is fully general there.
            std::vector<W> H, Qv;
            if constexpr (is_complex<datatype>::value) {
                auto pr = schurDecompComplex();
                H = std::move(pr.first);
                Qv = std::move(pr.second);
            } else {
                auto pr = schurDecomp();
                H = std::move(pr.first);
                Qv = std::move(pr.second);
                for (int i = 0; i + 1 < n; i++)
                    if (isSchurBlock(H, n, i))
                        throw std::domain_error(
                            "eig: the Schur form has a 2x2 block at index " + std::to_string(i) +
                            ", i.e. a complex-conjugate eigenvalue pair. Real eigenvalues "
                            "cannot represent it — use eigvals(), which returns "
                            "Matrix<std::complex<double>>");
            }
            Matrix<W> eigenvals(n, 1);
            for (int i = 0; i < n; i++)
                eigenvals(i, 0) = H[(std::size_t)i * n + i];
            return {eigenvals, eigenvectorsFromSchur<W>(H, Qv, n)};
        } catch (const std::exception& e) {
            std::cerr << "eig() error: " << e.what() << '\n';
            throw;
        }
    }

    // Every eigenvalue, complex ones included, as an n×1 complex column vector.
    // This is the block-aware counterpart to eig(): it walks the real Schur
    // form and, wherever a 2x2 block sits on the diagonal, takes both roots of
    // that block's characteristic polynomial λ² − tr·λ + det instead of reading
    // the diagonal entry. Ordering follows the Schur form, not magnitude.
    // Usage: auto lambda = A.eigvals();   // lambda(k,0) is std::complex<double>
    Matrix<std::complex<double>> eigvals() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("eigvals: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("eigvals: matrix must be non-empty");
            int n = (int)rowSize;
            Matrix<std::complex<double>> out(n, 1);
            if (n == 1) {
                if constexpr (is_complex<datatype>::value)
                    out(0, 0) = std::complex<double>(grid[0]);
                else
                    out(0, 0) = std::complex<double>(double(std::real(grid[0])), 0.0);
                return out;
            }

            // For a complex matrix the Schur form is triangular, so the
            // eigenvalues are simply its diagonal — none of the 2x2 block
            // unpacking below applies, because no such block can arise.
            if constexpr (is_complex<datatype>::value) {
                auto [Tc, Qc] = schurDecompComplex();
                (void)Qc;
                for (int i = 0; i < n; i++) out(i, 0) = Tc[(std::size_t)i * n + i];
                return out;
            } else {

            auto [H, Qv] = schurDecomp();
            (void)Qv;
            for (int i = 0; i < n;) {
                if (i + 1 < n && isSchurBlock(H, n, i)) {
                    double a = H[i * n + i], b = H[i * n + (i + 1)];
                    double c = H[(i + 1) * n + i], d = H[(i + 1) * n + (i + 1)];
                    double half = (a + d) / 2.0;
                    double disc = half * half - (a * d - b * c);
                    if (disc >= 0.0) {  // block did not actually pair up
                        double rt = std::sqrt(disc);
                        out(i, 0) = std::complex<double>(half + rt, 0.0);
                        out(i + 1, 0) = std::complex<double>(half - rt, 0.0);
                    } else {
                        double im = std::sqrt(-disc);
                        out(i, 0) = std::complex<double>(half, im);
                        out(i + 1, 0) = std::complex<double>(half, -im);
                    }
                    i += 2;
                } else {
                    out(i, 0) = std::complex<double>(H[i * n + i], 0.0);
                    i++;
                }
            }
            return out;
            }
        } catch (const std::exception& e) {
            std::cerr << "eigvals() error: " << e.what() << '\n';
            throw;
        }
    }

    // Singular value decomposition A = U * S * V^T. Works for any m×n.
    //   U — m×m orthogonal (left singular vectors)
    //   S — m×n diagonal, singular values in descending order, all >= 0
    //   V — n×n orthogonal (right singular vectors; note this returns V, not V^T)
    // The last major decomposition missing. Once it exists, norm(Two), cond(Two),
    // pinv() and a more robust rank() all fall out of it, as does PCA and
    // low-rank approximation.
    // Method: one-sided Jacobi, not the bidiagonalise-then-QR route. It
    // rotates pairs of columns of A until they are mutually orthogonal; at that
    // point A*V = U*S, so the column norms ARE the singular values and the
    // normalised columns ARE the left singular vectors. It is chosen here
    // because it computes the small singular values to high *relative*
    // accuracy, which is exactly what cond() and pinv() depend on, and because
    // it needs no shift strategy to converge.
    // Usage: auto [U, S, V] = A.svd();
    // U and V follow the input type (see work_t) and are orthogonal for a real
    // matrix, UNITARY for a complex one. S is always REAL — singular values are
    // magnitudes and cannot be complex, whatever went in — so it stays
    // Matrix<double> in both cases, and A == U * S * V^H throughout.
    std::tuple<Matrix<work_t<datatype>>, Matrix<double>, Matrix<work_t<datatype>>> svd()
        const {
        using W = work_t<datatype>;
        // conj() for the complex case, identity for the real one, so one body
        // serves both rather than two that can drift apart.
        auto cj = [](const W& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        try {
            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("svd: matrix must be non-empty");

            int m = (int)rowSize, n = (int)colSize;

            // One-sided Jacobi orthogonalises COLUMNS, so it needs at least as
            // many rows as columns. For a wide matrix, factor the transpose and
            // read the result back: A = (Aᵀ)ᵀ = (U'S'V'ᵀ)ᵀ = V' S'ᵀ U'ᵀ.
            if (m < n) {
                // The CONJUGATE transpose, not the plain one: A = (A^H)^H, and
                // A^H = V S^T U^H gives back A = U S V^H. Using the plain
                // transpose would conjugate the answer for a complex matrix.
                Matrix<datatype> At(n, m);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++) {
                        if constexpr (is_complex<datatype>::value)
                            At(j, i) = std::conj(grid[i * n + j]);
                        else
                            At(j, i) = grid[i * n + j];
                    }
                auto [U2, S2, V2] = At.svd();
                return std::make_tuple(V2, S2.T(), U2);
            }

            // ---- Layout: everything is stored TRANSPOSED. ----
            // One-sided Jacobi does all its work on COLUMNS: every rotation
            // reads and writes two whole columns of W and two of V. In a
            // row-major array a column is strided by n, so each rotation
            // touched one cache line per element, nothing vectorised, and the
            // whole factorisation ran at ~1.1 GFLOP/s. LAPACK does not have
            // this problem because Fortran is column-major — dgesvj's columns
            // are contiguous by construction.
            //
            // Storing the TRANSPOSES gets the same property here: row p of Wt
            // is column p of W, so a rotation is now two contiguous runs and
            // the compiler can vectorise it. This is the single biggest change
            // in the function.
            std::vector<W> Wt(std::size_t(n) * m), Vt(std::size_t(n) * n, W(0));
            for (long i = 0; i < m; i++)
                for (long j = 0; j < n; j++) Wt[j * m + i] = W(grid[i * n + j]);
            for (long j = 0; j < n; j++)
                Vt[j * n + j] = 1.0;
            W* MATRIXCPP_RESTRICT wt = Wt.data();
            W* MATRIXCPP_RESTRICT vt = Vt.data();

            // ---- Cached column norms, as in dgesvj's sva[] array. ----
            // The version this replaced recomputed all THREE inner products
            // (p·p, q·q, p·q) for every one of the n(n-1)/2 pairs in every
            // sweep. Only the cross term p·q actually changes unpredictably:
            // the two squared norms can be carried forward through the
            // rotation exactly, because zeroing the cross term means
            //     alpha' = alpha - t*gamma,   beta' = beta + t*gamma
            // (substitute s = c*t and 1 - t^2 = 2*zeta*t into the 2x2 Gram
            // update to check this). That drops the per-pair cost from three
            // dot products to one — and a pair that is already converged now
            // costs one dot product and no rotation at all.
            std::vector<double> sva(n);
            auto refreshNorms = [&]() {
                for (long j = 0; j < n; j++) {
                    const W* MATRIXCPP_RESTRICT wj = wt + j * m;
                    // Squared magnitudes, so the column norms stay real however
                    // complex the entries are — which is why the singular values
                    // come out real without any special handling.
                    double a0 = 0.0, a1 = 0.0;
                    long i = 0;
                    for (; i + 1 < m; i += 2) {
                        a0 += magnitudeSq(wj[i]);
                        a1 += magnitudeSq(wj[i + 1]);
                    }
                    for (; i < m; i++) a0 += magnitudeSq(wj[i]);
                    sva[j] = a0 + a1;
                }
            };
            refreshNorms();

            const double eps = std::numeric_limits<double>::epsilon();
            const int maxSweeps = 60;

            // One pair of columns: orthogonalise them and record how far from
            // orthogonal they were. Returns that relative off-diagonal size so
            // the sweep can decide whether it has converged.
            auto processPair = [&](long p, long q) -> double {
                const double alpha = sva[p], beta = sva[q];
                if (alpha == 0.0 || beta == 0.0)
                    return 0.0;
                W* MATRIXCPP_RESTRICT wp = wt + p * m;
                W* MATRIXCPP_RESTRICT wq = wt + q * m;

                // The inner product is conj(w_p)·w_q — HERMITIAN, so for a
                // complex matrix it is itself complex. For a real one cj() is
                // the identity and this is the ordinary dot product.
                W g0 = W(0), g1 = W(0);
                long i = 0;
                for (; i + 1 < m; i += 2) {
                    g0 += cj(wp[i]) * wq[i];
                    g1 += cj(wp[i + 1]) * wq[i + 1];
                }
                for (; i < m; i++) g0 += cj(wp[i]) * wq[i];
                const W gc = g0 + g1;
                const double gabs = magnitude(gc);
                if (gabs == 0.0)
                    return 0.0;

                // Relative, not absolute: this is what buys the small
                // singular values their relative accuracy.
                const double conv = gabs / std::sqrt(alpha * beta);
                if (conv <= eps)
                    return conv;

                // ── The complex case, in one extra step ────────────────────
                // A complex inner product cannot be zeroed by a real rotation.
                // But its PHASE can be rotated out first: scaling column q by
                // conj(g/|g|) makes conj(w_p)·w_q real and positive, after which
                // the ordinary real Jacobi rotation applies unchanged. The
                // scaling is by a unit modulus, so it is unitary and V absorbs
                // it exactly as it absorbs the rotation.
                //
                // gamma is then |g| rather than g. For a REAL matrix that
                // would change the sign convention of the existing path — a
                // negative g would flip column q — so the real path keeps the
                // signed value and skips the scaling entirely.
                double gamma;
                if constexpr (is_complex<datatype>::value) {
                    const W ph = cj(gc / gabs);          // conj of the phase
                    for (long k = 0; k < m; k++) wq[k] *= ph;
                    W* MATRIXCPP_RESTRICT vqp = vt + q * n;
                    for (long k = 0; k < n; k++) vqp[k] *= ph;
                    gamma = gabs;
                } else {
                    gamma = double(std::real(gc));
                }

                // Jacobi rotation zeroing the p-q inner product.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t =
                    (zeta >= 0.0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t), sn = c * t;
                for (long k = 0; k < m; k++) {
                    const W a = wp[k], b = wq[k];
                    wp[k] = c * a - sn * b;
                    wq[k] = sn * a + c * b;
                }
                W* MATRIXCPP_RESTRICT vp = vt + p * n;
                W* MATRIXCPP_RESTRICT vq = vt + q * n;
                for (long k = 0; k < n; k++) {
                    const W a = vp[k], b = vq[k];
                    vp[k] = c * a - sn * b;
                    vq[k] = sn * a + c * b;
                }
                sva[p] = alpha - t * gamma;
                sva[q] = beta + t * gamma;
                return conv;
            };

            // ---- Brent-Luk round-robin ("chess tournament") pair ordering. ----
            // The natural p<q double loop visits pairs in an order where
            // consecutive pairs share a column, so no two can be done at the
            // same time. The round-robin schedule instead splits a sweep into
            // np-1 ROUNDS of np/2 pairs each, where the pairs within a round
            // are column-disjoint by construction — seat the columns around a
            // circle, pair each seat with the one opposite, then rotate all but
            // one seat by a position. Every pair still occurs exactly once per
            // sweep, so this is a legitimate cyclic ordering with the same
            // convergence behaviour, but now a whole round runs in parallel.
            //
            // Being column-disjoint also makes the result DETERMINISTIC: no two
            // pairs in a round read or write the same column of W, V or sva, so
            // the answer does not depend on the thread count or the schedule.
            //
            // An odd number of columns gets one padding seat whose pairs are
            // skipped — the standard way to handle a bye in a round-robin.
            const long np = n + (n & 1);
            std::vector<long> ring(np);
            const long half = np / 2;

            // A sweep is np-1 rounds, so a 256-column matrix enters ~2000
            // parallel regions per factorisation. That makes the thread count
            // matter more than usual: too few and the cores idle, too many and
            // the ~360 ns region entry plus the barrier at the end of each
            // round costs more than the round does. Measured at n = 256:
            //
            //     threads   1     2     4     8    16    32
            //     ms       78.4  38.0  21.4  19.1  35.0  63.4
            //
            // so the useful range ends once a thread has fewer than ~8k
            // element-updates to do. Requesting that many threads and no more
            // tracks the optimum at every size. Below one thread's worth of
            // work the serial path runs with no OpenMP construct anywhere near
            // it — an `if` clause on the pragma is not enough, the runtime
            // still charges for evaluating it.
            const long roundWork = half * (m + n);
            const long WORK_PER_THREAD = 8192;
            (void)roundWork;
            (void)WORK_PER_THREAD;  // only read on the OpenMP path
#ifdef _OPENMP
            const int sweepThreads = (int)std::min<long>(
                std::max<long>(roundWork / WORK_PER_THREAD, 1), omp_get_max_threads());
#else
            const int sweepThreads = 1;
#endif
            const bool parallelSweep = (sweepThreads > 1);

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                double offMax = 0.0;
                for (long i = 0; i < np; i++)
                    ring[i] = i;
                for (long round = 0; round < np - 1; round++) {
#ifdef _OPENMP
                    if (parallelSweep) {
    #pragma omp parallel for schedule(static) reduction(max : offMax) num_threads(sweepThreads)
                        for (long i = 0; i < half; i++) {
                            const long a = ring[i], b = ring[np - 1 - i];
                            if (a >= n || b >= n)
                                continue;  // padding seat
                            const double conv = processPair(a < b ? a : b, a < b ? b : a);
                            if (conv > offMax)
                                offMax = conv;
                        }
                    } else
#endif
                    {
                        for (long i = 0; i < half; i++) {
                            const long a = ring[i], b = ring[np - 1 - i];
                            if (a >= n || b >= n)
                                continue;  // padding seat
                            const double conv = processPair(a < b ? a : b, a < b ? b : a);
                            if (conv > offMax)
                                offMax = conv;
                        }
                    }
                    // Rotate every seat but the first, so each column meets a
                    // different partner next round.
                    const long last = ring[np - 1];
                    for (long i = np - 1; i > 1; i--)
                        ring[i] = ring[i - 1];
                    ring[1] = last;
                }
                // The incremental norm update inside processPair is exact in
                // real arithmetic but drifts in floating point over many
                // sweeps. Recomputing once per sweep costs O(mn) against the
                // sweep's O(mn²), so it is free, and it keeps the convergence
                // test honest — dgesvj refreshes for the same reason.
                refreshNorms();
                if (offMax <= eps)
                    break;
            }
            (void)parallelSweep;

            auto w = [&](long i, long j) -> W& { return wt[j * m + i]; };

            // Column norms are the singular values; sort them descending and
            // carry the same permutation through the columns of W and V.
            // S stays REAL — these are magnitudes.
            std::vector<double> sigma(n, 0.0);
            for (long j = 0; j < n; j++) sigma[j] = std::sqrt(sva[j]);
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(
                order.begin(), order.end(), [&](int a, int b) { return sigma[a] > sigma[b]; });

            Matrix<double> S(m, n);
            Matrix<W> Vm(n, n), U(m, m);
            for (int j = 0; j < n; j++) {
                S(j, j) = sigma[order[j]];
                // Vt is stored transposed, so column order[j] of V is a
                // contiguous row of Vt.
                const W* MATRIXCPP_RESTRICT vsrc = vt + (long)order[j] * n;
                for (long i = 0; i < n; i++) Vm(int(i), j) = vsrc[i];
            }

            // Left singular vectors: the normalised columns of W, for every
            // singular value that is numerically non-zero.
            double sTol = double(std::max(m, n)) * eps * (n ? sigma[order[0]] : 0.0);
            int r = 0;
            while (r < n && sigma[order[r]] > sTol) r++;
            for (int j = 0; j < r; j++) {
                double sj = sigma[order[j]];
                for (int i = 0; i < m; i++) U(i, j) = w(i, order[j]) / sj;
            }

            // U must come back m×m orthogonal — UNITARY for a complex matrix —
            // but only r of its columns are determined by A. Fill the rest with
            // any orthonormal completion: push each canonical basis vector
            // through modified Gram-Schmidt and keep the ones with a surviving
            // component. Twice, because one pass loses orthogonality when the
            // residual is small.
            //
            // The projection uses conj(U)·x, the Hermitian inner product. With
            // the plain dot product the completion would not be unitary for a
            // complex matrix, and U^H U == I would fail only for the columns A
            // never determined — which is exactly the kind of bug that hides.
            int filled = r;
            for (int cand = 0; cand < m && filled < m; cand++) {
                std::vector<W> x((std::size_t)m, W(0));
                x[(std::size_t)cand] = W(1);
                for (int pass = 0; pass < 2; pass++)
                    for (int j = 0; j < filled; j++) {
                        W dot = W(0);
                        for (int i = 0; i < m; i++) dot += cj(U(i, j)) * x[(std::size_t)i];
                        for (int i = 0; i < m; i++) x[(std::size_t)i] -= dot * U(i, j);
                    }
                double nx = 0.0;
                for (int i = 0; i < m; i++) nx += magnitudeSq(x[(std::size_t)i]);
                nx = std::sqrt(nx);
                if (nx < 1e-8) continue;  // already spanned; try the next one
                for (int i = 0; i < m; i++) U(i, filled) = x[(std::size_t)i] / nx;
                filled++;
            }

            return std::make_tuple(U, S, Vm);
        } catch (const std::exception& e) {
            std::cerr << "svd() error: " << e.what() << '\n';
            throw;
        }
    }

    // Computes A^(-1) by solving A * X = I. Requires a square, non-singular
    // matrix. The factor-and-substitute machinery this used to inline now lives
    // in solve(), so there is one copy of it to get right rather than two. The
    // flop count is unchanged: both versions factor once and substitute n times.
    //
    // If you are about to write A.inverse() * b, write A.solve(b) instead:
    // fewer flops and better conditioned. See the note on solve().
    Matrix<work_t<datatype>> inverse() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("inverse: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            int n = (int)rowSize;
            Matrix<double> Id(n, n);
            for (int i = 0; i < n; i++)
                Id(i, i) = 1.0;
            return solve(Id);
        } catch (const std::exception& e) {
            std::cerr << "inverse() error: " << e.what() << '\n';
            throw;
        }
    }

    // Solves A * X = B for X. B may be a single column or several at once.
    //
    // Prefer this over A.inverse() * B: about a third of the flops and
    // numerically better conditioned. Forming an explicit inverse just to
    // multiply by it is the classic mistake this method exists to prevent.
    //
    // Square A       → exact solve via LU with partial pivoting.
    // rows > cols    → the least-squares solution argmin ||A*X - B||₂, via the
    //                  column-pivoted QR above: back-substitute R*y = (Qᵀ*B)
    //                  over the numerically non-zero diagonal of R, then undo
    //                  the column permutation. Rank-deficient input gives a
    //                  basic solution (free variables set to zero), not the
    //                  minimum-norm one — use pinv() if you need that.
    // rows < cols    → under-determined; throws, since "the" solution is not
    //                  unique. pinv() gives the minimum-norm one.
    //
    // Templated on B's element type so A.solve(b) works whatever b holds.
    // Usage: auto x = A.solve(b);
    template <typename dtB>
    Matrix<work_t<datatype>> solve(const Matrix<dtB>& B) const {
        using W = work_t<datatype>;
        try {
            int m = (int)rowSize, n = (int)colSize, nrhs = (int)B.cols();
            if (B.rows() != rowSize)
                throw std::invalid_argument(
                    "solve: B must have one row per row of A — A is " + std::to_string(rowSize) +
                    "x" + std::to_string(colSize) + " but B is " + std::to_string(B.rows()) + "x" +
                    std::to_string(B.cols()));
            if (m == 0 || n == 0 || nrhs == 0)
                throw std::invalid_argument("solve: matrices must be non-empty");
            if (m < n)
                throw std::invalid_argument("solve: system is under-determined (" +
                                            std::to_string(m) + " equations, " + std::to_string(n) +
                                            " unknowns) — infinitely many "
                                            "solutions. Use pinv() for the minimum-norm one");

            // ── Square: LU with partial pivoting ────────────────────────────
            if (m == n) {
                if (n == 1) {  // luPacked() requires 2x2; handle the scalar case here
                    W a = W(grid[0]);
                    if (a == W(0))
                        throw std::runtime_error("solve: 1x1 matrix is singular");
                    Matrix<W> X(1, nrhs);
                    for (int j = 0; j < nrhs; j++) X(0, j) = W(B(0, j)) / a;
                    return X;
                }
                auto [packed, pivots] = luPacked();
                // The substitution itself lives in luSubstitute so that this
                // and Decomposition::solve share ONE implementation — the
                // parallel-over-right-hand-sides path in particular is worth
                // having in exactly one place.
                Matrix<W> Bd(n, nrhs);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < nrhs; j++) Bd(i, j) = W(B(i, j));
                return luSubstitute(packed, pivots, n, Bd);
            }

            // ── Over-determined: least squares via column-pivoted QR ────────
            // A*P = Q*R, so ||A x - b|| = ||R (Pᵀx) - Qᵀb||.
            // The least-squares solution needs Q^H*B and R — never Q itself.
            // Applying the reflectors straight to B is O(m·r·nrhs) where
            // forming Q first is O(m²·r); for a tall thin system that is the
            // difference between milliseconds and seconds, and between a few
            // kilobytes and half a gigabyte. This is LAPACK's dormqr.
            QRFactored F = qrFactor();
            Matrix<W> R(m, n);
            for (int i = 0; i < m; i++)
                for (int j = i; j < n; j++) R(i, j) = F.wt[(std::size_t)j * m + i];
            Matrix<W> QtB(m, nrhs);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < nrhs; j++) QtB(i, j) = W(B(i, j));
            applyQH(F, QtB);

            double rTol =
                double(std::max(m, n)) * std::numeric_limits<double>::epsilon() * std::abs(R(0, 0));
            int rk = 0;
            while (rk < n && std::abs(R(rk, rk)) > rTol)
                rk++;

            Matrix<W> Y(n, nrhs);
            for (int col = 0; col < nrhs; col++) {
                // Back-substitute over the leading rk×rk block; the trailing
                // unknowns are the free variables and stay at zero.
                for (int i = rk - 1; i >= 0; i--) {
                    W acc = QtB(i, col);
                    for (int j = i + 1; j < rk; j++) acc -= R(i, j) * Y(j, col);
                    Y(i, col) = acc / R(i, i);
                }
            }
            // x = P*y — undo the column permutation. Straight from the pivot
            // list rather than by building and multiplying an n×n permutation
            // matrix, which was O(n²·nrhs) to move n·nrhs numbers.
            Matrix<W> X(n, nrhs);
            for (int k = 0; k < n; k++)
                for (int col = 0; col < nrhs; col++)
                    X(F.pivots[(std::size_t)k], col) = Y(k, col);
            return X;
        } catch (const std::exception& e) {
            std::cerr << "solve() error: " << e.what() << '\n';
            throw;
        }
    }

    // Moore-Penrose pseudo-inverse. Defined for any m×n, singular matrices
    // included, and coincides with inverse() when A is square and non-singular.
    // From the SVD: pinv(A) = V * S^+ * U^T, where S^+ inverts every singular
    // value above tol and leaves the rest at zero. Requires svd() first.
    // Follows the input type, like svd(): the pseudo-inverse of a complex matrix
    // is complex. Note it is V·S⁺·U^H, the CONJUGATE transpose — with the plain
    // transpose the Moore-Penrose conditions fail for a complex A.
    Matrix<work_t<datatype>> pinv(double tol = -1.0) const {
        using W = work_t<datatype>;
        try {
            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("pinv: matrix must be non-empty");
            auto [U, S, V] = svd();
            int m = (int)rowSize, n = (int)colSize;
            int d = std::min(m, n);

            if (tol < 0.0)
                tol = double(std::max(m, n)) * std::numeric_limits<double>::epsilon() * S(0, 0);

            // pinv(A) = V · S⁺ · Uᵀ, where S⁺ inverts the singular values above
            // tol and leaves the rest at zero. Folding S⁺ into V first keeps
            // this to one matrix product.
            Matrix<W> VS(n, m);
            for (int j = 0; j < d; j++) {
                double sj = S(j, j);
                if (sj <= tol)
                    continue;
                double inv = 1.0 / sj;
                for (int i = 0; i < n; i++) VS(i, j) = V(i, j) * inv;
            }
            return VS * U.H();
        } catch (const std::exception& e) {
            std::cerr << "pinv() error: " << e.what() << '\n';
            throw;
        }
    }

    // Adjugate (classical adjoint): the transpose of the cofactor matrix.
    // Satisfies A * adj(A) = det(A) * I, which is the identity behind the
    // textbook formula inverse(A) = adj(A) / det(A). Requires square A.
    // Mainly of symbolic/theoretical interest — inverse() and solve() are the
    // right tools numerically, since the cofactor route is O(n!) if done naively.
    Matrix<double> adjugate() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("adjugate: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("adjugate: matrix must be non-empty");
            int n = (int)rowSize;

            // adj of a 1x1 is [1] by convention: A·adj(A) = det(A)·I = a·I.
            if (n == 1) {
                Matrix<double> out(1, 1);
                out(0, 0) = 1.0;
                return out;
            }

            Matrix<double> Ad(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Ad(i, j) = double(std::real(grid[i * n + j]));

            // Non-singular: adj(A) = det(A)·A⁻¹ directly from the identity
            // A·adj(A) = det(A)·I. Two O(n³) steps instead of n² minors.
            double scale = norm(NormType::Inf);
            double dTol = std::numeric_limits<double>::epsilon() *
                          std::pow(scale > 0.0 ? scale : 1.0, n) * 100.0;
            double d = Ad.det();
            if (std::abs(d) > dTol)
                return Ad.inverse() * d;

            // Singular: the identity above says nothing, so fall back to the
            // definition — adj(A)[j,i] = (-1)^(i+j) · det(A with row i, col j
            // deleted). O(n⁵), but it is the only route that stays correct here.
            Matrix<double> out(n, n);
            Matrix<double> minor(n - 1, n - 1);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    for (int r = 0, mr = 0; r < n; r++) {
                        if (r == i)
                            continue;
                        for (int c = 0, mc = 0; c < n; c++) {
                            if (c == j)
                                continue;
                            minor(mr, mc) = Ad(r, c);
                            mc++;
                        }
                        mr++;
                    }
                    double md = (n == 2) ? minor(0, 0) : minor.det();
                    out(j, i) = ((i + j) % 2 ? -1.0 : 1.0) * md;
                }
            return out;
        } catch (const std::exception& e) {
            std::cerr << "adjugate() error: " << e.what() << '\n';
            throw;
        }
    }

    // deconstructor
    ~Matrix() { release(); }

  private:
    // Lets Matrix<double> reach into Matrix<int>'s internals and vice versa,
    // which real()/imag() and the mixed-type paths need.
    template <typename>
    friend class Matrix;
    // Decomposition drives luSubstitute/luSubstituteT and luPacked directly:
    // it IS the factor-once path, so it works at the same level solve() does.
    template <typename>
    friend class Decomposition;

    // Tag type selecting the constructor below.
    struct uninit_t {};

    // Allocates without value-initialising. `new T[n]()` zero-fills, which is
    // pure waste when the very next thing the caller does is overwrite every
    // element — about 16% of the cost of an element-wise operation at
    // n = 2000. Only ever use this when the buffer is fully written before
    // it can be read.
    Matrix(long i, long j, uninit_t) {
        rowSize = i;
        colSize = j;
        allocRaw(i * j);  // deliberately not zeroed
    }

    // ── Small-buffer storage ────────────────────────────────────────
    // Matrices of up to SBO_CAPACITY elements live inside the object; only
    // larger ones touch the heap. Measured on this machine, a 2x2 A+B cost
    // 19 ns of which 19 ns was new/delete — the arithmetic was free and the
    // allocator was the entire operation. Eigen solves this with fixed-size
    // types; this is the runtime equivalent, and it costs one branch on the
    // sizing path.
    //
    // 16 elements covers every matrix up to 4x4, which is where small-matrix
    // work actually concentrates: 2x2 and 4x4 gates for the quantum-circuit
    // goal, 3x3 and 4x4 for geometry.
    static constexpr long SBO_CAPACITY = 16;

    // True when grid points at the inline buffer rather than the heap.
    // Compared against the pointer rather than recomputed from the sizes, so
    // it stays correct even while the sizes are mid-update.
    bool isInline() const { return grid == sbo; }

    // Points grid at storage for n elements WITHOUT initialising it.
    void allocRaw(long n) {
        if (n <= SBO_CAPACITY) {
            grid = sbo;
            return;
        }
        grid = rawAlloc(n);
    }

    // Points grid at zero-initialised storage for n elements.
    void allocZero(long n) {
        if (n <= SBO_CAPACITY) {
            grid = sbo;
            for (long i = 0; i < n; i++)
                grid[i] = datatype();
        } else {
            grid = rawAlloc(n);
            for (long i = 0; i < n; i++)
                grid[i] = datatype();
        }
    }

    // Storage, huge-page advice and the parallel helpers all live in
    // namespace mstore at the top of this file, so that Tensor is built on
    // exactly the same ones. These are the in-class spellings; the comments
    // explaining WHY each exists are on the definitions there.
    static constexpr bool RAW_STORAGE_OK = mstore::raw_storage_ok<datatype>;
    static datatype* rawAlloc(long n) { return mstore::rawAlloc<datatype>(n); }
    static void rawFree(datatype* p) { mstore::rawFree<datatype>(p); }
    static void adviseHuge(void* p, std::size_t b) { mstore::adviseHuge(p, b); }

    // Frees heap storage if that is what grid points at. Safe to call twice.
    void release() {
        if (grid && grid != sbo)
            rawFree(grid);
        grid = nullptr;
    }

    long rowSize;
    long colSize;
    datatype* grid;
    datatype sbo[SBO_CAPACITY];

    // True when rows i..i+1 of a real Schur form T are a genuine 2x2 block —
    // the signature of a complex-conjugate eigenvalue pair — rather than two
    // separate real eigenvalues. The QR iteration drives converged
    // sub-diagonal entries towards zero without always setting them exactly
    // to zero, so the test has to be relative to the neighbouring diagonal.
    static bool isSchurBlock(const std::vector<double>& T, int n, int i) {
        double sub = std::abs(T[(i + 1) * n + i]);
        double nbr = std::abs(T[i * n + i]) + std::abs(T[(i + 1) * n + (i + 1)]);
        return sub > std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0);
    }

    // Shared guard for the reductions, which have no meaningful answer on an
    // empty matrix and would otherwise read grid[0] off a null pointer.
    void requireNonEmpty(const char* who) const {
        if (rowSize * colSize == 0)
            throw std::invalid_argument(std::string(who) + "(): matrix is empty (" +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize) +
                                        ")");
    }

    // ── Mask machinery ─────────────────────────────────────────────────
    // (CmpOp, LogOp and mask_scalar_t are declared just above the public
    // mask API — a member TYPE has to be declared before a member
    // declaration can name it, unlike a member function body.)

    void requireMaskShape(const Matrix<bool>& mask) const {
        if (mask.rows() != rowSize || mask.cols() != colSize)
            throw std::invalid_argument(
                "logical index: the mask is (" + std::to_string(mask.rows()) + "x" +
                std::to_string(mask.cols()) + ") but the matrix is (" + std::to_string(rowSize) +
                "x" + std::to_string(colSize) + ")");
    }

    // Ordering needs <, which std::complex deliberately does not provide.
    // Instantiating one on a complex matrix is a compile-time error rather
    // than a silent choice of some arbitrary ordering — the same rule
    // min()/max()/argmin()/argmax() already follow. eq() and ne() are fine
    // for complex and do not go through this check.
    static void requireOrdered() {
        static_assert(!is_complex<datatype>::value,
                      "lt/gt/le/ge (and <, >, <=, >=) need an ordering, which std::complex\n"
                      "deliberately does not provide. Compare a component or a magnitude\n"
                      "instead — for example A.real().gt(0.0) or A.abs().gt(1.0).\n"
                      "eq() and ne() DO work for complex.");
    }

    template <typename A, typename B>
    static bool applyCmp(CmpOp op, const A& x, const B& y) {
        switch (op) {
            case CmpOp::LT:
                return x < y;
            case CmpOp::GT:
                return x > y;
            case CmpOp::LE:
                return x <= y;
            case CmpOp::GE:
                return x >= y;
            case CmpOp::EQ:
                return x == y;
            default:
                return !(x == y);
        }
    }

    Matrix<bool> compare(const Matrix& M, const char* who, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE)
            requireOrdered();
        requireSameShape(M, who);
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = applyCmp(op, a[i], b[i]); });
        return out;
    }

    template <typename Scalar>
    Matrix<bool> compareScalar(const Scalar& v, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE)
            requireOrdered();
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype rhs = datatype(v);
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = applyCmp(op, a[i], rhs); });
        return out;
    }

    Matrix<bool> logical(const Matrix& M, const char* who, LogOp op) const {
        requireSameShape(M, who);
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype zero = datatype(0);
        forEachIndex(rowSize * colSize, [=](long i) {
            const bool x = !(a[i] == zero), y = !(b[i] == zero);
            r[i] = (op == LogOp::AND) ? (x && y) : (op == LogOp::OR) ? (x || y) : (x != y);
        });
        return out;
    }

    // Per-axis any()/all(). axis=0 walks columns, axis=1 walks rows,
    // matching sum(bool).
    Matrix<bool> reduceLogical(bool axis, bool wantAny) const {
        const datatype zero = datatype(0);
        if (!axis) {
            Matrix<bool> out(1, colSize);
            for (long j = 0; j < colSize; j++) {
                bool acc = !wantAny;
                for (long i = 0; i < rowSize; i++) {
                    const bool v = !(grid[i * colSize + j] == zero);
                    if (wantAny) {
                        if (v) {
                            acc = true;
                            break;
                        }
                    } else {
                        if (!v) {
                            acc = false;
                            break;
                        }
                    }
                }
                out(0, int(j)) = acc;
            }
            return out;
        }
        Matrix<bool> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            bool acc = !wantAny;
            for (long j = 0; j < colSize; j++) {
                const bool v = !(grid[i * colSize + j] == zero);
                if (wantAny) {
                    if (v) {
                        acc = true;
                        break;
                    }
                } else {
                    if (!v) {
                        acc = false;
                        break;
                    }
                }
            }
            out(int(i), 0) = acc;
        }
        return out;
    }

    // Default tolerance for the structure predicates: relative to the
    // largest row sum, so it scales with the matrix rather than assuming
    // entries are around 1.
    double structureTol(double tol) const {
        if (tol >= 0.0)
            return tol;
        const double scale = (rowSize * colSize > 0) ? norm(NormType::Inf) : 1.0;
        return std::numeric_limits<double>::epsilon() * 100.0 * (scale > 0.0 ? scale : 1.0);
    }

    // Default cut-off for "this singular value is numerically zero", the
    // same rule rank() uses: max(m,n) * eps * sigma_max.
    double svdTol(const Matrix<double>& S, double tol) const {
        if (tol >= 0.0)
            return tol;
        const double smax = (rowSize > 0 && colSize > 0) ? S(0, 0) : 0.0;
        return double(std::max(rowSize, colSize)) * std::numeric_limits<double>::epsilon() * smax;
    }

    // The rounding and real-valued maps have no complex meaning — there is no
    // "largest integer below" a complex number — so instantiating one is a
    // compile-time error rather than a silent choice. sign() and angle() DO
    // have complex meanings and do not go through this.
    static void requireRoundable(const char* who) {
        (void)who;
        static_assert(!is_complex<datatype>::value,
                      "floor/ceil/round/fix/mod/rem/atan2/hypot/expm1/log1p have no complex\n"
                      "meaning. Apply them to .real() / .imag() / .abs(), or use sign() and\n"
                      "angle(), which are defined for complex.");
    }

    // Two-operand element-wise map, the binary sibling of mapElems.
    template <class F>
    Matrix zipElems(const Matrix& B, F fn) const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = B.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        const long total = rowSize * colSize;
#ifdef _OPENMP
        if (total >= MAP_MIN_WORK) {
    #pragma omp parallel for schedule(static)
            for (long i = 0; i < total; i++)
                r[i] = fn(a[i], b[i]);
            return ans;
        }
#endif
        for (long i = 0; i < total; i++)
            r[i] = fn(a[i], b[i]);
        return ans;
    }

    // Solves L*U*x = P*b for every column of B, given the packed factors
    // luPacked() produces. Extracted from solve() so that solve(),
    // inverse() and Decomposition all drive the same code.
    static Matrix<work_t<datatype>> luSubstitute(
        const std::vector<work_t<datatype>>& packed, const std::vector<int>& pivots, int n,
        const Matrix<work_t<datatype>>& B) {
        using W = work_t<datatype>;
        const int nrhs = (int)B.cols();
        Matrix<W> X(n, nrhs);
        const W* MATRIXCPP_RESTRICT LU = packed.data();
        W* MATRIXCPP_RESTRICT Xg = X.grid;
        const W* MATRIXCPP_RESTRICT Bg = B.grid;

        // One right-hand side: permute, forward-substitute through L,
        // back-substitute through U. Both substitutions walk a row of the
        // packed factor, which is contiguous, and four accumulators keep the
        // dot product at FMA throughput rather than FMA latency.
        auto solveOne = [&](int col) {
            std::vector<W> bv((std::size_t)n);
            W* MATRIXCPP_RESTRICT b = bv.data();
            for (int i = 0; i < n; i++)
                b[i] = Bg[(std::size_t)i * nrhs + col];
            for (int i = 0; i < n; i++)
                if (pivots[(std::size_t)i] != i)
                    std::swap(b[i], b[pivots[(std::size_t)i]]);
            for (int i = 0; i < n; i++) {
                const W* MATRIXCPP_RESTRICT row = LU + (std::size_t)i * n;
                W a0 = W(0), a1 = W(0), a2 = W(0), a3 = W(0);
                int j = 0;
                for (; j + 3 < i; j += 4) {
                    a0 += row[j] * b[j];
                    a1 += row[j + 1] * b[j + 1];
                    a2 += row[j + 2] * b[j + 2];
                    a3 += row[j + 3] * b[j + 3];
                }
                for (; j < i; j++)
                    a0 += row[j] * b[j];
                b[i] -= (a0 + a1) + (a2 + a3);
            }
            for (int i = n - 1; i >= 0; i--) {
                const W* MATRIXCPP_RESTRICT row = LU + (std::size_t)i * n;
                W a0 = W(0), a1 = W(0), a2 = W(0), a3 = W(0);
                int j = i + 1;
                for (; j + 3 < n; j += 4) {
                    a0 += row[j] * b[j];
                    a1 += row[j + 1] * b[j + 1];
                    a2 += row[j + 2] * b[j + 2];
                    a3 += row[j + 3] * b[j + 3];
                }
                for (; j < n; j++)
                    a0 += row[j] * b[j];
                b[i] = (b[i] - ((a0 + a1) + (a2 + a3))) / row[i];
            }
            for (int i = 0; i < n; i++)
                Xg[(std::size_t)i * nrhs + col] = b[i];
        };

        // Every right-hand side is solved against the SAME factors and writes
        // a different column of X, so the columns are independent. LAPACK
        // exploits this differently — dgetrs hands all of them to dtrsm at
        // once for a level-3 blocked triangular solve — but the parallelism
        // is the same, and it is what makes inverse() (solve against n
        // columns of the identity) worth anything: one column at a time, a
        // 512x512 inverse cost 83 ms against 7.6 ms for a single solve.
        const long solveWork = (long)nrhs * n * n;
        (void)solveWork;  // only read on the OpenMP path
#ifdef _OPENMP
        if (solveWork >= PARALLEL_MIN_WORK && nrhs > 1) {
    #pragma omp parallel for schedule(static)
            for (int col = 0; col < nrhs; col++)
                solveOne(col);
            return X;
        }
#endif
        for (int col = 0; col < nrhs; col++)
            solveOne(col);
        return X;
    }

    // Solves the TRANSPOSED system, Uᵀ Lᵀ y = b then undoes the permutation.
    // Needed by the condition estimator, which alternates between A and Aᵀ.
    static Matrix<work_t<datatype>> luSubstituteT(
        const std::vector<work_t<datatype>>& packed, const std::vector<int>& pivots, int n,
        const Matrix<work_t<datatype>>& B) {
        using W = work_t<datatype>;
        const int nrhs = (int)B.cols();
        Matrix<W> X(n, nrhs);
        const W* LU = packed.data();
        for (int col = 0; col < nrhs; col++) {
            std::vector<W> b((std::size_t)n);
            for (int i = 0; i < n; i++)
                b[(std::size_t)i] = B(i, col);
            // Uᵀ is lower triangular with U's diagonal.
            for (int i = 0; i < n; i++) {
                W acc = b[(std::size_t)i];
                for (int j = 0; j < i; j++)
                    acc -= LU[(std::size_t)j * n + i] * b[(std::size_t)j];
                b[(std::size_t)i] = acc / LU[(std::size_t)i * n + i];
            }
            // Lᵀ is upper triangular with a unit diagonal.
            for (int i = n - 1; i >= 0; i--) {
                W acc = b[(std::size_t)i];
                for (int j = i + 1; j < n; j++)
                    acc -= LU[(std::size_t)j * n + i] * b[(std::size_t)j];
                b[(std::size_t)i] = acc;
            }
            // The row swaps were applied to b in the forward direction, so
            // they come off the answer in reverse.
            for (int i = n - 1; i >= 0; i--)
                if (pivots[(std::size_t)i] != i)
                    std::swap(b[(std::size_t)i], b[(std::size_t)pivots[(std::size_t)i]]);
            for (int i = 0; i < n; i++)
                X(i, col) = b[(std::size_t)i];
        }
        return X;
    }

    // Shared body of cumsum and cumprod. The scan runs along the axis the
    // matching reduction would collapse, so cumsum(false) accumulates down
    // each column exactly as sum(false) totals it.
    Matrix scan(bool axis, bool adding) const {
        Matrix out(rowSize, colSize, uninit_t{});
        if (rowSize == 0 || colSize == 0)
            return out;
        if (!axis) {
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype prev = out.grid[(i - 1) * colSize + j];
                    const datatype here = grid[i * colSize + j];
                    out.grid[i * colSize + j] = adding ? prev + here : prev * here;
                }
            return out;
        }
        for (long i = 0; i < rowSize; i++) {
            out.grid[i * colSize] = grid[i * colSize];
            for (long j = 1; j < colSize; j++) {
                const datatype prev = out.grid[i * colSize + j - 1];
                const datatype here = grid[i * colSize + j];
                out.grid[i * colSize + j] = adding ? prev + here : prev * here;
            }
        }
        return out;
    }

    // nth_element twice rather than a full sort: O(n) instead of O(n log n),
    // and the second call only has to look at the upper half.
    static double medianOf(std::vector<datatype>& buf) {
        const std::size_t n = buf.size();
        const std::size_t mid = n / 2;
        std::nth_element(buf.begin(), buf.begin() + (long)mid, buf.end());
        const double hi = double(buf[mid]);
        if (n % 2 == 1)
            return hi;
        const double lo = double(*std::max_element(buf.begin(), buf.begin() + (long)mid));
        return 0.5 * (lo + hi);
    }

    // Adds a constant to every element — the body behind A + scalar.
    Matrix offsetBy(datatype d) const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = a[i] + d; });
        return ans;
    }

    // Shared dimension check for the element-wise operations.
    void requireSameShape(const Matrix& M, const char* who) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            throw std::invalid_argument(std::string(who) +
                                        "(): element-wise operations need identical "
                                        "shapes, got (" +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize) +
                                        ") and (" + std::to_string(M.rowSize) + "x" +
                                        std::to_string(M.colSize) + ")");
    }

    // Computes the real Schur decomposition of this matrix.
    // Returns {T_flat, Q_flat} where A = Q * T * Q^T,
    // T is upper (quasi-)triangular and Q is orthogonal (both n×n, row-major
    // double). Public so that the free pow() function can access it; also useful
    // on its own.
    //
    // NOTE — "quasi-triangular" is the word carrying all the weight here.
    // T is triangular except for 2x2 blocks on the diagonal, one per pair of
    // complex-conjugate eigenvalues. Every caller in this header currently
    // assumes those blocks do not exist: eig() reads the bare diagonal, and
    // pow()/log() sidestep the issue by demanding positive eigenvalues.
    // Any block-aware consumer (a correct eig, exp(A), the matrix trig
    // functions) needs a Parlett recurrence that solves a small Sylvester
    // equation per block rather than dividing scalars. This is the single
    // change that unblocks the most of the roadmap at the top of the file.
  public:
    // ── Complex Schur: A = Q T Q^H with T UPPER TRIANGULAR ───────────────
    //
    // The REAL Schur form cannot represent a complex matrix. It parks a
    // conjugate pair in a 2x2 diagonal block, and that trick works only because
    // a real matrix's complex eigenvalues necessarily come in conjugate pairs.
    // A complex matrix has no such symmetry, so it needs the genuine complex
    // Schur form — where T comes out FULLY TRIANGULAR, eigenvalues on the
    // diagonal, no blocks to special-case anywhere downstream.
    //
    // Complex Hessenberg reduction, then a single-shift QR iteration by Givens
    // rotations. schurDecomp() needs a DOUBLE shift to keep a conjugate pair
    // inside real arithmetic; in complex arithmetic one Wilkinson shift does it,
    // which is why this is shorter than the real routine despite being strictly
    // more general. Everything the real one earned is kept: Givens rather than
    // Householder in the sweep (exploiting Hessenberg structure), a relative
    // deflation test, and an exceptional shift to break cycling.
    std::pair<std::vector<work_t<datatype>>, std::vector<work_t<datatype>>>
    schurDecompComplex() const {
        using C = work_t<datatype>;
        auto cj = [](const C& v) {
            if constexpr (is_complex<datatype>::value) return std::conj(v);
            else return v;
        };
        const int n = (int)rowSize;
        std::vector<C> H((std::size_t)n * n), Q((std::size_t)n * n, C(0));
        for (int k = 0; k < n * n; k++) H[(std::size_t)k] = C(grid[(std::size_t)k]);
        for (int i = 0; i < n; i++) Q[(std::size_t)i * n + i] = C(1);
        auto h  = [&](int i, int j) -> C& { return H[(std::size_t)i * n + j]; };
        auto qv = [&](int i, int j) -> C& { return Q[(std::size_t)i * n + j]; };

        // [ c         s ] [f]   [r]      c real and >= 0,  c^2 + |s|^2 = 1
        // [ -conj(s)  c ] [g] = [0]
        auto givens = [&](const C& f, const C& g, double& c, C& sv) {
            const double af = std::abs(f), ag = std::abs(g);
            if (ag == 0.0) { c = 1.0; sv = C(0); return; }
            if (af == 0.0) { c = 0.0; sv = C(1); return; }
            const double hh = std::hypot(af, ag);
            c = af / hh;
            // conj(g)*(f/|f|)/h, not the textbook c*conj(g/f): algebraically
            // identical, but it never divides by a tiny f.
            sv = cj(g) * (f / af) / hh;
        };

        // ── Complex Hessenberg reduction ──────────────────────────────────
        for (int k = 0; k < n - 2; k++) {
            double xn = 0.0;
            for (int i = k + 1; i < n; i++) xn += std::norm(h(i, k));
            xn = std::sqrt(xn);
            if (xn == 0.0) continue;
            const int sz = n - k - 1;
            // Opposite in PHASE, not in sign — the real rule does not carry over.
            const C hk = h(k + 1, k);
            const double a0 = std::abs(hk);
            const C alpha = -((a0 == 0.0) ? C(1) : hk / a0) * xn;
            std::vector<C> v((std::size_t)sz);
            for (int i = 0; i < sz; i++) v[(std::size_t)i] = h(k + 1 + i, k);
            v[0] -= alpha;
            double vtv = 0.0;
            for (int i = 0; i < sz; i++) vtv += std::norm(v[(std::size_t)i]);
            if (vtv == 0.0) continue;
            const double tau = 2.0 / vtv;
            for (int j = 0; j < n; j++) {                 // H <- (I - t v v^H) H
                C d = C(0);
                for (int i = 0; i < sz; i++) d += cj(v[(std::size_t)i]) * h(k + 1 + i, j);
                const C f = tau * d;
                for (int i = 0; i < sz; i++) h(k + 1 + i, j) -= f * v[(std::size_t)i];
            }
            for (int i = 0; i < n; i++) {                 // H <- H (I - t v v^H)
                C d = C(0);
                for (int j = 0; j < sz; j++) d += h(i, k + 1 + j) * v[(std::size_t)j];
                const C f = tau * d;
                for (int j = 0; j < sz; j++) h(i, k + 1 + j) -= f * cj(v[(std::size_t)j]);
            }
            for (int i = 0; i < n; i++) {                 // Q <- Q (I - t v v^H)
                C d = C(0);
                for (int j = 0; j < sz; j++) d += qv(i, k + 1 + j) * v[(std::size_t)j];
                const C f = tau * d;
                for (int j = 0; j < sz; j++) qv(i, k + 1 + j) -= f * cj(v[(std::size_t)j]);
            }
        }
        for (int i = 2; i < n; i++)
            for (int j = 0; j <= i - 2; j++) h(i, j) = C(0);

        // ── Single-shift QR iteration ─────────────────────────────────────
        const double eps = std::numeric_limits<double>::epsilon();
        int hi = n - 1, iter = 0;
        while (hi > 0) {
            // Deflate on a RELATIVE test. An absolute one is the bug that made
            // funm read a 2.8e-23 residual as a genuine coupling.
            int lo = hi;
            while (lo > 0) {
                double sc = std::abs(h(lo - 1, lo - 1)) + std::abs(h(lo, lo));
                if (sc == 0.0) sc = 1.0;
                if (std::abs(h(lo, lo - 1)) <= eps * sc) { h(lo, lo - 1) = C(0); break; }
                lo--;
            }
            if (lo == hi) { hi--; iter = 0; continue; }   // 1x1 converged
            if (++iter > 120)
                throw std::runtime_error(
                    "schurDecompComplex: QR iteration failed to converge at index " +
                    std::to_string(hi));

            C sh;
            if (iter % 15 == 0) {
                sh = h(hi, hi) + C(std::abs(h(hi, hi - 1)));      // exceptional
            } else {
                const C a = h(hi - 1, hi - 1), b = h(hi - 1, hi);
                const C c2 = h(hi, hi - 1), d = h(hi, hi);
                const C tr = a + d, det = a * d - b * c2;
                const C disc = std::sqrt(tr * tr - 4.0 * det);
                const C l1 = (tr + disc) * 0.5, l2 = (tr - disc) * 0.5;
                sh = (std::abs(l1 - d) < std::abs(l2 - d)) ? l1 : l2;   // Wilkinson
            }

            for (int i = lo; i <= hi; i++) h(i, i) -= sh;
            std::vector<double> cs((std::size_t)(hi - lo));
            std::vector<C> sn((std::size_t)(hi - lo));
            for (int i = lo; i < hi; i++) {               // H - sI  ->  R
                double cc; C ss;
                givens(h(i, i), h(i + 1, i), cc, ss);
                cs[(std::size_t)(i - lo)] = cc; sn[(std::size_t)(i - lo)] = ss;
                for (int j = i; j < n; j++) {
                    const C t1 = h(i, j), t2 = h(i + 1, j);
                    h(i, j)     = cc * t1 + ss * t2;
                    h(i + 1, j) = -cj(ss) * t1 + cc * t2;
                }
            }
            for (int i = lo; i < hi; i++) {               // R -> R G^H, and Q G^H
                const double cc = cs[(std::size_t)(i - lo)];
                const C ss = sn[(std::size_t)(i - lo)];
                for (int j = 0; j <= hi; j++) {
                    const C t1 = h(j, i), t2 = h(j, i + 1);
                    h(j, i)     = cc * t1 + cj(ss) * t2;
                    h(j, i + 1) = -ss * t1 + cc * t2;
                }
                for (int j = 0; j < n; j++) {
                    const C t1 = qv(j, i), t2 = qv(j, i + 1);
                    qv(j, i)     = cc * t1 + cj(ss) * t2;
                    qv(j, i + 1) = -ss * t1 + cc * t2;
                }
            }
            for (int i = lo; i <= hi; i++) h(i, i) += sh;
        }
        for (int i = 1; i < n; i++)                       // clear numerical dust
            for (int j = 0; j < i; j++) h(i, j) = C(0);
        return {H, Q};
    }

    std::pair<std::vector<double>, std::vector<double>> schurDecomp() const {
        int n = (int)rowSize;
        std::vector<double> H(n * n), Q(n * n, 0.0);
        for (int k = 0; k < n * n; k++)
            H[k] = double(grid[k]);
        for (int i = 0; i < n; i++)
            Q[i * n + i] = 1.0;
        auto h = [&](int i, int j) -> double& { return H[i * n + j]; };
        auto qv = [&](int i, int j) -> double& { return Q[i * n + j]; };

        // ── Hessenberg reduction ──────────────────────────────────────────
        for (int k = 0; k < n - 2; k++) {
            double xn = 0.0;
            for (int i = k + 1; i < n; i++)
                xn += h(i, k) * h(i, k);
            xn = std::sqrt(xn);
            if (xn < 1e-14)
                continue;
            int sz = n - k - 1;
            double alpha = (h(k + 1, k) >= 0.0 ? -1.0 : 1.0) * xn;
            std::vector<double> v(sz);
            for (int i = 0; i < sz; i++)
                v[i] = h(k + 1 + i, k);
            v[0] -= alpha;
            double vv = 0.0;
            for (double vi : v)
                vv += vi * vi;
            if (vv < 1e-28)
                continue;
            double tau = 2.0 / vv;
            for (int j = k; j < n; j++) {
                double s = 0.0;
                for (int i = 0; i < sz; i++)
                    s += v[i] * h(k + 1 + i, j);
                for (int i = 0; i < sz; i++)
                    h(k + 1 + i, j) -= tau * v[i] * s;
            }
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = 0; j < sz; j++)
                    s += h(i, k + 1 + j) * v[j];
                for (int j = 0; j < sz; j++)
                    h(i, k + 1 + j) -= tau * v[j] * s;
            }
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = 0; j < sz; j++)
                    s += qv(i, k + 1 + j) * v[j];
                for (int j = 0; j < sz; j++)
                    qv(i, k + 1 + j) -= tau * v[j] * s;
            }
        }

        // ── QR iteration with Wilkinson shift and deflation ───────────────
        // Two things this loop has to handle that a plain "iterate until the
        // sub-diagonal vanishes" version does not:
        //
        //  1. A complex-conjugate eigenvalue pair NEVER drives its
        //     sub-diagonal entry to zero. That is not a convergence failure,
        //     it is the definition of a real Schur form: the pair lives in a
        //     2x2 block. So once a trailing 2x2 is isolated it is deflated as
        //     a block rather than iterated on. Without this, a plain rotation
        //     matrix spins until the step limit and throws.
        //  2. An isolated 2x2 whose eigenvalues turn out to be REAL is split
        //     by one Givens rotation, so it does not then masquerade as a
        //     complex pair to eig()/eigvals() downstream.
        //
        // A block that stalls also gets a periodic exceptional shift — the
        // standard escape from the matrices a Wilkinson shift cycles on.
        const double eps = std::numeric_limits<double>::epsilon();

        // The QR iteration below updates COLUMNS k and k+1 of Q on every
        // rotation. In row-major storage a column walk strides by a whole row,
        // so that is one cache miss per row, n of them per rotation — and it
        // was the dominant cost of the whole routine once the O(n^4) work was
        // gone. Transposing Q once here turns each of those column updates
        // into two contiguous row updates; it is transposed back at the end.
        std::vector<double> Qt(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Qt[j * n + i] = Q[i * n + j];

        // Relative test for "this sub-diagonal entry has converged to zero".
        auto negligible = [&](int i) {
            double nbr = std::abs(h(i - 1, i - 1)) + std::abs(h(i, i));
            if (nbr == 0.0)
                nbr = 1.0;
            return std::abs(h(i, i - 1)) <= eps * nbr;
        };

        // Triangularises the 2x2 block at rows/cols k, k+1 if its eigenvalues
        // are real; leaves it intact if they are a conjugate pair. The
        // rotation is chosen so its first column is an eigenvector, which
        // sends the sub-diagonal entry to exactly zero.
        auto split2x2 = [&](int k) {
            // Standardise a real 2x2 diagonal block — LAPACK's dlanv2.
            //
            // The version this replaced computed the eigenvalue from
            //     disc = ((a+d)/2)^2 - (a*d - b*c)
            // and then FORCED h(k+1,k) = 0 afterwards. Both halves were wrong:
            // that discriminant cancels catastrophically when a and d are close
            // (for a = d = 1e3, b = c = 1e-3 the true 1e-6 is the difference of
            // two numbers near 1e6, keeping four digits), so the rotation built
            // from it did not actually zero the sub-diagonal — and forcing the
            // zero anyway broke the similarity silently. Measured across a
            // single call: ||Q^T A Q - H|| went 3.3e-16 -> 4.8e-10, compounding
            // to 3e-2 on a 19x19. It stayed hidden until the Francis shift
            // started handing this routine blocks the old iteration never
            // reached.
            //
            // dlanv2 avoids all of it: the discriminant is formed as
            // p^2 + b*c with p = (a-d)/2 (no cancellation), everything is
            // scaled against overflow, the SECOND eigenvalue comes from the
            // product rather than the other root, and the standardised block is
            // written out directly instead of being hoped for.
            double a = h(k, k), b = h(k, k + 1);
            double c = h(k + 1, k), d = h(k + 1, k + 1);
            if (c == 0.0)
                return;
            double cs, sn;
            if (b == 0.0) {
                // Swap the two, so the block comes out upper triangular.
                cs = 0.0;
                sn = 1.0;
                std::swap(a, d);
                b = -c;
                c = 0.0;
            } else {
                const double p = 0.5 * (a - d);
                const double bcmax = std::max(std::abs(b), std::abs(c));
                const double bcmis = std::min(std::abs(b), std::abs(c)) *
                                     ((b >= 0.0) ? 1.0 : -1.0) * ((c >= 0.0) ? 1.0 : -1.0);
                const double scale = std::max(std::abs(p), bcmax);
                if (scale == 0.0)
                    return;
                double z = (p / scale) * p + (bcmax / scale) * bcmis;
                if (z < 4.0 * eps)
                    return;  // a genuine complex pair — the block stays
                // Real, well separated eigenvalues.
                z = p + ((p >= 0.0) ? 1.0 : -1.0) * std::sqrt(scale) * std::sqrt(z);
                a = d + z;
                d -= (bcmax / z) * bcmis;   // from the PRODUCT, not the other root
                const double tau = std::hypot(c, z);
                if (tau == 0.0)
                    return;
                cs = z / tau;
                sn = c / tau;
                b -= c;
                c = 0.0;
            }
            for (int j = 0; j < n; j++) {  // rows: H <- G^T H
                double t1 = h(k, j), t2 = h(k + 1, j);
                h(k, j) = cs * t1 + sn * t2;
                h(k + 1, j) = -sn * t1 + cs * t2;
            }
            for (int i = 0; i < n; i++) {  // cols: H <- H G
                double t1 = h(i, k), t2 = h(i, k + 1);
                h(i, k) = cs * t1 + sn * t2;
                h(i, k + 1) = -sn * t1 + cs * t2;
            }
            {  // accumulate: Q <- Q G, on Qt
                double* qk = &Qt[k * n];
                double* qn = &Qt[(k + 1) * n];
                for (int i = 0; i < n; i++) {
                    double t1 = qk[i], t2 = qn[i];
                    qk[i] = cs * t1 + sn * t2;
                    qn[i] = -sn * t1 + cs * t2;
                }
            }
            // Write the standardised block. These are the accurately computed
            // values, not whatever the rotation happened to leave behind.
            h(k, k) = a;
            h(k, k + 1) = b;
            h(k + 1, k) = c;
            h(k + 1, k + 1) = d;
        };

        int maxSteps = 60 * n, ihi = n - 1, itersOnBlock = 0;
        while (ihi >= 1) {
            int ilo = ihi;
            while (ilo > 0 && !negligible(ilo))
                ilo--;

            if (ilo == ihi) {  // isolated 1x1: real eigenvalue
                if (ihi > 0)
                    h(ihi, ihi - 1) = 0.0;
                ihi--;
                itersOnBlock = 0;
                continue;
            }
            if (ilo == ihi - 1) {  // isolated 2x2: deflate as a block
                if (ilo > 0)
                    h(ilo, ilo - 1) = 0.0;
                split2x2(ilo);
                ihi -= 2;
                itersOnBlock = 0;
                continue;
            }
            if (maxSteps-- < 0)
                throw std::runtime_error("schurDecomp: QR iteration did not converge after " +
                                         std::to_string(60 * n) + " steps");

            // ── Francis DOUBLE-shift implicit QR step (LAPACK's dlahqr) ──
            //
            // The single real shift this replaced COULD NOT CONVERGE ON A
            // CONJUGATE PAIR. When the trailing 2x2 had complex eigenvalues it
            // fell back to sigma = d, a real shift, and no real shift converges
            // to a complex pair. The routine therefore threw on ordinary
            // matrices, intermittently and with no dependence on size — a plain
            // random 16x16 failed while a 64x64 succeeded, and a near-identity
            // 64x64 failed while its 32x32 sibling did not. Every user-facing
            // consumer went down with it: eig, schur, funm, sqrt(A), log(A),
            // pow(A,real).
            //
            // The Francis step applies BOTH roots of the trailing quadratic at
            // once while staying in real arithmetic. It never forms them: it
            // builds only the first column of
            //     (H - l1 I)(H - l2 I) = H^2 - sH + tI,   s = l1+l2, t = l1*l2
            // and Hessenberg structure leaves that column with just THREE
            // nonzero entries. So the step is one 3x3 reflector at the top of
            // the block, then the resulting bulge chased down the diagonal by
            // further 3x3 reflectors — O(n) work per column, O(n^2) per sweep,
            // exactly as the Givens version was.
            double s = h(ihi - 1, ihi - 1) + h(ihi, ihi);              // trace
            double t = h(ihi - 1, ihi - 1) * h(ihi, ihi) -
                       h(ihi - 1, ihi) * h(ihi, ihi - 1);              // determinant
            // Exceptional shift every 10 steps — the standard escape from the
            // matrices a Wilkinson shift cycles on. Perturbing s and t together
            // keeps the pair real-arithmetic-representable.
            if (++itersOnBlock % 10 == 0) {
                const double ex = std::abs(h(ihi, ihi - 1)) + std::abs(h(ihi - 1, ihi - 2));
                s = 1.5 * ex;
                t = ex * ex;
            }
            // First column of H^2 - sH + tI. Only three entries can be nonzero.
            double x = h(ilo, ilo) * h(ilo, ilo) + h(ilo, ilo + 1) * h(ilo + 1, ilo) -
                       s * h(ilo, ilo) + t;
            double y = h(ilo + 1, ilo) * (h(ilo, ilo) + h(ilo + 1, ilo + 1) - s);
            double z = (ilo + 2 <= ihi) ? h(ilo + 1, ilo) * h(ilo + 2, ilo + 1) : 0.0;

            for (int k = ilo; k <= ihi - 1; k++) {
                const int nr = std::min(3, ihi - k + 1);   // 3, then 2 at the bottom
                double u[3] = {0.0, 0.0, 0.0};
                if (k == ilo) {
                    u[0] = x; u[1] = y; u[2] = z;
                } else {
                    u[0] = h(k, k - 1);
                    u[1] = h(k + 1, k - 1);
                    u[2] = (nr == 3) ? h(k + 2, k - 1) : 0.0;
                }
                double nrm = 0.0;
                for (int i = 0; i < nr; i++) nrm += u[i] * u[i];
                nrm = std::sqrt(nrm);
                if (nrm == 0.0)
                    continue;
                // beta opposite in sign to u[0], so u[0]-beta cannot cancel.
                const double beta = (u[0] >= 0.0) ? -nrm : nrm;
                const double den = u[0] - beta;
                double v[3] = {1.0, 0.0, 0.0};
                for (int i = 1; i < nr; i++) v[i] = u[i] / den;
                const double tau = (beta - u[0]) / beta;   // == 2/(v.v)
                // From column k: columns to the left are zero in these rows
                // (Hessenberg), except k-1, which the reflector maps exactly to
                // (beta, 0, 0) and which is written out explicitly below.
                for (int j = std::max(k - 1, 0); j < n; j++) {          // H <- P H
                    double sum = 0.0;
                    for (int i = 0; i < nr; i++) sum += v[i] * h(k + i, j);
                    sum *= tau;
                    for (int i = 0; i < nr; i++) h(k + i, j) -= sum * v[i];
                }
                if (k > ilo) {   // force the exact structure the reflector implies
                    h(k, k - 1) = beta;
                    for (int i = 1; i < nr; i++) h(k + i, k - 1) = 0.0;
                }
                const int iMax = std::min(k + nr, ihi);
                for (int i = 0; i <= iMax; i++) {                       // H <- H P
                    double sum = 0.0;
                    for (int c2 = 0; c2 < nr; c2++) sum += v[c2] * h(i, k + c2);
                    sum *= tau;
                    for (int c2 = 0; c2 < nr; c2++) h(i, k + c2) -= sum * v[c2];
                }
                for (int i = 0; i < n; i++) {                           // Q <- Q P
                    double sum = 0.0;                                   // Qt holds Q^T
                    for (int c2 = 0; c2 < nr; c2++) sum += v[c2] * Qt[(k + c2) * n + i];
                    sum *= tau;
                    for (int c2 = 0; c2 < nr; c2++) Qt[(k + c2) * n + i] -= sum * v[c2];
                }
            }
            // Deflation is decided by the scan at the top of the loop, which
            // also recognises the isolated-2x2 case this used to miss.
        }

        for (int i = 0; i < n; i++)  // undo the transpose
            for (int j = 0; j < n; j++)
                Q[i * n + j] = Qt[j * n + i];
        return {H, Q};
    }

    // Shared Doolittle factorisation used by both LU() and det().
    // Returns {packedData, pivotVec} — packed lower/upper triangle + row-swap
    // record.
    // throwIfSingular = false makes a zero pivot a fact rather than an error:
    // the column below it is already zero, so the elimination simply steps past
    // it, U(k,k) stays 0, and P*A == L*U still holds. That is what det() and
    // LU() want — a singular matrix HAS a determinant (it is zero) and it HAS an
    // LU factorisation. solve(), inverse() and Decomposition keep the throw,
    // because a singular system genuinely has no unique solution to return.
    std::pair<std::vector<work_t<datatype>>, std::vector<int>> luPacked(
        bool throwIfSingular = true) const {
        using W = work_t<datatype>;
        if (rowSize != colSize)
            throw std::invalid_argument("LU: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        if (rowSize < 2)
            throw std::invalid_argument("LU: matrix must be at least 2x2, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));

        int n = (int)rowSize;
        std::vector<W> packed((std::size_t)(n * n));
        for (int k = 0; k < n * n; k++) packed[(std::size_t)k] = W(grid[k]);

        auto pat = [&](int i, int j) -> W& { return packed[(std::size_t)(i * n + j)]; };

        std::vector<int> pivotVec(n);
        for (int i = 0; i < n; i++)
            pivotVec[i] = i;

        for (int k = 0; k < n; k++) {
            // Pivot on MAGNITUDE, which is what partial pivoting means for a
            // complex matrix too — |z| is real whatever z is.
            int maxRow = k;
            double maxVal = magnitude(pat(k, k));
            for (int i = k + 1; i < n; i++) {
                double v = magnitude(pat(i, k));
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = i;
                }
            }
            if (maxVal == 0.0) {
                if (throwIfSingular)
                    throw std::runtime_error("LU: zero pivot in column " + std::to_string(k) +
                                             " — matrix is singular");
                // Everything at or below (k,k) in this column is already zero,
                // so there is nothing to eliminate and nothing to divide by:
                // record no swap and move on. U(k,k) is left at 0, which is
                // exactly what makes det() come out 0.
                pivotVec[k] = k;
                continue;
            }
            if (maxRow != k)
                for (int j = 0; j < n; j++)
                    std::swap(pat(k, j), pat(maxRow, j));
            pivotVec[k] = maxRow;
            for (int i = k + 1; i < n; i++)
                pat(i, k) /= pat(k, k);
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    pat(i, j) -= pat(i, k) * pat(k, j);
        }
        return {packed, pivotVec};
    }

    // Crossover point: matrices smaller than this use naive O(n³) multiplication.
    // 64 is a common empirical choice — below this the Strassen overhead
    // outweighs the asymptotic benefit.
    static constexpr long STRASSEN_THRESHOLD = 64;

    // Tuning constants and the parallel drivers — see namespace mstore.
    static constexpr long PARALLEL_MIN_WORK = mstore::PARALLEL_MIN_WORK;
    static constexpr long MAP_MIN_WORK = mstore::MAP_MIN_WORK;
    static constexpr long ELEMENTWISE_MIN_WORK = mstore::ELEMENTWISE_MIN_WORK;
    static int memoryThreads() { return mstore::memoryThreads(); }
    template <class F>
    static void forEachIndex(long total, F body) {
        mstore::forEachIndex(total, body);
    }

    // Shared body of every element-wise map (exp, ln, sin, ...). NumPy's
    // ufuncs are SIMD but strictly SINGLE-THREADED, so threading a
    // transcendental map is a gap that is not available to it: elem_ln at
    // n=2000 went 11.6 -> 0.62 ms. __restrict on both pointers as well,
    // without which the destination is assumed to alias the source.
    template <class F>
    Matrix mapElems(F fn) const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
#ifdef _OPENMP
        if (total >= MAP_MIN_WORK) {
    #pragma omp parallel for schedule(static)
            for (long i = 0; i < total; i++)
                r[i] = fn(a[i]);
            return ans;
        }
#endif
        for (long i = 0; i < total; i++)
            r[i] = fn(a[i]);
        return ans;
    }

    // Returns the smallest power of 2 >= n.
    static long nextPow2(long n) {
        long p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    // Cache-blocked, OpenMP-parallelised matrix multiplication.
    // Tile size: 64 elements × sizeof(datatype) fits comfortably in L1 cache.
    // Each outer ii-tile is an independent OpenMP task, so cores don't share
    // work. Used as the base case for Strassen-Winograd and for rectangular
    // matrices.
    static Matrix naiveMul(const Matrix& A, const Matrix& B) {
        // The kernel itself is mstore::gemm — see there for the blocking, the
        // restrict/by-value capture that roughly doubled its throughput, and
        // why the serial path branches around the OpenMP construct entirely.
        // Sharing it means Tensor's contractions use the same one.
        Matrix ans(A.rowSize, B.colSize, uninit_t{});
        mstore::gemm(A.grid, B.grid, ans.grid, A.rowSize, B.colSize, A.colSize);
        return ans;
    }

    // Extract the h×h sub-block of M starting at (r0, c0).
    static Matrix subBlock(const Matrix& M, long r0, long c0, long h) {
        Matrix out(h, h, uninit_t{});  // fully overwritten below
        for (long i = 0; i < h; i++) {
            const datatype* src = M.grid + (r0 + i) * M.colSize + c0;
            datatype* dst = out.grid + i * h;
            for (long j = 0; j < h; j++)
                dst[j] = src[j];
        }
        return out;
    }

    // Write block src (h×h) into dst at (r0, c0).
    static void setBlock(Matrix& dst, const Matrix& src, long r0, long c0, long h) {
        for (long i = 0; i < h; i++)
            for (long j = 0; j < h; j++)
                dst.grid[(r0 + i) * dst.colSize + (c0 + j)] = src.grid[i * h + j];
    }

    // Element-wise addition of two same-size matrices.
    static Matrix addMat(const Matrix& A, const Matrix& B) {
        Matrix out(A.rowSize, A.colSize, uninit_t{});
        const long total = A.rowSize * A.colSize;
        for (long k = 0; k < total; k++)
            out.grid[k] = A.grid[k] + B.grid[k];
        return out;
    }

    // Element-wise subtraction.
    static Matrix subMat(const Matrix& A, const Matrix& B) {
        Matrix out(A.rowSize, A.colSize, uninit_t{});
        const long total = A.rowSize * A.colSize;
        for (long k = 0; k < total; k++)
            out.grid[k] = A.grid[k] - B.grid[k];
        return out;
    }

    // Strassen-Winograd algorithm.
    //
    // Requires A and B to be square with size = power of 2.
    // Recursively splits into h×h quadrants and computes 7 recursive products
    // (versus 8 for standard multiplication), which is what buys the
    // O(n^2.807) exponent. Winograd's variant additionally cuts the additions
    // from 18 to 15 by sharing the auxiliary sums below.
    //
    //   S1 = A21 + A22        T1 = B12 - B11
    //   S2 = S1  - A11        T2 = B22 - T1
    //   S3 = A11 - A21        T3 = B22 - B12
    //   S4 = A12 - S2         T4 = T2  - B21
    //
    //   P1 = A11 * B11        P2 = A12 * B21
    //   P3 = S4  * B22        P4 = A22 * T4
    //   P5 = S1  * T1         P6 = S2  * T2
    //   P7 = S3  * T3
    //
    //   U1 = P1 + P2          U5 = U4 + P3
    //   U2 = P1 + P6          U6 = U3 - P4
    //   U3 = U2 + P7          U7 = U3 + P5
    //   U4 = U2 + P5
    //
    //   C11 = U1   C12 = U5   C21 = U6   C22 = U7
    //
    // Both T4 and U7 are easy to get subtly wrong, and a wrong version still
    // returns plausible-looking numbers of the right magnitude — the error is
    // only visible against a reference product. validate.cpp checks this path
    // against a naive multiply at every size class that reaches it.
    static Matrix strassenWinograd(const Matrix& A, const Matrix& B) {
        long n = A.rowSize;

        // Base case: fall back to naive multiplication
        if (n <= STRASSEN_THRESHOLD)
            return naiveMul(A, B);

        long h = n / 2;

        // Partition A into quadrants
        Matrix A11 = subBlock(A, 0, 0, h);
        Matrix A12 = subBlock(A, 0, h, h);
        Matrix A21 = subBlock(A, h, 0, h);
        Matrix A22 = subBlock(A, h, h, h);

        // Partition B into quadrants
        Matrix B11 = subBlock(B, 0, 0, h);
        Matrix B12 = subBlock(B, 0, h, h);
        Matrix B21 = subBlock(B, h, 0, h);
        Matrix B22 = subBlock(B, h, h, h);

        // Winograd auxiliary sums (saves additions vs plain Strassen)
        Matrix S1 = addMat(A21, A22);  // A21 + A22
        Matrix S2 = subMat(S1, A11);   // S1  - A11
        Matrix S3 = subMat(A11, A21);  // A11 - A21
        Matrix S4 = subMat(A12, S2);   // A12 - S2
        Matrix T1 = subMat(B12, B11);  // B12 - B11
        Matrix T2 = subMat(B22, T1);   // B22 - T1
        Matrix T3 = subMat(B22, B12);  // B22 - B12
        Matrix T4 = subMat(T2, B21);   // T2 - B21

        // 7 recursive multiplications
        Matrix P1 = strassenWinograd(A11, B11);
        Matrix P2 = strassenWinograd(A12, B21);
        Matrix P3 = strassenWinograd(S4, B22);
        Matrix P4 = strassenWinograd(A22, T4);
        Matrix P5 = strassenWinograd(S1, T1);
        Matrix P6 = strassenWinograd(S2, T2);
        Matrix P7 = strassenWinograd(S3, T3);

        // Combine into result quadrants — see the table in the comment above.
        Matrix U1 = addMat(P1, P2);
        Matrix U2 = addMat(P1, P6);
        Matrix U3 = addMat(U2, P7);
        Matrix U4 = addMat(U2, P5);
        Matrix U5 = addMat(U4, P3);
        Matrix U6 = subMat(U3, P4);
        Matrix U7 = addMat(U3, P5);

        // Assemble the n×n result from its four h×h quadrants
        Matrix C(n, n);
        setBlock(C, U1, 0, 0, h);  // C11
        setBlock(C, U5, 0, h, h);  // C12
        setBlock(C, U6, h, 0, h);  // C21
        setBlock(C, U7, h, h, h);  // C22
        return C;
    }
};

// pow(A, p) — matrix power A^p (not element-wise; use A.pow(p) for that).
//
// Integer p  — binary exponentiation using operator* (Strassen-accelerated).
//              Negative integers use A.inverse() then repeated squaring.
//
// Real p     — Higham Schur-Padé algorithm:
//   1. Schur decompose:  A = Q T Q^T
//   2. Compute T^p via Parlett recurrence on the upper triangular T
//      (diagonal entries λᵢ^p; super-diagonals via the commutativity equation
//      T·F = F·T)
//   3. Return Q * T^p * Q^T
//
// Requires square matrix. For real p, all eigenvalues must be positive
// (negative eigenvalues with non-integer p yield complex results — an exception
// is thrown).
//
// Usage: auto Ahalf = pow(A, 0.5);   // matrix square root
//        auto Ainv  = pow(A, -1);    // same as A.inverse()
//        auto A3    = pow(A, 3);     // A * A * A  via binary squaring
// Declared ahead of its definition because pow() and log() delegate to it for
// complex matrices, and it is defined further down beside the Schur-Parlett
// machinery it depends on.
template <typename datatype, typename F>
Matrix<std::complex<double>> funm(const Matrix<datatype>& A, F f);

template <typename datatype, typename scalar>
Matrix<work_t<datatype>> pow(const Matrix<datatype>& A, scalar p) {
    // ── complex: no real Schur form exists, so route through funm ──
    if constexpr (is_complex<datatype>::value) {
        using C = work_t<datatype>;
        if (A.rows() != A.cols())
            throw std::invalid_argument("pow: matrix must be square, got " +
                                        std::to_string(A.rows()) + "x" +
                                        std::to_string(A.cols()));
        const int nn = (int)A.rows();
        Matrix<C> Ac(nn, nn);
        for (int i = 0; i < nn; i++)
            for (int j = 0; j < nn; j++) Ac(i, j) = C(A(i, j));
        const double pd = double(std::real(p));
        // An INTEGER power needs no Schur form at all, and binary squaring has
        // no eigenvalue-separation requirement — so it stays exact where the
        // Parlett recurrence would refuse a repeated eigenvalue.
        if (pd == std::floor(pd) && std::abs(pd) < 1e9) {
            long long e = (long long)std::llabs((long long)pd);
            Matrix<C> base = Ac, result(nn, nn);
            for (int i = 0; i < nn; i++) result(i, i) = C(1);
            while (e) {
                if (e & 1) result = result * base;
                e >>= 1;
                if (e) base = base * base;
            }
            return (pd < 0) ? result.inverse() : result;
        }
        return funm(Ac, [pd](std::complex<double> z) { return std::pow(z, pd); });
    } else {
    if (A.rows() != A.cols())
        throw std::invalid_argument("pow: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    int n = (int)A.rows();

    // Convert to double for consistent arithmetic
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    auto identity = [&]() {
        Matrix<double> I(n, n);
        for (int i = 0; i < n; i++)
            I(i, i) = 1.0;
        return I;
    };

    // ── Integer fast path: binary exponentiation ──────────────────────────────
    long ip = (long)std::round(double(p));
    if (std::abs(double(p) - double(ip)) < 1e-9) {
        if (ip == 0)
            return identity();
        if (ip == 1)
            return Ad;
        Matrix<double> base = (ip < 0) ? Ad.inverse() : Ad;
        Matrix<double> result = identity();
        for (long exp = std::abs(ip); exp > 0; exp >>= 1) {
            if (exp & 1)
                result = result * base;
            if (exp > 1)
                base = base * base;
        }
        return result;
    }

    // ── Real power: Schur-Padé via Parlett recurrence ─────────────────────────
    // Step 1: Schur decompose Ad = Q * T * Q^T
    auto [Tv, Qv] = Ad.schurDecomp();
    // A 2x2 block in the real Schur form is a complex-conjugate eigenvalue pair.
    // The diagonal below reads such a block as its REAL PART, twice — and since
    // that part is usually positive, the positivity check passes and a wrong
    // answer comes back silently. (Measured: a 4x4 with eigenvalues 6.084+-0.403i
    // gave ||R*R - A|| = 6.7e-01 for sqrt.) funm complexifies the blocks
    // properly, and the answer is real again whenever a real matrix function
    // exists at all — which is the case the pair is there to describe.
    //
    // Same relative test as Matrix::isSchurBlock, repeated because that one is
    // private and this is a free function.
    auto schurBlockAt = [&](const std::vector<double>& T, int nn, int i) {
        const double sub = std::abs(T[(std::size_t)(i + 1) * nn + i]);
        const double nbr = std::abs(T[(std::size_t)i * nn + i]) +
                           std::abs(T[(std::size_t)(i + 1) * nn + (i + 1)]);
        return sub > std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0);
    };
    for (int bi = 0; bi + 1 < n; bi++)
        if (schurBlockAt(Tv, n, bi)) {
            const double pd = double(p);
            auto Fc = funm(Ad, [pd](std::complex<double> z) { return std::pow(z, pd); });
            double imagMax = 0.0, realMax = 0.0;
            for (int t = 0; t < n * n; t++) {
                imagMax = std::max(imagMax, std::abs(Fc[t].imag()));
                realMax = std::max(realMax, std::abs(Fc[t].real()));
            }
            if (imagMax > 1e-8 * std::max(1.0, realMax))
                throw std::domain_error(
                    "pow: the matrix has complex eigenvalues and no REAL matrix power "
                    "exists for exponent " + std::to_string(double(p)) +
                    " — the answer is genuinely complex. Call funm(A, f) with a "
                    "complex-valued f, which returns Matrix<std::complex<double>>");
            Matrix<double> outR(n, n);
            for (int t = 0; t < n * n; t++) outR[t] = Fc[t].real();
            return outR;
        }

    auto T = [&](int i, int j) -> double { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) -> double { return Qv[i * n + j]; };

    // Step 2: Parlett recurrence for F = T^p (upper triangular)
    // Diagonal: F[i,i] = T[i,i]^p  (eigenvalue must be positive for real result)
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error("pow: eigenvalue " + std::to_string(lam) +
                                    " is non-positive — real matrix power undefined "
                                    "for non-integer exponent");
        f(i, i) = std::pow(lam, double(p));
    }

    // Super-diagonals via commutativity equation T·F = F·T → Parlett recurrence.
    // For distinct eigenvalues (|λᵢ - λⱼ| > ε):
    //   F[i,j] = (T[i,j]·(F[i,i]−F[j,j]) + Σ_{k=i+1}^{j-1}(F[i,k]·T[k,j] −
    //   T[i,k]·F[k,j]))
    //            / (T[i,i] − T[j,j])
    // For repeated eigenvalues (|λᵢ−λⱼ| < relative eps), the standard formula
    // has a near-zero denominator. Use the L'Hôpital limit instead:
    //   lim_{λⱼ→λᵢ} (f(λᵢ)−f(λⱼ))/(λᵢ−λⱼ) = f'(λᵢ) = p·λᵢ^(p−1)
    // This is exact for adjacent super-diagonals (d=1) and a good approximation
    // for d>1 because when eigenvalues are equal the inner sum is also zero.
    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) * double(p) * std::pow(T(i, i), double(p) - 1.0);
        }
    }

    // Step 3: A^p = Q·F·Qᵀ  —  two O(n³) multiplications, not one O(n⁴) loop
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j) = f(i, j);
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
    }
}

// log(A, base) — matrix logarithm in an arbitrary base (free function).
// Distinct from A.log(base) which is the element-wise member function.
//
// Algorithm: Schur-Padé with f(x) = ln(x) / ln(base)
//   1. Schur decompose:  A = Q T Qᵀ
//   2. Compute F = ln(T) via Parlett recurrence on upper triangular T:
//        diagonal:       F[i,i] = ln(T[i,i])           (eigenvalue must be > 0)
//        super-diagonals: same recurrence as pow() with f'(λ) = 1/λ
//   3. Return Q · (F / ln(base)) · Qᵀ
//
// For symmetric (diagonalizable) A the Schur form is diagonal, so Parlett
// collapses to element-wise log on the eigenvalues — exact with no extra cost.
//
// Requires: square matrix, all eigenvalues positive, base > 0 and base ≠ 1.
// Usage: auto L2 = log(A, 2.0);   // log base-2 of matrix A
//        auto Le = log(A, M_E);   // natural matrix logarithm
template <typename datatype, typename scalar>
Matrix<work_t<datatype>> log(const Matrix<datatype>& A, scalar base) {
    if constexpr (is_complex<datatype>::value) {
        using C = work_t<datatype>;
        if (A.rows() != A.cols())
            throw std::invalid_argument("log: matrix must be square, got " +
                                        std::to_string(A.rows()) + "x" +
                                        std::to_string(A.cols()));
        if (double(std::real(base)) <= 0.0 || double(std::real(base)) == 1.0)
            throw std::invalid_argument("log: base must be positive and not equal to 1");
        const int nn = (int)A.rows();
        Matrix<C> Ac(nn, nn);
        for (int i = 0; i < nn; i++)
            for (int j = 0; j < nn; j++) Ac(i, j) = C(A(i, j));
        // No positive-eigenvalue restriction here: the complex logarithm is
        // defined on the whole cut plane, which is exactly what a complex
        // matrix needs and what the real path cannot offer.
        auto L = funm(Ac, [](std::complex<double> z) { return std::log(z); });
        const double lb = std::log(double(std::real(base)));
        for (int i = 0; i < nn * nn; i++) L[i] /= lb;
        return L;
    } else {
    if (A.rows() != A.cols())
        throw std::invalid_argument("log: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    if (double(base) <= 0.0 || double(base) == 1.0)
        throw std::invalid_argument("log: base must be positive and not equal to 1");

    int n = (int)A.rows();
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    // ── Schur decompose: A = Q T Qᵀ ─────────────────────────────────────────
    auto [Tv, Qv] = Ad.schurDecomp();
    // Same trap as pow(): a 2x2 block is a complex-conjugate pair, whose real
    // part alone passes the positivity check below and yields a silently wrong
    // logarithm. Measured at ||exp(log B) - B|| = 6.7e-01 before this guard.
    auto schurBlockAt = [&](const std::vector<double>& T, int nn, int i) {
        const double sub = std::abs(T[(std::size_t)(i + 1) * nn + i]);
        const double nbr = std::abs(T[(std::size_t)i * nn + i]) +
                           std::abs(T[(std::size_t)(i + 1) * nn + (i + 1)]);
        return sub > std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0);
    };
    for (int bi = 0; bi + 1 < n; bi++)
        if (schurBlockAt(Tv, n, bi)) {
            const double lbase = std::log(double(base));
            auto Fc = funm(Ad, [](std::complex<double> z) { return std::log(z); });
            double imagMax = 0.0, realMax = 0.0;
            for (int t = 0; t < n * n; t++) {
                imagMax = std::max(imagMax, std::abs(Fc[t].imag()));
                realMax = std::max(realMax, std::abs(Fc[t].real()));
            }
            if (imagMax > 1e-8 * std::max(1.0, realMax))
                throw std::domain_error(
                    "log: the matrix has complex eigenvalues and no REAL logarithm "
                    "exists — the answer is genuinely complex. Call "
                    "funm(A, [](std::complex<double> z){ return std::log(z); }), which "
                    "returns Matrix<std::complex<double>>");
            Matrix<double> outR(n, n);
            for (int t = 0; t < n * n; t++) outR[t] = Fc[t].real() / lbase;
            return outR;
        }

    auto T = [&](int i, int j) { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) { return Qv[i * n + j]; };

    // ── Parlett recurrence for F = ln(T), f(x)=ln(x), f'(x)=1/x ────────────
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error("log: eigenvalue " + std::to_string(lam) +
                                    " is non-positive — matrix logarithm is not real-valued");
        f(i, i) = std::log(lam);
    }

    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) / T(i, i);  // f'(λ) = 1/λ for ln
        }
    }

    // ── A^log = Q · (F/ln(base)) · Qᵀ  ─────────────────────────────────────
    double logBase = std::log(double(base));
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j) = f(i, j) / logBase;
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
    }
}

// Stream insertion, so std::cout << A works and so does writing to any other
// stream. Uses the default Pretty format; for anything else call
// A.print(os, fmt) or A.str(fmt).
template <typename datatype>
std::ostream& operator<<(std::ostream& os, const Matrix<datatype>& M) {
    return os << M.str();
}
