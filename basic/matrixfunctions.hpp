#pragma once

// ==========================================================================
//  Matrix functions of a matrix
// ==========================================================================
//
// exp(A), log(A), sqrt(A), pow(A,p), sin(A), cos(A), tan(A) — the MATRIX ones,
// not the element-wise members. A.exp() and exp(A) are different results; the
// convention is that the member dot is element-wise and the free function is
// the matrix operation.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "matrix.hpp"

// ─── Matrix functions (free) ────────────────────────────────────────────────
// Reminder on the convention this header follows:
//   A.exp()  → element-wise, e^(a_ij) for each element independently
//   exp(A)   → MATRIX exponential, the power series I + A + A²/2! + A³/3! + ...
// These are completely different results. The two above and the two below
// (A.sin() vs sin(A)) are the pairs most likely to be confused, so each doc
// comment says which one it is.
//
// All of these evaluate a Taylor series, and all of them accept an optional
// trailing TaylorOpts controlling how far that series runs — see TaylorOpts at
// the top of this header for the four call forms.

// Shared machinery for the Taylor-series matrix functions. Kept in a namespace
// because these are implementation details, not part of the Matrix API, and the
// names (eye, halvings) are ones a caller might reasonably want for themselves.
namespace taylor_detail {

    // n×n identity in whatever type the series is running in.
    template <class W>
    inline Matrix<W> eye(int n) {
        Matrix<W> E(n, n);
        for (int i = 0; i < n; i++) E(i, i) = W(1);
        return E;
    }

    // Validated double copy of A — every matrix function needs the same square,
    // non-empty check and the same conversion, so they share one.
    template <typename datatype>
    // Promotes A into the type the series should run in: double for a real
    // matrix, complex<double> for a complex one. The Taylor machinery below is
    // templated on that type, so exp/sin/cos/tan/sinh/cosh/tanh all work for
    // complex without a second implementation — the series are the same series,
    // only the arithmetic differs.
    Matrix<work_t<datatype>> squareAsWork(const Matrix<datatype>& A, const char* who) {
        if (A.rows() != A.cols())
            throw std::invalid_argument(std::string(who) + ": matrix must be square, got " +
                                        std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        if (A.rows() == 0)
            throw std::invalid_argument(std::string(who) + ": matrix must be non-empty");
        int n = (int)A.rows();
        Matrix<work_t<datatype>> Ad(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) Ad(i, j) = work_t<datatype>(A(i, j));
        return Ad;
    }

    // How many times to halve A so that ||A/2^s|| lands inside `radius`.
    //
    // This is the step that makes a Taylor series usable at all. The series for
    // e^A converges for every A in theory, but for ||A|| = 20 the terms grow to
    // about 20^20/20! ≈ 4e7 before they start shrinking, so the sum loses every
    // significant digit it had to cancellation. Halving first keeps every term
    // smaller than the last, and squaring afterwards costs only s extra products.
    inline int halvings(double nrm, double radius, bool enabled) {
        if (!enabled || !(nrm > radius))
            return 0;
        int s = (int)std::ceil(std::log2(nrm / radius));
        if (s < 0)
            s = 0;
        if (s > 64)
            s = 64;  // beyond this the squarings cost more than they buy
        return s;
    }

    // Evaluates {sin(A), cos(A)} — or {sinh(A), cosh(A)} when hyper is true.
    //
    // They are computed together on purpose: the double-angle identity that
    // undoes the scaling for sine, sin(2Y) = 2·sin(Y)·cos(Y), needs the cosine
    // anyway, so computing one alone would cost the same as computing both.
    //
    // Both series advance by a recurrence on the previous term:
    //     sin:  t₀ = X,  tₖ = tₖ₋₁ · X² / ( -(2k)(2k+1) )
    //     cos:  t₀ = I,  tₖ = tₖ₋₁ · X² / ( -(2k-1)(2k) )
    // with the sign folded into the denominator, and dropped for the hyperbolic
    // pair. One matrix product per term, and no factorial is ever formed — 171!
    // already overflows a double, so any implementation that divides by an
    // explicit factorial returns NaN long before 300 terms.
    template <class W>
    inline std::pair<Matrix<W>, Matrix<W>> sincosSeries(Matrix<W> X, TaylorOpts opts,
                                                        bool hyper) {
        int n = (int)X.rows();
        int s = halvings(X.norm(NormType::Inf), 1.0, opts.scaling);
        if (s)
            X = X / std::ldexp(1.0, s);

        Matrix<W> X2 = X * X;
        Matrix<W> Id = eye<W>(n);
        Matrix<W> S = X, sTerm = X;
        Matrix<W> C = Id, cTerm = Id;

        const double tol = opts.epsTol();
        bool sDone = false, cDone = false;
        for (long k = 1; k <= opts.maxTerms && !(sDone && cDone); k++) {
            double sDen = double(2 * k) * double(2 * k + 1);
            double cDen = double(2 * k - 1) * double(2 * k);
            if (!hyper) {
                sDen = -sDen;
                cDen = -cDen;
            }
            if (!sDone) {
                sTerm = sTerm * X2 / sDen;
                S += sTerm;
            }
            if (!cDone) {
                cTerm = cTerm * X2 / cDen;
                C += cTerm;
            }
            if (tol > 0.0) {
                if (sTerm.norm(NormType::Inf) <= tol * S.norm(NormType::Inf))
                    sDone = true;
                if (cTerm.norm(NormType::Inf) <= tol * C.norm(NormType::Inf))
                    cDone = true;
            }
        }

        // Undo the halving. The circular and hyperbolic identities have the same
        // shape, which is why one loop serves both:
        //   sin(2Y) = 2·sin·cos      sinh(2Y) = 2·sinh·cosh
        //   cos(2Y) = 2·cos² − I     cosh(2Y) = 2·cosh² − I
        for (int i = 0; i < s; i++) {
            Matrix<W> Snew = (S * C) * 2.0;
            C = (C * C) * 2.0 - Id;
            S = Snew;
        }
        return {S, C};
    }

}  // namespace taylor_detail

// exp(A) — the matrix exponential, sum of A^k / k!.
//
// The most valuable of the missing matrix functions: it is the closed-form
// solution of the linear ODE x' = A*x, the transition operator of a continuous
// Markov chain, the map from a Lie algebra to its Lie group, and — once complex
// support lands — exp(-i*H*t), the time-evolution operator the quantum-circuit
// goal in the README is built on.
//
// Unlike pow() and log() this does NOT need the Schur form, so it is unaffected
// by the 2x2 real-Schur block problem and is defined for every square matrix
// with no eigenvalue restrictions.
//
// Scaling and squaring around a Taylor series:
//   1. halve A until ||A/2^s||∞ <= 1/2, where the series converges fast and
//      monotonically
//   2. sum I + X + X²/2! + … by the recurrence tₖ = tₖ₋₁·X/k — one matrix
//      product per term, stopping as soon as a term stops contributing
//   3. square s times to undo the scaling, since e^A = (e^(A/2^s))^(2^s)
//
// Usage: auto E = exp(A);                  // defaults
//        auto E = exp(A, 25);              // at most 25 terms
//        auto E = exp(A, {25, 1e-12});     // ... or until terms fall below
//        1e-12 auto E = exp(A, {25, 1e-12, false});  // ... with scaling
//        disabled
template <typename datatype>
Matrix<work_t<datatype>> exp(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        using W = work_t<datatype>;
        Matrix<W> X = taylor_detail::squareAsWork(A, "exp");
        int n = (int)X.rows();

        int s = taylor_detail::halvings(X.norm(NormType::Inf), 0.5, opts.scaling);
        if (s)
            X = X / std::ldexp(1.0, s);

        Matrix<W> E = taylor_detail::eye<W>(n);
        Matrix<W> term = E;
        const double tol = opts.epsTol();
        for (long k = 1; k <= opts.maxTerms; k++) {
            term = term * X / double(k);
            E += term;
            if (tol > 0.0 && term.norm(NormType::Inf) <= tol * E.norm(NormType::Inf))
                break;
        }

        for (int i = 0; i < s; i++)
            E = E * E;
        return E;
    } catch (const std::exception& e) {
        std::cerr << "exp(A) error: " << e.what() << std::endl;
        throw;
    }
}

// ─── Matrix trigonometry (free) ─────────────────────────────────────────────
// sin(A), cos(A) and the hyperbolic pair are ENTIRE functions: their power
// series converge for every square matrix. So unlike log(A, base) — which
// requires positive eigenvalues — these have no domain restriction at all.
//
// That mattered for the route not taken. Evaluating them through the Schur form
// the way pow() and log() do would run straight into the 2x2 blocks that a real
// Schur form carries for complex-conjugate eigenvalue pairs, and a plain 2D
// rotation matrix — complex eigenvalues, perfectly ordinary real cos(A) — is
// enough to trigger it. Summing the series directly sidesteps the Schur form
// altogether, so these are correct for rotations today, with no dependency on
// the block-aware Parlett recurrence described at schurDecomp().
//
// Identities the test suite checks (matrix products, not element-wise):
//   sin(A)² + cos(A)²  == I
//   cosh(A)² − sinh(A)² == I
//   sin(A) is NOT A.sin(); the two agree only for diagonal A
//
// ACCURACY, and what to expect from those identities. If A has an eigenvalue
// with a large imaginary part, sin(A) and cos(A) genuinely grow like
// e^|Im(λ)| — for ||A||∞ ≈ 40 that is around 1e11 per entry. sin²+cos² then
// asks for ~1e22 of cancellation to land back on I, so the ABSOLUTE residual
// is bounded below by conditioning, not by the algorithm: no implementation in
// double precision returns a small one. The error relative to those
// intermediate magnitudes stays at machine epsilon, which is the meaningful
// measure. Symmetric A has real eigenvalues, so sin/cos stay bounded by 1 and
// the identity does hold to full absolute accuracy there.

// sin(A) — matrix sine, sum of (-1)^k A^(2k+1) / (2k+1)!.
// Optional TaylorOpts as for exp(): sin(A, 25), sin(A, {25, 1e-12}), etc.
template <typename datatype>
Matrix<work_t<datatype>> sin(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "sin"), opts, false);
        (void)C;
        return S;
    } catch (const std::exception& e) {
        std::cerr << "sin(A) error: " << e.what() << std::endl;
        throw;
    }
}

// cos(A) — matrix cosine, sum of (-1)^k A^(2k) / (2k)!.
// Optional TaylorOpts as for exp().
template <typename datatype>
Matrix<work_t<datatype>> cos(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "cos"), opts, false);
        (void)S;
        return C;
    } catch (const std::exception& e) {
        std::cerr << "cos(A) error: " << e.what() << std::endl;
        throw;
    }
}

// tan(A) — matrix tangent, the X solving cos(A)·X = sin(A).
//
// NOTE the operation: this is a linear SOLVE, cos(A)⁻¹·sin(A) — a LEFT
// division. sin(A)/cos(A) would now give the right division sin(A)·cos(A)⁻¹,
// which happens to be the same matrix here (sin(A) and cos(A) commute, both
// being functions of A) but is the wrong operation in general, and goes through
// an extra pair of transposes to get there. Undefined where cos(A) is singular.
template <typename datatype>
Matrix<work_t<datatype>> tan(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "tan"), opts, false);
        return C.solve(S);
    } catch (const std::exception& e) {
        std::cerr << "tan(A) error: " << e.what() << std::endl;
        throw;
    }
}

// sinh(A) — matrix hyperbolic sine, sum of A^(2k+1) / (2k+1)!.
//
// Summed directly rather than as (exp(A) − exp(−A))/2: that identity costs two
// matrix exponentials, and it cancels catastrophically for large ||A||, where
// exp(A) and exp(−A) agree to many digits before the subtraction throws them
// away.
template <typename datatype>
Matrix<work_t<datatype>> sinh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "sinh"), opts, true);
        (void)C;
        return S;
    } catch (const std::exception& e) {
        std::cerr << "sinh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// cosh(A) — matrix hyperbolic cosine, sum of A^(2k) / (2k)!.
template <typename datatype>
Matrix<work_t<datatype>> cosh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "cosh"), opts, true);
        (void)S;
        return C;
    } catch (const std::exception& e) {
        std::cerr << "cosh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// tanh(A) — matrix hyperbolic tangent, the X solving cosh(A)·X = sinh(A).
// A solve, not an element-wise division — see the note on tan(A).
template <typename datatype>
Matrix<work_t<datatype>> tanh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsWork(A, "tanh"), opts, true);
        return C.solve(S);
    } catch (const std::exception& e) {
        std::cerr << "tanh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// sqrt(A) — principal matrix square root, the X with X*X = A.
// Already reachable as pow(A, 0.5); this is the conventional spelling. Note
// this one goes through the Schur form, not a Taylor series, so it takes no
// TaylorOpts and it does inherit pow()'s positive-eigenvalue requirement.
template <typename datatype>
Matrix<work_t<datatype>> sqrt(const Matrix<datatype>& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("sqrt: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    return pow(A, 0.5);
}

// ─── Free-function spellings that mirror mathematical notation ──────────────

// sum(A) — every element. The member A.sum() is the same thing; both exist
// because capital sigma is written to the LEFT of what it sums, and because
// MATLAB only has the free spelling while NumPy has both.
template <typename datatype>
datatype sum(const Matrix<datatype>& A) {
    return A.sum();
}

// sum(A, ROW) / sum(A, COL) — along one axis. The flag names the axis you get
// ONE RESULT PER: sum(A, ROW) is an m x 1 column, sum(A, COL) a 1 x n row.
template <typename datatype>
Matrix<datatype> sum(const Matrix<datatype>& A, bool axis) {
    return A.sum(axis);
}

// tr(A) — sum of the main diagonal elements. Mirrors mathematical notation.
template <typename datatype>
datatype tr(const Matrix<datatype>& A) {
    return A.tr();
}

// det(A) — determinant. Mirrors mathematical notation.
template <typename datatype>
datatype det(const Matrix<datatype>& A) {
    return A.det();
}

// Thin wrappers over the members of the same name, for the callers who prefer
// f(A) notation. Each is a one-liner like tr/det above.

// solve(A, B) — solves A * X = B. See Matrix::solve.
template <typename datatype, typename dtB>
Matrix<work_t<datatype>> solve(const Matrix<datatype>& A, const Matrix<dtB>& B) {
    return A.solve(B);
}

// norm(A, type) — matrix norm. See Matrix::norm.
template <typename datatype>
double norm(const Matrix<datatype>& A, NormType type = NormType::Fro) {
    return A.norm(type);
}

// rank(A) — numerical rank. See Matrix::rank.
template <typename datatype>
long rank(const Matrix<datatype>& A, double tol = -1.0) {
    return A.rank(tol);
}

// cond(A, type) — condition number. See Matrix::cond.
template <typename datatype>
double cond(const Matrix<datatype>& A, NormType type = NormType::Two) {
    return A.cond(type);
}

// inverse(A) — matrix inverse. See Matrix::inverse.
template <typename datatype>
Matrix<work_t<datatype>> inverse(const Matrix<datatype>& A) {
    return A.inverse();
}

// pinv(A) — Moore-Penrose pseudo-inverse. See Matrix::pinv.
template <typename datatype>
Matrix<work_t<datatype>> pinv(const Matrix<datatype>& A, double tol = -1.0) {
    return A.pinv(tol);
}

// adj(A) — adjugate. See Matrix::adjugate.
template <typename datatype>
Matrix<double> adj(const Matrix<datatype>& A) {
    return A.adjugate();
}

// eigvals(A) — every eigenvalue, complex ones included. See Matrix::eigvals.
template <typename datatype>
Matrix<std::complex<double>> eigvals(const Matrix<datatype>& A) {
    return A.eigvals();
}

// diag(v) — builds a square diagonal matrix FROM a vector, the inverse
// direction of the member A.diag() which extracts a diagonal into a vector.
// v must be a row or column vector; an (n x 1) or (1 x n) input gives n x n.
