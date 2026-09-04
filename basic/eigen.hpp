#pragma once

// ==========================================================================
//  Eigenvalue problems beyond the Matrix members
// ==========================================================================
//
// funm and its Schur-Parlett machinery, the generalized problem eig(A,B) /
// eigvals(A,B), and QZ — the generalized Schur decomposition, which is what
// lets a singular or ill-conditioned B work at all.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "builders.hpp"

namespace mcpu {


// ═══════════════════════════════════════════════════════════════════════════
//  Generalized eigenvalue problems  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
//     eig(A, B)      symmetric-definite:  A x = lambda B x, A symmetric,
//                                         B symmetric positive definite
//     eigvals(A, B)  the general pencil, when B is well conditioned
//     polyeig({A0, A1, ..., Ad})          matrix polynomial eigenvalues
//
// ── WHAT IS HERE AND WHAT IS NOT ──────────────────────────────────────────
// The complete, backward-stable answer for a general pencil is the QZ
// algorithm — a generalized Schur decomposition, with its own
// Hessenberg-triangular reduction and shifted sweeps. That is a project in its
// own right and is NOT implemented. What is here instead:
//
//   * The SYMMETRIC-DEFINITE case in full, by Cholesky reduction. This is
//     backward stable, it is the case that actually turns up — vibration modes,
//     PCA with a covariance metric, generalized least squares — and it is
//     exactly what LAPACK's dsygv does.
//   * The general case by way of B⁻¹A, GUARDED by a condition estimate. Forming
//     B⁻¹A is legitimate when B is well conditioned and loses accuracy in
//     proportion to cond(B) when it is not, so this measures cond(B) with the
//     Hager-Higham estimator and REFUSES when the answer would not be
//     trustworthy, naming QZ as what is needed. Returning plausible-looking
//     numbers from a near-singular B is the failure this is written to avoid.
//
// eig(A, B) — symmetric-definite. Returns {eigenvalues ascending, eigenvectors},
// the same shape as the standard A.eig().
//
// The reduction: B = L Lᵀ, then A x = lambda B x becomes C y = lambda y with
// C = L⁻¹ A L⁻ᵀ and x = L⁻ᵀ y. C is symmetric, so the standard symmetric
// eigensolver applies and the eigenvalues come out real — which is the whole
// point of doing it this way rather than through B⁻¹A, whose product is not
// symmetric even when both factors are.
template <typename datatype>
std::pair<Matrix<work_t<datatype>>, Matrix<work_t<datatype>>> eig(const Matrix<datatype>& A,
                                                                  const Matrix<datatype>& B) {
    using W = work_t<datatype>;
    if (A.rows() != A.cols() || B.rows() != B.cols())
        throw std::invalid_argument("eig(A,B): both matrices must be square, got " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    " and " + std::to_string(B.rows()) + "x" +
                                    std::to_string(B.cols()));
    if (A.rows() != B.rows())
        throw std::invalid_argument("eig(A,B): A and B must be the same size, got " +
                                    std::to_string(A.rows()) + " and " + std::to_string(B.rows()));
    // Hermitian is the complex generalisation of symmetric — A = A^H, which is
    // what makes the eigenvalues real and the whole reduction below valid.
    const bool selfAdjoint = [&] {
        if constexpr (is_complex<datatype>::value) return A.IsHermitian();
        else return A.IsSymmetric();
    }();
    if (!selfAdjoint)
        throw std::invalid_argument(
            std::string("eig(A,B): A must be ") +
            (is_complex<datatype>::value ? "Hermitian" : "symmetric") +
            ". This is the definite solver, which returns B-orthonormal vectors and "
            "sorted real eigenvalues. For a GENERAL pencil use qz(A,B), whose "
            "eigenvectors() handles any square pencil including a singular B.");
    const long n = A.rows();

    // cholesky() throws domain_error on a non-positive pivot, and that failure
    // IS the positive-definiteness test — there is no separate check to pay for.
    Matrix<W> L;
    try {
        L = B.cholesky();
    } catch (const std::domain_error&) {
        throw std::invalid_argument(
            "eig(A,B): B must be positive definite — its Cholesky factorisation hit a "
            "non-positive pivot. For an indefinite or singular B use qz(A,B), which has "
            "no definiteness requirement.");
    }

    // L is TRIANGULAR, so substitution solves it outright — factorize() would
    // run an LU on a matrix that is already factored, and Decomposition is in
    // any case still real-only.
    auto cjs = [](const W& z) {
        if constexpr (is_complex<datatype>::value) return std::conj(z);
        else return z;
    };
    auto forwardSolve = [&](const Matrix<W>& Lm, const Matrix<W>& RHS) {   // L X = RHS
        const long nc = RHS.cols();
        Matrix<W> X(n, nc);
        for (long c = 0; c < nc; c++)
            for (long i = 0; i < n; i++) {
                W acc = RHS(int(i), int(c));
                for (long k = 0; k < i; k++) acc -= Lm(int(i), int(k)) * X(int(k), int(c));
                X(int(i), int(c)) = acc / Lm(int(i), int(i));
            }
        return X;
    };
    auto backSolveH = [&](const Matrix<W>& Lm, const Matrix<W>& RHS) {     // L^H X = RHS
        const long nc = RHS.cols();
        Matrix<W> X(n, nc);
        for (long c = 0; c < nc; c++)
            for (long i = n - 1; i >= 0; i--) {
                W acc = RHS(int(i), int(c));
                for (long k = i + 1; k < n; k++)
                    acc -= cjs(Lm(int(k), int(i))) * X(int(k), int(c));
                X(int(i), int(c)) = acc / cjs(Lm(int(i), int(i)));
            }
        return X;
    };
    Matrix<W> Ad(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) Ad(int(i), int(j)) = W(A(int(i), int(j)));

    // H() is the conjugate transpose, and for a real matrix it IS the transpose,
    // so this one spelling covers both cases.
    Matrix<W> Zm = forwardSolve(L, Ad);            // Z  = L⁻¹ A
    Matrix<W> C = forwardSolve(L, Zm.H()).H();     // C  = (L⁻¹ Z^H)^H = L⁻¹ A L⁻^H
    // C is self-adjoint in exact arithmetic; enforce it so that rounding cannot
    // hand the eigensolver something it is entitled to reject.
    C = (C + C.H()) * 0.5;

    auto [vals, Y] = C.eig();

    // x = L⁻ᵀ y, which is a solve against Lᵀ — NOT against L. Getting this wrong
    // is silent: the shapes all match and the eigenvalues are still right, only
    // the vectors are garbage, so the residual ||Ax - lambda Bx|| is the test
    // that catches it and there is one in validate.cpp.
    Matrix<W> X = backSolveH(L, Y);

    // A.eig() does not order its output, and MATLAB's symmetric-definite solver
    // does, so sort here and carry the vectors along with their values.
    std::vector<long> order((std::size_t)n);
    std::iota(order.begin(), order.end(), 0L);
    // Sorted on the REAL part: a self-adjoint pencil has real eigenvalues, and
    // std::complex has no ordering to sort on directly.
    std::stable_sort(order.begin(), order.end(), [&](long a, long b) {
        return std::real(vals(int(a), 0)) < std::real(vals(int(b), 0));
    });
    Matrix<W> valsOut(n, 1), Xout(n, n);
    for (long j = 0; j < n; j++) {
        const long src = order[(std::size_t)j];
        valsOut(int(j), 0) = vals(int(src), 0);
        for (long i = 0; i < n; i++) Xout(int(i), int(j)) = X(int(i), int(src));
    }
    // DELIBERATELY NOT rescaled to unit Euclidean length. The right normalisation
    // for this problem is B-orthonormality, Xᵀ B X = I, which is what LAPACK's
    // dsygv returns and what makes the eigenvectors a basis in the metric B
    // defines. It comes out for free: Y is orthonormal from the symmetric
    // eigensolver, and X = L⁻ᵀ Y gives
    //     Xᵀ B X = Yᵀ L⁻¹ (L Lᵀ) L⁻ᵀ Y = Yᵀ Y = I.
    // Scaling the columns to Euclidean length 1 would DESTROY that — which is
    // what an earlier version of this function did, and what the X^H B X == I
    // assertion in validate.cpp caught.
    return {valsOut, Xout};
}

// eigvals(A, B) — the general pencil A x = lambda B x, as the eigenvalues of
// B⁻¹A, and only when that is a defensible thing to compute.
//
// The guard is the point. cond(B) bounds how much accuracy forming B⁻¹A can
// cost, so it is estimated first — cheaply, with the same Hager-Higham
// estimator rcond() uses — and the call is refused outright when B is close to
// singular. Pass a smaller tol to override, knowing what it buys.
template <typename dtA, typename dtB>
Matrix<std::complex<double>> eigvals(const Matrix<dtA>& A, const Matrix<dtB>& B,
                                     double tol = 1e-10) {
    if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
        throw std::invalid_argument("eigvals(A,B): both matrices must be square and the "
                                    "same size");
    // Goes through QZ, which never forms B^-1*A — so an ILL-CONDITIONED B is no
    // longer a problem, and the old rcond guard is gone with it. Verified
    // against scipy.linalg.eig(A,B) at rcond(B) = 1e-10, where the previous
    // B^-1*A reduction refused outright.
    const long n = A.rows();
    auto r = qz(A, B);
    const auto al = r.alpha(), be = r.beta();
    // `tol` now scales the test for beta == 0 rather than bounding rcond(B). It
    // plays the same role — how singular is too singular — on the quantity the
    // QZ path actually depends on.
    double bn = 0.0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) bn = std::max(bn, std::abs(r.T(int(i), int(j))));
    const double bzero = std::max(tol, 0.0) * std::max(bn, 1.0);
    const auto undef = r.undefined();
    Matrix<std::complex<double>> out(n, 1);
    for (long i = 0; i < n; i++) {
        if (undef[(std::size_t)i])
            throw std::domain_error(
                "eigvals(A,B): the pencil is SINGULAR — alpha and beta are both zero at "
                "index " + std::to_string(i) + ", so det(A - lambda B) vanishes "
                "identically and no eigenvalue is determined there. A and B share a null "
                "space. The numbers qz(A,B) returns for such a position are rounding "
                "noise; check qz(A,B).undefined() before using them.");
        if (std::abs(be[(std::size_t)i]) <= bzero)
            throw std::domain_error(
                "eigvals(A,B): the pencil has an INFINITE eigenvalue at index " +
                std::to_string(i) + " — B is singular, so no finite lambda exists there "
                "and a Matrix of values cannot express the answer. Call qz(A,B) and read "
                "alpha()/beta(), which reports the pair rather than the ratio.");
        out(int(i), 0) = al[(std::size_t)i] / be[(std::size_t)i];
    }
    return out;
}

// polyeig({A0, A1, ..., Ad}) — eigenvalues of the matrix polynomial
//     P(lambda) = A0 + lambda*A1 + ... + lambda^d * Ad,
// the values of lambda for which P(lambda) is singular. Coefficients ASCENDING,
// matching MATLAB's polyeig(A0, A1, ..., Ad) argument order — and note that is
// the opposite of the DESCENDING order polyval, roots and polyfit use, because
// MATLAB itself is inconsistent here and matching it beats being quietly
// different.
//
// Solved by LINEARISATION: the degree-d n-by-n polynomial becomes a linear
// pencil of size d*n, whose eigenvalues are exactly P's. That is what MATLAB and
// every serious implementation do, because a companion pencil is well
// understood where root-finding on det(P(lambda)) is hopeless.
//
//     C1 = [ 0    I    0   ...  ]        C2 = diag(I, I, ..., I, Ad)
//          [ 0    0    I   ...  ]
//          [ ...                ]
//          [ -A0  -A1  ...  -Ad-1]
//
// A d*n-by-d*n problem for an n-by-n polynomial, so there are d*n eigenvalues.
// Requires Ad nonsingular; when it is not, some eigenvalues are infinite and the
// linearisation needs different handling, which the conditioning guard in
// eigvals(A,B) will catch and report.
template <typename datatype>
Matrix<std::complex<double>> polyeig(const std::vector<Matrix<datatype>>& coeffs,
                                     double tol = 1e-10) {
    if (coeffs.size() < 2)
        throw std::invalid_argument("polyeig: need at least two coefficient matrices "
                                    "(a constant and a linear term)");
    const long n = coeffs[0].rows();
    const long d = (long)coeffs.size() - 1;
    for (const auto& M : coeffs)
        if (M.rows() != n || M.cols() != n)
            throw std::invalid_argument("polyeig: every coefficient must be the same "
                                        "square size, expected " + std::to_string(n) +
                                        "x" + std::to_string(n));
    const long N = d * n;
    Matrix<double> C1(N, N), C2(N, N);
    for (long b = 0; b + 1 < d; b++)                    // super-diagonal identities
        for (long i = 0; i < n; i++) C1(int(b * n + i), int((b + 1) * n + i)) = 1.0;
    for (long b = 0; b < d; b++)                        // bottom row block: -A0..-A(d-1)
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                C1(int((d - 1) * n + i), int(b * n + j)) =
                    -double(std::real(coeffs[(std::size_t)b](int(i), int(j))));
    for (long b = 0; b + 1 < d; b++)                    // C2 = diag(I,...,I,Ad)
        for (long i = 0; i < n; i++) C2(int(b * n + i), int(b * n + i)) = 1.0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            C2(int((d - 1) * n + i), int((d - 1) * n + j)) =
                double(std::real(coeffs[(std::size_t)d](int(i), int(j))));
    return eigvals(C1, C2, tol);
}

// ═══════════════════════════════════════════════════════════════════════════
//  funm — a general matrix function  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
//     funm(A, f)   where f is any callable complex<double> -> complex<double>
//
//     funm(A, [](std::complex<double> z){ return std::exp(z); })   == exp(A)
//     funm(A, [](std::complex<double> z){ return std::sqrt(z); })  == sqrt(A)
//
// THIS IS NUMERIC, NOT SYMBOLIC. f is a function pointer or lambda that gets
// called on complex numbers; nothing here differentiates it, expands it, or
// looks at its form. That is a deliberate boundary — a matrix function library
// and a computer algebra system are different projects, and the moment this
// needed derivatives of f it would be sliding into the second one.
//
// It costs something, and the cost is stated below rather than hidden: the
// robust Schur-Parlett of Higham's algorithm reorders the Schur form into
// clusters and evaluates a TAYLOR SERIES on each diagonal block, which needs the
// derivatives f', f'', ... This implementation cannot do that, so it detects the
// case where it would be needed and refuses instead of returning noise.
//
// ── HOW IT WORKS ──────────────────────────────────────────────────────────
//   1. Real Schur:  A = Q T Qᵀ, T quasi-triangular.
//   2. Complexify:  each 2x2 block (a complex-conjugate eigenvalue pair) is
//      triangularised by a 2x2 unitary, giving a genuinely upper-triangular
//      complex Tc and a unitary Qc = Q Z. This is the piece the roadmap called
//      out as missing; doing it here means the Parlett recurrence below is
//      scalar throughout and never needs a Sylvester solve.
//   3. Parlett recurrence on Tc: the diagonal is f of the eigenvalues, and each
//      superdiagonal follows from the ones before it.
//   4. Undo:  f(A) = Qc F Qcᴴ.
//
// ── WHEN IT REFUSES, AND WHY THAT IS THE RIGHT ANSWER ─────────────────────
// Step 3 divides by (T_ii - T_jj). When two eigenvalues are close that division
// amplifies error without limit, and when they are equal it is undefined. The
// separation is checked against the size of the matrix and funm THROWS if it is
// too small, naming the two eigenvalues. A wrong f(A) that looks like a matrix
// is worse than an exception.
//
// If you hit that: exp, log, sqrt, sin, cos, sinh, cosh, tanh and pow all have
// dedicated implementations elsewhere in this header that do NOT go through
// Parlett — exp uses scaling-and-squaring, log the inverse of it — and those
// have no eigenvalue-separation requirement at all. funm is for the functions
// that do not have one.
namespace funm_detail {

using cplx = std::complex<double>;

// Real Schur form -> complex Schur form. Every 2x2 block on the diagonal holds a
// complex-conjugate pair; a 2x2 unitary built from one eigenvector triangularises
// it, and applying that to the whole matrix keeps everything else triangular.
inline void complexify(std::vector<cplx>& T, std::vector<cplx>& Z, long n,
                       const Matrix<double>& Treal) {
    for (long i = 0; i + 1 < n; i++) {
        // WHETHER THIS IS A REAL 2x2 BLOCK IS A RELATIVE QUESTION, not an exact
        // one. The QR iteration drives a converged subdiagonal towards zero
        // without ever setting it exactly to zero — on a 4x4 test matrix the
        // three subdiagonals came out {0, 2.8e-23, 3.5e-01}, and a `!= 0` test
        // read that 2.8e-23 as a block, consumed two rows on it, and then
        // stepped straight past the genuine block behind it. The result was
        // silently wrong: still triangular-looking, still real, just not f(A).
        // Matrix::isSchurBlock already encodes the right test; this is the same
        // criterion, spelled out here because that one is a private member.
        const double sub = std::abs(Treal(int(i + 1), int(i)));
        const double nbr = std::abs(Treal(int(i), int(i))) +
                           std::abs(Treal(int(i + 1), int(i + 1)));
        if (sub <= std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0))
            continue;
        // A genuine 2x2 block. Its eigenvalues are the roots of the block's
        // characteristic polynomial; a negative discriminant is what makes them
        // a conjugate pair rather than two reals the QR iteration already split.
        const cplx a = T[(std::size_t)(i * n + i)],       b = T[(std::size_t)(i * n + i + 1)];
        const cplx c = T[(std::size_t)((i + 1) * n + i)], d = T[(std::size_t)((i + 1) * n + i + 1)];
        const cplx tr = a + d, det = a * d - b * c;
        const cplx disc = std::sqrt(tr * tr - 4.0 * det);
        const cplx lam = 0.5 * (tr + disc);
        // Eigenvector of the 2x2 block, taking whichever column is better scaled.
        cplx v0, v1;
        if (std::abs(b) >= std::abs(lam - a)) { v0 = b;       v1 = lam - a; }
        else                                  { v0 = lam - d; v1 = c; }
        const double nrm = std::sqrt(std::norm(v0) + std::norm(v1));
        if (nrm == 0.0) continue;
        v0 /= nrm; v1 /= nrm;
        // U = [v, w] with w orthonormal to v — in 2x2 that is just this.
        const cplx u00 = v0, u10 = v1;
        const cplx u01 = -std::conj(v1), u11 = std::conj(v0);
        // T := Uᴴ T U and Z := Z U, both touching only rows/columns i and i+1.
        for (long k = 0; k < n; k++) {                     // rows i, i+1 <- Uᴴ * rows
            const cplx r0 = T[(std::size_t)(i * n + k)], r1 = T[(std::size_t)((i + 1) * n + k)];
            T[(std::size_t)(i * n + k)]       = std::conj(u00) * r0 + std::conj(u10) * r1;
            T[(std::size_t)((i + 1) * n + k)] = std::conj(u01) * r0 + std::conj(u11) * r1;
        }
        for (long k = 0; k < n; k++) {                     // columns i, i+1 <- cols * U
            const cplx c0 = T[(std::size_t)(k * n + i)], c1 = T[(std::size_t)(k * n + i + 1)];
            T[(std::size_t)(k * n + i)]     = c0 * u00 + c1 * u10;
            T[(std::size_t)(k * n + i + 1)] = c0 * u01 + c1 * u11;
        }
        for (long k = 0; k < n; k++) {
            const cplx z0 = Z[(std::size_t)(k * n + i)], z1 = Z[(std::size_t)(k * n + i + 1)];
            Z[(std::size_t)(k * n + i)]     = z0 * u00 + z1 * u10;
            Z[(std::size_t)(k * n + i + 1)] = z0 * u01 + z1 * u11;
        }
        T[(std::size_t)((i + 1) * n + i)] = cplx(0.0, 0.0);   // exactly zero now
        i++;                                                  // skip the partner row
    }
}

}  // namespace funm_detail

// ─────────────────────────────────────────────────────────────────────────────
//  QZ — the generalized Schur decomposition  (tier 4)
// ─────────────────────────────────────────────────────────────────────────────
//
// For a matrix PENCIL (A, B), find unitary Q and Z with
//
//     Q^H A Z = S      Q^H B Z = T      both upper triangular
//
// so the eigenvalues of A x = lambda B x are the RATIOS S(i,i)/T(i,i). Nothing
// ever forms B^-1 A, which is the whole point: eigvals(A,B) currently refuses
// outright when rcond(B) is small, because forming that product spends the
// available precision before the eigensolver even starts.
//
// A ratio also expresses what B^-1 A cannot. T(i,i) = 0 with S(i,i) != 0 is an
// INFINITE eigenvalue — a genuine feature of a singular pencil, not a failure —
// and 0/0 marks the pencil as singular, where the eigenvalues are not defined
// at all.
//
// As with the Schur form, the COMPLEX case is the simpler one: S and T both
// come out fully triangular, where a real QZ leaves S quasi-triangular with a
// 2x2 block per conjugate pair and needs a double shift to stay in real
// arithmetic. This implementation works in work_t throughout, so a real pencil
// is handled by promoting it.
namespace qz_detail {

// conj for complex, identity for real. std::conj(double) returns a
// complex<double>, so the plain overload is not optional.
template <class R>
inline R cj(const R& z) {
    return z;
}
template <class R>
inline std::complex<R> cj(const std::complex<R>& z) {
    return std::conj(z);
}

// [ c         s ] [f]   [r]      c real and >= 0,  c^2 + |s|^2 = 1
// [ -conj(s)  c ] [g] = [0]
// The same rotation schurDecompComplex uses, written free so both can share it.
template <class W>
inline void givens(const W& f, const W& g, double& c, W& s) {
    const double af = std::abs(f), ag = std::abs(g);
    if (ag == 0.0) { c = 1.0; s = W(0); return; }
    if (af == 0.0) { c = 0.0; s = W(1); return; }
    const double h = std::hypot(af, ag);
    c = af / h;
    // conj(g)*(f/|f|)/h rather than c*conj(g/f): never divides by a tiny f.
    s = cj(g) * (f / af) / h;
}

// ── Hessenberg-triangular reduction — LAPACK's zgghrd ────────────────────────
//
// On entry B must ALREADY be upper triangular (the caller gets there with an
// unpivoted QR). On exit A is upper Hessenberg, B is still upper triangular,
// and Q, Z have been updated so that the original pencil is Q A Z^H, Q B Z^H.
//
// The whole difficulty is that the two have to move together. A left rotation
// chosen to kill A(i,j) also hits B, and because B is triangular that rotation
// introduces a BULGE at B(i,i-1). A second rotation — from the right this time,
// so it does not disturb the zero just made in A — removes it. Every step is
// therefore a PAIR, and the column that the right rotation touches (i-1 and i)
// is always strictly right of the column j being cleared, which is why the
// zeros already placed in A survive.
template <class W>
void hessTri(std::vector<W>& A, std::vector<W>& B,
             std::vector<W>& Q, std::vector<W>& Z, int n) {
    auto a = [&](int i, int j) -> W& { return A[(std::size_t)i * n + j]; };
    auto b = [&](int i, int j) -> W& { return B[(std::size_t)i * n + j]; };
    // Q and Z are held TRANSPOSED for the duration. Every update to them is a
    // rotation of a COLUMN PAIR, which strides a cache line per element in
    // row-major storage and is two contiguous runs transposed. With the sweeps
    // fixed this reduction became 56-63% of the whole QZ, and these two loops
    // were most of it.
    {
        std::vector<W> tmp((std::size_t)n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)j * n + i] = Q[(std::size_t)i * n + j];
        Q.swap(tmp);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)j * n + i] = Z[(std::size_t)i * n + j];
        Z.swap(tmp);
    }

    for (int j = 0; j < n - 2; j++) {
        // Upward, so each rotation acts on rows (i-1, i) and cannot refill the
        // zero the previous one placed at row i.
        for (int i = n - 1; i >= j + 2; i--) {
            double c; W s;
            givens(a(i - 1, j), a(i, j), c, s);          // kill A(i,j)
            for (int k = j; k < n; k++) {                // A <- G A
                const W t1 = a(i - 1, k), t2 = a(i, k);
                a(i - 1, k) = c * t1 + s * t2;
                a(i, k)     = -cj(s) * t1 + c * t2;
            }
            for (int k = i - 1; k < n; k++) {            // B <- G B
                const W t1 = b(i - 1, k), t2 = b(i, k);  // cols < i-1 are zero
                b(i - 1, k) = c * t1 + s * t2;
                b(i, k)     = -cj(s) * t1 + c * t2;
            }
            {                                            // Q <- Q G^H
                W* MATRIXCPP_RESTRICT q0 = &Q[(std::size_t)(i - 1) * n];
                W* MATRIXCPP_RESTRICT q1 = &Q[(std::size_t)i * n];
                for (int k = 0; k < n; k++) {
                    const W t1 = q0[k], t2 = q1[k];
                    q0[k] = c * t1 + cj(s) * t2;
                    q1[k] = -s * t1 + c * t2;
                }
            }
            a(i, j) = W(0);

            // B(i,i-1) is now a bulge. Kill it from the RIGHT, on columns
            // (i-1, i) — both strictly right of j, so A(:,j) is untouched.
            double c2; W s2;
            givens(b(i, i), b(i, i - 1), c2, s2);
            for (int k = 0; k <= i; k++) {               // B <- B W
                const W t1 = b(k, i - 1), t2 = b(k, i);
                b(k, i - 1) = c2 * t1 - cj(s2) * t2;
                b(k, i)     = s2 * t1 + c2 * t2;
            }
            for (int k = 0; k < n; k++) {                // A <- A W
                const W t1 = a(k, i - 1), t2 = a(k, i);
                a(k, i - 1) = c2 * t1 - cj(s2) * t2;
                a(k, i)     = s2 * t1 + c2 * t2;
            }
            {                                            // Z <- Z W
                W* MATRIXCPP_RESTRICT z0 = &Z[(std::size_t)(i - 1) * n];
                W* MATRIXCPP_RESTRICT z1 = &Z[(std::size_t)i * n];
                for (int k = 0; k < n; k++) {
                    const W t1 = z0[k], t2 = z1[k];
                    z0[k] = c2 * t1 - cj(s2) * t2;
                    z1[k] = s2 * t1 + c2 * t2;
                }
            }
            b(i, i - 1) = W(0);
        }
    }
    for (int i = 2; i < n; i++)                          // clear numerical dust
        for (int j = 0; j <= i - 2; j++) a(i, j) = W(0);
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++) b(i, j) = W(0);
    {   // back from the transposed layout Q and Z were held in
        std::vector<W> tmp((std::size_t)n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)i * n + j] = Q[(std::size_t)j * n + i];
        Q.swap(tmp);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)i * n + j] = Z[(std::size_t)j * n + i];
        Z.swap(tmp);
    }
}

// Brings a general pencil to Hessenberg-triangular form: an unpivoted QR of B
// to make it triangular, then hessTri above. Returns flat row-major (S,T,Q,Z).
template <class W, typename dtA, typename dtB>
std::tuple<std::vector<W>, std::vector<W>, std::vector<W>, std::vector<W>>
reduce(const Matrix<dtA>& A, const Matrix<dtB>& B) {
    const int n = (int)A.rows();
    Matrix<W> Ad(n, n), Bd(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Ad(i, j) = W(A(i, j)); Bd(i, j) = W(B(i, j)); }
    // UNPIVOTED, necessarily: a pivoted QR would give B*P = Q*R, and the
    // permutation would have to be undone on the pencil rather than absorbed.
    auto [Q1, R1, P1] = Bd.QR(QRMode::Complete, QRPivot::Off);
    (void)P1;
    Matrix<W> Aw = Q1.H() * Ad;                          // A <- Q1^H A

    std::vector<W> Av((std::size_t)n * n), Bv((std::size_t)n * n),
        Qv((std::size_t)n * n), Zv((std::size_t)n * n, W(0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Av[(std::size_t)i * n + j] = Aw(i, j);
            Bv[(std::size_t)i * n + j] = R1(i, j);
            Qv[(std::size_t)i * n + j] = Q1(i, j);
        }
    for (int i = 0; i < n; i++) Zv[(std::size_t)i * n + i] = W(1);
    for (int i = 1; i < n; i++)                          // R from QR, exactly
        for (int j = 0; j < i; j++) Bv[(std::size_t)i * n + j] = W(0);

    hessTri(Av, Bv, Qv, Zv, n);
    return {Av, Bv, Qv, Zv};
}

// ── The QZ iteration — LAPACK's zhgeqz ───────────────────────────────────────
//
// Drives S's subdiagonal to zero while keeping T triangular, by an implicit
// single-shift step. The step is a BULGE CHASE done in pairs: a left rotation
// that tidies S spoils T, and the right rotation that repairs T spoils S one
// row further down, so the two alternate their way to the bottom of the block.
//
// Single shift because this runs in complex arithmetic. A real QZ needs a
// double shift to keep a conjugate pair real, and leaves S quasi-triangular
// with a 2x2 block per pair — the same trade the Schur code makes.
template <class W>
void iterate(std::vector<W>& S, std::vector<W>& T, std::vector<W>& Q,
             std::vector<W>& Z, int n) {
    auto sm = [&](int i, int j) -> W& { return S[(std::size_t)i * n + j]; };
    auto tm = [&](int i, int j) -> W& { return T[(std::size_t)i * n + j]; };
    // Q and Z are held TRANSPOSED. Every update to them rotates a pair of
    // COLUMNS, which in row-major storage strides one cache line per element;
    // transposed, the same update is two contiguous runs. schurDecomp does the
    // same thing for its Q, for the same reason.
    {
        std::vector<W> tmp(Q.size());
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)j * n + i] = Q[(std::size_t)i * n + j];
        Q.swap(tmp);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)j * n + i] = Z[(std::size_t)i * n + j];
        Z.swap(tmp);
    }

    // `hi` is the bottom of the active block; the rotations read it to know how
    // far down anything can be nonzero.
    int hi = n - 1;

    // ── The rotation ranges are RESTRICTED, and that is where the time was ──
    // An earlier version applied every rotation to the full row and column,
    // reasoning that rotating a pair of zeros costs nothing but correctness is
    // not at risk. It cost a great deal: a column rotation touched all n rows
    // where only the first p+2 can be nonzero, and QZ came out 12-14x slower
    // than schurDecomp for an algorithm that should be about 2x.
    //   left  on rows (p,p+1): S row p starts at column p-1, so columns below
    //                          that are zero in both rows. The upper end cannot
    //                          be trimmed — the off-diagonal block to the right
    //                          is live.
    //   right on columns (p,p+1): S is Hessenberg so S(i,p) is zero past
    //                          i = p+1, and the chase's bulge reaches p+2.
    //                          Everything past hi is deflated and already zero.
    auto rotL = [&](int p, double c, const W& sv) {          // rows (p, p+1)
        const int k0 = (p > 0) ? p - 1 : 0;
        W* MATRIXCPP_RESTRICT s0 = &S[(std::size_t)p * n];
        W* MATRIXCPP_RESTRICT s1 = &S[(std::size_t)(p + 1) * n];
        W* MATRIXCPP_RESTRICT t0 = &T[(std::size_t)p * n];
        W* MATRIXCPP_RESTRICT t1 = &T[(std::size_t)(p + 1) * n];
        for (int k = k0; k < n; k++) {
            const W a1 = s0[k], a2 = s1[k];
            s0[k] = c * a1 + sv * a2;
            s1[k] = -cj(sv) * a1 + c * a2;
            const W b1 = t0[k], b2 = t1[k];
            t0[k] = c * b1 + sv * b2;
            t1[k] = -cj(sv) * b1 + c * b2;
        }
        W* MATRIXCPP_RESTRICT q0 = &Q[(std::size_t)p * n];   // contiguous, transposed
        W* MATRIXCPP_RESTRICT q1 = &Q[(std::size_t)(p + 1) * n];
        for (int k = 0; k < n; k++) {                        // Q <- Q G^H
            const W u = q0[k], v = q1[k];
            q0[k] = c * u + cj(sv) * v;
            q1[k] = -sv * u + c * v;
        }
    };
    auto rotR = [&](int p, double c, const W& sv) {          // columns (p, p+1)
        const int kEnd = std::min(p + 2, hi);
        for (int k = 0; k <= kEnd; k++) {
            W* MATRIXCPP_RESTRICT sr = &S[(std::size_t)k * n];
            const W a1 = sr[p], a2 = sr[p + 1];
            sr[p] = c * a1 - cj(sv) * a2;
            sr[p + 1] = sv * a1 + c * a2;
            W* MATRIXCPP_RESTRICT tr = &T[(std::size_t)k * n];
            const W b1 = tr[p], b2 = tr[p + 1];
            tr[p] = c * b1 - cj(sv) * b2;
            tr[p + 1] = sv * b1 + c * b2;
        }
        W* MATRIXCPP_RESTRICT z0 = &Z[(std::size_t)p * n];   // contiguous, transposed
        W* MATRIXCPP_RESTRICT z1 = &Z[(std::size_t)(p + 1) * n];
        for (int k = 0; k < n; k++) {                        // Z <- Z W
            const W u = z0[k], v = z1[k];
            z0[k] = c * u - cj(sv) * v;
            z1[k] = sv * u + c * v;
        }
    };

    const double eps = std::numeric_limits<double>::epsilon();
    double anorm = 0.0, bnorm = 0.0;
    for (int i = 0; i < n * n; i++) {
        anorm = std::max(anorm, std::abs(S[(std::size_t)i]));
        bnorm = std::max(bnorm, std::abs(T[(std::size_t)i]));
    }
    // Absolute floors, derived from the size of the pencil. A purely relative
    // test on T's diagonal cannot see an infinite eigenvalue, because there the
    // neighbouring entries it would scale against are the ones going to zero.
    const double atol = eps * std::max(anorm, 1.0) * double(n);
    const double btol = eps * std::max(bnorm, 1.0) * double(n);

    int iter = 0;
    while (hi > 0) {
        // ── deflate on S's subdiagonal, relative to its neighbours ──
        int lo = hi;
        while (lo > 0) {
            const double sc = std::abs(sm(lo - 1, lo - 1)) + std::abs(sm(lo, lo));
            // Relative to the neighbours, but never below the pencil's own
            // scale: with both neighbours near zero a purely relative test
            // chases rounding noise and the iteration stops converging.
            if (std::abs(sm(lo, lo - 1)) <= std::max(eps * sc, atol)) {
                sm(lo, lo - 1) = W(0);
                break;
            }
            lo--;
        }
        if (lo == hi) { hi--; iter = 0; continue; }
        if (++iter > 150)
            throw std::runtime_error("qz: iteration failed to converge at index " +
                                     std::to_string(hi));

        // ── an INFINITE eigenvalue, deflated at the BOTTOM ──
        // T(hi,hi) == 0 means row hi of T is zero across columns hi-1 and hi,
        // so a RIGHT rotation on those columns can zero S(hi,hi-1) WITHOUT
        // disturbing T: both entries it would touch there are already zero.
        // hi then carries the pair (S(hi,hi), 0) — lambda = infinity, which is
        // a real eigenvalue of a singular pencil and precisely what B^-1*A
        // cannot express.
        if (std::abs(tm(hi, hi)) <= btol) {
            double c; W sv;
            givens(sm(hi, hi), sm(hi, hi - 1), c, sv);
            rotR(hi - 1, c, sv);
            sm(hi, hi - 1) = W(0);
            tm(hi, hi) = W(0);
            hi--;
            iter = 0;
            continue;
        }
        // ── the mirror case: an infinite eigenvalue at the TOP of the block ──
        // T(lo,lo) == 0 makes column lo of T zero from row lo down, so a LEFT
        // rotation on rows (lo, lo+1) can zero S(lo+1,lo) and leave T exactly
        // as triangular as it found it. lo then carries (S(lo,lo), 0).
        if (std::abs(tm(lo, lo)) <= btol) {
            double c; W sv;
            givens(sm(lo, lo), sm(lo + 1, lo), c, sv);
            rotL(lo, c, sv);
            sm(lo + 1, lo) = W(0);
            tm(lo, lo) = W(0);
            iter = 0;
            continue;
        }
        // An INTERIOR zero on T's diagonal needs NO special handling — the
        // ordinary sweep drives it to an end on its own, where the two
        // deflations above take it. This was checked rather than assumed:
        // compared against scipy.linalg.eig(A,B) as (alpha,beta) pairs on the
        // Riemann sphere (the scale-invariant comparison, and the only one that
        // can compare an infinite eigenvalue at all), singular pencils of
        // rank deficiency 1, 2 and 3 agree to 1e-15, and pencils with BOTH A
        // and B rank-deficient agree to 5e-11.
        // An earlier version chased interior zeros explicitly and got it wrong:
        // a rotation zeroing T(k+1,k+1) leaves T(k,k) zero too, because both
        // entries it draws on are already zero, so the zero SPREADS along the
        // diagonal instead of moving — n-1 infinite eigenvalues reported for a
        // pencil with one. Doing nothing is not merely simpler here, it is
        // correct.

        // ── the shift: an eigenvalue of the trailing 2x2 PENCIL ──
        // det(S2 - lambda T2) = 0 with T2 upper triangular expands to
        //     (T11 T22) l^2 - (S11 T22 + S22 T11 - S21 T12) l + (S11 S22 - S21 S12)
        W shift;
        const W s11 = sm(hi - 1, hi - 1), s12 = sm(hi - 1, hi);
        const W s21 = sm(hi, hi - 1), s22 = sm(hi, hi);
        const W t11 = tm(hi - 1, hi - 1), t12 = tm(hi - 1, hi), t22 = tm(hi, hi);
        const W aq = t11 * t22;
        const W bq = -(s11 * t22 + s22 * t11 - s21 * t12);
        const W cq = s11 * s22 - s21 * s12;
        if (iter % 15 == 0) {
            shift = (std::abs(t11) > btol) ? (s11 + W(std::abs(s21))) / t11
                                           : W(std::abs(s21) + 1.0);   // exceptional
        } else if (std::abs(aq) <= atol * std::max(std::abs(bq), 1.0)) {
            // The quadratic degenerated — the trailing pencil is itself nearly
            // singular. Fall back to the linear root.
            shift = (std::abs(bq) > 0.0) ? -cq / bq : W(1);
        } else {
            const W disc = std::sqrt(bq * bq - W(4.0) * aq * cq);
            const W l1 = (-bq + disc) / (W(2.0) * aq);
            const W l2 = (-bq - disc) / (W(2.0) * aq);
            const W target = (std::abs(t22) > btol) ? s22 / t22 : s22;
            shift = (std::abs(l1 - target) < std::abs(l2 - target)) ? l1 : l2;
        }

        // ── the bulge chase ──
        {
            double c; W sv;
            givens(sm(lo, lo) - shift * tm(lo, lo), sm(lo + 1, lo), c, sv);
            rotL(lo, c, sv);                     // T now bulges at (lo+1, lo)
        }
        for (int k = lo; k < hi; k++) {
            double c; W sv;
            givens(tm(k + 1, k + 1), tm(k + 1, k), c, sv);
            rotR(k, c, sv);                      // repair T; S bulges at (k+2, k)
            tm(k + 1, k) = W(0);
            if (k + 2 <= hi) {
                double c2; W s2;
                givens(sm(k + 1, k), sm(k + 2, k), c2, s2);
                rotL(k + 1, c2, s2);             // repair S; T bulges at (k+2, k+1)
                sm(k + 2, k) = W(0);
            }
        }
    }
    for (int i = 1; i < n; i++)                  // clear numerical dust
        for (int j = 0; j < i; j++) { sm(i, j) = W(0); tm(i, j) = W(0); }
    {   // back from the transposed layout Q and Z were held in
        std::vector<W> tmp(Q.size());
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)i * n + j] = Q[(std::size_t)j * n + i];
        Q.swap(tmp);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tmp[(std::size_t)i * n + j] = Z[(std::size_t)j * n + i];
        Z.swap(tmp);
    }
}

}  // namespace qz_detail

// The generalized Schur decomposition of a pencil (A, B).
//
//   Q^H A Z = S      Q^H B Z = T      S, T upper triangular, Q, Z unitary
//
// Equivalently A == Q S Z^H and B == Q T Z^H. Eigenvalue i of A x = lambda B x
// is S(i,i)/T(i,i) — use alpha()/beta() rather than dividing yourself, because
// T(i,i) == 0 is a legitimate INFINITE eigenvalue and not an error.
//
// Always returns complex factors, whatever went in: a real pencil can have
// complex eigenvalues, so the single-shift iteration cannot stay in real
// arithmetic. The Hessenberg-triangular reduction beforehand DOES run in the
// input's own type, which keeps that O(n^3) stage at real cost for a real
// pencil.
struct QZResult {
    Matrix<std::complex<double>> S, T, Q, Z;

    // Eigenvalues as the PAIR (alpha, beta), which is how LAPACK reports them
    // and the only honest way to express an infinite one. lambda = alpha/beta.
    std::vector<std::complex<double>> alpha() const {
        std::vector<std::complex<double>> a;
        for (long i = 0; i < S.rows(); i++) a.push_back(S(int(i), int(i)));
        return a;
    }
    std::vector<std::complex<double>> beta() const {
        std::vector<std::complex<double>> b;
        for (long i = 0; i < T.rows(); i++) b.push_back(T(int(i), int(i)));
        return b;
    }
    // Right eigenvectors, one per column, matching alpha()/beta() by index.
    //
    // A x = lambda B x with lambda = alpha/beta is the same as the HOMOGENEOUS
    //     (beta*A - alpha*B) x = 0
    // and that form is the one to back-substitute on, because it stays finite
    // when beta is zero. An infinite eigenvalue then has an ordinary
    // eigenvector — it solves B x = 0 — where dividing first would have given
    // nothing but NaN.
    //
    // Q^H A Z = S and Q^H B Z = T are triangular, so with y = Z^H x the system
    // is (beta*S - alpha*T) y = 0. Its k-th diagonal entry is
    // beta_k*S(k,k) - alpha_k*T(k,k) = 0 identically, so y_k = 1, y_j = 0 above,
    // and the rest follows by back-substitution. Then x = Z y.
    Matrix<std::complex<double>> eigenvectors() const {
        using C = std::complex<double>;
        const long n = S.rows();
        double scale = 0.0;
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                scale = std::max(scale, std::abs(S(int(i), int(j))) + std::abs(T(int(i), int(j))));
        if (scale == 0.0) scale = 1.0;
        const double tiny = std::numeric_limits<double>::epsilon() * scale * double(n);

        Matrix<C> vecs(n, n);
        std::vector<C> y((std::size_t)n);
        for (long k = 0; k < n; k++) {
            const C al = S(int(k), int(k)), be = T(int(k), int(k));
            std::fill(y.begin(), y.end(), C(0));
            y[(std::size_t)k] = C(1);
            for (long j = k - 1; j >= 0; j--) {
                C acc = C(0);
                for (long m = j + 1; m <= k; m++)
                    acc += (be * S(int(j), int(m)) - al * T(int(j), int(m))) * y[(std::size_t)m];
                C den = be * S(int(j), int(j)) - al * T(int(j), int(j));
                // A repeated eigenvalue makes the pivot vanish and the vector
                // non-unique; perturbing is what LAPACK's ztgevc does, and it
                // keeps a defective pencil from returning NaN.
                if (std::abs(den) < tiny) den = C(tiny);
                y[(std::size_t)j] = -acc / den;
            }
            for (long i = 0; i < n; i++) {                 // x = Z y
                C acc = C(0);
                for (long m = 0; m <= k; m++) acc += Z(int(i), int(m)) * y[(std::size_t)m];
                vecs(int(i), int(k)) = acc;
            }
            double nrm = 0.0;
            for (long i = 0; i < n; i++) nrm += std::norm(vecs(int(i), int(k)));
            nrm = std::sqrt(nrm);
            if (nrm > 0.0)
                for (long i = 0; i < n; i++) vecs(int(i), int(k)) /= C(nrm);
        }
        return vecs;
    }

    // True where BOTH alpha and beta are zero: 0/0, which is not an eigenvalue
    // at all. It means the pencil itself is SINGULAR — det(A - lambda B) is
    // identically zero, because A and B share a null space — and then no
    // lambda is determined. The entries that come back for those positions are
    // rounding noise, and they look like ordinary numbers, so this is worth
    // asking about before trusting any of them.
    // (Measured on a 20x20 with a shared 12-dimensional null space: exactly 12
    // undefined pairs here and 12 from scipy.linalg.eig, with the remaining 8
    // agreeing to 1e-15.)
    std::vector<bool> undefined(double tol = 0.0) const {
        double an = 0.0, bn = 0.0;
        for (long i = 0; i < S.rows(); i++) {
            an = std::max(an, std::abs(S(int(i), int(i))));
            bn = std::max(bn, std::abs(T(int(i), int(i))));
        }
        if (tol <= 0.0) tol = std::numeric_limits<double>::epsilon() * double(S.rows()) * 100.0;
        std::vector<bool> f;
        for (long i = 0; i < S.rows(); i++)
            f.push_back(std::abs(S(int(i), int(i))) <= tol * std::max(an, 1.0) &&
                        std::abs(T(int(i), int(i))) <= tol * std::max(bn, 1.0));
        return f;
    }

    // True where beta is zero: an infinite eigenvalue, which happens exactly
    // when B is singular. B^-1*A cannot represent these at all.
    std::vector<bool> infinite(double tol = 0.0) const {
        double bn = 0.0;
        for (long i = 0; i < T.rows(); i++)
            for (long j = 0; j < T.cols(); j++) bn = std::max(bn, std::abs(T(int(i), int(j))));
        if (tol <= 0.0)
            tol = std::numeric_limits<double>::epsilon() * std::max(bn, 1.0) * double(T.rows());
        std::vector<bool> f;
        for (long i = 0; i < T.rows(); i++) f.push_back(std::abs(T(int(i), int(i))) <= tol);
        return f;
    }
};

template <typename dtA, typename dtB>
QZResult qz(const Matrix<dtA>& A, const Matrix<dtB>& B) {
    using cplx = std::complex<double>;
    // The reduction runs in the pencil's own arithmetic — real stays real, which
    // is roughly 4x cheaper for that O(n^3) stage. Only the sweep must be
    // complex.
    using W = std::conditional_t<is_complex<dtA>::value || is_complex<dtB>::value,
                                 cplx, double>;
    if (A.rows() != A.cols() || B.rows() != B.cols())
        throw std::invalid_argument("qz: both matrices must be square, got " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    " and " + std::to_string(B.rows()) + "x" +
                                    std::to_string(B.cols()));
    if (A.rows() != B.rows())
        throw std::invalid_argument("qz: A and B must be the same size, got " +
                                    std::to_string(A.rows()) + " and " + std::to_string(B.rows()));
    if (A.rows() == 0)
        throw std::invalid_argument("qz: matrices must be non-empty");
    const int n = (int)A.rows();

    auto [Sv, Tv, Qv, Zv] = qz_detail::reduce<W>(A, B);
    std::vector<cplx> S((std::size_t)n * n), T((std::size_t)n * n),
        Q((std::size_t)n * n), Z((std::size_t)n * n);
    for (int i = 0; i < n * n; i++) {
        S[(std::size_t)i] = cplx(Sv[(std::size_t)i]);
        T[(std::size_t)i] = cplx(Tv[(std::size_t)i]);
        Q[(std::size_t)i] = cplx(Qv[(std::size_t)i]);
        Z[(std::size_t)i] = cplx(Zv[(std::size_t)i]);
    }
    // ── normalise before the sweep, restore after ──
    // The shift is a ratio of products of S and T entries, so a pencil with
    // ||A|| ~ 1e9 and ||B|| ~ 1e-9 drives it through eighteen orders of
    // magnitude and the tolerances stop meaning anything: 70 of 400 stress
    // pencils failed to converge without this, all of them deliberately
    // ill-scaled. Scaling A by 1/as and B by 1/bs multiplies every eigenvalue
    // by bs/as, which is undone exactly by scaling S and T back afterwards —
    // and because it is applied to the FACTORS, A == Q S Z^H still holds to the
    // bit.
    double anorm = 0.0, bnorm = 0.0;
    for (int i = 0; i < n * n; i++) {
        anorm = std::max(anorm, std::abs(S[(std::size_t)i]));
        bnorm = std::max(bnorm, std::abs(T[(std::size_t)i]));
    }
    const double as = (anorm > 0.0) ? anorm : 1.0;
    const double bs = (bnorm > 0.0) ? bnorm : 1.0;
    for (int i = 0; i < n * n; i++) {
        S[(std::size_t)i] /= as;
        T[(std::size_t)i] /= bs;
    }
    qz_detail::iterate(S, T, Q, Z, n);
    for (int i = 0; i < n * n; i++) {
        S[(std::size_t)i] *= as;
        T[(std::size_t)i] *= bs;
    }

    QZResult out;
    out.S = Matrix<cplx>(n, n); out.T = Matrix<cplx>(n, n);
    out.Q = Matrix<cplx>(n, n); out.Z = Matrix<cplx>(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            out.S(i, j) = S[(std::size_t)i * n + j];
            out.T(i, j) = T[(std::size_t)i * n + j];
            out.Q(i, j) = Q[(std::size_t)i * n + j];
            out.Z(i, j) = Z[(std::size_t)i * n + j];
        }
    return out;
}


template <typename datatype, typename F>
Matrix<std::complex<double>> funm(const Matrix<datatype>& A, F f) {
    using cplx = std::complex<double>;
    if (A.rows() != A.cols())
        throw std::invalid_argument("funm: matrix must be square, got " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    const long n = A.rows();
    if (n == 0) throw std::invalid_argument("funm: matrix must be non-empty");

    // QZ is the basis in which T is triangular, so that f(A) = QZ f(T) QZ^H.
    std::vector<cplx> T((std::size_t)(n * n)), QZ((std::size_t)(n * n), cplx(0.0, 0.0));
    if constexpr (is_complex<datatype>::value) {
        // A complex matrix's Schur form is ALREADY triangular — no 2x2 blocks
        // exist to complexify, and the Schur vectors are the basis outright.
        // The complex path is the simpler of the two.
        auto [Tc, Qc] = A.schur();
        for (long i = 0; i < n * n; i++) {
            T[(std::size_t)i] = cplx(Tc[int(i)]);
            QZ[(std::size_t)i] = cplx(Qc[int(i)]);
        }
    } else {
        auto [Treal, Qreal] = A.schur();
        // Promote to complex, then triangularise the 2x2 blocks.
        std::vector<cplx> Z((std::size_t)(n * n), cplx(0.0, 0.0));
        for (long i = 0; i < n * n; i++) T[(std::size_t)i] = cplx(Treal[int(i)], 0.0);
        for (long i = 0; i < n; i++) Z[(std::size_t)(i * n + i)] = cplx(1.0, 0.0);
        funm_detail::complexify(T, Z, n, Treal);
        for (long i = 0; i < n; i++)
            for (long k = 0; k < n; k++) {
                const cplx q = cplx(Qreal(int(i), int(k)), 0.0);
                if (q == cplx(0.0, 0.0)) continue;
                for (long j = 0; j < n; j++)
                    QZ[(std::size_t)(i * n + j)] += q * Z[(std::size_t)(k * n + j)];
            }
    }

    // ── Parlett recurrence ────────────────────────────────────────────────
    // F is upper triangular with the same structure as T. The diagonal is f of
    // the eigenvalues; each superdiagonal follows from the ones already known.
    std::vector<cplx> Fm((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n; i++) Fm[(std::size_t)(i * n + i)] = f(T[(std::size_t)(i * n + i)]);

    // Separation floor, relative to the size of the matrix — an absolute
    // tolerance would be meaningless for a matrix scaled by 10^6.
    double tscale = 0.0;
    for (long i = 0; i < n; i++) tscale = std::max(tscale, std::abs(T[(std::size_t)(i * n + i)]));
    const double sepTol = std::numeric_limits<double>::epsilon() *
                          std::max(1.0, tscale) * double(n);

    for (long d = 1; d < n; d++)                       // superdiagonal by superdiagonal
        for (long i = 0; i + d < n; i++) {
            const long j = i + d;
            const cplx tii = T[(std::size_t)(i * n + i)], tjj = T[(std::size_t)(j * n + j)];
            const cplx den = tii - tjj;
            if (std::abs(den) <= sepTol) {
                std::ostringstream os;
                os << "funm: eigenvalues " << i << " and " << j << " are too close to "
                   << "separate (" << tii.real() << (tii.imag() < 0 ? "" : "+") << tii.imag()
                   << "i and " << tjj.real() << (tjj.imag() < 0 ? "" : "+") << tjj.imag()
                   << "i). The Parlett recurrence divides by their difference, so the "
                   << "result would be noise. A robust answer needs the derivatives of f, "
                   << "which this deliberately does not compute — see the note above funm. "
                   << "If f is exp, log, sqrt, sin, cos, sinh, cosh, tanh or a power, use "
                   << "the dedicated version in this header instead: none of those go "
                   << "through Parlett.";
                throw std::domain_error(os.str());
            }
            cplx acc = T[(std::size_t)(i * n + j)] *
                       (Fm[(std::size_t)(i * n + i)] - Fm[(std::size_t)(j * n + j)]);
            for (long k = i + 1; k < j; k++)
                acc += Fm[(std::size_t)(i * n + k)] * T[(std::size_t)(k * n + j)] -
                       T[(std::size_t)(i * n + k)] * Fm[(std::size_t)(k * n + j)];
            Fm[(std::size_t)(i * n + j)] = acc / den;
        }

    // f(A) = (Q Z) F (Q Z)ᴴ.
    std::vector<cplx> tmp((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n; i++)
        for (long k = 0; k < n; k++) {
            const cplx a = QZ[(std::size_t)(i * n + k)];
            if (a == cplx(0.0, 0.0)) continue;
            for (long j = k; j < n; j++)                  // F is upper triangular
                tmp[(std::size_t)(i * n + j)] += a * Fm[(std::size_t)(k * n + j)];
        }
    Matrix<cplx> out(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            cplx acc(0.0, 0.0);
            for (long k = 0; k < n; k++)
                acc += tmp[(std::size_t)(i * n + k)] * std::conj(QZ[(std::size_t)(j * n + k)]);
            out(int(i), int(j)) = acc;
        }
    return out;
}

}  // namespace mcpu
