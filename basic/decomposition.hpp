#pragma once

// ==========================================================================
//  Factor once, solve many times
// ==========================================================================
//
// The Decomposition class and factorize(), plus solve / inverse / pinv /
// lsqminnorm / rcond built on it. Holds LU, Cholesky or QR and reuses it.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "matrixfunctions.hpp"

// ═══════════════════════════════════════════════════════════════════════════
//  Decomposition — factor once, solve many times  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
// A.solve(b) factors A from scratch on every call. That is the right default
// for a one-off, and completely wrong in a loop: at n = 512 the factorisation
// is ~7 ms and a triangular solve is ~0.1 ms, so a hundred right-hand sides
// solved one at a time cost a hundred factorisations.
//
//     auto dA = A.factorize();          // or decomposition(A), MATLAB's
//     spelling Matrix<double> x1 = dA.solve(b1); Matrix<double> x2 =
//     dA.solve(b2);
//
// The object OWNS its factors, so it stays valid after A is destroyed — it is a
// value, like everything else here, not a view onto the matrix it came from.
//
// It picks the factorisation the way MATLAB's decomposition does, from the
// structure of A rather than from a flag the caller has to get right:
//     square, symmetric, positive definite  ->  Cholesky   (half the flops)
//     square, otherwise                     ->  LU with partial pivoting
//     rectangular                           ->  column-pivoted QR (least
//     squares)
// kind() reports which was chosen, because "why is this slower than I expected"
// is a question the answer should be available for.
template <typename datatype>
class Decomposition {
  public:
    enum class Kind { Cholesky, LU, QR };
    // Every stored factor and every solve runs in the working type: double for
    // a real matrix, complex<double> for a complex one. This class used to be
    // real-only outright, which is why solve() once began by taking
    // std::real() of its right-hand side.
    using W = work_t<datatype>;

    explicit Decomposition(const Matrix<datatype>& A) : rows_(A.rows()), cols_(A.cols()) {
        if (rows_ == 0 || cols_ == 0)
            throw std::invalid_argument("Decomposition: matrix must be non-empty");
        if (rows_ == cols_) {
            // Cholesky is worth trying first: it is half the work of LU, and the
            // attempt is itself the positive-definiteness test — a non-positive
            // pivot IS the proof, so there is no separate check to pay for.
            // HERMITIAN, not symmetric, for a complex matrix: A = A^H is what
            // Cholesky needs, and a complex symmetric matrix does not qualify.
            const bool selfAdjoint = [&] {
                if constexpr (is_complex<datatype>::value) return A.IsHermitian();
                else return A.IsSymmetric();
            }();
            if (selfAdjoint) {
                try {
                    chol_ = A.cholesky();
                    kind_ = Kind::Cholesky;
                    return;
                } catch (const std::domain_error&) {
                    // Self-adjoint but not positive definite. Fall through to LU.
                }
            }
            auto [packed, piv] = A.luPacked();
            lu_ = std::move(packed);
            piv_ = std::move(piv);
            kind_ = Kind::LU;
            return;
        }
        if (rows_ < cols_)
            throw std::invalid_argument(
                "Decomposition: under-determined systems (" + std::to_string(rows_) +
                " equations, " + std::to_string(cols_) +
                " unknowns) have infinitely "
                "many solutions — use lsqminnorm() for the minimum-norm one");
        auto [Q, R, P] = A.QR();
        q_ = std::move(Q);
        r_ = std::move(R);
        perm_.resize((std::size_t)cols_);
        for (long k = 0; k < cols_; k++)
            for (long i = 0; i < cols_; i++)
                if (double(std::real(P(int(i), int(k)))) != 0.0)
                    perm_[(std::size_t)k] = i;
        kind_ = Kind::QR;
    }

    Kind kind() const { return kind_; }

    const char* kindName() const {
        switch (kind_) {
            case Kind::Cholesky:
                return "Cholesky";
            case Kind::LU:
                return "LU";
            default:
                return "QR";
        }
    }

    long rows() const { return rows_; }
    long cols() const { return cols_; }

    // Solves A * X = B using the stored factors. No refactorisation.
    template <typename dtB>
    Matrix<W> solve(const Matrix<dtB>& B) const {
        if (B.rows() != rows_)
            throw std::invalid_argument(
                "Decomposition::solve: B must have one row per row of A — A is " +
                std::to_string(rows_) + "x" + std::to_string(cols_) + " but B is " +
                std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
        const long nrhs = B.cols();
        Matrix<W> Bd(B.rows(), nrhs);
        for (long i = 0; i < B.rows(); i++)
            for (long j = 0; j < nrhs; j++)
                Bd(int(i), int(j)) = W(B(int(i), int(j)));

        if (kind_ == Kind::LU)
            return Matrix<datatype>::luSubstitute(lu_, piv_, (int)rows_, Bd);

        if (kind_ == Kind::Cholesky) {
            // A = L L^H, so forward-substitute through L then back through
            // L^H. The BACK pass takes the conjugate — for a real matrix that
            // is a no-op, which is why the real-only version could omit it.
            const long n = rows_;
            Matrix<W> X(n, nrhs);
            for (long c = 0; c < nrhs; c++) {
                std::vector<W> y((std::size_t)n);
                for (long i = 0; i < n; i++) {
                    W acc = Bd(int(i), int(c));
                    for (long j = 0; j < i; j++)
                        acc -= chol_(int(i), int(j)) * y[(std::size_t)j];
                    y[(std::size_t)i] = acc / chol_(int(i), int(i));
                }
                for (long i = n - 1; i >= 0; i--) {
                    W acc = y[(std::size_t)i];
                    for (long j = i + 1; j < n; j++)
                        acc -= cjw(chol_(int(j), int(i))) * X(int(j), int(c));
                    X(int(i), int(c)) = acc / cjw(chol_(int(i), int(i)));
                }
            }
            return X;
        }

        // QR least squares: A*P = Q*R, so minimise ||R (P^H x) - Q^H b||.
        Matrix<W> QtB = q_.H() * Bd;
        const double rTol = double(std::max(rows_, cols_)) *
                            std::numeric_limits<double>::epsilon() * std::abs(r_(0, 0));
        long rk = 0;
        while (rk < cols_ && std::abs(r_(int(rk), int(rk))) > rTol)
            rk++;
        Matrix<W> Y(cols_, nrhs);
        for (long c = 0; c < nrhs; c++)
            for (long i = rk - 1; i >= 0; i--) {
                W acc = QtB(int(i), int(c));
                for (long j = i + 1; j < rk; j++)
                    acc -= r_(int(i), int(j)) * Y(int(j), int(c));
                Y(int(i), int(c)) = acc / r_(int(i), int(i));
            }
        Matrix<W> X(cols_, nrhs);
        for (long k = 0; k < cols_; k++)
            for (long c = 0; c < nrhs; c++)
                X(int(perm_[(std::size_t)k]), int(c)) = Y(int(k), int(c));
        return X;
    }

    // Determinant, free from the factors already held. Only square systems have
    // one, so the QR case says so rather than returning something meaningless.
    W det() const {
        if (kind_ == Kind::QR)
            throw std::logic_error("Decomposition::det: only a square system has a determinant");
        if (kind_ == Kind::Cholesky) {
            W d = W(1);
            for (long i = 0; i < rows_; i++)
                d *= chol_(int(i), int(i));
            return d * cjw(d);  // det(L L^H) = det(L)*conj(det(L)) = |det(L)|²
        }
        W d = W(1);
        int swaps = 0;
        for (long i = 0; i < rows_; i++) {
            d *= lu_[(std::size_t)(i * rows_ + i)];
            if (piv_[(std::size_t)i] != (int)i)
                swaps++;
        }
        return (swaps % 2) ? -d : d;
    }

    // Reciprocal condition number in the 1-norm, by the Hager-Higham estimator
    // (LAPACK's dlacn2 / dgecon). It needs ||A⁻¹||₁, which would ordinarily mean
    // forming the inverse; the estimator instead finds a vector that nearly
    // maximises ||A⁻¹x||₁/||x||₁ using a handful of solves against factors we
    // already have. A few solves against O(n³) for an explicit inverse.
    //
    // It is an ESTIMATE, and a lower bound on the true condition number — it can
    // be optimistic, never pessimistic. Near 1 means well conditioned; near
    // machine epsilon means the matrix is numerically singular.
    double rcond(double oneNormA) const {
        if (kind_ == Kind::QR)
            throw std::logic_error("Decomposition::rcond: only defined for a square system");
        if (oneNormA == 0.0)
            return 0.0;
        const long n = rows_;
        Matrix<W> x(n, 1);
        for (long i = 0; i < n; i++)
            x(int(i), 0) = W(1.0 / double(n));
        double est = 0.0;
        for (int iter = 0; iter < 5; iter++) {
            Matrix<W> y = solve(x);
            double ynorm = 0.0;
            for (long i = 0; i < n; i++)
                ynorm += std::abs(y(int(i), 0));
            // The real estimator takes the SIGN of y. The complex one takes its
            // PHASE, y/|y|, which is what LAPACK's zlacn2 does and what reduces
            // to +-1 when the imaginary part is zero.
            Matrix<W> xi(n, 1);
            for (long i = 0; i < n; i++) {
                const double m = std::abs(y(int(i), 0));
                xi(int(i), 0) = (m == 0.0) ? W(1) : y(int(i), 0) / W(m);
            }
            Matrix<W> z = solveTransposed(xi);
            long jmax = 0;
            for (long i = 1; i < n; i++)
                if (std::abs(z(int(i), 0)) > std::abs(z(int(jmax), 0)))
                    jmax = i;
            double ztx = 0.0;
            for (long i = 0; i < n; i++)
                ztx += std::real(cjw(z(int(i), 0)) * x(int(i), 0));
            if (iter && std::abs(z(int(jmax), 0)) <= ztx) {
                est = ynorm;
                break;
            }
            est = ynorm;
            for (long i = 0; i < n; i++)
                x(int(i), 0) = W(0);
            x(int(jmax), 0) = W(1);
        }
        if (est == 0.0)
            return 0.0;
        return 1.0 / (oneNormA * est);
    }

  private:
    Kind kind_ = Kind::LU;
    long rows_ = 0, cols_ = 0;
    std::vector<W> lu_;
    std::vector<int> piv_;
    Matrix<W> chol_;
    Matrix<W> q_, r_;

    // Conjugate for a complex matrix, identity for a real one.
    static W cjw(const W& z) {
        if constexpr (is_complex<datatype>::value) return std::conj(z);
        else return z;
    }
    std::vector<long> perm_;

    // Aᵀ x = b, for the condition estimator. Cholesky is symmetric so its
    // transpose is itself.
    Matrix<W> solveTransposed(const Matrix<W>& B) const {
        if (kind_ == Kind::Cholesky)
            return solve(B);
        return Matrix<datatype>::luSubstituteT(lu_, piv_, (int)rows_, B);
    }
};

template <typename datatype>
Decomposition<datatype> Matrix<datatype>::factorize() const {
    return Decomposition<datatype>(*this);
}

// decomposition(A) — MATLAB's spelling of A.factorize().
template <typename datatype>
Decomposition<datatype> decomposition(const Matrix<datatype>& A) {
    return Decomposition<datatype>(A);
}

// rcond(A) — reciprocal condition estimate in the 1-norm. See
// Decomposition::rcond for what the estimate is and is not.
template <typename datatype>
double rcond(const Matrix<datatype>& A) {
    return Decomposition<datatype>(A).rcond(A.norm(NormType::One));
}

// condest(A) — 1-norm condition estimate, the reciprocal of rcond.
template <typename datatype>
double condest(const Matrix<datatype>& A) {
    const double r = rcond(A);
    return r == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / r;
}

// lsqminnorm(A, b) — the minimum-norm least-squares solution.
//
// solve() throws for an under-determined system, on the grounds that
// "infinitely many solutions" is usually a mistake the caller wants told about.
// When it is not a mistake, this is the one to reach for: of all the x
// minimising ||Ax-b||, it returns the one with the smallest ||x||. Computed
// through the SVD, so it is well defined for a rank-deficient A too, which is
// exactly when it is wanted.
template <typename datatype, typename dtB>
Matrix<double> lsqminnorm(const Matrix<datatype>& A, const Matrix<dtB>& B, double tol = -1.0) {
    return A.pinv(tol) * B;
}
