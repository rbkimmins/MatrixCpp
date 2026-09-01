// Assertion-based validation for Matrix1.0.hpp.
//
// Complements test.cpp, which prints results for a human to eyeball. This one
// checks them: every case below either matches a hand-computed value or an
// identity that must hold exactly, and the process exits non-zero if any fail.
//
// Build: g++ -std=c++17 -O2 -fopenmp -o validate validate.cpp && ./validate
#include "Matrix1.0.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>

static int checks = 0, failures = 0;
static std::string suite;

static void section(const std::string& s) {
    suite = s;
    std::cout << "\n\033[1m── " << s << " ──\033[0m\n";
}

static void ok(bool cond, const std::string& what) {
    checks++;
    if (cond) { std::cout << "  \033[32mPASS\033[0m  " << what << "\n"; return; }
    failures++;
    std::cout << "  \033[31mFAIL\033[0m  " << what << "   [" << suite << "]\n";
}

// Scalar comparison with an absolute+relative tolerance.
static bool near(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol + tol * std::abs(b);
}

template <typename T>
static bool same(const Matrix<T>& A, const Matrix<T>& B, double tol = 1e-9) {
    return A.allclose(B, tol, tol);
}

// True when f() throws — used for the deliberate error paths.
static bool threwMask(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

// n x n identity as Matrix<double>.
static Matrix<double> Id(int n) {
    Matrix<double> E(n, n);
    for (int i = 0; i < n; i++) E(i, i) = 1.0;
    return E;
}

int main() {
    std::cout << std::setprecision(12);

    // ═══════════════════════════════════════════════════════════════════
    section("Operators");
    {
        Matrix<double> A(2, 2); A = {{1, 2}, {3, 4}};
        Matrix<double> B(2, 2); B = {{5, 6}, {7, 8}};

        Matrix<double> negA(2, 2); negA = {{-1, -2}, {-3, -4}};
        ok(same(-A, negA), "unary minus negates every element");

        // --- MATLAB division semantics ---
        // A / B is matrix RIGHT division: the X solving X*B = A. Element-wise
        // division moved to div() / the /dot/ spelling when this changed, so
        // these check both meanings and that they are genuinely different.
        {
            Matrix<double> X = A / B;
            ok(same(X * B, A),               "A / B is right division: (A/B)*B == A");
            ok(same(X, A * B.inverse()),     "A / B agrees with A * inv(B)");
            Matrix<double> eA(2, 2); eA = {{1.0/5, 2.0/6}, {3.0/7, 4.0/8}};
            ok(same(A.div(B), eA),          "A.div(B) is element-wise division");
            ok(same(A /dot/ B, eA),          "A /dot/ B is the same as A.div(B)");
            ok(!same(X, eA),                 "right division and A.div(B) really do differ");
            ok(same(A % B, A.mul(B)),       "A.mul(B) is a named spelling of operator%");
            ok(same(A *dot* B, A % B),       "A *dot* B is the same as A % B");
            // Precedence: * and / share a level and associate left to right, so
            // the sugar has to group as (A/dot)/B and compose with + correctly.
            ok(same(A /dot/ B + B, eA + B),  "/dot/ binds tighter than +");
            // Non-square right division falls through to least squares, exactly
            // as solve() does — X (2x3) * B3 (3x3) = A23 (2x3).
            Matrix<double> A23(2, 3); A23 = {{1, 2, 3}, {4, 5, 6}};
            Matrix<double> B3(3, 3);  B3  = {{2, 0, 1}, {1, 3, 0}, {0, 1, 4}};
            ok(same(A23 / B3 * B3, A23),     "right division works for a wide left operand");
            // /= inherits the right-division meaning.
            Matrix<double> Q = A; Q /= B;
            ok(same(Q, X),                   "A /= B is A = A / B");
        }

        Matrix<double> A2 = A;
        ok(A2 == A,  "operator== is true for an exact copy");
        ok(!(A2 != A), "operator!= is its negation");
        ok(!(A == B), "operator== is false for different contents");

        Matrix<double> wrong(2, 3);
        ok(A != wrong, "operator!= is true across mismatched shapes");
        ok(!A.allclose(wrong), "allclose is false across mismatched shapes, without throwing");

        Matrix<double> Anudge(2, 2); Anudge = {{1, 2}, {3, 4.0000000001}};
        ok(A.allclose(Anudge), "allclose tolerates a 1e-10 perturbation");
        ok(!(A == Anudge),     "operator== does not");

        Matrix<double> sum(2, 2); sum = {{6, 8}, {10, 12}};
        Matrix<double> acc = A; acc += B;
        ok(same(acc, sum), "operator+= accumulates in place");

        Matrix<double> diff(2, 2); diff = {{-4, -4}, {-4, -4}};
        Matrix<double> dec = A; dec -= B;
        ok(same(dec, diff), "operator-= subtracts in place");
        ok(same(A - B, diff), "binary operator- matches");

        const Matrix<double> cA = A, cB = B;
        ok(same(cA + cB, sum), "operator+ is callable on const operands");
        ok(same(cA - cB, diff), "operator- is callable on const operands");

        // Subtraction used to route through A + B*-1, which is wrong for
        // unsigned element types; it now subtracts directly.
        Matrix<unsigned> U(1, 2); U = {{9u, 5u}};
        Matrix<unsigned> V(1, 2); V = {{4u, 5u}};
        Matrix<unsigned> UV(1, 2); UV = {{5u, 0u}};
        ok((U - V) == UV, "operator- is correct for an unsigned datatype");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Reductions");
    {
        Matrix<double> A(2, 3); A = {{3, 1, 4}, {1, 5, 9}};

        ok(near(A.min(), 1.0), "min() over all elements");
        ok(near(A.max(), 9.0), "max() over all elements");
        ok(near(A.sum(), 23.0), "sum() over all elements");
        ok(near(A.mean(), 23.0 / 6.0), "mean() over all elements");

        Matrix<double> colMin(1, 3); colMin = {{1, 1, 4}};
        Matrix<double> rowMin(2, 1); rowMin = {{1}, {1}};
        ok(same(A.min(0), colMin), "min(0) gives per-column minima");
        ok(same(A.min(1), rowMin), "min(1) gives per-row minima");

        Matrix<double> colMax(1, 3); colMax = {{3, 5, 9}};
        Matrix<double> rowMax(2, 1); rowMax = {{4}, {9}};
        ok(same(A.max(0), colMax), "max(0) gives per-column maxima");
        ok(same(A.max(1), rowMax), "max(1) gives per-row maxima");

        Matrix<double> colMean(1, 3); colMean = {{2, 3, 6.5}};
        ok(same(A.mean(0), colMean), "mean(0) gives per-column means");

        Matrix<double> colSum(1, 3); colSum = {{4, 6, 13}};
        Matrix<double> rowSum(2, 1); rowSum = {{8}, {15}};
        ok(same(A.sum(0), colSum), "sum(0) still gives per-column sums after the rewrite");
        ok(same(A.sum(1), rowSum), "sum(1) still gives per-row sums after the rewrite");

        // var: mean 23/6; population variance computed by hand.
        double mu = 23.0 / 6.0, ss = 0.0;
        double vals[] = {3, 1, 4, 1, 5, 9};
        for (double v : vals) ss += (v - mu) * (v - mu);
        ok(near(A.var(false), ss / 6.0), "var(population) divides by N");
        ok(near(A.var(true),  ss / 5.0), "var(sample) divides by N-1");
        ok(near(A.stddev(), std::sqrt(ss / 6.0)), "stddev is the root of var");

        ok(A.argmin() == std::make_pair(0L, 1L), "argmin gives {row,col} of the first minimum");
        ok(A.argmax() == std::make_pair(1L, 2L), "argmax gives {row,col} of the maximum");

        // Integral matrices must not truncate their mean.
        Matrix<int> Ai(1, 2); Ai = {{1, 2}};
        ok(near(Ai.mean(), 1.5), "mean() of an int matrix returns 1.5, not 1");

        bool threw = false;
        try { Matrix<double>().min(); } catch (const std::exception&) { threw = true; }
        ok(threw, "min() on an empty matrix throws instead of reading off the end");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Structural");
    {
        Matrix<double> A(3, 3); A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        Matrix<double> d(3, 1); d = {{1}, {5}, {9}};
        ok(same(A.diag(), d), "diag() extracts the main diagonal as a column");

        Matrix<double> R(2, 3); R = {{1, 2, 3}, {4, 5, 6}};
        Matrix<double> rd(2, 1); rd = {{1}, {5}};
        ok(same(R.diag(), rd), "diag() works on a non-square matrix");

        Matrix<double> up(3, 3); up = {{1, 2, 3}, {0, 5, 6}, {0, 0, 9}};
        ok(same(A.triu(), up), "triu() keeps the upper triangle");

        Matrix<double> up1(3, 3); up1 = {{0, 2, 3}, {0, 0, 6}, {0, 0, 0}};
        ok(same(A.triu(1), up1), "triu(1) moves above the main diagonal");

        Matrix<double> lo(3, 3); lo = {{1, 0, 0}, {4, 5, 0}, {7, 8, 9}};
        ok(same(A.tril(), lo), "tril() keeps the lower triangle");

        Matrix<double> lom1(3, 3); lom1 = {{0, 0, 0}, {4, 0, 0}, {7, 8, 0}};
        ok(same(A.tril(-1), lom1), "tril(-1) moves below the main diagonal");
        ok(same(A.triu() + A.tril(-1), A), "triu() + tril(-1) reassembles the matrix");

        Matrix<double> flat(1, 9); flat = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        ok(same(A.reshape(1, 9), flat), "reshape() reads row-major");
        ok(same(A.reshape(9, 1).reshape(3, 3), A), "reshape() round-trips");

        bool threw = false;
        try { A.reshape(2, 2); } catch (const std::exception&) { threw = true; }
        ok(threw, "reshape() rejects an element-count mismatch");

        Matrix<double> v(3, 1); v = {{2}, {3}, {4}};
        Matrix<double> dm(3, 3); dm = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
        ok(same(diag(v), dm), "free diag(v) builds a diagonal matrix from a vector");
        ok(same(diag(A.diag()), Matrix<double>(A.triu().tril())), "diag(A.diag()) round-trips the diagonal");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Complex support: H / conj / real / imag");
    {
        using cd = std::complex<double>;
        Matrix<cd> Z(2, 2);
        Z = {{cd(1, 2), cd(3, 4)}, {cd(5, 6), cd(7, 8)}};

        Matrix<cd> Zc(2, 2);
        Zc = {{cd(1, -2), cd(3, -4)}, {cd(5, -6), cd(7, -8)}};
        ok(Z.conj() == Zc, "conj() negates every imaginary part");

        Matrix<cd> Zh(2, 2);
        Zh = {{cd(1, -2), cd(5, -6)}, {cd(3, -4), cd(7, -8)}};
        ok(Z.H() == Zh, "H() is conjugate-transpose, not plain transpose");
        ok(!(Z.H() == Z.T()), "H() and T() genuinely differ for a complex matrix");

        Matrix<double> Zr(2, 2); Zr = {{1, 3}, {5, 7}};
        Matrix<double> Zi(2, 2); Zi = {{2, 4}, {6, 8}};
        ok(same(Z.real(), Zr), "real() extracts real parts");
        ok(same(Z.imag(), Zi), "imag() extracts imaginary parts");

        // Degrades gracefully on real matrices.
        Matrix<double> A(2, 2); A = {{1, 2}, {3, 4}};
        ok(same(A.conj(), A),      "conj() is the identity for a real datatype");
        ok(same(A.H(), A.T()),     "H() collapses to T() for a real datatype");
        ok(same(A.real(), A),      "real() is the identity for a real datatype");
        ok(same(A.imag(), Matrix<double>(2, 2)), "imag() is all zeros for a real datatype");

        // real_of keeps the element type honest rather than forcing double.
        static_assert(std::is_same<real_t<std::complex<float>>, float>::value,
                      "real_of<complex<float>> must be float");
        static_assert(std::is_same<real_t<double>, double>::value,
                      "real_of<double> must be double");
        ok(true, "real_of trait maps complex<float> to float, double to double");

        // The container layer must still work end-to-end for complex.
        Matrix<cd> P = Z * Z;
        ok(near(std::abs(P(0,0) - (Z(0,0)*Z(0,0) + Z(0,1)*Z(1,0))), 0.0, 1e-12),
           "complex matrix multiplication still works");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Norms, rank, condition number");
    {
        Matrix<double> A(2, 2); A = {{3, -4}, {0, 0}};
        ok(near(A.norm(NormType::Fro), 5.0), "Frobenius norm of [[3,-4],[0,0]] is 5");
        ok(near(A.norm(NormType::One), 4.0), "One norm is the max absolute column sum");
        ok(near(A.norm(NormType::Inf), 7.0), "Inf norm is the max absolute row sum");
        ok(near(A.norm(NormType::Two), 5.0, 1e-8), "Two norm equals the largest singular value");

        // Overflow safety: the naive sum of squares would overflow here.
        Matrix<double> Big(1, 2); Big = {{3e200, 4e200}};
        ok(near(Big.norm(NormType::Fro), 5e200, 1e-6),
           "Frobenius norm survives entries near the overflow boundary");

        Matrix<double> Full(3, 3); Full = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        ok(Full.rank() == 3, "rank() of a non-singular 3x3 is 3");

        Matrix<double> Def(3, 3); Def = {{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};
        ok(Def.rank() == 2, "rank() detects a linearly dependent row");

        Matrix<double> Zero(3, 3);
        ok(Zero.rank() == 0, "rank() of the zero matrix is 0");

        Matrix<double> Wide(2, 4); Wide = {{1, 0, 1, 0}, {0, 1, 0, 1}};
        ok(Wide.rank() == 2, "rank() works on a wide matrix");

        ok(near(Id(4).cond(NormType::Two), 1.0, 1e-9), "cond of the identity is 1");
        Matrix<double> D(2, 2); D = {{100, 0}, {0, 1}};
        ok(near(D.cond(NormType::Two), 100.0, 1e-8), "cond(Two) is sigma_max / sigma_min");
        ok(near(D.cond(NormType::One), 100.0, 1e-8), "cond(One) matches for a diagonal matrix");
        // Threshold rather than std::isinf: -ffast-math implies -ffinite-math-only,
        // under which isinf() folds to a constant false. cond() does return
        // infinity either way; only the test spelling has to survive the flag.
        double cSing = Def.cond(NormType::Two);
        ok(!(cSing < 1e15), "cond of a singular matrix blows up (infinite, or beyond 1e15 "
                            "under -ffast-math where isinf is unusable)");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Cholesky");
    {
        Matrix<double> A(3, 3);
        A = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
        Matrix<double> L = A.cholesky();

        Matrix<double> expected(3, 3);
        expected = {{2, 0, 0}, {6, 1, 0}, {-8, 5, 3}};
        ok(same(L, expected, 1e-9), "cholesky matches the textbook factor for the standard example");
        ok(same(L * L.T(), A, 1e-9), "L * L^T reconstructs A");
        ok(same(L.triu(1), Matrix<double>(3, 3)), "the returned factor is lower triangular");

        bool threw = false;
        try { Matrix<double> N(2,2); N = {{1, 2}, {2, 1}}; N.cholesky(); }
        catch (const std::exception&) { threw = true; }
        ok(threw, "cholesky throws on a symmetric but indefinite matrix");

        threw = false;
        try { Matrix<double> N(2,2); N = {{1, 2}, {3, 4}}; N.cholesky(); }
        catch (const std::exception&) { threw = true; }
        ok(threw, "cholesky throws on a non-symmetric matrix");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("solve / inverse");
    {
        Matrix<double> A(3, 3); A = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
        Matrix<double> b(3, 1); b = {{8}, {-11}, {-3}};
        Matrix<double> x = A.solve(b);

        Matrix<double> expected(3, 1); expected = {{2}, {3}, {-1}};
        ok(same(x, expected, 1e-9), "solve finds the known solution of a 3x3 system");
        ok(same(A * x, b, 1e-9), "A * x reproduces b");

        Matrix<double> inv = A.inverse();
        ok(same(A * inv, Id(3), 1e-9), "A * inverse(A) is the identity");
        ok(same(inv * A, Id(3), 1e-9), "inverse(A) * A is the identity");
        ok(same(A.solve(Id(3)), inv, 1e-9), "inverse() agrees with solve(I) after the refactor");

        // Multiple right-hand sides in one call.
        Matrix<double> B = b | (b * 2.0);
        Matrix<double> X = A.solve(B);
        ok(X.cols() == 2 && same(A * X, B, 1e-9), "solve handles several right-hand sides at once");

        // 1x1, which luPacked() refuses.
        Matrix<double> One(1, 1); One = {{4}};
        Matrix<double> rhs(1, 1); rhs = {{8}};
        ok(near(One.solve(rhs)(0, 0), 2.0), "solve handles the 1x1 case");

        // Mixed element types.
        Matrix<int> Ai(2, 2); Ai = {{2, 0}, {0, 4}};
        Matrix<double> bi(2, 1); bi = {{2}, {8}};
        Matrix<double> xi = Ai.solve(bi);
        ok(near(xi(0,0), 1.0) && near(xi(1,0), 2.0), "solve accepts an int matrix with a double rhs");

        // Over-determined least squares: fit y = 1 + x through (0,1),(1,2),(2,3).
        Matrix<double> M(3, 2); M = {{1, 0}, {1, 1}, {1, 2}};
        Matrix<double> y(3, 1); y = {{1}, {2}, {3}};
        Matrix<double> beta = M.solve(y);
        ok(near(beta(0,0), 1.0, 1e-9) && near(beta(1,0), 1.0, 1e-9),
           "solve returns the least-squares fit for an over-determined system");

        // With noise the residual must be orthogonal to the column space.
        Matrix<double> yn(3, 1); yn = {{1}, {2.5}, {3}};
        Matrix<double> bn = M.solve(yn);
        Matrix<double> resid = M * bn - yn;
        ok(near((M.T() * resid).norm(NormType::Fro), 0.0, 1e-9),
           "the least-squares residual is orthogonal to the column space");

        bool threw = false;
        try { Matrix<double> W(2, 3); Matrix<double> r(2,1); W.solve(r); }
        catch (const std::exception&) { threw = true; }
        ok(threw, "solve refuses an under-determined system rather than guessing");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("SVD");
    {
        auto checkSVD = [&](const Matrix<double>& A, const std::string& label) {
            auto [U, S, V] = A.svd();
            long m = A.rows(), n = A.cols();
            ok(U.rows() == m && U.cols() == m && S.rows() == m && S.cols() == n &&
               V.rows() == n && V.cols() == n, label + ": factor shapes are m*m, m*n, n*n");
            ok(same(U * S * V.T(), A, 1e-8), label + ": U * S * V^T reconstructs A");
            ok(same(U.T() * U, Id((int)m), 1e-8), label + ": U is orthogonal");
            ok(same(V.T() * V, Id((int)n), 1e-8), label + ": V is orthogonal");
            bool sorted = true, nonneg = true, offZero = true;
            for (long i = 0; i < std::min(m, n); i++) {
                if (S(int(i), int(i)) < -1e-12) nonneg = false;
                if (i && S(int(i), int(i)) > S(int(i-1), int(i-1)) + 1e-12) sorted = false;
            }
            for (long i = 0; i < m; i++)
                for (long j = 0; j < n; j++)
                    if (i != j && std::abs(S(int(i), int(j))) > 1e-12) offZero = false;
            ok(sorted && nonneg && offZero,
               label + ": singular values are non-negative, descending, and diagonal");
        };

        Matrix<double> Sq(3, 3); Sq = {{4, 0, 1}, {0, 3, 0}, {1, 0, 4}};
        checkSVD(Sq, "square");

        Matrix<double> Tall(4, 2); Tall = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        checkSVD(Tall, "tall");

        Matrix<double> Wide(2, 4); Wide = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        checkSVD(Wide, "wide");

        Matrix<double> Sing(3, 3); Sing = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
        checkSVD(Sing, "rank-1 singular");

        // Known singular values: diag(3,1) has exactly {3,1}.
        Matrix<double> D(2, 2); D = {{3, 0}, {0, 1}};
        auto [U2, S2, V2] = D.svd();
        ok(near(S2(0,0), 3.0, 1e-10) && near(S2(1,1), 1.0, 1e-10),
           "singular values of diag(3,1) are exactly 3 and 1");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("pinv / adjugate");
    {
        Matrix<double> A(3, 2); A = {{1, 0}, {0, 1}, {1, 1}};
        Matrix<double> Ap = A.pinv();
        ok(same(A * Ap * A, A, 1e-8),    "Moore-Penrose 1: A A+ A == A");
        ok(same(Ap * A * Ap, Ap, 1e-8),  "Moore-Penrose 2: A+ A A+ == A+");
        ok(same((A * Ap).T(), A * Ap, 1e-8),   "Moore-Penrose 3: (A A+) is symmetric");
        ok(same((Ap * A).T(), Ap * A, 1e-8),   "Moore-Penrose 4: (A+ A) is symmetric");

        Matrix<double> Inv(3, 3); Inv = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
        ok(same(Inv.pinv(), Inv.inverse(), 1e-8),
           "pinv coincides with inverse for a non-singular square matrix");

        Matrix<double> Sing(2, 2); Sing = {{1, 1}, {1, 1}};
        Matrix<double> Sp = Sing.pinv();
        ok(same(Sing * Sp * Sing, Sing, 1e-8), "pinv is defined for a singular matrix");

        // adjugate: A * adj(A) == det(A) * I
        Matrix<double> B(3, 3); B = {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
        Matrix<double> adjB = B.adjugate();
        ok(same(B * adjB, Id(3) * B.det(), 1e-7), "A * adj(A) == det(A) * I");

        Matrix<double> known(3, 3); known = {{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
        ok(same(adjB, known, 1e-7), "adjugate matches the hand-computed cofactor matrix");

        // Singular case goes through the cofactor route.
        Matrix<double> S2(3, 3); S2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        Matrix<double> adjS = S2.adjugate();
        ok(same(S2 * adjS, Matrix<double>(3, 3), 1e-7),
           "A * adj(A) == 0 for a singular A, via the cofactor fallback");

        Matrix<double> two(2, 2); two = {{1, 2}, {3, 4}};
        Matrix<double> adj2(2, 2); adj2 = {{4, -2}, {-3, 1}};
        ok(same(two.adjugate(), adj2, 1e-9), "adjugate of a 2x2 swaps and negates correctly");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Eigenvalues: the real-Schur 2x2 block");
    {
        // Symmetric: eigenvalues of [[2,1],[1,2]] are 3 and 1.
        Matrix<double> Sym(2, 2); Sym = {{2, 1}, {1, 2}};
        auto [ev, Q] = Sym.eig();
        double a = ev(0,0), b = ev(1,0);
        ok((near(a, 3.0, 1e-8) && near(b, 1.0, 1e-8)) ||
           (near(a, 1.0, 1e-8) && near(b, 3.0, 1e-8)),
           "eig() gives {3,1} for a symmetric matrix");
        ok(same(Q.T() * Q, Id(2), 1e-8), "eig() returns an orthogonal Q");

        // A rotation matrix has eigenvalues e^{+-i*theta} and no real ones.
        double th = 0.7;
        Matrix<double> R(2, 2);
        R = {{std::cos(th), -std::sin(th)}, {std::sin(th), std::cos(th)}};

        bool threw = false;
        try { R.eig(); } catch (const std::exception&) { threw = true; }
        ok(threw, "eig() now THROWS on a rotation matrix instead of silently "
                  "returning the real part twice");

        auto lam = R.eigvals();
        ok(lam.rows() == 2 && lam.cols() == 1, "eigvals() returns an n x 1 complex column");
        bool matched =
            near(lam(0,0).real(), std::cos(th), 1e-8) &&
            near(std::abs(lam(0,0).imag()), std::sin(th), 1e-8) &&
            near(lam(1,0).real(), std::cos(th), 1e-8) &&
            near(lam(0,0).imag(), -lam(1,0).imag(), 1e-8);
        ok(matched, "eigvals() recovers the conjugate pair cos(t) +- i*sin(t)");
        ok(near(std::abs(lam(0,0)), 1.0, 1e-8),
           "eigvals() of a rotation lie on the unit circle");

        // Symmetric input through eigvals(): imaginary parts must all vanish.
        auto slam = Sym.eigvals();
        ok(near(slam(0,0).imag(), 0.0, 1e-10) && near(slam(1,0).imag(), 0.0, 1e-10),
           "eigvals() of a symmetric matrix is purely real");

        // Trace and determinant are the sum and product of the eigenvalues.
        Matrix<double> G(3, 3); G = {{0, -1, 0}, {1, 0, 0}, {0, 0, 5}};
        auto glam = G.eigvals();
        std::complex<double> tsum = glam(0,0) + glam(1,0) + glam(2,0);
        std::complex<double> pprod = glam(0,0) * glam(1,0) * glam(2,0);
        ok(near(tsum.real(), G.tr(), 1e-8) && near(tsum.imag(), 0.0, 1e-8),
           "eigvals() sum to the trace, mixed real and complex spectrum");
        ok(near(pprod.real(), G.det(), 1e-8) && near(pprod.imag(), 0.0, 1e-8),
           "eigvals() multiply to the determinant");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Matrix exponential");
    {
        ok(same(exp(Matrix<double>(3, 3)), Id(3), 1e-12), "exp(0) == I");

        Matrix<double> D(2, 2); D = {{2, 0}, {0, 3}};
        Matrix<double> eD(2, 2); eD = {{std::exp(2.0), 0}, {0, std::exp(3.0)}};
        ok(same(exp(D), eD, 1e-9), "exp of a diagonal matrix exponentiates the diagonal");

        // Nilpotent: N^2 == 0, so exp(N) == I + N exactly.
        Matrix<double> N(2, 2); N = {{0, 1}, {0, 0}};
        Matrix<double> eN(2, 2); eN = {{1, 1}, {0, 1}};
        ok(same(exp(N), eN, 1e-12), "exp of a nilpotent matrix terminates at I + N");

        // exp of the rotation generator is the rotation matrix.
        double th = 0.9;
        Matrix<double> J(2, 2); J = {{0, -th}, {th, 0}};
        Matrix<double> R(2, 2);
        R = {{std::cos(th), -std::sin(th)}, {std::sin(th), std::cos(th)}};
        ok(same(exp(J), R, 1e-10), "exp of the rotation generator is the rotation matrix");

        // det(exp(A)) == exp(tr(A)) for every square A.
        Matrix<double> A(3, 3); A = {{0.3, -1.2, 0.5}, {0.7, 0.1, -0.4}, {-0.2, 0.9, 0.6}};
        ok(near(exp(A).det(), std::exp(A.tr()), 1e-7), "det(exp(A)) == exp(tr(A))");

        // exp(A) * exp(-A) == I.
        ok(same(exp(A) * exp(-A), Id(3), 1e-9), "exp(A) * exp(-A) == I");

        // Commuting matrices: exp(A+B) == exp(A) exp(B).
        Matrix<double> B = A * 2.0;
        ok(same(exp(A + B), exp(A) * exp(B), 1e-8),
           "exp(A+B) == exp(A)exp(B) for commuting A and B");

        // Large norm: this is what the unscaled series cannot survive.
        Matrix<double> Big = A * 30.0;
        Matrix<double> eBig = exp(Big);
        bool finite = true;
        for (long i = 0; i < eBig.rows() * eBig.cols(); i++)
            if (!std::isfinite(eBig[int(i)])) finite = false;
        ok(finite, "exp stays finite at ||A|| ~ 60 (no NaN from factorial overflow)");
        ok(near(eBig.det(), std::exp(Big.tr()), std::abs(std::exp(Big.tr())) * 1e-6),
           "det(exp(A)) == exp(tr(A)) still holds at ||A|| ~ 60");
        ok(same(exp(Big) * exp(-Big), Id(3), 1e-6), "exp(A)exp(-A) == I at ||A|| ~ 60");

        // Ground truth for a symmetric matrix: A = Q diag(l) Q^T exactly, so
        // exp(A) = Q diag(e^l) Q^T. This checks exp against a value derived a
        // completely different way, at norms where scaling is doing real work.
        {
            Matrix<double> S(6, 6); S.set_Ran_values(-1.0, 1.0, -4242);
            S = (S + S.T()) * 0.5;
            for (double sc : {1.0, 20.0, 100.0, 300.0}) {
                Matrix<double> Bs = S * sc;
                auto [ev, Qe] = Bs.eig();
                Matrix<double> Dg(6, 6);
                for (int i = 0; i < 6; i++) Dg(i, i) = std::exp(ev(i, 0));
                Matrix<double> exact = Qe * Dg * Qe.T();
                Matrix<double> got = exp(Bs);
                double num = 0.0, den = 0.0;
                for (long i = 0; i < 36; i++) {
                    num = std::max(num, std::abs(got[int(i)] - exact[int(i)]));
                    den = std::max(den, std::abs(exact[int(i)]));
                }
                ok(den > 0.0 && num / den < 1e-12,
                   "exp matches the exact eigendecomposition to 1e-12 relative at ||A||inf ~ "
                   + std::to_string((long)Bs.norm(NormType::Inf)));
            }
        }

        // Element-wise and matrix-wise must NOT agree except on diagonals.
        ok(!same(exp(A), A.exp(), 1e-6), "exp(A) differs from A.exp()");
        ok(same(exp(D), D.exp().triu().tril(), 1e-9),
           "exp(A) and A.exp() do agree on the diagonal of a diagonal matrix");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Matrix trigonometry");
    {
        Matrix<double> A(3, 3); A = {{0.4, -0.9, 0.2}, {0.6, 0.3, -0.7}, {-0.1, 0.8, 0.5}};

        ok(same(sin(Matrix<double>(3, 3)), Matrix<double>(3, 3), 1e-12), "sin(0) == 0");
        ok(same(cos(Matrix<double>(3, 3)), Id(3), 1e-12), "cos(0) == I");

        ok(same(sin(A) * sin(A) + cos(A) * cos(A), Id(3), 1e-9),
           "sin(A)^2 + cos(A)^2 == I");
        ok(same(cosh(A) * cosh(A) - sinh(A) * sinh(A), Id(3), 1e-9),
           "cosh(A)^2 - sinh(A)^2 == I");

        // Parity.
        ok(same(sin(-A), -sin(A), 1e-9),  "sin is odd");
        ok(same(cos(-A), cos(A), 1e-9),   "cos is even");
        ok(same(sinh(-A), -sinh(A), 1e-9), "sinh is odd");
        ok(same(cosh(-A), cosh(A), 1e-9),  "cosh is even");

        // Hyperbolic pair against the exponential definition.
        ok(same(cosh(A), (exp(A) + exp(-A)) * 0.5, 1e-9), "cosh(A) == (exp(A)+exp(-A))/2");
        ok(same(sinh(A), (exp(A) - exp(-A)) * 0.5, 1e-9), "sinh(A) == (exp(A)-exp(-A))/2");

        // tan and tanh are SOLVES, not element-wise divisions.
        ok(same(cos(A) * tan(A), sin(A), 1e-8),   "cos(A) * tan(A) == sin(A)");
        ok(same(cosh(A) * tanh(A), sinh(A), 1e-8), "cosh(A) * tanh(A) == sinh(A)");
        // Now that / is matrix right division, sin(A)/cos(A) IS tan(A) — the two
        // commute, being functions of the same A, so left and right division
        // agree. It is still the element-wise spelling that is wrong, and that is
        // what this pins down. (Before / changed meaning, this assertion ran the
        // other way round.)
        ok(same(tan(A), sin(A) / cos(A), 1e-6),
           "tan(A) == sin(A) / cos(A) under right division");
        ok(!same(tan(A), sin(A).div(cos(A)), 1e-6),
           "tan(A) is NOT the element-wise sin(A) ./ cos(A)");

        // Diagonal: matrix and element-wise trig must coincide.
        Matrix<double> D(3, 3); D = {{0.5, 0, 0}, {0, -1.1, 0}, {0, 0, 2.2}};
        ok(same(sin(D), D.sin().triu().tril(), 1e-10),
           "sin(D) equals element-wise sin on the diagonal of a diagonal matrix");
        ok(!same(sin(A), A.sin(), 1e-6), "sin(A) differs from A.sin() in general");

        // THE case the roadmap said Schur-based trig could not handle: a matrix
        // with purely imaginary eigenvalues. J = [[0,-t],[t,0]] has J^2 = -t^2 I,
        // so the series collapse to closed form: cos(J) = cosh(t) I and
        // sin(J) = (sinh(t)/t) J.
        double t = 0.8;
        Matrix<double> J(2, 2); J = {{0, -t}, {t, 0}};
        ok(same(cos(J), Id(2) * std::cosh(t), 1e-10),
           "cos(J) == cosh(t)*I for the rotation generator (complex eigenvalues)");
        ok(same(sin(J), J * (std::sinh(t) / t), 1e-10),
           "sin(J) == (sinh(t)/t)*J for the rotation generator");
        ok(same(cosh(J), Id(2) * std::cos(t), 1e-10), "cosh(J) == cos(t)*I");

        // A rotation matrix itself — also complex-eigenvalued.
        double th = 1.3;
        Matrix<double> R(2, 2);
        R = {{std::cos(th), -std::sin(th)}, {std::sin(th), std::cos(th)}};
        ok(same(sin(R) * sin(R) + cos(R) * cos(R), Id(2), 1e-9),
           "sin^2 + cos^2 == I for a rotation matrix");

        // Large norm, well-conditioned: a symmetric matrix has real eigenvalues,
        // so sin and cos stay bounded by 1 and the identity holds to full accuracy.
        Matrix<double> Sym(3, 3); Sym = {{6, 2, 1}, {2, -5, 3}, {1, 3, 8}};
        Matrix<double> BigSym = Sym * 4.0;   // ||.||inf = 48
        ok(same(sin(BigSym) * sin(BigSym) + cos(BigSym) * cos(BigSym), Id(3), 1e-10),
           "sin^2 + cos^2 == I at ||A|| ~ 48, symmetric");
        ok(sin(BigSym).abs().max() <= 1.0 + 1e-10,
           "sin of a symmetric matrix stays bounded by 1, as it must");

        // Large norm with COMPLEX eigenvalues. Here sin(A) and cos(A) genuinely
        // grow like e^|Im(lambda)| — about 1e11 for this matrix — so sin^2+cos^2
        // asks for ~1e22 of cancellation to land on I. The absolute residual is
        // therefore large no matter how it is computed; what has to be small is
        // the error RELATIVE to the intermediate magnitudes, and that is at
        // machine epsilon. Asserting a small absolute residual here would be
        // asserting something double precision cannot represent.
        Matrix<double> Big = A * 25.0;
        Matrix<double> S = sin(Big), C = cos(Big);
        double mag = S.abs().max() * S.abs().max();
        double resid = (S * S + C * C - Id(3)).norm(NormType::Fro);
        ok(resid / mag < 1e-13,
           "sin^2 + cos^2 == I at ||A|| ~ 40 to full RELATIVE accuracy "
           "(complex spectrum; absolute residual is bounded below by conditioning)");
        bool finite = true;
        for (long i = 0; i < 9; i++)
            if (!std::isfinite(S[int(i)]) || !std::isfinite(C[int(i)])) finite = false;
        ok(finite, "sin/cos stay finite at ||A|| ~ 40 with a complex spectrum");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("TaylorOpts: caller-supplied series limits");
    {
        Matrix<double> A(2, 2); A = {{0.1, 0.2}, {-0.15, 0.05}};

        // Scaling off and tol 0 forces exactly maxTerms terms, so the result is a
        // partial sum we can predict term by term.
        Matrix<double> one   = exp(A, {1, 0.0, false});
        Matrix<double> two   = exp(A, {2, 0.0, false});
        Matrix<double> three = exp(A, {3, 0.0, false});
        ok(same(one, Id(2) + A, 1e-12), "exp(A, {1,0,false}) == I + A");
        ok(same(two, Id(2) + A + (A * A) * 0.5, 1e-12),
           "exp(A, {2,0,false}) == I + A + A^2/2");
        ok(same(three, two + (A * A * A) * (1.0 / 6.0), 1e-12),
           "exp(A, {3,0,false}) adds exactly the A^3/6 term");

        // A truncated series is a worse approximation than the converged one.
        Matrix<double> full = exp(A);
        double e2 = (two - full).norm(NormType::Fro);
        double e3 = (three - full).norm(NormType::Fro);
        ok(e3 < e2, "adding a term moves the partial sum closer to the converged value");
        ok(e2 > 1e-12, "a 2-term truncation is measurably short of converged");

        // Enough terms and it agrees with the default.
        ok(same(exp(A, 60), full, 1e-12), "exp(A, 60) matches the default result");
        ok(same(exp(A, {60, 1e-14}), full, 1e-12), "the {terms, tol} form matches too");

        // The integer form must actually bind, not silently take the default.
        ok(!same(exp(A, {1, 0.0, false}), full, 1e-6),
           "a 1-term limit really does truncate rather than falling back to the default");

        // Same controls on the trig functions.
        ok(same(sin(A, {1, 0.0, false}), A - (A * A * A) * (1.0 / 6.0), 1e-12),
           "sin(A, {1,0,false}) == A - A^3/6");
        ok(same(cos(A, {1, 0.0, false}), Id(2) - (A * A) * 0.5, 1e-12),
           "cos(A, {1,0,false}) == I - A^2/2");
        ok(same(cosh(A, {1, 0.0, false}), Id(2) + (A * A) * 0.5, 1e-12),
           "cosh(A, {1,0,false}) == I + A^2/2");
        ok(same(sinh(A, {1, 0.0, false}), A + (A * A * A) * (1.0 / 6.0), 1e-12),
           "sinh(A, {1,0,false}) == A + A^3/6");
        ok(same(sin(A, 60), sin(A), 1e-12), "sin(A, 60) matches the default result");

        // Scaling is what rescues a large-norm argument; without it, a short
        // series is visibly wrong, and the library default is not.
        Matrix<double> Big = A * 200.0;
        Matrix<double> noScale = exp(Big, {20, 0.0, false});
        Matrix<double> scaled  = exp(Big);
        ok(!same(noScale, scaled, 1e-3),
           "disabling scaling with 20 terms gives a visibly different answer at large ||A||");
        ok(near(scaled.det(), std::exp(Big.tr()), std::abs(std::exp(Big.tr())) * 1e-6),
           "the scaled default is the one that satisfies det(exp(A)) == exp(tr(A))");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Regression: existing behaviour still intact");
    {
        Matrix<double> A(3, 3); A = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
        ok(near(A.det(), -1.0, 1e-9), "det() unchanged");

        auto [L, U, P] = A.LU();
        ok(same(P * A, L * U, 1e-9), "LU: P*A == L*U");

        auto [Q, R, Pq] = A.QR();
        ok(same(A * Pq, Q * R, 1e-9), "QR: A*P == Q*R");
        ok(same(Q.T() * Q, Id(3), 1e-9), "QR: Q is orthogonal");

        auto [T, Qs] = A.schurDecomp();
        Matrix<double> Tm(3, 3), Qm(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) { Tm(i, j) = T[i*3+j]; Qm(i, j) = Qs[i*3+j]; }
        ok(same(Qm * Tm * Qm.T(), A, 1e-8), "Schur: Q*T*Q^T == A");

        ok(same(pow(A, 3), A * A * A, 1e-8), "pow(A,3) == A*A*A");
        ok(same(pow(A, -1), A.inverse(), 1e-8), "pow(A,-1) == inverse(A)");

        Matrix<double> SPD(2, 2); SPD = {{4, 1}, {1, 3}};
        ok(same(pow(SPD, 0.5) * pow(SPD, 0.5), SPD, 1e-7), "pow(A,0.5) squares back to A");
        ok(same(sqrt(SPD), pow(SPD, 0.5), 1e-9), "sqrt(A) == pow(A, 0.5)");

        Matrix<double> E = exp(SPD);
        ok(same(log(E, M_E), SPD, 1e-7), "log(exp(A), e) round-trips");

        // operator* picks one of three paths by size, and they must all agree with
        // a hand-rolled reference product:
        //   n < 64                        naive
        //   n == 2^k                      Strassen-Winograd directly
        //   2/3*nextPow2(n) <= n < 2^k    zero-padded to 2^k, then Strassen
        //   otherwise                     naive (padding overhead too high)
        // n=70 alone does NOT exercise Strassen — nextPow2(70)=128 exceeds the
        // padding budget, so it falls through to naive. The recursion first runs
        // for real at n=128. Every class is covered here for that reason.
        for (long n : {63L, 64L, 86L, 100L, 128L, 129L, 171L, 256L}) {
            Matrix<double> Bg(n, n); Bg.set_Ran_values(-1.0, 1.0, -12345);
            Matrix<double> Cg(n, n); Cg.set_Ran_values(-1.0, 1.0, -54321);
            Matrix<double> prod = Bg * Cg;
            Matrix<double> ref(n, n);
            for (long i = 0; i < n; i++)
                for (long j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (long k = 0; k < n; k++) acc += Bg(int(i), int(k)) * Cg(int(k), int(j));
                    ref(int(i), int(j)) = acc;
                }
            ok(prod.allclose(ref, 1e-10, 1e-9),
               "operator* agrees with a reference product at n=" + std::to_string(n));
        }

        // B^T*B is symmetric by construction — the property that first exposed the
        // Strassen bug, at a size that routes through the padded path.
        Matrix<double> Bs(86, 86); Bs.set_Ran_values(-3.0, 3.0, -613);
        Matrix<double> Sq = Bs.T() * Bs;
        ok(Sq.allclose(Sq.T(), 1e-10, 1e-9),
           "B^T * B comes back symmetric at n=86 (padded Strassen path)");
        Matrix<double> Spd = Sq;
        for (int k = 0; k < 86; k++) Spd(k, k) += 1.0;
        bool posEig = true;
        { auto [evv, Qv] = Spd.eig(); (void)Qv;
          for (int k = 0; k < 86; k++) if (evv(k, 0) <= 0.0) posEig = false; }
        ok(posEig, "B^T*B + I has all-positive eigenvalues at n=86, as the maths requires");

        // Free-function spellings.
        ok(near(norm(A), A.norm()), "free norm(A) matches the member");
        ok(rank(A) == A.rank(), "free rank(A) matches the member");
        ok(same(inverse(A), A.inverse(), 1e-12), "free inverse(A) matches the member");
        Matrix<double> b(3,1); b = {{8},{-11},{-3}};
        ok(same(solve(A, b), A.solve(b), 1e-12), "free solve(A,b) matches the member");

        // kron() — renamed from tensor() once Tensor.hpp introduced a TYPE of
        // that name. MATLAB, NumPy and SciPy all spell it kron.
        Matrix<double> K1(2,2); K1 = {{1, 2}, {3, 4}};
        Matrix<double> K2(2,2); K2 = {{0, 5}, {6, 7}};
        Matrix<double> Kexp(4,4);
        Kexp = {{ 0,  5,  0, 10},
                { 6,  7, 12, 14},
                { 0, 15,  0, 20},
                {18, 21, 24, 28}};
        ok(same(K1.kron(K2), Kexp),          "kron() is the Kronecker product");
        ok(same(kron(K1, K2), K1.kron(K2)),  "free kron(A,B) matches the member");
        ok(K1.kron(K2).rows() == 4 && K1.kron(K2).cols() == 4,
                                             "kron of two 2x2 is 4x4");
    }

    // ═══════════════════════════════════════════════════════════════════
    // TIER 6 — WORK IN PROGRESS. Tests go here as each function lands.
    // The contracts and design notes are in the tier 6 block in Matrix1.0.hpp.
    //
    // Useful shapes for these, from what the rest of this file already does:
    //   * Round trips beat reference values: ifft(fft(x)) == x catches
    //     normalisation, bit-reversal and twiddle-sign errors at once, and
    //     deconv(conv(a,b), a) == {b, 0} does the same for polynomial division.
    //   * roots(poly(r)) == r ties tier 6 back to what already works.
    //   * conv against an explicit double loop, the way the contraction tests
    //     check tensordot — a hand-rolled reference is worth more than a
    //     hand-computed constant, because it keeps working when the case grows.
    //   * trapz of a straight line is exact; gradient of a straight line is
    //     constant, INCLUDING at the one-sided ends. That end behaviour is the
    //     whole difference between gradient and diff, so it deserves a test.
    //   * threwMask([]{ ... }) is the helper for the deliberate error paths.
    section("Signal, calculus and interpolation (tier 6)");
    {
        using cplx = std::complex<double>;
        auto maxdiff = [](const Matrix<cplx>& X, const Matrix<cplx>& Y) {
            double m = 0.0;
            for (long i = 0; i < X.numel(); i++)
                m = std::max(m, std::abs(X[int(i)] - Y[int(i)]));
            return m;
        };
        auto toC = [](const Matrix<double>& R) {
            Matrix<cplx> C(R.rows(), R.cols());
            for (long i = 0; i < R.numel(); i++) C[int(i)] = cplx(R[int(i)], 0.0);
            return C;
        };

        // --- a hand-computable case ---
        Matrix<double> x(1, 8); x = {{1, 2, 3, 4, 5, 6, 7, 8}};
        Matrix<cplx> X = fft(x);
        ok(X.rows() == 1 && X.cols() == 8, "fft() keeps a row vector a row vector");
        ok(near(X(0, 0).real(), 36.0) && std::abs(X(0, 0).imag()) < 1e-12,
                                             "X[0] is the sum of the input");
        ok(near(X(0, 4).real(), -4.0) && std::abs(X(0, 4).imag()) < 1e-12,
                                             "the Nyquist bin of 1..8 is -4");
        // A real input has a conjugate-symmetric transform: X[n-k] == conj(X[k]).
        bool herm = true;
        for (int k = 1; k < 8; k++)
            if (std::abs(X(0, 8 - k) - std::conj(X(0, k))) > 1e-12) herm = false;
        ok(herm,                             "a real signal transforms conjugate-symmetrically");

        // --- the round trip, which catches normalisation, bit-reversal and
        //     twiddle-sign errors all at once and needs no reference ---
        for (long n : {1L, 2L, 3L, 4L, 5L, 7L, 8L, 16L, 31L, 64L, 100L, 127L, 256L}) {
            Matrix<double> v(1, n); v.set_Ran_values(-1, 1, -n - 1);
            ok(maxdiff(ifft(fft(v)), toC(v)) < 1e-10,
               "ifft(fft(x)) == x at n = " + std::to_string(n));
        }
        // n = 127 and n = 31 are PRIME, so those went through Bluestein, not
        // radix-2 — the two paths are both exercised above.

        // --- known transforms ---
        Matrix<double> imp(1, 5); imp = {{1, 0, 0, 0, 0}};
        Matrix<cplx> I5 = fft(imp);
        bool allOne = true;
        for (int k = 0; k < 5; k++) if (std::abs(I5(0, k) - cplx(1, 0)) > 1e-12) allOne = false;
        ok(allOne,                           "the transform of a unit impulse is flat");
        Matrix<double> con(1, 16); for (int k = 0; k < 16; k++) con(0, k) = 3.0;
        Matrix<cplx> C16 = fft(con);
        double leak = 0.0;
        for (int k = 1; k < 16; k++) leak = std::max(leak, std::abs(C16(0, k)));
        ok(near(C16(0, 0).real(), 48.0) && leak < 1e-12,
                                             "the transform of a constant is a single spike");

        // --- linearity, and Parseval, which pins the scaling ---
        Matrix<double> a(1, 32), b(1, 32);
        a.set_Ran_values(-1, 1, -11); b.set_Ran_values(-1, 1, -22);
        ok(maxdiff(fft(a + b), fft(a) + fft(b)) < 1e-10, "fft is linear");
        double e1 = 0.0, e2 = 0.0;
        Matrix<cplx> A32 = fft(a);
        for (int k = 0; k < 32; k++) { e1 += a(0, k) * a(0, k); e2 += std::norm(A32(0, k)); }
        ok(near(e2, 32.0 * e1, 1e-9),        "Parseval: sum|X|^2 == n * sum|x|^2");

        // --- padding and truncation ---
        ok(fft(x, 16).cols() == 16,          "fft(x, n) zero-pads");
        ok(fft(x, 4).cols() == 4,            "fft(x, n) truncates");
        ok(near(fft(x, 16)(0, 0).real(), 36.0), "padding does not change the sum");
        Matrix<double> first4(1, 4); first4 = {{1, 2, 3, 4}};
        ok(maxdiff(fft(x, 4), fft(first4)) < 1e-12,
                                             "truncating transforms only the kept samples");

        // --- axis handling: a matrix goes column by column, MATLAB's rule ---
        Matrix<double> M(4, 3); M.set_Ran_values(-1, 1, -33);
        Matrix<cplx> FM = fft(M);
        ok(FM.rows() == 4 && FM.cols() == 3, "fft() of a matrix keeps its shape");
        bool percol = true;
        for (int c = 0; c < 3; c++) {
            Matrix<double> col(4, 1);
            for (int r = 0; r < 4; r++) col(r, 0) = M(r, c);
            Matrix<cplx> FC = fft(col);
            for (int r = 0; r < 4; r++)
                if (std::abs(FM(r, c) - FC(r, 0)) > 1e-12) percol = false;
        }
        ok(percol,                           "each column is transformed independently");
        // Along rows, with the same addcol flag sum() and cumsum() take.
        ok(maxdiff(fft(M, -1, true), fft(M.T(), -1, false).T()) < 1e-10,
                                             "fft(A, n, true) works along rows");
        ok(maxdiff(ifft(fft(M)), toC(M)) < 1e-10, "the matrix round trip");

        // --- complex input ---
        using namespace matrix_literals;
        Matrix<cplx> z(1, 8);
        for (int k = 0; k < 8; k++) z(0, k) = cplx(std::cos(k), std::sin(k));
        ok(maxdiff(ifft(fft(z)), z) < 1e-10, "a complex signal round-trips");
        // exp(2*pi*i*k/n) is a pure tone: one non-zero bin, at index 1.
        Matrix<cplx> tone(1, 16);
        for (int k = 0; k < 16; k++)
            tone(0, k) = std::exp(cplx(0, 1) * (2.0 * mconst::pi * double(k) / 16.0));
        Matrix<cplx> T16 = fft(tone);
        double off = 0.0;
        for (int k = 0; k < 16; k++) if (k != 1) off = std::max(off, std::abs(T16(0, k)));
        ok(near(std::abs(T16(0, 1)), 16.0, 1e-9) && off < 1e-10,
                                             "a pure tone lands in exactly one bin");

        // --- the convolution theorem, which is what conv() will lean on ---
        Matrix<double> p(1, 8), q(1, 8);
        p = {{1, 2, 3, 0, 0, 0, 0, 0}};
        q = {{4, 5, 0, 0, 0, 0, 0, 0}};
        Matrix<cplx> pq = ifft(fft(p) % fft(q));
        // (1,2,3) * (4,5) = (4, 13, 22, 15)
        ok(near(pq(0, 0).real(), 4.0, 1e-9) && near(pq(0, 1).real(), 13.0, 1e-9) &&
           near(pq(0, 2).real(), 22.0, 1e-9) && near(pq(0, 3).real(), 15.0, 1e-9),
                                             "ifft(fft(a) .* fft(b)) convolves");

        // --- fftshift ---
        Matrix<double> sh(1, 8); sh = {{0, 1, 2, 3, 4, 5, 6, 7}};
        ok(fftshift(sh)(0, 0) == 4.0,        "fftshift() moves the zero frequency to the middle");
        ok(same(ifftshift(fftshift(sh)), sh), "ifftshift() undoes fftshift() for even n");
        Matrix<double> so(1, 5); so = {{0, 1, 2, 3, 4}};
        ok(same(ifftshift(fftshift(so)), so), "and for odd n, where they differ");

        // --- edge cases ---
        ok(fft(Matrix<double>(1, 1)).numel() == 1, "a length-1 transform is the identity");
        ok(fft(Matrix<double>(0, 0)).numel() == 0, "an empty input gives an empty result");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Sequences, shape and constructors (tier 5)");
    {
        // --- sequences ---
        Matrix<double> ls = linspace(0, 1, 5);
        Matrix<double> lw(1, 5); lw = {{0, 0.25, 0.5, 0.75, 1}};
        ok(same(ls, lw),                     "linspace() spaces evenly");
        ok(ls.numel() == 5,                  "linspace() returns n points");
        // The endpoint is SET, not computed — a + i*(b-a)/(n-1) does not
        // reliably land on b, and code gets written around that.
        ok(linspace(0, 1, 101)(0, 100) == 1.0, "linspace() hits the endpoint exactly");
        ok(linspace(0, 1, 3)(0, 0) == 0.0,   "and the start exactly");
        ok(linspace(2, 5, 1)(0, 0) == 5.0,   "linspace(a,b,1) is b, as MATLAB gives");
        ok(linspace(0, 1, 0).numel() == 0,   "linspace(a,b,0) is empty");
        Matrix<double> rg = range(0, 5, 2);
        Matrix<double> rw(1, 3); rw = {{0, 2, 4}};
        ok(same(rg, rw),                     "range() includes the endpoint only if it lands");
        ok(range(0, 4).numel() == 5,         "range() defaults to unit step");
        ok(range(5, 0).numel() == 0,         "a range going the wrong way is empty");
        ok(threwMask([]{ range(0, 1, 0); }), "a zero step throws");
        Matrix<double> lg = logspace(0, 3, 4);
        Matrix<double> gw(1, 4); gw = {{1, 10, 100, 1000}};
        ok(same(lg, gw, 1e-9),               "logspace()");

        // --- shape ---
        Matrix<double> A(2, 3); A = {{1, 2, 3}, {4, 5, 6}};
        ok(A.numel() == 6,                   "numel()");
        Matrix<double> fl(2, 3); fl = {{3, 2, 1}, {6, 5, 4}};
        ok(same(A.fliplr(), fl),             "fliplr() reverses the columns");
        Matrix<double> fu(2, 3); fu = {{4, 5, 6}, {1, 2, 3}};
        ok(same(A.flipud(), fu),             "flipud() reverses the rows");
        ok(same(A.fliplr().fliplr(), A),     "fliplr() is its own inverse");
        ok(same(A.flipud().flipud(), A),     "flipud() likewise");
        // rot90 turns counterclockwise, and four turns is the identity.
        ok(A.rot90().rows() == 3 && A.rot90().cols() == 2, "rot90() transposes the shape");
        ok(same(A.rot90(4), A),              "four quarter-turns is the identity");
        ok(same(A.rot90(1).rot90(-1), A),    "rot90(-1) undoes rot90(1)");
        ok(same(A.rot90(2), A.fliplr().flipud()), "rot90(2) is a double mirror");
        ok(same(A.rot90(5), A.rot90(1)),     "k is taken modulo 4");
        ok(A.rot90()(0, 0) == 3.0,           "rot90() really is counterclockwise");

        Matrix<double> rp = A.repmat(2, 2);
        ok(rp.rows() == 4 && rp.cols() == 6, "repmat() tiles the shape");
        ok(rp(0, 0) == 1.0 && rp(3, 5) == 6.0 && rp(2, 3) == 1.0, "repmat() tiles the values");
        ok(same(A.repmat(1, 1), A),          "repmat(1,1) is the identity");

        Matrix<double> cs(2, 3); cs = {{4, 5, 6}, {1, 2, 3}};
        ok(same(A.circshift(1, 0), cs),      "circshift() on rows");
        Matrix<double> cc(2, 3); cc = {{2, 3, 1}, {5, 6, 4}};
        ok(same(A.circshift(-1, 1), cc),     "circshift() on columns, negative k");
        ok(same(A.circshift(2, 0), A),       "a full turn is the identity");
        ok(same(A.circshift(-1, 1).circshift(1, 1), A), "circshift() inverts");
        ok(threwMask([&]{ A.circshift(1, 2); }), "an invalid dim throws");

        Matrix<double> B(1, 2); B = {{7, 8}};
        Matrix<double> bd = A.blkdiag(B);
        ok(bd.rows() == 3 && bd.cols() == 5, "blkdiag() sums both dimensions");
        ok(bd(0, 0) == 1.0 && bd(2, 3) == 7.0 && bd(0, 4) == 0.0,
                                             "blkdiag() places the blocks and zeroes the rest");

        // --- structured test matrices ---
        ok(same(hilb(3)(0, 0) * Id(1), Id(1)), "hilb(3)(0,0) is 1");
        ok(hilb(5).IsSymmetric(),            "a Hilbert matrix is symmetric");
        ok(hilb(5).factorize().kind() == Decomposition<double>::Kind::Cholesky,
                                             "and positive definite");
        ok(condest(hilb(5)) > 1e5,           "hilb(5) is badly conditioned, as advertised");
        ok(near(pascal(5).det(), 1.0, 1e-6), "a Pascal matrix has determinant 1");
        ok(pascal(5).IsSymmetric(),          "and is symmetric");
        ok(wilkinson(7).IsSymmetric(),       "a Wilkinson matrix is symmetric");
        ok(wilkinson(7).bandwidth() == std::make_pair(1L, 1L), "and tridiagonal");

        // A magic square: every row, column and both diagonals share one sum.
        for (long n : {3L, 4L, 5L, 6L, 8L}) {
            Matrix<double> M = magic(n);
            const double want = double(n * (n * n + 1)) / 2.0;
            bool lines = true;
            for (long i = 0; i < n; i++) {
                if (!near(M.sum(true)(int(i), 0), want))  lines = false;
                if (!near(M.sum(false)(0, int(i)), want)) lines = false;
            }
            double d1 = 0, d2 = 0;
            for (long i = 0; i < n; i++) { d1 += M(int(i), int(i));
                                           d2 += M(int(i), int(n - 1 - i)); }
            ok(lines && near(d1, want) && near(d2, want),
               "magic(" + std::to_string(n) + ") rows, columns and both diagonals agree");
            ok(M.unique().numel() == n * n, "magic(" + std::to_string(n) +
                                            ") uses each of 1..n^2 exactly once");
        }
        ok(threwMask([]{ magic(2); }),       "no 2x2 magic square exists");

        Matrix<double> tc(1, 3); tc = {{1, 2, 3}};
        Matrix<double> tp = toeplitz(tc);
        ok(tp(0, 0) == 1 && tp(1, 0) == 2 && tp(0, 1) == 2, "toeplitz() is constant on diagonals");
        ok(tp.IsSymmetric(),                 "a one-vector toeplitz is symmetric");
        Matrix<double> vd = vander(tc);
        ok(vd(1, 0) == 4.0 && vd(1, 1) == 2.0 && vd(1, 2) == 1.0,
                                             "vander() uses descending powers");
        // Descending powers means V * p evaluates p, in polyval's coefficient order.
        Matrix<double> pp(3, 1); pp = {{1}, {0}, {-1}};        // x^2 - 1
        ok(same((vd * pp).T(), polyval(pp.T(), tc), 1e-12),
                                             "vander() * p agrees with polyval");

        // --- random constructors, all through random.hpp's ran2 ---
        Matrix<double> rn = randn(40, 40, -11);
        ok(rn.rows() == 40 && rn.cols() == 40, "randn() shape");
        ok(std::abs(rn.mean()) < 0.1,        "randn() is centred near 0");
        ok(std::abs(rn.stddev() - 1.0) < 0.1, "randn() has unit spread");
        ok(same(randn(3, 3, -5), randn(3, 3, -5)), "the same seed gives the same matrix");
        ok(!same(randn(3, 3, -5), randn(3, 3, -6)), "a different seed does not");
        ok(threwMask([]{ randn(2, 2, 7); }), "a non-negative seed is refused");

        Matrix<long> ri = randi(1, 6, 1, 400, -22);
        ok(ri.min() >= 1 && ri.max() <= 6,   "randi() stays inside its bounds");
        ok(ri.unique().numel() == 6,         "randi(1,6) reaches every value");
        ok(threwMask([]{ randi(6, 1, 2, 2, -1); }), "randi() rejects lo > hi");

        Matrix<long> pm = randperm(50, -33);
        ok(pm.numel() == 50,                 "randperm() returns n entries");
        ok(pm.unique().numel() == 50,        "randperm() is a permutation — no repeats");
        ok(pm.min() == 0 && pm.max() == 49,  "randperm() covers 0..n-1");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("funm and generalized eigenvalues (tier 4)");
    {
        using cplx = std::complex<double>;
        auto realpart = [](const Matrix<cplx>& Z) {
            Matrix<double> R(Z.rows(), Z.cols());
            for (long i = 0; i < Z.numel(); i++) R[int(i)] = Z[int(i)].real();
            return R;
        };

        // --- funm against the dedicated implementations it must agree with ---
        Matrix<double> A(4, 4); A.set_Ran_values(-1, 1, -11);
        ok(same(realpart(funm(A, [](cplx z) { return std::exp(z); })), exp(A), 1e-10),
                                             "funm(exp) matches exp(A)");
        ok(same(realpart(funm(A, [](cplx z) { return std::sin(z); })), sin(A), 1e-9),
                                             "funm(sin) matches sin(A)");
        Matrix<double> Spd(3, 3); Spd = {{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
        Matrix<double> Rt = realpart(funm(Spd, [](cplx z) { return std::sqrt(z); }));
        ok(same(Rt * Rt, Spd, 1e-10),        "funm(sqrt) squares back to A");

        // A function with NO dedicated version — the case funm exists for.
        Matrix<double> Inv = realpart(funm(A, [](cplx z) { return 1.0 / (1.0 + z); }));
        ok(same(Inv * (Id(4) + A), Id(4), 1e-10),
                                             "funm(1/(1+z)) inverts (I+A)");
        // f(A) commutes with A for any f — a property no reference is needed for.
        ok(same(Inv * A, A * Inv, 1e-10),    "f(A) commutes with A");

        // A matrix with COMPLEX-CONJUGATE eigenvalues, which exercises the 2x2
        // block handling in the real-Schur-to-complex-Schur step. The first
        // version of this suite only tested a matrix whose eigenvalues happened
        // to be all real, and a genuine bug in that step survived it: the block
        // test used `subdiagonal != 0` where it had to be relative, so a
        // converged 2.8e-23 residual was read as a block and the real one behind
        // it was skipped. The answer stayed real and triangular-looking and was
        // simply wrong.
        Matrix<double> Rot(4, 4);
        const double th = 0.7;
        Rot = {{std::cos(th), -std::sin(th), 0, 0},
               {std::sin(th),  std::cos(th), 0, 0},
               {0, 0, 2, 1},
               {0, 0, 0, 3}};
        ok(std::abs(Rot.eigvals()[0].imag()) > 1e-6,
                                             "the rotation block really has complex eigenvalues");
        Matrix<std::complex<double>> Fc = funm(Rot, [](cplx z) { return std::exp(z); });
        double imagLeak = 0.0;
        for (long i = 0; i < Fc.numel(); i++) imagLeak = std::max(imagLeak, std::abs(Fc[int(i)].imag()));
        ok(imagLeak < 1e-12,                 "f(A) of a real matrix comes back real");
        ok(same(realpart(Fc), exp(Rot), 1e-10),
                                             "funm(exp) matches exp(A) with a complex pair");
        Matrix<double> Rr = realpart(funm(Rot, [](cplx z) { return std::sqrt(z); }));
        ok(same(Rr * Rr, Rot, 1e-9),         "funm(sqrt) squares back with a complex pair");

        // Repeated eigenvalues must REFUSE. The Parlett recurrence divides by
        // the eigenvalue difference, so a defective matrix would give noise.
        Matrix<double> Jord(3, 3); Jord = {{2, 1, 0}, {0, 2, 1}, {0, 0, 2}};
        ok(threwMask([&]{ funm(Jord, [](cplx z) { return std::exp(z); }); }),
                                             "funm refuses repeated eigenvalues");
        // And exp(A) still works on that same matrix, because it does not use
        // Parlett — which is exactly what the error message tells you to do.
        ok(exp(Jord).numel() == 9,           "exp() handles what funm refuses");
        ok(threwMask([&]{ Matrix<double> ns(2, 3); funm(ns, [](cplx z) { return z; }); }),
                                             "funm rejects a non-square matrix");

        // --- symmetric-definite eig(A, B) ---
        Matrix<double> R5(5, 5); R5.set_Ran_values(-1, 1, -22);
        Matrix<double> Asym = R5 + R5.T();
        Matrix<double> Bspd = R5.T() * R5;
        for (int i = 0; i < 5; i++) Bspd(i, i) += 5.0;
        auto [lam, X] = eig(Asym, Bspd);
        ok(lam.rows() == 5 && X.rows() == 5 && X.cols() == 5, "eig(A,B) shapes");
        // THE test: every pair must satisfy A x = lambda B x. This is what
        // caught the eigenvector solve going against L instead of L-transpose —
        // the eigenvalues were right and only the vectors were wrong, so
        // nothing else would have noticed.
        double resid = 0.0;
        for (int j = 0; j < 5; j++) {
            Matrix<double> x(5, 1);
            for (int i = 0; i < 5; i++) x(i, 0) = X(i, j);
            resid = std::max(resid, (Asym * x - Bspd * x * lam(j, 0)).norm());
        }
        ok(resid < 1e-9,                     "eig(A,B): A x == lambda B x for every pair");
        bool asc = true;
        for (int i = 0; i + 1 < 5; i++) if (lam(i, 0) > lam(i + 1, 0) + 1e-12) asc = false;
        ok(asc,                              "eig(A,B) returns eigenvalues ascending");
        // B-orthonormality: X^T B X == I, the defining property of this problem.
        ok(same(X.T() * Bspd * X, Id(5), 1e-8),
                                             "eigenvectors are B-orthonormal");
        // With B = I it must reduce to the ordinary symmetric problem.
        Matrix<double> lamI = eig(Asym, Id(5)).first;
        Matrix<double> plain = Asym.eig().first.sort(false);
        ok(same(lamI, plain, 1e-9),          "eig(A,I) agrees with the standard eig(A)");
        ok(threwMask([&]{ eig(R5, Bspd); }), "eig(A,B) rejects a non-symmetric A");
        ok(threwMask([&]{ eig(Asym, Asym); }), "eig(A,B) rejects an indefinite B");

        // --- the general pencil ---
        Matrix<double> G(4, 4), H(4, 4);
        G.set_Ran_values(-1, 1, -33); H.set_Ran_values(-1, 1, -44);
        for (int i = 0; i < 4; i++) H(i, i) += 4.0;
        Matrix<cplx> ge = eigvals(G, H);
        ok(ge.rows() == 4,                   "eigvals(A,B) returns one value per dimension");
        // Each eigenvalue must make (A - lambda B) singular, which for a real
        // eigenvalue is checkable with the determinant.
        // The assertion is on the SMALLEST SINGULAR VALUE, not on rank() and not
        // on det(). det() throws on a singular matrix rather than returning zero,
        // and rank()'s cut-off lands right on top of the value being tested — the
        // first version of this check sat exactly on that boundary and failed for
        // a matrix whose sigma_min was 1.4e-15. sigma_min relative to ||A|| says
        // what is actually meant: singular to working precision.
        bool sing = true;
        for (int k = 0; k < 4; k++) {
            if (std::abs(ge[k].imag()) > 1e-12) continue;      // skip complex pairs
            Matrix<double> M = G - H * ge[k].real();
            auto [Um, Sm, Vm] = M.svd();
            if (Sm(3, 3) > 1e-12 * G.norm()) sing = false;
        }
        ok(sing,                             "(A - lambda B) is singular at each eigenvalue");
        // The guard: a singular B must be refused, not quietly reduced.
        Matrix<double> Bsing(4, 4); Bsing.set_Ran_values(-1, 1, -55);
        for (int j = 0; j < 4; j++) Bsing(3, j) = Bsing(0, j) * 2.0;   // rank deficient
        ok(threwMask([&]{ eigvals(G, Bsing); }),
                                             "eigvals(A,B) refuses a singular B");

        // --- polyeig ---
        // 1x1 coefficients reduce the matrix polynomial to a scalar one, so the
        // answer must be exactly what roots() gives for the same coefficients.
        std::vector<Matrix<double>> c(3, Matrix<double>(1, 1));
        c[0](0, 0) = 2; c[1](0, 0) = -3; c[2](0, 0) = 1;     // lambda^2 - 3 lambda + 2
        Matrix<cplx> pe = polyeig(c);
        ok(pe.rows() == 2,                   "polyeig returns degree * n eigenvalues");
        double r0 = pe[0].real(), r1 = pe[1].real();
        if (r0 > r1) std::swap(r0, r1);
        ok(near(r0, 1.0, 1e-9) && near(r1, 2.0, 1e-9),
                                             "polyeig on 1x1 matches the scalar roots");
        // A genuine 2x2 quadratic: each eigenvalue must make P(lambda) singular.
        std::vector<Matrix<double>> q(3, Matrix<double>(2, 2));
        q[0] = {{2, 0}, {0, 3}}; q[1] = {{-3, 1}, {0, -4}}; q[2] = {{1, 0}, {0, 1}};
        Matrix<cplx> pq = polyeig(q);
        ok(pq.rows() == 4,                   "a 2x2 quadratic has four eigenvalues");
        bool psing = true;
        for (int k = 0; k < 4; k++) {
            if (std::abs(pq[k].imag()) > 1e-10) continue;
            const double L = pq[k].real();
            Matrix<double> P = q[0] + q[1] * L + q[2] * (L * L);
            // Smallest singular value, for the reason given above.
            auto [Up, Sp, Vp] = P.svd();
            if (Sp(1, 1) > 1e-9 * (1.0 + std::abs(L))) psing = false;
        }
        ok(psing,                            "P(lambda) is singular at each polyeig value");
        ok(threwMask([&]{ polyeig(std::vector<Matrix<double>>{Matrix<double>(2, 2)}); }),
                                             "polyeig needs at least two coefficients");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Decomposition, condition estimates, minimum-norm (tier 4)");
    {
        // --- the factorisation is chosen from the structure, not from a flag ---
        Matrix<double> Rr(5, 5); Rr.set_Ran_values(-1, 1, -11);
        Matrix<double> Spd = Rr.T() * Rr;
        for (int i = 0; i < 5; i++) Spd(i, i) += 5.0;
        Matrix<double> Gen(5, 5); Gen.set_Ran_values(-1, 1, -22);
        Matrix<double> Rect(8, 3); Rect.set_Ran_values(-1, 1, -33);

        ok(Spd.factorize().kind()  == Decomposition<double>::Kind::Cholesky,
                                             "an SPD matrix factors by Cholesky");
        ok(Gen.factorize().kind()  == Decomposition<double>::Kind::LU,
                                             "a general square matrix factors by LU");
        ok(Rect.factorize().kind() == Decomposition<double>::Kind::QR,
                                             "a tall matrix factors by QR");
        ok(std::string(Spd.factorize().kindName()) == "Cholesky", "kindName() reports it");
        // Symmetric but INDEFINITE must fall through Cholesky to LU.
        Matrix<double> Ind(2, 2); Ind = {{1, 2}, {2, 1}};   // eigenvalues 3 and -1
        ok(Ind.IsSymmetric(),                "the indefinite test matrix is symmetric");
        ok(Ind.factorize().kind() == Decomposition<double>::Kind::LU,
                                             "symmetric but indefinite falls back to LU");

        // --- every path must agree with the one-shot solve ---
        Matrix<double> b5(5, 1); b5.set_Ran_values(-1, 1, -44);
        Matrix<double> b8(8, 1); b8.set_Ran_values(-1, 1, -55);
        ok(same(Spd.factorize().solve(b5),  Spd.solve(b5),  1e-10), "Cholesky solve matches solve()");
        ok(same(Gen.factorize().solve(b5),  Gen.solve(b5),  1e-10), "LU solve matches solve()");
        ok(same(Rect.factorize().solve(b8), Rect.solve(b8), 1e-8),  "QR solve matches solve()");
        // And must actually solve the system, not merely agree with a sibling.
        ok(same(Gen * Gen.factorize().solve(b5), b5, 1e-10), "A * (dA \\ b) == b");
        ok(same(Spd * Spd.factorize().solve(b5), b5, 1e-10), "the Cholesky path too");

        // --- reuse: one factorisation, many right-hand sides ---
        auto dG = Gen.factorize();
        bool reuse = true;
        for (int k = 0; k < 5; k++) {
            Matrix<double> bk(5, 1); bk.set_Ran_values(-1, 1, -100 - k);
            if (!same(Gen * dG.solve(bk), bk, 1e-10)) reuse = false;
        }
        ok(reuse,                            "the same object solves many right-hand sides");
        // Multiple columns at once must match column-by-column.
        Matrix<double> B3(5, 3); B3.set_Ran_values(-1, 1, -66);
        ok(same(Gen * dG.solve(B3), B3, 1e-10), "a multi-column right-hand side");

        // The object owns its factors, so it outlives the matrix it came from.
        Decomposition<double> kept = Gen.factorize();
        {
            Matrix<double> temp = Gen;
            kept = temp.factorize();
        }
        ok(same(Gen * kept.solve(b5), b5, 1e-10),
                                             "a Decomposition outlives its source matrix");

        // --- det from the factors already held ---
        ok(near(Gen.factorize().det(), Gen.det(), 1e-9),  "det() from the LU factors");
        ok(near(Spd.factorize().det(), Spd.det(), 1e-6),  "det() from the Cholesky factors");
        ok(threwMask([&]{ Rect.factorize().det(); }),     "a rectangular system has no det()");
        ok(threwMask([&]{ Matrix<double> u(3, 5); u.factorize(); }),
                                             "an under-determined system is refused");

        // --- rcond / condest ---
        // Hilbert matrices are the standard ill-conditioning test, and small ones
        // have known condition numbers, so this pins the estimator to a real value.
        Matrix<double> Hil(5, 5);
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) Hil(i, j) = 1.0 / (i + j + 1);
        ok(near(condest(Hil), Hil.cond(NormType::One), 1e-6),
                                             "condest() matches the true 1-norm condition of Hilbert(5)");
        ok(near(rcond(Hil), 1.0 / Hil.cond(NormType::One), 1e-6), "rcond() is its reciprocal");
        ok(rcond(Id(4)) > 0.99,              "rcond() of the identity is 1");
        // The estimator is a LOWER bound on the condition number: never pessimistic.
        ok(condest(Gen) <= Gen.cond(NormType::One) * (1.0 + 1e-9),
                                             "condest() never overestimates");
        ok(condest(Gen) >= 1.0,              "a condition number is at least 1");

        // --- lsqminnorm ---
        // Under-determined: solve() refuses, lsqminnorm gives the smallest x.
        Matrix<double> Wide(2, 4); Wide = {{1, 0, 1, 0}, {0, 1, 0, 1}};
        Matrix<double> rhs(2, 1);  rhs  = {{2}, {4}};
        ok(threwMask([&]{ Wide.solve(rhs); }), "solve() refuses an under-determined system");
        Matrix<double> xm = lsqminnorm(Wide, rhs);
        ok(same(Wide * xm, rhs, 1e-10),      "lsqminnorm() satisfies the equations");
        // Any other exact solution must have a larger norm.
        Matrix<double> other(4, 1); other = {{2}, {4}, {0}, {0}};
        ok(same(Wide * other, rhs, 1e-12),   "the comparison solution is also exact");
        ok(xm.norm() < other.norm(),         "lsqminnorm() really is the minimum-norm one");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Element-wise maths (tier 3)");
    {
        Matrix<double> A(1, 5); A = {{-2.5, -1.0, 0.0, 1.5, 2.5}};

        Matrix<double> sg(1, 5); sg = {{-1, -1, 0, 1, 1}};
        ok(same(A.sign(), sg),               "sign() is -1 / 0 / +1");
        ok(same(A.sign() % A.abs(), A),      "sign(x) * |x| == x");

        Matrix<double> fl(1, 5); fl = {{-3, -1, 0, 1, 2}};
        ok(same(A.floor(), fl),              "floor()");
        Matrix<double> ce(1, 5); ce = {{-2, -1, 0, 2, 3}};
        ok(same(A.ceil(), ce),               "ceil()");
        Matrix<double> rn(1, 5); rn = {{-3, -1, 0, 2, 3}};
        ok(same(A.round(), rn),              "round() sends halves away from zero");
        Matrix<double> fx(1, 5); fx = {{-2, -1, 0, 1, 2}};
        ok(same(A.fix(), fx),                "fix() truncates towards zero");
        // The one that catches people out: floor and fix differ on negatives.
        ok(A.floor()(0, 0) == -3.0 && A.fix()(0, 0) == -2.0,
                                             "floor(-2.5) is -3 but fix(-2.5) is -2");

        // mod follows the divisor's sign, rem the dividend's.
        Matrix<double> M(1, 4); M = {{-1, -4, 5, 7}};
        Matrix<double> md(1, 4); md = {{2, 2, 2, 1}};
        Matrix<double> rm(1, 4); rm = {{-1, -1, 2, 1}};
        ok(same(M.mod(3.0), md),             "mod() takes the divisor's sign");
        ok(same(M.rem(3.0), rm),             "rem() takes the dividend's sign");
        ok(!same(M.mod(3.0), M.rem(3.0)),    "mod and rem really are different");
        // Both satisfy the division identity with their own quotient.
        ok(near((M.rem(3.0) + M.fix() * 0.0)(0, 0), std::fmod(-1.0, 3.0)),
                                             "rem agrees with fmod");

        // atan2 resolves the quadrant that atan(y/x) cannot.
        Matrix<double> Y(1, 3); Y = {{1, -1, 0}};
        Matrix<double> X(1, 3); X = {{1, -1, -1}};
        ok(near(Y.atan2(X)(0, 0), mconst::pi / 4),      "atan2 in the first quadrant");
        ok(near(Y.atan2(X)(0, 1), -3 * mconst::pi / 4), "atan2 in the third quadrant");
        ok(near(Y.atan2(X)(0, 2), mconst::pi),          "atan2 on the negative x axis");

        Matrix<double> ha(1, 2); ha = {{3, 5}};
        Matrix<double> hb(1, 2); hb = {{4, 12}};
        Matrix<double> hw(1, 2); hw = {{5, 13}};
        ok(same(ha.hypot(hb), hw),           "hypot() gives the Pythagorean length");

        Matrix<double> sm(1, 1); sm = {{1e-18}};
        ok(near(sm.expm1()(0, 0), 1e-18, 1e-12),  "expm1() keeps precision for tiny x");
        ok(near(sm.log1p()(0, 0), 1e-18, 1e-12),  "log1p() likewise");
        ok(near(A.asinh()(0, 3), std::asinh(1.5)), "asinh()");
        ok(near(ce.acosh()(0, 4), std::acosh(3.0)), "acosh()");
        ok(near(sm.atanh()(0, 0), std::atanh(1e-18), 1e-12), "atanh()");

        // angle / arg — the piece complex work was missing.
        using namespace matrix_literals;
        Matrix<std::complex<double>> Z(1, 3);
        Z(0, 0) = 1.0 + 1.0i; Z(0, 1) = -1.0; Z(0, 2) = 1.0i;
        ok(near(Z.angle()(0, 0), mconst::pi / 4), "angle() of 1+i is pi/4");
        ok(near(Z.angle()(0, 1), mconst::pi),     "angle() of -1 is pi");
        ok(near(Z.angle()(0, 2), mconst::pi / 2), "angle() of i is pi/2");
        ok(same(Z.arg(), Z.angle()),              "arg() is angle()");
        // The polar identity: |z| * exp(i*angle(z)) == z.
        bool polar = true;
        for (int k = 0; k < 3; k++) {
            std::complex<double> z = Z(0, k);
            std::complex<double> rebuilt =
                std::abs(z) * std::exp(std::complex<double>(0, 1) * Z.angle()(0, k));
            if (std::abs(rebuilt - z) > 1e-12) polar = false;
        }
        ok(polar,                            "|z| * exp(i*angle(z)) reconstructs z");
        // For a real matrix angle is 0 or pi, as MATLAB gives.
        ok(near(A.angle()(0, 0), mconst::pi) && near(A.angle()(0, 3), 0.0),
                                             "angle() of a real matrix is 0 or pi");
        // sign() for complex is the unit vector z/|z|.
        ok(std::abs(std::abs(Z.sign()(0, 0)) - 1.0) < 1e-15,
                                             "complex sign() has modulus 1");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Structure, subspaces and polynomials (tier 4)");
    {
        // --- predicates ---
        Matrix<double> S(3, 3); S = {{2, 1, 0}, {1, 3, 1}, {0, 1, 4}};
        ok(S.IsSymmetric(),                  "IsSymmetric() on a symmetric matrix");
        ok(S.IsHermitian(),                  "a real symmetric matrix is Hermitian");
        Matrix<double> NS(2, 2); NS = {{1, 2}, {3, 4}};
        ok(!NS.IsSymmetric(),                "IsSymmetric() is false otherwise");
        using namespace matrix_literals;
        Matrix<std::complex<double>> H(2, 2);
        H(0, 0) = 1.0; H(0, 1) = 2.0 + 1.0i; H(1, 0) = 2.0 - 1.0i; H(1, 1) = 3.0;
        ok(H.IsHermitian(),                  "IsHermitian() on a complex Hermitian matrix");
        ok(!H.IsSymmetric(),                 "which is NOT the same as symmetric");

        Matrix<double> U(3, 3); U = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
        ok(U.IsUpper() && !U.IsLower(),      "IsUpper() / IsLower()");
        ok(U.T().IsLower(),                  "the transpose of an upper is lower");
        ok(same(U, U.triu()),                "IsUpper(k) agrees with triu(k)");
        Matrix<double> D(3, 3); D = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        ok(D.IsUpper() && D.IsLower() && D.IsDiagonal(), "a diagonal is both");
        ok(D.bandwidth() == std::make_pair(0L, 0L), "bandwidth() of a diagonal is {0,0}");
        Matrix<double> Tri(4, 4);
        Tri = {{2, 1, 0, 0}, {1, 2, 1, 0}, {0, 1, 2, 1}, {0, 0, 1, 2}};
        ok(Tri.bandwidth() == std::make_pair(1L, 1L), "bandwidth() of a tridiagonal");
        ok(Tri.IsBanded(1, 1) && !Tri.IsBanded(0, 0), "IsBanded()");

        // --- null and orth ---
        Matrix<double> R(3, 3); R = {{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};   // rank 2
        ok(R.rank() == 2,                    "the test matrix has rank 2");
        Matrix<double> N = R.null();
        ok(N.rows() == 3 && N.cols() == 1,   "null() has one column for a rank-2 3x3");
        ok((R * N).norm() < 1e-10,           "A * null(A) is zero");
        ok(near(N.dot(N), 1.0, 1e-12),       "null()'s columns are unit length");
        Matrix<double> O = R.orth();
        ok(O.rows() == 3 && O.cols() == 2,   "orth() has one column per rank");
        ok(same(O.T() * O, Id(2), 1e-10),    "orth()'s columns are orthonormal");
        Matrix<double> F(3, 3); F.set_Ran_values(-1, 1, -21);
        ok(F.null().cols() == 0,             "a full-rank matrix has an empty null space");

        // --- rref ---
        Matrix<double> Rr = R.rref();
        ok(Rr.rows() == 3 && Rr.cols() == 3, "rref() keeps the shape");
        // Leading ones, and zeros above and below each pivot.
        ok(near(Rr(0, 0), 1.0) && near(Rr(1, 0), 0.0) && near(Rr(2, 0), 0.0),
                                             "rref() has a leading one with zeros under it");
        ok(near(Rr(2, 0), 0.0) && near(Rr(2, 1), 0.0) && near(Rr(2, 2), 0.0),
                                             "the dependent row reduces to zero");
        ok(same(Id(3).rref(), Id(3)),        "rref() of the identity is the identity");

        // --- dot and cross ---
        Matrix<double> a(3, 1); a = {{1}, {0}, {0}};
        Matrix<double> b(3, 1); b = {{0}, {1}, {0}};
        ok(near(a.dot(b), 0.0),              "dot() of orthogonal vectors is 0");
        ok(near(a.dot(a), 1.0),              "dot() of a unit vector with itself is 1");
        Matrix<double> c(3, 1); c = {{0}, {0}, {1}};
        ok(same(a.cross(b), c),              "cross() of x and y is z");
        ok(same(b.cross(a), c * -1.0),       "cross() is antisymmetric");
        ok(near(a.cross(b).dot(a), 0.0),     "the cross product is orthogonal to both");
        // For complex, dot conjugates the LEFT operand, so dot(A,A) is ||A||_F^2.
        Matrix<std::complex<double>> zc(2, 1); zc(0, 0) = 3.0 + 4.0i; zc(1, 0) = 1.0i;
        ok(near(zc.dot(zc).real(), 26.0) && near(zc.dot(zc).imag(), 0.0),
                                             "dot() conjugates the left operand");
        ok(threwMask([&]{ Matrix<double> two(2, 1); Matrix<double> q = two.cross(two); (void)q; }),
                                             "cross() rejects non-3-element operands");

        // --- normest ---
        Matrix<double> E(20, 20); E.set_Ran_values(-1, 1, -77);
        ok(near(E.normest(1e-10), E.norm(NormType::Two), 1e-6),
                                             "normest() agrees with the true 2-norm");

        // --- hess ---
        Matrix<double> Hm(5, 5); Hm.set_Ran_values(-1, 1, -33);
        auto [Hh, Qh] = Hm.hess();
        ok(same(Qh * Hh * Qh.T(), Hm, 1e-10), "hess(): A == Q H Q^T");
        ok(same(Qh.T() * Qh, Id(5), 1e-10),   "hess(): Q is orthogonal");
        bool hessForm = true;
        for (int i = 2; i < 5; i++) for (int j = 0; j + 2 <= i; j++)
            if (std::abs(Hh(i, j)) > 1e-12) hessForm = false;
        ok(hessForm,                          "hess(): H is zero below the subdiagonal");

        // --- schur ---
        auto [Ts, Qs] = Hm.schur();
        ok(same(Qs * Ts * Qs.T(), Hm, 1e-8),  "schur(): A == Q T Q^T");
        ok(same(Qs.T() * Qs, Id(5), 1e-10),   "schur(): Q is orthogonal");
        // T is quasi-triangular: only 1x1 and 2x2 blocks on the diagonal, so
        // two consecutive sub-diagonal entries can never both be non-zero.
        bool quasi = true;
        for (int i = 0; i + 2 < 5; i++)
            if (std::abs(Ts(i + 1, i)) > 1e-10 && std::abs(Ts(i + 2, i + 1)) > 1e-10)
                quasi = false;
        ok(quasi,                             "schur(): T is quasi-triangular");

        // --- polynomials ---
        Matrix<double> p(1, 3); p = {{1, -3, 2}};            // x^2 - 3x + 2
        Matrix<double> xs(1, 3); xs = {{0, 1, 2}};
        Matrix<double> pv(1, 3); pv = {{2, 0, 0}};
        ok(same(polyval(p, xs), pv),          "polyval() by Horner");
        Matrix<std::complex<double>> rt = roots(p);
        ok(rt.rows() == 2,                    "roots() returns one per degree");
        double r0 = rt[0].real(), r1 = rt[1].real();
        if (r0 > r1) std::swap(r0, r1);
        ok(near(r0, 1.0, 1e-10) && near(r1, 2.0, 1e-10), "roots() of x^2-3x+2 are 1 and 2");
        // Every root must actually annihilate the polynomial.
        Matrix<double> rr(2, 1); rr = {{r0}, {r1}};
        ok(polyval(p, rr).norm() < 1e-10,     "polyval() at the roots is zero");
        // Trailing zeros become exact roots at the origin.
        Matrix<double> pz(1, 4); pz = {{1, -3, 2, 0}};       // x*(x^2-3x+2)
        ok(roots(pz).rows() == 3,             "roots() counts a root at zero");

        // polyfit recovers a polynomial it was given exactly.
        Matrix<double> fx2(1, 5); fx2 = {{-2, -1, 0, 1, 2}};
        Matrix<double> fy = polyval(p, fx2);
        Matrix<double> fit = polyfit(fx2, fy, 2);
        ok(same(fit.T(), p, 1e-8),            "polyfit() recovers an exact fit");
        ok(fit.rows() == 3,                   "polyfit() returns degree+1 coefficients");
        ok(threwMask([&]{ polyfit(fx2, fy, 9); }),
                                              "polyfit() rejects too few points");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Scans, orderings and multiset reductions");
    {
        Matrix<double> A(2, 3); A = {{3, 1, 4}, {1, 5, 9}};

        // --- prod ---
        ok(near(A.prod(), 3.0 * 1 * 4 * 1 * 5 * 9), "prod() over every element");
        Matrix<double> pc(1, 3); pc = {{3, 5, 36}};
        ok(same(A.prod(false), pc),          "prod(false) is per column");
        Matrix<double> pr(2, 1); pr = {{12}, {45}};
        ok(same(A.prod(true), pr),           "prod(true) is per row");
        // prod pairs with sum on the same axis convention.
        ok(A.prod(false).cols() == A.sum(false).cols(), "prod and sum agree on the axis flag");

        // --- cumulative scans keep the shape ---
        Matrix<double> cs(2, 3); cs = {{3, 1, 4}, {4, 6, 13}};
        ok(same(A.cumsum(false), cs),        "cumsum(false) accumulates down columns");
        Matrix<double> csr(2, 3); csr = {{3, 4, 8}, {1, 6, 15}};
        ok(same(A.cumsum(true), csr),        "cumsum(true) accumulates along rows");
        ok(A.cumsum(false).rows() == 2 && A.cumsum(false).cols() == 3,
                                             "a scan does not change the shape");
        // The last entry of a scan is the corresponding reduction.
        ok(near(A.cumsum(true)(1, 2), A.sum(true)(1, 0)), "cumsum ends at sum");
        Matrix<double> cp(2, 3); cp = {{3, 1, 4}, {3, 5, 36}};
        ok(same(A.cumprod(false), cp),       "cumprod(false) down columns");
        ok(near(A.cumprod(true)(1, 2), A.prod(true)(1, 0)), "cumprod ends at prod");

        // --- diff shrinks the scanned axis by one ---
        Matrix<double> d0(1, 3); d0 = {{-2, 4, 5}};
        ok(same(A.diff(false), d0),          "diff(false) is (rows-1 x cols)");
        Matrix<double> d1(2, 2); d1 = {{-2, 3}, {4, 4}};
        ok(same(A.diff(true), d1),           "diff(true) is (rows x cols-1)");
        ok(A.diff(false).rows() == 1 && A.diff(true).cols() == 2, "diff shapes");
        // diff undoes cumsum: differencing a running total gives back the
        // original terms, minus the first one that had nothing before it.
        // A = {{3,1,4},{1,5,9}} -> cumsum {{3,4,8},{1,6,15}} -> diff {{1,4},{5,9}}
        Matrix<double> tail(2, 2); tail = {{1, 4}, {5, 9}};
        ok(same(A.cumsum(true).diff(true), tail),
                                             "diff(cumsum(A)) is A without its first column");

        // --- sort keeps the shape ---
        Matrix<double> s0(2, 3); s0 = {{1, 1, 4}, {3, 5, 9}};
        ok(same(A.sort(false), s0),          "sort(false) sorts each column");
        Matrix<double> s1(2, 3); s1 = {{1, 3, 4}, {1, 5, 9}};
        ok(same(A.sort(true), s1),           "sort(true) sorts each row");
        Matrix<double> sd(2, 3); sd = {{4, 3, 1}, {9, 5, 1}};
        ok(same(A.sort(true, true), sd),     "sort(true, descending)");
        ok(A.sort(false).rows() == 2,        "sort does not change the shape");
        // Sorting is a permutation: same multiset, so the same sum and product.
        ok(near(A.sort(false).sum(), A.sum()) && near(A.sort(true).prod(), A.prod()),
                                             "sort permutes, it does not alter values");

        // --- sortrows carries whole rows ---
        Matrix<double> R(3, 2); R = {{3, 30}, {1, 10}, {2, 20}};
        Matrix<double> rs(3, 2); rs = {{1, 10}, {2, 20}, {3, 30}};
        ok(same(R.sortrows(0), rs),          "sortrows orders by the key column");
        ok(same(R.sortrows(0).sortrows(0), R.sortrows(0)), "sortrows is idempotent");
        Matrix<double> rd(3, 2); rd = {{3, 30}, {2, 20}, {1, 10}};
        ok(same(R.sortrows(0, true), rd),    "sortrows descending");
        // The companion column must travel with its key, not be sorted separately.
        ok(R.sortrows(0)(0, 1) == 10.0 && R.sortrows(0)(2, 1) == 30.0,
                                             "sortrows keeps each row intact");
        ok(threwMask([&]{ R.sortrows(5); }), "an out-of-range key throws");

        // --- median ---
        Matrix<double> M5(1, 5); M5 = {{5, 1, 3, 2, 4}};
        ok(near(M5.median(), 3.0),           "median of an odd count");
        Matrix<double> M4(1, 4); M4 = {{4, 1, 3, 2}};
        ok(near(M4.median(), 2.5),           "median of an even count averages the middle two");
        ok(near(A.median(), 3.5),            "median over every element");
        Matrix<double> mc = A.median(false);
        ok(mc.rows() == 1 && mc.cols() == 3, "median(false) is a row of column results");
        ok(near(mc(0, 0), 2.0) && near(mc(0, 2), 6.5), "median per column");

        // --- mode, ties to the smallest ---
        Matrix<double> Mo(1, 6); Mo = {{2, 3, 2, 3, 1, 3}};
        ok(near(Mo.mode(), 3.0),             "mode() finds the most frequent value");
        Matrix<double> Tie(1, 4); Tie = {{5, 5, 2, 2}};
        ok(near(Tie.mode(), 2.0),            "a tie goes to the smallest value");

        // --- unique ---
        Matrix<double> U(2, 3); U = {{3, 1, 3}, {2, 1, 3}};
        Matrix<double> uw(3, 1); uw = {{1}, {2}, {3}};
        ok(same(U.unique(), uw),             "unique() is ascending and distinct");
        ok(U.unique().cols() == 1,           "unique() returns a column vector");
        ok(A.unique().rows() == 5,           "unique() collapses the repeat in A");

        // These compose with the mask layer: count the distinct positives.
        ok(A.unique().gt(0.0).nnz() == 5,    "unique composes with masks");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Logical masks");
    {
        Matrix<double> A(2, 3); A = {{-1, 2, -3}, {4, -5, 6}};
        Matrix<double> B(2, 3); B = {{ 0, 2,  9}, {4, -9, 0}};

        // --- comparison against a scalar ---
        Matrix<bool> pos = A > 0.0;
        Matrix<bool> want(2, 3); want = {{false, true, false}, {true, false, true}};
        ok(pos == want,                       "A > scalar gives an element-wise mask");
        ok(pos.rows() == 2 && pos.cols() == 3, "the mask has the operand's shape");
        ok((A < 0.0) == !pos,                 "A < 0 is the complement of A > 0 here");
        ok((A >= 2.0).nnz() == 3,             "A >= scalar");
        ok((A <= -3.0).nnz() == 2,            "A <= scalar");
        ok(A.gt(0.0) == pos,                  "gt() is the named form of >");
        ok(A.lt(0.0) == (A < 0.0),            "lt() is the named form of <");

        // --- comparison against a matrix ---
        ok((A > B).nnz() == 2,                "A > B element-wise");
        ok((A < B).nnz() == 2,                "A < B element-wise");
        ok(A.eq(B).nnz() == 2,                "eq() is element-wise equality");
        ok(A.ne(B).nnz() == 4,                "ne() is its complement");
        ok(A.eq(B).nnz() + A.ne(B).nnz() == 6, "eq and ne partition every element");

        // --- == stays whole-matrix, which is the deliberate divergence ---
        Matrix<double> Acopy = A;
        ok(A == Acopy,                        "operator== is still whole-matrix equality");
        ok(!(A == B),                         "operator== is false when any element differs");
        ok(A.eq(Acopy).all(),                 "eq() on an identical matrix is all true");

        // --- combinators ---
        Matrix<bool> inRange = (A > 0.0).land(A < 5.0);
        ok(inRange.nnz() == 2,                "land() intersects two masks");
        ok(((A > 0.0) && (A < 5.0)) == inRange, "operator&& is land()");
        ok(((A > 3.0) || (A < -2.0)) == (A > 3.0).lor(A < -2.0),
                                              "operator|| is lor()");
        // Overloading && and || costs short-circuiting, which an
        // element-wise or never had: both operands are needed in full.
        // The result is a Matrix<bool> with no conversion to bool, so
        // `if (m1 || m2)` will not compile — .any()/.all() must be said.
        ok((pos || !pos).all(),               "m || !m covers everything");
        ok(!(pos && !pos).any(),              "m && !m selects nothing");
        // A > 3 selects {4, 6}; A < -2 selects {-3, -5}. Disjoint, so 4.
        ok((A > 3.0).lor(A < -2.0).nnz() == 4, "lor() unions two masks");
        ok((A > 0.0).lxor(A > 3.0).nnz() == 1, "lxor() is the symmetric difference");
        ok(((A > 0.0) ^ (A > 3.0)) == (A > 0.0).lxor(A > 3.0),
                                              "operator^ is lxor()");
        // ^ has famously low precedence — lower than the relational
        // operators — so it groups the way anyone writing it would mean.
        // GCC still emits -Wparentheses here, because an unparenthesised
        // comparison next to ^ is a classic bug in BITWISE code; the grouping
        // is right, but real code should parenthesise anyway to keep the build
        // quiet. The warning is silenced just for this one assertion, whose
        // whole point is to pin the grouping down.
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wparentheses"
#endif
        ok((A > 0.0 ^ A > 3.0) == ((A > 0.0) ^ (A > 3.0)),
                                              "A > 0 ^ A > 3 groups as (A>0) ^ (A>3)");
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
        ok((!pos).nnz() == 3,                 "operator! negates a mask");
        ok(pos.lnot() == !pos,                "lnot() is the named form of !");
        ok((pos.land(!pos)).nnz() == 0,       "a mask and its negation never overlap");
        ok((pos.lor(!pos)).all(),             "a mask or its negation covers everything");

        // --- reductions ---
        ok(pos.any() && !pos.all(),           "any() and all() on a mixed mask");
        ok(!(A > 100.0).any(),                "any() is false when nothing matches");
        ok((A > -100.0).all(),                "all() is true when everything matches");
        ok(pos.nnz() == 3,                    "nnz() counts the true entries");
        ok(Matrix<double>(2, 2).nnz() == 0,   "nnz() of a zero matrix is 0");
        // Counting a mask goes through nnz(), NOT sum(): sum() returns datatype,
        // and for Matrix<bool> that saturates at true instead of counting.
        ok(pos.sum() == true,                 "sum() on a mask saturates — use nnz()");

        // per-axis forms, mirroring sum(bool)
        Matrix<bool> anyCol = A.gt(3.0).any(false);
        ok(anyCol.rows() == 1 && anyCol.cols() == 3, "any(false) gives a row of column results");
        Matrix<bool> wantCol(1, 3); wantCol = {{true, false, true}};
        ok(anyCol == wantCol,                 "any() per column");
        Matrix<bool> allRow = A.gt(-10.0).all(true);
        ok(allRow.rows() == 2 && allRow.cols() == 1, "all(true) gives a column of row results");
        ok(allRow.all(),                      "all() per row");

        // --- find ---
        auto f = pos.find();
        ok(f.size() == 3,                     "find() returns one entry per true element");
        ok(f[0] == std::make_pair(0L, 1L),    "find() is in row-major order");
        ok(f[2] == std::make_pair(1L, 2L),    "find()'s last position");

        // --- logical indexing, read ---
        Matrix<double> picked = A(pos);
        ok(picked.rows() == 3 && picked.cols() == 1, "A(mask) is a column vector of nnz elements");
        Matrix<double> wantPick(3, 1); wantPick = {{2}, {4}, {6}};
        ok(same(picked, wantPick),            "A(mask) selects in row-major order");
        ok(same(Matrix<double>(A(A > 100.0)), Matrix<double>(0, 1)),
                                              "an empty mask selects nothing");

        // --- logical indexing, write ---
        Matrix<double> W = A;
        W(W < 0.0) = 0.0;
        Matrix<double> wantW(2, 3); wantW = {{0, 2, 0}, {4, 0, 6}};
        ok(same(W, wantW),                    "A(mask) = scalar writes through the mask");

        Matrix<double> W2 = A;
        Matrix<double> repl(3, 1); repl = {{10}, {20}, {30}};
        W2(W2 > 0.0) = repl;
        Matrix<double> wantW2(2, 3); wantW2 = {{-1, 10, -3}, {20, -5, 30}};
        ok(same(W2, wantW2),                  "A(mask) = vector writes one value per selection");

        // The write form must round-trip with the read form.
        Matrix<double> W3 = A;
        W3(pos) = Matrix<double>(A(pos)) * 2.0;
        ok(near(W3(0, 1), 4.0) && near(W3(1, 2), 12.0) && near(W3(0, 0), -1.0),
                                              "A(mask) = A(mask)*2 doubles only the selected");

        // --- errors ---
        ok(threwMask([&]{ Matrix<bool> bad(3, 3); Matrix<double> z = A(bad); (void)z; }),
                                              "a mask of the wrong shape throws");
        ok(threwMask([&]{ Matrix<double> bad(2, 1); Matrix<double> C = A; C(C > 0.0) = bad; }),
                                              "a replacement of the wrong length throws");

        // --- masks compose with the rest of the library ---
        Matrix<double> M(3, 3); M.set_Ran_values(-1, 1, -11);
        ok(M.gt(0.0).nnz() + M.le(0.0).nnz() == 9, "gt and le partition a random matrix");
        ok((M.T().gt(0.0)) == (M.gt(0.0)).T(),     "a mask transposes like any matrix");
    }

    // ═══════════════════════════════════════════════════════════════════
    section("Math constants and literals");
    {
        using mconst::pi;
        // Identities that must hold whatever standard this is built as, so they
        // check the C++17 fallback and the std::numbers alias with one body.
        ok(near(pi, std::acos(-1.0), 1e-15),                    "pi == acos(-1)");
        ok(near(mconst::e, std::exp(1.0), 1e-15),               "e == exp(1)");
        ok(near(mconst::ln2, std::log(2.0), 1e-15),             "ln2 == log(2)");
        ok(near(mconst::ln10, std::log(10.0), 1e-15),           "ln10 == log(10)");
        ok(near(mconst::log2e * mconst::ln2, 1.0, 1e-15),       "log2e == 1/ln2");
        ok(near(mconst::log10e * mconst::ln10, 1.0, 1e-15),     "log10e == 1/ln10");
        ok(near(mconst::sqrt2 * mconst::sqrt2, 2.0, 1e-15),     "sqrt2 squares to 2");
        ok(near(mconst::sqrt3 * mconst::sqrt3, 3.0, 1e-15),     "sqrt3 squares to 3");
        ok(near(mconst::inv_sqrt3 * mconst::sqrt3, 1.0, 1e-15), "inv_sqrt3 == 1/sqrt3");
        ok(near(mconst::inv_pi * pi, 1.0, 1e-15),               "inv_pi == 1/pi");
        ok(near(mconst::inv_sqrtpi * mconst::inv_sqrtpi * pi, 1.0, 1e-15),
                                                                "inv_sqrtpi == 1/sqrt(pi)");
        ok(near(mconst::phi * mconst::phi, mconst::phi + 1.0, 1e-15),
                                                                "phi satisfies phi^2 == phi + 1");
        ok(near(mconst::egamma, 0.577215664901532860606, 1e-15), "egamma");
        ok(mconst::pi_v<double> == pi,                          "pi_v<double> is the double alias");
        ok(mconst::pi_v<float> == float(pi),                    "pi_v<float> rounds like pi");

#if defined(__cpp_lib_math_constants)
        // Built at C++20 or later, so the standard's own constants exist and can
        // be compared against. In the DEFAULT C++20 build mconst simply aliases
        // them, so this confirms the alias is wired up; build with
        //     -std=c++20 -DMATRIXCPP_NO_STD_NUMBERS
        // to force the hand-written fallback and make it a real bit-for-bit
        // check of the literals. (Compare with ==, never memcmp: x86-64 long
        // double is an 80-bit value in a 16-byte slot, and memcmp would also
        // compare six bytes of padding.)
        ok(mconst::pi     == std::numbers::pi     &&
           mconst::e      == std::numbers::e      &&
           mconst::sqrt2  == std::numbers::sqrt2  &&
           mconst::ln2    == std::numbers::ln2    &&
           mconst::log2e  == std::numbers::log2e  &&
           mconst::egamma == std::numbers::egamma &&
           mconst::phi    == std::numbers::phi,
           "constants are bit-identical to std::numbers (double)");
        ok(mconst::pi_v<long double>    == std::numbers::pi_v<long double> &&
           mconst::ln2_v<long double>   == std::numbers::ln2_v<long double> &&
           mconst::log2e_v<long double> == std::numbers::log2e_v<long double>,
           "constants are bit-identical to std::numbers (long double)");
#endif

        // --- std::complex_literals, re-exported through matrix_literals ---
        using namespace matrix_literals;
        auto z = 3.0 + 4.0i;
        ok(z.real() == 3.0 && z.imag() == 4.0,     "3.0 + 4.0i builds complex<double>(3,4)");
        ok(std::abs(z) == 5.0,                     "|3+4i| == 5");

        Matrix<std::complex<double>> Az(2, 2);
        Az = {{1.0 + 2.0i, 3.0 + 0.0i}, {0.0 + 0.0i, 1.0i}};
        ok(Az(0, 0) == std::complex<double>(1, 2), "literals work inside an initializer list");
        ok(Az(1, 1) == std::complex<double>(0, 1), "bare 1.0i is the imaginary unit");
        ok(Az.H()(0, 0) == std::complex<double>(1, -2),
                                                   "H() conjugates a literal-built matrix");
        // A literal suffix cannot be shadowed by a local named i — which is the
        // whole reason it is preferred over a global imaginary-unit constant.
        { int i = 7; auto w = 2.0i; (void)i;
          ok(w == std::complex<double>(0, 2),      "4.0i still works with a local `i` in scope"); }

        // Constants and complex together: exp(i*pi) + 1 == 0.
        ok(std::abs(std::exp(std::complex<double>(0, 1) * pi) + 1.0) < 1e-15,
                                                   "Euler: exp(i*pi) + 1 == 0");
        // And on a matrix: rotation by pi/3 applied three times is -I.
        Matrix<double> R(2, 2);
        const double th = pi / 3.0;
        R = {{std::cos(th), -std::sin(th)}, {std::sin(th), std::cos(th)}};
        ok(same(R * R * R, Id(2) * -1.0, 1e-14),   "rotation by pi/3 cubed is -I");
    }

    // ═══════════════════════════════════════════════════════════════════
    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  " << (checks - failures) << " / " << checks << " checks passed\n";
    if (failures) std::cout << "  \033[31m" << failures << " FAILED\033[0m\n";
    else          std::cout << "  \033[32mall green\033[0m\n";
    std::cout << "════════════════════════════════════════\n";
    return failures ? 1 : 0;
}
