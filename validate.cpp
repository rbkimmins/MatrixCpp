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
        // division moved to ediv() / the /dot/ spelling when this changed, so
        // these check both meanings and that they are genuinely different.
        {
            Matrix<double> X = A / B;
            ok(same(X * B, A),               "A / B is right division: (A/B)*B == A");
            ok(same(X, A * B.inverse()),     "A / B agrees with A * inv(B)");
            Matrix<double> eA(2, 2); eA = {{1.0/5, 2.0/6}, {3.0/7, 4.0/8}};
            ok(same(A.ediv(B), eA),          "ediv is element-wise division");
            ok(same(A /dot/ B, eA),          "A /dot/ B is the same as A.ediv(B)");
            ok(!same(X, eA),                 "right division and ediv really do differ");
            ok(same(A % B, A.emul(B)),       "emul is a named spelling of operator%");
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
        ok(!same(tan(A), sin(A).ediv(cos(A)), 1e-6),
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
    }

    // ═══════════════════════════════════════════════════════════════════
    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  " << (checks - failures) << " / " << checks << " checks passed\n";
    if (failures) std::cout << "  \033[31m" << failures << " FAILED\033[0m\n";
    else          std::cout << "  \033[32mall green\033[0m\n";
    std::cout << "════════════════════════════════════════\n";
    return failures ? 1 : 0;
}
