// ==========================================================================
//  GPU correctness — every result checked against basic/
// ==========================================================================
//
// The CPU library is the reference. It is already validated against
// NumPy/SciPy (261 comparisons in benchmarks/), so agreeing with it transitively
// agrees with them, and any disagreement here is the GPU's.
//
// Where a comparison against the CPU is weak — a factorisation whose factors
// are only unique up to signs, an eigenvector scaled differently — the check
// is a RESIDUAL instead: Q*R == A is true regardless of which sign convention
// either library picked, and it is the property callers actually depend on.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// The full CPU package, because it is the reference this test
// measures against - gpu_matrix.hpp alone pulls only basic/matrix.hpp.
#include "../../basic/MatrixCpp.hpp"
#include "../MatrixGpu.hpp"
#include "../../shared/interop.hpp"

#include <sstream>

using namespace mcpu;   // bare Matrix<> is the CPU one; the GPU twin is mgpu::Matrix

static int g_pass = 0, g_fail = 0;

// Relative Frobenius distance, which is scale-free and so needs one tolerance
// rather than one per test size.
static double relerr(const Matrix<double>& A, const Matrix<double>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return 1e9;
    double num = 0, den = 0;
    for (long i = 0; i < A.rows(); ++i)
        for (long j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            num += d * d;
            den += B(i, j) * B(i, j);
        }
    return std::sqrt(num) / (std::sqrt(den) + 1e-300);
}

// Same measure for complex, so the two families of tests read alike.
static double relerr(const Matrix<std::complex<double>>& A,
                     const Matrix<std::complex<double>>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return 1e9;
    double num = 0, den = 0;
    for (long i = 0; i < A.rows(); ++i)
        for (long j = 0; j < A.cols(); ++j) {
            num += std::norm(A(i, j) - B(i, j));
            den += std::norm(B(i, j));
        }
    return std::sqrt(num) / (std::sqrt(den) + 1e-300);
}

static void check(const std::string& name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    (ok ? g_pass : g_fail)++;
    if (!ok) std::printf("  FAIL  %-38s  err = %.3e  (tol %.1e)\n", name.c_str(), err, tol);
}
static void checkTrue(const std::string& name, bool ok) {
    (ok ? g_pass : g_fail)++;
    if (!ok) std::printf("  FAIL  %-38s\n", name.c_str());
}

// EXPLICITLY SEEDED, and it matters. mcpu's unseeded set_Ran_values derives
// its seed from whole seconds since midnight, so every call inside the same
// second produces the SAME matrix -- which quietly turned tests like
// "A - B" into "A - A" and made them far weaker than they looked. A
// decrementing counter gives each matrix its own stream (ran2 requires the
// seed to be negative).
static long g_seed = -12345;
static Matrix<double> randMat(long r, long c, double lo = -1, double hi = 1) {
    Matrix<double> A(r, c);
    A.set_Ran_values(lo, hi, --g_seed);
    return A;
}

// Symmetric positive definite: A^T A + n I is SPD for any A, and the shift
// keeps the condition number sane so the comparison tests numerics, not luck.
static Matrix<double> spd(long n) {
    Matrix<double> A = randMat(n, n);
    Matrix<double> S = A.T() * A;
    for (long i = 0; i < n; ++i) S(i, i) += double(n);
    return S;
}

int main() {
    if (!mgpu::available()) {
        std::printf("no CUDA device - nothing to test\n");
        return 77;
    }
    std::printf("%s\n\n", mgpu::describe().c_str());

    const double TOL = 1e-11;   // fp64, relative

    // ── Transfer ───────────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(97, 43);
        check("round trip host->device->host", relerr(mgpu::upload(A).cpu(), A), 0.0);

        auto dA = mgpu::upload(A);
        auto dB = dA;                      // deep copy, device to device
        check("device copy constructor", relerr(dB.cpu(), A), 0.0);
        auto dC = std::move(dB);
        check("device move constructor", relerr(dC.cpu(), A), 0.0);
        checkTrue("moved-from buffer released", dB.size() == 0 || dB.data() != dC.data());
    }

    // ── Element-wise ───────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(64, 51), B = randMat(64, 51);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);

        check("A + B", relerr((dA + dB).cpu(), A + B), TOL);
        check("A - B", relerr((dA - dB).cpu(), A - B), TOL);
        check("A % B (Hadamard)", relerr((dA % dB).cpu(), A % B), TOL);
        check("A * 2.5", relerr((dA * 2.5).cpu(), A * 2.5), TOL);
        check("2.5 * A", relerr((2.5 * dA).cpu(), A * 2.5), TOL);
        check("A / 3.0", relerr((dA / 3.0).cpu(), A / 3.0), TOL);
        check("-A", relerr((-dA).cpu(), A * -1.0), TOL);

        auto dAcc = mgpu::upload(A);
        dAcc += dB;
        check("A += B", relerr(dAcc.cpu(), A + B), TOL);

        // Positive domain for the functions that need one.
        Matrix<double> P = randMat(64, 51, 0.1, 4.0);
        auto dP = mgpu::upload(P);
        check("exp",   relerr(dP.exp().cpu(),   P.exp()),   TOL);
        check("ln",    relerr(dP.ln().cpu(),    P.ln()),    TOL);
        check("log(base 7)", relerr(dP.log(7.0).cpu(), P.log(7.0)), TOL);
        check("log10", relerr(dP.log10().cpu(), P.log10()), TOL);
        check("lg (log2)", relerr(dP.lg().cpu(), P.lg()),   TOL);
        check("exp2",  relerr(dP.exp2().cpu(),  P.exp2()),  TOL);
        check("sqrt",  relerr(dP.sqrt().cpu(),  P.sqrt()),  TOL);
        check("sin",   relerr(dP.sin().cpu(),   P.sin()),   TOL);
        check("cos",   relerr(dP.cos().cpu(),   P.cos()),   TOL);
        check("tanh",  relerr(dP.tanh().cpu(),  P.tanh()),  TOL);
        check("abs",   relerr(dA.abs().cpu(),   A.abs()),   TOL);
        check("pow(A, 3)", relerr(dP.pow(3.0).cpu(), P.pow(3.0)), TOL);
        check("pow2 (square)", relerr(dP.pow2().cpu(), P.pow(2.0)), TOL);
    }

    // ── Fused expressions ──────────────────────────────────────────────
    //
    // The fused path must agree with the eager one BIT FOR BIT, not merely to
    // a tolerance: it runs the same operations in the same order on the same
    // hardware, so any difference at all would mean the postfix program does
    // not encode what the chain says.
    {
        Matrix<double> A = randMat(101, 67), B = randMat(101, 67), C = randMat(101, 67);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B), dC = mgpu::upload(C);

        check("fused: (A%B).exp().sqrt()",
              relerr((dA.lazy() % dB).exp().sqrt().eval().cpu(),
                     (dA % dB).exp().sqrt().cpu()),
              0.0);

        check("fused: A - B + C",
              relerr(((dA.lazy() - dB) + dC).eval().cpu(), (dA - dB + dC).cpu()), 0.0);

        check("fused: scalars both sides",
              relerr(((dA.lazy() * 3.0 + 1.5) / 2.0).eval().cpu(),
                     ((dA * 3.0 + 1.5) / 2.0).cpu()),
              0.0);

        check("fused: implicit conversion to Matrix",
              relerr(mgpu::Matrix<double>((dA.lazy() % dB).tanh()).cpu(), (dA % dB).tanh().cpu()),
              0.0);

        check("fused: matrix on the left of a lazy chain",
              relerr((dC + (dA.lazy() % dB)).eval().cpu(), (dC + (dA % dB)).cpu()), 0.0);

        check("fused: nested binary on both sides",
              relerr(((dA.lazy() + dB) % (dC.lazy() - dA)).eval().cpu(),
                     ((dA + dB) % (dC - dA)).cpu()),
              0.0);

        // Deduplication: the same operand used repeatedly costs one slot.
        auto rep = (dA.lazy() % dA) + (dA.lazy() - dA);
        checkTrue("fused: repeated operand uses one input slot", rep.inputs() == 1);
        check("fused: repeated operand result",
              relerr(rep.eval().cpu(), ((A % A) + (A - A))), 1e-14);

        // A chain longer than one program can hold must still be correct: it
        // spills into a second kernel rather than failing.
        auto e = dA.lazy();
        Matrix<double> ref = A;
        for (int i = 0; i < 40; ++i) {
            e = e.tanh() + 0.25;
            ref = ref.tanh() + 0.25;
        }
        checkTrue("fused: long chain spilled", e.spills() > 0);
        check("fused: long chain still correct", relerr(e.eval().cpu(), ref), 1e-13);

        // Stack depth: a deeply right-nested tree must spill rather than
        // overrun the per-thread stack.
        auto deep = dA.lazy();
        for (int i = 0; i < 12; ++i) deep = dB.lazy() % deep;
        Matrix<double> refDeep = A;
        for (int i = 0; i < 12; ++i) refDeep = B % refDeep;
        check("fused: deep right-nested tree", relerr(deep.eval().cpu(), refDeep), 1e-13);

        // Many distinct operands, up to and past the input limit.
        std::vector<Matrix<double>> hs;
        std::vector<mgpu::Matrix<double>> ds;
        for (int i = 0; i < 12; ++i) {
            hs.push_back(randMat(101, 67));
            ds.push_back(mgpu::upload(hs.back()));
        }
        auto many = ds[0].lazy();
        Matrix<double> refMany = hs[0];
        for (int i = 1; i < 12; ++i) {
            many = many + ds[(std::size_t)i];
            refMany = refMany + hs[(std::size_t)i];
        }
        check("fused: 12 distinct inputs", relerr(many.eval().cpu(), refMany), 1e-13);

        bool threw = false;
        try {
            mgpu::Matrix<double> small(3, 3);
            (void)(dA.lazy() + small).eval();
        } catch (const mgpu::Error&) { threw = true; }
        checkTrue("fused: shape mismatch throws", threw);

        Matrix<float> Af(64, 40), Bf(64, 40);
        Af.set_Ran_values(0.5, 2.0);
        Bf.set_Ran_values(0.5, 2.0);
        auto dAf = mgpu::upload(Af), dBf = mgpu::upload(Bf);
        Matrix<float> fused = (dAf.lazy() % dBf).exp().eval().cpu();
        Matrix<float> eager = (dAf % dBf).exp().cpu();
        double fe = 0;
        for (long i = 0; i < fused.rows(); ++i)
            for (long j = 0; j < fused.cols(); ++j)
                fe = std::max(fe, (double)std::fabs(fused(i, j) - eager(i, j)));
        check("fused: float path", fe, 0.0);
    }

    // ── GEMM ───────────────────────────────────────────────────────────
    {
        // Non-square and non-multiple-of-anything on purpose: the row-major /
        // column-major swap in the backend is exactly where a square-only test
        // would hide a transposed result.
        struct { long m, k, n; } shapes[] = {
            {64, 64, 64}, {129, 77, 203}, {1, 512, 1}, {512, 1, 512}, {333, 1, 1}, {1, 1, 333}};
        for (auto s : shapes) {
            Matrix<double> A = randMat(s.m, s.k), B = randMat(s.k, s.n);
            char nm[64];
            std::snprintf(nm, sizeof(nm), "gemm %ldx%ld * %ldx%ld", s.m, s.k, s.k, s.n);
            check(nm, relerr((mgpu::upload(A) * mgpu::upload(B)).cpu(), A * B), TOL);
        }

        Matrix<double> A = randMat(80, 37);
        check("transpose", relerr(mgpu::upload(A).T().cpu(), A.T()), 0.0);
        check("A.T()*A", relerr((mgpu::upload(A).T() * mgpu::upload(A)).cpu(), A.T() * A), TOL);

        // Fused C = alpha*A*B + beta*C
        Matrix<double> X = randMat(40, 40), Y = randMat(40, 40), C = randMat(40, 40);
        auto dC = mgpu::upload(C);
        dC.gemmInto(mgpu::upload(X), mgpu::upload(Y), 2.0, 3.0);
        check("gemmInto (alpha/beta)", relerr(dC.cpu(), (X * Y) * 2.0 + C * 3.0), TOL);
    }

    // ══════════════════════════════════════════════════════════════════
    //  COMPLEX
    // ══════════════════════════════════════════════════════════════════
    //
    // basic/ has had complex support throughout for a long time and is checked
    // against NumPy, so it is the reference here exactly as it is for the real
    // tests. std::complex<T> and the device buffer are the same bytes, so the
    // round trip below is also what proves the layout assumption the whole
    // complex path rests on.
    {
        using Cd = std::complex<double>;
        auto randCx = [&](long r, long c) {
            Matrix<double> a = randMat(r, c), b = randMat(r, c);
            Matrix<Cd> out(r, c);
            for (long i = 0; i < r; ++i)
                for (long j = 0; j < c; ++j) out(i, j) = Cd(a(i, j), b(i, j));
            return out;
        };

        Matrix<Cd> A = randCx(37, 29), B = randCx(37, 29);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);

        check("complex: round trip", relerr(dA.cpu(), A), 0.0);
        check("complex: A + B", relerr((dA + dB).cpu(), A + B), TOL);
        check("complex: A - B", relerr((dA - dB).cpu(), A - B), TOL);
        check("complex: A % B (Hadamard)", relerr((dA % dB).cpu(), A % B), TOL);
        check("complex: A / B", relerr((dA / dB).cpu(), A.div(B)), 1e-11);
        check("complex: scalar multiply", relerr((dA * Cd(2.0, -1.5)).cpu(), A * Cd(2.0, -1.5)),
              TOL);
        check("complex: -A", relerr((-dA).cpu(), A * Cd(-1.0, 0.0)), TOL);

        check("complex: exp", relerr(dA.exp().cpu(), A.exp()), 1e-11);
        check("complex: sqrt", relerr(dA.sqrt().cpu(), A.sqrt()), 1e-11);
        check("complex: ln", relerr(dA.ln().cpu(), A.ln()), 1e-11);
        check("complex: sin", relerr(dA.sin().cpu(), A.sin()), 1e-11);
        check("complex: tanh", relerr(dA.tanh().cpu(), A.tanh()), 1e-11);
        check("complex: conj", relerr(dA.conj().cpu(), A.conj()), 0.0);

        // Parts drop to a REAL matrix, so these check the type change as much
        // as the values.
        Matrix<double> re = dA.real().cpu(), im = dA.imag().cpu(), ab = dA.abs().cpu();
        Matrix<double> ar = dA.arg().cpu();
        double e = 0, e2 = 0, e3 = 0, e4 = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) {
                e = std::max(e, std::fabs(re(i, j) - A(i, j).real()));
                e2 = std::max(e2, std::fabs(im(i, j) - A(i, j).imag()));
                e3 = std::max(e3, std::fabs(ab(i, j) - std::abs(A(i, j))));
                e4 = std::max(e4, std::fabs(ar(i, j) - std::arg(A(i, j))));
            }
        check("complex: real()", e, 0.0);
        check("complex: imag()", e2, 0.0);
        check("complex: abs()", e3, 1e-14);
        check("complex: arg()", e4, 1e-14);

        // T() vs H() -- the distinction that matters most for complex data.
        check("complex: T() is the plain transpose", relerr(dA.T().cpu(), A.T()), 0.0);
        check("complex: H() is the conjugate transpose", relerr(dA.H().cpu(), A.H()), 0.0);

        check("complex: norm() is real", std::fabs(dA.norm() - A.norm()) / A.norm(), 1e-12);
        check("complex: sum", std::abs(dA.sum() - ::sum(A)) / std::abs(::sum(A)), 1e-12);

        // GEMM
        Matrix<Cd> P = randCx(40, 55), Q = randCx(55, 33);
        check("complex: gemm", relerr((mgpu::upload(P) * mgpu::upload(Q)).cpu(), P * Q), 1e-12);
        check("complex: A.H() * A", relerr((dA.H() * dA).cpu(), A.H() * A), 1e-12);

        // Fused expressions must work for complex too.
        check("complex: fused chain",
              relerr(((dA.lazy() % dB) + dA).exp().eval().cpu(), ((A % B) + A).exp()), 1e-11);
        // Not bit-identical here, unlike the real case: fusing the multiply
        // and the exp into one kernel lets nvcc contract them into an FMA
        // across what was an operation boundary, where the eager path's store
        // and reload force a rounding to double in between. About 0.1 ulp.
        check("complex: fused == eager to a rounding",
              relerr(((dA.lazy() % dB).exp()).eval().cpu(), (dA % dB).exp().cpu()), 1e-15);

        // --- factorisations ---
        const long n = 64;
        Matrix<Cd> M = randCx(n, n);

        // LU: L*U == P*A
        {
            auto [dL, dU, piv] = mgpu::upload(M).lu();
            Matrix<Cd> L = dL.cpu(), U = dU.cpu(), PA = M;
            for (long i = 0; i < n; ++i) {
                const long q = piv[(std::size_t)i] - 1;
                if (q != i)
                    for (long j = 0; j < n; ++j) std::swap(PA(i, j), PA(q, j));
            }
            check("complex: lu, L*U == P*A", relerr(L * U, PA), 1e-11);
        }

        // Hermitian positive definite: A^H A + nI.
        Matrix<Cd> Herm = M.H() * M;
        for (long i = 0; i < n; ++i) Herm(i, i) += Cd(double(n), 0.0);

        {
            Matrix<Cd> L = mgpu::upload(Herm).cholesky().cpu();
            check("complex: cholesky, L*L^H == A", relerr(L * L.H(), Herm), 1e-11);
        }

        // QR: Q must be UNITARY, not merely orthogonal -- Q^H Q == I.
        {
            auto [dQ, dR] = mgpu::upload(M).qr();
            Matrix<Cd> Q = dQ.cpu(), R = dR.cpu();
            check("complex: qr, Q*R == A", relerr(Q * R, M), 1e-11);
            Matrix<Cd> QhQ = Q.H() * Q;
            double u = 0;
            for (long i = 0; i < n; ++i)
                for (long j = 0; j < n; ++j)
                    u = std::max(u, std::abs(QhQ(i, j) - Cd(i == j ? 1.0 : 0.0, 0.0)));
            check("complex: qr, Q^H Q == I", u, 1e-11);
        }

        // SVD, including the wide case that goes through the conjugate
        // transpose identity in the backend.
        for (auto sh : {std::pair<long, long>{70, 40}, {50, 50}, {30, 64}}) {
            Matrix<Cd> X = randCx(sh.first, sh.second);
            auto [dU, dS, dV] = mgpu::upload(X).svd();
            char nm[64];
            std::snprintf(nm, sizeof(nm), "complex: svd %ldx%ld, U*S*V^H == A", sh.first,
                          sh.second);
            check(nm, relerr(dU.cpu() * dS.cpu() * dV.cpu().H(), X), 1e-10);

            std::snprintf(nm, sizeof(nm), "complex: svdvals %ldx%ld are real", sh.first,
                          sh.second);
            Matrix<double> sv = mgpu::upload(X).svdvals().cpu();
            auto [cU, cS, cV] = X.svd();
            const long kk = std::min(sh.first, sh.second);
            double se = 0;
            for (long i = 0; i < kk; ++i)
                se = std::max(se, std::fabs(sv(i, 0) - cS(i, i)) / (std::fabs(cS(i, i)) + 1e-300));
            check(nm, se, 1e-10);
        }

        // Hermitian eigenproblem: real eigenvalues, unitary eigenvectors.
        {
            auto [dW, dV] = mgpu::upload(Herm).eigSym();
            Matrix<double> w = dW.cpu();
            Matrix<Cd> V = dV.cpu();
            Matrix<Cd> D(n, n);
            for (long i = 0; i < n; ++i) D(i, i) = Cd(w(i, 0), 0.0);
            check("complex: eigSym, A*V == V*diag(w)", relerr(Herm * V, V * D), 1e-10);
            bool ascending = true;
            for (long i = 1; i < n; ++i)
                if (w(i, 0) < w(i - 1, 0)) ascending = false;
            checkTrue("complex: eigSym values real and ascending", ascending);
        }

        // solve
        {
            Matrix<Cd> rhs = randCx(n, 2);
            Matrix<Cd> X = mgpu::upload(Herm).solve(mgpu::upload(rhs)).cpu();
            check("complex: solve, A*X == B", relerr(Herm * X, rhs), 1e-10);
            check("complex: factorize + solve",
                  relerr(Herm * mgpu::upload(Herm).factorize().solve(mgpu::upload(rhs)).cpu(), rhs),
                  1e-10);
        }

        // det, against the CPU on a size where the value is trustworthy
        {
            Matrix<Cd> S9 = randCx(9, 9);
            check("complex: det", std::abs(mgpu::upload(S9).det() - S9.det()) /
                                      (std::abs(S9.det()) + 1e-300),
                  1e-10);
        }

        // complex<float>
        {
            using Cf = std::complex<float>;
            Matrix<Cf> Af(24, 24);
            for (long i = 0; i < 24; ++i)
                for (long j = 0; j < 24; ++j) Af(i, j) = Cf(float(A(i, j).real()), float(A(i, j).imag()));
            Matrix<Cf> back = mgpu::upload(Af).H().H().cpu();
            double fe = 0;
            for (long i = 0; i < 24; ++i)
                for (long j = 0; j < 24; ++j) fe = std::max(fe, (double)std::abs(back(i, j) - Af(i, j)));
            check("complex<float>: H() twice is identity", fe, 0.0);
            Matrix<Cf> prod = (mgpu::upload(Af) * mgpu::upload(Af)).cpu(), ref = Af * Af;
            fe = 0;
            for (long i = 0; i < 24; ++i)
                for (long j = 0; j < 24; ++j)
                    fe = std::max(fe, (double)std::abs(prod(i, j) - ref(i, j)));
            check("complex<float>: gemm", fe / 24.0, 1e-4);
        }
    }

    // ── Transposed products ────────────────────────────────────────────
    {
        Matrix<double> A = randMat(150, 40), B = randMat(150, 25);
        Matrix<double> C = randMat(40, 90), D = randMat(25, 90);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        auto dC = mgpu::upload(C), dD = mgpu::upload(D);
        check("tMul: A^T * B", relerr(dA.tMul(dB).cpu(), A.T() * B), TOL);
        check("mulT: A * B^T", relerr(dC.mulT(dD).cpu(), C * D.T()), TOL);
        check("gram: A^T * A", relerr(dA.gram().cpu(), A.T() * A), TOL);
        // Must agree exactly with the materialising route, since it is the
        // same cuBLAS kernel with a flag set rather than a copy made.
        check("tMul == T() * B", relerr(dA.tMul(dB).cpu(), (dA.T() * dB).cpu()), 0.0);

        bool threw = false;
        try { (void)dA.tMul(dC); } catch (const mgpu::Error&) { threw = true; }
        checkTrue("tMul shape mismatch throws", threw);
    }

    // ── Mixed-precision solve ──────────────────────────────────────────
    {
        const long n = 512;
        Matrix<double> A = spd(n), B = randMat(n, 3);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);

        int iters = 0;
        Matrix<double> X = dA.solveMixed(dB, mgpu::Matrix<double>::Factor::Single, &iters).cpu();
        // The whole claim is that a single-precision factorisation still gives
        // a double-precision answer, so the residual is checked at fp64
        // tolerance, not fp32.
        check("solveMixed(Single): A*X == B", relerr(A * X, B), 1e-11);
        checkTrue("solveMixed reports refinement steps", iters != 0);

        Matrix<double> Xd = dA.solveMixed(dB, mgpu::Matrix<double>::Factor::Double).cpu();
        check("solveMixed(Double): A*X == B", relerr(A * Xd, B), 1e-11);

        // Half converges on fewer matrices; when it cannot, cuSOLVER falls
        // back to full precision and says so with a negative count. Either
        // way the answer must be right, which is the point of the test.
        int hiters = 0;
        Matrix<double> Xh = dA.solveMixed(dB, mgpu::Matrix<double>::Factor::Half, &hiters).cpu();
        check("solveMixed(Half): A*X == B", relerr(A * Xh, B), 1e-9);

        check("solveMixed agrees with solve", relerr(X, dA.solve(dB).cpu()), 1e-10);
    }

    // ── Factor once, solve many ─────────────────────────────────────────
    {
        const long n = 200;
        Matrix<double> A = spd(n);
        auto dA = mgpu::upload(A);
        auto lu = dA.factorize();

        for (int k = 0; k < 3; ++k) {
            Matrix<double> B = randMat(n, 2);
            Matrix<double> X = lu.solve(mgpu::upload(B)).cpu();
            char nm[64];
            std::snprintf(nm, sizeof(nm), "factorize: solve #%d", k + 1);
            check(nm, relerr(A * X, B), 1e-11);
        }
        // Reusing the factor must give bit-identical results to a fresh one.
        Matrix<double> B = randMat(n, 2);
        check("factorize: matches solve()", relerr(lu.solve(mgpu::upload(B)).cpu(),
                                                   dA.solve(mgpu::upload(B)).cpu()), 0.0);

        Matrix<double> Small = randMat(9, 9);
        check("factorize: det matches basic/",
              std::fabs(mgpu::upload(Small).factorize().det() - Small.det()) /
                  (std::fabs(Small.det()) + 1e-300),
              1e-10);
        double e = 0;
        Matrix<double> P = A * lu.inv().cpu();
        for (long i = 0; i < n; ++i)
            for (long j = 0; j < n; ++j) e = std::max(e, std::fabs(P(i, j) - (i == j ? 1.0 : 0.0)));
        check("factorize: inv", e, 1e-10);
    }

    // ── Parity with mcpu: the rest of <cmath>, logicals, dot ───────────
    {
        Matrix<double> P = randMat(20, 15, 1.2, 4.0);    // >= 1, the acosh domain
        Matrix<double> Q = randMat(20, 15, 0.1, 0.9);    // safe for atanh
        Matrix<double> R = randMat(20, 15, -2.0, 2.0);
        auto dP = mgpu::upload(P), dQ = mgpu::upload(Q), dR = mgpu::upload(R);

        check("asinh vs basic/", relerr(dR.asinh().cpu(), R.asinh()), TOL);
        check("acosh vs basic/", relerr(dP.acosh().cpu(), P.acosh()), TOL);
        check("atanh vs basic/", relerr(dQ.atanh().cpu(), Q.atanh()), TOL);
        check("log1p vs basic/", relerr(dQ.log1p().cpu(), Q.log1p()), TOL);
        check("expm1 vs basic/", relerr(dQ.expm1().cpu(), Q.expm1()), TOL);
        check("fix (trunc) vs basic/", relerr(dR.fix().cpu(), R.fix()), 0.0);
        check("atan2 vs basic/", relerr(dR.atan2(dP).cpu(), R.atan2(P)), TOL);
        check("hypot vs basic/", relerr(dR.hypot(dP).cpu(), R.hypot(P)), TOL);
        // mcpu's mod/rem take a SCALAR only; the GPU also accepts a matrix,
        // so the scalar form is checked against basic/ and the matrix form
        // against fmod/remainder directly.
        check("mod(scalar) vs basic/", relerr(dR.mod(0.7).cpu(), R.mod(0.7)), TOL);
        check("rem(scalar) vs basic/", relerr(dR.rem(0.7).cpu(), R.rem(0.7)), TOL);
        {
            Matrix<double> gm = dR.mod(dP).cpu(), gr = dR.rem(dP).cpu();
            double em = 0, er = 0;
            for (long i = 0; i < R.rows(); ++i)
                for (long j = 0; j < R.cols(); ++j) {
                    double m = std::fmod(R(i, j), P(i, j));
                    if (m != 0.0 && ((m < 0.0) != (P(i, j) < 0.0))) m += P(i, j);
                    em = std::max(em, std::fabs(gm(i, j) - m));
                    er = std::max(er, std::fabs(gr(i, j) - std::fmod(R(i, j), P(i, j))));
                }
            check("mod(matrix), MATLAB convention", em, 0.0);
            check("rem(matrix) == fmod", er, 0.0);
        }

        // log1p is not shorthand: it must beat log(1+x) where x is tiny.
        Matrix<double> tiny(1, 4);
        tiny(0, 0) = 1e-16; tiny(0, 1) = 1e-14; tiny(0, 2) = 1e-12; tiny(0, 3) = 1e-10;
        Matrix<double> l1 = mgpu::upload(tiny).log1p().cpu();
        double worst = 0, naive = 0;
        for (long j = 0; j < 4; ++j) {
            const double want = std::log1p(tiny(0, j));   // the correct reference
            worst = std::max(worst, std::fabs(l1(0, j) - want) / want);
            naive = std::max(naive, std::fabs(std::log(1.0 + tiny(0, j)) - want) / want);
        }
        check("log1p matches std::log1p for tiny x", worst, 1e-15);
        // and it is not merely shorthand: log(1+x) is far worse here
        checkTrue("log1p beats log(1+x) for tiny x", naive > 1e3 * std::max(worst, 1e-17));

        // Logical masks
        Matrix<double> ma = dR.gt(0.0).cpu(), mb = dP.gt(1.5).cpu();
        auto dma = mgpu::upload(ma), dmb = mgpu::upload(mb);
        double e = 0, e2 = 0, e3 = 0, e4 = 0;
        Matrix<double> ga = dma.land(dmb).cpu(), go = dma.lor(dmb).cpu();
        Matrix<double> gx = dma.lxor(dmb).cpu(), gn = dma.lnot().cpu();
        for (long i = 0; i < ma.rows(); ++i)
            for (long j = 0; j < ma.cols(); ++j) {
                const bool a = ma(i, j) != 0.0, b = mb(i, j) != 0.0;
                e = std::max(e, std::fabs(ga(i, j) - double(a && b)));
                e2 = std::max(e2, std::fabs(go(i, j) - double(a || b)));
                e3 = std::max(e3, std::fabs(gx(i, j) - double(a != b)));
                e4 = std::max(e4, std::fabs(gn(i, j) - double(!a)));
            }
        check("land", e, 0.0);
        check("lor", e2, 0.0);
        check("lxor", e3, 0.0);
        check("lnot", e4, 0.0);

        // dot, both orientations, and the Hermitian form for complex
        Matrix<double> u = randMat(1, 40), v = randMat(1, 40);
        double want = 0;
        for (long j = 0; j < 40; ++j) want += u(0, j) * v(0, j);
        check("dot", std::fabs(mgpu::upload(u).dot(mgpu::upload(v)) - want) / std::fabs(want),
              1e-12);
        {
            using Cd = std::complex<double>;
            Matrix<Cd> cu(1, 20);
            for (long j = 0; j < 20; ++j) cu(0, j) = Cd(u(0, j), v(0, j));
            Cd wantc(0, 0);
            for (long j = 0; j < 20; ++j) wantc += std::conj(cu(0, j)) * cu(0, j);
            auto dcu = mgpu::upload(cu);
            check("complex dot is Hermitian (x.dot(x) is real)",
                  std::abs(dcu.dot(dcu) - wantc) / std::abs(wantc), 1e-12);
        }
    }

    // ── Output formats, bridged from mcpu ──────────────────────────────
    {
        Matrix<double> A = randMat(3, 4);
        auto dA = mgpu::upload(A);
        checkTrue("gpu str() matches cpu str()", dA.str() == A.str());
        mcpu::matio::Opts csv;
        csv.fmt = mcpu::matio::Fmt::CSV;
        checkTrue("gpu str(CSV) matches cpu", dA.str(csv) == A.str(csv));
        std::ostringstream a, b;
        dA.print(a);
        A.print(b);
        checkTrue("gpu print() matches cpu print()", a.str() == b.str());
    }

    // ── shared/interop.hpp ─────────────────────────────────────────────
    {
        Matrix<double> A = randMat(30, 20);
        check("mx::to_gpu / to_cpu round trip", relerr(mx::roundtrip(A), A), 0.0);
        check("mx::to_gpu matches mgpu::upload",
              relerr(mx::to_cpu(mx::to_gpu(A)), mgpu::upload(A).cpu()), 0.0);
        // The pass-throughs let one template take either side.
        auto same = [](const auto& m) { return mx::to_cpu(m).norm(); };
        check("mx::to_cpu passes a host matrix through",
              std::fabs(same(A) - A.norm()) / A.norm(), 0.0);
        check("mx::to_cpu converts a device matrix",
              std::fabs(same(mgpu::upload(A)) - A.norm()) / A.norm(), 1e-13);
        checkTrue("mx vocabulary aliases mcpu's", mx::ROW == ROW && mx::COL == COL);
    }

    // ── Scalar element access ──────────────────────────────────────────
    //
    // The gap a namespace swap exposes first: mcpu code indexes elements, and
    // without these the swap does not compile at all.
    {
        Matrix<double> A = randMat(6, 5);
        auto dA = mgpu::upload(A);

        double e = 0;
        for (long i = 0; i < 6; ++i)
            for (long j = 0; j < 5; ++j) e = std::max(e, std::fabs(dA(i, j) - A(i, j)));
        check("operator()(i,j) read", e, 0.0);

        e = 0;
        for (long k = 0; k < 30; ++k) e = std::max(e, std::fabs(dA[k] - A(k / 5, k % 5)));
        check("operator[] read (row-major flat)", e, 0.0);

        dA(2, 3) = 7.25;
        dA[7] = -1.5;
        dA(0, 0) += 10.0;
        Matrix<double> got = dA.cpu();
        checkTrue("operator()(i,j) write", got(2, 3) == 7.25);
        checkTrue("operator[] write", got(1, 2) == -1.5);
        checkTrue("element += ", got(0, 0) == A(0, 0) + 10.0);
        // and nothing else moved
        e = 0;
        for (long i = 0; i < 6; ++i)
            for (long j = 0; j < 5; ++j) {
                if ((i == 2 && j == 3) || (i == 1 && j == 2) || (i == 0 && j == 0)) continue;
                e = std::max(e, std::fabs(got(i, j) - A(i, j)));
            }
        check("element write touches nothing else", e, 0.0);

        // Negative indices wrap from the end -- correctly, unlike the CPU's
        // `i % rows`, which for -1 yields -1 and reads before the buffer.
        checkTrue("A(-1,-1) is the last element", dA(-1, -1) == got(5, 4));
        checkTrue("A[-1] is the last element", dA[-1] == got(5, 4));

        bool threw = false;
        try { (void)dA(6, 0); } catch (const mgpu::Error&) { threw = true; }
        checkTrue("out-of-range index throws", threw);
        threw = false;
        try { (void)dA[30]; } catch (const mgpu::Error&) { threw = true; }
        checkTrue("out-of-range flat index throws", threw);

        // A 1x1 IS a scalar.
        Matrix<double> one(1, 1);
        one(0, 0) = 3.75;
        double asScalar = mgpu::upload(one);
        checkTrue("1x1 converts to a scalar", asScalar == 3.75);
        threw = false;
        try { double bad = dA; (void)bad; } catch (const mgpu::Error&) { threw = true; }
        checkTrue("non-1x1 scalar conversion throws", threw);
    }

    // ── The brownian demo's expression, on both sides ──────────────────
    //
    // This is the line the namespace swap was tried on. It exercises row
    // views, slice subtraction, element-wise power, an axis reduction and the
    // 1x1-to-scalar conversion in one go.
    {
        Matrix<double> P = randMat(12, 3);
        auto dP = mgpu::upload(P);
        double worst = 0;
        for (long i = 0; i < 12; ++i)
            for (long j = i + 1; j < 12; ++j) {
                const double rc =
                    std::sqrt(::sum((P(i, all) - P(j, all)).pow(2), ROW));
                const double rg =
                    std::sqrt(double(mgpu::sum((dP(i, all).eval() - dP(j, all).eval()).pow(2.0),
                                               ROW)));
                worst = std::max(worst, std::fabs(rc - rg) / rc);
            }
        check("brownian distance expression, cpu vs gpu", worst, 1e-13);

        // And the element-write loop the demo uses for its Wiener increment.
        mgpu::Matrix<double> dW(1, 3);
        for (int k = 0; k < 3; k++) dW(0, k) = 0.1 * (k + 1);
        Matrix<double> hw = dW.cpu();
        // Not compared against the decimal it looks like -- 0.1 * 3 is not
        // 0.3 in binary -- and not demanded exact either: under -march=native
        // GCC contracts `hw - 0.1*(k+1)` into an fma, so the product is never
        // rounded and the difference comes out as its rounding error rather
        // than zero. One ulp is the honest tolerance.
        double we = 0;
        for (int k = 0; k < 3; k++) {
            const volatile double want = 0.1 * (k + 1);   // volatile: no contraction
            we = std::max(we, std::fabs(hw(0, k) - want));
        }
        check("element-write loop", we, 0.0);
    }

    // ── General (non-symmetric) eigenproblem ───────────────────────────
    //
    // Restored after an earlier version of this suite concluded, wrongly, that
    // cusolverDnXgeev was broken. It was a data-type mistake on our side --
    // see detail/backend.hpp.
    {
        const long n = 48;
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);

        // Eigenvalues are a SET: cuSOLVER and LAPACK need not order them the
        // same way, so both are sorted before comparing.
        Matrix<std::complex<double>> gw = dA.eigvals().cpu();
        Matrix<std::complex<double>> cw = A.eigvals();
        auto key = [](const std::complex<double>& z) {
            return std::make_pair(z.real(), z.imag());
        };
        std::vector<std::complex<double>> g, c;
        for (long i = 0; i < n; ++i) { g.push_back(gw(i, 0)); c.push_back(cw(i, 0)); }
        std::sort(g.begin(), g.end(), [&](auto a, auto b) { return key(a) < key(b); });
        std::sort(c.begin(), c.end(), [&](auto a, auto b) { return key(a) < key(b); });
        double worst = 0;
        for (long i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(g[(std::size_t)i] - c[(std::size_t)i]));
        check("eigvals vs basic/ (sorted)", worst, 1e-9);

        // Independent of either library: the trace and the determinant are the
        // sum and product of the eigenvalues.
        std::complex<double> sum(0, 0), prod(1, 0);
        for (long i = 0; i < n; ++i) { sum += gw(i, 0); prod *= gw(i, 0); }
        double tr = 0;
        for (long i = 0; i < n; ++i) tr += A(i, i);
        check("eigvals: sum == trace", std::abs(sum - std::complex<double>(tr, 0)) /
                                           (std::fabs(tr) + 1e-300), 1e-10);
        check("eigvals: product == det",
              std::abs(prod - std::complex<double>(A.det(), 0)) / (std::abs(A.det()) + 1e-300),
              1e-8);

        // A real spectrum, so eig() is allowed: symmetric matrices qualify.
        Matrix<double> S = spd(40);
        auto [w, V] = mgpu::upload(S).eig();
        Matrix<double> hw = w.cpu(), hv = V.cpu();
        Matrix<double> D(40, 40);
        for (long i = 0; i < 40; ++i) D(i, i) = hw(i, 0);
        check("eig: A*V == V*diag(w)", relerr(S * hv, hv * D), 1e-9);

        // ... and eig() must REFUSE a complex spectrum rather than silently
        // dropping the imaginary part. A rotation has none that are real.
        Matrix<double> Rot(2, 2);
        Rot(0, 0) = 0; Rot(0, 1) = -1; Rot(1, 0) = 1; Rot(1, 1) = 0;
        bool threw = false;
        try { (void)mgpu::upload(Rot).eig(); } catch (const mgpu::Error&) { threw = true; }
        checkTrue("eig refuses a complex spectrum", threw);
        // but eigvals handles it: the eigenvalues are +-i
        Matrix<std::complex<double>> rv = mgpu::upload(Rot).eigvals().cpu();
        double im = std::max(std::fabs(rv(0, 0).imag()), std::fabs(rv(1, 0).imag()));
        double re = std::max(std::fabs(rv(0, 0).real()), std::fabs(rv(1, 0).real()));
        check("eigvals of a rotation is +-i", std::fabs(im - 1.0) + re, 1e-12);

        // Complex eigenvectors, unpacked from LAPACK's real storage: check the
        // residual on a matrix that genuinely has a conjugate pair.
        Matrix<double> M = randMat(16, 16);
        auto dM = mgpu::upload(M);
        Matrix<std::complex<double>> mw = dM.eigvals().cpu();
        double pairs = 0;
        for (long i = 0; i < 16; ++i) if (std::fabs(mw(i, 0).imag()) > 1e-12) pairs++;
        checkTrue("the test matrix does have complex eigenvalues", pairs > 0);
        double resid = 0;
        for (long j = 0; j < 16; ++j) {
            // det(A - lambda I) must vanish; use the smallest singular value
            // as a conditioning-safe stand-in for that determinant.
            Matrix<std::complex<double>> Sh(16, 16);
            for (long r = 0; r < 16; ++r)
                for (long c2 = 0; c2 < 16; ++c2)
                    Sh(r, c2) = std::complex<double>(M(r, c2), 0.0) -
                                (r == c2 ? mw(j, 0) : std::complex<double>(0, 0));
            auto [su, ss, sv] = Sh.svd();
            resid = std::max(resid, ss(15, 15));
        }
        check("every eigenvalue makes A - lambda I singular", resid, 1e-9);
    }

    // ── Sorting ────────────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(40, 25);
        auto dA = mgpu::upload(A);

        check("sort(ROW) vs basic/", relerr(dA.sort(ROW).cpu(), A.sort(ROW)), 0.0);
        check("sort(COL) vs basic/", relerr(dA.sort(COL).cpu(), A.sort(COL)), 0.0);
        check("sort(ROW, descending) vs basic/",
              relerr(dA.sort(ROW, true).cpu(), A.sort(ROW, true)), 0.0);
        check("sort(COL, descending) vs basic/",
              relerr(dA.sort(COL, true).cpu(), A.sort(COL, true)), 0.0);

        // Non-square both ways, since the column path transposes and a square
        // matrix would hide a swapped dimension.
        Matrix<double> W = randMat(7, 61), Tt = randMat(61, 7);
        check("sort(COL) on a wide matrix",
              relerr(mgpu::upload(W).sort(COL).cpu(), W.sort(COL)), 0.0);
        check("sort(ROW) on a tall matrix",
              relerr(mgpu::upload(Tt).sort(ROW).cpu(), Tt.sort(ROW)), 0.0);

        // sorted(): every element, shape kept.
        Matrix<double> flat = dA.sorted().cpu();
        bool ascending = true;
        for (long k = 1; k < 40 * 25; ++k)
            if (flat(k / 25, k % 25) < flat((k - 1) / 25, (k - 1) % 25)) ascending = false;
        checkTrue("sorted() is ascending across the whole matrix", ascending);
        check("sorted() keeps the shape",
              double(flat.rows() != 40 || flat.cols() != 25), 0.0);

        // sortrows carries the other columns along.
        for (long key : {0L, 3L, 24L}) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "sortrows(key=%ld) vs basic/", key);
            check(nm, relerr(dA.sortrows(key).cpu(), A.sortrows(key)), 0.0);
        }
        check("sortrows descending vs basic/",
              relerr(dA.sortrows(1, true).cpu(), A.sortrows(1, true)), 0.0);

        // Stability: with a repeated key, the original row order must survive,
        // which is what lets two sortrows calls make a two-column sort.
        Matrix<double> D(8, 2);
        for (long i = 0; i < 8; ++i) { D(i, 0) = double(i % 2); D(i, 1) = double(i); }
        Matrix<double> st = mgpu::upload(D).sortrows(0).cpu();
        bool stable = true;
        for (long i = 1; i < 8; ++i)
            if (st(i, 0) == st(i - 1, 0) && st(i, 1) < st(i - 1, 1)) stable = false;
        checkTrue("sortrows is stable on equal keys", stable);

        // unique: distinct values ascending, as a column.
        Matrix<double> R(1, 200);
        for (long j = 0; j < 200; ++j) R(0, j) = double(long(j) % 17);
        Matrix<double> u = mgpu::upload(R).unique().cpu();
        checkTrue("unique count", u.rows() == 17 && u.cols() == 1);
        double ue = 0;
        for (long i = 0; i < 17; ++i) ue = std::max(ue, std::fabs(u(i, 0) - double(i)));
        check("unique values", ue, 0.0);
        check("unique vs basic/", relerr(mgpu::upload(R).unique().cpu(), R.unique()), 0.0);

        // median, odd and even counts, and both axes.
        Matrix<double> Odd = randMat(1, 41), Even = randMat(1, 40);
        check("median (odd count) vs basic/",
              std::fabs(mgpu::upload(Odd).median() - Odd.median()), 1e-13);
        check("median (even count) vs basic/",
              std::fabs(mgpu::upload(Even).median() - Even.median()), 1e-13);
        check("median(ROW) vs basic/", relerr(dA.median(ROW).cpu(), A.median(ROW)), 1e-13);
        check("median(COL) vs basic/", relerr(dA.median(COL).cpu(), A.median(COL)), 1e-13);
        // even-length runs on both axes, so the two-element average is exercised
        Matrix<double> Ev(10, 8);
        for (long i = 0; i < 10; ++i)
            for (long j = 0; j < 8; ++j) Ev(i, j) = A(i, j);
        check("median(ROW) even run", relerr(mgpu::upload(Ev).median(ROW).cpu(), Ev.median(ROW)),
              1e-13);
        check("median(COL) even run", relerr(mgpu::upload(Ev).median(COL).cpu(), Ev.median(COL)),
              1e-13);

        // mode, including the tie rule.
        check("mode vs basic/", std::fabs(mgpu::upload(R).mode() - R.mode()), 0.0);
        Matrix<double> Tie(1, 6);
        Tie(0,0)=5; Tie(0,1)=5; Tie(0,2)=2; Tie(0,3)=2; Tie(0,4)=9; Tie(0,5)=1;
        checkTrue("mode breaks ties toward the smallest", mgpu::upload(Tie).mode() == 2.0);
        checkTrue("mode agrees with basic/ on the tie", mgpu::upload(Tie).mode() == Tie.mode());

        // complex has no order, so these must refuse to compile-or-run
        bool threw = false;
        try {
            Matrix<double> one(1, 1);
            (void)mgpu::upload(one).mode();
        } catch (const mgpu::Error&) { threw = true; }
        checkTrue("mode of a single element works", !threw);
    }

    // ── Views ──────────────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(8, 6);
        auto dA = mgpu::upload(A);

        // Reading a row and a column out.
        check("view: dA(all, 2) reads a column", relerr(dA(all, 2).eval().cpu(), A(all, 2)), 0.0);
        check("view: dA(3, all) reads a row", relerr(dA(3, all).eval().cpu(), A(3, all)), 0.0);
        check("view: col()/row() accessors",
              relerr(dA.col(2).cpu(), A(all, 2)) + relerr(dA.row(3).cpu(), A(3, all)), 0.0);

        // Writing through one.
        Matrix<double> newCol = randMat(8, 1);
        dA(all, 1) = mgpu::upload(newCol);
        Matrix<double> got = dA.cpu();
        double e = 0;
        for (long i = 0; i < 8; ++i)
            for (long j = 0; j < 6; ++j)
                e = std::max(e, std::fabs(got(i, j) - (j == 1 ? newCol(i, 0) : A(i, j))));
        check("view: assigning a column", e, 0.0);

        // Scalar fill and random refill through a view.
        auto dB = mgpu::upload(A);
        dB(2, all) = 7.5;
        got = dB.cpu();
        e = 0;
        for (long i = 0; i < 8; ++i)
            for (long j = 0; j < 6; ++j)
                e = std::max(e, std::fabs(got(i, j) - (i == 2 ? 7.5 : A(i, j))));
        check("view: scalar fill of a row", e, 0.0);

        auto dC = mgpu::upload(A);
        dC(all, 0).set_Ran_values(10.0, 11.0, 42);
        got = dC.cpu();
        bool inRange = true, restIntact = true;
        for (long i = 0; i < 8; ++i) {
            if (got(i, 0) < 10.0 || got(i, 0) >= 11.0) inRange = false;
            for (long j = 1; j < 6; ++j)
                if (got(i, j) != A(i, j)) restIntact = false;
        }
        checkTrue("view: set_Ran_values fills only the view", inRange && restIntact);

        // Sub-block view.
        auto dD = mgpu::upload(A);
        Matrix<double> patch = randMat(3, 2);
        dD.view(4, 3, 3, 2) = mgpu::upload(patch);
        got = dD.cpu();
        e = 0;
        for (long i = 0; i < 8; ++i)
            for (long j = 0; j < 6; ++j) {
                const bool in = (i >= 4 && i < 7 && j >= 3 && j < 5);
                e = std::max(e, std::fabs(got(i, j) - (in ? patch(i - 4, j - 3) : A(i, j))));
            }
        check("view: sub-block assignment", e, 0.0);

        // Arithmetic on a view materialises.
        check("view: arithmetic", relerr((dA(all, 2) * 2.0).cpu(), A(all, 2) * 2.0), TOL);

        // Compound assignment through a view -- the family that was missing
        // on both sides of the package, and had to be written as members for
        // the same reason the binary operators were.
        {
            auto dE = mgpu::upload(A);
            dE(all, 0) += mgpu::upload(newCol);
            Matrix<double> g2 = dE.cpu();
            double e2 = 0;
            for (long i = 0; i < 8; ++i)
                for (long j = 0; j < 6; ++j)
                    e2 = std::max(e2, std::fabs(g2(i, j) -
                                                (j == 0 ? A(i, j) + newCol(i, 0) : A(i, j))));
            check("view: += a matrix", e2, 0.0);

            auto dF = mgpu::upload(A);
            dF(1, all) *= 3.0;
            g2 = dF.cpu();
            e2 = 0;
            for (long i = 0; i < 8; ++i)
                for (long j = 0; j < 6; ++j)
                    e2 = std::max(e2, std::fabs(g2(i, j) - (i == 1 ? A(i, j) * 3.0 : A(i, j))));
            check("view: *= a scalar", e2, 0.0);

            auto dG = mgpu::upload(A);
            dG(all, 3) += 1.5;
            g2 = dG.cpu();
            e2 = 0;
            for (long i = 0; i < 8; ++i)
                for (long j = 0; j < 6; ++j)
                    e2 = std::max(e2, std::fabs(g2(i, j) - (j == 3 ? A(i, j) + 1.5 : A(i, j))));
            check("view: += a scalar", e2, 0.0);

            auto dH = mgpu::upload(A);
            dH.view(2, 2, 3, 3) -= mgpu::upload(randMat(3, 3)) * 0.0;   // subtracting zero
            check("view: -= leaves the matrix unchanged when subtracting zero",
                  relerr(dH.cpu(), A), 0.0);
        }

        bool threw = false;
        try { (void)dA.view(6, 0, 5, 1); } catch (const mgpu::Error&) { threw = true; }
        checkTrue("view: out-of-range region throws", threw);
        threw = false;
        try { dA(all, 0) = mgpu::upload(randMat(3, 1)); } catch (const mgpu::Error&) { threw = true; }
        checkTrue("view: shape mismatch on assign throws", threw);
    }

    // ── Comparisons and masks ──────────────────────────────────────────
    {
        Matrix<double> A = randMat(30, 20), B = randMat(30, 20);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);

        auto maskErr = [&](const Matrix<double>& got, auto pred) {
            double e = 0;
            for (long i = 0; i < A.rows(); ++i)
                for (long j = 0; j < A.cols(); ++j)
                    e = std::max(e, std::fabs(got(i, j) - (pred(i, j) ? 1.0 : 0.0)));
            return e;
        };
        check("mask: lt", maskErr(dA.lt(dB).cpu(), [&](long i, long j) { return A(i, j) < B(i, j); }),
              0.0);
        check("mask: ge", maskErr(dA.ge(dB).cpu(), [&](long i, long j) { return A(i, j) >= B(i, j); }),
              0.0);
        check("mask: gt scalar",
              maskErr(dA.gt(0.0).cpu(), [&](long i, long j) { return A(i, j) > 0.0; }), 0.0);
        check("mask: ne scalar",
              maskErr(dA.ne(0.0).cpu(), [&](long i, long j) { return A(i, j) != 0.0; }), 0.0);

        // Masks compose with the rest of the package, which is the point.
        check("mask: A % A.gt(0) keeps only positives",
              relerr((dA % dA.gt(0.0)).cpu(),
                     [&] {
                         Matrix<double> r = A;
                         for (long i = 0; i < A.rows(); ++i)
                             for (long j = 0; j < A.cols(); ++j)
                                 if (!(A(i, j) > 0.0)) r(i, j) = 0.0;
                         return r;
                     }()),
              0.0);

        long expect = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j)
                if (A(i, j) != 0.0) ++expect;
        checkTrue("nnz", dA.nnz() == expect);
        checkTrue("any", dA.any());
        checkTrue("all on a nonzero matrix", dA.all());
        Matrix<double> Z(4, 4);
        checkTrue("any on zeros is false", !mgpu::upload(Z).any());

        checkTrue("allclose with itself", dA.allclose(dA));
        checkTrue("allclose rejects a different matrix", !dA.allclose(dB));
        Matrix<double> nudged = A;
        nudged(0, 0) += 1e-12;
        checkTrue("allclose tolerates a tiny nudge", dA.allclose(mgpu::upload(nudged)));
    }

    // ── Statistics, norms, derived linear algebra ──────────────────────
    {
        Matrix<double> A = randMat(40, 25, 0.5, 3.0);
        auto dA = mgpu::upload(A);

        double m = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) m += A(i, j);
        m /= double(A.rows() * A.cols());
        double v = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) v += (A(i, j) - m) * (A(i, j) - m);
        v /= double(A.rows() * A.cols() - 1);
        check("var", std::fabs(dA.var() - v) / v, 1e-11);
        check("stddev", std::fabs(dA.stddev() - std::sqrt(v)) / std::sqrt(v), 1e-11);

        long am = 0, an = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) {
                const long k = i * A.cols() + j;
                if (A(i, j) > A(am / A.cols(), am % A.cols())) am = k;
                if (A(i, j) < A(an / A.cols(), an % A.cols())) an = k;
            }
        checkTrue("argmax", dA.argmax() == am);
        checkTrue("argmin", dA.argmin() == an);

        check("norm(Fro) vs basic/", std::fabs(dA.norm() - A.norm()) / A.norm(), 1e-12);
        check("norm(One) vs basic/",
              std::fabs(dA.norm(mgpu::Matrix<double>::NormType::One) - A.norm(NormType::One)) /
                  A.norm(NormType::One),
              1e-11);
        check("norm(Inf) vs basic/",
              std::fabs(dA.norm(mgpu::Matrix<double>::NormType::Inf) - A.norm(NormType::Inf)) /
                  A.norm(NormType::Inf),
              1e-11);
        check("norm(Two) vs basic/",
              std::fabs(dA.norm(mgpu::Matrix<double>::NormType::Two) - A.norm(NormType::Two)) /
                  A.norm(NormType::Two),
              1e-10);

        Matrix<double> S = spd(30);
        auto dS = mgpu::upload(S);
        double tr = 0;
        for (long i = 0; i < 30; ++i) tr += S(i, i);
        check("trace", std::fabs(dS.trace() - tr) / tr, 1e-12);
        check("cond vs basic/", std::fabs(dS.cond() - S.cond()) / S.cond(), 1e-9);
        checkTrue("rank of a full-rank matrix", dS.rank() == 30);

        // A deliberately rank-deficient matrix: two identical columns.
        Matrix<double> R = randMat(20, 6);
        for (long i = 0; i < 20; ++i) R(i, 4) = R(i, 1);
        checkTrue("rank detects a repeated column", mgpu::upload(R).rank() == 5);

        // pinv: A * pinv(A) * A == A is the defining property and holds even
        // when A is not square or not full rank.
        Matrix<double> P = randMat(30, 12);
        Matrix<double> Pp = mgpu::upload(P).pinv().cpu();
        check("pinv: A * pinv(A) * A == A", relerr(P * Pp * P, P), 1e-9);
        check("pinv vs basic/", relerr(Pp, P.pinv()), 1e-8);
    }

    // ── Rearrangement ──────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(5, 4);
        auto dA = mgpu::upload(A);

        Matrix<double> r = dA.reshaped(4, 5).cpu();
        double e = 0;
        for (long k = 0; k < 20; ++k)
            e = std::max(e, std::fabs(r(k / 5, k % 5) - A(k / 4, k % 4)));
        check("reshape", e, 0.0);

        Matrix<double> rep = dA.repmat(2, 3).cpu();
        e = 0;
        for (long i = 0; i < 10; ++i)
            for (long j = 0; j < 12; ++j) e = std::max(e, std::fabs(rep(i, j) - A(i % 5, j % 4)));
        check("repmat", e, 0.0);

        check("fliplr vs basic/", relerr(dA.fliplr().cpu(), A.fliplr()), 0.0);
        check("flipud vs basic/", relerr(dA.flipud().cpu(), A.flipud()), 0.0);
        check("rot90 vs basic/", relerr(dA.rot90().cpu(), A.rot90()), 0.0);
        check("circshift(rows) vs basic/", relerr(dA.circshift(2, 0).cpu(), A.circshift(2, 0)), 0.0);
        check("circshift(cols) vs basic/", relerr(dA.circshift(-1, 1).cpu(), A.circshift(-1, 1)),
              0.0);

        Matrix<double> B = randMat(3, 2);
        check("kron vs basic/", relerr(dA.kron(mgpu::upload(B)).cpu(), kron(A, B)), 1e-12);
    }

    // ── Builders ───────────────────────────────────────────────────────
    {
        check("linspace vs basic/", relerr(mgpu::linspace(0.0, 10.0, 200).cpu(),
                                            linspace(0.0, 10.0, 200)), 1e-13);
        check("linspace with one point", relerr(mgpu::linspace(3.0, 9.0, 1).cpu(),
                                                 linspace(3.0, 9.0, 1)), 0.0);
        check("logspace vs basic/", relerr(mgpu::logspace(-2.0, 3.0, 60).cpu(),
                                            logspace(-2.0, 3.0, 60)), 1e-12);
        Matrix<double> rg = mgpu::range(0.0, 5.0, 0.5).cpu();
        checkTrue("range has the right count", rg.cols() == 10);
        double e = 0;
        for (long j = 0; j < rg.cols(); ++j) e = std::max(e, std::fabs(rg(0, j) - 0.5 * double(j)));
        check("range values", e, 1e-13);
    }

    // ── Blocks: setBlock, stacking, resize ─────────────────────────────
    {
        Matrix<double> A = randMat(6, 5), S = randMat(2, 3);
        auto dA = mgpu::upload(A);
        dA.setBlock(3, 1, mgpu::upload(S));
        Matrix<double> got = dA.cpu();
        double e = 0;
        for (long i = 0; i < 6; ++i)
            for (long j = 0; j < 5; ++j) {
                const bool in = (i >= 3 && i < 5 && j >= 1 && j < 4);
                e = std::max(e, std::fabs(got(i, j) - (in ? S(i - 3, j - 1) : A(i, j))));
            }
        check("setBlock", e, 0.0);

        Matrix<double> L = randMat(4, 3), R = randMat(4, 2);
        Matrix<double> h = mgpu::upload(L).hstack(mgpu::upload(R)).cpu();
        e = 0;
        for (long i = 0; i < 4; ++i)
            for (long j = 0; j < 5; ++j)
                e = std::max(e, std::fabs(h(i, j) - (j < 3 ? L(i, j) : R(i, j - 3))));
        check("hstack", e, 0.0);

        Matrix<double> U = randMat(3, 4), D = randMat(2, 4);
        Matrix<double> v = mgpu::upload(U).vstack(mgpu::upload(D)).cpu();
        e = 0;
        for (long i = 0; i < 5; ++i)
            for (long j = 0; j < 4; ++j)
                e = std::max(e, std::fabs(v(i, j) - (i < 3 ? U(i, j) : D(i - 3, j))));
        check("vstack", e, 0.0);

        Matrix<double> pad = mgpu::upload(L).resized(6, 5).cpu();
        e = 0;
        for (long i = 0; i < 6; ++i)
            for (long j = 0; j < 5; ++j)
                e = std::max(e, std::fabs(pad(i, j) - ((i < 4 && j < 3) ? L(i, j) : 0.0)));
        check("resized (zero pad)", e, 0.0);
    }

    // ── FFT ────────────────────────────────────────────────────────────
    //
    // basic/signal.hpp's fft is the reference, so these check the GPU against
    // an implementation already validated against NumPy.
    {
        auto cerr_ = [](const Matrix<std::complex<double>>& X,
                        const Matrix<std::complex<double>>& Y) {
            if (X.rows() != Y.rows() || X.cols() != Y.cols()) return 1e9;
            double num = 0, den = 0;
            for (long i = 0; i < X.rows(); ++i)
                for (long j = 0; j < X.cols(); ++j) {
                    num += std::norm(X(i, j) - Y(i, j));
                    den += std::norm(Y(i, j));
                }
            return std::sqrt(num) / (std::sqrt(den) + 1e-300);
        };

        // Row vector: transforms along itself, both on the CPU and here.
        Matrix<double> x = randMat(1, 256);
        check("fft: row vector vs basic/", cerr_(mgpu::fft(mgpu::upload(x)).cpu(), fft(x)), 1e-12);

        // A non-power-of-two length, where a bad plan or padding shows up.
        Matrix<double> x2 = randMat(1, 210);
        check("fft: length 210 vs basic/", cerr_(mgpu::fft(mgpu::upload(x2)).cpu(), fft(x2)),
              1e-12);

        // Column vector and matrix default to running DOWN COLUMNS.
        Matrix<double> col = randMat(128, 1);
        check("fft: column vector vs basic/", cerr_(mgpu::fft(mgpu::upload(col)).cpu(), fft(col)),
              1e-12);
        Matrix<double> M = randMat(64, 32);
        check("fft: matrix, default axis (columns)",
              cerr_(mgpu::fft(mgpu::upload(M)).cpu(), fft(M)), 1e-12);
        check("fft: matrix, along rows",
              cerr_(mgpu::fft(mgpu::upload(M), ROW).cpu(), fft(M, -1, ROW)), 1e-12);
        check("fft: matrix, down columns",
              cerr_(mgpu::fft(mgpu::upload(M), COL).cpu(), fft(M, -1, COL)), 1e-12);

        // Round trip, which pins the 1/n the inverse has to apply.
        check("ifft(fft(x)) == x", relerr(mgpu::ifft(mgpu::fft(mgpu::upload(x), ROW), ROW).cpu(), x),
              1e-12);
        check("ifft2(fft2(A)) == A", relerr(mgpu::ifft2(mgpu::fft2(mgpu::upload(M))).cpu(), M),
              1e-12);

        // Magnitude, and Parseval as an independent check on the scaling.
        Matrix<double> mag = mgpu::fft(mgpu::upload(x)).abs().cpu();
        Matrix<std::complex<double>> ref = fft(x);
        double e = 0;
        for (long j = 0; j < x.cols(); ++j)
            e = std::max(e, std::fabs(mag(0, j) - std::abs(ref(0, j))) / (std::abs(ref(0, j)) + 1e-300));
        check("Spectrum::abs", e, 1e-11);

        double energy = 0, spec = 0;
        for (long j = 0; j < x.cols(); ++j) energy += x(0, j) * x(0, j);
        Matrix<double> pw = mgpu::fft(mgpu::upload(x)).power().cpu();
        for (long j = 0; j < x.cols(); ++j) spec += pw(0, j);
        check("Parseval: sum|X|^2 == n*sum|x|^2",
              std::fabs(spec - double(x.cols()) * energy) / (double(x.cols()) * energy), 1e-12);

        check("fftshift vs basic/", relerr(mgpu::fftshift(mgpu::upload(x)).cpu(), fftshift(x)), 0.0);
        Matrix<double> odd = randMat(1, 51);
        check("fftshift, odd length", relerr(mgpu::fftshift(mgpu::upload(odd)).cpu(), fftshift(odd)),
              0.0);
        check("ifftshift inverts fftshift",
              relerr(mgpu::ifftshift(mgpu::fftshift(mgpu::upload(odd))).cpu(), odd), 0.0);

        Matrix<double> a = randMat(1, 100), b = randMat(1, 37);
        check("conv vs basic/", relerr(mgpu::conv(mgpu::upload(a), mgpu::upload(b)).cpu(), conv(a, b)),
              1e-11);

        // Complex in, complex out -- the path that matches basic/'s fft()
        // signature exactly, and skips the interleave/split the real one needs.
        {
            using Cd = std::complex<double>;
            Matrix<Cd> z(1, 256);
            for (long j = 0; j < 256; ++j) z(0, j) = Cd(x(0, j), x(0, (j + 7) % 256));
            Matrix<Cd> Z = mgpu::fft(mgpu::upload(z), ROW).cpu();
            check("fft: complex in/out vs basic/", relerr(Z, fft(z, -1, ROW)), 1e-11);
            check("ifft(fft(z)) == z, complex",
                  relerr(mgpu::ifft(mgpu::fft(mgpu::upload(z), ROW), ROW).cpu(), z), 1e-12);

            Matrix<Cd> Z2 = mgpu::fft2(mgpu::upload(z)).cpu();
            check("ifft2(fft2(z)) == z, complex",
                  relerr(mgpu::ifft2(mgpu::fft2(mgpu::upload(z))).cpu(), z), 1e-12);
            (void)Z2;

            // A Spectrum and a complex matrix must describe the same
            // spectrum; toComplex() is the bridge between the two forms.
            Matrix<Cd> xc(1, 256);
            for (long j = 0; j < 256; ++j) xc(0, j) = Cd(x(0, j), 0.0);
            check("Spectrum::toComplex matches the complex path",
                  relerr(mgpu::fft(mgpu::upload(x), ROW).toComplex().cpu(),
                         mgpu::fft(mgpu::upload(xc), ROW).cpu()),
                  1e-13);
        }

        // float path
        Matrix<float> xf(1, 128);
        xf.set_Ran_values(-1.0, 1.0);
        Matrix<float> back = mgpu::ifft(mgpu::fft(mgpu::upload(xf), ROW), ROW).cpu();
        double fe = 0;
        for (long j = 0; j < 128; ++j) fe = std::max(fe, (double)std::fabs(back(0, j) - xf(0, j)));
        check("float fft round trip", fe, 1e-4);
    }

    // ── Reductions ─────────────────────────────────────────────────────
    {
        Matrix<double> A = randMat(200, 173, 0.5, 2.0);
        auto dA = mgpu::upload(A);
        const double cpuSum = mcpu::sum(A);
        check("sum", std::fabs(dA.sum() - cpuSum) / std::fabs(cpuSum), 1e-12);
        check("mean", std::fabs(dA.mean() - cpuSum / double(A.rows() * A.cols())) /
                          std::fabs(cpuSum / double(A.rows() * A.cols())), 1e-12);

        double mx = A(0, 0), mn = A(0, 0);
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) {
                if (A(i, j) > mx) mx = A(i, j);
                if (A(i, j) < mn) mn = A(i, j);
            }
        check("max", std::fabs(dA.max() - mx), 0.0);
        check("min", std::fabs(dA.min() - mn), 0.0);

        double fro = 0;
        for (long i = 0; i < A.rows(); ++i)
            for (long j = 0; j < A.cols(); ++j) fro += A(i, j) * A(i, j);
        check("norm (Frobenius)", std::fabs(dA.norm() - std::sqrt(fro)) / std::sqrt(fro), 1e-12);

        check("sum(ROW)", relerr(dA.sum(true).cpu(), mcpu::sum(A, true)), 1e-12);
        check("sum(COL)", relerr(dA.sum(false).cpu(), mcpu::sum(A, false)), 1e-12);
        checkTrue("sum(ROW) shape", dA.sum(true).rows() == A.rows() && dA.sum(true).cols() == 1);
        checkTrue("sum(COL) shape", dA.sum(false).rows() == 1 && dA.sum(false).cols() == A.cols());
    }

    // ── Blocks, triangles, diagonal ────────────────────────────────────
    {
        Matrix<double> A = randMat(30, 25);
        auto dA = mgpu::upload(A);

        Matrix<double> blk = dA.block(7, 3, 11, 9).cpu();
        double e = 0;
        for (long i = 0; i < 11; ++i)
            for (long j = 0; j < 9; ++j) e = std::max(e, std::fabs(blk(i, j) - A(7 + i, 3 + j)));
        check("block(7,3,11,9)", e, 0.0);

        Matrix<double> d = dA.diag().cpu();
        e = 0;
        for (long i = 0; i < 25; ++i) e = std::max(e, std::fabs(d(i, 0) - A(i, i)));
        check("diag", e, 0.0);

        Matrix<double> up = dA.triu().cpu();
        e = 0;
        for (long i = 0; i < 30; ++i)
            for (long j = 0; j < 25; ++j)
                e = std::max(e, std::fabs(up(i, j) - (j >= i ? A(i, j) : 0.0)));
        check("triu", e, 0.0);

        Matrix<double> lo = dA.tril(true).cpu();
        e = 0;
        for (long i = 0; i < 30; ++i)
            for (long j = 0; j < 25; ++j) {
                const double want = (j < i) ? A(i, j) : (i == j ? 1.0 : 0.0);
                e = std::max(e, std::fabs(lo(i, j) - want));
            }
        check("tril(unit diagonal)", e, 0.0);

        Matrix<double> I = mgpu::Matrix<double>::eye(12).cpu();
        e = 0;
        for (long i = 0; i < 12; ++i)
            for (long j = 0; j < 12; ++j) e = std::max(e, std::fabs(I(i, j) - (i == j ? 1.0 : 0.0)));
        check("eye", e, 0.0);
    }

    // ── LU ─────────────────────────────────────────────────────────────
    {
        const long n = 120;
        Matrix<double> A = randMat(n, n);
        auto [dL, dU, piv] = mgpu::upload(A).lu();
        Matrix<double> L = dL.cpu(), U = dU.cpu();

        // cuSOLVER factorises P*A = L*U, so A has to be permuted the same way
        // before the product can match. ipiv is LAPACK's sequential-swap form:
        // at step i, row i was exchanged with row ipiv[i]-1.
        Matrix<double> PA = A;
        for (long i = 0; i < n; ++i) {
            const long p = piv[(std::size_t)i] - 1;
            if (p != i)
                for (long j = 0; j < n; ++j) std::swap(PA(i, j), PA(p, j));
        }
        check("lu: L*U == P*A", relerr(L * U, PA), 1e-12);

        bool lowerOk = true, upperOk = true;
        for (long i = 0; i < n; ++i)
            for (long j = 0; j < n; ++j) {
                if (j > i && L(i, j) != 0.0) lowerOk = false;
                if (j < i && U(i, j) != 0.0) upperOk = false;
            }
        checkTrue("lu: L is lower triangular", lowerOk);
        checkTrue("lu: U is upper triangular", upperOk);
        bool unit = true;
        for (long i = 0; i < n; ++i)
            if (L(i, i) != 1.0) unit = false;
        checkTrue("lu: L has unit diagonal", unit);
    }

    // ── Cholesky ───────────────────────────────────────────────────────
    {
        const long n = 100;
        Matrix<double> A = spd(n);
        Matrix<double> L = mgpu::upload(A).cholesky().cpu();
        check("cholesky: L*L^T == A", relerr(L * L.T(), A), 1e-12);
        bool lowerOk = true;
        for (long i = 0; i < n; ++i)
            for (long j = i + 1; j < n; ++j)
                if (L(i, j) != 0.0) lowerOk = false;
        checkTrue("cholesky: L is lower triangular", lowerOk);
        check("cholesky matches basic/", relerr(L, A.cholesky()), 1e-11);
    }

    // ── QR ─────────────────────────────────────────────────────────────
    {
        // Tall, square and wide: the reduced shapes differ in each case and
        // the m<n path in particular is where a shape bug would live.
        struct { long m, n; } sh[] = {{150, 80}, {100, 100}, {60, 130}};
        for (auto s : sh) {
            Matrix<double> A = randMat(s.m, s.n);
            auto [dQ, dR] = mgpu::upload(A).qr();
            Matrix<double> Q = dQ.cpu(), R = dR.cpu();
            const long k = std::min(s.m, s.n);

            char nm[64];
            std::snprintf(nm, sizeof(nm), "qr %ldx%ld: Q*R == A", s.m, s.n);
            check(nm, relerr(Q * R, A), 1e-12);

            std::snprintf(nm, sizeof(nm), "qr %ldx%ld: Q^T Q == I", s.m, s.n);
            Matrix<double> QtQ = Q.T() * Q;
            double e = 0;
            for (long i = 0; i < k; ++i)
                for (long j = 0; j < k; ++j)
                    e = std::max(e, std::fabs(QtQ(i, j) - (i == j ? 1.0 : 0.0)));
            check(nm, e, 1e-12);

            std::snprintf(nm, sizeof(nm), "qr %ldx%ld: R upper triangular", s.m, s.n);
            bool ok = true;
            for (long i = 0; i < R.rows(); ++i)
                for (long j = 0; j < i && j < R.cols(); ++j)
                    if (R(i, j) != 0.0) ok = false;
            checkTrue(nm, ok);
        }
    }

    // ── SVD ────────────────────────────────────────────────────────────
    {
        struct { long m, n; } sh[] = {{120, 70}, {90, 90}, {50, 110}};
        for (auto s : sh) {
            Matrix<double> A = randMat(s.m, s.n);
            auto [dU, dS, dV] = mgpu::upload(A).svd();
            Matrix<double> U = dU.cpu(), S = dS.cpu(), V = dV.cpu();

            char nm[64];
            std::snprintf(nm, sizeof(nm), "svd %ldx%ld: U*S*V^T == A", s.m, s.n);
            check(nm, relerr(U * S * V.T(), A), 1e-11);

            // Singular values are unique, so unlike the factors they can be
            // compared straight against the CPU library.
            std::snprintf(nm, sizeof(nm), "svd %ldx%ld: values match basic/", s.m, s.n);
            auto [cU, cS, cV] = A.svd();
            const long k = std::min(s.m, s.n);
            double e = 0;
            for (long i = 0; i < k; ++i)
                e = std::max(e, std::fabs(S(i, i) - cS(i, i)) / (std::fabs(cS(i, i)) + 1e-300));
            check(nm, e, 1e-10);

            std::snprintf(nm, sizeof(nm), "svdvals %ldx%ld", s.m, s.n);
            Matrix<double> sv = mgpu::upload(A).svdvals().cpu();
            e = 0;
            for (long i = 0; i < k; ++i)
                e = std::max(e, std::fabs(sv(i, 0) - cS(i, i)) / (std::fabs(cS(i, i)) + 1e-300));
            check(nm, e, 1e-10);
        }
    }

    // ── Symmetric eigenproblem ─────────────────────────────────────────
    {
        const long n = 90;
        Matrix<double> A = spd(n);
        auto [dW, dV] = mgpu::upload(A).eigSym();
        Matrix<double> w = dW.cpu(), V = dV.cpu();

        // A*V == V*diag(w), the property that does not care about sign or order
        Matrix<double> D(n, n);
        for (long i = 0; i < n; ++i) D(i, i) = w(i, 0);
        check("eigSym: A*V == V*diag(w)", relerr(A * V, V * D), 1e-11);

        Matrix<double> VtV = V.T() * V;
        double e = 0;
        for (long i = 0; i < n; ++i)
            for (long j = 0; j < n; ++j) e = std::max(e, std::fabs(VtV(i, j) - (i == j ? 1.0 : 0.0)));
        check("eigSym: V orthonormal", e, 1e-11);

        bool ascending = true;
        for (long i = 1; i < n; ++i)
            if (w(i, 0) < w(i - 1, 0)) ascending = false;
        checkTrue("eigSym: values ascending", ascending);
    }

    // ── Solve, inverse, determinant ────────────────────────────────────
    {
        const long n = 110;
        Matrix<double> A = spd(n);            // well conditioned on purpose
        Matrix<double> B = randMat(n, 7);
        Matrix<double> X = mgpu::upload(A).solve(mgpu::upload(B)).cpu();
        check("solve: A*X == B", relerr(A * X, B), 1e-11);

        Matrix<double> Ai = mgpu::upload(A).inv().cpu();
        double e = 0;
        Matrix<double> P = A * Ai;
        for (long i = 0; i < n; ++i)
            for (long j = 0; j < n; ++j) e = std::max(e, std::fabs(P(i, j) - (i == j ? 1.0 : 0.0)));
        check("inv: A*inv(A) == I", e, 1e-10);

        // Determinants of a 110x110 overflow any sane scale, so compare a
        // small one where the CPU value is trustworthy.
        Matrix<double> Small = randMat(9, 9);
        const double dg = mgpu::upload(Small).det(), dc = Small.det();
        check("det matches basic/", std::fabs(dg - dc) / (std::fabs(dc) + 1e-300), 1e-10);
    }

    // ── Random ─────────────────────────────────────────────────────────
    {
        mgpu::Matrix<double> R(400, 400);
        R.set_Ran_values(-2.0, 5.0, 1234);
        Matrix<double> h = R.cpu();
        double lo = h(0, 0), hi = h(0, 0), mean = 0;
        for (long i = 0; i < h.rows(); ++i)
            for (long j = 0; j < h.cols(); ++j) {
                lo = std::min(lo, h(i, j));
                hi = std::max(hi, h(i, j));
                mean += h(i, j);
            }
        mean /= double(h.rows() * h.cols());
        checkTrue("set_Ran_values within bounds", lo >= -2.0 && hi < 5.0);
        check("set_Ran_values mean ~ 1.5", std::fabs(mean - 1.5) / 1.5, 0.02);

        mgpu::Matrix<double> R2(400, 400);
        R2.set_Ran_values(-2.0, 5.0, 1234);
        check("set_Ran_values is seed-reproducible", relerr(R2.cpu(), h), 0.0);

        mgpu::Matrix<double> N(400, 400);
        N.randn(0.0, 1.0, 99);
        Matrix<double> nh = N.cpu();
        double m2 = 0, v2 = 0;
        const long cnt = nh.rows() * nh.cols();
        for (long i = 0; i < nh.rows(); ++i)
            for (long j = 0; j < nh.cols(); ++j) m2 += nh(i, j);
        m2 /= double(cnt);
        for (long i = 0; i < nh.rows(); ++i)
            for (long j = 0; j < nh.cols(); ++j) v2 += (nh(i, j) - m2) * (nh(i, j) - m2);
        v2 /= double(cnt - 1);
        checkTrue("randn mean ~ 0", std::fabs(m2) < 0.02);
        checkTrue("randn variance ~ 1", std::fabs(v2 - 1.0) < 0.03);

        // Odd count: cuRAND generates normals in pairs and rejects an odd n,
        // so this exercises the round-up-and-trim path in the backend.
        mgpu::Matrix<double> Odd(1, 777);
        Odd.randn(0.0, 1.0, 7);
        checkTrue("randn with odd element count", std::isfinite(Odd.sum()));
    }

    // ── float ──────────────────────────────────────────────────────────
    {
        Matrix<float> A(128, 96), B(96, 64);
        A.set_Ran_values(-1.0, 1.0);
        B.set_Ran_values(-1.0, 1.0);
        Matrix<float> C = (mgpu::upload(A) * mgpu::upload(B)).cpu();
        Matrix<float> Cc = A * B;
        double e = 0, d = 0;
        for (long i = 0; i < C.rows(); ++i)
            for (long j = 0; j < C.cols(); ++j) {
                e += double(C(i, j) - Cc(i, j)) * double(C(i, j) - Cc(i, j));
                d += double(Cc(i, j)) * double(Cc(i, j));
            }
        check("float gemm", std::sqrt(e) / std::sqrt(d), 1e-5);

        Matrix<float> F(64, 64);
        F.set_Ran_values(-1.0, 1.0);
        Matrix<float> Sf = F.T() * F;
        for (long i = 0; i < 64; ++i) Sf(i, i) += 64.0f;
        Matrix<float> Lf = mgpu::upload(Sf).cholesky().cpu();
        Matrix<float> back = Lf * Lf.T();
        e = d = 0;
        for (long i = 0; i < back.rows(); ++i)
            for (long j = 0; j < back.cols(); ++j) {
                e += double(back(i, j) - Sf(i, j)) * double(back(i, j) - Sf(i, j));
                d += double(Sf(i, j)) * double(Sf(i, j));
            }
        check("float cholesky", std::sqrt(e) / std::sqrt(d), 1e-4);
    }

    // ── Error handling ─────────────────────────────────────────────────
    {
        bool threw = false;
        try {
            mgpu::Matrix<double> a(3, 4), b(5, 6);
            (void)(a + b);
        } catch (const mgpu::Error&) { threw = true; }
        checkTrue("mismatched shapes throw mgpu::Error", threw);

        threw = false;
        try {
            mgpu::Matrix<double> a(3, 4), b(5, 6);
            (void)(a * b);
        } catch (const mgpu::Error&) { threw = true; }
        checkTrue("bad inner dimension throws", threw);

        threw = false;
        try {
            Matrix<double> N(4, 4);          // all zeros: not positive definite
            (void)mgpu::upload(N).cholesky();
        } catch (const mgpu::Error&) { threw = true; }
        checkTrue("non-SPD cholesky throws", threw);
    }

    std::printf("\n%d / %d checks passed\n", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
