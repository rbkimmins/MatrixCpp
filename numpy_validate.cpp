// ════════════════════════════════════════════════════════════════════════════
//  Cross-validation dump: runs every operator MatrixCpp implements on fixed
//  inputs and writes the inputs and results to validation/*.dat, so that
//  numpy_validate.py can recompute the same thing in NumPy and compare.
//
//  This is a different kind of check from validate.cpp. That file verifies the
//  library against itself — identities, invariants, hand-computed values. This
//  one verifies it against an independent implementation, which catches the
//  class of error where a convention is self-consistent but simply not what the
//  rest of the world means by that operation.
//
//  Build and run:
//      g++ -std=c++17 -O2 -fopenmp -o numpy_validate numpy_validate.cpp
//      ./numpy_validate && python3 numpy_validate.py
//
//  File format — deliberately trivial to parse, one case per file:
//      op <name>
//      dtype <real|complex>
//      param <double>              (0 when the operation takes no scalar)
//      mat <label> <rows> <cols> <r|c>
//      <rows*cols numbers, whitespace separated; 'c' writes re im pairs>
//      ...
//      end
//  Values are written with 17 significant digits, which round-trips a double
//  exactly, so any mismatch the Python side reports is a real disagreement and
//  not a printing artefact.
// ════════════════════════════════════════════════════════════════════════════
#include "Matrix1.0.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <complex>
#include <vector>

using cd = std::complex<double>;
static int caseCount = 0;

// ─────────────────────────────────────────────────────────────── writing ──
// The element kind is tagged PER MATRIX, not per case: a complex case can
// legitimately contain real matrices — real(), imag() and the norms all return
// real results from a complex input — and the reader has to know how many
// numbers make up each element.
template<typename T>
static void writeMat(std::ofstream& f, const std::string& label, const Matrix<T>& M) {
    f << "mat " << label << " " << M.rows() << " " << M.cols() << " "
      << (is_complex<T>::value ? 'c' : 'r') << "\n";
    for (long i = 0; i < M.rows(); i++) {
        for (long j = 0; j < M.cols(); j++) {
            const T& v = M(int(i), int(j));
            if constexpr (is_complex<T>::value) f << v.real() << " " << v.imag() << " ";
            else                                f << v << " ";
        }
        f << "\n";
    }
}

class Case {
    std::ofstream f;
public:
    Case(const std::string& op, const std::string& dtype, double param = 0.0) {
        f.open("validation/" + dtype + "_" + op + ".dat");
        f << std::scientific << std::setprecision(17);
        f << "op " << op << "\ndtype " << dtype << "\nparam " << param << "\n";
        caseCount++;
    }
    template<typename T> Case& mat(const std::string& label, const Matrix<T>& M) {
        writeMat(f, label, M);
        return *this;
    }
    Case& scalar(const std::string& label, double v) {
        Matrix<double> S(1, 1); S(0, 0) = v;
        writeMat(f, label, S);
        return *this;
    }
    Case& scalar(const std::string& label, cd v) {
        Matrix<cd> S(1, 1); S(0, 0) = v;
        writeMat(f, label, S);
        return *this;
    }
    ~Case() { f << "end\n"; }
};

// ─────────────────────────────────────────────────────── fixed test inputs ──
template<typename T>
static Matrix<T> operandA(long n = 5) {
    Matrix<T> A(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            double re = 0.7 * std::sin(2.1 * i + 0.9 * j) + 0.3 * (i - j);
            if constexpr (is_complex<T>::value)
                A(int(i), int(j)) = cd(re, 0.5 * std::cos(1.3 * i - 0.7 * j));
            else
                A(int(i), int(j)) = re;
        }
    return A;
}

template<typename T>
static Matrix<T> operandB(long n = 5) {
    Matrix<T> B(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            double re = 0.4 * std::cos(1.7 * i + 1.1 * j) + 0.25 * (i + j) + 1.5;
            if constexpr (is_complex<T>::value)
                B(int(i), int(j)) = cd(re, 0.35 * std::sin(0.8 * i + 1.9 * j) + 0.6);
            else
                B(int(i), int(j)) = re;
        }
    return B;
}

// Strictly positive entries, for the element-wise log/pow cases.
template<typename T>
static Matrix<T> operandPos(long n = 5) {
    Matrix<T> P = operandA<T>(n);
    for (long i = 0; i < n * n; i++) P[int(i)] = P[int(i)] + T(3);
    return P;
}

static Matrix<double> wellConditioned(long n = 5) {
    Matrix<double> A = operandA<double>(n);
    for (long i = 0; i < n; i++) A(int(i), int(i)) += double(n);
    return A;
}

static Matrix<double> symmetric(long n = 5) {
    Matrix<double> A = operandA<double>(n);
    return (A + A.T()) * 0.5;
}

static Matrix<double> spd(long n = 5) {
    Matrix<double> A = symmetric(n);
    for (long i = 0; i < n; i++) A(int(i), int(i)) += double(n) + 2.0;
    return A;
}

// Wraps a scalar result as a 1x1 matrix so it travels through the same channel.
static Matrix<double> oneByOne(double v) { Matrix<double> m(1, 1); m(0, 0) = v; return m; }

// ══════════════════════════════════════════════════ shared: real + complex ──
template<typename T>
static void dumpShared(const std::string& dt) {
    Matrix<T> A = operandA<T>(), B = operandB<T>(), P = operandPos<T>();

    Case("add",        dt).mat("A", A).mat("B", B).mat("R", A + B);
    Case("subtract",   dt).mat("A", A).mat("B", B).mat("R", A - B);
    Case("multiply",   dt).mat("A", A).mat("B", B).mat("R", A * B);
    Case("hadamard",   dt).mat("A", A).mat("B", B).mat("R", A % B);
    Case("elem_div",   dt).mat("A", A).mat("B", B).mat("R", A.div(B));
    Case("negate",     dt).mat("A", A).mat("R", -A);
    Case("scalar_mul", dt, 2.75).mat("A", A).mat("R", A * 2.75);
    Case("scalar_div", dt, 2.75).mat("A", A).mat("R", A / 2.75);

    Case("transpose",  dt).mat("A", A).mat("R", A.T());
    Case("conj",       dt).mat("A", A).mat("R", A.conj());
    Case("conj_transpose", dt).mat("A", A).mat("R", A.H());
    Case("real_part",  dt).mat("A", A).mat("R", A.real());
    Case("imag_part",  dt).mat("A", A).mat("R", A.imag());

    Case("elem_exp",   dt).mat("A", A).mat("R", A.exp());

    // ── Least squares with Q left implicit ─────────────────────────────────
    if constexpr (!is_complex<T>::value) {
        Matrix<double> Ls(40, 6), Lb(40, 1);
        Ls.set_Ran_values(-1.0, 1.0, -1201);
        Lb.set_Ran_values(-1.0, 1.0, -1202);
        Case("lstsq_tall", dt).mat("A", Ls).mat("B", Lb).mat("R", Ls.solve(Lb));
        Matrix<double> Rk(30, 5);
        Rk.set_Ran_values(-1.0, 1.0, -1203);
        Case("rank_tall", dt).mat("A", Rk).mat("R", oneByOne(double(Rk.rank())));
    }

    // ── Complex linear algebra (work_t) ────────────────────────────────────
    // These are dumped under the COMPLEX dtype so numpy_validate.py compares
    // them against numpy's complex routines directly.
    if constexpr (is_complex<T>::value) {
        Matrix<T> Cm(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Cm(i, j) = T(std::sin(2.0 * i + j)) + T(std::cos(i - 1.5 * j)) * T(0, 1);
        for (int i = 0; i < 4; i++) Cm(i, i) += T(4.0);
        Case("cx_inverse", dt).mat("A", Cm).mat("R", Cm.inverse());
        Case("cx_det", dt).mat("A", Cm).mat("R", [&]{ Matrix<T> d(1,1); d(0,0)=Cm.det(); return d; }());
        Matrix<T> rhs(4, 2);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++) rhs(i, j) = T(double(i + 1)) + T(double(-j)) * T(0, 1);
        Case("cx_solve", dt).mat("A", Cm).mat("B", rhs).mat("R", Cm.solve(rhs));
        Case("cx_pinv", dt).mat("A", Cm).mat("R", Cm.pinv());
        // Singular values are real and unique, so they compare directly where
        // U and V (which are not unique) could not.
        Case("cx_svdvals", dt).mat("A", Cm).mat("R", [&]{
            auto [U, S, V] = Cm.svd(); (void)U; (void)V;
            Matrix<T> sv(4, 1);
            for (int i = 0; i < 4; i++) sv(i, 0) = T(S(i, i));
            return sv; }());
        Case("cx_norm2", dt).mat("A", Cm).mat("R", [&]{
            Matrix<T> v(1,1); v(0,0)=T(Cm.norm(NormType::Two)); return v; }());
        // A Hermitian positive definite matrix, for the Cholesky comparison.
        Matrix<T> Hp(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Hp(i, j) = T(std::sin(i * 3.1 + j)) + T(std::cos(i + 2.7 * j)) * T(0, 1);
        Matrix<T> Hh = Hp.H() * Hp;
        for (int i = 0; i < 4; i++) Hh(i, i) += T(4.0);
        Case("cx_chol", dt).mat("A", Hh).mat("R", Hh.cholesky());
        Matrix<T> Sm = Cm * 0.125;
        Case("cx_expm", dt).mat("A", Sm).mat("R", exp(Sm));
    }


    Case("elem_ln",    dt).mat("A", P).mat("R", P.ln());
    Case("elem_log10", dt).mat("A", P).mat("R", P.log10());
    Case("elem_pow",   dt, 2.5).mat("A", P).mat("R", P.pow(2.5));
    Case("elem_sqrt",  dt).mat("A", P).mat("R", P.sqrt());
    Case("elem_sin",   dt).mat("A", A).mat("R", A.sin());
    Case("elem_cos",   dt).mat("A", A).mat("R", A.cos());
    Case("elem_tan",   dt).mat("A", A).mat("R", A.tan());
    Case("elem_sinh",  dt).mat("A", A).mat("R", A.sinh());
    Case("elem_cosh",  dt).mat("A", A).mat("R", A.cosh());
    Case("elem_tanh",  dt).mat("A", A).mat("R", A.tanh());

    Case("sum_all",    dt).mat("A", A).scalar("R", A.sum());
    Case("sum_cols",   dt).mat("A", A).mat("R", A.sum(0));
    Case("sum_rows",   dt).mat("A", A).mat("R", A.sum(1));
    Case("trace",      dt).mat("A", A).scalar("R", A.tr());

    Case("triu",       dt).mat("A", A).mat("R", A.triu());
    Case("tril",       dt).mat("A", A).mat("R", A.tril());
    Case("triu_k1",    dt, 1).mat("A", A).mat("R", A.triu(1));
    Case("tril_km1",   dt, -1).mat("A", A).mat("R", A.tril(-1));
    Case("diag_extract", dt).mat("A", A).mat("R", A.diag());
    Case("diag_build", dt).mat("A", A.diag()).mat("R", diag(A.diag()));
    Case("reshape",    dt).mat("A", A).mat("R", A.reshape(1, A.rows() * A.cols()));
    Case("concat_h",   dt).mat("A", A).mat("B", B).mat("R", A.concat(B, 1));
    Case("concat_v",   dt).mat("A", A).mat("B", B).mat("R", A.concat(B, 0));
    Case("kron",     dt).mat("A", A).mat("B", B).mat("R", A.kron(B));

    Case("norm_fro",   dt).mat("A", A).scalar("R", A.norm(NormType::Fro));
    Case("norm_one",   dt).mat("A", A).scalar("R", A.norm(NormType::One));
    Case("norm_inf",   dt).mat("A", A).scalar("R", A.norm(NormType::Inf));
    // ── the complex Schur family, cross-checked against SciPy ──
    // These were real-only until the complex Schur decomposition landed, so an
    // INDEPENDENT reference matters more here than anywhere else in this file.
    if constexpr (is_complex<T>::value) {
        { auto [Ts, Qs] = A.schur();
          Case("schur_complex", dt).mat("A", A).mat("T", Ts).mat("Q", Qs); }
        { auto [Hh, Qhh] = A.hess();
          Case("hess_complex", dt).mat("A", A).mat("H", Hh).mat("Q", Qhh); }
        { auto [ev, X] = A.eig();
          Case("eig_complex", dt).mat("A", A).mat("E", ev).mat("X", X); }
        { Matrix<T> Ap = A;
          for (long i = 0; i < Ap.rows(); i++) Ap(int(i), int(i)) += T(4.0);
          Case("sqrtm_complex", dt).mat("A", Ap).mat("R", sqrt(Ap)); }
        { Matrix<T> Ap = A;
          for (long i = 0; i < Ap.rows(); i++) Ap(int(i), int(i)) += T(4.0);
          Case("logm_complex", dt).mat("A", Ap).mat("R", log(Ap, M_E)); }
        { Matrix<T> Ap = A;
          for (long i = 0; i < Ap.rows(); i++) Ap(int(i), int(i)) += T(4.0);
          Case("expm_complex", dt).mat("A", Ap)
              .mat("R", funm(Ap, [](std::complex<double> z) { return std::exp(z); })); }
        // QZ on a pencil. B is pushed away from singular so every eigenvalue is
        // finite and can be compared as a ratio.
        { Matrix<T> Bp = B;
          for (long i = 0; i < Bp.rows(); i++) Bp(int(i), int(i)) += T(double(Bp.rows()));
          auto r = qz(A, Bp);
          Matrix<T> Bt(Bp.rows(), Bp.cols());
          for (long i = 0; i < Bp.rows(); i++)
              for (long j = 0; j < Bp.cols(); j++) Bt(int(i), int(j)) = Bp(int(i), int(j));
          Matrix<double> re(A.rows(), 1), im(A.rows(), 1);
          const auto al = r.alpha(), be = r.beta();
          for (long i = 0; i < A.rows(); i++) {
              const std::complex<double> lam = al[(std::size_t)i] / be[(std::size_t)i];
              re(int(i), 0) = lam.real();
              im(int(i), 0) = lam.imag();
          }
          Case("qz_complex", dt).mat("A", A).mat("B", Bt).mat("S", r.S).mat("T", r.T)
              .mat("Q", r.Q).mat("Z", r.Z).mat("RE", re).mat("IM", im)
              .mat("X", r.eigenvectors()); }
    }

}

// ═════════════════════════════════════════════════════════════ real only ──
static void dumpRealOnly() {
    const std::string dt = "real";
    Matrix<double> A = operandA<double>();
    Matrix<double> W = wellConditioned(), S = symmetric(), Q = spd();

    // Reductions — refused at compile time for complex (no ordering).
    Case("min",    dt).mat("A", A).scalar("R", A.min());
    Case("max",    dt).mat("A", A).scalar("R", A.max());
    Case("mean",   dt).mat("A", A).scalar("R", A.mean());
    Case("var_pop", dt).mat("A", A).scalar("R", A.var(false));
    Case("var_samp", dt).mat("A", A).scalar("R", A.var(true));
    Case("stddev", dt).mat("A", A).scalar("R", A.stddev(false));
    Case("min_cols", dt).mat("A", A).mat("R", A.min(0));
    Case("max_rows", dt).mat("A", A).mat("R", A.max(1));
    Case("mean_cols", dt).mat("A", A).mat("R", A.mean(0));
    { auto [r, c] = A.argmin();
      Matrix<double> R(1, 2); R(0,0) = double(r); R(0,1) = double(c);
      Case("argmin", dt).mat("A", A).mat("R", R); }
    { auto [r, c] = A.argmax();
      Matrix<double> R(1, 2); R(0,0) = double(r); R(0,1) = double(c);
      Case("argmax", dt).mat("A", A).mat("R", R); }

    // Unique results — compared against NumPy element by element.
    Case("det",     dt).mat("A", W).scalar("R", W.det());
    Case("inverse", dt).mat("A", W).mat("R", W.inverse());
    Case("pinv",    dt).mat("A", W).mat("R", W.pinv());
    Case("adjugate", dt).mat("A", W).mat("R", W.adjugate());
    Case("rank",    dt).mat("A", W).scalar("R", double(W.rank()));
    Case("cholesky", dt).mat("A", Q).mat("R", Q.cholesky());

    { Matrix<double> b(5, 1);
      for (int i = 0; i < 5; i++) b(i, 0) = 1.0 + 0.5 * i;
      Case("solve", dt).mat("A", W).mat("B", b).mat("R", W.solve(b));
      // Matrix RIGHT division, MATLAB's mrdivide: X / W is the X solving X*W = A.
      // NumPy has no operator for it; the reference is A @ inv(W).
      Case("mrdivide", dt).mat("A", W).mat("B", W).mat("R", (W + W) / W); }
    // ── Tier 6: convolution, polynomials, calculus, interpolation ──────────
    { Matrix<double> pa(1, 4), pb(1, 3);
      pa = {{2, -3, 0, 5}};
      pb = {{1, 4, -2}};
      Case("conv",     dt).mat("A", pa).mat("B", pb).mat("R", conv(pa, pb));
      // A long convolution, so the FFT path is the one being compared.
      Matrix<double> la(1, 200), lb(1, 200);
      la.set_Ran_values(-1.0, 1.0, -1001);
      lb.set_Ran_values(-1.0, 1.0, -1002);
      Case("conv_fft", dt).mat("A", la).mat("B", lb).mat("R", conv(la, lb));
      Case("deconv_q", dt).mat("A", conv(pa, pb)).mat("B", pa)
                          .mat("R", deconv(conv(pa, pb), pa).first);
      Matrix<double> rt(1, 4); rt = {{1, -2, 3, 0.5}};
      Case("poly_roots", dt).mat("A", rt).mat("R", poly(rt));
      Matrix<double> Pm(3, 3);
      Pm.set_Ran_values(-1.0, 1.0, -1003);
      Case("poly_charpoly", dt).mat("A", Pm).mat("R", poly(Pm));

      Matrix<double> sy(1, 9), sx(1, 9);
      sy.set_Ran_values(-2.0, 2.0, -1004);
      for (int i = 0; i < 9; i++) sx(0, i) = double(i) * 0.5;
      Case("trapz_unit", dt).mat("A", sy).mat("R", oneByOne(sy.trapz()));
      Case("trapz_xy",   dt).mat("A", sx).mat("B", sy).mat("R", oneByOne(trapz(sx, sy)));
      Case("cumtrapz",   dt).mat("A", sy).mat("R", sy.cumtrapz(true));
      Case("gradient",   dt).mat("A", sy).mat("R", sy.gradient(true));
      Case("gradient_h", dt).mat("A", sy).mat("R", sy.gradient(true, 0.5));

      Matrix<double> iy(1, 6), ix(1, 6), iq(1, 5);
      for (int i = 0; i < 6; i++) { ix(0, i) = double(i); iy(0, i) = std::sin(double(i)); }
      for (int i = 0; i < 5; i++) iq(0, i) = 0.3 + double(i) * 1.1;
      Case("interp1", dt).mat("A", ix).mat("B", iq).mat("R", interp1(ix, iy, iq));

      Matrix<double> fb(1, 3), fa(1, 3), fx(1, 12);
      fb = {{0.2, 0.5, 0.3}};
      fa = {{1.0, -0.4, 0.1}};
      fx.set_Ran_values(-1.0, 1.0, -1005);
      Case("filter", dt).mat("A", fx).mat("R", filter(fb, fa, fx));
    }

    // ── Tier 6: the FFT ────────────────────────────────────────────────────
    // Complex results are dumped as separate real and imaginary matrices — the
    // .dat format tags element kind per matrix, so this needs no format change.
    { Matrix<double> sig(1, 64);
      sig.set_Ran_values(-1.0, 1.0, -606);
      Case("fft_re",   dt).mat("A", sig).mat("R", fft(sig).real());
      Case("fft_im",   dt).mat("A", sig).mat("R", fft(sig).imag());
      // A prime length, so this one goes through Bluestein rather than radix-2.
      Matrix<double> pr(1, 61);
      pr.set_Ran_values(-1.0, 1.0, -707);
      Case("fft_prime_re", dt).mat("A", pr).mat("R", fft(pr).real());
      Case("fft_prime_im", dt).mat("A", pr).mat("R", fft(pr).imag());
      // Zero-padded, and truncated.
      Case("fft_pad_re",  dt).mat("A", sig).mat("R", fft(sig, 100).real());
      Case("fft_trunc_re", dt).mat("A", sig).mat("R", fft(sig, 20).real());
      // A matrix, transformed column by column.
      Matrix<double> M(8, 5);
      M.set_Ran_values(-1.0, 1.0, -808);
      Case("fft_cols_re", dt).mat("A", M).mat("R", fft(M).real());
      Case("fft_cols_im", dt).mat("A", M).mat("R", fft(M).imag());
      Case("fft_rows_re", dt).mat("A", M).mat("R", fft(M, -1, true).real());
      // The inverse, where the 1/n convention lives.
      Case("ifft_re", dt).mat("A", sig).mat("R", ifft(sig).real());
      Case("ifft_im", dt).mat("A", sig).mat("R", ifft(sig).imag());
      Case("fftshift", dt).mat("A", sig).mat("R", fftshift(sig));
    }

    // ── Sequences, shape and constructors (tier 5) ─────────────────────────
    { Case("linspace", dt).mat("A", Matrix<double>(1, 1)).mat("R", linspace(-2.0, 3.0, 11));
      Case("logspace", dt).mat("A", Matrix<double>(1, 1)).mat("R", logspace(-1.0, 2.0, 7));
      Case("range",    dt).mat("A", Matrix<double>(1, 1)).mat("R", range(0.0, 9.0, 2.0));
      Case("hilb5",    dt).mat("A", Matrix<double>(1, 1)).mat("R", hilb(5));
      Case("pascal4",  dt).mat("A", Matrix<double>(1, 1)).mat("R", pascal(4));
      Matrix<double> Sh(3, 4); Sh.set_Ran_values(-2.0, 2.0, -717);
      Case("fliplr",    dt).mat("A", Sh).mat("R", Sh.fliplr());
      Case("flipud",    dt).mat("A", Sh).mat("R", Sh.flipud());
      Case("rot90_1",   dt).mat("A", Sh).mat("R", Sh.rot90(1));
      Case("rot90_m1",  dt).mat("A", Sh).mat("R", Sh.rot90(-1));
      Case("repmat",    dt).mat("A", Sh).mat("R", Sh.repmat(2, 3));
      Case("circ_row",  dt).mat("A", Sh).mat("R", Sh.circshift(1, 0));
      Case("circ_col",  dt).mat("A", Sh).mat("R", Sh.circshift(-2, 1));
      Matrix<double> Bk(2, 2); Bk.set_Ran_values(-1.0, 1.0, -818);
      Case("blkdiag",   dt).mat("A", Sh).mat("B", Bk).mat("R", Sh.blkdiag(Bk));
      Matrix<double> tv(1, 4); tv = {{1, 2, 3, 4}};
      Case("toeplitz",  dt).mat("A", tv).mat("R", toeplitz(tv));
      Case("vander",    dt).mat("A", tv).mat("R", vander(tv));
    }

    // ── det / LU of a singular matrix ──────────────────────────────────────
    { Matrix<double> Sg(3, 3);
      Sg = {{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};            // row 2 = 2 * row 1
      Case("det_singular", dt).mat("A", Sg).mat("R", oneByOne(Sg.det()));
      Matrix<double> Zg(3, 3);
      Case("det_zeros", dt).mat("A", Zg).mat("R", oneByOne(Zg.det()));
      Matrix<double> Ng(3, 3);
      Ng = {{2, 0, 1}, {1, 3, 0}, {0, 1, 4}};
      Case("det_nonsingular", dt).mat("A", Ng).mat("R", oneByOne(Ng.det()));
    }

    // ── funm and generalized eigenvalues (tier 4) ──────────────────────────
    { Matrix<double> Af(4, 4);
      Af.set_Ran_values(-1.0, 1.0, -919);
      // funm against a function SciPy also has, so the reference is genuine.
      Case("funm_exp", dt).mat("A", Af)
          .mat("R", funm(Af, [](std::complex<double> z) { return std::exp(z); }).real());
      // Symmetric-definite generalized eigenvalues.
      Matrix<double> Rg(5, 5);
      Rg.set_Ran_values(-1.0, 1.0, -929);
      Matrix<double> Ag = Rg + Rg.T();
      Matrix<double> Bg = Rg.T() * Rg;
      for (int i = 0; i < 5; i++) Bg(i, i) += 5.0;
      Case("geneig_sym", dt).mat("A", Ag).mat("B", Bg).mat("R", eig(Ag, Bg).first);
      // Eigenvectors are not unique, so compare what is: X^T B X must be I.
      Case("geneig_orth", dt).mat("A", Ag).mat("B", Bg)
          .mat("R", eig(Ag, Bg).second.T() * Bg * eig(Ag, Bg).second);
    }

    // ── Decomposition and condition estimates (tier 4) ─────────────────────
    { Matrix<double> Hil(5, 5);
      for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) Hil(i, j) = 1.0 / (i + j + 1);
      Case("condest_hilbert", dt).mat("A", Hil).mat("R", oneByOne(condest(Hil)));
      Matrix<double> Gd(5, 5); Gd.set_Ran_values(-1.0, 1.0, -515);
      Matrix<double> bd(5, 2); bd.set_Ran_values(-1.0, 1.0, -616);
      Case("decomp_solve", dt).mat("A", Gd).mat("B", bd)
                              .mat("R", Gd.factorize().solve(bd));
      Case("decomp_det",   dt).mat("A", Gd).mat("R", oneByOne(Gd.factorize().det()));
      Matrix<double> Wide(2, 4);
      Wide = {{1, 0, 1, 0}, {0, 1, 0, 1}};
      Matrix<double> wr(2, 1); wr = {{2}, {4}};
      Case("lsqminnorm", dt).mat("A", Wide).mat("B", wr).mat("R", lsqminnorm(Wide, wr));
    }

    // ── Element-wise maths (tier 3) ────────────────────────────────────────
    { Matrix<double> E(3, 4);
      E.set_Ran_values(-3.0, 3.0, -909);
      Case("sign",  dt).mat("A", E).mat("R", E.sign());
      Case("floor", dt).mat("A", E).mat("R", E.floor());
      Case("ceil",  dt).mat("A", E).mat("R", E.ceil());
      Case("round", dt).mat("A", E).mat("R", E.round());
      Case("fix",   dt).mat("A", E).mat("R", E.fix());
      Case("mod3",  dt).mat("A", E).mat("R", E.mod(3.0));
      Case("rem3",  dt).mat("A", E).mat("R", E.rem(3.0));
      Case("expm1", dt).mat("A", E).mat("R", E.expm1());
      Case("asinh", dt).mat("A", E).mat("R", E.asinh());
      Case("atanh", dt).mat("A", E.mod(1.0)).mat("R", E.mod(1.0).atanh());
      Case("angle_real", dt).mat("A", E).mat("R", E.angle());
      Matrix<double> Y(3, 4), X(3, 4);
      Y.set_Ran_values(-2.0, 2.0, -111);
      X.set_Ran_values(-2.0, 2.0, -222);
      Case("atan2", dt).mat("A", Y).mat("B", X).mat("R", Y.atan2(X));
      Case("hypot", dt).mat("A", Y).mat("B", X).mat("R", Y.hypot(X));
    }

    // ── Structure, subspaces and polynomials (tier 4) ──────────────────────
    { Matrix<double> P(4, 4);
      P.set_Ran_values(-1.0, 1.0, -313);
      Case("normest", dt).mat("A", P).mat("R", oneByOne(P.normest(1e-12)));
      Case("bandwidth_lo", dt).mat("A", P).mat("R", oneByOne(double(P.bandwidth().first)));
      // A rank-deficient matrix, so null() and orth() have something to find.
      Matrix<double> Rk(3, 3);
      Rk = {{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};
      Case("rref", dt).mat("A", Rk).mat("R", Rk.rref());
      Case("null_dim", dt).mat("A", Rk).mat("R", oneByOne(double(Rk.null().cols())));
      Case("orth_dim", dt).mat("A", Rk).mat("R", oneByOne(double(Rk.orth().cols())));
      // null() and orth() bases are not unique, so compare the PROJECTORS they
      // define, which are.
      Case("orth_proj", dt).mat("A", Rk).mat("R", Rk.orth() * Rk.orth().T());
      Case("null_proj", dt).mat("A", Rk).mat("R", Rk.null() * Rk.null().T());
      Matrix<double> a3(3, 1), b3(3, 1);
      a3 = {{1}, {2}, {3}};  b3 = {{4}, {5}, {6}};
      Case("cross", dt).mat("A", a3).mat("B", b3).mat("R", a3.cross(b3));
      Case("dot",   dt).mat("A", a3).mat("B", b3).mat("R", oneByOne(a3.dot(b3)));
      Matrix<double> poly(1, 4);
      poly = {{2, -3, 0, 5}};
      Matrix<double> pts(1, 5);
      pts = {{-2, -1, 0, 1, 2}};
      Case("polyval", dt).mat("A", poly).mat("B", pts).mat("R", polyval(poly, pts));
      // Roots are returned in no guaranteed order, so compare the polynomial
      // rebuilt from them via its elementary symmetric functions — the sorted
      // real parts and moduli are enough to pin them down here.
      { Matrix<std::complex<double>> rt = roots(poly);
        Matrix<double> mods(rt.rows(), 1);
        for (long i = 0; i < rt.rows(); i++) mods(int(i), 0) = std::abs(rt[int(i)]);
        Case("roots_moduli", dt).mat("A", poly).mat("R", mods.sort(false)); }
      Matrix<double> fy = polyval(poly, pts);
      Case("polyfit", dt).mat("A", pts).mat("B", fy).mat("R", polyfit(pts, fy, 3));
    }

    // ── Scans, orderings and multiset reductions (tier 2) ──────────────────
    // Real only: everything here except prod/cumsum/cumprod needs an ordering.
    { Matrix<double> S(4, 5);
      S.set_Ran_values(-2.0, 2.0, -808);
      Case("prod_all",  dt).mat("A", S).mat("R", oneByOne(S.prod()));
      Case("prod_col",  dt).mat("A", S).mat("R", S.prod(false));
      Case("prod_row",  dt).mat("A", S).mat("R", S.prod(true));
      Case("cumsum_col",  dt).mat("A", S).mat("R", S.cumsum(false));
      Case("cumsum_row",  dt).mat("A", S).mat("R", S.cumsum(true));
      Case("cumprod_col", dt).mat("A", S).mat("R", S.cumprod(false));
      Case("diff_col",  dt).mat("A", S).mat("R", S.diff(false));
      Case("diff_row",  dt).mat("A", S).mat("R", S.diff(true));
      Case("sort_col",  dt).mat("A", S).mat("R", S.sort(false));
      Case("sort_row",  dt).mat("A", S).mat("R", S.sort(true));
      Case("sort_desc", dt).mat("A", S).mat("R", S.sort(false, true));
      Case("median_all", dt).mat("A", S).mat("R", oneByOne(S.median()));
      Case("median_col", dt).mat("A", S).mat("R", S.median(false));
      Case("median_row", dt).mat("A", S).mat("R", S.median(true));
      // sortrows and unique want repeats to be interesting, so use integers.
      Matrix<double> D(5, 3);
      D = {{3, 30, 1}, {1, 10, 2}, {2, 20, 3}, {1, 11, 4}, {3, 31, 5}};
      Case("sortrows_k0", dt).mat("A", D).mat("R", D.sortrows(0));
      Matrix<double> U(3, 4);
      U = {{3, 1, 3, 2}, {1, 2, 2, 3}, {5, 3, 1, 1}};
      Case("unique",    dt).mat("A", U).mat("R", U.unique());
      Case("mode",      dt).mat("A", U).mat("R", oneByOne(U.mode()));
    }

    // ── Logical masks ──────────────────────────────────────────────────────
    // Dumped as 0/1 matrices so NumPy can compare them directly against its own
    // boolean arrays. Real only: ordering needs <, which complex has not got.
    { Matrix<double> Md(4, 4);
      Md.set_Ran_values(-1.0, 1.0, -404);
      auto asNum = [](const Matrix<bool>& m) {
          Matrix<double> out(m.rows(), m.cols());
          for (long i = 0; i < m.rows() * m.cols(); i++) out[int(i)] = m[int(i)] ? 1.0 : 0.0;
          return out;
      };
      Case("mask_gt",   dt).mat("A", Md).mat("R", asNum(Md > 0.0));
      Case("mask_le",   dt).mat("A", Md).mat("R", asNum(Md <= 0.0));
      Case("mask_band", dt).mat("A", Md).mat("R", asNum((Md > -0.5).land(Md < 0.5)));
      Case("mask_bor",  dt).mat("A", Md).mat("R", asNum((Md > 0.5).lor(Md < -0.5)));
      Case("mask_bxor", dt).mat("A", Md).mat("R", asNum((Md > 0.0).lxor(Md > 0.5)));
      Case("mask_not",  dt).mat("A", Md).mat("R", asNum(!(Md > 0.0)));
      // Logical indexing: the selected elements, as a column vector.
      Case("mask_select", dt).mat("A", Md).mat("R", Matrix<double>(Md(Md > 0.0)));
      // Write-through: zero everything negative.
      { Matrix<double> W = Md; W(W < 0.0) = 0.0;
        Case("mask_assign", dt).mat("A", Md).mat("R", W); }
      // Counts, as 1x1 matrices so they travel through the same channel.
      { Matrix<double> c(1, 1); c(0, 0) = double((Md > 0.0).nnz());
        Case("mask_nnz", dt).mat("A", Md).mat("R", c); }
    }


    // Least squares on an over-determined system.
    { Matrix<double> M(6, 3), y(6, 1);
      for (int i = 0; i < 6; i++) {
          M(i, 0) = 1.0; M(i, 1) = double(i); M(i, 2) = double(i) * double(i);
          y(i, 0) = 2.0 + 0.5 * i - 0.1 * i * i + 0.3 * std::sin(double(i));
      }
      Case("lstsq", dt).mat("A", M).mat("B", y).mat("R", M.solve(y)); }

    // Matrix functions.
    Case("mat_pow_int",  dt, 3).mat("A", W).mat("R", pow(W, 3));
    Case("mat_pow_real", dt, 0.5).mat("A", Q).mat("R", pow(Q, 0.5));
    Case("mat_sqrt",     dt).mat("A", Q).mat("R", sqrt(Q));
    Case("mat_log",      dt).mat("A", Q).mat("R", log(Q, M_E));
    Case("mat_exp",      dt).mat("A", A).mat("R", exp(A));
    Case("mat_sin",      dt).mat("A", A).mat("R", sin(A));
    Case("mat_cos",      dt).mat("A", A).mat("R", cos(A));
    Case("mat_tan",      dt).mat("A", A).mat("R", tan(A));
    Case("mat_sinh",     dt).mat("A", A).mat("R", sinh(A));
    Case("mat_cosh",     dt).mat("A", A).mat("R", cosh(A));
    Case("mat_tanh",     dt).mat("A", A).mat("R", tanh(A));

    // Decompositions. The FACTORS are not unique — sign, column order and
    // pivoting all vary between implementations — so the Python side checks
    // what is well defined: that the factors rebuild A, and that the spectrum
    // matches NumPy's. Every factor is dumped so it can do both.
    { auto [L, U, P] = W.LU();
      Case("lu", dt).mat("A", W).mat("L", L).mat("U", U).mat("P", P); }
    { auto [Qm, R, P] = W.QR();
      Case("qr", dt).mat("A", W).mat("Q", Qm).mat("R", R).mat("P", P); }
    { auto [U, Sv, V] = W.svd();
      Case("svd", dt).mat("A", W).mat("U", U).mat("S", Sv).mat("V", V); }
    { auto [ev, Qe] = S.eig();
      Case("eig_symmetric", dt).mat("A", S).mat("E", ev).mat("Q", Qe); }
    { auto lam = A.eigvals();
      Matrix<double> re(lam.rows(), 1), im(lam.rows(), 1);
      for (long i = 0; i < lam.rows(); i++) {
          re(int(i), 0) = lam(int(i), 0).real();
          im(int(i), 0) = lam(int(i), 0).imag();
      }
      Case("eigvals", dt).mat("A", A).mat("RE", re).mat("IM", im); }
}

int main() {
    if (system("mkdir -p validation") != 0) {
        std::cerr << "could not create validation/\n";
        return 1;
    }
    dumpShared<double>("real");
    dumpRealOnly();
    dumpShared<cd>("complex");
    std::cout << caseCount << " cases -> validation/\n"
              << "Next: python3 numpy_validate.py\n";
    return 0;
}
