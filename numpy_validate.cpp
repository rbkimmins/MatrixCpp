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

// ══════════════════════════════════════════════════ shared: real + complex ──
template<typename T>
static void dumpShared(const std::string& dt) {
    Matrix<T> A = operandA<T>(), B = operandB<T>(), P = operandPos<T>();

    Case("add",        dt).mat("A", A).mat("B", B).mat("R", A + B);
    Case("subtract",   dt).mat("A", A).mat("B", B).mat("R", A - B);
    Case("multiply",   dt).mat("A", A).mat("B", B).mat("R", A * B);
    Case("hadamard",   dt).mat("A", A).mat("B", B).mat("R", A % B);
    Case("elem_div",   dt).mat("A", A).mat("B", B).mat("R", A.ediv(B));
    Case("negate",     dt).mat("A", A).mat("R", -A);
    Case("scalar_mul", dt, 2.75).mat("A", A).mat("R", A * 2.75);
    Case("scalar_div", dt, 2.75).mat("A", A).mat("R", A / 2.75);

    Case("transpose",  dt).mat("A", A).mat("R", A.T());
    Case("conj",       dt).mat("A", A).mat("R", A.conj());
    Case("conj_transpose", dt).mat("A", A).mat("R", A.H());
    Case("real_part",  dt).mat("A", A).mat("R", A.real());
    Case("imag_part",  dt).mat("A", A).mat("R", A.imag());

    Case("elem_exp",   dt).mat("A", A).mat("R", A.exp());
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
    Case("tensor",     dt).mat("A", A).mat("B", B).mat("R", A.tensor(B));

    Case("norm_fro",   dt).mat("A", A).scalar("R", A.norm(NormType::Fro));
    Case("norm_one",   dt).mat("A", A).scalar("R", A.norm(NormType::One));
    Case("norm_inf",   dt).mat("A", A).scalar("R", A.norm(NormType::Inf));
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
