#include "Matrix1.0.hpp"
#include <fstream>
#include <chrono>
#include <string>
#include <cmath>
using namespace std;
using namespace std::chrono;

// ════════════════════════════════════════════════════════════════════════════
//  BENCHMARK SELECTION — comment out any line to skip that operation
// ════════════════════════════════════════════════════════════════════════════
#define BENCH_MULTIPLY      // A * B            O(n³)  Strassen-Winograd
#define BENCH_ADD           // A + B            O(n²)
#define BENCH_SUBTRACT      // A - B            O(n²)
#define BENCH_SCALAR_MUL    // A * k            O(n²)
#define BENCH_HADAMARD      // A % B            O(n²)  element-wise product
#define BENCH_ELEM_DIV      // A / B            O(n²)  element-wise division
#define BENCH_SCALAR_DIV    // A / k            O(n²)
#define BENCH_TRANSPOSE     // A.T()            O(n²)
#define BENCH_ELEM_POW      // A.pow(p)         O(n²)
#define BENCH_ELEM_EXP      // A.exp()          O(n²)
#define BENCH_ELEM_LN       // A.ln()           O(n²)
#define BENCH_SUM_ALL       // A.sum()          O(n²)
#define BENCH_SUM_DIM       // A.sum(true)      O(n²)  row sums
#define BENCH_ISDIAGONAL    // A.IsDiagonal()   O(n²)
#define BENCH_CONCAT        // A.concat(B,1)    O(n²)  horizontal
#define BENCH_TRACE         // tr(A)            O(n)
#define BENCH_DET           // det(A)           O(n³)  via LU
#define BENCH_INVERSE       // A.inverse()      O(n³)  via LU
#define BENCH_LU            // A.LU()           O(n³)  Doolittle + partial pivoting
#define BENCH_QR            // A.QR()           O(n³)  Householder + column pivoting
#define BENCH_EIG           // A.eig()          O(n³)  implicit-shift QR
#define BENCH_MAT_POW_INT   // pow(A, 3)        O(n³ log p) binary squaring
#define BENCH_MAT_POW_REAL  // pow(A, 0.5)      O(n³)  Schur-Padé
#define BENCH_MAT_LOG       // log(A, M_E)      O(n³)  Schur-Padé
#define BENCH_TENSOR        // A.tensor(B)      O(n⁴)  Kronecker product

// ════════════════════════════════════════════════════════════════════════════
//  SIZE LIMITS per complexity class (max n for square n×n matrices)
// ════════════════════════════════════════════════════════════════════════════
static const long N_CUBIC  =  300;   // O(n³): multiply, LU, QR, det, inverse
static const long N_EIG    =  150;   // O(n³) iterative: eig, matrix pow/log
static const long N_SQUARE = 1500;   // O(n²): elementwise ops, transpose, etc.
static const long N_LINEAR = 8000;   // O(n):  trace
static const long N_TENSOR =   35;   // O(n⁴): tensor — result is n²×n²

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════
static void write_header(ofstream& f) { f << "size,time_seconds\n"; }

static double elapsed(high_resolution_clock::time_point a,
                      high_resolution_clock::time_point b) {
    return duration<double>(b - a).count();
}

int main() {
    system("mkdir -p plots");          // create plots output directory
    cout << "Running benchmarks — results written to rt_*.txt\n\n";

// ── O(n²) helpers (reused across several benchmarks) ────────────────────────
#if defined(BENCH_MULTIPLY) || defined(BENCH_ADD)     || defined(BENCH_SUBTRACT)  || \
    defined(BENCH_HADAMARD) || defined(BENCH_ELEM_DIV)|| defined(BENCH_SCALAR_MUL)|| \
    defined(BENCH_SCALAR_DIV)|| defined(BENCH_TRANSPOSE)|| defined(BENCH_ELEM_POW) || \
    defined(BENCH_ELEM_EXP) || defined(BENCH_ELEM_LN) || defined(BENCH_SUM_ALL)   || \
    defined(BENCH_SUM_DIM)  || defined(BENCH_ISDIAGONAL)|| defined(BENCH_CONCAT)  || \
    defined(BENCH_TRACE)    || defined(BENCH_DET)     || defined(BENCH_INVERSE)   || \
    defined(BENCH_LU)       || defined(BENCH_QR)
    // (variables declared per-benchmark below)
#endif

// ════════════════════════════════════════════════════════════════════════════
//  O(n) — TRACE
// ════════════════════════════════════════════════════════════════════════════
#ifdef BENCH_TRACE
    {
        ofstream out("rt_trace.txt"); write_header(out);
        for (long i = 2; i <= N_LINEAR; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto v = tr(A);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)v;
        }
        cout << "  trace done\n";
    }
#endif

// ════════════════════════════════════════════════════════════════════════════
//  O(n²) — ELEMENTWISE AND STRUCTURAL OPERATIONS
// ════════════════════════════════════════════════════════════════════════════
#ifdef BENCH_ADD
    {
        ofstream out("rt_add.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A + B;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  add done\n";
    }
#endif

#ifdef BENCH_SUBTRACT
    {
        ofstream out("rt_subtract.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A - B;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  subtract done\n";
    }
#endif

#ifdef BENCH_SCALAR_MUL
    {
        ofstream out("rt_scalar_mul.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A * 3.14;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  scalar_mul done\n";
    }
#endif

#ifdef BENCH_SCALAR_DIV
    {
        ofstream out("rt_scalar_div.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A / 3.14;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  scalar_div done\n";
    }
#endif

#ifdef BENCH_HADAMARD
    {
        ofstream out("rt_hadamard.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A % B;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  hadamard done\n";
    }
#endif

#ifdef BENCH_ELEM_DIV
    {
        ofstream out("rt_elem_div.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(1.0, 100.0);   // keep B away from zero
            auto t0 = high_resolution_clock::now();
            auto C = A / B;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  elem_div done\n";
    }
#endif

#ifdef BENCH_TRANSPOSE
    {
        ofstream out("rt_transpose.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.T();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  transpose done\n";
    }
#endif

#ifdef BENCH_ELEM_POW
    {
        ofstream out("rt_elem_pow.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(0.1, 10.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.pow(2.5);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  elem_pow done\n";
    }
#endif

#ifdef BENCH_ELEM_EXP
    {
        ofstream out("rt_elem_exp.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-5.0, 5.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.exp();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  elem_exp done\n";
    }
#endif

#ifdef BENCH_ELEM_LN
    {
        ofstream out("rt_elem_ln.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(0.1, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.ln();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  elem_ln done\n";
    }
#endif

#ifdef BENCH_SUM_ALL
    {
        ofstream out("rt_sum_all.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto v = A.sum();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)v;
        }
        cout << "  sum_all done\n";
    }
#endif

#ifdef BENCH_SUM_DIM
    {
        ofstream out("rt_sum_dim.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.sum(true);   // row sums → (n×1)
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  sum_dim done\n";
    }
#endif

#ifdef BENCH_ISDIAGONAL
    {
        ofstream out("rt_isdiagonal.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            bool r = A.IsDiagonal();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)r;
        }
        cout << "  isdiagonal done\n";
    }
#endif

#ifdef BENCH_CONCAT
    {
        ofstream out("rt_concat.txt"); write_header(out);
        for (long i = 2; i <= N_SQUARE; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.concat(B, true);   // horizontal: (i × 2i)
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  concat done\n";
    }
#endif

// ════════════════════════════════════════════════════════════════════════════
//  O(n³) — MATRIX OPERATIONS
// ════════════════════════════════════════════════════════════════════════════
#ifdef BENCH_MULTIPLY
    {
        ofstream out("rt_multiply.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-100.0, 100.0);
            B.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A * B;
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  multiply done\n";
    }
#endif

#ifdef BENCH_DET
    {
        ofstream out("rt_det.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto v = det(A);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)v;
        }
        cout << "  det done\n";
    }
#endif

#ifdef BENCH_INVERSE
    {
        ofstream out("rt_inverse.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.inverse();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  inverse done\n";
    }
#endif

#ifdef BENCH_LU
    {
        ofstream out("rt_lu.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto [L, U, P] = A.LU();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)L; (void)U; (void)P;
        }
        cout << "  lu done\n";
    }
#endif

#ifdef BENCH_QR
    {
        ofstream out("rt_qr.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-100.0, 100.0);
            auto t0 = high_resolution_clock::now();
            auto [Q, R, P] = A.QR();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)Q; (void)R; (void)P;
        }
        cout << "  qr done\n";
    }
#endif

#ifdef BENCH_MAT_POW_INT
    {
        ofstream out("rt_mat_pow_int.txt"); write_header(out);
        for (long i = 2; i <= N_CUBIC; i++) {
            Matrix<double> A(i, i); A.set_Ran_values(-10.0, 10.0);
            auto t0 = high_resolution_clock::now();
            auto C = pow(A, 3);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  mat_pow_int done\n";
    }
#endif

// ── These three require symmetric positive-definite matrices ─────────────────
#ifdef BENCH_EIG
    {
        ofstream out("rt_eig.txt"); write_header(out);
        for (long i = 2; i <= N_EIG; i++) {
            // A + A^T makes symmetric; diagonal shift ensures non-zero eigenvalues
            Matrix<double> B(i, i); B.set_Ran_values(-5.0, 5.0);
            Matrix<double> A = B + B.T();
            for (long k = 0; k < i; k++) A(k, k) += 2.0 * i;
            auto t0 = high_resolution_clock::now();
            auto [vals, vecs] = A.eig();
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)vals; (void)vecs;
        }
        cout << "  eig done\n";
    }
#endif

#ifdef BENCH_MAT_POW_REAL
    {
        ofstream out("rt_mat_pow_real.txt"); write_header(out);
        for (long i = 2; i <= N_EIG; i++) {
            // B^T*B is PSD; +I ensures strictly positive eigenvalues
            Matrix<double> B(i, i); B.set_Ran_values(-3.0, 3.0);
            Matrix<double> A = B.T() * B;
            for (long k = 0; k < i; k++) A(k, k) += 1.0;
            auto t0 = high_resolution_clock::now();
            auto C = pow(A, 0.5);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  mat_pow_real done\n";
    }
#endif

#ifdef BENCH_MAT_LOG
    {
        ofstream out("rt_mat_log.txt"); write_header(out);
        for (long i = 2; i <= N_EIG; i++) {
            Matrix<double> B(i, i); B.set_Ran_values(-3.0, 3.0);
            Matrix<double> A = B.T() * B;
            for (long k = 0; k < i; k++) A(k, k) += 1.0;
            auto t0 = high_resolution_clock::now();
            auto C = log(A, M_E);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  mat_log done\n";
    }
#endif

// ════════════════════════════════════════════════════════════════════════════
//  O(n⁴) — KRONECKER / TENSOR PRODUCT
// ════════════════════════════════════════════════════════════════════════════
#ifdef BENCH_TENSOR
    {
        ofstream out("rt_tensor.txt"); write_header(out);
        for (long i = 2; i <= N_TENSOR; i++) {
            Matrix<double> A(i, i), B(i, i);
            A.set_Ran_values(-10.0, 10.0);
            B.set_Ran_values(-10.0, 10.0);
            auto t0 = high_resolution_clock::now();
            auto C = A.tensor(B);
            auto t1 = high_resolution_clock::now();
            out << i << "," << elapsed(t0, t1) << "\n";
            (void)C;
        }
        cout << "  tensor done\n";
    }
#endif

    cout << "\nAll benchmarks complete.\n";
    return 0;
}
