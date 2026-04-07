#include "Matrix1.0.hpp"
using namespace std;

void section(const string& title) {
    cout << "\n========================================\n";
    cout << "  " << title << "\n";
    cout << "========================================\n";
}

int main(){

    // -------------------------------------------------------
    // Setup test matrices
    // -------------------------------------------------------
    section("Test Matrices (integer)");
    Matrix<int> A(3, 3);  A.set_Ran_values(1, 9, -11);
    Matrix<int> B(3, 3);  B.set_Ran_values(1, 9, -22);
    Matrix<int> C(3, 1);  C.set_Ran_values(1, 9, -33);  // column vector
    Matrix<int> D(1, 3);  D.set_Ran_values(1, 9, -44);  // row vector
    Matrix<int> E(2, 3);  E.set_Ran_values(1, 9, -55);  // non-square
    Matrix<int> F(3, 2);  F.set_Ran_values(1, 9, -66);  // non-square

    cout << "A (3x3):\n" << A << "\n";
    cout << "B (3x3):\n" << B << "\n";
    cout << "C (3x1 column vector):\n" << C << "\n";
    cout << "D (1x3 row vector):\n" << D << "\n";
    cout << "E (2x3):\n" << E << "\n";
    cout << "F (3x2):\n" << F << "\n";

    section("Test Matrices (double)");
    Matrix<double> Ad(3, 3);  Ad.set_Ran_values(1.0, 9.0, -11);
    Matrix<double> Bd(3, 3);  Bd.set_Ran_values(1.0, 9.0, -22);
    cout << "Ad (3x3, double):\n" << Ad << "\n";
    cout << "Bd (3x3, double):\n" << Bd << "\n";

    // -------------------------------------------------------
    // Assignment
    // -------------------------------------------------------
    section("Initializer List Assignment");
    Matrix<double> Init(2, 2);
    Init = {{1.5, 2.5}, {3.5, 4.5}};
    cout << "Init = {{1.5, 2.5}, {3.5, 4.5}}:\n" << Init << "\n";

    // -------------------------------------------------------
    // Addition and subtraction
    // -------------------------------------------------------
    section("Addition: A + B");
    cout << "A:\n" << A << "\n";
    cout << "B:\n" << B << "\n";
    cout << "A + B:\n" << (A + B) << "\n";

    section("Subtraction: A - B");
    cout << "A - B:\n" << (A - B) << "\n";

    // -------------------------------------------------------
    // Scalar multiplication and division
    // -------------------------------------------------------
    section("Scalar Multiplication: A * 3  and  3 * A");
    cout << "A * 3:\n" << (A * 3) << "\n";
    cout << "3 * A:\n" << (3 * A) << "\n";

    section("Scalar Division: Ad / 2.0");
    cout << "Ad / 2.0:\n" << (Ad / 2.0) << "\n";

    section("In-place Scalar Multiply: A *= 2");
    Matrix<int> Atmp = A;
    Atmp *= 2;
    cout << "A *= 2:\n" << Atmp << "\n";

    section("In-place Scalar Divide: Ad /= 2.0");
    Matrix<double> Adtmp = Ad;
    Adtmp /= 2.0;
    cout << "Ad /= 2.0:\n" << Adtmp << "\n";

    // -------------------------------------------------------
    // Matrix multiplication
    // -------------------------------------------------------
    section("Matrix Multiply: A * B  (3x3 * 3x3 -> 3x3)");
    printSideBySide(A, "*", B);
    cout << "Result:\n" << (A * B) << "\n";

    section("Matrix Multiply: E * F  (2x3 * 3x2 -> 2x2)");
    printSideBySide(E, "*", F);
    cout << "Result:\n" << (E * F) << "\n";

    section("Matrix Multiply: D * C  (1x3 * 3x1 -> 1x1, dot product)");
    printSideBySide(D, "*", C);
    cout << "Result:\n" << (D * C) << "\n";

    section("In-place Matrix Multiply: B *= A");
    Matrix<int> Btmp = B;
    Btmp *= A;
    cout << "B *= A:\n" << Btmp << "\n";

    // -------------------------------------------------------
    // Hadamard product
    // -------------------------------------------------------
    section("Hadamard Product: A % B  (element-wise multiply)");
    printSideBySide(A, "%", B);
    cout << "Result:\n" << (A % B) << "\n";

    section("In-place Hadamard: A %= B");
    Matrix<int> Ahadtmp = A;
    Ahadtmp %= B;
    cout << "A %= B:\n" << Ahadtmp << "\n";

    // -------------------------------------------------------
    // Integer modulo
    // -------------------------------------------------------
    section("Integer Modulo: A % 3  (each element mod 3)");
    cout << "A:\n" << A << "\n";
    cout << "A % 3:\n" << (A % 3) << "\n";

    section("In-place Modulo: A %= 3");
    Matrix<int> Amodtmp = A;
    Amodtmp %= 3;
    cout << "A %= 3:\n" << Amodtmp << "\n";

    // -------------------------------------------------------
    // Transpose
    // -------------------------------------------------------
    section("Transpose: A.T()");
    printSideBySide(A, "->", A.T());

    section("Transpose: E.T()  (2x3 -> 3x2)");
    printSideBySide(E, "->", E.T());

    // -------------------------------------------------------
    // Indexing
    // -------------------------------------------------------
    section("Element Indexing: A(i, j)");
    cout << "A:\n" << A << "\n";
    cout << "A(0,0) = " << A(0,0) << "\n";
    cout << "A(1,2) = " << A(1,2) << "\n";
    cout << "A(2,2) = " << A(2,2) << "\n";

    section("Flat Index: A[i]  (row-major)");
    cout << "A[0] = " << A[0] << "  (same as A(0,0))\n";
    cout << "A[4] = " << A[4] << "  (same as A(1,1))\n";

    section("Row Extraction: A(i, all)");
    cout << "A:\n" << A << "\n";
    cout << "A(0, all) = row 0:\n" << A(0, all) << "\n";
    cout << "A(2, all) = row 2:\n" << A(2, all) << "\n";

    section("Column Extraction: A(all, j)");
    cout << "A(all, 0) = col 0:\n" << A(all, 0) << "\n";
    cout << "A(all, 2) = col 2:\n" << A(all, 2) << "\n";

    // -------------------------------------------------------
    // Trace, determinant
    // -------------------------------------------------------
    section("Trace: A.trace()");
    cout << "A:\n" << A << "\n";
    cout << "trace(A) = " << A.trace() << "\n";

    section("Determinant: Ad.det()");
    cout << "Ad:\n" << Ad << "\n";
    cout << "det(Ad) = " << Ad.det() << "\n";

    section("Determinant (integer matrix): A.det()");
    cout << "A:\n" << A << "\n";
    cout << "det(A) = " << A.det() << "\n";

    // -------------------------------------------------------
    // LU factorisation
    // -------------------------------------------------------
    section("LU Factorisation: Ad.LU()  ->  L, U, P  where P*Ad = L*U");
    auto lu = Ad.LU();
    Matrix<double> L = lu.getL();
    Matrix<double> U = lu.getU();
    Matrix<double> P = lu.getPivot();
    cout << "Ad:\n" << Ad << "\n";
    cout << "L:\n" << L << "\n";
    cout << "U:\n" << U << "\n";
    cout << "P:\n" << P << "\n";
    cout << "Verification — P * Ad:\n" << (P * Ad) << "\n";
    cout << "Verification — L * U:\n" << (L * U) << "\n";

    // -------------------------------------------------------
    // Sum
    // -------------------------------------------------------
    section("Sum: A.sum()  (all elements)");
    cout << "A:\n" << A << "\n";
    cout << "sum(A) = " << A.sum() << "\n";

    section("Sum: A.sum(0)  (column sums -> 1 x cols row vector)");
    cout << "A.sum(0):\n" << A.sum(0) << "\n";

    section("Sum: A.sum(1)  (row sums -> rows x 1 column vector)");
    cout << "A.sum(1):\n" << A.sum(1) << "\n";

    // -------------------------------------------------------
    // Concat
    // -------------------------------------------------------
    section("Concat vertical (0): A on top of B  (6x3)");
    cout << "A:\n" << A << "\nB:\n" << B << "\n";
    cout << "A.concat(B, 0):\n" << A.concat(B, 0) << "\n";

    section("Concat horizontal (1): A beside B  (3x6)");
    cout << "A.concat(B, 1):\n" << A.concat(B, 1) << "\n";

    section("Concat with empty matrix (should return M)");
    Matrix<int> empty;
    cout << "empty.concat(A, 0):\n" << empty.concat(A, 0) << "\n";

    // -------------------------------------------------------
    // Tensor (Kronecker) product
    // -------------------------------------------------------
    section("Tensor (Kronecker) Product: A.tensor(B)");
    Matrix<int> T1(2, 2);  T1.set_Ran_values(1, 4, -77);
    Matrix<int> T2(2, 2);  T2.set_Ran_values(1, 4, -88);
    cout << "T1 (2x2):\n" << T1 << "\n";
    cout << "T2 (2x2):\n" << T2 << "\n";
    cout << "T1.tensor(T2) (4x4):\n" << T1.tensor(T2) << "\n";

    // -------------------------------------------------------
    // Identity matrix
    // -------------------------------------------------------
    section("Identity Matrix: I");
    Matrix<double> I3 = I(3);
    cout << "I(3):\n" << I3 << "\n";

    section("I + I  (dynamic, scale=2)");
    Matrix<double> G(3, 3);  G.set_Ran_values(1.0, 5.0, -99);
    cout << "G:\n" << G << "\n";
    cout << "G + I  (adds 1 to diagonal):\n" << (G + I) << "\n";
    cout << "G + 2*I  (adds 2 to diagonal):\n" << (G + 2*I) << "\n";
    cout << "G + I+I+I  (adds 3 to diagonal):\n" << (G + (I+I+I)) << "\n";

    section("G * I  (should return G)");
    cout << "G * I:\n" << (G * I) << "\n";

    section("I * G  (should return G)");
    cout << "I * G:\n" << (I * G) << "\n";

    // -------------------------------------------------------
    // Printing utilities
    // -------------------------------------------------------
    section("printSideBySide with operator string");
    printSideBySide(A, "+", B);
    cout << "=\n" << (A + B) << "\n";

    section("toString with precision 10 (for NumPy comparison)");
    cout << Ad.toString(10) << "\n";

    section("print() with precision 2");
    Ad.print(2);

    cout << "\n--- All tests complete ---\n";
    return 0;
}
