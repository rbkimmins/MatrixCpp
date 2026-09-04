#pragma once

// ==========================================================================
//  The global identity matrix
// ==========================================================================
//
// IdentityMatrix and the inline `I` instance, so that A * I and I * A read the
// way they do on paper without allocating an n x n array.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "signal.hpp"

namespace mcpu {



// ============================================================
// IdentityMatrix — lazy proxy for k*I
//   n == 0  : dynamic  (size inferred when used in an expression)
//   n  > 0  : fixed    (materialized via I(n) assignment)
//   scale   : multiplier (supports k*I, I+I, etc.)
// ============================================================
class IdentityMatrix {
  public:
    IdentityMatrix() : n(0), scale(1.0L) {}

    // I(size) — fix the dimension, return a new proxy
    IdentityMatrix operator()(unsigned int size) const {
        if (size == 0)
            throw std::invalid_argument("IdentityMatrix: size must be positive, got 0");
        return IdentityMatrix(size, scale);
    }

    // Materialize to a concrete Matrix<T> (triggered by: Matrix<T> A = I(n);)
    template <typename T>
    operator Matrix<T>() const {
        if (n == 0)
            throw std::invalid_argument(
                "IdentityMatrix: cannot materialize a dynamic identity matrix "
                "without a fixed size — use I(n) to specify one");
        Matrix<T> ans(n, n);
        // n is unsigned; the counter matches it so the comparison does not warn.
        for (unsigned int i = 0; i < n; i++)
            ans(int(i), int(i)) = static_cast<T>(scale);
        return ans;
    }

    // kI + kI  →  (k1+k2)I      handles I+I+I+... chains
    IdentityMatrix operator+(const IdentityMatrix& rhs) const {
        return IdentityMatrix(resolveSize(n, rhs.n, "operator+"), scale + rhs.scale);
    }

    // kI * kI  →  (k1*k2)I
    IdentityMatrix operator*(const IdentityMatrix& rhs) const {
        return IdentityMatrix(resolveSize(n, rhs.n, "operator*"), scale * rhs.scale);
    }

    // I * scalar  (scalar on right: I * k)
    template <typename scalar>
    IdentityMatrix operator*(const scalar k) const {
        return IdentityMatrix(n, scale * static_cast<long double>(k));
    }

    unsigned int size() const { return n; }
    long double getScale() const { return scale; }

  private:
    unsigned int n;
    long double scale;

    IdentityMatrix(unsigned int size, long double s) : n(size), scale(s) {}

    // size resolution rules:
    //   dynamic + dynamic → dynamic (0)
    //   dynamic + fixed   → fixed
    //   fixed   + fixed   → must match, else throw
    static unsigned int resolveSize(unsigned int a, unsigned int b, const char* op) {
        if (a == 0)
            return b;
        if (b == 0)
            return a;
        if (a != b)
            throw std::invalid_argument(std::string("IdentityMatrix size mismatch in ") + op +
                                        ": " + std::to_string(a) + " != " + std::to_string(b));
        return a;
    }
};

// scalar * I  (scalar on left)
template <typename scalar>
IdentityMatrix operator*(const scalar k, const IdentityMatrix& Id) {
    return Id * k;
}

// A + I  —  adds scale to each diagonal element; A must be square
template <typename datatype>
Matrix<datatype> operator+(Matrix<datatype> A, const IdentityMatrix& Id) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator+(Matrix, IdentityMatrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.rows())
            throw std::invalid_argument(
                "operator+(Matrix, IdentityMatrix): size mismatch: Matrix is " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + " but I(" +
                std::to_string(sz) + ") was requested");
        for (unsigned int i = 0; i < A.rows(); i++)
            A(i, i) += static_cast<datatype>(Id.getScale());
        return A;
    } catch (const std::exception& e) {
        std::cerr << "Matrix + IdentityMatrix error: " << e.what() << std::endl;
        throw;
    }
}

// I + A  —  commutative
template <typename datatype>
Matrix<datatype> operator+(const IdentityMatrix& Id, Matrix<datatype> A) {
    return A + Id;
}

// A * I  —  A must be square; returns scale * A
template <typename datatype>
Matrix<datatype> operator*(Matrix<datatype> A, const IdentityMatrix& Id) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator*(Matrix, IdentityMatrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.cols())
            throw std::invalid_argument(
                "operator*(Matrix, IdentityMatrix): size mismatch: Matrix cols=" +
                std::to_string(A.cols()) + " but I(" + std::to_string(sz) + ")");
        return A * static_cast<datatype>(Id.getScale());
    } catch (const std::exception& e) {
        std::cerr << "Matrix * IdentityMatrix error: " << e.what() << std::endl;
        throw;
    }
}

// I * A  —  A must be square; returns scale * A
template <typename datatype>
Matrix<datatype> operator*(const IdentityMatrix& Id, Matrix<datatype> A) {
    try {
        if (A.rows() != A.cols())
            throw std::invalid_argument(
                "operator*(IdentityMatrix, Matrix): Matrix must be square, got " +
                std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        unsigned int sz = Id.size();
        if (sz != 0 && sz != A.rows())
            throw std::invalid_argument("operator*(IdentityMatrix, Matrix): size mismatch: I(" +
                                        std::to_string(sz) +
                                        ") but Matrix rows=" + std::to_string(A.rows()));
        return A * static_cast<datatype>(Id.getScale());
    } catch (const std::exception& e) {
        std::cerr << "IdentityMatrix * Matrix error: " << e.what() << std::endl;
        throw;
    }
}

// Global instance — include this header and 'I' is ready to use, inline version
// is for std=C++17 and beyond
inline const IdentityMatrix I;
// const IdentityMatrix I;

}  // namespace mcpu
