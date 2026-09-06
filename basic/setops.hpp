#pragma once

// ==========================================================================
//  Set operations
// ==========================================================================
//
// The family MATLAB spells union / intersect / setdiff / setxor / ismember /
// unique, and NumPy spells union1d / intersect1d / setdiff1d / setxor1d /
// isin / unique. A matrix is treated as a bag of values; the result is the
// distinct ones, ascending, as a COLUMN vector — the same shape and ordering
// promise Matrix::unique() already makes.
//
//     Matrix<double> a = {{3, 1, 4, 1, 5}};
//     Matrix<double> b = {{1, 5, 9}};
//
//     setunion(a, b)   ->  1 3 4 5 9
//     intersect(a, b)  ->  1 5
//     setdiff(a, b)    ->  3 4          in a, not in b
//     setxor(a, b)     ->  3 4 9        in one but not both
//     ismember(a, b)   ->  0 1 0 1 1    SHAPED LIKE a, not reduced
//
// WHY THERE IS NO `union`. It is a C++ keyword, so the name is unavailable and
// `setunion` stands in. The rest keep MATLAB's spelling exactly.
//
// WHY NOT std::set. Every one of these is implemented over sorted vectors and
// <algorithm>'s set_union / set_intersection / set_difference /
// set_symmetric_difference, not std::set. Three reasons, in order of weight:
//
//   * The RESULT has to come back sorted and contiguous, which is what a
//     Matrix is. A std::set would have to be walked and copied out anyway.
//   * A std::set allocates a node per element and scatters them through
//     memory. Sorting a contiguous buffer is one allocation and one pass,
//     and for the sizes a matrix library sees it wins comfortably.
//   * The <algorithm> versions are already the operations, exactly. Writing
//     them out of set insertions would be reimplementing them by hand.
//
// std::set is the right container when membership is queried while the set is
// still being built. That is not this.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "matrix.hpp"

#include <algorithm>
#include <vector>

namespace mcpu {

namespace setops_detail {

    // Sorted, deduplicated contents, which is the form all four set
    // operations below require of their inputs.
    template <typename datatype>
    std::vector<datatype> bag(const Matrix<datatype>& A) {
        std::vector<datatype> v;
        v.reserve((std::size_t)(A.rows() * A.cols()));
        for (long i = 0; i < A.rows(); i++)
            for (long j = 0; j < A.cols(); j++) v.push_back(A(i, j));
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }

    template <typename datatype>
    Matrix<datatype> column(const std::vector<datatype>& v) {
        Matrix<datatype> out((long)v.size(), 1);
        for (std::size_t i = 0; i < v.size(); i++) out[i] = v[i];
        return out;
    }

    // The one place the ordering requirement is stated, so the message is the
    // same wherever it is hit.
    template <typename datatype>
    void requireOrdered() {
        static_assert(!is_complex<datatype>::value,
                      "set operations need an ordering, which std::complex deliberately does\n"
                      "not provide. Apply them to .real() / .imag() / .abs() instead — for\n"
                      "example intersect(A.real(), B.real()).");
    }

}  // namespace setops_detail

// The distinct values of A, ascending, as a column vector. The free spelling
// of A.unique(), for symmetry with the rest of this header and with MATLAB.
template <typename datatype>
Matrix<datatype> unique(const Matrix<datatype>& A) {
    setops_detail::requireOrdered<datatype>();
    return setops_detail::column(setops_detail::bag(A));
}

// Every value in either — MATLAB's union, whose name C++ has taken.
template <typename datatype>
Matrix<datatype> setunion(const Matrix<datatype>& A, const Matrix<datatype>& B) {
    setops_detail::requireOrdered<datatype>();
    const auto a = setops_detail::bag(A), b = setops_detail::bag(B);
    std::vector<datatype> out;
    out.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return setops_detail::column(out);
}

// The values in both.
template <typename datatype>
Matrix<datatype> intersect(const Matrix<datatype>& A, const Matrix<datatype>& B) {
    setops_detail::requireOrdered<datatype>();
    const auto a = setops_detail::bag(A), b = setops_detail::bag(B);
    std::vector<datatype> out;
    out.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return setops_detail::column(out);
}

// In A but not in B. NOT symmetric: setdiff(a, b) and setdiff(b, a) differ.
template <typename datatype>
Matrix<datatype> setdiff(const Matrix<datatype>& A, const Matrix<datatype>& B) {
    setops_detail::requireOrdered<datatype>();
    const auto a = setops_detail::bag(A), b = setops_detail::bag(B);
    std::vector<datatype> out;
    out.reserve(a.size());
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return setops_detail::column(out);
}

// In one but not both — the symmetric difference, and the one case where
// setxor(a, b) == setxor(b, a).
template <typename datatype>
Matrix<datatype> setxor(const Matrix<datatype>& A, const Matrix<datatype>& B) {
    setops_detail::requireOrdered<datatype>();
    const auto a = setops_detail::bag(A), b = setops_detail::bag(B);
    std::vector<datatype> out;
    out.reserve(a.size() + b.size());
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(out));
    return setops_detail::column(out);
}

// Which elements of A appear in S. The ONE function here that does not reduce:
// the answer is a mask SHAPED LIKE A, so it composes with the rest of the
// library — A(ismember(A, S)) selects them, ismember(A, S).nnz() counts them.
//
// S is sorted once and each element of A found by binary search, so this is
// O(n log m) rather than the O(n*m) a nested loop would give.
template <typename datatype>
Matrix<bool> ismember(const Matrix<datatype>& A, const Matrix<datatype>& S) {
    setops_detail::requireOrdered<datatype>();
    const auto s = setops_detail::bag(S);
    Matrix<bool> out(A.rows(), A.cols());
    for (long i = 0; i < A.rows(); i++)
        for (long j = 0; j < A.cols(); j++)
            out(i, j) = std::binary_search(s.begin(), s.end(), A(i, j));
    return out;
}

// True when every value of A also appears in S. The question ismember answers
// element by element, asked of the whole matrix.
template <typename datatype>
bool issubset(const Matrix<datatype>& A, const Matrix<datatype>& S) {
    setops_detail::requireOrdered<datatype>();
    const auto a = setops_detail::bag(A), s = setops_detail::bag(S);
    return std::includes(s.begin(), s.end(), a.begin(), a.end());
}

}  // namespace mcpu
