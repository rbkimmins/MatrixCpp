#pragma once

// ==========================================================================
//  Sequences, special matrices and polynomials
// ==========================================================================
//
// linspace, logspace, range, hilb, pascal, toeplitz, hankel, vander, magic,
// randn / randi / randperm, and the polynomial routines (poly, polyfit, roots).
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "decomposition.hpp"

// ═══════════════════════════════════════════════════════════════════════════
//  Sequences and constructors  (tier 5)
// ═══════════════════════════════════════════════════════════════════════════

// range(start, stop, step) — MATLAB's colon operator, a:step:b, as a row
// vector.
//
// Built on std::iota, which is the standard library's own sequence generator
// and means exactly this: fill a range with successive increments. Going
// through it rather than hand-rolling the loop keeps the index arithmetic in
// one place, and the index sequence is EXACT — 0, 1, 2, ... in long — so the
// only floating point in the result is the single multiply that scales it.
// Accumulating `v += step` instead would drift, and drift more the longer the
// vector.
//
// The endpoint is included only when it lands on a step, as in MATLAB: 0:2:5
// gives 0 2 4.
inline Matrix<double> range(double start, double stop, double step = 1.0) {
    if (step == 0.0)
        throw std::invalid_argument("range: step must be non-zero");
    const double span = (stop - start) / step;
    if (span < 0.0)
        return Matrix<double>(1, 0);
    const long n = (long)std::floor(span + 1e-12) + 1;
    std::vector<long> idx((std::size_t)n);
    std::iota(idx.begin(), idx.end(), 0L);
    Matrix<double> out(1, n);
    for (long i = 0; i < n; i++)
        out(0, int(i)) = start + double(idx[(std::size_t)i]) * step;
    return out;
}

// linspace(a, b, n) — n points from a to b inclusive, as a row vector.
//
// The endpoint is SET, not computed. a + i*(b-a)/(n-1) does not reliably land
// on b for the last i, and "does linspace(0,1,101) contain exactly 1.0" is the
// sort of question loops get written around. NumPy and MATLAB both special-case
// it, and so does this.
inline Matrix<double> linspace(double a, double b, long n = 100) {
    if (n < 0)
        throw std::invalid_argument("linspace: n must be non-negative");
    Matrix<double> out(1, n);
    if (n == 0)
        return out;
    if (n == 1) {
        out(0, 0) = b;
        return out;
    }  // MATLAB returns b, not a
    std::vector<long> idx((std::size_t)n);
    std::iota(idx.begin(), idx.end(), 0L);
    const double d = (b - a) / double(n - 1);
    for (long i = 0; i < n; i++)
        out(0, int(i)) = a + double(idx[(std::size_t)i]) * d;
    out(0, int(n - 1)) = b;  // exact, by construction
    return out;
}

// logspace(a, b, n) — n points from 10^a to 10^b, logarithmically spaced.
inline Matrix<double> logspace(double a, double b, long n = 50) {
    Matrix<double> e = linspace(a, b, n);
    for (long i = 0; i < n; i++)
        e(0, int(i)) = std::pow(10.0, e(0, int(i)));
    return e;
}

// ── Structured test matrices ───────────────────────────────────────────────
// These exist so that tests have something harder than random data to chew on.
// hilb in particular is THE standard ill-conditioning case — hilb(5) has a
// 1-norm condition number of 9.4e5, hilb(12) exceeds double precision entirely
// — and wilkinson is the standard eigenvalue stress case, with pairs of
// eigenvalues that are close but not equal.

// Hilbert matrix, H(i,j) = 1/(i+j-1) in 1-based terms. Symmetric, positive
// definite, and famously ill conditioned.
inline Matrix<double> hilb(long n) {
    Matrix<double> H(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            H(int(i), int(j)) = 1.0 / double(i + j + 1);
    return H;
}

// Pascal matrix — symmetric positive definite, built from binomial
// coefficients, with determinant exactly 1 for every n.
inline Matrix<double> pascal(long n) {
    Matrix<double> P(n, n);
    for (long i = 0; i < n; i++)
        P(int(i), 0) = 1.0;
    for (long j = 0; j < n; j++)
        P(0, int(j)) = 1.0;
    for (long i = 1; i < n; i++)
        for (long j = 1; j < n; j++)
            P(int(i), int(j)) = P(int(i - 1), int(j)) + P(int(i), int(j - 1));
    return P;
}

// Wilkinson's eigenvalue test matrix: symmetric tridiagonal, with eigenvalues
// in close pairs that a careless algorithm merges.
inline Matrix<double> wilkinson(long n) {
    Matrix<double> W(n, n);
    const double half = double(n - 1) / 2.0;
    for (long i = 0; i < n; i++)
        W(int(i), int(i)) = std::abs(half - double(i));
    for (long i = 0; i + 1 < n; i++) {
        W(int(i), int(i + 1)) = 1.0;
        W(int(i + 1), int(i)) = 1.0;
    }
    return W;
}

// Toeplitz matrix: constant along every diagonal. c is the first column, r the
// first row. r(0) is ignored — c(0) wins the corner, as in MATLAB.
template <typename datatype>
Matrix<datatype> toeplitz(const Matrix<datatype>& c, const Matrix<datatype>& r) {
    const long m = c.rows() * c.cols(), n = r.rows() * r.cols();
    if (m == 0 || n == 0)
        throw std::invalid_argument("toeplitz: vectors must be non-empty");
    Matrix<datatype> T(m, n);
    for (long i = 0; i < m; i++)
        for (long j = 0; j < n; j++)
            T(int(i), int(j)) = (i >= j) ? c[int(i - j)] : r[int(j - i)];
    return T;
}
// Symmetric Toeplitz from one vector.
template <typename datatype>
Matrix<datatype> toeplitz(const Matrix<datatype>& c) {
    return toeplitz(c, c);
}

// Hankel matrix: constant along every ANTI-diagonal — Toeplitz flipped. c is
// the first column, r the last row; the corner element comes from c.
template <typename datatype>
Matrix<datatype> hankel(const Matrix<datatype>& c, const Matrix<datatype>& r) {
    const long m = c.rows() * c.cols(), n = r.rows() * r.cols();
    if (m == 0 || n == 0)
        throw std::invalid_argument("hankel: vectors must be non-empty");
    Matrix<datatype> H(m, n);
    for (long i = 0; i < m; i++)
        for (long j = 0; j < n; j++) {
            const long k = i + j;
            H(int(i), int(j)) = (k < m) ? c[int(k)] : r[int(k - m + 1)];
        }
    return H;
}
template <typename datatype>
Matrix<datatype> hankel(const Matrix<datatype>& c) {
    Matrix<datatype> zero(1, c.rows() * c.cols());
    return hankel(c, zero);
}

// Vandermonde matrix, V(i,j) = v(i)^(n-1-j) — DESCENDING powers, so that
// V * p evaluates the polynomial p at every v(i) with p in the same
// descending-power order polyval, polyfit and roots all use.
template <typename datatype>
Matrix<double> vander(const Matrix<datatype>& v, long cols = -1) {
    const long n = v.rows() * v.cols();
    const long c = (cols < 0) ? n : cols;
    Matrix<double> V(n, c);
    for (long i = 0; i < n; i++) {
        const double x = double(std::real(v[int(i)]));
        double acc = 1.0;
        for (long j = c - 1; j >= 0; j--) {
            V(int(i), int(j)) = acc;
            acc *= x;
        }
    }
    return V;
}

// Magic square: every row, column and both diagonals sum to n(n²+1)/2.
// Three cases, as MATLAB has: odd by the Siamese method, doubly even (n % 4 ==
// 0) by a complement pattern, singly even (n % 4 == 2) by the LUX method.
inline Matrix<double> magic(long n) {
    if (n < 1)
        throw std::invalid_argument("magic: n must be >= 1");
    if (n == 2)
        throw std::invalid_argument("magic: no 2x2 magic square exists");
    Matrix<double> M(n, n);
    if (n % 2 == 1) {  // Siamese
        long i = 0, j = n / 2;
        for (long k = 1; k <= n * n; k++) {
            M(int(i), int(j)) = double(k);
            const long ni = (i - 1 + n) % n, nj = (j + 1) % n;
            if (M(int(ni), int(nj)) != 0.0)
                i = (i + 1) % n;
            else {
                i = ni;
                j = nj;
            }
        }
    } else if (n % 4 == 0) {  // doubly even
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++) {
                const long v = i * n + j + 1;
                const bool keep = ((i % 4 == 0 || i % 4 == 3) == (j % 4 == 0 || j % 4 == 3));
                M(int(i), int(j)) = keep ? double(n * n + 1 - v) : double(v);
            }
    } else {  // singly even, LUX
        const long h = n / 2, k = (n - 2) / 4;
        Matrix<double> A = magic(h);
        for (long i = 0; i < h; i++)
            for (long j = 0; j < h; j++) {
                const double a = A(int(i), int(j));
                M(int(i), int(j)) = a;
                M(int(i + h), int(j + h)) = a + double(h * h);
                M(int(i), int(j + h)) = a + double(2 * h * h);
                M(int(i + h), int(j)) = a + double(3 * h * h);
            }
        for (long i = 0; i < h; i++) {
            for (long j = 0; j < k; j++) {
                const long jj = (i == h / 2) ? j + h / 2 : j;  // shift the middle row
                if (jj < h)
                    std::swap(M(int(i), int(jj)), M(int(i + h), int(jj)));
            }
            for (long j = 0; j + 1 < k; j++) {
                const long jj = n - 1 - j;
                std::swap(M(int(i), int(jj)), M(int(i + h), int(jj)));
            }
        }
    }
    return M;
}

// ── Random constructors ────────────────────────────────────────────────────
// ALL of these go through ran2() from random.hpp — the L'Ecuyer generator from
// Numerical Recipes — and never through std::rand, whose low bits are famously
// poor and whose period can be as short as 32767. Seeds are NEGATIVE, which is
// how ran2 signals "initialise this stream"; the existing set_Ran_values()
// enforces the same rule and this stays consistent with it.
//
// NOT THREAD SAFE: ran2 keeps static state between calls, so two threads
// drawing at once would interleave and corrupt the sequence. Every filler here
// is deliberately serial for that reason — this is one place the OpenMP
// treatment the rest of the header gets would be actively wrong.

// Normally distributed values, by the Box-Muller transform on two uniforms.
// Marsaglia's polar variant would avoid the sin/cos, but Box-Muller consumes
// exactly two draws per pair, which keeps a given seed reproducible.
inline Matrix<double> randn(
    long rows, long cols, long seed, double mean = 0.0, double stddev = 1.0) {
    if (seed >= 0)
        throw std::invalid_argument(
            "randn: seed must be negative — that is how ran2 marks a fresh stream");
    Matrix<double> out(rows, cols);
    const long total = rows * cols;
    long s = seed;
    for (long i = 0; i < total; i += 2) {
        double u1 = double(ran2(&s)), u2 = double(ran2(&s));
        if (u1 < 1e-300)
            u1 = 1e-300;  // log(0) guard
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double t = 2.0 * mconst::pi * u2;
        out[int(i)] = mean + stddev * r * std::cos(t);
        if (i + 1 < total)
            out[int(i + 1)] = mean + stddev * r * std::sin(t);
    }
    return out;
}

// Uniform integers in [lo, hi], inclusive at both ends.
inline Matrix<long> randi(long lo, long hi, long rows, long cols, long seed) {
    if (seed >= 0)
        throw std::invalid_argument("randi: seed must be negative");
    if (lo > hi)
        throw std::invalid_argument("randi: lo must not exceed hi, got " + std::to_string(lo) +
                                    " > " + std::to_string(hi));
    Matrix<long> out(rows, cols);
    long s = seed;
    const long span = hi - lo + 1;
    for (long i = 0; i < rows * cols; i++) {
        long v = lo + (long)(double(ran2(&s)) * double(span));
        if (v > hi)
            v = hi;  // ran2 can return 1-eps
        out[int(i)] = v;
    }
    return out;
}

// A random permutation of 0 .. n-1, by the Fisher-Yates shuffle — which is the
// only shuffle that is uniform over all n! orderings, and is O(n).
inline Matrix<long> randperm(long n, long seed) {
    if (seed >= 0)
        throw std::invalid_argument("randperm: seed must be negative");
    Matrix<long> out(1, n);
    std::vector<long> v((std::size_t)n);
    std::iota(v.begin(), v.end(), 0L);  // 0,1,...,n-1
    long s = seed;
    for (long i = n - 1; i > 0; i--) {
        const long j = (long)(double(ran2(&s)) * double(i + 1));
        std::swap(v[(std::size_t)i], v[(std::size_t)(j > i ? i : j)]);
    }
    for (long i = 0; i < n; i++)
        out(0, int(i)) = v[(std::size_t)i];
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Polynomials  (tier 4 / tier 6)
// ═══════════════════════════════════════════════════════════════════════════
// A polynomial is a vector of coefficients in DESCENDING powers, the same
// convention MATLAB and NumPy both use:
//     {{1, -3, 2}}  is  x² - 3x + 2
// Row or column vector, either way.

// Evaluates p at every element of x, by Horner's rule — n multiplies and n adds
// per point, and far better conditioned than summing c_k * x^k separately.
template <typename datatype, typename dtX>
Matrix<double> polyval(const Matrix<datatype>& p, const Matrix<dtX>& x) {
    const long np = p.rows() * p.cols();
    if (np == 0)
        throw std::invalid_argument("polyval: the coefficient vector is empty");
    if (p.rows() != 1 && p.cols() != 1)
        throw std::invalid_argument("polyval: p must be a row or column vector");
    Matrix<double> out(x.rows(), x.cols());
    for (long i = 0; i < x.rows() * x.cols(); i++) {
        double acc = double(std::real(p[int(0)]));
        const double xi = double(std::real(x[int(i)]));
        for (long k = 1; k < np; k++)
            acc = acc * xi + double(std::real(p[int(k)]));
        out[int(i)] = acc;
    }
    return out;
}

// Roots of a polynomial, as the eigenvalues of its companion matrix — which is
// how MATLAB, NumPy and every serious library do it, because the QR iteration
// is backward-stable where root-finding by deflation is not.
//
// Leading zeros are stripped first (they are roots at infinity, not roots), and
// trailing zeros become exact roots at zero rather than being left for the
// eigenvalue solver to approximate.
template <typename datatype>
Matrix<std::complex<double>> roots(const Matrix<datatype>& p) {
    if (p.rows() != 1 && p.cols() != 1)
        throw std::invalid_argument("roots: p must be a row or column vector");
    std::vector<double> c;
    for (long i = 0; i < p.rows() * p.cols(); i++)
        c.push_back(double(std::real(p[int(i)])));
    std::size_t lead = 0;
    while (lead < c.size() && c[lead] == 0.0)
        lead++;
    if (lead == c.size())
        throw std::invalid_argument("roots: p is identically zero");
    c.erase(c.begin(), c.begin() + (long)lead);
    long zeroRoots = 0;
    while (c.size() > 1 && c.back() == 0.0) {
        c.pop_back();
        zeroRoots++;
    }
    const long deg = (long)c.size() - 1;
    Matrix<std::complex<double>> out(deg + zeroRoots, 1);
    if (deg > 0) {
        // Companion matrix: first row is -c1/c0 ... -cn/c0, subdiagonal ones.
        Matrix<double> C(deg, deg);
        for (long j = 0; j < deg; j++)
            C(0, int(j)) = -c[(std::size_t)(j + 1)] / c[0];
        for (long i = 1; i < deg; i++)
            C(int(i), int(i - 1)) = 1.0;
        Matrix<std::complex<double>> ev = C.eigvals();
        for (long i = 0; i < deg; i++)
            out[int(i)] = ev[int(i)];
    }
    for (long i = 0; i < zeroRoots; i++)
        out[int(deg + i)] = std::complex<double>(0.0, 0.0);
    return out;
}

// Least-squares polynomial fit of the given degree, returned in the same
// descending-power order polyval and roots expect.
//
// Built on solve(), which for an over-determined system goes through the
// column-pivoted QR — so this inherits that stability rather than forming and
// inverting the normal equations, which would square the condition number.
template <typename datatype>
Matrix<double> polyfit(const Matrix<datatype>& x, const Matrix<datatype>& y, long degree) {
    const long n = x.rows() * x.cols();
    if (n != y.rows() * y.cols())
        throw std::invalid_argument("polyfit: x and y must have the same number of points, got " +
                                    std::to_string(n) + " and " +
                                    std::to_string(y.rows() * y.cols()));
    if (degree < 0)
        throw std::invalid_argument("polyfit: degree must be >= 0");
    if (n < degree + 1)
        throw std::invalid_argument(
            "polyfit: need at least degree+1 = " + std::to_string(degree + 1) +
            " points to fit degree " + std::to_string(degree) + ", got " + std::to_string(n));
    // Vandermonde in descending powers, matching the coefficient order.
    Matrix<double> V(n, degree + 1);
    for (long i = 0; i < n; i++) {
        double xi = double(std::real(x[int(i)])), acc = 1.0;
        for (long k = degree; k >= 0; k--) {
            V(int(i), int(k)) = acc;
            acc *= xi;
        }
    }
    Matrix<double> b(n, 1);
    for (long i = 0; i < n; i++)
        b(int(i), 0) = double(std::real(y[int(i)]));
    return V.solve(b);
}
