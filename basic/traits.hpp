#pragma once

// ==========================================================================
//  Type traits, tags and the small numeric helpers
// ==========================================================================
//
// is_complex / real_t / work_t / mean_t, the pairwise summation used by every
// reduction, the ROW and COL axis selectors, the NormType / QRMode / QRPivot
// enums and TaylorOpts. Everything a signature further down needs to name.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "constants.hpp"
#include <array>
#include <chrono>
#include <functional>

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <class T>
struct real_of {
    using type = T;
};
template <class T>
struct real_of<std::complex<T>> {
    using type = T;
};
template <class T>
using real_t = typename real_of<T>::type;

// The type a numeric ALGORITHM should work in for a given element type: double
// for anything real, complex<double> for anything complex. This is roadmap item
// 1, and it is what lets cholesky, svd and the Taylor matrix functions stop
// being real-only — each of them used to convert to Matrix<double> and so had to
// refuse a complex input rather than silently drop its imaginary part.
//
// Note it is NOT the same question as real_t. real_t strips complex away, for
// results that genuinely are real whatever went in — abs(), angle(), the
// singular values. work_t preserves it, for results that are complex when the
// input is.
template <class T>
using work_t = std::conditional_t<is_complex<T>::value, std::complex<double>, double>;

// mean() has to follow the input: the mean of complex numbers is complex, while
// the mean of ints is not an int. work_t answers both.
template <class T>
using mean_t = work_t<T>;

// True for the types toLines() should format with fixed/setprecision.
// std::is_floating_point<std::complex<double>> is FALSE, which is why this
// wrapper exists — see roadmap item 5.
template <class T>
struct is_float_like : std::integral_constant<bool,
                                              std::is_floating_point<T>::value ||
                                                  (is_complex<T>::value &&
                                                   std::is_floating_point<real_t<T>>::value)> {};

// Magnitude of a scalar as a plain double, for any supported datatype.
// std::abs on a complex already returns the real modulus, so this is a single
// spelling that works for int, double and complex alike.
template <class T>
inline double magnitude(const T& x) {
    return double(std::abs(x));
}

// |x|^2, computed without ever taking a square root. std::abs on a complex
// evaluates sqrt(re*re + im*im), so squaring that result computes a square root
// only to undo it — pure waste in the Frobenius norm, which wants the sum of
// squared magnitudes. std::norm is exactly re*re + im*im.
template <class T>
inline double magnitudeSq(const T& x) {
    if constexpr (is_complex<T>::value)
        return double(std::norm(x));
    else {
        const double d = double(x);
        return d * d;
    }
}

// Largest absolute component: |x| for a real, max(|re|, |im|) for a complex.
// Used to decide whether squaring is safe BEFORE any squaring happens — testing
// the square would be too late, since that is the operation that overflows.
// Costs a bit-mask and a compare; unlike std::abs on a complex it takes no
// square root.
template <class T>
inline double maxComponent(const T& x) {
    if constexpr (is_complex<T>::value) {
        const double a = std::fabs(double(x.real())), b = std::fabs(double(x.imag()));
        return a > b ? a : b;
    } else {
        return std::fabs(double(x));
    }
}

// Pairwise summation, following the algorithm NumPy uses for every one of its
// sum/mean reductions (pairwise_sum_@TYPE@ in
// numpy/_core/src/umath/loops.c.src). Two problems with the obvious `for (i)
// acc += a[i]` loop, and this fixes both:
//
//   Speed.    A single accumulator makes every add depend on the one before it,
//             so the loop runs at the LATENCY of a floating-point add (~4
//             cycles) rather than its throughput, and cannot vectorise at all.
//             The eight independent partial sums in the 128-element base case
//             keep both FMA ports busy and give the vectoriser something to
//             widen.
//   Accuracy. Sequential summation accumulates rounding error as O(n·eps); the
//             recursive halving here makes it O(log n · eps), because no
//             partial sum ever grows large relative to the terms still being
//             added to it.
//
// The block size (128) and the unroll (8) are NumPy's; the recursion trims the
// split point to a multiple of 8 so the base case always sees whole groups.
template <class T>
inline T pairwiseSum(const T* MATRIXCPP_RESTRICT a, long n) {
    if (n < 8) {
        T r = T(0);
        for (long i = 0; i < n; i++)
            r += a[i];
        return r;
    }
    if (n <= 128) {
        T r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        long i = 8;
        for (; i + 7 < n; i += 8) {
            r0 += a[i];
            r1 += a[i + 1];
            r2 += a[i + 2];
            r3 += a[i + 3];
            r4 += a[i + 4];
            r5 += a[i + 5];
            r6 += a[i + 6];
            r7 += a[i + 7];
        }
        for (; i < n; i++)
            r0 += a[i];
        return ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    }
    // Recurse for scalar element types only. Measured on 4M elements:
    //
    //                        double    complex<double>
    //     4 accumulators     0.642 ms      0.940 ms
    //     8 accumulators     ~0.40 ms      0.899 ms
    //     recursive pairwise  0.404 ms     2.895 ms
    //
    // For double the split is a clear win. For complex it is a 3x LOSS, and
    // stays a loss at every base-case size tried up to 16384, so the recursion
    // is skipped entirely there and the flat eight-accumulator pass above
    // handles the whole array. The likely cause is that a complex add is not a
    // single vector instruction, so the block boundaries interrupt a loop that
    // was already only just keeping up with L3 bandwidth (a 4M complex array is
    // 64 MB, exactly this machine's L3). Accuracy for complex therefore stays at
    // the O(n·eps) of the previous four-accumulator version rather than
    // improving to O(log n · eps) — worth knowing, and the reason this is a
    // deliberate branch and not an oversight.
    if constexpr (is_complex<T>::value) {
        T r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        long i = 8;
        for (; i + 7 < n; i += 8) {
            r0 += a[i];
            r1 += a[i + 1];
            r2 += a[i + 2];
            r3 += a[i + 3];
            r4 += a[i + 4];
            r5 += a[i + 5];
            r6 += a[i + 6];
            r7 += a[i + 7];
        }
        for (; i < n; i++)
            r0 += a[i];
        return ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    } else {
        long half = n / 2;
        half -= half % 8;
        return pairwiseSum(a, half) + pairwiseSum(a + half, n - half);
    }
}

// Sum of squares, with the same pairwise structure and for the same two reasons
// (eight independent chains, O(log n · eps) error growth). Kept separate rather
// than expressed as pairwiseSum of a squared range so that no temporary buffer
// is needed: every caller wants this over a vector it already has.
inline double pairwiseSum_sq(const double* MATRIXCPP_RESTRICT a, long n) {
    if (n < 8) {
        double r = 0.0;
        for (long i = 0; i < n; i++)
            r += a[i] * a[i];
        return r;
    }
    if (n <= 128) {
        double r0 = a[0] * a[0], r1 = a[1] * a[1], r2 = a[2] * a[2], r3 = a[3] * a[3],
               r4 = a[4] * a[4], r5 = a[5] * a[5], r6 = a[6] * a[6], r7 = a[7] * a[7];
        long i = 8;
        for (; i + 7 < n; i += 8) {
            r0 += a[i] * a[i];
            r1 += a[i + 1] * a[i + 1];
            r2 += a[i + 2] * a[i + 2];
            r3 += a[i + 3] * a[i + 3];
            r4 += a[i + 4] * a[i + 4];
            r5 += a[i + 5] * a[i + 5];
            r6 += a[i + 6] * a[i + 6];
            r7 += a[i + 7] * a[i + 7];
        }
        for (; i < n; i++)
            r0 += a[i] * a[i];
        return ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    }
    long half = n / 2;
    half -= half % 8;
    return pairwiseSum_sq(a, half) + pairwiseSum_sq(a + half, n - half);
}

// Sum of squared MAGNITUDES — real whatever the element type is, which is what
// every norm and every Householder length needs. For a real array this is
// exactly pairwiseSum_sq; for a complex one it is the only correct reading,
// since summing z*z rather than |z|^2 would give a complex "length".
template <class T>
inline double sumSq(const T* MATRIXCPP_RESTRICT a, long n) {
    if constexpr (std::is_same<T, double>::value) {
        return pairwiseSum_sq(a, n);
    } else {
        double s0 = 0.0, s1 = 0.0;
        long i = 0;
        for (; i + 1 < n; i += 2) {
            s0 += magnitudeSq(a[i]);
            s1 += magnitudeSq(a[i + 1]);
        }
        for (; i < n; i++) s0 += magnitudeSq(a[i]);
        return s0 + s1;
    }
}
struct all_t {};
inline constexpr all_t all;

// Element-wise operator sugar. MATLAB writes .* and ./ ; C++ cannot define an
// operator with a leading dot, but it CAN read the dot as an operand:
//
//     A *dot* B      is  A .* B      (same as A % B, and as A.mul(B))
//     A /dot/ B      is  A ./ B      (same as A.div(B))
//
// `A *dot` yields a proxy holding a pointer to A, which the second * or /
// consumes immediately. * and / share one precedence level and associate
// left-to-right, so `A /dot/ B` parses as `(A / dot) / B` — the intended
// grouping — and mixes correctly with surrounding + and -.
//
// The proxy never outlives the full expression it appears in, so a temporary on
// the left (`(A + B) /dot/ C`) is still alive when it is used. It does mean the
// sugar always allocates, where `std::move(t).div(C)` would reuse t's buffer;
// use the named form in a hot loop if that matters.
struct dot_t {};
inline constexpr dot_t dot;

// Selects which norm norm()/cond() compute.
//   Fro — Frobenius, sqrt of the sum of squares of every element
//   One — maximum absolute column sum
//   Inf — maximum absolute row sum
//   Two — spectral norm, the largest singular value (requires svd())
// ── Which axis a reduction, scan or ordering runs along ──────────────────────
//
// Every member that takes an axis takes one of these, and the constant names
// the axis you get ONE RESULT PER:
//
//     A.sum(ROW)   one sum per row      -> an m x 1 column
//     A.sum(COL)   one sum per column   -> a 1 x n row
//
// They are plain bools, so the older spelling still compiles — sum(false) is
// sum(COL) and sum(true) is sum(ROW) — but nothing in this header spells it
// that way any more, because `sort(true)` tells the reader nothing and
// `sort(ROW)` tells them everything.
//
// constexpr rather than #define, so they respect scope and cannot be
// accidentally stringified or redefined by a downstream macro.
inline constexpr bool COL = false;
inline constexpr bool ROW = true;

enum class NormType { Fro, One, Inf, Two };

// How much of Q a QR factorisation should build.
//
//   Complete — Q is m x m and orthogonal/unitary in the full sense. MATLAB's
//              [Q,R] = qr(A), and the default here for that reason.
//   Reduced  — Q is m x k and R is k x n with k = min(m,n): the ECONOMY form,
//              MATLAB's qr(A,0) and NumPy's default mode='reduced'.
//
// For a tall matrix the difference is enormous and it is entirely in Q. On a
// 2000x100 the complete Q is 2000x2000 = 32 MB and takes 98% of the whole
// factorisation; the reduced Q is 2000x100 = 1.6 MB. The two agree exactly on
// the columns they share — Q_reduced IS the first k columns of Q_complete —
// so the choice is purely about how much of it you need.
enum class QRMode { Complete, Reduced };

// Whether a QR factorisation pivots its columns.
//
//   On  — column-pivoted Householder, LAPACK's dgeqp3. RANK-REVEALING: |R(i,i)|
//         comes out non-increasing, which is what rank() and the least-squares
//         path read. Cannot be blocked, because the column norms have to be
//         downdated before the next pivot can be chosen.
//   Off — plain Householder, LAPACK's dgeqrf, and MATLAB's two-output qr(A).
//         BLOCKED: b reflectors are accumulated into a compact-WY form
//         (I - V*T*V^H) so the trailing update becomes a matrix multiply
//         instead of a rank-1 update per column — BLAS level 3 rather than 2.
//         P comes back as the identity.
//
// LAPACK ships these as two routines and so does Eigen (HouseholderQR is
// blocked, ColPivHouseholderQR is not), for exactly this reason.
enum class QRPivot { On, Off };



// Controls how the Taylor-series matrix functions evaluate their series.
//
// exp, sin, cos, sinh and cosh (and tan/tanh, which are built on them) each
// take an optional trailing argument of this type. It converts implicitly from
// a plain integer, so every one of these call forms works:
//
//     exp(A)                     // library defaults
//     exp(A, 25)                 // stop after at most 25 terms
//     exp(A, 25, 1e-12)          // ... or earlier, once a term falls below
//     1e-12 exp(A, 25, 1e-12, false)   // ... and skip scaling-and-squaring
//     entirely
//
//   maxTerms — hard cap on series terms. The series normally exits well before
//              this on the tolerance test; set it to pin an exact term count.
//   tol      — relative size at which a term is deemed negligible and the sum
//              stops. Negative selects machine epsilon (the accurate default).
//              Set tol = 0 to force exactly maxTerms terms with no early exit.
//   scaling  — when true the argument is halved until its norm is below the
//              series' comfortable radius and the result is recovered by
//              repeated squaring (exp) or double-angle identities (trig).
//              This is what keeps the series accurate for large ||A||;
//              turning it off is a study aid, not a faster path.
struct TaylorOpts {
    long maxTerms = taylor_limit;
    double tol = -1.0;
    bool scaling = true;

    TaylorOpts() = default;
    TaylorOpts(long n) : maxTerms(n) {}
    TaylorOpts(long n, double t) : maxTerms(n), tol(t) {}
    TaylorOpts(long n, double t, bool s) : maxTerms(n), tol(t), scaling(s) {}

    // Resolved tolerance: negative means "use machine epsilon".
    double epsTol() const { return tol < 0.0 ? std::numeric_limits<double>::epsilon() : tol; }
};
// for std=C++11
// constexpr all_t all;
//  Declared here so Matrix::factorize() can name it; defined after the class,
//  which cannot happen sooner because Decomposition holds Matrix members.
