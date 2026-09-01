// #include <initializer_list>
// #include <utility>
#include <algorithm>  // std::min
#include <cmath>
#include <complex>  // std::complex, std::conj — see COMPLEX NUMBER SUPPORT below
#include <cstddef>  // std::size_t
#include <cstdint>  // std::uintptr_t
#include <iomanip>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>        // std::tuple, std::make_tuple
#include <type_traits>  // std::is_floating_point, std::enable_if, std::false_type
#include <utility>      // std::pair, std::swap
#include <vector>

#include "random.hpp"
// <version> carries the library feature-test macros. It is C++20, so it is only
// reached through __has_include (which IS C++17) — see the constants block
// below.
#if __has_include(<version>)
    #include <version>
#endif
#if defined(__cpp_lib_math_constants) && !defined(MATRIXCPP_NO_STD_NUMBERS)
    #include <numbers>  // C++20; only included when the feature macro says it exists
#endif
#if defined(__linux__) && !defined(MATRIXCPP_NO_HUGEPAGE)
    #include <sys/mman.h>  // madvise, MADV_HUGEPAGE — see adviseHuge()
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif

// Tells the compiler that two pointers cannot address the same memory. Without
// it, a loop reading through two pointers and writing through a third has to
// assume they may overlap, which blocks vectorisation.
#if defined(__GNUC__) || defined(__clang__)
    #define MATRIXCPP_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define MATRIXCPP_RESTRICT __restrict
#else
    #define MATRIXCPP_RESTRICT
#endif

// ─── Shared storage and kernels ─────────────────────────────────────────────
// Everything in here was measured into existence on Matrix (see the PERFORMANCE
// sections of the roadmap). It lives at namespace scope rather than inside the
// class so that Tensor — see Tensor.hpp — is built on exactly the same storage
// and the same multiply kernel, instead of a second copy that drifts.
namespace mstore {

    // Ask the kernel to back a large buffer with 2 MB transparent huge pages
    // instead of 4 KB ones. Lifted from NumPy's PyDataMem_NEW
    // (numpy/_core/src/multiarray/alloc.c), same 4 MB threshold, and proved by
    // toggling NumPy's own _set_madvise_hugepage:
    //     np.hstack of two 2000x2000 doubles, hugepages ON   6.5 ms / OFF 26.8 ms
    // A freshly allocated buffer is not resident until written, and the first write
    // to each page traps into the kernel: 64 MB costs 16384 faults at 4 KB, 32 at
    // 2 MB. Advisory — if the kernel has THP off, madvise fails harmlessly.
    // MATRIXCPP_NO_HUGEPAGE opts out.
    inline void adviseHuge(void* p, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE) && !defined(MATRIXCPP_NO_HUGEPAGE)
        constexpr std::size_t HUGE_MIN = std::size_t(4) << 20;
        constexpr std::size_t PAGE_SIZE = 4096;
        if (bytes < HUGE_MIN)
            return;
        const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(p);
        const std::size_t off = std::size_t((PAGE_SIZE - base % PAGE_SIZE) % PAGE_SIZE);
        if (bytes > off)
            ::madvise(reinterpret_cast<void*>(base + off), bytes - off, MADV_HUGEPAGE);
#else
        (void)p;
        (void)bytes;
#endif
    }

    // Storage that holds only bytes, with no per-element constructor call — the
    // strategy Eigen and Armadillo both use for their scalar types. `new T[n]` is
    // NOT equivalent: std::complex<double> has a user-provided default constructor,
    // so new writes a zero to every element, faulting the whole buffer in with 4 KB
    // pages before adviseHuge can apply and then costing a second pass when the
    // caller overwrites it. Complex concat: 49.8 -> 8.6 ms.
    template <class T>
    inline constexpr bool raw_storage_ok =
        std::is_trivially_copyable<T>::value && std::is_trivially_destructible<T>::value;

    template <class T>
    inline T* rawAlloc(long n) {
        const std::size_t bytes = std::size_t(n) * sizeof(T);
        T* p;
        if constexpr (raw_storage_ok<T>)
            p = static_cast<T*>(::operator new(bytes));
        else
            p = new T[n];
        adviseHuge(p, bytes);
        return p;
    }

    template <class T>
    inline void rawFree(T* p) {
        if (!p)
            return;
        if constexpr (raw_storage_ok<T>)
            ::operator delete(static_cast<void*>(p));
        else
            delete[] p;
    }

    // Below this much work (multiply-accumulate count) a product runs serially: the
    // parallel region costs more to set up than it saves.
    inline constexpr long PARALLEL_MIN_WORK = 65536;
    // Element count above which an element-wise loop is worth handing to OpenMP.
    inline constexpr long ELEMENTWISE_MIN_WORK = 32768;
    // Lower for the transcendental maps: 20-40 cycles of work per element rather
    // than ~1, so the ~360 ns region entry is repaid far sooner.
    inline constexpr long MAP_MIN_WORK = 4096;

    // Thread count for a loop limited by memory bandwidth rather than arithmetic.
    // These saturate with far fewer threads than a compute-bound loop, and past
    // that extra threads only add contention. Measured transposing a 2000x2000
    // complex matrix: 1->13.1 ms, 4->6.4, 8->5.8, 16->6.7, 32->12.1. Halving the
    // reported maximum lands on the physical core count wherever SMT is two-way.
    inline int memoryThreads() {
#ifdef _OPENMP
        const int t = omp_get_max_threads();
        return t > 1 ? t / 2 : 1;
#else
        return 1;
#endif
    }

    // Runs body(i) over [0,total), in parallel once the range is large enough.
    // Element-wise work is memory-bound and one core cannot saturate the memory
    // system: a 2000x2000 Hadamard (96 MB touched) runs at 44 GB/s on one thread
    // and 112 GB/s on sixteen. NumPy's ufuncs are SIMD but strictly
    // single-threaded, so this is a gap that is simply not available to it. body is
    // taken BY VALUE and callers pass restrict-qualified pointers captured by
    // value: by-reference capture costs about half the throughput, because the
    // compiler can no longer prove the pointers do not alias.
    template <class F>
    inline void forEachIndex(long total, F body) {
#ifdef _OPENMP
        if (total >= ELEMENTWISE_MIN_WORK) {
    #pragma omp parallel for schedule(static) num_threads(memoryThreads())
            for (long i = 0; i < total; i++)
                body(i);
            return;
        }
#endif
        for (long i = 0; i < total; i++)
            body(i);
    }

    // ── The multiply kernel. C (MxN) = A (MxK) * B (KxN), all row-major. ──
    // gemmAcc ACCUMULATES into C; gemm overwrites it. Both spellings exist because
    // of the std::linalg (P1673) lesson: an out-parameter that the caller already
    // owns removes the per-step allocation that made an 18-term Taylor loop cost
    // 3.2x its own arithmetic at n=256.
    template <class T>
    inline void gemmAcc(const T* MATRIXCPP_RESTRICT Ag,
                        const T* MATRIXCPP_RESTRICT Bg,
                        T* MATRIXCPP_RESTRICT Cg,
                        long M,
                        long N,
                        long K) {
        constexpr long BLOCK = 64;
        // Pointers hoisted and captured BY VALUE — by reference cost about half the
        // throughput at n=2048 (64 -> 175 GFLOP/s came from getting this right).
        auto tile = [=](long ii) {
            for (long kk = 0; kk < K; kk += BLOCK)
                for (long jj = 0; jj < N; jj += BLOCK)
                    for (long i = ii; i < std::min(ii + BLOCK, M); i++)
                        for (long k = kk; k < std::min(kk + BLOCK, K); k++) {
                            const T aik = Ag[i * K + k];
                            const T* MATRIXCPP_RESTRICT brow = Bg + k * N;
                            T* MATRIXCPP_RESTRICT crow = Cg + i * N;
                            const long jEnd = std::min(jj + BLOCK, N);
                            for (long j = jj; j < jEnd; j++)
                                crow[j] += aik * brow[j];
                        }
        };
        // Small products never touch the OpenMP runtime. An `if` clause on the
        // pragma is not enough — the runtime is still entered for ~360 ns even when
        // the condition is false, and a 2x2 multiply went 29 ns -> 3290 ns because
        // of it. The threshold is on M*N*K, not on n: a tall thin product and a
        // square one of the same n are different jobs.
        if (M * N * K <= PARALLEL_MIN_WORK) {
            for (long ii = 0; ii < M; ii += BLOCK)
                tile(ii);
            return;
        }
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
        for (long ii = 0; ii < M; ii += BLOCK)
            tile(ii);
    }

    template <class T>
    inline void gemm(const T* MATRIXCPP_RESTRICT Ag,
                     const T* MATRIXCPP_RESTRICT Bg,
                     T* MATRIXCPP_RESTRICT Cg,
                     long M,
                     long N,
                     long K) {
        const long total = M * N;
        for (long i = 0; i < total; i++)
            Cg[i] = T(0);
        gemmAcc(Ag, Bg, Cg, M, N, K);
    }

}  // namespace mstore

// ─── Mathematical constants ─────────────────────────────────────────────────
// Same names and same values as std::numbers, usable from C++17.
//
//     mconst::pi, mconst::e, mconst::sqrt2, ...        (double)
//     mconst::pi_v<float>, mconst::pi_v<long double>   (any floating type)
//
// WHY NOT <numbers> DIRECTLY: it is C++20. Verified on this toolchain — at
// -std=c++17 including <numbers> compiles but `std::numbers::pi` is "not
// declared", because the whole namespace sits behind an internal
// __cplusplus > 201703L guard. Depending on it would silently force every user
// of this header to C++20, which the header does not otherwise need.
//
// WHY NOT _USE_MATH_DEFINES / M_PI: M_PI is a POSIX and MSVC extension, not ISO
// C++. It happens to work on glibc without the define, does NOT on MSVC without
// it, is a macro (so it cannot be scoped, overloaded, or made a template), and
// the SHOUTY_NAMES are exactly what std::numbers was introduced to replace.
//
// SO: define them here, with the standard's own names, and simply ALIAS to
// std::numbers wherever it exists. Nothing about user code changes when the
// project moves to C++20 — the names are already the standard ones, they just
// stop being ours. Define MATRIXCPP_NO_STD_NUMBERS to force the fallback (which
// is what validate.cpp does, so that the two are checked against each other).
namespace mconst {
#if defined(__cpp_lib_math_constants) && !defined(MATRIXCPP_NO_STD_NUMBERS)
    using std::numbers::e;
    using std::numbers::e_v;
    using std::numbers::egamma;
    using std::numbers::egamma_v;
    using std::numbers::inv_pi;
    using std::numbers::inv_pi_v;
    using std::numbers::inv_sqrt3;
    using std::numbers::inv_sqrt3_v;
    using std::numbers::inv_sqrtpi;
    using std::numbers::inv_sqrtpi_v;
    using std::numbers::ln10;
    using std::numbers::ln10_v;
    using std::numbers::ln2;
    using std::numbers::ln2_v;
    using std::numbers::log10e;
    using std::numbers::log10e_v;
    using std::numbers::log2e;
    using std::numbers::log2e_v;
    using std::numbers::phi;
    using std::numbers::phi_v;
    using std::numbers::pi;
    using std::numbers::pi_v;
    using std::numbers::sqrt2;
    using std::numbers::sqrt2_v;
    using std::numbers::sqrt3;
    using std::numbers::sqrt3_v;
#else
    // Long-double literals so that pi_v<long double> is exact to its type; the
    // double specialisations then round once, the same way std::numbers does.
    template <class T>
    inline constexpr T e_v = T(2.718281828459045235360287471352662498L);
    template <class T>
    inline constexpr T log2e_v = T(1.442695040888963407359924681001892137L);
    template <class T>
    inline constexpr T log10e_v = T(0.434294481903251827651128918916605082L);
    template <class T>
    inline constexpr T pi_v = T(3.141592653589793238462643383279502884L);
    template <class T>
    inline constexpr T inv_pi_v = T(0.318309886183790671537767526745028724L);
    template <class T>
    inline constexpr T inv_sqrtpi_v = T(0.564189583547756286948079451560772586L);
    template <class T>
    inline constexpr T ln2_v = T(0.693147180559945309417232121458176568L);
    template <class T>
    inline constexpr T ln10_v = T(2.302585092994045684017991454684364208L);
    template <class T>
    inline constexpr T sqrt2_v = T(1.414213562373095048801688724209698079L);
    template <class T>
    inline constexpr T sqrt3_v = T(1.732050807568877293527446341505872367L);
    template <class T>
    inline constexpr T inv_sqrt3_v = T(0.577350269189625764509148780501957456L);
    template <class T>
    inline constexpr T egamma_v = T(0.577215664901532860606512090082402431L);
    template <class T>
    inline constexpr T phi_v = T(1.618033988749894848204586834365638118L);

    inline constexpr double e = e_v<double>;
    inline constexpr double log2e = log2e_v<double>;
    inline constexpr double log10e = log10e_v<double>;
    inline constexpr double pi = pi_v<double>;
    inline constexpr double inv_pi = inv_pi_v<double>;
    inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
    inline constexpr double ln2 = ln2_v<double>;
    inline constexpr double ln10 = ln10_v<double>;
    inline constexpr double sqrt2 = sqrt2_v<double>;
    inline constexpr double sqrt3 = sqrt3_v<double>;
    inline constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
    inline constexpr double egamma = egamma_v<double>;
    inline constexpr double phi = phi_v<double>;
#endif
}  // namespace mconst
// The namespace is NOT called `numbers`: a user with `using namespace std;` on
// C++20 would then find both ::numbers and std::numbers and the lookup would be
// ambiguous. `using namespace mconst;` if bare `pi` is wanted — note that `e`
// is a very common local variable name, and a local always wins, harmlessly.

// ─── Literals ───────────────────────────────────────────────────────────────
// The imaginary unit comes from the standard, not from this header:
//
//     using namespace matrix_literals;
//     auto z = 3.0 + 4.0i;                       // std::complex<double>(3,4)
//     Matrix<std::complex<double>> A(2,2);
//     A = {{1.0 + 2.0i, 3.0}, {0.0, 1.0i}};
//
// std::complex_literals has been in <complex> since C++14 and gives
// operator""i,
// ""if and ""il for complex<double>, <float> and <long double>. It is preferred
// over a global `inline constexpr std::complex<double> i(0,1)` for one concrete
// reason: nearly every loop in this header (and in any matrix code) uses `i` as
// a counter, and a local declaration shadows a global one. That still compiles
// — the loop variable simply wins — but it makes the imaginary unit unusable
// inside almost every function you would want to write. A literal suffix cannot
// be shadowed by a variable, so it has no such failure mode.
//
// matrix_literals re-exports it so a single using-directive brings in the
// complex literals together with anything this library adds later.
namespace matrix_literals {
    using namespace std::complex_literals;
}

// ─── Global tolerances ──────────────────────────────────────────────────────
// Default cap on the number of terms the Taylor-series matrix functions
// (exp, sin, cos, sinh, cosh) will evaluate. It is only a cap: every one of
// those functions scales its argument first and then stops as soon as the
// terms stop contributing, which in practice happens after 15-25 terms. Raise
// it only if you deliberately disable scaling — see TaylorOpts below.
const static long taylor_limit = 300;

// ─── Scalar type traits ─────────────────────────────────────────────────────
// is_complex<T>  — true only for std::complex<U>
// real_of<T>     — the underlying real type: double for complex<double>, T
// otherwise These let one body serve both real and complex datatypes via `if
// constexpr`, which is what makes conj()/real()/imag()/H() below degrade
// gracefully instead of failing to compile on Matrix<double>.
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

// ─────────────────────────────────────────────────────────────────────────────
// ROADMAP — where the project is at
//
// Naming convention already in use (keep it):
//   A.f()   member function  → ELEMENT-WISE   (A.pow(2), A.ln(), A.exp())
//   f(A)    free function    → MATRIX-WISE    (pow(A,2), log(A,base))
//   Operators are element-wise EXCEPT operator*(Matrix), which is matmul.
//   operator() is indexing/slicing, not arithmetic.
//
// Everything claimed below is checked by validate.cpp — build and run it before
// trusting any of it:
//     g++ -std=c++17 -O2 -fopenmp -o validate validate.cpp && ./validate
//
// DONE
//   [x] solve, norm, rank            — solve covers square (LU) and
//   over-determined
//                                       (least squares via column-pivoted QR);
//                                       inverse() is now solve(I), one copy of
//                                       the substitution code rather than two
//   [x] cholesky                     — SPD, with the failed-pivot case doubling
//   as
//                                       the positive-definiteness test
//   [x] svd, pinv, cond              — svd is one-sided Jacobi, chosen over
//                                       bidiagonalise-then-QR for its relative
//                                       accuracy on small singular values,
//                                       which is what cond() and pinv()
//                                       actually depend on
//   [x] diag, triu, tril, reshape    — plus the free diag(v) that goes the
//   other way [x] min/max/mean/var/stddev/argmin/argmax  — scalar and axis
//   forms, mirroring sum(bool) [x] ==, !=, allclose, unary -    — and
//   operator+=; operator+/-/| are now const,
//                                       so they work on const operands
//   [x] adjugate                     — det*inverse when non-singular, cofactor
//                                       expansion when not
//   [x] exp(A) matrix exponential    — scaling-and-squaring around a Taylor
//   series [x] sin/cos/tan/sinh/cosh/tanh   — element-wise members and
//   matrix-wise free fns.
//                                       tan/tanh are SOLVES (cos(A)·X =
//                                       sin(A)), not the element-wise division
//                                       they look like
//   [x] conj(), real(), imag(), H()  — see COMPLEX below
//   [x] Caller-supplied series limits on every Taylor-based function, via
//   TaylorOpts
//
// THE 2x2 SCHUR BLOCK — resolved, and it was worse than previously recorded.
//   The old note said eig() *read* complex-conjugate pairs wrongly. It did, but
//   schurDecomp() also never CONVERGED on them: a conjugate pair's sub-diagonal
//   entry does not go to zero (that is the definition of a real Schur form), so
//   the iteration ran to its step limit and threw. A plain rotation matrix was
//   enough. Fixed in schurDecomp() by deflating an isolated trailing 2x2 as a
//   block, splitting it with a Givens rotation when its roots turn out to be
//   real, and adding an exceptional shift for stalled blocks. Consequences:
//     - eigvals() returns every eigenvalue including complex ones, block-aware.
//     - eig() stays real-valued but now THROWS on a conjugate pair instead of
//       silently returning the real part twice.
//     - The matrix trig functions never needed this: they sum a Taylor series
//     and
//       so avoid the Schur form entirely. pow()/log() still go through it and
//       still require positive eigenvalues.
//
// COMPLEX NUMBER SUPPORT — Matrix<std::complex<double>>, via <complex> only.
// Status as measured, not guessed (see the probe in validate.cpp):
//   WORKING: constructors, =, + - * / %, matmul incl. Strassen, T(), H(),
//   conj(),
//     real(), imag(), tr(), sum(), IsDiagonal(), slicing/proxies,
//     set_Ran_values(), concat, tensor, reshape/triu/tril/diag, ==/!=/allclose,
//     unary -, norm() (all of Fro/One/Inf — std::abs already gives the complex
//     modulus), and operator<< now formats properly.
//   REFUSED AT COMPILE TIME, deliberately: min/max/argmin/argmax (complex has
//   no
//     ordering) and mean/var/stddev, svd, cholesky, and the Taylor matrix
//     functions. Each of those would otherwise compile by taking std::real() of
//     every entry and quietly discarding the imaginary part — a wrong answer
//     dressed as a working one. They static_assert with an explanation instead.
//   STILL TO DO: det, inverse, solve, LU, QR, eig, schurDecomp, pow, log. All
//   of
//     them fail on the same single construct, double(grid[k]).
//
//   [ ] 1. Replace the double(...) casts using the work_t idea.
//   real_of/is_complex
//          already exist at the top of this header; what is missing is
//            work_t = std::complex<double> when datatype is complex, else
//            double
//          plus turning the std::vector<double> scratch buffers into
//          std::vector<work_t> and widening the return types to Matrix<work_t>.
//          Real matrices are unaffected because work_t collapses to double.
//   [x] 2. conj(), real(), imag(), H() — done. H() is the one that matters: for
//          complex matrices it, not T(), is the adjoint. NOTE the corollary is
//          still outstanding — Q^T in QR, Q*T*Q^T in schurDecomp and every
//          symmetry check in this header still call .T(), and each must become
//          .H() when item 1 lands. Every one of those is a SILENT wrong answer
//          if missed; it compiles perfectly, it is just not the right matrix.
//   [ ] 3. Householder reflectors need their complex form. The real code picks
//          alpha = -sign(x_1)*||x||; the complex version is
//          alpha = -exp(i*arg(x_1))*||x||, and tau becomes complex. This is a
//          genuine algorithm change, not a type substitution. Same for the
//          Givens rotations and the Wilkinson shift inside schurDecomp.
//   [ ] 4. Pivoting is fine as written — std::abs() on a complex returns the
//          real magnitude, so the LU pivot search keeps working and keeps
//          meaning the right thing. Explicit sign tests like (W(k,k) >= 0.0) do
//          NOT survive; complex has no ordering. Those are the lines to hunt,
//          and split2x2()'s (half >= 0.0) is now one of them.
//   [x] 5. Printing — toLines() now tests is_float_like<datatype>, which is
//   true
//          for complex<double> where std::is_floating_point is false.
//   [x] 6. Complex eig() — eigvals() delivers this; see THE 2x2 SCHUR BLOCK
//   above.
//
// [x] The imaginary unit — DECIDED and implemented; see the Literals block near
//     the top of the file. std::complex_literals won over a global
//     `inline constexpr std::complex<double> i(0,1)`, for the reason weighed
//     here originally: nearly every loop in this header uses `i` as a counter,
//     and a local declaration shadows a global one. That still compiles (the
//     loop wins) but leaves the imaginary unit unusable inside almost every
//     function you would want to write matrix code in. A literal suffix cannot
//     be shadowed by a variable, so it has no such failure mode — validate.cpp
//     pins that down with a test that declares `int i = 7` and then uses 2.0i.
//     Reached through `using namespace matrix_literals;`, which re-exports
//     std::complex_literals so one using-directive covers future additions too.
//
// BUILD FLAG WARNING — this one is specific to the current compile line.
// -ffast-math enables -fcx-limited-range (verified with -Q --help=optimizers on
// this toolchain: disabled at -O3, enabled once -ffast-math is added). That
// switches complex multiply and divide to the naive textbook formulas with no
// range reduction, so complex division can overflow or underflow spuriously on
// operands that are individually well within double's range. Drop -ffast-math,
// or add -fno-cx-limited-range, before trusting any complex benchmark numbers.
//
// Also outstanding (not code):
//   [x] .gitignore — added; the committed binaries still need `git rm --cached`
//   [x] rt_mat_pow_real.txt was empty — root cause found and fixed. It was not
//   a
//       benchmark-harness problem at all: Strassen-Winograd was computing wrong
//       products (see below), so B.T()*B came back non-symmetric with negative
//       eigenvalues, pow(A,0.5) threw on the first size that used the padded
//       Strassen path (n=86), and the program aborted before flushing the file.
//   [x] README now documents the build line and the member/free convention
//   [ ] The committed rt_*.txt timings predate the Strassen fix. The fix does
//   not
//       change the operation count, so the timings should still stand, but they
//       were measured against a path that returned wrong answers — worth a
//       rerun before they are quoted anywhere.
//
// STRASSEN-WINOGRAD — was silently wrong, now correct but compiled OUT.
//   Two errors in the combination table (T4 had its operands reversed, and U7
//   used the wrong pair of intermediates) meant operator* returned incorrect
//   products for EVERY size that actually entered the recursion. It went
//   unnoticed because n=64 hits the base case and falls straight back to
//   naiveMul, so the recursion first ran for real at n=128. validate.cpp now
//   checks operator* against a reference product at every size class.
//   Once correct, it was benchmarked against the same blocked naive multiply
//   and lost at every size (0.43x at n=128, 0.11x at n=1024) — the recursion is
//   serial where naiveMul is OpenMP-parallel, and it allocates ~20 temporaries
//   per level. It is therefore behind MATRIXCPP_ENABLE_STRASSEN and off by
//   default; see the note at operator*(Matrix) for what would make it pay.
//
// PERFORMANCE WORK ALREADY DONE (all measured, all still green in validate.cpp
// and numpy_validate.py):
//   [x] NRVO — returning a local declared inside a try block suppresses the
//       named return value optimisation, adding a full allocate-zero-copy of
//       the result. 9x on Hadamard and element-wise division. The rule this
//       leaves behind: keep the try around the CHECKS, not around the result.
//   [x] Element-wise ops build into an uninitialised buffer in one pass instead
//       of copy-then-modify. 1.6-2.1x.
//   [x] schurDecomp QR step uses Givens rotations, exploiting the Hessenberg
//       structure the previous full-length Householders ignored: O(n^4) ->
//       O(n^3), and Q accumulates transposed so its updates are contiguous.
//       107x on eig, ~90x on pow(A,real) and log(A).
//   [x] QR accumulates Q transposed for the same reason. 1.5x.
//   [x] concat/kron index directly instead of through the wrapping operator(),
//       which ran an integer division per coordinate. Up to 5x.
//   [x] Frobenius norm uses |x|^2 directly rather than squaring std::abs, which
//       for a complex matrix was computing a square root only to undo it. 55x.
//   [x] sum() and the norms use four accumulators so the loop runs at add
//       throughput rather than add latency. 3.7x.
//
// IDEAS TAKEN FROM Eigen / Armadillo / uBLAS / MTL4 (all measured):
//   [x] Rule of FIVE. The class had a destructor, copy constructor and copy
//       assignment but no MOVE pair, so every `C = A + B;` deep-copied the
//       temporary operator+ had just built. Adding them, noexcept so that
//       std::vector<Matrix> actually moves on reallocation, took a 4-term
//       expression chain from 28.6ms to 7.2ms.
//   [x] Rvalue-qualified arithmetic. In A + B + C the left operand of the
//   second
//       + is the temporary the first + produced, so the && overloads write into
//       that buffer instead of allocating another. A chain of k operations now
//       allocates once, not k times. This is the cheap half of what expression
//       templates (Eigen, uBLAS, MTL4) do; the full version fuses the chain
//       into one pass, but it changes what `A + B` RETURNS, which breaks
//       template argument deduction in ordinary user code like f(A + B). Not
//       worth it here. CAVEAT: only catches temporaries on the left. A + (B +
//       C) still allocates.
//   [x] Small-buffer storage, the runtime cousin of Eigen's fixed-size types.
//       Matrices up to 4x4 live inside the object. A 2x2 A+B was 19ns of which
//       19ns was new/delete — the allocator WAS the operation.
//   [x] __restrict on the kernels. Biggest single win of the group: naiveMul
//       went from 64 to ~175 GFLOP/s at n=2048, because without it the compiler
//       must assume the result aliases the operands and refuses to vectorise.
//       Element-wise ops did NOT improve — at n=2000 they move 96MB and are
//       already at the memory roofline, so there is nothing for vectorisation
//       to recover.
//   [ ] Aligned allocation (Eigen aligns to 16/32/64). new[] gives 16 here and
//       never 32, so AVX loads are unaligned. Untested; likely small next to
//       the bandwidth limit above.
//   [ ] Register-blocked GEMM micro-kernel (Eigen/BLIS/Goto). The inner loop is
//       an axpy doing one FMA per two memory ops. Computing a 4x4 tile of C in
//       registers would reuse each loaded value four times. This is the single
//       biggest remaining item for multiply.
//   [ ] Sparse storage (PETSc/Trilinos). Out of scope for a dense library, but
//       it is what those two are actually for.
//
// A NOTE ON REF-QUALIFYING MEMBER OPERATORS, learned the hard way:
//   once ANY overload of an operator name is ref-qualified, every sibling
//   overload must be too. operator*(scalar) was qualified while
//   operator*(Matrix) was not, and for an rvalue left operand the &&-qualified
//   scalar template then beat the unqualified matrix one — silently routing
//   `Q.T() * B` into scalar multiplication. It compiled. validate.cpp caught
//   it.
//
// IDEAS TAKEN FROM WHAT NumPy AND LAPACK ACTUALLY DO (all measured).
// NumPy's speed comes from three places, and it turned out that copying them
// was mostly about STORAGE and MEMORY, not about cleverer arithmetic:
//
//   [x] madvise(MADV_HUGEPAGE) on large buffers. Lifted straight from NumPy's
//       PyDataMem_NEW (numpy/_core/src/multiarray/alloc.c), same 4 MB
//       threshold. Proved by toggling NumPy's own switch,
//       _set_madvise_hugepage:
//           np.hstack of two 2000x2000 doubles, hugepages ON   6.5 ms
//                                               hugepages OFF 26.8 ms
//           our concat, before the change                     24.8 ms
//       That is, our copy loop was ALREADY as good as NumPy's and the entire
//       4x gap was page-fault traffic. See adviseHuge().
//   [x] Constructor-free storage (also Eigen/Armadillo). `new T[n]` runs T's
//       default constructor, which for double is nothing but for
//       std::complex<double> writes a zero to every element — faulting the
//       whole buffer in before adviseHuge can apply, then paying a second full
//       pass when the caller overwrites it. Complex concat: 49.8 -> 8.6 ms.
//   [x] Pairwise summation, NumPy's reduction algorithm. Eight independent
//       accumulator chains instead of one latency-bound one, and O(log n · eps)
//       error growth instead of O(n · eps). sum(axis=1): 2.06 -> 0.36 ms.
//       Kept OFF for complex, where it measured 3x slower — see pairwiseSum.
//   [x] Column-major working arrays inside the factorisations. This is the
//       single most valuable thing LAPACK does that a row-major library gets
//       wrong for free. Every step of a Householder QR and every rotation of a
//       one-sided Jacobi SVD walks a COLUMN; in row-major storage that strides
//       by a whole row, one cache line per element, and no vectorisation.
//       Fortran has contiguous columns by construction. Transposing the working
//       copy (not the input) buys the same thing:
//           svd, n=256:  644 -> 60.6 ms       qr, n=512:  838 -> 72.3 ms
//   [x] Cached column norms in the Jacobi sweep, as dgesvj's sva[] array. Only
//       the cross term p·q has to be recomputed per pair; the two squared norms
//       update exactly through the rotation as alpha - t·gamma, beta + t·gamma.
//       Three dot products per pair become one.
//   [x] Brent-Luk round-robin pair ordering, so a sweep splits into rounds of
//       column-disjoint pairs that run in parallel. Deterministic regardless of
//       thread count. svd, n=256: 60.6 -> 19.7 ms, and 12.3 ms after the thread
//       cap below. Total 52x, and it now beats NumPy's LAPACK dgesdd.
//   [x] Parallelism where NumPy structurally cannot use it. NumPy's ufuncs are
//       SIMD but strictly SINGLE-THREADED, so every element-wise map and every
//       element-wise binary op is one core there. Threading them (mapElems,
//       forEachIndex) is a gap that is simply not available to it:
//           elem_ln 11.6 -> 0.62 ms      hadamard 2.55 -> 0.80 ms
//       Also the per-right-hand-side loop in solve(), which is what inverse()
//       is built on: 83 -> 10.7 ms at n=512.
//   [x] Index arithmetic. tr(), IsDiagonal() and cholesky() recovered (i,j)
//   with
//       / and %, or went through the wrapping operator() — up to four integer
//       divisions in the innermost loop of an O(n³) algorithm. cholesky was
//       held to 0.78 GFLOP/s by this alone: 26.8 -> 6.0 ms.
//   [x] Thread counts matched to the work. A memory-bound loop saturates with
//       about half the reported threads (physical cores, no SMT) and gets
//       SLOWER past that; a Jacobi sweep enters ~2000 parallel regions per
//       factorisation and wants only as many threads as leave each one ~8k
//       element-updates. See memoryThreads() and the table at svd's
//       sweepThreads.
//
// A CAVEAT ON THE BENCHMARK'S LARGEST SIZE, worth knowing before quoting it:
//   this machine has 64 MiB of L3, and a 2000x2000 double matrix is 32 MB — so
//   a one-input, one-output real op at n=2000 has a 64 MB working set that
//   fits ENTIRELY in L3, while the complex version at 128 MB does not.
//       A * scalar:  n=1000  794 GB/s | n=2000  771 GB/s | n=3000  34 GB/s
//   That cliff is why the real element-wise numbers look so much better than
//   the complex ones (2.3-23x against ~1.0x): above L3 both libraries are
//   pinned to the same DRAM roofline and the only remaining lever is thread
//   count.
//
// STILL SLOWER THAN LAPACK/NumPy, in rough order of how much is on the table:
//   [ ] eig (0.81x) and pow(A, real) (0.85x). schurDecomp is O(n^3) but
//       memory-bound at n >= 512. LAPACK's answer is the blocked multishift QR
//       of dlaqr0, which chases several bulges per pass over memory. The single
//       biggest algorithmic item left.
//   [ ] complex elem_div. libstdc++ implements complex division with Smith's
//       algorithm — branches and a range reduction that will not vectorise.
//       NumPy uses the naive formula. This is a correctness/speed trade, not an
//       oversight: -ffast-math would take our path too (via -fcx-limited-range)
//       and quietly change the answers, which is why the build flag warning
//       above exists.
//   [ ] reshape (0.95x) is a pure copy racing memcpy; there is nothing here.
//   [ ] qr is blocked in neither sense: LAPACK's dgeqrf accumulates reflectors
//       into a WY block so the trailing update is a matrix multiply. Column
//       pivoting rules that out (the norms must be downdated before the next
//       pivot is chosen), so matching dgeqrf would mean offering an unpivoted
//       path as well. We are 8.7x ahead of NumPy anyway, because NumPy's qr
//       returns the full m x m Q.
//
// ─────────────────────────────────────────────────────────────────────────────
//  GAP AGAINST BASE MATLAB  (no toolboxes, plotting ignored)
// ─────────────────────────────────────────────────────────────────────────────
//
// A NOTE ON WHAT DOES *NOT* GET AN OPERATOR.
// % is the element-wise (Hadamard) product and stays that way. Moving it to the
// Kronecker product was considered — kron() is the only element-wise-adjacent
// operation without an operator, and % would have freed the name `tensor` — and
// rejected for three reasons:
//   * Operators should go to FREQUENT, CHEAP operations. Hadamard is O(n²) and
//     everywhere; Kronecker is O(n⁴) and rare. Two 1000x1000 matrices kron to
//     10^12 elements, 8 TB. That belongs behind a name you have to type, not
//     two characters. Our own benchmark is the evidence: every other operation
//     runs to n = 512 or 2000, and this one stops at 48.
//   * % meaning Hadamard is the established C++ convention (Armadillo). Giving
//     it a different meaning here would mislead rather than merely surprise.
//   * MATLAB has no operator for it either — it is kron(A, B) there too — so an
//     operator would buy nothing in the MATLAB fidelity this header is aiming
//     at.
// The name collision that prompted the question was fixed at its source
// instead: tensor() became kron(), so an operation no longer shares a name with
// a type.
//
// OPERATOR MAPPING — settled, and the one deliberate divergence is documented.
// The organising rule is that THE MEMBER DOT MEANS ELEMENT-WISE: A.sin() is
// element-wise and sin(A) is the matrix function, A.pow(n) is element-wise and
// pow(A, n) is the matrix power. MATLAB's leading dot, put where C++ can hold
// it.
//     MATLAB    here                       note
//     A * B     A * B
//     A .* B    A.mul(B) / A % B / A*dot*B
//     kron(A,B) A.kron(B) / kron(A,B)      no operator, deliberately — see
//     above A / B     A / B                      right division; CHANGED to
//     match A ./ B    A.div(B) / A /dot/ B       C++ cannot spell a leading dot
//     A \ B     A.solve(B)                 no operator\ in C++
//     A ^ n     pow(A, n)                  ^ has the wrong precedence in C++
//     A .^ n    A.pow(n)
//     sin(A)    sin(A)                     matrix function (free)
//     sin(A) elementwise   A.sin()         element-wise (member)
//     A'        A.H()                      conjugate transpose — a member that
//     A.'       A.T()                      is NOT element-wise, but harmlessly
//                                          so: there is no element-wise T
//
// [x] TIER 1 — THE LOGICAL / MASKING LAYER. DONE. See the "Logical masks"
//     section in the class for the full API and the reasoning; in brief:
//       - a mask is a Matrix<bool>, so it is an ordinary matrix and inherits
//         shape, printing, T(), slicing and the rest for nothing;
//       - <  >  <=  >=  are ELEMENT-WISE operators, because no matrix-level
//         ordering exists for them to be confused with — the same licence that
//         lets A.T() be a non-element-wise member;
//       - ==  != stay WHOLE-MATRIX and return bool, because that meaning does
//         exist and `if (A == B)` is the idiom every C++ programmer reaches
//         for. .eq() / .ne() are the element-wise forms. This is a deliberate
//         divergence from MATLAB and the only one in the comparison family;
//       - logic in C syntax: &&, || and !, plus named land / lor / lxor / lnot.
//         NOT NumPy's & and | — `A | B` is already the augmented-matrix
//         operator, and taking & for `and` while `or` needed a named function
//         would have been lopsided, so the whole triple went to C instead. The
//         trap for NumPy habits survives — `(A>0) | (B>0)` concatenates — and
//         is written down at the definition rather than left to be discovered.
//         Overloading && and || costs short-circuiting, which an element-wise
//         or never had; and since Matrix<bool> has no conversion to bool, `if
//         (m1 || m2)` does not compile, which is a safety win. Both checked;
//       - any / all (both with the sum(bool) axis forms) / nnz / find;
//       - logical indexing both ways: A(mask) reads a column vector, and
//         A(mask) = scalar-or-vector writes through a MaskProxy.
//     Counting a mask goes through nnz(), not sum(): sum() returns datatype,
//     and for Matrix<bool> that saturates at true instead of counting. 46
//     assertions in validate.cpp, 9 more cross-checked against NumPy. [x]
//     Tensor has the same layer, same spellings — 29 more assertions. The
//         two deliberate differences: Tensor::find() returns FLAT row-major
//         indices (a tensor's positions are rank-long, so Matrix's (row, col)
//         pairs do not generalise; unravel() converts one back), and Tensor
//         uses
//         || even though it has no augmented-tensor operator to avoid —
//         matching Matrix matters more than claiming the free slot, since |
//         meaning `or` on a Tensor and `concatenate` on a Matrix would be a
//         worse trap.
//
// [x] TIER 2 — REDUCTIONS. DONE for Matrix. prod, cumsum, cumprod, diff, sort,
//     sortrows, median, mode and unique are all in, each following the same
//     addcol convention as sum(bool): 0 works DOWN columns, 1 ALONG rows, so
//     prod(false) pairs with sum(false) and cumsum(false) accumulates down the
//     same axis sum(false) totals. Scans (cumsum/cumprod/sort) keep the input
//     shape; diff shrinks the scanned axis by one; median returns double
//     because an even count averages the middle two. prod/cumsum/cumprod work
//     for complex, and everything that has to ORDER elements static_asserts
//     against it for the reason min()/max() already give. 35 assertions in
//     validate.cpp, 17 cross-checked against NumPy. STILL OPEN: Tensor has none
//     of these yet — it should get the same set, and unlike the mask layer the
//     axis handling is genuinely different there (an arbitrary axis rather than
//     a bool), so it is not a copy-paste.
//
// [x] TIER 3 — ELEMENT-WISE MATH. DONE. sign, floor, ceil, round, fix, mod,
//     rem, atan2, hypot, angle/arg, asinh/acosh/atanh, expm1 and log1p are all
//     in, all members (so all element-wise, by the rule), all through mapElems
//     so all inheriting its threading and restrict-qualified loop.
//     Three things worth knowing:
//       - mod and rem are NOT the same function. mod follows the DIVISOR's
//         sign, rem the DIVIDEND's: mod(-1,3) is 2, rem(-1,3) is -1. rem is
//         C's fmod; mod is the one you want for wrapping an index or an angle.
//       - floor and fix differ on negatives: floor(-2.5) is -3, fix(-2.5) is
//       -2.
//       - angle()/arg() closes the complex gap this list called out — real(),
//         imag() and conj() were all here but there was no way to get a phase.
//         sign() is defined for complex too, as z/|z|, so sign(z)*abs(z) == z
//         holds in both cases. The rounding family static_asserts against
//         complex, since there is no "largest integer below" a complex number.
//     29 assertions in validate.cpp, 13 cross-checked against NumPy.
//
// [x] TIER 4 — LINEAR ALGEBRA. DONE except QZ, which is called out below.
//     [x] null, orth      column subsets of V / U from svd(). Bases are not
//                         unique, so the NumPy cross-check compares the
//                         PROJECTORS they define, which are.
//     [x] roots           companion matrix + eigvals(), which is how MATLAB and
//                         NumPy both do it — the QR iteration is backward
//                         stable where deflation is not. Leading zeros are
//                         stripped and trailing zeros become exact roots at 0.
//     [x] hess            A = Q H Qᵀ. Written out rather than lifted from
//                         schurDecomp: that would have meant surgery on the
//                         routine eig, pow and log all depend on, to save 30
//                         lines. The duplication is deliberate.
//     [x] schur           public wrapper over schurDecomp, returning matrices.
//     [x] polyfit, polyval  polyfit goes through solve()'s pivoted QR rather
//                         than the normal equations, which would square the
//                         condition number. polyval is Horner.
//     [x] dot, cross      dot conjugates the LEFT operand, so A.dot(A) is
//                         ||A||_F² for complex as well as real.
//     [x] rref            Gauss-Jordan with partial pivoting. Documented as a
//                         teaching tool: on floating-point data the
//                         pivot-is-zero decision is a guess. Use rank(), null()
//                         and solve() for anything numerical.
//     [x] IsSymmetric, IsHermitian, IsUpper, IsLower, IsBanded, bandwidth.
//                         Named to match the IsDiagonal() that was already here
//                         rather than MATLAB's lowercase — a predicate family
//                         that agrees with itself beats one that is half and
//                         half. All take a tolerance relative to the entries.
//     [x] normest         power iteration on AᵀA, for when norm(A, Two)'s full
//                         SVD is a great deal of work to throw away.
//     [x] rcond, condest  the Hager-Higham 1-norm estimator (LAPACK's dlacn2 /
//                         dgecon), driven from the factors Decomposition
//                         already holds. It finds a vector that nearly
//                         maximises
//                         ||A⁻¹x||₁/||x||₁ in a handful of solves, instead of
//                         forming the inverse. Measured at n=512: 9.6 ms
//                         against 302 ms for a full cond(Two) — 31x. It is a
//                         LOWER bound on the true condition number, never
//                         pessimistic; on a 5x5 Hilbert matrix it lands on it
//                         exactly.
//     [x] lsqminnorm      minimum-norm least squares, through the SVD, so it is
//                         defined for a rank-deficient A too — which is when it
//                         is actually wanted. solve() still refuses an
//                         under-determined system, on the grounds that
//                         "infinitely many solutions" is usually a mistake
//                         worth being told about; this is the escape hatch.
//     [x] decomposition   DONE, and it was the highest value-per-line item as
//                         predicted. A.factorize() picks the factorisation from
//                         the STRUCTURE, the way MATLAB's decomposition does:
//                         symmetric positive definite -> Cholesky (half the
//                         flops, and the attempt is itself the definiteness
//                         test), square otherwise -> LU, rectangular -> pivoted
//                         QR. It owns its factors, so it outlives the matrix it
//                         came from. Measured, 100 right-hand sides:
//                             n=256   104 ms -> 3.1 ms   (33x)
//                             n=512   810 ms -> 16.3 ms  (50x)
//                         solve() and Decomposition share ONE substitution
//                         routine (luSubstitute), so the parallel-over-RHS path
//                         exists in exactly one place.
//     [x] funm            DONE, and NUMERIC not symbolic: f is any callable
//                         complex -> complex, and nothing here differentiates or
//                         expands it. The 2x2 block problem this entry used to
//                         describe is solved by converting the real Schur form
//                         to a genuinely complex triangular one first, which
//                         makes the Parlett recurrence scalar throughout and
//                         removes the need for a Sylvester solve per block.
//                         The price of staying numeric is stated rather than
//                         hidden: Higham's robust algorithm reorders into
//                         clusters and Taylor-expands each block, needing f',
//                         f''. Without those, close eigenvalues make the Parlett
//                         division meaningless — so funm DETECTS that and
//                         throws, naming the two eigenvalues and pointing at
//                         exp/log/sqrt/sin/cos/sinh/cosh/tanh/pow, none of which
//                         go through Parlett and none of which have any
//                         eigenvalue-separation requirement.
//     [x] eig(A,B)        symmetric-definite, by Cholesky reduction — the
//                         backward-stable route, and what LAPACK's dsygv does.
//                         Eigenvalues ascending, eigenvectors B-ORTHONORMAL
//                         (X^T B X = I) rather than Euclidean-normalised, which
//                         is the right normalisation here and falls out of the
//                         reduction for free.
//     [x] eigvals(A,B)    the general pencil via B^-1 A, GUARDED by the
//                         Hager-Higham condition estimate: it refuses outright
//                         when B is near-singular rather than returning
//                         plausible numbers, and names QZ as what is needed.
//     [x] polyeig         matrix polynomial eigenvalues by companion
//                         linearisation to a d*n pencil, then eigvals(A,B), so
//                         it inherits that same guard.
//     [ ] qz              STILL OPEN, and the one real gap left in this tier.
//                         A general pencil with a singular or ill-conditioned B
//                         needs the generalized Schur decomposition, with its
//                         own Hessenberg-triangular reduction and shifted
//                         sweeps. Everything above refuses that case loudly
//                         instead of guessing at it.
//
// [x] TIER 5 — CONSTRUCTION AND SHAPE. DONE. numel, repmat, fliplr, flipud,
//     rot90, circshift, blkdiag as members; linspace, logspace, range, randn,
//     randi, randperm and the structured matrices as free functions.
//     Four things worth knowing:
//       - SEQUENCES GO THROUGH std::iota, the standard library's own sequence
//         generator. The index run is exact (0,1,2,... in long), so the only
//         floating point in a linspace is the single multiply that scales it;
//         accumulating v += step instead would drift, and drift further the
//         longer the vector.
//       - linspace SETS its endpoint rather than computing it. a+i*(b-a)/(n-1)
//         does not reliably land on b, and "does linspace(0,1,101) contain
//         exactly 1.0" is a question people write loops around. NumPy and
//         MATLAB both special-case it. linspace(a,b,1) returns b, as MATLAB
//         does.
//       - EVERY RANDOM CONSTRUCTOR USES ran2() FROM random.hpp, never
//       std::rand,
//         whose low bits are poor and whose period can be 32767. Seeds stay
//         negative, matching set_Ran_values. randn is Box-Muller (two draws per
//         pair, so a seed stays reproducible), randperm is Fisher-Yates (the
//         only shuffle uniform over all n! orderings). NOT THREAD SAFE: ran2
//         keeps static state, so every filler here is deliberately serial — the
//         one place the OpenMP treatment the rest of the header gets would be
//         actively wrong.
//       - magic() implements all three of MATLAB's cases (odd by the Siamese
//         method, doubly even by a complement pattern, singly even by LUX), and
//         the tests check rows, columns, BOTH diagonals, and that 1..n² each
//         appear once, for n = 3, 4, 5, 6 and 8.
//     hilb and wilkinson now earn their keep in validate.cpp as the
//     ill-conditioning and eigenvalue stress cases they were wanted for.
//     67 assertions in validate.cpp, 15 cross-checked against NumPy/SciPy.
//
// [~] TIER 6 — IN BASE MATLAB, OUTSIDE LINEAR ALGEBRA. IN PROGRESS.
//     [x] polyval, polyfit, roots   landed with tier 4
//     [x] fft, ifft, fftshift, ifftshift
//         MATLAB semantics: a vector transforms along its own length, a matrix
//         column by column, n pads or truncates, and the whole 1/n sits on the
//         inverse. Correct for EVERY length — radix-2 where n is a power of two,
//         Bluestein's chirp-z otherwise — because the tempting shortcut of
//         zero-padding up to a power of two computes the transform of a
//         different signal, which is right for a convolution and silently wrong
//         for a spectrum. 37 assertions, 12 cross-checked against numpy.fft
//         including the Bluestein path. See the block further down for the
//         thread-count measurements and where it lands against NumPy.
//     [ ] Still to write: conv, deconv, poly, trapz, cumtrapz, gradient,
//         interp1, filter. Contracts and conventions are written out beside the
//         code, and the test spots are stubbed in numpy_validate.{cpp,py}.
//     Further out and probably not this library's job: ode45, fzero,
//     fminsearch, integral.
//
// [ ] ALSO STILL REAL-ONLY: svd, cholesky and the matrix functions
// static_assert
//     against complex. MATLAB is complex-capable throughout. Belongs on this
//     list even though it is already scheduled separately.
//
// ─────────────────────────────────────────────────────────────────────────────
//  C++26 std::linalg (P1673) — what is worth taking, measured
// ─────────────────────────────────────────────────────────────────────────────
//
// C++26 adds <linalg>: free functions over std::mdspan giving BLAS 1/2/3 —
// matrix_product, matrix_vector_product, triangular_matrix_matrix_solve, dot,
// vector_norm2, matrix_frob_norm, the rank-k updates, and so on.
//
// FIRST, THE LIMIT: it is a BLAS, NOT a LAPACK. There is no LU, QR, SVD, eig or
// Cholesky in it. Every factorisation in this header stays ours; what
// std::linalg could ever replace is naiveMul and the norms. So it is a source
// of DESIGN ideas here, not a dependency to plan around.
//
// Availability checked on this toolchain (GCC 13.3): <mdspan> and <linalg> do
// not exist at any -std setting. mdspan lands in libstdc++ 15. So none of the
// below can be adopted directly yet — but two of the three ideas can be copied
// without the header, and one turns out not to be worth copying at all.
//
// [ ] IDEA 1 — LAZY transposed() / conjugated() / scaled() VIEWS.
//     std::linalg never materialises a transpose: matrix_product(transposed(A),
//     B, C) reads A in place with swapped indices. Obviously appealing, since
//     A.T() * B here builds a whole temporary first.
//     MEASURED, AND IT DOES NOT PAY:
//         n=512    A*B 5.03 ms | A.T()*B 7.65 ms | direct Aᵀ·B kernel 8.02 ms
//         n=1024   A*B 13.9 ms | A.T()*B 17.1 ms | direct Aᵀ·B kernel 16.8 ms
//     i.e. 0.95x and 1.02x — a wash. The transpose KERNEL is only ~0.7% of
//     A.T()*B (T() alone is 0.07 ms of 7.65 ms); what makes the product slower
//     is the Aᵀ access pattern inside the GEMM, and a lazy view has exactly the
//     same access pattern. Materialising actually converts the bad pattern into
//     a good one and pays for itself. Verdict: do NOT build a lazy-view layer.
//     For the record, the three transposes inside operator/ cost 1.4% at n=512
//     and 0.5% at n=1024 — also not worth removing.
//
// [x] IDEA 2 — OUT-PARAMETER AND "UPDATING" FORMS. This is the one that pays.
//     std::linalg writes into a caller-supplied output: matrix_product(A,B,C)
//     sets C = A*B, and matrix_product(A,B,E,C) sets C = E + A*B in one pass.
//     Nothing allocates, and the accumulate is fused.
//     Every iterative routine here allocates a fresh temporary per step
//     instead. Measured on the shape of a Taylor / scaling-and-squaring loop (T
//     = T*A/k ; S = S + T, 18 terms):
//         n=128   loop 3.71 ms   vs 18 raw products 2.35 ms   -> 37% overhead
//         n=256   loop 33.5 ms   vs 18 raw products 10.6 ms   -> 68% overhead
//     At n=256 the loop costs 3.2x the arithmetic it actually performs. An
//     internal multiplyInto(A, B, C) plus a fused C += A*B would take most of
//     that back, and exp/sin/cos/sinh/cosh/log/pow all go through this shape.
//     Highest-value item to come out of reading P1673.
//
// [ ] IDEA 3 — mdspan's LAYOUT VOCABULARY for the tensor work below.
//     layout_right / layout_left / layout_stride is exactly the
//     shape-and-stride model the N-dimensional survey lands on, and layout_left
//     is precisely the column-major working array that made svd and qr 10-50x
//     faster. Matching the naming and the stride semantics costs nothing now
//     and means a Tensor<T> can later hand out a real std::mdspan — and
//     interoperate with std::linalg — without changing its storage.
//
// [ ] Also noted, not needed: std::linalg's ExecutionPolicy overloads
//     (matrix_product(std::execution::par, ...)) are the standard's version of
//     what the OpenMP paths here already do by hand.
//
// ─────────────────────────────────────────────────────────────────────────────
//  GOING N-DIMENSIONAL — a measured survey
// ─────────────────────────────────────────────────────────────────────────────
//
// MATLAB arrays are N-dimensional; Matrix is strictly rank 2. That is a real
// ceiling for the ML and quantum directions — a 5-qubit state is naturally
// 2x2x2x2x2, and a batch of images is (batch, channel, height, width). Four
// layouts were considered and three of them measured, on a
// (32, 32, 64, 64) tensor = 4.19M doubles = 33.6 MB:
//
//     operation                nested   vector<Matrix>     flat
//     allocate                11.77 ms      ~same        1.91 ms
//     element-wise add        14.22 ms     13.29 ms      3.83 ms
//     sum all                  0.83 ms      0.79 ms      0.43 ms
//     allocations                 1024         1024            1
//     contraction to a GEMM        no           no    3.10 ms @ 173 GFLOP/s
//
// STATUS: BUILT. Option C below is implemented in Tensor.hpp — include that
// instead of this header to get it. namespace mstore near the top of this file
// exists so that Tensor shares the storage, the huge-page advice, the threaded
// element-wise driver and the GEMM with Matrix rather than growing a second
// copy of each. Measured after building it, on the same tensor as the survey:
//     element-wise add  3.74 ms (survey predicted 3.83, nesting was 14.22)
//     sum all           0.37 ms (survey predicted 0.43, nesting was 0.83)
//     reshape / permute / contiguous   30-50 NANOseconds — metadata only
//     contract (1024x4096)*(4096x64)   4.39 ms @ 122 GFLOP/s
//     contractInto, same product       3.82 ms @ 141 GFLOP/s  (1.15x, no alloc)
// The out-parameter form also beats the same product through Matrix (3.82 ms
// against 5.46 ms) purely by not allocating — idea 2 from P1673, paying again.
// Cross-checked against NumPy: 14/14 operations agree (tensor_numpy_validate).
//
// Still open on Tensor, in rough order of value:
//   [ ] Broadcasting. NumPy's shape rules are not implemented; shapes must
//   match
//       exactly. This is the biggest missing convenience.
//   [ ] A strided element-wise path. Non-contiguous operands are materialised
//       first, which is one extra pass; a stride-aware loop would avoid it.
//   [ ] Single-qubit gates should not permute at all. Applying H to one qubit
//   of
//       a 22-qubit state costs 28 ms on the fast axis and 52 ms on a middle
//       one, and almost all of it is the permute-and-clone, not the 2x2
//       arithmetic. A strided kernel that walks the target axis directly would
//       remove it.
//   [ ] No small-buffer optimisation, so a 2x2 Tensor allocates where a 2x2
//       Matrix does not. Use Matrix for gate-sized objects, or add SBO.
//   [ ] Matrix is still a separate class rather than the rank-2 case of Tensor.
//       Sharing mstore was the safe 90% of that; unifying the classes would be
//       a rewrite of a 5000-line file that currently passes 236 checks.
//
// [ ] OPTION A — Matrix<Matrix<double>>, i.e. nesting the existing template.
//     It COMPILES, and element-wise addition even works, which makes it more
//     tempting than it should be. Three reasons it is the wrong answer:
//       1. operator* THROWS. naiveMul accumulates into datatype(0), and for a
//          nested element that is a 0x0 matrix, so the first `0x0 += 64x64`
//          fails. This is not a bug to fix but a structural problem: a generic
//          algorithm needs a zero of the right SHAPE, and a nested element type
//          cannot supply one without knowing the block dimensions. Every
//          algorithm here that starts from an accumulator has the same issue.
//       2. 1024 separate allocations instead of one, and 6x the allocation
//       cost.
//          Nothing is contiguous across the outer dimensions, so the huge-page
//          work, the __restrict work and the GEMM blocking all stop applying at
//          the block boundary.
//       3. 3.7x slower on element-wise work, for a layout holding the same
//       bytes.
//     Verdict: viable only for genuine BLOCK matrices where the blocks are the
//     mathematical objects (block-diagonal preconditioners, and so on) — not as
//     a general tensor.
//
// [ ] OPTION B — std::vector<Matrix<double>>, a list of rank-2 slices.
//     Measurably the same as option A (13.29 ms vs 14.22 ms): still one
//     allocation per slice, still no contiguity across the outer index, still
//     no way to hand the whole thing to a GEMM. It buys ordinary container
//     semantics and nothing else. Useful as a CONTAINER of matrices — a batch
//     of independent problems — but not as a tensor.
//
// [x] OPTION C — ONE FLAT BUFFER PLUS SHAPE AND STRIDE METADATA. This is what
//     NumPy, PyTorch and TensorFlow all do, and the measurements say the same
//     thing: 3.7x faster element-wise, 6x faster to allocate, one allocation.
//     The decisive argument is not those numbers though, it is this:
//
//       Every fast tensor contraction in every library is implemented as
//       reshape -> permute -> 2-D GEMM -> permute back.
//
//     With a flat buffer, reshape is FREE — it only rewrites the shape
//     metadata, no data moves — and the GEMM is the naiveMul that already runs
//     at 173 GFLOP/s. The (1024x4096)*(4096x64) contraction above IS that GEMM,
//     unmodified. With options A or B the reshape is impossible without first
//     copying everything into a flat buffer, at which point you have built
//     option C the slow way.
//     For quantum circuits this is exactly the operation that matters: applying
//     a k-qubit gate to an n-qubit state is reshape, permute the k target axes
//     to the front, multiply by the 2^k x 2^k gate, permute back.
//
// [ ] OPTION D — compile-time rank, Tensor<T, N> (Eigen's unsupported Tensor
//     module, xtensor). Same flat storage as C, but the rank is a template
//     parameter, so index arithmetic unrolls and shape checks happen at compile
//     time. Faster still for small ranks, at the cost of rank-generic code
//     being hard to write and error messages getting much worse. Worth
//     considering LATER as a typed layer over C's storage, not instead of it.
//
// EVEN ON A FLAT BUFFER THE AXIS MATTERS — measured on the same tensor:
//     reduce over the last axis   (contiguous)        0.98 ms
//     reduce over the first axis  (contiguous passes) 0.77 ms
//     reduce over a middle axis   (strided gather)    5.17 ms
// A 5.3x spread, which is the whole reason the permute-then-GEMM strategy earns
// its keep rather than reducing along whatever axis the caller asked for.
//
// WHAT THIS WOULD MEAN HERE, concretely:
//   - Add Tensor<T> holding { std::vector<long> shape, strides; T* data; },
//     using the SAME allocator path as Matrix so it inherits adviseHuge() and
//     the constructor-free storage.
//   - Make Matrix a rank-2 special case over that storage rather than a
//   separate
//     class, so every optimisation already measured carries over unchanged.
//   - reshape() becomes O(1) metadata for Tensor. NOTE it is a COPY today,
//   which
//     is the right behaviour for a value-semantics Matrix but would be a
//     serious performance bug in a tensor.
//   - permute()/transpose() sets strides; a materialise() forces contiguity
//   when
//     a GEMM needs it.
//   - Contraction = reshape + permute + naiveMul. No new kernel.
//   - Non-contiguous strides mean the element-wise fast paths need a
//   "contiguous?"
//     test before using the flat restrict loops, with a strided fallback. That
//     branch is the main new cost, and it is per-operation, not per-element.
// ─────────────────────────────────────────────────────────────────────────────

// Slice sentinel — used in place of all for row/column extraction.
// A dedicated type prevents ambiguity with operator()(int, int).
// Usage: A(i, all)  or  A(all, j)
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
enum class NormType { Fro, One, Inf, Two };

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
template <typename datatype>
class Decomposition;

template <typename datatype>
class Matrix {
  public:
    // --- Constructors ---

    // Default: creates an empty 0x0 matrix
    Matrix() {
        rowSize = 0;
        colSize = 0;
        grid = nullptr;
    }
    // Creates an i x j matrix, zero-initialised
    Matrix(long i, long j) {
        if (i < 0 || j < 0)
            throw std::invalid_argument("Matrix: dimensions must be non-negative, got " +
                                        std::to_string(i) + "x" + std::to_string(j));
        // Indexing operators use signed int, so both dimensions and their product
        // must fit within INT_MAX to guarantee every element is reachable.
        static constexpr long MAX_IDX = std::numeric_limits<int>::max();
        if (i > MAX_IDX || j > MAX_IDX || i * j > MAX_IDX)
            throw std::invalid_argument("Matrix: dimensions " + std::to_string(i) + "x" +
                                        std::to_string(j) + " exceed the maximum indexable size (" +
                                        std::to_string(MAX_IDX) + ")");
        rowSize = i;
        colSize = j;
        allocZero(rowSize * colSize);
    }
    // Move constructor: takes over M's buffer instead of copying it.
    //
    // Without this the class stopped at the rule of THREE, so every Matrix
    // built from a temporary — which is every arithmetic result — paid a
    // full allocate + zero-fill + element-by-element copy for data that was
    // about to be destroyed anyway. Eigen and Armadillo both rely on this;
    // it is the cheapest structural win available to a value-semantics
    // matrix type.
    //
    // noexcept matters as much as the move itself: std::vector<Matrix> only
    // moves its elements when reallocating if the move cannot throw,
    // otherwise it silently falls back to copying them.
    Matrix(Matrix&& M) noexcept {
        rowSize = M.rowSize;
        colSize = M.colSize;
        if (M.isInline()) {
            // The source's data lives inside the source object, so there is
            // no pointer to steal — copy it across. It is at most
            // SBO_CAPACITY elements, which is cheaper than an allocation.
            const long total = rowSize * colSize;
            grid = sbo;
            for (long i = 0; i < total; i++)
                grid[i] = M.grid[i];
        } else {
            grid = M.grid;
            M.grid = nullptr;  // release() tolerates null
        }
        M.rowSize = 0;
        M.colSize = 0;
    }

    // Copy constructor: deep copies M
    Matrix(const Matrix& M) {
        rowSize = M.rowSize;
        colSize = M.colSize;
        const long total = rowSize * colSize;
        allocRaw(total);  // every element is written below
        for (long index = 0; index < total; index++)
            grid[index] = M.grid[index];
    }

    // --- Assignment operators ---

    // Move assignment: steals M's buffer. This is the one that matters most
    // in practice, because `C = A + B;` on an already-existing C used to deep
    // copy the temporary that operator+ had just built.
    Matrix& operator=(Matrix&& M) noexcept {
        if (this == &M)
            return *this;
        release();
        rowSize = M.rowSize;
        colSize = M.colSize;
        if (M.isInline()) {  // nothing to steal — see the move ctor
            const long total = rowSize * colSize;
            grid = sbo;
            for (long i = 0; i < total; i++)
                grid[i] = M.grid[i];
        } else {
            grid = M.grid;
            M.grid = nullptr;
        }
        M.rowSize = 0;
        M.colSize = 0;
        return *this;
    }

    // Assigns from another Matrix (deep copy)
    Matrix& operator=(const Matrix& M) {
        if (this == &M)
            return *this;  // self-assignment guard

        release();  // empty the Matrix
        rowSize = M.rowSize;
        colSize = M.colSize;
        const long total = rowSize * colSize;
        allocRaw(total);
        for (long index = 0; index < total; index++)
            grid[index] = M.grid[index];
        return *this;
    }
    // Assigns from a 2D initializer list, e.g. A = {{1,2},{3,4}}
    Matrix& operator=(const std::initializer_list<std::initializer_list<datatype>>& M) {
        try {
            if (M.size() == 0)
                throw std::invalid_argument("Cannot assign empty initializer list to Matrix");

            release();

            rowSize = M.size();
            auto itr = M.begin();
            colSize = itr->size();

            allocZero(rowSize * colSize);  // deep copy

            int i = 0, j = 0, index = 0;
            for (auto row : M) {
                for (auto element : row) {
                    index = i * colSize + (j++ % colSize);
                    grid[index] = element;
                }
                i++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Matrix assignment error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // --- Arithmetic operators ---

    // Element-wise addition. Requires identical dimensions. Returns a new Matrix.
    // const so that it works on const operands — matrix functions such as
    // exp(const Matrix&) accumulate their series with it.
    Matrix operator+(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize)
                throw std::invalid_argument(
                    "Column size mismatch in operator+: " + std::to_string(colSize) +
                    " != " + std::to_string(M.colSize));
            if (this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Row size mismatch in operator+: " + std::to_string(rowSize) +
                    " != " + std::to_string(M.rowSize));
        } catch (const std::exception& e) {
            std::cerr << "Matrix addition error: " << e.what() << std::endl;
            throw;
        }
        // One pass into an uninitialised buffer. Copy-then-accumulate would
        // zero the result, copy into it, then read it back to add — three
        // passes over the data where one will do.
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] + b[i]; });
        return ans;
    }

    // Scalar multiplication: multiplies every element by num. Returns a new
    // Matrix.
    template <typename scalar>
    Matrix operator*(const scalar& num) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] * num; });
        return ans;
    }
    template <typename scalar>
    Matrix operator*(const scalar& num) && {
        *this *= num;
        return std::move(*this);
    }

    // In-place scalar multiplication. Written directly rather than as
    // `*this = *this * num`, which allocated a whole second matrix and
    // assigned it back over the first.
    template <typename scalar>
    Matrix& operator*=(const scalar& num) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] *= num; });
        return *this;
    }

    // Matrix multiplication (dot product). Requires this->cols == M.rows.
    // Uses Strassen-Winograd for square matrices >= STRASSEN_THRESHOLD,
    // falling back to naive O(n³) for smaller or rectangular matrices.
    // Ref-qualified purely for consistency with the scalar operator* below.
    // Once ANY overload of a name is ref-qualified, an unqualified sibling
    // stops competing on equal terms: for an rvalue left operand the
    // &&-qualified one wins outright, which silently routed `Q.T() * B` into
    // the scalar path. Matrix multiply cannot reuse either operand's buffer
    // (the result has different dimensions), so this simply forwards.
    Matrix operator*(const Matrix& M) && { return static_cast<const Matrix&>(*this) * M; }
    Matrix operator*(const Matrix& M) const& {
        try {
            if (this->colSize != M.rowSize)
                throw std::invalid_argument(
                    "Inner dimensions must match for operator*: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") * (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");

                // ── Strassen-Winograd: OFF by default ──────────────────────
                //
                // Measured against the same blocked naive multiply this file
                // already contains, Strassen is SLOWER at every size tested:
                //
                //     n      naive / strassen
                //     100        0.23x
                //     128        0.43x
                //     256        0.42x
                //     512        0.25x
                //    1024        0.11x
                //
                // The asymptotics are real, but two implementation facts swamp
                // them here. First, naiveMul is OpenMP-parallel across every
                // core, while the Strassen recursion is serial — only its
                // base-case calls enter a parallel region, each on a block small
                // enough that the thread overhead dominates. Second, every
                // recursion level heap-allocates about twenty temporaries
                // (eight sub-blocks, eight sums, seven products), and that
                // allocation and copy traffic costs more than the one saved
                // multiply out of eight returns.
                //
                // Turning it back on is worth doing once the recursion itself is
                // parallelised (an OpenMP task per independent product) and the
                // temporaries come from one preallocated arena instead of the
                // heap. Until then it is a pessimisation, so it is compiled out
                // rather than silently costing 2-9x on every product.
                //
                // Define MATRIXCPP_ENABLE_STRASSEN to opt back in.
#ifdef MATRIXCPP_ENABLE_STRASSEN
            if (rowSize == colSize && M.rowSize == M.colSize && rowSize == M.rowSize &&
                rowSize >= STRASSEN_THRESHOLD) {
                long sz = nextPow2(rowSize);
                if (sz == rowSize)
                    return strassenWinograd(*this, M);
                // Only use Strassen if padding stays within 41% overhead (i.e. the
                // padded size is at most sqrt(2)*n ≈ 1.41n, keeping work < 2×).
                // Otherwise fall through to naive — the plateau jump is not worth it.
                if (sz <= rowSize * 3 / 2) {
                    Matrix Ap(sz, sz), Bp(sz, sz);
                    long total = rowSize * colSize;
                    for (long k = 0; k < total; k++) {
                        Ap.grid[k / colSize * sz + k % colSize] = grid[k];
                        Bp.grid[k / M.colSize * sz + k % M.colSize] = M.grid[k];
                    }
                    Matrix Cp = strassenWinograd(Ap, Bp);
                    Matrix ans(rowSize, rowSize);
                    long ansTotal = rowSize * rowSize;
                    for (long k = 0; k < ansTotal; k++)
                        ans.grid[k] = Cp.grid[k / rowSize * sz + k % rowSize];
                    return ans;
                }
            }
#endif
            return naiveMul(*this, M);

        } catch (const std::exception& e) {
            std::cerr << "Matrix multiplication error: " << e.what() << std::endl;
            throw;
        }
    }

    // In-place matrix multiplication
    Matrix& operator*=(const Matrix& M) {
        (*this) = (*this) * M;
        return (*this);
    }

    // Hadamard (element-wise) product. Requires identical dimensions. Returns a
    // new Matrix.
    Matrix operator%(const Matrix& M) const& {
        // The try covers only the check that can throw. Building `ans` and
        // returning it from inside a try would suppress NRVO and cost a full
        // extra allocate-zero-copy of the result — 9x on a 2000x2000 product.
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in Hadamard product: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Hadamard product error: " << e.what() << std::endl;
            throw;
        }
        // uninit_t, and restrict-qualified locals — the same two points as
        // operator+ and operator-, both of which this loop was missing.
        // Zero-filling a result whose every element is written on the next
        // line costs a whole extra pass over the output: the product moves
        // 3n² elements, so the wasted n² was a third of the operation. That
        // was the entire gap against NumPy here.
        Matrix<datatype> ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] * b[i]; });
        return ans;
    }
    Matrix operator%(const Matrix& M) && {
        *this %= M;
        return std::move(*this);
    }

    // Integer modulo: applies modulo to every element. Returns a new Matrix.
    // e.g. A % 3 gives a matrix where each element is a_ij % 3.
    // Note: n % A has no defined meaning and is not supported.
    Matrix operator%(const int& modulo) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        for (long i = 0; i < total; i++)
            r[i] = a[i] % modulo;
        return ans;
    }
    // Qualified to match operator%(const Matrix&); reuses the temporary.
    Matrix operator%(const int& modulo) && {
        *this %= modulo;
        return std::move(*this);
    }

    // In-place integer modulo
    Matrix& operator%=(const int& modulo) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        for (long i = 0; i < total; i++)
            r[i] %= modulo;
        return *this;
    }

    // In-place Hadamard product (Element-wise Matrix multiplication)
    Matrix& operator%=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in Hadamard product: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Hadamard product error: " << e.what() << std::endl;
            throw;
        }
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] *= b[i]; });
        return *this;
    }

    // ── Matrix right division, MATLAB's mrdivide. A / B is the X solving
    //    X * B = A, i.e. A * inv(B) — WITHOUT ever forming inv(B).
    //
    // This used to be element-wise division. It was changed to match MATLAB,
    // because `A / B` meaning two entirely different things in two systems
    // that otherwise line up is the kind of difference that produces a wrong
    // answer rather than an error. Element-wise division is now div() — see
    // below, and the operator table in the header comment.
    //
    // Implemented as a solve, not as A * B.inverse(): fewer flops, and better
    // conditioned. Right division is left division on the transposes,
    //     X * B = A   <=>   Bᵀ * Xᵀ = Aᵀ   <=>   X = (Bᵀ \ Aᵀ)ᵀ
    // so it inherits everything solve() already does — LU with partial
    // pivoting when B is square, least squares via column-pivoted QR when it
    // is not, which is also what MATLAB's / does.
    //
    // Returns Matrix<double> because a solve does; see solve().
    Matrix<double> operator/(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize)
                throw std::invalid_argument(
                    "Dimension mismatch in matrix right division X*B=A: A is (" +
                    std::to_string(rowSize) + "x" + std::to_string(colSize) + ") and B is (" +
                    std::to_string(M.rowSize) + "x" + std::to_string(M.colSize) +
                    ") — they must agree in COLUMNS. "
                    "For element-wise division use A.div(B) or A /dot/ B");
            return M.T().solve(this->T()).T();
        } catch (const std::exception& e) {
            std::cerr << "Matrix right division error: " << e.what() << '\n';
            throw;
        }
    }

    // No buffer to reuse — the result of a solve is a fresh matrix of a
    // different shape in general — so the rvalue form just forwards. It still
    // has to exist: once one overload of an operator name is ref-qualified,
    // every sibling must be, or overload resolution silently picks the wrong
    // one for rvalue operands. See the note in the header comment.
    Matrix<double> operator/(const Matrix& M) && { return static_cast<const Matrix&>(*this) / M; }

    // A /= B is A = A / B, so it inherits the right-division meaning. Only
    // instantiable for Matrix<double>, since a solve produces doubles and
    // there is no narrowing conversion back.
    Matrix& operator/=(const Matrix& M) {
        static_assert(std::is_same<datatype, double>::value,
                      "A /= B is matrix RIGHT DIVISION and produces double results, so it\n"
                      "only applies to Matrix<double>. Write B = A / C for other element\n"
                      "types, or A = A.div(B) if you meant element-wise division.");
        *this = (*this) / M;
        return *this;
    }

    // ── Element-wise ("dot") operations ─────────────────────────────────
    // THE RULE THIS HEADER FOLLOWS, and the reason these are named the way
    // they are: the MEMBER DOT IS THE ELEMENT-WISE MARKER. A.sin() is
    // element-wise, sin(A) is the matrix function; A.pow(n) is element-wise,
    // pow(A, n) is the matrix power. That is the whole of MATLAB's leading
    // dot, moved to where C++ can actually put it.
    //
    //     MATLAB      here
    //     A .* B      A.mul(B)   or  A % B   or  A *dot* B
    //     A ./ B      A.div(B)                or  A /dot/ B
    //     A .^ n      A.pow(n)
    //     sin(A) elementwise      A.sin()
    //
    // So the names are div and mul, NOT ediv and emul: the dot has already
    // said element-wise, and an `e` prefix says it a second time. It also
    // keeps them the same shape as every other element-wise member — sin,
    // cos, exp, abs, pow are all three letters, and so are these.
    //
    // T() and H() are members that are NOT element-wise, which does bend the
    // rule. It cannot cause an ambiguity though: there is no such thing as an
    // element-wise transpose for A.T() to be mistaken for.
    //
    // The named forms are the primitives; % and the dot-sugar are spellings
    // of them. Both are ref-qualified so that a temporary on the left is
    // reused instead of reallocated, exactly like operator+ and operator-.

    // Element-wise division, MATLAB's ./
    Matrix div(const Matrix& M) const& {
        requireSameShape(M, "div");
        // uninit_t and restrict, for the reasons given on operator%.
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] / b[i]; });
        return ans;
    }

    // Rvalue form: divides in place and hands the same buffer back.
    Matrix div(const Matrix& M) && {
        requireSameShape(M, "div");
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] /= b[i]; });
        return std::move(*this);
    }

    // Element-wise multiplication, MATLAB's .* — a named spelling of operator%.
    Matrix mul(const Matrix& M) const& { return *this % M; }
    Matrix mul(const Matrix& M) && { return std::move(*this) % M; }
    // Scalar division: divides every element by n. Preserves datatype.
    // Note: n / A has no defined meaning and is not supported.
    template <typename scalar>
    Matrix operator/(const scalar& n) const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] / n; });
        return ans;
    }

    template <typename scalar>
    Matrix operator/(const scalar& n) && {
        *this /= n;
        return std::move(*this);
    }

    // In-place scalar division
    template <typename scalar>
    Matrix& operator/=(const scalar& n) {
        const long total = rowSize * colSize;
        datatype* MATRIXCPP_RESTRICT r = grid;
        forEachIndex(total, [=](long i) { r[i] /= n; });
        return *this;
    }

    // ── Rvalue-qualified arithmetic ─────────────────────────────────
    // In a chain like A + B + C, the left operand of the second + is the
    // temporary that the first + just produced. These overloads recognise
    // that and write into that temporary's buffer instead of allocating
    // another one, so a chain of k operations allocates once rather than k
    // times. Expression templates (Eigen, uBLAS, MTL4) solve the same
    // problem more completely — they fuse the whole chain into a single
    // pass — but they change what `A + B` returns, which breaks template
    // argument deduction in ordinary user code like f(A + B). These keep
    // every type exactly as it was.
    //
    // Note this only catches temporaries on the LEFT. A + (B + C) still
    // allocates for the inner sum, because there the temporary is the
    // argument, not the object. Left-to-right is how chains normally parse.
    Matrix operator+(const Matrix& M) && {
        *this += M;
        return std::move(*this);
    }

    // In-place element-wise addition. Requires identical dimensions.
    Matrix& operator+=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator+=: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
            for (long index = 0; index < colSize * rowSize; index++)
                grid[index] += M.grid[index];
        } catch (const std::exception& e) {
            std::cerr << "Matrix addition error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // Element-wise subtraction. Requires identical dimensions. Returns a new
    // Matrix. Subtracts directly rather than going via *this + M*-1: one pass
    // instead of two, no intermediate matrix, and it stays correct for unsigned
    // datatypes.
    Matrix operator-(const Matrix& M) const& {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator-: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
        } catch (const std::exception& e) {
            std::cerr << "Matrix subtraction error: " << e.what() << std::endl;
            throw;
        }
        Matrix ans(rowSize, colSize, uninit_t{});
        // restrict-qualified locals: without them the compiler must assume
        // the result may overlap the operands and cannot vectorise the loop.
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        forEachIndex(total, [=](long i) { r[i] = a[i] - b[i]; });
        return ans;
    }

    Matrix operator-(const Matrix& M) && {
        *this -= M;
        return std::move(*this);
    }

    // In-place element-wise subtraction. Requires identical dimensions.
    Matrix& operator-=(const Matrix& M) {
        try {
            if (this->colSize != M.colSize || this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "Dimension mismatch in operator-=: (" + std::to_string(rowSize) + "x" +
                    std::to_string(colSize) + ") vs (" + std::to_string(M.rowSize) + "x" +
                    std::to_string(M.colSize) + ")");
            for (long index = 0; index < colSize * rowSize; index++)
                grid[index] -= M.grid[index];
        } catch (const std::exception& e) {
            std::cerr << "Matrix subtraction error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }

    // Agrumented Matrix operator, allows for similar math notation.
    // Note, to preserve predence use with (), EX (A|B)
    Matrix operator|(const Matrix& M) const { return (*this).concat(M, 1); }

    // Unary negation: returns a new Matrix with every element negated.
    Matrix operator-() const& {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        for (long i = 0; i < total; i++)
            ans.grid[i] = -grid[i];
        return ans;
    }

    Matrix operator-() && {
        const long total = rowSize * colSize;
        for (long i = 0; i < total; i++)
            grid[i] = -grid[i];
        return std::move(*this);
    }

    // --- Comparison operators ---

    // Exact equality: same dimensions and every element compares equal.
    // For floating-point types prefer allclose() — exact == is rarely what you
    // want, because two mathematically equal results computed different ways
    // almost never agree bit-for-bit.
    bool operator==(const Matrix& M) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            return false;
        for (long index = 0; index < rowSize * colSize; index++)
            if (!(grid[index] == M.grid[index]))
                return false;
        return true;
    }

    bool operator!=(const Matrix& M) const { return !(*this == M); }

    // Approximate equality, mirroring numpy.allclose:
    //   |a_ij - b_ij| <= atol + rtol * |b_ij|  for every element.
    // Mismatched dimensions compare false rather than throwing, so it is safe
    // to use directly as a test assertion. This is what the test suite uses.
    bool allclose(const Matrix& M, double rtol = 1e-5, double atol = 1e-8) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            return false;
        for (long index = 0; index < rowSize * colSize; index++) {
            double diff = magnitude(grid[index] - M.grid[index]);
            if (!(diff <= atol + rtol * magnitude(M.grid[index])))
                return false;
        }
        return true;
    }

    // Write-through handle for logical indexing: A(mask) = x.
    //
    // The mask is stored BY VALUE, not by reference: `A(A > 0) = 0.0` builds
    // a temporary mask, and although that temporary does survive to the end
    // of the full expression, a proxy held any longer would dangle. One byte
    // per element against the datatype-sized assignment it is about to drive
    // makes the safety close to free.
    //
    // It is flattened into a vector<char> rather than held as a Matrix<bool>
    // because MaskProxy is a member of Matrix<datatype>, so a Matrix<bool>
    // member would make Matrix<bool> contain a MaskProxy containing a
    // Matrix<bool> — an infinitely recursive type. char, not bool, to keep
    // the plain-array indexing that vector<bool>'s bit-packing would take
    // away.
    class MaskProxy {
        Matrix& mat;
        std::vector<char> mask;

      public:
        MaskProxy(Matrix& m, const Matrix<bool>& k) : mat(m) {
            m.requireMaskShape(k);
            const long total = k.rows() * k.cols();
            mask.resize((std::size_t)total);
            for (long i = 0; i < total; i++)
                mask[(std::size_t)i] = k[int(i)] ? 1 : 0;
        }
        long selected() const {
            long n = 0;
            for (char c : mask)
                if (c)
                    n++;
            return n;
        }
        // A(mask) = scalar — sets every selected element.
        template <
            typename Scalar,
            typename = std::enable_if_t<!std::is_base_of<Matrix, std::decay_t<Scalar>>::value>>
        MaskProxy& operator=(const Scalar& v) {
            const long total = mat.rows() * mat.cols();
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    mat[int(i)] = datatype(v);
            return *this;
        }
        // A(mask) = column vector — one value per selected element, in the
        // same row-major order the read form produces.
        MaskProxy& operator=(const Matrix<datatype>& src) {
            const long n = selected();
            if (src.rows() * src.cols() != n)
                throw std::invalid_argument("A(mask) = src: the mask selects " + std::to_string(n) +
                                            " elements but src has " +
                                            std::to_string(src.rows() * src.cols()));
            const long total = mat.rows() * mat.cols();
            long k = 0;
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    mat[int(i)] = src[int(k++)];
            return *this;
        }
        // Read: A(mask) used in an expression.
        operator Matrix<datatype>() const {
            const long total = mat.rows() * mat.cols();
            Matrix<datatype> out(selected(), 1, uninit_t{});
            long k = 0;
            for (long i = 0; i < total; i++)
                if (mask[(std::size_t)i])
                    out[int(k++)] = mat[int(i)];
            return out;
        }
        friend std::ostream& operator<<(std::ostream& os, const MaskProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

  private:
    // Declared here rather than with the other private helpers below: a
    // member TYPE must be declared before any member declaration names it,
    // and the comparison templates just below use mask_scalar_t in their
    // template parameter lists. Member function BODIES are compiled after
    // the class is complete, which is why CmpOp can be used in them freely.
    enum class CmpOp { LT, GT, LE, GE, EQ, NE };
    enum class LogOp { AND, OR, XOR };

    // Keeps the scalar comparison templates from swallowing a Matrix
    // argument, so A > B picks the matrix overload and A > 0 the scalar one.
    template <typename Scalar>
    using mask_scalar_t = std::enable_if_t<!std::is_base_of<Matrix, std::decay_t<Scalar>>::value>;

  public:
    // ═══════════════════════════════════════════════════════════════════
    //  Logical masks — MATLAB's logical arrays
    // ═══════════════════════════════════════════════════════════════════
    //
    // A mask is a Matrix<bool>, so it is an ordinary matrix and gets shape,
    // printing, T(), slicing and everything else for free.
    //
    // WHICH COMPARISONS ARE OPERATORS, AND WHY:
    //   <  >  <=  >=   are operators and are ELEMENT-WISE.
    //   ==  !=         are operators and are WHOLE-MATRIX, returning bool.
    //   .eq() .ne()    are the element-wise forms of those two.
    //
    // That looks inconsistent for a moment and is not. The rule this header
    // follows is that the member dot marks element-wise where BOTH meanings
    // exist. For < > <= >= only one meaning exists — there is no ordering of
    // matrices for `A < B` to be mistaken for — so the operator is free to
    // take it, exactly as A.T() is free to be a non-element-wise member
    // because no element-wise transpose exists to collide with.
    //
    // For == the other meaning very much does exist, and `if (A == B)` is the
    // idiom every C++ programmer reaches for, so the operator keeps whole
    // matrix equality and the element-wise version is the member, .eq().
    // MATLAB's == is element-wise; this is a deliberate divergence, and it is
    // the one place in the comparison family where the two differ.
    //
    //     A > 0        A <= B       A.eq(B)      A.ne(0)
    //     m1 && m2     m1 || m2     m1 ^ m2      !m1
    //     A.any()      A.all()      A.nnz()      A.find()
    //     A(A > 0)                  read the selected elements
    //     A(A < 0) = 0.0            write through the mask

    // Element-wise comparisons against another matrix.
    Matrix<bool> lt(const Matrix& M) const { return compare(M, "lt", CmpOp::LT); }
    Matrix<bool> gt(const Matrix& M) const { return compare(M, "gt", CmpOp::GT); }
    Matrix<bool> le(const Matrix& M) const { return compare(M, "le", CmpOp::LE); }
    Matrix<bool> ge(const Matrix& M) const { return compare(M, "ge", CmpOp::GE); }
    Matrix<bool> eq(const Matrix& M) const { return compare(M, "eq", CmpOp::EQ); }
    Matrix<bool> ne(const Matrix& M) const { return compare(M, "ne", CmpOp::NE); }

    // Element-wise comparisons against a scalar.
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> lt(const Scalar& v) const {
        return compareScalar(v, CmpOp::LT);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> gt(const Scalar& v) const {
        return compareScalar(v, CmpOp::GT);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> le(const Scalar& v) const {
        return compareScalar(v, CmpOp::LE);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> ge(const Scalar& v) const {
        return compareScalar(v, CmpOp::GE);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> eq(const Scalar& v) const {
        return compareScalar(v, CmpOp::EQ);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> ne(const Scalar& v) const {
        return compareScalar(v, CmpOp::NE);
    }

    // The ordering operators. No matrix-level meaning exists, so these are
    // element-wise with nothing to be confused with.
    Matrix<bool> operator<(const Matrix& M) const { return lt(M); }
    Matrix<bool> operator>(const Matrix& M) const { return gt(M); }
    Matrix<bool> operator<=(const Matrix& M) const { return le(M); }
    Matrix<bool> operator>=(const Matrix& M) const { return ge(M); }

    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator<(const Scalar& v) const {
        return lt(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator>(const Scalar& v) const {
        return gt(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator<=(const Scalar& v) const {
        return le(v);
    }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Matrix<bool> operator>=(const Scalar& v) const {
        return ge(v);
    }

    // Logical combinators, spelled in C: &&, || and !. Non-zero counts as
    // true, as it does in MATLAB, so these work on a numeric matrix and not
    // only on a mask.
    //
    // WHY C's && AND || RATHER THAN NumPy's AND MATLAB's & AND |:
    // because | is not available. `A | B` in this header is the
    // AUGMENTED-MATRIX operator, (A|B), which horizontally concatenates —
    // it predates the mask layer and is worth keeping. Taking & for `and`
    // while `or` had to be a named function would have left the pair
    // lopsided, and worse, would have left `(A > 0) | (B > 0)` compiling and
    // quietly returning a mask of twice the width. Going to the full C
    // triple instead gives both halves an operator, makes neither collide,
    // and reads the way logic reads in C.
    //
    // ⚠ THAT TRAP STILL EXISTS FOR ANYONE TYPING FROM NumPy HABIT: `|` is
    // concatenation here, not or. Use ||.
    //
    // OVERLOADING && AND || COSTS SHORT-CIRCUIT EVALUATION — for a built-in
    // ||, the right operand is skipped when the left already decides the
    // answer; for an overloaded one both are always evaluated, and the
    // sequencing guarantee goes with it. That loss is real and here it is
    // free: an element-wise or has to look at every element of both operands
    // regardless, so there was never anything to skip. Note also that the
    // result is a Matrix<bool>, which has no conversion to bool, so
    // `if (m1 || m2)` does not compile — you have to say which you meant,
    // .any() or .all(). Scalar conditions like `if (A.any() || B.any())` are
    // plain bools and short-circuit normally.
    Matrix<bool> operator&&(const Matrix& M) const { return land(M); }
    Matrix<bool> operator||(const Matrix& M) const { return lor(M); }

    // ^ IS exclusive-or in C, so this is the language's own spelling, not a
    // borrowed one. Its famously low precedence is an argument against ever
    // using ^ for a POWER — `A * B ^ 2` would silently group as `(A*B) ^ 2`
    // — which is exactly why the matrix power here is the function pow(A, n)
    // and never an operator. For exclusive-or the precedence is harmless and
    // in fact convenient: the relational operators bind TIGHTER than ^, so
    //     A > 0 ^ A > 3     groups as     (A > 0) ^ (A > 3)
    // which is what anyone writing it would mean. Checked in validate.cpp.
    // GCC nevertheless emits -Wparentheses for the unparenthesised form,
    // because a bare comparison beside ^ is a classic bug in bitwise code —
    // so parenthesise in real code even though the grouping is already
    // right. The suggestion is about legibility, not correctness.
    Matrix<bool> operator^(const Matrix& M) const { return lxor(M); }

    // Named forms, uniform across the whole family, and deliberately kept
    // alongside the operators rather than instead of them: && || ^ ! read
    // naturally to anyone arriving from C or C++, and land/lor/lxor/lnot
    // read naturally to anyone arriving from Python, MATLAB or Fortran.
    // Both spellings are the same call.
    Matrix<bool> land(const Matrix& M) const { return logical(M, "land", LogOp::AND); }
    Matrix<bool> lor(const Matrix& M) const { return logical(M, "lor", LogOp::OR); }
    Matrix<bool> lxor(const Matrix& M) const { return logical(M, "lxor", LogOp::XOR); }
    Matrix<bool> lnot() const { return !(*this); }

    // Element-wise negation: true where the element is zero. MATLAB's ~A.
    Matrix<bool> operator!() const {
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype zero = datatype(0);
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = (a[i] == zero); });
        return out;
    }

    // ── Mask reductions ────────────────────────────────────────────────
    // Non-zero counts as true throughout, so any()/all()/nnz() read the same
    // on a mask and on the numeric matrix it came from.

    bool any() const {
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (!(grid[i] == zero))
                return true;
        return false;
    }
    bool all() const {
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (grid[i] == zero)
                return false;
        return true;
    }
    // Count of non-zero elements. This — not sum() — is how you count a mask:
    // sum() returns datatype, and for Matrix<bool> that would saturate at
    // true rather than counting.
    long nnz() const {
        long n = 0;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++)
            if (!(grid[i] == zero))
                n++;
        return n;
    }

    // Axis forms, mirroring sum(bool): addcol=0 gives a (1 x cols) row of
    // per-column results, addcol=1 a (rows x 1) column of per-row results.
    Matrix<bool> any(const bool& addcol) const { return reduceLogical(addcol, true); }
    Matrix<bool> all(const bool& addcol) const { return reduceLogical(addcol, false); }

    // Positions of the non-zero elements, as (row, column) pairs in
    // row-major order. Pairs rather than MATLAB's linear indices on purpose:
    // MATLAB's are COLUMN-major, and silently handing back numbers that mean
    // something different in the two systems is exactly the sort of thing
    // this header tries not to do.
    std::vector<std::pair<long, long>> find() const {
        std::vector<std::pair<long, long>> out;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (!(grid[i * colSize + j] == zero))
                    out.emplace_back(i, j);
        return out;
    }

    // ── Logical indexing ───────────────────────────────────────────────

    // Read: A(mask) gives a column vector of the selected elements, in
    // row-major order. (MATLAB's is column-major — same caveat as find().)
    Matrix operator()(const Matrix<bool>& mask) const {
        requireMaskShape(mask);
        const long n = mask.nnz();
        Matrix out(n, 1, uninit_t{});
        long k = 0;
        for (long i = 0; i < rowSize * colSize; i++)
            if (mask.grid[i])
                out.grid[k++] = grid[i];
        return out;
    }

    // Write: A(mask) = scalar, or A(mask) = column vector of nnz elements.
    MaskProxy operator()(const Matrix<bool>& mask) { return MaskProxy(*this, mask); }

    // --- Proxy classes for slice assignment ---
    // Returned by non-const slice operators. Holds a reference back to the
    // parent Matrix so that A(i, all) = B writes through to A.
    // Implicit Matrix<datatype> conversion lets them be used in read contexts
    // too.

    class RowProxy {
        Matrix& mat;
        int row;

      public:
        RowProxy(Matrix& m, int r) : mat(m), row(r) {}
        // Write: A(i, all) = src  — src must be a (1 x cols) row vector
        RowProxy& operator=(const Matrix<datatype>& src) {
            if (src.rows() != 1 || src.cols() != mat.cols())
                throw std::invalid_argument(
                    "RowProxy: source must be (1 x " + std::to_string(mat.cols()) + "), got (" +
                    std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
            for (long j = 0; j < mat.cols(); j++)
                mat(row, j) = src(0, j);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(1, mat.cols());
            for (long j = 0; j < mat.cols(); j++)
                ans[j] = mat(row, j);
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const RowProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    class ColProxy {
        Matrix& mat;
        int col;

      public:
        ColProxy(Matrix& m, int c) : mat(m), col(c) {}
        // Write: A(all, j) = src  — src must be a (rows x 1) column vector
        ColProxy& operator=(const Matrix<datatype>& src) {
            if (src.cols() != 1 || src.rows() != mat.rows())
                throw std::invalid_argument(
                    "ColProxy: source must be (" + std::to_string(mat.rows()) + " x 1), got (" +
                    std::to_string(src.rows()) + " x " + std::to_string(src.cols()) + ")");
            for (long i = 0; i < mat.rows(); i++)
                mat(i, col) = src(i, 0);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(mat.rows(), 1);
            for (long i = 0; i < mat.rows(); i++)
                ans[i] = mat(i, col);
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const ColProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    class SubProxy {
        Matrix& mat;
        int r1, c1, rStep, cStep, numRows, numCols;

      public:
        SubProxy(Matrix& m, int r1, int c1, int rStep, int cStep, int numRows, int numCols)
            : mat(m),
              r1(r1),
              c1(c1),
              rStep(rStep),
              cStep(cStep),
              numRows(numRows),
              numCols(numCols) {}
        // Write: A({r1,r2},{c1,c2}) = src  — src must match the slice dimensions
        SubProxy& operator=(const Matrix<datatype>& src) {
            if (src.rows() != numRows || src.cols() != numCols)
                throw std::invalid_argument("SubProxy: source is (" + std::to_string(src.rows()) +
                                            "x" + std::to_string(src.cols()) + ") but slice is (" +
                                            std::to_string(numRows) + "x" +
                                            std::to_string(numCols) + ")");
            for (int i = 0; i < numRows; i++)
                for (int j = 0; j < numCols; j++)
                    mat(r1 + i * rStep, c1 + j * cStep) = src(i, j);
            return *this;
        }
        // Read: implicit conversion to Matrix for use in expressions
        operator Matrix<datatype>() const {
            Matrix<datatype> ans(numRows, numCols);
            for (int idx = 0; idx < numRows * numCols; idx++) {
                int i = idx / numCols, j = idx % numCols;
                ans[idx] = mat(r1 + i * rStep, c1 + j * cStep);
            }
            return ans;
        }
        friend std::ostream& operator<<(std::ostream& os, const SubProxy& p) {
            return os << static_cast<Matrix<datatype>>(p).toString();
        }
    };

    // --- Indexing operators ---
    // Note: negative indices wrap backwards (e.g. -1 gives last element).
    // Matrix is indexed as A(i, j) where i = row, j = column (0-based).

    // Returns a reference to element (i, j) — supports A(i,j) = x
    datatype& operator()(const int& i, const int& j) {
        return grid[(i % rowSize) * colSize + (j % colSize)];
    }
    // Const element access
    const datatype& operator()(const int& i, const int& j) const {
        return grid[(i % rowSize) * colSize + (j % colSize)];
    }
    // Flat index access into the underlying row-major array — supports A[i] = x
    datatype& operator[](const int& i) { return grid[i % (rowSize * colSize)]; }
    // Const flat index access, so A[i] reads from a const Matrix too.
    const datatype& operator[](const int& i) const { return grid[i % (rowSize * colSize)]; }

    // Row extraction — non-const returns RowProxy: supports A(i, all) = B
    RowProxy operator()(const int& i, all_t) {
        int r = ((i % (int)rowSize) + (int)rowSize) % (int)rowSize;
        return RowProxy(*this, r);
    }
    // Row extraction — const returns Matrix by value for reading
    Matrix operator()(const int& i, all_t) const {
        Matrix<datatype> ans(1, this->colSize);
        for (long j = 0; j < this->colSize; j++)
            ans[j] = this->grid[(i % rowSize) * colSize + (j % colSize)];
        return ans;
    }

    // Column extraction — non-const returns ColProxy: supports A(all, j) = B
    ColProxy operator()(all_t, const int& i) {
        int c = ((i % (int)colSize) + (int)colSize) % (int)colSize;
        return ColProxy(*this, c);
    }
    // Column extraction — const returns Matrix by value for reading
    Matrix operator()(all_t, const int& i) const {
        Matrix<datatype> ans(rowSize, 1);
        for (long j = 0; j < rowSize; j++)
            ans[j] = grid[(j % rowSize) * colSize + (i % colSize)];
        return ans;
    }

    // Submatrix — non-const returns SubProxy: supports A({r1,r2},{c1,c2}) = B
    // Negative indices wrap; reversed ranges (e.g. {9,7}) return elements in
    // reverse order.
    SubProxy operator()(std::pair<int, int> rowRange, std::pair<int, int> colRange) {
        int rN = (int)rowSize, cN = (int)colSize;
        int r1 = ((rowRange.first % rN) + rN) % rN;
        int r2 = ((rowRange.second % rN) + rN) % rN;
        int c1 = ((colRange.first % cN) + cN) % cN;
        int c2 = ((colRange.second % cN) + cN) % cN;
        int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
        return SubProxy(*this, r1, c1, (r2 >= r1) ? 1 : -1, (c2 >= c1) ? 1 : -1, numRows, numCols);
    }
    // Submatrix — const returns Matrix by value for reading
    Matrix operator()(std::pair<int, int> rowRange, std::pair<int, int> colRange) const {
        int rN = (int)rowSize, cN = (int)colSize;
        int r1 = ((rowRange.first % rN) + rN) % rN;
        int r2 = ((rowRange.second % rN) + rN) % rN;
        int c1 = ((colRange.first % cN) + cN) % cN;
        int c2 = ((colRange.second % cN) + cN) % cN;
        int numRows = std::abs(r2 - r1) + 1, numCols = std::abs(c2 - c1) + 1;
        int rStep = (r2 >= r1) ? 1 : -1, cStep = (c2 >= c1) ? 1 : -1;
        Matrix<datatype> ans(numRows, numCols);
        for (int idx = 0; idx < numRows * numCols; idx++) {
            int i = idx / numCols, j = idx % numCols;
            ans[idx] = (*this)(r1 + i * rStep, c1 + j * cStep);
        }
        return ans;
    }
    // --- Inspection ---

    // Returns true if the matrix has no elements (0x0 or any zero dimension)
    inline bool empty() const { return rowSize * colSize == 0; }
    // Returns the number of rows
    long rows() const { return rowSize; }
    // Returns the number of columns
    long cols() const { return colSize; }

    // --- Printing and string conversion ---

    // Returns a vector of formatted row strings, one per row.
    // Used internally by toString() and printSideBySide().
    // precision: decimal places for floating-point types (ignored for integral
    // types).
    std::vector<std::string> toLines(int precision = 6) const {
        // Pre-pass: format every element to find the widest string
        std::vector<std::string> cells(rowSize * colSize);
        size_t colWidth = 0;
        for (long i = 0; i < rowSize * colSize; i++) {
            std::ostringstream oss;
            // is_float_like, not std::is_floating_point: the latter is false for
            // std::complex<double>, which would silently drop the formatting.
            if (is_float_like<datatype>::value)
                oss << std::fixed << std::setprecision(precision);
            oss << grid[i];
            cells[i] = oss.str();
            if (cells[i].size() > colWidth)
                colWidth = cells[i].size();
        }

        std::vector<std::string> lines(rowSize);
        for (long i = 0; i < rowSize; i++) {
            std::ostringstream row;
            row << "[ ";
            for (long j = 0; j < colSize; j++) {
                row << std::setw((int)colWidth) << cells[i * colSize + j];
                if (j < colSize - 1)
                    row << "  ";
            }
            row << " ]";
            lines[i] = row.str();
        }
        return lines;
    }

    // Returns the full matrix as a formatted multi-line string.
    // precision: decimal places (default 6, matching NumPy's default).
    std::string toString(int precision = 6) const {
        auto lines = toLines(precision);
        std::string result;
        for (size_t i = 0; i < lines.size(); i++) {
            result += lines[i];
            if (i + 1 < lines.size())
                result += '\n';
        }
        return result;
    }

    // Explicit cast to std::string using default precision (6dp).
    // Use toString(n) directly when a specific precision is needed.
    explicit operator std::string() const { return toString(); }

    // Prints the matrix to stdout. precision: decimal places (default 6).
    void print(int precision = 6) const { std::cout << toString(precision) << '\n'; }

    // --- Linear algebra ---

    // Returns the transpose of this matrix as a new (cols x rows) Matrix.
    // NOTE: for complex datatypes this is the PLAIN transpose, which is
    // usually not the operation you want — see H() below.
    Matrix T() const {
        Matrix<datatype> ans(colSize, rowSize, uninit_t{});
        // Blocked, because a transpose is inherently cache-hostile: one of the
        // two sides is always striding by a whole row. Walking the matrix in
        // tiles keeps both the source and destination tile resident in L1 for
        // the duration of the tile, instead of evicting a cache line per
        // element. The flat-index version this replaced also paid a division
        // and a modulo on every single element.
        constexpr long BLOCK = 32;
        const datatype* MATRIXCPP_RESTRICT src = grid;
        datatype* MATRIXCPP_RESTRICT dst = ans.grid;
        const long R = rowSize, C = colSize;
        // Each ii-strip of tiles writes a disjoint set of destination columns,
        // so the outer loop is independent and parallelises directly. A
        // transpose is pure memory traffic, and one core cannot saturate the
        // memory system on its own.
        auto strip = [=](long ii) {
            const long iMax = std::min(ii + BLOCK, R);
            for (long jj = 0; jj < C; jj += BLOCK) {
                const long jMax = std::min(jj + BLOCK, C);
                for (long i = ii; i < iMax; i++)
                    for (long j = jj; j < jMax; j++)
                        dst[j * R + i] = src[i * C + j];
            }
        };
#ifdef _OPENMP
        if (R * C >= PARALLEL_MIN_WORK) {
    #pragma omp parallel for schedule(static) num_threads(memoryThreads())
            for (long ii = 0; ii < R; ii += BLOCK)
                strip(ii);
            return ans;
        }
#endif
        for (long ii = 0; ii < R; ii += BLOCK)
            strip(ii);
        return ans;
    }

    // --- Complex support (see the COMPLEX NUMBER SUPPORT block at the top) ---
    // These four are the ones that make Matrix<std::complex<double>> usable.
    // For real datatypes they degrade gracefully: conj() and real() are the
    // identity, imag() is all zeros, and H() is exactly T().

    // Conjugate transpose (Hermitian adjoint), A^H = conj(A)^T.
    // THE important one. For complex matrices this — not T() — is what plays
    // the role the transpose plays in the real theory: it is the adjoint that
    // makes <Ax,y> = <x,A^H y>, it is what "orthogonal" becomes ("unitary",
    // Q^H Q = I), and it is what "symmetric" becomes ("Hermitian", A = A^H).
    // Every .T() inside QR, schurDecomp and the symmetry checks must become
    // .H() once complex is supported. Missing one is a silent wrong answer,
    // never a compile error.
    Matrix H() const {
        Matrix<datatype> ans(colSize, rowSize, uninit_t{});
        constexpr long BLOCK = 32;  // blocked for the same reason as T()
        for (long ii = 0; ii < rowSize; ii += BLOCK) {
            const long iMax = std::min(ii + BLOCK, rowSize);
            for (long jj = 0; jj < colSize; jj += BLOCK) {
                const long jMax = std::min(jj + BLOCK, colSize);
                for (long i = ii; i < iMax; i++)
                    for (long j = jj; j < jMax; j++) {
                        const datatype& v = grid[i * colSize + j];
                        if constexpr (is_complex<datatype>::value)
                            ans.grid[j * rowSize + i] = std::conj(v);
                        else
                            ans.grid[j * rowSize + i] = v;
                    }
            }
        }
        return ans;
    }

    // Element-wise complex conjugate. Identity for real datatypes.
    Matrix conj() const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        if constexpr (is_complex<datatype>::value)
            for (long i = 0; i < total; i++)
                ans.grid[i] = std::conj(grid[i]);
        else
            for (long i = 0; i < total; i++)
                ans.grid[i] = grid[i];
        return ans;
    }

    // Real parts of every element. The real_of trait keeps the element type
    // honest: Matrix<complex<float>> gives back Matrix<float>, and for a real
    // datatype this is the identity, returning the same type it started with.
    Matrix<real_t<datatype>> real() const {
        Matrix<real_t<datatype>> ans(rowSize, colSize);
        for (long i = 0; i < rowSize * colSize; i++) {
            if constexpr (is_complex<datatype>::value)
                ans[int(i)] = grid[i].real();
            else
                ans[int(i)] = grid[i];
        }
        return ans;
    }

    // Imaginary parts of every element. All zeros for real datatypes.
    Matrix<real_t<datatype>> imag() const {
        Matrix<real_t<datatype>> ans(rowSize, colSize);
        if constexpr (is_complex<datatype>::value)
            for (long i = 0; i < rowSize * colSize; i++)
                ans[int(i)] = grid[i].imag();
        return ans;
    }

    // element-wise power
    template <typename scalar>
    Matrix pow(scalar num) const {
        return mapElems([=](const datatype& x) { return std::pow(x, num); });
    }

    // element-wise exp
    Matrix exp() const {
        return mapElems([=](const datatype& x) { return std::exp(x); });
    }

    // element-wise log base 10
    Matrix log10() const {
        return mapElems([=](const datatype& x) { return std::log10(x); });
    }

    // element-wise log base 2, using computer science notation.
    Matrix lg() const {
        return mapElems([=](const datatype& x) { return std::log2(x); });
    }

    // element-wise natural log
    Matrix ln() const {
        return mapElems([=](const datatype& x) { return std::log(x); });
    }

    // element-wise log with arbitrary base
    template <typename scalar>
    Matrix log(scalar base) const {
        const auto invLogBase = 1.0 / std::log2(base);  // loop-invariant
        return mapElems([=](const datatype& x) { return std::log2(x) * invLogBase; });
    }

    // --- Element-wise trigonometry ---
    // Each applies std::<fn> to every element independently, exactly like
    // exp()/ln() above. These are NOT the matrix trig functions — for those
    // see the free functions sin(A)/cos(A)/... near the bottom of this header.
    // All follow the same shape: copy *this, map every element, return.

    // element-wise sine
    Matrix sin() const {
        return mapElems([=](const datatype& x) { return std::sin(x); });
    }

    // element-wise cosine
    Matrix cos() const {
        return mapElems([=](const datatype& x) { return std::cos(x); });
    }

    // element-wise tangent
    Matrix tan() const {
        return mapElems([=](const datatype& x) { return std::tan(x); });
    }

    // element-wise arcsine
    Matrix asin() const {
        return mapElems([=](const datatype& x) { return std::asin(x); });
    }

    // element-wise arccosine
    Matrix acos() const {
        return mapElems([=](const datatype& x) { return std::acos(x); });
    }

    // element-wise arctangent
    Matrix atan() const {
        return mapElems([=](const datatype& x) { return std::atan(x); });
    }

    // element-wise hyperbolic sine
    Matrix sinh() const {
        return mapElems([=](const datatype& x) { return std::sinh(x); });
    }

    // element-wise hyperbolic cosine
    Matrix cosh() const {
        return mapElems([=](const datatype& x) { return std::cosh(x); });
    }

    // element-wise hyperbolic tangent
    Matrix tanh() const {
        return mapElems([=](const datatype& x) { return std::tanh(x); });
    }

    // element-wise square root
    Matrix sqrt() const {
        return mapElems([=](const datatype& x) { return std::sqrt(x); });
    }

    // element-wise absolute value
    //  ═══════════════════════════════════════════════════════════════════
    //   Rounding, sign, phase and the two-argument functions  (tier 3)
    //  ═══════════════════════════════════════════════════════════════════
    //  All members, so all element-wise — the rule the header follows
    //  throughout. Each goes through mapElems, which means each inherits the
    //  threading and the restrict-qualified loop the transcendentals use.

    // -1, 0 or +1 for a real element. For a COMPLEX element this is z/|z|,
    // the unit vector in z's direction, which is what MATLAB's sign gives
    // and what makes sign(z)*abs(z) == z hold in both cases.
    Matrix sign() const {
        return mapElems([](const datatype& x) -> datatype {
            if constexpr (is_complex<datatype>::value) {
                const auto m = std::abs(x);
                return m == 0 ? datatype(0) : datatype(x / m);
            } else {
                return datatype(x > datatype(0) ? 1 : (x < datatype(0) ? -1 : 0));
            }
        });
    }

    // Largest integer <= x.
    Matrix floor() const {
        requireRoundable("floor");
        return mapElems([](const datatype& x) { return std::floor(x); });
    }
    // Smallest integer >= x.
    Matrix ceil() const {
        requireRoundable("ceil");
        return mapElems([](const datatype& x) { return std::ceil(x); });
    }
    // Nearest integer, halves away from zero — C's round, and MATLAB's.
    Matrix round() const {
        requireRoundable("round");
        return mapElems([](const datatype& x) { return std::round(x); });
    }
    // Truncation TOWARDS ZERO. Differs from floor for negatives:
    // floor(-2.5) is -3, fix(-2.5) is -2. MATLAB's fix, C's trunc.
    Matrix fix() const {
        requireRoundable("fix");
        return mapElems([](const datatype& x) { return std::trunc(x); });
    }

    // ── mod and rem are NOT the same function ──────────────────────────
    // They differ whenever the operands have opposite signs, and the
    // difference is which one the result follows:
    //     mod(-1, 3) ==  2     the sign of the DIVISOR
    //     rem(-1, 3) == -1     the sign of the DIVIDEND
    // rem is C's fmod. mod is the one you almost always want for wrapping
    // an index or an angle into a range. Both match MATLAB.
    template <typename Scalar>
    Matrix mod(const Scalar& y) const {
        requireRoundable("mod");
        const datatype d = datatype(y);
        return mapElems([=](const datatype& x) -> datatype {
            if (d == datatype(0))
                return x;  // MATLAB: mod(x,0) is x
            const datatype r = std::fmod(x, d);
            return (r != datatype(0) && ((r < datatype(0)) != (d < datatype(0)))) ? datatype(r + d)
                                                                                  : r;
        });
    }
    template <typename Scalar>
    Matrix rem(const Scalar& y) const {
        requireRoundable("rem");
        const datatype d = datatype(y);
        return mapElems([=](const datatype& x) -> datatype {
            return d == datatype(0) ? x : datatype(std::fmod(x, d));
        });
    }

    // Two-argument arctangent, element-wise: Y.atan2(X) is atan2(y, x),
    // the angle of the point (x, y) with the quadrant resolved — which is
    // exactly what plain atan(y/x) cannot do.
    Matrix atan2(const Matrix& X) const {
        requireRoundable("atan2");
        requireSameShape(X, "atan2");
        return zipElems(X, [](const datatype& y, const datatype& x) { return std::atan2(y, x); });
    }

    // sqrt(a² + b²) without the overflow that squaring would cause.
    Matrix hypot(const Matrix& B) const {
        requireRoundable("hypot");
        requireSameShape(B, "hypot");
        return zipElems(B, [](const datatype& a, const datatype& b) { return std::hypot(a, b); });
    }

    // Phase angle, in radians. THE missing piece for complex work: real(),
    // imag() and conj() were all here but there was no way to get an
    // argument out. Returns the real type, like real() and imag() do.
    // For a real matrix this is 0 where the element is >= 0 and pi where it
    // is negative, which is what MATLAB's angle gives.
    Matrix<real_t<datatype>> angle() const {
        // grid of another instantiation is reachable because Matrix
        // befriends every Matrix — see the friend declaration at the bottom.
        Matrix<real_t<datatype>> out(
            rowSize, colSize, typename Matrix<real_t<datatype>>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        real_t<datatype>* MATRIXCPP_RESTRICT r = out.grid;
        forEachIndex(rowSize * colSize, [=](long i) {
            if constexpr (is_complex<datatype>::value)
                r[i] = std::arg(a[i]);
            else
                r[i] = a[i] < datatype(0) ? real_t<datatype>(mconst::pi) : real_t<datatype>(0);
        });
        return out;
    }
    // arg() is the same function under the name the maths uses.
    Matrix<real_t<datatype>> arg() const { return angle(); }

    // Inverse hyperbolics, completing the set beside sinh/cosh/tanh.
    Matrix asinh() const {
        return mapElems([](const datatype& x) { return std::asinh(x); });
    }
    Matrix acosh() const {
        return mapElems([](const datatype& x) { return std::acosh(x); });
    }
    Matrix atanh() const {
        return mapElems([](const datatype& x) { return std::atanh(x); });
    }

    // exp(x) - 1 and log(1 + x), accurate for small x where the obvious
    // spelling loses every significant digit to cancellation.
    Matrix expm1() const {
        requireRoundable("expm1");
        return mapElems([](const datatype& x) { return std::expm1(x); });
    }
    Matrix log1p() const {
        requireRoundable("log1p");
        return mapElems([](const datatype& x) { return std::log1p(x); });
    }

    Matrix abs() const {
        return mapElems([=](const datatype& x) { return std::abs(x); });
    }

    // Returns whether a matrix is diagonal: all off-diagonal elements are zero.
    // Works for non-square matrices. Single flat loop — no allocations, early
    // exit.
    bool IsDiagonal() const {
        // The flat version this replaced recovered (i,j) from k with
        // k / colSize and k % colSize — two integer divisions for every
        // element of the matrix, which dominated the loop completely: the
        // scan ran at 6.5 GB/s against a machine that streams at ~45.
        // Walking rows explicitly makes the row/column comparison free and
        // splits each row into two runs the compiler can vectorise, since
        // neither contains the diagonal and so neither needs a test on j.
        const datatype* MATRIXCPP_RESTRICT g = grid;
        const datatype zero = datatype(0);
        for (long i = 0; i < rowSize; i++) {
            const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
            const long d = (i < colSize) ? i : colSize;  // diagonal, or past the end
            for (long j = 0; j < d; j++)
                if (row[j] != zero)
                    return false;
            for (long j = d + 1; j < colSize; j++)
                if (row[j] != zero)
                    return false;
        }
        return true;
    }

    // Returns the sum of the main diagonal elements. Requires a square matrix.
    datatype tr() const {
        try {
            if (!(rowSize == colSize && rowSize > 0))
                throw std::invalid_argument("tr() requires a square non-empty matrix, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            // Indexed straight into grid rather than through operator(),
            // which wraps negative indices and so runs a modulo on BOTH
            // coordinates — two integer divisions (~20-40 cycles each) per
            // element, for indices the loop bounds already prove in range.
            // The stride is colSize+1, so this is a diagonal walk with one
            // cache miss per element and nothing else to do; four
            // accumulators keep those misses in flight concurrently instead
            // of serialising on the add.
            const datatype* MATRIXCPP_RESTRICT g = grid;
            const long step = colSize + 1;
            datatype s0 = datatype(0), s1 = datatype(0), s2 = datatype(0), s3 = datatype(0);
            long i = 0;
            for (; i + 3 < rowSize; i += 4) {
                s0 += g[i * step];
                s1 += g[(i + 1) * step];
                s2 += g[(i + 2) * step];
                s3 += g[(i + 3) * step];
            }
            for (; i < rowSize; i++)
                s0 += g[i * step];
            return (s0 + s1) + (s2 + s3);
        } catch (const std::exception& e) {
            std::cerr << "Matrix trace error: " << e.what() << '\n';
            throw;
        }
    }
    // Returns the sum of all elements in the matrix.
    datatype sum() const {
        // Defers to pairwiseSum — NumPy's reduction algorithm — which is both
        // faster (eight independent accumulator chains instead of one
        // latency-bound one) and more accurate (O(log n · eps) error growth
        // instead of O(n · eps)). See its definition near the top of the file.
        return pairwiseSum(grid, rowSize * colSize);
    }
    // Dimensional sum. addcol=0: returns a (1 x cols) row matrix of column sums.
    //                  addcol=1: returns a (rows x 1) column matrix of row sums.
    Matrix sum(const bool& addcol) const {
        // Accumulates straight out of grid. The slice-based version this
        // replaced built a whole temporary Matrix per row/column, so a
        // (n x n) sum cost n allocations and n copies on top of the arithmetic.
        const datatype* MATRIXCPP_RESTRICT g = grid;
        if (!addcol) {
            // Column sums. The accumulator is the whole output row, so this
            // is already a vector operation — but only if the compiler can
            // prove the output does not alias the input. It cannot: both are
            // datatype* from the same allocator, so without __restrict every
            // store to out[j] has to be re-loaded before the next row. The
            // restrict qualifiers are what let this run at memory bandwidth.
            Matrix<datatype> total_vec(1, colSize);
            datatype* MATRIXCPP_RESTRICT out = total_vec.grid;
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
                for (long j = 0; j < colSize; j++)
                    out[j] += row[j];
            }
            return total_vec;
        }
        // Row sums. Each row is a contiguous run, so this is exactly the
        // reduction pairwiseSum exists for: the single-accumulator loop this
        // replaced was latency-bound and read 32 MB at 15 GB/s on a machine
        // that streams at 45.
        Matrix<datatype> total_vec(rowSize, 1);
        datatype* MATRIXCPP_RESTRICT out = total_vec.grid;
        for (long i = 0; i < rowSize; i++)
            out[i] = pairwiseSum(g + i * colSize, colSize);
        return total_vec;
    }

    // --- Reductions ---
    // Every reduction comes in two forms, mirroring sum() / sum(bool) above:
    //   f()      → scalar over all elements
    //   f(bool)  → addcol=0: (1 x cols) row matrix of per-column results
    //              addcol=1: (rows x 1) column matrix of per-row results
    // The axis versions can reuse the (*this)(all, i) / (*this)(i, all) slice
    // pattern that sum(bool) already uses.

    // min/max and the arg- variants order their elements with <, which
    // std::complex deliberately does not provide. Instantiating them on a
    // complex Matrix is a compile-time error rather than a silent choice of
    // some arbitrary ordering; every other reduction here works for complex.

    // Smallest element in the matrix.
    datatype min() const {
        static_assert(!is_complex<datatype>::value,
                      "min() needs an ordering; std::complex has none. Use A.abs().min().");
        requireNonEmpty("min");
        datatype best = grid[0];
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[i] < best)
                best = grid[i];
        return best;
    }

    // Per-column (addcol=0) or per-row (addcol=1) minima.
    Matrix min(const bool& addcol) const {
        static_assert(!is_complex<datatype>::value,
                      "min() needs an ordering; std::complex has none. Use A.abs().min().");
        requireNonEmpty("min");
        if (!addcol) {
            Matrix<datatype> out(1, colSize);
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype& v = grid[i * colSize + j];
                    if (v < out.grid[j])
                        out.grid[j] = v;
                }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype best = grid[i * colSize];
            for (long j = 1; j < colSize; j++)
                if (grid[i * colSize + j] < best)
                    best = grid[i * colSize + j];
            out.grid[i] = best;
        }
        return out;
    }

    // Largest element in the matrix.
    datatype max() const {
        static_assert(!is_complex<datatype>::value,
                      "max() needs an ordering; std::complex has none. Use A.abs().max().");
        requireNonEmpty("max");
        datatype best = grid[0];
        for (long i = 1; i < rowSize * colSize; i++)
            if (best < grid[i])
                best = grid[i];
        return best;
    }

    // Per-column (addcol=0) or per-row (addcol=1) maxima.
    Matrix max(const bool& addcol) const {
        static_assert(!is_complex<datatype>::value,
                      "max() needs an ordering; std::complex has none. Use A.abs().max().");
        requireNonEmpty("max");
        if (!addcol) {
            Matrix<datatype> out(1, colSize);
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype& v = grid[i * colSize + j];
                    if (out.grid[j] < v)
                        out.grid[j] = v;
                }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype best = grid[i * colSize];
            for (long j = 1; j < colSize; j++)
                if (best < grid[i * colSize + j])
                    best = grid[i * colSize + j];
            out.grid[i] = best;
        }
        return out;
    }

    // Arithmetic mean of all elements. Returns double so integral matrices
    // do not truncate — note this differs from sum(), which preserves datatype.
    double mean() const {
        static_assert(!is_complex<datatype>::value,
                      "mean(): returns double, which cannot hold a complex mean — it would "
                      "silently average only the real parts. Use A.real().mean() if that is "
                      "what you want.");
        requireNonEmpty("mean");
        double acc = 0.0;
        for (long i = 0; i < rowSize * colSize; i++)
            acc += double(std::real(grid[i]));
        return acc / double(rowSize * colSize);
    }

    // Per-column (addcol=0) or per-row (addcol=1) means.
    Matrix<double> mean(const bool& addcol) const {
        static_assert(!is_complex<datatype>::value,
                      "mean(): returns double, which cannot hold a complex mean. "
                      "Use A.real().mean(addcol).");
        requireNonEmpty("mean");
        if (!addcol) {
            Matrix<double> out(1, colSize);
            for (long i = 0; i < rowSize; i++)
                for (long j = 0; j < colSize; j++)
                    out[int(j)] += double(std::real(grid[i * colSize + j]));
            for (long j = 0; j < colSize; j++)
                out[int(j)] /= double(rowSize);
            return out;
        }
        Matrix<double> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            double acc = 0.0;
            for (long j = 0; j < colSize; j++)
                acc += double(std::real(grid[i * colSize + j]));
            out[int(i)] = acc / double(colSize);
        }
        return out;
    }

    // Variance of all elements. sample=false divides by N (population variance),
    // sample=true divides by N-1 (Bessel-corrected sample variance).
    // Two-pass: the mean first, then the squared deviations from it. The
    // one-pass E[x²]-E[x]² shortcut loses most of its significant digits when
    // the mean is large relative to the spread, so it is not used here.
    double var(bool sample = false) const {
        static_assert(!is_complex<datatype>::value,
                      "var(): defined here for real datatypes only — it would silently use "
                      "only the real parts. Use A.real().var() or A.abs().var().");
        requireNonEmpty("var");
        long N = rowSize * colSize;
        if (sample && N < 2)
            throw std::invalid_argument("var(sample=true) needs at least 2 elements, got " +
                                        std::to_string(N));
        double mu = mean(), acc = 0.0;
        for (long i = 0; i < N; i++) {
            double d = double(std::real(grid[i])) - mu;
            acc += d * d;
        }
        return acc / double(sample ? N - 1 : N);
    }

    // Standard deviation of all elements — sqrt of var(sample).
    double stddev(bool sample = false) const { return std::sqrt(var(sample)); }

    // Position {row, col} of the smallest element. Ties resolve to the first
    // encountered in row-major order.
    std::pair<long, long> argmin() const {
        static_assert(!is_complex<datatype>::value,
                      "argmin() needs an ordering; std::complex has none.");
        requireNonEmpty("argmin");
        long best = 0;
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[i] < grid[best])
                best = i;
        return {best / colSize, best % colSize};
    }

    // Position {row, col} of the largest element. Ties resolve to the first
    // encountered in row-major order.
    std::pair<long, long> argmax() const {
        static_assert(!is_complex<datatype>::value,
                      "argmax() needs an ordering; std::complex has none.");
        requireNonEmpty("argmax");
        long best = 0;
        for (long i = 1; i < rowSize * colSize; i++)
            if (grid[best] < grid[i])
                best = i;
        return {best / colSize, best % colSize};
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Hessenberg and Schur, exposed  (tier 4)
    // ═══════════════════════════════════════════════════════════════════

    // Upper Hessenberg form: A = Q * H * Qᵀ with H zero below the first
    // subdiagonal and Q orthogonal. This is the first half of every dense
    // eigenvalue algorithm — reducing to Hessenberg costs O(n³) once, after
    // which each QR sweep is O(n²) instead of O(n³).
    //
    // Written out here rather than lifted out of schurDecomp: that function
    // computes the same reduction, but pulling it apart would mean surgery
    // on the routine that eig, pow(A,real) and log(A) all depend on, for a
    // 30-line saving. The duplication is deliberate and small.
    //
    // Usage: auto [H, Q] = A.hess();
    std::pair<Matrix<double>, Matrix<double>> hess() const {
        static_assert(!is_complex<datatype>::value,
                      "hess: not yet implemented for complex datatypes — it would "
                      "compile by\n"
                      "taking std::real() of each entry and silently discard the "
                      "imaginary part.");
        if (rowSize != colSize)
            throw std::invalid_argument("hess: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        const long n = rowSize;
        Matrix<double> H(n, n), Q(n, n);
        for (long i = 0; i < n * n; i++)
            H[int(i)] = double(std::real(grid[i]));
        for (long i = 0; i < n; i++)
            Q(int(i), int(i)) = 1.0;

        std::vector<double> v((std::size_t)n);
        for (long k = 0; k + 2 < n; k++) {
            // Householder reflector zeroing H(k+2..n-1, k).
            double nrm = 0.0;
            for (long i = k + 1; i < n; i++)
                nrm += H(int(i), int(k)) * H(int(i), int(k));
            nrm = std::sqrt(nrm);
            if (nrm == 0.0)
                continue;
            const double alpha = H(int(k + 1), int(k)) >= 0.0 ? -nrm : nrm;
            for (long i = k + 1; i < n; i++)
                v[(std::size_t)i] = H(int(i), int(k));
            v[(std::size_t)(k + 1)] -= alpha;
            double vtv = 0.0;
            for (long i = k + 1; i < n; i++)
                vtv += v[(std::size_t)i] * v[(std::size_t)i];
            if (vtv == 0.0)
                continue;
            const double tau = 2.0 / vtv;
            // H := (I - tau v vᵀ) H (I - tau v vᵀ), applied from both sides.
            for (long j = 0; j < n; j++) {
                double d = 0.0;
                for (long i = k + 1; i < n; i++)
                    d += v[(std::size_t)i] * H(int(i), int(j));
                d *= tau;
                for (long i = k + 1; i < n; i++)
                    H(int(i), int(j)) -= d * v[(std::size_t)i];
            }
            for (long i = 0; i < n; i++) {
                double d = 0.0;
                for (long j = k + 1; j < n; j++)
                    d += H(int(i), int(j)) * v[(std::size_t)j];
                d *= tau;
                for (long j = k + 1; j < n; j++)
                    H(int(i), int(j)) -= d * v[(std::size_t)j];
            }
            // Accumulate Q the same way, so that A == Q H Qᵀ afterwards.
            for (long i = 0; i < n; i++) {
                double d = 0.0;
                for (long j = k + 1; j < n; j++)
                    d += Q(int(i), int(j)) * v[(std::size_t)j];
                d *= tau;
                for (long j = k + 1; j < n; j++)
                    Q(int(i), int(j)) -= d * v[(std::size_t)j];
            }
        }
        // Clean the numerical dust below the subdiagonal so the result is
        // exactly Hessenberg rather than Hessenberg to within rounding.
        for (long i = 2; i < n; i++)
            for (long j = 0; j + 2 <= i; j++)
                H(int(i), int(j)) = 0.0;
        return {H, Q};
    }

    // Real Schur form: A = Q * T * Qᵀ with Q orthogonal and T quasi-upper
    // triangular — 1x1 blocks for real eigenvalues, 2x2 blocks for each
    // complex-conjugate pair, which is why it is "quasi".
    //
    // schurDecomp() below computes this already but hands back flat
    // std::vectors because it is the internal engine for eig, pow and log.
    // This is the public spelling.
    //
    // Factor once, solve many times. See the Decomposition class below for
    // which factorisation is chosen and why. Declared here and defined after
    // that class, which cannot be complete before Matrix is.
    //
    // Usage: auto dA = A.factorize();  then  dA.solve(b1), dA.solve(b2), ...
    Decomposition<datatype> factorize() const;

    // Usage: auto [T, Q] = A.schur();
    std::pair<Matrix<double>, Matrix<double>> schur() const {
        static_assert(!is_complex<datatype>::value,
                      "schur: not yet implemented for complex datatypes.");
        if (rowSize != colSize)
            throw std::invalid_argument("schur: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        auto [Tv, Qv] = schurDecomp();
        const long n = rowSize;
        Matrix<double> T(n, n, Matrix<double>::uninit_t{});
        Matrix<double> Q(n, n, Matrix<double>::uninit_t{});
        for (long i = 0; i < n * n; i++) {
            T[int(i)] = Tv[(std::size_t)i];
            Q[int(i)] = Qv[(std::size_t)i];
        }
        return {T, Q};
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Structure predicates  (tier 4)
    // ═══════════════════════════════════════════════════════════════════
    // Named to match the IsDiagonal() that was already here. That capital I
    // is not this header's usual style, but a predicate family that agrees
    // with itself beats one where half the members match MATLAB's lowercase
    // and half match the neighbour they sit next to.
    //
    // Every one takes a tolerance relative to the size of the entries, so
    // they answer "is this matrix symmetric" rather than "did these floats
    // come out bit-identical", which is almost never the useful question.

    bool IsSymmetric(double tol = -1.0) const {
        if (rowSize != colSize)
            return false;
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < i; j++)
                if (magnitude(grid[i * colSize + j] - grid[j * colSize + i]) > t)
                    return false;
        return true;
    }

    // A == conj(A)^T. For a real matrix this is the same question as
    // IsSymmetric; for a complex one it is the one that actually matters,
    // since it is Hermitian — not symmetric — that gives real eigenvalues.
    bool IsHermitian(double tol = -1.0) const {
        if (rowSize != colSize)
            return false;
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j <= i; j++) {
                datatype conjugated;
                if constexpr (is_complex<datatype>::value)
                    conjugated = std::conj(grid[j * colSize + i]);
                else
                    conjugated = grid[j * colSize + i];
                if (magnitude(grid[i * colSize + j] - conjugated) > t)
                    return false;
            }
        return true;
    }

    // Upper triangular: everything below the k-th diagonal is zero.
    // k > 0 tests above the main diagonal, k < 0 below — same convention as
    // triu()/tril(), so IsUpper(k) is exactly "A == A.triu(k)".
    bool IsUpper(int k = 0, double tol = -1.0) const {
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (j - i < k && magnitude(grid[i * colSize + j]) > t)
                    return false;
        return true;
    }
    bool IsLower(int k = 0, double tol = -1.0) const {
        const double t = structureTol(tol);
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (j - i > k && magnitude(grid[i * colSize + j]) > t)
                    return false;
        return true;
    }

    // How far the non-zeros reach below and above the main diagonal.
    // {0, 0} is diagonal, {rows-1, cols-1} is dense. MATLAB's bandwidth.
    std::pair<long, long> bandwidth(double tol = -1.0) const {
        const double t = structureTol(tol);
        long below = 0, above = 0;
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                if (magnitude(grid[i * colSize + j]) > t) {
                    if (i > j && i - j > below)
                        below = i - j;
                    if (j > i && j - i > above)
                        above = j - i;
                }
        return {below, above};
    }

    bool IsBanded(long lower, long upper, double tol = -1.0) const {
        const std::pair<long, long> b = bandwidth(tol);
        return b.first <= lower && b.second <= upper;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Subspaces, elimination and vector products  (tier 4)
    // ═══════════════════════════════════════════════════════════════════

    // Orthonormal basis for the null space: the columns of V belonging to
    // singular values that are numerically zero. A * null(A) == 0.
    // Empty (n x 0) when A has full column rank.
    Matrix<double> null(double tol = -1.0) const {
        auto [U, S, V] = svd();
        (void)U;
        const long n = colSize;
        const double t = svdTol(S, tol);
        std::vector<long> keep;
        for (long j = 0; j < n; j++)
            if (j >= std::min(rowSize, colSize) || S(int(j), int(j)) <= t)
                keep.push_back(j);
        Matrix<double> out(n, (long)keep.size(), Matrix<double>::uninit_t{});
        for (std::size_t c = 0; c < keep.size(); c++)
            for (long i = 0; i < n; i++)
                out(int(i), int(c)) = V(int(i), int(keep[c]));
        return out;
    }

    // Orthonormal basis for the range (column space): the columns of U
    // belonging to the non-zero singular values. m x rank(A).
    Matrix<double> orth(double tol = -1.0) const {
        auto [U, S, V] = svd();
        (void)V;
        const double t = svdTol(S, tol);
        std::vector<long> keep;
        for (long j = 0; j < std::min(rowSize, colSize); j++)
            if (S(int(j), int(j)) > t)
                keep.push_back(j);
        Matrix<double> out(rowSize, (long)keep.size(), Matrix<double>::uninit_t{});
        for (std::size_t c = 0; c < keep.size(); c++)
            for (long i = 0; i < rowSize; i++)
                out(int(i), int(c)) = U(int(i), int(keep[c]));
        return out;
    }

    // Reduced row echelon form, by Gauss-Jordan with partial pivoting.
    //
    // A NOTE ON WHAT THIS IS FOR: rref is a teaching and exact-arithmetic
    // tool. On floating-point data the pivot-is-zero decision is a guess,
    // and a slightly different tolerance can change the reported rank. Use
    // rank(), null() and solve(), which go through the SVD or a pivoted QR,
    // for anything numerical. MATLAB's documentation says the same thing
    // about its rref, and it is worth repeating here.
    Matrix<double> rref(double tol = -1.0) const {
        Matrix<double> R(rowSize, colSize);
        for (long i = 0; i < rowSize * colSize; i++)
            R[int(i)] = double(std::real(grid[i]));
        const double t = tol >= 0.0 ? tol
                                    : std::numeric_limits<double>::epsilon() *
                                          double(std::max(rowSize, colSize)) *
                                          std::max(1.0, norm(NormType::Inf));
        long row = 0;
        for (long col = 0; col < colSize && row < rowSize; col++) {
            // Partial pivoting: the largest remaining entry in this column.
            long piv = row;
            for (long i = row + 1; i < rowSize; i++)
                if (std::abs(R(int(i), int(col))) > std::abs(R(int(piv), int(col))))
                    piv = i;
            if (std::abs(R(int(piv), int(col))) <= t) {
                for (long i = row; i < rowSize; i++)
                    R(int(i), int(col)) = 0.0;
                continue;  // no pivot in this column
            }
            if (piv != row)
                for (long j = 0; j < colSize; j++)
                    std::swap(R(int(row), int(j)), R(int(piv), int(j)));
            const double d = R(int(row), int(col));
            for (long j = 0; j < colSize; j++)
                R(int(row), int(j)) /= d;
            for (long i = 0; i < rowSize; i++) {
                if (i == row)
                    continue;
                const double f = R(int(i), int(col));
                if (f == 0.0)
                    continue;
                for (long j = 0; j < colSize; j++)
                    R(int(i), int(j)) -= f * R(int(row), int(j));
            }
            row++;
        }
        return R;
    }

    // Inner product of two equally shaped operands, sum(conj(a) * b) — the
    // conjugate goes on the LEFT, which is the convention that makes
    // A.dot(A) equal ||A||_F² for complex as well as real.
    datatype dot(const Matrix& B) const {
        requireSameShape(B, "dot");
        datatype acc = datatype(0);
        for (long i = 0; i < rowSize * colSize; i++) {
            if constexpr (is_complex<datatype>::value)
                acc += std::conj(grid[i]) * B.grid[i];
            else
                acc += grid[i] * B.grid[i];
        }
        return acc;
    }

    // Cross product. Three elements only — it is the one dimension where a
    // vector product of two vectors is again a vector.
    Matrix cross(const Matrix& B) const {
        if (rowSize * colSize != 3 || B.rows() * B.cols() != 3)
            throw std::invalid_argument("cross: both operands must have exactly 3 elements, got " +
                                        std::to_string(rowSize * colSize) + " and " +
                                        std::to_string(B.rows() * B.cols()));
        Matrix out(rowSize, colSize, uninit_t{});
        const datatype* a = grid;
        const datatype* b = B.grid;
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
        return out;
    }

    // Estimate of the 2-norm by power iteration on AᵀA, which is what
    // MATLAB's normest is for: norm(A, Two) computes a full SVD, and when
    // all you want is the largest singular value to a few digits that is a
    // great deal of work to throw away.
    double normest(double tol = 1e-6, int maxIter = 100) const {
        if (rowSize * colSize == 0)
            return 0.0;
        const long n = colSize;
        std::vector<double> x((std::size_t)n, 1.0 / std::sqrt(double(n)));
        std::vector<double> Ax((std::size_t)rowSize), y((std::size_t)n);
        double est = 0.0;
        for (int it = 0; it < maxIter; it++) {
            for (long i = 0; i < rowSize; i++) {
                double acc = 0.0;
                for (long j = 0; j < n; j++)
                    acc += double(std::real(grid[i * n + j])) * x[(std::size_t)j];
                Ax[(std::size_t)i] = acc;
            }
            for (long j = 0; j < n; j++) {
                double acc = 0.0;
                for (long i = 0; i < rowSize; i++)
                    acc += double(std::real(grid[i * n + j])) * Ax[(std::size_t)i];
                y[(std::size_t)j] = acc;
            }
            double nrm = 0.0;
            for (long j = 0; j < n; j++)
                nrm += y[(std::size_t)j] * y[(std::size_t)j];
            nrm = std::sqrt(nrm);
            if (nrm == 0.0)
                return 0.0;
            const double next = std::sqrt(nrm);
            for (long j = 0; j < n; j++)
                x[(std::size_t)j] = y[(std::size_t)j] / nrm;
            if (it && std::abs(next - est) <= tol * next)
                return next;
            est = next;
        }
        return est;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Scans, orderings and multiset reductions  (roadmap tier 2)
    // ═══════════════════════════════════════════════════════════════════
    //
    // Every one of these takes the same `addcol` flag the existing
    // reductions use, and means the same thing by it:
    //     addcol = 0  work DOWN each column  (result is per-column)
    //     addcol = 1  work ALONG each row    (result is per-row)
    // so prod(false) pairs with sum(false), cumsum(false) accumulates down
    // columns, sort(false) sorts each column, and so on.
    //
    // prod / cumsum / cumprod are defined for complex. Anything that has to
    // ORDER elements — sort, median, mode, unique — is not, for the reason
    // min() and max() already give: std::complex deliberately has no <, and
    // inventing one silently is worse than refusing to compile.

    // Product of every element.
    datatype prod() const {
        requireNonEmpty("prod");
        datatype p = datatype(1);
        for (long i = 0; i < rowSize * colSize; i++)
            p *= grid[i];
        return p;
    }

    // Per-column (addcol=0) or per-row (addcol=1) products.
    Matrix prod(const bool& addcol) const {
        const datatype* MATRIXCPP_RESTRICT g = grid;
        if (!addcol) {
            Matrix<datatype> out(1, colSize);
            datatype* MATRIXCPP_RESTRICT o = out.grid;
            for (long j = 0; j < colSize; j++)
                o[j] = datatype(1);
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
                for (long j = 0; j < colSize; j++)
                    o[j] *= row[j];
            }
            return out;
        }
        Matrix<datatype> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            datatype p = datatype(1);
            const datatype* MATRIXCPP_RESTRICT row = g + i * colSize;
            for (long j = 0; j < colSize; j++)
                p *= row[j];
            out.grid[i] = p;
        }
        return out;
    }

    // Cumulative sum. Same shape as the input — a scan, not a reduction.
    Matrix cumsum(const bool& addcol) const { return scan(addcol, true); }
    // Cumulative product, likewise.
    Matrix cumprod(const bool& addcol) const { return scan(addcol, false); }

    // Successive differences along an axis. The scanned dimension shrinks by
    // one, so diff(false) on (m x n) gives (m-1 x n) — MATLAB's diff.
    Matrix diff(const bool& addcol) const {
        if (!addcol) {
            if (rowSize < 2)
                return Matrix(0, colSize);
            Matrix out(rowSize - 1, colSize, uninit_t{});
            for (long i = 0; i + 1 < rowSize; i++)
                for (long j = 0; j < colSize; j++)
                    out.grid[i * colSize + j] = grid[(i + 1) * colSize + j] - grid[i * colSize + j];
            return out;
        }
        if (colSize < 2)
            return Matrix(rowSize, 0);
        Matrix out(rowSize, colSize - 1, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j + 1 < colSize; j++)
                out.grid[i * (colSize - 1) + j] = grid[i * colSize + j + 1] - grid[i * colSize + j];
        return out;
    }

    // Sorts each column (addcol=0) or each row (addcol=1). Same shape as the
    // input, like MATLAB's sort — this rearranges, it does not reduce.
    Matrix sort(const bool& addcol, bool descending = false) const {
        static_assert(!is_complex<datatype>::value,
                      "sort() needs an ordering; std::complex has none. Sort a component "
                      "or a magnitude instead — for example A.abs().sort(0).");
        Matrix out = *this;
        std::vector<datatype> buf((std::size_t)(addcol ? colSize : rowSize));
        const long outer = addcol ? rowSize : colSize;
        const long inner = addcol ? colSize : rowSize;
        for (long k = 0; k < outer; k++) {
            for (long t = 0; t < inner; t++)
                buf[(std::size_t)t] = addcol ? grid[k * colSize + t] : grid[t * colSize + k];
            if (descending)
                std::sort(buf.begin(), buf.end(), std::greater<datatype>());
            else
                std::sort(buf.begin(), buf.end());
            for (long t = 0; t < inner; t++) {
                if (addcol)
                    out.grid[k * colSize + t] = buf[(std::size_t)t];
                else
                    out.grid[t * colSize + k] = buf[(std::size_t)t];
            }
        }
        return out;
    }

    // Sorts whole ROWS, ordered by column `key`, carrying every other column
    // along — MATLAB's sortrows. The rows keep their contents; only their
    // order changes.
    Matrix sortrows(long key = 0, bool descending = false) const {
        static_assert(!is_complex<datatype>::value,
                      "sortrows() needs an ordering; std::complex has none.");
        if (rowSize == 0 || colSize == 0)
            return *this;
        if (key < 0 || key >= colSize)
            throw std::out_of_range("sortrows: key column " + std::to_string(key) +
                                    " is out of range for " + std::to_string(colSize) + " columns");
        std::vector<long> order((std::size_t)rowSize);
        std::iota(order.begin(), order.end(), 0L);
        const datatype* g = grid;
        const long cs = colSize;
        // stable_sort so that rows tying on the key keep their input order,
        // which is what makes repeated sortrows calls compose predictably.
        std::stable_sort(order.begin(), order.end(), [=](long a, long b) {
            return descending ? (g[b * cs + key] < g[a * cs + key])
                              : (g[a * cs + key] < g[b * cs + key]);
        });
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                out.grid[i * colSize + j] = grid[order[(std::size_t)i] * colSize + j];
        return out;
    }

    // Middle value; the mean of the two middle values when the count is even,
    // which is why this returns double rather than datatype.
    double median() const {
        static_assert(!is_complex<datatype>::value,
                      "median() needs an ordering; std::complex has none.");
        requireNonEmpty("median");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        return medianOf(buf);
    }

    // Per-column (addcol=0) or per-row (addcol=1) medians.
    Matrix<double> median(const bool& addcol) const {
        static_assert(!is_complex<datatype>::value,
                      "median() needs an ordering; std::complex has none.");
        requireNonEmpty("median");
        const long outer = addcol ? rowSize : colSize;
        const long inner = addcol ? colSize : rowSize;
        Matrix<double> out(addcol ? rowSize : 1, addcol ? 1 : colSize);
        std::vector<datatype> buf((std::size_t)inner);
        for (long k = 0; k < outer; k++) {
            for (long t = 0; t < inner; t++)
                buf[(std::size_t)t] = addcol ? grid[k * colSize + t] : grid[t * colSize + k];
            out[int(k)] = medianOf(buf);
        }
        return out;
    }

    // Most frequently occurring value. Ties go to the SMALLEST such value,
    // which is what MATLAB's mode does.
    datatype mode() const {
        static_assert(!is_complex<datatype>::value,
                      "mode() needs an ordering; std::complex has none.");
        requireNonEmpty("mode");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        std::sort(buf.begin(), buf.end());
        datatype best = buf[0];
        long bestRun = 0, run = 0;
        for (std::size_t i = 0; i < buf.size(); i++) {
            run = (i && buf[i] == buf[i - 1]) ? run + 1 : 1;
            if (run > bestRun) {
                bestRun = run;
                best = buf[i];
            }
        }
        return best;
    }

    // The distinct values, ascending, as a column vector — MATLAB's unique.
    Matrix unique() const {
        static_assert(!is_complex<datatype>::value,
                      "unique() needs an ordering; std::complex has none.");
        std::vector<datatype> buf(grid, grid + rowSize * colSize);
        std::sort(buf.begin(), buf.end());
        buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
        Matrix out((long)buf.size(), 1, uninit_t{});
        for (std::size_t i = 0; i < buf.size(); i++)
            out.grid[i] = buf[i];
        return out;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Shape manipulation  (tier 5)
    // ═══════════════════════════════════════════════════════════════════

    // Total element count. rows()*cols(), but spelled the way MATLAB and
    // NumPy both spell it, and without the chance of writing rows*rows.
    long numel() const { return rowSize * colSize; }

    // Tiles this matrix m times down and n times across.
    Matrix repmat(long m, long n) const {
        if (m < 0 || n < 0)
            throw std::invalid_argument("repmat: counts must be non-negative, got " +
                                        std::to_string(m) + " and " + std::to_string(n));
        Matrix out(rowSize * m, colSize * n, uninit_t{});
        for (long bi = 0; bi < m; bi++)
            for (long i = 0; i < rowSize; i++) {
                const datatype* MATRIXCPP_RESTRICT src = grid + i * colSize;
                datatype* dst = out.grid + (bi * rowSize + i) * colSize * n;
                for (long bj = 0; bj < n; bj++)
                    std::copy(src, src + colSize, dst + bj * colSize);
            }
        return out;
    }

    // Reverses the column order — a mirror about the vertical axis.
    Matrix fliplr() const {
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++)
                out.grid[i * colSize + j] = grid[i * colSize + (colSize - 1 - j)];
        return out;
    }

    // Reverses the row order — a mirror about the horizontal axis. Rows are
    // contiguous, so this is a run of whole-row copies.
    Matrix flipud() const {
        Matrix out(rowSize, colSize, uninit_t{});
        for (long i = 0; i < rowSize; i++)
            std::copy(grid + (rowSize - 1 - i) * colSize,
                      grid + (rowSize - i) * colSize,
                      out.grid + i * colSize);
        return out;
    }

    // Rotates by k quarter-turns COUNTERCLOCKWISE, as MATLAB's rot90 does.
    // Negative k turns the other way; k is taken modulo 4, so rot90(5) is
    // rot90(1) and rot90(-1) is rot90(3).
    Matrix rot90(int k = 1) const {
        int t = k % 4;
        if (t < 0)
            t += 4;
        switch (t) {
            case 0:
                return *this;
            case 1:
                return T().flipud();  // counterclockwise
            case 2:
                return fliplr().flipud();  // half turn
            default:
                return T().fliplr();  // clockwise
        }
    }

    // Circularly shifts elements by k places. dim=0 shifts rows (down for
    // positive k), dim=1 shifts columns (right). Nothing is lost: what falls
    // off one end reappears at the other.
    Matrix circshift(long k, int dim = 0) const {
        if (dim != 0 && dim != 1)
            throw std::invalid_argument("circshift: dim must be 0 (rows) or 1 (columns), got " +
                                        std::to_string(dim));
        Matrix out(rowSize, colSize, uninit_t{});
        const long n = (dim == 0) ? rowSize : colSize;
        if (n == 0)
            return out;
        // C's % keeps the sign of the dividend, so a negative shift needs
        // wrapping back into [0, n) before it can be used as an offset.
        const long shift = ((k % n) + n) % n;
        for (long i = 0; i < rowSize; i++)
            for (long j = 0; j < colSize; j++) {
                const long si = (dim == 0) ? ((i - shift) % rowSize + rowSize) % rowSize : i;
                const long sj = (dim == 1) ? ((j - shift) % colSize + colSize) % colSize : j;
                out.grid[i * colSize + j] = grid[si * colSize + sj];
            }
        return out;
    }

    // Block-diagonal concatenation: this and M placed on the diagonal of a
    // larger matrix, everything else zero. Shapes need not match, and need
    // not even be square.
    Matrix blkdiag(const Matrix& M) const {
        Matrix out(rowSize + M.rowSize, colSize + M.colSize);
        for (long i = 0; i < rowSize; i++)
            std::copy(grid + i * colSize, grid + (i + 1) * colSize, out.grid + i * out.colSize);
        for (long i = 0; i < M.rowSize; i++)
            std::copy(M.grid + i * M.colSize,
                      M.grid + (i + 1) * M.colSize,
                      out.grid + (rowSize + i) * out.colSize + colSize);
        return out;
    }

    // Concatenates M to this matrix. concatCol=0: vertical (stack rows, cols must
    // match).
    //                               concatCol=1: horizontal (stack cols, rows
    //                               must match).
    // If this matrix is empty, returns M directly.
    Matrix concat(const Matrix& M, const bool& concatCol) const {
        if (this->empty())
            return M;
        try {
            if (!concatCol && this->colSize != M.colSize)
                throw std::invalid_argument(
                    "concat: column size mismatch: " + std::to_string(colSize) +
                    " != " + std::to_string(M.colSize));
            if (concatCol && this->rowSize != M.rowSize)
                throw std::invalid_argument(
                    "concat: row size mismatch: " + std::to_string(rowSize) +
                    " != " + std::to_string(M.rowSize));
        } catch (const std::exception& e) {
            std::cerr << "Matrix concat error: " << e.what() << std::endl;
            throw;
        }
        // Both branches used to go through operator(), which wraps negative
        // indices and therefore runs a modulo on BOTH coordinates — four
        // integer divisions per element copied, for indices already known to
        // be in range. These are plain contiguous copies instead.
        // std::copy rather than a hand-written loop: for a trivially copyable
        // element type it lowers to memmove, which the C library implements
        // with wide vector loads and non-temporal stores. concat is purely
        // memory-bound, so that is the whole cost of the operation.
        if (!concatCol) {
            // Vertical: rows stack and the row length is unchanged, so the
            // two source blocks are already contiguous runs.
            Matrix<datatype> out(rowSize + M.rowSize, colSize, uninit_t{});
            const long a = rowSize * colSize;
            std::copy(grid, grid + a, out.grid);
            std::copy(M.grid, M.grid + M.rowSize * colSize, out.grid + a);
            return out;
        }
        // Horizontal: rows interleave, so copy one row segment at a time.
        const long outCols = colSize + M.colSize;
        Matrix<datatype> out(rowSize, outCols, uninit_t{});
        for (long i = 0; i < rowSize; i++) {
            datatype* dst = out.grid + i * outCols;
            std::copy(grid + i * colSize, grid + (i + 1) * colSize, dst);
            std::copy(M.grid + i * M.colSize, M.grid + (i + 1) * M.colSize, dst + colSize);
        }
        return out;
    }

    // Kronecker product: replaces every element A_ij with the block A_ij * M.
    //
    // Named kron(), not tensor(), for two reasons. It is what MATLAB, NumPy
    // and SciPy all call it, and — since Tensor.hpp added a TYPE called
    // Tensor — `A.tensor(B)` would have been an operation sharing its name
    // with a data type while returning the other one (a Matrix). That is the
    // kind of collision that reads fine to whoever wrote it and confuses
    // everyone else.
    //
    // It stays a named function rather than getting an operator. The result
    // is (m*p) x (n*q), so it is the one operation here that explodes: two
    // 1000x1000 matrices give 10^12 elements, 8 TB. That cost should be
    // visible at the call site, not two characters away. It is also why the
    // benchmark only takes this one to n = 48 while everything else runs to
    // 512 or 2000.
    // Returns a (rows*M.rows x cols*M.cols) Matrix.
    Matrix kron(const Matrix& M) const {
        const long rM = M.rowSize, cM = M.colSize;
        const long cOut = colSize * cM;
        Matrix<datatype> ans(rowSize * rM, cOut, uninit_t{});
        // Iterating by output block rather than by flat index. The flat-index
        // version cost six integer divisions per element — two to split idx
        // into (r, c), two more to split those into block and offset, and two
        // more inside operator() — for a loop whose structure already knows
        // every one of those values. Nested loops carry them for free, and
        // the A element is loaded once per block instead of per element.
        for (long i = 0; i < rowSize; i++) {
            for (long j = 0; j < colSize; j++) {
                const datatype a = grid[i * colSize + j];
                for (long p = 0; p < rM; p++) {
                    datatype* dst = ans.grid + (i * rM + p) * cOut + j * cM;
                    const datatype* src = M.grid + p * cM;
                    for (long q = 0; q < cM; q++)
                        dst[q] = a * src[q];
                }
            }
        }
        return ans;
    }

    // --- Structural extraction and reshaping ---

    // Extracts the main diagonal as a (min(rows,cols) x 1) column vector.
    // Works for non-square matrices. To go the other way — build a diagonal
    // matrix FROM a vector — use the free function diag(v) near the bottom.
    Matrix diag() const {
        long d = std::min(rowSize, colSize);
        Matrix<datatype> out(d, 1);
        for (long i = 0; i < d; i++)
            out.grid[i] = grid[i * colSize + i];
        return out;
    }

    // Upper triangle: copies elements on and above the k-th diagonal, zeros the
    // rest. k=0 is the main diagonal, k>0 moves above it, k<0 below. Mirrors
    // numpy.triu.
    Matrix triu(int k = 0) const {
        Matrix<datatype> out(rowSize, colSize);
        for (long i = 0; i < rowSize; i++)
            for (long j = std::max(0L, i + k); j < colSize; j++)
                out.grid[i * colSize + j] = grid[i * colSize + j];
        return out;
    }

    // Lower triangle: copies elements on and below the k-th diagonal, zeros the
    // rest. k=0 is the main diagonal, k>0 moves above it, k<0 below. Mirrors
    // numpy.tril.
    Matrix tril(int k = 0) const {
        Matrix<datatype> out(rowSize, colSize);
        for (long i = 0; i < rowSize; i++) {
            long hi = std::min(colSize - 1, i + k);
            for (long j = 0; j <= hi; j++)
                out.grid[i * colSize + j] = grid[i * colSize + j];
        }
        return out;
    }

    // Reinterprets the elements as a (newRows x newCols) matrix in row-major
    // order. Requires newRows * newCols == rows * cols. Returns a new Matrix; the
    // underlying data is copied, not aliased.
    Matrix reshape(long newRows, long newCols) const {
        if (newRows < 0 || newCols < 0)
            throw std::invalid_argument("reshape: dimensions must be non-negative, got " +
                                        std::to_string(newRows) + "x" + std::to_string(newCols));
        if (newRows * newCols != rowSize * colSize)
            throw std::invalid_argument("reshape: cannot reshape " + std::to_string(rowSize) + "x" +
                                        std::to_string(colSize) + " (" +
                                        std::to_string(rowSize * colSize) + " elements) into " +
                                        std::to_string(newRows) + "x" + std::to_string(newCols) +
                                        " (" + std::to_string(newRows * newCols) + ")");
        // uninit_t, not the zeroing constructor: every element is about to
        // be overwritten, so zero-filling first writes the whole buffer twice.
        // std::copy rather than an element loop, so a trivially copyable type
        // lowers to memmove and its wide vector stores.
        Matrix<datatype> out(newRows, newCols, uninit_t{});
        std::copy(grid, grid + rowSize * colSize, out.grid);
        return out;
    }

    // --- Random initialisation ---

    // Fills every element with a random value in [lowBound, highBound).
    // Seed is derived from the current time (unique within a 24-hour window).
    // Integral types are rounded to the nearest integer.
    // Returns *this to allow chaining.
    Matrix& set_Ran_values(double lowBound, double highBound) {
        try {
            if (lowBound >= highBound)
                throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) +
                    " > high: " + std::to_string(highBound));
            double range = highBound - lowBound;
            long seed;
            setRan(seed);
            for (long i = 0; i < rowSize * colSize; i++) {
                double val = range * ran2(&seed) + lowBound;
                grid[i] =
                    std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
            }
        } catch (const std::exception& e) {
            std::cerr << "Bounds error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }
    // Fills every element with a random value in [lowBound, highBound) using a
    // custom seed. customSeed MUST be negative (required by ran2 to trigger
    // initialisation). Integral types are rounded to the nearest integer. Returns
    // *this to allow chaining.
    Matrix& set_Ran_values(double lowBound, double highBound, long customSeed) {
        try {
            if (customSeed >= 0)
                throw std::invalid_argument("Seed, " + std::to_string(customSeed) + " is >= 0");
            if (lowBound >= highBound)
                throw std::invalid_argument(
                    "Improper boundaries given, low: " + std::to_string(lowBound) +
                    " > high: " + std::to_string(highBound));
            double range = highBound - lowBound;
            long seed = customSeed;
            for (long i = 0; i < rowSize * colSize; i++) {
                double val = range * ran2(&seed) + lowBound;
                grid[i] =
                    std::is_integral<datatype>::value ? datatype(std::round(val)) : datatype(val);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
        return *this;
    }
    //

    // QR factorisation with column pivoting using Householder reflections.
    // Mirrors LAPACK's DGEQP3: at each step the column with the largest remaining
    // norm is pivoted to the front, then a Householder reflector eliminates the
    // sub-diagonal entries of that column. Column norms are maintained via the
    // Bischof-Pan rank-1 downdate, avoiding a full norm recomputation each step.
    //
    // For m×n matrix A, computes  A * P = Q * R  where:
    //   Q — m×m orthogonal (product of Householder reflectors)
    //   R — m×n upper triangular
    //   P — n×n permutation matrix (column pivoting for stability)
    //
    // Works for any m×n, including m < n.
    // Usage: auto [Q, R, P] = A.QR();
    std::tuple<Matrix<double>, Matrix<double>, Matrix<datatype>> QR() const {
        try {
            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("QR: matrix must be non-empty");

            int m = (int)rowSize, n = (int)colSize;
            int r = std::min(m, n);

            // Working copy in double, stored COLUMN BY COLUMN — Wt[j*m + i]
            // is A(i,j). Every step of a Householder QR works down columns: the
            // reflector is built from a column, applied to each trailing column,
            // and the pivot search swaps whole columns. Row-major storage makes
            // all of that stride by n, one cache line per element, which is why
            // the factorisation ran at ~0.3 GFLOP/s. LAPACK's dgeqp3 never has
            // this problem because Fortran arrays are column-major; storing the
            // transpose gets the same contiguity here, and the reflector
            // application becomes a plain dot product followed by an axpy.
            // Q is already accumulated transposed for exactly this reason —
            // see the Q block below.
            std::vector<double> Wt((size_t)n * m);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Wt[(size_t)j * m + i] = double(grid[(size_t)i * n + j]);
            double* MATRIXCPP_RESTRICT wt = Wt.data();

            // Column pivot tracking — pivots[k] = original column index at position k
            std::vector<int> pivots(n);
            std::iota(pivots.begin(), pivots.end(), 0);

            // Squared column norms for Bischof-Pan pivot selection
            std::vector<double> sqNorms(n, 0.0);
            for (int j = 0; j < n; j++)
                sqNorms[j] = pairwiseSum_sq(wt + (size_t)j * m, m);

            // Householder taus + vectors stored for Q accumulation
            std::vector<double> taus(r, 0.0);
            std::vector<std::vector<double>> hvecs(r);

            for (int k = 0; k < r; k++) {
                // ── Pivot: bring the largest-norm remaining column to position k ──
                int jmax = k;
                for (int j = k + 1; j < n; j++)
                    if (sqNorms[j] > sqNorms[jmax])
                        jmax = j;
                if (jmax != k) {
                    // A column is a contiguous run now, so the swap is one
                    // memory-to-memory exchange instead of m strided ones.
                    std::swap_ranges(
                        wt + (size_t)k * m, wt + (size_t)k * m + m, wt + (size_t)jmax * m);
                    std::swap(pivots[k], pivots[jmax]);
                    std::swap(sqNorms[k], sqNorms[jmax]);
                }

                double* MATRIXCPP_RESTRICT colk = wt + (size_t)k * m;
                const int sz = m - k;

                // ── Householder reflector for column k, rows k:m-1 ──
                // Choose alpha opposite in sign to x[0] to avoid cancellation.
                double xnorm = std::sqrt(pairwiseSum_sq(colk + k, sz));

                if (xnorm == 0.0) {
                    hvecs[k].assign(sz, 0.0);
                    continue;
                }

                double alpha = (colk[k] >= 0.0 ? -1.0 : 1.0) * xnorm;
                std::vector<double> v(colk + k, colk + m);
                v[0] -= alpha;  // v = x - alpha*e_1

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;
                double tau = 2.0 / vTv;
                taus[k] = tau;
                hvecs[k] = v;

                // Apply H_k = I - tau*v*v^T to trailing block W(k:m-1, k:n-1).
                // Every column j is updated independently of every other, so
                // this — the O(mn²) bulk of the factorisation — is embarrassingly
                // parallel. It is also the loop that column pivoting forces to
                // stay at BLAS level 2: the norms have to be downdated before
                // the next pivot can be chosen, so unlike LAPACK's unpivoted
                // blocked dgeqrf there is no way to batch several reflectors
                // into one matrix-matrix product.
                const double* MATRIXCPP_RESTRICT vp = v.data();
                auto applyCol = [=](int j) {
                    double* MATRIXCPP_RESTRICT wj = wt + (size_t)j * m + k;
                    double d0 = 0.0, d1 = 0.0;
                    int i = 0;
                    for (; i + 1 < sz; i += 2) {
                        d0 += vp[i] * wj[i];
                        d1 += vp[i + 1] * wj[i + 1];
                    }
                    for (; i < sz; i++)
                        d0 += vp[i] * wj[i];
                    const double f = tau * (d0 + d1);
                    for (i = 0; i < sz; i++)
                        wj[i] -= f * vp[i];
                };
                const long applyWork = (long)(n - k) * sz;
                (void)applyWork;  // only read on the OpenMP path
#ifdef _OPENMP
                if (applyWork >= 32768) {
                    const int th = (int)std::min<long>(std::max<long>(applyWork / 8192, 1),
                                                       omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(th)
                    for (int j = k; j < n; j++)
                        applyCol(j);
                } else
#endif
                {
                    for (int j = k; j < n; j++)
                        applyCol(j);
                }

                // Bischof-Pan downdate: H_k is orthogonal so column norms are
                // preserved; the squared norm below row k shrinks by W(k,j)^2.
                for (int j = k + 1; j < n; j++) {
                    const double wkj = wt[(size_t)j * m + k];
                    sqNorms[j] -= wkj * wkj;
                    if (sqNorms[j] < 0.0)
                        sqNorms[j] = 0.0;
                }
            }

            // ── Materialise R (upper triangle of the worked array) ──
            Matrix<double> R(m, n);
            for (int i = 0; i < m; i++)
                for (int j = i; j < n; j++)
                    R(i, j) = wt[(size_t)j * m + i];

            // ── Accumulate Q = H_0 * H_1 * … * H_{r-1} ──
            // Apply reflectors in reverse order to the m×m identity.
            // At descending step k, columns 0:k-1 of Q are zero in rows k:m-1,
            // so only columns k:m-1 need updating.
            //
            // Accumulated TRANSPOSED. Each reflector touches rows k..m-1 of a
            // fixed column, which in row-major storage strides by a whole row
            // — a cache miss per element, and this loop is the bulk of the
            // factorisation. Working on Qt makes the same update contiguous;
            // it is transposed back once at the end, which is O(m²) against
            // the O(m³) it saves. Same reasoning as the Q accumulation in
            // schurDecomp().
            std::vector<double> Qt((size_t)m * m, 0.0);
            for (int i = 0; i < m; i++)
                Qt[(size_t)i * m + i] = 1.0;
            double* MATRIXCPP_RESTRICT qt = Qt.data();
            for (int k = r - 1; k >= 0; k--) {
                if (taus[k] == 0.0)
                    continue;
                const double* MATRIXCPP_RESTRICT v = hvecs[k].data();
                const int sz = (int)hvecs[k].size();
                const double tau = taus[k];
                // Independent per column j, exactly like the panel update, so
                // it parallelises the same way.
                auto applyQ = [=](int j) {
                    double* MATRIXCPP_RESTRICT qj = qt + (size_t)j * m + k;  // Q(k.., j)
                    double d0 = 0.0, d1 = 0.0;
                    int i = 0;
                    for (; i + 1 < sz; i += 2) {
                        d0 += v[i] * qj[i];
                        d1 += v[i + 1] * qj[i + 1];
                    }
                    for (; i < sz; i++)
                        d0 += v[i] * qj[i];
                    const double f = tau * (d0 + d1);
                    for (i = 0; i < sz; i++)
                        qj[i] -= f * v[i];
                };
                const long qWork = (long)(m - k) * sz;
                (void)qWork;  // only read on the OpenMP path
#ifdef _OPENMP
                if (qWork >= 32768) {
                    const int th =
                        (int)std::min<long>(std::max<long>(qWork / 8192, 1), omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(th)
                    for (int j = k; j < m; j++)
                        applyQ(j);
                } else
#endif
                {
                    for (int j = k; j < m; j++)
                        applyQ(j);
                }
            }
            Matrix<double> Q(m, m, Matrix<double>::uninit_t{});
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    Q(i, j) = Qt[(size_t)j * m + i];

            // ── Materialise P: A*P = Q*R, so P[pivots[k], k] = 1 ──
            Matrix<datatype> P(n, n);
            for (int k = 0; k < n; k++)
                P(pivots[k], k) = datatype(1);

            return std::make_tuple(Q, R, P);

        } catch (const std::exception& e) {
            std::cerr << "QR factorization error: " << e.what() << '\n';
            throw;
        }
    }

    // Matrix norm. Works for any m×n.
    //   NormType::Fro — sqrt(sum of squares of every element)
    //   NormType::One — max absolute column sum
    //   NormType::Inf — max absolute row sum
    //   NormType::Two — largest singular value; needs svd(), so implement that
    //   first
    // Usage: double e = (A*x - b).norm();
    double norm(NormType type = NormType::Fro) const {
        if (rowSize * colSize == 0)
            return 0.0;
        switch (type) {
            case NormType::One: {
                double best = 0.0;
                for (long j = 0; j < colSize; j++) {
                    double acc = 0.0;
                    for (long i = 0; i < rowSize; i++)
                        acc += magnitude(grid[i * colSize + j]);
                    if (acc > best)
                        best = acc;
                }
                return best;
            }
            case NormType::Inf: {
                double best = 0.0;
                for (long i = 0; i < rowSize; i++) {
                    double acc = 0.0;
                    for (long j = 0; j < colSize; j++)
                        acc += magnitude(grid[i * colSize + j]);
                    if (acc > best)
                        best = acc;
                }
                return best;
            }
            case NormType::Two: {
                // if constexpr, not a plain if: svd() static_asserts against
                // complex, and a runtime branch would still instantiate it —
                // which would make even A.norm() (Frobenius) fail to compile
                // for a complex matrix, though Fro/One/Inf are perfectly well
                // defined there via the complex modulus.
                if constexpr (is_complex<datatype>::value) {
                    throw std::invalid_argument(
                        "norm(Two): the spectral norm needs svd(), which is not yet "
                        "implemented for complex datatypes. Fro/One/Inf all work.");
                } else {
                    auto [U, S, V] = svd();
                    (void)U;
                    (void)V;
                    return S.rows() && S.cols() ? S(0, 0) : 0.0;  // sorted descending
                }
            }
            case NormType::Fro:
            default: {
                // Fast path: one pass, four accumulators, tracking the largest
                // magnitude alongside the sum so the safety of the result can
                // be decided by arithmetic rather than by inspecting it.
                //
                // Deliberately NOT std::isfinite(acc): -ffast-math implies
                // -ffinite-math-only, under which that folds to a constant
                // true and the guard silently disappears. Comparing the
                // tracked maximum against a bound no optimisation flag can
                // reason away keeps this correct under every build line.
                const long total = rowSize * colSize;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, mx = 0.0;
                long i = 0;
                // Squared magnitudes throughout — the overflow guard compares
                // against the squared bound, so no square root is needed here
                // either. For a complex matrix that removes one sqrt per element.
                // The accumulators may overflow to infinity here; that is
                // harmless, because whether to trust them is decided by mx,
                // which tracks the largest COMPONENT and so cannot itself
                // overflow. Testing the accumulated square would be too late.
                for (; i + 3 < total; i += 4) {
                    a0 += magnitudeSq(grid[i]);
                    a1 += magnitudeSq(grid[i + 1]);
                    a2 += magnitudeSq(grid[i + 2]);
                    a3 += magnitudeSq(grid[i + 3]);
                    const double c0 = maxComponent(grid[i]), c1 = maxComponent(grid[i + 1]);
                    const double c2 = maxComponent(grid[i + 2]), c3 = maxComponent(grid[i + 3]);
                    const double p0 = c0 > c1 ? c0 : c1, p1 = c2 > c3 ? c2 : c3;
                    const double p = p0 > p1 ? p0 : p1;
                    if (p > mx)
                        mx = p;
                }
                for (; i < total; i++) {
                    a0 += magnitudeSq(grid[i]);
                    const double c = maxComponent(grid[i]);
                    if (c > mx)
                        mx = c;
                }
                // Safe when total * mx^2 cannot overflow, and when the squares
                // are still above the subnormal floor. Outside that window the
                // scaled pass below is the only way to get the right answer.
                // Safe when total * 2 * mx^2 stays inside double's range (the
                // 2 covers re^2 + im^2 for a complex element), and when the
                // squares stay above the subnormal floor.
                const double hiBound = std::sqrt(std::numeric_limits<double>::max() /
                                                 (2.0 * double(total > 0 ? total : 1)));
                const double loBound = std::sqrt(std::numeric_limits<double>::min()) * 1e3;
                if (mx == 0.0)
                    return 0.0;
                if (mx < hiBound && mx > loBound)
                    return std::sqrt((a0 + a1) + (a2 + a3));

                // Slow path: rescale by the largest magnitude so the squares
                // stay representable. Reached only for matrices whose entries
                // sit near the top or bottom of double's range.
                double sc = 0.0;
                for (long k = 0; k < total; k++) {
                    double t = magnitude(grid[k]) / mx;
                    sc += t * t;
                }
                return mx * std::sqrt(sc);
            }
        }
    }

    // Numerical rank: the number of linearly independent columns.
    // The column-pivoted QR above is already the rank-revealing tool — count the
    // diagonal entries of R whose magnitude exceeds tol. Passing tol < 0 selects
    // the LAPACK-style default, max(m,n) * eps * |R(0,0)|.
    long rank(double tol = -1.0) const {
        if (rowSize * colSize == 0)
            return 0;
        auto [Q, R, P] = QR();
        (void)Q;
        (void)P;
        long d = std::min(rowSize, colSize);
        // Column pivoting orders |R(i,i)| non-increasingly, so the first entry
        // is the largest and the count can stop at the first one below tol.
        if (tol < 0.0)
            tol = double(std::max(rowSize, colSize)) * std::numeric_limits<double>::epsilon() *
                  std::abs(R(0, 0));
        long r = 0;
        for (long i = 0; i < d; i++) {
            if (std::abs(R(int(i), int(i))) <= tol)
                break;
            r++;
        }
        return r;
    }

    // Condition number ||A|| * ||A^-1|| in the given norm — how much a relative
    // error in b is amplified when solving A*x = b. Large means ill-conditioned.
    // NormType::Two is the usual choice and equals sigma_max / sigma_min from
    // svd().
    //
    // A singular matrix returns infinity. CAUTION under this project's compile
    // line: -ffast-math implies -ffinite-math-only, which lets the compiler
    // assume infinities never occur, and std::isinf() then folds to false. The
    // value returned is still infinity; it is the TEST that stops working. Use
    // a magnitude threshold (c > 1e15) rather than std::isinf() if you build
    // with -ffast-math, or drop the flag.
    double cond(NormType type = NormType::Two) const {
        const double inf = std::numeric_limits<double>::infinity();
        if (rowSize * colSize == 0)
            return 0.0;
        if (type == NormType::Two) {
            auto [U, S, V] = svd();
            (void)U;
            (void)V;
            long d = std::min(S.rows(), S.cols());
            double smax = S(0, 0), smin = S(int(d - 1), int(d - 1));
            return smin == 0.0 ? inf : smax / smin;  // singular, infinitely ill-conditioned
        }
        if (rowSize != colSize)
            throw std::invalid_argument(
                "cond: the One/Inf/Fro condition number needs a square matrix, got " +
                std::to_string(rowSize) + "x" + std::to_string(colSize) +
                " — use NormType::Two, which is defined for any shape");
        try {
            return norm(type) * inverse().norm(type);
        } catch (const std::exception&) {
            return inf;  // inverse() throws on a singular matrix
        }
    }

    // Performs LU factorization with partial pivoting (Doolittle's method).
    // Requires a square matrix of at least 2x2. Throws if singular.
    // Returns std::tuple<L, U, P> where PA = LU.
    // Usage: auto [L, U, P] = A.LU();
    std::tuple<Matrix<double>, Matrix<double>, Matrix<datatype>> LU() const {
        try {
            auto [packed, pivotVec] = luPacked();
            int n = (int)rowSize;
            auto pat = [&](int i, int j) { return packed[i * n + j]; };

            Matrix<double> L(n, n), U(n, n);
            for (int i = 0; i < n; i++) {
                L(i, i) = 1.0;
                for (int j = 0; j < i; j++)
                    L(i, j) = pat(i, j);
                for (int j = i; j < n; j++)
                    U(i, j) = pat(i, j);
            }

            Matrix<datatype> P(n, n);
            for (int i = 0; i < n; i++)
                P(i, i) = datatype(1);
            for (int k = 0; k < n; k++) {
                if (pivotVec[k] != k)
                    for (int j = 0; j < n; j++)
                        std::swap(P(k, j), P(pivotVec[k], j));
            }

            return std::make_tuple(L, U, P);
        } catch (const std::exception& e) {
            std::cerr << "LU factorization error: " << e.what() << '\n';
            throw;
        }
    }

    // Cholesky factorisation A = L * L^T for symmetric positive-definite A.
    // Roughly half the work of LU since it exploits symmetry — the reason it is
    // the default for covariance matrices, normal equations, Kalman filters and
    // GP kernels. Returns the lower-triangular L (upper triangle zeroed).
    // Throws if A is not square, not symmetric, or not positive definite —
    // a failed Cholesky is in fact the standard *test* for positive definiteness.
    // Usage: auto L = A.cholesky();
    Matrix<double> cholesky() const {
        static_assert(!is_complex<datatype>::value,
                      "cholesky: not yet implemented for complex datatypes. It would compile "
                      "by\n"
                      "taking std::real() of each entry and silently discard the imaginary\n"
                      "part — a wrong answer, not a limitation. See roadmap item 1 "
                      "(work_t).");
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("cholesky: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("cholesky: matrix must be non-empty");
            int n = (int)rowSize;

            // Symmetry check, relative to the size of the entries involved.
            double scale = norm(NormType::Inf);
            double symTol =
                std::numeric_limits<double>::epsilon() * 100.0 * (scale > 0.0 ? scale : 1.0);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < i; j++)
                    if (magnitude(grid[i * n + j] - grid[j * n + i]) > symTol)
                        throw std::domain_error("cholesky: matrix is not symmetric — A(" +
                                                std::to_string(i) + "," + std::to_string(j) +
                                                ") != A(" + std::to_string(j) + "," +
                                                std::to_string(i) + ")");

            // Right-looking Cholesky in the shape of LAPACK's unblocked dpotf2:
            // row i of L is built from dot products of two ALREADY-COMPLETED
            // rows of L, which are contiguous runs in row-major order. Two
            // things were costing far more than the arithmetic:
            //
            //   * L(i,k) and L(j,k) went through operator(), which wraps
            //     negative indices and therefore runs a modulo on both
            //     coordinates — four integer divisions in the innermost loop
            //     of an O(n³/6) algorithm. This alone held the factorisation
            //     to 0.78 GFLOP/s.
            //   * one accumulator makes each multiply-add wait for the
            //     previous one, so the dot product ran at the latency of an
            //     FMA rather than its throughput. Four independent chains fix
            //     that, exactly as sum() does.
            //
            // The i==j case is split out of the j loop rather than tested
            // inside it: the diagonal needs L(i,k)² and a square root, the
            // off-diagonal needs a division, and branching on that in the
            // hot loop blocks vectorisation for both.
            Matrix<double> L(n, n);
            double* MATRIXCPP_RESTRICT Lg = L.grid;
            for (long i = 0; i < n; i++) {
                double* MATRIXCPP_RESTRICT Li = Lg + i * n;
                for (long j = 0; j < i; j++) {
                    const double* MATRIXCPP_RESTRICT Lj = Lg + j * n;
                    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                    long k = 0;
                    for (; k + 3 < j; k += 4) {
                        a0 += Li[k] * Lj[k];
                        a1 += Li[k + 1] * Lj[k + 1];
                        a2 += Li[k + 2] * Lj[k + 2];
                        a3 += Li[k + 3] * Lj[k + 3];
                    }
                    for (; k < j; k++)
                        a0 += Li[k] * Lj[k];
                    const double acc = double(std::real(grid[i * n + j])) - ((a0 + a1) + (a2 + a3));
                    Li[j] = acc / Lj[j];
                }
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
                long k = 0;
                for (; k + 3 < i; k += 4) {
                    d0 += Li[k] * Li[k];
                    d1 += Li[k + 1] * Li[k + 1];
                    d2 += Li[k + 2] * Li[k + 2];
                    d3 += Li[k + 3] * Li[k + 3];
                }
                for (; k < i; k++)
                    d0 += Li[k] * Li[k];
                const double piv = double(std::real(grid[i * n + i])) - ((d0 + d1) + (d2 + d3));
                // A non-positive pivot IS the proof that A is not positive
                // definite — this failure is the standard PD test.
                if (piv <= 0.0)
                    throw std::domain_error(
                        "cholesky: matrix is not positive definite — non-positive "
                        "pivot " +
                        std::to_string(piv) + " at index " + std::to_string(i));
                Li[i] = std::sqrt(piv);
            }
            return L;
        } catch (const std::exception& e) {
            std::cerr << "cholesky() error: " << e.what() << '\n';
            throw;
        }
    }

    // Returns the determinant via LU factorisation.
    // Integer types are rounded to avoid floating-point drift (e.g. 2.9999 → 3).
    datatype det() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("det() requires a square matrix, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            auto [packed, pivotVec] = luPacked();
            int n = (int)rowSize;
            int sign = 1;
            for (int k = 0; k < n; k++)
                if (pivotVec[k] != k)
                    sign = -sign;
            double d = double(sign);
            for (int i = 0; i < n; i++)
                d *= packed[i * n + i];
            return std::is_integral<datatype>::value ? datatype(std::round(d)) : datatype(d);
        } catch (const std::exception& e) {
            std::cerr << "det() error: " << e.what() << '\n';
            throw;
        }
    }

    // Eigendecomposition via the implicit-shift QR algorithm.
    // Usage: auto [eigenvalues, Q] = A.eig();
    //   eigenvalues — n×1 column vector (diagonal of Schur form)
    //   Q           — n×n orthogonal matrix (eigenvectors for symmetric A,
    //                 Schur vectors for general A)
    //
    // REAL eigenvalues only. schurDecomp() returns a REAL Schur form, in which
    // a complex-conjugate pair a±bi occupies a 2x2 diagonal block rather than
    // a single entry; reading the bare diagonal there yields `a` twice and
    // discards ±bi. This used to happen silently — a plain 2D rotation matrix
    // is enough to trigger it — so eig() now detects such a block and THROWS
    // rather than returning a plausible-looking wrong answer.
    //
    // If the matrix may have complex eigenvalues, call eigvals() below, which
    // is block-aware and returns Matrix<std::complex<double>>. Symmetric
    // matrices are always real-eigenvalued, so eig() is safe for those by
    // construction. schurDecomp() remains available for the raw factors.
    std::pair<Matrix<double>, Matrix<double>> eig() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("eig: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("eig: matrix must be non-empty");
            int n = (int)rowSize;
            auto [H, Qv] = schurDecomp();
            for (int i = 0; i + 1 < n; i++)
                if (isSchurBlock(H, n, i))
                    throw std::domain_error(
                        "eig: the Schur form has a 2x2 block at index " + std::to_string(i) +
                        ", i.e. a complex-conjugate eigenvalue pair. Real eigenvalues "
                        "cannot represent it — use eigvals(), which returns "
                        "Matrix<std::complex<double>>");
            Matrix<double> eigenvals(n, 1), eigenvecs(n, n);
            for (int i = 0; i < n; i++)
                eigenvals(i, 0) = H[i * n + i];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    eigenvecs(i, j) = Qv[i * n + j];
            return {eigenvals, eigenvecs};
        } catch (const std::exception& e) {
            std::cerr << "eig() error: " << e.what() << '\n';
            throw;
        }
    }

    // Every eigenvalue, complex ones included, as an n×1 complex column vector.
    // This is the block-aware counterpart to eig(): it walks the real Schur
    // form and, wherever a 2x2 block sits on the diagonal, takes both roots of
    // that block's characteristic polynomial λ² − tr·λ + det instead of reading
    // the diagonal entry. Ordering follows the Schur form, not magnitude.
    // Usage: auto lambda = A.eigvals();   // lambda(k,0) is std::complex<double>
    Matrix<std::complex<double>> eigvals() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("eigvals: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("eigvals: matrix must be non-empty");
            int n = (int)rowSize;
            Matrix<std::complex<double>> out(n, 1);
            if (n == 1) {
                out(0, 0) = std::complex<double>(double(std::real(grid[0])), 0.0);
                return out;
            }

            auto [H, Qv] = schurDecomp();
            (void)Qv;
            for (int i = 0; i < n;) {
                if (i + 1 < n && isSchurBlock(H, n, i)) {
                    double a = H[i * n + i], b = H[i * n + (i + 1)];
                    double c = H[(i + 1) * n + i], d = H[(i + 1) * n + (i + 1)];
                    double half = (a + d) / 2.0;
                    double disc = half * half - (a * d - b * c);
                    if (disc >= 0.0) {  // block did not actually pair up
                        double rt = std::sqrt(disc);
                        out(i, 0) = std::complex<double>(half + rt, 0.0);
                        out(i + 1, 0) = std::complex<double>(half - rt, 0.0);
                    } else {
                        double im = std::sqrt(-disc);
                        out(i, 0) = std::complex<double>(half, im);
                        out(i + 1, 0) = std::complex<double>(half, -im);
                    }
                    i += 2;
                } else {
                    out(i, 0) = std::complex<double>(H[i * n + i], 0.0);
                    i++;
                }
            }
            return out;
        } catch (const std::exception& e) {
            std::cerr << "eigvals() error: " << e.what() << '\n';
            throw;
        }
    }

    // Singular value decomposition A = U * S * V^T. Works for any m×n.
    //   U — m×m orthogonal (left singular vectors)
    //   S — m×n diagonal, singular values in descending order, all >= 0
    //   V — n×n orthogonal (right singular vectors; note this returns V, not V^T)
    // The last major decomposition missing. Once it exists, norm(Two), cond(Two),
    // pinv() and a more robust rank() all fall out of it, as does PCA and
    // low-rank approximation.
    // Method: one-sided Jacobi, not the bidiagonalise-then-QR route. It
    // rotates pairs of columns of A until they are mutually orthogonal; at that
    // point A*V = U*S, so the column norms ARE the singular values and the
    // normalised columns ARE the left singular vectors. It is chosen here
    // because it computes the small singular values to high *relative*
    // accuracy, which is exactly what cond() and pinv() depend on, and because
    // it needs no shift strategy to converge.
    // Usage: auto [U, S, V] = A.svd();
    std::tuple<Matrix<double>, Matrix<double>, Matrix<double>> svd() const {
        static_assert(!is_complex<datatype>::value,
                      "svd: not yet implemented for complex datatypes. It would compile by\n"
                      "taking std::real() of each entry and silently discard the imaginary\n"
                      "part — a wrong answer, not a limitation. See roadmap item 1 "
                      "(work_t).");
        try {
            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("svd: matrix must be non-empty");

            int m = (int)rowSize, n = (int)colSize;

            // One-sided Jacobi orthogonalises COLUMNS, so it needs at least as
            // many rows as columns. For a wide matrix, factor the transpose and
            // read the result back: A = (Aᵀ)ᵀ = (U'S'V'ᵀ)ᵀ = V' S'ᵀ U'ᵀ.
            if (m < n) {
                Matrix<double> At(n, m);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        At(j, i) = double(std::real(grid[i * n + j]));
                auto [U2, S2, V2] = At.svd();
                return std::make_tuple(V2, S2.T(), U2);
            }

            // ---- Layout: everything is stored TRANSPOSED. ----
            // One-sided Jacobi does all its work on COLUMNS: every rotation
            // reads and writes two whole columns of W and two of V. In a
            // row-major array a column is strided by n, so each rotation
            // touched one cache line per element, nothing vectorised, and the
            // whole factorisation ran at ~1.1 GFLOP/s. LAPACK does not have
            // this problem because Fortran is column-major — dgesvj's columns
            // are contiguous by construction.
            //
            // Storing the TRANSPOSES gets the same property here: row p of Wt
            // is column p of W, so a rotation is now two contiguous runs and
            // the compiler can vectorise it. This is the single biggest change
            // in the function.
            std::vector<double> Wt(std::size_t(n) * m), Vt(std::size_t(n) * n, 0.0);
            for (long i = 0; i < m; i++)
                for (long j = 0; j < n; j++)
                    Wt[j * m + i] = double(std::real(grid[i * n + j]));
            for (long j = 0; j < n; j++)
                Vt[j * n + j] = 1.0;
            double* MATRIXCPP_RESTRICT wt = Wt.data();
            double* MATRIXCPP_RESTRICT vt = Vt.data();

            // ---- Cached column norms, as in dgesvj's sva[] array. ----
            // The version this replaced recomputed all THREE inner products
            // (p·p, q·q, p·q) for every one of the n(n-1)/2 pairs in every
            // sweep. Only the cross term p·q actually changes unpredictably:
            // the two squared norms can be carried forward through the
            // rotation exactly, because zeroing the cross term means
            //     alpha' = alpha - t*gamma,   beta' = beta + t*gamma
            // (substitute s = c*t and 1 - t^2 = 2*zeta*t into the 2x2 Gram
            // update to check this). That drops the per-pair cost from three
            // dot products to one — and a pair that is already converged now
            // costs one dot product and no rotation at all.
            std::vector<double> sva(n);
            auto refreshNorms = [&]() {
                for (long j = 0; j < n; j++) {
                    const double* MATRIXCPP_RESTRICT wj = wt + j * m;
                    double a0 = 0.0, a1 = 0.0;
                    long i = 0;
                    for (; i + 1 < m; i += 2) {
                        a0 += wj[i] * wj[i];
                        a1 += wj[i + 1] * wj[i + 1];
                    }
                    for (; i < m; i++)
                        a0 += wj[i] * wj[i];
                    sva[j] = a0 + a1;
                }
            };
            refreshNorms();

            const double eps = std::numeric_limits<double>::epsilon();
            const int maxSweeps = 60;

            // One pair of columns: orthogonalise them and record how far from
            // orthogonal they were. Returns that relative off-diagonal size so
            // the sweep can decide whether it has converged.
            auto processPair = [&](long p, long q) -> double {
                const double alpha = sva[p], beta = sva[q];
                if (alpha == 0.0 || beta == 0.0)
                    return 0.0;
                double* MATRIXCPP_RESTRICT wp = wt + p * m;
                double* MATRIXCPP_RESTRICT wq = wt + q * m;

                double g0 = 0.0, g1 = 0.0;
                long i = 0;
                for (; i + 1 < m; i += 2) {
                    g0 += wp[i] * wq[i];
                    g1 += wp[i + 1] * wq[i + 1];
                }
                for (; i < m; i++)
                    g0 += wp[i] * wq[i];
                const double gamma = g0 + g1;
                if (gamma == 0.0)
                    return 0.0;

                // Relative, not absolute: this is what buys the small
                // singular values their relative accuracy.
                const double conv = std::abs(gamma) / std::sqrt(alpha * beta);
                if (conv <= eps)
                    return conv;

                // Jacobi rotation zeroing the p-q inner product.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t =
                    (zeta >= 0.0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t), sn = c * t;
                for (long k = 0; k < m; k++) {
                    const double a = wp[k], b = wq[k];
                    wp[k] = c * a - sn * b;
                    wq[k] = sn * a + c * b;
                }
                double* MATRIXCPP_RESTRICT vp = vt + p * n;
                double* MATRIXCPP_RESTRICT vq = vt + q * n;
                for (long k = 0; k < n; k++) {
                    const double a = vp[k], b = vq[k];
                    vp[k] = c * a - sn * b;
                    vq[k] = sn * a + c * b;
                }
                sva[p] = alpha - t * gamma;
                sva[q] = beta + t * gamma;
                return conv;
            };

            // ---- Brent-Luk round-robin ("chess tournament") pair ordering. ----
            // The natural p<q double loop visits pairs in an order where
            // consecutive pairs share a column, so no two can be done at the
            // same time. The round-robin schedule instead splits a sweep into
            // np-1 ROUNDS of np/2 pairs each, where the pairs within a round
            // are column-disjoint by construction — seat the columns around a
            // circle, pair each seat with the one opposite, then rotate all but
            // one seat by a position. Every pair still occurs exactly once per
            // sweep, so this is a legitimate cyclic ordering with the same
            // convergence behaviour, but now a whole round runs in parallel.
            //
            // Being column-disjoint also makes the result DETERMINISTIC: no two
            // pairs in a round read or write the same column of W, V or sva, so
            // the answer does not depend on the thread count or the schedule.
            //
            // An odd number of columns gets one padding seat whose pairs are
            // skipped — the standard way to handle a bye in a round-robin.
            const long np = n + (n & 1);
            std::vector<long> ring(np);
            const long half = np / 2;

            // A sweep is np-1 rounds, so a 256-column matrix enters ~2000
            // parallel regions per factorisation. That makes the thread count
            // matter more than usual: too few and the cores idle, too many and
            // the ~360 ns region entry plus the barrier at the end of each
            // round costs more than the round does. Measured at n = 256:
            //
            //     threads   1     2     4     8    16    32
            //     ms       78.4  38.0  21.4  19.1  35.0  63.4
            //
            // so the useful range ends once a thread has fewer than ~8k
            // element-updates to do. Requesting that many threads and no more
            // tracks the optimum at every size. Below one thread's worth of
            // work the serial path runs with no OpenMP construct anywhere near
            // it — an `if` clause on the pragma is not enough, the runtime
            // still charges for evaluating it.
            const long roundWork = half * (m + n);
            const long WORK_PER_THREAD = 8192;
            (void)roundWork;
            (void)WORK_PER_THREAD;  // only read on the OpenMP path
#ifdef _OPENMP
            const int sweepThreads = (int)std::min<long>(
                std::max<long>(roundWork / WORK_PER_THREAD, 1), omp_get_max_threads());
#else
            const int sweepThreads = 1;
#endif
            const bool parallelSweep = (sweepThreads > 1);

            for (int sweep = 0; sweep < maxSweeps; sweep++) {
                double offMax = 0.0;
                for (long i = 0; i < np; i++)
                    ring[i] = i;
                for (long round = 0; round < np - 1; round++) {
#ifdef _OPENMP
                    if (parallelSweep) {
    #pragma omp parallel for schedule(static) reduction(max : offMax) num_threads(sweepThreads)
                        for (long i = 0; i < half; i++) {
                            const long a = ring[i], b = ring[np - 1 - i];
                            if (a >= n || b >= n)
                                continue;  // padding seat
                            const double conv = processPair(a < b ? a : b, a < b ? b : a);
                            if (conv > offMax)
                                offMax = conv;
                        }
                    } else
#endif
                    {
                        for (long i = 0; i < half; i++) {
                            const long a = ring[i], b = ring[np - 1 - i];
                            if (a >= n || b >= n)
                                continue;  // padding seat
                            const double conv = processPair(a < b ? a : b, a < b ? b : a);
                            if (conv > offMax)
                                offMax = conv;
                        }
                    }
                    // Rotate every seat but the first, so each column meets a
                    // different partner next round.
                    const long last = ring[np - 1];
                    for (long i = np - 1; i > 1; i--)
                        ring[i] = ring[i - 1];
                    ring[1] = last;
                }
                // The incremental norm update inside processPair is exact in
                // real arithmetic but drifts in floating point over many
                // sweeps. Recomputing once per sweep costs O(mn) against the
                // sweep's O(mn²), so it is free, and it keeps the convergence
                // test honest — dgesvj refreshes for the same reason.
                refreshNorms();
                if (offMax <= eps)
                    break;
            }
            (void)parallelSweep;

            auto w = [&](long i, long j) -> double& { return wt[j * m + i]; };

            // Column norms are the singular values; sort them descending and
            // carry the same permutation through the columns of W and V.
            std::vector<double> sigma(n, 0.0);
            for (long j = 0; j < n; j++)
                sigma[j] = std::sqrt(sva[j]);
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(
                order.begin(), order.end(), [&](int a, int b) { return sigma[a] > sigma[b]; });

            Matrix<double> S(m, n), Vm(n, n), U(m, m);
            for (int j = 0; j < n; j++) {
                S(j, j) = sigma[order[j]];
                // Vt is stored transposed, so column order[j] of V is a
                // contiguous row of Vt.
                const double* MATRIXCPP_RESTRICT vsrc = vt + (long)order[j] * n;
                for (long i = 0; i < n; i++)
                    Vm(int(i), j) = vsrc[i];
            }

            // Left singular vectors: the normalised columns of W, for every
            // singular value that is numerically non-zero.
            double sTol = double(std::max(m, n)) * eps * (n ? sigma[order[0]] : 0.0);
            int r = 0;
            while (r < n && sigma[order[r]] > sTol)
                r++;
            for (int j = 0; j < r; j++) {
                double sj = sigma[order[j]];
                for (int i = 0; i < m; i++)
                    U(i, j) = w(i, order[j]) / sj;
            }

            // U must come back m×m orthogonal, but only r of its columns are
            // determined by A. Fill the remaining ones with any orthonormal
            // completion: push each canonical basis vector through modified
            // Gram-Schmidt and keep the ones with a surviving component.
            // Twice — one pass loses orthogonality when the residual is small.
            int filled = r;
            for (int cand = 0; cand < m && filled < m; cand++) {
                std::vector<double> x(m, 0.0);
                x[cand] = 1.0;
                for (int pass = 0; pass < 2; pass++)
                    for (int j = 0; j < filled; j++) {
                        double dot = 0.0;
                        for (int i = 0; i < m; i++)
                            dot += U(i, j) * x[i];
                        for (int i = 0; i < m; i++)
                            x[i] -= dot * U(i, j);
                    }
                double nx = 0.0;
                for (int i = 0; i < m; i++)
                    nx += x[i] * x[i];
                nx = std::sqrt(nx);
                if (nx < 1e-8)
                    continue;  // already spanned; try the next one
                for (int i = 0; i < m; i++)
                    U(i, filled) = x[i] / nx;
                filled++;
            }

            return std::make_tuple(U, S, Vm);
        } catch (const std::exception& e) {
            std::cerr << "svd() error: " << e.what() << '\n';
            throw;
        }
    }

    // Computes A^(-1) by solving A * X = I. Requires a square, non-singular
    // matrix. The factor-and-substitute machinery this used to inline now lives
    // in solve(), so there is one copy of it to get right rather than two. The
    // flop count is unchanged: both versions factor once and substitute n times.
    //
    // If you are about to write A.inverse() * b, write A.solve(b) instead:
    // fewer flops and better conditioned. See the note on solve().
    Matrix<double> inverse() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("inverse: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            int n = (int)rowSize;
            Matrix<double> Id(n, n);
            for (int i = 0; i < n; i++)
                Id(i, i) = 1.0;
            return solve(Id);
        } catch (const std::exception& e) {
            std::cerr << "inverse() error: " << e.what() << '\n';
            throw;
        }
    }

    // Solves A * X = B for X. B may be a single column or several at once.
    //
    // Prefer this over A.inverse() * B: about a third of the flops and
    // numerically better conditioned. Forming an explicit inverse just to
    // multiply by it is the classic mistake this method exists to prevent.
    //
    // Square A       → exact solve via LU with partial pivoting.
    // rows > cols    → the least-squares solution argmin ||A*X - B||₂, via the
    //                  column-pivoted QR above: back-substitute R*y = (Qᵀ*B)
    //                  over the numerically non-zero diagonal of R, then undo
    //                  the column permutation. Rank-deficient input gives a
    //                  basic solution (free variables set to zero), not the
    //                  minimum-norm one — use pinv() if you need that.
    // rows < cols    → under-determined; throws, since "the" solution is not
    //                  unique. pinv() gives the minimum-norm one.
    //
    // Templated on B's element type so A.solve(b) works whatever b holds.
    // Usage: auto x = A.solve(b);
    template <typename dtB>
    Matrix<double> solve(const Matrix<dtB>& B) const {
        try {
            int m = (int)rowSize, n = (int)colSize, nrhs = (int)B.cols();
            if (B.rows() != rowSize)
                throw std::invalid_argument(
                    "solve: B must have one row per row of A — A is " + std::to_string(rowSize) +
                    "x" + std::to_string(colSize) + " but B is " + std::to_string(B.rows()) + "x" +
                    std::to_string(B.cols()));
            if (m == 0 || n == 0 || nrhs == 0)
                throw std::invalid_argument("solve: matrices must be non-empty");
            if (m < n)
                throw std::invalid_argument("solve: system is under-determined (" +
                                            std::to_string(m) + " equations, " + std::to_string(n) +
                                            " unknowns) — infinitely many "
                                            "solutions. Use pinv() for the minimum-norm one");

            // ── Square: LU with partial pivoting ────────────────────────────
            if (m == n) {
                if (n == 1) {  // luPacked() requires 2x2; handle the scalar case here
                    double a = double(std::real(grid[0]));
                    if (a == 0.0)
                        throw std::runtime_error("solve: 1x1 matrix is singular");
                    Matrix<double> X(1, nrhs);
                    for (int j = 0; j < nrhs; j++)
                        X(0, j) = double(std::real(B(0, j))) / a;
                    return X;
                }
                auto [packed, pivots] = luPacked();
                // The substitution itself lives in luSubstitute so that this
                // and Decomposition::solve share ONE implementation — the
                // parallel-over-right-hand-sides path in particular is worth
                // having in exactly one place.
                Matrix<double> Bd(n, nrhs);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < nrhs; j++)
                        Bd(i, j) = double(std::real(B(i, j)));
                return luSubstitute(packed, pivots, n, Bd);
            }

            // ── Over-determined: least squares via column-pivoted QR ────────
            // A*P = Q*R, so ||A x - b|| = ||R (Pᵀx) - Qᵀb||.
            auto [Q, R, P] = QR();
            Matrix<double> Bd(m, nrhs);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < nrhs; j++)
                    Bd(i, j) = double(std::real(B(i, j)));
            Matrix<double> QtB = Q.T() * Bd;

            double rTol =
                double(std::max(m, n)) * std::numeric_limits<double>::epsilon() * std::abs(R(0, 0));
            int rk = 0;
            while (rk < n && std::abs(R(rk, rk)) > rTol)
                rk++;

            Matrix<double> Y(n, nrhs);
            for (int col = 0; col < nrhs; col++) {
                // Back-substitute over the leading rk×rk block; the trailing
                // unknowns are the free variables and stay at zero.
                for (int i = rk - 1; i >= 0; i--) {
                    double acc = QtB(i, col);
                    for (int j = i + 1; j < rk; j++)
                        acc -= R(i, j) * Y(j, col);
                    Y(i, col) = acc / R(i, i);
                }
            }
            // x = P*y — undo the column permutation.
            Matrix<double> X(n, nrhs);
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++) {
                    double pik = double(std::real(P(i, k)));
                    if (pik == 0.0)
                        continue;
                    for (int col = 0; col < nrhs; col++)
                        X(i, col) += pik * Y(k, col);
                }
            return X;
        } catch (const std::exception& e) {
            std::cerr << "solve() error: " << e.what() << '\n';
            throw;
        }
    }

    // Moore-Penrose pseudo-inverse. Defined for any m×n, singular matrices
    // included, and coincides with inverse() when A is square and non-singular.
    // From the SVD: pinv(A) = V * S^+ * U^T, where S^+ inverts every singular
    // value above tol and leaves the rest at zero. Requires svd() first.
    Matrix<double> pinv(double tol = -1.0) const {
        try {
            if (rowSize == 0 || colSize == 0)
                throw std::invalid_argument("pinv: matrix must be non-empty");
            auto [U, S, V] = svd();
            int m = (int)rowSize, n = (int)colSize;
            int d = std::min(m, n);

            if (tol < 0.0)
                tol = double(std::max(m, n)) * std::numeric_limits<double>::epsilon() * S(0, 0);

            // pinv(A) = V · S⁺ · Uᵀ, where S⁺ inverts the singular values above
            // tol and leaves the rest at zero. Folding S⁺ into V first keeps
            // this to one matrix product.
            Matrix<double> VS(n, m);
            for (int j = 0; j < d; j++) {
                double sj = S(j, j);
                if (sj <= tol)
                    continue;
                double inv = 1.0 / sj;
                for (int i = 0; i < n; i++)
                    VS(i, j) = V(i, j) * inv;
            }
            return VS * U.T();
        } catch (const std::exception& e) {
            std::cerr << "pinv() error: " << e.what() << '\n';
            throw;
        }
    }

    // Adjugate (classical adjoint): the transpose of the cofactor matrix.
    // Satisfies A * adj(A) = det(A) * I, which is the identity behind the
    // textbook formula inverse(A) = adj(A) / det(A). Requires square A.
    // Mainly of symbolic/theoretical interest — inverse() and solve() are the
    // right tools numerically, since the cofactor route is O(n!) if done naively.
    Matrix<double> adjugate() const {
        try {
            if (rowSize != colSize)
                throw std::invalid_argument("adjugate: matrix must be square, got " +
                                            std::to_string(rowSize) + "x" +
                                            std::to_string(colSize));
            if (rowSize == 0)
                throw std::invalid_argument("adjugate: matrix must be non-empty");
            int n = (int)rowSize;

            // adj of a 1x1 is [1] by convention: A·adj(A) = det(A)·I = a·I.
            if (n == 1) {
                Matrix<double> out(1, 1);
                out(0, 0) = 1.0;
                return out;
            }

            Matrix<double> Ad(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Ad(i, j) = double(std::real(grid[i * n + j]));

            // Non-singular: adj(A) = det(A)·A⁻¹ directly from the identity
            // A·adj(A) = det(A)·I. Two O(n³) steps instead of n² minors.
            double scale = norm(NormType::Inf);
            double dTol = std::numeric_limits<double>::epsilon() *
                          std::pow(scale > 0.0 ? scale : 1.0, n) * 100.0;
            double d = Ad.det();
            if (std::abs(d) > dTol)
                return Ad.inverse() * d;

            // Singular: the identity above says nothing, so fall back to the
            // definition — adj(A)[j,i] = (-1)^(i+j) · det(A with row i, col j
            // deleted). O(n⁵), but it is the only route that stays correct here.
            Matrix<double> out(n, n);
            Matrix<double> minor(n - 1, n - 1);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    for (int r = 0, mr = 0; r < n; r++) {
                        if (r == i)
                            continue;
                        for (int c = 0, mc = 0; c < n; c++) {
                            if (c == j)
                                continue;
                            minor(mr, mc) = Ad(r, c);
                            mc++;
                        }
                        mr++;
                    }
                    double md = (n == 2) ? minor(0, 0) : minor.det();
                    out(j, i) = ((i + j) % 2 ? -1.0 : 1.0) * md;
                }
            return out;
        } catch (const std::exception& e) {
            std::cerr << "adjugate() error: " << e.what() << '\n';
            throw;
        }
    }

    // deconstructor
    ~Matrix() { release(); }

  private:
    // Lets Matrix<double> reach into Matrix<int>'s internals and vice versa,
    // which real()/imag() and the mixed-type paths need.
    template <typename>
    friend class Matrix;
    // Decomposition drives luSubstitute/luSubstituteT and luPacked directly:
    // it IS the factor-once path, so it works at the same level solve() does.
    template <typename>
    friend class Decomposition;

    // Tag type selecting the constructor below.
    struct uninit_t {};

    // Allocates without value-initialising. `new T[n]()` zero-fills, which is
    // pure waste when the very next thing the caller does is overwrite every
    // element — about 16% of the cost of an element-wise operation at
    // n = 2000. Only ever use this when the buffer is fully written before
    // it can be read.
    Matrix(long i, long j, uninit_t) {
        rowSize = i;
        colSize = j;
        allocRaw(i * j);  // deliberately not zeroed
    }

    // ── Small-buffer storage ────────────────────────────────────────
    // Matrices of up to SBO_CAPACITY elements live inside the object; only
    // larger ones touch the heap. Measured on this machine, a 2x2 A+B cost
    // 19 ns of which 19 ns was new/delete — the arithmetic was free and the
    // allocator was the entire operation. Eigen solves this with fixed-size
    // types; this is the runtime equivalent, and it costs one branch on the
    // sizing path.
    //
    // 16 elements covers every matrix up to 4x4, which is where small-matrix
    // work actually concentrates: 2x2 and 4x4 gates for the quantum-circuit
    // goal, 3x3 and 4x4 for geometry.
    static constexpr long SBO_CAPACITY = 16;

    // True when grid points at the inline buffer rather than the heap.
    // Compared against the pointer rather than recomputed from the sizes, so
    // it stays correct even while the sizes are mid-update.
    bool isInline() const { return grid == sbo; }

    // Points grid at storage for n elements WITHOUT initialising it.
    void allocRaw(long n) {
        if (n <= SBO_CAPACITY) {
            grid = sbo;
            return;
        }
        grid = rawAlloc(n);
    }

    // Points grid at zero-initialised storage for n elements.
    void allocZero(long n) {
        if (n <= SBO_CAPACITY) {
            grid = sbo;
            for (long i = 0; i < n; i++)
                grid[i] = datatype();
        } else {
            grid = rawAlloc(n);
            for (long i = 0; i < n; i++)
                grid[i] = datatype();
        }
    }

    // Storage, huge-page advice and the parallel helpers all live in
    // namespace mstore at the top of this file, so that Tensor is built on
    // exactly the same ones. These are the in-class spellings; the comments
    // explaining WHY each exists are on the definitions there.
    static constexpr bool RAW_STORAGE_OK = mstore::raw_storage_ok<datatype>;
    static datatype* rawAlloc(long n) { return mstore::rawAlloc<datatype>(n); }
    static void rawFree(datatype* p) { mstore::rawFree<datatype>(p); }
    static void adviseHuge(void* p, std::size_t b) { mstore::adviseHuge(p, b); }

    // Frees heap storage if that is what grid points at. Safe to call twice.
    void release() {
        if (grid && grid != sbo)
            rawFree(grid);
        grid = nullptr;
    }

    long rowSize;
    long colSize;
    datatype* grid;
    datatype sbo[SBO_CAPACITY];

    // True when rows i..i+1 of a real Schur form T are a genuine 2x2 block —
    // the signature of a complex-conjugate eigenvalue pair — rather than two
    // separate real eigenvalues. The QR iteration drives converged
    // sub-diagonal entries towards zero without always setting them exactly
    // to zero, so the test has to be relative to the neighbouring diagonal.
    static bool isSchurBlock(const std::vector<double>& T, int n, int i) {
        double sub = std::abs(T[(i + 1) * n + i]);
        double nbr = std::abs(T[i * n + i]) + std::abs(T[(i + 1) * n + (i + 1)]);
        return sub > std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0);
    }

    // Shared guard for the reductions, which have no meaningful answer on an
    // empty matrix and would otherwise read grid[0] off a null pointer.
    void requireNonEmpty(const char* who) const {
        if (rowSize * colSize == 0)
            throw std::invalid_argument(std::string(who) + "(): matrix is empty (" +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize) +
                                        ")");
    }

    // ── Mask machinery ─────────────────────────────────────────────────
    // (CmpOp, LogOp and mask_scalar_t are declared just above the public
    // mask API — a member TYPE has to be declared before a member
    // declaration can name it, unlike a member function body.)

    void requireMaskShape(const Matrix<bool>& mask) const {
        if (mask.rows() != rowSize || mask.cols() != colSize)
            throw std::invalid_argument(
                "logical index: the mask is (" + std::to_string(mask.rows()) + "x" +
                std::to_string(mask.cols()) + ") but the matrix is (" + std::to_string(rowSize) +
                "x" + std::to_string(colSize) + ")");
    }

    // Ordering needs <, which std::complex deliberately does not provide.
    // Instantiating one on a complex matrix is a compile-time error rather
    // than a silent choice of some arbitrary ordering — the same rule
    // min()/max()/argmin()/argmax() already follow. eq() and ne() are fine
    // for complex and do not go through this check.
    static void requireOrdered() {
        static_assert(!is_complex<datatype>::value,
                      "lt/gt/le/ge (and <, >, <=, >=) need an ordering, which std::complex\n"
                      "deliberately does not provide. Compare a component or a magnitude\n"
                      "instead — for example A.real().gt(0.0) or A.abs().gt(1.0).\n"
                      "eq() and ne() DO work for complex.");
    }

    template <typename A, typename B>
    static bool applyCmp(CmpOp op, const A& x, const B& y) {
        switch (op) {
            case CmpOp::LT:
                return x < y;
            case CmpOp::GT:
                return x > y;
            case CmpOp::LE:
                return x <= y;
            case CmpOp::GE:
                return x >= y;
            case CmpOp::EQ:
                return x == y;
            default:
                return !(x == y);
        }
    }

    Matrix<bool> compare(const Matrix& M, const char* who, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE)
            requireOrdered();
        requireSameShape(M, who);
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = applyCmp(op, a[i], b[i]); });
        return out;
    }

    template <typename Scalar>
    Matrix<bool> compareScalar(const Scalar& v, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE)
            requireOrdered();
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype rhs = datatype(v);
        forEachIndex(rowSize * colSize, [=](long i) { r[i] = applyCmp(op, a[i], rhs); });
        return out;
    }

    Matrix<bool> logical(const Matrix& M, const char* who, LogOp op) const {
        requireSameShape(M, who);
        Matrix<bool> out(rowSize, colSize, typename Matrix<bool>::uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = M.grid;
        bool* MATRIXCPP_RESTRICT r = out.grid;
        const datatype zero = datatype(0);
        forEachIndex(rowSize * colSize, [=](long i) {
            const bool x = !(a[i] == zero), y = !(b[i] == zero);
            r[i] = (op == LogOp::AND) ? (x && y) : (op == LogOp::OR) ? (x || y) : (x != y);
        });
        return out;
    }

    // Per-axis any()/all(). addcol=0 walks columns, addcol=1 walks rows,
    // matching sum(bool).
    Matrix<bool> reduceLogical(bool addcol, bool wantAny) const {
        const datatype zero = datatype(0);
        if (!addcol) {
            Matrix<bool> out(1, colSize);
            for (long j = 0; j < colSize; j++) {
                bool acc = !wantAny;
                for (long i = 0; i < rowSize; i++) {
                    const bool v = !(grid[i * colSize + j] == zero);
                    if (wantAny) {
                        if (v) {
                            acc = true;
                            break;
                        }
                    } else {
                        if (!v) {
                            acc = false;
                            break;
                        }
                    }
                }
                out(0, int(j)) = acc;
            }
            return out;
        }
        Matrix<bool> out(rowSize, 1);
        for (long i = 0; i < rowSize; i++) {
            bool acc = !wantAny;
            for (long j = 0; j < colSize; j++) {
                const bool v = !(grid[i * colSize + j] == zero);
                if (wantAny) {
                    if (v) {
                        acc = true;
                        break;
                    }
                } else {
                    if (!v) {
                        acc = false;
                        break;
                    }
                }
            }
            out(int(i), 0) = acc;
        }
        return out;
    }

    // Default tolerance for the structure predicates: relative to the
    // largest row sum, so it scales with the matrix rather than assuming
    // entries are around 1.
    double structureTol(double tol) const {
        if (tol >= 0.0)
            return tol;
        const double scale = (rowSize * colSize > 0) ? norm(NormType::Inf) : 1.0;
        return std::numeric_limits<double>::epsilon() * 100.0 * (scale > 0.0 ? scale : 1.0);
    }

    // Default cut-off for "this singular value is numerically zero", the
    // same rule rank() uses: max(m,n) * eps * sigma_max.
    double svdTol(const Matrix<double>& S, double tol) const {
        if (tol >= 0.0)
            return tol;
        const double smax = (rowSize > 0 && colSize > 0) ? S(0, 0) : 0.0;
        return double(std::max(rowSize, colSize)) * std::numeric_limits<double>::epsilon() * smax;
    }

    // The rounding and real-valued maps have no complex meaning — there is no
    // "largest integer below" a complex number — so instantiating one is a
    // compile-time error rather than a silent choice. sign() and angle() DO
    // have complex meanings and do not go through this.
    static void requireRoundable(const char* who) {
        (void)who;
        static_assert(!is_complex<datatype>::value,
                      "floor/ceil/round/fix/mod/rem/atan2/hypot/expm1/log1p have no complex\n"
                      "meaning. Apply them to .real() / .imag() / .abs(), or use sign() and\n"
                      "angle(), which are defined for complex.");
    }

    // Two-operand element-wise map, the binary sibling of mapElems.
    template <class F>
    Matrix zipElems(const Matrix& B, F fn) const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const datatype* MATRIXCPP_RESTRICT a = grid;
        const datatype* MATRIXCPP_RESTRICT b = B.grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
        const long total = rowSize * colSize;
#ifdef _OPENMP
        if (total >= MAP_MIN_WORK) {
    #pragma omp parallel for schedule(static)
            for (long i = 0; i < total; i++)
                r[i] = fn(a[i], b[i]);
            return ans;
        }
#endif
        for (long i = 0; i < total; i++)
            r[i] = fn(a[i], b[i]);
        return ans;
    }

    // Solves L*U*x = P*b for every column of B, given the packed factors
    // luPacked() produces. Extracted from solve() so that solve(),
    // inverse() and Decomposition all drive the same code.
    static Matrix<double> luSubstitute(const std::vector<double>& packed,
                                       const std::vector<int>& pivots,
                                       int n,
                                       const Matrix<double>& B) {
        const int nrhs = (int)B.cols();
        Matrix<double> X(n, nrhs);
        const double* MATRIXCPP_RESTRICT LU = packed.data();
        double* MATRIXCPP_RESTRICT Xg = X.grid;
        const double* MATRIXCPP_RESTRICT Bg = B.grid;

        // One right-hand side: permute, forward-substitute through L,
        // back-substitute through U. Both substitutions walk a row of the
        // packed factor, which is contiguous, and four accumulators keep the
        // dot product at FMA throughput rather than FMA latency.
        auto solveOne = [&](int col) {
            std::vector<double> bv((std::size_t)n);
            double* MATRIXCPP_RESTRICT b = bv.data();
            for (int i = 0; i < n; i++)
                b[i] = Bg[(std::size_t)i * nrhs + col];
            for (int i = 0; i < n; i++)
                if (pivots[(std::size_t)i] != i)
                    std::swap(b[i], b[pivots[(std::size_t)i]]);
            for (int i = 0; i < n; i++) {
                const double* MATRIXCPP_RESTRICT row = LU + (std::size_t)i * n;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                int j = 0;
                for (; j + 3 < i; j += 4) {
                    a0 += row[j] * b[j];
                    a1 += row[j + 1] * b[j + 1];
                    a2 += row[j + 2] * b[j + 2];
                    a3 += row[j + 3] * b[j + 3];
                }
                for (; j < i; j++)
                    a0 += row[j] * b[j];
                b[i] -= (a0 + a1) + (a2 + a3);
            }
            for (int i = n - 1; i >= 0; i--) {
                const double* MATRIXCPP_RESTRICT row = LU + (std::size_t)i * n;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                int j = i + 1;
                for (; j + 3 < n; j += 4) {
                    a0 += row[j] * b[j];
                    a1 += row[j + 1] * b[j + 1];
                    a2 += row[j + 2] * b[j + 2];
                    a3 += row[j + 3] * b[j + 3];
                }
                for (; j < n; j++)
                    a0 += row[j] * b[j];
                b[i] = (b[i] - ((a0 + a1) + (a2 + a3))) / row[i];
            }
            for (int i = 0; i < n; i++)
                Xg[(std::size_t)i * nrhs + col] = b[i];
        };

        // Every right-hand side is solved against the SAME factors and writes
        // a different column of X, so the columns are independent. LAPACK
        // exploits this differently — dgetrs hands all of them to dtrsm at
        // once for a level-3 blocked triangular solve — but the parallelism
        // is the same, and it is what makes inverse() (solve against n
        // columns of the identity) worth anything: one column at a time, a
        // 512x512 inverse cost 83 ms against 7.6 ms for a single solve.
        const long solveWork = (long)nrhs * n * n;
        (void)solveWork;  // only read on the OpenMP path
#ifdef _OPENMP
        if (solveWork >= PARALLEL_MIN_WORK && nrhs > 1) {
    #pragma omp parallel for schedule(static)
            for (int col = 0; col < nrhs; col++)
                solveOne(col);
            return X;
        }
#endif
        for (int col = 0; col < nrhs; col++)
            solveOne(col);
        return X;
    }

    // Solves the TRANSPOSED system, Uᵀ Lᵀ y = b then undoes the permutation.
    // Needed by the condition estimator, which alternates between A and Aᵀ.
    static Matrix<double> luSubstituteT(const std::vector<double>& packed,
                                        const std::vector<int>& pivots,
                                        int n,
                                        const Matrix<double>& B) {
        const int nrhs = (int)B.cols();
        Matrix<double> X(n, nrhs);
        const double* LU = packed.data();
        for (int col = 0; col < nrhs; col++) {
            std::vector<double> b((std::size_t)n);
            for (int i = 0; i < n; i++)
                b[(std::size_t)i] = B(i, col);
            // Uᵀ is lower triangular with U's diagonal.
            for (int i = 0; i < n; i++) {
                double acc = b[(std::size_t)i];
                for (int j = 0; j < i; j++)
                    acc -= LU[(std::size_t)j * n + i] * b[(std::size_t)j];
                b[(std::size_t)i] = acc / LU[(std::size_t)i * n + i];
            }
            // Lᵀ is upper triangular with a unit diagonal.
            for (int i = n - 1; i >= 0; i--) {
                double acc = b[(std::size_t)i];
                for (int j = i + 1; j < n; j++)
                    acc -= LU[(std::size_t)j * n + i] * b[(std::size_t)j];
                b[(std::size_t)i] = acc;
            }
            // The row swaps were applied to b in the forward direction, so
            // they come off the answer in reverse.
            for (int i = n - 1; i >= 0; i--)
                if (pivots[(std::size_t)i] != i)
                    std::swap(b[(std::size_t)i], b[(std::size_t)pivots[(std::size_t)i]]);
            for (int i = 0; i < n; i++)
                X(i, col) = b[(std::size_t)i];
        }
        return X;
    }

    // Shared body of cumsum and cumprod. The scan runs along the axis the
    // matching reduction would collapse, so cumsum(false) accumulates down
    // each column exactly as sum(false) totals it.
    Matrix scan(bool addcol, bool adding) const {
        Matrix out(rowSize, colSize, uninit_t{});
        if (rowSize == 0 || colSize == 0)
            return out;
        if (!addcol) {
            for (long j = 0; j < colSize; j++)
                out.grid[j] = grid[j];
            for (long i = 1; i < rowSize; i++)
                for (long j = 0; j < colSize; j++) {
                    const datatype prev = out.grid[(i - 1) * colSize + j];
                    const datatype here = grid[i * colSize + j];
                    out.grid[i * colSize + j] = adding ? prev + here : prev * here;
                }
            return out;
        }
        for (long i = 0; i < rowSize; i++) {
            out.grid[i * colSize] = grid[i * colSize];
            for (long j = 1; j < colSize; j++) {
                const datatype prev = out.grid[i * colSize + j - 1];
                const datatype here = grid[i * colSize + j];
                out.grid[i * colSize + j] = adding ? prev + here : prev * here;
            }
        }
        return out;
    }

    // nth_element twice rather than a full sort: O(n) instead of O(n log n),
    // and the second call only has to look at the upper half.
    static double medianOf(std::vector<datatype>& buf) {
        const std::size_t n = buf.size();
        const std::size_t mid = n / 2;
        std::nth_element(buf.begin(), buf.begin() + (long)mid, buf.end());
        const double hi = double(buf[mid]);
        if (n % 2 == 1)
            return hi;
        const double lo = double(*std::max_element(buf.begin(), buf.begin() + (long)mid));
        return 0.5 * (lo + hi);
    }

    // Shared dimension check for the element-wise operations.
    void requireSameShape(const Matrix& M, const char* who) const {
        if (rowSize != M.rowSize || colSize != M.colSize)
            throw std::invalid_argument(std::string(who) +
                                        "(): element-wise operations need identical "
                                        "shapes, got (" +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize) +
                                        ") and (" + std::to_string(M.rowSize) + "x" +
                                        std::to_string(M.colSize) + ")");
    }

    // Computes the real Schur decomposition of this matrix.
    // Returns {T_flat, Q_flat} where A = Q * T * Q^T,
    // T is upper (quasi-)triangular and Q is orthogonal (both n×n, row-major
    // double). Public so that the free pow() function can access it; also useful
    // on its own.
    //
    // NOTE — "quasi-triangular" is the word carrying all the weight here.
    // T is triangular except for 2x2 blocks on the diagonal, one per pair of
    // complex-conjugate eigenvalues. Every caller in this header currently
    // assumes those blocks do not exist: eig() reads the bare diagonal, and
    // pow()/log() sidestep the issue by demanding positive eigenvalues.
    // Any block-aware consumer (a correct eig, exp(A), the matrix trig
    // functions) needs a Parlett recurrence that solves a small Sylvester
    // equation per block rather than dividing scalars. This is the single
    // change that unblocks the most of the roadmap at the top of the file.
  public:
    std::pair<std::vector<double>, std::vector<double>> schurDecomp() const {
        int n = (int)rowSize;
        std::vector<double> H(n * n), Q(n * n, 0.0);
        for (int k = 0; k < n * n; k++)
            H[k] = double(grid[k]);
        for (int i = 0; i < n; i++)
            Q[i * n + i] = 1.0;
        auto h = [&](int i, int j) -> double& { return H[i * n + j]; };
        auto qv = [&](int i, int j) -> double& { return Q[i * n + j]; };

        // ── Hessenberg reduction ──────────────────────────────────────────
        for (int k = 0; k < n - 2; k++) {
            double xn = 0.0;
            for (int i = k + 1; i < n; i++)
                xn += h(i, k) * h(i, k);
            xn = std::sqrt(xn);
            if (xn < 1e-14)
                continue;
            int sz = n - k - 1;
            double alpha = (h(k + 1, k) >= 0.0 ? -1.0 : 1.0) * xn;
            std::vector<double> v(sz);
            for (int i = 0; i < sz; i++)
                v[i] = h(k + 1 + i, k);
            v[0] -= alpha;
            double vv = 0.0;
            for (double vi : v)
                vv += vi * vi;
            if (vv < 1e-28)
                continue;
            double tau = 2.0 / vv;
            for (int j = k; j < n; j++) {
                double s = 0.0;
                for (int i = 0; i < sz; i++)
                    s += v[i] * h(k + 1 + i, j);
                for (int i = 0; i < sz; i++)
                    h(k + 1 + i, j) -= tau * v[i] * s;
            }
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = 0; j < sz; j++)
                    s += h(i, k + 1 + j) * v[j];
                for (int j = 0; j < sz; j++)
                    h(i, k + 1 + j) -= tau * v[j] * s;
            }
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = 0; j < sz; j++)
                    s += qv(i, k + 1 + j) * v[j];
                for (int j = 0; j < sz; j++)
                    qv(i, k + 1 + j) -= tau * v[j] * s;
            }
        }

        // ── QR iteration with Wilkinson shift and deflation ───────────────
        // Two things this loop has to handle that a plain "iterate until the
        // sub-diagonal vanishes" version does not:
        //
        //  1. A complex-conjugate eigenvalue pair NEVER drives its
        //     sub-diagonal entry to zero. That is not a convergence failure,
        //     it is the definition of a real Schur form: the pair lives in a
        //     2x2 block. So once a trailing 2x2 is isolated it is deflated as
        //     a block rather than iterated on. Without this, a plain rotation
        //     matrix spins until the step limit and throws.
        //  2. An isolated 2x2 whose eigenvalues turn out to be REAL is split
        //     by one Givens rotation, so it does not then masquerade as a
        //     complex pair to eig()/eigvals() downstream.
        //
        // A block that stalls also gets a periodic exceptional shift — the
        // standard escape from the matrices a Wilkinson shift cycles on.
        const double eps = std::numeric_limits<double>::epsilon();

        // The QR iteration below updates COLUMNS k and k+1 of Q on every
        // rotation. In row-major storage a column walk strides by a whole row,
        // so that is one cache miss per row, n of them per rotation — and it
        // was the dominant cost of the whole routine once the O(n^4) work was
        // gone. Transposing Q once here turns each of those column updates
        // into two contiguous row updates; it is transposed back at the end.
        std::vector<double> Qt(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Qt[j * n + i] = Q[i * n + j];

        // Relative test for "this sub-diagonal entry has converged to zero".
        auto negligible = [&](int i) {
            double nbr = std::abs(h(i - 1, i - 1)) + std::abs(h(i, i));
            if (nbr == 0.0)
                nbr = 1.0;
            return std::abs(h(i, i - 1)) <= eps * nbr;
        };

        // Triangularises the 2x2 block at rows/cols k, k+1 if its eigenvalues
        // are real; leaves it intact if they are a conjugate pair. The
        // rotation is chosen so its first column is an eigenvector, which
        // sends the sub-diagonal entry to exactly zero.
        auto split2x2 = [&](int k) {
            double a = h(k, k), b = h(k, k + 1);
            double c = h(k + 1, k), d = h(k + 1, k + 1);
            if (c == 0.0)
                return;
            double half = (a + d) / 2.0;
            double disc = half * half - (a * d - b * c);
            if (disc < 0.0)
                return;  // genuine complex pair — keep the block

            // Take the root of larger magnitude directly and the other from
            // the product, so the small one does not lose digits to cancellation.
            double rt = std::sqrt(disc);
            double lam = half + (half >= 0.0 ? rt : -rt);
            if (lam == 0.0)
                lam = half;

            double x = lam - d, y = c;  // eigenvector direction for lam
            if (std::abs(x) + std::abs(y) < eps * (std::abs(a) + std::abs(d))) {
                x = b;
                y = lam - a;  // the other spelling, when that one degenerates
            }
            double r = std::hypot(x, y);
            if (r == 0.0)
                return;
            double cs = x / r, sn = y / r;

            for (int j = 0; j < n; j++) {  // rows: H <- G^T H
                double t1 = h(k, j), t2 = h(k + 1, j);
                h(k, j) = cs * t1 + sn * t2;
                h(k + 1, j) = -sn * t1 + cs * t2;
            }
            for (int i = 0; i < n; i++) {  // cols: H <- H G
                double t1 = h(i, k), t2 = h(i, k + 1);
                h(i, k) = cs * t1 + sn * t2;
                h(i, k + 1) = -sn * t1 + cs * t2;
            }
            {  // accumulate: Q <- Q G, on Qt
                double* qk = &Qt[k * n];
                double* qn = &Qt[(k + 1) * n];
                for (int i = 0; i < n; i++) {
                    double t1 = qk[i], t2 = qn[i];
                    qk[i] = cs * t1 + sn * t2;
                    qn[i] = -sn * t1 + cs * t2;
                }
            }
            h(k + 1, k) = 0.0;
        };

        int maxSteps = 60 * n, ihi = n - 1, itersOnBlock = 0;
        // Rotation scratch, sized once. The previous code allocated a fresh
        // std::vector per reflector per sweep.
        std::vector<double> csv(n > 0 ? n : 1), snv(n > 0 ? n : 1);
        while (ihi >= 1) {
            int ilo = ihi;
            while (ilo > 0 && !negligible(ilo))
                ilo--;

            if (ilo == ihi) {  // isolated 1x1: real eigenvalue
                if (ihi > 0)
                    h(ihi, ihi - 1) = 0.0;
                ihi--;
                itersOnBlock = 0;
                continue;
            }
            if (ilo == ihi - 1) {  // isolated 2x2: deflate as a block
                if (ilo > 0)
                    h(ilo, ilo - 1) = 0.0;
                split2x2(ilo);
                ihi -= 2;
                itersOnBlock = 0;
                continue;
            }
            if (maxSteps-- < 0)
                throw std::runtime_error("schurDecomp: QR iteration did not converge after " +
                                         std::to_string(60 * n) + " steps");

            double a = h(ihi - 1, ihi - 1), b = h(ihi - 1, ihi);
            double c = h(ihi, ihi - 1), d = h(ihi, ihi);
            double tr2 = (a + d) / 2.0, disc = tr2 * tr2 - (a * d - b * c);
            double sigma =
                (disc >= 0.0)
                    ? (std::abs(tr2 + std::sqrt(disc) - d) < std::abs(tr2 - std::sqrt(disc) - d)
                           ? tr2 + std::sqrt(disc)
                           : tr2 - std::sqrt(disc))
                    : d;
            // Exceptional shift: break out of a cycling block every 10 steps.
            if (++itersOnBlock % 10 == 0)
                sigma = std::abs(h(ihi, ihi - 1)) + std::abs(h(ihi - 1, ihi - 2));
            // ── Shifted QR step, Givens instead of Householder ──────────
            //
            // This is the step that dominates the whole routine, and the
            // Hessenberg structure is what makes it cheap. Column k has
            // exactly ONE entry below the diagonal — everything from row k+2
            // down is already zero — so zeroing it needs a 2x2 rotation
            // touching two rows, not a Householder reflector spanning
            // rows k..ihi.
            //
            // The reflector version this replaced built a vector of length
            // (ihi-k+1) and applied it across that many rows, making each
            // sweep O(n * sz^2) and the whole decomposition O(n^4). Rotations
            // bring the sweep to O(n * sz) and the decomposition to O(n^3),
            // which is what LAPACK's dlahqr does. It also drops the
            // std::vector allocated per reflector per sweep.
            //
            // Factor H - sigma*I = Q*R with Q = (G_ilo ... G_{ihi-1})^T,
            // then form H <- R*Q + sigma*I.
            for (int i = ilo; i <= ihi; i++)
                h(i, i) -= sigma;

            const int sz = ihi - ilo;
            if ((int)csv.size() < sz) {
                csv.resize(sz);
                snv.resize(sz);
            }

            // Left sweep: G_k zeroes h(k+1,k). By the time we reach column k,
            // G_{k-1} has already zeroed h(k,k-1), so starting at column k
            // leaves the earlier columns untouched and Hessenberg intact.
            for (int k = ilo; k < ihi; k++) {
                double a = h(k, k), b = h(k + 1, k);
                double r = std::hypot(a, b);
                double c, sn2;
                if (r == 0.0) {
                    c = 1.0;
                    sn2 = 0.0;
                } else {
                    c = a / r;
                    sn2 = b / r;
                }
                csv[k - ilo] = c;
                snv[k - ilo] = sn2;
                for (int j = k; j < n; j++) {
                    double t1 = h(k, j), t2 = h(k + 1, j);
                    h(k, j) = c * t1 + sn2 * t2;
                    h(k + 1, j) = -sn2 * t1 + c * t2;
                }
            }

            // Right sweep: H <- R * G_k^T, and accumulate Q <- Q * G_k^T.
            // Columns k and k+1 of R are zero below row k+1, so the update
            // stops at row k+2 — rows above ilo are still included, since
            // they hold the off-block part of H.
            for (int k = ilo; k < ihi; k++) {
                const double c = csv[k - ilo], sn2 = snv[k - ilo];
                const int iMax = std::min(ihi, k + 2);
                for (int i = 0; i <= iMax; i++) {
                    double t1 = h(i, k), t2 = h(i, k + 1);
                    h(i, k) = c * t1 + sn2 * t2;
                    h(i, k + 1) = -sn2 * t1 + c * t2;
                }
                // Contiguous, because Qt holds Q transposed — see the note
                // where Qt is built.
                double* qk = &Qt[k * n];
                double* qn = &Qt[(k + 1) * n];
                for (int i = 0; i < n; i++) {
                    double t1 = qk[i], t2 = qn[i];
                    qk[i] = c * t1 + sn2 * t2;
                    qn[i] = -sn2 * t1 + c * t2;
                }
            }

            for (int i = ilo; i <= ihi; i++)
                h(i, i) += sigma;
            // Deflation is decided by the scan at the top of the loop, which
            // also recognises the isolated-2x2 case this used to miss.
        }

        for (int i = 0; i < n; i++)  // undo the transpose
            for (int j = 0; j < n; j++)
                Q[i * n + j] = Qt[j * n + i];
        return {H, Q};
    }

    // Shared Doolittle factorisation used by both LU() and det().
    // Returns {packedData, pivotVec} — packed lower/upper triangle + row-swap
    // record.
    std::pair<std::vector<double>, std::vector<int>> luPacked() const {
        if (rowSize != colSize)
            throw std::invalid_argument("LU: matrix must be square, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));
        if (rowSize < 2)
            throw std::invalid_argument("LU: matrix must be at least 2x2, got " +
                                        std::to_string(rowSize) + "x" + std::to_string(colSize));

        int n = (int)rowSize;
        std::vector<double> packed(n * n);
        for (int k = 0; k < n * n; k++)
            packed[k] = double(grid[k]);

        auto pat = [&](int i, int j) -> double& { return packed[i * n + j]; };

        std::vector<int> pivotVec(n);
        for (int i = 0; i < n; i++)
            pivotVec[i] = i;

        for (int k = 0; k < n; k++) {
            int maxRow = k;
            double maxVal = std::abs(pat(k, k));
            for (int i = k + 1; i < n; i++) {
                double v = std::abs(pat(i, k));
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = i;
                }
            }
            if (maxVal == 0.0)
                throw std::runtime_error("LU: zero pivot in column " + std::to_string(k) +
                                         " — matrix is singular");
            if (maxRow != k)
                for (int j = 0; j < n; j++)
                    std::swap(pat(k, j), pat(maxRow, j));
            pivotVec[k] = maxRow;
            for (int i = k + 1; i < n; i++)
                pat(i, k) /= pat(k, k);
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    pat(i, j) -= pat(i, k) * pat(k, j);
        }
        return {packed, pivotVec};
    }

    // Crossover point: matrices smaller than this use naive O(n³) multiplication.
    // 64 is a common empirical choice — below this the Strassen overhead
    // outweighs the asymptotic benefit.
    static constexpr long STRASSEN_THRESHOLD = 64;

    // Tuning constants and the parallel drivers — see namespace mstore.
    static constexpr long PARALLEL_MIN_WORK = mstore::PARALLEL_MIN_WORK;
    static constexpr long MAP_MIN_WORK = mstore::MAP_MIN_WORK;
    static constexpr long ELEMENTWISE_MIN_WORK = mstore::ELEMENTWISE_MIN_WORK;
    static int memoryThreads() { return mstore::memoryThreads(); }
    template <class F>
    static void forEachIndex(long total, F body) {
        mstore::forEachIndex(total, body);
    }

    // Shared body of every element-wise map (exp, ln, sin, ...). NumPy's
    // ufuncs are SIMD but strictly SINGLE-THREADED, so threading a
    // transcendental map is a gap that is not available to it: elem_ln at
    // n=2000 went 11.6 -> 0.62 ms. __restrict on both pointers as well,
    // without which the destination is assumed to alias the source.
    template <class F>
    Matrix mapElems(F fn) const {
        Matrix ans(rowSize, colSize, uninit_t{});
        const long total = rowSize * colSize;
        const datatype* MATRIXCPP_RESTRICT a = grid;
        datatype* MATRIXCPP_RESTRICT r = ans.grid;
#ifdef _OPENMP
        if (total >= MAP_MIN_WORK) {
    #pragma omp parallel for schedule(static)
            for (long i = 0; i < total; i++)
                r[i] = fn(a[i]);
            return ans;
        }
#endif
        for (long i = 0; i < total; i++)
            r[i] = fn(a[i]);
        return ans;
    }

    // Returns the smallest power of 2 >= n.
    static long nextPow2(long n) {
        long p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    // Cache-blocked, OpenMP-parallelised matrix multiplication.
    // Tile size: 64 elements × sizeof(datatype) fits comfortably in L1 cache.
    // Each outer ii-tile is an independent OpenMP task, so cores don't share
    // work. Used as the base case for Strassen-Winograd and for rectangular
    // matrices.
    static Matrix naiveMul(const Matrix& A, const Matrix& B) {
        // The kernel itself is mstore::gemm — see there for the blocking, the
        // restrict/by-value capture that took it from 64 to 175 GFLOP/s, and
        // why the serial path branches around the OpenMP construct entirely.
        // Sharing it means Tensor's contractions use the same one.
        Matrix ans(A.rowSize, B.colSize, uninit_t{});
        mstore::gemm(A.grid, B.grid, ans.grid, A.rowSize, B.colSize, A.colSize);
        return ans;
    }

    // Extract the h×h sub-block of M starting at (r0, c0).
    static Matrix subBlock(const Matrix& M, long r0, long c0, long h) {
        Matrix out(h, h, uninit_t{});  // fully overwritten below
        for (long i = 0; i < h; i++) {
            const datatype* src = M.grid + (r0 + i) * M.colSize + c0;
            datatype* dst = out.grid + i * h;
            for (long j = 0; j < h; j++)
                dst[j] = src[j];
        }
        return out;
    }

    // Write block src (h×h) into dst at (r0, c0).
    static void setBlock(Matrix& dst, const Matrix& src, long r0, long c0, long h) {
        for (long i = 0; i < h; i++)
            for (long j = 0; j < h; j++)
                dst.grid[(r0 + i) * dst.colSize + (c0 + j)] = src.grid[i * h + j];
    }

    // Element-wise addition of two same-size matrices.
    static Matrix addMat(const Matrix& A, const Matrix& B) {
        Matrix out(A.rowSize, A.colSize, uninit_t{});
        const long total = A.rowSize * A.colSize;
        for (long k = 0; k < total; k++)
            out.grid[k] = A.grid[k] + B.grid[k];
        return out;
    }

    // Element-wise subtraction.
    static Matrix subMat(const Matrix& A, const Matrix& B) {
        Matrix out(A.rowSize, A.colSize, uninit_t{});
        const long total = A.rowSize * A.colSize;
        for (long k = 0; k < total; k++)
            out.grid[k] = A.grid[k] - B.grid[k];
        return out;
    }

    // Strassen-Winograd algorithm.
    //
    // Requires A and B to be square with size = power of 2.
    // Recursively splits into h×h quadrants and computes 7 recursive products
    // (versus 8 for standard multiplication), which is what buys the
    // O(n^2.807) exponent. Winograd's variant additionally cuts the additions
    // from 18 to 15 by sharing the auxiliary sums below.
    //
    //   S1 = A21 + A22        T1 = B12 - B11
    //   S2 = S1  - A11        T2 = B22 - T1
    //   S3 = A11 - A21        T3 = B22 - B12
    //   S4 = A12 - S2         T4 = T2  - B21
    //
    //   P1 = A11 * B11        P2 = A12 * B21
    //   P3 = S4  * B22        P4 = A22 * T4
    //   P5 = S1  * T1         P6 = S2  * T2
    //   P7 = S3  * T3
    //
    //   U1 = P1 + P2          U5 = U4 + P3
    //   U2 = P1 + P6          U6 = U3 - P4
    //   U3 = U2 + P7          U7 = U3 + P5
    //   U4 = U2 + P5
    //
    //   C11 = U1   C12 = U5   C21 = U6   C22 = U7
    //
    // Both T4 and U7 are easy to get subtly wrong, and a wrong version still
    // returns plausible-looking numbers of the right magnitude — the error is
    // only visible against a reference product. validate.cpp checks this path
    // against a naive multiply at every size class that reaches it.
    static Matrix strassenWinograd(const Matrix& A, const Matrix& B) {
        long n = A.rowSize;

        // Base case: fall back to naive multiplication
        if (n <= STRASSEN_THRESHOLD)
            return naiveMul(A, B);

        long h = n / 2;

        // Partition A into quadrants
        Matrix A11 = subBlock(A, 0, 0, h);
        Matrix A12 = subBlock(A, 0, h, h);
        Matrix A21 = subBlock(A, h, 0, h);
        Matrix A22 = subBlock(A, h, h, h);

        // Partition B into quadrants
        Matrix B11 = subBlock(B, 0, 0, h);
        Matrix B12 = subBlock(B, 0, h, h);
        Matrix B21 = subBlock(B, h, 0, h);
        Matrix B22 = subBlock(B, h, h, h);

        // Winograd auxiliary sums (saves additions vs plain Strassen)
        Matrix S1 = addMat(A21, A22);  // A21 + A22
        Matrix S2 = subMat(S1, A11);   // S1  - A11
        Matrix S3 = subMat(A11, A21);  // A11 - A21
        Matrix S4 = subMat(A12, S2);   // A12 - S2
        Matrix T1 = subMat(B12, B11);  // B12 - B11
        Matrix T2 = subMat(B22, T1);   // B22 - T1
        Matrix T3 = subMat(B22, B12);  // B22 - B12
        Matrix T4 = subMat(T2, B21);   // T2 - B21

        // 7 recursive multiplications
        Matrix P1 = strassenWinograd(A11, B11);
        Matrix P2 = strassenWinograd(A12, B21);
        Matrix P3 = strassenWinograd(S4, B22);
        Matrix P4 = strassenWinograd(A22, T4);
        Matrix P5 = strassenWinograd(S1, T1);
        Matrix P6 = strassenWinograd(S2, T2);
        Matrix P7 = strassenWinograd(S3, T3);

        // Combine into result quadrants — see the table in the comment above.
        Matrix U1 = addMat(P1, P2);
        Matrix U2 = addMat(P1, P6);
        Matrix U3 = addMat(U2, P7);
        Matrix U4 = addMat(U2, P5);
        Matrix U5 = addMat(U4, P3);
        Matrix U6 = subMat(U3, P4);
        Matrix U7 = addMat(U3, P5);

        // Assemble the n×n result from its four h×h quadrants
        Matrix C(n, n);
        setBlock(C, U1, 0, 0, h);  // C11
        setBlock(C, U5, 0, h, h);  // C12
        setBlock(C, U6, h, 0, h);  // C21
        setBlock(C, U7, h, h, h);  // C22
        return C;
    }
};

// pow(A, p) — matrix power A^p (not element-wise; use A.pow(p) for that).
//
// Integer p  — binary exponentiation using operator* (Strassen-accelerated).
//              Negative integers use A.inverse() then repeated squaring.
//
// Real p     — Higham Schur-Padé algorithm:
//   1. Schur decompose:  A = Q T Q^T
//   2. Compute T^p via Parlett recurrence on the upper triangular T
//      (diagonal entries λᵢ^p; super-diagonals via the commutativity equation
//      T·F = F·T)
//   3. Return Q * T^p * Q^T
//
// Requires square matrix. For real p, all eigenvalues must be positive
// (negative eigenvalues with non-integer p yield complex results — an exception
// is thrown).
//
// Usage: auto Ahalf = pow(A, 0.5);   // matrix square root
//        auto Ainv  = pow(A, -1);    // same as A.inverse()
//        auto A3    = pow(A, 3);     // A * A * A  via binary squaring
template <typename datatype, typename scalar>
Matrix<double> pow(const Matrix<datatype>& A, scalar p) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("pow: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    int n = (int)A.rows();

    // Convert to double for consistent arithmetic
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    auto identity = [&]() {
        Matrix<double> I(n, n);
        for (int i = 0; i < n; i++)
            I(i, i) = 1.0;
        return I;
    };

    // ── Integer fast path: binary exponentiation ──────────────────────────────
    long ip = (long)std::round(double(p));
    if (std::abs(double(p) - double(ip)) < 1e-9) {
        if (ip == 0)
            return identity();
        if (ip == 1)
            return Ad;
        Matrix<double> base = (ip < 0) ? Ad.inverse() : Ad;
        Matrix<double> result = identity();
        for (long exp = std::abs(ip); exp > 0; exp >>= 1) {
            if (exp & 1)
                result = result * base;
            if (exp > 1)
                base = base * base;
        }
        return result;
    }

    // ── Real power: Schur-Padé via Parlett recurrence ─────────────────────────
    // Step 1: Schur decompose Ad = Q * T * Q^T
    auto [Tv, Qv] = Ad.schurDecomp();
    auto T = [&](int i, int j) -> double { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) -> double { return Qv[i * n + j]; };

    // Step 2: Parlett recurrence for F = T^p (upper triangular)
    // Diagonal: F[i,i] = T[i,i]^p  (eigenvalue must be positive for real result)
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error("pow: eigenvalue " + std::to_string(lam) +
                                    " is non-positive — real matrix power undefined "
                                    "for non-integer exponent");
        f(i, i) = std::pow(lam, double(p));
    }

    // Super-diagonals via commutativity equation T·F = F·T → Parlett recurrence.
    // For distinct eigenvalues (|λᵢ - λⱼ| > ε):
    //   F[i,j] = (T[i,j]·(F[i,i]−F[j,j]) + Σ_{k=i+1}^{j-1}(F[i,k]·T[k,j] −
    //   T[i,k]·F[k,j]))
    //            / (T[i,i] − T[j,j])
    // For repeated eigenvalues (|λᵢ−λⱼ| < relative eps), the standard formula
    // has a near-zero denominator. Use the L'Hôpital limit instead:
    //   lim_{λⱼ→λᵢ} (f(λᵢ)−f(λⱼ))/(λᵢ−λⱼ) = f'(λᵢ) = p·λᵢ^(p−1)
    // This is exact for adjacent super-diagonals (d=1) and a good approximation
    // for d>1 because when eigenvalues are equal the inner sum is also zero.
    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) * double(p) * std::pow(T(i, i), double(p) - 1.0);
        }
    }

    // Step 3: A^p = Q·F·Qᵀ  —  two O(n³) multiplications, not one O(n⁴) loop
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j) = f(i, j);
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
}

// log(A, base) — matrix logarithm in an arbitrary base (free function).
// Distinct from A.log(base) which is the element-wise member function.
//
// Algorithm: Schur-Padé with f(x) = ln(x) / ln(base)
//   1. Schur decompose:  A = Q T Qᵀ
//   2. Compute F = ln(T) via Parlett recurrence on upper triangular T:
//        diagonal:       F[i,i] = ln(T[i,i])           (eigenvalue must be > 0)
//        super-diagonals: same recurrence as pow() with f'(λ) = 1/λ
//   3. Return Q · (F / ln(base)) · Qᵀ
//
// For symmetric (diagonalizable) A the Schur form is diagonal, so Parlett
// collapses to element-wise log on the eigenvalues — exact with no extra cost.
//
// Requires: square matrix, all eigenvalues positive, base > 0 and base ≠ 1.
// Usage: auto L2 = log(A, 2.0);   // log base-2 of matrix A
//        auto Le = log(A, M_E);   // natural matrix logarithm
template <typename datatype, typename scalar>
Matrix<double> log(const Matrix<datatype>& A, scalar base) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("log: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    if (double(base) <= 0.0 || double(base) == 1.0)
        throw std::invalid_argument("log: base must be positive and not equal to 1");

    int n = (int)A.rows();
    Matrix<double> Ad(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ad(i, j) = double(A(i, j));

    // ── Schur decompose: A = Q T Qᵀ ─────────────────────────────────────────
    auto [Tv, Qv] = Ad.schurDecomp();
    auto T = [&](int i, int j) { return Tv[i * n + j]; };
    auto Qm = [&](int i, int j) { return Qv[i * n + j]; };

    // ── Parlett recurrence for F = ln(T), f(x)=ln(x), f'(x)=1/x ────────────
    std::vector<double> F(n * n, 0.0);
    auto f = [&](int i, int j) -> double& { return F[i * n + j]; };

    for (int i = 0; i < n; i++) {
        double lam = T(i, i);
        if (lam <= 0.0)
            throw std::domain_error("log: eigenvalue " + std::to_string(lam) +
                                    " is non-positive — matrix logarithm is not real-valued");
        f(i, i) = std::log(lam);
    }

    for (int d = 1; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            double num = T(i, j) * (f(i, i) - f(j, j));
            for (int k = i + 1; k < j; k++)
                num += f(i, k) * T(k, j) - T(i, k) * f(k, j);
            double denom = T(i, i) - T(j, j);
            double scale = std::max(std::abs(T(i, i)), std::abs(T(j, j)));
            if (std::abs(denom) > std::numeric_limits<double>::epsilon() * 1e4 * scale)
                f(i, j) = num / denom;
            else
                f(i, j) = T(i, j) / T(i, i);  // f'(λ) = 1/λ for ln
        }
    }

    // ── A^log = Q · (F/ln(base)) · Qᵀ  ─────────────────────────────────────
    double logBase = std::log(double(base));
    Matrix<double> Fm(n, n), Qmat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            Fm(i, j) = f(i, j) / logBase;
            Qmat(i, j) = Qm(i, j);
        }
    return Qmat * Fm * Qmat.T();
}

// ─── Matrix functions (free) ────────────────────────────────────────────────
// Reminder on the convention this header follows:
//   A.exp()  → element-wise, e^(a_ij) for each element independently
//   exp(A)   → MATRIX exponential, the power series I + A + A²/2! + A³/3! + ...
// These are completely different results. The two above and the two below
// (A.sin() vs sin(A)) are the pairs most likely to be confused, so each doc
// comment says which one it is.
//
// All of these evaluate a Taylor series, and all of them accept an optional
// trailing TaylorOpts controlling how far that series runs — see TaylorOpts at
// the top of this header for the four call forms.

// Shared machinery for the Taylor-series matrix functions. Kept in a namespace
// because these are implementation details, not part of the Matrix API, and the
// names (eye, halvings) are ones a caller might reasonably want for themselves.
namespace taylor_detail {

    // n×n identity as Matrix<double>.
    inline Matrix<double> eye(int n) {
        Matrix<double> E(n, n);
        for (int i = 0; i < n; i++)
            E(i, i) = 1.0;
        return E;
    }

    // Validated double copy of A — every matrix function needs the same square,
    // non-empty check and the same conversion, so they share one.
    template <typename datatype>
    Matrix<double> squareAsDouble(const Matrix<datatype>& A, const char* who) {
        static_assert(!is_complex<datatype>::value,
                      "The matrix functions (exp/sin/cos/tan/sinh/cosh/tanh) are not yet "
                      "implemented for complex datatypes. They would compile by taking "
                      "std::real() of each entry and silently discard the imaginary part. "
                      "See roadmap item 1 (work_t) at the top of this header.");
        if (A.rows() != A.cols())
            throw std::invalid_argument(std::string(who) + ": matrix must be square, got " +
                                        std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
        if (A.rows() == 0)
            throw std::invalid_argument(std::string(who) + ": matrix must be non-empty");
        int n = (int)A.rows();
        Matrix<double> Ad(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Ad(i, j) = double(std::real(A(i, j)));
        return Ad;
    }

    // How many times to halve A so that ||A/2^s|| lands inside `radius`.
    //
    // This is the step that makes a Taylor series usable at all. The series for
    // e^A converges for every A in theory, but for ||A|| = 20 the terms grow to
    // about 20^20/20! ≈ 4e7 before they start shrinking, so the sum loses every
    // significant digit it had to cancellation. Halving first keeps every term
    // smaller than the last, and squaring afterwards costs only s extra products.
    inline int halvings(double nrm, double radius, bool enabled) {
        if (!enabled || !(nrm > radius))
            return 0;
        int s = (int)std::ceil(std::log2(nrm / radius));
        if (s < 0)
            s = 0;
        if (s > 64)
            s = 64;  // beyond this the squarings cost more than they buy
        return s;
    }

    // Evaluates {sin(A), cos(A)} — or {sinh(A), cosh(A)} when hyper is true.
    //
    // They are computed together on purpose: the double-angle identity that
    // undoes the scaling for sine, sin(2Y) = 2·sin(Y)·cos(Y), needs the cosine
    // anyway, so computing one alone would cost the same as computing both.
    //
    // Both series advance by a recurrence on the previous term:
    //     sin:  t₀ = X,  tₖ = tₖ₋₁ · X² / ( -(2k)(2k+1) )
    //     cos:  t₀ = I,  tₖ = tₖ₋₁ · X² / ( -(2k-1)(2k) )
    // with the sign folded into the denominator, and dropped for the hyperbolic
    // pair. One matrix product per term, and no factorial is ever formed — 171!
    // already overflows a double, so any implementation that divides by an
    // explicit factorial returns NaN long before 300 terms.
    inline std::pair<Matrix<double>, Matrix<double>> sincosSeries(Matrix<double> X,
                                                                  TaylorOpts opts,
                                                                  bool hyper) {
        int n = (int)X.rows();
        int s = halvings(X.norm(NormType::Inf), 1.0, opts.scaling);
        if (s)
            X = X / std::ldexp(1.0, s);

        Matrix<double> X2 = X * X;
        Matrix<double> Id = eye(n);
        Matrix<double> S = X, sTerm = X;
        Matrix<double> C = Id, cTerm = Id;

        const double tol = opts.epsTol();
        bool sDone = false, cDone = false;
        for (long k = 1; k <= opts.maxTerms && !(sDone && cDone); k++) {
            double sDen = double(2 * k) * double(2 * k + 1);
            double cDen = double(2 * k - 1) * double(2 * k);
            if (!hyper) {
                sDen = -sDen;
                cDen = -cDen;
            }
            if (!sDone) {
                sTerm = sTerm * X2 / sDen;
                S += sTerm;
            }
            if (!cDone) {
                cTerm = cTerm * X2 / cDen;
                C += cTerm;
            }
            if (tol > 0.0) {
                if (sTerm.norm(NormType::Inf) <= tol * S.norm(NormType::Inf))
                    sDone = true;
                if (cTerm.norm(NormType::Inf) <= tol * C.norm(NormType::Inf))
                    cDone = true;
            }
        }

        // Undo the halving. The circular and hyperbolic identities have the same
        // shape, which is why one loop serves both:
        //   sin(2Y) = 2·sin·cos      sinh(2Y) = 2·sinh·cosh
        //   cos(2Y) = 2·cos² − I     cosh(2Y) = 2·cosh² − I
        for (int i = 0; i < s; i++) {
            Matrix<double> Snew = (S * C) * 2.0;
            C = (C * C) * 2.0 - Id;
            S = Snew;
        }
        return {S, C};
    }

}  // namespace taylor_detail

// exp(A) — the matrix exponential, sum of A^k / k!.
//
// The most valuable of the missing matrix functions: it is the closed-form
// solution of the linear ODE x' = A*x, the transition operator of a continuous
// Markov chain, the map from a Lie algebra to its Lie group, and — once complex
// support lands — exp(-i*H*t), the time-evolution operator the quantum-circuit
// goal in the README is built on.
//
// Unlike pow() and log() this does NOT need the Schur form, so it is unaffected
// by the 2x2 real-Schur block problem and is defined for every square matrix
// with no eigenvalue restrictions.
//
// Scaling and squaring around a Taylor series:
//   1. halve A until ||A/2^s||∞ <= 1/2, where the series converges fast and
//      monotonically
//   2. sum I + X + X²/2! + … by the recurrence tₖ = tₖ₋₁·X/k — one matrix
//      product per term, stopping as soon as a term stops contributing
//   3. square s times to undo the scaling, since e^A = (e^(A/2^s))^(2^s)
//
// Usage: auto E = exp(A);                  // defaults
//        auto E = exp(A, 25);              // at most 25 terms
//        auto E = exp(A, {25, 1e-12});     // ... or until terms fall below
//        1e-12 auto E = exp(A, {25, 1e-12, false});  // ... with scaling
//        disabled
template <typename datatype>
Matrix<double> exp(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        Matrix<double> X = taylor_detail::squareAsDouble(A, "exp");
        int n = (int)X.rows();

        int s = taylor_detail::halvings(X.norm(NormType::Inf), 0.5, opts.scaling);
        if (s)
            X = X / std::ldexp(1.0, s);

        Matrix<double> E = taylor_detail::eye(n);
        Matrix<double> term = E;
        const double tol = opts.epsTol();
        for (long k = 1; k <= opts.maxTerms; k++) {
            term = term * X / double(k);
            E += term;
            if (tol > 0.0 && term.norm(NormType::Inf) <= tol * E.norm(NormType::Inf))
                break;
        }

        for (int i = 0; i < s; i++)
            E = E * E;
        return E;
    } catch (const std::exception& e) {
        std::cerr << "exp(A) error: " << e.what() << std::endl;
        throw;
    }
}

// ─── Matrix trigonometry (free) ─────────────────────────────────────────────
// sin(A), cos(A) and the hyperbolic pair are ENTIRE functions: their power
// series converge for every square matrix. So unlike log(A, base) — which
// requires positive eigenvalues — these have no domain restriction at all.
//
// That mattered for the route not taken. Evaluating them through the Schur form
// the way pow() and log() do would run straight into the 2x2 blocks that a real
// Schur form carries for complex-conjugate eigenvalue pairs, and a plain 2D
// rotation matrix — complex eigenvalues, perfectly ordinary real cos(A) — is
// enough to trigger it. Summing the series directly sidesteps the Schur form
// altogether, so these are correct for rotations today, with no dependency on
// the block-aware Parlett recurrence described at schurDecomp().
//
// Identities the test suite checks (matrix products, not element-wise):
//   sin(A)² + cos(A)²  == I
//   cosh(A)² − sinh(A)² == I
//   sin(A) is NOT A.sin(); the two agree only for diagonal A
//
// ACCURACY, and what to expect from those identities. If A has an eigenvalue
// with a large imaginary part, sin(A) and cos(A) genuinely grow like
// e^|Im(λ)| — for ||A||∞ ≈ 40 that is around 1e11 per entry. sin²+cos² then
// asks for ~1e22 of cancellation to land back on I, so the ABSOLUTE residual
// is bounded below by conditioning, not by the algorithm: no implementation in
// double precision returns a small one. The error relative to those
// intermediate magnitudes stays at machine epsilon, which is the meaningful
// measure. Symmetric A has real eigenvalues, so sin/cos stay bounded by 1 and
// the identity does hold to full absolute accuracy there.

// sin(A) — matrix sine, sum of (-1)^k A^(2k+1) / (2k+1)!.
// Optional TaylorOpts as for exp(): sin(A, 25), sin(A, {25, 1e-12}), etc.
template <typename datatype>
Matrix<double> sin(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "sin"), opts, false);
        (void)C;
        return S;
    } catch (const std::exception& e) {
        std::cerr << "sin(A) error: " << e.what() << std::endl;
        throw;
    }
}

// cos(A) — matrix cosine, sum of (-1)^k A^(2k) / (2k)!.
// Optional TaylorOpts as for exp().
template <typename datatype>
Matrix<double> cos(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "cos"), opts, false);
        (void)S;
        return C;
    } catch (const std::exception& e) {
        std::cerr << "cos(A) error: " << e.what() << std::endl;
        throw;
    }
}

// tan(A) — matrix tangent, the X solving cos(A)·X = sin(A).
//
// NOTE the operation: this is a linear SOLVE, cos(A)⁻¹·sin(A) — a LEFT
// division. sin(A)/cos(A) would now give the right division sin(A)·cos(A)⁻¹,
// which happens to be the same matrix here (sin(A) and cos(A) commute, both
// being functions of A) but is the wrong operation in general, and goes through
// an extra pair of transposes to get there. Undefined where cos(A) is singular.
template <typename datatype>
Matrix<double> tan(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "tan"), opts, false);
        return C.solve(S);
    } catch (const std::exception& e) {
        std::cerr << "tan(A) error: " << e.what() << std::endl;
        throw;
    }
}

// sinh(A) — matrix hyperbolic sine, sum of A^(2k+1) / (2k+1)!.
//
// Summed directly rather than as (exp(A) − exp(−A))/2: that identity costs two
// matrix exponentials, and it cancels catastrophically for large ||A||, where
// exp(A) and exp(−A) agree to many digits before the subtraction throws them
// away.
template <typename datatype>
Matrix<double> sinh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "sinh"), opts, true);
        (void)C;
        return S;
    } catch (const std::exception& e) {
        std::cerr << "sinh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// cosh(A) — matrix hyperbolic cosine, sum of A^(2k) / (2k)!.
template <typename datatype>
Matrix<double> cosh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "cosh"), opts, true);
        (void)S;
        return C;
    } catch (const std::exception& e) {
        std::cerr << "cosh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// tanh(A) — matrix hyperbolic tangent, the X solving cosh(A)·X = sinh(A).
// A solve, not an element-wise division — see the note on tan(A).
template <typename datatype>
Matrix<double> tanh(const Matrix<datatype>& A, TaylorOpts opts = {}) {
    try {
        auto [S, C] =
            taylor_detail::sincosSeries(taylor_detail::squareAsDouble(A, "tanh"), opts, true);
        return C.solve(S);
    } catch (const std::exception& e) {
        std::cerr << "tanh(A) error: " << e.what() << std::endl;
        throw;
    }
}

// sqrt(A) — principal matrix square root, the X with X*X = A.
// Already reachable as pow(A, 0.5); this is the conventional spelling. Note
// this one goes through the Schur form, not a Taylor series, so it takes no
// TaylorOpts and it does inherit pow()'s positive-eigenvalue requirement.
template <typename datatype>
Matrix<double> sqrt(const Matrix<datatype>& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("sqrt: matrix must be square, got " + std::to_string(A.rows()) +
                                    "x" + std::to_string(A.cols()));
    return pow(A, 0.5);
}

// ─── Free-function spellings that mirror mathematical notation ──────────────

// tr(A) — sum of the main diagonal elements. Mirrors mathematical notation.
template <typename datatype>
datatype tr(const Matrix<datatype>& A) {
    return A.tr();
}

// det(A) — determinant. Mirrors mathematical notation.
template <typename datatype>
datatype det(const Matrix<datatype>& A) {
    return A.det();
}

// Thin wrappers over the members of the same name, for the callers who prefer
// f(A) notation. Each is a one-liner like tr/det above.

// solve(A, B) — solves A * X = B. See Matrix::solve.
template <typename datatype, typename dtB>
Matrix<double> solve(const Matrix<datatype>& A, const Matrix<dtB>& B) {
    return A.solve(B);
}

// norm(A, type) — matrix norm. See Matrix::norm.
template <typename datatype>
double norm(const Matrix<datatype>& A, NormType type = NormType::Fro) {
    return A.norm(type);
}

// rank(A) — numerical rank. See Matrix::rank.
template <typename datatype>
long rank(const Matrix<datatype>& A, double tol = -1.0) {
    return A.rank(tol);
}

// cond(A, type) — condition number. See Matrix::cond.
template <typename datatype>
double cond(const Matrix<datatype>& A, NormType type = NormType::Two) {
    return A.cond(type);
}

// inverse(A) — matrix inverse. See Matrix::inverse.
template <typename datatype>
Matrix<double> inverse(const Matrix<datatype>& A) {
    return A.inverse();
}

// pinv(A) — Moore-Penrose pseudo-inverse. See Matrix::pinv.
template <typename datatype>
Matrix<double> pinv(const Matrix<datatype>& A, double tol = -1.0) {
    return A.pinv(tol);
}

// adj(A) — adjugate. See Matrix::adjugate.
template <typename datatype>
Matrix<double> adj(const Matrix<datatype>& A) {
    return A.adjugate();
}

// eigvals(A) — every eigenvalue, complex ones included. See Matrix::eigvals.
template <typename datatype>
Matrix<std::complex<double>> eigvals(const Matrix<datatype>& A) {
    return A.eigvals();
}

// diag(v) — builds a square diagonal matrix FROM a vector, the inverse
// direction of the member A.diag() which extracts a diagonal into a vector.
// v must be a row or column vector; an (n x 1) or (1 x n) input gives n x n.
// ═══════════════════════════════════════════════════════════════════════════
//  Decomposition — factor once, solve many times  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
// A.solve(b) factors A from scratch on every call. That is the right default
// for a one-off, and completely wrong in a loop: at n = 512 the factorisation
// is ~7 ms and a triangular solve is ~0.1 ms, so a hundred right-hand sides
// solved one at a time cost a hundred factorisations.
//
//     auto dA = A.factorize();          // or decomposition(A), MATLAB's
//     spelling Matrix<double> x1 = dA.solve(b1); Matrix<double> x2 =
//     dA.solve(b2);
//
// The object OWNS its factors, so it stays valid after A is destroyed — it is a
// value, like everything else here, not a view onto the matrix it came from.
//
// It picks the factorisation the way MATLAB's decomposition does, from the
// structure of A rather than from a flag the caller has to get right:
//     square, symmetric, positive definite  ->  Cholesky   (half the flops)
//     square, otherwise                     ->  LU with partial pivoting
//     rectangular                           ->  column-pivoted QR (least
//     squares)
// kind() reports which was chosen, because "why is this slower than I expected"
// is a question the answer should be available for.
template <typename datatype>
class Decomposition {
  public:
    enum class Kind { Cholesky, LU, QR };

    explicit Decomposition(const Matrix<datatype>& A) : rows_(A.rows()), cols_(A.cols()) {
        if (rows_ == 0 || cols_ == 0)
            throw std::invalid_argument("Decomposition: matrix must be non-empty");
        if (rows_ == cols_) {
            // Cholesky is worth trying first: it is half the work of LU, and the
            // attempt is itself the positive-definiteness test — a non-positive
            // pivot IS the proof, so there is no separate check to pay for.
            if (A.IsSymmetric()) {
                try {
                    chol_ = A.cholesky();
                    kind_ = Kind::Cholesky;
                    return;
                } catch (const std::domain_error&) {
                    // Symmetric but not positive definite. Fall through to LU.
                }
            }
            auto [packed, piv] = A.luPacked();
            lu_ = std::move(packed);
            piv_ = std::move(piv);
            kind_ = Kind::LU;
            return;
        }
        if (rows_ < cols_)
            throw std::invalid_argument(
                "Decomposition: under-determined systems (" + std::to_string(rows_) +
                " equations, " + std::to_string(cols_) +
                " unknowns) have infinitely "
                "many solutions — use lsqminnorm() for the minimum-norm one");
        auto [Q, R, P] = A.QR();
        q_ = std::move(Q);
        r_ = std::move(R);
        perm_.resize((std::size_t)cols_);
        for (long k = 0; k < cols_; k++)
            for (long i = 0; i < cols_; i++)
                if (double(std::real(P(int(i), int(k)))) != 0.0)
                    perm_[(std::size_t)k] = i;
        kind_ = Kind::QR;
    }

    Kind kind() const { return kind_; }

    const char* kindName() const {
        switch (kind_) {
            case Kind::Cholesky:
                return "Cholesky";
            case Kind::LU:
                return "LU";
            default:
                return "QR";
        }
    }

    long rows() const { return rows_; }
    long cols() const { return cols_; }

    // Solves A * X = B using the stored factors. No refactorisation.
    template <typename dtB>
    Matrix<double> solve(const Matrix<dtB>& B) const {
        if (B.rows() != rows_)
            throw std::invalid_argument(
                "Decomposition::solve: B must have one row per row of A — A is " +
                std::to_string(rows_) + "x" + std::to_string(cols_) + " but B is " +
                std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
        const long nrhs = B.cols();
        Matrix<double> Bd(B.rows(), nrhs);
        for (long i = 0; i < B.rows(); i++)
            for (long j = 0; j < nrhs; j++)
                Bd(int(i), int(j)) = double(std::real(B(int(i), int(j))));

        if (kind_ == Kind::LU)
            return Matrix<datatype>::luSubstitute(lu_, piv_, (int)rows_, Bd);

        if (kind_ == Kind::Cholesky) {
            // A = L Lᵀ, so forward-substitute through L then back through Lᵀ.
            const long n = rows_;
            Matrix<double> X(n, nrhs);
            for (long c = 0; c < nrhs; c++) {
                std::vector<double> y((std::size_t)n);
                for (long i = 0; i < n; i++) {
                    double acc = Bd(int(i), int(c));
                    for (long j = 0; j < i; j++)
                        acc -= chol_(int(i), int(j)) * y[(std::size_t)j];
                    y[(std::size_t)i] = acc / chol_(int(i), int(i));
                }
                for (long i = n - 1; i >= 0; i--) {
                    double acc = y[(std::size_t)i];
                    for (long j = i + 1; j < n; j++)
                        acc -= chol_(int(j), int(i)) * X(int(j), int(c));
                    X(int(i), int(c)) = acc / chol_(int(i), int(i));
                }
            }
            return X;
        }

        // QR least squares: A*P = Q*R, so minimise ||R (Pᵀx) - Qᵀb||.
        Matrix<double> QtB = q_.T() * Bd;
        const double rTol = double(std::max(rows_, cols_)) *
                            std::numeric_limits<double>::epsilon() * std::abs(r_(0, 0));
        long rk = 0;
        while (rk < cols_ && std::abs(r_(int(rk), int(rk))) > rTol)
            rk++;
        Matrix<double> Y(cols_, nrhs);
        for (long c = 0; c < nrhs; c++)
            for (long i = rk - 1; i >= 0; i--) {
                double acc = QtB(int(i), int(c));
                for (long j = i + 1; j < rk; j++)
                    acc -= r_(int(i), int(j)) * Y(int(j), int(c));
                Y(int(i), int(c)) = acc / r_(int(i), int(i));
            }
        Matrix<double> X(cols_, nrhs);
        for (long k = 0; k < cols_; k++)
            for (long c = 0; c < nrhs; c++)
                X(int(perm_[(std::size_t)k]), int(c)) = Y(int(k), int(c));
        return X;
    }

    // Determinant, free from the factors already held. Only square systems have
    // one, so the QR case says so rather than returning something meaningless.
    double det() const {
        if (kind_ == Kind::QR)
            throw std::logic_error("Decomposition::det: only a square system has a determinant");
        if (kind_ == Kind::Cholesky) {
            double d = 1.0;
            for (long i = 0; i < rows_; i++)
                d *= chol_(int(i), int(i));
            return d * d;  // det(L Lᵀ) = det(L)²
        }
        double d = 1.0;
        int swaps = 0;
        for (long i = 0; i < rows_; i++) {
            d *= lu_[(std::size_t)(i * rows_ + i)];
            if (piv_[(std::size_t)i] != (int)i)
                swaps++;
        }
        return (swaps % 2) ? -d : d;
    }

    // Reciprocal condition number in the 1-norm, by the Hager-Higham estimator
    // (LAPACK's dlacn2 / dgecon). It needs ||A⁻¹||₁, which would ordinarily mean
    // forming the inverse; the estimator instead finds a vector that nearly
    // maximises ||A⁻¹x||₁/||x||₁ using a handful of solves against factors we
    // already have. A few solves against O(n³) for an explicit inverse.
    //
    // It is an ESTIMATE, and a lower bound on the true condition number — it can
    // be optimistic, never pessimistic. Near 1 means well conditioned; near
    // machine epsilon means the matrix is numerically singular.
    double rcond(double oneNormA) const {
        if (kind_ == Kind::QR)
            throw std::logic_error("Decomposition::rcond: only defined for a square system");
        if (oneNormA == 0.0)
            return 0.0;
        const long n = rows_;
        Matrix<double> x(n, 1);
        for (long i = 0; i < n; i++)
            x(int(i), 0) = 1.0 / double(n);
        double est = 0.0;
        for (int iter = 0; iter < 5; iter++) {
            Matrix<double> y = solve(x);
            double ynorm = 0.0;
            for (long i = 0; i < n; i++)
                ynorm += std::abs(y(int(i), 0));
            Matrix<double> xi(n, 1);
            for (long i = 0; i < n; i++)
                xi(int(i), 0) = y(int(i), 0) >= 0.0 ? 1.0 : -1.0;
            Matrix<double> z = solveTransposed(xi);
            long jmax = 0;
            for (long i = 1; i < n; i++)
                if (std::abs(z(int(i), 0)) > std::abs(z(int(jmax), 0)))
                    jmax = i;
            double ztx = 0.0;
            for (long i = 0; i < n; i++)
                ztx += z(int(i), 0) * x(int(i), 0);
            if (iter && std::abs(z(int(jmax), 0)) <= ztx) {
                est = ynorm;
                break;
            }
            est = ynorm;
            for (long i = 0; i < n; i++)
                x(int(i), 0) = 0.0;
            x(int(jmax), 0) = 1.0;
        }
        if (est == 0.0)
            return 0.0;
        return 1.0 / (oneNormA * est);
    }

  private:
    Kind kind_ = Kind::LU;
    long rows_ = 0, cols_ = 0;
    std::vector<double> lu_;
    std::vector<int> piv_;
    Matrix<double> chol_;
    Matrix<double> q_, r_;
    std::vector<long> perm_;

    // Aᵀ x = b, for the condition estimator. Cholesky is symmetric so its
    // transpose is itself.
    Matrix<double> solveTransposed(const Matrix<double>& B) const {
        if (kind_ == Kind::Cholesky)
            return solve(B);
        return Matrix<datatype>::luSubstituteT(lu_, piv_, (int)rows_, B);
    }
};

template <typename datatype>
Decomposition<datatype> Matrix<datatype>::factorize() const {
    return Decomposition<datatype>(*this);
}

// decomposition(A) — MATLAB's spelling of A.factorize().
template <typename datatype>
Decomposition<datatype> decomposition(const Matrix<datatype>& A) {
    return Decomposition<datatype>(A);
}

// rcond(A) — reciprocal condition estimate in the 1-norm. See
// Decomposition::rcond for what the estimate is and is not.
template <typename datatype>
double rcond(const Matrix<datatype>& A) {
    return Decomposition<datatype>(A).rcond(A.norm(NormType::One));
}

// condest(A) — 1-norm condition estimate, the reciprocal of rcond.
template <typename datatype>
double condest(const Matrix<datatype>& A) {
    const double r = rcond(A);
    return r == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / r;
}

// lsqminnorm(A, b) — the minimum-norm least-squares solution.
//
// solve() throws for an under-determined system, on the grounds that
// "infinitely many solutions" is usually a mistake the caller wants told about.
// When it is not a mistake, this is the one to reach for: of all the x
// minimising ||Ax-b||, it returns the one with the smallest ||x||. Computed
// through the SVD, so it is well defined for a rank-deficient A too, which is
// exactly when it is wanted.
template <typename datatype, typename dtB>
Matrix<double> lsqminnorm(const Matrix<datatype>& A, const Matrix<dtB>& B, double tol = -1.0) {
    return A.pinv(tol) * B;
}

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

// ═══════════════════════════════════════════════════════════════════════════
//  TIER 6 — SIGNAL, CALCULUS AND INTERPOLATION            *** WORK IN PROGRESS
// ═══════════════════════════════════════════════════════════════════════════
//
// Everything below this line is SCAFFOLDING: contracts, conventions and design
// decisions written down, implementations deliberately not. Fill them in here.
//
// Already landed from this tier, in the Polynomials block above:
//     polyval, polyfit, roots
// Still to write, in a sensible order (each is useful on its own, and the later
// ones get easier once the earlier ones exist):
//     conv, deconv, poly           polynomial arithmetic — no new machinery
//     trapz, cumtrapz, gradient    calculus on samples — no new machinery
//     interp1                      needs a sorted-lookup helper
//     filter                       a recurrence; the one with real edge cases
//     fft, ifft                    the big one; see the design notes at the end
//
// ── CONVENTIONS THIS HEADER ALREADY COMMITTED TO ──────────────────────────
// Worth having in front of you before writing any of these, because breaking
// one of them is the kind of thing that only shows up much later:
//
//   * THE MEMBER DOT MARKS ELEMENT-WISE. A.sin() is element-wise, sin(A) is the
//     matrix function. So anything here that acts on a whole vector as a signal
//     — conv, filter, fft, trapz — is a FREE function, not a member. cumtrapz
//     and gradient are the interesting case: they act along an axis, like
//     cumsum, so they are arguably members taking the same `addcol` flag. Pick
//     one and say why in the comment; do not leave it implied.
//   * addcol = false works DOWN columns, true works ALONG rows. Every reduction
//     and scan in the header uses that flag with that meaning.
//   * Polynomial coefficients are DESCENDING, matching polyval/polyfit/roots.
//     conv and poly must agree, or roots(conv(a,b)) will silently be wrong.
//   * Anything random goes through ran2() from random.hpp, never std::rand, and
//     stays serial — ran2 keeps static state.
//   * A free function cannot use uninit_t; it is private, deliberately. Use the
//     ordinary constructor and accept the extra pass, as the tier 5
//     constructors do.
//   * Throw std::invalid_argument with the ACTUAL sizes in the message. Every
//     error path in this header names the numbers it saw.
//
// ── THE CONTRACTS ─────────────────────────────────────────────────────────
// Signatures are suggestions, but the SEMANTICS are not: these are what MATLAB
// and NumPy do, and agreeing with them is what lets numpy_validate.py check the
// results rather than just the shapes.
//
//   Matrix<double> conv(const Matrix<A>& a, const Matrix<B>& b)
//       Discrete convolution, equivalently polynomial multiplication. Result
//       length is na + nb - 1. MATLAB conv, NumPy np.convolve.
//       The direct O(na*nb) double loop is the right implementation up to
//       roughly a thousand terms; above that the FFT route wins, which is a
//       reason to write fft first and come back with a threshold — exactly the
//       shape of the STRASSEN_THRESHOLD decision in operator*.
//
//   std::pair<Matrix<double>, Matrix<double>> deconv(const Matrix<A>& y,
//                                                    const Matrix<B>& a)
//       Polynomial long division: returns {quotient, remainder} with
//       y == conv(a, quotient) + remainder. MATLAB returns [q, r].
//       Edge case worth deciding: a(0) == 0. MATLAB errors. Leading zeros in
//       `a` are the same problem roots() already strips — reuse that reasoning.
//
//   Matrix<double> poly(const Matrix<A>& r)
//       The monic polynomial whose roots are r, in descending order, so that
//       roots(poly(r)) returns r back up to ordering and rounding. MATLAB poly.
//       NOTE MATLAB overloads this: poly(A) for a SQUARE MATRIX gives the
//       characteristic polynomial, which is poly(eigvals(A)). Decide whether to
//       support that too; if so it needs a separate overload, since a square
//       matrix is also a valid list of numbers and the two would be ambiguous
//       for a 1x1.
//
//   double trapz(const Matrix<A>& y)                        uniform spacing 1
//   double trapz(const Matrix<A>& x, const Matrix<A>& y)    given sample points
//       Trapezoidal integral. Sum of (x[i+1]-x[i]) * (y[i]+y[i+1]) / 2.
//       With n < 2 samples the integral is 0, not an error — MATLAB agrees.
//
//   cumtrapz, gradient
//       Cumulative trapezoid and the central-difference derivative. gradient
//       uses a CENTRED difference in the interior and a one-sided difference at
//       each end, so the result is the same length as the input — that is the
//       whole difference between gradient and diff, and it is worth a comment.
//
//   Matrix<double> interp1(const Matrix<A>& x, const Matrix<A>& y,
//                          const Matrix<A>& xi)
//       Linear interpolation of the samples (x, y) at the points xi.
//       Decisions to make and document:
//         - x must be sorted ascending. Check it, or sort internally? Checking
//           is cheaper and catches a real class of caller bug.
//         - Out-of-range xi: MATLAB returns NaN, NumPy's np.interp clamps to
//         the
//           end values. They disagree, so pick one and say which.
//         - std::lower_bound is the right lookup, and it is already included
//         via
//           <algorithm>.
//
//   Matrix<double> filter(const Matrix<A>& b, const Matrix<A>& a,
//                         const Matrix<A>& x)
//       The rational-transfer-function difference equation, MATLAB filter:
//           a(0)*y(n) = b(0)*x(n) + b(1)*x(n-1) + ... - a(1)*y(n-1) - ...
//       Normalise by a(0); error if it is zero. This is a RECURRENCE, so unlike
//       everything else in this tier it cannot be parallelised over the output
//       — worth stating in the comment so nobody tries later.
//
// ── fft / ifft: the decisions to make before writing a line ───────────────
// This is the one with real design freedom, and the choices interact. Directly
// relevant to the quantum-circuit goal, since the QFT is exactly this.
//
//   1. WHERE DOES IT LIVE, AND ON WHAT?
//      A signal is a vector. Matrix<complex<double>> is the obvious carrier and
//      keeps everything in one type. Deciding fft(Matrix) -> Matrix now avoids
//      a painful move later.
//      MATLAB's fft(A) on a MATRIX transforms each COLUMN. That falls out of
//      the addcol convention if you want it, and is a natural second overload.
//
//   2. WHAT ABOUT LENGTHS THAT ARE NOT A POWER OF TWO?
//      Radix-2 Cooley-Tukey is thirty lines and only handles 2^k. The honest
//      options, in increasing order of work:
//        (a) radix-2 only, and THROW for other lengths. Clean, and enough for
//            quantum work where everything is 2^n by construction.
//        (b) radix-2, and zero-pad to the next power of two. Fast and easy, but
//            it changes the answer — padding computes the transform of a
//            different, longer signal. Only correct for convolution, never for
//            spectra. If you take this route, do it INSIDE conv() and not
//            inside fft(), or the result will be quietly wrong.
//        (c) Bluestein's chirp-z for arbitrary n, which turns any length into a
//            power-of-two convolution. Correct for every n, and about a hundred
//            more lines.
//      (a) now with a clear error message, (c) later, is a defensible path.
//      What is NOT defensible is (b) hidden inside fft().
//
//   3. NORMALISATION. MATLAB and NumPy both put the whole 1/n on the INVERSE
//   and
//      none on the forward transform. Match them — a different convention here
//      would make every cross-check against numpy_validate.py fail for a reason
//      that has nothing to do with correctness.
//
//   4. THE PROPERTY TO TEST FIRST is ifft(fft(x)) == x. It catches
//   normalisation,
//      bit-reversal and twiddle-sign errors all at once, and it needs no
//      reference implementation. After that, check a known pair by hand — the
//      transform of a constant vector is a spike at index 0 — and only then
//      compare against numpy.fft in numpy_validate.py.
//
//   5. std::complex<double> already has everything needed: std::polar for the
//      twiddle factors, and the arithmetic operators. is_complex<T> and
//      real_t<T> at the top of this header are there for writing the guards.
//
// ── WHERE THE TESTS GO ────────────────────────────────────────────────────
//   validate.cpp        — section("Signal, calculus and interpolation (tier
//   6)")
//                         is already there, empty, waiting.
//   numpy_validate.cpp  — the "tier 6" block is stubbed with the Case() calls
//                         commented out; uncomment them as each lands.
//   numpy_validate.py   — matching NumPy references are stubbed the same way.
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//  Generalized eigenvalue problems  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
//     eig(A, B)      symmetric-definite:  A x = lambda B x, A symmetric,
//                                         B symmetric positive definite
//     eigvals(A, B)  the general pencil, when B is well conditioned
//     polyeig({A0, A1, ..., Ad})          matrix polynomial eigenvalues
//
// ── WHAT IS HERE AND WHAT IS NOT ──────────────────────────────────────────
// The complete, backward-stable answer for a general pencil is the QZ
// algorithm — a generalized Schur decomposition, with its own
// Hessenberg-triangular reduction and shifted sweeps. That is a project in its
// own right and is NOT implemented. What is here instead:
//
//   * The SYMMETRIC-DEFINITE case in full, by Cholesky reduction. This is
//     backward stable, it is the case that actually turns up — vibration modes,
//     PCA with a covariance metric, generalized least squares — and it is
//     exactly what LAPACK's dsygv does.
//   * The general case by way of B⁻¹A, GUARDED by a condition estimate. Forming
//     B⁻¹A is legitimate when B is well conditioned and loses accuracy in
//     proportion to cond(B) when it is not, so this measures cond(B) with the
//     Hager-Higham estimator and REFUSES when the answer would not be
//     trustworthy, naming QZ as what is needed. Returning plausible-looking
//     numbers from a near-singular B is the failure this is written to avoid.
//
// eig(A, B) — symmetric-definite. Returns {eigenvalues ascending, eigenvectors},
// the same shape as the standard A.eig().
//
// The reduction: B = L Lᵀ, then A x = lambda B x becomes C y = lambda y with
// C = L⁻¹ A L⁻ᵀ and x = L⁻ᵀ y. C is symmetric, so the standard symmetric
// eigensolver applies and the eigenvalues come out real — which is the whole
// point of doing it this way rather than through B⁻¹A, whose product is not
// symmetric even when both factors are.
template <typename datatype>
std::pair<Matrix<double>, Matrix<double>> eig(const Matrix<datatype>& A,
                                              const Matrix<datatype>& B) {
    static_assert(!is_complex<datatype>::value,
                  "eig(A,B): not yet implemented for complex datatypes.");
    if (A.rows() != A.cols() || B.rows() != B.cols())
        throw std::invalid_argument("eig(A,B): both matrices must be square, got " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    " and " + std::to_string(B.rows()) + "x" +
                                    std::to_string(B.cols()));
    if (A.rows() != B.rows())
        throw std::invalid_argument("eig(A,B): A and B must be the same size, got " +
                                    std::to_string(A.rows()) + " and " + std::to_string(B.rows()));
    if (!A.IsSymmetric())
        throw std::invalid_argument(
            "eig(A,B): A must be symmetric. This is the symmetric-definite solver; the "
            "general pencil needs QZ, which is not implemented — see eigvals(A,B) for "
            "what is available.");
    const long n = A.rows();

    // cholesky() throws domain_error on a non-positive pivot, and that failure
    // IS the positive-definiteness test — there is no separate check to pay for.
    Matrix<double> L;
    try {
        L = B.cholesky();
    } catch (const std::domain_error&) {
        throw std::invalid_argument(
            "eig(A,B): B must be positive definite — its Cholesky factorisation hit a "
            "non-positive pivot. A symmetric indefinite B needs QZ, which is not "
            "implemented.");
    }

    // One factorisation of L, reused for all four triangular solves below.
    Decomposition<double> dL = L.factorize();
    Matrix<double> Ad(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) Ad(int(i), int(j)) = double(std::real(A(int(i), int(j))));

    Matrix<double> Zm = dL.solve(Ad);              // Z  = L⁻¹ A
    Matrix<double> C = dL.solve(Zm.T()).T();       // C  = (L⁻¹ Zᵀ)ᵀ = L⁻¹ A L⁻ᵀ
    // C is symmetric in exact arithmetic; symmetrise so that rounding cannot
    // hand the symmetric eigensolver something it is entitled to reject.
    C = (C + C.T()) * 0.5;

    auto [vals, Y] = C.eig();

    // x = L⁻ᵀ y, which is a solve against Lᵀ — NOT against L. Getting this wrong
    // is silent: the shapes all match and the eigenvalues are still right, only
    // the vectors are garbage, so the residual ||Ax - lambda Bx|| is the test
    // that catches it and there is one in validate.cpp.
    Matrix<double> X = L.T().factorize().solve(Y);

    // A.eig() does not order its output, and MATLAB's symmetric-definite solver
    // does, so sort here and carry the vectors along with their values.
    std::vector<long> order((std::size_t)n);
    std::iota(order.begin(), order.end(), 0L);
    std::stable_sort(order.begin(), order.end(),
                     [&](long a, long b) { return vals(int(a), 0) < vals(int(b), 0); });
    Matrix<double> valsOut(n, 1), Xout(n, n);
    for (long j = 0; j < n; j++) {
        const long src = order[(std::size_t)j];
        valsOut(int(j), 0) = vals(int(src), 0);
        for (long i = 0; i < n; i++) Xout(int(i), int(j)) = X(int(i), int(src));
    }
    // DELIBERATELY NOT rescaled to unit Euclidean length. The right normalisation
    // for this problem is B-orthonormality, Xᵀ B X = I, which is what LAPACK's
    // dsygv returns and what makes the eigenvectors a basis in the metric B
    // defines. It comes out for free: Y is orthonormal from the symmetric
    // eigensolver, and X = L⁻ᵀ Y gives
    //     Xᵀ B X = Yᵀ L⁻¹ (L Lᵀ) L⁻ᵀ Y = Yᵀ Y = I.
    // Scaling the columns to Euclidean length 1 would DESTROY that — which is
    // what an earlier version of this function did, and what the Xᵀ B X == I
    // assertion in validate.cpp caught.
    return {valsOut, Xout};
}

// eigvals(A, B) — the general pencil A x = lambda B x, as the eigenvalues of
// B⁻¹A, and only when that is a defensible thing to compute.
//
// The guard is the point. cond(B) bounds how much accuracy forming B⁻¹A can
// cost, so it is estimated first — cheaply, with the same Hager-Higham
// estimator rcond() uses — and the call is refused outright when B is close to
// singular. Pass a smaller tol to override, knowing what it buys.
template <typename dtA, typename dtB>
Matrix<std::complex<double>> eigvals(const Matrix<dtA>& A, const Matrix<dtB>& B,
                                     double tol = 1e-10) {
    if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
        throw std::invalid_argument("eigvals(A,B): both matrices must be square and the "
                                    "same size");
    const double rc = rcond(B);
    if (rc < tol) {
        std::ostringstream os;
        os << "eigvals(A,B): B has a reciprocal condition number of " << rc
           << ", so forming B^-1*A would lose most of the available precision. This "
              "solver reduces the pencil that way and is only trustworthy for a well "
              "conditioned B; a singular or nearly singular B needs the QZ algorithm, "
              "which is not implemented. If you know what you are doing, lower the tol "
              "argument.";
        throw std::domain_error(os.str());
    }
    return B.solve(A).eigvals();
}

// polyeig({A0, A1, ..., Ad}) — eigenvalues of the matrix polynomial
//     P(lambda) = A0 + lambda*A1 + ... + lambda^d * Ad,
// the values of lambda for which P(lambda) is singular. Coefficients ASCENDING,
// matching MATLAB's polyeig(A0, A1, ..., Ad) argument order — and note that is
// the opposite of the DESCENDING order polyval, roots and polyfit use, because
// MATLAB itself is inconsistent here and matching it beats being quietly
// different.
//
// Solved by LINEARISATION: the degree-d n-by-n polynomial becomes a linear
// pencil of size d*n, whose eigenvalues are exactly P's. That is what MATLAB and
// every serious implementation do, because a companion pencil is well
// understood where root-finding on det(P(lambda)) is hopeless.
//
//     C1 = [ 0    I    0   ...  ]        C2 = diag(I, I, ..., I, Ad)
//          [ 0    0    I   ...  ]
//          [ ...                ]
//          [ -A0  -A1  ...  -Ad-1]
//
// A d*n-by-d*n problem for an n-by-n polynomial, so there are d*n eigenvalues.
// Requires Ad nonsingular; when it is not, some eigenvalues are infinite and the
// linearisation needs different handling, which the conditioning guard in
// eigvals(A,B) will catch and report.
template <typename datatype>
Matrix<std::complex<double>> polyeig(const std::vector<Matrix<datatype>>& coeffs,
                                     double tol = 1e-10) {
    if (coeffs.size() < 2)
        throw std::invalid_argument("polyeig: need at least two coefficient matrices "
                                    "(a constant and a linear term)");
    const long n = coeffs[0].rows();
    const long d = (long)coeffs.size() - 1;
    for (const auto& M : coeffs)
        if (M.rows() != n || M.cols() != n)
            throw std::invalid_argument("polyeig: every coefficient must be the same "
                                        "square size, expected " + std::to_string(n) +
                                        "x" + std::to_string(n));
    const long N = d * n;
    Matrix<double> C1(N, N), C2(N, N);
    for (long b = 0; b + 1 < d; b++)                    // super-diagonal identities
        for (long i = 0; i < n; i++) C1(int(b * n + i), int((b + 1) * n + i)) = 1.0;
    for (long b = 0; b < d; b++)                        // bottom row block: -A0..-A(d-1)
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                C1(int((d - 1) * n + i), int(b * n + j)) =
                    -double(std::real(coeffs[(std::size_t)b](int(i), int(j))));
    for (long b = 0; b + 1 < d; b++)                    // C2 = diag(I,...,I,Ad)
        for (long i = 0; i < n; i++) C2(int(b * n + i), int(b * n + i)) = 1.0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            C2(int((d - 1) * n + i), int((d - 1) * n + j)) =
                double(std::real(coeffs[(std::size_t)d](int(i), int(j))));
    return eigvals(C1, C2, tol);
}

// ═══════════════════════════════════════════════════════════════════════════
//  funm — a general matrix function  (tier 4)
// ═══════════════════════════════════════════════════════════════════════════
//
//     funm(A, f)   where f is any callable complex<double> -> complex<double>
//
//     funm(A, [](std::complex<double> z){ return std::exp(z); })   == exp(A)
//     funm(A, [](std::complex<double> z){ return std::sqrt(z); })  == sqrt(A)
//
// THIS IS NUMERIC, NOT SYMBOLIC. f is a function pointer or lambda that gets
// called on complex numbers; nothing here differentiates it, expands it, or
// looks at its form. That is a deliberate boundary — a matrix function library
// and a computer algebra system are different projects, and the moment this
// needed derivatives of f it would be sliding into the second one.
//
// It costs something, and the cost is stated below rather than hidden: the
// robust Schur-Parlett of Higham's algorithm reorders the Schur form into
// clusters and evaluates a TAYLOR SERIES on each diagonal block, which needs the
// derivatives f', f'', ... This implementation cannot do that, so it detects the
// case where it would be needed and refuses instead of returning noise.
//
// ── HOW IT WORKS ──────────────────────────────────────────────────────────
//   1. Real Schur:  A = Q T Qᵀ, T quasi-triangular.
//   2. Complexify:  each 2x2 block (a complex-conjugate eigenvalue pair) is
//      triangularised by a 2x2 unitary, giving a genuinely upper-triangular
//      complex Tc and a unitary Qc = Q Z. This is the piece the roadmap called
//      out as missing; doing it here means the Parlett recurrence below is
//      scalar throughout and never needs a Sylvester solve.
//   3. Parlett recurrence on Tc: the diagonal is f of the eigenvalues, and each
//      superdiagonal follows from the ones before it.
//   4. Undo:  f(A) = Qc F Qcᴴ.
//
// ── WHEN IT REFUSES, AND WHY THAT IS THE RIGHT ANSWER ─────────────────────
// Step 3 divides by (T_ii - T_jj). When two eigenvalues are close that division
// amplifies error without limit, and when they are equal it is undefined. The
// separation is checked against the size of the matrix and funm THROWS if it is
// too small, naming the two eigenvalues. A wrong f(A) that looks like a matrix
// is worse than an exception.
//
// If you hit that: exp, log, sqrt, sin, cos, sinh, cosh, tanh and pow all have
// dedicated implementations elsewhere in this header that do NOT go through
// Parlett — exp uses scaling-and-squaring, log the inverse of it — and those
// have no eigenvalue-separation requirement at all. funm is for the functions
// that do not have one.
namespace funm_detail {

using cplx = std::complex<double>;

// Real Schur form -> complex Schur form. Every 2x2 block on the diagonal holds a
// complex-conjugate pair; a 2x2 unitary built from one eigenvector triangularises
// it, and applying that to the whole matrix keeps everything else triangular.
inline void complexify(std::vector<cplx>& T, std::vector<cplx>& Z, long n,
                       const Matrix<double>& Treal) {
    for (long i = 0; i + 1 < n; i++) {
        // WHETHER THIS IS A REAL 2x2 BLOCK IS A RELATIVE QUESTION, not an exact
        // one. The QR iteration drives a converged subdiagonal towards zero
        // without ever setting it exactly to zero — on a 4x4 test matrix the
        // three subdiagonals came out {0, 2.8e-23, 3.5e-01}, and a `!= 0` test
        // read that 2.8e-23 as a block, consumed two rows on it, and then
        // stepped straight past the genuine block behind it. The result was
        // silently wrong: still triangular-looking, still real, just not f(A).
        // Matrix::isSchurBlock already encodes the right test; this is the same
        // criterion, spelled out here because that one is a private member.
        const double sub = std::abs(Treal(int(i + 1), int(i)));
        const double nbr = std::abs(Treal(int(i), int(i))) +
                           std::abs(Treal(int(i + 1), int(i + 1)));
        if (sub <= std::numeric_limits<double>::epsilon() * 100.0 * (nbr > 0.0 ? nbr : 1.0))
            continue;
        // A genuine 2x2 block. Its eigenvalues are the roots of the block's
        // characteristic polynomial; a negative discriminant is what makes them
        // a conjugate pair rather than two reals the QR iteration already split.
        const cplx a = T[(std::size_t)(i * n + i)],       b = T[(std::size_t)(i * n + i + 1)];
        const cplx c = T[(std::size_t)((i + 1) * n + i)], d = T[(std::size_t)((i + 1) * n + i + 1)];
        const cplx tr = a + d, det = a * d - b * c;
        const cplx disc = std::sqrt(tr * tr - 4.0 * det);
        const cplx lam = 0.5 * (tr + disc);
        // Eigenvector of the 2x2 block, taking whichever column is better scaled.
        cplx v0, v1;
        if (std::abs(b) >= std::abs(lam - a)) { v0 = b;       v1 = lam - a; }
        else                                  { v0 = lam - d; v1 = c; }
        const double nrm = std::sqrt(std::norm(v0) + std::norm(v1));
        if (nrm == 0.0) continue;
        v0 /= nrm; v1 /= nrm;
        // U = [v, w] with w orthonormal to v — in 2x2 that is just this.
        const cplx u00 = v0, u10 = v1;
        const cplx u01 = -std::conj(v1), u11 = std::conj(v0);
        // T := Uᴴ T U and Z := Z U, both touching only rows/columns i and i+1.
        for (long k = 0; k < n; k++) {                     // rows i, i+1 <- Uᴴ * rows
            const cplx r0 = T[(std::size_t)(i * n + k)], r1 = T[(std::size_t)((i + 1) * n + k)];
            T[(std::size_t)(i * n + k)]       = std::conj(u00) * r0 + std::conj(u10) * r1;
            T[(std::size_t)((i + 1) * n + k)] = std::conj(u01) * r0 + std::conj(u11) * r1;
        }
        for (long k = 0; k < n; k++) {                     // columns i, i+1 <- cols * U
            const cplx c0 = T[(std::size_t)(k * n + i)], c1 = T[(std::size_t)(k * n + i + 1)];
            T[(std::size_t)(k * n + i)]     = c0 * u00 + c1 * u10;
            T[(std::size_t)(k * n + i + 1)] = c0 * u01 + c1 * u11;
        }
        for (long k = 0; k < n; k++) {
            const cplx z0 = Z[(std::size_t)(k * n + i)], z1 = Z[(std::size_t)(k * n + i + 1)];
            Z[(std::size_t)(k * n + i)]     = z0 * u00 + z1 * u10;
            Z[(std::size_t)(k * n + i + 1)] = z0 * u01 + z1 * u11;
        }
        T[(std::size_t)((i + 1) * n + i)] = cplx(0.0, 0.0);   // exactly zero now
        i++;                                                  // skip the partner row
    }
}

}  // namespace funm_detail

template <typename datatype, typename F>
Matrix<std::complex<double>> funm(const Matrix<datatype>& A, F f) {
    using cplx = std::complex<double>;
    static_assert(!is_complex<datatype>::value,
                  "funm: not yet implemented for complex datatypes — it goes through the "
                  "REAL Schur form, which a complex matrix does not have.");
    if (A.rows() != A.cols())
        throw std::invalid_argument("funm: matrix must be square, got " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    const long n = A.rows();
    if (n == 0) throw std::invalid_argument("funm: matrix must be non-empty");

    auto [Treal, Qreal] = A.schur();

    // Promote to complex, then triangularise the 2x2 blocks.
    std::vector<cplx> T((std::size_t)(n * n)), Z((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n * n; i++) T[(std::size_t)i] = cplx(Treal[int(i)], 0.0);
    for (long i = 0; i < n; i++) Z[(std::size_t)(i * n + i)] = cplx(1.0, 0.0);
    funm_detail::complexify(T, Z, n, Treal);

    // ── Parlett recurrence ────────────────────────────────────────────────
    // F is upper triangular with the same structure as T. The diagonal is f of
    // the eigenvalues; each superdiagonal follows from the ones already known.
    std::vector<cplx> Fm((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n; i++) Fm[(std::size_t)(i * n + i)] = f(T[(std::size_t)(i * n + i)]);

    // Separation floor, relative to the size of the matrix — an absolute
    // tolerance would be meaningless for a matrix scaled by 10^6.
    double tscale = 0.0;
    for (long i = 0; i < n; i++) tscale = std::max(tscale, std::abs(T[(std::size_t)(i * n + i)]));
    const double sepTol = std::numeric_limits<double>::epsilon() *
                          std::max(1.0, tscale) * double(n);

    for (long d = 1; d < n; d++)                       // superdiagonal by superdiagonal
        for (long i = 0; i + d < n; i++) {
            const long j = i + d;
            const cplx tii = T[(std::size_t)(i * n + i)], tjj = T[(std::size_t)(j * n + j)];
            const cplx den = tii - tjj;
            if (std::abs(den) <= sepTol) {
                std::ostringstream os;
                os << "funm: eigenvalues " << i << " and " << j << " are too close to "
                   << "separate (" << tii.real() << (tii.imag() < 0 ? "" : "+") << tii.imag()
                   << "i and " << tjj.real() << (tjj.imag() < 0 ? "" : "+") << tjj.imag()
                   << "i). The Parlett recurrence divides by their difference, so the "
                   << "result would be noise. A robust answer needs the derivatives of f, "
                   << "which this deliberately does not compute — see the note above funm. "
                   << "If f is exp, log, sqrt, sin, cos, sinh, cosh, tanh or a power, use "
                   << "the dedicated version in this header instead: none of those go "
                   << "through Parlett.";
                throw std::domain_error(os.str());
            }
            cplx acc = T[(std::size_t)(i * n + j)] *
                       (Fm[(std::size_t)(i * n + i)] - Fm[(std::size_t)(j * n + j)]);
            for (long k = i + 1; k < j; k++)
                acc += Fm[(std::size_t)(i * n + k)] * T[(std::size_t)(k * n + j)] -
                       T[(std::size_t)(i * n + k)] * Fm[(std::size_t)(k * n + j)];
            Fm[(std::size_t)(i * n + j)] = acc / den;
        }

    // f(A) = (Q Z) F (Q Z)ᴴ.
    std::vector<cplx> QZ((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n; i++)
        for (long k = 0; k < n; k++) {
            const cplx q = cplx(Qreal(int(i), int(k)), 0.0);
            if (q == cplx(0.0, 0.0)) continue;
            for (long j = 0; j < n; j++)
                QZ[(std::size_t)(i * n + j)] += q * Z[(std::size_t)(k * n + j)];
        }
    std::vector<cplx> tmp((std::size_t)(n * n), cplx(0.0, 0.0));
    for (long i = 0; i < n; i++)
        for (long k = 0; k < n; k++) {
            const cplx a = QZ[(std::size_t)(i * n + k)];
            if (a == cplx(0.0, 0.0)) continue;
            for (long j = k; j < n; j++)                  // F is upper triangular
                tmp[(std::size_t)(i * n + j)] += a * Fm[(std::size_t)(k * n + j)];
        }
    Matrix<cplx> out(n, n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            cplx acc(0.0, 0.0);
            for (long k = 0; k < n; k++)
                acc += tmp[(std::size_t)(i * n + k)] * std::conj(QZ[(std::size_t)(j * n + k)]);
            out(int(i), int(j)) = acc;
        }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fast Fourier transform  (tier 6)
// ═══════════════════════════════════════════════════════════════════════════
//
//     fft(A)            fft(A, n)            fft(A, n, addcol)
//     ifft(A)           ifft(A, n)           ifft(A, n, addcol)
//     fftshift(A)       ifftshift(A)
//
// MATLAB's semantics throughout, because that is what makes the results
// checkable against numpy.fft rather than merely self-consistent:
//
//   * A VECTOR is transformed along its own length, whichever way it is
//     oriented, and comes back the same shape. A MATRIX is transformed COLUMN by
//     column. That is MATLAB's rule, and it is also the addcol=false convention
//     the rest of this header uses, so the two agree for free.
//   * n pads with zeros or truncates, exactly like MATLAB's fft(x, n).
//   * THE WHOLE 1/n GOES ON THE INVERSE. Forward is unnormalised. MATLAB and
//     NumPy both do this; a different split here would make every cross-check
//     fail for a reason that has nothing to do with correctness.
//   * The return type is Matrix<complex<double>> whatever went in — the
//     transform of real data is complex, so it cannot be Matrix<datatype>.
//
// ── THE DECISION THAT MATTERED: NON-POWER-OF-TWO LENGTHS ──────────────────
// Radix-2 Cooley-Tukey is thirty lines and handles only 2^k. The tempting
// shortcut is to zero-pad up to the next power of two — and it is WRONG, because
// padding computes the transform of a different, longer signal. It gives the
// right answer for a convolution and a quietly wrong one for a spectrum, which
// is the worst possible failure mode: no error, plausible numbers.
//
// So this does what MATLAB does and handles every length:
//     n a power of two  ->  iterative radix-2, the fast path
//     anything else     ->  Bluestein's chirp-z, which re-expresses the DFT as a
//                           convolution of length 2^k and hands it back to the
//                           radix-2 path
// Bluestein costs about 6x a same-size radix-2 and is O(n log n) for every n,
// including primes. A 1013-point transform (prime) stays microseconds, where the
// O(n^2) direct sum would not.
//
// ── THE OPTIMISATIONS, all of them earned earlier in this header ──────────
//   * PRECOMPUTED TWIDDLE TABLE, built once per call and shared by every column.
//     std::polar in the butterfly would put a sin and a cos on the critical path
//     of the innermost loop; the table turns that into one load.
//   * ITERATIVE, IN PLACE, with an explicit bit-reversal permutation. The
//     textbook recursive formulation allocates two vectors per level.
//   * __restrict on the working pointers, the single biggest win in this file
//     when it was applied to naiveMul (64 -> 175 GFLOP/s).
//   * NO operator() IN THE COPY IN AND OUT. It wraps negative indices, so it
//     runs two integer divisions per element — the same trap that held cholesky
//     to 0.78 GFLOP/s. One &A(0,0) and plain indexing after that: 17% at
//     n = 262144.
//   * PARALLEL OVER LINES when a matrix has several columns to transform. They
//     are completely independent, which is the cheapest parallelism there is.
//   * PARALLEL OVER BUTTERFLIES for one large transform. Within a stage the
//     butterflies touch disjoint pairs, so the stage parallelises; which loop to
//     split changes as the stage widens, so it splits blocks while there are
//     blocks and switches to splitting within the block for the last stages.
//   * ONE ALLOCATION PER LINE, reused across every stage, and the buffer is
//     declared inside the parallel loop so each thread owns its own.
//   * The inverse is conj -> forward -> conj -> scale rather than a second
//     kernel with flipped twiddles. It is provably correct given the forward
//     transform is, and it halves the code that can be wrong.
//
// ── WHERE THIS LANDS AGAINST NumPy, measured ──────────────────────────────
// NumPy uses pocketfft: mixed-radix, with hand-written codelets for radices
// 2/3/4/5/7/11 and cache blocking on top. This is a single-radix-2 kernel, so
// the honest summary is that it wins where its parallelism applies and loses
// where pocketfft's radix choice does.
//
//   ONE 1-D TRANSFORM (in the permanent benchmark, bench/*_fft.csv):
//       n = 2048     0.53x        n = 262144   0.65x
//       n = 65536    0.43x        n = 1048576  0.89x
//     Slower throughout, converging towards parity as n grows and memory
//     bandwidth rather than radix choice starts to decide it.
//
//   MANY TRANSFORMS AT ONCE — the case this wins, and it wins clearly:
//       1024 x 64    0.114 ms  against NumPy's 0.226 ms    1.98x
//       1024 x 512   1.232 ms  against NumPy's 3.185 ms    2.59x
//     pocketfft does not thread. Transforming the columns of a matrix is
//     perfect parallelism and it is simply not available to NumPy here.
//
//   NON-POWER-OF-TWO: n = 1013 (prime) 0.62x, n = 65537 (prime) 0.48x, but
//     n = 1000 only 0.10x — because 1000 = 2^3 * 5^3 and pocketfft factors it
//     with radix-5 codelets, where this has to fall back to Bluestein and do
//     three transforms of length 2048. That gap is the price of one radix.
//
// THE ONE OPTIMISATION LEFT THAT WOULD MOVE THIS: a radix-4 butterfly. It
// halves the number of passes over memory against radix-2, which is where the
// remaining 2x on mid-sized transforms lives. Mixed-radix beyond that (3, 5)
// would close the n = 1000 case too, at a lot more code.
namespace fft_detail {

    using cplx = std::complex<double>;

    inline bool isPow2(long n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
    inline long ceilPow2(long n) {
        long p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    // roots[j] = exp(-2*pi*i*j/n) for j < n/2 — every twiddle any stage needs, since
    // a stage of length `len` uses roots[j * (n/len)].
    inline std::vector<cplx> twiddles(long n) {
        std::vector<cplx> w((std::size_t)(n / 2));
        const double s = -2.0 * mconst::pi / double(n);
        for (long j = 0; j < n / 2; j++)
            w[(std::size_t)j] = std::polar(1.0, s * double(j));
        return w;
    }

    // ── Thread tuning, measured rather than assumed ───────────────────────────
    // A single transform, milliseconds, on a 16-core / 32-thread machine:
    //
    //     n           t=1     t=2     t=4     t=8    t=16    t=32
    //     4096       0.064   0.065   0.065   0.065   0.065   0.065
    //     16384      0.412   0.418   0.401   0.411   0.427   0.489
    //     65536      1.838   1.606   1.480   1.566   1.638   1.831
    //     262144     7.736   6.353   5.671   5.383   6.330   7.512
    //     1048576   35.677  27.683  24.414  23.329  26.026  26.412
    //
    // Two things fall out of that, and both are the opposite of the naive choice:
    //   * Below ~65536 there is NOTHING to gain — at 16384 every thread count is
    //     within noise of serial, and 32 threads is measurably worse. The butterfly
    //     stages are short and the whole array is in cache.
    //   * The optimum is FOUR TO EIGHT threads, never sixteen or thirty-two. Early
    //     stages stride across the whole array, so a transform is bound by memory
    //     latency rather than arithmetic and saturates long before the core count.
    //     Erring low costs under 5% (t=4 against t=8); erring high costs over 30%.
    // Whole LINES are different — independent transforms, each one cache-resident —
    // and those do scale to memoryThreads(), so they get their own threshold.
    inline constexpr long FFT_PARALLEL_MIN = 65536;  // butterflies within one line
    inline constexpr long FFT_LINES_MIN = 32768;     // elements across all lines

    inline int butterflyThreads() {
        return std::min(8, mstore::memoryThreads());
    }

    // In-place forward radix-2, n a power of two, w from twiddles(n).
    inline void radix2(cplx* MATRIXCPP_RESTRICT a,
                       long n,
                       const cplx* MATRIXCPP_RESTRICT w,
                       bool allowParallel) {
        if (n < 2)
            return;
        // Bit-reversal permutation, by incrementing a reversed counter rather than
        // reversing each index from scratch.
        for (long i = 1, j = 0; i < n; i++) {
            long bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap(a[i], a[j]);
        }
        const bool par = allowParallel && n >= FFT_PARALLEL_MIN;
        (void)par;
        for (long len = 2; len <= n; len <<= 1) {
            const long half = len >> 1;
            const long step = n / len;
            const long blocks = n / len;
            (void)blocks;  // only read on the OpenMP path
#ifdef _OPENMP
            const int nth = butterflyThreads();
            if (par && blocks > 1) {
    // Early stages: many short blocks, split those.
    #pragma omp parallel for schedule(static) num_threads(nth)
                for (long b = 0; b < blocks; b++) {
                    const long i = b * len;
                    for (long j = 0; j < half; j++) {
                        const cplx u = a[i + j];
                        const cplx v = a[i + j + half] * w[j * step];
                        a[i + j] = u + v;
                        a[i + j + half] = u - v;
                    }
                }
                continue;
            }
            if (par) {
    // Late stages: one wide block, so split inside it instead.
    #pragma omp parallel for schedule(static) num_threads(nth)
                for (long j = 0; j < half; j++) {
                    const cplx u = a[j];
                    const cplx v = a[j + half] * w[j * step];
                    a[j] = u + v;
                    a[j + half] = u - v;
                }
                continue;
            }
#endif
            for (long i = 0; i < n; i += len)
                for (long j = 0; j < half; j++) {
                    const cplx u = a[i + j];
                    const cplx v = a[i + j + half] * w[j * step];
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
        }
    }

    // Bluestein's chirp-z: a DFT of ANY length as a convolution of power-of-two
    // length. X_k = conj(c_k) * sum_j (x_j c_j) * conj(c_{k-j}) with c_k the chirp
    // exp(-i*pi*k^2/n), and that sum is a convolution the radix-2 path can do.
    inline void bluestein(cplx* MATRIXCPP_RESTRICT a, long n, bool allowParallel) {
        const long m = ceilPow2(2 * n - 1);
        std::vector<cplx> chirp((std::size_t)n), A((std::size_t)m, cplx(0.0, 0.0)),
            B((std::size_t)m, cplx(0.0, 0.0));
        for (long k = 0; k < n; k++) {
            // k*k reduced modulo 2n BEFORE it becomes an angle. k^2 overflows the
            // useful range of a double's mantissa long before it overflows long, and
            // the phase only depends on k^2 mod 2n, so reducing first keeps every
            // digit that matters.
            const long kk = (k * k) % (2 * n);
            chirp[(std::size_t)k] = std::polar(1.0, -mconst::pi * double(kk) / double(n));
        }
        for (long k = 0; k < n; k++)
            A[(std::size_t)k] = a[k] * chirp[(std::size_t)k];
        B[0] = std::conj(chirp[0]);
        for (long k = 1; k < n; k++) {
            B[(std::size_t)k] = std::conj(chirp[(std::size_t)k]);
            B[(std::size_t)(m - k)] = std::conj(chirp[(std::size_t)k]);
        }
        const std::vector<cplx> w = twiddles(m);
        radix2(A.data(), m, w.data(), allowParallel);
        radix2(B.data(), m, w.data(), allowParallel);
        for (long k = 0; k < m; k++)
            A[(std::size_t)k] *= B[(std::size_t)k];
        // Inverse of the size-m transform, by the conjugate identity.
        for (long k = 0; k < m; k++)
            A[(std::size_t)k] = std::conj(A[(std::size_t)k]);
        radix2(A.data(), m, w.data(), allowParallel);
        const double inv = 1.0 / double(m);
        for (long k = 0; k < n; k++)
            a[k] = std::conj(A[(std::size_t)k]) * inv * chirp[(std::size_t)k];
    }

    // One line, forward, any length.
    inline void forward(cplx* a, long n, const std::vector<cplx>& w, bool allowParallel) {
        if (n < 2)
            return;
        if (isPow2(n))
            radix2(a, n, w.data(), allowParallel);
        else
            bluestein(a, n, allowParallel);
    }

    // ifft(x) == conj(fft(conj(x))) / n. One kernel, not two.
    inline void backward(cplx* a, long n, const std::vector<cplx>& w, bool allowParallel) {
        if (n < 1)
            return;
        for (long k = 0; k < n; k++)
            a[k] = std::conj(a[k]);
        forward(a, n, w, allowParallel);
        const double inv = 1.0 / double(n);
        for (long k = 0; k < n; k++)
            a[k] = std::conj(a[k]) * inv;
    }

    // Shared driver. Pulls each line out into a contiguous buffer (padding or
    // truncating to L), transforms it, writes it back.
    template <typename datatype>
    inline Matrix<cplx> run(const Matrix<datatype>& A, long L, bool addcol, bool invert) {
        const long rows = A.rows(), cols = A.cols();
        if (rows == 0 || cols == 0)
            return Matrix<cplx>(rows, cols);
        const long lineLen = addcol ? cols : rows;
        if (L < 0)
            L = lineLen;
        if (L == 0)
            return Matrix<cplx>(addcol ? rows : 0, addcol ? 0 : cols);

        const long outRows = addcol ? rows : L;
        const long outCols = addcol ? L : cols;
        const long nLines = addcol ? rows : cols;
        Matrix<cplx> out(outRows, outCols);

        // The twiddle table is built ONCE and shared by every line — it depends only
        // on the length. Bluestein builds its own for the padded size internally.
        const std::vector<cplx> w = isPow2(L) ? twiddles(L) : std::vector<cplx>();

        // Lines are independent, so this is the cheap parallelism; a single line
        // only splits its butterflies when there is nothing else to split.
        const bool parLines = nLines > 1 && (long)(nLines * L) >= FFT_LINES_MIN;
        const bool parInner = !parLines;

        // Storage is row-major and contiguous, so one operator() call gives the base
        // pointer and every element after that is plain indexing. Going through
        // operator() per element instead would run TWO integer divisions each time —
        // it wraps negative indices — which is the same trap that was holding
        // cholesky to 0.78 GFLOP/s before it was found there.
        const datatype* MATRIXCPP_RESTRICT Ap = &A(0, 0);
        cplx* MATRIXCPP_RESTRICT Op = &out(0, 0);

        auto doLine = [&](long line) {
            std::vector<cplx> buf((std::size_t)L, cplx(0.0, 0.0));
            const long copy = std::min(L, lineLen);
            for (long k = 0; k < copy; k++) {
                const datatype v = addcol ? Ap[line * cols + k] : Ap[k * cols + line];
                if constexpr (is_complex<datatype>::value)
                    buf[(std::size_t)k] = cplx(v.real(), v.imag());
                else
                    buf[(std::size_t)k] = cplx(double(v), 0.0);
            }
            if (invert)
                backward(buf.data(), L, w, parInner);
            else
                forward(buf.data(), L, w, parInner);
            for (long k = 0; k < L; k++) {
                if (addcol)
                    Op[line * outCols + k] = buf[(std::size_t)k];
                else
                    Op[k * outCols + line] = buf[(std::size_t)k];
            }
        };

#ifdef _OPENMP
        if (parLines) {
    // Lines ARE independent and each is cache-resident, so unlike the
    // butterfly stages these scale all the way to memoryThreads().
    #pragma omp parallel for schedule(static) num_threads(mstore::memoryThreads())
            for (long line = 0; line < nLines; line++)
                doLine(line);
            return out;
        }
#endif
        for (long line = 0; line < nLines; line++)
            doLine(line);
        return out;
    }

    // A vector is transformed along its own length whichever way it is oriented —
    // MATLAB's rule — and a matrix column by column.
    template <typename datatype>
    inline bool autoAxis(const Matrix<datatype>& A) {
        return A.rows() == 1 && A.cols() != 1;
    }

}  // namespace fft_detail

// fft(A) / fft(A, n) — MATLAB's default axis: along a vector, down a matrix's
// columns. n pads with zeros or truncates.
template <typename datatype>
Matrix<std::complex<double>> fft(const Matrix<datatype>& A, long n = -1) {
    return fft_detail::run(A, n, fft_detail::autoAxis(A), false);
}
// fft(A, n, addcol) — explicit axis. false works down columns, true along rows,
// the same flag sum() and cumsum() take.
template <typename datatype>
Matrix<std::complex<double>> fft(const Matrix<datatype>& A, long n, bool addcol) {
    return fft_detail::run(A, n, addcol, false);
}

template <typename datatype>
Matrix<std::complex<double>> ifft(const Matrix<datatype>& A, long n = -1) {
    return fft_detail::run(A, n, fft_detail::autoAxis(A), true);
}
template <typename datatype>
Matrix<std::complex<double>> ifft(const Matrix<datatype>& A, long n, bool addcol) {
    return fft_detail::run(A, n, addcol, true);
}

// fftshift — swaps the halves of each line, moving the zero frequency from index
// 0 to the middle, which is how a spectrum is almost always looked at.
// For an ODD length the two halves differ by one, so fftshift and ifftshift are
// NOT the same operation; ifftshift is the exact inverse.
template <typename datatype>
Matrix<datatype> fftshift(const Matrix<datatype>& A) {
    const bool addcol = fft_detail::autoAxis(A);
    const long n = addcol ? A.cols() : A.rows();
    return A.circshift(n / 2, addcol ? 1 : 0);
}
template <typename datatype>
Matrix<datatype> ifftshift(const Matrix<datatype>& A) {
    const bool addcol = fft_detail::autoAxis(A);
    const long n = addcol ? A.cols() : A.rows();
    return A.circshift(-(n / 2), addcol ? 1 : 0);
}
// kron(A, B) — Kronecker product, MATLAB's spelling. See Matrix::kron.
template <typename datatype>
Matrix<datatype> kron(const Matrix<datatype>& A, const Matrix<datatype>& B) {
    return A.kron(B);
}

template <typename datatype>
Matrix<datatype> diag(const Matrix<datatype>& v) {
    long n = v.rows() * v.cols();
    if (v.rows() != 1 && v.cols() != 1)
        throw std::invalid_argument(
            "diag(v): expected a row or column vector, got " + std::to_string(v.rows()) + "x" +
            std::to_string(v.cols()) +
            " — to extract a diagonal from a matrix use the member A.diag()");
    if (n == 0)
        throw std::invalid_argument("diag(v): vector must be non-empty");
    Matrix<datatype> out(n, n);
    for (long i = 0; i < n; i++)
        out(int(i), int(i)) = v[int(i)];
    return out;
}

// Scalar multiplication with scalar on the left: k * A.
// Complements the member operator A * k so both orderings work.
// --- Element-wise dot-operator sugar (see dot_t above) ---
// Deliberately a distinct type per side so that a stray `A * dot` cannot be
// mistaken for anything else, and so the second operand is checked at compile
// time rather than silently deducing scalar = dot_t in Matrix::operator*.
template <typename datatype>
struct ElemMulLhs {
    const Matrix<datatype>* a;
};
template <typename datatype>
struct ElemDivLhs {
    const Matrix<datatype>* a;
};

template <typename datatype>
ElemMulLhs<datatype> operator*(const Matrix<datatype>& A, dot_t) {
    return {&A};
}
template <typename datatype>
ElemDivLhs<datatype> operator/(const Matrix<datatype>& A, dot_t) {
    return {&A};
}

template <typename datatype>
Matrix<datatype> operator*(ElemMulLhs<datatype> lhs, const Matrix<datatype>& B) {
    return lhs.a->mul(B);
}
template <typename datatype>
Matrix<datatype> operator/(ElemDivLhs<datatype> lhs, const Matrix<datatype>& B) {
    return lhs.a->div(B);
}

template <typename datatype, typename scalar>
Matrix<datatype> operator*(const scalar k, Matrix<datatype> A) {
    return A * k;
}

// Stream insertion: allows std::cout << A and writing to any std::ostream.
// Uses default precision (6dp). For custom precision call A.toString(n)
// directly.
template <typename datatype>
std::ostream& operator<<(std::ostream& os, const Matrix<datatype>& M) {
    return os << M.toString();
}

// Prints two matrices side by side with an operator symbol centred on the
// middle row. A:        left matrix op:       operator string shown between
// them, e.g. "*", "+", "=" B:        right matrix precision: decimal places for
// floating-point types (default 6) Handles mismatched row counts by padding the
// shorter matrix with blank lines.
template <typename datatype>
void printSideBySide(const Matrix<datatype>& A,
                     const std::string& op,
                     const Matrix<datatype>& B,
                     int precision = 6) {
    auto linesA = A.toLines(precision);
    auto linesB = B.toLines(precision);

    size_t rowsA = linesA.size();
    size_t rowsB = linesB.size();
    size_t totalRows = rowsA > rowsB ? rowsA : rowsB;

    // Width of a blank line matching A's and B's row width
    size_t widthA = rowsA > 0 ? linesA[0].size() : 0;
    size_t widthB = rowsB > 0 ? linesB[0].size() : 0;
    std::string blankA(widthA, ' ');
    std::string blankB(widthB, ' ');

    // op column: symbol on middle row, spaces elsewhere
    size_t midRow = totalRows / 2;
    std::string opPad(op.size(), ' ');

    for (size_t r = 0; r < totalRows; r++) {
        const std::string& rowA = r < rowsA ? linesA[r] : blankA;
        const std::string& rowB = r < rowsB ? linesB[r] : blankB;
        const std::string& sym = r == midRow ? op : opPad;
        std::cout << rowA << "   " << sym << "   " << rowB << '\n';
    }
}

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