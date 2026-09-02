#pragma once

// ==========================================================================
//  Mathematical constants and literals
// ==========================================================================
//
// mconst::pi and friends, aliasing std::numbers where the compiler has it,
// plus the _i / _deg user-defined literals.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "mstore.hpp"

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
