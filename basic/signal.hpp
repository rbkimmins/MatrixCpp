#pragma once

// ==========================================================================
//  Transforms, calculus and interpolation
// ==========================================================================
//
// fft / ifft / fftshift, conv / deconv, filter, interp1, trapz / cumtrapz and
// gradient. The tier-6 group: things that treat a matrix as sampled data.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "eigen.hpp"

namespace mcpu {


// ═══════════════════════════════════════════════════════════════════════════
//  Fast Fourier transform  (tier 6)
// ═══════════════════════════════════════════════════════════════════════════
//
//     fft(A)            fft(A, n)            fft(A, n, axis)
//     ifft(A)           ifft(A, n)           ifft(A, n, axis)
//     fftshift(A)       ifftshift(A)
//
// MATLAB's semantics throughout, because that is what makes the results
// checkable against numpy.fft rather than merely self-consistent:
//
//   * A VECTOR is transformed along its own length, whichever way it is
//     oriented, and comes back the same shape. A MATRIX is transformed COLUMN by
//     column. That is MATLAB's rule, and it is also the axis=false convention
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
//     when it was applied to naiveMul.
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
    inline Matrix<cplx> run(const Matrix<datatype>& A, long L, bool axis, bool invert) {
        const long rows = A.rows(), cols = A.cols();
        if (rows == 0 || cols == 0)
            return Matrix<cplx>(rows, cols);
        const long lineLen = axis ? cols : rows;
        if (L < 0)
            L = lineLen;
        if (L == 0)
            return Matrix<cplx>(axis ? rows : 0, axis ? 0 : cols);

        const long outRows = axis ? rows : L;
        const long outCols = axis ? L : cols;
        const long nLines = axis ? rows : cols;
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
                const datatype v = axis ? Ap[line * cols + k] : Ap[k * cols + line];
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
                if (axis)
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
// fft(A, n, axis) — explicit axis. false works down columns, true along rows,
// the same flag sum() and cumsum() take.
template <typename datatype>
Matrix<std::complex<double>> fft(const Matrix<datatype>& A, long n, bool axis) {
    return fft_detail::run(A, n, axis, false);
}

template <typename datatype>
Matrix<std::complex<double>> ifft(const Matrix<datatype>& A, long n = -1) {
    return fft_detail::run(A, n, fft_detail::autoAxis(A), true);
}
template <typename datatype>
Matrix<std::complex<double>> ifft(const Matrix<datatype>& A, long n, bool axis) {
    return fft_detail::run(A, n, axis, true);
}

// fftshift — swaps the halves of each line, moving the zero frequency from index
// 0 to the middle, which is how a spectrum is almost always looked at.
// For an ODD length the two halves differ by one, so fftshift and ifftshift are
// NOT the same operation; ifftshift is the exact inverse.
template <typename datatype>
Matrix<datatype> fftshift(const Matrix<datatype>& A) {
    const bool axis = fft_detail::autoAxis(A);
    const long n = axis ? A.cols() : A.rows();
    return A.circshift(n / 2, axis ? 1 : 0);
}
template <typename datatype>
Matrix<datatype> ifftshift(const Matrix<datatype>& A) {
    const bool axis = fft_detail::autoAxis(A);
    const long n = axis ? A.cols() : A.rows();
    return A.circshift(-(n / 2), axis ? 1 : 0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Convolution, polynomials, interpolation and filtering  (tier 6)
// ═══════════════════════════════════════════════════════════════════════════
//
// These are FREE functions, not members, because each treats a whole vector as
// one signal rather than working along an axis of an array. trapz, cumtrapz and
// gradient went the other way, and are members taking the same axis flag as
// sum() and cumsum(), for exactly that reason.
//
// Polynomial coefficients are DESCENDING throughout, matching polyval, polyfit
// and roots. conv is the same operation either way — a convolution does not care
// which end you start from, as long as both operands agree — but poly() and
// deconv() return coefficients, and those have to match or roots(conv(a,b))
// would be quietly wrong.

namespace conv_detail {
// Above this many multiply-adds the FFT route beats the direct double loop.
// MEASURED, and the first guess was 100000 — about ten times too high, which
// would have skipped the FFT across a whole range where it is twice as fast.
// Direct against FFT, milliseconds, for two length-n operands:
//
//     n         48      64      80      96     112     128     192     512
//     work    2304    4096    6400    9216   12544   16384   36864  262144
//     direct 0.0059  0.0105  0.0164  0.0236  0.0321  0.0401  0.0901  0.6390
//     FFT    0.0155  0.0170  0.0308  0.0324  0.0332  0.0338  0.0654  0.1839
//
// The crossover sits at work ~= 13000-16000, so 16384 is the threshold and the
// direct loop keeps everything below it: no transform, no padding to a power of
// two, no complex arithmetic. Same shape of decision as STRASSEN_THRESHOLD, and
// the same lesson — the number has to be measured, not reasoned about.
inline constexpr long CONV_FFT_MIN_WORK = 16384;
}  // namespace conv_detail

// conv(a, b) — discrete convolution, equivalently polynomial multiplication.
// Result length is na + nb - 1. MATLAB conv, NumPy np.convolve.
template <typename dtA, typename dtB>
Matrix<double> conv(const Matrix<dtA>& a, const Matrix<dtB>& b) {
    const long na = a.rows() * a.cols(), nb = b.rows() * b.cols();
    if (na == 0 || nb == 0)
        throw std::invalid_argument("conv: both operands must be non-empty");
    const long nc = na + nb - 1;
    Matrix<double> out(1, nc);

    if (na * nb < conv_detail::CONV_FFT_MIN_WORK) {
        for (long i = 0; i < na; i++) {
            const double av = double(std::real(a[int(i)]));
            if (av == 0.0) continue;
            for (long j = 0; j < nb; j++)
                out(0, int(i + j)) += av * double(std::real(b[int(j)]));
        }
        return out;
    }

    // The convolution theorem: pad BOTH operands to the same length nc (or more)
    // and the transform of the product is the product of the transforms. The
    // padding here is legitimate precisely because the answer wanted IS the
    // longer signal's — which is the one case where zero-padding an FFT is
    // correct, and the reason fft() itself refuses to do it silently.
    Matrix<double> ap(1, nc), bp(1, nc);
    for (long i = 0; i < na; i++) ap(0, int(i)) = double(std::real(a[int(i)]));
    for (long j = 0; j < nb; j++) bp(0, int(j)) = double(std::real(b[int(j)]));
    Matrix<std::complex<double>> prod = fft(ap) % fft(bp);
    Matrix<std::complex<double>> back = ifft(prod);
    for (long k = 0; k < nc; k++) out(0, int(k)) = back(0, int(k)).real();
    return out;
}

// deconv(y, a) — polynomial long division. Returns {quotient, remainder} with
//     y == conv(a, quotient) + remainder
// exactly, which is the identity worth testing rather than the coefficients.
template <typename dtY, typename dtA>
std::pair<Matrix<double>, Matrix<double>> deconv(const Matrix<dtY>& y,
                                                 const Matrix<dtA>& a) {
    const long ny = y.rows() * y.cols(), na = a.rows() * a.cols();
    if (na == 0 || ny == 0)
        throw std::invalid_argument("deconv: both operands must be non-empty");
    if (double(std::real(a[0])) == 0.0)
        throw std::invalid_argument(
            "deconv: the divisor's leading coefficient is zero, so the division is "
            "undefined. Strip the leading zeros first — roots() does the same thing "
            "for the same reason.");
    if (ny < na) {                       // nothing to divide into
        Matrix<double> q(1, 1);
        Matrix<double> r(1, ny);
        for (long i = 0; i < ny; i++) r(0, int(i)) = double(std::real(y[int(i)]));
        return {q, r};
    }
    const long nq = ny - na + 1;
    std::vector<double> work((std::size_t)ny);
    for (long i = 0; i < ny; i++) work[(std::size_t)i] = double(std::real(y[int(i)]));
    Matrix<double> q(1, nq);
    const double lead = double(std::real(a[0]));
    for (long k = 0; k < nq; k++) {
        const double c = work[(std::size_t)k] / lead;
        q(0, int(k)) = c;
        if (c == 0.0) continue;
        for (long j = 0; j < na; j++)
            work[(std::size_t)(k + j)] -= c * double(std::real(a[int(j)]));
    }
    Matrix<double> r(1, ny);             // same length as y, as MATLAB returns
    for (long i = 0; i < ny; i++) r(0, int(i)) = work[(std::size_t)i];
    return {q, r};
}

// poly(v) — MATLAB's overloaded one, and the overloading is MATLAB's rule too:
//   * v a VECTOR  -> the monic polynomial whose roots are v, so that
//                    roots(poly(v)) returns v back up to ordering and rounding.
//   * v a SQUARE MATRIX -> its characteristic polynomial, det(lambda*I - A),
//                    which is poly(eigvals(A)).
// One function rather than two overloads, because a vector IS a matrix and the
// two would be ambiguous for a 1x1 — this way the rule is explicit and checked.
//
// Built up with complex arithmetic even for a real input, because the roots of a
// real polynomial come in conjugate pairs whose imaginary parts only cancel once
// the whole product is formed. The result is real to rounding, and that is
// asserted rather than assumed.
template <typename datatype>
Matrix<double> poly(const Matrix<datatype>& v) {
    using cplx = std::complex<double>;
    std::vector<cplx> r;
    const bool isVector = (v.rows() == 1 || v.cols() == 1);
    if (isVector) {
        for (long i = 0; i < v.rows() * v.cols(); i++) {
            const auto e = v[int(i)];
            if constexpr (is_complex<datatype>::value) r.push_back(cplx(e.real(), e.imag()));
            else                                       r.push_back(cplx(double(e), 0.0));
        }
    } else if (v.rows() == v.cols()) {
        Matrix<cplx> ev = v.eigvals();
        for (long i = 0; i < ev.rows(); i++) r.push_back(ev[int(i)]);
    } else {
        throw std::invalid_argument(
            "poly: expected a vector (its roots) or a square matrix (its characteristic "
            "polynomial), got " + std::to_string(v.rows()) + "x" + std::to_string(v.cols()));
    }

    std::vector<cplx> p{cplx(1.0, 0.0)};          // monic, descending
    for (const cplx& root : r) {
        std::vector<cplx> next(p.size() + 1, cplx(0.0, 0.0));
        for (std::size_t i = 0; i < p.size(); i++) {
            next[i]     += p[i];
            next[i + 1] -= p[i] * root;
        }
        p.swap(next);
    }
    Matrix<double> out(1, (long)p.size());
    for (std::size_t i = 0; i < p.size(); i++) out(0, int(i)) = p[i].real();
    return out;
}

// interp1(x, y, xi) — linear interpolation of the samples (x, y) at xi.
//
// Two decisions, both MATLAB's:
//   * x MUST BE SORTED ASCENDING, and that is CHECKED rather than assumed.
//     Sorting internally would hide a caller bug that is almost always real;
//     the check is one pass and the error names the offending index.
//   * OUT OF RANGE GIVES THE FILL VALUE, which defaults to NaN. MATLAB returns
//     NaN, NumPy's np.interp clamps to the end values instead. They genuinely
//     disagree, and this follows MATLAB — extrapolating off the end of a sample
//     set is a decision the caller should make, not one made for them.
//
//     ⚠ THE NaN DEFAULT DOES NOT SURVIVE -ffast-math. That flag implies
//     -ffinite-math-only, under which the compiler may assume no NaN ever
//     exists: std::isnan() folds to false and the sentinel becomes
//     undetectable. This is the third time that flag has bitten this header (see
//     the BUILD FLAG WARNING near the top), and here it is not just a test
//     artifact — the feature genuinely stops working.
//
//     Pass an explicit `fill` when that matters. It is MATLAB's own escape
//     hatch, it works in every build, and any finite value is detectable:
//         interp1(x, y, xq, 0.0)                 // zero outside the range
//         interp1(x, y, xq, y[0])                // clamp-ish, NumPy's habit
template <typename datatype>
Matrix<double> interp1(const Matrix<datatype>& x, const Matrix<datatype>& y,
                       const Matrix<datatype>& xi,
                       double fill = std::numeric_limits<double>::quiet_NaN()) {
    const long n = x.rows() * x.cols();
    if (n != y.rows() * y.cols())
        throw std::invalid_argument("interp1: x and y must have the same length, got " +
                                    std::to_string(n) + " and " +
                                    std::to_string(y.rows() * y.cols()));
    if (n < 2) throw std::invalid_argument("interp1: need at least two sample points");
    std::vector<double> xs((std::size_t)n), ys((std::size_t)n);
    for (long i = 0; i < n; i++) {
        xs[(std::size_t)i] = double(std::real(x[int(i)]));
        ys[(std::size_t)i] = double(std::real(y[int(i)]));
        if (i && !(xs[(std::size_t)i] > xs[(std::size_t)(i - 1)]))
            throw std::invalid_argument(
                "interp1: x must be strictly ascending, but x[" + std::to_string(i - 1) +
                "] = " + std::to_string(xs[(std::size_t)(i - 1)]) + " and x[" +
                std::to_string(i) + "] = " + std::to_string(xs[(std::size_t)i]));
    }
    Matrix<double> out(xi.rows(), xi.cols());
    for (long k = 0; k < xi.rows() * xi.cols(); k++) {
        const double q = double(std::real(xi[int(k)]));
        if (q < xs.front() || q > xs.back()) { out[int(k)] = fill; continue; }
        // lower_bound over a sorted range — O(log n) per query, which is why the
        // ascending requirement is worth enforcing rather than working around.
        const auto it = std::lower_bound(xs.begin(), xs.end(), q);
        std::size_t hi = (std::size_t)(it - xs.begin());
        if (hi == 0) { out[int(k)] = ys[0]; continue; }         // exactly the first point
        const std::size_t lo = hi - 1;
        const double t = (q - xs[lo]) / (xs[hi] - xs[lo]);
        out[int(k)] = ys[lo] + t * (ys[hi] - ys[lo]);
    }
    return out;
}

// filter(b, a, x) — the rational-transfer-function difference equation:
//     a(0)*y(n) = b(0)*x(n) + b(1)*x(n-1) + ... - a(1)*y(n-1) - a(2)*y(n-2) - ...
// MATLAB's filter, with the same argument order and the same normalisation by
// a(0).
//
// THIS IS A RECURRENCE, so unlike everything else in this tier it cannot be
// parallelised over the output: y(n) depends on y(n-1). Worth stating plainly so
// nobody comes back later and tries. It is also why an IIR filter and a
// convolution are different operations even though both look like sliding sums.
template <typename dtB, typename dtA, typename dtX>
Matrix<double> filter(const Matrix<dtB>& b, const Matrix<dtA>& a, const Matrix<dtX>& x) {
    const long nb = b.rows() * b.cols(), na = a.rows() * a.cols();
    const long nx = x.rows() * x.cols();
    if (nb == 0 || na == 0) throw std::invalid_argument("filter: b and a must be non-empty");
    const double a0 = double(std::real(a[0]));
    if (a0 == 0.0)
        throw std::invalid_argument("filter: a(0) must be non-zero — the difference "
                                    "equation is divided through by it");
    Matrix<double> out(x.rows(), x.cols());
    for (long n = 0; n < nx; n++) {
        double acc = 0.0;
        for (long j = 0; j < nb && j <= n; j++)
            acc += double(std::real(b[int(j)])) * double(std::real(x[int(n - j)]));
        for (long j = 1; j < na && j <= n; j++)
            acc -= double(std::real(a[int(j)])) * out[int(n - j)];
        out[int(n)] = acc / a0;
    }
    return out;
}

// trapz(x, y) — trapezoidal integral over given sample points, MATLAB's
// two-argument form. The member y.trapz() covers the unit-spacing case.
template <typename datatype>
double trapz(const Matrix<datatype>& x, const Matrix<datatype>& y) {
    const long n = x.rows() * x.cols();
    if (n != y.rows() * y.cols())
        throw std::invalid_argument("trapz: x and y must have the same length, got " +
                                    std::to_string(n) + " and " +
                                    std::to_string(y.rows() * y.cols()));
    if (n < 2) return 0.0;
    double acc = 0.0;
    for (long i = 0; i + 1 < n; i++) {
        const double dx = double(std::real(x[int(i + 1)])) - double(std::real(x[int(i)]));
        acc += 0.5 * dx * (double(std::real(y[int(i)])) + double(std::real(y[int(i + 1)])));
    }
    return acc;
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

// scalar + A and scalar - A, so both orderings work. `3 - A` negates then
// offsets, which is what MATLAB gives and is NOT the same as A - 3.
template <typename datatype, typename scalar,
          typename = std::enable_if_t<!std::is_base_of<Matrix<datatype>,
                                                       std::decay_t<scalar>>::value>>
Matrix<datatype> operator+(const scalar k, const Matrix<datatype>& A) {
    return A + k;
}
template <typename datatype, typename scalar,
          typename = std::enable_if_t<!std::is_base_of<Matrix<datatype>,
                                                       std::decay_t<scalar>>::value>>
Matrix<datatype> operator-(const scalar k, const Matrix<datatype>& A) {
    return (-A) + k;
}

// The guard matters: without it this template also matches
// Matrix<double> * Matrix<complex<double>>, deducing scalar = Matrix<double>,
// and then fails deep inside with a conversion error instead of simply not
// being a candidate. The sibling operator+ and operator- already had it.
template <typename datatype, typename scalar,
          typename = std::enable_if_t<
              !std::is_base_of<Matrix<datatype>, std::decay_t<scalar>>::value>>
Matrix<datatype> operator*(const scalar k, Matrix<datatype> A) {
    return A * k;
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

}  // namespace mcpu
