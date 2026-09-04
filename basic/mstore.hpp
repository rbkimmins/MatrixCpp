#pragma once

// ==========================================================================
//  Raw storage and the multiply kernel
// ==========================================================================
//
// Allocation that does not run constructors, transparent huge pages, the
// parallel-work thresholds, and the blocked GEMM every level-3 path calls.
// Nothing here knows what a Matrix is — it is the layer underneath.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

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

#if defined(__GNUC__) || defined(__clang__)
    #define MATRIXCPP_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define MATRIXCPP_RESTRICT __restrict
#else
    #define MATRIXCPP_RESTRICT
#endif

namespace mcpu {


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

    // RAII around rawAlloc, for a scratch buffer the caller is about to
    // overwrite completely. std::vector value-initialises, and zeroing a buffer
    // that is then written in full is pure waste — measured at tens of
    // milliseconds inside a single LU.
    template <class T>
    class RawBuf {
      public:
        explicit RawBuf(long n) : p_(rawAlloc<T>(n)) {}
        ~RawBuf() { rawFree(p_); }
        RawBuf(const RawBuf&) = delete;
        RawBuf& operator=(const RawBuf&) = delete;
        T* get() const { return p_; }

      private:
        T* p_;
    };

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
    // MEASURED, and the BUILD FLAGS are half the number — quiet 32-core Zen 4,
    // square double product at n=1024:
    //         -O2                 5.2 GFLOP/s (1 thread)    71 GFLOP/s (32)
    //         -O3 -march=native  13.5 GFLOP/s (1 thread)   194 GFLOP/s (32)
    // So -march=native alone is worth 2.7x, for free, and any figure quoted
    // without its flags is meaningless. (An earlier edit here deleted a "175
    // GFLOP/s" note as unreproducible; it reproduces perfectly well at
    // -O3 -march=native, and the deletion was made off an -O2 measurement.)
    //
    // A packed, register-blocked AVX-512 microkernel — MR x NR tile of C held
    // in zmm registers across the whole k loop, so C is loaded and stored once
    // instead of once per k — was prototyped and measured against this:
    //         1 thread    13.5 ->  75.0 GFLOP/s   5.6x
    //         32 threads   194 ->   438 GFLOP/s   2.2x
    // The threaded gain is the smaller one because 32 cores reach the memory
    // wall first. That 2.2x is the honest ceiling for this kernel, and it is
    // the case FOR a rewrite — but see the QR notes: it is NOT enough to make
    // blocked QR win, which an earlier version of that note claimed.
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
        // ── The unit of parallel work is a TILE OF C, indexed over BOTH M and N ──
        // Parallelising over M alone meant a product with M < BLOCK had exactly
        // one tile and ran SINGLE-THREADED. That is not a corner case: it is the
        // shape every panel algorithm produces. Measured on 32 cores, M=48,
        // N=K=1024 ran at 6.5 GFLOP/s against 8.0 on ONE thread — the OpenMP
        // entry cost more than the parallelism it failed to provide, and it is
        // the single reason the blocked QR lost to a level-2 update.
        //
        // Tiles are disjoint in C, so both dimensions parallelise safely. The kk
        // loop stays INSIDE the tile, which is also better for locality than the
        // old order: one 64x64 tile of C is 32 KB and now stays hot across the
        // whole K sweep, where before each C row-block was re-read K/BLOCK times.
        const long nTiles = (N + BLOCK - 1) / BLOCK;
        // When N is SMALLER than a block there is only one column of tiles, and
        // all the parallelism has to come from M — 16 tasks for 32 threads at
        // M=1024, N=48, which measured 27 GFLOP/s against 94 for the same
        // arithmetic laid out the other way. Shrinking the M block until there
        // is work for everyone costs some reuse of A and buys far more.
        long mB = BLOCK;
#if defined(_OPENMP)
        {
            const long want = 2L * omp_get_max_threads();
            while (mB > 8 && ((M + mB - 1) / mB) * nTiles < want) mB /= 2;
        }
#endif
        const long mTiles = (M + mB - 1) / mB;
        const long tiles = mTiles * nTiles;
        // Pointers hoisted and captured BY VALUE — by reference cost about half
        // the throughput at n=2048.
        auto tile = [=](long t) {
            const long ii = (t / nTiles) * mB;
            const long jj = (t % nTiles) * BLOCK;
            const long iEnd = std::min(ii + mB, M);
            const long jEnd = std::min(jj + BLOCK, N);
            for (long kk = 0; kk < K; kk += BLOCK) {
                const long kEnd = std::min(kk + BLOCK, K);
                for (long i = ii; i < iEnd; i++) {
                    T* MATRIXCPP_RESTRICT crow = Cg + i * N;
                    for (long k = kk; k < kEnd; k++) {
                        const T aik = Ag[i * K + k];
                        const T* MATRIXCPP_RESTRICT brow = Bg + k * N;
                        for (long j = jj; j < jEnd; j++)
                            crow[j] += aik * brow[j];
                    }
                }
            }
        };
        // Small products never touch the OpenMP runtime. An `if` clause on the
        // pragma is not enough — the runtime is still entered for ~360 ns even when
        // the condition is false, and a 2x2 multiply went 29 ns -> 3290 ns because
        // of it. The threshold is on M*N*K, not on n: a tall thin product and a
        // square one of the same n are different jobs.
        if (M * N * K <= PARALLEL_MIN_WORK) {
            for (long t = 0; t < tiles; t++)
                tile(t);
            return;
        }
        // The CHUNK is sized from the work in a tile, not left at 1. A tile is
        // BLOCK*BLOCK*K multiply-adds, so a short K makes it tiny and the
        // per-grab cost of a dynamic schedule dominates: at K=48 a chunk of 1
        // measured 26.9 GFLOP/s where a coarser grain reached 68.9, while at
        // K=1024 the coarse grain loses to load imbalance. Scaling the chunk so
        // each grab is a comparable amount of arithmetic gets both.
#ifdef _OPENMP
        const long perTile = mB * BLOCK * std::max<long>(K, 1);
        long chunk = (long)(1 << 21) / perTile;
        const long cap = std::max<long>(1, tiles / (4L * omp_get_max_threads()));
        if (chunk < 1) chunk = 1;
        if (chunk > cap) chunk = cap;
    #pragma omp parallel for schedule(dynamic, chunk)
#endif
        for (long t = 0; t < tiles; t++)
            tile(t);
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

}  // namespace mcpu
