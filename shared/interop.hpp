#pragma once

// ==========================================================================
//  Shared vocabulary and CPU <-> GPU interop
// ==========================================================================
//
//     #include "shared/interop.hpp"
//
//     mcpu::Matrix<double> A(4096, 4096);
//     auto dA = mx::to_gpu(A);          // host -> device
//     auto B  = mx::to_cpu(dA * dA);    // device -> host
//
// WHAT THIS IS FOR, and what it deliberately is not.
//
// The two packages are meant to read as one library with two address spaces:
// mcpu::Matrix and mgpu::Matrix, the same name a namespace apart. That only
// works if the words they share mean the same thing in both, and if crossing
// between them is one obvious call rather than a different spelling on each
// side.
//
// It is NOT an attempt to factor the algorithms. Those are exactly what
// differs -- a blocked LU written for 32 MB of L3 and a cuSOLVER call have
// nothing in common but their name -- and pretending otherwise would produce
// an abstraction that fits neither.
//
// WHY THE ENUMS ARE ALIASED RATHER THAN MOVED. NormType, all_t, ROW/COL and
// the QR modes already live in basic/traits.hpp and gpu/ already uses those
// same definitions, so there is only ever one of each. Relocating them into
// this header would churn every include in basic/ to buy nothing; aliasing
// gives generic code a neutral spelling at no risk. is_complex and real_t are
// the one genuine duplication, and mgpu's are aliased onto mcpu's below.

#include "../basic/matrix.hpp"
#include "../gpu/MatrixGpu.hpp"

namespace mx {

    // ── The shared vocabulary ──────────────────────────────────────────
    //
    // Neutral names for the things that mean the same on both sides, so code
    // that is generic over where the data lives does not have to pick a
    // namespace arbitrarily.

    using mcpu::all;
    using mcpu::all_t;
    using mcpu::COL;
    using mcpu::NormType;
    using mcpu::ROW;

    template <class T>
    using is_complex = mcpu::is_complex<T>;
    template <class T>
    inline constexpr bool is_complex_v = mcpu::is_complex<T>::value;
    // The real type behind a possibly-complex one: real_t<complex<double>> is
    // double, real_t<double> is double.
    template <class T>
    using real_t = mgpu::real_t<T>;

    // ── Where a matrix lives ───────────────────────────────────────────

    template <class T>
    using Host = mcpu::Matrix<T>;
    template <class T>
    using Device = mgpu::Matrix<T>;

    // ── Crossing ───────────────────────────────────────────────────────
    //
    // Both directions spelled the same way, as verbs, because a transfer is
    // the one operation in this library that is worth seeing at the call site:
    // on this hardware PCIe moves 9-22 GB/s against a card that does 340
    // GFLOP/s in double, so a crossing inside a loop is usually the whole cost.
    //
    // These are the same operations as mgpu::upload() and .cpu(); they exist so
    // that neither direction has to be spelled from inside one of the two
    // namespaces.

    template <class T>
    mgpu::Matrix<T> to_gpu(const mcpu::Matrix<T>& host) {
        return mgpu::Matrix<T>(host);
    }
    template <class T>
    mcpu::Matrix<T> to_cpu(const mgpu::Matrix<T>& device) {
        return device.cpu();
    }
    // An already-resident matrix passed through, so a template that may be
    // handed either does not need to branch.
    template <class T>
    const mgpu::Matrix<T>& to_gpu(const mgpu::Matrix<T>& device) {
        return device;
    }
    template <class T>
    const mcpu::Matrix<T>& to_cpu(const mcpu::Matrix<T>& host) {
        return host;
    }

    // Round trip, mostly for tests: whatever comes back must equal what went in.
    template <class T>
    mcpu::Matrix<T> roundtrip(const mcpu::Matrix<T>& host) {
        return to_cpu(to_gpu(host));
    }

    // ── Running the same code on either side ───────────────────────────
    //
    // The subset of the interface that both classes implement with the same
    // meaning, so a function template can be written once against it:
    //
    //     template <class M> auto residual(const M& A, const M& x, const M& b) {
    //         return (A * x - b).norm();      // works for either Matrix
    //     }
    //     double rc = residual(A, x, b);              // on the CPU
    //     double rg = residual(dA, dx, db);           // on the GPU
    //
    // What is common today: construction from (rows, cols), rows()/cols()/
    // size()/empty(), operator() indexing on the host side only, + - % and
    // scalar * /, matrix *, T(), H(), the element-wise maths, sum/prod/mean/
    // norm, lu/cholesky/qr/svd/solve/inv/det, and diag/triu/tril/block.
    //
    // gpu/README.md lists what the GPU side still lacks. The short version is
    // that anything needing a sort (median, sortrows, unique) or a
    // non-symmetric eigensolver (eig, schur, hess, funm) is CPU-only.

}  // namespace mx
