#pragma once

// ==========================================================================
//  The Basic Matrix Package
// ==========================================================================
//
// One include for everything:
//
//     #include "basic/MatrixCpp.hpp"
//
//     Matrix<double> A(3, 3);
//     Tensor<double> T({2, 3, 4});
//
// The package is split by theme, and each header stands on its own if that is
// all you need — they chain their own dependencies, so including any one of
// them pulls exactly what it requires and nothing else:
//
//     mstore.hpp            raw storage, huge pages, the GEMM kernel
//     constants.hpp         mconst::pi and the _i / _deg literals
//     traits.hpp            is_complex / work_t, ROW and COL, the enums
//     io.hpp                output formats: Pretty, CSV, JSON, MATLAB, ...
//     matrix.hpp            the Matrix class and its factorisations
//     matrixfunctions.hpp   exp(A), log(A), sqrt(A) — matrix, not element-wise
//     decomposition.hpp     factorize(): factor once, solve many times
//     builders.hpp          linspace, magic, vander, polynomials
//     eigen.hpp             funm, eig(A,B), and QZ
//     signal.hpp            fft, conv, filter, interp1, gradient
//     identity.hpp          the global `I`
//     tensor.hpp            the Tensor class
//     random.hpp            the ran2 generator both classes use
//
// The reasoning behind the code — every design decision, every measurement,
// and the negative results worth not repeating — is in docs/DESIGN_NOTES.md
// rather than in these headers, which is most of why they are readable.
//
// BUILD WITH -O3 -march=native. It is worth 2.7x on the multiply kernel alone,
// and every GFLOP/s figure in the notes assumes it.

#include "identity.hpp"   // chains the whole matrix side
#include "tensor.hpp"
