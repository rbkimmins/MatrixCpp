#pragma once

// ==========================================================================
//  The GPU Matrix Package
// ==========================================================================
//
// One include for everything:
//
//     #include "basic/MatrixCpp.hpp"
//     #include "gpu/MatrixGpu.hpp"
//     using namespace mcpu;                        // bare Matrix<> is the CPU one
//
//     Matrix<double> A(4096, 4096);  A.set_Ran_values(0.0, 1.0);
//
//     mgpu::Matrix<double> dA = mgpu::upload(A);   // host -> device, once
//     auto dC = (dA * dA).tanh();                  // stays on the device
//     Matrix<double> C = dC.cpu();                 // device -> host, once
//
// mcpu::Matrix and mgpu::Matrix are the same name one namespace apart, which
// is the whole point of the split: `namespace np = mcpu; namespace cp = mgpu;`
// makes the numpy/cupy correspondence exact. Do NOT `using namespace` both --
// they each export Matrix, so the unqualified name becomes ambiguous.
//
// This is the GPU companion to basic/, kept in its own directory for the same
// reason MATLAB and NumPy ship GPU support as separate packages rather than
// folding it into the core: `basic/` stays header-only, dependency-free and
// compilable anywhere, and nothing in it ever includes a CUDA header.
//
// BUILDING
//
//   make -C gpu                       builds libmatrixcpp_gpu.a
//   make -C gpu test                  correctness against the CPU library
//   make -C gpu bench                 timings against basic/ and CuPy
//
// Your own code compiles with a PLAIN C++ COMPILER — no nvcc:
//
//   g++ -std=c++17 -O3 -march=native -fopenmp yours.cpp
//       -Lgpu -lmatrixcpp_gpu -L/usr/local/cuda/lib64
//       -lcudart -lcublas -lcusolver -lcurand -o yours
//
// Only gpu/src/backend.cu needs a CUDA compiler, and only when the library
// itself is built. That is why every public header here is ordinary C++17.
//
// LAYOUT
//
//   MatrixGpu.hpp      this file
//   device.hpp         which GPU, how much memory, is there one at all
//   gpu_matrix.hpp     Matrix<T>: the class
//   detail/backend.hpp declarations of everything CUDA implements
//   src/backend.cu     the only file nvcc sees
//
// WHAT IS AND IS NOT WORTH MOVING HERE
//
// Read gpu/README.md before assuming the GPU wins. On a GeForce card fp64 is
// throttled to 1/64 of fp32, so in `double` this hardware is only ~1.3-1.9x
// the tuned CPU GEMM, and a single operation that has to cross PCIe both ways
// is roughly break-even. The wins are real but specific: long chains that stay
// resident, the factorisations basic/ is slowest at, and anything in float.

#include "device.hpp"
#include "gpu_matrix.hpp"
