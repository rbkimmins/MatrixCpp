#pragma once
// Compatibility shim. The library now lives in basic/ as the Basic Matrix
// Package, split by theme — see basic/MatrixCpp.hpp. Existing code that
// includes "Matrix1.0.hpp" keeps working unchanged.
#include "basic/MatrixCpp.hpp"

// The package now lives in namespace mcpu, so that mcpu::Matrix and
// mgpu::Matrix read as the one-to-one pair they are — the C++ spelling of
// numpy/cupy. This shim hoists it back to global scope so code written before
// the move keeps compiling with no edit at all.
//
// New code should prefer `using namespace mcpu;` (or `namespace np = mcpu;`)
// over this header, and MUST do so if it also uses mgpu — a global Matrix and
// an mgpu::Matrix in the same translation unit is the one combination that
// makes the unqualified name ambiguous.
using namespace mcpu;
