// Cross-validates Tensor.hpp against NumPy.
//
// NumPy IS the reference implementation of the flat-buffer-plus-strides layout
// Tensor uses, so "does permute/reshape/tensordot mean the same thing here as
// there" is exactly the question worth asking. This dumps every case; the
// companion tensor_numpy_validate.py recomputes each one in NumPy and compares.
//
//   g++ -std=c++17 -O2 -fopenmp -o tensor_numpy_validate tensor_numpy_validate.cpp
//   ./tensor_numpy_validate > tensor_cases.txt && python3 tensor_numpy_validate.py
//
// Format, two lines per case:
//   <name> <rank> <dim0> <dim1> ...
//   <values, row-major, 17 significant digits>
#include "Tensor.hpp"
#include <cstdio>

static void dump(const char* name, const Tensor<double>& t) {
    Tensor<double> c = t.contiguous();
    printf("%s %ld", name, c.rank());
    for (long k = 0; k < c.rank(); k++) printf(" %ld", c.shape(k));
    printf("\n");
    Tensor<double> f = c.reshape(-1);
    for (long i = 0; i < c.size(); i++)
        printf("%.17g%c", f(i), (i + 1 == c.size()) ? '\n' : ' ');
}

int main() {
    Tensor<double> A(2, 3, 4), B(4, 5), C(3, 4, 5);
    // Deterministic, and spread over a range where cancellation would show up.
    long n = 0;
    { auto f = A.reshape(-1); for (long i = 0; i < A.size(); i++) f(i) = std::sin(double(n++)); }
    n = 0;
    { auto f = B.reshape(-1); for (long i = 0; i < B.size(); i++) f(i) = std::cos(double(n++)); }
    n = 0;
    { auto f = C.reshape(-1); for (long i = 0; i < C.size(); i++) f(i) = std::sin(double(n++) * 0.5); }

    dump("A", A); dump("B", B); dump("C", C);

    // Views — the metadata-only operations, which is where a stride bug hides.
    dump("transposeT",   A.T());
    dump("permute201",   A.permute({2, 0, 1}));
    dump("swap02",       A.swapAxes(0, 2));
    dump("slice_ax1_i2", A.slice(1, 2));
    dump("reshape_6_4",  A.reshape(6, 4));

    // Reductions along every axis.
    dump("sum_ax0", A.sum(0));
    dump("sum_ax1", A.sum(1));
    dump("sum_ax2", A.sum(2));

    // Contractions: one axis, two axes, and explicitly named axes.
    dump("tensordot1", contract(A, B, 1));
    dump("tensordot2", contract(A, C, 2));
    dump("named_2_0",  contract(A, B, {2}, {0}));

    // Element-wise, including through a strided view.
    dump("elem_add", A + A);
    dump("elem_mul", A % A);
    dump("perm_add", A.permute({2, 1, 0}) + A.permute({2, 1, 0}));
    return 0;
}
