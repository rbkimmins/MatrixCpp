# Benchmarks

How fast each operation is, and how that compares to NumPy.

```
g++ -std=c++17 -O3 -march=native -fopenmp -I. benchmarks/running_time.cpp -o running_time
./running_time
```

That times 57 operations across both `Matrix<double>` and
`Matrix<complex<double>>` and **draws the results itself**, into
`benchmarks/plots/speed_<dtype>_<group>.png` — one figure per family, every
operation in it overlaid on log-log axes. About 45 s for the timing plus a
second per figure.

No Julia scripts, no matplotlib, no intermediate file. The plots come out of
the C++ program that took the measurements, through `plotting/`.

## Comparing against NumPy

This is the one place data is written to disk, and only because the two sides
are measured by two different languages and have to meet somewhere:

```
./running_time                          # also writes bench/cpp_<dtype>_<op>.csv
python3 benchmarks/numpy_timings.py     # writes bench/numpy_*.csv + a table
g++ -std=c++17 -O2 -fopenmp -I. benchmarks/plot_comparison.cpp -o plot_comparison
./plot_comparison                       # C++ draws the comparison
```

`numpy_timings.py` does not plot. It times NumPy and prints a table; the
figures are `benchmarks/plots/speedup_<dtype>_<group>.png`, drawn by
`plot_comparison.cpp`.

The size points are not chosen twice: the Python side reads them back out of the
C++ CSVs, so both implementations are measured at exactly the same n. No
interpolation, no mismatched ranges.

**What the comparison is worth knowing about:** NumPy on this machine links the
*reference* BLAS (`libblas.so.3`), not OpenBLAS or MKL — its matmul runs at
~4.8 GFLOP/s single-threaded against ~200 for ours. Speedups against it flatter
us considerably. See the note in the root README.

## Why speedup rather than absolute times

`plot_comparison.cpp` plots the ratio, with a parity line at 1. Fifty operations'
absolute times on one pair of axes is unreadable, and the absolute curves are
already what `running_time.cpp` draws. Above the line we are faster.

## Grouping

Operations are sorted into `elementwise`, `reductions`, `factorisations`,
`matrix_functions` and `structure` by name, in `groupOf()` — which both programs
define identically so their figures line up. A new operation lands in the right
figure without touching either.

## `-ffast-math` is deliberately absent

It implies `-fcx-limited-range`, which changes how complex multiply and divide
are evaluated. Complex timings taken with it are not comparable to NumPy's.
