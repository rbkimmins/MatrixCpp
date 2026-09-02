# The Plotting Package

Julia's [Plots.jl](https://docs.juliaplots.org) driven from C++, with no Julia
syntax in your code and no build flags beyond the include path.

```cpp
#include "plotting/MatrixPlot.hpp"

Matrix<double> x = linspace(0.0, 10.0, 200);
plt::plot(x, x.sin(), "sin");
plt::plot(x, x.cos(), "cos");
plt::title("trig");  plt::xlabel("x");  plt::legend();  plt::grid();
plt::save("trig.png");
```

```
g++ -std=c++17 -O2 -fopenmp -I. demo.cpp -o demo
```

That is the whole build. **Requires** `julia` on `PATH` with `Plots.jl`
installed; nothing else.

## The model

matplotlib's, not MATLAB's. There is a **current figure**, drawing calls add to
it, and `save()` or `show()` finishes it and starts a new one. `figure()` clears
it early. Two `plot()` calls in a row therefore share a pair of axes, which is
what makes the example above draw both curves.

Nothing runs until `save()` or `show()` — everything before that just records
what to draw. So one Julia process starts per *figure*, not per call. Expect
about a second there while Plots loads.

## What is there

| | |
|---|---|
| `plot` `scatter` `bar` `stairs` | series; `(y)` or `(x, y)`, optional label |
| `semilogx` `semilogy` `loglog` | log axes — these set the scale *and* draw |
| `heatmap` `surface` `contour` | a whole matrix |
| `spy(A, tol)` | sparsity pattern: 1 where non-zero |
| `hist(v, bins)` | histogram |
| `title` `xlabel` `ylabel` `zlabel` `legend` `grid` `size` `xlim` `ylim` | |
| `set(key, value)` | any other Plots attribute |
| `figure()` `save(path)` `show()` `script()` | |

`plot(A)` on a matrix with more than one column draws **one series per column**,
as MATLAB and Plots both do.

Values starting with `:` are passed to Plots as Symbols, so
`legend(":bottomleft")` and `set("seriescolor", ":red")` work as they would in
Julia. `save()` picks the format from the extension — `.png`, `.pdf`, `.svg`,
`.html`.

`script()` returns the Julia that would run, for when a figure comes out wrong
and the question is whether C++ or Plots is at fault.

## Why a subprocess and not an embedded runtime

Embedding was tried first, properly: `jl_init`, `jl_eval_string`, GC-rooted
argument marshalling, the lot. It crashed — reliably but unpredictably — inside
`FreeType2_jll.__init__` during `using Plots`, a segfault in `JLLWrappers`'
`unique!` over the library path list, deep in Julia's package loader and nowhere
near this code.

It was verified *not* to be the marshalling: arrays built with
`jl_alloc_array_1d` round-tripped correctly, a bare `jl_init` + module eval ran
10/10, and every combination of our header, OpenMP and `jl_get_function` passed
in isolation. What decided whether a given binary crashed was its **size** and
ASLR — the signature of a layout-sensitive fault, not a logical one. Plain
`julia` runs the identical script 100% of the time.

One process per figure costs a second of startup and buys a plotting package
that cannot corrupt the caller's heap, needs no `-I/-L/-ljulia`, and works
whether or not the Julia development headers are installed. For plotting, that
is the right trade.

`juliaflags.sh` is kept for anyone who wants to revisit embedding.
