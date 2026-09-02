# Demos

Five small programs, each using nothing but this library and the standard
library. Run them all:

```
sh demos/run_all.sh
```

Figures land in `demos/out/`.

| demo | what it shows off |
|---|---|
| `mandelbrot.cpp` | complex matrices, element-wise operators, masks, `heatmap` |
| `svd_compress.cpp` | `svd`, low-rank approximation, `rank`, `semilogy` |
| `eigenmodes.cpp` | `eig` on a symmetric matrix, eigenvectors as physical shapes |
| `heat_expm.cpp` | `exp(A)` — the **matrix** exponential, and why it beats stepping |
| `fft_denoise.cpp` | `fft` / `ifft`, spectra, recovering a signal from noise |

## The Mandelbrot set, on whole matrices

The escape-time loop is normally written per pixel. Here the entire 600×800
image is one `Matrix<complex<double>>` and each iteration is a single line:

```cpp
Z = Z % Z + C;        // % is element-wise multiply — 480 000 points at once
```

102 406 of 480 000 points never escape.

## Truncated SVD

`A = U S V^T` orders the picture by importance, and keeping the largest k
singular values gives the best possible rank-k approximation. The demo builds a
synthetic image, reports `rank(A) = 20`, and the error confirms it exactly:

```
k=  1  relative error 0.6782
k=  3  relative error 0.1934
k=  8  relative error 0.0779
k= 20  relative error 0.0000     <- rank(A) = 20
```

Note `S` comes back as an **m×n diagonal matrix**, not a vector — so the k-th
singular value is `S(k,k)`.

## Standing waves

Discretising `-u'' = λu` gives a tridiagonal matrix whose eigenvectors *are* the
standing waves. The overtones come out as integer multiples of the fundamental
without that ever being put in:

```
mode   frequency   exact (k*pi)   relative error
   1      3.1416         3.1416        1.02e-05
   2      6.2829         6.2832        4.07e-05
   5     15.7040        15.7080        2.54e-04
```

## Heat flow, exactly

`u' = Au` has the exact solution `u(t) = exp(At)·u(0)`. That `exp` is the free
function — the **matrix** exponential — not the element-wise `A.exp()` member.
The distinction the library is built around, and the demo shows why it is worth
having: explicit stepping has a stability limit and this does not.

```
euler dt = 0.90 x limit :  peak |u| = 0.1798
euler dt = 1.60 x limit :  peak |u| = 4.902e+123   <- blown up
```

## Denoising

Two tones at 12 Hz and 47 Hz buried in noise. The FFT keeps **4 of 1024**
frequency bins, and the inverse transform rebuilds the signal:

```
noisy    vs clean : relative error 0.966
denoised vs clean: relative error 0.057
```

## One thing to watch

`linspace` returns a **1×N row**, so index it as `x(0,i)` — or build a column
directly when you need `x(i,0)`. `plot()` flattens either way, so it only
matters when you index by hand.
