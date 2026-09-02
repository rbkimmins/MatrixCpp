// ── The Mandelbrot set, computed on WHOLE MATRICES ────────────────────────
// The escape-time loop is usually written per pixel. Here the entire image is
// one Matrix<complex<double>> and each iteration is a single line:
//
//     Z = Z % Z + C;        // % is element-wise multiply
//
// so the library's complex arithmetic, element-wise operators and logical
// masks do the work for every pixel at once.
#include "plotting/MatrixPlot.hpp"
#include <cstdio>
int main() {
    using cplx = std::complex<double>;
    const int H = 600, W = 800, MAXIT = 120;
    Matrix<cplx> C(H, W), Z(H, W);
    Matrix<double> iters(H, W);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            C(i, j) = cplx(-2.2 + 3.0 * j / (W - 1), -1.2 + 2.4 * i / (H - 1));

    for (int k = 0; k < MAXIT; k++) {
        Z = Z % Z + C;                       // one line, 480 000 points
        Matrix<double> r = Z.abs().real();   // magnitudes (abs keeps the type)
        // Points still bounded get another iteration credited to them. Escaped
        // points are pinned so they cannot overflow into inf/NaN.
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                if (r(i, j) <= 2.0) iters(i, j) += 1.0;
                else Z(i, j) = cplx(1e3, 0.0);   // park it well OUTSIDE the disc:
                                                 // pinning to exactly 2 leaves
                                                 // |Z| <= 2 true and it counts on
            }
    }
    printf("  %dx%d points, %d iterations\n", H, W, MAXIT);
    printf("  in the set (never escaped): %ld of %d\n",
           iters.eq(double(MAXIT)).nnz(), H * W);
    plt::heatmap(iters);
    plt::title("Mandelbrot set - escape time");
    plt::size(900, 650);
    plt::save("demos/out/mandelbrot.png");
    printf("  -> demos/out/mandelbrot.png\n");
}
