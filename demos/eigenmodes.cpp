// ── The shapes a guitar string can vibrate in ─────────────────────────────
// Discretising -u'' = lambda u on [0,1] gives a tridiagonal matrix whose
// EIGENVECTORS are the standing waves and whose EIGENVALUES are the squared
// frequencies. The overtones come out as integer multiples of the fundamental
// without that ever being put in - it falls out of the eigenvalue problem.
#include "plotting/MatrixPlot.hpp"
#include <algorithm>
#include <cstdio>

using namespace mcpu;   // the package lives in mcpu; mgpu is its GPU twin
int main() {
    const int n = 200;
    const double h = 1.0 / (n + 1);
    Matrix<double> K(n, n);
    for (int i = 0; i < n; i++) {
        K(i, i) = 2.0 / (h * h);
        if (i > 0) K(i, i - 1) = -1.0 / (h * h);
        if (i + 1 < n) K(i, i + 1) = -1.0 / (h * h);
    }
    auto [lam, V] = K.eig();                    // symmetric: real, and eig is safe

    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lam(a, 0) < lam(b, 0); });

    printf("  mode   frequency   exact (k*pi)   relative error\n");
    for (int k = 0; k < 5; k++) {
        const double f = std::sqrt(lam(idx[k], 0)), exact = (k + 1) * mconst::pi;
        printf("  %4d   %9.4f   %12.4f   %13.2e\n", k + 1, f, exact,
               std::abs(f - exact) / exact);
    }
    Matrix<double> x = linspace(h, 1.0 - h, n);
    for (int k = 0; k < 4; k++) {
        Matrix<double> mode(n, 1);
        double s = 0.0;
        for (int i = 0; i < n; i++) s = std::max(s, std::abs(V(i, idx[k])));
        for (int i = 0; i < n; i++) mode(i, 0) = V(i, idx[k]) / s;   // scale to +-1
        if (mode(n / (2 * (k + 1)), 0) < 0) for (int i = 0; i < n; i++) mode(i, 0) *= -1;
        plt::plot(x, mode, "mode " + std::to_string(k + 1));
    }
    plt::title("standing waves of a fixed string (eigenvectors of -u'')");
    plt::xlabel("position"); plt::ylabel("amplitude");
    plt::legend(":outerright"); plt::grid(); plt::margin(6.0); plt::size(900, 500);
    plt::save("demos/out/eigenmodes.png");
    printf("  -> demos/out/eigenmodes.png\n");
}
