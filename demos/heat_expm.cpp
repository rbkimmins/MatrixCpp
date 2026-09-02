// ── Heat flow, solved EXACTLY with a matrix exponential ───────────────────
// Discretising u_t = u_xx gives u' = A u, whose exact solution is
// u(t) = exp(A t) u(0). exp() here is the MATRIX exponential (the free
// function), not the element-wise member - the distinction the whole library
// is built around.
//
// The point of the demo: explicit stepping has a stability limit and blows up
// past it. The matrix exponential has none, and takes one step of any size.
#include "plotting/MatrixPlot.hpp"
#include <cstdio>
int main() {
    const int n = 120;
    const double h = 1.0 / (n + 1), T = 0.02;
    Matrix<double> A(n, n);
    for (int i = 0; i < n; i++) {
        A(i, i) = -2.0 / (h * h);
        if (i > 0) A(i, i - 1) = 1.0 / (h * h);
        if (i + 1 < n) A(i, i + 1) = 1.0 / (h * h);
    }
    Matrix<double> u0(n, 1);                       // a hot spike in the middle
    for (int i = 0; i < n; i++) u0(i, 0) = (std::abs(i - n / 2) < 6) ? 1.0 : 0.0;

    Matrix<double> exact = exp(A * T) * u0;        // MATRIX exponential: one step
    printf("  exact:  one exp(A*T), any T, unconditionally stable\n");

    Matrix<double> x = linspace(h, 1.0 - h, n);
    plt::plot(x, u0, "t = 0");
    plt::plot(x, exact, "t = T  (exp(A*T))");

    const double dtStable = 0.5 * h * h;           // the explicit stability limit
    for (double mult : {0.9, 1.6}) {
        const double dt = mult * dtStable;
        Matrix<double> u = u0;
        for (int s = 0; s < int(T / dt); s++) u = u + (A * u) * dt;
        const double peak = u.abs().max();
        printf("  euler dt = %.2f x limit :  peak |u| = %-12.4g %s\n", mult, peak,
               peak > 10.0 ? "<- blown up" : "");
        if (peak < 10.0) plt::plot(x, u, "euler (stable dt)");
    }
    plt::title("heat equation: exp(A*t) versus explicit stepping");
    plt::xlabel("position"); plt::ylabel("temperature");
    plt::legend(); plt::grid(); plt::margin(6.0);
    plt::save("demos/out/heat.png");
    printf("  -> demos/out/heat.png\n");
}
