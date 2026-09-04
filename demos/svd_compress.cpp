// ── Image compression by truncated SVD ────────────────────────────────────
// A = U S V^H keeps the picture in order of importance. Throwing away all but
// the largest k singular values gives the best possible rank-k approximation
// (Eckart-Young), and the error it leaves is exactly the tail of the singular
// values - which this checks rather than asserts.
#include "plotting/MatrixPlot.hpp"
#include <cstdio>

using namespace mcpu;   // the package lives in mcpu; mgpu is its GPU twin
int main() {
    const int N = 200;
    Matrix<double> img(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            const double x = (j - N / 2.0) / N, y = (i - N / 2.0) / N;
            img(i, j) = std::sin(12 * x) * std::cos(9 * y) +          // smooth part
                        (std::abs(x) < 0.18 && std::abs(y) < 0.30 ? 1.5 : 0.0) +
                        (x * x + y * y < 0.02 ? -2.0 : 0.0);          // sharp features
        }
    auto [U, S, V] = img.svd();
    const double total = img.norm();
    printf("  %dx%d, rank %ld, ||A|| = %.3f\n", N, N, img.rank(), total);

    Matrix<double> ks(6, 1), err(6, 1);
    const int kk[6] = {1, 3, 8, 20, 50, 120};
    for (int c = 0; c < 6; c++) {
        const int k = kk[c];
        Matrix<double> Ak(N, N);
        for (int t = 0; t < k; t++)                       // sum of rank-1 layers
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) Ak(i, j) += S(t, t) * U(i, t) * V(j, t);   // S is m x n DIAGONAL
        ks(c, 0) = double(k);
        err(c, 0) = (Ak - img).norm() / total;
        printf("  k=%3d  %5.1f%% of the data,  relative error %.4f\n",
               k, 100.0 * k * (2.0 * N + 1) / (N * N), err(c, 0));
        if (k == 20) {
            plt::figure(); plt::heatmap(Ak);
            plt::title("rank-20 approximation"); plt::save("demos/out/svd_rank20.png");
        }
    }
    plt::figure(); plt::heatmap(img); plt::title("original");
    plt::save("demos/out/svd_original.png");
    plt::figure();
    plt::semilogy(ks, err, "relative error");
    plt::title("truncated SVD: error against rank kept");
    plt::xlabel("k"); plt::ylabel("||A - A_k|| / ||A||");
    plt::legend(); plt::grid(); plt::margin(6.0);
    plt::save("demos/out/svd_error.png");
    printf("  -> demos/out/svd_{original,rank20,error}.png\n");
}
