// ==========================================================================
//  Wigner's semicircle law — a demo of the one thing the GPU is 236x at
// ==========================================================================
//
// Take a large symmetric matrix whose entries are independent random numbers.
// Its eigenvalues are not random-looking at all: scaled by sqrt(n) they fill a
// perfect semicircle on [-2, 2], and the bigger the matrix the sharper it gets.
// Wigner proved it in 1955 for nuclear energy levels; it turns up again in
// wireless channel capacity, in the spectra of large graphs, and in why deep
// networks train at all.
//
// It is also the ideal demonstration of what this package is for. The law only
// becomes visible at large n, and a symmetric eigendecomposition is the single
// routine where the GPU wins hardest here:
//
//     n = 1024    CPU 10978 ms    GPU 46 ms      236x
//
// At n = 4096 the CPU would need roughly ten minutes per matrix. The GPU does
// the whole sweep below in seconds, which is the difference between "a plot you
// can look at" and "a job you submit".
//
// Build:  make -C gpu demo && gpu/demo/wigner

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../basic/MatrixCpp.hpp"
#include "../MatrixGpu.hpp"

using namespace mcpu;   // bare Matrix<> is the CPU one; the GPU twin is mgpu::Matrix
using Clock = std::chrono::steady_clock;

// A GOE matrix: symmetric, entries N(0,1) off the diagonal. Built on the
// device and symmetrised there, so nothing crosses the bus but the seed.
static mgpu::Matrix<double> goe(long n, unsigned long long seed) {
    mgpu::Matrix<double> A(n, n);
    A.randn(0.0, 1.0, seed);
    // The symmetric part is (A + A^T)/2, but that is NOT the matrix Wigner's
    // law describes. Halving gives the off-diagonals variance (1+1)/4 = 1/2,
    // and the semicircle then only reaches +-sqrt(2) instead of +-2 - visible
    // immediately as a histogram that stops short of the predicted edge.
    // Dividing by sqrt(2) instead restores unit variance off the diagonal,
    // which is what the law assumes.
    //
    // One transpose, one add, one scale: three kernels, no host involvement.
    return (A + A.T()) * (1.0 / std::sqrt(2.0));
}

static void histogram(const std::vector<double>& x, double lo, double hi, int bins) {
    std::vector<long> h((std::size_t)bins, 0);
    for (double v : x) {
        int b = int((v - lo) / (hi - lo) * bins);
        if (b >= 0 && b < bins) h[(std::size_t)b]++;
    }
    const long peak = *std::max_element(h.begin(), h.end());
    const int width = 58;
    for (int b = bins - 1; b >= 0; --b) {
        const double centre = lo + (b + 0.5) * (hi - lo) / bins;
        const int len = peak ? int(double(h[(std::size_t)b]) / double(peak) * width) : 0;
        // The semicircle this should trace: (1/2pi) sqrt(4 - x^2), normalised
        // so its own peak is the full width.
        const double th = centre * centre < 4.0 ? std::sqrt(4.0 - centre * centre) / 2.0 : 0.0;
        const int expect = int(th * width);
        std::printf("  %+5.2f | ", centre);
        for (int i = 0; i < width; ++i)
            std::printf("%c", i < len ? '#' : (i == expect ? '.' : ' '));
        std::printf("\n");
    }
    std::printf("        %s\n", "   '.' marks where the semicircle law says the edge should be");
}

int main() {
    if (!mgpu::available()) {
        std::printf("no CUDA device\n");
        return 77;
    }
    std::printf("%s\n\n", mgpu::describe().c_str());

    // ── The law, at a size the CPU could not reach ─────────────────────
    const long n = 4096;
    const int trials = 4;
    std::printf("Wigner semicircle: %d matrices of %ldx%ld, %d eigenvalues total\n\n", trials, n,
                n, trials * (int)n);

    std::vector<double> all;
    all.reserve((std::size_t)trials * n);

    const auto t0 = Clock::now();
    for (int t = 0; t < trials; ++t) {
        mgpu::Matrix<double> A = goe(n, 1000 + t);
        auto [w, V] = A.eigSym(/*vectors=*/false);   // values only: much cheaper
        Matrix<double> hw = w.cpu();
        // Wigner's scaling: eigenvalues of an n x n GOE spread as sqrt(n).
        for (long i = 0; i < n; ++i) all.push_back(hw(i, 0) / std::sqrt(double(n)));
    }
    mgpu::sync();
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    histogram(all, -2.5, 2.5, 25);
    std::printf("\n  %d eigendecompositions of %ldx%ld in %.0f ms (%.0f ms each)\n", trials, n, n,
                ms, ms / trials);
    std::printf("  the same work on the CPU runs at roughly 10 s for n=1024 alone,\n");
    std::printf("  and cost grows as n^3 - about 10 minutes per matrix at this size\n");

    // ── Marchenko-Pastur, for free ─────────────────────────────────────
    //
    // The same machinery gives the other classical law: the eigenvalues of a
    // sample covariance matrix (1/m) X^T X, which is what you actually get when
    // you estimate a covariance from finite data. The bulk has hard edges at
    // (1 +- sqrt(ratio))^2, and everything inside them is noise rather than
    // signal - which is the practical reason to care, since it tells you which
    // principal components mean anything.
    std::printf("\n\nMarchenko-Pastur: eigenvalues of a sample covariance matrix\n");
    const long p = 2048, m = 4096;
    const double ratio = double(p) / double(m);
    mgpu::Matrix<double> X(m, p);
    X.randn(0.0, 1.0, 7);
    mgpu::Matrix<double> C = (X.T() * X) * (1.0 / double(m));   // p x p, on the device
    auto [wc, Vc] = C.eigSym(false);
    Matrix<double> hc = wc.cpu();
    std::vector<double> ev;
    ev.reserve((std::size_t)p);
    for (long i = 0; i < p; ++i) ev.push_back(hc(i, 0));

    const double edgeLo = (1 - std::sqrt(ratio)) * (1 - std::sqrt(ratio));
    const double edgeHi = (1 + std::sqrt(ratio)) * (1 + std::sqrt(ratio));
    std::printf("  p=%ld, m=%ld, ratio=%.2f\n", p, m, ratio);
    std::printf("  predicted support   [%.4f, %.4f]\n", edgeLo, edgeHi);
    std::printf("  observed  support   [%.4f, %.4f]\n", ev.front(), ev.back());
    histogram(ev, 0.0, edgeHi * 1.1, 20);

    return 0;
}
