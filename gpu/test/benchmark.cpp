// ==========================================================================
//  GPU timings — against basic/ on the CPU, and against CuPy on the same card
// ==========================================================================
//
// METHOD, matching the rest of this project:
//
//   * min of N repeats, not mean. The minimum is the run least disturbed by
//     the scheduler, and it is the only statistic that is stable enough to
//     compare across machines under load.
//   * every GPU timing brackets a mgpu::sync(). Kernel launches are asynchronous, so
//     a timer without one measures the launch, which is ~5 us regardless of
//     how much work was queued behind it.
//   * one untimed warm-up per case. The first call to any cuBLAS or cuSOLVER
//     routine loads its module and picks a kernel, which is milliseconds of
//     one-time cost that has nothing to do with the operation.
//   * both sides built -O3 -march=native, because comparing a tuned build to
//     an untuned one measures the flags.
//
// Writes test/results/cpp.csv (run it from gpu/, which is what `make run-bench`
// does), which cupy_timings.py reads to build the three-way
// table.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "../../basic/MatrixCpp.hpp"
#include "../MatrixGpu.hpp"

using namespace mcpu;   // bare Matrix<> is the CPU one; the GPU twin is mgpu::Matrix
using Clock = std::chrono::steady_clock;

static std::vector<std::string> g_rows;

static double timeIt(int reps, const std::function<void()>& f) {
    f();   // warm-up, untimed
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        f();
        const auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// Same, but with a device sync inside the timed region so the measurement
// covers the work rather than the launch.
static double timeGpu(int reps, const std::function<void()>& f) {
    f();
    mgpu::sync();
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        f();
        mgpu::sync();
        const auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

static void report(const char* op, long n, const char* dtype, double cpuMs, double gpuMs,
                   double gflop = 0) {
    char line[256];
    if (cpuMs > 0)
        std::printf("  %-22s %5ld  %-6s  cpu %9.2f ms   gpu %8.2f ms   %6.2fx", op, n, dtype,
                    cpuMs, gpuMs, cpuMs / gpuMs);
    else
        std::printf("  %-22s %5ld  %-6s  cpu %9s      gpu %8.2f ms   %8s", op, n, dtype, "-",
                    gpuMs, "-");
    if (gflop > 0) std::printf("   %7.1f GFLOP/s", gflop / (gpuMs * 1e-3) / 1e9);
    std::printf("\n");
    std::snprintf(line, sizeof(line), "%s,%ld,%s,%.4f,%.4f", op, n, dtype, cpuMs, gpuMs);
    g_rows.emplace_back(line);
}

static Matrix<double> randMat(long r, long c) {
    Matrix<double> A(r, c);
    A.set_Ran_values(-1.0, 1.0);
    return A;
}
static Matrix<double> spd(long n) {
    Matrix<double> A = randMat(n, n);
    Matrix<double> S = A.T() * A;
    for (long i = 0; i < n; ++i) S(i, i) += double(n);
    return S;
}

int main() {
    if (!mgpu::available()) {
        std::printf("no CUDA device\n");
        return 77;
    }
    std::printf("%s\n\n", mgpu::describe().c_str());

    // ── Transfer ───────────────────────────────────────────────────────
    //
    // First, because it is the number that decides everything else: any
    // operation whose data has to cross the bus both ways is competing with
    // this, not with the CPU's arithmetic.
    std::printf("PCIe transfer\n");
    for (long n : {1024L, 4096L}) {
        Matrix<double> A = randMat(n, n);
        const double mb = double(n) * n * 8 / (1 << 20);
        const double up = timeGpu(10, [&] { volatile auto d = mgpu::upload(A); (void)d; });
        auto dA = mgpu::upload(A);
        const double down = timeGpu(10, [&] { volatile auto h = dA.cpu(); (void)h; });
        std::printf("  %-22s %5ld  %-6s  up  %7.2f ms (%6.1f GB/s)   down %7.2f ms (%6.1f GB/s)\n",
                    "host <-> device", n, "f64", up, mb / 1024 / (up * 1e-3), down,
                    mb / 1024 / (down * 1e-3));
    }

    // ── GEMM ───────────────────────────────────────────────────────────
    std::printf("\nGEMM  (2n^3 flops)\n");
    for (long n : {512L, 1024L, 2048L, 4096L}) {
        Matrix<double> A = randMat(n, n), B = randMat(n, n);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double flops = 2.0 * n * n * n;
        const int reps = n <= 1024 ? 10 : 3;
        const double cpu = timeIt(reps, [&] { volatile auto C = A * B; (void)C; });
        const double g = timeGpu(reps, [&] { auto C = dA * dB; mgpu::sync(); });
        report("gemm", n, "f64", cpu, g, flops);
    }
    // fp32, where this card is not throttled and the picture changes entirely
    for (long n : {1024L, 2048L, 4096L}) {
        Matrix<float> A(n, n), B(n, n);
        A.set_Ran_values(-1.0, 1.0);
        B.set_Ran_values(-1.0, 1.0);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double flops = 2.0 * n * n * n;
        const int reps = n <= 2048 ? 10 : 5;
        const double cpu = timeIt(n >= 4096 ? 2 : 3, [&] { volatile auto C = A * B; (void)C; });
        const double g = timeGpu(reps, [&] { auto C = dA * dB; mgpu::sync(); });
        report("gemm", n, "f32", cpu, g, flops);
    }

    // ── Element-wise ───────────────────────────────────────────────────
    //
    // Memory-bound, so the comparison is DRAM bandwidth against DRAM
    // bandwidth: ~448 GB/s on the card against ~80 GB/s for this CPU. No
    // amount of fp64 throttling touches that ratio, which is why the
    // element-wise side wins by more than GEMM does.
    std::printf("\nElement-wise (device-resident chain)\n");
    for (long n : {1024L, 4096L}) {
        Matrix<double> A = randMat(n, n), B = randMat(n, n);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double cpu = timeIt(5, [&] { volatile auto C = (A % B).exp().sqrt(); (void)C; });
        const double g = timeGpu(20, [&] { auto C = (dA % dB).exp().sqrt(); mgpu::sync(); });
        report("(A%B).exp().sqrt()", n, "f64", cpu, g);
    }

    // Fusion: the same expression as one kernel instead of three.
    std::printf("\nFused vs eager element-wise\n");
    for (long n : {1024L, 4096L}) {
        Matrix<double> A = randMat(n, n), B = randMat(n, n);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double cpu = timeIt(5, [&] { volatile auto C = (A % B).exp().sqrt(); (void)C; });
        const double eager = timeGpu(20, [&] { auto C = (dA % dB).exp().sqrt(); mgpu::sync(); });
        const double fused =
            timeGpu(20, [&] { auto C = (dA.lazy() % dB).exp().sqrt().eval(); mgpu::sync(); });
        std::printf("  %-22s %5ld  f64   cpu %8.2f   eager %7.3f   fused %7.3f   %.2fx\n",
                    "(A%B).exp().sqrt()", n, cpu, eager, fused, eager / fused);
        char nm[64];
        std::snprintf(nm, sizeof(nm), "fused_%s", "chain3");
        g_rows.push_back(std::string(nm) + "," + std::to_string(n) + ",f64," +
                         std::to_string(cpu) + "," + std::to_string(fused));
    }
    // A longer chain, where the eager path pays for six temporaries and the
    // fused one still pays for a single pass.
    for (long n : {4096L}) {
        Matrix<double> A = randMat(n, n), B = randMat(n, n), C = randMat(n, n);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B), dC = mgpu::upload(C);
        const double cpu =
            timeIt(3, [&] { volatile auto R = ((A % B) + C).exp().sqrt().tanh() % A; (void)R; });
        const double eager =
            timeGpu(10, [&] { auto R = ((dA % dB) + dC).exp().sqrt().tanh() % dA; mgpu::sync(); });
        const double fused = timeGpu(10, [&] {
            auto R = (((dA.lazy() % dB) + dC).exp().sqrt().tanh() % dA).eval();
            mgpu::sync();
        });
        std::printf("  %-22s %5ld  f64   cpu %8.2f   eager %7.3f   fused %7.3f   %.2fx\n",
                    "6-op chain, 3 inputs", n, cpu, eager, fused, eager / fused);
        g_rows.push_back("fused_chain6," + std::to_string(n) + ",f64," + std::to_string(cpu) +
                         "," + std::to_string(fused));
    }

    // Cheap ops only. The chains above use fp64 exp/sqrt, which on a card
    // that throttles fp64 to 1/64 may well be the bottleneck rather than
    // memory -- in which case removing memory traffic cannot help much. This
    // is the control: same number of steps, arithmetic that costs nothing.
    for (long n : {4096L}) {
        Matrix<double> A = randMat(n, n), B = randMat(n, n), C = randMat(n, n);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B), dC = mgpu::upload(C);
        const double eager = timeGpu(20, [&] {
            auto R = ((dA % dB) + dC) % dA - dB;
            mgpu::sync();
        });
        const double fused = timeGpu(20, [&] {
            auto R = (((dA.lazy() % dB) + dC) % dA - dB).eval();
            mgpu::sync();
        });
        // Traffic: eager writes and rereads a temporary per step (4 steps,
        // 3 distinct inputs); fused reads each input once and writes once.
        const double bytes = double(n) * n * 8;
        std::printf("  %-22s %5ld  f64   %s   eager %7.3f   fused %7.3f   %.2fx\n",
                    "4 cheap ops, 3 inputs", n, "          ", eager, fused, eager / fused);
        std::printf("       fused effective bandwidth %.0f GB/s of 448 peak (4 buffers touched)\n",
                    4 * bytes / (fused * 1e-3) / 1e9);
        g_rows.push_back("fused_cheap," + std::to_string(n) + ",f64,-1," + std::to_string(fused));
    }

    // The design argument, measured. Same arithmetic, two ownership models.
    std::printf("\nWhy results stay on the device\n");
    {
        const long n = 2048;
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double resident = timeGpu(5, [&] {
            auto t = dA;
            for (int i = 0; i < 8; ++i) t = t.tanh() + dA;
            mgpu::sync();
        });
        const double naive = timeGpu(5, [&] {
            Matrix<double> h = A;
            for (int i = 0; i < 8; ++i) h = (mgpu::upload(h).tanh() + mgpu::upload(A)).cpu();   // crosses every step
            mgpu::sync();
        });
        const double cpu = timeIt(3, [&] {
            Matrix<double> h = A;
            for (int i = 0; i < 8; ++i) h = h.tanh() + A;
        });
        std::printf("  8-step chain, n=2048, f64\n");
        std::printf("    device-resident        %8.2f ms   (%.2fx vs CPU)\n", resident,
                    cpu / resident);
        std::printf("    copying every step     %8.2f ms   (%.2fx vs CPU)  <- the trap\n", naive,
                    cpu / naive);
        std::printf("    CPU (basic/)           %8.2f ms\n", cpu);
        g_rows.push_back("chain_resident,2048,f64," + std::to_string(cpu) + "," +
                         std::to_string(resident));
        g_rows.push_back("chain_copying,2048,f64," + std::to_string(cpu) + "," +
                         std::to_string(naive));
    }

    // ── Factorisations ─────────────────────────────────────────────────
    std::printf("\nFactorisations\n");
    for (long n : {512L, 1024L, 2048L}) {
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(3, [&] { volatile auto r = A.luPacked(); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dA.lu(); mgpu::sync(); });
        report("lu", n, "f64", cpu, g);
    }
    for (long n : {512L, 1024L, 2048L}) {
        Matrix<double> A = spd(n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(3, [&] { volatile auto r = A.cholesky(); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dA.cholesky(); mgpu::sync(); });
        report("cholesky", n, "f64", cpu, g);
    }
    for (long n : {512L, 1024L, 2048L}) {
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(3, [&] { volatile auto r = A.QR(); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dA.qr(); mgpu::sync(); });
        report("qr", n, "f64", cpu, g);
    }
    for (long n : {256L, 512L, 1024L}) {
        // The CPU one-sided Jacobi runs at ~1.4 GFLOP/s, so 2048 would take
        // minutes; the GPU column is measured there separately below.
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(2, [&] { volatile auto r = A.svd(); (void)r; });
        const double g = timeGpu(3, [&] { auto r = dA.svd(); mgpu::sync(); });
        report("svd", n, "f64", cpu, g);
    }
    {
        Matrix<double> A = randMat(2048, 2048);
        auto dA = mgpu::upload(A);
        report("svd", 2048, "f64", -1, timeGpu(3, [&] { auto r = dA.svd(); mgpu::sync(); }));
    }
    for (long n : {256L, 512L, 1024L}) {
        Matrix<double> A = spd(n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(2, [&] { volatile auto r = A.eig(); (void)r; });
        const double g = timeGpu(3, [&] { auto r = dA.eigSym(); mgpu::sync(); });
        report("eigSym", n, "f64", cpu, g);
    }
    for (long n : {512L, 1024L, 2048L}) {
        Matrix<double> A = spd(n), B = randMat(n, 1);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double cpu = timeIt(3, [&] { volatile auto r = A.factorize().solve(B); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dA.solve(dB); mgpu::sync(); });
        report("solve", n, "f64", cpu, g);
    }

    // ── Tier 1 additions ───────────────────────────────────────────────

    std::printf("\nMixed-precision solve (fp32 factor + fp64 refinement)\n");
    for (long n : {1024L, 2048L, 4096L}) {
        Matrix<double> A = spd(n), B = randMat(n, 1);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double fp64 = timeGpu(5, [&] { auto r = dA.solve(dB); mgpu::sync(); });
        int iters = 0;
        const double mixed = timeGpu(5, [&] {
            auto r = dA.solveMixed(dB, mgpu::Matrix<double>::Factor::Single, &iters);
            mgpu::sync();
        });
        // The speedup is only meaningful if the answer is still fp64-accurate,
        // so the residual is reported next to it rather than trusted.
        Matrix<double> X = dA.solveMixed(dB, mgpu::Matrix<double>::Factor::Single).cpu();
        Matrix<double> R = A * X - B;
        double rn = 0, bn = 0;
        for (long i = 0; i < n; ++i) {
            rn = std::max(rn, std::fabs(R(i, 0)));
            bn = std::max(bn, std::fabs(B(i, 0)));
        }
        std::printf("  solve %5ld  f64  fp64 %8.2f ms   mixed %8.2f ms   %5.2fx   "
                    "%d refinements   resid %.1e\n",
                    n, fp64, mixed, fp64 / mixed, iters, rn / bn);
        g_rows.push_back("solve_mixed," + std::to_string(n) + ",f64," + std::to_string(fp64) +
                         "," + std::to_string(mixed));
    }

    std::printf("\nFactor once, solve many (10 right-hand sides, one at a time)\n");
    for (long n : {1024L, 2048L}) {
        Matrix<double> A = spd(n);
        std::vector<mgpu::Matrix<double>> rhs;
        for (int k = 0; k < 10; ++k) rhs.push_back(mgpu::upload(randMat(n, 1)));
        auto dA = mgpu::upload(A);
        const double refactor = timeGpu(3, [&] {
            for (auto& b : rhs) { auto x = dA.solve(b); }
            mgpu::sync();
        });
        const double once = timeGpu(3, [&] {
            auto lu = dA.factorize();
            for (auto& b : rhs) { auto x = lu.solve(b); }
            mgpu::sync();
        });
        std::printf("  %5ld  f64   solve() x10 %8.2f ms   factorize+10 %8.2f ms   %5.2fx\n", n,
                    refactor, once, refactor / once);
        g_rows.push_back("factorize10," + std::to_string(n) + ",f64," + std::to_string(refactor) +
                         "," + std::to_string(once));
    }

    std::printf("\nTransposed product: does skipping the transpose matter?\n");
    {
        // Square, where the transpose is a rounding error against the product.
        Matrix<double> A = randMat(2048, 2048), B = randMat(2048, 2048);
        auto dA = mgpu::upload(A), dB = mgpu::upload(B);
        const double mat = timeGpu(5, [&] { auto C = dA.T() * dB; mgpu::sync(); });
        const double flag = timeGpu(5, [&] { auto C = dA.tMul(dB); mgpu::sync(); });
        std::printf("  2048x2048 ^T * 2048x2048   materialise %7.2f ms   flag %7.2f ms   %.2fx\n",
                    mat, flag, mat / flag);
    }
    {
        // Tall and skinny, where the transpose is O(mn) against a product that
        // is only O(mn*k) with a tiny k -- the least-squares/covariance shape.
        Matrix<double> A = randMat(2000000, 8);
        auto dA = mgpu::upload(A);
        const double mat = timeGpu(10, [&] { auto C = dA.T() * dA; mgpu::sync(); });
        const double flag = timeGpu(10, [&] { auto C = dA.gram(); mgpu::sync(); });
        std::printf("  2000000x8 gram             materialise %7.2f ms   flag %7.2f ms   %.2fx\n",
                    mat, flag, mat / flag);
        g_rows.push_back("gram_skinny,2000000,f64," + std::to_string(mat) + "," +
                         std::to_string(flag));
    }

    std::printf("\nFFT (cuFFT)\n");
    for (long n : {4096L, 65536L, 1048576L}) {
        Matrix<double> x = randMat(1, n);
        auto dx = mgpu::upload(x);
        const double cpu = timeIt(3, [&] { volatile auto X = fft(x); (void)X; });
        const double g = timeGpu(20, [&] { auto X = mgpu::fft(dx, ROW); mgpu::sync(); });
        std::printf("  fft 1-D %8ld  f64   cpu %9.2f ms   gpu %8.3f ms   %7.2fx\n", n, cpu, g,
                    cpu / g);
        g_rows.push_back("fft1d," + std::to_string(n) + ",f64," + std::to_string(cpu) + "," +
                         std::to_string(g));
    }
    for (long n : {512L, 2048L}) {
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(2, [&] { volatile auto X = fft(A); (void)X; });
        const double g = timeGpu(10, [&] { auto X = mgpu::fft(dA, COL); mgpu::sync(); });
        std::printf("  fft batched %4ldx%-4ld f64   cpu %9.2f ms   gpu %8.3f ms   %7.2fx\n", n, n,
                    cpu, g, cpu / g);
        g_rows.push_back("fft_batch," + std::to_string(n) + ",f64," + std::to_string(cpu) + "," +
                         std::to_string(g));
    }
    {
        Matrix<double> a = randMat(1, 200000), b = randMat(1, 20000);
        auto da = mgpu::upload(a), db = mgpu::upload(b);
        const double cpu = timeIt(2, [&] { volatile auto c = conv(a, b); (void)c; });
        const double g = timeGpu(10, [&] { auto c = mgpu::conv(da, db); mgpu::sync(); });
        std::printf("  conv 200000 * 20000  f64   cpu %9.2f ms   gpu %8.3f ms   %7.2fx\n", cpu, g,
                    cpu / g);
        g_rows.push_back("conv,200000,f64," + std::to_string(cpu) + "," + std::to_string(g));
    }

    std::printf("\nComplex\n");
    {
        using Cd = std::complex<double>;
        auto randCx = [&](long r, long c) {
            Matrix<double> a = randMat(r, c), b = randMat(r, c);
            Matrix<Cd> out(r, c);
            for (long i = 0; i < r; ++i)
                for (long j = 0; j < c; ++j) out(i, j) = Cd(a(i, j), b(i, j));
            return out;
        };
        for (long n : {1024L, 2048L}) {
            Matrix<Cd> A = randCx(n, n), B = randCx(n, n);
            auto dA = mgpu::upload(A), dB = mgpu::upload(B);
            // A complex multiply is 4 real multiplies and 2 adds per term, so
            // 8n^3 flops against the real 2n^3.
            const double flops = 8.0 * n * n * n;
            const double cpu = timeIt(2, [&] { volatile auto C = A * B; (void)C; });
            const double g = timeGpu(5, [&] { auto C = dA * dB; mgpu::sync(); });
            std::printf("  zgemm %5ld       cpu %9.2f ms   gpu %8.2f ms   %6.2fx   %6.1f GFLOP/s\n",
                        n, cpu, g, cpu / g, flops / (g * 1e-3) / 1e9);
            g_rows.push_back("zgemm," + std::to_string(n) + ",c128," + std::to_string(cpu) + "," +
                             std::to_string(g));
        }
        for (long n : {512L, 1024L}) {
            Matrix<Cd> M = randCx(n, n);
            Matrix<Cd> H = M.H() * M;
            for (long i = 0; i < n; ++i) H(i, i) += Cd(double(n), 0.0);
            auto dH = mgpu::upload(H);
            const double cpu = timeIt(2, [&] { volatile auto r = H.cholesky(); (void)r; });
            const double g = timeGpu(5, [&] { auto r = dH.cholesky(); mgpu::sync(); });
            std::printf("  zpotrf %4ld      cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, cpu, g,
                        cpu / g);
        }
        for (long n : {256L, 512L}) {
            Matrix<Cd> M = randCx(n, n);
            Matrix<Cd> H = M.H() * M;
            for (long i = 0; i < n; ++i) H(i, i) += Cd(double(n), 0.0);
            auto dH = mgpu::upload(H);
            const double cpu = timeIt(2, [&] { volatile auto r = H.eig(); (void)r; });
            const double g = timeGpu(3, [&] { auto r = dH.eigSym(); mgpu::sync(); });
            std::printf("  zheevd %4ld      cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, cpu, g,
                        cpu / g);
            g_rows.push_back("zheevd," + std::to_string(n) + ",c128," + std::to_string(cpu) + "," +
                             std::to_string(g));
        }
        {
            const long n = 1048576;
            Matrix<Cd> z = randCx(1, n);
            auto dz = mgpu::upload(z);
            const double cpu = timeIt(2, [&] { volatile auto Z = fft(z, -1, ROW); (void)Z; });
            const double g = timeGpu(20, [&] { auto Z = mgpu::fft(dz, ROW); mgpu::sync(); });
            std::printf("  zfft %8ld    cpu %9.2f ms   gpu %8.3f ms   %6.2fx\n", n, cpu, g,
                        cpu / g);
            g_rows.push_back("zfft," + std::to_string(n) + ",c128," + std::to_string(cpu) + "," +
                             std::to_string(g));
        }
    }

    std::printf("\nSorting (Thrust / CUB)\n");
    for (long n : {1L << 20, 1L << 24}) {
        Matrix<double> v = randMat(1, n);
        auto dv = mgpu::upload(v);
        const double cpu = timeIt(3, [&] { volatile auto r = v.sort(ROW); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dv.sorted(); mgpu::sync(); });
        std::printf("  sort %10ld  f64   cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, cpu, g,
                    cpu / g);
        g_rows.push_back("sort," + std::to_string(n) + ",f64," + std::to_string(cpu) + "," +
                         std::to_string(g));
    }
    for (long n : {1024L, 4096L}) {
        Matrix<double> A = randMat(n, n);
        auto dA = mgpu::upload(A);
        const double cpu = timeIt(2, [&] { volatile auto r = A.sort(ROW); (void)r; });
        const double g = timeGpu(5, [&] { auto r = dA.sort(ROW); mgpu::sync(); });
        std::printf("  sort rows of %4ldx%-4ld  cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, n,
                    cpu, g, cpu / g);
        const double cpuc = timeIt(2, [&] { volatile auto r = A.sort(COL); (void)r; });
        const double gc = timeGpu(5, [&] { auto r = dA.sort(COL); mgpu::sync(); });
        std::printf("  sort cols of %4ldx%-4ld  cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, n,
                    cpuc, gc, cpuc / gc);
        g_rows.push_back("sort_rows," + std::to_string(n) + ",f64," + std::to_string(cpu) + "," +
                         std::to_string(g));
    }
    {
        const long n = 1L << 22;
        Matrix<double> v = randMat(1, n);
        auto dv = mgpu::upload(v);
        const double cpu = timeIt(2, [&] { volatile auto r = v.median(ROW); (void)r; });
        const double g = timeGpu(5, [&] { volatile double r = dv.median(); (void)r; mgpu::sync(); });
        std::printf("  median %8ld f64   cpu %9.2f ms   gpu %8.2f ms   %6.2fx\n", n, cpu, g,
                    cpu / g);
    }

    {
        std::ofstream f("test/results/cpp.csv");
        f << "op,n,dtype,cpu_ms,gpu_ms\n";
        for (const auto& r : g_rows) f << r << "\n";
    }
    std::printf("\nwrote test/results/cpp.csv\n");
    return 0;
}
