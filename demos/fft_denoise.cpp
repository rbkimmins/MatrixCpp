// ── Pulling a signal back out of noise with the FFT ───────────────────────
// Two tones buried in noise. The FFT says which frequencies are really there,
// everything below a threshold is zeroed, and the inverse transform puts the
// signal back together. The recovered curve is compared against the clean one
// it never saw.
#include "plotting/MatrixPlot.hpp"
#include <cstdio>
int main() {
    const int N = 1024;
    const double fs = 512.0;                       // samples per second
    Matrix<double> t(N, 1), clean(N, 1), noisy(N, 1), noise(N, 1);
    for (int i = 0; i < N; i++) t(i, 0) = i / fs;
    noise.set_Ran_values(-1.4, 1.4, -20240902);
    for (int i = 0; i < N; i++) {
        clean(i, 0) = std::sin(2 * mconst::pi * 12 * t(i, 0)) +
                      0.6 * std::sin(2 * mconst::pi * 47 * t(i, 0));
        noisy(i, 0) = clean(i, 0) + noise(i, 0);
    }
    Matrix<std::complex<double>> F = fft(noisy);
    Matrix<double> mag = F.abs().real();
    const double cut = 0.35 * mag.max();
    long kept = 0;
    for (int i = 0; i < N; i++)
        if (mag(i, 0) < cut) F(i, 0) = 0.0;
        else kept++;
    Matrix<double> rec = ifft(F).real();

    printf("  %d samples, %ld of %d frequency bins kept (%.1f%%)\n", N, kept, N,
           100.0 * kept / N);
    printf("  noisy   vs clean : relative error %.3f\n", (noisy - clean).norm() / clean.norm());
    printf("  denoised vs clean: relative error %.3f\n", (rec - clean).norm() / clean.norm());

    // Only the first 200 samples, so the waveform is actually visible.
    Matrix<double> ts(200, 1), c2(200, 1), n2(200, 1), r2(200, 1);
    for (int i = 0; i < 200; i++) {
        ts(i, 0) = t(i, 0); c2(i, 0) = clean(i, 0);
        n2(i, 0) = noisy(i, 0); r2(i, 0) = rec(i, 0);
    }
    plt::plot(ts, n2, "noisy");
    plt::plot(ts, c2, "clean (unseen)");
    plt::plot(ts, r2, "recovered by FFT");
    plt::title("denoising: two tones at 12 Hz and 47 Hz");
    plt::xlabel("time (s)"); plt::ylabel("amplitude");
    plt::legend(); plt::grid(); plt::margin(6.0); plt::size(950, 500);
    plt::save("demos/out/fft_denoise.png");

    plt::figure();
    Matrix<double> freq(N / 2, 1), spec(N / 2, 1);
    for (int i = 0; i < N / 2; i++) { freq(i, 0) = i * fs / N; spec(i, 0) = mag(i, 0); }
    plt::plot(freq, spec, "|FFT|");
    plt::title("spectrum: the two tones stand above the noise floor");
    plt::xlabel("frequency (Hz)"); plt::legend(); plt::grid(); plt::margin(6.0);
    plt::save("demos/out/fft_spectrum.png");
    printf("  -> demos/out/fft_{denoise,spectrum}.png\n");
}
