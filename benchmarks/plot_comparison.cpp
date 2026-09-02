// ══════════════════════════════════════════════════════════════════════════
//  MatrixCpp vs NumPy — drawn in C++
//
//  Reads bench/cpp_<dtype>_<op>.csv and bench/numpy_<dtype>_<op>.csv and plots
//  the comparison with the plotting package. This is the ONLY part of the
//  benchmark that reads data back off disk, and it is only here because the two
//  sides are measured by two different languages and have to meet somewhere.
//
//  What it draws, per (dtype, group):
//    speedup_<dtype>_<group>.png   NumPy time / MatrixCpp time against n, with
//                                  a parity line at 1. Above the line we are
//                                  faster. A ratio is the right thing to plot
//                                  here: absolute times for fifty operations on
//                                  one pair of axes is unreadable, and the
//                                  absolute curves are already drawn by
//                                  running_time.cpp.
//
//  Build and run from the REPO ROOT, after ./running_time and numpy_timings.py:
//      g++ -std=c++17 -O2 -fopenmp -I. benchmarks/plot_comparison.cpp -o plot_comparison
//      ./plot_comparison
// ══════════════════════════════════════════════════════════════════════════
#include "plotting/MatrixPlot.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Curve {
    std::vector<double> n, t;
};

// size,time_seconds — one header line, then pairs.
static Curve readCsv(const std::string& path) {
    Curve c;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line)) {
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        try {
            c.n.push_back(std::stod(line.substr(0, comma)));
            c.t.push_back(std::stod(line.substr(comma + 1)));
        } catch (const std::exception&) {
        }
    }
    return c;
}

// Same rule running_time.cpp uses, so the two sets of figures line up.
static std::string groupOf(const std::string& op) {
    auto has = [&](const char* s) { return op.find(s) != std::string::npos; };
    if (has("lu") || has("qr") || has("svd") || has("eig") || has("det") ||
        has("inverse") || has("cholesky") || has("schur") || has("solve") ||
        has("rank") || has("pinv"))
        return "factorisations";
    if (has("mat_pow") || has("expm") || has("logm") || has("sqrtm") || has("funm"))
        return "matrix_functions";
    if (has("sum") || has("trace") || has("norm") || has("mean") || has("min") ||
        has("max") || has("prod"))
        return "reductions";
    if (has("transpose") || has("concat") || has("isdiagonal") || has("conj") ||
        has("diag") || has("reshape"))
        return "structure";
    return "elementwise";
}

int main() {
    const std::string dir = "bench";
    if (!fs::exists(dir)) {
        std::cerr << "no bench/ — run ./running_time and "
                     "python3 benchmarks/numpy_timings.py first\n";
        return 1;
    }
    // (dtype, group) -> list of (op, speedup curve)
    std::map<std::string, std::vector<std::pair<std::string, Curve>>> figures;
    int paired = 0, unpaired = 0;

    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("cpp_", 0) != 0 || name.size() < 9) continue;
        const std::string stem = name.substr(4, name.size() - 8);   // <dtype>_<op>
        const std::size_t us = stem.find('_');
        if (us == std::string::npos) continue;
        const std::string dtype = stem.substr(0, us), op = stem.substr(us + 1);
        const std::string npath = dir + "/numpy_" + stem + ".csv";
        if (!fs::exists(npath)) { unpaired++; continue; }

        const Curve c = readCsv(e.path().string());
        const Curve np = readCsv(npath);
        Curve ratio;
        const std::size_t m = std::min(c.n.size(), np.n.size());
        for (std::size_t i = 0; i < m; i++) {
            if (c.t[i] <= 0.0) continue;
            ratio.n.push_back(c.n[i]);
            ratio.t.push_back(np.t[i] / c.t[i]);
        }
        if (ratio.n.empty()) continue;
        figures[dtype + "|" + groupOf(op)].push_back({op, ratio});
        paired++;
    }

    if (figures.empty()) {
        std::cerr << "no cpp_/numpy_ pairs found in bench/\n";
        return 1;
    }
    std::cout << paired << " operations paired";
    if (unpaired) std::cout << ", " << unpaired << " without a NumPy counterpart";
    std::cout << "\nplotting -> benchmarks/plots/\n";
    if (std::system("mkdir -p benchmarks/plots") != 0) return 1;

    for (const auto& kv : figures) {
        const std::size_t bar = kv.first.find('|');
        const std::string dtype = kv.first.substr(0, bar);
        const std::string group = kv.first.substr(bar + 1);

        plt::figure();
        double lo = 1e300, hi = -1e300;
        for (const auto& oc : kv.second) {
            plt::plot(oc.second.n, oc.second.t, oc.first);
            for (double v : oc.second.t) { lo = std::min(lo, v); hi = std::max(hi, v); }
        }
        // Parity, drawn across the measured range so it is unmistakable which
        // side of it each curve is on.
        double nlo = 1e300, nhi = -1e300;
        for (const auto& oc : kv.second)
            for (double v : oc.second.n) { nlo = std::min(nlo, v); nhi = std::max(nhi, v); }
        plt::plot(std::vector<double>{nlo, nhi}, std::vector<double>{1.0, 1.0}, "parity");

        plt::set("xscale", ":log10");
        plt::set("yscale", ":log10");
        plt::title(dtype + " — " + group + ": NumPy time / MatrixCpp time");
        plt::xlabel("n");
        plt::ylabel("speedup  (>1 = ours faster)");
        plt::margin(6.0);
        plt::legend(":outerright");
        plt::size(1000, 600);
        const std::string out =
            "benchmarks/plots/speedup_" + dtype + "_" + group + ".png";
        plt::save(out);
        std::cout << "  " << out << "  (" << kv.second.size() << " ops)\n";
    }
    return 0;
}
