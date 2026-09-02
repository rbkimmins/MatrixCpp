#pragma once

// ==========================================================================
//  The Plotting Package
// ==========================================================================
//
// Julia's Plots.jl, driven from C++, with no Julia syntax anywhere in the
// caller's code and no build flags beyond the include path:
//
//     #include "plotting/MatrixPlot.hpp"
//
//     Matrix<double> x = linspace(0.0, 10.0, 200);
//     plt::plot(x, x.sin(), "sin");
//     plt::plot(x, x.cos(), "cos");
//     plt::title("trig");  plt::xlabel("x");  plt::legend();
//     plt::save("trig.png");
//
//     g++ -std=c++17 -O2 -fopenmp demo.cpp -o demo      # that is the whole build
//
// The model is matplotlib's, not MATLAB's: there is a CURRENT FIGURE, drawing
// calls add to it, and save() or show() finishes it. figure() starts a new one.
// That is the idiom the NumPy side of the world already knows, which is the
// point — this is meant to feel like matplotlib, not like a binding.
//
// Nothing runs until save() or show(). Everything before that just records
// what to draw, so one Julia process is started per figure rather than per
// call. Expect roughly a second there while Plots loads. See julia_script.hpp
// for why this is a subprocess and not an embedded runtime.
//
// REQUIRES: julia on PATH, with Plots.jl installed.

#include "julia_script.hpp"

#include <string>
#include <vector>

namespace plt {

namespace detail {

inline jlx::Script& S() { return jlx::Script::current(); }

// Julia string literal, quoted.
inline std::string q(const std::string& s) {
    return "\"" + jlx::detail::jstr(s) + "\"";
}
// A label of "" means "no entry in the legend", which Plots spells `false`.
inline std::string lab(const std::string& s) { return s.empty() ? "false" : q(s); }
// A value written ":something" is a Plots Symbol; anything else is a string.
inline std::string sym(const std::string& s) {
    return (!s.empty() && s[0] == ':') ? s : q(s);
}

template <typename T>
std::vector<double> flat(const Matrix<T>& m) {
    std::vector<double> v((std::size_t)(m.rows() * m.cols()));
    for (long i = 0; i < m.rows(); i++)
        for (long j = 0; j < m.cols(); j++)
            v[(std::size_t)(i * m.cols() + j)] = double(std::real(m(int(i), int(j))));
    return v;
}
// Column-major, because that is how Julia reads a matrix.
template <typename T>
std::vector<double> colMajor(const Matrix<T>& m) {
    std::vector<double> v((std::size_t)(m.rows() * m.cols()));
    for (long j = 0; j < m.cols(); j++)
        for (long i = 0; i < m.rows(); i++)
            v[(std::size_t)(j * m.rows() + i)] = double(std::real(m(int(i), int(j))));
    return v;
}
inline std::vector<double> indices(long n) {
    std::vector<double> v((std::size_t)n);
    for (long i = 0; i < n; i++) v[(std::size_t)i] = double(i + 1);
    return v;
}

// Every series goes through here: create the figure if this is the first one,
// add to it otherwise.
inline void series(const std::vector<double>& x, const std::vector<double>& y,
                   const std::string& label, const char* kind) {
    const std::string nx = S().bind(x), ny = S().bind(y);
    S().add("_cur = _cur === nothing ? plot(" + nx + ", " + ny + "; label=" + lab(label) +
            ", seriestype=:" + kind + ") : plot!(_cur, " + nx + ", " + ny +
            "; label=" + lab(label) + ", seriestype=:" + kind + ")");
}

inline void attr(const std::string& key, const std::string& value) {
    S().add("_cur = _cur === nothing ? plot(; " + key + "=" + value + ") : plot!(_cur; " + key +
            "=" + value + ")");
}

template <typename T>
void matview(const Matrix<T>& A, const char* fn, bool yflip) {
    const std::string n = S().bind(colMajor(A), A.rows(), A.cols());
    S().add(std::string("_cur = ") + fn + "(" + n + (yflip ? "; yflip=true)" : ")"));
}

}  // namespace detail

// ── Figures ─────────────────────────────────────────────────────────────────

// Start a fresh figure. Without it, drawing calls accumulate into the current
// one — which is exactly what lets two plot() calls share a pair of axes.
inline void figure() { detail::S().reset(); }

// The Julia this would run, for when a figure comes out wrong and the question
// is whether C++ or Plots is at fault.
inline std::string script() { return detail::S().text(); }

// ── Line and point series ───────────────────────────────────────────────────
//
// Each takes (y) with the x axis implied as 1..n, or (x, y). A matrix with more
// than one column is drawn as one series PER COLUMN, which is what MATLAB and
// Plots both do and what makes plot(A) useful for a family of curves.

template <typename Tx, typename Ty>
void plot(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "line");
}
template <typename T>
void plot(const Matrix<T>& y, const std::string& label = "") {
    if (y.rows() == 1 || y.cols() == 1) {
        detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), label, "line");
        return;
    }
    for (long j = 0; j < y.cols(); j++) {
        std::vector<double> col((std::size_t)y.rows());
        for (long i = 0; i < y.rows(); i++)
            col[(std::size_t)i] = double(std::real(y(int(i), int(j))));
        detail::series(detail::indices(y.rows()), col,
                       label.empty() ? "" : label + " " + std::to_string(j + 1), "line");
    }
}
inline void plot(const std::vector<double>& x, const std::vector<double>& y,
                 const std::string& label = "") {
    detail::series(x, y, label, "line");
}
inline void plot(const std::vector<double>& y, const std::string& label = "") {
    detail::series(detail::indices((long)y.size()), y, label, "line");
}

template <typename Tx, typename Ty>
void scatter(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "scatter");
}
template <typename T>
void scatter(const Matrix<T>& y, const std::string& label = "") {
    detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), label, "scatter");
}
template <typename Tx, typename Ty>
void bar(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "bar");
}
template <typename Tx, typename Ty>
void stairs(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "steppost");
}

// Log axes set the scale AND draw, because setting a scale and forgetting to
// draw is the usual way to end up with an empty figure.
template <typename Tx, typename Ty>
void semilogy(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("yscale", ":log10");
}
template <typename Tx, typename Ty>
void semilogx(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("xscale", ":log10");
}
template <typename Tx, typename Ty>
void loglog(const Matrix<Tx>& x, const Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("xscale", ":log10");
    detail::attr("yscale", ":log10");
}

// ── Whole-matrix views ──────────────────────────────────────────────────────

// Rows run downward, the way a matrix is written on paper — Plots would
// otherwise put row 1 at the bottom.
template <typename T>
void heatmap(const Matrix<T>& A) { detail::matview(A, "heatmap", true); }
template <typename T>
void surface(const Matrix<T>& A) { detail::matview(A, "surface", false); }
template <typename T>
void contour(const Matrix<T>& A) { detail::matview(A, "contour", false); }

// Sparsity pattern: 1 where the matrix is non-zero. Useful for seeing what a
// factorisation did to the structure.
template <typename T>
void spy(const Matrix<T>& A, double tol = 0.0) {
    Matrix<double> P(A.rows(), A.cols());
    for (long i = 0; i < A.rows(); i++)
        for (long j = 0; j < A.cols(); j++)
            P(int(i), int(j)) = (magnitude(A(int(i), int(j))) > tol) ? 1.0 : 0.0;
    heatmap(P);
}

template <typename T>
void hist(const Matrix<T>& v, int bins = 20, const std::string& label = "") {
    const std::string n = detail::S().bind(detail::flat(v));
    detail::S().add("_cur = _cur === nothing ? histogram(" + n + "; bins=" +
                    std::to_string(bins) + ", label=" + detail::lab(label) +
                    ") : histogram!(_cur, " + n + "; bins=" + std::to_string(bins) +
                    ", label=" + detail::lab(label) + ")");
}

// ── Labels and axes ─────────────────────────────────────────────────────────

inline void title(const std::string& s) { detail::attr("title", detail::q(s)); }
inline void xlabel(const std::string& s) { detail::attr("xlabel", detail::q(s)); }
inline void ylabel(const std::string& s) { detail::attr("ylabel", detail::q(s)); }
inline void zlabel(const std::string& s) { detail::attr("zlabel", detail::q(s)); }
// Position is a Plots symbol: ":topright", ":bottomleft", ":outertop", ...
inline void legend(const std::string& pos = ":topright") {
    detail::attr("legend", detail::sym(pos));
}
inline void grid(bool on = true) { detail::attr("grid", on ? "true" : "false"); }
// Extra space around the axes, in millimetres. Plots measures margins in real
// units, so a long y-label needs this rather than a bigger figure — growing the
// figure scales the label too and clips it just the same.
inline void margin(double mm) {
    const std::string v = std::to_string(mm) + "Plots.mm";
    detail::attr("left_margin", v);
    detail::attr("bottom_margin", v);
    detail::attr("top_margin", v);
    detail::attr("right_margin", v);
}
inline void size(int w, int h) {
    detail::attr("size", "(" + std::to_string(w) + ", " + std::to_string(h) + ")");
}
inline void xlim(double lo, double hi) {
    detail::attr("xlims", "(" + std::to_string(lo) + ", " + std::to_string(hi) + ")");
}
inline void ylim(double lo, double hi) {
    detail::attr("ylims", "(" + std::to_string(lo) + ", " + std::to_string(hi) + ")");
}
// Anything this header does not wrap. The key is a Plots attribute name; a
// value starting ':' becomes a Symbol — set("linewidth", 3.0),
// set("seriescolor", ":red").
inline void set(const std::string& key, double v) { detail::attr(key, std::to_string(v)); }
inline void set(const std::string& key, const std::string& v) { detail::attr(key, detail::sym(v)); }

// ── Output ──────────────────────────────────────────────────────────────────

// Format comes from the extension: .png, .pdf, .svg, .html, and whatever else
// the active Plots backend supports.
inline void save(const std::string& path) {
    detail::S().add("savefig(_cur, " + detail::q(path) + ")");
    jlx::run("save");
    detail::S().reset();
}
// Opens a window; needs a display. On a headless machine use save().
inline void show() {
    detail::S().add("display(_cur); readline()");
    jlx::run("show");
    detail::S().reset();
}

}  // namespace plt
