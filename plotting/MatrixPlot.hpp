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
//     using namespace mcpu;                       // or: namespace np = mcpu;
//     mcpu::Matrix<double> x = linspace(0.0, 10.0, 200);
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
//
// TWO OUTPUT FORMATS WORTH KNOWING ABOUT, both one call away:
//
//     plt::interactive();  ...  plt::save("fig.html");   pan/zoom/hover
//     plt::latex();        ...  plt::save("fig.tex");    \input into a paper
//
// The first needs nothing installed beyond Plots. The second needs PGFPlotsX
// and a LaTeX, and also writes .pdf and .svg. See the backends section.
//
// ── What is here ────────────────────────────────────────────────────────────
//
//   2-D series   plot  scatter  bar  bars  barh  area  stem  stairs
//                stairs_pre  semilogx  semilogy  loglog  hist  hist2d
//                hexbin*  pie  hline  vline  hspan  vspan  quiver
//                fill_between  errorbar  stackplot
//   distributions boxplot**  violin**  density**   (call stats() first)
//   matrix views heatmap  contour  contourf  surface  wireframe  image  spy
//                contour3d*  contour_labels
//   3-D series   plot3  scatter3  surface3
//   styling      Style().label().color().width().dash().marker().markersize()
//                       .markercolor().alpha().fill().fillto().fillalpha()
//                       .fillcolor().yerror().xerror().ribbon()
//                       .sizes().colorby().linecolorby().set().raw()
//   axes         title  suptitle  xlabel  ylabel  zlabel  xlim  ylim  zlim
//                xticks  yticks  xscale  yscale  zscale  xflip  yflip
//                xrotation  yrotation  minorticks  aspect  view  polar
//                framestyle  axis_off  twinx  link
//   appearance   legend  grid  size  margin  dpi  fontsize  titlefontsize
//                colorbar  colorbar_title  colormap  clim  annotate  text
//                theme  transparent  set
//   layout       panel  layout
//   output       save  show  frame  gif  figure  script
//   backends     backend  interactive  latex  latex_engine  stats
//
// *  hexbin and contour3d are NOT supported by GR, the default backend, and
//    Plots will refuse them. Call plt::backend("pyplot") first, with PyPlot
//    installed in Julia.
// ** boxplot, violin and density need StatsPlots -- call plt::stats() once
//    before the first of them. Everything NOT marked is verified rendering
//    on GR.
//
// ── Styling and subplots ────────────────────────────────────────────────────
//
// Plots takes appearance settings as keyword arguments on the series itself,
// so they are given when the series is drawn. C++17 has no named arguments, so
// Style is a chainable stand-in:
//
//     plt::plot(x, y, plt::Style().label("fit").color(":red").width(3).dash(":dash"));
//     plt::plot(x, y, plt::Style().ribbon(sigma).fillalpha(0.3));
//
// Panels are banked as they are finished, and combined at output:
//
//     plt::plot(x, y);  plt::title("raw");       plt::panel();
//     plt::plot(x, f);  plt::title("filtered");  plt::layout(2, 1);
//     plt::save("both.png");
//
// ── Animation ───────────────────────────────────────────────────────────────
//
//     for (...) { draw(); plt::frame(); }
//     plt::gif("out.gif", 20);
//
// One Julia process for the whole movie. Fix the axes inside the loop, or
// Plots rescales per frame and the world appears to move instead of the data.

#include "julia_script.hpp"

#include <string>
#include <utility>
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
std::vector<double> flat(const mcpu::Matrix<T>& m) {
    std::vector<double> v((std::size_t)(m.rows() * m.cols()));
    for (long i = 0; i < m.rows(); i++)
        for (long j = 0; j < m.cols(); j++)
            v[(std::size_t)(i * m.cols() + j)] = double(std::real(m(i, j)));
    return v;
}
// Column-major, because that is how Julia reads a matrix.
template <typename T>
std::vector<double> colMajor(const mcpu::Matrix<T>& m) {
    std::vector<double> v((std::size_t)(m.rows() * m.cols()));
    for (long j = 0; j < m.cols(); j++)
        for (long i = 0; i < m.rows(); i++)
            v[(std::size_t)(j * m.rows() + i)] = double(std::real(m(i, j)));
    return v;
}
inline std::vector<double> indices(long n) {
    std::vector<double> v((std::size_t)n);
    for (long i = 0; i < n; i++) v[(std::size_t)i] = double(i + 1);
    return v;
}

inline std::string num(double v) { return std::to_string(v); }

// Every series in the package goes through this one emitter: create the figure
// if this is the first thing drawn, add to it otherwise. `args` is the
// positional part (one, two or three bound vector names) and `opts` the
// keyword part, so a new series type is a one-line addition rather than
// another copy of this string.
inline void emit(const std::string& args, const std::string& opts) {
    S().add("_cur = _cur === nothing ? plot(" + args + "; " + opts + ") : plot!(_cur, " + args +
            "; " + opts + ")");
}

}  // namespace detail

// ── Per-series style ────────────────────────────────────────────────────────
//
// Plots takes its appearance settings as keyword arguments on the series
// itself, so they have to be given when the series is drawn -- there is no
// "current series" to modify afterwards. C++17 has no named arguments, so a
// chainable builder stands in for them:
//
//     plt::plot(x, y, plt::Style().label("sin").color(":red").width(3));
//     plt::scatter(x, y, plt::Style().marker(":diamond").markersize(6).alpha(0.5));
//
// Anything this does not name is reachable with .set(), which passes a Plots
// attribute straight through.
class Style {
  public:
    Style& label(const std::string& s) { return put("label", detail::lab(s)); }
    // Colours are Plots symbols (":red", ":dodgerblue") or hex ("#ff8800").
    Style& color(const std::string& c) { return put("seriescolor", detail::sym(c)); }
    Style& width(double w) { return put("linewidth", detail::num(w)); }
    // ":dash", ":dot", ":dashdot", ":solid"
    Style& dash(const std::string& d) { return put("linestyle", detail::sym(d)); }
    // ":circle", ":square", ":diamond", ":utriangle", ":star5", ":none", ...
    Style& marker(const std::string& m) { return put("markershape", detail::sym(m)); }
    Style& markersize(double v) { return put("markersize", detail::num(v)); }
    Style& markercolor(const std::string& c) { return put("markercolor", detail::sym(c)); }
    Style& alpha(double a) { return put("seriesalpha", detail::num(a)); }
    // Fill down to a level -- 0 gives the usual filled area under a curve.
    Style& fill(double to = 0.0) { return put("fillrange", detail::num(to)); }
    Style& fillalpha(double a) { return put("fillalpha", detail::num(a)); }
    Style& fillcolor(const std::string& c) { return put("fillcolor", detail::sym(c)); }
    // Symmetric error bars and shaded bands, from a matrix of half-widths.
    template <typename T>
    Style& yerror(const mcpu::Matrix<T>& e) {
        return put("yerror", detail::S().bind(detail::flat(e)));
    }
    template <typename T>
    Style& xerror(const mcpu::Matrix<T>& e) {
        return put("xerror", detail::S().bind(detail::flat(e)));
    }
    template <typename T>
    Style& ribbon(const mcpu::Matrix<T>& r) {
        return put("ribbon", detail::S().bind(detail::flat(r)));
    }
    // matplotlib's scatter(s=...) and scatter(c=...): a size and a colour PER
    // POINT rather than one for the series. This is what turns a scatter into
    // a four-dimensional plot, and it is the single most-used thing matplotlib
    // has that a plain series API does not.
    template <typename T>
    Style& sizes(const mcpu::Matrix<T>& s) {
        return put("markersize", detail::S().bind(detail::flat(s)));
    }
    // Colour markers by a value, mapped through the current colormap. Add
    // plt::colorbar() to show the scale.
    template <typename T>
    Style& colorby(const mcpu::Matrix<T>& z) {
        return put("marker_z", detail::S().bind(detail::flat(z)));
    }
    // The same for a line, so a trajectory can be coloured by time or speed.
    template <typename T>
    Style& linecolorby(const mcpu::Matrix<T>& z) {
        return put("line_z", detail::S().bind(detail::flat(z)));
    }
    // Fill up to another CURVE rather than to a level -- matplotlib's
    // fill_between. fill(level) above is the constant case.
    template <typename T>
    Style& fillto(const mcpu::Matrix<T>& y2) {
        return put("fillrange", detail::S().bind(detail::flat(y2)));
    }
    Style& z_order(const std::string& o) { return put("z_order", detail::sym(o)); }
    // The escape hatch, same convention as plt::set.
    Style& set(const std::string& k, double v) { return put(k, detail::num(v)); }
    Style& set(const std::string& k, const std::string& v) { return put(k, detail::sym(v)); }
    // Integers matter: some Plots attributes (bins, for one) dispatch on Int
    // and reject a Float64 that happens to be whole.
    Style& set(const std::string& k, int v) { return put(k, std::to_string(v)); }
    // Verbatim Julia, for values that are neither a number, a symbol nor a
    // string -- a tuple of bound vectors, say. set() would quote this into a
    // String and Plots would reject it.
    Style& raw(const std::string& k, const std::string& v) { return put(k, v); }

    // Rendered as Julia keyword arguments, in the order they were set.
    std::string julia() const {
        std::string out;
        for (const auto& kv : kv_) {
            if (!out.empty()) out += ", ";
            out += kv.first + "=" + kv.second;
        }
        return out;
    }
    bool has(const std::string& k) const {
        for (const auto& kv : kv_)
            if (kv.first == k) return true;
        return false;
    }

  private:
    Style& put(const std::string& k, const std::string& v) {
        for (auto& kv : kv_)
            if (kv.first == k) {   // last setting of a key wins
                kv.second = v;
                return *this;
            }
        kv_.emplace_back(k, v);
        return *this;
    }
    std::vector<std::pair<std::string, std::string>> kv_;
};

namespace detail {

// Joins a seriestype with a style, defaulting the label to "no legend entry"
// when the caller did not give one.
inline std::string opts(const char* kind, const Style& st) {
    std::string o = "seriestype=:" + std::string(kind);
    if (!st.has("label")) o += ", label=false";
    const std::string s = st.julia();
    if (!s.empty()) o += ", " + s;
    return o;
}

inline void series(const std::vector<double>& x, const std::vector<double>& y, const char* kind,
                   const Style& st) {
    emit(S().bind(x) + ", " + S().bind(y), opts(kind, st));
}
inline void series(const std::vector<double>& x, const std::vector<double>& y,
                   const std::string& label, const char* kind) {
    series(x, y, kind, Style().label(label));
}
inline void series1(const std::vector<double>& y, const char* kind, const Style& st) {
    emit(S().bind(y), opts(kind, st));
}

// The 3-D counterpart. Plots takes a third positional vector and a
// seriestype ending in 3d; everything else is the same.
inline void series3(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& z, const char* kind, const Style& st) {
    emit(S().bind(x) + ", " + S().bind(y) + ", " + S().bind(z), opts(kind, st));
}
inline void series3(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& z, const std::string& label, const char* kind) {
    series3(x, y, z, kind, Style().label(label));
}

// Whether a twin y axis is currently the drawing target. Plots' twinx()
// returns a SUBPLOT that shares the x axis; series go to it, but the figure
// that gets saved is still the original Plot, so the root has to be put back
// before any output happens.
inline bool& twinned() {
    static bool on = false;
    return on;
}
inline void untwin() {
    if (!twinned()) return;
    S().add("_cur = _twinroot");
    twinned() = false;
}

// How many panels have been banked with panel(), and the grid they should be
// arranged into. Held on this side so save()/show()/frame() know whether a
// combine has to be emitted before the figure is finished.
inline int& panelCount() {
    static int n = 0;
    return n;
}
inline std::pair<int, int>& panelGrid() {
    static std::pair<int, int> rc{0, 0};
    return rc;
}

// Emits the combine when panels are in play, so the rest of the output path
// does not have to care whether it is looking at one plot or nine.
inline void finishPanels() {
    untwin();
    if (panelCount() == 0) return;
    S().add("push!(_panels, _cur)");
    const int n = panelCount() + 1;
    int r = panelGrid().first, c = panelGrid().second;
    if (r <= 0 || c <= 0) {   // no explicit grid: one row
        r = 1;
        c = n;
    }
    S().add("_cur = plot(_panels...; layout=(" + std::to_string(r) + ", " + std::to_string(c) +
            "))");
    panelCount() = 0;
    panelGrid() = {0, 0};
}

// Whether an animation is open in the current script. Kept on this side
// rather than probed in Julia so the first frame() knows to create the
// Animation and later ones know not to.
inline bool& animating() {
    static bool on = false;
    return on;
}

inline void attr(const std::string& key, const std::string& value) {
    S().add("_cur = _cur === nothing ? plot(; " + key + "=" + value + ") : plot!(_cur; " + key +
            "=" + value + ")");
}

template <typename T>
void matview(const mcpu::Matrix<T>& A, const char* fn, bool yflip) {
    const std::string n = S().bind(colMajor(A), A.rows(), A.cols());
    S().add(std::string("_cur = ") + fn + "(" + n + (yflip ? "; yflip=true)" : ")"));
}

}  // namespace detail

// ── Figures ─────────────────────────────────────────────────────────────────

// Start a fresh figure. Without it, drawing calls accumulate into the current
// one — which is exactly what lets two plot() calls share a pair of axes.
inline void figure() {
    detail::S().reset();
    detail::animating() = false;
    detail::twinned() = false;
    detail::panelCount() = 0;
    detail::panelGrid() = {0, 0};
}

// ── Subplots ────────────────────────────────────────────────────────────────
//
// Draw a panel, bank it with panel(), draw the next. layout() says how they
// should be arranged; without it they go in a single row. save(), show() and
// frame() all combine automatically, so nothing else changes:
//
//     plt::plot(x, y);   plt::title("raw");       plt::panel();
//     plt::plot(x, f);   plt::title("filtered");  plt::panel();
//     plt::layout(2, 1);
//     plt::save("both.png");
//
// The panel being drawn when the figure is finished is included too, so the
// last one needs no panel() call -- though one does no harm.

inline void panel() {
    detail::untwin();
    if (detail::panelCount() == 0) detail::S().add("_panels = Any[]");
    detail::S().add("push!(_panels, _cur)");
    detail::S().add("_cur = nothing");
    detail::panelCount()++;
}
inline void layout(int rows, int cols) { detail::panelGrid() = {rows, cols}; }

// The Julia this would run, for when a figure comes out wrong and the question
// is whether C++ or Plots is at fault.
inline std::string script() { return detail::S().text(); }

// ── Line and point series ───────────────────────────────────────────────────
//
// Each takes (y) with the x axis implied as 1..n, or (x, y). A matrix with more
// than one column is drawn as one series PER COLUMN, which is what MATLAB and
// Plots both do and what makes plot(A) useful for a family of curves.

template <typename Tx, typename Ty>
void plot(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "line");
}
template <typename T>
void plot(const mcpu::Matrix<T>& y, const std::string& label = "") {
    if (y.rows() == 1 || y.cols() == 1) {
        detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), label, "line");
        return;
    }
    for (long j = 0; j < y.cols(); j++) {
        std::vector<double> col((std::size_t)y.rows());
        for (long i = 0; i < y.rows(); i++)
            col[(std::size_t)i] = double(std::real(y(i, j)));
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

// The same two, taking a Style. Raw vectors are what a program that has read
// its data from a file usually holds, so they should not be second-class.
inline void plot(const std::vector<double>& x, const std::vector<double>& y, const Style& st) {
    detail::series(x, y, "line", st);
}
inline void plot(const std::vector<double>& y, const Style& st) {
    detail::series(detail::indices((long)y.size()), y, "line", st);
}
inline void scatter(const std::vector<double>& x, const std::vector<double>& y, const Style& st) {
    detail::series(x, y, "scatter", st);
}

template <typename Tx, typename Ty>
void scatter(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "scatter");
}
template <typename T>
void scatter(const mcpu::Matrix<T>& y, const std::string& label = "") {
    detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), label, "scatter");
}
// ── matplotlib's other verbs ────────────────────────────────────────────────

// fill_between: shade the region between two curves.
template <typename Tx, typename T1, typename T2>
void fill_between(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<T1>& y1,
                  const mcpu::Matrix<T2>& y2, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y1), "line",
                   Style().label(label).fillto(y2).fillalpha(0.35));
}

// errorbar: points with error bars, matplotlib's signature form. Pass only
// yerr for vertical bars, or both.
template <typename Tx, typename Ty, typename Te>
void errorbar(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Te>& yerr,
              const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), "scatter",
                   Style().label(label).yerror(yerr).markersize(4));
}

// stackplot: cumulative bands, one per column of A. Each series is filled down
// to the running total beneath it, which is what makes the bands stack rather
// than overlap.
template <typename Tx, typename T>
void stackplot(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<T>& A) {
    const long n = A.rows(), k = A.cols();
    mcpu::Matrix<double> running(n, 1);
    for (long c = 0; c < k; c++) {
        mcpu::Matrix<double> below = running;
        for (long i = 0; i < n; i++) running(i, 0) += double(A(i, c));
        detail::series(detail::flat(x), detail::flat(running), "line",
                       Style().label("series " + std::to_string(c + 1)).fillto(below)
                           .fillalpha(0.7));
    }
}

// Grouped and stacked bars, matplotlib's bar(bottom=...) and the width trick.
// One series per COLUMN of A.
template <typename Tx, typename T>
void bars(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<T>& A, bool stacked = false) {
    for (long c = 0; c < A.cols(); c++) {
        mcpu::Matrix<double> col(A.rows(), 1);
        for (long i = 0; i < A.rows(); i++) col(i, 0) = double(A(i, c));
        detail::series(detail::flat(x), detail::flat(col), "bar",
                       Style().label("series " + std::to_string(c + 1))
                           .set("bar_position", stacked ? ":stack" : ":dodge"));
    }
}

// ── Distributions (needs StatsPlots) ────────────────────────────────────────
//
// Plots itself has no box, violin or KDE recipe -- they live in StatsPlots.
// Call plt::stats() before the first of these in a figure to load it.
//
// These were long marked unverified here, because the install failed with
//
//     Pkg.add("StatsPlots")
//     ERROR: version 6.10.2+1 of package Qt6Base_jll is not available
//
// That was never StatsPlots' fault: a stale GR_jll in this depot pinned a Qt6
// build the registry had retired. Adding PGFPlotsX and PlotlyJS moved GR_jll
// 0.73.24 -> 0.73.27, the pin went with it, and StatsPlots v0.15.8 resolved on
// the next try. All three render.

inline void stats() { detail::S().add("using StatsPlots"); }

template <typename T>
void boxplot(const mcpu::Matrix<T>& v, const std::string& label = "") {
    detail::series1(detail::flat(v), "boxplot", Style().label(label));
}
template <typename T>
void violin(const mcpu::Matrix<T>& v, const std::string& label = "") {
    detail::series1(detail::flat(v), "violin", Style().label(label));
}
template <typename T>
void density(const mcpu::Matrix<T>& v, const std::string& label = "") {
    detail::series1(detail::flat(v), "density", Style().label(label));
}

// ── Three dimensions ────────────────────────────────────────────────────────
//
// Same call shape as plot/scatter with a third coordinate. Julia's Plots draws
// these with whichever backend is active; GR (the default) handles them without
// anything extra having to be installed.
//
//     plt::scatter3(x, y, z);
//     plt::zlim(0.0, 1000.0);
//     plt::view(45, 30);          // azimuth, elevation

template <typename Tx, typename Ty, typename Tz>
void plot3(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tz>& z,
           const std::string& label = "") {
    detail::series3(detail::flat(x), detail::flat(y), detail::flat(z), label, "path3d");
}
template <typename Tx, typename Ty, typename Tz>
void scatter3(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tz>& z,
              const std::string& label = "") {
    detail::series3(detail::flat(x), detail::flat(y), detail::flat(z), label, "scatter3d");
}
template <typename Tx, typename Ty, typename Tz>
void plot3(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tz>& z,
           const Style& st) {
    detail::series3(detail::flat(x), detail::flat(y), detail::flat(z), "path3d", st);
}
template <typename Tx, typename Ty, typename Tz>
void scatter3(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tz>& z,
              const Style& st) {
    detail::series3(detail::flat(x), detail::flat(y), detail::flat(z), "scatter3d", st);
}
// A surface over scattered (x, y, z) rather than over a grid -- surface(A)
// above is the gridded form.
template <typename Tx, typename Ty, typename Tz>
void surface3(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tz>& z,
              const Style& st = Style()) {
    detail::series3(detail::flat(x), detail::flat(y), detail::flat(z), "surface", st);
}

template <typename Tx, typename Ty>
void bar(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "bar");
}
template <typename Tx, typename Ty>
void stairs(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "steppost");
}

// Log axes set the scale AND draw, because setting a scale and forgetting to
// draw is the usual way to end up with an empty figure.
template <typename Tx, typename Ty>
void semilogy(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("yscale", ":log10");
}
template <typename Tx, typename Ty>
void semilogx(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("xscale", ":log10");
}
template <typename Tx, typename Ty>
void loglog(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    plot(x, y, label);
    detail::attr("xscale", ":log10");
    detail::attr("yscale", ":log10");
}

// ── Whole-matrix views ──────────────────────────────────────────────────────

// Rows run downward, the way a matrix is written on paper — Plots would
// otherwise put row 1 at the bottom.
// ── Styled forms ────────────────────────────────────────────────────────────
//
// Every series function above also takes a Style instead of a bare label.

template <typename Tx, typename Ty>
void plot(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const Style& st) {
    detail::series(detail::flat(x), detail::flat(y), "line", st);
}
template <typename T>
void plot(const mcpu::Matrix<T>& y, const Style& st) {
    detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), "line", st);
}
template <typename Tx, typename Ty>
void scatter(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const Style& st) {
    detail::series(detail::flat(x), detail::flat(y), "scatter", st);
}
template <typename T>
void scatter(const mcpu::Matrix<T>& y, const Style& st) {
    detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), "scatter", st);
}
template <typename Tx, typename Ty>
void bar(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const Style& st) {
    detail::series(detail::flat(x), detail::flat(y), "bar", st);
}

// ── More 2-D series ─────────────────────────────────────────────────────────

// Filled area under a curve. Same as plot() with Style().fill(0), named
// because it is common enough to deserve a verb.
template <typename Tx, typename Ty>
void area(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), "line", Style().label(label).fill(0.0));
}
// Horizontal bars. Plots has no :barh seriestype -- it is an ordinary bar with
// its orientation turned, which is also why it takes (x, y) in the same order.
template <typename Tx, typename Ty>
void barh(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), "bar",
                   Style().label(label).set("orientation", ":h"));
}
// Stems, as a discrete signal is usually drawn.
template <typename Tx, typename Ty>
void stem(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label,
                   "sticks");
}
template <typename T>
void stem(const mcpu::Matrix<T>& y, const std::string& label = "") {
    detail::series(detail::indices(y.rows() * y.cols()), detail::flat(y), label, "sticks");
}
// The other staircase: stairs() holds the value AFTER each point (steppost),
// this one holds it BEFORE (steppre). Which is right depends on whether a
// sample marks the start or the end of its interval.
template <typename Tx, typename Ty>
void stairs_pre(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y,
                const std::string& label = "") {
    detail::series(detail::flat(x), detail::flat(y), label, "steppre");
}

// Reference lines and shaded bands spanning the whole axes.
template <typename T>
void hline(const mcpu::Matrix<T>& ys, const Style& st = Style()) {
    detail::series1(detail::flat(ys), "hline", st);
}
template <typename T>
void vline(const mcpu::Matrix<T>& xs, const Style& st = Style()) {
    detail::series1(detail::flat(xs), "vline", st);
}
inline void hline(double y, const Style& st = Style()) {
    detail::series1(std::vector<double>{y}, "hline", st);
}
inline void vline(double x, const Style& st = Style()) {
    detail::series1(std::vector<double>{x}, "vline", st);
}
// Bands: pairs of edges, so {a, b} shades between a and b.
inline void hspan(double lo, double hi, const Style& st = Style()) {
    detail::series1(std::vector<double>{lo, hi}, "hspan", st);
}
inline void vspan(double lo, double hi, const Style& st = Style()) {
    detail::series1(std::vector<double>{lo, hi}, "vspan", st);
}

// Vector field. u and v are the components at each (x, y).
template <typename Tx, typename Ty, typename Tu, typename Tv>
void quiver(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, const mcpu::Matrix<Tu>& u,
            const mcpu::Matrix<Tv>& v, const Style& st = Style()) {
    const std::string nu = detail::S().bind(detail::flat(u));
    const std::string nv = detail::S().bind(detail::flat(v));
    Style s2 = st;
    s2.raw("quiver", "(" + nu + ", " + nv + ")");
    detail::series(detail::flat(x), detail::flat(y), "quiver", s2);
}

// Two-dimensional histogram, for a scatter dense enough that points overlap.
template <typename Tx, typename Ty>
void hist2d(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, int bins = 30) {
    detail::series(detail::flat(x), detail::flat(y), "histogram2d", Style().set("bins", bins));
}
// NOT AVAILABLE ON GR, the default backend -- it has no hexbin implementation
// and Plots raises "the backend must not support the series type". Call
// plt::backend("pyplot") or plt::backend("plotlyjs") first, with that package
// installed in Julia. hist2d above does the same job on any backend.
template <typename Tx, typename Ty>
void hexbin(const mcpu::Matrix<Tx>& x, const mcpu::Matrix<Ty>& y, int bins = 30) {
    detail::series(detail::flat(x), detail::flat(y), "hexbin", Style().set("bins", bins));
}

// Pie chart. `values` are the slice sizes; labels are optional.
template <typename T>
void pie(const mcpu::Matrix<T>& values, const std::vector<std::string>& labels = {}) {
    const std::string nv = detail::S().bind(detail::flat(values));
    if (labels.empty()) {
        detail::S().add("_cur = pie(" + nv + ")");
    } else {
        std::string l = "[";
        for (std::size_t i = 0; i < labels.size(); i++)
            l += (i ? ", " : "") + detail::q(labels[i]);
        l += "]";
        detail::S().add("_cur = pie(" + l + ", " + nv + ")");
    }
}

template <typename T>
void heatmap(const mcpu::Matrix<T>& A) { detail::matview(A, "heatmap", true); }
template <typename T>
void surface(const mcpu::Matrix<T>& A) { detail::matview(A, "surface", false); }
template <typename T>
void contour(const mcpu::Matrix<T>& A) { detail::matview(A, "contour", false); }
// Filled contours, and the wireframe / 3-D contour views of the same matrix.
template <typename T>
void contourf(const mcpu::Matrix<T>& A) { detail::matview(A, "contourf", false); }
template <typename T>
void wireframe(const mcpu::Matrix<T>& A) { detail::matview(A, "wireframe", false); }
// NOT AVAILABLE ON GR either, for the same reason -- verified against
// contour3d(z), contour3d(x, y, z) and seriestype=:contour3d, all of which GR
// rejects. Use plt::backend("pyplot") for it, or contour()/contourf() for the
// flat view, which GR does support.
template <typename T>
void contour3d(const mcpu::Matrix<T>& A) { detail::matview(A, "contour3d", false); }

// A matrix shown as an image: square pixels, no colour bar, first row at the
// top. Implemented as a heatmap with those settings rather than through
// Images.jl, so it needs no package beyond Plots.
template <typename T>
void image(const mcpu::Matrix<T>& A) {
    detail::matview(A, "heatmap", true);
    detail::attr("aspect_ratio", ":equal");
    detail::attr("colorbar", "false");
}

// Sparsity pattern: 1 where the matrix is non-zero. Useful for seeing what a
// factorisation did to the structure.
template <typename T>
void spy(const mcpu::Matrix<T>& A, double tol = 0.0) {
    mcpu::Matrix<double> P(A.rows(), A.cols());
    for (long i = 0; i < A.rows(); i++)
        for (long j = 0; j < A.cols(); j++)
            // mcpu:: is required, not optional: the argument here is a plain
            // scalar, so argument-dependent lookup has no mcpu type to
            // follow back into the namespace.
            P(i, j) = (mcpu::magnitude(A(i, j)) > tol) ? 1.0 : 0.0;
    heatmap(P);
}

template <typename T>
void hist(const mcpu::Matrix<T>& v, int bins = 20, const std::string& label = "") {
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
inline void zlim(double lo, double hi) {
    detail::attr("zlims", "(" + std::to_string(lo) + ", " + std::to_string(hi) + ")");
}
// Camera angle for a 3-D plot: azimuth and elevation in degrees. Worth setting
// explicitly in an animation, since the default depends on the backend and a
// drifting viewpoint is indistinguishable from drifting data.
inline void view(double azimuth, double elevation) {
    detail::attr("camera", "(" + std::to_string(azimuth) + ", " + std::to_string(elevation) + ")");
}
// Log axes without going through semilogy/loglog, for when only one call in a
// figure needs it. Scale is ":log10", ":log2", ":ln" or ":identity".
inline void xscale(const std::string& s = ":log10") { detail::attr("xscale", detail::sym(s)); }
inline void yscale(const std::string& s = ":log10") { detail::attr("yscale", detail::sym(s)); }
inline void zscale(const std::string& s = ":log10") { detail::attr("zscale", detail::sym(s)); }

// Reverse an axis, for data that reads naturally downwards (depth, magnitude).
inline void xflip(bool on = true) { detail::attr("xflip", on ? "true" : "false"); }
inline void yflip(bool on = true) { detail::attr("yflip", on ? "true" : "false"); }

// Explicit tick positions, and optionally their labels.
template <typename T>
void xticks(const mcpu::Matrix<T>& at) {
    detail::attr("xticks", detail::S().bind(detail::flat(at)));
}
// Raw-vector forms, for data that came from a file rather than a Matrix.
//
// NOTE the two-argument form is usually the one wanted on a LOG axis. Giving
// positions alone lets Plots label them with the transformed value -- ticks at
// 128 and 512 come out as 10^2.107 and 10^2.709 -- so the labels have to be
// supplied explicitly to read as the numbers they are.
inline void xticks(const std::vector<double>& at) {
    detail::attr("xticks", detail::S().bind(at));
}
inline void xticks(const std::vector<double>& at, const std::vector<std::string>& labels) {
    std::string l = "[";
    for (std::size_t i = 0; i < labels.size(); i++) l += (i ? ", " : "") + detail::q(labels[i]);
    l += "]";
    detail::attr("xticks", "(" + detail::S().bind(at) + ", " + l + ")");
}
inline void yticks(const std::vector<double>& at) {
    detail::attr("yticks", detail::S().bind(at));
}
template <typename T>
void yticks(const mcpu::Matrix<T>& at) {
    detail::attr("yticks", detail::S().bind(detail::flat(at)));
}
template <typename T>
void xticks(const mcpu::Matrix<T>& at, const std::vector<std::string>& labels) {
    std::string l = "[";
    for (std::size_t i = 0; i < labels.size(); i++) l += (i ? ", " : "") + detail::q(labels[i]);
    l += "]";
    detail::attr("xticks", "(" + detail::S().bind(detail::flat(at)) + ", " + l + ")");
}

// 1.0 keeps the axes square in DATA units; ":equal" does it in screen units.
inline void aspect(double ratio) { detail::attr("aspect_ratio", detail::num(ratio)); }
inline void aspect(const std::string& mode) { detail::attr("aspect_ratio", detail::sym(mode)); }

inline void colorbar(bool on = true) { detail::attr("colorbar", on ? "true" : "false"); }
// Colour scheme for heatmaps, surfaces and contours: ":viridis", ":plasma",
// ":thermal", ":greys", ...
inline void colormap(const std::string& name) { detail::attr("seriescolor", detail::sym(name)); }
inline void clim(double lo, double hi) {
    detail::attr("clims", "(" + detail::num(lo) + ", " + detail::num(hi) + ")");
}

// Text placed at a data coordinate.
inline void annotate(double x, double y, const std::string& text) {
    detail::S().add("_cur = annotate!(_cur, " + detail::num(x) + ", " + detail::num(y) + ", " +
                    detail::q(text) + ")");
}

// Font sizes, in points.
inline void fontsize(double pt) {
    detail::attr("guidefontsize", detail::num(pt));
    detail::attr("tickfontsize", detail::num(pt));
    detail::attr("legendfontsize", detail::num(pt));
}
inline void titlefontsize(double pt) { detail::attr("titlefontsize", detail::num(pt)); }
inline void dpi(int d) { detail::attr("dpi", std::to_string(d)); }

// ── Backends ────────────────────────────────────────────────────────────────
//
// Call before drawing; the choice applies to the figure being built.
//
//   "gr"         the default. Fast, needs nothing installed, raster or vector.
//   "plotly"     INTERACTIVE HTML -- pan, zoom, hover-to-read-values. Built
//                into Plots, so it needs no extra package at all. save() to a
//                .html and open it in a browser.
//   "plotlyjs"   the same figures with a native window for show(). Needs
//                PlotlyJS.jl.
//   "pgfplotsx"  LATEX-QUALITY vector output. The figure is compiled by a real
//                TeX run, so the fonts and maths match the document it lands
//                in. Needs PGFPlotsX.jl and a working LaTeX. Saves .pdf, .svg
//                and -- most usefully -- .tex, which you can \input straight
//                into a paper and then edit by hand.
//   "pyplot"     matplotlib, for the handful of series GR does not implement
//                (hexbin, contour3d). Needs PyPlot.jl.
//
// WHY pgfplotsx GETS SPECIAL HANDLING. PGFPlotsX defaults to lualatex, and a
// LuaTeX installation missing luaotfload -- which is the state of this machine
// -- fails with "The latex command `lualatex ...` failed" and no hint that the
// engine is the problem rather than the plot. pdflatex is both more commonly
// present and enough for everything pgfplots emits, so it is selected here.
// Override with latex_engine() if your document needs lualatex or xelatex.
inline void latex_engine(const std::string& engine = "PDFLATEX") {
    detail::S().add("using PGFPlotsX");
    detail::S().add("PGFPlotsX.latexengine!(PGFPlotsX." + engine + ")");
}

inline void backend(const std::string& name) {
    if (name == "pgfplotsx" || name == "pgfplots") latex_engine();
    detail::S().add(name + "()");
}

// Named shortcuts for the two worth reaching for by intent rather than by
// package name.
inline void interactive() { backend("plotly"); }   // -> save("figure.html")
inline void latex() { backend("pgfplotsx"); }      // -> save("figure.tex" or ".pdf")
// ":dark", ":ggplot2", ":juno", ":solarized", ":wong", ":default", ...
inline void theme(const std::string& name) {
    detail::S().add("theme(" + detail::sym(name) + ")");
}

// ── A second y axis ─────────────────────────────────────────────────────────
//
// matplotlib's twinx(). Everything drawn after this goes on a right-hand axis
// sharing the same x, until the figure or panel is finished:
//
//     plt::plot(t, signal, "signal");
//     plt::twinx();
//     plt::plot(t, temp, plt::Style().label("temp").color(":red"));
//     plt::ylabel("degrees");        // the RIGHT axis, from here on
//     plt::ylabel_left("volts");     // the left one, which needs its own call
//
// There is no untwin() to call: panel(), save(), show() and frame() all put
// the root axis back on their own, because the figure that gets saved is the
// original plot and not the twin subplot.
//
// TWO THINGS PLOTS DOES THAT WILL SURPRISE YOU, both verified rather than
// guessed at:
//
//   * twinx() CLOBBERS the left axis label. Setting ylabel before the call
//     does not survive it, which is why ylabel_left() exists -- it writes to
//     the root subplot after the twin has been made.
//   * the colour cycle RESTARTS on the twin, so the second series comes out
//     the same colour as the first. Give it an explicit .color(), or the two
//     axes are indistinguishable.
//
// matplotlib has the same second problem and the same remedy.
inline void twinx() {
    if (detail::twinned()) return;
    detail::S().add("_twinroot = _cur");
    detail::S().add("_cur = twinx(_twinroot)");
    detail::twinned() = true;
}

// The LEFT axis label once a twin exists. Plain ylabel() targets whichever
// axis is current, which after twinx() is the right one.
inline void ylabel_left(const std::string& s) {
    if (!detail::twinned()) {
        detail::attr("ylabel", detail::q(s));
        return;
    }
    detail::S().add("plot!(_twinroot[1]; ylabel=" + detail::q(s) + ")");
}

// ── Text ────────────────────────────────────────────────────────────────────
//
// annotate() above places plain text. This is the full form: size, colour,
// alignment (":left", ":center", ":right") and rotation in degrees, which is
// what matplotlib's text(..., ha=, rotation=) gives.
inline void text(double x, double y, const std::string& s, double pt = 10,
                 const std::string& align = ":center", double rotation = 0.0) {
    detail::S().add("_cur = annotate!(_cur, " + detail::num(x) + ", " + detail::num(y) +
                    ", Plots.text(" + detail::q(s) + ", " + detail::num(pt) + ", " +
                    detail::sym(align) + ", rotation=" + detail::num(rotation) + "))");
}

// A title across a whole grid of panels, above the individual panel titles.
// Set it before save(); it applies to the combined figure.
inline void suptitle(const std::string& s) { detail::attr("plot_title", detail::q(s)); }

// ── Frame, ticks and projection ─────────────────────────────────────────────

// ":box" draws all four sides, ":axes" only left and bottom, ":none" removes
// the frame entirely, ":grid" keeps the grid alone. matplotlib's spines.
inline void framestyle(const std::string& style = ":box") {
    detail::attr("framestyle", detail::sym(style));
}
inline void axis_off() { framestyle(":none"); }

// Rotate tick labels, for the usual case of long category names colliding.
inline void xrotation(double degrees) { detail::attr("xrotation", detail::num(degrees)); }
inline void yrotation(double degrees) { detail::attr("yrotation", detail::num(degrees)); }

inline void minorticks(bool on = true) {
    detail::attr("minorticks", on ? "true" : "false");
    detail::attr("minorgrid", on ? "true" : "false");
}

// Polar projection: x is then the angle in radians and y the radius.
inline void polar(bool on = true) {
    detail::attr("proj", on ? ":polar" : ":none");
}

// Draw the level values onto a contour plot -- matplotlib's clabel.
inline void contour_labels(bool on = true) {
    detail::attr("contour_labels", on ? "true" : "false");
}

// A caption on the colour bar.
inline void colorbar_title(const std::string& s) {
    detail::attr("colorbar_title", detail::q(s));
}

// Tie the axes of a panel grid together, so they zoom and scale as one:
// ":x", ":y", ":both", ":none". matplotlib's sharex/sharey. Set it before
// save(), like layout().
inline void link(const std::string& which = ":x") { detail::attr("link", detail::sym(which)); }

// Anything this header does not wrap. The key is a Plots attribute name; a
// value starting ':' becomes a Symbol — set("linewidth", 3.0),
// set("seriescolor", ":red").
inline void set(const std::string& key, double v) { detail::attr(key, std::to_string(v)); }
inline void set(const std::string& key, const std::string& v) { detail::attr(key, detail::sym(v)); }

// ── Output ──────────────────────────────────────────────────────────────────

// Format comes from the extension: .png, .pdf, .svg, .html, and whatever else
// the active Plots backend supports.
// A transparent background, for dropping a figure onto a slide.
inline void transparent(bool on = true) {
    detail::attr("background_color", on ? ":transparent" : ":white");
    detail::attr("foreground_color", on ? ":black" : ":black");
}

inline void save(const std::string& path) {
    detail::finishPanels();
    detail::S().add("savefig(_cur, " + detail::q(path) + ")");
    jlx::run("save");
    detail::S().reset();
    detail::animating() = false;
}
// Opens a window; needs a display. On a headless machine use save().
inline void show() {
    detail::finishPanels();
    detail::S().add("display(_cur); readline()");
    jlx::run("show");
    detail::S().reset();
    detail::animating() = false;
}

// ── Animation ───────────────────────────────────────────────────────────────
//
// A movie is built the same way a figure is: draw, then frame(), repeated, and
// gif() at the end writes the whole thing out. Everything still happens in ONE
// Julia process, which is the reason it is done this way rather than by calling
// show() in a loop -- that would start a process per frame and, since show()
// waits on the window, never get past the first one.
//
//     for (int t = 0; t < steps; t++) {
//         simulate();
//         plt::scatter(x, y);
//         plt::title("t = " + std::to_string(t * dt));
//         plt::frame();
//     }
//     plt::gif("out.gif", 20);
//
// FIX THE AXES with xlim/ylim inside the loop. Without them Plots rescales to
// each frame's data and the picture jitters, which reads as the world moving
// rather than the particles.
//
// Do NOT call figure() between frames: it clears the script, and the frames
// already captured go with it. frame() starts the next figure by itself.

// Capture the current figure as one frame, then begin a fresh one.
inline void frame() {
    detail::untwin();
    if (detail::S().empty())
        throw std::runtime_error("plt::frame: nothing has been drawn yet");
    detail::finishPanels();
    if (!detail::animating()) {
        detail::S().add("_anim = Animation()");
        detail::animating() = true;
    }
    detail::S().add("frame(_anim, _cur)");
    detail::S().add("_cur = nothing");
}

// Write the captured frames out. The extension picks the format: .gif works
// everywhere, .mp4 needs ffmpeg on the machine running Julia.
inline void gif(const std::string& path, int fps = 15) {
    if (!detail::animating())
        throw std::runtime_error(
            "plt::gif: no frames captured - draw something and call plt::frame() first");
    detail::S().add("gif(_anim, " + detail::q(path) + ", fps=" + std::to_string(fps) + ")");
    jlx::run("gif");
    detail::S().reset();
    detail::animating() = false;
}

}  // namespace plt
