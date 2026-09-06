#pragma once

// ==========================================================================
//  Bare-bones plotting, through gnuplot
// ==========================================================================
//
// The plotting/ package is the powerful one: 136 entry points, statistical
// plots, interactive HTML, LaTeX vector output. It is also the only part of
// MatrixCpp that is not just a header -- it needs Julia and Plots.jl, which on
// this machine is a 3.2 GB depot against gnuplot's 3.3 MB.
//
// This header is the other end of that trade. It draws through gnuplot, which
// is in every distribution's base repository and is very often already
// installed, and it covers the plots people actually reach for:
//
//     #include "basic/MatrixCpp.hpp"
//     using namespace mcpu;
//
//     Matrix<double> x = linspace(0.0, 10.0, 200);
//     gp::plot(x, x.sin(), "sin");
//     gp::plot(x, x.cos(), "cos");
//     gp::title("trig");
//     gp::save("trig.png");
//
// The two are INDEPENDENT and can be used side by side -- plt:: for a figure
// that goes in a paper, gp:: for a look at what the numbers are doing. Neither
// includes the other, and nothing here is a fallback path for plt::.
//
// ── What is here ────────────────────────────────────────────────────────────
//
//   2-D series   plot  scatter  steps  stem  bar  hist  series
//   matrix views heatmap  surface  contour
//   3-D series   plot3  scatter3
//   axes         title  xlabel  ylabel  zlabel  xlim  ylim  zlim  view
//                legend  grid  logscale  size
//   output       save  show  draw  figure  gif_begin  frame  gif_end
//   escape       raw  term  echo  sync
//
// If you want a violin plot, a twin axis, a panel layout, hover-and-zoom HTML
// or a \input-able .tex, use plt::. This header is deliberately not that.
//
// ── Why a live pipe, and not a script ───────────────────────────────────────
//
// plt:: builds a Julia script and runs it once at the end, because starting a
// Julia process costs about a second and you do not want to pay that per
// frame. gnuplot starts in about 30 ms and, more to the point, is designed to
// be driven down a pipe: one process is opened on first use and stays open for
// the life of the program, so drawing again is just more text on a file
// descriptor.
//
// That is what makes draw() a genuine live view rather than a GIF assembled
// afterwards -- measured at 7.7 ms a frame, about 130 a second. In a
// simulation loop, call draw() and watch the window update:
//
//     for (int t = 0; t < steps; t++) {
//         step();
//         gp::scatter(x, y);
//         gp::xlim(0.0, L); gp::ylim(0.0, L);
//         gp::title("t = " + std::to_string(t * dt));
//         gp::draw();                      // window updates in place
//     }
//
// Swap draw() for frame() between gif_begin() and gif_end() and the same loop
// writes an animated GIF instead -- gnuplot has an `animate` terminal, so the
// frames are encoded as they are produced and nothing is buffered.
//
// ── Data goes down the pipe, not through a temp file ─────────────────────────
//
// Every series is written as a gnuplot DATABLOCK, an inline heredoc:
//
//     $D0 << EOD
//     0 0.0
//     1 0.841470984807896
//     EOD
//     plot $D0 using 1:2 with lines title "sin"
//
// so there are no temporary files to name, collide over, or clean up, and the
// data never touches the disk. Values are written with %.17g, which is the
// shortest form that round-trips a double exactly.
//
// ── Requirements and limits, stated plainly ─────────────────────────────────
//
// REQUIRES: gnuplot on PATH. Nothing else -- no Julia, no Python, no LaTeX
// unless you ask for a LaTeX terminal.
//
//   * Series take mcpu::Matrix, not the proxies that A(all, 0) returns.
//     Template deduction runs before user-defined conversions, so a proxy will
//     not bind; wrap it, Matrix<double>(A(all, 0)), exactly as plt:: requires.
//   * A matrix is drawn by index, not by coordinate. heatmap/surface/contour
//     use gnuplot's `matrix` reader, whose axes are the column and row number.
//   * Complex matrices are drawn by their real part, silently, as in plt::.
//   * show() opens a window with the `persist` flag so it survives the program
//     exiting. On a headless machine use save().
//
// Part of the Basic Matrix Package -- include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if a plot is genuinely all you need.

#include "matrix.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
    #include <unistd.h>
    #include <sys/wait.h>
#endif

namespace gp {

namespace detail {

#if defined(_WIN32)
    inline const char* probeCmd() { return "gnuplot --version > NUL 2>&1"; }
#else
    inline const char* probeCmd() { return "gnuplot --version > /dev/null 2>&1"; }
#endif

// A gnuplot string literal. Both the quote and the backslash are special
// inside one, and a stray quote in a title would otherwise swallow the rest of
// the command and produce an error message about something else entirely.
inline std::string q(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

// %.17g is the shortest decimal form guaranteed to round-trip an IEEE double.
// std::to_string would give 6 decimals and quietly lose the rest.
inline std::string num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

// The real part, so complex matrices plot rather than fail to compile. This
// matches what plt:: does with them.
template <typename T>
inline double re(const T& v) { return double(std::real(v)); }

// ONE gnuplot process for the life of the program, opened on first use.
//
// Series data is written to the pipe the moment it is handed over, but the
// `plot` command that references it is held back: gnuplot draws every series
// of a figure in a single command, so the list has to be complete first.
// Settings are held back for the same reason and because it lets them be given
// in any order relative to the series, which is what callers expect.
class Session {
  public:
    static Session& current() {
        static Session s;
        return s;
    }

    void cmd(const std::string& s) {
        ensure();
        if (echo_) std::fprintf(stderr, "gnuplot| %s\n", s.c_str());
        std::fputs(s.c_str(), in_);
        std::fputc('\n', in_);
    }

    void flush() {
        if (in_) {
            std::fflush(in_);
            // A gnuplot that died takes the pipe with it. Without this the
            // program carries on writing into a broken descriptor and the
            // plots simply never appear, with nothing said.
            if (std::ferror(in_))
                throw std::runtime_error(
                    "gnuplot: the plotting process has died; no further plots will appear");
        }
    }

    // Blocks until gnuplot has worked through everything sent so far.
    //
    // WHY THIS EXISTS. Commands go down a pipe, so they return the moment they
    // are queued -- measured here, save() returned 11 to 43 ms before the file
    // it named existed on disk, and stat() on it failed outright. A function
    // called save() that has not saved when it returns is a genuine trap: any
    // program that writes a figure and then reads, uploads or checks it would
    // race.
    //
    // gnuplot reads commands strictly in order, so a sentinel printed after the
    // output is closed cannot come back early. `set print "-"` sends the
    // `print` command to stdout, which is the half of the connection we can
    // read; gnuplot's own warnings and errors keep going to stderr and so
    // cannot be mistaken for the sentinel.
    void sync() {
        if (!in_ || !out_) return;
        cmd("set print \"-\"");
        cmd("print \"" + std::string(kSentinel) + "\"");
        flush();
        char buf[512];
        while (std::fgets(buf, sizeof(buf), out_)) {
            if (std::string(buf).find(kSentinel) != std::string::npos) return;
        }
        throw std::runtime_error(
            "gnuplot: the process closed its output before finishing; the figure may be "
            "incomplete");
    }

    void setting(const std::string& s) { settings_.push_back(s); }
    void element(const std::string& s) { elements_.push_back(s); }
    void need3D() { threeD_ = true; }
    bool is3D() const { return threeD_; }

    // Names a fresh datablock and streams the rows into it.
    std::string block(const std::vector<std::string>& rows) {
        const std::string name = "$D" + std::to_string(blocks_++);
        std::string payload = name + " << EOD\n";
        for (const std::string& r : rows) {
            payload += r;
            payload += '\n';
        }
        payload += "EOD";
        cmd(payload);
        return name;
    }

    // Emits the whole figure: settings, then the one plot command.
    //
    // `reset` first, so a figure is fully described by its own calls and
    // nothing leaks in from the last one -- a stale xrange is a genuinely
    // confusing thing to debug. It does not disturb the terminal or the output
    // file, which is what makes it safe to do this inside an animation.
    void render() {
        if (elements_.empty()) {
            clear();
            return;
        }
        cmd("reset");
        for (const std::string& s : settings_) cmd(s);
        std::string c = threeD_ ? "splot " : "plot ";
        for (std::size_t i = 0; i < elements_.size(); i++) {
            if (i) c += ", ";
            c += elements_[i];
        }
        cmd(c);
        flush();
        clear();
    }

    void clear() {
        settings_.clear();
        elements_.clear();
        threeD_ = false;
        blocks_ = 0;
    }

    void size(int w, int h) { w_ = w; h_ = h; }
    int width() const { return w_; }
    int height() const { return h_; }
    void echo(bool on) { echo_ = on; }

  private:
    Session() = default;
    ~Session() {
        if (in_) {
            std::fflush(in_);
            std::fclose(in_);  // gnuplot sees EOF on stdin and exits
        }
        if (out_) std::fclose(out_);
#if !defined(_WIN32)
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);  // reap it rather than leave a zombie
        }
#endif
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    static constexpr const char* kSentinel = "__MATRIXCPP_GP_SYNC__";

    // popen() on POSIX hands back a valid FILE* even when the command does not
    // exist -- the shell it forks is what fails, on its own stderr, and every
    // write after that goes nowhere. Probing first turns that silent nothing
    // into a message that says what to install.
    void ensure() {
        if (in_) return;
        static const bool present = (std::system(probeCmd()) == 0);
        if (!present)
            throw std::runtime_error(
                "gnuplot was not found on PATH. This header needs it and nothing else --\n"
                "  Debian/Ubuntu:  sudo apt install gnuplot\n"
                "  Fedora:         sudo dnf install gnuplot\n"
                "  macOS:          brew install gnuplot\n"
                "Or use the plotting/ package instead, which draws through Julia.");
        spawn();
    }

#if defined(_WIN32)
    // No sync() on Windows: _popen gives one direction only, so there is
    // nothing to read the sentinel back on. save() is asynchronous there, and
    // the file is complete by the time the program exits.
    void spawn() {
        in_ = _popen("gnuplot", "w");
        if (!in_) throw std::runtime_error("gnuplot: could not open a pipe to the process");
    }
#else
    // Two pipes and a fork rather than popen(), because popen() opens ONE
    // direction and sync() needs to hear gnuplot answer. The child execs
    // immediately, which is what makes this safe to do from a program that has
    // OpenMP threads running.
    void spawn() {
        int toChild[2], fromChild[2];
        if (::pipe(toChild) != 0) throw std::runtime_error("gnuplot: could not create a pipe");
        if (::pipe(fromChild) != 0) {
            ::close(toChild[0]);
            ::close(toChild[1]);
            throw std::runtime_error("gnuplot: could not create a pipe");
        }
        pid_ = ::fork();
        if (pid_ < 0) throw std::runtime_error("gnuplot: could not fork the plotting process");
        if (pid_ == 0) {
            ::dup2(toChild[0], STDIN_FILENO);
            ::dup2(fromChild[1], STDOUT_FILENO);
            ::close(toChild[0]);
            ::close(toChild[1]);
            ::close(fromChild[0]);
            ::close(fromChild[1]);
            ::execlp("gnuplot", "gnuplot", (char*)nullptr);
            ::_exit(127);  // only reached if exec failed
        }
        ::close(toChild[0]);
        ::close(fromChild[1]);
        in_ = ::fdopen(toChild[1], "w");
        out_ = ::fdopen(fromChild[0], "r");
        if (!in_ || !out_) throw std::runtime_error("gnuplot: could not attach to the process");
    }
    ::pid_t pid_ = -1;
#endif

    std::FILE* in_ = nullptr;
    std::FILE* out_ = nullptr;
    std::vector<std::string> settings_;
    std::vector<std::string> elements_;
    int blocks_ = 0;
    bool threeD_ = false;
    bool echo_ = false;
    int w_ = 800, h_ = 600;
};

inline Session& S() { return Session::current(); }

// A label of "" means "keep this out of the key", which gnuplot spells
// `notitle` rather than with an empty string.
inline std::string titleOf(const std::string& label) {
    return label.empty() ? "notitle" : ("title " + q(label));
}

template <typename T>
std::vector<double> flat(const mcpu::Matrix<T>& m) {
    std::vector<double> v;
    v.reserve((std::size_t)(m.rows() * m.cols()));
    for (long i = 0; i < m.rows(); i++)
        for (long j = 0; j < m.cols(); j++) v.push_back(re(m(i, j)));
    return v;
}

inline std::vector<std::string> pairRows(const std::vector<double>& x,
                                         const std::vector<double>& y) {
    if (x.size() != y.size())
        throw std::invalid_argument("gnuplot: x and y have different lengths (" +
                                    std::to_string(x.size()) + " vs " + std::to_string(y.size()) +
                                    ")");
    std::vector<std::string> rows;
    rows.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); i++) rows.push_back(num(x[i]) + " " + num(y[i]));
    return rows;
}

inline std::vector<double> ramp(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; i++) v[i] = double(i + 1);
    return v;
}

// Every 2-D series in the header funnels through here, so adding one is a line
// rather than another copy of this.
inline void xy(const std::vector<double>& x, const std::vector<double>& y,
               const std::string& label, const std::string& style) {
    const std::string b = S().block(pairRows(x, y));
    S().element(b + " using 1:2 " + style + " " + titleOf(label));
}

}  // namespace detail

// ── 2-D series ──────────────────────────────────────────────────────────────

// Against the index 1..n, the way MATLAB's plot(y) does.
template <typename T>
void plot(const mcpu::Matrix<T>& y, const std::string& label = "") {
    const auto v = detail::flat(y);
    detail::xy(detail::ramp(v.size()), v, label, "with lines");
}

template <typename T, typename U>
void plot(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label = "") {
    detail::xy(detail::flat(x), detail::flat(y), label, "with lines");
}

template <typename T, typename U>
void scatter(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label = "") {
    detail::xy(detail::flat(x), detail::flat(y), label, "with points pt 7 ps 0.8");
}

template <typename T>
void scatter(const mcpu::Matrix<T>& y, const std::string& label = "") {
    const auto v = detail::flat(y);
    detail::xy(detail::ramp(v.size()), v, label, "with points pt 7 ps 0.8");
}

template <typename T, typename U>
void steps(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label = "") {
    detail::xy(detail::flat(x), detail::flat(y), label, "with steps");
}

template <typename T, typename U>
void stem(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label = "") {
    detail::xy(detail::flat(x), detail::flat(y), label, "with impulses");
}

template <typename T, typename U>
void bar(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label = "") {
    detail::S().setting("set boxwidth 0.8 relative");
    detail::S().setting("set style fill solid 0.5");
    detail::xy(detail::flat(x), detail::flat(y), label, "with boxes");
}

// The bins are counted HERE rather than with gnuplot's `smooth freq`, because
// that needs a binning function defined in the script and gets the edges
// subtly wrong at the top of the range. Counting in C++ is a dozen lines and
// the answer is exactly the one basic/ would give.
template <typename T>
void hist(const mcpu::Matrix<T>& v, int bins = 20, const std::string& label = "") {
    const auto d = detail::flat(v);
    if (d.empty()) throw std::invalid_argument("gnuplot: hist() of an empty matrix");
    if (bins < 1) throw std::invalid_argument("gnuplot: hist() needs at least one bin");
    double lo = d[0], hi = d[0];
    for (double z : d) {
        if (z < lo) lo = z;
        if (z > hi) hi = z;
    }
    if (hi == lo) hi = lo + 1.0;  // a constant sample still deserves one bar
    const double w = (hi - lo) / bins;
    std::vector<double> centre((std::size_t)bins), count((std::size_t)bins, 0.0);
    for (int b = 0; b < bins; b++) centre[(std::size_t)b] = lo + w * (b + 0.5);
    for (double z : d) {
        int b = int((z - lo) / w);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;  // the maximum lands in the last bin, not past it
        count[(std::size_t)b] += 1.0;
    }
    detail::S().setting("set boxwidth " + detail::num(w * 0.9));
    detail::S().setting("set style fill solid 0.5");
    detail::xy(centre, count, label, "with boxes");
}

// The escape hatch: `style` is handed to gnuplot verbatim, so anything the
// `plot` command understands is reachable without this header having to wrap
// it -- "with linespoints lw 2 lc rgb '#cc0000' dt 2", for instance.
template <typename T, typename U>
void series(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const std::string& label,
            const std::string& style) {
    detail::xy(detail::flat(x), detail::flat(y), label, style);
}

// ── Matrix views ────────────────────────────────────────────────────────────
//
// These use gnuplot's `matrix` reader, which takes a plain grid of z values and
// uses the COLUMN NUMBER as x and the ROW NUMBER as y. There is no way to give
// it coordinate axes without switching to a different data format, so a
// heatmap here is indexed like the matrix it came from.

namespace detail {
template <typename T>
std::string grid(const mcpu::Matrix<T>& A) {
    std::vector<std::string> rows;
    rows.reserve((std::size_t)A.rows());
    for (long i = 0; i < A.rows(); i++) {
        std::string r;
        for (long j = 0; j < A.cols(); j++) {
            if (j) r += ' ';
            r += num(re(A(i, j)));
        }
        rows.push_back(r);
    }
    return S().block(rows);
}
}  // namespace detail

template <typename T>
void heatmap(const mcpu::Matrix<T>& A, const std::string& label = "") {
    const std::string b = detail::grid(A);
    detail::S().setting("set view map");
    detail::S().need3D();
    detail::S().element(b + " matrix with image " + detail::titleOf(label));
}

template <typename T>
void surface(const mcpu::Matrix<T>& A, const std::string& label = "") {
    const std::string b = detail::grid(A);
    detail::S().setting("set pm3d");
    detail::S().need3D();
    detail::S().element(b + " matrix with pm3d " + detail::titleOf(label));
}

template <typename T>
void contour(const mcpu::Matrix<T>& A, const std::string& label = "") {
    const std::string b = detail::grid(A);
    detail::S().setting("set contour base");
    detail::S().setting("unset surface");
    detail::S().setting("set view map");
    detail::S().need3D();
    detail::S().element(b + " matrix with lines " + detail::titleOf(label));
}

// ── 3-D series ──────────────────────────────────────────────────────────────

namespace detail {
inline void xyz(const std::vector<double>& x, const std::vector<double>& y,
                const std::vector<double>& z, const std::string& label,
                const std::string& style) {
    if (x.size() != y.size() || x.size() != z.size())
        throw std::invalid_argument("gnuplot: x, y and z have different lengths (" +
                                    std::to_string(x.size()) + ", " + std::to_string(y.size()) +
                                    ", " + std::to_string(z.size()) + ")");
    std::vector<std::string> rows;
    rows.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); i++)
        rows.push_back(num(x[i]) + " " + num(y[i]) + " " + num(z[i]));
    const std::string b = S().block(rows);
    S().need3D();
    S().element(b + " using 1:2:3 " + style + " " + titleOf(label));
}
}  // namespace detail

template <typename T, typename U, typename V>
void plot3(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const mcpu::Matrix<V>& z,
           const std::string& label = "") {
    detail::xyz(detail::flat(x), detail::flat(y), detail::flat(z), label, "with lines");
}

template <typename T, typename U, typename V>
void scatter3(const mcpu::Matrix<T>& x, const mcpu::Matrix<U>& y, const mcpu::Matrix<V>& z,
              const std::string& label = "") {
    detail::xyz(detail::flat(x), detail::flat(y), detail::flat(z), label,
                "with points pt 7 ps 0.8");
}

// ── Axes and appearance ─────────────────────────────────────────────────────

inline void title(const std::string& t) { detail::S().setting("set title " + detail::q(t)); }
inline void xlabel(const std::string& t) { detail::S().setting("set xlabel " + detail::q(t)); }
inline void ylabel(const std::string& t) { detail::S().setting("set ylabel " + detail::q(t)); }
inline void zlabel(const std::string& t) { detail::S().setting("set zlabel " + detail::q(t)); }

inline void xlim(double lo, double hi) {
    detail::S().setting("set xrange [" + detail::num(lo) + ":" + detail::num(hi) + "]");
}
inline void ylim(double lo, double hi) {
    detail::S().setting("set yrange [" + detail::num(lo) + ":" + detail::num(hi) + "]");
}
inline void zlim(double lo, double hi) {
    detail::S().setting("set zrange [" + detail::num(lo) + ":" + detail::num(hi) + "]");
}

// Camera for 3-D plots: rotation about x then z, in degrees.
inline void view(double azimuth, double elevation) {
    detail::S().setting("set view " + detail::num(elevation) + ", " + detail::num(azimuth));
}

inline void legend(bool on = true, const std::string& where = "top right") {
    detail::S().setting(on ? ("set key " + where) : "unset key");
}
inline void grid(bool on = true) { detail::S().setting(on ? "set grid" : "unset grid"); }

// axis is any of "x", "y", "z", or several at once as "xy".
inline void logscale(const std::string& axis = "y", double base = 10.0) {
    detail::S().setting("set logscale " + axis + " " + detail::num(base));
}

// In pixels. Vector terminals measure in inches instead, and save() converts
// at 96 dpi so that one call means the same shape everywhere.
inline void size(int w, int h) { detail::S().size(w, h); }

// ── Output ──────────────────────────────────────────────────────────────────

// Throws away anything drawn but not yet rendered. Rendering already clears,
// so this is only needed to abandon a half-built figure.
inline void figure() { detail::S().clear(); }

// Verbatim gnuplot, for the settings this header does not wrap. It joins the
// same queue as the rest, so it is applied in order with them.
inline void raw(const std::string& command) { detail::S().setting(command); }

inline void term(const std::string& spec) { detail::S().cmd("set terminal " + spec); }

// Mirrors every command to stderr. The first thing to reach for when a plot
// comes out wrong -- what gnuplot was actually told is usually the answer.
inline void echo(bool on = true) { detail::S().echo(on); }

namespace detail {

// Extension to terminal. The size unit is the wart being papered over here:
// the cairo vector terminals measure in inches while the raster ones measure
// in pixels, so the same size() call would otherwise give a figure 96 times
// too large in a PDF.
inline std::string terminalFor(const std::string& path, int w, int h) {
    const std::size_t dot = path.rfind('.');
    const std::string ext = (dot == std::string::npos) ? "" : path.substr(dot + 1);
    const std::string px = std::to_string(w) + "," + std::to_string(h);
    const std::string in = num(w / 96.0) + "in," + num(h / 96.0) + "in";

    if (ext == "png") return "pngcairo size " + px;
    if (ext == "gif") return "gif size " + px;
    if (ext == "jpg" || ext == "jpeg") return "jpeg size " + px;
    if (ext == "webp") return "webp size " + px;
    if (ext == "svg") return "svg size " + px;
    if (ext == "html") return "canvas size " + px;      // mousing, but not plotly
    if (ext == "pdf") return "pdfcairo size " + in;
    if (ext == "eps") return "epscairo size " + in;
    if (ext == "tex") return "cairolatex pdf size " + in;
    throw std::invalid_argument(
        "gnuplot: nothing is known about the extension \"" + ext +
        "\". save() understands png, gif, jpg, webp, svg, html, pdf, eps and tex.");
}

// The interactive terminal is set only when a window is actually wanted. Doing
// it when the pipe opens would make every headless run complain about a
// display it was never going to use.
inline void interactiveTerm() {
    // `persist` is what keeps the window on screen after the program exits;
    // without it the window dies with the pipe and a short program appears to
    // have drawn nothing at all.
    S().cmd("set terminal qt persist size " + std::to_string(S().width()) + "," +
            std::to_string(S().height()));
}

}  // namespace detail

// Blocks until gnuplot has caught up with everything sent so far. save() and
// gif_end() already do this, so it is only needed after raw() commands that
// produce a file of their own.
inline void sync() { detail::S().sync(); }

// Writes the figure to a file, with the terminal chosen from the extension.
//
// This RETURNS ONLY ONCE THE FILE IS ON DISK. Everything else in this header
// is fire-and-forget down the pipe, which is what makes draw() cheap, but a
// save() that returned early would hand back a path to a file that was not
// there yet. Waiting costs about 20 ms against the 1050 ms a plt::save()
// through Julia takes, so this is still the fast one by roughly 50x.
inline void save(const std::string& path) {
    detail::S().cmd("set terminal " +
                    detail::terminalFor(path, detail::S().width(), detail::S().height()));
    detail::S().cmd("set output " + detail::q(path));
    detail::S().render();
    detail::S().cmd("unset output");
    detail::S().sync();
}

// Opens a window that survives the program exiting. Needs a display.
inline void show() {
    detail::interactiveTerm();
    detail::S().render();
}

// Draws into the SAME window, replacing what was there. This is the one that
// makes a simulation loop into a live view; it costs a write to a pipe, not a
// process.
inline void draw() {
    static bool armed = false;
    if (!armed) {
        detail::interactiveTerm();
        armed = true;
    }
    detail::S().render();
}

// ── Animation ───────────────────────────────────────────────────────────────
//
// gnuplot encodes an animated GIF as the frames arrive, so unlike plt::gif()
// nothing is held in memory and the terminal has to be chosen UP FRONT. Hence
// three calls rather than two:
//
//     gp::gif_begin("out/run.gif", 5);
//     for (...) { gp::scatter(x, y); gp::frame(); }
//     gp::gif_end();
//
// `delay` is in hundredths of a second between frames, which is the unit the
// GIF format itself uses.
inline void gif_begin(const std::string& path, int delay = 5) {
    detail::S().cmd("set terminal gif animate delay " + std::to_string(delay) + " size " +
                    std::to_string(detail::S().width()) + "," +
                    std::to_string(detail::S().height()));
    detail::S().cmd("set output " + detail::q(path));
}

// Deliberately does NOT wait: frames queue up in the pipe while the next one
// is being computed, which is the whole point of streaming them. gif_end()
// waits once, at the end, for all of them.
inline void frame() { detail::S().render(); }

// Closes the GIF and waits for it, on the same reasoning as save().
inline void gif_end() {
    detail::S().cmd("unset output");
    detail::S().sync();
}

}  // namespace gp
