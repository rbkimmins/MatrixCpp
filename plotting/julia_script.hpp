#pragma once

// ==========================================================================
//  Talking to Julia
// ==========================================================================
//
// A figure is accumulated as a list of Julia statements plus the arrays they
// refer to. At save() or show() the whole thing is written out as one script,
// with the arrays as raw Float64 files beside it, and handed to `julia` as a
// subprocess. Nothing above this header writes any Julia.
//
// WHY A SUBPROCESS RATHER THAN EMBEDDING
//
// Embedding was tried first — jl_init, jl_eval_string, GC-rooted argument
// marshalling, the lot — and it is in the git history of this file's
// predecessor. It crashed, reliably but unpredictably, inside
// FreeType2_jll.__init__ during `using Plots`: a segfault in JLLWrappers'
// unique! over the library path list, deep in Julia's package loader and
// nowhere near anything this code does. It was verified NOT to be the
// marshalling — arrays built with jl_alloc_array_1d round-tripped correctly,
// and a bare jl_init + module eval ran 10/10 — but whether a given binary
// crashed depended on its size and on ASLR, which is the signature of a
// layout-sensitive fault rather than a logical one. Plain `julia` is 100%
// reliable on the same machine and the same script.
//
// So: one Julia process per figure. It costs about a second of startup while
// Plots loads, and buys a plotting package that cannot corrupt the caller's
// heap, needs no -I/-L/-ljulia flags, and works whether or not the Julia
// development headers are installed.
//
// Part of the Plotting Package — include <plotting/MatrixPlot.hpp>.

#include "../basic/MatrixCpp.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>   // getpid, for a per-process scratch directory
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jlx {

// ── The figure under construction ───────────────────────────────────────────
class Script {
  public:
    static Script& current() {
        static Script s;
        return s;
    }

    void reset() {
        lines_.clear();
        blobs_.clear();
        next_ = 0;
    }

    void add(const std::string& stmt) { lines_.push_back(stmt); }
    bool empty() const { return lines_.empty(); }

    // Registers an array and returns the Julia name it will be bound to. The
    // data goes out as raw Float64 rather than as text in the script: exact,
    // compact, and it keeps a 10 000-point series from becoming a megabyte of
    // source.
    std::string bind(const std::vector<double>& v, long rows = 0, long cols = 0) {
        const std::string name = "_d" + std::to_string(next_++);
        blobs_.push_back({name, v, rows, cols});
        return name;
    }

    // Emits the script (and its data files) into `dir` and returns the path.
    std::string emit(const std::string& dir) const {
        for (const Blob& b : blobs_) {
            const std::string path = dir + "/" + b.name + ".bin";
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("plot: cannot write " + path);
            f.write(reinterpret_cast<const char*>(b.data.data()),
                    (std::streamsize)(b.data.size() * sizeof(double)));
        }
        const std::string path = dir + "/figure.jl";
        std::ofstream f(path);
        if (!f) throw std::runtime_error("plot: cannot write " + path);
        f << "using Plots\n";
        for (const Blob& b : blobs_) {
            f << b.name << " = Vector{Float64}(undef, " << b.data.size() << ")\n"
              << "read!(raw\"" << dir << "/" << b.name << ".bin\", " << b.name << ")\n";
            if (b.rows > 0)   // a matrix: the blob was written column-major
                f << b.name << " = reshape(" << b.name << ", " << b.rows << ", " << b.cols
                  << ")\n";
        }
        f << "_cur = nothing\n";
        for (const std::string& l : lines_) f << l << '\n';
        return path;
    }

    std::string text(const std::string& dir = "<dir>") const {
        std::ostringstream ss;
        ss << "using Plots\n";
        for (const Blob& b : blobs_)
            ss << b.name << " = <" << b.data.size() << " Float64 from " << dir << "/" << b.name
               << ".bin>\n";
        ss << "_cur = nothing\n";
        for (const std::string& l : lines_) ss << l << '\n';
        return ss.str();
    }

  private:
    struct Blob {
        std::string name;
        std::vector<double> data;
        long rows, cols;
    };
    std::vector<std::string> lines_;
    std::vector<Blob> blobs_;
    long next_ = 0;
};

// ── Running it ──────────────────────────────────────────────────────────────
namespace detail {

inline std::string tempDir() {
    std::string base = "/tmp/matrixcpp_plot_" + std::to_string(::getpid());
    std::string cmd = "mkdir -p '" + base + "'";
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("plot: cannot create " + base);
    return base;
}

// Escapes a path for a Julia raw string literal. Julia's raw"..." handles
// backslashes, so only a stray quote needs care.
inline std::string jstr(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

}  // namespace detail

// Writes the script out, runs julia on it, and throws with Julia's own error
// text if it fails. Everything Julia prints on stderr is forwarded, so a
// mistyped attribute name reads the way it would at the REPL.
inline void run(const std::string& what) {
    Script& s = Script::current();
    if (s.empty())
        throw std::runtime_error("plot: nothing to " + what + " — no plot has been drawn yet");
    const std::string dir = detail::tempDir();
    const std::string script = s.emit(dir);
    const std::string log = dir + "/stderr.txt";
    const std::string cmd = "julia --startup-file=no '" + script + "' 2> '" + log + "'";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::ifstream f(log);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string msg = ss.str();
        if (msg.size() > 2000) msg = msg.substr(0, 2000) + "\n  ...";
        throw std::runtime_error("plot: julia failed during " + what + "\n" + msg +
                                 "\n  script kept at " + script);
    }
    // Best effort; a leftover scratch directory is not worth failing over.
    const int cleaned = std::system(("rm -rf '" + dir + "'").c_str());
    (void)cleaned;
}

}  // namespace jlx
