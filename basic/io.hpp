#pragma once

// ==========================================================================
//  Output formatting
// ==========================================================================
//
// One format engine, shared by Matrix and Tensor. It knows nothing about
// either — it takes a size and a callable that returns element (i,j), which is
// why the same eight formats work for both without either class knowing how
// any of them are spelled.
//
// The C++ idiom is that anything printable takes a std::ostream, so print()
// does: no argument means std::cout, and any other stream — a file, a
// std::ostringstream, a socket — works the same way.
//
//     A.print();                          // to the terminal, aligned
//     A.print(std::cerr);                 // to any stream
//     A.print(matio::Fmt::CSV);           // comma-separated, to the terminal
//     A.print(file, matio::Fmt::CSV);     // ... to a file
//     A.save("data.csv");                 // format chosen from the extension
//     std::string s = A.str(matio::Fmt::Markdown);
//
// Reading these back in is deliberately not here yet — the formats were chosen
// so that CSV, TSV and Plain are trivially parseable when that lands.
//
// Part of the Basic Matrix Package — include <basic/MatrixCpp.hpp> for all of
// it, or this header alone if that is genuinely all you need.

#include "traits.hpp"

#include <fstream>
#include <functional>
#include <ostream>

namespace matio {

// Auto means "decide from context": the terminal gets Pretty, a file gets
// whatever its extension says. Everything else is explicit.
enum class Fmt {
    Auto,
    Pretty,    // aligned and bracketed — for a human reading a terminal
    Plain,     // whitespace separated, nothing else. Trivially parseable.
    CSV,       // comma separated
    TSV,       // tab separated
    Markdown,  // | a | b | with the separator row, pastes into a document
    MATLAB,    // [1, 2; 3, 4]
    NumPy,     // np.array([[1, 2], [3, 4]])
    JSON       // [[1, 2], [3, 4]]
};

struct Opts {
    Fmt fmt = Fmt::Auto;
    int precision = 6;
    bool scientific = false;  // 1.234568e+03 rather than 1234.567890
    bool header = false;      // CSV/TSV/Markdown: emit col1,col2,... first
    std::string name;         // MATLAB/NumPy: emit "name = ..." instead of bare

    Opts() = default;
    Opts(Fmt f) : fmt(f) {}                              // implicit, so
    Opts(Fmt f, int p) : fmt(f), precision(p) {}         // print(Fmt::CSV) works
};

// ── One value, formatted ────────────────────────────────────────────────────
// Complex comes out as 3+4i, NOT the (3,4) that std::complex streams by
// default: that spelling contains a comma, which would silently corrupt every
// CSV it appeared in.
template <class T>
inline std::string value(const T& x, const Opts& o) {
    auto setup = [&](std::ostringstream& ss) {
        if constexpr (is_float_like<T>::value) {
            if (o.scientific) ss << std::scientific << std::setprecision(o.precision);
            else              ss << std::fixed << std::setprecision(o.precision);
        }
    };
    std::ostringstream ss;
    setup(ss);
    if constexpr (is_complex<T>::value) {
        std::ostringstream im;
        setup(im);
        const double re = double(std::real(x)), imag = double(std::imag(x));
        im << std::abs(imag);
        ss << re << (imag < 0.0 ? '-' : '+') << im.str() << 'i';
    } else {
        ss << x;
    }
    return ss.str();
}

// ── Format from a filename ──────────────────────────────────────────────────
inline Fmt fromExtension(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return Fmt::Plain;
    std::string e = path.substr(dot + 1);
    for (char& c : e) c = char(std::tolower((unsigned char)c));
    if (e == "csv") return Fmt::CSV;
    if (e == "tsv" || e == "tab") return Fmt::TSV;
    if (e == "md" || e == "markdown") return Fmt::Markdown;
    if (e == "json") return Fmt::JSON;
    if (e == "m") return Fmt::MATLAB;
    if (e == "py") return Fmt::NumPy;
    if (e == "txt" || e == "dat") return Fmt::Plain;
    return Fmt::Plain;
}

namespace detail {
inline const char* delim(Fmt f) {
    switch (f) {
        case Fmt::CSV:  return ",";
        case Fmt::TSV:  return "\t";
        default:        return " ";
    }
}
inline std::string colName(long j) { return "col" + std::to_string(j + 1); }
}  // namespace detail

// ── A rows x cols table ─────────────────────────────────────────────────────
// `get(i, j)` returns the ELEMENT, not a string — formatting is this function's
// job, so every caller gets the same treatment of precision and complex.
template <class Get>
void table(std::ostream& os, long rows, long cols, Get get, const Opts& oIn) {
    Opts o = oIn;
    if (o.fmt == Fmt::Auto) o.fmt = Fmt::Pretty;

    // Format every cell up front: Pretty and Markdown need the widest one, and
    // it costs one pass either way.
    std::vector<std::string> cell((std::size_t)(rows * cols));
    std::size_t w = 0;
    for (long i = 0; i < rows; i++)
        for (long j = 0; j < cols; j++) {
            std::string s = value(get(i, j), o);
            w = std::max(w, s.size());
            cell[(std::size_t)(i * cols + j)] = std::move(s);
        }
    auto at = [&](long i, long j) -> const std::string& {
        return cell[(std::size_t)(i * cols + j)];
    };

    switch (o.fmt) {
        case Fmt::Pretty: {
            for (long i = 0; i < rows; i++) {
                os << "[ ";
                for (long j = 0; j < cols; j++) {
                    os << std::setw((int)w) << at(i, j);
                    if (j + 1 < cols) os << "  ";
                }
                os << " ]";
                if (i + 1 < rows) os << '\n';
            }
            break;
        }
        case Fmt::Plain:
        case Fmt::CSV:
        case Fmt::TSV: {
            const char* d = detail::delim(o.fmt);
            if (o.header) {
                for (long j = 0; j < cols; j++) {
                    os << detail::colName(j);
                    if (j + 1 < cols) os << d;
                }
                os << '\n';
            }
            for (long i = 0; i < rows; i++) {
                for (long j = 0; j < cols; j++) {
                    os << at(i, j);
                    if (j + 1 < cols) os << d;
                }
                if (i + 1 < rows) os << '\n';
            }
            break;
        }
        case Fmt::Markdown: {
            // Markdown REQUIRES a header row, so this one always names the
            // columns — a blank header row is legal and looks broken.
            os << "|";
            for (long j = 0; j < cols; j++) os << ' ' << detail::colName(j) << " |";
            os << "\n|";
            for (long j = 0; j < cols; j++) os << " --- |";
            for (long i = 0; i < rows; i++) {
                os << "\n|";
                for (long j = 0; j < cols; j++) os << ' ' << at(i, j) << " |";
            }
            break;
        }
        case Fmt::MATLAB: {
            if (!o.name.empty()) os << o.name << " = ";
            os << '[';
            for (long i = 0; i < rows; i++) {
                for (long j = 0; j < cols; j++) {
                    os << at(i, j);
                    if (j + 1 < cols) os << ", ";
                }
                if (i + 1 < rows) os << "; ";
            }
            os << ']';
            if (!o.name.empty()) os << ';';
            break;
        }
        case Fmt::NumPy:
        case Fmt::JSON: {
            const bool np = (o.fmt == Fmt::NumPy);
            if (np && !o.name.empty()) os << o.name << " = ";
            if (np) os << "np.array(";
            os << '[';
            for (long i = 0; i < rows; i++) {
                os << '[';
                for (long j = 0; j < cols; j++) {
                    os << at(i, j);
                    if (j + 1 < cols) os << ", ";
                }
                os << ']';
                if (i + 1 < rows) os << ", ";
            }
            os << ']';
            if (np) os << ')';
            break;
        }
        case Fmt::Auto: break;  // resolved above
    }
}

// Opens the file, picks the format from its extension when none was asked for,
// and throws rather than failing quietly — a save that silently did nothing is
// the worst outcome here.
template <class Writer>
void saveWith(const std::string& path, Opts o, Writer w) {
    if (o.fmt == Fmt::Auto) o.fmt = fromExtension(path);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("save: could not open '" + path + "' for writing");
    w(f, o);
    f << '\n';
    if (!f)
        throw std::runtime_error("save: write to '" + path + "' failed");
}

}  // namespace matio
