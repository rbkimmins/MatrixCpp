#pragma once

// ==========================================================================
//  Plotting a device matrix
// ==========================================================================
//
//     #include "shared/plot_gpu.hpp"     // instead of plotting/MatrixPlot.hpp
//
//     mgpu::Matrix<double> dx = ..., dy = ...;
//     plt::scatter(dx, dy);              // just works
//
// WHY THIS IS A SEPARATE HEADER, and why the overloads are this thin.
//
// Plotting is HOST work: it walks elements one at a time, formats numbers, and
// hands a few kilobytes to Julia. There is nothing in it to parallelise, so
// there is no such thing as a "GPU plot" -- only a plot of data that happens
// to be on the GPU. The right implementation is therefore to download ONCE and
// reuse every line of the existing code, which is what each overload below is.
//
// The naive alternative is much worse and worth naming: plt::detail::flat()
// reads a matrix through operator(), and on a device matrix each of those is a
// separate PCIe round trip. Letting the existing templates bind to
// mgpu::Matrix would turn one transfer into rows*cols of them -- microseconds
// each, so a 512x512 heatmap would take minutes instead of a millisecond. The
// overloads exist to make sure that never happens by accident.
//
// It is a separate header so that plotting/ does not depend on gpu/: a program
// that only draws CPU data should not have to link CUDA.

#include "../gpu/MatrixGpu.hpp"
#include "../plotting/MatrixPlot.hpp"

namespace plt {

// One download, then the ordinary host path. Every overload here is exactly
// that shape -- if one of them grows a second .cpu() call, it is a bug.

template <typename T>
void plot(const mgpu::Matrix<T>& y, const std::string& label = "") { plot(y.cpu(), label); }
template <typename Tx, typename Ty>
void plot(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const std::string& label = "") {
    plot(x.cpu(), y.cpu(), label);
}
template <typename Tx, typename Ty>
void plot(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const Style& st) {
    plot(x.cpu(), y.cpu(), st);
}

template <typename T>
void scatter(const mgpu::Matrix<T>& y, const std::string& label = "") { scatter(y.cpu(), label); }
template <typename Tx, typename Ty>
void scatter(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const std::string& label = "") {
    scatter(x.cpu(), y.cpu(), label);
}
template <typename Tx, typename Ty>
void scatter(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const Style& st) {
    scatter(x.cpu(), y.cpu(), st);
}

template <typename Tx, typename Ty, typename Tz>
void plot3(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const mgpu::Matrix<Tz>& z,
           const std::string& label = "") {
    plot3(x.cpu(), y.cpu(), z.cpu(), label);
}
template <typename Tx, typename Ty, typename Tz>
void scatter3(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const mgpu::Matrix<Tz>& z,
              const std::string& label = "") {
    scatter3(x.cpu(), y.cpu(), z.cpu(), label);
}
template <typename Tx, typename Ty, typename Tz>
void scatter3(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const mgpu::Matrix<Tz>& z,
              const Style& st) {
    scatter3(x.cpu(), y.cpu(), z.cpu(), st);
}

template <typename Tx, typename Ty>
void bar(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const std::string& label = "") {
    bar(x.cpu(), y.cpu(), label);
}
template <typename Tx, typename Ty>
void stem(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const std::string& label = "") {
    stem(x.cpu(), y.cpu(), label);
}
template <typename Tx, typename Ty>
void area(const mgpu::Matrix<Tx>& x, const mgpu::Matrix<Ty>& y, const std::string& label = "") {
    area(x.cpu(), y.cpu(), label);
}

template <typename T>
void heatmap(const mgpu::Matrix<T>& A) { heatmap(A.cpu()); }
template <typename T>
void surface(const mgpu::Matrix<T>& A) { surface(A.cpu()); }
template <typename T>
void contour(const mgpu::Matrix<T>& A) { contour(A.cpu()); }
template <typename T>
void contourf(const mgpu::Matrix<T>& A) { contourf(A.cpu()); }
template <typename T>
void wireframe(const mgpu::Matrix<T>& A) { wireframe(A.cpu()); }
template <typename T>
void image(const mgpu::Matrix<T>& A) { image(A.cpu()); }
template <typename T>
void spy(const mgpu::Matrix<T>& A, double tol = 0.0) { spy(A.cpu(), tol); }
template <typename T>
void hist(const mgpu::Matrix<T>& v, int bins = 20, const std::string& label = "") {
    hist(v.cpu(), bins, label);
}

}  // namespace plt
