#pragma once

// ==========================================================================
//  mgpu::Matrix — a device-resident matrix
// ==========================================================================
//
// The companion to Matrix<T>, the way Bandicoot is to Armadillo or CuPy is to
// NumPy: same spelling, different address space.
//
//     mcpu::Matrix<double> A(2048, 2048);   A.set_Ran_values(0.0, 1.0);
//
//     mgpu::Matrix<double> dA = mgpu::upload(A);   // one crossing of the bus
//     mgpu::Matrix<double> dC = dA * dA + dA;      // stays on the device
//     mcpu::Matrix<double> C  = dC.cpu();          // one crossing back
//
// The two spell the same, one namespace apart, which is the whole point of the
// split: mcpu::Matrix and mgpu::Matrix are the C++ rendering of numpy and
// cupy. `namespace np = mcpu; namespace cp = mgpu;` completes the analogy.
//
// THE ONE DESIGN RULE, and the reason the class exists at all:
//
//   Every operation takes device memory and returns device memory. Nothing
//   here touches the host except the explicit crossings — gpu()/cpu() — and
//   the scalar reductions, which have nowhere else to put their answer.
//
// That is not a stylistic preference, it is the entire performance argument.
// On this hardware (GeForce, fp64 throttled to 1/64) a double-precision GEMM
// runs about 1.3-1.9x the tuned CPU one, while PCIe moves 12-29 GB/s. A
// library that copied host->device->host per operation would lose to the CPU
// on every single call, no matter whose kernels ran in the middle. Keeping
// results resident is what turns a sequence of ops into a win: the bus is paid
// once for the sequence instead of once per operation.
//
// WHERE THE WORK ACTUALLY HAPPENS
//
//   cuBLAS     GEMM, transpose, scaled sums
//   cuSOLVER   LU, Cholesky, QR, SVD, symmetric and general eigenproblems
//   cuRAND     uniform and normal streams
//   us         element-wise ops and reductions, because nobody ships those
//
// We deliberately do not hand-write a GEMM. cuBLAS's is NVIDIA's own, tuned
// per architecture, and beating it is not a side quest — it is the whole job
// of a team. CUTLASS was considered and rejected for this layer: it covers
// only GEMM-shaped work, leaves every factorisation uncovered, and its
// advantage lives on Tensor Cores, which do not exist for fp64 on consumer
// Blackwell. It remains the right answer later for a FUSED float path.
//
// PRECISION
//
//   Only float and double instantiate. Complex and integral types are a CPU
//   affair for now; see gpu/README.md for what adding them would cost.

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <cstddef>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../basic/matrix.hpp"
#include "detail/backend.hpp"

namespace mgpu {

    namespace detail {
        // Dependent false, so the static_assert fires only when the function is
        // actually instantiated rather than the moment the class is parsed.
        template <typename>
        inline constexpr bool always_false = false;
    }  // namespace detail

    // ── Device-side storage ────────────────────────────────────────────
    //
    // RAII around cudaMalloc. Move-only on purpose: an accidental copy of a
    // device buffer is a silent multi-gigabyte allocation plus a full-bandwidth
    // copy, and it should have to be spelled out. Matrix's copy constructor
    // spells it out.

    class DeviceBuffer {
      public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(std::size_t bytes)
            : p_(detail::devAlloc(bytes)), bytes_(bytes) {}

        ~DeviceBuffer() { detail::devFree(p_); }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), bytes_(o.bytes_) {
            o.p_ = nullptr;
            o.bytes_ = 0;
        }
        DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
            if (this != &o) {
                detail::devFree(p_);
                p_ = o.p_;
                bytes_ = o.bytes_;
                o.p_ = nullptr;
                o.bytes_ = 0;
            }
            return *this;
        }

        void* get() const { return p_; }
        std::size_t bytes() const { return bytes_; }

      private:
        void* p_ = nullptr;
        std::size_t bytes_ = 0;
    };

    // ── Shared vocabulary, re-exported ─────────────────────────────────
    //
    // These are RE-EXPORTS, not copies: mgpu::all and mcpu::all name the same
    // object, so `using namespace mcpu;` and `using namespace mgpu;` in one
    // file stay unambiguous, and a matrix built on one side can be indexed
    // with the other's tag.
    //
    // WHY THESE AND NOT THE REST. The test is whether the thing has any
    // dependence on where the data lives:
    //
    //   all / all_t   an EMPTY tag struct. No data, no behaviour -- overload
    //                 resolution reads its type and nothing else. There is
    //                 literally nothing about it that could differ per device.
    //   ROW / COL     two bools naming an axis convention.
    //   NormType      an enum of four names.
    //
    // None of those could have a device-specific version even in principle,
    // so a second definition would only be a way to get them out of step.
    //
    // The random-number vocabulary splits, and the split is the interesting
    // one. ran2, setRan and the value-returning set_Ran_values are HOST
    // functions -- a sequential generator producing one number at a time --
    // so they are shared as-is. Filling a MATRIX is not shared: that is
    // Matrix::set_Ran_values, and the GPU one runs cuRAND across every element
    // at once and cannot reproduce a sequential stream's order. Same name,
    // genuinely different implementation, which is why it is a member.

    using mcpu::all;
    using mcpu::all_t;
    using mcpu::COL;
    using mcpu::NormType;
    using mcpu::ROW;

    using mcpu::ran2;
    using mcpu::set_Ran_values;
    using mcpu::setRan;

    // ── Complex traits ─────────────────────────────────────────────────
    //
    // The same split basic/traits.hpp makes, restated here so the GPU headers
    // stay usable without pulling the CPU ones in for anything but Matrix.

    template <class T>
    struct is_complex : std::false_type {};
    template <class R>
    struct is_complex<std::complex<R>> : std::true_type {};
    template <class T>
    inline constexpr bool is_complex_v = is_complex<T>::value;

    // real_t<std::complex<double>> is double; real_t<double> is double. Used
    // wherever a result is necessarily real whatever went in -- a magnitude, a
    // singular value, a Hermitian eigenvalue.
    template <class T>
    struct real_of {
        using type = T;
    };
    template <class R>
    struct real_of<std::complex<R>> {
        using type = R;
    };
    template <class T>
    using real_t = typename real_of<T>::type;

    template <typename datatype>
    class Expr;

    // ── Matrix ──────────────────────────────────────────────────────
    //
    // The template parameter is named `datatype` rather than `T` to match
    // mcpu::Matrix<datatype>, and because T() is the transpose.
    //
    // Unqualified `Matrix` inside this namespace means THIS class. Every
    // reference to the CPU one is spelled mcpu::Matrix, deliberately and
    // without exception -- the two are one character apart at a glance and
    // the compiler will not always tell you which you got.

    template <typename datatype>
    class Matrix {
        static_assert(std::is_same<datatype, float>::value ||
                          std::is_same<datatype, double>::value ||
                          std::is_same<datatype, std::complex<float>>::value ||
                          std::is_same<datatype, std::complex<double>>::value,
                      "mgpu::Matrix supports float, double, complex<float> and complex<double>. "
                      "Integral types stay on the CPU - see gpu/README.md.");

      public:
        static constexpr bool isComplex = is_complex_v<datatype>;
        // The type a necessarily-real result takes: a magnitude, a singular
        // value, a Hermitian eigenvalue. Equal to datatype for real matrices,
        // so one spelling serves both.
        using real_type = real_t<datatype>;

      private:

      public:
        // --- Construction ---

        Matrix() = default;

        // Zero-filled, matching Matrix<datatype>(rows, cols).
        Matrix(long rows, long cols) : rows_(rows), cols_(cols) {
            checkDims(rows, cols);
            buf_ = DeviceBuffer(bytesFor(rows, cols));
            detail::devZero(buf_.get(), buf_.bytes());
        }

        // Uninitialised. Separate tag type for the same reason Matrix has one:
        // skipping the zero-fill is worth a full pass over the buffer, and it
        // must never happen by accident.
        struct uninit_t {};
        Matrix(long rows, long cols, uninit_t) : rows_(rows), cols_(cols) {
            checkDims(rows, cols);
            buf_ = DeviceBuffer(bytesFor(rows, cols));
        }

        // Uploads. The only host->device crossing in the class.
        explicit Matrix(const mcpu::Matrix<datatype>& host)
            : Matrix(host.rows(), host.cols(), uninit_t{}) {
            if (size())
                detail::copyH2D(buf_.get(), hostData(host), buf_.bytes());
        }

        Matrix(const Matrix& o) : Matrix(o.rows_, o.cols_, uninit_t{}) {
            if (size()) detail::copyD2D(buf_.get(), o.buf_.get(), buf_.bytes());
        }
        Matrix& operator=(const Matrix& o) {
            if (this != &o) {
                Matrix tmp(o);
                *this = std::move(tmp);
            }
            return *this;
        }

        Matrix(Matrix&&) noexcept = default;
        Matrix& operator=(Matrix&&) noexcept = default;

        // --- Shape ---

        long rows() const { return rows_; }
        long cols() const { return cols_; }
        std::size_t size() const { return (std::size_t)rows_ * (std::size_t)cols_; }
        std::size_t bytes() const { return buf_.bytes(); }
        bool empty() const { return size() == 0; }

        // DEVICE pointer. Dereferencing it on the host is undefined; it is
        // exposed so callers can hand it to their own kernels or to cuBLAS.
        datatype* data() { return (datatype*)buf_.get(); }
        const datatype* data() const { return (const datatype*)buf_.get(); }

        // ── Scalar element access ──────────────────────────────────────
        //
        // A(i, j) and A[k], read and write, so code written against
        // mcpu::Matrix compiles unchanged after a namespace swap.
        //
        // THESE ARE EXPENSIVE AND THE COST DOES NOT SHRINK. Every one is a
        // separate 8-byte transfer across PCIe, which costs microseconds
        // whatever its size -- a latency, not a bandwidth. Filling a 1000x1000
        // matrix element by element is some seconds of pure round trips
        // against under a millisecond for set_Ran_values. They exist for the
        // handful of scalars a program genuinely needs (a boundary value, one
        // entry of a small vector), not as a way to build data.
        //
        // Build on the host and upload, or use fill/set_Ran_values/linspace and
        // the views, all of which run on the device.
        //
        // Indices may be negative and wrap from the end, which is what the CPU
        // side's documentation promises; anything still out of range throws
        // rather than reading past the buffer.

        class ElementRef {
          public:
            ElementRef(Matrix& m, std::size_t k) : m_(&m), k_(k) {}

            operator datatype() const {
                datatype v{};
                detail::copyD2H(&v, m_->data() + k_, sizeof(datatype));
                return v;
            }
            ElementRef& operator=(datatype v) {
                detail::copyH2D(m_->data() + k_, &v, sizeof(datatype));
                return *this;
            }
            // Element-to-element assignment goes through the host, since there
            // is no cheaper route for one value.
            ElementRef& operator=(const ElementRef& o) { return *this = datatype(o); }

            ElementRef& operator+=(datatype v) { return *this = datatype(*this) + v; }
            ElementRef& operator-=(datatype v) { return *this = datatype(*this) - v; }
            ElementRef& operator*=(datatype v) { return *this = datatype(*this) * v; }
            ElementRef& operator/=(datatype v) { return *this = datatype(*this) / v; }

          private:
            Matrix* m_;
            std::size_t k_;
        };

// A floating-point index TRUNCATES rather than rounds -- A[2.9] means
        // A[2] -- and nothing warns. Refused here for the same reason as on the
        // CPU side, and with the same message, so ml:: code written against one
        // backend behaves identically on the other.
        template <typename U, typename = std::enable_if_t<std::is_floating_point<U>::value>>
        datatype operator[](U) const {
            static_assert(!std::is_floating_point<U>::value,
                          "a floating-point index TRUNCATES: A[2.9] means A[2], not A[3]. "
                          "Round first and say which you meant, or keep the index integral.");
            return datatype{};
        }
        template <typename U, typename V,
                  typename = std::enable_if_t<std::is_floating_point<U>::value ||
                                              std::is_floating_point<V>::value>>
        datatype operator()(U, V) const {
            static_assert(!(std::is_floating_point<U>::value || std::is_floating_point<V>::value),
                          "a floating-point index TRUNCATES: A(1.9, 1.9) means A(1, 1). "
                          "Round first and say which you meant, or keep the indices integral.");
            return datatype{};
        }

                ElementRef operator()(long i, long j) { return ElementRef(*this, flatIndex(i, j)); }
        datatype operator()(long i, long j) const {
            datatype v{};
            detail::copyD2H(&v, data() + flatIndex(i, j), sizeof(datatype));
            return v;
        }
        ElementRef operator[](long k) { return ElementRef(*this, flatIndex(k)); }
        datatype operator[](long k) const {
            datatype v{};
            detail::copyD2H(&v, data() + flatIndex(k), sizeof(datatype));
            return v;
        }

        // ── A 1x1 matrix IS a scalar ───────────────────────────────────
        //
        //     double r = std::sqrt(sum((a - b).pow(2), ROW));
        //
        // sum along a 1 x n row gives a 1 x 1, and this is what lets it be
        // used as the number it is. Same rule as mcpu::Matrix: exactly one
        // element or it throws, because there is no other sensible answer.
        operator datatype() const {
            if (size() != 1)
                throw Error("a Matrix converts to a scalar only when it holds exactly one "
                            "element, but this one is " + std::to_string(rows_) + "x" +
                            std::to_string(cols_) + " (" + std::to_string(size()) +
                            " elements). Index it, or reduce it first - sum(), det(), dot() and "
                            "the other reductions already return scalars.");
            datatype v{};
            detail::copyD2H(&v, data(), sizeof(datatype));
            return v;
        }

        // ── No iterators, deliberately ─────────────────────────────────
        //
        // A range-for over a device matrix would be one PCIe round trip PER
        // ELEMENT -- about 5 us each, so a 1000x1000 matrix is roughly a
        // minute of pure latency for what the CPU does in under a millisecond.
        //
        // It would also compile silently. Code written against the ml:: layer
        // is meant to build for either backend, and a loop that works on the
        // CPU and quietly crawls on the GPU is worse than one that refuses:
        // the refusal is found at compile time, on the machine writing it.
        //
        // Download once and iterate the host copy, or express the loop as a
        // whole-matrix operation:
        //
        //     for (double v : dA.cpu()) ...      // one transfer, then free
        //     double s = dA.sum();               // better: no transfer at all
        template <typename U = datatype>
        U* begin() {
            static_assert(detail::always_false<U>,
                          "mgpu::Matrix has no begin()/end(): iterating device memory costs a "
                          "PCIe round trip per element. Use .cpu() to get an iterable host "
                          "matrix, or a whole-matrix operation instead.");
            return nullptr;
        }
        template <typename U = datatype>
        U* end() {
            static_assert(detail::always_false<U>,
                          "mgpu::Matrix has no begin()/end(): iterating device memory costs a "
                          "PCIe round trip per element. Use .cpu() to get an iterable host "
                          "matrix, or a whole-matrix operation instead.");
            return nullptr;
        }

        // --- Transfer ---

        // Downloads. The only device->host crossing in the class.
        mcpu::Matrix<datatype> cpu() const {
            mcpu::Matrix<datatype> out(rows_, cols_);
            if (size()) detail::copyD2H(hostData(out), buf_.get(), buf_.bytes());
            return out;
        }

        // --- Output ---
        //
        // Formatting is a HOST job: it walks elements one at a time, builds
        // strings, and produces at most a few kilobytes. There is nothing to
        // parallelise and nothing to gain from doing it on the device, so all
        // of these download once and hand the work to the CPU package's
        // formatter -- which means every format matio knows (Pretty, CSV, TSV,
        // Markdown, MATLAB, NumPy, JSON) works on a device matrix for free.
        //
        // matio lives in mcpu, so it is qualified: the formats are shared, the
        // namespace split is not.

        void print(std::ostream& os, const mcpu::matio::Opts& o) const { cpu().print(os, o); }
        void print(std::ostream& os) const { cpu().print(os); }
        void print(const mcpu::matio::Opts& o) const { cpu().print(o); }
        void print() const { cpu().print(); }
        void print(int precision) const { cpu().print(precision); }
        std::string str(const mcpu::matio::Opts& o = {}) const { return cpu().str(o); }
        void save(const std::string& path, mcpu::matio::Opts o = {}) const { cpu().save(path, o); }

        // --- Fills ---

        Matrix& fill(datatype v) {
            detail::fill(size(), data(), v);
            return *this;
        }

        // Uniform on [lo, hi), from cuRAND's own stream. It has nothing to do
        // with ran2 on the CPU side and cannot: reproducing a sequential
        // generator's order on the GPU would mean serialising it, which gives
        // up the only reason to be here.
        //
        // seed != 0 restarts the stream, so the same seed always gives the
        // same matrix. seed == 0 (the default) continues it, so consecutive
        // calls differ - the same split set_Ran_values has on the CPU.
        // For a complex matrix both parts are filled independently, which
        // needs no separate code path: std::complex<R> is R[2], so the buffer
        // already IS 2n contiguous reals and cuRAND can fill it directly.
        Matrix& set_Ran_values(real_type lo, real_type hi, unsigned long long seed = 0) {
            detail::randUniform(size() * (isComplex ? 2 : 1), (real_type*)data(), lo, hi, seed);
            return *this;
        }
        Matrix& randn(real_type mean = 0, real_type sd = 1, unsigned long long seed = 0) {
            detail::randNormal(size() * (isComplex ? 2 : 1), (real_type*)data(), mean, sd, seed);
            return *this;
        }

      private:
        // Negative indices count from the end -- A(-1, -1) is the last
        // element. The CPU side documents this too but implements it as
        // `i % rows`, and C++ gives -1 % 3 == -1, so it reads BEFORE the
        // buffer instead of wrapping. This does the arithmetic that the
        // documentation describes, and refuses anything still out of range: a
        // stray index on the device corrupts memory silently rather than
        // segfaulting where you can see it.
        std::size_t flatIndex(long i, long j) const {
            const long r = i < 0 ? i + rows_ : i;
            const long c = j < 0 ? j + cols_ : j;
            if (r < 0 || r >= rows_ || c < 0 || c >= cols_)
                throw Error("index (" + std::to_string(i) + ", " + std::to_string(j) +
                            ") is outside a " + std::to_string(rows_) + "x" +
                            std::to_string(cols_) + " matrix");
            return (std::size_t)r * (std::size_t)cols_ + (std::size_t)c;
        }
        std::size_t flatIndex(long k) const {
            const long n = (long)size();
            const long q = k < 0 ? k + n : k;
            if (q < 0 || q >= n)
                throw Error("flat index " + std::to_string(k) + " is outside a matrix of " +
                            std::to_string(n) + " elements");
            return (std::size_t)q;
        }

        static void checkDims(long r, long c) {
            if (r < 0 || c < 0)
                throw Error("Matrix: negative dimension (" + std::to_string(r) + "x" +
                               std::to_string(c) + ")");
        }
        static std::size_t bytesFor(long r, long c) {
            return sizeof(datatype) * (std::size_t)r * (std::size_t)c;
        }

        // Matrix<T> exposes elements through operator(), not a raw pointer, so
        // reach the contiguous buffer through the first element. Matrix is
        // row-major and contiguous, which is what makes the single memcpy
        // valid; that is checked by the round-trip test in gpu/test.
        static datatype* hostData(mcpu::Matrix<datatype>& m) { return &m(0, 0); }
        static const datatype* hostData(const mcpu::Matrix<datatype>& m) {
            return &const_cast<mcpu::Matrix<datatype>&>(m)(0, 0);
        }

        void requireSame(const Matrix& o, const char* what) const {
            if (rows_ != o.rows_ || cols_ != o.cols_)
                throw Error(std::string(what) + ": dimension mismatch (" +
                               std::to_string(rows_) + "x" + std::to_string(cols_) + ") vs (" +
                               std::to_string(o.rows_) + "x" + std::to_string(o.cols_) + ")");
        }
        void requireSquare(const char* what) const {
            if (rows_ != cols_)
                throw Error(std::string(what) + ": matrix must be square, got " +
                               std::to_string(rows_) + "x" + std::to_string(cols_));
        }
        void requireNonEmpty(const char* what) const {
            if (empty()) throw Error(std::string(what) + ": matrix is empty");
        }

        // Compile-time gate for the operations that need an ORDER. Written as a
        // function rather than a class-level static_assert so a complex matrix
        // is perfectly usable -- you only hear about it if you call one of
        // them, and then the message says which.
        static void requireReal(const char* what) {
            static_assert(!isComplex || sizeof(what) == 0,
                          "this operation needs an ordering and the complex numbers do not have "
                          "one: max, min, floor, ceil, round and sign are real-only. Take .abs() "
                          "or .real() first.");
        }

        long rows_ = 0, cols_ = 0;
        DeviceBuffer buf_;

        // Carrier holding *this in column-major order, which is what cuSOLVER
        // speaks. A column-major m x n buffer is bit-for-bit a row-major n x m
        // one, so the conversion is a transpose and the carrier's declared
        // shape is deliberately the transposed one: it is a buffer with a
        // shape attached, not a matrix anyone should read.
        Matrix colMajor() const {
            Matrix out(cols_, rows_, uninit_t{});
            // The PLAIN transpose, never the conjugate one: this is a layout
            // change, not a mathematical operation, and conjugating here would
            // silently corrupt every complex factorisation.
            if constexpr (isComplex)
                detail::transposeCx((int)rows_, (int)cols_, data(), out.data(), false);
            else
                detail::transpose((int)rows_, (int)cols_, data(), out.data());
            return out;
        }

        // The inverse reading: `cm` points at a column-major m x n buffer,
        // which is a row-major n x m one, so transposing it gives row-major
        // m x n. Takes a bare pointer because cuSOLVER often leaves the piece
        // we want as a PREFIX of a larger buffer (orgqr's first k columns),
        // and a prefix needs no copy to reinterpret.
        static Matrix rowMajorFrom(const datatype* cm, long m, long n) {
            Matrix out(m, n, uninit_t{});
            if (m && n) {
                if constexpr (isComplex)
                    detail::transposeCx((int)n, (int)m, cm, out.data(), false);
                else
                    detail::transpose((int)n, (int)m, cm, out.data());
            }
            return out;
        }

      public:
        // Opens a FUSED expression: everything chained onto the result is
        // compiled into a single kernel with a single temporary instead of one
        // of each per step. Defined below Expr, which it returns.
        //
        //     Matrix<double> C = (dA.lazy() % dB).exp().sqrt();
        //
        // Once one operand is lazy the rest of the chain follows, so only the
        // first needs saying. See Expr for what is and is not fusable.
        Expr<datatype> lazy() const;

        // --- Element-wise binary ---
        //
        // Each returns a fresh device matrix immediately. No host round trip,
        // so chains like (A % B + C).exp() cross the bus exactly zero times --
        // but they do allocate and traverse a temporary per step. Add .lazy()
        // to collapse the whole chain into one kernel.

        Matrix operator+(const Matrix& o) const { return zip(o, detail::BinOp::Add, "operator+"); }
        Matrix operator-(const Matrix& o) const { return zip(o, detail::BinOp::Sub, "operator-"); }
        // Hadamard, matching Matrix<datatype>::operator%.
        Matrix operator%(const Matrix& o) const { return zip(o, detail::BinOp::Mul, "operator%"); }
        Matrix operator/(const Matrix& o) const { return zip(o, detail::BinOp::Div, "operator/"); }
        Matrix emax(const Matrix& o) const {
            requireReal("emax");
            return zip(o, detail::BinOp::Max, "emax");
        }
        Matrix emin(const Matrix& o) const {
            requireReal("emin");
            return zip(o, detail::BinOp::Min, "emin");
        }

        Matrix& operator+=(const Matrix& o) { return zipInto(o, detail::BinOp::Add, "operator+="); }
        Matrix& operator-=(const Matrix& o) { return zipInto(o, detail::BinOp::Sub, "operator-="); }
        Matrix& operator%=(const Matrix& o) { return zipInto(o, detail::BinOp::Mul, "operator%="); }

        // --- Scalar ---

        Matrix operator+(datatype s) const { return scalar(detail::BinOp::Add, s, false); }
        Matrix operator-(datatype s) const { return scalar(detail::BinOp::Sub, s, false); }
        Matrix operator*(datatype s) const { return scalar(detail::BinOp::Mul, s, false); }
        Matrix operator/(datatype s) const { return scalar(detail::BinOp::Div, s, false); }

        // Scalar += and -=, which mcpu::Matrix has and this did not.
        Matrix& operator+=(datatype s) {
            detail::binaryScalar(detail::BinOp::Add, size(), data(), s, data(), false);
            return *this;
        }
        Matrix& operator-=(datatype s) {
            detail::binaryScalar(detail::BinOp::Sub, size(), data(), s, data(), false);
            return *this;
        }
        Matrix& operator*=(datatype s) {
            detail::binaryScalar(detail::BinOp::Mul, size(), data(), s, data(), false);
            return *this;
        }
        Matrix& operator/=(datatype s) {
            detail::binaryScalar(detail::BinOp::Div, size(), data(), s, data(), false);
            return *this;
        }

        // --- Element-wise unary ---

        Matrix operator-() const { return map(detail::UnOp::Neg); }

        // |z| is REAL even when z is not, so this returns real_type. For a real
        // matrix real_type is datatype and nothing changes.
        Matrix<real_type> abs() const {
            if constexpr (isComplex) {
                Matrix<real_type> out(rows_, cols_, typename Matrix<real_type>::uninit_t{});
                detail::project(2, size(), data(), out.data());
                return out;
            } else {
                return map(detail::UnOp::Abs);
            }
        }
        Matrix sqrt() const { return map(detail::UnOp::Sqrt); }
        Matrix exp() const { return map(detail::UnOp::Exp); }
        // Naming follows basic/ exactly: ln() is the natural log and log()
        // takes a base. Spelling log() as the natural one here would be the
        // more common convention and precisely the wrong choice - a call that
        // compiles on both sides and means different things is worse than one
        // that does not compile at all.
        Matrix ln() const { return map(detail::UnOp::Log); }
        Matrix lg() const { return map(detail::UnOp::Log2); }
        Matrix log10() const { return map(detail::UnOp::Log10); }
        Matrix log(datatype base) const {
            // Same identity basic/ uses, and for the same reason: one log2
            // pass plus a multiply beats a per-element division.
            return lg() * datatype(1.0 / std::log2((double)base));
        }
        Matrix exp2() const { return map(detail::UnOp::Exp2); }
        Matrix sin() const { return map(detail::UnOp::Sin); }
        Matrix cos() const { return map(detail::UnOp::Cos); }
        Matrix tan() const { return map(detail::UnOp::Tan); }
        Matrix asin() const { return map(detail::UnOp::Asin); }
        Matrix acos() const { return map(detail::UnOp::Acos); }
        Matrix atan() const { return map(detail::UnOp::Atan); }
        Matrix sinh() const { return map(detail::UnOp::Sinh); }
        Matrix cosh() const { return map(detail::UnOp::Cosh); }
        Matrix tanh() const { return map(detail::UnOp::Tanh); }
        // Ordering-based, so real only. The complex numbers are a field but
        // not an ordered one; there is no defensible answer for floor(1+2i),
        // and inventing one would be worse than not offering it.
        Matrix floor() const { requireReal("floor"); return map(detail::UnOp::Floor); }
        Matrix ceil() const { requireReal("ceil"); return map(detail::UnOp::Ceil); }
        Matrix round() const { requireReal("round"); return map(detail::UnOp::Round); }
        Matrix sign() const { requireReal("sign"); return map(detail::UnOp::Sign); }

        // --- Complex parts ---
        //
        // real() and imag() drop to a real matrix; conj() and H() stay complex.
        // All four exist for real matrices too, where they are the identity or
        // the plain transpose, so generic code does not have to branch.

        Matrix<real_type> real() const {
            if constexpr (isComplex) {
                Matrix<real_type> out(rows_, cols_, typename Matrix<real_type>::uninit_t{});
                detail::project(0, size(), data(), out.data());
                return out;
            } else {
                return *this;
            }
        }
        Matrix<real_type> imag() const {
            if constexpr (isComplex) {
                Matrix<real_type> out(rows_, cols_, typename Matrix<real_type>::uninit_t{});
                detail::project(1, size(), data(), out.data());
                return out;
            } else {
                return Matrix<real_type>(rows_, cols_);   // all zeros
            }
        }
        Matrix<real_type> arg() const {
            Matrix<real_type> out(rows_, cols_, typename Matrix<real_type>::uninit_t{});
            if constexpr (isComplex) detail::project(3, size(), data(), out.data());
            else detail::unary(detail::UnOp::Sign, size(), data(), out.data());
            return out;
        }
        Matrix conj() const {
            if constexpr (isComplex) {
                Matrix out(rows_, cols_, uninit_t{});
                detail::conj(size(), data(), out.data());
                return out;
            } else {
                return *this;
            }
        }

        // Builds a complex matrix from real parts. For a real datatype this is
        // just a copy of `re`, so it stays callable from generic code.
        static Matrix fromParts(const Matrix<real_type>& re, const Matrix<real_type>* im) {
            Matrix out(re.rows(), re.cols(), uninit_t{});
            if constexpr (isComplex)
                detail::compose(out.size(), re.data(), im ? im->data() : nullptr, out.data());
            else
                detail::copyD2D(out.data(), re.data(), out.bytes());
            return out;
        }
        Matrix pow2() const { return map(detail::UnOp::Square); }   // square, as in basic/
        Matrix pow(datatype e) const { return scalar(detail::BinOp::Pow, e, false); }

        // The rest of <cmath>, matching mcpu::Matrix. Real only: acosh and its
        // relatives do have complex branches, but they need a branch-cut
        // convention to be pinned down and guessing one is worse than not
        // offering it.
        Matrix asinh() const { requireReal("asinh"); return map(detail::UnOp::Asinh); }
        Matrix acosh() const { requireReal("acosh"); return map(detail::UnOp::Acosh); }
        Matrix atanh() const { requireReal("atanh"); return map(detail::UnOp::Atanh); }
        Matrix cbrt() const { requireReal("cbrt"); return map(detail::UnOp::Cbrt); }
        // log(1+x) and exp(x)-1, accurate for small x where the naive forms
        // lose every significant digit.
        Matrix log1p() const { requireReal("log1p"); return map(detail::UnOp::Log1p); }
        Matrix expm1() const { requireReal("expm1"); return map(detail::UnOp::Expm1); }
        // Toward zero, unlike floor (down) and round (to nearest). basic/ calls
        // this fix(), after MATLAB; both spellings are here.
        Matrix trunc() const { requireReal("trunc"); return map(detail::UnOp::Trunc); }
        Matrix fix() const { return trunc(); }

        Matrix atan2(const Matrix& o) const { return zip(o, detail::BinOp::Atan2, "atan2"); }
        Matrix hypot(const Matrix& o) const { return zip(o, detail::BinOp::Hypot, "hypot"); }
        // mod keeps the sign of the dividend; rem rounds to the nearest
        // multiple and may be negative -- the same split basic/ makes.
        Matrix mod(const Matrix& o) const { return zip(o, detail::BinOp::Mod, "mod"); }
        Matrix rem(const Matrix& o) const { return zip(o, detail::BinOp::Rem, "rem"); }
        Matrix mod(datatype v) const { return scalar(detail::BinOp::Mod, v, false); }
        Matrix rem(datatype v) const { return scalar(detail::BinOp::Rem, v, false); }

        // Logical combination of masks, matching mcpu::Matrix. Any non-zero
        // counts as true, so these compose with the comparison results above.
        Matrix land(const Matrix& o) const { return zip(o, detail::BinOp::And, "land"); }
        Matrix lor(const Matrix& o) const { return zip(o, detail::BinOp::Or, "lor"); }
        Matrix lxor(const Matrix& o) const { return zip(o, detail::BinOp::Xor, "lxor"); }
        Matrix lnot() const { requireReal("lnot"); return map(detail::UnOp::Not); }

        // --- Vector products ---
        //
        // Both take either orientation, since a "vector" here is any matrix
        // with one row or one column.
        datatype dot(const Matrix& o) const {
            if (size() != o.size())
                throw Error("dot: lengths differ (" + std::to_string(size()) + " vs " +
                            std::to_string(o.size()) + ")");
            if constexpr (isComplex) {
                // The HERMITIAN inner product, conjugating the left operand --
                // which is what makes dot(x, x) real and equal to norm squared.
                return (conj() % o).sum();
            } else {
                return (*this % o).sum();
            }
        }

        // --- Matrix multiply ---

        Matrix operator*(const Matrix& o) const {
            if (cols_ != o.rows_)
                throw Error("operator*: inner dimensions disagree (" + std::to_string(rows_) +
                               "x" + std::to_string(cols_) + ") * (" + std::to_string(o.rows_) +
                               "x" + std::to_string(o.cols_) + ")");
            Matrix out(rows_, o.cols_, uninit_t{});
            detail::gemm((int)rows_, (int)o.cols_, (int)cols_, datatype(1), data(), o.data(),
                         datatype(0), out.data());
            return out;
        }

        // C = alpha*A*B + beta*C, in place. The fused form is why cuBLAS is
        // here: it saves reading and rewriting C, which for a memory-bound
        // shape is most of the cost.
        Matrix& gemmInto(const Matrix& A, const Matrix& B, datatype alpha = 1,
                            datatype beta = 0) {
            if (A.cols_ != B.rows_ || A.rows_ != rows_ || B.cols_ != cols_)
                throw Error("gemmInto: shapes do not form C = A*B");
            detail::gemm((int)A.rows_, (int)B.cols_, (int)A.cols_, alpha, A.data(), B.data(), beta,
                         data());
            return *this;
        }

        // The PLAIN transpose. For complex data this is almost never the one
        // you want -- see H() below.
        Matrix T() const {
            Matrix out(cols_, rows_, uninit_t{});
            if (size()) {
                if constexpr (isComplex)
                    detail::transposeCx((int)rows_, (int)cols_, data(), out.data(), false);
                else
                    detail::transpose((int)rows_, (int)cols_, data(), out.data());
            }
            return out;
        }

        // The CONJUGATE transpose, matching Matrix::H() on the CPU side. This
        // is the transpose that matters for complex matrices: it is what makes
        // Q^H Q = I and A = U S V^H true. For a real matrix it is exactly T(),
        // so generic code can always say H() and be right.
        Matrix H() const {
            if constexpr (isComplex) {
                Matrix out(cols_, rows_, uninit_t{});
                if (size()) detail::transposeCx((int)rows_, (int)cols_, data(), out.data(), true);
                return out;
            } else {
                return T();
            }
        }

        // --- Reductions ---
        //
        // These return HOST scalars and therefore synchronise. That is the one
        // place a device-resident design has to stop and wait, so they are
        // worth being deliberate about inside a loop.

        datatype sum() const { return red(detail::RedOp::Sum, "sum"); }
        datatype prod() const { return red(detail::RedOp::Prod, "prod"); }
        datatype max() const { requireReal("max"); return red(detail::RedOp::Max, "max"); }
        datatype min() const { requireReal("min"); return red(detail::RedOp::Min, "min"); }
        datatype mean() const { return red(detail::RedOp::Sum, "mean") / datatype(size()); }
        // Frobenius norm, matching Matrix<datatype>::norm() with no argument.
        real_type norm() const {
            requireNonEmpty("norm");
            if constexpr (isComplex) {
                // sum |z|^2, which is real -- not sum z^2, which is not.
                return std::sqrt(detail::normSq(size(), data()));
            } else {
                return std::sqrt(red(detail::RedOp::SumSq, "norm"));
            }
        }

        // Axis reductions, with basic/'s convention: ROW (true) reduces ALONG
        // each row and gives one value per row; COL (false) gives one per
        // column. Shapes match Matrix: (rows x 1) and (1 x cols).
        Matrix sum(bool axis) const { return redAxis(detail::RedOp::Sum, axis); }
        Matrix max(bool axis) const {
            requireReal("max(axis)");
            return redAxis(detail::RedOp::Max, axis);
        }
        Matrix min(bool axis) const {
            requireReal("min(axis)");
            return redAxis(detail::RedOp::Min, axis);
        }

        // ── Views ──────────────────────────────────────────────────────
        //
        // A non-owning rectangular reference into a matrix, so a row, a column
        // or a submatrix can be READ or WRITTEN without the caller doing the
        // index arithmetic. The CPU proxies (RowProxy, ColProxy, SubProxy)
        // spell the same thing:
        //
        //     dA(all, 0) = b;                    // overwrite a column
        //     dA(3, all).set_Ran_values(0, 1);   // refill a row
        //     Matrix<double> c = dA(all, 2);     // read one out
        //
        // WHAT THIS IS NOT: a strided view that every kernel understands. A
        // view materialises when it is read and writes back through setBlock
        // when it is assigned, which is exactly what the CPU proxies do. Making
        // every operation stride-aware would touch every kernel in the package
        // to save a copy that is O(size of the view), not of the matrix.
        //
        // LIFETIME: a view points at its parent and does not keep it alive.
        // Consume it in the statement that creates it, like any proxy.

        class View {
          public:
            View(Matrix& m, long r0, long c0, long nr, long nc)
                : m_(&m), r0_(r0), c0_(c0), nr_(nr), nc_(nc) {
                if (r0 < 0 || c0 < 0 || r0 + nr > m.rows() || c0 + nc > m.cols())
                    throw Error("view: region falls outside the matrix");
            }

            long rows() const { return nr_; }
            long cols() const { return nc_; }

            Matrix eval() const { return m_->block(r0_, c0_, nr_, nc_); }
            operator Matrix() const { return eval(); }

            View& operator=(const Matrix& src) {
                if (src.rows() != nr_ || src.cols() != nc_)
                    throw Error("view assignment: shape mismatch (" + std::to_string(nr_) + "x" +
                                std::to_string(nc_) + ") vs (" + std::to_string(src.rows()) + "x" +
                                std::to_string(src.cols()) + ")");
                m_->setBlock(r0_, c0_, src);
                return *this;
            }
            View& operator=(const View& src) { return *this = src.eval(); }
            View& operator=(const Expr<datatype>& e) { return *this = e.eval(); }

            // Scalar fill and random refill go through a temporary of the
            // view's own size rather than a masked kernel: the region is
            // usually small, the pool makes the allocation nearly free, and it
            // reuses code that is already tested.
            View& fill(datatype v) {
                Matrix t(nr_, nc_, uninit_t{});
                t.fill(v);
                return *this = t;
            }
            View& operator=(datatype v) { return fill(v); }

            View& set_Ran_values(real_type lo, real_type hi, unsigned long long seed = 0) {
                Matrix t(nr_, nc_, uninit_t{});
                t.set_Ran_values(lo, hi, seed);
                return *this = t;
            }
            View& randn(real_type mean = 0, real_type sd = 1, unsigned long long seed = 0) {
                Matrix t(nr_, nc_, uninit_t{});
                t.randn(mean, sd, seed);
                return *this = t;
            }

            // Arithmetic materialises first, so a view reads like the matrix it
            // stands for.
            Matrix operator+(const Matrix& r) const { return eval() + r; }
            Matrix operator-(const Matrix& r) const { return eval() - r; }
            Matrix operator%(const Matrix& r) const { return eval() % r; }
            Matrix operator*(const Matrix& r) const { return eval() * r; }
            Matrix operator*(datatype s) const { return eval() * s; }
            Matrix operator/(datatype s) const { return eval() / s; }
            Matrix operator+(datatype s) const { return eval() + s; }
            Matrix operator-(datatype s) const { return eval() - s; }
            Expr<datatype> lazy() const { return Expr<datatype>(eval()); }

            // Compound assignment. The same gap the CPU proxies had: the binary
            // operators above are members so they can be found at all, but
            // operator+= must ALSO be a member -- there is no conversion path
            // that lets Matrix's own be found through a view.
            //
            // Each materialises, hands the work to Matrix's own compound
            // operator, and writes the result back through setBlock. Delegating
            // rather than reimplementing means a view accepts exactly what a
            // Matrix accepts, and the two cannot drift apart.
            template <typename Rhs>
            View& operator+=(const Rhs& r) {
                Matrix t = eval();
                t += r;
                return *this = t;
            }
            template <typename Rhs>
            View& operator-=(const Rhs& r) {
                Matrix t = eval();
                t -= r;
                return *this = t;
            }
            template <typename Rhs>
            View& operator%=(const Rhs& r) {
                Matrix t = eval();
                t %= r;
                return *this = t;
            }
            template <typename Rhs>
            View& operator*=(const Rhs& r) {
                Matrix t = eval();
                t *= r;
                return *this = t;
            }
            template <typename Rhs>
            View& operator/=(const Rhs& r) {
                Matrix t = eval();
                t /= r;
                return *this = t;
            }

          private:
            Matrix* m_;
            long r0_, c0_, nr_, nc_;
        };

        // Non-const gives a writable view; const gives the materialised copy,
        // so a read never needs the caller to think about which they have.
        View operator()(mcpu::all_t, long j) { return View(*this, 0, j, rows_, 1); }
        View operator()(long i, mcpu::all_t) { return View(*this, i, 0, 1, cols_); }
        View operator()(mcpu::all_t, mcpu::all_t) { return View(*this, 0, 0, rows_, cols_); }
        View view(long r0, long c0, long nr, long nc) { return View(*this, r0, c0, nr, nc); }

        Matrix operator()(mcpu::all_t, long j) const { return block(0, j, rows_, 1); }
        Matrix operator()(long i, mcpu::all_t) const { return block(i, 0, 1, cols_); }
        Matrix view(long r0, long c0, long nr, long nc) const { return block(r0, c0, nr, nc); }

        Matrix col(long j) const { return block(0, j, rows_, 1); }
        Matrix row(long i) const { return block(i, 0, 1, cols_); }

        // --- Blocks ---

        Matrix block(long r0, long c0, long nr, long nc) const {
            Matrix out(nr, nc, uninit_t{});
            detail::copyBlock((int)rows_, (int)cols_, data(), (int)r0, (int)c0, (int)nr, (int)nc,
                              out.data());
            return out;
        }
        // Writes `src` into this matrix with its top-left corner at (r0, c0).
        Matrix& setBlock(long r0, long c0, const Matrix& src) {
            detail::setBlock((int)rows_, (int)cols_, data(), (int)r0, (int)c0, (int)src.rows_,
                             (int)src.cols_, src.data());
            return *this;
        }

        // Side by side, and stacked. Both are one allocation plus two block
        // writes, which is all hstack/vstack ever are once setBlock exists.
        Matrix hstack(const Matrix& r) const {
            if (rows_ != r.rows_)
                throw Error("hstack: row counts differ (" + std::to_string(rows_) + " vs " +
                            std::to_string(r.rows_) + ")");
            Matrix out(rows_, cols_ + r.cols_, uninit_t{});
            out.setBlock(0, 0, *this);
            out.setBlock(0, cols_, r);
            return out;
        }
        Matrix vstack(const Matrix& r) const {
            if (cols_ != r.cols_)
                throw Error("vstack: column counts differ (" + std::to_string(cols_) + " vs " +
                            std::to_string(r.cols_) + ")");
            Matrix out(rows_ + r.rows_, cols_, uninit_t{});
            out.setBlock(0, 0, *this);
            out.setBlock(rows_, 0, r);
            return out;
        }

        // Zero-padded (or truncated) copy — what every FFT of a non-power-of-two
        // length needs before it starts.
        Matrix resized(long r, long c) const {
            Matrix out(r, c);
            out.setBlock(0, 0, block(0, 0, std::min(r, rows_), std::min(c, cols_)));
            return out;
        }

        Matrix diag() const {
            const long k = std::min(rows_, cols_);
            Matrix out(k, 1, uninit_t{});
            detail::getDiagonal((int)rows_, (int)cols_, data(), out.data());
            return out;
        }
        Matrix triu() const {
            Matrix out(*this);
            detail::triangle((int)rows_, (int)cols_, out.data(), true, false);
            return out;
        }
        Matrix tril(bool unitDiag = false) const {
            Matrix out(*this);
            detail::triangle((int)rows_, (int)cols_, out.data(), false, unitDiag);
            return out;
        }

        // ── Comparisons and masks ──────────────────────────────────────
        //
        // These return a matrix of 1.0 and 0.0 in the matrix's own type, not a
        // packed bool array. A numeric mask composes with everything else here
        // -- `A % A.gt(0)` zeroes the negatives, `mask.sum()` counts the hits --
        // where a separate bool type would need its own parallel vocabulary
        // before it was good for anything. Real only: complex is not ordered.

        Matrix lt(const Matrix& o) const { return cmp(o, detail::CmpOp::LT, "lt"); }
        Matrix le(const Matrix& o) const { return cmp(o, detail::CmpOp::LE, "le"); }
        Matrix gt(const Matrix& o) const { return cmp(o, detail::CmpOp::GT, "gt"); }
        Matrix ge(const Matrix& o) const { return cmp(o, detail::CmpOp::GE, "ge"); }
        Matrix eq(const Matrix& o) const { return cmp(o, detail::CmpOp::EQ, "eq"); }
        Matrix ne(const Matrix& o) const { return cmp(o, detail::CmpOp::NE, "ne"); }

        Matrix lt(datatype v) const { return cmpScalar(detail::CmpOp::LT, v); }
        Matrix le(datatype v) const { return cmpScalar(detail::CmpOp::LE, v); }
        Matrix gt(datatype v) const { return cmpScalar(detail::CmpOp::GT, v); }
        Matrix ge(datatype v) const { return cmpScalar(detail::CmpOp::GE, v); }
        Matrix eq(datatype v) const { return cmpScalar(detail::CmpOp::EQ, v); }
        Matrix ne(datatype v) const { return cmpScalar(detail::CmpOp::NE, v); }

        long nnz() const {
            requireNonEmpty("nnz");
            if constexpr (isComplex) return (long)std::llround(abs().gt(real_type(0)).sum());
            else return (long)std::llround(ne(datatype(0)).sum());
        }
        bool any() const { return nnz() > 0; }
        bool all() const { return nnz() == (long)size(); }

        // Matches Matrix::allclose on the CPU side, tolerances included.
        bool allclose(const Matrix& o, double rtol = 1e-5, double atol = 1e-8) const {
            requireSame(o, "allclose");
            if (empty()) return true;
            Matrix<real_type> d = (*this - o).abs();
            Matrix<real_type> lim = o.abs() * real_type(rtol) + real_type(atol);
            return d.le(lim).min() > real_type(0);
        }

        // ── Statistics ─────────────────────────────────────────────────

        // Two-pass, on purpose. The one-pass E[x^2] - E[x]^2 loses every
        // significant digit when the mean is large next to the spread, and the
        // extra pass is bandwidth we can afford.
        real_type var() const {
            requireNonEmpty("var");
            const long n = (long)size();
            if (n < 2) return real_type(0);
            const datatype m = mean();
            if constexpr (isComplex) {
                Matrix d = *this - m;
                return detail::normSq(size(), d.data()) / real_type(n - 1);
            } else {
                Matrix d = (this->lazy() - m).pow2().eval();
                return d.sum() / real_type(n - 1);
            }
        }
        real_type stddev() const { return std::sqrt(var()); }

        // Row-major index of the extreme element; ties go to the lowest index.
        long argmax() const {
            requireReal("argmax");
            requireNonEmpty("argmax");
            return detail::argExtreme(true, size(), data());
        }
        long argmin() const {
            requireReal("argmin");
            requireNonEmpty("argmin");
            return detail::argExtreme(false, size(), data());
        }

        // ── Norms ──────────────────────────────────────────────────────
        //
        // Same four as basic/: Frobenius (the default), the maximum absolute
        // column sum, the maximum absolute row sum, and the spectral norm.
        using NormType = mcpu::NormType;

        real_type norm(NormType type) const {
            requireNonEmpty("norm");
            Matrix<real_type> a = abs();
            switch (type) {
                case NormType::One:
                    // max over columns of the column sums
                    return a.sum(false).max();
                case NormType::Inf:
                    return a.sum(true).max();
                case NormType::Two:
                    // The largest singular value. Genuinely expensive -- it is
                    // a full SVD -- which is why Fro stays the default.
                    return svdvals().max();
                default:
                    return norm();
            }
        }

        // ── Derived linear algebra ─────────────────────────────────────
        //
        // All of these fall out of the singular values, so they share one
        // factorisation apiece rather than deriving anything by hand.

        datatype trace() const {
            requireNonEmpty("trace");
            return diag().sum();
        }

        // Ratio of the largest singular value to the smallest. Infinite for a
        // singular matrix, which is the honest answer rather than an error.
        real_type cond() const {
            Matrix<real_type> s = svdvals();
            const real_type lo = s.min(), hi = s.max();
            return lo == real_type(0) ? std::numeric_limits<real_type>::infinity() : hi / lo;
        }

        // Numerical rank: singular values above max(m,n) * eps * sigma_max,
        // which is LAPACK's and NumPy's default cut.
        long rank(double tol = -1) const {
            Matrix<real_type> s = svdvals();
            const real_type hi = s.max();
            const real_type cut =
                tol >= 0 ? real_type(tol)
                         : real_type(std::max(rows_, cols_)) *
                               std::numeric_limits<real_type>::epsilon() * hi;
            return (long)std::llround(s.gt(cut).sum());
        }

        // Moore-Penrose pseudo-inverse, V * diag(1/s) * U^H with the small
        // singular values dropped rather than inverted -- inverting them is
        // what makes a naive pinv explode.
        Matrix pinv(double tol = -1) const {
            auto [U, S, V] = svd(/*full=*/false);
            Matrix<real_type> s = svdvals();
            const real_type hi = s.max();
            const real_type cut =
                tol >= 0 ? real_type(tol)
                         : real_type(std::max(rows_, cols_)) *
                               std::numeric_limits<real_type>::epsilon() * hi;
            const long k = std::min(rows_, cols_);
            Matrix<real_type> inv(k, 1, typename Matrix<real_type>::uninit_t{});
            // 1/s where s is above the cut, 0 where it is not: keep = (s > cut),
            // so keep/(s + (1-keep)) never divides by zero.
            Matrix<real_type> keep = s.gt(cut);
            inv = keep / (s + (keep * real_type(-1) + real_type(1)));
            Matrix D(k, k);
            Matrix ic = fromParts(inv, nullptr);
            detail::setDiagonal((int)k, (int)k, D.data(), ic.data());
            return V * D * U.H();
        }

        // ── Sorting ────────────────────────────────────────────────────
        //
        // Same signatures and the same conventions as mcpu::Matrix, so these
        // survive a namespace swap. Real only -- sorting needs an order.

        // Sorts each column (axis = COL) or each row (axis = ROW). Same shape
        // as the input, like MATLAB's sort: this rearranges, it does not reduce.
        Matrix sort(bool axis, bool descending = false) const {
            requireReal("sort");
            Matrix out(*this);
            detail::sortAxis((int)rows_, (int)cols_, out.data(), axis, descending);
            return out;
        }
        // Every element, ascending, keeping the shape.
        Matrix sorted(bool descending = false) const {
            requireReal("sorted");
            Matrix out(*this);
            detail::sortFlat(size(), out.data(), descending);
            return out;
        }

        // Sorts whole ROWS by column `key`, carrying every other column along.
        // Stable, so repeated calls compose into a multi-column sort.
        Matrix sortrows(long key = 0, bool descending = false) const {
            requireReal("sortrows");
            Matrix out(*this);
            detail::sortRowsBy((int)rows_, (int)cols_, out.data(), (int)key, descending);
            return out;
        }

        // The distinct values, ascending, as a COLUMN vector -- MATLAB's unique.
        Matrix unique() const {
            requireReal("unique");
            requireNonEmpty("unique");
            Matrix buf(*this);
            const long k = detail::uniqueInPlace(size(), buf.data());
            // The survivors sit at the front of the buffer; take that prefix.
            Matrix out(k, 1, uninit_t{});
            detail::copyD2D(out.data(), buf.data(), sizeof(datatype) * (std::size_t)k);
            return out;
        }

        // Middle value, or the mean of the two middle ones when the count is
        // even -- which is why it returns a real rather than datatype.
        real_type median() const {
            requireReal("median");
            requireNonEmpty("median");
            Matrix s = sorted();
            const long n = (long)size();
            if (n % 2) return real_type(s[n / 2]);
            return (real_type(s[n / 2 - 1]) + real_type(s[n / 2])) / real_type(2);
        }

        // Per-column (axis = COL) or per-row (axis = ROW), matching sum().
        Matrix median(bool axis) const {
            requireReal("median");
            requireNonEmpty("median");
            Matrix s = sort(axis);
            // Along a row the run has cols_ entries; down a column, rows_.
            const long n = axis ? cols_ : rows_;
            const long mid = n / 2;
            auto slice = [&](long k) {
                return axis ? s.block(0, k, rows_, 1) : s.block(k, 0, 1, cols_);
            };
            if (n % 2) return slice(mid);
            return (slice(mid - 1) + slice(mid)) * datatype(0.5);
        }

        // Most frequent value; ties go to the SMALLEST, as MATLAB's mode does.
        datatype mode() const {
            requireReal("mode");
            requireNonEmpty("mode");
            return detail::modeOf(size(), data());
        }

        // ── Rearrangement ──────────────────────────────────────────────

        // Free: row-major contiguous data with new dimensions IS the reshape,
        // so only the two integers change.
        Matrix& reshape(long r, long c) {
            if (r * c != (long)size())
                throw Error("reshape: " + std::to_string(rows_) + "x" + std::to_string(cols_) +
                            " has " + std::to_string(size()) + " elements, cannot become " +
                            std::to_string(r) + "x" + std::to_string(c));
            rows_ = r;
            cols_ = c;
            return *this;
        }
        Matrix reshaped(long r, long c) const {
            Matrix out(*this);
            out.reshape(r, c);
            return out;
        }

        Matrix repmat(long p, long q) const {
            if (p <= 0 || q <= 0) throw Error("repmat: tile counts must be positive");
            Matrix out(rows_ * p, cols_ * q, uninit_t{});
            detail::rearrange(detail::Rearrange::Repmat, (int)rows_, (int)cols_, data(), (int)p,
                              (int)q, out.data());
            return out;
        }
        Matrix fliplr() const { return rearranged(detail::Rearrange::FlipLR, rows_, cols_, 0, 0); }
        Matrix flipud() const { return rearranged(detail::Rearrange::FlipUD, rows_, cols_, 0, 0); }
        Matrix rot90() const { return rearranged(detail::Rearrange::Rot90, cols_, rows_, 0, 0); }
        // Signature matches mcpu::Matrix::circshift exactly: shift by k along
        // dim 0 (rows) or dim 1 (columns).
        Matrix circshift(long k, int dim = 0) const {
            if (dim != 0 && dim != 1)
                throw Error("circshift: dim must be 0 (rows) or 1 (columns), got " +
                            std::to_string(dim));
            return rearranged(detail::Rearrange::CircShift, rows_, cols_, dim == 0 ? k : 0,
                              dim == 0 ? 0 : k);
        }

        Matrix kron(const Matrix& B) const {
            Matrix out(rows_ * B.rows_, cols_ * B.cols_, uninit_t{});
            detail::kron((int)rows_, (int)cols_, data(), (int)B.rows_, (int)B.cols_, B.data(),
                         out.data());
            return out;
        }

        // --- Builders ---

        static Matrix eye(long n) {
            Matrix out(n, n);
            Matrix ones(n, 1, uninit_t{});
            ones.fill(datatype(1));
            detail::setDiagonal((int)n, (int)n, out.data(), ones.data());
            return out;
        }
        static Matrix zeros(long r, long c) { return Matrix(r, c); }
        static Matrix diagonal(const Matrix& d) {
            const long n = d.size();
            Matrix out(n, n);
            detail::setDiagonal((int)n, (int)n, out.data(), d.data());
            return out;
        }

        // ==============================================================
        //  Factorisations
        // ==============================================================
        //
        // All of these are cuSOLVER's, which is the point: LAPACK-grade
        // algorithms nobody here has to re-derive. The only work done on our
        // side is the layout crossing, and it happens exactly twice per call
        // (in, and out) rather than being smeared through the algorithm.
        //
        // cuSOLVER reports failures through an `info` code, in LAPACK's
        // convention: > 0 names the step that failed, < 0 the bad argument.
        // Each wrapper turns a non-zero info into a Error naming what it
        // means, because "info = 3" at a call site is not a diagnosis.

        // LU with partial pivoting. Returns L (m x k, unit diagonal),
        // U (k x n), and the pivots in LAPACK's 1-based ipiv form.
        std::tuple<Matrix, Matrix, std::vector<int>> lu() const {
            requireNonEmpty("lu");
            const int m = (int)rows_, n = (int)cols_, k = std::min(m, n);
            Matrix cm = colMajor();
            DeviceBuffer piv(sizeof(int) * (std::size_t)k);
            const int info = detail::getrf(m, n, cm.data(), (int*)piv.get());
            if (info < 0) throw Error("lu: cuSOLVER rejected argument " + std::to_string(-info));
            if (info > 0)
                throw Error("lu: exact zero pivot at position " + std::to_string(info) +
                               " - the matrix is singular");
            Matrix packed = rowMajorFrom(cm.data(), rows_, cols_);
            Matrix L = packed.block(0, 0, m, k);
            detail::triangle(m, k, L.data(), false, true);
            Matrix U = packed.block(0, 0, k, n);
            detail::triangle(k, n, U.data(), true, false);
            std::vector<int> ipiv((std::size_t)k);
            detail::copyD2H(ipiv.data(), piv.get(), sizeof(int) * (std::size_t)k);
            return {std::move(L), std::move(U), std::move(ipiv)};
        }

        // Cholesky. Returns the LOWER factor: A = L * L^T for a real matrix,
        // A = L * L^H for a complex Hermitian one. Matches
        // mcpu::Matrix::cholesky() in both cases.
        Matrix cholesky() const {
            requireSquare("cholesky");
            requireNonEmpty("cholesky");
            const int n = (int)rows_;
            Matrix cm = colMajor();
            const int info = detail::potrf(n, cm.data(), false);
            if (info < 0)
                throw Error("cholesky: cuSOLVER rejected argument " + std::to_string(-info));
            if (info > 0)
                throw Error("cholesky: leading minor of order " + std::to_string(info) +
                               " is not positive definite");
            Matrix L = rowMajorFrom(cm.data(), rows_, cols_);
            // potrf leaves the untouched triangle as whatever was there.
            detail::triangle(n, n, L.data(), false, false);
            return L;
        }

        // Reduced (economy) QR: Q is m x k, R is k x n, k = min(m, n).
        // Unpivoted — cuSOLVER offers no pivoted geqrf, so unlike
        // Matrix::QR() there is no permutation to return.
        std::pair<Matrix, Matrix> qr() const {
            requireNonEmpty("qr");
            const int m = (int)rows_, n = (int)cols_, k = std::min(m, n);
            Matrix cm = colMajor();
            Matrix tau(k, 1, uninit_t{});
            int info = detail::geqrf(m, n, cm.data(), tau.data());
            if (info != 0) throw Error("qr: geqrf failed with info " + std::to_string(info));

            // R first: orgqr overwrites the reflectors that encode Q, so the
            // triangle has to be lifted out before that happens.
            Matrix packed = rowMajorFrom(cm.data(), rows_, cols_);
            Matrix R = packed.block(0, 0, k, n);
            detail::triangle(k, n, R.data(), true, false);

            info = detail::orgqr(m, k, k, cm.data(), tau.data());
            if (info != 0) throw Error("qr: orgqr failed with info " + std::to_string(info));
            // orgqr wrote Q's k columns into the first k*m entries, and in
            // column-major those columns are contiguous — so the prefix is
            // already a valid column-major m x k matrix, no copy needed.
            Matrix Q = rowMajorFrom(cm.data(), rows_, k);
            return {std::move(Q), std::move(R)};
        }

        // SVD, A = U * S * V^T. S is returned as a matrix with the singular
        // values on its diagonal, matching Matrix<datatype>::svd() so the same
        // A == U * S * V.T() identity holds on both sides.
        // S comes back as a Matrix<datatype> DIAGONAL rather than a real
        // vector, deliberately: it keeps A == U * S * V.H() a device-resident
        // identity for complex matrices too, where a real S could not multiply
        // a complex U without leaving the GPU. The values themselves are real
        // and svdvals() returns them as such.
        std::tuple<Matrix, Matrix, Matrix> svd(bool full = true) const {
            requireNonEmpty("svd");
            const int m = (int)rows_, n = (int)cols_, k = std::min(m, n);
            Matrix cm = colMajor();
            const int uCols = full ? m : k;
            const int vtRows = full ? n : k;
            Matrix Ucm(uCols, m, uninit_t{});      // carrier: col-major m x uCols
            Matrix VTcm(n, vtRows, uninit_t{});    // carrier: col-major vtRows x n
            Matrix<real_type> s(k, 1, typename Matrix<real_type>::uninit_t{});
            const int info =
                detail::gesvd(m, n, cm.data(), s.data(), Ucm.data(), VTcm.data(), full);
            if (info != 0)
                throw Error("svd: gesvd failed to converge, info " + std::to_string(info));

            Matrix U = rowMajorFrom(Ucm.data(), m, uCols);
            Matrix VT = rowMajorFrom(VTcm.data(), vtRows, n);
            Matrix S(uCols, vtRows);
            Matrix sc = fromParts(s, nullptr);
            detail::setDiagonal(uCols, vtRows, S.data(), sc.data());
            // V, not V^H: the caller reconstructs with U * S * V.H(), which is
            // the same convention basic/ uses.
            return {std::move(U), std::move(S), VT.H()};
        }

        // Just the singular values, k x 1, descending. Cheaper than svd():
        // cuSOLVER skips forming the orthogonal factors entirely.
        Matrix<real_type> svdvals() const {
            requireNonEmpty("svdvals");
            const int m = (int)rows_, n = (int)cols_, k = std::min(m, n);
            Matrix cm = colMajor();
            Matrix<real_type> s(k, 1, typename Matrix<real_type>::uninit_t{});
            // Null factors tell the backend to pass cuSOLVER 'N'. The earlier
            // version passed two undersized dummy buffers instead, which was a
            // straight overflow: economy mode writes an m x k U, and a 1 x m
            // buffer is k times too small.
            const int info = detail::gesvd(m, n, cm.data(), s.data(), nullptr, nullptr, false);
            if (info != 0)
                throw Error("svdvals: gesvd failed to converge, info " + std::to_string(info));
            return s;
        }

        // Symmetric / Hermitian eigenproblem. Values ascending (cuSOLVER's
        // order, which is also LAPACK's), vectors as COLUMNS of the returned
        // matrix. Reads the lower triangle only.
        // For a complex matrix this is the HERMITIAN eigenproblem (heevd).
        // Its eigenvalues are real even though its eigenvectors are not, which
        // is why the values come back as Matrix<real_type> in both cases.
        std::pair<Matrix<real_type>, Matrix> eigSym(bool vectors = true) const {
            requireSquare("eigSym");
            requireNonEmpty("eigSym");
            const int n = (int)rows_;
            Matrix cm = colMajor();
            Matrix<real_type> w(n, 1, typename Matrix<real_type>::uninit_t{});
            // if constexpr, not a ternary: a ternary would instantiate BOTH
            // calls, and heevd has no real overload.
            int info;
            if constexpr (isComplex) info = detail::heevd(n, cm.data(), w.data(), vectors);
            else info = detail::syevd(n, cm.data(), w.data(), vectors);
            if (info != 0)
                throw Error("eigSym: failed to converge, info " + std::to_string(info));
            Matrix V = vectors ? rowMajorFrom(cm.data(), n, n) : Matrix();
            return {std::move(w), std::move(V)};
        }

        // --- Transposed products ---
        //
        // cuBLAS honours a transpose flag inside the kernel, so A.T() * B need
        // not build A^T first. Whether that matters is entirely a question of
        // shape: measured at 1.00x for a square multiply and 1.05x even for a
        // 2000000x8 Gram matrix, so these are here for clarity at the call
        // site rather than for speed.

        // A^T * B, with A stored as-is.
        Matrix tMul(const Matrix& B) const {
            if (rows_ != B.rows_)
                throw Error("tMul: A^T * B needs A and B to have the same number of rows, got " +
                            std::to_string(rows_) + " and " + std::to_string(B.rows_));
            Matrix out(cols_, B.cols_, uninit_t{});
            detail::gemmT(true, false, (int)cols_, (int)B.cols_, (int)rows_, (int)cols_,
                          (int)B.cols_, datatype(1), data(), B.data(), datatype(0), out.data());
            return out;
        }
        // A * B^T.
        Matrix mulT(const Matrix& B) const {
            if (cols_ != B.cols_)
                throw Error("mulT: A * B^T needs A and B to have the same number of columns, got " +
                            std::to_string(cols_) + " and " + std::to_string(B.cols_));
            Matrix out(rows_, B.rows_, uninit_t{});
            detail::gemmT(false, true, (int)rows_, (int)B.rows_, (int)cols_, (int)cols_,
                          (int)B.cols_, datatype(1), data(), B.data(), datatype(0), out.data());
            return out;
        }
        // A^T * A, the Gram matrix.
        Matrix gram() const { return tMul(*this); }

        // ── General (non-symmetric) eigenproblem ───────────────────────
        //
        // Same two entry points as mcpu::Matrix, with the same contracts:
        // eigvals() always works and returns complex; eig() returns the real
        // pair and REFUSES a complex spectrum, because a real matrix cannot
        // hold it.
        //
        // WHETHER THIS BELONGS ON A GPU AT ALL. The reduction to Hessenberg
        // form parallelises; the QR sweep that follows is sequential and
        // shift-dependent, so this is one of the least GPU-shaped routines in
        // the package. gpu/README.md carries the measurement. Reach for
        // eigSym() instead whenever the matrix is symmetric -- that one has a
        // divide-and-conquer algorithm and is over 100x.

        // Every eigenvalue, complex, as an n x 1 column. Matches
        // mcpu::Matrix::eigvals().
        Matrix<std::complex<real_type>> eigvals() const {
            requireReal("eigvals");
            requireSquare("eigvals");
            requireNonEmpty("eigvals");
            const int n = (int)rows_;
            Matrix cm = colMajor();
            Matrix<real_type> wr(n, 1, typename Matrix<real_type>::uninit_t{});
            Matrix<real_type> wi(n, 1, typename Matrix<real_type>::uninit_t{});
            const int info = detail::geev(n, cm.data(), wr.data(), wi.data(), nullptr, nullptr);
            if (info != 0)
                throw Error("eigvals: the QR iteration failed to converge, info " +
                            std::to_string(info));
            return Matrix<std::complex<real_type>>::fromParts(wr, &wi);
        }

        // Eigenvalues and right eigenvectors, both REAL. Throws when the
        // spectrum is complex, with the same advice mcpu gives -- there is no
        // way to put a conjugate pair into a real matrix, and silently
        // dropping the imaginary part would be worse than refusing.
        std::pair<Matrix, Matrix> eig() const {
            requireReal("eig");
            requireSquare("eig");
            requireNonEmpty("eig");
            const int n = (int)rows_;
            Matrix cm = colMajor();
            Matrix<real_type> wr(n, 1, typename Matrix<real_type>::uninit_t{});
            Matrix<real_type> wi(n, 1, typename Matrix<real_type>::uninit_t{});
            Matrix<real_type> vr(n, n, typename Matrix<real_type>::uninit_t{});
            Matrix<real_type> vi(n, n, typename Matrix<real_type>::uninit_t{});
            const int info =
                detail::geev(n, cm.data(), wr.data(), wi.data(), vr.data(), vi.data());
            if (info != 0)
                throw Error("eig: the QR iteration failed to converge, info " +
                            std::to_string(info));
            // Scaled by the spectral radius, so the test is relative: an
            // eigenvalue of 1e8 + 1e-9i is real, one of 1e-9 + 1e-9i is not.
            const real_type scale = std::max(wr.abs().max(), real_type(1));
            if (wi.abs().max() > real_type(1e-12) * scale)
                throw Error("eig: this matrix has a complex-conjugate eigenvalue pair, which a "
                            "real result cannot represent - use eigvals(), which returns "
                            "Matrix<std::complex<>>");
            return {wr, rowMajorFrom(vr.data(), rows_, cols_)};
        }

        // --- Solving ---

        // Solves A X = B by LU with partial pivoting.
        Matrix solve(const Matrix& B) const {
            requireSquare("solve");
            requireNonEmpty("solve");
            if (B.rows_ != rows_)
                throw Error("solve: A is " + std::to_string(rows_) + "x" +
                               std::to_string(cols_) + " but B has " + std::to_string(B.rows_) +
                               " rows");
            const int n = (int)rows_, nrhs = (int)B.cols_;
            Matrix Acm = colMajor(), Bcm = B.colMajor();
            DeviceBuffer piv(sizeof(int) * (std::size_t)n);
            int info = detail::getrf(n, n, Acm.data(), (int*)piv.get());
            if (info > 0)
                throw Error("solve: exact zero pivot at position " + std::to_string(info) +
                               " - the matrix is singular");
            if (info < 0)
                throw Error("solve: cuSOLVER rejected argument " + std::to_string(-info));
            info = detail::getrs(n, nrhs, Acm.data(), (const int*)piv.get(), Bcm.data());
            if (info != 0) throw Error("solve: getrs failed with info " + std::to_string(info));
            return rowMajorFrom(Bcm.data(), rows_, B.cols_);
        }

        Matrix inv() const { return solve(eye(rows_)); }

        // Determinant as the product of U's diagonal, sign-corrected by the
        // pivot count. Computed from the same LU the solve uses, so a singular
        // matrix throws there rather than silently returning zero here.
        datatype det() const {
            requireSquare("det");
            requireNonEmpty("det");
            const int n = (int)rows_;
            Matrix cm = colMajor();
            DeviceBuffer piv(sizeof(int) * (std::size_t)n);
            const int info = detail::getrf(n, n, cm.data(), (int*)piv.get());
            if (info < 0) throw Error("det: cuSOLVER rejected argument " + std::to_string(-info));
            if (info > 0) return datatype(0);   // exact zero pivot: singular
            Matrix packed = rowMajorFrom(cm.data(), rows_, cols_);
            datatype d = packed.diag().prod();
            std::vector<int> ipiv((std::size_t)n);
            detail::copyD2H(ipiv.data(), piv.get(), sizeof(int) * (std::size_t)n);
            // ipiv is 1-based; an entry differing from its own index is a swap.
            for (int i = 0; i < n; ++i)
                if (ipiv[(std::size_t)i] != i + 1) d = -d;
            return d;
        }

        // Solve by factoring in a LOWER precision and refining the answer back
        // to full accuracy. On this card fp64 runs at 1/64 of fp32, so moving
        // the O(n^3) factorisation to fp32 and leaving only the O(n^2)
        // refinement in fp64 is close to free accuracy:
        //
        //     n=2048   solve() 24.8 ms      solveMixed() 5.6 ms, residual 2e-14
        //
        // cuSOLVER refines until the fp64 residual is as good as a full fp64
        // factorisation would have given, and if that fails to converge it
        // silently redoes the whole thing in fp64 rather than returning a worse
        // answer -- reported as a NEGATIVE refinement count. So this is safe to
        // reach for by default on a well-conditioned system; the failure mode
        // is losing the speedup, not the accuracy.
        //
        // Half and BFloat16 factor even faster and converge on fewer matrices.
        using Factor = detail::Factor;

        Matrix solveMixed(const Matrix& B, Factor f = Factor::Single,
                          int* refinements = nullptr) const {
            requireReal("solveMixed");
            requireSquare("solveMixed");
            requireNonEmpty("solveMixed");
            if (B.rows_ != rows_)
                throw Error("solveMixed: A is " + std::to_string(rows_) + "x" +
                            std::to_string(cols_) + " but B has " + std::to_string(B.rows_) +
                            " rows");
            const int n = (int)rows_, nrhs = (int)B.cols_;
            Matrix Acm = colMajor(), Bcm = B.colMajor();
            Matrix Xcm(B.cols_, rows_, uninit_t{});   // carrier for column-major n x nrhs
            int iters = 0;
            const int info = detail::gesvMixed(n, nrhs, Acm.data(), Bcm.data(), Xcm.data(), f,
                                               &iters);
            if (info > 0)
                throw Error("solveMixed: exact zero pivot at position " + std::to_string(info) +
                            " - the matrix is singular");
            if (info < 0)
                throw Error("solveMixed: cuSOLVER rejected argument " + std::to_string(-info));
            if (refinements) *refinements = iters;
            return rowMajorFrom(Xcm.data(), rows_, B.cols_);
        }

        // --- Factor once, solve many ---
        //
        // solve() runs a fresh getrf on every call, which is the whole cost;
        // the triangular substitution after it is O(n^2) and nearly free. Any
        // algorithm that solves against the same matrix repeatedly -- Newton
        // steps, implicit time stepping, an inverse-power iteration -- should
        // pay for the factorisation once. This mirrors Matrix::factorize() on
        // the CPU side.
        class LU {
          public:
            LU() = default;
            LU(const Matrix& A) : n_(A.rows()) {
                A.requireSquare("factorize");
                A.requireNonEmpty("factorize");
                lu_ = A.colMajor();
                piv_ = DeviceBuffer(sizeof(int) * (std::size_t)n_);
                const int info = detail::getrf((int)n_, (int)n_, lu_.data(), (int*)piv_.get());
                if (info > 0)
                    throw Error("factorize: exact zero pivot at position " + std::to_string(info) +
                                " - the matrix is singular");
                if (info < 0)
                    throw Error("factorize: cuSOLVER rejected argument " + std::to_string(-info));
                hostPiv_.resize((std::size_t)n_);
                detail::copyD2H(hostPiv_.data(), piv_.get(), sizeof(int) * (std::size_t)n_);
            }

            Matrix solve(const Matrix& B) const {
                if (B.rows() != n_)
                    throw Error("LU::solve: factor is " + std::to_string(n_) + "x" +
                                std::to_string(n_) + " but B has " + std::to_string(B.rows()) +
                                " rows");
                Matrix Bcm = B.colMajor();
                const int info = detail::getrs((int)n_, (int)B.cols(), lu_.data(),
                                               (const int*)piv_.get(), Bcm.data());
                if (info != 0)
                    throw Error("LU::solve: getrs failed with info " + std::to_string(info));
                return rowMajorFrom(Bcm.data(), n_, B.cols());
            }

            Matrix inv() const { return solve(Matrix::eye(n_)); }

            datatype det() const {
                Matrix packed = rowMajorFrom(lu_.data(), n_, n_);
                datatype d = packed.diag().prod();
                for (int i = 0; i < (int)n_; ++i)
                    if (hostPiv_[(std::size_t)i] != i + 1) d = -d;
                return d;
            }

            long rows() const { return n_; }

          private:
            Matrix lu_;                  // packed factors, column-major carrier
            DeviceBuffer piv_;
            std::vector<int> hostPiv_;   // n ints; needed on the host for det()
            long n_ = 0;
        };

        LU factorize() const { return LU(*this); }

      private:
        Matrix cmp(const Matrix& o, detail::CmpOp op, const char* what) const {
            requireReal(what);
            requireSame(o, what);
            Matrix out(rows_, cols_, uninit_t{});
            detail::compare(op, size(), data(), o.data(), out.data());
            return out;
        }
        Matrix cmpScalar(detail::CmpOp op, datatype v) const {
            requireReal("comparison");
            Matrix out(rows_, cols_, uninit_t{});
            detail::compareScalar(op, size(), data(), v, out.data());
            return out;
        }
        Matrix rearranged(detail::Rearrange how, long dr, long dc, long p, long q) const {
            Matrix out(dr, dc, uninit_t{});
            detail::rearrange(how, (int)rows_, (int)cols_, data(), (int)p, (int)q, out.data());
            return out;
        }

        Matrix zip(const Matrix& o, detail::BinOp op, const char* what) const {
            requireSame(o, what);
            Matrix out(rows_, cols_, uninit_t{});
            detail::binary(op, size(), data(), o.data(), out.data());
            return out;
        }
        Matrix& zipInto(const Matrix& o, detail::BinOp op, const char* what) {
            requireSame(o, what);
            detail::binary(op, size(), data(), o.data(), data());
            return *this;
        }
        Matrix scalar(detail::BinOp op, datatype s, bool left) const {
            Matrix out(rows_, cols_, uninit_t{});
            detail::binaryScalar(op, size(), data(), s, out.data(), left);
            return out;
        }
        Matrix map(detail::UnOp op) const {
            Matrix out(rows_, cols_, uninit_t{});
            detail::unary(op, size(), data(), out.data());
            return out;
        }
        datatype red(detail::RedOp op, const char* what) const {
            requireNonEmpty(what);
            return detail::reduce(op, size(), data());
        }
        Matrix redAxis(detail::RedOp op, bool byRow) const {
            requireNonEmpty("axis reduction");
            Matrix out(byRow ? rows_ : 1, byRow ? 1 : cols_, uninit_t{});
            detail::reduceAxis(op, (int)rows_, (int)cols_, data(), out.data(), byRow);
            return out;
        }
    };

    // ── Expr: fused element-wise expressions ────────────────────────
    //
    // Holds an unevaluated element-wise expression and turns the whole thing
    // into ONE kernel when it lands in a Matrix. The measured reason it
    // exists, at n=4096 f64:
    //
    //     (A%B).exp().sqrt()   eager  3.24 ms   fused  ~0.9 ms
    //
    // and the arithmetic is identical -- the difference is entirely memory
    // traffic, 7 passes over the data against 3.
    //
    // HOW: each operator appends to a postfix program (push input, push
    // constant, apply unary, apply binary) that one precompiled kernel
    // interprets with a per-thread stack. That is what lets fusion exist in a
    // package whose users have no CUDA compiler -- MatX and cupy.fuse both
    // reach for NVRTC here, which would mean shipping a compiler.
    //
    // LIFETIME. An expression stores raw device pointers to its operands and
    // does not keep them alive. It is built to be consumed in the statement
    // that creates it, exactly like an Armadillo or Eigen expression:
    //
    //     Matrix<double> C = (dA.lazy() % dB).exp();   // fine
    //     auto e = (dA.lazy() % dB).exp();                // fine while dA, dB live
    //     auto bad = (make().lazy() + 1).exp();           // DANGLING: temporary died
    //
    // NOT FUSABLE, and deliberately absent: matmul, transpose, reductions and
    // the factorisations. They are not element-wise -- each needs to see other
    // elements -- so they terminate a chain. Call .eval() and carry on:
    //
    //     auto C = (dA.lazy() % dB).exp().eval() * dC;

    template <typename datatype>
    class Expr {
      public:
        explicit Expr(const Matrix<datatype>& m)
            : rows_(m.rows()), cols_(m.cols()) {
            inputs_[0] = m.data();
            nInputs_ = 1;
            push(detail::FusedCode::PushInput, 0, 0.0);
            depth_ = maxDepth_ = 1;
        }

        long rows() const { return rows_; }
        long cols() const { return cols_; }

        // --- Unary ---

        Expr operator-() const { return withUnary(detail::UnOp::Neg); }
        Expr abs() const { return withUnary(detail::UnOp::Abs); }
        Expr sqrt() const { return withUnary(detail::UnOp::Sqrt); }
        Expr exp() const { return withUnary(detail::UnOp::Exp); }
        Expr ln() const { return withUnary(detail::UnOp::Log); }
        Expr lg() const { return withUnary(detail::UnOp::Log2); }
        Expr log10() const { return withUnary(detail::UnOp::Log10); }
        Expr exp2() const { return withUnary(detail::UnOp::Exp2); }
        Expr sin() const { return withUnary(detail::UnOp::Sin); }
        Expr cos() const { return withUnary(detail::UnOp::Cos); }
        Expr tan() const { return withUnary(detail::UnOp::Tan); }
        Expr asin() const { return withUnary(detail::UnOp::Asin); }
        Expr acos() const { return withUnary(detail::UnOp::Acos); }
        Expr atan() const { return withUnary(detail::UnOp::Atan); }
        Expr sinh() const { return withUnary(detail::UnOp::Sinh); }
        Expr cosh() const { return withUnary(detail::UnOp::Cosh); }
        Expr tanh() const { return withUnary(detail::UnOp::Tanh); }
        Expr floor() const { return withUnary(detail::UnOp::Floor); }
        Expr ceil() const { return withUnary(detail::UnOp::Ceil); }
        Expr round() const { return withUnary(detail::UnOp::Round); }
        Expr sign() const { return withUnary(detail::UnOp::Sign); }
        Expr pow2() const { return withUnary(detail::UnOp::Square); }
        Expr asinh() const { return withUnary(detail::UnOp::Asinh); }
        Expr acosh() const { return withUnary(detail::UnOp::Acosh); }
        Expr atanh() const { return withUnary(detail::UnOp::Atanh); }
        Expr cbrt() const { return withUnary(detail::UnOp::Cbrt); }
        Expr log1p() const { return withUnary(detail::UnOp::Log1p); }
        Expr expm1() const { return withUnary(detail::UnOp::Expm1); }
        Expr trunc() const { return withUnary(detail::UnOp::Trunc); }
        Expr atan2(const Expr& r) const { return join(r, detail::BinOp::Atan2); }
        Expr hypot(const Expr& r) const { return join(r, detail::BinOp::Hypot); }
        Expr mod(const Expr& r) const { return join(r, detail::BinOp::Mod); }
        Expr log(datatype base) const {
            return lg() * datatype(1.0 / std::log2((double)base));
        }

        // --- Binary ---
        //
        // Each comes in three forms so a chain can mix freely: expression on
        // both sides, a plain matrix on one side, or a scalar.

        Expr operator+(const Expr& r) const { return join(r, detail::BinOp::Add); }
        Expr operator-(const Expr& r) const { return join(r, detail::BinOp::Sub); }
        Expr operator%(const Expr& r) const { return join(r, detail::BinOp::Mul); }
        Expr operator/(const Expr& r) const { return join(r, detail::BinOp::Div); }
        Expr emax(const Expr& r) const { return join(r, detail::BinOp::Max); }
        Expr emin(const Expr& r) const { return join(r, detail::BinOp::Min); }
        Expr pow(const Expr& r) const { return join(r, detail::BinOp::Pow); }

        Expr operator+(const Matrix<datatype>& r) const { return *this + Expr(r); }
        Expr operator-(const Matrix<datatype>& r) const { return *this - Expr(r); }
        Expr operator%(const Matrix<datatype>& r) const { return *this % Expr(r); }
        Expr operator/(const Matrix<datatype>& r) const { return *this / Expr(r); }
        Expr emax(const Matrix<datatype>& r) const { return emax(Expr(r)); }
        Expr emin(const Matrix<datatype>& r) const { return emin(Expr(r)); }

        Expr operator+(datatype s) const { return withScalar(detail::BinOp::Add, s); }
        Expr operator-(datatype s) const { return withScalar(detail::BinOp::Sub, s); }
        Expr operator*(datatype s) const { return withScalar(detail::BinOp::Mul, s); }
        Expr operator/(datatype s) const { return withScalar(detail::BinOp::Div, s); }
        Expr pow(datatype e) const { return withScalar(detail::BinOp::Pow, e); }

        // --- Termination ---

        Matrix<datatype> eval() const {
            Matrix<datatype> out(rows_, cols_, typename Matrix<datatype>::uninit_t{});
            detail::FusedProgram p = prog_;
            p.maxDepth = maxDepth_;   // picks the register-only kernel when <= 2
            detail::fusedElementwise(p, (std::size_t)rows_ * (std::size_t)cols_, inputs_,
                                     nInputs_, out.data());
            return out;
        }
        operator Matrix<datatype>() const { return eval(); }

        // How many ops and inputs the chain compiled to. Exposed because it is
        // the only way to confirm from outside that a chain actually fused
        // rather than silently spilling; the test suite checks it.
        int ops() const { return prog_.nOps; }
        int inputs() const { return nInputs_; }
        int spills() const { return (int)owned_.size(); }

      private:
        void push(detail::FusedCode c, int arg, double imm) {
            prog_.code[prog_.nOps] = (int)c;
            prog_.arg[prog_.nOps] = arg;
            prog_.imm[prog_.nOps] = imm;
            ++prog_.nOps;
        }

        // Returns the slot holding p, adding it only if it is not already
        // there. Deduplicating matters: (dA.lazy() % dA) reads one buffer, not
        // two, and an expression reusing the same operand five times still
        // costs one input slot.
        int slotFor(const datatype* p) {
            for (int i = 0; i < nInputs_; ++i)
                if (inputs_[i] == p) return i;
            inputs_[nInputs_] = p;
            return nInputs_++;
        }

        void requireSame(const Expr& r) const {
            if (rows_ != r.rows_ || cols_ != r.cols_)
                throw Error("fused expression: dimension mismatch (" + std::to_string(rows_) +
                               "x" + std::to_string(cols_) + ") vs (" + std::to_string(r.rows_) +
                               "x" + std::to_string(r.cols_) + ")");
        }

        // Collapses the expression so far into a real matrix and restarts the
        // program from it. This is what keeps an arbitrarily long chain
        // working instead of hitting a hard cliff at MaxOps: a chain too big
        // for one kernel becomes two, which is still better than one kernel
        // per step. The spilled matrix is kept alive by owned_, because unlike
        // a user's operand nothing else refers to it.
        void spill() {
            auto held = std::make_shared<Matrix<datatype>>(eval());
            prog_ = detail::FusedProgram{};
            nInputs_ = 0;
            inputs_[0] = held->data();
            nInputs_ = 1;
            push(detail::FusedCode::PushInput, 0, 0.0);
            depth_ = maxDepth_ = 1;
            owned_.push_back(std::move(held));
        }

        Expr withUnary(detail::UnOp op) const {
            Expr out = *this;
            if (out.prog_.nOps + 1 > detail::FusedProgram::MaxOps) out.spill();
            out.push(detail::FusedCode::ApplyUnary, (int)op, 0.0);
            return out;
        }

        Expr withScalar(detail::BinOp op, datatype s) const {
            Expr out = *this;
            if (out.prog_.nOps + 2 > detail::FusedProgram::MaxOps ||
                out.depth_ + 1 > detail::FusedProgram::StackDepth)
                out.spill();
            out.push(detail::FusedCode::PushScalar, 0, (double)s);
            out.push(detail::FusedCode::ApplyBinary, (int)op, 0.0);
            out.maxDepth_ = std::max(out.maxDepth_, out.depth_ + 1);
            return out;
        }

        // Postfix concatenation: [lhs][rhs][op] evaluates correctly because
        // lhs leaves exactly one value on the stack before rhs begins. The
        // stack therefore peaks at max(lhs peak, 1 + rhs peak).
        Expr join(const Expr& r, detail::BinOp op) const {
            requireSame(r);
            Expr out = *this;
            Expr rhs = r;

            // Spill whichever side would overflow. Spilling the right-hand
            // side first is usually enough, since it collapses to one push.
            if (out.prog_.nOps + rhs.prog_.nOps + 1 > detail::FusedProgram::MaxOps ||
                out.nInputs_ + rhs.nInputs_ > detail::FusedProgram::MaxInputs ||
                1 + rhs.maxDepth_ > detail::FusedProgram::StackDepth)
                rhs.spill();
            if (out.prog_.nOps + rhs.prog_.nOps + 1 > detail::FusedProgram::MaxOps ||
                out.nInputs_ + rhs.nInputs_ > detail::FusedProgram::MaxInputs)
                out.spill();

            for (auto& h : rhs.owned_) out.owned_.push_back(h);
            for (int k = 0; k < rhs.prog_.nOps; ++k) {
                const int code = rhs.prog_.code[k];
                int arg = rhs.prog_.arg[k];
                if (code == (int)detail::FusedCode::PushInput)
                    arg = out.slotFor(rhs.inputs_[arg]);
                out.push((detail::FusedCode)code, arg, rhs.prog_.imm[k]);
            }
            out.push(detail::FusedCode::ApplyBinary, (int)op, 0.0);
            out.maxDepth_ = std::max(out.maxDepth_, 1 + rhs.maxDepth_);
            out.depth_ = 1;
            return out;
        }

        detail::FusedProgram prog_;
        const datatype* inputs_[detail::FusedProgram::MaxInputs] = {};
        int nInputs_ = 0;
        long rows_ = 0, cols_ = 0;
        int depth_ = 0, maxDepth_ = 0;
        std::vector<std::shared_ptr<Matrix<datatype>>> owned_;
    };

    template <typename datatype>
    Expr<datatype> Matrix<datatype>::lazy() const {
        return Expr<datatype>(*this);
    }

    // Scalar on the left, and a plain matrix on the left of a lazy chain.
    template <typename datatype>
    Expr<datatype> operator*(datatype s, const Expr<datatype>& e) { return e * s; }
    template <typename datatype>
    Expr<datatype> operator+(datatype s, const Expr<datatype>& e) { return e + s; }
    template <typename datatype>
    Expr<datatype> operator+(const Matrix<datatype>& m, const Expr<datatype>& e) {
        return Expr<datatype>(m) + e;
    }
    template <typename datatype>
    Expr<datatype> operator-(const Matrix<datatype>& m, const Expr<datatype>& e) {
        return Expr<datatype>(m) - e;
    }
    template <typename datatype>
    Expr<datatype> operator%(const Matrix<datatype>& m, const Expr<datatype>& e) {
        return Expr<datatype>(m) % e;
    }
    template <typename datatype>
    Expr<datatype> operator/(const Matrix<datatype>& m, const Expr<datatype>& e) {
        return Expr<datatype>(m) / e;
    }

    template <typename datatype>
    Matrix<datatype> eval(const Expr<datatype>& e) { return e.eval(); }

    // ── Spectrum: complex data, as two real matrices ───────────────────
    //
    // mgpu::Matrix is real-only, so a transform's output travels as separate
    // real and imaginary parts rather than as a complex matrix. That is not a
    // workaround: split arrays are what the rest of this package can operate
    // on, so a spectrum stays usable -- filtered, masked, fused -- without
    // leaving the device or needing a complex Matrix that does not exist.
    //
    // .cpu() hands back an mcpu::Matrix<std::complex<T>>, which is exactly what
    // basic/signal.hpp's fft() returns, so the two are directly comparable.

    template <typename datatype>
    struct Spectrum {
        Matrix<datatype> re, im;

        long rows() const { return re.rows(); }
        long cols() const { return re.cols(); }

        // Magnitude and power. Written through .lazy() so each is a single
        // kernel over both parts rather than three.
        Matrix<datatype> abs() const { return (re.lazy().pow2() + im.lazy().pow2()).sqrt(); }
        Matrix<datatype> power() const { return (re.lazy().pow2() + im.lazy().pow2()).eval(); }
        Spectrum conj() const { return {re, -im}; }

        // Complex element-wise product: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
        // This is what turns a pair of transforms into a convolution.
        Spectrum operator%(const Spectrum& o) const {
            return {((re.lazy() % o.re) - (im.lazy() % o.im)).eval(),
                    ((re.lazy() % o.im) + (im.lazy() % o.re)).eval()};
        }
        Spectrum operator*(datatype s) const { return {re * s, im * s}; }

        // The interleaved form, for handing a spectrum to anything that wants
        // a complex matrix -- including fft() itself.
        Matrix<std::complex<datatype>> toComplex() const {
            return Matrix<std::complex<datatype>>::fromParts(re, &im);
        }

        mcpu::Matrix<std::complex<datatype>> cpu() const {
            mcpu::Matrix<datatype> hr = re.cpu(), hi = im.cpu();
            mcpu::Matrix<std::complex<datatype>> out(hr.rows(), hr.cols());
            for (long i = 0; i < hr.rows(); ++i)
                for (long j = 0; j < hr.cols(); ++j)
                    out(i, j) = std::complex<datatype>(hr(i, j), hi(i, j));
            return out;
        }
    };

    // ── FFT ────────────────────────────────────────────────────────────
    //
    // Axis convention is basic/signal.hpp's, which is MATLAB's: false (COL)
    // works DOWN COLUMNS, true (ROW) works ALONG ROWS, and the no-axis form
    // picks the one a vector obviously wants and columns for a matrix.
    //
    // Both directions run without transposing anything: cuFFT takes a stride
    // and a batch distance, and row-major storage makes along-rows contiguous
    // (stride 1, distance cols) and down-columns strided (stride cols,
    // distance 1). See the note in detail/backend.hpp.

    namespace fftaxis {
        // A row vector transforms along itself; anything else defaults to
        // columns, matching MATLAB and basic/signal.hpp.
        inline bool automatic(long rows, long) { return rows == 1; }

        struct Layout {
            int batch, n, stride, dist;
        };
        inline Layout of(long rows, long cols, bool alongRows) {
            if (alongRows) return {(int)rows, (int)cols, 1, (int)cols};
            return {(int)cols, (int)rows, (int)cols, 1};
        }
    }  // namespace fftaxis

    template <typename datatype>
    Spectrum<datatype> fft(const Spectrum<datatype>& x, bool alongRows) {
        const auto L = fftaxis::of(x.rows(), x.cols(), alongRows);
        Spectrum<datatype> out{Matrix<datatype>(x.rows(), x.cols(),
                                                typename Matrix<datatype>::uninit_t{}),
                               Matrix<datatype>(x.rows(), x.cols(),
                                                typename Matrix<datatype>::uninit_t{})};
        detail::fft1d(L.batch, L.n, L.stride, L.dist, x.re.data(), x.im.data(), out.re.data(),
                      out.im.data(), false);
        return out;
    }

    // A REAL input gives a Spectrum (split parts); a COMPLEX input gives a
    // complex Matrix, matching basic/signal.hpp's fft() exactly. The return
    // type therefore depends on the argument, which is what `auto` plus
    // `if constexpr` is for -- and the complex path is the cheaper one, since
    // cuFFT wants interleaved complex and that is already the storage.
    template <typename datatype>
    auto fft(const Matrix<datatype>& x, bool alongRows) {
        const auto L = fftaxis::of(x.rows(), x.cols(), alongRows);
        if constexpr (is_complex_v<datatype>) {
            Matrix<datatype> out(x.rows(), x.cols(), typename Matrix<datatype>::uninit_t{});
            detail::fft1dCx(L.batch, L.n, L.stride, L.dist, x.data(), out.data(), false);
            return out;
        } else {
            Spectrum<datatype> out{Matrix<datatype>(x.rows(), x.cols(),
                                                    typename Matrix<datatype>::uninit_t{}),
                                   Matrix<datatype>(x.rows(), x.cols(),
                                                    typename Matrix<datatype>::uninit_t{})};
            // A null imaginary pointer tells the backend the input is purely
            // real, so no zero matrix is built just to be read once.
            detail::fft1d(L.batch, L.n, L.stride, L.dist, x.data(), (const datatype*)nullptr,
                          out.re.data(), out.im.data(), false);
            return out;
        }
    }
    template <typename datatype>
    auto fft(const Matrix<datatype>& x) {
        return fft(x, fftaxis::automatic(x.rows(), x.cols()));
    }

    // Complex inverse, complex out. The real-input inverse is ifft(Spectrum)
    // further down.
    template <typename datatype,
              typename = std::enable_if_t<is_complex_v<datatype>>>
    Matrix<datatype> ifft(const Matrix<datatype>& X, bool alongRows) {
        const auto L = fftaxis::of(X.rows(), X.cols(), alongRows);
        Matrix<datatype> out(X.rows(), X.cols(), typename Matrix<datatype>::uninit_t{});
        detail::fft1dCx(L.batch, L.n, L.stride, L.dist, X.data(), out.data(), true);
        return out;
    }
    template <typename datatype,
              typename = std::enable_if_t<is_complex_v<datatype>>>
    Matrix<datatype> ifft(const Matrix<datatype>& X) {
        return ifft(X, fftaxis::automatic(X.rows(), X.cols()));
    }

    // Full complex inverse.
    template <typename datatype>
    Spectrum<datatype> ifftc(const Spectrum<datatype>& X, bool alongRows) {
        const auto L = fftaxis::of(X.rows(), X.cols(), alongRows);
        Spectrum<datatype> out{Matrix<datatype>(X.rows(), X.cols(),
                                                typename Matrix<datatype>::uninit_t{}),
                               Matrix<datatype>(X.rows(), X.cols(),
                                                typename Matrix<datatype>::uninit_t{})};
        detail::fft1d(L.batch, L.n, L.stride, L.dist, X.re.data(), X.im.data(), out.re.data(),
                      out.im.data(), true);
        return out;
    }
    template <typename datatype>
    Spectrum<datatype> ifftc(const Spectrum<datatype>& X) {
        return ifftc(X, fftaxis::automatic(X.rows(), X.cols()));
    }

    // Real inverse: the imaginary part of a conjugate-symmetric spectrum is
    // rounding error, and callers almost always want it dropped.
    template <typename datatype>
    Matrix<datatype> ifft(const Spectrum<datatype>& X, bool alongRows) {
        return ifftc(X, alongRows).re;
    }
    template <typename datatype>
    Matrix<datatype> ifft(const Spectrum<datatype>& X) {
        return ifftc(X).re;
    }

    template <typename datatype>
    auto fft2(const Matrix<datatype>& x) {
        if constexpr (is_complex_v<datatype>) {
            Matrix<datatype> out(x.rows(), x.cols(), typename Matrix<datatype>::uninit_t{});
            detail::fft2dCx((int)x.rows(), (int)x.cols(), x.data(), out.data(), false);
            return out;
        } else {
            Spectrum<datatype> out{Matrix<datatype>(x.rows(), x.cols(),
                                                    typename Matrix<datatype>::uninit_t{}),
                                   Matrix<datatype>(x.rows(), x.cols(),
                                                    typename Matrix<datatype>::uninit_t{})};
            detail::fft2d((int)x.rows(), (int)x.cols(), x.data(), (const datatype*)nullptr,
                          out.re.data(), out.im.data(), false);
            return out;
        }
    }
    template <typename datatype,
              typename = std::enable_if_t<is_complex_v<datatype>>>
    Matrix<datatype> ifft2(const Matrix<datatype>& X) {
        Matrix<datatype> out(X.rows(), X.cols(), typename Matrix<datatype>::uninit_t{});
        detail::fft2dCx((int)X.rows(), (int)X.cols(), X.data(), out.data(), true);
        return out;
    }
    template <typename datatype>
    Spectrum<datatype> ifft2c(const Spectrum<datatype>& X) {
        Spectrum<datatype> out{Matrix<datatype>(X.rows(), X.cols(),
                                                typename Matrix<datatype>::uninit_t{}),
                               Matrix<datatype>(X.rows(), X.cols(),
                                                typename Matrix<datatype>::uninit_t{})};
        detail::fft2d((int)X.rows(), (int)X.cols(), X.re.data(), X.im.data(), out.re.data(),
                      out.im.data(), true);
        return out;
    }
    template <typename datatype>
    Matrix<datatype> ifft2(const Spectrum<datatype>& X) {
        return ifft2c(X).re;
    }

    // fftshift moves the zero frequency from index 0 to the middle. For an ODD
    // length the halves differ by one, so this and ifftshift are NOT the same
    // operation -- ifftshift is the exact inverse. Same split as basic/.
    namespace fftshift_detail {
        template <typename datatype>
        Matrix<datatype> roll(const Matrix<datatype>& A, bool inverse) {
            const long r = A.rows(), c = A.cols();
            Matrix<datatype> out(r, c, typename Matrix<datatype>::uninit_t{});
            // For ODD n the two halves differ by one, and which half moves is
            // the entire difference between fftshift and its inverse:
            // fftshift([0,1,2,3,4]) = [3,4,0,1,2] splits after n-n/2 = 3,
            // ifftshift gives [2,3,4,0,1] and splits after n/2 = 2. For even n
            // the two coincide, which is why an even-length test cannot tell
            // them apart.
            auto half = [&](long n) { return inverse ? n / 2 : n - n / 2; };
            const long hr = (r > 1) ? half(r) : 0;
            const long hc = (c > 1) ? half(c) : 0;
            // Four quadrants swapped diagonally; degenerate sizes fall out as
            // empty blocks that setBlock skips.
            out.setBlock(r - hr, c - hc, A.block(0, 0, hr, hc));
            out.setBlock(r - hr, 0, A.block(0, hc, hr, c - hc));
            out.setBlock(0, c - hc, A.block(hr, 0, r - hr, hc));
            out.setBlock(0, 0, A.block(hr, hc, r - hr, c - hc));
            return out;
        }
    }  // namespace fftshift_detail

    template <typename datatype>
    Matrix<datatype> fftshift(const Matrix<datatype>& A) {
        return fftshift_detail::roll(A, false);
    }
    template <typename datatype>
    Matrix<datatype> ifftshift(const Matrix<datatype>& A) {
        return fftshift_detail::roll(A, true);
    }

    // Linear convolution, by transform. Matches basic/signal.hpp's conv():
    // both operands are treated as vectors and the result is 1 x (na + nb - 1).
    template <typename datatype>
    Matrix<datatype> conv(const Matrix<datatype>& a, const Matrix<datatype>& b) {
        const long na = (long)a.size(), nb = (long)b.size();
        if (!na || !nb) throw Error("conv: both operands must be non-empty");
        const long nc = na + nb - 1;
        // Zero-pad both to the output length so the circular convolution the
        // transform computes equals the linear one that was asked for.
        Matrix<datatype> ap = a.resized(1, nc), bp = b.resized(1, nc);
        Spectrum<datatype> P = fft(ap, true) % fft(bp, true);
        return ifft(P, true);
    }

    // ── Free functions ─────────────────────────────────────────────────

    // The upload, spelled as a verb so the crossing is visible at the call
    // site. mgpu::upload(A) and dA.cpu() should be the only two places in any
    // program where data changes address space.
    //
    // Equivalent to the explicit constructor, mgpu::Matrix<double> dA(A); the
    // free function exists so `auto` can be used without losing the fact that
    // a bus transfer just happened.
    // ── Builders ───────────────────────────────────────────────────────
    //
    // Built ON the device, so a grid or a sweep never crosses the bus at all.
    // Shapes match basic/: a 1 x n ROW, both endpoints included.

    template <typename datatype = double>
    Matrix<datatype> linspace(datatype lo, datatype hi, long n) {
        if (n < 0) throw Error("linspace: negative count");
        Matrix<datatype> out(1, n, typename Matrix<datatype>::uninit_t{});
        detail::linspace((std::size_t)n, out.data(), lo, hi, false);
        return out;
    }

    // Powers of ten from 10^lo to 10^hi, matching MATLAB and basic/ -- the
    // bounds are EXPONENTS, not values.
    template <typename datatype = double>
    Matrix<datatype> logspace(datatype lo, datatype hi, long n) {
        if (n < 0) throw Error("logspace: negative count");
        Matrix<datatype> out(1, n, typename Matrix<datatype>::uninit_t{});
        detail::linspace((std::size_t)n, out.data(), lo, hi, true);
        return out;
    }

    // Half-open [lo, hi) with a fixed step, the way a for-loop runs -- unlike
    // linspace, which is closed at both ends and counts points instead.
    template <typename datatype = double>
    Matrix<datatype> range(datatype lo, datatype hi, datatype step = datatype(1)) {
        if (step == datatype(0)) throw Error("range: step must be non-zero");
        long n = (long)std::ceil((double)(hi - lo) / (double)step);
        if (n < 0) n = 0;
        Matrix<datatype> out(1, n, typename Matrix<datatype>::uninit_t{});
        if (n > 0)
            detail::linspace((std::size_t)n, out.data(), lo, lo + step * datatype(n - 1), false);
        return out;
    }

    template <typename datatype>
    Matrix<datatype> upload(const mcpu::Matrix<datatype>& host) {
        return Matrix<datatype>(host);
    }

    template <typename datatype>
    Matrix<datatype> operator*(datatype s, const Matrix<datatype>& m) {
        return m * s;
    }
    template <typename datatype>
    Matrix<datatype> operator+(datatype s, const Matrix<datatype>& m) {
        return m + s;
    }

    template <typename datatype>
    datatype sum(const Matrix<datatype>& m) {
        return m.sum();
    }
    template <typename datatype>
    Matrix<datatype> sum(const Matrix<datatype>& m, bool axis) {
        return m.sum(axis);
    }

    template <typename datatype>
    std::ostream& operator<<(std::ostream& os, const Matrix<datatype>& m) {
        return os << m.cpu();
    }

}  // namespace mgpu
