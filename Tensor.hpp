#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  Tensor<T> — N-dimensional arrays for MatrixCpp
// ═══════════════════════════════════════════════════════════════════════════
//
// Include this INSTEAD of Matrix1.0.hpp; it pulls the matrix header in and adds
// the N-dimensional layer on top. C++17, header-only, no third-party anything —
// the same rules as the rest of the project.
//
// ── WHY THIS LAYOUT ────────────────────────────────────────────────────────
// One flat buffer plus shape and stride metadata, which is what NumPy, PyTorch
// and TensorFlow all do. The two alternatives were measured on a
// (32, 32, 64, 64) tensor (4.19M doubles, 33.6 MB) before this was written:
//
//     operation           Matrix<Matrix<double>>  vector<Matrix<double>>      flat
//     allocate                        11.77 ms            ~same         1.91 ms
//     element-wise add                14.22 ms         13.29 ms         3.83 ms
//     sum all                          0.83 ms          0.79 ms         0.43 ms
//     allocations                         1024             1024               1
//     usable as a GEMM                      no               no  173 GFLOP/s
//
// Nesting Matrix inside Matrix compiles, and addition even works, which makes it
// more tempting than it should be. It fails structurally on multiplication:
// the accumulator starts as datatype(0), which for a nested element is a 0x0
// matrix, so the first `0x0 += 64x64` throws. A generic algorithm needs a zero
// of the right SHAPE, and a nested element type cannot supply one.
//
// The decisive argument is not the 3.7x though. It is that every fast tensor
// contraction in every library is
//
//     reshape  ->  permute  ->  2-D GEMM  ->  permute back
//
// With a flat buffer, reshape and permute are FREE (metadata only, no data
// moves) and the GEMM is mstore::gemm from Matrix1.0.hpp, unmodified, at
// 196 GFLOP/s. With nesting neither step is expressible without first copying
// everything into a flat buffer — at which point you have built this the slow
// way. For quantum circuits that IS the operation: applying a k-qubit gate to an
// n-qubit state is reshape, permute the target axes forward, multiply by the
// 2^k x 2^k gate, permute back.
//
// ── WHAT IS SHARED WITH Matrix ─────────────────────────────────────────────
// Everything expensive, through namespace mstore:
//   * mstore::rawAlloc / rawFree — constructor-free storage, so a complex
//     tensor is not zero-filled before you overwrite it, and
//   * mstore::adviseHuge — NumPy's madvise(MADV_HUGEPAGE) on buffers over 4 MB,
//     worth 4x on anything memory-bound,
//   * mstore::gemm / gemmAcc — the blocked, restrict-qualified, OpenMP multiply,
//   * mstore::forEachIndex — threaded element-wise loops with the measured
//     work thresholds, and memoryThreads() for the memory-bound ones,
//   * pairwiseSum — NumPy's reduction, eight accumulator chains and
//     O(log n · eps) error growth.
// There is one implementation of each, not two that drift apart.
//
// ── COPY SEMANTICS, read this once ─────────────────────────────────────────
// Matrix is a value: copying it deep-copies. Tensor keeps that for ordinary
// copies, and adds VIEWS, which is the whole point of the stride design:
//
//     Tensor<double> B = A;                 // deep copy, independent
//     Tensor<double> V = A.permute({1,0});  // VIEW — shares A's buffer, O(1)
//     auto C = V.clone();                   // explicit deep copy of a view
//
// reshape(), permute(), swapAxes(), T() and slice() return views. They cost
// nothing and they alias: writing through a view writes through to the original.
// Copy-assigning or copy-constructing from one materialises it into an
// independent contiguous tensor, so the aliasing never outlives an explicit
// view variable. The buffer is reference counted, so a view keeps its storage
// alive even if the tensor it came from is destroyed.
//
// ── SYNTAX ─────────────────────────────────────────────────────────────────
//     Tensor<double> A(2, 3, 4);        // variadic shape, zero-filled
//     A(1, 2, 3) = 5.0;                 // variadic indexing
//     A.rank(); A.size(); A.shape(0);
//
//     A + B      A - B      -A          element-wise, shapes must match
//     A.mul(B)   A % B    A *dot* B  element-wise product  (MATLAB .*)
//     A.div(B)            A /dot/ B  element-wise division (MATLAB ./)
//     A * 2.0    A / 2.0                scalar
//     A * B                             CONTRACTION over A's last axis and B's
//                                       first — for rank 2 that is exactly the
//                                       matrix product, same as Matrix
//
//     A.reshape(6, 4)     A.permute({1,0,2})    A.swapAxes(0,2)    A.T()
//     A.contiguous()      A.clone()             A.toMatrix()
//     A.sum()   A.sum(axis)   A.min()  A.max()  A.mean()
//     contract(A, B, k)                 tensordot: A's last k with B's first k
//     contractInto(A, B, k, out)        same, into storage you already own
// ═══════════════════════════════════════════════════════════════════════════

#include "Matrix1.0.hpp"
#include <memory>
#include <initializer_list>

template <typename datatype>
class Tensor {
public:
    // ── Construction ───────────────────────────────────────────────────────

    Tensor() = default;   // rank 0, size 0 — the empty tensor

    // Tensor<double> A(2, 3, 4);
    // Deliberately constrained to integral arguments so it cannot hijack the
    // copy constructor, and so Tensor<double> A(someOtherTensor) is a copy.
    template <typename... Dims,
              typename = std::enable_if_t<(sizeof...(Dims) > 0) &&
                                          (std::is_integral<Dims>::value && ...)>>
    explicit Tensor(Dims... dims) : Tensor(std::vector<long>{ long(dims)... }) {}

    // Tensor<double> A({2, 3, 4});  — for a shape computed at runtime
    explicit Tensor(std::vector<long> shape) {
        for (long d : shape)
            if (d < 0) throw std::invalid_argument(
                "Tensor: dimensions must be non-negative, got " + std::to_string(d));
        shape_   = std::move(shape);
        strides_ = rowMajorStrides(shape_);
        size_    = numel(shape_);
        allocZeroed(size_);
    }

    // ── Rule of five ───────────────────────────────────────────────────────
    // Copying MATERIALISES: the result is always independent and contiguous,
    // even when the source was a strided view. That is what keeps view aliasing
    // confined to variables the caller explicitly made with a view method.
    Tensor(const Tensor& other) { assignFrom(other); }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) assignFrom(other);
        return *this;
    }

    // Moving preserves whatever the source was, view or owner. noexcept so that
    // std::vector<Tensor> moves on reallocation rather than copying.
    Tensor(Tensor&& other) noexcept
        : buf_(std::move(other.buf_)), data_(other.data_),
          shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            buf_     = std::move(other.buf_);
            data_    = other.data_;
            shape_   = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_    = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~Tensor() = default;   // shared_ptr releases through mstore::rawFree

    // ── Shape ──────────────────────────────────────────────────────────────

    long rank() const { return (long)shape_.size(); }
    long size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const std::vector<long>& shape() const { return shape_; }
    const std::vector<long>& strides() const { return strides_; }

    long shape(long axis) const {
        return shape_[(std::size_t)checkAxis(axis, "shape")];
    }
    long stride(long axis) const {
        return strides_[(std::size_t)checkAxis(axis, "stride")];
    }

    // True when the elements sit in memory in row-major order with no gaps —
    // i.e. when the flat fast paths and mstore::gemm can be used directly.
    bool isContiguous() const {
        long expect = 1;
        for (long k = rank() - 1; k >= 0; k--) {
            if (shape_[(std::size_t)k] == 1) continue;   // a length-1 axis can
            if (strides_[(std::size_t)k] != expect) return false;  // hold any stride
            expect *= shape_[(std::size_t)k];
        }
        return true;
    }

    // ── Indexing ───────────────────────────────────────────────────────────
    // A(i, j, k) — one index per axis, checked against the rank.

    // Constrained to integral arguments so that the logical-index overloads
    // below (which take a Tensor<bool>) are not shadowed by this template.
    template <typename... Idx,
              typename = std::enable_if_t<(std::is_integral<Idx>::value && ...)>>
    datatype& operator()(Idx... idx) {
        return data_[offsetOf(idx...)];
    }
    template <typename... Idx,
              typename = std::enable_if_t<(std::is_integral<Idx>::value && ...)>>
    const datatype& operator()(Idx... idx) const {
        return data_[offsetOf(idx...)];
    }

    datatype& at(const std::vector<long>& idx)             { return data_[offsetOfVec(idx)]; }
    const datatype& at(const std::vector<long>& idx) const { return data_[offsetOfVec(idx)]; }

    // Flat access in memory order. Only meaningful on a contiguous tensor, and
    // says so rather than quietly reading the wrong element of a view.
    datatype& operator[](long i) {
        requireContiguous("operator[]");
        return data_[i];
    }
    const datatype& operator[](long i) const {
        requireContiguous("operator[]");
        return data_[i];
    }

    // Raw storage, for handing to a kernel. Contiguous tensors only.
    datatype* data() { requireContiguous("data"); return data_; }
    const datatype* data() const { requireContiguous("data"); return data_; }

    // ── Views: all O(1), all share storage ─────────────────────────────────

    // Same elements, new shape. One dimension may be -1 and is inferred.
    template <typename... Dims,
              typename = std::enable_if_t<(std::is_integral<Dims>::value && ...)>>
    Tensor reshape(Dims... dims) const {
        return reshape(std::vector<long>{ long(dims)... });
    }

    Tensor reshape(std::vector<long> newShape) const {
        // Resolve a single -1 against the element count.
        long known = 1; long freeAxis = -1;
        for (std::size_t k = 0; k < newShape.size(); k++) {
            if (newShape[k] == -1) {
                if (freeAxis >= 0) throw std::invalid_argument(
                    "reshape: at most one dimension may be -1");
                freeAxis = (long)k;
            } else if (newShape[k] < 0) {
                throw std::invalid_argument("reshape: dimensions must be >= 0 or -1");
            } else {
                known *= newShape[k];
            }
        }
        if (freeAxis >= 0) {
            if (known == 0 || size_ % known != 0) throw std::invalid_argument(
                "reshape: cannot infer a dimension for " + shapeStr(shape_) +
                " -> " + shapeStr(newShape));
            newShape[(std::size_t)freeAxis] = size_ / known;
        } else if (numel(newShape) != size_) {
            throw std::invalid_argument(
                "reshape: " + shapeStr(shape_) + " has " + std::to_string(size_) +
                " elements, " + shapeStr(newShape) + " needs " +
                std::to_string(numel(newShape)));
        }

        // Reshaping only rewrites metadata, which is only valid if the elements
        // are already in row-major order. A strided view has to be materialised
        // first — that is a real copy, and the only place reshape is not free.
        if (!isContiguous()) return contiguous().reshape(std::move(newShape));

        Tensor out;
        out.buf_     = buf_;
        out.data_    = data_;
        out.shape_   = std::move(newShape);
        out.strides_ = rowMajorStrides(out.shape_);
        out.size_    = size_;
        return out;
    }

    // Reorder the axes. permute({1,0,2}) makes axis 1 the new axis 0.
    Tensor permute(const std::vector<long>& order) const {
        if ((long)order.size() != rank()) throw std::invalid_argument(
            "permute: expected " + std::to_string(rank()) + " axes, got " +
            std::to_string(order.size()));
        std::vector<bool> seen((std::size_t)rank(), false);
        for (long a : order) {
            if (a < 0 || a >= rank()) throw std::invalid_argument(
                "permute: axis " + std::to_string(a) + " is out of range for rank " +
                std::to_string(rank()));
            if (seen[(std::size_t)a]) throw std::invalid_argument(
                "permute: axis " + std::to_string(a) + " appears twice");
            seen[(std::size_t)a] = true;
        }
        Tensor out;
        out.buf_  = buf_;
        out.data_ = data_;
        out.size_ = size_;
        out.shape_.resize(order.size());
        out.strides_.resize(order.size());
        for (std::size_t k = 0; k < order.size(); k++) {
            out.shape_[k]   = shape_[(std::size_t)order[k]];
            out.strides_[k] = strides_[(std::size_t)order[k]];
        }
        return out;
    }

    Tensor swapAxes(long a, long b) const {
        checkAxis(a, "swapAxes"); checkAxis(b, "swapAxes");
        std::vector<long> order((std::size_t)rank());
        for (long k = 0; k < rank(); k++) order[(std::size_t)k] = k;
        std::swap(order[(std::size_t)a], order[(std::size_t)b]);
        return permute(order);
    }

    // Reverses every axis — NumPy's .datatype. For rank 2 this is the ordinary
    // transpose, so it agrees with Matrix::T().
    Tensor T() const {
        std::vector<long> order((std::size_t)rank());
        for (long k = 0; k < rank(); k++) order[(std::size_t)k] = rank() - 1 - k;
        return permute(order);
    }

    // Fixes index `i` along `axis`, dropping that axis. A view.
    Tensor slice(long axis, long index) const {
        const long a = checkAxis(axis, "slice");
        if (index < 0 || index >= shape_[(std::size_t)a]) throw std::out_of_range(
            "slice: index " + std::to_string(index) + " is out of range for axis " +
            std::to_string(a) + " of length " + std::to_string(shape_[(std::size_t)a]));
        Tensor out;
        out.buf_  = buf_;
        out.data_ = data_ + index * strides_[(std::size_t)a];
        for (long k = 0; k < rank(); k++) {
            if (k == a) continue;
            out.shape_.push_back(shape_[(std::size_t)k]);
            out.strides_.push_back(strides_[(std::size_t)k]);
        }
        out.size_ = numel(out.shape_);
        return out;
    }

    // ── Materialising ──────────────────────────────────────────────────────

    // Row-major and gap-free. Returns *this (sharing) when already contiguous,
    // so calling it defensively before a kernel costs nothing.
    Tensor contiguous() const {
        // shareView(), NOT `return *this` — returning *this would run the copy
        // constructor, which materialises, so every element-wise operation would
        // deep-copy both operands before touching them. The suite pins this down
        // with "contiguous() is a no-op share when already contiguous".
        if (isContiguous()) return shareView();
        Tensor out(shape_);
        gather(out.data_);
        return out;
    }

    // Unconditional independent copy, contiguous.
    Tensor clone() const {
        Tensor out(shape_);
        gather(out.data_);
        return out;
    }

    // ── Interop with Matrix ────────────────────────────────────────────────

    // rank 1 -> (n x 1) column, rank 2 -> (rows x cols).
    Matrix<datatype> toMatrix() const {
        if (rank() != 1 && rank() != 2) throw std::invalid_argument(
            "toMatrix: only rank 1 or 2 can become a Matrix, this is rank " +
            std::to_string(rank()) + " " + shapeStr(shape_) +
            " — reshape it first");
        const long r = shape_[0];
        const long c = rank() == 2 ? shape_[1] : 1;
        Matrix<datatype> M(r, c);
        if (isContiguous()) {
            std::copy(data_, data_ + size_, &M(0, 0));
        } else {
            std::vector<datatype> tmp((std::size_t)size_);
            gather(tmp.data());
            std::copy(tmp.begin(), tmp.end(), &M(0, 0));
        }
        return M;
    }

    static Tensor fromMatrix(const Matrix<datatype>& M) {
        Tensor out(std::vector<long>{ M.rows(), M.cols() });
        const long n = M.rows() * M.cols();
        const datatype* src = &M(0, 0);
        std::copy(src, src + n, out.data_);
        return out;
    }

    // ── Element-wise arithmetic ────────────────────────────────────────────
    // Same vocabulary and the same naming rule as Matrix: the MEMBER DOT IS THE
    // ELEMENT-WISE MARKER, so these are div and mul rather than ediv and emul —
    // A.div(B) has already said element-wise once by being a member, the way
    // A.sin() does. Ref-qualified so a temporary on
    // the left is reused rather than reallocated — and note the rule the matrix
    // header learned the hard way: once ONE overload of an operator name is
    // ref-qualified, every sibling must be, or overload resolution silently
    // picks the wrong one for rvalue operands.

    Tensor operator+(const Tensor& o) const & { return binary(o, "+", std::plus<datatype>{}); }
    Tensor operator+(const Tensor& o) &&      { return inPlace(o, "+", std::plus<datatype>{}); }
    Tensor operator-(const Tensor& o) const & { return binary(o, "-", std::minus<datatype>{}); }
    Tensor operator-(const Tensor& o) &&      { return inPlace(o, "-", std::minus<datatype>{}); }
    Tensor operator%(const Tensor& o) const & { return binary(o, "%", std::multiplies<datatype>{}); }
    Tensor operator%(const Tensor& o) &&      { return inPlace(o, "%", std::multiplies<datatype>{}); }

    Tensor mul(const Tensor& o) const & { return binary(o, "mul", std::multiplies<datatype>{}); }
    Tensor mul(const Tensor& o) &&      { return inPlace(o, "mul", std::multiplies<datatype>{}); }
    Tensor div(const Tensor& o) const & { return binary(o, "div", std::divides<datatype>{}); }
    Tensor div(const Tensor& o) &&      { return inPlace(o, "div", std::divides<datatype>{}); }

    Tensor& operator+=(const Tensor& o) { *this = std::move(*this) + o; return *this; }
    Tensor& operator-=(const Tensor& o) { *this = std::move(*this) - o; return *this; }
    Tensor& operator%=(const Tensor& o) { *this = std::move(*this) % o; return *this; }

    Tensor operator-() const {
        Tensor a = contiguous(), out(a.shape_);
        const datatype* MATRIXCPP_RESTRICT p = a.data_;
        datatype* MATRIXCPP_RESTRICT r = out.data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] = -p[i]; });
        return out;
    }

    template <typename Scalar,
              typename = std::enable_if_t<!std::is_base_of<Tensor, std::decay_t<Scalar>>::value>>
    Tensor operator*(const Scalar& k) const & { return scalarOp(k, true); }
    template <typename Scalar,
              typename = std::enable_if_t<!std::is_base_of<Tensor, std::decay_t<Scalar>>::value>>
    Tensor operator/(const Scalar& k) const & { return scalarOp(k, false); }

    template <typename Scalar>
    Tensor& operator*=(const Scalar& k) {
        makeContiguousInPlace();
        datatype* MATRIXCPP_RESTRICT r = data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] *= k; });
        return *this;
    }
    template <typename Scalar>
    Tensor& operator/=(const Scalar& k) {
        makeContiguousInPlace();
        datatype* MATRIXCPP_RESTRICT r = data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] /= k; });
        return *this;
    }

    // ── Contraction ────────────────────────────────────────────────────────
    // A * B contracts A's LAST axis with B's FIRST. For two rank-2 tensors that
    // is exactly the matrix product, so it agrees with Matrix::operator*.
    Tensor operator*(const Tensor& B) const { return contractLast(*this, B, 1); }

private:
    // A member TYPE has to be declared before any member declaration names it,
    // and the comparison templates below use mask_scalar_t in their template
    // parameter lists. Member function BODIES are compiled after the class is
    // complete, which is why CmpOp can be used freely in them.
    enum class CmpOp { LT, GT, LE, GE, EQ, NE };
    enum class LogOp { AND, OR, XOR };

    template <typename Scalar>
    using mask_scalar_t = std::enable_if_t<
        !std::is_base_of<Tensor, std::decay_t<Scalar>>::value>;

public:
    // Write-through handle for logical indexing: A(mask) = x.
    //
    // The mask is flattened into a vector<char> and stored BY VALUE. By value
    // because `A(A > 0) = 0.0` builds a temporary mask and a proxy held past the
    // full expression would dangle; flattened because a Tensor<bool> member
    // would make Tensor<bool> contain a MaskProxy containing a Tensor<bool>, an
    // infinitely recursive type. char rather than bool keeps plain-array
    // indexing that vector<bool>'s bit packing would take away.
    class MaskProxy {
        Tensor&           ten;
        std::vector<char> mask;
    public:
        MaskProxy(Tensor& t, const Tensor<bool>& k) : ten(t) {
            t.requireMaskShape(k);
            Tensor<bool> kc = k.contiguous();
            mask.resize((std::size_t)k.size());
            for (long i = 0; i < k.size(); i++) mask[(std::size_t)i] = kc.rawData()[i] ? 1 : 0;
        }
        long selected() const {
            long n = 0;
            for (char c : mask) if (c) n++;
            return n;
        }
        template <typename Scalar,
                  typename = std::enable_if_t<
                      !std::is_base_of<Tensor, std::decay_t<Scalar>>::value>>
        MaskProxy& operator=(const Scalar& v) {
            ten.makeContiguousInPlace();
            for (long i = 0; i < ten.size(); i++)
                if (mask[(std::size_t)i]) ten.rawData()[i] = datatype(v);
            return *this;
        }
        MaskProxy& operator=(const Tensor& src) {
            const long n = selected();
            if (src.size() != n) throw std::invalid_argument(
                "A(mask) = src: the mask selects " + std::to_string(n) +
                " elements but src has " + std::to_string(src.size()));
            ten.makeContiguousInPlace();
            Tensor sc = src.contiguous();
            long k = 0;
            for (long i = 0; i < ten.size(); i++)
                if (mask[(std::size_t)i]) ten.rawData()[i] = sc.rawData()[k++];
            return *this;
        }
        operator Tensor() const {
            Tensor a = ten.contiguous();
            Tensor out(std::vector<long>{ selected() });
            long k = 0;
            for (long i = 0; i < ten.size(); i++)
                if (mask[(std::size_t)i]) out.rawData()[k++] = a.rawData()[i];
            return out;
        }
        friend std::ostream& operator<<(std::ostream& os, const MaskProxy& p) {
            return os << static_cast<Tensor>(p).toString();
        }
    };

    // ══════════════════════════════════════════════════════════════════════
    //  Logical masks — the same layer Matrix has, same spellings
    // ══════════════════════════════════════════════════════════════════════
    //
    //     A > 0        A <= B       A.eq(B)      A.ne(0)
    //     m1 && m2     m1 || m2     m1 ^ m2      !m1
    //     A.any()      A.all()      A.nnz()      A.find()
    //     A(A > 0)                  read the selected elements (rank 1)
    //     A(A < 0) = 0.0            write through the mask
    //
    // A mask is a Tensor<bool>, so it reshapes, permutes and slices like any
    // other tensor. The rule is the one Matrix follows: < > <= >= are
    // element-wise operators because no tensor-level ordering exists to confuse
    // them with, while == and != stay whole-tensor and return bool, with .eq()
    // and .ne() as their element-wise forms.
    //
    // Note that || is used here even though Tensor has no augmented-tensor
    // operator to collide with. Matching Matrix matters more than claiming the
    // free slot: | meaning `or` on a Tensor and `concatenate` on a Matrix would
    // be a far worse trap than not having | at all.

    Tensor<bool> lt(const Tensor& o) const { return compare(o, "lt", CmpOp::LT); }
    Tensor<bool> gt(const Tensor& o) const { return compare(o, "gt", CmpOp::GT); }
    Tensor<bool> le(const Tensor& o) const { return compare(o, "le", CmpOp::LE); }
    Tensor<bool> ge(const Tensor& o) const { return compare(o, "ge", CmpOp::GE); }
    Tensor<bool> eq(const Tensor& o) const { return compare(o, "eq", CmpOp::EQ); }
    Tensor<bool> ne(const Tensor& o) const { return compare(o, "ne", CmpOp::NE); }

    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> lt(const Scalar& v) const { return compareScalar(v, CmpOp::LT); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> gt(const Scalar& v) const { return compareScalar(v, CmpOp::GT); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> le(const Scalar& v) const { return compareScalar(v, CmpOp::LE); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> ge(const Scalar& v) const { return compareScalar(v, CmpOp::GE); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> eq(const Scalar& v) const { return compareScalar(v, CmpOp::EQ); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> ne(const Scalar& v) const { return compareScalar(v, CmpOp::NE); }

    Tensor<bool> operator<(const Tensor& o) const  { return lt(o); }
    Tensor<bool> operator>(const Tensor& o) const  { return gt(o); }
    Tensor<bool> operator<=(const Tensor& o) const { return le(o); }
    Tensor<bool> operator>=(const Tensor& o) const { return ge(o); }

    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> operator<(const Scalar& v) const  { return lt(v); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> operator>(const Scalar& v) const  { return gt(v); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> operator<=(const Scalar& v) const { return le(v); }
    template <typename Scalar, typename = mask_scalar_t<Scalar>>
    Tensor<bool> operator>=(const Scalar& v) const { return ge(v); }

    Tensor<bool> land(const Tensor& o) const { return logical(o, "land", LogOp::AND); }
    Tensor<bool> lor (const Tensor& o) const { return logical(o, "lor",  LogOp::OR);  }
    Tensor<bool> lxor(const Tensor& o) const { return logical(o, "lxor", LogOp::XOR); }
    Tensor<bool> lnot() const { return !(*this); }

    Tensor<bool> operator&&(const Tensor& o) const { return land(o); }
    Tensor<bool> operator||(const Tensor& o) const { return lor(o);  }
    Tensor<bool> operator^(const Tensor& o) const  { return lxor(o); }

    Tensor<bool> operator!() const {
        Tensor a = contiguous();
        Tensor<bool> out(shape_);
        const datatype* MATRIXCPP_RESTRICT p = a.data_;
        bool* MATRIXCPP_RESTRICT r = out.rawData();
        const datatype zero = datatype(0);
        mstore::forEachIndex(size_, [=](long i) { r[i] = (p[i] == zero); });
        return out;
    }

    // Non-zero counts as true throughout, so these read the same on a mask and
    // on the numeric tensor it came from.
    bool any() const {
        Tensor a = contiguous();
        const datatype zero = datatype(0);
        for (long i = 0; i < size_; i++) if (!(a.data_[i] == zero)) return true;
        return false;
    }
    bool all() const {
        Tensor a = contiguous();
        const datatype zero = datatype(0);
        for (long i = 0; i < size_; i++) if (a.data_[i] == zero) return false;
        return true;
    }
    // Count the true entries with this, not sum(): sum() returns datatype, and
    // for Tensor<bool> that saturates at true rather than counting.
    long nnz() const {
        Tensor a = contiguous();
        long n = 0;
        const datatype zero = datatype(0);
        for (long i = 0; i < size_; i++) if (!(a.data_[i] == zero)) n++;
        return n;
    }

    // Flat row-major positions of the non-zero elements.
    //
    // Matrix::find() returns (row, column) pairs, which it can because its rank
    // is fixed at two. A tensor's positions are rank-long, so a vector of
    // vectors would be the honest analogue and is clumsy to use; flat indices
    // compose directly with reshape(-1) instead, and unravel() turns one back
    // into per-axis indices when that is what is wanted.
    std::vector<long> find() const {
        Tensor a = contiguous();
        std::vector<long> out;
        const datatype zero = datatype(0);
        for (long i = 0; i < size_; i++) if (!(a.data_[i] == zero)) out.push_back(i);
        return out;
    }

    // Flat row-major index -> one index per axis. NumPy's unravel_index.
    std::vector<long> unravel(long flat) const {
        if (flat < 0 || flat >= size_) throw std::out_of_range(
            "unravel: " + std::to_string(flat) + " is out of range for a tensor of " +
            std::to_string(size_) + " elements");
        std::vector<long> idx((std::size_t)rank());
        for (long k = rank() - 1; k >= 0; k--) {
            idx[(std::size_t)k] = flat % shape_[(std::size_t)k];
            flat /= shape_[(std::size_t)k];
        }
        return idx;
    }

    // Read: A(mask) gives a rank-1 tensor of the selected elements, row-major.
    Tensor operator()(const Tensor<bool>& mask) const {
        requireMaskShape(mask);
        Tensor a = contiguous();
        Tensor<bool> m = mask.contiguous();
        const long n = mask.nnz();
        Tensor out(std::vector<long>{n});
        long k = 0;
        for (long i = 0; i < size_; i++) if (m.rawData()[i]) out.data_[k++] = a.data_[i];
        return out;
    }

    // Write: A(mask) = scalar, or A(mask) = a rank-1 tensor of nnz elements.
    MaskProxy operator()(const Tensor<bool>& mask) { return MaskProxy(*this, mask); }

    // ── Comparison ─────────────────────────────────────────────────────────

    bool sameShape(const Tensor& o) const { return shape_ == o.shape_; }

    bool operator==(const Tensor& o) const {
        if (shape_ != o.shape_) return false;
        Tensor a = contiguous(), b = o.contiguous();
        for (long i = 0; i < size_; i++) if (!(a.data_[i] == b.data_[i])) return false;
        return true;
    }
    bool operator!=(const Tensor& o) const { return !(*this == o); }

    bool allclose(const Tensor& o, double rtol = 1e-5, double atol = 1e-8) const {
        if (shape_ != o.shape_) return false;
        Tensor a = contiguous(), b = o.contiguous();
        for (long i = 0; i < size_; i++) {
            const double diff = magnitude(a.data_[i] - b.data_[i]);
            if (diff > atol + rtol * magnitude(b.data_[i])) return false;
        }
        return true;
    }

    // ── Reductions ─────────────────────────────────────────────────────────

    // pairwiseSum, the same one Matrix uses: eight independent accumulator
    // chains rather than one latency-bound one, and O(log n · eps) error growth
    // instead of O(n · eps).
    datatype sum() const {
        if (size_ == 0) return datatype(0);
        Tensor a = contiguous();
        return pairwiseSum(a.data_, a.size_);
    }

    double mean() const {
        if (size_ == 0) throw std::invalid_argument("mean: tensor is empty");
        return magnitudeSigned(sum()) / double(size_);
    }

    datatype min() const { return extremum(true); }
    datatype max() const { return extremum(false); }

    // Sum along one axis, dropping it. sum(rank-1) needs no permutation and is
    // the fast case; other axes are rotated to the end first, which materialises.
    Tensor sum(long axis) const {
        const long a = checkAxis(axis, "sum");
        if (rank() == 1) {
            Tensor out(std::vector<long>{});
            out = Tensor(std::vector<long>{1});
            out.data_[0] = sum();
            return out;
        }
        // Move `a` last, then every output element is a contiguous run.
        std::vector<long> order;
        for (long k = 0; k < rank(); k++) if (k != a) order.push_back(k);
        order.push_back(a);
        Tensor rolled = permute(order).contiguous();

        std::vector<long> outShape(rolled.shape_.begin(), rolled.shape_.end() - 1);
        Tensor out(outShape);
        const long run = rolled.shape_.back();
        const long rows = out.size_;
        const datatype* MATRIXCPP_RESTRICT src = rolled.data_;
        datatype* MATRIXCPP_RESTRICT dst = out.data_;
        for (long i = 0; i < rows; i++) dst[i] = pairwiseSum(src + i * run, run);
        return out;
    }

    // ── Filling ────────────────────────────────────────────────────────────

    Tensor& fill(const datatype& v) {
        makeContiguousInPlace();
        datatype* MATRIXCPP_RESTRICT r = data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] = v; });
        return *this;
    }

    // Uniform random values, reusing the same generator Matrix uses so a seed
    // means the same thing in both.
    Tensor& set_Ran_values(double lo, double hi, long seed) {
        makeContiguousInPlace();
        Matrix<double> tmp(size_, 1);
        tmp.set_Ran_values(lo, hi, seed);
        for (long i = 0; i < size_; i++) data_[i] = datatype(tmp(int(i), 0));
        return *this;
    }

    // ── Printing ───────────────────────────────────────────────────────────

    std::string toString(int precision = 6) const {
        std::ostringstream os;
        os << "Tensor" << shapeStr(shape_) << '\n';
        if (size_ == 0) return os.str();
        if (rank() <= 2) { os << sliceToString(*this, precision); return os.str(); }
        // Higher rank: print each trailing 2-D slice under its index prefix.
        const long lead = size_ / (shape_[(std::size_t)rank() - 2] *
                                   shape_[(std::size_t)rank() - 1]);
        std::vector<long> idx((std::size_t)rank() - 2, 0);
        for (long s = 0; s < lead; s++) {
            Tensor sub = *this;
            os << '[';
            for (std::size_t k = 0; k < idx.size(); k++) {
                os << idx[k] << (k + 1 < idx.size() ? "," : "");
                sub = sub.slice(0, idx[k]);
            }
            os << ",:,:]\n" << sliceToString(sub, precision);
            for (long k = (long)idx.size() - 1; k >= 0; k--) {
                if (++idx[(std::size_t)k] < shape_[(std::size_t)k]) break;
                idx[(std::size_t)k] = 0;
            }
        }
        return os.str();
    }

    void print(int precision = 6) const { std::cout << toString(precision) << '\n'; }

    // ═══════════════════════════════════════════════════════════════════════
    //  Contraction — the operation the whole layout exists for
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Contracts the last `k` axes of A with the first `k` axes of B, exactly
    // like np.tensordot(A, B, k). Those axes are flattened into one matrix
    // dimension and the work is handed to mstore::gemm — the same blocked,
    // restrict-qualified, OpenMP kernel Matrix::operator* uses. No new kernel,
    // and no copy at all when both operands are already contiguous.
    static Tensor contractLast(const Tensor& A, const Tensor& B, long k) {
        long M, N, K;
        std::vector<long> outShape;
        contractDims(A, B, k, M, N, K, outShape);
        Tensor out(outShape);
        Tensor Ac = A.contiguous(), Bc = B.contiguous();
        mstore::gemm(Ac.data_, Bc.data_, out.data_, M, N, K);
        return out;
    }

    // Out-parameter form. This is the std::linalg (P1673) lesson made concrete:
    // an 18-term Taylor-shaped loop that allocates a temporary per step costs
    // 3.2x its own arithmetic at n=256 (68% overhead; 37% at n=128). Writing
    // into storage the caller already owns removes all of it.
    static void contractInto(const Tensor& A, const Tensor& B, long k, Tensor& out) {
        long M, N, K;
        std::vector<long> outShape;
        contractDims(A, B, k, M, N, K, outShape);
        if (out.shape_ != outShape) throw std::invalid_argument(
            "contractInto: output has shape " + shapeStr(out.shape_) + ", needs " +
            shapeStr(outShape));
        out.requireContiguous("contractInto (output)");
        Tensor Ac = A.contiguous(), Bc = B.contiguous();
        mstore::gemm(Ac.data_, Bc.data_, out.data_, M, N, K);
    }

    // Accumulating form: out += A·B, no allocation and no separate add pass.
    static void contractAccInto(const Tensor& A, const Tensor& B, long k, Tensor& out) {
        long M, N, K;
        std::vector<long> outShape;
        contractDims(A, B, k, M, N, K, outShape);
        if (out.shape_ != outShape) throw std::invalid_argument(
            "contractAccInto: output has shape " + shapeStr(out.shape_) + ", needs " +
            shapeStr(outShape));
        out.requireContiguous("contractAccInto (output)");
        Tensor Ac = A.contiguous(), Bc = B.contiguous();
        mstore::gemmAcc(Ac.data_, Bc.data_, out.data_, M, N, K);
    }

    // General contraction over named axis pairs. Permutes the contracted axes
    // to the end of A and the front of B, then defers to contractLast — which
    // is exactly the reshape -> permute -> GEMM pipeline described at the top.
    static Tensor contractAxes(const Tensor& A, const Tensor& B,
                               const std::vector<long>& axesA,
                               const std::vector<long>& axesB) {
        if (axesA.size() != axesB.size()) throw std::invalid_argument(
            "contract: axesA and axesB must name the same number of axes");
        const long k = (long)axesA.size();
        std::vector<bool> inA((std::size_t)A.rank(), false), inB((std::size_t)B.rank(), false);
        for (long i = 0; i < k; i++) {
            const long a = A.checkAxis(axesA[(std::size_t)i], "contract");
            const long b = B.checkAxis(axesB[(std::size_t)i], "contract");
            if (inA[(std::size_t)a] || inB[(std::size_t)b]) throw std::invalid_argument(
                "contract: an axis is named twice");
            inA[(std::size_t)a] = inB[(std::size_t)b] = true;
            if (A.shape_[(std::size_t)a] != B.shape_[(std::size_t)b])
                throw std::invalid_argument(
                    "contract: axis " + std::to_string(a) + " of A has length " +
                    std::to_string(A.shape_[(std::size_t)a]) + " but axis " +
                    std::to_string(b) + " of B has length " +
                    std::to_string(B.shape_[(std::size_t)b]));
        }
        std::vector<long> orderA, orderB;
        for (long i = 0; i < A.rank(); i++) if (!inA[(std::size_t)i]) orderA.push_back(i);
        for (long i = 0; i < k; i++) orderA.push_back(axesA[(std::size_t)i]);
        for (long i = 0; i < k; i++) orderB.push_back(axesB[(std::size_t)i]);
        for (long i = 0; i < B.rank(); i++) if (!inB[(std::size_t)i]) orderB.push_back(i);
        return contractLast(A.permute(orderA), B.permute(orderB), k);
    }

private:
    // ── Storage ────────────────────────────────────────────────────────────
    // The buffer is reference counted so that a view keeps it alive after the
    // tensor it came from is gone. data_ is the cached start pointer (buf_.get()
    // plus any slice offset) so that indexing never pays for the shared_ptr.
    std::shared_ptr<datatype>  buf_;
    datatype*                  data_ = nullptr;
    std::vector<long>   shape_;
    std::vector<long>   strides_;
    long                size_ = 0;

    template <typename> friend class Tensor;

    static long numel(const std::vector<long>& s) {
        long n = 1;
        for (long d : s) n *= d;
        return s.empty() ? 0 : n;
    }

    static std::vector<long> rowMajorStrides(const std::vector<long>& s) {
        std::vector<long> st(s.size());
        long acc = 1;
        for (long k = (long)s.size() - 1; k >= 0; k--) {
            st[(std::size_t)k] = acc;
            acc *= s[(std::size_t)k];
        }
        return st;
    }

    static std::string shapeStr(const std::vector<long>& s) {
        std::string out = "(";
        for (std::size_t k = 0; k < s.size(); k++) {
            out += std::to_string(s[k]);
            if (k + 1 < s.size()) out += ", ";
        }
        return out + ")";
    }

    void allocZeroed(long n) {
        if (n <= 0) { buf_.reset(); data_ = nullptr; return; }
        // mstore::rawAlloc: no per-element constructor call, and madvise
        // MADV_HUGEPAGE above 4 MB. Zeroing is explicit and threaded.
        datatype* p = mstore::rawAlloc<datatype>(n);
        buf_ = std::shared_ptr<datatype>(p, [](datatype* q) { mstore::rawFree<datatype>(q); });
        data_ = p;
        datatype* MATRIXCPP_RESTRICT r = p;
        mstore::forEachIndex(n, [=](long i) { r[i] = datatype(); });
    }

    void allocRaw(long n) {
        if (n <= 0) { buf_.reset(); data_ = nullptr; return; }
        datatype* p = mstore::rawAlloc<datatype>(n);
        buf_ = std::shared_ptr<datatype>(p, [](datatype* q) { mstore::rawFree<datatype>(q); });
        data_ = p;
    }

    // Deep copy that also flattens a strided view into row-major order.
    void assignFrom(const Tensor& other) {
        shape_   = other.shape_;
        strides_ = rowMajorStrides(shape_);
        size_    = other.size_;
        allocRaw(size_);
        if (size_) other.gather(data_);
    }

    // Walks this tensor in row-major logical order and writes the elements out
    // contiguously. The offset is carried and updated by the odometer rather
    // than recomputed, so it is O(1) per element rather than O(rank).
    void gather(datatype* MATRIXCPP_RESTRICT dst) const {
        if (size_ == 0) return;
        if (isContiguous()) { std::copy(data_, data_ + size_, dst); return; }
        const long r = rank();
        std::vector<long> idx((std::size_t)r, 0);
        long off = 0;
        for (long c = 0; c < size_; c++) {
            dst[c] = data_[off];
            for (long k = r - 1; k >= 0; k--) {
                if (++idx[(std::size_t)k] < shape_[(std::size_t)k]) {
                    off += strides_[(std::size_t)k];
                    break;
                }
                idx[(std::size_t)k] = 0;
                off -= strides_[(std::size_t)k] * (shape_[(std::size_t)k] - 1);
            }
        }
    }

    // A second handle on the same buffer, with the same shape and strides. The
    // copy constructor deliberately materialises instead, so anything that wants
    // to share has to say so explicitly through here.
    Tensor shareView() const {
        Tensor out;
        out.buf_     = buf_;
        out.data_    = data_;
        out.shape_   = shape_;
        out.strides_ = strides_;
        out.size_    = size_;
        return out;
    }

    // Raw storage without the contiguity check that data() applies — used
    // internally where the caller has already made the tensor contiguous.
    datatype*       rawData()       { return data_; }
    const datatype* rawData() const { return data_; }

    void requireMaskShape(const Tensor<bool>& mask) const {
        if (mask.shape() != shape_) throw std::invalid_argument(
            "logical index: the mask has shape " + shapeStr(mask.shape()) +
            " but the tensor has " + shapeStr(shape_));
    }

    // Ordering needs <, which std::complex deliberately does not provide.
    // eq() and ne() are fine for complex and do not go through this.
    static void requireOrdered() {
        static_assert(!is_complex<datatype>::value,
            "lt/gt/le/ge (and <, >, <=, >=) need an ordering, which std::complex\n"
            "deliberately does not provide. Compare a component instead.\n"
            "eq() and ne() DO work for complex.");
    }

    template <typename A, typename B>
    static bool applyCmp(CmpOp op, const A& x, const B& y) {
        switch (op) {
            case CmpOp::LT: return x <  y;
            case CmpOp::GT: return x >  y;
            case CmpOp::LE: return x <= y;
            case CmpOp::GE: return x >= y;
            case CmpOp::EQ: return x == y;
            default:        return !(x == y);
        }
    }

    Tensor<bool> compare(const Tensor& o, const char* who, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE) requireOrdered();
        requireSameShape(o, who);
        Tensor a = contiguous(), b = o.contiguous();
        Tensor<bool> out(shape_);
        const datatype* MATRIXCPP_RESTRICT pa = a.data_;
        const datatype* MATRIXCPP_RESTRICT pb = b.data_;
        bool* MATRIXCPP_RESTRICT r = out.rawData();
        mstore::forEachIndex(size_, [=](long i) { r[i] = applyCmp(op, pa[i], pb[i]); });
        return out;
    }

    template <typename Scalar>
    Tensor<bool> compareScalar(const Scalar& v, CmpOp op) const {
        if (op != CmpOp::EQ && op != CmpOp::NE) requireOrdered();
        Tensor a = contiguous();
        Tensor<bool> out(shape_);
        const datatype* MATRIXCPP_RESTRICT p = a.data_;
        bool* MATRIXCPP_RESTRICT r = out.rawData();
        const datatype rhs = datatype(v);
        mstore::forEachIndex(size_, [=](long i) { r[i] = applyCmp(op, p[i], rhs); });
        return out;
    }

    Tensor<bool> logical(const Tensor& o, const char* who, LogOp op) const {
        requireSameShape(o, who);
        Tensor a = contiguous(), b = o.contiguous();
        Tensor<bool> out(shape_);
        const datatype* MATRIXCPP_RESTRICT pa = a.data_;
        const datatype* MATRIXCPP_RESTRICT pb = b.data_;
        bool* MATRIXCPP_RESTRICT r = out.rawData();
        const datatype zero = datatype(0);
        mstore::forEachIndex(size_, [=](long i) {
            const bool x = !(pa[i] == zero), y = !(pb[i] == zero);
            r[i] = (op == LogOp::AND) ? (x && y)
                 : (op == LogOp::OR)  ? (x || y)
                                      : (x != y);
        });
        return out;
    }

    void makeContiguousInPlace() {
        if (isContiguous()) return;
        Tensor c = clone();
        *this = std::move(c);
    }

    long checkAxis(long axis, const char* who) const {
        if (axis < 0 || axis >= rank()) throw std::out_of_range(
            std::string(who) + ": axis " + std::to_string(axis) +
            " is out of range for rank " + std::to_string(rank()));
        return axis;
    }

    void requireContiguous(const char* who) const {
        if (!isContiguous()) throw std::logic_error(
            std::string(who) + ": tensor is a strided view, so its elements are not "
            "laid out in memory order — call .contiguous() first");
    }

    void requireSameShape(const Tensor& o, const char* who) const {
        if (shape_ != o.shape_) throw std::invalid_argument(
            std::string(who) + ": shapes must match, got " + shapeStr(shape_) +
            " and " + shapeStr(o.shape_));
    }

    template <typename... Idx>
    long offsetOf(Idx... idx) const {
        static_assert((std::is_integral<Idx>::value && ...),
                      "Tensor indices must be integers");
        constexpr long n = (long)sizeof...(Idx);
        if (n != rank()) throw std::invalid_argument(
            "Tensor: got " + std::to_string(n) + " indices for a rank-" +
            std::to_string(rank()) + " tensor " + shapeStr(shape_));
        const long ix[] = { long(idx)... };
        long off = 0;
        for (long k = 0; k < n; k++) {
            if (ix[k] < 0 || ix[k] >= shape_[(std::size_t)k]) throw std::out_of_range(
                "Tensor: index " + std::to_string(ix[k]) + " is out of range for axis " +
                std::to_string(k) + " of length " + std::to_string(shape_[(std::size_t)k]));
            off += ix[k] * strides_[(std::size_t)k];
        }
        return off;
    }

    long offsetOfVec(const std::vector<long>& idx) const {
        if ((long)idx.size() != rank()) throw std::invalid_argument(
            "Tensor::at: got " + std::to_string(idx.size()) + " indices for a rank-" +
            std::to_string(rank()) + " tensor");
        long off = 0;
        for (long k = 0; k < rank(); k++) {
            if (idx[(std::size_t)k] < 0 || idx[(std::size_t)k] >= shape_[(std::size_t)k])
                throw std::out_of_range("Tensor::at: index out of range on axis " +
                                        std::to_string(k));
            off += idx[(std::size_t)k] * strides_[(std::size_t)k];
        }
        return off;
    }

    // Element-wise driver. Both operands are made contiguous first (a strided
    // view is materialised once, rather than paying O(rank) index arithmetic on
    // every element), then the loop is the same flat restrict-qualified,
    // threaded one Matrix uses.
    template <typename Op>
    Tensor binary(const Tensor& o, const char* who, Op op) const {
        requireSameShape(o, who);
        Tensor a = contiguous(), b = o.contiguous();
        Tensor out;
        out.shape_   = shape_;
        out.strides_ = rowMajorStrides(shape_);
        out.size_    = size_;
        out.allocRaw(size_);
        const datatype* MATRIXCPP_RESTRICT pa = a.data_;
        const datatype* MATRIXCPP_RESTRICT pb = b.data_;
        datatype* MATRIXCPP_RESTRICT r = out.data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] = op(pa[i], pb[i]); });
        return out;
    }

    // Rvalue path: *this is already a temporary, so write through its own buffer.
    template <typename Op>
    Tensor inPlace(const Tensor& o, const char* who, Op op) {
        requireSameShape(o, who);
        makeContiguousInPlace();
        Tensor b = o.contiguous();
        const datatype* MATRIXCPP_RESTRICT pb = b.data_;
        datatype* MATRIXCPP_RESTRICT r = data_;
        mstore::forEachIndex(size_, [=](long i) { r[i] = op(r[i], pb[i]); });
        return std::move(*this);
    }

    template <typename Scalar>
    Tensor scalarOp(const Scalar& k, bool multiply) const {
        Tensor a = contiguous();
        Tensor out;
        out.shape_   = shape_;
        out.strides_ = rowMajorStrides(shape_);
        out.size_    = size_;
        out.allocRaw(size_);
        const datatype* MATRIXCPP_RESTRICT p = a.data_;
        datatype* MATRIXCPP_RESTRICT r = out.data_;
        if (multiply) mstore::forEachIndex(size_, [=](long i) { r[i] = p[i] * k; });
        else          mstore::forEachIndex(size_, [=](long i) { r[i] = p[i] / k; });
        return out;
    }

    datatype extremum(bool wantMin) const {
        static_assert(!is_complex<datatype>::value,
            "min()/max() need an ordering, which std::complex deliberately does not "
            "provide. Use abs() on the elements, or compare a specific component.");
        if (size_ == 0) throw std::invalid_argument("min/max: tensor is empty");
        Tensor a = contiguous();
        datatype best = a.data_[0];
        for (long i = 1; i < size_; i++) {
            if (wantMin ? (a.data_[i] < best) : (best < a.data_[i])) best = a.data_[i];
        }
        return best;
    }

    // mean() has to divide, which needs a real number even for a complex tensor.
    static double magnitudeSigned(const datatype& v) {
        if constexpr (is_complex<datatype>::value) return double(v.real());
        else                                return double(v);
    }

    // Shape and dimension check shared by every contraction entry point.
    static void contractDims(const Tensor& A, const Tensor& B, long k,
                             long& M, long& N, long& K, std::vector<long>& outShape) {
        if (k < 0) throw std::invalid_argument("contract: k must be >= 0");
        if (k > A.rank() || k > B.rank()) throw std::invalid_argument(
            "contract: cannot contract " + std::to_string(k) + " axes of a rank-" +
            std::to_string(A.rank()) + " and a rank-" + std::to_string(B.rank()) +
            " tensor");
        for (long i = 0; i < k; i++) {
            const long da = A.shape_[(std::size_t)(A.rank() - k + i)];
            const long db = B.shape_[(std::size_t)i];
            if (da != db) throw std::invalid_argument(
                "contract: contracted axes disagree — A" + shapeStr(A.shape_) +
                " axis " + std::to_string(A.rank() - k + i) + " has length " +
                std::to_string(da) + ", B" + shapeStr(B.shape_) + " axis " +
                std::to_string(i) + " has length " + std::to_string(db));
        }
        M = 1; for (long i = 0; i < A.rank() - k; i++) M *= A.shape_[(std::size_t)i];
        K = 1; for (long i = A.rank() - k; i < A.rank(); i++) K *= A.shape_[(std::size_t)i];
        N = 1; for (long i = k; i < B.rank(); i++) N *= B.shape_[(std::size_t)i];
        outShape.clear();
        for (long i = 0; i < A.rank() - k; i++) outShape.push_back(A.shape_[(std::size_t)i]);
        for (long i = k; i < B.rank(); i++)     outShape.push_back(B.shape_[(std::size_t)i]);
        if (outShape.empty()) outShape.push_back(1);   // full contraction -> scalar
    }

    static std::string sliceToString(const Tensor& t, int precision) {
        Tensor c = t.contiguous();
        const long rows = c.rank() == 0 ? 1 : c.shape_[0];
        const long cols = c.rank() == 2 ? c.shape_[1] : (c.rank() == 0 ? 1 : 1);
        if (c.rank() <= 1) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(precision) << "[ ";
            for (long i = 0; i < c.size_; i++) os << c.data_[i] << (i + 1 < c.size_ ? "  " : "");
            os << " ]\n";
            return os.str();
        }
        std::ostringstream os;
        os << std::fixed << std::setprecision(precision);
        for (long i = 0; i < rows; i++) {
            os << "[ ";
            for (long j = 0; j < cols; j++) os << c.data_[i * cols + j] << (j + 1 < cols ? "  " : "");
            os << " ]\n";
        }
        return os.str();
    }
};

// ── Free functions, mirroring the member spellings ─────────────────────────

// tensordot: contract A's last k axes with B's first k.
template <typename datatype>
Tensor<datatype> contract(const Tensor<datatype>& A, const Tensor<datatype>& B, long k = 1) {
    return Tensor<datatype>::contractLast(A, B, k);
}

// Contract over explicitly named axis pairs.
template <typename datatype>
Tensor<datatype> contract(const Tensor<datatype>& A, const Tensor<datatype>& B,
                   const std::vector<long>& axesA, const std::vector<long>& axesB) {
    return Tensor<datatype>::contractAxes(A, B, axesA, axesB);
}

// Out-parameter forms — see contractInto's comment for why these exist.
template <typename datatype>
void contractInto(const Tensor<datatype>& A, const Tensor<datatype>& B, long k, Tensor<datatype>& out) {
    Tensor<datatype>::contractInto(A, B, k, out);
}
template <typename datatype>
void contractAccInto(const Tensor<datatype>& A, const Tensor<datatype>& B, long k, Tensor<datatype>& out) {
    Tensor<datatype>::contractAccInto(A, B, k, out);
}

template <typename datatype, typename Scalar,
          typename = std::enable_if_t<!std::is_base_of<Tensor<datatype>, std::decay_t<Scalar>>::value>>
Tensor<datatype> operator*(const Scalar& k, const Tensor<datatype>& A) { return A * k; }

template <typename datatype>
std::ostream& operator<<(std::ostream& os, const Tensor<datatype>& t) {
    return os << t.toString();
}

// ── Element-wise dot-operator sugar, same spelling as Matrix ───────────────
//     A *dot* B   is  A .* B      A /dot/ B   is  A ./ B
template <typename datatype> struct TensorMulLhs { const Tensor<datatype>* a; };
template <typename datatype> struct TensorDivLhs { const Tensor<datatype>* a; };

template <typename datatype>
TensorMulLhs<datatype> operator*(const Tensor<datatype>& A, dot_t) { return {&A}; }
template <typename datatype>
TensorDivLhs<datatype> operator/(const Tensor<datatype>& A, dot_t) { return {&A}; }
template <typename datatype>
Tensor<datatype> operator*(TensorMulLhs<datatype> l, const Tensor<datatype>& B) { return l.a->mul(B); }
template <typename datatype>
Tensor<datatype> operator/(TensorDivLhs<datatype> l, const Tensor<datatype>& B) { return l.a->div(B); }
