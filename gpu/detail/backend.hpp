#pragma once

// ==========================================================================
//  Backend interface — the only place the two worlds meet
// ==========================================================================
//
// Everything below is declared in PURE C++17. No CUDA headers, no __device__,
// no nvcc-only syntax. That is deliberate and it is the whole reason the
// public headers stay compilable by plain g++:
//
//     backend.hpp      declarations  (g++ reads this)
//     src/backend.cu   definitions   (nvcc reads this)
//
// User code includes <gpu/MatrixGpu.hpp>, compiles with g++, and links
// libmatrixcpp_gpu.a. Only this library needs a CUDA compiler, and only when
// it is built. A CPU-only build of `basic/` never sees any of it.
//
// The device pointers travel as void*/T* through here. They are NOT
// dereferenceable on the host; every function taking one is documented with
// which side of the bus its arguments live on.

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mgpu {

    // Thrown by every entry point below when CUDA, cuBLAS or cuSOLVER reports
    // a failure. Carries the API's own message so the caller sees, e.g.,
    // "cudaMalloc: out of memory" rather than a bare error code.
    class Error : public std::runtime_error {
      public:
        explicit Error(const std::string& what) : std::runtime_error(what) {}
    };

    namespace detail {

        // ── Device management ──────────────────────────────────────────

        struct DeviceInfo {
            char name[256];
            std::size_t totalMem;   // bytes
            std::size_t freeMem;    // bytes, at the moment of the call
            int major, minor;       // compute capability
            int smCount;
            int clockKHz;
            double memBandwidthGBs; // theoretical peak
        };

        // How many CUDA devices the driver can see. Returns 0 rather than
        // throwing when there is no driver at all, so a caller can degrade to
        // the CPU instead of dying.
        int deviceCount() noexcept;

        DeviceInfo deviceInfo(int dev);
        void setDevice(int dev);
        void deviceSync();

        // ── Memory ─────────────────────────────────────────────────────

        // Both go through a caching pool - see the long note in backend.cu.
        // devFree returns a block to the pool rather than to the driver, so a
        // matrix's destructor is cheap and the next same-sized allocation is
        // nearly free.
        void* devAlloc(std::size_t bytes);
        void devFree(void* p) noexcept;

        // Trims the pool back to nothing. Only needed when another library
        // has to be given room on the same card; the pool already trims itself
        // if an allocation would otherwise fail. Synchronises.
        void poolRelease();

        struct PoolStats {
            std::size_t reservedBytes;   // held by the pool, free or not
            std::size_t usedBytes;       // currently handed out
            std::size_t capBytes;        // release threshold
            bool enabled;
        };
        PoolStats poolStats();

        void copyH2D(void* dst, const void* src, std::size_t bytes);
        void copyD2H(void* dst, const void* src, std::size_t bytes);
        void copyD2D(void* dst, const void* src, std::size_t bytes);
        void devZero(void* p, std::size_t bytes);

        // Page-locked host memory. Worth it only for buffers that cross the
        // bus repeatedly: measured 28.7 GB/s pinned against 12.1 GB/s for a
        // plain std::vector, which the driver has to stage through its own
        // pinned bounce buffer anyway.
        void* hostAlloc(std::size_t bytes);
        void hostFree(void* p) noexcept;

        // ── GEMM ───────────────────────────────────────────────────────
        //
        // ROW-MAJOR, matching Matrix<T>. C[M*N] = alpha * A[M*K] * B[K*N]
        //                                       + beta * C.
        // No transposes are performed: cuBLAS is column-major, and a row-major
        // buffer read column-major is already its own transpose, so asking for
        // C^T = B^T * A^T with the operands swapped lands on the right answer.
        // See gemmRowMajor() in backend.cu for the index bookkeeping.
        void gemm(int M, int N, int K, float alpha, const float* A, const float* B,
                  float beta, float* C);
        void gemm(int M, int N, int K, double alpha, const double* A, const double* B,
                  double beta, double* C);

        // The same product with either operand used TRANSPOSED, without
        // materialising the transpose. cuBLAS takes an op flag per operand and
        // honours it inside the kernel for free, so A.T()*B should never cost
        // a separate pass over A -- and for a tall, skinny A that pass can be a
        // large fraction of the total, since the transpose is O(mn) against a
        // product that is only O(mn * cols(B)).
        //
        // lda / ldb are the operands' STORED column counts (their row-major
        // leading dimensions), which is what cuBLAS needs and what the logical
        // M/N/K no longer tell it once a transpose is in play.
        void gemmT(bool transA, bool transB, int M, int N, int K, int lda, int ldb, float alpha,
                   const float* A, const float* B, float beta, float* C);
        void gemmT(bool transA, bool transB, int M, int N, int K, int lda, int ldb, double alpha,
                   const double* A, const double* B, double beta, double* C);

        // Out-of-place transpose of a row-major rows x cols matrix. Uses
        // cublas geam, which is a tuned library kernel, not a hand-rolled one.
        void transpose(int rows, int cols, const float* A, float* out);
        void transpose(int rows, int cols, const double* A, double* out);

        // out = alpha*A + beta*B, both row-major rows x cols. Also geam.
        void geam(int rows, int cols, float alpha, const float* A, float beta,
                  const float* B, float* out);
        void geam(int rows, int cols, double alpha, const double* A, double beta,
                  const double* B, double* out);

        // ── Element-wise ───────────────────────────────────────────────
        //
        // These are the ops no vendor library provides, so they are the only
        // kernels in the package that we write. All are memory-bound and
        // trivially correct: one thread per element, grid-stride loop.

        // Atan2/Hypot/Mod/Rem and the three logical ops are REAL ONLY -- see
        // the complex section for why the ordered and integral operations have
        // no complex counterpart.
        enum class BinOp { Add, Sub, Mul, Div, Pow, Max, Min,
                           Atan2, Hypot, Mod, Rem, And, Or, Xor };
        enum class UnOp {
            Neg, Abs, Sqrt, Exp, Log, Log2, Log10, Exp2,
            Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh,
            Floor, Ceil, Round, Sign, Recip, Square,
            Asinh, Acosh, Atanh, Cbrt, Log1p, Expm1, Trunc, Not
        };
        enum class RedOp { Sum, Max, Min, SumAbs, SumSq, Prod };

        void binary(BinOp op, std::size_t n, const float* a, const float* b, float* out);
        void binary(BinOp op, std::size_t n, const double* a, const double* b, double* out);

        // scalarLeft == true evaluates op(s, a) rather than op(a, s); it only
        // matters for the non-commutative ops (Sub, Div, Pow).
        void binaryScalar(BinOp op, std::size_t n, const float* a, float s, float* out,
                          bool scalarLeft);
        void binaryScalar(BinOp op, std::size_t n, const double* a, double s, double* out,
                          bool scalarLeft);

        void unary(UnOp op, std::size_t n, const float* a, float* out);
        void unary(UnOp op, std::size_t n, const double* a, double* out);

        void fill(std::size_t n, float* a, float v);
        void fill(std::size_t n, double* a, double v);

        // ── Fused element-wise evaluation ──────────────────────────────
        //
        // A whole element-wise expression in ONE kernel, one temporary, one
        // pass over memory. This is the idea MatX is built around and the one
        // real deficit the CuPy comparison turned up:
        //
        //     (A%B).exp().sqrt()  eager   3 kernels, 3 temporaries
        //                                 read A, read B, write T1
        //                                 read T1, write T2
        //                                 read T2, write C     = 7n traffic
        //                         fused   1 kernel, 1 temporary
        //                                 read A, read B, write C = 3n traffic
        //
        // These are bandwidth-bound, so 7n -> 3n is the whole speedup.
        //
        // HOW IT WORKS WITHOUT RUNTIME COMPILATION. MatX and CuPy's fuse both
        // reach for NVRTC: build the source for the exact expression, compile
        // it, cache it. That would break this package's central promise, which
        // is that user code needs no CUDA compiler and no CUDA toolchain at
        // run time. So instead the expression is compiled -- at C++ compile
        // time, by the expression templates in gpu_matrix.hpp -- into a tiny
        // POSTFIX PROGRAM, and one precompiled kernel interprets it with a
        // small per-thread stack.
        //
        // The interpretation is not free, but it is the right trade: these
        // kernels wait on DRAM, and the switch is uniform across the warp
        // (every thread runs the same program in lockstep), so the decode
        // hides under memory latency that we have to pay either way.

        struct FusedProgram {
            static constexpr int MaxOps = 32;
            static constexpr int MaxInputs = 8;
            static constexpr int StackDepth = 8;

            // code[i] selects what step i does; arg and imm carry its operand.
            //   0  push input matrix arg[i]
            //   1  push the constant imm[i]
            //   2  replace the top of the stack with UnOp(arg[i]) of it
            //   3  pop two, push BinOp(arg[i]) of them
            int nOps = 0;
            // Peak stack depth the program reaches. Anything that never needs
            // more than two slots runs on a register-only kernel; see the
            // dispatch in backend.cu for why that is worth distinguishing.
            int maxDepth = 0;
            int code[MaxOps] = {};
            int arg[MaxOps] = {};
            double imm[MaxOps] = {};
        };

        enum class FusedCode { PushInput = 0, PushScalar = 1, ApplyUnary = 2, ApplyBinary = 3 };

        void fusedElementwise(const FusedProgram& prog, std::size_t n, const float* const* inputs,
                              int nInputs, float* out);
        void fusedElementwise(const FusedProgram& prog, std::size_t n, const double* const* inputs,
                              int nInputs, double* out);

        // ── Shape utilities ────────────────────────────────────────────
        //
        // Extracting a triangle, a block, or a diagonal is the other thing no
        // vendor library exposes, and every factorisation below needs at least
        // one of them to turn cuSOLVER's packed output into separate factors.
        // All row-major, all element-wise, all trivial.

        // Zeroes the triangle OPPOSITE to `upper`, in place. With unitDiag the
        // diagonal is set to 1 as well, which is what unpacking L from getrf
        // wants (LAPACK stores L's unit diagonal implicitly).
        void triangle(int rows, int cols, float* A, bool upper, bool unitDiag);
        void triangle(int rows, int cols, double* A, bool upper, bool unitDiag);

        // dst[nr x nc] = src[r0 .. r0+nr, c0 .. c0+nc]. Both row-major.
        void copyBlock(int srcRows, int srcCols, const float* src, int r0, int c0, int nr, int nc,
                       float* dst);
        void copyBlock(int srcRows, int srcCols, const double* src, int r0, int c0, int nr, int nc,
                       double* dst);

        // dst[r0.., c0..] <- src. The inverse of copyBlock, and the primitive
        // behind fftshift, hstack/vstack and any in-place assembly.
        void setBlock(int dstRows, int dstCols, float* dst, int r0, int c0, int srcRows,
                      int srcCols, const float* src);
        void setBlock(int dstRows, int dstCols, double* dst, int r0, int c0, int srcRows,
                      int srcCols, const double* src);

        // A's main diagonal <- d, and the reverse. d holds min(rows, cols).
        void setDiagonal(int rows, int cols, float* A, const float* d);
        void setDiagonal(int rows, int cols, double* A, const double* d);
        void getDiagonal(int rows, int cols, const float* A, float* d);
        void getDiagonal(int rows, int cols, const double* A, double* d);

        // Whole-buffer reduction. Returns a HOST scalar; synchronises.
        float reduce(RedOp op, std::size_t n, const float* a);
        double reduce(RedOp op, std::size_t n, const double* a);

        // Row/column reductions of a row-major rows x cols matrix.
        // byRow == true  -> one result per row,    out has `rows` elements
        // byRow == false -> one result per column, out has `cols` elements
        void reduceAxis(RedOp op, int rows, int cols, const float* a, float* out, bool byRow);
        void reduceAxis(RedOp op, int rows, int cols, const double* a, double* out, bool byRow);

        // ── Comparisons and masks ──────────────────────────────────────
        //
        // Results are 1.0 / 0.0 in the matrix's own type, not a packed bool
        // array. That is deliberate: a numeric mask composes directly with
        // everything else here -- A % mask zeroes what the mask rejects, and
        // sum(mask) counts it -- where a bool type would need its own parallel
        // set of operations before it was useful for anything.
        //
        // Real only: the complex numbers are not ordered, and offering equality
        // by itself would be a strange half-measure.

        enum class CmpOp { LT, LE, GT, GE, EQ, NE };

        void compare(CmpOp op, std::size_t n, const float* a, const float* b, float* out);
        void compare(CmpOp op, std::size_t n, const double* a, const double* b, double* out);
        void compareScalar(CmpOp op, std::size_t n, const float* a, float s, float* out);
        void compareScalar(CmpOp op, std::size_t n, const double* a, double s, double* out);

        // Index of the largest / smallest element in row-major order. Ties go
        // to the LOWEST index, matching basic/ and NumPy.
        long argExtreme(bool maximum, std::size_t n, const float* a);
        long argExtreme(bool maximum, std::size_t n, const double* a);

        // ── Rearrangement ──────────────────────────────────────────────
        //
        // All gathers over the OUTPUT: one thread per destination element,
        // reading wherever that element comes from. Written that way because it
        // needs no atomics and no bounds reasoning -- every output is written
        // exactly once, by construction.

        enum class Rearrange { Repmat, FlipLR, FlipUD, Rot90, CircShift };

        // For Repmat p/q are the tile counts; for CircShift they are the row and
        // column shifts; otherwise they are unused.
        void rearrange(Rearrange how, int srcRows, int srcCols, const float* src, int p, int q,
                       float* dst);
        void rearrange(Rearrange how, int srcRows, int srcCols, const double* src, int p, int q,
                       double* dst);
        void rearrange(Rearrange how, int srcRows, int srcCols, const std::complex<float>* src,
                       int p, int q, std::complex<float>* dst);
        void rearrange(Rearrange how, int srcRows, int srcCols, const std::complex<double>* src,
                       int p, int q, std::complex<double>* dst);

        // Kronecker product: dst[(i*br + k), (j*bc + l)] = A(i,j) * B(k,l).
        void kron(int ar, int ac, const float* A, int br, int bc, const float* B, float* dst);
        void kron(int ar, int ac, const double* A, int br, int bc, const double* B, double* dst);
        void kron(int ar, int ac, const std::complex<float>* A, int br, int bc,
                  const std::complex<float>* B, std::complex<float>* dst);
        void kron(int ar, int ac, const std::complex<double>* A, int br, int bc,
                  const std::complex<double>* B, std::complex<double>* dst);

        // ── Sorting ────────────────────────────────────────────────────
        //
        // Thrust and CUB do the work. CUDA 13 relocated them to
        // include/cccl/, which nvcc adds to the include path on its own, so
        // <thrust/sort.h> and <cub/cub.cuh> just work -- no extra flags and no
        // hand-written sort network.
        //
        // Real only: sorting needs an order and the complex numbers have none.

        // Sorts the whole buffer in place.
        void sortFlat(std::size_t n, float* data, bool descending);
        void sortFlat(std::size_t n, double* data, bool descending);

        // Sorts each row (byRow) or each column independently, in place, via
        // CUB's segmented sort -- one launch for the whole matrix rather than
        // one per row. Columns are handled by transposing in and back out,
        // since a segment has to be contiguous.
        void sortAxis(int rows, int cols, float* data, bool byRow, bool descending);
        void sortAxis(int rows, int cols, double* data, bool byRow, bool descending);

        // Reorders whole ROWS by the values in column `key`, carrying every
        // other column along.
        void sortRowsBy(int rows, int cols, float* data, int key, bool descending);
        void sortRowsBy(int rows, int cols, double* data, int key, bool descending);

        // Sorts, removes adjacent duplicates, and returns how many distinct
        // values are left at the front of the buffer.
        long uniqueInPlace(std::size_t n, float* data);
        long uniqueInPlace(std::size_t n, double* data);

        // Most frequent value; ties go to the smallest, as MATLAB does.
        float modeOf(std::size_t n, const float* data);
        double modeOf(std::size_t n, const double* data);

        // ── Builders ───────────────────────────────────────────────────
        //
        // Both endpoints included, so the step is (hi - lo) / (n - 1) rather
        // than / n. For n == 1 the single value is the UPPER bound, which is
        // MATLAB's convention and basic/'s -- NumPy returns the lower one, so
        // this is a place the two references genuinely disagree.
        void linspace(std::size_t n, float* out, float lo, float hi, bool logarithmic);
        void linspace(std::size_t n, double* out, double lo, double hi, bool logarithmic);

        // ── Random ─────────────────────────────────────────────────────
        //
        // cuRAND. Uniform is [lo, hi); normal is mean/stddev.
        void randUniform(std::size_t n, float* a, float lo, float hi, unsigned long long seed);
        void randUniform(std::size_t n, double* a, double lo, double hi, unsigned long long seed);
        void randNormal(std::size_t n, float* a, float mean, float sd, unsigned long long seed);
        void randNormal(std::size_t n, double* a, double mean, double sd, unsigned long long seed);

        // ── Factorisations (cuSOLVER) ──────────────────────────────────
        //
        // IMPORTANT: unlike everything above, these take COLUMN-MAJOR data,
        // because that is what cuSOLVER speaks and transposing inside each
        // call would hide a cost the caller may be able to avoid. Matrix
        // does the conversion at its boundary, once.
        //
        // Each returns cuSOLVER's `info`: 0 on success, >0 for the
        // decomposition-specific failure (a zero pivot, a non-positive-definite
        // leading minor), <0 for a bad argument.

        int getrf(int m, int n, float* A, int* ipiv);    // LU, in place
        int getrf(int m, int n, double* A, int* ipiv);
        int getrs(int n, int nrhs, const float* A, const int* ipiv, float* B);
        int getrs(int n, int nrhs, const double* A, const int* ipiv, double* B);

        // ── Mixed-precision solve ──────────────────────────────────────
        //
        // Factor in a LOW precision, then refine the solution back to the
        // input's precision with a few fp64 residual corrections. This is
        // MAGMA's signature technique and the direct answer to a consumer card
        // throttling fp64 to 1/64 of fp32: the O(n^3) factorisation runs where
        // the hardware is fast, and only the O(n^2) refinement runs where it is
        // slow. Measured here, against 24.8 ms for the plain fp64 solve:
        //
        //     n=2048   5.63 ms   2 refinement steps   rel. residual 2.2e-14
        //     n=4096  16.38 ms   2 refinement steps   rel. residual 3.7e-14
        //
        // Full fp64 accuracy, about 4x the speed. cuSOLVER falls back to a
        // straight fp64 factorisation on its own if refinement fails to
        // converge, and reports that by returning a NEGATIVE iteration count,
        // so the low-precision path can never silently return a bad answer.
        enum class Factor { Double, Single, Half, BFloat16 };

        // A is overwritten. `iters` receives the refinement count, or a
        // negative value when cuSOLVER fell back to full precision.
        int gesvMixed(int n, int nrhs, double* A, double* B, double* X, Factor f, int* iters);
        int gesvMixed(int n, int nrhs, float* A, float* B, float* X, Factor f, int* iters);

        int potrf(int n, float* A, bool upper);          // Cholesky, in place
        int potrf(int n, double* A, bool upper);
        int potrs(int n, int nrhs, const float* A, float* B, bool upper);
        int potrs(int n, int nrhs, const double* A, double* B, bool upper);

        int geqrf(int m, int n, float* A, float* tau);   // QR, in place
        int geqrf(int m, int n, double* A, double* tau);
        // Forms the explicit m x k orthogonal factor from geqrf's output.
        int orgqr(int m, int n, int k, float* A, const float* tau);
        int orgqr(int m, int n, int k, double* A, const double* tau);

        // SVD. S has min(m,n) entries; U is m x m and VT is n x n when
        // `full` is set, otherwise the economy m x min and min x n.
        int gesvd(int m, int n, float* A, float* S, float* U, float* VT, bool full);
        int gesvd(int m, int n, double* A, double* S, double* U, double* VT, bool full);

        // Symmetric eigenproblem. w gets the n eigenvalues ascending; A is
        // overwritten with the eigenvectors when `vectors` is set.
        int syevd(int n, float* A, float* w, bool vectors);
        int syevd(int n, double* A, double* w, bool vectors);

        // ══════════════════════════════════════════════════════════════
        //  COMPLEX
        // ══════════════════════════════════════════════════════════════
        //
        // std::complex<T> has the same object representation as T[2] and
        // cuDoubleComplex is a double2, so the host matrix, the device buffer
        // and what cuBLAS/cuSOLVER expect are all the same bytes -- uploads
        // stay a single memcpy and nothing is repacked.
        //
        // Ordering-based operations (Max, Min, floor, ceil, round, sign) have
        // no complex counterpart and are absent rather than faked.
        //
        // project(): complex -> real. part 0 real, 1 imaginary, 2 magnitude,
        // 3 argument. compose(): real -> complex, im may be null.
        // transposeCx(): conjugate = true gives A^H rather than A^T.
        // normSq(): sum |z|^2, real-valued -- the Frobenius norm's inside.
        // heevd(): the Hermitian eigenproblem, with real eigenvalues.

        void binary(BinOp op, std::size_t n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* o);
        void binary(BinOp op, std::size_t n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* o);
        void binaryScalar(BinOp op, std::size_t n, const std::complex<float>* a, std::complex<float> s, std::complex<float>* o, bool l);
        void binaryScalar(BinOp op, std::size_t n, const std::complex<double>* a, std::complex<double> s, std::complex<double>* o, bool l);
        void unary(UnOp op, std::size_t n, const std::complex<float>* a, std::complex<float>* o);
        void unary(UnOp op, std::size_t n, const std::complex<double>* a, std::complex<double>* o);
        void conj(std::size_t n, const std::complex<float>* a, std::complex<float>* o);
        void conj(std::size_t n, const std::complex<double>* a, std::complex<double>* o);
        void fill(std::size_t n, std::complex<float>* a, std::complex<float> v);
        void fill(std::size_t n, std::complex<double>* a, std::complex<double> v);
        void project(int part, std::size_t n, const std::complex<float>* a, float* o);
        void project(int part, std::size_t n, const std::complex<double>* a, double* o);
        void compose(std::size_t n, const float* re, const float* im, std::complex<float>* o);
        void compose(std::size_t n, const double* re, const double* im, std::complex<double>* o);
        std::complex<float> reduce(RedOp op, std::size_t n, const std::complex<float>* a);
        std::complex<double> reduce(RedOp op, std::size_t n, const std::complex<double>* a);
        float normSq(std::size_t n, const std::complex<float>* a);
        double normSq(std::size_t n, const std::complex<double>* a);
        void triangle(int r, int c, std::complex<float>* A, bool u, bool ud);
        void triangle(int r, int c, std::complex<double>* A, bool u, bool ud);
        void copyBlock(int sr, int sc, const std::complex<float>* s, int r0, int c0, int nr, int nc, std::complex<float>* d);
        void copyBlock(int sr, int sc, const std::complex<double>* s, int r0, int c0, int nr, int nc, std::complex<double>* d);
        void setBlock(int dr, int dc, std::complex<float>* d, int r0, int c0, int nr, int nc, const std::complex<float>* s);
        void setBlock(int dr, int dc, std::complex<double>* d, int r0, int c0, int nr, int nc, const std::complex<double>* s);
        void setDiagonal(int r, int c, std::complex<float>* A, const std::complex<float>* d);
        void setDiagonal(int r, int c, std::complex<double>* A, const std::complex<double>* d);
        void getDiagonal(int r, int c, const std::complex<float>* A, std::complex<float>* d);
        void getDiagonal(int r, int c, const std::complex<double>* A, std::complex<double>* d);
        void gemm(int M, int N, int K, std::complex<float> alpha, const std::complex<float>* A, const std::complex<float>* B, std::complex<float> beta, std::complex<float>* C);
        void gemm(int M, int N, int K, std::complex<double> alpha, const std::complex<double>* A, const std::complex<double>* B, std::complex<double> beta, std::complex<double>* C);
        void transposeCx(int rows, int cols, const std::complex<float>* A, std::complex<float>* out, bool conjugate);
        void transposeCx(int rows, int cols, const std::complex<double>* A, std::complex<double>* out, bool conjugate);
        void fusedElementwise(const FusedProgram& prog, std::size_t n, const std::complex<float>* const* inputs, int nInputs, std::complex<float>* out);
        void fusedElementwise(const FusedProgram& prog, std::size_t n, const std::complex<double>* const* inputs, int nInputs, std::complex<double>* out);
        int getrf(int m, int n, std::complex<double>* A, int* ipiv);
        int getrf(int m, int n, std::complex<float>* A, int* ipiv);
        int getrs(int n, int nrhs, const std::complex<double>* A, const int* ipiv, std::complex<double>* B);
        int getrs(int n, int nrhs, const std::complex<float>* A, const int* ipiv, std::complex<float>* B);
        int potrf(int n, std::complex<double>* A, bool upper);
        int potrf(int n, std::complex<float>* A, bool upper);
        int potrs(int n, int nrhs, const std::complex<double>* A, std::complex<double>* B, bool upper);
        int potrs(int n, int nrhs, const std::complex<float>* A, std::complex<float>* B, bool upper);
        int geqrf(int m, int n, std::complex<double>* A, std::complex<double>* tau);
        int geqrf(int m, int n, std::complex<float>* A, std::complex<float>* tau);
        int orgqr(int m, int n, int k, std::complex<double>* A, const std::complex<double>* tau);
        int orgqr(int m, int n, int k, std::complex<float>* A, const std::complex<float>* tau);
        int gesvd(int m, int n, std::complex<double>* A, double* S, std::complex<double>* U, std::complex<double>* VT, bool full);
        int gesvd(int m, int n, std::complex<float>* A, float* S, std::complex<float>* U, std::complex<float>* VT, bool full);
        int heevd(int n, std::complex<double>* A, double* w, bool vectors);
        int heevd(int n, std::complex<float>* A, float* w, bool vectors);
        void fft1dCx(int b, int n, int st, int di, const std::complex<float>* in, std::complex<float>* out, bool inv);
        void fft1dCx(int b, int n, int st, int di, const std::complex<double>* in, std::complex<double>* out, bool inv);
        void fft2dCx(int r, int c, const std::complex<float>* in, std::complex<float>* out, bool inv);
        void fft2dCx(int r, int c, const std::complex<double>* in, std::complex<double>* out, bool inv);
        void compare(CmpOp op, std::size_t n, const float* a, const float* b, float* o);
        void compare(CmpOp op, std::size_t n, const double* a, const double* b, double* o);
        void compareScalar(CmpOp op, std::size_t n, const float* a, float s, float* o);
        void compareScalar(CmpOp op, std::size_t n, const double* a, double s, double* o);
        long argExtreme(bool mx, std::size_t n, const float* a);
        long argExtreme(bool mx, std::size_t n, const double* a);
        void rearrange(Rearrange h, int sr, int sc, const float* s, int p, int q, float* d);
        void rearrange(Rearrange h, int sr, int sc, const double* s, int p, int q, double* d);
        void rearrange(Rearrange h, int sr, int sc, const std::complex<float>* s, int p, int q, std::complex<float>* d);
        void rearrange(Rearrange h, int sr, int sc, const std::complex<double>* s, int p, int q, std::complex<double>* d);
        void kron(int ar, int ac, const float* A, int br, int bc, const float* B, float* d);
        void kron(int ar, int ac, const double* A, int br, int bc, const double* B, double* d);
        void kron(int ar, int ac, const std::complex<float>* A, int br, int bc, const std::complex<float>* B, std::complex<float>* d);
        void kron(int ar, int ac, const std::complex<double>* A, int br, int bc, const std::complex<double>* B, std::complex<double>* d);
        void sortFlat(std::size_t n, float* d, bool r);
        void sortFlat(std::size_t n, double* d, bool r);
        void sortAxis(int r, int c, float* d, bool byRow, bool desc);
        void sortAxis(int r, int c, double* d, bool byRow, bool desc);
        void sortRowsBy(int r, int c, float* d, int k, bool desc);
        void sortRowsBy(int r, int c, double* d, int k, bool desc);
        long uniqueInPlace(std::size_t n, float* d);
        long uniqueInPlace(std::size_t n, double* d);
        float modeOf(std::size_t n, const float* d);
        double modeOf(std::size_t n, const double* d);
        void linspace(std::size_t n, float* o, float lo, float hi, bool lg);
        void linspace(std::size_t n, double* o, double lo, double hi, bool lg);
        int geev(int n, double* A, double* wr, double* wi, double* VRr, double* VRi);
        int geev(int n, float* A, float* wr, float* wi, float* VRr, float* VRi);

        // ── FFT (cuFFT) ────────────────────────────────────────────────
        //
        // Complex data travels as SPLIT real/imaginary arrays for the real
        // entry points, because mgpu::Matrix is real-only there; the Cx forms
        // take interleaved complex straight through, which is what cuFFT wants
        // and what std::complex already is.
        //
        // stride is the gap between elements of one transform, dist the gap
        // between transforms. Row-major m x n: along rows n=cols batch=rows
        // stride=1 dist=cols; down columns n=rows batch=cols stride=cols dist=1.
        // The inverse is scaled by 1/n, matching NumPy, MATLAB and basic/.

        void fft1d(int b, int n, int st, int di, const float* ir, const float* ii, float* orr, float* oi, bool inv);
        void fft1d(int b, int n, int st, int di, const double* ir, const double* ii, double* orr, double* oi, bool inv);
        void fft2d(int r, int c, const float* ir, const float* ii, float* orr, float* oi, bool inv);
        void fft2d(int r, int c, const double* ir, const double* ii, double* orr, double* oi, bool inv);

        // ── General (non-symmetric) eigenproblem ───────────────────────
        //
        // cusolverDnXgeev, and the DATA TYPES ARE THE WHOLE TRICK. For a real
        // matrix:
        //
        //     dataTypeA   real       the matrix
        //     dataTypeW   COMPLEX    eigenvalues, which are complex in general
        //     dataTypeVL  real       <- real, NOT complex
        //     dataTypeVR  real       <- real, NOT complex
        //     computeType real
        //
        // Passing complex for the eigenvector arrays -- which looks right,
        // since the vectors are complex whenever the values are -- makes
        // bufferSize return CUSOLVER_STATUS_INVALID_VALUE, and an earlier
        // version of this file concluded from that the routine was broken. It
        // is not. cuSOLVER returns the vectors in LAPACK's PACKED REAL form:
        // a real eigenvalue gets one column, and a conjugate pair gets two,
        // holding the shared real part and the imaginary part. assembleEvs()
        // below turns that back into complex columns.
        //
        // VL must be a valid pointer even when jobvl is NOVECTOR.

        // wr/wi receive the eigenvalues split into real and imaginary parts,
        // n each. VRr/VRi, when non-null, receive the right eigenvectors
        // already unpacked into complex, n*n each, column-major.
        int geev(int n, double* A, double* wr, double* wi, double* VRr, double* VRi);
        int geev(int n, float* A, float* wr, float* wi, float* VRr, float* VRi);

        // ── Diagnostics ────────────────────────────────────────────────

        const char* backendVersions();  // "CUDA 13.2 / cuBLAS 13.4 / cuSOLVER ..."

    }  // namespace detail
}  // namespace mgpu
