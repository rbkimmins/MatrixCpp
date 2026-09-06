// ==========================================================================
//  Backend implementation — the only file that needs nvcc
// ==========================================================================
//
// Everything CUDA lives behind gpu/detail/backend.hpp. Nothing here is visible
// to user code; the public headers see declarations only.
//
// Division of labour, which is the whole design argument for this package:
//
//   cuBLAS    GEMM, transpose, rank-k updates          NVIDIA's, hand-tuned
//   cuSOLVER  LU, Cholesky, QR, SVD, eigenproblems     NVIDIA's, hand-tuned
//   cuRAND    uniform and normal streams               NVIDIA's
//   us        element-wise ops and reductions          ~200 lines, below
//
// We write the fourth row only because no vendor library covers it. Every one
// of those kernels is memory-bound and structurally trivial: one thread per
// element, grid-stride loop, no tiling, no shared memory except in the
// reductions. There is nothing to tune there and nothing to get subtly wrong,
// which is exactly why it is the only part worth hand-writing.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <cusolverDn.h>

// CUDA 13 ships Thrust and CUB under include/cccl/, which nvcc puts on the
// include path itself -- these need no extra -I.
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../detail/backend.hpp"

namespace mgpu {
    namespace detail {

        // ── Error handling ─────────────────────────────────────────────
        //
        // Every CUDA call goes through one of these. They throw Error with
        // the API's own message plus the call site, because a bare
        // "CUBLAS_STATUS_EXECUTION_FAILED" three frames down is unactionable.

        static void checkCuda(cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw Error(std::string(what) + ": " + cudaGetErrorString(e));
        }

        static const char* cublasMsg(cublasStatus_t s) {
            switch (s) {
                case CUBLAS_STATUS_SUCCESS: return "success";
                case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
                case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
                case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
                case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
                case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
                case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
                case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
                case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
                default: return "unknown cuBLAS error";
            }
        }
        static void checkBlas(cublasStatus_t s, const char* what) {
            if (s != CUBLAS_STATUS_SUCCESS)
                throw Error(std::string(what) + ": " + cublasMsg(s));
        }

        static const char* solverMsg(cusolverStatus_t s) {
            switch (s) {
                case CUSOLVER_STATUS_SUCCESS: return "success";
                case CUSOLVER_STATUS_NOT_INITIALIZED: return "not initialized";
                case CUSOLVER_STATUS_ALLOC_FAILED: return "allocation failed";
                case CUSOLVER_STATUS_INVALID_VALUE: return "invalid value";
                case CUSOLVER_STATUS_ARCH_MISMATCH: return "architecture mismatch";
                case CUSOLVER_STATUS_EXECUTION_FAILED: return "execution failed";
                case CUSOLVER_STATUS_INTERNAL_ERROR: return "internal error";
                case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "matrix type not supported";
                default: return "unknown cuSOLVER error";
            }
        }
        static void checkSolver(cusolverStatus_t s, const char* what) {
            if (s != CUSOLVER_STATUS_SUCCESS)
                throw Error(std::string(what) + ": " + solverMsg(s));
        }

        static void checkRand(curandStatus_t s, const char* what) {
            if (s != CURAND_STATUS_SUCCESS)
                throw Error(std::string(what) + ": cuRAND status " + std::to_string((int)s));
        }

        // Kernel launches report asynchronously. This catches the launch
        // itself; a fault inside the kernel surfaces at the next sync.
        static void checkLaunch(const char* what) {
            checkCuda(cudaGetLastError(), what);
        }

        // ── Handles ────────────────────────────────────────────────────
        //
        // cuBLAS and cuSOLVER handles cost milliseconds to create, so they are
        // made once and reused. std::call_once rather than a plain static so
        // the creation itself is thread-safe, not merely the assignment.
        // Deliberately never destroyed: a static destructor racing CUDA's own
        // teardown at exit is a well-known source of phantom crashes, and the
        // driver reclaims everything on process exit regardless.

        namespace {
            cublasHandle_t g_blas = nullptr;
            cusolverDnHandle_t g_solver = nullptr;
            std::once_flag g_blasOnce, g_solverOnce;
        }  // namespace

        static cublasHandle_t blas() {
            std::call_once(g_blasOnce, [] {
                checkBlas(cublasCreate(&g_blas), "cublasCreate");
            });
            return g_blas;
        }
        static cusolverDnHandle_t solver() {
            std::call_once(g_solverOnce, [] {
                checkSolver(cusolverDnCreate(&g_solver), "cusolverDnCreate");
            });
            return g_solver;
        }

        // ── Device management ──────────────────────────────────────────

        int deviceCount() noexcept {
            int n = 0;
            // Swallow the error on purpose: "no driver" is a legitimate state
            // a caller may want to fall back from, not an exception.
            if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
            return n;
        }

        DeviceInfo deviceInfo(int dev) {
            cudaDeviceProp p{};
            checkCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
            DeviceInfo d{};
            std::snprintf(d.name, sizeof(d.name), "%s", p.name);
            d.totalMem = p.totalGlobalMem;
            d.major = p.major;
            d.minor = p.minor;
            d.smCount = p.multiProcessorCount;
            // CUDA 13 dropped clockRate and memoryClockRate from
            // cudaDeviceProp; both now come from cudaDeviceGetAttribute.
            int clk = 0, memClk = 0, busBits = 0;
            cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
            cudaDeviceGetAttribute(&memClk, cudaDevAttrMemoryClockRate, dev);
            cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, dev);
            d.clockKHz = clk;
            // Bus width is in bits and the clock counts one transfer per edge.
            d.memBandwidthGBs = 2.0 * memClk * (busBits / 8) / 1.0e6;
            std::size_t freeB = 0, totB = 0;
            if (cudaMemGetInfo(&freeB, &totB) == cudaSuccess) d.freeMem = freeB;
            return d;
        }

        void setDevice(int dev) { checkCuda(cudaSetDevice(dev), "cudaSetDevice"); }
        void deviceSync() { checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

        // ── Memory ─────────────────────────────────────────────────────

        // POOLED DEVICE MEMORY, via the driver's own stream-ordered allocator.
        //
        // Why a pool at all: measured against CuPy, the element-wise chain came
        // out 2.1x slower for kernels doing identical work. The cause was
        // cudaMalloc, which is not a heap bump - it talks to the driver and
        // synchronises, costing tens of microseconds regardless of size. A
        // chain like (A%B).exp().sqrt() allocates one temporary per step, so at
        // n=1024 the three allocations cost more than the three kernels. CuPy
        // has had a memory pool since its first release for exactly this
        // reason, and without one no expression-level API can compete.
        //
        // Why the DRIVER's pool and not one of ours: the first version here was
        // hand-rolled - free lists keyed by exact size, a cap, a flush on OOM.
        // It produced the right speedup and then failed cusolverDnDgesvd with
        // an internal error, because a caching allocator has a hazard that
        // cudaMalloc/cudaFree does not:
        //
        //     cudaFree SYNCHRONISES THE DEVICE. A pool does not.
        //
        // Kernel launches are asynchronous, so when a scratch buffer's
        // destructor runs the kernel using it may still be executing. With
        // cudaFree that is safe by accident - the free waits. With a naive pool
        // the block is handed straight to the next allocation and two live
        // kernels end up writing the same memory. It is a genuinely nasty bug:
        // silent, size-dependent, and it looks like a library fault.
        //
        // cudaMallocAsync/cudaFreeAsync solve it properly. They are
        // STREAM-ORDERED: a freed block is only reissued once the work queued
        // before the free has actually finished, which the driver tracks
        // without a device-wide sync. Same speedup, none of the hazard, and
        // roughly a hundred lines less code than getting it right by hand.

        namespace {
            std::once_flag g_poolOnce;
            bool g_poolUsable = false;

            void initPool() {
                int dev = 0, supported = 0;
                if (cudaGetDevice(&dev) != cudaSuccess) return;
                if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev) !=
                        cudaSuccess ||
                    !supported)
                    return;   // pre-11.2 semantics: fall back to cudaMalloc

                cudaMemPool_t pool;
                if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) return;
                // Without this the pool returns everything to the OS at each
                // sync point, which is most of the cost we are trying to avoid.
                // A quarter of the card is kept: enough for a chain's working
                // set, not so much that other libraries are starved.
                std::size_t freeB = 0, totB = 0;
                std::uint64_t threshold = std::uint64_t(256) << 20;
                if (cudaMemGetInfo(&freeB, &totB) == cudaSuccess) threshold = totB / 4;
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
                g_poolUsable = true;
            }
        }  // namespace

        void* devAlloc(std::size_t bytes) {
            if (bytes == 0) return nullptr;
            std::call_once(g_poolOnce, initPool);
            void* p = nullptr;
            // Stream 0, the LEGACY default stream, deliberately. cuBLAS,
            // cuSOLVER and cudaMemcpy all run there because no stream is ever
            // set on the handles, and stream-ordered allocation only orders
            // against the stream it is given. Passing cudaStreamPerThread here
            // instead - which looks more modern and is what the first version
            // did - orders the allocations against a stream no work runs on,
            // which is the same as no ordering at all: cuSOLVER then reads
            // scratch that a previous call is still writing and reports
            // uninitialised `info` values.
            cudaError_t e = g_poolUsable ? cudaMallocAsync(&p, bytes, 0) : cudaMalloc(&p, bytes);
            if (e != cudaSuccess && g_poolUsable) {
                // Out of memory with blocks cached is recoverable: trim and
                // retry once before giving up.
                poolRelease();
                e = cudaMallocAsync(&p, bytes, 0);
            }
            checkCuda(e, "cudaMalloc");
            return p;
        }

        void devFree(void* p) noexcept {
            if (!p) return;
            if (g_poolUsable) cudaFreeAsync(p, 0);
            else cudaFree(p);
        }

        void poolRelease() {
            if (!g_poolUsable) return;
            int dev = 0;
            if (cudaGetDevice(&dev) != cudaSuccess) return;
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) return;
            cudaDeviceSynchronize();   // nothing may still be using what we trim
            cudaMemPoolTrimTo(pool, 0);
        }

        PoolStats poolStats() {
            PoolStats s{};
            if (!g_poolUsable) return s;
            int dev = 0;
            if (cudaGetDevice(&dev) != cudaSuccess) return s;
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) return s;
            std::uint64_t reserved = 0, used = 0, threshold = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            s.reservedBytes = reserved;
            s.usedBytes = used;
            s.capBytes = threshold;
            s.enabled = true;
            return s;
        }

        void copyH2D(void* d, const void* s, std::size_t b) {
            if (b) checkCuda(cudaMemcpy(d, s, b, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        }
        void copyD2H(void* d, const void* s, std::size_t b) {
            if (b) checkCuda(cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        }
        void copyD2D(void* d, const void* s, std::size_t b) {
            if (b) checkCuda(cudaMemcpy(d, s, b, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
        }
        void devZero(void* p, std::size_t b) {
            if (b) checkCuda(cudaMemset(p, 0, b), "cudaMemset");
        }
        void* hostAlloc(std::size_t bytes) {
            if (bytes == 0) return nullptr;
            void* p = nullptr;
            checkCuda(cudaMallocHost(&p, bytes), "cudaMallocHost");
            return p;
        }
        void hostFree(void* p) noexcept {
            if (p) cudaFreeHost(p);
        }

        // ── GEMM ───────────────────────────────────────────────────────
        //
        // The row-major/column-major reconciliation, once, so no caller has to
        // think about it again:
        //
        //   Our C (M x N, row-major) = A (M x K) * B (K x N).
        //
        //   A row-major buffer reinterpreted as column-major IS its transpose.
        //   So the same bytes cuBLAS sees are A^T (K x M), B^T (N x K),
        //   C^T (N x M). Transposing the product identity,
        //
        //       C^T = B^T * A^T
        //
        //   which is a plain no-transpose column-major GEMM of the buffers we
        //   already have, with the operands swapped and (M, N) exchanged. Zero
        //   data movement — the "conversion" is entirely in the arguments.

        template <class T, class F>
        static void gemmImpl(F f, const char* name, int M, int N, int K, T alpha, const T* A,
                             const T* B, T beta, T* C) {
            if (M <= 0 || N <= 0 || K <= 0) return;
            checkBlas(f(blas(), CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,          // swapped: C^T is N x M
                        &alpha,
                        B, N,             // B^T, leading dim N
                        A, K,             // A^T, leading dim K
                        &beta,
                        C, N),            // C^T, leading dim N
                      name);
        }

        void gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                  float* C) {
            gemmImpl<float>(cublasSgemm, "cublasSgemm", M, N, K, alpha, A, B, beta, C);
        }
        void gemm(int M, int N, int K, double alpha, const double* A, const double* B, double beta,
                  double* C) {
            gemmImpl<double>(cublasDgemm, "cublasDgemm", M, N, K, alpha, A, B, beta, C);
        }

        // Transpose-aware form. The reconciliation, worked through once:
        //
        //   Logical (row-major):  C[M x N] = opA(A)[M x K] * opB(B)[K x N]
        //
        //   A row-major buffer read column-major is its own transpose, so the
        //   buffer cuBLAS sees for A is A_stored^T with leading dimension
        //   lda = (A's stored column count). Transposing the product identity,
        //
        //       C^T = opB(B)^T * opA(A)^T
        //
        //   and opA(A)^T is A_stored when transA, which is (what cuBLAS sees)^T
        //   -> OP_T; and A_stored^T when !transA, which is exactly what cuBLAS
        //   sees -> OP_N. So the flag passes straight through, and the operands
        //   swap as before. Setting transA = transB = false reproduces the
        //   plain gemm above exactly, which is what the test asserts.
        template <class T, class F>
        static void gemmTImpl(F f, const char* name, bool transA, bool transB, int M, int N, int K,
                              int lda, int ldb, T alpha, const T* A, const T* B, T beta, T* C) {
            if (M <= 0 || N <= 0 || K <= 0) return;
            checkBlas(f(blas(), transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                        transA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda, &beta,
                        C, N),
                      name);
        }

        void gemmT(bool tA, bool tB, int M, int N, int K, int lda, int ldb, float alpha,
                   const float* A, const float* B, float beta, float* C) {
            gemmTImpl<float>(cublasSgemm, "cublasSgemm(T)", tA, tB, M, N, K, lda, ldb, alpha, A, B,
                             beta, C);
        }
        void gemmT(bool tA, bool tB, int M, int N, int K, int lda, int ldb, double alpha,
                   const double* A, const double* B, double beta, double* C) {
            gemmTImpl<double>(cublasDgemm, "cublasDgemm(T)", tA, tB, M, N, K, lda, ldb, alpha, A, B,
                              beta, C);
        }

        // Transpose via geam. Same reinterpretation: transposing a row-major
        // rows x cols matrix is asking cuBLAS to transpose the column-major
        // cols x rows one it already sees.
        template <class T, class F>
        static void transposeImpl(F f, const char* name, int rows, int cols, const T* A, T* out) {
            if (rows <= 0 || cols <= 0) return;
            const T one = T(1), zero = T(0);
            checkBlas(f(blas(), CUBLAS_OP_T, CUBLAS_OP_N,
                        rows, cols,       // result is rows x cols column-major
                        &one, A, cols,    // A seen as cols x rows, ld = cols
                        // beta is 0 so B is never read, but cuBLAS still
                        // validates its leading dimension: pass `out`, which
                        // has exactly the right shape, rather than nullptr.
                        &zero, out, rows,
                        out, rows),
                      name);
        }
        void transpose(int rows, int cols, const float* A, float* out) {
            transposeImpl<float>(cublasSgeam, "cublasSgeam(transpose)", rows, cols, A, out);
        }
        void transpose(int rows, int cols, const double* A, double* out) {
            transposeImpl<double>(cublasDgeam, "cublasDgeam(transpose)", rows, cols, A, out);
        }

        template <class T, class F>
        static void geamImpl(F f, const char* name, int rows, int cols, T alpha, const T* A, T beta,
                             const T* B, T* out) {
            if (rows <= 0 || cols <= 0) return;
            // Shape-agnostic: treat both as the column-major cols x rows they
            // physically are, so no transpose is needed for an elementwise sum.
            checkBlas(f(blas(), CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, A, cols, &beta, B,
                        cols, out, cols),
                      name);
        }
        void geam(int rows, int cols, float alpha, const float* A, float beta, const float* B,
                  float* out) {
            geamImpl<float>(cublasSgeam, "cublasSgeam", rows, cols, alpha, A, beta, B, out);
        }
        void geam(int rows, int cols, double alpha, const double* A, double beta, const double* B,
                  double* out) {
            geamImpl<double>(cublasDgeam, "cublasDgeam", rows, cols, alpha, A, beta, B, out);
        }

        // ── Complex on the device ──────────────────────────────────────
        //
        // cuComplex.h supplies add/sub/mul/div/conj and nothing else -- no
        // exp, log, sqrt or trigonometry -- and CUDA 13 does not ship libcu++'s
        // <complex> at an include path we can rely on. So the arithmetic lives
        // here. It is short, it is textbook, and writing it out avoids making
        // the package depend on a header that may or may not be installed.
        //
        // LAYOUT MATTERS AND IT LINES UP. std::complex<T> is required by the
        // standard to have the same object representation as T[2], and
        // cuDoubleComplex is a double2, i.e. {double x, y}. So a host
        // std::complex<double> buffer, this Cx<double>, and what cuBLAS and
        // cuSOLVER expect are all the same bytes -- uploads stay a single
        // memcpy and no repacking happens anywhere.

        template <class R>
        struct Cx {
            R x, y;
            // TRIVIALLY default-constructible on purpose. A user-provided
            // `Cx() : x(0), y(0) {}` would count as dynamic initialisation,
            // which CUDA cannot do for a __shared__ array -- and the block
            // reductions below declare exactly that. Value-initialising with
            // `Cx<R> z{}` still zeroes both parts; only `Cx<R> z;` leaves them
            // unset, and every such declaration here assigns before reading.
            Cx() = default;
            __host__ __device__ Cx(R re) : x(re), y(0) {}
            __host__ __device__ Cx(R re, R im) : x(re), y(im) {}
        };

        template <class R>
        __device__ __forceinline__ Cx<R> operator+(Cx<R> a, Cx<R> b) {
            return Cx<R>(a.x + b.x, a.y + b.y);
        }
        template <class R>
        __device__ __forceinline__ Cx<R> operator-(Cx<R> a, Cx<R> b) {
            return Cx<R>(a.x - b.x, a.y - b.y);
        }
        template <class R>
        __device__ __forceinline__ Cx<R> operator-(Cx<R> a) {
            return Cx<R>(-a.x, -a.y);
        }
        template <class R>
        __device__ __forceinline__ Cx<R> operator*(Cx<R> a, Cx<R> b) {
            return Cx<R>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
        // Smith's formula rather than the naive (ac+bd)/(c^2+d^2): scaling by
        // the larger denominator component first keeps c^2+d^2 from overflowing
        // or flushing to zero when the parts are far apart in magnitude.
        template <class R>
        __device__ __forceinline__ Cx<R> operator/(Cx<R> a, Cx<R> b) {
            if (fabs(b.x) >= fabs(b.y)) {
                const R r = b.y / b.x, d = b.x + b.y * r;
                return Cx<R>((a.x + a.y * r) / d, (a.y - a.x * r) / d);
            }
            const R r = b.x / b.y, d = b.x * r + b.y;
            return Cx<R>((a.x * r + a.y) / d, (a.y * r - a.x) / d);
        }

        template <class R>
        __device__ __forceinline__ R cxAbs(Cx<R> a) { return hypot(a.x, a.y); }
        template <class R>
        __device__ __forceinline__ R cxArg(Cx<R> a) { return atan2(a.y, a.x); }
        template <class R>
        __device__ __forceinline__ Cx<R> cxConj(Cx<R> a) { return Cx<R>(a.x, -a.y); }

        template <class R>
        __device__ __forceinline__ Cx<R> cxExp(Cx<R> a) {
            const R e = exp(a.x);
            return Cx<R>(e * cos(a.y), e * sin(a.y));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxLog(Cx<R> a) {
            return Cx<R>(log(cxAbs(a)), cxArg(a));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxSqrt(Cx<R> a) {
            // Via the half-angle form, which avoids the cancellation that
            // sqrt((|z|+x)/2) suffers when x is large and negative.
            if (a.x == R(0) && a.y == R(0)) return Cx<R>(0, 0);
            const R m = cxAbs(a);
            const R re = sqrt((m + a.x) / R(2));
            R im = sqrt((m - a.x) / R(2));
            if (a.y < R(0)) im = -im;
            return Cx<R>(re, im);
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxPow(Cx<R> a, Cx<R> b) {
            if (a.x == R(0) && a.y == R(0)) return Cx<R>(0, 0);
            return cxExp(b * cxLog(a));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxSin(Cx<R> a) {
            return Cx<R>(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxCos(Cx<R> a) {
            return Cx<R>(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxSinh(Cx<R> a) {
            return Cx<R>(sinh(a.x) * cos(a.y), cosh(a.x) * sin(a.y));
        }
        template <class R>
        __device__ __forceinline__ Cx<R> cxCosh(Cx<R> a) {
            return Cx<R>(cosh(a.x) * cos(a.y), sinh(a.x) * sin(a.y));
        }

        // ── Element-wise kernels ───────────────────────────────────────
        //
        // The only kernels we write. Grid-stride loops so one launch
        // configuration covers every size, and the block/grid choice below is
        // the standard occupancy-driven one rather than anything tuned: these
        // are bandwidth-bound, so the arithmetic layout does not matter and
        // only the memory access pattern does. All accesses here are unit
        // stride and therefore fully coalesced.

        static constexpr int kBlock = 256;

        static int gridFor(std::size_t n) {
            std::size_t g = (n + kBlock - 1) / kBlock;
            if (g > 65535) g = 65535;   // grid-stride loop covers the rest
            if (g == 0) g = 1;
            return (int)g;
        }

        // Device-side op dispatch. The switch is on a compile-time-unknown
        // enum, but it is uniform across the whole grid, so every warp takes
        // one branch and there is no divergence to pay for. Templating a
        // kernel per op would be faster in principle and is not worth the
        // instantiation count for ops that are waiting on DRAM anyway.

        template <class T>
        __device__ __forceinline__ T applyBin(int op, T a, T b) {
            switch ((BinOp)op) {
                case BinOp::Add: return a + b;
                case BinOp::Sub: return a - b;
                case BinOp::Mul: return a * b;
                case BinOp::Div: return a / b;
                case BinOp::Pow: return pow(a, b);
                case BinOp::Max: return a > b ? a : b;
                case BinOp::Min: return a < b ? a : b;
                case BinOp::Atan2: return atan2(a, b);
                case BinOp::Hypot: return hypot(a, b);
                // MATLAB's convention, which is basic/'s:
                //     mod(-1, 3) ==  2   the sign of the DIVISOR
                //     rem(-1, 3) == -1   the sign of the DIVIDEND
                // rem is C's fmod. mod is the one wanted for wrapping an angle
                // or an index into a range. Both leave x alone when b is 0.
                // Not to be confused with C's remainder(), which rounds to the
                // NEAREST multiple and is neither of these.
                case BinOp::Mod: {
                    if (b == T(0)) return a;
                    const T r = fmod(a, b);
                    return (r != T(0) && ((r < T(0)) != (b < T(0)))) ? T(r + b) : r;
                }
                case BinOp::Rem: return b == T(0) ? a : fmod(a, b);
                // Logical on 0/1 masks, returned as 0.0 or 1.0 so the result
                // composes with the arithmetic like any other mask.
                case BinOp::And: return T((a != T(0)) && (b != T(0)));
                case BinOp::Or: return T((a != T(0)) || (b != T(0)));
                case BinOp::Xor: return T((a != T(0)) != (b != T(0)));
            }
            return T(0);
        }

        template <class T>
        __device__ __forceinline__ T applyUn(int op, T x) {
            switch ((UnOp)op) {
                case UnOp::Neg: return -x;
                case UnOp::Abs: return fabs(x);
                case UnOp::Sqrt: return sqrt(x);
                case UnOp::Exp: return exp(x);
                case UnOp::Log: return log(x);
                case UnOp::Log2: return log2(x);
                case UnOp::Log10: return log10(x);
                case UnOp::Exp2: return exp2(x);
                case UnOp::Sin: return sin(x);
                case UnOp::Cos: return cos(x);
                case UnOp::Tan: return tan(x);
                case UnOp::Asin: return asin(x);
                case UnOp::Acos: return acos(x);
                case UnOp::Atan: return atan(x);
                case UnOp::Sinh: return sinh(x);
                case UnOp::Cosh: return cosh(x);
                case UnOp::Tanh: return tanh(x);
                case UnOp::Floor: return floor(x);
                case UnOp::Ceil: return ceil(x);
                case UnOp::Round: return round(x);
                case UnOp::Sign: return T(x > T(0)) - T(x < T(0));
                case UnOp::Recip: return T(1) / x;
                case UnOp::Square: return x * x;
                case UnOp::Asinh: return asinh(x);
                case UnOp::Acosh: return acosh(x);
                case UnOp::Atanh: return atanh(x);
                case UnOp::Cbrt: return cbrt(x);
                // log1p and expm1 exist because log(1+x) and exp(x)-1 lose
                // every significant digit for small x; they are not shorthand.
                case UnOp::Log1p: return log1p(x);
                case UnOp::Expm1: return expm1(x);
                case UnOp::Trunc: return trunc(x);
                case UnOp::Not: return T(x == T(0));
            }
            return T(0);
        }

        template <class T>
        __global__ void kBinary(int op, std::size_t n, const T* a, const T* b, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = applyBin<T>(op, a[i], b[i]);
        }

        template <class T>
        __global__ void kBinaryScalar(int op, std::size_t n, const T* a, T s, T* out,
                                      bool scalarLeft) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = scalarLeft ? applyBin<T>(op, s, a[i]) : applyBin<T>(op, a[i], s);
        }

        template <class T>
        __global__ void kUnary(int op, std::size_t n, const T* a, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = applyUn<T>(op, a[i]);
        }

        template <class T>
        __global__ void kFill(std::size_t n, T* a, T v) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                a[i] = v;
        }

        template <class T>
        static void binaryImpl(BinOp op, std::size_t n, const T* a, const T* b, T* out) {
            if (!n) return;
            kBinary<T><<<gridFor(n), kBlock>>>((int)op, n, a, b, out);
            checkLaunch("element-wise binary kernel");
        }
        void binary(BinOp op, std::size_t n, const float* a, const float* b, float* o) {
            binaryImpl<float>(op, n, a, b, o);
        }
        void binary(BinOp op, std::size_t n, const double* a, const double* b, double* o) {
            binaryImpl<double>(op, n, a, b, o);
        }

        template <class T>
        static void binaryScalarImpl(BinOp op, std::size_t n, const T* a, T s, T* out, bool left) {
            if (!n) return;
            kBinaryScalar<T><<<gridFor(n), kBlock>>>((int)op, n, a, s, out, left);
            checkLaunch("element-wise scalar kernel");
        }
        void binaryScalar(BinOp op, std::size_t n, const float* a, float s, float* o, bool l) {
            binaryScalarImpl<float>(op, n, a, s, o, l);
        }
        void binaryScalar(BinOp op, std::size_t n, const double* a, double s, double* o, bool l) {
            binaryScalarImpl<double>(op, n, a, s, o, l);
        }

        template <class T>
        static void unaryImpl(UnOp op, std::size_t n, const T* a, T* out) {
            if (!n) return;
            kUnary<T><<<gridFor(n), kBlock>>>((int)op, n, a, out);
            checkLaunch("element-wise unary kernel");
        }
        void unary(UnOp op, std::size_t n, const float* a, float* o) { unaryImpl<float>(op, n, a, o); }
        void unary(UnOp op, std::size_t n, const double* a, double* o) {
            unaryImpl<double>(op, n, a, o);
        }

        template <class T>
        static void fillImpl(std::size_t n, T* a, T v) {
            if (!n) return;
            kFill<T><<<gridFor(n), kBlock>>>(n, a, v);
            checkLaunch("fill kernel");
        }
        void fill(std::size_t n, float* a, float v) { fillImpl<float>(n, a, v); }
        void fill(std::size_t n, double* a, double v) { fillImpl<double>(n, a, v); }

        // ── Fused element-wise evaluation ──────────────────────────────
        //
        // A stack machine, one instance per thread, running the same postfix
        // program over its own element. See the long note in backend.hpp for
        // why an interpreter rather than NVRTC.

        template <class T>
        struct FusedInputs {
            const T* p[FusedProgram::MaxInputs];
        };

        template <class T>
        __global__ void kFused(FusedProgram prog, std::size_t n, FusedInputs<T> in, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                // Lives in local memory rather than registers, because the
                // indices are not compile-time constants. That is fine here:
                // it is L1-resident, reused every iteration of the grid-stride
                // loop, and these kernels are waiting on global memory anyway.
                T stack[FusedProgram::StackDepth];
                int sp = 0;
                for (int k = 0; k < prog.nOps; ++k) {
                    switch (prog.code[k]) {
                        case 0: stack[sp++] = in.p[prog.arg[k]][i]; break;
                        case 1: stack[sp++] = T(prog.imm[k]); break;
                        case 2: stack[sp - 1] = applyUn<T>(prog.arg[k], stack[sp - 1]); break;
                        default: {
                            const T rhs = stack[--sp];
                            stack[sp - 1] = applyBin<T>(prog.arg[k], stack[sp - 1], rhs);
                            break;
                        }
                    }
                }
                out[i] = stack[0];
            }
        }

        // The stack above is indexed by a runtime value, so nvcc must place it
        // in LOCAL memory -- which is off-chip, and on a memory-bound chain
        // that is most of the cost. Measured against cupy.fuse, whose NVRTC
        // path emits straight-line code with no stack at all, the interpreter
        // was 1.58x behind on a chain of cheap ops (and level with it on one
        // dominated by fp64 transcendentals, where neither is memory-bound).
        //
        // Almost every real expression is LEFT-LINEAR -- a running value with
        // one operand folded in at a time -- and never needs more than two
        // stack slots. Two slots fit in two named registers, and "which slot"
        // becomes a predicated select rather than an address, so the local
        // memory disappears entirely. Deeper trees fall back to the stack
        // machine, which is still correct, just slower.
        template <class T>
        __global__ void kFusedLinear(FusedProgram prog, std::size_t n, FusedInputs<T> in, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                T a = T(0), b = T(0);
                int sp = 0;
                for (int k = 0; k < prog.nOps; ++k) {
                    switch (prog.code[k]) {
                        case 0: {
                            const T v = in.p[prog.arg[k]][i];
                            if (sp == 0) a = v; else b = v;
                            ++sp;
                            break;
                        }
                        case 1: {
                            const T v = T(prog.imm[k]);
                            if (sp == 0) a = v; else b = v;
                            ++sp;
                            break;
                        }
                        case 2:
                            if (sp == 1) a = applyUn<T>(prog.arg[k], a);
                            else b = applyUn<T>(prog.arg[k], b);
                            break;
                        default:
                            a = applyBin<T>(prog.arg[k], a, b);
                            sp = 1;
                            break;
                    }
                }
                out[i] = a;
            }
        }

        template <class T>
        static void fusedImpl(const FusedProgram& prog, std::size_t n, const T* const* inputs,
                              int nInputs, T* out) {
            if (!n) return;
            if (prog.nOps <= 0 || prog.nOps > FusedProgram::MaxOps)
                throw Error("fusedElementwise: program has " + std::to_string(prog.nOps) +
                               " ops, limit is " + std::to_string(FusedProgram::MaxOps));
            if (nInputs < 0 || nInputs > FusedProgram::MaxInputs)
                throw Error("fusedElementwise: " + std::to_string(nInputs) +
                               " inputs, limit is " + std::to_string(FusedProgram::MaxInputs));
            FusedInputs<T> in{};
            for (int i = 0; i < nInputs; ++i) in.p[i] = inputs[i];
            // Both structs go in by value as kernel arguments; together they
            // are well under the 4 KB parameter limit, so the program never
            // has to be staged through device memory.
            if (prog.maxDepth <= 2)
                kFusedLinear<T><<<gridFor(n), kBlock>>>(prog, n, in, out);
            else
                kFused<T><<<gridFor(n), kBlock>>>(prog, n, in, out);
            checkLaunch("fused element-wise kernel");
        }

        void fusedElementwise(const FusedProgram& prog, std::size_t n, const float* const* inputs,
                              int nInputs, float* out) {
            fusedImpl<float>(prog, n, inputs, nInputs, out);
        }
        void fusedElementwise(const FusedProgram& prog, std::size_t n, const double* const* inputs,
                              int nInputs, double* out) {
            fusedImpl<double>(prog, n, inputs, nInputs, out);
        }

        // ── Shape utilities ────────────────────────────────────────────

        template <class T>
        __global__ void kTriangle(int rows, int cols, T* A, bool upper, bool unitDiag) {
            const std::size_t n = (std::size_t)rows * cols;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / cols), c = (int)(i % cols);
                if (upper ? (c < r) : (c > r)) A[i] = T(0);
                else if (unitDiag && r == c) A[i] = T(1);
            }
        }

        template <class T>
        __global__ void kCopyBlock(int srcCols, const T* src, int r0, int c0, int nr, int nc,
                                   T* dst) {
            const std::size_t n = (std::size_t)nr * nc;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / nc), c = (int)(i % nc);
                dst[i] = src[(std::size_t)(r0 + r) * srcCols + (c0 + c)];
            }
        }

        template <class T>
        __global__ void kSetBlock(int dstCols, T* dst, int r0, int c0, int nr, int nc,
                                  const T* src) {
            const std::size_t n = (std::size_t)nr * nc;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / nc), c = (int)(i % nc);
                dst[(std::size_t)(r0 + r) * dstCols + (c0 + c)] = src[i];
            }
        }

        template <class T>
        static void setBlockImpl(int dstRows, int dstCols, T* dst, int r0, int c0, int nr, int nc,
                                 const T* src) {
            if (nr <= 0 || nc <= 0) return;
            if (r0 < 0 || c0 < 0 || r0 + nr > dstRows || c0 + nc > dstCols)
                throw Error("setBlock: the block does not fit inside the destination");
            kSetBlock<T><<<gridFor((std::size_t)nr * nc), kBlock>>>(dstCols, dst, r0, c0, nr, nc,
                                                                    src);
            checkLaunch("set block kernel");
        }
        void setBlock(int dr, int dc, float* d, int r0, int c0, int nr, int nc, const float* s) {
            setBlockImpl<float>(dr, dc, d, r0, c0, nr, nc, s);
        }
        void setBlock(int dr, int dc, double* d, int r0, int c0, int nr, int nc, const double* s) {
            setBlockImpl<double>(dr, dc, d, r0, c0, nr, nc, s);
        }

        template <class T>
        __global__ void kDiag(int cols, int k, T* A, T* d, bool set) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
                 i += blockDim.x * gridDim.x) {
                const std::size_t at = (std::size_t)i * cols + i;
                if (set) A[at] = d[i];
                else d[i] = A[at];
            }
        }

        template <class T>
        static void triangleImpl(int rows, int cols, T* A, bool upper, bool unitDiag) {
            if (rows <= 0 || cols <= 0) return;
            const std::size_t n = (std::size_t)rows * cols;
            kTriangle<T><<<gridFor(n), kBlock>>>(rows, cols, A, upper, unitDiag);
            checkLaunch("triangle kernel");
        }
        void triangle(int r, int c, float* A, bool u, bool ud) { triangleImpl<float>(r, c, A, u, ud); }
        void triangle(int r, int c, double* A, bool u, bool ud) {
            triangleImpl<double>(r, c, A, u, ud);
        }

        template <class T>
        static void copyBlockImpl(int srcRows, int srcCols, const T* src, int r0, int c0, int nr,
                                  int nc, T* dst) {
            if (nr <= 0 || nc <= 0) return;
            if (r0 < 0 || c0 < 0 || r0 + nr > srcRows || c0 + nc > srcCols)
                throw Error("copyBlock: requested block falls outside the source matrix");
            kCopyBlock<T><<<gridFor((std::size_t)nr * nc), kBlock>>>(srcCols, src, r0, c0, nr, nc,
                                                                    dst);
            checkLaunch("block copy kernel");
        }
        void copyBlock(int sr, int sc, const float* s, int r0, int c0, int nr, int nc, float* d) {
            copyBlockImpl<float>(sr, sc, s, r0, c0, nr, nc, d);
        }
        void copyBlock(int sr, int sc, const double* s, int r0, int c0, int nr, int nc, double* d) {
            copyBlockImpl<double>(sr, sc, s, r0, c0, nr, nc, d);
        }

        template <class T>
        static void diagImpl(int rows, int cols, T* A, T* d, bool set) {
            const int k = rows < cols ? rows : cols;
            if (k <= 0) return;
            kDiag<T><<<(k + kBlock - 1) / kBlock, kBlock>>>(cols, k, A, d, set);
            checkLaunch("diagonal kernel");
        }
        void setDiagonal(int r, int c, float* A, const float* d) {
            diagImpl<float>(r, c, A, const_cast<float*>(d), true);
        }
        void setDiagonal(int r, int c, double* A, const double* d) {
            diagImpl<double>(r, c, A, const_cast<double*>(d), true);
        }
        void getDiagonal(int r, int c, const float* A, float* d) {
            diagImpl<float>(r, c, const_cast<float*>(A), d, false);
        }
        void getDiagonal(int r, int c, const double* A, double* d) {
            diagImpl<double>(r, c, const_cast<double*>(A), d, false);
        }

        // ── Reductions ─────────────────────────────────────────────────
        //
        // Two-stage, no atomics. Stage 1 reduces the input to one partial per
        // block; stage 2 reduces the partials with a single block. Atomics
        // would be shorter but make float sums non-deterministic run to run,
        // and a library whose answers wobble is a library nobody can test
        // against NumPy.

        template <class T>
        __device__ __forceinline__ T redInit(int op) {
            switch ((RedOp)op) {
                case RedOp::Sum:
                case RedOp::SumAbs:
                case RedOp::SumSq: return T(0);
                case RedOp::Prod: return T(1);
                // INFINITY is a float constant but converts exactly to a
                // double infinity, so one spelling serves both instantiations.
                case RedOp::Max: return -T(INFINITY);
                case RedOp::Min: return T(INFINITY);
            }
            return T(0);
        }

        // Folds a raw input element into the accumulator.
        template <class T>
        __device__ __forceinline__ T redTake(int op, T acc, T x) {
            switch ((RedOp)op) {
                case RedOp::Sum: return acc + x;
                case RedOp::SumAbs: return acc + fabs(x);
                case RedOp::SumSq: return acc + x * x;
                case RedOp::Prod: return acc * x;
                case RedOp::Max: return acc > x ? acc : x;
                case RedOp::Min: return acc < x ? acc : x;
            }
            return acc;
        }

        // Combines two ALREADY-folded partials. Distinct from redTake: SumAbs
        // and SumSq must not re-apply abs/square to a partial that has already
        // had them applied.
        template <class T>
        __device__ __forceinline__ T redJoin(int op, T a, T b) {
            switch ((RedOp)op) {
                case RedOp::Sum:
                case RedOp::SumAbs:
                case RedOp::SumSq: return a + b;
                case RedOp::Prod: return a * b;
                case RedOp::Max: return a > b ? a : b;
                case RedOp::Min: return a < b ? a : b;
            }
            return a;
        }

        template <class T>
        __device__ T blockReduce(int op, T v) {
            __shared__ T s[kBlock];
            const int t = threadIdx.x;
            s[t] = v;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (t < stride) s[t] = redJoin<T>(op, s[t], s[t + stride]);
                __syncthreads();
            }
            return s[0];
        }

        // stage == 0 folds raw input (applies abs/square); stage == 1 joins
        // partials that have already been folded.
        template <class T>
        __global__ void kReduce(int op, int stage, std::size_t n, const T* a, T* partial) {
            T acc = redInit<T>(op);
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                acc = stage == 0 ? redTake<T>(op, acc, a[i]) : redJoin<T>(op, acc, a[i]);
            T r = blockReduce<T>(op, acc);
            if (threadIdx.x == 0) partial[blockIdx.x] = r;
        }

        template <class T>
        static T reduceImpl(RedOp op, std::size_t n, const T* a) {
            if (n == 0)
                throw Error("reduce: empty input has no defined result");
            int blocks = gridFor(n);
            if (blocks > 1024) blocks = 1024;   // stage 2 is a single block
            T* partial = (T*)devAlloc(sizeof(T) * (std::size_t)blocks);
            T out{};
            try {
                kReduce<T><<<blocks, kBlock>>>((int)op, 0, n, a, partial);
                checkLaunch("reduce stage 1");
                if (blocks > 1) {
                    kReduce<T><<<1, kBlock>>>((int)op, 1, (std::size_t)blocks, partial, partial);
                    checkLaunch("reduce stage 2");
                }
                copyD2H(&out, partial, sizeof(T));
            } catch (...) {
                devFree(partial);
                throw;
            }
            devFree(partial);
            return out;
        }
        float reduce(RedOp op, std::size_t n, const float* a) { return reduceImpl<float>(op, n, a); }
        double reduce(RedOp op, std::size_t n, const double* a) {
            return reduceImpl<double>(op, n, a);
        }

        // Row reduction: one block per row. A row is contiguous in row-major,
        // so threads sweeping it read consecutive addresses — coalesced.
        template <class T>
        __global__ void kReduceRows(int op, int rows, int cols, const T* a, T* out) {
            const int r = blockIdx.x;
            if (r >= rows) return;
            T acc = redInit<T>(op);
            for (int c = threadIdx.x; c < cols; c += blockDim.x)
                acc = redTake<T>(op, acc, a[(std::size_t)r * cols + c]);
            T v = blockReduce<T>(op, acc);
            if (threadIdx.x == 0) out[r] = v;
        }

        // Column reduction: one THREAD per column, walking down the rows.
        // The obvious alternative — one block per column — would have every
        // thread in a warp reading the same column at a stride of `cols`,
        // which is the worst possible pattern. This way adjacent threads read
        // adjacent columns of the same row, so each warp's loads coalesce.
        template <class T>
        __global__ void kReduceCols(int op, int rows, int cols, const T* a, T* out) {
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= cols) return;
            T acc = redInit<T>(op);
            for (int r = 0; r < rows; ++r) acc = redTake<T>(op, acc, a[(std::size_t)r * cols + c]);
            out[c] = acc;
        }

        template <class T>
        static void reduceAxisImpl(RedOp op, int rows, int cols, const T* a, T* out, bool byRow) {
            if (rows <= 0 || cols <= 0) return;
            if (byRow) {
                kReduceRows<T><<<rows, kBlock>>>((int)op, rows, cols, a, out);
                checkLaunch("row reduction");
            } else {
                kReduceCols<T><<<(cols + kBlock - 1) / kBlock, kBlock>>>((int)op, rows, cols, a,
                                                                        out);
                checkLaunch("column reduction");
            }
        }
        void reduceAxis(RedOp op, int r, int c, const float* a, float* o, bool byRow) {
            reduceAxisImpl<float>(op, r, c, a, o, byRow);
        }
        void reduceAxis(RedOp op, int r, int c, const double* a, double* o, bool byRow) {
            reduceAxisImpl<double>(op, r, c, a, o, byRow);
        }

        // ── Random (cuRAND) ────────────────────────────────────────────

        namespace {
            curandGenerator_t g_rng = nullptr;
            std::once_flag g_rngOnce;
        }  // namespace

        // Seed convention, matching what set_Ran_values does on the CPU side:
        //
        //   seed != 0   reset the stream to that seed, so the same seed always
        //               produces the same numbers
        //   seed == 0   leave the stream where it is, so consecutive calls
        //               keep drawing fresh values
        //
        // The earlier version of this cached the last seed and skipped the
        // reseed when it matched, which quietly broke reproducibility: asking
        // twice for seed 1234 continued the stream instead of restarting it,
        // and the two results differed. Caching a seed saves nothing anyway -
        // curandSetPseudoRandomGeneratorSeed is a host-side field write.
        static curandGenerator_t rng(unsigned long long seed) {
            std::call_once(g_rngOnce, [] {
                checkRand(curandCreateGenerator(&g_rng, CURAND_RNG_PSEUDO_DEFAULT),
                          "curandCreateGenerator");
            });
            if (seed != 0) {
                checkRand(curandSetPseudoRandomGeneratorSeed(g_rng, seed),
                          "curandSetPseudoRandomGeneratorSeed");
                // Setting the seed alone does not rewind the position within
                // the sequence; without this the second call to a given seed
                // starts wherever the first one stopped.
                checkRand(curandSetGeneratorOffset(g_rng, 0), "curandSetGeneratorOffset");
            }
            return g_rng;
        }

        // cuRAND's uniform is (0, 1]. Rescaling with the two scalar kernels we
        // already have beats writing a third one: both passes are pure
        // bandwidth and the generator itself dominates.
        template <class T, class F>
        static void randUniformImpl(F gen, const char* name, std::size_t n, T* a, T lo, T hi,
                                    unsigned long long seed) {
            if (!n) return;
            checkRand(gen(rng(seed), a, n), name);
            if (lo != T(0) || hi != T(1)) {
                binaryScalarImpl<T>(BinOp::Mul, n, a, hi - lo, a, false);
                binaryScalarImpl<T>(BinOp::Add, n, a, lo, a, false);
            }
        }
        void randUniform(std::size_t n, float* a, float lo, float hi, unsigned long long s) {
            randUniformImpl<float>(curandGenerateUniform, "curandGenerateUniform", n, a, lo, hi, s);
        }
        void randUniform(std::size_t n, double* a, double lo, double hi, unsigned long long s) {
            randUniformImpl<double>(curandGenerateUniformDouble, "curandGenerateUniformDouble", n,
                                    a, lo, hi, s);
        }

        // cuRAND generates normals in Box-Muller PAIRS, so it rejects an odd
        // count outright. Round up into scratch and copy back the prefix.
        template <class T, class F>
        static void randNormalImpl(F gen, const char* name, std::size_t n, T* a, T mean, T sd,
                                   unsigned long long seed) {
            if (!n) return;
            if (n % 2 == 0) {
                checkRand(gen(rng(seed), a, n, mean, sd), name);
                return;
            }
            T* tmp = (T*)devAlloc(sizeof(T) * (n + 1));
            try {
                checkRand(gen(rng(seed), tmp, n + 1, mean, sd), name);
                copyD2D(a, tmp, sizeof(T) * n);
            } catch (...) {
                devFree(tmp);
                throw;
            }
            devFree(tmp);
        }
        void randNormal(std::size_t n, float* a, float m, float sd, unsigned long long s) {
            randNormalImpl<float>(curandGenerateNormal, "curandGenerateNormal", n, a, m, sd, s);
        }
        void randNormal(std::size_t n, double* a, double m, double sd, unsigned long long s) {
            randNormalImpl<double>(curandGenerateNormalDouble, "curandGenerateNormalDouble", n, a,
                                   m, sd, s);
        }

        // ── cuSOLVER plumbing ──────────────────────────────────────────
        //
        // Every routine follows the same three steps: ask for a workspace
        // size, allocate it, run. These two helpers make that RAII so an
        // exception mid-factorisation cannot leak device memory.

        namespace {
            struct Scratch {
                void* p = nullptr;
                explicit Scratch(std::size_t bytes) : p(bytes ? devAlloc(bytes) : nullptr) {}
                ~Scratch() { devFree(p); }
                Scratch(const Scratch&) = delete;
                Scratch& operator=(const Scratch&) = delete;
                template <class T>
                T* as() const { return (T*)p; }
            };

            // cuSOLVER reports per-call status through a device int. Reading
            // it back is also the synchronisation point that surfaces any
            // asynchronous kernel fault from inside the factorisation.
            struct Info {
                int* d = nullptr;
                Info() : d((int*)devAlloc(sizeof(int))) { devZero(d, sizeof(int)); }
                ~Info() { devFree(d); }
                Info(const Info&) = delete;
                Info& operator=(const Info&) = delete;
                int get() const {
                    int h = 0;
                    copyD2H(&h, d, sizeof(int));
                    return h;
                }
            };
        }  // namespace

        // Column-major transpose of a rows x cols matrix into a cols x rows
        // one. The public transpose() above is this same geam call with its
        // arguments read row-major; kept separate because the factorisations
        // genuinely work in column-major and conflating the two is how sign
        // and shape bugs get in.
        template <class T, class F>
        static void transposeCM(F f, int rows, int cols, const T* A, T* out) {
            const T one = T(1), zero = T(0);
            checkBlas(f(blas(), CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &one, A, rows, &zero, out,
                        cols, out, cols),
                      "cublas geam (column-major transpose)");
        }

        // ── LU ─────────────────────────────────────────────────────────

        int getrf(int m, int n, double* A, int* ipiv) {
            int lwork = 0;
            checkSolver(cusolverDnDgetrf_bufferSize(solver(), m, n, A, m, &lwork),
                        "cusolverDnDgetrf_bufferSize");
            Scratch w(sizeof(double) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnDgetrf(solver(), m, n, A, m, w.as<double>(), ipiv, info.d),
                        "cusolverDnDgetrf");
            return info.get();
        }
        int getrf(int m, int n, float* A, int* ipiv) {
            int lwork = 0;
            checkSolver(cusolverDnSgetrf_bufferSize(solver(), m, n, A, m, &lwork),
                        "cusolverDnSgetrf_bufferSize");
            Scratch w(sizeof(float) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnSgetrf(solver(), m, n, A, m, w.as<float>(), ipiv, info.d),
                        "cusolverDnSgetrf");
            return info.get();
        }

        int getrs(int n, int nrhs, const double* A, const int* ipiv, double* B) {
            Info info;
            checkSolver(cusolverDnDgetrs(solver(), CUBLAS_OP_N, n, nrhs, A, n, ipiv, B, n, info.d),
                        "cusolverDnDgetrs");
            return info.get();
        }
        int getrs(int n, int nrhs, const float* A, const int* ipiv, float* B) {
            Info info;
            checkSolver(cusolverDnSgetrs(solver(), CUBLAS_OP_N, n, nrhs, A, n, ipiv, B, n, info.d),
                        "cusolverDnSgetrs");
            return info.get();
        }

        // ── Mixed-precision solve ──────────────────────────────────────

        int gesvMixed(int n, int nrhs, double* A, double* B, double* X, Factor f, int* iters) {
            Scratch piv(sizeof(int) * (std::size_t)n);
            int* ipiv = (int*)piv.p;
            std::size_t lwork = 0;
            cusolverStatus_t st;
            int local = 0;
            int* it = iters ? iters : &local;

            // One shape of call per factorisation precision. They differ only
            // in the letter, but the entry points are distinct symbols, so a
            // switch is the whole dispatch.
            #define MG_GESV(FN)                                                                   \
                st = FN##_bufferSize(solver(), n, nrhs, A, n, ipiv, B, n, X, n, nullptr, &lwork);  \
                if (st != CUSOLVER_STATUS_SUCCESS) checkSolver(st, #FN "_bufferSize");             \
                {                                                                                  \
                    Scratch w(lwork);                                                              \
                    Info info;                                                                     \
                    checkSolver(FN(solver(), n, nrhs, A, n, ipiv, B, n, X, n, w.p, lwork, it,      \
                                   info.d),                                                        \
                                #FN);                                                              \
                    return info.get();                                                             \
                }

            switch (f) {
                case Factor::Single: MG_GESV(cusolverDnDSgesv)
                case Factor::Half: MG_GESV(cusolverDnDHgesv)
                case Factor::BFloat16: MG_GESV(cusolverDnDBgesv)
                default: MG_GESV(cusolverDnDDgesv)
            }
            #undef MG_GESV
        }

        int gesvMixed(int n, int nrhs, float* A, float* B, float* X, Factor f, int* iters) {
            Scratch piv(sizeof(int) * (std::size_t)n);
            int* ipiv = (int*)piv.p;
            std::size_t lwork = 0;
            cusolverStatus_t st;
            int local = 0;
            int* it = iters ? iters : &local;

            #define MG_GESVF(FN)                                                                  \
                st = FN##_bufferSize(solver(), n, nrhs, A, n, ipiv, B, n, X, n, nullptr, &lwork);  \
                if (st != CUSOLVER_STATUS_SUCCESS) checkSolver(st, #FN "_bufferSize");             \
                {                                                                                  \
                    Scratch w(lwork);                                                              \
                    Info info;                                                                     \
                    checkSolver(FN(solver(), n, nrhs, A, n, ipiv, B, n, X, n, w.p, lwork, it,      \
                                   info.d),                                                        \
                                #FN);                                                              \
                    return info.get();                                                             \
                }

            switch (f) {
                case Factor::Half: MG_GESVF(cusolverDnSHgesv)
                case Factor::BFloat16:
                    throw Error("solveMixed: cuSOLVER has no bfloat16 path for a float system");
                default: MG_GESVF(cusolverDnSSgesv)
            }
            #undef MG_GESVF
        }

        // ── Cholesky ───────────────────────────────────────────────────

        int potrf(int n, double* A, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            int lwork = 0;
            checkSolver(cusolverDnDpotrf_bufferSize(solver(), uplo, n, A, n, &lwork),
                        "cusolverDnDpotrf_bufferSize");
            Scratch w(sizeof(double) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnDpotrf(solver(), uplo, n, A, n, w.as<double>(), lwork, info.d),
                        "cusolverDnDpotrf");
            return info.get();
        }
        int potrf(int n, float* A, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            int lwork = 0;
            checkSolver(cusolverDnSpotrf_bufferSize(solver(), uplo, n, A, n, &lwork),
                        "cusolverDnSpotrf_bufferSize");
            Scratch w(sizeof(float) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnSpotrf(solver(), uplo, n, A, n, w.as<float>(), lwork, info.d),
                        "cusolverDnSpotrf");
            return info.get();
        }

        int potrs(int n, int nrhs, const double* A, double* B, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            Info info;
            checkSolver(cusolverDnDpotrs(solver(), uplo, n, nrhs, A, n, B, n, info.d),
                        "cusolverDnDpotrs");
            return info.get();
        }
        int potrs(int n, int nrhs, const float* A, float* B, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            Info info;
            checkSolver(cusolverDnSpotrs(solver(), uplo, n, nrhs, A, n, B, n, info.d),
                        "cusolverDnSpotrs");
            return info.get();
        }

        // ── QR ─────────────────────────────────────────────────────────

        int geqrf(int m, int n, double* A, double* tau) {
            int lwork = 0;
            checkSolver(cusolverDnDgeqrf_bufferSize(solver(), m, n, A, m, &lwork),
                        "cusolverDnDgeqrf_bufferSize");
            Scratch w(sizeof(double) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnDgeqrf(solver(), m, n, A, m, tau, w.as<double>(), lwork, info.d),
                        "cusolverDnDgeqrf");
            return info.get();
        }
        int geqrf(int m, int n, float* A, float* tau) {
            int lwork = 0;
            checkSolver(cusolverDnSgeqrf_bufferSize(solver(), m, n, A, m, &lwork),
                        "cusolverDnSgeqrf_bufferSize");
            Scratch w(sizeof(float) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnSgeqrf(solver(), m, n, A, m, tau, w.as<float>(), lwork, info.d),
                        "cusolverDnSgeqrf");
            return info.get();
        }

        int orgqr(int m, int n, int k, double* A, const double* tau) {
            int lwork = 0;
            checkSolver(cusolverDnDorgqr_bufferSize(solver(), m, n, k, A, m, tau, &lwork),
                        "cusolverDnDorgqr_bufferSize");
            Scratch w(sizeof(double) * (std::size_t)lwork);
            Info info;
            checkSolver(
                cusolverDnDorgqr(solver(), m, n, k, A, m, tau, w.as<double>(), lwork, info.d),
                "cusolverDnDorgqr");
            return info.get();
        }
        int orgqr(int m, int n, int k, float* A, const float* tau) {
            int lwork = 0;
            checkSolver(cusolverDnSorgqr_bufferSize(solver(), m, n, k, A, m, tau, &lwork),
                        "cusolverDnSorgqr_bufferSize");
            Scratch w(sizeof(float) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnSorgqr(solver(), m, n, k, A, m, tau, w.as<float>(), lwork, info.d),
                        "cusolverDnSorgqr");
            return info.get();
        }

        // ── SVD ────────────────────────────────────────────────────────
        //
        // cusolverDnXgesvd only accepts m >= n. Rather than push that
        // restriction onto every caller, the wide case is handled here by the
        // standard transpose identity:
        //
        //     A = U S V^T   <=>   A^T = V S U^T
        //
        // so factorising A^T (which is tall, hence legal) and swapping the
        // roles of the two orthogonal factors gives A's decomposition. The
        // singular values are identical either way, and both the full and
        // economy shapes come out right — worked through in the comments at
        // the swap below.

        template <class T, class FBuf, class FSvd, class FGeam>
        static int gesvdImpl(FBuf fbuf, FSvd fsvd, FGeam fgeam, const char* name, int m, int n,
                             T* A, T* S, T* U, T* VT, bool full) {
            const int k = m < n ? m : n;
            const signed char job = full ? 'A' : 'S';
            // A null factor means "do not compute it". cuSOLVER spells that
            // 'N', and it then touches neither the pointer nor its size, which
            // is what makes a values-only SVD cheaper than throwing the
            // factors away afterwards.
            const signed char jobu = U ? job : 'N';
            const signed char jobvt = VT ? job : 'N';

            if (m >= n) {
                int lwork = 0;
                checkSolver(fbuf(solver(), m, n, &lwork), "gesvd_bufferSize");
                Scratch w(sizeof(T) * (std::size_t)lwork);
                Info info;
                const int ldu = m;
                const int ldvt = full ? n : k;
                checkSolver(fsvd(solver(), jobu, jobvt, m, n, A, m, S, U, ldu, VT, ldvt,
                                 w.template as<T>(), lwork, nullptr, info.d),
                            name);
                return info.get();
            }

            // Wide case. Factorise A^T (n x m, tall).
            Scratch At(sizeof(T) * (std::size_t)m * n);
            transposeCM<T>(fgeam, m, n, A, At.template as<T>());

            // Shapes of the transposed problem, where min(n, m) is still k = m:
            //   full   U_t is n x n, VT_t is m x m
            //   econ   U_t is n x k, VT_t is k x m   (k == m here)
            const int utCols = full ? n : k;
            const int vtRows = k;
            const int vtCols = full ? m : m;
            // Roles swap in the transposed problem, so a caller who wanted
            // only U needs only VT_t, and vice versa.
            const bool needUt = (VT != nullptr);
            const bool needVTt = (U != nullptr);
            Scratch Ut(needUt ? sizeof(T) * (std::size_t)n * utCols : 0);
            Scratch VTt(needVTt ? sizeof(T) * (std::size_t)vtRows * vtCols : 0);

            int lwork = 0;
            checkSolver(fbuf(solver(), n, m, &lwork), "gesvd_bufferSize");
            Scratch w(sizeof(T) * (std::size_t)lwork);
            Info info;
            checkSolver(fsvd(solver(), needUt ? job : 'N', needVTt ? job : 'N', n, m,
                             At.template as<T>(), n, S, Ut.template as<T>(), n,
                             VTt.template as<T>(), vtRows, w.template as<T>(), lwork, nullptr,
                             info.d),
                        name);
            const int st = info.get();
            if (st != 0) return st;

            // A = (A^T)^T = (U_t S VT_t)^T = VT_t^T S U_t^T, so
            //     U  = VT_t^T   (full: m x m,  econ: m x k)
            //     VT = U_t^T    (full: n x n,  econ: k x n)
            if (U) transposeCM<T>(fgeam, vtRows, vtCols, VTt.template as<T>(), U);
            if (VT) transposeCM<T>(fgeam, n, utCols, Ut.template as<T>(), VT);
            return 0;
        }

        int gesvd(int m, int n, double* A, double* S, double* U, double* VT, bool full) {
            return gesvdImpl<double>(cusolverDnDgesvd_bufferSize, cusolverDnDgesvd, cublasDgeam,
                                     "cusolverDnDgesvd", m, n, A, S, U, VT, full);
        }
        int gesvd(int m, int n, float* A, float* S, float* U, float* VT, bool full) {
            return gesvdImpl<float>(cusolverDnSgesvd_bufferSize, cusolverDnSgesvd, cublasSgeam,
                                    "cusolverDnSgesvd", m, n, A, S, U, VT, full);
        }

        // ── Symmetric eigenproblem ─────────────────────────────────────

        int syevd(int n, double* A, double* w, bool vectors) {
            cusolverEigMode_t jobz =
                vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
            int lwork = 0;
            checkSolver(cusolverDnDsyevd_bufferSize(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, A, n,
                                                    w, &lwork),
                        "cusolverDnDsyevd_bufferSize");
            Scratch ws(sizeof(double) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnDsyevd(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, A, n, w,
                                         ws.as<double>(), lwork, info.d),
                        "cusolverDnDsyevd");
            return info.get();
        }
        int syevd(int n, float* A, float* w, bool vectors) {
            cusolverEigMode_t jobz =
                vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
            int lwork = 0;
            checkSolver(cusolverDnSsyevd_bufferSize(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, A, n,
                                                    w, &lwork),
                        "cusolverDnSsyevd_bufferSize");
            Scratch ws(sizeof(float) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnSsyevd(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, A, n, w,
                                         ws.as<float>(), lwork, info.d),
                        "cusolverDnSsyevd");
            return info.get();
        }

        // ── FFT (cuFFT) ────────────────────────────────────────────────

        static const char* fftMsg(cufftResult r) {
            switch (r) {
                case CUFFT_SUCCESS: return "success";
                case CUFFT_INVALID_PLAN: return "invalid plan";
                case CUFFT_ALLOC_FAILED: return "allocation failed";
                case CUFFT_INVALID_VALUE: return "invalid value";
                case CUFFT_INTERNAL_ERROR: return "internal error";
                case CUFFT_EXEC_FAILED: return "execution failed";
                case CUFFT_SETUP_FAILED: return "setup failed";
                case CUFFT_INVALID_SIZE: return "invalid size";
                case CUFFT_NOT_SUPPORTED: return "not supported";
                default: return "unknown cuFFT error";
            }
        }
        static void checkFft(cufftResult r, const char* what) {
            if (r != CUFFT_SUCCESS) throw Error(std::string(what) + ": " + fftMsg(r));
        }

        // Plans cost milliseconds to build and are entirely determined by the
        // shape, so they are cached. Without this a loop calling fft() on the
        // same size pays the setup every iteration and the transform stops
        // being the cost.
        namespace {
            struct PlanKey {
                int rank, n0, n1, batch, stride, dist, type;
                bool operator==(const PlanKey& o) const {
                    return rank == o.rank && n0 == o.n0 && n1 == o.n1 && batch == o.batch &&
                           stride == o.stride && dist == o.dist && type == o.type;
                }
            };
            struct PlanHash {
                std::size_t operator()(const PlanKey& k) const {
                    std::size_t h = 1469598103934665603ull;
                    for (int v : {k.rank, k.n0, k.n1, k.batch, k.stride, k.dist, k.type})
                        h = (h ^ (std::size_t)v) * 1099511628211ull;
                    return h;
                }
            };
            std::mutex g_planMu;
            std::unordered_map<PlanKey, cufftHandle, PlanHash> g_plans;
        }  // namespace

        static cufftHandle planFor(const PlanKey& k) {
            std::lock_guard<std::mutex> lk(g_planMu);
            auto it = g_plans.find(k);
            if (it != g_plans.end()) return it->second;
            cufftHandle p{};
            if (k.rank == 1) {
                int n = k.n0;
                // inembed/onembed MUST be non-null. With NULL, cuFFT assumes a
                // simple contiguous layout and SILENTLY IGNORES istride and
                // idist -- so the down-columns plan, which is entirely a
                // question of stride, came back transforming rows instead. The
                // along-rows case happened to be contiguous and looked fine,
                // which is exactly how this hides.
                checkFft(cufftPlanMany(&p, 1, &n, &n, k.stride, k.dist, &n, k.stride, k.dist,
                                       (cufftType)k.type, k.batch),
                         "cufftPlanMany");
            } else {
                checkFft(cufftPlan2d(&p, k.n0, k.n1, (cufftType)k.type), "cufftPlan2d");
            }
            g_plans.emplace(k, p);
            return p;
        }

        void fftRelease() {
            std::lock_guard<std::mutex> lk(g_planMu);
            for (auto& kv : g_plans) cufftDestroy(kv.second);
            g_plans.clear();
        }

        // Split <-> interleaved. inIm may be null, meaning a purely real input,
        // which is the common case and saves the caller allocating a zero
        // matrix just to throw it away.
        template <class T>
        __global__ void kInterleave(std::size_t n, const T* re, const T* im, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                out[2 * i] = re[i];
                out[2 * i + 1] = im ? im[i] : T(0);
            }
        }
        template <class T>
        __global__ void kDeinterleave(std::size_t n, const T* in, T* re, T* im, T scale) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                re[i] = in[2 * i] * scale;
                im[i] = in[2 * i + 1] * scale;
            }
        }

        template <class T>
        struct FftTraits;
        template <>
        struct FftTraits<float> {
            static constexpr cufftType type = CUFFT_C2C;
            static cufftResult exec(cufftHandle p, void* d, int dir) {
                return cufftExecC2C(p, (cufftComplex*)d, (cufftComplex*)d, dir);
            }
            static cufftResult exec(cufftHandle p, std::complex<float>* d, int dir) {
                return cufftExecC2C(p, (cufftComplex*)d, (cufftComplex*)d, dir);
            }
        };
        template <>
        struct FftTraits<double> {
            static constexpr cufftType type = CUFFT_Z2Z;
            static cufftResult exec(cufftHandle p, void* d, int dir) {
                return cufftExecZ2Z(p, (cufftDoubleComplex*)d, (cufftDoubleComplex*)d, dir);
            }
            static cufftResult exec(cufftHandle p, std::complex<double>* d, int dir) {
                return cufftExecZ2Z(p, (cufftDoubleComplex*)d, (cufftDoubleComplex*)d, dir);
            }
        };

        template <class T>
        static void fftRun(const PlanKey& key, std::size_t total, int n, const T* inRe,
                           const T* inIm, T* outRe, T* outIm, bool inverse) {
            if (!total) return;
            Scratch buf(sizeof(T) * 2 * total);
            kInterleave<T><<<gridFor(total), kBlock>>>(total, inRe, inIm, buf.template as<T>());
            checkLaunch("fft interleave");
            checkFft(FftTraits<T>::exec(planFor(key), buf.p,
                                        inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                     "cufftExec");
            // cuFFT's inverse is unnormalised; 1/n here is what makes
            // ifft(fft(x)) == x, matching NumPy, MATLAB and basic/signal.hpp.
            const T scale = inverse ? T(1) / T(n) : T(1);
            kDeinterleave<T><<<gridFor(total), kBlock>>>(total, buf.template as<T>(), outRe, outIm,
                                                         scale);
            checkLaunch("fft deinterleave");
        }

        template <class T>
        static void fft1dImpl(int batch, int n, int stride, int dist, const T* inRe, const T* inIm,
                              T* outRe, T* outIm, bool inverse) {
            if (batch <= 0 || n <= 0) return;
            PlanKey k{1, n, 0, batch, stride, dist, (int)FftTraits<T>::type};
            fftRun<T>(k, (std::size_t)batch * n, n, inRe, inIm, outRe, outIm, inverse);
        }
        void fft1d(int b, int n, int st, int di, const float* ir, const float* ii, float* orr,
                   float* oi, bool inv) {
            fft1dImpl<float>(b, n, st, di, ir, ii, orr, oi, inv);
        }
        void fft1d(int b, int n, int st, int di, const double* ir, const double* ii, double* orr,
                   double* oi, bool inv) {
            fft1dImpl<double>(b, n, st, di, ir, ii, orr, oi, inv);
        }

        template <class T>
        static void fft2dImpl(int rows, int cols, const T* inRe, const T* inIm, T* outRe, T* outIm,
                              bool inverse) {
            if (rows <= 0 || cols <= 0) return;
            PlanKey k{2, rows, cols, 1, 1, 1, (int)FftTraits<T>::type};
            // A 2-D inverse divides by the FULL element count, not one side.
            fftRun<T>(k, (std::size_t)rows * cols, rows * cols, inRe, inIm, outRe, outIm, inverse);
        }
        void fft2d(int r, int c, const float* ir, const float* ii, float* orr, float* oi, bool inv) {
            fft2dImpl<float>(r, c, ir, ii, orr, oi, inv);
        }
        void fft2d(int r, int c, const double* ir, const double* ii, double* orr, double* oi,
                   bool inv) {
            fft2dImpl<double>(r, c, ir, ii, orr, oi, inv);
        }

        // ══════════════════════════════════════════════════════════════
        //  COMPLEX
        // ══════════════════════════════════════════════════════════════
        //
        // The complex half of the package. Everything here mirrors a real
        // counterpart above; what differs is only that Max/Min/floor/ceil and
        // the rest of the ordering-based operations are absent, because they
        // have no meaning on a field that is not ordered. The public API does
        // not expose them for complex types, so the gaps are unreachable
        // rather than silently wrong.

        template <class R>
        static Cx<R>* cx(std::complex<R>* p) { return reinterpret_cast<Cx<R>*>(p); }
        template <class R>
        static const Cx<R>* cx(const std::complex<R>* p) {
            return reinterpret_cast<const Cx<R>*>(p);
        }

        template <class R>
        __device__ __forceinline__ Cx<R> applyBinCx(int op, Cx<R> a, Cx<R> b) {
            switch ((BinOp)op) {
                case BinOp::Add: return a + b;
                case BinOp::Sub: return a - b;
                case BinOp::Mul: return a * b;
                case BinOp::Div: return a / b;
                case BinOp::Pow: return cxPow(a, b);
                default: return a;   // Max/Min: unreachable, see above
            }
        }

        template <class R>
        __device__ __forceinline__ Cx<R> applyUnCx(int op, Cx<R> z) {
            switch ((UnOp)op) {
                case UnOp::Neg: return -z;
                case UnOp::Sqrt: return cxSqrt(z);
                case UnOp::Exp: return cxExp(z);
                case UnOp::Log: return cxLog(z);
                case UnOp::Log2: return cxLog(z) / Cx<R>(log(R(2)));
                case UnOp::Log10: return cxLog(z) / Cx<R>(log(R(10)));
                case UnOp::Exp2: return cxPow(Cx<R>(2), z);
                case UnOp::Sin: return cxSin(z);
                case UnOp::Cos: return cxCos(z);
                case UnOp::Tan: return cxSin(z) / cxCos(z);
                case UnOp::Sinh: return cxSinh(z);
                case UnOp::Cosh: return cxCosh(z);
                case UnOp::Tanh: return cxSinh(z) / cxCosh(z);
                case UnOp::Square: return z * z;
                case UnOp::Recip: return Cx<R>(1) / z;
                // Abs is deliberately NOT here: |z| is real, so it changes the
                // result type and goes through project() below instead.
                default: return z;
            }
        }

        template <class R>
        __global__ void kBinaryCx(int op, std::size_t n, const Cx<R>* a, const Cx<R>* b,
                                  Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = applyBinCx<R>(op, a[i], b[i]);
        }
        template <class R>
        __global__ void kBinaryScalarCx(int op, std::size_t n, const Cx<R>* a, Cx<R> s, Cx<R>* out,
                                        bool left) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = left ? applyBinCx<R>(op, s, a[i]) : applyBinCx<R>(op, a[i], s);
        }
        template <class R>
        __global__ void kUnaryCx(int op, std::size_t n, const Cx<R>* a, Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = applyUnCx<R>(op, a[i]);
        }
        template <class R>
        __global__ void kConjCx(std::size_t n, const Cx<R>* a, Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = cxConj(a[i]);
        }
        template <class R>
        __global__ void kFillCx(std::size_t n, Cx<R>* a, Cx<R> v) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                a[i] = v;
        }

        // Complex -> real (part 0 real, 1 imag, 2 |z|, 3 arg z) and back.
        template <class R>
        __global__ void kProject(int part, std::size_t n, const Cx<R>* a, R* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const Cx<R> z = a[i];
                out[i] = part == 0 ? z.x : part == 1 ? z.y : part == 2 ? cxAbs(z) : cxArg(z);
            }
        }
        template <class R>
        __global__ void kCompose(std::size_t n, const R* re, const R* im, Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = Cx<R>(re[i], im ? im[i] : R(0));
        }

        // Sum / product only: the other reductions rank their inputs, and the
        // complex numbers do not rank.
        template <class R>
        __global__ void kReduceCx(int op, int stage, std::size_t n, const Cx<R>* a, Cx<R>* part) {
            Cx<R> acc = ((RedOp)op == RedOp::Prod) ? Cx<R>(1) : Cx<R>(0);
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                acc = ((RedOp)op == RedOp::Prod) ? acc * a[i] : acc + a[i];
            (void)stage;
            __shared__ Cx<R> sh[kBlock];
            const int t = threadIdx.x;
            sh[t] = acc;
            __syncthreads();
            for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
                if (t < s2) sh[t] = ((RedOp)op == RedOp::Prod) ? sh[t] * sh[t + s2]
                                                               : sh[t] + sh[t + s2];
                __syncthreads();
            }
            if (t == 0) part[blockIdx.x] = sh[0];
        }

        // sum |z|^2, real-valued -- the Frobenius norm's inside.
        template <class R>
        __global__ void kNormSq(std::size_t n, const Cx<R>* a, R* part) {
            R acc = 0;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const Cx<R> z = a[i];
                acc += z.x * z.x + z.y * z.y;
            }
            R r = blockReduce<R>((int)RedOp::Sum, acc);
            if (threadIdx.x == 0) part[blockIdx.x] = r;
        }

        // --- entry points ---

        template <class R>
        static void binaryCxImpl(BinOp op, std::size_t n, const std::complex<R>* a,
                                 const std::complex<R>* b, std::complex<R>* o) {
            if (!n) return;
            kBinaryCx<R><<<gridFor(n), kBlock>>>((int)op, n, cx(a), cx(b), cx(o));
            checkLaunch("complex element-wise binary");
        }
        void binary(BinOp op, std::size_t n, const std::complex<float>* a,
                    const std::complex<float>* b, std::complex<float>* o) {
            binaryCxImpl<float>(op, n, a, b, o);
        }
        void binary(BinOp op, std::size_t n, const std::complex<double>* a,
                    const std::complex<double>* b, std::complex<double>* o) {
            binaryCxImpl<double>(op, n, a, b, o);
        }

        template <class R>
        static void binaryScalarCxImpl(BinOp op, std::size_t n, const std::complex<R>* a,
                                       std::complex<R> s, std::complex<R>* o, bool left) {
            if (!n) return;
            kBinaryScalarCx<R><<<gridFor(n), kBlock>>>((int)op, n, cx(a),
                                                       Cx<R>(s.real(), s.imag()), cx(o), left);
            checkLaunch("complex element-wise scalar");
        }
        void binaryScalar(BinOp op, std::size_t n, const std::complex<float>* a,
                          std::complex<float> s, std::complex<float>* o, bool l) {
            binaryScalarCxImpl<float>(op, n, a, s, o, l);
        }
        void binaryScalar(BinOp op, std::size_t n, const std::complex<double>* a,
                          std::complex<double> s, std::complex<double>* o, bool l) {
            binaryScalarCxImpl<double>(op, n, a, s, o, l);
        }

        template <class R>
        static void unaryCxImpl(UnOp op, std::size_t n, const std::complex<R>* a,
                                std::complex<R>* o) {
            if (!n) return;
            kUnaryCx<R><<<gridFor(n), kBlock>>>((int)op, n, cx(a), cx(o));
            checkLaunch("complex element-wise unary");
        }
        void unary(UnOp op, std::size_t n, const std::complex<float>* a, std::complex<float>* o) {
            unaryCxImpl<float>(op, n, a, o);
        }
        void unary(UnOp op, std::size_t n, const std::complex<double>* a, std::complex<double>* o) {
            unaryCxImpl<double>(op, n, a, o);
        }

        template <class R>
        static void conjImpl(std::size_t n, const std::complex<R>* a, std::complex<R>* o) {
            if (!n) return;
            kConjCx<R><<<gridFor(n), kBlock>>>(n, cx(a), cx(o));
            checkLaunch("conj");
        }
        void conj(std::size_t n, const std::complex<float>* a, std::complex<float>* o) {
            conjImpl<float>(n, a, o);
        }
        void conj(std::size_t n, const std::complex<double>* a, std::complex<double>* o) {
            conjImpl<double>(n, a, o);
        }

        template <class R>
        static void fillCxImpl(std::size_t n, std::complex<R>* a, std::complex<R> v) {
            if (!n) return;
            kFillCx<R><<<gridFor(n), kBlock>>>(n, cx(a), Cx<R>(v.real(), v.imag()));
            checkLaunch("complex fill");
        }
        void fill(std::size_t n, std::complex<float>* a, std::complex<float> v) {
            fillCxImpl<float>(n, a, v);
        }
        void fill(std::size_t n, std::complex<double>* a, std::complex<double> v) {
            fillCxImpl<double>(n, a, v);
        }

        template <class R>
        static void projectImpl(int part, std::size_t n, const std::complex<R>* a, R* o) {
            if (!n) return;
            kProject<R><<<gridFor(n), kBlock>>>(part, n, cx(a), o);
            checkLaunch("complex projection");
        }
        void project(int part, std::size_t n, const std::complex<float>* a, float* o) {
            projectImpl<float>(part, n, a, o);
        }
        void project(int part, std::size_t n, const std::complex<double>* a, double* o) {
            projectImpl<double>(part, n, a, o);
        }

        template <class R>
        static void composeImpl(std::size_t n, const R* re, const R* im, std::complex<R>* o) {
            if (!n) return;
            kCompose<R><<<gridFor(n), kBlock>>>(n, re, im, cx(o));
            checkLaunch("complex compose");
        }
        void compose(std::size_t n, const float* re, const float* im, std::complex<float>* o) {
            composeImpl<float>(n, re, im, o);
        }
        void compose(std::size_t n, const double* re, const double* im, std::complex<double>* o) {
            composeImpl<double>(n, re, im, o);
        }

        template <class R>
        static std::complex<R> reduceCxImpl(RedOp op, std::size_t n, const std::complex<R>* a) {
            if (!n) throw Error("reduce: empty input has no defined result");
            if (op != RedOp::Sum && op != RedOp::Prod)
                throw Error("reduce: only sum and product are defined for complex values");
            int blocks = gridFor(n);
            if (blocks > 1024) blocks = 1024;
            Scratch part(sizeof(std::complex<R>) * (std::size_t)blocks);
            kReduceCx<R><<<blocks, kBlock>>>((int)op, 0, n, cx(a), (Cx<R>*)part.p);
            checkLaunch("complex reduce stage 1");
            if (blocks > 1) {
                kReduceCx<R><<<1, kBlock>>>((int)op, 1, (std::size_t)blocks, (const Cx<R>*)part.p,
                                            (Cx<R>*)part.p);
                checkLaunch("complex reduce stage 2");
            }
            std::complex<R> out;
            copyD2H(&out, part.p, sizeof(out));
            return out;
        }
        std::complex<float> reduce(RedOp op, std::size_t n, const std::complex<float>* a) {
            return reduceCxImpl<float>(op, n, a);
        }
        std::complex<double> reduce(RedOp op, std::size_t n, const std::complex<double>* a) {
            return reduceCxImpl<double>(op, n, a);
        }

        template <class R>
        static R normSqImpl(std::size_t n, const std::complex<R>* a) {
            if (!n) return R(0);
            int blocks = gridFor(n);
            if (blocks > 1024) blocks = 1024;
            Scratch part(sizeof(R) * (std::size_t)blocks);
            kNormSq<R><<<blocks, kBlock>>>(n, cx(a), part.template as<R>());
            checkLaunch("complex normSq stage 1");
            if (blocks > 1) {
                kReduce<R><<<1, kBlock>>>((int)RedOp::Sum, 1, (std::size_t)blocks,
                                          part.template as<R>(), part.template as<R>());
                checkLaunch("complex normSq stage 2");
            }
            R out{};
            copyD2H(&out, part.p, sizeof(R));
            return out;
        }
        float normSq(std::size_t n, const std::complex<float>* a) { return normSqImpl<float>(n, a); }
        double normSq(std::size_t n, const std::complex<double>* a) {
            return normSqImpl<double>(n, a);
        }

        // --- shape utilities: pure data movement, so the real kernels serve ---

        void triangle(int r, int c, std::complex<float>* A, bool u, bool ud) {
            triangleImpl<Cx<float>>(r, c, cx(A), u, ud);
        }
        void triangle(int r, int c, std::complex<double>* A, bool u, bool ud) {
            triangleImpl<Cx<double>>(r, c, cx(A), u, ud);
        }
        void copyBlock(int sr, int sc, const std::complex<float>* s, int r0, int c0, int nr, int nc,
                       std::complex<float>* d) {
            copyBlockImpl<Cx<float>>(sr, sc, cx(s), r0, c0, nr, nc, cx(d));
        }
        void copyBlock(int sr, int sc, const std::complex<double>* s, int r0, int c0, int nr,
                       int nc, std::complex<double>* d) {
            copyBlockImpl<Cx<double>>(sr, sc, cx(s), r0, c0, nr, nc, cx(d));
        }
        void setBlock(int dr, int dc, std::complex<float>* d, int r0, int c0, int nr, int nc,
                      const std::complex<float>* s) {
            setBlockImpl<Cx<float>>(dr, dc, cx(d), r0, c0, nr, nc, cx(s));
        }
        void setBlock(int dr, int dc, std::complex<double>* d, int r0, int c0, int nr, int nc,
                      const std::complex<double>* s) {
            setBlockImpl<Cx<double>>(dr, dc, cx(d), r0, c0, nr, nc, cx(s));
        }
        void setDiagonal(int r, int c, std::complex<float>* A, const std::complex<float>* d) {
            diagImpl<Cx<float>>(r, c, cx(A), const_cast<Cx<float>*>(cx(d)), true);
        }
        void setDiagonal(int r, int c, std::complex<double>* A, const std::complex<double>* d) {
            diagImpl<Cx<double>>(r, c, cx(A), const_cast<Cx<double>*>(cx(d)), true);
        }
        void getDiagonal(int r, int c, const std::complex<float>* A, std::complex<float>* d) {
            diagImpl<Cx<float>>(r, c, const_cast<Cx<float>*>(cx(A)), cx(d), false);
        }
        void getDiagonal(int r, int c, const std::complex<double>* A, std::complex<double>* d) {
            diagImpl<Cx<double>>(r, c, const_cast<Cx<double>*>(cx(A)), cx(d), false);
        }

        // --- BLAS ---

        void gemm(int M, int N, int K, std::complex<float> alpha, const std::complex<float>* A,
                  const std::complex<float>* B, std::complex<float> beta, std::complex<float>* C) {
            if (M <= 0 || N <= 0 || K <= 0) return;
            const cuComplex a = make_cuComplex(alpha.real(), alpha.imag());
            const cuComplex b = make_cuComplex(beta.real(), beta.imag());
            checkBlas(cublasCgemm(blas(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a,
                                  (const cuComplex*)B, N, (const cuComplex*)A, K, &b,
                                  (cuComplex*)C, N),
                      "cublasCgemm");
        }
        void gemm(int M, int N, int K, std::complex<double> alpha, const std::complex<double>* A,
                  const std::complex<double>* B, std::complex<double> beta,
                  std::complex<double>* C) {
            if (M <= 0 || N <= 0 || K <= 0) return;
            const cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
            const cuDoubleComplex b = make_cuDoubleComplex(beta.real(), beta.imag());
            checkBlas(cublasZgemm(blas(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a,
                                  (const cuDoubleComplex*)B, N, (const cuDoubleComplex*)A, K, &b,
                                  (cuDoubleComplex*)C, N),
                      "cublasZgemm");
        }

        // conjugate = true gives the CONJUGATE transpose A^H, which for complex
        // matrices is almost always the one wanted: it is what makes Q^H Q = I
        // and A = U S V^H hold.
        void transposeCx(int rows, int cols, const std::complex<float>* A, std::complex<float>* out,
                         bool conjugate) {
            if (rows <= 0 || cols <= 0) return;
            const cuComplex one = make_cuComplex(1, 0), zero = make_cuComplex(0, 0);
            checkBlas(cublasCgeam(blas(), conjugate ? CUBLAS_OP_C : CUBLAS_OP_T, CUBLAS_OP_N, rows,
                                  cols, &one, (const cuComplex*)A, cols, &zero, (cuComplex*)out,
                                  rows, (cuComplex*)out, rows),
                      "cublasCgeam(transpose)");
        }
        void transposeCx(int rows, int cols, const std::complex<double>* A,
                         std::complex<double>* out, bool conjugate) {
            if (rows <= 0 || cols <= 0) return;
            const cuDoubleComplex one = make_cuDoubleComplex(1, 0),
                                  zero = make_cuDoubleComplex(0, 0);
            checkBlas(cublasZgeam(blas(), conjugate ? CUBLAS_OP_C : CUBLAS_OP_T, CUBLAS_OP_N, rows,
                                  cols, &one, (const cuDoubleComplex*)A, cols, &zero,
                                  (cuDoubleComplex*)out, rows, (cuDoubleComplex*)out, rows),
                      "cublasZgeam(transpose)");
        }

        // Fused expressions for complex, so .lazy() is not a real-only
        // feature. Same postfix program, same register-only fast path for the
        // shallow left-linear case that covers nearly every real expression.
        template <class R>
        __global__ void kFusedCx(FusedProgram prog, std::size_t n, FusedInputs<Cx<R>> in,
                                 Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                Cx<R> a{}, b{};
                int sp = 0;
                for (int k = 0; k < prog.nOps; ++k) {
                    switch (prog.code[k]) {
                        case 0: {
                            const Cx<R> v = in.p[prog.arg[k]][i];
                            if (sp == 0) a = v; else b = v;
                            ++sp;
                            break;
                        }
                        case 1: {
                            const Cx<R> v = Cx<R>(R(prog.imm[k]));
                            if (sp == 0) a = v; else b = v;
                            ++sp;
                            break;
                        }
                        case 2:
                            if (sp == 1) a = applyUnCx<R>(prog.arg[k], a);
                            else b = applyUnCx<R>(prog.arg[k], b);
                            break;
                        default:
                            a = applyBinCx<R>(prog.arg[k], a, b);
                            sp = 1;
                            break;
                    }
                }
                out[i] = a;
            }
        }

        // The deep case needs a real stack; complex expressions deep enough to
        // need it are rare, so it gets the straightforward version.
        template <class R>
        __global__ void kFusedCxDeep(FusedProgram prog, std::size_t n, FusedInputs<Cx<R>> in,
                                     Cx<R>* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                Cx<R> stack[FusedProgram::StackDepth];
                int sp = 0;
                for (int k = 0; k < prog.nOps; ++k) {
                    switch (prog.code[k]) {
                        case 0: stack[sp++] = in.p[prog.arg[k]][i]; break;
                        case 1: stack[sp++] = Cx<R>(R(prog.imm[k])); break;
                        case 2: stack[sp - 1] = applyUnCx<R>(prog.arg[k], stack[sp - 1]); break;
                        default: {
                            const Cx<R> rhs = stack[--sp];
                            stack[sp - 1] = applyBinCx<R>(prog.arg[k], stack[sp - 1], rhs);
                            break;
                        }
                    }
                }
                out[i] = stack[0];
            }
        }

        template <class R>
        static void fusedCxImpl(const FusedProgram& prog, std::size_t n,
                                const std::complex<R>* const* inputs, int nInputs,
                                std::complex<R>* out) {
            if (!n) return;
            if (prog.nOps <= 0 || prog.nOps > FusedProgram::MaxOps)
                throw Error("fusedElementwise: program too long");
            if (nInputs < 0 || nInputs > FusedProgram::MaxInputs)
                throw Error("fusedElementwise: too many inputs");
            FusedInputs<Cx<R>> in{};
            for (int i = 0; i < nInputs; ++i) in.p[i] = cx(inputs[i]);
            if (prog.maxDepth <= 2)
                kFusedCx<R><<<gridFor(n), kBlock>>>(prog, n, in, cx(out));
            else
                kFusedCxDeep<R><<<gridFor(n), kBlock>>>(prog, n, in, cx(out));
            checkLaunch("complex fused element-wise kernel");
        }
        void fusedElementwise(const FusedProgram& prog, std::size_t n,
                              const std::complex<float>* const* inputs, int nInputs,
                              std::complex<float>* out) {
            fusedCxImpl<float>(prog, n, inputs, nInputs, out);
        }
        void fusedElementwise(const FusedProgram& prog, std::size_t n,
                              const std::complex<double>* const* inputs, int nInputs,
                              std::complex<double>* out) {
            fusedCxImpl<double>(prog, n, inputs, nInputs, out);
        }

        // --- complex factorisations ---
        //
        // Same shapes as the real ones, with two differences worth naming:
        // complex gesvd needs a real `rwork` array cuSOLVER uses internally,
        // and the symmetric eigenproblem becomes the HERMITIAN one (heevd),
        // whose eigenvalues are real even though the matrix is not.

        int getrf(int m, int n, std::complex<double>* A, int* ipiv) {
            int lwork = 0;
            auto* a = (cuDoubleComplex*)A;
            checkSolver(cusolverDnZgetrf_bufferSize(solver(), m, n, a, m, &lwork),
                        "cusolverDnZgetrf_bufferSize");
            Scratch w(sizeof(cuDoubleComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnZgetrf(solver(), m, n, a, m, (cuDoubleComplex*)w.p, ipiv, info.d),
                        "cusolverDnZgetrf");
            return info.get();
        }
        int getrf(int m, int n, std::complex<float>* A, int* ipiv) {
            int lwork = 0;
            auto* a = (cuComplex*)A;
            checkSolver(cusolverDnCgetrf_bufferSize(solver(), m, n, a, m, &lwork),
                        "cusolverDnCgetrf_bufferSize");
            Scratch w(sizeof(cuComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnCgetrf(solver(), m, n, a, m, (cuComplex*)w.p, ipiv, info.d),
                        "cusolverDnCgetrf");
            return info.get();
        }
        int getrs(int n, int nrhs, const std::complex<double>* A, const int* ipiv,
                  std::complex<double>* B) {
            Info info;
            checkSolver(cusolverDnZgetrs(solver(), CUBLAS_OP_N, n, nrhs, (const cuDoubleComplex*)A,
                                         n, ipiv, (cuDoubleComplex*)B, n, info.d),
                        "cusolverDnZgetrs");
            return info.get();
        }
        int getrs(int n, int nrhs, const std::complex<float>* A, const int* ipiv,
                  std::complex<float>* B) {
            Info info;
            checkSolver(cusolverDnCgetrs(solver(), CUBLAS_OP_N, n, nrhs, (const cuComplex*)A, n,
                                         ipiv, (cuComplex*)B, n, info.d),
                        "cusolverDnCgetrs");
            return info.get();
        }

        int potrf(int n, std::complex<double>* A, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            int lwork = 0;
            auto* a = (cuDoubleComplex*)A;
            checkSolver(cusolverDnZpotrf_bufferSize(solver(), uplo, n, a, n, &lwork),
                        "cusolverDnZpotrf_bufferSize");
            Scratch w(sizeof(cuDoubleComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnZpotrf(solver(), uplo, n, a, n, (cuDoubleComplex*)w.p, lwork,
                                         info.d),
                        "cusolverDnZpotrf");
            return info.get();
        }
        int potrf(int n, std::complex<float>* A, bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            int lwork = 0;
            auto* a = (cuComplex*)A;
            checkSolver(cusolverDnCpotrf_bufferSize(solver(), uplo, n, a, n, &lwork),
                        "cusolverDnCpotrf_bufferSize");
            Scratch w(sizeof(cuComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnCpotrf(solver(), uplo, n, a, n, (cuComplex*)w.p, lwork, info.d),
                        "cusolverDnCpotrf");
            return info.get();
        }
        int potrs(int n, int nrhs, const std::complex<double>* A, std::complex<double>* B,
                  bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            Info info;
            checkSolver(cusolverDnZpotrs(solver(), uplo, n, nrhs, (const cuDoubleComplex*)A, n,
                                         (cuDoubleComplex*)B, n, info.d),
                        "cusolverDnZpotrs");
            return info.get();
        }
        int potrs(int n, int nrhs, const std::complex<float>* A, std::complex<float>* B,
                  bool upper) {
            cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
            Info info;
            checkSolver(cusolverDnCpotrs(solver(), uplo, n, nrhs, (const cuComplex*)A, n,
                                         (cuComplex*)B, n, info.d),
                        "cusolverDnCpotrs");
            return info.get();
        }

        int geqrf(int m, int n, std::complex<double>* A, std::complex<double>* tau) {
            int lwork = 0;
            auto* a = (cuDoubleComplex*)A;
            checkSolver(cusolverDnZgeqrf_bufferSize(solver(), m, n, a, m, &lwork),
                        "cusolverDnZgeqrf_bufferSize");
            Scratch w(sizeof(cuDoubleComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnZgeqrf(solver(), m, n, a, m, (cuDoubleComplex*)tau,
                                         (cuDoubleComplex*)w.p, lwork, info.d),
                        "cusolverDnZgeqrf");
            return info.get();
        }
        int geqrf(int m, int n, std::complex<float>* A, std::complex<float>* tau) {
            int lwork = 0;
            auto* a = (cuComplex*)A;
            checkSolver(cusolverDnCgeqrf_bufferSize(solver(), m, n, a, m, &lwork),
                        "cusolverDnCgeqrf_bufferSize");
            Scratch w(sizeof(cuComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnCgeqrf(solver(), m, n, a, m, (cuComplex*)tau, (cuComplex*)w.p,
                                         lwork, info.d),
                        "cusolverDnCgeqrf");
            return info.get();
        }
        // ungqr is unmqr's generator: the complex name for orgqr, and it builds
        // a UNITARY factor rather than an orthogonal one.
        int orgqr(int m, int n, int k, std::complex<double>* A, const std::complex<double>* tau) {
            int lwork = 0;
            auto* a = (cuDoubleComplex*)A;
            checkSolver(cusolverDnZungqr_bufferSize(solver(), m, n, k, a, m,
                                                    (const cuDoubleComplex*)tau, &lwork),
                        "cusolverDnZungqr_bufferSize");
            Scratch w(sizeof(cuDoubleComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnZungqr(solver(), m, n, k, a, m, (const cuDoubleComplex*)tau,
                                         (cuDoubleComplex*)w.p, lwork, info.d),
                        "cusolverDnZungqr");
            return info.get();
        }
        int orgqr(int m, int n, int k, std::complex<float>* A, const std::complex<float>* tau) {
            int lwork = 0;
            auto* a = (cuComplex*)A;
            checkSolver(cusolverDnCungqr_bufferSize(solver(), m, n, k, a, m,
                                                    (const cuComplex*)tau, &lwork),
                        "cusolverDnCungqr_bufferSize");
            Scratch w(sizeof(cuComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnCungqr(solver(), m, n, k, a, m, (const cuComplex*)tau,
                                         (cuComplex*)w.p, lwork, info.d),
                        "cusolverDnCungqr");
            return info.get();
        }

        // Complex SVD. Singular values are REAL whatever went in -- they are
        // magnitudes -- which is why S is a real array here and Matrix<double>
        // on the CPU side.
        template <class C, class R, class FBuf, class FSvd, class FGeam>
        static int gesvdCxImpl(FBuf fbuf, FSvd fsvd, FGeam fgeam, const char* name, int m, int n,
                               C* A, R* S, C* U, C* VT, bool full) {
            const int k = m < n ? m : n;
            const signed char job = full ? 'A' : 'S';
            const signed char jobu = U ? job : 'N';
            const signed char jobvt = VT ? job : 'N';
            // cuSOLVER needs a real scratch array for the complex path.
            Scratch rwork(sizeof(R) * (std::size_t)(k > 1 ? 5 * k : 5));

            if (m >= n) {
                int lwork = 0;
                checkSolver(fbuf(solver(), m, n, &lwork), "gesvd_bufferSize");
                Scratch w(sizeof(C) * (std::size_t)lwork);
                Info info;
                checkSolver(fsvd(solver(), jobu, jobvt, m, n, A, m, S, U, m, VT, full ? n : k,
                                 (C*)w.p, lwork, (R*)rwork.p, info.d),
                            name);
                return info.get();
            }

            // Wide case. For complex the identity is A = (A^H)^H, and
            // A^H = V S U^H, so factoring the CONJUGATE transpose and swapping
            // the factors gives A = U S V^H back. Using the plain transpose
            // here would conjugate the answer -- the same trap basic/ documents.
            const int utCols = full ? n : k;
            Scratch At(sizeof(C) * (std::size_t)m * n);
            const C one{1, 0}, zero{0, 0};
            checkBlas(fgeam(blas(), CUBLAS_OP_C, CUBLAS_OP_N, n, m, &one, A, m, &zero, (C*)At.p, n,
                            (C*)At.p, n),
                      "geam(conjugate transpose)");
            Scratch Ut(U || VT ? sizeof(C) * (std::size_t)n * utCols : 0);
            Scratch VTt(U ? sizeof(C) * (std::size_t)k * m : 0);
            int lwork = 0;
            checkSolver(fbuf(solver(), n, m, &lwork), "gesvd_bufferSize");
            Scratch w(sizeof(C) * (std::size_t)lwork);
            Info info;
            checkSolver(fsvd(solver(), VT ? job : 'N', U ? job : 'N', n, m, (C*)At.p, n, S,
                             (C*)Ut.p, n, (C*)VTt.p, k, (C*)w.p, lwork, (R*)rwork.p, info.d),
                        name);
            const int st = info.get();
            if (st != 0) return st;
            if (U) checkBlas(fgeam(blas(), CUBLAS_OP_C, CUBLAS_OP_N, m, k, &one, (const C*)VTt.p, k,
                                   &zero, U, m, U, m),
                             "geam(U)");
            if (VT) checkBlas(fgeam(blas(), CUBLAS_OP_C, CUBLAS_OP_N, utCols, n, &one,
                                    (const C*)Ut.p, n, &zero, VT, utCols, VT, utCols),
                              "geam(VT)");
            return 0;
        }

        int gesvd(int m, int n, std::complex<double>* A, double* S, std::complex<double>* U,
                  std::complex<double>* VT, bool full) {
            return gesvdCxImpl<cuDoubleComplex, double>(
                cusolverDnZgesvd_bufferSize, cusolverDnZgesvd, cublasZgeam, "cusolverDnZgesvd", m,
                n, (cuDoubleComplex*)A, S, (cuDoubleComplex*)U, (cuDoubleComplex*)VT, full);
        }
        int gesvd(int m, int n, std::complex<float>* A, float* S, std::complex<float>* U,
                  std::complex<float>* VT, bool full) {
            return gesvdCxImpl<cuComplex, float>(cusolverDnCgesvd_bufferSize, cusolverDnCgesvd,
                                                 cublasCgeam, "cusolverDnCgesvd", m, n,
                                                 (cuComplex*)A, S, (cuComplex*)U, (cuComplex*)VT,
                                                 full);
        }

        // Hermitian eigenproblem. A Hermitian matrix has REAL eigenvalues, so w
        // is a real array even though the eigenvectors are complex.
        int heevd(int n, std::complex<double>* A, double* w, bool vectors) {
            cusolverEigMode_t jobz =
                vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
            auto* a = (cuDoubleComplex*)A;
            int lwork = 0;
            checkSolver(cusolverDnZheevd_bufferSize(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, a, n,
                                                    w, &lwork),
                        "cusolverDnZheevd_bufferSize");
            Scratch ws(sizeof(cuDoubleComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnZheevd(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, a, n, w,
                                         (cuDoubleComplex*)ws.p, lwork, info.d),
                        "cusolverDnZheevd");
            return info.get();
        }
        int heevd(int n, std::complex<float>* A, float* w, bool vectors) {
            cusolverEigMode_t jobz =
                vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
            auto* a = (cuComplex*)A;
            int lwork = 0;
            checkSolver(cusolverDnCheevd_bufferSize(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, a, n,
                                                    w, &lwork),
                        "cusolverDnCheevd_bufferSize");
            Scratch ws(sizeof(cuComplex) * (std::size_t)lwork);
            Info info;
            checkSolver(cusolverDnCheevd(solver(), jobz, CUBLAS_FILL_MODE_LOWER, n, a, n, w,
                                         (cuComplex*)ws.p, lwork, info.d),
                        "cusolverDnCheevd");
            return info.get();
        }

        // Complex FFT, straight through. cuFFT works on INTERLEAVED complex,
        // which is exactly how std::complex<T> is laid out, so a complex matrix
        // needs neither the interleave on the way in nor the split on the way
        // out that the real path pays. Two fewer passes over the data.
        template <class R>
        static void fftCxRun(const PlanKey& key, std::size_t total, int n,
                             const std::complex<R>* in, std::complex<R>* out, bool inverse) {
            if (!total) return;
            if (in != out) copyD2D(out, in, sizeof(std::complex<R>) * total);
            checkFft(FftTraits<R>::exec(planFor(key), out, inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                     "cufftExec");
            if (inverse)
                binaryScalarCxImpl<R>(BinOp::Mul, total, out, std::complex<R>(R(1) / R(n), 0), out,
                                      false);
        }

        template <class R>
        static void fft1dCxImpl(int batch, int n, int stride, int dist, const std::complex<R>* in,
                                std::complex<R>* out, bool inverse) {
            if (batch <= 0 || n <= 0) return;
            PlanKey k{1, n, 0, batch, stride, dist, (int)FftTraits<R>::type};
            fftCxRun<R>(k, (std::size_t)batch * n, n, in, out, inverse);
        }
        void fft1dCx(int b, int n, int st, int di, const std::complex<float>* in,
                     std::complex<float>* out, bool inv) {
            fft1dCxImpl<float>(b, n, st, di, in, out, inv);
        }
        void fft1dCx(int b, int n, int st, int di, const std::complex<double>* in,
                     std::complex<double>* out, bool inv) {
            fft1dCxImpl<double>(b, n, st, di, in, out, inv);
        }

        template <class R>
        static void fft2dCxImpl(int rows, int cols, const std::complex<R>* in,
                                std::complex<R>* out, bool inverse) {
            if (rows <= 0 || cols <= 0) return;
            PlanKey k{2, rows, cols, 1, 1, 1, (int)FftTraits<R>::type};
            fftCxRun<R>(k, (std::size_t)rows * cols, rows * cols, in, out, inverse);
        }
        void fft2dCx(int r, int c, const std::complex<float>* in, std::complex<float>* out,
                     bool inv) {
            fft2dCxImpl<float>(r, c, in, out, inv);
        }
        void fft2dCx(int r, int c, const std::complex<double>* in, std::complex<double>* out,
                     bool inv) {
            fft2dCxImpl<double>(r, c, in, out, inv);
        }

        // ── Comparisons and masks ──────────────────────────────────────

        template <class T>
        __device__ __forceinline__ T cmpResult(int op, T a, T b) {
            switch ((CmpOp)op) {
                case CmpOp::LT: return T(a < b);
                case CmpOp::LE: return T(a <= b);
                case CmpOp::GT: return T(a > b);
                case CmpOp::GE: return T(a >= b);
                case CmpOp::EQ: return T(a == b);
                default: return T(a != b);
            }
        }

        template <class T>
        __global__ void kCompare(int op, std::size_t n, const T* a, const T* b, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = cmpResult<T>(op, a[i], b[i]);
        }
        template <class T>
        __global__ void kCompareScalar(int op, std::size_t n, const T* a, T s, T* out) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x)
                out[i] = cmpResult<T>(op, a[i], s);
        }

        template <class T>
        static void compareImpl(CmpOp op, std::size_t n, const T* a, const T* b, T* out) {
            if (!n) return;
            kCompare<T><<<gridFor(n), kBlock>>>((int)op, n, a, b, out);
            checkLaunch("compare kernel");
        }
        void compare(CmpOp op, std::size_t n, const float* a, const float* b, float* o) {
            compareImpl<float>(op, n, a, b, o);
        }
        void compare(CmpOp op, std::size_t n, const double* a, const double* b, double* o) {
            compareImpl<double>(op, n, a, b, o);
        }
        template <class T>
        static void compareScalarImpl(CmpOp op, std::size_t n, const T* a, T s, T* out) {
            if (!n) return;
            kCompareScalar<T><<<gridFor(n), kBlock>>>((int)op, n, a, s, out);
            checkLaunch("compare-scalar kernel");
        }
        void compareScalar(CmpOp op, std::size_t n, const float* a, float s, float* o) {
            compareScalarImpl<float>(op, n, a, s, o);
        }
        void compareScalar(CmpOp op, std::size_t n, const double* a, double s, double* o) {
            compareScalarImpl<double>(op, n, a, s, o);
        }

        // Index-carrying reduction. The value alone is not enough -- two
        // elements can tie -- so the index rides along and the comparison
        // breaks ties toward the SMALLER index, which is what makes the answer
        // deterministic and matches basic/ and NumPy.
        template <class T>
        __global__ void kArgExtreme(bool maximum, int stage, std::size_t n, const T* a,
                                    const long* inIdx, T* outVal, long* outIdx) {
            T best = maximum ? -T(INFINITY) : T(INFINITY);
            long bestIdx = -1;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const T v = a[i];
                const long idx = stage == 0 ? (long)i : inIdx[i];
                const bool better = maximum ? (v > best) : (v < best);
                if (better || (v == best && idx < bestIdx)) {
                    best = v;
                    bestIdx = idx;
                }
            }
            __shared__ T sv[kBlock];
            __shared__ long si[kBlock];
            const int t = threadIdx.x;
            sv[t] = best;
            si[t] = bestIdx;
            __syncthreads();
            for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
                if (t < s2) {
                    const bool better = maximum ? (sv[t + s2] > sv[t]) : (sv[t + s2] < sv[t]);
                    const bool tie = (sv[t + s2] == sv[t]) && si[t + s2] >= 0 &&
                                     (si[t] < 0 || si[t + s2] < si[t]);
                    if (better || tie) {
                        sv[t] = sv[t + s2];
                        si[t] = si[t + s2];
                    }
                }
                __syncthreads();
            }
            if (t == 0) {
                outVal[blockIdx.x] = sv[0];
                outIdx[blockIdx.x] = si[0];
            }
        }

        template <class T>
        static long argExtremeImpl(bool maximum, std::size_t n, const T* a) {
            if (!n) throw Error("argmax/argmin: empty input has no answer");
            int blocks = gridFor(n);
            if (blocks > 1024) blocks = 1024;
            Scratch v(sizeof(T) * (std::size_t)blocks);
            Scratch idx(sizeof(long) * (std::size_t)blocks);
            kArgExtreme<T><<<blocks, kBlock>>>(maximum, 0, n, a, nullptr, v.template as<T>(),
                                               (long*)idx.p);
            checkLaunch("argExtreme stage 1");
            if (blocks > 1) {
                kArgExtreme<T><<<1, kBlock>>>(maximum, 1, (std::size_t)blocks, v.template as<T>(),
                                              (const long*)idx.p, v.template as<T>(), (long*)idx.p);
                checkLaunch("argExtreme stage 2");
            }
            long out = 0;
            copyD2H(&out, idx.p, sizeof(long));
            return out;
        }
        long argExtreme(bool mx, std::size_t n, const float* a) {
            return argExtremeImpl<float>(mx, n, a);
        }
        long argExtreme(bool mx, std::size_t n, const double* a) {
            return argExtremeImpl<double>(mx, n, a);
        }

        // ── Rearrangement ──────────────────────────────────────────────

        template <class T>
        __global__ void kRearrange(int how, int sr, int sc, const T* src, int p, int q, int dr,
                                   int dc, T* dst) {
            const std::size_t n = (std::size_t)dr * dc;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / dc), c = (int)(i % dc);
                int sr_i = r, sc_i = c;
                switch ((Rearrange)how) {
                    case Rearrange::Repmat:
                        sr_i = r % sr;
                        sc_i = c % sc;
                        break;
                    case Rearrange::FlipLR: sc_i = sc - 1 - c; break;
                    case Rearrange::FlipUD: sr_i = sr - 1 - r; break;
                    // 90 degrees anticlockwise: destination (r, c) comes from
                    // source (c, sc-1-r), and the destination is sc x sr.
                    case Rearrange::Rot90:
                        sr_i = c;
                        sc_i = sc - 1 - r;
                        break;
                    default: {
                        // Shift by (p, q) with wraparound. The +sr / +sc before
                        // the modulo keeps a negative shift in range, since C's
                        // % follows the sign of the dividend.
                        sr_i = ((r - p) % sr + sr) % sr;
                        sc_i = ((c - q) % sc + sc) % sc;
                        break;
                    }
                }
                dst[i] = src[(std::size_t)sr_i * sc + sc_i];
            }
        }

        template <class T>
        static void rearrangeImpl(Rearrange how, int sr, int sc, const T* src, int p, int q,
                                  T* dst) {
            if (sr <= 0 || sc <= 0) return;
            int dr = sr, dc = sc;
            if (how == Rearrange::Repmat) { dr = sr * p; dc = sc * q; }
            else if (how == Rearrange::Rot90) { dr = sc; dc = sr; }
            kRearrange<T><<<gridFor((std::size_t)dr * dc), kBlock>>>((int)how, sr, sc, src, p, q,
                                                                     dr, dc, dst);
            checkLaunch("rearrange kernel");
        }
        void rearrange(Rearrange h, int sr, int sc, const float* s, int p, int q, float* d) {
            rearrangeImpl<float>(h, sr, sc, s, p, q, d);
        }
        void rearrange(Rearrange h, int sr, int sc, const double* s, int p, int q, double* d) {
            rearrangeImpl<double>(h, sr, sc, s, p, q, d);
        }
        void rearrange(Rearrange h, int sr, int sc, const std::complex<float>* s, int p, int q,
                       std::complex<float>* d) {
            rearrangeImpl<Cx<float>>(h, sr, sc, cx(s), p, q, cx(d));
        }
        void rearrange(Rearrange h, int sr, int sc, const std::complex<double>* s, int p, int q,
                       std::complex<double>* d) {
            rearrangeImpl<Cx<double>>(h, sr, sc, cx(s), p, q, cx(d));
        }

        template <class T>
        __global__ void kKron(int ar, int ac, const T* A, int br, int bc, const T* B, T* dst) {
            const int dc = ac * bc;
            const std::size_t n = (std::size_t)ar * br * dc;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / dc), c = (int)(i % dc);
                dst[i] = A[(std::size_t)(r / br) * ac + (c / bc)] *
                         B[(std::size_t)(r % br) * bc + (c % bc)];
            }
        }
        template <class T>
        static void kronImpl(int ar, int ac, const T* A, int br, int bc, const T* B, T* dst) {
            if (ar <= 0 || ac <= 0 || br <= 0 || bc <= 0) return;
            kKron<T><<<gridFor((std::size_t)ar * br * ac * bc), kBlock>>>(ar, ac, A, br, bc, B,
                                                                          dst);
            checkLaunch("kron kernel");
        }
        void kron(int ar, int ac, const float* A, int br, int bc, const float* B, float* d) {
            kronImpl<float>(ar, ac, A, br, bc, B, d);
        }
        void kron(int ar, int ac, const double* A, int br, int bc, const double* B, double* d) {
            kronImpl<double>(ar, ac, A, br, bc, B, d);
        }
        void kron(int ar, int ac, const std::complex<float>* A, int br, int bc,
                  const std::complex<float>* B, std::complex<float>* d) {
            kronImpl<Cx<float>>(ar, ac, cx(A), br, bc, cx(B), cx(d));
        }
        void kron(int ar, int ac, const std::complex<double>* A, int br, int bc,
                  const std::complex<double>* B, std::complex<double>* d) {
            kronImpl<Cx<double>>(ar, ac, cx(A), br, bc, cx(B), cx(d));
        }

        // ── Sorting ────────────────────────────────────────────────────

        template <class T>
        static void sortFlatImpl(std::size_t n, T* data, bool desc) {
            if (n < 2) return;
            thrust::device_ptr<T> p(data);
            if (desc) thrust::sort(thrust::device, p, p + n, thrust::greater<T>());
            else thrust::sort(thrust::device, p, p + n);
            checkLaunch("sort");
        }
        void sortFlat(std::size_t n, float* d, bool r) { sortFlatImpl<float>(n, d, r); }
        void sortFlat(std::size_t n, double* d, bool r) { sortFlatImpl<double>(n, d, r); }

        // Segment boundaries for a row-major matrix: row i spans
        // [i*cols, (i+1)*cols). CUB wants the begin and end arrays separately,
        // but they overlap by one, so one array of rows+1 offsets serves both.
        __global__ void kRowOffsets(int rows, int cols, int* off) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <= rows;
                 i += blockDim.x * gridDim.x)
                off[i] = i * cols;
        }

        template <class T>
        static void sortSegments(int segments, int width, T* data, bool desc) {
            const std::size_t n = (std::size_t)segments * width;
            Scratch off(sizeof(int) * (std::size_t)(segments + 1));
            kRowOffsets<<<gridFor((std::size_t)segments + 1), kBlock>>>(segments, width,
                                                                       (int*)off.p);
            checkLaunch("segment offsets");
            // CUB's segmented sort is not in-place: it needs a distinct output.
            Scratch out(sizeof(T) * n);
            std::size_t bytes = 0;
            const int* beg = (const int*)off.p;
            const int* end = beg + 1;
            if (desc)
                cub::DeviceSegmentedSort::SortKeysDescending(nullptr, bytes, data, out.as<T>(),
                                                             (int)n, segments, beg, end);
            else
                cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, data, out.as<T>(), (int)n,
                                                   segments, beg, end);
            Scratch tmp(bytes);
            if (desc)
                cub::DeviceSegmentedSort::SortKeysDescending(tmp.p, bytes, data, out.as<T>(),
                                                             (int)n, segments, beg, end);
            else
                cub::DeviceSegmentedSort::SortKeys(tmp.p, bytes, data, out.as<T>(), (int)n,
                                                   segments, beg, end);
            checkLaunch("segmented sort");
            copyD2D(data, out.p, sizeof(T) * n);
        }

        template <class T, class FGeam>
        static void sortAxisImpl(FGeam fgeam, int rows, int cols, T* data, bool byRow, bool desc) {
            if (rows <= 0 || cols <= 0) return;
            if (byRow) {
                sortSegments<T>(rows, cols, data, desc);
                return;
            }
            // A column is strided, and a CUB segment must be contiguous, so the
            // matrix is transposed, sorted by row, and transposed back. Two
            // geam passes is far cheaper than one sort launch per column.
            const std::size_t n = (std::size_t)rows * cols;
            Scratch t(sizeof(T) * n);
            transposeCM<T>(fgeam, cols, rows, data, t.as<T>());
            sortSegments<T>(cols, rows, t.as<T>(), desc);
            transposeCM<T>(fgeam, rows, cols, t.as<T>(), data);
        }
        void sortAxis(int r, int c, float* d, bool byRow, bool desc) {
            sortAxisImpl<float>(cublasSgeam, r, c, d, byRow, desc);
        }
        void sortAxis(int r, int c, double* d, bool byRow, bool desc) {
            sortAxisImpl<double>(cublasDgeam, r, c, d, byRow, desc);
        }

        // Pulls one column out as the sort key.
        template <class T>
        __global__ void kExtractCol(int rows, int cols, const T* A, int key, T* out) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
                 i += blockDim.x * gridDim.x)
                out[i] = A[(std::size_t)i * cols + key];
        }
        // Moves whole rows into the order the permutation gives.
        template <class T>
        __global__ void kPermuteRows(int rows, int cols, const T* A, const int* order, T* out) {
            const std::size_t n = (std::size_t)rows * cols;
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const int r = (int)(i / cols), c = (int)(i % cols);
                out[i] = A[(std::size_t)order[r] * cols + c];
            }
        }

        template <class T>
        static void sortRowsByImpl(int rows, int cols, T* data, int key, bool desc) {
            if (rows <= 1 || cols <= 0) return;
            if (key < 0 || key >= cols)
                throw Error("sortrows: key column " + std::to_string(key) +
                            " is outside a matrix with " + std::to_string(cols) + " columns");
            Scratch keys(sizeof(T) * (std::size_t)rows);
            Scratch order(sizeof(int) * (std::size_t)rows);
            kExtractCol<T><<<gridFor((std::size_t)rows), kBlock>>>(rows, cols, data, key,
                                                                   keys.as<T>());
            checkLaunch("extract key column");
            thrust::device_ptr<int> o((int*)order.p);
            thrust::sequence(thrust::device, o, o + rows);
            thrust::device_ptr<T> k(keys.as<T>());
            // STABLE, so rows with equal keys keep their original order --
            // which is what makes a sequence of sortrows calls compose into a
            // multi-column sort, and what MATLAB does.
            if (desc) thrust::stable_sort_by_key(thrust::device, k, k + rows, o,
                                                 thrust::greater<T>());
            else thrust::stable_sort_by_key(thrust::device, k, k + rows, o);
            Scratch out(sizeof(T) * (std::size_t)rows * cols);
            kPermuteRows<T><<<gridFor((std::size_t)rows * cols), kBlock>>>(
                rows, cols, data, (const int*)order.p, out.as<T>());
            checkLaunch("permute rows");
            copyD2D(data, out.p, sizeof(T) * (std::size_t)rows * cols);
        }
        void sortRowsBy(int r, int c, float* d, int k, bool desc) {
            sortRowsByImpl<float>(r, c, d, k, desc);
        }
        void sortRowsBy(int r, int c, double* d, int k, bool desc) {
            sortRowsByImpl<double>(r, c, d, k, desc);
        }

        template <class T>
        static long uniqueImpl(std::size_t n, T* data) {
            if (n == 0) return 0;
            thrust::device_ptr<T> p(data);
            thrust::sort(thrust::device, p, p + n);
            auto last = thrust::unique(thrust::device, p, p + n);
            checkLaunch("unique");
            return (long)(last - p);
        }
        long uniqueInPlace(std::size_t n, float* d) { return uniqueImpl<float>(n, d); }
        long uniqueInPlace(std::size_t n, double* d) { return uniqueImpl<double>(n, d); }

        // Longest run in the sorted copy. Ties go to the SMALLEST value because
        // the scan keeps the first run of a given length and the data is
        // ascending -- the same rule MATLAB's mode follows.
        template <class T>
        static T modeImpl(std::size_t n, const T* data) {
            if (n == 0) throw Error("mode: empty input has no answer");
            Scratch buf(sizeof(T) * n);
            copyD2D(buf.p, data, sizeof(T) * n);
            thrust::device_ptr<T> p(buf.as<T>());
            thrust::sort(thrust::device, p, p + n);
            // The run structure is cheap to read on the host once sorted, and
            // n is the count of DISTINCT runs at most -- but copying n values
            // back would defeat the point, so the scan runs on the device via
            // reduce_by_key into counts, then one max.
            Scratch vals(sizeof(T) * n);
            Scratch cnts(sizeof(int) * n);
            thrust::device_ptr<T> vp(vals.as<T>());
            thrust::device_ptr<int> cp((int*)cnts.p);
            auto ends = thrust::reduce_by_key(thrust::device, p, p + n,
                                              thrust::constant_iterator<int>(1), vp, cp);
            const long runs = (long)(ends.first - vp);
            checkLaunch("mode: run lengths");
            // argmax over the run counts, ties to the lowest index = smallest
            // value, since the runs are in ascending order.
            std::vector<int> hc((std::size_t)runs);
            copyD2H(hc.data(), cnts.p, sizeof(int) * (std::size_t)runs);
            long best = 0;
            for (long i = 1; i < runs; ++i)
                if (hc[(std::size_t)i] > hc[(std::size_t)best]) best = i;
            T out{};
            copyD2H(&out, vals.as<T>() + best, sizeof(T));
            return out;
        }
        float modeOf(std::size_t n, const float* d) { return modeImpl<float>(n, d); }
        double modeOf(std::size_t n, const double* d) { return modeImpl<double>(n, d); }

        // ── Builders ───────────────────────────────────────────────────

        template <class T>
        __global__ void kLinspace(std::size_t n, T* out, T lo, T step, bool logarithmic) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)blockDim.x * gridDim.x) {
                const T v = lo + T(i) * step;
                out[i] = logarithmic ? pow(T(10), v) : v;
            }
        }
        template <class T>
        static void linspaceImpl(std::size_t n, T* out, T lo, T hi, bool lg) {
            if (!n) return;
            // Both endpoints included, so n-1 gaps. For n == 1 MATLAB -- and
            // basic/ after it -- returns the UPPER bound, not the lower; NumPy
            // returns the lower. Following basic/ is the point of the exercise.
            const T step = (n > 1) ? (hi - lo) / T(n - 1) : T(0);
            kLinspace<T><<<gridFor(n), kBlock>>>(n, out, n == 1 ? hi : lo, step, lg);
            checkLaunch("linspace kernel");
        }
        void linspace(std::size_t n, float* o, float lo, float hi, bool lg) {
            linspaceImpl<float>(n, o, lo, hi, lg);
        }
        void linspace(std::size_t n, double* o, double lo, double hi, bool lg) {
            linspaceImpl<double>(n, o, lo, hi, lg);
        }

        // ── General (non-symmetric) eigenproblem ───────────────────────

        // Splits cuSOLVER's interleaved complex eigenvalues into two real
        // arrays, which is the layout Matrix<complex<T>> and the rest of this
        // package want.
        template <class T>
        __global__ void kSplitW(int n, const T* interleaved, T* re, T* im) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += blockDim.x * gridDim.x) {
                re[i] = interleaved[2 * i];
                im[i] = interleaved[2 * i + 1];
            }
        }

        // Unpacks LAPACK's real eigenvector storage into complex columns.
        //
        // A real eigenvalue owns one column of V and its vector is that column
        // with zero imaginary part. A CONJUGATE PAIR owns two adjacent columns
        // j and j+1: the first holds the shared real part, the second the
        // imaginary part, and the two eigenvectors are
        //
        //     v_j   = V[:,j] + i V[:,j+1]        (the one with wi > 0)
        //     v_j+1 = V[:,j] - i V[:,j+1]
        //
        // The pair is identified by wi[j] > 0, which cuSOLVER always puts
        // first, exactly as LAPACK does.
        template <class T>
        __global__ void kAssembleEvs(int n, const T* V, const T* wi, T* re, T* im) {
            const std::size_t total = (std::size_t)n * n;
            for (std::size_t k = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; k < total;
                 k += (std::size_t)blockDim.x * gridDim.x) {
                const int col = (int)(k / n);   // column-major: column is the outer index
                const int row = (int)(k % n);
                const T wcol = wi[col];
                if (wcol == T(0)) {
                    re[k] = V[k];
                    im[k] = T(0);
                } else if (wcol > T(0)) {
                    re[k] = V[k];
                    im[k] = V[(std::size_t)(col + 1) * n + row];
                } else {
                    re[k] = V[(std::size_t)(col - 1) * n + row];
                    im[k] = -V[k];
                }
            }
        }

        template <class T>
        static int geevImpl(cudaDataType realType, cudaDataType cplxType, int n, T* A, T* wr,
                            T* wi, T* VRr, T* VRi) {
            cusolverDnParams_t params = nullptr;
            checkSolver(cusolverDnCreateParams(&params), "cusolverDnCreateParams");
            const bool wantVec = (VRr != nullptr);
            const cusolverEigMode_t jobvr =
                wantVec ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

            Scratch W(sizeof(T) * 2 * (std::size_t)n);          // complex eigenvalues
            Scratch V(sizeof(T) * (std::size_t)n * n);          // REAL packed eigenvectors
            int result = 0;
            try {
                std::size_t devBytes = 0, hostBytes = 0;
                // VL is passed as a valid pointer even though jobvl is
                // NOVECTOR: cuSOLVER validates it regardless.
                checkSolver(cusolverDnXgeev_bufferSize(
                                solver(), params, CUSOLVER_EIG_MODE_NOVECTOR, jobvr, (int64_t)n,
                                realType, A, (int64_t)n, cplxType, W.p, realType, V.p, (int64_t)n,
                                realType, V.p, (int64_t)n, realType, &devBytes, &hostBytes),
                            "cusolverDnXgeev_bufferSize");
                Scratch devW(devBytes);
                std::vector<char> hostW(hostBytes ? hostBytes : 1);
                Info info;
                checkSolver(cusolverDnXgeev(solver(), params, CUSOLVER_EIG_MODE_NOVECTOR, jobvr,
                                            (int64_t)n, realType, A, (int64_t)n, cplxType, W.p,
                                            realType, V.p, (int64_t)n, realType, V.p, (int64_t)n,
                                            realType, devW.p, devBytes, hostW.data(), hostBytes,
                                            info.d),
                            "cusolverDnXgeev");
                result = info.get();
                if (result == 0) {
                    kSplitW<T><<<gridFor((std::size_t)n), kBlock>>>(n, W.template as<T>(), wr, wi);
                    checkLaunch("eigenvalue split");
                    if (wantVec) {
                        kAssembleEvs<T><<<gridFor((std::size_t)n * n), kBlock>>>(
                            n, V.template as<T>(), wi, VRr, VRi);
                        checkLaunch("eigenvector assembly");
                    }
                }
            } catch (...) {
                cusolverDnDestroyParams(params);
                throw;
            }
            cusolverDnDestroyParams(params);
            return result;
        }

        int geev(int n, double* A, double* wr, double* wi, double* VRr, double* VRi) {
            return geevImpl<double>(CUDA_R_64F, CUDA_C_64F, n, A, wr, wi, VRr, VRi);
        }
        int geev(int n, float* A, float* wr, float* wi, float* VRr, float* VRi) {
            return geevImpl<float>(CUDA_R_32F, CUDA_C_32F, n, A, wr, wi, VRr, VRi);
        }

        // ── Diagnostics ────────────────────────────────────────────────

        const char* backendVersions() {
            static std::string s;
            if (s.empty()) {
                int rt = 0, drv = 0;
                cudaRuntimeGetVersion(&rt);
                cudaDriverGetVersion(&drv);
                int bmaj = 0, bmin = 0, bpatch = 0;
                cublasGetProperty(MAJOR_VERSION, &bmaj);
                cublasGetProperty(MINOR_VERSION, &bmin);
                cublasGetProperty(PATCH_LEVEL, &bpatch);
                int smaj = 0, smin = 0, spatch = 0;
                cusolverGetProperty(MAJOR_VERSION, &smaj);
                cusolverGetProperty(MINOR_VERSION, &smin);
                cusolverGetProperty(PATCH_LEVEL, &spatch);
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "CUDA runtime %d.%d / driver %d.%d / cuBLAS %d.%d.%d / cuSOLVER "
                              "%d.%d.%d",
                              rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10, bmaj,
                              bmin, bpatch, smaj, smin, spatch);
                s = buf;
            }
            return s.c_str();
        }

    }  // namespace detail
}  // namespace mgpu
