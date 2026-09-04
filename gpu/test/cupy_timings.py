#!/usr/bin/env python3
"""
CuPy timings, for the third column of the comparison table.

CuPy is the fair reference for this package: it is NumPy's API over the same
cuBLAS/cuSOLVER this library calls, on the same card. So the comparison is not
"is a GPU faster than a CPU" -- benchmark.cpp already answers that -- but
"does our C++ front-end cost anything against a mature one". Both sides end up
in the same NVIDIA kernels; a large gap either way would mean one of us is
marshalling badly.

Run:
    PYTHONPATH=~/.local/lib/matrixcpp-cupy python3 gpu/test/cupy_timings.py

CuPy lives in its own directory rather than in system site-packages so that
benchmarks/numpy_timings.py keeps seeing numpy 1.26.4 + scipy 1.11.4; setting
PYTHONPATH is what opts in to the newer numpy CuPy needs. See gpu/README.md.

Timing method matches benchmark.cpp exactly: min of N, one untimed warm-up,
and an explicit stream synchronise inside the timed region, because a CuPy
call returns as soon as the kernel is queued.
"""

import csv
import os
import sys
import time

try:
    import cupy as cp
except ImportError:
    sys.exit("cupy not importable - see the header of this file for PYTHONPATH")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def timed(reps, fn):
    fn()
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"{props['name'].decode()}  cupy {cp.__version__}\n")
    rows = []

    def record(op, n, dtype, ms, flops=0.0):
        rate = f"{flops / (ms * 1e-3) / 1e9:9.1f} GFLOP/s" if flops else ""
        print(f"  {op:<22} {n:>5}  {dtype:<6}  {ms:9.2f} ms  {rate}")
        rows.append((op, n, dtype, f"{ms:.4f}"))

    print("GEMM")
    for n in (512, 1024, 2048, 4096):
        a = cp.random.rand(n, n, dtype=cp.float64)
        b = cp.random.rand(n, n, dtype=cp.float64)
        record("gemm", n, "f64", timed(5, lambda: a @ b), 2.0 * n ** 3)
    for n in (1024, 2048, 4096):
        a = cp.random.rand(n, n, dtype=cp.float32)
        b = cp.random.rand(n, n, dtype=cp.float32)
        record("gemm", n, "f32", timed(10, lambda: a @ b), 2.0 * n ** 3)

    print("\nElement-wise")
    for n in (1024, 4096):
        a = cp.random.rand(n, n, dtype=cp.float64)
        b = cp.random.rand(n, n, dtype=cp.float64)
        record("(A%B).exp().sqrt()", n, "f64", timed(20, lambda: cp.sqrt(cp.exp(a * b))))

    # CuPy has fusion too (cupy.fuse), so the honest comparison for our
    # GpuExpr is against CuPy's fused path, not only its unfused one. Both
    # collapse an element-wise chain into a single kernel; the difference is
    # that CuPy compiles the exact expression with NVRTC at first call, while
    # we interpret a postfix program in a precompiled kernel. NVRTC should win
    # on arithmetic-heavy chains -- it emits straight-line code with no decode
    # -- and the two should converge when memory is the limit.
    print("\nElement-wise, fused")

    @cp.fuse()
    def chain3(a, b):
        return cp.sqrt(cp.exp(a * b))

    @cp.fuse()
    def cheap4(a, b, c):
        return ((a * b) + c) * a - b

    for n in (1024, 4096):
        a = cp.random.rand(n, n, dtype=cp.float64)
        b = cp.random.rand(n, n, dtype=cp.float64)
        record("fused_chain3", n, "f64", timed(20, lambda: chain3(a, b)))
    for n in (4096,):
        a = cp.random.rand(n, n, dtype=cp.float64)
        b = cp.random.rand(n, n, dtype=cp.float64)
        c = cp.random.rand(n, n, dtype=cp.float64)
        record("fused_cheap", n, "f64", timed(20, lambda: cheap4(a, b, c)))

    print("\nFactorisations")
    for n in (512, 1024, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        record("lu", n, "f64", timed(5, lambda: cp.linalg.lu_factor(a))
               if hasattr(cp.linalg, "lu_factor") else timed(5, lambda: _lu(a)))
    for n in (512, 1024, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        s = a.T @ a + n * cp.eye(n, dtype=cp.float64)
        record("cholesky", n, "f64", timed(5, lambda: cp.linalg.cholesky(s)))
    for n in (512, 1024, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        record("qr", n, "f64", timed(5, lambda: cp.linalg.qr(a, mode="reduced")))
    for n in (256, 512, 1024, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        record("svd", n, "f64", timed(3, lambda: cp.linalg.svd(a)))
    for n in (256, 512, 1024):
        a = cp.random.rand(n, n, dtype=cp.float64)
        s = a.T @ a + n * cp.eye(n, dtype=cp.float64)
        record("eigSym", n, "f64", timed(3, lambda: cp.linalg.eigh(s)))
    for n in (512, 1024, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        s = a.T @ a + n * cp.eye(n, dtype=cp.float64)
        b = cp.random.rand(n, 1, dtype=cp.float64)
        record("solve", n, "f64", timed(5, lambda: cp.linalg.solve(s, b)))

    print("\nFFT")
    for n in (4096, 65536, 1048576):
        x = cp.random.rand(1, n, dtype=cp.float64)
        record("fft1d", n, "f64", timed(20, lambda: cp.fft.fft(x, axis=1)))
    for n in (512, 2048):
        a = cp.random.rand(n, n, dtype=cp.float64)
        record("fft_batch", n, "f64", timed(10, lambda: cp.fft.fft(a, axis=0)))
    # Convolution by transform, written out rather than taken from
    # cupyx.scipy.signal: that module imports the system scipy, which was built
    # against numpy 1.x and will not load beside the numpy 2.x CuPy needs. Doing
    # it directly is also the fairer comparison, since it is exactly the
    # algorithm mgpu::conv uses -- pad, transform both, multiply, invert.
    nc = 200000 + 20000 - 1
    ap = cp.zeros(nc, dtype=cp.float64); ap[:200000] = cp.random.rand(200000)
    bp = cp.zeros(nc, dtype=cp.float64); bp[:20000] = cp.random.rand(20000)

    def fftconv():
        return cp.real(cp.fft.ifft(cp.fft.fft(ap) * cp.fft.fft(bp)))

    record("conv", 200000, "f64", timed(10, fftconv))

    # CuPy exposes no mixed-precision gesv, so its solve column is fp64 only --
    # which is itself the comparison worth seeing.
    print("\nSolve")
    for n in (1024, 2048, 4096):
        a = cp.random.rand(n, n, dtype=cp.float64)
        s = a.T @ a + n * cp.eye(n, dtype=cp.float64)
        b = cp.random.rand(n, 1, dtype=cp.float64)
        record("solve_mixed", n, "f64", timed(5, lambda: cp.linalg.solve(s, b)))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "cupy.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "n", "dtype", "cupy_ms"])
        w.writerows(rows)
    print(f"\nwrote {out}")
    compare()


def _lu(a):
    # Older CuPy exposes the factorisation only through the solver module.
    from cupyx.scipy.linalg import lu_factor
    return lu_factor(a)


def compare():
    """Three-way table: basic/ on the CPU, this package, and CuPy."""
    cpp_path = os.path.join(RESULTS, "cpp.csv")
    if not os.path.exists(cpp_path):
        print("\n(run `make run-bench` first for the C++ columns)")
        return
    cpp = {}
    with open(cpp_path) as f:
        for r in csv.DictReader(f):
            cpp[(r["op"], r["n"], r["dtype"])] = (float(r["cpu_ms"]), float(r["gpu_ms"]))
    cup = {}
    with open(os.path.join(RESULTS, "cupy.csv")) as f:
        for r in csv.DictReader(f):
            cup[(r["op"], r["n"], r["dtype"])] = float(r["cupy_ms"])

    print("\n" + "=" * 78)
    print("  basic/ (CPU)  vs  MatrixCpp GPU  vs  CuPy      [min of N, ms]")
    print("=" * 78)
    print(f"  {'op':<22}{'n':>6} {'dt':<5}{'CPU':>10}{'ours':>10}{'CuPy':>10}{'ours/CuPy':>11}")
    print("-" * 78)
    for key in cpp:
        if key not in cup:
            continue
        op, n, dt = key
        cpu_ms, gpu_ms = cpp[key]
        cu = cup[key]
        cpu_s = f"{cpu_ms:10.2f}" if cpu_ms > 0 else f"{'-':>10}"
        print(f"  {op:<22}{n:>6} {dt:<5}{cpu_s}{gpu_ms:10.2f}{cu:10.2f}{gpu_ms / cu:10.2f}x")
    print("=" * 78)
    print("  ours/CuPy near 1.00 is the expected result: both call the same")
    print("  cuBLAS and cuSOLVER routines, so this column measures front-end")
    print("  overhead -- marshalling, layout conversion, allocation -- only.")


if __name__ == "__main__":
    main()
