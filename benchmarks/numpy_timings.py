#!/usr/bin/env python3
"""
NumPy half of the MatrixCpp benchmark.

Runs the same operation set as benchmarks/running_time.cpp — for both real and
complex matrices — and writes bench/numpy_<dtype>_<op>.csv alongside the C++
results, then prints a comparison table.

IT DOES NOT PLOT. Every figure in this project is drawn by C++, through the
plotting package: see benchmarks/plot_comparison.cpp, which reads the same CSVs
this writes. The CSVs exist only because a CROSS-LANGUAGE comparison has to
persist data somewhere — the C++-only speed plots keep nothing on disk.

The size points are not chosen here: they are read back out of the C++ CSVs, so
both implementations are measured at exactly the same n. No interpolation, no
mismatched ranges, and any op the C++ side skipped is skipped here too.

Usage, from the repo root:
    ./running_time                            # writes bench/cpp_*.csv  (first)
    python3 benchmarks/numpy_timings.py       # writes bench/numpy_*.csv + table
    python3 benchmarks/numpy_timings.py --summary-only   # table, no re-timing
    ./plot_comparison                         # C++ draws the comparison

Fairness notes, because several NumPy operations are lazy:
  * A.T and A.conj().T return VIEWS in NumPy and cost O(1). The C++ T()/H()
    materialise a new matrix, so these are wrapped in ascontiguousarray() to
    force the copy that is actually being compared.
  * reshape is likewise a view in NumPy and a copy in C++, so it is .copy()ed.
  * QR is compared against scipy's pivoting=True, since the C++ QR does column
    pivoting and NumPy's does not.
  * eig uses np.linalg.eig, not eigh, because the C++ eig() runs its general
    implicit-shift QR even when handed a symmetric matrix.
"""
import os
import sys
import csv
import time
import numpy as np

try:
    import scipy.linalg as sla
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("scipy not found — lu / sqrtm / logm / expm will be skipped")

BENCH_DIR = "bench"

MIN_SECONDS = 0.05
MAX_REPS = 100
WARMUP = 1

rng = np.random.default_rng(12345)


# ───────────────────────────────────────────────────────────── timing ──
def time_best(fn):
    """Best-of-k, matching timeBest() in Running_time.cpp."""
    for _ in range(WARMUP):
        fn()
    best, total, reps = float("inf"), 0.0, 0
    while reps < MAX_REPS and total < MIN_SECONDS:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        reps += 1
    return best


# ──────────────────────────────────────────────────────── operand setup ──
def rand(n, m, complex_):
    A = rng.uniform(-1.0, 1.0, size=(n, m))
    if complex_:
        A = A + 1j * rng.uniform(-1.0, 1.0, size=(n, m))
    return np.ascontiguousarray(A)


def well_conditioned(n):
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    A[np.diag_indices(n)] += n
    return np.ascontiguousarray(A)


def symmetric(n):
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    return np.ascontiguousarray((A + A.T) * 0.5)


def spd(n):
    A = symmetric(n)
    A[np.diag_indices(n)] += n + 1.0
    return np.ascontiguousarray(A)


# ─────────────────────────────────────────────────────── operation table ──
# key -> callable(n, complex_) returning the zero-argument thunk to be timed.
def _two(n, cx):
    return rand(n, n, cx), rand(n, n, cx)


SHARED_OPS = {
    "add":            lambda n, cx: (lambda a, b: (lambda: a + b))(*_two(n, cx)),
    "subtract":       lambda n, cx: (lambda a, b: (lambda: a - b))(*_two(n, cx)),
    "hadamard":       lambda n, cx: (lambda a, b: (lambda: a * b))(*_two(n, cx)),
    "elem_div":       lambda n, cx: (lambda a, b: (lambda: a / b))(rand(n, n, cx), rand(n, n, cx) + 2),
    "scalar_mul":     lambda n, cx: (lambda a: (lambda: a * 1.0000001))(rand(n, n, cx)),
    "scalar_div":     lambda n, cx: (lambda a: (lambda: a / 1.0000001))(rand(n, n, cx)),
    "elem_exp":       lambda n, cx: (lambda a: (lambda: np.exp(a)))(rand(n, n, cx)),
    # The C++ side rounds the size down to a power of two, so match that here or
    # the two would be transforming different lengths.
    "fft":            lambda n, cx: (lambda a: (lambda: np.fft.fft(a)))(
                          rand(1, 1 << (int(n).bit_length() - 1), cx).ravel()),
    "elem_ln":        lambda n, cx: (lambda a: (lambda: np.log(a)))(rand(n, n, cx) + 2),
    "elem_pow":       lambda n, cx: (lambda a: (lambda: a ** 2.5))(rand(n, n, cx) + 2),
    # ascontiguousarray forces the copy; bare .T would time a no-op view.
    "transpose":      lambda n, cx: (lambda a: (lambda: np.ascontiguousarray(a.T)))(rand(n, n, cx)),
    "triu":           lambda n, cx: (lambda a: (lambda: np.triu(a)))(rand(n, n, cx)),
    "reshape":        lambda n, cx: (lambda a: (lambda: a.reshape(1, n * n).copy()))(rand(n, n, cx)),
    "concat":         lambda n, cx: (lambda a, b: (lambda: np.hstack((a, b))))(*_two(n, cx)),
    "conj_transpose": lambda n, cx: (lambda a: (lambda: np.ascontiguousarray(a.conj().T)))(rand(n, n, cx)),
    "conjugate":      lambda n, cx: (lambda a: (lambda: np.conj(a)))(rand(n, n, cx)),
    "sum_all":        lambda n, cx: (lambda a: (lambda: a.sum()))(rand(n, n, cx)),
    "sum_dim":        lambda n, cx: (lambda a: (lambda: a.sum(axis=1)))(rand(n, n, cx)),
    "norm_fro":       lambda n, cx: (lambda a: (lambda: np.linalg.norm(a)))(rand(n, n, cx)),
    "isdiagonal":     lambda n, cx: (lambda a: (lambda: np.array_equal(a, np.diag(np.diag(a)))))(
                          np.diag(np.diag(rand(n, n, cx)))),
    "trace":          lambda n, cx: (lambda a: (lambda: np.trace(a)))(rand(n, n, cx)),
    "multiply":       lambda n, cx: (lambda a, b: (lambda: a @ b))(*_two(n, cx)),
    "kron":         lambda n, cx: (lambda a, b: (lambda: np.kron(a, b)))(*_two(n, cx)),
}

REAL_ONLY_OPS = {
    "det":          lambda n, cx: (lambda a: (lambda: np.linalg.det(a)))(well_conditioned(n)),
    "inverse":      lambda n, cx: (lambda a: (lambda: np.linalg.inv(a)))(well_conditioned(n)),
    "solve":        lambda n, cx: (lambda a, b: (lambda: np.linalg.solve(a, b)))(
                        well_conditioned(n), rand(n, 1, False)),
    "qr":           lambda n, cx: (lambda a: (lambda: sla.qr(a, pivoting=True)))(well_conditioned(n)),
    "svd":          lambda n, cx: (lambda a: (lambda: np.linalg.svd(a)))(well_conditioned(n)),
    "cholesky":     lambda n, cx: (lambda a: (lambda: np.linalg.cholesky(a)))(spd(n)),
    "rank":         lambda n, cx: (lambda a: (lambda: np.linalg.matrix_rank(a)))(well_conditioned(n)),
    "eig":          lambda n, cx: (lambda a: (lambda: np.linalg.eig(a)))(symmetric(n)),
    "mat_pow_int":  lambda n, cx: (lambda a: (lambda: np.linalg.matrix_power(a, 3)))(well_conditioned(n)),
    "lu":           lambda n, cx: (lambda a: (lambda: sla.lu(a)))(well_conditioned(n)),
    "mat_pow_real": lambda n, cx: (lambda a: (lambda: sla.sqrtm(a)))(spd(n)),
    "mat_log":      lambda n, cx: (lambda a: (lambda: sla.logm(a)))(spd(n)),
    "mat_exp":      lambda n, cx: (lambda a: (lambda: sla.expm(a)))(rng.uniform(-0.5, 0.5, (n, n))),
}
SCIPY_OPS = {"lu", "qr", "mat_pow_real", "mat_log", "mat_exp"}


# ─────────────────────────────────────────────────────────────── io ──
def read_csv(path):
    sizes, times = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                sizes.append(int(row[0]))
                times.append(float(row[1]))
    return np.array(sizes), np.array(times)


def cpp_files():
    """Every C++ result present, as {(dtype, op): path}."""
    out = {}
    if not os.path.isdir(BENCH_DIR):
        return out
    for fn in sorted(os.listdir(BENCH_DIR)):
        if fn.startswith("cpp_") and fn.endswith(".csv"):
            stem = fn[len("cpp_"):-len(".csv")]
            dtype, _, op = stem.partition("_")
            out[(dtype, op)] = os.path.join(BENCH_DIR, fn)
    return out


# ─────────────────────────────────────────────────────── benchmarking ──
def run_benchmarks():
    os.makedirs(BENCH_DIR, exist_ok=True)
    targets = cpp_files()
    if not targets:
        sys.exit("No bench/cpp_*.csv found — build and run ./Running_time first.")

    print(f"NumPy {np.__version__}  |  {len(targets)} operations to match\n")
    for (dtype, op), path in sorted(targets.items()):
        cx = dtype == "complex"
        table = SHARED_OPS if op in SHARED_OPS else REAL_ONLY_OPS
        if op not in table:
            print(f"  skip   {dtype:8s} {op:16s} (no NumPy equivalent registered)")
            continue
        if op in SCIPY_OPS and not HAVE_SCIPY:
            print(f"  skip   {dtype:8s} {op:16s} (needs scipy)")
            continue

        sizes, _ = read_csv(path)
        out_path = os.path.join(BENCH_DIR, f"numpy_{dtype}_{op}.csv")
        rows = []
        for n in sizes:
            try:
                thunk = table[op](int(n), cx)
                rows.append((int(n), time_best(thunk)))
            except Exception as exc:                      # noqa: BLE001
                print(f"  warn   {dtype} {op} n={n}: {type(exc).__name__}: {exc}")
                break
        if not rows:
            continue
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size", "time_seconds"])
            w.writerows(rows)
        print(f"  wrote  {out_path}  ({len(rows)} points)")


# ─────────────────────────────────────────────────────────── summary ──
def summarise():
    """Console table only. THE PLOTS ARE DRAWN BY C++ — see
    benchmarks/plot_comparison.cpp, which reads the same CSVs this writes."""
    pairs = {}
    for (dtype, op), path in cpp_files().items():
        npath = os.path.join(BENCH_DIR, f"numpy_{dtype}_{op}.csv")
        if not os.path.exists(npath):
            continue
        n, ct = read_csv(path)
        _, nt = read_csv(npath)
        if len(n) and len(ct) and len(nt):
            pairs[(dtype, op)] = (n, ct, nt)
    if not pairs:
        print("No matching cpp_/numpy_ pairs in bench/.")
        return
    print("\n" + "=" * 74)
    print(f"{'dtype':9s} {'operation':17s} {'n':>6s} {'MatrixCpp':>12s} "
          f"{'NumPy':>12s} {'speedup':>8s}")
    print("=" * 74)
    wins = 0
    for (dtype, op), (n, ct, nt) in sorted(pairs.items()):
        r = nt[-1] / ct[-1]
        if r > 1.0:
            wins += 1
        mark = "" if 0.9 <= r <= 1.1 else ("  <<" if r < 1 else "  >>")
        print(f"{dtype:9s} {op:17s} {int(n[-1]):6d} {ct[-1]:12.6f} "
              f"{nt[-1]:12.6f} {r:8.2f}{mark}")
    print("=" * 74)
    print(f">> MatrixCpp faster    << NumPy faster     ({wins}/{len(pairs)} to MatrixCpp)")
    print("\nPlots:  g++ -std=c++17 -O2 -fopenmp -I. benchmarks/plot_comparison.cpp "
          "-o plot_comparison && ./plot_comparison")



if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        run_benchmarks()
    summarise()
