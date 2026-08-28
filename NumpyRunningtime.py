#!/usr/bin/env python3
"""
NumPy half of the MatrixCpp benchmark, plus the Python plots.

Runs the same operation set as Running_time.cpp — for both real and complex
matrices — and writes bench/numpy_<dtype>_<op>.csv alongside the C++ results.
Then draws the comparison.

The size points are not chosen here: they are read back out of the C++ CSVs, so
both implementations are measured at exactly the same n. No interpolation, no
mismatched ranges, and any op the C++ side skipped is skipped here too.

Usage:
    ./Running_time                  # writes bench/cpp_*.csv     (run this first)
    python3 NumpyRunningtime.py     # writes bench/numpy_*.csv, then plots
    python3 NumpyRunningtime.py --plot-only     # re-draw without re-timing

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
PLOT_DIR = "plots"

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
    "tensor":         lambda n, cx: (lambda a, b: (lambda: np.kron(a, b)))(*_two(n, cx)),
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


# ──────────────────────────────────────────────────────────── plotting ──
def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    os.makedirs(os.path.join(PLOT_DIR, "ops"), exist_ok=True)

    CPP = "#2a9d5c"      # green  — C++
    NPY = "#3d7ebd"      # blue   — NumPy
    WIN = "#2a9d5c"
    LOSE = "#c0392b"

    pairs = {}
    for (dtype, op), cpath in cpp_files().items():
        npath = os.path.join(BENCH_DIR, f"numpy_{dtype}_{op}.csv")
        if not os.path.exists(npath):
            continue
        cs, ct = read_csv(cpath)
        ns, nt = read_csv(npath)
        common = np.intersect1d(cs, ns)
        if len(common) < 2:
            continue
        ct = np.array([ct[list(cs).index(s)] for s in common])
        nt = np.array([nt[list(ns).index(s)] for s in common])
        good = (ct > 0) & (nt > 0)
        if good.sum() < 2:
            continue
        pairs[(dtype, op)] = (common[good], ct[good], nt[good])

    if not pairs:
        sys.exit("No matched cpp/numpy pairs to plot.")

    # ── per-operation figure: times on the left, speedup on the right ──
    for (dtype, op), (n, ct, nt) in sorted(pairs.items()):
        ratio = nt / ct
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.loglog(n, ct, "o-", color=CPP, lw=2, ms=4, label="MatrixCpp")
        ax1.loglog(n, nt, "s-", color=NPY, lw=2, ms=4, label="NumPy")
        ax1.set_xlabel("Matrix size n  (n × n)")
        ax1.set_ylabel("Time (s)   — lower is better")
        ax1.set_title(f"{op}  [{dtype}]")
        ax1.grid(True, which="both", alpha=0.25)
        ax1.legend()

        ax2.axhline(1.0, color="#555", lw=1.2, ls="--", zorder=2)
        ax2.fill_between(n, 1.0, ratio, where=ratio >= 1, color=WIN, alpha=0.22,
                         interpolate=True, zorder=1)
        ax2.fill_between(n, 1.0, ratio, where=ratio < 1, color=LOSE, alpha=0.22,
                         interpolate=True, zorder=1)
        ax2.plot(n, ratio, "o-", color="#222", lw=2, ms=4, zorder=3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
        ax2.set_xlabel("Matrix size n")
        ax2.set_ylabel("NumPy time ÷ MatrixCpp time")
        ax2.set_title(f"speedup  —  above 1 = MatrixCpp wins  ({ratio[-1]:.2f}× at n={n[-1]})")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.text(0.02, 0.95, "MatrixCpp faster", transform=ax2.transAxes,
                 color=WIN, fontweight="bold", va="top", fontsize=9)
        ax2.text(0.02, 0.05, "NumPy faster", transform=ax2.transAxes,
                 color=LOSE, fontweight="bold", va="bottom", fontsize=9)

        fig.tight_layout()
        out = os.path.join(PLOT_DIR, "ops", f"{dtype}_{op}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)

    print(f"\n  {len(pairs)} per-op figures -> {PLOT_DIR}/ops/")

    # ── summary: one bar per operation, at the largest common size ──
    for dtype in ("real", "complex"):
        items = []
        for (dt, op), (n, ct, nt) in pairs.items():
            if dt != dtype:
                continue
            items.append((op, nt[-1] / ct[-1], int(n[-1])))
        if not items:
            continue
        items.sort(key=lambda r: r[1])
        labels = [f"{op}  (n={nn})" for op, _, nn in items]
        vals = [r[1] for r in items]
        colors = [WIN if v >= 1 else LOSE for v in vals]

        fig, ax = plt.subplots(figsize=(11, 0.42 * len(items) + 2.2))
        ax.barh(labels, vals, color=colors, alpha=0.85)
        ax.axvline(1.0, color="#333", lw=1.4, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("NumPy time ÷ MatrixCpp time   (log scale, >1 = MatrixCpp faster)")
        ax.set_title(f"MatrixCpp vs NumPy — {dtype}, at the largest measured size")
        ax.grid(True, axis="x", which="both", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(v * (1.06 if v >= 1 else 0.94), i, f"{v:.2f}×",
                    va="center", ha="left" if v >= 1 else "right", fontsize=8.5)
        fig.tight_layout()
        out = os.path.join(PLOT_DIR, f"summary_{dtype}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  summary -> {out}")

    # ── console table, so the numbers are readable without opening a PNG ──
    print("\n" + "=" * 74)
    print(f"{'dtype':9s} {'operation':17s} {'n':>6s} {'MatrixCpp':>12s} {'NumPy':>12s} {'speedup':>9s}")
    print("=" * 74)
    for (dtype, op), (n, ct, nt) in sorted(pairs.items()):
        r = nt[-1] / ct[-1]
        mark = "" if 0.9 <= r <= 1.1 else ("  <<" if r < 1 else "  >>")
        print(f"{dtype:9s} {op:17s} {int(n[-1]):6d} {ct[-1]:12.6f} {nt[-1]:12.6f} {r:8.2f}x{mark}")
    print("=" * 74)
    print(">> MatrixCpp faster    << NumPy faster")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_benchmarks()
    make_plots()
