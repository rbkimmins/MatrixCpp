import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

n         = 1000
LOW_BOUND = -1000.0
HIGH_BOUND =  1000.0

rng   = np.random.default_rng()
sizes = []
times = []

for i in range(2, n + 1):
    A = rng.uniform(LOW_BOUND, HIGH_BOUND, size=(i, i))
    B = rng.uniform(LOW_BOUND, HIGH_BOUND, size=(i, i))

    start = perf_counter()
    C = A @ B           # matrix multiply (not element-wise)
    end   = perf_counter()

    sizes.append(i)
    times.append(end - start)

# ── Write CSV (same format as running_time.txt for easy comparison) ───────────
with open("numpy_running_time.txt", "w") as f:
    f.write("size,time_seconds\n")
    for s, t in zip(sizes, times):
        f.write(f"{s},{t}\n")
print("Results written to numpy_running_time.txt")

# ── Load C++ benchmark data ───────────────────────────────────────────────────
cpp_sizes, cpp_times = [], []
try:
    with open("running_time.txt") as f:
        next(f)  # skip header
        for line in f:
            s, t = line.strip().split(",")
            cpp_sizes.append(int(s))
            cpp_times.append(float(t))
except FileNotFoundError:
    print("Warning: running_time.txt not found — C++ row will be omitted")

# ── Helpers ───────────────────────────────────────────────────────────────────
def empirical_slope(xs, ys):
    lx = np.log10(xs)
    ly = np.log10(ys)
    mx, my = lx.mean(), ly.mean()
    slope = ((lx - mx) * (ly - my)).sum() / ((lx - mx) ** 2).sum()
    intercept = my - slope * mx
    return slope, intercept, lx, ly

# ── Trim to common range if C++ data exists ───────────────────────────────────
if cpp_sizes:
    common_max = min(sizes[-1], cpp_sizes[-1])
    np_s  = np.array([s for s in sizes     if s <= common_max])
    np_t  = np.array([t for s, t in zip(sizes, times)         if s <= common_max])
    cpp_s = np.array([s for s in cpp_sizes if s <= common_max])
    cpp_t = np.array([t for s, t in zip(cpp_sizes, cpp_times) if s <= common_max])
else:
    np_s, np_t = np.array(sizes), np.array(times)
    cpp_s = cpp_t = np.array([])

# ── O(n³) reference (scaled to last NumPy point) ─────────────────────────────
ref_scale = np_t[-1] / np_s[-1] ** 3
t_ref     = ref_scale * np_s ** 3

# ── Empirical slopes ──────────────────────────────────────────────────────────
np_slope,  np_int,  np_lx,  np_ly  = empirical_slope(np_s, np_t)
np_fit = np_int + np_slope * np_lx

if cpp_sizes:
    cpp_slope, cpp_int, cpp_lx, cpp_ly = empirical_slope(cpp_s, cpp_t)

ref_lx = np_lx
ref_ly = np.log10(ref_scale * 10 ** (3 * ref_lx))

# ── Single figure: 2 rows × 2 cols ───────────────────────────────────────────
has_cpp  = len(cpp_sizes) > 0
nrows    = 2 if has_cpp else 1
fig, axes = plt.subplots(nrows, 2, figsize=(13, 5 * nrows))
if nrows == 1:
    axes = axes[np.newaxis, :]   # keep 2-D indexing consistent

# ── Row 0: NumPy only ─────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(np_s, np_t, color="steelblue", linewidth=2,   label="NumPy measured")
ax.plot(np_s, t_ref, color="tomato",   linewidth=1.5,
        linestyle="--", label="O(n³) reference")
ax.set_xlabel("Matrix size n  (n × n)")
ax.set_ylabel("Time (seconds)")
ax.set_title("NumPy Matrix Multiplication Running Time")
ax.legend(loc="upper left")

ax2 = axes[0, 1]
ax2.plot(np_lx, np_ly,  color="steelblue", linewidth=2,
         label="log₁₀(time) vs log₁₀(n)")
ax2.plot(np_lx, np_fit, color="tomato",    linewidth=1.5,
         linestyle="--", label=f"Linear fit  (slope = {np_slope:.2f})")
ax2.set_xlabel("log₁₀(n)")
ax2.set_ylabel("log₁₀(time / s)")
ax2.set_title(f"NumPy Log-Log  (slope ≈ {np_slope:.2f})")
ax2.legend(loc="upper left")

# ── Row 1: NumPy vs C++ comparison ───────────────────────────────────────────
if has_cpp:
    ax3 = axes[1, 0]
    ax3.plot(np_s,  np_t,  color="steelblue", linewidth=2,   label="NumPy")
    ax3.plot(cpp_s, cpp_t, color="seagreen",  linewidth=2,   label="C++")
    ax3.plot(np_s,  t_ref, color="tomato",    linewidth=1.5,
             linestyle="--", label="O(n³) reference")
    ax3.set_xlabel("Matrix size n  (n × n)")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("NumPy vs C++ Matrix Multiplication")
    ax3.legend(loc="upper left")

    ax4 = axes[1, 1]
    ax4.plot(np_lx,  np_ly,  color="steelblue", linewidth=2,
             label=f"NumPy  (slope = {np_slope:.2f})")
    ax4.plot(cpp_lx, cpp_ly, color="seagreen",  linewidth=2,
             label=f"C++    (slope = {cpp_slope:.2f})")
    ax4.plot(ref_lx, ref_ly, color="tomato",    linewidth=1.5,
             linestyle="--", label="O(n³) reference  (slope = 3)")
    ax4.set_xlabel("log₁₀(n)")
    ax4.set_ylabel("log₁₀(time / s)")
    ax4.set_title("Log-Log Comparison  (slope → algorithmic complexity)")
    ax4.legend(loc="upper left")

plt.tight_layout()
plt.savefig("running_time_comparison.png", dpi=150)
plt.show()
print("Plot saved to running_time_comparison.png")
print(f"NumPy empirical slope : {np_slope:.3f}")
if has_cpp:
    print(f"C++   empirical slope : {cpp_slope:.3f}")
