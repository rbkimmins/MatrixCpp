import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

n         = 300
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

# ── O(n³) reference curve ─────────────────────────────────────────────────────
ref_scale = times[-1] / sizes[-1] ** 3
t_ref     = [ref_scale * s ** 3 for s in sizes]

# ── Linear plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(sizes, times, color="steelblue", linewidth=2,   label="NumPy measured")
ax.plot(sizes, t_ref, color="tomato",    linewidth=1.5,
        linestyle="--", label="O(n³) reference")
ax.set_xlabel("Matrix size n  (n × n)")
ax.set_ylabel("Time (seconds)")
ax.set_title("NumPy Matrix Multiplication Running Time")
ax.legend(loc="upper left")

# ── Log-log plot ──────────────────────────────────────────────────────────────
log_n = [np.log10(s) for s in sizes]
log_t = [np.log10(t) for t in times]

# Empirical slope via least-squares
mean_n = sum(log_n) / len(log_n)
mean_t = sum(log_t) / len(log_t)
slope     = sum((x - mean_n) * (y - mean_t) for x, y in zip(log_n, log_t)) \
          / sum((x - mean_n) ** 2            for x          in log_n)
intercept = mean_t - slope * mean_n
fit       = [intercept + slope * x for x in log_n]

ax2 = axes[1]
ax2.plot(log_n, log_t, color="steelblue", linewidth=2,
         label="log₁₀(time) vs log₁₀(n)")
ax2.plot(log_n, fit,   color="tomato",    linewidth=1.5,
         linestyle="--", label=f"Linear fit  (slope = {slope:.2f})")
ax2.set_xlabel("log₁₀(n)")
ax2.set_ylabel("log₁₀(time / s)")
ax2.set_title(f"Log-Log Plot  (slope ≈ {slope:.2f}  →  O(n^{slope:.2f}))")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("numpy_running_time.png", dpi=150)
plt.show()
print(f"Plot saved to numpy_running_time.png")
print(f"Empirical slope: {slope:.3f}  (expected ≈ 3.0 for O(n³), lower → BLAS optimisation)")
