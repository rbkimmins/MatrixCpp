using CSV, DataFrames, Plots

# ── Load data ──────────────────────────────────────────────────────────────────
df = CSV.read("running_time.txt", DataFrame)
n  = df.size
t  = df.time_seconds

# ── O(n³) reference curve ─────────────────────────────────────────────────────
# Scale it so it passes through the last measured point for easy visual comparison
ref_scale = t[end] / n[end]^3
t_ref = ref_scale .* n .^ 3

# ── Plot ───────────────────────────────────────────────────────────────────────
p = plot(n, t,
    label      = "Measured",
    xlabel     = "Matrix size n  (n × n)",
    ylabel     = "Time (seconds)",
    title      = "Matrix Multiplication Running Time",
    linewidth  = 2,
    color      = :steelblue,
    legend     = :topleft)

plot!(p, n, t_ref,
    label     = "O(n³) reference",
    linewidth = 1.5,
    linestyle = :dash,
    color     = :tomato)

savefig(p, "running_time.png")
println("Plot saved to running_time.png")

# ── Log-log plot (slope ≈ 3 confirms O(n³)) ────────────────────────────────────
p2 = plot(log10.(n), log10.(t),
    label      = "log₁₀(time) vs log₁₀(n)",
    xlabel     = "log₁₀(n)",
    ylabel     = "log₁₀(time / s)",
    title      = "Log-Log Plot  (slope ≈ 3 → O(n³))",
    linewidth  = 2,
    color      = :steelblue,
    legend     = :topleft)

# Fit a line to confirm the empirical slope
using Statistics
slope = cov(log10.(n), log10.(t)) / var(log10.(n))
intercept = mean(log10.(t)) - slope * mean(log10.(n))
t_fit = intercept .+ slope .* log10.(n)

plot!(p2, log10.(n), t_fit,
    label     = "Linear fit  (slope = $(round(slope, digits=2)))",
    linewidth = 1.5,
    linestyle = :dash,
    color     = :tomato)

savefig(p2, "running_time_loglog.png")
println("Log-log plot saved to running_time_loglog.png")
println("Empirical slope: $(round(slope, digits=3))  (expected ≈ 3.0 for O(n³))")
