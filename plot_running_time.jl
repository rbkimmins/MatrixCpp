using CSV, DataFrames, Plots, Statistics

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — maps each benchmark name → (plot title, expected O(n^k))
#  Comment out any row to skip that plot even if the data file exists.
# ════════════════════════════════════════════════════════════════════════════
const BENCHMARKS = [
    # name               title                                        complexity
    ("trace",           "Trace  tr(A)",                              1),
    ("add",             "Element-wise Add  A + B",                   2),
    ("subtract",        "Element-wise Subtract  A - B",              2),
    ("scalar_mul",      "Scalar Multiply  A * k",                    2),
    ("scalar_div",      "Scalar Divide  A / k",                      2),
    ("hadamard",        "Hadamard Product  A % B",                   2),
    ("elem_div",        "Element-wise Divide  A / B",                2),
    ("transpose",       "Transpose  A.T()",                          2),
    ("elem_pow",        "Element-wise Power  A.pow(p)",              2),
    ("elem_exp",        "Element-wise Exp  A.exp()",                 2),
    ("elem_ln",         "Element-wise Ln  A.ln()",                   2),
    ("sum_all",         "Total Sum  A.sum()",                        2),
    ("sum_dim",         "Row Sums  A.sum(true)",                     2),
    ("isdiagonal",      "IsDiagonal  A.IsDiagonal()",               2),
    ("concat",          "Horizontal Concat  A.concat(B, true)",      2),
    ("multiply",        "Matrix Multiply  A * B  (Strassen-Winograd)", 3),
    ("det",             "Determinant  det(A)  via LU",               3),
    ("inverse",         "Matrix Inverse  A.inverse()",               3),
    ("lu",              "LU Factorization  A.LU()",                  3),
    ("qr",              "QR Factorization  A.QR()",                  3),
    ("mat_pow_int",     "Matrix Power  pow(A, 3)  binary squaring",  3),
    ("eig",             "Eigendecomposition  A.eig()",               3),
    ("mat_pow_real",    "Matrix Square Root  pow(A, 0.5)  Schur-Padé", 3),
    ("mat_log",         "Matrix Log  log(A, e)  Schur-Padé",        3),
    ("tensor",          "Kronecker Product  A.tensor(B)",            4),
]

# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════
function empirical_slope(n, t)
    ln = log10.(n); lt = log10.(t)
    m = cov(ln, lt) / var(ln)
    b = mean(lt) - m * mean(ln)
    m, b
end

function ref_curve(n, t, k)
    # Scale O(n^k) to pass through the last data point
    scale = t[end] / n[end]^k
    scale .* n .^ k
end

# Divergence label for the reference curve
ref_label(k) = k == 1 ? "O(n) reference" :
               k == 2 ? "O(n²) reference" :
               k == 3 ? "O(n³) reference" :
                        "O(n⁴) reference"

# ════════════════════════════════════════════════════════════════════════════
#  Main plotting loop
# ════════════════════════════════════════════════════════════════════════════
mkpath("plots")
plotted = 0

for (name, title, complexity) in BENCHMARKS
    fname = "rt_$(name).txt"
    if !isfile(fname)
        println("  skip  $fname  (not found)")
        continue
    end

    df = CSV.read(fname, DataFrame)
    n  = Float64.(df.size)
    t  = Float64.(df.time_seconds)

    # Drop any non-positive times (timing noise on very fast ops)
    valid = t .> 0
    n, t = n[valid], t[valid]
    isempty(n) && continue

    t_ref  = ref_curve(n, t, complexity)
    slope, intercept = empirical_slope(n, t)
    t_fit  = intercept .+ slope .* log10.(n)

    # ── Figure: 1 row × 2 columns ──────────────────────────────────────────
    fig = plot(
        layout  = (1, 2),
        size    = (1200, 500),
        margin  = 5Plots.mm,
    )

    # Left panel: linear scale
    plot!(fig[1],
        n, t,
        label     = "Measured",
        color     = :steelblue,
        linewidth = 2,
        xlabel    = "Matrix size n  (n × n)",
        ylabel    = "Time (s)",
        title     = title,
        legend    = :topleft,
    )
    plot!(fig[1],
        n, t_ref,
        label     = ref_label(complexity),
        color     = :tomato,
        linewidth = 1.5,
        linestyle = :dash,
    )

    # Right panel: log-log
    plot!(fig[2],
        log10.(n), log10.(t),
        label     = "log₁₀(time) vs log₁₀(n)",
        color     = :steelblue,
        linewidth = 2,
        xlabel    = "log₁₀(n)",
        ylabel    = "log₁₀(time / s)",
        title     = "Log-Log  (slope ≈ $(round(slope, digits=2)))",
        legend    = :topleft,
    )
    plot!(fig[2],
        log10.(n), t_fit,
        label     = "Fit  slope = $(round(slope, digits=2))",
        color     = :tomato,
        linewidth = 1.5,
        linestyle = :dash,
    )

    out = "plots/$(name).png"
    savefig(fig, out)
    println("  saved  $out   (slope=$(round(slope,digits=3)), expected≈$complexity)")
    global plotted += 1
end

println("\n$plotted plot(s) saved to plots/")
