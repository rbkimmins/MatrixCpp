# ════════════════════════════════════════════════════════════════════════════
#  Julia plots for the MatrixCpp vs NumPy benchmark.
#
#  Reads the CSVs written by Running_time.cpp and NumpyRunningtime.py:
#      bench/cpp_<dtype>_<op>.csv
#      bench/numpy_<dtype>_<op>.csv
#  and draws, for every operation present in both:
#      plots/julia/ops/<dtype>_<op>.png   times (log-log) + speedup ratio
#      plots/julia/summary_<dtype>.png    every operation on one bar chart
#      plots/julia/scaling_<dtype>.png    measured exponent vs the ideal
#
#  Run:
#      ./Running_time                     # writes bench/cpp_*.csv
#      python3 NumpyRunningtime.py        # writes bench/numpy_*.csv
#      julia --project=. plot_running_time.jl
# ════════════════════════════════════════════════════════════════════════════
using CSV, DataFrames, Plots, Statistics, Printf

const BENCH = "bench"
const OUT   = joinpath("plots", "julia")

const CPP_COLOR = RGB(0.165, 0.616, 0.361)   # green — MatrixCpp
const NPY_COLOR = RGB(0.239, 0.494, 0.741)   # blue  — NumPy
const WIN       = RGB(0.165, 0.616, 0.361)
const LOSE      = RGB(0.753, 0.224, 0.169)

# Expected complexity exponent per operation, for the scaling panel.
const EXPECTED = Dict(
    "trace" => 1,
    "add" => 2, "subtract" => 2, "hadamard" => 2, "elem_div" => 2,
    "scalar_mul" => 2, "scalar_div" => 2, "elem_exp" => 2, "elem_ln" => 2,
    "elem_pow" => 2, "transpose" => 2, "triu" => 2, "reshape" => 2,
    "concat" => 2, "conj_transpose" => 2, "conjugate" => 2,
    "sum_all" => 2, "sum_dim" => 2, "norm_fro" => 2, "isdiagonal" => 2,
    "multiply" => 3, "det" => 3, "inverse" => 3, "solve" => 3, "lu" => 3,
    "qr" => 3, "svd" => 3, "cholesky" => 3, "rank" => 3, "eig" => 3,
    "mat_pow_int" => 3, "mat_pow_real" => 3, "mat_log" => 3, "mat_exp" => 3,
    "tensor" => 4,
)

read_bench(path) = begin
    df = CSV.read(path, DataFrame)
    Float64.(df.size), Float64.(df.time_seconds)
end

# Least-squares slope of log10(t) against log10(n) — the measured exponent.
function empirical_slope(n, t)
    ln, lt = log10.(n), log10.(t)
    length(ln) < 2 && return NaN
    cov(ln, lt) / var(ln)
end

"""Every (dtype, op) that has BOTH a cpp and a numpy CSV, aligned on shared sizes."""
function collect_pairs()
    pairs = Dict{Tuple{String,String},NTuple{3,Vector{Float64}}}()
    isdir(BENCH) || return pairs
    for f in sort(readdir(BENCH))
        (startswith(f, "cpp_") && endswith(f, ".csv")) || continue
        stem  = f[5:end-4]
        parts = split(stem, "_", limit = 2)
        length(parts) == 2 || continue
        dtype, op = String(parts[1]), String(parts[2])

        npath = joinpath(BENCH, "numpy_$(dtype)_$(op).csv")
        isfile(npath) || continue

        cs, ct = read_bench(joinpath(BENCH, f))
        ns, nt = read_bench(npath)

        # Align on the sizes both sides actually measured.
        common = intersect(cs, ns)
        length(common) < 2 && continue
        sort!(common)
        cidx = [findfirst(==(s), cs) for s in common]
        nidx = [findfirst(==(s), ns) for s in common]
        cts, nts = ct[cidx], nt[nidx]

        keep = (cts .> 0) .& (nts .> 0)
        sum(keep) < 2 && continue
        pairs[(dtype, op)] = (common[keep], cts[keep], nts[keep])
    end
    return pairs
end

function per_op_figure(dtype, op, n, ct, nt)
    ratio = nt ./ ct
    last_r = ratio[end]

    p1 = plot(n, ct;
        label = "MatrixCpp", color = CPP_COLOR, lw = 2, marker = :circle, ms = 3,
        xscale = :log10, yscale = :log10,
        xlabel = "Matrix size n  (n × n)", ylabel = "Time (s) — lower is better",
        title = "$op  [$dtype]", legend = :topleft, grid = true)
    plot!(p1, n, nt;
        label = "NumPy", color = NPY_COLOR, lw = 2, marker = :rect, ms = 3)

    # Speedup panel. Shading makes the win/lose side readable at a glance.
    p2 = plot(n, ratio;
        label = "", color = :black, lw = 2, marker = :circle, ms = 3,
        xscale = :log10, yscale = :log10,
        fillrange = 1.0, fillalpha = 0.22,
        fillcolor = last_r >= 1 ? WIN : LOSE,
        xlabel = "Matrix size n",
        ylabel = "NumPy time ÷ MatrixCpp time",
        title = @sprintf("speedup — above 1 = MatrixCpp wins (%.2f× at n=%d)",
                         last_r, Int(n[end])),
        legend = false, grid = true)
    hline!(p2, [1.0]; color = :gray30, ls = :dash, lw = 1.2, label = "")

    plot(p1, p2; layout = (1, 2), size = (1250, 500), margin = 6Plots.mm)
end

function summary_figure(dtype, pairs)
    rows = [(op, nt[end] / ct[end], Int(n[end]))
            for ((dt, op), (n, ct, nt)) in pairs if dt == dtype]
    isempty(rows) && return nothing
    sort!(rows, by = r -> r[2])

    labels = ["$(r[1])  (n=$(r[3]))" for r in rows]
    vals   = [r[2] for r in rows]
    colors = [v >= 1 ? WIN : LOSE for v in vals]

    p = bar(labels, vals;
        orientation = :horizontal, color = colors, alpha = 0.85,
        xscale = :log10, legend = false,
        xlabel = "NumPy time ÷ MatrixCpp time  (log scale, >1 = MatrixCpp faster)",
        title  = "MatrixCpp vs NumPy — $dtype, at the largest measured size",
        size   = (1150, max(420, 34 * length(rows) + 260)),
        left_margin = 30Plots.mm, right_margin = 12Plots.mm,
        bottom_margin = 10Plots.mm, top_margin = 6Plots.mm,
        yticks = :all, tickfontsize = 7, grid = true)
    vline!(p, [1.0]; color = :gray20, ls = :dash, lw = 1.4, label = "")
    return p
end

# Measured complexity exponent for each side, against the textbook value.
function scaling_figure(dtype, pairs)
    rows = [(op, empirical_slope(n, ct), empirical_slope(n, nt),
             get(EXPECTED, op, NaN))
            for ((dt, op), (n, ct, nt)) in pairs if dt == dtype]
    filter!(r -> isfinite(r[2]) && isfinite(r[3]), rows)
    isempty(rows) && return nothing
    sort!(rows, by = r -> r[1])

    ops = [r[1] for r in rows]
    # Built from the first series rather than from an empty plot(): seeding a
    # figure with attributes but no data leaves Plots without the extents it
    # needs for a categorical axis, and the layout assertion fails at savefig
    # time — taking the GR session down with it for every later figure too.
    p = scatter([r[2] for r in rows], ops;
        label = "MatrixCpp", color = CPP_COLOR, ms = 7, marker = :circle,
        size = (1150, max(420, 34 * length(rows) + 260)), legend = :bottomright,
        xlabel = "measured exponent k  in  time ∝ n^k",
        title  = "Scaling exponent — $dtype",
        left_margin = 30Plots.mm, right_margin = 12Plots.mm,
        bottom_margin = 10Plots.mm, top_margin = 6Plots.mm,
        yticks = :all, tickfontsize = 7, grid = true)
    scatter!(p, [r[3] for r in rows], ops;
        label = "NumPy", color = NPY_COLOR, ms = 7, marker = :rect)
    scatter!(p, [r[4] for r in rows], ops;
        label = "textbook", color = :gray50, ms = 8, marker = :diamond)
    return p
end

# ════════════════════════════════════════════════════════════════════════════
function main()
    pairs = collect_pairs()
    if isempty(pairs)
        println("No matched bench/cpp_*.csv + bench/numpy_*.csv pairs found.")
        println("Run  ./Running_time  then  python3 NumpyRunningtime.py  first.")
        return
    end

    # savefig is where a Plots layout error actually surfaces, so each figure is
    # saved independently — one that fails should not take the rest of the run
    # down with it.
    function save_safely(fig, path)
        fig === nothing && return false
        try
            savefig(fig, path)
            Plots.closeall()
            return true
        catch err
            println("  WARN  could not save $path: ",
                    first(sprint(showerror, err), 160))
            return false
        end
    end

    mkpath(joinpath(OUT, "ops"))

    # The overview figures are drawn FIRST and each is closed after saving. GR
    # keeps global state across figures, and a long run of them leaves the
    # backend unable to lay out the tall categorical charts that follow.
    for dtype in ("real", "complex")
        save_safely(summary_figure(dtype, pairs),
                    joinpath(OUT, "summary_$(dtype).png")) &&
            println("  summary  -> $(joinpath(OUT, "summary_$(dtype).png"))")
        save_safely(scaling_figure(dtype, pairs),
                    joinpath(OUT, "scaling_$(dtype).png")) &&
            println("  scaling  -> $(joinpath(OUT, "scaling_$(dtype).png"))")
    end

    saved = 0
    for ((dtype, op), (n, ct, nt)) in sort(collect(pairs), by = first)
        save_safely(per_op_figure(dtype, op, n, ct, nt),
                    joinpath(OUT, "ops", "$(dtype)_$(op).png")) && (saved += 1)
    end
    println("  $saved/$(length(pairs)) per-op figures -> $(joinpath(OUT, "ops"))/")

    # Console table, matching the one NumpyRunningtime.py prints.
    println("\n" * "="^74)
    @printf("%-9s %-17s %6s %12s %12s %9s\n",
            "dtype", "operation", "n", "MatrixCpp", "NumPy", "speedup")
    println("="^74)
    for ((dtype, op), (n, ct, nt)) in sort(collect(pairs), by = first)
        r = nt[end] / ct[end]
        mark = (0.9 <= r <= 1.1) ? "" : (r < 1 ? "  <<" : "  >>")
        @printf("%-9s %-17s %6d %12.6f %12.6f %8.2fx%s\n",
                dtype, op, Int(n[end]), ct[end], nt[end], r, mark)
    end
    println("="^74)
    println(">> MatrixCpp faster    << NumPy faster")
end

main()
