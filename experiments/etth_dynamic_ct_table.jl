# ETTh1/ETTh2 x {96, 192, 336, 720}: the linear-66d 2-level log-precision head, fit with
# **native natural-gradient message passing**, against the paper's Dynamic ensemble,
# scored with the paper's own MSE/NLL confidence-interval recipe.
#
# The arm is defined once in experiments/dynamic_deep_kernel_precision.jl -- same model,
# same priors, same NGMP damping -- and this script only sweeps cells and formats tables.
#
# Usage (one cell per process -- see "staging" below):
#   julia --project=. experiments/etth_dynamic_ct_table.jl prepare ETTh1 192
#   julia --project=. experiments/etth_dynamic_ct_table.jl fit     ETTh1 192
#   julia --project=. experiments/etth_dynamic_ct_table.jl table
#
# Staging: `prepare` (expert + VAE evaluation) is separated from `fit` (variational
# inference) so a slow cache never blocks a fit, and every cell runs as its own process --
# multi-cell shell loops have been truncated at ~10 minutes in this project before,
# silently dropping later configurations.
#
# Env knobs: CT_ITERATIONS (default 20), CT_N_OBS (0 = full validation split),
#            CT_REFERENCE_REPO, CT_RESULTS_DIR.

using Distributions
using JLD2
using Printf
using Serialization
using Statistics
using YAML

using ProbabilisticEnsembling
using RxInfer
using SurrogateModelling

# The models, priors, feature maps, `fit_arm`, `fit_named_arm` and `predict_arm` all live
# in the 2x2 script, whose driver is guarded -- including it defines them without running
# the ETTh1 h96 comparison. Deliberately NOT including dynamic_vmp_vs_ngmp_etth2.jl as
# well: it defines its own `predictive_metrics(mean, std, target)` with the same positional
# arity, which would silently overwrite this one.
include(joinpath(@__DIR__, "dynamic_deep_kernel_precision.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATASETS = ("ETTh1", "ETTh2")
const HORIZONS = (96, 192, 336, 720)
const ARM = :linear2_ngmp           # linear 66d, 2 levels, native NGMP
const ITERATIONS = parse(Int, get(ENV, "CT_ITERATIONS", "20"))
const N_OBS_OVERRIDE = parse(Int, get(ENV, "CT_N_OBS", "0"))
const RESULTS_DIR = get(ENV, "CT_RESULTS_DIR", joinpath(ROOT, "results", "etth_ct_table"))
const REFERENCE_REPO = get(
    ENV, "CT_REFERENCE_REPO",
    joinpath(homedir(), "repos", "probabilistic_ensemble_forecasting"),
)

# scripts/build_comparision_table.jl:273 and scripts/gp_ssm_ci_vs_dynamic.jl:65 both use
# this z value, and the paper's CI is Z * std(per-test-point term) / sqrt(n).
const Z95 = 1.959963984540054

# The paper's fixed test size for both ETT sets (build_comparision_table.jl:23), used ONLY
# to reproduce its published interval widths for the reference column.
const REFERENCE_N = 2881

# ---------------------------------------------------------------------------
# Cells and caches
# ---------------------------------------------------------------------------

session_path(dataset, horizon) =
    joinpath(ROOT, "sessions", "dynamic", "vae", "dynamic_$(dataset)_$(horizon).yaml")

"""
    cache_candidates(dataset, horizon)

Where this cell's prepared arrays may already live. The canonical location is
`cache/`, but two cells were built earlier under different names and are reused as-is
rather than recomputed: ETTh1 h96 sits next to the notebook that created it, and
ETTh2 h96 already uses the canonical path.
"""
function cache_candidates(dataset, horizon)
    canonical = joinpath(ROOT, "cache",
        "dynamic_$(lowercase(dataset))_h$(horizon)_cache.jld2")
    legacy = joinpath(ROOT, "notebooks", "vmp_vs_ngmp",
        "dynamic_$(lowercase(dataset))_h$(horizon)_cache.jld2")
    return (canonical, legacy)
end

"""
    verify_prerequisites(dataset, horizon)

Check the session file and every artifact it names. Generalized from
`dynamic_vmp_vs_ngmp_etth2.jl:264`, which hardcoded both the ETTh2 VAE path and the
`etth2` cache filename and so could not be pointed at ETTh1.
"""
function verify_prerequisites(dataset, horizon)
    dataset in DATASETS || throw(ArgumentError("dataset must be one of $DATASETS"))
    horizon in HORIZONS || throw(ArgumentError("horizon must be one of $HORIZONS"))
    session = session_path(dataset, horizon)
    isfile(session) || error("session file not found: $session")
    raw = YAML.load_file(session)
    params = raw["params"]
    params["dataset"] == dataset ||
        error("session $session configures $(params["dataset"]), expected $dataset")
    Int(params["horizon"]) == horizon ||
        error("session $session configures h$(params["horizon"]), expected h$horizon")

    needed = String[params["dataset_path"]]
    append!(needed, String.(params["experts"]))
    push!(needed, "models/$(dataset)_s96_VAE_enzyme.jld2")
    absent = filter(p -> !isfile(joinpath(ROOT, p)), needed)
    isempty(absent) || error("missing prerequisites for $dataset h$horizon:\n" *
        join(absent, "\n"))
    return raw
end

"""
    prepare_cell(dataset, horizon; rebuild = false)

Load the prepared arrays for one cell, building them with
`ProbabilisticEnsembling.before_rxinfer` if absent. PE resolves `data/` and `models/`
relative to the working directory, hence the `cd(ROOT)`.
"""
function prepare_cell(dataset, horizon; rebuild::Bool = false)
    raw = verify_prerequisites(dataset, horizon)
    canonical, _ = cache_candidates(dataset, horizon)
    existing = rebuild ? nothing :
        findfirst(isfile, collect(cache_candidates(dataset, horizon)))
    if isnothing(existing)
        @printf("building cache for %s h%d (expert + VAE evaluation)\n", dataset, horizon)
        started = time()
        spec = ProbabilisticEnsembling._parse_spec(raw)
        prepared = cd(() -> ProbabilisticEnsembling.before_rxinfer(spec), ROOT)
        mkpath(dirname(canonical))
        jldsave(canonical;
            y_val = prepared[1], y_test = prepared[2],
            predictions_val = prepared[3], predictions_test = prepared[4],
            features_val = prepared[5], features_test = prepared[6])
        @printf("  built in %.0f s -> %s\n", time() - started, canonical)
        return JLD2.load(canonical), canonical
    end
    path = cache_candidates(dataset, horizon)[existing]
    return JLD2.load(path), path
end

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

"""
    ci_summary(terms)

The paper's interval: mean of the per-test-point terms with half-width
`Z95 * std / sqrt(n)`. Note this uses the ACTUAL n for the cell, matching
`dynamic_vmp_vs_ngmp_etth2.jl:220`; the paper's own table instead pins
`TEST_SIZE_BY_DATASET = 2881` for both ETT sets while its split yields ~3446, which makes
its intervals ~9% wider than the data supports. The tables below state which n they used.
"""
ci_summary(terms) = (mean = mean(terms),
                     half_width = Z95 * std(terms) / sqrt(length(terms)))

"""
    score_cell(means, sigmas, targets)

MSE and NLL with intervals, on the standardized OT target.

`nll` here is a genuine **negative** log predictive density -- positive, lower is better.
The reference repo stores a field named `nll` that is actually `mean(logpdf(...))`
(`src/model_zoo/shared_pipeline.jl:87`), i.e. a log density held negative, and its table
then declares `larger_is_better = false` on that stored sign. Reporting `-logpdf` here
removes the ambiguity rather than inheriting it. `std` is unchanged by the sign flip, so
the interval carries over untouched.
"""
function score_cell(means, sigmas, targets)
    squared_error = (means .- targets) .^ 2
    nll_terms = [-logpdf(Normal(means[j], sigmas[j]), targets[j])
                 for j in eachindex(targets)]
    covered = (targets .>= means .- Z95 .* sigmas) .& (targets .<= means .+ Z95 .* sigmas)
    mse, nll = ci_summary(squared_error), ci_summary(nll_terms)
    return (n = length(targets),
            mse = mse.mean, mse_ci = mse.half_width,
            nll = nll.mean, nll_ci = nll.half_width,
            mae = mean(abs.(means .- targets)), cov95 = mean(covered))
end

# ---------------------------------------------------------------------------
# One cell, end to end
# ---------------------------------------------------------------------------

result_path(dataset, horizon) =
    joinpath(RESULTS_DIR, "$(dataset)_h$(horizon)_$(ARM).jls")

function fit_cell(dataset, horizon)
    cache, cache_path = prepare_cell(dataset, horizon)
    n_available = length(cache["y_val"])
    n_obs = N_OBS_OVERRIDE > 0 ? min(N_OBS_OVERRIDE, n_available) : n_available
    config = merge(DDK_CONFIG, (; n_obs, iterations = ITERATIONS))

    println("\n", "="^100)
    @printf("%s h%d -- %s, %d iterations\n", dataset, horizon, ARM, ITERATIONS)
    @printf("cache %s\n", cache_path)
    println("="^100)

    train_features = cache["features_val"][1:n_obs]
    targets = cache["y_val"][1:n_obs]
    predictions = cache["predictions_val"][:, 1:n_obs]

    started = time()
    arm = fit_arm(train_features, targets, predictions, config)
    means, sigmas = predict_arm(
        arm, cache["features_test"], cache["predictions_test"], config,
    )
    metrics = score_cell(means, sigmas, cache["y_test"])
    elapsed = time() - started

    @printf("\n  %d features, %d experts, %d train, %d test\n",
        arm.dimension, size(predictions, 1), n_obs, metrics.n)
    @printf("  MSE %.5f +- %.5f | NLL %.5f +- %.5f | MAE %.4f | cov95 %.3f | %.0f s\n",
        metrics.mse, metrics.mse_ci, metrics.nll, metrics.nll_ci,
        metrics.mae, metrics.cov95, elapsed)

    mkpath(RESULTS_DIR)
    record = (; dataset, horizon, arm = ARM, iterations = ITERATIONS, n_obs,
              dimension = arm.dimension, n_experts = size(predictions, 1),
              metrics, means, sigmas, elapsed)
    serialize(result_path(dataset, horizon), record)
    @printf("  saved %s\n", result_path(dataset, horizon))
    return record
end

# ---------------------------------------------------------------------------
# Reference numbers
# ---------------------------------------------------------------------------

"""
    reference_cell(dataset, horizon)

The published Dynamic ensemble metrics, read live from the reference repo rather than
transcribed -- `scripts/gp_ssm_ci_vs_dynamic.jl:33` hardcodes these and in doing so
sign-flipped `nll` without saying so. Returns NLL in the same positive,
lower-is-better convention as `score_cell`.
"""
function reference_cell(dataset, horizon)
    directory = joinpath(REFERENCE_REPO, "paper", "results_vae_std", "dynamic")
    isdir(directory) || return nothing
    prefix = "$(dataset)_h$(horizon)_OT_dynamic"
    candidates = filter(f -> startswith(f, prefix), readdir(directory))
    isempty(candidates) && return nothing
    metrics = JLD2.load(joinpath(directory, first(candidates)))["ensemble_metrics"]
    # `nll` is stored as a mean log density (negative); flip it to a true NLL.
    return (mse = metrics.mse, mse_ci = Z95 * metrics.mse_std / sqrt(REFERENCE_N),
            nll = -metrics.nll, nll_ci = Z95 * metrics.nll_std / sqrt(REFERENCE_N),
            cov95 = metrics.ci95_target_overlap)
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

load_cell(dataset, horizon) =
    isfile(result_path(dataset, horizon)) ?
        deserialize(result_path(dataset, horizon)) : nothing

function console_table(io = stdout)
    println(io, "\n", "="^108)
    println(io, "Dynamic ensemble on standardized OT: NGMP arm (linear 66d, 2 levels) vs " *
                "the paper's Dynamic row")
    println(io, "="^108)
    @printf(io, "%-6s %5s | %-22s %-22s | %-22s %-22s\n",
        "data", "h", "CT MSE", "Dyn MSE", "CT NLL", "Dyn NLL")
    println(io, "-"^108)
    for dataset in DATASETS
        for horizon in HORIZONS
            cell, reference = load_cell(dataset, horizon), reference_cell(dataset, horizon)
            if isnothing(cell)
                @printf(io, "%-6s %5d | %s\n", dataset, horizon, "not fitted yet")
                continue
            end
            m = cell.metrics
            fmt(value, half) = @sprintf("%7.5f +- %-7.5f", value, half)
            @printf(io, "%-6s %5d | %-22s %-22s | %-22s %-22s\n",
                dataset, horizon,
                fmt(m.mse, m.mse_ci),
                isnothing(reference) ? "-" : fmt(reference.mse, reference.mse_ci),
                fmt(m.nll, m.nll_ci),
                isnothing(reference) ? "-" : fmt(reference.nll, reference.nll_ci))
        end
    end
    println(io, "-"^108)
    println(io, "NLL is a NEGATIVE log predictive density: positive, lower is better.")
    println(io, "CI half-width = $(round(Z95; digits = 3))*std(per-test-point term)/sqrt(n).")
    println(io, "CT uses the actual n per cell; the Dyn column uses the paper's fixed " *
                "n = $REFERENCE_N, reproducing its published widths.")
    println(io, "Intervals are UNPAIRED per-model intervals. Overlap means the interval " *
                "does not resolve the\ndifference, not that there is none -- a paired " *
                "per-point test on identical test points is far\nmore powerful.")
    return nothing
end

"""LaTeX in the shape of `build_comparision_table.jl`: `\$x \\pm y\$`, better value bolded."""
function latex_table(io = stdout)
    cell_tex(value, half, better) = begin
        body = @sprintf("%.3f \\pm %.3f", value, half)
        better ? "\$\\mathbf{$body}\$" : "\$$body\$"
    end
    println(io, "\\begin{tabular}{llcccc}")
    println(io, "\\toprule")
    println(io, "Data & \$H\$ & \\multicolumn{2}{c}{MSE} & \\multicolumn{2}{c}{NLL} \\\\")
    println(io, "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    println(io, " & & CT & Dyn. & CT & Dyn. \\\\")
    println(io, "\\midrule")
    for dataset in DATASETS
        for horizon in HORIZONS
            cell, reference = load_cell(dataset, horizon), reference_cell(dataset, horizon)
            isnothing(cell) && continue
            m = cell.metrics
            if isnothing(reference)
                @printf(io, "%s & %d & %s & -- & %s & -- \\\\\n", dataset, horizon,
                    cell_tex(m.mse, m.mse_ci, false), cell_tex(m.nll, m.nll_ci, false))
                continue
            end
            @printf(io, "%s & %d & %s & %s & %s & %s \\\\\n", dataset, horizon,
                cell_tex(m.mse, m.mse_ci, m.mse <= reference.mse),
                cell_tex(reference.mse, reference.mse_ci, reference.mse < m.mse),
                cell_tex(m.nll, m.nll_ci, m.nll <= reference.nll),
                cell_tex(reference.nll, reference.nll_ci, reference.nll < m.nll))
        end
        dataset == last(DATASETS) || println(io, "\\midrule")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "% NLL = negative log predictive density (lower is better).")
    println(io, "% CI half-width = $(round(Z95; digits = 3)) std/sqrt(n); " *
                "CT uses actual n, Dyn. uses the paper's n = $REFERENCE_N.")
    println(io, "% Bold marks the better point estimate; intervals are unpaired and " *
                "frequently overlap.")
    return nothing
end

function write_tables()
    mkpath(RESULTS_DIR)
    console_table()
    console_path = joinpath(RESULTS_DIR, "table.txt")
    open(io -> console_table(io), console_path, "w")
    latex_path = joinpath(RESULTS_DIR, "table.tex")
    open(io -> latex_table(io), latex_path, "w")
    println("\nwrote $console_path\n      $latex_path")
    return nothing
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main(args)
    isempty(args) && error("usage: prepare|fit <dataset> <horizon> | table")
    mode = args[1]
    if mode == "table"
        write_tables()
    elseif mode in ("prepare", "fit")
        length(args) == 3 || error("usage: $mode <dataset> <horizon>")
        dataset, horizon = args[2], parse(Int, args[3])
        if mode == "prepare"
            _, path = prepare_cell(dataset, horizon)
            println("cache ready: $path")
        else
            fit_cell(dataset, horizon)
        end
    else
        error("unknown mode $mode; expected prepare, fit or table")
    end
end

main(ARGS)
