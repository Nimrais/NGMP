#!/usr/bin/env julia

using CSV
using DataFrames
using Printf
using Statistics

const ROOT = dirname(@__DIR__)
const SOURCE_STEM =
    "uci_deep_kernel_direct_final_decreasing_matern1000_depth1_5_ablation"
const SOURCE_RUNS = joinpath(ROOT, "results", "$SOURCE_STEM.csv")
const SOURCE_SUMMARY = joinpath(ROOT, "results", "$(SOURCE_STEM)_summary.csv")
const OUTPUT = joinpath(ROOT, "paper_materials", "ngmp_uci_depth_ablation")
const MANIFEST_SOURCE =
    joinpath(ROOT, "paper_materials", "ngmp_uci", "split_manifest.jld2")
const DATASET_ORDER = ["concrete", "energy", "housing", "power", "wine", "yacht"]
const DATASET_NAMES = Dict(
    "concrete" => "Concrete",
    "energy" => "Energy",
    "housing" => "Boston",
    "power" => "Power",
    "wine" => "Wine",
    "yacht" => "Yacht",
)

ci95_halfwidth(values) = 1.96 * std(values) / sqrt(length(values))

function summarized_rows(runs)
    rows = NamedTuple[]
    for dataset in DATASET_ORDER, depth in 1:5
        selected = filter(
            row -> row.dataset == dataset && row.depth == depth,
            runs,
        )
        nrow(selected) == 20 || error(
            "expected 20 successful rows for $dataset depth $depth",
        )
        all(selected.status .== "ok") || error(
            "non-successful result for $dataset depth $depth",
        )
        n = nrow(selected)
        push!(rows, (;
            dataset,
            dataset_name = DATASET_NAMES[dataset],
            depth,
            likelihood = depth == 1 ? "homoscedastic" : "heteroscedastic",
            optimizer = first(selected.optimizer),
            beta = first(selected.beta),
            n,
            lpd_standardized_mean = mean(selected.logpdf_standardized),
            lpd_standardized_std = std(selected.logpdf_standardized),
            lpd_standardized_se = std(selected.logpdf_standardized) / sqrt(n),
            lpd_standardized_ci95 = ci95_halfwidth(selected.logpdf_standardized),
            lpd_original_mean = mean(selected.logpdf),
            lpd_original_std = std(selected.logpdf),
            lpd_original_se = std(selected.logpdf) / sqrt(n),
            lpd_original_ci95 = ci95_halfwidth(selected.logpdf),
            rmse_original_mean = mean(selected.rmse),
            rmse_original_std = std(selected.rmse),
            rmse_original_se = std(selected.rmse) / sqrt(n),
            rmse_original_ci95 = ci95_halfwidth(selected.rmse),
            paper_dvi = first(selected.paper_dvi),
        ))
    end
    return DataFrame(rows)
end

metric(mean, ci95) = @sprintf("%.4f ± %.4f", mean, ci95)
tex_metric(mean, ci95) = @sprintf("%.4f \$\\pm\$ %.4f", mean, ci95)

function write_markdown(path, summary)
    open(path, "w") do io
        println(io, "| Dataset | Depth | Runs | LPD (original) | RMSE | LPD (standardized) |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for row in eachrow(summary)
            println(
                io,
                "| $(row.dataset_name) | $(row.depth) | $(row.n) | " *
                "$(metric(row.lpd_original_mean, row.lpd_original_ci95)) | " *
                "$(metric(row.rmse_original_mean, row.rmse_original_ci95)) | " *
                "$(metric(row.lpd_standardized_mean, row.lpd_standardized_ci95)) |",
            )
        end
        println(io)
        println(io, "Values are means ± 95% confidence-interval half-widths across")
        println(io, "20 repeated-holdout splits, computed as `1.96 × standard error`.")
    end
end

function write_latex(path, summary)
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrr}")
        println(io, "\\toprule")
        println(io, "Dataset & Depth & Runs & LPD (original) & RMSE & LPD (standardized) \\\\")
        println(io, "\\midrule")
        for row in eachrow(summary)
            println(
                io,
                "$(row.dataset_name) & $(row.depth) & $(row.n) & " *
                "$(tex_metric(row.lpd_original_mean, row.lpd_original_ci95)) & " *
                "$(tex_metric(row.rmse_original_mean, row.rmse_original_ci95)) & " *
                "$(tex_metric(row.lpd_standardized_mean, row.lpd_standardized_ci95)) \\\\",
            )
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "% Values are means plus/minus 95% CI half-widths (1.96 times SE).")
    end
end

function write_config(path)
    open(path, "w") do io
        print(io, """
method = "NGMP direct deep kernel depth ablation"
split_protocol = "repeated-holdout-v1"
split_seed = 20260726
n_splits = 20
test_fraction = 0.1
datasets = ["yacht", "housing", "energy", "concrete", "wine", "power"]

preprocessing = "multiscale_matern32_linear"
feature_dimension = 1000
depths = [1, 2, 3, 4, 5]
lengthscale_schedule = "decreasing"
base_lengthscale = 1.5
lengthscale_factor = 0.7071067811865476

depth1_optimizer = "damped"
deep_optimizer = "vector_transport"
alpha = 0.6
beta = 0.8
prior_gain = 1.0
residual_variance_init_min = 1.0e-3
residual_variance_init_max = 1.0e3
""")
    end
end

function write_readme(path)
    open(path, "w") do io
        print(io, """
# NGMP UCI depth-ablation paper artifacts

Final 1,000-feature NGMP configuration evaluated at hierarchy depths 1--5 on
the shared BBB-aligned `repeated-holdout-v1` splits.

- `runs.csv`: all 600 dataset/depth/split results.
- `benchmark_summary.csv`: summary emitted by the benchmark script.
- `summary.csv`: paper summary with standard deviation, standard error, and
  95% CI half-width for every metric.
- `table.md` and `table.tex`: one row per dataset and depth, reporting
  mean ± 95% CI half-width.
- `split_manifest.jld2`: exact shared split specifications.
- `config.toml`: fixed configuration and the single ablated factor.

Depth 1 is the homoscedastic damped baseline. Depths 2--5 use vector transport
with beta 0.8 and add progressively more precision-hierarchy levels. Mean
features remain fixed; this experiment measures uncertainty-hierarchy depth.
""")
    end
end

function main()
    isfile(SOURCE_RUNS) || error("missing $SOURCE_RUNS")
    isfile(SOURCE_SUMMARY) || error("missing $SOURCE_SUMMARY")
    isfile(MANIFEST_SOURCE) || error("missing $MANIFEST_SOURCE")
    mkpath(OUTPUT)
    runs = CSV.read(SOURCE_RUNS, DataFrame)
    nrow(runs) == 600 || error("expected 600 per-split rows")
    all(runs.status .== "ok") || error("depth ablation contains failed rows")
    summary = summarized_rows(runs)
    CSV.write(joinpath(OUTPUT, "summary.csv"), summary)
    cp(SOURCE_RUNS, joinpath(OUTPUT, "runs.csv"); force = true)
    cp(SOURCE_SUMMARY, joinpath(OUTPUT, "benchmark_summary.csv"); force = true)
    cp(MANIFEST_SOURCE, joinpath(OUTPUT, "split_manifest.jld2"); force = true)
    write_markdown(joinpath(OUTPUT, "table.md"), summary)
    write_latex(joinpath(OUTPUT, "table.tex"), summary)
    write_config(joinpath(OUTPUT, "config.toml"))
    write_readme(joinpath(OUTPUT, "README.md"))
    println("Packaged 600 runs and 30 dataset-depth summaries in $OUTPUT")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
