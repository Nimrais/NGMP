#!/usr/bin/env julia

using CSV
using DataFrames
using JLD2
using Printf

struct ResultSource
    method::String
    likelihood::String
    propagation::Union{Nothing,String}
    directory::String
end

const PAPER_ROOT = @__DIR__
const SOURCES = [
    ResultSource(
        "BBB-Homo",
        "homoscedastic",
        nothing,
        joinpath(
            PAPER_ROOT,
            "bbb_uci",
            "blundell2015_repeated_holdout_v1_20splits_20260728",
        ),
    ),
    ResultSource(
        "BBB-Hetero",
        "heteroscedastic",
        nothing,
        joinpath(
            PAPER_ROOT,
            "bbb_uci",
            "blundell2015_repeated_holdout_v1_20splits_20260728",
        ),
    ),
    ResultSource(
        "dDVI-Homo",
        "homoscedastic",
        "diagonal",
        joinpath(
            PAPER_ROOT,
            "dvi_uci",
            "wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804",
            "ddvi",
        ),
    ),
    ResultSource(
        "dDVI-Hetero",
        "heteroscedastic",
        "diagonal",
        joinpath(
            PAPER_ROOT,
            "dvi_uci",
            "wu2019_repeated_holdout_v1_20splits_robust_v1_20260731",
            "ddvi",
        ),
    ),
    ResultSource(
        "DVI-Homo",
        "homoscedastic",
        "full",
        joinpath(
            PAPER_ROOT,
            "dvi_uci",
            "wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804",
            "dvi",
        ),
    ),
    ResultSource(
        "DVI-Hetero",
        "heteroscedastic",
        "full",
        joinpath(
            PAPER_ROOT,
            "dvi_uci",
            "wu2019_repeated_holdout_v1_20splits_robust_selection_optimized_v2_20260803",
            "dvi",
        ),
    ),
    ResultSource(
        "BPC-Homo",
        "homoscedastic",
        nothing,
        joinpath(
            PAPER_ROOT,
            "bpc_uci",
            "tschantz2025_repeated_holdout_v1_20splits_homoscedastic_20260805",
        ),
    ),
]

const DATASET_ORDER = [
    "concrete" => "Concrete",
    "energy" => "Energy",
    "housing" => "Boston",
    "power" => "Power",
    "wine" => "Wine",
    "yacht" => "Yacht",
]

function validate_split_manifests(sources)
    manifest_paths = unique(
        joinpath(source.directory, "split_manifest.jld2") for source in sources
    )
    all(isfile, manifest_paths) || error("one or more split manifests are missing")

    reference_path = first(manifest_paths)
    reference = JLD2.load(reference_path)
    scalar_keys = [
        "base_seed",
        "n_splits",
        "protocol_version",
        "schema_version",
        "test_fraction",
        "validation_fraction",
    ]

    for path in Iterators.drop(manifest_paths, 1)
        candidate = JLD2.load(path)
        for key in scalar_keys
            isequal(candidate[key], reference[key]) || error(
                "split-manifest field $key differs between $reference_path and $path",
            )
        end
        isequal(candidate["datasets"], reference["datasets"]) || error(
            "exact outer/inner split indices differ between $reference_path and $path",
        )
    end

    reference["n_splits"] == 20 || error("expected exactly 20 saved splits")
    return nothing
end

function load_results(sources)
    expected_datasets = Set(first.(DATASET_ORDER))
    results = Dict{Tuple{String,String},NamedTuple}()

    for source in sources
        runs_path = joinpath(source.directory, "runs.csv")
        summary_path = joinpath(source.directory, "summary.csv")
        isfile(runs_path) || error("missing runs table: $runs_path")
        isfile(summary_path) || error("missing summary: $summary_path")
        runs = CSV.read(runs_path, DataFrame)
        summary = CSV.read(summary_path, DataFrame)
        selected_runs = filter(
            row -> string(row.likelihood) == source.likelihood,
            runs,
        )
        selected = filter(
            row -> string(row.likelihood) == source.likelihood,
            summary,
        )

        if source.propagation !== nothing
            selected_runs = filter(
                row -> string(row.propagation) == source.propagation,
                selected_runs,
            )
        end
        nrow(selected_runs) == length(DATASET_ORDER) * 20 || error(
            "$(source.method) must contain exactly 120 run rows",
        )
        all(string.(selected_runs.status) .== "success") || error(
            "$(source.method) contains a non-success run row",
        )
        for dataset in expected_datasets
            dataset_runs = filter(
                row -> string(row.dataset) == dataset,
                selected_runs,
            )
            Set(Int.(dataset_runs.split)) == Set(1:20) || error(
                "$(source.method) does not contain splits 1:20 for $dataset",
            )
            nrow(dataset_runs) == 20 || error(
                "$(source.method) contains duplicate splits for $dataset",
            )
        end

        nrow(selected) == length(DATASET_ORDER) || error(
            "$(source.method) must have one summary row for every dataset",
        )
        Set(string.(selected.dataset)) == expected_datasets || error(
            "$(source.method) does not contain the canonical six datasets",
        )
        all(selected.n .== 20) || error(
            "$(source.method) is not aggregated over exactly 20 splits",
        )
        if source.propagation !== nothing
            all(string.(selected.propagation) .== source.propagation) || error(
                "$(source.method) has the wrong propagation setting",
            )
        end

        for row in eachrow(selected)
            key = (string(row.dataset), source.method)
            haskey(results, key) && error("duplicate table row for $key")
            results[key] = (
                n = Int(row.n),
                lpd_mean = Float64(row.lpd_original_mean),
                lpd_std = Float64(row.lpd_original_std),
                rmse_mean = Float64(row.rmse_original_mean),
                rmse_std = Float64(row.rmse_original_std),
            )
        end
    end

    length(results) == length(DATASET_ORDER) * length(sources) || error(
        "final table is incomplete",
    )
    return results
end

function render_markdown(io, sources, results)
    println(io, "| Dataset | Method | Runs | LPD (original) | RMSE |")
    println(io, "|---|---|---:|---:|---:|")
    for (dataset, dataset_name) in DATASET_ORDER
        for source in sources
            row = results[(dataset, source.method)]
            lpd = @sprintf("%.4f ± %.4f", row.lpd_mean, row.lpd_std)
            rmse = @sprintf("%.4f ± %.4f", row.rmse_mean, row.rmse_std)
            println(
                io,
                "| $dataset_name | $(source.method) | $(row.n) | $lpd | $rmse |",
            )
        end
    end
end

isempty(ARGS) || error(
    "usage: julia --project=. " *
    "paper_materials/make_final_baseline_table.jl",
)

validate_split_manifests(SOURCES)
results = load_results(SOURCES)
render_markdown(stdout, SOURCES, results)
println(
    stderr,
    "Validated $(length(SOURCES)) method variants on " *
    "$(length(DATASET_ORDER)) datasets × 20 exact paired splits.",
)
