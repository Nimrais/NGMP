#!/usr/bin/env julia

using CSV
using DataFrames
using JLD2
using Printf
using TOML

struct ResultSource
    method::String
    likelihood::String
    propagation::Union{Nothing,String}
    directory::String
    success_status::String
    hidden_layers::Union{Nothing,Int}
end

ResultSource(method, likelihood, propagation, directory, success_status = "success";
    hidden_layers = nothing) =
    ResultSource(method, likelihood, propagation, directory, success_status, hidden_layers)

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
        "dDVI-2HL-Hetero",
        "heteroscedastic",
        "diagonal",
        joinpath(PAPER_ROOT, "dvi_uci",
            "wu2019_repeated_holdout_v1_20splits_heteroscedastic_2hidden50_v1_20261024",
            "ddvi"),
        hidden_layers = 2,
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
    ResultSource(
        "NGMP-Hetero",
        "heteroscedastic",
        nothing,
        joinpath(PAPER_ROOT, "ngmp_uci"),
        "ok",
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

const DVI_BIG_ROOT = joinpath(PAPER_ROOT, "dvi_uci",
    "wu2019_repeated_holdout_v1_20splits_full_heteroscedastic_2hidden50_v1_20261025")
const HOMO_BIG_ROOT = joinpath(PAPER_ROOT, "dvi_uci",
    "wu2019_repeated_holdout_v1_20splits_homoscedastic_2hidden50_budget50k_v1_20261025")

function table_sources(; include_dvi_big = false, dvi_big_directory = joinpath(DVI_BIG_ROOT, "dvi"),
    include_ddvi_big_homo = false, include_dvi_big_homo = false,
    ddvi_big_homo_directory = joinpath(HOMO_BIG_ROOT, "ddvi"),
    dvi_big_homo_directory = joinpath(HOMO_BIG_ROOT, "dvi"))
    sources = copy(SOURCES)
    if include_dvi_big
        position = findfirst(source -> source.method == "DVI-Hetero", sources)
        insert!(sources, position + 1, ResultSource("DVI-2HL-Hetero", "heteroscedastic",
            "full", dvi_big_directory; hidden_layers = 2))
    end
    for (enabled, small, label, propagation, directory) in (
        (include_ddvi_big_homo, "dDVI-Homo", "dDVI-2HL-Homo", "diagonal", ddvi_big_homo_directory),
        (include_dvi_big_homo, "DVI-Homo", "DVI-2HL-Homo", "full", dvi_big_homo_directory))
        enabled || continue
        position = findfirst(source -> source.method == small, sources)
        insert!(sources, position + 1, ResultSource(label, "homoscedastic",
            propagation, directory; hidden_layers = 2))
    end
    return sources
end

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
        validate_source_architecture(source, runs, summary)
        selected_runs = if hasproperty(runs, :likelihood)
            filter(row -> string(row.likelihood) == source.likelihood, runs)
        else
            runs
        end
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
        all(string.(selected_runs.status) .== source.success_status) || error(
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
            all(isfinite, (row.lpd_original_mean, row.lpd_original_se,
                row.rmse_original_mean, row.rmse_original_se)) ||
                error("$(source.method) contains non-finite summary metrics")
            key = (string(row.dataset), source.method)
            haskey(results, key) && error("duplicate table row for $key")
            results[key] = (
                n = Int(row.n),
                lpd_mean = Float64(row.lpd_original_mean),
                lpd_ci95 = 1.96 * Float64(row.lpd_original_se),
                rmse_mean = Float64(row.rmse_original_mean),
                rmse_ci95 = 1.96 * Float64(row.rmse_original_se),
            )
        end
    end

    length(results) == length(DATASET_ORDER) * length(sources) || error(
        "final table is incomplete",
    )
    return results
end

function validate_source_architecture(source, runs, summary)
    source.hidden_layers === nothing && return nothing
    config = TOML.parsefile(joinpath(source.directory, "config.toml"))
    get(config, "hidden_layers", nothing) == source.hidden_layers ||
        error("$(source.method) has the wrong hidden-layer count")
    get(config, "hidden_units", nothing) == 50 ||
        error("$(source.method) must have 50 units per hidden layer")
    get(config, "propagation", nothing) == source.propagation ||
        error("$(source.method) has the wrong propagation setting")
    get(config, "likelihoods", nothing) == [source.likelihood] ||
        error("$(source.method) has the wrong likelihood")
    for table in (runs, summary)
        all(name -> name in propertynames(table),
            (:hidden_layers, :hidden_units, :model_coefficients)) ||
            error("$(source.method) is missing architecture metadata")
        all(table.hidden_layers .== source.hidden_layers) &&
            all(table.hidden_units .== 50) ||
            error("$(source.method) contains mixed architectures")
        dimensions = Dict("concrete" => 8, "energy" => 8, "housing" => 13,
            "power" => 4, "wine" => 11, "yacht" => 6)
        output_units = source.likelihood == "homoscedastic" ? 1 : 2
        all(row.model_coefficients == 50 * dimensions[string(row.dataset)] + 2600 + 51 * output_units
            for row in eachrow(table)) || error("$(source.method) has incorrect coefficient counts")
    end
    return nothing
end

function render_markdown(io, sources, results)
    println(io, "| Dataset | Method | Runs | LPD (original, 95% CI) | RMSE (95% CI) |")
    println(io, "|---|---|---:|---:|---:|")
    for (dataset, dataset_name) in DATASET_ORDER
        rows = [results[(dataset, source.method)] for source in sources]
        best_lpd = maximum(row.lpd_mean for row in rows)
        best_rmse = minimum(row.rmse_mean for row in rows)
        for source in sources
            row = results[(dataset, source.method)]
            lpd = @sprintf("%.4f ± %.4f", row.lpd_mean, row.lpd_ci95)
            rmse = @sprintf("%.4f ± %.4f", row.rmse_mean, row.rmse_ci95)
            row.lpd_mean + row.lpd_ci95 >= best_lpd && (lpd = "**$lpd**")
            row.rmse_mean - row.rmse_ci95 <= best_rmse && (rmse = "**$rmse**")
            println(
                io,
                "| $dataset_name | $(source.method) | $(row.n) | $lpd | $rmse |",
            )
        end
    end
end

function main(args = ARGS)
    allowed = ["--include-dvi-big", "--include-ddvi-big-homo", "--include-dvi-big-homo"]
    (all(arg -> arg in allowed, args) && length(unique(args)) == length(args)) || error(
        "usage: julia --project=. " *
        "paper_materials/make_final_baseline_table.jl [--include-dvi-big] " *
        "[--include-ddvi-big-homo] [--include-dvi-big-homo]",
    )
    sources = table_sources(; include_dvi_big = "--include-dvi-big" in args,
        include_ddvi_big_homo = "--include-ddvi-big-homo" in args,
        include_dvi_big_homo = "--include-dvi-big-homo" in args)
    validate_split_manifests(sources)
    results = load_results(sources)
    render_markdown(stdout, sources, results)
    println(
        stderr,
        "Validated $(length(sources)) method variants on " *
        "$(length(DATASET_ORDER)) datasets × 20 exact paired splits.",
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
