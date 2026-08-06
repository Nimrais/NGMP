#!/usr/bin/env julia

# Determine whether the useful random-feature dimension grows with UCI dataset
# size. This uses the selected direct deep-kernel configuration and changes
# only dataset, depth, and random-feature dimension.
#
# Full run (6 datasets * 3 dimensions * 20 splits * 5 depths = 1,800 fits):
#   julia --project=. scripts/uci_deep_kernel_direct_feature_scaling_benchmark.jl
#
# Quick smoke test:
#   UCI_SPLITS=1 UCI_DATASETS=yacht,housing julia --project=. \
#     scripts/uci_deep_kernel_direct_feature_scaling_benchmark.jl

ENV["UCI_DATASETS"] = get(
    ENV,
    "UCI_DATASETS",
    "yacht,housing,energy,concrete,wine,power",
)
ENV["UCI_DEPTHS"] = get(ENV, "UCI_DEPTHS", "1,2,3,4,5")
ENV["UCI_DIRECT_FEATURE_DIMENSIONS"] = get(
    ENV,
    "UCI_DIRECT_FEATURE_DIMENSIONS",
    "400,600,800",
)
ENV["UCI_DIRECT_PREPROCESSING"] = get(
    ENV,
    "UCI_DIRECT_PREPROCESSING",
    "multiscale_matern32_linear",
)
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"

include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))
using CSV
using DataFrames

const FEATURE_SCALING_PRIOR_GAIN = parse(
    Float64,
    get(ENV, "UCI_FEATURE_SCALING_PRIOR_GAIN", "1.5"),
)
const FEATURE_SCALING_LENGTHSCALE = parse(
    Float64,
    get(ENV, "UCI_FEATURE_SCALING_LENGTHSCALE", "1.5"),
)
const FEATURE_SCALING_BETAS = parse.(
    Float64,
    split(get(ENV, "UCI_FEATURE_SCALING_BETAS", "0.5,0.6"), ','),
)

feature_scaling_optimizers(depth) = depth == 1 ?
    ((:damped, 0.0),) :
    [(:vector_transport, beta) for beta in FEATURE_SCALING_BETAS]

feature_scaling_beta_label() = join(
    (replace(@sprintf("%.2f", beta), "." => "_") for beta in FEATURE_SCALING_BETAS),
    "_",
)

function feature_scaling_prior(depth, Φ, targets)
    prior = direct_prior_parameters(depth, Φ, targets)
    gained_precisions = map(prior.level_prior_precisions) do precision
        gained = copy(precision)
        gained[1:(end - 1), 1:(end - 1)] ./=
            FEATURE_SCALING_PRIOR_GAIN^2
        gained
    end
    return merge(prior, (; level_prior_precisions = gained_precisions))
end

const FEATURE_SCALING_DATASET_SHAPES = Dict(
    :yacht => (rows = 308, inputs = 6),
    :housing => (rows = 506, inputs = 13),
    :energy => (rows = 768, inputs = 8),
    :concrete => (rows = 1_030, inputs = 8),
    :wine => (rows = 1_599, inputs = 11),
    :power => (rows = 9_568, inputs = 4),
)

function write_feature_dependencies(summary_path, output_path)
    summary = CSV.read(summary_path, DataFrame)
    successful = filter(
        row -> row.successful_splits > 0 && isfinite(row.mean_logpdf),
        summary,
    )
    selected = NamedTuple[]
    for group in groupby(successful, [:dataset, :depth])
        best = group[argmax(group.mean_logpdf), :]
        shape = FEATURE_SCALING_DATASET_SHAPES[Symbol(best.dataset)]
        push!(selected, (;
            dataset = best.dataset,
            observations = shape.rows,
            input_dimension = shape.inputs,
            depth = best.depth,
            best_feature_dimension = best.feature_dimension,
            mean_logpdf = best.mean_logpdf,
            mean_rmse = best.mean_rmse,
        ))
    end
    write_results(output_path, selected)

    dependency_rows = NamedTuple[]
    for depth_group in groupby(DataFrame(selected), :depth)
        length(depth_group.depth) >= 3 || continue
        features = Float64.(depth_group.best_feature_dimension)
        log_size = log2.(Float64.(depth_group.observations) ./ 308)
        shifted_inputs = Float64.(depth_group.input_dimension) .- 6
        design = hcat(ones(length(features)), log_size, shifted_inputs)
        coefficients = design \ features
        fitted = design * coefficients
        total_variation = sum(abs2, features .- mean(features))
        r_squared = total_variation <= eps() ? NaN :
            1 - sum(abs2, features .- fitted) / total_variation
        push!(dependency_rows, (;
            depth = first(depth_group.depth),
            intercept = coefficients[1],
            log2_rows_coefficient = coefficients[2],
            input_dimension_coefficient = coefficients[3],
            rows_correlation = cor(features, log_size),
            input_dimension_correlation = cor(features, shifted_inputs),
            r_squared,
        ))
    end
    dependency_path = replace(output_path, ".csv" => "_dependencies.csv")
    write_results(dependency_path, dependency_rows)

    if !isempty(dependency_rows)
        println("\nDescriptive equation fitted separately at each depth:")
        println("  M = a + b*log2(N/308) + c*(D-6)")
        for row in dependency_rows
            @printf(
                "  depth %d: M = %.1f %+.1f*log2(N/308) %+.1f*(D-6), R²=%.3f\n",
                row.depth,
                row.intercept,
                row.log2_rows_coefficient,
                row.input_dimension_coefficient,
                row.r_squared,
            )
        end
    end
    println("Best-feature CSV: $output_path")
    println("Dependency CSV: $dependency_path")
    return selected
end

function feature_scaling_main()
    beta_label = feature_scaling_beta_label()
    output_stem = "uci_deep_kernel_direct_feature_scaling_beta_$beta_label"
    direct_main(
        ;
        prior_builder = feature_scaling_prior,
        output_stem,
        backend_label = "direct-feature-scaling",
        fixed_lengthscale = FEATURE_SCALING_LENGTHSCALE,
        optimizer_configs_for_depth = feature_scaling_optimizers,
    )
    results_dir = joinpath(dirname(@__DIR__), "results")
    write_feature_dependencies(
        joinpath(results_dir, "$(output_stem)_summary.csv"),
        joinpath(results_dir, "$(output_stem)_best_features.csv"),
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && feature_scaling_main()
