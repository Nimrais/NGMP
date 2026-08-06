#!/usr/bin/env julia

# Same six-dataset feature-scaling experiment, with a separate lengthscale at
# every hierarchy layer. The two schedules are:
#   increasing: lengthscale[layer] = 1.5 * sqrt(2)^(layer - 1)
#   decreasing: lengthscale[layer] = 1.5 / sqrt(2)^(layer - 1)

include(joinpath(@__DIR__, "uci_deep_kernel_direct_feature_scaling_benchmark.jl"))

function layerwise_lengthscale_main()
    schedules = (
        increasing = sqrt(2.0),
        decreasing = inv(sqrt(2.0)),
    )
    beta_label = feature_scaling_beta_label()
    for (name, factor) in pairs(schedules)
        @printf("\n=== layerwise lengthscale schedule: %s (factor %.6f) ===\n",
            name, factor)
        output_stem =
            "uci_deep_kernel_direct_layerwise_lengthscale_$(name)_beta_$beta_label"
        direct_main(
            ;
            prior_builder = feature_scaling_prior,
            output_stem,
            backend_label = "direct-layerwise-$name",
            fixed_lengthscale = FEATURE_SCALING_LENGTHSCALE,
            layerwise_lengthscale_factor = factor,
            optimizer_configs_for_depth = feature_scaling_optimizers,
        )
        results_dir = joinpath(dirname(@__DIR__), "results")
        write_feature_dependencies(
            joinpath(results_dir, "$(output_stem)_summary.csv"),
            joinpath(results_dir, "$(output_stem)_best_features.csv"),
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && layerwise_lengthscale_main()
