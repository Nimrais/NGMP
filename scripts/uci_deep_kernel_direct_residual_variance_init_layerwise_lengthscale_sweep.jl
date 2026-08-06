#!/usr/bin/env julia

# Successful residual-variance initialization combined with layer-specific
# lengthscales. Two separate schedules are run:
#
#   increasing: lengthscale[layer] = 1.5 * sqrt(2)^(layer - 1)
#   decreasing: lengthscale[layer] = 1.5 / sqrt(2)^(layer - 1)
#
# Other focused-sweep settings remain: all six datasets, 800 Matérn features,
# depths 1:4, prior gain 1.0, damped inference, and vector transport at
# beta = 0.1, 0.2, 0.5, 0.8.

ENV["UCI_RESIDUAL_VARIANCE_INIT_MIN"] = get(
    ENV,
    "UCI_RESIDUAL_VARIANCE_INIT_MIN",
    "1e-3",
)
ENV["UCI_RESIDUAL_VARIANCE_INIT_MAX"] = get(
    ENV,
    "UCI_RESIDUAL_VARIANCE_INIT_MAX",
    "1e3",
)
ENV["UCI_PRIOR_OPTIMIZER_GAINS"] = get(
    ENV,
    "UCI_PRIOR_OPTIMIZER_GAINS",
    "1.0",
)

include(joinpath(
    @__DIR__,
    "uci_deep_kernel_direct_residual_variance_init_sweep.jl",
))

const RESIDUAL_LAYERWISE_BASE_LENGTHSCALE = parse(
    Float64,
    get(ENV, "UCI_RESIDUAL_LAYERWISE_BASE_LENGTHSCALE", "1.5"),
)

function residual_variance_layerwise_main()
    validate_residual_variance_init_bounds()
    RESIDUAL_LAYERWISE_BASE_LENGTHSCALE > 0 || throw(ArgumentError(
        "layerwise base lengthscale must be positive",
    ))
    PRIOR_OPTIMIZER_GAINS == [1.0] || throw(ArgumentError(
        "this focused layerwise run requires prior gain 1.0",
    ))
    empty!(RESIDUAL_VARIANCE_ANCHOR_CACHE)
    range_label = residual_variance_init_range_label()
    schedules = (
        increasing = sqrt(2.0),
        decreasing = inv(sqrt(2.0)),
    )

    for (schedule, factor) in pairs(schedules)
        output_stem =
            "uci_deep_kernel_direct_matern800_depth1_4_residual_variance_init_$(range_label)_prior_gain_1_00_layerwise_$(schedule)_optimizer_sweep"
        @printf(
            "\n=== residual init [%.1e, %.1e], layerwise %s, base lengthscale %.3f ===\n",
            RESIDUAL_VARIANCE_INIT_MIN,
            RESIDUAL_VARIANCE_INIT_MAX,
            schedule,
            RESIDUAL_LAYERWISE_BASE_LENGTHSCALE,
        )
        prior_builder = (depth, Φ, targets) ->
            residual_variance_prior_parameters(depth, Φ, targets, 1.0)
        direct_main(
            ;
            prior_builder,
            output_stem,
            backend_label = "direct-residual-init-layerwise-$schedule",
            fixed_lengthscale = RESIDUAL_LAYERWISE_BASE_LENGTHSCALE,
            layerwise_lengthscale_factor = factor,
            optimizer_configs_for_depth = prior_optimizer_configs,
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    residual_variance_layerwise_main()
