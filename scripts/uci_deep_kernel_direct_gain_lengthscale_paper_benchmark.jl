#!/usr/bin/env julia

# Joint UCI sweep over precision-level prior gain and fixed lengthscale.
#
# Only the Matérn-3/2 multiscale preprocessing is used. Deep models compare
# damped inference with regular vector transport; VT Nesterov is intentionally
# excluded.
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
ENV["UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA"] = "0.6"
ENV["UCI_DIRECT_PREPROCESSING"] = "multiscale_matern32_linear"
include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))

const JOINT_PRIOR_GAINS = parse.(Float64, split(
    get(ENV, "UCI_JOINT_PRIOR_GAINS", "1.25,1.5"),
    ',',
))
const JOINT_LENGTHSCALES = parse.(Float64, split(
    get(ENV, "UCI_JOINT_LENGTHSCALES", "1.5,1.75,2.0"),
    ',',
))
const JOINT_OPTIMIZERS = [
    (:damped, 0.0),
    [(:vector_transport, beta)
     for beta in (0.05, 0.10, 0.20, 0.50, 0.80)]...,
]

joint_optimizers_for_depth(depth) =
    depth == 1 ? ((:damped, 0.0),) : JOINT_OPTIMIZERS

parameter_label(value) =
    replace(@sprintf("%.2f", value), "." => "_")

function joint_prior_parameters(depth, Φ, targets, gain)
    gain > 0 || throw(ArgumentError("prior gain must be positive"))
    prior = direct_prior_parameters(depth, Φ, targets)
    gain_squared = abs2(gain)
    level_prior_precisions = map(prior.level_prior_precisions) do precision
        gained = copy(precision)
        gained[1:(end - 1), 1:(end - 1)] ./= gain_squared
        gained
    end
    return merge(prior, (; level_prior_precisions))
end

function gain_lengthscale_main()
    for gain in JOINT_PRIOR_GAINS, lengthscale in JOINT_LENGTHSCALES
        gain > 0 || throw(ArgumentError("prior gain must be positive"))
        lengthscale > 0 ||
            throw(ArgumentError("lengthscale must be positive"))
        gain_text = parameter_label(gain)
        lengthscale_text = parameter_label(lengthscale)
        experiment = "gain_$(gain_text)_lengthscale_$(lengthscale_text)"
        @printf(
            "\n=== prior gain %.2f, fixed lengthscale %.2f ===\n",
            gain,
            lengthscale,
        )
        prior_builder = (depth, Φ, targets) ->
            joint_prior_parameters(depth, Φ, targets, gain)
        direct_main(
            ;
            prior_builder,
            output_stem = "uci_deep_kernel_direct_$(experiment)_paper",
            backend_label = "direct-$experiment",
            fixed_lengthscale = lengthscale,
            optimizer_configs_for_depth = joint_optimizers_for_depth,
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    gain_lengthscale_main()
