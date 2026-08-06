#!/usr/bin/env julia

# Focused UCI interaction sweep:
#   preprocessing: multiscale Matérn-3/2 + linear
#   random features: 800
#   depths: 1:4
#   optimizers: damped (beta=0) and vector transport beta=0.1,0.2,0.5,0.8
#   precision-level prior gains: 1.0, 1.5
# All unmentioned settings retain the direct benchmark defaults.

ENV["UCI_DATASETS"] = get(
    ENV,
    "UCI_DATASETS",
    "yacht,housing,energy,concrete,wine,power",
)
ENV["UCI_DEPTHS"] = get(ENV, "UCI_DEPTHS", "1,2,3,4")
ENV["UCI_DIRECT_FEATURE_DIMENSIONS"] = get(
    ENV,
    "UCI_DIRECT_FEATURE_DIMENSIONS",
    "800",
)
ENV["UCI_DIRECT_PREPROCESSING"] = get(
    ENV,
    "UCI_DIRECT_PREPROCESSING",
    "multiscale_matern32_linear",
)
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = get(
    ENV,
    "UCI_DIRECT_VECTOR_TRANSPORT_ALPHA",
    "0.6",
)

include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))

const PRIOR_OPTIMIZER_GAINS = parse.(Float64, split(
    get(ENV, "UCI_PRIOR_OPTIMIZER_GAINS", "1.0,1.5"),
    ',',
))
const PRIOR_OPTIMIZER_VT_BETAS = parse.(Float64, split(
    get(ENV, "UCI_PRIOR_OPTIMIZER_VT_BETAS", "0.1,0.2,0.5,0.8"),
    ',',
))

function prior_optimizer_configs(depth)
    # At depth 1 there are no learned precision sites, so vector transport has
    # nothing to update and would only duplicate the damped baseline.
    depth == 1 && return ((:damped, 0.0),)
    return [
        (:damped, 0.0),
        [(:vector_transport, beta) for beta in PRIOR_OPTIMIZER_VT_BETAS]...,
    ]
end

function prior_optimizer_gain_parameters(depth, Φ, targets, gain)
    gain > 0 || throw(ArgumentError("prior gain must be positive"))
    prior = direct_prior_parameters(depth, Φ, targets)
    level_prior_precisions = map(prior.level_prior_precisions) do precision
        gained = copy(precision)
        gained[1:(end - 1), 1:(end - 1)] ./= gain^2
        gained
    end
    return merge(prior, (; level_prior_precisions))
end

gain_file_label(gain) = replace(@sprintf("%.2f", gain), "." => "_")

function prior_gain_optimizer_sweep_main()
    for gain in PRIOR_OPTIMIZER_GAINS
        label = gain_file_label(gain)
        output_stem =
            "uci_deep_kernel_direct_matern800_depth1_4_prior_gain_$(label)_optimizer_sweep"
        @printf(
            "\n=== Matérn-3/2, 800 features, prior gain %.2f ===\n",
            gain,
        )
        prior_builder = (depth, Φ, targets) ->
            prior_optimizer_gain_parameters(depth, Φ, targets, gain)
        direct_main(
            ;
            prior_builder,
            output_stem,
            backend_label = "direct-matern800-prior-gain-$label",
            optimizer_configs_for_depth = prior_optimizer_configs,
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    prior_gain_optimizer_sweep_main()
