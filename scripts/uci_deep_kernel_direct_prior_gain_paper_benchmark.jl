#!/usr/bin/env julia

# Direct UCI deep-kernel benchmark for precision-level prior gains.
#
# This keeps the successful baseline prior as the reference and changes only
# the standard deviation of non-intercept weights in the precision hierarchy:
#
#     sd(w_level,feature) = gain * 0.4
#
# The observation-mean prior and all intercept/anchor priors remain unchanged.
# Gain 1.0 is therefore an exact baseline-prior control.
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
ENV["UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA"] = "0.6"
include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))

const DIRECT_PRIOR_GAINS = parse.(Float64, split(
    get(ENV, "UCI_DIRECT_PRIOR_GAINS", "1.25,1.5"),
    ',',
))

function gain_label(gain)
    text = replace(@sprintf("%.2f", gain), "." => "_")
    return "gain_$text"
end

function gained_level_prior_parameters(depth, Φ, targets, gain)
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

function prior_gain_main()
    for gain in DIRECT_PRIOR_GAINS
        label = gain_label(gain)
        @printf(
            "\n=== precision-level prior gain %.2f (weight SD %.3f) ===\n",
            gain,
            0.4 * gain,
        )
        prior_builder = (depth, Φ, targets) ->
            gained_level_prior_parameters(depth, Φ, targets, gain)
        direct_main(
            ;
            prior_builder,
            output_stem = "uci_deep_kernel_direct_prior_$(label)_paper",
            backend_label = "direct-prior-$label",
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && prior_gain_main()
