#!/usr/bin/env julia

# Direct UCI deep-kernel benchmark for a lengthscale-only sweep.
#
# The selected absolute lengthscale replaces the median-distance heuristic
# before constructing the existing multiscale feature map. Inputs are already
# standardized. Priors, hierarchy, and all other settings remain at baseline.
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
ENV["UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA"] = "0.6"
ENV["UCI_DIRECT_PREPROCESSING"] = "multiscale_matern32_linear"
include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))

const DIRECT_LENGTHSCALES = parse.(Float64, split(
    get(ENV, "UCI_DIRECT_LENGTHSCALES", "1.5,2.0"),
    ',',
))

function lengthscale_label(lengthscale)
    text = replace(@sprintf("%.2f", lengthscale), "." => "_")
    return "lengthscale_$text"
end

function lengthscale_main()
    for lengthscale in DIRECT_LENGTHSCALES
        lengthscale > 0 ||
            throw(ArgumentError("lengthscale must be positive"))
        label = lengthscale_label(lengthscale)
        @printf(
            "\n=== fixed lengthscale %.2f ===\n",
            lengthscale,
        )
        direct_main(
            ;
            output_stem = "uci_deep_kernel_direct_$(label)_paper",
            backend_label = "direct-$label",
            fixed_lengthscale = lengthscale,
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && lengthscale_main()
