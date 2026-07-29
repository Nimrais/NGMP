#!/usr/bin/env julia

# Direct UCI deep-kernel benchmark with design-aware prior scaling.
#
# The baseline assigns the same variance to every feature weight. This version
# instead fixes the average prior variance of the feature-dependent score,
#
#     E_x Var[w' * phi(x)] = target variance,
#
# using the training design matrix. This is the Bayesian analogue of a
# Xavier-style fan-in correction: changing the RFF count, input dimension, or
# feature-map energy does not change the initial score SNR merely by rescaling
# the design. The intercept prior is kept separate so the precision anchors are
# unchanged.
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
ENV["UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA"] = "0.6"
include(joinpath(@__DIR__, "uci_deep_kernel_direct_paper_benchmark.jl"))

const FIXED_SNR_MEAN_FEATURE_VARIANCE = parse(
    Float64,
    get(ENV, "UCI_FIXED_SNR_MEAN_FEATURE_VARIANCE", "1.0"),
)
const FIXED_SNR_LEVEL_FEATURE_VARIANCE = parse(
    Float64,
    get(ENV, "UCI_FIXED_SNR_LEVEL_FEATURE_VARIANCE", string(0.4^2)),
)

function fixed_snr_weight_variance(Φ, target_variance)
    target_variance > 0 ||
        throw(ArgumentError("fixed-SNR target variance must be positive"))
    size(Φ, 2) > 1 ||
        throw(ArgumentError("fixed-SNR design must include features and intercept"))

    feature_design = @view Φ[:, 1:(end - 1)]
    mean_feature_energy = mean(vec(sum(abs2, feature_design; dims = 2)))
    mean_feature_energy > DIRECT_JITTER ||
        throw(ArgumentError("fixed-SNR design has zero feature energy"))
    return target_variance / mean_feature_energy
end

function fixed_snr_prior_parameters(depth, Φ, targets)
    p = size(Φ, 2)
    anchor = -log(max(mean(abs2, diff(sort(targets))) / 2, 1e-8))
    mean_weight_variance = fixed_snr_weight_variance(
        Φ,
        FIXED_SNR_MEAN_FEATURE_VARIANCE,
    )
    level_weight_variance = fixed_snr_weight_variance(
        Φ,
        FIXED_SNR_LEVEL_FEATURE_VARIANCE,
    )

    mean_prior_mean = zeros(p)
    mean_prior_precision = Matrix(Diagonal(vcat(
        fill(inv(mean_weight_variance), p - 1),
        1.0,
    )))
    level_prior_means = [
        vcat(zeros(p - 1), level == 1 ? anchor : log(TOP_CARRIER))
        for level in 1:(depth - 1)
    ]
    level_prior_precisions = [
        Matrix(Diagonal(vcat(
            fill(inv(level_weight_variance), p - 1),
            1.0,
        )))
        for _ in 1:(depth - 1)
    ]

    return (;
        mean_prior_mean,
        mean_prior_precision,
        level_prior_means,
        level_prior_precisions,
    )
end

function fixed_snr_main()
    direct_main(
        ;
        prior_builder = fixed_snr_prior_parameters,
        output_stem = "uci_deep_kernel_direct_fixed_snr_paper",
        backend_label = "direct-fixed-snr",
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && fixed_snr_main()
