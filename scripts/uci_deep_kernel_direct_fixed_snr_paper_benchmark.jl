#!/usr/bin/env julia

# Direct UCI deep-kernel benchmark with design-aware prior scaling.
#
# The baseline assigns the same variance to every feature weight even though
# RFF and appended linear columns have very different scales. This version
# gives each non-intercept feature an equal share of the target prior score
# variance,
#
#     Var(w_j) = target variance / (fan-in * E_x[phi_j(x)^2]),
#
# so E_x Var[w' * phi(x)] equals the target variance. This feature-aware
# Xavier correction is important for the mixed design: RFF columns have energy
# O(1 / n_rff), while standardized linear columns have energy O(1). The
# intercept prior is kept separate so the precision anchors are unchanged.
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

function fixed_snr_weight_variances(Φ, target_variance)
    target_variance > 0 ||
        throw(ArgumentError("fixed-SNR target variance must be positive"))
    size(Φ, 2) > 1 ||
        throw(ArgumentError("fixed-SNR design must include features and intercept"))

    feature_design = @view Φ[:, 1:(end - 1)]
    feature_energies = vec(mean(abs2, feature_design; dims = 1))
    all(energy -> energy > DIRECT_JITTER, feature_energies) ||
        throw(ArgumentError("fixed-SNR design has a zero-energy feature"))
    fan_in = size(feature_design, 2)
    return target_variance ./ (fan_in .* feature_energies)
end

function fixed_snr_prior_parameters(depth, Φ, targets)
    p = size(Φ, 2)
    anchor = -log(max(mean(abs2, diff(sort(targets))) / 2, 1e-8))
    mean_weight_variances = fixed_snr_weight_variances(
        Φ,
        FIXED_SNR_MEAN_FEATURE_VARIANCE,
    )
    level_weight_variances = fixed_snr_weight_variances(
        Φ,
        FIXED_SNR_LEVEL_FEATURE_VARIANCE,
    )

    mean_prior_mean = zeros(p)
    mean_prior_precision = Matrix(Diagonal(vcat(
        inv.(mean_weight_variances),
        1.0,
    )))
    level_prior_means = [
        vcat(zeros(p - 1), level == 1 ? anchor : log(TOP_CARRIER))
        for level in 1:(depth - 1)
    ]
    level_prior_precisions = [
        Matrix(Diagonal(vcat(
            inv.(level_weight_variances),
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
