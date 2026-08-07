#!/usr/bin/env julia

# Final NGMP configuration with only hierarchy depth varied from 1 to 5.
# This is a precision-hierarchy depth ablation: depth 1 is the homoscedastic
# baseline; depths 2:5 add increasingly many heteroscedastic precision levels.

ENV["UCI_DATASETS"] = get(
    ENV,
    "UCI_DATASETS",
    "yacht,housing,energy,concrete,wine,power",
)
ENV["UCI_DEPTHS"] = "1,2,3,4,5"
ENV["UCI_DIRECT_FEATURE_DIMENSIONS"] = "1000"
ENV["UCI_DIRECT_PREPROCESSING"] = "multiscale_matern32_linear"
ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
ENV["UCI_PRIOR_OPTIMIZER_GAINS"] = "1.0"
ENV["UCI_RESIDUAL_VARIANCE_INIT_MIN"] = "1e-3"
ENV["UCI_RESIDUAL_VARIANCE_INIT_MAX"] = "1e3"
ENV["UCI_RESIDUAL_LAYERWISE_BASE_LENGTHSCALE"] = get(
    ENV,
    "UCI_RESIDUAL_LAYERWISE_BASE_LENGTHSCALE",
    "1.5",
)

include(joinpath(
    @__DIR__,
    "uci_deep_kernel_direct_residual_variance_init_layerwise_lengthscale_sweep.jl",
))

final_depth_ablation_optimizer(depth) = depth == 1 ?
    ((:damped, 0.0),) :
    ((:vector_transport, 0.8),)

function final_decreasing_matern1000_depth_ablation_main()
    validate_residual_variance_init_bounds()
    empty!(RESIDUAL_VARIANCE_ANCHOR_CACHE)
    prior_builder = (depth, Φ, targets) ->
        residual_variance_prior_parameters(depth, Φ, targets, 1.0)
    direct_main(
        ;
        prior_builder,
        output_stem =
            "uci_deep_kernel_direct_final_decreasing_matern1000_depth1_5_ablation",
        backend_label = "direct-final-decreasing-matern1000-depth-ablation",
        fixed_lengthscale = RESIDUAL_LAYERWISE_BASE_LENGTHSCALE,
        layerwise_lengthscale_factor = inv(sqrt(2.0)),
        optimizer_configs_for_depth = final_depth_ablation_optimizer,
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    final_decreasing_matern1000_depth_ablation_main()
