#!/usr/bin/env julia

# Final single-configuration UCI benchmark selected from the layerwise sweep:
#   - 1,000 Matérn-3/2 random features
#   - depth 2
#   - decreasing layerwise lengthscale: 1.5, 1.5/sqrt(2)
#   - vector transport, alpha=0.6, beta=0.8
#   - shallow residual-variance initialization clamped to [1e-3, 1e3]
#   - precision prior gain 1.0
#   - all six available DVI/UCI datasets, 20 splits by default

ENV["UCI_DATASETS"] = get(
    ENV,
    "UCI_DATASETS",
    "yacht,housing,energy,concrete,wine,power",
)
ENV["UCI_DEPTHS"] = "2"
ENV["UCI_DIRECT_FEATURE_DIMENSIONS"] = "1200"
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

final_decreasing_optimizer(depth) = ((:vector_transport, 0.8),)

function final_decreasing_matern1000_main()
    validate_residual_variance_init_bounds()
    empty!(RESIDUAL_VARIANCE_ANCHOR_CACHE)
    prior_builder = (depth, Φ, targets) ->
        residual_variance_prior_parameters(depth, Φ, targets, 1.0)
    direct_main(
        ;
        prior_builder,
        output_stem =
            "uci_deep_kernel_direct_final_decreasing_matern1000_depth2_vt_beta_0_80",
        backend_label = "direct-final-decreasing-matern1200",
        fixed_lengthscale = RESIDUAL_LAYERWISE_BASE_LENGTHSCALE,
        layerwise_lengthscale_factor = inv(sqrt(2.0)),
        optimizer_configs_for_depth = final_decreasing_optimizer,
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    final_decreasing_matern1000_main()
