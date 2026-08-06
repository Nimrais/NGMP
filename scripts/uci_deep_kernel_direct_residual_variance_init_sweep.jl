#!/usr/bin/env julia

# Same focused Matérn-800 prior-gain/optimizer sweep as
# `uci_deep_kernel_direct_prior_gain_optimizer_sweep.jl`, but the first
# precision layer is initialized from the fitted depth-1 residual variance.
# This avoids the dataset-size-dependent sorted-target-spacing anchor.

ENV["UCI_PRIOR_OPTIMIZER_GAINS"] = get(
    ENV,
    "UCI_PRIOR_OPTIMIZER_GAINS",
    "1.0",
)

include(joinpath(
    @__DIR__,
    "uci_deep_kernel_direct_prior_gain_optimizer_sweep.jl",
))

# `direct_main` reuses the same design matrix object across depths and
# optimizers for a split. Identity caching therefore computes the shallow
# noise estimate only once for each dataset/split/feature map.
const RESIDUAL_VARIANCE_ANCHOR_CACHE = WeakKeyDict{Any, Float64}()
const RESIDUAL_VARIANCE_INIT_MIN = parse(
    Float64,
    get(ENV, "UCI_RESIDUAL_VARIANCE_INIT_MIN", "1e-1"),
)
const RESIDUAL_VARIANCE_INIT_MAX = parse(
    Float64,
    get(ENV, "UCI_RESIDUAL_VARIANCE_INIT_MAX", "1e1"),
)

function validate_residual_variance_init_bounds()
    0 < RESIDUAL_VARIANCE_INIT_MIN <= RESIDUAL_VARIANCE_INIT_MAX ||
        throw(ArgumentError(
            "residual-variance init bounds must satisfy 0 < min <= max",
        ))
end

function residual_variance_init_range_label()
    minimum_label = replace(
        @sprintf("%.0e", RESIDUAL_VARIANCE_INIT_MIN),
        "-" => "m",
        "+" => "p",
    )
    maximum_label = replace(
        @sprintf("%.0e", RESIDUAL_VARIANCE_INIT_MAX),
        "-" => "m",
        "+" => "p",
    )
    return "$(minimum_label)_to_$(maximum_label)"
end

function fitted_residual_precision_anchor(Φ, targets)
    return get!(RESIDUAL_VARIANCE_ANCHOR_CACHE, Φ) do
        shallow = fit_direct_depth_one(
            Φ,
            targets;
            prior_builder = direct_prior_parameters,
        )
        # Match the aleatoric predictive variance used by
        # `predict_direct_model` for the depth-1 baseline.
        residual_variance = shallow.noise_rate /
            max(shallow.noise_shape - 1, DIRECT_JITTER)
        residual_variance = clamp(
            residual_variance,
            RESIDUAL_VARIANCE_INIT_MIN,
            RESIDUAL_VARIANCE_INIT_MAX,
        )
        -log(residual_variance)
    end
end

function residual_variance_prior_parameters(depth, Φ, targets, gain)
    gain > 0 || throw(ArgumentError("prior gain must be positive"))
    prior = direct_prior_parameters(depth, Φ, targets)
    depth == 1 && return prior

    anchor = fitted_residual_precision_anchor(Φ, targets)
    level_prior_means = copy.(prior.level_prior_means)
    level_prior_means[1][end] = anchor

    level_prior_precisions = map(prior.level_prior_precisions) do precision
        gained = copy(precision)
        # Prior gain continues to affect only non-intercept weights. The new
        # residual-derived intercept remains directly comparable across gains.
        gained[1:(end - 1), 1:(end - 1)] ./= gain^2
        gained
    end
    return merge(prior, (; level_prior_means, level_prior_precisions))
end

function residual_variance_init_sweep_main()
    validate_residual_variance_init_bounds()
    empty!(RESIDUAL_VARIANCE_ANCHOR_CACHE)
    range_label = residual_variance_init_range_label()
    for gain in PRIOR_OPTIMIZER_GAINS
        label = gain_file_label(gain)
        output_stem =
            "uci_deep_kernel_direct_matern800_depth1_4_residual_variance_init_$(range_label)_prior_gain_$(label)_optimizer_sweep"
        @printf(
            "\n=== residual-variance init [%.1e, %.1e], Matérn-3/2, 800 features, prior gain %.2f ===\n",
            RESIDUAL_VARIANCE_INIT_MIN,
            RESIDUAL_VARIANCE_INIT_MAX,
            gain,
        )
        prior_builder = (depth, Φ, targets) ->
            residual_variance_prior_parameters(depth, Φ, targets, gain)
        direct_main(
            ;
            prior_builder,
            output_stem,
            backend_label = "direct-residual-variance-init-gain-$label",
            optimizer_configs_for_depth = prior_optimizer_configs,
        )
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    residual_variance_init_sweep_main()
