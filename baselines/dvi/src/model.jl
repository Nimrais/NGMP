function initialize_dvi_layer(
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
    config::DVIConfig,
)
    weight_variance = Float32(config.initialization_scale / input_dimension)
    bias_variance = Float32(
        weight_variance / config.bias_variance_divisor,
    )
    return (
        weight_mu = sqrt(weight_variance) .* randn(
            rng, Float32, output_dimension, input_dimension,
        ),
        weight_log_std = fill(
            0.5f0 * log(weight_variance),
            output_dimension,
            input_dimension,
        ),
        bias_mu = zeros(Float32, output_dimension),
        bias_log_std = fill(
            0.5f0 * log(bias_variance), output_dimension,
        ),
    )
end

function initialize_model(
    input_dimension::Int,
    likelihood::String,
    config::DVIConfig;
    seed::Int,
)
    likelihood in ("homoscedastic", "heteroscedastic") ||
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    rng = StableRNG(seed)
    output_dimension = likelihood == "homoscedastic" ? 1 : 2
    hidden = initialize_dvi_layer(
        rng, input_dimension, config.hidden_units, config,
    )
    output = initialize_dvi_layer(
        rng, config.hidden_units, output_dimension, config,
    )
    if likelihood == "heteroscedastic"
        output.weight_mu[2, :] .= 0f0
        output.weight_log_std[2, :] .= Float32(log(
            config.log_variance_head_posterior_std,
        ))
        output.bias_mu[2] = Float32(config.log_variance_head_initial_bias)
        output.bias_log_std[2] = Float32(log(
            config.log_variance_head_posterior_std,
        ))
    end
    return (
        hidden = hidden,
        output = output,
    )
end

"""
    propagate_dvi(params, features, config)

Deterministically propagate Gaussian activation moments through the
one-hidden-layer Bayesian ReLU network. `config.propagation == "full"`
implements DVI; `"diagonal"` implements dDVI.
"""
function propagate_dvi(
    params,
    features::AbstractMatrix,
    config::DVIConfig;
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    if config.propagation == "full"
        hidden_pre = linear_certain_full(
            features, params.hidden, config, tracker,
        )
        hidden_post = relu_moments_full(
            hidden_pre.mean, hidden_pre.covariance,
        )
        return linear_full(hidden_post, params.output, config, tracker)
    elseif config.propagation == "diagonal"
        hidden_pre = linear_certain_diagonal(
            features, params.hidden, config, tracker,
        )
        hidden_post = relu_moments_diagonal(
            hidden_pre.mean, hidden_pre.variance,
        )
        return linear_diagonal(hidden_post, params.output, config, tracker)
    end
    throw(ArgumentError("unknown propagation mode '$(config.propagation)'"))
end

function output_covariance_entry(
    output,
    first_index::Int,
    second_index::Int,
    config::DVIConfig,
)
    if config.propagation == "full"
        return vec(output.covariance[:, first_index, second_index])
    elseif first_index == second_index
        return vec(output.variance[:, first_index])
    end
    return zero.(vec(output.mean[:, first_index]))
end

function empirical_bayes_group_kl(
    means::AbstractArray,
    log_stds::AbstractArray,
    alpha::Real,
    beta::Real,
    config::DVIConfig = DVIConfig(),
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    length(means) == length(log_stds) ||
        throw(DimensionMismatch("posterior mean and scale disagree"))
    m = Float64(length(means))
    posterior_variances = Float64.(safe_exp(
        2f0 .* log_stds, config, tracker,
    ))
    means64 = Float64.(means)
    second_moment_sum = sum(posterior_variances .+ means64 .^ 2)
    degrees = m + 2 * Float64(alpha) + 2
    regularized_second_moment =
        second_moment_sum + 2 * Float64(beta)
    return 0.5 * (
        m * log(regularized_second_moment / degrees) +
        second_moment_sum * degrees / regularized_second_moment -
        (m + 2f0 * sum(log_stds))
    )
end

function empirical_bayes_layer_kl(
    layer,
    alpha::Real,
    beta::Real,
    config::DVIConfig = DVIConfig(),
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    m = Float64(length(layer.weight_mu) + length(layer.bias_mu))
    second_moment_sum =
        sum(Float64.(safe_exp(
            2f0 .* layer.weight_log_std, config, tracker,
        )) .+ Float64.(layer.weight_mu) .^ 2) +
        sum(Float64.(safe_exp(
            2f0 .* layer.bias_log_std, config, tracker,
        )) .+ Float64.(layer.bias_mu) .^ 2)
    log_std_sum =
        sum(layer.weight_log_std) + sum(layer.bias_log_std)
    degrees = m + 2 * Float64(alpha) + 2
    regularized_second_moment =
        second_moment_sum + 2 * Float64(beta)
    return 0.5 * (
        m * log(regularized_second_moment / degrees) +
        second_moment_sum * degrees / regularized_second_moment -
        (m + 2f0 * log_std_sum)
    )
end

function empirical_bayes_layer_prior_variance(
    layer,
    alpha::Real,
    beta::Real,
    config::DVIConfig = DVIConfig(),
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    second_moment_sum =
        sum(Float64.(safe_exp(
            2f0 .* layer.weight_log_std, config, tracker,
        )) .+ Float64.(layer.weight_mu) .^ 2) +
        sum(Float64.(safe_exp(
            2f0 .* layer.bias_log_std, config, tracker,
        )) .+ Float64.(layer.bias_mu) .^ 2)
    return (
        second_moment_sum + 2 * Float64(beta)
    ) / (
        Float64(length(layer.weight_mu) + length(layer.bias_mu)) +
        2 * Float64(alpha) +
        2
    )
end

function empirical_bayes_group_prior_variance(
    means::AbstractArray,
    log_stds::AbstractArray,
    alpha::Real,
    beta::Real,
    config::DVIConfig = DVIConfig(),
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    second_moment_sum = sum(Float64.(safe_exp(
        2f0 .* log_stds, config, tracker,
    )) .+ Float64.(means) .^ 2)
    return (
        second_moment_sum + 2 * Float64(beta)
    ) / (
        Float64(length(means)) + 2 * Float64(alpha) + 2
    )
end

function empirical_bayes_kl(
    params,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    layer_kl(layer) = empirical_bayes_layer_kl(
        layer, config.eb_alpha, config.eb_beta, config, tracker,
    )
    return layer_kl(params.hidden) + layer_kl(params.output)
end

function empirical_bayes_prior_variances(params, config::DVIConfig)
    layer_variance(layer) = Float64(
        empirical_bayes_layer_prior_variance(
            layer, config.eb_alpha, config.eb_beta, config,
        ),
    )
    return (
        hidden = layer_variance(params.hidden),
        output = layer_variance(params.output),
    )
end

"""
    expected_log_likelihood(output, targets, likelihood, config)

Equation (8) of Wu et al. (2019), evaluated pointwise.
"""
function expected_log_likelihood(
    output,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    mean = vec(output.mean[:, 1])
    mean_variance = output_covariance_entry(output, 1, 1, config)
    if likelihood == "heteroscedastic"
        log_variance = vec(output.mean[:, 2])
        log_variance_variance =
            output_covariance_entry(output, 2, 2, config)
        mean_log_variance_covariance =
            output_covariance_entry(output, 1, 2, config)
    elseif likelihood == "homoscedastic"
        log_variance = fill(
            Float32(config.homo_log_variance), length(mean),
        )
        log_variance_variance = zero.(mean)
        mean_log_variance_covariance = zero.(mean)
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end
    precision_log_moment =
        -log_variance .+ 0.5f0 .* log_variance_variance
    precision_log_moment = Float64.(precision_log_moment)
    precision_expectation = safe_exp(
        precision_log_moment, config, tracker,
    )
    residual = mean .- mean_log_variance_covariance .- targets
    squared_residual = Float64.(residual) .^ 2
    return -0.5 .* (
        log(2pi) .+
        Float64.(log_variance) .+
        precision_expectation .* (mean_variance .+ squared_residual)
    )
end

function kl_weight(progress::Int, config::DVIConfig)
    warmup = config.kl_schedule_unit == "steps" ?
        config.kl_warmup_steps : config.kl_warmup_epochs
    anneal = config.kl_schedule_unit == "steps" ?
        config.kl_anneal_steps : config.kl_anneal_epochs
    progress <= warmup && return 0.0
    anneal == 0 && return 1.0
    return clamp((progress - warmup) / anneal, 0.0, 1.0)
end

kl_progress(epoch::Int, optimizer_step::Int, config::DVIConfig) =
    config.kl_schedule_unit == "steps" ? optimizer_step : epoch

function dvi_loss(
    params,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
    n_training::Int;
    epoch::Int = config.max_epochs,
    optimizer_step::Int = config.kl_schedule_unit == "steps" ?
        config.kl_warmup_steps + config.kl_anneal_steps : 0,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    n_training >= 1 || throw(ArgumentError("n_training must be positive"))
    progress = kl_progress(epoch, optimizer_step, config)
    return dvi_loss_with_kl_weight(
        params,
        features,
        targets,
        likelihood,
        config,
        n_training,
        kl_weight(progress, config);
        tracker = tracker,
    )
end

function dvi_loss_with_kl_weight(
    params,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
    n_training::Int,
    weight;
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    n_training >= 1 || throw(ArgumentError("n_training must be positive"))
    output = propagate_dvi(params, features, config; tracker = tracker)
    reconstruction = mean(expected_log_likelihood(
        output, targets, likelihood, config, tracker,
    ))
    return weight *
        empirical_bayes_kl(params, config, tracker) / n_training -
        reconstruction
end

function clamp_tally(values, config::DVIConfig)
    lower = convert(eltype(values), config.safe_exp_min)
    upper = convert(eltype(values), config.safe_exp_max)
    return (
        calls = 1,
        elements = length(values),
        lower_clamps = sum(values .< lower),
        upper_clamps = sum(values .> upper),
    )
end

function dvi_loss_clamp_statistics(
    params,
    features::AbstractMatrix,
    likelihood::String,
    config::DVIConfig,
)
    parameter_inputs = (
        2f0 .* params.hidden.weight_log_std,
        2f0 .* params.hidden.bias_log_std,
        2f0 .* params.output.weight_log_std,
        2f0 .* params.output.bias_log_std,
    )
    parameter_tallies = map(
        values -> clamp_tally(values, config), parameter_inputs,
    )

    output = propagate_dvi(params, features, config)
    precision_input = if likelihood == "heteroscedastic"
        -vec(output.mean[:, 2]) .+
        0.5f0 .* output_covariance_entry(output, 2, 2, config)
    elseif likelihood == "homoscedastic"
        fill(-Float32(config.homo_log_variance), size(features, 1))
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end
    precision_tally = clamp_tally(precision_input, config)

    return (
        calls = 2 * sum(tally.calls for tally in parameter_tallies) +
            precision_tally.calls,
        elements = 2 * sum(tally.elements for tally in parameter_tallies) +
            precision_tally.elements,
        lower_clamps = 2 * sum(
            tally.lower_clamps for tally in parameter_tallies
        ) + precision_tally.lower_clamps,
        upper_clamps = 2 * sum(
            tally.upper_clamps for tally in parameter_tallies
        ) + precision_tally.upper_clamps,
    )
end

function sample_layer(layer, rng::AbstractRNG, config::DVIConfig)
    return (
        weight = layer.weight_mu .+
            safe_exp(layer.weight_log_std, config) .*
            randn(rng, Float32, size(layer.weight_mu)),
        bias = layer.bias_mu .+
            safe_exp(layer.bias_log_std, config) .*
            randn(rng, Float32, size(layer.bias_mu)),
    )
end

function sample_forward(
    params,
    features::AbstractMatrix,
    rng::AbstractRNG,
    config::DVIConfig = DVIConfig(),
)
    hidden = sample_layer(params.hidden, rng, config)
    output = sample_layer(params.output, rng, config)
    hidden_activation = max.(
        features * transpose(hidden.weight) .+ transpose(hidden.bias),
        0f0,
    )
    return hidden_activation * transpose(output.weight) .+
        transpose(output.bias)
end

clip_gradient(::Nothing, limit::Real) = nothing
clip_gradient(array::AbstractArray, limit::Real) =
    isinf(limit) ? array : clamp.(array, -limit, limit)
function clip_gradient(tuple::NamedTuple, limit::Real)
    return NamedTuple{keys(tuple)}(
        map(value -> clip_gradient(value, limit), values(tuple)),
    )
end

parameters_are_finite(::Nothing) = true
parameters_are_finite(array::AbstractArray) = all(isfinite, array)
parameters_are_finite(tuple::NamedTuple) =
    all(parameters_are_finite, values(tuple))

gradient_maximum_absolute(::Nothing) = 0.0
gradient_maximum_absolute(array::AbstractArray) =
    isempty(array) ? 0.0 : maximum(abs, array)
gradient_maximum_absolute(tuple::NamedTuple) =
    maximum(gradient_maximum_absolute, values(tuple); init = 0.0)
