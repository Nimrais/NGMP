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
    return (
        hidden = initialize_dvi_layer(
            rng, input_dimension, config.hidden_units, config,
        ),
        output = initialize_dvi_layer(
            rng, config.hidden_units, output_dimension, config,
        ),
    )
end

"""
    propagate_dvi(params, features, config)

Deterministically propagate Gaussian activation moments through the
one-hidden-layer Bayesian ReLU network. `config.propagation == "full"`
implements DVI; `"diagonal"` implements dDVI.
"""
function propagate_dvi(params, features::AbstractMatrix, config::DVIConfig)
    if config.propagation == "full"
        hidden_pre = linear_certain_full(features, params.hidden)
        hidden_post = relu_moments_full(
            hidden_pre.mean, hidden_pre.covariance,
        )
        return linear_full(hidden_post, params.output)
    elseif config.propagation == "diagonal"
        hidden_pre = linear_certain_diagonal(features, params.hidden)
        hidden_post = relu_moments_diagonal(
            hidden_pre.mean, hidden_pre.variance,
        )
        return linear_diagonal(hidden_post, params.output)
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
)
    length(means) == length(log_stds) ||
        throw(DimensionMismatch("posterior mean and scale disagree"))
    m = Float32(length(means))
    posterior_variances = exp.(2f0 .* log_stds)
    second_moment_sum = sum(posterior_variances .+ means .^ 2)
    degrees = m + 2f0 * Float32(alpha) + 2f0
    regularized_second_moment =
        second_moment_sum + 2f0 * Float32(beta)
    return 0.5f0 * (
        m * log(regularized_second_moment / degrees) +
        second_moment_sum * degrees / regularized_second_moment -
        (m + 2f0 * sum(log_stds))
    )
end

function empirical_bayes_layer_kl(layer, alpha::Real, beta::Real)
    m = Float32(length(layer.weight_mu) + length(layer.bias_mu))
    second_moment_sum =
        sum(exp.(2f0 .* layer.weight_log_std) .+ layer.weight_mu .^ 2) +
        sum(exp.(2f0 .* layer.bias_log_std) .+ layer.bias_mu .^ 2)
    log_std_sum =
        sum(layer.weight_log_std) + sum(layer.bias_log_std)
    degrees = m + 2f0 * Float32(alpha) + 2f0
    regularized_second_moment =
        second_moment_sum + 2f0 * Float32(beta)
    return 0.5f0 * (
        m * log(regularized_second_moment / degrees) +
        second_moment_sum * degrees / regularized_second_moment -
        (m + 2f0 * log_std_sum)
    )
end

function empirical_bayes_layer_prior_variance(
    layer,
    alpha::Real,
    beta::Real,
)
    second_moment_sum =
        sum(exp.(2f0 .* layer.weight_log_std) .+ layer.weight_mu .^ 2) +
        sum(exp.(2f0 .* layer.bias_log_std) .+ layer.bias_mu .^ 2)
    return (
        second_moment_sum + 2f0 * Float32(beta)
    ) / (
        Float32(length(layer.weight_mu) + length(layer.bias_mu)) +
        2f0 * Float32(alpha) +
        2f0
    )
end

function empirical_bayes_group_prior_variance(
    means::AbstractArray,
    log_stds::AbstractArray,
    alpha::Real,
    beta::Real,
)
    second_moment_sum = sum(exp.(2f0 .* log_stds) .+ means .^ 2)
    return (
        second_moment_sum + 2f0 * Float32(beta)
    ) / (
        Float32(length(means)) + 2f0 * Float32(alpha) + 2f0
    )
end

function empirical_bayes_kl(params, config::DVIConfig)
    layer_kl(layer) = empirical_bayes_layer_kl(
        layer, config.eb_alpha, config.eb_beta,
    )
    return layer_kl(params.hidden) + layer_kl(params.output)
end

function empirical_bayes_prior_variances(params, config::DVIConfig)
    layer_variance(layer) = Float64(
        empirical_bayes_layer_prior_variance(
            layer, config.eb_alpha, config.eb_beta,
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
    precision_expectation =
        exp.(-log_variance .+ 0.5f0 .* log_variance_variance)
    squared_residual =
        (mean .- mean_log_variance_covariance .- targets) .^ 2
    return -0.5f0 .* (
        DVI_LOG2PI .+
        log_variance .+
        precision_expectation .* (mean_variance .+ squared_residual)
    )
end

function kl_weight(epoch::Int, config::DVIConfig)
    epoch <= config.kl_warmup_epochs && return 0.0
    config.kl_anneal_epochs == 0 && return 1.0
    anneal_epoch = epoch - config.kl_warmup_epochs
    return clamp(anneal_epoch / config.kl_anneal_epochs, 0.0, 1.0)
end

function dvi_loss(
    params,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
    n_training::Int;
    epoch::Int = config.max_epochs,
)
    n_training >= 1 || throw(ArgumentError("n_training must be positive"))
    output = propagate_dvi(params, features, config)
    reconstruction = mean(expected_log_likelihood(
        output, targets, likelihood, config,
    ))
    return Float32(kl_weight(epoch, config)) *
        empirical_bayes_kl(params, config) / n_training -
        reconstruction
end

function sample_layer(layer, rng::AbstractRNG)
    return (
        weight = layer.weight_mu .+
            exp.(layer.weight_log_std) .*
            randn(rng, Float32, size(layer.weight_mu)),
        bias = layer.bias_mu .+
            exp.(layer.bias_log_std) .*
            randn(rng, Float32, size(layer.bias_mu)),
    )
end

function sample_forward(
    params,
    features::AbstractMatrix,
    rng::AbstractRNG,
)
    hidden = sample_layer(params.hidden, rng)
    output = sample_layer(params.output, rng)
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
