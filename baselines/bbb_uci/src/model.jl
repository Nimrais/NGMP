const LOG2PI_F32 = Float32(log(2pi))

stable_softplus(x::Real) = max(x, zero(x)) + log1p(exp(-abs(x)))

function inverse_softplus(y::Real)
    y > 0 || throw(ArgumentError("softplus inverse requires y > 0"))
    return y > 20 ? float(y) : log(expm1(float(y)))
end

function initialize_bayesian_layer(
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
    config::BBBConfig,
)
    bound = Float32(inv(sqrt(input_dimension)))
    rho = Float32(inverse_softplus(config.initial_posterior_std))
    return (
        weight_mu = rand(
            rng, Float32, output_dimension, input_dimension,
        ) .* (2f0 * bound) .- bound,
        weight_rho = fill(rho, output_dimension, input_dimension),
        bias_mu = rand(rng, Float32, output_dimension) .*
            (2f0 * bound) .- bound,
        bias_rho = fill(rho, output_dimension),
    )
end

"""
    initialize_model(input_dimension, likelihood, config; seed)

Construct the two-hidden-layer, 50-unit Bayesian ReLU network used for the
UCI comparison in Tschantz et al. (2025), with the Bayes-by-Backprop
posterior of Blundell et al. (2015). Every weight and bias has an independent
Gaussian variational posterior.
"""
function initialize_model(
    input_dimension::Int,
    likelihood::String,
    config::BBBConfig;
    seed::Int,
)
    likelihood in ("homoscedastic", "heteroscedastic") ||
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    rng = StableRNG(seed)
    output_dimension = likelihood == "homoscedastic" ? 1 : 2
    return (
        layer1 = initialize_bayesian_layer(
            rng, input_dimension, config.hidden_units, config,
        ),
        layer2 = initialize_bayesian_layer(
            rng, config.hidden_units, config.hidden_units, config,
        ),
        output = initialize_bayesian_layer(
            rng, config.hidden_units, output_dimension, config,
        ),
    )
end

function sample_layer_epsilon(rng::AbstractRNG, layer)
    return (
        weight = randn(rng, Float32, size(layer.weight_mu)),
        bias = randn(rng, Float32, size(layer.bias_mu)),
    )
end

function sample_epsilon(params, rng::AbstractRNG)
    return (
        layer1 = sample_layer_epsilon(rng, params.layer1),
        layer2 = sample_layer_epsilon(rng, params.layer2),
        output = sample_layer_epsilon(rng, params.output),
    )
end

function sampled_layer(layer, epsilon)
    weight_std = stable_softplus.(layer.weight_rho)
    bias_std = stable_softplus.(layer.bias_rho)
    return (
        weight = layer.weight_mu .+ weight_std .* epsilon.weight,
        bias = layer.bias_mu .+ bias_std .* epsilon.bias,
    )
end

function sampled_affine(
    features::AbstractMatrix,
    layer,
    epsilon,
)
    sampled = sampled_layer(layer, epsilon)
    return features * transpose(sampled.weight) .+ transpose(sampled.bias)
end

"""
    forward_sample(params, epsilon, features, likelihood, config)

One reparameterized network draw. Returns standardized-target means and
strictly positive standardized-target observation scales. The
heteroscedastic output is `(m, ℓ)` with variance `exp(ℓ)`, matching equation
(8) of Wu et al. (2019), as referenced by the BPC uncertainty experiments.
"""
function forward_sample(
    params,
    epsilon,
    features::AbstractMatrix,
    likelihood::String,
    config::BBBConfig,
)
    hidden1 = max.(sampled_affine(features, params.layer1, epsilon.layer1), 0f0)
    hidden2 = max.(sampled_affine(hidden1, params.layer2, epsilon.layer2), 0f0)
    output = sampled_affine(hidden2, params.output, epsilon.output)
    means = vec(@view output[:, 1])
    scales = if likelihood == "homoscedastic"
        fill(exp(0.5f0 * Float32(config.homo_log_variance)), length(means))
    elseif likelihood == "heteroscedastic"
        log_variances = clamp.(
            vec(@view output[:, 2]),
            Float32(config.minimum_log_variance),
            Float32(config.maximum_log_variance),
        )
        exp.(0.5f0 .* log_variances)
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end
    return means, scales
end

function normal_logdensity_sum(
    values::AbstractArray,
    means,
    standard_deviations,
)
    return sum(
        -0.5f0 .* LOG2PI_F32 .-
        log.(standard_deviations) .-
        0.5f0 .* ((values .- means) ./ standard_deviations) .^ 2,
    )
end

function sampled_layer_complexity(
    layer,
    epsilon,
    prior_mean::Real,
    prior_std::Real,
)
    sampled = sampled_layer(layer, epsilon)
    weight_std = stable_softplus.(layer.weight_rho)
    bias_std = stable_softplus.(layer.bias_rho)
    log_q = normal_logdensity_sum(
        sampled.weight, layer.weight_mu, weight_std,
    ) + normal_logdensity_sum(
        sampled.bias, layer.bias_mu, bias_std,
    )
    prior_mean_f32 = Float32(prior_mean)
    prior_std_f32 = Float32(prior_std)
    log_p = normal_logdensity_sum(
        sampled.weight, prior_mean_f32, prior_std_f32,
    ) + normal_logdensity_sum(
        sampled.bias, prior_mean_f32, prior_std_f32,
    )
    return log_q - log_p
end

"""
    sampled_complexity_cost(params, epsilon, config)

Compute `log q(w|θ) - log p(w)` using exactly the same reparameterized weight
draw that is used by the likelihood. This is the Monte Carlo complexity term
in equation (2) of Blundell et al. (2015), rather than an analytic KL.
"""
function sampled_complexity_cost(params, epsilon, config::BBBConfig)
    layer_cost(layer, layer_epsilon) = sampled_layer_complexity(
        layer,
        layer_epsilon,
        config.prior_mean,
        config.prior_std,
    )
    return layer_cost(params.layer1, epsilon.layer1) +
        layer_cost(params.layer2, epsilon.layer2) +
        layer_cost(params.output, epsilon.output)
end

function gaussian_nll(
    targets::AbstractVector,
    means::AbstractVector,
    scales::AbstractVector,
)
    residual = (targets .- means) ./ scales
    return mean(0.5f0 .* (LOG2PI_F32 .+ residual .^ 2) .+ log.(scales))
end

"""
    elbo_loss(params, epsilon_samples, x, y, likelihood, config, n_training)

The mini-batch objective averages Blundell et al.'s sampled variational free
energy: Gaussian NLL plus sampled `log q(w|θ) - log p(w)`, normalized by the
number of training observations. Epsilon is sampled outside automatic
differentiation, making the stochastic objective reproducible.
"""
function elbo_loss(
    params,
    epsilon_samples::AbstractVector,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::BBBConfig,
    n_training::Int,
)
    n_training >= 1 || throw(ArgumentError("n_training must be positive"))
    sampled_free_energy = zero(Float32)
    for epsilon in epsilon_samples
        means, scales = forward_sample(
            params, epsilon, features, likelihood, config,
        )
        sampled_free_energy +=
            gaussian_nll(targets, means, scales) +
            sampled_complexity_cost(params, epsilon, config) / n_training
    end
    return sampled_free_energy / length(epsilon_samples)
end

gradient_sqnorm(::Nothing) = 0.0
gradient_sqnorm(array::AbstractArray) = sum(abs2, array)
gradient_sqnorm(tuple::NamedTuple) = sum(gradient_sqnorm, values(tuple))

scale_gradient(::Nothing, factor) = nothing
scale_gradient(array::AbstractArray, factor) = array .* factor
function scale_gradient(tuple::NamedTuple, factor)
    return NamedTuple{keys(tuple)}(map(value -> scale_gradient(value, factor), values(tuple)))
end

function clip_gradient(gradient, maximum_norm::Real)
    isinf(maximum_norm) && return gradient
    norm_value = sqrt(gradient_sqnorm(gradient))
    factor = norm_value > maximum_norm ? maximum_norm / norm_value : 1.0
    return scale_gradient(gradient, factor)
end

parameters_are_finite(::Nothing) = true
parameters_are_finite(array::AbstractArray) = all(isfinite, array)
parameters_are_finite(tuple::NamedTuple) = all(parameters_are_finite, values(tuple))
