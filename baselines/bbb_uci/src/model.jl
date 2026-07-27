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
    posterior_std::Real,
)
    bound = Float32(inv(sqrt(input_dimension)))
    rho = Float32(inverse_softplus(posterior_std))
    return (
        weight_mu = rand(
            rng, Float32, output_dimension, input_dimension,
        ) .* (2f0 * bound) .- bound,
        weight_rho = fill(rho, output_dimension, input_dimension),
        bias_mu = rand(rng, Float32, output_dimension) .* (2f0 * bound) .- bound,
        bias_rho = fill(rho, output_dimension),
    )
end

"""
    initialize_model(input_dimension, likelihood, config; seed)

Construct a 50-50 Bayesian ReLU network (or the configured hidden width).
Every weight and bias has an independent Gaussian variational posterior.
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
    common = (
        layer1 = initialize_bayesian_layer(
            rng, input_dimension, config.hidden_units,
            config.initial_posterior_std,
        ),
        layer2 = initialize_bayesian_layer(
            rng, config.hidden_units, config.hidden_units,
            config.initial_posterior_std,
        ),
        output = initialize_bayesian_layer(
            rng, config.hidden_units, output_dimension,
            config.initial_posterior_std,
        ),
    )
    if likelihood == "homoscedastic"
        initial_noise = max(1.0 - config.noise_floor, config.noise_floor)
        return merge(common, (
            noise_raw = Float32[inverse_softplus(initial_noise)],
        ))
    end
    return common
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

function sampled_affine(features::AbstractMatrix, layer, epsilon)
    weight_std = stable_softplus.(layer.weight_rho)
    bias_std = stable_softplus.(layer.bias_rho)
    weight = layer.weight_mu .+ weight_std .* epsilon.weight
    bias = layer.bias_mu .+ bias_std .* epsilon.bias
    return features * transpose(weight) .+ transpose(bias)
end

"""
    forward_sample(params, epsilon, features, likelihood, noise_floor)

One reparameterized network draw. Returns standardized-target means and
strictly positive standardized-target observation scales.
"""
function forward_sample(
    params,
    epsilon,
    features::AbstractMatrix,
    likelihood::String,
    noise_floor::Real,
)
    hidden1 = max.(sampled_affine(features, params.layer1, epsilon.layer1), 0f0)
    hidden2 = max.(sampled_affine(hidden1, params.layer2, epsilon.layer2), 0f0)
    output = sampled_affine(hidden2, params.output, epsilon.output)
    means = vec(@view output[:, 1])
    scales = if likelihood == "homoscedastic"
        fill(stable_softplus(params.noise_raw[1]) + Float32(noise_floor), length(means))
    elseif likelihood == "heteroscedastic"
        stable_softplus.(vec(@view output[:, 2])) .+ Float32(noise_floor)
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end
    return means, scales
end

function layer_gaussian_kl(layer, prior_std::Real)
    prior_variance = Float32(prior_std^2)
    weight_std = stable_softplus.(layer.weight_rho)
    bias_std = stable_softplus.(layer.bias_rho)
    weight_kl = 0.5f0 .* sum(
        (weight_std .^ 2 .+ layer.weight_mu .^ 2) ./ prior_variance .-
        1f0 .+ 2f0 .* (Float32(log(prior_std)) .- log.(weight_std)),
    )
    bias_kl = 0.5f0 .* sum(
        (bias_std .^ 2 .+ layer.bias_mu .^ 2) ./ prior_variance .-
        1f0 .+ 2f0 .* (Float32(log(prior_std)) .- log.(bias_std)),
    )
    return weight_kl + bias_kl
end

function gaussian_kl(params, prior_std::Real = 1.0)
    return layer_gaussian_kl(params.layer1, prior_std) +
        layer_gaussian_kl(params.layer2, prior_std) +
        layer_gaussian_kl(params.output, prior_std)
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

The mini-batch objective is mean Gaussian NLL plus the analytic global KL
divided by the number of training observations. Epsilon is sampled outside
automatic differentiation, making the stochastic objective reproducible.
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
    likelihood_term = zero(Float32)
    for epsilon in epsilon_samples
        means, scales = forward_sample(
            params, epsilon, features, likelihood, config.noise_floor,
        )
        likelihood_term += gaussian_nll(targets, means, scales)
    end
    likelihood_term /= length(epsilon_samples)
    return likelihood_term + gaussian_kl(params, config.prior_std) / n_training
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
