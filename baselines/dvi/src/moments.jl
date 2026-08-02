const DVI_EPSILON = 1f-6
const DVI_HALF_EPSILON = DVI_EPSILON / 2f0
const DVI_LOG2PI = Float32(log(2pi))
const DVI_INV_SQRT2PI = Float32(inv(sqrt(2pi)))
const DVI_INV_SQRT2 = Float32(inv(sqrt(2)))
const DVI_TWOPI = Float32(2pi)

Base.@kwdef mutable struct NumericalTracker
    calls::Int = 0
    elements::Int = 0
    lower_clamps::Int = 0
    upper_clamps::Int = 0
end

function record_exp_clamps!(tracker::NumericalTracker, values, lower, upper)
    lower_clamps = values isa Number ? Int(values < lower) :
        count(value -> value < lower, values)
    upper_clamps = values isa Number ? Int(values > upper) :
        count(value -> value > upper, values)
    elements = values isa Number ? 1 : length(values)
    Zygote.ignore() do
        tracker.calls += 1
        tracker.elements += elements
        tracker.lower_clamps += lower_clamps
        tracker.upper_clamps += upper_clamps
    end
    return nothing
end

function safe_exp(
    values,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    tracker === nothing || record_exp_clamps!(
        tracker, values, config.safe_exp_min, config.safe_exp_max,
    )
    return exp.(clamp.(
        values,
        convert(eltype(values), config.safe_exp_min),
        convert(eltype(values), config.safe_exp_max),
    ))
end

function safe_exp(
    value::Number,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    tracker === nothing || record_exp_clamps!(
        tracker, value, config.safe_exp_min, config.safe_exp_max,
    )
    return exp(clamp(value, config.safe_exp_min, config.safe_exp_max))
end

function tracker_record(tracker::NumericalTracker)
    clamps = tracker.lower_clamps + tracker.upper_clamps
    return (
        calls = tracker.calls,
        elements = tracker.elements,
        lower_clamps = tracker.lower_clamps,
        upper_clamps = tracker.upper_clamps,
        clamp_count = clamps,
        clamp_rate = tracker.elements == 0 ? 0.0 : clamps / tracker.elements,
    )
end

function accumulate_tracker!(tracker::NumericalTracker, record)
    tracker.calls += Int(record.calls)
    tracker.elements += Int(record.elements)
    tracker.lower_clamps += Int(record.lower_clamps)
    tracker.upper_clamps += Int(record.upper_clamps)
    return tracker
end

standard_gaussian(x) = DVI_INV_SQRT2PI .* exp.(-0.5f0 .* x .^ 2)
gaussian_cdf(x) = 0.5f0 .* (1f0 .+ erf.(DVI_INV_SQRT2 .* x))
softrelu(x) = standard_gaussian(x) .+ x .* gaussian_cdf(x)

"""
    dvi_relu_g(rho, mu1, mu2)

Gaussian correction in Table 1 of Wu et al. (2019). This is a direct Julia
translation of `g` in the authors' `bayes_util.py`.
"""
function dvi_relu_g(rho, mu1, mu2)
    one_plus_sqrt = 1f0 .+ sqrt.(1f0 .- rho .^ 2)
    a = asin.(rho) .- rho ./ one_plus_sqrt
    safe_a = abs.(a) .+ DVI_HALF_EPSILON
    safe_rho = abs.(rho) .+ DVI_EPSILON
    amplitude = a ./ DVI_TWOPI
    sxx = safe_a .* one_plus_sqrt ./ safe_rho
    inverse_sxy = (asin.(rho) .- rho) ./ (safe_a .* safe_rho)
    return amplitude .* exp.(
        .-(mu1 .^ 2 .+ mu2 .^ 2) ./ (2f0 .* sxx) .+
        inverse_sxy .* mu1 .* mu2,
    )
end

dvi_relu_delta(rho, mu1, mu2) =
    gaussian_cdf(mu1) .* gaussian_cdf(mu2) .+
    dvi_relu_g(rho, mu1, mu2)

function batch_diagonal(diagonals::AbstractMatrix)
    batch, dimension = size(diagonals)
    identity_matrix = Matrix{eltype(diagonals)}(I, dimension, dimension)
    return reshape(diagonals, batch, dimension, 1) .*
        reshape(identity_matrix, 1, dimension, dimension)
end

function covariance_diagonal(covariance::AbstractArray{T, 3}) where {T}
    batch, dimension, other_dimension = size(covariance)
    dimension == other_dimension ||
        throw(DimensionMismatch("covariance matrices must be square"))
    flattened = reshape(covariance, batch, dimension * dimension)
    return flattened[:, 1:(dimension + 1):(dimension * dimension)]
end

function batch_quadratic(
    weight_mean::AbstractMatrix,
    covariance::AbstractArray{T, 3},
) where {T}
    size(weight_mean, 2) == size(covariance, 2) == size(covariance, 3) ||
        throw(DimensionMismatch("weight and activation covariance disagree"))
    @tullio result[batch, output, other_output] :=
        weight_mean[output, input] *
        covariance[batch, input, other_input] *
        weight_mean[other_output, other_input]
    return result
end

"""
    relu_moments_full(mean, covariance)

Approximate the mean and full covariance of `ReLU(a)` for Gaussian `a` using
equation (6) and Table 1 of Wu et al. (2019).
"""
function relu_moments_full(
    mean::AbstractMatrix,
    covariance::AbstractArray{T, 3},
) where {T}
    batch, dimension = size(mean)
    variance = max.(covariance_diagonal(covariance), zero(eltype(covariance)))
    standard_deviation = sqrt.(variance)
    standardized_mean = mean ./ (standard_deviation .+ DVI_EPSILON)

    denominator =
        reshape(standard_deviation, batch, dimension, 1) .*
        reshape(standard_deviation, batch, 1, dimension)
    rho = covariance ./ max.(denominator, DVI_EPSILON)
    rho_limit = 1f0 / (1f0 + DVI_EPSILON)
    rho = clamp.(rho, -rho_limit, rho_limit)
    mu1 = reshape(standardized_mean, batch, dimension, 1)
    mu2 = reshape(standardized_mean, batch, 1, dimension)

    relu_mean = standard_deviation .* softrelu(standardized_mean)
    relu_covariance =
        covariance .* dvi_relu_delta(rho, mu1, mu2)
    return (mean = relu_mean, covariance = relu_covariance)
end

"""
    relu_moments_diagonal(mean, variance)

Exact univariate ReLU mean and variance used by diagonal-DVI.
"""
function relu_moments_diagonal(
    mean::AbstractMatrix,
    variance::AbstractMatrix,
)
    safe_variance = max.(variance, zero(eltype(variance)))
    standard_deviation = sqrt.(safe_variance)
    standardized_mean = mean ./ (standard_deviation .+ DVI_EPSILON)
    density = standard_gaussian(standardized_mean)
    probability = gaussian_cdf(standardized_mean)
    softened = density .+ standardized_mean .* probability
    relu_mean = standard_deviation .* softened
    relu_variance = safe_variance .* (
        probability .+
        standardized_mean .* softened .-
        softened .^ 2
    )
    return (
        mean = relu_mean,
        variance = max.(relu_variance, zero(eltype(relu_variance))),
    )
end

function linear_certain_full(
    features::AbstractMatrix,
    layer,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    weight_variance = safe_exp(2f0 .* layer.weight_log_std, config, tracker)
    bias_variance = safe_exp(2f0 .* layer.bias_log_std, config, tracker)
    mean = features * transpose(layer.weight_mu) .+
        transpose(layer.bias_mu)
    variance_diagonal =
        (features .^ 2) * transpose(weight_variance) .+
        transpose(bias_variance)
    return (mean = mean, covariance = batch_diagonal(variance_diagonal))
end

function linear_full(
    activations,
    layer,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    weight_variance = safe_exp(2f0 .* layer.weight_log_std, config, tracker)
    bias_variance = safe_exp(2f0 .* layer.bias_log_std, config, tracker)
    input_variance = covariance_diagonal(activations.covariance)
    input_second_moment = input_variance .+ activations.mean .^ 2
    mean = activations.mean * transpose(layer.weight_mu) .+
        transpose(layer.bias_mu)
    independent_weight_diagonal =
        input_second_moment * transpose(weight_variance) .+
        transpose(bias_variance)
    correlated_term = batch_quadratic(
        layer.weight_mu, activations.covariance,
    )
    covariance =
        correlated_term .+ batch_diagonal(independent_weight_diagonal)
    return (mean = mean, covariance = covariance)
end

function linear_certain_diagonal(
    features::AbstractMatrix,
    layer,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    weight_variance = safe_exp(2f0 .* layer.weight_log_std, config, tracker)
    bias_variance = safe_exp(2f0 .* layer.bias_log_std, config, tracker)
    return (
        mean = features * transpose(layer.weight_mu) .+
            transpose(layer.bias_mu),
        variance = (features .^ 2) * transpose(weight_variance) .+
            transpose(bias_variance),
    )
end

function linear_diagonal(
    activations,
    layer,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    weight_variance = safe_exp(2f0 .* layer.weight_log_std, config, tracker)
    bias_variance = safe_exp(2f0 .* layer.bias_log_std, config, tracker)
    input_second_moment =
        activations.variance .+ activations.mean .^ 2
    return (
        mean = activations.mean * transpose(layer.weight_mu) .+
            transpose(layer.bias_mu),
        variance =
            input_second_moment * transpose(weight_variance) .+
            activations.variance * transpose(layer.weight_mu .^ 2) .+
            transpose(bias_variance),
    )
end
