module DVIReactantBackend

using Enzyme
using LinearAlgebra
import Optimisers
import Reactant
using Random
using Statistics

import ..DVIUCI

mutable struct ReactantTrainingBackend
    params::Any
    optimizer_state::Any
    features::Any
    targets::Any
    likelihood::String
    config::DVIUCI.DVIConfig
    validation_features::Any
    validation_kernel::Any
end

const COMPILED_BATCH_KERNELS = Dict{Tuple, Any}()
const COMPILED_VALIDATION_KERNELS = Dict{Tuple, Any}()

function reactant_gaussian_cdf(values)
    scaled = DVIUCI.DVI_INV_SQRT2 .* values
    magnitude = abs.(scaled)
    t = 1f0 ./ (1f0 .+ 0.3275911f0 .* magnitude)
    polynomial =
        (((
            (1.061405429f0 .* t .- 1.453152027f0) .* t .+
            1.421413741f0
        ) .* t .- 0.284496736f0) .* t .+ 0.254829592f0) .* t
    erf_magnitude = 1f0 .- polynomial .* exp.(-magnitude .^ 2)
    sign = ifelse.(scaled .>= 0f0, 1f0, -1f0)
    return 0.5f0 .* (1f0 .+ sign .* erf_magnitude)
end

function DVIUCI.gaussian_cdf(
    values::Reactant.TracedRArray{T, N},
) where {T, N}
    return reactant_gaussian_cdf(values)
end

function DVIUCI.dvi_relu_delta(
    rho::Reactant.TracedRArray{T, N}, mu1, mu2,
) where {T, N}
    return reactant_gaussian_cdf(mu1) .* reactant_gaussian_cdf(mu2) .+
        DVIUCI.dvi_relu_g(rho, mu1, mu2)
end

function DVIUCI.batch_diagonal(
    diagonals::Reactant.TracedRArray{T, 2},
) where {T}
    batch, dimension = size(diagonals)
    identity_matrix = Matrix{T}(I, dimension, dimension)
    return reshape(diagonals, batch, dimension, 1) .*
        reshape(identity_matrix, 1, dimension, dimension)
end

function DVIUCI.batch_quadratic(
    weight_mean::Reactant.TracedRArray{T, 2},
    covariance::Reactant.TracedRArray{S, 3},
) where {T, S}
    batch, input_dimension, other_input_dimension = size(covariance)
    output_dimension, weight_input_dimension = size(weight_mean)
    input_dimension == other_input_dimension == weight_input_dimension ||
        throw(DimensionMismatch("weight and activation covariance disagree"))

    entries = map(CartesianIndices((output_dimension, output_dimension))) do index
        first_output, second_output = Tuple(index)
        first_weights = reshape(
            weight_mean[first_output, :], 1, input_dimension, 1,
        )
        second_weights = reshape(
            weight_mean[second_output, :], 1, 1, input_dimension,
        )
        return vec(sum(
            covariance .* first_weights .* second_weights; dims = (2, 3),
        ))
    end
    return reshape(
        reduce(hcat, vec(entries)), batch, output_dimension, output_dimension,
    )
end

function reactant_empirical_bayes_layer_kl(layer, config::DVIUCI.DVIConfig)
    parameter_count = Float32(
        length(layer.weight_mu) + length(layer.bias_mu),
    )
    second_moment_sum =
        sum(
            DVIUCI.safe_exp(2f0 .* layer.weight_log_std, config) .+
            layer.weight_mu .^ 2,
        ) +
        sum(
            DVIUCI.safe_exp(2f0 .* layer.bias_log_std, config) .+
            layer.bias_mu .^ 2,
        )
    log_std_sum =
        sum(layer.weight_log_std) + sum(layer.bias_log_std)
    degrees = parameter_count + 2f0 * Float32(config.eb_alpha) + 2f0
    regularized_second_moment =
        second_moment_sum + 2f0 * Float32(config.eb_beta)
    return 0.5f0 * (
        parameter_count * log(regularized_second_moment / degrees) +
        second_moment_sum * degrees / regularized_second_moment -
        (parameter_count + 2f0 * log_std_sum)
    )
end

function reactant_expected_log_likelihood(
    output,
    targets,
    likelihood::String,
    config::DVIUCI.DVIConfig,
)
    mean_prediction = vec(output.mean[:, 1])
    mean_variance = DVIUCI.output_covariance_entry(output, 1, 1, config)
    if likelihood == "heteroscedastic"
        log_variance = vec(output.mean[:, 2])
        log_variance_variance =
            DVIUCI.output_covariance_entry(output, 2, 2, config)
        mean_log_variance_covariance =
            DVIUCI.output_covariance_entry(output, 1, 2, config)
    elseif likelihood == "homoscedastic"
        log_variance =
            zero.(mean_prediction) .+ Float32(config.homo_log_variance)
        log_variance_variance = zero.(mean_prediction)
        mean_log_variance_covariance = zero.(mean_prediction)
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end

    precision_log_moment =
        -log_variance .+ 0.5f0 .* log_variance_variance
    precision_expectation = DVIUCI.safe_exp(
        precision_log_moment, config,
    )
    residual =
        mean_prediction .- mean_log_variance_covariance .- targets
    return -0.5f0 .* (
        DVIUCI.DVI_LOG2PI .+
        log_variance .+
        precision_expectation .* (mean_variance .+ residual .^ 2)
    )
end

function reactant_dvi_loss(
    params,
    features,
    targets,
    likelihood::String,
    config::DVIUCI.DVIConfig,
    n_training::Int,
    weight,
)
    output = DVIUCI.propagate_dvi(params, features, config)
    reconstruction = mean(reactant_expected_log_likelihood(
        output, targets, likelihood, config,
    ))
    kl =
        reactant_empirical_bayes_layer_kl(params.hidden, config) +
        reactant_empirical_bayes_layer_kl(params.output, config)
    return weight * kl / n_training - reconstruction
end

reactant_clip_gradient(::Nothing, limit::Real) = nothing
reactant_clip_gradient(array::AbstractArray, limit::Real) =
    clamp.(array, -Float32(limit), Float32(limit))
function reactant_clip_gradient(tuple::NamedTuple, limit::Real)
    return NamedTuple{keys(tuple)}(
        map(value -> reactant_clip_gradient(value, limit), values(tuple)),
    )
end

reactant_parameters_are_finite(::Nothing) = true
reactant_parameters_are_finite(array::AbstractArray) = all(isfinite, array)
function reactant_parameters_are_finite(tuple::NamedTuple)
    result = nothing
    for value in values(tuple)
        current = reactant_parameters_are_finite(value)
        result = result === nothing ? current : result & current
    end
    return something(result, true)
end

function reactant_training_step(
    params,
    optimizer_state,
    all_features,
    all_targets,
    indices,
    weight,
    likelihood::String,
    config::DVIUCI.DVIConfig,
)
    features = all_features[indices, :]
    targets = all_targets[indices]
    n_training = size(all_features, 1)
    objective = candidate -> reactant_dvi_loss(
        candidate,
        features,
        targets,
        likelihood,
        config,
        n_training,
        weight,
    )
    differentiated = Enzyme.gradient(
        Enzyme.ReverseWithPrimal, objective, params,
    )
    gradient = reactant_clip_gradient(
        differentiated.derivs[1], config.gradient_clip,
    )
    next_optimizer_state, next_params = Optimisers.update(
        optimizer_state, params, gradient,
    )
    numerical = DVIUCI.dvi_loss_clamp_statistics(
        params, features, likelihood, config,
    )
    return (
        params = next_params,
        optimizer_state = next_optimizer_state,
        loss = differentiated.val,
        gradient = gradient,
        loss_finite = isfinite(differentiated.val),
        gradient_finite = reactant_parameters_are_finite(gradient),
        params_finite = reactant_parameters_are_finite(next_params),
        numerical = numerical,
    )
end

function initialize_training_backend(
    params,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIUCI.DVIConfig,
)
    Reactant.set_default_backend(config.execution_device)
    device_params = Reactant.to_rarray(params)
    device_optimizer_state = Reactant.@jit Optimisers.setup(
        Optimisers.Adam(
            Float32(config.learning_rate),
            (0.9f0, 0.999f0),
            1f-8,
        ),
        device_params,
    )
    return ReactantTrainingBackend(
        device_params,
        device_optimizer_state,
        Reactant.to_rarray(features),
        Reactant.to_rarray(targets),
        likelihood,
        config,
        nothing,
        nothing,
    )
end

function batch_kernel_key(
    backend::ReactantTrainingBackend,
    batch_size::Int,
)
    config = backend.config
    return (
        config.execution_device,
        config.implementation_version,
        config.propagation,
        backend.likelihood,
        size(backend.features),
        batch_size,
        eltype(backend.params.hidden.weight_mu),
        config.hidden_units,
        config.safe_exp_min,
        config.safe_exp_max,
        config.eb_alpha,
        config.eb_beta,
        config.homo_log_variance,
        config.gradient_clip,
    )
end

function compile_batch_kernel!(
    backend::ReactantTrainingBackend,
    device_indices,
    device_weight,
)
    batch_size = length(device_indices)
    key = batch_kernel_key(backend, batch_size)
    return get!(COMPILED_BATCH_KERNELS, key) do
        likelihood = backend.likelihood
        config = backend.config
        kernel = (
            params,
            optimizer_state,
            features,
            targets,
            indices,
            weight,
        ) -> reactant_training_step(
            params,
            optimizer_state,
            features,
            targets,
            indices,
            weight,
            likelihood,
            config,
        )
        Reactant.@compile sync = true kernel(
            backend.params,
            backend.optimizer_state,
            backend.features,
            backend.targets,
            device_indices,
            device_weight,
        )
    end
end

function validation_kernel_key(
    backend::ReactantTrainingBackend,
    validation_features,
)
    config = backend.config
    return (
        config.execution_device,
        config.implementation_version,
        config.propagation,
        backend.likelihood,
        size(validation_features),
        eltype(backend.params.hidden.weight_mu),
        config.hidden_units,
        config.safe_exp_min,
        config.safe_exp_max,
    )
end

function compile_validation_kernel!(
    backend::ReactantTrainingBackend,
    validation_features,
)
    key = validation_kernel_key(backend, validation_features)
    return get!(COMPILED_VALIDATION_KERNELS, key) do
        config = backend.config
        kernel = (params, features) -> (
            output = DVIUCI.propagate_dvi(params, features, config),
            numerical = DVIUCI.parameter_variance_clamp_statistics(
                params, config,
            ),
        )
        Reactant.@compile sync = true kernel(
            backend.params, validation_features,
        )
    end
end

host_scalar(value::Number) = value
host_scalar(value) = only(Array(value))
host_bool(value) = Bool(host_scalar(value))

function materialize(value::NamedTuple)
    return NamedTuple{keys(value)}(map(materialize, values(value)))
end
materialize(value::Reactant.RArray) = Array(value)
materialize(value) = value

function training_backend_validation_output(
    backend::ReactantTrainingBackend,
    features::AbstractMatrix;
    tracker::DVIUCI.NumericalTracker,
)
    if backend.validation_features === nothing
        backend.validation_features = Reactant.to_rarray(features)
        backend.validation_kernel = compile_validation_kernel!(
            backend, backend.validation_features,
        )
    else
        size(backend.validation_features) == size(features) || throw(
            DimensionMismatch("validation feature shape changed"),
        )
    end
    observed = backend.validation_kernel(
        backend.params, backend.validation_features,
    )
    DVIUCI.accumulate_tracker!(tracker, observed.numerical)
    return materialize(observed.output)
end

function training_backend_epoch(
    backend::ReactantTrainingBackend,
    rng::AbstractRNG,
    epoch::Int,
    optimizer_step::Int;
    phase::String,
    tracker::DVIUCI.NumericalTracker,
)
    n_training = size(backend.features, 1)
    order = randperm(rng, n_training)
    weighted_loss = 0.0

    for (batch, first_index) in enumerate(
        1:backend.config.batch_size:n_training,
    )
        last_index = min(
            first_index + backend.config.batch_size - 1, n_training,
        )
        host_indices = Int32.(order[first_index:last_index])
        next_optimizer_step = optimizer_step + 1
        progress = DVIUCI.kl_progress(
            epoch, next_optimizer_step, backend.config,
        )
        host_weight = Float32(DVIUCI.kl_weight(progress, backend.config))
        device_indices = Reactant.to_rarray(host_indices)
        device_weight = Reactant.to_rarray(
            host_weight; track_numbers = Number,
        )
        kernel = compile_batch_kernel!(
            backend, device_indices, device_weight,
        )
        observed = kernel(
            backend.params,
            backend.optimizer_state,
            backend.features,
            backend.targets,
            device_indices,
            device_weight,
        )

        DVIUCI.accumulate_tracker!(tracker, observed.numerical)
        loss = Float64(host_scalar(observed.loss))
        current_params = backend.params
        if !host_bool(observed.loss_finite)
            DVIUCI.throw_numerical_error(
                "non-finite DVI loss",
                materialize(current_params),
                Array(backend.features)[host_indices, :],
                backend.likelihood,
                backend.config;
                phase = phase,
                epoch = epoch,
                batch = batch,
                optimizer_step = next_optimizer_step,
                loss = loss,
                gradient = materialize(observed.gradient),
                tracker = tracker,
            )
        end
        if !host_bool(observed.gradient_finite)
            DVIUCI.throw_numerical_error(
                "non-finite DVI gradient",
                materialize(current_params),
                Array(backend.features)[host_indices, :],
                backend.likelihood,
                backend.config;
                phase = phase,
                epoch = epoch,
                batch = batch,
                optimizer_step = next_optimizer_step,
                loss = loss,
                gradient = materialize(observed.gradient),
                tracker = tracker,
            )
        end
        if !host_bool(observed.params_finite)
            DVIUCI.throw_numerical_error(
                "non-finite model parameters",
                materialize(observed.params),
                Array(backend.features)[host_indices, :],
                backend.likelihood,
                backend.config;
                phase = phase,
                epoch = epoch,
                batch = batch,
                optimizer_step = next_optimizer_step,
                loss = loss,
                gradient = materialize(observed.gradient),
                tracker = tracker,
            )
        end

        backend.params = observed.params
        backend.optimizer_state = observed.optimizer_state
        optimizer_step = next_optimizer_step
        weighted_loss += length(host_indices) * loss
    end
    return backend, weighted_loss / n_training, optimizer_step
end

training_backend_parameters(backend::ReactantTrainingBackend) =
    materialize(backend.params)

end
