"""
Matrix-Normal--Wishart posterior for one affine BPC layer.

The natural parameters are stored as `(K, H, G, eta4)`, where
`K = V^-1`, `H = M*K`, `G = Psi^-1 + M*K*M'`, and
`eta4 = nu - output_dimension + augmented_input_dimension - 1`.
"""
mutable struct BPCLayer{A<:AbstractMatrix}
    input_dimension::Int
    output_dimension::Int
    prior_K::A
    prior_H::A
    prior_G::A
    prior_eta4::Float64
    K::A
    H::A
    G::A
    eta4::Float64
    M::A
    V::A
    Psi::A
    expected_precision::A
    nu::Float64
end

mutable struct BPCModel{L<:BPCLayer}
    layers::Vector{L}
    activate_input::Bool
    update_step::Int
    jitter::Float64
end

function symmetric_with_jitter(matrix::AbstractMatrix, jitter::Real)
    dimension = size(matrix, 1)
    identity = device_identity(matrix, dimension)
    return 0.5f0 .* (matrix .+ transpose(matrix)) .+
        Float32(jitter) .* identity
end

function refresh_moments!(layer::BPCLayer, jitter::Real)
    layer.V = stable_spd_inverse(layer.K, jitter)
    layer.M = layer.H * layer.V
    psi_inverse = layer.G - layer.M * transpose(layer.H)
    layer.Psi = stable_spd_inverse(psi_inverse, jitter)
    layer.nu = layer.eta4 + layer.output_dimension -
        (layer.input_dimension + 1) + 1
    layer.nu > layer.output_dimension - 1 || throw(ArgumentError(
        "Wishart degrees of freedom must exceed output_dimension - 1",
    ))
    layer.expected_precision = Float32(layer.nu) .* layer.Psi
    return layer
end

function initialize_layer(
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
    config::BPCConfig,
)
    augmented_input = input_dimension + 1
    bound = Float32(inv(sqrt(input_dimension)))
    initial_M_host = rand(
        rng, Float32, output_dimension, augmented_input,
    ) .* (2f0 * bound) .- bound
    backend = backend_value(config)
    initial_M = to_device(initial_M_host, backend)
    reference = initial_M
    identity_input = device_identity(reference, augmented_input)
    identity_output = device_identity(reference, output_dimension)

    prior_K = Float32(inv(config.prior_v_scale)) .* identity_input
    prior_H = similar(initial_M)
    fill!(prior_H, 0f0)
    prior_G = Float32(inv(config.prior_psi_scale)) .* identity_output
    prior_nu = output_dimension + config.prior_nu_offset
    prior_eta4 = prior_nu - output_dimension + augmented_input - 1

    K = copy(prior_K)
    H = initial_M * K
    G = prior_G + initial_M * K * transpose(initial_M)
    layer = BPCLayer(
        input_dimension,
        output_dimension,
        prior_K,
        prior_H,
        prior_G,
        prior_eta4,
        K,
        H,
        G,
        prior_eta4,
        initial_M,
        copy(prior_K),
        copy(prior_G),
        copy(prior_G),
        prior_nu,
    )
    return refresh_moments!(layer, config.posterior_jitter)
end

"""Create the two-hidden-layer ReLU network specified in BPC Appendix F.3."""
function initialize_model(
    input_dimension::Int,
    config::BPCConfig;
    seed::Int,
)
    rng = StableRNG(seed)
    dimensions = (input_dimension, config.hidden_units, config.hidden_units, 1)
    layers = [
        initialize_layer(rng, dimensions[index], dimensions[index + 1], config)
        for index in 1:(length(dimensions) - 1)
    ]
    return BPCModel(
        layers, config.activate_input, 0, config.posterior_jitter,
    )
end

relu(array) = max.(array, zero(eltype(array)))

function layer_nonlinearity(model::BPCModel, layer_index::Int, state)
    layer_index == 1 && !model.activate_input && return state
    return relu(state)
end

function augmented_layer_input(
    model::BPCModel,
    layer_index::Int,
    previous_state::AbstractMatrix,
)
    activated = layer_nonlinearity(model, layer_index, previous_state)
    return vcat(
        activated,
        device_ones(activated, 1, size(activated, 2)),
    )
end

function deterministic_forward_columns(model::BPCModel, features::AbstractMatrix)
    state = features
    for (index, layer) in enumerate(model.layers)
        state = layer.M * augmented_layer_input(model, index, state)
    end
    return state
end

function posterior_update!(
    model::BPCModel,
    states::AbstractVector{<:AbstractMatrix},
    kappa::Real,
    stat_scale::Real = 1,
)
    length(states) == length(model.layers) + 1 ||
        throw(DimensionMismatch("one state is required for every network level"))
    0 < kappa <= 1 || throw(ArgumentError("kappa must be in (0, 1]"))
    stat_scale > 0 || throw(ArgumentError("stat_scale must be positive"))
    blend = Float32(kappa)
    scale = Float32(stat_scale)

    for (index, layer) in enumerate(model.layers)
        inputs = augmented_layer_input(model, index, states[index])
        outputs = states[index + 1]
        size(inputs, 1) == layer.input_dimension + 1 ||
            throw(DimensionMismatch("layer input dimension mismatch"))
        size(outputs, 1) == layer.output_dimension ||
            throw(DimensionMismatch("layer output dimension mismatch"))
        size(inputs, 2) == size(outputs, 2) ||
            throw(DimensionMismatch("input and output batch sizes disagree"))

        target_K = layer.prior_K + scale .* (inputs * transpose(inputs))
        target_H = layer.prior_H + scale .* (outputs * transpose(inputs))
        target_G = layer.prior_G + scale .* (outputs * transpose(outputs))
        target_eta4 = layer.prior_eta4 + stat_scale * size(inputs, 2)

        layer.K = (1f0 - blend) .* layer.K .+ blend .* target_K
        layer.H = (1f0 - blend) .* layer.H .+ blend .* target_H
        layer.G = (1f0 - blend) .* layer.G .+ blend .* target_G
        layer.eta4 = (1 - kappa) * layer.eta4 + kappa * target_eta4
        refresh_moments!(layer, model.jitter)
    end
    model.update_step += 1
    return model
end

function latent_gradients(
    model::BPCModel,
    states::AbstractVector{<:AbstractMatrix},
)
    hidden_count = length(model.layers) - 1
    gradients = Vector{typeof(states[1])}(undef, hidden_count)
    for hidden_index in 1:hidden_count
        current_layer = model.layers[hidden_index]
        current_input = augmented_layer_input(
            model, hidden_index, states[hidden_index],
        )
        current_residual = states[hidden_index + 1] -
            current_layer.M * current_input
        current_gradient = current_layer.expected_precision * current_residual

        next_layer_index = hidden_index + 1
        next_layer = model.layers[next_layer_index]
        hidden_state = states[hidden_index + 1]
        next_input = augmented_layer_input(
            model, next_layer_index, hidden_state,
        )
        next_residual = states[hidden_index + 2] - next_layer.M * next_input
        mean_feedback = transpose(next_layer.M) *
            (next_layer.expected_precision * next_residual)
        covariance_feedback = Float32(next_layer.output_dimension) .*
            (next_layer.V * next_input)
        input_gradient = -mean_feedback + covariance_feedback
        nonlinear_gradient = @view input_gradient[1:next_layer.input_dimension, :]
        relu_derivative = hidden_state .> zero(eltype(hidden_state))
        gradients[hidden_index] = current_gradient .+
            relu_derivative .* nonlinear_gradient
    end
    return gradients
end

"""
Infer the hidden MAP states while clamping `z0=x` and `zL=y`.

The paper uses Adam for 10 hidden-state steps per batch. The implementation
keeps every matrix on the selected backend and computes all observations in
the batch together.
"""
function infer_latent_states(
    model::BPCModel,
    features::AbstractMatrix,
    targets::AbstractMatrix,
    config::BPCConfig,
)
    size(features, 1) == model.layers[1].input_dimension ||
        throw(DimensionMismatch("feature dimension mismatch"))
    size(targets, 1) == last(model.layers).output_dimension ||
        throw(DimensionMismatch("target dimension mismatch"))
    size(features, 2) == size(targets, 2) ||
        throw(DimensionMismatch("feature and target batch sizes disagree"))

    states = Vector{typeof(features)}()
    push!(states, features)
    state = features
    for index in 1:(length(model.layers) - 1)
        state = model.layers[index].M * augmented_layer_input(model, index, state)
        push!(states, state)
    end
    push!(states, targets)
    hidden_count = length(states) - 2
    hidden_count == 0 && return states

    first_moments = [zero(state) for state in states[2:(end - 1)]]
    second_moments = [zero(state) for state in states[2:(end - 1)]]
    beta1 = Float32(config.latent_adam_beta1)
    beta2 = Float32(config.latent_adam_beta2)
    learning_rate = Float32(config.latent_learning_rate)
    epsilon = Float32(config.latent_adam_epsilon)

    for step in 1:config.latent_steps
        gradients = latent_gradients(model, states)
        correction1 = 1f0 - beta1^step
        correction2 = 1f0 - beta2^step
        for index in 1:hidden_count
            gradient = gradients[index]
            first_moments[index] = beta1 .* first_moments[index] .+
                (1f0 - beta1) .* gradient
            second_moments[index] = beta2 .* second_moments[index] .+
                (1f0 - beta2) .* abs2.(gradient)
            first_unbiased = first_moments[index] ./ correction1
            second_unbiased = second_moments[index] ./ correction2
            states[index + 1] = states[index + 1] .-
                learning_rate .* first_unbiased ./ (sqrt.(second_unbiased) .+ epsilon)
        end
    end
    return states
end

function expected_energy(
    model::BPCModel,
    states::AbstractVector{<:AbstractMatrix},
)
    observations = size(first(states), 2)
    total = 0.0
    for (index, layer) in enumerate(model.layers)
        inputs = augmented_layer_input(model, index, states[index])
        residual = states[index + 1] - layer.M * inputs
        precision_term = sum(residual .* (layer.expected_precision * residual))
        weight_uncertainty = layer.output_dimension *
            sum(inputs .* (layer.V * inputs))
        total += 0.5 * (Float64(precision_term) + Float64(weight_uncertainty))
    end
    return total / observations
end

function host_layer(layer::BPCLayer)
    arrays = map(to_host, (
        layer.prior_K, layer.prior_H, layer.prior_G,
        layer.K, layer.H, layer.G, layer.M, layer.V,
        layer.Psi, layer.expected_precision,
    ))
    return BPCLayer(
        layer.input_dimension, layer.output_dimension,
        arrays[1], arrays[2], arrays[3], layer.prior_eta4,
        arrays[4], arrays[5], arrays[6], layer.eta4,
        arrays[7], arrays[8], arrays[9], arrays[10], layer.nu,
    )
end

function host_model(model::BPCModel)
    layers = [host_layer(layer) for layer in model.layers]
    return BPCModel(layers, model.activate_input, model.update_step, model.jitter)
end

function stabilized_symmetric(
    matrix::Matrix,
    jitter::Real;
    promote_to_float64::Bool = false,
)
    symmetrized = promote_to_float64 ?
        Matrix(0.5 .* (matrix .+ transpose(matrix))) :
        Matrix(0.5f0 .* (matrix .+ transpose(matrix)))
    element_type = eltype(symmetrized)
    scale = max(maximum(abs, symmetrized), one(element_type))
    fallback_floor = eps(element_type) * scale
    shift = element_type(jitter)
    for _ in 1:8
        candidate = Symmetric(symmetrized + shift * I)
        factor = cholesky(candidate; check = false)
        issuccess(factor) && return candidate
        shift = max(shift * element_type(10), fallback_floor)
    end
    throw(PosDefException(size(matrix, 1)))
end

function stable_spd_inverse(matrix::Matrix, jitter::Real)
    return Matrix(inv(stabilized_symmetric(matrix, jitter)))
end

function stable_spd_inverse(matrix::AbstractMatrix, jitter::Real)
    return inv(symmetric_with_jitter(matrix, jitter))
end

function stable_cholesky(matrix::AbstractMatrix, jitter::Real)
    candidate = stabilized_symmetric(
        Matrix(matrix), jitter; promote_to_float64 = true,
    )
    return cholesky(candidate; check = false)
end

function sample_layer_parameters(
    layer::BPCLayer{<:Matrix},
    rng::AbstractRNG,
    jitter::Real,
)
    output_dimension = layer.output_dimension
    input_dimension = layer.input_dimension + 1
    psi_factor = stable_cholesky(layer.Psi, jitter).L
    bartlett = zeros(Float32, output_dimension, output_dimension)
    for row in 1:output_dimension
        degrees = layer.nu - row + 1
        bartlett[row, row] = sqrt(Float32(rand(rng, Chisq(degrees))))
        for column in 1:(row - 1)
            bartlett[row, column] = randn(rng, Float32)
        end
    end
    precision = Matrix(psi_factor * bartlett * transpose(bartlett) *
        transpose(psi_factor))
    precision_factor = stable_cholesky(precision, jitter).L
    row_factor = inv(Matrix(transpose(precision_factor)))
    column_factor = stable_cholesky(layer.V, jitter).L
    noise = randn(rng, Float32, output_dimension, input_dimension)
    weight = layer.M + row_factor * noise * transpose(column_factor)
    covariance = inv(Symmetric(precision))
    return weight, Matrix(covariance)
end

function host_layer_input(
    model::BPCModel,
    layer_index::Int,
    previous_state::AbstractMatrix,
)
    activated = layer_index == 1 && !model.activate_input ?
        previous_state : max.(previous_state, zero(eltype(previous_state)))
    return vcat(
        activated,
        ones(eltype(activated), 1, size(activated, 2)),
    )
end

"""Draw complete MNW parameter samples and return Gaussian mixture components."""
function predictive_samples(
    model::BPCModel,
    features::AbstractMatrix,
    config::BPCConfig;
    seed::Int,
    n_samples::Int = config.eval_samples,
)
    n_samples >= 2 || throw(ArgumentError("n_samples must be at least 2"))
    posterior = host_model(model)
    feature_columns = Matrix{Float32}(transpose(features))
    observations = size(features, 1)
    means = Matrix{Float64}(undef, n_samples, observations)
    variances = Matrix{Float64}(undef, n_samples, observations)
    rng = StableRNG(seed)

    for sample in 1:n_samples
        state = feature_columns
        output_covariance = nothing
        for (index, layer) in enumerate(posterior.layers)
            weight, covariance = sample_layer_parameters(
                layer, rng, config.posterior_jitter,
            )
            state = weight * host_layer_input(posterior, index, state)
            output_covariance = covariance
        end
        means[sample, :] .= vec(state)
        variances[sample, :] .= output_covariance[1, 1]
    end
    return (means = means, variances = variances)
end
