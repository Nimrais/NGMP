using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using ExponentialFamily
using LinearAlgebra: Diagonal, dot
using ProbabilisticEnsembling
using Random
using RxInfer
using StableRNGs
using Statistics
using StatsPlots
using SurrogateModelling

# Two-level Softplus precision-gated PoE model.
#
# For every observation and hidden branch, `n_neurons` first-level experts share
# one `hidden_out`, so their Gaussian factors form a product of experts. The
# hidden output is transformed by `Softplus` into the precision of a final expert
# whose mean is computed directly from the original features. All final experts
# share `y`, forming a second product of experts.
#
# When `connect_hidden_to_y` is true, each completed hidden PoE also contributes
# one auxiliary Gaussian factor to `y`. That factor is deliberately created after
# the inner neuron loop, so it appears once per hidden branch rather than once per
# first-level expert.
#
# Expected entries in `priors`:
#
# - Scalars: `:τ_mean`, `:τ_gate`, `:τ_h`, and `:β_hidden`.
# - Per-hidden arrays: `:w_h`, `:β_out`, and, when requested, `:τ_hidden`.
# - Hidden-by-neuron arrays: `:w_mean` and `:w_a`.
@model function xor_softplus_hidden_poe(
    n_hidden,
    n_neurons,
    features,
    y,
    priors,
    softplus_dependencies,
    softplus_damping,
    normal_dependencies,
    normal_damping,
    connect_hidden_to_y,
)
    local w_mean, w_a, w_h
    local z_mean, za, γ_hidden, hidden_out
    local mean_contribution, gate_contribution
    local τ_mean, τ_gate, τ_h
    local β_hidden, β_out, τ_hidden

    τ_mean ~ priors[:τ_mean]
    τ_gate ~ priors[:τ_gate]
    τ_h ~ priors[:τ_h]
    β_hidden ~ priors[:β_hidden]

    for hidden in 1:n_hidden
        w_h[hidden] ~ priors[:w_h][hidden]
        β_out[hidden] ~ priors[:β_out][hidden]

        if connect_hidden_to_y
            τ_hidden[hidden] ~ priors[:τ_hidden][hidden]
        end

        for neuron in 1:n_neurons
            w_mean[hidden, neuron] ~ priors[:w_mean][hidden, neuron]
            w_a[hidden, neuron] ~ priors[:w_a][hidden, neuron]
        end
    end

    for observation in eachindex(y)
        for hidden in 1:n_hidden
            # The first-level Gaussian factors share this hidden output and
            # therefore form one PoE for each (hidden, observation) pair.
            for neuron in 1:n_neurons
                z_mean[hidden, neuron, observation] ~ softdot(
                    features[observation],
                    w_mean[hidden, neuron],
                    τ_mean,
                )

                za[hidden, neuron, observation] ~ softdot(
                    features[observation],
                    w_a[hidden, neuron],
                    τ_gate,
                ) where {
                    meta = LowRankMeta(),
                }

                γ_hidden[hidden, neuron, observation] ~
                    GammaShapeRate(1.0, β_hidden)

                γ_hidden[hidden, neuron, observation] ~ Softplus(
                    za[hidden, neuron, observation],
                ) where {
                    dependencies = softplus_dependencies,
                    meta = softplus_damping,
                }

                hidden_out[hidden, observation] ~ NormalMeanPrecision(
                    z_mean[hidden, neuron, observation],
                    γ_hidden[hidden, neuron, observation],
                ) where {
                    dependencies = normal_dependencies,
                    meta = normal_damping,
                }
            end

            # Optional deep supervision: exactly one factor per completed
            # hidden PoE, never one copy per contributing neuron.
            if connect_hidden_to_y
                y[observation] ~ NormalMeanPrecision(
                    hidden_out[hidden, observation],
                    τ_hidden[hidden],
                )
            end

            # The final expert mean depends on the original features, not on
            # hidden_out. The hidden output controls only its precision.
            mean_contribution[hidden, observation] ~ softdot(
                features[observation],
                w_h[hidden],
                τ_h,
            )

            gate_contribution[hidden, observation] ~
                GammaShapeRate(1.0, β_out[hidden])

            gate_contribution[hidden, observation] ~ Softplus(
                hidden_out[hidden, observation],
            ) where {
                dependencies = softplus_dependencies,
                meta = softplus_damping,
            }

            # Repeating this factor over `hidden` is intentional: the final
            # prediction is a PoE over the hidden-specific mean/gate pairs.
            y[observation] ~ NormalMeanPrecision(
                mean_contribution[hidden, observation],
                gate_contribution[hidden, observation],
            ) where {
                dependencies = normal_dependencies,
                meta = normal_damping,
            }
        end
    end
end

# -----------------------------------------------------------------------------
# Standalone pilot experiment
# -----------------------------------------------------------------------------

hidden_poe_env_int(name, default) =
    parse(Int, get(ENV, name, string(default)))
hidden_poe_env_float(name, default) =
    parse(Float64, get(ENV, name, string(default)))
hidden_poe_env_bool(name, default = false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

function hidden_poe_parse_ints(value)
    values = parse.(Int, strip.(split(value, ',')))
    isempty(values) && throw(ArgumentError("integer list must not be empty"))
    all(>(0), values) || throw(ArgumentError("integer list must be positive"))
    return Tuple(values)
end

function hidden_poe_parse_connections(value)
    parse_entry(entry) = begin
        normalized = lowercase(strip(entry))
        normalized in ("1", "true", "yes", "on") && return true
        normalized in ("0", "false", "no", "off") && return false
        throw(ArgumentError("invalid connection mode: $entry"))
    end
    values = parse_entry.(split(value, ','))
    isempty(values) && throw(ArgumentError("connection list must not be empty"))
    return Tuple(unique(values))
end

const HIDDEN_POE_SMOKE = hidden_poe_env_bool("XOR_HIDDEN_POE_SMOKE")
const HIDDEN_POE_CONFIG = (
    n_samples = hidden_poe_env_int(
        "N_SAMPLES",
        HIDDEN_POE_SMOKE ? 24 : 400,
    ),
    train_fraction = hidden_poe_env_float("TRAIN_FRACTION", 0.40),
    iterations = hidden_poe_env_int(
        "N_ITERATIONS",
        HIDDEN_POE_SMOKE ? 1 : 30,
    ),
    hidden_counts = hidden_poe_parse_ints(
        get(ENV, "HIDDEN_COUNTS", HIDDEN_POE_SMOKE ? "1" : "1,2,4"),
    ),
    neuron_counts = hidden_poe_parse_ints(
        get(ENV, "NEURON_COUNTS", HIDDEN_POE_SMOKE ? "1" : "2,4"),
    ),
    connections = hidden_poe_parse_connections(
        get(ENV, "CONNECT_HIDDEN_TO_Y", "false,true"),
    ),
    checkerboard_size = (2, 2),
    noise_std = hidden_poe_env_float("NOISE_STD", 0.10),
    data_seed = hidden_poe_env_int("DATA_SEED", 2_026),
    split_seed = hidden_poe_env_int("SPLIT_SEED", 2_027),
    prior_seed = hidden_poe_env_int("PRIOR_SEED", 42),
    mean_prior_precision = hidden_poe_env_float(
        "MEAN_PRIOR_PRECISION",
        1e-4,
    ),
    gate_prior_precision = hidden_poe_env_float(
        "GATE_PRIOR_PRECISION",
        1.0,
    ),
    mean_precision_shape = hidden_poe_env_float(
        "MEAN_PRECISION_SHAPE",
        1e4,
    ),
    mean_precision_rate = hidden_poe_env_float(
        "MEAN_PRECISION_RATE",
        1.0,
    ),
    gate_precision_shape = hidden_poe_env_float(
        "GATE_PRECISION_SHAPE",
        1e3,
    ),
    gate_precision_rate = hidden_poe_env_float(
        "GATE_PRECISION_RATE",
        1.0,
    ),
    gamma_rate_shape = hidden_poe_env_float("GAMMA_RATE_SHAPE", 10.0),
    gamma_rate_rate = hidden_poe_env_float("GAMMA_RATE_RATE", 10.0),
    auxiliary_precision_shape = hidden_poe_env_float(
        "AUXILIARY_PRECISION_SHAPE",
        10.0,
    ),
    auxiliary_precision_rate = hidden_poe_env_float(
        "AUXILIARY_PRECISION_RATE",
        40.0,
    ),
    softplus_alpha = hidden_poe_env_float("SOFTPLUS_ALPHA", 0.1),
    softplus_max_step = hidden_poe_env_float("SOFTPLUS_MAX_STEP", 0.25),
    normal_alpha = hidden_poe_env_float("NORMAL_ALPHA", 0.1),
    normal_max_step = hidden_poe_env_float("NORMAL_MAX_STEP", 0.5),
    grid_size = hidden_poe_env_int(
        "GRID_SIZE",
        HIDDEN_POE_SMOKE ? 16 : 60,
    ),
    showprogress = hidden_poe_env_bool("SHOW_PROGRESS", false),
    save_outputs = hidden_poe_env_bool("SAVE_OUTPUTS", !HIDDEN_POE_SMOKE),
    output_prefix = get(
        ENV,
        "OUTPUT_PREFIX",
        joinpath(@__DIR__, "..", "viz", "xor_softplus_hidden_poe_pilot"),
    ),
)

0 < HIDDEN_POE_CONFIG.train_fraction < 1 ||
    throw(ArgumentError("TRAIN_FRACTION must be in (0, 1)"))
HIDDEN_POE_CONFIG.n_samples > 1 ||
    throw(ArgumentError("N_SAMPLES must exceed one"))
HIDDEN_POE_CONFIG.iterations > 0 ||
    throw(ArgumentError("N_ITERATIONS must be positive"))
HIDDEN_POE_CONFIG.grid_size > 1 ||
    throw(ArgumentError("GRID_SIZE must exceed one"))

function hidden_poe_checkerboard_label(x1, x2, checkerboard_size)
    nx, ny = checkerboard_size
    nx > 0 && ny > 0 ||
        throw(ArgumentError("checkerboard dimensions must be positive"))
    cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
    cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
    return Float64(isodd(cell_x + cell_y))
end

function make_hidden_poe_dataset(; n, checkerboard_size, noise_std, seed)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    clean = hidden_poe_checkerboard_label.(
        x1,
        x2,
        Ref(checkerboard_size),
    )
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
    return (x1 = x1, x2 = x2, clean = clean, target = target)
end

hidden_poe_subset(data, indices) = (
    x1 = data.x1[indices],
    x2 = data.x2[indices],
    clean = data.clean[indices],
    target = data.target[indices],
)

function split_hidden_poe_dataset(data; train_fraction, seed)
    rng = StableRNG(seed)
    indices = randperm(rng, length(data.target))
    n_train = round(Int, train_fraction * length(indices))
    n_train = clamp(n_train, 1, length(indices) - 1)
    return (
        hidden_poe_subset(data, indices[1:n_train]),
        hidden_poe_subset(data, indices[(n_train + 1):end]),
    )
end

hidden_poe_features(data) = [
    [1.0, data.x1[index], data.x2[index]] for
    index in eachindex(data.target)
]

hidden_poe_invsoftplus(value::Real) = value + log1p(-exp(-value))
hidden_poe_stable_softplus(value::Real) =
    max(value, zero(value)) + log1p(exp(-abs(value)))

function hidden_poe_weight_prior(mean_vector, precision_matrix)
    return MvNormalWeightedMeanPrecision(
        precision_matrix * mean_vector,
        precision_matrix,
    )
end

function make_hidden_poe_priors(
    ;
    n_hidden,
    n_neurons,
    output_mean,
    prior_seed,
)
    rng = MersenneTwister(prior_seed)
    n_features = 3
    mean_precision = Diagonal(fill(
        HIDDEN_POE_CONFIG.mean_prior_precision,
        n_features,
    ))
    gate_precision = Diagonal(fill(
        HIDDEN_POE_CONFIG.gate_prior_precision,
        n_features,
    ))
    gate_intercept = hidden_poe_invsoftplus(1.0)

    mean_prior() = hidden_poe_weight_prior(
        [
            output_mean + 0.1 * randn(rng),
            0.25 * randn(rng),
            0.25 * randn(rng),
        ],
        mean_precision,
    )
    gate_prior() = hidden_poe_weight_prior(
        [
            gate_intercept + 0.1 * randn(rng),
            0.1 * randn(rng),
            0.1 * randn(rng),
        ],
        gate_precision,
    )

    return Dict{Symbol, Any}(
        :w_mean => [mean_prior() for _ in 1:n_hidden, _ in 1:n_neurons],
        :w_a => [gate_prior() for _ in 1:n_hidden, _ in 1:n_neurons],
        :w_h => [mean_prior() for _ in 1:n_hidden],
        :τ_mean => GammaShapeRate(
            HIDDEN_POE_CONFIG.mean_precision_shape,
            HIDDEN_POE_CONFIG.mean_precision_rate,
        ),
        :τ_gate => GammaShapeRate(
            HIDDEN_POE_CONFIG.gate_precision_shape,
            HIDDEN_POE_CONFIG.gate_precision_rate,
        ),
        :τ_h => GammaShapeRate(
            HIDDEN_POE_CONFIG.mean_precision_shape,
            HIDDEN_POE_CONFIG.mean_precision_rate,
        ),
        :β_hidden => GammaShapeRate(
            HIDDEN_POE_CONFIG.gamma_rate_shape,
            HIDDEN_POE_CONFIG.gamma_rate_rate,
        ),
        :β_out => [
            GammaShapeRate(
                HIDDEN_POE_CONFIG.gamma_rate_shape,
                HIDDEN_POE_CONFIG.gamma_rate_rate,
            ) for _ in 1:n_hidden
        ],
        :τ_hidden => [
            GammaShapeRate(
                HIDDEN_POE_CONFIG.auxiliary_precision_shape,
                HIDDEN_POE_CONFIG.auxiliary_precision_rate,
            ) for _ in 1:n_hidden
        ],
    )
end

const HIDDEN_POE_INITIAL_GATE_MEAN = 1.0
const HIDDEN_POE_INITIAL_GATE_VARIANCE = 0.01
const HIDDEN_POE_INITIAL_GATE_INPUT_MEAN =
    hidden_poe_invsoftplus(HIDDEN_POE_INITIAL_GATE_MEAN)
const HIDDEN_POE_INITIAL_GATE_INPUT_DERIVATIVE =
    1 - exp(-HIDDEN_POE_INITIAL_GATE_MEAN)
const HIDDEN_POE_INITIAL_GATE_INPUT_VARIANCE =
    HIDDEN_POE_INITIAL_GATE_VARIANCE /
    HIDDEN_POE_INITIAL_GATE_INPUT_DERIVATIVE^2

@initialization function xor_softplus_hidden_poe_initialization(
    priors,
    output_mean,
    connect_hidden_to_y,
)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(w_h) = deepcopy(priors[:w_h])
    q(z_mean) = NormalMeanVariance(output_mean, 1.0)
    q(za) = NormalMeanVariance(
        HIDDEN_POE_INITIAL_GATE_INPUT_MEAN,
        HIDDEN_POE_INITIAL_GATE_INPUT_VARIANCE,
    )
    q(γ_hidden) = GammaShapeRate(100.0, 100.0)
    q(hidden_out) = NormalMeanVariance(output_mean, 1.0)
    q(mean_contribution) = NormalMeanVariance(output_mean, 1.0)
    q(gate_contribution) = GammaShapeRate(100.0, 100.0)
    q(τ_mean) = priors[:τ_mean]
    q(τ_gate) = priors[:τ_gate]
    q(τ_h) = priors[:τ_h]
    q(β_hidden) = priors[:β_hidden]
    q(β_out) = deepcopy(priors[:β_out])
    if connect_hidden_to_y
        q(τ_hidden) = deepcopy(priors[:τ_hidden])
    end
end

function make_hidden_poe_dependencies()
    softplus_dependencies = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    normal_dependencies = NGMPDependencies(
        out = nothing,
        μ = nothing,
        τ = nothing,
        projection = TangentProjection(type = Unscented),
    )
    softplus_damping = DampingMeta(
        alpha = HIDDEN_POE_CONFIG.softplus_alpha,
        beta = 0.0,
        max_step = HIDDEN_POE_CONFIG.softplus_max_step,
    )
    normal_damping = DampingMeta(
        alpha = HIDDEN_POE_CONFIG.normal_alpha,
        beta = 0.0,
        max_step = HIDDEN_POE_CONFIG.normal_max_step,
    )
    return (
        softplus_dependencies = softplus_dependencies,
        normal_dependencies = normal_dependencies,
        softplus_damping = softplus_damping,
        normal_damping = normal_damping,
    )
end

function hidden_poe_returnvars(connect_hidden_to_y)
    if connect_hidden_to_y
        return (
            w_mean = KeepEach(),
            w_a = KeepEach(),
            w_h = KeepEach(),
            τ_hidden = KeepEach(),
            γ_hidden = KeepLast(),
            gate_contribution = KeepLast(),
        )
    end
    return (
        w_mean = KeepEach(),
        w_a = KeepEach(),
        w_h = KeepEach(),
        γ_hidden = KeepLast(),
        gate_contribution = KeepLast(),
    )
end

function run_hidden_poe_training(
    train_data,
    train_features;
    n_hidden,
    n_neurons,
    connect_hidden_to_y,
    iterations = HIDDEN_POE_CONFIG.iterations,
)
    output_mean = mean(train_data.target)
    priors = make_hidden_poe_priors(
        n_hidden = n_hidden,
        n_neurons = n_neurons,
        output_mean = output_mean,
        prior_seed = HIDDEN_POE_CONFIG.prior_seed,
    )
    controls = make_hidden_poe_dependencies()

    timed = @timed infer(
        model = xor_softplus_hidden_poe(
            n_hidden = n_hidden,
            n_neurons = n_neurons,
            priors = priors,
            softplus_dependencies = controls.softplus_dependencies,
            softplus_damping = controls.softplus_damping,
            normal_dependencies = controls.normal_dependencies,
            normal_damping = controls.normal_damping,
            connect_hidden_to_y = connect_hidden_to_y,
        ),
        data = (features = train_features, y = train_data.target),
        constraints = xor_softplus_hidden_poe_training_constraints(
            connect_hidden_to_y,
        ),
        initialization = xor_softplus_hidden_poe_initialization(
            priors,
            output_mean,
            connect_hidden_to_y,
        ),
        iterations = iterations,
        free_energy = false,
        showprogress = HIDDEN_POE_CONFIG.showprogress,
        returnvars = hidden_poe_returnvars(connect_hidden_to_y),
        options = (limit_stack_depth = 300,),
        disable_inference_error_hint = true,
    )

    return (
        result = timed.value,
        seconds = timed.time,
        bytes = timed.bytes,
        priors = priors,
        controls = controls,
    )
end

function hidden_poe_checked_mean(weights, values, label)
    denominator = sum(weights)
    if !(isfinite(denominator) && denominator > eps(Float64))
        throw(DomainError(denominator, "$label has no finite positive precision"))
    end
    numerator = dot(weights, values)
    isfinite(numerator) ||
        throw(DomainError(numerator, "$label has a non-finite numerator"))
    return numerator / denominator, numerator, denominator
end

function hidden_poe_plugin_prediction(
    feature,
    w_mean,
    w_a,
    w_h;
    τ_hidden = nothing,
)
    n_hidden, n_neurons = size(w_mean)
    size(w_a) == (n_hidden, n_neurons) ||
        throw(DimensionMismatch("w_mean and w_a shapes differ"))
    length(w_h) == n_hidden ||
        throw(DimensionMismatch("w_h length differs from n_hidden"))

    hidden_means = Vector{Float64}(undef, n_hidden)
    final_means = Vector{Float64}(undef, n_hidden)
    final_gates = Vector{Float64}(undef, n_hidden)

    for hidden in 1:n_hidden
        local_means = [
            dot(w_mean[hidden, neuron], feature) for neuron in 1:n_neurons
        ]
        local_gates = [
            hidden_poe_stable_softplus(dot(w_a[hidden, neuron], feature)) for
            neuron in 1:n_neurons
        ]
        hidden_means[hidden], _, _ = hidden_poe_checked_mean(
            local_gates,
            local_means,
            "hidden PoE",
        )
        final_means[hidden] = dot(w_h[hidden], feature)
        final_gates[hidden] =
            hidden_poe_stable_softplus(hidden_means[hidden])
    end

    head_mean, head_numerator, head_precision = hidden_poe_checked_mean(
        final_gates,
        final_means,
        "final head PoE",
    )

    graph_mean = head_mean
    if !isnothing(τ_hidden)
        length(τ_hidden) == n_hidden ||
            throw(DimensionMismatch("τ_hidden length differs from n_hidden"))
        auxiliary_precision = sum(τ_hidden)
        graph_precision = head_precision + auxiliary_precision
        graph_numerator = head_numerator + dot(τ_hidden, hidden_means)
        if !(
            isfinite(graph_precision) &&
            graph_precision > eps(Float64) &&
            isfinite(graph_numerator)
        )
            throw(DomainError(
                (graph_numerator, graph_precision),
                "full graph PoE is not finite and proper",
            ))
        end
        graph_mean = graph_numerator / graph_precision
    end

    return (
        head = head_mean,
        graph = graph_mean,
        hidden = hidden_means,
        final_gates = final_gates,
    )
end

function hidden_poe_iteration_parameters(result, iteration, connected)
    w_mean = mean.(result.posteriors[:w_mean][iteration])
    w_a = mean.(result.posteriors[:w_a][iteration])
    w_h = mean.(result.posteriors[:w_h][iteration])
    τ_hidden = connected ?
        Float64.(mean.(result.posteriors[:τ_hidden][iteration])) : nothing
    return (w_mean = w_mean, w_a = w_a, w_h = w_h, τ_hidden = τ_hidden)
end

function hidden_poe_prediction_metrics(predictions, data, constant_mse)
    nonfinite = count(value -> !isfinite(value), predictions)
    mse = nonfinite == 0 ? mean(abs2, predictions .- data.target) : NaN
    accuracy = nonfinite == 0 ? mean(
        (predictions .>= 0.5) .== (data.clean .>= 0.5),
    ) : NaN
    return (
        mse = mse,
        nmse = mse / constant_mse,
        accuracy = accuracy,
        minimum = minimum(predictions),
        maximum = maximum(predictions),
        nonfinite = nonfinite,
    )
end

function evaluate_hidden_poe_iteration(
    result,
    iteration,
    test_data,
    test_features,
    constant_mse,
    connected,
)
    parameters = hidden_poe_iteration_parameters(result, iteration, connected)
    point_predictions = [
        hidden_poe_plugin_prediction(
            feature,
            parameters.w_mean,
            parameters.w_a,
            parameters.w_h;
            τ_hidden = parameters.τ_hidden,
        ) for feature in test_features
    ]
    head = getproperty.(point_predictions, :head)
    graph = getproperty.(point_predictions, :graph)
    return (
        head = hidden_poe_prediction_metrics(head, test_data, constant_mse),
        graph = hidden_poe_prediction_metrics(graph, test_data, constant_mse),
        head_predictions = head,
        graph_predictions = graph,
        parameters = parameters,
    )
end

function hidden_poe_positive_diagnostics(result)
    γ = vec(result.posteriors[:γ_hidden])
    gates = vec(result.posteriors[:gate_contribution])
    γ_means = mean.(γ)
    gate_means = mean.(gates)
    all(isfinite, γ_means) && all(>(0), γ_means) ||
        error("γ_hidden contains a non-positive or non-finite mean")
    all(isfinite, gate_means) && all(>(0), gate_means) ||
        error("gate_contribution contains a non-positive or non-finite mean")
    return (
        minimum_hidden_gate = minimum(γ_means),
        maximum_hidden_gate = maximum(γ_means),
        minimum_final_gate = minimum(gate_means),
        maximum_final_gate = maximum(gate_means),
    )
end

function hidden_poe_grid()
    x = range(-2.0, 2.0; length = HIDDEN_POE_CONFIG.grid_size)
    y = range(-2.0, 2.0; length = HIDDEN_POE_CONFIG.grid_size)
    features = vec([
        [1.0, x_value, y_value] for y_value in y, x_value in x
    ])
    clean = [
        hidden_poe_checkerboard_label(
            x_value,
            y_value,
            HIDDEN_POE_CONFIG.checkerboard_size,
        ) for y_value in y, x_value in x
    ]
    return (x = x, y = y, features = features, clean = clean)
end

function hidden_poe_surface(parameters, grid, estimator)
    values = [
        getproperty(
            hidden_poe_plugin_prediction(
                feature,
                parameters.w_mean,
                parameters.w_a,
                parameters.w_h;
                τ_hidden = parameters.τ_hidden,
            ),
            estimator,
        ) for feature in grid.features
    ]
    return reshape(values, length(grid.y), length(grid.x))
end

function hidden_poe_failure_row(
    n_hidden,
    n_neurons,
    connected,
    constant_mse,
    error_message,
)
    return (
        n_hidden = n_hidden,
        n_neurons = n_neurons,
        connected = connected,
        estimator = connected ? "graph" : "head",
        iterations = HIDDEN_POE_CONFIG.iterations,
        seconds = NaN,
        allocated_gib = NaN,
        final_mse = NaN,
        best_mse = NaN,
        best_iteration = 0,
        constant_mse = constant_mse,
        final_nmse = NaN,
        best_nmse = NaN,
        final_accuracy = NaN,
        prediction_minimum = NaN,
        prediction_maximum = NaN,
        nonfinite = -1,
        learned = false,
        status = "error",
        error = error_message,
    )
end

function run_hidden_poe_size_sweep(train_data, test_data)
    train_features = hidden_poe_features(train_data)
    test_features = hidden_poe_features(test_data)
    constant_prediction = mean(train_data.target)
    constant_mse = mean(abs2, constant_prediction .- test_data.target)
    grid = hidden_poe_grid()
    learning_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    surfaces = Dict{Tuple{Bool, Int, Int, Symbol}, Matrix{Float64}}()

    for connected in HIDDEN_POE_CONFIG.connections
        for n_hidden in HIDDEN_POE_CONFIG.hidden_counts
            for n_neurons in HIDDEN_POE_CONFIG.neuron_counts
                label = "connected=$connected H=$n_hidden K=$n_neurons"
                println("Training $label")
                try
                    run = run_hidden_poe_training(
                        train_data,
                        train_features;
                        n_hidden = n_hidden,
                        n_neurons = n_neurons,
                        connect_hidden_to_y = connected,
                    )
                    diagnostics = hidden_poe_positive_diagnostics(run.result)
                    n_iterations = length(run.result.posteriors[:w_mean])
                    println(
                        "  completed in $(round(run.seconds; digits = 2))s; " *
                        "hidden gate range=$(round(diagnostics.minimum_hidden_gate; digits = 4))..$(round(diagnostics.maximum_hidden_gate; digits = 4)); " *
                        "final gate range=$(round(diagnostics.minimum_final_gate; digits = 4))..$(round(diagnostics.maximum_final_gate; digits = 4))",
                    )

                    iteration_evaluations = Vector{Any}(undef, n_iterations)
                    for iteration in 1:n_iterations
                        evaluation = evaluate_hidden_poe_iteration(
                            run.result,
                            iteration,
                            test_data,
                            test_features,
                            constant_mse,
                            connected,
                        )
                        iteration_evaluations[iteration] = evaluation
                        estimators = connected ? (:head, :graph) : (:head,)
                        for estimator in estimators
                            metrics = getproperty(evaluation, estimator)
                            push!(learning_rows, (
                                n_hidden = n_hidden,
                                n_neurons = n_neurons,
                                connected = connected,
                                estimator = String(estimator),
                                iteration = iteration,
                                mse = metrics.mse,
                                nmse = metrics.nmse,
                                accuracy = metrics.accuracy,
                            ))
                        end
                    end

                    estimators = connected ? (:head, :graph) : (:head,)
                    final_evaluation = last(iteration_evaluations)
                    for estimator in estimators
                        metric_history = [
                            getproperty(evaluation, estimator) for
                            evaluation in iteration_evaluations
                        ]
                        mse_history = getproperty.(metric_history, :mse)
                        best_iteration = argmin(mse_history)
                        final_metrics = last(metric_history)
                        best_metrics = metric_history[best_iteration]
                        push!(summary_rows, (
                            n_hidden = n_hidden,
                            n_neurons = n_neurons,
                            connected = connected,
                            estimator = String(estimator),
                            iterations = n_iterations,
                            seconds = run.seconds,
                            allocated_gib = run.bytes / 2.0^30,
                            final_mse = final_metrics.mse,
                            best_mse = best_metrics.mse,
                            best_iteration = best_iteration,
                            constant_mse = constant_mse,
                            final_nmse = final_metrics.nmse,
                            best_nmse = best_metrics.nmse,
                            final_accuracy = final_metrics.accuracy,
                            prediction_minimum = final_metrics.minimum,
                            prediction_maximum = final_metrics.maximum,
                            nonfinite = final_metrics.nonfinite,
                            learned = best_metrics.mse < constant_mse,
                            status = "ok",
                            error = "",
                        ))
                        surfaces[(connected, n_hidden, n_neurons, estimator)] =
                            hidden_poe_surface(
                                final_evaluation.parameters,
                                grid,
                                estimator,
                            )
                    end
                catch exception
                    message = sprint(showerror, exception, catch_backtrace())
                    println(stderr, "  ERROR: $message")
                    push!(
                        summary_rows,
                        hidden_poe_failure_row(
                            n_hidden,
                            n_neurons,
                            connected,
                            constant_mse,
                            message,
                        ),
                    )
                end
            end
        end
    end

    return (
        summary = DataFrame(summary_rows),
        learning = DataFrame(learning_rows),
        surfaces = surfaces,
        grid = grid,
        constant_mse = constant_mse,
    )
end

function save_hidden_poe_learning_plot(results, connected, estimator)
    subset = filter(
        row -> row.connected == connected && row.estimator == String(estimator),
        results.learning,
    )
    nrow(subset) == 0 && return nothing
    figure = plot(
        xlabel = "Iteration",
        ylabel = "Test MSE (cheap mean)",
        title = "Hidden PoE connected=$connected estimator=$estimator",
        legend = :topright,
    )
    hline!(
        figure,
        [results.constant_mse];
        color = :black,
        linestyle = :dash,
        label = "constant",
    )
    for n_hidden in HIDDEN_POE_CONFIG.hidden_counts
        for n_neurons in HIDDEN_POE_CONFIG.neuron_counts
            curve = filter(
                row -> row.n_hidden == n_hidden && row.n_neurons == n_neurons,
                subset,
            )
            nrow(curve) == 0 && continue
            sort!(curve, :iteration)
            plot!(
                figure,
                curve.iteration,
                curve.mse;
                linewidth = 2,
                label = "H=$n_hidden K=$n_neurons",
            )
        end
    end
    filename = HIDDEN_POE_CONFIG.output_prefix *
               "_learning_connected_$(connected)_$(estimator).png"
    savefig(figure, filename)
    return filename
end

function save_hidden_poe_surface_plot(results, connected, estimator)
    entries = [
        (n_hidden, n_neurons) for n_hidden in HIDDEN_POE_CONFIG.hidden_counts for
        n_neurons in HIDDEN_POE_CONFIG.neuron_counts if haskey(
            results.surfaces,
            (connected, n_hidden, n_neurons, estimator),
        )
    ]
    isempty(entries) && return nothing
    matrices = [
        results.surfaces[(connected, n_hidden, n_neurons, estimator)] for
        (n_hidden, n_neurons) in entries
    ]
    lower = min(0.0, minimum(minimum, matrices))
    upper = max(1.0, maximum(maximum, matrices))
    lower == upper && (upper = nextfloat(upper))
    panels = map(zip(entries, matrices)) do ((n_hidden, n_neurons), surface)
        panel = contourf(
            results.grid.x,
            results.grid.y,
            surface;
            color = :RdBu,
            levels = 20,
            clims = (lower, upper),
            xlabel = "x1",
            ylabel = "x2",
            title = "H=$n_hidden K=$n_neurons",
            linewidth = 0,
            aspect_ratio = :equal,
        )
        contour!(
            panel,
            results.grid.x,
            results.grid.y,
            surface;
            levels = [0.5],
            color = :black,
            linewidth = 1.5,
            colorbar = false,
        )
        panel
    end
    columns = min(3, length(panels))
    rows = cld(length(panels), columns)
    figure = plot(
        panels...;
        layout = (rows, columns),
        size = (420 * columns, 390 * rows),
        plot_title = "Cheap mean connected=$connected estimator=$estimator",
    )
    filename = HIDDEN_POE_CONFIG.output_prefix *
               "_surfaces_connected_$(connected)_$(estimator).png"
    savefig(figure, filename)
    return filename
end

function save_hidden_poe_outputs(results)
    HIDDEN_POE_CONFIG.save_outputs || return String[]
    mkpath(dirname(HIDDEN_POE_CONFIG.output_prefix))
    summary_path = HIDDEN_POE_CONFIG.output_prefix * "_summary.csv"
    learning_path = HIDDEN_POE_CONFIG.output_prefix * "_learning.csv"
    CSV.write(summary_path, results.summary)
    CSV.write(learning_path, results.learning)
    paths = [summary_path, learning_path]
    for connected in HIDDEN_POE_CONFIG.connections
        estimators = connected ? (:head, :graph) : (:head,)
        for estimator in estimators
            learning_plot = save_hidden_poe_learning_plot(
                results,
                connected,
                estimator,
            )
            surface_plot = save_hidden_poe_surface_plot(
                results,
                connected,
                estimator,
            )
            isnothing(learning_plot) || push!(paths, learning_plot)
            isnothing(surface_plot) || push!(paths, surface_plot)
        end
    end
    return paths
end

function print_hidden_poe_summary(results)
    println("\nConstant predictor test MSE: $(results.constant_mse)")
    if nrow(results.summary) == 0
        println("No completed runs")
        return
    end
    primary = filter(results.summary) do row
        row.status == "ok" &&
            ((!row.connected && row.estimator == "head") ||
             (row.connected && row.estimator == "graph"))
    end
    sort!(primary, :best_mse)
    println("\nPrimary cheap-mean results (graph is primary when connected):")
    show(
        stdout,
        primary[
            :,
            [
                :connected,
                :n_hidden,
                :n_neurons,
                :final_mse,
                :best_mse,
                :best_iteration,
                :final_nmse,
                :final_accuracy,
                :learned,
                :seconds,
            ],
        ];
        allrows = true,
        allcols = true,
    )
    println()
end

function main_hidden_poe_experiment()
    println("Hidden PoE pilot configuration:")
    println(HIDDEN_POE_CONFIG)
    dataset = make_hidden_poe_dataset(
        n = HIDDEN_POE_CONFIG.n_samples,
        checkerboard_size = HIDDEN_POE_CONFIG.checkerboard_size,
        noise_std = HIDDEN_POE_CONFIG.noise_std,
        seed = HIDDEN_POE_CONFIG.data_seed,
    )
    train_data, test_data = split_hidden_poe_dataset(
        dataset;
        train_fraction = HIDDEN_POE_CONFIG.train_fraction,
        seed = HIDDEN_POE_CONFIG.split_seed,
    )
    println(
        "Dataset: $(length(train_data.target)) train / " *
        "$(length(test_data.target)) test",
    )
    results = run_hidden_poe_size_sweep(train_data, test_data)
    print_hidden_poe_summary(results)
    paths = save_hidden_poe_outputs(results)
    isempty(paths) || println("\nSaved outputs:\n", join(paths, '\n'))
    return results
end

@constraints function xor_softplus_hidden_poe_training_constraints(connect_hidden_to_y)
    if connect_hidden_to_y
      q(
          w_mean,
          w_a,
          w_h,
          z_mean,
          za,
          γ_hidden,
          hidden_out,
          mean_contribution,
          gate_contribution,
          τ_mean,
          τ_gate,
          τ_h,
          β_hidden,
          β_out,
          τ_hidden,
      ) =
          q(w_mean) *
          q(w_a) *
          q(w_h) *
          q(
              z_mean,
              za,
              γ_hidden,
              hidden_out,
              mean_contribution,
              gate_contribution,
          ) *
          q(τ_mean) *
          q(τ_gate) *
          q(τ_h) *
          q(β_hidden) *
          q(β_out) *
          q(τ_hidden)
    else
        q(
          w_mean,
          w_a,
          w_h,
          z_mean,
          za,
          γ_hidden,
          hidden_out,
          mean_contribution,
          gate_contribution,
          τ_mean,
          τ_gate,
          τ_h,
          β_hidden,
          β_out
      ) =
          q(w_mean) *
          q(w_a) *
          q(w_h) *
          q(
              z_mean,
              za,
              γ_hidden,
              hidden_out,
              mean_contribution,
              gate_contribution,
          ) *
          q(τ_mean) *
          q(τ_gate) *
          q(τ_h) *
          q(β_hidden) *
          q(β_out)
    end 

    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()
    q(w_h)::MomentForm()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_hidden_poe_experiment()
end
