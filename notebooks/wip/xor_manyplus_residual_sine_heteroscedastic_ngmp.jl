# Hierarchical observation-noise experiment for the additive ManyPlus XOR model.
#
# The study separates three questions which are easy to conflate in a single
# predictive-variance heatmap:
#
#   1. Does a variance head stay flat when observation noise is homoscedastic?
#   2. Can it recover noise concentrated near the XOR class borders?
#   3. Can it localise a smooth high-noise region in the top-right corner?
#
# Two learning schedules are compared.  `joint` resets the mean hierarchy to
# its original priors and learns mean and variance together.  `fixed_mean`
# first learns the constant-noise mean model and then calibrates a variance
# hierarchy against its frozen predictive means.  The latter is deliberately
# labelled empirical Bayes: the observations are used once for the mean fit
# and again for residual calibration.
#
# The mean head includes a learned Normal intercept.  Its per-observation
# identity output is appended directly to ManyPlus, without passing through
# ResidualSine.  This separates the global target level from the nonlinear
# basis coefficients.
#
# Every hierarchy run starts from the constant model's learned average
# precision.  If gamma_bar is that precision, the Squareplus score intercept is
#
#       s0 = inverse_squareplus(gamma_bar) = gamma_bar - inv(gamma_bar),
#
# and all deviation-head output means are initialised at zero.  Thus the new
# model nests the old constant-noise model at its initial mean.
#
# Run a cheap graph audit (the default):
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_manyplus_residual_sine_heteroscedastic_ngmp.jl
#
# Staged studies:
#   XOR_HETERO_STAGE=screen julia --project=. experiments/xor_manyplus_residual_sine_heteroscedastic_ngmp.jl
#   XOR_HETERO_STAGE=full   julia --project=. experiments/xor_manyplus_residual_sine_heteroscedastic_ngmp.jl
#   XOR_HETERO_STAGE=all    julia --project=. experiments/xor_manyplus_residual_sine_heteroscedastic_ngmp.jl
#
# Useful knobs:
#   XOR_HETERO_TASKS=constant,border,corner
#   XOR_HETERO_NOISE_NEURONS=4 XOR_HETERO_KAPPA=50
#   XOR_HETERO_GRID_SIZE=32
#   XOR_HETERO_GRID_X_MIN=-4 XOR_HETERO_GRID_X_MAX=4
#   XOR_HETERO_GRID_Y_MIN=-3 XOR_HETERO_GRID_Y_MAX=3

ENV["GKSwstype"] = "100"

using SurrogateModelling
using RxInfer
using StableRNGs
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using Plots
using Distributions: Normal, quantile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

hetero_env_int(name, default) = parse(Int, get(ENV, name, string(default)))
hetero_env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
hetero_env_bool(name, default) = lowercase(strip(get(
    ENV, name, default ? "true" : "false",
))) in ("1", "true", "yes", "on")

function hetero_env_symbols(name, default)
    tokens = strip.(split(get(ENV, name, join(string.(default), ',')), ','))
    values = Tuple(Symbol(lowercase(token)) for token in tokens if !isempty(token))
    isempty(values) && throw(ArgumentError("$name must not be empty"))
    return values
end

function hetero_env_ints(name, default)
    tokens = strip.(split(get(ENV, name, join(string.(default), ',')), ','))
    values = Tuple(parse(Int, token) for token in tokens if !isempty(token))
    isempty(values) && throw(ArgumentError("$name must not be empty"))
    return values
end

function heteroscedastic_config()
    stage = Symbol(lowercase(get(ENV, "XOR_HETERO_STAGE", "smoke")))
    stage in (:smoke, :screen, :full, :all) || throw(ArgumentError(
        "XOR_HETERO_STAGE must be smoke, screen, full, or all",
    ))
    smoke = stage === :smoke
    screen = stage === :screen
    default_samples = smoke ? 96 : 800
    default_batches = smoke ? 2 : 8
    default_iterations = smoke ? 2 : (screen ? 60 : 125)
    default_stop_after = smoke ? 1 : (screen ? 40 : 75)
    default_tasks = smoke ? (:constant,) : (:constant, :border, :corner)
    default_seeds = stage in (:full, :all) ? (2_026, 2_027, 2_028) : (2_026,)

    tasks = hetero_env_symbols("XOR_HETERO_TASKS", default_tasks)
    all(task -> task in (:constant, :border, :corner), tasks) ||
        throw(ArgumentError("XOR_HETERO_TASKS accepts constant,border,corner"))

    return (
        stage = stage,
        n_samples = hetero_env_int("XOR_HETERO_N_SAMPLES", default_samples),
        n_neurons = hetero_env_int("XOR_HETERO_MEAN_NEURONS", 8),
        n_training_batches = hetero_env_int(
            "XOR_HETERO_BATCHES", default_batches,
        ),
        max_batch_iterations = hetero_env_int(
            "XOR_HETERO_ITERATIONS", default_iterations,
        ),
        stop_after_iteration = hetero_env_int(
            "XOR_HETERO_STOP_AFTER", default_stop_after,
        ),
        stop_atol = 0.0,
        stop_rtol = 1e-6,
        train_fraction = 0.40,
        validation_fraction = 0.20,
        tasks = tasks,
        seeds = hetero_env_ints("XOR_HETERO_SEEDS", default_seeds),
        split_seed = hetero_env_int("XOR_HETERO_SPLIT_SEED", 2_027),
        prior_seed = hetero_env_int("XOR_HETERO_PRIOR_SEED", 42),
        base_noise_std = hetero_env_float("XOR_HETERO_BASE_STD", 0.10),
        elevated_noise_std = hetero_env_float("XOR_HETERO_HIGH_STD", 0.30),
        border_width = hetero_env_float("XOR_HETERO_BORDER_WIDTH", 0.25),
        corner_x_threshold = hetero_env_float("XOR_HETERO_CORNER_X", 0.0),
        corner_y_threshold = hetero_env_float("XOR_HETERO_CORNER_Y", 0.0),
        corner_transition = hetero_env_float(
            "XOR_HETERO_CORNER_TRANSITION", 0.25,
        ),
        phi_rho = 0.9,
        phi_omega = 1.0,
        w_prior_scale = 2.2,
        w_prior_variance = 0.005,
        bias_prior_variance = 0.5,
        v_prior_scale = 0.5,
        v_prior_variance = 0.01,
        mean_intercept_prior_mean = hetero_env_float(
            "XOR_HETERO_MEAN_INTERCEPT_MEAN",
            0.5,
        ),
        mean_intercept_prior_variance = hetero_env_float(
            "XOR_HETERO_MEAN_INTERCEPT_VARIANCE",
            0.001,
        ),
        tau_prior = (1e3, 1.0),
        tau_c_prior = (1e4, 1.0),
        obs_noise_prior = (100.0, 1.0),
        mean_ngmp_alpha = hetero_env_float("XOR_HETERO_MEAN_ALPHA", 0.05),
        ngmp_beta = 0.0,
        mean_ngmp_max_step = 1.0,
        noise_w_prior_scale = hetero_env_float(
            "XOR_HETERO_NOISE_W_SCALE", 1.2,
        ),
        noise_w_prior_variance = hetero_env_float(
            "XOR_HETERO_NOISE_W_VARIANCE", 0.05,
        ),
        noise_bias_prior_variance = 0.5,
        noise_v_prior_variance = hetero_env_float(
            "XOR_HETERO_NOISE_V_VARIANCE", 25.0,
        ),
        noise_v_initial_variance = 1e-10,
        noise_intercept_prior_variance = hetero_env_float(
            "XOR_HETERO_INTERCEPT_VARIANCE", 25.0,
        ),
        noise_intercept_initial_variance = 1e-10,
        noise_tau_prior = (1e3, 1.0),
        noise_tau_c_prior = (1e4, 1.0),
        beta_prior_shape = hetero_env_float(
            "XOR_HETERO_BETA_PRIOR_SHAPE", 100.0,
        ),
        squareplus_alpha = hetero_env_float(
            "XOR_HETERO_SQUAREPLUS_ALPHA", 0.05,
        ),
        squareplus_max_step = hetero_env_float(
            "XOR_HETERO_SQUAREPLUS_MAX_STEP", 1.0,
        ),
        likelihood_alpha = hetero_env_float(
            "XOR_HETERO_LIKELIHOOD_ALPHA", 0.05,
        ),
        likelihood_max_step = hetero_env_float(
            "XOR_HETERO_LIKELIHOOD_MAX_STEP", 0.5,
        ),
        prediction_iterations = hetero_env_int(
            "XOR_HETERO_PREDICTION_ITERATIONS", smoke ? 2 : 8,
        ),
        prediction_batch_size = hetero_env_int(
            "XOR_HETERO_PREDICTION_BATCH", 512,
        ),
        grid_size = hetero_env_int("XOR_HETERO_GRID_SIZE", smoke ? 8 : 32),
        grid_x_min = hetero_env_float("XOR_HETERO_GRID_X_MIN", -2.0),
        grid_x_max = hetero_env_float("XOR_HETERO_GRID_X_MAX", 2.0),
        grid_y_min = hetero_env_float("XOR_HETERO_GRID_Y_MIN", -2.0),
        grid_y_max = hetero_env_float("XOR_HETERO_GRID_Y_MAX", 2.0),
        screen_specs = ((2, 10.0), (2, 50.0), (4, 10.0), (4, 50.0)),
        full_noise_neurons = hetero_env_int(
            "XOR_HETERO_NOISE_NEURONS", 4,
        ),
        full_kappa = hetero_env_float("XOR_HETERO_KAPPA", 50.0),
        save_outputs = hetero_env_bool("SAVE_OUTPUTS", true),
        save_plots = hetero_env_bool(
            "SAVE_PLOTS", stage in (:full, :all),
        ),
        output_dir = get(
            ENV,
            "OUTPUT_DIR",
            joinpath(@__DIR__, "xor_manyplus_residual_sine_heteroscedastic_output"),
        ),
    )
end

function validate_config(config)
    iseven(config.n_neurons) ||
        throw(ArgumentError("mean paired priors require an even neuron count"))
    config.n_samples >= 12 || throw(ArgumentError("n_samples must be at least 12"))
    0 < config.train_fraction < 1 || throw(ArgumentError("bad train fraction"))
    0 < config.validation_fraction < 1 ||
        throw(ArgumentError("bad validation fraction"))
    config.train_fraction + config.validation_fraction < 1 ||
        throw(ArgumentError("train + validation fractions must be below one"))
    1 <= config.n_training_batches <= round(Int, config.train_fraction * config.n_samples) ||
        throw(ArgumentError("invalid number of batches"))
    0 <= config.stop_after_iteration < config.max_batch_iterations ||
        throw(ArgumentError("stop_after_iteration must be below max iterations"))
    0 < config.base_noise_std < config.elevated_noise_std ||
        throw(ArgumentError("noise standard deviations must satisfy 0 < base < elevated"))
    config.border_width > 0 || throw(ArgumentError("border width must be positive"))
    config.corner_transition > 0 ||
        throw(ArgumentError("corner transition must be positive"))
    config.grid_size >= 2 || throw(ArgumentError("grid size must be at least two"))
    config.grid_x_min < config.grid_x_max ||
        throw(ArgumentError("grid x minimum must be below its maximum"))
    config.grid_y_min < config.grid_y_max ||
        throw(ArgumentError("grid y minimum must be below its maximum"))
    return config
end

# ---------------------------------------------------------------------------
# Clean paired benchmark data
# ---------------------------------------------------------------------------

smooth_xor_mean(x1, x2) =
    0.5 * (1 - sinpi(x1 / 2) * sinpi(x2 / 2))

function border_intensity(x1, x2, width)
    vertical = exp(-0.5 * abs2(x1 / width))
    horizontal = exp(-0.5 * abs2(x2 / width))
    return 1 - (1 - vertical) * (1 - horizontal)
end

logistic_gate(value, threshold, transition) =
    inv(1 + exp(-(value - threshold) / transition))

function corner_intensity(x1, x2, config)
    return logistic_gate(
        x1, config.corner_x_threshold, config.corner_transition,
    ) * logistic_gate(
        x2, config.corner_y_threshold, config.corner_transition,
    )
end

function make_paired_datasets(config, seed)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, config.n_samples) .- 2
    x2 = 4 .* rand(rng, config.n_samples) .- 2
    epsilon = randn(rng, config.n_samples)
    clean_mean = smooth_xor_mean.(x1, x2)
    border = border_intensity.(x1, x2, config.border_width)
    corner = corner_intensity.(x1, x2, Ref(config))

    noise_std = Dict(
        :constant => fill(config.base_noise_std, config.n_samples),
        :border => config.base_noise_std .+
                   (config.elevated_noise_std - config.base_noise_std) .* border,
        :corner => config.base_noise_std .+
                   (config.elevated_noise_std - config.base_noise_std) .* corner,
    )

    return Dict(task => (
        x1 = copy(x1),
        x2 = copy(x2),
        y = clean_mean .+ noise_std[task] .* epsilon,
        clean_mean = copy(clean_mean),
        true_variance = abs2.(noise_std[task]),
        border_intensity = copy(border),
        corner_intensity = copy(corner),
    ) for task in keys(noise_std))
end

function deterministic_split(n, config, seed)
    rng = StableRNG(seed)
    indices = randperm(rng, n)
    n_train = round(Int, config.train_fraction * n)
    n_validation = round(Int, config.validation_fraction * n)
    n_train > 0 && n_validation > 0 && n_train + n_validation < n ||
        error("split produced an empty partition")
    return (
        train = sort(indices[1:n_train]),
        validation = sort(indices[(n_train + 1):(n_train + n_validation)]),
        test = sort(indices[(n_train + n_validation + 1):end]),
    )
end

subset_data(data, indices) = NamedTuple(
    name => getproperty(data, name)[indices] for name in propertynames(data)
)

build_features(data) = [
    [1.0, data.x1[index], data.x2[index]] for index in eachindex(data.x1)
]

function deterministic_batch_ranges(n_observations, n_batches)
    1 <= n_batches <= n_observations || throw(ArgumentError(
        "n_batches must be between 1 and the observation count",
    ))
    quotient, remainder = divrem(n_observations, n_batches)
    ranges = UnitRange{Int}[]
    first_index = 1
    for batch in 1:n_batches
        batch_size = quotient + (batch <= remainder ? 1 : 0)
        last_index = first_index + batch_size - 1
        push!(ranges, first_index:last_index)
        first_index = last_index + 1
    end
    vcat(collect.(ranges)...) == collect(1:n_observations) ||
        error("batch ranges do not cover every observation exactly once")
    return ranges
end

# ---------------------------------------------------------------------------
# Priors and initial moments
# ---------------------------------------------------------------------------

function paired_weight_priors(
    n_neurons,
    radius_scale,
    direction_variance,
    bias_variance,
    seed,
)
    iseven(n_neurons) || throw(ArgumentError("paired priors need an even width"))
    rng = StableRNG(seed)
    n_pairs = div(n_neurons, 2)
    prior_precision = Diagonal([
        inv(bias_variance),
        inv(direction_variance),
        inv(direction_variance),
    ])
    weights = Vector{Any}(undef, n_neurons)
    for pair in 1:n_pairs
        angle = pi * (pair - 1) / n_pairs + 0.1 * randn(rng)
        radius = radius_scale * (1 + 0.05 * randn(rng))
        direction = [radius * cos(angle), radius * sin(angle)]
        bias = (isodd(pair) ? 1.0 : -1.0) *
               (pi / 4) * (1 + 0.1 * randn(rng))
        for (slot, sign) in ((2pair - 1, 1.0), (2pair, -1.0))
            prior_mean = [sign * bias, direction[1], direction[2]]
            weights[slot] = MvNormalWeightedMeanPrecision(
                prior_precision * prior_mean,
                prior_precision,
            )
        end
    end
    return weights
end

function make_mean_priors(config)
    weights = paired_weight_priors(
        config.n_neurons,
        config.w_prior_scale,
        config.w_prior_variance,
        config.bias_prior_variance,
        config.prior_seed,
    )
    n_pairs = div(config.n_neurons, 2)
    coefficients = Vector{Any}(undef, config.n_neurons)
    for pair in 1:n_pairs
        coefficients[2pair - 1] = NormalMeanVariance(
            config.v_prior_scale, config.v_prior_variance,
        )
        coefficients[2pair] = NormalMeanVariance(
            -config.v_prior_scale, config.v_prior_variance,
        )
    end
    return Dict{Symbol, Any}(
        :w => weights,
        :v => coefficients,
        :mean_intercept => NormalMeanVariance(
            config.mean_intercept_prior_mean,
            config.mean_intercept_prior_variance,
        ),
        :tau => GammaShapeRate(config.tau_prior...),
        :tau_c => GammaShapeRate(config.tau_c_prior...),
        :obs_noise => GammaShapeRate(config.obs_noise_prior...),
    )
end

inverse_squareplus(value) = value - inv(value)
squareplus_value(value) = exp(asinh(value / 2))

function make_noise_priors(config, n_noise_neurons, kappa, baseline_precision)
    iseven(n_noise_neurons) ||
        throw(ArgumentError("noise paired priors require an even width"))
    baseline_precision > 0 || throw(ArgumentError("baseline precision must be positive"))
    kappa > 1 || throw(ArgumentError("kappa must exceed one"))
    score_intercept = inverse_squareplus(baseline_precision)
    beta_mean = kappa / baseline_precision
    beta_shape = config.beta_prior_shape
    return Dict{Symbol, Any}(
        :noise_w => paired_weight_priors(
            n_noise_neurons,
            config.noise_w_prior_scale,
            config.noise_w_prior_variance,
            config.noise_bias_prior_variance,
            config.prior_seed + 10_000 + n_noise_neurons,
        ),
        :noise_v => [
            NormalMeanVariance(0.0, config.noise_v_prior_variance)
            for _ in 1:n_noise_neurons
        ],
        :noise_tau => GammaShapeRate(config.noise_tau_prior...),
        :noise_tau_c => GammaShapeRate(config.noise_tau_c_prior...),
        :noise_intercept => NormalMeanVariance(
            score_intercept, config.noise_intercept_prior_variance,
        ),
        :noise_beta => GammaShapeRate(
            beta_shape, beta_shape / beta_mean,
        ),
        :baseline_obs_noise => nothing,
        :baseline_precision => Float64(baseline_precision),
        :score_intercept => Float64(score_intercept),
        :kappa => Float64(kappa),
    )
end

function merge_hierarchy_priors(mean_priors, noise_priors, baseline_obs_noise)
    merged = Dict{Symbol, Any}()
    for (name, value) in mean_priors
        merged[name] = deepcopy(value)
    end
    for (name, value) in noise_priors
        merged[name] = deepcopy(value)
    end
    merged[:baseline_obs_noise] = deepcopy(baseline_obs_noise)
    return merged
end

function activation_meta(config)
    return ResidualSineMeta(rho = config.phi_rho, omega = config.phi_omega)
end

function activation_dependencies(config; alpha = config.mean_ngmp_alpha, max_step = 1.0)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = alpha,
            beta = config.ngmp_beta,
            max_step = max_step,
        ),
    )
end

function projection_from_symbol(name)
    name === :unscented && return TangentProjection(type = Unscented)
    name === :quadrature32 && return TangentProjection(type = Quadrature(32))
    throw(ArgumentError("projection must be unscented or quadrature32"))
end

function squareplus_dependencies(config, projection, alpha, max_step)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = projection_from_symbol(projection),
        damping = DampingMeta(
            alpha = alpha,
            beta = 0.0,
            max_step = max_step,
        ),
    )
end

function likelihood_dependencies(config, update, alpha, max_step)
    update === :vmp && return nothing
    update === :ngmp || throw(ArgumentError("likelihood update must be vmp or ngmp"))
    return NGMPDependencies(
        μ = nothing,
        τ = nothing,
        projection = TangentProjection(type = Unscented),
        damping = DampingMeta(alpha = alpha, beta = 0.0, max_step = max_step),
    )
end

function mean_pushforward_inits(priors, features, config)
    activation = activation_meta(config)
    phi(value) = SurrogateModelling._residual_sine(value, activation)
    w_means = mean.(priors[:w])
    v_means = mean.(priors[:v])
    n = length(features)
    za = [
        NormalMeanVariance(dot(w_means[k], features[i]), 0.5)
        for k in 1:config.n_neurons, i in 1:n
    ]
    h = [
        NormalMeanVariance(phi(mean(za[k, i])), 1.0)
        for k in 1:config.n_neurons, i in 1:n
    ]
    c = [
        NormalMeanVariance(v_means[k] * mean(h[k, i]), 1.0)
        for k in 1:config.n_neurons, i in 1:n
    ]
    intercept_mean = mean(priors[:mean_intercept])
    mean_offset = [
        NormalMeanVariance(
            intercept_mean,
            config.mean_intercept_prior_variance,
        )
        for _ in 1:n
    ]
    out = [
        NormalMeanVariance(
            sum(mean(c[k, i]) for k in 1:config.n_neurons) +
            mean(mean_offset[i]),
            1.0,
        ) for i in 1:n
    ]
    return (; za, h, c, mean_offset, out)
end

function noise_pushforward_inits(
    priors,
    features,
    n_noise_neurons,
    config;
    cold_start = false,
)
    activation = activation_meta(config)
    phi(value) = SurrogateModelling._residual_sine(value, activation)
    w_means = mean.(priors[:noise_w])
    intercept_mean = mean(priors[:noise_intercept])
    n = length(features)
    noise_v = cold_start ? [
        NormalMeanVariance(0.0, config.noise_v_initial_variance)
        for _ in 1:n_noise_neurons
    ] : deepcopy(priors[:noise_v])
    v_means = mean.(noise_v)
    za = [
        NormalMeanVariance(dot(w_means[k], features[i]), 0.5)
        for k in 1:n_noise_neurons, i in 1:n
    ]
    h = [
        NormalMeanVariance(phi(mean(za[k, i])), 1.0)
        for k in 1:n_noise_neurons, i in 1:n
    ]
    c = Matrix{Any}(undef, n_noise_neurons + 1, n)
    for i in 1:n, k in 1:n_noise_neurons
        c[k, i] = NormalMeanVariance(
            v_means[k] * mean(h[k, i]), 1e-6,
        )
    end
    for i in 1:n
        c[n_noise_neurons + 1, i] = NormalMeanVariance(intercept_mean, 1e-6)
    end
    score_means = [
        sum(mean(c[k, i]) for k in 1:(n_noise_neurons + 1))
        for i in 1:n
    ]
    score = [
        NormalMeanVariance(
            score_means[i],
            cold_start ? config.noise_intercept_initial_variance : 1e-4,
        ) for i in 1:n
    ]
    baseline_distribution = priors[:baseline_obs_noise]
    gamma = cold_start ? [deepcopy(baseline_distribution) for _ in 1:n] : [
        GammaShapeRate(
            priors[:kappa],
            priors[:kappa] / squareplus_value(score_means[i]),
        ) for i in 1:n
    ]
    intercept = cold_start ? NormalMeanVariance(
        priors[:score_intercept], config.noise_intercept_initial_variance,
    ) : deepcopy(priors[:noise_intercept])

    if cold_start
        initial_score_means = mean.(score)
        all(isapprox(value, intercept_mean; atol = 1e-10, rtol = 0) for value in initial_score_means) ||
            error("noise score initialization is not constant")
        initial_precision_means = mean.(gamma)
        baseline_precision = priors[:baseline_precision]
        all(isapprox(value, baseline_precision; atol = 1e-10, rtol = 1e-10) for value in initial_precision_means) ||
            error("local precision initialization does not match the baseline")
    end

    return (; noise_v, za, h, c, score, gamma, intercept)
end

# ---------------------------------------------------------------------------
# Constant-noise baseline
# ---------------------------------------------------------------------------

@model function hetero_constant_mean_model(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, mean_offset, out

    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]
    obs_noise ~ priors[:obs_noise]
    mean_intercept ~ priors[:mean_intercept]
    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end
    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], tau,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {dependencies = activation_deps, meta = activation}
            c[neuron, observation] ~ softdot(
                v[neuron], h[neuron, observation], tau_c,
            )
        end
        mean_offset[observation] := mean_intercept * 1.0
        out[observation] ~ ManyPlus(inputs = [
            [c[neuron, observation] for neuron in 1:n_neurons]...,
            mean_offset[observation],
        ])
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

@constraints function hetero_constant_mean_constraints()
    q(
        w, v, za, h, c, mean_offset, out,
        tau, tau_c, obs_noise, mean_intercept,
    ) = q(w, za, h, c, mean_offset, out) *
        q(v)q(tau)q(tau_c)q(obs_noise)q(mean_intercept)
    q(w)::MomentForm()
end

@initialization function hetero_constant_mean_initialization(priors, inits)
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(mean_offset) = inits.mean_offset
    q(out) = inits.out
    q(tau) = priors[:tau]
    q(tau_c) = priors[:tau_c]
    q(obs_noise) = priors[:obs_noise]
    q(mean_intercept) = priors[:mean_intercept]
    μ(w) = deepcopy(priors[:w])
end

# ---------------------------------------------------------------------------
# Hierarchical variance models
# ---------------------------------------------------------------------------

@model function hetero_joint_model(
    n_neurons,
    n_noise_neurons,
    kappa,
    features,
    y,
    priors,
    activation,
    mean_activation_deps,
    noise_activation_deps,
    squareplus_deps,
    observation_deps,
)
    local w, v, za, h, c, mean_offset, out
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]
    mean_intercept ~ priors[:mean_intercept]
    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    noise_beta ~ priors[:noise_beta]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], tau,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {dependencies = mean_activation_deps, meta = activation}
            c[neuron, observation] ~ softdot(
                v[neuron], h[neuron, observation], tau_c,
            )
        end
        mean_offset[observation] := mean_intercept * 1.0
        out[observation] ~ ManyPlus(inputs = [
            [c[neuron, observation] for neuron in 1:n_neurons]...,
            mean_offset[observation],
        ])

        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {dependencies = noise_activation_deps, meta = activation}
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron], noise_h[neuron, observation], noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept, 1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation] for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ GammaShapeRate(kappa, noise_beta)
        gamma[observation] ~ Squareplus(noise_score[observation]) where {
            dependencies = squareplus_deps,
        }
        y[observation] ~ NormalMeanPrecision(
            out[observation], gamma[observation],
        ) where {dependencies = observation_deps}
    end
end

@constraints function hetero_joint_meanfield_constraints()
    q(
        w, v, za, h, c, out, tau, tau_c,
        mean_offset, mean_intercept,
        noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma,
        noise_tau, noise_tau_c, noise_intercept, noise_beta,
    ) = q(w, za, h, c, mean_offset, out) *
        q(v)q(tau)q(tau_c)q(mean_intercept) *
        q(noise_w, noise_za, noise_h, noise_c, noise_score, gamma) *
        q(noise_v)q(noise_tau)q(noise_tau_c)q(noise_intercept)q(noise_beta)
    q(w)::MomentForm()
    q(noise_w)::MomentForm()
end

# The NGMP likelihood needs the uncertain mean and precision in the same local
# BP cluster.  This deliberately forms a larger structured cluster than the VMP
# control and is treated as a high-risk ablation by the runner.
@constraints function hetero_joint_ngmp_constraints()
    q(
        w, v, za, h, c, out, tau, tau_c,
        mean_offset, mean_intercept,
        noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma,
        noise_tau, noise_tau_c, noise_intercept, noise_beta,
    ) = q(
        w, za, h, c, mean_offset, out,
        noise_w, noise_za, noise_h, noise_c, noise_score, gamma,
    ) * q(v)q(tau)q(tau_c)q(mean_intercept) *
        q(noise_v)q(noise_tau)q(noise_tau_c) *
        q(noise_intercept)q(noise_beta)
    q(w)::MomentForm()
    q(noise_w)::MomentForm()
end

@initialization function hetero_joint_initialization(priors, mean_inits, noise_inits)
    q(v) = deepcopy(priors[:v])
    q(za) = mean_inits.za
    q(h) = mean_inits.h
    q(c) = mean_inits.c
    q(mean_offset) = mean_inits.mean_offset
    q(out) = mean_inits.out
    q(tau) = priors[:tau]
    q(tau_c) = priors[:tau_c]
    q(mean_intercept) = priors[:mean_intercept]
    μ(w) = deepcopy(priors[:w])

    q(noise_v) = noise_inits.noise_v
    q(noise_za) = noise_inits.za
    q(noise_h) = noise_inits.h
    q(noise_c) = noise_inits.c
    q(noise_score) = noise_inits.score
    q(gamma) = noise_inits.gamma
    q(noise_tau) = priors[:noise_tau]
    q(noise_tau_c) = priors[:noise_tau_c]
    q(noise_intercept) = noise_inits.intercept
    q(noise_beta) = priors[:noise_beta]
    μ(noise_w) = deepcopy(priors[:noise_w])
end

@model function hetero_fixed_mean_model(
    n_noise_neurons,
    kappa,
    features,
    fixed_mean,
    y,
    priors,
    activation,
    noise_activation_deps,
    squareplus_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    noise_beta ~ priors[:noise_beta]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(y)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {dependencies = noise_activation_deps, meta = activation}
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron], noise_h[neuron, observation], noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept, 1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation] for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ GammaShapeRate(kappa, noise_beta)
        gamma[observation] ~ Squareplus(noise_score[observation]) where {
            dependencies = squareplus_deps,
        }
        y[observation] ~ NormalMeanPrecision(
            fixed_mean[observation], gamma[observation],
        )
    end
end

@constraints function hetero_fixed_mean_constraints()
    q(
        noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma,
        noise_tau, noise_tau_c, noise_intercept, noise_beta,
    ) = q(noise_w, noise_za, noise_h, noise_c, noise_score, gamma) *
        q(noise_v)q(noise_tau)q(noise_tau_c)q(noise_intercept)q(noise_beta)
    q(noise_w)::MomentForm()
end

@initialization function hetero_fixed_mean_initialization(priors, noise_inits)
    q(noise_v) = noise_inits.noise_v
    q(noise_za) = noise_inits.za
    q(noise_h) = noise_inits.h
    q(noise_c) = noise_inits.c
    q(noise_score) = noise_inits.score
    q(gamma) = noise_inits.gamma
    q(noise_tau) = priors[:noise_tau]
    q(noise_tau_c) = priors[:noise_tau_c]
    q(noise_intercept) = noise_inits.intercept
    q(noise_beta) = priors[:noise_beta]
    μ(noise_w) = deepcopy(priors[:noise_w])
end

# ---------------------------------------------------------------------------
# Batched fitting and posterior transport
# ---------------------------------------------------------------------------

function validate_distribution(distribution, label)
    distribution_mean = mean(distribution)
    mean_values = distribution_mean isa Number ?
                  [Float64(distribution_mean)] :
                  Float64.(vec(distribution_mean))
    all(isfinite, mean_values) || error("$label has a non-finite mean")
    variance_values = if distribution_mean isa Number
        [Float64(var(distribution))]
    else
        Float64.(diag(Matrix(cov(distribution))))
    end
    all(value -> isfinite(value) && value > 0, variance_values) ||
        error("$label has a non-positive or non-finite variance")
    if distribution isa GammaDistributionsFamily
        isfinite(shape(distribution)) && shape(distribution) > 0 ||
            error("$label has an invalid Gamma shape")
        isfinite(rate(distribution)) && rate(distribution) > 0 ||
            error("$label has an invalid Gamma rate")
    end
    return nothing
end

function validate_mean_priors(priors)
    for (index, distribution) in enumerate(priors[:w])
        validate_distribution(distribution, "w[$index]")
    end
    for (index, distribution) in enumerate(priors[:v])
        validate_distribution(distribution, "v[$index]")
    end
    for name in (:tau, :tau_c, :obs_noise)
        haskey(priors, name) && validate_distribution(priors[name], String(name))
    end
    haskey(priors, :mean_intercept) && validate_distribution(
        priors[:mean_intercept],
        "mean_intercept",
    )
    return nothing
end

function validate_noise_priors(priors)
    for (index, distribution) in enumerate(priors[:noise_w])
        validate_distribution(distribution, "noise_w[$index]")
    end
    for (index, distribution) in enumerate(priors[:noise_v])
        validate_distribution(distribution, "noise_v[$index]")
    end
    for name in (:noise_tau, :noise_tau_c, :noise_beta)
        validate_distribution(priors[name], String(name))
    end
    validate_distribution(priors[:noise_intercept], "noise_intercept")
    return nothing
end

function make_delayed_stopper(config)
    stopper = StopEarlyIterationStrategy(config.stop_atol, config.stop_rtol)
    return function (event)
        event.iteration > config.stop_after_iteration && stopper(event)
        return nothing
    end
end

function baseline_posterior_priors(result)
    return Dict{Symbol, Any}(
        :w => deepcopy(collect(vec(result.posteriors[:w]))),
        :v => deepcopy(collect(vec(result.posteriors[:v]))),
        :mean_intercept => deepcopy(result.posteriors[:mean_intercept]),
        :tau => deepcopy(result.posteriors[:tau]),
        :tau_c => deepcopy(result.posteriors[:tau_c]),
        :obs_noise => deepcopy(result.posteriors[:obs_noise]),
    )
end

function run_baseline_batch(priors, observations, features, config; alpha)
    dependencies = activation_dependencies(config; alpha, max_step = 1.0)
    local result
    elapsed = @elapsed result = infer(
        model = hetero_constant_mean_model(
            n_neurons = config.n_neurons,
            priors = priors,
            activation = activation_meta(config),
            activation_deps = dependencies,
        ),
        data = (y = observations, features = features),
        constraints = hetero_constant_mean_constraints(),
        initialization = hetero_constant_mean_initialization(
            priors, mean_pushforward_inits(priors, features, config),
        ),
        returnvars = (
            w = KeepLast(), v = KeepLast(), mean_intercept = KeepLast(),
            tau = KeepLast(),
            tau_c = KeepLast(), obs_noise = KeepLast(),
        ),
        iterations = config.max_batch_iterations,
        free_energy = true,
        callbacks = (after_iteration = make_delayed_stopper(config),),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    all(isfinite, result.free_energy) || error("baseline batch has non-finite FE")
    updated = baseline_posterior_priors(result)
    validate_mean_priors(updated)
    return (
        priors = updated,
        iterations = length(result.free_energy),
        elapsed_seconds = elapsed,
        free_energy = Float64.(result.free_energy),
        activation_states = length(dependencies.states),
    )
end

function run_baseline_training(train, features, batches, config; alpha)
    carried = make_mean_priors(config)
    validate_mean_priors(carried)
    reports = NamedTuple[]
    for (batch, indices) in enumerate(batches)
        fitted = run_baseline_batch(
            carried, train.y[indices], features[indices], config; alpha,
        )
        carried = fitted.priors
        push!(reports, (
            batch = batch,
            observations = length(indices),
            iterations = fitted.iterations,
            elapsed_seconds = fitted.elapsed_seconds,
            free_energy = fitted.free_energy,
            final_free_energy = last(fitted.free_energy),
            activation_states = fitted.activation_states,
        ))
    end
    return (; priors = carried, reports)
end

function hierarchy_posterior_priors(result, previous; include_mean)
    carried = Dict{Symbol, Any}()
    if include_mean
        carried[:w] = deepcopy(collect(vec(result.posteriors[:w])))
        carried[:v] = deepcopy(collect(vec(result.posteriors[:v])))
        carried[:mean_intercept] = deepcopy(
            result.posteriors[:mean_intercept],
        )
        carried[:tau] = deepcopy(result.posteriors[:tau])
        carried[:tau_c] = deepcopy(result.posteriors[:tau_c])
    else
        for name in (
            :w,
            :v,
            :mean_intercept,
            :tau,
            :tau_c,
            :obs_noise,
        )
            haskey(previous, name) && (carried[name] = deepcopy(previous[name]))
        end
    end
    carried[:noise_w] = deepcopy(collect(vec(result.posteriors[:noise_w])))
    carried[:noise_v] = deepcopy(collect(vec(result.posteriors[:noise_v])))
    carried[:noise_tau] = deepcopy(result.posteriors[:noise_tau])
    carried[:noise_tau_c] = deepcopy(result.posteriors[:noise_tau_c])
    carried[:noise_intercept] = deepcopy(result.posteriors[:noise_intercept])
    carried[:noise_beta] = deepcopy(result.posteriors[:noise_beta])
    for name in (:baseline_obs_noise, :baseline_precision, :score_intercept, :kappa)
        carried[name] = deepcopy(previous[name])
    end
    return carried
end

function run_joint_batch(
    priors,
    observations,
    features,
    config,
    spec;
    cold_start,
)
    mean_deps = activation_dependencies(
        config; alpha = spec.mean_alpha, max_step = spec.mean_max_step,
    )
    noise_activation_deps = activation_dependencies(
        config; alpha = spec.squareplus_alpha, max_step = spec.squareplus_max_step,
    )
    squareplus_deps = squareplus_dependencies(
        config, spec.projection, spec.squareplus_alpha, spec.squareplus_max_step,
    )
    observation_deps = likelihood_dependencies(
        config, spec.likelihood, spec.likelihood_alpha, spec.likelihood_max_step,
    )
    noise_inits = noise_pushforward_inits(
        priors, features, spec.n_noise_neurons, config; cold_start,
    )
    constraints = spec.likelihood === :ngmp ?
                  hetero_joint_ngmp_constraints() :
                  hetero_joint_meanfield_constraints()
    local result
    elapsed = @elapsed result = infer(
        model = hetero_joint_model(
            n_neurons = config.n_neurons,
            n_noise_neurons = spec.n_noise_neurons,
            kappa = spec.kappa,
            priors = priors,
            activation = activation_meta(config),
            mean_activation_deps = mean_deps,
            noise_activation_deps = noise_activation_deps,
            squareplus_deps = squareplus_deps,
            observation_deps = observation_deps,
        ),
        data = (y = observations, features = features),
        constraints = constraints,
        initialization = hetero_joint_initialization(
            priors,
            mean_pushforward_inits(priors, features, config),
            noise_inits,
        ),
        returnvars = (
            w = KeepLast(), v = KeepLast(),
            mean_intercept = KeepLast(),
            tau = KeepLast(), tau_c = KeepLast(),
            noise_w = KeepLast(), noise_v = KeepLast(),
            noise_tau = KeepLast(), noise_tau_c = KeepLast(),
            noise_intercept = KeepLast(), noise_beta = KeepLast(),
        ),
        iterations = config.max_batch_iterations,
        free_energy = true,
        callbacks = (after_iteration = make_delayed_stopper(config),),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    all(isfinite, result.free_energy) || error("joint batch has non-finite FE")
    updated = hierarchy_posterior_priors(result, priors; include_mean = true)
    validate_mean_priors(updated)
    validate_noise_priors(updated)
    expected_squareplus_states = 2length(observations)
    length(squareplus_deps.states) == expected_squareplus_states || error(
        "Squareplus state count $(length(squareplus_deps.states)) != $expected_squareplus_states",
    )
    expected_likelihood_states = spec.likelihood === :ngmp ?
                                 2length(observations) : 0
    actual_likelihood_states = isnothing(observation_deps) ? 0 :
                               length(observation_deps.states)
    actual_likelihood_states == expected_likelihood_states || error(
        "likelihood state count $actual_likelihood_states != $expected_likelihood_states",
    )
    return (
        priors = updated,
        iterations = length(result.free_energy),
        elapsed_seconds = elapsed,
        free_energy = Float64.(result.free_energy),
        squareplus_states = length(squareplus_deps.states),
        likelihood_states = actual_likelihood_states,
    )
end

function run_fixed_batch(
    priors,
    observations,
    fixed_mean,
    features,
    config,
    spec;
    cold_start,
)
    spec.likelihood === :vmp || throw(ArgumentError(
        "the fixed-mean likelihood is already conjugate; use likelihood=:vmp",
    ))
    noise_activation_deps = activation_dependencies(
        config; alpha = spec.squareplus_alpha, max_step = spec.squareplus_max_step,
    )
    squareplus_deps = squareplus_dependencies(
        config, spec.projection, spec.squareplus_alpha, spec.squareplus_max_step,
    )
    noise_inits = noise_pushforward_inits(
        priors, features, spec.n_noise_neurons, config; cold_start,
    )
    local result
    elapsed = @elapsed result = infer(
        model = hetero_fixed_mean_model(
            n_noise_neurons = spec.n_noise_neurons,
            kappa = spec.kappa,
            priors = priors,
            activation = activation_meta(config),
            noise_activation_deps = noise_activation_deps,
            squareplus_deps = squareplus_deps,
        ),
        data = (
            y = observations, fixed_mean = fixed_mean, features = features,
        ),
        constraints = hetero_fixed_mean_constraints(),
        initialization = hetero_fixed_mean_initialization(priors, noise_inits),
        returnvars = (
            noise_w = KeepLast(), noise_v = KeepLast(),
            noise_tau = KeepLast(), noise_tau_c = KeepLast(),
            noise_intercept = KeepLast(), noise_beta = KeepLast(),
        ),
        iterations = config.max_batch_iterations,
        free_energy = true,
        callbacks = (after_iteration = make_delayed_stopper(config),),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    all(isfinite, result.free_energy) || error("fixed-mean batch has non-finite FE")
    updated = hierarchy_posterior_priors(result, priors; include_mean = false)
    validate_noise_priors(updated)
    expected_states = 2length(observations)
    length(squareplus_deps.states) == expected_states || error(
        "Squareplus state count $(length(squareplus_deps.states)) != $expected_states",
    )
    return (
        priors = updated,
        iterations = length(result.free_energy),
        elapsed_seconds = elapsed,
        free_energy = Float64.(result.free_energy),
        squareplus_states = length(squareplus_deps.states),
        likelihood_states = 0,
    )
end

function run_hierarchy_training(
    mode,
    train,
    features,
    fixed_means,
    batches,
    initial_priors,
    config,
    spec,
)
    mode in (:joint, :fixed_mean) || throw(ArgumentError("unknown mode $mode"))
    carried = deepcopy(initial_priors)
    reports = NamedTuple[]
    for (batch, indices) in enumerate(batches)
        cold_start = batch == 1
        fitted = if mode === :joint
            run_joint_batch(
                carried, train.y[indices], features[indices], config, spec;
                cold_start,
            )
        else
            run_fixed_batch(
                carried,
                train.y[indices],
                fixed_means[indices],
                features[indices],
                config,
                spec;
                cold_start,
            )
        end
        carried = fitted.priors
        push!(reports, (
            batch = batch,
            observations = length(indices),
            iterations = fitted.iterations,
            elapsed_seconds = fitted.elapsed_seconds,
            free_energy = fitted.free_energy,
            final_free_energy = last(fitted.free_energy),
            squareplus_states = fitted.squareplus_states,
            likelihood_states = fitted.likelihood_states,
        ))
    end
    return (; priors = carried, reports)
end

# ---------------------------------------------------------------------------
# Forward prediction and uncertainty decomposition
# ---------------------------------------------------------------------------

@model function hetero_mean_prediction_model(
    n_neurons,
    features,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, mean_offset, out
    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]
    mean_intercept ~ priors[:mean_intercept]
    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end
    for observation in eachindex(features)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], tau,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {dependencies = activation_deps, meta = activation}
            c[neuron, observation] ~ softdot(
                v[neuron], h[neuron, observation], tau_c,
            )
        end
        mean_offset[observation] := mean_intercept * 1.0
        out[observation] ~ ManyPlus(inputs = [
            [c[neuron, observation] for neuron in 1:n_neurons]...,
            mean_offset[observation],
        ])
        # Terminate the otherwise dangling output edge without materially
        # changing its forward moments.
        out[observation] ~ NormalMeanVariance(0.0, 1e12)
    end
end

@constraints function hetero_mean_prediction_constraints(priors)
    q(
        w, v, za, h, c, mean_offset, out,
        tau, tau_c, mean_intercept,
    ) = q(w)q(v)q(tau)q(tau_c)q(mean_intercept) *
        q(za, h, c, mean_offset, out)
    q(tau)::RxInfer.FixedMarginalFormConstraint(priors[:tau])
    q(tau_c)::RxInfer.FixedMarginalFormConstraint(priors[:tau_c])
    q(mean_intercept)::RxInfer.FixedMarginalFormConstraint(
        priors[:mean_intercept],
    )
    for (neuron, prior) in enumerate(priors[:w])
        q(w[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:v])
        q(v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function hetero_mean_prediction_initialization(priors, inits)
    q(w) = deepcopy(priors[:w])
    q(v) = deepcopy(priors[:v])
    q(tau) = priors[:tau]
    q(tau_c) = priors[:tau_c]
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(mean_offset) = inits.mean_offset
    q(out) = inits.out
    q(mean_intercept) = priors[:mean_intercept]
end

function predict_mean_batch(priors, features, config)
    isempty(features) && return Any[]
    dependencies = activation_dependencies(config)
    result = infer(
        model = hetero_mean_prediction_model(
            n_neurons = config.n_neurons,
            priors = priors,
            activation = activation_meta(config),
            activation_deps = dependencies,
        ),
        data = (features = features,),
        constraints = hetero_mean_prediction_constraints(priors),
        initialization = hetero_mean_prediction_initialization(
            priors, mean_pushforward_inits(priors, features, config),
        ),
        returnvars = (out = KeepLast(),),
        iterations = max(1, config.prediction_iterations),
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    return collect(vec(result.posteriors[:out]))
end

function predict_mean(priors, features, config)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1, length(features),
        )
        marginals[indices] = predict_mean_batch(priors, features[indices], config)
    end
    means = Float64.(mean.(marginals))
    variances = Float64.(var.(marginals))
    all(isfinite, means) || error("mean prediction contains non-finite means")
    all(value -> isfinite(value) && value > 0, variances) ||
        error("mean prediction contains invalid epistemic variances")
    return (; mean = means, epistemic_variance = variances, marginals)
end

@model function hetero_noise_prediction_model(
    n_noise_neurons,
    kappa,
    features,
    priors,
    activation,
    activation_deps,
    squareplus_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma
    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    noise_beta ~ priors[:noise_beta]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(features)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {dependencies = activation_deps, meta = activation}
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron], noise_h[neuron, observation], noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept, 1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation] for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ GammaShapeRate(kappa, noise_beta)
        gamma[observation] ~ Squareplus(noise_score[observation]) where {
            dependencies = squareplus_deps,
        }
    end
end

@constraints function hetero_noise_prediction_constraints(priors)
    q(
        noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma,
        noise_tau, noise_tau_c, noise_intercept, noise_beta,
    ) = q(noise_w)q(noise_v)q(noise_tau)q(noise_tau_c) *
        q(noise_intercept)q(noise_beta)q(noise_za, noise_h, noise_c, noise_score, gamma)
    q(noise_tau)::RxInfer.FixedMarginalFormConstraint(priors[:noise_tau])
    q(noise_tau_c)::RxInfer.FixedMarginalFormConstraint(priors[:noise_tau_c])
    q(noise_intercept)::RxInfer.FixedMarginalFormConstraint(priors[:noise_intercept])
    q(noise_beta)::RxInfer.FixedMarginalFormConstraint(priors[:noise_beta])
    for (neuron, prior) in enumerate(priors[:noise_w])
        q(noise_w[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:noise_v])
        q(noise_v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function hetero_noise_prediction_initialization(priors, inits)
    q(noise_w) = deepcopy(priors[:noise_w])
    q(noise_v) = deepcopy(priors[:noise_v])
    q(noise_tau) = priors[:noise_tau]
    q(noise_tau_c) = priors[:noise_tau_c]
    q(noise_intercept) = priors[:noise_intercept]
    q(noise_beta) = priors[:noise_beta]
    q(noise_za) = inits.za
    q(noise_h) = inits.h
    q(noise_c) = inits.c
    q(noise_score) = inits.score
    q(gamma) = inits.gamma
end

function predict_noise_batch(priors, features, config, spec)
    isempty(features) && return Any[]
    deps = squareplus_dependencies(
        config, spec.projection, spec.squareplus_alpha, spec.squareplus_max_step,
    )
    noise_activation_deps = activation_dependencies(
        config; alpha = spec.squareplus_alpha, max_step = spec.squareplus_max_step,
    )
    inits = noise_pushforward_inits(
        priors, features, spec.n_noise_neurons, config; cold_start = false,
    )
    result = infer(
        model = hetero_noise_prediction_model(
            n_noise_neurons = spec.n_noise_neurons,
            kappa = spec.kappa,
            priors = priors,
            activation = activation_meta(config),
            activation_deps = noise_activation_deps,
            squareplus_deps = deps,
        ),
        data = (features = features,),
        constraints = hetero_noise_prediction_constraints(priors),
        initialization = hetero_noise_prediction_initialization(priors, inits),
        returnvars = (gamma = KeepLast(),),
        iterations = max(1, config.prediction_iterations),
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    length(deps.states) == 2length(features) || error(
        "noise prediction Squareplus state count is wrong",
    )
    return collect(vec(result.posteriors[:gamma]))
end

function gamma_inverse_mean(distribution)
    posterior_shape = shape(distribution)
    posterior_shape > 1 || throw(DomainError(
        posterior_shape, "precision shape must exceed one for E[1/gamma]",
    ))
    value = rate(distribution) / (posterior_shape - 1)
    isfinite(value) && value > 0 || error("invalid inverse-precision moment")
    return Float64(value)
end

function predict_noise(priors, features, config, spec)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1, length(features),
        )
        marginals[indices] = predict_noise_batch(
            priors, features[indices], config, spec,
        )
    end
    precision_mean = Float64.(mean.(marginals))
    aleatoric_variance = gamma_inverse_mean.(marginals)
    all(value -> isfinite(value) && value > 0, precision_mean) ||
        error("noise prediction contains invalid precision means")
    return (; precision_mean, aleatoric_variance, marginals)
end

function baseline_prediction(priors, features, config)
    mean_prediction = predict_mean(priors, features, config)
    aleatoric = gamma_inverse_mean(priors[:obs_noise])
    return (
        mean = mean_prediction.mean,
        epistemic_variance = mean_prediction.epistemic_variance,
        aleatoric_variance = fill(aleatoric, length(features)),
        total_variance = mean_prediction.epistemic_variance .+ aleatoric,
        precision_mean = fill(mean(priors[:obs_noise]), length(features)),
    )
end

function hierarchy_prediction(priors, features, config, spec)
    mean_prediction = predict_mean(priors, features, config)
    noise_prediction = predict_noise(priors, features, config, spec)
    return (
        mean = mean_prediction.mean,
        epistemic_variance = mean_prediction.epistemic_variance,
        aleatoric_variance = noise_prediction.aleatoric_variance,
        total_variance = mean_prediction.epistemic_variance .+
                         noise_prediction.aleatoric_variance,
        precision_mean = noise_prediction.precision_mean,
    )
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

function gaussian_nll(targets, means, variances)
    all(value -> isfinite(value) && value > 0, variances) ||
        error("Gaussian NLL received invalid variances")
    return mean(0.5 .* (
        log.(2pi .* variances) .+ abs2.(targets .- means) ./ variances
    ))
end

function normal_coverage(targets, means, variances, probability)
    z = quantile(Normal(), (1 + probability) / 2)
    radius = z .* sqrt.(variances)
    return mean(abs.(targets .- means) .<= radius)
end

function pairwise_auc(scores, labels)
    positives = scores[labels]
    negatives = scores[.!labels]
    if isempty(positives) || isempty(negatives)
        return NaN
    end
    wins = 0.0
    for positive in positives, negative in negatives
        wins += positive > negative ? 1.0 : (positive == negative ? 0.5 : 0.0)
    end
    return wins / (length(positives) * length(negatives))
end

function safe_ratio(numerator, denominator)
    return isfinite(numerator) && isfinite(denominator) && denominator > 0 ?
           numerator / denominator : NaN
end

function evaluate_prediction(prediction, data, config)
    mean_mse = mean(abs2, prediction.mean .- data.y)
    constant_mse = mean(abs2, mean(data.y) .- data.y)
    aleatoric_rmse = sqrt(mean(abs2,
        prediction.aleatoric_variance .- data.true_variance,
    ))
    aleatoric_correlation = std(prediction.aleatoric_variance) > 0 &&
                            std(data.true_variance) > 0 ?
                            cor(prediction.aleatoric_variance, data.true_variance) : 0.0
    border_labels = data.border_intensity .>= 0.5
    corner_labels = (data.x1 .>= config.corner_x_threshold) .&
                    (data.x2 .>= config.corner_y_threshold)
    border_inside = any(border_labels) ?
                    mean(prediction.aleatoric_variance[border_labels]) : NaN
    border_outside = any(.!border_labels) ?
                     mean(prediction.aleatoric_variance[.!border_labels]) : NaN
    corner_inside = any(corner_labels) ?
                    mean(prediction.aleatoric_variance[corner_labels]) : NaN
    corner_outside = any(.!corner_labels) ?
                     mean(prediction.aleatoric_variance[.!corner_labels]) : NaN
    return (
        mean_mse = mean_mse,
        constant_mse = constant_mse,
        nll = gaussian_nll(data.y, prediction.mean, prediction.total_variance),
        conditional_nll = gaussian_nll(
            data.y,
            prediction.mean,
            prediction.aleatoric_variance,
        ),
        coverage50 = normal_coverage(
            data.y, prediction.mean, prediction.total_variance, 0.50,
        ),
        coverage80 = normal_coverage(
            data.y, prediction.mean, prediction.total_variance, 0.80,
        ),
        coverage95 = normal_coverage(
            data.y, prediction.mean, prediction.total_variance, 0.95,
        ),
        aleatoric_rmse = aleatoric_rmse,
        aleatoric_correlation = aleatoric_correlation,
        epistemic_variance_mean = mean(prediction.epistemic_variance),
        aleatoric_variance_mean = mean(prediction.aleatoric_variance),
        total_variance_mean = mean(prediction.total_variance),
        minimum_total_variance = minimum(prediction.total_variance),
        maximum_total_variance = maximum(prediction.total_variance),
        border_inside_variance = border_inside,
        border_outside_variance = border_outside,
        border_variance_ratio = safe_ratio(border_inside, border_outside),
        border_auc = pairwise_auc(
            prediction.aleatoric_variance, border_labels,
        ),
        corner_inside_variance = corner_inside,
        corner_outside_variance = corner_outside,
        corner_variance_ratio = safe_ratio(corner_inside, corner_outside),
        corner_auc = pairwise_auc(prediction.aleatoric_variance, corner_labels),
    )
end

# ---------------------------------------------------------------------------
# Failure-tolerant experiment runner
# ---------------------------------------------------------------------------

function hierarchy_spec(
    config,
    n_noise_neurons,
    kappa;
    projection = :unscented,
    likelihood = :vmp,
    squareplus_alpha = config.squareplus_alpha,
    squareplus_max_step = config.squareplus_max_step,
    likelihood_alpha = config.likelihood_alpha,
    likelihood_max_step = config.likelihood_max_step,
)
    return (
        n_noise_neurons = Int(n_noise_neurons),
        kappa = Float64(kappa),
        projection = Symbol(projection),
        likelihood = Symbol(likelihood),
        mean_alpha = config.mean_ngmp_alpha,
        mean_max_step = config.mean_ngmp_max_step,
        squareplus_alpha = Float64(squareplus_alpha),
        squareplus_max_step = Float64(squareplus_max_step),
        likelihood_alpha = Float64(likelihood_alpha),
        likelihood_max_step = Float64(likelihood_max_step),
    )
end

function hierarchy_retry_specs(config, requested)
    # Each candidate is run from the same cold hierarchy priors.  A failed
    # attempt therefore cannot contaminate the next candidate's posterior or
    # NGMP damping state.
    candidates = [
        requested,
        merge(requested, (
            squareplus_alpha = min(requested.squareplus_alpha, 0.02),
            squareplus_max_step = min(requested.squareplus_max_step, 0.5),
            likelihood_alpha = min(requested.likelihood_alpha, 0.02),
            likelihood_max_step = min(requested.likelihood_max_step, 0.25),
        )),
        merge(requested, (
            squareplus_alpha = min(requested.squareplus_alpha, 0.005),
            squareplus_max_step = min(requested.squareplus_max_step, 0.1),
            likelihood_alpha = min(requested.likelihood_alpha, 0.01),
            likelihood_max_step = min(requested.likelihood_max_step, 0.1),
        )),
        merge(requested, (
            projection = :quadrature32,
            squareplus_alpha = min(requested.squareplus_alpha, 0.005),
            squareplus_max_step = min(requested.squareplus_max_step, 0.1),
            likelihood_alpha = min(requested.likelihood_alpha, 0.01),
            likelihood_max_step = min(requested.likelihood_max_step, 0.1),
        )),
    ]
    unique_candidates = NamedTuple[]
    for candidate in candidates
        candidate in unique_candidates || push!(unique_candidates, candidate)
    end
    return unique_candidates
end

function empty_metrics()
    return (
        mean_mse = NaN,
        constant_mse = NaN,
        nll = NaN,
        conditional_nll = NaN,
        coverage50 = NaN,
        coverage80 = NaN,
        coverage95 = NaN,
        aleatoric_rmse = NaN,
        aleatoric_correlation = NaN,
        epistemic_variance_mean = NaN,
        aleatoric_variance_mean = NaN,
        total_variance_mean = NaN,
        minimum_total_variance = NaN,
        maximum_total_variance = NaN,
        border_inside_variance = NaN,
        border_outside_variance = NaN,
        border_variance_ratio = NaN,
        border_auc = NaN,
        corner_inside_variance = NaN,
        corner_outside_variance = NaN,
        corner_variance_ratio = NaN,
        corner_auc = NaN,
    )
end

function experiment_row(config; kwargs...)
    base = (
        stage = String(config.stage),
        task = "",
        seed = 0,
        split = "",
        mode = "",
        likelihood = "",
        projection = "",
        n_noise_neurons = 0,
        kappa = NaN,
        attempt = 0,
        status = "failure",
        error = "",
        n_samples = config.n_samples,
        n_train = 0,
        n_validation = 0,
        n_test = 0,
        n_batches = config.n_training_batches,
        total_iterations = 0,
        elapsed_seconds = NaN,
        final_free_energy = NaN,
        initial_precision = NaN,
        initial_score = NaN,
        final_precision_mean = NaN,
        mean_alpha = NaN,
        squareplus_alpha = NaN,
        squareplus_max_step = NaN,
        likelihood_alpha = NaN,
        likelihood_max_step = NaN,
        batch_iterations = "",
        batch_seconds = "",
        mean_coefficients = "",
        mean_intercept_mean = NaN,
        noise_coefficients = "",
        noise_intercept_mean = NaN,
    )
    return merge(base, empty_metrics(), (; kwargs...))
end

function concise_error(exception)
    message = replace(sprint(showerror, exception), r"\s+" => " ")
    return first(message, min(length(message), 1_200))
end

function ensure_output_directory(config)
    (config.save_outputs || config.save_plots) && mkpath(config.output_dir)
    return config.output_dir
end

function persist_rows(rows, config)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    CSV.write(joinpath(config.output_dir, "attempts.csv"), DataFrame(rows))
    return nothing
end

function record_row!(rows, row, config)
    push!(rows, row)
    # Incremental persistence is intentional: a later numerical failure does
    # not erase already completed conditions from a long screen/full run.
    persist_rows(rows, config)
    return row
end

function compact_means(distributions)
    values = Float64.(mean.(distributions))
    return join(string.(round.(values; digits = 6)), ';')
end

function batch_summary(reports)
    return (
        total_iterations = sum(report.iterations for report in reports),
        elapsed_seconds = sum(report.elapsed_seconds for report in reports),
        final_free_energy = last(reports).final_free_energy,
        batch_iterations = join((report.iterations for report in reports), ';'),
        batch_seconds = join(
            (round(report.elapsed_seconds; digits = 4) for report in reports), ';',
        ),
    )
end

function split_metadata(split)
    return (
        n_train = length(split.train),
        n_validation = length(split.validation),
        n_test = length(split.test),
    )
end

function fit_baseline_with_retries!(
    rows,
    task,
    seed,
    split_name,
    split,
    train,
    train_features,
    evaluation,
    evaluation_features,
    batches,
    config,
)
    alphas = unique([
        config.mean_ngmp_alpha,
        min(config.mean_ngmp_alpha, 0.02),
        min(config.mean_ngmp_alpha, 0.005),
    ])
    for (attempt, alpha) in enumerate(alphas)
        try
            fitted = run_baseline_training(
                train, train_features, batches, config; alpha,
            )
            prediction = baseline_prediction(
                fitted.priors, evaluation_features, config,
            )
            metrics = evaluate_prediction(prediction, evaluation, config)
            summary = batch_summary(fitted.reports)
            row = experiment_row(
                config;
                task = String(task), seed, split = String(split_name),
                mode = "baseline", likelihood = "constant",
                projection = "closed_form", attempt, status = "success",
                split_metadata(split)..., summary..., metrics...,
                initial_precision = mean(make_mean_priors(config)[:obs_noise]),
                final_precision_mean = mean(fitted.priors[:obs_noise]),
                mean_alpha = alpha,
                mean_coefficients = compact_means(fitted.priors[:v]),
                mean_intercept_mean = mean(
                    fitted.priors[:mean_intercept],
                ),
            )
            record_row!(rows, row, config)
            println(
                "baseline task=$task seed=$seed attempt=$attempt ",
                "MSE=$(round(metrics.mean_mse; digits=5)) ",
                "NLL=$(round(metrics.nll; digits=5))",
            )
            return (; fitted, prediction, alpha)
        catch exception
            message = concise_error(exception)
            record_row!(rows, experiment_row(
                config;
                task = String(task), seed, split = String(split_name),
                mode = "baseline", likelihood = "constant",
                projection = "closed_form", attempt, status = "failure",
                error = message, split_metadata(split)...,
                mean_alpha = alpha,
            ), config)
            println("baseline retry $attempt failed for $task/$seed: $message")
        end
    end
    return nothing
end

function initial_hierarchy_priors(mode, baseline_priors, config, spec)
    baseline_obs_noise = baseline_priors[:obs_noise]
    baseline_precision = mean(baseline_obs_noise)
    noise_priors = make_noise_priors(
        config, spec.n_noise_neurons, spec.kappa, baseline_precision,
    )
    # Joint learning starts the mean head from its original priors.  The
    # fixed-mean schedule freezes the already fitted baseline mean posterior.
    mean_priors = mode === :joint ? make_mean_priors(config) : baseline_priors
    return merge_hierarchy_priors(
        mean_priors, noise_priors, baseline_obs_noise,
    )
end

function fit_hierarchy_with_retries!(
    rows,
    task,
    seed,
    split_name,
    split,
    mode,
    requested_spec,
    baseline,
    train,
    train_features,
    fixed_means,
    evaluation,
    evaluation_features,
    batches,
    config,
)
    for (attempt, spec) in enumerate(hierarchy_retry_specs(config, requested_spec))
        initial_priors = initial_hierarchy_priors(
            mode, baseline.fitted.priors, config, spec,
        )
        initial_precision = initial_priors[:baseline_precision]
        initial_score = initial_priors[:score_intercept]
        try
            fitted = run_hierarchy_training(
                mode,
                train,
                train_features,
                fixed_means,
                batches,
                initial_priors,
                config,
                spec,
            )
            prediction = hierarchy_prediction(
                fitted.priors, evaluation_features, config, spec,
            )
            metrics = evaluate_prediction(prediction, evaluation, config)
            summary = batch_summary(fitted.reports)
            row = experiment_row(
                config;
                task = String(task), seed, split = String(split_name),
                mode = String(mode), likelihood = String(spec.likelihood),
                projection = String(spec.projection),
                n_noise_neurons = spec.n_noise_neurons, kappa = spec.kappa,
                attempt, status = "success", split_metadata(split)...,
                summary..., metrics..., initial_precision, initial_score,
                final_precision_mean = mean(prediction.precision_mean),
                mean_alpha = spec.mean_alpha,
                squareplus_alpha = spec.squareplus_alpha,
                squareplus_max_step = spec.squareplus_max_step,
                likelihood_alpha = spec.likelihood_alpha,
                likelihood_max_step = spec.likelihood_max_step,
                mean_coefficients = compact_means(fitted.priors[:v]),
                mean_intercept_mean = mean(
                    fitted.priors[:mean_intercept],
                ),
                noise_coefficients = compact_means(fitted.priors[:noise_v]),
                noise_intercept_mean = mean(fitted.priors[:noise_intercept]),
            )
            record_row!(rows, row, config)
            println(
                "$mode/$(spec.likelihood) task=$task seed=$seed ",
                "attempt=$attempt MSE=$(round(metrics.mean_mse; digits=5)) ",
                "NLL=$(round(metrics.nll; digits=5)) ",
                "noise-corr=$(round(metrics.aleatoric_correlation; digits=4))",
            )
            return (; fitted, prediction, spec, row)
        catch exception
            message = concise_error(exception)
            record_row!(rows, experiment_row(
                config;
                task = String(task), seed, split = String(split_name),
                mode = String(mode), likelihood = String(spec.likelihood),
                projection = String(spec.projection),
                n_noise_neurons = spec.n_noise_neurons, kappa = spec.kappa,
                attempt, status = "failure", error = message,
                split_metadata(split)..., initial_precision, initial_score,
                mean_alpha = spec.mean_alpha,
                squareplus_alpha = spec.squareplus_alpha,
                squareplus_max_step = spec.squareplus_max_step,
                likelihood_alpha = spec.likelihood_alpha,
                likelihood_max_step = spec.likelihood_max_step,
            ), config)
            println(
                "$mode/$(spec.likelihood) retry $attempt failed for ",
                "$task/$seed: $message",
            )
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

function save_free_energy_plot(reports, path)
    figure = plot(
        xlabel = "iteration (batches concatenated)",
        ylabel = "Bethe free energy",
        title = "Free energy through one-pass batches",
        legend = :topright,
        size = (900, 480),
    )
    offset = 0
    for report in reports
        iterations = offset .+ collect(eachindex(report.free_energy))
        plot!(
            figure,
            iterations,
            report.free_energy;
            linewidth = 2,
            label = "batch $(report.batch)",
        )
        offset += length(report.free_energy)
    end
    savefig(figure, path)
    return path
end

function grid_data(task, config)
    x_axis = range(
        config.grid_x_min, config.grid_x_max; length = config.grid_size,
    )
    y_axis = range(
        config.grid_y_min, config.grid_y_max; length = config.grid_size,
    )
    x1 = [x for x in x_axis for _ in y_axis]
    x2 = [y for _ in x_axis for y in y_axis]
    clean_mean = smooth_xor_mean.(x1, x2)
    border = border_intensity.(x1, x2, config.border_width)
    corner = corner_intensity.(x1, x2, Ref(config))
    intensity = task === :constant ? zeros(length(x1)) :
                task === :border ? border : corner
    noise_std = config.base_noise_std .+
                (config.elevated_noise_std - config.base_noise_std) .* intensity
    return (; x_axis, y_axis), (
        x1 = x1,
        x2 = x2,
        y = copy(clean_mean),
        clean_mean = clean_mean,
        true_variance = abs2.(noise_std),
        border_intensity = border,
        corner_intensity = corner,
    )
end

function heatmap_panel(axes, values, config, title; color = :viridis, clims = nothing)
    matrix = reshape(values, config.grid_size, config.grid_size)
    options = (
        aspect_ratio = :equal,
        xlims = (config.grid_x_min, config.grid_x_max),
        ylims = (config.grid_y_min, config.grid_y_max),
        xlabel = "x₁", ylabel = "x₂", title = title,
        color = color,
    )
    return isnothing(clims) ?
           heatmap(axes.x_axis, axes.y_axis, matrix; options...) :
           heatmap(axes.x_axis, axes.y_axis, matrix; options..., clims = clims)
end

function save_uncertainty_diagnostics(
    fitted,
    task,
    seed,
    mode,
    config,
    spec,
)
    (config.save_outputs || config.save_plots) || return nothing
    ensure_output_directory(config)
    axes, grid = grid_data(task, config)
    prediction = hierarchy_prediction(
        fitted.priors, build_features(grid), config, spec,
    )
    prefix = join((
        String(config.stage), String(task), string(seed), String(mode),
        String(spec.likelihood), "n$(spec.n_noise_neurons)",
        "k$(replace(string(spec.kappa), '.' => '_'))",
    ), "_")

    if config.save_outputs
        CSV.write(joinpath(config.output_dir, prefix * "_grid.csv"), DataFrame(
            x1 = grid.x1,
            x2 = grid.x2,
            true_mean = grid.clean_mean,
            true_aleatoric_variance = grid.true_variance,
            predictive_mean = prediction.mean,
            epistemic_variance = prediction.epistemic_variance,
            aleatoric_variance = prediction.aleatoric_variance,
            total_variance = prediction.total_variance,
        ))
    end
    if config.save_plots
        maximum_total_variance = maximum(vcat(
            prediction.epistemic_variance,
            prediction.total_variance,
        ))
        maximum_aleatoric_variance = maximum(vcat(
            prediction.aleatoric_variance,
            grid.true_variance,
        ))
        mean_radius = max(0.5, maximum(abs.(prediction.mean .- 0.5)))
        panels = [
            heatmap_panel(axes, prediction.mean, config, "predictive mean";
                          color = :balance,
                          clims = (0.5 - mean_radius, 0.5 + mean_radius)),
            heatmap_panel(axes, prediction.epistemic_variance, config,
                          "epistemic variance";
                          clims = (0, maximum_total_variance)),
            heatmap_panel(axes, prediction.aleatoric_variance, config,
                          "learned aleatoric variance";
                          clims = (0, maximum_aleatoric_variance)),
            heatmap_panel(axes, prediction.total_variance, config,
                          "total predictive variance";
                          clims = (0, maximum_total_variance)),
            heatmap_panel(axes, grid.true_variance, config,
                          "true aleatoric variance";
                          clims = (0, maximum_aleatoric_variance)),
        ]
        if config.grid_x_min < -2 || config.grid_x_max > 2 ||
           config.grid_y_min < -2 || config.grid_y_max > 2
            data_boundary_x = [-2.0, 2.0, 2.0, -2.0, -2.0]
            data_boundary_y = [-2.0, -2.0, 2.0, 2.0, -2.0]
            for panel in panels
                plot!(
                    panel, data_boundary_x, data_boundary_y;
                    color = :black, linestyle = :dash, linewidth = 1.2,
                    label = false,
                )
            end
        end
        figure = plot(panels...; layout = (1, 5), size = (1900, 380))
        savefig(figure, joinpath(config.output_dir, prefix * "_uncertainty.png"))
        save_free_energy_plot(
            fitted.reports,
            joinpath(config.output_dir, prefix * "_free_energy.png"),
        )
    end
    return prediction
end

# ---------------------------------------------------------------------------
# Study stages and selection
# ---------------------------------------------------------------------------

function stage_conditions(config, stage; winners = Dict{Symbol, Any}())
    if stage === :smoke
        vmp = hierarchy_spec(config, 2, 50.0; likelihood = :vmp)
        ngmp = hierarchy_spec(config, 2, 50.0; likelihood = :ngmp)
        return [
            (mode = :fixed_mean, spec = vmp),
            (mode = :joint, spec = vmp),
            (mode = :joint, spec = ngmp),
        ]
    elseif stage === :screen
        conditions = NamedTuple[]
        for (n_noise_neurons, kappa) in config.screen_specs
            spec = hierarchy_spec(
                config, n_noise_neurons, kappa; likelihood = :vmp,
            )
            push!(conditions, (mode = :fixed_mean, spec = spec))
            push!(conditions, (mode = :joint, spec = spec))
        end
        return conditions
    elseif stage === :full
        fixed_choice = get(winners, :fixed_mean, (
            n_noise_neurons = config.full_noise_neurons,
            kappa = config.full_kappa,
            projection = :unscented,
        ))
        joint_choice = get(winners, :joint, (
            n_noise_neurons = config.full_noise_neurons,
            kappa = config.full_kappa,
            projection = :unscented,
        ))
        fixed = hierarchy_spec(
            config, fixed_choice.n_noise_neurons, fixed_choice.kappa;
            projection = fixed_choice.projection, likelihood = :vmp,
        )
        joint_vmp = hierarchy_spec(
            config, joint_choice.n_noise_neurons, joint_choice.kappa;
            projection = joint_choice.projection, likelihood = :vmp,
        )
        joint_ngmp = hierarchy_spec(
            config, joint_choice.n_noise_neurons, joint_choice.kappa;
            projection = joint_choice.projection, likelihood = :ngmp,
        )
        return [
            (mode = :fixed_mean, spec = fixed),
            (mode = :joint, spec = joint_vmp),
            (mode = :joint, spec = joint_ngmp),
        ]
    end
    throw(ArgumentError("unknown study stage $stage"))
end

function execute_stage!(rows, config, conditions; evaluation_split)
    for seed in config.seeds
        paired = make_paired_datasets(config, seed)
        split = deterministic_split(
            config.n_samples, config, config.split_seed + seed,
        )
        for task in config.tasks
            println("\n=== $(config.stage): task=$task seed=$seed ===")
            data = paired[task]
            train = subset_data(data, split.train)
            evaluation_indices = getproperty(split, evaluation_split)
            evaluation = subset_data(data, evaluation_indices)
            train_features = build_features(train)
            evaluation_features = build_features(evaluation)
            batches = deterministic_batch_ranges(
                length(train.y), config.n_training_batches,
            )
            baseline = fit_baseline_with_retries!(
                rows,
                task,
                seed,
                evaluation_split,
                split,
                train,
                train_features,
                evaluation,
                evaluation_features,
                batches,
                config,
            )
            isnothing(baseline) && continue

            fixed_mean_prediction = try
                predict_mean(baseline.fitted.priors, train_features, config).mean
            catch exception
                println("fixed-mean prediction failed: $(concise_error(exception))")
                fill(NaN, length(train.y))
            end

            for condition in conditions
                if condition.mode === :fixed_mean && !all(isfinite, fixed_mean_prediction)
                    record_row!(rows, experiment_row(
                        config;
                        task = String(task), seed, split = String(evaluation_split),
                        mode = "fixed_mean", likelihood = "vmp",
                        n_noise_neurons = condition.spec.n_noise_neurons,
                        kappa = condition.spec.kappa,
                        status = "failure",
                        error = "baseline training-point prediction failed",
                        split_metadata(split)...,
                    ), config)
                    continue
                end
                fitted = fit_hierarchy_with_retries!(
                    rows,
                    task,
                    seed,
                    evaluation_split,
                    split,
                    condition.mode,
                    condition.spec,
                    baseline,
                    train,
                    train_features,
                    fixed_mean_prediction,
                    evaluation,
                    evaluation_features,
                    batches,
                    config,
                )
                isnothing(fitted) && continue
                if config.stage in (:smoke, :full) || config.save_plots
                    try
                        save_uncertainty_diagnostics(
                            fitted.fitted,
                            task,
                            seed,
                            condition.mode,
                            config,
                            fitted.spec,
                        )
                    catch exception
                        println("diagnostic plot failed: $(concise_error(exception))")
                    end
                end
            end
        end
    end
    return rows
end

function select_screen_winners(rows, config)
    baseline_mse = Dict{Tuple{String, Int}, Float64}()
    for row in rows
        row.stage == "screen" && row.mode == "baseline" &&
        row.status == "success" && (baseline_mse[(row.task, row.seed)] = row.mean_mse)
    end
    grouped = Dict{Tuple{Symbol, Int, Float64, Symbol}, Vector{Float64}}()
    for row in rows
        row.stage == "screen" || continue
        row.status == "success" || continue
        row.likelihood == "vmp" || continue
        row.mode in ("fixed_mean", "joint") || continue
        baseline = get(baseline_mse, (row.task, row.seed), NaN)
        isfinite(baseline) && row.mean_mse <= 1.05baseline || continue
        key = (
            Symbol(row.mode), row.n_noise_neurons, row.kappa,
            Symbol(row.projection),
        )
        push!(get!(grouped, key, Float64[]), row.nll)
    end
    winners = Dict{Symbol, Any}()
    winner_table = DataFrame(
        mode = String[],
        n_noise_neurons = Int[],
        kappa = Float64[],
        projection = String[],
        mean_validation_nll = Float64[],
    )
    expected_conditions = length(config.tasks) * length(config.seeds)
    for mode in (:fixed_mean, :joint)
        candidates = [
            (key = key, score = mean(values))
            for (key, values) in grouped
            if first(key) === mode && length(values) == expected_conditions
        ]
        if !isempty(candidates)
            winner = candidates[argmin(getproperty.(candidates, :score))]
            winners[mode] = (
                n_noise_neurons = winner.key[2],
                kappa = winner.key[3],
                projection = winner.key[4],
            )
            push!(winner_table, (
                String(mode), winner.key[2], winner.key[3],
                String(winner.key[4]), winner.score,
            ))
            println("screen winner $mode: $(winners[mode]), NLL=$(winner.score)")
        else
            println("screen found no mean-MSE-safe winner for $mode; using defaults")
        end
    end
    if config.save_outputs
        ensure_output_directory(config)
        CSV.write(joinpath(config.output_dir, "screen_winners.csv"), winner_table)
    end
    return winners
end

function write_study_note(config)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    path = joinpath(config.output_dir, "README.md")
    open(path, "w") do io
        println(io, "# Hierarchical XOR observation-noise study")
        println(io)
        println(io, "The mean and variance heads use paired additive residual-sine bases. The variance score is mapped to a positive local precision by Squareplus, while a shared Gamma rate pools those local precisions. Every variance head starts exactly at the constant model's learned average precision.")
        println(io)
        println(io, "`joint` learns both heads from their priors. `fixed_mean` freezes the fitted mean and is an empirical-Bayes residual-calibration comparison. Training uses $(config.n_training_batches) one-pass batches. Repeating the batches as epochs would count the same observations again and is therefore intentionally unsupported.")
        println(io)
        println(io, "`attempts.csv` includes failed retries as well as held-out MSE, Gaussian NLL, interval coverage, variance recovery, border/corner ratios, and posterior summaries. `screen_winners.csv` records configurations that pass the mean-MSE guard. Grid CSVs and plots decompose epistemic, learned aleatoric, and total predictive variance.")
        println(io)
        println(io, "The diagnostic rectangle and resolution are controlled by `XOR_HETERO_GRID_X_MIN/MAX`, `XOR_HETERO_GRID_Y_MIN/MAX`, and `XOR_HETERO_GRID_SIZE`. If the rectangle extends beyond the observed [-2,2]² domain, plots mark that training boundary with a dashed box.")
    end
    return path
end

function config_for_stage(config, stage)
    if stage === :smoke
        return validate_config(merge(config, (
            stage = :smoke,
            n_samples = min(config.n_samples, 96),
            n_training_batches = 2,
            max_batch_iterations = 2,
            stop_after_iteration = 1,
            tasks = (:constant,),
            seeds = (first(config.seeds),),
            prediction_iterations = min(config.prediction_iterations, 2),
            grid_size = min(config.grid_size, 8),
            save_plots = false,
        )))
    elseif stage === :screen
        return validate_config(merge(config, (
            stage = :screen,
            max_batch_iterations = min(config.max_batch_iterations, 60),
            stop_after_iteration = min(config.stop_after_iteration, 40),
            seeds = (first(config.seeds),),
            save_plots = false,
        )))
    elseif stage === :full
        return validate_config(merge(config, (stage = :full,)))
    end
    throw(ArgumentError("unknown concrete stage $stage"))
end

function run_study(config = validate_config(heteroscedastic_config()))
    ensure_output_directory(config)
    write_study_note(config)
    rows = NamedTuple[]
    if config.stage === :all
        smoke_config = config_for_stage(config, :smoke)
        execute_stage!(
            rows,
            smoke_config,
            stage_conditions(smoke_config, :smoke);
            evaluation_split = :validation,
        )
        screen_config = config_for_stage(config, :screen)
        execute_stage!(
            rows,
            screen_config,
            stage_conditions(screen_config, :screen);
            evaluation_split = :validation,
        )
        winners = select_screen_winners(rows, screen_config)
        full_config = config_for_stage(config, :full)
        execute_stage!(
            rows,
            full_config,
            stage_conditions(full_config, :full; winners);
            evaluation_split = :test,
        )
    else
        execute_stage!(
            rows,
            config,
            stage_conditions(config, config.stage);
            evaluation_split = config.stage === :full ? :test : :validation,
        )
        config.stage === :screen && select_screen_winners(rows, config)
    end
    persist_rows(rows, config)
    successes = count(row -> row.status == "success", rows)
    primary_successes = count(
        row -> row.status == "success" &&
               row.mode in ("fixed_mean", "joint") &&
               row.likelihood == "vmp",
        rows,
    )
    println("\ncompleted: $successes successful fits, $(length(rows) - successes) failed attempts")
    primary_successes > 0 || error(
        "no primary hierarchical VMP condition completed successfully",
    )
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_study()
end
