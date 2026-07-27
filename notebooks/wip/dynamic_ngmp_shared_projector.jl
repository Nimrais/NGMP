# Shared nonlinear low-rank projector for the ETTh dynamic NGMP ensemble.
#
# The trusted baseline is the working model from
# notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl. This experiment leaves that
# notebook untouched and compares
#
#   raw features -> CT(65 -> h) -> residual sine -> CT(h -> h)
#                -> expert-specific softdot -> Gamma/Log precision gate
#
# against both the native linear model and a matched two-CT linear ablation.
#
# Examples:
#   DYNAMIC_PROJECTOR_STAGE=smoke julia --project=. \
#       experiments/dynamic_ngmp_shared_projector.jl
#   DYNAMIC_PROJECTOR_STAGE=xor julia --project=. \
#       experiments/dynamic_ngmp_shared_projector.jl
#   DYNAMIC_PROJECTOR_STAGE=screen julia --project=. \
#       experiments/dynamic_ngmp_shared_projector.jl
#   DYNAMIC_PROJECTOR_STAGE=full PROJECTOR_HIDDEN_WIDTH=8 julia --project=. \
#       experiments/dynamic_ngmp_shared_projector.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Distributions
using ExponentialFamily
using JLD2
using LinearAlgebra: Diagonal, I, Symmetric, dot, eigen, svd, tr
using Plots
using ProbabilisticEnsembling
using Random
using ReactiveMP
using RxInfer
using Statistics
using SurrogateModelling
using YAML

import BayesBase: mean_cov

const PROJECTOR_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECTOR_ETTH1_H96_CACHE = joinpath(
    PROJECTOR_ROOT,
    "notebooks",
    "vmp_vs_ngmp",
    "dynamic_etth1_h96_cache.jld2",
)

projector_env_bool(name, default = false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
projector_env_int(name, default) = parse(Int, get(ENV, name, string(default)))
projector_env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

function projector_dataset_name()
    token = lowercase(strip(get(ENV, "PROJECTOR_DATASET", "ETTh1")))
    token == "etth1" && return "ETTh1"
    token == "etth2" && return "ETTh2"
    throw(ArgumentError("PROJECTOR_DATASET must be ETTh1 or ETTh2"))
end

function dynamic_projector_data_paths(config)
    default_cache = if config.dataset == "ETTh1" && config.horizon == 96
        PROJECTOR_ETTH1_H96_CACHE
    else
        joinpath(
            PROJECTOR_ROOT,
            "cache",
            "dynamic_$(lowercase(config.dataset))_h$(config.horizon)_cache.jld2",
        )
    end
    default_spec = joinpath(
        PROJECTOR_ROOT,
        "sessions",
        "dynamic",
        "vae",
        "dynamic_$(config.dataset)_$(config.horizon).yaml",
    )
    return (;
        cache = normpath(get(ENV, "PROJECTOR_CACHE", default_cache)),
        spec = normpath(get(ENV, "PROJECTOR_SPEC", default_spec)),
    )
end

function projector_env_widths(name, default)
    raw = get(ENV, name, join(default, ','))
    widths = unique(parse.(Int, strip.(split(raw, ','))))
    all(>(1), widths) || throw(ArgumentError("all projector widths must exceed one"))
    return widths
end

function dynamic_projector_config()
    stage = Symbol(lowercase(get(ENV, "DYNAMIC_PROJECTOR_STAGE", "smoke")))
    stage in (:xor, :smoke, :screen, :full, :all) || throw(ArgumentError(
        "DYNAMIC_PROJECTOR_STAGE must be xor, smoke, screen, full, or all",
    ))
    dataset = projector_dataset_name()
    horizon = projector_env_int("PROJECTOR_HORIZON", 96)
    horizon > 0 || throw(ArgumentError("PROJECTOR_HORIZON must be positive"))
    dataset_token = lowercase(dataset)
    default_output_name = dataset == "ETTh1" ?
        "dynamic_ngmp_shared_projector" :
        "dynamic_ngmp_shared_projector_$(dataset_token)_h$(horizon)"
    return (;
        stage,
        dataset,
        horizon,
        widths = projector_env_widths("PROJECTOR_WIDTHS", (4, 8, 16)),
        hidden_width = projector_env_int("PROJECTOR_HIDDEN_WIDTH", 8),
        rank_multiplier = projector_env_int("PROJECTOR_RANK_MULTIPLIER", 2),
        prior_variance = projector_env_float("PROJECTOR_PRIOR_VARIANCE", 0.25),
        head_prior_precision = projector_env_float(
            "PROJECTOR_HEAD_PRIOR_PRECISION",
            0.01,
        ),
        head_prior_mean_scale = projector_env_float(
            "PROJECTOR_HEAD_PRIOR_MEAN_SCALE",
            0.0,
        ),
        initial_jitter = projector_env_float("PROJECTOR_INITIAL_JITTER", 0.05),
        projector_seed = projector_env_int("PROJECTOR_SEED", 2_026),
        ct_precision_mean = projector_env_float("PROJECTOR_CT_PRECISION_MEAN", 10.0),
        ct_dof_multiplier = projector_env_float(
            "PROJECTOR_CT_DOF_MULTIPLIER",
            1.0,
        ),
        rho = projector_env_float("PROJECTOR_RHO", 0.9),
        omega = projector_env_float("PROJECTOR_OMEGA", 1.0),
        alpha = projector_env_float("PROJECTOR_DAMPING_ALPHA", 0.2),
        beta = projector_env_float("PROJECTOR_DAMPING_BETA", 0.0),
        activation_alpha = projector_env_float("PROJECTOR_ACTIVATION_ALPHA", 0.4),
        activation_beta = projector_env_float("PROJECTOR_ACTIVATION_BETA", 0.2),
        activation_max_step = projector_env_float(
            "PROJECTOR_ACTIVATION_MAX_STEP",
            1.0,
        ),
        ct_alpha = projector_env_float("PROJECTOR_CT_ALPHA", 0.5),
        ct_beta = projector_env_float("PROJECTOR_CT_BETA", 0.2),
        ct_max_step = projector_env_float("PROJECTOR_CT_MAX_STEP", Inf),
        kappa = projector_env_float("PROJECTOR_BETA_FLOOR_WEIGHT", 1.0),
        smoke_observations = projector_env_int("PROJECTOR_SMOKE_OBSERVATIONS", 64),
        smoke_iterations = projector_env_int("PROJECTOR_SMOKE_ITERATIONS", 2),
        xor_train = projector_env_int("PROJECTOR_XOR_TRAIN", 256),
        xor_validation = projector_env_int("PROJECTOR_XOR_VALIDATION", 256),
        xor_iterations = projector_env_int("PROJECTOR_XOR_ITERATIONS", 80),
        xor_hidden_width = projector_env_int("PROJECTOR_XOR_HIDDEN_WIDTH", 8),
        xor_prior_variance = projector_env_float(
            "PROJECTOR_XOR_PRIOR_VARIANCE",
            1.0,
        ),
        xor_initial_jitter = projector_env_float(
            "PROJECTOR_XOR_INITIAL_JITTER",
            0.5,
        ),
        xor_ct_precision_mean = projector_env_float(
            "PROJECTOR_XOR_CT_PRECISION_MEAN",
            1_000.0,
        ),
        xor_seed = projector_env_int("PROJECTOR_XOR_SEED", 42),
        xor_error = projector_env_float("PROJECTOR_XOR_ERROR", 1.5),
        screen_train = projector_env_int("PROJECTOR_SCREEN_TRAIN", 512),
        screen_validation = projector_env_int("PROJECTOR_SCREEN_VALIDATION", 256),
        screen_iterations = projector_env_int("PROJECTOR_SCREEN_ITERATIONS", 10),
        full_iterations = projector_env_int("PROJECTOR_FULL_ITERATIONS", 20),
        show_progress = projector_env_bool("SHOW_PROGRESS", false),
        save_outputs = projector_env_bool("SAVE_OUTPUTS", true),
        run_baseline = projector_env_bool("RUN_NATIVE_BASELINE", true),
        run_linear = projector_env_bool("RUN_LINEAR_PROJECTOR", true),
        run_nonlinear = projector_env_bool("RUN_NONLINEAR_PROJECTOR", true),
        free_energy = projector_env_bool("PROJECTOR_FREE_ENERGY", false),
        benchmark_warmup = projector_env_bool("PROJECTOR_BENCHMARK_WARMUP", true),
        output_prefix = get(
            ENV,
            "OUTPUT_PREFIX",
            joinpath(PROJECTOR_ROOT, "viz", default_output_name),
        ),
    )
end

function load_dynamic_projector_data(config = dynamic_projector_config())
    paths = dynamic_projector_data_paths(config)
    prepare_hint = config.dataset == "ETTh2" ?
        "run `julia --project=. dynamic_vmp_vs_ngmp_etth2.jl " *
        "--prepare-only --horizon $(config.horizon)` first" :
        "run the trusted notebook data cell first"
    isfile(paths.cache) || error(
        "missing prepared $(config.dataset) cache at $(paths.cache); $prepare_hint",
    )
    isfile(paths.spec) || error("missing projector session specification at $(paths.spec)")
    cache = load(paths.cache)
    spec = ProbabilisticEnsembling._parse_spec(YAML.load_file(paths.spec))
    return (;
        dataset = config.dataset,
        horizon = config.horizon,
        cache_path = paths.cache,
        spec_path = paths.spec,
        y_validation = collect(Float64, cache["y_val"]),
        y_test = collect(Float64, cache["y_test"]),
        predictions_validation = Matrix{Float64}(cache["predictions_val"]),
        predictions_test = Matrix{Float64}(cache["predictions_test"]),
        features_validation = [collect(Float64, feature) for feature in cache["features_val"]],
        features_test = [collect(Float64, feature) for feature in cache["features_test"]],
        base_priors = spec.priors,
    )
end

function dynamic_projector_slice(data, indices; test = false)
    if test
        return (;
            y = data.y_test[indices],
            features = data.features_test[indices],
            predictions = data.predictions_test[:, indices],
        )
    end
    return (;
        y = data.y_validation[indices],
        features = data.features_validation[indices],
        predictions = data.predictions_validation[:, indices],
    )
end

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@model function dynamic_projector_native(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    log_deps,
    log_damping,
)
    local w, z, gamma, tau, beta
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end
    for j in 1:n_obs, i in 1:n_forecasters
        z[i, j] ~ softdot(features[j], w[i], tau[i]) where {meta = LowRankMeta()}
        gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
        z[i, j] ~ Log(gamma[i, j]) where {
            dependencies = log_deps,
            meta = log_damping
        }
        y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
    end
end

@constraints function dynamic_projector_native_constraints()
    q(w, z, gamma, tau, beta) = q(w)q(z, gamma)q(tau)q(beta)
    q(w)::MomentForm()
end

@model function dynamic_ngmp_shared_projector(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_deps,
    hidden_deps,
    activation,
    activation_deps,
    log_deps,
    log_damping,
)
    local h1, s, h2, w, z, gamma, tau, beta

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]

    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end

    for j in 1:n_obs
        h1[j] ~ ContinuousTransition(features[j], a_map, P_map) where {
            dependencies = map_deps,
            meta = map_meta
        }
        s[j] ~ MvResidualSine(h1[j]) where {
            dependencies = activation_deps,
            meta = activation
        }
        h2[j] ~ ContinuousTransition(s[j], a_hidden, P_hidden) where {
            dependencies = hidden_deps,
            meta = hidden_meta
        }
        for i in 1:n_forecasters
            z[i, j] ~ softdot(h2[j], w[i], tau[i])
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            z[i, j] ~ Log(gamma[i, j]) where {
                dependencies = log_deps,
                meta = log_damping
            }
            y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
        end
    end
end

@constraints function dynamic_ngmp_shared_projector_constraints()
    q(
        h1,
        s,
        h2,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        w,
        z,
        gamma,
        tau,
        beta,
    ) = q(h1)q(s, h2)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(w)q(z, gamma)q(tau)q(beta)
    q(a_map)::MomentForm()
    q(a_hidden)::MomentForm()
    q(h2)::MomentForm()
    q(w)::MomentForm()
end

@model function dynamic_ngmp_linear_projector(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_deps,
    hidden_deps,
    log_deps,
    log_damping,
)
    local h1, h2, w, z, gamma, tau, beta

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]

    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end

    for j in 1:n_obs
        h1[j] ~ ContinuousTransition(features[j], a_map, P_map) where {
            dependencies = map_deps,
            meta = map_meta
        }
        h2[j] ~ ContinuousTransition(h1[j], a_hidden, P_hidden) where {
            dependencies = hidden_deps,
            meta = hidden_meta
        }
        for i in 1:n_forecasters
            z[i, j] ~ softdot(h2[j], w[i], tau[i])
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            z[i, j] ~ Log(gamma[i, j]) where {
                dependencies = log_deps,
                meta = log_damping
            }
            y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
        end
    end
end

@constraints function dynamic_ngmp_linear_projector_constraints()
    q(
        h1,
        h2,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        w,
        z,
        gamma,
        tau,
        beta,
    ) = q(h1, h2)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(w)q(z, gamma)q(tau)q(beta)
    q(a_map)::MomentForm()
    q(a_hidden)::MomentForm()
    q(h2)::MomentForm()
    q(w)::MomentForm()
end

# ---------------------------------------------------------------------------
# Low-rank setup and priors
# ---------------------------------------------------------------------------

function projector_rectangular_diagonal(output_dim, input_dim, scale)
    diagonal = fill(Float64(scale), min(output_dim, input_dim))
    matrix = zeros(Float64, output_dim, input_dim)
    @inbounds for index in eachindex(diagonal)
        matrix[index, index] = diagonal[index]
    end
    return diagonal, matrix
end

"Build a seeded Frobenius-orthogonal rank-one CT dictionary."
function projector_svd_low_rank_meta(
    reference_mean,
    parameter_count;
    offset_scale,
    prior_variance,
    initial_jitter = 0.0,
    rng = Random.default_rng(),
)
    output_dim, input_dim = size(reference_mean)
    matrix_parameter_count = output_dim * input_dim
    0 < parameter_count <= matrix_parameter_count || throw(ArgumentError(
        "low-rank parameter count must be in 1:$matrix_parameter_count",
    ))
    a0_diagonal, A0 = projector_rectangular_diagonal(
        output_dim,
        input_dim,
        offset_scale,
    )
    decomposition = svd(reference_mean - A0; full = true)
    atom_scale = sqrt(matrix_parameter_count / parameter_count)
    pairs = [(index, index) for index in 1:min(output_dim, input_dim)]
    append!(pairs, [
        (output_index, input_index)
        for output_index in 1:output_dim for input_index in 1:input_dim
        if output_index != input_index
    ])

    U = zeros(Float64, output_dim, parameter_count)
    V = zeros(Float64, input_dim, parameter_count)
    coefficient_mean = zeros(Float64, parameter_count)
    singular_count = min(parameter_count, length(decomposition.S))
    for (index, (output_index, input_index)) in
        enumerate(Iterators.take(pairs, parameter_count))
        U[:, index] .= atom_scale .* decomposition.U[:, output_index]
        V[:, index] .= decomposition.V[:, input_index]
        if output_index == input_index && index <= singular_count
            coefficient_mean[index] = decomposition.S[index] / atom_scale
        end
    end

    reference_coefficients = copy(coefficient_mean)
    coefficient_mean .+= initial_jitter .* randn(rng, parameter_count)
    meta = LinearLowRankMeta(a0_diagonal, U, V)
    prior = MvNormalMeanCovariance(
        coefficient_mean,
        prior_variance .* Diagonal(ones(parameter_count)),
    )
    reference_reconstruction = A0 + U * Diagonal(reference_coefficients) * V'
    prior_mean_matrix = A0 + U * Diagonal(coefficient_mean) * V'
    return (;
        meta,
        prior,
        reference_mean,
        prior_mean_matrix,
        atom_scale,
        reconstruction_error = maximum(
            abs,
            reference_reconstruction - reference_mean,
        ),
        initialization_shift = maximum(abs, prior_mean_matrix - reference_mean),
    )
end

function projector_reference_map(features, hidden_width)
    feature_dimension = length(first(features))
    feature_dimension > 1 || throw(DimensionMismatch(
        "the projector requires a bias plus at least one raw feature",
    ))
    all(feature -> length(feature) == feature_dimension, features) || throw(
        DimensionMismatch("all feature vectors must have the same length"),
    )
    all(feature -> isapprox(feature[1], 1.0; atol = 1e-10), features) || throw(
        ArgumentError("the first feature coordinate must be the existing bias"),
    )

    feature_matrix = reduce(vcat, (permutedims(feature) for feature in features))
    raw = @view feature_matrix[:, 2:end]
    location = vec(mean(raw; dims = 1))
    scale = vec(std(raw; dims = 1))
    scale = max.(scale, sqrt(eps(Float64)))
    standardized = (raw .- transpose(location)) ./ transpose(scale)
    decomposition = svd(standardized; full = false)

    reference = zeros(Float64, hidden_width, feature_dimension)
    reference[1, 1] = 1.0
    directions = zeros(Float64, feature_dimension - 1, max(hidden_width - 1, 0))
    available_components = size(decomposition.V, 2)
    @inbounds for hidden_index in 2:hidden_width
        output_component = hidden_index - 1
        direction = if output_component <= available_components
            decomposition.V[:, output_component]
        else
            phase = 2pi * (output_component - 1) / (hidden_width - 1)
            coefficients = [
                sin(component * phase + component^2) / component
                for component in 1:available_components
            ]
            candidate = decomposition.V * coefficients
            candidate ./ sqrt(dot(candidate, candidate))
        end
        score = standardized * direction
        score_scale = max(std(score), sqrt(eps(Float64)))
        raw_coefficient = direction ./ scale ./ score_scale
        reference[hidden_index, 1] = -dot(raw_coefficient, location)
        reference[hidden_index, 2:end] .= raw_coefficient
        directions[:, output_component] .= direction
    end
    return (;
        reference,
        location,
        scale,
        directions,
        singular_values = decomposition.S,
    )
end

function make_dynamic_projector_setup(
    features,
    base_priors,
    hidden_width,
    n_forecasters,
    config,
)
    feature_setup = projector_reference_map(features, hidden_width)
    rank = min(
        config.rank_multiplier * hidden_width,
        hidden_width * length(first(features)),
    )
    hidden_rank = min(
        config.rank_multiplier * hidden_width,
        hidden_width^2,
    )
    rng = MersenneTwister(config.projector_seed + hidden_width)
    map = projector_svd_low_rank_meta(
        feature_setup.reference,
        rank;
        offset_scale = 0.0,
        prior_variance = config.prior_variance,
        initial_jitter = config.initial_jitter,
        rng,
    )
    hidden = projector_svd_low_rank_meta(
        Matrix{Float64}(I, hidden_width, hidden_width),
        hidden_rank;
        offset_scale = 1.0,
        prior_variance = config.prior_variance,
        initial_jitter = config.initial_jitter,
        rng,
    )
    degrees_of_freedom = config.ct_dof_multiplier * (hidden_width + 2.0)
    degrees_of_freedom > hidden_width - 1 || throw(ArgumentError(
        "PROJECTOR_CT_DOF_MULTIPLIER gives invalid Wishart degrees of freedom " *
        "$degrees_of_freedom for hidden width $hidden_width",
    ))
    precision_prior = ExponentialFamily.WishartFast(
        degrees_of_freedom,
        Matrix(Diagonal(fill(
            degrees_of_freedom / config.ct_precision_mean,
            hidden_width,
        ))),
    )
    priors = Dict{Symbol, Any}(
        :a_map => map.prior,
        :a_hidden => hidden.prior,
        :P_map => precision_prior,
        :P_hidden => deepcopy(precision_prior),
        :w => [
            MvNormalMeanScalePrecision(
                config.head_prior_mean_scale .* randn(rng, hidden_width),
                config.head_prior_precision,
            )
            for _ in 1:n_forecasters
        ],
        :tau => deepcopy(base_priors[:τ]),
        :beta => deepcopy(base_priors[:β]),
    )
    return (;
        priors,
        map_meta = map.meta,
        hidden_meta = hidden.meta,
        map,
        hidden,
        feature_setup,
        hidden_width,
        map_rank = rank,
        hidden_rank,
        parameter_count = rank + hidden_rank + n_forecasters * hidden_width,
    )
end

function normalized_native_priors(base_priors)
    return Dict{Symbol, Any}(
        :w => deepcopy(base_priors[:w]),
        :tau => deepcopy(base_priors[:τ]),
        :beta => deepcopy(base_priors[:β]),
    )
end

# ---------------------------------------------------------------------------
# Inference and analytic predictive propagation
# ---------------------------------------------------------------------------

function native_projector_initialization(priors)
    return @initialization begin
        q(w) = deepcopy(priors[:w])
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
    end
end

function nonlinear_projector_initialization(priors, activation, hidden_width)
    s_mean, s_covariance = SurrogateModelling._mv_residual_sine_mean_cov(
        zeros(hidden_width),
        Matrix(Diagonal(ones(hidden_width))),
        activation,
    )
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_hidden) = priors[:a_hidden]
        q(P_map) = priors[:P_map]
        q(P_hidden) = priors[:P_hidden]
        q(w) = deepcopy(priors[:w])
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
        q(h1) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(s) = MvNormalMeanCovariance(s_mean, s_covariance)
        q(h2) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
    end
end

function linear_projector_initialization(priors, hidden_width)
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_hidden) = priors[:a_hidden]
        q(P_map) = priors[:P_map]
        q(P_hidden) = priors[:P_hidden]
        q(w) = deepcopy(priors[:w])
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
        q(h1) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(h2) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
    end
end

function run_native_projector_fit(training_data, base_priors, config; iterations)
    n_forecasters = size(training_data.predictions, 1)
    priors = normalized_native_priors(base_priors)
    log_deps = NGMPDependencies(out = nothing, in = nothing)
    log_damping = DampingMeta(alpha = config.alpha, beta = config.beta)
    measured = @timed infer(
        model = dynamic_projector_native(
            n_forecasters = n_forecasters,
            n_obs = length(training_data.y),
            priors = priors,
            log_deps = log_deps,
            log_damping = log_damping,
        ),
        data = (
            y = training_data.y,
            features = training_data.features,
            predictions = training_data.predictions,
        ),
        constraints = dynamic_projector_native_constraints(),
        initialization = native_projector_initialization(priors),
        iterations = iterations,
        free_energy = config.free_energy,
        showprogress = config.show_progress,
        options = (limit_stack_depth = 500,),
        disable_inference_error_hint = true,
        returnvars = (w = KeepLast(), tau = KeepLast(), beta = KeepLast()),
    )
    result = measured.value
    expected_states = 2 * n_forecasters * length(training_data.y)
    length(log_deps.states) == expected_states || error(
        "native Log state count $(length(log_deps.states)) != $expected_states",
    )
    return (;
        result,
        posterior = (
            w = result.posteriors[:w],
            tau = result.posteriors[:tau],
            beta = result.posteriors[:beta],
        ),
        training_seconds = measured.time,
        training_bytes = measured.bytes,
        log_deps,
        parameter_count = n_forecasters * length(first(training_data.features)),
    )
end

function run_shared_projector_fit(
    mode,
    training_data,
    base_priors,
    config;
    hidden_width,
    iterations,
)
    mode in (:linear, :nonlinear) || throw(ArgumentError(
        "projector mode must be linear or nonlinear",
    ))
    n_forecasters = size(training_data.predictions, 1)
    setup = make_dynamic_projector_setup(
        training_data.features,
        base_priors,
        hidden_width,
        n_forecasters,
        config,
    )
    log_deps = NGMPDependencies(out = nothing, in = nothing)
    log_damping = DampingMeta(alpha = config.alpha, beta = config.beta)
    map_deps = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_alpha,
            beta = config.ct_beta,
            max_step = config.ct_max_step,
        ),
    )
    hidden_deps = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_alpha,
            beta = config.ct_beta,
            max_step = config.ct_max_step,
        ),
    )
    activation = ResidualSineMeta(rho = config.rho, omega = config.omega)
    activation_deps = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = config.activation_alpha,
            beta = config.activation_beta,
            max_step = config.activation_max_step,
        ),
    )

    measured = if mode === :nonlinear
        @timed infer(
            model = dynamic_ngmp_shared_projector(
                n_forecasters = n_forecasters,
                n_obs = length(training_data.y),
                priors = setup.priors,
                map_meta = setup.map_meta,
                hidden_meta = setup.hidden_meta,
                map_deps = map_deps,
                hidden_deps = hidden_deps,
                activation = activation,
                activation_deps = activation_deps,
                log_deps = log_deps,
                log_damping = log_damping,
            ),
            data = (
                y = training_data.y,
                features = training_data.features,
                predictions = training_data.predictions,
            ),
            constraints = dynamic_ngmp_shared_projector_constraints(),
            initialization = nonlinear_projector_initialization(
                setup.priors,
                activation,
                hidden_width,
            ),
            iterations = iterations,
            free_energy = config.free_energy,
            showprogress = config.show_progress,
            options = (limit_stack_depth = 500,),
            disable_inference_error_hint = true,
            returnvars = (
                a_map = KeepLast(),
                a_hidden = KeepLast(),
                P_map = KeepLast(),
                P_hidden = KeepLast(),
                w = KeepLast(),
                tau = KeepLast(),
                beta = KeepLast(),
            ),
        )
    else
        @timed infer(
            model = dynamic_ngmp_linear_projector(
                n_forecasters = n_forecasters,
                n_obs = length(training_data.y),
                priors = setup.priors,
                map_meta = setup.map_meta,
                hidden_meta = setup.hidden_meta,
                map_deps = map_deps,
                hidden_deps = hidden_deps,
                log_deps = log_deps,
                log_damping = log_damping,
            ),
            data = (
                y = training_data.y,
                features = training_data.features,
                predictions = training_data.predictions,
            ),
            constraints = dynamic_ngmp_linear_projector_constraints(),
            initialization = linear_projector_initialization(
                setup.priors,
                hidden_width,
            ),
            iterations = iterations,
            free_energy = config.free_energy,
            showprogress = config.show_progress,
            options = (limit_stack_depth = 500,),
            disable_inference_error_hint = true,
            returnvars = (
                a_map = KeepLast(),
                a_hidden = KeepLast(),
                P_map = KeepLast(),
                P_hidden = KeepLast(),
                w = KeepLast(),
                tau = KeepLast(),
                beta = KeepLast(),
            ),
        )
    end

    result = measured.value
    expected_log_states = 2 * n_forecasters * length(training_data.y)
    length(log_deps.states) == expected_log_states || error(
        "projector Log state count $(length(log_deps.states)) != $expected_log_states",
    )
    if mode === :nonlinear
        expected_activation_states = 2 * length(training_data.y)
        length(activation_deps.states) == expected_activation_states || error(
            "activation state count $(length(activation_deps.states)) != $expected_activation_states",
        )
    end
    expected_ct_states = length(training_data.y)
    length(map_deps.states) == expected_ct_states || error(
        "map CT state count $(length(map_deps.states)) != $expected_ct_states",
    )
    length(hidden_deps.states) == expected_ct_states || error(
        "hidden CT state count $(length(hidden_deps.states)) != $expected_ct_states",
    )
    posterior = (
        a_map = result.posteriors[:a_map],
        a_hidden = result.posteriors[:a_hidden],
        P_map = result.posteriors[:P_map],
        P_hidden = result.posteriors[:P_hidden],
        w = result.posteriors[:w],
        tau = result.posteriors[:tau],
        beta = result.posteriors[:beta],
    )
    return (;
        result,
        posterior,
        setup,
        activation,
        predictive_activation = mode === :nonlinear ?
            ResidualSineProjectorActivation(activation) :
            IdentityProjectorActivation(),
        mode,
        training_seconds = measured.time,
        training_bytes = measured.bytes,
        log_deps,
        map_deps,
        hidden_deps,
        activation_deps,
        parameter_count = setup.parameter_count,
    )
end

function projector_psd_covariance(covariance; floor = 1e-10)
    decomposition = eigen(Symmetric(Matrix((covariance + covariance') ./ 2)))
    return decomposition.vectors *
           Diagonal(max.(decomposition.values, floor)) *
           decomposition.vectors'
end

abstract type DynamicProjectorActivation end

struct IdentityProjectorActivation <: DynamicProjectorActivation end

struct ResidualSineProjectorActivation{M <: ResidualSineMeta} <:
       DynamicProjectorActivation
    meta::M
end

projector_activation_moments(
    ::IdentityProjectorActivation,
    input_mean,
    input_covariance,
) = (input_mean, input_covariance)

projector_activation_moments(
    activation::ResidualSineProjectorActivation,
    input_mean,
    input_covariance,
) = SurrogateModelling._mv_residual_sine_mean_cov(
    input_mean,
    input_covariance,
    activation.meta,
)

function pointmass_ct_moments(x, q_a, q_W, meta)
    ma, Va = mean_cov(q_a)
    transformed_mean = SurrogateModelling._linear_low_rank_mul(meta, ma, x)
    projected_x = transpose(meta.V) * x
    parameter_covariance = meta.U *
        (Va .* (projected_x * projected_x')) * meta.U'
    noise_covariance = ReactiveMP.cholinv(mean(q_W))
    return transformed_mean, projector_psd_covariance(
        parameter_covariance + noise_covariance,
    )
end

function gaussian_ct_moments(mean_x, covariance_x, q_a, q_W, meta)
    ma, Va = mean_cov(q_a)
    second_x = covariance_x + mean_x * mean_x'
    transformed_mean = SurrogateModelling._linear_low_rank_mul(meta, ma, mean_x)
    transformed_second =
        SurrogateModelling._linear_low_rank_ASAt(meta, ma, second_x) +
        SurrogateModelling._linear_low_rank_output_uncertainty(meta, second_x, Va)
    noise_covariance = ReactiveMP.cholinv(mean(q_W))
    transformed_covariance = transformed_second -
                             transformed_mean * transformed_mean' +
                             noise_covariance
    return transformed_mean, projector_psd_covariance(transformed_covariance)
end

"Noise variance used by the softdot output rule: inverse posterior mean precision."
softdot_noise_variance(distribution) = rate(distribution) / shape(distribution)

function independent_softdot_moments(
    left_mean,
    left_covariance,
    right_distribution,
    precision_distribution,
)
    right_mean, right_covariance = mean_cov(right_distribution)
    left_second = left_covariance + left_mean * left_mean'
    right_second = right_covariance + right_mean * right_mean'
    output_mean = dot(left_mean, right_mean)
    output_variance = tr(left_second * right_second) - output_mean^2 +
                      softdot_noise_variance(precision_distribution)
    return output_mean, max(output_variance, 0.0)
end

function independent_softdot_variance_components(
    left_mean,
    left_covariance,
    right_distribution,
    precision_distribution,
)
    right_mean, right_covariance = mean_cov(right_distribution)
    hidden_uncertainty = dot(right_mean, left_covariance * right_mean)
    head_uncertainty = dot(left_mean, right_covariance * left_mean)
    interaction = tr(left_covariance * right_covariance)
    observation_noise = softdot_noise_variance(precision_distribution)
    total = hidden_uncertainty + head_uncertainty + interaction + observation_noise
    return (;
        hidden_uncertainty,
        head_uncertainty,
        interaction,
        observation_noise,
        total,
    )
end

function native_log_precision_moments(posterior, features)
    n_forecasters = length(posterior.w)
    n_observations = length(features)
    means = Matrix{Float64}(undef, n_forecasters, n_observations)
    variances = similar(means)
    @inbounds for j in 1:n_observations, i in 1:n_forecasters
        weight_mean, weight_covariance = mean_cov(posterior.w[i])
        feature = features[j]
        means[i, j] = dot(feature, weight_mean)
        variances[i, j] = dot(feature, weight_covariance * feature) +
                          softdot_noise_variance(posterior.tau[i])
    end
    return means, variances
end

function projector_hidden_moments(fit, features)
    means = Vector{Vector{Float64}}(undef, length(features))
    covariances = Vector{Matrix{Float64}}(undef, length(features))
    @inbounds for j in eachindex(features)
        h1_mean, h1_covariance = pointmass_ct_moments(
            features[j],
            fit.posterior.a_map,
            fit.posterior.P_map,
            fit.setup.map_meta,
        )
        input_mean, input_covariance = projector_activation_moments(
            fit.predictive_activation,
            h1_mean,
            h1_covariance,
        )
        means[j], covariances[j] = gaussian_ct_moments(
            input_mean,
            input_covariance,
            fit.posterior.a_hidden,
            fit.posterior.P_hidden,
            fit.setup.hidden_meta,
        )
    end
    return means, covariances
end

function projector_log_precision_moments(fit, features)
    hidden_means, hidden_covariances = projector_hidden_moments(fit, features)
    n_forecasters = length(fit.posterior.w)
    n_observations = length(features)
    means = Matrix{Float64}(undef, n_forecasters, n_observations)
    variances = similar(means)
    @inbounds for j in 1:n_observations, i in 1:n_forecasters
        means[i, j], variances[i, j] = independent_softdot_moments(
            hidden_means[j],
            hidden_covariances[j],
            fit.posterior.w[i],
            fit.posterior.tau[i],
        )
    end
    return means, variances
end

function projector_log_precision_variance_components(fit, features)
    hidden_means, hidden_covariances = projector_hidden_moments(fit, features)
    n_forecasters = length(fit.posterior.w)
    n_observations = length(features)
    hidden = Matrix{Float64}(undef, n_forecasters, n_observations)
    head = similar(hidden)
    interaction = similar(hidden)
    noise = similar(hidden)
    total = similar(hidden)
    @inbounds for j in 1:n_observations, i in 1:n_forecasters
        components = independent_softdot_variance_components(
            hidden_means[j],
            hidden_covariances[j],
            fit.posterior.w[i],
            fit.posterior.tau[i],
        )
        hidden[i, j] = components.hidden_uncertainty
        head[i, j] = components.head_uncertainty
        interaction[i, j] = components.interaction
        noise[i, j] = components.observation_noise
        total[i, j] = components.total
    end
    return (; hidden, head, interaction, noise, total)
end

function summarize_projector_uncertainty(fit, features)
    components = projector_log_precision_variance_components(fit, features)
    component_means = (;
        hidden = mean(components.hidden),
        head = mean(components.head),
        interaction = mean(components.interaction),
        noise = mean(components.noise),
        total = mean(components.total),
    )
    total = max(component_means.total, eps(Float64))
    component_fractions = (;
        hidden = component_means.hidden / total,
        head = component_means.head / total,
        interaction = component_means.interaction / total,
        noise = component_means.noise / total,
    )
    head_mean_norms = [sqrt(sum(abs2, mean(q_w))) for q_w in fit.posterior.w]
    head_covariance_traces = [tr(last(mean_cov(q_w))) for q_w in fit.posterior.w]
    return (;
        component_means,
        component_fractions,
        total_extrema = extrema(components.total),
        hidden_precision_mean = mean((
            tr(mean(fit.posterior.P_map)) / fit.setup.hidden_width,
            tr(mean(fit.posterior.P_hidden)) / fit.setup.hidden_width,
        )),
        map_precision_mean = tr(mean(fit.posterior.P_map)) / fit.setup.hidden_width,
        second_precision_mean = tr(mean(fit.posterior.P_hidden)) /
                                fit.setup.hidden_width,
        head_mean_norm_mean = mean(head_mean_norms),
        head_covariance_trace_mean = mean(head_covariance_traces),
    )
end

function ensemble_predictive_statistics(
    log_precision_mean,
    log_precision_variance,
    predictions,
    beta_posteriors;
    kappa,
)
    n_forecasters, n_observations = size(log_precision_mean)
    beta_mean = mean.(beta_posteriors)
    exponent = clamp.(
        .-log_precision_mean .+ log_precision_variance ./ 2,
        -50.0,
        50.0,
    )
    component_variance = exp.(exponent) .+ kappa .* reshape(beta_mean, :, 1)
    component_precision = clamp.(1 ./ component_variance, 1e-10, 1e10)
    normalized_weights = similar(component_precision)
    predictive_mean = Vector{Float64}(undef, n_observations)
    predictive_std = similar(predictive_mean)
    @inbounds for j in 1:n_observations
        total_precision = sum(@view component_precision[:, j])
        normalized_weights[:, j] .= component_precision[:, j] ./ total_precision
        predictive_mean[j] = dot(
            @view(normalized_weights[:, j]),
            @view(predictions[:, j]),
        )
        predictive_std[j] = sqrt(inv(total_precision))
    end
    return (;
        mean = predictive_mean,
        std = predictive_std,
        weights = normalized_weights,
        component_variance,
        log_precision_mean,
        log_precision_variance,
    )
end

function dynamic_predictive_metrics(prediction, y, expert_predictions)
    residual = prediction.mean .- y
    log_likelihood = [
        logpdf(Normal(prediction.mean[j], prediction.std[j]), y[j])
        for j in eachindex(y)
    ]
    z95 = 1.959963984540054
    coverage95 = mean(
        (y .>= prediction.mean .- z95 .* prediction.std) .&
        (y .<= prediction.mean .+ z95 .* prediction.std),
    )
    pinball = mean([
        mean(max.(q .* (y .- (prediction.mean .+ quantile(Normal(), q) .* prediction.std)),
                  (q - 1) .* (y .- (prediction.mean .+ quantile(Normal(), q) .* prediction.std))))
        for q in (0.1, 0.9)
    ])
    entropy = -sum(
        prediction.weights .* log.(max.(prediction.weights, eps(Float64)));
        dims = 1,
    )
    effective_experts = vec(exp.(entropy))
    weight_error_correlations = map(axes(expert_predictions, 1)) do i
        weights = vec(@view prediction.weights[i, :])
        negative_squared_error = .-abs2.(
            vec(@view expert_predictions[i, :]) .- y,
        )
        if std(weights) <= sqrt(eps(Float64)) ||
           std(negative_squared_error) <= sqrt(eps(Float64))
            0.0
        else
            cor(weights, negative_squared_error)
        end
    end
    return (;
        mae = mean(abs, residual),
        rmse = sqrt(mean(abs2, residual)),
        mean_log_likelihood = mean(log_likelihood),
        log_likelihood_std = std(log_likelihood),
        coverage95,
        pinball,
        effective_experts_mean = mean(effective_experts),
        effective_experts_min = minimum(effective_experts),
        effective_experts_max = maximum(effective_experts),
        weight_error_correlation_mean = mean(weight_error_correlations),
        weight_error_correlations,
    )
end

function evaluate_dynamic_fit(arm, fit, evaluation_data, config; hidden_width = 0)
    measured = @timed begin
        log_precision_mean, log_precision_variance = arm === :native ?
            native_log_precision_moments(fit.posterior, evaluation_data.features) :
            projector_log_precision_moments(fit, evaluation_data.features)
        ensemble_predictive_statistics(
            log_precision_mean,
            log_precision_variance,
            evaluation_data.predictions,
            fit.posterior.beta;
            kappa = config.kappa,
        )
    end
    prediction = measured.value
    metrics = dynamic_predictive_metrics(
        prediction,
        evaluation_data.y,
        evaluation_data.predictions,
    )
    scalar_metrics = (
        metrics.mae,
        metrics.rmse,
        metrics.mean_log_likelihood,
        metrics.log_likelihood_std,
        metrics.coverage95,
        metrics.pinball,
        metrics.effective_experts_mean,
        metrics.effective_experts_min,
        metrics.effective_experts_max,
        metrics.weight_error_correlation_mean,
    )
    finite = all(isfinite, prediction.mean) &&
             all(isfinite, prediction.std) &&
             all(isfinite, prediction.weights) &&
             all(isfinite, scalar_metrics) &&
             all(isfinite, metrics.weight_error_correlations)
    return (;
        arm = String(arm),
        hidden_width,
        fit.training_seconds,
        fit.training_bytes,
        prediction_seconds = measured.time,
        prediction_bytes = measured.bytes,
        fit.parameter_count,
        finite,
        metrics...,
        prediction,
    )
end

# ---------------------------------------------------------------------------
# Staged comparison and artifacts
# ---------------------------------------------------------------------------

function dynamic_metrics_row(stage, evaluation)
    return (;
        stage = String(stage),
        arm = evaluation.arm,
        hidden_width = evaluation.hidden_width,
        parameter_count = evaluation.parameter_count,
        training_seconds = evaluation.training_seconds,
        training_gib = evaluation.training_bytes / 2.0^30,
        prediction_seconds = evaluation.prediction_seconds,
        prediction_mib = evaluation.prediction_bytes / 2.0^20,
        finite = evaluation.finite,
        mae = evaluation.mae,
        rmse = evaluation.rmse,
        mean_log_likelihood = evaluation.mean_log_likelihood,
        log_likelihood_std = evaluation.log_likelihood_std,
        coverage95 = evaluation.coverage95,
        pinball = evaluation.pinball,
        effective_experts_mean = evaluation.effective_experts_mean,
        effective_experts_min = evaluation.effective_experts_min,
        effective_experts_max = evaluation.effective_experts_max,
        weight_error_correlation_mean = evaluation.weight_error_correlation_mean,
    )
end

function print_dynamic_metrics(stage, evaluation)
    println()
    println("=== $(stage): $(evaluation.arm), h=$(evaluation.hidden_width) ===")
    println("finite                     : ", evaluation.finite)
    println("parameters                 : ", evaluation.parameter_count)
    println("training / prediction      : ", round(evaluation.training_seconds; digits = 3),
            "s / ", round(evaluation.prediction_seconds; digits = 3), "s")
    println("allocated train / predict  : ", round(evaluation.training_bytes / 2.0^30; digits = 3),
            " GiB / ", round(evaluation.prediction_bytes / 2.0^20; digits = 3), " MiB")
    println("MAE / RMSE                 : ", round(evaluation.mae; digits = 5),
            " / ", round(evaluation.rmse; digits = 5))
    println("mean LL / std              : ", round(evaluation.mean_log_likelihood; digits = 5),
            " / ", round(evaluation.log_likelihood_std; digits = 5))
    println("coverage95 / pinball       : ", round(evaluation.coverage95; digits = 5),
            " / ", round(evaluation.pinball; digits = 5))
    println("effective experts mean/rng : ", round(evaluation.effective_experts_mean; digits = 3),
            " / ", round(evaluation.effective_experts_min; digits = 3), "..",
            round(evaluation.effective_experts_max; digits = 3))
    println("weight/error correlation   : ",
            round(evaluation.weight_error_correlation_mean; digits = 4), " ",
            evaluation.weight_error_correlations)
end

function run_dynamic_arm(
    stage,
    arm,
    training_data,
    evaluation_data,
    base_priors,
    config;
    hidden_width = 0,
    iterations,
)
    fit = if arm === :native
        run_native_projector_fit(training_data, base_priors, config; iterations)
    else
        run_shared_projector_fit(
            arm,
            training_data,
            base_priors,
            config;
            hidden_width,
            iterations,
        )
    end
    evaluation = evaluate_dynamic_fit(
        arm,
        fit,
        evaluation_data,
        config;
        hidden_width,
    )
    print_dynamic_metrics(stage, evaluation)
    return (; fit, evaluation)
end

function warm_dynamic_projector_models(
    training_data,
    base_priors,
    config;
    hidden_width,
)
    config.benchmark_warmup || return nothing
    n = min(4, length(training_data.y))
    tiny = (;
        y = training_data.y[1:n],
        features = training_data.features[1:n],
        predictions = training_data.predictions[:, 1:n],
    )
    warm_config = merge(config, (; free_energy = false, show_progress = false))
    println("Warming enabled model paths before timed comparisons ...")
    for arm in (:native, :linear, :nonlinear)
        enabled = arm === :native ? config.run_baseline :
                  arm === :linear ? config.run_linear : config.run_nonlinear
        enabled || continue
        fit = arm === :native ?
            run_native_projector_fit(tiny, base_priors, warm_config; iterations = 1) :
            run_shared_projector_fit(
                arm,
                tiny,
                base_priors,
                warm_config;
                hidden_width,
                iterations = 1,
            )
        evaluate_dynamic_fit(
            arm,
            fit,
            tiny,
            warm_config;
            hidden_width = arm === :native ? 0 : hidden_width,
        )
    end
    GC.gc()
    return nothing
end

function dynamic_projector_xor_split(rng, n, error_size)
    coordinates = 2pi .* rand(rng, 2, n) .- pi
    features = [
        [1.0, coordinates[1, observation], coordinates[2, observation]]
        for observation in 1:n
    ]
    y = @. 0.25 * sin(coordinates[1, :]) + 0.25 * cos(coordinates[2, :])
    oracle_expert = [
        coordinates[1, observation] * coordinates[2, observation] >= 0 ? 1 : 2
        for observation in 1:n
    ]
    predictions = Matrix{Float64}(undef, 2, n)
    @inbounds for observation in 1:n
        if oracle_expert[observation] == 1
            predictions[1, observation] = y[observation]
            predictions[2, observation] = y[observation] - error_size
        else
            predictions[1, observation] = y[observation] + error_size
            predictions[2, observation] = y[observation]
        end
    end
    return (; y, features, predictions, oracle_expert)
end

function dynamic_projector_xor_data(config)
    rng = MersenneTwister(config.xor_seed)
    training = dynamic_projector_xor_split(rng, config.xor_train, config.xor_error)
    validation = dynamic_projector_xor_split(
        rng,
        config.xor_validation,
        config.xor_error,
    )
    n_forecasters = size(training.predictions, 1)
    feature_dimension = length(first(training.features))
    priors = Dict{Symbol, Any}(
        :w => [
            MvNormalMeanScalePrecision(zeros(feature_dimension), 0.01)
            for _ in 1:n_forecasters
        ],
        :τ => [GammaShapeRate(1.0, 1e-3) for _ in 1:n_forecasters],
        :β => [GammaShapeRate(1.0, 1e3) for _ in 1:n_forecasters],
    )
    return (; training, validation, priors)
end

function dynamic_projector_routing_metrics(prediction, oracle_expert)
    selected = [
        findmax(@view prediction.weights[:, observation])[2]
        for observation in eachindex(oracle_expert)
    ]
    oracle_weight = [
        prediction.weights[oracle_expert[observation], observation]
        for observation in eachindex(oracle_expert)
    ]
    selected_by_mean = [
        findmax(@view prediction.log_precision_mean[:, observation])[2]
        for observation in eachindex(oracle_expert)
    ]
    return (;
        routing_accuracy = mean(selected .== oracle_expert),
        mean_routing_accuracy = mean(selected_by_mean .== oracle_expert),
        oracle_weight_mean = mean(oracle_weight),
    )
end

function run_dynamic_projector_xor(config; warmup = config.benchmark_warmup)
    data = dynamic_projector_xor_data(config)
    hidden_width = config.xor_hidden_width
    xor_config = merge(config, (;
        prior_variance = config.xor_prior_variance,
        initial_jitter = config.xor_initial_jitter,
        ct_precision_mean = config.xor_ct_precision_mean,
    ))
    if warmup
        warm_dynamic_projector_models(
            data.training,
            data.priors,
            xor_config;
            hidden_width,
        )
    end
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    for arm in (:native, :linear, :nonlinear)
        enabled = arm === :native ? config.run_baseline :
                  arm === :linear ? config.run_linear : config.run_nonlinear
        enabled || continue
        run = run_dynamic_arm(
            :xor,
            arm,
            data.training,
            data.validation,
            data.priors,
            xor_config;
            hidden_width = arm === :native ? 0 : hidden_width,
            iterations = xor_config.xor_iterations,
        )
        routing = dynamic_projector_routing_metrics(
            run.evaluation.prediction,
            data.validation.oracle_expert,
        )
        println(
            "routing accuracy / oracle weight: ",
            round(routing.routing_accuracy; digits = 4),
            " (mean-only ",
            round(routing.mean_routing_accuracy; digits = 4),
            ")",
            " / ",
            round(routing.oracle_weight_mean; digits = 4),
        )
        runs[arm] = merge(run, (; routing))
        push!(rows, merge(dynamic_metrics_row(:xor, run.evaluation), routing))
    end
    path = config.save_outputs ?
        save_dynamic_projector_results(:xor, rows, xor_config) : nothing
    return (; rows, runs, path, selected_width = hidden_width)
end

function save_dynamic_projector_plots(run, evaluation_data, output_prefix)
    prediction = run.evaluation.prediction
    mkpath(dirname(output_prefix))
    heatmap_path = output_prefix * "_expert_weights.png"
    heatmap(
        1:size(prediction.weights, 2),
        1:size(prediction.weights, 1),
        prediction.weights;
        xlabel = "test observation",
        ylabel = "expert",
        title = "Shared-projector normalized expert precision",
        color = :viridis,
    )
    savefig(heatmap_path)

    last_index = min(length(evaluation_data.y), 500)
    first_index = min(250, last_index)
    window = first_index:last_index
    trace_path = output_prefix * "_predictive_trace.png"
    plot(
        window,
        evaluation_data.y[window];
        color = :black,
        linewidth = 1.5,
        label = "target",
        xlabel = "test observation",
        ylabel = "scaled OT",
        title = "Shared nonlinear projector predictive",
    )
    plot!(
        window,
        prediction.mean[window];
        ribbon = 1.96 .* prediction.std[window],
        fillalpha = 0.18,
        linewidth = 2,
        color = :dodgerblue,
        label = "predictive mean / 95%",
    )
    savefig(trace_path)
    return (; heatmap_path, trace_path)
end

function save_dynamic_projector_results(stage, rows, config; suffix = "")
    isempty(rows) && return nothing
    filename = config.output_prefix * "_$(stage)$(suffix)_metrics.csv"
    mkpath(dirname(filename))
    CSV.write(filename, DataFrame(rows))
    return filename
end

function run_dynamic_projector_smoke(data, config)
    n = min(config.smoke_observations, length(data.y_validation) - 1)
    training_data = dynamic_projector_slice(data, 1:n)
    evaluation_data = dynamic_projector_slice(data, (n + 1):min(2n, length(data.y_validation)))
    width = first(config.widths)
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    for arm in (:native, :linear, :nonlinear)
        enabled = arm === :native ? config.run_baseline :
                  arm === :linear ? config.run_linear : config.run_nonlinear
        enabled || continue
        runs[arm] = run_dynamic_arm(
            :smoke,
            arm,
            training_data,
            evaluation_data,
            data.base_priors,
            config;
            hidden_width = arm === :native ? 0 : width,
            iterations = config.smoke_iterations,
        )
        push!(rows, dynamic_metrics_row(:smoke, runs[arm].evaluation))
    end
    path = config.save_outputs ?
        save_dynamic_projector_results(:smoke, rows, config) : nothing
    return (; rows, runs, path, selected_width = width)
end

function run_dynamic_projector_screen(
    data,
    config;
    warmup = config.benchmark_warmup,
)
    n_train = min(config.screen_train, length(data.y_validation) - 1)
    validation_stop = min(
        n_train + config.screen_validation,
        length(data.y_validation),
    )
    training_data = dynamic_projector_slice(data, 1:n_train)
    evaluation_data = dynamic_projector_slice(data, (n_train + 1):validation_stop)
    if warmup
        warm_dynamic_projector_models(
            training_data,
            data.base_priors,
            config;
            hidden_width = first(config.widths),
        )
    end
    rows = NamedTuple[]
    runs = Dict{Tuple{Symbol, Int}, Any}()
    if config.run_baseline
        run = run_dynamic_arm(
            :screen,
            :native,
            training_data,
            evaluation_data,
            data.base_priors,
            config;
            iterations = config.screen_iterations,
        )
        runs[(:native, 0)] = run
        push!(rows, dynamic_metrics_row(:screen, run.evaluation))
    end
    for width in config.widths
        for arm in (:linear, :nonlinear)
            enabled = arm === :linear ? config.run_linear : config.run_nonlinear
            enabled || continue
            run = run_dynamic_arm(
                :screen,
                arm,
                training_data,
                evaluation_data,
                data.base_priors,
                config;
                hidden_width = width,
                iterations = config.screen_iterations,
            )
            runs[(arm, width)] = run
            push!(rows, dynamic_metrics_row(:screen, run.evaluation))
            GC.gc()
        end
    end
    nonlinear_runs = [
        run for ((arm, _), run) in pairs(runs)
        if arm === :nonlinear && run.evaluation.finite
    ]
    isempty(nonlinear_runs) && error("no finite nonlinear projector screen arm")
    selected = nonlinear_runs[argmax([
        run.evaluation.mean_log_likelihood for run in nonlinear_runs
    ])]
    selected_width = selected.evaluation.hidden_width
    println("Selected nonlinear width by validation mean LL: ", selected_width)
    path = config.save_outputs ?
        save_dynamic_projector_results(:screen, rows, config) : nothing
    return (; rows, runs, path, selected_width)
end

function run_dynamic_projector_full(
    data,
    config;
    hidden_width = config.hidden_width,
    warmup = config.benchmark_warmup,
)
    training_data = dynamic_projector_slice(data, eachindex(data.y_validation))
    evaluation_data = dynamic_projector_slice(data, eachindex(data.y_test); test = true)
    if warmup
        warm_dynamic_projector_models(
            training_data,
            data.base_priors,
            config;
            hidden_width,
        )
    end
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    for arm in (:native, :linear, :nonlinear)
        enabled = arm === :native ? config.run_baseline :
                  arm === :linear ? config.run_linear : config.run_nonlinear
        enabled || continue
        run = run_dynamic_arm(
            :full,
            arm,
            training_data,
            evaluation_data,
            data.base_priors,
            config;
            hidden_width = arm === :native ? 0 : hidden_width,
            iterations = config.full_iterations,
        )
        runs[arm] = run
        push!(rows, dynamic_metrics_row(:full, run.evaluation))
        if config.save_outputs && arm === :nonlinear
            save_dynamic_projector_plots(
                run,
                evaluation_data,
                config.output_prefix * "_full_h$(hidden_width)",
            )
            jldsave(
                config.output_prefix * "_full_h$(hidden_width)_predictions.jld2";
                target = evaluation_data.y,
                predictive_mean = run.evaluation.prediction.mean,
                predictive_std = run.evaluation.prediction.std,
                expert_weights = run.evaluation.prediction.weights,
                log_precision_mean = run.evaluation.prediction.log_precision_mean,
                log_precision_variance = run.evaluation.prediction.log_precision_variance,
            )
        end
        GC.gc()
    end
    path = config.save_outputs ?
        save_dynamic_projector_results(:full, rows, config; suffix = "_h$(hidden_width)") : nothing
    return (; rows, runs, path, selected_width = hidden_width)
end

function run_dynamic_projector_study(config = dynamic_projector_config())
    if config.stage === :xor
        return run_dynamic_projector_xor(config)
    end
    data = load_dynamic_projector_data(config)
    println(
        "Dynamic shared-projector study: dataset=$(config.dataset), " *
        "horizon=$(config.horizon), stage=$(config.stage), " *
        "n_validation=$(length(data.y_validation)), n_test=$(length(data.y_test)), " *
        "n_forecasters=$(size(data.predictions_validation, 1)), " *
        "feature_dimension=$(length(first(data.features_validation)))",
    )
    if config.stage === :smoke
        return run_dynamic_projector_smoke(data, config)
    elseif config.stage === :screen
        return run_dynamic_projector_screen(data, config)
    elseif config.stage === :full
        return run_dynamic_projector_full(data, config)
    end

    smoke = run_dynamic_projector_smoke(data, config)
    xor = run_dynamic_projector_xor(config; warmup = false)
    screen = run_dynamic_projector_screen(data, config; warmup = false)
    full = run_dynamic_projector_full(
        data,
        config;
        hidden_width = screen.selected_width,
        warmup = false,
    )
    return (; xor, smoke, screen, full, selected_width = screen.selected_width)
end

if abspath(PROGRAM_FILE) == @__FILE__
    dynamic_projector_study = run_dynamic_projector_study()
end
