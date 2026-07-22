# Binary XOR with a ContinuousTransition / MvSoftplus latent-score model:
#
#   x_f -> ContinuousTransition -> h1 -> MvSoftplus -> s
#       -> ContinuousTransition -> h2 -> softdot(theta, h2, gamma_score)
#       -> z -> Probit -> y
#
# Two paired arms use identical inputs, train/test splits, global priors, and
# initialization.  The clean arm sees exact XOR labels.  In the corrupted arm,
# labels are independently flipped only inside one training quadrant.  Test
# labels are always clean.  The model receives neither a quadrant indicator nor
# a local observation-noise variable, so localization of score uncertainty is
# an empirical result rather than a modeled guarantee.
#
# Smoke run:
#   XOR_CT_PROBIT_SMOKE=true julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit.jl
#
# Full paired run:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Distributions
using ExponentialFamily
using LinearAlgebra: Diagonal
using Random
using ReactiveMP: @call_rule
using RxInfer
using SpecialFunctions: erfc
using StableRNGs
using Statistics
using StatsPlots
using SurrogateModelling

import ExponentialFamily: WishartFast

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default = false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

const QUADRANTS = (:upper_right, :upper_left, :lower_left, :lower_right)

function normalize_quadrant(value)
    quadrant = Symbol(replace(lowercase(string(value)), '-' => '_', ' ' => '_'))
    quadrant in QUADRANTS || throw(ArgumentError(
        "UNTRUSTED_QUADRANT must be one of $(join(string.(QUADRANTS), ", "))",
    ))
    return quadrant
end

const SMOKE = env_bool("XOR_CT_PROBIT_SMOKE")
# The non-smoke score-precision defaults were selected using clean labels only
# by xor_ctransition_mvsoftplus_probit_tuning.jl (8-candidate screen followed
# by two-seed checkpoint/prediction refinement).
const CONFIG = (
    n_samples = env_int("N_SAMPLES", SMOKE ? 60 : 2_000),
    d_hidden = 4,
    iterations = env_int("N_ITERATIONS", SMOKE ? 2 : 160),
    train_fraction = env_float("TRAIN_FRACTION", 0.80),
    feature_jitter = env_float("FEATURE_JITTER", 1e-4),
    ct_precision_mean = env_float("CT_PRECISION_MEAN", 10.0),
    a_prior_mean_scale = env_float("A_PRIOR_MEAN_SCALE", 1.0),
    a_prior_variance = env_float("A_PRIOR_VARIANCE", 1.0),
    theta_prior_mean_scale = env_float("THETA_PRIOR_MEAN_SCALE", 0.0),
    theta_prior_variance = env_float("THETA_PRIOR_VARIANCE", 0.3),
    score_precision_mean = env_float("SCORE_PRECISION_MEAN", 10.0),
    score_precision_concentration = env_float("SCORE_PRECISION_CONCENTRATION", 10.0),
    ngmp_alpha = env_float("NGMP_ALPHA", 0.4),
    ngmp_beta = env_float("NGMP_BETA", 0.2),
    ngmp_max_step = env_float("NGMP_MAX_STEP", 1.0),
    ct_a_alpha = env_float("CT_A_ALPHA", 0.5),
    ct_a_beta = env_float("CT_A_BETA", 0.2),
    ct_a_max_step = env_float("CT_A_MAX_STEP", Inf),
    softplus_output_initial_mean = env_float("SP_OUTPUT_INITIAL_MEAN", log(2.0)),
    softplus_output_initial_variance = env_float("SP_OUTPUT_INITIAL_VARIANCE", 0.04),
    score_initial_variance = env_float("SCORE_INITIAL_VARIANCE", 1.0),
    prediction_iterations = env_int("PREDICTION_ITERATIONS", SMOKE ? 2 : 10),
    prediction_batch_size = env_int("PREDICTION_BATCH_SIZE", SMOKE ? 64 : 1_024),
    prediction_prior_variance = env_float("PREDICTION_PRIOR_VARIANCE", 1e12),
    grid_size = env_int("GRID_SIZE", SMOKE ? 10 : 60),
    untrusted_quadrant = normalize_quadrant(
        get(ENV, "UNTRUSTED_QUADRANT", "upper_right"),
    ),
    quadrant_flip_prob = env_float("QUADRANT_FLIP_PROB", 0.35),
    save_outputs = env_bool("SAVE_OUTPUTS", !SMOKE),
    output_prefix = get(
        ENV,
        "OUTPUT_PREFIX",
        joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvsoftplus_probit"),
    ),
    data_seed = env_int("DATA_SEED", 2_030),
    split_seed = env_int("SPLIT_SEED", 2_031),
    prior_seed = env_int("PRIOR_SEED", 44),
    flip_seed = env_int("FLIP_SEED", 9_031),
    show_progress = env_bool("SHOW_PROGRESS", true),
    require_clean_baseline = env_bool("REQUIRE_CLEAN_BASELINE", !SMOKE),
)

function validate_config(config)
    config.d_hidden == 4 || throw(ArgumentError("this experiment fixes D_HIDDEN at 4"))
    config.n_samples > 1 || throw(ArgumentError("N_SAMPLES must exceed one"))
    0 < config.train_fraction < 1 ||
        throw(ArgumentError("TRAIN_FRACTION must be in (0, 1)"))
    config.iterations > 0 || throw(ArgumentError("N_ITERATIONS must be positive"))
    config.prediction_iterations > 0 ||
        throw(ArgumentError("PREDICTION_ITERATIONS must be positive"))
    config.prediction_batch_size > 0 ||
        throw(ArgumentError("PREDICTION_BATCH_SIZE must be positive"))
    config.prediction_prior_variance > 0 ||
        throw(ArgumentError("PREDICTION_PRIOR_VARIANCE must be positive"))
    config.score_initial_variance > 0 ||
        throw(ArgumentError("SCORE_INITIAL_VARIANCE must be positive"))
    config.grid_size > 1 || throw(ArgumentError("GRID_SIZE must exceed one"))
    0 <= config.quadrant_flip_prob <= 1 ||
        throw(ArgumentError("QUADRANT_FLIP_PROB must be in [0, 1]"))
    config.untrusted_quadrant in QUADRANTS ||
        throw(ArgumentError("invalid untrusted quadrant"))
    config.ct_precision_mean > 0 ||
        throw(ArgumentError("CT_PRECISION_MEAN must be positive"))
    config.a_prior_mean_scale >= 0 ||
        throw(ArgumentError("A_PRIOR_MEAN_SCALE must be nonnegative"))
    config.a_prior_variance > 0 ||
        throw(ArgumentError("A_PRIOR_VARIANCE must be positive"))
    config.theta_prior_mean_scale >= 0 ||
        throw(ArgumentError("THETA_PRIOR_MEAN_SCALE must be nonnegative"))
    config.theta_prior_variance > 0 ||
        throw(ArgumentError("THETA_PRIOR_VARIANCE must be positive"))
    config.score_precision_mean > 0 ||
        throw(ArgumentError("SCORE_PRECISION_MEAN must be positive"))
    config.score_precision_concentration > 0 ||
        throw(ArgumentError("SCORE_PRECISION_CONCENTRATION must be positive"))
    return config
end

validate_config(CONFIG)

# --- Exact binary XOR data and deterministic paired corruption

xor_label(x1, x2) = Float64((x1 >= 0) != (x2 >= 0))

function quadrant_name(x1, x2)
    if x1 >= 0
        return x2 >= 0 ? :upper_right : :lower_right
    end
    return x2 >= 0 ? :upper_left : :lower_left
end

quadrant_mask(x1, x2, quadrant) =
    map((a, b) -> quadrant_name(a, b) == quadrant, x1, x2)

function make_clean_xor_dataset(; n, seed)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    labels = xor_label.(x1, x2)
    all(label -> label == 0.0 || label == 1.0, labels) ||
        error("XOR generator produced a non-binary label")
    return DataFrame(x1 = x1, x2 = x2, label = labels)
end

function split_dataset(df; train_fraction, seed)
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

build_features(df) = [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]

function make_paired_training_labels(
    train_data;
    quadrant = :upper_right,
    flip_probability = 0.35,
    seed = 9_031,
)
    quadrant = normalize_quadrant(quadrant)
    0 <= flip_probability <= 1 ||
        throw(ArgumentError("flip_probability must be in [0, 1]"))
    clean = Float64.(train_data.label)
    all(label -> label == 0.0 || label == 1.0, clean) ||
        error("clean training labels must be binary")

    # Generate one draw for every training item.  The flip RNG is independent
    # of data, split, and prior RNGs, and changing the selected quadrant does
    # not shift the draw assigned to any sample.
    draws = rand(StableRNG(seed), length(clean))
    selected = quadrant_mask(train_data.x1, train_data.x2, quadrant)
    flipped = selected .& (draws .< flip_probability)
    corrupted = copy(clean)
    corrupted[flipped] .= 1.0 .- corrupted[flipped]

    changed = corrupted .!= clean
    changed == flipped || error("corruption bookkeeping does not match changed labels")
    any(changed .& .!selected) && error("a label changed outside the selected quadrant")
    all(label -> label == 0.0 || label == 1.0, corrupted) ||
        error("corruption produced a non-binary label")
    flip_probability == 0 && corrupted != clean &&
        error("zero flip probability must exactly reproduce clean labels")

    return (
        clean = clean,
        corrupted = corrupted,
        selected = selected,
        flipped = flipped,
        draws = draws,
    )
end

# --- Training and frozen-global prediction graphs

@model function xor_ct_mvsoftplus_probit(
    y,
    features,
    priors,
    feature_cov,
    meta_map,
    meta_pred,
    ct_a_deps,
    ct2_deps,
    sp_deps,
    sp_damping,
)
    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_score ~ priors[:gamma_score]
    for i in eachindex(y)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {
            dependencies = ct_a_deps, meta = meta_map
        }
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {
            dependencies = ct2_deps, meta = meta_pred
        }
        z[i] ~ softdot(theta, h2[i], gamma_score)
        y[i] ~ Probit(z[i])
    end
end

@constraints function xor_ct_mvsoftplus_probit_constraints()
    q(x_f, h1, s, h2, z, a_map, a_pred, P, Gamma2, theta, gamma_score) =
        q(x_f, h1)q(s, h2, z)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_score)
end

@model function xor_ct_mvsoftplus_score_prediction(
    features,
    priors,
    feature_cov,
    meta_map,
    meta_pred,
    sp_deps,
    sp_damping,
    score_prior_variance,
)
    local x_f, h1, s, h2, z

    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_score ~ priors[:gamma_score]
    for i in eachindex(features)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {meta = meta_map}
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {meta = meta_pred}
        z[i] ~ softdot(theta, h2[i], gamma_score)
        z[i] ~ NormalMeanVariance(0.0, score_prior_variance)
    end
end

@constraints function xor_ct_mvsoftplus_score_prediction_constraints(priors)
    q(x_f, h1, s, h2, z, a_map, a_pred, P, Gamma2, theta, gamma_score) =
        q(x_f, h1)q(s, h2, z)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_score)

    q(a_map)::RxInfer.FixedMarginalFormConstraint(priors[:a_map])
    q(a_pred)::RxInfer.FixedMarginalFormConstraint(priors[:a_pred])
    q(theta)::RxInfer.FixedMarginalFormConstraint(priors[:theta])
    q(P)::RxInfer.FixedMarginalFormConstraint(priors[:P])
    q(Gamma2)::RxInfer.FixedMarginalFormConstraint(priors[:Gamma2])
    q(gamma_score)::RxInfer.FixedMarginalFormConstraint(priors[:gamma_score])
end

function make_priors(config; seed = config.prior_seed, d_f = 3)
    d_h = config.d_hidden
    rng = StableRNG(seed)
    nu = d_h + 2.0
    inv_scale = Matrix(Diagonal(fill(nu / config.ct_precision_mean, d_h)))
    return Dict{Symbol, Any}(
        :a_map => MvNormalMeanCovariance(
            config.a_prior_mean_scale .* randn(rng, d_h * d_f),
            config.a_prior_variance .* Diagonal(ones(d_h * d_f)),
        ),
        :a_pred => MvNormalMeanCovariance(
            config.a_prior_mean_scale .* randn(rng, d_h * d_h),
            config.a_prior_variance .* Diagonal(ones(d_h * d_h)),
        ),
        :theta => MvNormalMeanCovariance(
            config.theta_prior_mean_scale .* randn(rng, d_h),
            config.theta_prior_variance .* Diagonal(ones(d_h)),
        ),
        :P => WishartFast(nu, inv_scale),
        :Gamma2 => WishartFast(nu, inv_scale),
        :gamma_score => GammaShapeRate(
            config.score_precision_concentration,
            config.score_precision_concentration / config.score_precision_mean,
        ),
    )
end

function make_initialization(config, priors)
    d_h = config.d_hidden
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(gamma_score) = priors[:gamma_score]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(
            fill(config.softplus_output_initial_mean, d_h),
            Diagonal(fill(config.softplus_output_initial_variance, d_h)),
        )
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(z) = NormalMeanVariance(0.0, config.score_initial_variance)
    end
end

function make_prediction_initialization(config, priors)
    d_h = config.d_hidden
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(gamma_score) = priors[:gamma_score]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(
            fill(config.softplus_output_initial_mean, d_h),
            Diagonal(fill(config.softplus_output_initial_variance, d_h)),
        )
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(z) = NormalMeanVariance(0.0, config.prediction_prior_variance)
        μ(z) = NormalMeanVariance(0.0, config.prediction_prior_variance)
    end
end

make_mvsoftplus_dependencies() = NGMPDependencies(
    out = nothing,
    in = nothing,
    projection = TangentProjection(type = Unscented),
)

function make_ct_dependencies(config)
    damping = DampingMeta(
        alpha = config.ct_a_alpha,
        beta = config.ct_a_beta,
        max_step = config.ct_a_max_step,
    )
    return NGMPDependencies(a = nothing, damping = damping)
end

function fit_probit_arm(
    config,
    features,
    labels;
    iterations = config.iterations,
    priors = nothing,
    meta_map = nothing,
    meta_pred = nothing,
    compute_free_energy = true,
)
    validate_config(config)
    all(label -> label == 0.0 || label == 1.0, labels) ||
        error("Probit observations must be exact binary labels")
    length(features) == length(labels) || throw(DimensionMismatch(
        "features and labels must contain the same number of samples",
    ))

    d_f = length(first(features))
    priors = isnothing(priors) ? make_priors(config; d_f = d_f) : priors
    meta_map = isnothing(meta_map) ? LinearReshapeMeta(config.d_hidden, d_f) : meta_map
    meta_pred = isnothing(meta_pred) ?
        LinearReshapeMeta(config.d_hidden, config.d_hidden) : meta_pred
    ct_a_deps = make_ct_dependencies(config)
    ct2_deps = make_ct_dependencies(config)
    sp_deps = make_mvsoftplus_dependencies()
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )

    elapsed = @elapsed result = infer(
        model = xor_ct_mvsoftplus_probit(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = meta_map,
            meta_pred = meta_pred,
            ct_a_deps = ct_a_deps,
            ct2_deps = ct2_deps,
            sp_deps = sp_deps,
            sp_damping = sp_damping,
        ),
        data = (y = labels, features = features),
        constraints = xor_ct_mvsoftplus_probit_constraints(),
        initialization = make_initialization(config, priors),
        iterations = iterations,
        free_energy = compute_free_energy,
        showprogress = config.show_progress,
        returnvars = (
            a_map = KeepEach(),
            a_pred = KeepEach(),
            theta = KeepEach(),
            P = KeepEach(),
            Gamma2 = KeepEach(),
            gamma_score = KeepEach(),
        ),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    fit = (
        result = result,
        priors = priors,
        ct_a_deps = ct_a_deps,
        ct2_deps = ct2_deps,
        sp_deps = sp_deps,
        meta_map = meta_map,
        meta_pred = meta_pred,
        iterations = iterations,
        elapsed = elapsed,
        compute_free_energy = compute_free_energy,
    )
    validate_training_fit(fit, length(labels))
    return fit
end

const GLOBAL_KEYS = (:a_map, :a_pred, :theta, :P, :Gamma2, :gamma_score)

function prediction_priors(fit, iteration = nothing)
    return Dict{Symbol, Any}(
        key => deepcopy(
            isnothing(iteration) ?
            last(fit.result.posteriors[key]) :
            fit.result.posteriors[key][iteration],
        ) for key in GLOBAL_KEYS
    )
end

function run_score_prediction_batch(
    priors,
    features;
    config,
    meta_map = nothing,
    meta_pred = nothing,
)
    isempty(features) && return Any[]
    d_f = length(first(features))
    meta_map = isnothing(meta_map) ? LinearReshapeMeta(config.d_hidden, d_f) : meta_map
    meta_pred = isnothing(meta_pred) ?
        LinearReshapeMeta(config.d_hidden, config.d_hidden) : meta_pred
    sp_deps = make_mvsoftplus_dependencies()
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )
    result = infer(
        model = xor_ct_mvsoftplus_score_prediction(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = meta_map,
            meta_pred = meta_pred,
            sp_deps = sp_deps,
            sp_damping = sp_damping,
            score_prior_variance = config.prediction_prior_variance,
        ),
        data = (features = features,),
        constraints = xor_ct_mvsoftplus_score_prediction_constraints(priors),
        initialization = make_prediction_initialization(config, priors),
        iterations = config.prediction_iterations,
        free_energy = false,
        showprogress = false,
        returnvars = (z = KeepLast(),),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    marginals = collect(vec(result.posteriors[:z]))
    length(marginals) == length(features) ||
        error("prediction graph returned the wrong number of score marginals")
    return marginals
end

function predict_score_marginals(fit, features; config, iteration = nothing)
    priors = prediction_priors(fit, iteration)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1,
            length(features),
        )
        marginals[indices] = run_score_prediction_batch(
            priors,
            features[indices];
            config = config,
            meta_map = fit.meta_map,
            meta_pred = fit.meta_pred,
        )
    end
    return marginals
end

standard_normal_cdf(x) = 0.5 * erfc(-x / sqrt(2.0))
probit_probability(m, v) = standard_normal_cdf(m / sqrt(1 + v))

function native_probit_probability(qz)
    predictive = @call_rule(
        Probit(:out, Marginalisation),
        (m_in = qz, meta = nothing),
    )
    return Float64(mean(predictive))
end

function verify_native_probit_probability()
    probes = (
        NormalMeanVariance(-1.7, 0.05),
        NormalMeanVariance(0.0, 1.0),
        NormalMeanVariance(2.2, 4.0),
    )
    for qz in probes
        analytic = probit_probability(mean(qz), var(qz))
        native = native_probit_probability(qz)
        isapprox(analytic, native; atol = 2e-14, rtol = 2e-14) || error(
            "analytic Probit prediction disagrees with ReactiveMP: $analytic != $native",
        )
    end
    return true
end

function score_statistics(marginals; verify_native = true)
    score_mean = Float64.(mean.(marginals))
    score_variance = Float64.(var.(marginals))
    all(isfinite, score_mean) || error("prediction produced a non-finite score mean")
    all(value -> isfinite(value) && value > 0, score_variance) ||
        error("prediction produced a non-positive or non-finite score variance")

    probability = probit_probability.(score_mean, score_variance)
    bernoulli_variance = probability .* (1 .- probability)
    all(value -> isfinite(value) && 0 <= value <= 1, probability) ||
        error("predictive probability lies outside [0, 1]")
    all(value -> isfinite(value) && 0 <= value <= 0.25 + 8eps(Float64), bernoulli_variance) ||
        error("Bernoulli variance lies outside [0, 0.25]")

    if verify_native
        for index in eachindex(marginals)
            isapprox(
                probability[index],
                native_probit_probability(marginals[index]);
                atol = 2e-12,
                rtol = 2e-12,
            ) || error("analytic and native Probit predictions disagree at index $index")
        end
    end

    epsilon = eps(Float64)
    clipped = clamp.(probability, epsilon, 1 - epsilon)
    entropy = @. -clipped * log(clipped) - (1 - clipped) * log(1 - clipped)
    return (
        probability = probability,
        bernoulli_variance = bernoulli_variance,
        score_mean = score_mean,
        score_variance = score_variance,
        entropy = entropy,
    )
end

function validate_training_fit(fit, n_train)
    if fit.compute_free_energy
        all(isfinite, fit.result.free_energy) || error("training free energy is non-finite")
    end
    for key in GLOBAL_KEYS
        posterior_mean = mean(last(fit.result.posteriors[key]))
        all(isfinite, posterior_mean) || error("posterior mean for $key is non-finite")
    end

    length(fit.ct_a_deps.states) == n_train || error(
        "first CT layer created $(length(fit.ct_a_deps.states)) damping states; expected $n_train",
    )
    length(fit.ct2_deps.states) == n_train || error(
        "second CT layer created $(length(fit.ct2_deps.states)) damping states; expected $n_train",
    )
    all(state -> state.nfired == fit.iterations, fit.ct_a_deps.states) ||
        error("first CT layer did not fire exactly once per iteration")
    all(state -> state.nfired == fit.iterations, fit.ct2_deps.states) ||
        error("second CT layer did not fire exactly once per iteration")
    return true
end

# --- Metrics and localization diagnostics

function classification_metrics(labels, probabilities)
    length(labels) == length(probabilities) || throw(DimensionMismatch(
        "labels and probabilities must have equal length",
    ))
    isempty(labels) && return (accuracy = NaN, nll = NaN)
    clipped = clamp.(probabilities, eps(Float64), 1 - eps(Float64))
    accuracy = mean((probabilities .>= 0.5) .== (labels .== 1.0))
    nll = -mean(@. labels * log(clipped) + (1 - labels) * log(1 - clipped))
    return (accuracy = accuracy, nll = nll)
end

function diagnostic_row(arm, quadrant, indices, labels, statistics)
    metrics = classification_metrics(labels[indices], statistics.probability[indices])
    if isempty(indices)
        return (
            arm = string(arm),
            quadrant = string(quadrant),
            accuracy = metrics.accuracy,
            nll = metrics.nll,
            latent_variance = NaN,
            bernoulli_variance = NaN,
            entropy = NaN,
            probability_min = NaN,
            probability_max = NaN,
            sample_count = 0,
        )
    end
    return (
        arm = string(arm),
        quadrant = string(quadrant),
        accuracy = metrics.accuracy,
        nll = metrics.nll,
        latent_variance = mean(statistics.score_variance[indices]),
        bernoulli_variance = mean(statistics.bernoulli_variance[indices]),
        entropy = mean(statistics.entropy[indices]),
        probability_min = minimum(statistics.probability[indices]),
        probability_max = maximum(statistics.probability[indices]),
        sample_count = length(indices),
    )
end

function quadrant_diagnostics(arm, test_data, statistics)
    labels = Float64.(test_data.label)
    rows = NamedTuple[
        diagnostic_row(arm, :overall, eachindex(labels), labels, statistics),
    ]
    for quadrant in QUADRANTS
        mask = quadrant_mask(test_data.x1, test_data.x2, quadrant)
        indices = findall(mask)
        push!(rows, diagnostic_row(arm, quadrant, indices, labels, statistics))
    end
    return DataFrame(rows)
end

function localization_metrics(statistics, test_data, untrusted_quadrant)
    untrusted = quadrant_mask(test_data.x1, test_data.x2, untrusted_quadrant)
    reliable = .!untrusted
    if !any(untrusted) || !any(reliable)
        return (
            untrusted_latent_variance = NaN,
            reliable_latent_variance = NaN,
            latent_variance_ratio = NaN,
            latent_variance_difference = NaN,
        )
    end
    untrusted_variance = mean(statistics.score_variance[untrusted])
    reliable_variance = mean(statistics.score_variance[reliable])
    return (
        untrusted_latent_variance = untrusted_variance,
        reliable_latent_variance = reliable_variance,
        latent_variance_ratio = untrusted_variance / reliable_variance,
        latent_variance_difference = untrusted_variance - reliable_variance,
    )
end

function class_prior_baseline(train_labels, test_labels)
    probability = clamp(mean(train_labels), eps(Float64), 1 - eps(Float64))
    metrics = classification_metrics(test_labels, fill(probability, length(test_labels)))
    return merge((probability = probability,), metrics)
end

# --- Five-panel matched surfaces

function make_prediction_grid(grid_size)
    x = range(-2.0, 2.0; length = grid_size)
    y = range(-2.0, 2.0; length = grid_size)
    actual = [xor_label(x_value, y_value) for y_value in y, x_value in x]
    features = vec([[1.0, x_value, y_value] for y_value in y, x_value in x])
    return (x = x, y = y, actual = actual, features = features)
end

function reshape_grid_statistics(statistics, grid)
    shape = (length(grid.y), length(grid.x))
    return (
        probability = reshape(statistics.probability, shape),
        bernoulli_variance = reshape(statistics.bernoulli_variance, shape),
        score_mean = reshape(statistics.score_mean, shape),
        score_variance = reshape(statistics.score_variance, shape),
    )
end

function nondegenerate_limits(lower, upper)
    lower == upper && return (lower, nextfloat(upper))
    return (lower, upper)
end

function save_five_panel_surface(
    arm,
    grid,
    prediction,
    output_prefix;
    score_mean_limits,
    score_variance_limits,
)
    probability_panel = contourf(
        grid.x, grid.y, prediction.probability;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Class probability",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    bernoulli_panel = contourf(
        grid.x, grid.y, prediction.bernoulli_variance;
        color = :viridis,
        levels = 20,
        clims = (0, 0.25),
        xlabel = "x1",
        ylabel = "x2",
        title = "Bernoulli variance",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    score_mean_panel = contourf(
        grid.x, grid.y, prediction.score_mean;
        color = :RdBu,
        levels = 20,
        clims = score_mean_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Latent-score mean",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    score_variance_panel = contourf(
        grid.x, grid.y, prediction.score_variance;
        color = :viridis,
        levels = 20,
        clims = score_variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Latent-score variance",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    actual_panel = heatmap(
        grid.x, grid.y, grid.actual;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Clean XOR target",
        aspect_ratio = :equal,
    )
    figure = plot(
        probability_panel,
        bernoulli_panel,
        score_mean_panel,
        score_variance_panel,
        actual_panel;
        layout = (1, 5),
        size = (2_250, 430),
    )
    path = output_prefix * "_$(arm)_surfaces.png"
    mkpath(dirname(path))
    savefig(figure, path)
    return path
end

function save_matched_surfaces(grid, clean, corrupted, output_prefix)
    all_score_means = vcat(vec(clean.score_mean), vec(corrupted.score_mean))
    all_score_variances = vcat(vec(clean.score_variance), vec(corrupted.score_variance))
    score_mean_limits = nondegenerate_limits(
        minimum(all_score_means),
        maximum(all_score_means),
    )
    score_variance_limits = nondegenerate_limits(
        minimum(all_score_variances),
        maximum(all_score_variances),
    )
    return (
        clean = save_five_panel_surface(
            :clean,
            grid,
            clean,
            output_prefix;
            score_mean_limits = score_mean_limits,
            score_variance_limits = score_variance_limits,
        ),
        corrupted = save_five_panel_surface(
            :corrupted,
            grid,
            corrupted,
            output_prefix;
            score_mean_limits = score_mean_limits,
            score_variance_limits = score_variance_limits,
        ),
    )
end

function verify_matched_priors(clean_fit, corrupted_fit)
    for key in GLOBAL_KEYS
        mean(clean_fit.priors[key]) == mean(corrupted_fit.priors[key]) ||
            error("paired arms do not share the same prior mean for $key")
    end
    return true
end

function print_paired_report(report)
    println()
    println(
        "=== xor_ctransition_mvsoftplus_probit " *
        "(width = 4, iterations = $(report.config.iterations), " *
        "n_train = $(nrow(report.train_data)), n_test = $(nrow(report.test_data)))",
    )
    println(
        "untrusted quadrant / flip probability : ",
        report.config.untrusted_quadrant,
        " / ",
        report.config.quadrant_flip_prob,
    )
    println(
        "selected / flipped training labels    : ",
        count(report.labels.selected),
        " / ",
        count(report.labels.flipped),
    )
    for arm in (:clean, :corrupted)
        arm_report = getproperty(report.arms, arm)
        overall = arm_report.diagnostics[arm_report.diagnostics.quadrant .== "overall", :]
        println(
            rpad(string(arm), 10),
            " accuracy/NLL = ",
            round(overall.accuracy[1], digits = 4),
            " / ",
            round(overall.nll[1], digits = 4),
            "; free energy first/last = ",
            first(arm_report.fit.result.free_energy),
            " / ",
            last(arm_report.fit.result.free_energy),
            "; elapsed = ",
            round(arm_report.fit.elapsed, digits = 1),
            "s",
        )
    end
    println(
        "class-prior baseline accuracy/NLL     : ",
        round(report.clean_baseline.accuracy, digits = 4),
        " / ",
        round(report.clean_baseline.nll, digits = 4),
    )
    println("--- quadrant test diagnostics")
    show(stdout, MIME("text/plain"), report.diagnostics)
    println()
    println("--- uncertainty localization (test set)")
    show(stdout, MIME("text/plain"), DataFrame([report.localization]))
    println()
    isnothing(report.surface_paths) || println("surface files: ", report.surface_paths)
end

function run_paired_experiment(config = CONFIG)
    validate_config(config)
    verify_native_probit_probability()

    dataset = make_clean_xor_dataset(n = config.n_samples, seed = config.data_seed)
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    labels = make_paired_training_labels(
        train_data;
        quadrant = config.untrusted_quadrant,
        flip_probability = config.quadrant_flip_prob,
        seed = config.flip_seed,
    )
    labels.clean == train_data.label || error("the clean arm changed a training label")
    all(test_data.label .== xor_label.(test_data.x1, test_data.x2)) ||
        error("test labels are not clean XOR labels")

    train_features = build_features(train_data)
    test_features = build_features(test_data)
    clean_fit = fit_probit_arm(config, train_features, labels.clean)
    corrupted_fit = fit_probit_arm(config, train_features, labels.corrupted)
    verify_matched_priors(clean_fit, corrupted_fit)

    clean_statistics = score_statistics(predict_score_marginals(
        clean_fit,
        test_features;
        config = config,
    ))
    corrupted_statistics = score_statistics(predict_score_marginals(
        corrupted_fit,
        test_features;
        config = config,
    ))
    if config.quadrant_flip_prob == 0
        for field in (
            :probability,
            :bernoulli_variance,
            :score_mean,
            :score_variance,
            :entropy,
        )
            clean_values = getproperty(clean_statistics, field)
            corrupted_values = getproperty(corrupted_statistics, field)
            all(isapprox.(clean_values, corrupted_values; atol = 1e-12, rtol = 1e-12)) ||
                error("zero flip probability did not reproduce clean $field predictions")
        end
    end
    clean_diagnostics = quadrant_diagnostics(:clean, test_data, clean_statistics)
    corrupted_diagnostics = quadrant_diagnostics(
        :corrupted,
        test_data,
        corrupted_statistics,
    )
    diagnostics = vcat(clean_diagnostics, corrupted_diagnostics)

    clean_localization = localization_metrics(
        clean_statistics,
        test_data,
        config.untrusted_quadrant,
    )
    corrupted_localization = localization_metrics(
        corrupted_statistics,
        test_data,
        config.untrusted_quadrant,
    )
    localization = (
        untrusted_quadrant = string(config.untrusted_quadrant),
        clean_untrusted_latent_variance = clean_localization.untrusted_latent_variance,
        clean_reliable_latent_variance = clean_localization.reliable_latent_variance,
        clean_latent_variance_ratio = clean_localization.latent_variance_ratio,
        clean_latent_variance_difference = clean_localization.latent_variance_difference,
        corrupted_untrusted_latent_variance = corrupted_localization.untrusted_latent_variance,
        corrupted_reliable_latent_variance = corrupted_localization.reliable_latent_variance,
        corrupted_latent_variance_ratio = corrupted_localization.latent_variance_ratio,
        corrupted_latent_variance_difference = corrupted_localization.latent_variance_difference,
        latent_variance_difference_in_differences =
            corrupted_localization.latent_variance_difference -
            clean_localization.latent_variance_difference,
    )

    clean_baseline = class_prior_baseline(labels.clean, Float64.(test_data.label))
    clean_overall = only(eachrow(clean_diagnostics[clean_diagnostics.quadrant .== "overall", :]))
    if config.require_clean_baseline
        clean_overall.accuracy > clean_baseline.accuracy || error(
            "clean arm accuracy $(clean_overall.accuracy) did not beat " *
            "class-prior baseline $(clean_baseline.accuracy)",
        )
        clean_overall.nll < clean_baseline.nll || error(
            "clean arm NLL $(clean_overall.nll) did not beat " *
            "class-prior baseline $(clean_baseline.nll)",
        )
    end

    surface_paths = nothing
    if config.save_outputs
        grid = make_prediction_grid(config.grid_size)
        clean_grid = reshape_grid_statistics(score_statistics(predict_score_marginals(
            clean_fit,
            grid.features;
            config = config,
        ); verify_native = false), grid)
        corrupted_grid = reshape_grid_statistics(score_statistics(predict_score_marginals(
            corrupted_fit,
            grid.features;
            config = config,
        ); verify_native = false), grid)
        surface_paths = save_matched_surfaces(
            grid,
            clean_grid,
            corrupted_grid,
            config.output_prefix,
        )
        mkpath(dirname(config.output_prefix))
        CSV.write(config.output_prefix * "_quadrant_metrics.csv", diagnostics)
        CSV.write(
            config.output_prefix * "_localization.csv",
            DataFrame([localization]),
        )
    end

    report = (
        config = config,
        train_data = train_data,
        test_data = test_data,
        labels = labels,
        arms = (
            clean = (
                fit = clean_fit,
                statistics = clean_statistics,
                diagnostics = clean_diagnostics,
                localization = clean_localization,
            ),
            corrupted = (
                fit = corrupted_fit,
                statistics = corrupted_statistics,
                diagnostics = corrupted_diagnostics,
                localization = corrupted_localization,
            ),
        ),
        diagnostics = diagnostics,
        localization = localization,
        clean_baseline = clean_baseline,
        surface_paths = surface_paths,
    )
    print_paired_report(report)
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    report = run_paired_experiment(CONFIG)
end
