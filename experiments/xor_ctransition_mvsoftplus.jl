# XOR (2x2 checkerboard) with a ContinuousTransition / MvSoftplus sandwich:
#
#   x_f -> CTransition(a_map, P) -> h1 -> MvSoftplus -> s
#       -> CTransition(a_pred, Gamma2) -> h2 -> softdot(theta, gamma_obs) -> y
#
# The idea under test: ContinuousTransition maintains a structured JOINT
# Gaussian q(y, x) between each linear layer's output and input, while the new
# MvSoftplus node passes the elementwise nonlinearity with NGMP messages —
# a moment-matched Gaussian pushforward forward, and a genuine unscented
# tangent projection of the exact backward log-message backward. This replaces
# the per-scalar softplus gating of notebooks/xor_softplus_ut_ngmp.jl with a
# real vector-valued hidden layer.
#
# Smoke run: XOR_CT_SMOKE=true julia --project=. experiments/xor_ctransition_mvsoftplus.jl
# Full run:  OPENBLAS_NUM_THREADS=1 julia --project=. experiments/xor_ctransition_mvsoftplus.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DataFrames
using Distributions
using ExponentialFamily
using LinearAlgebra: Diagonal, dot
using Random
using RxInfer
using StableRNGs
using StatsPlots
using Statistics
using SurrogateModelling

import ExponentialFamily: WishartFast
import SurrogateModelling: _softplus

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default = false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

const SMOKE = env_bool("XOR_CT_SMOKE")
const CONFIG = (
    n_samples = env_int("N_SAMPLES", SMOKE ? 60 : 1_600),
    d_hidden = env_int("D_HIDDEN", SMOKE ? 2 : 8),
    iterations = env_int("N_ITERATIONS", SMOKE ? 3 : 50),
    train_fraction = env_float("TRAIN_FRACTION", 0.40),
    noise_std = env_float("NOISE_STD", 0.10),
    feature_jitter = env_float("FEATURE_JITTER", 1e-4),
    ct_precision_mean = env_float("CT_PRECISION_MEAN", 10.0),
    ngmp_alpha = env_float("NGMP_ALPHA", 0.2),
    ngmp_beta = env_float("NGMP_BETA", 0.0),
    ngmp_max_step = env_float("NGMP_MAX_STEP", 1.0),
    prediction_iterations = env_int("PREDICTION_ITERATIONS", SMOKE ? 3 : 20),
    prediction_batch_size = env_int("PREDICTION_BATCH_SIZE", SMOKE ? 64 : 1_024),
    prediction_prior_variance = env_float("PREDICTION_PRIOR_VARIANCE", 1e12),
    grid_size = env_int("GRID_SIZE", SMOKE ? 16 : 60),
    save_outputs = env_bool("SAVE_OUTPUTS", !SMOKE),
    output_prefix = get(
        ENV,
        "OUTPUT_PREFIX",
        joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvsoftplus"),
    ),
    data_seed = env_int("DATA_SEED", 2_026),
    split_seed = env_int("SPLIT_SEED", 2_027),
    prior_seed = env_int("PRIOR_SEED", 42),
    diagnostics = env_bool("DIAGNOSTICS"),
)

0 < CONFIG.train_fraction < 1 || throw(ArgumentError("TRAIN_FRACTION must be in (0, 1)"))
CONFIG.iterations > 0 || throw(ArgumentError("N_ITERATIONS must be positive"))
CONFIG.prediction_iterations > 0 || throw(ArgumentError("PREDICTION_ITERATIONS must be positive"))
CONFIG.prediction_batch_size > 0 || throw(ArgumentError("PREDICTION_BATCH_SIZE must be positive"))
CONFIG.prediction_prior_variance > 0 ||
    throw(ArgumentError("PREDICTION_PRIOR_VARIANCE must be positive"))
CONFIG.grid_size > 1 || throw(ArgumentError("GRID_SIZE must exceed one"))

# --- Data: 2x2 checkerboard = XOR on [-2, 2]^2 (same generator as
# --- notebooks/xor_softplus_ut_ngmp.jl).

function checkerboard_label(x1, x2, checkerboard_size)
    nx, ny = checkerboard_size
    cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
    cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
    return Float64(isodd(cell_x + cell_y))
end

function make_checkerboard_dataset(; n, checkerboard_size = (2, 2), noise_std = 0.10, seed = 1011)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    clean = checkerboard_label.(x1, x2, Ref(checkerboard_size))
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
    return DataFrame(x1 = x1, x2 = x2, OT = target)
end

function split_dataset(df; train_fraction = 0.30, seed = 42)
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

build_features(df) = [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]

# --- Model

@model function xor_ct_mvsoftplus(y, features, priors, feature_cov, meta_map, meta_pred, sp_deps, sp_damping)
    a_map ~ priors[:a_map]         # vec of the d_h x d_f layer-1 matrix
    a_pred ~ priors[:a_pred]       # vec of the d_h x d_h layer-2 matrix
    theta ~ priors[:theta]         # readout weights
    P ~ priors[:P]                 # layer-1 transition precision (Wishart)
    Gamma2 ~ priors[:Gamma2]       # layer-2 transition precision (Wishart)
    gamma_obs ~ priors[:gamma_obs] # readout precision (Gamma)
    for i in eachindex(y)
        # ContinuousTransition has no PointMass rules on its x interface, so
        # observed features enter through a tight pseudo-observed latent.
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {meta = meta_map}
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {meta = meta_pred}
        y[i] ~ softdot(theta, h2[i], gamma_obs)
    end
end

# The joint pairs give each ContinuousTransition its structured q(y, x);
# everything else is mean-field. MvSoftplus is deterministic — ReactiveMP
# forces full factorization on it regardless of this statement.
@constraints function xor_ct_constraints()
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs) =
        q(x_f, h1)q(s, h2)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)
end

function make_priors(; d_h, d_f, seed, ct_precision_mean)
    rng = StableRNG(seed)
    ν = d_h + 2.0
    # WishartFast(ν, invS): mean = ν inv(invS) = ct_precision_mean I. Plain
    # Wishart breaks the free-energy scorer (no closed-form KL in Distributions;
    # the bridge in src/kl_divergences.jl covers the WishartFast path).
    inv_scale = Matrix(Diagonal(fill(ν / ct_precision_mean, d_h)))
    return Dict{Symbol, Any}(
        # Random prior means on the layer matrices break the hidden-unit symmetry.
        :a_map => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_f), Diagonal(ones(d_h * d_f))),
        :a_pred => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_h), Diagonal(ones(d_h * d_h))),
        :theta => MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h))),
        :P => WishartFast(ν, inv_scale),
        :Gamma2 => WishartFast(ν, inv_scale),
        :gamma_obs => GammaShapeRate(1.0, 1.0),
    )
end

function make_initialization(priors, d_h)
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(gamma_obs) = priors[:gamma_obs]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        # Positive-mean, tight init keeps the layer-2 input in the softplus range.
        q(s) = MvNormalMeanCovariance(fill(log(2.0), d_h), Diagonal(fill(0.04, d_h)))
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
    end
end

# --- Prediction

# Prediction is inference in a second graph. The learned global posteriors are
# frozen, while the local ContinuousTransition / MvSoftplus pipeline is inferred
# again for every new feature vector. The diffuse Normal factor makes y an
# effectively unobserved output whose marginal carries predictive uncertainty.
@model function xor_ct_mvsoftplus_prediction(
    features,
    priors,
    feature_cov,
    meta_map,
    meta_pred,
    sp_deps,
    sp_damping,
    y_prior_variance,
)
    local x_f, h1, s, h2, y

    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_obs ~ priors[:gamma_obs]
    for i in eachindex(features)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {meta = meta_map}
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {meta = meta_pred}
        y[i] ~ softdot(theta, h2[i], gamma_obs)
        y[i] ~ NormalMeanVariance(0.0, y_prior_variance)
    end
end

@constraints function xor_ct_prediction_constraints(priors)
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs, y) =
        q(x_f, h1)q(s, h2, y)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)

    q(a_map)::RxInfer.FixedMarginalFormConstraint(priors[:a_map])
    q(a_pred)::RxInfer.FixedMarginalFormConstraint(priors[:a_pred])
    q(theta)::RxInfer.FixedMarginalFormConstraint(priors[:theta])
    q(P)::RxInfer.FixedMarginalFormConstraint(priors[:P])
    q(Gamma2)::RxInfer.FixedMarginalFormConstraint(priors[:Gamma2])
    q(gamma_obs)::RxInfer.FixedMarginalFormConstraint(priors[:gamma_obs])
end

function make_prediction_initialization(priors, d_h, output_mean, y_prior_variance)
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(gamma_obs) = priors[:gamma_obs]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(fill(log(2.0), d_h), Diagonal(fill(0.04, d_h)))
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(y) = NormalMeanVariance(output_mean, y_prior_variance)
        μ(y) = NormalMeanVariance(output_mean, y_prior_variance)
    end
end

function prediction_priors(result)
    return Dict{Symbol, Any}(
        key => deepcopy(last(result.posteriors[key])) for
        key in (:a_map, :a_pred, :theta, :P, :Gamma2, :gamma_obs)
    )
end

function run_prediction_batch(priors, features; config, d_h, d_f, output_mean)
    isempty(features) && return Any[]

    # NGMPDependencies keeps mutable edge state, so every batch receives a
    # fresh instance rather than inheriting damping state from another batch.
    sp_deps = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )
    result = infer(
        model = xor_ct_mvsoftplus_prediction(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = CTMeta(a -> reshape(a, d_h, d_f)),
            meta_pred = CTMeta(a -> reshape(a, d_h, d_h)),
            sp_deps = sp_deps,
            sp_damping = sp_damping,
            y_prior_variance = config.prediction_prior_variance,
        ),
        data = (features = features,),
        constraints = xor_ct_prediction_constraints(priors),
        initialization = make_prediction_initialization(
            priors,
            d_h,
            output_mean,
            config.prediction_prior_variance,
        ),
        iterations = config.prediction_iterations,
        free_energy = false,
        showprogress = false,
        returnvars = (y = KeepLast(),),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    marginals = collect(vec(result.posteriors[:y]))
    length(marginals) == length(features) ||
        error("prediction graph returned the wrong number of y marginals")
    return marginals
end

function predict_marginals(priors, features; batch_size, kwargs...)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:batch_size:length(features)
        indices = first_index:min(first_index + batch_size - 1, length(features))
        marginals[indices] = run_prediction_batch(priors, features[indices]; kwargs...)
    end
    return marginals
end

function predictive_statistics(marginals)
    means = Float64.(mean.(marginals))
    variances = Float64.(var.(marginals))
    all(isfinite, means) || error("prediction graph produced a non-finite mean")
    all(variance -> isfinite(variance) && variance > 0, variances) ||
        error("prediction graph produced a non-positive or non-finite variance")
    return (mean = means, variance = variances)
end

function plugin_prediction_means(result, features, d_h, d_f)
    A_map = reshape(mean(last(result.posteriors[:a_map])), d_h, d_f)
    A_pred = reshape(mean(last(result.posteriors[:a_pred])), d_h, d_h)
    theta_hat = mean(last(result.posteriors[:theta]))
    return [dot(theta_hat, A_pred * _softplus.(A_map * feature)) for feature in features]
end

function make_prediction_grid(grid_size)
    x = range(-2.0, 2.0; length = grid_size)
    y = range(-2.0, 2.0; length = grid_size)
    actual = [checkerboard_label(x_value, y_value, (2, 2)) for y_value in y, x_value in x]
    features = vec([[1.0, x_value, y_value] for y_value in y, x_value in x])
    return (x = x, y = y, actual = actual, features = features)
end

function save_predictive_surface(grid, prediction, output_prefix)
    variance_lower = minimum(prediction.variance)
    variance_upper = maximum(prediction.variance)
    variance_limits = variance_lower == variance_upper ?
        (variance_lower, nextfloat(variance_upper)) : (variance_lower, variance_upper)

    mean_panel = contourf(
        grid.x,
        grid.y,
        prediction.mean;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive mean",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    variance_panel = contourf(
        grid.x,
        grid.y,
        prediction.variance;
        color = :viridis,
        levels = 20,
        clims = variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive variance q(y)",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    actual_panel = heatmap(
        grid.x,
        grid.y,
        grid.actual;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Clean 2x2 target",
        aspect_ratio = :equal,
    )
    figure = plot(
        mean_panel,
        variance_panel,
        actual_panel;
        layout = (1, 3),
        size = (1_350, 420),
        plot_title = "ContinuousTransition / MvSoftplus posterior prediction",
    )
    filename = output_prefix * "_predictive.png"
    mkpath(dirname(filename))
    savefig(figure, filename)
    return filename
end

# --- Run

function run_experiment(config)
    d_h = config.d_hidden
    d_f = 3

    dataset = make_checkerboard_dataset(
        n = config.n_samples, noise_std = config.noise_std, seed = config.data_seed
    )
    train_data, test_data = split_dataset(
        dataset; train_fraction = config.train_fraction, seed = config.split_seed
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)

    priors = make_priors(
        d_h = d_h, d_f = d_f, seed = config.prior_seed,
        ct_precision_mean = config.ct_precision_mean,
    )
    sp_deps = NGMPDependencies(
        out = nothing, in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha, beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )

    elapsed = @elapsed result = infer(
        model = xor_ct_mvsoftplus(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = CTMeta(a -> reshape(a, d_h, d_f)),
            meta_pred = CTMeta(a -> reshape(a, d_h, d_h)),
            sp_deps = sp_deps,
            sp_damping = sp_damping,
        ),
        data = (y = train_data.OT, features = train_features),
        constraints = xor_ct_constraints(),
        initialization = make_initialization(priors, d_h),
        iterations = config.iterations,
        free_energy = true,
        options = (limit_stack_depth = 100,),
        showprogress = true,
        disable_inference_error_hint = true,
    )

    priors_for_prediction = prediction_priors(result)
    prediction_output_mean = mean(train_data.OT)
    prediction_kwargs = (
        batch_size = config.prediction_batch_size,
        config = config,
        d_h = d_h,
        d_f = d_f,
        output_mean = prediction_output_mean,
    )
    train_prediction = predictive_statistics(predict_marginals(
        priors_for_prediction,
        train_features;
        prediction_kwargs...,
    ))
    test_prediction = predictive_statistics(predict_marginals(
        priors_for_prediction,
        test_features;
        prediction_kwargs...,
    ))
    train_mse = mean(abs2, train_prediction.mean .- train_data.OT)
    test_mse = mean(abs2, test_prediction.mean .- test_data.OT)
    baseline = mean(abs2, mean(train_data.OT) .- test_data.OT)

    surface_path = nothing
    if config.save_outputs
        grid = make_prediction_grid(config.grid_size)
        grid_statistics = predictive_statistics(predict_marginals(
            priors_for_prediction,
            grid.features;
            prediction_kwargs...,
        ))
        grid_prediction = (
            mean = reshape(grid_statistics.mean, length(grid.y), length(grid.x)),
            variance = reshape(grid_statistics.variance, length(grid.y), length(grid.x)),
        )
        surface_path = save_predictive_surface(grid, grid_prediction, config.output_prefix)
    end

    fe = result.free_energy
    println()
    println("=== xor_ctransition_mvsoftplus (d_h = $d_h, iterations = $(config.iterations), " *
            "n_train = $(nrow(train_data)), n_test = $(nrow(test_data)), $(round(elapsed, digits = 1))s)")
    println("free energy first/last : ", first(fe), " / ", last(fe))
    println("free energy finite     : ", all(isfinite, fe),
            "   decreasing steps: ", count(<(0), diff(fe)), "/", length(fe) - 1)
    println("train MSE              : ", round(train_mse, digits = 4))
    println("test MSE               : ", round(test_mse, digits = 4))
    println("all-mean baseline MSE  : ", round(baseline, digits = 4))
    println("test q(y) variance     : ",
            round(minimum(test_prediction.variance), digits = 5), " / ",
            round(mean(test_prediction.variance), digits = 5), " / ",
            round(maximum(test_prediction.variance), digits = 5))
    isnothing(surface_path) || println("predictive surface     : ", surface_path)

    if config.diagnostics
        prior_a_map = mean(priors[:a_map])
        prior_a_pred = mean(priors[:a_pred])
        A_map = reshape(mean(last(result.posteriors[:a_map])), d_h, d_f)
        A_pred = reshape(mean(last(result.posteriors[:a_pred])), d_h, d_h)
        theta_hat = mean(last(result.posteriors[:theta]))
        preds = test_prediction.mean
        plugin_train_prediction = plugin_prediction_means(result, train_features, d_h, d_f)
        plugin_test_prediction = plugin_prediction_means(result, test_features, d_h, d_f)
        qγ = last(result.posteriors[:gamma_obs])
        s_means = [mean(q) for q in last(result.posteriors[:s])]
        h1_means = [mean(q) for q in last(result.posteriors[:h1])]
        println("--- diagnostics")
        println("|Δ a_map| / |prior|    : ",
                round(sqrt(sum(abs2, vec(A_map) .- prior_a_map)) / sqrt(sum(abs2, prior_a_map)), digits = 3))
        println("|Δ a_pred| / |prior|   : ",
                round(sqrt(sum(abs2, vec(A_pred) .- prior_a_pred)) / sqrt(sum(abs2, prior_a_pred)), digits = 3))
        println("|theta|                : ", round(sqrt(sum(abs2, theta_hat)), digits = 3))
        println("E[gamma_obs]           : ", round(mean(qγ), digits = 3))
        println("prediction range       : ", round(minimum(preds), digits = 3), " .. ", round(maximum(preds), digits = 3))
        println("plugin train/test MSE  : ",
                round(mean(abs2, plugin_train_prediction .- train_data.OT), digits = 4), " / ",
                round(mean(abs2, plugin_test_prediction .- test_data.OT), digits = 4))
        println("RxInfer/plugin RMSE    : ",
                round(sqrt(mean(abs2, preds .- plugin_test_prediction)), digits = 4))
        println("q(h1) mean range       : ", round(minimum(minimum.(h1_means)), digits = 3), " .. ",
                round(maximum(maximum.(h1_means)), digits = 3))
        println("q(s) mean range        : ", round(minimum(minimum.(s_means)), digits = 3), " .. ",
                round(maximum(maximum.(s_means)), digits = 3))
    end
    return result, (
        train_mse = train_mse,
        test_mse = test_mse,
        baseline = baseline,
        test_predictive_variance = (
            minimum = minimum(test_prediction.variance),
            mean = mean(test_prediction.variance),
            maximum = maximum(test_prediction.variance),
        ),
        surface_path = surface_path,
        free_energy = fe,
    )
end

result, metrics = run_experiment(CONFIG)
