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
# Full run:  julia --project=. experiments/xor_ctransition_mvsoftplus.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DataFrames
using Distributions
using ExponentialFamily
using LinearAlgebra: Diagonal, dot
using Random
using RxInfer
using StableRNGs
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
    data_seed = env_int("DATA_SEED", 2_026),
    split_seed = env_int("SPLIT_SEED", 2_027),
    prior_seed = env_int("PRIOR_SEED", 42),
    diagnostics = env_bool("DIAGNOSTICS"),
)

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

    # Plug-in posterior-mean forward pass.
    A_map = reshape(mean(last(result.posteriors[:a_map])), d_h, d_f)
    A_pred = reshape(mean(last(result.posteriors[:a_pred])), d_h, d_h)
    theta_hat = mean(last(result.posteriors[:theta]))
    predict(x) = dot(theta_hat, A_pred * _softplus.(A_map * x))

    train_mse = mean(abs2, predict.(train_features) .- train_data.OT)
    test_mse = mean(abs2, predict.(test_features) .- test_data.OT)
    baseline = mean(abs2, mean(train_data.OT) .- test_data.OT)

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

    if config.diagnostics
        prior_a_map = mean(priors[:a_map])
        prior_a_pred = mean(priors[:a_pred])
        preds = predict.(test_features)
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
        println("q(h1) mean range       : ", round(minimum(minimum.(h1_means)), digits = 3), " .. ",
                round(maximum(maximum.(h1_means)), digits = 3))
        println("q(s) mean range        : ", round(minimum(minimum.(s_means)), digits = 3), " .. ",
                round(maximum(maximum.(s_means)), digits = 3))
    end
    return result, (train_mse = train_mse, test_mse = test_mse, baseline = baseline, free_energy = fe)
end

result, metrics = run_experiment(CONFIG)
