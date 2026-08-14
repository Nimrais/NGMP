# The dynamic ensemble's log-precision head, ONE arm: raw linear features, two levels,
# fit with **native natural-gradient message passing**.
#
#   z[i,j] ~ softdot(features[j], w[i], tau[i])      per-expert log-precision
#   z[i,j] ~ Log(gamma[i,j])                          non-conjugate link, NGMP rules
#   y[j]   ~ NormalMeanPrecision(predictions[i,j], gamma[i,j])
#
# The mean is GIVEN -- `predictions` are pretrained expert forecasts -- so the only thing
# learned is how reliable each expert is at each timestep. `q(w)` is dense (66x66,
# MomentForm), which is what lets the head be uncertain about reliability at all.
#
# Why NGMP and not form constraints
# ---------------------------------
# `NGMPDependencies(out = nothing, in = nothing)` makes each of the `Log` node's two
# outbound messages subscribe to the receiving edge's own marginal and dispatches the
# `NaturalGradientMessage` rules, so every message stays in-family -- Gaussian toward `z`,
# Gamma toward `gamma` -- and **no `ProjectedTo` form constraints are needed**. That is the
# point of this project: the missing exact-BP message for a non-conjugate factor IS a
# natural-gradient projection, supplied natively rather than by projecting marginals after
# the fact. It is also far cheaper: projection ran a per-marginal optimisation for each of
# the 7 x n_obs `z` and `gamma` edges on every sweep.
#
# Stability comes from damping (`DampingMeta`, alpha in eta-space) and from the anchored
# prior below -- not from guards or clipping.
#
# The 66th feature is a CONSTANT coordinate. It is the only coordinate whose value is
# identically 1, hence the only place an anchor on the log-precision can live; without it
# the "anchored" prior was only a shrunk prior.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/dynamic_deep_kernel_precision.jl
#
# Env: DDK_N_OBS (0 = full validation split), DDK_ITERATIONS, DDK_ALPHA, DDK_RESULTS

using JLD2
using LinearAlgebra
using Printf
using Serialization
using Statistics

using Distributions
using RxInfer
using SurrogateModelling

import ProbabilisticEnsembling: Log, LowRankMeta

ddk_integer(key, default) = parse(Int, get(ENV, key, string(default)))
ddk_float(key, default) = parse(Float64, get(ENV, key, string(default)))

const DDK_CONFIG = (
    n_obs = ddk_integer("DDK_N_OBS", 0),
    iterations = ddk_integer("DDK_ITERATIONS", 20),
    # beta = 0 deliberately: Gamma momentum can leave the natural domain.
    alpha = ddk_float("DDK_ALPHA", 0.2),
    kappa = ddk_float("DDK_KAPPA", 1.0),
    # Anchored prior on the constant coordinate; diffuse on the rest.
    signal_sd = ddk_float("DDK_SIGNAL_SD", 2.0),
    anchor_variance = ddk_float("DDK_ANCHOR_VARIANCE", 2.0),
    # Spec priors from sessions/dynamic/vae/dynamic_ETTh1_96.yaml.
    tau_shape = ddk_float("DDK_TAU_SHAPE", 1.0),
    tau_rate = ddk_float("DDK_TAU_RATE", 1e-3),
    beta_shape = ddk_float("DDK_BETA_SHAPE", 1.0),
    beta_rate = ddk_float("DDK_BETA_RATE", 1e3),
    results_path = get(ENV, "DDK_RESULTS", "/tmp/dynamic_deep_kernel_precision.jls"),
)

# ---------------------------------------------------------------------------
# Features and priors
# ---------------------------------------------------------------------------

"""
    raw_standardizer(train_features)

Frozen standardizer for the raw features with a **constant coordinate appended** last, so
`anchor_index = length(phi)`.
"""
function raw_standardizer(train_features)
    matrix = reduce(hcat, train_features)'
    centre = vec(mean(matrix; dims = 1))
    spread = max.(vec(std(matrix; dims = 1)), 1e-9)
    return raw -> vcat((raw .- centre) ./ spread, [1.0])
end

"""
    log_precision_anchors(predictions, targets)

Per-expert `log(1 / residual variance)`: the prior mean of each expert's constant
coordinate, so the head learns deviations from a sane reliability level instead of
travelling several units from zero.
"""
function log_precision_anchors(predictions, targets)
    return map(axes(predictions, 1)) do expert
        residual = mean(abs2.(view(predictions, expert, :) .- targets))
        -log(max(residual, 1e-8))
    end
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@model function dynamic_ngmp(
    n_forecasters, n_obs, y, features, predictions, priors, deps, damping,
)
    local w, z, γ, τ, β
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where { meta = LowRankMeta() }
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            # The non-conjugate link, native: Gaussian message toward z, Gamma toward
            # gamma, both damped in natural-parameter space.
            z[i, j] ~ Log(γ[i, j]) where { dependencies = deps, meta = damping }
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_ngmp_constraints()
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
    # The softdot rules read mean/cov of q(w), so converting the information-form marginal
    # ONCE per update instead of a 66x66 solve in every rule call is a ~7x win.
    q(w)::MomentForm()
end

@initialization function dynamic_init(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(β) = priors[:β]
end

# ---------------------------------------------------------------------------
# Fit and predict
# ---------------------------------------------------------------------------

"""
    fit_arm(train_features, targets, predictions, config)

Fit the linear two-level arm. `train_features` is the RAW feature vectors, already sliced
to the observations being used -- passed in rather than read from a module global so this
file can be `include`d by a driver that sweeps several (dataset, horizon) cells.
"""
function fit_arm(train_features, targets, predictions, config)
    n_forecasters = size(predictions, 1)
    feature_map = raw_standardizer(train_features)
    design = [feature_map(f) for f in train_features]
    dimension = length(first(design))
    n_obs = length(design)
    anchors = log_precision_anchors(predictions, targets)

    # `LowRankMeta` folds its rank-1 messages into the prior in place via `BLAS.syr!`, and
    # its `prod` rules are defined for `MvNormalWeightedMeanPrecision` but NOT for
    # `MvNormalMeanCovariance` -- so the priors are built in information form, with a dense
    # `Matrix` precision so the in-place path is taken.
    information_form(means, variances) = begin
        precision = Matrix(Diagonal(inv.(variances)))
        MvNormalWeightedMeanPrecision(precision * means, precision)
    end
    w_priors = map(1:n_forecasters) do expert
        information_form(
            vcat(zeros(dimension - 1), [anchors[expert]]),
            vcat(fill(abs2(config.signal_sd), dimension - 1), [config.anchor_variance]),
        )
    end
    priors = Dict{Symbol, Any}(
        :w => w_priors,
        :τ => [GammaShapeRate(config.tau_shape, config.tau_rate) for _ in 1:n_forecasters],
        :β => [GammaShapeRate(config.beta_shape, config.beta_rate) for _ in 1:n_forecasters],
    )

    @printf("  %d features (65 standardized + 1 constant), %d experts, %d observations\n",
        dimension, n_forecasters, n_obs)
    @printf("  NGMP: alpha = %.2f, beta = 0, %d iterations\n",
        config.alpha, config.iterations)

    deps = NGMPDependencies(out = nothing, in = nothing)
    damping = DampingMeta(alpha = config.alpha, beta = 0.0)

    started = time()
    result = infer(
        model = dynamic_ngmp(
            n_forecasters = n_forecasters, n_obs = n_obs, priors = priors,
            deps = deps, damping = damping,
        ),
        data = (y = targets, features = design, predictions = predictions),
        constraints = dynamic_ngmp_constraints(),
        initialization = dynamic_init(priors),
        returnvars = (w = KeepLast(), τ = KeepLast(), β = KeepLast()),
        iterations = config.iterations,
        free_energy = false, showprogress = false,
        options = (limit_stack_depth = 500,),
    )
    elapsed = time() - started
    # One damping state per Log edge: out + in, per forecaster per observation. A mismatch
    # means the NGMP rules were not the ones that fired.
    expected_states = 2 * n_forecasters * n_obs
    length(deps.states) == expected_states || error(
        "NGMP rules did not dispatch: $(length(deps.states)) damping states, " *
        "expected $expected_states",
    )

    q_w = collect(vec(result.posteriors[:w]))
    q_τ = collect(vec(result.posteriors[:τ]))
    q_β = collect(vec(result.posteriors[:β]))
    # Unspanned directions per expert: the quantity that decides whether the model can ever
    # say "I do not know how reliable this expert is here".
    retained = map(enumerate(q_w)) do (expert, posterior)
        scaling = sqrt(inv(Diagonal(diag(cov(w_priors[expert])))))
        count(>(0.5), eigvals(Symmetric(Matrix(scaling * cov(posterior) * scaling))))
    end
    @printf("  fit in %.1f s; %d NGMP damping states; directions keeping >50%% of prior variance: %s of %d\n",
        elapsed, length(deps.states), join(retained, ","), dimension)
    @printf("  learned E[tau] per expert (prior mean %.3g): %s\n",
        config.tau_shape / config.tau_rate,
        join(map(v -> @sprintf("%.3g", mean(v)), q_τ), " "))
    @printf("  learned E[beta] per expert (prior mean %.3g): %s\n",
        config.beta_shape / config.beta_rate,
        join(map(v -> @sprintf("%.3g", mean(v)), q_β), " "))

    return (; q_w, q_τ, q_β, feature_map, elapsed, retained, dimension, n_obs)
end

"""
    predict_arm(arm, features, predictions, config)

The notebook's own predictive rule, `V = E[1/gamma] + kappa * E[beta]`, precision-weighted
across experts. Closed form: with PointMass features and mean-field `q(w)`,
`m_z = phi' m_w` and `v_z = phi' V_w phi + 1/E[tau]`, so no inner `infer_qz` pass is
needed.
"""
function predict_arm(arm, features, predictions, config)
    n_forecasters, n_points = size(predictions, 1), length(features)
    design = reduce(hcat, (arm.feature_map(f) for f in features))'
    precision = Matrix{Float64}(undef, n_forecasters, n_points)
    for expert in 1:n_forecasters
        m_w, V_w = mean_cov(arm.q_w[expert])
        score_mean = design * m_w
        score_variance =
            vec(sum((design * V_w) .* design; dims = 2)) .+ inv(mean(arm.q_τ[expert]))
        expected_inverse = exp.(.-score_mean .+ score_variance ./ 2)
        variance = expected_inverse .+ config.kappa * mean(arm.q_β[expert])
        precision[expert, :] = clamp.(1 ./ variance, 1e-6, 1e6)
    end
    means = Vector{Float64}(undef, n_points)
    sigmas = Vector{Float64}(undef, n_points)
    for point in 1:n_points
        total = sum(view(precision, :, point))
        means[point] =
            sum(precision[i, point] * predictions[i, point] for i in 1:n_forecasters) / total
        sigmas[point] = sqrt(inv(total))
    end
    return means, sigmas
end

"""The notebook's `predictive_metrics`, verbatim in behaviour."""
function predictive_metrics(means, sigmas, targets; quantiles = (0.1, 0.9))
    terms = [logpdf(Normal(means[j], sigmas[j]), targets[j]) for j in eachindex(targets)]
    z95 = 1.959963984540054
    coverage = mean(
        (targets .>= means .- z95 .* sigmas) .& (targets .<= means .+ z95 .* sigmas),
    )
    pinball = mean(map(quantiles) do q
        zq = quantile(Normal(), q)
        estimate = means .+ zq .* sigmas
        mean(max.(q .* (targets .- estimate), (q - 1) .* (targets .- estimate)))
    end)
    return (; mae = mean(abs.(means .- targets)),
        rmse = sqrt(mean(abs2.(means .- targets))),
        ll = mean(terms), ll_std = std(terms), cov95 = coverage, pinball)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Guarded so this file can be `include`d for its model and `fit_arm` without running the
# ETTh1 h96 fit as a side effect.
if abspath(PROGRAM_FILE) == @__FILE__

cache = JLD2.load(joinpath(
    @__DIR__, "..", "notebooks", "vmp_vs_ngmp", "dynamic_etth1_h96_cache.jld2",
))
n_obs = DDK_CONFIG.n_obs > 0 ? DDK_CONFIG.n_obs : length(cache["y_val"])
train_features = cache["features_val"][1:n_obs]
targets = cache["y_val"][1:n_obs]
predictions = cache["predictions_val"][:, 1:n_obs]

println("\n", "="^90)
println("Dynamic ensemble log-precision head -- linear features, 2 levels, native NGMP")
println("="^90)

arm = fit_arm(train_features, targets, predictions, DDK_CONFIG)
means, sigmas = predict_arm(
    arm, cache["features_test"], cache["predictions_test"], DDK_CONFIG,
)
metrics = predictive_metrics(means, sigmas, cache["y_test"])

println("\n", "-"^90)
@printf("%9s %9s %8s %8s %9s   (%d test points)\n",
    "logpdf", "cov95", "MAE", "RMSE", "pinball", length(cache["y_test"]))
@printf("%9.4f %9.3f %8.4f %8.4f %9.4f\n",
    metrics.ll, metrics.cov95, metrics.mae, metrics.rmse, metrics.pinball)

serialize(DDK_CONFIG.results_path,
    (; config = DDK_CONFIG, metrics, means, sigmas, arm.retained, arm.dimension,
       arm.elapsed))
@info "saved" DDK_CONFIG.results_path

end  # driver guard
