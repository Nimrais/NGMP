#!/usr/bin/env julia

# Model library for the heteroscedastic hierarchy: shared graph, fitting arms,
# priors, prediction, and the aleatoric benchmark data. Used by
# streaming_hetero.jl; has no study entry point of its own.
#
# Model (identical graph and factorization for both arms):
#
#     v ~ N(0, σ_v² I)                       # mean weights
#     w ~ N(anchored mean, diag)             # log-precision weights
#     f[o] := dot(φ(x_o), v)                 # deterministic mean pathway (exact BP)
#     s[o] ~ softdot(ψ(x_o), w, c_top)       # level-2 log-precision
#     y[o] ~ MvNormalExpPrecision(f[o], s[o])   #  y ~ N(f, e^{-s})
#
# The arms differ ONLY in how the non-conjugate message toward each s[o] is
# executed: the VMP arm multiplies the exact mean-field site exp(E_q[log f])
# into the marginal and projects the product onto a Gaussian (ProjectedTo +
# ClosedFormStrategy); the NGMP arms send natural-gradient tangent projections
# (hybrid site or true cavity messages, NGMPDependencies + DampingMeta).

include(joinpath(@__DIR__, "common.jl"))
using .WhenNGMPHelpsCommon

using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using LinearAlgebra
using Random
using RxInfer
using StableRNGs
using Statistics
using SurrogateModelling

import ExponentialFamilyProjection: BoundedNormUpdateRule

# ---------------------------------------------------------------------------
# models: same graph, arms differ only by the `where` clause on the likelihood
# ---------------------------------------------------------------------------

@model function hetero_vmp_model(y, mean_features, level_features, v_prior, w_prior, top_carrier)
    local f, s
    v ~ v_prior
    w ~ w_prior
    for o in eachindex(y)
        f[o] := dot(mean_features[o], v)
        s[o] ~ softdot(level_features[o], w, top_carrier)
        y[o] ~ MvNormalExpPrecision(f[o], s[o])
    end
end

@model function hetero_ngmp_model(y, mean_features, level_features, v_prior, w_prior, top_carrier, dependencies, damping)
    local f, s
    v ~ v_prior
    w ~ w_prior
    for o in eachindex(y)
        f[o] := dot(mean_features[o], v)
        s[o] ~ softdot(level_features[o], w, top_carrier)
        y[o] ~ MvNormalExpPrecision(f[o], s[o]) where {
            dependencies = dependencies,
            meta = damping,
        }
    end
end

# Norm bound 100: the default projection budget can trap marginals far from
# their optimum (see poisson_state_space.jl) — widen it so the VMP arm is a
# fair baseline.
@constraints function hetero_vmp_constraints()
    q(f, w, s) = q(f)q(w)q(s)
    q(s) :: ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            direction = BoundedNormUpdateRule(100.0),
        ),
    )
end

# NGMP's projected site toward s is Gaussian, so it composes with the exact
# conjugate (w, s) cluster — the notebook's structured configuration. The VMP
# arm cannot join this cluster: its raw ExpGamma site is projected at the
# MARGINAL, which forces the mean-field split between w and s.
@constraints function hetero_ngmp_constraints()
    q(f, w, s) = q(f)q(w, s)
end

# ---------------------------------------------------------------------------
# features, priors, initialization (why_hierarchy_deep_kernel.jl choices)
# ---------------------------------------------------------------------------

function make_feature_map(n_basis, lengthscale; feature_seed)
    rng = StableRNG(feature_seed)
    frequencies = randn(rng, n_basis) ./ lengthscale
    phases = 2pi .* rand(rng, n_basis)
    scale = sqrt(2 / n_basis)
    return x -> vcat(scale .* cos.(frequencies .* x .+ phases), [1.0])
end

design(feature_map, xs) = [feature_map(x) for x in xs]

# Rice/Gasser first-difference noise estimate: a basis-free anchor for the
# constant coordinate of the log-precision weights.
noise_anchor(xs, ys) = -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

gaussian(means, variances) =
    MvNormalMeanCovariance(collect(means), Matrix(Diagonal(collect(variances))))

function make_priors(config, xs, ys)
    n_basis = config.n_basis
    anchor = noise_anchor(xs, ys)
    return (;
        v = gaussian(
            zeros(n_basis + 1),
            fill(abs2(config.signal_sd), n_basis + 1),
        ),
        w = gaussian(
            vcat(zeros(n_basis), [anchor]),
            vcat(fill(abs2(config.level_sd), n_basis), [abs2(config.anchor_sd)]),
        ),
    )
end

# Push the priors forward so the first damped step is well scaled.
function initial_marginals(priors, rows, top_carrier)
    Φ = reduce(hcat, rows)'
    mv, Vv = mean_cov(priors.v)
    mw, Vw = mean_cov(priors.w)
    f_means = Φ * mv
    f_variances = vec(sum((Φ * Vv) .* Φ; dims = 2))
    s_means = Φ * mw
    s_variances = vec(sum((Φ * Vw) .* Φ; dims = 2)) .+ inv(top_carrier)
    return (;
        f = NormalMeanVariance.(f_means, f_variances),
        s = NormalMeanVariance.(s_means, s_variances),
    )
end

# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

function fit_arm(method, observations, rows, priors, config)
    states = initial_marginals(priors, rows, config.top_carrier)
    initialization = @initialization begin
        q(v) = deepcopy(priors.v)
        q(w) = deepcopy(priors.w)
        q(f) = states.f
        q(s) = states.s
    end
    if method == "NGMP-cavity"
        # True NGMP: no factorization constraints (the fused node keeps its
        # (μ, s) cluster, the softdot keeps (w, s)), both latent interfaces
        # named so the cavity rules receive messages; message inits break the
        # message-marginal cycle.
        dependencies = NGMPDependencies(
            s = nothing,
            μ = nothing,
            projection = TangentProjection(type = Quadrature(config.cavity_quadrature)),
        )
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
            method = :damped,
        )
        cavity_initialization = @initialization begin
            q(v) = deepcopy(priors.v)
            q(w) = deepcopy(priors.w)
            q(f) = states.f
            q(s) = states.s
            μ(f) = states.f
            μ(s) = states.s
        end
        elapsed = @elapsed result = infer(
            model = hetero_ngmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
                dependencies = dependencies,
                damping = damping,
            ),
            data = (y = observations,),
            initialization = cavity_initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
        return (;
            qv = result.posteriors[:v],
            qw = result.posteriors[:w],
            free_energy = Float64[],
            elapsed,
        )
    elseif method == "NGMP"
        dependencies = NGMPDependencies(
            s = nothing,
            projection = TangentProjection(type = ClosedForm),
        )
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
            method = :damped,
        )
        elapsed = @elapsed result = infer(
            model = hetero_ngmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
                dependencies = dependencies,
                damping = damping,
            ),
            data = (y = observations,),
            constraints = hetero_ngmp_constraints(),
            initialization = initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = true,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
    else
        elapsed = @elapsed result = infer(
            model = hetero_vmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
            ),
            data = (y = observations,),
            constraints = hetero_vmp_constraints(),
            initialization = initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = true,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
    end
    return (;
        qv = result.posteriors[:v],
        qw = result.posteriors[:w],
        free_energy = collect(Float64.(result.free_energy)),
        elapsed,
    )
end

# ---------------------------------------------------------------------------
# closed-form prediction: everything follows from q(v), q(w)
# ---------------------------------------------------------------------------

function predict(fit, rows, top_carrier)
    Φ = reduce(hcat, rows)'
    mv, Vv = mean_cov(fit.qv)
    mw, Vw = mean_cov(fit.qw)
    f_mean = Φ * mv
    f_variance = vec(sum((Φ * Vv) .* Φ; dims = 2))
    s_mean = Φ * mw
    s_variance = vec(sum((Φ * Vw) .* Φ; dims = 2)) .+ inv(top_carrier)
    # lognormal moments of the noise variance e^{-s}
    noise_mean = exp.(-s_mean .+ s_variance ./ 2)
    noise_lower = exp.(-(s_mean .+ 1.96 .* sqrt.(s_variance)))
    noise_upper = exp.(-(s_mean .- 1.96 .* sqrt.(s_variance)))
    return (;
        mean = f_mean,
        latent_variance = f_variance,
        total_variance = f_variance .+ noise_mean,
        noise_mean,
        noise_lower,
        noise_upper,
        s_mean,
        s_variance,
    )
end

# p(y*) = ∫ N(y*; f_mean, f_var + e^{-s}) N(s; m, v) ds, by quadrature over s.
function predictive_logpdf(f_mean, f_variance, s_mean, s_variance, y)
    sd = sqrt(s_variance)
    points = range(s_mean - 8sd, s_mean + 8sd; length = 201)
    weights = pdf.(Normal(s_mean, sd), points)
    weights ./= sum(weights)
    density = sum(
        weights .* pdf.(Normal.(f_mean, sqrt.(f_variance .+ exp.(-points))), y),
    )
    return log(max(density, floatmin(Float64)))
end

mean_predictive_logpdf(prediction, ys) = mean(
    predictive_logpdf(
        prediction.mean[i],
        prediction.latent_variance[i],
        prediction.s_mean[i],
        prediction.s_variance[i],
        ys[i],
    ) for i in eachindex(ys)
)

# ---------------------------------------------------------------------------
# aleatoric benchmark (why_hierarchy_deep_kernel.jl):
#     y = -(x+1/2)sin(3πx) + ε,  ε ~ N(0, [0.45(x+1/2)]²),  x ~ N(0,1)
# The noise level s(x_o) is a per-observation latent: its uncertainty cannot
# concentrate no matter how long the dataset grows.
# ---------------------------------------------------------------------------

aleatoric_mean(x) = -(x + 0.5) * sin(3pi * x)
aleatoric_noise_variance(x) = abs2(0.45 * (x + 0.5))

function aleatoric_data(config, rng)
    n = config.aleatoric_samples
    x = randn(rng, n)
    y = aleatoric_mean.(x) .+ sqrt.(aleatoric_noise_variance.(x)) .* randn(rng, n)
    order = randperm(rng, n)
    n_test = round(Int, config.holdout_fraction * n)
    test, train = order[1:n_test], order[(n_test + 1):end]
    return (;
        x_train = x[train], y_train = y[train],
        x_test = x[test], y_test = y[test],
    )
end
