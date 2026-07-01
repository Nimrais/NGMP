# =============================================================================
# NGMP surrogate inference for the dynamic-exp ensemble (univariate ETTh1, h336)
# =============================================================================
# The dynamic-exp ensemble (Lukashchuk-2026, "Learn Experts, Infer Gates") routes
# probabilistic forecasts through, per forecaster i and observation j,
#
#     z[i,j] ~ softdot(features[j], w[i], τ[i])      (Gaussian regression on log-precision)
#     γ[i,j] = exp(z[i,j])                            (deterministic exp link)
#     y[j]   ~ N(predictions[i,j], γ[i,j]^{-1})       (precision-weighted ensemble)
#
# Its RxInfer inference is structured VMP, q(w)q(z,γ)q(τ), with a moment-matched
# projection through the Exp node. This script replaces the γ=exp(z) handling with
# the EXACT natural-gradient message passing (NGMP) surrogate of the TMLR paper:
# the non-conjugate exp link is replaced by conjugate surrogate leaves whose
# natural parameters are the Fisher-tangent projection of the exact BP log-message
# (computed in closed form by ClosedFormExpectations.jl — NOT a 2nd-order delta).
#
# Two regimes are compared against VMP on the SAME series:
#   (a) prediction-only : reuse VMP-trained w,τ, swap NGMP routing in at predict time.
#   (b) full surrogate  : also learn w,τ through the surrogate loop.
#
# Data + the VMP baseline are produced once by scratchpad/prep_etth1.jl (it needs the
# neural expert/VAE stack) and cached to a JLD2 that this lightweight script loads.
# =============================================================================



using JLD2, Distributions, LinearAlgebra, Printf, Statistics
using RxInfer
using ProbabilisticEnsembling
import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf, LogGamma
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters
import SpecialFunctions: trigamma, digamma
using ProgressMeter
using YAML


cfg = YAML.load_file("sessions/dynamic/vae/dynamic_ETTh1_336.yaml")  # model_type already "dynamic"
spec = ProbabilisticEnsembling._parse_spec(cfg)

y_val, y_test, predictions_val, predictions_test, features_val, features_test =
    ProbabilisticEnsembling.before_rxinfer(spec);

function project_loggamma_to_normal(μ, m, σ)      # NGMP delta: observed curvature at the point z=m
    # deterministic (delta) projection: evaluate the Williams product at Normal(m, 0),
    # i.e. z=m with zero variance. ∂m = ℓ'(m); Λ = -ℓ''(m) = β - ∂m (drops the e^{v/2}
    # Fisher factor of the expected form, avoiding NGMP's 0/0 for Λ = -∂σ/σ at σ=0). σ unused.
    ∂m, _ = mean(ClosedWilliamsProduct(), Logpdf(μ), Normal(m, 0.0))
    Λ = μ.β - ∂m
    ξ = ∂m + m * Λ
    return ξ, Λ
end

# -----------------------------------------------------------------------------
# Inner conjugate model: z[i,j] ~ softdot(features[j], w[i], τ[i]) plus a Gaussian
# pseudo-observation obsz[i,j] ~ N(z[i,j], Rz[i,j]) carrying the γ/y backward
# message. The softdot runs under the structured factorization q(w,z)q(τ): w and z
# stay jointly Gaussian (the low-rank rule keeps their correlation, so w-uncertainty
# propagates into z), only τ is factored out — the key difference from a plain
# mean-field q(w)q(z), which severs that correlation. w,τ are not pinned by a
# FixedMarginalFormConstraint: at prediction y is missing, so passing the trained
# posteriors as priors leaves them put while the q(w,z) joint still updates z.
# -----------------------------------------------------------------------------
@model function inner_exp_ensemble(n_forecasters, n_obs, features, w_priors, τ_priors, obsz, Rz)
    local w, z, τ
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            obsz[i, j] ~ NormalMeanVariance(z[i, j], Rz[i, j])
        end
    end
end

@constraints function inner_constraints()
    q(w, z, τ) = q(w, z)q(τ)
end

inner_init(w_priors, τ_priors) = @initialization begin
    q(w) = w_priors
    q(τ) = τ_priors
    q(z) = NormalMeanVariance(0.0, 1.0)
end

function form_loggamma_messages(predictions, ys; αmax = 1e6)
    nf, n = size(predictions, 1), length(ys)
    q_prediction = PointMass.(predictions)              # nf × n
    q_y = [PointMass(y) for y in ys]                    # n
    μ_likelihood_γ = [                                  # nf × n  Gamma(3/2, (y-pred)^2/2)
        @call_rule NormalMeanPrecision(:τ, Marginalisation) (q_out = q_y[j], q_μ = q_prediction[i, j])
        for i in 1:nf, j in 1:n
    ]
    μz = [                                              # nf × n  LogGamma(α=1/rate, β=3/2) toward z
        @call_rule Exp(:in, Marginalisation) (m_out = μ_likelihood_γ[i, j],)
        for i in 1:nf, j in 1:n
    ]
    # perfect forecast (residual→0) ⇒ rate→0 ⇒ α=1/rate=Inf; cap it to a big finite number
    capα(m) = LogGamma(min(m.α, αmax), m.β; check_args = false)
    return capα.(μz)
end

# NGMP training loop — mimics notebooks/poisson_surrogate.jl (project → damped
# step → one inline infer → repeat), with LogGamma leaves instead of Poisson.
# No zforward_gamma, no β, no clamps, no anti-windup: damping α + momentum β only.
function ngmp_train(features, predictions, y, w_priors0, τ_priors0;
                    outer = 2, inner = 2, α = 0.5, β = 0.2, update_priors = false, verbose = true)
    nf, no = size(predictions)
    μz = form_loggamma_messages(predictions, y)     # fixed LogGamma(α=1/rate, β=3/2) leaves, built once
    # first-iteration init at each leaf's mode z*=log(β·α)=log(β/rate): linearize the
    # exp link AT its operating point (Λ=β there), not at 0 where the tangent overshoots
    # to obsz≈β/rate. Only the warm-start; from iter 2 q(z) is the coupled infer marginal.
    m = [log(μz[i, j].β * μz[i, j].α) for i in 1:nf, j in 1:no]
    v = ones(nf, no)
    ξ = zeros(nf, no); Λ = zeros(nf, no)
    vξ = zeros(nf, no); vΛ = zeros(nf, no)
    wpri, τpri = w_priors0, τ_priors0               # priors used this outer
    local res
    @showprogress for t in 1:outer
        ηξ = similar(ξ); ηΛ = similar(Λ)
        for j in 1:no, i in 1:nf                    # project each leaf at the current q(z)
            ηξ[i, j], ηΛ[i, j] = project_loggamma_to_normal(μz[i, j], m[i, j], sqrt(v[i, j]))
        end
        @. vξ = β * vξ + α * (ηξ - ξ); @. ξ += vξ   # damped NG step — NO clamps
        @. vΛ = β * vΛ + α * (ηΛ - Λ); @. Λ += vΛ
        res = infer(model = inner_exp_ensemble(n_forecasters = nf, n_obs = no,
                        w_priors = wpri, τ_priors = τpri),
                    data = (features = features, obsz = ξ ./ Λ, Rz = 1 ./ Λ),
                    constraints = inner_constraints(),
                    initialization = inner_init(wpri, τpri),
                    iterations = inner, free_energy = false,
                    options = (limit_stack_depth = 200,))
        qz = res.posteriors[:z][end]; m = map(mean, qz); v = map(var, qz)
        if update_priors                            # carry posteriors forward as next outer's prior
            wpri = res.posteriors[:w][end]; τpri = res.posteriors[:τ][end]
        end
        verbose && (@printf("  outer %2d  meanΛ=%.3f  meanE[z]=%.3f\n", t, mean(Λ), mean(m)); flush(stdout))
    end
    return res.posteriors[:w][end], res.posteriors[:τ][end]
end

# -----------------------------------------------------------------------------
# Driver: build priors from the YAML, train BOTH prior-update modes on val.
# Subset / iteration counts are ENV-overridable for tuning inner vs outer.
# -----------------------------------------------------------------------------
nf    = size(predictions_val, 1)
nfeat = cfg["params"]["priors"]["w"]["n_features"]
w0 = [MvNormalMeanScalePrecision(zeros(nfeat), cfg["params"]["priors"]["w"]["scale"]) for _ in 1:nf]
τ0 = [GammaShapeRate(cfg["params"]["priors"]["τ"]["shape"], cfg["params"]["priors"]["τ"]["rate"]) for _ in 1:nf]

nsub  = parse(Int, get(ENV, "NGMP_NSUB",  string(length(y_val))))
outer = 2
inner = 2
sub   = 1:min(nsub, length(y_val))
fsub, psub, ysub = features_val[sub], predictions_val[:, sub], y_val[sub]
@printf("== training on %d/%d val obs, outer=%d inner=%d ==\n", length(sub), length(y_val), outer, inner)

println("-- fixed prior (update_priors = false) --")
wb_fix, τb_fix = ngmp_train(fsub, psub, ysub, w0, τ0; outer, inner, update_priors = false)
println("-- carry-forward prior (update_priors = true) --")
wb_upd, τb_upd = ngmp_train(fsub, psub, ysub, w0, τ0; outer, inner, update_priors = true)

println("\nmean(τ) fixed : ", round.(mean.(τb_fix), sigdigits = 3))
println("mean(τ) carry : ", round.(mean.(τb_upd), sigdigits = 3))
