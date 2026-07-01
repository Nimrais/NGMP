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
using YAML


cfg = YAML.load_file("sessions/dynamic/vae/dynamic_ETTh1_336.yaml")  # model_type already "dynamic"
spec = ProbabilisticEnsembling._parse_spec(cfg)

y_val, y_test, predictions_val, predictions_test, features_val, features_test =
    ProbabilisticEnsembling.before_rxinfer(spec);

function project_loggamma_to_normal(α, β, m, σ)
    ∂μ, ∂σ = mean(ClosedWilliamsProduct(), Logpdf(LogGamma(α, β; check_args = false)), Normal(m, σ))
    Λ = -∂σ / σ
    ξ = ∂μ + m * Λ
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
            z[i, j] ~ softdot(features[j], w[i], τ[i])
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

function form_loggamma_messages(predictions, ys)
    nf, n = size(predictions_val)[1], length(y_val)
    q_prediction_val = [PointMass(prediction) for prediction in predictions];
    q_y_val = [PointMass(y) for y in ys];
    μ_likelihood_γ = [
        @call_rule NormalMeanPrecision(:τ, Marginalisation) (q_out = q_y_val[i], q_μ = q_prediction_val[i, j])
        for i in 1:nf, j in 1:n
    ];
    return [@call_rule Exp(:in, Marginalisation) (m_out = μ_γ,) for μ_γ in μ_likelihood_γ]
end

# function ngmp_train(features, predictions, y, w_priors0, τ_priors0;
#                     outer = 40, α_d = 0.3, β_m = 0.7, verbose = true)
#     nf = length(w_priors0)
#     no = length(features)
#     gy_shape = 1.5

#     q_predictions = [for prediction in predictions]
    
#     vξ = zeros(nf, no); vΛ = zeros(nf, no)
#     local res
#     for t in 1:outer
#         res, mz, vz = infer_qz(features, w_priors0, τ_priors0, obsz, Rz; iters = inner)
#         ηξ = similar(ξ); ηΛ = similar(Λ)
#         for j in 1:no, i in 1:nf
#             mc = clamp(mz[i, j], -30.0, 30.0)
#             vc = clamp(vz[i, j], 1e-6, 8.0)
#             ξt, Λt = project_loggamma_to_normal(α_lg[i, j], gy_shape, mc, sqrt(vc))
#             ηξ[i, j] = ξt; ηΛ[i, j] = clamp(Λt, 1e-6, 1e3)
#         end
#         @. vξ = β_m * vξ + α_d * (ηξ - ξ); @. ξ += vξ
#         @. vΛ = β_m * vΛ + α_d * (ηΛ - Λ); @. Λ += vΛ
#         # Anti-windup: z = log-precision is physically bounded, so clamp the z-target
#         # and RE-SYNC the momentum state (ξ,Λ) to the clamped pseudo-obs. Without this
#         # a few near-perfect-forecast points (rate→0) drive Λ to the floor and obsz→∞,
#         # corrupting the w fit; the bulk already converges (mean Λ→3/2).
#         Λc = clamp.(Λ, 1e-3, 1e3)
#         nobsz = clamp.(ξ ./ Λc, -25.0, 25.0)
#         Δ = maximum(abs.(obsz .- nobsz))
#         ξ .= nobsz .* Λc; Λ .= Λc                     # re-sync state to clamped values
#         obsz .= nobsz; Rz .= 1 ./ Λc
#         verbose && (@printf("  outer %2d  ‖Δz-target‖∞=%.4e  mean Λ=%.3f\n", t, Δ, mean(Λc)); flush(stdout))
#     end
#     return res.posteriors[:w][end], res.posteriors[:τ][end]
# end


