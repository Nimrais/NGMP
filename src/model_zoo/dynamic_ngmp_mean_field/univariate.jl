# ============================================================================
# DynamicExpNGMP — univariate inner surrogate model + the exp-link projections.
#
# The NGMP surrogate replaces the deterministic γ=exp(z) link with a Gaussian
# pseudo-observation on z. The INNER graph that the outer loop solves repeatedly
# is therefore conjugate: a softdot regression with one Gaussian pseudo-obs per
# (i,j). Structured factorization q(w,z)q(τ) keeps the w–z correlation (plain
# `SoftDot`, NOT LowRankMeta — the low-rank meta only ships mean-field rules).
#
# The pseudo-observations (obsz, Rz) are NOT data in the YAML sense — they are
# recomputed every outer sweep by the driver in pipeline.jl from the y→z message
# projected at the current q(z). They enter the model as datavars.
# ============================================================================

using RxInfer
import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf, LogGamma
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters
import SpecialFunctions: trigamma

function project_loggamma_to_normal(α, β, m, σ)
    ∂μ, ∂σ = mean(ClosedWilliamsProduct(), Logpdf(LogGamma(α, β; check_args = false)), Normal(m, σ))
    Λ = -∂σ / σ
    ξ = ∂μ + m * Λ
    return ξ, Λ
end

# ---------------------------------------------------------------------------
# Inner conjugate model: softdot + Gaussian pseudo-obs on z.
# ---------------------------------------------------------------------------
@model function univariate_dynamic_exp_ngmp(n_forecasters, n_obs, features, priors, obsz, Rz)
    local w, z, τ
    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
    end
    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            obsz[i, j] ~ NormalMeanVariance(z[i, j], Rz[i, j])
        end
    end
end

@constraints function univariate_dynamic_exp_ngmp_constraints(priors, prediction)
    q(w, z, τ) = q(w, z)q(τ)
    # prediction: pin w, τ to the trained posteriors (as in dynamic_exp).
    # TODO: with q(w,z) structured, FixedMarginalFormConstraint on w needs care;
    # the driver currently passes trained posteriors as priors and runs 1 sweep.
end

@initialization function univariate_dynamic_exp_ngmp_init(priors)
    q(w) = deepcopy(priors[:w])
    q(τ) = priors[:τ]
    q(z) = NormalMeanVariance(0.0, 1.0)
end

_shape_rate_ef(q) = (getnaturalparameters(q)[1] + 1, -getnaturalparameters(q)[2])

# z→γ : project the LogNormal(m,√v) forward message onto a Gamma belief (exact
# ∇_η E = Cov_q[T,ℓ] + Gamma inverse-Fisher). Returns natural-param increments
# (Δa, Δb). Used for E[γ] / the β model; not strictly needed for dynamic_exp.
function project_lognormal_to_gamma(m, v, q_ef)
    c1, c2 = mean(ClosedWilliamsProduct(), Logpdf(LogNormal(m, sqrt(v))), q_ef)
    a, b = _shape_rate_ef(q_ef)
    f11, f12, f22 = trigamma(a), 1 / b, a / b^2
    detF = f11 * f22 - f12^2
    return (f22 * c1 - f12 * c2) / detF, -(f11 * c2 - f12 * c1) / detF
end
