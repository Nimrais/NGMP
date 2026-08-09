# Structured-VMP rules for MvNormalExpPrecision with a JOINT q(out, μ) cluster
# (s still factorized apart). This is the transition-node role of the fused
# likelihood: ν ~ N(μ, diag(exp.(s))⁻¹) as a latent-feature noise layer, where
# evidence must propagate through ν into μ without the mean-field cut.
#
# Under q(s), the tilted factor is Gaussian with per-dimension precision
# ρⱼ = E[e^{sⱼ}] = exp(mⱼ + Vⱼⱼ/2), so within the (out, μ) cluster the node is
# exactly an MvNormalMeanPrecision with precision diag(ρ):
#
#   toward out:  ∫ Ñ(ν | μ) m(μ) dμ = N(m̃_μ, Ṽ_μ + diag(ρ)⁻¹)
#   toward μ:    symmetric
#   q(ν, μ):     precision [Λ_ν + P  −P; −P  Λ_μ + P],  P = diag(ρ)
#
# The s-site then uses the JOINT second moment — the cross-covariance term is
# the whole point of the structured cluster:
#
#   Eⱼ = E[(νⱼ − μⱼ)²] = (m_νⱼ − m_μⱼ)² + V_ννⱼⱼ + V_μμⱼⱼ − 2·V_νμⱼⱼ.

import ExponentialFamily: MvNormalWeightedMeanPrecision
import BayesBase: weightedmean_precision

# toward out (ν): Gaussian with the μ-cavity integrated through the tilted factor
@rule MvNormalExpPrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    m̃, Ṽ = mean_cov(m_μ)
    return MvNormalMeanCovariance(m̃, Ṽ + Diagonal(inv.(ρ)))
end

# toward μ: symmetric, ν-cavity integrated
@rule MvNormalExpPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    m̃, Ṽ = mean_cov(m_out)
    return MvNormalMeanCovariance(m̃, Ṽ + Diagonal(inv.(ρ)))
end

@marginalrule MvNormalExpPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    P = Diagonal(ρ)
    ξν, Λν = weightedmean_precision(m_out)
    ξμ, Λμ = weightedmean_precision(m_μ)
    return MvNormalWeightedMeanPrecision(
        [ξν; ξμ],
        [(Λν + P) (-Matrix(P)); (-Matrix(P)) (Λμ + P)],
    )
end

# per-coordinate E[(νⱼ − μⱼ)²] under the joint cluster marginal
function _mnep_joint_square_residuals(q_out_μ)
    m, V = mean_cov(q_out_μ)
    d = div(length(m), 2)
    r = view(m, 1:d) .- view(m, (d + 1):2d)
    E = [abs2(r[j]) + V[j, j] + V[d + j, d + j] - 2 * V[j, d + j] for j in 1:d]
    return _mnep_floor_residuals(E)
end

@rule MvNormalExpPrecision(:s, NaturalGradientMessage) (
    q_out_μ::MultivariateNormalDistributionsFamily,
    q_s::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    E = _mnep_joint_square_residuals(q_out_μ)
    exact = Logpdf(MvExpGammaSiteMessage(fill(one(eltype(E)) / 2, length(E)), E ./ 2))
    site = project(resolve_projection(getprojection(vconstraint)), q_s, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@average_energy MvNormalExpPrecision (q_out_μ::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    E = _mnep_joint_square_residuals(q_out_μ)
    return length(ms) * log(2π) / 2 - sum(ms) / 2 + dot(ρ, E) / 2
end
