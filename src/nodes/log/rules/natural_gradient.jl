# Natural-gradient rules for ProbabilisticEnsembling's `Log` node (z = log(γ),
# i.e. γ = exp(z)) — the non-conjugate link of the dynamic ensemble model.
#
# Under the factorization q(w)q(z,γ)q(τ)q(β) the Log node's two edges form one
# cluster, so each outbound message sees the OTHER edge's BP message (`m_in` /
# `m_out`) plus — injected by `NGMPDependencies` — the receiving edge's own
# marginal (`q_out` / `q_in`), the projection point of the tangent projection.
# `meta` is the per-edge damping state created at activation.
#
# Exact BP log-messages through the link (change-of-variables convention of the
# existing `Marginalisation` rules in PE's src/log.jl):
#   toward z: a Gamma(a, b) message on γ pulls back to LogGamma(α = scale, β = a),
#             ℓ(z) = a·z − e^z/α + const — projected at q(z) = N(m, v) via the
#             expected Williams product (Λ = e^{m+v/2}/α, ξ = a + (m−1)Λ).
#   toward γ: a N(m, v) message on z pushes forward to LogNormal(m, √v) —
#             projected at q(γ) = Gamma(a, b) via the Gamma inverse-Fisher map,
#             giving the site γ^Δa e^{−Δb γ} = GammaShapeRate(Δa+1, Δb).

import ProbabilisticEnsembling: Log

@rule Log(:out, NaturalGradientMessage) (m_in::GammaDistributionsFamily, q_out::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    exact = Logpdf(LogGamma(scale(m_in), shape(m_in); check_args = false))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Log(:in, NaturalGradientMessage) (m_out::UnivariateGaussianDistributionsFamily, q_in::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m, v = mean_var(m_out)
    exact = Logpdf(LogNormal(m, sqrt(v)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
