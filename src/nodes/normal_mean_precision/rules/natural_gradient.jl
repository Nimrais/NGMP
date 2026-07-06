# Natural-gradient rules for the standard `NormalMeanPrecision` node with BOTH the
# location and the precision unknown — the textbook non-conjugate pair.
#
# With the observation on `out` (data, auto-factorized) and NO factorization
# constraint, the μ and τ edges form one cluster, so each outbound message sees the
# other edge's BP message (`m_μ` / `m_τ`) plus — injected by `NGMPDependencies` —
# the receiving edge's own marginal, the projection point.
#
# Neither exact BP message has a closed-form Williams product, so both rules use the
# QUADRATURE-EXACT tangent projection (`Quadrature`, generalized Gauss–Laguerre on the
# Gamma edge / Gauss–Hermite on the Gaussian edge): the second-order delta expansion
# (`project_to_gamma`/`project_to_normal`) is biased when the receiving marginal is
# wide — exactly the small-N regime — because it integrates the touching quadratic
# of ℓ far from the expansion point. The exact BP messages (src/expressions/):
#
#   toward τ: μ_{f→τ}(τ) = ∫ 𝒩(y|z,τ⁻¹) 𝒩(z|m̃,ṽ) dz = 𝒩(y | m̃, ṽ + τ⁻¹)
#             — a `NormalPrecisionMessage`, projected at q(τ) → damped Gamma site.
#   toward μ: μ_{f→x}(x) = ∫ 𝒩(y|x,τ⁻¹) Gamma(τ; ã, b̃) dτ ∝ (2b̃ + (x−y)²)^{−(ã+½)}
#             — a `StudentTMessage`, projected at q(μ) → damped Gaussian site.

@rule NormalMeanPrecision(:τ, NaturalGradientMessage) (m_μ::UnivariateNormalDistributionsFamily, q_out::PointMass, q_τ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m̃, ṽ = mean_var(m_μ)
    exact = Logpdf(NormalPrecisionMessage(mean(q_out), m̃, ṽ))
    site = project(TangentProjection(type = Quadrature(128)), q_τ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule NormalMeanPrecision(:μ, NaturalGradientMessage) (m_τ::GammaDistributionsFamily, q_out::PointMass, q_μ::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    exact = Logpdf(StudentTMessage(mean(q_out), shape(m_τ), rate(m_τ)))
    site = project(TangentProjection(type = Quadrature(128)), q_μ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
