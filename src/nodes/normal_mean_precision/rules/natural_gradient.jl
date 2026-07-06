# Natural-gradient rules for the standard `NormalMeanPrecision` node with BOTH the
# location and the precision unknown — the textbook non-conjugate pair.
#
# With the observation on `out` (data, auto-factorized) and NO factorization
# constraint, the μ and τ edges form one cluster, so each outbound message sees the
# other edge's BP message (`m_μ` / `m_τ`) plus — injected by `NGMPDependencies` —
# the receiving edge's own marginal, the projection point.
#
# Neither exact BP message has a closed-form Williams product, so the projection
# strategy matters here and MUST be chosen explicitly. Both rules read it from the
# `NaturalGradientMessage` carried in `vconstraint` (select via
# `NGMPDependencies(...; projection = ...)`): `DeltaApproximation` is the
# analytic-derivative second-order projection (`project_to_gamma`/
# `project_to_normal` — biased when the receiving marginal is wide, exactly the
# small-N regime), `Unscented` is the 3-sigma-point middle ground, and
# `Quadrature(n)` is exact to quadrature precision. The default `ClosedForm`
# raises an informative error for these messages, since no exact product exists.
# The exact BP messages (src/expressions/):
#
#   toward τ: μ_{f→τ}(τ) = ∫ 𝒩(y|z,τ⁻¹) 𝒩(z|m̃,ṽ) dz = 𝒩(y | m̃, ṽ + τ⁻¹)
#             — a `NormalPrecisionMessage`, projected at q(τ) → damped Gamma site.
#   toward μ: μ_{f→x}(x) = ∫ 𝒩(y|x,τ⁻¹) Gamma(τ; ã, b̃) dτ ∝ (2b̃ + (x−y)²)^{−(ã+½)}
#             — a `StudentTMessage`, projected at q(μ) → damped Gaussian site.

@rule NormalMeanPrecision(:τ, NaturalGradientMessage) (m_μ::UnivariateNormalDistributionsFamily, q_out::PointMass, q_τ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m̃, ṽ = mean_var(m_μ)
    exact = Logpdf(NormalPrecisionMessage(mean(q_out), m̃, ṽ))
    site = project(resolve_projection(getprojection(vconstraint)), q_τ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule NormalMeanPrecision(:μ, NaturalGradientMessage) (m_τ::GammaDistributionsFamily, q_out::PointMass, q_μ::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    exact = Logpdf(StudentTMessage(mean(q_out), shape(m_τ), rate(m_τ)))
    site = project(resolve_projection(getprojection(vconstraint)), q_μ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
