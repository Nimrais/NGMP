# Cavity-based (true NGMP) rules for the univariate MvNormalExpPrecision node.
#
# The marginal-based rules in univariate.jl project the VMP tilted site
# ½s − ½ E_{q(μ)}[(y−μ)²] e^s — the expected log-factor over the OTHER edge's
# marginal, which shares its stationary points with projected VMP. These rules
# instead project the EXACT belief-propagation log-messages obtained by
# integrating the cavity message on the other edge inside the log:
#
#   toward s:  ℓ(s) = log 𝒩(y | m̃_μ, ṽ_μ + e^{-s})   (determinant correction)
#   toward μ:  ℓ(μ) = log ∫ 𝒩(y | μ, e^{-s}) 𝒩(s | m̃_s, ṽ_s) ds
#                                                     (log-normal scale mixture)
#
# They fire when μ and s share a factorization cluster (no mean-field split
# between them), with `NGMPDependencies(s = nothing, μ = nothing)` — the
# dependency port then routes the other edge as a MESSAGE (m_·) and injects the
# receiving edge's own marginal last. Signatures mirror the NormalMeanPrecision
# cavity rules one interface over.
#
# The sites have no ClosedForm projection; use a quadrature/unscented tangent
# projection, e.g. `projection = TangentProjection(type = Quadrature(32))`.
# Cost per edge and sweep is a handful of scalar evaluations — no inner
# optimization. Transiently improper INCOMING cavities damp toward a flat
# site (the softdot idiom), never clip. Projected OUTGOING sites may
# legitimately be improper (both exact log-messages are non-concave in
# places); they are handled by the η-space damping, and flattening them
# measurably degrades the small-data regime — do not "repair" them. Known
# boundary: under very wide level priors (log-precision prior sd ≳ 6) the
# accumulated marginals can go non-PSD; that regime is outside the model's
# healthy range for every method (see the level_sd sweep in the study).

@rule MvNormalExpPrecision(:s, NaturalGradientMessage) (
    m_μ::UnivariateNormalDistributionsFamily,
    q_out::PointMass,
    q_s::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    m̃, ṽ = mean_var(m_μ)
    (isfinite(m̃) && isfinite(ṽ) && ṽ > 0) || return NaturalGradientMP.apply_damping!(
        meta, NormalWeightedMeanPrecision(0.0, 0.0),
    )
    exact = Logpdf(NormalLogPrecisionMessage(mean(q_out), m̃, ṽ))
    site = project(resolve_projection(getprojection(vconstraint)), q_s, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule MvNormalExpPrecision(:μ, NaturalGradientMessage) (
    m_s::UnivariateNormalDistributionsFamily,
    q_out::PointMass,
    q_μ::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    m̃, ṽ = mean_var(m_s)
    (isfinite(m̃) && isfinite(ṽ) && ṽ > 0) || return NaturalGradientMP.apply_damping!(
        meta, NormalWeightedMeanPrecision(0.0, 0.0),
    )
    exact = Logpdf(GaussianLogNormalScaleMessage(mean(q_out), m̃, ṽ))
    site = project(resolve_projection(getprojection(vconstraint)), q_μ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
