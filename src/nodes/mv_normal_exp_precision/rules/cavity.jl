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

# --- Bethe diagnostics for the cavity configuration --------------------------
#
# The joint cluster marginal q(μ, s) ∝ m_μ(μ) m_s(s) 𝒩(y | μ, e^{-s}) is a
# continuous Gaussian mixture over s (conditional on s the μ-factor is
# conjugate). Moment-match it to a 2-D Gaussian on a dense s-grid; together
# with the closed-form average energy below this makes `free_energy = true`
# usable on the cavity graph. Like the Poisson NGMP trace, the result is a
# surrogate Bethe diagnostic — the local objects change between sweeps, so
# it certifies stability, not descent of VMP's objective.

@marginalrule MvNormalExpPrecision(:μ_s) (
    m_μ::UnivariateNormalDistributionsFamily,
    m_s::UnivariateNormalDistributionsFamily,
    q_out::PointMass,
    meta::Any,
) = begin
    y = mean(q_out)
    mμ, vμ = mean_var(m_μ)
    ms, vs = mean_var(m_s)
    # diagnostic-only regularization: transiently improper incoming messages
    # must not crash the free-energy observer
    vμ = (isfinite(vμ) && vμ > 0) ? vμ : 1e-8
    vs = (isfinite(vs) && vs > 0) ? vs : 1e-8
    sd = sqrt(vs)
    points = collect(range(ms - 8sd, ms + 8sd; length = 129))
    precisions = exp.(_mnep_clamp_exponent.(points))
    variances = exp.(_mnep_clamp_exponent.(.-points))
    log_weights = [
        -abs2(s - ms) / (2vs) -
        (log(vμ + variances[i]) + abs2(y - mμ) / (vμ + variances[i])) / 2
        for (i, s) in enumerate(points)
    ]
    log_weights .-= maximum(log_weights)
    weights = exp.(log_weights)
    weights ./= sum(weights)
    conditional_precision = inv(vμ) .+ precisions
    conditional_mean = (mμ / vμ .+ precisions .* y) ./ conditional_precision
    conditional_variance = inv.(conditional_precision)
    Eμ = sum(weights .* conditional_mean)
    Es = sum(weights .* points)
    Vμ = max(
        sum(weights .* (conditional_variance .+ conditional_mean .^ 2)) - Eμ^2,
        1e-12,
    )
    Vs = max(sum(weights .* points .^ 2) - Es^2, 1e-12)
    C = sum(weights .* points .* conditional_mean) - Es * Eμ
    # keep the matched covariance strictly PSD
    limit = 0.999 * sqrt(Vμ * Vs)
    C = clamp(C, -limit, limit)
    return MvNormalMeanCovariance([Eμ, Es], [Vμ C; C Vs])
end

# U = E_q[−log f] under the joint cluster Gaussian, closed form via the
# lognormal tilt: with a = y − μ, E[e^s a²] = e^{E[s]+V[s]/2}·((y−E[μ]−c)² + V[μ])
# where c = Cov(μ, s) (tilting by e^s shifts a by Cov(a, s) = −c).
@average_energy MvNormalExpPrecision (
    q_out::PointMass,
    q_μ_s::MultivariateNormalDistributionsFamily,
    meta::Any,
) = begin
    y = mean(q_out)
    m, V = mean_cov(q_μ_s)
    length(m) == 2 || error(
        "the joint (μ, s) average energy of MvNormalExpPrecision is scalar-edge only",
    )
    mμ, ms = m
    vμ, vs, c = V[1, 1], V[2, 2], V[1, 2]
    ρ = exp(_mnep_clamp_exponent(ms + vs / 2))
    return log(2π) / 2 - ms / 2 + ρ * (abs2(y - mμ - c) + vμ) / 2
end
