# Natural-gradient rules for the stock `MvNormalMeanScalePrecision` node used as a
# GATE — the factor 𝒩(w | μ, κ⁻¹I) with the scalar precision κ latent, produced by
# an upper softdot → Gamma → Log pipeline. Two usages, distinguished by dispatch:
#
# (1) Zero-pseudo-observation gate (out = 0⃗ data, μ = w latent): shrink-to-zero
#     prior on w. The μ and γ edges form one cluster; each outbound message sees
#     the other edge's BP message plus — injected by `NGMPDependencies` — the
#     receiving edge's own marginal (the multivariate `NormalMeanPrecision` analogue).
# (2) Local-deviation gate (out = w latent, μ = w̄ latent, all three edges in one
#     cluster): w ~ 𝒩(w̄, κ⁻¹I) with κ deciding how far the local weights may
#     deviate from the shared profile. All inbound dependencies are BP messages.
#
#   toward κ (1): μ_{f→κ}(κ) = ∫ 𝒩(y|w,κ⁻¹I) 𝒩(w|m̃,Ṽ) dw = 𝒩(y | m̃, Ṽ + κ⁻¹I)
#   toward κ (2): ∫∫ 𝒩(w|w̄,κ⁻¹I) 𝒩(w|m̃,Ṽ) 𝒩(w̄|m̃ᵦ,Ṽᵦ) dw dw̄
#                 = 𝒩(m̃ | m̃ᵦ, Ṽ + Ṽᵦ + κ⁻¹I)
#             — both are an `MvNormalPrecisionMessage`, projected at q(κ) → damped
#             Gamma site. No exact Williams product exists, so the strategy MUST
#             be chosen explicitly (`Unscented` or `Quadrature(n)`; the default
#             `ClosedForm` errors informatively).
#   toward w / w̄: the exact message with κ integrated against Gamma(ã, b̃) is a
#             multivariate Student-t; projecting it would need a multivariate
#             Gaussian projection point, which does not exist yet. Instead κ is
#             collapsed to the mean ã/b̃ of the incoming Gamma MESSAGE — a plug-in
#             conjugate Gaussian (in (2) additionally convolved with the other
#             Gaussian edge's cavity). Covariance matching (ã−1)/b̃ is deliberately
#             not used: damped sites routinely drive ã ≤ 1, where it flips sign.

@rule MvNormalMeanScalePrecision(:γ, NaturalGradientMessage) (m_μ::MultivariateNormalDistributionsFamily, q_out::PointMass, q_γ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m̃, Ṽ = mean_cov(m_μ)
    exact = Logpdf(MvNormalPrecisionMessage(mean(q_out), m̃, Ṽ))
    site = project(resolve_projection(getprojection(vconstraint)), q_γ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_γ::GammaDistributionsFamily, q_out::PointMass) = begin
    return MvNormalMeanScalePrecision(mean(q_out), shape(m_γ) / rate(m_γ))
end

# The node-level `meta = damping` reaches the non-NGMP μ interface unwrapped
# (NGMPDependencies only wraps spec-named edges in NGMPEdgeState); the plug-in
# message is deterministic in m_γ, so the damping meta is simply ignored here.
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_γ::GammaDistributionsFamily, q_out::PointMass, meta::DampingMeta) = begin
    return MvNormalMeanScalePrecision(mean(q_out), shape(m_γ) / rate(m_γ))
end

# --- local-deviation gate (usage 2): out = w, μ = w̄, γ = κ, one cluster ---
#
# The cavity on the deviating edge `w` is the bare structured-softdot message,
# whose precision is rank-1 — it must stay in information form throughout: a
# moment-form conversion of a singular message produces garbage covariances.
# The anchor cavity on `w̄` (prior × the other gate nodes' messages) is proper.

@rule MvNormalMeanScalePrecision(:γ, NaturalGradientMessage) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_γ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    ξ, Λ = weightedmean_precision(m_out)
    m̃ᵦ, Ṽᵦ = mean_cov(m_μ)
    exact = Logpdf(MvNormalDeviationPrecisionMessage(ξ, Λ, m̃ᵦ, Ṽᵦ))
    site = project(resolve_projection(getprojection(vconstraint)), q_γ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# toward w: the anchor cavity is proper, so the plug-in convolution is moment-form
_gate_out_plugin(m_μ, m_γ) = begin
    m̃ᵦ, Ṽᵦ = mean_cov(m_μ)
    MvNormalMeanCovariance(m̃ᵦ, Ṽᵦ + Matrix((rate(m_γ) / shape(m_γ)) * LinearAlgebra.I, length(m̃ᵦ), length(m̃ᵦ)))
end

@rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily) = begin
    return _gate_out_plugin(m_μ, m_γ)
end

@rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, meta::DampingMeta) = begin
    return _gate_out_plugin(m_μ, m_γ)
end

# toward w̄: convolve the (possibly rank-deficient) info-form cavity with κ̄⁻¹I
# analytically — for exp(ξᵀw − ½wᵀΛw) ∗ 𝒩(κ̄⁻¹I) the result is the info-form
# Gaussian with Λ̃ = κ̄I − κ̄²(Λ + κ̄I)⁻¹ (= κ̄Λ(Λ + κ̄I)⁻¹, PSD, same rank as Λ)
# and ξ̃ = κ̄ (Λ + κ̄I)⁻¹ ξ; no inversion of Λ itself is ever required.
_gate_anchor_plugin(m_out, m_γ) = begin
    ξ, Λ = weightedmean_precision(m_out)
    κ̄ = shape(m_γ) / rate(m_γ)
    d = length(ξ)
    C = cholesky(LinearAlgebra.Symmetric(Λ + Matrix(κ̄ * LinearAlgebra.I, d, d)))
    MvNormalWeightedMeanPrecision(κ̄ * (C \ ξ), Matrix(κ̄ * LinearAlgebra.I, d, d) - κ̄^2 * inv(C))
end

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily) = begin
    return _gate_anchor_plugin(m_out, m_γ)
end

@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, meta::DampingMeta) = begin
    return _gate_anchor_plugin(m_out, m_γ)
end
