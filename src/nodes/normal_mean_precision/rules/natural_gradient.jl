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

# All three interfaces latent and in one BP cluster. Integrating the Gaussian
# cavity on the opposite location edge and the Gamma precision cavity produces a
# Gaussian-Student-t convolution, projected at the receiving Gaussian marginal.
@rule NormalMeanPrecision(:out, NaturalGradientMessage) (m_μ::UnivariateNormalDistributionsFamily, m_τ::GammaDistributionsFamily, q_out::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    m, v = mean_var(m_μ)
    exact = Logpdf(GaussianStudentTMessage(m, v, shape(m_τ), rate(m_τ)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule NormalMeanPrecision(:μ, NaturalGradientMessage) (m_out::UnivariateNormalDistributionsFamily, m_τ::GammaDistributionsFamily, q_μ::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    m, v = mean_var(m_out)
    exact = Logpdf(GaussianStudentTMessage(m, v, shape(m_τ), rate(m_τ)))
    site = project(resolve_projection(getprojection(vconstraint)), q_μ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# The difference of two independent Gaussian cavities is Gaussian, so the exact
# precision-edge BP message stays the existing NormalPrecisionMessage.
@rule NormalMeanPrecision(:τ, NaturalGradientMessage) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, q_τ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m_out_mean, m_out_var = mean_var(m_out)
    m_μ_mean, m_μ_var = mean_var(m_μ)
    exact = Logpdf(
        NormalPrecisionMessage(
            m_out_mean,
            m_μ_mean,
            m_out_var + m_μ_var,
        ),
    )
    site = project(resolve_projection(getprojection(vconstraint)), q_τ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# --- product-of-experts usage: ALL THREE edges latent (out in a structured cluster
# --- with μ; τ mean-field apart) — the ReLU-diffusion model's gate consumer.
#
# toward τ: the dependency is the joint local marginal q(out, μ); the exact message
# is NormalPrecisionMessage built from its moments with the cross-covariance
# DELIBERATELY dropped — keeping it is the τ-trap cancellation (the joint hides the
# misfit), the same diagonalization proven at the softdot hybrid gate.
@rule NormalMeanPrecision(:τ, NaturalGradientMessage) (q_out_μ::MultivariateNormalDistributionsFamily, q_τ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m, V = mean_cov(q_out_μ)
    exact = Logpdf(NormalPrecisionMessage(m[1], m[2], V[1, 1] + V[2, 2]))
    site = project(resolve_projection(getprojection(vconstraint)), q_τ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# DampingMeta-tolerant twins of the stock structured rules for the non-NGMP
# interfaces of the same node (the recurring meta-unwrapping gotcha): the (out, μ)
# cluster passes messages; τ enters through its mean-field marginal.
@rule NormalMeanPrecision(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, q_τ::Any, meta::DampingMeta) = begin
    return NormalMeanVariance(mean(m_μ), var(m_μ) + inv(mean(q_τ)))
end

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, q_τ::Any, meta::DampingMeta) = begin
    return NormalMeanVariance(mean(m_out), var(m_out) + inv(mean(q_τ)))
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any, meta::DampingMeta) = begin
    τ̄ = mean(q_τ)
    ξo, po = weightedmean_precision(m_out)
    ξμ, pμ = weightedmean_precision(m_μ)
    W = [ (po + τ̄)  (-τ̄) ; (-τ̄)  (pμ + τ̄) ]
    return MvNormalWeightedMeanPrecision([ξo; ξμ], W)
end
