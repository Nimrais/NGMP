# Natural-gradient rules for the stock `softdot` node with a LATENT per-observation
# precision — the gated-precision design: z ~ 𝒩(wᵀf, κ⁻¹) where κ[i,j] is produced
# by an upper softdot → Gamma → Log pipeline and decides how strongly the lower
# weight layer w is coupled to z at this input (κ high = layer ON, κ low = OFF).
#
# Interfaces [y = z, θ = features (PointMass data), x = w, γ = κ]; y, x and γ sit in
# one BP cluster, so rules see the other edges' messages plus — injected by
# `NGMPDependencies` on the γ interface — the receiving edge's own marginal.
#
#   toward κ: μ_{f→κ}(κ) = ∫∫ 𝒩(z | wᵀf, κ⁻¹) m(z) m(w) dz dw
#                        = 𝒩(m_z | fᵀm̃_w, ṽ_z + fᵀṼ_w f + κ⁻¹)
#             — a scalar `NormalPrecisionMessage`, projected at q(κ) → damped Gamma
#             site (the strategy MUST be chosen explicitly; ClosedForm errors).
#   toward z / w: the exact messages with κ integrated against its Gamma message are
#             Student-t-like; κ is collapsed to the Gamma-MESSAGE mean ã/b̃ — the
#             plug-in conjugate pattern proven at the gate node ((ã−1)/b̃ rejected:
#             damped sites routinely drive ã ≤ 1, where it flips sign).
#
# The w-edge cavity `m_x` is the product of the weight prior and the other
# observations' rank-1 softdot sites — proper, unlike the bare-site cavity at the
# MvNormalMeanScalePrecision gate, so moment-form conversion is safe here.

@rule softdot(:γ, NaturalGradientMessage) (m_y::UnivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_θ::PointMass, q_γ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    f = mean(q_θ)
    m̃w, Ṽw = mean_cov(m_x)
    mz, vz = mean_var(m_y)
    exact = Logpdf(NormalPrecisionMessage(mz, dot(f, m̃w), vz + dot(f, Ṽw * f)))
    site = project(resolve_projection(getprojection(vconstraint)), q_γ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Hybrid gate for a STRUCTURED q(w, z) cluster with κ factorized apart. The rule
# receives the joint local marginal over (z, w), but the exact precision message
# needs the independent Gaussian CAVITIES on z and w. Using the marginal directly
# is another form of the τ-trap: the current softdot factor has already narrowed
# both variances and correlated z with w, so feeding those moments back toward κ
# rewards κ for explaining its own posterior contraction.
#
# Under the structured softdot marginal, the current factor contributes
#
#     κ̄ [1; -f] [1; -f]'
#
# to the joint precision and no weighted-mean term. Subtracting that block
# therefore recovers the two cavity precision blocks exactly. We then integrate
# those cavities as in the ordinary BP rule above and project the resulting
# NormalPrecisionMessage at q(κ).
@rule softdot(:γ, NaturalGradientMessage) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::PointMass, q_γ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    f = mean(q_θ)
    ξ, Λ = weightedmean_precision(q_y_x)
    # The reactive q(y, x) stream and q(κ) stream can be one update out of sync.
    # Read the factor strength that was actually used to construct THIS joint
    # marginal from its off-diagonal precision block:
    #
    #     Λ[y, x] = -κ̄ f'.
    #
    # Using mean(q_γ) here can over-subtract during that transient and create a
    # spurious non-positive cavity precision in deeper asynchronous graphs.
    feature_energy = dot(f, f)
    κbar = feature_energy > eps(eltype(Λ)) ?
        -dot(view(Λ, 1, 2:size(Λ, 2)), f) / feature_energy :
        mean(q_γ)

    py = Λ[1, 1] - κbar
    # In a deep loopy graph the Exp-side Gaussian site can be locally convex,
    # hence the exact cavity site need not be normalisable on its own even though
    # q(y, x) is proper. NormalPrecisionMessage needs a proper Gaussian cavity;
    # project such a transient to the vague boundary instead of feeding the
    # current factor's own contraction back into κ.
    cavity_floor =
        sqrt(eps(eltype(Λ))) *
        max(abs(Λ[1, 1]), abs(κbar), one(eltype(Λ)))
    py = max(py, cavity_floor)
    my = ξ[1] / py
    vy = inv(py)

    ξx = view(ξ, 2:length(ξ))
    Λx = Matrix(view(Λ, 2:size(Λ, 1), 2:size(Λ, 2))) .- κbar .* (f * f')
    Cx = cholesky(Hermitian(Λx))
    cavity_solutions = Cx \ hcat(ξx, f)
    mx = view(cavity_solutions, :, 1)
    fx_variance = dot(f, view(cavity_solutions, :, 2))

    exact = Logpdf(NormalPrecisionMessage(
        my,
        dot(f, mx),
        vy + fx_variance,
    ))
    site = project(resolve_projection(getprojection(vconstraint)), q_γ, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# toward z: 𝒩(fᵀm̃_w, fᵀṼ_w f + κ̄⁻¹) — info-form fast path (one Cholesky, as in
# structured_info_form.jl) plus a generic moment-form fallback
_softdot_y_plugin(ξx, Λx, f, κ̄) = begin
    C = cholesky(Hermitian(Λx))
    sol = C \ hcat(ξx, f)
    NormalMeanVariance(dot(f, view(sol, :, 1)), dot(f, view(sol, :, 2)) + inv(κ̄))
end

_softdot_y_plugin_mom(m_x, m_γ, q_θ) = begin
    f = mean(q_θ)
    m̃w, Ṽw = mean_cov(m_x)
    NormalMeanVariance(dot(f, m̃w), dot(f, Ṽw * f) + rate(m_γ) / shape(m_γ))
end

# the node-level `meta = damping` reaches the non-NGMP interfaces unwrapped
# (NGMPDependencies only wraps spec-named edges), hence the meta-accepting twins
@rule softdot(:y, Marginalisation) (m_x::MvNormalWeightedMeanPrecision, m_γ::GammaDistributionsFamily, q_θ::PointMass) = begin
    return _softdot_y_plugin(weightedmean(m_x), precision(m_x), mean(q_θ), shape(m_γ) / rate(m_γ))
end

@rule softdot(:y, Marginalisation) (m_x::MvNormalWeightedMeanPrecision, m_γ::GammaDistributionsFamily, q_θ::PointMass, meta::DampingMeta) = begin
    return _softdot_y_plugin(weightedmean(m_x), precision(m_x), mean(q_θ), shape(m_γ) / rate(m_γ))
end

@rule softdot(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, q_θ::PointMass) = begin
    return _softdot_y_plugin_mom(m_x, m_γ, q_θ)
end

@rule softdot(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, q_θ::PointMass, meta::DampingMeta) = begin
    return _softdot_y_plugin_mom(m_x, m_γ, q_θ)
end

# toward w: the stock structured rank-1 site with κ̄ plugged in for mean(q_γ)
_softdot_x_plugin(m_y, m_γ, q_θ) = begin
    f = mean(q_θ)
    my, vy = mean_var(m_y)
    c = inv(vy + rate(m_γ) / shape(m_γ))
    MvNormalWeightedMeanPrecision((c * my) .* f, c .* (f * f'))
end

# --- DampingMeta-tolerant twins of the STOCK structured rules -----------------
# In the hybrid gate the softdot node carries `meta = damping` for its NGMP κ
# edge; the non-NGMP interfaces receive that meta unwrapped, which would other-
# wise break dispatch of the stock (meta-less) structured rules. Same formulas,
# q_γ marginal (κ mean-field w.r.t. the (z,w) cluster).

@rule softdot(:y, Marginalisation) (m_x::MvNormalWeightedMeanPrecision, q_θ::PointMass, q_γ::Any, meta::DampingMeta) = begin
    return _softdot_y_plugin(weightedmean(m_x), precision(m_x), mean(q_θ), mean(q_γ))
end

@rule softdot(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_θ::PointMass, q_γ::Any, meta::DampingMeta) = begin
    f = mean(q_θ)
    m̃w, Ṽw = mean_cov(m_x)
    return NormalMeanVariance(dot(f, m̃w), dot(f, Ṽw * f) + inv(mean(q_γ)))
end

@rule softdot(:x, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, q_θ::PointMass, q_γ::Any, meta::DampingMeta) = begin
    f = mean(q_θ)
    my, vy = mean_var(m_y)
    c = inv(vy + inv(mean(q_γ)))
    return MvNormalWeightedMeanPrecision((c * my) .* f, c .* (f * f'))
end

@marginalrule SoftDot(:y_x) (m_y::UnivariateNormalDistributionsFamily, m_x::MvNormalWeightedMeanPrecision, q_θ::PointMass, q_γ::Any, meta::DampingMeta) = begin
    f = mean(q_θ)
    mγ = mean(q_γ)
    ξy, py = weightedmean_precision(m_y)
    ξx = weightedmean(m_x)
    Λx = precision(m_x)
    W = [ (py + mγ)  (-mγ .* f)' ; (-mγ .* f)  (Λx .+ mγ .* (f * f')) ]
    return MvNormalWeightedMeanPrecision([ξy; ξx], W)
end

@marginalrule SoftDot(:y_x) (m_y::UnivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_θ::PointMass, q_γ::Any, meta::DampingMeta) = begin
    f = mean(q_θ)
    mγ = mean(q_γ)
    ξy, py = weightedmean_precision(m_y)
    ξx, Λx = weightedmean_precision(m_x)
    W = [ (py + mγ)  (-mγ .* f)' ; (-mγ .* f)  (Λx .+ mγ .* (f * f')) ]
    return MvNormalWeightedMeanPrecision([ξy; ξx], W)
end

@rule softdot(:x, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, q_θ::PointMass) = begin
    return _softdot_x_plugin(m_y, m_γ, q_θ)
end

@rule softdot(:x, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, m_γ::GammaDistributionsFamily, q_θ::PointMass, meta::DampingMeta) = begin
    return _softdot_x_plugin(m_y, m_γ, q_θ)
end
