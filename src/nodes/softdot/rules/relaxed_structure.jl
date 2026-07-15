# Relaxed structured SoftDot with a random Gaussian multiplier.
#
# The local factorization represented here is
#
#     q(y, x) q(θ) q(γ),       y ~ Normal(θ' * x, γ^-1),
#
# where q(θ) is Gaussian rather than a PointMass. ReactiveMP 6.3 already has
# all closed-form structured VMP targets for this factorization. These methods
# therefore delegate to the stock rules; they do not reimplement the algebra.
#
# Why `NormalDistributionsFamily` instead of `Any`?
# -------------------------------------------------
# The structured targets require both mθ = E[θ] and Vθ = Cov(θ). Restricting
# the adapter to Gaussian beliefs documents the approximation family used by the
# hierarchy and prevents a broad `Any` method from claiming support for a random
# distribution whose required moments or Gaussian projection are undefined.
#
# Why does Vθ matter for a message toward y?
# -------------------------------------------
# Averaging the factor log-density over q(θ) gives
#
#   E_qθ[log f] = const - E[γ]/2 * ((y - mθ'x)^2 + x'Vθ*x).
#
# The second term is absent only for a PointMass. With an inbound information-
# form x message (ξx, Λx), define D = Λx + E[γ]Vθ. Marginalizing x from
# this Gaussian potential gives
#
#   m(f -> y) = Normal(mθ' D^-1 ξx,
#                      E[γ]^-1 + mθ' D^-1 mθ).
#
# Consequently, replacing `PointMass` by `Any` in the old fixed-feature helper
# would silently drop Vθ and would be wrong. The stock structured rule below
# includes D exactly.

# The γ edge is still routed through NaturalGradientMessage only to retain the
# package's stateful natural-parameter damping. For random θ we deliberately use
# ReactiveMP's conjugate structured VMP Gamma target, which contains Vθ and the
# y-x cross-covariance, and then damp that target. We do NOT extend the old
# NormalPrecisionMessage rule: after integrating a random θ, θ'x is generally
# non-Gaussian, so that old PointMass rule is no longer exact.
@rule softdot(:γ, NaturalGradientMessage) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::GammaDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    stock_vmp_message = @call_rule softdot(:γ, Marginalisation) (
        q_y_x = q_y_x,
        q_θ = q_θ,
    )
    return NaturalGradientMP.apply_damping!(meta, stock_vmp_message)
end

# Metadata-tolerant stock γ edge for models that damp θ and/or x but leave γ as
# ordinary VMP. This adapter does not damp γ.
@rule softdot(:γ, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:γ, Marginalisation) (
        q_y_x = q_y_x,
        q_θ = q_θ,
    )
end

# DampingMeta belongs to the complete SoftDot node, so it is also passed to its
# ordinary structured interfaces. The Marginalisation adapters below remove that
# metadata and call the corresponding stock ReactiveMP 6.3 VMP rule. When θ or x
# is selected in NGMPDependencies, the NaturalGradientMessage adapters instead
# damp the same stock VMP target in Gaussian natural coordinates.

# VMP message toward the now-random θ. It uses the complete joint q(y, x):
#
#   Λθ = E[γ] E[xx'],
#   ξθ = E[γ] E[xy].
#
# Notice that q(θ) itself is not an input to its VMP message.
@rule softdot(:θ, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:θ, Marginalisation) (
        q_y_x = q_y_x,
        q_γ = q_γ,
    )
end

# Stateful damping of the structured VMP target toward θ. q_θ is the receiving
# marginal inserted by NGMPDependencies; it is deliberately not passed into the
# stock rule because a factor-to-θ VMP target excludes q(θ) itself.
@rule softdot(:θ, NaturalGradientMessage) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::NGMPEdgeState,
) = begin
    stock_vmp_message = @call_rule softdot(:θ, Marginalisation) (
        q_y_x = q_y_x,
        q_γ = q_γ,
    )
    return NaturalGradientMP.apply_damping!(meta, stock_vmp_message)
end

# Structured message toward y. ReactiveMP includes Vθ through
# D = precision(m_x) + E[γ]Vθ, as derived above.
@rule softdot(:y, Marginalisation) (
    m_x::NormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:y, Marginalisation) (
        m_x = m_x,
        q_θ = q_θ,
        q_γ = q_γ,
    )
end

# Structured message toward x. Here uncertainty in θ contributes the positive
# precision E[γ]Vθ in addition to the rank-one mean term.
@rule softdot(:x, Marginalisation) (
    m_y::UnivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:x, Marginalisation) (
        m_y = m_y,
        q_θ = q_θ,
        q_γ = q_γ,
    )
end

# Stateful damping of the structured VMP target toward x. As on the θ edge,
# q_x is only the receiving marginal required by NGMPDependencies to maintain a
# per-edge update state; the undamped target remains ReactiveMP's stock rule.
@rule softdot(:x, NaturalGradientMessage) (
    m_y::UnivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_x::NormalDistributionsFamily,
    q_γ::Any,
    meta::NGMPEdgeState,
) = begin
    stock_vmp_message = @call_rule softdot(:x, Marginalisation) (
        m_y = m_y,
        q_θ = q_θ,
        q_γ = q_γ,
    )
    return NaturalGradientMP.apply_damping!(meta, stock_vmp_message)
end

# Construct q(y, x). The stock lower-right precision block is
#
#   precision(m_x) + E[γ](Vθ + mθmθ'),
#
# so the multiplier uncertainty is retained inside the structured cluster.
@marginalrule SoftDot(:y_x) (
    m_y::NormalDistributionsFamily,
    m_x::NormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_marginalrule SoftDot(:y_x) (
        m_y = m_y,
        m_x = m_x,
        q_θ = q_θ,
        q_γ = q_γ,
    )
end

# Free-energy scoring is also delegated to ReactiveMP. This method only removes
# the damping metadata; no energy expression is copied into this package.
@average_energy softdot (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::NormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    marginals = (
        Marginal(q_y_x, false, false),
        Marginal(q_θ, false, false),
        Marginal(q_γ, false, false),
    )
    return score(
        AverageEnergy(),
        SoftDot,
        Val{(:y_x, :θ, :γ)}(),
        marginals,
        nothing,
    )
end
