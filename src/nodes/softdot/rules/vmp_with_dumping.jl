# Damped adapter around ReactiveMP's stock mean-field VMP rules for the scalar
# random-random product
#
#     y ~ Normal(θ * x, γ^-1),    q(y)q(θ)q(x)q(γ).
#
# Only the message toward θ is changed: ReactiveMP computes the ordinary VMP
# target and we damp that target in Gaussian natural coordinates. The y and x
# rules below do not change or damp their stock messages; they only accept the
# node-level DampingMeta that is needed by the θ edge.

@rule softdot(:θ, NaturalGradientMessage) (
    q_y::Any,
    q_θ::UnivariateNormalDistributionsFamily,
    q_x::UnivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::NGMPEdgeState,
) = begin
    # q_θ is intentionally not passed to the stock rule. It is the receiving
    # marginal injected by NGMPDependencies so this edge has persistent damping
    # state; the mean-field VMP target itself depends only on q(y), q(x), q(γ).
    stock_vmp_message = @call_rule softdot(:θ, Marginalisation) (
        q_y = q_y,
        q_x = q_x,
        q_γ = q_γ,
    )

    stock_weighted_mean = weightedmean(stock_vmp_message)
    stock_precision = precision(stock_vmp_message)
    return NaturalGradientMP.apply_damping!(
        meta,
        stock_weighted_mean,
        stock_precision,
    )
end

# DampingMeta is attached to the whole SoftDot node, so ReactiveMP also passes
# it to the ordinary y/x interfaces. Forward those calls to the stock VMP rules
# without metadata. No damping is applied on either edge.
@rule softdot(:y, Marginalisation) (
    q_θ::UnivariateNormalDistributionsFamily,
    q_x::UnivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:y, Marginalisation) (
        q_θ = q_θ,
        q_x = q_x,
        q_γ = q_γ,
    )
end

@rule softdot(:x, Marginalisation) (
    q_y::Any,
    q_θ::UnivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    return @call_rule softdot(:x, Marginalisation) (
        q_y = q_y,
        q_θ = q_θ,
        q_γ = q_γ,
    )
end

@average_energy softdot (
    q_y::Any,
    q_θ::UnivariateNormalDistributionsFamily,
    q_x::UnivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::DampingMeta,
) = begin
    marginals = (
        Marginal(q_y, false, false),
        Marginal(q_θ, false, false),
        Marginal(q_x, false, false),
        Marginal(q_γ, false, false),
    )
    return score(
        AverageEnergy(),
        SoftDot,
        Val{(:y, :θ, :x, :γ)}(),
        marginals,
        nothing,
    )
end
