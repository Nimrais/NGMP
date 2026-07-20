import ExponentialFamily: ExponentialFamilyDistribution, MultivariateNormalDistributionsFamily, MvNormalMeanCovariance
import ClosedFormExpectations: Logpdf

# Forward (:out): the exact pushforward log-density is supported only on the
# positive orthant. `DeltaApproximation` uses its finite local quadratic at
# mean(q_out), whereas Unscented uses the support-safe moment-matched Gaussian
# pushforward of m_in (its sigma points are mapped through softplus first).
@rule MvSoftplus(:out, NaturalGradientMessage) (
    m_in::MultivariateNormalDistributionsFamily,
    q_out::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    projection = resolve_projection(getprojection(vconstraint))
    site = if projection isa TangentProjection{<:DeltaApproximation}
        # A genuine q(out)-dependent tangent projection of the exact forward
        # pushforward. It intentionally raises a clear error if mean(q_out)
        # leaves softplus' positive support.
        exact = Logpdf(MvSoftplusForwardMessage(mean_cov(m_in)...))
        project(projection, q_out, exact)
    elseif projection isa TangentProjection{<:UnscentedTransform}
        # Moment-matched Gaussian pushforward, used by the Unscented strategy.
        α, β, κ = UnscentedTransforms.ut_parameters(UnscentedTransform)
        m, V = mean_cov(m_in)
        x, wm, wc = UnscentedTransforms.gaussian_sigma_points(α, β, κ, m, V)
        s = map(xk -> _softplus.(xk), x)
        μ = sum(wm .* s)
        Σ = sum(wc[k] .* ((s[k] .- μ) * (s[k] .- μ)') for k in eachindex(s))
        Σ = (Σ .+ Σ') ./ 2
        MvNormalMeanCovariance(μ, Σ)
    else
        throw(ArgumentError(
            "MvSoftplus(:out) supports `TangentProjection(type = DeltaApproximation)` " *
            "or `TangentProjection(type = Unscented)`; got $(typeof(projection))",
        ))
    end
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Forward (:out) onto an MvInverseSoftplusNormal edge: the exact pushforward of
# the Gaussian m_in through softplus IS a member of the receiving family, with
# m_in's own natural parameters. The projection is closed-form and exact
# (KL = 0) — no strategy, no support workaround; q_out only selects the edge
# family. Damping still applies in natural-parameter space.
@rule MvSoftplus(:out, NaturalGradientMessage) (
    m_in::MultivariateNormalDistributionsFamily,
    q_out::_MvISNPoint,
    meta::NGMPEdgeState,
) = begin
    ξ, Λ = weightedmean_precision(m_in)
    site = ExponentialFamilyDistribution(
        MvInverseSoftplusNormal,
        vcat(collect(Float64, ξ), vec(collect(Float64, -Λ ./ 2))),
        nothing,
        nothing,
    )
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Delta-method path: unlike the moment-matched forward path above, this is a
# genuine tangent projection of the exact pushforward message at mean(q_out).
# It is valid only while the Gaussian q_out mean lies in softplus' positive
# support; the derivative bundle raises a domain error otherwise rather than
# silently taking a projection of an undefined log-density.
function DerivativeEnhancedFunction(
    p::Logpdf{<:MvSoftplusForwardMessage},
    expansion_point::AbstractVector,
)
    all(>(0), expansion_point) || throw(DomainError(
        expansion_point,
        "MvSoftplus forward delta projection requires mean(q_out) in the positive orthant",
    ))
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        x -> first(_mv_softplus_forward_logderivatives(p.dist, x)),
        x -> last(_mv_softplus_forward_logderivatives(p.dist, x)),
    )
end

function DerivativeEnhancedFunction(
    p::Logpdf{<:MvSoftplusGaussianBackwardMessage},
    expansion_point::AbstractVector,
)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        x -> first(_mv_softplus_backward_logderivatives(p.dist, x)),
        x -> last(_mv_softplus_backward_logderivatives(p.dist, x)),
    )
end

const _MvSoftplusGaussianLogMessage = Union{
    MvSoftplusForwardMessage,
    MvSoftplusGaussianBackwardMessage,
}

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_MvGaussianProjectionPoint,
    exact::Logpdf{<:_MvSoftplusGaussianLogMessage},
)
    qmean, _ = _mv_mean_cov(q)
    ξ, Λ = project_to_mvnormal(DerivativeEnhancedFunction(exact, qmean), q)
    return ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        vcat(ξ, vec(-Λ ./ 2)),
        nothing,
        nothing,
    )
end

# Backward (:in): genuine tangent projection of the EXACT backward
# log-message (finite on all of R^d - no support issue) onto the Gaussian
# input edge at the current marginal q(in).
@rule MvSoftplus(:in, NaturalGradientMessage) (
    m_out::MultivariateNormalDistributionsFamily,
    q_in::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    ξ, Λ = weightedmean_precision(m_out)
    exact = Logpdf(MvSoftplusGaussianBackwardMessage(ξ, Λ))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Backward (:in) from an MvInverseSoftplusNormal out-edge: the message is an
# exponential tilt exp(ηᵀ T(y)) with T(y) = (invsoftplus.(y), ⋯). Substituting
# y = softplus.(x) collapses T to the plain Gaussian statistics (x, x xᵀ), so
# the exact backward message is the Gaussian tilt with the SAME natural
# parameters — closed form, no projection. The softplus nonlinearity is a pure
# change of coordinates between the two edge families.
@rule MvSoftplus(:in, NaturalGradientMessage) (
    m_out::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
    q_in::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    site = ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        collect(Float64, getnaturalparameters(m_out)),
        nothing,
        nothing,
    )
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule MvSoftplus(:in, NaturalGradientMessage) (
    m_out::MvInverseSoftplusNormal,
    q_in::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    site = ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        collect(Float64, getnaturalparameters(_mvisn_site(m_out))),
        nothing,
        nothing,
    )
    return NaturalGradientMP.apply_damping!(meta, site)
end
