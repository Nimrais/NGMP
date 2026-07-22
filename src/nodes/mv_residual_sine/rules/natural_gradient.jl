import ClosedFormExpectations: Logpdf
import ExponentialFamily:
    ExponentialFamilyDistribution,
    MultivariateNormalDistributionsFamily,
    MvNormalMeanCovariance,
    getnaturalparameters

function DerivativeEnhancedFunction(
    p::Logpdf{<:MvResidualSineForwardMessage},
    expansion_point::AbstractVector,
)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        y -> first(_mv_residual_sine_forward_logderivatives(p.dist, y)),
        y -> last(_mv_residual_sine_forward_logderivatives(p.dist, y)),
    )
end

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_MvGaussianProjectionPoint,
    exact::Logpdf{<:MvResidualSineForwardMessage},
)
    qmean, _ = _mv_mean_cov(q)
    weighted_mean, precision = project_to_mvnormal(
        DerivativeEnhancedFunction(exact, qmean),
        q,
    )
    return ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        vcat(weighted_mean, vec(-precision ./ 2)),
        nothing,
        nothing,
    )
end

function project(
    ::TangentProjection{<:ClosedForm},
    q::_MvGaussianProjectionPoint,
    exact::Logpdf{<:MvResidualSineGaussianBackwardMessage},
)
    return _project_mv_residual_sine_backward(q, exact.dist)
end

"""
    _mv_residual_sine_forward_site(activation, projection, m_in, q_out)

Construct the Gaussian forward site for `MvResidualSine`. The approximation is
selected entirely by dispatch on `projection`; `activation` contains only the
parameters of the nonlinear map.

The delta method is a genuine `q_out`-dependent tangent approximation of the
exact pushforward log-density. The closed-form method exactly matches the first
two pushforward moments, so it is an assumed-density update rather than a Fisher
tangent projection.
"""
function _mv_residual_sine_forward_site(
    activation::ResidualSineMeta,
    projection::TangentProjection{<:DeltaApproximation},
    m_in::MultivariateNormalDistributionsFamily,
    q_out::_MvGaussianProjectionPoint,
)
    exact = Logpdf(MvResidualSineForwardMessage(mean_cov(m_in)..., activation))
    return project(projection, q_out, exact)
end

function _mv_residual_sine_forward_site(
    activation::ResidualSineMeta,
    ::TangentProjection{<:ClosedForm},
    m_in::MultivariateNormalDistributionsFamily,
    ::_MvGaussianProjectionPoint,
)
    transformed_mean, transformed_covariance =
        _mv_residual_sine_mean_cov(mean_cov(m_in)..., activation)
    return MvNormalMeanCovariance(transformed_mean, transformed_covariance)
end

function _mv_residual_sine_forward_site(
    ::ResidualSineMeta,
    projection::TangentProjection,
    ::MultivariateNormalDistributionsFamily,
    ::_MvGaussianProjectionPoint,
)
    throw(ArgumentError(
        "MvResidualSine(:out) supports TangentProjection(type = " *
        "DeltaApproximation) or TangentProjection(type = ClosedForm); got " *
        "$(typeof(projection))",
    ))
end

# Forward Gaussian edge: strategy selection is a single multiple-dispatch call.
@rule MvResidualSine(:out, NaturalGradientMessage) (
    m_in::MultivariateNormalDistributionsFamily,
    q_out::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    activation = _residual_sine_meta(meta)
    projection = resolve_projection(getprojection(vconstraint))
    site = _mv_residual_sine_forward_site(activation, projection, m_in, q_out)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Backward Gaussian edge. This is the exact Fisher tangent site under q(in) for
# both forward modes; no sigma points or local delta expansion are used.
@rule MvResidualSine(:in, NaturalGradientMessage) (
    m_out::MultivariateNormalDistributionsFamily,
    q_in::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    activation = _residual_sine_meta(meta)
    xi, Lambda = weightedmean_precision(m_out)
    exact = Logpdf(MvResidualSineGaussianBackwardMessage(xi, Lambda, activation))
    site = project(TangentProjection(type = ClosedForm), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
