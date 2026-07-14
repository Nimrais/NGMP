export Softplus

import ExponentialFamily: getnaturalparameters, weightedmean_precision

"""
    Softplus

Deterministic positive transform `out = log(1 + exp(in))` with interfaces
`[out, in]`. The output can be represented either as Gamma (the original
positive-support path) or by an NGMP tangent-projected Gaussian for direct
composition with Gaussian nodes such as `softdot`; the input remains Gaussian.
"""
struct Softplus end

@node Softplus Deterministic [out, in]

# Stable scalar transform and inverse. The inverse identity
# x = y + log(1 - exp(-y)) avoids overflow for large positive y.
_softplus(x::Real) = max(x, zero(x)) + log1p(exp(-abs(x)))
_logsoftplus(x::Real) = x < -37 ? float(x) : log(_softplus(x))

function _inverse_softplus(y::Real)
    y > zero(y) || return oftype(float(y), -Inf)
    return y + log(-expm1(-y))
end

struct SoftplusForwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    mean::T
    variance::T
end

function SoftplusForwardMessage(mean::Real, variance::Real)
    m, v = promote(float(mean), float(variance))
    return SoftplusForwardMessage{typeof(m)}(m, v)
end

function Base.log(message::SoftplusForwardMessage, y::Real)
    y > zero(y) || return oftype(float(y), -Inf)
    x = _inverse_softplus(y)
    lognormal = -(log(2π * message.variance) + (x - message.mean)^2 / message.variance) / 2
    loginversejacobian = -log(-expm1(-y))
    return lognormal + loginversejacobian
end

(message::SoftplusForwardMessage)(y::Real) = exp(log(message, y))

struct SoftplusBackwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    shape::T
    rate::T
end

function SoftplusBackwardMessage(shape::Real, rate::Real)
    a, b = promote(float(shape), float(rate))
    return SoftplusBackwardMessage{typeof(a)}(a, b)
end

function Base.log(message::SoftplusBackwardMessage, x::Real)
    y = _softplus(x)
    # The Gamma normalizer is constant in x and cancels from a tangent projection.
    return (message.shape - 1) * _logsoftplus(x) - message.rate * y
end

(message::SoftplusBackwardMessage)(x::Real) = exp(log(message, x))

"""
    SoftplusGaussianBackwardMessage(mean, variance)

Exact backward log-message through the deterministic transform when the
outgoing cavity is Gaussian:

    log μ(in) = log 𝒩(softplus(in); mean, variance).

There is no Jacobian in the backward direction.  The message is not Gaussian,
so the rule below projects it onto the Gaussian input edge.
"""
struct SoftplusGaussianBackwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    mean::T
    variance::T
end

function SoftplusGaussianBackwardMessage(mean::Real, variance::Real)
    variance > 0 || throw(DomainError(variance, "Gaussian variance must be positive"))
    m, v = promote(float(mean), float(variance))
    return SoftplusGaussianBackwardMessage{typeof(m)}(m, v)
end

function Base.log(message::SoftplusGaussianBackwardMessage, x::Real)
    residual = _softplus(x) - message.mean
    return -(log(2π * message.variance) + residual^2 / message.variance) / 2
end

(message::SoftplusGaussianBackwardMessage)(x::Real) = exp(log(message, x))

# Analytic derivatives of the exact forward log-message.  The receiving
# Gaussian's expansion point must be positive; the exact pushforward density is
# zero on y <= 0, so no finite touching quadratic exists there.
function _softplus_forward_first_derivative(p::Logpdf{<:SoftplusForwardMessage}, y::Real)
    y > 0 || throw(DomainError(
        y,
        "Softplus forward delta projection requires a positive expansion point",
    ))
    message = p.dist
    r = exp(-y)
    d = -expm1(-y)                 # 1 - exp(-y), stable near zero
    x = _inverse_softplus(y)
    return -(x - message.mean) / (message.variance * d) - r / d
end

function _softplus_forward_second_derivative(p::Logpdf{<:SoftplusForwardMessage}, y::Real)
    y > 0 || throw(DomainError(
        y,
        "Softplus forward delta projection requires a positive expansion point",
    ))
    message = p.dist
    r = exp(-y)
    d = -expm1(-y)
    x = _inverse_softplus(y)
    inverse_first = inv(d)
    inverse_second = -r / d^2
    return -(inverse_first^2 + (x - message.mean) * inverse_second) / message.variance + r / d^2
end

function _softplus_gaussian_backward_first_derivative(
    p::Logpdf{<:SoftplusGaussianBackwardMessage}, x::Real
)
    message = p.dist
    slope = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
    return -(_softplus(x) - message.mean) * slope / message.variance
end

function _softplus_gaussian_backward_second_derivative(
    p::Logpdf{<:SoftplusGaussianBackwardMessage}, x::Real
)
    message = p.dist
    slope = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
    curvature = slope * (1 - slope)
    return -(slope^2 + (_softplus(x) - message.mean) * curvature) / message.variance
end

function DerivativeEnhancedFunction(p::Logpdf{<:SoftplusForwardMessage}, expansion_point)
    expansion_point > 0 || throw(DomainError(
        expansion_point,
        "Softplus forward delta projection requires mean(q_out) > 0",
    ))
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_softplus_forward_first_derivative, p),
        Base.Fix1(_softplus_forward_second_derivative, p),
    )
end

function DerivativeEnhancedFunction(p::Logpdf{<:SoftplusGaussianBackwardMessage}, expansion_point)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_softplus_gaussian_backward_first_derivative, p),
        Base.Fix1(_softplus_gaussian_backward_second_derivative, p),
    )
end

const _SoftplusGaussianLogMessage = Union{
    SoftplusForwardMessage,
    SoftplusGaussianBackwardMessage,
}

function project(
    ::TangentProjection{<:ClosedForm},
    ::_GaussianProjectionPoint,
    ::Logpdf{<:_SoftplusGaussianLogMessage},
)
    throw(ArgumentError(
        "Gaussian Softplus NGMP requires an explicit DeltaApproximation, " *
        "Unscented, or Quadrature tangent projection",
    ))
end

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_GaussianProjectionPoint,
    exact::Logpdf{<:_SoftplusGaussianLogMessage},
)
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    m, _ = _mean_var(q_ef)
    weighted_mean, message_precision = project_to_normal(
        DerivativeEnhancedFunction(exact, m), q_ef
    )
    natural = promote(weighted_mean, -message_precision / 2)
    return ExponentialFamilyDistribution(
        NormalMeanVariance, collect(natural), nothing, nothing
    )
end

@rule Softplus(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SoftplusForwardMessage(mean(m_in), var(m_in)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Softplus(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::GammaDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SoftplusForwardMessage(mean(m_in), var(m_in)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Softplus(:in, NaturalGradientMessage) (
    m_out::GammaDistributionsFamily,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SoftplusBackwardMessage(shape(m_out), rate(m_out)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Softplus(:in, NaturalGradientMessage) (
    m_out::UnivariateGaussianDistributionsFamily,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SoftplusGaussianBackwardMessage(mean(m_out), var(m_out)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# ReactiveMP scores unary deterministic nodes as -H(q_in). Rebuild the local
# Gaussian input marginal from both cavities using the inexpensive Gaussian UT.
@marginalrule Softplus(:in) (
    m_out::GammaDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
    meta::Any,
) = begin
    exact = Logpdf(SoftplusBackwardMessage(shape(m_out), rate(m_out)))
    site = project(TangentProjection(type = Unscented), m_in, exact)
    site_parameters = getnaturalparameters(site)
    cavity_weighted_mean, cavity_precision = weightedmean_precision(m_in)
    weighted_mean = cavity_weighted_mean + site_parameters[1]
    precision = cavity_precision - 2 * site_parameters[2]

    # This marginal is used only by the deterministic free-energy scorer. A
    # locally non-concave projected site can be improper while the variable
    # belief remains valid; keep the scoring approximation finite in that case.
    scoring_precision = max(float(precision), sqrt(eps(Float64)))
    return NormalWeightedMeanPrecision(weighted_mean, scoring_precision)
end

@marginalrule Softplus(:in) (
    m_out::UnivariateGaussianDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
    meta::Any,
) = begin
    exact = Logpdf(SoftplusGaussianBackwardMessage(mean(m_out), var(m_out)))
    site = project(TangentProjection(type = Unscented), m_in, exact)
    site_parameters = getnaturalparameters(site)
    cavity_weighted_mean, cavity_precision = weightedmean_precision(m_in)
    weighted_mean = cavity_weighted_mean + site_parameters[1]
    precision = cavity_precision - 2 * site_parameters[2]
    scoring_precision = max(float(precision), sqrt(eps(Float64)))
    return NormalWeightedMeanPrecision(weighted_mean, scoring_precision)
end
