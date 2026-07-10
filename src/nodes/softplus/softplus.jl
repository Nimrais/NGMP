export Softplus

import ExponentialFamily: getnaturalparameters, weightedmean_precision

"""
    Softplus

Deterministic positive transform `out = log(1 + exp(in))` with interfaces
`[out, in]`. NGMP rules keep `out` in the Gamma family and `in` Gaussian.
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
