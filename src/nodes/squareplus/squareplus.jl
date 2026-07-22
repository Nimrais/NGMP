export Squareplus

import ExponentialFamily:
    ExponentialFamilyDistribution,
    NormalMeanVariance,
    getnaturalparameters,
    weightedmean_precision

"""
    Squareplus

Deterministic positive transform

```math
\\operatorname{squareplus}(x) = \\frac{x + \\sqrt{x^2 + 4}}{2}
                              = \\exp(\\operatorname{asinh}(x/2)),
```

with interfaces `[out, in]`.  The scale is fixed so that
`squareplus(0) == 1` and `inv(squareplus(x)) == squareplus(-x)`.  The output is
represented by a Gamma marginal and the input by a Gaussian marginal.
"""
struct Squareplus end

@node Squareplus Deterministic [out, in]

_log_squareplus(x::Real) = asinh(float(x) / 2)
_squareplus(x::Real) = exp(_log_squareplus(x))

function _inverse_squareplus(y::Real)
    y > zero(y) || return oftype(float(y), -Inf)
    return y - inv(y)
end

function _squareplus_log_inverse_jacobian(y::Real)
    y > zero(y) || return oftype(float(y), Inf)
    # log(1 + 1/y^2), evaluated without forming 1/y^2.
    return _softplus(-2log(y))
end

struct SquareplusForwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    mean::T
    variance::T

    function SquareplusForwardMessage{T}(mean::T, variance::T) where {T <: Real}
        variance > zero(T) || throw(DomainError(
            variance,
            "Squareplus forward-message variance must be positive",
        ))
        return new{T}(mean, variance)
    end
end

function SquareplusForwardMessage(mean::Real, variance::Real)
    m, v = promote(float(mean), float(variance))
    return SquareplusForwardMessage{typeof(m)}(m, v)
end

function Base.log(message::SquareplusForwardMessage, y::Real)
    y > zero(y) || return oftype(float(y), -Inf)
    x = _inverse_squareplus(y)
    lognormal = -(
        log(2π * message.variance) +
        (x - message.mean)^2 / message.variance
    ) / 2
    return lognormal + _squareplus_log_inverse_jacobian(y)
end

(message::SquareplusForwardMessage)(y::Real) = exp(log(message, y))

struct SquareplusBackwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    shape::T
    rate::T

    function SquareplusBackwardMessage{T}(shape::T, rate::T) where {T <: Real}
        shape > zero(T) || throw(DomainError(
            shape,
            "Squareplus backward-message Gamma shape must be positive",
        ))
        rate > zero(T) || throw(DomainError(
            rate,
            "Squareplus backward-message Gamma rate must be positive",
        ))
        return new{T}(shape, rate)
    end
end

function SquareplusBackwardMessage(shape::Real, rate::Real)
    a, b = promote(float(shape), float(rate))
    return SquareplusBackwardMessage{typeof(a)}(a, b)
end

function Base.log(message::SquareplusBackwardMessage, x::Real)
    logvalue = _log_squareplus(x)
    value = exp(logvalue)
    # The Gamma normalizer is constant in x and cancels from a tangent
    # projection.
    return (message.shape - 1) * logvalue - message.rate * value
end

(message::SquareplusBackwardMessage)(x::Real) = exp(log(message, x))

function _squareplus_forward_first_derivative(
    p::Logpdf{<:SquareplusForwardMessage},
    y::Real,
)
    y > 0 || throw(DomainError(
        y,
        "Squareplus forward delta projection requires a positive expansion point",
    ))
    message = p.dist
    inverse_value = _inverse_squareplus(y)
    inverse_first = 1 + inv(y^2)
    log_jacobian_first = -2 / (y * (y^2 + 1))
    return -(
        (inverse_value - message.mean) * inverse_first /
        message.variance
    ) + log_jacobian_first
end

function _squareplus_forward_second_derivative(
    p::Logpdf{<:SquareplusForwardMessage},
    y::Real,
)
    y > 0 || throw(DomainError(
        y,
        "Squareplus forward delta projection requires a positive expansion point",
    ))
    message = p.dist
    inverse_value = _inverse_squareplus(y)
    inverse_first = 1 + inv(y^2)
    inverse_second = -2 / y^3
    log_jacobian_second =
        2 * (3y^2 + 1) / (y^2 * (y^2 + 1)^2)
    return -(
        inverse_first^2 +
        (inverse_value - message.mean) * inverse_second
    ) / message.variance + log_jacobian_second
end

function _squareplus_backward_first_derivative(
    p::Logpdf{<:SquareplusBackwardMessage},
    x::Real,
)
    message = p.dist
    value = _squareplus(x)
    inverse_root = inv(hypot(float(x), 2))
    return (message.shape - 1 - message.rate * value) * inverse_root
end

function _squareplus_backward_second_derivative(
    p::Logpdf{<:SquareplusBackwardMessage},
    x::Real,
)
    message = p.dist
    xf = float(x)
    root = hypot(xf, 2)
    inverse_root = inv(root)
    value = _squareplus(xf)
    residual = message.shape - 1 - message.rate * value
    return -message.rate * value * inverse_root^2 -
           residual * xf / root^3
end

function DerivativeEnhancedFunction(
    p::Logpdf{<:SquareplusForwardMessage},
    expansion_point,
)
    expansion_point > 0 || throw(DomainError(
        expansion_point,
        "Squareplus forward delta projection requires mean(q_out) > 0",
    ))
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_squareplus_forward_first_derivative, p),
        Base.Fix1(_squareplus_forward_second_derivative, p),
    )
end

function DerivativeEnhancedFunction(
    p::Logpdf{<:SquareplusBackwardMessage},
    expansion_point,
)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_squareplus_backward_first_derivative, p),
        Base.Fix1(_squareplus_backward_second_derivative, p),
    )
end

function project(
    ::TangentProjection{<:ClosedForm},
    ::_GammaProjectionPoint,
    ::Logpdf{<:SquareplusForwardMessage},
)
    throw(ArgumentError(
        "Gamma Squareplus NGMP has no closed-form Williams product; use " *
        "DeltaApproximation, Unscented, or Quadrature",
    ))
end

function project(
    ::TangentProjection{<:ClosedForm},
    ::_GaussianProjectionPoint,
    ::Logpdf{<:SquareplusBackwardMessage},
)
    throw(ArgumentError(
        "Gaussian Squareplus NGMP has no closed-form Williams product; use " *
        "DeltaApproximation, Unscented, or Quadrature",
    ))
end

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_GammaProjectionPoint,
    exact::Logpdf{<:SquareplusForwardMessage},
)
    q_ef = q isa ExponentialFamilyDistribution ?
           q : convert(ExponentialFamilyDistribution, q)
    expansion_point = mean(q_ef)
    delta_shape, delta_rate = project_to_gamma(
        DerivativeEnhancedFunction(exact, expansion_point),
        q_ef,
    )
    natural = promote(delta_shape, -delta_rate)
    return ExponentialFamilyDistribution(
        Distributions.Gamma,
        collect(natural),
        nothing,
        nothing,
    )
end

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_GaussianProjectionPoint,
    exact::Logpdf{<:SquareplusBackwardMessage},
)
    q_ef = q isa ExponentialFamilyDistribution ?
           q : convert(ExponentialFamilyDistribution, q)
    expansion_point, _ = _mean_var(q_ef)
    weighted_mean, message_precision = project_to_normal(
        DerivativeEnhancedFunction(exact, expansion_point),
        q_ef,
    )
    natural = promote(weighted_mean, -message_precision / 2)
    return ExponentialFamilyDistribution(
        NormalMeanVariance,
        collect(natural),
        nothing,
        nothing,
    )
end

@rule Squareplus(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::GammaDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SquareplusForwardMessage(mean(m_in), var(m_in)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Squareplus(:in, NaturalGradientMessage) (
    m_out::GammaDistributionsFamily,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(SquareplusBackwardMessage(shape(m_out), rate(m_out)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# ReactiveMP scores a unary deterministic node as -H(q_in). Reconstruct the
# local Gaussian input marginal from its cavity and an inexpensive UT site.
@marginalrule Squareplus(:in) (
    m_out::GammaDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
    meta::Any,
) = begin
    exact = Logpdf(SquareplusBackwardMessage(shape(m_out), rate(m_out)))
    site = project(TangentProjection(type = Unscented), m_in, exact)
    site_parameters = getnaturalparameters(site)
    cavity_weighted_mean, cavity_precision = weightedmean_precision(m_in)
    weighted_mean = cavity_weighted_mean + site_parameters[1]
    precision = cavity_precision - 2site_parameters[2]
    scoring_precision = max(float(precision), sqrt(eps(Float64)))
    return NormalWeightedMeanPrecision(weighted_mean, scoring_precision)
end
