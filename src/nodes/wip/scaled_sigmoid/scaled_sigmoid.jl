export ScaledSigmoid

"""
    ScaledSigmoid

Deterministic bounded positive transform `out = scale · σ(in)` with interfaces
`[out, in, scale]`; `scale` is a model constant (PointMass). NGMP rules keep
`out` in the Gamma family and `in` Gaussian, mirroring [`Softplus`](@ref).

The bound is the point: a softplus gate must trade responsibility sharpness
against total ensemble precision (`Σγ` is both the mixture normalizer and the
predictive precision), while a scaled sigmoid saturates the winning gate at
`scale`, so gate weights can grow — and responsibilities sharpen — without the
ensemble precision collapsing or exploding.

Support caveat: `out` lives in `(0, scale)` but the Gamma projection point has
support `(0, ∞)`, so the log-space sigma points of the unscented Gamma
projection can land above `scale`. The forward log-message therefore extends
beyond the bound with a steep finite quadratic penalty instead of `-Inf`,
keeping the UT covariance sums finite.
"""
struct ScaledSigmoid end

@node ScaledSigmoid Deterministic [out, in, scale]

_logistic(x::Real) = inv(one(x) + exp(-x))
# log σ(x) = -softplus(-x), stable on the whole real line.
_log_logistic(x::Real) = -_softplus(-x)

# Fraction of `scale` past which the forward log-density switches from the
# exact pushforward to the clamped-plus-penalty extension. Deliberately soft:
# the Gamma-edge sigma points regularly land beyond the bound, and extension
# values far below the interior log-density scale (≈ -4 … -30) would blow up
# the UT covariance sums and destabilize the sites.
const _SCALED_SIGMOID_SUPPORT_MARGIN = 1e-3
# Width (as a fraction of `scale`) of the quadratic penalty beyond the bound.
const _SCALED_SIGMOID_PENALTY_WIDTH = 0.1

struct ScaledSigmoidForwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    mean::T
    variance::T
    scale::T
end

function ScaledSigmoidForwardMessage(mean::Real, variance::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "ScaledSigmoid scale must be positive"))
    m, v, s = promote(float(mean), float(variance), float(scale))
    return ScaledSigmoidForwardMessage{typeof(m)}(m, v, s)
end

function Base.log(message::ScaledSigmoidForwardMessage, y::Real)
    y > zero(y) || return oftype(float(y), -Inf)
    s = message.scale
    edge = s * (1 - _SCALED_SIGMOID_SUPPORT_MARGIN)
    yc = min(y, edge)
    x = log(yc) - log(s - yc)
    lognormal = -(log(2π * message.variance) + (x - message.mean)^2 / message.variance) / 2
    # |dx/dy| = s / (y (s - y))
    loginversejacobian = log(s) - log(yc) - log(s - yc)
    value = lognormal + loginversejacobian
    if y > edge
        value -= ((y - edge) / (_SCALED_SIGMOID_PENALTY_WIDTH * s))^2
    end
    return value
end

(message::ScaledSigmoidForwardMessage)(y::Real) = exp(log(message, y))

struct ScaledSigmoidBackwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    shape::T
    rate::T
    scale::T
end

function ScaledSigmoidBackwardMessage(shape::Real, rate::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "ScaledSigmoid scale must be positive"))
    a, b, s = promote(float(shape), float(rate), float(scale))
    return ScaledSigmoidBackwardMessage{typeof(a)}(a, b, s)
end

function Base.log(message::ScaledSigmoidBackwardMessage, x::Real)
    # log(s·σ(x)) and s·σ(x), both stable on ℝ. The Gamma normalizer is
    # constant in x and cancels from a tangent projection.
    log_gamma_value = log(message.scale) + _log_logistic(x)
    gamma_value = message.scale * _logistic(x)
    return (message.shape - 1) * log_gamma_value - message.rate * gamma_value
end

(message::ScaledSigmoidBackwardMessage)(x::Real) = exp(log(message, x))

@rule ScaledSigmoid(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    m_scale::PointMass,
    q_out::GammaDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    # A transiently improper edge marginal cannot serve as a projection point
    # (no sigma points). Send a decaying message instead — the flat target
    # shrinks the cumulative site by (1 - α) per firing until the marginal
    # recovers properness.
    if !(isfinite(shape(q_out)) && isfinite(rate(q_out)) && shape(q_out) > 0 && rate(q_out) > 0)
        return NaturalGradientMP.apply_damping!(meta, GammaShapeRate(1.0, 0.0))
    end
    exact = Logpdf(ScaledSigmoidForwardMessage(mean(m_in), var(m_in), mean(m_scale)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule ScaledSigmoid(:in, NaturalGradientMessage) (
    m_out::GammaDistributionsFamily,
    m_scale::PointMass,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    if !(isfinite(mean(q_in)) && isfinite(precision(q_in)) && precision(q_in) > 0)
        return NaturalGradientMP.apply_damping!(meta, NormalWeightedMeanPrecision(0.0, 0.0))
    end
    exact = Logpdf(ScaledSigmoidBackwardMessage(shape(m_out), rate(m_out), mean(m_scale)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Free-energy scoring marginal, mirroring the Softplus(:in) rule: rebuild the
# local Gaussian input marginal from the cavity and the projected backward
# site. The scorer asks for the (in, scale) cluster; the constant scale is a
# point mass and contributes nothing, so the returned belief is the Gaussian.
@marginalrule ScaledSigmoid(:in_scale) (
    m_out::GammaDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
    m_scale::PointMass,
    meta::Any,
) = begin
    exact = Logpdf(ScaledSigmoidBackwardMessage(shape(m_out), rate(m_out), mean(m_scale)))
    site = project(TangentProjection(type = Unscented), m_in, exact)
    site_parameters = getnaturalparameters(site)
    cavity_weighted_mean, cavity_precision = weightedmean_precision(m_in)
    weighted_mean = cavity_weighted_mean + site_parameters[1]
    precision = cavity_precision - 2 * site_parameters[2]
    scoring_precision = max(float(precision), sqrt(eps(Float64)))
    return NormalWeightedMeanPrecision(weighted_mean, scoring_precision)
end
