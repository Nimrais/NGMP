export GaussianStudentTMessage

import FastGaussQuadrature

const _GAUSSIAN_STUDENT_T_ORDER = 16
const _GAUSSIAN_STUDENT_T_NODES, _GAUSSIAN_STUDENT_T_LOGWEIGHTS = let
    nodes, weights = FastGaussQuadrature.gausshermite(_GAUSSIAN_STUDENT_T_ORDER)
    nodes, log.(weights ./ sqrt(pi))
end

"""
    GaussianStudentTMessage(m, v, a, b)

Exact belief-propagation message sent toward one latent Gaussian edge of
`NormalMeanPrecision(out, mu, tau)` after integrating a Gaussian cavity
`Normal(m, v)` on the other location edge and a `GammaShapeRate(a, b)` cavity on
the precision edge:

    m(x) = integral Normal(z; m, v) StudentTMessage(z, a, b)(x) dz.

The density is a Gaussian convolution of a Student-t density. It exists for
`v >= 0`, `a > 0`, and `b > 0`, but generally has no elementary closed form.
Evaluation uses a shared 16-point Gauss-Hermite rule; constants independent of
`x` are omitted because tangent projection only consumes the log-message up to
an additive constant.
"""
struct GaussianStudentTMessage{T <: Real} <: ClosedFormExpectations.Expression
    m::T
    v::T
    a::T
    b::T
end

function GaussianStudentTMessage(m::Real, v::Real, a::Real, b::Real)
    mp, vp, ap, bp = promote(m, v, a, b)
    vp >= 0 || throw(DomainError(v, "Gaussian cavity variance must be nonnegative"))
    ap > 0 || throw(DomainError(a, "Gamma cavity shape must be positive"))
    bp > 0 || throw(DomainError(b, "Gamma cavity rate must be positive"))
    return GaussianStudentTMessage{typeof(mp)}(mp, vp, ap, bp)
end

function _gaussian_student_t_component_log(p::GaussianStudentTMessage, x, index, scale)
    center = p.m + scale * _GAUSSIAN_STUDENT_T_NODES[index]
    residual = x - center
    return _GAUSSIAN_STUDENT_T_LOGWEIGHTS[index] -
           (2 * p.a + 1) / 2 * log(2 * p.b + residual^2)
end

function Base.log(p::GaussianStudentTMessage, x)
    scale = sqrt(2 * p.v)
    maximum_log = -Inf
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        component_log = _gaussian_student_t_component_log(p, x, index, scale)
        maximum_log = max(maximum_log, component_log)
    end

    total = zero(promote_type(typeof(x), typeof(p.m)))
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        component_log = _gaussian_student_t_component_log(p, x, index, scale)
        total += exp(component_log - maximum_log)
    end
    return maximum_log + log(total)
end

(p::GaussianStudentTMessage)(x) = exp(log(p, x))

function _gaussian_student_t_logderivatives(p::GaussianStudentTMessage, x)
    scale = sqrt(2 * p.v)
    multiplier = 2 * p.a + 1
    maximum_log = -Inf
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        component_log = _gaussian_student_t_component_log(p, x, index, scale)
        maximum_log = max(maximum_log, component_log)
    end

    total = zero(promote_type(typeof(x), typeof(p.m)))
    first_total = zero(total)
    second_total = zero(total)
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        center = p.m + scale * _GAUSSIAN_STUDENT_T_NODES[index]
        residual = x - center
        denominator = 2 * p.b + residual^2
        component_log = _GAUSSIAN_STUDENT_T_LOGWEIGHTS[index] -
                        multiplier / 2 * log(denominator)
        weight = exp(component_log - maximum_log)
        first = -multiplier * residual / denominator
        second = -multiplier * (2 * p.b - residual^2) / denominator^2
        total += weight
        first_total += weight * first
        second_total += weight * (second + first^2)
    end

    first = first_total / total
    second = second_total / total - first^2
    return first, second
end

