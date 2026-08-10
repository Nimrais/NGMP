export GaussianLogNormalScaleMessage

"""
    GaussianLogNormalScaleMessage(y, m_s, v_s)

Exact belief-propagation message that a Normal factor with log-precision
parametrization `f(s, z) = 𝒩(y | z, e^{-s})` sends toward its **mean** edge,
once the incoming Gaussian cavity `𝒩(s | m_s, v_s)` on the log-precision edge
has been integrated out:

    μ_{f→z}(z) = ∫ 𝒩(y | z, e^{-s}) 𝒩(s | m_s, v_s) ds.

A log-normal scale mixture of Gaussians: heavier-tailed than any single
Gaussian, with no elementary closed form. Evaluation uses a shared 16-point
Gauss–Hermite rule over `s`; each component is exactly quadratic in `z`.
Constants independent of `z` are kept only up to the mixture weights, which is
sufficient for tangent projection (the log-message is consumed up to an
additive constant).

In the zero-cavity limit `v_s → 0` the message collapses to the Gaussian
`𝒩(y | z, e^{-m_s})` — the mean-field tilted site with a plug-in precision.

# Fields
- `y::T`: the observed value entering the Normal factor.
- `m_s::T`: mean of the Gaussian cavity message on the log-precision edge.
- `v_s::T`: variance of that cavity message (`v_s ≥ 0`).
"""
struct GaussianLogNormalScaleMessage{T <: Real} <: ClosedFormExpectations.Expression
    y::T
    m_s::T
    v_s::T
    function GaussianLogNormalScaleMessage{T}(y::T, m_s::T, v_s::T) where {T <: Real}
        v_s >= 0 || throw(DomainError(v_s, "Gaussian cavity variance must be nonnegative"))
        return new{T}(y, m_s, v_s)
    end
end

function GaussianLogNormalScaleMessage(y::Real, m_s::Real, v_s::Real)
    yp, mp, vp = promote(y, m_s, v_s)
    return GaussianLogNormalScaleMessage{typeof(yp)}(yp, mp, vp)
end

# log w̃ₖ + ½ sₖ − ½ e^{sₖ} (z − y)², with sₖ = m_s + √(2 v_s) uₖ. Shares the
# Gauss–Hermite rule of GaussianStudentTMessage; exponent clamped against
# overflow at extreme cavity points.
function _lognormal_scale_component_log(p::GaussianLogNormalScaleMessage, z, index, scale)
    node = p.m_s + scale * _GAUSSIAN_STUDENT_T_NODES[index]
    precision = exp(clamp(node, -700.0, 700.0))
    return _GAUSSIAN_STUDENT_T_LOGWEIGHTS[index] + node / 2 -
           precision * (z - p.y)^2 / 2
end

function Base.log(p::GaussianLogNormalScaleMessage, z)
    scale = sqrt(2 * p.v_s)
    maximum_log = -Inf
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        component_log = _lognormal_scale_component_log(p, z, index, scale)
        maximum_log = max(maximum_log, component_log)
    end
    isfinite(maximum_log) || return maximum_log

    total = zero(promote_type(typeof(z), typeof(p.m_s)))
    for index in eachindex(_GAUSSIAN_STUDENT_T_NODES)
        component_log = _lognormal_scale_component_log(p, z, index, scale)
        total += exp(component_log - maximum_log)
    end
    return maximum_log + log(total)
end

(p::GaussianLogNormalScaleMessage)(z) = exp(log(p, z))
