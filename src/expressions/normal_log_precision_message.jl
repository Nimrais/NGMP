export NormalLogPrecisionMessage

"""
    NormalLogPrecisionMessage(y, m̃, ṽ)

Exact belief-propagation message that a Normal factor with **log-precision**
parametrization

    f(s, z) = 𝒩(y | z, e^{-s})

sends toward its log-precision edge `s`, once the incoming Gaussian cavity
`𝒩(z | m̃, ṽ)` on the mean edge has been integrated out:

    μ_{f→s}(s) = ∫ 𝒩(y | z, e^{-s}) 𝒩(z | m̃, ṽ) dz = 𝒩(y | m̃, ṽ + e^{-s}).

This is [`NormalPrecisionMessage`](@ref) evaluated at `τ = e^s` — a message
function of the edge variable, so the reparametrization carries no Jacobian.
The `ṽ`-dependent log-determinant `-½ log(ṽ + e^{-s})` is the beyond-mean-field
correction: in the zero-cavity limit `ṽ → 0` the message collapses to the
tilted mean-field site `½ s - ½ (y - m̃)² e^{s}` (the scalar
`ExpGammaSiteMessage`), which is exactly what the marginal-based rule uses.

# Fields
- `y::T`: the observed value entering the Normal factor.
- `m̃::T`: mean of the Gaussian cavity message on the mean edge.
- `ṽ::T`: variance of the Gaussian cavity message (`ṽ ≥ 0`).
"""
struct NormalLogPrecisionMessage{T <: Real} <: ClosedFormExpectations.Expression
    y::T
    m̃::T
    ṽ::T
    function NormalLogPrecisionMessage{T}(y::T, m̃::T, ṽ::T) where {T <: Real}
        ṽ >= 0 || throw(DomainError(ṽ, "Gaussian cavity variance must be nonnegative"))
        return new{T}(y, m̃, ṽ)
    end
end

function NormalLogPrecisionMessage(y::Real, m̃::Real, ṽ::Real)
    yp, mp, vp = promote(y, m̃, ṽ)
    return NormalLogPrecisionMessage{typeof(yp)}(yp, mp, vp)
end

# ṽ + e^{-s} with the exponent clamped so that extreme quadrature/sigma points
# neither overflow nor collapse the total variance to an exact zero.
function _log_precision_total_variance(p::NormalLogPrecisionMessage, s)
    return p.ṽ + exp(clamp(-s, -700.0, 700.0)) + floatmin(Float64)
end

function Base.log(p::NormalLogPrecisionMessage, s)
    total = _log_precision_total_variance(p, s)
    return -(log(2π * total) + (p.y - p.m̃)^2 / total) / 2
end

(p::NormalLogPrecisionMessage)(s) = exp(log(p, s))
