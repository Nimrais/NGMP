export ProductPartnerMessage, ProductOutMessage

import FastGaussQuadrature: gausshermite

"""
    ProductPartnerMessage(m_z, v_z, m_p, v_p)

Exact belief-propagation message that the latent×latent product factor
`δ(z − g·h)` sends toward one factor edge (`g`), once the Gaussian cavities on
the product `z` and on the partner edge `h` are integrated out:

    μ_{f→g}(g) = ∫∫ δ(z − g·h) 𝒩(z | m_z, v_z) 𝒩(h | m_p, v_p) dz dh
               = 𝒩(m_z | g·m_p, v_z + g²·v_p).

The integrand is linear-Gaussian in the partner, so the message is closed form —
but as a function of `g` it is not conjugate to a Gaussian belief: `g` scales the
mean AND enters the variance (the same structure class as
[`NormalPrecisionMessage`](@ref)). It is therefore the object a Gaussian tangent
projection (`Unscented`, `Quadrature`) consumes to produce the natural-parameter
message on the `g` edge. One struct serves both directions (swap the partner).

ReactiveMP's stock `typeof(*)` node emits this same density for two latent
Gaussians as a raw `ContinuousUnivariateLogPdf` (with `|x|` factored out of the
variance term); the tests pin the two forms against each other.

# Fields
- `m_z::T`, `v_z::T`: mean and variance of the cavity on the product edge `z`.
- `m_p::T`, `v_p::T`: mean and variance of the cavity on the partner edge.
"""
struct ProductPartnerMessage{T<:Real} <: ClosedFormExpectations.Expression
    m_z::T
    v_z::T
    m_p::T
    v_p::T
end

function ProductPartnerMessage(m_z::Real, v_z::Real, m_p::Real, v_p::Real)
    a, b, c, d = promote(float(m_z), float(v_z), float(m_p), float(v_p))
    return ProductPartnerMessage{typeof(a)}(a, b, c, d)
end

(p::ProductPartnerMessage)(x) = exp(log(p, x))

function Base.log(p::ProductPartnerMessage, x)
    s = p.v_z + x^2 * p.v_p
    return -(log(2π * s) + (p.m_z - x * p.m_p)^2 / s) / 2
end

"""
    ProductOutMessage(m_g, v_g, m_h, v_h; n = 64)

Exact belief-propagation message that the product factor `δ(z − g·h)` sends
toward the product edge `z`, with Gaussian cavities on both factors:

    μ_{f→z}(z) = ∫∫ δ(z − g·h) 𝒩(g | m_g, v_g) 𝒩(h | m_h, v_h) dg dh
               = ∫ 𝒩(z | g·m_h, g²·v_h) 𝒩(g | m_g, v_g) dg.

The density of a product of two Gaussians has no elementary closed form
(ReactiveMP evaluates it as a Bessel-function series); here the remaining 1-D
integral is evaluated on a fixed Gauss–Hermite grid over `g`, precomputed at
construction (`n` even so no node lands on the degenerate `g = 0`). Exact first
and second moments DO exist in closed form (`E[z] = m_g m_h`,
`V[z] = m_g²v_h + m_h²v_g + v_g v_h`) — the moment-matched Gaussian used by the
`ScalarProduct(:out)` rule; this expression serves as the exact reference and as
the target for an (optional) tangent projection of the out edge.
"""
struct ProductOutMessage{T<:Real} <: ClosedFormExpectations.Expression
    g::Vector{T}       # Gauss–Hermite abscissae mapped through 𝒩(m_g, v_g)
    logw::Vector{T}
    m_h::T
    v_h::T
end

function ProductOutMessage(m_g::Real, v_g::Real, m_h::Real, v_h::Real; n::Int = 64)
    t, w = gausshermite(n)
    g = float(m_g) .+ sqrt(2 * float(v_g)) .* t
    logw = log.(w ./ sqrt(π))
    mh, vh = promote(float(m_h), float(v_h))
    T = promote_type(eltype(g), typeof(mh))
    return ProductOutMessage{T}(convert(Vector{T}, g), convert(Vector{T}, logw), mh, vh)
end

(p::ProductOutMessage)(z) = exp(log(p, z))

function Base.log(p::ProductOutMessage, z)
    acc_max = -Inf
    n = length(p.g)
    terms = Vector{typeof(zero(eltype(p.g)) + zero(z))}(undef, n)
    @inbounds for i in 1:n
        gi = p.g[i]
        s = gi^2 * p.v_h
        terms[i] = p.logw[i] - (log(2π * s) + (z - gi * p.m_h)^2 / s) / 2
        acc_max = max(acc_max, terms[i])
    end
    acc = zero(acc_max)
    @inbounds for i in 1:n
        acc += exp(terms[i] - acc_max)
    end
    return acc_max + log(acc)
end
