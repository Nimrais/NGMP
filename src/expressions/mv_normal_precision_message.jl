export MvNormalPrecisionMessage, MvNormalDeviationPrecisionMessage

import LinearAlgebra
import LinearAlgebra: Symmetric, eigen, Diagonal, cholesky, logdet, dot

"""
    MvNormalPrecisionMessage(y, m̃, Ṽ)

`MvNormalPrecisionMessage` represents the exact belief-propagation message that a
multivariate Normal factor with **unknown scalar precision**

    f(κ, w) = 𝒩(y | w, κ⁻¹ I)

sends toward its precision edge `κ`, once the incoming Gaussian cavity
`𝒩(w | m̃, Ṽ)` on the mean edge `w` has been integrated out — the multivariate
analogue of [`NormalPrecisionMessage`](@ref):

    μ_{f→κ}(κ) = ∫ 𝒩(y | w, κ⁻¹ I) 𝒩(w | m̃, Ṽ) dw = 𝒩(y | m̃, Ṽ + κ⁻¹ I).

The isotropic `κ⁻¹ I` commutes with the eigendecomposition `Ṽ = Q Λ Qᵀ`, so the
message factorizes over the cavity's eigendirections. The constructor performs
that decomposition **once** and stores the eigenvalues `λ` together with the
rotated residual `c = Qᵀ(y − m̃)`; each subsequent evaluation is then O(d):

    log μ_{f→κ}(κ) = -½ Σₖ [ log(2π (λₖ + κ⁻¹)) + cₖ² / (λₖ + κ⁻¹) ].

As a function of `κ` this is not affine in the Gamma sufficient statistics
`(log κ, κ)` — same non-conjugacy as the scalar case — so it is the object a
Gamma tangent projection (`Unscented`, `Quadrature`) consumes to produce the
natural-parameter message on the `κ` edge. For `d = 1` it coincides with
`NormalPrecisionMessage(y, m̃, ṽ)`.

# Fields
- `λ::Vector{T}`: eigenvalues of the cavity covariance `Ṽ` (`λₖ ≥ 0`).
- `c::Vector{T}`: cavity residual `y − m̃` rotated into the eigenbasis, `Qᵀ(y − m̃)`.
"""
struct MvNormalPrecisionMessage{T<:Real} <: ClosedFormExpectations.Expression
    λ::Vector{T}
    c::Vector{T}
end

function MvNormalPrecisionMessage(y::AbstractVector, m̃::AbstractVector, Ṽ::AbstractMatrix)
    F = eigen(Symmetric(Matrix(Ṽ)))
    λ = float.(F.values)
    c = F.vectors' * float.(y .- m̃)
    T = promote_type(eltype(λ), eltype(c))
    return MvNormalPrecisionMessage{T}(convert(Vector{T}, λ), convert(Vector{T}, c))
end

# μ_{f→κ}(κ): the message value, i.e. the density 𝒩(y | m̃, Ṽ + κ⁻¹ I).
(p::MvNormalPrecisionMessage)(κ) = exp(log(p, κ))

# log μ_{f→κ}(κ): the exact log-message that is projected onto the Gamma κ edge.
function Base.log(p::MvNormalPrecisionMessage, κ)
    κinv = inv(κ)
    acc = zero(promote_type(eltype(p.λ), typeof(κinv)))
    @inbounds for k in eachindex(p.λ)
        s = p.λ[k] + κinv
        acc += log(2π * s) + p.c[k]^2 / s
    end
    return -acc / 2
end

"""
    MvNormalDeviationPrecisionMessage(ξ, Λ, m_b, V_b)

Exact BP message that the local-deviation gate `𝒩(w | w̄, κ⁻¹ I)` sends toward its
scalar precision `κ` when **both** Gaussian neighbours are latent: the cavity on
the deviating edge `w` arrives in **information form** `exp(ξᵀw − ½ wᵀΛw)` — and
may be rank-deficient (e.g. the rank-1 precision of a structured softdot message),
so it must NOT be converted to moment form — while the cavity on the anchor edge
`w̄` is a proper Gaussian `𝒩(m_b, V_b)`. Integrating both out,

    μ_{f→κ}(κ) ∝ ∫∫ 𝒩(w | w̄, κ⁻¹I) e^{ξᵀw − ½wᵀΛw} 𝒩(w̄ | m_b, V_b) dw dw̄,

gives, with `S = V_b + κ⁻¹I`, `A = Λ + S⁻¹` and `h = ξ + S⁻¹ m_b`,

    log μ_{f→κ}(κ) = -½ [ log|S| + log|A| + m_bᵀ S⁻¹ m_b − hᵀ A⁻¹ h ]

**up to a κ-independent constant** (irrelevant to the tangent projection, which
only uses covariances against the Gamma statistics). The anchor covariance is
eigendecomposed once (`V_b = Q D Qᵀ`), so `S`-terms are O(d) per evaluation; `A`
mixes the two bases and costs one d×d Cholesky per κ (3 evaluations under the
unscented strategy). `A ≻ 0` for any PSD `Λ` since `S⁻¹ ≻ 0`.

# Fields
- `ξ::Vector{T}`, `Λ::Matrix{T}`: information form of the deviating-edge cavity.
- `D::Vector{T}`, `Q::Matrix{T}`: eigendecomposition of the anchor covariance `V_b`.
- `c::Vector{T}`: rotated anchor mean `Qᵀ m_b`.
"""
struct MvNormalDeviationPrecisionMessage{T<:Real} <: ClosedFormExpectations.Expression
    ξ::Vector{T}
    Λ::Matrix{T}
    D::Vector{T}
    Q::Matrix{T}
    c::Vector{T}
end

function MvNormalDeviationPrecisionMessage(ξ::AbstractVector, Λ::AbstractMatrix, m_b::AbstractVector, V_b::AbstractMatrix)
    F = eigen(Symmetric(Matrix(V_b)))
    D = float.(F.values)
    Q = float.(F.vectors)
    c = Q' * float.(m_b)
    T = promote_type(eltype(D), eltype(c), eltype(float.(ξ)))
    return MvNormalDeviationPrecisionMessage{T}(
        convert(Vector{T}, float.(ξ)), convert(Matrix{T}, float.(Λ)),
        convert(Vector{T}, D), convert(Matrix{T}, Q), convert(Vector{T}, c),
    )
end

(p::MvNormalDeviationPrecisionMessage)(κ) = exp(log(p, κ))

function Base.log(p::MvNormalDeviationPrecisionMessage, κ)
    κinv = inv(κ)
    sd = p.D .+ κinv                             # eigenvalues of S = V_b + κ⁻¹I
    Sinv = p.Q * Diagonal(inv.(sd)) * p.Q'
    A = cholesky(Symmetric(p.Λ + Sinv))
    h = p.ξ + p.Q * (p.c ./ sd)
    return -(sum(log, sd) + logdet(A) + sum(abs2(p.c[k]) / sd[k] for k in eachindex(sd)) - dot(h, A \ h)) / 2
end
