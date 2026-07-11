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
only uses covariances against the Gamma statistics).

The constructor exploits the low rank `r` of the cavity precision (`r = 1` for a
structured softdot message): `Λ = U diag(λr) Uᵀ` and everything is rotated once
into the anchor eigenbasis `V_b = Q D Qᵀ`, where `S` is diagonal. Each κ
evaluation then costs O(d·r + r³) with only O(d)-sized allocations, via the
matrix determinant lemma and the Woodbury identity:

    log|A| = -Σₖ log sₖ + log|I_r + diag(λr) ŨᵀSŨ|,          sₖ = Dₖ + κ⁻¹,
    hᵀA⁻¹h = h̃ᵀSh̃ − (ŨᵀSh̃)ᵀ (diag(λr)⁻¹ + ŨᵀSŨ)⁻¹ (ŨᵀSh̃).

This matters at scale: the gate rule evaluates the message at 3 unscented points
per node per iteration, so a dense O(d³)-with-O(d²)-allocations evaluation
produces hundreds of GB of GC churn on the full dataset.

# Fields
- `D::Vector{T}`: eigenvalues of the anchor covariance `V_b`.
- `c::Vector{T}`: rotated anchor mean `Qᵀ m_b`.
- `ξ̃::Vector{T}`: rotated cavity weighted mean `Qᵀ ξ`.
- `Ũ::Matrix{T}`: rotated cavity precision eigenvectors `Qᵀ U` (d×r).
- `λr::Vector{T}`: the `r` retained cavity precision eigenvalues.
"""
struct MvNormalDeviationPrecisionMessage{T<:Real} <: ClosedFormExpectations.Expression
    D::Vector{T}
    c::Vector{T}
    ξ̃::Vector{T}
    Ũ::Matrix{T}
    λr::Vector{T}
end

function MvNormalDeviationPrecisionMessage(ξ::AbstractVector, Λ::AbstractMatrix, m_b::AbstractVector, V_b::AbstractMatrix)
    Fb = eigen(Symmetric(Matrix(V_b)))
    FΛ = eigen(Symmetric(Matrix(Λ)))
    λmax = maximum(FΛ.values)
    # √eps relative truncation: LAPACK's numerical zeros are O(d·eps·λmax), and a
    # genuine direction this small contributes negligibly to the message anyway
    tol = (λmax > 0 ? float(λmax) : one(float(λmax))) * sqrt(eps(one(float(λmax))))
    keep = findall(>(tol), FΛ.values)
    Q = float.(Fb.vectors)
    D = float.(Fb.values)
    c = Q' * float.(m_b)
    ξ̃ = Q' * float.(ξ)
    Ũ = Q' * float.(FΛ.vectors[:, keep])
    λr = float.(FΛ.values[keep])
    T = promote_type(eltype(D), eltype(c), eltype(ξ̃), eltype(λr))
    return MvNormalDeviationPrecisionMessage{T}(
        convert(Vector{T}, D), convert(Vector{T}, c), convert(Vector{T}, ξ̃),
        convert(Matrix{T}, Ũ), convert(Vector{T}, λr),
    )
end

(p::MvNormalDeviationPrecisionMessage)(κ) = exp(log(p, κ))

function Base.log(p::MvNormalDeviationPrecisionMessage, κ)
    κinv = inv(κ)
    T = promote_type(eltype(p.D), typeof(κinv))
    d, r = length(p.D), length(p.λr)
    logdetS = zero(T)
    mSm = zero(T)
    hSh = zero(T)
    w = zeros(T, r)                      # ŨᵀSh̃
    G = zeros(T, r, r)                   # ŨᵀSŨ
    @inbounds for k in 1:d
        s = p.D[k] + κinv
        logdetS += log(s)
        ck = p.c[k]
        mSm += ck^2 / s
        h = p.ξ̃[k] + ck / s
        sh = s * h
        hSh += h * sh
        for a in 1:r
            w[a] += p.Ũ[k, a] * sh
            su = s * p.Ũ[k, a]
            for b in a:r
                G[a, b] += su * p.Ũ[k, b]
            end
        end
    end
    logdetA = -logdetS
    quad = zero(T)
    if r == 1
        g = G[1, 1]
        logdetA += log1p(p.λr[1] * g)
        quad = w[1]^2 / (inv(p.λr[1]) + g)
    elseif r > 1
        @inbounds for a in 1:r, b in (a+1):r
            G[b, a] = G[a, b]
        end
        M = cholesky(Symmetric(G + Diagonal(inv.(p.λr))))
        logdetA += logdet(M) + sum(log, p.λr)   # |I + diag(λ)G| = |diag(λ)|·|diag(λ)⁻¹ + G|
        quad = dot(w, M \ w)
    end
    return -(logdetS + logdetA + mSm - (hSh - quad)) / 2   # hᵀA⁻¹h = h̃ᵀSh̃ − wᵀM⁻¹w
end
