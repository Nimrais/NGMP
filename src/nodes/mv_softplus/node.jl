export MvSoftplus

import ExponentialFamily: getnaturalparameters, weightedmean_precision
import BayesBase: mean_cov
import LinearAlgebra: Cholesky, Symmetric, cholesky, dot, logdet

"""
    MvSoftplus

Deterministic elementwise positive transform `out = softplus.(in)` with
interfaces `[out, in]`; both edges are multivariate Gaussian. Messages in both
directions are non-Gaussian, so the NGMP rules project them onto the receiving
edge's Gaussian tangent space (`TangentProjection(type = Unscented)`).
"""
struct MvSoftplus end

@node MvSoftplus Deterministic [out, in]

"""
    MvSoftplusForwardMessage(mean, covariance)

Exact forward log-message: the pushforward density of `y = softplus.(x)` for
`x ~ N(mean, covariance)`, including the per-element inverse-Jacobian term
`-∑ₖ log(1 - exp(-yₖ))`. Zero (log = -Inf) off the positive orthant — which is
why the `:out` NGMP rule sends the moment-matched Gaussian pushforward instead
of tangent-projecting this expression (see rules/natural_gradient.jl); the
exact expression is kept as the reference density.
"""
struct MvSoftplusForwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    mean::Vector{T}
    chol::Cholesky{T, Matrix{T}}
    logdet_cov::T
end

function MvSoftplusForwardMessage(mean::AbstractVector, covariance::AbstractMatrix)
    C = cholesky(Symmetric(Matrix(float.(covariance))))
    m = collect(float.(mean))
    return MvSoftplusForwardMessage{eltype(m)}(m, C, logdet(C))
end

function Base.log(message::MvSoftplusForwardMessage, y::AbstractVector)
    T = float(eltype(y))
    all(>(zero(T)), y) || return T(-Inf)
    x = _inverse_softplus.(y)
    r = x .- message.mean
    lognormal = -(length(y) * log(2π) + message.logdet_cov + dot(r, message.chol \ r)) / 2
    loginversejacobian = -sum(yk -> log(-expm1(-yk)), y)
    return lognormal + loginversejacobian
end

(message::MvSoftplusForwardMessage)(y::AbstractVector) = exp(log(message, y))

"""
    MvSoftplusGaussianBackwardMessage(ξ, Λ)

Exact backward log-message through the deterministic transform when the
outgoing cavity is Gaussian, kept in INFORMATION form:

    log μ(x) = ξᵀ s(x) - ½ s(x)ᵀ Λ s(x) + const,   s(x) = softplus.(x).

The x-independent normalizer drops out of every tangent-projection covariance,
and the (ξ, Λ) form never inverts the cavity — `ContinuousTransition`'s `:x`
rule emits `MvNormalWeightedMeanPrecision` whose precision may be singular.
"""
struct MvSoftplusGaussianBackwardMessage{T <: Real} <: ClosedFormExpectations.Expression
    xi::Vector{T}
    Lambda::Matrix{T}
end

function MvSoftplusGaussianBackwardMessage(xi::AbstractVector, Lambda::AbstractMatrix)
    ξ = collect(float.(xi))
    return MvSoftplusGaussianBackwardMessage{eltype(ξ)}(ξ, Matrix(float.(Lambda)))
end

function Base.log(message::MvSoftplusGaussianBackwardMessage, x::AbstractVector)
    s = _softplus.(x)
    return dot(message.xi, s) - dot(s, message.Lambda * s) / 2
end

(message::MvSoftplusGaussianBackwardMessage)(x::AbstractVector) = exp(log(message, x))
