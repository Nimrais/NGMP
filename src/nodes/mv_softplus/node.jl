export MvSoftplus

import ExponentialFamily: getnaturalparameters, weightedmean_precision
import BayesBase: mean_cov
import LinearAlgebra: Cholesky, Diagonal, I, Symmetric, cholesky, dot, logdet

"""
    MvSoftplus

Deterministic elementwise positive transform `out = softplus.(in)` with
interfaces `[out, in]`. With a multivariate Gaussian out-edge, the `:in` NGMP
message is a tangent projection of its exact log-message, and for `:out`
`TangentProjection(type = DeltaApproximation)` is likewise an exact-log-message
local quadratic at `mean(q(out))` while `TangentProjection(type = Unscented)`
uses a support-safe moment-matched Gaussian pushforward instead.

With an [`MvInverseSoftplusNormal`](@ref) out-edge both NGMP messages are exact
closed-form in-family sites — the pushforward family shares the Gaussian
natural parameters, so softplus becomes a pure change of coordinates between
the two edges and no approximation is made at this node at all.
"""
struct MvSoftplus end

@node MvSoftplus Deterministic [out, in]

"""
    MvSoftplusForwardMessage(mean, covariance)

Exact forward log-message: the pushforward density of `y = softplus.(x)` for
`x ~ N(mean, covariance)`, including the per-element inverse-Jacobian term
`-∑ₖ log(1 - exp(-yₖ))`. Zero (log = -Inf) off the positive orthant — which is
why, on a Gaussian out-edge, the `:out` NGMP rule sends the moment-matched
Gaussian pushforward instead of tangent-projecting this expression (see
rules/natural_gradient.jl). This density IS the [`MvInverseSoftplusNormal`](@ref)
distribution; constraining the out-edge to that family makes the forward
message exact and in-family.
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

# Analytic derivatives of the exact forward log-message.  They make the
# `DeltaApproximation` a genuine tangent projection: its touching quadratic is
# evaluated at mean(q(out)), rather than solely from the incoming message.  The
# exact pushforward has positive-orthant support, so that expansion point must
# remain strictly positive.
function _mv_softplus_forward_logderivatives(
    message::MvSoftplusForwardMessage,
    y::AbstractVector,
)
    all(>(0), y) || throw(DomainError(
        y,
        "MvSoftplus forward delta projection requires mean(q_out) in the positive orthant",
    ))
    d = length(y)
    x = _inverse_softplus.(y)
    invcov = Matrix(message.chol \ Matrix{eltype(message.mean)}(I, d, d))
    residual_precision = invcov * (x .- message.mean)
    expneg = exp.(-y)
    denominator = .-expm1.(-y)             # 1 - exp(-y), stable near zero
    inverse_slope = inv.(denominator)
    jacobian_term = expneg ./ denominator

    gradient = .-inverse_slope .* residual_precision .- jacobian_term
    hessian = -Diagonal(inverse_slope) * invcov * Diagonal(inverse_slope) +
        Diagonal(inverse_slope .* jacobian_term .* (residual_precision .+ 1))
    return gradient, Matrix((hessian .+ hessian') ./ 2)
end

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

# Analytic derivatives of the exact backward log-message
#   ξᵀ softplus(x) - 1/2 softplus(x)ᵀΛ softplus(x).
function _mv_softplus_backward_logderivatives(
    message::MvSoftplusGaussianBackwardMessage,
    x::AbstractVector,
)
    s = _softplus.(x)
    slope = ifelse.(x .>= 0, inv.(1 .+ exp.(-x)), exp.(x) ./ (1 .+ exp.(x)))
    curvature = slope .* (1 .- slope)
    residual_precision = message.xi .- message.Lambda * s
    gradient = slope .* residual_precision
    hessian = -Diagonal(slope) * message.Lambda * Diagonal(slope) +
        Diagonal(curvature .* residual_precision)
    return gradient, Matrix((hessian .+ hessian') ./ 2)
end
