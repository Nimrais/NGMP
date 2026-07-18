# Multivariate Gaussian tangent projection: the Williams product
# ∇_η E_q[ℓ] = Cov_q[T, ℓ] with T = (x, x xᵀ) approximated with the 2d²+1-point
# degree-5 Gaussian cubature, mapped through the inverse Fisher via the Stein
# identities
#   Cov_q[x, ℓ] = V g,                    g = E_q[∇ℓ]
#   Cov_q[(x−m)(x−m)ᵀ, ℓ] = V H V,        H = E_q[∇²ℓ]
# so the projected site is the canonical Gaussian message
#   μ̂(x) ∝ exp(ξᵀx − ½ xᵀΛx),   Λ = −H,   ξ = g + Λ m.
# The degree-5 rule (with its pair points) is required: an axis-only 2d+1 UT
# set misrepresents the cross fourth moments and gets H wrong even for exactly
# quadratic ℓ. Reduces exactly to the univariate sigma-point route at d = 1
# (identical points, identical Fisher map).

import ExponentialFamily: MvNormalMeanCovariance, MultivariateNormalDistributionsFamily
import BayesBase: mean_cov
import LinearAlgebra: Symmetric, cholesky, vec

const _MvGaussianProjectionPoint = Union{
    MultivariateNormalDistributionsFamily,
    ExponentialFamilyDistribution{MvNormalMeanCovariance},
}

_mv_mean_cov(q::MultivariateNormalDistributionsFamily) = mean_cov(q)
_mv_mean_cov(q::ExponentialFamilyDistribution{MvNormalMeanCovariance}) =
    mean_cov(convert(Distribution, q))

"""
    project(TangentProjection(type = Unscented), q::MultivariateNormalDistributionsFamily, f::Logpdf)

Cubature tangent projection onto a multivariate Gaussian edge: the
McNamee–Stenger degree-5 points at `q = N(m, V)` (2d²+1 points — the pair
points are required for the cross fourth moments in `Cov_q[x xᵀ, ℓ]`),
covariance sums against `T = (x, x xᵀ)` with discrete-mean centering (the same
convention as the scalar rule in `unscented.jl`), then the inverse-Fisher map
via the Stein identities. Exact for quadratic `ℓ`; identical to the scalar
sigma-point rule at `d = 1`. The scaled-UT parameters `α, β, κ` are ignored on
this edge, as on the Gamma edge. Returns the unchecked
`ExponentialFamilyDistribution{MvNormalMeanCovariance}` site with flat natural
parameters `vcat(ξ, vec(-Λ/2))` — improper sites are legitimate mid-iteration
and must never be round-tripped through checked conversions.
"""
function project(::TangentProjection{U}, q::_MvGaussianProjectionPoint, f::Logpdf) where {U <: UnscentedTransform}
    m, V = _mv_mean_cov(q)
    x, w = UnscentedTransforms.degree5_cubature_points(m, V)
    ℓ = map(xk -> _eval_logmessage(f, xk), x)

    Ex = sum(w .* x)
    Exx = sum(w[k] .* (x[k] * x[k]') for k in eachindex(x))
    Eℓ = sum(w .* ℓ)

    c1 = sum(w[k] .* (x[k] .- Ex) .* (ℓ[k] - Eℓ) for k in eachindex(x))               # Cov_q[x, ℓ]
    C2 = sum(w[k] .* (x[k] * x[k]' .- Exx) .* (ℓ[k] - Eℓ) for k in eachindex(x))      # Cov_q[x xᵀ, ℓ]
    C2c = C2 .- Ex * c1' .- c1 * Ex'                                                  # Cov_q[(x−m)(x−m)ᵀ, ℓ]

    F = cholesky(Symmetric(Matrix(float.(V))))
    g = F \ c1
    H = F \ (F \ Symmetric(C2c))'                      # V⁻¹ C2c V⁻¹
    H = (H .+ H') ./ 2

    Λ = -H
    ξ = g .+ Λ * collect(float.(m))
    return ExponentialFamilyDistribution(MvNormalMeanCovariance, vcat(ξ, vec(-Λ ./ 2)), nothing, nothing)
end

function project(::TangentProjection{<:ClosedForm}, ::_MvGaussianProjectionPoint, ::Logpdf)
    return error(
        "No closed-form Williams product against a multivariate Gaussian belief. ",
        "Use `TangentProjection(type = Unscented)` (2d+1 sigma points).",
    )
end

function project(::TangentProjection{<:DeltaApproximation}, ::_MvGaussianProjectionPoint, ::Logpdf)
    return error(
        "No analytic-derivative delta projection for multivariate Gaussian edges. ",
        "Use `TangentProjection(type = Unscented)` (2d+1 sigma points).",
    )
end

function project(::TangentProjection{<:Quadrature}, ::_MvGaussianProjectionPoint, ::Logpdf)
    return error(
        "Tensor-product quadrature is not implemented for multivariate Gaussian edges. ",
        "Use `TangentProjection(type = Unscented)` (2d+1 sigma points).",
    )
end
