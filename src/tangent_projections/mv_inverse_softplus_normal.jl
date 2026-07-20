# MvInverseSoftplusNormal tangent projections: the family shares the Gaussian
# natural parameters over the sufficient statistics T(y) = (u, u uᵀ) with
# u = invsoftplus.(y), so the Williams product against an
# MvInverseSoftplusNormal belief is EXACTLY the multivariate Gaussian Williams
# product in latent coordinates x = invsoftplus.(y):
#
#   Cov_q[T(y), ℓ(y)] = Cov_{N(μ,Σ)}[(x, x xᵀ), ℓ(softplus.(x))],
#
# and the two families' Fisher matrices coincide. Every projection below
# therefore pulls the log-message back through softplus and delegates to the
# Gaussian-edge machinery in `mv_normal.jl`, then re-tags the natural
# parameters as an MvInverseSoftplusNormal site.

import ClosedFormExpectations: Logpdf
import ExponentialFamily: weightedmean_precision

const _MvISNProjectionPoint = _MvISNPoint

"""
    SoftplusPullback(logmessage::Logpdf)

The pullback `x ↦ ℓ(softplus.(x))` of a positive-orthant log-message onto the
latent Gaussian coordinates of an `MvInverseSoftplusNormal` edge. Wrapping any
`Logpdf` makes the generic Gaussian cubature projection applicable; for
Gaussian log-messages the pullback is instead constructed analytically as an
[`MvSoftplusGaussianBackwardMessage`](@ref), which also carries derivatives
for the delta projection.
"""
struct SoftplusPullback{F <: Logpdf} <: ClosedFormExpectations.Expression
    logmessage::F
end

Base.log(pullback::SoftplusPullback, x::AbstractVector) =
    _eval_logmessage(pullback.logmessage, _softplus.(x))
(pullback::SoftplusPullback)(x::AbstractVector) = exp(log(pullback, x))

_mvisn_retag(site) = ExponentialFamilyDistribution(
    MvInverseSoftplusNormal,
    collect(Float64, getnaturalparameters(site)),
    nothing,
    nothing,
)

_mvisn_latent_point(q::_MvISNProjectionPoint) =
    MvNormalMeanCovariance(_inverse_softplus_normal_latent(q)...)

# Analytic pullback of a Gaussian log-message ξᵀy − ½ yᵀΛy: substituting
# y = softplus.(x) gives exactly the exact backward expression of the
# MvSoftplus node, derivatives included.
function _mvisn_gaussian_pullback(f::Logpdf{<:MultivariateNormalDistributionsFamily})
    ξ, Λ = weightedmean_precision(f.dist)
    return Logpdf(MvSoftplusGaussianBackwardMessage(ξ, Λ))
end

# Cubature projection of an arbitrary positive-orthant log-message.
function project(
    strategy::TangentProjection{U},
    q::_MvISNProjectionPoint,
    f::Logpdf,
) where {U <: UnscentedTransform}
    site = project(strategy, _mvisn_latent_point(q), Logpdf(SoftplusPullback(f)))
    return _mvisn_retag(site)
end

# Gaussian log-message specializations: use the analytic pullback (identical
# values, but the delta path additionally needs its exact derivatives).
function project(
    strategy::TangentProjection{U},
    q::_MvISNProjectionPoint,
    f::Logpdf{<:MultivariateNormalDistributionsFamily},
) where {U <: UnscentedTransform}
    site = project(strategy, _mvisn_latent_point(q), _mvisn_gaussian_pullback(f))
    return _mvisn_retag(site)
end

function project(
    strategy::TangentProjection{<:DeltaApproximation},
    q::_MvISNProjectionPoint,
    f::Logpdf{<:MultivariateNormalDistributionsFamily},
)
    site = project(strategy, _mvisn_latent_point(q), _mvisn_gaussian_pullback(f))
    return _mvisn_retag(site)
end

function project(::TangentProjection{<:DeltaApproximation}, ::_MvISNProjectionPoint, ::Logpdf)
    return error(
        "The delta projection onto an MvInverseSoftplusNormal belief requires analytic ",
        "pullback derivatives, provided only for multivariate Gaussian log-messages. ",
        "Use `TangentProjection(type = Unscented)` for other log-messages.",
    )
end

function project(::TangentProjection{<:ClosedForm}, ::_MvISNProjectionPoint, ::Logpdf)
    return error(
        "No closed-form Williams product against an MvInverseSoftplusNormal belief. ",
        "Use `TangentProjection(type = Unscented)` (degree-5 cubature in latent space).",
    )
end

function project(::TangentProjection{<:Quadrature}, ::_MvISNProjectionPoint, ::Logpdf)
    return error(
        "Tensor-product quadrature is not implemented for MvInverseSoftplusNormal edges. ",
        "Use `TangentProjection(type = Unscented)` (degree-5 cubature in latent space).",
    )
end
