export MvInverseSoftplusNormal

import ExponentialFamily
import ExponentialFamily:
    ExponentialFamilyDistribution,
    MvNormalMeanCovariance,
    NaturalParametersSpace,
    DefaultParametersSpace,
    MeanToNatural,
    NaturalToMean,
    getnaturalparameters,
    getbasemeasure,
    getlogbasemeasure,
    getsufficientstatistics,
    getlogpartition,
    getgradlogpartition,
    getfisherinformation,
    isproper,
    isbasemeasureconstant,
    NonConstantBaseMeasure,
    unpack_parameters,
    exponential_family_typetag
import BayesBase
import BayesBase: PreserveTypeProd
import Distributions
import Random
import LinearAlgebra: Diagonal, Symmetric, cholesky, diag, dot, logdet

"""
    MvInverseSoftplusNormal(μ, Σ)

The pushforward of `x ~ MvNormal(μ, Σ)` through the elementwise softplus,
`y = softplus.(x)` — named, like `LogNormal`, after the inverse map that
Gaussianizes it: `invsoftplus.(y) ~ MvNormal(μ, Σ)`. Support is the positive
orthant.

It is a genuine exponential family sharing the multivariate Gaussian natural
structure: with `u = invsoftplus.(y)`,

    T(y) = (u, u uᵀ),   η = (Σ⁻¹μ, -Σ⁻¹/2),
    h(y) = (2π)^(-d/2) ∏ₖ 1/(1 - e^(-yₖ)),

so the log-partition, its gradient and the Fisher information are *identical*
to `MvNormalMeanCovariance`'s (they depend on η only) and are delegated, never
re-derived. Unlike `LogNormal` the base measure is non-constant — the inverse
Jacobian cannot be absorbed into the sufficient statistics — so products must
not use the constant-base-measure fast path (explicit `prod` methods below).

The mean and covariance have no closed form; they are the exact moments of a
well-defined density, computed with the same degree-5 Gaussian cubature the
tangent projections use. `median` is exact: `softplus.(μ)`.
"""
struct MvInverseSoftplusNormal{T <: Real} <: Distributions.ContinuousMultivariateDistribution
    μ::Vector{T}
    Σ::Matrix{T}
end

function MvInverseSoftplusNormal(μ::AbstractVector, Σ::AbstractMatrix)
    length(μ) == size(Σ, 1) == size(Σ, 2) || throw(DimensionMismatch(
        "MvInverseSoftplusNormal requires length(μ) == size(Σ, 1) == size(Σ, 2); " *
        "got length(μ) = $(length(μ)), size(Σ) = $(size(Σ))",
    ))
    T = promote_type(float(eltype(μ)), float(eltype(Σ)))
    return MvInverseSoftplusNormal{T}(convert(Vector{T}, μ), convert(Matrix{T}, Σ))
end

Base.length(dist::MvInverseSoftplusNormal) = length(dist.μ)
Base.eltype(::MvInverseSoftplusNormal{T}) where {T} = T
Distributions.params(dist::MvInverseSoftplusNormal) = (dist.μ, dist.Σ)

Distributions.insupport(dist::MvInverseSoftplusNormal, y::AbstractVector) =
    length(y) == length(dist.μ) && all(>(zero(eltype(y))), y)

function Distributions._logpdf(dist::MvInverseSoftplusNormal, y::AbstractVector{<:Real})
    T = promote_type(eltype(dist.μ), float(eltype(y)))
    Distributions.insupport(dist, y) || return T(-Inf)
    x = _inverse_softplus.(y)
    r = x .- dist.μ
    C = cholesky(Symmetric(dist.Σ))
    lognormal = -(length(y) * log(2π) + logdet(C) + dot(r, C \ r)) / 2
    loginversejacobian = -sum(yk -> log(-expm1(-yk)), y)
    return lognormal + loginversejacobian
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    dist::MvInverseSoftplusNormal,
    y::AbstractVector{<:Real},
)
    L = cholesky(Symmetric(dist.Σ)).L
    y .= _softplus.(dist.μ .+ L * randn(rng, length(dist.μ)))
    return y
end

Distributions.median(dist::MvInverseSoftplusNormal) = _softplus.(dist.μ)

"""
    _inverse_softplus_normal_latent(q) -> (μ, Σ)

Parameters of the latent Gaussian of an `MvInverseSoftplusNormal` belief, from
either the distribution form or an (unchecked) natural-parameter site. This is
the bridge every tangent projection uses: an `MvInverseSoftplusNormal` edge is
a Gaussian edge in `x = invsoftplus.(y)` coordinates.
"""
_inverse_softplus_normal_latent(dist::MvInverseSoftplusNormal) = (dist.μ, dist.Σ)

function _inverse_softplus_normal_latent(
    ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
)
    η₁, η₂ = unpack_parameters(MvInverseSoftplusNormal, getnaturalparameters(ef))
    C = cholesky(Symmetric(Matrix(-2 .* η₂)))
    Σ = Matrix(Symmetric(inv(C)))
    return Σ * η₁, Σ
end

function _inverse_softplus_normal_moments(μ::AbstractVector, Σ::AbstractMatrix)
    x, w = UnscentedTransforms.degree5_cubature_points(
        collect(float.(μ)), Matrix(float.(Σ)),
    )
    s = map(xk -> _softplus.(xk), x)
    m = sum(w .* s)
    C = sum(w[k] .* ((s[k] .- m) * (s[k] .- m)') for k in eachindex(s))
    return m, Matrix((C .+ C') ./ 2)
end

BayesBase.mean_cov(dist::MvInverseSoftplusNormal) =
    _inverse_softplus_normal_moments(dist.μ, dist.Σ)
BayesBase.mean_cov(ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal}) =
    _inverse_softplus_normal_moments(_inverse_softplus_normal_latent(ef)...)

const _MvISNPoint = Union{
    MvInverseSoftplusNormal,
    ExponentialFamilyDistribution{MvInverseSoftplusNormal},
}

Distributions.mean(q::_MvISNPoint) = first(BayesBase.mean_cov(q))
Distributions.cov(q::_MvISNPoint) = last(BayesBase.mean_cov(q))
Distributions.var(q::_MvISNPoint) = diag(Distributions.cov(q))

"""
    entropy(q::MvInverseSoftplusNormal)

Differential entropy: `H[y] = H[x] + E[log σ(x)]` with `σ` the logistic slope
of softplus and `log σ(x) = -softplus(-x)`, the expectation taken by the same
degree-5 cubature as the moments.
"""
function Distributions.entropy(q::_MvISNPoint)
    μ, Σ = _inverse_softplus_normal_latent(q)
    d = length(μ)
    C = cholesky(Symmetric(Matrix(float.(Σ))))
    gaussian_entropy = (d * (1 + log(2π)) + logdet(C)) / 2
    x, w = UnscentedTransforms.degree5_cubature_points(collect(float.(μ)), Matrix(float.(Σ)))
    jacobian_term = -sum(w[k] * sum(xi -> _softplus(-xi), x[k]) for k in eachindex(x))
    return gaussian_entropy + jacobian_term
end

# ---------------------------------------------------------------------------
# ExponentialFamily.jl type-backed interface. Everything that depends only on
# the natural parameters is delegated to `MvNormalMeanCovariance`; only the
# sufficient statistics, base measure, and support are softplus-specific.
# ---------------------------------------------------------------------------

exponential_family_typetag(::MvInverseSoftplusNormal) = MvInverseSoftplusNormal

isproper(space::NaturalParametersSpace, ::Type{MvInverseSoftplusNormal}, η, conditioner) =
    isproper(space, MvNormalMeanCovariance, η, conditioner)
isproper(space::DefaultParametersSpace, ::Type{MvInverseSoftplusNormal}, θ, conditioner) =
    isproper(space, MvNormalMeanCovariance, θ, conditioner)

(::MeanToNatural{MvInverseSoftplusNormal})(tuple_of_θ::Tuple{Any, Any}) =
    (MeanToNatural{MvNormalMeanCovariance}())(tuple_of_θ)
(::NaturalToMean{MvInverseSoftplusNormal})(tuple_of_η::Tuple{Any, Any}) =
    (NaturalToMean{MvNormalMeanCovariance}())(tuple_of_η)

unpack_parameters(::Type{MvInverseSoftplusNormal}, packed) =
    unpack_parameters(MvNormalMeanCovariance, packed)

isbasemeasureconstant(::Type{MvInverseSoftplusNormal}) = NonConstantBaseMeasure()

getlogbasemeasure(::Type{MvInverseSoftplusNormal}) =
    (y) -> -(length(y) * log(2π)) / 2 - sum(yk -> log(-expm1(-yk)), y)
getbasemeasure(::Type{MvInverseSoftplusNormal}) =
    (y) -> exp(getlogbasemeasure(MvInverseSoftplusNormal)(y))

getsufficientstatistics(::Type{MvInverseSoftplusNormal}) = (
    (y) -> _inverse_softplus.(y),
    (y) -> begin
        u = _inverse_softplus.(y)
        return u * u'
    end,
)

getlogpartition(space::NaturalParametersSpace, ::Type{MvInverseSoftplusNormal}) =
    getlogpartition(space, MvNormalMeanCovariance)
getgradlogpartition(space::NaturalParametersSpace, ::Type{MvInverseSoftplusNormal}) =
    getgradlogpartition(space, MvNormalMeanCovariance)
getfisherinformation(space::NaturalParametersSpace, ::Type{MvInverseSoftplusNormal}) =
    getfisherinformation(space, MvNormalMeanCovariance)

_mvisn_naturals_dim(ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal}) =
    div(isqrt(1 + 4 * length(getnaturalparameters(ef))) - 1, 2)

BayesBase.insupport(ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal}, y) = false
BayesBase.insupport(
    ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
    y::AbstractVector,
) = length(y) == _mvisn_naturals_dim(ef) && all(>(zero(eltype(y))), y)

BayesBase.vague(::Type{MvInverseSoftplusNormal}, dims::Int) =
    MvInverseSoftplusNormal(zeros(dims), Matrix(Diagonal(fill(1e12, dims))))

# ---------------------------------------------------------------------------
# Products. The generic same-family ExponentialFamilyDistribution product is
# explicitly unimplemented for non-constant base measures, and the message
# semantics here is exponential-tilt addition of natural parameters — one base
# measure per edge, never squared. Sites may be improper mid-iteration, so the
# natural-parameter path always uses the unchecked 4-argument constructor.
# ---------------------------------------------------------------------------

BayesBase.default_prod_rule(
    ::Type{<:MvInverseSoftplusNormal},
    ::Type{<:MvInverseSoftplusNormal},
) = PreserveTypeProd(Distributions.Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distributions.Distribution},
    left::MvInverseSoftplusNormal,
    right::MvInverseSoftplusNormal,
)
    Cl = cholesky(Symmetric(left.Σ))
    Cr = cholesky(Symmetric(right.Σ))
    Λ = Matrix(Symmetric(inv(Cl) .+ inv(Cr)))
    ξ = (Cl \ left.μ) .+ (Cr \ right.μ)
    C = cholesky(Symmetric(Λ))
    Σ = Matrix(Symmetric(inv(C)))
    return MvInverseSoftplusNormal(C \ ξ, Σ)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
    right::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
)
    return ExponentialFamilyDistribution(
        MvInverseSoftplusNormal,
        getnaturalparameters(left) .+ getnaturalparameters(right),
        nothing,
        nothing,
    )
end

_mvisn_site(dist::MvInverseSoftplusNormal) = ExponentialFamilyDistribution(
    MvInverseSoftplusNormal,
    getnaturalparameters(convert(ExponentialFamilyDistribution, dist)),
    nothing,
    nothing,
)
_mvisn_site(ef::ExponentialFamilyDistribution{MvInverseSoftplusNormal}) = ef

BayesBase.default_prod_rule(
    ::Type{<:MvInverseSoftplusNormal},
    ::Type{<:ExponentialFamilyDistribution{MvInverseSoftplusNormal}},
) = PreserveTypeProd(ExponentialFamilyDistribution)
BayesBase.default_prod_rule(
    ::Type{<:ExponentialFamilyDistribution{MvInverseSoftplusNormal}},
    ::Type{<:MvInverseSoftplusNormal},
) = PreserveTypeProd(ExponentialFamilyDistribution)

BayesBase.prod(
    strategy::PreserveTypeProd{ExponentialFamilyDistribution},
    left::MvInverseSoftplusNormal,
    right::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
) = BayesBase.prod(strategy, _mvisn_site(left), right)
BayesBase.prod(
    strategy::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{MvInverseSoftplusNormal},
    right::MvInverseSoftplusNormal,
) = BayesBase.prod(strategy, left, _mvisn_site(right))

# Family-generic NGMP damping bridge: reconstruct the (possibly improper)
# message from its natural parameters without checked conversions.
NaturalGradientMP.from_natural(::Type{MvInverseSoftplusNormal}, η) =
    ExponentialFamilyDistribution(MvInverseSoftplusNormal, collect(Float64, η), nothing, nothing)
