export TangentProjection, ClosedForm, project

import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf
import ExponentialFamily:
    ExponentialFamilyDistribution,
    NormalMeanVariance,
    UnivariateGaussianDistributionsFamily

struct ClosedForm end

struct TangentProjection{S} end

TangentProjection(::Type{S}) where {S} = TangentProjection{S}()
TangentProjection(::S) where {S} = TangentProjection{S}()
TangentProjection(; type = ClosedForm) = TangentProjection(type)

"""
The stable for the tangent projection interface.
"""
function project end

const _GaussianProjectionPoint = Union{
    UnivariateGaussianDistributionsFamily,
    ExponentialFamilyDistribution{NormalMeanVariance},
}

"""
    project(TangentProjection(type = ClosedForm), q, f::Logpdf)

Project the exact log-message `f` onto the Gaussian tangent space at the
receiving marginal `q` using `ClosedFormExpectations.ClosedWilliamsProduct`.

The result is returned as an `ExponentialFamilyDistribution{NormalMeanVariance}`
with natural parameters `(ξ, -Λ/2)`, representing the Gaussian site

    μ̂(z) ∝ exp(ξ z - 1/2 Λ z^2).

The four-argument `ExponentialFamilyDistribution` constructor intentionally skips
properness checks: projected messages are site factors, so during damping or
intermediate iterations their precision can be non-positive even though the final
belief remains proper.
"""
function project(
    ::TangentProjection{<:ClosedForm},
    q::_GaussianProjectionPoint,
    f::Logpdf,
)
    normal_q = convert(Distributions.Normal, q)
    m = Distributions.mean(normal_q)
    σ = Distributions.std(normal_q)
    ∂μ, ∂σ = ClosedFormExpectations.mean(ClosedWilliamsProduct(), f, normal_q)
    Λ = -∂σ / σ
    ξ = ∂μ + m * Λ
    η = promote(ξ, -Λ / 2)
    return ExponentialFamilyDistribution(NormalMeanVariance, collect(η), nothing, nothing)
end
