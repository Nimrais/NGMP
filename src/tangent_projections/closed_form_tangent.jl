export TangentProjection, ClosedForm, Quadrature, project

import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf
import ExponentialFamily:
    ExponentialFamilyDistribution,
    NormalMeanVariance,
    UnivariateGaussianDistributionsFamily

struct ClosedForm end

"""
    Quadrature(n = 64)

Tangent-projection strategy that computes the Williams product `∇_η E_q[ℓ] =
Cov_q[T, ℓ]` **exactly (to quadrature precision)** with a Gaussian quadrature
matched to the receiving family — generalized Gauss–Laguerre on a Gamma edge,
Gauss–Hermite on a Gaussian edge — instead of a closed form (unavailable for
e.g. `NormalPrecisionMessage`) or the second-order delta expansion (biased when
the receiving marginal is wide: the touching quadratic of `ℓ` is integrated far
from the expansion point). The node count `n` lives in the type domain so it
survives the `TangentProjection(type = Quadrature(64))` constructor.
"""
struct Quadrature{N} end

Quadrature(n::Int = 64) = Quadrature{n}()

struct TangentProjection{S} end

TangentProjection(::Type{S}) where {S} = TangentProjection{S}()
TangentProjection(::S) where {S} = TangentProjection{S}()
TangentProjection(; type = ClosedForm) = TangentProjection(type)

# Uniform evaluation of a log-message at a point: SurrogateModelling expressions
# (`NormalPrecisionMessage`, `StudentTMessage`, ...) subtype
# `ClosedFormExpectations.Expression` and define `Base.log(p, x)`; everything else
# (CFE / Distributions.jl distributions such as `LogGamma`, `LogNormal`) goes
# through `logpdf`.
_eval_logmessage(f::Logpdf{<:ClosedFormExpectations.Expression}, x) = log(f.dist, x)
_eval_logmessage(f::Logpdf, x) = Distributions.logpdf(f.dist, x)

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
