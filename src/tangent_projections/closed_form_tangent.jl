export TangentProjection, ClosedForm, DeltaApproximation, Quadrature, project, resolve_projection

import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf
import ExponentialFamily:
    ExponentialFamilyDistribution,
    NormalMeanVariance,
    UnivariateGaussianDistributionsFamily

"""
    ClosedForm

Tangent-projection strategy computing the Williams product with an **exact**
`ClosedFormExpectations.ClosedWilliamsProduct` — available only for messages
whose expectations against the receiving family are closed-form (the Poisson
`LogGamma`, the `Log` node's `LogGamma`/`LogNormal`, ...). For messages WITHOUT
a closed-form product (`StudentTMessage`, `NormalPrecisionMessage`) this
strategy raises an informative error — pick an approximation explicitly:
[`DeltaApproximation`](@ref) (2 analytic derivatives, biased for wide marginals),
`Unscented` (3 sigma points), or [`Quadrature`](@ref) (exact to quadrature
precision).
"""
struct ClosedForm end

"""
    DeltaApproximation

Tangent-projection strategy computing the Williams product from the
**second-order (delta-method) expansion** of the log-message at the mean of the
receiving marginal: 2 analytic derivative evaluations per message
(`project_to_gamma` / `project_to_normal` with the `DerivativeEnhancedFunction`
bundles). Exact when ℓ is quadratic in the sufficient statistics and cheap
everywhere — but **biased when the receiving marginal is wide** (the touching
quadratic is integrated far from the expansion point); prefer `Unscented` or
`Quadrature` in that regime. Available for the messages with registered analytic
derivatives: `NormalPrecisionMessage` (Gamma edge), `StudentTMessage`
(Gaussian edge).
"""
struct DeltaApproximation end

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

"""
    resolve_projection(projection) -> TangentProjection

Resolve the `projection` field of a [`NaturalGradientMessage`](@ref) inside a
rule body: the [`NaturalGradientMP.ClosedFormDefault`](@ref) sentinel (the
submodule cannot see the strategy types defined here) becomes
`TangentProjection(type = ClosedForm)`; anything else passes through unchanged.
"""
resolve_projection(::NaturalGradientMP.ClosedFormDefault) = TangentProjection(type = ClosedForm)
resolve_projection(projection) = projection

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
