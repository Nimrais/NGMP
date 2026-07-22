# Closed-form Wishart KL divergence for the Bethe free-energy scorer.
#
# ReactiveMP scores a Wishart prior node as `kldivergence(q_marginal, prior)`,
# but Distributions.jl has no closed-form KL for (matrix-variate) Wisharts and
# its generic fallback dies in a Monte-Carlo `expectation`. The divergence is
# closed-form inside the exponential family:
#
#     KL(p ‖ q) = ⟨η_p − η_q, ∇A(η_p)⟩ − A(η_p) + A(η_q),
#
# with `logpartition`/`gradlogpartition` supplied by ExponentialFamily.jl.
# The EF construction is UNCHECKED: the checked `convert` path rejects valid
# small-ν parameters, and scoring must not round-trip through checked
# conversions. (Type piracy on `Distributions.kldivergence` — deliberate, the
# method simply does not exist upstream.)

import Distributions: Wishart
import ExponentialFamily:
    WishartFast, gradlogpartition, logpartition, pack_parameters, getnaturalparameters, ExponentialFamilyDistribution
import LinearAlgebra: dot

_wishart_fast(p::WishartFast) = p
function _wishart_fast(p::Wishart)
    ν, S = Distributions.params(p)
    return WishartFast(ν, Matrix(inv(S)))
end

function _wishart_ef_unchecked(p::WishartFast)
    d = size(p.invS, 1)
    η = pack_parameters(WishartFast, ((p.ν - d - 1) / 2, -p.invS ./ 2))
    return ExponentialFamilyDistribution(WishartFast, η, nothing, nothing)
end

const _AnyWishart = Union{Wishart, WishartFast}

function Distributions.kldivergence(p::_AnyWishart, q::_AnyWishart)
    efp = _wishart_ef_unchecked(_wishart_fast(p))
    efq = _wishart_ef_unchecked(_wishart_fast(q))
    ηp = getnaturalparameters(efp)
    ηq = getnaturalparameters(efq)
    return dot(ηp .- ηq, gradlogpartition(efp)) - logpartition(efp) + logpartition(efq)
end

# GammaShapeRate priors passed as distribution INSTANCES (`γ ~ priors[:γ]`)
# become StandaloneDistributionNodes scored with `kldivergence(marginal, prior)`.
# Distributions.jl has the closed form only for its own `Gamma` (= shape/scale)
# type; a GammaShapeRate on either side falls into the slow quadgk fallback.
# Route through the closed form instead.
import ExponentialFamily: GammaShapeRate

_gamma_shape_scale(p::GammaShapeRate) = Gamma(shape(p), 1 / rate(p))

Distributions.kldivergence(p::GammaShapeRate, q::GammaShapeRate) =
    Distributions.kldivergence(_gamma_shape_scale(p), _gamma_shape_scale(q))
Distributions.kldivergence(p::GammaShapeRate, q::Gamma) =
    Distributions.kldivergence(_gamma_shape_scale(p), q)
Distributions.kldivergence(p::Gamma, q::GammaShapeRate) =
    Distributions.kldivergence(p, _gamma_shape_scale(q))
