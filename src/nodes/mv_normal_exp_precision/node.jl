export MvNormalExpPrecision, MvExpGammaSiteMessage

import BayesBase: mean_cov
import ExponentialFamily:
    MultivariateNormalDistributionsFamily,
    MvNormalMeanPrecision,
    MvNormalMeanCovariance
import LinearAlgebra: Diagonal, diag, dot

# ============================================================================
# MvNormalExpPrecision — a multivariate Gaussian likelihood with a PER-DIMENSION
# precision carried in log-space on a plain Gaussian edge:
#
#       y ~ MvNormalExpPrecision(μ, s)   ⟺   y ~ N(μ, diag(exp.(s))⁻¹)
#
# The fused-Exp generalization of the scalar softdot precision: where
# `y ~ softdot(x, w, τ)` forces ONE τ across all output dimensions
# (homoscedastic), this node gives every dimension its own precision e^{sⱼ}, so
# an upper layer can learn a different uncertainty function per output. Fusing
# the exp link into the node avoids a vector-of-Gammas edge (no such
# distribution exists in ExponentialFamily); `s` is an ordinary MvNormal edge
# that upstream softdot/MvStack/ContinuousTransition machinery can drive.
#
# The diagonal precision factorizes the likelihood over dimensions,
#
#     log f(y, μ, s) = Σⱼ [ ½sⱼ − ½e^{sⱼ}(yⱼ−μⱼ)² ] − (d/2)·log 2π,
#
# so every message is a per-coordinate copy of the scalar Exp-node machinery:
#
#   (1) @node             — register the stochastic factor on [out, μ, s]
#   (2) MvExpGammaSiteMessage — the exact mean-field site toward s: a product of
#                           independent ExpGamma-shaped sites (see below)
#   (3) rules/vmp.jl      — Marginalisation rules toward :out and :μ (Gaussian
#                           with precision diag(E[e^{sⱼ}]), lognormal moment) and
#                           the raw site toward :s
#   (4) rules/natural_gradient.jl — the NGMP rule toward :s with an EXACT
#                           closed-form tangent projection (per-coordinate
#                           Williams product) and damping
#   (5) @average_energy   — the Bethe energy contribution U = E_q[−log f]
#
# FACTORIZATION: the marginal-based rules ((3)-(4) above) require mean-field
# `q(out, μ, s) = q(out)q(μ)q(s)` (y is data in the target models, so q(out) is
# a PointMass automatically). Alternatively, keeping μ and s in ONE cluster and
# naming both in `NGMPDependencies(s = nothing, μ = nothing)` activates the
# CAVITY-based rules in rules/cavity.jl — the true NGMP messages that project
# the exact BP log-messages (with the determinant correction) instead of the
# expected-log-factor sites; those need a quadrature/unscented projection and
# run with `free_energy = false` (no joint (μ, s) marginal rule is provided).
# ============================================================================
"""
MvNormalExpPrecision — multivariate Gaussian with per-dimension log-precision.

      y ~ MvNormalExpPrecision(μ, s)  ⟺   y ~ N(μ, diag(exp.(s))⁻¹)

Heteroscedastic (per-output-dimension) generalization of the scalar softdot
precision. `s` is a plain multivariate Gaussian edge; the exp link is fused
into the node. Requires the mean-field constraint `q(μ)q(s)` and, for the NGMP
site toward `s`, `NGMPDependencies(s = nothing)` with node-level `DampingMeta`.
"""
struct MvNormalExpPrecision end

@node MvNormalExpPrecision Stochastic [out, (μ, aliases = [m]), (s, aliases = [logprecision])]

"""
    MvExpGammaSiteMessage(log_coefficients, rates)

The exact mean-field VMP site toward the log-precision vector `s`:

    log μ(s) = Σⱼ ( cⱼ sⱼ − bⱼ e^{sⱼ} ),

with `cⱼ = 1/2` and `bⱼ = ½·E[(yⱼ − μⱼ)²]` for this node — a product of
independent ExpGamma-shaped sites, the vectorized twin of the scalar
[`ExpGammaSiteMessage`](@ref). Like its scalar twin this is a SITE, not a
distribution: either natural parameter may sit outside the proper ExpGamma
domain, which is why it is projected (closed-form, per coordinate) rather than
normalized.
"""
struct MvExpGammaSiteMessage{T <: Real} <: ClosedFormExpectations.Expression
    log_coefficients::Vector{T}
    rates::Vector{T}
end

function MvExpGammaSiteMessage(log_coefficients::AbstractVector, rates::AbstractVector)
    length(log_coefficients) == length(rates) || throw(DimensionMismatch(
        "MvExpGammaSiteMessage needs matching coefficient/rate lengths, got " *
        "$(length(log_coefficients)) and $(length(rates))",
    ))
    c = collect(float.(log_coefficients))
    b = collect(promote_type(eltype(c), eltype(float.(rates))).(rates))
    return MvExpGammaSiteMessage{eltype(c)}(c, b)
end

function Base.log(message::MvExpGammaSiteMessage, s::AbstractVector)
    return dot(message.log_coefficients, s) - dot(message.rates, exp.(s))
end

(message::MvExpGammaSiteMessage)(s::AbstractVector) = exp(log(message, s))

# BayesBase glue (mirroring PoissonExpression): lets the raw site ride RxInfer's
# generic message product and the `ProjectedTo` marginal path (the VMP arm).
BayesBase.insupport(::MvExpGammaSiteMessage, ::AbstractVector) = true
BayesBase.logpdf(message::MvExpGammaSiteMessage, s::AbstractVector) = log(message, s)

# --- shared numerics ---------------------------------------------------------

# exp overflow guard for the lognormal moment ρⱼ = E[e^{sⱼ}] = e^{mⱼ + Vⱼⱼ/2};
# the half-floatmax budget leaves room for the downstream ρⱼ·Eⱼ products
_mnep_clamp_exponent(a) = min(a, log(floatmax(Float64)) / 2)

_mnep_rho(ms::AbstractVector, vs::AbstractVector) =
    exp.(_mnep_clamp_exponent.(ms .+ vs ./ 2))

# marginal mean and per-coordinate variance of an edge marginal; a PointMass
# (observed data) contributes zero variance
_mnep_mean_vardiag(q::PointMass) = (mean(q), zero(float.(mean(q))))
_mnep_mean_vardiag(q) = begin
    m, V = mean_cov(q)
    return (m, diag(V))
end

# residual floor (cavity-floor idiom): a zero residual would make the site
# log-linear in sⱼ and launch sⱼ → ∞
_mnep_floor_residuals(E::AbstractVector) = max.(E, sqrt(eps(Float64)))
_mnep_floor_residuals(E::Real) = max(E, sqrt(eps(Float64)))

# E[(yⱼ − μⱼ)²] per coordinate under independent q(out), q(μ)
function _mnep_expected_square_residuals(q_out, q_μ)
    my, vy = _mnep_mean_vardiag(q_out)
    mμ, vμ = _mnep_mean_vardiag(q_μ)
    return _mnep_floor_residuals(abs2.(my .- mμ) .+ vy .+ vμ)
end

# --- Bethe average energy ----------------------------------------------------
#
#   U = E_q[−log f] = (d/2)·log 2π − ½ Σⱼ E[sⱼ] + ½ Σⱼ E[e^{sⱼ}]·E[(yⱼ−μⱼ)²]

@average_energy MvNormalExpPrecision (q_out::Any, q_μ::Any, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    E = _mnep_expected_square_residuals(q_out, q_μ)
    return length(ms) * log(2π) / 2 - sum(ms) / 2 + dot(ρ, E) / 2
end
