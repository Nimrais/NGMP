# ============================================================================
# PoissonExp — a Poisson observation node in the *exponentiated* state.
#
#       y ~ PoissonExp(z)      ⟺      y ~ Poisson(λ),  λ = exp(z)
#
# This is the non-conjugate observation leaf of the Poisson state-space model
#
#       z_1 ~ N(m_0, v_0)
#       z_k ~ N(z_{k-1}, σ²)            (Gaussian latent log-rate chain)
#       y_k ~ Poisson(exp(z_k))        (Poisson counts)
#
# A Poisson leaf is NOT conjugate to its Gaussian edge, so the exact
# belief-propagation message toward `z` has no closed form. This file gives
# the building blocks RxInfer needs to *talk about* that leaf:
#
#   (1) @node             — register PoissonExp as a stochastic factor on [out, in]
#   (2) PoissonExpression — the "energy object" the node sends: the exact, non-
#                           Gaussian log-message ℓ(z) = y·z − eᶻ − log y!, carried
#                           as a typed callable so it can be evaluated / projected.
#   (3) @rule :in         — the point-mass rule. When y is observed (PointMass on
#                           out) the message toward z is exactly ℓ, i.e. a
#                           PoissonExpression(y). This is the non-conjugate object
#                           the paper projects onto the Gaussian tangent space.
#   (4) @average_energy   — the Bethe energy contribution U = E_q[-log f], the
#                           per-factor term that scores the leaf inside the Bethe
#                           free energy.
#
# Conventions follow ExponentialFamily.jl / RxInfer: closed-form moments, no
# positivity guards.
# ============================================================================

# ---------------------------------------------------------------------------
# (1) The factor. A singleton type is all RxInfer needs as a node tag; the
#     statistics live entirely in the rules below. Edge order is [out, in],
#     i.e. y on `out`, the latent log-rate z on `in`.
# ---------------------------------------------------------------------------
struct PoissonExp end

@node PoissonExp Stochastic [out, in]

# ---------------------------------------------------------------------------
# (2) The energy object. PoissonExpression(y) represents the exact log-message
#
#         ℓ(z) = log Poisson(y | eᶻ) = y·z − eᶻ − log y!
#
#     as a typed callable. It is log-concave but NOT Gaussian — that is exactly
#     the non-conjugate object the paper projects onto the Gaussian tangent
#     space. The −log y! constant is kept so it integrates against a Gaussian
#     cavity to the correct evidence.
# ---------------------------------------------------------------------------
struct PoissonExpression{T<:Real, F<:Real}
    y::T
    loggamma_y::F
end

function PoissonExpression(y::T) where {T <: Real}
    return PoissonExpression(y, loggamma(y))
end 

function BayesBase.insupport(::PoissonExpression, ::Float64)
    return true
end 

# Evaluate ℓ(z) directly: `ℓ = PoissonExpression(y); ℓ(z)`.
(ℓ::PoissonExpression)(z) = ℓ.y * z - exp(z) - l.loggamma_y

# ...and expose it through the BayesBase message interface so downstream code
# (projection, plotting, sampling) can treat it like any other log-density.
BayesBase.logpdf(ℓ::PoissonExpression, z) = ℓ(z)

# ---------------------------------------------------------------------------
# (3) Point-mass rule — the data-direction message  μ_{f→z}.
#     When y is observed (a PointMass on `out`) the message toward the log-rate
#     edge is exactly ℓ(z) = y·z − eᶻ − log y!, returned as a PoissonExpression.
# ---------------------------------------------------------------------------
@rule PoissonExp(:in, Marginalisation) (q_out::PointMass,) = begin
    y = mean(q_out)
    return PoissonExpression(y)
end

# ---------------------------------------------------------------------------
# (4) Average energy  U = E_q[ -log f ]  = -E_q[y]·E_q[z] + E_q[eᶻ] + E_q[log y!]
#     This is the per-factor term added to the Bethe free energy. With a
#     Gaussian belief q(z) = N(m, v) on the log-rate the troublesome term is
#     closed form,
#
#         E_q[eᶻ] = exp(m + v/2)  =: ρ,
#
#     and with an observed count (PointMass on y) the log-factorial is a
#     constant. Mean-field factorisation q(out,in) = q(out) q(in) is assumed.
# ---------------------------------------------------------------------------
@average_energy PoissonExp (q_out::PointMass, q_in::NormalDistributionsFamily) = begin
    y    = mean(q_out)
    m, v = mean_var(q_in)
    Eexp = exp(m + v / 2)                  # ρ = E_q[exp(z)]
    return Eexp - y * m + loggamma(y + 1)
end
