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
"""
PoissonExp — a Poisson observation node in the *exponentiated* state.

      y ~ PoissonExp(z)  ⟺   y ~ Poisson(λ),  λ = exp(z)

This is the non-conjugate observation leaf of the Poisson state-space model

      z_1 ~ N(m_0, v_0)
      z_k ~ N(z_{k-1}, σ²)            (Gaussian latent log-rate chain)
      y_k ~ Poisson(exp(z_k))        (Poisson counts)
"""
struct PoissonExp end

@node PoissonExp Stochastic [out, in]

@average_energy PoissonExp (q_out::PointMass, q_in::NormalDistributionsFamily, meta::Any) = begin
    y    = mean(q_out)
    m, v = mean_var(q_in)
    Eexp = exp(m + v / 2)                  # ρ = E_q[exp(z)]
    return Eexp - y * m + loggamma(y + 1)
end
