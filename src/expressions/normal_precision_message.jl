export NormalPrecisionMessage

"""
    NormalPrecisionMessage(y, m̃, ṽ)

`NormalPrecisionMessage` represents the exact belief-propagation message that a
Normal factor with **unknown precision**

    f(τ, z) = 𝒩(y | z, τ⁻¹)

sends toward its precision edge `τ`, once the incoming Gaussian cavity
`𝒩(z | m̃, ṽ)` on the mean edge `z` has been integrated out. Convolving two
Gaussians adds variances, so the message is the marginal likelihood of the
observation at the cavity-inflated variance `ṽ + τ⁻¹`:

    μ_{f→τ}(τ) = ∫ 𝒩(y | z, τ⁻¹) 𝒩(z | m̃, ṽ) dz = 𝒩(y | m̃, ṽ + τ⁻¹).

Calling the object evaluates this message, `(p::NormalPrecisionMessage)(τ)`, and
`log(p, τ)` evaluates its logarithm

    log μ_{f→τ}(τ) = -½ log(2π (ṽ + τ⁻¹)) - (y - m̃)² / (2 (ṽ + τ⁻¹)).

As a function of the precision `τ` this is **not** affine in the Gamma
sufficient statistics `(log τ, τ)`: the `τ⁻¹` sitting inside both the
log-determinant and the quadratic make it non-conjugate to a Gamma belief. It is
therefore the object a Gamma tangent projection consumes to produce the
natural-parameter message on the `τ` edge.

In the zero-cavity limit `ṽ → 0` the message collapses to the conjugate Gamma
form `½ log τ - ½ (y - m̃)² τ` — exactly the variational mean-field update. The
`ṽ`-dependent log-determinant `-½ log(ṽ + τ⁻¹)` is the beyond-mean-field
correction the projection keeps and mean-field drops. The additive `-½ log 2π`
is constant in `τ` and irrelevant to the projection; it is retained so the
message integrates to the correct evidence.

# Fields
- `y::T`: the observed value entering the Normal factor.
- `m̃::T`: mean of the Gaussian cavity message on the latent mean edge `z`.
- `ṽ::T`: variance of the Gaussian cavity message on `z` (`ṽ ≥ 0`).
"""
struct NormalPrecisionMessage{T<:Real} <: ClosedFormExpectations.Expression
    y::T
    m̃::T
    ṽ::T
end

# Promote mixed-type arguments to a common element type, mirroring the
# constructors in ClosedFormExpectations (e.g. `LinearLogGamma`).
function NormalPrecisionMessage(y::Real, m̃::Real, ṽ::Real)
    yp, mp, vp = promote(y, m̃, ṽ)
    return NormalPrecisionMessage{typeof(yp)}(yp, mp, vp)
end

# Total variance the observation sees at precision `τ`: the cavity variance plus
# the factor's own variance `τ⁻¹`. Defined for `τ > 0`.
_total_variance(p::NormalPrecisionMessage, τ) = p.ṽ + inv(τ)

# μ_{f→τ}(τ): the message value, i.e. the Normal density 𝒩(y | m̃, ṽ + τ⁻¹).
function (p::NormalPrecisionMessage)(τ)
    s = _total_variance(p, τ)
    return exp(-(p.y - p.m̃)^2 / (2s)) / sqrt(2π * s)
end

# log μ_{f→τ}(τ): the exact log-message that is projected onto the Gamma τ edge.
# Subtyping `Expression` lets `log ∘ p` dispatch here through
# ClosedFormExpectations' `ComposedFunction{typeof(log), <:Expression}` hook.
function Base.log(p::NormalPrecisionMessage, τ)
    s = _total_variance(p, τ)
    return -(log(2π * s) + (p.y - p.m̃)^2 / s) / 2
end
