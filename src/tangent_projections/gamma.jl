export project_to_gamma

using SpecialFunctions: trigamma
import SpecialFunctions
import FastGaussQuadrature
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters
import ClosedFormExpectations: Logpdf

"""
    DerivativeEnhancedFunction(p::Logpdf{<:NormalPrecisionMessage}, expansion_point)

Bundle the log-message `p = Logpdf(NormalPrecisionMessage(...))` — already in log
space, so nothing is converted behind your back — with its analytic first and
second τ-derivatives, ready for [`project_to_gamma`](@ref). Pass the mean of the
receiving Gamma belief as `expansion_point` (the default; see the note above).
"""
function DerivativeEnhancedFunction(p::Logpdf{<:NormalPrecisionMessage}, expansion_point)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_normal_precision_first_derivative, p),   # τ ↦ ℓ′(τ)
        Base.Fix1(_normal_precision_second_derivative, p),  # τ ↦ ℓ″(τ)
    )
end

# Second-order Cov_q[T, ℓ] with T = (log τ, τ), from the two derivatives of ℓ at
# the expansion point x₀. The 2nd-order Taylor of ℓ about x₀, in monomial form,
# is  A·τ + B·τ² + const  with
#   A = ℓ′(x₀) - x₀ ℓ″(x₀),   B = ½ ℓ″(x₀);
# the constant drops out of every covariance, so
#   Cov_q[T, ℓ] ≈ A·Cov_q[T, τ] + B·Cov_q[T, τ²]
# with the exact Gamma raw-moment covariances
#   Cov_q[T, τ]  = (1/b,       a/b²),
#   Cov_q[T, τ²] = ((2a+1)/b², 2a(a+1)/b³).
# The expansion point enters only through A; at x₀ = a/b (the Gamma mean) this
# reduces to the central-moment form  d1/b + d2/(2b²),  (a/b²)(d1 + d2/b).
function _second_order_williams(d1, d2, x₀, a, b)
    A = d1 - x₀ * d2
    B = d2 / 2
    c1 = A / b         + B * (2 * a + 1) / b^2         # Cov_q[log τ, ℓ]
    c2 = A * (a / b^2) + B * 2 * a * (a + 1) / b^3     # Cov_q[τ, ℓ]
    return c1, c2
end

# (Δa, -Δb) = F⁻¹ (c1, c2), with the symmetric 2×2 Gamma Fisher solved by hand.
# F = [trigamma(a) 1/b; 1/b a/b²] matches ExponentialFamily's natural-space Fisher.
function _increments_from_williams(c1, c2, a, b)
    f11, f12, f22 = trigamma(a), 1 / b, a / b^2
    detF = f11 * f22 - f12^2
    η1 = (f22 * c1 - f12 * c2) / detF
    η2 = (f11 * c2 - f12 * c1) / detF
    return η1, -η2                          # (Δa, Δb)
end

# Natural parameters of an ExponentialFamilyDistribution{Gamma} are η = (a-1, -b).
_shape_rate(q::ExponentialFamilyDistribution{Distributions.Gamma}) =
    (getnaturalparameters(q)[1] + 1, -getnaturalparameters(q)[2])

"""
    mean(::SecondOrderClosedWilliamsProduct, f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{Gamma})

Second-order Williams product of `f` against the Gamma belief `q`, returned as the
gradient with respect to the **natural parameters** of `q`:

    ∇_η E_q[ℓ] = Cov_q[T, ℓ] = (Cov_q[log τ, ℓ], Cov_q[τ, ℓ]),   T = (log τ, τ).

The expectations come from the second-order expansion of `f` about
`f.expansion_point` (the Gamma mean `a/b` by default), so only
`f.first_derivative` and `f.second_derivative` evaluated there are used — no
quadrature. This is the Euclidean natural-coordinate gradient; the natural
gradient (the projected message) is the inverse-Fisher map of it, computed by
[`project_to_gamma`](@ref).
"""
function ClosedFormExpectations.mean(
    ::SecondOrderClosedWilliamsProduct,
    f::DerivativeEnhancedFunction,
    q::ExponentialFamilyDistribution{Distributions.Gamma},
)
    a, b = _shape_rate(q)
    d1 = f.first_derivative(f.expansion_point)
    d2 = f.second_derivative(f.expansion_point)
    return _second_order_williams(d1, d2, f.expansion_point, a, b)
end

"""
    project_to_gamma(f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{Gamma}) -> (Δa, Δb)

Second-order (delta-method) tangent projection of `f` onto the Gamma edge `q`.
Returns the natural-parameter increments `(Δa, Δb)` of the projected message

    μ̂(τ) ∝ τ^{Δa} exp(-Δb τ),

so a belief `Gamma(a₀, b₀)` updates to `Gamma(a₀ + Δa, b₀ + Δb)`. This is the
**natural gradient** — the inverse-Fisher map of the second-order Williams product
[`mean`](@ref):

    (Δa, -Δb) = F⁻¹ · (Cov_q[log τ, ℓ], Cov_q[τ, ℓ]),
    F = Fisher(q) = [trigamma(a)  1/b ; 1/b  a/b²].

`f.expansion_point` sets where the touching quadratic is taken; the Gamma mean
`a/b` is the default and most accurate choice. This is delta-method logic, not a
Laplace approximation: the default expansion point is the mean, not a mode. It is
**exact** when the expanded function is quadratic in `τ` (for *any* expansion
point), and otherwise most accurate when `q` is concentrated (large `a`); for
wide cavities prefer the quadrature form.
"""
function project_to_gamma(f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{Distributions.Gamma})
    a, b = _shape_rate(q)
    c1, c2 = ClosedFormExpectations.mean(SecondOrderClosedWilliamsProduct(), f, q)
    return _increments_from_williams(c1, c2, a, b)
end

"""
    project_to_gamma(p::NormalPrecisionMessage, q::ExponentialFamilyDistribution{Gamma}) -> (Δa, Δb)

Convenience projection for a [`NormalPrecisionMessage`](@ref): builds the
[`DerivativeEnhancedFunction`](@ref) for `log(p, ·)` expanded at the Gamma mean
`a/b` — so the expansion point is set correctly for you — and projects it onto
`q`. Equivalent to `project_to_gamma(DerivativeEnhancedFunction(p, a/b), q)`.
"""
function project_to_gamma(p::NormalPrecisionMessage, q::ExponentialFamilyDistribution{Distributions.Gamma})
    a, b = _shape_rate(q)
    return project_to_gamma(DerivativeEnhancedFunction(Logpdf(p), a / b), q)
end

const _GammaProjectionPoint = Union{
    GammaDistributionsFamily,
    ExponentialFamilyDistribution{Distributions.Gamma},
}

"""
    project(TangentProjection(type = ClosedForm), q::GammaDistributionsFamily, f::Logpdf)

Exact tangent projection of the log-message `f` onto the Gamma edge at the
receiving marginal `q` — the Gamma analogue of the Gaussian `project` method in
`closed_form_tangent.jl`. The `ClosedWilliamsProduct` against the
`ExponentialFamilyDistribution{Gamma}` form of `q` yields the natural-coordinate
gradient `∇_η E_q[ℓ] = (Cov_q[log τ, ℓ], Cov_q[τ, ℓ])`; the inverse-Fisher map
turns it into the natural-gradient increments `(Δa, Δb)` of the projected site

    μ̂(τ) ∝ τ^{Δa} exp(-Δb τ),

returned as an unchecked `ExponentialFamilyDistribution{Gamma}` with natural
parameters `(Δa, -Δb)` — sites may be improper during damping, so properness is
deliberately not validated.
"""
function project(::TangentProjection{<:ClosedForm}, q::_GammaProjectionPoint, f::Logpdf)
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    c1, c2 = ClosedFormExpectations.mean(ClosedWilliamsProduct(), f, q_ef)
    Δa, Δb = _increments_from_williams(c1, c2, _shape_rate(q_ef)...)
    η = promote(Δa, -Δb)
    return ExponentialFamilyDistribution(Distributions.Gamma, collect(η), nothing, nothing)
end

"""
    project(TangentProjection(type = Quadrature(n)), q::GammaDistributionsFamily, f::Logpdf)

Quadrature-exact tangent projection onto the Gamma edge: the Williams product
`∇_η E_q[ℓ] = (Cov_q[log τ, ℓ], Cov_q[τ, ℓ])` is evaluated on a **log-space
trapezoid grid**. Substituting `s = log τ`, the Gamma(a, b) expectation becomes
`∫ f(eˢ) exp(a·s − b·eˢ) ds` (up to normalization) — a smooth, unimodal integrand
decaying (double-)exponentially in both directions, for which the trapezoid rule
converges geometrically for **any** shape `a` (generalized Gauss–Laguerre, by
contrast, converges only algebraically against the `log τ` statistic and its
weights overflow for large `a`). The grid spans ±12 log-space standard deviations
(`√ψ₁(a)`) around `E[log τ] = ψ(a) − log b`; weights are self-normalized and both
covariances use the discrete means, so normalization errors cancel exactly.

Unlike the second-order `project_to_gamma`, this is exact for any width of `q` —
the delta expansion is only trustworthy when `q` is concentrated. Returns the same
unchecked `ExponentialFamilyDistribution{Gamma}` site `(Δa, -Δb)` as the
`ClosedForm` method.
"""
function project(::TangentProjection{Quadrature{n}}, q::_GammaProjectionPoint, f::Logpdf) where {n}
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    a, b = _shape_rate(q_ef)
    sμ = SpecialFunctions.digamma(a) - log(b)
    sσ = sqrt(trigamma(a))
    s = range(sμ - 12 * sσ, sμ + 12 * sσ; length = n)
    τ = exp.(s)
    logw = a .* s .- b .* τ                  # τ^{a-1} e^{-bτ} dτ = e^{as - be^s} ds
    logw .-= maximum(logw)
    w̃ = exp.(logw)
    w̃ ./= sum(w̃)
    ℓ = map(τk -> _eval_logmessage(f, τk), τ)
    Eℓ = sum(w̃ .* ℓ)
    Es = sum(w̃ .* s)
    Eτ = sum(w̃ .* τ)
    c1 = sum(w̃ .* (s .- Es) .* (ℓ .- Eℓ))   # Cov_q[log τ, ℓ]
    c2 = sum(w̃ .* (τ .- Eτ) .* (ℓ .- Eℓ))   # Cov_q[τ, ℓ]
    Δa, Δb = _increments_from_williams(c1, c2, a, b)
    η = promote(Δa, -Δb)
    return ExponentialFamilyDistribution(Distributions.Gamma, collect(η), nothing, nothing)
end

# Analytic τ-derivatives of  log μ_{f→τ}(τ) = -½ log(ṽ + τ⁻¹) - (y - m̃)²/(2(ṽ + τ⁻¹))
# (the τ-independent -½ log 2π drops out). With s = ṽ + τ⁻¹ and r = (y - m̃)²:
function _normal_precision_first_derivative(p::Logpdf{<:NormalPrecisionMessage}, τ)
    msg = p.dist
    r = (msg.y - msg.m̃)^2
    s = msg.ṽ + inv(τ)
    return (s - r) / (2 * s^2 * τ^2)                            # ℓ′(τ)
end

function _normal_precision_second_derivative(p::Logpdf{<:NormalPrecisionMessage}, τ)
    msg = p.dist
    r  = (msg.y - msg.m̃)^2
    s  = msg.ṽ + inv(τ)
    t2 = τ^2
    return (s - 2r) / (2 * s^3 * t2^2) + (r - s) / (s^2 * τ * t2)  # ℓ″(τ)
end

"""
    project(TangentProjection(type = DeltaApproximation), q::GammaDistributionsFamily, f::Logpdf{<:NormalPrecisionMessage})

Second-order (delta-method) tangent projection of a `NormalPrecisionMessage` —
a message with NO exact Williams product — via its analytic τ-derivatives
([`project_to_gamma`](@ref)). Trustworthy only when `q` is concentrated; for
wide `q` prefer `Unscented`/`Quadrature`.
"""
function project(::TangentProjection{<:DeltaApproximation}, q::_GammaProjectionPoint, f::Logpdf{<:NormalPrecisionMessage})
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    Δa, Δb = project_to_gamma(f.dist, q_ef)
    η = promote(Δa, -Δb)
    return ExponentialFamilyDistribution(Distributions.Gamma, collect(η), nothing, nothing)
end

# `ClosedForm` means an EXACT Williams product — which this message does not have.
function project(::TangentProjection{<:ClosedForm}, q::_GammaProjectionPoint, f::Logpdf{<:NormalPrecisionMessage})
    return error(
        "`NormalPrecisionMessage` has no closed-form Williams product against a Gamma belief. ",
        "Choose the approximation explicitly via `NGMPDependencies(...; projection = ...)`: ",
        "`TangentProjection(type = DeltaApproximation)` (2 analytic derivatives, biased for wide q), ",
        "`TangentProjection(type = Unscented)` (3 sigma points), or ",
        "`TangentProjection(type = Quadrature(n))` (exact to quadrature precision)."
    )
end
