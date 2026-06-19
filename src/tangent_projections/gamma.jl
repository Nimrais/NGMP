export SecondOrderClosedWilliamsProduct, DerivativeEnhancedFunction, project_to_gamma

using SpecialFunctions: trigamma
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters
import ClosedFormExpectations: Logpdf

"""
    SecondOrderClosedWilliamsProduct()

Strategy for the **second-order, quadrature-free** Williams product against a
Gamma belief — the Gamma-edge analogue of
`ClosedFormExpectations.ClosedWilliamsProduct`.

`ClosedWilliamsProduct` returns the exact `∇_θ E_q[ℓ]` for a Gaussian edge; this
strategy returns the *second-order* value for a Gamma edge by expanding the
target to second order about its `expansion_point`, so the expectations become
closed-form Gamma moments instead of a Gauss–Legendre / Laguerre quadrature.

Following the ClosedFormExpectations convention, the coordinate system of the
returned gradient is fixed by the type of `q`: passing an
`ExponentialFamilyDistribution{Gamma}` returns the gradient with respect to the
**natural parameters** `η = (a-1, -b)` (a `Distributions.Gamma` would instead ask
for the gradient with respect to the Gamma parameters). It consumes a
[`DerivativeEnhancedFunction`](@ref); see [`mean`](@ref) for the returned quantity
and [`project_to_gamma`](@ref) for the natural gradient / projected message.
"""
struct SecondOrderClosedWilliamsProduct end

"""
    DerivativeEnhancedFunction(true_function, expansion_point, first_derivative, second_derivative)

A function bundled with everything needed for its second-order Taylor expansion
about `expansion_point`: the function itself and its first and second derivatives
(each a callable of one argument). This is the universal input to the Gamma
tangent projection — the projection only ever touches the two derivatives at the
expansion point, so any function that can supply them is accepted, regardless of
what it represents.

!!! note "Choosing the expansion point"
    The projection is computed exactly for the quadratic that touches the
    function at `expansion_point`, for *any* point — so the result is always a
    valid second-order projection. But that quadratic only resembles the true
    function near `expansion_point`, weighted by where `q` has mass, so the Gamma
    mean `a/b` is the default and most accurate choice; expanding far from where
    `q` concentrates stays valid but loses accuracy.

# Fields
- `true_function`: the function being expanded (informational; the projection
  uses only the derivatives below).
- `expansion_point`: the point to expand about — set it to the Gamma mean `a/b`.
- `first_derivative`: `f′`, a callable of one argument.
- `second_derivative`: `f″`, a callable of one argument.
"""
struct DerivativeEnhancedFunction{F, P, F1, F2}
    true_function::F
    expansion_point::P
    first_derivative::F1
    second_derivative::F2
end

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
