export project_to_normal

import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, NormalMeanVariance
import ClosedFormExpectations: Logpdf

# (m, v) from the natural parameters η = (m/v, -1/(2v)) of an EF Normal.
function _mean_var(q::ExponentialFamilyDistribution{NormalMeanVariance})
    η = getnaturalparameters(q)
    v = -1 / (2 * η[2])
    m = -η[1] / (2 * η[2])
    return m, v
end

# Second-order Cov_q[T, ℓ] with T = (x, x²), from the two derivatives of ℓ at the
# expansion point x₀. The 2nd-order Taylor of ℓ about x₀, in monomial form, is
#   A·x + B·x² + const  with  A = ℓ′(x₀) - x₀ ℓ″(x₀),  B = ½ ℓ″(x₀);
# the constant drops out of every covariance, so
#   Cov_q[T, ℓ] ≈ A·Cov_q[T, x] + B·Cov_q[T, x²]
# with the exact Gaussian raw-moment covariances
#   Cov_q[T, x]  = (v,   2mv),
#   Cov_q[T, x²] = (2mv, 2v² + 4m²v).
function _second_order_williams_normal(d1, d2, x₀, m, v)
    A = d1 - x₀ * d2
    B = d2 / 2
    c1 = A * v         + B * 2 * m * v
    c2 = A * 2 * m * v + B * (2 * v^2 + 4 * m^2 * v)
    return c1, c2
end

# (ξ, Λ) from the inverse-Fisher map, read as a canonical Gaussian message
# μ̂(x) ∝ exp(ξ x - ½ Λ x²). G = [v 2mv; 2mv 2v²+4m²v] is the EF Normal
# natural-space Fisher, detG = 2v³.
function _increments_from_williams_normal(c1, c2, m, v)
    detG = 2 * v^3
    η1 = ((2 * v^2 + 4 * m^2 * v) * c1 - 2 * m * v * c2) / detG   # ξ
    η2 = (-2 * m * v * c1 + v * c2) / detG                        # coeff of x² = -½Λ
    return η1, -2 * η2                                            # (ξ, Λ)
end

"""
    mean(::SecondOrderClosedWilliamsProduct, f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{NormalMeanVariance})

Second-order Williams product of `f` against the Gaussian belief `q`, returned as
the gradient with respect to the **natural parameters** of `q`:

    ∇_η E_q[ℓ] = Cov_q[T, ℓ] = (Cov_q[x, ℓ], Cov_q[x², ℓ]),   T = (x, x²).

Computed from the second-order expansion of `f` about `f.expansion_point` (the
Gaussian mean by default), so only `f.first_derivative` and `f.second_derivative`
evaluated there are used — no Gauss–Hermite quadrature. This is the delta-method
analogue of a quadrature projection onto the Gaussian edge; the projected message
is the inverse-Fisher map of it, computed by [`project_to_normal`](@ref).
"""
function ClosedFormExpectations.mean(
    ::SecondOrderClosedWilliamsProduct,
    f::DerivativeEnhancedFunction,
    q::ExponentialFamilyDistribution{NormalMeanVariance},
)
    m, v = _mean_var(q)
    d1 = f.first_derivative(f.expansion_point)
    d2 = f.second_derivative(f.expansion_point)
    return _second_order_williams_normal(d1, d2, f.expansion_point, m, v)
end

"""
    project_to_normal(f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{NormalMeanVariance}) -> (ξ, Λ)

Second-order (delta-method) tangent projection of `f` onto the Gaussian edge `q`.
Returns the canonical parameters `(ξ, Λ)` of the projected message

    μ̂(x) ∝ exp(ξ x - ½ Λ x²),

a Gaussian pseudo-observation of precision `Λ` and mean `ξ/Λ`. It is the
inverse-Fisher map of the second-order Williams product [`mean`](@ref):

    (ξ, -½Λ) = G⁻¹ · (Cov_q[x, ℓ], Cov_q[x², ℓ]).

Because the second-order Taylor of `ℓ` is already quadratic in `x`, this collapses
to the pointwise delta surrogate at the expansion point `x₀`:

    Λ = -ℓ″(x₀),   ξ = ℓ′(x₀) - x₀ ℓ″(x₀) = ℓ′(x₀) + x₀ Λ,

i.e. the Gauss–Hermite `project_to_x` with the cavity-variance averaging dropped
(the Poisson-notebook delta approximation). It is **exact** when `ℓ` is quadratic
in `x` (for any expansion point) and otherwise most accurate at the Gaussian mean;
the convenience method below defaults `x₀` to that mean.
"""
function project_to_normal(f::DerivativeEnhancedFunction, q::ExponentialFamilyDistribution{NormalMeanVariance})
    m, v = _mean_var(q)
    c1, c2 = ClosedFormExpectations.mean(SecondOrderClosedWilliamsProduct(), f, q)
    return _increments_from_williams_normal(c1, c2, m, v)
end

"""
    project_to_normal(p::StudentTMessage, q::ExponentialFamilyDistribution{NormalMeanVariance}) -> (ξ, Λ)

Convenience projection for a [`StudentTMessage`](@ref): builds the
[`DerivativeEnhancedFunction`](@ref) for `log(p, ·)` expanded at the Gaussian mean
and projects it onto `q`. This is the second-order / delta-method version of the
Gauss–Hermite `project_to_x`.
"""
function project_to_normal(p::StudentTMessage, q::ExponentialFamilyDistribution{NormalMeanVariance})
    m, _ = _mean_var(q)
    return project_to_normal(DerivativeEnhancedFunction(Logpdf(p), m), q)
end

"""
    DerivativeEnhancedFunction(p::Logpdf{<:StudentTMessage}, expansion_point)

Bundle the Student-t log-message `p = Logpdf(StudentTMessage(...))` with its
analytic first and second x-derivatives, ready for [`project_to_normal`](@ref).
Pass the Gaussian mean as `expansion_point` (the default; see the note on
[`DerivativeEnhancedFunction`](@ref)).
"""
function DerivativeEnhancedFunction(p::Logpdf{<:StudentTMessage}, expansion_point)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        Base.Fix1(_student_t_first_derivative, p),   # x ↦ ℓ′(x)
        Base.Fix1(_student_t_second_derivative, p),  # x ↦ ℓ″(x)
    )
end

# Analytic x-derivatives of  log μ_{f→x}(x) = -(ã + ½) log(2b̃ + (x - y)²).
# With d = x - y:
function _student_t_first_derivative(p::Logpdf{<:StudentTMessage}, x)
    msg = p.dist
    d = x - msg.y
    return -(2 * msg.ã + 1) * d / (2 * msg.b̃ + d^2)               # ℓ′(x)
end

function _student_t_second_derivative(p::Logpdf{<:StudentTMessage}, x)
    msg = p.dist
    d = x - msg.y
    return -(2 * msg.ã + 1) * (2 * msg.b̃ - d^2) / (2 * msg.b̃ + d^2)^2  # ℓ″(x)
end
