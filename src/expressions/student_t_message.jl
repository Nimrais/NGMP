export StudentTMessage

"""
    StudentTMessage(y, ã, b̃)

`StudentTMessage` represents the exact belief-propagation message that a Normal
factor with **unknown precision**

    f(τ, x) = 𝒩(y | x, τ⁻¹)

sends toward its **location** edge `x`, once the incoming Gamma cavity
`Gamma(τ; ã, b̃)` on the precision edge `τ` has been integrated out:

    μ_{f→x}(x) = ∫ 𝒩(y | x, τ⁻¹) Gamma(τ; ã, b̃) dτ ∝ (2b̃ + (x - y)²)^{-(ã + ½)}.

This is a Student-t in `x` (location `y`, `2ã` degrees of freedom) — the dual of
[`NormalPrecisionMessage`](@ref), which is the message of the same factor toward
`τ`. Calling the object evaluates the (unnormalised) message,
`(p::StudentTMessage)(x)`, and `log(p, x)` its logarithm

    log μ_{f→x}(x) = -(ã + ½) log(2b̃ + (x - y)²).

It is **not** quadratic in `x`, so it is non-conjugate to a Gaussian belief; it is
exactly the object projected onto the Gaussian `x` edge (see `project_to_normal`).
The overall normalising constant is dropped — it is irrelevant to the projection.

# Fields
- `y::T`: the observed value entering the Normal factor.
- `ã::T`: shape of the Gamma cavity on the precision `τ`.
- `b̃::T`: rate of the Gamma cavity on the precision `τ`.
"""
struct StudentTMessage{T<:Real} <: ClosedFormExpectations.Expression
    y::T
    ã::T
    b̃::T
end

function StudentTMessage(y::Real, ã::Real, b̃::Real)
    yp, ap, bp = promote(y, ã, b̃)
    return StudentTMessage{typeof(yp)}(yp, ap, bp)
end

# μ_{f→x}(x) ∝ (2b̃ + (x - y)²)^{-(ã + ½)}
function (p::StudentTMessage)(x)
    return (2 * p.b̃ + (x - p.y)^2)^(-(2 * p.ã + 1) / 2)
end

# log μ_{f→x}(x) = -(ã + ½) log(2b̃ + (x - y)²)
function Base.log(p::StudentTMessage, x)
    return -(2 * p.ã + 1) / 2 * log(2 * p.b̃ + (x - p.y)^2)
end
