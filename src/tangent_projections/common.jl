export SecondOrderClosedWilliamsProduct, DerivativeEnhancedFunction

"""
    SecondOrderClosedWilliamsProduct()

Strategy for the **second-order, quadrature-free** Williams product against an
exponential-family belief — the delta-method analogue of
`ClosedFormExpectations.ClosedWilliamsProduct`.

`ClosedWilliamsProduct` returns the exact `∇_θ E_q[ℓ]`; this strategy returns the
*second-order* value by expanding the target to second order about its
`expansion_point`, so the expectations become closed-form moments of `q` instead
of a quadrature. It consumes a [`DerivativeEnhancedFunction`](@ref).

Following the ClosedFormExpectations convention, the coordinate system of the
returned gradient is fixed by the type of `q`: passing an
`ExponentialFamilyDistribution` returns the gradient with respect to the
**natural parameters**. An edge-specific `mean` method supplies the moments — see
[`project_to_gamma`](@ref) (Gamma edge) and [`project_to_normal`](@ref) (Normal
edge) for the resulting natural gradient / projected message.
"""
struct SecondOrderClosedWilliamsProduct end

"""
    DerivativeEnhancedFunction(true_function, expansion_point, first_derivative, second_derivative)

A function bundled with everything needed for its second-order Taylor expansion
about `expansion_point`: the function itself and its first and second derivatives
(each a callable of one argument). This is the universal input to a tangent
projection — the projection only ever touches the two derivatives at the
expansion point, so any function that can supply them is accepted, regardless of
what it represents.

!!! note "Choosing the expansion point"
    The projection is computed exactly for the quadratic that touches the
    function at `expansion_point`, for *any* point — so the result is always a
    valid second-order projection. But that quadratic only resembles the true
    function near `expansion_point`, weighted by where `q` has mass, so the mean
    of the receiving belief `q` is the default and most accurate choice;
    expanding far from where `q` concentrates stays valid but loses accuracy.

# Fields
- `true_function`: the function being expanded (informational; the projection
  uses only the derivatives below).
- `expansion_point`: the point to expand about — set it to the mean of `q`.
- `first_derivative`: `f′`, a callable of one argument.
- `second_derivative`: `f″`, a callable of one argument.
"""
struct DerivativeEnhancedFunction{F, P, F1, F2}
    true_function::F
    expansion_point::P
    first_derivative::F1
    second_derivative::F2
end
