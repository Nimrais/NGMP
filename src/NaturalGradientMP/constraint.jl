"""
    ClosedFormDefault

Sentinel for the default tangent-projection strategy of a
[`NaturalGradientMessage`](@ref). The concrete strategy types
(`TangentProjection`, `ClosedForm`, `Quadrature`, `UnscentedTransform`) live in
the parent `SurrogateModelling` module, which is loaded after this submodule —
so the default is stored as this sentinel and resolved by the parent module's
`resolve_projection(::ClosedFormDefault) = TangentProjection(type = ClosedForm)`.
"""
struct ClosedFormDefault end

"""
    NaturalGradientMessage(projection = ClosedFormDefault())

A rule-dispatch tag used in the `vconstraint` slot of ReactiveMP rules, marking a
*natural-gradient* message update: the outbound message toward an edge is the tangent
projection of the exact BP log-message at the edge's current marginal,

    η_{a→i} = ∇_μᵢ E_{q(zᵢ)}[ ℓ_{a→i}(zᵢ) ],

so the rule additionally receives the receiving edge's own marginal as `q_<edge>`.
Rules of this type coexist with ordinary `Marginalisation` (BP/VMP) rules for the
same node and are selected per-interface by [`NGMPDependencies`](@ref).

`projection` selects HOW the tangent projection is computed and is read by the rule
body via the `vconstraint` argument (`resolve_projection(vconstraint.projection)`):
the exact closed-form Williams product by default (`ClosedForm`; messages without
one raise an informative error), or an explicit approximation —
`TangentProjection(type = DeltaApproximation)` (second-order, 2 derivative evals),
`TangentProjection(type = Unscented)` (3 sigma points), or
`TangentProjection(type = Quadrature(n))`. Select it per model through
`NGMPDependencies(...; projection = ...)`.

```julia
@rule PoissonExp(:in, NaturalGradientMessage) (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = ...
```
"""
struct NaturalGradientMessage{P}
    projection::P
end

NaturalGradientMessage() = NaturalGradientMessage(ClosedFormDefault())

getprojection(constraint::NaturalGradientMessage) = constraint.projection

# `@call_rule Node(:edge, Constraint)` instantiates the constraint as `Constraint()`;
# making instances self-callable lets tests pass a specific projection through the
# same syntax: `@call_rule Node(:edge, NaturalGradientMessage(TangentProjection(...))) (...)`.
(constraint::NaturalGradientMessage)() = constraint
