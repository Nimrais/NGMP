"""
    NaturalGradientMessage

A rule-dispatch tag used in the `vconstraint` slot of ReactiveMP rules, marking a
*natural-gradient* message update: the outbound message toward an edge is the tangent
projection of the exact BP log-message at the edge's current marginal,

    η_{a→i} = ∇_μᵢ E_{q(zᵢ)}[ ℓ_{a→i}(zᵢ) ],

so the rule additionally receives the receiving edge's own marginal as `q_<edge>`.
Rules of this type coexist with ordinary `Marginalisation` (BP/VMP) rules for the
same node and are selected per-interface by [`NGMPDependencies`](@ref).

```julia
@rule PoissonExp(:in, NaturalGradientMessage) (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = ...
```
"""
struct NaturalGradientMessage end
