# NaturalGradientMP.jl

Natural-gradient message passing (NGMP) as a first-class rule type for
[ReactiveMP.jl](https://github.com/ReactiveBayes/ReactiveMP.jl) /
[RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) — an external-package
prototype (Phase 1).

## The idea

For an edge carrying an exponential-family marginal `q(x)`, the NGMP message from an
adjacent factor toward `x` is the *tangent projection of the exact BP log-message at
the current marginal*:

```
η_{a→x} = ∇_μ E_{q(x)}[ ℓ_{a→x} ]        (natural gradient in mean coordinates)
```

Unlike BP or VMP, the message therefore depends on the marginal of the edge it is
sent *to*. This package wires that dependency into ReactiveMP's reactive machinery,
so a non-conjugate model runs as a single `infer(...; iterations = n)` call — no
outer loop rebuilding surrogate pseudo-observations.

## What's inside

- **`NaturalGradientMessage`** — a new rule-dispatch tag for the `vconstraint` slot
  of `@rule`. NGMP rules coexist with ordinary `Marginalisation` (BP/VMP) rules:

  ```julia
  @rule PoissonExp(:in, NaturalGradientMessage) (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = ...
  ```

- **`NGMPDependencies`** — a `ReactiveMP.FunctionalDependencies` policy. For the
  interfaces named in its specification it subscribes the outbound message to the
  *same edge's current marginal* (delivered as `q_<edge>`) and dispatches the
  `NaturalGradientMessage` rule; all other interfaces behave exactly as with
  `DefaultFunctionalDependencies`. Because dependencies are derived from the node's
  local factorization clusters, this composes freely with BP / structured / mean-field
  factorization constraints.

- **`DampingMeta(; alpha, beta, max_step=Inf)`** — heavy-ball damping in
  natural-parameter coordinates, `v ← β·v + α·(η⋆ − η); η ← η + v`, optionally
  bounding `‖v‖₂` by `max_step`. Parameter-only by design:
  GraphPPL's `@meta` evaluates its right-hand side once and shares the instance
  across all matched nodes. The mutable per-node-per-edge state
  (**`NGMPEdgeState`**: previous message, momentum buffers, firing counter) is
  created automatically at node activation, one per constrained interface.

- **`PoissonExp`** — `y ~ Poisson(exp(z))`, the non-conjugate leaf of the Poisson
  state-space model, with the closed-form NGMP rule
  (`ρ = exp(m + v/2)`, `ξ⋆ = y + (m−1)ρ`, `Λ⋆ = ρ`), a predictive `:out` rule,
  and the average energy for Bethe free-energy tracking.

## Usage

```julia
using RxInfer, NaturalGradientMP

@model function poisson_ssm(y, σ, m0, v0, deps, damping)
    z[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ PoissonExp(z[1]) where { dependencies = deps, meta = damping }
    for k in 2:length(y)
        z[k] ~ Normal(mean = z[k - 1], variance = σ)
        y[k] ~ PoissonExp(z[k]) where { dependencies = deps, meta = damping }
    end
end

# NGMP messages need a starting marginal on the constrained edges
@initialization function ngmp_init(y)
    q(z) = NormalMeanVariance.(log.(y .+ 1.0), 1.0)
end

result = infer(
    model = poisson_ssm(σ = 0.1, m0 = 0.0, v0 = 10.0,
                        deps = NGMPDependencies(in = nothing),
                        damping = DampingMeta(alpha = 0.5, beta = 0.2)),
    data = (y = counts,),
    initialization = ngmp_init(counts),
    iterations = 10,
    free_energy = true
)
```

See `examples/poisson_sunspots.jl` for the full sunspot-smoothing example. On the
monthly sunspot series (3288 observations) the native run matches the hand-rolled
surrogate outer loop (projection → damping → Kalman/RTS sweep, as in the
NGMP paper's experiments) to machine precision (`max |Δmean| ≈ 3e-15`).

## Notes and conventions

- **`q_out`, not `m_out`.** RxInfer auto-factorizes data variables into their own
  clusters, so an observed edge reaches rules as a marginal (`q_out::PointMass`).
  NGMP rules for likelihood nodes should follow that convention.
- **`meta::Any` on sibling rules.** Attaching any meta to a node changes dispatch
  for *all* of its rules and its average energy; declare `meta::Any` everywhere
  except the NGMP rule itself (which takes the `NGMPEdgeState` wrapper).
- **Initialization is required** on NGMP-constrained edges: the message depends on
  the marginal, which depends on the message — `@initialization q(x) = ...` breaks
  the cycle.
- **One firing per iteration.** The outbound stream re-emits only when both the data
  message and the edge marginal are fresh, i.e. exactly once per `infer` iteration —
  which makes the stateful damping trajectory identical to the reference outer loop.
  `NGMPDependencies.states[i].nfired` exposes the per-edge counter for diagnostics.
- **Free energy** is the Bethe free energy of the *current* Gaussianized surrogate
  (as in the reference notebook); it settles near the fixed point but is not a
  single monotone objective.
- **Missing observations** are handled out of the box: the NGMP rule never fires on
  a missing leaf (equivalent to a flat pseudo-observation) and the `:out` rule
  returns a plug-in Poisson predictive.
- **Internal-API coupling.** `NGMPDependencies` ports two unexported ReactiveMP
  bodies (`activate!` and the `RequireMarginalFunctionalDependencies` dependency
  builder); compat is pinned to `ReactiveMP ~6.3`.

## Roadmap (Phase 2)

Derive the NGMP pipeline automatically from a form constraint
(`q(x) :: NaturalGradient(NormalMeanVariance)`) via a GraphPPL plugin that rewrites
adjacent nodes' dependencies, instead of the explicit `where { dependencies = ... }`
opt-in. Generic (non-closed-form) rules via ExponentialFamilyProjection are also
planned.
