# Recall History — notebooks


## Session 3a8400f4 — 2026-07-06 19:57

**You:** (base) ➜  Lukashchuk-TMLR-Natural-gradient-message-passing git:(main) ✗ pwd
/Users/mykola/repos/papers/Lukashchuk-TMLR-Natural-gradient-message-passing Read my paper and the ReactiveMP.jl in this folder to understand how the following idea can be implemented once I have an Exponential family constraint on an edge let say FormConstraint(Type) when we are computing the message torwards this edge in the subscrition rule we would have the maringal on this edge as part of the subscription it should frely stack with all other possible factroization constraints so BP, VMP and q_x subscrption should be easily composable but it should be a new type of the rule Don't write any code just come up with a plan how external package can extend RxInfer in this way

**Claude:** I'll first look at what's in both folders, then dig into the paper and ReactiveMP's rule/subscription machinery.
  - `Bash: ls /Users/mykola/repos/papers/Lukashchuk-TMLR-Natural-gradient-message-passing/ && echo --- && ls /Users/mykola/repos/ReactiveBayes/`
  - `Agent: Summarize TMLR paper method`
  - `Agent: Map ReactiveMP rule machinery`
  - `Agent: Trace constraints flow into ReactiveMP`

**Claude:** All three explorations are done, and the good news is that the machinery you need already has ~80% of its extension points in place. Here's the assessment and plan.

## TLDR

Your NGMP idea maps almost one-to-one onto an existing ReactiveMP primitive: `RequireMarginalFunctionalDependencies` (`ReactiveMP.jl/src/nodes/dependencies.jl:241-302`) already makes an outbound message on edge `x` subscribe to the *current marginal q(x) of that same edge* and pass it into the rule as `q_x` — composably with any factorization, because dependency policies derive the `m_*`/`q_*` split from the node's local clusters. What's missing, and what your external package would add, is: (1) a way to make this happen *automatically* from a form constraint `q(x) :: ExponentialFamily(T)`, and (2) a genuinely new rule type so NGMP rules don't masquerade as ordinary `(m_..., q_x)` rules. Both are achievable externally with at most two small upstream PRs.

## Why it composes for free

The load-bearing fact: `DefaultFunctionalDependencies` computes, for an outbound interface, messages = *other edges in the same factorization cluster*, marginals = *marginals of the other clusters*. That single function is what makes BP (one big cluster → all `m_*`), mean-field (singleton clusters → all `q_*`), and structured VMP fall out of the same code. Crucially, the marginal of the cluster *containing* the target edge is never among the dependencies — so the name `q_x` for the target edge's own marginal is guaranteed free, and splicing it in stacks cleanly on top of any factorization. This is exactly the paper's structure: the tangent projection $\mathsf{T}_i^{\lambda_i}[\ell_{a\to i}]$ needs the standard BP/VMP inputs (to form $\ell_{a\to i}$, or its tilted-factor variant under structured factorization) *plus* $q_{\lambda_i}$ (the projection point).

## The plan

**Package**: a new external package (say `RxInferNGMP.jl`), depending on ReactiveMP, ExponentialFamily, ExponentialFamilyProjection, GraphPPL; following the `RxInfer.jl/ext/ProjectionExt` + `ReactiveMP.jl/ext/ReactiveMPProjectionExt` pattern as templates.

**1. The constraint object.** Define a plain struct, e.g. `NaturalGradient(NormalMeanVariance)` — like `ProjectedTo`, *not* a subtype of `AbstractFormConstraint`, so it flows through `WrappedFormConstraint`. Implement the standard variable-side surface so `q(x) :: NaturalGradient(T)` works in `@constraints` today: `constrain_form`, `default_prod_constraint`, `default_form_check_strategy`, and `prepare_context` (the stateful-context hook `ProjectionExt` uses for warm starts — you'll want it for damping state, see item 5). On the variable side this constraint is cheap: at the fixed point the marginal is just $\lambda_i = \eta_{b\to i} + \eta_{c\to i}$, a closed-form product of two same-family messages, so `constrain_form` is nearly a no-op / type-assertion once the messages are NGMP messages.

**2. A new dependency policy.** `NaturalGradientDependencies <: ReactiveMP.FunctionalDependencies`, wrapping the specification of which interfaces are constrained (and with what family). Its `functional_dependencies(policy, node, interface, iindex)` delegates to the default cluster-based logic, then splices a `FactorNodeLocalMarginal` connected to `get_stream_of_marginals(getvariable(interface))` into the marginal dependencies — literally the `RequireMarginalFunctionalDependencies` body, generalized to carry the EF family info. Because it builds *on top of* the cluster partition, BP/structured/mean-field composition is automatic. Only the constrained interfaces get the extra dependency; unconstrained edges on the same node keep pure BP/VMP behavior — matching the paper's "unconstrained edges keep the classical update."

**3. The new rule type — dispatch on the variable-constraint slot.** This is the "new type of rule" you asked for, and there's a dormant dispatch dimension built for exactly this: every `rule` call carries a `vconstraint` argument, currently hardcoded to `Marginalisation()` in the generic `activate!` (`dependencies.jl:55`). Your policy defines its own `activate!(::NaturalGradientDependencies, node, options)` method (the `MixtureNode` policy shows this is a supported pattern) that passes `vconstraint = NaturalGradientMessage(family)` for constrained interfaces. Then NGMP rules are written as

- `@rule Node(:out, NaturalGradientMessage) (m_in::..., q_out::...) = ...` — BP-factorized NGMP,
- `@rule Node(:out, NaturalGradientMessage) (q_in::..., q_out::...) = ...` — tilted/SVMP-factorized NGMP,

and they can never collide with or shadow existing `Marginalisation` rules, while BP/VMP/NGMP rules for the same node coexist in one method table. This is semantically honest too: the paper's message *is* a different mathematical object (a tangent projection at $q_{\lambda_i}$), not a marginalization.

**4. Generic fallback rule.** Register a `rulefallback` (a first-class slot in `FactorNodeActivationOptions` / `MessageMapping`) for `NaturalGradientMessage`: when no analytic rule exists, compute $\eta_{a\to i} = \nabla_{\mu_i}\mathbb{E}_{q_{\lambda_i}}[\ell_{a\to i}]$ numerically — the covariance/Fisher form via ExponentialFamily's `fisherinformation`/sufficient statistics plus quadrature or sampling of $\mathrm{Cov}_{q}[T_i, \ell]$. Ship analytic rules for the paper's worked cases (Gaussian-with-precision node toward the state edge and toward the Gamma/Wishart edge, Poisson leaf) as the fast path.

**5. Damping and state.** The undamped fixed-point map oscillates (your paper needs α ≈ 0.25–0.5), so damping is not optional. Two composable places to put it, both existing mechanisms: geometric averaging in natural parameters inside the rule via a stateful meta/context object holding $\eta^{(t-1)}$ (the `ProjectionContext.previous` pattern), or as a message `postprocessor` on the constrained interfaces. I'd put it in the rule context, since damping in natural coordinates is family-aware. The cavity-point refinement your experiment script uses (projecting at $q$ with the own site divided out) also lives naturally here: the previous outbound $\eta^{(t-1)}$ is in the context, and $q$ arrives as `q_x`, so the cavity is computable inside the rule without extra subscriptions.

**6. Auto-wiring from the constraint (the RxInfer-level glue).** The activation order in `RxInfer.jl/src/model/plugins/reactivemp_inference.jl:272-326` is favorable: constraints are written onto variable NodeData first (`VariationalConstraintsMarginalFormConstraintKey`), variables activate, factor nodes activate last — and factor activation already reads a per-node `:dependencies` extra (`ReactiveMPExtraDependenciesKey`). So the wiring is a GraphPPL plugin in your package whose `postprocess_plugin` runs after `VariationalConstraintsPlugin` and before `ReactiveMPInferencePlugin`: for each factor node, inspect neighbors' marginal-form-constraint extras; where it finds your `NaturalGradient` constraint, write (or merge into) the node's dependencies extra a `NaturalGradientDependencies` spec for that interface. Stage it:

- *Phase 1 (works today, zero upstream changes)*: users opt in per node with `y ~ Node(x, τ) where { dependencies = NaturalGradientDependencies(x = ...) }`, alongside the ordinary `q(x)::NaturalGradient(T)` constraint. Ugly but functional; proves the rules and policy.
- *Phase 2 (one small upstream PR)*: RxInfer's `infer` gains an `extra_plugins` kwarg appended to the `PluginsCollection` in `batch.jl:177-199` / `streaming.jl:613-628`. Then `q(x)::NaturalGradient(T)` alone activates everything — the user-facing story you want: add one constraint line, and BP/VMP/NGMP mix per-edge.

**7. Initialization / cycle-breaking.** The subscription graph gains a cycle: message toward x ← q(x) ← messages toward x. This is the same cycle `RequireMarginalFunctionalDependencies` already manages via `set_initial_marginal!`; require `@initialization q(x) = ...` for constrained edges (your experiments warm-start from a VMP fit — worth recommending in docs as the default recipe) and fail with a clear error if missing.

**Upstream PRs worth proposing** (both tiny and non-breaking): the `extra_plugins` kwarg above, and optionally refactoring the hardcoded `vconstraint = Marginalisation()` in the generic `activate!` into a `message_constraint(dependencies, interface)` function with a `Marginalisation()` default — that would let you reuse the generic `activate!` loop instead of copying it.

**Risks / open items to validate early**: Bethe free energy reporting (the `score`/BFE plugin averages over node-local marginals; check the constrained-edge terms match the paper's constrained-Bethe objective, or mark BFE unsupported initially); reactive re-triggering granularity (message toward x now fires on every q(x) update — fine for the iterative scheme, but confirm the `PushNew`-gated `combineLatest` doesn't double-fire per iteration); and multi-degree edges (the theory is stated for degree-2 Forney edges; ReactiveMP variables of degree > 2 go through `EqualityChain`, which conveniently *is* the Forney-style equality expansion, so q(x) as the product over the chain should still be the right projection point — verify on a ≥3-degree example).

The natural first milestone: Phase 1 package with the `NaturalGradientDependencies` policy, the `NaturalGradientMessage` rule tag, analytic rules for `NormalMeanPrecision`-toward-state and toward-Wishart, damping context — then reproduce your `ssm_ng_bp_wishart_experiment.jl` result natively (no surrogate-graph outer loop, just `infer` with iterations) as the correctness test against the surrogate implementation.

**You:** 5. dumping and momentum can be organized through meta I guess it can hold previous message for every local edge if needed we can call it DampingMeta okay you can do phase 1 check how Poisson is implemented here reproduce this example /Users/mykola/repos/surrogate-modelling/notebooks/poisson_surrogate.jl with the Phase 1 in a new package as an example

**You:** 5. dumping and momentum can be organized through meta I guess it can hold previous message for every local edge if needed we can call it DampingMeta okay you can do phase 1 check how Poisson is implemented here reproduce this example /Users/mykola/repos/surrogate-modelling/notebooks/poisson_surrogate.jl with the Phase 1 in a new package as an example

**Claude:** Two things to explore before designing: the Poisson notebook you want reproduced, and how the Poisson node/rules are implemented in ReactiveMP. I'll fan those out now.
  - `Agent: Read poisson surrogate notebook`
  - `Agent: Map Poisson node and external rule registration`
  - `Agent: Design NGMP package implementation`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `ToolSearch: select:ExitPlanMode`
  - `ExitPlanMode: # NaturalGradientMP.jl — Phase 1: NGMP rule type + native Poisson sunspot example ## Context The TMLR paper (Natural-gradient message passing) computes the mess`
  - `ToolSearch: select:TaskCreate,TaskUpdate`
  - `Read: /Users/mykola/repos/ReactiveBayes/ReactiveMP.jl/src/nodes/dependencies.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/nodes/poisson_exp.jl`
  - `TaskCreate: Scaffold package: Project.toml, module, constraint.jl, damping.jl`
  - `TaskCreate: Implement PoissonExp node and rules`
  - `TaskCreate: Implement NGMPDependencies policy and custom activate!`
  - `TaskCreate: Write tests: rule tests, reference impl, integration`
  - `TaskCreate: Create sunspot example and README`
  - `TaskUpdate: 1`
  - `Bash: head -30 /Users/mykola/repos/ReactiveBayes/ReactiveMP.jl/Project.toml && julia --version`
  - `Bash: grep -E "TupleTools|Unrolled" /Users/mykola/repos/ReactiveBayes/ReactiveMP.jl/Project.toml; grep -E "^(ReactiveMP|RxInfer|Rocket|BayesBase|ExponentialFamily|Dis`
  - `Bash: head -5 /Users/mykola/repos/ReactiveBayes/RxInfer.jl/Project.toml /Users/mykola/repos/ReactiveBayes/BayesBase.jl/Project.toml /Users/mykola/repos/ReactiveBayes/`

**Claude:** Now I'll create the package. First generate a UUID and scaffold the structure.
  - `Bash: julia -e 'using UUIDs; println(uuid4())' && mkdir -p /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/{src/nodes,test,examples}`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/Project.toml`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/NaturalGradientMP.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/constraint.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/damping.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/dependencies.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/nodes/poisson_exp.jl`

**Claude:** The `⋆` character in `ξ⋆`/`Λ⋆` is a Julia infix operator, not a valid identifier suffix — fixing the argument names:
  - `Edit: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/damping.jl`

**Claude:** Now let me instantiate the environment and check the package loads (this validates the `@rule`/`@node` macro usage that the LSP can't parse):
  - `Bash: cd /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.status()' 2>&1 | tail -20`
  - `Bash: julia --project=. -e ' using NaturalGradientMP, ReactiveMP, BayesBase, ExponentialFamily # α=1, β=0 from zero state → message should equal the undamped target s`
  - `Bash: julia --project=. -e ' using NaturalGradientMP, ReactiveMP, BayesBase, ExponentialFamily, SpecialFunctions, Distributions state = NGMPEdgeState(DampingMeta(alph`

**Claude:** Core rule math verified. Tasks 1–2 done; now the critical piece — testing the dependency policy inside a real `infer` run. Setting up a scratch environment with RxInfer:
  - `TaskUpdate: 1`
  - `TaskUpdate: 2`
  - `TaskUpdate: 3`
  - `Bash: julia -e ' using Pkg Pkg.activate("/private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad/ngmp-env") Pkg.deve`

**Claude:** Now the end-to-end smoke test — a small chain with NGMP dependencies vs a hand-rolled Kalman reference:
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad/smoke.jl`
  - `Bash: julia --project=/private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad/ngmp-env /private/tmp/claude-501/-User`
  - `Bash: julia --project=/private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad/ngmp-env /private/tmp/claude-501/-User`

**Claude:** `connect!` lives in ReactiveMP, not Rocket:
  - `Edit: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/dependencies.jl`
  - `Bash: julia --project=/private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad/ngmp-env /private/tmp/claude-501/-User`
  - `Edit: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/dependencies.jl`
  - `Edit: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/src/dependencies.jl`

**Claude:** Now let me check how a `missing` observation behaves (for the prediction path) before writing tests:
  - `Bash: cd /private/tmp/claude-501/-Users-mykola-repos-ReactiveBayes/3a8400f4-e6e3-4c9a-8421-8e705af90a6b/scratchpad && julia --project=ngmp-env -e ' using RxInfer, Nat`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/test/reference.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/test/rule_tests.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/test/integration_tests.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/test/runtests.jl`
  - `Bash: cd /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -B3 "Stacktrace" | head -15`

**Claude:** Wrong UUID for the Test stdlib — let me fix it:
  - `Bash: julia -e 'using Pkg; println([(k,v) for (k,v) in Pkg.Types.stdlibs() if v isa Tuple ? first(v) == "Test" : v == "Test"])' 2>/dev/null || julia -e 'import TOML;`
  - `Edit: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/Project.toml`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -15`
  - `TaskUpdate: 3`
  - `TaskUpdate: 4`
  - `TaskUpdate: 5`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/datasets/sunspots.jl`

**Claude:** Now the example files and README:
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/examples/sunspots.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/examples/poisson_sunspots.jl`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/examples/Project.toml`

**Claude:** Running the example end-to-end (this instantiates Plots and runs on the full ~3300-month series):
  - `Bash: julia /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/examples/poisson_sunspots.jl 2>&1 | grep -vE "^\s*(\[|✓|◐|◑|◒|◓|Precompiling|Resolving|Installed|Up`

**Claude:** Machine-precision agreement on the full 3288-month series (2.7e-15). Let me check the figure and write the README:
  - `Read: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/examples/poisson_sunspots.png`
  - `Write: /Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl/README.md`

**Claude:** Final verification — full test suite one more time on the finished package:
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4 && ls -R src examples test | head -30`
  - `TaskUpdate: 5`
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-ReactiveBayes/memory/ngmp-package-phase1.md`
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-ReactiveBayes/memory/MEMORY.md`

**Claude:** Phase 1 is done. **`NaturalGradientMP.jl` is built, all 136 tests pass, and the native NGMP run reproduces your `poisson_surrogate.jl` notebook to machine precision** — max |Δmean| = 2.7e-15 against the surrogate outer loop on the full 3288-month sunspot series, with a single `infer(iterations = 10)` call instead of the rebuild-and-rerun loop.

The package lives at `/Users/mykola/repos/ReactiveBayes/NaturalGradientMP.jl`:

- **`NaturalGradientMessage`** — the new rule type, dispatched via the previously-dormant `vconstraint` slot, so NGMP rules coexist with BP/VMP `Marginalisation` rules for the same node.
- **`NGMPDependencies`** — the policy that subscribes the outbound message to the *receiving edge's own marginal* (arrives as `q_in`) and swaps in the new rule tag; it derives everything else from the factorization clusters, so it composes with BP/structured/mean-field unchanged.
- **`DampingMeta` + `NGMPEdgeState`** — your damping-through-meta idea, with one twist discovered during verification: GraphPPL's `@meta` evaluates its right-hand side **once** and shares that instance across all matched nodes, so `DampingMeta` holds only (α, β) and the mutable per-node-per-edge state (previous η, momentum buffers, firing counter) is created automatically at activation — each interface gets its own `MessageMapping`, which gives per-edge state for free.
- **`PoissonExp`** with the closed-form NGMP rule (ρ = e^{m+v/2}, ξ⋆ = y+(m−1)ρ, Λ⋆ = ρ), predictive `:out` rule, and average energy for free-energy tracking.

Things that worked out better than planned: the free energy computes with no extra marginal rules (trace settles exactly like the notebook's), and **missing observations work out of the box** — the NGMP rule simply never fires on a missing leaf (verified `nfired = 0`, identical to your masked flat pseudo-observation), while the `:out` rule returns the plug-in Poisson predictive. The firing-count test confirms exactly one rule invocation per edge per iteration, so the stateful damping trajectory matches the reference loop step for step.

Two caveats documented in the README: data edges reach rules as `q_out::PointMass` (not `m_out`) because RxInfer auto-factorizes data variables, and the policy ports two unexported ReactiveMP internals, so compat is pinned to `ReactiveMP ~6.3`. To try it: `julia examples/poisson_sunspots.jl` (self-bootstrapping; saves the smoothing figure), or `Pkg.test()` in the package. Phase 2 — deriving all this from `q(x) :: NaturalGradient(T)` via a GraphPPL plugin — is scoped in the plan file when you want it.

**You:** all good but just explain two things: 1. data edges reach rules as q_out::PointMass (not m_out) because RxInfer auto-factorizes data variables, and the policy ports two unexported ReactiveMP internals, so compat is pinned to ReactiveMP ~6.3. why is it specifc to ReactiveMP 6.3?

**Claude:** Both come down to specifics of how ReactiveMP wires things, so let me unpack each.

**1. Why `q_out::PointMass` instead of `m_out`**

In ReactiveMP, whether a dependency arrives in a rule as `m_*` or `q_*` is not about what the thing *is* — it's decided by the node's local factorization clusters. When computing the outbound message on an interface, edges in the **same cluster** contribute their inbound *messages* (`m_*`), and every **other cluster** contributes its *marginal* (`q_*`). That's the single mechanism behind BP (one big cluster → all `m_*`) and mean-field (singleton clusters → all `q_*`).

Now, when RxInfer builds the graph, it creates every data/constant variable with `factorized = true`, and GraphPPL's constraint engine then unconditionally splits each data neighbor into its own singleton cluster — regardless of what your `@constraints` say. The reasoning: a joint belief `q(out, in)` over an *observed* `out` is meaningless (the observation is a delta; there's nothing to marginalize jointly), and the "marginal" on a data edge is simply the `PointMass` observation itself. So for `PoissonExp(out, in)` with observed `y`, the clusters are always `((out,), (in,))`: from the perspective of the `:in` interface, `out` sits in a *different* cluster, hence it arrives as the marginal `q_out = PointMass(y)`.

Note there's no information difference — for an observed variable the message and the marginal are the same `PointMass`. It's purely a naming/dispatch convention: the rule signature has to say `q_out` or it won't be found. (This is also why ReactiveMP's own VMP-style likelihood rules and your `surrogate-modelling` `PoissonExp` rule already used `q_out`.)

**2. Why the `~6.3` pin**

It's not that anything is known to break outside 6.3 — it's that `NGMPDependencies` contains verbatim *copies* of two function bodies that are **unexported internals** of ReactiveMP 6.3.0:

- the generic `activate!(::FunctionalDependencies, node, options)` loop (copied so we could replace the hardcoded `vconstraint = Marginalisation()` on line 56 with `NaturalGradientMessage()`, and wrap the meta into a per-edge `NGMPEdgeState`), and
- the `RequireMarginalFunctionalDependencies` dependency builder (copied to splice the same-edge marginal in).

Julia's semver contract only covers public API. Internals like `collect_latest_messages`, the `MessageMapping` constructor's arity and field order, `node_if_required`, `postprocess_stream_of_outbound_messages` — all of which our ports call — can legitimately change in a 6.4 *minor* release without it counting as breaking. If that happens, the failure mode isn't necessarily a load error (the good case); the bad case is *silent divergence*: say 6.4's `activate!` gains an extra step (a new annotations pass, a new stream hook), every stock node gets it, and our copied loop doesn't — NGMP nodes would then behave subtly differently from the rest of the graph with no error anywhere.

`~6.3` means "allow 6.3.x patches, refuse 6.4 until a human re-diffs the two ported bodies against upstream and bumps compat deliberately." It's a verified-against pin, insurance rather than a known incompatibility. The clean way to relax it is the small upstream PR we scoped in the design: if ReactiveMP made the constraint policy-derived (e.g. `vconstraint = message_constraint(dependencies, interface)` with a `Marginalisation()` default, and similarly a hook for wrapping the meta), we could delete the copied `activate!` entirely, keep only the dependency-builder override, and widen compat to plain `"6.3"` (any 6.x ≥ 6.3).

**You:** I moved your implementation and tested it in this repo and now I want you to use to implement a bit more important goal look our constaraints are easily composing with VMP or BP so I wonder can you implement me this model (base) ➜  dynamic git:(main) ✗ pwd
  /Users/mykola/repos/probabilistic_ensemble_forecasting/src/model_zoo/dynamic here for now with vmp constraints on the toop     q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β) but the messages torwards z and \gamma will be projected also you would need to adapt your dumping
  meta to be able to support not only univariate normal but also gamma, it can be done if you will store a distribution and inplace of the summation of the natural parameters you will do the product of the messages from previous iteration and this one, do you
  understand what needs to be done, do it for ETTh1 dataset yaml is here sessions/dynamic/vae/dynamic_ETTh1_96.yamlfor the priors implement it as a notebook here notebooks/vmp_vs_ngmp/Poisson_VMP_vs_NGMP.jl and example with surrogate modelling is here
  dynamic_ngmp_surrogate.jl and all possible dynamic files but we are doing it with NGMP normal beatiful implementation that you made not with the surrogate models, surrogate models are here just to show the point it's not easy to stack them again in RxInfer so for
  me be able to extend dynamic please implement it in style such that I would be able to use it like here notebooks/vmp_vs_ngmp/Poisson_VMP_vs_NGMP.jl. I moved the folder structure don't be confused you can explore the surrogate modelling now, it has NaturalGradientMP as a subpackage

**Claude:** New scope: bring native NGMP to the *dynamic* model with a structured VMP factorization `q(w)q(z,γ)q(τ)q(β)` and NGMP-projected messages toward `z` and `γ`, with a generalized damping meta (distribution-valued, works for Gamma too). Before designing, I need to understand the dynamic model, the reference surrogate scripts, the notebook style to follow, and the moved package. Fanning out three explorers:
  - `Agent: Explore dynamic model and ETTh1 config`
  - `Agent: Explore surrogate-modelling new layout and notebook style`
  - `Agent: Explore dynamic surrogate scripts and equations`
  - `Bash: sed -n '1,40p' dynamic_ngmp_surrogate.jl && echo ===== && sed -n '150,230p' dynamic_ngmp_surrogate.jl`
  - `Bash: grep -rn "Log\b" src/ --include="*.jl" -l | head; echo ---; grep -rn "@node Log\|@node Exp\|struct Log\|struct Exp\b" src/ ../probabilistic_ensemble_forecasting`
  - `Bash: cat ../probabilistic_ensemble_forecasting/src/log.jl; echo =====EXP; sed -n '1,60p' ../probabilistic_ensemble_forecasting/src/exp.jl; echo =====SESSIONS; ls ses`
  - `Bash: ls sessions/dynamic/vae/ 2>/dev/null | head -8; echo "---PREP---"; grep -rn "function before_rxinfer" ../probabilistic_ensemble_forecasting/src/model_zoo/model_`
  - `Bash: sed -n '799,850p' ../probabilistic_ensemble_forecasting/src/model_zoo/model_specifier.jl; echo "---SPEC-FROM-YAML---"; sed -n '630,700p' ../probabilistic_ensemb`
  - `Bash: grep -n "_parse_spec\|function read_spec\|load_experiment\|YAML.load" ../probabilistic_ensemble_forecasting/src/model_zoo/model_specifier.jl | head; echo "---";`
  - `Bash: cat src/SurrogateModelling.jl; echo "---NB-ENV---"; ls notebooks/ notebooks/vmp_vs_ngmp/ 2>/dev/null; echo "---YAML-READER---"; sed -n '145,160p' ../probabilist`
  - `Agent: Design dynamic NGMP implementation`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `ExitPlanMode: # Dynamic ensemble VMP vs NGMP: generic damping + Log-node NGMP rules + ETTh1 notebook ## Context Phase 1 (native NGMP for `PoissonExp`, now living in `surrogat`
  - `TaskCreate: Generalize damping to distribution-valued EF state`
  - `TaskCreate: Add Gamma-edge projection + Log NGMP rules`
  - `TaskCreate: Add damping + Log rule tests, keep suite green`
  - `TaskCreate: Create Dynamic_VMP_vs_NGMP notebook and run it`
  - `TaskUpdate: 6`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/damping.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/NaturalGradientMP.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/nodes/poisson/rules/natural_gradient.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/damping.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/NaturalGradientMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/poisson/rules/natural_gradient.jl`

**Claude:** Step 1 done. Now reading the tangent-projection files to add the Gamma `project` method:
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/common.jl`
  - `TaskUpdate: 6`
  - `TaskUpdate: 7`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/nodes/log/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`

**Claude:** Now a quick compile + `@call_rule` smoke test before writing the formal tests:
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, Distributions import ClosedFormExpectations: LogGamma import SurrogateModelling: NaturalGradientMP # -`

**Claude:** Damping works; just need `Log` in scope for `@call_rule`:
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, Distributions import ClosedFormExpectations: LogGamma import ProbabilisticEnsembling: Log import Surro`

**Claude:** Both rules match the reference closed forms exactly. Now the critical de-risk: NGMPDependencies on a Deterministic node in a real `infer`:
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, Distributions import ProbabilisticEnsembling: Log @model function toy(y, pred, deps, damping) γ ~ Gamm`

**Claude:** The Deterministic-node path works end-to-end: proper posteriors, both edge states fire exactly once per iteration. Now the formal tests — first checking the existing test structure:
  - `TaskUpdate: 7`
  - `TaskUpdate: 8`
  - `Read: /Users/mykola/repos/surrogate-modelling/test/runtests.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/damping_tests.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/log_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/runtests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -8`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -B8 "ERROR" | head -30`

**Claude:** One test errored under `Pkg.test` (it passed interactively). Getting the error detail:
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -B3 -A25 "integration: NGMP through a Deterministic Log node: Error" | head -40`

**Claude:** `@model` can't be defined inside a `@testset` (macro-generated function ends up in the wrong scope). Moving it to file top level:
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/log_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/log_rule_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -12`
  - `Bash: julia --project=. -e ' using YAML, JLD2 import ProbabilisticEnsembling ROOT = "/Users/mykola/repos/surrogate-modelling" spec = ProbabilisticEnsembling._parse_sp`

**Claude:** While the cache builds, I'll generate UUIDs and write the notebook:
  - `TaskUpdate: 8`
  - `TaskUpdate: 9`
  - `Bash: julia -e 'using UUIDs; foreach(_ -> println(uuid4()), 1:24)'`

**Claude:** Let me read the prediction/metrics part of the surrogate reference to port it faithfully:
  - `Bash: sed -n '150,260p' /Users/mykola/repos/surrogate-modelling/dynamic_ngmp_surrogate.jl`
  - `Bash: sed -n '76,112p' /Users/mykola/repos/surrogate-modelling/dynamic_ngmp_surrogate.jl; grep -n "LowRankMeta" ../probabilistic_ensemble_forecasting/src/low_rank_sof`
  - `Bash: tail -5 /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bwpm4ehex.output; ls -la notebooks/vmp_vs_ngm`

**Claude:** Cache is ready (7 forecasters, 3446 val/test points, 65-dim features). Writing the notebook:
  - `Write: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`

**Claude:** Now a fast smoke test of the full pipeline (small `n_obs`) before running the notebook at full size:
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/dynamic_smoke.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/dynamic_smoke.jl 2>&1 | grep -vE "^\s*(\[|`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Bash: cd /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp && julia Dynamic_VMP_vs_NGMP.jl 2>&1 | grep -vE "^\s*(\[|┌|└|│|✓|Precompiling|Activating)" | ta`
  - `ToolSearch: select:Monitor`
  - `Bash: tail -5 /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bcb15jo84.output 2>/dev/null; echo "---"; ps`

**Claude:** While the notebook runs, let me update the project memory with the new state:
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-native-implementation.md`
  - `Bash: grep -q "ngmp-native-implementation" /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/MEMORY.md || cat /Users/mykola/.claude/projec`
  - `Bash: cat >> /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/MEMORY.md << 'EOF' - [NGMP native implementation](ngmp-native-implementatio`
  - `Bash: sleep 90; ps aux | grep "Dynamic_VMP" | grep -v grep | awk '{print $3, $4}' | head -1; wc -c /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a70`

**Claude:** All the code work is done and tested — I'm now just waiting on the full-size notebook run (n_obs = 500, 20 iterations, both arms + test-set prediction), which is running in the background and will notify me when it finishes. Status so far:

- **Generic damping**: `NGMPEdgeState` now stores the previous message as a distribution with its natural-parameter vector; `apply_damping!` does the heavy-ball update in η-space (your "product of messages" — exactly equivalent for β = 0) and works for any family with an unchecked bridge (Normal + Gamma provided). The unchecked bridges matter: ExponentialFamily's checked conversions throw on improper sites, which NGMP messages legitimately are mid-iteration.
- **Log-node NGMP rules**: `Log(:out)` projects the pulled-back LogGamma at q(z) → damped Gaussian; `Log(:in)` projects the pushed-forward LogNormal at q(γ) via the Gamma inverse-Fisher map → damped Gamma. Both verified against closed forms and the `project_lognormal_to_gamma` reference.
- **Tests**: 226 pass, including a toy integration test of NGMP on the Deterministic `Log` node (first such use — works, fires once per edge per iteration).
- **Smoke run** (n_obs = 60): both arms converge; NGMP arm ran ~6× faster than the ProjectedTo arm since there's no per-marginal Manopt optimization.

**You:** Can you not speed the inference using src/moment_form.jl on the w edge, because to my understanding you are converting into mean-form in every rule call can you check it until the notebook runs?

**Claude:** Good question — let me look at `moment_form.jl` and what actually happens on the `w` edge in our notebook models while the run finishes.
  - `Read: /Users/mykola/repos/surrogate-modelling/src/moment_form.jl`
  - `Bash: grep -rn "MomentForm" /Users/mykola/repos/surrogate-modelling --include="*.jl" -l | grep -v moment_form.jl; echo ---; grep -n "MomentForm" /Users/mykola/repos/s`
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/momentform_ab.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/momentform_ab.jl 2>&1 | grep -vE "^\s*(\[|`
  - `ToolSearch: select:TaskStop`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Bash: cd /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp && julia Dynamic_VMP_vs_NGMP.jl 2>&1 | grep -vE "^\s*(\[ Info: Loading|┌|└|│|✓|Precompiling|Act`

**Claude:** While the rerun goes, let me record this in memory — then report:
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-native-implementation.md`

**Claude:** Checked, confirmed, and applied. Here's the full picture:

**Your understanding is exactly right.** Without the constraint, `q(w)` is a `MvNormalWeightedMeanPrecision` — the message product on the `w` edge naturally accumulates in information form (prior message × 500 softdot messages). The softdot rules toward `z` and `τ` each read `mean`/`cov` of `q(w)`, and on an information-form marginal that triggers a fresh 65×65 solve **per rule call**: 7 forecasters × 500 obs × 2 rules × 20 iterations ≈ 140k Choleskys. `MomentForm` (with `FormConstraintCheckLast`) converts the marginal to `MvNormalMeanCovariance` **once per marginal update** instead — ~140 conversions total.

**A/B result** (n_obs = 120, 10 iterations, NGMP arm): 41.2s → 5.5s, a **7.4× speedup**, with posteriors identical to numerical noise (max |ΔE[w]| ≈ 8e-11, |ΔE[τ]| ≈ 1e-12, |ΔE[β]| ≈ 1e-15) — it's purely a representation change, same fixed point.

I've added `q(w)::MomentForm()` to **both** arms' constraint blocks in the notebook (fairness — the VMP arm pays the same per-call conversion tax) with a comment explaining why, and restarted the full notebook run. Two other notes:

- The previous (slow-path) full run had already completed with exit 0, so the notebook is verified end-to-end; this rerun is to produce the final numbers with the fast path, plus an `@info` line in the metrics cell so the test-set comparison lands in the log.
- I left the prediction-time `inner_z` model without `MomentForm`: there `q(w, z)` is a structured joint cluster, so the `w` variable marginal isn't what the rules consume, and it matches the `dyn_predict` reference from `dynamic_ngmp_surrogate.jl` verbatim.

I'll report the final VMP-vs-NGMP metrics when the run notifies.

**You:** <task-notification>
<task-id>b93b151bq</task-id>
<tool-use-id>toolu_01R5SBtj3wBTnfeq7C3MFyY3</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/b93b151bq.output</output-file>
<status>completed</status>
<summary>Background command "Rerun full notebook with MomentForm and metric prints" completed (exit code 0)</summary>
</task-notification>
  - `Bash: cat /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/b93b151bq.output`
  - `Bash: cd /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp && julia Dynamic_VMP_vs_NGMP.jl 2>&1 | tr "\r" "\n" | grep -vE "Progress|Activating|Loading exp`

**Claude:** Rerunning now with the metrics preserved (about 4–5 minutes: the VMP arm dominates at ~1:36, the NGMP arm takes ~2s, plus the two full-test-set prediction sweeps). I'll report the final comparison table when it lands.

**You:** <task-notification>
<task-id>bbqvjck4m</task-id>
<tool-use-id>toolu_0113biHnXCaUymyBiiupqUtw</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bbqvjck4m.output</output-file>
<status>completed</status>
<summary>Background command "Rerun notebook keeping the metrics output" completed (exit code 0)</summary>
</task-notification>
  - `Bash: grep -A6 "test-set predictive" /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bbqvjck4m.output | hea`
  - `TaskUpdate: 9`

**Claude:** Done — the full run completed and the comparison is in. **NGMP beats VMP on every uncertainty metric while matching it on point accuracy, and its training arm runs ~50× faster** (2s vs 96s for 500 obs × 20 iterations).

Test-set predictive on all 3446 ETTh1 h96 test points (trained on 500 validation obs, identical factorization `q(w)q(z,γ)q(τ)q(β)`, α = 0.2, κ = 1):

| metric | VMP (ProjectedTo) | NGMP (native) |
|---|---|---|
| MAE | **0.3184** | 0.3275 |
| RMSE | **0.3987** | 0.4081 |
| mean log-lik | −0.876 | **−0.681** |
| LL std | 1.743 | **1.262** |
| cov95 | 0.763 | **0.829** |
| pinball (0.1, 0.9) | 0.264 | **0.204** |

The pattern matches the paper's story: point forecasts are essentially tied (VMP marginally better MAE/RMSE), but NGMP's predictive distributions are better calibrated — higher and less volatile log-likelihood, coverage closer to nominal 95%, better quantile loss. The speed difference comes from where the projection lives: VMP runs a Manopt descent per marginal per iteration to project `q(z)` and `q(γ)`, while NGMP's messages are one closed-form Williams product + inverse-Fisher map each, so the graph stays conjugate throughout.

Everything delivered this round:

- **`src/NaturalGradientMP/damping.jl`** — family-generic damping: `NGMPEdgeState` stores the previous message as a distribution + natural-parameter vector; `apply_damping!` does the heavy-ball update in η-space (your "product of powered messages") through unchecked bridges, since improper sites would throw in ExponentialFamily's checked conversions. Old Gaussian behavior preserved exactly.
- **`src/nodes/log/rules/natural_gradient.jl`** + a Gamma `project` method in `src/tangent_projections/gamma.jl` — NGMP rules for PE's `Log` node toward both `z` (LogGamma → damped Gaussian) and `γ` (LogNormal → Gamma inverse-Fisher → damped Gamma).
- **`q(w)::MomentForm()`** in both arms per your suggestion — verified 7.4× A/B speedup with posteriors identical to ~1e-11.
- **Tests**: 226 pass, including Gamma damping recursion replays, improper-site safety, both Log rules vs closed forms, and a Deterministic-node integration test.
- **`notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`** — runs end-to-end (verified twice), data cached in `dynamic_etth1_h96_cache.jld2`; knobs (`n_obs`, `iterations`, `damping`, `κ`) are in one cell. To push further toward the surrogate reference numbers, the natural next step is raising `n_obs` toward the full 3446 — with MomentForm the NGMP arm should stay well under a minute.

**You:** where is this cache is created notebooks/vmp_vs_ngmp/dynamic_etth1_h96_cache.jld2?

**Claude:** It lives next to the notebook: `/Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/dynamic_etth1_h96_cache.jld2` (~4.8 MB).

Two things wrote/use it:

1. **Originally created** by a one-off script I ran before writing the notebook (to pre-warm it), which called `ProbabilisticEnsembling.before_rxinfer(spec)` under `cd(ROOT)` and saved the six arrays.
2. **The notebook itself owns the logic** — the data cell (`48eaeab6`, `notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl:76-93`) does:

```julia
cache_path = joinpath(@__DIR__, "dynamic_etth1_h96_cache.jld2")
if !isfile(cache_path)
    prep = cd(() -> ProbabilisticEnsembling.before_rxinfer(spec), ROOT)
    jldsave(cache_path; y_val = prep[1], y_test = prep[2],
            predictions_val = prep[3], predictions_test = prep[4],
            features_val = prep[5], features_test = prep[6])
end
dcache = load(cache_path)
```

So if you delete the file, the next notebook run regenerates it by loading the 5 expert models + the VAE from `models/`, evaluating them on the val/test windows of `data/ETTh1.csv` (that's the expensive step, a few minutes); every subsequent run just `load`s the JLD2. The `cd(ROOT)` is required because PE resolves `data/` and `models/` relative to the working directory. Contents: `y_val`, `y_test` (3446 each), `predictions_val/test` (7×3446), `features_val/test` (3446 vectors of length 65).

One thing to decide: it's a generated binary sitting in the repo — you may want to add `notebooks/vmp_vs_ngmp/*.jld2` to `.gitignore` so it doesn't get committed.

**You:** okay now you can do one more trick in this notebook to see one more thing techincally the vmp constraints in @constraints function dynamic_ngmp_constraints()
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β) can be reduced to q(w, z, γ, β) do you see why? Can you show how the performance improves if we will do that basically we will have VMP (CVI)/NGMP/NGMP with relaxed constraints?

**Claude:** Interesting move — let me make sure I read the relaxation right and check what rules exist before planning. Quick look at the GammaShapeRate and softdot rule coverage:
  - `Bash: ls ~/.julia/packages/ReactiveMP/4saqY/src/rules/gamma_shape_rate/ 2>/dev/null; echo ---; grep -n "@rule" ~/.julia/packages/ReactiveMP/4saqY/src/rules/gamma_shap`

**You:** [Request interrupted by user]

**You:** but before you start check on small subset like 60 points because maybe LowRankMeta normals can not be properly multiplied in this way however meta exists

**Claude:** No response requested.
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`

**You:** Look so much implemented cool do you remember how the method works because I have a last task for you to re-write this notebook from the gaussian surrogate model to the tangential projection unforneatly it will not be closed form like with Log but messages are prepared for you (bp messages I mean) here src/expressions

**Claude:** Yes, I remember the method: the NGMP message toward an edge is the tangent projection of the exact BP log-message at the receiving edge's own marginal — η = ∇_μ E_{q}[ℓ], damped heavy-ball in natural parameters. When E_q[ℓ] has a closed Williams product (PoissonExp, Log) we project exactly; when it doesn't, we fall back to the **second-order delta projection**: expand ℓ to a touching quadratic at the marginal's mean via `DerivativeEnhancedFunction` and use the exact Gamma/Normal raw-moment covariances — that's the `project_to_gamma`/`project_to_normal` machinery, and `src/expressions` holds the exact BP messages (`NormalPrecisionMessage` toward τ, Student-t toward the mean) with their analytic derivatives. That's exactly the non-closed-form case you're pointing at.

Let me read the notebook and the prepared messages:
  - `Read: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`

**You:** [Request interrupted by user]

**You:** continue
  - `Read: /Users/mykola/repos/surrogate-modelling/src/expressions/normal_precision_message.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/expressions/student_t_message.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`
  - `ExitPlanMode: # Third notebook arm: NGMP with relaxed factorization q(w,z,γ,β)q(τ) ## Context The dynamic notebook currently compares VMP-CVI vs NGMP under the same factoriza`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Bash: cat ~/.julia/packages/ReactiveMP/4saqY/src/rules/softdot/marginals.jl; echo "---GAMMA-RULE---"; cat ~/.julia/packages/ReactiveMP/4saqY/src/rules/softdot/gamma.j`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `AskUserQuestion`

**Claude:** Got it — dynamic only. Restoring the relaxed-arm plan to the file:
  - `Read: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `ExitPlanMode: # Native NG-BP for the Normal(mean, precision) notebook — second-order tangent projections ## Context `notebooks/gaussian_surrogate.jl` is the textbook model `x`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Read: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `TaskCreate: Add NormalMeanPrecision NGMP rules (τ and μ edges)`
  - `TaskCreate: Tests for the new rules + integration vs exact grid`
  - `TaskCreate: Rewrite gaussian_surrogate.jl notebook with native NG-BP`
  - `TaskUpdate: 10`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `ExitPlanMode: # Third notebook arm: NGMP with relaxed factorization q(w,z,γ)q(τ)q(β) ## Context The dynamic notebook compares VMP-CVI vs NGMP under `q(w)q(z,γ)q(τ)q(β)`. The`

**You:** [Request interrupted by user for tool use]

**Claude:** Now the sanity check — both rules via `@call_rule` against direct projection calls, and the end-to-end toy vs the exact grid posterior:
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl 2>&1 | grep -vE "^\s*(\[|┌|└`

**You:** any ideas how we can use whole data for q(w, z)? maybe filtering over observations? maybe there is a trick to actually do that?
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl 2>&1 | grep -B5 "Stacktrace"`
  - `AskUserQuestion`
  - `Edit: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Edit: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Edit: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl`
  - `ExitPlanMode: # Third notebook arm: NGMP with relaxed factorization q(w,z,γ)q(τ)q(β) ## Context The dynamic notebook compares VMP-CVI vs NGMP under `q(w)q(z,γ)q(τ)q(β)`. The`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl 2>&1 | grep -vE "^\s*(\[|┌|└`
  - `TaskCreate: Run staged 60-point gate for relaxed arm`
  - `TaskCreate: Full-data batch attempt for NGMP arms`
  - `TaskCreate: Add relaxed arm to notebook and run 3-arm comparison`
  - `Bash: ls src/nodes/normal_mean_precision/rules/ 2>/dev/null && git -C . status --short src/ | head -10`
  - `TaskUpdate: 13`
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/relaxed_gate.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl 2>&1 | grep -E "τ rule|μ rul`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/relaxed_gate.jl 2>&1 | grep -vE "^\s*(\[|┌`
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_debug.jl`
  - `Bash: grep -n "@rule\|@marginalrule" ../probabilistic_ensemble_forecasting/src/low_rank_softdot.jl`
  - `Bash: grep -rn "LowRankMeta" ../probabilistic_ensemble_forecasting/src/*.jl | grep -v "model_zoo" | head; echo ---; grep -rln "meta::LowRankMeta\|meta::Union{Nothing,`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_debug.jl 2>&1 | grep -vE "^\s*(\[|┌|└`
  - `Bash: grep -n "@rule\|@marginalrule" ../probabilistic_ensemble_forecasting/src/low_rank_softdot/structured.jl ../probabilistic_ensemble_forecasting/src/low_rank_softd`
  - `Read: /Users/mykola/repos/probabilistic_ensemble_forecasting/src/low_rank_softdot/structured.jl`
  - `Bash: julia -e ' include("/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/relaxed_gate.jl_header.jl")`
  - `Bash: cd /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad && julia -e ' using Pkg; Pkg.activate("/Users`

**Claude:** N=5 converges cleanly with message initialization. Now the full three-way comparison at N=1 and N=5:
  - `Edit: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/ngbp_smoke.jl 2>&1 | grep -E "rule|N = |x`
  - `TaskUpdate: 13`
  - `TaskUpdate: 14`
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/full_three_arm.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/full_three_arm.jl 2>&1 | grep -vE "^\s*(\[`

**Claude:** While the full-data validation runs (~15–20 min), let me wire the third arm into the notebook:
  - `Bash: julia -e 'using UUIDs; foreach(_ -> println(uuid4()), 1:5)'`
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/runtests.jl`
  - `Bash: grep -n "StableRNG" test/*.jl test/ngmp/*.jl Project.toml | head; grep -A3 "\[extras\]" Project.toml`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -6`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`

**Claude:** Now the intro, discussion, and the cell-order footer:
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`

**Claude:** Notebook is wired. Checking the full-data run progress:
  - `Bash: tr "\r" "\n" < /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/b2mggm1ys.output | grep -vE "Progress|`
  - `Bash: cd notebooks/vmp_vs_ngmp && grep -c "^# ╔═╡ [0-9a-f]" Dynamic_VMP_vs_NGMP.jl && grep -A40 "Cell order:" Dynamic_VMP_vs_NGMP.jl | grep -cE "^# (╠═|╟─)" && julia`

**Claude:** Notebook is consistent (27 cells, footer matches, parses cleanly) and now trains all three arms on the **full 3446-observation validation set**. The three-arm validation script is still running in the background (CVI arm extrapolates to ~11 min of its ~15 min total); when it completes I'll report the full-data comparison and, if it's clean, do the final notebook run to regenerate the displayed metrics.

Status of the gate findings that shaped this: relaxed arm uses **plain softdot** (LowRankMeta's structured rules run but disagree with stock structured rules by up to 66% on E[τ] — your "can't be properly multiplied" suspicion, now documented in the notebook md); no MomentForm analogue needed for the joint `q(z,w)` (consumed once per node per iteration — no redundant conversions); full batch confirmed feasible so no ADF fallback needed.

**Claude:** All 266 tests pass. Now the notebook rewrite — generating UUIDs and writing:
  - `TaskUpdate: 10`
  - `TaskUpdate: 11`
  - `TaskUpdate: 12`
  - `Bash: julia -e 'using UUIDs; foreach(_ -> println(uuid4()), 1:8)'`
  - `Write: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Bash: julia notebooks/gaussian_surrogate.jl 2>&1 | grep -vE "^\s*(\[|┌|└|│|✓|Precompiling|Activating)" | tail -8`

**Claude:** Clean run (asserts inside the notebook passed). Let me pull the actual numbers for the seeded data to make sure the discussion text matches reality:
  - `Bash: julia -e ' using Pkg; Pkg.activate(".") include("notebooks/gaussian_surrogate.jl") println("ys = ", round.(ys, digits = 3)) println(" E[x] V[x] E[τ] V[τ]") prin`
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-native-implementation.md`
  - `TaskUpdate: 12`

**Claude:** Done — `notebooks/gaussian_surrogate.jl` now runs Method 3 natively, and the whole suite (266 tests) is green. Summary of what was built:

**New rules** (`src/nodes/normal_mean_precision/rules/natural_gradient.jl`): NGMP messages for the standard `NormalMeanPrecision` node with both location and precision unknown. Since neither exact BP message has a closed-form Williams product, both rules use the **second-order tangent projection** at the receiving marginal — the prepared expressions did all the heavy lifting:

- toward τ: `NormalPrecisionMessage(y, m̃, ṽ)` (= 𝒩(y | m̃, ṽ+τ⁻¹)) → `project_to_gamma` → damped Gamma site;
- toward x: `StudentTMessage(y, ã, b̃)` (∝ (2b̃+(x−y)²)^−(ã+½)) → `project_to_normal` → damped Gaussian site.

**Two real findings surfaced during verification** (both now documented in the notebook and memory):

1. **Loopy start-up deadlock**: for N ≥ 2 the shared x and τ edges make every NGMP message wait on the equality-chain product of the *other* nodes' messages — nothing ever fires. Fix: `@initialization` must seed **messages** (`μ(x)`, `μ(τ)`), not just marginals. N=1 masked this because the inbound message is just the prior.
2. **Convex-tail initialization**: the Student-t log-message is concave only in its bulk ((m−y)² < 2b̃); seeding `q(x)` at N(0,1) with data near 10 expands in the convex tail, producing negative-precision sites that flip the vague-prior marginal improper. Seeding at the data mean fixes it — damping and init only, no guards, per your convention.

**Results** (seeded, N = 5, in the notebook's moments table): all three x-means agree at 10.238; **NG-BP's x-variance is closest to exact** (0.0887 vs exact 0.1064; VMP overconfident at 0.0709); τ shows the expected loopy-regime degradation (E[τ] = 2.13 vs exact 2.82, while at **N = 1 NG-BP recovers the exact τ posterior** and VMP doesn't — I verified both regimes and the discussion cell states this honestly, matching your paper's N-analysis).

**Tests**: `test/ngmp/normal_mean_precision_rule_tests.jl` — both rules vs direct projection calls, damped η-replay on the τ edge, and a deterministic N=3 loopy integration test against the exact grid posterior (ports `exact_posterior` as the reference). 259 NGMP tests + 7 projection tests pass.

The notebook keeps your original exact-grid and VMP cells untouched (data cell now seeded for reproducibility), adds the NG-BP section with an explainer of where the second-order projection differs from the closed-form Log/Poisson cases, overlays all three posteriors on the existing plots, and ends with the moments table.

**You:** <task-notification>
<task-id>b2mggm1ys</task-id>
<tool-use-id>toolu_01X3i9WYJBBnPfewMEqJoZAa</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/b2mggm1ys.output</output-file>
<status>completed</status>
<summary>Background command "Run three-arm full-data validation (background, ~20 min)" completed (exit code 0)</summary>
</task-notification>
  - `Bash: tr "\r" "\n" < /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/b2mggm1ys.output | grep -vE "Progress|`

**Claude:** Full-data validation is in — kicking off the final notebook run to confirm the file itself, then the summary:
  - `TaskUpdate: 14`
  - `TaskUpdate: 15`
  - `Bash: julia Dynamic_VMP_vs_NGMP.jl 2>&1 | tr "\r" "\n" | grep -vE "Progress|Activating|Loading expert" | tail -15`
  - `Bash: pwd; head -3 /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bl5u8a5ee.output 2>/dev/null; ps aux | g`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`

**Claude:** Discussion cell updated with the τ-trap caveat (the running process already read the file into memory, so the in-flight run is unaffected — and the changed cell is markdown-only anyway). Waiting on the final notebook run (~45 min remaining); I'll confirm its numbers match the validation script when it completes.

**You:** <task-notification>
<task-id>bl5u8a5ee</task-id>
<tool-use-id>toolu_014kV17Qc59dLrT2hU66qYqc</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bl5u8a5ee.output</output-file>
<status>completed</status>
<summary>Background command "Final full-data 3-arm notebook run (~50 min background)" completed (exit code 0)</summary>
</task-notification>
  - `Bash: tr "\r" "\n" < /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bl5u8a5ee.output | grep -vE "Progress|`

**Claude:** The notebook's own run reproduces the validation numbers to full precision. Wrapping up with a memory update:
  - `TaskUpdate: 15`
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/relaxed-factorization-results.md`
  - `Bash: grep -q "relaxed-factorization-results" /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/MEMORY.md || cat >> /Users/mykola/.claude/`

**Claude:** Done — the three-arm comparison is complete, verified twice (standalone script + the notebook itself, numbers identical to full precision).

**Final results** (full 3446-obs ETTh1 h96 validation, 20 iterations, α = 0.2, β = 0):

| method | MAE | RMSE | mean-LL | cov95 | train |
|---|---|---|---|---|---|
| VMP-CVI, `q(w)q(z,γ)q(τ)q(β)` | **0.2814** | **0.3525** | −0.4197 | 0.986 | 34 min |
| NGMP, same factorization | 0.2894 | 0.3627 | −0.4514 | 0.987 | **39 s** |
| NGMP relaxed, `q(w,z,γ)q(τ)q(β)` | 0.2820 | 0.3538 | **−0.3915** | **0.937** | 13 min |

What landed in `notebooks/vmp_vs_ngmp/Dynamic_VMP_vs_NGMP.jl`:

- The **relaxed third arm** — plain `softdot` (structured `q(w,z)` via stock ReactiveMP rules), same NGMP `Log`-link, constraints reduced to `q(w,z,γ)q(τ)q(β)` + `q(w)::MomentForm()`. Full data (`n_obs = length(y_val)`), all three arms in the plots, table, and the `@info` metrics line.
- An md cell documenting the two gate findings: **LowRankMeta's structured rules run but disagree with the stock rules** (E[τ] off by up to 66% — your "meta exists but can't multiply properly" suspicion, confirmed empirically), and why no `MomentForm` analogue is needed for the joint `q(z,w)` (consumed once per node per iteration — no redundant conversions).
- A discussion cell that reports the **τ-trap honestly**: under structured `q(w,z)`, E[τ] locks at the prior mean (~1020) for all seven forecasters — yet the relaxed arm is still the best-calibrated because `E[β]` takes over the noise-floor role.

The headline for the paper's narrative: the factorization split `q(w)‖q(z)` existed only to keep CVI tractable; NGMP removes that necessity, and on real data the relaxed family it enables gives the best log-likelihood and the closest-to-nominal coverage of the three — at 2.7× less compute than the CVI baseline. Two natural follow-ups, both parked deliberately: the γ–β merge (needs second-order NGMP rules for GammaShapeRate's compound messages — the machinery for it, `DerivativeEnhancedFunction` + `project_to_gamma`, already exists), and unlocking `q(τ)` via the `:marginal`-scatter update.

**You:** could be that our tangential projection for gamma is just not good enough because x posterior is actually better with more data, it could be we just using bad approximation for @rule NormalMeanPrecision(:τ, NaturalGradientMessage) (m_μ::UnivariateNormalDistributionsFamily, q_out::PointMass, q_τ::GammaDistributionsFamily, meta::NGMPEdgeState) = begin
    m̃, ṽ = mean_var(m_μ)
    exact = NormalPrecisionMessage(mean(q_out), m̃, ṽ)
    Δa, Δb = project_to_gamma(exact, convert(ExponentialFamilyDistribution, q_τ))
    site = ExponentialFamilyDistribution(Distributions.Gamma, [Δa, -Δb], nothing, nothing)
    return NaturalGradientMP.apply_damping!(meta, site)
end, also can you implement it as src/tangent_projections/closed_form_tangent.jl tangent? any ideas how to make it better?

**You:** why did perforamce of NGMP did change it had better NLL before?

**You:** why did perforamce of NGMP did change it had better NLL before? or is it another horizon?

**Claude:** Your hypothesis is well-founded, and I can pin down exactly why. The current τ-rule uses the **second-order delta projection**: it replaces `Cov_q[T, ℓ]` with the covariance against the *touching quadratic* of ℓ at the Gamma mean. That's exact only if ℓ were quadratic in τ — but `log 𝒩(y | m̃, ṽ + τ⁻¹)` is strongly non-quadratic (the log-determinant saturates for large τ), so for a **wide** q(τ) (small shape = small N) the quadratic is integrated over regions where it's badly wrong → biased (Δa, Δb) → biased fixed point. As N grows, q(τ) concentrates (a ≈ N/2), the delta method converges, and only the x-side benefit remains — matching exactly what you observed.

A fully closed-form Williams product doesn't exist here (`E[log(1+ṽτ)]` and its covariances under Gamma have no elementary form), but an **exact-to-machine-precision** one does: generalized **Gauss–Laguerre quadrature** — the Gamma(a,b) expectation is ∫f(u/b)u^{a−1}e^{−u}du/Γ(a), which is exactly the `gausslaguerre(n, α)` weight with α = a−1, and `FastGaussQuadrature` is already a dependency. Let me verify the API, then plan:
  - `Bash: julia --project=. -e ' using FastGaussQuadrature, SpecialFunctions u, w = gausslaguerre(32, 1.5) # weight u^1.5 e^-u println("sum(w) = ", sum(w), " Γ(2.5) = ",`

**Claude:** Same horizon, same test set — the only thing that changed is the **training set size**: the earlier table was trained on the 500-observation subsample, this one on all 3446. Let me first check empirically whether NGMP's remaining 0.03-nat gap at full data is a convergence shortfall (damped α = 0.2 × 20 iterations) or a genuine fixed-point difference — mean-field NGMP is cheap enough to just run longer:

**You:** [Request interrupted by user]

**You:** okay

**Claude:** Right — to wrap the answer up: it's the **same horizon (h96) and the same 3446-point test set**; what changed between the two tables is only the *training* size — 500-observation subsample before, the full 3446 validation set now.

Why that flips the VMP-vs-NGMP ordering on mean-LL:

- **At 500 obs both arms were data-starved and far from their fixed points** — you can see it in the absolute numbers: mean-LL was −0.68…−0.88 there vs −0.39…−0.45 at full data. In that under-trained regime, VMP was badly overconfident: coverage 0.76 (far under nominal) and LL-std 1.74, meaning the mean-LL was dominated by a heavy tail of confidently-wrong test points. NGMP's damped updates (α = 0.2) are conservative by construction — messages only move 20% toward the target per sweep — so its predictives stayed broader, which is exactly what wins log-likelihood when neither model has enough data. The 500-point comparison was really measuring *who degrades more gracefully when undertrained*.
- **At 3446 both are well constrained and essentially converged**: LL-std drops to ~0.47–0.49 for both, both slightly *over*-cover (0.986), and the two methods — which target very similar fixed points under the identical factorization — land 0.03 nats apart, now in CVI's favor. That residual gap is either a genuinely small fixed-point difference (CVI projects *marginals*, NGMP projects *messages* — they coincide exactly only at stationarity of the same objective) or simply NGMP not fully settled at α = 0.2 × 20 iterations. If you ever want to pin that down, a 40-iteration mean-field NGMP run costs ~80 seconds; if the gap closes, it was convergence, not the method.

And the framing that holds at full data: mean-field NGMP reaches parity with CVI at ~52× less compute, and the relaxed-factorization arm — which only NGMP enables — beats both on likelihood (−0.3915) regardless.

**Claude:** The API works — machine precision on polynomial moments, ~1e-4 on log moments at n=32 (the log singularity converges slower; n=64 and centered statistics will handle it). Writing the plan:
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Read: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`

**You:** can we make │ NGMP relaxed faster somehow? don't write code only ideas where we can save computations?
  - `ExitPlanMode: # Quadrature-exact tangent projection for the Gamma/Normal edges ## Context The NG-BP τ-marginal degrades for N ≥ 2 in `notebooks/gaussian_surrogate.jl`, and th`
  - `TaskCreate: Implement Quadrature tangent projection + switch rules`
  - `TaskCreate: Quadrature tests + notebook diagnostic + four-way comparison`

**Claude:** Yes — the 773s has a lot of fat. Ranked by payoff-per-effort, with where the computation actually hides:

**1. Warm-start from the mean-field arm (cheapest, no new rules).** The mean-field NGMP arm costs 39s and lands close to the relaxed fixed point. Initialize `q(w), q(τ), q(β), q(z), q(γ)` (and even the damping states' η) from it, and the relaxed arm needs ~5 sweeps instead of 20. Combined with early stopping on marginal change instead of a fixed iteration count (the last ~8 sweeps barely move anything at α = 0.2), that's a ~3–4× cut for an afternoon of plumbing.

**2. Run forecasters in parallel — the graph already factorizes.** Nothing is shared across `i`: `w[i], τ[i], β[i], z[i,:], γ[i,:]` couple only through *observed* `y`. So the 7-forecaster graph is literally 7 independent inference problems — run 7 `infer` calls on 7 threads for ~7× wall-clock. (Applies to all three arms; CVI's 34 min would drop to ~5.)

**3. Kill the redundant info↔moment round-trips (the real per-node waste).** Per softdot node per iteration, the stock structured rules do something absurd: the inbound `w`-cavity message `m_x` arrives in *information form*, the `SoftDot(:y_x)` marginal rule calls `mean_cov(m_x)` (65×65 solve) and then `cholinv`s the result **back to the precision it started with**; the `:y` rule does `mean_cov(m_x)` *again* on the same message. A specialized marginal-rule method for weighted-mean-precision inputs reads ξ and W directly — zero solves — leaving exactly one unavoidable 66×66 `mean_cov` per node (in the τ rule). This is the same idea as your `MomentForm` catch, but at the message/rule level, and it's probably 2–3× on its own.

**4. Never materialize the 66-dim joint at all (the big algorithmic one).** The joint `q(z,w)` exists solely to feed the τ update, and τ only needs three scalars: `Vy`, `f'Σ_w f`, and the z–w cross term. The joint's precision has known structure (scalar + rank-1 coupling + dense w-block), so those scalars come from one solve against `f` via block elimination — no 66×66 assembly, no inversion. This is exactly what your `cavi_inner.jl` structured update does in closed form, and it was measured at ~594× vs the RxInfer inner solve. Fusing "marginal + τ-rule" into one custom structured rule imports that win natively.

**5. Share the w-solve across observations (cavity → marginal approximation).** All 3446 nodes of a forecaster need `f_j`-functionals of what is *almost* the same object — the cavity differs from the full `q(w)` by one rank-1 downdate, an O(1/n) difference. Projecting at the full marginal instead (which is what the paper does elsewhere anyway) means **one** 65×65 factorization per forecaster per iteration serves every node, and the equality chain stops recomputing 3446 distinct cavity products. Combined with #4 this is essentially the CAVI cost structure: O(n·d²) per sweep, which is the 594×-regime.

**6. Fix LowRankMeta-structured rather than avoiding it.** It was *designed* for exactly this (rank-1 w-messages, no dense products), and it runs — it just disagrees with the stock rules by up to 66% on E[τ], which smells like a bug in its product/marginal path rather than a fundamental limitation. Finding that bug gets you #3 and half of #4 in code that already exists.

One caveat before investing in 3–6: profile one relaxed sweep first. My reading of the rule code says the double conversions and joint materialization dominate, but a 10-minute `@profile` will tell you whether the equality chain or the marginal rule is actually the top of the flame graph — that decides between #3/#5 as the first structural fix. And note #1+#2 alone (~10–20× combined, zero new math) already bring the relaxed arm under a minute.
  - `TaskUpdate: 16`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`

**Claude:** Now switch the rules to the quadrature projection:
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`

**Claude:** Quick validation of the new projections before touching tests — cross-checks and the key question (does the τ fixed point improve at N=5?):
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, Distributions, Statistics import SurrogateModelling: NaturalGradientMP, NormalPrecisionMessage, Studen`

**You:** okay try to implement a new rule for softdot and see how much speed we will obtain 
3. Kill the redundant info↔moment round-trips (the real per-node waste). Per softdot node per iteration, the stock structured rules do something absurd: the inbound w-cavity message m_x arrives in information form, the SoftDot(:y_x) marginal rule calls mean_cov(m_x) (65×65 solve) and then cholinvs the result back to the precision it started with; the :y rule does mean_cov(m_x) again on the same message. A specialized marginal-rule method for weighted-mean-precision inputs reads ξ and W directly — zero solves — leaving exactly one unavoidable 66×66 mean_cov per node (in the τ rule). This is the same idea as your MomentForm catch, but at the message/rule level, and it's probably 2–3× on its own.

**You:** okay try to implement a new rule for softdot and see how much speed we will obtain 
3. Kill the redundant info↔moment round-trips (the real per-node waste). Per softdot node per iteration, the stock structured rules do something absurd: the inbound w-cavity message m_x arrives in information form, the SoftDot(:y_x) marginal rule calls mean_cov(m_x) (65×65 solve) and then cholinvs the result back to the precision it started with; the :y rule does mean_cov(m_x) again on the same message. A specialized marginal-rule method for weighted-mean-precision inputs reads ξ and W directly — zero solves — leaving exactly one unavoidable 66×66 mean_cov per node (in the τ rule). This is the same idea as your MomentForm catch, but at the message/rule level, and it's probably 2–3× on its own. implement and test it in Surrogate moddelling
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`

**Claude:** Default node counts matter for the trapezoid — let me validate accuracy vs brute force and ClosedForm at several `n`, including the large-shape case:
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, Distributions, Statistics import SurrogateModelling: NormalPrecisionMessage, project_to_gamma import E`

**Claude:** Direct implementation task — first a baseline benchmark (stock rules), then the specialized rules, tests, and the A/B. Baseline at n_obs = 500, 10 iterations:
  - `TaskCreate: Info-form softdot rules: implement, test, benchmark`
  - `Write: /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/softdot_bench.jl`
  - `TaskUpdate: 18`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/softdot_bench.jl 2>&1 | grep -vE "^\s*(\[|`
  - `Bash: sed -i '' 's/Quadrature(64)/Quadrature(128)/g' src/nodes/normal_mean_precision/rules/natural_gradient.jl && julia --project=. -e ' using SurrogateModelling, RxI`
  - `Bash: cd /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad && sed -i '' 's/r1 = run_relaxed()/res_a = ru`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -5`

**Claude:** Baseline captured: **26.53 s** (500 obs × 10 iters), with exact checksums for the correctness comparison. Now implementing the specialized rules:
  - `Bash: grep -n "LinearAlgebra" Project.toml || echo "NO LinearAlgebra dep"`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.add("LinearAlgebra")' 2>&1 | tail -2`

**Claude:** All 273 tests pass. Now the notebook — reading the current state first (it has your N=20 edit):
  - `Read: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/nodes/softdot/rules/structured_info_form.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Bash: julia /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/scratchpad/softdot_bench.jl 2>&1 | grep -vE "^\s*(\[|`
  - `TaskUpdate: 16`
  - `TaskUpdate: 17`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`

**Claude:** Now the end-to-end notebook run (with your N=20 setting):
  - `Bash: julia notebooks/gaussian_surrogate.jl 2>&1 | grep -vE "^\s*(\[|┌|└|│|✓|Precompiling|Activating)" | tail -6 && julia -e ' using Pkg; Pkg.activate(".") include("n`
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/runtests.jl`
  - `Bash: grep -n "StableRNG" test/ngmp/*.jl Project.toml | head -5; grep -A4 "\[extras\]" Project.toml`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -6`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-tau-loopy-regime.md`
  - `Read: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-tau-loopy-regime.md`
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/ngmp-tau-loopy-regime.md`
  - `Bash: sed -i '' 's|NGMP τ loopy regime](ngmp-tau-loopy-regime.md) — NGMP beats VMP on precision τ only at N=1; loop degrades it for N≥2 (location/scale asymmetry), ta`
  - `Bash: ls test/ngmp/ && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`

**Claude:** All green (274 + 7 tests, including the new softdot comparisons). Final measurement — the full-3446 relaxed arm that took 773s before:
  - `Bash: julia -e ' using Pkg; Pkg.activate("/Users/mykola/repos/surrogate-modelling") using RxInfer, ExponentialFamily, SurrogateModelling, ProbabilisticEnsembling, YAM`

**Claude:** Full-data measurement running (~5–7 min expected at the 2× rate). Summary of what's in place while it finishes:

- **`src/nodes/softdot/rules/structured_info_form.jl`** — two specializations that dispatch whenever `m_x` arrives in information form (which is always, under the structured cluster):
  - `SoftDot(:y_x)` marginal rule: reads `(ξ, Λ)` directly — **zero matrix solves**, versus the stock rule's `mean_cov(m_x)` (65×65 solve) followed by `cholinv` right back to the precision it started with;
  - `softdot(:y)` rule: one Cholesky + two triangular solves instead of a full `mean_cov` inverse.
- **A/B at 500 obs × 10 iters: 26.53s → 13.29s (2.0×)**, posteriors identical to ~1e-12, and the diagnostic counters prove both rules fire on every node call (105k invocations).
- **Tests**: new `softdot_info_form_tests.jl` pins both specializations against the stock rules on identical inputs (forcing stock dispatch via moment-form conversion) — full suite green, 281 tests.

As predicted, the remaining per-node O(d³) is now the single 66×66 `mean_cov` in the structured `softdot(:γ)` rule — that one needs the fused-τ-update idea (#4) to eliminate.
  - `TaskUpdate: 17`

**Claude:** Done — and **your hypothesis was correct**: the τ degradation was the second-order projection, not the method. Full picture:

**The mechanism, quantified.** The delta projection replaces `Cov_q[T, ℓ]` with the covariance against ℓ's touching quadratic at the Gamma mean. For a wide `q(τ)` that quadratic gets integrated where it's badly wrong: at `Gamma(1.5, 2)` the second-order rate increment is `Δb = 0.443` vs the exact `0.131` — off by **3.4×**. At concentrated marginals (shape ≳ 100) both agree — which is why the problem faded with more data.

**The implementation** (`Quadrature` strategy, as you asked, in the `TangentProjection` interface):

- `src/tangent_projections/closed_form_tangent.jl`: `Quadrature(n = 64)` strategy type (node count in the type domain) + a uniform log-message evaluator for both `Expression`-style messages and CFE/Distributions densities.
- **Gamma edge** (`gamma.jl`): my first attempt with generalized Gauss–Laguerre failed validation — it converges only algebraically against the `log τ` statistic and its weights NaN at large shape. The working method is a **log-space trapezoid**: in `s = log τ` the integrand `e^{as − be^s}` is smooth and decays double-exponentially, so the trapezoid converges geometrically for *any* shape — n=64 matches a 4-million-point brute-force reference to 6 decimals, and `Gamma(200, 100)` works fine.
- **Gaussian edge** (`normal.jl`): Gauss–Hermite; matches the closed-form `ClosedForm` projection to 1e-15.
- Both `NormalMeanPrecision` rules now use `Quadrature(128)`; the second-order `project_to_gamma`/`project_to_normal` remain as the cheap alternative and diagnostic baseline.

**The result** (seeded notebook data):

| | exact | NG-BP quad | NG-BP 2nd-order | VMP |
|---|---|---|---|---|
| N=5: E[τ], V[τ] | 2.819, 2.650 | **2.830, 2.390** | 2.13, 1.31 | 2.819, 2.271 |
| N=20: E[τ], V[τ] | 6.957, 4.610 | **6.969, 4.571** | — | 6.957, 4.400 |

NG-BP's τ now essentially matches the exact posterior at every N and reports the least-understated τ-variance. The notebook has a diagnostic cell showing the side-by-side (Δa, Δb) at a wide vs converged marginal, and the discussion is rewritten accordingly. All 281 tests pass (including cross-checks vs ClosedForm, brute force, and the delta-regime consistency tests).

**One scientific caveat I preserved**: your older `normal_ng_bp_*` sweep found genuine negative-shape increments *with a converged quadrature* — that pathology lives in the opposite regime (`v_cav ≫ 1/τ`, spread data), where the message's leading direction leaves the Gamma span entirely. It stands; it just wasn't what bit the native rules here. I amended the `ngmp-tau-loopy-regime` memory to record both regimes rather than overwriting your earlier verified findings — when τ misbehaves, check the projection-accuracy regime first, then the `v_cav/(1/τ)` ratio.

**Further ideas** (beyond what's implemented): a closed-form "second-order in log-space" projection (expand ℓ in `s = log τ` — all needed Gamma moments are polygamma-analytic, and it should inherit most of the quadrature's accuracy for free), and if the fixed point ever needs more than Gamma can express on the τ edge, a richer family (GIG/log-Normal) is the structural upgrade — the family-generic damping already supports adding bridges.

**You:** <task-notification>
<task-id>bgv6f56y9</task-id>
<tool-use-id>toolu_01AmRpMxiPZ5egidDM9jFpvn</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bgv6f56y9.output</output-file>
<status>completed</status>
<summary>Background command "Measure full-3446 relaxed arm with specialized rules" completed (exit code 0)</summary>
</task-notification>
  - `Bash: cat /private/tmp/claude-501/-Users-mykola-repos-surrogate-modelling/a700ca6b-a191-4815-8bd8-40418ca7e016/tasks/bgv6f56y9.output`
  - `TaskUpdate: 18`
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/relaxed-factorization-results.md`

**Claude:** Done — and it paid off more at scale than the small-size A/B suggested.

**Full-data relaxed arm: 772.8s → 223.0s, a 3.5× speedup** (the 500-obs A/B showed 2.0×; at 3446 observations the redundant per-cavity conversions weighed relatively more). Correctness is airtight on three counts: posteriors identical to the previous full run (E[β] matches to 4 significant digits, E[τ] the same τ-trap plateau), the unit tests pin both specializations against the stock rules on identical inputs, and the counters show both rules fired exactly `7 × 3446 × 20 = 482,440` times — every node, every iteration, no silent fallback to the stock path.

What was shipped:

- **`src/nodes/softdot/rules/structured_info_form.jl`** — two strictly-more-specific methods that dispatch when `m_x` is `MvNormalWeightedMeanPrecision` (always the case under the structured cluster):
  - `SoftDot(:y_x)` marginal rule: reads `(ξ, Λ)` off the message directly and assembles the joint in information form with **zero matrix solves** — eliminating the stock rule's absurd `mean_cov` → `cholinv`-back round-trip;
  - `softdot(:y)`: one Cholesky factorization + two triangular solves (`Λ⁻¹ξ` and `f'Λ⁻¹f`) instead of a full `mean_cov` inverse.
- **`test/ngmp/softdot_info_form_tests.jl`** — specialized-vs-stock equality on random inputs plus dispatch-counter assertions; full suite green (281 tests).
- The notebook needs no changes — the rules are global, so its relaxed arm inherits the speedup automatically (next run: ~3.7 min instead of ~13).

Updated arm costs at full data: mean-field NGMP 39s, **relaxed NGMP 223s**, VMP-CVI 2050s. The remaining per-node O(d³) in the relaxed arm is now exactly one 66-dim `mean_cov` per softdot node (in the structured `:γ` rule) — killing that requires the fused τ-update (idea #4: two solves against vectors via the joint's block structure, or the full CAVI closed form), which alone would likely bring the relaxed arm near the mean-field arm's cost. Warm-starting from the 39s mean-field fixed point and the 7-way per-forecaster parallelism remain on the table too if you want to push further.

**You:** why do we need _infoform_marginal_calls?

**Claude:** Purely as a **dispatch tripwire** — they're not needed for correctness, only for observability of a failure mode that is otherwise invisible.

The whole optimization rests on Julia picking the specialized method (`m_x::MvNormalWeightedMeanPrecision`) over the stock one (`m_x::Any`). If that ever stops happening — a ReactiveMP version changes what type the equality-chain product returns, someone changes the `w` prior type, a constraint change routes a moment-form message there — nothing breaks: Julia silently falls back to the stock rule, the posteriors stay **identical** (both paths compute the same thing), and the only symptom is that the 3.5× quietly evaporates. You can't detect that regression by checking results, and timing-based tests are flaky. The counters make it checkable: the benchmark printed `482,440 = 7 × 3446 × 20` (every node, every iteration — proof the fast path carried the whole run), and the unit tests assert the counter increments when an info-form message goes in.

Cost side: one `Ref` increment per rule call — nanoseconds against a 65-dim Cholesky, so performance is a non-issue. The legitimate objection is aesthetic/architectural: it's mutable global state in a hot path, and it's not thread-safe (harmless for a diagnostic under RxInfer's single-threaded inference, but still a data race if that ever changes).

If you'd rather keep the rules clean, two alternatives:

1. **Delete the counters and move the dispatch check into the test suite** via `Base.which` on the generated `ReactiveMP.rule`/`marginalrule` signature — asserts at test time that the specialized method is the one selected for these argument types. Slightly brittle to `@rule` macro internals, but zero runtime state.
2. **Keep them but demote to a comment-documented debug facility** — which is basically what they are now.

Happy to strip them and switch the tests to the `which`-based assertion if you prefer — say the word.

**You:** yes please do that this counters looks strange 1. Delete the counters and move the dispatch check into the test suite via Base.which on the generated ReactiveMP.rule/marginalrule signature — asserts at test time that the specialized method is the one selected for these argument types. Slightly brittle to @rule macro internals, but zero runtime state.
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/softdot/rules/structured_info_form.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/softdot/rules/structured_info_form.jl`

**Claude:** Now the test rewrite — counters out, method-table dispatch assertion in:
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -B10 "ERROR\|Fail\|Error" | head -30`

**Claude:** The `@rule` macro records its own source location, not our file — let me probe what's actually in the method metadata:
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily ms = filter(methods(ReactiveMP.rule)) do m occursin("MvNormalWeightedMeanPrecision",`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`

**You:** [Request interrupted by user for tool use]

**You:** Skip to where you left off (post 2) Skip to last reply Skip to top
Skip to main content

 Julia Programming Language

​


​
Hi everyone,

The registration is open and tickets are available now for JuliaCon Global 2026!

Please find more information on the ticketing page here:


JuliaCon Global 2026

JuliaCon Global 2026

JuliaCon Global 2026, Mainz
You can also head directly to our registration portal:


app.hi.events

JuliaCon Global 2026 | IANW e.V.

JuliaCon Global 2026: Registration and Ticketing Welcome to the registration page for JuliaCon Gl...
The early-bird rates are in effect until May 1, 2026. If you have any questions, please go through the FAQs, ask on the JuliaCon channel on Slack, or write to us at juliacon@julialang.org.

We look forward to seeing you in Mainz!

Best,
JuliaCon Global 2026 Organizing Committee
How to check what method is used by dispatch?
General Usage

154
views
 
 
Jul 2025
Jul 2025

​

post by lrnv on Jul 29, 2025
 
lrnv
Jul 2025
Hey,

Say I have an abstract type A and many subtypes B, C, D:

abstract type A end
struct B<:A end
struct C<:A end

should_return_one(::A) = 1
should_return_one(x::B) = # something else. 
# remark that, on purpose, I did not overwrote the method for x::C
Suppose now that for testing reasons, I want to ensure that the subtypes are implementing the method correctly. I am testing it like that :

objects = (B(), C()) # there are a lot more objects here. 
for o in objets 
    @test should_return_one(o) == 1
end
Now, if the method is very costly, testing the method for C is not worth it because it will dispatch to the generic version. Thus, I’d like to only test when object do indeed overwrite the generic method:

objects = (B(), C()) # there are a lot more objects here. 
for o in objets
    condition = # the call will not dispatch to should_return_one(::A)
    if condition
        @test should_return_one(o) == 1
    end
end
How can i write this condition ?


SOLVED

giordano
Jul 2025

​
Hopefully there are better ways to do this, but one option is to use which and check the (internal, private, unreliable, use at your own risk) field sig:

julia> using Test, InteractiveUtils

julia> abstract type A end

julia> struct B<:A end

julia> struct C<:A end

julia> should_return_one(::A) = 1
should_return_one (generic function with 2 methods)

julia> should_return_one(x::B) = 2
should_return_one (generic function with 2 methods)

julia> which(should_return_one, (B,)).sig
Tuple{typeof(should_return_one), B}

julia> which(should_return_one, (C,)).sig
Tuple{typeof(should_return_one), A}

read more

154
views
 
 
post by giordano on Jul 29, 2025
 
giordano
Jul 2025
Hopefully there are better ways to do this, but one option is to use which and check the (internal, private, unreliable, use at your own risk) field sig:

julia> using Test, InteractiveUtils

julia> abstract type A end

julia> struct B<:A end

julia> struct C<:A end

julia> should_return_one(::A) = 1
should_return_one (generic function with 2 methods)

julia> should_return_one(x::B) = 2
should_return_one (generic function with 2 methods)

julia> which(should_return_one, (B,)).sig
Tuple{typeof(should_return_one), B}

julia> which(should_return_one, (C,)).sig
Tuple{typeof(should_return_one), A}

post by lrnv on Jul 29, 2025
 
lrnv
Jul 2025
How, which sound nice indeed… What about simply which(should_return_one, (A,)) == which(should_return_one, (B,)) ? To avoid internals.


post by giordano on Jul 29, 2025
 
giordano
Jul 2025
I guess that’d work (I didn’t realise which accepts abstract types, but it’s documented to do so, that’s good)


 
Share

Bookmark

Flag

Reply

Normal
You will be notified if someone mentions your @name or replies to you.

Topic list, column headers with buttons are sortable.

Suggested

Related
Replies    Views    Activity
New Working Local Coding Agent for Julia 
General Usage
3    212    13h
Joovy - Faster compilation 
Tooling
9    349    2h
Mooncake.jl with variable sized input 
Specific Domains
autodiff
note
5    222    2h
Blog post on GPU ecosystem 
GPU
packages
blog-post
0    75    3h
Optimization.jl - Define the objective function and its gradient in the same function 
Optimization (Mathematical)
question
2    86    6h
There are 3 unread and 8 new topics remaining, or browse other topics in 
General Usage
Julia Discourse Board
The Julia programming language forum: discuss usage, development, packages, and community.
FAQ Terms of service About Privacy ❤️ Thanks to Discourse for their generous support
 this should help?
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily mm = filter(m -> occursin("MvNormalWeightedMeanPrecision", string(m.sig)) && occursi`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/softdot_info_form_tests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`

**Claude:** All green. Updating the memory note that referenced the counters, then done:
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/relaxed-factorization-results.md`

**Claude:** Done — counters are gone, and the dispatch guard now uses exactly the idiom from that Discourse thread.

**What changed:**

- `src/nodes/softdot/rules/structured_info_form.jl`: both `Ref` counters and their increments deleted — the rules are now pure functions with zero global state. The header comment explains that dispatch regressions are silent and points to the test guard.
- `test/ngmp/softdot_info_form_tests.jl`: the guard is now `Base.which(ReactiveMP.rule, argtypes)` resolved twice with identical argument types **except** the form of `m_x` — info-form vs moment-form — asserting the two resolve to **different methods**. If the specialization is ever deleted, shadowed by an upstream change, or stops matching the message type, both calls collapse onto the same stock `::Any` method and the test fails loudly. Same check for the `marginalrule`. No `.sig` string matching in the assertion (it's only `which` + method identity, per the `which(f, (B,)) != which(f, (A,))` trick from the thread).

One practical detail worth knowing from the intermediate failure: my first attempt located our methods by `m.file` — that doesn't work because `@rule`-generated methods record ReactiveMP's own `rule.jl` as their source file (the macro's location), not the file that invoked the macro. The `which`-with-concrete-argtypes approach sidesteps that entirely; the exact rule signatures it needs (10 args for `rule`, 8 for `marginalrule`, with `Message`/`Marginal` wrappers built via their 3-arg constructors) are probed from the live method table, so the test constructs them the same way the runtime does.

Full suite: **281 tests pass** (7 + 274). The 3.5× full-data speedup is unchanged — the counters were nanoseconds; this was purely about removing the global state.

**You:** we implemented TangentProjection{Quadrature} I wonder can we implement a new subpackage UncentedTransform and try to see how well it would work to replace our TangentsProjection{Quadrature} with UncentedTransform for the uncented transform implementation look into this package (base) ➜  ExpectationApproximations.jl git:(main) pwd
/Users/mykola/repos/ReactiveBayes/ExpectationApproximations.jl you can take implementation from there notebooks/gaussian_surrogate.jl because basically if I am using delta rule in this notebook my method NGMP works worse than VMP but with Quadratures it works better than VMP but I wonder maybe there is uncented transform middle ground? notebooks/gaussian_surrogate.jl. I wonder can you into NaturalGradientMessage struct a new subfield that is TangentProjection type which by default will be closed form but we can put there Quadratures, Uncented Transform and so on

**Claude:** New task: an **UnscentedTransform** projection strategy as a middle ground between the delta/closed-form and Quadrature, plus making `NaturalGradientMessage` carry the projection strategy as a field (defaulting to ClosedForm). I need to see the current state (the Quadrature work landed since I last looked), the ExpectationApproximations.jl UT implementation, and whether rule bodies can access the `vconstraint` instance. Exploring:
  - `Agent: Explore current tangent-projection state`
  - `Agent: Explore ExpectationApproximations.jl UT`
  - `Bash: sed -n '343,380p' ~/.julia/packages/ReactiveMP/4saqY/src/rule.jl; echo ---CALLRULE---; sed -n '650,665p' ~/.julia/packages/ReactiveMP/4saqY/src/rule.jl`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Read: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `Write: /Users/mykola/.claude/plans/5-dumping-and-momentum-quirky-starfish.md`
  - `ExitPlanMode: # UnscentedTransform projection strategy + projection-carrying NaturalGradientMessage ## Context The gaussian_surrogate notebook showed: NGMP with the **delta**`
  - `TaskCreate: Projection-carrying NaturalGradientMessage + NGMPDependencies`
  - `TaskCreate: UnscentedTransforms submodule + project methods`
  - `TaskCreate: Rules read projection; ClosedForm delta fallbacks; tests`
  - `TaskCreate: Five-way gaussian notebook comparison`
  - `TaskUpdate: 19`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/constraint.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/constraint.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/dependencies.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/dependencies.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/dependencies.jl`
  - `TaskUpdate: 19`
  - `TaskUpdate: 20`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/NaturalGradientMP.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/UnscentedTransforms/UnscentedTransforms.jl`

**Claude:** Now let me read the exact quadrature centering conventions to mirror them:
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`

**Claude:** Now the `project` methods and the ClosedForm delta fallbacks:
  - `Write: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Write: /Users/mykola/repos/surrogate-modelling/src/UnscentedTransforms/UnscentedTransforms.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`

**Claude:** Wiring into the module and adding `resolve_projection` + the ClosedForm delta fallbacks:
  - `Read: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`

**Claude:** Now the ClosedForm delta fallbacks and the rule rewiring:
  - `Bash: tail -5 src/tangent_projections/gamma.jl && tail -5 src/tangent_projections/normal.jl && cat src/nodes/log/rules/natural_gradient.jl | sed -n '20,35p'`
  - `Bash: cat >> src/tangent_projections/gamma.jl << 'EOF' """ project(TangentProjection(type = ClosedForm), q::GammaDistributionsFamily, f::Logpdf{<:NormalPrecisionMessa`
  - `TaskUpdate: 20`
  - `TaskUpdate: 21`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/log/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/log/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/poisson/rules/natural_gradient.jl`
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily, Distributions import ClosedFormExpectations: Logpdf import SurrogateModelling: Norm`

**Claude:** Name clash — **ReactiveMP already exports `UnscentedTransform`** (its Delta-node approximation). Let me see what it is; maybe I can bridge to it instead of colliding:
  - `Bash: grep -rn "UnscentedTransform\|struct Unscented" ~/.julia/packages/ReactiveMP/4saqY/src/approximations/unscented.jl | head -8; sed -n '1,40p' ~/.julia/packages/R`
  - `Bash: sed -n '40,68p' ~/.julia/packages/ReactiveMP/4saqY/src/approximations/unscented.jl; grep -n "getalpha\|getbeta\|getkappa" ~/.julia/packages/ReactiveMP/4saqY/src`
  - `Bash: grep -rn "const default_alpha\|const default_beta\|const default_kappa" ~/.julia/packages/ReactiveMP/4saqY/src/ | head -3; grep -n "getα\|getalpha" ~/.julia/pac`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/SurrogateModelling.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily, Distributions import ClosedFormExpectations: Logpdf import SurrogateModelling: Norm`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/UnscentedTransforms/UnscentedTransforms.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily, Distributions import ClosedFormExpectations: Logpdf import SurrogateModelling: Norm`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/unscented.jl`
  - `Bash: grep -n "s = log" src/tangent_projections/unscented.jl; grep -n "import SpecialFunctions\|trigamma" src/tangent_projections/gamma.jl | head -3`
  - `Bash: julia --project=. -e ' using SurrogateModelling, RxInfer, ExponentialFamily, Distributions import ClosedFormExpectations: Logpdf import SurrogateModelling: Norm`
  - `Read: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/normal_mean_precision_rule_tests.jl`

**Claude:** Now the UT test file:
  - `Write: /Users/mykola/repos/surrogate-modelling/test/ngmp/unscented_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/unscented_tests.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/runtests.jl`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -6`
  - `TaskUpdate: 21`
  - `TaskUpdate: 22`
  - `Read: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`

**Claude:** Now run the notebook and capture the actual five-way numbers before writing the verdict text:
  - `Bash: cd notebooks && julia -e ' include("gaussian_surrogate.jl") using Statistics, Distributions fmt(qx, qτ) = (round(mean(qx), digits=4), round(var(qx), digits=4),`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `TaskUpdate: 22`
  - `Write: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/unscented-tangent-projection.md`
  - `Bash: cat >> /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/MEMORY.md << 'EOF' - [Unscented tangent projection](unscented-tangent-proje`

**Claude:** Done — the unscented transform is in, validated, and the answer to your hypothesis is **yes, it's a genuine middle ground** (with one crucial twist discovered along the way).

**The headline numbers** (gaussian notebook, N = 2, exact E[τ] = 1.5):

| method | evals/message | E[τ] | V[τ] (exact: 1.5) |
|---|---|---|---|
| NG-BP delta | 2 derivatives | 1.086 | 0.699 |
| mean-field VMP | — | 1.4995 | 1.124 |
| **NG-BP unscented** | **3 points** | **1.549** | **1.317** |
| NG-BP quadrature | 128 nodes | 1.505 | 1.258 |

Delta ≪ VMP < UT ≈ quadrature — the UT kills the delta bias at 3 evaluations, and its V[τ] is actually the closest to exact of all approximations. At the single-projection level it's 10–25× more accurate than delta across every q(τ) width.

**The twist that made it work:** naive GenUT sigma points in τ-space (EA's approach + constrained resampling) were *worse than delta* at mid widths and NaN-prone at small shapes. The fix was building the moment-matched points **in log space** — the log-Gamma's central moments are exact polygamma expressions, and the exp-mapped points always stay in support. That mirrors how your trapezoid quadrature handles the same edge.

**What was built:**

- **`src/UnscentedTransforms/`** submodule — scaled UT + generalized (asymmetric, moment-matched) 3-point rules, ported from ExpectationApproximations.jl (no dependency on it). Deliberately *not* copying EA's α = 1e-3 default, which degenerates to a finite-difference delta; our defaults (α=1, β=0, κ=2) make the Gaussian-edge UT exactly 3-point Gauss–Hermite.
- **`NaturalGradientMessage{P}`** now carries the projection (default ClosedForm, which falls back to the delta for messages without an exact Williams product — so "delta" is now a selectable arm, not a hardcoded one). Selection: `NGMPDependencies(...; projection = TangentProjection(type = ...))`. Instances are self-callable so `@call_rule` accepts explicit projections.
- **Name-clash resolution:** ReactiveMP already exports `Unscented`/`UnscentedTransform`, so instead of colliding, `TangentProjection` bridges: `type = Unscented` → our GH(3) defaults, `type = Unscented(alpha=…, beta=…, kappa=…)` → parameters mapped verbatim (with a documented warning about ReactiveMP's near-delta defaults).
- **Notebook** now runs the five-way comparison (exact / VMP / delta / UT / quadrature) where the NG-BP arms differ only in the `projection` argument; the diagnostic table shows all three strategies on one τ-message.
- **Tests:** 312 pass — sigma-point identities, moment-matching exactness, UT ≡ Quadrature(3) on the Gaussian edge, the middle-ground assertion (UT strictly closer to quadrature truth than delta at every width), rule dispatch via explicit instances, and the updated NMP tests.

One honest caveat for the paper narrative: the notebook's UT slightly overshoots where quadrature is near-exact (1.549 vs 1.505), so quadrature remains the reference-quality choice — UT is the right default when message evaluations are the bottleneck (e.g. the dynamic model's 480k projections per run).

**You:** but should you not call Delta, DeltaApproximation not ClosedForm? closed form is like Poisson when we can use ClosedWilliamsProduct?

**Claude:** You're absolutely right — `ClosedForm` should mean *exact* (a `ClosedWilliamsProduct` exists, like Poisson/LogGamma/LogNormal), and silently downgrading it to a biased second-order expansion muddies the semantics. Making the delta an explicit, honestly-named strategy:
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/closed_form_tangent.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`
  - `Bash: grep -n "ClosedForm fallback\|project(::TangentProjection{<:ClosedForm}, q::_GammaProjectionPoint, f::Logpdf{<:NormalPrecisionMessage})\|project(::TangentProjec`
  - `Bash: pwd; ls src/ 2>/dev/null | head; ls /Users/mykola/repos/surrogate-modelling/src/tangent_projections/`

**Claude:** Shell was in `notebooks/`. Re-reading the fallback sections to reconcile before editing:
  - `Bash: cd /Users/mykola/repos/surrogate-modelling && grep -n "ClosedForm fallback\|DeltaApproximation\|Logpdf{<:NormalPrecisionMessage}\|Logpdf{<:StudentTMessage}" src`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/gamma.jl`
  - `Read: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/tangent_projections/normal.jl`

**Claude:** Now updating every place that relied on "ClosedForm = delta":
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/NaturalGradientMP/constraint.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/src/nodes/normal_mean_precision/rules/natural_gradient.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/notebooks/gaussian_surrogate.jl`
  - `Edit: /Users/mykola/repos/surrogate-modelling/test/ngmp/unscented_tests.jl`
  - `Edit: /Users/mykola/.claude/projects/-Users-mykola-repos-surrogate-modelling/memory/unscented-tangent-projection.md`
  - `Bash: julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`
  - `Bash: cd notebooks && julia -e ' include("gaussian_surrogate.jl") using Statistics, Distributions fmt(qx, qτ) = (round(mean(qx), digits=4), round(var(qx), digits=4),`
