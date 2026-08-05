# Recall History — vmp_vs_ngmp


## Session 3a8400f4 — 2026-07-06 16:33

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
