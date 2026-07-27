# Static RxInfer Acceleration Attempt: Detailed Postmortem

> **Archive notice:** The compiler and specialized executor described here are
> benchmark artifacts, not production backends. Historical paths in this
> document describe their locations during development; the artifacts now live
> under `benchmarks/rxinfer_compiler/`. The Rocket experiments described below
> were subsequently reset after review because their maintenance cost was not
> justified by the measured end-to-end benefit.

## Executive summary

The original goal was to build a general `backend=:compiled` for RxInfer that
would:

- Trace or lower an arbitrary supported RxInfer graph.
- Remove Rocket scheduling from profitable regions.
- Fuse repeated observation plates.
- Execute independent regions across Julia threads.
- Fall back safely to reactive execution for unsupported regions.
- Preserve requested marginals within the final tolerance of `1e-10`.

That goal was not completed.

What was accomplished falls into four categories:

1. **Generic runtime improvements that work**

   - Rocket readiness counters: approximately 7.1% faster reactive iteration
     waves.
   - RxInfer prepared-constraint caching: approximately 54.5% faster
     rematerialization.
   - ReactiveMP callback filtering: a large improvement when callbacks are
     installed.

2. **A generic fusion primitive that works in isolation**

   - ReactiveMP homogeneous bucket fusion: approximately 3.1x faster in its
     microbenchmark, with zero warm allocations.
   - It is not connected to RxInfer graph lowering.

3. **General compiled-backend attempts that were correct but too slow**

   - The most complete general component compiler reproduced the target
     exactly, but ran slower than reactive execution at scale and had
     catastrophic preparation costs.
   - It was removed according to the performance kill gate.

4. **A specialized executor for one model that is fast**

   - Approximately 4.1x faster than reactive execution for the specific
     prediction graph.
   - Approximately half the allocations.
   - It is handwritten around one graph topology and therefore is not a
     general compiler.

The central lesson is that eliminating individual Rocket operations is not
enough. A successful compiler must automatically recognize and fuse an entire
repeated observation tile, while preserving only the few live shared frontiers
between tiles.

## 1. Original objective

The target model contained a large repeated prediction plate:

```text
features
   |
   v
SoftDot
   |
   v
ResidualSine
   |
   v
SoftDot
   |
   v
ManyPlus
   |
   v
out + shared_intercept
   |
   v
NormalMeanPrecision
   |
   v
predicted y
```

For 1,024 observations and eight neurons, the complete prepared graph
contained approximately:

| Component | Count |
|---|---:|
| Factors | 28,692 |
| Variables | 30,740 |
| `ResidualSine` factors | 8,192 |
| First `SoftDot` factors | 8,192 |
| Second `SoftDot` factors | 8,192 |
| `ManyPlus{8}` factors | 1,024 |
| Addition factors | 1,024 |
| `NormalMeanPrecision` factors | 1,024 |
| NGMP edge states | 16,384 |
| Expected NGMP firings over eight iterations | 131,072 |

The original frozen reactive baseline was:

| Metric | Baseline |
|---|---:|
| Runtime rematerialization | 1.299 s |
| Eight-iteration wave | 3.166 s |
| Time per iteration | 395.8 ms |
| Rematerialization allocations | 871.9 MB |
| Iteration-wave allocations | 1.119 GB |

The intended compiled targets were:

| Metric | Target |
|---|---:|
| Single-thread wave | <= 1.5 s |
| Six-thread wave | <= 0.75 s |
| Rematerialization | <= 0.617 s |
| Wave allocations | <= 0.60 GB |
| Marginal error | finally relaxed to <= `1e-10` |

## 2. How the correctness requirement evolved

Initially, the work pursued bitwise identity with reactive execution:

- Same messages.
- Same product grouping.
- Same update order.
- Same deferred-message force points.
- Same marginal history.
- Same callback order.
- Same free energy.
- Same NGMP state and firing counts.

That was too restrictive for the actual user requirement.

The requirement was subsequently relaxed:

1. Internal reactive messages did not need to be reproduced.
2. Only externally requested marginals mattered.
3. The tolerance was first relaxed to approximately `1e-12`.
4. It was finally relaxed to `1e-10`.

This relaxation allowed:

- Replacing hundreds of message deliveries with one fused tile call.
- Avoiding exact reconstruction of irrelevant messages.
- Replacing the reactive equality-chain implementation with an equivalent
  O(N) frontier calculation.
- Executing independent tile work in parallel.

However, considerable time had already been spent preserving internal reactive
behavior that was not externally required.

## 3. Establishing a reliable benchmark

A dedicated target benchmark was created in:

- `benchmarks/rxinfer_hotpaths.jl`
- `benchmarks/rxinfer_hotpaths_report.md`

The benchmark:

- Prepared one reusable RxInfer model.
- Separated rematerialization from iteration execution.
- Warmed Julia compilation before collecting samples.
- Forced BLAS to one thread.
- Checked posterior means and variances.
- Checked NGMP state counts.
- Checked every NGMP state fired exactly once per iteration.
- Recorded time, allocations, GC time, and callback spans.

A generic backend differential harness was also created:

- `benchmarks/RxInferBackendHarness.jl`
- `benchmarks/RXINFER_BACKEND_HARNESS.md`

It covers:

- Beta-Bernoulli.
- A Gaussian chain.
- A high-degree Gaussian node.
- Posterior histories.
- Free-energy histories.
- Callback consistency.
- Preparation and inference measurements.

This infrastructure was valuable because several candidates appeared
dramatically faster only because they silently stopped performing required
inference work.

## 4. Generic optimizations that succeeded

### 4.1 Rocket readiness counters

Rocket combinators repeatedly checked status arrays with operations equivalent
to:

```julia
all(vstatus)
all(cstatus)
```

For high-degree combinators, this rescanned a bit array on every incoming
notification.

The implementation added maintained counters:

```julia
nvalid
ncompleted
```

Each status transition updates the relevant counter, making readiness checks
O(1):

```julia
all_vstatus(wrapper) =
    wrapper.nvalid == length(wrapper.vstatus)
```

This was implemented in:

- `GenericUpdatesStatus`
- `collectLatest`
- Associated completion and reset paths

Measured combined impact:

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Eight-iteration wave | 3.166 s | 2.942 s | 7.1% faster |
| Time per iteration | 395.8 ms | 367.8 ms | 28.0 ms saved |
| Wave allocations | 1.119 GB | 1.099 GB | 1.8% lower |
| Allocation reduction | - | - | 19.9 MB |

The complete Rocket suite passed 21,936 assertions.

This is the broadest low-level performance improvement retained from the
work.

#### Limitation

The counter changes improve reactive scheduling, but they do not change the
fundamental cost model:

- There are still many subjects.
- There are still many subscriptions.
- There are still per-edge offers.
- There are still deferred messages.
- There is still one scheduling action for nearly every graph operation.

Therefore, this work could provide a useful 7% improvement but could not
produce the requested 2-4x end-to-end breakthrough by itself.

### 4.2 ReactiveMP callback-interest filtering

ReactiveMP previously constructed callback event objects and UUID span
identifiers even when the installed callback handler did not observe those
event types.

A new predicate was added:

```julia
is_callback_interested(callbacks, Val(:event_name))
```

Specialized behavior was added for:

- `nothing`
- Named tuples
- Merged callbacks
- RxInfer benchmark callback handlers

Message-rule calls, products, form constraints, and marginal computations now
construct events only when necessary.

Measured callback-enabled target result:

| Metric | Before | After |
|---|---:|---:|
| Wave time | 8.681 s | 2.755 s |
| Allocations | 1.475 GB | 1.226 GB |

This was a real generic improvement, but its largest benefit applies only when
callbacks are enabled. With `callbacks = nothing`, the baseline was already
much closer to the fast path.

### 4.3 RxInfer prepared-constraint metadata caching

Prepared RxInfer models reused GraphPPL topology but still rebuilt unchanged
variational-constraint metadata during rematerialization.

The implementation now tracks whether the prepared constraint metadata remains
valid:

```julia
prepared_constraints_materialized::Bool
```

Behavior:

- Repeated `infer!` calls using the prepared constraint specification reuse
  its materialized metadata.
- Explicit constraint overrides invalidate the cache.
- Returning to the prepared constraints rematerializes them once, then reuses
  them again.
- Initialization and ReactiveMP runtime objects are still rebuilt fresh.

Measured result:

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Rematerialization time | 1.299 s | approximately 0.590 s | 54.5% faster |
| Rematerialization allocations | 871.9 MB | approximately 459-466 MB | approximately 47% lower |

This was one of the strongest generic improvements, although it affects
preparation/rematerialization rather than the inference iteration wave.

### 4.4 Free-energy callback reuse

`StopEarly` previously opened a second free-energy score subscription after
every iteration, even when inference was already collecting free energy.

`AfterIterationEvent` was extended to carry the current collected free energy.
`StopEarly` now reuses it when available.

Measured focused result:

| Metric | Before | After |
|---|---:|---:|
| Runtime | 12.689 ms | 7.682 ms |
| Allocations | 12.31 MB | 4.94 MB |

This is a valid generic improvement, but it is relevant only to
free-energy-enabled inference with compatible callbacks.

## 5. Generic optimization attempts that failed

### 5.1 Eliminating callback branches and annotations more aggressively

An experiment introduced separate no-hook message product implementations
intended to skip callback and annotation machinery completely.

Expected result:

- Less branching.
- Fewer annotation operations.
- Fewer callback checks.

Actual target result:

```text
2.755 s -> 3.495 s
```

The supposedly simpler implementation was approximately 27% slower.

Likely causes:

- Additional dispatch paths.
- Worse Julia specialization.
- More code and type instability.
- Loss of favorable inlining from the existing implementation.
- Increased allocations in representative paths.

#### Lesson

A logically simpler Julia path is not automatically a faster path. Changes to
specialization and inference can dominate the cost of a few branches.

The experiment was reverted.

### 5.2 Fixed-marginal product elision

The target model uses many `FixedMarginalFormConstraint` values. The idea was
to identify products whose result would be replaced by a fixed marginal and
skip their mathematical work.

The candidate bypassed approximately 278,528 products per wave.

However:

| Path | Time |
|---|---:|
| Control | 2.678 s |
| Generic elision | 3.045 s |
| Specialized elision | 2.913 s |

Allocations decreased by about 5.7%, but runtime increased.

The problem was that scanning abstract message collections, checking flags,
determining eligibility, and handling generic cases cost more than multiplying
the cheap Gaussian messages.

#### Lesson

Do not optimize a cheap mathematical kernel by adding an expensive generic
eligibility test to every invocation. Such optimization belongs in a
compile-time liveness pass, not a runtime branch.

### 5.3 Fixed-marginal one-shot dependency streams

A more aggressive experiment treated fixed marginals as completed, one-shot
dependency sources.

It appeared spectacular:

| Metric | Control | Candidate |
|---|---:|---:|
| Wave time | 2.801 s | 0.971 s |
| Allocations | 1.023 GB | 462 MB |

But the result was invalid:

| Correctness metric | Expected | Actual |
|---|---:|---:|
| NGMP firings | 131,072 | 65,536 |
| NGMP states firing zero times | 0 | 8,192 |

The posterior changed as well.

The conceptual mistake was treating a fixed marginal as equivalent to a static
outbound message.

A fixed marginal constraint fixes the variable's marginal form or value, but
its outbound cavities can still depend on incoming messages from other
factors. Staticness must be propagated through the full dependency graph, not
inferred from a local constraint annotation.

Changing the live `fixed_value` also updated the public marginal while leaving
the one-shot dependency stale.

#### Lesson

A fixed marginal is a potential graph separator for liveness analysis, but it
is not automatically a constant message source.

### 5.4 Raising the `LimitStackScheduler` threshold

The hypothesis was that Rocket's stack-depth limiter caused too many
task/condition transitions.

Changing the threshold from 100 to 500 produced:

```text
2.978 s -> 2.928 s
```

approximately 1.7%, with essentially unchanged allocations.

This was too small to retain as the breakthrough path and increased stack-risk
tradeoffs.

#### Lesson

Scheduler tuning can shave small percentages, but it cannot remove the
fundamental per-event execution cost.

### 5.5 Rocket transaction scheduler

A transaction-oriented scheduler was prototyped to batch notifications.

Two modes were considered:

- Preserve all events.
- Keep only the latest event.

The semantics were incompatible:

- Preserving all events changed depth-first reactive delivery into FIFO
  behavior.
- Keeping only the latest event dropped intermediate notifications.

Both can alter inference behavior, particularly with:

- Reentrancy.
- Deferred messages.
- Structured marginals.
- NGMP state updates.
- Shared equality chains.

#### Lesson

A scheduler cannot reorder or coalesce events generically unless the compiler
has proven that the affected region is independent and confluent.

The prototype was removed.

### 5.6 Typed Rocket subject delivery cache

A typed listener cache improved isolated subject delivery:

| Listeners | Isolated improvement |
|---:|---:|
| 1 | approximately 1.3x |
| 2 | approximately 1.5x |
| 8 | approximately 2.9x |

But end-to-end:

| Metric | Before | After |
|---|---:|
| Wave time | 2.678 s | 3.094 s |
| Allocations | 1.023 GB | 1.416 GB |

Constructing and maintaining typed listener tuples throughout a large dynamic
graph cost more than the faster delivery saved.

#### Lesson

Optimizing individual subject delivery is insufficient when graph
construction, subscription management, and tuple construction dominate.

The candidate was reverted.

## 6. First general compiled-backend attempts

### 6.1 Partial static compiled backend

An early backend replaced selected reactive operations with compiled
equivalents.

Results on generic small cases varied:

| Case | Change |
|---|---:|
| Beta-Bernoulli | approximately 3.2% slower |
| Gaussian chain | approximately 5.1% faster |
| High-degree Gaussian | approximately 19.4% faster |

The scalar stream microbenchmark was approximately 3.4% faster but allocated
approximately 5.3% more.

More importantly:

- The structured target produced incorrect behavior.
- Fallback was per specification instead of per connected component.
- Reactive and compiled pieces disagreed about activation ownership.
- Some custom activation paths broke.
- Compatibility with ordinary RxInfer behavior was not maintained.

#### Root problem

A factor graph cannot safely be split at arbitrary individual operations. A
compiled region must own a semantically complete component or region,
including:

- Messages.
- Marginals.
- Cavities.
- Deferred forces.
- State updates.
- Public boundary publication.

The partial backend was removed.

### 6.2 Typed CSR scalar executor

A generic executor was built with:

- Typed integer ports.
- CSR fanout.
- Reusable continuation frames.
- Iterative rather than recursive propagation.
- Zero warmed allocations.

It handled deep chains safely, including a 20,000-gate chain.

Isolated results:

| Case | Speedup |
|---|---:|
| Fanout 1 | 1.61x |
| Fanout 2 | 2.06x |
| Fanout 8 | 4.35x |
| Fanout 32 | 6.31x |
| Chain depth 128 | 3.68x |
| Abstract payload fanout 2 | 1.71x |
| Repeated arity-2 wave | 1.04x |
| Repeated arity-9 wave | 1.04x |

Geometric mean: approximately 2.24x.

It failed the chosen 3x primitive gate, and the repeated-wave cases, the ones
most representative of inference, were almost tied with Rocket.

#### Root problem

It replaced Rocket scheduling one scalar operation at a time. It did not reduce
the number of logical scheduling units.

#### Lesson

The required unit of compilation is a whole repeated tile, not one factor or
one edge.

The scalar executor was removed.

## 7. Homogeneous bucket fusion

The scalar executor was replaced with a homogeneous bucket-fusion primitive
in:

```text
ReactiveMP.jl/src/execution/bucket_fusion.jl
```

It combines:

- Typed payload planes.
- CSR notification runs.
- Homogeneous lane kernels.
- Stable lane order.
- Depth-first descendant draining.
- Reentrancy support.
- Error and reset handling.
- Per-lane mutable-state isolation.

Measured results over repeated 500-sample runs:

| Lanes | Shared-fanout speedup | Complete-wave speedup |
|---:|---:|---:|
| 4 | 2.46-2.50x | 1.96-2.02x |
| 16 | 3.09-3.12x | 2.14-2.17x |
| 64 | 3.89-3.95x | 2.24-2.29x |

The shared-fanout geometric mean was approximately 3.11x, with zero warmed
allocations.

This primitive passed 214 focused assertions covering:

- Stable Rocket order.
- Depth-first descendants.
- Reentrancy.
- Error propagation.
- Reset.
- Mutable state alias rejection.
- Reference payloads.

### Why this still did not solve the original goal

It is a manually constructed execution primitive. There is no compiler that:

- Finds eligible graph regions.
- Converts RxInfer factor nodes into bucket kernels.
- Allocates slots.
- Determines live boundaries.
- Preserves deferred-force semantics.
- Connects outputs back to reactive boundaries.

It is useful infrastructure, but not a backend.

## 8. Demand-aware component compiler

The most complete general compiler attempt lowered connected components and
supported:

- Aliases.
- Data ingress.
- Factor messages.
- Structured messages.
- Variable marginals.
- Directional cavities.
- Deferred messages.
- Lazy demand.
- Predictions.
- Depth-first reentrancy.
- Stack-safe delivery.

Focused runtime tests passed 178 assertions.

Correctness on the real target was excellent:

- Count 2: byte-identical.
- Count 64: byte-identical.
- Count 1,024: byte-identical.
- Correct NGMP states.
- Correct 131,072 firings.

Performance was not:

| Count | Reactive wave | Compiled wave | Compiled rematerialization | Compiled bytes |
|---:|---:|---:|---:|---:|
| 2 | 21.23 ms | 7.20 ms | 12.95 ms | 2.03 MB |
| 64 | 150.19 ms | 185.27 ms | 276.19 ms | 52.33 MB |
| 1,024 | 2.678 s | 3.023 s | 19.941 s | 1.159 GB |

It missed every important large-scale gate:

- Slower than reactive at count 1,024.
- Rematerialization was more than 30x above target.
- Allocations were almost twice the target.
- No reason remained to test six-thread execution.

### Why it failed

The compiler was still too close to the reactive graph's granularity.

It lowered operations and edges but retained too much machinery:

- Too many individual operation descriptors.
- Too many per-edge relationships.
- Too much topology reconstruction.
- Too much generic dispatch.
- Too much work representing reactive demand exactly.
- Insufficient fusion across the entire repeated observation tile.

It successfully replaced the implementation of scheduling without eliminating
enough scheduling units.

### Key lesson

Correctly compiling 200-400 scalar graph events per observation into 200-400
static instructions does not produce the required breakthrough.

The compiler must turn the entire observation region into approximately one
region call.

The implementation was removed according to the hard kill gate.

## 9. The specialized one-model executor

After the general compiler failed, a model-specific executor was built to prove
that region-level fusion could actually reach the target.

The current implementation is in:

```text
src/static_prediction.jl
```

It performs one reactive bootstrap iteration, extracts current state, and runs
the remaining iterations without Rocket.

### 9.1 What it fuses

For each observation lane, it directly calls existing ReactiveMP and Surrogate
mathematical kernels for:

- Natural-parameter message products.
- `ResidualSine` forward NGMP message.
- `SoftDot` output message.
- `ManyPlus` sum.
- Addition with intercept.
- `NormalMeanPrecision` output message.
- Posterior `y` product.
- `ManyPlus` backward cavities.
- `SoftDot` backward message.
- `ResidualSine` backward NGMP message.

It does not duplicate the inference formulas for the nonlinear rules. It
reuses `@call_rule`.

### 9.2 Initial incorrect boundary approximation

The first implementation treated the shared intercept as if every lane could
independently use the fixed prior.

That worked at small counts and produced very small errors, but count 1,024
exceeded tolerance.

An attempted approximation multiplied repeated boundary messages and
calibrated offsets. This was the wrong abstraction.

The trace later showed why.

### 9.3 Critical shared-intercept discovery

`FixedMarginalFormConstraint` fixed the intercept's marginal but did not make
each outbound cavity equal to the prior.

For lane `k`, the addition factor receives an intercept cavity containing:

- The intercept prior.
- Messages from the other observation lanes.
- Excluding the lane's own addition-to-intercept message.

Thus all supposedly independent observation tiles meet at one shared frontier
per iteration.

Each lane produces:

```text
addition -> intercept
```

Then the intercept equality chain produces a leave-one-out cavity:

```text
intercept -> addition[k]
```

The correct architecture became:

```text
parallel local forward tiles
          |
          v
serial O(N) intercept frontier
          |
          v
parallel local backward tiles
```

### 9.4 Synchronous versus reactive update order

A fully synchronous prefix/suffix computation made all current iteration
messages visible before computing every lane cavity.

This produced a maximum count-1,024 mean error of approximately:

```text
1.032e-10
```

just outside the requested tolerance.

Field-by-field diagnostics showed:

- Local forward messages were exact.
- Variances were exact.
- The first mismatch appeared in the intercept cavity mean.
- The mismatch propagated unchanged through addition and the requested `y`
  posterior.

ReactiveMP was effectively exposing a Gauss-Seidel frontier for most lanes:

- Current-iteration messages from earlier lanes.
- Previous-iteration messages from later lanes.

The specialized executor was changed to snapshot the old right suffix, compute
the current left prefix, and combine them per lane.

That reduced the verified maximum mean error to approximately:

```text
9.85e-11
```

with exact variances.

This is uncomfortably close to the `1e-10` threshold, but within it.

### 9.5 Threading

The local forward and backward phases use deterministic contiguous lane chunks:

```text
thread 1: lanes   1 ... a
thread 2: lanes a+1 ... b
...
```

Each thread writes only its own lane state.

The shared intercept frontier remains serial and deterministic.

Verified static serial/threaded difference:

```text
mean error     = 0
variance error = 0
```

### 9.6 Measured specialized performance

Verified one-sample results before the final unverified boundary
micro-adjustment:

| Path | Time | Allocations |
|---|---:|---:|
| Reactive | 2.579 s | 0.961 GB |
| Specialized, 1 thread | 0.630 s | 0.499 GB |
| Specialized, 6 threads | 0.613 s | 0.499 GB |

This is approximately:

- 4.09x faster single-threaded.
- 4.16x faster in the six-thread run relative to its reactive reference.
- 48% lower allocations.

The six-thread static path was only slightly faster than the single-thread path
because the total includes:

- One reactive bootstrap iteration.
- Extraction of runtime state.
- Shared serial frontier work.
- Result construction.

The local fused phases scale, but they were no longer the entire measured
runtime.

### 9.7 Important verification caveat

After the last verified count-1,024 result, one additional special-case change
was made for first-lane liveness. The required five-sample benchmark was
started but interrupted when the implementation was correctly identified as
model-specific rather than general.

Therefore:

- The 0.630 s and 0.613 s numbers belong to the last verified revision before
  that final micro-adjustment.
- The current working-tree version contains one later change that has not
  completed the full count-1,024 validation.
- No five-sample median exists for the final working tree.

## 10. Why the specialized path succeeded

The specialized executor obtained the breakthrough because it changed the unit
of execution.

Reactive execution approximately does this for each lane:

```text
hundreds of scheduler operations
hundreds of edge deliveries
many deferred-message checks
many subject notifications
many readiness checks
many generic dispatches
```

The specialized executor does this:

```text
one typed forward lane call
one O(N) shared frontier phase
one typed backward lane call
```

The mathematical rule computations remain, but most orchestration disappears.

This validates the core performance thesis:

> Whole-region fusion can achieve the target. Scalar operation replacement
> cannot.

## 11. Why the specialized path is not a compiler

The executor directly assumes:

- Variables named `za`, `h`, `c`, `out`, `intercept`, `mean_output`, and `y`.
- Exactly the factor sequence used by this prediction model.
- Eight-neuron `ManyPlus`-style summation.
- Specific message directions.
- Specific NGMP state ordering.
- A shared scalar intercept.
- Specific fixed priors.
- Specific initialization and factorization patterns.
- A specific reactive lane-order effect at the shared frontier.

There is no general representation of:

- Operations.
- Slots.
- Factors.
- Regions.
- Frontiers.
- Output publication.
- Guards.
- Topology signatures.

The type called `StaticPredictionProgram` is therefore not a general compiled
program. It is typed configuration for one handwritten executor.

## 12. Current useful code

### SurrogateModelling

- `src/static_prediction.jl`
- `benchmarks/static_prediction_tile_benchmark.jl`
- `benchmarks/static_boundary_trace.jl`
- `benchmarks/rxinfer_hotpaths.jl`
- `benchmarks/rxinfer_hotpaths_report.md`
- `benchmarks/RxInferBackendHarness.jl`

`marginal_tile_probe.jl` is stale because it refers to a moved benchmark-local
source file.

### ReactiveMP

- Generic callback-interest filtering.
- Reusable `compute_marginal_mapping`.
- `src/execution/bucket_fusion.jl`.
- Focused tests and benchmarks.

### Rocket

- O(1) readiness counters.
- O(1) `collectLatest` validity/completion checks.
- `cache_recent!` and `notify_recent!` split.
- Expanded combinator and subject tests.

### RxInfer

- Prepared-constraint metadata cache.
- Current free energy in `AfterIterationEvent`.
- Free-energy reuse in `StopEarly`.
- Callback-interest specialization for benchmark callbacks.
- Tests and documentation.

Everything remains uncommitted.

## 13. What I would do differently from scratch

### 13.1 Separate generic optimizations from compiler work

I would create four independent branches or commits:

1. Rocket readiness counters.
2. ReactiveMP callback filtering.
3. RxInfer prepared-constraint caching.
4. Static compiler.

This would prevent:

- Benchmark contamination.
- Confusing generic gains with compiler gains.
- Accumulating unrelated dirty changes.
- Uncertainty about which revision produced a result.

The first three changes already provide a cleaner reactive baseline and should
be evaluated independently.

### 13.2 Define the external correctness contract immediately

I would freeze this contract at the start.

Required:

- Requested posterior marginals within `1e-10`.
- Correct iteration history length.
- Correct prediction shapes.
- Correct NGMP state behavior in compiled regions.
- Deterministic serial/threaded results.
- Safe fallback before compiled execution when guards fail.

Initially excluded:

- Internal message identity.
- Bitwise equality.
- Unrequested marginals.
- Exact scheduler event counts.
- Arbitrary callbacks inside compiled regions.
- Free energy inside compiled regions.

Callbacks and free energy would be introduced later as explicit frontier
consumers.

This would avoid spending time reproducing irrelevant internal reactive events.

### 13.3 Build explicit compiler hooks during graph activation

I would not attempt to reverse-engineer a general compiler solely from callback
traces.

Callbacks reveal:

- That a rule executed.
- Its concrete mapping.
- Its concrete inputs and outputs.

But they do not naturally provide stable compile-time identities for:

- Input slots.
- Output slots.
- Stream ownership.
- Deferred-message dependencies.
- Variable cavity directions.
- Public publication boundaries.

Instead, ReactiveMP activation should optionally emit compiler descriptors
while it builds the reactive runtime.

For every factor interface it already knows:

- `MessageMapping`.
- Message dependencies.
- Marginal dependencies.
- Factor node.
- Variable.
- Interface direction.
- Constraint.
- Metadata.
- Output stream.

The compiler should receive that information directly.

A trace can still validate order, but it should not be the sole source of graph
structure.

### 13.4 Introduce a genuinely general IR

I would define something approximately like:

```julia
struct StaticProgram{Regions, Frontiers, Outputs, Guards}
    regions::Regions
    frontiers::Frontiers
    outputs::Outputs
    guards::Guards
    topology_hash::UInt64
end

struct StaticRegion{Signature, Kernel, Lanes, Inputs, Outputs}
    signature::Signature
    kernel::Kernel
    lanes::Lanes
    inputs::Inputs
    outputs::Outputs
end

struct StaticState{Planes, Epochs, NGMP, Scratch}
    planes::Planes
    epochs::Epochs
    ngmp::NGMP
    scratch::Scratch
end
```

Generic operation classes would include:

```text
IngressOp
AliasOp
MessageRuleOp{MessageMapping}
MarginalRuleOp{MarginalMapping}
MessageProductOp
VariableMarginalOp
DeferredForceOp
PublishOp
FrontierOp
```

The important point is that `MessageRuleOp` stores the existing reusable
`MessageMapping`; it does not duplicate the rule's mathematics.

Likewise, `MarginalRuleOp` calls the extracted
`compute_marginal_mapping`.

### 13.5 Give every value a stable typed slot

The earlier general compiler spent too much time on generic graph objects and
per-edge runtime machinery.

A new compiler should assign stable slot IDs during lowering:

```text
message slot
marginal slot
data slot
fixed-prior slot
NGMP-state slot
public-output slot
```

Slots should be grouped into typed planes:

```julia
struct NormalMeanVariancePlane
    values::Vector{NormalMeanVariance{Float64}}
end

struct NormalWeightedMeanPrecisionPlane
    values::Vector{NormalWeightedMeanPrecision{Float64}}
end
```

Avoid:

```julia
Vector{Any}
```

and avoid one heap object per slot.

A `StaticSlotRef` would contain:

```text
plane ID
index
epoch
```

The program is immutable. Each run receives fresh state planes.

### 13.6 Perform backward liveness before region construction

Start from externally live outputs:

- Requested posteriors.
- Requested predictions.
- Free energy, if enabled.
- Callback-observed values.
- NGMP state effects.
- Public reactive boundaries.

Walk backward through slot producers.

Dead operations are removed before execution planning.

Fixed marginal constraints can serve as separators only after proving which
outbound messages remain live.

This would have prevented the mistaken assumption that a fixed marginal makes
all neighboring messages constant.

### 13.7 Treat deferred messages explicitly

ReactiveMP's deferred messages are not merely lazy performance wrappers. Their
force points affect:

- Whether a rule executes.
- NGMP firing counts.
- State mutation.
- Callback visibility.
- Reentrant scheduling.

The IR should contain explicit `DeferredForceOp` instructions at traced or
statically derived force points.

A deferred rule should not execute merely because it exists in the graph.

However, once liveness proves that a whole repeated region always forces the
same sequence, that sequence can be fused.

### 13.8 Detect repeated regions automatically

This is the key missing breakthrough.

After lowering and liveness, construct an operation signature for each
candidate observation region:

```text
operation kind
mapping type
constraint type
metadata type
input relative-slot pattern
output relative-slot pattern
stateful/static classification
frontier dependencies
```

Canonicalize slot numbers relative to the region.

Two regions are isomorphic when their signatures match after relative
renumbering.

For the target model, the compiler should discover 1,024 copies of the same
observation tile.

Instead of storing 1,024 times hundreds of individual instructions, produce:

```julia
StaticRegion(
    kernel = CompiledObservationTile(...),
    lanes = 1:1024,
)
```

The kernel must still be constructed generically from operation mappings. It
should not mention `ResidualSine` or `SoftDot` in the compiler itself.

Possible implementation strategies:

1. Generate a Julia callable type containing a small tuple of concrete
   mappings.
2. Use a generated function to unroll the region's short operation sequence.
3. Store lane-varying state in structure-of-arrays form.
4. Call the same region kernel for every lane.

The specialized executor proves that this region granularity can achieve the
performance target.

### 13.9 Discover shared frontiers from graph cuts

After cutting repeated regions, examine edges that cross region boundaries.

Classify them as:

- Immutable ingress.
- Per-iteration shared reduction.
- Serial stateful frontier.
- Public reactive output.
- Unsupported boundary.

For the target, the compiler should discover:

```text
1,024 repeated observation regions
          |
          v
one shared intercept variable frontier
```

The equality-chain behavior should lower to a generic prefix/suffix frontier,
not a model-specific intercept formula.

For a product-capable distribution family:

```text
prefix[k] = product(message[k], prefix[k-1])
suffix[k] = product(message[k], suffix[k+1])
cavity[k] = product(prefix[k-1], suffix[k+1])
```

The exact reactive update schedule must be represented in the frontier plan:

- Jacobi/current-current.
- Gauss-Seidel/current-old.
- A traced mixed publication schedule.

This should be inferred from dependency order and demand, not manually coded
for lane 1.

### 13.10 Thread only proven-independent regions

Parallel execution should occur in phases:

```text
Phase A: parallel region-local forward work
Barrier
Phase B: deterministic serial/shared frontier
Barrier
Phase C: parallel region-local backward work
Barrier
Phase D: serial callbacks/reductions/publication
```

Each thread receives:

- A contiguous lane range.
- Disjoint state planes.
- Thread-local scratch.
- No access to another lane's mutable NGMP state.

Shared outputs should be buffered and replayed serially in the original global
order when order is externally visible.

This is safer than trying to make Rocket delivery itself concurrent.

### 13.11 Cache topology, not state

For prepared inference:

Cached:

- Static program.
- Region signatures.
- Slot layouts.
- Operation mappings whose types and topology are stable.
- Frontier plans.
- Guards.

Fresh per run:

- Data.
- Initialization messages.
- Fixed-prior values.
- NGMP state.
- Slot values.
- Epochs.
- Thread scratch.
- Public result buffers.

Invalidation guards should include:

- Graph topology hash.
- Mapping types.
- Constraint types.
- Factorization structure.
- Initialization shapes.
- Slot payload types.
- Region lane count.

Changing prior values should restage state without rebuilding topology.

### 13.12 Integrate only after the standalone executor wins

I would stage implementation as follows.

#### Stage A: generic serial IR replay

Goal:

- Correctness only.
- Arbitrary supported operations.
- No requirement to beat reactive execution.

Gates:

- Beta-Bernoulli.
- Gaussian chain.
- High-degree Gaussian.
- Count-2 target.
- Requested marginals within `1e-10`.

#### Stage B: repeated-region discovery

Goal:

- Automatically identify the observation plate.
- Produce one canonical region and N lane states.

Gate:

- Count 64 must beat reactive execution.
- No model-name or variable-name checks.

#### Stage C: shared frontier lowering

Goal:

- Automatically lower the intercept equality chain.
- Pass count 1,024 correctness.

Gate:

- Single-thread <= 1.5 s.
- Allocations <= 0.60 GB.

#### Stage D: threading

Goal:

- Parallel region-local phases.
- Deterministic frontier replay.

Gate:

- Six-thread <= 0.75 s.
- Serial/threaded marginal agreement within `1e-10`.

#### Stage E: RxInfer integration

Only after the previous gates:

- Add `backend=:compiled`.
- Add `backend=:auto`.
- Cache the program in prepared inference.
- Add guard-based fallback.
- Ensure unsupported regions remain reactive boundaries.

This order avoids spending time integrating a compiler that has not yet
demonstrated region-level performance.

## 14. What I would retain

If restarting, I would retain as separate, reviewed changes:

1. Rocket readiness counters.
2. ReactiveMP callback-interest filtering.
3. RxInfer prepared-constraint caching.
4. Free-energy callback reuse.
5. The backend differential harness.
6. The target hot-path benchmark.
7. The homogeneous bucket-fusion primitive, but clearly marked internal until
   integrated.
8. The specialized executor only as an oracle for what a correctly fused
   target region should compute.

The specialized executor is valuable as:

- A performance upper-bound reference.
- A correctness fixture.
- A guide for region phase boundaries.
- Evidence that whole-tile fusion can meet the target.

It should not be presented as a backend.

## 15. What I would remove or quarantine

I would remove from the production module until a real compiler exists:

- Exported `StaticPredictionProgram`.
- Exported `StaticPredictionExecutor`.
- Public-looking specialized compiler functions.

I would move them under something like:

```text
benchmarks/oracles/specialized_prediction_executor.jl
```

I would also:

- Remove or update the stale `marginal_tile_probe.jl`.
- Keep `cache_recent!` and `notify_recent!` only if the next compiler design
  actually requires them.
- Avoid shipping bucket fusion as user-visible functionality until an
  integrated consumer exists.
- Keep all compiler experiments on isolated branches rather than one large
  dirty worktree.

## 16. Final assessment

### What worked

- The benchmark and correctness infrastructure became substantially better.
- Several generic reactive-path optimizations produced real gains.
- Prepared inference rematerialization improved dramatically.
- A generic bucket-fusion primitive demonstrated that plate-level
  amortization works.
- The specialized executor proved the target graph can run approximately 4x
  faster with about half the allocations.

### What failed

- The requested general compiler was not delivered.
- Early work optimized scalar event delivery instead of eliminating scalar
  events.
- Fixed-marginal semantics were misunderstood several times.
- Too much time was spent preserving internal messages and bitwise ordering.
- The complete general compiler preserved reactive granularity and therefore
  failed its performance gates.
- The final fast implementation encoded one model directly.

### Most important architectural conclusion

A viable implementation should not be "Rocket, but represented with arrays."

It should be:

> A compiler that lowers complete live repeated regions into typed region
> kernels, with explicit, minimal shared frontiers and reactive execution only
> at unsupported boundaries.

That is the path supported by the evidence from this run.
