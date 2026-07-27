# RxInfer hot-path optimization ledger

This archived ledger records every optimization evaluated against
`benchmarks/rxinfer_compiler/rxinfer_hotpaths.jl`. Rejected experiments are
reverted from their package repositories but remain listed here.

## Protocol

- Julia 1.11.9, one Julia thread, one BLAS thread.
- Eight warmed samples, with rematerialization outside the iteration-wave
  measurement and an explicit full GC before each stage.
- Target graph: 1,024 prediction points, eight neurons, eight iterations.
- Correctness gate: 16,384 NGMP states, exactly 131,072 firings, stable
  posterior means and variances within `atol = 1e-8`, `rtol = 1e-7`.
- Retention gate: at least 3% target wall-time reduction or 10% allocation
  reduction without a time regression, plus a general-library improvement and
  no representative regression above 2%.

## Frozen baseline

| Component | Revision |
|---|---|
| RxInfer | `1e0ecb79c8162a854b7fa71e5b87ab8d3b27610d` |
| ReactiveMP | `3b5481cd891638e243fb3fc5c3d5561f652b2171` |
| Rocket | `9ddfba930cd36b31b4c63adba646bf504dea666c` |

| Metric | Median |
|---|---:|
| Runtime rematerialization | 1.298981 s |
| Eight-iteration wave | 3.166057 s |
| Time per iteration | 0.395757 s |
| Iteration-wave allocations | 1,118,606,240 B |
| Allocations per iteration | 139,825,780 B |

These are the paired-clean medians; a single 66-second operating-system
outlier remains in the raw samples but does not affect the median. The frozen
raw samples and posterior moments are stored outside the repository under
`/tmp/rxinfer-perf-baseline.FLwpD1/`.

## Candidate ledger

| Candidate | Target result | General result | Correctness | Decision |
|---|---:|---:|---|---|
| ReactiveMP callback-interest filtering | Callback-enabled wave: 8.681 s → 2.755 s (-68.3%); 1.475 GB → 1.226 GB (-16.9%) | Irrelevant-callback `MessageMapping` microbenchmark ≈47× faster | Exact event order and paired span IDs | **Keep** |
| Rocket readiness counters | 3.166 s → 2.942 s (-7.1%); 1.119 GB → 1.099 GB (-1.8%) | Full Rocket suite: 21,936/21,936 | Byte-identical target posterior | **Keep** |
| ReactiveMP no-hook message products | 2.755 s → 3.495 s (+26.9%) when added to callback filtering | Higher allocations in representative path | Exact but slower | **Reverted** |
| ReactiveMP lazy no-hook annotations | Included in the rejected no-hook product candidate | No independent qualifying result | Exact but slower as bundled | **Reverted** |
| Fixed-marginal product elision | Generic: 2.678 s → 3.045 s (+13.7%); specialized: 2.913 s (+8.8%); bytes -5.7% | Bypassed 278,528 products/wave, but scanning abstract messages for flags cost more than the cheap Normal products | Byte-identical posterior; exact states/firings | **Rejected** |
| Higher `LimitStackScheduler` threshold | Paired 100 → 500: 2.978 s → 2.928 s (-1.7%); allocations essentially unchanged | Reduces Task/Condition stack splits but does not remove the dominant delivery work | Byte-identical posterior; exact states/firings | **Rejected** |
| Rocket transaction scheduler | No target improvement; not used by RxInfer | `PreserveAll` changes depth-first delivery to FIFO, while `KeepLatest` drops intermediate notifications | Focused 40/40 only; not semantically eligible for automatic use | **Remove/split** |
| Rocket typed `Subject` delivery cache | Direct-data version: 2.678 s → 3.094 s (+15.5%); 1.023 GB → 1.416 GB (+38.5%). The earlier boxed-notification version was worse at 4.418 s / 1.858 GB | Isolated 1/2/8-listener abstract-payload delivery was 1.3×/1.5×/2.9× faster, but constructing typed listener tuples across the full reactive graph outweighed delivery savings | Byte-identical posterior; exact states/firings; focused mutation/reentrancy/scheduler tests passed | **Reverted** |
| RxInfer prepared-constraint metadata cache | Rematerialization: 1.299 s → 0.591 s (-54.5%); 871.9 MB → 458.6 MB (-47.4%) | Reuses immutable prepared constraint topology | Exact override/default restoration smokes | **Keep** |
| RxInfer free-energy callback reuse | Focused FE workload: 12.689 ms → 7.682 ms (1.65×); 12.31 MB → 4.94 MB (-59.9%) | Avoids a duplicate score subscription for `StopEarly` | Identical FE/stopping trace | **Keep** |
| RxInfer benchmark-callback interest specialization | Part of the 68.3% callback-enabled target win | Avoids unrelated ReactiveMP events | Exact benchmark rows | **Keep** |
| Partial static compiled backend | Scalar stream micro: -3.4% but +5.3% bytes; one-sample RxInfer suite: Beta +3.2%, Gaussian chain -5.1%, high-degree Gaussian -19.4% | Small cases are exact, but structured target stalls because per-spec reactive/compiled fallback is not component-atomic | Target incorrect; legacy custom activation and RxInfer compat also broken | **Remove/quarantine** |
| Demand-aware component compiler with homogeneous buckets | Count 2: 21.23 ms → 7.20 ms; count 64: 150.19 ms → 185.27 ms best case; count 1,024: 2.678 s current reactive → 3.023 s compiled, 1.159 GB | Lowered aliases, ingress, factors, structured messages, marginals, cavities, deferred messages, and exact lazy demand; fused homogeneous factor/random lanes | Byte-identical count-2/count-64 posterior CSVs; count-1,024 SHA `eb8978395c43...`; exact 16,384 states and 131,072 firings | **Rejected and removed at the single-thread gate** |
| ReactiveMP typed-CSR scalar executor, phase 0 | Not integrated into RxInfer because the isolated general gate failed first | Equal-weight geometric mean 2.24×; range 1.04×–6.31×; 0 B versus 1.60 MB across the gate | 1,127/1,127 focused assertions, including randomized Rocket differential traces and a 20,000-gate chain | **Rejected; pivot to bucket/plate fusion** |
| ReactiveMP homogeneous bucket-fusion kernel | Not integrated into RxInfer yet; no end-to-end claim | Warm shared-fanout geometric mean 3.11× across three 500-sample repetitions; complete private-plus-shared waves 1.96×–2.29×; zero warm allocations | 214/214 focused assertions covering stable Rocket order, depth-first descendants, reentrancy, errors, reset, and reference payloads | **Keep as backend primitive; require target gate after integration** |
| Fixed-marginal one-shot dependency stream | Apparent 2.801 s → 0.971 s and 1.023 GB → 462 MB, but only because half the required NGMP edges stopped updating | Completion/staticness was not propagated transitively through outbound messages; live `.fixed_value` mutation also left the internal dependency stale | Failed: 65,536/131,072 firings, 8,192 states never fired, posterior SHA changed to `630073...` | **Rejected and reverted** |

### Typed-CSR scalar executor rejection

The isolated executor used typed integer ports, stable-order CSR fanout, and
reusable iterative continuation frames. At 1,000 rounds and 20 samples it
measured:

| Case | Compiled | Rocket | Speedup |
|---|---:|---:|---:|
| Fanout 1 | 19.396 ns/unit | 31.230 ns/unit | 1.61× |
| Fanout 2 | 10.542 ns/unit | 21.729 ns/unit | 2.06× |
| Fanout 8 | 3.622 ns/unit | 15.755 ns/unit | 4.35× |
| Fanout 32 | 2.121 ns/unit | 13.392 ns/unit | 6.31× |
| Arity-2 repeated wave | 18.073 ns/unit | 18.823 ns/unit | 1.04× |
| Arity-9 repeated wave | 17.398 ns/unit | 18.123 ns/unit | 1.04× |
| Chain depth 128 | 9.036 ns/unit | 33.271 ns/unit | 3.68× |
| Abstract heap payload, fanout 2 | 13.792 ns/unit | 23.625 ns/unit | 1.71× |

The implementation removed warmed allocations and scaled iteratively to a
20,000-gate chain, but its 2.24× geometric mean missed the 3× backend gate.
Most importantly, scalar repeated waves remained effectively tied with Rocket.
The experiment was therefore removed from ReactiveMP rather than adding a
second execution path whose review and maintenance cost exceeded its general
benefit. The next backend design must fuse homogeneous gate buckets or plates,
amortizing scheduling and status work across many nodes.

### Homogeneous bucket-fusion gate

The replacement microkernel separates a typed value plane from stable CSR
notification runs and drains each lane's descendants before advancing to the
next lane. Three independent 500-sample repetitions measured:

| Lanes | Warm shared-fanout speedup | Complete-wave speedup |
|---:|---:|---:|
| 4 | 2.46×–2.50× | 1.96×–2.02× |
| 16 | 3.09×–3.12× | 2.14×–2.17× |
| 64 | 3.89×–3.95× | 2.24×–2.29× |

The warm shared-fanout geometric means were 3.103×, 3.117×, and 3.112×,
with zero allocations in every fused case. Construction was slower for the
four-lane microcase (625 ns versus 292 ns), so construction remains a prepared
plan cost rather than an iteration-path win. The kernel is retained only as a
backend building block: it has not yet earned an RxInfer end-to-end speedup
claim. Raw results are stored at
`/private/tmp/reactivemp_bucket_fusion_gate_20260725_results.txt`.

### Fixed-marginal one-shot dependency rejection

A separate internal dependency stream treated fixed marginals as completed
one-shot sources while leaving public marginal computation and subscriptions
live. Its apparent target improvement was large:

| Metric | Control | Candidate |
|---|---:|---:|
| Eight-iteration wave | 2.801428 s | 0.970826 s |
| Iteration allocations | 1,022,552,944 B | 462,462,832 B |
| NGMP firings | 131,072 | 65,536 |

The missing work made the timing invalid: 8,192 of 16,384 NGMP states remained
at zero firings, and the posterior SHA changed from the frozen
`eb8978395c43...` to `6300735381fc...`. A live mutation probe also showed that
changing the documented mutable `FixedMarginalFormConstraint.fixed_value`
updated the public marginal but left the one-shot internal dependency stale.
The candidate was fully reverted. Any future compiled constant propagation
must track transitive liveness and treat mutable constraints as explicit
invalidations.

### Demand-aware component compiler rejection

A second compiled candidate lowered complete connected components before graph
mutation and activated them through exact lazy demand. Its focused runtime
suite passed 178/178 assertions covering deferred messages, structured factor
marginals, directional cavities, PushNew coalescing, predictions, late replay,
Gaussian inference, depth-first reentrancy, and stack-safe delivery. The real
target also remained exact:

| Gate | Reactive wave | Best compiled wave | Compiled rematerialization | Compiled wave bytes | Correctness |
|---|---:|---:|---:|---:|---|
| Count 2, eight iterations | 21.231 ms | 7.199 ms | 12.952 ms | 2.03 MB | Byte-identical CSV; 32 states, 256 firings |
| Count 64, eight iterations | 150.187 ms | 185.274 ms | 276.186 ms | 52.33 MB | Byte-identical CSV; 1,024 states, 8,192 firings |
| Count 1,024, eight iterations | 2.678 s retained reference | 3.023 s | 19.941 s | 1.159 GB | SHA `eb8978395c43...`; 16,384 states, 131,072 firings |

The count-1,024 run missed the 1.5 s single-thread, 0.617 s
rematerialization, and 0.60 GB allocation gates. It therefore did not proceed
to a six-thread claim. Following the hard acceptance rule, the component
compiler, RxInfer backend option, model-specific recipe adapters, and their
tests were removed. The separately qualified generic bucket-fusion primitive
remains available for a future region compiler that can amortize whole
observation tiles rather than individual operation deliveries.

## Target graph census for compiled lowering

The prepared target contains 28,692 factors and 30,740 variables. Its repeated
factor buckets are:

| Factor/factorization | Count |
|---|---:|
| `ResidualSine`, one structured `(out, in)` cluster | 8,192 |
| first `SoftDot`, clusters `(2, 1, 1)` | 8,192 |
| second `SoftDot`, mean-field `(1, 1, 1, 1)` | 8,192 |
| `ManyPlus{8}` | 1,024 |
| `NormalMeanVariance`, mean-field `(1, 1, 1)` | 1,024 |
| addition, one structured three-edge cluster | 1,024 |
| `NormalMeanPrecision`, clusters `(2, 1)` | 1,024 |
| standalone prior-distribution factors | 20 |

The variable side contains 27,648 degree-two random variables and 1,024
degree-eight data variables, plus 18 shared random variables of degree 1,025
and two of degree 8,193. Consequently, a useful automatic backend must support
structured local marginals, NGMP, `ManyPlus`, and a scalable equality-chain
kernel; optimizing only small mean-field nodes cannot reach the target.

## Current retained stack

A clean four-sample validation after retaining the accepted optimizations and
reverting the rejected no-hook message-product experiment produced:

| Metric | Frozen baseline | Current | Change |
|---|---:|---:|---:|
| Runtime rematerialization | 1.298981 s | 0.589761 s | -54.6% |
| Rematerialization allocations | 871,909,856 B | 466,209,632 B | -46.5% |
| Eight-iteration wave | 3.166057 s | 2.677582 s | -15.4% |
| Time per iteration | 0.395757 s | 0.334698 s | -15.4% |
| Iteration-wave allocations | 1,118,606,240 B | 1,022,552,944 B | -8.6% |

The current run retained exactly 16,384 NGMP states and 131,072 firings. Its
posterior CSV is byte-for-byte identical to the frozen baseline (SHA-256
`eb8978395c4399c784cc32002deab6100848e7de61d1dbe743f4740a9da846a2`).
Raw current samples are stored at
`/tmp/rxinfer-reactive-filter-only.csv`.

## Iteration versus run setup

The no-callback reference intentionally omits lifecycle instrumentation.
A separate four-sample run with the lifecycle-only
`RxInferBenchmarkCallbacks` measured the split without changing the posterior:

| Instrumented metric | Median |
|---|---:|
| Complete inference call | 3.174754 s |
| Sum of eight iteration callback spans | 2.822937 s |
| Everything outside the iteration spans | 0.325076 s |
| Propagation time per iteration | 0.352867 s |

Thus about 89% of the instrumented call is iteration propagation and about
10% is per-run subscription, actor, teardown, and result setup. The callback
instrumentation itself is visible—the no-callback retained reference is
2.677582 s—so the two totals must not be compared as if they used the same
configuration. Both configurations produced the same byte-identical posterior.

The corresponding sampling profile identifies generic execution overhead in
Rocket `Subject` delivery, `LimitStackScheduler`, lazy subscription wiring,
`DeferredMessage` materialization, and message-product folds. These are the
inputs to the typed-Subject and compiled-plan candidates; rule computations
remain part of the irreducible workload unless homogeneous rule instances are
batched into typed kernels.
