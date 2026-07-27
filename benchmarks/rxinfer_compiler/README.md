# RxInfer compiler experiments archive

This directory contains the useful artifacts from the unsuccessful general
RxInfer compiler effort. Nothing here is loaded by `SurrogateModelling`, and
none of it implements `backend = :compiled`.

## Status

| Artifact | Status |
|---|---|
| Model-specific static prediction executor | Handwritten one-graph oracle; no accepted final performance gate |
| General backend differential harness | Useful and runnable |
| Target hot-path benchmark | Useful and runnable |
| Shared-intercept trace | Useful diagnostic |
| ReactiveMP bucket fusion | Passed focused tests/microbenchmarks; never integrated end to end |
| General RxInfer compiler | Removed after failing the count-1,024 performance gate |

The complete technical history and proposed clean redesign are in
[`POSTMORTEM.md`](POSTMORTEM.md).

## Directory contents

- `static_prediction.jl`: handwritten executor for the one supported
  `ResidualSine -> SoftDot -> ManyPlus -> intercept -> NormalMeanPrecision`
  prediction graph.
- `static_prediction_tile_benchmark.jl`: reactive-versus-static correctness and
  performance gate.
- `static_boundary_trace.jl`: trace that exposed the shared intercept cavity.
- `marginal_tile_probe.jl`: early small-count correctness probe.
- `rxinfer_hotpaths.jl`: reproducible 1,024-lane ReactiveMP/RxInfer workload.
- `rxinfer_iteration_profile.jl`: sampling/allocation profiler for that
  workload.
- `rxinfer_hotpaths_report.md`: optimization ledger with retained and rejected
  experiments.
- `RxInferBackendHarness.jl`, `rxinfer_backend_differential.jl`, and
  `RXINFER_BACKEND_HARNESS.md`: generic backend differential harness.
- `rxinfer_free_energy_callback.jl`: focused benchmark for the retained RxInfer
  free-energy callback reuse.
- `reactivemp/`: complete archived bucket-fusion implementation, tests, and
  standalone benchmark. The test file is the original `TestItemRunner`
  snapshot and can be included from a test environment that provides
  `@testitem`.

## Running the archived benchmarks

From the `surrogate-modelling` repository:

```sh
julia --project=. benchmarks/rxinfer_compiler/rxinfer_backend_differential.jl
```

Small static-oracle check:

```sh
PERF_BATCH_SIZE=2 PERF_SAMPLES=1 OPENBLAS_NUM_THREADS=1 \
    julia --threads=1 --project=. \
    benchmarks/rxinfer_compiler/static_prediction_tile_benchmark.jl
```

The archive defaults to the repository project two levels above its files.
To test local ReactiveMP/RxInfer checkouts, develop those checkouts into the
active environment or set the benchmark's project environment variables
explicitly.

## Retained production changes outside this archive

The following changes were not compiler code and had measured benefits, so
they remain in their owning repositories:

### ReactiveMP

- Callback-interest filtering in `src/callbacks.jl`.
- Conditional message/product/marginal callback construction in
  `src/message.jl` and `src/variables/random.jl`.
- Focused callback tests and microbenchmark.

Measured callback-enabled target result:

```text
8.681 s -> 2.755 s
1.475 GB -> 1.226 GB
```

### RxInfer

- Prepared-constraint metadata reuse in `src/inference/batch.jl`.
- Free-energy reuse by `AfterIterationEvent`/`StopEarly`.
- Benchmark-callback interest specialization.
- Focused tests and documentation.

Measured prepared rematerialization result:

```text
1.299 s -> approximately 0.590 s
871.9 MB -> approximately 459-466 MB
```

Focused free-energy callback result:

```text
12.689 ms -> 7.682 ms
12.31 MB -> 4.94 MB
```

## Archive smoke verification

The archived executor was rerun after relocation with two lanes, eight
iterations, one Julia thread, and `1e-10` marginal tolerance:

```text
maximum mean error     = 0
maximum variance error = 0
reactive               = 0.167 s
static                 = 0.196 s
```

This confirms that the archived code still executes and reproduces the small
reference case. It is not a performance success: that small run was about 17%
slower and allocated about 2% more. The historical 1,024-lane speedup in the
postmortem belongs to the last verified pre-adjustment revision; the current
archive has not passed the final large five-sample gate.

## Deliberately removed from production

- `SurrogateModelling` exports for the model-specific executor.
- ReactiveMP `BucketFusion` module inclusion.
- ReactiveMP bucket-fusion benchmark registration.
- The compiler-only `compute_marginal_mapping` extraction.
- Local path pins in `surrogate-modelling/Manifest.toml`.

Rocket's experimental tracked changes were reset separately after review.
