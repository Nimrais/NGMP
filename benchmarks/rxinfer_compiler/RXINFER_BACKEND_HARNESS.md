# RxInfer backend differential harness

`rxinfer_backend_differential.jl` is a correctness and steady-state
performance harness for prepared RxInfer models. It registers the normal
reactive backend as its reference and accepts additional adapters without
coupling the harness to a backend implementation.

Run the default suite:

```sh
julia --project=. benchmarks/rxinfer_compiler/rxinfer_backend_differential.jl
```

The suite covers Beta–Bernoulli, a small Gaussian chain, and a high-degree
Gaussian node. Each measured inference follows an untimed `infer!` call on the
same prepared model. This makes the inference measurement cover topology
reuse and fresh ReactiveMP runtime materialization. Preparation time and
allocations are reported separately.

By default output is written to
`joinpath(tempdir(), "rxinfer_backend_differential")`:

- `runs.csv`: preparation/inference time, allocations, GC/compile time,
  requested and executed iterations, and callback consistency.
- `trace.csv`: posterior means, posterior variances, and free energy for every
  retained iteration.
- `summary.csv`: correctness deltas and median performance relative to the
  reference backend.
- `metadata.csv`: versions and the run configuration.
- `hooks.csv`: availability and status of the existing surrogate hot-path
  benchmark.

Other backends can be plugged in by pointing `RXINFER_DIFF_ADAPTER` at a
Julia file:

```julia
using .RxInferBackendHarness
using MyBackendPackage

register_backend!(
    make_rxinfer_adapter(
        "my-backend";
        prepare_transform = MyBackendPackage.prepare_keywords,
    ),
)
```

The transform may instead target inference keywords with `run_transform`.
Both transforms can accept `keywords`, `(keywords, case)`, or
`(keywords, case, sample)`. For an API that is not keyword-compatible, create
`BackendAdapter("my-backend", execute)` directly; `execute` receives
`(case, sample, config)` and returns a `BackendRun`. `BackendRun` and
`IterationSnapshot` are exported so a fully custom adapter can return the
same normalized trace and measurement schema.

Useful environment variables:

- `RXINFER_DIFF_SAMPLES` (default `3`)
- `RXINFER_DIFF_WARMUP` (default `true`)
- `RXINFER_DIFF_CHAIN_STATES` (default `24`)
- `RXINFER_DIFF_HIGH_DEGREE` (default `128`)
- `RXINFER_DIFF_ATOL` / `RXINFER_DIFF_RTOL`
- `RXINFER_DIFF_OUTPUT_DIR`
- `RXINFER_DIFF_FAIL_ON_MISMATCH` (default `true`)
- `RXINFER_DIFF_RUN_HOTPATH` (default `false`)
- `RXINFER_DIFF_HOTPATH_SAMPLES` (default `1`)

Setting `RXINFER_DIFF_RUN_HOTPATH=true` runs
`rxinfer_hotpaths.jl` after the differential suite and records its result in
`hooks.csv`. A requested hook failure makes the harness fail. The hook is
intentionally opt-in because it is much larger than the generic smoke cases.
