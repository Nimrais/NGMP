# When does NGMP help?

This directory is the canonical, script-based reproduction entry point for the
three exhibits of the paper's comparison study ("Comparing VMP to NGMP: When
Does NGMP Matter?"), its convergence-diagnostics appendix, and the damping
diagnostic of the "Damping and Momentum" appendix. It generates exactly the
artifacts the paper uses — nothing else.

Every study is split into two stages:

- **compute** — runs the inference and persists everything downstream needs to
  `results/*.csv` (plus a `*_config.toml` snapshot). This is the expensive
  part (up to ~2 h for the full streaming study).
- **render** — reads *only* `results/` and redraws every figure, `.tex` table,
  and markdown summary in seconds. `run_all.jl` always renders through this
  path, so the CSV boundary is exercised on every run.

The result CSVs are committed via **git LFS**, so on a fresh clone

```sh
git lfs pull
julia --project=. when_ngmp_helps/render_all.jl
```

regenerates every paper figure and table without running any inference.

## The three studies

1. `normal_mean_precision.jl`: in an IID Normal model with jointly unknown mean
   and precision, 20 seeds draw different true means and precisions. Marginal
   `KL(p_exact || q_method)` is averaged across seeds as the sample size grows,
   producing separate state/mean and precision panels
   (`normal_kl_{state,precision}`).
2. `poisson_state_space.jl`: projected mean-field VMP makes the non-conjugate
   Poisson model tractable but discards temporal posterior dependence; native
   NGMP preserves the Gaussian chain after projecting each Poisson message. The
   protocol removes nested 5–50% subsets for 20 mask seeds and reports the
   metrics table (`poisson_state_space_metrics_table.tex`, including the
   budget-matched one-sweep VMP arm), the deepest-gap panels
   (`poisson_gap50_*`), the gap-depth profiles (`poisson_depth_*`), and the
   Bethe free-energy panels (`poisson_bethe_heldout_*`). Overridable:
   `WHEN_NGMP_POISSON_REPETITIONS`.
   - `poisson_damping.jl` is the companion damping diagnostic for the paper's
     "Damping and Momentum" appendix. It includes `poisson_state_space.jl` and
     compares three outer-loop settings of the same NGMP model: the undamped
     fixed-point map (α = 1), pure natural damping (α = 0.25), and the setting
     used for every NGMP sunspot result above (α = 0.5, β = 0.2), tracking the
     model's Bethe free energy (RxInfer `free_energy`) per variational iteration. On the sunspot masks all
     three converge (undamped fastest; supplementary panels
     `poisson_damping_heldout_{5,10,20,50}`, first 20 of 60 sweeps), so the
     paper panels use the synthetic random-walk series of the earlier synthetic
     damping figure (`Random.seed!(seed)`, z_k = z_{k-1} + N(0, 0.1), all counts
     observed) at chain lengths N ∈ {100, 250, 500, 1000}, 20 seeds, 200 sweeps
     (`poisson_damping_synthetic_n{100,250,500,1000}`; free energy above each
     seed's converged value, geometric mean and 95 % band over seeds, log axis): longer chains wander to extreme log-rates, where
     the undamped map diverges and the momentum setting can leave the natural
     domain (such fits are kept through `catch_exception = true` and counted
     as failed). `results/poisson_damping_summary.md` holds the settle sweeps
     and the per-chain-length convergence/divergence/failure counts quoted in
     the appendix. Overridable: `WHEN_NGMP_DAMPING_ITERATIONS`,
     `WHEN_NGMP_DAMPING_SYNTHETIC_ITERATIONS`, `WHEN_NGMP_POISSON_REPETITIONS`.
3. `streaming_hetero.jl`: the heteroscedastic hierarchy (model code in
   `hetero_model.jl`) with a Matérn-3/2 RFF-128 mean path (`signal_sd = 2`)
   and a separately frozen RBF-32 log-precision path (`level_sd = 1.6`), fitted
   once on all data (smoothing) versus on ten sequential batches with
   posterior-as-prior chaining (batching), 20 paired seeds, three arms
   (projected VMP, its one-sweep budget-matched ablation, and cavity/true
   NGMP). Produces four prediction panels
   (`streaming_{vmp,cavity}_{full,sequential}`), four posterior predictive
   variance panels with pointwise credible intervals, mean-weight uncertainty,
   and the true aleatoric variance
   (`streaming_{vmp,cavity}_variance_{full,sequential}`), each exported as a
   compact 480-by-320-point vector PDF (so axes remain readable
   when embedded at half-column width) and a 1200-by-800 PNG, in addition to
   the combined 2-by-2 inspection figure. Curve semantics are stated in the
   paper caption instead of repeated legends that obscure the panels,
   the `q(w)` collapse figure
   (`streaming_collapse`), the projected-VMP Bethe free-energy convergence
   panels (`streaming_bethe_{full,sequential}` — evidence the comparison
   probes fixed points, not truncation), and the summary table.

`hetero_model.jl` is a library (graph, `fit_arm` with the three inference
arms, priors, prediction, aleatoric benchmark data); it has no entry point.

The fitted variance-panel quantities are persisted in
`results/streaming_hetero_panels.csv`. Regenerate the four standalone variance
plots and the combined preview without rerunning inference using:

```sh
julia --project=. when_ngmp_helps/streaming_hetero.jl --render-only
```

### NGMP-only latent-mean capacity diagnostic

`ngmp_mean_capacity.jl` isolates the heteroscedastic model's mean pathway
without fitting either VMP arm. It tests a wider mean prior alone, 128 RBFs,
and 128 Matérn-3/2 RFFs against the historical 32-feature RBF mean map. In
every preset the exponentiated log-precision pathway remains fixed at its
original 32 RBF features and prior scale. It reports RMSE against the
benchmark's known latent mean in addition to noisy-observation RMSE/NLL:

```sh
julia --project=. when_ngmp_helps/ngmp_mean_capacity.jl
```

Use `--repetitions N` or `--iterations N` for a different diagnostic budget;
add `--vary-mean-feature-seed` to hold the data fixed and assess RFF-draw
robustness. This script writes only `ngmp_mean_capacity_*` artifacts and is
intentionally not part of `run_all.jl`.

## Method display names

Legends, table headers, and summaries never hardcode method names. The CSVs
store stable keys (`vmp`, `vmp1`, `ngmp`, `exact`); the display names live in
one place — `METHOD_LABELS` (and `METHOD_LABELS_SHORT` for tight table
headers) in `common.jl`. To relabel every artifact:

1. edit `METHOD_LABELS` in `common.jl`,
2. `julia --project=. when_ngmp_helps/render_all.jl`.

## Reproduce from scratch

Run from the repository root:

```sh
JULIA_NUM_THREADS=4 DATADEPS_ALWAYS_ACCEPT=true julia --project=. when_ngmp_helps/run_all.jl
```

The Sunspots dataset used by the Poisson study is obtained through DataDeps on
the first full run. Each study can also be run independently with the same
`julia --project=.` prefix; append `--render-only` (or set
`WHEN_NGMP_RENDER_ONLY=true`) to redraw its artifacts from the existing CSVs.

Generated PDFs (and PNG previews of the combined inspection figures) are
written to `figures/`, which stays untracked — figures are regenerable in
seconds from the tracked results. Scripts overwrite only their own named
artifacts.

## Smoke validation

Smoke mode reduces the simulation grids and uses a deterministic synthetic
Poisson series, so it does not need a dataset download:

```sh
WHEN_NGMP_SMOKE=true julia --project=. when_ngmp_helps/run_all.jl
```

To keep smoke artifacts outside the repository, set
`WHEN_NGMP_OUTPUT_DIR=/path/to/output`. The automated check in
`test/runtests.jl` uses a temporary directory this way, then deletes all
figures and re-renders them through `render_all.jl` to verify the
render-from-CSV path.

## Interpretation

Across the exhibits, NGMP uses the projected exact BP log-message while VMP
uses an expected log-factor. When neighboring cavity beliefs concentrate these
local objects agree. Bottlenecked edges (a state deep inside a held-out gap,
a per-observation noise level) keep cavity beliefs broad, which is the regime
where NGMP retains an advantage; the streaming study shows how the mean-field
arm's local error compounds through the hierarchy and through posterior reuse.
