# IVON posterior-gate ensemble benchmark

This isolated Julia environment retrains only the top gating head of the
PrecisionGatedExperts (PGE) ETTh1/ETTh2 mixture of experts with IVON. The five
neural experts, q10/q90 constant experts, train-only scaler, preprocessing, and
64-dimensional VAE are read from the published PGE checkpoints and never
updated. No Adam or AdamW baseline is rerun.

The primary estimator is an ensemble of 1,000 gate networks sampled from the
learned diagonal IVON posterior. The single gate at the posterior-mean
parameters is reported only in the appendix output.

## Reproducibility contract

- PGE is pinned to commit `8cfdef63384d48ba302b3dd25ffb0ff356fecaea`,
  which places ReLU after the hidden layer of the large gate.
- IVONRepro is pinned to commit
  `871dc5e0cc85a0ed00ea9fb4bad42fc1a52799ed`. That upstream tree declares an
  unused Git submodule which Julia's package materializer cannot check out, so
  its exact package source is stored under `vendor/IVONRepro`. `UPSTREAM.toml`
  records the commit, Git tree, and SHA-256 of every vendored package file; the
  CLI verifies those hashes before running.
- Protocol: `OT`, 96 input steps, horizons 96/192/336/720, chronological
  60/20/20 split, and scaler fitted only on the upstream training partition.
- Gates: MoE `Dense(65 => 7)` (462 parameters) and MoE Big
  `Dense(65 => 192, relu) -> Dense(192 => 7)` (14,023 parameters).
- Training retains PGE's softmax-weighted expert-SSE objective, 100-epoch
  budget, and `train_set=false` direction: update on validation observations,
  monitor the training observations with patience 1 and `min_delta=1e-3`.
- Base seed is 12345. Stable cell-specific seeds are derived from the complete
  effective configuration, so resuming cells cannot change other cells.

`frozen_hashes.toml` records SHA-256 digests for every CNN, NLinear (called
`MLP` in the upstream filename), LSTM, DLinear, NConv, and VAE checkpoint. The
q10 and q90 forecast vectors are hashed immediately before and after training.
All hashes are checked before a result is accepted.

## Set up

Run from this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The benchmark reads `../../data/ETTh1.csv` and `../../data/ETTh2.csv`. Large
published model files remain in the pinned PGE package; they are neither copied
nor modified.

## Commands

```bash
julia --project=. run.jl smoke
julia --project=. run.jl pilot
julia --project=. run.jl full
julia --project=. run.jl summarize
```

Add `--force` to `smoke`, `pilot`, or `full` to replace a matching benchmark
checkpoint. Without it, a completed checkpoint is resumed only after checking
its configuration fingerprint and prediction digest.

- `smoke` runs one epoch on four ETTh1/H96 observations for both heads and
  evaluates eight posterior samples on four observations. It exercises the real
  frozen PGE pipeline without claiming benchmark-quality numbers.
- `pilot` is restricted to ETTh1/H96. It truncates the raw time series at the
  final upstream validation target before constructing data, then splits that
  gate-training partition chronologically 80/20. It evaluates learning rates
  `{0.001, 0.01, 0.1}` and ESS multipliers `{1, 100}` for both heads. The shared
  configuration minimizes mean posterior-predictive NLL across the two heads,
  with mean MSE as tie-breaker. The test partition cannot be materialized by the
  pilot loader.
- `full` requires the pilot selection and performs the 16 final fits:
  2 datasets × 4 horizons × 2 gate architectures.
- `summarize` reads valid final checkpoints and rewrites result tables without
  training or prediction.

## Prediction semantics

For every posterior gate draw and time point, the seven logits are used in two
ways, matching upstream PGE:

1. softmax(logits) weights the seven frozen forecasts for the point forecast;
2. exp(logits) supplies expert precisions for the product-of-Gaussians PGE
   prediction. Its variance is `1 / sum(exp(logits))`.

The 1,000 Gaussian predictions form an equally weighted predictive mixture.
Mixture log density uses log-sum-exp. The reported point prediction is the mean
of component means. Epistemic variance is the variance of component means,
aleatoric variance is the mean component variance, and total variance is their
sum. CRPS and 95% intervals use one seeded observation draw per Gaussian
component. The checkpoint records that Monte Carlo convention.

Standardized metrics are NLL/log predictive density, MSE, RMSE, MAE, CRPS,
95% coverage, interval width, interval score, and epistemic/aleatoric/total
variance. Original `OT` metrics apply the training scaler; density values use
the exact Jacobian (`NLL_original = NLL_standardized + log(σ_OT)`). Gate
diagnostics include probability means and standard deviations, entropy,
top-expert shares, and sample/consensus switching rates.

## Outputs

Outputs live under `results/`:

- `checkpoints/*.jld2`: posterior mean parameters, Lux state, the complete IVON
  optimizer/Hessian tree at the selected epoch, seeds, dependency commits,
  scaler, split metadata, effective configuration, metrics, prediction traces,
  and replay digest;
- `runs.csv`: chronological run/resume ledger;
- `summary.csv` and `main_ensemble_table.{csv,md}`: primary posterior ensemble;
- `appendix_mean_head_table.{csv,md}`: posterior-mean head;
- `pilot/pilot_heads.csv`, `pilot_grid.csv`, and `selection.jld2`: selection
  audit trail.

Acceptance of the full benchmark requires 16 complete checkpoints with finite
metrics. Checkpoint loading recomputes a digest over posterior component means
and variances, which makes saved predictions replay-verifiable without rerunning
the expensive frozen expert pipeline.
