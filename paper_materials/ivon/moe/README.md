# IVON posterior-gate PGE results

This directory is the canonical result root for the IVON top-gate benchmark on
ETTh1 and ETTh2. The five neural experts, q10/q90 experts, preprocessing,
train-only scaler, and 64-dimensional VAE are frozen. Only the 65-input PGE gate
is retrained, using IVON.

The primary result uses 1,000 gate networks drawn from the learned diagonal
posterior. The posterior-sample sensitivity table evaluates deterministic,
nested K=10 and K=100 subsets from that same 1,000-component bank. IVON
selection uses only K=1,000 on a chronological 80/20 split of the ETTh1/H96
gate-training partition and never reads its test targets.

## Contents

- `config.toml` records the effective protocol, dependency commits, seeds, and
  selected learning-rate/ESS configuration.
- `frozen_hashes.toml` snapshots the accepted neural-expert and VAE hashes.
- `pilot/`, `smoke/`, and `checkpoints/` contain replayable JLD2 artifacts.
- `summary.csv` and `main_ensemble_table.{csv,md}` contain the primary K=1,000
  results.
- `posterior_sample_sensitivity_table.{csv,md}` contains K=10/100/1,000.
- `appendix_mean_head_table.{csv,md}` evaluates the posterior-mean gate.
- `runs.csv` is the chronological completion/resume ledger.

## Reproduction

From `benchmarks/ivon/` run:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. run.jl smoke
julia --project=. run.jl pilot
julia --project=. run.jl full
julia --project=. run.jl summarize
```

Completed checkpoints are resumed after their configuration, component digest,
and posterior-subset digest are validated. See `benchmarks/ivon/README.md` for
the complete data protocol and prediction semantics.
