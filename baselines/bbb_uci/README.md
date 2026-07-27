# Bayes by Backprop UCI baseline

This is an isolated, learned-feature Bayesian neural-network baseline for the
six UCI regression datasets already exposed by `SurrogateModelling`.

The default experiment is deliberately self-contained:

- two Bayesian hidden layers with 50 ReLU units each;
- factorized Gaussian weight and bias posteriors and a standard-normal prior;
- Adam at `1e-3`, batch size 100, one weight sample per training batch;
- validation early stopping followed by a fresh full-training-set refit;
- 20 posterior samples for predictive metrics;
- both a global (homoscedastic) and input-dependent (heteroscedastic) Gaussian
  observation scale;
- five independent 90/10 splits per dataset.

Run the mathematical and integration tests in the resolved environment:

```sh
julia --project=baselines/bbb_uci baselines/bbb_uci/test/runtests.jl
```

`Pkg.test()` is also supported, but Julia creates and precompiles a temporary
test environment for it, which is much slower with the repository-local
dataset dependency.

Run a short Yacht smoke test:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BBB_DATASETS=yacht BBB_SPLITS=1 BBB_LIKELIHOODS=heteroscedastic \
BBB_MAX_EPOCHS=10 BBB_MIN_EPOCHS=5 BBB_PATIENCE=2 \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

Run the complete 60-run table:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

Outputs are written incrementally under `results/bbb_uci/<timestamp>/`.
Set `BBB_OUTPUT_DIR` to reuse a directory; successful configurations in its
`runs.csv` are skipped. Failures are recorded and do not stop the remaining
runs.

Important environment overrides include `BBB_DATASETS`, `BBB_SPLITS`,
`BBB_LIKELIHOODS`, `BBB_MAX_EPOCHS`, `BBB_EVAL_SAMPLES`, `BBB_BATCH_SIZE`,
`BBB_HIDDEN_UNITS`, `BBB_OUTPUT_DIR`, and `BBB_RESUME`.

`lpd_standardized` is the predictive log density after target
standardization. `lpd_original` includes the required Jacobian correction,
`lpd_original = lpd_standardized - log(y_scale)`. RMSE and interval coverage
are always reported in original target units.
