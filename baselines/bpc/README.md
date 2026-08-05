# Bayesian Predictive Coding on UCI regression

This folder contains an independent Julia implementation of Bayesian
Predictive Coding (BPC) from:

> Alexander Tschantz, Magnus Koudahl, Hampus Linander, Lancelot Da Costa,
> Conor Heins, Jeff Beck, and Christopher Buckley. *Bayesian Predictive
> Coding*. arXiv:2503.24016, 2025.

The downloaded TeX bundle is preserved under [`paper/`](paper/), with the
method in [`paper/main.tex`](paper/main.tex) and derivations and experimental
settings in [`paper/appendix.tex`](paper/appendix.tex).

## Implemented method

The implementation follows equations (1)--(10), Algorithm 1, and Appendix
B/C/F of the source:

- a hierarchical Gaussian ReLU network with weights outside the nonlinearity;
- a full Matrix-Normal--Wishart posterior for every weight-and-bias matrix;
- MAP hidden-state inference using 10 batched Adam steps by default;
- closed-form local updates to the four posterior natural parameters;
- the paper's stochastic natural learning rate `kappa_t = t^-0.25`;
- deterministic posterior-mean forward passes and complete posterior Monte
  Carlo sampling, including Wishart precision and conditional matrix-normal
  weight draws;
- a learned homoscedastic output covariance from the final MNW layer.

Biases are handled by appending a row of ones to each layer input. The TeX
cross statistic is printed as `f(z_(l-1)) z_l'`, but its dimensions conflict
with the paper's own `M V^-1` natural parameter for weights shaped
`output × input`. The code uses the equivalent dimensionally consistent
orientation `z_l f(z_(l-1))'` (`Y X'`) and tests this formula directly.

## Comparison with DVI and BBB

The benchmark deliberately shares the repository's comparison contract:

- the same six UCI loaders;
- the versioned `repeated-holdout-v1` 90/10 outer splits and inner validation
  indices used by `baselines/dvi` and `baselines/bbb_uci`;
- train-only feature and target standardization;
- two hidden layers of 50 ReLU units, minibatches of 100, and 20 posterior
  draws, as specified for BPC's UCI experiment in Appendix F.3;
- validation epoch selection followed by reinitialization and refitting on
  the complete outer-training partition;
- finite-mixture `logmeanexp` LPD, the exact target-scale Jacobian, original
  unit RMSE, interval coverage, and epistemic/aleatoric variance summaries;
- incremental result rows, split manifests, and replayable checkpoints.

BPC's published conjugate output is homoscedastic. The local robust DVI and
default BBB runs use a heteroscedastic `(mean, log variance)` head. Those
models can be compared under identical rows and metrics, but they are not the
same likelihood family. For the closest BBB likelihood match, run BBB with
`BBB_LIKELIHOODS=homoscedastic`. This implementation does not invent a
heteroscedastic conjugate BPC update that is absent from the paper.

Result rows use `likelihood=homoscedastic`, matching BBB's label, and record
`posterior_family=matrix_normal_wishart` separately.

The paper leaves its exact UCI split, stopping rule, target standardization,
and minibatch sufficient-statistic scaling unspecified. The local choices
above are recorded in `config.toml`; this is a method implementation under a
shared protocol, not a claim of reproducing the paper's table exactly. The
default `BPC_MINIBATCH_STAT_SCALE=1` follows the printed minibatch update.

## CPU and CUDA execution

CPU is the portable reference:

```sh
julia --project=baselines/bpc baselines/bpc/test/runtests.jl
```

Run a short Yacht smoke benchmark:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BPC_DATASETS=yacht BPC_SPLITS=1 BPC_MAX_EPOCHS=5 \
BPC_MIN_EPOCHS=2 BPC_PATIENCE=2 \
julia --project=baselines/bpc baselines/bpc/run.jl
```

The checked two-epoch end-to-end driver also reloads and replays its saved
checkpoint:

```sh
julia --project=baselines/bpc baselines/bpc/test/smoke.jl /tmp/bpc-smoke
julia --project=baselines/bpc baselines/bpc/test/smoke.jl /tmp/bpc-smoke-gpu cuda
```

Select CUDA for training and latent inference:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BPC_BACKEND=cuda BPC_DATASETS=yacht BPC_SPLITS=1 \
julia --project=baselines/bpc baselines/bpc/run.jl
```

CUDA accelerates batched matrix operations, covariance recovery, hidden-state
inference, and sufficient-statistic updates. Evaluation materializes the
small posterior matrices on the host so seeded Wishart and matrix-normal
draws are reproducible across CPU and GPU runs. The original paper reports
CPU experiments only; CUDA is a local performance extension.

Important overrides include `BPC_DATASETS`, `BPC_SPLITS`,
`BPC_HIDDEN_UNITS`, `BPC_BATCH_SIZE`, `BPC_MAX_EPOCHS`, `BPC_LATENT_STEPS`,
`BPC_LATENT_LEARNING_RATE`, `BPC_NATURAL_LEARNING_EXPONENT`,
`BPC_MINIBATCH_STAT_SCALE`, `BPC_EVAL_SAMPLES`, `BPC_BACKEND`, and
`BPC_OUTPUT_DIR`.

To keep startup lean, `BPCUCI` directly includes the repository's canonical
`src/datasets/uci_regression.jl` and `src/datasets/uci_benchmark.jl` files
instead of loading the full top-level modelling package (and its unrelated
RxInfer, plotting, and Reactant stack). It therefore uses the same executable
loader and split definitions, not a forked reimplementation.

Outputs are written under `results/bpc/<timestamp>/`. Replay a saved holdout:

```julia
using BPCUCI

checkpoint = load_posterior_checkpoint(
    "results/bpc/<run>/checkpoints/yacht_split01_homoscedastic.jld2",
)
prediction = predict_holdout(checkpoint)
prediction.metrics
```
