# Deterministic Variational Inference for UCI regression

This folder implements the full-covariance **Deterministic Variational
Inference (DVI)** method from:

> Anqi Wu, Sebastian Nowozin, Edward Meeds, Richard E. Turner,
> José Miguel Hernández-Lobato, and Alexander L. Gaunt.
> *Deterministic Variational Inference for Robust Bayesian Neural Networks*.
> ICLR 2019. [arXiv:1810.03958](https://arxiv.org/abs/1810.03958)

The authors' reference TensorFlow implementation is available at
[microsoft/deterministic-variational-inference](https://github.com/microsoft/deterministic-variational-inference).

## What is implemented

The implementation is an independent Julia port of the following parts of the
paper and reference code:

- Equation (3): propagation of activation means and covariances through
  uncertain affine transformations.
- Equation (6) and Table 1: the full-covariance Gaussian approximation for
  moments of correlated ReLU activations. The functions `g`, `delta`,
  `softrelu`, and the covariance propagation in
  [`src/moments.jl`](src/moments.jl) are direct formula-level translations of
  the authors' `bayes_util.py` and `bayes_layers.py`.
- Equation (8): deterministic expected log-likelihood for heteroscedastic
  Gaussian regression, including the output covariance
  `Σ_mm`, `Σ_mℓ`, and `Σ_ℓℓ`.
- Equation (9): the Gaussian posterior-predictive approximation,
  with mean `E[m]` and variance
  `Σ_mm + exp(E[ℓ] + Σ_ℓℓ/2)`.
- Equations (10)–(13): diagonal Gaussian weight posteriors and the
  empirical-Bayes inverse-Gamma hierarchy. The default hyperprior is
  `α = 1`, `β = 10`.
- Full DVI, which propagates the complete covariance matrix between hidden
  activations, and optional diagonal-DVI (`dDVI`), selected with
  `DVI_PROPAGATION=diagonal`.

There is no weight sampling in the DVI training objective or prediction
path. Monte Carlo weight sampling exists only in the tests, where it checks
the propagated moments.

## UCI settings taken from the paper

The architectural and statistical defaults follow the settings stated for
Table 2:

- one hidden layer with 50 ReLU units;
- factorized Gaussian posterior over every weight and bias;
- a layer-partitioned empirical-Bayes prior. This implementation treats a
  layer's weight matrix and bias vector as one parameter set, following the
  paper's statement that sets typically coincide with layer partitions;
- heteroscedastic output `(m, ℓ)` with observation variance `exp(ℓ)`;
- `α = 1`, `β = 10`;
- random 90/10 train/test splits repeated 20 times.

The repository currently exposes six of the paper's nine datasets: Boston,
Concrete, Energy, Power, Wine, and Yacht. Kin8nm, Naval, and Protein are not
available through `src/datasets/uci_regression.jl`. The local Wine loader has
1,599 rows, whereas the paper reports 1,588, so its numerical result is not a
like-for-like reproduction.

## Choices not fixed by the paper

The paper and public repository do not include the authors' complete UCI
experiment driver. Consequently this is a reproduction of the published
method and stated protocol, not a claim of bit-for-bit reproduction of Table
2. The remaining defaults are recorded in every output `config.toml`:

- Adam with learning rate `0.0003`;
- minibatches of 100;
- posterior initialization based on the `wider_he` setting in the authors'
  public code: weight variance `5/fan_in`, zero bias means, and bias variance
  divided by 10. For the heteroscedastic log-variance output only, the robust
  protocol uses zero mean weights/bias and posterior standard deviation `0.05`;
- elementwise gradient clipping to `[-0.1, 0.1]`, matching the authors'
  public training utility;
- bounded exponent arguments in `[-20, 20]` for posterior variances,
  likelihood precision moments, and predictive variance moments. Likelihood,
  KL, and metric scalar arithmetic use `Float64`; model parameters remain
  `Float32`;
- 10,000 maximum epochs, with the KL off for 14,000 optimizer steps and
  annealed over 1,000 steps. This preserves the authors' released toy
  notebook's 14,000-update warmup without exposing larger datasets to many
  more unregularized Adam updates. It is a local robust UCI protocol, not a
  UCI setting stated in the paper;
- train-only feature and target standardization;
- inner validation for selecting the epoch, followed by a fresh refit on all
  outer-training observations; selection and early stopping begin only after
  any configured KL warmup and annealing are complete;
- validation every five epochs and patience of 500 checks. Full DVI also caps
  a no-improvement tail at 50,000 optimizer steps; diagonal-dDVI retains only
  the original check-based patience and is not affected by this cap;
- full DVI skips validation before the selectable KL phase. With Reactant it
  propagates validation moments on the selected device and transfers only the
  small output moments back for the unchanged host-side `Float64` metrics.

The exact repeated-holdout algorithm is shared with BBB through
`SurrogateModelling`'s versioned `repeated-holdout-v1` protocol. Split IDs
use sorted original-row indices, base seed `20260726`, and a fixed inner-seed
offset. Every run writes the resulting index sets to `split_manifest.jld2`
and to each versioned checkpoint, allowing independently launched baselines
to be compared directly.

The optimizer, minibatch size, stopping rule, standardization convention,
posterior initialization, and exact UCI split seeds are not jointly specified
by the paper. They must not be attributed to Wu et al.

## Metrics

LPD is evaluated directly from the deterministic predictive Gaussian in
Equation (9), not from sampled networks. Both scales are saved:

```text
lpd_original = lpd_standardized - log(y_scale)
nll = -lpd
```

RMSE and interval coverage are reported in original target units. Predictive
variance is decomposed into:

```text
epistemic = Σ_mm
aleatoric = exp(E[ℓ] + Σ_ℓℓ / 2)
total     = epistemic + aleatoric
```

## Run

Run the mathematical and synthetic integration tests:

```sh
julia --project=baselines/dvi baselines/dvi/test/runtests.jl
```

Run one full-covariance Yacht split:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
DVI_DATASETS=yacht DVI_SPLITS=1 DVI_PROPAGATION=full \
julia --project=baselines/dvi baselines/dvi/run.jl
```

Full DVI is substantially slower than dDVI because it propagates a hidden
covariance matrix. Run the paper-protocol six-dataset, 20-split driver with:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
julia --project=baselines/dvi baselines/dvi/run.jl
```

The default `zygote`/`cpu` execution backend is the portable numerical
reference. The optional Reactant backend uses Enzyme differentiation and keeps
the Float32 model, Adam state, and training data on the selected device. Its
XLA-compatible Gaussian CDF uses a high-accuracy elementary approximation;
cross-backend results are checked for numerical, rather than bitwise, parity:

```sh
DVI_BACKEND=reactant DVI_DEVICE=gpu \
DATADEPS_ALWAYS_ACCEPT=true DVI_DATASETS=yacht DVI_SPLITS=1 \
julia --project=baselines/dvi baselines/dvi/run.jl
```

Backend, device, and implementation version are saved as result-affecting
configuration fields. A result directory therefore cannot be resumed with a
different execution implementation.

`DVI_FULL_PATIENCE_STEPS` controls the full-DVI-only optimizer-step cap and
defaults to `50000`. It never changes diagonal-dDVI stopping behavior.

`DVI_SELECTION_MAX_STEPS` and `DVI_REFIT_MAX_STEPS` are optional hard
training-budget ceilings checked at epoch boundaries; zero (the default)
disables each ceiling. A selection ceiling forces one final validation when
reached. `DVI_TRAINING_BUDGET_PROTOCOL` records a human-readable identifier in
the configuration, run rows, and checkpoints. These settings are
result-affecting and therefore protected by resume validation.

Before launching the paper run, measure warmed gradient/Adam steps without
writing any result artifact:

```sh
DVI_PROFILE_BACKENDS=zygote \
julia --project=baselines/dvi baselines/dvi/profile_step.jl

DVI_PROFILE_BACKENDS=reactant DVI_DEVICE=gpu \
julia --project=baselines/dvi baselines/dvi/profile_step.jl
```

The profiler reports compilation time, steady-state time and allocations for
100-row batches with 6 and 13 inputs. Its conservative projection includes a
25% margin and must be at most three days before a 120-configuration run is
started.

Run diagonal-DVI:

```sh
DATADEPS_ALWAYS_ACCEPT=true DVI_PROPAGATION=diagonal \
julia --project=baselines/dvi baselines/dvi/run.jl
```

### Checked Yacht reproduction

The following legacy one-split command was run locally before the robust
numerical protocol was introduced:

```sh
DATADEPS_ALWAYS_ACCEPT=true OPENBLAS_NUM_THREADS=1 \
DVI_DATASETS=yacht DVI_SPLITS=1 DVI_PROPAGATION=diagonal \
DVI_MAX_EPOCHS=10000 DVI_KL_SCHEDULE_UNIT=epochs \
DVI_KL_WARMUP_EPOCHS=7000 \
DVI_KL_ANNEAL_EPOCHS=500 DVI_PATIENCE=500 \
DVI_OUTPUT_DIR=results/dvi/wu2019_yacht_ddvi_public_warmup_ratio_fixed_20260727 \
DVI_RESUME=false DVI_SHOW_PROGRESS=false \
julia --project=baselines/dvi baselines/dvi/run.jl
```

It selected epoch 7,925 after the KL was fully enabled and obtained:

| Dataset | Method | Splits | LPD | RMSE |
|---|---:|---:|---:|---:|
| Yacht | dDVI | 1 | -0.5284 | 0.7462 |
| Yacht, Wu et al. Table 2 | dDVI | 20 | -0.47 ± 0.03 | not reported |

This historical result is a close one-split check, not a result from the
current `bounded-exp-step-kl-v1` protocol and not a reproduction of the
paper's 20-split mean.

Outputs are written incrementally under `results/dvi/<timestamp>/`.
Successful configurations can be resumed by reusing `DVI_OUTPUT_DIR`; a
configuration is skipped only when its checkpoint is present and valid.
Long runs can be sharded safely with a comma-separated subset such as
`DVI_SPLIT_IDS=1,2,3,4,5`. `DVI_SPLITS=20` must remain unchanged so every
shard records the complete shared manifest; use a distinct output directory
per shard and combine the resulting run rows only after all shards finish.

For a paper comparison against BBB, run only the heteroscedastic likelihood
(the DVI regression model from Wu et al.) for both propagation variants.
Keep the shared split and evaluation settings at their defaults and use
separate output directories because `propagation` is a result-affecting
setting:

```sh
DATADEPS_ALWAYS_ACCEPT=true DVI_PROPAGATION=full \
DVI_LIKELIHOODS=heteroscedastic \
DVI_OUTPUT_DIR=paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits/dvi \
julia --project=baselines/dvi baselines/dvi/run.jl

DATADEPS_ALWAYS_ACCEPT=true DVI_PROPAGATION=diagonal \
DVI_LIKELIHOODS=heteroscedastic \
DVI_OUTPUT_DIR=paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits/ddvi \
julia --project=baselines/dvi baselines/dvi/run.jl
```

These runs use the same versioned outer and inner row indices, train-only
standardization, 20 repeated 90/10 holdouts, validation-based epoch selection,
full-training refit, target-scale Jacobian for LPD, and aggregate mean, sample
standard deviation, and standard error as BBB. DVI-specific architecture,
objective, robust initialization, empirical-Bayes prior, numerical bounds,
step-based KL schedule, stopping rule, and gradient clipping remain those
documented above and are saved in each `config.toml`. A 500-epoch equal-budget
pilot was rejected because both an
immediate KL and a proportionally shortened warmup failed to converge on the
checked Yacht split. Each directory contains the same paper artifacts as BBB:
`runs.csv`, `summary.csv`, `table.md`, `table.tex`, `split_manifest.jld2`,
histories, and versioned checkpoints.

Losses and gradients are checked before each optimizer update. Validation
histories are written incrementally, clamp counts/rates are included in run
rows, and failures have structured diagnostics under `failures/`. Resume uses
configuration-identity upserts, so a successful retry replaces its failed row.
Shard merging refuses incomplete, failed, duplicate, or selection-only runs.

Before a paper run, replay the 19 failures from the stopped run plus three
previously successful anchors without evaluating any outer-test targets:

```sh
baselines/dvi/run_ddvi_failure_replay.sh
```

After parity and the three-day projection gate pass,
`run_bbb_matched_paper.sh` runs full DVI in a fresh optimized output directory.
It defaults to four CPU shards. Select the validated Reactant GPU layout with,
for example:

```sh
DVI_PAPER_PROFILE_APPROVED=true DVI_PAPER_BACKEND=reactant \
DVI_PAPER_DEVICE=gpu DVI_PAPER_SHARDS=1 \
baselines/dvi/run_bbb_matched_paper.sh
```

`DVI_PAPER_SHARDS` accepts 1, 2, or 4. The launcher now refuses any
`DVI_PAPER_METHODS` value other than `full`, ensuring the completed dDVI result
cannot be rerun accidentally.

The separate homoscedastic campaign uses the explicitly reduced
`selection-refit-max-50000-v1` budget for both propagation methods. It keeps
the original 14,000-step KL warmup and 1,000-step anneal, but limits selection
and refit to 50,000 optimizer steps apiece:

```sh
DVI_HOMO_PROFILE_APPROVED=true \
baselines/dvi/run_homoscedastic_budgeted_paper.sh
```

This launcher hardcodes the homoscedastic likelihood, defaults to Reactant on
CUDA with four shards, and refuses to start until the Power split-1 quality
and timing pilots have approved the reduced protocol. Its results must be
reported as the budgeted homoscedastic protocol, not as the unlimited
heteroscedastic training protocol.

Reproduce the four-process timing and GPU-memory gate with:

```sh
baselines/dvi/run_homoscedastic_power_scale_gate.sh
```

It runs dDVI Power splits 2--5 concurrently by default. Set
`DVI_HOMO_GATE_METHOD=full` to repeat the gate for full DVI.
The script refuses an existing output directory so resume time cannot be
mistaken for a fresh timing result, and exits nonzero when the conservative
two-method projection exceeds ten hours.

After running independent `DVI_SPLIT_IDS` shards, combine them with:

```sh
julia --project=baselines/dvi baselines/dvi/merge_shards.jl \
  paper_materials/dvi_uci/<run>/dvi \
  paper_materials/dvi_uci/<run>/dvi_shards/*
```

## BibTeX

```bibtex
@inproceedings{wu2019deterministic,
  title     = {Deterministic Variational Inference for Robust Bayesian Neural Networks},
  author    = {Wu, Anqi and Nowozin, Sebastian and Meeds, Edward and
               Turner, Richard E. and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and
               Gaunt, Alexander L.},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  url       = {https://arxiv.org/abs/1810.03958}
}
```
