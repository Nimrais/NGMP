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

- Adam with learning rate `0.001`;
- minibatches of 100;
- posterior initialization based on the `wider_he` setting in the authors'
  public code: weight variance `5/fan_in`, zero bias means, and bias variance
  divided by 10;
- elementwise gradient clipping to `[-0.1, 0.1]`, matching the authors'
  public training utility;
- 10,000 maximum epochs, with the KL off for 7,000 epochs and annealed over
  500 epochs. This preserves the 70% warmup ratio used by the authors'
  released toy notebook (14,000 of 20,000 updates), but is a local adaptation,
  not a UCI setting stated in the paper;
- train-only feature and target standardization;
- inner validation for selecting the epoch, followed by a fresh refit on all
  outer-training observations; selection and early stopping begin only after
  any configured KL warmup and annealing are complete;
- validation patience of 500 checks.

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

Run diagonal-DVI:

```sh
DATADEPS_ALWAYS_ACCEPT=true DVI_PROPAGATION=diagonal \
julia --project=baselines/dvi baselines/dvi/run.jl
```

### Checked Yacht reproduction

The following one-split command was run locally:

```sh
DATADEPS_ALWAYS_ACCEPT=true OPENBLAS_NUM_THREADS=1 \
DVI_DATASETS=yacht DVI_SPLITS=1 DVI_PROPAGATION=diagonal \
DVI_MAX_EPOCHS=10000 DVI_KL_WARMUP_EPOCHS=7000 \
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

This is a close one-split check, not a reproduction of the paper's 20-split
mean. The split seed and the undisclosed original UCI optimizer details need
not match the authors' experiment.

Outputs are written incrementally under `results/dvi/<timestamp>/`.
Successful configurations can be resumed by reusing `DVI_OUTPUT_DIR`; a
configuration is skipped only when its checkpoint is present and valid.

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
