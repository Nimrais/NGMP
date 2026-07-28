# Bayes by Backprop (Monte Carlo VI) on UCI regression

This folder implements **Bayes by Backprop (BBB)**, the Monte Carlo
variational-inference algorithm introduced by:

> Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra.
> *Weight Uncertainty in Neural Networks*. arXiv preprint arXiv:1505.05424,
> 2015. Published as *Weight Uncertainty in Neural Network* in the Proceedings
> of the 32nd International Conference on Machine Learning,
> PMLR 37:1613–1622.
> [PMLR paper](https://proceedings.mlr.press/v37/blundell15.html) ·
> [arXiv:1505.05424](https://arxiv.org/abs/1505.05424)

“MCVI” is used here descriptively for Monte Carlo variational inference. The
name used by Blundell et al. is **Bayes by Backprop**.

## What is implemented from Blundell et al. (2015)

The implementation in [`src/model.jl`](src/model.jl) follows these parts of
the paper:

- Equation (1): minimize variational free energy,
  `KL[q(w|θ) || p(w)] - E_q[log p(D|w)]`.
- Equation (2): estimate the complete free energy with weight samples,
  `log q(w|θ) - log p(w) - log p(D|w)`. The code does not substitute an
  analytic KL. The same sampled weights are used in all three terms, as
  required by the paper's common-random-number estimator.
- Section 3.2: a fully factorized Gaussian posterior with
  `σ = log(1 + exp(ρ))` and reparameterized samples
  `w = μ + σ ⊙ ε`, `ε ~ N(0,I)`.
- Section 3.4, Equation (8): uniform minibatch allocation of the complexity
  cost. In the per-observation objective this is implemented as
  `(log q - log p) / N + mean_batch(NLL)`.

Every weight and bias in all three affine layers has its own learned `μ` and
`ρ`. Thus the model learns epistemic uncertainty in every neuron feature,
without supplied RFF features.

Blundell et al. study both Gaussian and scale-mixture Gaussian priors. This
benchmark uses the paper's **Gaussian-prior variant**, not the scale-mixture
variant, because the UCI comparison protocol below explicitly fixes a
Gaussian prior with mean `0` and standard deviation `1`.

## Where the UCI setup comes from

Blundell et al. do not report this six-dataset UCI benchmark. The architecture
and optimizer defaults come from Appendix F.3 of:

> Alexander Tschantz, Magnus Koudahl, Hampus Linander, Lancelot Da Costa,
> Conor Heins, Jeff Beck, and Christopher Buckley. *Bayesian Predictive
> Coding*. arXiv:2503.24016, 2025.
> [paper](https://arxiv.org/abs/2503.24016)

Specifically, Appendix F.3 states:

- two hidden layers with 50 hidden units;
- Adam with learning rate `0.001`;
- Gaussian prior mean `0`, standard deviation `1`;
- minibatch size `100`;
- 20 posterior samples when computing LPD.

Those values are the defaults here. The network uses ReLU activations and a
factorized posterior over the two hidden layers and output layer.

## Explicit local choices and non-reproduction caveats

This is an implementation of the cited algorithm under the cited UCI
settings; it is **not claimed to be an exact reproduction of either paper's
reported table**. The papers do not jointly specify all details needed to
reproduce that table. The following choices are local and are recorded in
`config.toml` for every run:

- one posterior weight sample per training minibatch; Blundell et al.
  consider 1, 2, 5, or 10, while Tschantz et al. specify only the 20 samples
  used for LPD evaluation;
- train-only feature and target standardization;
- deterministic 90/10 train/test splits, repeated over 20 seeds;
- inner validation for epoch selection, followed by reinitialization and
  refitting on the complete outer training partition;
- uniform initialization of posterior means and initial posterior standard
  deviation `0.05`; posterior-scale initialization is not given by the UCI
  protocol;
- a heteroscedastic Gaussian output by default, parameterized as
  `(mean, log variance)` with variance `exp(log variance)`. This output
  parameterization follows Anqi Wu et al., *Deterministic Variational
  Inference for Robust Bayesian Neural Networks*,
  [arXiv:1810.03958](https://arxiv.org/abs/1810.03958), which is the
  uncertainty setup referenced by Tschantz et al.; it is not an explicit UCI
  specification in Blundell et al.;
- clipping predicted log variance to `[-20, 20]` for numerical stability;
- finite-mixture predictive density computed with `logmeanexp` over the 20
  posterior draws.

Both standardized and original-unit densities are saved.
`lpd_original = lpd_standardized - log(y_scale)` applies the exact Jacobian
for target standardization. RMSE and coverage are in original target units.

## Run

Run the mathematical and integration tests:

```sh
julia --project=baselines/bbb_uci baselines/bbb_uci/test/runtests.jl
```

Run a short Yacht smoke test:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BBB_DATASETS=yacht BBB_SPLITS=1 BBB_LIKELIHOODS=heteroscedastic \
BBB_MAX_EPOCHS=10 BBB_MIN_EPOCHS=5 BBB_PATIENCE=2 \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

Run the default 120 configurations (six datasets × 20 splits × one
likelihood):

```sh
DATADEPS_ALWAYS_ACCEPT=true \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

Run both likelihoods for all 20 splits (240 configurations):

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BBB_LIKELIHOODS=homoscedastic,heteroscedastic \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

Outputs are written incrementally under `results/bbb_uci/<timestamp>/`.
Every `runs.csv` row contains the method and protocol references. Set
`BBB_OUTPUT_DIR` to reuse a directory; successful configurations are skipped.
Each result directory also contains `split_manifest.jld2`. BBB and DVI both
use the versioned `repeated-holdout-v1` implementation in
`SurrogateModelling`, so independently launched runs produce identical outer
and inner row indices for the same dataset, split seed, and fractions.

Versioned posterior checkpoints are written atomically under `checkpoints/`.
They contain the complete factorized weight posterior, standardizer, exact
split indices, prediction configuration, and evaluation seed. Replay the
saved holdout prediction with:

```julia
using BBBUCI

checkpoint = load_posterior_checkpoint(
    "results/bbb_uci/<run>/checkpoints/yacht_split01_heteroscedastic.jld2",
)
prediction = predict_holdout(checkpoint)
prediction.metrics
```

Resume skips a successful configuration only when its versioned checkpoint
also loads successfully. Reusing an output directory with different
result-affecting settings is rejected.

Result directories created by the earlier implementation lack the `method`,
`method_reference`, and `uci_protocol_reference` columns. They used a
closed-form Gaussian KL and must not be presented as results of the sampled
Equation (2) estimator implemented here.

Important overrides include `BBB_DATASETS`, `BBB_SPLITS`,
`BBB_LIKELIHOODS`, `BBB_TRAIN_SAMPLES`, `BBB_EVAL_SAMPLES`,
`BBB_MAX_EPOCHS`, `BBB_BATCH_SIZE`, `BBB_HIDDEN_UNITS`,
`BBB_INITIAL_POSTERIOR_STD`, `BBB_OUTPUT_DIR`, and `BBB_RESUME`.

## BibTeX

```bibtex
@InProceedings{pmlr-v37-blundell15,
  title     = {Weight Uncertainty in Neural Network},
  author    = {Blundell, Charles and Cornebise, Julien and
               Kavukcuoglu, Koray and Wierstra, Daan},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
  pages     = {1613--1622},
  year      = {2015},
  volume    = {37},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v37/blundell15.html}
}

@misc{tschantz2025bayesian,
  title         = {Bayesian Predictive Coding},
  author        = {Tschantz, Alexander and Koudahl, Magnus and Linander, Hampus
                   and Da Costa, Lancelot and Heins, Conor and Beck, Jeff
                   and Buckley, Christopher},
  year          = {2025},
  eprint        = {2503.24016},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
