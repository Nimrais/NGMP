# ManyPlus ResidualSine uncertainty experiments

Date: 2026-07-25

## Executive summary

The original regression model can learn a nonlinear mean, but its single
global `obs_noise` variable cannot learn input-dependent aleatoric uncertainty.
That limitation is structural: no amount of data or additional mean neurons
can make a global scalar noise variable depend on `x`.

An explicit neural precision head was therefore added:

```text
mean network:       x -> ResidualSine features -> summed mean
precision network:  x -> ResidualSine features -> summed score -> positive link
likelihood:          y ~ NormalMeanPrecision(mean(x), precision(x))
```

Both heads are inferred together in one full-data `infer` call. There is no
training batching, no posterior-as-prior batching, and no frozen pretrained
mean. The prediction graph keeps the learned weights joint with the local
forward variables and reports the direct posterior `var(q(y*))`.

The current conclusions are:

1. **Input-dependent aleatoric uncertainty is representable and is learned on
   the sine benchmark.** With 240 observations, the learned mean aleatoric
   variance is `0.1981`, versus the true grid mean `0.2050`, with correlation
   `0.8290`.
2. **Epistemic uncertainty responds to data density.** On the cubic gap
   benchmark it is larger in the empty interval than in the observed wings.
   Depending on width, the gap/observed epistemic ratio is `1.68` to `2.08`.
3. **The cubic benchmark is not calibrated yet.** Squareplus makes the
   aleatoric variance far too large; Exp makes it too small. The cubic mean is
   also visibly underfit after the 100-iteration cap.
4. **Network width currently changes the uncertainty scale.** Doubling both
   heads from 8 to 16 neurons improves the mean but greatly increases absolute
   epistemic variance and drives the Exp aleatoric variance toward zero.
   Scaling the output-weight priors reduces, but does not eliminate, this
   problem.
5. **An early prediction plot was invalid because prediction message passing
   had not converged.** A 16-step prediction produced mean total variance
   `7.66`; a conservative 1,000-step audit produced `0.551`, matching the
   corrected accelerated 100-step prediction (`0.558`).

The present model is therefore a useful experimental heteroscedastic Bayesian
neural network, but it is not yet a calibrated general-purpose uncertainty
model.

## Synthetic benchmarks

### Aleatoric benchmark

```math
x \sim \mathcal N(0,1)
```

```math
y = -(x + 0.5)\sin(3\pi x)
    + \mathcal N\left(0,\,[0.45(x+0.5)]^2\right).
```

The true conditional variance is:

```math
\operatorname{Var}(y\mid x)=[0.45(x+0.5)]^2.
```

It is computed analytically from the data-generating equation. It is not
estimated from the training samples.

### Epistemic benchmark

```math
y=x^3+\mathcal N(0,9)
```

with half the inputs drawn from `Uniform(-5,-3)` and half from
`Uniform(3,5)`. There are no training observations in `(-3,3)`.

The correct aleatoric variance is constant and equal to `9`. A successful
epistemic model should have more mean-function uncertainty inside the empty
interval than in the observed wings.

All reported runs use fixed random seeds. Synthetic observations are sampled
to create a training set, but posterior predictive uncertainty is propagated
by message passing; it is not estimated with posterior Monte Carlo.

## Model evolution

### 1. Original model: nonlinear mean with global noise

The first notebook preserved the structure of the XOR model:

```julia
obs_noise ~ priors[:obs_noise]
y[observation] ~ NormalMeanPrecision(
    out[observation] + intercept,
    obs_noise,
)
```

This model has one `obs_noise` variable for every `x`. It can learn a nonlinear
mean and epistemic uncertainty from uncertain weights, but it can learn only
static aleatoric uncertainty.

### 2. Shared-hidden two-head model

The first heteroscedastic graph reused the same hidden activations for the
mean and precision heads:

```math
\begin{aligned}
h_i(x) &= \operatorname{ResidualSine}
          (\operatorname{softdot}(x,w_i,\tau)),\\
\mu(x) &= b_\mu + \sum_i v_i h_i(x),\\
\eta(x) &= b_\lambda + \sum_i g_i h_i(x),\\
\lambda(x) &= L(\eta(x)),\\
y(x) &\sim \operatorname{NormalMeanPrecision}(\mu(x),\lambda(x)).
\end{aligned}
```

`L` is either `Exp` or `Squareplus`. There is one shared `tau` and one shared
`tau_c`, rather than a different pair for every neuron.

This made precision depend on `x`, but sharing the complete hidden
representation forced the mean and noise functions to use the same basis.
The resulting direct predictive variance was badly inflated.

### 3. Independent mean and precision networks

The next graph uses two independent ResidualSine networks:

```text
mean:       x -> w       -> za       -> h       -> v       -> sum -> intercept
precision:  x -> noise_w -> noise_za -> noise_h -> noise_v -> sum
                                                               |
                                                     precision intercept
                                                               |
                                                     Exp or Squareplus
```

The networks have separate:

- input weights;
- hidden activation variables;
- output weights;
- `tau`;
- `tau_c`.

They still meet in the same `NormalMeanPrecision` likelihood and are learned
together in one inference call.

The optional per-observation Gamma precision prior was tested and then removed
for the best sine benchmark. Removing it avoids the Gamma prior and neural
positive-link factor simultaneously competing to determine each local
precision.

### 4. Width-scaled output priors

The 16-neuron width-scaled run uses the 8-neuron model as a reference:

```math
\text{weight mean and standard deviation scale}
=\sqrt{8/16}=0.70710678,
```

and therefore:

```math
\text{weight variance scale}=8/16=0.5.
```

This scaling is applied to the output weights of both the mean and precision
networks, including the precision-head initialization. It changes the prior
parameterization only; it does not insert a fixed normalization node in the
graph.

## Training and prediction protocol

The final comparable runs use:

- one full-data training `infer` call;
- a hard cap of 100 training iterations;
- an intercept in the mean;
- one shared `tau` and `tau_c` within each network;
- 8 or 16 neurons per network;
- a separate direct prediction graph;
- 100 prediction iterations;
- prediction damping `alpha = 0.5` for the nonlinear activation and link;
- prediction NGMP `max_step = 1.0`;
- a diffuse `NormalMeanVariance(0, 10^12)` terminal message on each unobserved
  `y*`.

The structured prediction cluster keeps the uncertain weights with the local
forward path. The learned posterior is therefore not copied into an
independent point estimate before prediction.

The plotted total uncertainty is:

```math
\operatorname{Var}(q(y^*)),
```

read directly from the prediction graph.

For diagnosis, it is compared with:

```math
\operatorname{Var}(q(\mu(x)))
+\frac{1}{\mathbb E_q[\lambda(x)]}.
```

The second term is the effective conditional variance used by the
`NormalMeanPrecision` VMP update. It is not generally identical to
`E_q[1/lambda]`. The direct `var(q(y*))` remains the primary reported
predictive quantity.

## Prediction convergence correction

The first direct-`q(y*)` image used only 16 slowly damped prediction
iterations. It showed:

| Prediction iterations | Total `Var(q(y*))` | Mean-function variance | Aleatoric variance |
|---:|---:|---:|---:|
| 16 | 7.6596 | 7.5522 | 0.1108 |
| 100, conservative damping | 1.4271 | 1.2402 | 0.1872 |
| 200, conservative damping | 0.8580 | 0.6529 | 0.2052 |
| 500, conservative damping | 0.5953 | 0.3697 | 0.2257 |
| 1,000, conservative damping | 0.5509 | 0.3056 | 0.2454 |
| 100, accelerated prediction | 0.5585 | 0.3056 | 0.2528 |

Thus the large early band was mainly an un-converged prediction graph, not a
learned posterior property. The accelerated 100-step schedule agrees closely
with the conservative 1,000-step audit and is used by the later experiments.

## Aleatoric benchmark results

The comparable re-audited 60-observation ablations are:

| Training configuration | Mean RMSE | Direct `Var(q(y*))` | Mean-function variance | Aleatoric variance | Aleatoric correlation |
|---|---:|---:|---:|---:|---:|
| No Gamma prior, slow training | 0.6732 | 0.9141 | 0.6599 | 0.2542 | 0.8373 |
| Gamma prior, faster damping | 0.5058 | 5.1099 | 0.4911 | 4.6280 | 0.7704 |
| No Gamma prior + faster damping | **0.4064** | **0.5585** | **0.3056** | **0.2528** | **0.8372** |

The true mean conditional variance over the evaluated grid is `0.2050`.

The combined no-Gamma/faster-damping training is the best current
60-observation result. The Gamma-prior/faster-damping case demonstrates that a
high correlation alone is insufficient: its aleatoric curve has roughly the
right shape but the wrong magnitude.

Several earlier plots show total variances around `9` to `13`. Those plots
used the old under-converged prediction schedule and must not be used for a
quantitative comparison.

### Four times more observations

Using 240 instead of 60 observations, with the same 8+8 architecture and
100-iteration cap:

| Metric | 60 observations | 240 observations |
|---|---:|---:|
| Mean RMSE | 0.4064 | **0.2884** |
| Direct mean `Var(q(y*))` | 0.5585 | **0.3213** |
| Mean-function variance | 0.3056 | **0.1231** |
| Aleatoric variance | 0.2528 | **0.1981** |
| True variance | 0.2050 | 0.2050 |
| Aleatoric correlation | 0.8372 | 0.8290 |

Four times more data reduced mean epistemic variance by about 60% and direct
total predictive variance by about 42%. The average aleatoric magnitude became
nearly correct, although the learned curve still underestimates the rapidly
increasing variance near the outer edges.

## Cubic epistemic benchmark results

All cubic runs use the same 80 observations, seeds, priors, and 100 training
plus 100 prediction iterations. Only the positive link, width, or width
scaling changes.

| Model | Observed mean RMSE | Gap mean RMSE | Observed epistemic | Gap epistemic | Gap/observed epistemic | Observed aleatoric | Direct observed total | Direct gap total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Squareplus, 8+8 | 29.06 | 24.54 | 724.2 | 1337.5 | 1.847 | 1590.49 | 2315.6 | 2882.9 |
| Exp, 8+8 | 27.48 | 22.64 | 582.9 | 980.8 | 1.683 | 2.093 | 585.1 | 983.7 |
| Exp, 16+16 | **24.76** | 18.58 | 1755.0 | 3539.7 | 2.017 | 0.031 | 1755.0 | 3539.8 |
| Exp, 16+16, scaled | 24.87 | **17.19** | 905.9 | 1883.3 | **2.079** | 0.347 | 906.2 | 1885.0 |

The true aleatoric variance is `9`.

Interpretation:

- All four models detect the empty interval because their gap epistemic
  variance exceeds their observed-domain epistemic variance.
- More neurons improve the mean approximation and strengthen relative gap
  detection.
- More neurons do **not** improve absolute uncertainty calibration.
- Width-scaled priors approximately halve the 16-neuron epistemic variance
  relative to the unscaled model, but it remains higher than in the
  8-neuron model.
- The predictive intervals are so wide that gap coverage is `1.0`; that value
  is not evidence of good calibration.

### Why Squareplus and Exp fail in opposite directions

The cubic response standard deviation is approximately `76.6`. Consequently,
the true standardized noise variance is about:

```math
9 / 76.6^2 \approx 0.00153,
```

which corresponds to standardized precision near `652`.

For Squareplus, large positive outputs grow approximately linearly. Its score
therefore needs to reach roughly `652`. The learned mean precisions remained
only about `3.60` to `3.85`, producing data-scale aleatoric variances around
`1526` to `1632`.

For Exp, the required score is only:

```math
\log(652)\approx 6.48.
```

The 8-neuron Exp model instead learned mean precisions around `1630` to
`6057`, producing data-scale aleatoric variances around `0.97` to `3.60`.
Thus Exp solves the Squareplus dynamic-range problem but overshoots and becomes
overconfident about observation noise.

## Script map

### Original reference and first benchmark

| Script | Purpose |
|---|---|
| `notebooks/xor_manyplus_residual_sine_batched_ngmp.jl` | User's original nonlinear XOR/checkerboard model used as the architectural reference. |
| `notebooks/manyplus_residual_sine_aleatoric_epistemic.jl` | First full-data synthetic benchmark. Includes intercept and direct structured prediction, but retains one global `obs_noise`; therefore it cannot represent heteroscedastic aleatoric noise. |

### Shared-hidden joint precision head

| Script | Purpose |
|---|---|
| `notebooks/manyplus_residual_sine_joint_heteroscedastic.jl` | Core shared-hidden model and direct `q(y*)` prediction. Supports Exp or Squareplus. No batching; shared `tau` and `tau_c`; default full run is 60 observations, 8 neurons, and 100 iterations. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_shared_tau_qy_squareplus.jl` | Controlled Squareplus wrapper for the shared-hidden model. |

### Independent mean and precision networks

| Script | Purpose |
|---|---|
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl` | Core independent-network model. Despite the historical filename, it now supports both Exp and Squareplus. It also contains optional reference-width output-prior scaling. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_separate_variance_no_gamma.jl` | Removes only the per-observation Gamma precision prior. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_separate_variance_fast_damping.jl` | Keeps the Gamma prior and increases nonlinear training damping from `0.005` to `0.05`. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_separate_variance_no_gamma_fast_damping.jl` | Combines no Gamma precision prior with `0.05` training damping; best current 60-observation sine configuration. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_more_data.jl` | Runs the best sine configuration with 240 observations by default. |
| `notebooks/manyplus_residual_sine_joint_heteroscedastic_prediction_convergence_audit.jl` | Reloads learned posteriors without retraining and audits direct prediction convergence at retained checkpoints. |

### Cubic epistemic benchmark

| Script | Purpose |
|---|---|
| `notebooks/manyplus_residual_sine_joint_cubic_epistemic.jl` | Base cubic gap benchmark; Squareplus is the default link. |
| `notebooks/manyplus_residual_sine_joint_cubic_epistemic_exp.jl` | Controlled 8+8-neuron Exp-link runner. |
| `notebooks/manyplus_residual_sine_joint_cubic_epistemic_exp_16neurons.jl` | Controlled 16+16-neuron Exp capacity test without width scaling. |
| `notebooks/manyplus_residual_sine_joint_cubic_epistemic_exp_16neurons_widthscaled.jl` | Controlled 16+16-neuron Exp test with output priors scaled relative to the 8-neuron reference. |

## Important saved plots and posterior artifacts

The current artifacts are stored under `/tmp`, so they are useful now but are
not permanent repository assets.

| Experiment | Plot | Serialized posterior/result |
|---|---|---|
| Corrected 60-point sine | `/tmp/manyplus_joint_separate_variance_no_gamma_fast_damping_qy.png` | `/tmp/manyplus_joint_separate_variance_no_gamma_fast_damping_qy_posteriors.jls` |
| Conservative prediction audit | `/tmp/manyplus_joint_prediction_convergence_audit_1000.png` | `/tmp/manyplus_joint_prediction_convergence_audit_1000.jls` |
| 240-point sine | `/tmp/manyplus_joint_separate_variance_n240_qy.png` | `/tmp/manyplus_joint_separate_variance_n240_qy_posteriors.jls` |
| Cubic Squareplus 8+8 | `/tmp/manyplus_joint_cubic_epistemic.png` | `/tmp/manyplus_joint_cubic_epistemic_posteriors.jls` |
| Cubic Exp 8+8 | `/tmp/manyplus_joint_cubic_epistemic_exp.png` | `/tmp/manyplus_joint_cubic_epistemic_exp_posteriors.jls` |
| Cubic Exp 16+16 | `/tmp/manyplus_joint_cubic_epistemic_exp_16neurons.png` | `/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_posteriors.jls` |
| Cubic Exp 16+16 scaled | `/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_widthscaled.png` | `/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_widthscaled_posteriors.jls` |

The older `/tmp/manyplus_joint_shared_tau_qy*.png` and early
`separate_variance_*.png` files are exploratory. Some contain the
under-converged prediction band and should not be treated as final results.

## How to rerun the principal experiments

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. \
  notebooks/manyplus_residual_sine_joint_heteroscedastic_separate_variance_no_gamma_fast_damping.jl
```

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. \
  notebooks/manyplus_residual_sine_joint_heteroscedastic_more_data.jl
```

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. \
  notebooks/manyplus_residual_sine_joint_cubic_epistemic_exp.jl
```

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. \
  notebooks/manyplus_residual_sine_joint_cubic_epistemic_exp_16neurons_widthscaled.jl
```

The environment currently resolves local development copies of RxInfer
`5.5.0` and ReactiveMP `6.3.3`, plus ProbabilisticEnsembling `0.0.1`.

## What works

- Nonlinear mean dependencies can be learned.
- Training can be performed in one full-data inference call without batches.
- An intercept is included.
- `tau` and `tau_c` can be shared within a network.
- A second neural graph can make precision explicitly dependent on `x`.
- Exp and Squareplus are implemented as nodes inside the RxInfer graph.
- Mean and precision are learned jointly.
- Learned weight uncertainty is retained during prediction.
- The plotted total variance comes directly from `q(y*)`.
- The prediction decomposition is numerically consistent: later runs have
  decomposition residuals tiny relative to their total variances.
- More observations reduce epistemic uncertainty on the sine task.
- The cubic empty interval produces higher relative epistemic uncertainty.

## What does not work yet

### 1. Cubic aleatoric calibration

Squareplus overestimates the true variance `9` by roughly two orders of
magnitude, while Exp underestimates it. The positive link and score scale are
not calibrated to the standardized target precision.

### 2. Cubic mean fit

Even 16 neurons do not fit the observed cubic wings well after 100 training
iterations. The model reaches the iteration cap rather than demonstrating
clear convergence. A wide variance head can hide mean error, while an
overconfident Exp head can make the coupled updates difficult in the opposite
direction.

### 3. Width-invariant epistemic uncertainty

`ManyPlus` currently sums neuron contributions without a fixed width
normalization. Additional uncertain neurons therefore add posterior variance.
Prior scaling helps, but learned weights can undo it. A fixed graph operation
equivalent to:

```math
\frac{1}{\sqrt H}\sum_{i=1}^H c_i
```

should be tested for both the mean score and precision score.

### 4. Robust convergence evidence

The full fits use a 100-iteration safety cap. Most reach the cap; this is not
proof that the variational fixed point has converged. Free-energy changes in
the sine experiments remain around `10^-3` in the final iterations.

### 5. Exact aleatoric decomposition

`1/E[q(precision)]` is the effective VMP conditional variance, not the exact
`E_q[1/precision]`. Direct `var(q(y*))` is valid as the graph's predictive
variance, but a rigorous law-of-total-variance report should also compute or
approximate `E_q[1/precision]`.

### 6. Statistical robustness

The comparisons use one fixed seed. There is no multi-seed distribution of
RMSE, NLL, calibration error, or gap ratios yet.

### 7. No GP/state-space uncertainty process

The GP-via-SSM idea was discussed but not implemented. The present epistemic
dependence on `x` comes from propagating uncertain neural weights through
`h(x)`. The aleatoric dependence comes from the explicit neural precision
head. There is no correlated latent variance process over ordered `x`.

### 8. Repository persistence

The experiment notebooks are currently untracked by Git, and the result
artifacts are in `/tmp`. They need to be reviewed and committed or moved to a
stable result directory before they can be considered permanent.

## Recommended next experiments

In priority order:

1. Insert a **fixed** `1/sqrt(n_neurons)` scaling node after both `ManyPlus`
   sums and rerun normalized 8- and 16-neuron models.
2. Calibrate the precision score in standardized units, initially with a
   stronger prior centered near the known standardized precision on the cubic
   diagnostic.
3. Run beyond 100 training iterations while retaining free-energy and
   posterior checkpoints to determine whether the cubic mean is merely slow
   or stuck.
4. Compare joint-from-scratch training with a controlled mean warm start,
   followed by joint updates. The final model can still be joint; the warm
   start only changes initialization.
5. Report `E_q[1/precision]` in addition to `1/E_q[precision]`.
6. Repeat all final comparisons over multiple seeds and report RMSE, NLL,
   interval coverage, calibration error, and gap/observed uncertainty ratios.

