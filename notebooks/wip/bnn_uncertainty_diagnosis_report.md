 # Why the message-passing BNN was not GP-like, and what fixes it

Date: 2026-07-25

Companion to `notebooks/manyplus_uncertainty_experiment_report.md`, which
documented the ManyPlus/ResidualSine heteroscedastic arm and left open the
question of why the cubic benchmark stays confident in the interval it never
observed. This report answers that, retracts one of that report's conclusions,
and records the architecture that produces GP behaviour.

## Executive summary

1. **The predictive variance of the production arm cannot dip where the data
   are.** This is a structural fact, not a tuning problem: `ManyPlus` sums
   variances and the output weights carry one scalar marginal each, so the
   mean-function variance is a non-negative combination of fixed non-negative
   functions. Data can scale the whole curve down uniformly; carving a dip
   requires negative off-diagonal entries of `Cov(v)`, which mean field deleted.

2. **The reported cubic "gap detection" was an artifact and should be
   retracted.** With the same model, seeds and priors, moving the empty interval
   from `(-3, 3)` to `(1, 3)` drops the gap/observed ratio to **0.999** — no
   detection whatsoever — while the variance peak stays at `x ≈ 0`. Filling the
   gap entirely still leaves a 1.19x bulge at the origin. The published
   1.68–2.08 ratios measure the basis envelope, not data density.

3. **The `softdot` carrier noise was the mechanism that pinned the output
   weights.** It is not a small benign floor: under the mean-field split
   `q(out, h) q(v)` it acts as a pseudo-observation of `v'h` with precision
   `E[tau_c] = 1e4`. Hence `Var(v_k) ≈ 5e-7` against a prior of `1e-2`, with the
   posterior mean still at the prior mean.

4. **A dense output-weight posterior with `softdot` as the likelihood is exactly
   a GP,** verified against two independent analytic routes to `1e-12`. On the
   cubic benchmark it reaches a gap/observed contraction of **2892** against the
   production arm's 2.079, with observed-domain variance *below* the noise level.

## 1. The structural blocker

For any last-layer-linear model `f(x) = v' phi(x)`,

```math
\operatorname{Var}(f(x_*)) = \phi(x_*)^\top \Sigma_v \phi(x_*).
```

In exact Bayesian linear regression, `Σ_v = (Σ_0^{-1} + Φ'ΓΦ)^{-1}` is dense, and
the entire "pinch at the data, balloon in the gap" behaviour comes from its
negative off-diagonal entries. The production arm has none, for three
independent reasons, each verified in code:

| # | Mechanism | Location |
|---|---|---|
| 1 | `total_variance += var(m_i)` — variances add, no covariance term, no `1/H` | `src/ManyPlusNode/rules.jl:1-19` |
| 2 | `collect_factorisation` is overridden, so a declared joint cluster is *discarded at that node*; `NGMPDependencies` can never attach either | `src/ManyPlusNode/node.jl:25-26, 85-89` |
| 3 | No factor ever couples two neurons, and `softdot(θ=v, x=h, γ)` factorizes as `q(y,x)q(θ)q(γ)`, so `Cov(w_k, v_k) = 0` by construction | model at `notebooks/…separate_variance_squareplus.jl:281-285, 351-356` |

So

```math
\operatorname{Var}(q(\text{mean}))(x) = \sum_k\Big[\operatorname{Var}(v_k)\,\mathbb E[h_k(x)]^2
 + \mathbb E[v_k]^2\operatorname{Var}(h_k(x)) + \dots\Big] + \operatorname{Var}(\text{intercept}) + \text{floors},
```

which depends on `x` only through each neuron's own `|w_k' x̃|`.

`ManyPlus` has one further inconsistency worth recording: its free-energy term
uses the *exact dense joint entropy* (`score.jl:1-17`, rank-one sum coupling
retained) while its messages assume independence. Objective and updates disagree.

## 2. Variance budget (`experiments/bnn_uncertainty_diagnostics.jl`)

Plug-in decomposition from the four saved cubic posteriors, no retraining.
Shares of the observed-domain total:

| Term | Exp 8+8 | Exp 16+16 | Exp 16+16 scaled |
|---|---:|---:|---:|
| `T1 = Σ Var(v_k) E[h_k]²` | 0.0% | 0.0% | 0.0% |
| `T2 = Σ E[v_k]² Var(h_k)` | 86.9% | 92.4% | 90.5% |
| `T3 = Σ Var(v_k) Var(h_k)` | 0.0% | 0.0% | 0.0% |
| `T4 = H / E[tau_c]` | 2.2% | 2.0% | 3.7% |
| `T5 = Var(intercept)` | 10.8% | 5.7% | 5.7% |

Two readings, both important.

**T1 is zero because the output weights are pinned.** `Var(v_k)` ranges
`3.7e-7 … 7.1e-7` against a prior of `1e-2`; posterior sd / |mean| is `1.4e-3`;
and `|E[v_k]| = 0.4997` against a prior mean of `0.5`. The weights have neither
moved nor retained uncertainty. So the epistemic channel carries no output-weight
information at all — it is input-weight uncertainty pushed through the activation.

**The hump is the activation's amplification factor.** The exact residual-sine
variance is `Var(φ(Z)) = v·(1 + 2ρ E[cos ωZ]) + (ρ/ω)² Var(sin ωZ)`. Measured on
the 16-neuron scaled arm:

| x | mean pre-activation variance | mean amplification | mean `Var(h)` |
|---:|---:|---:|---:|
| 0.0 | 0.0720 | 1.894 | 0.1649 |
| 2.0 | 0.0732 | 1.420 | 0.1310 |
| 5.0 | 0.0797 | 0.214 | 0.0336 |

The pre-activation variance *rises* monotonically with `|x|`, so it cannot make a
hump. The amplification *falls* monotonically, and `Var(h)` follows it. The bulge
at `x = 0` is the neurons' prior biases (`±π/4`, where `cos > 0`) aligning there
and decohering further out. The cubic benchmark's gap happens to be centred on
exactly that point.

Also worth noting: the iterated prediction graph inflates the variance by
20–40% over the plug-in forward budget (budget/artifact ratio 0.71–0.84).

## 3. The decisive control (`experiments/cubic_gap_placement_control.jl`)

Same model, priors, seeds, architecture and inference settings; three data
designs over the same support; `n = 80`, Exp link, 8 neurons per head.

| design | empty interval | peak `x` | own gap / obs | `|x| ≤ 1` / obs |
|---|---|---:|---:|---:|
| centered | `(-3, 3)` | **0.10** | 1.631 | 1.811 |
| filled | none | **0.15** | – | 1.190 |
| shifted | `(1, 3)` | **-0.15** | **0.999** | 1.117 |

- The variance peak sits at `x ≈ 0` in **all three** designs, including `filled`,
  which has data everywhere.
- With the gap at `(1, 3)`, the model is **exactly as uncertain over its own empty
  interval as over the observed region** (0.999), and *more* uncertain at the
  origin where it does have data (1.117).

This is the direct answer to "why is it still so confident there": it was never
responding to the gap at all. A ratio near 2 on a positive-basis model is the
null result, not a weak positive.

## 4. The `softdot` carrier is the pinning mechanism

Under `q(y, x) q(θ) q(γ)`, the message toward `θ` carries site precision
`E[γ]·E[x x']`. So `c ~ softdot(v, h, tau_c)` with `E[tau_c] = 1e4` informs `v` as
if `v'h` had been observed with precision `1e4`. Raising the carrier to "remove"
it makes this strictly worse: a probe at `1e6` drove `Var(v)` to `~0` with `E[v]`
still at zero.

The asymmetry matters. `za ~ softdot(features, w, tau)` with `E[tau] = 1e3` is
**harmless**, because `w` shares a *structured* cluster with `za`, so the
near-determinism is handled exactly rather than as pseudo-data. Only the
mean-field-split edge is damaged. This is the concrete form of the project's
standing position that softdot is not a carrier substitute.

The correct construction is to make `softdot` **be** the likelihood,
`y ~ softdot(features, weights, real_precision)`, so the information reaching the
weights is exactly the BLR amount. Additive terms then belong as extra
coordinates of the weight vector rather than as separate mean-field variables.

## 5. The verified GP reference (`experiments/parametric_gp_reference.jl`)

```text
phi(x) in R^H            fixed features (random Fourier, or a frozen ResidualSine draw)
v ~ MvNormal(0, I)       DENSE H-dimensional weight posterior
y_o ~ softdot(phi(x_o), v, gamma)
```

PointMass features on the `θ` edge make this exact BLR, so the info-form fast path
at `src/nodes/softdot/rules/structured_info_form.jl:34-62` is the selected
dispatch. Weight-space/function-space duality makes it *identically* a GP with
kernel `k(x, x') = φ(x)'φ(x')`.

**Exactness**, cubic benchmark, `H = 128`, message passing vs analytic BLR vs the
function-space GP computed by an independent `n × n` route:

| quantity | max abs error |
|---|---:|
| weight mean | 9.5e-12 |
| weight covariance | 9.1e-13 |
| predictive mean | 5.3e-12 |
| predictive variance | 2.1e-13 |
| BLR vs function-space GP mean | 4.0e-12 |
| BLR vs function-space GP variance | 1.5e-13 |

**Behaviour**, same data as the production arm:

| arm | obs RMSE | gap RMSE | Var obs | Var gap | Var outer | gap/obs | out/obs |
|---|---:|---:|---:|---:|---:|---:|---:|
| production ManyPlus (Exp 16+16 scaled) | 24.87 | 17.19 | 905.9 | 1883.3 | 678.0 | 2.079 | 0.748 |
| GP reference, fixed noise | 1.12 | 18.13 | **1.17** | 3378 | 838 | **2892** | **717** |
| GP reference, learned noise | 1.15 | 15.35 | 4.66 | 3754 | 1430 | 806 | 307 |

The observed-domain variance falls *below* the true noise variance of 9, which is
what a GP does with ~40 points per wing. On the sine benchmark the reference
reaches observed RMSE **0.217** against the production arm's best 0.2884.

Two honest limitations of this arm: it is homoscedastic by construction, so it
cannot represent the sine benchmark's input-dependent noise (the production arm's
neural precision head genuinely wins there); and the learned-noise variant
overestimates the noise level (40.3 vs 9 on cubic), a standard mean-field bias
between `v` and `γ`. The fixed-noise variant is the exactness reference; the
learned-noise variant is the practical one.

## 6. Applying the fix to the actual BNN

### Arm A — dense output layer (`experiments/joint_output_layer_bnn.jl`)

Same ResidualSine network, but: `MvStack` instead of `ManyPlus`; `softdot` as the
likelihood with no output carrier; one dense `v`; intercept and an uncertain linear
slope as extra stack coordinates rather than mean-field additive terms; the noise
head's constant coordinate anchored at the ridge-fit residual log-precision; and
slope means **log-spaced** over `[0.35, 5] × 2.2` rad per standardized unit.

That last change matters independently of the output layer. The production arm's
slope means covered only 1.65–2.75 rad/unit, and the slope prior variance (0.05) is
far too tight for the posterior to travel. The sine benchmark's `sin(3 pi x)` needs
about 9.4 rad/unit, so the basis was **frequency-starved**. Widening the band moved
sine RMSE from 0.703 to 0.304.

| benchmark | metric | production | **Arm A** | GP reference |
|---|---|---:|---:|---:|
| cubic | obs RMSE | 24.87 | **3.53** | 1.15 |
| cubic | gap RMSE | 17.19 | **11.59** | 15.35 |
| cubic | Var obs (truth 9) | 905.9 | **28.4** | 4.66 |
| cubic | gap/obs | 2.079 | **2.65** | 806 |
| cubic | outer/obs | 0.748 | **2.22** | 307 |
| cubic | aleatoric (truth 9) | 0.347 | **29.3** | 40.3 |
| cubic | excess NLL | 2.240 | **0.605** | 0.418 |
| sine | obs RMSE | 0.2884 | 0.299 | 0.217 |
| sine | aleatoric (truth 0.204) | 0.1981 | 0.026 | 0.235 |
| sine | aleatoric correlation | 0.8290 | **0.972** | 0.000 |

The noise-head anchor went through one correction worth recording, because the first
version was wrong for an instructive reason. Anchoring on the residual of a ridge fit
to the raw `[1, x]` features cannot work on the cubic benchmark: that basis cannot
represent `x^3`, so almost all of its residual is *signal*, the anchor lands far too
low, and aleatoric comes out too high. Replacing it with the basis-free
first-difference (Rice / Gasser) estimator `Var ~ mean(diff(y)^2) / 2` improved cubic
across the board (aleatoric 35.6 -> 29.3, excess NLL 0.681 -> 0.605, obs RMSE
3.84 -> 3.53) and left sine essentially unchanged.

The qualitative change is clearest in the plot: `Var(q(mean))` now has **minima at
x ≈ ±3.7, inside both data wings, and a maximum at x = 0 in the gap** — a genuine
dumbbell. The production arm had a single smooth hump with no dips anywhere. Note
also `outer/obs` rising from 0.748 to 2.20: the production arm was *more* confident
beyond its data than inside it.

Two honest weaknesses remain, and they are the same weakness in two directions:

- cubic aleatoric is ~4x too high (35.6 vs 9); sine aleatoric is ~7x too low
  (0.028 vs 0.204), even though its *shape* is nearly perfect (correlation 0.976
  against the production arm's 0.829).
- With the wide frequency band the mean interpolates the sine noise, so the noise
  head compensates downward; with the cubic the mean slightly underfits the wings
  and the noise head compensates upward.

**This is a kernel-hyperparameter problem, not a message-passing problem.** The
frequency band plays the role of a GP lengthscale, and it is currently a *fixed
prior*. The GP reference gets aleatoric right (9.000 fixed / 0.235 on sine) because
its lengthscale was chosen by hand for each benchmark. Learning the band — or
selecting it by free energy — is the obvious next step, and it is a standard,
well-posed problem rather than a structural one.

### The same control applied to Arm A (`experiments/arm_a_gap_placement_control.jl`)

A larger contraction ratio proves nothing on its own — a larger *artifact* would
also report a larger ratio. So Arm A was put through the identical placement
control:

| design | empty interval | peak x | Var obs | own gap / obs | `|x| ≤ 1` / obs |
|---|---|---:|---:|---:|---:|
| centered | (−3, 3) | 0.00 | 32.07 | 2.503 | 3.314 |
| filled | none | **5.00** | 18.52 | – | **1.214** |
| shifted | (1, 3) | **5.00** | 22.08 | **1.313** | **1.142** |

Compare the production arm on the same control: peak 0.10 / 0.15 / −0.15, own-gap
ratio 0.999, origin ratio 1.117.

`results/uncertainty_diagnosis/arm_a_gap_placement_control.png` makes the result
plainer than the table does, and it is unambiguous:

- **centered**: a clean dumbbell — minima at x ≈ ±3.7 inside both data wings, a
  single maximum at x = 0 in the gap, spanning 27 to 113.
- **filled**: with data everywhere the profile *flattens* to 15–25 with no dominant
  central hump, rising only at the extrapolation edges. The production arm kept its
  central peak here.
- **shifted**: a **local maximum sits inside the shaded (1, 3) band**, peaking at
  x ≈ 2 at ~30 while the origin region sits at ~24. The uncertainty followed the
  gap.

Two things therefore changed qualitatively:

- **The profile is no longer pinned to the origin.** With the gap filled, the
  interior hump disappears and the origin ratio falls to 1.214. The production
  arm's peak stayed at x ≈ 0.15 in every design.
- **The ordering flips the right way.** On the shifted design Arm A is more
  uncertain over its *own* empty interval (1.313) than at the origin (1.142). The
  production arm was the other way round (0.999 vs 1.117) — less uncertain in its
  gap than where it had data.

So Arm A's gap response is **genuinely data-driven, but weak**: 1.31x over its own
gap, against the GP reference's 806x. The residual difference is explained by what
Arm A still keeps from the old design — the input weights remain uncertain, and
`E[v]' V_h E[v]` is a non-contracting additive floor exactly like the production
arm's dominant term, just no longer the *only* term. The GP reference has no such
floor because its features are deterministic.

The "peak inside the gap" half of the pass condition turned out to be a badly chosen
criterion: the linear stack coordinate makes variance grow with `|x|`, so the global
maximum over `|x| <= 5` always lands at the boundary. The ratio comparison is the
informative half.

The convergence audit is the one place where the retained iterated-prediction
protocol still bites, and it bites unevenly:

| benchmark | max relative drift, 200 -> 400 prediction iterations |
|---|---:|
| cubic gap | 2.1e-2 |
| heteroscedastic sine | **3.9e-1** |

The cubic numbers are stable to ~2%. The **sine numbers are not converged at all** --
38% drift on doubling. That changes the interpretation above: the sine aleatoric
underestimate (0.026 against 0.204) may be substantially an artifact of an
unconverged prediction graph rather than of the mean absorbing the noise, and it
should not be quoted as a model property until the graph converges. This is the
quantitative form of the concern raised before the protocol was chosen -- iterating a
mean-field fixed point on an unobserved subgraph is mode-seeking, and the earlier
report's own table showed variance falling monotonically 7.66 -> 0.55 over 16 -> 1000
iterations. The audit at least makes the problem visible per benchmark instead of
hiding it behind a fixed cap. The placement-control runs drift 3.2e-2, so the
cubic-based conclusions in this report are safe.

### Arm B — dense over all weights (`experiments/joint_all_weights_bnn.jl`)

Negative result, reported as such. Replacing the per-neuron input layer with
`ContinuousTransition` + `LinearReshapeMeta` (dense `vec(W)`, including cross-neuron
covariance) and `MvResidualSine` (exact joint activation covariance) makes things
**worse**:

| | Arm A | Arm B |
|---|---:|---:|
| cubic obs RMSE | 3.84 | 16.79 |
| cubic gap RMSE | 11.24 | 19.00 |
| cubic gap/obs | 2.66 | **0.778** |
| cubic aleatoric (truth 9) | 35.6 | 313.4 |

`gap/obs` below 1 means it is *less* uncertain in the gap than at the data. The
output weight posterior collapses again (`q(v)` diagonal `9e-5 … 2.6e-4` against a
prior of `0.083`), and the dense `vec(W)` covariance that Arm A cannot represent
turns out to be small anyway (max off-diagonal `5e-4` against diagonals of `3e-2`).

A transition-precision sweep rules out the obvious explanation: `P` mean
20 / 200 / 2000 gives `gap/obs` 0.778 / 0.695 / 0.688 and *identical* weight
posteriors to four digits, so the injected transition noise is not the cause.

Per the plan's stated criterion — Arm B should match or beat Arm A, otherwise the
extra cost is not justified — **Arm A is the answer**, and cross-neuron input-weight
covariance is not what was missing. Arm B does carry two forced differences worth
noting before writing it off entirely: it has no dedicated constant/linear stack
coordinates (there is no node that concatenates a vector edge with scalars), and its
hidden layer is genuinely stochastic.

## 7. Supporting node: `MvStack`

`src/MvStackNode/` — the lossless counterpart of `ManyPlus`. It gathers scalar
neurons into a vector edge instead of pre-summing them:

- forward: the joint Gaussian of the independent inbound scalars;
- backward to `in_k`: the `k`-th coordinate marginal of
  `m_out × Π_{j≠k} m_j`, which *does* carry back the correlations the dense weight
  posterior has learned — the step `ManyPlus` cannot express;
- free energy: the exact negative entropy of the node's `H`-dimensional local
  belief, consistent with the messages (unlike `ManyPlus`).

Stacking is a bijection of `R^H` with unit Jacobian, so all of this is exact BP.
25 tests in `test/mv_stack_tests.jl`, registered in `runtests.jl`. One
implementation note: the forward covariance must be a dense `Matrix`, not a
`Diagonal` — downstream softdot marginals accumulate off-diagonal mass in place.

## What is retracted from the earlier report

- §"Epistemic uncertainty responds to data density" and the claim that all four
  models "detect the empty interval": the gap/observed ratios of 1.68–2.08 do not
  measure data density. See §3.
- The framing of `tau_c` as a benign width-scaling floor. Its share of the
  variance budget is small (2–4%), but its effect on the output weights is the
  dominant pathology. See §4.

What stands: the mean-fitting results, the heteroscedastic aleatoric head's
success on the sine benchmark (0.1981 against a truth of 0.2050), the
prediction-convergence correction, and the observation that width changes the
uncertainty scale.

## Reproducing

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. experiments/bnn_uncertainty_diagnostics.jl
OPENBLAS_NUM_THREADS=1 julia --project=. experiments/cubic_gap_placement_control.jl
GP_REF_H=128 GP_REF_LENGTHSCALE=0.2 GP_REF_ITERATIONS=60 \
  OPENBLAS_NUM_THREADS=1 julia --project=. experiments/parametric_gp_reference.jl
OPENBLAS_NUM_THREADS=1 julia --project=. experiments/joint_output_layer_bnn.jl
```

Benchmarks and metrics are defined once in `experiments/uq_benchmarks.jl` so
every arm is scored on byte-identical data. The contraction-ratio triple
(observed / gap / beyond-data) is the statistic that encodes GP-likeness; RMSE and
coverage alone hid this failure for the whole earlier report.
