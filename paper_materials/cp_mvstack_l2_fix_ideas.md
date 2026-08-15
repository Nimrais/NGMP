# Making L=2 Beat L=1 on Yacht: Diagnosis and Fix Catalogue

Companion to `cp_mvstack_yacht_20splits_findings.md`. That document records
*that* CP+MvStack `L = 2` loses to `L = 1` (paired ΔLPD −0.3298 ± 0.0346,
1/20 wins, RMSE 2.01 → 3.41). This document records *why*, and every idea on
the table for fixing it, with the reason to believe each one and its cost.

Date: 2026-07-29, updated same day with Fix 5 results. Model and line
references: `scripts/uci_yacht_cp_mvstack_l2_all_splits.jl` (constants at
`:13-17`, model at `:90-143`, constraints at `:145-166`, predictive at
`:468-524`) and `scripts/uci_yacht_tensor_kernel_benchmark.jl` (priors at
`:261-288`).

---

## Status log

| Date | Step | Outcome |
|---|---|---|
| 2026-07-29 | Fix 5 (predictive repairs), 20 splits, `scripts/uci_yacht_cp_mvstack_l2_fix5_all_splits.jl` | **Confirmed but small on L2**: ΔLPD +0.0334 ± 0.0147 vs legacy L2 predictive, 13/20 wins, RMSE unchanged. Gap to raw L1 remains −0.2964 ± 0.0367 (1/20 wins). Verdict: predictive bugs were a real but minor component (~10% of the gap); the mean damage dominates, exactly as §1.1 predicted. **Fix 5 L1 arm exposed a scale bug, not over-conservatism — see Fix 5b.** Results: `results/cp_mvstack_uci/yacht_l2_fix5_20splits/`. |

Interpretation after Fix 5: the remaining −0.296 is almost entirely the
destroyed mean (RMSE 2.01 → 3.41 unchanged by construction). The critical
path is now Fix 1 + Fix 2 (frozen mean + variance-follows-mean link), then
Fixes 3/4 to un-freeze.

| 2026-07-29 | MvStack notebook revision (damped copy edge, routing flag, exact-variance scoring, frozen `FixedMarginalFormConstraint` prediction, moment-consistent Gamma init) run at L=2..5, both routings, copy α ∈ {0.3, 1.0} | **The upper levels are inert on the 1-D benchmark once inference is honest.** Upper-level SNR ≈ 0.03–0.07 in BOTH routings and at BOTH copy-α values (so the collapse is not a damping artifact); readout coefficients shrink BELOW their skip-weight prior means (z ≈ −0.6…−2.1); held-out logpdf mildly prefers L=2 (−1.161…−1.167) over L=3–5 (−1.17…−1.18). The `d695ddd` "MvStack keeps all layers alive" claim does not survive exact-variance scoring + frozen prediction — the earlier upper-coefficient life was substantially prior means + the second free predictive infer. Predict drift (T vs 2T) ≈ 5.2e-2 with frozen globals; exact-variance logpdf beats plug-in by ~0.01–0.02 as Jensen predicts. |
| 2026-07-29 | Yacht rev script, split-1 smoke: `l2_rev` (damped copy edge) | **Training stabilized**: train MSE ≈ 0.0395 flat across sweeps (fix5 control: 0.0402 → 0.0423 worsening), precision floor 24 vs 14. Damping the copy echo removes the within-training degradation drift, but the sweep-1 mean damage (0.010 → 0.0395) remains — that part is the likelihood-side E[λ] echo, not the copy edge. |
| 2026-07-29 | Yacht rev script, split-1 smoke: `l2_cavity` v1 (damped-from-flat BP Student-t sites toward v) | **DIVERGED** (train MSE 7 → 907 → 7e4): a damped-from-flat rank-one site cannot reach full strength inside a 12-iteration ALS block, so each block writes back a near-prior slab and destroys the warm start. Lesson: **full BP-with-damping for the mean conflicts with the ALS architecture**; the ALS-compatible form of Fix 4 is the gated-design plug-in — the undamped CONJUGATE rank-one site toward v with the weight collapsed to the cavity mean ã/b̃ of m_γ (per-observation echo still cut), with only the γ direction NGMP-projected. Implemented as v2. |
| 2026-07-29 | Yacht rev script, split-1 smoke: `l2_direct` (η = score + intercept, no MvStack) | Trains WORSE than rev (0.068 → 0.104): removing the MvStack readout also removes its tight identity/intercept priors, freeing the log-precision level to drift and amplifying the likelihood echo. The MvStack overhead at L=2 is not the binding problem; its priors are protective. |
| 2026-07-29 | Notebook α-A/B (`MVSTACK_LIKELIHOOD ∈ {stock, cavity0, cavity1}`, dual routing, single-infer, 240 iterations) | **The exact Student-t tails buy nothing at fixed cavity, and α=1 is stable outside ALS.** L=2 held-out logpdf: stock −1.1668, cavity0 −1.1692, cavity1 −1.1702 (spread ~0.003 nats ≪ holdout noise); RMSE/coverage/SNR/shrinkage identical to 2–3 decimals; cavity1 (damped tangent-projected Student-t toward v) runs 240 iterations with no instability — confirming the Yacht divergence was purely the ALS block schedule, not the message. Also: cavity ≈ stock on THIS benchmark — the echo is only destructive when the variance path can absorb mean misfit (Yacht's regime), not when noise is genuinely heteroscedastic and the mean is over-determined. Conclusion: deploying α→0 on Yacht sacrifices nothing relative to exact BP; the α-hierarchy claim is closed empirically end-to-end. |
| 2026-07-29 | **Yacht rev script, FULL 20 SPLITS** (`results/cp_mvstack_uci/yacht_l2_rev_20splits/`) | **`l2_cavity` is the first L=2 model to beat L=1 on Yacht**: paired vs the STRENGTHENED honest L1 (Fix 5b rescale + validation inflation, itself +0.049 better than the raw L1 of the original findings doc): **ΔLPD +0.0731 ± 0.0432 (15/20 wins), ΔRMSE −0.0966 ± 0.0125** (better mean on essentially every split), total predictive variance 2.37 vs L1's raw-scale 2.53. Train MSE ≈ 0.0093 — the cavity likelihood *improves* the mean while learning input-dependent precision (92–109 range, at the anchor). LPD margin is ~1.7 SE — suggestive by the project's own standard, decisive on RMSE (~7.7 SE). Controls: `l2_rev` (copy damping alone) −0.401 vs L1 (1/20) — the copy echo is NOT the binding constraint; `l2_direct` −1.700 (0/20). Versus the ORIGINAL comparison: legacy L2 was −0.3298 ± 0.0346 (1/20); the cavity likelihood moved the paired gap by **+0.40 nats**. |
| 2026-07-29 | **`l2_slim` — the simplified model — 20 splits** (`results/cp_mvstack_uci/yacht_l2_rev2_20splits/`) | Model reduced to `score → damped η edge → Exp → cavity likelihood` (no MvStack, no readout vector, no stack bias, no copy/readout pair; the readout's tight intercept prior transplanted into the score slab's intercept coordinate). **Best RMSE ever on this benchmark: 1.5912 ± 0.0841** (paired vs L1: −0.4149 ± 0.0273, ≈15 SE; vs cavity: −0.3183 ± 0.0244) and best mean LPD −1.9884 ± 0.1007 (slim−L1 +0.1810 ± 0.0840, 14/20; slim−cavity +0.1079 ± 0.0524, 16/20). Mechanism: removing the η-path noise floors lets λ track the genuinely improving residuals (train MSE now IMPROVES across sweeps, 0.0078 → 0.0060 — a virtuous cavity-weight cycle; also an implicit ridge reduction, so L1 was over-regularized). **Caveat: under-calibrated** — coverage 0.826, total variance 0.77, because λ equilibrates to TRAIN residuals (≈0.006) while test residuals are ≈1.8× larger; hence the fat LPD SE. Finishing step: validation-selected variance inflation for the L=2 arm (the same treatment L1 already gets). Two structural lessons: (1) the fully bufferless variant (Exp directly on the score) fits sweep 1 best (0.0072) then oscillates apart across ALS sweeps — exactly ONE damped buffer edge is load-bearing; (2) the copy edge's true role was the proper-message variational boundary, now replaced by flat-fallback guards in the Exp NGMP rules. |
| 2026-07-29 | **Mean-refinement hierarchy** (`scripts/uci_yacht_mean_refinement_hierarchy.jl`, depth-parameterized; L≥3 = gated mean corrections `c_ℓ ~ softdot(f, u_ℓ, λ_ℓ^gate)`, `m = ManyPlus(base, c…)`, NMP likelihood with cavity messages) — split 1 | **First depth ladder that improves the MEAN: L3 −1.636/RMSE 1.346 > L2 −1.811/1.642 > L1 −2.024/1.824**, coverage 1.0/0.935. Train MSE flat while test RMSE drops 18% at L3 — the gate acts as a generalizing input-dependent trust region, not a train-fit booster. Required for stability: correction ridge boost ×50 at gate anchor 1000 (base/corrections are collinear — same features — ALS thrashes otherwise), finiteness guards in the buffer rules (both-sites-flat edge ⇒ marginal mean 0/0), and properness guards on ALL Gamma projection points (6 NMP τ-rules, 2 gated softdot γ-rules). New src rule: `NMP(:μ, Marginalisation)(m_τ, q_out::PointMass)` cavity plug-in. 20-split run (depths 2,3,4) in progress. |
| 2026-07-29 | Frequency-banded variant (`CP_MR_BANDED=true`: correction level ℓ at lengthscale/2^ℓ, gates/base coarse) — split 1 | Structural fix for the base↔correction collinearity (disjoint octaves instead of the ridge boost). Split 1: banded L3 −1.676/1.415 vs unbanded −1.636/1.346 — slightly WORSE; consistent with the data-density ceiling (277 pts, 6-D: an isotropic half-lengthscale band dilutes capacity across 5 smooth directions). 3-split probe running; refinements if banding underperforms: gentler ratio (ls/√2), spectral-decay amplitude priors instead of the ridge boost, ARD-banding (shorten only the Froude coordinate). NOTE: an initial version leaked the test set into the lengthscale heuristic (bases built per partition) — fixed to train-only bases; verified by L2 band-invariance (−1.8113 reproduced exactly). |

---

## 0. Why the goal is achievable at all

Heteroscedastic BBB beats homoscedastic BBB on the same 20 splits by
**+1.04 nats** (−2.6565 vs −3.6925). Input-dependent variance genuinely pays
on Yacht — the residual structure is close to multiplicative in the original
units. The L=2 loss is therefore not "depth is useless here"; it is an
optimization/inference pathology plus predictive-computation bugs. The
headroom is real and large.

A second encouraging fact: L=2's *coverage* is better than L=1's
(0.940 vs 0.892). The variance side of the model works; the LPD loss is
driven by the destroyed mean (RMSE +1.40) and by variance inflation
(mean predictive variance 7.36 vs 2.46).

---

## 1. Diagnosis

### 1.1 The main killer: the mean-field E[λ] echo

At the likelihood `y[i] ~ softdot(f_i, v, λ_i)` with `q(v)` and
`q(precision)` in different clusters:

1. Message toward `λ_i` is the stock mean-field rule
   `Gamma(3/2, ((y_i − f_iᵀm_v)² + f_iᵀV_v f_i)/2)` — a per-point
   inverse-squared-residual estimate, **exact from iteration 1**.
2. Message toward `v` reweights every observation by `E[λ_i]` — a quantity
   that was just fit to the current mean's own residuals. Poorly fit points
   are down-weighted, their residuals grow, their λ falls further.
3. The only counterweight (the damped `Exp(:out)` NGMP site pulling λ toward
   the CP score function) starts from flat η = 0, moves at α = 0.15 per
   iteration, gets 12 iterations, and its damping state is **discarded at
   each of the 18 per-block `infer` calls**. The residual-chasing side is
   instant; the regularizing side never equilibrates.

This is the same pathology named in
`src/nodes/softdot/rules/natural_gradient.jl:37-40` for the κ direction
("rewards κ for explaining its own posterior contraction" — the τ-trap),
running uncorrected in the opposite direction, toward the mean. It is also
the message-passing incarnation of the known heteroscedastic-NLL failure
(Skafte & Hauberg 2019; Seitzer et al., β-NLL, 2022; Stirn et al., faithful
heteroscedastic regression, 2023).

Evidence: `split01/training.csv` shows train MSE already at 0.0402 at the
end of sweep 1 (L=1 reaches 0.010–0.011) and *worsening* across sweeps
(0.0402 → 0.0407 → 0.0421) while the precision range widens (28–38 → 14–43).
The damage happens inside the first 12-iteration block and is never undone.

### 1.2 Structural aggravators

- **No trust region on the mean.** `active_factor_prior` sets the slab prior
  mean to zero every block (only the intercept is anchored); only posterior
  means are carried between blocks, covariance discarded. Nothing resists
  the drift.
- **Prior/likelihood scale shear.** The mean-slab prior precision is scaled
  by `noise_precision_anchor` (≈95), which matches the ridge to L=1 only if
  the fitted λ stays near the anchor. Fitted λ collapses to ~14–43, so the
  effective regularization on the mean is ~3× stronger than what L=1 saw.
- **Readout evidence never accumulates.** The readout prior is re-applied
  fresh in all 18 blocks; its posterior is only used as the next block's
  initialization.

### 1.3 Predictive-computation bugs (cost LPD independently of training)

- **Score uncertainty is hard-coded at test time.**
  `stacked_covariance = Diagonal([1/25 + 1/100, 1e-4])` — a constant 0.05
  for every test input, instead of the input-dependent
  `f(x)ᵀ V_w f(x) + 1/top_carrier + 1/copy`. No off-data growth of
  log-precision uncertainty; also inconsistent with the 0.11 used at
  training initialization.
- **Mean epistemic variance counted at fit, dropped at test.**
  `f_iᵀV_v f_i` inflates the Gamma rate during training (absorbed into the
  aleatoric head) but `f*ᵀV_v f*` is never added to the test predictive
  variance. Epistemic mass is paid for twice at train and credited zero
  times at test.
- **`q(v, y)` is a no-op.** Data interfaces are created `factorized = true`;
  GraphPPL splits `y` out of any declared cluster. Numerically harmless
  (observed y has zero cavity) but the constraints document structure the
  graph does not have. (Same idiom in the base tensor script and the RFF
  script; `UnfactorizedData` is never used.)
- Minor: the Gamma initializer uses `exp(η_mean)` (lognormal median) while
  everything else uses exact lognormal moments; the score CP intercept
  (`N(0,1)`) and the readout intercept (`N(log λ̄, 0.15²)`) are
  non-identifiable duplicates of the same constant.

### 1.4 What is NOT the problem here

- **There is no ContinuousTransition in this model.** The readout is
  `softdot(score_vector, readout_weight, 100.0)`. That is the right choice:
  CT forces mean-field `q(a)`, and the deep-learned-features notebook found
  a CT head "too strongly regularized to move".
- **SNR starvation is not the L=2 bottleneck.** The serial-carrier
  information bottleneck (`why_hierarchy_deep_kernel.jl`) bites at L ≥ 3;
  MvStack already solved it (commit `d695ddd`). At L=2 the score path
  trains fine — coverage improves. Do not conflate the two failure modes
  in the paper.

---

## 2. Fix catalogue

Ranked roughly by (probability of flipping the LPD sign) × (1/cost).

### Fix 1 — Frozen-mean anchored residual (the control; run first)

Fit CP L=1; freeze the mean tensor; learn only the CP precision tensor and
the centred MvStack readout; select the residual scale on the
training-only validation split. RMSE is then pinned at 2.006 by
construction and the LPD comparison isolates whether input-dependent
variance pays. Given the BBB het/homo gap (+1.04 nats), this alone should
flip the sign. Already sketched in the findings doc; still not run.

- Cost: small script change (skip `update_active_factor!` for
  `functions[1]`, or pass the mean as PointMass features into the score
  path only).
- Risk: none — it cannot be worse than L=1 in RMSE; only the variance model
  is on trial.

### Fix 2 — Add the predicted mean to the stack (variance-follows-mean link)

Yacht noise is approximately multiplicative, so give the precision head the
one feature it most needs:

    η(x) = a₁·s(x) + a₂·μ̂(x) + b,   MvStack inputs = [score, mean_copy, bias]

With the frozen mean (Fix 1), `μ̂(x)` is a fixed input — one extra readout
coefficient, same constraints, near-zero identifiability risk. This is the
GLM variance-function idea and it composes with Fix 1 into the arm most
likely to produce a decisive LPD win.

- Cost: one more MvStack input + one prior line.
- Reason to believe: the residual-vs-|y| correlation on Yacht; het-BBB's
  entire win comes from tracking exactly this structure.

### Fix 3 — β-tempered mean update (β-NLL analogue in message passing)

In the mean update only, replace the plug-in weight:

    γ̃_i = λ̄^β · E[λ_i]^(1−β)        (λ̄ = noise_precision_anchor)

β = 1 → "faithful" homoscedastic mean fitting; β = 0 → current model.
Validation-select β. This is *not* equivalent to a Gamma prior on λ:
a prior shrinks q(λ) symmetrically and the echo persists (the model already
has an effective prior — carrier 25, anchored intercept — and the mean
still collapsed in block 1). β-tempering is an asymmetric surgery on one
message direction, the MP analogue of a stop-gradient; not expressible as
any prior. Known from the β-NLL literature to recover MSE-quality means
where variance regularization does not.

- Cost: a small rule adapter on the softdot `:x`/`:θ` messages (scale the
  consumed `mean(q_γ)`), plus a validation loop over β ∈ {0, 0.25, 0.5,
  0.75, 1}.
- This is the lever that lets the mean be *unfrozen* safely after Fix 1
  establishes the ceiling.

### Fix 4 — BP cavity (Student-t) message toward v: the paper-machinery fix

The principled version of Fix 3. The exact BP message from the likelihood
toward `v` integrates λ under its **cavity** (the Exp-side Gamma, not the
residual-contaminated marginal):

    ∫ N(y | fᵀv, λ⁻¹) Gamma_cav(λ) dλ  =  Student-t factor in fᵀv

Non-conjugate → supply it as an NG projection. This is literally the
paper's thesis (the missing exact-BP message for a non-conjugate factor IS
a natural-gradient projection) applied to the heteroscedastic likelihood.
The rule slot exists (`GaussianStudentTMessage` in `src/expressions/`
currently errors "no closed form"); the multivariate v-edge has the
degree-5 cubature; the univariate pieces have quadrature/UT.

- Reason to believe: two in-repo precedents where cavity-based beats
  marginal-based on precision-coupled edges — the gated-κ cavity extraction
  (`softdot/rules/natural_gradient.jl:47-90`) and the ETTh dynamic arm
  (native NGMP/BP) beating VMP Dynamic 6/8.
- Cost: one new tangent-projection rule + tests. Highest effort, highest
  paper value: turns the failure into a result ("VMP marginal reweighting
  destroys the mean; the exact BP message does not").

### Fix 5 — Repair the predictive variance (do this regardless)

1. Input-dependent log-precision uncertainty at test:
   `s_η²(x) = a₁²·(f(x)ᵀV_w f(x) + 1/top_carrier + 1/copy) + (V_a terms) + 1/readout_precision`,
   replacing the hard-coded `Diagonal([0.05, 1e-4])`.
2. Add the mean's epistemic term `f*ᵀV_v f*` to the predictive variance
   (requires keeping `V_v` for the final state instead of discarding it at
   `:371-380`).
3. Make the Gamma initializer use `exp(m + s²/2)` moments consistently.
4. Fairness note: if L=2 gets an epistemic term, the honest L=1 baseline is
   the validation-inflated one (L=1's constant train-MSE variance gives
   coverage 0.892 — itself overconfident).

- Cost: prediction-path code only; no retraining changes.
- Effect: sharpens LPD on both tails; will not fix RMSE by itself.
- **Status: RUN (2026-07-29).** L2: +0.0334 ± 0.0147 over legacy predictive,
  13/20 wins, mean epistemic added 1.259. As expected, does not close the
  gap to L1 (−0.2964 remains). Keep these repairs in every later arm.

### Fix 5b — Rescale the retained L=1 slab covariance (new, from Fix 5 run)

The Fix 5 L1 arm (epistemic added to L=1) produced mean epistemic 21.10,
total variance 23.56, coverage 1.000, and LPD −2.5196 — *worse* than raw
L1. This is **not** evidence that "a single final ALS conditional
covariance is too conservative". It is a scale bug:
`fit_fix5_l1` fits every block with `observation_precision = 1.0`
(`uci_yacht_cp_mvstack_l2_fix5_all_splits.jl:161`, inherited from
`update_depth_one!`). In the conjugate solve both the likelihood and the
prior scale with γ, so the posterior covariance obeys
`V(γ) = γ⁻¹ (XᵀX + reg·Gram)⁻¹` — the retained matrix is the γ = 1
covariance, a factor `1/σ̂² ≈ 95` too large. The point estimate is
unaffected (γ cancels in the mean), which is why the bug was invisible
until Fix 5 started using the covariance.

Correction: multiply the retained `V_v` by `homoscedastic_variance`
(equivalently, compute epistemic as `σ̂² · f*ᵀ(XᵀX + reg·Gram)⁻¹ f*`).
Expected magnitude after rescaling: ≈ 0.22 original-units mean epistemic —
small and sane next to the 2.46 aleatoric.

Two follow-ons:

1. Re-run the L1 validation inflation *after* rescaling. It selected 1.0 on
   every split only because the epistemic term was 95× inflated; with the
   corrected (small) epistemic, L1 coverage returns to ≈ 0.89 and the
   selector should pick a genuine inflation > 1, raising the honest L1
   baseline.
2. After rescaling, the conditional ALS covariance is if anything too
   *narrow* (it conditions on the other five CP factors at point
   estimates, missing cross-factor uncertainty). If a calibrated global CP
   posterior is ever needed, that is the direction of the remaining error
   — the opposite of what the buggy run suggested.

Note the L2 arms do not share this bug: their mean-slab blocks run at the
actual `E[λ]` (≈14–43) with the anchor-scaled prior, so the retained
covariance there is approximately correctly scaled (its 1.259 contribution
is plausible).

### Fix 6 — Trust region / proximal ALS on the mean slab

Set the slab prior's weighted mean to the precision-weighted previous slab
value (`weighted_mean = precision * vec(previous_slab)`) instead of zero in
`active_factor_prior`'s caller. Each block then solves a proximal step
around the current tensor instead of a fresh zero-mean ridge. Optionally a
separate trust-radius hyperparameter for the mean vs the score tensors.

- Cost: ~one line + one hyperparameter.
- This is the findings doc's "strong trust-region prior around its L=1
  factors", made concrete.

### Fix 7 — Scheduling and damping-state repairs

- **Mean warmup**: sweep 1 with λ frozen at the anchor (homoscedastic),
  then release. Standard remedy in the het-regression literature; also the
  deep-learned-features notebook's own recommendation ("shallow pretraining
  followed by an identity-initialized second block").
- **Persist NGMP damping state across the 18 blocks** (or run one global
  `infer` for the precision path between CP coordinate sweeps) so the
  `Exp(:out)` site can actually reach its fixed point. Today it restarts
  from flat η = 0 every block.
- **More iterations where they matter**: `block_iterations = 12` with
  α = 0.15 gives the regularizing site no chance against an exact
  residual message. Either raise iterations or raise α on the Exp edges
  specifically (separate `DampingMeta` per edge is already supported).
- **Free-energy monitoring**: `free_energy = false` currently; the base
  script's `update_deep!` checks it. Turn it on for diagnosis runs so
  divergent blocks are visible.

### Fix 8 — Weaken or re-place the copy-edge pseudo-observation

`readout_score ~ N(score, 1/100)` is mean-field-split from `q(w, score)`,
so its return message is a precision-100 pseudo-observation pinning `score`
to the readout's current belief — the "carrier over-informs when
mean-field-split" pathology from `wip/bnn_uncertainty_diagnosis_report.md`.
Try `COPY_PRECISION ∈ {10, 25}` (validation-select), or restructure so the
variational boundary sits at a lower-precision edge. Keep the finite copy
edge itself — it exists to give MvStack proper inputs.

### Fix 9 — Prior-side control (cheap, run as ablation, not as the fix)

Raise `top_carrier` and the score-tensor regularization so λ(x) is forced
smooth and near-anchor. Prediction: reduces mean damage but does not
eliminate it (the echo is structural) and trades away exactly the
input-dependent variance needed to win LPD. Useful as the negative control
that justifies Fixes 3/4 in the paper.

### Fix 10 — Hyperparameter selection that currently does not exist

The protocol carries `validation_fraction = 0.1` and **nothing uses it**.
Candidates to select on validation, roughly in order of expected impact:

| Knob | Current | Note |
|---|---|---|
| β (Fix 3) | — (implicitly 0) | the big one |
| residual/readout scale (Fix 1) | — | required by the anchored protocol |
| `top_carrier` | 25.0 fixed | score smoothness |
| score `cp_regularization` | 2e-3 (shared with mean) | decouple from the mean's |
| `COPY_PRECISION` | 100.0 | Fix 8 |
| L=1 variance inflation | — | honest baseline |
| `sweeps` / `block_iterations` | 3 / 12 | with persisted damping |

---

## 3. Ideas considered and rejected (with reasons)

- **Joint/structured `q(v, precision)`.** The dyn_exp finding: structured
  joints lock the precision at its prior (any τ is a fixed point). That is
  the opposite trap. The asymmetric fixes (3/4) are strictly better targeted.
- **Merging the readout clusters** (e.g. `q(w, score)` with
  `q(readout_weight, eta, precision)`). No softdot rules exist for
  `q(θ, x)` joints; and the notebook record shows dense multi-path readouts
  produce non-positive-definite Gaussian cavities.
- **More precision-side routes from upper layers to y.** At L=2 the score
  already reaches y through MvStack → η → Exp → λ, and the precision path
  demonstrably trains (coverage improves). Connectivity is an L ≥ 3 (SNR)
  issue, already solved by MvStack.
- **Upper layers into the mean of y** (`y ~ N(fᵀv + c·s(x), λ⁻¹)`).
  Tempting (mean paths get Fisher information λ·(∂μ)² per point vs the flat
  1/2 for log-precision paths), but it changes the question — that arm is a
  wider mean model, fairly compared against L=1 at higher rank, not against
  this L=2 — and sharing s(x) between mean and η re-imports the joint-training
  damage. If tried anyway: centred residual `(s − anchor)/σ_prior` with a
  small validation-selected coefficient, per the skip-notebook design rules.
- **Gamma prior on λ as the fix.** See Fix 9 — symmetric shrinkage cannot
  remove an asymmetric feedback; the model's existing prior side already
  failed to prevent block-1 collapse.

---

## 4. Suggested experiment order

1. ~~**Fix 5** (predictive repairs)~~ — **DONE.** +0.0334 on L2; new
   baseline for all later arms is Fix 5 L2 = −2.5149 ± 0.0535. Apply
   **Fix 5b** (rescale L1 slab covariance by σ̂², re-run the validation
   inflation) to finalize the honest L1 baseline before the depth
   comparison is scored again.
2. **Fix 1 + Fix 2** (frozen mean + μ̂ in the stack, residual scale on
   validation) — the arm expected to produce the decisive LPD win with
   RMSE pinned by construction. **Now the critical path**: Fix 5 confirmed
   the remaining −0.296 gap is the mean, and the frozen-mean arm removes
   it by construction while keeping the coverage advantage L2 already has
   (0.940 vs 0.892).
3. **Fix 6 + Fix 7** (trust region + warmup + persisted damping) — un-freeze
   the mean under protection; check whether joint training can now hold the
   L=1 mean.
4. **Fix 3** (β-temper, validation-selected) — if step 3 still bleeds RMSE.
5. **Fix 4** (Student-t cavity message) — the paper-grade replacement for
   step 4; worth doing even if step 4 suffices, because it is the
   NGMP-native statement of the same repair.
6. **Fix 9** as the published negative control; **Fix 8/10** folded into
   whichever arm survives.

Success criterion throughout: paired per-split ΔLPD (L=2 − L=1) > 0 with
the 20-split SE, RMSE non-inferior, on the locked `repeated-holdout-v1`
splits; validation split used for every selected knob, test touched once.
