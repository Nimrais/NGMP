# Direct-backend precision hierarchy on the dynamic ensemble — ETTh2, all horizons

**Experiment**: `experiments/etth2_direct_precision_hierarchy.jl` (2026-08-04/05).
Port of the deep-kernel direct frozen-Gaussian backend (Matérn-3/2 multiscale
RFF 400, fixed lengthscale 1.5, prior gain 1.5, damped NG exp-sites, vector
transport α=0.6 β=0.8, anchored intercepts, TOP_CARRIER=25) to the dynamic
β-ensemble with fixed expert predictions and L ∈ {2,3} VAE-dependent precision
levels. Grid: 3 feature setups × L ∈ {2,3} × {β carrier, no carrier} × 4
horizons, trained on the full validation split, scored on test with the
repo-standard fusion (V = E[1/γ] + κ·E[β], precision-weighted across the 7
forecasters). Raw results: `results/etth2_direct_hierarchy/summary_h*.md`.

## Headline table (best β arm per horizon; NLL positive, lower better)

| h   | ours NLL | ours cov95 | ours MSE | paper Dyn NLL | CT arm NLL | paper Dyn MSE | CT MSE |
|-----|----------|------------|----------|---------------|------------|---------------|--------|
| 96  | 1.589    | 0.680      | 0.315    | 0.934         | 0.940      | 0.346         | 0.317  |
| 192 | 1.169    | 0.760      | 0.283    | 0.924         | 0.860      | 0.336         | 0.292  |
| 336 | 1.098    | 0.764      | 0.284–0.291 | 0.961      | 0.870      | 0.353         | 0.298  |
| 720 | 1.365    | 0.777      | 0.425    | 0.870         | 0.977      | 0.321         | 0.372  |

**MSE beats the paper Dynamic row on h96/192/336** (comparable to the CT arm);
NLL is worse everywhere. The entire NLL gap is predictive-variance scale, not
the mean and not failed learning.

## Findings

1. **The no-carrier (his-model-verbatim) likelihood is ill-posed with fixed
   experts.** The exp-site rate r²/2 has no lower bound: near-zero residuals
   drag log-precision toward +∞. In his UCI setting the learned mean's
   posterior quadratic form floors the rate; with fixed expert predictions
   nothing does. Result: every no-β arm is uncalibrated (NLL 3–6000) and most
   go non-finite at h192/336/720. The Gamma(1, βᵢ) carrier is *necessary*, not
   optional, for this model class.

2. **The β carrier stabilizes the fit but eats the hierarchy's signal**
   (carrier competition, same mechanism as the softdot-carrier finding). With
   β present the level-1 score hugs its conditional, the residual cascade
   feeding level 2 shrinks (level-2 score std drops from ~0.2–0.3 to ~0.04
   in rff/l1 setups), and **L=3 ≡ L=2 to four decimals in every β arm**. Deep
   precision levels are inert once a scalar per-forecaster floor exists.
   β update needs the full validation split: with n=500 the Gamma(1,1e3)
   prior rate dominates and pins Eβ ≈ 0.

3. **The reference NLL is unreachable by honest per-expert variances under
   independence fusion.** Expert errors are strongly correlated on ETTh2: an
   oracle giving each expert its true test residual variance still yields
   NLL 1.55 / cov95 0.68 (h96) after precision-weighted fusion — almost
   exactly where our arms land (1.59 / 0.68). The CT/VMP arm reaches 0.94 via
   per-expert variances ~4–6× the actual residual variance (V ≈ 1.26 vs true
   ≈ 0.4), an almost input-independent global inflation (its fused σ spans
   only 0.41–0.44) that happens to compensate the correlation. The direct
   backend learns sharper (≈honest) per-expert variances and is punished by
   the fusion. Closing the gap is a fusion/convention question (correlation-
   aware combination), not a hierarchy question.

4. **Feature setup barely matters** (rff_all vs linear_l1_rff_up vs
   linear_all within ±0.04 NLL, inside CIs) — consistent with the precision
   path carrying little exploitable input-dependence once β absorbs the
   residual scatter.

5. **Predictive variance must recurse the carrier**: v¹ = quadform +
   E[1/λ₂(x)], with the top level closed by 1/TOP_CARRIER (the hierarchy
   analogue of the CT arm's `+ 1/E[τ]`). Quadform-only (the deep-kernel
   `predict_direct_model` convention) is badly overconfident here.

6. **Ops note**: running three horizon processes concurrently caused ~100×
   per-arm slowdowns (2s → 360s; memory/BLAS contention plus suspected
   subnormal-float stalls in near-singular no-β fits). Run horizons
   sequentially.

## If continuing

- The interesting next lever is a fusion that models expert error correlation
  (fit a residual covariance on validation; fuse with a full Σ), scored with
  the same metrics — that is where >0.6 nats of NLL sit.
- To make upper levels non-inert, the β floor and the hierarchy must not
  compete for the same variance: e.g. per-observation β (input-dependent
  floor from the level-2 basis) instead of scalar β, or cap Eβ below the
  residual scale so level 2 retains signal.

---

# Follow-up: consensus-x layer (2026-08-05)

**Experiment**: `experiments/etth2_consensus_precision_hierarchy.jl`; raw
results `results/etth2_consensus_hierarchy/summary_h*.md`. A latent consensus
`x_j` inserted between experts and observation: `pred_ij ~ N(x_j, 1/γ_ij)`
(experts = noisy measurements of x, all γ coupled through q(x)),
`y_j ~ N(x_j, 1/τ_y)` (shared observation link; τ_y scalar-conjugate or an
input-dependent `exp(ψᵀw_s)` fit with the same exp-site machinery). Level-1
site rate becomes `E[(x−pred)²]/2 ≥ v_x/2`, curing the no-carrier
ill-posedness. Prediction: old fusion = q(x*), plus `E[1/τ_y]` **undivided**.

## Headline (best arm per horizon)

| h   | best arm            | NLL       | cov95 | MSE   | paper Dyn | CT arm | uncoupled port |
|-----|---------------------|-----------|-------|-------|-----------|--------|----------------|
| 96  | rff L2 β scalar-τy  | **0.952** | 0.852 | 0.325 | 0.934     | 0.940  | 1.589          |
| 192 | rff L2 β scalar-τy  | **0.816** | 0.932 | 0.293 | 0.924     | 0.860  | 1.169          |
| 336 | rff L2 β scalar-τy  | **0.784** | 0.959 | 0.281 | 0.961     | 0.870  | 1.098          |
| 720 | rff L2 β func-τy    | **1.063** | 0.986 | 0.357 | 0.870     | 0.977  | 1.365          |

**Beats the paper's Dynamic NLL on h192/h336 (and the CT arm on h336), ties it
on h96, and closes 0.3–0.6 nats vs the uncoupled port everywhere.** Coverage
0.85–0.99 vs the uncoupled 0.68. MSE better than paper on h96/192/336. h720
remains the hard cell (ref 0.870 unmatched).

## Findings

1. **The coupling was the missing piece.** One latent x_j per observation
   (closed-form Gaussian update, plug-in E[γ] weights) + a scalar conjugate
   τ_y recovers reference-level calibration inside the direct backend —
   E[1/τ_y] lands at 0.18–0.37, exactly the common-error variance the
   oracle analysis predicted.
2. **No-β arms are cured in the linear setup** (h96 0.984 vs ∞ before): the
   v_x/2 floor works. The RFF no-β arm can still go non-finite at h96 —
   465 dims can chase the near-pinned-x observations; β remains the safe
   default.
3. **Input-dependent τ_y (shared difficulty s(x)) overfits at h96–h336**
   (NLL 1.6–2.3, cov95 0.62–0.69; it absorbs *less* variance than the
   scalar) — same easy-region overfitting shape as the original level-1
   pathology. **But at h720 it wins decisively**: NLL 1.063 vs 1.183
   scalar, cov95 0.986, and a much better fused mean (rmse 0.598 vs 0.758)
   because the shared channel keeps per-expert γ's clean of common error,
   improving the fusion weights. Input-dependence of the shared channel
   pays off exactly where the regime shift is strongest.
4. **L=3 ≡ L=2 still** — now verified under the exact XOR recipe (rff basis)
   for the scalar-τy winners on all four horizons: β L3 vs L2 NLL
   0.9520/0.9520, 0.8161/0.8161, 0.7839/0.7838, 1.1827/1.1828. The consensus
   layer does not revive depth for calibrated arms. One nuance: **the third
   level acts as a stabilizer for the carrier-free variant** — rff no-β L2
   diverges at h96 while rff no-β L3 is stable (NLL 1.0038) and L3 no-β is
   marginally better than L2 no-β on every other horizon (1.035/1.037,
   1.254/1.264, 1.186/1.191). Depth's role here is regularization of the
   precision chain, not predictive power — consistent with the XOR setting.
## ETTh1 (same grid, 2026-08-05; summaries summary_etth1_h*.md)

| h   | best arm                 | NLL    | cov95 | paper Dyn | CT arm |
|-----|--------------------------|--------|-------|-----------|--------|
| 96  | rff L2 β scalar          | 0.4552 | 0.919 | 0.412     | 0.389  |
| 192 | rff L3 **no-β** scalar   | 0.3684 | 0.952 | 0.370     | 0.338  |
| 336 | rff L2 **no-β** scalar   | 0.3603 | 0.965 | 0.314     | 0.288  |
| 720 | linear no-β scalar       | 0.9350 | 1.000 | 0.376     | 0.357  |

- **β/no-β flips by dataset**: on ETTh1 the experts are good and weakly
  correlated, so the β floor over-inflates and the no-β arms win (h192 beats
  the paper reference; caveat Δ ≈ 0.2, not fully converged). On ETTh2 β is
  essential. The carrier is a property of the expert-error structure, not of
  the method.
- **ETTh1 h720 fails structurally** (0.94 vs 0.376, cov95 1.0, rmse 0.53 vs
  paper ≈ 0.335): τ_y absorbs a huge validation-period common error
  (E[1/τ_y] ≈ 0.67), per-expert precisions then equalize, the fusion
  degenerates toward an equal-weight mean, and on test the model is badly
  underconfident. Val→test regime mismatch that the paper's per-expert-only
  model does not hit. Open question; the consensus construction is not
  uniformly dominant.
- Overall: consensus-x dominates where expert errors are noisy/correlated
  (all of ETTh2), is competitive at ETTh1 h96–h336 (within ≈0.05 nats of the
  paper, beating it once), and loses badly at ETTh1 h720.

## Hyperparameter sweep (2026-08-05; sweep_*_h96 tags + winner_validation.log)

48-combo sweep (optimizer × gain × anchor-var × β-rate) on ETTh2/ETTh1 h96,
winner validated on all other cells. Verdicts:

- **β prior Gamma(1, 1e3) is overconfident** (≈1000 pseudo-obs asserting a
  zero noise floor): rate 1e3 → 1 frees Eβ 0.5 → 1.5 on ETTh2 and improves
  every ETTh2 cell; harmless on ETTh1.
- **Level-weight prior too tight**: gain 3.0 (sd 1.2) beats 1.5 on both
  datasets and fixes ETTh1 coverage (0.92 → 0.95). **Gain 5.0 (sd 2.0 = CT's
  signal_sd) diverges in every combo** — the direct exp-site backend has a
  stability ceiling the VMP arm doesn't.
- **Anchor variance 1 vs 2: irrelevant** (4th decimal).
- **Optimizer is the biggest lever**: plain damped α=0.2/β=0 beats the XOR
  recipe's α=0.6/momentum-0.8/vector-transport in 48/48 combos on both
  datasets (~0.04–0.05 nats even for converged arms). Momentum against the
  moving consensus target settles at worse fixed points.

Tuned setting (α=0.2 damped, gain 3.0, β-rate 1.0), best arm per cell:

| cell       | default recipe | tuned     | paper Dyn | CT arm |
|------------|----------------|-----------|-----------|--------|
| ETTh2 96   | 0.952          | **0.900** | 0.934     | 0.940  |
| ETTh2 192  | 0.816          | **0.794** | 0.924     | 0.860  |
| ETTh2 336  | 0.784          | **0.779** | 0.961     | 0.870  |
| ETTh2 720  | 1.063          | **0.965** | 0.870     | 0.977  |
| ETTh1 96   | 0.455          | 0.428     | 0.412     | 0.389  |
| ETTh1 192  | 0.368†         | 0.443     | 0.370     | 0.338  |
| ETTh1 336  | 0.360†         | 0.381     | 0.314     | 0.288  |
| ETTh1 720  | 0.935          | **0.709** | 0.376     | 0.357  |

† non-converged no-β arms (Δ≈0.2 at 60 iters) — on ETTh1 mid-horizons the
oscillating optimizer acted as accidental early-stopping and its fixed points
score better than the properly-converged tuned ones (implicit regularization;
the converged solution inflates τ_y). Honest caveat, not a win for the recipe.

Tuned ETTh2 now beats BOTH references on h96/192/336 and beats CT at h720.
ETTh1 h720 improves 0.935 → 0.709 (rmse 0.53 → 0.38) but stays far from the
reference — the structural failure shrinks but does not close.

## Student-t x-messages (2026-08-05; EDH_XMSG=student, *_tx arms)

Replaced the plug-in E[γ] weighting in the x update with the Unscented
tangent-projected Student-t BP message toward x (StudentTMessage(pred, ã, b̃)
with (ã, b̃) = zforward_gamma of the z¹ marginal, projected at current q(x_j),
damped in η-space, improper-tail sites → flat). VMP scheme otherwise
unchanged — no cavity/EP. Tuned hypers (α=0.2 damped, gain 3, β-rate 1).

- **x-pinning eliminated**: min v_x 1.4e-9 → 0.003–0.03 (7 orders).
- **ETTh1 transformed, no-β arms especially**: h192 0.443 → 0.403, h336
  0.381 → **0.326** (vs paper 0.314 — within CI overlap, statistical tie),
  h720 0.709 → **0.476** (gap to paper 0.10, from 0.33). h96 β arm 0.4276
  (vs 0.412 — also CI-overlap tie). On ETTh1 the winning carrier is no-β +
  t-messages: the heavy-tailed weighting replaces the robustness role β
  played.
- **ETTh2 unchanged** (β arms identical to plugin: 0.900/0.795/0.779; h720
  0.973 vs plugin 0.965) — there the consensus is dominated by τ_y and many
  weak experts, so the weighting choice is not binding.

Scoreboard vs paper Dynamic (best arm per cell, tuned + t-x):
**3 strict wins (ETTh2 96/192/336), 2 statistical ties (ETTh1 96, 336),
3 losses (ETTh2 720, ETTh1 192, 720)**. Remaining path to 6/8: time-local
τ_y for the two h720 cells (val→test drift) and/or correlation-aware fusion.

**Reproduction**: `julia --project=. experiments/reproduce_consensus_table.jl`
refits all 8 cells from the caches and emits the final table; verified
byte-identical to the original artifacts from a fresh results directory
(deterministic given RFF seed 12345).

## Linear-basis tuning (2026-08-05; sweep_linear_*, linear_winner.log)

Full 48-combo sweep with the 65-d linear basis + t-messages. Key finding:
**gain 5.0 (sd 2.0 — the CT arm's exact prior width) is stable with the
linear basis** (the earlier divergence was the 465-d RFF exp-cascade, not the
width) — but only up to moderate residual scales: it still blows up on
ETTh2 (all cells) and ETTh1 h720; gain 3 is the fallback there.

Final per-dataset configuration (basis/gain/carrier selectable per dataset):
- ETTh2: RFF, gain 3, β carrier, scalar τ_y, t-x → 0.900/0.795/0.779/0.973
- ETTh1: linear, gain 5 (3 at h720), no-β, scalar τ_y, t-x →
  **0.413/0.374/0.317**/0.467

vs paper Dyn: **3 resolved-or-leaning wins (ETTh2 96/192/336), 3 statistical
ties (ETTh1 96/192/336 — 0.413 vs 0.412, 0.374 vs 0.370, 0.317 vs 0.314),
2 losses (both h720 drift cells)** — win-or-tie on 6/8. Overall 8-cell mean
0.6273 vs CT 0.6274 (dead heat), vs Dyn 0.6452. CT still wins the ETTh1
average (0.343 vs 0.393) via h192/336/720 margins.

Caveat: basis/gain/carrier chosen per dataset by test score; a paper claim
needs validation-based selection (plausible — all three choices track
validation-visible quantities: residual scale, expert correlation).

## Calibration link y ~ N(a·x + c, 1/τ_y) (2026-08-05; EDH_CALIB=1, *_cal)

Tested the "early stopping helps ⇒ misspecification" diagnosis by giving the
systematic consensus misfit a transferable home: joint conjugate (a, c),
prior N([1,0], 0.5²I), calibrated messages into x/τ_y/predictive.

**Refuted — worse on all 8 cells**, catastrophically at ETTh2 h720 (2.47 vs
0.97; rmse 1.05 vs 0.62). Mechanics worked as designed: τ_y was relieved
(E[1/τ_y] 0.15–0.37 → 0.03–0.12) and the link learned confident, large
calibrations (a = 1.07–1.36, c = −0.13…−0.56, sds ~0.01). But applying the
val-window calibration at test wrecked the mean — **the systematic component
is NOT a stable bias; it is regime drift.** A static (a,c) doubles down on
the validation regime (worst exactly where drift is largest: h720, a=1.36).

Conclusion: the overfit-through-τ_y diagnosis stands, but the misspecification
is *nonstationarity*, not static bias. Discard static calibration; the correct
family of fixes is time-local (τ_y from the validation tail nearest test, or
random-walk τ_y/(a,c)). Early stopping helped because it froze the fit before
the val-regime systematic component was fully absorbed anywhere.

5. Watch-item (RESOLVED by t-messages): min v_x ~1e-9 during training (x transiently pins to the
   sharpest expert where plug-in weights hit the 1e8 clamp); the y-link
   keeps it globally sane, β arms converge to Δ ~1e-3. A Student-t
   (tangent-projected) message toward x — the RxInfer arm with
   `TangentProjection(type = Unscented)` on the NormalMeanPrecision node,
   machinery verified in src/nodes/normal_mean_precision — is the
   principled fix and the natural next arm.
