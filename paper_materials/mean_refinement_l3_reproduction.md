# Why banded L=3 beats L=2, and how to reproduce it

Status 2026-07-30: **CONFIRMED over all 20 splits**
(`results/cp_mvstack_uci/yacht_mean_refine_banded_20splits/`):

| Paired, 20 splits | ΔLPD | wins |
|---|---:|---:|
| banded L3 − L2 | **+0.0937 ± 0.0219** (~4.3 SE) | 17/20 |
| banded L3 − L1 | **+0.2565 ± 0.0518** (~5 SE) | 18/20 |
| unbanded L3 − L2 (control) | −0.0185 ± 0.0355 | 11/20 |

Absolute ladder: L1 −2.1694 → L2 −2.0067 → L3 **−1.9129**, coverage 0.937 at
every depth — the best calibrated LPD recorded on this benchmark. The band is
the mechanism: same depth/gates/inference without the octave separation gains
nothing. Nuance: at 20 splits the L3 gain is a DENSITY gain (predictive
variance 2.41 vs 2.55 at identical coverage), not an RMSE gain (ΔRMSE
+0.03 ± 0.10 neutral; the splits-1–3 RMSE win regressed to noise).

## The mechanism, in four parts

1. **The third level enters through the mean, not the precision.** A
   log-precision layer receives at most ½ nat of Fisher information per
   observation; a mean correction receives first-order information
   λ·(∂μ)². This is why every earlier L=3 (precision-of-precision) was
   inert and this one is not:

       gate_score₃ ~ softdot(f_coarse, g, top_carrier)
       λ^gate(x)  = exp(buffered gate_score₃)         # the trust field
       c(x)       ~ softdot(f_fine, u, λ^gate(x))     # gated mean correction
       y          ~ N(base(x) + c(x), λ_noise(x)⁻¹)

   "Level 3 predicts the variance of level 2's refinement" — the upper
   level still speaks the hierarchy's language (a precision), but the
   thing it gates now carries first-order information about y.

2. **Corrections are OFF by default, so depth cannot hurt.** u is
   zero-anchored and the gate is anchored at high precision (κ = 1000):
   at the anchor the model reduces *exactly* to L=2. The gate opens only
   where residual evidence is consistent across observations. Empirically
   the correction acts as a generalizing trust region: train MSE stays
   ≈ flat while test RMSE drops — the opposite of overfit.

3. **The frequency band removes the collinearity that kills same-band
   depth.** The correction's basis lives at `lengthscale/2` (one octave
   above the base). Base and correction then span near-disjoint function
   spaces: they cannot compete for the same signal (the unbanded control
   shows what happens when they can — no 20-split gain), and the octave
   contains exactly the frequencies the coarse basis cannot represent
   (Yacht's sharp Froude-number response; the M12/R16 probe showed more
   rank at the SAME lengthscale buys ~1%). Gates stay at the coarse scale
   (trust fields should be smooth).

4. **Echo-free inference throughout** (the session's cavity discipline):
   the likelihood exchanges cavity messages in both directions (conjugate
   ã/b̃ plug-in toward the mean, projected NormalPrecisionMessage toward
   λ), the correction softdot uses the gated-cluster rules (a correction
   cannot inflate its own gate), each Exp chain has exactly one damped
   buffer edge, and all Gamma projection points carry flat-fallback
   impropriety guards. Without these, the same graph either destroys the
   mean (marginal-message echo) or NaNs (improper transients).

## Reproduce

```bash
cd ~/repos/surrogate-modelling
# 3-split result (validated):
CP_MR_SPLITS=3 CP_MR_BANDED=true \
  CP_MR_OUTPUT_DIR=results/cp_mvstack_uci/yacht_mean_refine_banded_3splits \
  OPENBLAS_NUM_THREADS=1 julia --project=. scripts/uci_yacht_mean_refinement_hierarchy.jl
# 20-split confirmation:
CP_MR_SPLITS=20 CP_MR_BANDED=true \
  CP_MR_OUTPUT_DIR=results/cp_mvstack_uci/yacht_mean_refine_banded_20splits \
  OPENBLAS_NUM_THREADS=1 julia --project=. scripts/uci_yacht_mean_refinement_hierarchy.jl
```

Expected (splits 1–3): l1 −2.159 / RMSE 2.023; mr_l2 −1.959 / 1.913;
mr_l3 −1.801 / 1.753, coverage 0.957–0.968; paired table in
`<output_dir>/table.md`.

Defaults that matter (all ENV-overridable): depths `2,3`; `CP_MR_BANDED`
must be `true` (default false = the failing same-band control);
`CP_MR_GATE_ANCHOR=1000`, `CP_MR_CORRECTION_RIDGE_BOOST=50` (collinearity
suppression — still needed within a level), `CP_MR_BUFFER_PRECISION=100`,
`CP_MR_CORRECTION_FLOOR=0` (the 4^ℓ spectral floor at 25 crushed the
useful octave — measured, reverted).

## Known limits (measured, not speculative)

- **L=4 diverges in all three tested variants, and the trio isolates the
  cause by elimination**: same-band (no separation; −1146), dyadic /2,/4
  (separated but the /4 octave crosses the isotropic Nyquist wall; −103),
  √2-ladder /1.41,/2 (both bands resolvable but only half-octave apart —
  inter-correction spectral overlap recreates the degeneracy; −477).
  Two corrections need band separation r ≥ 2 AND all bands under the wall;
  with ~1.5–2 resolvable isotropic octaves on Yacht those constraints have
  an empty intersection — **L=3 is the isotropic maximum on this dataset**.
  L≥4 requires ARD-banding (Froude's ~14 sample levels afford 2–3 separated
  octaves along one axis) or stagewise freezing of overlapping bands.
  Side result: √2-L3 works (+0.123 ± 0.023 over L2) but dyadic is better
  (+0.158); `CP_MR_BAND_RATIO` defaults to 2.
- **No automatic Occam**: an unresolvable level does not stay inert — the
  gate pins the correction to fᵀu, not to zero, and the aliased directions
  receive real (spurious) likelihood pull, so u walks off its anchor. Even
  a level-selective spectral floor (first correction free, 400 on the
  second) only softens the failure (per-split L4: −2.40 / −4.20 / −28.2 vs
  L3's −1.80 ± 0.07): soft priors make the walk expensive, not impossible.
  Depth beyond the resolvable spectrum must be excluded by a HARD
  mechanism — validation-selected depth (rejects L4 on every split),
  stagewise freezing, or ARD-banding (which makes the level resolvable
  instead of suppressed).
- **The per-level κ-ARD gate is ANTI-Occam here (measured)**: with
  `u_ℓ ~ MvN(0, κ_ℓ⁻¹I)`, `κ_ℓ ~ Gamma` (the zero-anchored gated-weights
  device, `CP_MR_ARD=true`), the conjugate update
  `κ ← Gamma(a₀+d/2, b₀+E‖u‖²/2)` reads a LOUD slab as evidence for LOW
  prior precision — gate opens, slab grows, positive feedback: κ → 0,
  train MSE 1.8e6, PosDefException within one sweep. ARD prunes quiet
  redundancy; it cannot clamp aliased components because they genuinely
  reduce training error. Generalizes the conclusion: NO amplitude-inferred
  prior (floor, EB ridge, κ-gate) can supply automatic Occam for
  unresolvable levels — within-sample amplitude cannot distinguish
  fit-to-signal from fit-to-noise. Flag kept (default false) as the
  documented negative control.
- Requires the session's src rules: cavity likelihood plug-ins
  (`NMP(:μ)(m_τ, q_out)`, softdot gated/cavity set), constant-τ damped
  buffer rules with finiteness guards, Exp/Gamma impropriety guards.
  `julia --project=. -e 'using Pkg; Pkg.test()'` must be green first.
