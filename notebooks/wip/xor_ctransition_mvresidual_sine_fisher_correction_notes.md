# Receiving-marginal Fisher correction for the residual-sine forward message

Date investigated: 2026-07-21

Status: historical experiment. The implementation described below was removed
after evaluation. The production experiment remains limited to the original
`delta` and `moments` forward modes.

## Question

Can the analytic moment-matched forward message through `MvResidualSine` be
corrected using the Fisher information of the current receiving marginal, as
suggested by the natural-gradient message-passing paper?

The short answer is that the paper projects the **exact BP log-message**, not an
already moment-matched Gaussian message. A Gaussian log-message is already in
the Gaussian tangent space, so applying the Fisher projector to it alone returns
its existing natural parameters and makes no correction.

The nontrivial construction is to use the analytic moment-matched Gaussian as a
baseline and Fisher-project the non-Gaussian residual.

## Mathematical identity

Let

- \(\ell_{\mathrm{exact}}(s)\) be the exact residual-sine pushforward
  log-density;
- \(m_{\mathrm{MM}}(s)\) be its analytic moment-matched Gaussian;
- \(q(s)\) be the current Gaussian receiving marginal;
- \(T(s)=(s,ss^\mathsf{T})\) be the Gaussian sufficient statistics; and
- \(\mathcal P_q\) be the Fisher tangent projector

\[
\mathcal P_q[r]
=
F(q)^{-1}\operatorname{Cov}_q[T,r]
=
\nabla_{\mu_q}\mathbb E_q[r].
\]

Because `log(m_MM)` is a Gaussian quadratic,

\[
\mathcal P_q[\log m_{\mathrm{MM}}]=\eta_{\mathrm{MM}}
\]

for every valid Gaussian projection point \(q\). Therefore the proposed
correction is

\[
\begin{aligned}
\eta_{\mathrm{corrected}}
&=
\eta_{\mathrm{MM}}
+
\mathcal P_q[
  \ell_{\mathrm{exact}}-\log m_{\mathrm{MM}}
]\\
&=
\mathcal P_q[\ell_{\mathrm{exact}}].
\end{aligned}
\]

Thus a **full** residual correction is algebraically identical to directly
Fisher-projecting the exact BP log-message. It is not generally a small or
stabilizing adjustment to moment matching. The moment-matched baseline cancels
from the final mathematical target; its practical value is only that cubature
evaluates the smaller non-Gaussian residual rather than the entire log-density.

## Temporary implementation that was evaluated

The temporary implementation made four changes, all of which have now been
removed.

### 1. Exact-minus-Gaussian correction expression

`src/nodes/mv_residual_sine/node.jl` temporarily defined an internal
`MvResidualSineForwardCorrection` expression containing:

- the exact `MvResidualSineForwardMessage`;
- the analytic moment-matched Gaussian mean;
- its Cholesky factor and log determinant.

It evaluated

```julia
log_correction(s) = log_exact_pushforward(s) - log_moment_gaussian(s)
```

without converting either message through a checked distribution constructor.

### 2. Fisher-corrected forward rule

`src/nodes/mv_residual_sine/rules/natural_gradient.jl` temporarily accepted
`TangentProjection(type = Unscented)` for the forward message. It performed:

```julia
baseline_mean, baseline_cov = exact_residual_sine_moments(m_in)
eta_baseline = gaussian_natural_parameters(baseline_mean, baseline_cov)

correction = log_exact_pushforward - log_moment_gaussian
eta_correction = fisher_project_with_degree5_cubature(q_out, correction)

eta_target = eta_baseline + eta_correction
```

The existing `NGMPEdgeState` then applied natural-coordinate damping/momentum
to `eta_target` in the usual way.

The backward message was unchanged: every forward arm used the existing exact
analytic Gaussian Fisher projection backward.

### 3. Experiment selector

`experiments/xor_ctransition_mvresidual_sine.jl` temporarily accepted

```text
PHI_FORWARD_MODE=fisher
```

and mapped it to `TangentProjection(type = Unscented)`. The original default
mode list was not changed; `fisher` was opt-in.

### 4. Tests

`test/ngmp/mv_residual_sine_rule_tests.jl` temporarily tested that:

1. baseline plus residual correction equaled a direct degree-5 cubature
   projection of the exact forward log-message;
2. the result differed from plain moment matching;
3. the result changed when the receiving marginal changed; and
4. the rule and full toy graph worked for `delta`, `moments`, and `fisher`.

The focused residual-sine suite passed all 86 tests with the temporary arm.

## Experimental results

No result files or predictive images were saved. The numbers below came from
terminal output.

### Smoke test

Configuration: width 2, 60 samples, 3 training iterations, 3 prediction
iterations, softplus baseline disabled.

| Arm | Train time | Prediction time | Test MSE | Normalized test MSE |
|---|---:|---:|---:|---:|
| Fisher-corrected | 0.264 s | 0.048 s | 0.41258 | 1.79458 |

The smoke test established that the graph was wired correctly, all activation
states fired once per iteration, and the free-energy values were finite. Three
iterations are not enough to assess XOR learning.

### Full width-4 run with the original damping schedule

Configuration: 2,000 samples, 80/20 split, 160 training iterations, 10
prediction iterations, `rho=0.9`, `omega=1.0`, one OpenBLAS thread,
`alpha=0.4`, `beta=0.2`, `max_step=1.0`.

| Arm | Train time | Prediction time | Test MSE | Normalized test MSE |
|---|---:|---:|---:|---:|
| Fisher-corrected | 61.903 s | 5.546 s | 0.45042 | 2.10325 |

Additional observations:

- The forward-message microbenchmark was about `47.8 us`.
- The model essentially remained at the initial predictor.
- Many downstream `FastCholesky` covariance-asymmetry warnings appeared.
- The free-energy diagnostic remained finite but moved from approximately
  `-11257.49` to `14043.77`.

### Conservative natural damping

The run was repeated with activation-message momentum removed and a smaller,
bounded update:

```text
NGMP_ALPHA=0.1
NGMP_BETA=0.0
NGMP_MAX_STEP=0.25
```

| Arm | Train time | Prediction time | Test MSE | Normalized test MSE |
|---|---:|---:|---:|---:|
| Fisher-corrected, conservative damping | 84.939 s | 14.899 s | 0.25045 | 1.16949 |

This schedule greatly reduced the warnings and improved the prediction, but it
still performed worse than the constant-mean baseline and far worse than plain
analytic moment matching.

### Reference results from the existing experiment summary

| Arm | Train time | Prediction time | Test MSE | Normalized test MSE |
|---|---:|---:|---:|---:|
| Residual sine, analytic moments | 40.466 s | 3.536 s | **0.10699** | **0.49959** |
| Residual sine, local delta | 43.519 s | 3.609 s | 0.45093 | 2.10561 |
| Softplus, unscented baseline | 56.808 s | 4.111 s | 0.11357 | 0.53031 |

The full Fisher-corrected result was close to the failed local-delta result,
although its predictive variance did not explode to the same degree.

## Site-curvature audit

A separate scalar diagnostic evaluated the corrected forward site on a grid of
360 combinations of:

- input means `0.0`, `1.5`, and `3.0`;
- input variances `0.5`, `1.0`, `2.0`, and `4.0`;
- receiving means `-4`, `-2`, `0`, `2`, and `4`; and
- receiving variances `0.1`, `0.5`, `1`, `2`, `4`, and `8`.

Results:

- `120 / 360` corrected Gaussian sites had non-positive precision.
- The smallest projected precision was approximately `-16.1774`, at input
  `(mean=3.0, variance=0.5)` and receiving marginal
  `(mean=-4.0, variance=0.5)`.

An improper factor-to-edge site is mathematically permissible: only the product
forming the complete marginal must be proper. In this graph, however, strongly
negative or highly conditioned site precisions make the surrounding structured
Gaussian and `ContinuousTransition` calculations numerically delicate. Plain
moment matching always returns a proper Gaussian covariance by construction.

## Interpretation

The outcome is compatible with the paper rather than a contradiction of it.
The theorem identifies a constrained-Bethe stationary equation; it does not
guarantee that:

- every projected factor site is a proper distribution;
- the corresponding fixed-point map is contractive;
- the default heavy-ball schedule converges;
- the stationary point predicts better than a forward moment projection; or
- a low-order numerical approximation accurately evaluates every Fisher
  expectation.

For this activation, `rho=0.9` permits derivatives as small as `0.1`. The exact
pushforward can consequently have sharp and non-log-concave local structure.
The receiving-marginal Fisher projection follows that local geometry, whereas
moment matching summarizes the complete transformed distribution with a proper
mean and covariance.

The implementation used the package's McNamee--Stenger degree-5 Gaussian
cubature (`2d^2 + 1` points). The tests prove equivalence between the residual
construction and a direct projection **under that same cubature rule**. They do
not prove that degree-5 cubature equals the exact Fisher integral for the
inverse-residual-sine log-density. Thus the poor result may combine an
unfavorable Fisher target with numerical integration error.

## Possible future experiments

1. **Validate the projection at low dimension.** Compare degree-5 cubature with
   high-order tensor Gauss--Hermite quadrature in dimensions one and two before
   another full training run.
2. **Use an adaptive trust region.** Propose the full Fisher target but line
   search the natural-parameter step until the resulting complete marginal has
   a well-conditioned positive-definite precision. If the accepted step can
   approach one near convergence, this controls the path without deliberately
   replacing the paper's fixed point.
3. **Investigate a partial residual hybrid.** Use
   `eta_MM + kappa * eta_residual`, with `0 < kappa < 1`. This may retain the
   useful moment solution, but a permanently restricted `kappa` changes the
   stationary equations and should be described as a hybrid rather than NGMP.
4. **Reduce activation stiffness.** Repeat for smaller `rho`, where the
   residual-sine derivative stays farther from zero and the pushforward density
   is less sharply distorted.
5. **Audit actual inference states.** Record eigenvalues of the forward site,
   the opposing cavity, and their summed marginal precision at the first
   warning, instead of diagnosing only a synthetic scalar grid.

## Commands that were used

Smoke run:

```bash
XOR_RS_SMOKE=true PHI_FORWARD_MODE=fisher \
RUN_SOFTPLUS_BASELINE=false SAVE_OUTPUTS=false SHOW_PROGRESS=false \
OPENBLAS_NUM_THREADS=1 \
julia --project=. experiments/xor_ctransition_mvresidual_sine.jl
```

Full run:

```bash
PHI_FORWARD_MODE=fisher RUN_SOFTPLUS_BASELINE=false \
SAVE_OUTPUTS=false SHOW_PROGRESS=false OPENBLAS_NUM_THREADS=1 \
julia --project=. experiments/xor_ctransition_mvresidual_sine.jl
```

Conservative-damping run:

```bash
PHI_FORWARD_MODE=fisher RUN_SOFTPLUS_BASELINE=false \
COMPARISON_WARMUP=false SAVE_OUTPUTS=false SHOW_PROGRESS=false \
OPENBLAS_NUM_THREADS=1 NGMP_ALPHA=0.1 NGMP_BETA=0.0 \
NGMP_MAX_STEP=0.25 \
julia --project=. experiments/xor_ctransition_mvresidual_sine.jl
```

