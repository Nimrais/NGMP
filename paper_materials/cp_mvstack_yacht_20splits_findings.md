# CP Fourier Features and MvStack Depth on Yacht

## Question

Can an end-to-end learned CP Fourier representation make an `L = 2`
MvStack heteroscedastic model outperform the corresponding `L = 1` model on
Yacht Hydrodynamics?

The experiment also tests whether the CP representation improves on the
50-feature random Fourier feature (RFF) basis previously used by the MvStack
model, and how both models compare with Bayes by Backprop (BBB).

## Evaluation protocol

All reported models use the same 20 splits from the versioned
`repeated-holdout-v1` UCI protocol:

- 308 Yacht observations with 6 input variables;
- 277 training observations and 31 locked test observations per split;
- standardization fitted only on the training partition;
- log predictive density (LPD) evaluated in the original target units;
- uncertainty reported as the standard error across the 20 splits.

Because every method uses the same held-out rows, paired differences across
splits provide the most informative comparison.

## Models

### CP Fourier `L = 1`

Each input coordinate is represented by an order-10 deterministic
Gauss–Hermite Fourier basis. A rank-8 CP decomposition represents their
tensor product without explicitly constructing all \(10^6\) product
features. The predictive mean is

\[
\mu(x)
=
\sum_{r=1}^{8}
\prod_{d=1}^{6}
z_d(x_d)^\top a_{d,r}
+ b_\mu.
\]

The observation variance is constant and estimated from the training
residuals. This model has 481 learned global coefficients.

### CP Fourier with MvStack `L = 2`

The `L = 2` model learns two CP functions end to end: one for the predictive
mean and one for the residual log-precision score. The score and a constant
bias are preserved as a vector by `MvStack`, followed by a learned readout:

\[
\begin{aligned}
\mu(x) &= f_{\mathrm{CP},\mu}(x),\\
s_1(x) &= f_{\mathrm{CP},\lambda}(x),\\
\eta(x) &= a_1 s_1(x) + b,\\
\lambda(x) &= \exp(\eta(x)),\\
y \mid x &\sim
\mathcal N\!\left(\mu(x),\lambda(x)^{-1}\right).
\end{aligned}
\]

Alternating conditional inference updates both CP tensors and the MvStack
readout. The `L = 2` model has 964 learned global coefficients.

The fixed configuration uses three alternating sweeps, 12 message-passing
iterations per coordinate block, CP regularization \(2\times10^{-3}\), and a
rank-8, order-10 basis.

## Main results

| Model | Learned coefficients | LPD (mean ± SE) | RMSE (mean ± SE) | Coverage 95% | Mean predictive variance |
|---|---:|---:|---:|---:|---:|
| CP Fourier `L = 1` | 481 | **−2.2185 ± 0.0741** | **2.0061 ± 0.0756** | 0.892 | 2.4601 |
| CP Fourier + MvStack `L = 2` | 964 | −2.5483 ± 0.0679 | 3.4078 ± 0.2154 | 0.940 | 7.3581 |
| 50-RFF MvStack `L = 2` | 104 | −3.3444 ± 0.1577 | 5.9227 ± 0.3857 | 0.840 | 8.5940 |
| Heteroscedastic BBB | 3,002 | −2.6565 ± 0.0230 | 3.0494 ± 0.2117 | 1.000 | 51.5749 |
| Homoscedastic BBB | 2,951 | −3.6925 ± 0.0030 | 3.4875 ± 0.2073 | 1.000 | 244.7431 |

BBB learns a mean and scale for every network coefficient, corresponding to
6,004 optimized variational scalars in the heteroscedastic model. CP and
MvStack use structured Gaussian message passing, so coefficient counts are
more comparable than raw variational-state sizes.

## Paired comparisons

### Does CP–MvStack `L = 2` beat CP `L = 1`?

No. Across matched test splits:

| Paired difference: `L = 2` minus `L = 1` | Mean ± SE |
|---|---:|
| LPD | **−0.3298 ± 0.0346** |
| RMSE | **+1.4016 ± 0.1548** |
| Coverage 95% | **+0.0484 ± 0.0113** |

`L = 2` has higher LPD on only 1 of the 20 splits, and that improvement is
only \(+0.0021\) nats per point. The median paired LPD change is
\(-0.3153\), while the observed range is \([-0.6296,+0.0021]\).

The extra precision function therefore improves interval coverage, but the
gain is not enough to compensate for the deterioration of the predictive
mean.

### Do CP features improve the MvStack model?

Yes. Relative to the 50-RFF MvStack `L = 2` model:

| Paired difference: CP `L = 2` minus RFF `L = 2` | Mean ± SE |
|---|---:|
| LPD | **+0.7961 ± 0.1014** |
| RMSE | **−2.5150 ± 0.2244** |

The CP model has higher LPD on all 20 splits. This is strong evidence that
the 50-feature RFF basis was an important bottleneck on Yacht.

### Comparison with heteroscedastic BBB

| Paired difference | LPD (mean ± SE) | RMSE (mean ± SE) | LPD wins |
|---|---:|---:|---:|
| CP `L = 1` minus BBB | **+0.4381 ± 0.0728** | **−1.0433 ± 0.1912** | 19/20 |
| CP–MvStack `L = 2` minus BBB | +0.1082 ± 0.0655 | +0.3583 ± 0.2120 | 14/20 |

CP `L = 1` clearly outperforms heteroscedastic BBB on these splits. The mean
LPD of CP–MvStack `L = 2` is also higher than BBB, but its paired advantage is
only about 1.7 standard errors and its RMSE is worse. That comparison should
be described as suggestive rather than decisive.

BBB attains 100% empirical 95% coverage with much larger predictive
variances. Its intervals are therefore conservative on this benchmark.

## Interpretation

The CP representation succeeds as a basis but unrestricted end-to-end depth
does not succeed as an optimization strategy.

The `L = 1` CP tensor reaches training mean-squared errors of approximately
0.010–0.011 in standardized units. During `L = 2` training, the MvStack
precision path remains numerically stable and learns nonconstant precision,
but the joint likelihood updates move the mean tensor to training errors of
approximately 0.040–0.045. The test RMSE increase is consequently systematic.

This separates two effects:

1. **Representation:** deterministic coordinate Fourier features with an
   implicit CP tensor product are substantially more effective than 50 joint
   RFFs on Yacht.
2. **Depth:** allowing the heteroscedastic likelihood to refit the already
   strong mean tensor damages the predictive mean. The resulting increase in
   uncertainty improves coverage but lowers LPD overall.

The result does not imply that a CP precision function is unhelpful. It shows
that the variance head must not be allowed to erase the strong `L = 1` mean
solution.

## Next controlled experiment

The natural follow-up is an anchored residual construction:

1. fit the CP `L = 1` mean;
2. initialize `L = 2` exactly at the `L = 1` predictive distribution;
3. keep the mean tensor fixed, or place a strong trust-region prior around
   its `L = 1` factors;
4. learn only the CP precision tensor and centred MvStack readout;
5. select the residual scale using training-only validation data.

With a fixed mean, RMSE must remain unchanged, and the LPD comparison isolates
whether input-dependent variance is genuinely useful. The current end-to-end
result indicates that this separation is necessary.

## Reproduction

Run:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. \
    scripts/uci_yacht_cp_mvstack_l2_all_splits.jl
```

The runner is resumable and writes:

- per-split metrics and predictions under
  `results/cp_mvstack_uci/yacht_l2_20splits/splitXX/`;
- `runs.csv` and `summary.csv` for aggregate results;
- `paired.csv` and `paired_summary.csv` for matched `L = 2` versus `L = 1`
  comparisons;
- `table.md` for the compact result table.
