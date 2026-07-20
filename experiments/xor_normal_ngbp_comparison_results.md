# Latent NormalMeanPrecision NG-BP experiment

## Does the true BP message exist?

Yes. For

```math
f(o, m, \tau) = \mathcal{N}(o \mid m, \tau^{-1}),
```

with independent cavity messages
`m_o = Normal(mu_o, v_o)`, `m_m = Normal(mu_m, v_m)`, and
`m_tau = Gamma(a, b)`, all three factor-to-variable BP messages are defined.

Toward the precision edge, the difference of the two Gaussian cavities is
Gaussian, so the exact message has the existing `NormalPrecisionMessage` form:

```math
m_{f \to \tau}(\tau)
= \mathcal{N}(\mu_o \mid \mu_m, v_o + v_m + \tau^{-1}).
```

Toward either Gaussian edge, integrating `tau` first gives a Student-t density
and integrating the other Gaussian cavity gives a Gaussian-Student-t
convolution:

```math
m_{f \to o}(o)
\propto \int \mathcal{N}(m \mid \mu_m, v_m)
\left(2b + (o-m)^2\right)^{-(a + 1/2)} dm.
```

The message exists but generally has no elementary closed-form density. The new
`GaussianStudentTMessage` represents this integral and evaluates it with a
shared 16-point Gauss-Hermite rule. Its first two log-density derivatives are
computed analytically from the resulting Gaussian mixture, so delta, unscented,
and quadrature tangent projections all use the same underlying BP target.

## Implemented arms

The comparison script is `experiments/xor_normal_ngbp_comparison.jl`.

| Arm | Factorization around `NormalMeanPrecision` | NGMP edges |
|---|---|---|
| `structured` | `q(w_mean, z_mean, out) q(za, gamma)` | none |
| `precision_ngmp` | same as structured | `gamma` only |
| `full_ngbp` | `q(w_mean, z_mean, out, za, gamma)` | `out`, `mu`, `gamma` |

The Softplus node uses UT NGMP in every arm. The global exponential gate-rate
parameter `beta` is inferred in every arm.

## Results

Configuration: `(1,3)` checkerboard, 400 samples, 40% training, 16 neurons,
fixed seeds, damping `alpha=0.2`, maximum natural step `0.1`.

### Projection comparison at 50 iterations

| Arm | Projection | Test MSE | Seconds |
|---|---:|---:|---:|
| structured | none | 0.158434 | 9.46 |
| precision NGMP | delta | **0.155579** | 9.93 |
| precision NGMP | UT | 0.157531 | 9.58 |
| precision NGMP | quadrature(16) | 0.157304 | 9.87 |
| full NG-BP | delta | 0.156758 | 9.83 |
| full NG-BP | UT | 0.157937 | 10.40 |
| full NG-BP | quadrature(16) | 0.157755 | 16.03 |

### Convergence comparison at 200 iterations

| Arm | Projection | Test MSE | Seconds | Relative time |
|---|---:|---:|---:|---:|
| structured | none | 0.149851 | 33.18 | 1.00x |
| precision NGMP | delta | **0.148188** | 35.08 | 1.06x |
| precision NGMP | UT | 0.148951 | 35.07 | 1.06x |
| full NG-BP | delta | 0.148431 | 35.50 | 1.07x |
| full NG-BP | UT | 0.148791 | 35.78 | 1.08x |

All Gamma posteriors remained proper. Every Normal NGMP state fired exactly once
per iteration. All arms were still improving slightly over the final ten
iterations, but their ordering was stable.

### Canonical `(2,2)` XOR at 100 iterations

The same 400-sample, 16-neuron setup gives the same ordering on XOR:

| Arm | Projection | Test MSE | Seconds |
|---|---:|---:|---:|
| structured | none | 0.155378 | 17.53 |
| precision NGMP | delta | **0.151316** | 18.07 |
| precision NGMP | UT | 0.153531 | 17.86 |
| full NG-BP | delta | 0.152312 | 18.05 |
| full NG-BP | UT | 0.153589 | 18.76 |

## Interpretation

1. Full latent BP is feasible and inexpensive with UT or delta. It does not
   improve predictive MSE over precision-only NGMP on this experiment.
2. Precision-only NGMP captures the useful correction at lower conceptual and
   implementation cost. It is the best next notebook arm.
3. Delta gives the best predictive MSE here. This does not make it the most
   accurate BP projection: on wide Gamma beliefs it can be biased, as shown in
   `notebooks/gaussian_surrogate.jl`.
4. UT remains the safer general default. It tracks quadrature closely while
   avoiding the full-BP quadrature runtime increase. Use delta when predictive
   MSE is the criterion and verify it against UT or quadrature on a representative
   run.

The full-BP arm currently runs with `free_energy=false`. Its edge messages are
defined, but a Bethe free-energy term additionally requires a tractable or
approximated three-edge local factor belief and entropy; that marginal rule is a
separate implementation problem.
