# Poisson damping diagnostic

## Sunspot masks (supplementary panels)

Data source: Sunspots; 20 masks; 60 variational iterations per fit; Bethe free energy per observed count.

Settle sweep: first sweep from which the relative sweep-to-sweep change of a mask's per-count free energy stays below 1e-4; reported as the mean over the masks that settle, with the number of settling masks. Amplitude: mean peak-to-peak range of the per-count free energy over the last 10 sweeps.

| Held out | Arm | α | β | settle sweep (masks settled) | last-10 amplitude |
|---:|:---|---:|---:|:---|---:|
| 5% | undamped | 1.00 | 0.0 | 5.0 (20/20) | 1.11e-15 |
| 5% | damped25 | 0.25 | 0.0 | 14.0 (20/20) | 1.36e-12 |
| 5% | damped50m20 | 0.50 | 0.2 | 8.0 (20/20) | 1.42e-15 |
| 10% | undamped | 1.00 | 0.0 | 5.0 (20/20) | 1.15e-15 |
| 10% | damped25 | 0.25 | 0.0 | 14.0 (20/20) | 1.38e-12 |
| 10% | damped50m20 | 0.50 | 0.2 | 8.0 (20/20) | 1.51e-15 |
| 20% | undamped | 1.00 | 0.0 | 5.0 (20/20) | 1.29e-15 |
| 20% | damped25 | 0.25 | 0.0 | 14.0 (20/20) | 1.43e-12 |
| 20% | damped50m20 | 0.50 | 0.2 | 8.0 (20/20) | 1.69e-15 |
| 50% | undamped | 1.00 | 0.0 | 5.0 (20/20) | 9.77e-16 |
| 50% | damped25 | 0.25 | 0.0 | 14.0 (20/20) | 1.72e-12 |
| 50% | damped50m20 | 0.50 | 0.2 | 8.0 (20/20) | 1.38e-15 |

## Synthetic random walk (paper figure)

z_k = z_{k-1} + N(0, 0.10), all counts observed, 200 outer sweeps, seeds 42-61 (`Random.seed!(seed)`, the draw order of `poisson_surrogate_model.jl`; seed 42 with N = 1000 is the earlier synthetic series). Per seed and setting: settled = the free energy per observation changes by less than 1e-4 (relative) from some sweep on; oscillating = never settles within the budget but ends within 10x of the same seed's α = 0.25 final value; diverged = ends more than 10x above it; failed = the fit threw (a message left the natural domain). Medians are over the seeds that finished. RMSE is against the true latent log-rate.

| N | Arm | α | β | settled / oscillating / diverged / failed | median settle sweep | median peak F | median final F | median RMSE | log-rate min (median over seeds) | zero counts (median) |
|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 100 | undamped | 1.00 | 0.0 | 20 / 0 / 0 / 0 | 5 | 166.2 | 154.5 | 0.395 | -1.8 | 41 |
| 100 | damped25 | 0.25 | 0.0 | 20 / 0 / 0 / 0 | 14 | 165 | 154.5 | 0.395 | -1.8 | 41 |
| 100 | damped50m20 | 0.50 | 0.2 | 20 / 0 / 0 / 0 | 8 | 162.7 | 154.5 | 0.395 | -1.8 | 41 |
| 250 | undamped | 1.00 | 0.0 | 17 / 3 / 0 / 0 | 5 | 362.5 | 327.1 | 0.470 | -3.5 | 118 |
| 250 | damped25 | 0.25 | 0.0 | 20 / 0 / 0 / 0 | 20 | 359.4 | 327.1 | 0.470 | -3.5 | 118 |
| 250 | damped50m20 | 0.50 | 0.2 | 20 / 0 / 0 / 0 | 9 | 354.6 | 327.1 | 0.470 | -3.5 | 118 |
| 500 | undamped | 1.00 | 0.0 | 11 / 9 / 0 / 0 | 5 | 3330 | 789.1 | 0.661 | -6.7 | 234 |
| 500 | damped25 | 0.25 | 0.0 | 20 / 0 / 0 / 0 | 29 | 1023 | 789.1 | 0.693 | -6.7 | 234 |
| 500 | damped50m20 | 0.50 | 0.2 | 17 / 0 / 0 / 3 | 13 | 1919 | 911.3 | 0.442 | -6.7 | 234 |
| 1000 | undamped | 1.00 | 0.0 | 11 / 5 / 3 / 1 | 5 | 7.41e+04 | 4644 | 0.834 | -8.2 | 436 |
| 1000 | damped25 | 0.25 | 0.0 | 20 / 0 / 0 / 0 | 37 | 9066 | 2125 | 0.905 | -8.2 | 436 |
| 1000 | damped50m20 | 0.50 | 0.2 | 11 / 0 / 0 / 9 | 10 | 7.397e+05 | 6439 | 0.177 | -8.2 | 436 |

Seed 42, N = 1000 (the earlier synthetic series):

| Arm | status | peak F / obs | final F / obs | RMSE | settle sweep |
|:---|:---|---:|---:|---:|---:|
| undamped | ok | 2.682e+07 | 6354 | 5.255 | never |
| damped25 | ok | 0.607 | 0.2157 | 2.498 | 69 |
| damped50m20 | failed: DomainError with -386.55747643905113: | 0.6099 | 0.2193 | - | never |
