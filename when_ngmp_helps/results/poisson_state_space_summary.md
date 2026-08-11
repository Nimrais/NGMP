# Poisson state-space study

Data source: Sunspots
NLL uses the notebook plug-in predictive rate `exp(m + v/2)`; RMSE is computed from that rate and the held-out count within each mask.
NLL and RMSE are shown as means ± 95% confidence intervals across 20 mask seeds.

| Held out | VMP NLL | VMP, 1 sweep NLL | NGMP NLL | VMP RMSE | VMP, 1 sweep RMSE | NGMP RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 4.534 ± 0.087 | 89.030 ± 2.808 | 4.532 ± 0.085 | 13.918 ± 0.385 | 63.217 ± 1.605 | 13.935 ± 0.378 |
| 10% | 4.569 ± 0.058 | 91.584 ± 1.594 | 4.566 ± 0.058 | 13.972 ± 0.260 | 63.371 ± 0.735 | 13.995 ± 0.260 |
| 20% | 4.672 ± 0.039 | 98.302 ± 1.421 | 4.661 ± 0.037 | 14.429 ± 0.214 | 63.925 ± 0.502 | 14.446 ± 0.208 |
| 50% | 5.411 ± 0.162 | 120.209 ± 1.282 | 4.930 ± 0.035 | 16.839 ± 0.471 | 65.393 ± 0.350 | 15.789 ± 0.162 |

**Bethe free-energy caption.** Mean per-observed-count Bethe free-energy diagnostics with 95% intervals across 20 masks after removing nested 5%, 10%, 20%, and 50% subsets of likelihood factors. Each VMP trace evaluates its holdout graph's variational objective; each NGMP trace is a surrogate Bethe diagnostic because its local Gaussian surrogates change between outer iterations.
