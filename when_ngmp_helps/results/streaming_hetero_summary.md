# Streaming heteroscedastic study (aleatoric benchmark)

Mean path: Matérn-3/2 RFF-128 with `signal_sd = 2.0`; log-precision path: RBF RFF-32 with `level_sd = 1.6`.
Full batch vs sequential (10 batches), 20 paired seeds.
Mean RMSE is measured against the benchmark's known latent mean; observed RMSE uses noisy held-out targets.

| arm | full NLL | full observed RMSE | full mean RMSE | sequential NLL | sequential observed RMSE | sequential mean RMSE | streaming penalty |
|---|---|---|---|---|---|---|---|
| PVMP | 0.276 ± 0.031 | 0.588 ± 0.037 | 0.306 ± 0.04 | 0.512 ± 0.044 | 0.583 ± 0.039 | 0.284 ± 0.044 | -0.235 ± 0.03 |
| NCVMP | 0.913 ± 0.015 | 0.577 ± 0.035 | 0.284 ± 0.036 | 0.916 ± 0.017 | 0.576 ± 0.035 | 0.277 ± 0.036 | -0.003 ± 0.005 |
| NGMP | 0.235 ± 0.031 | 0.591 ± 0.038 | 0.311 ± 0.041 | 0.244 ± 0.031 | 0.607 ± 0.042 | 0.334 ± 0.044 | -0.009 ± 0.01 |
