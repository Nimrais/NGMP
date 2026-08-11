# Streaming heteroscedastic study (aleatoric benchmark, n_basis = 32)

Full batch vs sequential (10 batches), 20 paired seeds.

| arm | full NLL | full RMSE | sequential NLL | sequential RMSE | streaming penalty |
|---|---|---|---|---|---|
| VMP | 0.818 ± 0.029 | 0.948 ± 0.041 | 0.946 ± 0.046 | 0.958 ± 0.049 | -0.128 ± 0.048 |
| VMP, 1 sweep | 1.424 ± 0.105 | 0.943 ± 0.078 | 1.424 ± 0.096 | 0.913 ± 0.054 | -0.0 ± 0.047 |
| NGMP | 0.809 ± 0.032 | 0.984 ± 0.046 | 0.816 ± 0.033 | 0.965 ± 0.04 | -0.007 ± 0.011 |
