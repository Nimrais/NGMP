# Streaming heteroscedastic study (aleatoric benchmark, n_basis = 32)

Full batch vs sequential (10 batches), 20 paired seeds.

| arm | full NLL | full RMSE | sequential NLL | sequential RMSE | streaming penalty |
|---|---|---|---|---|---|
| VMP | 0.818 ± 0.029 | 0.948 ± 0.041 | 0.946 ± 0.046 | 0.958 ± 0.049 | -0.128 ± 0.048 |
| VMP, 1-step projection | 1.104 ± 0.022 | 0.912 ± 0.047 | 1.13 ± 0.027 | 0.914 ± 0.05 | -0.026 ± 0.011 |
| NGMP | 0.809 ± 0.032 | 0.984 ± 0.046 | 0.816 ± 0.033 | 0.965 ± 0.04 | -0.007 ± 0.011 |
