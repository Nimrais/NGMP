# NGMP-only heteroscedastic mean-capacity diagnostic

The log-precision path is frozen at 32 RBF features with `level_sd = 1.6`.
Results use 240 NGMP sweeps over 1 data-seed setting(s).
Latent-mean RMSE is measured against the known benchmark mean; the core grid is `|x| <= 2`.

| preset | full NLL | full latent-mean RMSE | full core-grid RMSE | sequential NLL | sequential latent-mean RMSE | sequential core-grid RMSE |
|---|---:|---:|---:|---:|---:|---:|
| NGMP, mean RBF-32 | 0.808 | 0.948 | 1.221 | 0.813 | 0.863 | 1.04 |
| NGMP, mean RBF-32, wide prior | 0.807 | 1.047 | 1.39 | 0.828 | 0.993 | 1.163 |
| NGMP, mean RBF-128 | 0.21 | 0.339 | 0.232 | 0.218 | 0.369 | 0.266 |
| NGMP, mean Matérn-3/2 RFF-128 | 0.214 | 0.344 | 0.241 | 0.235 | 0.427 | 0.296 |
