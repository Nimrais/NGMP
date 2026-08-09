# NGMP UCI paper artifacts

Final selected NGMP direct deep-kernel benchmark on the repository's shared
`repeated-holdout-v1` UCI splits (base seed `20260726`, 20 repetitions, 90/10
outer train/test partition). The selected default is hierarchy depth 3, with
two heteroscedastic precision levels above the mean layer.

- `runs.csv`: one row per dataset and split (120 successful runs).
- `summary.csv`: mean, split standard deviation, and standard error.
- `table.md` and `table.tex`: paper tables reporting mean ± approximate 95%
  confidence-interval half-width (`1.96 × standard error`). This intentionally
  corrects the BBB artifact table's use of standard deviation; the other author
  will update that baseline separately.
- `split_manifest.jld2`: exact shared split specifications.
- `config.toml`: selected model and inference configuration.

The local Wine loader contains 1,599 observations, while Wu et al.'s DVI table
reports 1,588; Wine is therefore not an exact reproduction of that paper row.
