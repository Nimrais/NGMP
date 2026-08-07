# NGMP UCI depth-ablation paper artifacts

Final 1,000-feature NGMP configuration evaluated at hierarchy depths 1--5 on
the shared BBB-aligned `repeated-holdout-v1` splits.

- `runs.csv`: all 600 dataset/depth/split results.
- `benchmark_summary.csv`: summary emitted by the benchmark script.
- `summary.csv`: paper summary with standard deviation, standard error, and
  95% CI half-width for every metric.
- `table.md` and `table.tex`: one row per dataset and depth, reporting
  mean ± 95% CI half-width.
- `split_manifest.jld2`: exact shared split specifications.
- `config.toml`: fixed configuration and the single ablated factor.

Depth 1 is the homoscedastic damped baseline. Depths 2--5 use vector transport
with beta 0.8 and add progressively more precision-hierarchy levels. Mean
features remain fixed; this experiment measures uncertainty-hierarchy depth.
