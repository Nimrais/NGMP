# Joint Normal mean–precision study

Each of the 20 seeds draws a new true mean and precision from the inference priors, then reuses prefixes of one observation stream across sample sizes.
The two figures report mean marginal KL ± 95% confidence intervals across seeds 42–61.
KL is oriented as `KL(p_exact || q_method)` for the shared mean/state and precision marginals separately.
The machine-readable method-wise values are stored in `normal_mean_precision_aggregate.csv`.
