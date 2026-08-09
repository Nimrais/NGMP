# When does NGMP help?

This directory is the canonical, script-based reproduction entry point for the
three controlled examples used to explain when natural-gradient message passing
(NGMP) differs from variational message passing (VMP). The Pluto notebooks in
`notebooks/` remain useful as tutorials, but are not required to regenerate the
figures here.

The examples answer one question in three steps:

1. `normal_mean_precision.jl`: in an IID Normal model with jointly unknown mean
   and precision (inverse observation variance), 20 seeds draw different true
   means and precisions. Marginal `KL(p_exact || q_method)` is averaged across
   seeds as the sample size grows, producing separate state/mean and precision
   performance plots for VMP and NGMP.
2. `poisson_state_space.jl`: projected mean-field VMP makes the non-conjugate
   Poisson model tractable, but discards temporal posterior dependence; native
   NGMP preserves the Gaussian chain after projecting each Poisson message. The
   prediction protocol removes nested 5%, 10%, 20%, and 50% subsets for 20 mask
   seeds, reports seed-averaged predictive NLL and RMSE, and generates separate
   trajectory and four-panel Bethe free-energy figures.
3. `gaussian_state_space.jl`: in a dynamic Normal model, increasing the chain
   length does not drive each local state's smoothing uncertainty to zero. The
   projected exact-BP correction therefore remains relevant.

## Reproduce all figures

Run from the repository root:

```sh
JULIA_NUM_THREADS=4 DATADEPS_ALWAYS_ACCEPT=true julia --project=. when_ngmp_helps/run_all.jl
```

The Sunspots dataset used by the Poisson example is obtained through DataDeps on
the first full run. Each example can also be run independently with the same
`julia --project=.` prefix.

Generated PDF and PNG files are written to `figures/`. Every publication panel
is also saved as a separate, title-free PDF for assembly in LaTeX/TikZ; the
combined files are inspection previews. Raw per-repetition values,
machine-readable summaries, and effective configurations are written to
`results/`. Scripts overwrite only their own named artifacts.

The Poisson study additionally writes a wide CSV table and a booktabs-compatible
`poisson_state_space_metrics_table.tex`. For a quicker non-publication run, the
number of Poisson masks can be overridden with
`WHEN_NGMP_POISSON_REPETITIONS`; the default remains 20.

## Smoke validation

Smoke mode reduces the simulation grids and uses a deterministic synthetic
Poisson series, so it does not need a dataset download:

```sh
WHEN_NGMP_SMOKE=true julia --project=. when_ngmp_helps/run_all.jl
```

To keep smoke artifacts outside the repository, set
`WHEN_NGMP_OUTPUT_DIR=/path/to/output`. The automated check in `test/runtests.jl`
uses a temporary directory this way.

## Interpretation

The projected-VMP Poisson baseline is labelled **Projected VMP (mean-field)**.
Projection is what makes the non-conjugate factor executable, while the chosen
mean-field constraint also removes temporal posterior dependence. The plots
therefore report marginal uncertainty and held-out predictive NLL directly,
without attributing every difference to projection alone.

Across the examples, NGMP uses the projected exact BP log-message while VMP uses
an expected log-factor. When neighboring cavity beliefs concentrate these local
objects agree. State-space process and observation noise can keep local cavity
beliefs broad even as the sequence grows, which is the regime where NGMP retains
an advantage.
