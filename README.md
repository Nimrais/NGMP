# NGMP: natural-gradient message passing in RxInfer

This repository is the code behind

> M. Lukashchuk et al., *Natural-Gradient Message Passing* (TMLR submission).

The Julia package `SurrogateModelling` (`src/`) implements the natural-gradient
message-passing (NGMP) rules on top of forks of RxInfer/ReactiveMP: the
`NGMPDependencies` dispatch, the natural-parameter `DampingMeta` optimizer
(damping, heavy-ball momentum, vector-transport momentum), and the
non-conjugate nodes used in the paper (`PoissonExp`, `Log`, `softdot`, the
exponential precision links, ...). Everything below explains how each figure
and table in the paper is produced from this repository.

## Setup

Requirements: Julia 1.11, Git LFS, network access on the first run (UCI and
sunspot data are fetched through DataDeps; ETTh CSVs live in `data/`).

```sh
git clone git@github.com:Nimrais/NGMP.git && cd NGMP
git lfs install && git lfs pull            # result CSVs, baseline checkpoints
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` pins the RxInfer and ReactiveMP forks under `[sources]`;
`Manifest.toml` is committed. The neural baselines have their own environments
(`baselines/bbb_uci`, `baselines/dvi`, `baselines/bpc`, `baselines/ivon`), each
instantiated the same way with `--project=baselines/<name>`.

Useful environment variables: `DATADEPS_ALWAYS_ACCEPT=true` (accept dataset
downloads without a prompt) and `JULIA_NUM_THREADS=<n>` (all studies below use
threads over seeds or masks).

## Reproducing the paper

Every paper artifact is listed below with the script that produces it. The
"cheap" studies (Section *Comparing VMP to NGMP* and its appendices) rerun in
minutes to a couple of hours and their persisted results are committed, so
figures can be redrawn without inference. The UCI and ETTh benchmarks are
expensive and depend on trained baselines; their result files are committed and
their tables are regenerated from those files by validated assembler scripts.

### 1. Comparison study: `when_ngmp_helps/`

`when_ngmp_helps/` is a self-contained reproduction package (see its
[README](when_ngmp_helps/README.md)). Each study is one script with a compute
stage that writes `when_ngmp_helps/results/*.csv` (tracked with git LFS) and a
render stage that draws every figure and table from those CSVs.

```sh
# rerun everything (inference + rendering); ~2-3 h, dominated by streaming_hetero
JULIA_NUM_THREADS=8 DATADEPS_ALWAYS_ACCEPT=true julia --project=. when_ngmp_helps/run_all.jl

# redraw every figure/table from the committed CSVs, no inference (seconds)
julia --project=. when_ngmp_helps/render_all.jl

# a single study; add --render-only to skip inference
julia --project=. when_ngmp_helps/poisson_state_space.jl [--render-only]
```

Figures land in `when_ngmp_helps/figures/` (untracked), tables and summaries in
`when_ngmp_helps/results/`.

| Paper item | Script | Output files |
|---|---|---|
| Fig. `edge-uncertainty-normal-kl` (Normal mean-precision KL, 20 instances) | `when_ngmp_helps/normal_mean_precision.jl` | `figures/normal_kl_state.pdf`, `figures/normal_kl_precision.pdf` |
| Table `edge-uncertainty-poisson` (sunspot NLL/RMSE, 20 masks) and appendix Table `app-budget-poisson` (with the NCVMP control) | `when_ngmp_helps/poisson_state_space.jl` | `results/poisson_state_space_metrics_table.tex`, `results/poisson_state_space_metrics_table_full.tex` |
| Fig. `edge-uncertainty-poisson` (50 % holdout gap window, depth profiles) | same | `figures/poisson_gap50_vmp.pdf`, `poisson_gap50_ngmp.pdf`, `poisson_depth_nll.pdf`, `poisson_depth_variance.pdf` |
| Appendix Fig. `app-poisson-fe` (Bethe traces per holdout fraction, PVMP/NCVMP/NGMP) | same | `figures/poisson_bethe_heldout_{5,10,20,50}.pdf` |
| Appendix Fig. `poisson-bethe-damping-progress` (undamped vs damped outer map on the synthetic random walk, four chain lengths) and the sunspot-mask settle sweeps quoted in the same appendix | `when_ngmp_helps/poisson_damping.jl` (includes `poisson_state_space.jl`; settings α=1 / α=0.25 / α=0.5, β=0.2; synthetic: 20 seeds × N ∈ {100,250,500,1000} × 200 sweeps; sunspot: 20 masks × 60 sweeps; ~20 min with 8 threads) | `figures/poisson_damping_synthetic_n{100,250,500,1000}.pdf` (paper), `figures/poisson_damping_heldout_{5,10,20,50}.pdf` (supplementary), `results/poisson_damping_summary.md`, `results/poisson_damping_synthetic{,_fits}.csv`, `results/poisson_damping_free_energy.csv` |
| Table `edge-uncertainty-streaming` (full batch vs ten sequential batches, 20 seeds) and appendix Table `app-budget-streaming` | `when_ngmp_helps/streaming_hetero.jl` (model in `hetero_model.jl`) | `results/streaming_hetero_summary.md` (numbers are typed into the `.tex` table) |
| Figs. `edge-uncertainty-streaming`, `edge-uncertainty-variance`, appendix `app-streaming-predictive`, `app-streaming-variance` | same | `figures/streaming_{vmp,cavity}_{full,sequential}.pdf`, `figures/streaming_{vmp,cavity}_variance_{full,sequential}.pdf` |
| Appendix Figs. `app-streaming-fe`, `edge-uncertainty-collapse` | same | `figures/streaming_bethe_{full,sequential}.pdf`, `figures/streaming_collapse.pdf` |

Knobs: `WHEN_NGMP_POISSON_REPETITIONS` (masks and synthetic seeds),
`WHEN_NGMP_DAMPING_ITERATIONS` / `WHEN_NGMP_DAMPING_SYNTHETIC_ITERATIONS` (outer
sweeps of the damping diagnostic), `WHEN_NGMP_SMOKE=true` (tiny grids,
synthetic Poisson series, no download), `WHEN_NGMP_OUTPUT_DIR` (write elsewhere).
The smoke test `julia --project=. when_ngmp_helps/test/runtests.jl` runs the
whole package in smoke mode, deletes the figures, and re-renders them from the
CSVs.

### 2. UCI regression (Section *Regression*, Tables `uci-nll`, `uci-rmse`, `app-uci-nll`)

All methods share the `repeated-holdout-v1` protocol (six UCI data sets, 20
deterministic 90/10 splits based at seed `20260726`, train-only
standardization, inner validation split, refit on the full training part). The
exact split indices are stored in every result directory as
`split_manifest.jld2`, and the table generator refuses to run if any manifest
differs.

| Row(s) | How it is produced | Result directory |
|---|---|---|
| NGMP (depth-3 heteroscedastic hierarchy, 1,000 Matérn-3/2 features, vector-transport momentum α=0.6, β=0.8) | `julia --project=. scripts/uci_deep_kernel_direct_final_decreasing_matern1000_depth_ablation.jl` runs depths 1–5 on all six data sets (`UCI_DATASETS`, `UCI_SPLITS` override); the depth-3 rows are the paper's NGMP row. `scripts/package_ngmp_uci_depth_ablation.jl` packages the full ablation. | `paper_materials/ngmp_uci/` (`runs.csv`, `summary.csv`, `table.tex`, `config.toml`), ablation in `paper_materials/ngmp_uci_depth_ablation/` |
| BBB, hoBBB | `baselines/bbb_uci/run.jl`, command in [`paper_materials/README.md`](paper_materials/README.md) §1 | `paper_materials/bbb_uci/...` |
| dDVI, hodDVI, DVI, hoDVI | `baselines/dvi/run.jl` and its shell launchers, commands in `paper_materials/README.md` §2–4 (GPU/Reactant, ~48 GPU hours in total) | `paper_materials/dvi_uci/...` |
| hoBPC | `baselines/bpc/run.jl`, `paper_materials/README.md` §5 | `paper_materials/bpc_uci/...` |
| hoIVON | `julia --project=baselines/ivon baselines/ivon/run_uci.jl pilot`, then `full`, then `summarize` (see [`baselines/ivon/README.md`](baselines/ivon/README.md), section "Standalone UCI regression benchmark") | `paper_materials/ivon/uci/` |
| Final combined table (all rows, bold = CI-compatible with the best) | `julia --project=. paper_materials/make_final_baseline_table.jl` (validates 840 rows and identical split manifests before printing) | stdout Markdown; the LaTeX rows in `sections/modelling.tex` and `sections/apendix_modelling.tex` are these numbers |

`paper_materials/README.md` documents the compute budget of the baselines
(about 100 single-GPU machine hours end to end) and the provenance of every
committed baseline directory. Baseline checkpoints are in git LFS.

### 3. ETTh ensemble forecasting (Section *Ensemble Forecasting*, Tables `etth-precision-gated`, `etth-precision-gated-full`, Fig. ETTh1 h=192)

The frozen expert bank (CNN, DLinear, NLinear, LSTM, NConv, q10/q90) and the
64-dimensional VAE context are the published PrecisionGatedExperts (PGE)
checkpoints of Lukashchuk et al. (2026). They are **not** in this repository:
copy the PGE repository's `models/*.jld2` into `models/` and its ETTh CSVs into
`data/` (both directories are gitignored), and point `CT_REFERENCE_REPO` /
`ETTH_PGE_REPO` at a checkout of that repository (default
`~/repos/probabilistic_ensemble_forecasting`) so that the historical PVMP
(Dynamic PGE) and Adam-gate predictions can be replayed.

| Row(s) | How it is produced | Result location |
|---|---|---|
| NGMP, affine precision gate (`linear2_ngmp`: 66-d context, two-level log-precision head, native NGMP) | per cell: `julia --project=. experiments/etth_dynamic_ct_table.jl prepare ETTh1 192` then `... fit ETTh1 192`, for `ETTh1`/`ETTh2` × `96,192,336,720`; `... table` prints the summary. Model definition in `experiments/dynamic_deep_kernel_precision.jl`. | `results/etth_ct_table/<dataset>_h<H>_linear2_ngmp.jls` (gitignored) |
| IVON affine and ReLU gates (K = 1000 posterior draws) | `cd baselines/ivon && julia --project=. run.jl pilot && julia --project=. run.jl full && julia --project=. run.jl summarize` | `paper_materials/ivon/moe/` (checkpoints in git LFS) |
| PVMP (Dynamic PGE) and Adam gates | numbers replayed from the PGE repository's `paper/results_vae_std/dynamic` artifacts; nothing is retrained | external |
| Main table, appendix table, ETTh1 h=192 figure and its caption | `julia --startup-file=no --project=. scripts/assemble_etth_precision_study.jl` (recomputes every metric and interval from stored predictions, checks frozen-checkpoint hashes, replays stored metrics to 1e-10) | `paper_materials/etth_precision_gated/{main_table,appendix_table}.tex`, `etth1_h192_ngmp_vmp.pdf`, `etth1_h192_ngmp_vmp_caption.tex`, `summary.csv` |

`paper_materials/etth_precision_gated/ensemble_forecasting.tex` is the
hand-written section text that surrounds the table; see that directory's
[README](paper_materials/etth_precision_gated/README.md) for the protocol
details.

### 4. Copying artifacts into the paper

The paper repository expects the files under `figures/` with these names
(everything else keeps its name):

| Repository file | Paper file |
|---|---|
| `when_ngmp_helps/figures/*.pdf` | `figures/*.pdf` |
| `when_ngmp_helps/results/poisson_state_space_metrics_table{,_full}.tex` | `figures/poisson_state_space_metrics_table{,_full}.tex` |
| `paper_materials/etth_precision_gated/main_table.tex` | `figures/etth_precision_main_table.tex` |
| `paper_materials/etth_precision_gated/appendix_table.tex` | `figures/etth_precision_appendix_table.tex` |
| `paper_materials/etth_precision_gated/ensemble_forecasting.tex` | `figures/etth_ensemble_forecasting.tex` |
| `paper_materials/etth_precision_gated/etth1_h192_ngmp_vmp.{pdf,tex}` | `figures/etth1_h192_ngmp_vmp.pdf`, `figures/etth1_h192_ngmp_vmp_caption.tex` |

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'            # library (NGMP rules, damping, nodes)
julia --project=. when_ngmp_helps/test/runtests.jl      # comparison-study smoke reproduction
julia --project=. test/etth_precision_study_tests.jl    # ETTh assembler
```

Baseline test commands are listed in `paper_materials/README.md`.

## What is not part of the paper

`poisson_surrogate_model.jl`, `ssm_ng_bp_1d.jl`, the `etth2_*` and
`dynamic_vmp_vs_ngmp_etth2.jl` scripts at the repository root, `cavi/`,
`notebooks/`, `mnist_experiments/`, and most of `scripts/` and `experiments/`
are exploratory or superseded material (development history, ablations, and
the Wishart state-space study that is not in the submitted paper). Nothing in
the paper depends on them.
