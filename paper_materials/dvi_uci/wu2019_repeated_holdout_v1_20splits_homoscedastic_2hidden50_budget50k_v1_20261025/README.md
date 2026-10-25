# Homoscedastic two-hidden-layer DVI campaigns

Two independent experiments use `d -> 50 -> 50 -> 1`: diagonal propagation
(`ddvi/`) and full propagation (`dvi/`). Observation noise variance is fixed
at 1 in standardized target units, matching the small homoscedastic baselines.
Selection and refitting retain the `selection-refit-max-50000-v1` protocol,
with 50,000-step budgets and the existing epoch-boundary budget checks.
The 14,000-step KL warmup and 1,000-step anneal are unchanged.

Each experiment covers Concrete Compressive Strength, Energy Efficiency
(heating load), Boston Housing, Combined Cycle Power Plant, Wine Quality Red,
and Yacht Hydrodynamics on the same 20 paired splits (120 fits per method).
The dDVI campaign completed on 2026-10-25 at 20:05 UTC: all 120 fits
succeeded and passed checkpoint metric replay. Its canonical results are in
`ddvi/`, including `checkpoint_replay.csv` and `final_baseline_table.md`.
The full-covariance DVI campaign has not yet produced a completed bundle.

After cloning, fetch binary artifacts with `git lfs pull`. To verify and
regenerate the saved dDVI comparison without retraining, run:

```sh
bash baselines/dvi/run_ddvi_homoscedastic_two_hidden_paper.sh finish
```

From the repository root, run either campaign independently:

```sh
bash baselines/dvi/run_ddvi_homoscedastic_two_hidden_paper.sh run
bash baselines/dvi/run_dvi_homoscedastic_two_hidden_paper.sh run
```

On one GPU, run these sequentially. Both default to Julia 1.11.9 and two
Reactant/GPU workers. `check` checks data and paired splits without training;
`smoke` verifies two-epoch CPU/GPU Yacht fits, checkpoint replay and resume.
`run` automatically performs that smoke check if the current source/environment
has not passed it. Smoke results live in a separate subdirectory and never
enter the paper comparison. They do not establish long-run convergence.

Rerun the same `run` command after interruption: successful fits with compatible
checkpoints are skipped; interrupted individual fits restart. Keep worker count,
backend, device, source and training settings unchanged when resuming. A complete
merged bundle can be reused without the local shard directories. `finish`
replays all saved metrics and regenerates the tables without training or a GPU.

Each method directory stores its own config, runs, summary, histories,
checkpoints (including refit losses), split manifest, replay report, logs,
campaign record, status, and source/environment snapshots. Its
`final_baseline_table.md` includes the ten completed baseline variants plus
that method. `paired_small_*` reports big-minus-small homoscedastic differences;
negative NLL/RMSE differences favor the big model.

To include both completed experiments in the full table:

```sh
julia +1.11 --project=. paper_materials/make_final_baseline_table.jl \
  --include-dvi-big --include-ddvi-big-homo --include-dvi-big-homo
```

The two homoscedastic flags also work independently. Missing, incomplete, or
incompatible results cause an error. The default table stays unchanged.

Optional environment variables: `DVI_HOMO_BIG_OUTPUT_ROOT` changes the parent
directory for both method bundles; `DVI_HOMO_BIG_GPU` selects the GPU (default
0); `DVI_HOMO_BIG_WORKERS` selects 1 or 2 GPU workers, or 1/2/4 CPU workers.
For CPU use, set `DVI_HOMO_BIG_BACKEND=zygote DVI_HOMO_BIG_DEVICE=cpu` explicitly.
`DVI_HOMO_BIG_JULIA_BIN` and `DVI_HOMO_BIG_DEPOT_PATH` override the executable
and package depot. Julia must report exactly 1.11.9. Custom output roots are
supported by each campaign's `finish`; standalone table flags use this default root.

Existing Git LFS rules cover binary checkpoints, split manifests and source
archives. Configs, metrics, logs, and reports use regular Git. Shard copies and
lock files are ignored; no result files need to be deleted to resume.
