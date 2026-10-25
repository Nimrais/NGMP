# Completed dDVI big UCI experiment

Heteroscedastic diagonal DVI with architecture `d -> 50 -> 50 -> 2`.
The outputs model the predictive mean and log observation-noise variance.
All 120 fits completed: 20 paired splits each for Concrete, Energy (heating
load), Boston Housing, Power, Wine Quality Red, and Yacht. All 120 saved
posteriors passed metric replay verification.

## Saved artifacts

- `ddvi/runs.csv`, `summary.csv`, and `table.*`: per-split and aggregate results.
- `ddvi/checkpoints/`: all 120 refitted posterior checkpoints, including
  preprocessing, training settings, and refit loss histories.
- `ddvi/histories/`: model-selection training histories.
- `ddvi/config.toml` and `split_manifest.jld2`: configuration and exact splits.
- `ddvi/checkpoint_replay.csv`: verification results for the saved posteriors.
- `final_baseline_table.md`: the combined table generated after completion.
- `paired_differences.csv`, `paired_summary.csv`, and `comparison.md`:
  comparisons against the original one-hidden-layer heteroscedastic dDVI.
- `logs/` and `status.toml`: validation logs and completed campaign status.
- `provenance/20261024T204953Z/`: source snapshot, Git revision and diff,
  Julia version, package environments, and dataset checksums used for the run.

Checkpoints, split manifests, and the compressed source snapshot use Git LFS.
Text artifacts and code use regular Git. Local shard copies, preflight output,
process files, and the incomplete first snapshot are excluded from Git.

## Reuse without training

After cloning, retrieve LFS artifacts with `git lfs pull`. From the repository
root, regenerate the combined table using only saved results:

```sh
julia --project=. paper_materials/make_final_baseline_table.jl > /tmp/full_baseline_table.md
```

This reads saved results for all methods and checks their split manifests;
it does not train models. Keep the experiment directory at its recorded
repository-relative location so the generator can find it.

The training launcher `baselines/dvi/run_ddvi_two_hidden_paper.sh` is for
reproducing the experiment. Its resume checks use the local shard directories,
which are not part of this saved result bundle; running it in a fresh clone
will train the fits again.
