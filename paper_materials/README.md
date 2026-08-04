# Reproducing the UCI paper results

This directory contains the complete BBB, diagonal-DVI (`dDVI`), and
full-covariance DVI results used for the paper comparison. The final result
directories contain 720 successful fits and no failed rows:

| Method | Likelihood | Final directory | Successful fits |
|---|---|---|---:|
| BBB | homoscedastic and heteroscedastic | `bbb_uci/blundell2015_repeated_holdout_v1_20splits_20260728` | 240 |
| dDVI | heteroscedastic | `dvi_uci/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/ddvi` | 120 |
| DVI | heteroscedastic | `dvi_uci/wu2019_repeated_holdout_v1_20splits_robust_selection_optimized_v2_20260803/dvi` | 120 |
| dDVI | homoscedastic | `dvi_uci/wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804/ddvi` | 120 |
| DVI | homoscedastic | `dvi_uci/wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804/dvi` | 120 |

Each set covers Yacht, Concrete, Energy, Housing, Power, and Wine with 20
deterministic repeated holdouts per dataset. Use the merged `bbb`, `ddvi`, and
`dvi` directories above for tables. Directories named `*_shards` and
`pilots` preserve execution provenance and timing/quality checks; they are not
additional independent paper results.

## Reproducibility scope

The commands below reproduce the data splits, model definitions, training
budgets, selection/refit procedure, and metrics. Every result directory also
records the complete effective settings in `config.toml`; that file is the
authoritative protocol record.

The saved CSV values are not promised to be bit-for-bit identical on another
machine. GPU reductions, compiler versions, and CPU math libraries can change
the last digits or an early-stopping decision. A reproduction should have the
same complete set of successful rows and numerically/statistically comparable
metrics. The original runs used Julia 1.11.9 on an Intel Core i7-11700
(8 cores/16 threads); accelerated runs used an NVIDIA RTX 3090. The baseline
`Project.toml` files constrain compatible dependency
families, but baseline-local manifests from the original executions were not
archived. Consequently `Pkg.instantiate()` can select newer compatible patch
versions; the saved checkpoints, configurations, source revisions, and
hardware record provide protocol-level rather than bitwise reproducibility.

There is one provenance detail to keep in mind. The final heteroscedastic DVI
directory is a validated consolidation of:

- 100 optimized Reactant/GPU rows for the five non-Yacht datasets;
- legacy Zygote/CPU Yacht splits 1--19; and
- optimized Reactant/GPU Yacht split 20.

Its `provenance.toml` records the source of every component. A clean rerun of
all 120 rows with the current GPU launcher is scientifically equivalent, but
is not a byte-for-byte recreation of that mixed-backend artifact. Instructions
for both a clean rerun and exact reconstruction from its raw components are
given below.

## Prerequisites and setup

Run all commands from the repository root:

```sh
cd /path/to/NGMP
```

Install Julia 1.11, Git LFS, and, for accelerated DVI, an NVIDIA GPU with a
working driver. Then fetch the saved checkpoints and instantiate both Julia
environments:

```sh
git lfs install
git lfs pull

julia --project=baselines/bbb_uci -e 'using Pkg; Pkg.instantiate()'
julia --project=baselines/dvi -e 'using Pkg; Pkg.instantiate()'
```

The UCI loaders use DataDeps. The launch commands set
`DATADEPS_ALWAYS_ACCEPT=true`, so a first run downloads and accepts the
datasets automatically and therefore requires network access.

Do not reproduce into one of the committed result directories. Successful
configurations there are resumable and will be skipped. Use a fresh output
root instead:

```sh
export REPRO_ROOT="$PWD/reproduced_results"
mkdir -p "$REPRO_ROOT"
```

Before a long run, the implementation tests can be checked with:

```sh
julia --project=baselines/bbb_uci baselines/bbb_uci/test/runtests.jl
julia --project=baselines/dvi -e 'using Pkg; Pkg.test()'
```

## Expected runtime

The following estimates are for the original machine described above, with a
warm package depot. They are wall-clock estimates, not the sum of CPU/GPU time
over concurrent processes:

| Stage | Execution layout | Estimated wall time |
|---|---|---:|
| BBB, both likelihoods | one CPU process, 240 fits | 45--60 minutes |
| Heteroscedastic dDVI | four concurrent CPU shards, 120 fits | 14.5--16 hours |
| Heteroscedastic DVI clean rerun | one Reactant/GPU shard, 120 fits | 14--16 hours |
| Homoscedastic dDVI | four concurrent Reactant/GPU shards, 120 fits | 4.1--4.6 hours |
| Homoscedastic DVI | four concurrent Reactant/GPU shards, 120 fits | 4.3--4.8 hours |
| All training stages, sequentially | layouts above | approximately 38--42 hours |

The estimate is derived from each successful row's recorded `total_seconds`,
which is selection time plus refit time plus evaluation time. For a serial
run, these values are summed. For four shards, split `s` belongs to shard
`1 + (s - 1) mod 4`; the values are summed within each shard and the maximum
shard sum estimates wall time:

```text
serial estimate = sum(total_seconds)
four-shard estimate = max over shards(sum(total_seconds in that shard))
```

The measured workload behind the table was:

- BBB: `0.702` hours summed over all rows;
- heteroscedastic dDVI: `50.230` CPU-hours in total and `14.187` hours for
  the slowest of the four shards;
- homoscedastic dDVI: `16.299` GPU-process-hours in total and `4.120` hours
  for the slowest shard;
- homoscedastic DVI: `17.089` GPU-process-hours in total and `4.309` hours
  for the slowest shard; and
- optimized heteroscedastic DVI on the five non-Yacht datasets: `10.862`
  hours on its single GPU shard. Two completed optimized Yacht measurements
  took `407` and `760` seconds; their mean extrapolates to `3.24` hours for
  20 Yacht splits, giving about `14.10` hours of recorded training work.

The ranges add allowance for process startup, package loading, XLA
compilation, shard imbalance, and final merging. A cold Julia depot or slow
dataset/network access can add roughly 10--30 minutes. Different CPUs and GPUs
can change the estimates substantially, so run `profile_step.jl` before the
heteroscedastic GPU campaign and the Power timing gate before the
homoscedastic campaign.

Reconstructing the canonical heteroscedastic DVI artifact from already
downloaded components does not retrain anything and should take less than five
minutes locally. `git lfs pull` transfers approximately 283 MB for all current
paper materials, so its time depends on the network connection.

## Shared comparison protocol

All three methods use the versioned `repeated-holdout-v1` split
implementation:

- six UCI regression datasets;
- 20 deterministic 90/10 outer train/test splits, based at seed `20260726`;
- train-only feature and target standardization;
- a deterministic inner validation split for epoch selection;
- fresh reinitialization and refitting on the complete outer training set;
- LPD with the exact target-standardization Jacobian, plus RMSE and coverage
  in original target units; and
- aggregate mean, sample standard deviation, and standard error over the 20
  splits.

`split_manifest.jld2` stores the exact original-row indices. Thus BBB, dDVI,
and DVI are paired on the same held-out examples. The method-specific
architecture, posterior, optimizer, KL schedule, and numerical safeguards are
documented in `baselines/bbb_uci/README.md` and `baselines/dvi/README.md`.

## 1. Reproduce BBB

This launches both likelihoods for all six datasets and 20 splits, producing
240 rows:

```sh
DATADEPS_ALWAYS_ACCEPT=true \
BBB_DATASETS=all \
BBB_SPLITS=20 \
BBB_LIKELIHOODS=homoscedastic,heteroscedastic \
BBB_OUTPUT_DIR="$REPRO_ROOT/bbb" \
BBB_MAKE_PLOT=false \
BBB_SHOW_PROGRESS=false \
julia --project=baselines/bbb_uci baselines/bbb_uci/run.jl
```

The important effective defaults are two 50-unit hidden layers, minibatches
of 100, Adam at `0.001`, one posterior sample per training minibatch, 20
posterior samples for evaluation, a standard-normal prior, and at most 500
epochs. Selection is evaluated every five epochs with patience 20, followed
by full-training refit.

## 2. Reproduce heteroscedastic dDVI

The published dDVI artifact used the portable Zygote/CPU implementation and
four disjoint shards. This shell block recreates the same scientific
configuration. Four processes run concurrently, so ensure that the machine
has enough CPU memory:

```bash
DDVI_ROOT="$REPRO_ROOT/heteroscedastic_ddvi"
mkdir -p "$DDVI_ROOT/ddvi_shards"

run_ddvi_shard() {
    shard="$1"
    split_ids="$2"
    env \
        DATADEPS_ALWAYS_ACCEPT=true \
        OPENBLAS_NUM_THREADS=1 \
        JULIA_NUM_THREADS=1 \
        DVI_DATASETS=all \
        DVI_PROPAGATION=diagonal \
        DVI_BACKEND=zygote \
        DVI_DEVICE=cpu \
        DVI_LIKELIHOODS=heteroscedastic \
        DVI_SPLITS=20 \
        DVI_SPLIT_IDS="$split_ids" \
        DVI_MAX_EPOCHS=10000 \
        DVI_MIN_EPOCHS=25 \
        DVI_VALIDATION_EVERY=5 \
        DVI_PATIENCE=500 \
        DVI_SELECTION_MAX_STEPS=0 \
        DVI_REFIT_MAX_STEPS=0 \
        DVI_TRAINING_BUDGET_PROTOCOL=unlimited-v1 \
        DVI_LEARNING_RATE=0.0003 \
        DVI_NUMERICAL_PROTOCOL=bounded-exp-step-kl-v1 \
        DVI_SAFE_EXP_MIN=-20 \
        DVI_SAFE_EXP_MAX=20 \
        DVI_KL_SCHEDULE_UNIT=steps \
        DVI_KL_WARMUP_STEPS=14000 \
        DVI_KL_ANNEAL_STEPS=1000 \
        DVI_OUTPUT_DIR="$DDVI_ROOT/ddvi_shards/$shard" \
        DVI_SELECTION_ONLY=false \
        DVI_MAKE_PLOT=false \
        DVI_SHOW_PROGRESS=false \
        julia --project=baselines/dvi baselines/dvi/run.jl &
}

run_ddvi_shard 01 1,5,9,13,17
run_ddvi_shard 02 2,6,10,14,18
run_ddvi_shard 03 3,7,11,15,19
run_ddvi_shard 04 4,8,12,16,20
wait

julia --project=baselines/dvi baselines/dvi/merge_shards.jl \
    "$DDVI_ROOT/ddvi" \
    "$DDVI_ROOT/ddvi_shards/01" \
    "$DDVI_ROOT/ddvi_shards/02" \
    "$DDVI_ROOT/ddvi_shards/03" \
    "$DDVI_ROOT/ddvi_shards/04"
```

Commit `945dc80` is the historical source/result snapshot for this run. The
current implementation retains the same dDVI protocol while adding stronger
validation and accelerated paths.

## 3. Reproduce heteroscedastic DVI

First profile the Reactant backend on the target GPU. The profiler does not
write paper results:

```sh
DVI_PROFILE_BACKENDS=reactant DVI_DEVICE=gpu \
julia --project=baselines/dvi baselines/dvi/profile_step.jl
```

If the projection is acceptable, run a clean 120-row full-DVI reproduction:

```sh
DVI_DATASETS=all \
DVI_PAPER_PROFILE_APPROVED=true \
DVI_PAPER_BACKEND=reactant \
DVI_PAPER_DEVICE=gpu \
DVI_PAPER_SHARDS=1 \
DVI_PAPER_OUTPUT_ROOT="$REPRO_ROOT/heteroscedastic_dvi_clean" \
baselines/dvi/run_bbb_matched_paper.sh
```

The launcher writes the raw run to `dvi_shards/01` and the verified merged
result to `dvi`. It uses the heteroscedastic likelihood, full covariance,
10,000 maximum epochs, a 14,000-step KL warmup followed by 1,000 annealing
steps, and the optimized selection path. On GPUs with less memory, lower XLA
preallocation or use CPU execution; changing backend/device is recorded in
`config.toml` and can lead to small numerical differences.

### Reconstruct the exact committed heterogeneous DVI artifact

The raw components are retained in Git LFS. This command validates their
split manifests and coverage, copies the selected histories/checkpoints, and
regenerates the canonical tables. The destination must not already exist:

```sh
HET_ROOT="paper_materials/dvi_uci"
CANONICAL_REBUILD="$REPRO_ROOT/heteroscedastic_dvi_canonical"

julia --project=baselines/dvi baselines/dvi/consolidate_paper_results.jl \
    "$CANONICAL_REBUILD" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_selection_optimized_v2_non_yacht_20260803/dvi" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_selection_optimized_v2_yacht_20260803/dvi_shards/20" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/dvi_shards/01" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/dvi_shards/02" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/dvi_shards/03" \
    "$HET_ROOT/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/dvi_shards/04"
```

The resulting `runs.csv`, derived summaries, and per-run artifacts represent
the same 120 source runs as the committed canonical directory.

## 4. Reproduce both homoscedastic DVI methods

The homoscedastic comparison uses a deliberately bounded protocol so both
methods complete in a practical runtime. Selection and refit are each capped
at 50,000 optimizer steps, checked at epoch boundaries. The same cap is used
for dDVI and DVI, and its identifier is stored in every row and checkpoint.

The final run used four concurrent Reactant/GPU shards per method and executed
dDVI first, then full DVI:

```sh
DVI_HOMO_OUTPUT_ROOT="$REPRO_ROOT/homoscedastic_budget50k" \
DVI_HOMO_PROFILE_APPROVED=true \
DVI_HOMO_METHODS=diagonal,full \
DVI_HOMO_BACKEND=reactant \
DVI_HOMO_DEVICE=gpu \
DVI_HOMO_SHARDS=4 \
baselines/dvi/run_homoscedastic_budgeted_paper.sh
```

The launcher sets each process to one Julia and one BLAS thread, disables XLA
GPU preallocation, and gives each of the four processes a default GPU-memory
fraction of `0.22`. Override `DVI_XLA_GPU_MEM_FRACTION` if required by the
available device. The merged outputs are:

```text
$REPRO_ROOT/homoscedastic_budget50k/ddvi
$REPRO_ROOT/homoscedastic_budget50k/dvi
```

The explicit `DVI_HOMO_PROFILE_APPROVED=true` records a human decision, not an
automatic quality claim. Before using a different machine or budget, run the
Power pilot/gate and inspect its generated `GATE.md`:

```sh
DVI_HOMO_GATE_OUTPUT_ROOT="$REPRO_ROOT/homoscedastic_gate" \
baselines/dvi/run_homoscedastic_power_scale_gate.sh
```

The committed pilot materials are under the homoscedastic result root's
`pilots/` directory. The 50,000-step pilot changed validation LPD by only
`0.000140` relative to its uncapped comparison and had no numerical failures
or clamp events; the conservative timing projection exceeded the original
automatic ten-hour gate, so the final run was launched after explicit manual
approval.

## Output layout

Every final result directory contains:

| Artifact | Purpose |
|---|---|
| `config.toml` | Complete effective, result-affecting configuration |
| `runs.csv` | One row per dataset/split/likelihood/method fit |
| `summary.csv` | Dataset-level aggregate metrics |
| `table.md`, `table.tex` | Ready-to-use rendered summaries |
| `split_manifest.jld2` | Exact outer and inner row indices |
| `histories/` | Selection/refit training histories |
| `checkpoints/` | Model, standardizer, split, and prediction state |
| `failures/` | Failure diagnostics; empty for the final DVI sets |

`merge_shards.jl` refuses overlapping/incomplete split coverage and rejects
result-affecting configuration mismatches. It regenerates the summary and
tables only after all expected rows and artifacts are present.

## Validate a reproduction

After all runs finish, this check verifies row counts, success status, and
summary coverage:

```sh
julia --project=baselines/dvi - "$REPRO_ROOT" <<'JULIA'
using CSV
using DataFrames

root = only(ARGS)
expected = [
    ("bbb", 240, 12),
    ("heteroscedastic_ddvi/ddvi", 120, 6),
    ("heteroscedastic_dvi_clean/dvi", 120, 6),
    ("homoscedastic_budget50k/ddvi", 120, 6),
    ("homoscedastic_budget50k/dvi", 120, 6),
]

for (relative_path, expected_runs, expected_summaries) in expected
    directory = joinpath(root, relative_path)
    runs = CSV.read(joinpath(directory, "runs.csv"), DataFrame)
    summary = CSV.read(joinpath(directory, "summary.csv"), DataFrame)
    @assert nrow(runs) == expected_runs
    @assert all(==("success"), string.(runs.status))
    @assert nrow(summary) == expected_summaries
    println("validated $relative_path: $(nrow(runs)) successful rows")
end
JULIA
```

Compare reproduced `config.toml` files with the corresponding committed
files before comparing metrics. Runtime-only fields such as `output_dir`,
plotting, and resume state may differ; model-, split-, stopping-, likelihood-,
and budget-related fields should not.

## Git LFS

All `.jld2` files below `paper_materials/bbb_uci` and
`paper_materials/dvi_uci` are tracked by Git LFS through `.gitattributes`.
CSV, TOML, Markdown, and TeX files remain ordinary Git files. To add a new
reproduction to the repository safely:

```sh
git add .gitattributes paper_materials
git lfs ls-files
git check-attr filter -- paper_materials/path/to/checkpoint.jld2
git status
```

The final `git check-attr` command must report `filter: lfs`, and staged
`.jld2` blobs should appear in `git lfs ls-files` before committing.
