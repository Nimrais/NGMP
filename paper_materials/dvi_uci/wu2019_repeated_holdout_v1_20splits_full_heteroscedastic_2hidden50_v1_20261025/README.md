# Full-covariance DVI big

Architecture: `d -> 50 -> 50 -> 2`, heteroscedastic likelihood, full
activation covariance propagation. The full experiment completed on
2026-10-25 at 17:42 UTC: all 120 fits succeeded and their saved posterior
metrics passed checkpoint replay. See `status.toml` and
`dvi/checkpoint_replay.csv`.

The canonical results are in `dvi/`; `final_baseline_table.md` contains the
ten-method comparison. The campaign resumed after interruptions, and the
initial numerical failures succeeded on retry. Logs and all three source
snapshots are retained to document those attempts.

The completed pilot recommends **Reactant/GPU with two workers** on the
RTX 3090. The projected full-run scenarios are **3.49 and 5.35 hours**,
including a 25% margin; actual early stopping may differ. See
[`pilots/report.md`](pilots/report.md). CPU-only reference work ran separately
from the GPU budget; `logs/gpu_budget.toml` records the conservative GPU-time
accounting. Earlier interrupted attempts remain in `logs/pilot_attempt*.log`.

The target datasets are Concrete Compressive Strength, Energy Efficiency
(heating load), Boston Housing, Combined Cycle Power Plant, Wine Quality
Red, and Yacht Hydrodynamics. The full campaign uses 20 paired splits each
and the existing small-DVI stopping protocol, including the 50,000-step
no-improvement limit.

## Pilot artifacts

- `logs/cpu_tests.log`: numerical and artifact regression checks.
- `logs/pilot.log`: GPU parity, data checks, and short training pilots.
- `logs/pilot_gpu_memory.csv`: sampled whole-device GPU memory and utilization.
- `pilots/reactant_gpu/` and `pilots/zygote_cpu/`: measured per-dataset step
  and validation times plus short-fit checkpoints, histories, replay results,
  and configuration. These two-epoch fits use a shortened KL schedule and
  are not final experiment results.
- `pilots/gpu_one/`, `pilots/gpu_two/`, and `pilots/cpu_four/`: simultaneous
  throughput measurements on Power.
- `pilots/report.md`, `runtime_projection.csv`, and `recommendation.toml`:
  generated only after the complete pilot succeeds. The recommendation
  records a source/environment fingerprint and the measured execution layout.

## Run and reuse

From the repository root, use:

```sh
bash baselines/dvi/run_dvi_two_hidden_paper.sh finish
```

Fetch the binary artifacts with `git lfs pull` after cloning. The `finish`
command regenerates tables and verifies saved models without training.
The `run` command also detects this complete canonical bundle and verifies
it without retraining; local shard copies are not required.

For a new experiment root, use `pilot` followed by `run`. The launcher checks
the pilot fingerprint, resumes matching completed fits, and saves the full
canonical bundle under `dvi/`.

The combined table is written as `final_baseline_table.md` only after all
120 fits complete and their saved posterior metrics are verified. It includes
`DVI-2HL-Hetero`; pilot fits are never used. Paired differences against small
DVI and dDVI big are saved separately, with negative NLL/RMSE differences
favoring DVI big.

Binary checkpoints, split manifests, and compressed source snapshots use
Git LFS. Code, configurations, CSVs, and reports use regular Git. Local shard
copies, preflight artifacts, and synchronization files are excluded from Git.
