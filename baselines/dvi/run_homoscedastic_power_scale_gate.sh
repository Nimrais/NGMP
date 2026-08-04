#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
propagation="${DVI_HOMO_GATE_METHOD:-diagonal}"
if [[ "$propagation" != "diagonal" && "$propagation" != "full" ]]; then
    echo "DVI_HOMO_GATE_METHOD must be diagonal or full" >&2
    exit 2
fi

output_root="${DVI_HOMO_GATE_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804/pilots/power_four_shard/$propagation}"
if [[ -e "$output_root" ]]; then
    echo "Refusing timing gate because output already exists: $output_root" >&2
    echo "Set a fresh DVI_HOMO_GATE_OUTPUT_ROOT for a valid wall-time measurement." >&2
    exit 2
fi
split_ids=(2 3 4 5)
pids=()

cleanup() {
    if ((${#pids[@]})); then
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

start_seconds=$SECONDS
for index in "${!split_ids[@]}"; do
    shard_id="$(printf '%02d' "$((index + 1))")"
    env \
        DATADEPS_ALWAYS_ACCEPT=true \
        OPENBLAS_NUM_THREADS=1 \
        JULIA_NUM_THREADS=1 \
        XLA_REACTANT_GPU_MEM_FRACTION="${DVI_XLA_GPU_MEM_FRACTION:-0.22}" \
        XLA_REACTANT_GPU_PREALLOCATE="${DVI_XLA_GPU_PREALLOCATE:-false}" \
        DVI_DATASETS=power \
        DVI_SPLITS=20 \
        DVI_SPLIT_IDS="${split_ids[$index]}" \
        DVI_PROPAGATION="$propagation" \
        DVI_BACKEND=reactant \
        DVI_DEVICE=gpu \
        DVI_LIKELIHOODS=homoscedastic \
        DVI_MAX_EPOCHS=10000 \
        DVI_MIN_EPOCHS=25 \
        DVI_VALIDATION_EVERY=5 \
        DVI_PATIENCE=500 \
        DVI_FULL_PATIENCE_STEPS=50000 \
        DVI_SELECTION_MAX_STEPS=50000 \
        DVI_REFIT_MAX_STEPS=50000 \
        DVI_TRAINING_BUDGET_PROTOCOL=selection-refit-max-50000-v1 \
        DVI_LEARNING_RATE=0.0003 \
        DVI_NUMERICAL_PROTOCOL=bounded-exp-step-kl-v1 \
        DVI_SAFE_EXP_MIN=-20 \
        DVI_SAFE_EXP_MAX=20 \
        DVI_HOMO_LOG_VARIANCE=0 \
        DVI_KL_SCHEDULE_UNIT=steps \
        DVI_KL_WARMUP_STEPS=14000 \
        DVI_KL_ANNEAL_STEPS=1000 \
        DVI_OUTPUT_DIR="$output_root/$shard_id" \
        DVI_RESUME=false \
        DVI_SELECTION_ONLY=false \
        DVI_MAKE_PLOT=false \
        DVI_SHOW_PROGRESS=false \
        julia --project="$script_dir" "$script_dir/run.jl" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
elapsed_seconds=$((SECONDS - start_seconds))

if ((status != 0)); then
    echo "Four-shard homoscedastic Power gate failed after ${elapsed_seconds}s" >&2
    exit 1
fi

pilot_workers=${#split_ids[@]}
configurations_per_method=120
paper_methods=2
target_seconds=$((10 * 60 * 60))
projected_single_method_seconds=$((
    elapsed_seconds * configurations_per_method / pilot_workers
))
projected_both_methods_seconds=$((
    projected_single_method_seconds * paper_methods
))

single_hours=$((projected_single_method_seconds / 3600))
single_minutes=$(((projected_single_method_seconds % 3600) / 60))
both_hours=$((projected_both_methods_seconds / 3600))
both_minutes=$(((projected_both_methods_seconds % 3600) / 60))
printf 'Four-shard Power gate completed in %ss\n' "$elapsed_seconds"
printf 'Projected one-method campaign: %dh %02dm\n' \
    "$single_hours" "$single_minutes"
printf 'Projected two-method campaign: %dh %02dm\n' \
    "$both_hours" "$both_minutes"

if ((projected_both_methods_seconds > target_seconds)); then
    echo "Runtime gate FAIL: projected DVI+dDVI campaign exceeds 10 hours" >&2
    exit 3
fi

echo "Runtime gate PASS: projected DVI+dDVI campaign is at most 10 hours"
