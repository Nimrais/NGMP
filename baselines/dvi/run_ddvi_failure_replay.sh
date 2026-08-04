#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
output_root="${DVI_REPLAY_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731/selection_replay}"
pids=()

cleanup() {
    if ((${#pids[@]})); then
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

run_replay() {
    local dataset="$1"
    local split_ids="$2"
    local output_dir="$output_root/$dataset"
    env \
        DATADEPS_ALWAYS_ACCEPT=true \
        OPENBLAS_NUM_THREADS=1 \
        DVI_DATASETS="$dataset" \
        DVI_PROPAGATION=diagonal \
        DVI_LIKELIHOODS=heteroscedastic \
        DVI_SPLITS=20 \
        DVI_SPLIT_IDS="$split_ids" \
        DVI_LEARNING_RATE=0.0003 \
        DVI_MAX_EPOCHS=10000 \
        DVI_MIN_EPOCHS=25 \
        DVI_VALIDATION_EVERY=5 \
        DVI_PATIENCE=500 \
        DVI_NUMERICAL_PROTOCOL=bounded-exp-step-kl-v1 \
        DVI_SAFE_EXP_MIN=-20 \
        DVI_SAFE_EXP_MAX=20 \
        DVI_LOG_VARIANCE_HEAD_POSTERIOR_STD=0.05 \
        DVI_LOG_VARIANCE_HEAD_INITIAL_BIAS=0 \
        DVI_KL_SCHEDULE_UNIT=steps \
        DVI_KL_WARMUP_STEPS=14000 \
        DVI_KL_ANNEAL_STEPS=1000 \
        DVI_SELECTION_ONLY=true \
        DVI_SAVE_CHECKPOINTS=false \
        DVI_OUTPUT_DIR="$output_dir" \
        DVI_MAKE_PLOT=false \
        DVI_SHOW_PROGRESS=false \
        julia --project="$script_dir" "$script_dir/run.jl" &
    pids+=("$!")
}

# Nineteen stopped-run failures plus successful Concrete and Yacht anchors.
run_replay concrete "2,3,4,5,7,8,11,12,13,14,15,16,17,19,20"
run_replay housing "3,7,11,15,19"
run_replay power "3"
run_replay yacht "1"

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
((status == 0)) || exit "$status"

mapfile -d '' run_tables < <(find "$output_root" -name runs.csv -print0)
((${#run_tables[@]} == 4)) || {
    echo "expected four replay run tables" >&2
    exit 1
}
if rg -q ',failure,' "${run_tables[@]}"; then
    echo "one or more dDVI selection replays failed" >&2
    exit 1
fi
successes="$(rg --no-filename ',selection_success,' "${run_tables[@]}" | wc -l)"
[[ "$successes" -eq 22 ]] || {
    echo "expected 22 successful selection replays, found $successes" >&2
    exit 1
}
echo "Validated 19 prior failures and 3 successful anchors in $output_root"
