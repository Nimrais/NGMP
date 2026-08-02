#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
output_root="${DVI_PAPER_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_robust_v1_20260731}"
split_groups=(
    "1,5,9,13,17"
    "2,6,10,14,18"
    "3,7,11,15,19"
    "4,8,12,16,20"
)
pids=()

cleanup() {
    if ((${#pids[@]})); then
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

run_method() {
    local propagation="$1"
    local method_directory
    if [[ "$propagation" == "full" ]]; then
        method_directory="dvi_shards"
    else
        method_directory="ddvi_shards"
    fi
    pids=()
    for index in "${!split_groups[@]}"; do
        shard_id="$(printf '%02d' "$((index + 1))")"
        env \
            DATADEPS_ALWAYS_ACCEPT=true \
            OPENBLAS_NUM_THREADS=1 \
            DVI_PROPAGATION="$propagation" \
            DVI_LIKELIHOODS=heteroscedastic \
            DVI_SPLITS=20 \
            DVI_SPLIT_IDS="${split_groups[$index]}" \
            DVI_MAX_EPOCHS=10000 \
            DVI_MIN_EPOCHS=25 \
            DVI_VALIDATION_EVERY=5 \
            DVI_PATIENCE=500 \
            DVI_LEARNING_RATE=0.0003 \
            DVI_NUMERICAL_PROTOCOL=bounded-exp-step-kl-v1 \
            DVI_SAFE_EXP_MIN=-20 \
            DVI_SAFE_EXP_MAX=20 \
            DVI_LOG_VARIANCE_HEAD_POSTERIOR_STD=0.05 \
            DVI_LOG_VARIANCE_HEAD_INITIAL_BIAS=0 \
            DVI_KL_SCHEDULE_UNIT=steps \
            DVI_KL_WARMUP_STEPS=14000 \
            DVI_KL_ANNEAL_STEPS=1000 \
            DVI_OUTPUT_DIR="$output_root/$method_directory/$shard_id" \
            DVI_SELECTION_ONLY=false \
            DVI_MAKE_PLOT=false \
            DVI_SHOW_PROGRESS=false \
            julia --project="$script_dir" "$script_dir/run.jl" &
        pids+=("$!")
    done
    local status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    ((status == 0)) || return "$status"
    julia --project="$script_dir" "$script_dir/merge_shards.jl" \
        "$output_root/${method_directory%_shards}" \
        "$output_root/$method_directory/"*
}

# Validate diagonal DVI completely before spending the full-DVI budget.
run_method diagonal
run_method full
