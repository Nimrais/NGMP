#!/usr/bin/env bash

set -euo pipefail

if [[ "${DVI_PAPER_PROFILE_APPROVED:-false}" != "true" ]]; then
    echo "Refusing paper run: set DVI_PAPER_PROFILE_APPROVED=true only after the profiler projects at most three days." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
output_root="${DVI_PAPER_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_robust_optimized_v1_20260802}"
execution_backend="${DVI_PAPER_BACKEND:-zygote}"
execution_device="${DVI_PAPER_DEVICE:-cpu}"
methods="${DVI_PAPER_METHODS:-full}"
default_shards=4
if [[ "$execution_backend" == "reactant" && "$execution_device" == "gpu" ]]; then
    default_shards=1
fi
shard_count="${DVI_PAPER_SHARDS:-$default_shards}"

case "$shard_count" in
    1)
        split_groups=("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
        ;;
    2)
        split_groups=(
            "1,3,5,7,9,11,13,15,17,19"
            "2,4,6,8,10,12,14,16,18,20"
        )
        ;;
    4)
        split_groups=(
            "1,5,9,13,17"
            "2,6,10,14,18"
            "3,7,11,15,19"
            "4,8,12,16,20"
        )
        ;;
    *)
        echo "DVI_PAPER_SHARDS must be 1, 2, or 4" >&2
        exit 2
        ;;
esac
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
            OPENBLAS_NUM_THREADS="${DVI_OPENBLAS_NUM_THREADS:-1}" \
            JULIA_NUM_THREADS="${DVI_JULIA_NUM_THREADS:-1}" \
            DVI_PROPAGATION="$propagation" \
            DVI_BACKEND="$execution_backend" \
            DVI_DEVICE="$execution_device" \
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

case ",$methods," in
    *,diagonal,*) run_method diagonal ;;
esac
case ",$methods," in
    *,full,*) run_method full ;;
esac
