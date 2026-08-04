#!/usr/bin/env bash

set -euo pipefail

if [[ "${DVI_HOMO_PROFILE_APPROVED:-false}" != "true" ]]; then
    echo "Refusing homoscedastic paper run: approve the Power split-1 quality and four-shard timing gates with DVI_HOMO_PROFILE_APPROVED=true." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
output_root="${DVI_HOMO_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_homoscedastic_budget50k_v1_20260804}"
execution_backend="${DVI_HOMO_BACKEND:-reactant}"
execution_device="${DVI_HOMO_DEVICE:-gpu}"
shard_count="${DVI_HOMO_SHARDS:-4}"
methods="${DVI_HOMO_METHODS:-diagonal,full}"

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
        echo "DVI_HOMO_SHARDS must be 1, 2, or 4" >&2
        exit 2
        ;;
esac

IFS=',' read -r -a method_list <<< "$methods"
declare -A observed_methods=()
for method in "${method_list[@]}"; do
    if [[ "$method" != "diagonal" && "$method" != "full" ]]; then
        echo "DVI_HOMO_METHODS may contain only diagonal and full" >&2
        exit 2
    fi
    if [[ -n "${observed_methods[$method]:-}" ]]; then
        echo "DVI_HOMO_METHODS contains duplicate method: $method" >&2
        exit 2
    fi
    observed_methods[$method]=true
done
((${#method_list[@]} > 0)) || {
    echo "DVI_HOMO_METHODS must select at least one method" >&2
    exit 2
}

pids=()
cleanup() {
    if ((${#pids[@]})); then
        kill -INT "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

run_method() {
    local propagation="$1"
    local method_directory="dvi_shards"
    [[ "$propagation" == "diagonal" ]] && method_directory="ddvi_shards"

    pids=()
    for index in "${!split_groups[@]}"; do
        local shard_id
        shard_id="$(printf '%02d' "$((index + 1))")"
        env \
            DATADEPS_ALWAYS_ACCEPT=true \
            OPENBLAS_NUM_THREADS="${DVI_OPENBLAS_NUM_THREADS:-1}" \
            JULIA_NUM_THREADS="${DVI_JULIA_NUM_THREADS:-1}" \
            XLA_REACTANT_GPU_MEM_FRACTION="${DVI_XLA_GPU_MEM_FRACTION:-0.22}" \
            XLA_REACTANT_GPU_PREALLOCATE="${DVI_XLA_GPU_PREALLOCATE:-false}" \
            DVI_PROPAGATION="$propagation" \
            DVI_BACKEND="$execution_backend" \
            DVI_DEVICE="$execution_device" \
            DVI_LIKELIHOODS=homoscedastic \
            DVI_SPLITS=20 \
            DVI_SPLIT_IDS="${split_groups[$index]}" \
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

    local merged_directory="${method_directory%_shards}"
    julia --project="$script_dir" "$script_dir/merge_shards.jl" \
        "$output_root/$merged_directory" \
        "$output_root/$method_directory/"*
}

for method in "${method_list[@]}"; do
    run_method "$method"
done

