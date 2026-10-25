#!/usr/bin/env bash
set -euo pipefail
method="${1:-}"
mode="${2:-}"
[[ "$method" == ddvi || "$method" == dvi ]] || { echo "Expected ddvi or dvi." >&2; exit 2; }
case "$mode" in check|smoke|run|finish) ;; *) echo "Usage: run_${method}_homoscedastic_two_hidden_paper.sh check|smoke|run|finish" >&2; exit 2;; esac
[[ $# == 2 ]] || { echo "Unexpected arguments." >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
export DVI_HOMO_BIG_OUTPUT_ROOT="${DVI_HOMO_BIG_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_homoscedastic_2hidden50_budget50k_v1_20261025}"
output_dir="$DVI_HOMO_BIG_OUTPUT_ROOT/$method"
julia_bin="${DVI_HOMO_BIG_JULIA_BIN:-}"
if [[ -z "$julia_bin" ]]; then
    for candidate in "$HOME"/.julia/juliaup/julia-1.11.9*/bin/julia; do
        if [[ -x "$candidate" ]]; then julia_bin="$candidate"; break; fi
    done
    julia_bin="${julia_bin:-julia}"
fi
export JULIA_DEPOT_PATH="${DVI_HOMO_BIG_DEPOT_PATH:-$repo_dir/cache/ddvi_depot:$HOME/.julia:}"
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DATADEPS_ALWAYS_ACCEPT=true
export DATADEPS_LOAD_PATH="$repo_dir/cache/ddvi_depot/datadeps${DATADEPS_LOAD_PATH:+:$DATADEPS_LOAD_PATH}"
export DVI_HOMO_BIG_BACKEND="${DVI_HOMO_BIG_BACKEND:-reactant}"
export DVI_HOMO_BIG_DEVICE="${DVI_HOMO_BIG_DEVICE:-gpu}"
export CUDA_VISIBLE_DEVICES="${DVI_HOMO_BIG_GPU:-0}"
export XLA_REACTANT_GPU_PREALLOCATE=false XLA_REACTANT_GPU_MEM_FRACTION=0.40
workers="${DVI_HOMO_BIG_WORKERS:-2}"
[[ "$workers" =~ ^[124]$ ]] || { echo "DVI_HOMO_BIG_WORKERS must be 1, 2, or 4." >&2; exit 2; }
if [[ "$DVI_HOMO_BIG_DEVICE" == gpu ]] && ((workers > 2)); then
    echo "Use one or two GPU workers." >&2; exit 2
fi
version="$("$julia_bin" --startup-file=no -e 'print(VERSION)')"
[[ "$version" == 1.11.9 ]] || { echo "This experiment requires Julia 1.11.9; selected executable reports $version." >&2; exit 2; }
echo "Using Julia $version: $julia_bin"
mkdir -p "$output_dir/logs"
exec 9>"$output_dir/campaign.lock"
flock -n 9 || { echo "$method homoscedastic big is already running." >&2; exit 1; }
experiment() { "$julia_bin" --startup-file=no --project="$script_dir" "$script_dir/homoscedastic_two_hidden_experiment.jl" "$method" "$@"; }
pids=()
training_started=0
cleanup() {
    local result=$?
    if ((${#pids[@]})); then
        kill -TERM "${pids[@]}" 2>/dev/null || true
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
    if ((result != 0 && training_started)); then
        printf 'status = "failed"\n' > "$output_dir/status.toml"
    fi
    return "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$mode" == finish ]]; then experiment finish; exit 0; fi
if [[ "$mode" == run && -f "$output_dir/runs.csv" ]]; then
    if experiment check-complete; then experiment finish; exit 0; fi
    [[ -d "$DVI_HOMO_BIG_OUTPUT_ROOT/${method}_shards" ]] || {
        echo "Incomplete canonical bundle: restore missing artifacts or local shards before retrying." >&2; exit 1;
    }
fi
if [[ "$DVI_HOMO_BIG_DEVICE" == gpu ]]; then
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$output_dir/logs/gpu.txt"
fi
if [[ "$mode" == check || "$mode" == smoke ]]; then
    experiment "$mode" 2>&1 | tee "$output_dir/logs/$mode.log"
    exit 0
fi
experiment prepare "$workers" 2>&1 | tee -a "$output_dir/logs/prepare.log"
snapshot="$output_dir/provenance/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$snapshot"
git rev-parse HEAD > "$snapshot/git_revision.txt"
git diff HEAD -- baselines/dvi paper_materials/make_final_baseline_table.jl > "$snapshot/source.diff"
cp Project.toml Manifest.toml "$snapshot/"
cp "$script_dir/Project.toml" "$snapshot/dvi_Project.toml"
cp "$script_dir/Manifest.toml" "$snapshot/dvi_Manifest.toml"
tar -czf "$snapshot/source.tar.gz" --exclude=Manifest.toml baselines/dvi src Project.toml paper_materials/make_final_baseline_table.jl
"$julia_bin" --startup-file=no --version > "$snapshot/julia_version.txt"
find "$repo_dir/cache/ddvi_depot/datadeps" -name data.txt -type f -exec sha256sum {} + > "$snapshot/dataset_sha256.txt"
printf 'status = "training"\nworkers = %s\n' "$workers" > "$output_dir/status.toml"
training_started=1
shards=()
for ((i=1;i<=workers;i++)); do
    printf -v id '%02d' "$i"
    splits="$(seq -s, "$i" "$workers" 20)"
    shards+=("$DVI_HOMO_BIG_OUTPUT_ROOT/${method}_shards/$id")
    "$julia_bin" --startup-file=no --project="$script_dir" "$script_dir/homoscedastic_two_hidden_experiment.jl" \
        "$method" run "$id" "$splits" >> "$output_dir/logs/shard_$id.log" 2>&1 &
    pids+=("$!")
    echo "Started $method worker $id, PID $!, splits $splits"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
((failed == 0)) || { echo "A worker failed; inspect shard logs and rerun with unchanged settings to resume." >&2; exit 1; }
"$julia_bin" --startup-file=no --project="$script_dir" "$script_dir/merge_shards.jl" \
    "$output_dir" "${shards[@]}" > "$output_dir/logs/merge.log" 2>&1
experiment finish > "$output_dir/logs/finish.log" 2>&1
echo "Completed $method homoscedastic big: $output_dir"
