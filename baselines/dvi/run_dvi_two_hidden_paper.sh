#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
export DVI_BIG_OUTPUT_ROOT="${DVI_BIG_OUTPUT_ROOT:-$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_full_heteroscedastic_2hidden50_v1_20261025}"
output_root="$DVI_BIG_OUTPUT_ROOT"
julia_bin="${DVI_BIG_JULIA_BIN:-}"
if [[ -z "$julia_bin" ]]; then
    # Prefer the tested installation over Juliaup's unrelated default channel.
    for candidate in "$HOME"/.julia/juliaup/julia-1.11.9*/bin/julia; do
        if [[ -x "$candidate" ]]; then julia_bin="$candidate"; break; fi
    done
    julia_bin="${julia_bin:-julia}"
fi
export JULIA_DEPOT_PATH="${DVI_BIG_DEPOT_PATH:-$repo_dir/cache/ddvi_depot:$HOME/.julia:}"
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DATADEPS_ALWAYS_ACCEPT=true
export DATADEPS_LOAD_PATH="$repo_dir/cache/ddvi_depot/datadeps${DATADEPS_LOAD_PATH:+:$DATADEPS_LOAD_PATH}"
export CUDA_VISIBLE_DEVICES="${DVI_BIG_GPU:-0}"
export XLA_REACTANT_GPU_PREALLOCATE=false
export XLA_REACTANT_GPU_MEM_FRACTION=0.40
mode="${1:-}"
case "$mode" in check|pilot|run|finish|_pilot) ;; *) echo "Usage: $0 check|pilot|run|finish" >&2; exit 2;; esac
if ! julia_version="$("$julia_bin" --startup-file=no -e 'print(VERSION)')"; then
    echo "Cannot start Julia. Set DVI_BIG_JULIA_BIN to a Julia 1.11.9 executable." >&2
    exit 2
fi
if [[ "$julia_version" != 1.11.9 ]]; then
    echo "This experiment requires the tested Julia 1.11.9; '$julia_bin' reports $julia_version." >&2
    echo "Set DVI_BIG_JULIA_BIN to its executable, or install Julia 1.11.9 before retrying." >&2
    exit 2
fi
echo "Using Julia $julia_version: $julia_bin" >&2
mkdir -p "$output_root/logs" "$output_root/pilots"
if [[ "$mode" != _pilot ]]; then
    exec 9>"$output_root/campaign.lock"
    flock -n 9 || { echo "DVI big is already running." >&2; exit 1; }
else
    [[ "${DVI_BIG_PILOT_CHILD:-}" == 1 ]] || { echo "Use pilot, not _pilot." >&2; exit 2; }
fi
jl() { "$julia_bin" --startup-file=no --project="$script_dir" "$@"; }
experiment() { jl "$script_dir/full_two_hidden_experiment.jl" "$@"; }
if [[ "$mode" == check ]]; then
    experiment gate
    echo "Package loading and saved pilot checks passed. No training started."
    exit 0
fi
pids=()
monitor_pid=""
training_started=0
cleanup() {
    local result=$?
    if [[ -n "$monitor_pid" ]]; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if ((${#pids[@]})); then
        kill -TERM "${pids[@]}" 2>/dev/null || true
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
    if ((result != 0 && training_started)); then
        printf 'status = "failed"\n' > "$output_root/status.toml"
    fi
    return "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
require_gpu() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$output_root/logs/gpu.txt"
}
throughput_group() {
    local backend="$1" device="$2" group="$3" count="$4"
    local directory="$output_root/pilots/$group"
    mkdir -p "$directory"
    rm -f "$directory"/ready_* "$directory"/worker_*.toml "$directory/go"
    pids=()
    for ((i=1;i<=count;i++)); do
        experiment throughput "$backend" "$device" "$group" "$i" > "$directory/worker_$i.log" 2>&1 &
        pids+=("$!")
    done
    local deadline=$((SECONDS + 600)) ready=0
    while ((ready < count)); do
        ready=0
        for ((i=1;i<=count;i++)); do
            if [[ -f "$directory/ready_$i" ]]; then ready=$((ready+1)); fi
            kill -0 "${pids[$((i-1))]}" 2>/dev/null || { echo "Throughput worker failed: $directory/worker_$i.log" >&2; return 1; }
        done
        ((SECONDS < deadline)) || { echo "Throughput compilation timed out." >&2; return 1; }
        sleep 0.2
    done
    date -u -d '+1 second' +%s.%N > "$directory/go"
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
}
if [[ "$mode" == pilot ]]; then
    require_gpu
    rm -f "$output_root/pilots/recommendation.toml"
    budget="${DVI_BIG_PILOT_SECONDS:-1800}"
    [[ "$budget" =~ ^[0-9]+$ ]] && ((budget >= 1 && budget <= 1800)) || { echo "Pilot budget must be 1–1800 seconds." >&2; exit 2; }
    # The user's budget is GPU time. Run the slower CPU reference outside
    # the GPU timeout, and keep all CUDA work within the bounded child.
    experiment profile zygote cpu > "$output_root/logs/cpu_profile.log" 2>&1
    throughput_group zygote cpu cpu_four 4
    export DVI_BIG_PILOT_CHILD=1
    if timeout --signal=TERM --kill-after=20s "${budget}s" bash "$0" _pilot > "$output_root/logs/pilot.log" 2>&1; then
        experiment report > "$output_root/logs/pilot_report.log" 2>&1
        cat "$output_root/pilots/report.md"
    else
        result=$?
        rm -f "$output_root/pilots/recommendation.toml"
        echo "Pilot failed or exceeded its time budget (exit $result). Full run remains blocked. See $output_root/logs/pilot.log" >&2
        exit "$result"
    fi
    exit 0
fi
if [[ "$mode" == _pilot ]]; then
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$output_root/logs/pilot_gpu_memory.csv" &
    monitor_pid=$!
    # Reuse a successful parity check only for exactly the same numerical
    # sources, package environment, Julia executable and GPU/driver.
    parity_key="$(
        sha256sum "$script_dir"/src/*.jl "$script_dir/test/full_two_hidden_reactant.jl" "$script_dir/Project.toml"
        for manifest in "$repo_dir/Manifest.toml" "$script_dir/Manifest.toml"; do
            [[ ! -f "$manifest" ]] || sha256sum "$manifest"
        done
        "$julia_bin" --startup-file=no --version
        cat "$output_root/logs/gpu.txt"
    )"
    if [[ -f "$output_root/pilots/parity.key" ]] && [[ "$(cat "$output_root/pilots/parity.key")" == "$parity_key" ]]; then
        echo "Reusing successful GPU parity for unchanged numerical sources and environment."
    else
        DVI_DEVICE=gpu jl "$script_dir/test/full_two_hidden_reactant.jl" > "$output_root/logs/gpu_parity.log" 2>&1
        cat "$output_root/logs/gpu_parity.log"
        printf '%s\n' "$parity_key" > "$output_root/pilots/parity.key"
    fi
    experiment preflight
    experiment profile reactant gpu
    throughput_group reactant gpu gpu_one 1
    throughput_group reactant gpu gpu_two 2
    exit 0
fi
if [[ "$mode" == finish ]]; then experiment finish; exit 0; fi
# A complete merged bundle is sufficient after cloning; shard copies are optional.
if [[ -f "$output_root/dvi/runs.csv" ]]; then
    if experiment check-complete; then
        experiment finish
        exit 0
    fi
    [[ -d "$output_root/dvi_shards" ]] || {
        echo "The merged bundle is incomplete and no local shards are available. Restore its missing artifacts before retrying." >&2
        exit 1
    }
    echo "Rechecking local shards to recover an incomplete merge."
fi
read -r recommended_backend recommended_device recommended_workers < <(experiment gate)
export DVI_BIG_BACKEND="${DVI_BIG_BACKEND:-$recommended_backend}"
export DVI_BIG_DEVICE="${DVI_BIG_DEVICE:-$recommended_device}"
workers="${DVI_BIG_WORKERS:-$recommended_workers}"
[[ "$workers" =~ ^[124]$ ]] || { echo "DVI_BIG_WORKERS must be 1, 2, or 4." >&2; exit 2; }
if [[ "$DVI_BIG_DEVICE" == gpu ]]; then
    require_gpu
    ((workers <= 2)) || { echo "Only one or two GPU workers were profiled." >&2; exit 2; }
fi
experiment preflight
snapshot="$output_root/provenance/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$snapshot"
git rev-parse HEAD > "$snapshot/git_revision.txt"
git diff HEAD -- baselines/dvi paper_materials/make_final_baseline_table.jl > "$snapshot/source.diff"
cp Project.toml "$snapshot/"
cp baselines/dvi/Project.toml "$snapshot/dvi_Project.toml"
[[ ! -f Manifest.toml ]] || cp Manifest.toml "$snapshot/"
[[ ! -f baselines/dvi/Manifest.toml ]] || cp baselines/dvi/Manifest.toml "$snapshot/dvi_Manifest.toml"
tar -czf "$snapshot/source.tar.gz" --exclude=Manifest.toml baselines/dvi src Project.toml paper_materials/make_final_baseline_table.jl
"$julia_bin" --startup-file=no --version > "$snapshot/julia_version.txt"
find "$repo_dir/cache/ddvi_depot/datadeps" -name data.txt -type f -exec sha256sum {} + > "$snapshot/dataset_sha256.txt"
printf 'status = "training"\nworkers = %s\n' "$workers" > "$output_root/status.toml"
training_started=1
shards=()
for ((i=1;i<=workers;i++)); do
    printf -v id '%02d' "$i"
    splits="$(seq -s, "$i" "$workers" 20)"
    shards+=("$output_root/dvi_shards/$id")
    experiment run "$id" "$splits" >> "$output_root/logs/shard_$id.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
if ((failed)); then
    printf 'status = "failed"\n' > "$output_root/status.toml"
    echo "A worker failed; inspect shard logs and rerun with the same settings to resume." >&2
    exit 1
fi
jl "$script_dir/merge_shards.jl" "$output_root/dvi" "${shards[@]}" > "$output_root/logs/merge.log" 2>&1
experiment finish > "$output_root/logs/finish.log" 2>&1
echo "DVI big complete: $output_root"
