#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
output_root="$repo_dir/paper_materials/dvi_uci/wu2019_repeated_holdout_v1_20splits_heteroscedastic_2hidden50_v1_20261024"
julia_bin="${DDVI_JULIA_BIN:-julia}"
export JULIA_DEPOT_PATH="${DDVI_DEPOT_PATH:-$repo_dir/cache/ddvi_depot:$HOME/.julia:}"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DATADEPS_ALWAYS_ACCEPT=true
export DATADEPS_LOAD_PATH="$repo_dir/cache/ddvi_depot/datadeps${DATADEPS_LOAD_PATH:+:$DATADEPS_LOAD_PATH}"
mkdir -p "$output_root/logs" "$output_root/provenance" "$repo_dir/cache/ddvi_depot"
exec 9>"$output_root/campaign.lock"
flock -n 9 || { echo "This campaign is already running." >&2; exit 1; }
cd "$repo_dir"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_seconds=$SECONDS
pids=()
write_status() {
    local state="$1"
    printf 'status = "%s"\nstarted_utc = "%s"\nupdated_utc = "%s"\nelapsed_seconds = %s\n' \
        "$state" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$((SECONDS-start_seconds))" \
        > "$output_root/status.toml"
}
cleanup() {
    local status=$?
    if ((status != 0)); then
        if ((${#pids[@]})); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
        write_status failed
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status preflight
snapshot="$output_root/provenance/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$snapshot"
git rev-parse HEAD > "$snapshot/git_revision.txt"
git diff -- baselines/dvi paper_materials/make_final_baseline_table.jl > "$snapshot/source.diff"
cp Project.toml Manifest.toml "$snapshot/"
cp baselines/dvi/Project.toml "$snapshot/dvi_Project.toml"
if [[ -f baselines/dvi/Manifest.toml ]]; then cp baselines/dvi/Manifest.toml "$snapshot/dvi_Manifest.toml"; fi
tar -czf "$snapshot/source.tar.gz" --exclude=Manifest.toml \
    baselines/dvi src Project.toml paper_materials/make_final_baseline_table.jl
"$julia_bin" --startup-file=no --version > "$snapshot/julia_version.txt"
if [[ -d cache/ddvi_depot/datadeps ]]; then
    find cache/ddvi_depot/datadeps -name data.txt -type f -exec sha256sum {} + > "$snapshot/dataset_sha256.txt"
fi
"$julia_bin" --startup-file=no --project="$script_dir" \
    "$script_dir/two_hidden_experiment.jl" preflight 2>&1 | tee "$output_root/logs/preflight.log"
write_status training
split_groups=("1,5,9,13,17" "2,6,10,14,18" "3,7,11,15,19" "4,8,12,16,20")
for index in 0 1 2 3; do
    shard_id=$(printf '%02d' "$((index+1))")
    "$julia_bin" --startup-file=no --project="$script_dir" \
        "$script_dir/two_hidden_experiment.jl" run "$shard_id" "${split_groups[$index]}" \
        >> "$output_root/logs/shard_${shard_id}.log" 2>&1 &
    pids+=("$!")
    echo "Started shard $shard_id, PID $!, splits ${split_groups[$index]}"
done
printf '%s\n' "${pids[@]}" > "$output_root/worker_pids.txt"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
((failed == 0)) || { echo "A worker failed; see shard logs. Rerun to resume completed fits." >&2; exit 1; }
write_status merging
"$julia_bin" --startup-file=no --project="$script_dir" "$script_dir/merge_shards.jl" \
    "$output_root/ddvi" "$output_root/ddvi_shards/01" "$output_root/ddvi_shards/02" \
    "$output_root/ddvi_shards/03" "$output_root/ddvi_shards/04" 2>&1 | tee "$output_root/logs/merge.log"
write_status verifying
"$julia_bin" --startup-file=no --project="$script_dir" "$script_dir/two_hidden_experiment.jl" finish \
    2>&1 | tee "$output_root/logs/finish.log"
write_status complete
echo "Campaign complete: $output_root"
