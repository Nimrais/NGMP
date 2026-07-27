#!/usr/bin/env bash

set -uo pipefail

# ResidualSine Gaussian-surrogate comparison.
#
# Default matrix:
#   2 datasets × 2 depths × 5 optimizers × 3 widths = 60 runs.
# The default "full" split is 40k train / 10k validation / 10k test.
#
# This runner does not include mnist_manyplus_residual_sine_precision_classifier.jl:
# that literal learned-precision RxInfer graph is experimental and currently has
# no complete ReactiveMP update schedule.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/residual_sine_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/logs"
SUMMARY="${OUTPUT_DIR}/summary.tsv"
STATUS="${OUTPUT_DIR}/run_status.tsv"

JULIA_BIN="${JULIA_BIN:-julia}"
DATASETS="${DATASETS:-mnist fashion_mnist}"
MODELS="${MODELS:-residual_1h residual_2h}"
OPTIMIZERS="${OPTIMIZERS:-damped vector_transport_05 vector_transport_08 vector_transport_nesterov_05 vector_transport_nesterov_08}"
HIDDEN_COUNTS="${HIDDEN_COUNTS:-32 64 128}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-direct}"
NTRAIN="${NTRAIN:-40000}"
NVAL="${NVAL:-10000}"
NTEST="${NTEST:-10000}"
SPLIT_SAMPLING="${SPLIT_SAMPLING:-auto}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_INNER="${MAX_INNER:-3}"
SEED="${SEED:-1}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-10000}"
RESIDUAL_SINE_RHO="${RESIDUAL_SINE_RHO:-0.9}"
RESIDUAL_SINE_OMEGA="${RESIDUAL_SINE_OMEGA:-1.0}"

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' \
        'Run ResidualSine flattened-MNIST optimizer comparisons.' \
        '' \
        'Environment variables:' \
        '  DATASETS="mnist fashion_mnist"' \
        '  MODELS="residual_1h residual_2h"' \
        '  OPTIMIZERS="damped vector_transport_05 vector_transport_08 vector_transport_nesterov_05 vector_transport_nesterov_08"' \
        '  HIDDEN_COUNTS="32 64 128"' \
        '  INFERENCE_BACKEND=direct|rxinfer' \
        '  NTRAIN=40000 NVAL=10000 NTEST=10000 SPLIT_SAMPLING=auto' \
        '  auto uses natural sampling for the full 10k test set and balanced sampling for smaller subsets' \
        '  EPOCHS=50 BATCH_SIZE=32 MAX_INNER=3' \
        '' \
        'Quick example:' \
        '  DATASETS=mnist HIDDEN_COUNTS=32 EPOCHS=2 NTRAIN=1000 NVAL=100 NTEST=100 ./mnist_experiments/run_residual_sine_mnist_sweep.sh'
    exit 0
elif (($#)); then
    printf 'Unknown argument: %s\n' "$1" >&2
    exit 2
fi

case "${INFERENCE_BACKEND}" in
    direct|rxinfer) ;;
    *)
        printf 'INFERENCE_BACKEND must be direct or rxinfer, got %s\n' \
            "${INFERENCE_BACKEND}" >&2
        exit 2
        ;;
esac

case "${SPLIT_SAMPLING}" in
    auto|natural|balanced) ;;
    *)
        printf 'SPLIT_SAMPLING must be auto, natural, or balanced, got %s\n' \
            "${SPLIT_SAMPLING}" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}" "${REPO_DIR}/.data"
export DATADEPS_LOAD_PATH="${DATADEPS_LOAD_PATH:-${REPO_DIR}/.data}"
export DATADEPS_ALWAYS_ACCEPT="${DATADEPS_ALWAYS_ACCEPT:-true}"

printf 'dataset\tmodel\toptimizer\tbackend\thidden_count\tbest_val_epoch\tbest_val_acc\ttest_acc\telapsed_seconds\tlog\n' > "${SUMMARY}"
printf 'dataset\tmodel\toptimizer\tbackend\thidden_count\tstatus\texit_code\telapsed_seconds\tlog\n' > "${STATUS}"

optimizer_arguments() {
    case "$1" in
        damped)
            printf 'alpha=0.5, projected_nesterov=false, vector_transport=false'
            ;;
        vector_transport_05)
            printf 'alpha=0.2, projected_nesterov=false, vector_transport=true, vector_transport_momentum=0.5'
            ;;
        vector_transport_08)
            printf 'alpha=0.2, projected_nesterov=false, vector_transport=true, vector_transport_momentum=0.8'
            ;;
        vector_transport_nesterov_05)
            printf 'alpha=0.2, projected_nesterov=false, vector_transport=false, vector_transport_nesterov=true, vector_transport_momentum=0.5'
            ;;
        vector_transport_nesterov_08)
            printf 'alpha=0.2, projected_nesterov=false, vector_transport=false, vector_transport_nesterov=true, vector_transport_momentum=0.8'
            ;;
        *)
            printf 'Unknown optimizer: %s\n' "$1" >&2
            return 2
            ;;
    esac
}

run_one() {
    local dataset="$1" model="$2" optimizer="$3" width="$4"
    local source_file function_name hidden_args optimizer_args
    local effective_split_sampling
    local run_name log_file start_time elapsed exit_code final_line
    local best_epoch best_val test_acc run_status

    case "${model}" in
        residual_1h)
            source_file="mnist_residual_sine_rxinfer_mlp_flattened.jl"
            function_name="train_residual_sine_mlp_flattened"
            hidden_args="hidden_count=${width}"
            ;;
        residual_2h)
            source_file="mnist_residual_sine_rxinfer_mlp_two_hidden.jl"
            function_name="train_residual_sine_mlp_two_hidden"
            hidden_args="hidden1_count=${width}, hidden2_count=${width}"
            ;;
        *)
            printf 'Unknown model: %s\n' "${model}" >&2
            return 2
            ;;
    esac
    optimizer_args="$(optimizer_arguments "${optimizer}")" || return $?
    if [[ "${SPLIT_SAMPLING}" == "auto" ]]; then
        if [[ "${NTEST}" == "10000" ]]; then
            effective_split_sampling=natural
        else
            effective_split_sampling=balanced
        fi
    else
        effective_split_sampling="${SPLIT_SAMPLING}"
    fi
    run_name="${dataset}_${model}_${optimizer}_h${width}_${INFERENCE_BACKEND}"
    log_file="${LOG_DIR}/${run_name}.log"
    start_time="$(date +%s)"

    printf 'starting %s split_sampling=%s\n' \
        "${run_name}" "${effective_split_sampling}"
    (
        cd "${REPO_DIR}" || exit
        "${JULIA_BIN}" --project=. -e "
            include(joinpath(\"mnist_experiments\", \"${source_file}\"))
            ${function_name}(
                dataset=:${dataset},
                ntrain=${NTRAIN}, nval=${NVAL}, ntest=${NTEST},
                split_sampling=:${effective_split_sampling},
                ${hidden_args},
                batch_size=${BATCH_SIZE}, epochs=${EPOCHS},
                seed=${SEED}, max_inner=${MAX_INNER},
                eval_max_images=${EVAL_MAX_IMAGES},
                inference_backend=:${INFERENCE_BACKEND},
                residual_sine_rho=${RESIDUAL_SINE_RHO},
                residual_sine_omega=${RESIDUAL_SINE_OMEGA},
                ${optimizer_args},
            )
        "
    ) 2>&1 | tee "${log_file}"
    exit_code="${PIPESTATUS[0]}"
    elapsed="$(( $(date +%s) - start_time ))"

    if ((exit_code == 0)); then
        run_status=passed
        final_line="$(sed -n 's/.*best_val_epoch=/best_val_epoch=/p' "${log_file}" | tail -n 1)"
        best_epoch="$(sed -n 's/.*best_val_epoch=\\([^ ]*\\).*/\\1/p' <<< "${final_line}")"
        best_val="$(sed -n 's/.*best_val_acc=\\([^ ]*\\).*/\\1/p' <<< "${final_line}")"
        test_acc="$(sed -n 's/.*test_acc=\\([^ ]*\\).*/\\1/p' <<< "${final_line}")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${dataset}" "${model}" "${optimizer}" "${INFERENCE_BACKEND}" \
            "${width}" "${best_epoch}" "${best_val}" "${test_acc}" \
            "${elapsed}" "${log_file}" >> "${SUMMARY}"
    else
        run_status=failed
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${dataset}" "${model}" "${optimizer}" "${INFERENCE_BACKEND}" \
        "${width}" "${run_status}" "${exit_code}" "${elapsed}" \
        "${log_file}" >> "${STATUS}"
    return "${exit_code}"
}

failure_count=0
for dataset in ${DATASETS}; do
    for model in ${MODELS}; do
        for optimizer in ${OPTIMIZERS}; do
            for width in ${HIDDEN_COUNTS}; do
                run_one "${dataset}" "${model}" "${optimizer}" "${width}" ||
                    failure_count=$((failure_count + 1))
            done
        done
    done
done

printf 'summary=%s\nstatus=%s\nfailed_runs=%s\n' \
    "${SUMMARY}" "${STATUS}" "${failure_count}"
((failure_count == 0))
