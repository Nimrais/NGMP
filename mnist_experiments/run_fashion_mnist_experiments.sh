#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/fashion_mnist_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/logs"
ALL_RUNS="${OUTPUT_DIR}/all_runs.txt"
SUMMARY="${OUTPUT_DIR}/summary.tsv"
RUN_STATUS="${OUTPUT_DIR}/run_status.tsv"
JULIA_BIN="${JULIA_BIN:-julia}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_COUNTS="${HIDDEN_COUNTS:-${HIDDEN_COUNT:-32 64 128}}"
RXINFER_OPTIMIZERS="${RXINFER_OPTIMIZERS:-vector_transport vector_transport_nesterov projected_nesterov}"
EXPERIMENT_CONFIGS="${EXPERIMENT_CONFIGS:-1000,100,100 10000,1000,1000}"
NN_LAYERS="${NN_LAYERS:-1 2 3 4 5 6 7}"

mkdir -p "${LOG_DIR}" "${REPO_DIR}/.data"
export DATADEPS_LOAD_PATH="${DATADEPS_LOAD_PATH:-${REPO_DIR}/.data}"
export DATADEPS_ALWAYS_ACCEPT="${DATADEPS_ALWAYS_ACCEPT:-true}"

: > "${ALL_RUNS}"
printf 'model\toptimizer\thidden_layers\thidden_count\tntrain\tnval\tntest\tbest_val_epoch\tbest_val_acc\ttest_acc\telapsed_seconds\tlog\n' \
    > "${SUMMARY}"
printf 'model\toptimizer\thidden_layers\thidden_count\tntrain\tnval\tntest\tstatus\texit_code\telapsed_seconds\tlog\n' \
    > "${RUN_STATUS}"

run_experiment() {
    local model="$1"
    local optimizer="$2"
    local hidden_layers="$3"
    local hidden_count="$4"
    local ntrain="$5"
    local nval="$6"
    local ntest="$7"
    local script function_call run_name log_file start_time end_time elapsed exit_code
    local final_line best_epoch best_val test_acc status
    local projected_nesterov=false vector_transport=false
    local vector_transport_nesterov=false

    case "${optimizer}" in
        projected_nesterov) projected_nesterov=true ;;
        vector_transport) vector_transport=true ;;
        vector_transport_nesterov) vector_transport_nesterov=true ;;
        standard)
            if [[ "${model}" != "neural" ]]; then
                printf 'Optimizer standard is only valid for the neural model\n' >&2
                return 2
            fi
            ;;
        *)
            printf 'Unknown optimizer: %s\n' "${optimizer}" >&2
            return 2
            ;;
    esac

    case "${model}" in
        rxinfer_1h)
            script="${SCRIPT_DIR}/mnist_softplus_rxinfer_mlp_flattened.jl"
            function_call="train_mlp_rxinfer_demo"
            ;;
        rxinfer_2h)
            script="${SCRIPT_DIR}/mnist_softplus_rxinfer_mlp_two_hidden.jl"
            function_call="train_two_hidden_mlp_rxinfer_demo"
            ;;
        neural)
            script="${SCRIPT_DIR}/mnist_softplus_nn_mlp_flattened.jl"
            function_call="train_nn_mlp_demo"
            ;;
        *)
            printf 'Unknown model: %s\n' "${model}" >&2
            return 2
            ;;
    esac

    run_name="${model}_optimizer-${optimizer}_layers-${hidden_layers}_hidden-${hidden_count}_train-${ntrain}_val-${nval}_test-${ntest}"
    log_file="${LOG_DIR}/${run_name}.txt"

    {
        printf '\n================================================================================\n'
        printf 'run=%s\n' "${run_name}"
        printf 'started_at=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
        printf 'epochs=%s batch_size=%s hidden_count=%s\n' \
            "${EPOCHS}" "${BATCH_SIZE}" "${hidden_count}"
        printf '================================================================================\n'
    } | tee -a "${ALL_RUNS}"

    start_time="$(date +%s)"
    if [[ "${model}" == "neural" ]]; then
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:fashion_mnist,
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                hidden_layers=${hidden_layers},
                hidden_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    elif [[ "${model}" == "rxinfer_1h" ]]; then
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:fashion_mnist,
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                hidden_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
                projected_nesterov=${projected_nesterov},
                vector_transport=${vector_transport},
                vector_transport_nesterov=${vector_transport_nesterov},
                max_inner=3,
                inference_backend=:direct,
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    else
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:fashion_mnist,
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                hidden1_count=${hidden_count},
                hidden2_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
                projected_nesterov=${projected_nesterov},
                vector_transport=${vector_transport},
                vector_transport_nesterov=${vector_transport_nesterov},
                max_inner=3,
                inference_backend=:direct,
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    fi
    end_time="$(date +%s)"
    elapsed="$((end_time - start_time))"

    status="failed"
    if [[ "${exit_code}" -eq 0 ]]; then
        status="completed"
        final_line="$(grep 'best_val_epoch=.*best_val_acc=.*test_acc=' "${log_file}" | tail -n 1 || true)"
        best_epoch="$(sed -n 's/.*best_val_epoch=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        best_val="$(sed -n 's/.*best_val_acc=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        test_acc="$(sed -n 's/.*test_acc=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        if [[ -n "${best_epoch}" && -n "${best_val}" && -n "${test_acc}" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${model}" "${optimizer}" "${hidden_layers}" "${hidden_count}" \
                "${ntrain}" "${nval}" "${ntest}" \
                "${best_epoch}" "${best_val}" "${test_acc}" "${elapsed}" "${log_file}" \
                >> "${SUMMARY}"
        else
            status="parse_failed"
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model}" "${optimizer}" "${hidden_layers}" "${hidden_count}" \
        "${ntrain}" "${nval}" "${ntest}" \
        "${status}" "${exit_code}" "${elapsed}" "${log_file}" >> "${RUN_STATUS}"
    printf 'finished run=%s status=%s exit_code=%s elapsed_seconds=%s\n' \
        "${run_name}" "${status}" "${exit_code}" "${elapsed}" | tee -a "${ALL_RUNS}"
}

for config in ${EXPERIMENT_CONFIGS}; do
    IFS=',' read -r ntrain nval ntest <<< "${config}"
    if [[ -z "${ntrain}" || -z "${nval}" || -z "${ntest}" ]]; then
        printf 'Invalid experiment config: %s (expected ntrain,nval,ntest)\n' \
            "${config}" >&2
        exit 2
    fi
    for hidden_count in ${HIDDEN_COUNTS}; do
        for optimizer in ${RXINFER_OPTIMIZERS}; do
            run_experiment rxinfer_1h "${optimizer}" 1 "${hidden_count}" "${ntrain}" "${nval}" "${ntest}"
            run_experiment rxinfer_2h "${optimizer}" 2 "${hidden_count}" "${ntrain}" "${nval}" "${ntest}"
        done
        for hidden_layers in ${NN_LAYERS}; do
            run_experiment neural standard "${hidden_layers}" "${hidden_count}" "${ntrain}" "${nval}" "${ntest}"
        done
    done
done

printf '\nAll FashionMNIST experiments finished.\n'
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Summary: %s\n' "${SUMMARY}"
printf 'Run status: %s\n' "${RUN_STATUS}"
