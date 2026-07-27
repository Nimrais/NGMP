#!/usr/bin/env bash

set -uo pipefail

# Default: 90 comparison runs =
#   72 direct: 2 datasets × 2 depths × 6 optimizers × 3 hidden widths
#   42 neural: 2 datasets × 7 depths × 3 hidden widths.
# The `full` split uses 40k train / 10k validation / 10k official test images.
# Set FULL_TRAIN=remaining to use the remaining 50k training images.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/mnist_comparison_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/logs"
ALL_RUNS="${OUTPUT_DIR}/all_runs.txt"
SUMMARY="${OUTPUT_DIR}/summary.tsv"
RUN_STATUS="${OUTPUT_DIR}/run_status.tsv"

JULIA_BIN="${JULIA_BIN:-julia}"
DATASETS="${DATASETS:-mnist fashion_mnist}"
DIRECT_MODELS="${DIRECT_MODELS:-${MODELS:-direct_1h direct_2h}}"
OPTIMIZERS="${OPTIMIZERS:-damped vector_transport_05 vector_transport_08 vector_transport_nesterov_05 vector_transport_nesterov_08 projected_nesterov_09}"
HIDDEN_COUNTS="${HIDDEN_COUNTS:-32 64 128}"
NN_LAYERS="${NN_LAYERS:-1 2 3 4 5 6 7}"
RUN_DIRECT="${RUN_DIRECT:-true}"
RUN_NEURAL="${RUN_NEURAL:-true}"
EXPERIMENT_CONFIGS="${EXPERIMENT_CONFIGS:-full}"
FULL_TRAIN="${FULL_TRAIN:-40000}"
FULL_NVAL="${FULL_NVAL:-10000}"
FULL_NTEST="${FULL_NTEST:-10000}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_INNER="${MAX_INNER:-3}"
SEED="${SEED:-1}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-10000}"
LOSS_BACKTRACKING="${LOSS_BACKTRACKING:-false}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-10}"

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' \
        'Compare direct Gaussian-surrogate optimizers and neural baselines.' \
        '' \
        'Default:' \
        '  datasets:  mnist fashion_mnist' \
        '  direct:    one- and two-hidden-layer surrogate models' \
        '  optimizers: damped, vector_transport_05, vector_transport_08,' \
        '              vector_transport_nesterov_05,' \
        '              vector_transport_nesterov_08, projected_nesterov_09' \
        '  neural:    Adam-trained Softplus MLPs with 1 through 7 hidden layers' \
        '  widths:    32 64 128' \
        '  split:     40000 train, 10000 validation, 10000 test' \
        '' \
        'Examples:' \
        '  ./mnist_experiments/run_mnist_direct_optimizer_sweep.sh' \
        '  FULL_TRAIN=remaining ./mnist_experiments/run_mnist_direct_optimizer_sweep.sh' \
        '  DATASETS=mnist EPOCHS=20 ./mnist_experiments/run_mnist_direct_optimizer_sweep.sh' \
        '  EXPERIMENT_CONFIGS=10000,1000,1000 ./mnist_experiments/run_mnist_direct_optimizer_sweep.sh' \
        '' \
        'Optional experimental line search:' \
        '  LOSS_BACKTRACKING=true ./mnist_experiments/run_mnist_direct_optimizer_sweep.sh'
    exit 0
elif (($# > 0)); then
    printf 'Unknown argument: %s (only --help is supported)\n' "$1" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}" "${REPO_DIR}/.data"
export DATADEPS_LOAD_PATH="${DATADEPS_LOAD_PATH:-${REPO_DIR}/.data}"
export DATADEPS_ALWAYS_ACCEPT="${DATADEPS_ALWAYS_ACCEPT:-true}"

: > "${ALL_RUNS}"
printf 'dataset\tmodel\toptimizer\talpha\tbeta\thidden_layers\thidden_count\tntrain\tnval\tntest\tsplit_sampling\tloss_backtracking\tbest_val_epoch\tbest_val_acc\ttest_acc\telapsed_seconds\tlog\n' \
    > "${SUMMARY}"
printf 'dataset\tmodel\toptimizer\talpha\tbeta\thidden_layers\thidden_count\tntrain\tnval\tntest\tsplit_sampling\tloss_backtracking\tstatus\texit_code\telapsed_seconds\tlog\n' \
    > "${RUN_STATUS}"

validate_boolean() {
    case "$2" in
        true|false) ;;
        *)
            printf '%s must be true or false, got: %s\n' "$1" "$2" >&2
            exit 2
            ;;
    esac
}

validate_boolean LOSS_BACKTRACKING "${LOSS_BACKTRACKING}"
validate_boolean RUN_DIRECT "${RUN_DIRECT}"
validate_boolean RUN_NEURAL "${RUN_NEURAL}"

validate_nonnegative_integer() {
    [[ "$2" =~ ^[0-9]+$ ]] || {
        printf '%s must be a nonnegative integer, got: %s\n' "$1" "$2" >&2
        exit 2
    }
}

validate_positive_integer() {
    validate_nonnegative_integer "$1" "$2"
    ((10#$2 > 0)) || {
        printf '%s must be positive, got: %s\n' "$1" "$2" >&2
        exit 2
    }
}

validate_nonnegative_integer FULL_NVAL "${FULL_NVAL}"
validate_nonnegative_integer FULL_NTEST "${FULL_NTEST}"
validate_positive_integer EPOCHS "${EPOCHS}"
validate_positive_integer BATCH_SIZE "${BATCH_SIZE}"
validate_positive_integer MAX_INNER "${MAX_INNER}"
validate_nonnegative_integer SEED "${SEED}"
validate_positive_integer EVAL_MAX_IMAGES "${EVAL_MAX_IMAGES}"
validate_nonnegative_integer MAX_BACKTRACKS "${MAX_BACKTRACKS}"

resolve_full_train() {
    case "${FULL_TRAIN}" in
        remaining|all|everything)
            # MNIST and Fashion-MNIST both contain 60,000 official training
            # examples. FULL_NVAL is held out from that split.
            printf '%s' "$((60000 - FULL_NVAL))"
            ;;
        *)
            [[ "${FULL_TRAIN}" =~ ^[0-9]+$ ]] || {
                printf 'FULL_TRAIN must be an integer or remaining, got: %s\n' \
                    "${FULL_TRAIN}" >&2
                exit 2
            }
            printf '%s' "${FULL_TRAIN}"
            ;;
    esac
}

run_experiment() {
    local dataset="$1"
    local model="$2"
    local optimizer="$3"
    local hidden_count="$4"
    local ntrain="$5"
    local nval="$6"
    local ntest="$7"
    local split_sampling="$8"
    local requested_hidden_layers="${9:-}"
    local projected_nesterov=false
    local vector_transport=false
    local vector_transport_nesterov=false
    local alpha beta hidden_layers script function_call run_backtracking
    local run_name log_file start_time end_time elapsed exit_code
    local final_line best_epoch best_val test_acc status

    case "${optimizer}" in
        damped)
            alpha=0.5
            beta=0.0
            ;;
        vector_transport_05)
            alpha=0.2
            beta=0.5
            vector_transport=true
            ;;
        vector_transport_08)
            alpha=0.2
            beta=0.8
            vector_transport=true
            ;;
        vector_transport_nesterov_05)
            alpha=0.2
            beta=0.5
            vector_transport_nesterov=true
            ;;
        vector_transport_nesterov_08)
            alpha=0.2
            beta=0.8
            vector_transport_nesterov=true
            ;;
        projected_nesterov_09)
            alpha=0.2
            beta=0.9
            projected_nesterov=true
            ;;
        standard)
            alpha=NA
            beta=NA
            ;;
        *)
            printf 'Unknown optimizer: %s\n' "${optimizer}" >&2
            return 2
            ;;
    esac

    case "${model}" in
        direct_1h)
            hidden_layers=1
            script="${SCRIPT_DIR}/mnist_softplus_rxinfer_mlp_flattened.jl"
            function_call=train_mlp_rxinfer_demo
            ;;
        direct_2h)
            hidden_layers=2
            script="${SCRIPT_DIR}/mnist_softplus_rxinfer_mlp_two_hidden.jl"
            function_call=train_two_hidden_mlp_rxinfer_demo
            ;;
        neural)
            [[ "${optimizer}" == "standard" ]] || {
                printf 'Neural model requires optimizer=standard\n' >&2
                return 2
            }
            validate_positive_integer hidden_layers "${requested_hidden_layers}"
            hidden_layers="${requested_hidden_layers}"
            script="${SCRIPT_DIR}/mnist_softplus_nn_mlp_flattened.jl"
            function_call=train_nn_mlp_demo
            ;;
        *)
            printf 'Unknown model: %s\n' "${model}" >&2
            return 2
            ;;
    esac

    if [[ "${model}" == "neural" ]]; then
        run_backtracking=NA
        run_name="${dataset}_${model}_${optimizer}_layers-${hidden_layers}_hidden-${hidden_count}_train-${ntrain}_val-${nval}_test-${ntest}"
    else
        [[ "${optimizer}" != "standard" ]] || {
            printf 'Direct models do not support optimizer=standard\n' >&2
            return 2
        }
        run_backtracking="${LOSS_BACKTRACKING}"
        run_name="${dataset}_${model}_${optimizer}_hidden-${hidden_count}_train-${ntrain}_val-${nval}_test-${ntest}_backtracking-${LOSS_BACKTRACKING}"
    fi
    log_file="${LOG_DIR}/${run_name}.txt"

    {
        printf '\n================================================================================\n'
        printf 'run=%s\n' "${run_name}"
        printf 'started_at=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
        printf 'dataset=%s model=%s optimizer=%s alpha=%s beta=%s\n' \
            "${dataset}" "${model}" "${optimizer}" "${alpha}" "${beta}"
        printf 'hidden_count=%s epochs=%s batch_size=%s max_inner=%s seed=%s\n' \
            "${hidden_count}" "${EPOCHS}" "${BATCH_SIZE}" "${MAX_INNER}" "${SEED}"
        printf 'train=%s val=%s test=%s split_sampling=%s loss_backtracking=%s\n' \
            "${ntrain}" "${nval}" "${ntest}" "${split_sampling}" "${run_backtracking}"
        printf '================================================================================\n'
    } | tee -a "${ALL_RUNS}"

    start_time="$(date +%s)"
    if [[ "${model}" == "neural" ]]; then
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:${dataset},
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                split_sampling=:${split_sampling},
                hidden_layers=${hidden_layers},
                hidden_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
                seed=${SEED},
                eval_max_images=${EVAL_MAX_IMAGES},
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    elif [[ "${model}" == "direct_1h" ]]; then
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:${dataset},
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                split_sampling=:${split_sampling},
                hidden_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
                seed=${SEED},
                alpha=${alpha},
                projected_nesterov=${projected_nesterov},
                nesterov_beta=${beta},
                vector_transport=${vector_transport},
                vector_transport_nesterov=${vector_transport_nesterov},
                vector_transport_momentum=${beta},
                max_inner=${MAX_INNER},
                eval_max_images=${EVAL_MAX_IMAGES},
                inference_backend=:direct,
                loss_backtracking=${LOSS_BACKTRACKING},
                max_backtracks=${MAX_BACKTRACKS},
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    else
        "${JULIA_BIN}" --project="${REPO_DIR}" -e "
            include(raw\"${script}\")
            ${function_call}(
                dataset=:${dataset},
                ntrain=${ntrain},
                nval=${nval},
                ntest=${ntest},
                split_sampling=:${split_sampling},
                hidden1_count=${hidden_count},
                hidden2_count=${hidden_count},
                batch_size=${BATCH_SIZE},
                epochs=${EPOCHS},
                seed=${SEED},
                alpha=${alpha},
                projected_nesterov=${projected_nesterov},
                nesterov_beta=${beta},
                vector_transport=${vector_transport},
                vector_transport_nesterov=${vector_transport_nesterov},
                vector_transport_momentum=${beta},
                max_inner=${MAX_INNER},
                eval_max_images=${EVAL_MAX_IMAGES},
                inference_backend=:direct,
                loss_backtracking=${LOSS_BACKTRACKING},
                max_backtracks=${MAX_BACKTRACKS},
            )
        " 2>&1 | tee "${log_file}" | tee -a "${ALL_RUNS}"
        exit_code="${PIPESTATUS[0]}"
    fi
    end_time="$(date +%s)"
    elapsed="$((end_time - start_time))"

    status=failed
    if [[ "${exit_code}" -eq 0 ]]; then
        status=completed
        final_line="$(grep 'best_val_epoch=.*best_val_acc=.*test_acc=' "${log_file}" | tail -n 1 || true)"
        best_epoch="$(sed -n 's/.*best_val_epoch=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        best_val="$(sed -n 's/.*best_val_acc=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        test_acc="$(sed -n 's/.*test_acc=\([^ ]*\).*/\1/p' <<< "${final_line}")"
        if [[ -n "${best_epoch}" && -n "${best_val}" && -n "${test_acc}" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${dataset}" "${model}" "${optimizer}" "${alpha}" "${beta}" \
                "${hidden_layers}" "${hidden_count}" "${ntrain}" "${nval}" "${ntest}" \
                "${split_sampling}" "${run_backtracking}" \
                "${best_epoch}" "${best_val}" "${test_acc}" \
                "${elapsed}" "${log_file}" >> "${SUMMARY}"
        else
            status=parse_failed
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${dataset}" "${model}" "${optimizer}" "${alpha}" "${beta}" \
        "${hidden_layers}" "${hidden_count}" "${ntrain}" "${nval}" "${ntest}" \
        "${split_sampling}" "${run_backtracking}" \
        "${status}" "${exit_code}" "${elapsed}" "${log_file}" \
        >> "${RUN_STATUS}"
    printf 'finished run=%s status=%s exit_code=%s elapsed_seconds=%s\n' \
        "${run_name}" "${status}" "${exit_code}" "${elapsed}" | tee -a "${ALL_RUNS}"
}

for config in ${EXPERIMENT_CONFIGS}; do
    if [[ "${config}" == "full" ]]; then
        ntrain="$(resolve_full_train)"
        nval="${FULL_NVAL}"
        ntest="${FULL_NTEST}"
        split_sampling=natural
    else
        IFS=',' read -r ntrain nval ntest split_sampling <<< "${config}"
        split_sampling="${split_sampling:-balanced}"
        if [[ -z "${ntrain}" || -z "${nval}" || -z "${ntest}" ]]; then
            printf 'Invalid config: %s (use full or ntrain,nval,ntest[,sampling])\n' \
                "${config}" >&2
            exit 2
        fi
    fi

    validate_nonnegative_integer ntrain "${ntrain}"
    validate_nonnegative_integer nval "${nval}"
    validate_nonnegative_integer ntest "${ntest}"
    case "${split_sampling}" in
        balanced|natural) ;;
        *)
            printf 'Sampling must be balanced or natural, got: %s\n' \
                "${split_sampling}" >&2
            exit 2
            ;;
    esac

    if ((ntrain + nval > 60000)); then
        printf 'Training plus validation exceeds 60,000: %s + %s\n' \
            "${ntrain}" "${nval}" >&2
        exit 2
    fi
    if ((ntest > 10000)); then
        printf 'Test size exceeds 10,000: %s\n' "${ntest}" >&2
        exit 2
    fi

    for dataset in ${DATASETS}; do
        for hidden_count in ${HIDDEN_COUNTS}; do
            validate_positive_integer hidden_count "${hidden_count}"
            if [[ "${RUN_DIRECT}" == "true" ]]; then
                for optimizer in ${OPTIMIZERS}; do
                    for model in ${DIRECT_MODELS}; do
                        run_experiment "${dataset}" "${model}" "${optimizer}" \
                            "${hidden_count}" "${ntrain}" "${nval}" "${ntest}" \
                            "${split_sampling}"
                    done
                done
            fi
            if [[ "${RUN_NEURAL}" == "true" ]]; then
                for hidden_layers in ${NN_LAYERS}; do
                    run_experiment "${dataset}" neural standard \
                        "${hidden_count}" "${ntrain}" "${nval}" "${ntest}" \
                        "${split_sampling}" "${hidden_layers}"
                done
            fi
        done
    done
done

printf '\nAll MNIST/Fashion-MNIST comparisons finished.\n'
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Summary: %s\n' "${SUMMARY}"
printf 'Run status: %s\n' "${RUN_STATUS}"
