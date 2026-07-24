#!/usr/bin/env bash

set -Eeuo pipefail


###############################################################################
# Basic environment
###############################################################################

REPO_ROOT="/home/rzwang/data/gofa-demo"

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "gofa" ]]; then
    echo "ERROR: conda environment 'gofa' is not active." >&2
    echo "Run: conda activate gofa" >&2
    exit 1
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

source scripts/gofa_formal_trace_env.sh
source scripts/gofa_canonical_m4k2v2_intqkpv_h100.sh
source scripts/make_h100_per_query_from_frozen_baseline.sh

if ! declare -F make_h100_per_query_from_frozen_baseline >/dev/null; then
    echo "ERROR: make_h100_per_query_from_frozen_baseline is unavailable." >&2
    exit 1
fi

export H100_QUANT_ROOT=\
"${CANON_ROOT}/latency_m4k2v2_w4a8_intqkpv"

export H100_QUERY_ROOT=\
"${CANON_ROOT}/latency_m4k2v2_w4a8_intqkpv_per_query"

mkdir -p "${H100_QUERY_ROOT}"


###############################################################################
# Tasks
###############################################################################

TASKS=(
    cora_link
    pubmed_node
    wikics
    arxiv
)

declare -A TASK_WAYS=(
    [cora_link]=2
    [pubmed_node]=3
    [wikics]=10
    [arxiv]=40
)


###############################################################################
# Runtime state
###############################################################################

MONITOR_PID=""
CURRENT_TASK=""
CURRENT_REP=""


cleanup_monitor() {
    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}


on_exit() {
    local rc=$?

    cleanup_monitor

    if [[ ${rc} -ne 0 ]]; then
        echo
        echo "=================================================="
        echo "RUN ABORTED"
        echo "TASK=${CURRENT_TASK:-unknown}"
        echo "REP=${CURRENT_REP:-unknown}"
        echo "EXIT_CODE=${rc}"
        echo "=================================================="
    fi
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


###############################################################################
# GPU checks
###############################################################################

check_gpu_idle() {
    local gpu_processes

    gpu_processes=$(
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits \
            2>/dev/null \
        | sed '/^[[:space:]]*$/d'
    )

    if [[ -n "${gpu_processes}" ]]; then
        echo "ERROR: GPU already has active compute process(es):" >&2
        echo "${gpu_processes}" >&2
        echo >&2
        echo "Wait until the GPU is idle before continuing." >&2
        return 1
    fi

    echo "GPU preflight check: idle"
}


start_gpu_monitor() {
    local monitor_log=$1
    local contamination_flag=$2

    mkdir -p "$(dirname "${monitor_log}")"
    rm -f "${contamination_flag}"

    (
        while true; do
            timestamp=$(date --iso-8601=seconds)

            process_output=$(
                nvidia-smi \
                    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                    --format=csv,noheader,nounits \
                    2>/dev/null || true
            )

            echo "===== ${timestamp} ====="
            echo "${process_output}"

            process_count=$(
                printf '%s\n' "${process_output}" \
                | sed '/^[[:space:]]*$/d' \
                | wc -l
            )

            # Single-GPU GOFA should expose one CUDA compute process.
            # More than one process indicates possible GPU contention.
            if (( process_count > 1 )); then
                echo "${timestamp}: detected ${process_count} GPU compute processes" \
                    >> "${contamination_flag}"
            fi

            sleep 1
        done
    ) > "${monitor_log}" 2>&1 &

    MONITOR_PID=$!

    echo "GPU monitor started: PID=${MONITOR_PID}"
    echo "GPU monitor log: ${monitor_log}"
}


###############################################################################
# Configuration validation
###############################################################################

validate_generated_config() {
    local config_path=$1
    local expected_task=$2
    local expected_ways=$3
    local expected_append=$4
    local expected_csv=$5

    python3 - \
        "${config_path}" \
        "${expected_task}" \
        "${expected_ways}" \
        "${expected_append}" \
        "${expected_csv}" <<'PY'
import sys
from pathlib import Path

import yaml


config_path = Path(sys.argv[1])
expected_task = sys.argv[2]
expected_ways = int(sys.argv[3])
expected_append = sys.argv[4].lower() == "true"
expected_csv = str(Path(sys.argv[5]).resolve())

with config_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

errors = []

quant = cfg.get("scheme_b_quant", {})
int_gemm = cfg.get("scheme_b_int_gemm", {})
kv_attention = cfg.get("scheme_b_quant_kv_attention", {})
per_query = cfg.get("gofa_per_query_latency", {})


def check(name, actual, expected):
    if actual != expected:
        errors.append(
            f"{name}: expected={expected!r}, actual={actual!r}"
        )


check("run_mode", cfg.get("run_mode"), "inf")
check("mode", cfg.get("mode"), "generate")
check("seed", int(cfg.get("seed", -1)), 1)
check("batch_size", int(cfg.get("batch_size", -1)), 1)
check("eval_batch_size", int(cfg.get("eval_batch_size", -1)), 1)
check("skip_validation", bool(cfg.get("skip_validation")), False)

# The frozen canonical workload uses eval_sample_size=-1 and
# inf_sample_size_per_task=[100].
check("eval_sample_size", int(cfg.get("eval_sample_size", 0)), -1)

check("eval_task_names", cfg.get("eval_task_names"), [expected_task])
check("inf_sample_size_per_task", cfg.get("inf_sample_size_per_task"), [100])
check("inf_hops", cfg.get("inf_hops"), [3])
check("inf_max_nodes_per_hops", cfg.get("inf_max_nodes_per_hops"), [10])
check("inf_ways", cfg.get("inf_ways"), [expected_ways])
check("ways", int(cfg.get("ways", -1)), expected_ways)

check("use_encoder_cache", bool(cfg.get("use_encoder_cache")), True)
check("encoder_cache_mode", cfg.get("encoder_cache_mode"), "memory_kv")
check("encoder_cache_skip_nog", bool(cfg.get("encoder_cache_skip_nog")), True)

check("quant.enabled", bool(quant.get("enabled")), True)
check("quant.strict", bool(quant.get("strict")), True)
check("quant.memory_base_bits", int(quant.get("memory_base_bits", -1)), 4)
check("quant.key_base_bits", int(quant.get("key_base_bits", -1)), 2)
check("quant.value_base_bits", int(quant.get("value_base_bits", -1)), 2)
check("quant.target_aware_delta", bool(quant.get("target_aware_delta")), False)
check("quant.load_memory_delta", bool(quant.get("load_memory_delta")), False)
check("quant.load_key_delta", bool(quant.get("load_key_delta")), False)
check("quant.load_value_delta", bool(quant.get("load_value_delta")), False)
check("quant.load_key_base", bool(quant.get("load_key_base")), True)
check("quant.load_value_base", bool(quant.get("load_value_base")), True)
check("quant.kv_base_load_policy", quant.get("kv_base_load_policy"), "target_1hop")
check("quant.kv_base_target_hops", int(quant.get("kv_base_target_hops", -1)), 1)

check("int_gemm.enabled", bool(int_gemm.get("enabled")), True)
check("int_gemm.target", int_gemm.get("target"), "suffix_transformer")
check("int_gemm.weight_bits", int(int_gemm.get("weight_bits", -1)), 4)
check("int_gemm.activation_bits", int(int_gemm.get("activation_bits", -1)), 8)
check("int_gemm.backend", int_gemm.get("backend"), "torch_int_mm")
check("int_gemm.fallback_to_fake_quant",
      bool(int_gemm.get("fallback_to_fake_quant")), False)

check("kv_attention.enabled", bool(kv_attention.get("enabled")), True)
check("kv_attention.backend",
      kv_attention.get("backend"), "torch_int_mm_qscale_fold")
check("kv_attention.key_bits", int(kv_attention.get("key_bits", -1)), 2)
check("kv_attention.value_bits", int(kv_attention.get("value_bits", -1)), 2)
check("kv_attention.use_int_qk", bool(kv_attention.get("use_int_qk")), True)
check("kv_attention.pv_compute_mode",
      kv_attention.get("pv_compute_mode"), "int_pv")
check("kv_attention.fallback_to_fp_attention",
      bool(kv_attention.get("fallback_to_fp_attention")), False)
check("kv_attention.fallback_to_scale_delayed_v",
      bool(kv_attention.get("fallback_to_scale_delayed_v")), False)

check("per_query.enabled", bool(per_query.get("enabled")), True)
check("per_query.strict_trace_match",
      bool(per_query.get("strict_trace_match")), True)
check("per_query.cuda_sync", bool(per_query.get("cuda_sync")), True)
check("per_query.export_wall_time",
      bool(per_query.get("export_wall_time")), True)
check("per_query.export_gpu_time",
      bool(per_query.get("export_gpu_time")), True)
check("per_query.append",
      bool(per_query.get("append")), expected_append)

actual_csv = str(
    Path(per_query.get("output_csv", "")).resolve()
)
check("per_query.output_csv", actual_csv, expected_csv)

trace_path = Path(per_query.get("trace_index_path", ""))
if not trace_path.is_file():
    errors.append(
        f"trace index does not exist: {trace_path}"
    )

full_cache = Path(cfg.get("encoder_cache_dir", ""))
if not full_cache.is_dir():
    errors.append(
        f"full cache directory does not exist: {full_cache}"
    )

quant_cache = Path(quant.get("cache_dir", ""))
if not quant_cache.is_dir():
    errors.append(
        f"quant cache directory does not exist: {quant_cache}"
    )

if errors:
    print("CONFIG VALIDATION FAILED", file=sys.stderr)

    for error in errors:
        print(f"  - {error}", file=sys.stderr)

    raise SystemExit(1)

print(
    "CONFIG VALIDATION PASSED: "
    f"task={expected_task}, ways={expected_ways}, "
    f"append={expected_append}"
)
PY
}


###############################################################################
# CSV validation
###############################################################################

validate_task_csv() {
    local task=$1
    local csv_path=$2

    python3 \
        scripts/validate_h100_per_query_latency.py \
        --input "${csv_path}" \
        --trace-index-path \
        "${QUERY_TRACE_ROOT}/${task}_formal_v1/trace_index.jsonl"
}


check_csv_line_count() {
    local csv_path=$1
    local completed_reps=$2

    local expected_lines=$((1 + 200 * completed_reps))
    local actual_lines

    actual_lines=$(wc -l < "${csv_path}")

    if [[ "${actual_lines}" -ne "${expected_lines}" ]]; then
        echo "ERROR: unexpected CSV line count." >&2
        echo "CSV=${csv_path}" >&2
        echo "EXPECTED=${expected_lines}" >&2
        echo "ACTUAL=${actual_lines}" >&2
        return 1
    fi

    echo "CSV line count passed: ${actual_lines}"
}


###############################################################################
# One repetition
###############################################################################

run_one_rep() {
    local task=$1
    local ways=$2
    local rep=$3

    CURRENT_TASK="${task}"
    CURRENT_REP="${rep}"

    local append_value

    if [[ "${rep}" -eq 0 ]]; then
        append_value=false
    else
        append_value=true
    fi

    local task_root="${H100_QUERY_ROOT}/${task}"
    local rep_root="${task_root}/rep${rep}"
    local csv_path="${task_root}/per_query.csv"
    local csv_backup="${task_root}/per_query.before_rep${rep}.csv"
    local log_path="${rep_root}/run_stdout.log"
    local monitor_log="${rep_root}/gpu_processes.log"
    local contamination_flag="${rep_root}/gpu_contamination_detected.txt"

    mkdir -p "${rep_root}"

    if [[ -f "${csv_path}" ]]; then
        cp -a "${csv_path}" "${csv_backup}"
        echo "Backed up CSV: ${csv_backup}"
    else
        rm -f "${csv_backup}"
    fi

    local config_path

    config_path=$(
        make_h100_per_query_from_frozen_baseline \
            "${task}" \
            "${ways}" \
            "${rep}" \
            "${append_value}"
    )

    if [[ ! -f "${config_path}" ]]; then
        echo "ERROR: generated config is missing: ${config_path}" >&2
        return 1
    fi

    validate_generated_config \
        "${config_path}" \
        "${task}" \
        "${ways}" \
        "${append_value}" \
        "${csv_path}"

    check_gpu_idle

    echo
    echo "=================================================="
    echo "H100 PER-QUERY LATENCY"
    echo "TASK=${task}"
    echo "WAYS=${ways}"
    echo "REP=${rep}"
    echo "APPEND=${append_value}"
    echo "CONFIG=${config_path}"
    echo "CSV=${csv_path}"
    echo "LOG=${log_path}"
    echo "=================================================="
    echo

    start_gpu_monitor \
        "${monitor_log}" \
        "${contamination_flag}"

    set +e

    /usr/bin/time \
        -f $'PROCESS_ELAPSED_S=%e\nMAX_RSS_KB=%M' \
        env PYTHONUNBUFFERED=1 \
        python3 run_gofa.py \
        --override "${config_path}" \
        2>&1 | tee "${log_path}"

    local run_rc=${PIPESTATUS[0]}

    set -e

    cleanup_monitor

    if [[ "${run_rc}" -ne 0 ]]; then
        echo "ERROR: GOFA run failed: task=${task}, rep=${rep}, rc=${run_rc}" >&2

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
            echo "Restored CSV from ${csv_backup}" >&2
        else
            rm -f "${csv_path}"
            echo "Removed partial CSV ${csv_path}" >&2
        fi

        return "${run_rc}"
    fi

    if [[ -s "${contamination_flag}" ]]; then
        echo "ERROR: multiple GPU compute processes were observed." >&2
        cat "${contamination_flag}" >&2

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
            echo "Restored CSV from ${csv_backup}" >&2
        else
            rm -f "${csv_path}"
            echo "Removed partial CSV ${csv_path}" >&2
        fi

        return 20
    fi

    if grep -qE \
        'Traceback|CUDA out of memory|torch\.OutOfMemoryError|RuntimeError:' \
        "${log_path}"; then

        echo "ERROR: runtime error text found in ${log_path}" >&2

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
        else
            rm -f "${csv_path}"
        fi

        return 21
    fi

    if grep -qiE \
        'falling back to|int_pv failed' \
        "${log_path}"; then

        echo "ERROR: quantization fallback found in ${log_path}" >&2

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
        else
            rm -f "${csv_path}"
        fi

        return 22
    fi

    validate_task_csv \
        "${task}" \
        "${csv_path}"

    check_csv_line_count \
        "${csv_path}" \
        "$((rep + 1))"

    echo
    echo "REP COMPLETED"
    echo "TASK=${task}"
    echo "REP=${rep}"
    echo
}


###############################################################################
# One task
###############################################################################

run_one_task() {
    local task=$1
    local ways=$2

    local task_root="${H100_QUERY_ROOT}/${task}"

    if [[ -e "${task_root}" ]]; then
        local backup_root
        backup_root=\
"${H100_QUERY_ROOT}/previous_runs/${task}_$(date +%Y%m%d_%H%M%S)"

        mkdir -p "$(dirname "${backup_root}")"

        echo "Existing task result found."
        echo "Moving:"
        echo "  ${task_root}"
        echo "to:"
        echo "  ${backup_root}"

        mv "${task_root}" "${backup_root}"
    fi

    mkdir -p "${task_root}"

    for rep in 0 1 2; do
        run_one_rep \
            "${task}" \
            "${ways}" \
            "${rep}"
    done

    echo
    echo "=================================================="
    echo "TASK COMPLETED"
    echo "TASK=${task}"
    echo "WAYS=${ways}"
    echo "CSV=${task_root}/per_query.csv"
    echo "=================================================="
    echo
}


###############################################################################
# Main
###############################################################################

echo "=================================================="
echo "START REMAINING H100 PER-QUERY EXPERIMENTS"
echo "ROOT=${H100_QUERY_ROOT}"
echo "TASKS=${TASKS[*]}"
echo "=================================================="

for task in "${TASKS[@]}"; do
    run_one_task \
        "${task}" \
        "${TASK_WAYS[${task}]}"
done


###############################################################################
# Final validation: all five tasks
###############################################################################

echo
echo "=================================================="
echo "FINAL FIVE-TASK VALIDATION"
echo "=================================================="

ALL_CSVS=(
    "${H100_QUERY_ROOT}/cora_node/per_query.csv"
    "${H100_QUERY_ROOT}/cora_link/per_query.csv"
    "${H100_QUERY_ROOT}/pubmed_node/per_query.csv"
    "${H100_QUERY_ROOT}/wikics/per_query.csv"
    "${H100_QUERY_ROOT}/arxiv/per_query.csv"
)

for csv_path in "${ALL_CSVS[@]}"; do
    if [[ ! -f "${csv_path}" ]]; then
        echo "ERROR: required final CSV is missing: ${csv_path}" >&2
        exit 1
    fi
done

python3 \
    scripts/validate_h100_per_query_latency.py \
    --input "${ALL_CSVS[@]}" \
    --trace-index-path \
    "${QUERY_TRACE_ROOT}/\${TASK}_formal_v1/trace_index.jsonl"


###############################################################################
# Final row counts
###############################################################################

echo
echo "Final CSV counts:"

for csv_path in "${ALL_CSVS[@]}"; do
    printf '%-90s %6d lines\n' \
        "${csv_path}" \
        "$(wc -l < "${csv_path}")"
done

total_data_rows=$(
    python3 - "${ALL_CSVS[@]}" <<'PY'
import csv
import sys

total = 0

for path in sys.argv[1:]:
    with open(path, newline="") as handle:
        total += sum(1 for _ in csv.DictReader(handle))

print(total)
PY
)

echo
echo "TOTAL_DATA_ROWS=${total_data_rows}"

if [[ "${total_data_rows}" -ne 3000 ]]; then
    echo "ERROR: expected 3000 total data rows." >&2
    exit 1
fi

echo
echo "=================================================="
echo "ALL REMAINING TASKS COMPLETED"
echo "Expected validator results: 15 PASS entries"
echo "Total data rows: 3000"
echo "=================================================="
