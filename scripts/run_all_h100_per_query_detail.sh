#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="/home/rzwang/data/gofa-demo"
cd "${REPO_ROOT}"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

source scripts/gofa_formal_trace_env.sh
source scripts/gofa_canonical_m4k2v2_intqkpv_h100.sh
source scripts/make_h100_per_query_from_frozen_baseline.sh

###############################################################################
# Output root
###############################################################################

export H100_DETAIL_ROOT=\
"${CANON_ROOT}/latency_m4k2v2_w4a8_intqkpv_per_query_detail"

# The local config helper uses H100_QUERY_ROOT as its output root.
export H100_QUERY_ROOT="${H100_DETAIL_ROOT}"

TASKS=(
    cora_node
    cora_link
    pubmed_node
    wikics
    arxiv
)

declare -A WAYS=(
    [cora_node]=7
    [cora_link]=2
    [pubmed_node]=3
    [wikics]=10
    [arxiv]=40
)

CURRENT_TASK=""
CURRENT_REP=""
MONITOR_PID=""

###############################################################################
# Cleanup
###############################################################################

stop_gpu_monitor() {
    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

on_exit() {
    local rc=$?
    stop_gpu_monitor

    if [[ ${rc} -ne 0 ]]; then
        echo
        echo "============================================================"
        echo "DETAIL EXPERIMENT FAILED"
        echo "task=${CURRENT_TASK:-unknown}"
        echo "rep=${CURRENT_REP:-unknown}"
        echo "exit_code=${rc}"
        echo "============================================================"
    fi
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

###############################################################################
# Preflight
###############################################################################

if [[ "$(git rev-parse HEAD)" != \
      "0020695cb66bbbccf41e026fc1b47151b5b3ce02" ]]; then
    echo "ERROR: unexpected Git HEAD:"
    git rev-parse HEAD
    exit 1
fi

if ! declare -F make_h100_per_query_from_frozen_baseline >/dev/null; then
    echo "ERROR: make_h100_per_query_from_frozen_baseline is unavailable."
    exit 1
fi

python3 -m py_compile \
    run_gofa.py \
    modules/gofa/per_query_latency.py \
    modules/gofa/latency_event_accumulator.py \
    scripts/validate_h100_per_query_latency.py

# The frozen canonical traces/caches require eval_sample_size=-1.
if ! grep -q 'eval_sample_size not in {-1, 100}' run_gofa.py; then
    echo "ERROR: run_gofa.py does not contain the local eval_sample_size compatibility patch."
    exit 1
fi

###############################################################################
# GPU helpers
###############################################################################

check_gpu_idle() {
    local processes

    processes=$(
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits \
            2>/dev/null \
        | sed '/^[[:space:]]*$/d'
    )

    if [[ -n "${processes}" ]]; then
        echo "ERROR: GPU already has active compute processes:"
        echo "${processes}"
        return 1
    fi

    echo "GPU preflight: idle"
}

start_gpu_monitor() {
    local log_path=$1
    local contamination_path=$2

    rm -f "${contamination_path}"

    (
        while true; do
            local_output=$(
                nvidia-smi \
                    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                    --format=csv,noheader,nounits \
                    2>/dev/null || true
            )

            echo "===== $(date --iso-8601=seconds) ====="
            echo "${local_output}"

            process_count=$(
                printf '%s\n' "${local_output}" \
                | sed '/^[[:space:]]*$/d' \
                | wc -l
            )

            if (( process_count > 1 )); then
                echo "$(date --iso-8601=seconds): ${process_count} compute processes" \
                    >> "${contamination_path}"
            fi

            sleep 1
        done
    ) > "${log_path}" 2>&1 &

    MONITOR_PID=$!
}

###############################################################################
# Generate and audit one config
###############################################################################

prepare_detail_config() {
    local task=$1
    local ways=$2
    local rep=$3
    local append=$4

    local cfg

    cfg=$(
        make_h100_per_query_from_frozen_baseline \
            "${task}" \
            "${ways}" \
            "${rep}" \
            "${append}"
    )

    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: generated config not found: ${cfg}" >&2
        return 1
    fi

    TASK="${task}" \
    WAYS_VALUE="${ways}" \
    APPEND_VALUE="${append}" \
    CFG_PATH="${cfg}" \
    DETAIL_ROOT="${H100_DETAIL_ROOT}" \
    TRACE_ROOT="${QUERY_TRACE_ROOT}" \
    python3 - <<'PY'
import os
from pathlib import Path

import yaml


task = os.environ["TASK"]
ways = int(os.environ["WAYS_VALUE"])
append = os.environ["APPEND_VALUE"].lower() == "true"
cfg_path = Path(os.environ["CFG_PATH"])
detail_root = Path(os.environ["DETAIL_ROOT"])
trace_root = Path(os.environ["TRACE_ROOT"])

with cfg_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Preserve the frozen canonical workload.
cfg["eval_sample_size"] = -1
cfg["sample_size_per_task"] = 100
cfg["task_names"] = [task]
cfg["train_task_names"] = [task]
cfg["eval_task_names"] = [task]
cfg["ways"] = ways
cfg["inf_sample_size_per_task"] = [100]
cfg["inf_hops"] = [3]
cfg["inf_max_nodes_per_hops"] = [10]
cfg["inf_ways"] = [ways]
cfg["inf_instructs"] = [True]
cfg["inf_selections"] = [True]
cfg["batch_size"] = 1
cfg["eval_batch_size"] = 1
cfg["seed"] = 1
cfg["skip_validation"] = False
cfg["run_mode"] = "inf"
cfg["mode"] = "generate"

quant = dict(cfg["scheme_b_quant"])
quant["kv_base_load_policy"] = "target_1hop"
quant["kv_base_target_hops"] = 1
quant["target_aware_delta"] = False
quant["load_memory_delta"] = False
quant["load_key_delta"] = False
quant["load_value_delta"] = False
quant["load_key_base"] = True
quant["load_value_base"] = True
cfg["scheme_b_quant"] = quant

task_root = detail_root / task
output_csv = task_root / "per_query.csv"
trace_index = trace_root / f"{task}_formal_v1" / "trace_index.jsonl"

per_query = dict(cfg.get("gofa_per_query_latency", {}))
per_query.update({
    "enabled": True,
    "output_csv": str(output_csv),
    "trace_index_path": str(trace_index),
    "strict_trace_match": True,
    "cuda_sync": True,
    "export_wall_time": True,
    "export_gpu_time": True,
    "export_detail_gpu_time": True,
    "append": append,
    "rank_zero_only": True,
})
cfg["gofa_per_query_latency"] = per_query

cfg["gofa_query_trace"] = {"enabled": False}
cfg["gofa_query_trace_enabled"] = False
cfg["gofa_trace_source_audit"] = {"enabled": False}
cfg["gofa_trace_source_audit_enabled"] = False
cfg["profile_memory_kv_transformer_breakdown"] = False
cfg["print_generation_samples"] = 0

with cfg_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(cfg_path)
PY

    echo "${cfg}"
}

audit_config() {
    local cfg=$1
    local expected_task=$2
    local expected_ways=$3
    local expected_append=$4

    python3 - \
        "${cfg}" \
        "${expected_task}" \
        "${expected_ways}" \
        "${expected_append}" <<'PY'
import sys
from pathlib import Path

import yaml


path = Path(sys.argv[1])
task = sys.argv[2]
ways = int(sys.argv[3])
append = sys.argv[4].lower() == "true"

with path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

q = cfg["scheme_b_quant"]
p = cfg["gofa_per_query_latency"]
ig = cfg["scheme_b_int_gemm"]
qa = cfg["scheme_b_quant_kv_attention"]

checks = {
    "eval_sample_size": (cfg.get("eval_sample_size"), -1),
    "eval_task_names": (cfg.get("eval_task_names"), [task]),
    "ways": (cfg.get("ways"), ways),
    "inf_ways": (cfg.get("inf_ways"), [ways]),
    "inf_sample_size_per_task": (
        cfg.get("inf_sample_size_per_task"),
        [100],
    ),
    "inf_hops": (cfg.get("inf_hops"), [3]),
    "inf_max_nodes_per_hops": (
        cfg.get("inf_max_nodes_per_hops"),
        [10],
    ),
    "kv_policy": (
        q.get("kv_base_load_policy"),
        "target_1hop",
    ),
    "memory_bits": (q.get("memory_base_bits"), 4),
    "key_bits": (q.get("key_base_bits"), 2),
    "value_bits": (q.get("value_base_bits"), 2),
    "int_gemm": (ig.get("enabled"), True),
    "weight_bits": (ig.get("weight_bits"), 4),
    "activation_bits": (ig.get("activation_bits"), 8),
    "int_qk": (qa.get("use_int_qk"), True),
    "pv_mode": (qa.get("pv_compute_mode"), "int_pv"),
    "detail_enabled": (
        p.get("export_detail_gpu_time"),
        True,
    ),
    "append": (p.get("append"), append),
}

errors = []

for name, (actual, expected) in checks.items():
    if actual != expected:
        errors.append(
            f"{name}: expected={expected!r}, actual={actual!r}"
        )

for name, value in (
    ("full cache", cfg.get("encoder_cache_dir")),
    ("quant cache", q.get("cache_dir")),
    ("trace index", p.get("trace_index_path")),
):
    if not value or not Path(value).exists():
        errors.append(f"{name} does not exist: {value}")

if errors:
    print("CONFIG AUDIT FAILED", file=sys.stderr)

    for error in errors:
        print(f"  - {error}", file=sys.stderr)

    raise SystemExit(1)

print(
    f"CONFIG AUDIT PASS: task={task}, ways={ways}, "
    f"append={append}, detail=True"
)
PY
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

    local append
    if [[ "${rep}" -eq 0 ]]; then
        append=false
    else
        append=true
    fi

    local task_root="${H100_DETAIL_ROOT}/${task}"
    local rep_root="${task_root}/rep${rep}"
    local csv_path="${task_root}/per_query.csv"
    local csv_backup="${task_root}/per_query.before_rep${rep}.csv"
    local run_log="${rep_root}/run_stdout.log"
    local gpu_log="${rep_root}/gpu_processes.log"
    local contamination="${rep_root}/gpu_contamination_detected.txt"

    mkdir -p "${rep_root}"

    if [[ -f "${csv_path}" ]]; then
        cp -a "${csv_path}" "${csv_backup}"
    else
        rm -f "${csv_backup}"
    fi

    local cfg
    cfg=$(
        prepare_detail_config \
            "${task}" \
            "${ways}" \
            "${rep}" \
            "${append}" \
        | tail -1
    )

    audit_config \
        "${cfg}" \
        "${task}" \
        "${ways}" \
        "${append}"

    check_gpu_idle

    echo
    echo "============================================================"
    echo "RUN DETAIL PROFILE"
    echo "task=${task}"
    echo "ways=${ways}"
    echo "rep=${rep}"
    echo "append=${append}"
    echo "config=${cfg}"
    echo "csv=${csv_path}"
    echo "============================================================"

    start_gpu_monitor \
        "${gpu_log}" \
        "${contamination}"

    set +e

    /usr/bin/time \
        -f $'PROCESS_ELAPSED_S=%e\nMAX_RSS_KB=%M' \
        env PYTHONUNBUFFERED=1 \
        python3 run_gofa.py \
        --override "${cfg}" \
        2>&1 | tee "${run_log}"

    local run_rc=${PIPESTATUS[0]}

    set -e
    stop_gpu_monitor

    if [[ "${run_rc}" -ne 0 ]]; then
        echo "ERROR: run failed: task=${task}, rep=${rep}, rc=${run_rc}"

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
        else
            rm -f "${csv_path}"
        fi

        return "${run_rc}"
    fi

    if [[ -s "${contamination}" ]]; then
        echo "ERROR: multiple GPU compute processes were observed."
        cat "${contamination}"

        if [[ -f "${csv_backup}" ]]; then
            cp -a "${csv_backup}" "${csv_path}"
        else
            rm -f "${csv_path}"
        fi

        return 20
    fi

    python3 \
        scripts/validate_h100_per_query_latency.py \
        --input "${csv_path}" \
        --trace-index-path \
        "${QUERY_TRACE_ROOT}/${task}_formal_v1/trace_index.jsonl"

    local expected_lines=$((1 + (rep + 1) * 200))
    local actual_lines

    actual_lines=$(wc -l < "${csv_path}")

    if [[ "${actual_lines}" -ne "${expected_lines}" ]]; then
        echo "ERROR: CSV line count mismatch."
        echo "expected=${expected_lines}"
        echo "actual=${actual_lines}"
        return 21
    fi

    echo
    echo "REP PASS: task=${task}, rep=${rep}, lines=${actual_lines}"
}

###############################################################################
# Main
###############################################################################

if [[ -e "${H100_DETAIL_ROOT}" ]]; then
    BACKUP_ROOT=\
"${H100_DETAIL_ROOT}_backup_$(date +%Y%m%d_%H%M%S)"

    echo "Existing detail root found."
    echo "Moving:"
    echo "  ${H100_DETAIL_ROOT}"
    echo "to:"
    echo "  ${BACKUP_ROOT}"

    mv "${H100_DETAIL_ROOT}" "${BACKUP_ROOT}"
fi

mkdir -p "${H100_DETAIL_ROOT}"

echo "============================================================"
echo "START FORMAL H100 DETAIL EXPERIMENT"
echo "root=${H100_DETAIL_ROOT}"
echo "tasks=${TASKS[*]}"
echo "repetitions=3"
echo "expected_rows=3000"
echo "============================================================"

for task in "${TASKS[@]}"; do
    for rep in 0 1 2; do
        run_one_rep \
            "${task}" \
            "${WAYS[${task}]}" \
            "${rep}"
    done
done

###############################################################################
# Final validation
###############################################################################

ALL_CSVS=()

for task in "${TASKS[@]}"; do
    ALL_CSVS+=(
        "${H100_DETAIL_ROOT}/${task}/per_query.csv"
    )
done

echo
echo "============================================================"
echo "FINAL VALIDATION"
echo "============================================================"

python3 \
    scripts/validate_h100_per_query_latency.py \
    --input "${ALL_CSVS[@]}" \
    --trace-index-path \
    "${QUERY_TRACE_ROOT}/\${TASK}_formal_v1/trace_index.jsonl"

python3 - "${ALL_CSVS[@]}" <<'PY'
import csv
import sys

total = 0

for path in sys.argv[1:]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    print(f"{path}: {len(rows)} rows")
    total += len(rows)

print(f"TOTAL_DATA_ROWS={total}")

if total != 3000:
    raise SystemExit(
        f"Expected 3000 rows, got {total}"
    )
PY

echo
echo "============================================================"
echo "ALL FORMAL DETAIL EXPERIMENTS COMPLETED"
echo "Expected validator entries: 15 PASS"
echo "Total detail rows: 3000"
echo "Output root: ${H100_DETAIL_ROOT}"
echo "============================================================"
