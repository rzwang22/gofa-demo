#!/usr/bin/env bash

# GOFA formal query-trace environment
#
# Usage:
#   cd ~/data/gofa-demo
#   conda activate gofa
#   source scripts/gofa_formal_trace_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: this script must be sourced, not executed."
  echo
  echo "Use:"
  echo "  cd ~/data/gofa-demo"
  echo "  conda activate gofa"
  echo "  source scripts/gofa_formal_trace_env.sh"
  exit 1
fi


# ============================================================
# Repository and runtime environment
# ============================================================

export GOFA_REPO=/home/rzwang/data/gofa-demo

export WANDB_MODE=offline

export DATA_ROOT=/home/rzwang/data/GOFA/TAGDataset

export MODEL_PATH=/home/rzwang/data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2

export CHECKPOINT_DIR=/home/rzwang/data/GOFA/cache_data/model

export LOAD_DIR=/home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth


# ============================================================
# Canonical cache and trace roots
# ============================================================

export EXP_ROOT=/home/rzwang/data/GOFA/cache_data/gofa_cache_exp

export CANON_ROOT=${EXP_ROOT}/canonical_seed1_h3_n10_s100

export FULL_ROOT=${CANON_ROOT}/full

export MANIFEST_ROOT=${CANON_ROOT}/manifest

export QUANT_ROOT=${CANON_ROOT}/quant

export AUDIT_ROOT=${CANON_ROOT}/trace_audit

export QUERY_TRACE_ROOT=${CANON_ROOT}/query_traces


mkdir -p \
  "${FULL_ROOT}" \
  "${MANIFEST_ROOT}" \
  "${QUANT_ROOT}" \
  "${AUDIT_ROOT}" \
  "${QUERY_TRACE_ROOT}"


# ============================================================
# Show current environment
# ============================================================

show_gofa_trace_env () {
  echo
  echo "=================================================="
  echo "GOFA FORMAL TRACE ENVIRONMENT"
  echo "=================================================="
  echo "GOFA_REPO=${GOFA_REPO}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "LOAD_DIR=${LOAD_DIR}"
  echo "EXP_ROOT=${EXP_ROOT}"
  echo "CANON_ROOT=${CANON_ROOT}"
  echo "FULL_ROOT=${FULL_ROOT}"
  echo "MANIFEST_ROOT=${MANIFEST_ROOT}"
  echo "QUANT_ROOT=${QUANT_ROOT}"
  echo "AUDIT_ROOT=${AUDIT_ROOT}"
  echo "QUERY_TRACE_ROOT=${QUERY_TRACE_ROOT}"
  echo "WANDB_MODE=${WANDB_MODE}"
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none}"
  echo "=================================================="
}


# ============================================================
# Validate required paths
# ============================================================

check_gofa_trace_environment () {
  local failed=0

  for path in \
    "${GOFA_REPO}" \
    "${DATA_ROOT}" \
    "${MODEL_PATH}" \
    "${CHECKPOINT_DIR}" \
    "${LOAD_DIR}"
  do
    if [ ! -e "${path}" ]; then
      echo "ERROR: required path does not exist:"
      echo "${path}"
      failed=1
    fi
  done

  for script in \
    "${GOFA_REPO}/run_gofa.py" \
    "${GOFA_REPO}/scripts/validate_gofa_query_trace.py" \
    "${GOFA_REPO}/scripts/summarize_gofa_query_traces.py"
  do
    if [ ! -f "${script}" ]; then
      echo "ERROR: required script does not exist:"
      echo "${script}"
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi

  echo "GOFA formal trace environment check passed."
}


# ============================================================
# Generate canonical M4K2V2 exact-path configuration
# ============================================================

make_canonical_audit_m4k2v2_exact_config () {
  local TASK=$1
  local WAYS=$2
  local BATCH_SIZE=$3
  local AUDIT_MAX_QUERIES=${4:-1}

  if [ -z "${TASK}" ] || \
     [ -z "${WAYS}" ] || \
     [ -z "${BATCH_SIZE}" ]; then
    echo "Usage:"
    echo "  make_canonical_audit_m4k2v2_exact_config \\"
    echo "    <task> <ways> <batch_size> [audit_max_queries]"
    return 2
  fi

  local FULL_CACHE=${FULL_ROOT}/${TASK}
  local QUANT_CACHE=${QUANT_ROOT}/${TASK}_m4k2v2

  local AUDIT_OUT=${AUDIT_ROOT}/${TASK}_m4k2v2_bs${BATCH_SIZE}_exact_path

  local CFG=/tmp/gofa_audit_${TASK}_m4k2v2_bs${BATCH_SIZE}_exact.yaml

  cat > "${CFG}" <<YAML
run_mode: "inf"
mode: "generate"

offline_log: true

data_root_path: "${DATA_ROOT}"

model_name_or_path: "${MODEL_PATH}"
checkpoint_dir: "${CHECKPOINT_DIR}"

load_model: true
load_dir: "${LOAD_DIR}"

seed: 1

batch_size: ${BATCH_SIZE}
eval_batch_size: ${BATCH_SIZE}
num_workers: 0

num_layers: 6
gnn_type: "index"
fuse_type: "interleave"

dec_lora: true
llm_max_length: 256

train_sample_size: -1

# Must remain identical to canonical full-cache construction.
eval_sample_size: -1

# Must preserve validation -> test execution order.
skip_validation: false

sample_size_per_task: 100

train_task_names:
  - "${TASK}"

eval_task_names:
  - "${TASK}"

ways: ${WAYS}

inf_sample_size_per_task:
  - 100

inf_hops:
  - 3

inf_max_nodes_per_hops:
  - 10

inf_ways:
  - ${WAYS}

inf_instructs:
  - true

inf_selections:
  - true

use_encoder_cache: true
encoder_cache_mode: "memory_kv"
encoder_cache_skip_nog: true
encoder_cache_verify: false
encoder_cache_dir: "${FULL_CACHE}"

encoder_cache_manifest:
  enabled: false
  output_path: ""
  append: false
  log_interval: 20

scheme_b_quant:
  enabled: true

  cache_dir: "${QUANT_CACHE}"

  strict: true
  fake_quant: true

  base_bits: 4
  delta_bits: 4

  memory_base_bits: 4
  key_base_bits: 2
  value_base_bits: 2

  memory_delta_bits: 4
  key_delta_bits: 4
  value_delta_bits: 4

  target_aware_delta: false

  load_memory_delta: false
  load_key_delta: false
  load_value_delta: false

  load_key_base: true
  load_value_base: true

  kv_base_load_policy: "target_1hop"
  kv_base_target_hops: 1
  kv_base_local_degree_top_ratio: 0.0
  kv_base_keep_target_edges: true

scheme_b_weight_quant:
  enabled: false

scheme_b_activation_quant:
  enabled: false

scheme_b_int_gemm:
  enabled: false

scheme_b_quant_kv_attention:
  enabled: false

scheme_b_ablation:
  enabled: false

gofa_trace_source_audit:
  enabled: true

  output_dir: "${AUDIT_OUT}"

  max_queries: ${AUDIT_MAX_QUERIES}

  include_full_arrays: true
  flush_interval: 1
  rank_zero_only: true
  strict: true
YAML

  if [ ! -s "${CFG}" ]; then
    echo "ERROR: failed to generate configuration:"
    echo "${CFG}"
    return 3
  fi

  echo "${CFG}"
}


# ============================================================
# Generate a complete formal query trace
# ============================================================

run_formal_query_trace_full () {
  local TASK=$1
  local WAYS=$2

  if [ -z "${TASK}" ] || [ -z "${WAYS}" ]; then
    echo "Usage:"
    echo "  run_formal_query_trace_full <task> <ways>"
    return 2
  fi

  local OUT_DIR=${QUERY_TRACE_ROOT}/${TASK}_formal_v1

  local FULL_CACHE=${FULL_ROOT}/${TASK}
  local QUANT_CACHE=${QUANT_ROOT}/${TASK}_m4k2v2

  if [ ! -d "${FULL_CACHE}" ]; then
    echo
    echo "ERROR: full cache directory does not exist:"
    echo "${FULL_CACHE}"
    return 3
  fi

  if [ ! -d "${QUANT_CACHE}" ]; then
    echo
    echo "ERROR: M4K2V2 quant-cache directory does not exist:"
    echo "${QUANT_CACHE}"
    return 4
  fi

  local CFG

  CFG=$(make_canonical_audit_m4k2v2_exact_config \
    "${TASK}" \
    "${WAYS}" \
    1 \
    1)

  local CFG_RC=$?

  if [ "${CFG_RC}" -ne 0 ] || [ ! -f "${CFG}" ]; then
    echo
    echo "ERROR: failed to generate runtime configuration."
    echo "TASK=${TASK}"
    echo "CFG=${CFG}"
    return 5
  fi

  echo
  echo "=================================================="
  echo "GOFA FORMAL QUERY TRACE"
  echo "TASK=${TASK}"
  echo "WAYS=${WAYS}"
  echo "BATCH_SIZE=1"
  echo "MAX_QUERIES=0"
  echo "CONFIG=${CFG}"
  echo "OUTPUT=${OUT_DIR}"
  echo "=================================================="

  if [ -d "${OUT_DIR}" ]; then
    echo
    echo "ERROR:"
    echo "Formal trace directory already exists:"
    echo "${OUT_DIR}"
    echo
    echo "Formal exporter uses an append-only index."
    echo "Validate, rename, or remove the directory explicitly before rerunning."
    return 6
  fi

  mkdir -p "${OUT_DIR}" || return 7

  (
    set -o pipefail

    cd "${GOFA_REPO}" || exit 8

    python3 run_gofa.py \
      --override "${CFG}" \
      gofa_trace_source_audit_enabled False \
      gofa_trace_source_audit_dump_pre_cache_snapshot False \
      gofa_trace_source_audit_dump_cache_miss_snapshot False \
      gofa_query_trace_enabled True \
      gofa_query_trace_output_dir "${OUT_DIR}" \
      gofa_query_trace_max_queries 0 \
      gofa_query_trace_include_token_ids False \
      gofa_query_trace_include_text_preview True \
      gofa_query_trace_rank_zero_only True \
      gofa_query_trace_strict True \
      2>&1 | tee "${OUT_DIR}/run_stdout.log"

    exit "${PIPESTATUS[0]}"
  )

  local RUN_RC=$?

  if [ "${RUN_RC}" -ne 0 ]; then
    echo
    echo "ERROR:"
    echo "Formal trace generation failed."
    echo "TASK=${TASK}"
    echo "EXIT_CODE=${RUN_RC}"
    return "${RUN_RC}"
  fi

  echo
  echo "===== VALIDATE TRACE DIRECTORY ====="

  python3 \
    "${GOFA_REPO}/scripts/validate_gofa_query_trace.py" \
    --input "${OUT_DIR}"

  local RC=$?

  if [ "${RC}" -ne 0 ]; then
    echo
    echo "ERROR:"
    echo "Formal trace validation failed."
    return "${RC}"
  fi

  echo
  echo "===== VALIDATE TRACE INDEX ====="

  python3 \
    "${GOFA_REPO}/scripts/validate_gofa_query_trace.py" \
    --input "${OUT_DIR}/trace_index.jsonl"

  RC=$?

  if [ "${RC}" -ne 0 ]; then
    echo
    echo "ERROR:"
    echo "Formal trace index validation failed."
    return "${RC}"
  fi

  echo
  echo "===== GENERATE SUMMARY ====="

  python3 \
    "${GOFA_REPO}/scripts/summarize_gofa_query_traces.py" \
    --input "${OUT_DIR}" \
    --output "${OUT_DIR}/summary.json"

  RC=$?

  if [ "${RC}" -ne 0 ]; then
    echo
    echo "ERROR:"
    echo "Formal trace summary failed."
    return "${RC}"
  fi

  echo
  echo "===== CHECK QUERY AND SPLIT COUNTS ====="

  python3 - "${OUT_DIR}" "${TASK}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected_task = sys.argv[2]

trace_files = sorted(root.glob("query_*.json"))

split_counts = Counter()
task_counts = Counter()

for path in trace_files:
    with path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    split_counts[str(trace.get("split"))] += 1
    task_counts[str(trace.get("task_name"))] += 1

index_path = root / "trace_index.jsonl"

if not index_path.exists():
    raise SystemExit(
        f"ERROR: trace index does not exist: {index_path}"
    )

with index_path.open("r", encoding="utf-8") as f:
    index_entries = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print("trace_count =", len(trace_files))
print("index_count =", len(index_entries))
print("split_counts =", dict(split_counts))
print("task_counts =", dict(task_counts))

if len(trace_files) != len(index_entries):
    raise SystemExit(
        "ERROR: trace count and index count differ"
    )

if split_counts.get("val", 0) != 100:
    raise SystemExit(
        "ERROR: expected 100 validation traces"
    )

if split_counts.get("test", 0) != 100:
    raise SystemExit(
        "ERROR: expected 100 test traces"
    )

if len(trace_files) != 200:
    raise SystemExit(
        "ERROR: expected 200 total traces"
    )

if task_counts.get(expected_task, 0) != 200:
    raise SystemExit(
        f"ERROR: expected 200 traces for {expected_task}"
    )

print("FULL TRACE COUNT CHECK PASSED")
PY

  RC=$?

  if [ "${RC}" -ne 0 ]; then
    echo
    echo "ERROR:"
    echo "Formal trace count check failed."
    return "${RC}"
  fi

  echo
  echo "===== OUTPUT SIZE ====="

  du -sh "${OUT_DIR}"

  echo
  echo "=================================================="
  echo "FORMAL QUERY TRACE PASSED"
  echo "TASK=${TASK}"
  echo "=================================================="
}


# ============================================================
# Validate an already generated formal trace
# ============================================================

check_formal_query_trace () {
  local TASK=$1
  local TRACE_DIR=${QUERY_TRACE_ROOT}/${TASK}_formal_v1

  if [ -z "${TASK}" ]; then
    echo "Usage:"
    echo "  check_formal_query_trace <task>"
    return 2
  fi

  if [ ! -d "${TRACE_DIR}" ]; then
    echo "ERROR: trace directory does not exist:"
    echo "${TRACE_DIR}"
    return 3
  fi

  python3 \
    "${GOFA_REPO}/scripts/validate_gofa_query_trace.py" \
    --input "${TRACE_DIR}" || return $?

  python3 \
    "${GOFA_REPO}/scripts/validate_gofa_query_trace.py" \
    --input "${TRACE_DIR}/trace_index.jsonl" || return $?

  python3 - "${TRACE_DIR}" "${TASK}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected_task = sys.argv[2]

trace_files = sorted(root.glob("query_*.json"))

split_counts = Counter()
task_counts = Counter()

for path in trace_files:
    with path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    split_counts[str(trace.get("split"))] += 1
    task_counts[str(trace.get("task_name"))] += 1

index_path = root / "trace_index.jsonl"

with index_path.open("r", encoding="utf-8") as f:
    index_entries = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print("trace_count =", len(trace_files))
print("index_count =", len(index_entries))
print("split_counts =", dict(split_counts))
print("task_counts =", dict(task_counts))

if len(trace_files) != 200:
    raise SystemExit("ERROR: expected 200 trace files")

if len(index_entries) != 200:
    raise SystemExit("ERROR: expected 200 index entries")

if split_counts.get("val", 0) != 100:
    raise SystemExit("ERROR: expected 100 validation traces")

if split_counts.get("test", 0) != 100:
    raise SystemExit("ERROR: expected 100 test traces")

if task_counts.get(expected_task, 0) != 200:
    raise SystemExit(
        f"ERROR: expected 200 traces for {expected_task}"
    )

print("FORMAL TRACE CHECK PASSED")
PY
}


echo
echo "GOFA formal trace environment loaded."
echo "Repository: ${GOFA_REPO}"
echo "Query trace root: ${QUERY_TRACE_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "gofa" ]]; then
  echo
  echo "WARNING: current Conda environment is:"
  echo "${CONDA_DEFAULT_ENV:-none}"
  echo
  echo "Expected:"
  echo "gofa"
fi
