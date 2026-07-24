#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script instead of executing it."
  echo
  echo "Use:"
  echo "  source scripts/gofa_formal_trace_env.sh"
  echo "  source scripts/gofa_canonical_m4k2v2_intqkpv_h100.sh"
  exit 1
fi

export GOFA_REPO=${GOFA_REPO:-/home/rzwang/data/gofa-demo}

export EXP_ROOT=${EXP_ROOT:-/home/rzwang/data/GOFA/cache_data/gofa_cache_exp}

export CANON_ROOT=${CANON_ROOT:-${EXP_ROOT}/canonical_seed1_h3_n10_s100}

export FULL_ROOT=${FULL_ROOT:-${CANON_ROOT}/full}
export QUANT_ROOT=${QUANT_ROOT:-${CANON_ROOT}/quant}
export QUERY_TRACE_ROOT=${QUERY_TRACE_ROOT:-${CANON_ROOT}/query_traces}

export H100_QUANT_ROOT=${H100_QUANT_ROOT:-${CANON_ROOT}/latency_m4k2v2_w4a8_intqkpv}

mkdir -p "${H100_QUANT_ROOT}"


gofa_task_ways () {
  local TASK=$1

  case "${TASK}" in
    cora_node)
      echo 7
      ;;
    cora_link)
      echo 2
      ;;
    pubmed_node)
      echo 3
      ;;
    wikics)
      echo 10
      ;;
    arxiv)
      echo 40
      ;;
    *)
      echo "ERROR: unsupported task: ${TASK}" >&2
      return 2
      ;;
  esac
}


check_canonical_quant_workload () {
  local TASK=$1

  if [ -z "${TASK}" ]; then
    echo "Usage:"
    echo "  check_canonical_quant_workload <task>"
    return 2
  fi

  local FULL_CACHE=${FULL_ROOT}/${TASK}
  local QUANT_CACHE=${QUANT_ROOT}/${TASK}_m4k2v2
  local TRACE_DIR=${QUERY_TRACE_ROOT}/${TASK}_formal_v1

  local FAILED=0

  echo
  echo "=================================================="
  echo "CANONICAL WORKLOAD CHECK"
  echo "TASK=${TASK}"
  echo "=================================================="

  if [ -d "${FULL_CACHE}" ]; then
    echo "FULL CACHE: OK"
    echo "${FULL_CACHE}"
  else
    echo "FULL CACHE: MISSING"
    echo "${FULL_CACHE}"
    FAILED=1
  fi

  if [ -d "${QUANT_CACHE}" ]; then
    echo "M4K2V2 CACHE: OK"
    echo "${QUANT_CACHE}"
  else
    echo "M4K2V2 CACHE: MISSING"
    echo "${QUANT_CACHE}"
    FAILED=1
  fi

  if [ -d "${TRACE_DIR}" ]; then
    echo "FORMAL TRACE: OK"
    echo "${TRACE_DIR}"
  else
    echo "FORMAL TRACE: MISSING"
    echo "${TRACE_DIR}"
    FAILED=1
  fi

  if [ -d "${TRACE_DIR}" ]; then
    local QUERY_COUNT
    local INDEX_COUNT

    QUERY_COUNT=$(
      find "${TRACE_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'query_*.json' \
        | wc -l
    )

    INDEX_COUNT=0

    if [ -f "${TRACE_DIR}/trace_index.jsonl" ]; then
      INDEX_COUNT=$(
        wc -l < "${TRACE_DIR}/trace_index.jsonl"
      )
    fi

    echo "QUERY_COUNT=${QUERY_COUNT}"
    echo "INDEX_COUNT=${INDEX_COUNT}"

    if [ "${QUERY_COUNT}" -ne 200 ]; then
      echo "ERROR: expected 200 formal query files."
      FAILED=1
    fi

    if [ "${INDEX_COUNT}" -ne 200 ]; then
      echo "ERROR: expected 200 trace-index entries."
      FAILED=1
    fi
  fi

  if [ "${FAILED}" -ne 0 ]; then
    echo
    echo "CANONICAL WORKLOAD CHECK FAILED"
    return 1
  fi

  echo
  echo "CANONICAL WORKLOAD CHECK PASSED"
}


make_canonical_m4k2v2_intqkpv_h100_config () {
  local TASK=$1
  local WAYS=$2
  local REP=${3:-1}
  local PROFILE_MODE=${4:-timing}

  if [ -z "${TASK}" ] || \
     [ -z "${WAYS}" ] || \
     [ -z "${REP}" ]; then
    echo "Usage:" >&2
    echo "  make_canonical_m4k2v2_intqkpv_h100_config \\" >&2
    echo "    <task> <ways> <rep> <timing|breakdown>" >&2
    return 2
  fi

  if [ "${PROFILE_MODE}" != "timing" ] && \
     [ "${PROFILE_MODE}" != "breakdown" ]; then
    echo "ERROR: profile mode must be timing or breakdown." >&2
    return 3
  fi

  local BASE_CFG

  BASE_CFG=$(
    make_canonical_audit_m4k2v2_exact_config \
      "${TASK}" \
      "${WAYS}" \
      1 \
      1
  ) || return $?

  local OUT_CFG=/tmp/gofa_${TASK}_m4k2v2_w4a8_intqkpv_${PROFILE_MODE}_rep${REP}.yaml

  BASE_CFG="${BASE_CFG}" \
  OUT_CFG="${OUT_CFG}" \
  TASK="${TASK}" \
  WAYS="${WAYS}" \
  PROFILE_MODE="${PROFILE_MODE}" \
  FULL_ROOT="${FULL_ROOT}" \
  QUANT_ROOT="${QUANT_ROOT}" \
  python3 - <<'PY'
import os
from pathlib import Path

import yaml


base_path = Path(os.environ["BASE_CFG"])
out_path = Path(os.environ["OUT_CFG"])

task = os.environ["TASK"]
ways = int(os.environ["WAYS"])
profile_mode = os.environ["PROFILE_MODE"]

full_root = Path(os.environ["FULL_ROOT"])
quant_root = Path(os.environ["QUANT_ROOT"])

with base_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# Canonical workload identity
# ============================================================

cfg["seed"] = 1

cfg["batch_size"] = 1
cfg["eval_batch_size"] = 1

cfg["num_workers"] = 0

cfg["eval_sample_size"] = -1

cfg["skip_validation"] = False

cfg["sample_size_per_task"] = 100

cfg["train_task_names"] = [task]
cfg["eval_task_names"] = [task]

cfg["ways"] = ways

cfg["inf_sample_size_per_task"] = [100]
cfg["inf_hops"] = [3]
cfg["inf_max_nodes_per_hops"] = [10]
cfg["inf_ways"] = [ways]
cfg["inf_instructs"] = [True]
cfg["inf_selections"] = [True]


# ============================================================
# Canonical full cache
# ============================================================

cfg["use_encoder_cache"] = True

cfg["encoder_cache_mode"] = "memory_kv"

cfg["encoder_cache_skip_nog"] = True

cfg["encoder_cache_verify"] = False

cfg["encoder_cache_dir"] = str(
    full_root / task
)

cfg["encoder_cache_manifest"] = {
    "enabled": False,
    "output_path": "",
    "append": False,
    "log_interval": 20,
}


# ============================================================
# Disable trace/audit output during timing
# ============================================================

cfg["gofa_trace_source_audit"] = {
    "enabled": False,
}

cfg["gofa_trace_source_audit_enabled"] = False

cfg["gofa_query_trace"] = {
    "enabled": False,
    "output_dir": "",
    "max_queries": 0,
    "include_token_ids": False,
    "include_text_preview": True,
    "rank_zero_only": True,
    "strict": True,
}

cfg["gofa_query_trace_enabled"] = False


# ============================================================
# Canonical M4K2V2 cache
# ============================================================

cfg["scheme_b_quant"] = {
    "enabled": True,

    "cache_dir": str(
        quant_root / f"{task}_m4k2v2"
    ),

    "strict": True,

    # Preserve the canonical cache representation.
    "fake_quant": True,

    "base_bits": 4,
    "delta_bits": 4,

    "memory_base_bits": 4,
    "key_base_bits": 2,
    "value_base_bits": 2,

    "memory_delta_bits": 4,
    "key_delta_bits": 4,
    "value_delta_bits": 4,

    "target_aware_delta": False,

    "load_memory_delta": False,
    "load_key_delta": False,
    "load_value_delta": False,

    "load_key_base": True,
    "load_value_base": True,

    "kv_base_load_policy": "target_1hop",
    "kv_base_target_hops": 1,
    "kv_base_local_degree_top_ratio": 0.0,
    "kv_base_keep_target_edges": True,

    "debug_zero_base": False,
}


# ============================================================
# W4A8 suffix Transformer
# ============================================================

cfg["scheme_b_weight_quant"] = {
    "enabled": False,
}

cfg["scheme_b_activation_quant"] = {
    "enabled": False,
}

cfg["scheme_b_int_gemm"] = {
    "enabled": True,

    "target": "suffix_transformer",

    "weight_bits": 4,
    "activation_bits": 8,

    "backend": "torch_int_mm",

    "quantize_attention": True,
    "quantize_mlp": True,
    "quantize_layernorm": False,

    "fallback_to_fake_quant": False,

    "log_modules": False,
}


# ============================================================
# Integer QK/PV with 2-bit-quantized K/V operands
# ============================================================

cfg["scheme_b_quant_kv_attention"] = {
    "enabled": True,

    "backend": "torch_int_mm_qscale_fold",

    "key_scale_fold_into_q": True,

    "quantize_query_bits": 8,

    "key_bits": 2,
    "value_bits": 2,

    "use_int_qk": True,

    "pv_compute_mode": "int_pv",

    "quantize_prob_bits": 8,

    "prob_quant_granularity": "per_query",

    "prob_quant_unsigned": False,

    "prob_quant_qmax": 127,

    "fallback_to_fp_attention": False,

    "fallback_to_scale_delayed_v": False,

    "compare_int_pv_with_fp_pv": False,

    "log_interval": 20,
}


cfg["scheme_b_ablation"] = {
    "enabled": False,
}


# ============================================================
# Profiling
# ============================================================

cfg["profile_stage_times"] = True

cfg["profile_stage_log_interval"] = 20

cfg["profile_memory_kv_transformer_breakdown"] = (
    profile_mode == "breakdown"
)

cfg["print_generation_samples"] = 0

cfg["offline_log"] = True


with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        sort_keys=False,
    )
PY

  local RC=$?

  if [ "${RC}" -ne 0 ] || \
     [ ! -s "${OUT_CFG}" ]; then
    echo "ERROR: failed to generate config:" >&2
    echo "${OUT_CFG}" >&2
    return 4
  fi

  printf '%s\n' "${OUT_CFG}"
}


run_canonical_m4k2v2_intqkpv_h100_one () {
  local TASK=$1
  local WAYS=$2
  local REP=${3:-1}
  local PROFILE_MODE=${4:-timing}

  check_canonical_quant_workload \
    "${TASK}" || return $?

  local OUT_DIR=\
${H100_QUANT_ROOT}/${TASK}/${PROFILE_MODE}/rep${REP}

  if [ -d "${OUT_DIR}" ]; then
    echo
    echo "ERROR: output directory already exists:"
    echo "${OUT_DIR}"
    echo
    echo "Remove or rename it explicitly before rerunning."
    return 5
  fi

  local CFG

  CFG=$(
    make_canonical_m4k2v2_intqkpv_h100_config \
      "${TASK}" \
      "${WAYS}" \
      "${REP}" \
      "${PROFILE_MODE}"
  ) || return $?

  local CFG_LINE_COUNT

  CFG_LINE_COUNT=$(
    printf '%s\n' "${CFG}" | wc -l
  )

  if [ "${CFG_LINE_COUNT}" -ne 1 ] || \
     [ ! -f "${CFG}" ]; then
    echo "ERROR: invalid generated config path:"
    printf '<%s>\n' "${CFG}"
    return 6
  fi

  mkdir -p "${OUT_DIR}" || return 7

  cp \
    "${CFG}" \
    "${OUT_DIR}/runtime_config.yaml"

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "task=${TASK}"
    echo "ways=${WAYS}"
    echo "rep=${REP}"
    echo "profile_mode=${PROFILE_MODE}"
    echo "batch_size=1"
    echo "seed=1"
    echo "execution_path=validation_then_test"
    echo "validation_queries=100"
    echo "test_queries=100"
    echo "cache=M4K2V2"
    echo "suffix_linear=W4A8"
    echo "qk=Q8_with_2bit_quantized_K_via_torch_int_mm"
    echo "pv=P8_with_2bit_quantized_V_via_torch_int_mm"
    echo
    echo "GOFA_GIT_COMMIT=$(git -C "${GOFA_REPO}" rev-parse HEAD 2>/dev/null || true)"
    echo
    nvidia-smi \
      --query-gpu=index,name,uuid,driver_version,memory.total \
      --format=csv,noheader
    echo
    python3 - <<'PY'
import torch

print("torch_version =", torch.__version__)
print("cuda_version =", torch.version.cuda)
print("cudnn_version =", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("gpu_name =", torch.cuda.get_device_name(0))
PY
  } > "${OUT_DIR}/metadata.txt"

  echo
  echo "=================================================="
  echo "GOFA CANONICAL H100 QUANTIZED BASELINE"
  echo "TASK=${TASK}"
  echo "WAYS=${WAYS}"
  echo "REP=${REP}"
  echo "PROFILE_MODE=${PROFILE_MODE}"
  echo "CONFIG=${CFG}"
  echo "OUTPUT=${OUT_DIR}"
  echo "=================================================="

  set -o pipefail

  (
    cd "${GOFA_REPO}" || exit 8

    /usr/bin/time \
      -f $'PROCESS_ELAPSED_S=%e\nMAX_RSS_KB=%M' \
      env PYTHONUNBUFFERED=1 \
      python3 run_gofa.py \
      --override "${CFG}"
  ) 2>&1 | tee "${OUT_DIR}/run_stdout.log"

  local RUN_RC=${PIPESTATUS[0]}

  if [ "${RUN_RC}" -ne 0 ]; then
    echo
    echo "ERROR: H100 quantized run failed."
    echo "TASK=${TASK}"
    echo "REP=${REP}"
    echo "EXIT_CODE=${RUN_RC}"
    return "${RUN_RC}"
  fi

  echo
  echo "===== FINAL CACHE STATUS ====="

  grep \
    'GOFA encoder memory/text-KV cache:' \
    "${OUT_DIR}/run_stdout.log" \
    | tail -1 || true

  echo
  echo "===== FINAL CACHE TIMING ====="

  grep \
    'GOFA encoder cache timing:' \
    "${OUT_DIR}/run_stdout.log" \
    | tail -1 || true

  echo
  echo "===== FINAL STAGE SUMMARY ====="

  grep -A30 \
    'GOFA stage timing summary: report=200' \
    "${OUT_DIR}/run_stdout.log" \
    | head -31 || true

  echo
  echo "===== QUANTIZED-KV / INT-PV STATUS ====="

  grep -E \
    'quantized-KV|int_pv|pv_compute_mode|quant_kv_attention|pv_int_mm|fallback_count|prob_quant' \
    "${OUT_DIR}/run_stdout.log" \
    | tail -30 || true

  echo
  echo "=================================================="
  echo "H100 QUANTIZED RUN COMPLETED"
  echo "TASK=${TASK}"
  echo "REP=${REP}"
  echo "PROFILE_MODE=${PROFILE_MODE}"
  echo "=================================================="
}


check_canonical_m4k2v2_intqkpv_h100_log () {
  local TASK=$1
  local REP=${2:-1}
  local PROFILE_MODE=${3:-timing}

  local LOG=\
${H100_QUANT_ROOT}/${TASK}/${PROFILE_MODE}/rep${REP}/run_stdout.log

  if [ ! -f "${LOG}" ]; then
    echo "ERROR: log does not exist:"
    echo "${LOG}"
    return 2
  fi

  echo
  echo "=================================================="
  echo "H100 QUANTIZED LOG CHECK"
  echo "TASK=${TASK}"
  echo "REP=${REP}"
  echo "PROFILE_MODE=${PROFILE_MODE}"
  echo "=================================================="

  if ! grep -q \
    'PROCESS_ELAPSED_S=' \
    "${LOG}"; then
    echo "ERROR: run did not reach process completion."
    return 3
  fi

  if grep -qE \
    'Traceback|CUDA out of memory|FileNotFoundError|RuntimeError:' \
    "${LOG}"; then
    echo "ERROR: fatal runtime error detected."
    grep -nE \
      'Traceback|CUDA out of memory|FileNotFoundError|RuntimeError:' \
      "${LOG}" \
      | tail -20
    return 4
  fi

  local FINAL_CACHE

  FINAL_CACHE=$(
    grep \
      'GOFA encoder memory/text-KV cache:' \
      "${LOG}" \
      | tail -1
  )

  echo "${FINAL_CACHE}"

  if [ -z "${FINAL_CACHE}" ]; then
    echo "ERROR: final cache status is missing."
    return 5
  fi

  if ! printf '%s\n' "${FINAL_CACHE}" \
      | grep -q 'total_misses=0'; then
    echo "ERROR: non-NOG cache miss detected."
    return 6
  fi

  if ! printf '%s\n' "${FINAL_CACHE}" \
      | grep -q 'total_skips=200'; then
    echo "WARNING: expected 200 NOG skips."
  fi

  if ! grep -q \
    'GOFA stage timing summary: report=200' \
    "${LOG}"; then
    echo "ERROR: final 200-query stage summary is missing."
    return 7
  fi

  if grep -qiE \
    'falling back to|fallback_to_fp_attention=True|fallback_to_scale_delayed_v=True' \
    "${LOG}"; then
    echo "ERROR: quantized attention fallback detected."
    grep -niE \
      'falling back to|fallback_to_fp_attention=True|fallback_to_scale_delayed_v=True' \
      "${LOG}" \
      | tail -20
    return 8
  fi

  echo
  echo "FINAL STAGE SUMMARY:"

  grep -A25 \
    'GOFA stage timing summary: report=200' \
    "${LOG}" \
    | head -26

  echo
  echo "FINAL CACHE TIMING:"

  grep \
    'GOFA encoder cache timing:' \
    "${LOG}" \
    | tail -1

  echo
  echo "QUANTIZED EXECUTION LINES:"

  grep -E \
    'quantized-KV|int_pv|pv_compute_mode|torch_int_mm|quant_kv_attention|pv_int_mm|prob_quant' \
    "${LOG}" \
    | tail -30 || true

  echo
  echo "H100 QUANTIZED LOG CHECK PASSED"
}


run_canonical_m4k2v2_intqkpv_h100_task () {
  local TASK=$1
  local PROFILE_MODE=${2:-timing}
  local REPEATS=${3:-3}

  local WAYS

  WAYS=$(
    gofa_task_ways "${TASK}"
  ) || return $?

  local REP

  for REP in $(seq 1 "${REPEATS}")
  do
    run_canonical_m4k2v2_intqkpv_h100_one \
      "${TASK}" \
      "${WAYS}" \
      "${REP}" \
      "${PROFILE_MODE}" || return $?

    check_canonical_m4k2v2_intqkpv_h100_log \
      "${TASK}" \
      "${REP}" \
      "${PROFILE_MODE}" || return $?
  done
}


run_canonical_m4k2v2_intqkpv_h100_all () {
  local PROFILE_MODE=${1:-timing}
  local REPEATS=${2:-3}

  local TASK

  for TASK in \
    cora_node \
    cora_link \
    pubmed_node \
    wikics \
    arxiv
  do
    run_canonical_m4k2v2_intqkpv_h100_task \
      "${TASK}" \
      "${PROFILE_MODE}" \
      "${REPEATS}" || return $?
  done
}


echo
echo "GOFA canonical M4K2V2 + W4A8 + INT-QK/PV H100 functions loaded."
echo "Output root:"
echo "${H100_QUANT_ROOT}"
