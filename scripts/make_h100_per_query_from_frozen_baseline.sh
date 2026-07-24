#!/usr/bin/env bash

make_h100_per_query_from_frozen_baseline() {
  if [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "  make_h100_per_query_from_frozen_baseline TASK WAYS REP APPEND"
    return 2
  fi

  local TASK=$1
  local WAYS=$2
  local REP=$3
  local APPEND=$4

  case "${APPEND}" in
    true|false)
      ;;
    *)
      echo "ERROR: APPEND must be true or false." >&2
      return 3
      ;;
  esac

  local BASE_CFG

  BASE_CFG=$(
    make_canonical_m4k2v2_intqkpv_h100_config \
      "${TASK}" \
      "${WAYS}" \
      "${REP}" \
      timing
  ) || return $?

  if [ ! -f "${BASE_CFG}" ]; then
    echo "ERROR: baseline config does not exist: ${BASE_CFG}" >&2
    return 4
  fi

  local TASK_ROOT=${H100_QUERY_ROOT}/${TASK}
  local OUT_CFG=${TASK_ROOT}/rep${REP}/runtime_config.yaml
  local OUT_CSV=${TASK_ROOT}/per_query.csv
  local TRACE_INDEX=${QUERY_TRACE_ROOT}/${TASK}_formal_v1/trace_index.jsonl

  mkdir -p "$(dirname "${OUT_CFG}")"

  BASE_CFG="${BASE_CFG}" \
  OUT_CFG="${OUT_CFG}" \
  OUT_CSV="${OUT_CSV}" \
  TRACE_INDEX="${TRACE_INDEX}" \
  APPEND="${APPEND}" \
  python3 - <<'PY'
import os
from pathlib import Path

import yaml


base_cfg = Path(os.environ["BASE_CFG"])
out_cfg = Path(os.environ["OUT_CFG"])
out_csv = Path(os.environ["OUT_CSV"])
trace_index = Path(os.environ["TRACE_INDEX"])

append = os.environ["APPEND"].lower() == "true"

with base_cfg.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

# Keep the previously validated aggregate timing path unchanged.
cfg["profile_stage_times"] = True
cfg["profile_memory_kv_transformer_breakdown"] = False

cfg["gofa_query_trace"] = {
    "enabled": False,
}

cfg["gofa_query_trace_enabled"] = False

cfg["gofa_trace_source_audit"] = {
    "enabled": False,
}

cfg["gofa_trace_source_audit_enabled"] = False

cfg["gofa_per_query_latency"] = {
    "enabled": True,
    "output_csv": str(out_csv),
    "trace_index_path": str(trace_index),
    "strict_trace_match": True,
    "cuda_sync": True,
    "export_wall_time": True,
    "export_gpu_time": True,
    "append": append,
    "rank_zero_only": True,
}

cfg["print_generation_samples"] = 0

out_cfg.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with out_cfg.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(
        cfg,
        handle,
        sort_keys=False,
    )

print(out_cfg)
PY
}

echo "Loaded make_h100_per_query_from_frozen_baseline"
