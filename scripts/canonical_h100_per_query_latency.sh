#!/usr/bin/env bash

# Generate one task-specific override. This function never launches GOFA.
generate_canonical_h100_per_query_latency_yaml() {
    if [[ $# -ne 6 ]]; then
        printf '%s\n' \
            "usage: generate_canonical_h100_per_query_latency_yaml TASK OUTPUT_YAML FULL_CACHE_ROOT QUANT_CACHE_DIR QUERY_TRACE_ROOT OUTPUT_CSV" >&2
        return 2
    fi

    local task="$1"
    local output_yaml="$2"
    local full_cache_root="$3"
    local quant_cache_dir="$4"
    local query_trace_root="$5"
    local output_csv="$6"
    case "${task}" in
        cora_node|cora_link|pubmed_node|wikics|arxiv) ;;
        *)
            printf 'unsupported canonical task: %s\n' "${task}" >&2
            return 2
            ;;
    esac

    mkdir -p "$(dirname "${output_yaml}")" "$(dirname "${output_csv}")"
    cat >"${output_yaml}" <<EOF
run_mode: "inf"
mode: "generate"
seed: 1
batch_size: 1
eval_sample_size: 100
skip_validation: False
eval_task_names: [${task}]
inf_sample_size_per_task: [100]
inf_hops: [3]
inf_max_nodes_per_hops: [10]
inf_ways: [2]
inf_instructs: [True]
inf_selections: [True]

use_encoder_cache: True
encoder_cache_dir: "${full_cache_root}"
encoder_cache_mode: "memory_kv"
encoder_cache_skip_nog: True
encoder_cache_manifest:
  enabled: False

scheme_b_quant:
  enabled: True
  base_bits: 4
  delta_bits: 4
  memory_base_bits: 4
  key_base_bits: 2
  value_base_bits: 2
  memory_delta_bits: 4
  key_delta_bits: 2
  value_delta_bits: 2
  cache_dir: "${quant_cache_dir}"
  fake_quant: True
  strict: True
  debug_zero_base: False
  target_aware_delta: False
  load_memory_delta: False
  load_key_delta: False
  load_value_delta: False
  load_key_base: True
  load_value_base: True
  kv_base_load_policy: "all"

scheme_b_weight_quant:
  enabled: False
scheme_b_activation_quant:
  enabled: False
scheme_b_int_gemm:
  enabled: True
  target: "suffix_transformer"
  weight_bits: 4
  activation_bits: 8
  backend: "torch_int_mm"
  quantize_attention: True
  quantize_mlp: True
  quantize_layernorm: False
  fallback_to_fake_quant: False

scheme_b_quant_kv_attention:
  enabled: True
  backend: "torch_int_mm_qscale_fold"
  key_scale_fold_into_q: True
  quantize_query_bits: 8
  key_bits: 2
  value_bits: 2
  use_int_qk: True
  pv_compute_mode: "int_pv"
  quantize_prob_bits: 8
  prob_quant_granularity: "per_query"
  prob_quant_unsigned: False
  prob_quant_qmax: 127
  fallback_to_fp_attention: False
  fallback_to_scale_delayed_v: False
  compare_int_pv_with_fp_pv: False

gofa_query_trace:
  enabled: False
gofa_trace_source_audit:
  enabled: False
gofa_per_query_latency:
  enabled: True
  output_csv: "${output_csv}"
  trace_index_path: "${query_trace_root}/${task}_formal_v1/trace_index.jsonl"
  strict_trace_match: True
  cuda_sync: True
  export_wall_time: True
  export_gpu_time: True
  append: False
  rank_zero_only: True
EOF
    printf 'wrote %s\n' "${output_yaml}"
}
