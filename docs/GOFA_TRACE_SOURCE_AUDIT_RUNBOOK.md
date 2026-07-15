# GOFA Trace Source Audit Runbook

Agent does not run these commands. Run them on the real server with GPU, datasets, full Scheme-B cache, and quantized cache available.

All commands set `max_queries=1`, `num_workers=0`, and fixed seeds for the first audit pass.

The commands use the repository's current inference defaults for `ways=2` and `inf_ways=[2]`, as shown in `configs/inference_config.yaml`. If a saved task was generated with a different `way`, update both `ways` and `inf_ways` in the temporary YAML before running.

## Common Paths

```bash
cd /home/rzwang/data/GOFA/gofa-demo

export GOFA_DATA_ROOT=/home/rzwang/data/GOFA/TAGDataset
export GOFA_MODEL_DIR=/home/rzwang/data/GOFA/cache_data/model
export GOFA_LOAD_DIR=/home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth
export GOFA_CACHE_EXP=/home/rzwang/data/GOFA/cache_data/gofa_cache_exp
```

## Helper Command Template

The project CLI does not parse list-valued flat overrides. Use a tiny temporary YAML for task lists, and use flat CLI for audit toggles.

### cora_node, batch size 1

```bash
cat > /tmp/gofa_trace_audit_cora_node_bs1.yaml <<'YAML'
run_mode: "inf"
mode: "generate"
data_root_path: "/home/rzwang/data/GOFA/TAGDataset"
model_name_or_path: "/home/rzwang/data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2"
checkpoint_dir: "/home/rzwang/data/GOFA/cache_data/model"
load_model: true
load_dir: "/home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth"
seed: 1
batch_size: 1
num_workers: 0
num_layers: 6
gnn_type: "index"
fuse_type: "interleave"
dec_lora: true
llm_max_length: 256
train_sample_size: -1
eval_sample_size: -1
sample_size_per_task: 1
eval_task_names: ["cora_node"]
train_task_names: ["cora_node"]
hops: 3
train_max_nodes_per_hops: 5
ways: 2
instructs: true
selections: true
inf_sample_size_per_task: [1]
inf_hops: [3]
inf_max_nodes_per_hops: [10]
inf_ways: [2]
inf_instructs: [true]
inf_selections: [true]
use_encoder_cache: true
encoder_cache_mode: "memory_kv"
encoder_cache_skip_nog: true
encoder_cache_dir: "/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/cora_node"
scheme_b_quant:
  enabled: true
  cache_dir: "/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_node_m4k4v4d4"
  strict: true
  base_bits: 4
  delta_bits: 4
  memory_base_bits: 4
  key_base_bits: 4
  value_base_bits: 4
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
  kv_base_keep_target_edges: true
scheme_b_quant_kv_attention:
  enabled: false
YAML

python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_node_bs1.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True
```

### cora_link, batch size 1

```bash
cat > /tmp/gofa_trace_audit_cora_link_bs1.yaml <<'YAML'
run_mode: "inf"
mode: "generate"
data_root_path: "/home/rzwang/data/GOFA/TAGDataset"
model_name_or_path: "/home/rzwang/data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2"
checkpoint_dir: "/home/rzwang/data/GOFA/cache_data/model"
load_model: true
load_dir: "/home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth"
seed: 1
batch_size: 1
num_workers: 0
num_layers: 6
gnn_type: "index"
fuse_type: "interleave"
dec_lora: true
llm_max_length: 256
train_sample_size: -1
eval_sample_size: -1
sample_size_per_task: 1
eval_task_names: ["cora_link"]
train_task_names: ["cora_link"]
hops: 3
train_max_nodes_per_hops: 5
ways: 2
instructs: true
selections: true
inf_sample_size_per_task: [1]
inf_hops: [3]
inf_max_nodes_per_hops: [10]
inf_ways: [2]
inf_instructs: [true]
inf_selections: [true]
use_encoder_cache: true
encoder_cache_mode: "memory_kv"
encoder_cache_skip_nog: true
encoder_cache_dir: "/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/cora_link"
scheme_b_quant:
  enabled: true
  cache_dir: "/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_link_m4k4v4d4"
  strict: true
  base_bits: 4
  delta_bits: 4
  memory_base_bits: 4
  key_base_bits: 4
  value_base_bits: 4
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
  kv_base_keep_target_edges: true
scheme_b_quant_kv_attention:
  enabled: false
YAML

python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_link_bs1.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True
```

### pubmed_node, batch size 1

```bash
cp /tmp/gofa_trace_audit_cora_node_bs1.yaml /tmp/gofa_trace_audit_pubmed_node_bs1.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/gofa_trace_audit_pubmed_node_bs1.yaml")
s = p.read_text()
s = s.replace('["cora_node"]', '["pubmed_node"]')
s = s.replace('/full/cora_node', '/full/pubmed_node')
s = s.replace('/quant/cora_node_m4k4v4d4', '/quant/pubmed_node_m4k4v4d4')
p.write_text(s)
PY

python3 run_gofa.py --override /tmp/gofa_trace_audit_pubmed_node_bs1.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/pubmed_node_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True
```

### wikics, batch size 1

```bash
cp /tmp/gofa_trace_audit_cora_node_bs1.yaml /tmp/gofa_trace_audit_wikics_bs1.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/gofa_trace_audit_wikics_bs1.yaml")
s = p.read_text()
s = s.replace('["cora_node"]', '["wikics"]')
s = s.replace('/full/cora_node', '/full/wikics')
s = s.replace('/quant/cora_node_m4k4v4d4', '/quant/wikics_m4k4v4d4')
p.write_text(s)
PY

python3 run_gofa.py --override /tmp/gofa_trace_audit_wikics_bs1.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/wikics_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True
```

### cora_node, batch size 2

```bash
cp /tmp/gofa_trace_audit_cora_node_bs1.yaml /tmp/gofa_trace_audit_cora_node_bs2.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/gofa_trace_audit_cora_node_bs2.yaml")
s = p.read_text().replace("batch_size: 1", "batch_size: 2")
p.write_text(s)
PY

python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_node_bs2.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs2 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True \
  gofa_trace_source_audit_dump_pre_cache_snapshot True \
  gofa_trace_source_audit_dump_cache_miss_snapshot True
```

### cora_link, batch size 2

```bash
cp /tmp/gofa_trace_audit_cora_link_bs1.yaml /tmp/gofa_trace_audit_cora_link_bs2.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/gofa_trace_audit_cora_link_bs2.yaml")
s = p.read_text().replace("batch_size: 1", "batch_size: 2")
p.write_text(s)
PY

python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_link_bs2.yaml \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs2 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True \
  gofa_trace_source_audit_dump_pre_cache_snapshot True \
  gofa_trace_source_audit_dump_cache_miss_snapshot True
```

After the expected strict miss, classify the missing item and check whether both batched NOG items
were resolved into the skip set:

```bash
python3 scripts/analyze_gofa_bs2_cache_miss.py \
  --input /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs2/cache_miss_000000.json
```

## K-Only and V-Only Audit Stress Tests

These are source-semantics tests only. The current implementation may empty effective text K/V unless both key and value are selected.

```bash
python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_node_bs1.yaml \
  scheme_b_quant_load_key_base True \
  scheme_b_quant_load_value_base False \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_k_only_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True

python3 run_gofa.py --override /tmp/gofa_trace_audit_cora_node_bs1.yaml \
  scheme_b_quant_load_key_base False \
  scheme_b_quant_load_value_base True \
  gofa_trace_source_audit_enabled True \
  gofa_trace_source_audit_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_v_only_bs1 \
  gofa_trace_source_audit_max_queries 1 \
  gofa_trace_source_audit_include_full_arrays True \
  gofa_trace_source_audit_flush_interval 1 \
  gofa_trace_source_audit_rank_zero_only True \
  gofa_trace_source_audit_strict True
```

## Validate Outputs

```bash
python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs1

python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs1

python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/pubmed_node_bs1

python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/wikics_bs1

python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs2

python3 scripts/validate_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs2
```

## Render Runtime Report

```bash
python3 scripts/render_gofa_trace_source_audit.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs1 \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs1 \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/pubmed_node_bs1 \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/wikics_bs1 \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_node_bs2 \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/trace_audit/cora_link_bs2 \
  --output GOFA_TRACE_SOURCE_RUNTIME_RESULTS.md
```
