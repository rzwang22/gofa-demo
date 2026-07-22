# Canonical H100 per-query latency

The exporter is read-only with respect to formal query traces and Scheme-B caches. It records one CSV row after each successful batch-size-1 query and leaves the existing aggregate profiler counters unchanged.

`gofa_per_query_latency.profile_mode` selects `nocache_bf16`, `cache_bf16`, or `cache_w8a8_m4k2v2`. All modes strictly match the formal trace task, split, query index, stable query UID, workload profile, and graph signature. Cache modes additionally match cache keys; BF16 cache requires zero misses, while W8A8/M4K2V2 also requires zero quantization fallback.

Set `gofa_per_query_latency.export_detail_gpu_time: False` to disable all fine-grained Event creation while retaining the original per-query exporter fields and behavior. Detailed CSV columns remain present and are written as zero in that mode.

## Detailed GPU boundaries

All detailed times use `torch.cuda.Event`. Region code records start/end events only; elapsed times are evaluated after the existing end-of-query CUDA synchronization.

| CSV field | Start boundary | End boundary |
| --- | --- | --- |
| `quant_kv_attention_gpu_ms` | Entry to each successful quantized-KV attention call, before cached K/V payload materialization | After cached/current PV outputs are combined, before returning from that attention call |
| `prefix_transformer_gpu_ms` | Before the first encoder prefix Transformer layer for a full query or online NOG item | After the last prefix Transformer layer |
| `attention_gpu_ms` | Entry to each encoder Transformer attention module or cached-attention implementation | After the attention output is produced |
| `dense_fc_gpu_ms` | Entry to each encoder Transformer MLP | After the MLP output is produced |
| `kv_prepare_gpu_ms` | Cached K/V unpack/scale materialization, per-head Q-scale folding and INT-QK input preparation, or INT-PV P/V padding | Immediately before the corresponding attention arithmetic region |
| `int_qk_gpu_ms` | Immediately before the cached QK `torch._int_mm` | Immediately after that `torch._int_mm`, before slicing or logits dequantization |
| `softmax_prob_quant_gpu_ms` | Before attention softmax, and separately before each per-query P-to-INT8 quantization | After dropout for softmax, or after the INT8 probability tensor is produced |
| `int_pv_gpu_ms` | Immediately before the cached PV `torch._int_mm` | Immediately after that `torch._int_mm`, before slicing or output dequantization |
| `gnn_score_gpu_ms` | Before GNN node/edge normalization and QKV/edge projections | After edge attention scores are normalized and dropout is applied |
| `gnn_message_gpu_ms` | After normalized scores are available, before weighted message construction | After PyG neighborhood aggregation/scatter returns from `propagate` |
| `gnn_update_gpu_ms` | Before the aggregated-message output projection | After attention residual/gating, post-GNN normalization, FFN, and FFN residual/gating |

`gnn_other_gpu_ms` is `max(0, suffix_gnn_gpu_ms - gnn_score_gpu_ms - gnn_message_gpu_ms - gnn_update_gpu_ms)`. It contains outer suffix-GNN work such as slicing, concatenation, dtype conversion, and Event granularity remainder.

## Generate an override

Source the helper and pass task-specific cache paths explicitly:

```bash
source scripts/canonical_h100_per_query_latency.sh
generate_canonical_h100_per_query_latency_yaml \
  cora_node \
  /home/rzwang/data/gofa-demo/configs/h100_latency/cora_node.yaml \
  /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/m4k2v2/cora_node \
  /home/rzwang/data/GOFA/query_traces \
  /home/rzwang/data/GOFA/h100_latency/cora_node.csv
```

Repeat the command for `cora_link`, `pubmed_node`, `wikics`, and `arxiv`. The function only writes YAML; it does not launch inference.

Run the generated override through the normal entry point:

```bash
cd /home/rzwang/data/gofa-demo
python3 run_gofa.py --override configs/h100_latency/cora_node.yaml
```

The canonical run is strict M4K2V2, suffix W4A8 `torch_int_mm`, INT-QK/INT-PV, validation first, and test second. A trace mismatch, cache miss, or quantized attention fallback aborts before a CSV row is written.

## Validate output

```bash
python3 scripts/validate_h100_per_query_latency.py \
  --input /home/rzwang/data/GOFA/h100_latency/cora_node.csv \
  --trace-index-path '/home/rzwang/data/GOFA/query_traces/${TASK}_formal_v1/trace_index.jsonl'
```

The validator requires 100 validation and 100 test rows for every task and repetition, exact trace-index alignment, zero cache misses, zero fallback, unique trace order, and nonnegative timing fields.
