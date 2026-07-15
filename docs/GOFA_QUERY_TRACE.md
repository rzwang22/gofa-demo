# GOFA Formal Query Trace

The formal exporter writes one trace per query for the Scheme-B M4K2V2 path. It is a workload description for downstream analysis; it does not model PIM/NPU cycles, latency, energy, bank utilization, or bank imbalance.

## Constraints

- `batch_size=1` only. TAGLAS symbolic `NODEID` assignment is batch dependent, so batched query traces are rejected.
- `use_encoder_cache=True` and `encoder_cache_mode=memory_kv`.
- `scheme_b_quant.enabled=True` with memory base 4-bit, key base 2-bit, and value base 2-bit.
- `encoder_cache_skip_nog=True`; NOG is computed online and is never persistent.
- `node_map` means graph-local node to encoder/text item. It is not a global node ID.
- `edge_map` means structural edge to edge-text-item offset.
- Selective K/V access is shared by suffix Transformer layers 26-31.

## Configuration

```yaml
gofa_query_trace:
  enabled: true
  output_dir: /path/to/query_traces/cora_node
  max_queries: 0
  include_token_ids: false
  include_text_preview: true
  rank_zero_only: true
  strict: true
```

`max_queries=0` means unlimited. All fields also support the flat CLI form, for example:

```bash
python3 run_gofa.py --override configs/inference_config.yaml \
  gofa_query_trace_enabled True \
  gofa_query_trace_output_dir /path/to/query_traces/cora_node \
  gofa_query_trace_max_queries 100
```

The exporter writes `query_000000.json`, `query_000001.json`, and so on, plus an append-only `trace_index.jsonl`. Existing query IDs in the index are not overwritten.

## Trace Contents

Each trace includes metadata, model dimensions, the graph-local node/edge mapping, a cache item inventory, selective K/V accesses, actual suffix runtime tensor shapes, quantized-data traffic, and a summary. Cache tensor values, quantized payloads, scales, and zero points are never written.

Traffic uses `ceil(numel * bits / 8)` for the quantized data body. It excludes scale tensors and Python/PyTorch serialization metadata; the trace records this as `quantized_data_only_excluding_scale_and_container_metadata`. Selected K/V ratios use cacheable item-layer accesses as the denominator; `selected_KV_ratio` counts accesses where both K and V are present.

GNN layers produce updated node states but no independent edge output tensor. Consequently, `gnn.edge_output_shape` is `null` with status `not_produced_by_gofa_gnn_layer`.

## Validate

Validate one trace, an index, or an output directory:

```bash
python3 scripts/validate_gofa_query_trace.py --input /path/to/query_traces/cora_node
python3 scripts/validate_gofa_query_trace.py --input /path/to/query_traces/cora_node/trace_index.jsonl
python3 scripts/validate_gofa_query_trace.py --input /path/to/query_traces/cora_node/query_000000.json
```

The validator checks graph-local map ranges, NOG eligibility, inventory and mask lengths, layer access consistency, runtime shapes, one-query batching, and M4K2V2 byte accounting.

## Summarize

```bash
python3 scripts/summarize_gofa_query_traces.py \
  --input /path/to/query_traces/cora_node \
  --output /path/to/query_traces/cora_node_summary.json
```

The summary groups traces by task and reports query count; node/edge count min, mean, p50, p95, and max; node/edge text item counts; selected K/V ratios; persistent and runtime-loaded cache bytes; and online NOG count.
