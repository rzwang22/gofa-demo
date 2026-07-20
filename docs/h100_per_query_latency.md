# Canonical H100 per-query latency

The exporter is read-only with respect to formal query traces and Scheme-B caches. It records one CSV row after each successful batch-size-1 query and leaves the existing aggregate profiler counters unchanged.

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
