# GOFA Large Workload Suite

The suite freezes one batch-size-1 query sequence and reuses its formal trace identity across the no-cache BF16, full-cache BF16, and W8A8/M4K2V2 GPU baselines. The default large profile is `large_h6_n32_s100`: seed 1, 100 validation queries, 100 test queries, 6 hops, and at most 32 sampled nodes per hop.

Every script accepts the same identity arguments:

```bash
COMMON=(
  --profile-root /path/to/gofa_profiles
  --data-root /path/to/TAGDataset
  --model-name-or-path /path/to/Mistral-7B-Instruct-v0.2
  --checkpoint-dir /path/to/model/checkpoints
  --load-dir /path/to/model/checkpoints/instruct_2_ckpt.pth
  --profile-name large_h6_n32_s100
  --hops 6
  --max-nodes-per-hop 32
  --samples-per-split 100
  --seed 1
  --tasks cora_node cora_link pubmed_node wikics arxiv
  --reps 3
  --kv-policy target_1hop
  --kv-target-hops 1
)
```

The task-specific `ways` values are fixed to 7/2/3/10/40 for Cora node, Cora link, PubMed node, WikiCS, and Arxiv respectively. Every generated stage config explicitly loads `--load-dir`; it does not inherit dataset or checkpoint identity from `default_config.yaml`. Formal traces and the `cache_w8a8_m4k2v2` baseline use the same KV policy. The formal importance-aware profile is `target_1hop` with `--kv-target-hops 1`; use `--kv-policy all` only for an explicit all-KV comparison suite.

## Pipeline

1. Generate and save the workload once. This writes task/split save names containing profile, task, split, seed, hops, and max-nodes. Omit `--execute` to inspect the command plan only.

   ```bash
   python3 scripts/prepare_large_workload_suite.py "${COMMON[@]}" --execute
   ```

2. Build the shared full Scheme-B cache and task manifests from the frozen workload.

   ```bash
   scripts/build_large_full_cache.sh "${COMMON[@]}" --execute
   ```

3. Convert only manifest-listed items to task-specific M4K2V2 caches.

   ```bash
   python3 scripts/build_large_quant_cache.py "${COMMON[@]}" --execute
   ```

4. Generate formal traces. Each trace contains the workload profile, graph SHA256, stable query UID, cache inventory, operation shapes, traffic, and aligned query-local logical address layout.

   ```bash
   scripts/generate_large_formal_traces.sh "${COMMON[@]}" --execute
   ```

5. Run `nocache_bf16`, `cache_bf16`, and `cache_w8a8_m4k2v2`. Repetitions append to each mode/task CSV.

   ```bash
   scripts/run_large_gpu_suite.sh "${COMMON[@]}" --execute
   ```

6. Validate, summarize, and package simulator inputs. The package intentionally excludes full and quantized cache tensors; formal traces provide logical addresses and byte traffic.

   ```bash
   python3 scripts/validate_large_workload_suite.py "${COMMON[@]}"
   python3 scripts/summarize_large_workload_suite.py "${COMMON[@]}"
   python3 scripts/package_large_simulator_handoff.py "${COMMON[@]}"
   ```

Without `--execute`, preparation/build/run entry points only emit auditable shell plans. Formal trace matching checks task, split, query index, stable query UID, graph signature, workload profile, and cache keys where a cache mode is active.
