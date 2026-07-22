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

`--data-root`, `--model-name-or-path`, and `--checkpoint-dir` must be existing directories. `--load-dir` must be an existing checkpoint file. The first preparation locks `suite_manifest.json` to the current Git commit, profile, ordered tasks, repetitions, runtime paths, and KV policy. Every later stage validates that identity. Use `--overwrite-suite` only when intentionally replacing the suite definition; it does not make old traces or latency rows compatible with the new identity.

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

4. Generate formal traces. Existing output is rejected by default. Use `--fresh` to remove each task's previous trace directory before rebuilding, or `--resume` to validate a complete continuous prefix and replay-check its task, split, query index, and graph signature before appending.

   ```bash
   scripts/generate_large_formal_traces.sh "${COMMON[@]}" --fresh --execute
   # After an interrupted run:
   scripts/generate_large_formal_traces.sh "${COMMON[@]}" --resume --execute
   ```

   Each trace keeps packed M4/K2/V2 logical data bytes, FP32 per-channel scale bytes, and uint32 gather/index metadata bytes in separate traffic categories. Scale or indexing overhead is never folded into the low-bit data byte count.

5. Run `nocache_bf16`, `cache_bf16`, and `cache_w8a8_m4k2v2`. Repetitions append to each mode/task CSV.

   ```bash
   scripts/run_large_gpu_suite.sh "${COMMON[@]}" --execute
   ```

   The GPU runner checks for an idle GPU before every repetition and polls compute PIDs once per second. Multiple compute processes terminate the current process group, roll the CSV back to its pre-repetition byte offset, and retain a contaminated log under `logs/gpu/`. Complete validated repetitions are skipped on restart; partial repetitions are removed and rerun.

6. Validate, summarize, and package simulator inputs. The package intentionally excludes full and quantized cache tensors; formal traces provide logical addresses and byte traffic.

   ```bash
   python3 scripts/validate_large_workload_suite.py "${COMMON[@]}"
   python3 scripts/summarize_large_workload_suite.py "${COMMON[@]}"
   python3 scripts/package_large_simulator_handoff.py "${COMMON[@]}"
   ```

   The raw mode/task latency CSV remains unchanged at 600 rows for a three-repetition, 200-query task. Summarization first takes rep0/1/2 medians per `(mode, task, split, query_uid)`, writes `summary/per_query_median.csv`, then reports task-level mean/p50/p95 across the 200 unique query medians.

Without `--execute`, preparation/build/run entry points only emit auditable shell plans. Formal trace matching checks task, split, query index, stable query UID, graph signature, workload profile, and cache keys where a cache mode is active.
