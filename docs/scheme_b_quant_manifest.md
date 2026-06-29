# Scheme-B Task Manifest Quantization

Use this flow when the full Scheme-B cache root is shared across tasks, but the quantized cache should contain only one task's accessed cache items.

## 1. Record a Task Manifest

Run full Scheme-B inference with manifest recording enabled:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  encoder_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  encoder_cache_manifest_enabled True \
  encoder_cache_manifest_output_path /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/manifest/cora_node.json \
  encoder_cache_manifest_append False
```

The manifest records unique cache keys actually accessed by this run. NOG/prompt skip items are written under `skip_items` and are not used for quantization.

## 2. Quantize Only Manifest Items

```bash
python scripts/quantize_scheme_b_cache.py \
  --input-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  --output-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_node_b2d4 \
  --manifest /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/manifest/cora_node.json \
  --base-bits 2 \
  --delta-bits 4
```

Output layout matches strict runtime:

```text
<quant_cache_dir>/<cache_tag>/<prefix>/<cache_key>.pt
<quant_cache_dir>/delta/<cache_tag>/<prefix>/<cache_key>.pt
```

## 3. Check Quant Cache

```bash
python scripts/check_scheme_b_quant_cache.py \
  --full-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  --quant-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_node_b2d4 \
  --manifest /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/manifest/cora_node.json \
  --base-bits 2 \
  --max-items 10
```

## 4. Run Strict Quant Inference

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  encoder_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  scheme_b_quant_enabled True \
  scheme_b_quant_strict True \
  scheme_b_quant_base_bits 2 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_node_b2d4
```

If a different task accesses cache items outside this manifest-specific quant cache, strict mode should raise on the missing quant base item instead of falling back.

## Suffix Weight Quant Profiling

Full Scheme-B baseline:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False
```

Suffix W4A16 only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True
```

Cache 4-bit only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled False
```

Cache 4-bit plus suffix W4A16:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True
```

## Suffix Activation Quant Profiling

A8 only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 8 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True
```

W4A8 only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 8 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True
```

Cache 4-bit plus A8:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 8 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True
```

Cache 4-bit plus W4A8:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 8 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True
```

A4 only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True \
  scheme_b_activation_quant_clip_ratio 1.0
```

W4A4 only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True \
  scheme_b_activation_quant_clip_ratio 1.0
```

Cache 4-bit plus A4:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True \
  scheme_b_activation_quant_clip_ratio 1.0
```

Cache 4-bit plus W4A4:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_fake_quant True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_per_token True \
  scheme_b_activation_quant_clip_ratio 1.0
```

## Attention Projection Activation Ablation

A4 q_proj only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj True \
  scheme_b_activation_quant_quantize_k_proj False \
  scheme_b_activation_quant_quantize_v_proj False \
  scheme_b_activation_quant_quantize_o_proj False \
  scheme_b_activation_quant_quantize_mlp False
```

A4 k_proj only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj False \
  scheme_b_activation_quant_quantize_k_proj True \
  scheme_b_activation_quant_quantize_v_proj False \
  scheme_b_activation_quant_quantize_o_proj False \
  scheme_b_activation_quant_quantize_mlp False
```

A4 v_proj only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj False \
  scheme_b_activation_quant_quantize_k_proj False \
  scheme_b_activation_quant_quantize_v_proj True \
  scheme_b_activation_quant_quantize_o_proj False \
  scheme_b_activation_quant_quantize_mlp False
```

A4 o_proj only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj False \
  scheme_b_activation_quant_quantize_k_proj False \
  scheme_b_activation_quant_quantize_v_proj False \
  scheme_b_activation_quant_quantize_o_proj True \
  scheme_b_activation_quant_quantize_mlp False
```

A4 q+k only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj True \
  scheme_b_activation_quant_quantize_k_proj True \
  scheme_b_activation_quant_quantize_v_proj False \
  scheme_b_activation_quant_quantize_o_proj False \
  scheme_b_activation_quant_quantize_mlp False
```

A4 v+o only:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj False \
  scheme_b_activation_quant_quantize_k_proj False \
  scheme_b_activation_quant_quantize_v_proj True \
  scheme_b_activation_quant_quantize_o_proj True \
  scheme_b_activation_quant_quantize_mlp False
```

Mixed precision target configuration:

Current config has one activation bit-width per run. Use full attention A8 as the reference, then run MLP-only A4 with attention unquantized:

```bash
python run_gofa.py --override configs/inference_config.yaml \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_quantize_attention False \
  scheme_b_activation_quant_quantize_mlp True
```

Use the per-projection bit fields below when one run needs different activation bit-widths for q/k/v/o and MLP.

## Per-Projection Activation Bits

Cache 4-bit plus W4 plus Q8/K4/V8/O4/MLP4:

```bash
python run_gofa.py \
  --override ./configs/inference_config.yaml \
  data_root_path /home/rzwang/data/GOFA/TAGDataset \
  load_dir /home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth \
  train_task_names cora_link \
  eval_task_names cora_link \
  sample_size_per_task 100 \
  inf_sample_size_per_task 100 \
  ways 2 \
  inf_ways 2 \
  inf_hops 3 \
  inf_max_nodes_per_hops 10 \
  inf_instructs True \
  inf_selections True \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  encoder_cache_skip_nog True \
  encoder_cache_verify False \
  profile_stage_times True \
  profile_stage_log_interval 20 \
  profile_memory_kv_transformer_breakdown False \
  encoder_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  encoder_cache_manifest_enabled False \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 4 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_target_aware_delta False \
  scheme_b_quant_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_link_b4d4 \
  scheme_b_quant_fake_quant True \
  scheme_b_quant_debug_zero_base False \
  scheme_b_quant_strict True \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_target suffix_transformer \
  scheme_b_weight_quant_fake_quant True \
  scheme_b_weight_quant_quantize_attention True \
  scheme_b_weight_quant_quantize_mlp True \
  scheme_b_weight_quant_quantize_layernorm False \
  scheme_b_weight_quant_log_quantized_modules True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_target suffix_transformer \
  scheme_b_activation_quant_fake_quant True \
  scheme_b_activation_quant_quantize_attention True \
  scheme_b_activation_quant_quantize_q_proj True \
  scheme_b_activation_quant_quantize_k_proj True \
  scheme_b_activation_quant_quantize_v_proj True \
  scheme_b_activation_quant_quantize_o_proj True \
  scheme_b_activation_quant_quantize_mlp True \
  scheme_b_activation_quant_q_proj_bits 8 \
  scheme_b_activation_quant_k_proj_bits 4 \
  scheme_b_activation_quant_v_proj_bits 8 \
  scheme_b_activation_quant_o_proj_bits 4 \
  scheme_b_activation_quant_mlp_bits 4 \
  scheme_b_activation_quant_quantize_qkv_outputs False \
  scheme_b_activation_quant_quantize_attn_output False \
  scheme_b_activation_quant_quantize_mlp_output False \
  scheme_b_activation_quant_per_token True \
  scheme_b_activation_quant_clip_ratio 1.0 \
  scheme_b_activation_quant_log_quantized_modules True \
  scheme_b_ablation_enabled False \
  offline_log True \
  num_workers 4
```

## Suffix Activation Observer

Use the observer to sample original suffix Transformer Linear input activations for layers 26, 29, and 31 without enabling cache, weight, or activation quantization:

```bash
python run_gofa.py \
  --override ./configs/inference_config.yaml \
  data_root_path /home/rzwang/data/GOFA/TAGDataset \
  load_dir /home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth \
  train_task_names cora_link \
  eval_task_names cora_link \
  sample_size_per_task 20 \
  inf_sample_size_per_task 20 \
  ways 2 \
  inf_ways 2 \
  inf_hops 3 \
  inf_max_nodes_per_hops 10 \
  inf_instructs True \
  inf_selections True \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  encoder_cache_skip_nog True \
  encoder_cache_verify False \
  profile_stage_times True \
  profile_stage_log_interval 10 \
  profile_memory_kv_transformer_breakdown False \
  encoder_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  encoder_cache_manifest_enabled False \
  scheme_b_quant_enabled False \
  scheme_b_weight_quant_enabled False \
  scheme_b_activation_quant_enabled False \
  scheme_b_activation_observer_enabled True \
  scheme_b_activation_observer_output_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/observer/cora_link_full \
  scheme_b_activation_observer_max_batches 2 \
  scheme_b_activation_observer_max_items_per_module 2 \
  scheme_b_activation_observer_layers 26,29,31 \
  scheme_b_activation_observer_projections q_proj,k_proj,v_proj,o_proj,mlp \
  scheme_b_activation_observer_save_tensor True \
  scheme_b_activation_observer_save_stats True \
  scheme_b_activation_observer_sample_tokens 512 \
  scheme_b_activation_observer_sample_channels 256 \
  scheme_b_activation_observer_compute_quant_error True \
  scheme_b_activation_observer_quant_bits 4,8 \
  scheme_b_activation_observer_per_token True \
  scheme_b_activation_observer_clip_ratio 1.0 \
  scheme_b_ablation_enabled False \
  offline_log True \
  num_workers 4
```

Plot the saved tensors and aggregate the JSONL statistics:

```bash
python scripts/plot_activation_observer.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/observer/cora_link_full \
  --output-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/observer/cora_link_full/plots \
  --max-files 50
```

The observer writes sampled tensors under `tensors/`, per-sample stats to `activation_stats.jsonl`, and a final `activation_observer_summary.json`.

## Scheme-B Two-Level Cache Policy

Inspect whether a task exposes full-graph degree metadata and sampled-subgraph local metadata:

```bash
python scripts/inspect_gofa_graph_metadata.py \
  --data-root-path /home/rzwang/data/GOFA/TAGDataset \
  --task-name cora_link \
  --max-samples 20
```

The script writes `/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/metadata/cora_link_graph_inspect.json` by default. If no full graph is found, use the sampled subgraph local degree and target-distance policy below.

Fixed compute configuration for cache policy profiling:

```text
W4 + Q8/K4/V8/O4/MLP4
```

Cache policy matrix:

1. Full cache + fixed compute: `scheme_b_quant_enabled False`
2. Cache4bit base-only + fixed compute: `scheme_b_quant_enabled True`, `scheme_b_quant_base_bits 4`, `scheme_b_quant_target_aware_delta False`
3. Cache2bit base-only + fixed compute: `scheme_b_quant_enabled True`, `scheme_b_quant_base_bits 2`, `scheme_b_quant_target_aware_delta False`
4. Cache2bit + all-delta + fixed compute: `scheme_b_quant_target_aware_policy all_delta`
5. Cache2bit + target-only delta + fixed compute: `scheme_b_quant_target_aware_policy target_only`
6. Cache2bit + target-1hop delta + fixed compute: `scheme_b_quant_target_aware_policy target_1hop`
7. Cache2bit + target-1hop-local-degree delta + fixed compute: `scheme_b_quant_target_aware_policy target_1hop_local_degree`

Example cache2bit + target-1hop policy:

```bash
python run_gofa.py \
  --override ./configs/inference_config.yaml \
  data_root_path /home/rzwang/data/GOFA/TAGDataset \
  load_dir /home/rzwang/data/GOFA/cache_data/model/instruct_2_ckpt.pth \
  train_task_names cora_link \
  eval_task_names cora_link \
  sample_size_per_task 100 \
  inf_sample_size_per_task 100 \
  ways 2 \
  inf_ways 2 \
  inf_hops 3 \
  inf_max_nodes_per_hops 10 \
  inf_instructs True \
  inf_selections True \
  use_encoder_cache True \
  encoder_cache_mode memory_kv \
  encoder_cache_skip_nog True \
  encoder_cache_verify False \
  profile_stage_times True \
  profile_stage_log_interval 20 \
  encoder_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  scheme_b_quant_enabled True \
  scheme_b_quant_base_bits 2 \
  scheme_b_quant_delta_bits 4 \
  scheme_b_quant_cache_dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_link_b2d4 \
  scheme_b_quant_strict True \
  scheme_b_quant_target_aware_delta True \
  scheme_b_quant_target_aware_policy target_1hop \
  scheme_b_quant_target_delta_hops 1 \
  scheme_b_quant_keep_target_edges True \
  scheme_b_quant_local_degree_top_ratio 0.0 \
  scheme_b_quant_local_degree_threshold None \
  scheme_b_quant_max_delta_items_per_batch None \
  scheme_b_weight_quant_enabled True \
  scheme_b_weight_quant_bits 4 \
  scheme_b_weight_quant_quantize_attention True \
  scheme_b_weight_quant_quantize_mlp True \
  scheme_b_activation_quant_enabled True \
  scheme_b_activation_quant_bits 4 \
  scheme_b_activation_quant_q_proj_bits 8 \
  scheme_b_activation_quant_k_proj_bits 4 \
  scheme_b_activation_quant_v_proj_bits 8 \
  scheme_b_activation_quant_o_proj_bits 4 \
  scheme_b_activation_quant_mlp_bits 4 \
  offline_log True \
  num_workers 4
```

For the all-delta upper bound, change only:

```bash
scheme_b_quant_target_aware_policy all_delta
```

For local-degree union with target 1-hop, use:

```bash
scheme_b_quant_target_aware_policy target_1hop_local_degree
scheme_b_quant_local_degree_top_ratio 0.2
```

## Scheme-B Cache Tensor Visualization

Inspect sampled full/quantized Scheme-B cache tensors offline. This analyzes `memory_state` and suffix text-side KV cache tensors without changing inference.

Example 1: inspect 4-bit cache:

```bash
python scripts/inspect_scheme_b_cache_tensors.py \
  --full-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  --quant-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_link_b4d4 \
  --manifest /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/manifest/cora_link.json \
  --output-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/cache_observer/cora_link_b4d4 \
  --sample-items 20 \
  --sample-policy random \
  --include-memory-state True \
  --include-text-kv True \
  --layers 26,27,28,29,30,31 \
  --kv-types key,value \
  --reconstruct-mode full,base,base_delta \
  --sample-tokens 512 \
  --sample-channels 256 \
  --compute-quant-error True
```

Example 2: inspect 2-bit base plus 4-bit delta cache:

```bash
python scripts/inspect_scheme_b_cache_tensors.py \
  --full-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/full/shared \
  --quant-cache-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/quant/cora_link_b2d4 \
  --manifest /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/manifest/cora_link.json \
  --output-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/cache_observer/cora_link_b2d4 \
  --sample-items 20 \
  --sample-policy random \
  --include-memory-state True \
  --include-text-kv True \
  --layers 26,27,28,29,30,31 \
  --kv-types key,value \
  --reconstruct-mode full,base,base_delta \
  --sample-tokens 512 \
  --sample-channels 256 \
  --compute-quant-error True
```

Plot saved tensors and aggregate CSV summaries:

```bash
python scripts/plot_scheme_b_cache_tensors.py \
  --input-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/cache_observer/cora_link_b2d4 \
  --output-dir /home/rzwang/data/GOFA/cache_data/gofa_cache_exp/cache_observer/cora_link_b2d4/plots \
  --max-files 100
```

Outputs:

- `tensors/*.pt`: sampled standardized `[tokens, channels]` tensors for full/base/base_delta.
- `stats/cache_tensor_stats.jsonl`: per-tensor distribution and reconstruction-error metrics.
- `plots/*.png`: heatmap, 3D surface, token/channel stats, histogram, and full/base/base_delta comparison plots.
- `plots/summary_by_*.csv`: aggregate summaries by tensor kind, layer, and reconstruct mode.
