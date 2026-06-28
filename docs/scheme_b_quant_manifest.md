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
