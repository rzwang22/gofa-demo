from collections import OrderedDict

from typing import List, Optional, Tuple, Union
import math
import time

import torch
import torch.utils.checkpoint
from torch import nn

from .gnn import GOFADecoderLayer, GOFAGatedDecoderLayer, GOFAGNNConv
from transformers import MistralConfig, GenerationMixin
from transformers.models.mistral.modeling_mistral import (
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralModel,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    LossKwargs, logging, )
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast, )
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from .cache_quant import dequantize_tensor, quantized_tensor_int, quantized_tensor_scale

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mistralai/Mistral-7B-v0.1"
_CONFIG_FOR_DOC = "MistralConfig"


_TORCH_INT_MM_M_ALIGNMENT = 32


class _QuantKVPVError(RuntimeError):
    pass


def _pad_int8_rows_to_multiple(tensor: torch.Tensor, multiple: int = _TORCH_INT_MM_M_ALIGNMENT):
    original_rows = int(tensor.size(0))
    padded_rows = ((original_rows + multiple - 1) // multiple) * multiple
    if padded_rows == original_rows:
        return tensor.contiguous(), original_rows
    padded = torch.zeros((padded_rows, int(tensor.size(1))), dtype=torch.int8, device=tensor.device)
    padded[:original_rows].copy_(tensor)
    return padded.contiguous(), original_rows


def _pad_int8_2d_to_multiple(tensor: torch.Tensor, multiple: int = _TORCH_INT_MM_M_ALIGNMENT):
    if tensor.dim() != 2:
        raise RuntimeError(f"Expected a 2D int8 tensor to pad for torch._int_mm, got shape={tuple(tensor.shape)}.")
    original_rows = int(tensor.size(0))
    original_cols = int(tensor.size(1))
    padded_rows = ((original_rows + multiple - 1) // multiple) * multiple
    padded_cols = ((original_cols + multiple - 1) // multiple) * multiple
    if padded_rows == original_rows and padded_cols == original_cols:
        return tensor.contiguous(), original_rows, original_cols
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.int8, device=tensor.device)
    padded[:original_rows, :original_cols].copy_(tensor)
    return padded.contiguous(), original_rows, original_cols


class _SingleLayerKVCache:
    """
    Minimal cache for one Mistral decoder layer.

    It is used only by the encoder memory/text-KV cache path, where each item is
    run independently and the cached prefix is the item's text-side K/V.
    """
    def __init__(self, key_states=None, value_states=None, key_quant_payload=None, value_quant_payload=None):
        self.key_states = key_states
        self.value_states = value_states
        self.key_quant_payload = key_quant_payload
        self.value_quant_payload = value_quant_payload
        self.quantized_kv_attention = key_quant_payload is not None or value_quant_payload is not None
        self.key_cache = [] if key_states is None else [key_states]
        self.value_cache = [] if value_states is None else [value_states]

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        if self.key_states is None:
            return 0
        return self.key_states.shape[-2]

    def get_usable_length(self, seq_len, layer_idx: Optional[int] = 0):
        return self.get_seq_length(layer_idx)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if self.key_states is None:
            updated_key_states = key_states
            updated_value_states = value_states
        else:
            cached_keys = self.key_states.to(device=key_states.device, dtype=key_states.dtype)
            cached_values = self.value_states.to(device=value_states.device, dtype=value_states.dtype)
            updated_key_states = torch.cat([cached_keys, key_states], dim=-2)
            updated_value_states = torch.cat([cached_values, value_states], dim=-2)
        self.key_states = updated_key_states
        self.value_states = updated_value_states
        self.key_cache = [updated_key_states]
        self.value_cache = [updated_value_states]
        return updated_key_states, updated_value_states


class GOFAMistralModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig, gofa_config):
        super().__init__(config)
        self.gofa_config = gofa_config
        self.profile_stage_times = False
        self.profile_memory_kv_transformer_breakdown = False
        self.quant_kv_attention_config = {"enabled": False}
        self.quant_kv_attention_stats = self._new_quant_kv_attention_stats()
        self.quant_kv_attention_warned_fallback = False
        self.reset_stage_profile()

        # self.g_layers = nn.ModuleList([GOFAGatedDecoderLayer(gofa_config, layer_idx=i) for i in range(gofa_config.num_layers)])
        self.g_layers = nn.ModuleList(
            [GOFAGNNConv(gofa_config) for i in range(gofa_config.num_layers)])

        self.post_init()

    def align_weight(self):
        n_layers = len(self.layers)
        inactive_layers = n_layers - len(self.g_layers)
        partial_state_dict = OrderedDict()
        source_dict = self.layers.state_dict()
        for layer_name in source_dict:
            name_split = layer_name.split(".")
            layer_ind = int(name_split[0])
            if layer_ind >= inactive_layers:
                name_split[0] = str(layer_ind - inactive_layers)
                if name_split[2] in ["v_proj", "q_proj", "k_proj", "o_proj"]:
                    name_split[2] = "g"+name_split[2]
                partial_state_dict[".".join(name_split)] = source_dict[layer_name]

        self.g_layers.load_state_dict(partial_state_dict, strict=False)

    @property
    def gnn_start_layer(self):
        return self.config.num_hidden_layers - self.gofa_config.num_layers

    def reset_stage_profile(self):
        num_gnn_layers = self.gofa_config.num_layers
        self.stage_profile = {
            "encoder_full_calls": 0,
            "encoder_prefix_calls": 0,
            "encoder_suffix_calls": 0,
            "encoder_prefix_transformer_s": 0.0,
            "encoder_gnn_layer_s": [0.0 for _ in range(num_gnn_layers)],
            "encoder_suffix_transformer_layer_s": [0.0 for _ in range(num_gnn_layers)],
            "encoder_norm_s": 0.0,
            "boundary_input_norm_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_qkv_proj_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_rope_repeat_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_attn_scores_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_o_proj_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_post_attn_norm_s": [0.0 for _ in range(num_gnn_layers)],
            "boundary_mlp_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_text_kv_to_device_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_input_norm_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_qkv_proj_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_rope_cache_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_attn_scores_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_o_proj_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_post_attn_norm_s": [0.0 for _ in range(num_gnn_layers)],
            "memory_kv_mlp_s": [0.0 for _ in range(num_gnn_layers)],
        }

    def _stage_profile_enabled(self):
        return bool(getattr(self, "profile_stage_times", False))

    def _stage_profile_sync(self, ref_tensor):
        if ref_tensor is not None and ref_tensor.device.type == "cuda":
            torch.cuda.synchronize(ref_tensor.device)

    def _stage_profile_start(self, ref_tensor):
        if not self._stage_profile_enabled():
            return None
        self._stage_profile_sync(ref_tensor)
        return time.perf_counter()

    def _stage_profile_add(self, key, start_time, ref_tensor, index=None):
        if start_time is None:
            return
        self._stage_profile_sync(ref_tensor)
        elapsed = time.perf_counter() - start_time
        if index is None:
            self.stage_profile[key] += elapsed
        else:
            self.stage_profile[key][index] += elapsed

    def _stage_profile_increment(self, key):
        if self._stage_profile_enabled():
            self.stage_profile[key] += 1

    def _new_quant_kv_attention_stats(self):
        return {
            "quant_kv_attention_call_count": 0,
            "k_unpack_time_s": 0.0,
            "q_scale_fold_time_s": 0.0,
            "q_eff_quant_time_s": 0.0,
            "qk_int_mm_time_s": 0.0,
            "logits_dequant_time_s": 0.0,
            "softmax_time_s": 0.0,
            "v_unpack_time_s": 0.0,
            "pv_matmul_time_s": 0.0,
            "v_scale_apply_time_s": 0.0,
            "prob_quant_time_s": 0.0,
            "prob_quant_numel": 0,
            "prob_quant_scale_min": None,
            "prob_quant_scale_max": None,
            "prob_quant_scale_sum": 0.0,
            "prob_quant_scale_count": 0,
            "prob_quant_zero_count": 0,
            "prob_quant_saturation_count": 0,
            "prob_quant_zero_ratio": 0.0,
            "prob_quant_saturation_ratio": 0.0,
            "pv_int_mm_time_s": 0.0,
            "pv_dequant_time_s": 0.0,
            "pv_int_mm_shape_examples": [],
            "fallback_count_pv": 0,
            "kv_empty_count": 0,
            "int_pv_compare_rel_l2_error": None,
            "int_pv_compare_cosine_similarity": None,
            "int_pv_compare_max_abs_error": None,
            "fallback_count": 0,
            "example_shapes": {},
        }

    def configure_quant_kv_attention(self, config):
        self.quant_kv_attention_config = dict(config or {"enabled": False})
        self.quant_kv_attention_stats = self._new_quant_kv_attention_stats()
        self.quant_kv_attention_warned_fallback = False
        self.quant_kv_attention_warned_pv_fallback = False

    def _make_block_causal_mask(self, query_len, key_len, past_len, dtype, device):
        mask = torch.zeros((query_len, key_len), dtype=dtype, device=device)
        query_positions = torch.arange(query_len, device=device).unsqueeze(1) + past_len
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        blocked = key_positions > query_positions
        if blocked.any():
            mask = mask.masked_fill(blocked, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def _quant_kv_attention_enabled(self):
        return bool(getattr(self, "quant_kv_attention_config", {}).get("enabled", False))

    def _quant_kv_stat_add(self, key, elapsed):
        self.quant_kv_attention_stats[key] = self.quant_kv_attention_stats.get(key, 0.0) + float(elapsed)

    def _quant_kv_payload_to_q_scale(self, payload, expected_bits, device, name):
        if not isinstance(payload, dict):
            raise RuntimeError(f"quantized-KV attention expected {name} quant payload, got {type(payload)}.")
        if payload.get("encoding") != "symmetric_per_channel":
            raise RuntimeError(f"quantized-KV attention expected {name} symmetric_per_channel payload.")
        if int(payload.get("bits", -1)) != int(expected_bits):
            raise RuntimeError(
                f"quantized-KV attention {name} bits mismatch: payload={payload.get('bits')}, expected={expected_bits}."
            )
        q = quantized_tensor_int(payload, device=device)
        scale = quantized_tensor_scale(payload, device=device)
        if q.dim() == 4 and q.size(0) == 1:
            q = q.squeeze(0)
        if q.dim() != 3:
            raise RuntimeError(f"quantized-KV attention expected {name} shape [kv_heads, seq, head_dim], got {tuple(q.shape)}.")
        head_dim = int(q.size(-1))
        if scale.numel() != head_dim:
            raise RuntimeError(
                f"quantized-KV attention expected {name} per-channel scale with {head_dim} values, "
                f"got shape={tuple(scale.shape)}."
            )
        return q.contiguous(), scale.reshape(head_dim).contiguous()

    def _quant_kv_record_prob_quant_stats(self, probs_q, scale_p, qmax, elapsed):
        stats = self.quant_kv_attention_stats
        stats["prob_quant_time_s"] = stats.get("prob_quant_time_s", 0.0) + float(elapsed)
        numel = int(probs_q.numel())
        stats["prob_quant_numel"] = int(stats.get("prob_quant_numel", 0)) + numel
        if scale_p.numel() > 0:
            scale_min = float(scale_p.min().item())
            scale_max = float(scale_p.max().item())
            scale_sum = float(scale_p.sum().item())
            scale_count = int(scale_p.numel())
            current_min = stats.get("prob_quant_scale_min")
            current_max = stats.get("prob_quant_scale_max")
            stats["prob_quant_scale_min"] = scale_min if current_min is None else min(float(current_min), scale_min)
            stats["prob_quant_scale_max"] = scale_max if current_max is None else max(float(current_max), scale_max)
            stats["prob_quant_scale_sum"] = float(stats.get("prob_quant_scale_sum", 0.0)) + scale_sum
            stats["prob_quant_scale_count"] = int(stats.get("prob_quant_scale_count", 0)) + scale_count
            stats["prob_quant_scale_mean"] = (
                stats["prob_quant_scale_sum"] / max(int(stats["prob_quant_scale_count"]), 1)
            )
        if numel > 0:
            zero_count = int((probs_q == 0).sum().item())
            saturation_count = int((probs_q == int(qmax)).sum().item())
            stats["prob_quant_zero_count"] = int(stats.get("prob_quant_zero_count", 0)) + zero_count
            stats["prob_quant_saturation_count"] = int(stats.get("prob_quant_saturation_count", 0)) + saturation_count
            total = max(int(stats["prob_quant_numel"]), 1)
            stats["prob_quant_zero_ratio"] = float(stats["prob_quant_zero_count"]) / float(total)
            stats["prob_quant_saturation_ratio"] = float(stats["prob_quant_saturation_count"]) / float(total)

    def _quant_kv_quantize_probs_int8(self, probs, cfg):
        if int(cfg.get("quantize_prob_bits", 8)) != 8:
            raise RuntimeError("quantized-KV int_pv currently supports only quantize_prob_bits=8.")
        if str(cfg.get("prob_quant_granularity", "per_query")) != "per_query":
            raise RuntimeError("quantized-KV int_pv currently supports only prob_quant_granularity=per_query.")
        if bool(cfg.get("prob_quant_unsigned", False)):
            raise RuntimeError("quantized-KV int_pv stores non-negative probabilities in int8; uint8 is unsupported.")
        qmax = int(cfg.get("prob_quant_qmax", 127))
        if qmax <= 0 or qmax > 127:
            raise RuntimeError("quantized-KV int_pv requires 1 <= prob_quant_qmax <= 127.")
        quant_start = time.perf_counter()
        probs_f = probs.float()
        scale_p = (probs_f.amax(dim=-1, keepdim=True) / float(qmax)).clamp(min=1e-12)
        probs_q = torch.round(probs_f / scale_p).clamp(0, qmax).to(torch.int8).contiguous()
        self._quant_kv_record_prob_quant_stats(probs_q, scale_p, qmax, time.perf_counter() - quant_start)
        return probs_q, scale_p

    def _quant_kv_scale_delayed_v_output(self, cached_probs, v_int, scale_v, num_key_value_groups, dtype, record_stats=True):
        bsz, num_heads, q_len, text_len = cached_probs.shape
        head_dim = int(v_int.size(-1))
        device = cached_probs.device
        cached_output = torch.zeros((bsz, num_heads, q_len, head_dim), dtype=dtype, device=device)
        if text_len == 0:
            return cached_output
        for batch_idx in range(bsz):
            for head_idx in range(num_heads):
                kv_head_idx = head_idx // num_key_value_groups
                pv_start = time.perf_counter()
                pv = torch.matmul(cached_probs[batch_idx, head_idx], v_int[kv_head_idx].to(dtype=dtype))
                if record_stats:
                    self._quant_kv_stat_add("pv_matmul_time_s", time.perf_counter() - pv_start)

                scale_start = time.perf_counter()
                cached_output[batch_idx, head_idx] = (pv.float() * scale_v.reshape(1, head_dim)).to(dtype)
                if record_stats:
                    self._quant_kv_stat_add("v_scale_apply_time_s", time.perf_counter() - scale_start)
        return cached_output

    def _quant_kv_maybe_record_int_pv_compare(self, int_output, fp_output, cfg):
        if not bool(cfg.get("compare_int_pv_with_fp_pv", False)):
            return
        diff = int_output.float() - fp_output.float()
        ref = fp_output.float()
        rel_l2 = (
            torch.linalg.vector_norm(diff) /
            torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            int_output.float().reshape(1, -1),
            ref.reshape(1, -1),
            dim=-1,
        ).item()
        max_abs = diff.abs().max().item() if diff.numel() else 0.0
        stats = self.quant_kv_attention_stats
        stats["int_pv_compare_rel_l2_error"] = float(rel_l2)
        stats["int_pv_compare_cosine_similarity"] = float(cosine)
        stats["int_pv_compare_max_abs_error"] = float(max_abs)
        call_count = int(stats.get("quant_kv_attention_call_count", 0))
        interval = max(int(cfg.get("log_interval", 20)), 1)
        if call_count <= 3 or call_count % interval == 0:
            print(
                "GOFA quantized-KV int_pv debug compare: "
                f"rel_l2_error={rel_l2:.6g}, "
                f"cosine_similarity={cosine:.6g}, "
                f"max_abs_error={max_abs:.6g}"
            )

    def _quant_kv_int_pv_output(self, cached_probs, v_int, scale_v, num_key_value_groups, dtype, cfg):
        bsz, num_heads, q_len, text_len = cached_probs.shape
        head_dim = int(v_int.size(-1))
        device = cached_probs.device
        if text_len == 0:
            return torch.zeros((bsz, num_heads, q_len, head_dim), dtype=dtype, device=device)
        try:
            if not hasattr(torch, "_int_mm"):
                raise RuntimeError("torch._int_mm is unavailable for quantized-KV int_pv.")
            cached_output = torch.zeros((bsz, num_heads, q_len, head_dim), dtype=dtype, device=device)
            for batch_idx in range(bsz):
                for head_idx in range(num_heads):
                    kv_head_idx = head_idx // num_key_value_groups
                    probs_q, scale_p = self._quant_kv_quantize_probs_int8(cached_probs[batch_idx, head_idx], cfg)
                    probs_q_padded, original_m, original_k = _pad_int8_2d_to_multiple(probs_q)
                    v_q_padded, v_original_k, original_n = _pad_int8_2d_to_multiple(v_int[kv_head_idx].contiguous())
                    if original_k != v_original_k:
                        raise RuntimeError(
                            "quantized-KV int_pv K dimension mismatch after padding: "
                            f"P={original_k}, V={v_original_k}."
                        )
                    pv_start = time.perf_counter()
                    out_int = torch._int_mm(probs_q_padded, v_q_padded)
                    out_int = out_int[:original_m, :original_n].contiguous()
                    self._quant_kv_stat_add("pv_int_mm_time_s", time.perf_counter() - pv_start)

                    dequant_start = time.perf_counter()
                    out_fp = out_int.float() * scale_p.reshape(original_m, 1) * scale_v.reshape(1, head_dim)
                    cached_output[batch_idx, head_idx] = out_fp.to(dtype)
                    self._quant_kv_stat_add("pv_dequant_time_s", time.perf_counter() - dequant_start)

                    shape_examples = self.quant_kv_attention_stats.setdefault("pv_int_mm_shape_examples", [])
                    if len(shape_examples) < 3:
                        shape_examples.append(
                            {
                                "p_q": tuple(probs_q.shape),
                                "p_q_padded": tuple(probs_q_padded.shape),
                                "v_q": tuple(v_int[kv_head_idx].shape),
                                "v_q_padded": tuple(v_q_padded.shape),
                                "out": tuple(out_int.shape),
                            }
                        )
            if bool(cfg.get("compare_int_pv_with_fp_pv", False)):
                fp_output = self._quant_kv_scale_delayed_v_output(
                    cached_probs,
                    v_int,
                    scale_v,
                    num_key_value_groups,
                    dtype,
                    record_stats=False,
                )
                self._quant_kv_maybe_record_int_pv_compare(cached_output, fp_output, cfg)
            return cached_output
        except Exception as exc:
            if not bool(cfg.get("fallback_to_scale_delayed_v", False)):
                raise _QuantKVPVError(
                    "quantized-KV int_pv failed and fallback_to_scale_delayed_v=False."
                ) from exc
            self.quant_kv_attention_stats["fallback_count_pv"] += 1
            if not getattr(self, "quant_kv_attention_warned_pv_fallback", False):
                print(
                    "GOFA quantized-KV attention warning: int_pv failed; falling back to scale_delayed_v. "
                    f"reason={type(exc).__name__}: {exc}"
                )
                self.quant_kv_attention_warned_pv_fallback = True
            return self._quant_kv_scale_delayed_v_output(
                cached_probs,
                v_int,
                scale_v,
                num_key_value_groups,
                dtype,
                record_stats=True,
            )

    def _quant_kv_fallback_to_fp(
        self,
        query_states,
        key_states,
        value_states,
        text_kv,
        attention_mask,
        attention,
        num_key_value_groups,
        head_dim,
        reason=None,
    ):
        cfg = getattr(self, "quant_kv_attention_config", {})
        if not bool(cfg.get("fallback_to_fp_attention", False)):
            if reason is not None:
                raise RuntimeError("quantized-KV attention failed and fallback_to_fp_attention=False.") from reason
            raise RuntimeError("quantized-KV attention failed and fallback_to_fp_attention=False.")
        self.quant_kv_attention_stats["fallback_count"] += 1
        if not self.quant_kv_attention_warned_fallback:
            print("GOFA quantized-KV attention warning: falling back to fp attention for unsupported payload/backend.")
            self.quant_kv_attention_warned_fallback = True
        cached_keys = dequantize_tensor(text_kv.key_quant_payload, dtype=key_states.dtype).unsqueeze(0).to(
            device=key_states.device,
            dtype=key_states.dtype,
        )
        cached_values = dequantize_tensor(text_kv.value_quant_payload, dtype=value_states.dtype).unsqueeze(0).to(
            device=value_states.device,
            dtype=value_states.dtype,
        )
        key_states = torch.cat([cached_keys, key_states], dim=-2)
        value_states = torch.cat([cached_values, value_states], dim=-2)
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        softmax_start = time.perf_counter()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._quant_kv_stat_add("softmax_time_s", time.perf_counter() - softmax_start)
        attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)
        return torch.matmul(attn_weights, value_states), attn_weights

    def _memory_kv_quantized_attention(
        self,
        attention,
        query_states,
        key_states,
        value_states,
        text_kv,
        attention_mask,
        output_attentions,
        num_key_value_groups,
        head_dim,
    ):
        cfg = getattr(self, "quant_kv_attention_config", {})
        if cfg.get("backend") != "torch_int_mm_qscale_fold":
            raise RuntimeError("quantized-KV attention supports only backend=torch_int_mm_qscale_fold.")
        if not bool(cfg.get("key_scale_fold_into_q", True)):
            raise RuntimeError("quantized-KV attention requires key_scale_fold_into_q=True.")
        if int(cfg.get("quantize_query_bits", 8)) != 8:
            raise RuntimeError("quantized-KV attention currently supports only quantize_query_bits=8.")
        if not bool(cfg.get("use_int_qk", True)):
            raise RuntimeError("quantized-KV attention requires use_int_qk=True.")
        if cfg.get("pv_compute_mode") not in {"scale_delayed_v", "int_pv"}:
            raise RuntimeError("quantized-KV attention supports pv_compute_mode=scale_delayed_v or int_pv.")
        try:
            if not hasattr(torch, "_int_mm"):
                raise RuntimeError("torch._int_mm is unavailable.")
            if query_states.device.type != "cuda":
                raise RuntimeError("torch._int_mm quantized-KV attention requires CUDA tensors.")

            stats = self.quant_kv_attention_stats
            stats["quant_kv_attention_call_count"] += 1
            device = query_states.device
            dtype = query_states.dtype
            bsz, num_heads, q_len, _ = query_states.shape
            _, num_kv_heads, current_len, _ = key_states.shape
            key_bits = int(cfg.get("key_bits", 4))
            value_bits = int(cfg.get("value_bits", 4))

            k_unpack_start = time.perf_counter()
            k_int, scale_k = self._quant_kv_payload_to_q_scale(text_kv.key_quant_payload, key_bits, device, "key")
            self._quant_kv_stat_add("k_unpack_time_s", time.perf_counter() - k_unpack_start)
            v_unpack_start = time.perf_counter()
            v_int, scale_v = self._quant_kv_payload_to_q_scale(text_kv.value_quant_payload, value_bits, device, "value")
            self._quant_kv_stat_add("v_unpack_time_s", time.perf_counter() - v_unpack_start)

            if k_int.size(0) != num_kv_heads or v_int.size(0) != num_kv_heads:
                raise RuntimeError(
                    "quantized-KV attention kv head mismatch: "
                    f"k_heads={k_int.size(0)}, v_heads={v_int.size(0)}, expected={num_kv_heads}."
                )
            if k_int.size(-1) != head_dim or v_int.size(-1) != head_dim:
                raise RuntimeError(
                    "quantized-KV attention head_dim mismatch: "
                    f"k={k_int.size(-1)}, v={v_int.size(-1)}, expected={head_dim}."
                )
            text_len = int(k_int.size(1))
            if int(v_int.size(1)) != text_len:
                raise RuntimeError(f"quantized-KV attention K/V seq mismatch: k={text_len}, v={v_int.size(1)}.")

            current_key_states = repeat_kv(key_states, num_key_value_groups)
            current_value_states = repeat_kv(value_states, num_key_value_groups)
            current_logits = torch.matmul(
                query_states.float(),
                current_key_states.float().transpose(2, 3),
            ) / math.sqrt(head_dim)
            cached_logits = torch.empty((bsz, num_heads, q_len, text_len), dtype=torch.float32, device=device)

            if text_len > 0:
                for batch_idx in range(bsz):
                    for head_idx in range(num_heads):
                        kv_head_idx = head_idx // num_key_value_groups
                        q_fp = query_states[batch_idx, head_idx].float()

                        fold_start = time.perf_counter()
                        q_eff = q_fp * scale_k.reshape(1, head_dim)
                        self._quant_kv_stat_add("q_scale_fold_time_s", time.perf_counter() - fold_start)

                        quant_start = time.perf_counter()
                        scale_q_eff = (q_eff.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
                        q_eff_int8 = torch.round(q_eff / scale_q_eff).clamp(-127, 127).to(torch.int8).contiguous()
                        q_eff_int8, original_m = _pad_int8_rows_to_multiple(q_eff_int8)
                        self._quant_kv_stat_add("q_eff_quant_time_s", time.perf_counter() - quant_start)

                        qk_start = time.perf_counter()
                        k_head_int8, original_n = _pad_int8_rows_to_multiple(k_int[kv_head_idx].contiguous())
                        logits_int = torch._int_mm(q_eff_int8, k_head_int8.t().contiguous())
                        if logits_int.size(0) != original_m or logits_int.size(1) != original_n:
                            logits_int = logits_int[:original_m, :original_n].contiguous()
                        self._quant_kv_stat_add("qk_int_mm_time_s", time.perf_counter() - qk_start)

                        dequant_start = time.perf_counter()
                        # scale_k is already folded into q_eff, so do not multiply scale_k here.
                        cached_logits[batch_idx, head_idx] = logits_int.float() * scale_q_eff / math.sqrt(head_dim)
                        self._quant_kv_stat_add("logits_dequant_time_s", time.perf_counter() - dequant_start)

            attn_weights = torch.cat([cached_logits, current_logits], dim=-1)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : text_len + current_len]
            softmax_start = time.perf_counter()
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
            self._quant_kv_stat_add("softmax_time_s", time.perf_counter() - softmax_start)
            attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)

            cached_probs = attn_weights[:, :, :, :text_len]
            if cfg.get("pv_compute_mode") == "int_pv":
                cached_output = self._quant_kv_int_pv_output(
                    cached_probs,
                    v_int,
                    scale_v,
                    num_key_value_groups,
                    dtype,
                    cfg,
                )
            else:
                cached_output = self._quant_kv_scale_delayed_v_output(
                    cached_probs,
                    v_int,
                    scale_v,
                    num_key_value_groups,
                    dtype,
                    record_stats=True,
                )
            current_probs = attn_weights[:, :, :, text_len:]
            current_output = torch.matmul(current_probs, current_value_states)
            attn_output = cached_output + current_output

            if not stats.get("example_shapes"):
                stats["example_shapes"] = {
                    "q": tuple(query_states.shape),
                    "k_int": tuple(k_int.shape),
                    "v_int": tuple(v_int.shape),
                    "scale_k": tuple(scale_k.shape),
                    "scale_v": tuple(scale_v.shape),
                    "logits": tuple(attn_weights.shape),
                }
            if not output_attentions:
                returned_attn_weights = None
            else:
                returned_attn_weights = attn_weights
            return attn_output, returned_attn_weights
        except _QuantKVPVError:
            raise
        except Exception as exc:
            return self._quant_kv_fallback_to_fp(
                query_states,
                key_states,
                value_states,
                text_kv,
                attention_mask,
                attention,
                num_key_value_groups,
                head_dim,
                reason=exc,
            )

    def _memory_kv_attention_breakdown(
        self,
        attention,
        hidden_states,
        attention_mask,
        position_ids,
        text_kv,
        output_attentions,
        cache_position,
        position_embeddings,
        g_layer_idx,
    ):
        bsz, q_len, _ = hidden_states.size()
        attention_config = attention.config
        num_heads = getattr(attention, "num_heads", attention_config.num_attention_heads)
        num_key_value_heads = getattr(attention, "num_key_value_heads", attention_config.num_key_value_heads)
        num_key_value_groups = getattr(attention, "num_key_value_groups", num_heads // num_key_value_heads)
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            head_dim = attention_config.hidden_size // num_heads

        qkv_start = self._stage_profile_start(hidden_states)
        query_states = attention.q_proj(hidden_states)
        key_states = attention.k_proj(hidden_states)
        value_states = attention.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        self._stage_profile_add("memory_kv_qkv_proj_s", qkv_start, query_states, g_layer_idx)

        rope_start = self._stage_profile_start(query_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if getattr(text_kv, "quantized_kv_attention", False) and self._quant_kv_attention_enabled():
            self._stage_profile_add("memory_kv_rope_cache_s", rope_start, key_states, g_layer_idx)
            attn_start = self._stage_profile_start(query_states)
            attn_output, attn_weights = self._memory_kv_quantized_attention(
                attention,
                query_states,
                key_states,
                value_states,
                text_kv,
                attention_mask,
                output_attentions,
                num_key_value_groups,
                head_dim,
            )
        else:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = text_kv.update(
                key_states, value_states, getattr(attention, "layer_idx", g_layer_idx), cache_kwargs
            )
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
            self._stage_profile_add("memory_kv_rope_cache_s", rope_start, key_states, g_layer_idx)

            attn_start = self._stage_profile_start(query_states)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)
            attn_output = torch.matmul(attn_weights, value_states)
        self._stage_profile_add("memory_kv_attn_scores_s", attn_start, attn_output, g_layer_idx)

        o_proj_start = self._stage_profile_start(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attention.o_proj(attn_output)
        self._stage_profile_add("memory_kv_o_proj_s", o_proj_start, attn_output, g_layer_idx)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, text_kv

    def _boundary_attention_breakdown(
        self,
        attention,
        hidden_states,
        attention_mask,
        position_ids,
        output_attentions,
        cache_position,
        position_embeddings,
        g_layer_idx,
    ):
        bsz, q_len, _ = hidden_states.size()
        attention_config = attention.config
        num_heads = getattr(attention, "num_heads", attention_config.num_attention_heads)
        num_key_value_heads = getattr(attention, "num_key_value_heads", attention_config.num_key_value_heads)
        num_key_value_groups = getattr(attention, "num_key_value_groups", num_heads // num_key_value_heads)
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            head_dim = attention_config.hidden_size // num_heads

        qkv_start = self._stage_profile_start(hidden_states)
        query_states = attention.q_proj(hidden_states)
        key_states = attention.k_proj(hidden_states)
        value_states = attention.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        self._stage_profile_add("boundary_qkv_proj_s", qkv_start, query_states, g_layer_idx)

        rope_start = self._stage_profile_start(query_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        self._stage_profile_add("boundary_rope_repeat_s", rope_start, key_states, g_layer_idx)

        attn_start = self._stage_profile_start(query_states)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)
        attn_output = torch.matmul(attn_weights, value_states)
        self._stage_profile_add("boundary_attn_scores_s", attn_start, attn_output, g_layer_idx)

        o_proj_start = self._stage_profile_start(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attention.o_proj(attn_output)
        self._stage_profile_add("boundary_o_proj_s", o_proj_start, attn_output, g_layer_idx)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    def _boundary_decoder_layer_breakdown(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        output_attentions,
        cache_position,
        position_embeddings,
        g_layer_idx,
    ):
        residual = hidden_states

        norm_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        self._stage_profile_add("boundary_input_norm_s", norm_start, hidden_states, g_layer_idx)

        attn_output, self_attn_weights = self._boundary_attention_breakdown(
            decoder_layer.self_attn,
            hidden_states,
            attention_mask,
            position_ids,
            output_attentions,
            cache_position,
            position_embeddings,
            g_layer_idx,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        post_norm_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        self._stage_profile_add("boundary_post_attn_norm_s", post_norm_start, hidden_states, g_layer_idx)

        mlp_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        self._stage_profile_add("boundary_mlp_s", mlp_start, hidden_states, g_layer_idx)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    def _memory_kv_decoder_layer_breakdown(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        text_kv,
        output_attentions,
        cache_position,
        position_embeddings,
        g_layer_idx,
    ):
        residual = hidden_states

        norm_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        self._stage_profile_add("memory_kv_input_norm_s", norm_start, hidden_states, g_layer_idx)

        attn_output, self_attn_weights, present_key_value = self._memory_kv_attention_breakdown(
            decoder_layer.self_attn,
            hidden_states,
            attention_mask,
            position_ids,
            text_kv,
            output_attentions,
            cache_position,
            position_embeddings,
            g_layer_idx,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        post_norm_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        self._stage_profile_add("memory_kv_post_attn_norm_s", post_norm_start, hidden_states, g_layer_idx)

        mlp_start = self._stage_profile_start(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        self._stage_profile_add("memory_kv_mlp_s", mlp_start, hidden_states, g_layer_idx)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (present_key_value,)
        return outputs

    def build_memory_text_kv_cache_item(
        self,
        boundary_hidden_states: torch.FloatTensor,
        text_len: int,
        output_attentions: Optional[bool] = None,
        partial_grad: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        memory_state = boundary_hidden_states[text_len:text_len + self.gofa_config.mem_token]
        if text_len == 0:
            text_kv = []
            for decoder_layer in self.layers[self.gnn_start_layer:self.config.num_hidden_layers]:
                attention = decoder_layer.self_attn
                num_kv_heads = attention.num_key_value_heads
                head_dim = getattr(attention, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
                text_kv.append(
                    {
                        "key": torch.empty(
                            (num_kv_heads, 0, head_dim),
                            dtype=boundary_hidden_states.dtype,
                            device="cpu",
                        ),
                        "value": torch.empty(
                            (num_kv_heads, 0, head_dim),
                            dtype=boundary_hidden_states.dtype,
                            device="cpu",
                        ),
                    }
                )
            return {
                "text_len": text_len,
                "memory_state": memory_state.detach().cpu(),
                "text_kv": text_kv,
            }
        text_hidden_states = boundary_hidden_states[:text_len].unsqueeze(0).contiguous()
        text_kv = []

        for decoder_layer in self.layers[self.gnn_start_layer:self.config.num_hidden_layers]:
            cache_position = torch.arange(0, text_len, device=text_hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask = self._make_block_causal_mask(
                text_len,
                text_len,
                0,
                text_hidden_states.dtype,
                text_hidden_states.device,
            )
            position_embeddings = self.rotary_emb(text_hidden_states, position_ids)
            layer_cache = _SingleLayerKVCache()
            if partial_grad:
                with torch.no_grad():
                    layer_outputs = self.llm_forward(
                        decoder_layer,
                        text_hidden_states,
                        causal_mask,
                        position_ids,
                        layer_cache,
                        output_attentions,
                        True,
                        cache_position,
                        position_embeddings,
                        flash_attn_kwargs,
                    )
            else:
                layer_outputs = self.llm_forward(
                    decoder_layer,
                    text_hidden_states,
                    causal_mask,
                    position_ids,
                    layer_cache,
                    output_attentions,
                    True,
                    cache_position,
                    position_embeddings,
                    flash_attn_kwargs,
                )
            text_kv.append(
                {
                    "key": layer_cache.key_states.squeeze(0).detach().cpu(),
                    "value": layer_cache.value_states.squeeze(0).detach().cpu(),
                }
            )
            text_hidden_states = layer_outputs[0]

        return {
            "text_len": text_len,
            "memory_state": memory_state.detach().cpu(),
            "text_kv": text_kv,
        }

    def forward_memory_with_text_kv(
        self,
        memory_states: torch.FloatTensor,
        text_kv_items,
        graph=None,
        map_node=None,
        output_attentions: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        cur_node_size = graph.num_node_feat if graph is not None else 0
        mapped_items = list(text_kv_items)

        if graph is not None and map_node:
            item_order = graph.node_map.detach().cpu().tolist() + list(range(cur_node_size, len(mapped_items)))
            memory_states = memory_states[item_order]
            mapped_items = [mapped_items[i] for i in item_order]
            cur_node_size = len(graph.node_map)

        self._stage_profile_increment("encoder_suffix_calls")
        for i, decoder_layer in enumerate(self.layers[self.gnn_start_layer:self.config.num_hidden_layers],
                                          start=self.gnn_start_layer):
            g_layer_idx = i - self.gnn_start_layer
            if graph is not None:
                gnn_start = self._stage_profile_start(memory_states)
                gnn_input = memory_states[:cur_node_size]
                gnn_edge_input = memory_states[cur_node_size:][graph.edge_map]
                output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                memory_states = torch.cat([output, memory_states[cur_node_size:]], dim=0)
                memory_states = memory_states.to(self.gofa_config.llama_dtype)
                self._stage_profile_add("encoder_gnn_layer_s", gnn_start, memory_states, g_layer_idx)

            llm_start = self._stage_profile_start(memory_states)
            next_memory_states = []
            for item_idx, item in enumerate(mapped_items):
                text_len = item["text_len"]
                text_kv_len = int(item.get("kv_text_len", text_len))
                mem_hidden_states = memory_states[item_idx:item_idx + 1]
                cache_position = torch.arange(
                    text_len,
                    text_len + self.gofa_config.mem_token,
                    device=mem_hidden_states.device,
                )
                position_ids = cache_position.unsqueeze(0)
                causal_mask = self._make_block_causal_mask(
                    self.gofa_config.mem_token,
                    text_kv_len + self.gofa_config.mem_token,
                    text_kv_len,
                    mem_hidden_states.dtype,
                    mem_hidden_states.device,
                )
                kv = item["text_kv"][g_layer_idx]
                kv_to_device_start = self._stage_profile_start(mem_hidden_states)
                use_quant_kv_attention = (
                    self._quant_kv_attention_enabled() and
                    isinstance(kv, dict) and
                    bool(kv.get("quantized", False))
                )
                if use_quant_kv_attention:
                    text_cache = _SingleLayerKVCache(
                        key_quant_payload=kv["key"],
                        value_quant_payload=kv["value"],
                    )
                    profile_ref = mem_hidden_states
                else:
                    text_cache = _SingleLayerKVCache(
                        kv["key"].unsqueeze(0).to(device=mem_hidden_states.device, dtype=mem_hidden_states.dtype),
                        kv["value"].unsqueeze(0).to(device=mem_hidden_states.device, dtype=mem_hidden_states.dtype),
                    )
                    profile_ref = text_cache.key_states
                if text_kv_len == 0:
                    self.quant_kv_attention_stats["kv_empty_count"] = (
                        int(self.quant_kv_attention_stats.get("kv_empty_count", 0)) + 1
                    )
                self._stage_profile_add(
                    "memory_kv_text_kv_to_device_s", kv_to_device_start, profile_ref, g_layer_idx
                )
                position_embeddings = self.rotary_emb(mem_hidden_states, position_ids)
                if use_quant_kv_attention or (
                        self._stage_profile_enabled() and getattr(self, "profile_memory_kv_transformer_breakdown", False)):
                    layer_outputs = self._memory_kv_decoder_layer_breakdown(
                        decoder_layer,
                        mem_hidden_states,
                        causal_mask,
                        position_ids,
                        text_cache,
                        output_attentions,
                        cache_position,
                        position_embeddings,
                        g_layer_idx,
                    )
                else:
                    layer_outputs = self.llm_forward(
                        decoder_layer,
                        mem_hidden_states,
                        causal_mask,
                        position_ids,
                        text_cache,
                        output_attentions,
                        True,
                        cache_position,
                        position_embeddings,
                        flash_attn_kwargs,
                    )
                next_memory_states.append(layer_outputs[0])
            memory_states = torch.cat(next_memory_states, dim=0)
            self._stage_profile_add("encoder_suffix_transformer_layer_s", llm_start, memory_states, g_layer_idx)

        norm_start = self._stage_profile_start(memory_states)
        memory_states = self.norm(memory_states)
        self._stage_profile_add("encoder_norm_s", norm_start, memory_states)
        return memory_states, mapped_items

    def forward_llm_prefix(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        partial_grad: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, None, output_attentions
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attns = () if output_attentions else None

        self._stage_profile_increment("encoder_prefix_calls")
        prefix_start = self._stage_profile_start(hidden_states)
        for decoder_layer in self.layers[:self.gnn_start_layer]:
            if partial_grad:
                with torch.no_grad():
                    layer_outputs = self.llm_forward(
                        decoder_layer, hidden_states, causal_mask, position_ids, None, output_attentions,
                        False, cache_position, position_embeddings, flash_attn_kwargs
                    )
            else:
                layer_outputs = self.llm_forward(
                    decoder_layer, hidden_states, causal_mask, position_ids, None, output_attentions,
                    False, cache_position, position_embeddings, flash_attn_kwargs
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        self._stage_profile_add("encoder_prefix_transformer_s", prefix_start, hidden_states)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def forward_from_gnn_boundary(
        self,
        boundary_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph=None,
        mem_mask=None,
        partial_grad=None,
        map_node=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cache_position = torch.arange(0, boundary_hidden_states.shape[1], device=boundary_hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, boundary_hidden_states, cache_position, None, output_attentions
        )
        hidden_states = boundary_hidden_states
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attns = () if output_attentions else None
        cur_node_size = graph.num_node_feat if graph is not None else 0

        self._stage_profile_increment("encoder_suffix_calls")
        for i, decoder_layer in enumerate(self.layers[self.gnn_start_layer:self.config.num_hidden_layers],
                                          start=self.gnn_start_layer):
            g_layer_idx = i - self.gnn_start_layer
            if graph is not None:
                gnn_start = self._stage_profile_start(hidden_states)
                if g_layer_idx == 0 and map_node:
                    hidden_states = torch.cat(
                        [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                    mem_mask = torch.cat([mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]], dim=0)
                    if causal_mask is not None:
                        causal_mask = torch.cat(
                            [causal_mask[:cur_node_size][graph.node_map], causal_mask[cur_node_size:]], dim=0)
                    cur_node_size = len(graph.node_map)
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.gofa_config.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
                hidden_states = hidden_states.to(self.gofa_config.llama_dtype)
                self._stage_profile_add("encoder_gnn_layer_s", gnn_start, hidden_states, g_layer_idx)

            llm_start = self._stage_profile_start(hidden_states)
            if self._stage_profile_enabled() and getattr(self, "profile_memory_kv_transformer_breakdown", False):
                layer_outputs = self._boundary_decoder_layer_breakdown(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                    g_layer_idx,
                )
            else:
                layer_outputs = self.llm_forward(
                    decoder_layer, hidden_states, causal_mask, position_ids, None, output_attentions,
                    False, cache_position, position_embeddings, flash_attn_kwargs
                )
            hidden_states = layer_outputs[0]
            self._stage_profile_add("encoder_suffix_transformer_layer_s", llm_start, hidden_states, g_layer_idx)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        norm_start = self._stage_profile_start(hidden_states)
        hidden_states = self.norm(hidden_states)
        self._stage_profile_add("encoder_norm_s", norm_start, hidden_states)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph=None,
        mem_mask=None,
        partial_grad=None,
        map_node=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Copied from Huggingface Mistral implementation.

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            raise ValueError("You cannot specify input_ids for GOFA, please construct input embeddings manually")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        ##########################################################
        #             Key model implementation of GOFA           #
        ##########################################################

        cur_node_size = graph.num_node_feat if graph is not None else 0

        if graph is not None:
            self._stage_profile_increment("encoder_full_calls")
        prefix_start = self._stage_profile_start(hidden_states) if graph is not None else None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - (self.config.num_hidden_layers - self.gofa_config.num_layers)
            if g_layer_idx >= 0 and graph is not None:
                if g_layer_idx == 0:
                    self._stage_profile_add("encoder_prefix_transformer_s", prefix_start, hidden_states)
                    prefix_start = None
                gnn_start = self._stage_profile_start(hidden_states)
                if g_layer_idx == 0 and map_node:
                    hidden_states = torch.cat(
                        [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                    mem_mask = torch.cat([mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]], dim=0)
                    if causal_mask is not None:
                        causal_mask = torch.cat(
                            [causal_mask[:cur_node_size][graph.node_map], causal_mask[cur_node_size:]], dim=0)
                    cur_node_size = len(graph.node_map)
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.gofa_config.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
                hidden_states = hidden_states.to(self.gofa_config.llama_dtype)
                self._stage_profile_add("encoder_gnn_layer_s", gnn_start, hidden_states, g_layer_idx)
            llm_start = self._stage_profile_start(hidden_states) if g_layer_idx >= 0 and graph is not None else None
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    layer_outputs = self.llm_forward(decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, flash_attn_kwargs)
            else:
                layer_outputs = self.llm_forward(decoder_layer, hidden_states, causal_mask, position_ids,
                                                 past_key_values, output_attentions, use_cache, cache_position,
                                                 position_embeddings, flash_attn_kwargs)

            hidden_states = layer_outputs[0]
            if g_layer_idx >= 0 and graph is not None:
                self._stage_profile_add("encoder_suffix_transformer_layer_s", llm_start, hidden_states, g_layer_idx)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        norm_start = self._stage_profile_start(hidden_states) if graph is not None else None
        hidden_states = self.norm(hidden_states)
        self._stage_profile_add("encoder_norm_s", norm_start, hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def llm_forward(self, decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, flash_attn_kwargs):
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, causal_mask,
                position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, )
        else:
            layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **flash_attn_kwargs, )
        return layer_outputs



class LLMGraphCombiner(torch.nn.Module):
    def __init__(self, init_theta=0.0, hidden_size=4096):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([init_theta]))
        self.norm = MistralRMSNorm(hidden_size)

    def forward(self, target_feat, additional_feat, val_mask=None):
        alpha = self.theta.tanh().to(additional_feat.dtype)
        if val_mask is None:
            return target_feat + additional_feat * alpha
        # print(alpha)
        # print((target_feat[val_mask]**2).sum(dim=-1).mean())
        # print((additional_feat ** 2).sum(dim=-1).mean())
        output = torch.zeros_like(target_feat, dtype=additional_feat.dtype)
        output[val_mask] = additional_feat.view(-1, additional_feat.size()[-1]) * alpha

        # val_multiplier = torch.zeros_like(target_feat)
        # val_multiplier[torch.logical_not(val_mask)] = 1
        # val_multiplier[val_mask] = alpha

        return self.norm(target_feat + output)


class GOFAMistralParallelModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig, gofa_config):
        super().__init__(config)
        self.gofa_config = gofa_config

        self.g_layers = nn.ModuleList()
        self.g_layers.append(nn.ModuleList([GOFADecoderLayer(gofa_config, i) for i in range(gofa_config.num_layers)]))
        self.g_layers.append(nn.ModuleList([LLMGraphCombiner() for _ in range(gofa_config.num_layers)]))
        self.g_layers.append(nn.ModuleList(
            [MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(gofa_config.num_layers)]))
        self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

    def align_weight(self):
        n_layers = len(self.layers)
        inactive_layers = n_layers - len(self.g_layers[0])
        partial_state_dict = OrderedDict()
        source_dict = self.layers.state_dict()
        for layer_name in source_dict:
            name_split = layer_name.split(".")
            layer_ind = int(name_split[0])
            if layer_ind >= inactive_layers:
                name_split[0] = str(layer_ind - inactive_layers)
                if name_split[2] in ["v_proj", "q_proj", "k_proj", "o_proj"]:
                    name_split[2] = "g"+name_split[2]
                partial_state_dict[".".join(name_split)] = source_dict[layer_name]

        self.g_layers[0].load_state_dict(partial_state_dict)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph=None,
        mem_mask=None,
        partial_grad=None,
        map_node=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            raise ValueError("You cannot specify input_ids for GOFA, please construct input embeddings manually")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        cur_node_size = graph.num_node_feat if graph is not None else 0

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - (self.config.num_hidden_layers - self.gofa_config.num_layers)
            if g_layer_idx == 0 and map_node:
                hidden_states = torch.cat(
                    [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                mem_mask = torch.cat([mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]], dim=0)
                if causal_mask is not None:
                    causal_mask = torch.cat(
                        [causal_mask[:cur_node_size][graph.node_map], causal_mask[cur_node_size:]], dim=0)
                cur_node_size = len(graph.node_map)
            if g_layer_idx >= 0 and graph is not None:
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.gofa_config.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[0][g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = self.g_layers[2][g_layer_idx](output)
                graph_output = torch.cat([output, mem_repr[cur_node_size:]],
                                         dim=0)  # gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)  # gnn_output[mem_mask] = output.view(-1, output.size()[-1])  # hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
                graph_output = graph_output.to(self.gofa_config.llama_dtype)
            else:
                graph_output = None
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    layer_outputs = self.llm_forward(decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, flash_attn_kwargs)
            else:
                layer_outputs = self.llm_forward(decoder_layer, hidden_states, causal_mask, position_ids,
                                                 past_key_values, output_attentions, use_cache, cache_position,
                                                 position_embeddings, flash_attn_kwargs)

            hidden_states = layer_outputs[0]
            if graph_output is not None:
                hidden_states = self.g_layers[1][g_layer_idx](hidden_states, graph_output, mem_mask)
                hidden_states = hidden_states.to(self.gofa_config.llama_dtype)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if graph is not None:
            hidden_states = self.g_layers[3](hidden_states)
            hidden_states = hidden_states.to(self.gofa_config.llama_dtype)
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def llm_forward(self, decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, flash_attn_kwargs):
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, causal_mask,
                position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, )
        else:
            layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **flash_attn_kwargs, )
        return layer_outputs

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class GOFAMistralForCausalLM(MistralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules = ["g_layers"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, gofa_config):
        super().__init__(config)
        if gofa_config.fuse_type == "parallel":
            self.model = GOFAMistralParallelModel(config, gofa_config)
        else:
            self.model = GOFAMistralModel(config, gofa_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0, graph=None, mem_mask=None, partial_grad=None, map_node=None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=map_node,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_base_model(self):
        return self
