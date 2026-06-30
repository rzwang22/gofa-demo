# example code for running inference with fine-tuned checkpoint
import atexit
from contextlib import ExitStack
from typing import Any, Dict, Optional

import hashlib
import json
import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from datetime import datetime
from transformers import MistralConfig

from modules.gofa.gofa_icae import MistralICAE
from collections import OrderedDict
from safetensors.torch import load_file
from modules.utils import safe_download_hf_file
from modules.gofa.cache_policy import (
    BASE_DELTA,
    build_scheme_b_load_policy,
    local_neighbors_by_hop,
    local_node_degrees,
    summarize_load_policy,
)
from modules.gofa.cache_quant import (
    QUANT_BASE_FORMAT,
    QUANT_DELTA_FORMAT,
    estimate_quantized_tensor_bits,
    reconstruct_scheme_b_cache_item,
)
from modules.gofa.activation_quant import (
    activation_quant_context,
    maybe_create_suffix_transformer_activation_quantizer,
)
from modules.gofa.activation_observer import (
    activation_observer_context,
    maybe_create_suffix_transformer_activation_observer,
)
from modules.gofa.weight_quant import (
    maybe_create_suffix_transformer_weight_quantizer,
    weight_quant_context,
)
from modules.gofa.int_gemm_quant import (
    int_gemm_context,
    maybe_create_suffix_transformer_int_gemm_quantizer,
)

###################################################################
#                 Configurations                                  #
###################################################################


class GOFAMistralConfig(MistralConfig):
    def __init__(self, dim=4096, num_layers=6, mem_token=128, head=32, add_self_loops=True, dropout=0.0,
                 llama_dtype=torch.float16, gnn_hidden_act="relu", gnn_mlp_type="gp", gnn_type="index", position_encoding="none", pretraining_tp=0, gating=True, interleave=True, mp_att="concat", trainable_layer=5, fuse_type="interleave", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mem_token = mem_token
        self.head = head
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.num_layers = num_layers
        self.llama_dtype = llama_dtype
        self.gnn_hidden_act = gnn_hidden_act
        self.gnn_mlp_type = gnn_mlp_type
        self.gnn_type = gnn_type
        self.pretraining_tp = pretraining_tp
        self.position_encoding = position_encoding
        self.interleave = interleave
        self.gating = gating
        self.mp_att = mp_att
        self.trainable_layer = trainable_layer
        self.fuse_type = fuse_type


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    attn_implementation: str = field(default="eager", metadata={"help": "Mistral attention implementation"})
    lora_r: int = field(default=512, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    mem_size: int = field(default=128, metadata={"help": "Memory size"}, )
    dec_lora: bool = field(default=False, metadata={"help": "Whether using lora in the decoder LLM"})
    checkpoint_dir: str = field(default="./cache_data/model/")
    use_encoder_cache: bool = field(default=False, metadata={"help": "Cache graph-independent encoder prefix states"})
    encoder_cache_dir: str = field(default="./cache_data/encoder_cache")
    encoder_cache_tag: str = field(default="")
    encoder_cache_skip_nog: bool = field(default=False, metadata={"help": "Do not cache NOG/prompt node states"})
    encoder_cache_mode: str = field(default="boundary", metadata={"help": "Encoder cache mode: boundary or memory_kv"})
    encoder_cache_manifest: Optional[Dict[str, Any]] = field(default_factory=dict)
    encoder_cache_manifest_enabled: Optional[bool] = field(default=None)
    encoder_cache_manifest_output_path: Optional[str] = field(default=None)
    encoder_cache_manifest_append: Optional[bool] = field(default=None)
    encoder_cache_manifest_log_interval: Optional[int] = field(default=None)
    encoder_cache_verify: bool = field(default=False, metadata={"help": "Compare memory_kv cache output against the full encoder path"})
    encoder_cache_verify_tolerance: float = field(default=1e-3, metadata={"help": "Strict max-absolute tolerance for exact verification reporting"})
    encoder_cache_verify_mean_tolerance: float = field(default=3e-2)
    encoder_cache_verify_p99_tolerance: float = field(default=2.5e-1)
    encoder_cache_verify_relative_l2_tolerance: float = field(default=8e-2)
    encoder_cache_verify_max_tolerance: float = field(default=1.5)
    encoder_cache_verify_log_interval: int = field(default=1)
    encoder_cache_verify_quantile_sample_size: int = field(default=1_000_000)
    profile_stage_times: bool = field(default=False, metadata={"help": "Synchronize and profile encoder/decoder stages"})
    profile_stage_log_interval: int = field(default=50, metadata={"help": "Log stage timing every N decoder calls"})
    profile_memory_kv_transformer_breakdown: bool = field(
        default=False,
        metadata={"help": "Break down cached suffix transformer time into attention and MLP parts"},
    )
    scheme_b_quant: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_quant_enabled: Optional[bool] = field(default=None)
    scheme_b_quant_base_bits: Optional[int] = field(default=None)
    scheme_b_quant_delta_bits: Optional[int] = field(default=None)
    scheme_b_quant_memory_base_bits: Optional[int] = field(default=None)
    scheme_b_quant_key_base_bits: Optional[int] = field(default=None)
    scheme_b_quant_value_base_bits: Optional[int] = field(default=None)
    scheme_b_quant_memory_delta_bits: Optional[int] = field(default=None)
    scheme_b_quant_key_delta_bits: Optional[int] = field(default=None)
    scheme_b_quant_value_delta_bits: Optional[int] = field(default=None)
    scheme_b_quant_static_high_ratio: Optional[float] = field(default=None)
    scheme_b_quant_static_mid_ratio: Optional[float] = field(default=None)
    scheme_b_quant_target_aware_delta: Optional[bool] = field(default=None)
    scheme_b_quant_target_aware_policy: Optional[str] = field(default=None)
    scheme_b_quant_target_delta_hops: Optional[int] = field(default=None)
    scheme_b_quant_keep_target_edges: Optional[bool] = field(default=None)
    scheme_b_quant_local_degree_top_ratio: Optional[float] = field(default=None)
    scheme_b_quant_local_degree_threshold: Optional[Any] = field(default=None)
    scheme_b_quant_max_delta_items_per_batch: Optional[Any] = field(default=None)
    scheme_b_quant_cache_dir: Optional[str] = field(default=None)
    scheme_b_quant_fake_quant: Optional[bool] = field(default=None)
    scheme_b_quant_debug_zero_base: Optional[bool] = field(default=None)
    scheme_b_quant_strict: Optional[bool] = field(default=None)
    scheme_b_quant_load_memory_delta: Optional[bool] = field(default=None)
    scheme_b_quant_load_key_delta: Optional[bool] = field(default=None)
    scheme_b_quant_load_value_delta: Optional[bool] = field(default=None)
    scheme_b_weight_quant: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_weight_quant_enabled: Optional[bool] = field(default=None)
    scheme_b_weight_quant_bits: Optional[int] = field(default=None)
    scheme_b_weight_quant_target: Optional[str] = field(default=None)
    scheme_b_weight_quant_fake_quant: Optional[bool] = field(default=None)
    scheme_b_weight_quant_quantize_attention: Optional[bool] = field(default=None)
    scheme_b_weight_quant_quantize_mlp: Optional[bool] = field(default=None)
    scheme_b_weight_quant_quantize_layernorm: Optional[bool] = field(default=None)
    scheme_b_weight_quant_log_quantized_modules: Optional[bool] = field(default=None)
    scheme_b_activation_quant: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_activation_quant_enabled: Optional[bool] = field(default=None)
    scheme_b_activation_quant_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_target: Optional[str] = field(default=None)
    scheme_b_activation_quant_fake_quant: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_attention: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_q_proj: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_k_proj: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_v_proj: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_o_proj: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_mlp: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_qkv_outputs: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_attn_output: Optional[bool] = field(default=None)
    scheme_b_activation_quant_quantize_mlp_output: Optional[bool] = field(default=None)
    scheme_b_activation_quant_per_token: Optional[bool] = field(default=None)
    scheme_b_activation_quant_clip_ratio: Optional[float] = field(default=None)
    scheme_b_activation_quant_q_proj_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_k_proj_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_v_proj_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_o_proj_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_mlp_bits: Optional[int] = field(default=None)
    scheme_b_activation_quant_log_quantized_modules: Optional[bool] = field(default=None)
    scheme_b_int_gemm: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_int_gemm_enabled: Optional[bool] = field(default=None)
    scheme_b_int_gemm_target: Optional[str] = field(default=None)
    scheme_b_int_gemm_weight_bits: Optional[int] = field(default=None)
    scheme_b_int_gemm_activation_bits: Optional[int] = field(default=None)
    scheme_b_int_gemm_backend: Optional[str] = field(default=None)
    scheme_b_int_gemm_quantize_attention: Optional[bool] = field(default=None)
    scheme_b_int_gemm_quantize_mlp: Optional[bool] = field(default=None)
    scheme_b_int_gemm_quantize_layernorm: Optional[bool] = field(default=None)
    scheme_b_int_gemm_fallback_to_fake_quant: Optional[bool] = field(default=None)
    scheme_b_int_gemm_log_modules: Optional[bool] = field(default=None)
    scheme_b_int_gemm_log_interval: Optional[int] = field(default=None)
    scheme_b_quant_kv_attention: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_quant_kv_attention_enabled: Optional[bool] = field(default=None)
    scheme_b_quant_kv_attention_backend: Optional[str] = field(default=None)
    scheme_b_quant_kv_attention_key_scale_fold_into_q: Optional[bool] = field(default=None)
    scheme_b_quant_kv_attention_quantize_query_bits: Optional[int] = field(default=None)
    scheme_b_quant_kv_attention_key_bits: Optional[int] = field(default=None)
    scheme_b_quant_kv_attention_value_bits: Optional[int] = field(default=None)
    scheme_b_quant_kv_attention_use_int_qk: Optional[bool] = field(default=None)
    scheme_b_quant_kv_attention_pv_compute_mode: Optional[str] = field(default=None)
    scheme_b_quant_kv_attention_fallback_to_fp_attention: Optional[bool] = field(default=None)
    scheme_b_quant_kv_attention_log_interval: Optional[int] = field(default=None)
    scheme_b_activation_observer: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_activation_observer_enabled: Optional[bool] = field(default=None)
    scheme_b_activation_observer_output_dir: Optional[str] = field(default=None)
    scheme_b_activation_observer_max_batches: Optional[int] = field(default=None)
    scheme_b_activation_observer_max_items_per_module: Optional[int] = field(default=None)
    scheme_b_activation_observer_target: Optional[str] = field(default=None)
    scheme_b_activation_observer_layers: Optional[Any] = field(default=None)
    scheme_b_activation_observer_projections: Optional[Any] = field(default=None)
    scheme_b_activation_observer_save_tensor: Optional[bool] = field(default=None)
    scheme_b_activation_observer_save_stats: Optional[bool] = field(default=None)
    scheme_b_activation_observer_sample_tokens: Optional[int] = field(default=None)
    scheme_b_activation_observer_sample_channels: Optional[int] = field(default=None)
    scheme_b_activation_observer_compute_quant_error: Optional[bool] = field(default=None)
    scheme_b_activation_observer_quant_bits: Optional[Any] = field(default=None)
    scheme_b_activation_observer_per_token: Optional[bool] = field(default=None)
    scheme_b_activation_observer_clip_ratio: Optional[float] = field(default=None)
    scheme_b_activation_observer_log_interval: Optional[int] = field(default=None)
    scheme_b_ablation: Optional[Dict[str, Any]] = field(default_factory=dict)
    scheme_b_ablation_enabled: Optional[bool] = field(default=None)
    scheme_b_ablation_mode: Optional[str] = field(default=None)
    scheme_b_ablation_zero_memory_state: Optional[bool] = field(default=None)
    scheme_b_ablation_zero_text_kv: Optional[bool] = field(default=None)
    scheme_b_ablation_zero_edge_cache: Optional[bool] = field(default=None)
    scheme_b_ablation_keep_target_edges: Optional[bool] = field(default=None)
    scheme_b_ablation_log_interval: Optional[int] = field(default=None)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=False)
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length per node. Sequences will be right padded (and possibly truncated)."}, )
    fixed_mem_size: int = field(default=128, metadata={"help": "Enalbing the fixed mem size."}, )
    mean_compression_rate: int = field(default=4, metadata={"help": "Mean compression rate; default=4"}, )
    min_tokens_for_lm: int = field(default=64, metadata={"help": "Minimum tokens for lm objective learning"}, )
    leave_tokens_for_lm: int = field(default=8, metadata={"help": "Leave some tokens without loss for lm objective"}, )
    lm_ratio: float = field(default=0.0, metadata={"help": "Ratio for LM training."}, )
    add_special_token_for_lm: bool = field(default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"}, )
    restore_from: str = field(default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"})


###################################################################
#                 Model                                           #
###################################################################


class GOFAMistral(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MistralICAE(model_args, training_args, gofa_args)  # restored llama2-7b-chat model
        dir = safe_download_hf_file("sggetao/icae", "mistral_7b_ft_icae.safetensors", model_args.checkpoint_dir, repo_type=None)
        state_dict = load_file(dir)  # change the path for your model
        new_state_dict = OrderedDict()
        for layer_name, weight in state_dict.items():
            new_state_dict[layer_name.replace("default", "encadapt")] = weight
        model.load_state_dict(new_state_dict, strict=False)
        # model.merge_lora()
        self.model_args = model_args
        self.dec_lora = model_args.dec_lora
        self.mem_tokens = list(range(model.vocab_size, model.vocab_size + model_args.mem_size))
        self.mem_size = model_args.mem_size
        self.model = model
        self.profile_stage_times = bool(model_args.profile_stage_times)
        self.profile_stage_log_interval = model_args.profile_stage_log_interval
        self.profile_memory_kv_transformer_breakdown = bool(model_args.profile_memory_kv_transformer_breakdown)
        self.stage_profile_reports = 0
        self.decoder_stage_calls = 0
        self.decoder_stage_time_s = 0.0
        self.encoder_cache_enabled = bool(model_args.use_encoder_cache)
        self.encoder_cache_dir = model_args.encoder_cache_dir
        self.encoder_cache_mode = model_args.encoder_cache_mode
        self.scheme_b_quant = self._normalize_scheme_b_quant_config(model_args)
        self.scheme_b_quant_enabled = bool(self.scheme_b_quant["enabled"])
        self.scheme_b_quant_path_example_logged = False
        self.scheme_b_quant_stats = self._new_scheme_b_quant_stats()
        self.scheme_b_weight_quant = self._normalize_scheme_b_weight_quant_config(model_args)
        self.scheme_b_weight_quant_enabled = bool(self.scheme_b_weight_quant["enabled"])
        self.scheme_b_weight_quantizer = None
        self.scheme_b_weight_quant_stats = {
            "quantized_module_count": 0,
            "quantized_weight_original_bytes": 0,
            "quantized_weight_effective_bytes": 0,
            "compression_ratio": 0.0,
        }
        self.scheme_b_activation_quant = self._normalize_scheme_b_activation_quant_config(model_args)
        self.scheme_b_activation_quant_enabled = bool(self.scheme_b_activation_quant["enabled"])
        self.scheme_b_activation_quantizer = None
        self.scheme_b_activation_quant_last_logged_call_count = -1
        self.scheme_b_int_gemm = self._normalize_scheme_b_int_gemm_config(model_args)
        self.scheme_b_int_gemm_enabled = bool(self.scheme_b_int_gemm["enabled"])
        self.scheme_b_int_gemm_quantizer = None
        self.scheme_b_int_gemm_last_logged_call_count = -1
        self.scheme_b_int_gemm_last_logged_signature = None
        self.scheme_b_quant_kv_attention = self._normalize_scheme_b_quant_kv_attention_config(model_args)
        self.scheme_b_quant_kv_attention_enabled = bool(self.scheme_b_quant_kv_attention["enabled"])
        self.scheme_b_quant_kv_attention_last_logged_signature = None
        self.scheme_b_activation_observer = self._normalize_scheme_b_activation_observer_config(model_args)
        self.scheme_b_activation_observer_enabled = bool(self.scheme_b_activation_observer["enabled"])
        self.scheme_b_activation_observer_instance = None
        self.encoder_cache_manifest = self._normalize_encoder_cache_manifest_config(model_args)
        self.encoder_cache_manifest_enabled = bool(self.encoder_cache_manifest["enabled"])
        self.encoder_cache_manifest_items = OrderedDict()
        self.encoder_cache_manifest_skip_items = OrderedDict()
        self.encoder_cache_manifest_samples = 0
        self.encoder_cache_manifest_dumps = 0
        self.encoder_cache_manifest_append_loaded = False
        self.encoder_cache_manifest_existing_items = OrderedDict()
        self.encoder_cache_manifest_existing_skip_items = OrderedDict()
        self.encoder_cache_manifest_existing_total_samples = 0
        self.scheme_b_ablation = self._normalize_scheme_b_ablation_config(model_args)
        self.scheme_b_ablation_enabled = bool(self.scheme_b_ablation["enabled"])
        self.scheme_b_ablation_calls = 0
        self.encoder_cache_verify = bool(model_args.encoder_cache_verify)
        self.encoder_cache_verify_tolerance = model_args.encoder_cache_verify_tolerance
        self.encoder_cache_verify_mean_tolerance = model_args.encoder_cache_verify_mean_tolerance
        self.encoder_cache_verify_p99_tolerance = model_args.encoder_cache_verify_p99_tolerance
        self.encoder_cache_verify_relative_l2_tolerance = model_args.encoder_cache_verify_relative_l2_tolerance
        self.encoder_cache_verify_max_tolerance = model_args.encoder_cache_verify_max_tolerance
        self.encoder_cache_verify_log_interval = model_args.encoder_cache_verify_log_interval
        self.encoder_cache_verify_quantile_sample_size = model_args.encoder_cache_verify_quantile_sample_size
        self.encoder_cache_verify_calls = 0
        self.encoder_cache_verify_exact_failures = 0
        self.encoder_cache_verify_practical_failures = 0
        self.encoder_cache_verify_running = {
            "mean_abs_sum": 0.0,
            "p99_abs_sum": 0.0,
            "relative_l2_sum": 0.0,
            "max_abs_worst": 0.0,
        }
        self.encoder_cache_calls = 0
        self.encoder_cache_hits = 0
        self.encoder_cache_misses = 0
        self.encoder_cache_skips = 0
        self.encoder_cache_timing = {
            "load_s": 0.0,
            "quant_load_s": 0.0,
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "dequant_s": 0.0,
            "delta_load_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
            "cache_size_bytes": 0.0,
        }
        self.encoder_full_calls = 0
        self.encoder_full_time_s = 0.0
        self.encoder_cache_namespace = self._build_encoder_cache_namespace(dir, model_args, training_args, gofa_args)
        base_model = self.model.icae.get_base_model().model
        if hasattr(base_model, "profile_stage_times"):
            base_model.profile_stage_times = self.profile_stage_times
            base_model.profile_memory_kv_transformer_breakdown = self.profile_memory_kv_transformer_breakdown
            base_model.reset_stage_profile()
        if self.scheme_b_int_gemm_enabled and (self.scheme_b_weight_quant_enabled or self.scheme_b_activation_quant_enabled):
            raise ValueError(
                "scheme_b_int_gemm.enabled=True cannot be combined with "
                "scheme_b_weight_quant.enabled=True or scheme_b_activation_quant.enabled=True. "
                "Disable fake quant hooks to avoid double quantization."
            )
        if self.scheme_b_int_gemm_enabled:
            print(
                "GOFA scheme-B suffix Transformer W4A8 integer GEMM enabled: "
                f"target={self.scheme_b_int_gemm['target']}, "
                f"weight_bits={self.scheme_b_int_gemm['weight_bits']}, "
                f"activation_bits={self.scheme_b_int_gemm['activation_bits']}, "
                f"backend={self.scheme_b_int_gemm['backend']}, "
                f"quantize_attention={self.scheme_b_int_gemm['quantize_attention']}, "
                f"quantize_mlp={self.scheme_b_int_gemm['quantize_mlp']}, "
                f"quantize_layernorm={self.scheme_b_int_gemm['quantize_layernorm']}, "
                f"fallback_to_fake_quant={self.scheme_b_int_gemm['fallback_to_fake_quant']}, "
                f"log_interval={self.scheme_b_int_gemm['log_interval']}"
            )
            self.scheme_b_int_gemm_quantizer = maybe_create_suffix_transformer_int_gemm_quantizer(
                base_model,
                self.scheme_b_int_gemm,
                logger=print,
            )
        if hasattr(base_model, "configure_quant_kv_attention"):
            base_model.configure_quant_kv_attention(self.scheme_b_quant_kv_attention)
        if self.scheme_b_weight_quant_enabled:
            print(
                "GOFA scheme-B suffix Transformer weight quantization enabled: "
                f"target={self.scheme_b_weight_quant['target']}, "
                f"bits={self.scheme_b_weight_quant['bits']}, "
                f"fake_quant={self.scheme_b_weight_quant['fake_quant']}, "
                f"quantize_attention={self.scheme_b_weight_quant['quantize_attention']}, "
                f"quantize_mlp={self.scheme_b_weight_quant['quantize_mlp']}, "
                f"quantize_layernorm={self.scheme_b_weight_quant['quantize_layernorm']}"
            )
            self.scheme_b_weight_quantizer = maybe_create_suffix_transformer_weight_quantizer(
                base_model,
                self.scheme_b_weight_quant,
                logger=print,
            )
            self.scheme_b_weight_quant_stats = dict(self.scheme_b_weight_quantizer.stats)
        if self.scheme_b_activation_observer_enabled:
            print(
                "GOFA scheme-B suffix Transformer activation observer enabled: "
                f"target={self.scheme_b_activation_observer['target']}, "
                f"output_dir={self.scheme_b_activation_observer['output_dir']}, "
                f"max_batches={self.scheme_b_activation_observer['max_batches']}, "
                f"max_items_per_module={self.scheme_b_activation_observer['max_items_per_module']}, "
                f"layers={self.scheme_b_activation_observer['layers']}, "
                f"projections={self.scheme_b_activation_observer['projections']}, "
                f"save_tensor={self.scheme_b_activation_observer['save_tensor']}, "
                f"save_stats={self.scheme_b_activation_observer['save_stats']}, "
                f"sample_tokens={self.scheme_b_activation_observer['sample_tokens']}, "
                f"sample_channels={self.scheme_b_activation_observer['sample_channels']}, "
                f"compute_quant_error={self.scheme_b_activation_observer['compute_quant_error']}, "
                f"quant_bits={self.scheme_b_activation_observer['quant_bits']}, "
                f"per_token={self.scheme_b_activation_observer['per_token']}, "
                f"clip_ratio={self.scheme_b_activation_observer['clip_ratio']}"
            )
            self.scheme_b_activation_observer_instance = maybe_create_suffix_transformer_activation_observer(
                base_model,
                self.scheme_b_activation_observer,
                task=self._encoder_cache_manifest_task_names(),
                cache_mode=self.encoder_cache_mode,
                logger=print,
            )
        if self.scheme_b_activation_quant_enabled:
            print(
                "GOFA scheme-B suffix Transformer activation quantization enabled: "
                f"target={self.scheme_b_activation_quant['target']}, "
                f"bits={self.scheme_b_activation_quant['bits']}, "
                f"fake_quant={self.scheme_b_activation_quant['fake_quant']}, "
                f"quantize_attention={self.scheme_b_activation_quant['quantize_attention']}, "
                f"quantize_q_proj={self.scheme_b_activation_quant['quantize_q_proj']}, "
                f"quantize_k_proj={self.scheme_b_activation_quant['quantize_k_proj']}, "
                f"quantize_v_proj={self.scheme_b_activation_quant['quantize_v_proj']}, "
                f"quantize_o_proj={self.scheme_b_activation_quant['quantize_o_proj']}, "
                f"quantize_mlp={self.scheme_b_activation_quant['quantize_mlp']}, "
                f"quantize_qkv_outputs={self.scheme_b_activation_quant['quantize_qkv_outputs']}, "
                f"quantize_attn_output={self.scheme_b_activation_quant['quantize_attn_output']}, "
                f"quantize_mlp_output={self.scheme_b_activation_quant['quantize_mlp_output']}, "
                f"per_token={self.scheme_b_activation_quant['per_token']}, "
                f"clip_ratio={self.scheme_b_activation_quant['clip_ratio']}, "
                f"q_proj_bits={self.scheme_b_activation_quant['q_proj_bits']}, "
                f"k_proj_bits={self.scheme_b_activation_quant['k_proj_bits']}, "
                f"v_proj_bits={self.scheme_b_activation_quant['v_proj_bits']}, "
                f"o_proj_bits={self.scheme_b_activation_quant['o_proj_bits']}, "
                f"mlp_bits={self.scheme_b_activation_quant['mlp_bits']}"
            )
            self.scheme_b_activation_quantizer = maybe_create_suffix_transformer_activation_quantizer(
                base_model,
                self.scheme_b_activation_quant,
                logger=print,
            )
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token
        for param in self.model.icae.parameters():
            param.requires_grad = False
        for param in self.model.icae.get_base_model().model.g_layers.parameters():
            param.requires_grad = True
        if self.dec_lora:
            for name, param in self.model.icae.named_parameters():
                if "default" in name:
                    param.requires_grad = True
        if self.encoder_cache_enabled:
            if self.encoder_cache_mode not in {"boundary", "memory_kv"}:
                raise ValueError("encoder_cache_mode must be either 'boundary' or 'memory_kv'.")
            if self.scheme_b_quant_enabled and self.encoder_cache_mode != "memory_kv":
                print("GOFA scheme-B quant cache is only valid for encoder_cache_mode=memory_kv; disabling quant cache.")
                self.scheme_b_quant_enabled = False
            if self.scheme_b_quant_kv_attention_enabled and self.encoder_cache_mode != "memory_kv":
                print("GOFA quantized-KV attention is only valid for encoder_cache_mode=memory_kv; disabling it.")
                self.scheme_b_quant_kv_attention_enabled = False
                self.scheme_b_quant_kv_attention["enabled"] = False
                if hasattr(base_model, "configure_quant_kv_attention"):
                    base_model.configure_quant_kv_attention(self.scheme_b_quant_kv_attention)
            if self.encoder_cache_manifest_enabled and self.encoder_cache_mode != "memory_kv":
                print("GOFA encoder cache manifest is only valid for encoder_cache_mode=memory_kv; disabling manifest.")
                self.encoder_cache_manifest_enabled = False
            if self.scheme_b_ablation_enabled and self.encoder_cache_mode != "memory_kv":
                print("GOFA scheme-B ablation is only valid for encoder_cache_mode=memory_kv; disabling ablation.")
                self.scheme_b_ablation_enabled = False
            if gofa_args.fuse_type != "interleave":
                print("GOFA encoder cache is only implemented for fuse_type=interleave; disabling cache.")
                self.encoder_cache_enabled = False
            else:
                os.makedirs(self._encoder_cache_root(), exist_ok=True)
                print(
                    "GOFA encoder cache enabled: "
                    f"dir={self._encoder_cache_root()}, boundary_layer="
                    f"{self.model.icae.get_base_model().model.gnn_start_layer}, "
                    f"skip_nog={model_args.encoder_cache_skip_nog}, "
                    f"mode={self.encoder_cache_mode}"
                )
                if self.encoder_cache_mode == "memory_kv":
                    print(
                        "GOFA encoder memory/text-KV cache enabled: "
                        "stores layer-26 memory tokens plus text-side KV for the last GNN/LLM layers"
                    )
                    if self.encoder_cache_verify:
                        print(
                            "GOFA encoder memory/text-KV cache verification enabled: "
                            f"strict_max_tol={self.encoder_cache_verify_tolerance}, "
                            f"mean_tol={self.encoder_cache_verify_mean_tolerance}, "
                            f"p99_tol={self.encoder_cache_verify_p99_tolerance}, "
                            f"rel_l2_tol={self.encoder_cache_verify_relative_l2_tolerance}, "
                            f"max_tol={self.encoder_cache_verify_max_tolerance}, "
                            f"log_interval={self.encoder_cache_verify_log_interval}, "
                            f"quantile_sample_size={self.encoder_cache_verify_quantile_sample_size}"
                        )
                if self.scheme_b_quant_enabled:
                    print(
                        "GOFA scheme-B quant cache enabled: "
                        f"dir={self._encoder_quant_cache_root()}, "
                        f"base_bits={self.scheme_b_quant['base_bits']}, "
                        f"delta_bits={self.scheme_b_quant['delta_bits']}, "
                        f"memory_base_bits={self.scheme_b_quant['memory_base_bits']}, "
                        f"key_base_bits={self.scheme_b_quant['key_base_bits']}, "
                        f"value_base_bits={self.scheme_b_quant['value_base_bits']}, "
                        f"memory_delta_bits={self.scheme_b_quant['memory_delta_bits']}, "
                        f"key_delta_bits={self.scheme_b_quant['key_delta_bits']}, "
                        f"value_delta_bits={self.scheme_b_quant['value_delta_bits']}, "
                        f"target_aware_delta={self.scheme_b_quant['target_aware_delta']}, "
                        f"target_aware_policy={self.scheme_b_quant['target_aware_policy']}, "
                        f"target_delta_hops={self.scheme_b_quant['target_delta_hops']}, "
                        f"keep_target_edges={self.scheme_b_quant['keep_target_edges']}, "
                        f"local_degree_top_ratio={self.scheme_b_quant['local_degree_top_ratio']}, "
                        f"local_degree_threshold={self.scheme_b_quant['local_degree_threshold']}, "
                        f"max_delta_items_per_batch={self.scheme_b_quant['max_delta_items_per_batch']}, "
                        f"load_memory_delta={self.scheme_b_quant['load_memory_delta']}, "
                        f"load_key_delta={self.scheme_b_quant['load_key_delta']}, "
                        f"load_value_delta={self.scheme_b_quant['load_value_delta']}, "
                        f"fake_quant={self.scheme_b_quant['fake_quant']}, "
                        f"debug_zero_base={self.scheme_b_quant['debug_zero_base']}, "
                        f"strict={self.scheme_b_quant['strict']}"
                    )
                    print(
                        "GOFA scheme-B quant cache path roots: "
                        f"cache_tag={self.encoder_cache_namespace}, "
                        f"full_cache_root={self._encoder_cache_root()}, "
                        f"quant_cache_root={self._encoder_quant_cache_root()}, "
                        f"quant_base_tag_root={self._encoder_quant_cache_tag_root(delta=False)}, "
                        f"quant_delta_tag_root={self._encoder_quant_cache_tag_root(delta=True)}, "
                        "full_cache_path_example=<full_cache_root>/<cache_key[:2]>/<cache_key>.pt, "
                        "quant_base_cache_path_example=<quant_cache_root>/<cache_tag>/<cache_key[:2]>/<cache_key>.pt, "
                        "quant_delta_cache_path_example=<quant_cache_root>/delta/<cache_tag>/<cache_key[:2]>/<cache_key>.pt"
                    )
                    self._validate_scheme_b_quant_cache_root()
                if self.scheme_b_quant_kv_attention_enabled:
                    if not self.scheme_b_quant_enabled:
                        raise ValueError(
                            "scheme_b_quant_kv_attention.enabled=True requires scheme_b_quant.enabled=True "
                            "for memory_kv cache payloads."
                        )
                    if not self.scheme_b_quant["strict"]:
                        print(
                            "GOFA quantized-KV attention warning: scheme_b_quant.strict=False permits fallback "
                            "before quantized-KV attention sees payloads."
                        )
                    if self.scheme_b_quant["load_key_delta"] or self.scheme_b_quant["load_value_delta"]:
                        print(
                            "GOFA quantized-KV attention warning: first version supports base-only text-side K/V. "
                            "If a policy selects key/value delta, reconstruct will raise unless fallback is enabled."
                        )
                    if int(self.scheme_b_quant["key_base_bits"]) != int(self.scheme_b_quant_kv_attention["key_bits"]):
                        raise ValueError(
                            "scheme_b_quant_kv_attention.key_bits must match scheme_b_quant.key_base_bits: "
                            f"{self.scheme_b_quant_kv_attention['key_bits']} != {self.scheme_b_quant['key_base_bits']}."
                        )
                    if int(self.scheme_b_quant["value_base_bits"]) != int(self.scheme_b_quant_kv_attention["value_bits"]):
                        raise ValueError(
                            "scheme_b_quant_kv_attention.value_bits must match scheme_b_quant.value_base_bits: "
                            f"{self.scheme_b_quant_kv_attention['value_bits']} != {self.scheme_b_quant['value_base_bits']}."
                        )
                    print(
                        "GOFA quantized-KV attention enabled: "
                        f"backend={self.scheme_b_quant_kv_attention['backend']}, "
                        f"key_scale_fold_into_q={self.scheme_b_quant_kv_attention['key_scale_fold_into_q']}, "
                        f"quantize_query_bits={self.scheme_b_quant_kv_attention['quantize_query_bits']}, "
                        f"key_bits={self.scheme_b_quant_kv_attention['key_bits']}, "
                        f"value_bits={self.scheme_b_quant_kv_attention['value_bits']}, "
                        f"use_int_qk={self.scheme_b_quant_kv_attention['use_int_qk']}, "
                        f"pv_compute_mode={self.scheme_b_quant_kv_attention['pv_compute_mode']}, "
                        f"fallback_to_fp_attention={self.scheme_b_quant_kv_attention['fallback_to_fp_attention']}"
                    )
                if self.encoder_cache_manifest_enabled:
                    if not self.encoder_cache_manifest["output_path"]:
                        raise ValueError(
                            "encoder_cache_manifest.enabled=True requires encoder_cache_manifest.output_path."
                        )
                    print(
                        "GOFA encoder cache manifest enabled: "
                        f"output_path={self.encoder_cache_manifest['output_path']}, "
                        f"append={self.encoder_cache_manifest['append']}, "
                        f"log_interval={self.encoder_cache_manifest['log_interval']}, "
                        f"cache_tag={self.encoder_cache_namespace}, "
                        f"full_cache_root={os.path.abspath(self.encoder_cache_dir)}"
                    )
                    atexit.register(self._maybe_dump_encoder_cache_manifest, current_batch_seen=0, force=True)
                if self.scheme_b_ablation_enabled:
                    print(
                        "GOFA scheme-B cache ablation enabled: "
                        f"mode={self.scheme_b_ablation['mode']}, "
                        f"zero_memory_state={self.scheme_b_ablation['zero_memory_state']}, "
                        f"zero_text_kv={self.scheme_b_ablation['zero_text_kv']}, "
                        f"zero_edge_cache={self.scheme_b_ablation['zero_edge_cache']}, "
                        f"keep_target_edges={self.scheme_b_ablation['keep_target_edges']}, "
                        f"log_interval={self.scheme_b_ablation['log_interval']}"
                    )
        if self.profile_stage_times:
            print(
                "GOFA stage profiler enabled: "
                f"log_interval={self.profile_stage_log_interval}, "
                "timings include cuda synchronization overhead"
            )
            if self.profile_memory_kv_transformer_breakdown:
                print(
                    "GOFA suffix transformer breakdown enabled: "
                    "diagnostic mode with extra synchronization overhead"
                )

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        self.model.icae.set_adapter("encadapt")
        for param in self.model.icae.parameters():
            param.requires_grad = False

    def load_pretrained(self, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = safe_download_hf_file("WFRaain/GOFA", "mistral_qamag03_best_ckpt.pth", self.model_args.checkpoint_dir,
                                                     repo_type=None)
        self.load_partial(pretrained_path)

    def save_partial(self, save_dir):
        """
        Save the GNN and lora weight (if available).
        """
        state_dict = self.model.icae.get_base_model().model.g_layers.state_dict()
        full_state_dict = self.state_dict()
        for k in full_state_dict:
            if "default" in k:
                state_dict[k] = full_state_dict[k]
        torch.save(state_dict, save_dir)

    def load_partial(self, load_dir):
        """
        Load the GNN and lora weight (if available).
        """
        state_dict = torch.load(load_dir, map_location="cpu")
        normalized_state_dict = OrderedDict()
        for key, value in state_dict.items():
            normalized_state_dict[key] = value
            if key.startswith("llm_model."):
                normalized_state_dict[key[len("llm_model."):]] = value

        raw_lora_keys = [key for key in state_dict if "default" in key and "lora" in key.lower()]
        loadable_lora_keys = [
            key for key in normalized_state_dict
            if not key.startswith("llm_model.") and "default" in key and "lora" in key.lower()
        ]
        print(
            "Loaded partial checkpoint keys: "
            f"total={len(state_dict)}, normalized={len(normalized_state_dict)}, "
            f"raw_decoder_lora={len(raw_lora_keys)}, loadable_decoder_lora={len(loadable_lora_keys)}"
        )
        missing_keys, unexpected_keys = self.model.icae.get_base_model().model.g_layers.load_state_dict(state_dict, strict=False)
        print("GNN module is missing the following keys:", missing_keys)
        if unexpected_keys:
            print("GNN module skipped non-GNN keys:", len(unexpected_keys))
        missing_keys, unexpected_keys = self.load_state_dict(normalized_state_dict, strict=False)
        if self.dec_lora:
            missing_lora = [key for key in missing_keys if "default" in key and "lora" in key.lower()]
            print("Decoder LoRA keys in checkpoint:", len(raw_lora_keys))
            print("Decoder LoRA keys after prefix normalization:", len(loadable_lora_keys))
            print("Decoder LoRA keys still missing after load:", len(missing_lora))
            if not loadable_lora_keys:
                raise RuntimeError("dec_lora=True, but no loadable decoder LoRA keys were found in the checkpoint.")
            if missing_lora:
                raise RuntimeError("dec_lora=True, but decoder LoRA keys were not fully loaded.")
        if unexpected_keys:
            print("Full GOFA module skipped unexpected keys:", len(unexpected_keys))

    def _normalize_scheme_b_quant_config(self, model_args):
        cfg = {
            "enabled": False,
            "base_bits": 8,
            "delta_bits": 4,
            "memory_base_bits": None,
            "key_base_bits": None,
            "value_base_bits": None,
            "memory_delta_bits": None,
            "key_delta_bits": None,
            "value_delta_bits": None,
            "static_high_ratio": 0.10,
            "static_mid_ratio": 0.40,
            "target_aware_delta": True,
            "target_aware_policy": "target_1hop",
            "target_delta_hops": 1,
            "keep_target_edges": True,
            "local_degree_top_ratio": 0.0,
            "local_degree_threshold": None,
            "max_delta_items_per_batch": None,
            "cache_dir": "",
            "fake_quant": True,
            "debug_zero_base": False,
            "strict": True,
            "load_memory_delta": True,
            "load_key_delta": True,
            "load_value_delta": True,
        }
        nested = getattr(model_args, "scheme_b_quant", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_quant_enabled",
            "base_bits": "scheme_b_quant_base_bits",
            "delta_bits": "scheme_b_quant_delta_bits",
            "memory_base_bits": "scheme_b_quant_memory_base_bits",
            "key_base_bits": "scheme_b_quant_key_base_bits",
            "value_base_bits": "scheme_b_quant_value_base_bits",
            "memory_delta_bits": "scheme_b_quant_memory_delta_bits",
            "key_delta_bits": "scheme_b_quant_key_delta_bits",
            "value_delta_bits": "scheme_b_quant_value_delta_bits",
            "static_high_ratio": "scheme_b_quant_static_high_ratio",
            "static_mid_ratio": "scheme_b_quant_static_mid_ratio",
            "target_aware_delta": "scheme_b_quant_target_aware_delta",
            "target_aware_policy": "scheme_b_quant_target_aware_policy",
            "target_delta_hops": "scheme_b_quant_target_delta_hops",
            "keep_target_edges": "scheme_b_quant_keep_target_edges",
            "local_degree_top_ratio": "scheme_b_quant_local_degree_top_ratio",
            "local_degree_threshold": "scheme_b_quant_local_degree_threshold",
            "max_delta_items_per_batch": "scheme_b_quant_max_delta_items_per_batch",
            "cache_dir": "scheme_b_quant_cache_dir",
            "fake_quant": "scheme_b_quant_fake_quant",
            "debug_zero_base": "scheme_b_quant_debug_zero_base",
            "strict": "scheme_b_quant_strict",
            "load_memory_delta": "scheme_b_quant_load_memory_delta",
            "load_key_delta": "scheme_b_quant_load_key_delta",
            "load_value_delta": "scheme_b_quant_load_value_delta",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["base_bits"] = int(cfg["base_bits"])
        cfg["delta_bits"] = int(cfg["delta_bits"])
        for bits_key, fallback_key in (
            ("memory_base_bits", "base_bits"),
            ("key_base_bits", "base_bits"),
            ("value_base_bits", "base_bits"),
            ("memory_delta_bits", "delta_bits"),
            ("key_delta_bits", "delta_bits"),
            ("value_delta_bits", "delta_bits"),
        ):
            value = cfg[bits_key]
            if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
                value = None
            cfg[bits_key] = int(cfg[fallback_key]) if value is None else int(value)
            if cfg[bits_key] not in {2, 4, 8, 16}:
                raise ValueError(f"scheme_b_quant.{bits_key} must be one of 2, 4, 8, 16.")
        cfg["static_high_ratio"] = float(cfg["static_high_ratio"])
        cfg["static_mid_ratio"] = float(cfg["static_mid_ratio"])
        cfg["target_aware_delta"] = bool(cfg["target_aware_delta"])
        cfg["target_aware_policy"] = str(cfg["target_aware_policy"] or "target_1hop")
        if cfg["target_aware_policy"] not in {
            "target_only",
            "target_1hop",
            "local_degree_top",
            "target_1hop_local_degree",
            "all_delta",
        }:
            raise ValueError(
                "scheme_b_quant.target_aware_policy must be one of target_only, target_1hop, "
                "local_degree_top, target_1hop_local_degree, all_delta."
            )
        cfg["target_delta_hops"] = max(int(cfg["target_delta_hops"]), 0)
        cfg["keep_target_edges"] = bool(cfg["keep_target_edges"])
        cfg["local_degree_top_ratio"] = max(float(cfg["local_degree_top_ratio"]), 0.0)
        if isinstance(cfg["local_degree_threshold"], str) and cfg["local_degree_threshold"].strip().lower() in {"", "none", "null"}:
            cfg["local_degree_threshold"] = None
        if cfg["local_degree_threshold"] is not None:
            cfg["local_degree_threshold"] = float(cfg["local_degree_threshold"])
        if isinstance(cfg["max_delta_items_per_batch"], str) and cfg["max_delta_items_per_batch"].strip().lower() in {"", "none", "null"}:
            cfg["max_delta_items_per_batch"] = None
        if cfg["max_delta_items_per_batch"] is not None:
            cfg["max_delta_items_per_batch"] = max(int(cfg["max_delta_items_per_batch"]), 0)
        cfg["cache_dir"] = str(cfg["cache_dir"] or "")
        cfg["fake_quant"] = bool(cfg["fake_quant"])
        cfg["debug_zero_base"] = bool(cfg["debug_zero_base"])
        cfg["strict"] = bool(cfg["strict"])
        cfg["load_memory_delta"] = bool(cfg["load_memory_delta"])
        cfg["load_key_delta"] = bool(cfg["load_key_delta"])
        cfg["load_value_delta"] = bool(cfg["load_value_delta"])
        return cfg

    def _normalize_encoder_cache_manifest_config(self, model_args):
        cfg = {
            "enabled": False,
            "output_path": "",
            "append": False,
            "log_interval": 20,
        }
        nested = getattr(model_args, "encoder_cache_manifest", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "encoder_cache_manifest_enabled",
            "output_path": "encoder_cache_manifest_output_path",
            "append": "encoder_cache_manifest_append",
            "log_interval": "encoder_cache_manifest_log_interval",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["output_path"] = str(cfg["output_path"] or "")
        cfg["append"] = bool(cfg["append"])
        cfg["log_interval"] = max(int(cfg["log_interval"]), 1)
        return cfg

    def _normalize_scheme_b_weight_quant_config(self, model_args):
        cfg = {
            "enabled": False,
            "bits": 4,
            "target": "suffix_transformer",
            "fake_quant": True,
            "quantize_attention": True,
            "quantize_mlp": True,
            "quantize_layernorm": False,
            "log_quantized_modules": True,
        }
        nested = getattr(model_args, "scheme_b_weight_quant", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_weight_quant_enabled",
            "bits": "scheme_b_weight_quant_bits",
            "target": "scheme_b_weight_quant_target",
            "fake_quant": "scheme_b_weight_quant_fake_quant",
            "quantize_attention": "scheme_b_weight_quant_quantize_attention",
            "quantize_mlp": "scheme_b_weight_quant_quantize_mlp",
            "quantize_layernorm": "scheme_b_weight_quant_quantize_layernorm",
            "log_quantized_modules": "scheme_b_weight_quant_log_quantized_modules",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["bits"] = int(cfg["bits"])
        if cfg["bits"] not in {4, 8}:
            raise ValueError("scheme_b_weight_quant.bits must be 4 or 8.")
        cfg["target"] = str(cfg["target"] or "suffix_transformer")
        if cfg["target"] != "suffix_transformer":
            raise ValueError("scheme_b_weight_quant.target currently supports only 'suffix_transformer'.")
        cfg["fake_quant"] = bool(cfg["fake_quant"])
        cfg["quantize_attention"] = bool(cfg["quantize_attention"])
        cfg["quantize_mlp"] = bool(cfg["quantize_mlp"])
        cfg["quantize_layernorm"] = bool(cfg["quantize_layernorm"])
        cfg["log_quantized_modules"] = bool(cfg["log_quantized_modules"])
        return cfg

    def _normalize_optional_activation_bits(self, value, field_name):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        value = int(value)
        if value not in {4, 8}:
            raise ValueError(f"scheme_b_activation_quant.{field_name} must be None, 4, or 8.")
        return value

    def _normalize_csv_int_list(self, value, default, field_name):
        if value is None:
            return list(default)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return list(default)
            values = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = [value]
        try:
            return [int(item) for item in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scheme_b_activation_observer.{field_name} must be an int list or comma-separated ints.") from exc

    def _normalize_csv_str_list(self, value, default, field_name):
        if value is None:
            return list(default)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return list(default)
            values = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value if str(item).strip()]
        else:
            values = [str(value).strip()]
        if not values:
            raise ValueError(f"scheme_b_activation_observer.{field_name} must not be empty.")
        return values

    def _normalize_scheme_b_activation_quant_config(self, model_args):
        cfg = {
            "enabled": False,
            "bits": 8,
            "target": "suffix_transformer",
            "fake_quant": True,
            "quantize_attention": True,
            "quantize_q_proj": True,
            "quantize_k_proj": True,
            "quantize_v_proj": True,
            "quantize_o_proj": True,
            "quantize_mlp": True,
            "quantize_qkv_outputs": False,
            "quantize_attn_output": False,
            "quantize_mlp_output": False,
            "per_token": True,
            "clip_ratio": 1.0,
            "q_proj_bits": None,
            "k_proj_bits": None,
            "v_proj_bits": None,
            "o_proj_bits": None,
            "mlp_bits": None,
            "log_quantized_modules": True,
        }
        nested = getattr(model_args, "scheme_b_activation_quant", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_activation_quant_enabled",
            "bits": "scheme_b_activation_quant_bits",
            "target": "scheme_b_activation_quant_target",
            "fake_quant": "scheme_b_activation_quant_fake_quant",
            "quantize_attention": "scheme_b_activation_quant_quantize_attention",
            "quantize_q_proj": "scheme_b_activation_quant_quantize_q_proj",
            "quantize_k_proj": "scheme_b_activation_quant_quantize_k_proj",
            "quantize_v_proj": "scheme_b_activation_quant_quantize_v_proj",
            "quantize_o_proj": "scheme_b_activation_quant_quantize_o_proj",
            "quantize_mlp": "scheme_b_activation_quant_quantize_mlp",
            "quantize_qkv_outputs": "scheme_b_activation_quant_quantize_qkv_outputs",
            "quantize_attn_output": "scheme_b_activation_quant_quantize_attn_output",
            "quantize_mlp_output": "scheme_b_activation_quant_quantize_mlp_output",
            "per_token": "scheme_b_activation_quant_per_token",
            "clip_ratio": "scheme_b_activation_quant_clip_ratio",
            "q_proj_bits": "scheme_b_activation_quant_q_proj_bits",
            "k_proj_bits": "scheme_b_activation_quant_k_proj_bits",
            "v_proj_bits": "scheme_b_activation_quant_v_proj_bits",
            "o_proj_bits": "scheme_b_activation_quant_o_proj_bits",
            "mlp_bits": "scheme_b_activation_quant_mlp_bits",
            "log_quantized_modules": "scheme_b_activation_quant_log_quantized_modules",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["bits"] = int(cfg["bits"])
        if cfg["bits"] not in {4, 8}:
            raise ValueError("scheme_b_activation_quant.bits must be 4 or 8.")
        cfg["target"] = str(cfg["target"] or "suffix_transformer")
        if cfg["target"] != "suffix_transformer":
            raise ValueError("scheme_b_activation_quant.target currently supports only 'suffix_transformer'.")
        cfg["fake_quant"] = bool(cfg["fake_quant"])
        cfg["quantize_attention"] = bool(cfg["quantize_attention"])
        cfg["quantize_q_proj"] = bool(cfg["quantize_q_proj"])
        cfg["quantize_k_proj"] = bool(cfg["quantize_k_proj"])
        cfg["quantize_v_proj"] = bool(cfg["quantize_v_proj"])
        cfg["quantize_o_proj"] = bool(cfg["quantize_o_proj"])
        if not cfg["quantize_attention"]:
            cfg["quantize_q_proj"] = False
            cfg["quantize_k_proj"] = False
            cfg["quantize_v_proj"] = False
            cfg["quantize_o_proj"] = False
        cfg["quantize_mlp"] = bool(cfg["quantize_mlp"])
        cfg["quantize_qkv_outputs"] = bool(cfg["quantize_qkv_outputs"])
        cfg["quantize_attn_output"] = bool(cfg["quantize_attn_output"])
        cfg["quantize_mlp_output"] = bool(cfg["quantize_mlp_output"])
        cfg["per_token"] = bool(cfg["per_token"])
        cfg["clip_ratio"] = float(cfg["clip_ratio"])
        if cfg["clip_ratio"] <= 0.0 or cfg["clip_ratio"] > 1.0:
            raise ValueError("scheme_b_activation_quant.clip_ratio must be in (0, 1].")
        cfg["q_proj_bits"] = self._normalize_optional_activation_bits(cfg["q_proj_bits"], "q_proj_bits")
        cfg["k_proj_bits"] = self._normalize_optional_activation_bits(cfg["k_proj_bits"], "k_proj_bits")
        cfg["v_proj_bits"] = self._normalize_optional_activation_bits(cfg["v_proj_bits"], "v_proj_bits")
        cfg["o_proj_bits"] = self._normalize_optional_activation_bits(cfg["o_proj_bits"], "o_proj_bits")
        cfg["mlp_bits"] = self._normalize_optional_activation_bits(cfg["mlp_bits"], "mlp_bits")
        cfg["log_quantized_modules"] = bool(cfg["log_quantized_modules"])
        return cfg

    def _normalize_scheme_b_int_gemm_config(self, model_args):
        cfg = {
            "enabled": False,
            "target": "suffix_transformer",
            "weight_bits": 4,
            "activation_bits": 8,
            "backend": "torch_int_mm",
            "quantize_attention": True,
            "quantize_mlp": True,
            "quantize_layernorm": False,
            "fallback_to_fake_quant": False,
            "log_modules": True,
            "log_interval": 20,
        }
        nested = getattr(model_args, "scheme_b_int_gemm", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_int_gemm_enabled",
            "target": "scheme_b_int_gemm_target",
            "weight_bits": "scheme_b_int_gemm_weight_bits",
            "activation_bits": "scheme_b_int_gemm_activation_bits",
            "backend": "scheme_b_int_gemm_backend",
            "quantize_attention": "scheme_b_int_gemm_quantize_attention",
            "quantize_mlp": "scheme_b_int_gemm_quantize_mlp",
            "quantize_layernorm": "scheme_b_int_gemm_quantize_layernorm",
            "fallback_to_fake_quant": "scheme_b_int_gemm_fallback_to_fake_quant",
            "log_modules": "scheme_b_int_gemm_log_modules",
            "log_interval": "scheme_b_int_gemm_log_interval",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["target"] = str(cfg["target"] or "suffix_transformer")
        if cfg["target"] != "suffix_transformer":
            raise ValueError("scheme_b_int_gemm.target currently supports only 'suffix_transformer'.")
        cfg["weight_bits"] = int(cfg["weight_bits"])
        if cfg["weight_bits"] != 4:
            raise ValueError("scheme_b_int_gemm.weight_bits currently supports only 4.")
        cfg["activation_bits"] = int(cfg["activation_bits"])
        if cfg["activation_bits"] != 8:
            raise ValueError("scheme_b_int_gemm.activation_bits currently supports only 8.")
        cfg["backend"] = str(cfg["backend"] or "torch_int_mm")
        if cfg["backend"] != "torch_int_mm":
            raise ValueError("scheme_b_int_gemm.backend currently supports only 'torch_int_mm'.")
        cfg["quantize_attention"] = bool(cfg["quantize_attention"])
        cfg["quantize_mlp"] = bool(cfg["quantize_mlp"])
        cfg["quantize_layernorm"] = bool(cfg["quantize_layernorm"])
        cfg["fallback_to_fake_quant"] = bool(cfg["fallback_to_fake_quant"])
        cfg["log_modules"] = bool(cfg["log_modules"])
        cfg["log_interval"] = max(int(cfg["log_interval"]), 1)
        return cfg

    def _normalize_scheme_b_quant_kv_attention_config(self, model_args):
        cfg = {
            "enabled": False,
            "backend": "torch_int_mm_qscale_fold",
            "key_scale_fold_into_q": True,
            "quantize_query_bits": 8,
            "key_bits": 4,
            "value_bits": 4,
            "use_int_qk": True,
            "pv_compute_mode": "scale_delayed_v",
            "fallback_to_fp_attention": False,
            "log_interval": 20,
        }
        nested = getattr(model_args, "scheme_b_quant_kv_attention", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_quant_kv_attention_enabled",
            "backend": "scheme_b_quant_kv_attention_backend",
            "key_scale_fold_into_q": "scheme_b_quant_kv_attention_key_scale_fold_into_q",
            "quantize_query_bits": "scheme_b_quant_kv_attention_quantize_query_bits",
            "key_bits": "scheme_b_quant_kv_attention_key_bits",
            "value_bits": "scheme_b_quant_kv_attention_value_bits",
            "use_int_qk": "scheme_b_quant_kv_attention_use_int_qk",
            "pv_compute_mode": "scheme_b_quant_kv_attention_pv_compute_mode",
            "fallback_to_fp_attention": "scheme_b_quant_kv_attention_fallback_to_fp_attention",
            "log_interval": "scheme_b_quant_kv_attention_log_interval",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["backend"] = str(cfg["backend"] or "torch_int_mm_qscale_fold")
        if cfg["backend"] != "torch_int_mm_qscale_fold":
            raise ValueError("scheme_b_quant_kv_attention.backend currently supports only 'torch_int_mm_qscale_fold'.")
        cfg["key_scale_fold_into_q"] = bool(cfg["key_scale_fold_into_q"])
        if not cfg["key_scale_fold_into_q"]:
            raise ValueError("scheme_b_quant_kv_attention.key_scale_fold_into_q must be True.")
        cfg["quantize_query_bits"] = int(cfg["quantize_query_bits"])
        if cfg["quantize_query_bits"] != 8:
            raise ValueError("scheme_b_quant_kv_attention.quantize_query_bits currently supports only 8.")
        cfg["key_bits"] = int(cfg["key_bits"])
        cfg["value_bits"] = int(cfg["value_bits"])
        if cfg["key_bits"] not in {2, 4}:
            raise ValueError("scheme_b_quant_kv_attention.key_bits must be 2 or 4.")
        if cfg["value_bits"] not in {2, 4}:
            raise ValueError("scheme_b_quant_kv_attention.value_bits must be 2 or 4.")
        cfg["use_int_qk"] = bool(cfg["use_int_qk"])
        if not cfg["use_int_qk"]:
            raise ValueError("scheme_b_quant_kv_attention.use_int_qk must be True.")
        cfg["pv_compute_mode"] = str(cfg["pv_compute_mode"] or "scale_delayed_v")
        if cfg["pv_compute_mode"] != "scale_delayed_v":
            raise ValueError("scheme_b_quant_kv_attention.pv_compute_mode currently supports only 'scale_delayed_v'.")
        cfg["fallback_to_fp_attention"] = bool(cfg["fallback_to_fp_attention"])
        cfg["log_interval"] = max(int(cfg["log_interval"]), 1)
        return cfg

    def _normalize_scheme_b_activation_observer_config(self, model_args):
        cfg = {
            "enabled": False,
            "output_dir": "",
            "max_batches": 2,
            "max_items_per_module": 4,
            "target": "suffix_transformer",
            "layers": [26, 29, 31],
            "projections": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"],
            "save_tensor": True,
            "save_stats": True,
            "sample_tokens": 512,
            "sample_channels": 256,
            "compute_quant_error": True,
            "quant_bits": [4, 8],
            "per_token": True,
            "clip_ratio": 1.0,
            "log_interval": 20,
        }
        nested = getattr(model_args, "scheme_b_activation_observer", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_activation_observer_enabled",
            "output_dir": "scheme_b_activation_observer_output_dir",
            "max_batches": "scheme_b_activation_observer_max_batches",
            "max_items_per_module": "scheme_b_activation_observer_max_items_per_module",
            "target": "scheme_b_activation_observer_target",
            "layers": "scheme_b_activation_observer_layers",
            "projections": "scheme_b_activation_observer_projections",
            "save_tensor": "scheme_b_activation_observer_save_tensor",
            "save_stats": "scheme_b_activation_observer_save_stats",
            "sample_tokens": "scheme_b_activation_observer_sample_tokens",
            "sample_channels": "scheme_b_activation_observer_sample_channels",
            "compute_quant_error": "scheme_b_activation_observer_compute_quant_error",
            "quant_bits": "scheme_b_activation_observer_quant_bits",
            "per_token": "scheme_b_activation_observer_per_token",
            "clip_ratio": "scheme_b_activation_observer_clip_ratio",
            "log_interval": "scheme_b_activation_observer_log_interval",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["output_dir"] = str(cfg["output_dir"] or "")
        if cfg["enabled"] and not cfg["output_dir"]:
            raise ValueError("scheme_b_activation_observer.output_dir must be set when observer is enabled.")
        cfg["max_batches"] = max(int(cfg["max_batches"]), 1)
        cfg["max_items_per_module"] = max(int(cfg["max_items_per_module"]), 1)
        cfg["target"] = str(cfg["target"] or "suffix_transformer")
        if cfg["target"] != "suffix_transformer":
            raise ValueError("scheme_b_activation_observer.target currently supports only 'suffix_transformer'.")
        cfg["layers"] = self._normalize_csv_int_list(cfg["layers"], [26, 29, 31], "layers")
        cfg["projections"] = self._normalize_csv_str_list(
            cfg["projections"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"],
            "projections",
        )
        valid_projections = {"q_proj", "k_proj", "v_proj", "o_proj", "mlp", "gate_proj", "up_proj", "down_proj"}
        invalid_projections = [projection for projection in cfg["projections"] if projection not in valid_projections]
        if invalid_projections:
            raise ValueError(f"scheme_b_activation_observer.projections has unsupported values: {invalid_projections}")
        cfg["save_tensor"] = bool(cfg["save_tensor"])
        cfg["save_stats"] = bool(cfg["save_stats"])
        cfg["sample_tokens"] = max(int(cfg["sample_tokens"]), 1)
        cfg["sample_channels"] = max(int(cfg["sample_channels"]), 1)
        cfg["compute_quant_error"] = bool(cfg["compute_quant_error"])
        cfg["quant_bits"] = self._normalize_csv_int_list(cfg["quant_bits"], [4, 8], "quant_bits")
        invalid_bits = [bit for bit in cfg["quant_bits"] if bit not in {4, 8}]
        if invalid_bits:
            raise ValueError(f"scheme_b_activation_observer.quant_bits must contain only 4 and 8; got {invalid_bits}.")
        cfg["per_token"] = bool(cfg["per_token"])
        cfg["clip_ratio"] = float(cfg["clip_ratio"])
        if cfg["clip_ratio"] <= 0.0 or cfg["clip_ratio"] > 1.0:
            raise ValueError("scheme_b_activation_observer.clip_ratio must be in (0, 1].")
        cfg["log_interval"] = max(int(cfg["log_interval"]), 1)
        return cfg

    def _normalize_scheme_b_ablation_config(self, model_args):
        cfg = {
            "enabled": False,
            "mode": "none",
            "zero_memory_state": True,
            "zero_text_kv": True,
            "zero_edge_cache": True,
            "keep_target_edges": False,
            "log_interval": 20,
        }
        nested = getattr(model_args, "scheme_b_ablation", None)
        if isinstance(nested, dict):
            cfg.update({key: value for key, value in nested.items() if key in cfg})
        direct_fields = {
            "enabled": "scheme_b_ablation_enabled",
            "mode": "scheme_b_ablation_mode",
            "zero_memory_state": "scheme_b_ablation_zero_memory_state",
            "zero_text_kv": "scheme_b_ablation_zero_text_kv",
            "zero_edge_cache": "scheme_b_ablation_zero_edge_cache",
            "keep_target_edges": "scheme_b_ablation_keep_target_edges",
            "log_interval": "scheme_b_ablation_log_interval",
        }
        for cfg_key, field_name in direct_fields.items():
            value = getattr(model_args, field_name, None)
            if value is not None:
                cfg[cfg_key] = value
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["mode"] = str(cfg["mode"] or "none")
        if cfg["mode"] not in {"none", "target_only_zero_others"}:
            raise ValueError("scheme_b_ablation.mode must be 'none' or 'target_only_zero_others'.")
        cfg["zero_memory_state"] = bool(cfg["zero_memory_state"])
        cfg["zero_text_kv"] = bool(cfg["zero_text_kv"])
        cfg["zero_edge_cache"] = bool(cfg["zero_edge_cache"])
        cfg["keep_target_edges"] = bool(cfg["keep_target_edges"])
        cfg["log_interval"] = max(int(cfg["log_interval"]), 1)
        return cfg

    def _encoder_suffix_quant_context(self):
        stack = ExitStack()
        stack.enter_context(int_gemm_context(self.scheme_b_int_gemm_quantizer))
        stack.enter_context(weight_quant_context(self.scheme_b_weight_quantizer))
        stack.enter_context(activation_observer_context(self.scheme_b_activation_observer_instance))
        stack.enter_context(activation_quant_context(self.scheme_b_activation_quantizer))
        return stack

    def _maybe_log_scheme_b_int_gemm_stats(self):
        quantizer = self.scheme_b_int_gemm_quantizer
        if quantizer is None:
            return
        stats = quantizer.stats
        call_count = int(stats.get("int_gemm_call_count", 0))
        fallback_count = int(stats.get("fallback_count", 0))
        log_signature = (call_count, fallback_count)
        if log_signature == self.scheme_b_int_gemm_last_logged_signature:
            return
        interval = max(int(self.scheme_b_int_gemm.get("log_interval", 20)), 1)
        encoder_call_idx = self.encoder_cache_calls + self.encoder_full_calls
        if not (encoder_call_idx <= 3 or encoder_call_idx % interval == 0):
            return
        self.scheme_b_int_gemm_last_logged_call_count = call_count
        self.scheme_b_int_gemm_last_logged_signature = log_signature
        print(
            "GOFA suffix int GEMM runtime stats: "
            f"int_gemm_call_count={call_count}, "
            f"int_gemm_numel_input={stats.get('int_gemm_numel_input', 0)}, "
            f"int_gemm_numel_output={stats.get('int_gemm_numel_output', 0)}, "
            f"int_gemm_int_mm_time_s={stats.get('int_gemm_int_mm_time_s', 0.0):.6f}, "
            f"int_gemm_quant_time_s={stats.get('int_gemm_quant_time_s', 0.0):.6f}, "
            f"int_gemm_dequant_time_s={stats.get('int_gemm_dequant_time_s', 0.0):.6f}, "
            f"int_gemm_total_time_s={stats.get('int_gemm_total_time_s', 0.0):.6f}, "
            f"int_gemm_backend={stats.get('int_gemm_backend', self.scheme_b_int_gemm.get('backend'))}, "
            f"fallback_count={fallback_count}, "
            f"quantized_module_count={stats.get('int_gemm_quantized_module_count', 0)}, "
            f"weight_bits={stats.get('weight_bits', self.scheme_b_int_gemm.get('weight_bits'))}, "
            f"activation_bits={stats.get('activation_bits', self.scheme_b_int_gemm.get('activation_bits'))}"
        )

    def _maybe_log_scheme_b_quant_kv_attention_stats(self):
        if not self.scheme_b_quant_kv_attention_enabled:
            return
        base_model = self.model.icae.get_base_model().model
        stats = getattr(base_model, "quant_kv_attention_stats", None)
        if not isinstance(stats, dict):
            return
        call_count = int(stats.get("quant_kv_attention_call_count", 0))
        fallback_count = int(stats.get("fallback_count", 0))
        log_signature = (call_count, fallback_count)
        if log_signature == self.scheme_b_quant_kv_attention_last_logged_signature:
            return
        interval = max(int(self.scheme_b_quant_kv_attention.get("log_interval", self.profile_stage_log_interval)), 1)
        encoder_call_idx = self.encoder_cache_calls + self.encoder_full_calls
        if not (encoder_call_idx <= 3 or encoder_call_idx % interval == 0 or fallback_count > 0):
            return
        self.scheme_b_quant_kv_attention_last_logged_signature = log_signature
        print(
            "GOFA quantized-KV attention runtime stats: "
            f"quant_kv_attention_call_count={call_count}, "
            f"k_unpack_time_s={stats.get('k_unpack_time_s', 0.0):.6f}, "
            f"q_scale_fold_time_s={stats.get('q_scale_fold_time_s', 0.0):.6f}, "
            f"q_eff_quant_time_s={stats.get('q_eff_quant_time_s', 0.0):.6f}, "
            f"qk_int_mm_time_s={stats.get('qk_int_mm_time_s', 0.0):.6f}, "
            f"logits_dequant_time_s={stats.get('logits_dequant_time_s', 0.0):.6f}, "
            f"softmax_time_s={stats.get('softmax_time_s', 0.0):.6f}, "
            f"v_unpack_time_s={stats.get('v_unpack_time_s', 0.0):.6f}, "
            f"pv_matmul_time_s={stats.get('pv_matmul_time_s', 0.0):.6f}, "
            f"v_scale_apply_time_s={stats.get('v_scale_apply_time_s', 0.0):.6f}, "
            f"fallback_count={fallback_count}, "
            f"backend={self.scheme_b_quant_kv_attention.get('backend')}, "
            f"key_scale_fold_into_q={self.scheme_b_quant_kv_attention.get('key_scale_fold_into_q')}, "
            f"key_bits={self.scheme_b_quant_kv_attention.get('key_bits')}, "
            f"value_bits={self.scheme_b_quant_kv_attention.get('value_bits')}, "
            f"pv_compute_mode={self.scheme_b_quant_kv_attention.get('pv_compute_mode')}, "
            f"example_shapes={stats.get('example_shapes', {})}"
        )

    def _maybe_log_scheme_b_activation_quant_stats(self):
        quantizer = self.scheme_b_activation_quantizer
        if quantizer is None:
            return
        stats = quantizer.stats
        call_count = int(stats.get("activation_quant_call_count", 0))
        if call_count == self.scheme_b_activation_quant_last_logged_call_count:
            return
        interval = max(int(self.profile_stage_log_interval), 1)
        encoder_call_idx = self.encoder_cache_calls + self.encoder_full_calls
        if not (encoder_call_idx <= 3 or encoder_call_idx % interval == 0):
            return
        self.scheme_b_activation_quant_last_logged_call_count = call_count
        print(
            "GOFA suffix activation quant runtime stats: "
            f"activation_quantized_module_count={stats.get('activation_quantized_module_count', 0)}, "
            f"activation_quant_call_count={call_count}, "
            f"activation_quant_tensor_count={stats.get('activation_quant_tensor_count', 0)}, "
            f"activation_quant_numel={stats.get('activation_quant_numel', 0)}, "
            f"activation_quant_time_s={stats.get('activation_quant_time_s', 0.0):.6f}, "
            f"activation_effective_bits={stats.get('activation_effective_bits', 0)}, "
            f"q_proj_bits={stats.get('q_proj_bits', self.scheme_b_activation_quant.get('q_proj_bits'))}, "
            f"k_proj_bits={stats.get('k_proj_bits', self.scheme_b_activation_quant.get('k_proj_bits'))}, "
            f"v_proj_bits={stats.get('v_proj_bits', self.scheme_b_activation_quant.get('v_proj_bits'))}, "
            f"o_proj_bits={stats.get('o_proj_bits', self.scheme_b_activation_quant.get('o_proj_bits'))}, "
            f"mlp_bits={stats.get('mlp_bits', self.scheme_b_activation_quant.get('mlp_bits'))}, "
            f"activation_quant_call_count_by_bits={stats.get('activation_quant_call_count_by_bits', {})}, "
            f"activation_quant_numel_by_bits={stats.get('activation_quant_numel_by_bits', {})}, "
            f"activation_quant_call_count_by_projection={stats.get('activation_quant_call_count_by_projection', {})}, "
            f"activation_quant_numel_by_projection={stats.get('activation_quant_numel_by_projection', {})}, "
            f"per_token={self.scheme_b_activation_quant.get('per_token', True)}, "
            f"clip_ratio={stats.get('clip_ratio', self.scheme_b_activation_quant.get('clip_ratio', 1.0))}, "
            f"quantize_attention={self.scheme_b_activation_quant.get('quantize_attention', True)}, "
            f"quantize_q_proj={self.scheme_b_activation_quant.get('quantize_q_proj', True)}, "
            f"quantize_k_proj={self.scheme_b_activation_quant.get('quantize_k_proj', True)}, "
            f"quantize_v_proj={self.scheme_b_activation_quant.get('quantize_v_proj', True)}, "
            f"quantize_o_proj={self.scheme_b_activation_quant.get('quantize_o_proj', True)}, "
            f"quantize_mlp={self.scheme_b_activation_quant.get('quantize_mlp', True)}"
        )

    def _maybe_dump_activation_observer(self):
        observer = self.scheme_b_activation_observer_instance
        if observer is None:
            return
        summary = observer.dump_summary()
        print(
            "GOFA activation observer summary dumped: "
            f"output_dir={summary.get('output_dir')}, "
            f"observed_batches={summary.get('observed_batches')}, "
            f"saved_tensors={summary.get('saved_tensors')}, "
            f"stats_records={summary.get('stats_records')}, "
            f"stats_path={summary.get('stats_path')}, "
            f"tensor_dir={summary.get('tensor_dir')}"
        )

    def _build_encoder_cache_namespace(self, icae_path, model_args, training_args, gofa_args):
        stat = os.stat(icae_path)
        base_config = self.model.icae.get_base_model().config
        metadata = {
            "format": 1,
            "model_name_or_path": model_args.model_name_or_path,
            "model_max_length": training_args.model_max_length,
            "mem_size": model_args.mem_size,
            "vocab_size": self.model.vocab_size,
            "hidden_size": self.model.dim,
            "num_hidden_layers": base_config.num_hidden_layers,
            "num_gnn_layers": gofa_args.num_layers,
            "fuse_type": gofa_args.fuse_type,
            "encoder_cache_mode": model_args.encoder_cache_mode,
            "attn_implementation": model_args.attn_implementation,
            "bf16": training_args.bf16,
            "icae_path": os.path.abspath(icae_path),
            "icae_size": stat.st_size,
            "icae_mtime_ns": stat.st_mtime_ns,
            "base_model_fingerprint": self._local_model_fingerprint(model_args.model_name_or_path),
            "tag": model_args.encoder_cache_tag,
        }
        payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _local_model_fingerprint(self, model_path):
        if not os.path.exists(model_path):
            return None
        if os.path.isfile(model_path):
            stat = os.stat(model_path)
            return [{"path": os.path.basename(model_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]

        fingerprint = []
        for name in sorted(os.listdir(model_path)):
            if not (name.endswith(".json") or name.endswith(".safetensors") or name.endswith(".bin")):
                continue
            path = os.path.join(model_path, name)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            fingerprint.append({"path": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return fingerprint

    def _encoder_cache_root(self):
        return os.path.join(self.encoder_cache_dir, self.encoder_cache_namespace)

    def _encoder_quant_cache_root(self):
        if self.scheme_b_quant.get("cache_dir"):
            return self.scheme_b_quant["cache_dir"]
        uniform_base = (
            self.scheme_b_quant["memory_base_bits"] == self.scheme_b_quant["base_bits"] and
            self.scheme_b_quant["key_base_bits"] == self.scheme_b_quant["base_bits"] and
            self.scheme_b_quant["value_base_bits"] == self.scheme_b_quant["base_bits"]
        )
        uniform_delta = (
            self.scheme_b_quant["memory_delta_bits"] == self.scheme_b_quant["delta_bits"] and
            self.scheme_b_quant["key_delta_bits"] == self.scheme_b_quant["delta_bits"] and
            self.scheme_b_quant["value_delta_bits"] == self.scheme_b_quant["delta_bits"]
        )
        if uniform_base and uniform_delta:
            suffix = f"_quant_b{self.scheme_b_quant['base_bits']}d{self.scheme_b_quant['delta_bits']}"
        else:
            suffix = (
                f"_quant_m{self.scheme_b_quant['memory_base_bits']}"
                f"k{self.scheme_b_quant['key_base_bits']}"
                f"v{self.scheme_b_quant['value_base_bits']}"
                f"_dm{self.scheme_b_quant['memory_delta_bits']}"
                f"dk{self.scheme_b_quant['key_delta_bits']}"
                f"dv{self.scheme_b_quant['value_delta_bits']}"
            )
        return self.encoder_cache_dir.rstrip(os.sep) + suffix

    def _encoder_quant_cache_tag_root(self, delta=False):
        root = self._encoder_quant_cache_root()
        if delta:
            root = os.path.join(root, "delta")
        return os.path.join(root, self.encoder_cache_namespace)

    def _validate_scheme_b_quant_cache_root(self):
        if not (self.scheme_b_quant_enabled and self.scheme_b_quant["strict"]):
            return
        quant_root = self._encoder_quant_cache_root()
        quant_tag_root = self._encoder_quant_cache_tag_root(delta=False)
        if not os.path.isdir(quant_root):
            raise RuntimeError(
                "GOFA scheme-B quant strict mode requires quant cache root to exist: "
                f"{quant_root}"
            )
        if not os.path.isdir(quant_tag_root):
            raise RuntimeError(
                "GOFA scheme-B quant strict mode requires quant cache tag directory to exist: "
                f"{quant_tag_root}"
            )

    def _scheme_b_payload_component_bits(self, payload, prefix):
        if prefix == "base":
            fallback_key = "base_bits"
            component_keys = ("memory_base_bits", "key_base_bits", "value_base_bits")
        elif prefix == "delta":
            fallback_key = "delta_bits"
            component_keys = ("memory_delta_bits", "key_delta_bits", "value_delta_bits")
        else:
            raise ValueError(f"Unsupported scheme-B quant bit prefix: {prefix}")
        fallback_value = payload.get(fallback_key)
        return {
            key: int(payload.get(key, fallback_value if fallback_value is not None else -1))
            for key in component_keys
        }

    def _scheme_b_quant_component_bits_match(self, payload, prefix):
        payload_bits = self._scheme_b_payload_component_bits(payload, prefix)
        mismatches = {}
        for key, payload_value in payload_bits.items():
            configured_value = int(self.scheme_b_quant[key])
            if int(payload_value) != configured_value:
                mismatches[key] = {
                    "payload": int(payload_value),
                    "configured": configured_value,
                }
        return mismatches

    def _encoder_cache_key(self, token_ids):
        token_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(token_bytes).hexdigest()

    def _encoder_cache_path(self, cache_key):
        return os.path.join(self._encoder_cache_root(), cache_key[:2], cache_key + ".pt")

    def _encoder_cache_relpath(self, cache_key):
        return os.path.join(self.encoder_cache_namespace, cache_key[:2], cache_key + ".pt")

    def _encoder_quant_cache_path(self, cache_key, delta=False):
        return os.path.join(self._encoder_quant_cache_tag_root(delta=delta), cache_key[:2], cache_key + ".pt")

    def _manifest_jsonify(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._manifest_jsonify(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._manifest_jsonify(item) for item in value]
        if isinstance(value, set):
            return [self._manifest_jsonify(item) for item in sorted(value)]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _encoder_cache_manifest_task_names(self):
        eval_task_names = getattr(self.model_args, "eval_task_names", None)
        train_task_names = getattr(self.model_args, "train_task_names", None)
        task_names = eval_task_names if eval_task_names is not None else train_task_names
        return self._manifest_jsonify(task_names)

    def _encoder_cache_manifest_cache_type(self, item_index, graph):
        if graph is None:
            return "unknown"
        num_node_feat = int(getattr(graph, "num_node_feat", 0))
        if item_index < num_node_feat:
            return "node"
        return "edge"

    def _manifest_sequence_item(self, value, index):
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                if index < value.numel():
                    return self._manifest_jsonify(value.detach().cpu().reshape(-1)[index])
                return None
            if isinstance(value, np.ndarray):
                if index < value.shape[0]:
                    return self._manifest_jsonify(value[index])
                return None
            if isinstance(value, (list, tuple)):
                if index < len(value):
                    return self._manifest_jsonify(value[index])
                return None
        except Exception:
            return None
        return None

    def _manifest_node_global_id(self, graph, local_node_idx):
        if graph is None:
            return None
        for attr_name in ("node_ids", "node_id", "global_node_id", "global_node_ids"):
            if hasattr(graph, attr_name):
                value = self._manifest_sequence_item(getattr(graph, attr_name), local_node_idx)
                if value is not None:
                    return value
        node_map = getattr(graph, "node_map", None)
        return self._manifest_sequence_item(node_map, local_node_idx)

    def _manifest_node_id_string(self, graph, local_node_idx):
        if graph is None:
            return None
        for attr_name in ("node_id_string", "node_id_strings"):
            if hasattr(graph, attr_name):
                value = self._manifest_sequence_item(getattr(graph, attr_name), local_node_idx)
                if value is not None:
                    return value
        x = getattr(graph, "x", None)
        value = self._manifest_sequence_item(x, local_node_idx)
        return str(value)[:200] if value is not None else None

    def _manifest_node_global_degree(self, graph, local_node_idx):
        if graph is None:
            return None
        for attr_name in ("global_degree", "global_degrees", "node_degree", "node_degrees", "degree", "degrees"):
            if hasattr(graph, attr_name):
                value = self._manifest_sequence_item(getattr(graph, attr_name), local_node_idx)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return value
        return None

    def _manifest_local_degrees(self, graph):
        if graph is None:
            return []
        return local_node_degrees(graph, num_nodes=int(getattr(graph, "num_node_feat", 0)))

    def _manifest_edge_position_for_item(self, graph, local_edge_idx):
        if graph is None:
            return int(local_edge_idx)
        edge_map = getattr(graph, "edge_map", None)
        if isinstance(edge_map, torch.Tensor):
            edge_map_cpu = edge_map.detach().cpu().reshape(-1)
            matches = (edge_map_cpu == int(local_edge_idx)).nonzero(as_tuple=False)
            if matches.numel() > 0:
                return int(matches[0].item())
        return int(local_edge_idx)

    def _encoder_cache_manifest_graph_metadata(self, item_index, graph):
        cache_type = self._encoder_cache_manifest_cache_type(item_index, graph)
        if graph is None:
            return {}
        num_node_feat = int(getattr(graph, "num_node_feat", 0))
        target_local = set(self._scheme_b_target_local_indices(graph))
        hop_distances = local_neighbors_by_hop(graph, target_local, max_hops=1)
        local_degrees = self._manifest_local_degrees(graph)
        if cache_type == "node":
            local_node_idx = int(item_index)
            hop_distance = hop_distances.get(local_node_idx)
            return {
                "local_node_idx": local_node_idx,
                "global_node_id": self._manifest_node_global_id(graph, local_node_idx),
                "node_id_string": self._manifest_node_id_string(graph, local_node_idx),
                "is_target": local_node_idx in target_local,
                "is_target_neighbor": hop_distance == 1,
                "hop_distance_to_target": hop_distance,
                "local_degree": int(local_degrees[local_node_idx]) if local_node_idx < len(local_degrees) else None,
                "global_degree": self._manifest_node_global_degree(graph, local_node_idx),
            }
        if cache_type == "edge":
            local_edge_idx = int(item_index) - num_node_feat
            edge_pos = self._manifest_edge_position_for_item(graph, local_edge_idx)
            edge_index = getattr(graph, "edge_index", None)
            src_local = dst_local = None
            if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2 and edge_pos < edge_index.size(1):
                src_local = int(edge_index[0, edge_pos].detach().cpu().item())
                dst_local = int(edge_index[1, edge_pos].detach().cpu().item())
            target_or_neighbor = set(idx for idx, distance in hop_distances.items() if distance is not None and distance <= 1)
            endpoints = {idx for idx in (src_local, dst_local) if idx is not None}
            return {
                "local_edge_idx": local_edge_idx,
                "edge_pos": edge_pos,
                "src_local": src_local,
                "dst_local": dst_local,
                "src_global": self._manifest_node_global_id(graph, src_local) if src_local is not None else None,
                "dst_global": self._manifest_node_global_id(graph, dst_local) if dst_local is not None else None,
                "is_incident_to_target": bool(endpoints & target_local),
                "both_endpoints_target_or_neighbor": bool(endpoints) and endpoints.issubset(target_or_neighbor),
            }
        return {}

    def _encoder_cache_manifest_shape_metadata(self, token_ids=None, cache_item=None, payload=None):
        seq_len = len(token_ids) if token_ids is not None else None
        text_len = None
        mem_size = None
        if isinstance(payload, dict):
            seq_len = payload.get("seq_len", seq_len)
            text_len = payload.get("text_len", text_len)
            mem_size = payload.get("mem_size", mem_size)
        if isinstance(cache_item, dict):
            text_len = cache_item.get("text_len", text_len)
            memory_state = cache_item.get("memory_state")
            if memory_state is not None and hasattr(memory_state, "size"):
                mem_size = int(memory_state.size(0))
        if mem_size is None:
            mem_size = self.mem_size
        return {
            "seq_len": int(seq_len) if seq_len is not None else None,
            "text_len": int(text_len) if text_len is not None else None,
            "mem_size": int(mem_size) if mem_size is not None else None,
        }

    def _merge_manifest_item(self, existing, incoming):
        merged = dict(existing)
        existing_count = int(merged.get("access_count", 0))
        incoming_count = int(incoming.get("access_count", 1))
        merged.update({key: value for key, value in incoming.items() if value is not None})
        merged["access_count"] = existing_count + incoming_count
        counts = dict(existing.get("hit_or_miss_counts", {}))
        for key, value in incoming.get("hit_or_miss_counts", {}).items():
            counts[key] = counts.get(key, 0) + int(value)
        merged["hit_or_miss_counts"] = counts
        return merged

    def _record_encoder_cache_manifest_item(
            self,
            cache_key,
            item_index,
            token_ids=None,
            cache_item=None,
            payload=None,
            graph=None,
            hit_or_miss="unknown"):
        if not self.encoder_cache_manifest_enabled or not cache_key:
            return False
        shape_meta = self._encoder_cache_manifest_shape_metadata(
            token_ids=token_ids,
            cache_item=cache_item,
            payload=payload,
        )
        item = {
            "cache_key": cache_key,
            "relpath": self._encoder_cache_relpath(cache_key),
            "cache_type": self._encoder_cache_manifest_cache_type(item_index, graph),
            "skip_nog": False,
            "seq_len": shape_meta["seq_len"],
            "text_len": shape_meta["text_len"],
            "mem_size": shape_meta["mem_size"],
            "hit_or_miss": hit_or_miss,
            "hit_or_miss_counts": {hit_or_miss: 1},
            "access_count": 1,
            "item_index": int(item_index),
        }
        item.update(self._encoder_cache_manifest_graph_metadata(item_index, graph))
        if cache_key in self.encoder_cache_manifest_items:
            self.encoder_cache_manifest_items[cache_key] = self._merge_manifest_item(
                self.encoder_cache_manifest_items[cache_key],
                item,
            )
            return False
        self.encoder_cache_manifest_items[cache_key] = item
        return True

    def _record_encoder_cache_manifest_skip_item(self, cache_key, item_index, token_ids=None, graph=None, reason="skip_nog"):
        if not self.encoder_cache_manifest_enabled or not cache_key:
            return
        skip_id = f"{cache_key}:{int(item_index)}"
        if skip_id in self.encoder_cache_manifest_skip_items:
            self.encoder_cache_manifest_skip_items[skip_id]["access_count"] += 1
            return
        self.encoder_cache_manifest_skip_items[skip_id] = {
            "cache_key": cache_key,
            "relpath": self._encoder_cache_relpath(cache_key),
            "cache_type": self._encoder_cache_manifest_cache_type(item_index, graph),
            "skip_nog": True,
            "seq_len": int(len(token_ids)) if token_ids is not None else None,
            "item_index": int(item_index),
            "reason": reason,
            "access_count": 1,
        }

    def _load_encoder_cache_manifest_append_state(self):
        if self.encoder_cache_manifest_append_loaded:
            return
        self.encoder_cache_manifest_append_loaded = True
        if not self.encoder_cache_manifest["append"]:
            return
        output_path = self.encoder_cache_manifest["output_path"]
        if not output_path or not os.path.exists(output_path):
            return
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
        except Exception as exc:
            print(f"GOFA encoder cache manifest warning: failed to read append manifest {output_path}: {exc}")
            return
        for item in existing.get("cache_items", []):
            cache_key = item.get("cache_key")
            if cache_key:
                self.encoder_cache_manifest_existing_items[cache_key] = item
        for item in existing.get("skip_items", []):
            skip_id = f"{item.get('cache_key')}:{item.get('item_index')}"
            self.encoder_cache_manifest_existing_skip_items[skip_id] = item
        self.encoder_cache_manifest_existing_total_samples = int(existing.get("total_samples", 0) or 0)

    def _combined_encoder_cache_manifest_items(self):
        self._load_encoder_cache_manifest_append_state()
        combined = OrderedDict()
        for cache_key, item in self.encoder_cache_manifest_existing_items.items():
            combined[cache_key] = item
        for cache_key, item in self.encoder_cache_manifest_items.items():
            if cache_key in combined:
                combined[cache_key] = self._merge_manifest_item(combined[cache_key], item)
            else:
                combined[cache_key] = item
        return combined

    def _combined_encoder_cache_manifest_skip_items(self):
        self._load_encoder_cache_manifest_append_state()
        combined = OrderedDict()
        for skip_id, item in self.encoder_cache_manifest_existing_skip_items.items():
            combined[skip_id] = item
        for skip_id, item in self.encoder_cache_manifest_skip_items.items():
            if skip_id in combined:
                merged = dict(combined[skip_id])
                merged["access_count"] = int(merged.get("access_count", 0)) + int(item.get("access_count", 1))
                combined[skip_id] = merged
            else:
                combined[skip_id] = item
        return combined

    def _maybe_dump_encoder_cache_manifest(self, current_batch_seen=0, force=False):
        if not self.encoder_cache_manifest_enabled:
            return
        output_path = self.encoder_cache_manifest["output_path"]
        if not output_path:
            return
        interval = self.encoder_cache_manifest["log_interval"]
        should_dump = (
            force or
            self.encoder_cache_manifest_samples <= 3 or
            self.encoder_cache_manifest_samples % interval == 0
        )
        if not should_dump:
            return
        cache_items = self._combined_encoder_cache_manifest_items()
        skip_items = self._combined_encoder_cache_manifest_skip_items()
        cache_item_values = list(cache_items.values())
        manifest_node_item_count = sum(1 for item in cache_item_values if item.get("cache_type") == "node")
        manifest_edge_item_count = sum(1 for item in cache_item_values if item.get("cache_type") == "edge")
        manifest_global_degree_count = sum(1 for item in cache_item_values if item.get("global_degree") is not None)
        manifest_local_degree_count = sum(1 for item in cache_item_values if item.get("local_degree") is not None)
        manifest_target_item_count = sum(1 for item in cache_item_values if item.get("is_target") is True)
        manifest_target_neighbor_item_count = sum(1 for item in cache_item_values if item.get("is_target_neighbor") is True)
        total_samples = self.encoder_cache_manifest_existing_total_samples + self.encoder_cache_manifest_samples
        manifest = {
            "manifest_format": "gofa_scheme_b_cache_manifest_v1",
            "task_names": self._encoder_cache_manifest_task_names(),
            "eval_task_names": self._manifest_jsonify(getattr(self.model_args, "eval_task_names", None)),
            "train_task_names": self._manifest_jsonify(getattr(self.model_args, "train_task_names", None)),
            "run_mode": self._manifest_jsonify(getattr(self.model_args, "run_mode", None)),
            "cache_tag": self.encoder_cache_namespace,
            "encoder_cache_namespace": self.encoder_cache_namespace,
            "full_cache_root": os.path.abspath(self.encoder_cache_dir),
            "cache_mode": self.encoder_cache_mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_samples": total_samples,
            "unique_cache_item_count": len(cache_items),
            "manifest_node_item_count": manifest_node_item_count,
            "manifest_edge_item_count": manifest_edge_item_count,
            "manifest_items_with_global_degree_count": manifest_global_degree_count,
            "manifest_items_with_local_degree_count": manifest_local_degree_count,
            "manifest_target_items_count": manifest_target_item_count,
            "manifest_target_neighbor_items_count": manifest_target_neighbor_item_count,
            "cache_items": cache_item_values,
            "skip_item_count": len(skip_items),
            "skip_items": list(skip_items.values()),
        }
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = f"{output_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, output_path)
        self.encoder_cache_manifest_dumps += 1
        print(
            "GOFA encoder cache manifest: "
            f"current_batch_seen_cache_items={current_batch_seen}, "
            f"cumulative_unique_cache_keys={len(cache_items)}, "
            f"skip_items={len(skip_items)}, "
            f"node_items={manifest_node_item_count}, "
            f"edge_items={manifest_edge_item_count}, "
            f"items_with_global_degree={manifest_global_degree_count}, "
            f"items_with_local_degree={manifest_local_degree_count}, "
            f"target_items={manifest_target_item_count}, "
            f"target_neighbor_items={manifest_target_neighbor_item_count}, "
            f"output_path={output_path}"
        )

    def _load_encoder_cache_item(self, token_ids):
        cache_key = self._encoder_cache_key(token_ids)
        cache_path = self._encoder_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None, cache_key
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("cache_key") != cache_key or payload.get("seq_len") != len(token_ids):
            return None, cache_key
        return payload["hidden_state"], cache_key

    def _save_encoder_cache_item(self, cache_key, token_ids, hidden_state):
        cache_path = self._encoder_cache_path(cache_key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        payload = {
            "cache_key": cache_key,
            "seq_len": len(token_ids),
            "dtype": str(hidden_state.dtype),
            "hidden_state": hidden_state.detach().cpu(),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def _load_encoder_memory_kv_cache_item(self, token_ids):
        cache_key = self._encoder_cache_key(token_ids)
        cache_path = self._encoder_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None, cache_key
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("cache_format") != "memory_text_kv_v1":
            return None, cache_key
        if payload.get("cache_key") != cache_key or payload.get("seq_len") != len(token_ids):
            return None, cache_key
        if payload.get("mem_size") != self.mem_size:
            return None, cache_key
        text_len = payload.get("text_len")
        if text_len is None or text_len + self.mem_size != len(token_ids):
            return None, cache_key
        text_kv = payload.get("text_kv")
        base_model = self.model.icae.get_base_model().model
        if not isinstance(text_kv, list) or len(text_kv) != base_model.gofa_config.num_layers:
            return None, cache_key
        return {
            "text_len": text_len,
            "memory_state": payload["memory_state"],
            "text_kv": text_kv,
        }, cache_key

    def _load_encoder_quant_memory_kv_base_payload(self, token_ids, strict=False):
        cache_key = self._encoder_cache_key(token_ids)
        cache_path = self._encoder_quant_cache_path(cache_key, delta=False)
        if not os.path.exists(cache_path):
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode missing quant base cache: "
                    f"cache_key={cache_key}, path={cache_path}"
                )
            return None, cache_key, 0, "missing"
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode failed to load quant base cache: "
                    f"cache_key={cache_key}, path={cache_path}"
                ) from exc
            return None, cache_key, os.path.getsize(cache_path), "format_error"
        cache_size = os.path.getsize(cache_path)
        if payload.get("cache_format") != QUANT_BASE_FORMAT:
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found invalid quant base cache format: "
                    f"cache_key={cache_key}, path={cache_path}, "
                    f"cache_format={payload.get('cache_format')}, expected={QUANT_BASE_FORMAT}"
                )
            return None, cache_key, cache_size, "format_error"
        bit_mismatches = self._scheme_b_quant_component_bits_match(payload, "base")
        if bit_mismatches:
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found mismatched quant base component bits: "
                    f"cache_key={cache_key}, path={cache_path}, "
                    f"payload_base_bits={payload.get('base_bits')}, "
                    f"payload_component_bits={self._scheme_b_payload_component_bits(payload, 'base')}, "
                    f"configured_component_bits={{"
                    f"'memory_base_bits': {self.scheme_b_quant['memory_base_bits']}, "
                    f"'key_base_bits': {self.scheme_b_quant['key_base_bits']}, "
                    f"'value_base_bits': {self.scheme_b_quant['value_base_bits']}"
                    f"}}, mismatches={bit_mismatches}"
                )
            return None, cache_key, cache_size, "format_error"
        if payload.get("cache_key") != cache_key or payload.get("seq_len") != len(token_ids):
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found quant base metadata mismatch: "
                    f"cache_key={cache_key}, path={cache_path}, "
                    f"payload_cache_key={payload.get('cache_key')}, payload_seq_len={payload.get('seq_len')}, "
                    f"expected_seq_len={len(token_ids)}"
                )
            return None, cache_key, cache_size, "format_error"
        if payload.get("mem_size") != self.mem_size:
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found mem_size mismatch: "
                    f"cache_key={cache_key}, path={cache_path}, "
                    f"payload_mem_size={payload.get('mem_size')}, expected_mem_size={self.mem_size}"
                )
            return None, cache_key, cache_size, "format_error"
        text_len = payload.get("text_len")
        if text_len is None or text_len + self.mem_size != len(token_ids):
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found text_len mismatch: "
                    f"cache_key={cache_key}, path={cache_path}, text_len={text_len}, "
                    f"seq_len={len(token_ids)}, mem_size={self.mem_size}"
                )
            return None, cache_key, cache_size, "format_error"
        text_kv = payload.get("text_kv")
        base_model = self.model.icae.get_base_model().model
        if not isinstance(text_kv, list) or len(text_kv) != base_model.gofa_config.num_layers:
            if strict:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode found invalid text_kv structure: "
                    f"cache_key={cache_key}, path={cache_path}, "
                    f"text_kv_layers={len(text_kv) if isinstance(text_kv, list) else None}, "
                    f"expected_layers={base_model.gofa_config.num_layers}"
                )
            return None, cache_key, cache_size, "format_error"
        return payload, cache_key, cache_size, "hit"

    def _load_encoder_quant_memory_kv_delta_payload(self, cache_key):
        cache_path = self._encoder_quant_cache_path(cache_key, delta=True)
        if not os.path.exists(cache_path):
            return None, 0, "missing"
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            return None, os.path.getsize(cache_path), "missing"
        cache_size = os.path.getsize(cache_path)
        if payload.get("cache_format") != QUANT_DELTA_FORMAT:
            return None, cache_size, "missing"
        if payload.get("cache_key") != cache_key:
            return None, cache_size, "missing"
        if payload.get("mem_size") != self.mem_size:
            return None, cache_size, "missing"
        if self._scheme_b_quant_component_bits_match(payload, "delta"):
            return None, cache_size, "missing"
        return payload, cache_size, "hit"

    def _save_encoder_memory_kv_cache_item(self, cache_key, token_ids, cache_item):
        cache_path = self._encoder_cache_path(cache_key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        payload = {
            "cache_format": "memory_text_kv_v1",
            "cache_key": cache_key,
            "seq_len": len(token_ids),
            "text_len": cache_item["text_len"],
            "mem_size": self.mem_size,
            "memory_state": cache_item["memory_state"],
            "text_kv": cache_item["text_kv"],
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def _sync_encoder_cache_timer(self, device):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _stage_timer_start(self, device):
        if not self.profile_stage_times:
            return None
        self._sync_encoder_cache_timer(device)
        return time.perf_counter()

    def _stage_timer_add_decoder(self, start_time, device):
        if start_time is None:
            return
        self._sync_encoder_cache_timer(device)
        self.decoder_stage_calls += 1
        self.decoder_stage_time_s += time.perf_counter() - start_time

    def _stage_timing_percent(self, value, total):
        return 100.0 * value / total if total > 0 else 0.0

    def _new_scheme_b_quant_stats(self):
        return {
            "full_scheme_b_cache_hit": 0,
            "full_scheme_b_cache_miss": 0,
            "quant_base_cache_hit": 0,
            "quant_base_cache_missing": 0,
            "quant_base_cache_format_error": 0,
            "quant_delta_cache_hit": 0,
            "quant_delta_cache_missing": 0,
            "quant_reconstruct_count": 0,
            "fallback_to_full_cache_count": 0,
            "online_compute_count_under_quant": 0,
            "quant_base_loaded_bytes": 0,
            "quant_delta_loaded_bytes": 0,
            "memory_delta_load_count": 0,
            "key_delta_load_count": 0,
            "value_delta_load_count": 0,
            "memory_delta_loaded_bytes": 0,
            "key_delta_loaded_bytes": 0,
            "value_delta_loaded_bytes": 0,
        }

    def _scheme_b_quant_bytes_from_bits(self, bit_count):
        return (int(bit_count) + 7) // 8

    def _scheme_b_quant_delta_component_bytes(self, delta_payload):
        component_bytes = {"memory": 0, "key": 0, "value": 0}
        if not isinstance(delta_payload, dict):
            return component_bytes
        component_bytes["memory"] = self._scheme_b_quant_bytes_from_bits(
            estimate_quantized_tensor_bits(delta_payload.get("memory_state"))
        )
        for layer_delta in delta_payload.get("text_kv", []):
            if not isinstance(layer_delta, dict):
                continue
            component_bytes["key"] += self._scheme_b_quant_bytes_from_bits(
                estimate_quantized_tensor_bits(layer_delta.get("key"))
            )
            component_bytes["value"] += self._scheme_b_quant_bytes_from_bits(
                estimate_quantized_tensor_bits(layer_delta.get("value"))
            )
        return component_bytes

    def _add_scheme_b_quant_stats(self, current):
        for key, value in current.items():
            self.scheme_b_quant_stats[key] = self.scheme_b_quant_stats.get(key, 0) + value

    def _maybe_log_scheme_b_quant_stats(self, current):
        if not self.scheme_b_quant_enabled:
            return
        interval = max(int(self.profile_stage_log_interval), 1)
        call_idx = self.encoder_cache_calls
        if not (call_idx <= 3 or call_idx % interval == 0):
            return
        current_parts = [f"{key}={current.get(key, 0)}" for key in self._new_scheme_b_quant_stats()]
        cumulative_parts = [f"cum_{key}={self.scheme_b_quant_stats.get(key, 0)}" for key in self._new_scheme_b_quant_stats()]
        print("GOFA scheme-B quant cache stats: " + ", ".join(current_parts + cumulative_parts))

    def _maybe_log_scheme_b_quant_path_example(self, cache_key):
        if not self.scheme_b_quant_enabled or self.scheme_b_quant_path_example_logged:
            return
        full_path = self._encoder_cache_path(cache_key)
        quant_base_path = self._encoder_quant_cache_path(cache_key, delta=False)
        quant_delta_path = self._encoder_quant_cache_path(cache_key, delta=True)
        print(
            "GOFA scheme-B quant cache path check: "
            f"cache_key={cache_key}, "
            f"cache_tag={self.encoder_cache_namespace}, "
            f"full_cache_path={full_path}, full_cache_exists={os.path.exists(full_path)}, "
            f"quant_base_cache_path={quant_base_path}, quant_base_cache_exists={os.path.exists(quant_base_path)}, "
            f"quant_delta_cache_path={quant_delta_path}, quant_delta_cache_exists={os.path.exists(quant_delta_path)}"
        )
        self.scheme_b_quant_path_example_logged = True

    def _maybe_log_stage_profile(self):
        if not self.profile_stage_times:
            return
        self.stage_profile_reports += 1
        interval = max(int(self.profile_stage_log_interval), 1)
        if not (self.stage_profile_reports <= 3 or self.stage_profile_reports % interval == 0):
            return

        base_model = self.model.icae.get_base_model().model
        profile = getattr(base_model, "stage_profile", None)
        if profile is None:
            return

        prefix = profile["encoder_prefix_transformer_s"]
        gnn_layers = profile["encoder_gnn_layer_s"]
        suffix_layers = profile["encoder_suffix_transformer_layer_s"]
        norm = profile["encoder_norm_s"]
        decoder = self.decoder_stage_time_s
        gnn_total = sum(gnn_layers)
        suffix_total = sum(suffix_layers)
        model_total = prefix + gnn_total + suffix_total + norm + decoder
        if model_total <= 0:
            return

        print(
            "GOFA stage timing summary: "
            f"report={self.stage_profile_reports}, model_total={model_total:.4f}s, "
            f"encoder_full_calls={profile['encoder_full_calls']}, "
            f"prefix_calls={profile['encoder_prefix_calls']}, "
            f"suffix_calls={profile['encoder_suffix_calls']}, "
            f"decoder_calls={self.decoder_stage_calls}"
        )
        print(
            "  encoder_prefix_transformer: "
            f"{prefix:.4f}s ({self._stage_timing_percent(prefix, model_total):.2f}%)"
        )
        for i, value in enumerate(gnn_layers):
            print(
                f"  encoder_gnn_layer_{i}: "
                f"{value:.4f}s ({self._stage_timing_percent(value, model_total):.2f}%)"
            )
        for i, value in enumerate(suffix_layers):
            print(
                f"  encoder_transformer_layer_{base_model.gnn_start_layer + i}: "
                f"{value:.4f}s ({self._stage_timing_percent(value, model_total):.2f}%)"
            )
        boundary_breakdown = [
            ("boundary_input_norm_s", "input_norm"),
            ("boundary_qkv_proj_s", "qkv_proj"),
            ("boundary_rope_repeat_s", "rope_repeat"),
            ("boundary_attn_scores_s", "attn_score_softmax_value"),
            ("boundary_o_proj_s", "o_proj"),
            ("boundary_post_attn_norm_s", "post_attn_norm"),
            ("boundary_mlp_s", "mlp"),
        ]
        if any(sum(profile.get(key, [])) > 0 for key, _ in boundary_breakdown):
            aggregate_parts = []
            for key, label in boundary_breakdown:
                value = sum(profile.get(key, []))
                if value > 0:
                    aggregate_parts.append(f"{label}={value:.4f}s")
            print("  boundary_transformer_breakdown_total: " + ", ".join(aggregate_parts))
            if self.profile_memory_kv_transformer_breakdown:
                for layer_idx in range(len(suffix_layers)):
                    layer_parts = []
                    for key, label in boundary_breakdown:
                        values = profile.get(key, [])
                        value = values[layer_idx] if layer_idx < len(values) else 0.0
                        if value > 0:
                            layer_parts.append(f"{label}={value:.4f}s")
                    print(
                        f"  boundary_transformer_layer_{base_model.gnn_start_layer + layer_idx}_breakdown: "
                        + ", ".join(layer_parts)
                    )
        memory_kv_breakdown = [
            ("memory_kv_text_kv_to_device_s", "kv_to_device"),
            ("memory_kv_input_norm_s", "input_norm"),
            ("memory_kv_qkv_proj_s", "qkv_proj"),
            ("memory_kv_rope_cache_s", "rope_cache_repeat"),
            ("memory_kv_attn_scores_s", "attn_score_softmax_value"),
            ("memory_kv_o_proj_s", "o_proj"),
            ("memory_kv_post_attn_norm_s", "post_attn_norm"),
            ("memory_kv_mlp_s", "mlp"),
        ]
        if any(sum(profile.get(key, [])) > 0 for key, _ in memory_kv_breakdown):
            aggregate_parts = []
            for key, label in memory_kv_breakdown:
                value = sum(profile.get(key, []))
                if value > 0:
                    aggregate_parts.append(f"{label}={value:.4f}s")
            print("  memory_kv_transformer_breakdown_total: " + ", ".join(aggregate_parts))
            if self.profile_memory_kv_transformer_breakdown:
                for layer_idx in range(len(suffix_layers)):
                    layer_parts = []
                    for key, label in memory_kv_breakdown:
                        values = profile.get(key, [])
                        value = values[layer_idx] if layer_idx < len(values) else 0.0
                        if value > 0:
                            layer_parts.append(f"{label}={value:.4f}s")
                    print(
                        f"  memory_kv_transformer_layer_{base_model.gnn_start_layer + layer_idx}_breakdown: "
                        + ", ".join(layer_parts)
                    )
        print(
            "  encoder_norm: "
            f"{norm:.4f}s ({self._stage_timing_percent(norm, model_total):.2f}%)"
        )
        print(
            "  decoder: "
            f"{decoder:.4f}s ({self._stage_timing_percent(decoder, model_total):.2f}%)"
        )

        cache_overhead = (
            self.encoder_cache_timing["load_s"]
            + self.encoder_cache_timing["save_s"]
            + self.encoder_cache_timing["assemble_s"]
        )
        if cache_overhead > 0:
            print(
                "  cache_overhead_load_save_assemble: "
                f"{cache_overhead:.4f}s "
                f"(load={self.encoder_cache_timing['load_s']:.4f}s, "
                f"save={self.encoder_cache_timing['save_s']:.4f}s, "
                f"assemble={self.encoder_cache_timing['assemble_s']:.4f}s)"
            )

    def _encoder_cache_log_timing(self, current):
        total_cacheable_items = self.encoder_cache_hits + self.encoder_cache_misses
        total_items = total_cacheable_items + self.encoder_cache_skips
        hit_rate = self.encoder_cache_hits / total_cacheable_items if total_cacheable_items else 0.0
        skip_rate = self.encoder_cache_skips / total_items if total_items else 0.0
        timing = self.encoder_cache_timing
        message = (
            "GOFA encoder cache timing: "
            f"call_total={self.encoder_cache_calls}, hit_rate={hit_rate:.2%}, "
            f"skip_rate={skip_rate:.2%}, "
            f"current_total={current['total_s']:.4f}s, "
            f"current_load={current['load_s']:.4f}s, "
            f"current_miss_compute={current['miss_compute_s']:.4f}s, "
            f"current_save={current['save_s']:.4f}s, "
            f"current_assemble={current['assemble_s']:.4f}s, "
            f"current_suffix={current['suffix_compute_s']:.4f}s, "
            f"cum_total={timing['total_s']:.4f}s, "
            f"cum_load={timing['load_s']:.4f}s, "
            f"cum_miss_compute={timing['miss_compute_s']:.4f}s, "
            f"cum_save={timing['save_s']:.4f}s, "
            f"cum_assemble={timing['assemble_s']:.4f}s, "
            f"cum_suffix={timing['suffix_compute_s']:.4f}s"
        )
        if any(current.get(key, 0.0) for key in ("quant_load_s", "dequant_s", "delta_load_s", "cache_size_bytes")):
            message += (
                f", current_quant_load={current.get('quant_load_s', 0.0):.4f}s, "
                f"current_dequant={current.get('dequant_s', 0.0):.4f}s, "
                f"current_delta_load={current.get('delta_load_s', 0.0):.4f}s, "
                f"current_cache_size={int(current.get('cache_size_bytes', 0))}B, "
                f"cum_quant_load={timing.get('quant_load_s', 0.0):.4f}s, "
                f"cum_dequant={timing.get('dequant_s', 0.0):.4f}s, "
                f"cum_delta_load={timing.get('delta_load_s', 0.0):.4f}s, "
                f"cum_cache_size={int(timing.get('cache_size_bytes', 0))}B"
            )
        print(message)

    def _encoder_cache_verify_sample(self, flat_values):
        sample_size = int(self.encoder_cache_verify_quantile_sample_size)
        if sample_size <= 0 or flat_values.numel() <= sample_size:
            return flat_values.contiguous()
        stride = max(int(np.ceil(flat_values.numel() / sample_size)), 1)
        sampled = flat_values[::stride]
        if sampled.numel() > sample_size:
            sampled = sampled[:sample_size]
        return sampled.contiguous()

    def _encoder_cache_verify_quantile(self, sampled_values, quantile):
        if sampled_values.numel() == 0:
            return 0.0
        kth = int(np.ceil(quantile * sampled_values.numel()))
        kth = max(1, min(kth, sampled_values.numel()))
        return torch.kthvalue(sampled_values, kth).values.item()

    def _encoder_cache_skip_indices(self, graph):
        if not self.model_args.encoder_cache_skip_nog or graph is None:
            return []
        if not hasattr(graph, "question_index") or not hasattr(graph, "node_map"):
            return []
        question_index = graph.question_index
        if question_index is None or question_index.numel() == 0:
            return []
        question_index = question_index.to(graph.node_map.device)
        valid_mask = (question_index >= 0) & (question_index < graph.node_map.numel())
        if not torch.any(valid_mask):
            return []
        skip_indices = torch.unique(graph.node_map[question_index[valid_mask]]).detach().cpu().tolist()
        return [int(idx) for idx in skip_indices if 0 <= int(idx) < graph.num_node_feat]

    def _encode_with_encoder_cache(
            self,
            token_ids,
            padded_token_ids,
            mem_mask,
            graph=None,
            partial_grad=None,
            skip_cache_indices=None):
        if self.training:
            return None
        base_model = self.model.icae.get_base_model().model
        if not hasattr(base_model, "forward_llm_prefix") or not hasattr(base_model, "forward_from_gnn_boundary"):
            return None

        device = padded_token_ids.device
        self._sync_encoder_cache_timer(device)
        total_start = time.perf_counter()
        current_timing = {
            "load_s": 0.0,
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
        }

        skip_cache_indices = set(skip_cache_indices or [])
        cached_states = [None] * len(token_ids)
        missing = []
        missing_keys = []
        skipped = []
        load_start = time.perf_counter()
        for i, ids in enumerate(token_ids):
            if i in skip_cache_indices:
                missing.append(i)
                missing_keys.append(None)
                skipped.append(i)
                continue
            cached_state, cache_key = self._load_encoder_cache_item(ids)
            if cached_state is None:
                missing.append(i)
                missing_keys.append(cache_key)
            else:
                cached_states[i] = cached_state
        current_timing["load_s"] = time.perf_counter() - load_start

        if missing:
            self._sync_encoder_cache_timer(device)
            miss_compute_start = time.perf_counter()
            missing_token_ids = [token_ids[i] for i in missing]
            missing_padded = self.model.tokenizer.pad(
                {"input_ids": missing_token_ids}, padding=True, return_tensors="pt"
            )["input_ids"].to(padded_token_ids.device)
            missing_embeddings = self.model.tokens_to_embeddings(missing_padded)
            prefix_output = base_model.forward_llm_prefix(
                inputs_embeds=missing_embeddings,
                partial_grad=partial_grad,
                return_dict=True,
            ).last_hidden_state
            self._sync_encoder_cache_timer(device)
            current_timing["miss_compute_s"] = time.perf_counter() - miss_compute_start

            save_start = time.perf_counter()
            for batch_idx, original_idx in enumerate(missing):
                seq_len = len(token_ids[original_idx])
                cached_state = prefix_output[batch_idx, :seq_len].detach().cpu()
                cached_states[original_idx] = cached_state
                if missing_keys[batch_idx] is not None:
                    self._save_encoder_cache_item(missing_keys[batch_idx], token_ids[original_idx], cached_state)
            current_timing["save_s"] = time.perf_counter() - save_start

        self._sync_encoder_cache_timer(device)
        assemble_start = time.perf_counter()
        hidden_dtype = cached_states[0].dtype
        boundary_hidden_states = torch.zeros(
            (len(cached_states), padded_token_ids.size(1), cached_states[0].size(-1)),
            dtype=hidden_dtype,
            device=padded_token_ids.device,
        )
        for i, cached_state in enumerate(cached_states):
            seq_len = cached_state.size(0)
            boundary_hidden_states[i, :seq_len] = cached_state.to(padded_token_ids.device)
        self._sync_encoder_cache_timer(device)
        current_timing["assemble_s"] = time.perf_counter() - assemble_start

        self.encoder_cache_calls += 1
        current_skips = len(skipped)
        current_misses = len(missing) - current_skips
        current_hits = len(token_ids) - len(missing)
        self.encoder_cache_hits += current_hits
        self.encoder_cache_misses += current_misses
        self.encoder_cache_skips += current_skips

        self._sync_encoder_cache_timer(device)
        suffix_start = time.perf_counter()
        with self._encoder_suffix_quant_context():
            final_hidden_states = base_model.forward_from_gnn_boundary(
                boundary_hidden_states=boundary_hidden_states,
                graph=graph,
                mem_mask=mem_mask,
                partial_grad=partial_grad,
                map_node=True,
                return_dict=True,
            ).last_hidden_state
        self._sync_encoder_cache_timer(device)
        current_timing["suffix_compute_s"] = time.perf_counter() - suffix_start
        current_timing["total_s"] = time.perf_counter() - total_start

        for key, value in current_timing.items():
            self.encoder_cache_timing[key] += value

        if self.encoder_cache_calls <= 3 or self.encoder_cache_calls % 50 == 0:
            print(
                "GOFA encoder cache: "
                f"call={self.encoder_cache_calls}, hits={current_hits}, "
                f"misses={current_misses}, skips={current_skips}, "
                f"total_hits={self.encoder_cache_hits}, total_misses={self.encoder_cache_misses}, "
                f"total_skips={self.encoder_cache_skips}"
            )
            self._encoder_cache_log_timing(current_timing)
        self._maybe_log_scheme_b_int_gemm_stats()
        self._maybe_log_scheme_b_quant_kv_attention_stats()
        self._maybe_log_scheme_b_activation_quant_stats()

        return final_hidden_states

    def _memory_kv_mapped_mem_mask(self, mapped_items, max_seq_len, device):
        mem_mask = torch.zeros((len(mapped_items), max_seq_len), dtype=torch.bool, device=device)
        for i, item in enumerate(mapped_items):
            start = item["text_len"]
            end = start + self.mem_size
            mem_mask[i, start:end] = True
        return mem_mask

    def _scheme_b_ablation_active(self):
        return (
            self.scheme_b_ablation_enabled and
            self.scheme_b_ablation["mode"] == "target_only_zero_others"
        )

    def _scheme_b_int_list(self, value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, np.ndarray):
            return [int(v) for v in value.reshape(-1).tolist()]
        if isinstance(value, (list, tuple, set)):
            result = []
            for item in value:
                result.extend(self._scheme_b_int_list(item))
            return result
        return [int(value)]

    def _scheme_b_target_local_indices(self, graph):
        if graph is None or not hasattr(graph, "target_index"):
            return []
        num_node_feat = int(getattr(graph, "num_node_feat", 0))
        target_indices = []
        for idx in self._scheme_b_int_list(graph.target_index):
            if 0 <= idx < num_node_feat:
                target_indices.append(idx)
        return sorted(set(target_indices))

    def _scheme_b_target_edge_item_indices(self, graph, target_local_indices, num_items):
        if graph is None or not target_local_indices:
            return set()
        edge_index = getattr(graph, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
            return set()
        num_node_feat = int(getattr(graph, "num_node_feat", 0))
        if num_node_feat >= num_items:
            return set()
        target_set = set(int(idx) for idx in target_local_indices)
        edge_index_cpu = edge_index.detach().cpu()
        edge_map = getattr(graph, "edge_map", None)
        edge_map_cpu = edge_map.detach().cpu() if isinstance(edge_map, torch.Tensor) else None
        kept_edges = set()
        for edge_pos, (src, dst) in enumerate(edge_index_cpu.t().tolist()):
            if int(src) not in target_set and int(dst) not in target_set:
                continue
            if edge_map_cpu is not None and edge_pos < edge_map_cpu.numel():
                edge_item_offset = int(edge_map_cpu[edge_pos].item())
            else:
                edge_item_offset = edge_pos
            item_idx = num_node_feat + edge_item_offset
            if num_node_feat <= item_idx < num_items:
                kept_edges.add(item_idx)
        return kept_edges

    def _zero_scheme_b_tensor_or_quant_payload(self, value):
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            value.zero_()
            return True
        if not isinstance(value, dict):
            return False
        zeroed = False
        tensor = value.get("tensor")
        if isinstance(tensor, torch.Tensor):
            tensor.zero_()
            zeroed = True
        q = value.get("q")
        if isinstance(q, torch.Tensor):
            q.zero_()
            zeroed = True
        q_packed = value.get("q_packed")
        if isinstance(q_packed, torch.Tensor):
            bits = int(value.get("pack_bits") or value.get("bits") or 0)
            if bits not in {2, 4}:
                q_packed.zero_()
            else:
                values_per_byte = 8 // bits
                zero_code = 1 << (bits - 1)
                packed_zero = 0
                for value_idx in range(values_per_byte):
                    packed_zero |= zero_code << (bits * value_idx)
                q_packed.fill_(packed_zero)
            zeroed = True
        return zeroed

    def _zero_scheme_b_cache_item(self, cache_item, zero_memory_state=True, zero_text_kv=True):
        if zero_memory_state and cache_item.get("memory_state") is not None:
            cache_item["memory_state"].zero_()
        if zero_text_kv:
            for layer_kv in cache_item.get("text_kv", []):
                self._zero_scheme_b_tensor_or_quant_payload(layer_kv.get("key"))
                self._zero_scheme_b_tensor_or_quant_payload(layer_kv.get("value"))

    def _apply_scheme_b_quant_debug_zero_base(self, cache_items, quant_base_payloads, non_skip_item_count):
        if not (self.scheme_b_quant_enabled and self.scheme_b_quant["debug_zero_base"]):
            return
        quant_base_item_count = sum(1 for payload in quant_base_payloads if payload is not None)
        if non_skip_item_count > 0 and quant_base_item_count == 0:
            raise RuntimeError(
                "GOFA scheme-B quant debug_zero_base found zero quant base cache items for a non-empty batch: "
                f"non_skip_item_count={non_skip_item_count}, quant_cache_root={self._encoder_quant_cache_root()}, "
                f"cache_tag={self.encoder_cache_namespace}"
            )
        zeroed_base_item_count = 0
        zeroed_memory_state_count = 0
        zeroed_text_kv_count = 0
        for item_idx, payload in enumerate(quant_base_payloads):
            if payload is None:
                continue
            cache_item = cache_items[item_idx]
            if cache_item is None:
                continue
            zeroed_base_item_count += 1
            if cache_item.get("memory_state") is not None:
                cache_item["memory_state"].zero_()
                zeroed_memory_state_count += 1
            for layer_kv in cache_item.get("text_kv", []):
                zeroed_key = self._zero_scheme_b_tensor_or_quant_payload(layer_kv.get("key"))
                zeroed_value = self._zero_scheme_b_tensor_or_quant_payload(layer_kv.get("value"))
                if zeroed_key or zeroed_value:
                    zeroed_text_kv_count += 1
        print(
            "GOFA scheme-B quant debug_zero_base:\n"
            f"  quant_cache_base_item_count={quant_base_item_count}\n"
            f"  zeroed_base_item_count={zeroed_base_item_count}\n"
            f"  zeroed_memory_state_count={zeroed_memory_state_count}\n"
            f"  zeroed_text_kv_count={zeroed_text_kv_count}"
        )

    def _apply_scheme_b_ablation(self, cache_items, graph, skip_cache_indices=None):
        if not self._scheme_b_ablation_active():
            return
        self.scheme_b_ablation_calls += 1
        cfg = self.scheme_b_ablation
        num_items = len(cache_items)
        num_node_feat = int(getattr(graph, "num_node_feat", num_items)) if graph is not None else num_items
        num_node_feat = min(num_node_feat, num_items)
        skip_cache_indices = set(int(idx) for idx in (skip_cache_indices or []))
        target_local_indices = self._scheme_b_target_local_indices(graph)
        kept_node_items = {idx for idx in target_local_indices if 0 <= idx < num_node_feat}
        skipped_node_items = {idx for idx in skip_cache_indices if 0 <= idx < num_node_feat}

        keep_edge_items = set()
        if cfg["keep_target_edges"]:
            keep_edge_items = self._scheme_b_target_edge_item_indices(graph, target_local_indices, num_items)

        zeroed_node_items = 0
        zeroed_edge_items = 0
        for item_idx, cache_item in enumerate(cache_items):
            if cache_item is None:
                continue
            if item_idx in skip_cache_indices:
                continue
            if item_idx < num_node_feat:
                if item_idx in kept_node_items:
                    continue
                if cfg["zero_memory_state"] or cfg["zero_text_kv"]:
                    self._zero_scheme_b_cache_item(
                        cache_item,
                        zero_memory_state=cfg["zero_memory_state"],
                        zero_text_kv=cfg["zero_text_kv"],
                    )
                    zeroed_node_items += 1
            elif cfg["zero_edge_cache"] and item_idx not in keep_edge_items:
                self._zero_scheme_b_cache_item(cache_item, zero_memory_state=True, zero_text_kv=True)
                zeroed_edge_items += 1

        if not target_local_indices:
            print(
                "GOFA scheme-B ablation warning: "
                "target_index is missing or contains no valid local node index for this sampled subgraph."
            )
        elif not kept_node_items:
            print(
                "GOFA scheme-B ablation warning: "
                f"target_index={target_local_indices} did not map to cache node item range [0, {num_node_feat})."
            )

        log_interval = cfg["log_interval"]
        should_log = self.scheme_b_ablation_calls <= 3 or self.scheme_b_ablation_calls % log_interval == 0
        if should_log:
            edge_items = max(num_items - num_node_feat, 0)
            print(
                "GOFA scheme-B ablation: "
                f"call={self.scheme_b_ablation_calls}, "
                f"mode={cfg['mode']}, "
                f"kept_target_node_cache_items={len(kept_node_items)}, "
                f"zeroed_node_cache_items={zeroed_node_items}, "
                f"zeroed_edge_cache_items={zeroed_edge_items}, "
                f"target_index={target_local_indices}, "
                f"zero_memory_state={cfg['zero_memory_state']}, "
                f"zero_text_kv={cfg['zero_text_kv']}, "
                f"zero_edge_cache={cfg['zero_edge_cache']}, "
                f"keep_target_edges={cfg['keep_target_edges']}, "
                f"target_cache_item_indices={sorted(kept_node_items)}, "
                f"node_item_range=[0,{num_node_feat}), "
                f"edge_item_range=[{num_node_feat},{num_items}), "
                f"edge_items={edge_items}, "
                "local_node_to_cache_item=identity, "
                f"skipped_online_node_items={sorted(skipped_node_items)}"
            )

    def _encode_with_memory_kv_cache(
            self,
            token_ids,
            padded_token_ids,
            mem_mask,
            graph=None,
            partial_grad=None,
            skip_cache_indices=None):
        if self.training:
            return None
        base_model = self.model.icae.get_base_model().model
        required_methods = ("forward_llm_prefix", "build_memory_text_kv_cache_item", "forward_memory_with_text_kv")
        if not all(hasattr(base_model, method) for method in required_methods):
            return None

        device = padded_token_ids.device
        self._sync_encoder_cache_timer(device)
        total_start = time.perf_counter()
        current_timing = {
            "load_s": 0.0,
            "quant_load_s": 0.0,
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "dequant_s": 0.0,
            "delta_load_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
            "cache_size_bytes": 0.0,
        }
        current_quant_stats = self._new_scheme_b_quant_stats()
        strict_quant = self.scheme_b_quant_enabled and self.scheme_b_quant["strict"]
        manifest_batch_seen = 0
        if self.encoder_cache_manifest_enabled:
            self.encoder_cache_manifest_samples += 1

        skip_cache_indices = set(skip_cache_indices or [])
        cache_items = [None] * len(token_ids)
        quant_base_payloads = [None] * len(token_ids)
        quant_cache_keys = [None] * len(token_ids)
        static_tiers = [None] * len(token_ids)
        missing = []
        missing_keys = []
        skipped = []
        load_start = time.perf_counter()
        for i, ids in enumerate(token_ids):
            if i in skip_cache_indices:
                if self.encoder_cache_manifest_enabled:
                    skip_cache_key = self._encoder_cache_key(ids)
                    self._record_encoder_cache_manifest_skip_item(
                        skip_cache_key,
                        i,
                        token_ids=ids,
                        graph=graph,
                        reason="skip_nog",
                    )
                missing.append(i)
                missing_keys.append(None)
                skipped.append(i)
                continue
            if self.scheme_b_quant_enabled:
                cache_key = self._encoder_cache_key(ids)
                self._maybe_log_scheme_b_quant_path_example(cache_key)
                quant_start = time.perf_counter()
                base_payload, cache_key, cache_size, quant_status = self._load_encoder_quant_memory_kv_base_payload(
                    ids,
                    strict=strict_quant,
                )
                current_timing["quant_load_s"] += time.perf_counter() - quant_start
                if quant_status == "hit":
                    current_quant_stats["quant_base_cache_hit"] += 1
                    current_quant_stats["quant_base_loaded_bytes"] += cache_size
                    current_timing["cache_size_bytes"] += cache_size
                    quant_base_payloads[i] = base_payload
                    quant_cache_keys[i] = cache_key
                    static_tiers[i] = base_payload.get("static_tier", "low")
                    if self.encoder_cache_manifest_enabled:
                        self._record_encoder_cache_manifest_item(
                            cache_key,
                            i,
                            token_ids=ids,
                            payload=base_payload,
                            graph=graph,
                            hit_or_miss="quant_hit",
                        )
                        manifest_batch_seen += 1
                    continue
                if quant_status == "missing":
                    current_quant_stats["quant_base_cache_missing"] += 1
                else:
                    current_quant_stats["quant_base_cache_format_error"] += 1
            cache_item, cache_key = self._load_encoder_memory_kv_cache_item(ids)
            if cache_item is None:
                if self.scheme_b_quant_enabled:
                    current_quant_stats["full_scheme_b_cache_miss"] += 1
                missing.append(i)
                missing_keys.append(cache_key)
            else:
                if self.scheme_b_quant_enabled:
                    current_quant_stats["full_scheme_b_cache_hit"] += 1
                    current_quant_stats["fallback_to_full_cache_count"] += 1
                cache_items[i] = cache_item
                if self.encoder_cache_manifest_enabled:
                    self._record_encoder_cache_manifest_item(
                        cache_key,
                        i,
                        token_ids=ids,
                        cache_item=cache_item,
                        graph=graph,
                        hit_or_miss="full_fallback_hit" if self.scheme_b_quant_enabled else "hit",
                    )
                    manifest_batch_seen += 1
        current_timing["load_s"] = time.perf_counter() - load_start

        non_skip_item_count = len(token_ids) - len(skipped)
        if strict_quant and non_skip_item_count > 0 and current_quant_stats["quant_base_cache_hit"] == 0:
            raise RuntimeError(
                "GOFA scheme-B quant strict mode loaded zero quant base cache items for a non-empty batch: "
                f"non_skip_item_count={non_skip_item_count}, quant_cache_root={self._encoder_quant_cache_root()}, "
                f"cache_tag={self.encoder_cache_namespace}"
            )

        if self.scheme_b_quant_enabled and any(payload is not None for payload in quant_base_payloads):
            load_policy, policy_details = build_scheme_b_load_policy(
                graph,
                static_tiers=static_tiers,
                num_items=len(token_ids),
                target_aware_delta=self.scheme_b_quant["target_aware_delta"],
                target_aware_policy=self.scheme_b_quant["target_aware_policy"],
                target_delta_hops=self.scheme_b_quant["target_delta_hops"],
                keep_target_edges=self.scheme_b_quant["keep_target_edges"],
                local_degree_top_ratio=self.scheme_b_quant["local_degree_top_ratio"],
                local_degree_threshold=self.scheme_b_quant["local_degree_threshold"],
                max_delta_items_per_batch=self.scheme_b_quant["max_delta_items_per_batch"],
                return_details=True,
            )
            policy_stats = summarize_load_policy([
                policy for policy, payload in zip(load_policy, quant_base_payloads)
                if payload is not None
            ])
            for i, base_payload in enumerate(quant_base_payloads):
                if base_payload is None:
                    continue
                delta_payload = None
                should_load_selected_delta = (
                    load_policy[i] == BASE_DELTA and
                    base_payload.get("has_delta", True) and
                    (
                        self.scheme_b_quant["load_memory_delta"] or
                        self.scheme_b_quant["load_key_delta"] or
                        self.scheme_b_quant["load_value_delta"]
                    )
                )
                if should_load_selected_delta:
                    delta_start = time.perf_counter()
                    delta_payload, delta_size, delta_status = self._load_encoder_quant_memory_kv_delta_payload(quant_cache_keys[i])
                    current_timing["delta_load_s"] += time.perf_counter() - delta_start
                    if delta_status == "hit":
                        current_quant_stats["quant_delta_cache_hit"] += 1
                        current_quant_stats["quant_delta_loaded_bytes"] += delta_size
                        current_timing["cache_size_bytes"] += delta_size
                        component_bytes = self._scheme_b_quant_delta_component_bytes(delta_payload)
                        if self.scheme_b_quant["load_memory_delta"] and component_bytes["memory"] > 0:
                            current_quant_stats["memory_delta_load_count"] += 1
                            current_quant_stats["memory_delta_loaded_bytes"] += component_bytes["memory"]
                        if self.scheme_b_quant["load_key_delta"] and component_bytes["key"] > 0:
                            current_quant_stats["key_delta_load_count"] += 1
                            current_quant_stats["key_delta_loaded_bytes"] += component_bytes["key"]
                        if self.scheme_b_quant["load_value_delta"] and component_bytes["value"] > 0:
                            current_quant_stats["value_delta_load_count"] += 1
                            current_quant_stats["value_delta_loaded_bytes"] += component_bytes["value"]
                    else:
                        current_quant_stats["quant_delta_cache_missing"] += 1
                        if strict_quant:
                            raise RuntimeError(
                                "GOFA scheme-B quant strict mode missing selected delta cache: "
                                f"index={i}, cache_key={quant_cache_keys[i]}, "
                                f"policy={self.scheme_b_quant['target_aware_policy']}, "
                                f"path={self._encoder_quant_cache_path(quant_cache_keys[i], delta=True)}"
                            )
                dequant_start = time.perf_counter()
                try:
                    cache_items[i] = reconstruct_scheme_b_cache_item(
                        base_payload,
                        delta_payload=delta_payload,
                        load_delta=load_policy[i] == BASE_DELTA,
                        load_memory_delta=self.scheme_b_quant["load_memory_delta"],
                        load_key_delta=self.scheme_b_quant["load_key_delta"],
                        load_value_delta=self.scheme_b_quant["load_value_delta"],
                        preserve_quantized_text_kv=self.scheme_b_quant_kv_attention_enabled,
                    )
                    current_quant_stats["quant_reconstruct_count"] += 1
                    current_timing["dequant_s"] += time.perf_counter() - dequant_start
                except Exception as exc:
                    current_timing["dequant_s"] += time.perf_counter() - dequant_start
                    if strict_quant:
                        raise RuntimeError(
                            "GOFA scheme-B quant strict mode failed to reconstruct quant cache item: "
                            f"index={i}, cache_key={quant_cache_keys[i]}"
                        ) from exc
                    quant_base_payloads[i] = None
                    cache_item, cache_key = self._load_encoder_memory_kv_cache_item(token_ids[i])
                    if cache_item is None:
                        current_quant_stats["full_scheme_b_cache_miss"] += 1
                        missing.append(i)
                        missing_keys.append(cache_key)
                    else:
                        current_quant_stats["full_scheme_b_cache_hit"] += 1
                        current_quant_stats["fallback_to_full_cache_count"] += 1
                        cache_items[i] = cache_item
                        if self.encoder_cache_manifest_enabled:
                            self._record_encoder_cache_manifest_item(
                                cache_key,
                                i,
                                token_ids=token_ids[i],
                                cache_item=cache_item,
                                graph=graph,
                                hit_or_miss="reconstruct_fallback_hit",
                            )
                            manifest_batch_seen += 1
            quant_log_interval = max(int(self.profile_stage_log_interval), 1)
            if self.encoder_cache_calls < 3 or (self.encoder_cache_calls + 1) % quant_log_interval == 0:
                print(
                    "GOFA scheme-B quant policy: "
                    f"policy={self.scheme_b_quant['target_aware_policy']}, "
                    f"base_only={policy_stats['base_only']}, "
                    f"base_delta={policy_stats['base_delta']}, "
                    f"delta_load_ratio={policy_stats['delta_load_ratio']:.2%}, "
                    f"selected_target_node_local_idx={policy_details['target_node_indices']}, "
                    f"selected_1hop_node_idx={policy_details['one_hop_node_indices']}, "
                    f"selected_local_degree_node_idx={policy_details['local_degree_node_indices']}, "
                    f"selected_edge_count={policy_details['selected_edge_count']}, "
                    f"selected_edge_item_idx={policy_details['edge_item_indices']}, "
                    f"quant_base_cache_hit={current_quant_stats['quant_base_cache_hit']}, "
                    f"quant_delta_cache_hit={current_quant_stats['quant_delta_cache_hit']}, "
                    f"delta_loaded_bytes={current_quant_stats['quant_delta_loaded_bytes']}, "
                    f"memory_delta_load_count={current_quant_stats['memory_delta_load_count']}, "
                    f"key_delta_load_count={current_quant_stats['key_delta_load_count']}, "
                    f"value_delta_load_count={current_quant_stats['value_delta_load_count']}, "
                    f"memory_delta_loaded_bytes={current_quant_stats['memory_delta_loaded_bytes']}, "
                    f"key_delta_loaded_bytes={current_quant_stats['key_delta_loaded_bytes']}, "
                    f"value_delta_loaded_bytes={current_quant_stats['value_delta_loaded_bytes']}"
                )

        if self.scheme_b_quant_enabled:
            online_under_quant = sum(1 for idx in missing if idx not in skip_cache_indices)
            current_quant_stats["online_compute_count_under_quant"] += online_under_quant
            if strict_quant and current_quant_stats["fallback_to_full_cache_count"] > 0:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode unexpectedly fell back to full Scheme-B cache: "
                    f"fallback_to_full_cache_count={current_quant_stats['fallback_to_full_cache_count']}"
                )
            if strict_quant and online_under_quant > 0:
                raise RuntimeError(
                    "GOFA scheme-B quant strict mode would execute online cache miss compute: "
                    f"online_compute_count_under_quant={online_under_quant}, missing_indices={missing}"
                )

        if missing:
            self._sync_encoder_cache_timer(device)
            miss_compute_start = time.perf_counter()
            missing_token_ids = [token_ids[i] for i in missing]
            missing_padded = self.model.tokenizer.pad(
                {"input_ids": missing_token_ids}, padding=True, return_tensors="pt"
            )["input_ids"].to(padded_token_ids.device)
            missing_embeddings = self.model.tokens_to_embeddings(missing_padded)
            prefix_output = base_model.forward_llm_prefix(
                inputs_embeds=missing_embeddings,
                partial_grad=partial_grad,
                return_dict=True,
            ).last_hidden_state

            built_items = []
            for batch_idx, original_idx in enumerate(missing):
                seq_len = len(token_ids[original_idx])
                text_len = seq_len - self.mem_size
                with self._encoder_suffix_quant_context():
                    cache_item = base_model.build_memory_text_kv_cache_item(
                        prefix_output[batch_idx, :seq_len],
                        text_len=text_len,
                        partial_grad=partial_grad,
                    )
                built_items.append(cache_item)
                cache_items[original_idx] = cache_item
            self._sync_encoder_cache_timer(device)
            current_timing["miss_compute_s"] = time.perf_counter() - miss_compute_start

            save_start = time.perf_counter()
            for batch_idx, original_idx in enumerate(missing):
                if missing_keys[batch_idx] is not None:
                    self._save_encoder_memory_kv_cache_item(
                        missing_keys[batch_idx],
                        token_ids[original_idx],
                        built_items[batch_idx],
                    )
                    if self.encoder_cache_manifest_enabled:
                        self._record_encoder_cache_manifest_item(
                            missing_keys[batch_idx],
                            original_idx,
                            token_ids=token_ids[original_idx],
                            cache_item=built_items[batch_idx],
                            graph=graph,
                            hit_or_miss="online_miss_under_quant" if self.scheme_b_quant_enabled else "miss",
                        )
                        manifest_batch_seen += 1
            current_timing["save_s"] = time.perf_counter() - save_start

        self._apply_scheme_b_quant_debug_zero_base(cache_items, quant_base_payloads, non_skip_item_count)
        self._apply_scheme_b_ablation(cache_items, graph, skip_cache_indices=skip_cache_indices)

        self._sync_encoder_cache_timer(device)
        assemble_start = time.perf_counter()
        hidden_dtype = cache_items[0]["memory_state"].dtype
        memory_states = torch.zeros(
            (len(cache_items), self.mem_size, cache_items[0]["memory_state"].size(-1)),
            dtype=hidden_dtype,
            device=padded_token_ids.device,
        )
        for i, cache_item in enumerate(cache_items):
            memory_states[i] = cache_item["memory_state"].to(padded_token_ids.device)
        self._sync_encoder_cache_timer(device)
        current_timing["assemble_s"] = time.perf_counter() - assemble_start

        self.encoder_cache_calls += 1
        current_skips = len(skipped)
        current_misses = len(missing) - current_skips
        current_hits = len(token_ids) - len(missing)
        self.encoder_cache_hits += current_hits
        self.encoder_cache_misses += current_misses
        self.encoder_cache_skips += current_skips
        self._add_scheme_b_quant_stats(current_quant_stats)

        self._sync_encoder_cache_timer(device)
        suffix_start = time.perf_counter()
        with self._encoder_suffix_quant_context():
            final_memory_states, mapped_items = base_model.forward_memory_with_text_kv(
                memory_states=memory_states,
                text_kv_items=cache_items,
                graph=graph,
                map_node=True,
            )
        final_hidden_states = torch.zeros(
            (len(mapped_items), padded_token_ids.size(1), final_memory_states.size(-1)),
            dtype=final_memory_states.dtype,
            device=padded_token_ids.device,
        )
        mapped_mem_mask = self._memory_kv_mapped_mem_mask(mapped_items, padded_token_ids.size(1), padded_token_ids.device)
        final_hidden_states[mapped_mem_mask] = final_memory_states.reshape(-1, final_memory_states.size(-1))
        self._sync_encoder_cache_timer(device)
        current_timing["suffix_compute_s"] = time.perf_counter() - suffix_start

        if self.encoder_cache_verify:
            previous_profile_state = getattr(base_model, "profile_stage_times", False)
            base_model.profile_stage_times = False
            try:
                with torch.no_grad():
                    reference_embeddings = self.model.tokens_to_embeddings(padded_token_ids)
                    with self._encoder_suffix_quant_context():
                        reference_hidden_states = self.model.icae(
                            inputs_embeds=reference_embeddings,
                            output_hidden_states=True,
                            graph=graph,
                            mem_mask=mem_mask,
                            partial_grad=partial_grad,
                            map_node=True,
                        ).hidden_states[-1]
            finally:
                base_model.profile_stage_times = previous_profile_state
            diff_abs = (
                final_hidden_states[mapped_mem_mask].float()
                - reference_hidden_states[mapped_mem_mask].float()
            ).abs()
            reference = reference_hidden_states[mapped_mem_mask].float()
            if diff_abs.numel():
                flat_diff = diff_abs.flatten()
                sampled_diff = self._encoder_cache_verify_sample(flat_diff)
                max_abs = flat_diff.max().item()
                mean_abs = flat_diff.mean().item()
                p95_abs = self._encoder_cache_verify_quantile(sampled_diff, 0.95)
                p99_abs = self._encoder_cache_verify_quantile(sampled_diff, 0.99)
                relative_l2 = (
                    torch.linalg.vector_norm(flat_diff) /
                    torch.clamp(torch.linalg.vector_norm(reference), min=1e-12)
                ).item()
                above_strict_ratio = (
                    sampled_diff > self.encoder_cache_verify_tolerance
                ).float().mean().item()
            else:
                max_abs = 0.0
                mean_abs = 0.0
                p95_abs = 0.0
                p99_abs = 0.0
                relative_l2 = 0.0
                above_strict_ratio = 0.0

            exact_passed = max_abs <= self.encoder_cache_verify_tolerance
            practical_passed = (
                mean_abs <= self.encoder_cache_verify_mean_tolerance and
                p99_abs <= self.encoder_cache_verify_p99_tolerance and
                relative_l2 <= self.encoder_cache_verify_relative_l2_tolerance and
                max_abs <= self.encoder_cache_verify_max_tolerance
            )
            self.encoder_cache_verify_calls += 1
            if not exact_passed:
                self.encoder_cache_verify_exact_failures += 1
            if not practical_passed:
                self.encoder_cache_verify_practical_failures += 1
            self.encoder_cache_verify_running["mean_abs_sum"] += mean_abs
            self.encoder_cache_verify_running["p99_abs_sum"] += p99_abs
            self.encoder_cache_verify_running["relative_l2_sum"] += relative_l2
            self.encoder_cache_verify_running["max_abs_worst"] = max(
                self.encoder_cache_verify_running["max_abs_worst"], max_abs
            )

            verify_interval = max(int(self.encoder_cache_verify_log_interval), 1)
            should_log_verify = (
                self.encoder_cache_verify_calls <= 3 or
                self.encoder_cache_verify_calls % verify_interval == 0 or
                not practical_passed
            )
            if should_log_verify:
                calls = self.encoder_cache_verify_calls
                practical_pass_rate = 1.0 - self.encoder_cache_verify_practical_failures / calls
                print(
                    "GOFA memory/text-KV cache verification: "
                    f"call={self.encoder_cache_calls}, "
                    f"exact_passed={exact_passed}, practical_passed={practical_passed}, "
                    f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
                    f"p95_abs={p95_abs:.6g}, p99_abs={p99_abs:.6g}, "
                    f"relative_l2={relative_l2:.6g}, "
                    f"above_strict_ratio={above_strict_ratio:.4%}, "
                    f"practical_pass_rate={practical_pass_rate:.2%}, "
                    f"running_mean_abs={self.encoder_cache_verify_running['mean_abs_sum'] / calls:.6g}, "
                    f"running_p99_abs={self.encoder_cache_verify_running['p99_abs_sum'] / calls:.6g}, "
                    f"running_relative_l2={self.encoder_cache_verify_running['relative_l2_sum'] / calls:.6g}, "
                    f"worst_max_abs={self.encoder_cache_verify_running['max_abs_worst']:.6g}"
                )

        current_timing["total_s"] = time.perf_counter() - total_start
        for key, value in current_timing.items():
            self.encoder_cache_timing[key] += value

        if self.encoder_cache_calls <= 3 or self.encoder_cache_calls % 50 == 0:
            print(
                "GOFA encoder memory/text-KV cache: "
                f"call={self.encoder_cache_calls}, hits={current_hits}, "
                f"misses={current_misses}, skips={current_skips}, "
                f"total_hits={self.encoder_cache_hits}, total_misses={self.encoder_cache_misses}, "
                f"total_skips={self.encoder_cache_skips}"
            )
            self._encoder_cache_log_timing(current_timing)
        self._maybe_log_scheme_b_quant_stats(current_quant_stats)
        self._maybe_log_scheme_b_int_gemm_stats()
        self._maybe_log_scheme_b_quant_kv_attention_stats()
        self._maybe_log_scheme_b_activation_quant_stats()
        self._maybe_dump_encoder_cache_manifest(current_batch_seen=manifest_batch_seen)

        return final_hidden_states

    def forward(self, g):
        """
        Encode the graph and generate logits for answer tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        if not hasattr(g, "answer"):
            raise ValueError("Forward stage graph should contain answer.")
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        # Legacy hard coding TODO: remove when TAGLAS is fixed.
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        answer_logits, answer_id, masks = self.decode(answer_texts, emb, prompt=prompt_input_texts)
        return answer_logits, answer_id, masks, answer_texts

    def generate(self, g, max_length=128):
        """
        Autoregressively generate tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        generated_text = self.infer(emb, prompt=prompt_input_texts, max_length=max_length)
        return generated_text

    def encode(self, data, graph=None, partial_grad=None):
        cur_device = self.model.memory_token_embed.weight.device
        batch_size = len(data)
        text_output = \
        self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length, padding=False,
                             return_attention_mask=False)["input_ids"]

        text_output = [t + self.mem_tokens for t in text_output]
        padded_text_output = {"input_ids": text_output}
        padded_text_output = self.model.tokenizer.pad(padded_text_output, padding=True, return_tensors="pt")["input_ids"].to(
            cur_device)
        mem_mask = padded_text_output >= self.model.vocab_size

        mem_mask = mem_mask.to(cur_device)

        # Use ICAE lora only in the encoder.
        self.model.icae.set_adapter("encadapt")
        self.model.icae.enable_adapter_layers()
        for name, param in self.model.icae.named_parameters():
            if "encadapt" in name:
                param.requires_grad = False
        compress_outputs = None
        if self.encoder_cache_enabled and graph is not None:
            if self.encoder_cache_mode == "memory_kv":
                compress_outputs = self._encode_with_memory_kv_cache(
                    text_output,
                    padded_text_output,
                    mem_mask,
                    graph=graph,
                    partial_grad=partial_grad,
                    skip_cache_indices=self._encoder_cache_skip_indices(graph),
                )
            else:
                compress_outputs = self._encode_with_encoder_cache(
                    text_output,
                    padded_text_output,
                    mem_mask,
                    graph=graph,
                    partial_grad=partial_grad,
                    skip_cache_indices=self._encoder_cache_skip_indices(graph),
                )
        if compress_outputs is None:
            self._sync_encoder_cache_timer(cur_device)
            full_start = time.perf_counter()
            autoencoder_input_embedding = self.model.tokens_to_embeddings(padded_text_output)
            with self._encoder_suffix_quant_context():
                compress_outputs = self.model.icae(
                    inputs_embeds=autoencoder_input_embedding,
                    output_hidden_states=True,
                    graph=graph,
                    mem_mask=mem_mask,
                    partial_grad=partial_grad,
                    map_node=True,
                )
            compress_outputs = compress_outputs.hidden_states[-1]
            self._sync_encoder_cache_timer(cur_device)
            full_elapsed = time.perf_counter() - full_start
            self.encoder_full_calls += 1
            self.encoder_full_time_s += full_elapsed
            if self.encoder_full_calls <= 3 or self.encoder_full_calls % 50 == 0:
                print(
                    "GOFA encoder full path timing: "
                    f"call={self.encoder_full_calls}, "
                    f"current={full_elapsed:.4f}s, "
                    f"cum_total={self.encoder_full_time_s:.4f}s"
                )
            self._maybe_log_scheme_b_int_gemm_stats()
            self._maybe_log_scheme_b_quant_kv_attention_stats()
            self._maybe_log_scheme_b_activation_quant_stats()
        self.model.icae.disable_adapter_layers()

        if graph is not None:
            node_emb = compress_outputs[:len(graph.node_map)]
            map_mem_mask = mem_mask[:graph.num_node_feat][graph.node_map]
            memory_embedding = node_emb[map_mem_mask].view(len(node_emb), self.mem_size, -1)
        else:
            memory_embedding = compress_outputs[mem_mask].view(batch_size, self.mem_size, -1)
        return memory_embedding

    def decode(self, data, mem_embs, graph=None, prompt=None):
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] if len(p) < self.model.training_args.model_max_length else p[:self.model.training_args.model_max_length] for p in prompt_output]
        original_prompt_output = prompt_output

        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.left_tokenizer(prompt, add_special_tokens=False, padding=False, truncation=True, max_length=512)["input_ids"]
        batch_size = len(prompt_input)

        # For Mistral, decode contains: prefix, memory slots and suffix
        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]
        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]
        prompt_ids = [a + [self.model.tokenizer.pad_token_id] * self.mem_size + b + c for a, b, c in
                      zip(prompt_left_ids, prompt_right_ids, prompt_output)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                prompt_output[i]) + [False] for i in range(batch_size)]

        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(
            mem_embs.device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")
        prompt_answer_ids = prompt_output["input_ids"].to(mem_embs.device)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * (
                len(prompt_output["input_ids"][i]) - len(prompt_left_ids[i]) - self.mem_size) for i in
                    range(batch_size)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                original_prompt_output[i]) + [False] * (1 + len(prompt_output["input_ids"][i]) - len(prompt_ids[i])) for
            i in range(batch_size)]

        prompt_answer_embs[torch.tensor(mem_mask)] = mem_embs.view(-1, mem_embs.size()[-1])

        target_mask = torch.tensor(prompt_mask, dtype=torch.long, device=mem_embs.device).to(torch.bool)

        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        decoder_start = self._stage_timer_start(mem_embs.device)
        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs).logits
        self._stage_timer_add_decoder(decoder_start, mem_embs.device)
        self._maybe_log_stage_profile()

        return output_emb, answer_prompt, target_mask

    def infer(self, mem_embs, graph=None, prompt=None, max_length=128):
        cur_device = self.model.memory_token_embed.weight.device

        if prompt is None:
            prompt = [""] * len(mem_embs)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        batch_size = len(prompt_input)

        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]

        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * len(prompt_right_ids[i]) for
                    i in range(batch_size)]
        att_mask = [[True] * (len(prompt_left_ids[i]) + self.mem_size + len(prompt_right_ids[i])) for i in
                    range(batch_size)]
        prompt_ids = [prompt_left_ids[i] + [self.model.tokenizer.pad_token_id] * self.mem_size + prompt_right_ids[i] for
                      i in range(batch_size)]

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": mem_mask},
                                                         padding=True, return_tensors="pt")
        mem_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device, dtype=torch.bool)

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")
        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device)

        prompt_answer_ids = prompt_ids.to(device=mem_embs.device, dtype=torch.long)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)
        prompt_answer_embs[mem_mask] = mem_embs.view(-1, mem_embs.size()[-1])

        decode_embed = prompt_answer_embs
        output = decode_embed.clone()

        generate_text = []
        eos_reached = torch.zeros(len(output), dtype=torch.bool).to(output.device)

        past_key_values = None
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        decoder_start = self._stage_timer_start(mem_embs.device)
        for i in range(max_length):
            out = self.model.icae(inputs_embeds=output, attention_mask=att_mask, past_key_values=past_key_values,
                                 use_cache=True)

            logits = out.logits[:, -1, :self.model.vocab_size - 1]

            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id==self.model.tokenizer.bos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id>=32000).view(-1))

            output = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(mem_embs.device)

            generate_text.append(next_token_id.view(-1, 1))
            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)

            if torch.all(eos_reached):
                break
        self._stage_timer_add_decoder(decoder_start, mem_embs.device)
        self._maybe_log_stage_profile()

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

        return generated_text
