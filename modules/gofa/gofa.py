# example code for running inference with fine-tuned checkpoint
from typing import Optional

import hashlib
import json
import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import MistralConfig

from modules.gofa.gofa_icae import MistralICAE
from collections import OrderedDict
from safetensors.torch import load_file
from modules.utils import safe_download_hf_file

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
    encoder_cache_verify: bool = field(default=False, metadata={"help": "Compare memory_kv cache output against the full encoder path"})
    encoder_cache_verify_tolerance: float = field(default=1e-3, metadata={"help": "Strict max-absolute tolerance for exact verification reporting"})
    encoder_cache_verify_mean_tolerance: float = field(default=3e-2)
    encoder_cache_verify_p99_tolerance: float = field(default=2.5e-1)
    encoder_cache_verify_relative_l2_tolerance: float = field(default=8e-2)
    encoder_cache_verify_max_tolerance: float = field(default=1.5)
    encoder_cache_verify_log_interval: int = field(default=1)
    profile_stage_times: bool = field(default=False, metadata={"help": "Synchronize and profile encoder/decoder stages"})
    profile_stage_log_interval: int = field(default=50, metadata={"help": "Log stage timing every N decoder calls"})


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
        self.stage_profile_reports = 0
        self.decoder_stage_calls = 0
        self.decoder_stage_time_s = 0.0
        self.encoder_cache_enabled = bool(model_args.use_encoder_cache)
        self.encoder_cache_dir = model_args.encoder_cache_dir
        self.encoder_cache_mode = model_args.encoder_cache_mode
        self.encoder_cache_verify = bool(model_args.encoder_cache_verify)
        self.encoder_cache_verify_tolerance = model_args.encoder_cache_verify_tolerance
        self.encoder_cache_verify_mean_tolerance = model_args.encoder_cache_verify_mean_tolerance
        self.encoder_cache_verify_p99_tolerance = model_args.encoder_cache_verify_p99_tolerance
        self.encoder_cache_verify_relative_l2_tolerance = model_args.encoder_cache_verify_relative_l2_tolerance
        self.encoder_cache_verify_max_tolerance = model_args.encoder_cache_verify_max_tolerance
        self.encoder_cache_verify_log_interval = model_args.encoder_cache_verify_log_interval
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
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
        }
        self.encoder_full_calls = 0
        self.encoder_full_time_s = 0.0
        self.encoder_cache_namespace = self._build_encoder_cache_namespace(dir, model_args, training_args, gofa_args)
        base_model = self.model.icae.get_base_model().model
        if hasattr(base_model, "profile_stage_times"):
            base_model.profile_stage_times = self.profile_stage_times
            base_model.reset_stage_profile()
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
                            f"log_interval={self.encoder_cache_verify_log_interval}"
                        )
        if self.profile_stage_times:
            print(
                "GOFA stage profiler enabled: "
                f"log_interval={self.profile_stage_log_interval}, "
                "timings include cuda synchronization overhead"
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

    def _encoder_cache_key(self, token_ids):
        token_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(token_bytes).hexdigest()

    def _encoder_cache_path(self, cache_key):
        return os.path.join(self._encoder_cache_root(), cache_key[:2], cache_key + ".pt")

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
        print(
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

        return final_hidden_states

    def _memory_kv_mapped_mem_mask(self, mapped_items, max_seq_len, device):
        mem_mask = torch.zeros((len(mapped_items), max_seq_len), dtype=torch.bool, device=device)
        for i, item in enumerate(mapped_items):
            start = item["text_len"]
            end = start + self.mem_size
            mem_mask[i, start:end] = True
        return mem_mask

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
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
        }

        skip_cache_indices = set(skip_cache_indices or [])
        cache_items = [None] * len(token_ids)
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
            cache_item, cache_key = self._load_encoder_memory_kv_cache_item(ids)
            if cache_item is None:
                missing.append(i)
                missing_keys.append(cache_key)
            else:
                cache_items[i] = cache_item
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

            built_items = []
            for batch_idx, original_idx in enumerate(missing):
                seq_len = len(token_ids[original_idx])
                text_len = seq_len - self.mem_size
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
            current_timing["save_s"] = time.perf_counter() - save_start

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

        self._sync_encoder_cache_timer(device)
        suffix_start = time.perf_counter()
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
                quantiles = torch.quantile(
                    flat_diff,
                    torch.tensor([0.95, 0.99], device=flat_diff.device, dtype=flat_diff.dtype),
                )
                max_abs = flat_diff.max().item()
                mean_abs = flat_diff.mean().item()
                p95_abs = quantiles[0].item()
                p99_abs = quantiles[1].item()
                relative_l2 = (
                    torch.linalg.vector_norm(flat_diff) /
                    torch.clamp(torch.linalg.vector_norm(reference), min=1e-12)
                ).item()
                above_strict_ratio = (
                    flat_diff > self.encoder_cache_verify_tolerance
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
            compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                               graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=True)
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
