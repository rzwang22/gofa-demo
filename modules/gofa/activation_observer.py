from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from modules.gofa.weight_quant import (
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
    _module_weight,
    _resolve_submodule,
)


SUPPORTED_OBSERVER_QUANT_BITS = {4, 8}
SUPPORTED_OBSERVER_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "mlp", *MLP_PROJECTIONS}


@dataclass
class ActivationObservedModule:
    layer_idx: int
    projection_type: str
    module_name: str
    module: nn.Module
    weight_shape: Optional[Tuple[int, ...]]


def _as_int_list(value, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(default)
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    return [int(value)]


def _as_str_list(value, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(default)
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_clip_ratio(clip_ratio: float) -> float:
    clip_ratio = float(clip_ratio)
    if clip_ratio <= 0.0 or clip_ratio > 1.0:
        raise ValueError("scheme_b_activation_observer.clip_ratio must be in (0, 1].")
    return clip_ratio


def _normalize_quant_bits(bits) -> List[int]:
    result = _as_int_list(bits, [4, 8])
    invalid = [bit for bit in result if bit not in SUPPORTED_OBSERVER_QUANT_BITS]
    if invalid:
        raise ValueError(
            "scheme_b_activation_observer.quant_bits must contain only "
            f"{sorted(SUPPORTED_OBSERVER_QUANT_BITS)}; got {invalid}."
        )
    return result


def _json_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return numerator / denominator.clamp(min=eps)


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _tensor_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.reshape(1, tensor.size(0))
    return tensor.reshape(-1, tensor.size(-1))


def _uniform_indices(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if length <= limit:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=limit, device=device).round().long()


def _sample_2d(tensor_2d: torch.Tensor, sample_tokens: int, sample_channels: int) -> torch.Tensor:
    token_limit = max(int(sample_tokens), 1)
    channel_limit = max(int(sample_channels), 1)
    token_idx = _uniform_indices(int(tensor_2d.size(0)), token_limit, tensor_2d.device)
    channel_idx = _uniform_indices(int(tensor_2d.size(1)), channel_limit, tensor_2d.device)
    return tensor_2d.index_select(0, token_idx).index_select(1, channel_idx)


def quantize_activation_with_q(
        activation: torch.Tensor,
        bits: int,
        per_token: bool = True,
        clip_ratio: float = 1.0,
        eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = int(bits)
    if bits not in SUPPORTED_OBSERVER_QUANT_BITS:
        raise ValueError(f"Unsupported activation observer quant bits={bits}.")
    activation_f = activation.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    abs_activation = activation_f.abs()
    if per_token:
        if clip_ratio < 1.0:
            max_abs = torch.quantile(abs_activation, clip_ratio, dim=-1, keepdim=True)
        else:
            max_abs = abs_activation.amax(dim=-1, keepdim=True)
    else:
        if clip_ratio < 1.0:
            max_abs = torch.quantile(abs_activation.flatten(), clip_ratio).reshape([1] * activation_f.dim())
        else:
            max_abs = abs_activation.amax().reshape([1] * activation_f.dim())
    if clip_ratio < 1.0:
        activation_f = activation_f.clamp(-max_abs, max_abs)
    scale = (max_abs / float(qmax)).clamp(min=eps)
    q = torch.round(activation_f / scale).clamp(qmin, qmax)
    return q, q * scale, scale


class SuffixTransformerActivationObserver:
    """
    Captures sampled suffix Transformer Linear input activations.

    Hooks are registered at construction time but record only inside activate().
    Register this observer before activation quantization hooks if both are
    enabled, so the observer sees the original input activation.
    """

    def __init__(
            self,
            base_model: nn.Module,
            output_dir: str,
            max_batches: int = 2,
            max_items_per_module: int = 4,
            layers: Optional[Sequence[int]] = None,
            projections: Optional[Sequence[str]] = None,
            save_tensor: bool = True,
            save_stats: bool = True,
            sample_tokens: int = 512,
            sample_channels: int = 256,
            compute_quant_error: bool = True,
            quant_bits: Optional[Sequence[int]] = None,
            per_token: bool = True,
            clip_ratio: float = 1.0,
            log_interval: int = 20,
            task="",
            cache_mode: str = "",
            log_observed_modules: bool = True,
            logger=print):
        if not output_dir:
            raise ValueError("scheme_b_activation_observer.output_dir must be set when observer is enabled.")
        self.base_model = base_model
        self.output_dir = os.path.abspath(str(output_dir))
        self.tensor_dir = os.path.join(self.output_dir, "tensors")
        self.max_batches = max(int(max_batches), 1)
        self.max_items_per_module = max(int(max_items_per_module), 1)
        self.layers = _as_int_list(layers, [26, 29, 31])
        self.projections = _as_str_list(projections, ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"])
        invalid_projections = [projection for projection in self.projections if projection not in SUPPORTED_OBSERVER_PROJECTIONS]
        if invalid_projections:
            raise ValueError(f"Unsupported scheme_b_activation_observer.projections={invalid_projections}.")
        self.save_tensor = bool(save_tensor)
        self.save_stats = bool(save_stats)
        self.sample_tokens = max(int(sample_tokens), 1)
        self.sample_channels = max(int(sample_channels), 1)
        self.compute_quant_error = bool(compute_quant_error)
        self.quant_bits = _normalize_quant_bits(quant_bits)
        self.per_token = bool(per_token)
        self.clip_ratio = _normalize_clip_ratio(clip_ratio)
        self.log_interval = max(int(log_interval), 1)
        self.task = task
        self.cache_mode = str(cache_mode)
        self.log_observed_modules = bool(log_observed_modules)
        self.logger = logger
        self.active_depth = 0
        self.current_batch_idx: Optional[int] = None
        self.next_batch_idx = 0
        self.modules: List[ActivationObservedModule] = []
        self.module_to_item: Dict[int, ActivationObservedModule] = {}
        self.module_batch_item_counts: Dict[Tuple[str, int], int] = {}
        self.handles = []
        self.stats = {
            "enabled": True,
            "output_dir": self.output_dir,
            "observed_batches": 0,
            "saved_tensors": 0,
            "stats_records": 0,
            "hooked_module_count": 0,
            "observer_time_s": 0.0,
            "layers": list(self.layers),
            "projections": list(self.projections),
        }
        self.stats_path = os.path.join(self.output_dir, "activation_stats.jsonl")
        self.summary_path = os.path.join(self.output_dir, "activation_observer_summary.json")
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_tensor:
            os.makedirs(self.tensor_dir, exist_ok=True)
        if self.save_stats and os.path.exists(self.stats_path):
            os.remove(self.stats_path)
        self._collect_modules()
        self._register_hooks()
        self._log_summary()
        atexit.register(self.dump_summary)

    def _suffix_layer_range(self) -> Tuple[int, int]:
        start = int(getattr(self.base_model, "gnn_start_layer"))
        end = int(self.base_model.config.num_hidden_layers)
        return start, end

    def _target_paths(self) -> List[Tuple[str, str]]:
        paths: List[Tuple[str, str]] = []
        requested = set(self.projections)
        for name in ATTENTION_PROJECTIONS:
            if name in requested:
                paths.append((f"self_attn.{name}", name))
        if "mlp" in requested:
            paths.extend((f"mlp.{name}", "mlp") for name in MLP_PROJECTIONS)
        else:
            for name in MLP_PROJECTIONS:
                if name in requested:
                    paths.append((f"mlp.{name}", "mlp"))
        return paths

    def _collect_modules(self):
        if not hasattr(self.base_model, "layers") or not hasattr(self.base_model, "gnn_start_layer"):
            raise ValueError("Activation observer requires a GOFA Mistral base model with layers and gnn_start_layer.")
        start, end = self._suffix_layer_range()
        target_layers = [layer_idx for layer_idx in self.layers if start <= int(layer_idx) < end]
        skipped_layers = [layer_idx for layer_idx in self.layers if int(layer_idx) < start or int(layer_idx) >= end]
        if skipped_layers:
            self.logger(
                "GOFA activation observer skipped non-suffix layers: "
                f"requested={skipped_layers}, suffix_range=[{start}, {end})"
            )
        for layer_idx in target_layers:
            decoder_layer = self.base_model.layers[int(layer_idx)]
            for path, projection_type in self._target_paths():
                module = _resolve_submodule(decoder_layer, path)
                if module is None:
                    continue
                weight = _module_weight(module)
                item = ActivationObservedModule(
                    layer_idx=int(layer_idx),
                    projection_type=projection_type,
                    module_name=f"layers.{int(layer_idx)}.{path}",
                    module=module,
                    weight_shape=tuple(weight.shape) if weight is not None else None,
                )
                self.modules.append(item)
                self.module_to_item[id(module)] = item
        self.stats["hooked_module_count"] = len(self.modules)

    def _register_hooks(self):
        for item in self.modules:
            self.handles.append(item.module.register_forward_pre_hook(self._pre_forward_hook))

    def _maybe_log_runtime(self):
        observed_batches = int(self.stats["observed_batches"])
        if observed_batches <= 3 or observed_batches % self.log_interval == 0:
            self.logger(
                "GOFA activation observer: "
                f"enabled=True, output_dir={self.output_dir}, "
                f"observed_batches={self.stats['observed_batches']}, "
                f"saved_tensors={self.stats['saved_tensors']}, "
                f"stats_records={self.stats['stats_records']}, "
                f"layers={self.layers}, projections={self.projections}"
            )

    def _pre_forward_hook(self, module: nn.Module, inputs):
        if self.active_depth <= 0 or self.current_batch_idx is None:
            return
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        item = self.module_to_item.get(id(module))
        if item is None:
            return
        start_time = time.perf_counter()
        activation = inputs[0].detach()
        batch_idx = int(self.current_batch_idx)
        key = (item.module_name, batch_idx)
        already_saved = int(self.module_batch_item_counts.get(key, 0))
        if already_saved >= self.max_items_per_module:
            return
        record_item_idx = already_saved
        sampled = _sample_2d(_tensor_to_2d(activation), self.sample_tokens, self.sample_channels).detach().cpu()
        record = self._build_record(item, sampled, batch_idx, record_item_idx, tuple(activation.shape))
        if self.save_tensor:
            self._save_tensor_payload(item, sampled, batch_idx, record_item_idx, tuple(activation.shape))
        if self.save_stats:
            self._append_stats_record(record)
        already_saved += 1
        self.stats["stats_records"] += 1
        self.module_batch_item_counts[key] = already_saved
        self.stats["observer_time_s"] += time.perf_counter() - start_time

    def _save_tensor_payload(
            self,
            item: ActivationObservedModule,
            sampled: torch.Tensor,
            batch_idx: int,
            item_idx: int,
            original_shape: Tuple[int, ...]):
        module_slug = _sanitize_filename(item.module_name)
        filename = f"layer{item.layer_idx}_{item.projection_type}_{module_slug}_batch{batch_idx}_item{item_idx}.pt"
        path = os.path.join(self.tensor_dir, filename)
        payload = {
            "layer_idx": item.layer_idx,
            "projection_type": item.projection_type,
            "module_name": item.module_name,
            "batch_idx": batch_idx,
            "item_idx": item_idx,
            "original_shape": tuple(original_shape),
            "saved_tensor": sampled,
            "dtype": str(sampled.dtype).replace("torch.", ""),
            "task": self.task,
            "cache_mode": self.cache_mode,
        }
        torch.save(payload, path)
        self.stats["saved_tensors"] += 1

    def _append_stats_record(self, record: Dict):
        with open(self.stats_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _build_record(
            self,
            item: ActivationObservedModule,
            sampled: torch.Tensor,
            batch_idx: int,
            item_idx: int,
            original_shape: Tuple[int, ...]) -> Dict:
        tensor = sampled.to(torch.float32)
        abs_tensor = tensor.abs()
        flat_abs = abs_tensor.flatten()
        abs_p99 = torch.quantile(flat_abs, 0.99)
        abs_p999 = torch.quantile(flat_abs, 0.999)
        token_abs_max = abs_tensor.amax(dim=1)
        token_abs_p99 = torch.quantile(abs_tensor, 0.99, dim=1)
        channel_abs_max = abs_tensor.amax(dim=0)
        channel_abs_p99 = torch.quantile(abs_tensor, 0.99, dim=0)
        record = {
            "layer_idx": item.layer_idx,
            "projection_type": item.projection_type,
            "module_name": item.module_name,
            "batch_idx": batch_idx,
            "item_idx": item_idx,
            "shape": [int(tensor.size(0)), int(tensor.size(1))],
            "original_shape": list(original_shape),
            "dtype": str(sampled.dtype).replace("torch.", ""),
            "task": self.task,
            "cache_mode": self.cache_mode,
            "abs_max": _json_float(flat_abs.max()),
            "abs_mean": _json_float(flat_abs.mean()),
            "abs_std": _json_float(flat_abs.std(unbiased=False)),
            "abs_p50": _json_float(torch.quantile(flat_abs, 0.50)),
            "abs_p90": _json_float(torch.quantile(flat_abs, 0.90)),
            "abs_p99": _json_float(abs_p99),
            "abs_p999": _json_float(abs_p999),
            "max_over_p99": _json_float(_safe_ratio(flat_abs.max(), abs_p99)),
            "max_over_p999": _json_float(_safe_ratio(flat_abs.max(), abs_p999)),
            "token_abs_max_mean": _json_float(token_abs_max.mean()),
            "token_abs_max_max": _json_float(token_abs_max.max()),
            "token_max_over_p99_mean": _json_float(_safe_ratio(token_abs_max, token_abs_p99).mean()),
            "token_max_over_p99_max": _json_float(_safe_ratio(token_abs_max, token_abs_p99).max()),
            "channel_abs_max_mean": _json_float(channel_abs_max.mean()),
            "channel_abs_max_max": _json_float(channel_abs_max.max()),
            "channel_max_over_p99_mean": _json_float(_safe_ratio(channel_abs_max, channel_abs_p99).mean()),
            "channel_max_over_p99_max": _json_float(_safe_ratio(channel_abs_max, channel_abs_p99).max()),
        }
        if self.compute_quant_error:
            record.update(self._quant_error_stats(tensor))
        return record

    def _quant_error_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        result: Dict[str, float] = {}
        flat = tensor.flatten()
        flat_norm = flat.norm().clamp(min=1e-12)
        flat_abs = flat.abs()
        for bits in self.quant_bits:
            q, dq, _scale = quantize_activation_with_q(
                tensor,
                bits=bits,
                per_token=self.per_token,
                clip_ratio=self.clip_ratio,
            )
            err = (dq - tensor).flatten()
            dq_flat = dq.flatten()
            denom = (flat.norm() * dq_flat.norm()).clamp(min=1e-12)
            cosine = torch.dot(flat, dq_flat) / denom
            q_abs = q.abs()
            qmax = (1 << (int(bits) - 1)) - 1
            prefix = f"quant_error_{bits}bit"
            result[f"{prefix}_rel_l2"] = _json_float(err.norm() / flat_norm)
            result[f"{prefix}_cosine"] = _json_float(cosine)
            result[f"{prefix}_zero_ratio"] = _json_float((q == 0).to(torch.float32).mean())
            result[f"{prefix}_saturation_ratio"] = _json_float((q_abs == qmax).to(torch.float32).mean())
            result[f"{prefix}_mean_abs_error"] = _json_float(err.abs().mean())
            result[f"{prefix}_max_abs_error"] = _json_float(err.abs().max())
            result[f"{prefix}_error_over_abs_mean"] = _json_float(_safe_ratio(err.abs().mean(), flat_abs.mean()))
        return result

    @contextmanager
    def activate(self):
        outermost = self.active_depth == 0
        if outermost:
            if self.next_batch_idx < self.max_batches:
                self.current_batch_idx = self.next_batch_idx
                self.next_batch_idx += 1
                self.stats["observed_batches"] = self.next_batch_idx
                self._maybe_log_runtime()
            else:
                self.current_batch_idx = None
        self.active_depth += 1
        try:
            yield self
        finally:
            self.active_depth = max(self.active_depth - 1, 0)
            if outermost:
                self.current_batch_idx = None

    def dump_summary(self):
        os.makedirs(self.output_dir, exist_ok=True)
        summary = dict(self.stats)
        summary.update(
            {
                "max_batches": self.max_batches,
                "max_items_per_module": self.max_items_per_module,
                "sample_tokens": self.sample_tokens,
                "sample_channels": self.sample_channels,
                "compute_quant_error": self.compute_quant_error,
                "quant_bits": list(self.quant_bits),
                "per_token": self.per_token,
                "clip_ratio": self.clip_ratio,
                "task": self.task,
                "cache_mode": self.cache_mode,
                "stats_path": self.stats_path if self.save_stats else "",
                "tensor_dir": self.tensor_dir if self.save_tensor else "",
            }
        )
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary

    def _log_summary(self):
        if self.log_observed_modules:
            for item in self.modules:
                self.logger(
                    "GOFA activation observer hooked module: "
                    f"layer={item.layer_idx}, projection={item.projection_type}, module={item.module_name}, "
                    f"weight_shape={item.weight_shape}"
                )
        self.logger(
            "GOFA activation observer initialized: "
            f"enabled=True, output_dir={self.output_dir}, max_batches={self.max_batches}, "
            f"max_items_per_module={self.max_items_per_module}, layers={self.layers}, "
            f"projections={self.projections}, hooked_module_count={self.stats['hooked_module_count']}, "
            f"save_tensor={self.save_tensor}, save_stats={self.save_stats}, "
            f"sample_tokens={self.sample_tokens}, sample_channels={self.sample_channels}, "
            f"compute_quant_error={self.compute_quant_error}, quant_bits={self.quant_bits}, "
            f"per_token={self.per_token}, clip_ratio={self.clip_ratio}"
        )


def maybe_create_suffix_transformer_activation_observer(
        base_model: nn.Module,
        cfg: Dict,
        task="",
        cache_mode: str = "",
        logger=print) -> Optional[SuffixTransformerActivationObserver]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    target = str(cfg.get("target", "suffix_transformer"))
    if target != "suffix_transformer":
        raise ValueError("scheme_b_activation_observer.target currently supports only 'suffix_transformer'.")
    return SuffixTransformerActivationObserver(
        base_model,
        output_dir=str(cfg.get("output_dir", "")),
        max_batches=int(cfg.get("max_batches", 2)),
        max_items_per_module=int(cfg.get("max_items_per_module", 4)),
        layers=cfg.get("layers", [26, 29, 31]),
        projections=cfg.get("projections", ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]),
        save_tensor=bool(cfg.get("save_tensor", True)),
        save_stats=bool(cfg.get("save_stats", True)),
        sample_tokens=int(cfg.get("sample_tokens", 512)),
        sample_channels=int(cfg.get("sample_channels", 256)),
        compute_quant_error=bool(cfg.get("compute_quant_error", True)),
        quant_bits=cfg.get("quant_bits", [4, 8]),
        per_token=bool(cfg.get("per_token", True)),
        clip_ratio=float(cfg.get("clip_ratio", 1.0)),
        log_interval=int(cfg.get("log_interval", 20)),
        task=task,
        cache_mode=cache_mode,
        logger=logger,
    )


def activation_observer_context(observer: Optional[SuffixTransformerActivationObserver]):
    if observer is None:
        return nullcontext()
    return observer.activate()
