from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


SUPPORTED_WEIGHT_BITS = {4, 8}
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass
class WeightQuantizedModule:
    name: str
    module: nn.Module
    weight: torch.nn.Parameter
    shape: Tuple[int, ...]
    scale_shape: Tuple[int, ...]
    original_bytes: int
    effective_bytes: int


def _normalize_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_WEIGHT_BITS:
        raise ValueError(f"Unsupported suffix weight quantization bits={bits}; expected one of {sorted(SUPPORTED_WEIGHT_BITS)}.")
    return bits


def _scale_view(scale: torch.Tensor, shape: Iterable[int], channel_axis: int) -> torch.Tensor:
    view_shape = [1] * len(tuple(shape))
    view_shape[int(channel_axis)] = scale.numel()
    return scale.reshape(view_shape)


def quantize_dequant_weight_symmetric_per_channel(
        weight: torch.Tensor,
        bits: int = 4,
        channel_axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    bits = _normalize_bits(bits)
    if weight.dim() == 0:
        raise ValueError("Per-channel weight quantization requires a tensor with at least one dimension.")
    channel_axis = int(channel_axis)
    if channel_axis < 0:
        channel_axis += weight.dim()
    if channel_axis < 0 or channel_axis >= weight.dim():
        raise ValueError(f"Invalid channel_axis={channel_axis} for weight shape={tuple(weight.shape)}.")

    original_dtype = weight.dtype
    weight_f = weight.detach().to(torch.float32)
    reduce_dims = tuple(dim for dim in range(weight_f.dim()) if dim != channel_axis)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    max_abs = weight_f.abs().amax(dim=reduce_dims) if reduce_dims else weight_f.abs()
    scale = (max_abs / float(qmax)).clamp(min=1e-12).to(torch.float32)
    scale_view = _scale_view(scale, weight_f.shape, channel_axis)
    q = torch.round(weight_f / scale_view).clamp(qmin, qmax)
    dequantized = (q * scale_view).to(original_dtype)
    return dequantized, scale


def weight_symmetric_per_channel_scale(weight: torch.Tensor, bits: int = 4, channel_axis: int = 0) -> torch.Tensor:
    bits = _normalize_bits(bits)
    if weight.dim() == 0:
        raise ValueError("Per-channel weight quantization requires a tensor with at least one dimension.")
    channel_axis = int(channel_axis)
    if channel_axis < 0:
        channel_axis += weight.dim()
    if channel_axis < 0 or channel_axis >= weight.dim():
        raise ValueError(f"Invalid channel_axis={channel_axis} for weight shape={tuple(weight.shape)}.")
    weight_f = weight.detach().to(torch.float32)
    reduce_dims = tuple(dim for dim in range(weight_f.dim()) if dim != channel_axis)
    qmax = (1 << (bits - 1)) - 1
    max_abs = weight_f.abs().amax(dim=reduce_dims) if reduce_dims else weight_f.abs()
    return (max_abs / float(qmax)).clamp(min=1e-12).to(torch.float32)


def _effective_weight_bytes(weight: torch.Tensor, bits: int, scale: torch.Tensor) -> int:
    packed_weight_bytes = (int(weight.numel()) * int(bits) + 7) // 8
    scale_bytes = int(scale.numel()) * torch.empty((), dtype=torch.float32).element_size()
    return packed_weight_bytes + scale_bytes


def _resolve_submodule(root: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    module = root
    for part in dotted_path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def _module_weight(module: nn.Module) -> Optional[torch.nn.Parameter]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.nn.Parameter):
        return weight
    base_layer = getattr(module, "base_layer", None)
    weight = getattr(base_layer, "weight", None)
    if isinstance(weight, torch.nn.Parameter):
        return weight
    return None


class SuffixTransformerWeightQuantizer:
    """
    Temporarily fake-quantizes selected suffix Transformer Linear weights.

    GOFA uses the same Mistral weights for encoder and decoder paths, so this
    controller only activates hooks while encoder suffix code runs. Decoder
    calls see the original full-precision weights.
    """

    def __init__(
            self,
            base_model: nn.Module,
            bits: int = 4,
            fake_quant: bool = True,
            quantize_attention: bool = True,
            quantize_mlp: bool = True,
            quantize_layernorm: bool = False,
            log_quantized_modules: bool = True,
            logger=print):
        self.base_model = base_model
        self.bits = _normalize_bits(bits)
        self.fake_quant = bool(fake_quant)
        self.quantize_attention = bool(quantize_attention)
        self.quantize_mlp = bool(quantize_mlp)
        self.quantize_layernorm = bool(quantize_layernorm)
        self.log_quantized_modules = bool(log_quantized_modules)
        self.logger = logger
        self.active_depth = 0
        self.modules: List[WeightQuantizedModule] = []
        self.handles = []
        self.stats = {
            "quantized_module_count": 0,
            "quantized_weight_original_bytes": 0,
            "quantized_weight_effective_bytes": 0,
            "compression_ratio": 0.0,
        }
        if not self.fake_quant:
            raise NotImplementedError(
                "scheme_b_weight_quant.fake_quant=False is not implemented; first version supports fake/dequant only."
            )
        if self.quantize_layernorm:
            self.logger("GOFA suffix weight quantization: quantize_layernorm=True is ignored; layer norms are not quantized.")
        self._collect_modules()
        self._register_hooks()
        self._log_summary()

    def _suffix_layer_range(self) -> Tuple[int, int]:
        start = int(getattr(self.base_model, "gnn_start_layer"))
        end = int(self.base_model.config.num_hidden_layers)
        return start, end

    def _target_paths(self) -> List[str]:
        paths = []
        if self.quantize_attention:
            paths.extend(f"self_attn.{name}" for name in ATTENTION_PROJECTIONS)
        if self.quantize_mlp:
            paths.extend(f"mlp.{name}" for name in MLP_PROJECTIONS)
        return paths

    def _collect_modules(self):
        if not hasattr(self.base_model, "layers") or not hasattr(self.base_model, "gnn_start_layer"):
            raise ValueError("Suffix weight quantization requires a GOFA Mistral base model with layers and gnn_start_layer.")
        start, end = self._suffix_layer_range()
        target_paths = self._target_paths()
        for layer_idx in range(start, end):
            decoder_layer = self.base_model.layers[layer_idx]
            for path in target_paths:
                module = _resolve_submodule(decoder_layer, path)
                if module is None:
                    continue
                weight = _module_weight(module)
                if weight is None:
                    continue
                with torch.no_grad():
                    scale = weight_symmetric_per_channel_scale(weight.detach(), bits=self.bits, channel_axis=0)
                original_bytes = int(weight.numel()) * int(weight.element_size())
                effective_bytes = _effective_weight_bytes(weight, self.bits, scale)
                self.modules.append(
                    WeightQuantizedModule(
                        name=f"layers.{layer_idx}.{path}",
                        module=module,
                        weight=weight,
                        shape=tuple(weight.shape),
                        scale_shape=tuple(scale.shape),
                        original_bytes=original_bytes,
                        effective_bytes=effective_bytes,
                    )
                )
        original_total = sum(item.original_bytes for item in self.modules)
        effective_total = sum(item.effective_bytes for item in self.modules)
        self.stats = {
            "quantized_module_count": len(self.modules),
            "quantized_weight_original_bytes": original_total,
            "quantized_weight_effective_bytes": effective_total,
            "compression_ratio": (original_total / effective_total) if effective_total else 0.0,
        }

    def _register_hooks(self):
        for item in self.modules:
            self.handles.append(item.module.register_forward_pre_hook(self._pre_forward_hook))
            self.handles.append(item.module.register_forward_hook(self._post_forward_hook))

    def _pre_forward_hook(self, module: nn.Module, _inputs):
        if self.active_depth <= 0:
            return
        if getattr(module, "_gofa_weight_quant_active", False):
            return
        weight = _module_weight(module)
        if weight is None:
            return
        with torch.no_grad():
            module._gofa_weight_quant_backup = weight.detach().clone()
            dequantized, _scale = quantize_dequant_weight_symmetric_per_channel(weight.detach(), bits=self.bits, channel_axis=0)
            weight.copy_(dequantized.to(device=weight.device, dtype=weight.dtype))
            module._gofa_weight_quant_active = True

    def _post_forward_hook(self, module: nn.Module, _inputs, _output):
        self._restore_module(module)

    def _restore_module(self, module: nn.Module):
        backup = getattr(module, "_gofa_weight_quant_backup", None)
        if backup is None:
            return
        weight = _module_weight(module)
        if weight is not None:
            with torch.no_grad():
                weight.copy_(backup.to(device=weight.device, dtype=weight.dtype))
        module._gofa_weight_quant_backup = None
        module._gofa_weight_quant_active = False

    def restore_all(self):
        for item in self.modules:
            self._restore_module(item.module)

    @contextmanager
    def activate(self):
        self.active_depth += 1
        try:
            yield self
        finally:
            self.active_depth = max(self.active_depth - 1, 0)
            self.restore_all()

    def _log_summary(self):
        if self.log_quantized_modules:
            for item in self.modules:
                self.logger(
                    "GOFA suffix weight quantized module: "
                    f"name={item.name}, shape={item.shape}, bits={self.bits}, scale_shape={item.scale_shape}"
                )
        self.logger(
            "GOFA suffix weight quantization stats: "
            f"quantized_module_count={self.stats['quantized_module_count']}, "
            f"quantized_weight_original_bytes={self.stats['quantized_weight_original_bytes']}, "
            f"quantized_weight_effective_bytes={self.stats['quantized_weight_effective_bytes']}, "
            f"compression_ratio={self.stats['compression_ratio']:.3f}x"
        )


def maybe_create_suffix_transformer_weight_quantizer(
        base_model: nn.Module,
        cfg: Dict,
        logger=print) -> Optional[SuffixTransformerWeightQuantizer]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    target = str(cfg.get("target", "suffix_transformer"))
    if target != "suffix_transformer":
        raise ValueError("scheme_b_weight_quant.target currently supports only 'suffix_transformer'.")
    return SuffixTransformerWeightQuantizer(
        base_model,
        bits=int(cfg.get("bits", 4)),
        fake_quant=bool(cfg.get("fake_quant", True)),
        quantize_attention=bool(cfg.get("quantize_attention", True)),
        quantize_mlp=bool(cfg.get("quantize_mlp", True)),
        quantize_layernorm=bool(cfg.get("quantize_layernorm", False)),
        log_quantized_modules=bool(cfg.get("log_quantized_modules", True)),
        logger=logger,
    )


def weight_quant_context(quantizer: Optional[SuffixTransformerWeightQuantizer]):
    if quantizer is None:
        return nullcontext()
    return quantizer.activate()
