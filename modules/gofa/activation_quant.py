from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from modules.gofa.weight_quant import (
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
    _module_weight,
    _resolve_submodule,
)


SUPPORTED_ACTIVATION_BITS = {8}


@dataclass
class ActivationQuantizedModule:
    layer_idx: int
    name: str
    module: nn.Module
    weight_shape: Optional[Tuple[int, ...]]


def _normalize_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_ACTIVATION_BITS:
        raise ValueError("scheme_b_activation_quant.bits currently supports only 8.")
    return bits


def fake_quant_activation_symmetric(
        activation: torch.Tensor,
        bits: int = 8,
        per_token: bool = True,
        eps: float = 1e-12) -> torch.Tensor:
    bits = _normalize_bits(bits)
    if not torch.is_floating_point(activation):
        return activation
    original_dtype = activation.dtype
    activation_f = activation.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    if per_token:
        max_abs = activation_f.abs().amax(dim=-1, keepdim=True)
    else:
        max_abs = activation_f.abs().amax().reshape([1] * activation_f.dim())
    scale = (max_abs / float(qmax)).clamp(min=eps)
    q = torch.round(activation_f / scale).clamp(qmin, qmax)
    return (q * scale).to(original_dtype)


class SuffixTransformerActivationQuantizer:
    """
    Temporarily fake-quantizes suffix Transformer Linear input activations.

    Hooks are always registered, but they only modify inputs while activate()
    is active. This keeps decoder calls at original precision.
    """

    def __init__(
            self,
            base_model: nn.Module,
            bits: int = 8,
            fake_quant: bool = True,
            quantize_attention: bool = True,
            quantize_mlp: bool = True,
            quantize_qkv_outputs: bool = False,
            quantize_attn_output: bool = False,
            quantize_mlp_output: bool = False,
            per_token: bool = True,
            log_quantized_modules: bool = True,
            logger=print):
        self.base_model = base_model
        self.bits = _normalize_bits(bits)
        self.fake_quant = bool(fake_quant)
        self.quantize_attention = bool(quantize_attention)
        self.quantize_mlp = bool(quantize_mlp)
        self.quantize_qkv_outputs = bool(quantize_qkv_outputs)
        self.quantize_attn_output = bool(quantize_attn_output)
        self.quantize_mlp_output = bool(quantize_mlp_output)
        self.per_token = bool(per_token)
        self.log_quantized_modules = bool(log_quantized_modules)
        self.logger = logger
        self.active_depth = 0
        self.modules: List[ActivationQuantizedModule] = []
        self.handles = []
        self.stats = {
            "activation_quantized_module_count": 0,
            "activation_quant_call_count": 0,
            "activation_quant_tensor_count": 0,
            "activation_quant_numel": 0,
            "activation_quant_time_s": 0.0,
            "activation_effective_bits": self.bits,
        }
        if not self.fake_quant:
            raise NotImplementedError(
                "scheme_b_activation_quant.fake_quant=False is not implemented; first version supports fake/dequant only."
            )
        if self.quantize_qkv_outputs or self.quantize_attn_output or self.quantize_mlp_output:
            self.logger(
                "GOFA suffix activation quantization: output activation quantization flags are ignored in v1; "
                "only Linear input activations are quantized."
            )
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
            raise ValueError("Suffix activation quantization requires a GOFA Mistral base model with layers and gnn_start_layer.")
        start, end = self._suffix_layer_range()
        target_paths = self._target_paths()
        for layer_idx in range(start, end):
            decoder_layer = self.base_model.layers[layer_idx]
            for path in target_paths:
                module = _resolve_submodule(decoder_layer, path)
                if module is None:
                    continue
                weight = _module_weight(module)
                self.modules.append(
                    ActivationQuantizedModule(
                        layer_idx=layer_idx,
                        name=f"layers.{layer_idx}.{path}",
                        module=module,
                        weight_shape=tuple(weight.shape) if weight is not None else None,
                    )
                )
        self.stats["activation_quantized_module_count"] = len(self.modules)

    def _register_hooks(self):
        for item in self.modules:
            self.handles.append(item.module.register_forward_pre_hook(self._pre_forward_hook))

    def _pre_forward_hook(self, _module: nn.Module, inputs):
        if self.active_depth <= 0:
            return
        if not inputs:
            return
        first_input = inputs[0]
        if not isinstance(first_input, torch.Tensor):
            return
        start_time = time.perf_counter()
        quantized = fake_quant_activation_symmetric(
            first_input,
            bits=self.bits,
            per_token=self.per_token,
        )
        elapsed = time.perf_counter() - start_time
        self.stats["activation_quant_call_count"] += 1
        self.stats["activation_quant_tensor_count"] += 1
        self.stats["activation_quant_numel"] += int(first_input.numel())
        self.stats["activation_quant_time_s"] += elapsed
        return (quantized,) + tuple(inputs[1:])

    @contextmanager
    def activate(self):
        self.active_depth += 1
        try:
            yield self
        finally:
            self.active_depth = max(self.active_depth - 1, 0)

    def _log_summary(self):
        if self.log_quantized_modules:
            for item in self.modules:
                self.logger(
                    "GOFA suffix activation quantized module: "
                    f"layer_index={item.layer_idx}, module={item.name}, "
                    f"weight_shape={item.weight_shape}, activation_bits={self.bits}, per_token={self.per_token}"
                )
        self.logger(
            "GOFA suffix activation quantization stats: "
            f"activation_quantized_module_count={self.stats['activation_quantized_module_count']}, "
            f"activation_effective_bits={self.stats['activation_effective_bits']}, "
            f"per_token={self.per_token}, target=suffix_transformer"
        )


def maybe_create_suffix_transformer_activation_quantizer(
        base_model: nn.Module,
        cfg: Dict,
        logger=print) -> Optional[SuffixTransformerActivationQuantizer]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    target = str(cfg.get("target", "suffix_transformer"))
    if target != "suffix_transformer":
        raise ValueError("scheme_b_activation_quant.target currently supports only 'suffix_transformer'.")
    return SuffixTransformerActivationQuantizer(
        base_model,
        bits=int(cfg.get("bits", 8)),
        fake_quant=bool(cfg.get("fake_quant", True)),
        quantize_attention=bool(cfg.get("quantize_attention", True)),
        quantize_mlp=bool(cfg.get("quantize_mlp", True)),
        quantize_qkv_outputs=bool(cfg.get("quantize_qkv_outputs", False)),
        quantize_attn_output=bool(cfg.get("quantize_attn_output", False)),
        quantize_mlp_output=bool(cfg.get("quantize_mlp_output", False)),
        per_token=bool(cfg.get("per_token", True)),
        log_quantized_modules=bool(cfg.get("log_quantized_modules", True)),
        logger=logger,
    )


def activation_quant_context(quantizer: Optional[SuffixTransformerActivationQuantizer]):
    if quantizer is None:
        return nullcontext()
    return quantizer.activate()
