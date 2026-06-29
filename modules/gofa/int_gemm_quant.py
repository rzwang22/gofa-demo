from __future__ import annotations

import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from modules.gofa.weight_quant import (
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
    _module_weight,
    _resolve_submodule,
)


SUPPORTED_INT_GEMM_WEIGHT_BITS = {4}
SUPPORTED_INT_GEMM_ACTIVATION_BITS = {8}
SUPPORTED_INT_GEMM_BACKENDS = {"torch_int_mm"}


@dataclass
class IntGemmQuantizedModule:
    layer_idx: int
    name: str
    module: nn.Module
    forward_module: nn.Module
    q_weight_cpu: torch.Tensor
    scale_w_cpu: torch.Tensor
    weight_shape: Tuple[int, ...]
    scale_shape: Tuple[int, ...]
    original_forward: Optional[object] = None
    device_payloads: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None


def _normalize_weight_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_INT_GEMM_WEIGHT_BITS:
        raise ValueError("scheme_b_int_gemm.weight_bits currently supports only 4.")
    return bits


def _normalize_activation_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_INT_GEMM_ACTIVATION_BITS:
        raise ValueError("scheme_b_int_gemm.activation_bits currently supports only 8.")
    return bits


def _linear_forward_module(module: nn.Module) -> nn.Module:
    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None and isinstance(_module_weight(base_layer), torch.nn.Parameter):
        return base_layer
    return module


def _linear_bias(module: nn.Module) -> Optional[torch.Tensor]:
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.nn.Parameter) or isinstance(bias, torch.Tensor):
        return bias
    return None


def quantize_weight_symmetric_int4_per_output(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"W4A8 int GEMM expects 2D Linear weight, got shape={tuple(weight.shape)}.")
    weight_f = weight.detach().to(torch.float32)
    qmax = 7
    scale_w = (weight_f.abs().amax(dim=1, keepdim=True) / float(qmax)).clamp(min=1e-12).to(torch.float32)
    q_weight = torch.round(weight_f / scale_w).clamp(-qmax, qmax).to(torch.int8).contiguous()
    return q_weight.cpu(), scale_w.cpu()


def _fake_quant_activation_symmetric_a8(activation: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(activation):
        return activation
    original_dtype = activation.dtype
    activation_f = activation.to(torch.float32)
    qmax = 127
    scale = (activation_f.abs().amax(dim=-1, keepdim=True) / float(qmax)).clamp(min=1e-12)
    q = torch.round(activation_f / scale).clamp(-qmax, qmax)
    return (q * scale).to(original_dtype)


def _fake_quant_dequant_weight_symmetric_int4(weight: torch.Tensor) -> torch.Tensor:
    original_dtype = weight.dtype
    q_weight, scale_w = quantize_weight_symmetric_int4_per_output(weight)
    return (q_weight.to(device=weight.device, dtype=torch.float32) * scale_w.to(device=weight.device)).to(original_dtype)


class SuffixTransformerIntGemmQuantizer:
    """
    Replaces selected suffix Transformer Linear forwards with an experimental
    W4A8 int8 x int8 -> int32 GEMM path while active.
    """

    def __init__(
            self,
            base_model: nn.Module,
            weight_bits: int = 4,
            activation_bits: int = 8,
            backend: str = "torch_int_mm",
            quantize_attention: bool = True,
            quantize_mlp: bool = True,
            quantize_layernorm: bool = False,
            fallback_to_fake_quant: bool = False,
            log_modules: bool = True,
            log_interval: int = 20,
            logger=print):
        self.base_model = base_model
        self.weight_bits = _normalize_weight_bits(weight_bits)
        self.activation_bits = _normalize_activation_bits(activation_bits)
        self.backend = str(backend or "torch_int_mm")
        if self.backend not in SUPPORTED_INT_GEMM_BACKENDS:
            raise ValueError("scheme_b_int_gemm.backend currently supports only 'torch_int_mm'.")
        self.quantize_attention = bool(quantize_attention)
        self.quantize_mlp = bool(quantize_mlp)
        self.quantize_layernorm = bool(quantize_layernorm)
        self.fallback_to_fake_quant = bool(fallback_to_fake_quant)
        self.log_modules = bool(log_modules)
        self.log_interval = max(int(log_interval), 1)
        self.logger = logger
        self.active_depth = 0
        self.modules: List[IntGemmQuantizedModule] = []
        self.module_by_forward_id: Dict[int, IntGemmQuantizedModule] = {}
        self.warned_fallback = False
        self.stats = {
            "int_gemm_quantized_module_count": 0,
            "int_gemm_call_count": 0,
            "int_gemm_numel_input": 0,
            "int_gemm_numel_output": 0,
            "int_gemm_int_mm_time_s": 0.0,
            "int_gemm_quant_time_s": 0.0,
            "int_gemm_dequant_time_s": 0.0,
            "int_gemm_total_time_s": 0.0,
            "int_gemm_backend": self.backend,
            "fallback_count": 0,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "quantized_weight_original_bytes": 0,
            "quantized_weight_effective_bytes": 0,
            "compression_ratio": 0.0,
        }
        if self.quantize_layernorm:
            self.logger("GOFA suffix int GEMM: quantize_layernorm=True is ignored; layer norms are not quantized.")
        self._collect_modules()
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
            raise ValueError("Suffix int GEMM quantization requires a GOFA Mistral base model with layers and gnn_start_layer.")
        start, end = self._suffix_layer_range()
        target_paths = self._target_paths()
        seen_forward_modules = set()
        original_bytes = 0
        effective_bytes = 0
        for layer_idx in range(start, end):
            decoder_layer = self.base_model.layers[layer_idx]
            for path in target_paths:
                module = _resolve_submodule(decoder_layer, path)
                if module is None:
                    continue
                forward_module = _linear_forward_module(module)
                if id(forward_module) in seen_forward_modules:
                    continue
                weight = _module_weight(forward_module)
                if weight is None:
                    continue
                q_weight, scale_w = quantize_weight_symmetric_int4_per_output(weight.detach())
                item = IntGemmQuantizedModule(
                    layer_idx=layer_idx,
                    name=f"layers.{layer_idx}.{path}",
                    module=module,
                    forward_module=forward_module,
                    q_weight_cpu=q_weight,
                    scale_w_cpu=scale_w,
                    weight_shape=tuple(weight.shape),
                    scale_shape=tuple(scale_w.shape),
                    device_payloads={},
                )
                seen_forward_modules.add(id(forward_module))
                self.modules.append(item)
                self.module_by_forward_id[id(forward_module)] = item
                original_bytes += int(weight.numel()) * int(weight.element_size())
                effective_bytes += (int(weight.numel()) * self.weight_bits + 7) // 8
                effective_bytes += int(scale_w.numel()) * torch.empty((), dtype=torch.float32).element_size()
        self.stats["int_gemm_quantized_module_count"] = len(self.modules)
        self.stats["quantized_weight_original_bytes"] = original_bytes
        self.stats["quantized_weight_effective_bytes"] = effective_bytes
        self.stats["compression_ratio"] = (original_bytes / effective_bytes) if effective_bytes else 0.0

    def _device_payload(self, item: IntGemmQuantizedModule, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        device_key = str(device)
        payloads = item.device_payloads if item.device_payloads is not None else {}
        if device_key not in payloads:
            payloads[device_key] = (
                item.q_weight_cpu.to(device=device, dtype=torch.int8, non_blocking=True).contiguous(),
                item.scale_w_cpu.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
            )
            item.device_payloads = payloads
        return payloads[device_key]

    def _fake_quant_forward(self, item: IntGemmQuantizedModule, module: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        self.stats["fallback_count"] += 1
        if not self.warned_fallback:
            self.logger(
                "GOFA suffix int GEMM warning: falling back to fake quant Linear; "
                "this path is for debugging only and is not an integer GEMM measurement."
            )
            self.warned_fallback = True
        weight = _module_weight(module)
        if weight is None:
            raise RuntimeError(f"GOFA suffix int GEMM fallback failed to find Linear weight for {item.name}.")
        activation_dq = _fake_quant_activation_symmetric_a8(input_tensor)
        weight_dq = _fake_quant_dequant_weight_symmetric_int4(weight)
        return F.linear(activation_dq, weight_dq, _linear_bias(module))

    def _int_gemm_forward(self, item: IntGemmQuantizedModule, module: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        total_start = time.perf_counter()
        if not isinstance(input_tensor, torch.Tensor):
            raise RuntimeError(f"GOFA suffix int GEMM expected Tensor input for {item.name}.")
        if not torch.is_floating_point(input_tensor):
            return self._call_original_forward(item, input_tensor)
        original_dtype = input_tensor.dtype
        original_shape = tuple(input_tensor.shape)
        in_features = int(item.weight_shape[1])
        out_features = int(item.weight_shape[0])
        if not original_shape or original_shape[-1] != in_features:
            raise RuntimeError(
                "GOFA suffix int GEMM input shape mismatch: "
                f"module={item.name}, input_shape={original_shape}, expected_in_features={in_features}"
            )
        if not hasattr(torch, "_int_mm"):
            if not self.fallback_to_fake_quant:
                raise RuntimeError("GOFA suffix int GEMM requires torch._int_mm, but it is unavailable.")
            return self._fake_quant_forward(item, module, input_tensor)
        if input_tensor.device.type != "cuda":
            if not self.fallback_to_fake_quant:
                raise RuntimeError(
                    "GOFA suffix int GEMM torch_int_mm backend requires CUDA input tensors. "
                    "Set scheme_b_int_gemm_fallback_to_fake_quant True only for debugging on non-CUDA devices."
                )
            return self._fake_quant_forward(item, module, input_tensor)

        quant_start = time.perf_counter()
        x_2d = input_tensor.reshape(-1, in_features)
        x_f = x_2d.to(torch.float32)
        qmax = 127
        scale_x = (x_f.abs().amax(dim=-1, keepdim=True) / float(qmax)).clamp(min=1e-12)
        q_x = torch.round(x_f / scale_x).clamp(-qmax, qmax).to(torch.int8).contiguous()
        q_w, scale_w = self._device_payload(item, input_tensor.device)
        q_w_t = q_w.t().contiguous()
        quant_elapsed = time.perf_counter() - quant_start

        int_mm_start = time.perf_counter()
        try:
            y_int = torch._int_mm(q_x, q_w_t)
        except Exception as exc:
            if not self.fallback_to_fake_quant:
                raise RuntimeError(
                    "GOFA suffix int GEMM torch._int_mm failed. "
                    "It usually requires contiguous CUDA int8 inputs; set "
                    "scheme_b_int_gemm_fallback_to_fake_quant True only for debugging."
                ) from exc
            return self._fake_quant_forward(item, module, input_tensor)
        int_mm_elapsed = time.perf_counter() - int_mm_start

        dequant_start = time.perf_counter()
        y = y_int.to(torch.float32) * scale_x.to(torch.float32) * scale_w.reshape(1, out_features).to(torch.float32)
        bias = _linear_bias(module)
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=torch.float32).reshape(1, out_features)
        output = y.reshape(*original_shape[:-1], out_features).to(dtype=original_dtype)
        dequant_elapsed = time.perf_counter() - dequant_start
        total_elapsed = time.perf_counter() - total_start

        self.stats["int_gemm_call_count"] += 1
        self.stats["int_gemm_numel_input"] += int(input_tensor.numel())
        self.stats["int_gemm_numel_output"] += int(output.numel())
        self.stats["int_gemm_quant_time_s"] += quant_elapsed
        self.stats["int_gemm_int_mm_time_s"] += int_mm_elapsed
        self.stats["int_gemm_dequant_time_s"] += dequant_elapsed
        self.stats["int_gemm_total_time_s"] += total_elapsed
        return output

    def _call_original_forward(self, item: IntGemmQuantizedModule, input_tensor: torch.Tensor):
        if item.original_forward is None:
            raise RuntimeError(f"GOFA suffix int GEMM original forward is not available for {item.name}.")
        return item.original_forward(input_tensor)

    def _patched_forward(self, module: nn.Module, input_tensor: Optional[torch.Tensor] = None, *args, **kwargs):
        item = self.module_by_forward_id.get(id(module))
        if item is None:
            raise RuntimeError("GOFA suffix int GEMM internal error: patched module is not registered.")
        if input_tensor is None and "input" in kwargs:
            input_tensor = kwargs.pop("input")
        if args or kwargs:
            if item.original_forward is None:
                raise RuntimeError(f"GOFA suffix int GEMM original forward is not available for {item.name}.")
            return item.original_forward(input_tensor, *args, **kwargs)
        if input_tensor is None:
            raise RuntimeError(f"GOFA suffix int GEMM expected Tensor input for {item.name}.")
        return self._int_gemm_forward(item, module, input_tensor)

    def _patch_modules(self):
        for item in self.modules:
            if item.original_forward is not None:
                continue
            item.original_forward = item.forward_module.forward
            def patched_forward(module_self, input_tensor=None, *args, _quantizer=self, **kwargs):
                return _quantizer._patched_forward(module_self, input_tensor, *args, **kwargs)
            item.forward_module.forward = types.MethodType(patched_forward, item.forward_module)

    def _restore_modules(self):
        for item in self.modules:
            if item.original_forward is None:
                continue
            item.forward_module.forward = item.original_forward
            item.original_forward = None

    @contextmanager
    def activate(self):
        self.active_depth += 1
        if self.active_depth == 1:
            self._patch_modules()
        try:
            yield self
        finally:
            self.active_depth = max(self.active_depth - 1, 0)
            if self.active_depth == 0:
                self._restore_modules()

    def _log_summary(self):
        start, end = self._suffix_layer_range()
        if self.log_modules:
            for item in self.modules:
                self.logger(
                    "GOFA suffix int GEMM module: "
                    f"name={item.name}, weight_shape={item.weight_shape}, "
                    f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
                    f"scale_w_shape={item.scale_shape}"
                )
        self.logger(
            "GOFA suffix int GEMM enabled: "
            f"backend={self.backend}, target_layers={list(range(start, end))}, "
            f"quantized_module_count={self.stats['int_gemm_quantized_module_count']}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
            f"quantize_attention={self.quantize_attention}, quantize_mlp={self.quantize_mlp}, "
            f"fallback_to_fake_quant={self.fallback_to_fake_quant}, "
            f"quantized_weight_original_bytes={self.stats['quantized_weight_original_bytes']}, "
            f"quantized_weight_effective_bytes={self.stats['quantized_weight_effective_bytes']}, "
            f"compression_ratio={self.stats['compression_ratio']:.3f}x"
        )


def maybe_create_suffix_transformer_int_gemm_quantizer(
        base_model: nn.Module,
        cfg: Dict,
        logger=print) -> Optional[SuffixTransformerIntGemmQuantizer]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    target = str(cfg.get("target", "suffix_transformer"))
    if target != "suffix_transformer":
        raise ValueError("scheme_b_int_gemm.target currently supports only 'suffix_transformer'.")
    return SuffixTransformerIntGemmQuantizer(
        base_model,
        weight_bits=int(cfg.get("weight_bits", 4)),
        activation_bits=int(cfg.get("activation_bits", 8)),
        backend=str(cfg.get("backend", "torch_int_mm")),
        quantize_attention=bool(cfg.get("quantize_attention", True)),
        quantize_mlp=bool(cfg.get("quantize_mlp", True)),
        quantize_layernorm=bool(cfg.get("quantize_layernorm", False)),
        fallback_to_fake_quant=bool(cfg.get("fallback_to_fake_quant", False)),
        log_modules=bool(cfg.get("log_modules", True)),
        log_interval=int(cfg.get("log_interval", 20)),
        logger=logger,
    )


def int_gemm_context(quantizer: Optional[SuffixTransformerIntGemmQuantizer]):
    if quantizer is None:
        return nullcontext()
    return quantizer.activate()
