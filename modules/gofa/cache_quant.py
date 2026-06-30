from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


QUANT_CACHE_FORMAT = "memory_text_kv_quant_v1"
QUANT_BASE_FORMAT = "memory_text_kv_quant_base_v1"
QUANT_DELTA_FORMAT = "memory_text_kv_quant_delta_v1"


def _normalize_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in {2, 4, 8, 16}:
        raise ValueError(f"Unsupported cache quantization bits={bits}; expected one of 2, 4, 8, 16.")
    return bits


def _effective_bits(bits: Optional[int], fallback_bits: int) -> int:
    if bits is None:
        return _normalize_bits(fallback_bits)
    return _normalize_bits(bits)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: Optional[str], fallback: torch.dtype = torch.float16) -> torch.dtype:
    if name is None:
        return fallback
    if isinstance(name, torch.dtype):
        return name
    name = str(name).replace("torch.", "")
    return getattr(torch, name, fallback)


def _canonical_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"channel_axis={axis} is invalid for ndim={ndim}.")
    return axis


def _scale_view(scale: torch.Tensor, shape: Iterable[int], axis: int) -> torch.Tensor:
    shape = list(shape)
    axis = _canonical_axis(axis, len(shape))
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    return scale.reshape(view_shape)


def _pack_low_bit(q: torch.Tensor, bits: int) -> torch.Tensor:
    bits = int(bits)
    values_per_byte = 8 // bits
    offset = 1 << (bits - 1)
    mask = (1 << bits) - 1
    q_u8 = (q.to(torch.int16) + offset).clamp_(0, mask).to(torch.uint8).flatten()
    padding = (-q_u8.numel()) % values_per_byte
    if padding:
        q_u8 = torch.cat([q_u8, torch.zeros(padding, dtype=torch.uint8)])
    q_u8 = q_u8.reshape(-1, values_per_byte)
    packed = torch.zeros(q_u8.shape[0], dtype=torch.uint8)
    for value_idx in range(values_per_byte):
        packed |= q_u8[:, value_idx] << (bits * value_idx)
    return packed


def _unpack_low_bit(packed: torch.Tensor, bits: int, numel: int, shape: Iterable[int]) -> torch.Tensor:
    bits = int(bits)
    values_per_byte = 8 // bits
    offset = 1 << (bits - 1)
    mask = (1 << bits) - 1
    packed = packed.to(torch.uint8).flatten()
    values = torch.empty(packed.numel() * values_per_byte, dtype=torch.uint8)
    for value_idx in range(values_per_byte):
        values[value_idx::values_per_byte] = (packed >> (bits * value_idx)) & mask
    values = values[:numel]
    return (values.to(torch.int16) - offset).to(torch.int8).reshape(tuple(shape))


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    channel_axis: int = -1,
    fake_quant: bool = True,
) -> Dict:
    bits = _normalize_bits(bits)
    tensor = tensor.detach().cpu()
    original_dtype = _dtype_name(tensor.dtype)
    shape = list(tensor.shape)

    if bits == 16 or not fake_quant:
        return {
            "encoding": "identity",
            "bits": 16,
            "shape": shape,
            "dtype": original_dtype,
            "tensor": tensor.contiguous(),
        }

    if tensor.numel() == 0:
        axis = _canonical_axis(channel_axis, tensor.dim()) if tensor.dim() else 0
        scale_shape = (shape[axis],) if tensor.dim() else ()
        return {
            "encoding": "symmetric_per_channel",
            "bits": bits,
            "shape": shape,
            "dtype": original_dtype,
            "channel_axis": axis,
            "scale": torch.ones(scale_shape, dtype=torch.float32),
            "q": torch.empty(shape, dtype=torch.int8),
            "packed": False,
            "pack_bits": None,
            "numel": 0,
        }

    axis = _canonical_axis(channel_axis, tensor.dim())
    reduce_dims = tuple(dim for dim in range(tensor.dim()) if dim != axis)
    tensor_f = tensor.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if reduce_dims:
        max_abs = tensor_f.abs().amax(dim=reduce_dims)
    else:
        max_abs = tensor_f.abs()
    scale = (max_abs / float(qmax)).clamp(min=1e-12).to(torch.float32).contiguous()
    q = torch.round(tensor_f / _scale_view(scale, shape, axis)).clamp(qmin, qmax).to(torch.int8).contiguous()

    payload = {
        "encoding": "symmetric_per_channel",
        "bits": bits,
        "shape": shape,
        "dtype": original_dtype,
        "channel_axis": axis,
        "scale": scale,
        "packed": bits in {2, 4},
        "pack_bits": bits if bits in {2, 4} else None,
        "numel": tensor.numel(),
    }
    if bits in {2, 4}:
        payload["q_packed"] = _pack_low_bit(q, bits)
    else:
        payload["q"] = q
    return payload


def dequantize_tensor(payload: Dict, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    encoding = payload.get("encoding")
    target_dtype = dtype or _dtype_from_name(payload.get("dtype"))
    if encoding == "identity":
        return payload["tensor"].to(dtype=target_dtype).contiguous()
    if encoding != "symmetric_per_channel":
        raise ValueError(f"Unsupported tensor cache encoding: {encoding}")

    shape = list(payload["shape"])
    if payload.get("packed"):
        q = _unpack_low_bit(
            payload["q_packed"],
            int(payload.get("pack_bits") or payload.get("bits")),
            int(payload["numel"]),
            shape,
        )
    else:
        q = payload["q"].reshape(tuple(shape)).to(torch.int8)
    scale = _scale_view(payload["scale"].to(torch.float32), shape, int(payload.get("channel_axis", -1)))
    tensor = q.to(torch.float32) * scale
    return tensor.to(dtype=target_dtype).contiguous()


def quantized_tensor_int(payload: Dict, device: Optional[torch.device] = None) -> torch.Tensor:
    encoding = payload.get("encoding")
    if encoding != "symmetric_per_channel":
        raise ValueError(f"Expected symmetric_per_channel quant payload, got encoding={encoding}.")
    shape = list(payload["shape"])
    if payload.get("packed"):
        q = _unpack_low_bit(
            payload["q_packed"],
            int(payload.get("pack_bits") or payload.get("bits")),
            int(payload["numel"]),
            shape,
        )
    else:
        q = payload["q"].reshape(tuple(shape)).to(torch.int8)
    if device is not None:
        q = q.to(device=device, dtype=torch.int8, non_blocking=True)
    return q.contiguous()


def quantized_tensor_scale(payload: Dict, device: Optional[torch.device] = None) -> torch.Tensor:
    encoding = payload.get("encoding")
    if encoding != "symmetric_per_channel":
        raise ValueError(f"Expected symmetric_per_channel quant payload, got encoding={encoding}.")
    scale = payload["scale"].to(torch.float32)
    if device is not None:
        scale = scale.to(device=device, dtype=torch.float32, non_blocking=True)
    return scale.contiguous()


def quantize_base_delta_tensor(
    tensor: torch.Tensor,
    base_bits: int = 8,
    delta_bits: int = 4,
    channel_axis: int = -1,
    fake_quant: bool = True,
) -> Tuple[Dict, Optional[Dict]]:
    base = quantize_tensor(tensor, bits=base_bits, channel_axis=channel_axis, fake_quant=fake_quant)
    if int(base.get("bits", base_bits)) == 16 or not fake_quant:
        return base, None
    base_reconstructed = dequantize_tensor(base, dtype=torch.float32)
    residual = tensor.detach().cpu().to(torch.float32) - base_reconstructed
    delta = quantize_tensor(residual, bits=delta_bits, channel_axis=channel_axis, fake_quant=fake_quant)
    delta["dtype"] = "float32"
    return base, delta


def reconstruct_base_delta_tensor(
    base_payload: Dict,
    delta_payload: Optional[Dict] = None,
    load_delta: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    original_dtype = dtype or _dtype_from_name(base_payload.get("dtype"))
    base = dequantize_tensor(base_payload, dtype=torch.float32)
    if load_delta and delta_payload is not None:
        base = base + dequantize_tensor(delta_payload, dtype=torch.float32)
    return base.to(dtype=original_dtype).contiguous()


def _quantize_kv_list(
    text_kv,
    key_base_bits: int,
    value_base_bits: int,
    key_delta_bits: int,
    value_delta_bits: int,
    fake_quant: bool,
):
    base_text_kv = []
    delta_text_kv = []
    for layer_kv in text_kv:
        base_key, delta_key = quantize_base_delta_tensor(
            layer_kv["key"],
            base_bits=key_base_bits,
            delta_bits=key_delta_bits,
            channel_axis=-1,
            fake_quant=fake_quant,
        )
        base_value, delta_value = quantize_base_delta_tensor(
            layer_kv["value"],
            base_bits=value_base_bits,
            delta_bits=value_delta_bits,
            channel_axis=-1,
            fake_quant=fake_quant,
        )
        base_text_kv.append({"key": base_key, "value": base_value})
        delta_text_kv.append({"key": delta_key, "value": delta_value})
    return base_text_kv, delta_text_kv


def quantize_scheme_b_cache_item(
    cache_item: Dict,
    base_bits: int = 8,
    delta_bits: int = 4,
    memory_base_bits: Optional[int] = None,
    key_base_bits: Optional[int] = None,
    value_base_bits: Optional[int] = None,
    memory_delta_bits: Optional[int] = None,
    key_delta_bits: Optional[int] = None,
    value_delta_bits: Optional[int] = None,
    static_tier: str = "low",
    fake_quant: bool = True,
) -> Tuple[Dict, Dict]:
    base_bits = _normalize_bits(base_bits)
    delta_bits = _normalize_bits(delta_bits)
    memory_base_bits = _effective_bits(memory_base_bits, base_bits)
    key_base_bits = _effective_bits(key_base_bits, base_bits)
    value_base_bits = _effective_bits(value_base_bits, base_bits)
    memory_delta_bits = _effective_bits(memory_delta_bits, delta_bits)
    key_delta_bits = _effective_bits(key_delta_bits, delta_bits)
    value_delta_bits = _effective_bits(value_delta_bits, delta_bits)
    memory_base, memory_delta = quantize_base_delta_tensor(
        cache_item["memory_state"],
        base_bits=memory_base_bits,
        delta_bits=memory_delta_bits,
        channel_axis=-1,
        fake_quant=fake_quant,
    )
    text_kv_base, text_kv_delta = _quantize_kv_list(
        cache_item["text_kv"],
        key_base_bits=key_base_bits,
        value_base_bits=value_base_bits,
        key_delta_bits=key_delta_bits,
        value_delta_bits=value_delta_bits,
        fake_quant=fake_quant,
    )
    base_payload = {
        "cache_format": QUANT_BASE_FORMAT,
        "source_cache_format": cache_item.get("cache_format", "memory_text_kv_v1"),
        "text_len": cache_item["text_len"],
        "mem_size": cache_item["mem_size"],
        "dtype": cache_item.get("dtype"),
        "base_bits": int(base_bits),
        "delta_bits": int(delta_bits),
        "fake_quant": bool(fake_quant),
        "static_tier": static_tier,
        "memory_base_bits": int(memory_base_bits),
        "key_base_bits": int(key_base_bits),
        "value_base_bits": int(value_base_bits),
        "memory_delta_bits": int(memory_delta_bits),
        "key_delta_bits": int(key_delta_bits),
        "value_delta_bits": int(value_delta_bits),
        "memory_state": memory_base,
        "text_kv": text_kv_base,
        "has_delta": memory_delta is not None,
    }
    delta_payload = {
        "cache_format": QUANT_DELTA_FORMAT,
        "text_len": cache_item["text_len"],
        "mem_size": cache_item["mem_size"],
        "delta_bits": int(delta_bits),
        "memory_delta_bits": int(memory_delta_bits),
        "key_delta_bits": int(key_delta_bits),
        "value_delta_bits": int(value_delta_bits),
        "memory_state": memory_delta,
        "text_kv": text_kv_delta,
    }
    return base_payload, delta_payload


def reconstruct_scheme_b_cache_item(
    base_payload: Dict,
    delta_payload: Optional[Dict] = None,
    load_delta: bool = True,
    load_memory_delta: bool = True,
    load_key_delta: bool = True,
    load_value_delta: bool = True,
    preserve_quantized_text_kv: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Dict:
    should_load_delta = load_delta and delta_payload is not None and bool(base_payload.get("has_delta", True))
    should_load_memory_delta = should_load_delta and bool(load_memory_delta)
    should_load_key_delta = should_load_delta and bool(load_key_delta)
    should_load_value_delta = should_load_delta and bool(load_value_delta)
    memory_state = reconstruct_base_delta_tensor(
        base_payload["memory_state"],
        None if not should_load_memory_delta else delta_payload.get("memory_state"),
        load_delta=should_load_memory_delta,
        dtype=dtype,
    )
    text_kv = []
    for layer_idx, layer_base in enumerate(base_payload["text_kv"]):
        layer_delta = None
        if should_load_delta and delta_payload is not None:
            layer_delta = delta_payload["text_kv"][layer_idx]
        if preserve_quantized_text_kv:
            if should_load_key_delta or should_load_value_delta:
                raise NotImplementedError(
                    "preserve_quantized_text_kv=True currently supports base-only text-side K/V. "
                    "Disable load_key_delta/load_value_delta or target-aware text-KV delta loading."
                )
            text_kv.append(
                {
                    "key": layer_base["key"],
                    "value": layer_base["value"],
                    "quantized": True,
                }
            )
            continue
        text_kv.append(
            {
                "key": reconstruct_base_delta_tensor(
                    layer_base["key"],
                    None if layer_delta is None else layer_delta.get("key"),
                    load_delta=should_load_key_delta,
                    dtype=dtype,
                ),
                "value": reconstruct_base_delta_tensor(
                    layer_base["value"],
                    None if layer_delta is None else layer_delta.get("value"),
                    load_delta=should_load_value_delta,
                    dtype=dtype,
                ),
            }
        )
    return {
        "text_len": base_payload["text_len"],
        "memory_state": memory_state,
        "text_kv": text_kv,
        "static_tier": base_payload.get("static_tier", "low"),
    }


def estimate_quantized_tensor_bits(payload: Optional[Dict]) -> int:
    if payload is None:
        return 0
    if payload.get("encoding") == "identity":
        return int(payload["tensor"].numel() * payload["tensor"].element_size() * 8)
    bits = int(payload.get("bits", 0))
    value_bits = int(payload.get("numel", 0)) * bits
    scale = payload.get("scale")
    scale_bits = 0 if scale is None else int(scale.numel() * scale.element_size() * 8)
    return value_bits + scale_bits
