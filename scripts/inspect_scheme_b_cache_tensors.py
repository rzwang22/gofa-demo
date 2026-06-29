#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gofa.cache_quant import QUANT_BASE_FORMAT, QUANT_DELTA_FORMAT, reconstruct_scheme_b_cache_item


METADATA_FIELDS = (
    "cache_type",
    "local_node_idx",
    "global_node_id",
    "is_target",
    "is_target_neighbor",
    "hop_distance_to_target",
    "local_degree",
    "global_degree",
)


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _csv_list(value: str, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _csv_int_list(value: str, default: Optional[List[int]] = None) -> List[int]:
    if value is None:
        return list(default or [])
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _manifest_item_relpath(item: Dict[str, Any]) -> Path:
    relpath = item.get("relpath") or item.get("source_cache_relpath")
    if not relpath:
        raise RuntimeError(f"Manifest cache item is missing relpath: cache_key={item.get('cache_key')}")
    relpath = Path(relpath)
    if relpath.is_absolute():
        raise RuntimeError(f"Manifest relpath must be relative, got: {relpath}")
    if ".." in relpath.parts:
        raise RuntimeError(f"Manifest relpath must not contain '..': {relpath}")
    return relpath


def _resolve_manifest_full_path(full_cache_dir: Path, relpath: Path) -> Path:
    candidate = full_cache_dir / relpath
    if candidate.exists():
        return candidate
    if len(relpath.parts) >= 3 and full_cache_dir.name == relpath.parts[0]:
        return full_cache_dir / Path(*relpath.parts[1:])
    return candidate


def _iter_full_cache_files(full_cache_dir: Path) -> Iterable[Path]:
    for path in sorted(full_cache_dir.rglob("*.pt")):
        if "delta" in path.relative_to(full_cache_dir).parts:
            continue
        yield path


def _cache_tag_and_relpath(full_cache_dir: Path, full_cache_file: Path) -> Tuple[str, Path]:
    relpath = full_cache_file.relative_to(full_cache_dir)
    if len(relpath.parts) >= 3:
        return relpath.parts[0], Path(*relpath.parts[1:])
    if len(relpath.parts) == 2:
        return full_cache_dir.name, relpath
    raise RuntimeError(f"Unexpected full cache relative path: {relpath}")


def _quant_paths(
        full_cache_dir: Path,
        quant_cache_dir: Path,
        full_cache_file: Path,
        manifest_relpath: Optional[Path]) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
    if quant_cache_dir is None:
        return None, None, None
    if manifest_relpath is not None:
        if len(manifest_relpath.parts) < 3:
            raise RuntimeError(f"Manifest relpath must include <cache_tag>/<prefix>/<cache_key>.pt: {manifest_relpath}")
        cache_tag = manifest_relpath.parts[0]
        return cache_tag, quant_cache_dir / manifest_relpath, quant_cache_dir / "delta" / manifest_relpath
    cache_tag, relpath = _cache_tag_and_relpath(full_cache_dir, full_cache_file)
    return cache_tag, quant_cache_dir / cache_tag / relpath, quant_cache_dir / "delta" / cache_tag / relpath


def _load_full_payload(path: Path) -> Optional[Dict]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("cache_format") != "memory_text_kv_v1":
        return None
    return payload


def _load_quant_payloads(base_path: Optional[Path], delta_path: Optional[Path]) -> Tuple[Optional[Dict], Optional[Dict]]:
    base_payload = None
    delta_payload = None
    if base_path is not None and base_path.exists():
        candidate = torch.load(base_path, map_location="cpu")
        if candidate.get("cache_format") == QUANT_BASE_FORMAT:
            base_payload = candidate
    if delta_path is not None and delta_path.exists():
        candidate = torch.load(delta_path, map_location="cpu")
        if candidate.get("cache_format") == QUANT_DELTA_FORMAT:
            delta_payload = candidate
    return base_payload, delta_payload


def _load_manifest_entries(
        manifest_path: Path,
        full_cache_dir: Path,
        sample_items: int,
        sample_policy: str,
        cache_type: str,
        seed: int) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = manifest.get("cache_items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Manifest cache_items must be a list: {manifest_path}")
    if cache_type != "all":
        items = [item for item in items if item.get("cache_type") == cache_type]
    seen = set()
    deduped = []
    for item in items:
        relpath = _manifest_item_relpath(item)
        cache_key = item.get("cache_key")
        dedupe_key = cache_key or str(relpath)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    if sample_policy == "random":
        rng = random.Random(seed)
        rng.shuffle(deduped)
    elif sample_policy == "target_related_first":
        deduped.sort(
            key=lambda item: (
                not bool(item.get("is_target")),
                not bool(item.get("is_target_neighbor")),
                item.get("hop_distance_to_target") if item.get("hop_distance_to_target") is not None else 1_000_000,
            )
        )
    elif sample_policy not in {"manifest_head", "by_cache_type"}:
        raise ValueError("sample-policy must be manifest_head, random, target_related_first, or by_cache_type.")
    selected = deduped[:sample_items]
    entries = []
    for item in selected:
        relpath = _manifest_item_relpath(item)
        entries.append(
            {
                "full_path": _resolve_manifest_full_path(full_cache_dir, relpath),
                "relpath": relpath,
                "cache_key": item.get("cache_key"),
                "manifest_item": item,
            }
        )
    return entries


def _scan_full_entries(full_cache_dir: Path, sample_items: int, sample_policy: str, seed: int) -> List[Dict[str, Any]]:
    paths = list(_iter_full_cache_files(full_cache_dir))
    if sample_policy == "random":
        rng = random.Random(seed)
        rng.shuffle(paths)
    else:
        paths = sorted(paths)
    entries = []
    for path in paths[:sample_items]:
        cache_tag, relpath_without_tag = _cache_tag_and_relpath(full_cache_dir, path)
        entries.append(
            {
                "full_path": path,
                "relpath": Path(cache_tag) / relpath_without_tag,
                "cache_key": path.stem,
                "manifest_item": {},
            }
        )
    return entries


def _sample_indices(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if length <= limit:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=limit, device=device).round().long()


def _sample_2d(tensor: torch.Tensor, sample_tokens: int, sample_channels: int) -> torch.Tensor:
    if tensor.dim() != 2:
        tensor = tensor.reshape(-1, tensor.size(-1))
    if tensor.size(0) > sample_tokens:
        tensor = tensor.index_select(0, _sample_indices(tensor.size(0), sample_tokens, tensor.device))
    if tensor.size(1) > sample_channels:
        tensor = tensor.index_select(1, _sample_indices(tensor.size(1), sample_channels, tensor.device))
    return tensor.detach().cpu().contiguous()


def _standardize_memory_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.reshape(tensor.size(0), 1)
    return tensor.reshape(-1, tensor.size(-1))


def _standardize_kv_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.reshape(tensor.size(0), 1)
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3:
        a, b, c = tensor.shape
        if a <= 64 and (b > a or b == 0):
            return tensor.permute(1, 0, 2).reshape(b, a * c)
        return tensor.reshape(a, b * c)
    if tensor.dim() == 4:
        # Common cache layout: [batch, heads, seq, head_dim].
        bsz, heads, seq_len, head_dim = tensor.shape
        if heads <= 64:
            return tensor.permute(0, 2, 1, 3).reshape(bsz * seq_len, heads * head_dim)
    return tensor.reshape(-1, tensor.size(-1))


def _standardize_tensor(tensor: torch.Tensor, tensor_kind: str) -> torch.Tensor:
    if tensor_kind == "memory_state":
        return _standardize_memory_tensor(tensor)
    return _standardize_kv_tensor(tensor)


def _json_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return numerator / denominator.clamp(min=eps)


def _tensor_stats(tensor_2d: torch.Tensor) -> Dict[str, Any]:
    tensor = tensor_2d.to(torch.float32)
    abs_tensor = tensor.abs()
    flat_abs = abs_tensor.flatten()
    if flat_abs.numel() == 0:
        return {
            "abs_max": 0.0,
            "abs_mean": 0.0,
            "abs_std": 0.0,
            "abs_p50": 0.0,
            "abs_p90": 0.0,
            "abs_p99": 0.0,
            "abs_p999": 0.0,
            "max_over_p99": 0.0,
            "max_over_p999": 0.0,
            "token_abs_max_mean": 0.0,
            "token_abs_max_max": 0.0,
            "token_abs_p99_mean": 0.0,
            "token_max_over_p99_mean": 0.0,
            "channel_abs_max_mean": 0.0,
            "channel_abs_max_max": 0.0,
            "channel_abs_p99_mean": 0.0,
            "channel_max_over_p99_mean": 0.0,
        }
    abs_p99 = torch.quantile(flat_abs, 0.99)
    abs_p999 = torch.quantile(flat_abs, 0.999)
    token_abs_max = abs_tensor.amax(dim=1)
    token_abs_p99 = torch.quantile(abs_tensor, 0.99, dim=1)
    channel_abs_max = abs_tensor.amax(dim=0)
    channel_abs_p99 = torch.quantile(abs_tensor, 0.99, dim=0)
    return {
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
        "token_abs_p99_mean": _json_float(token_abs_p99.mean()),
        "token_max_over_p99_mean": _json_float(_safe_ratio(token_abs_max, token_abs_p99).mean()),
        "channel_abs_max_mean": _json_float(channel_abs_max.mean()),
        "channel_abs_max_max": _json_float(channel_abs_max.max()),
        "channel_abs_p99_mean": _json_float(channel_abs_p99.mean()),
        "channel_max_over_p99_mean": _json_float(_safe_ratio(channel_abs_max, channel_abs_p99).mean()),
    }


def _error_stats(full: torch.Tensor, reconstructed: Optional[torch.Tensor], suffix: str) -> Dict[str, Optional[float]]:
    if reconstructed is None:
        return {
            f"rel_l2_error_{suffix}": None,
            f"cosine_similarity_{suffix}": None,
            f"mse_{suffix}": None,
        }
    full_f = full.to(torch.float32).flatten()
    rec_f = reconstructed.to(torch.float32).flatten()
    if full_f.numel() != rec_f.numel():
        min_len = min(full_f.numel(), rec_f.numel())
        full_f = full_f[:min_len]
        rec_f = rec_f[:min_len]
    diff = rec_f - full_f
    denom = full_f.norm().clamp(min=1e-12)
    cosine_denom = (full_f.norm() * rec_f.norm()).clamp(min=1e-12)
    return {
        f"rel_l2_error_{suffix}": _json_float(diff.norm() / denom),
        f"cosine_similarity_{suffix}": _json_float(torch.dot(full_f, rec_f) / cosine_denom),
        f"mse_{suffix}": _json_float((diff * diff).mean()),
    }


def _metadata_from_manifest(item: Dict[str, Any]) -> Dict[str, Any]:
    return {field: item.get(field) for field in METADATA_FIELDS}


def _safe_cache_key(cache_key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(cache_key))[:32]


def _save_tensor_payload(
        tensor_dir: Path,
        item_idx: int,
        cache_key: str,
        cache_tag: Optional[str],
        cache_type: Optional[str],
        layer_idx: Optional[int],
        tensor_kind: str,
        reconstruct_mode: str,
        original_shape: Tuple[int, ...],
        tensor_2d: torch.Tensor,
        metadata: Dict[str, Any]):
    layer_part = "" if layer_idx is None else f"_layer{layer_idx}"
    filename = (
        f"item{item_idx}_cachekey_{_safe_cache_key(cache_key)}"
        f"{layer_part}_{tensor_kind}_{reconstruct_mode}.pt"
    )
    payload = {
        "cache_key": cache_key,
        "item_idx": item_idx,
        "cache_tag": cache_tag,
        "cache_type": cache_type,
        "layer_idx": layer_idx,
        "tensor_kind": tensor_kind,
        "reconstruct_mode": reconstruct_mode,
        "original_shape": tuple(original_shape),
        "standardized_shape": tuple(tensor_2d.shape),
        "tensor": tensor_2d,
        "dtype": str(tensor_2d.dtype).replace("torch.", ""),
        "metadata": metadata,
    }
    tensor_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tensor_dir / filename)


def _layer_offset(layer_idx: int, text_kv_len: int, suffix_layer_start: int) -> Optional[int]:
    if 0 <= layer_idx < text_kv_len:
        return layer_idx
    offset = int(layer_idx) - int(suffix_layer_start)
    if 0 <= offset < text_kv_len:
        return offset
    return None


def _collect_tensor_versions(
        full_payload: Dict,
        base_item: Optional[Dict],
        base_delta_item: Optional[Dict],
        include_memory_state: bool,
        include_text_kv: bool,
        layers: List[int],
        kv_types: List[str],
        reconstruct_modes: List[str],
        suffix_layer_start: int) -> List[Dict[str, Any]]:
    tensors = []
    version_items = {"full": full_payload, "base": base_item, "base_delta": base_delta_item}
    if include_memory_state:
        for mode in reconstruct_modes:
            item = version_items.get(mode)
            if item is not None and item.get("memory_state") is not None:
                tensors.append(
                    {
                        "tensor_kind": "memory_state",
                        "layer_idx": None,
                        "mode": mode,
                        "tensor": item["memory_state"],
                    }
                )
    if include_text_kv:
        text_kv_len = len(full_payload.get("text_kv", []))
        for layer_idx in layers:
            offset = _layer_offset(layer_idx, text_kv_len, suffix_layer_start)
            if offset is None:
                continue
            for kv_type in kv_types:
                for mode in reconstruct_modes:
                    item = version_items.get(mode)
                    text_kv = item.get("text_kv", []) if item is not None else []
                    if offset >= len(text_kv) or kv_type not in text_kv[offset]:
                        continue
                    tensors.append(
                        {
                            "tensor_kind": f"text_kv_{kv_type}",
                            "layer_idx": layer_idx,
                            "mode": mode,
                            "tensor": text_kv[offset][kv_type],
                        }
                    )
    return tensors


def _append_jsonl(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect and sample GOFA Scheme-B cache tensors.")
    parser.add_argument("--full-cache-dir", required=True)
    parser.add_argument("--quant-cache-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-items", type=int, default=30)
    parser.add_argument("--sample-policy", default="manifest_head", choices=("manifest_head", "random", "target_related_first", "by_cache_type"))
    parser.add_argument("--cache-type", default="node", choices=("node", "edge", "all"))
    parser.add_argument("--include-memory-state", type=_str_to_bool, default=True)
    parser.add_argument("--include-text-kv", type=_str_to_bool, default=True)
    parser.add_argument("--layers", default="26,27,28,29,30,31")
    parser.add_argument("--kv-types", default="key,value")
    parser.add_argument("--reconstruct-mode", default="full,base,base_delta")
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--sample-channels", type=int, default=256)
    parser.add_argument("--compute-quant-error", type=_str_to_bool, default=True)
    parser.add_argument("--suffix-layer-start", type=int, default=26)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    full_cache_dir = Path(args.full_cache_dir).resolve()
    quant_cache_dir = Path(args.quant_cache_dir).resolve() if args.quant_cache_dir else None
    output_dir = Path(args.output_dir).resolve()
    tensor_dir = output_dir / "tensors"
    stats_dir = output_dir / "stats"
    stats_path = stats_dir / "cache_tensor_stats.jsonl"
    stats_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    if stats_path.exists():
        stats_path.unlink()

    layers = _csv_int_list(args.layers, [26, 27, 28, 29, 30, 31])
    kv_types = _csv_list(args.kv_types, ["key", "value"])
    reconstruct_modes = _csv_list(args.reconstruct_mode, ["full", "base", "base_delta"])
    invalid_modes = [mode for mode in reconstruct_modes if mode not in {"full", "base", "base_delta"}]
    if invalid_modes:
        raise ValueError(f"Unsupported reconstruct modes: {invalid_modes}")
    if not full_cache_dir.is_dir():
        raise FileNotFoundError(full_cache_dir)
    if quant_cache_dir is not None and not quant_cache_dir.is_dir():
        raise FileNotFoundError(quant_cache_dir)

    if args.manifest:
        entries = _load_manifest_entries(
            Path(args.manifest).resolve(),
            full_cache_dir,
            sample_items=int(args.sample_items),
            sample_policy=args.sample_policy,
            cache_type=args.cache_type,
            seed=args.seed,
        )
    else:
        entries = _scan_full_entries(
            full_cache_dir,
            sample_items=int(args.sample_items),
            sample_policy=args.sample_policy,
            seed=args.seed,
        )

    processed_items = 0
    saved_tensors = 0
    missing_full = 0
    missing_quant_base = 0
    missing_quant_delta = 0
    for item_idx, entry in enumerate(entries):
        full_path = entry["full_path"]
        manifest_item = entry.get("manifest_item") or {}
        if not full_path.exists():
            missing_full += 1
            print(f"missing full cache item skipped: {full_path}")
            continue
        full_payload = _load_full_payload(full_path)
        if full_payload is None:
            continue
        cache_key = full_payload.get("cache_key") or entry.get("cache_key") or full_path.stem
        if entry.get("cache_key") and full_payload.get("cache_key") != entry.get("cache_key"):
            raise RuntimeError(
                "Manifest cache_key does not match full cache payload: "
                f"manifest={entry.get('cache_key')}, payload={full_payload.get('cache_key')}, path={full_path}"
            )
        cache_tag, quant_base_path, quant_delta_path = _quant_paths(
            full_cache_dir,
            quant_cache_dir,
            full_path,
            entry.get("relpath"),
        )
        base_payload, delta_payload = _load_quant_payloads(quant_base_path, quant_delta_path)
        if quant_cache_dir is not None and base_payload is None:
            missing_quant_base += 1
        if quant_cache_dir is not None and delta_payload is None:
            missing_quant_delta += 1
        base_item = None
        base_delta_item = None
        if base_payload is not None:
            base_item = reconstruct_scheme_b_cache_item(base_payload, delta_payload=None, load_delta=False)
            if delta_payload is not None:
                base_delta_item = reconstruct_scheme_b_cache_item(base_payload, delta_payload=delta_payload, load_delta=True)
        metadata = _metadata_from_manifest(manifest_item)
        metadata.update(
            {
                "full_cache_path": str(full_path),
                "quant_base_path": str(quant_base_path) if quant_base_path is not None else None,
                "quant_delta_path": str(quant_delta_path) if quant_delta_path is not None else None,
                "quant_base_exists": quant_base_path.exists() if quant_base_path is not None else False,
                "quant_delta_exists": quant_delta_path.exists() if quant_delta_path is not None else False,
            }
        )
        tensors = _collect_tensor_versions(
            full_payload,
            base_item,
            base_delta_item,
            include_memory_state=args.include_memory_state,
            include_text_kv=args.include_text_kv,
            layers=layers,
            kv_types=kv_types,
            reconstruct_modes=reconstruct_modes,
            suffix_layer_start=args.suffix_layer_start,
        )
        grouped: Dict[Tuple[Optional[int], str], Dict[str, torch.Tensor]] = {}
        original_shapes: Dict[Tuple[Optional[int], str, str], Tuple[int, ...]] = {}
        for tensor_entry in tensors:
            tensor_kind = tensor_entry["tensor_kind"]
            layer_idx = tensor_entry["layer_idx"]
            mode = tensor_entry["mode"]
            original_shape = tuple(tensor_entry["tensor"].shape)
            tensor_2d = _sample_2d(
                _standardize_tensor(tensor_entry["tensor"], tensor_kind),
                sample_tokens=args.sample_tokens,
                sample_channels=args.sample_channels,
            )
            grouped.setdefault((layer_idx, tensor_kind), {})[mode] = tensor_2d
            original_shapes[(layer_idx, tensor_kind, mode)] = original_shape

        for (layer_idx, tensor_kind), versions in grouped.items():
            full_tensor = versions.get("full")
            error_metrics = {}
            if args.compute_quant_error and full_tensor is not None:
                error_metrics.update(_error_stats(full_tensor, versions.get("base"), "base"))
                error_metrics.update(_error_stats(full_tensor, versions.get("base_delta"), "base_delta"))
            for mode, tensor_2d in versions.items():
                original_shape = original_shapes[(layer_idx, tensor_kind, mode)]
                _save_tensor_payload(
                    tensor_dir,
                    item_idx=item_idx,
                    cache_key=cache_key,
                    cache_tag=cache_tag,
                    cache_type=manifest_item.get("cache_type"),
                    layer_idx=layer_idx,
                    tensor_kind=tensor_kind,
                    reconstruct_mode=mode,
                    original_shape=original_shape,
                    tensor_2d=tensor_2d,
                    metadata=metadata,
                )
                record = {
                    "cache_key": cache_key,
                    "cache_tag": cache_tag,
                    "cache_type": manifest_item.get("cache_type"),
                    "layer_idx": layer_idx,
                    "tensor_kind": tensor_kind,
                    "reconstruct_mode": mode,
                    "shape": [int(tensor_2d.size(0)), int(tensor_2d.size(1))],
                    "original_shape": list(original_shape),
                    **metadata,
                    **_tensor_stats(tensor_2d),
                    **error_metrics,
                }
                _append_jsonl(stats_path, record)
                saved_tensors += 1
        processed_items += 1
        if processed_items <= 3 or processed_items % 10 == 0:
            print(
                "processed cache item: "
                f"idx={item_idx}, cache_key={cache_key}, cache_type={manifest_item.get('cache_type')}, "
                f"saved_tensors={saved_tensors}, quant_base_exists={base_payload is not None}, "
                f"quant_delta_exists={delta_payload is not None}"
            )

    summary = {
        "full_cache_dir": str(full_cache_dir),
        "quant_cache_dir": str(quant_cache_dir) if quant_cache_dir is not None else None,
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "sample_items": args.sample_items,
        "sample_policy": args.sample_policy,
        "cache_type": args.cache_type,
        "processed_items": processed_items,
        "saved_tensors": saved_tensors,
        "missing_full_items": missing_full,
        "missing_quant_base_items": missing_quant_base,
        "missing_quant_delta_items": missing_quant_delta,
        "stats_path": str(stats_path),
        "tensor_dir": str(tensor_dir),
    }
    with (stats_dir / "cache_tensor_inspect_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "GOFA Scheme-B cache tensor inspection complete: "
        f"processed_items={processed_items}, saved_tensors={saved_tensors}, "
        f"missing_full={missing_full}, missing_quant_base={missing_quant_base}, "
        f"missing_quant_delta={missing_quant_delta}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
