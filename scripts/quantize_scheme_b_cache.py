#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gofa.cache_policy import compute_static_tiers
from modules.gofa.cache_quant import (
    QUANT_BASE_FORMAT,
    QUANT_DELTA_FORMAT,
    estimate_quantized_tensor_bits,
    quantize_scheme_b_cache_item,
)


def _load_degree_metadata(path: Optional[str]) -> Any:
    if not path:
        return None
    degree_path = Path(path)
    if degree_path.suffix == ".json":
        with degree_path.open("r") as f:
            data = json.load(f)
    elif degree_path.suffix in {".pt", ".pth"}:
        data = torch.load(degree_path, map_location="cpu")
    elif degree_path.suffix == ".npy":
        data = np.load(degree_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
    else:
        raise ValueError(f"Unsupported degree metadata format: {degree_path}")
    if isinstance(data, dict) and "degrees" in data:
        return data["degrees"]
    return data


def _lookup_degree(degrees: Any, payload: Dict, cache_key: str, ordinal: int) -> Optional[float]:
    for field in ("degree", "node_degree", "static_degree"):
        if field in payload:
            return float(payload[field])
    if degrees is None:
        return None
    if isinstance(degrees, dict):
        for key in (cache_key, str(ordinal), ordinal, payload.get("node_id"), payload.get("global_node_id")):
            if key is not None and key in degrees:
                return float(degrees[key])
        return None
    if isinstance(degrees, np.ndarray):
        if ordinal < degrees.shape[0]:
            return float(degrees[ordinal])
        return None
    if isinstance(degrees, (list, tuple)):
        if ordinal < len(degrees):
            return float(degrees[ordinal])
        return None
    return None


def _iter_cache_files(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.rglob("*.pt")
        if path.is_file() and "delta" not in path.relative_to(input_dir).parts
    )


def _manifest_item_relpath(item: Dict[str, Any]) -> Path:
    relpath = item.get("relpath") or item.get("source_cache_relpath")
    if not relpath:
        cache_key = item.get("cache_key")
        raise RuntimeError(f"Manifest cache item is missing relpath: cache_key={cache_key}")
    relpath = Path(relpath)
    if relpath.is_absolute():
        raise RuntimeError(f"Manifest relpath must be relative, got: {relpath}")
    if ".." in relpath.parts:
        raise RuntimeError(f"Manifest relpath must not contain '..': {relpath}")
    return relpath


def _resolve_manifest_full_path(input_dir: Path, relpath: Path) -> Path:
    candidate = input_dir / relpath
    if candidate.exists():
        return candidate
    if len(relpath.parts) >= 3 and input_dir.name == relpath.parts[0]:
        tag_root_candidate = input_dir / Path(*relpath.parts[1:])
        if tag_root_candidate.exists():
            return tag_root_candidate
        return tag_root_candidate
    return candidate


def _load_manifest_entries(manifest_path: Path, input_dir: Path, skip_missing: bool) -> tuple[List[Dict[str, Any]], int]:
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    items = manifest.get("cache_items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Manifest cache_items must be a list: {manifest_path}")
    entries = []
    missing = 0
    seen = set()
    for ordinal, item in enumerate(items):
        relpath = _manifest_item_relpath(item)
        cache_key = item.get("cache_key")
        dedupe_key = cache_key or str(relpath)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        full_path = _resolve_manifest_full_path(input_dir, relpath)
        if not full_path.exists():
            missing += 1
            if skip_missing:
                print(f"missing manifest cache item skipped: cache_key={cache_key}, full_path={full_path}")
                continue
            raise RuntimeError(f"Missing manifest full cache item: cache_key={cache_key}, full_path={full_path}")
        entries.append({
            "path": full_path,
            "relpath": relpath,
            "cache_key": cache_key,
            "manifest_item": item,
            "manifest_ordinal": ordinal,
        })
    return entries, missing


def _load_scheme_b_payload(path: Path) -> Optional[Dict]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("cache_format") != "memory_text_kv_v1":
        return None
    required = ("cache_key", "seq_len", "text_len", "mem_size", "memory_state", "text_kv")
    if any(key not in payload for key in required):
        return None
    return payload


def _save_atomic(payload: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _relative_output_path(input_dir: Path, output_dir: Path, cache_file: Path) -> Path:
    relpath = cache_file.relative_to(input_dir)
    if len(relpath.parts) == 2:
        relpath = Path(input_dir.name) / relpath
    return output_dir / relpath


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _optional_bits(value: Optional[int], fallback: int) -> int:
    return int(fallback) if value is None else int(value)


def _ceil_bits_to_bytes(num_bits: int) -> int:
    return (int(num_bits) + 7) // 8


def _component_original_bytes(payload: Dict) -> Dict[str, int]:
    memory_bytes = int(payload["memory_state"].numel() * payload["memory_state"].element_size())
    key_bytes = 0
    value_bytes = 0
    for layer_kv in payload.get("text_kv", []):
        key = layer_kv.get("key")
        value = layer_kv.get("value")
        if key is not None:
            key_bytes += int(key.numel() * key.element_size())
        if value is not None:
            value_bytes += int(value.numel() * value.element_size())
    return {"memory": memory_bytes, "key": key_bytes, "value": value_bytes}


def _component_quantized_bytes(base_payload: Dict, delta_payload: Dict) -> Dict[str, Dict[str, int]]:
    memory_base = _ceil_bits_to_bytes(estimate_quantized_tensor_bits(base_payload.get("memory_state")))
    memory_delta = _ceil_bits_to_bytes(estimate_quantized_tensor_bits(delta_payload.get("memory_state")))
    key_base = 0
    key_delta = 0
    value_base = 0
    value_delta = 0
    for layer_base, layer_delta in zip(base_payload.get("text_kv", []), delta_payload.get("text_kv", [])):
        key_base += _ceil_bits_to_bytes(estimate_quantized_tensor_bits(layer_base.get("key")))
        key_delta += _ceil_bits_to_bytes(estimate_quantized_tensor_bits(layer_delta.get("key")))
        value_base += _ceil_bits_to_bytes(estimate_quantized_tensor_bits(layer_base.get("value")))
        value_delta += _ceil_bits_to_bytes(estimate_quantized_tensor_bits(layer_delta.get("value")))
    return {
        "memory": {"base": memory_base, "delta": memory_delta},
        "key": {"base": key_base, "delta": key_delta},
        "value": {"base": value_base, "delta": value_delta},
    }


def main():
    parser = argparse.ArgumentParser(description="Quantize GOFA scheme-B memory/text-KV cache.")
    parser.add_argument("--input-cache-dir", required=True, help="Existing full-precision scheme-B cache root.")
    parser.add_argument("--output-cache-dir", required=True, help="Output root for quantized base cache.")
    parser.add_argument("--manifest", default=None, help="Optional task-specific encoder cache manifest JSON.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing manifest full cache items instead of raising.")
    parser.add_argument("--base-bits", type=int, default=8, choices=(2, 4, 8, 16))
    parser.add_argument("--delta-bits", type=int, default=4, choices=(2, 4, 8, 16))
    parser.add_argument("--memory-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--key-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--value-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--memory-delta-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--key-delta-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--value-delta-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--static-high-ratio", type=float, default=0.10)
    parser.add_argument("--static-mid-ratio", type=float, default=0.40)
    parser.add_argument(
        "--degree-metadata",
        default=None,
        help="Optional JSON/PT/NPY mapping cache_key or sorted-file ordinal to node degree.",
    )
    parser.add_argument(
        "--fake-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable for identity/debug cache; base_bits=16 is also identity.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_cache_dir).resolve()
    output_dir = Path(args.output_cache_dir).resolve()
    delta_dir = output_dir / "delta"
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    effective_memory_base_bits = _optional_bits(args.memory_base_bits, args.base_bits)
    effective_key_base_bits = _optional_bits(args.key_base_bits, args.base_bits)
    effective_value_base_bits = _optional_bits(args.value_base_bits, args.base_bits)
    effective_memory_delta_bits = _optional_bits(args.memory_delta_bits, args.delta_bits)
    effective_key_delta_bits = _optional_bits(args.key_delta_bits, args.delta_bits)
    effective_value_delta_bits = _optional_bits(args.value_delta_bits, args.delta_bits)
    print(
        "GOFA scheme-B component quantization bits: "
        f"base_bits={args.base_bits}, delta_bits={args.delta_bits}, "
        f"memory_base_bits={effective_memory_base_bits}, key_base_bits={effective_key_base_bits}, "
        f"value_base_bits={effective_value_base_bits}, memory_delta_bits={effective_memory_delta_bits}, "
        f"key_delta_bits={effective_key_delta_bits}, value_delta_bits={effective_value_delta_bits}"
    )

    manifest_entries = None
    manifest_item_count = 0
    missing_item_count = 0
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r") as f:
            manifest_item_count = len(json.load(f).get("cache_items", []))
        manifest_entries, missing_item_count = _load_manifest_entries(
            manifest_path,
            input_dir,
            skip_missing=args.skip_missing,
        )
        cache_files = [entry["path"] for entry in manifest_entries]
        print(
            "Using task-specific cache manifest: "
            f"manifest={manifest_path}, manifest_item_count={manifest_item_count}, "
            f"candidate_item_count={len(cache_files)}, missing_item_count={missing_item_count}"
        )
    else:
        cache_files = _iter_cache_files(input_dir)
    degree_metadata = _load_degree_metadata(args.degree_metadata)
    entries = []
    missing_degrees = 0
    original_bytes = 0

    for ordinal, cache_file in enumerate(cache_files):
        payload = _load_scheme_b_payload(cache_file)
        if payload is None:
            continue
        manifest_entry = manifest_entries[ordinal] if manifest_entries is not None else None
        expected_cache_key = manifest_entry.get("cache_key") if manifest_entry is not None else None
        if expected_cache_key and payload.get("cache_key") != expected_cache_key:
            raise RuntimeError(
                "Manifest cache_key does not match full cache payload: "
                f"manifest_cache_key={expected_cache_key}, payload_cache_key={payload.get('cache_key')}, "
                f"path={cache_file}"
            )
        cache_key = payload["cache_key"]
        degree = _lookup_degree(degree_metadata, payload, cache_key, ordinal)
        manifest_item = manifest_entry.get("manifest_item") if manifest_entry is not None else None
        if degree is None and isinstance(manifest_item, dict):
            for degree_field in ("global_degree", "local_degree"):
                if manifest_item.get(degree_field) is not None:
                    degree = float(manifest_item[degree_field])
                    break
        if degree is None:
            missing_degrees += 1
            degree = 0.0
        entries.append({
            "path": cache_file,
            "relpath": manifest_entry.get("relpath") if manifest_entry is not None else None,
            "cache_key": cache_key,
            "degree": degree,
            "manifest_item": manifest_item,
        })
        original_bytes += cache_file.stat().st_size

    if not entries:
        raise RuntimeError(f"No full-precision scheme-B cache payloads found under {input_dir}")

    if missing_degrees == len(entries):
        tiers = ["low"] * len(entries)
        print("No degree metadata found; assigning static_tier=low to every cache item.")
    else:
        tiers = compute_static_tiers(
            [entry["degree"] for entry in entries],
            high_ratio=args.static_high_ratio,
            mid_ratio=args.static_mid_ratio,
        )
        if missing_degrees:
            print(f"Degree metadata missing for {missing_degrees}/{len(entries)} items; missing entries use degree=0.")

    base_bytes = 0
    delta_bytes = 0
    component_original_bytes = {"memory": 0, "key": 0, "value": 0}
    component_base_bytes = {"memory": 0, "key": 0, "value": 0}
    component_delta_bytes = {"memory": 0, "key": 0, "value": 0}
    converted = 0
    for ordinal, entry in enumerate(entries):
        cache_file = entry["path"]
        payload = _load_scheme_b_payload(cache_file)
        if payload is None:
            continue
        base_payload, delta_payload = quantize_scheme_b_cache_item(
            payload,
            base_bits=args.base_bits,
            delta_bits=args.delta_bits,
            memory_base_bits=args.memory_base_bits,
            key_base_bits=args.key_base_bits,
            value_base_bits=args.value_base_bits,
            memory_delta_bits=args.memory_delta_bits,
            key_delta_bits=args.key_delta_bits,
            value_delta_bits=args.value_delta_bits,
            static_tier=tiers[ordinal],
            fake_quant=args.fake_quant,
        )
        original_component = _component_original_bytes(payload)
        quantized_component = _component_quantized_bytes(base_payload, delta_payload)
        for component in ("memory", "key", "value"):
            component_original_bytes[component] += original_component[component]
            component_base_bytes[component] += quantized_component[component]["base"]
            component_delta_bytes[component] += quantized_component[component]["delta"]
        common_metadata = {
            "cache_key": payload["cache_key"],
            "seq_len": payload["seq_len"],
            "source_cache_relpath": str(entry["relpath"] or cache_file.relative_to(input_dir)),
            "static_degree": entry["degree"],
            "static_high_ratio": args.static_high_ratio,
            "static_mid_ratio": args.static_mid_ratio,
        }
        if entry.get("manifest_item"):
            manifest_item = entry["manifest_item"]
            common_metadata["manifest_cache_type"] = manifest_item.get("cache_type", "unknown")
            common_metadata["manifest_relpath"] = str(entry["relpath"])
            for metadata_field in (
                "local_node_idx",
                "global_node_id",
                "node_id_string",
                "is_target",
                "is_target_neighbor",
                "hop_distance_to_target",
                "local_degree",
                "global_degree",
                "local_edge_idx",
                "src_local",
                "dst_local",
                "src_global",
                "dst_global",
                "is_incident_to_target",
                "both_endpoints_target_or_neighbor",
            ):
                if metadata_field in manifest_item:
                    common_metadata[f"manifest_{metadata_field}"] = manifest_item.get(metadata_field)
        base_payload.update(common_metadata)
        delta_payload.update(common_metadata)

        if entry["relpath"] is not None:
            base_path = output_dir / entry["relpath"]
            delta_path = delta_dir / entry["relpath"]
        else:
            base_path = _relative_output_path(input_dir, output_dir, cache_file)
            delta_path = _relative_output_path(input_dir, delta_dir, cache_file)
        _save_atomic(base_payload, base_path)
        _save_atomic(delta_payload, delta_path)
        base_bytes += base_path.stat().st_size
        delta_bytes += delta_path.stat().st_size
        converted += 1
        if converted <= 3 or converted % 100 == 0:
            print(
                "converted "
                f"{converted}/{len(entries)} cache_key={payload['cache_key'][:8]} "
                f"tier={tiers[ordinal]} base_format={QUANT_BASE_FORMAT} delta_format={QUANT_DELTA_FORMAT}"
            )

    quant_bytes = base_bytes + delta_bytes
    compression_ratio = (original_bytes / quant_bytes) if quant_bytes else 0.0
    tier_counts = {tier: tiers.count(tier) for tier in ("high", "mid", "low")}
    logical_original_bytes = sum(component_original_bytes.values())
    logical_base_bytes = sum(component_base_bytes.values())
    logical_delta_bytes = sum(component_delta_bytes.values())
    print(
        "GOFA scheme-B quantized cache stats: "
        f"manifest_item_count={manifest_item_count if args.manifest else 0}, "
        f"converted_item_count={converted}, missing_item_count={missing_item_count}, "
        f"items={converted}, tiers={tier_counts}, "
        f"original={_format_bytes(original_bytes)}, "
        f"base={_format_bytes(base_bytes)}, "
        f"delta={_format_bytes(delta_bytes)}, "
        f"total_quantized={_format_bytes(quant_bytes)}, "
        f"compression_ratio={compression_ratio:.3f}x"
    )
    print(
        "GOFA scheme-B component logical cache stats: "
        f"memory_original={_format_bytes(component_original_bytes['memory'])}, "
        f"memory_base={_format_bytes(component_base_bytes['memory'])}, "
        f"memory_delta={_format_bytes(component_delta_bytes['memory'])}, "
        f"key_original={_format_bytes(component_original_bytes['key'])}, "
        f"key_base={_format_bytes(component_base_bytes['key'])}, "
        f"key_delta={_format_bytes(component_delta_bytes['key'])}, "
        f"value_original={_format_bytes(component_original_bytes['value'])}, "
        f"value_base={_format_bytes(component_base_bytes['value'])}, "
        f"value_delta={_format_bytes(component_delta_bytes['value'])}, "
        f"total_original={_format_bytes(logical_original_bytes)}, "
        f"total_base={_format_bytes(logical_base_bytes)}, "
        f"total_delta={_format_bytes(logical_delta_bytes)}, "
        f"total_quantized={_format_bytes(logical_base_bytes + logical_delta_bytes)}"
    )


if __name__ == "__main__":
    main()
