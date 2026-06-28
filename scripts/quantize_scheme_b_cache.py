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
from modules.gofa.cache_quant import QUANT_BASE_FORMAT, QUANT_DELTA_FORMAT, quantize_scheme_b_cache_item


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
    return output_dir / cache_file.relative_to(input_dir)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def main():
    parser = argparse.ArgumentParser(description="Quantize GOFA scheme-B memory/text-KV cache.")
    parser.add_argument("--input-cache-dir", required=True, help="Existing full-precision scheme-B cache root.")
    parser.add_argument("--output-cache-dir", required=True, help="Output root for quantized base cache.")
    parser.add_argument("--base-bits", type=int, default=8, choices=(4, 8, 16))
    parser.add_argument("--delta-bits", type=int, default=4, choices=(4, 8, 16))
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

    cache_files = _iter_cache_files(input_dir)
    degree_metadata = _load_degree_metadata(args.degree_metadata)
    entries = []
    missing_degrees = 0
    original_bytes = 0

    for ordinal, cache_file in enumerate(cache_files):
        payload = _load_scheme_b_payload(cache_file)
        if payload is None:
            continue
        cache_key = payload["cache_key"]
        degree = _lookup_degree(degree_metadata, payload, cache_key, ordinal)
        if degree is None:
            missing_degrees += 1
            degree = 0.0
        entries.append({"path": cache_file, "cache_key": cache_key, "degree": degree})
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
            static_tier=tiers[ordinal],
            fake_quant=args.fake_quant,
        )
        common_metadata = {
            "cache_key": payload["cache_key"],
            "seq_len": payload["seq_len"],
            "source_cache_relpath": str(cache_file.relative_to(input_dir)),
            "static_degree": entry["degree"],
            "static_high_ratio": args.static_high_ratio,
            "static_mid_ratio": args.static_mid_ratio,
        }
        base_payload.update(common_metadata)
        delta_payload.update(common_metadata)

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
    print(
        "GOFA scheme-B quantized cache stats: "
        f"items={converted}, tiers={tier_counts}, "
        f"original={_format_bytes(original_bytes)}, "
        f"base={_format_bytes(base_bytes)}, "
        f"delta={_format_bytes(delta_bytes)}, "
        f"total_quantized={_format_bytes(quant_bytes)}, "
        f"compression_ratio={compression_ratio:.3f}x"
    )


if __name__ == "__main__":
    main()

