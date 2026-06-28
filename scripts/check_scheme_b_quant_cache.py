#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gofa.cache_quant import QUANT_BASE_FORMAT, reconstruct_scheme_b_cache_item


def _iter_full_cache_files(full_cache_dir: Path) -> Iterable[Path]:
    for path in sorted(full_cache_dir.rglob("*.pt")):
        if "delta" in path.relative_to(full_cache_dir).parts:
            continue
        yield path


def _load_full_scheme_b_payload(path: Path) -> Optional[Dict]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("cache_format") != "memory_text_kv_v1":
        return None
    return payload


def _cache_tag_and_relpath(full_cache_dir: Path, full_cache_file: Path) -> Tuple[str, Path]:
    relpath = full_cache_file.relative_to(full_cache_dir)
    if len(relpath.parts) >= 3:
        return relpath.parts[0], Path(*relpath.parts[1:])
    if len(relpath.parts) == 2:
        return full_cache_dir.name, relpath
    raise RuntimeError(f"Unexpected full cache relative path: {relpath}")


def _quant_paths(full_cache_dir: Path, quant_cache_dir: Path, full_cache_file: Path) -> Tuple[str, Path, Path]:
    cache_tag, relpath = _cache_tag_and_relpath(full_cache_dir, full_cache_file)
    return cache_tag, quant_cache_dir / cache_tag / relpath, quant_cache_dir / "delta" / cache_tag / relpath


def _assert_shape_match(full_item: Dict, reconstructed_item: Dict, quant_path: Path):
    if tuple(full_item["memory_state"].shape) != tuple(reconstructed_item["memory_state"].shape):
        raise RuntimeError(
            f"memory_state shape mismatch for {quant_path}: "
            f"full={tuple(full_item['memory_state'].shape)}, "
            f"reconstructed={tuple(reconstructed_item['memory_state'].shape)}"
        )
    if len(full_item["text_kv"]) != len(reconstructed_item["text_kv"]):
        raise RuntimeError(
            f"text_kv layer count mismatch for {quant_path}: "
            f"full={len(full_item['text_kv'])}, reconstructed={len(reconstructed_item['text_kv'])}"
        )
    for layer_idx, (full_layer, reconstructed_layer) in enumerate(zip(full_item["text_kv"], reconstructed_item["text_kv"])):
        for name in ("key", "value"):
            if tuple(full_layer[name].shape) != tuple(reconstructed_layer[name].shape):
                raise RuntimeError(
                    f"text_kv {name} shape mismatch for {quant_path}, layer={layer_idx}: "
                    f"full={tuple(full_layer[name].shape)}, "
                    f"reconstructed={tuple(reconstructed_layer[name].shape)}"
                )


def main():
    parser = argparse.ArgumentParser(description="Validate GOFA Scheme-B quant cache files against full cache files.")
    parser.add_argument("--full-cache-dir", required=True)
    parser.add_argument("--quant-cache-dir", required=True)
    parser.add_argument("--base-bits", type=int, required=True)
    parser.add_argument("--max-items", type=int, default=10)
    args = parser.parse_args()

    full_cache_dir = Path(args.full_cache_dir).resolve()
    quant_cache_dir = Path(args.quant_cache_dir).resolve()
    if not full_cache_dir.is_dir():
        raise FileNotFoundError(full_cache_dir)
    if not quant_cache_dir.is_dir():
        raise FileNotFoundError(quant_cache_dir)

    checked = 0
    for full_cache_file in _iter_full_cache_files(full_cache_dir):
        full_payload = _load_full_scheme_b_payload(full_cache_file)
        if full_payload is None:
            continue
        cache_tag, quant_base_path, quant_delta_path = _quant_paths(full_cache_dir, quant_cache_dir, full_cache_file)
        if not quant_base_path.exists():
            raise RuntimeError(
                f"Missing quant base cache: cache_tag={cache_tag}, "
                f"full={full_cache_file}, quant_base={quant_base_path}"
            )
        quant_payload = torch.load(quant_base_path, map_location="cpu")
        if quant_payload.get("cache_format") != QUANT_BASE_FORMAT:
            raise RuntimeError(
                f"Invalid quant cache format for {quant_base_path}: "
                f"{quant_payload.get('cache_format')} != {QUANT_BASE_FORMAT}"
            )
        if int(quant_payload.get("base_bits", -1)) != int(args.base_bits):
            raise RuntimeError(
                f"base_bits mismatch for {quant_base_path}: "
                f"payload={quant_payload.get('base_bits')}, expected={args.base_bits}"
            )
        reconstructed = reconstruct_scheme_b_cache_item(quant_payload, delta_payload=None, load_delta=False)
        _assert_shape_match(full_payload, reconstructed, quant_base_path)
        checked += 1
        print(
            "checked "
            f"{checked}: cache_key={full_payload.get('cache_key')}, cache_tag={cache_tag}, "
            f"full_memory_shape={tuple(full_payload['memory_state'].shape)}, "
            f"reconstructed_memory_shape={tuple(reconstructed['memory_state'].shape)}, "
            f"quant_base={quant_base_path}, quant_base_exists={quant_base_path.exists()}, "
            f"quant_delta={quant_delta_path}, quant_delta_exists={quant_delta_path.exists()}"
        )
        if checked >= args.max_items:
            break

    if checked == 0:
        raise RuntimeError(f"No memory_text_kv_v1 full cache items found under {full_cache_dir}")
    print(f"GOFA Scheme-B quant cache check passed: checked={checked}")


if __name__ == "__main__":
    main()

