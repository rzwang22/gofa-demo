#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gofa.cache_quant import QUANT_BASE_FORMAT, QUANT_DELTA_FORMAT, reconstruct_scheme_b_cache_item


def _optional_bits(value: Optional[int], fallback: int) -> int:
    return int(fallback) if value is None else int(value)


def _iter_full_cache_files(full_cache_dir: Path) -> Iterable[Path]:
    for path in sorted(full_cache_dir.rglob("*.pt")):
        if "delta" in path.relative_to(full_cache_dir).parts:
            continue
        yield path


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


def _load_manifest_entries(manifest_path: Path, full_cache_dir: Path, max_items: int) -> List[Tuple[Path, Path, Optional[str]]]:
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    items = manifest.get("cache_items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Manifest cache_items must be a list: {manifest_path}")
    entries = []
    seen = set()
    for item in items:
        relpath = _manifest_item_relpath(item)
        cache_key = item.get("cache_key")
        dedupe_key = cache_key or str(relpath)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append((_resolve_manifest_full_path(full_cache_dir, relpath), relpath, cache_key))
        if len(entries) >= max_items:
            break
    return entries


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


def _manifest_quant_paths(quant_cache_dir: Path, relpath: Path) -> Tuple[str, Path, Path]:
    if len(relpath.parts) < 3:
        raise RuntimeError(f"Manifest relpath must include <cache_tag>/<prefix>/<cache_key>.pt: {relpath}")
    cache_tag = relpath.parts[0]
    return cache_tag, quant_cache_dir / relpath, quant_cache_dir / "delta" / relpath


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


def _payload_component_base_bits(payload: Dict) -> Dict[str, int]:
    fallback = payload.get("base_bits")
    return {
        "memory_base_bits": int(payload.get("memory_base_bits", fallback if fallback is not None else -1)),
        "key_base_bits": int(payload.get("key_base_bits", fallback if fallback is not None else -1)),
        "value_base_bits": int(payload.get("value_base_bits", fallback if fallback is not None else -1)),
    }


def _assert_component_base_bits(payload: Dict, expected: Dict[str, int], quant_path: Path):
    payload_bits = _payload_component_base_bits(payload)
    mismatches = {
        key: {"payload": payload_bits[key], "expected": expected[key]}
        for key in expected
        if payload_bits[key] != expected[key]
    }
    if mismatches:
        raise RuntimeError(
            f"base component bits mismatch for {quant_path}: "
            f"payload_base_bits={payload.get('base_bits')}, "
            f"payload_component_bits={payload_bits}, expected={expected}, mismatches={mismatches}"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate GOFA Scheme-B quant cache files against full cache files.")
    parser.add_argument("--full-cache-dir", required=True)
    parser.add_argument("--quant-cache-dir", required=True)
    parser.add_argument("--base-bits", type=int, required=True, choices=(2, 4, 8, 16))
    parser.add_argument("--memory-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--key-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--value-base-bits", type=int, default=None, choices=(2, 4, 8, 16))
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--manifest", default=None, help="Optional task-specific encoder cache manifest JSON.")
    args = parser.parse_args()

    full_cache_dir = Path(args.full_cache_dir).resolve()
    quant_cache_dir = Path(args.quant_cache_dir).resolve()
    if not full_cache_dir.is_dir():
        raise FileNotFoundError(full_cache_dir)
    if not quant_cache_dir.is_dir():
        raise FileNotFoundError(quant_cache_dir)
    expected_component_base_bits = {
        "memory_base_bits": _optional_bits(args.memory_base_bits, args.base_bits),
        "key_base_bits": _optional_bits(args.key_base_bits, args.base_bits),
        "value_base_bits": _optional_bits(args.value_base_bits, args.base_bits),
    }

    checked = 0
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        full_entries = _load_manifest_entries(manifest_path, full_cache_dir, args.max_items)
    else:
        full_entries = ((path, None, None) for path in _iter_full_cache_files(full_cache_dir))

    for full_cache_file, manifest_relpath, expected_cache_key in full_entries:
        if not full_cache_file.exists():
            raise RuntimeError(f"Missing full cache item: {full_cache_file}")
        full_payload = _load_full_scheme_b_payload(full_cache_file)
        if full_payload is None:
            continue
        if expected_cache_key and full_payload.get("cache_key") != expected_cache_key:
            raise RuntimeError(
                "Manifest cache_key does not match full cache payload: "
                f"manifest_cache_key={expected_cache_key}, payload_cache_key={full_payload.get('cache_key')}, "
                f"path={full_cache_file}"
            )
        if manifest_relpath is not None:
            cache_tag, quant_base_path, quant_delta_path = _manifest_quant_paths(quant_cache_dir, manifest_relpath)
        else:
            cache_tag, quant_base_path, quant_delta_path = _quant_paths(full_cache_dir, quant_cache_dir, full_cache_file)
        if not quant_base_path.exists():
            raise RuntimeError(
                f"Missing quant base cache: cache_tag={cache_tag}, "
                f"full={full_cache_file}, quant_base={quant_base_path}"
            )
        if not quant_delta_path.exists():
            raise RuntimeError(
                f"Missing quant delta cache: cache_tag={cache_tag}, "
                f"full={full_cache_file}, quant_delta={quant_delta_path}"
            )
        quant_payload = torch.load(quant_base_path, map_location="cpu")
        if quant_payload.get("cache_format") != QUANT_BASE_FORMAT:
            raise RuntimeError(
                f"Invalid quant cache format for {quant_base_path}: "
                f"{quant_payload.get('cache_format')} != {QUANT_BASE_FORMAT}"
            )
        delta_payload = torch.load(quant_delta_path, map_location="cpu")
        if delta_payload.get("cache_format") != QUANT_DELTA_FORMAT:
            raise RuntimeError(
                f"Invalid quant delta cache format for {quant_delta_path}: "
                f"{delta_payload.get('cache_format')} != {QUANT_DELTA_FORMAT}"
            )
        _assert_component_base_bits(quant_payload, expected_component_base_bits, quant_base_path)
        reconstructed = reconstruct_scheme_b_cache_item(quant_payload, delta_payload=delta_payload, load_delta=True)
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
