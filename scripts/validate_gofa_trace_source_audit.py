#!/usr/bin/env python3
"""Validate minimal GOFA trace-source audit snapshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


REQUIRED_STAGES = (
    "dataset_sample_or_pre_batch",
    "model_encoder_entry",
    "after_skip_nog_resolution",
    "after_selective_kv_policy",
    "before_forward_memory_with_text_kv",
    "after_map_node_reorder",
)


def _failures() -> list[str]:
    return []


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _walk_numbers(value: Any, path: str = "$") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk_numbers(item, f"{path}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{path}.{key}")


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _numel_from_shape(shape: Any) -> int | None:
    if not isinstance(shape, list):
        return None
    total = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        total *= dim
    return total


def _check_field_shape(name: str, field: Any, errors: list[str]) -> None:
    if not isinstance(field, dict):
        errors.append(f"{name} is not an object")
        return
    status = field.get("status")
    if status == "missing":
        return
    raw = field.get("raw")
    shape = field.get("shape")
    numel = field.get("numel")
    if shape is not None and numel is not None:
        expected = _numel_from_shape(shape)
        if expected is not None and int(numel) != expected:
            errors.append(f"{name} numel={numel} does not match shape product={expected}")
    if raw is not None and numel is not None:
        raw_numel = len(_flatten(raw))
        if raw_numel != int(numel) and raw_numel > 32:
            errors.append(f"{name} raw length={raw_numel} does not match numel={numel}")


def _check_index_array(name: str, value: Any, errors: list[str]) -> None:
    if value is None:
        return
    for idx, item in enumerate(_flatten(value)):
        if item is None:
            continue
        if not isinstance(item, int) or isinstance(item, bool):
            errors.append(f"{name}[{idx}] is not an integer/null value: {item!r}")
            return


def validate_query(path: Path) -> list[str]:
    errors = _failures()
    query = _load_json(path)
    if "query_id" not in query:
        errors.append("query_id missing")
    stages = query.get("stages")
    if not isinstance(stages, dict):
        errors.append("stages missing or not an object")
    else:
        for stage in REQUIRED_STAGES:
            if stage not in stages:
                errors.append(f"stage missing: {stage}")

    for number_path, number in _walk_numbers(query):
        if not math.isfinite(number):
            errors.append(f"non-finite number at {number_path}")

    for top_field in ("target_index", "node_map", "edge_map"):
        if top_field not in query:
            errors.append(f"{top_field} top-level field missing")
    if "nog" not in query:
        errors.append("nog field missing")
    elif not isinstance(query["nog"], dict) or "question_index_raw" not in query["nog"]:
        errors.append("nog.question_index_raw missing")

    if "batching" not in query:
        errors.append("batching field missing")
    else:
        batching = query["batching"]
        if isinstance(batching, dict):
            for field_name in ("target_index", "question_index", "node_map", "edge_map"):
                if field_name not in batching:
                    errors.append(f"batching.{field_name} missing")
                else:
                    _check_field_shape(f"batching.{field_name}", batching[field_name], errors)

    target = query.get("target_index", {})
    if isinstance(target, dict):
        _check_index_array("target_index.raw", target.get("raw"), errors)
        if "question_index_raw" not in target:
            errors.append("target_index.question_index_raw missing")

    node_map = query.get("node_map", {})
    if isinstance(node_map, dict):
        _check_index_array("node_map.raw", node_map.get("raw"), errors)
        if "values_at_target_index" not in node_map:
            errors.append("node_map.values_at_target_index missing")
        if "values_at_question_index" not in node_map:
            errors.append("node_map.values_at_question_index missing")

    edge_map = query.get("edge_map", {})
    if isinstance(edge_map, dict):
        _check_index_array("edge_map.raw", edge_map.get("raw"), errors)

    selective_kv = query.get("selective_kv")
    if not isinstance(selective_kv, dict):
        errors.append("selective_kv missing or not an object")
    else:
        for field_name in (
            "policy_selected_key_item_indices",
            "policy_selected_value_item_indices",
            "effective_consumed_key_item_indices",
            "effective_consumed_value_item_indices",
            "complete_kv_item_indices",
            "eligible_item_indices",
            "skip_cache_indices",
        ):
            if field_name not in selective_kv:
                errors.append(f"selective_kv.{field_name} missing")
            else:
                _check_index_array(f"selective_kv.{field_name}", selective_kv[field_name], errors)

    if isinstance(stages, dict):
        policy_stage = stages.get("after_selective_kv_policy", {})
        if isinstance(policy_stage, dict):
            for mask_name in ("key_policy_mask", "value_policy_mask", "complete_kv_policy_mask"):
                mask = policy_stage.get(mask_name)
                if not isinstance(mask, list):
                    errors.append(f"after_selective_kv_policy.{mask_name} missing or not a list")
                elif not all(isinstance(item, bool) for item in mask):
                    errors.append(f"after_selective_kv_policy.{mask_name} contains non-bool values")
        before_forward = stages.get("before_forward_memory_with_text_kv", {})
        if isinstance(before_forward, dict):
            for field_name in ("effective_key_item_indices", "effective_value_item_indices"):
                if field_name not in before_forward:
                    errors.append(f"before_forward_memory_with_text_kv.{field_name} missing")

    token_item_count = query.get("token_item_count")
    if isinstance(token_item_count, int) and isinstance(stages, dict):
        masks = stages.get("after_selective_kv_policy", {})
        if isinstance(masks, dict):
            for mask_name in ("key_policy_mask", "value_policy_mask", "complete_kv_policy_mask"):
                mask = masks.get(mask_name)
                if isinstance(mask, list) and len(mask) != token_item_count:
                    errors.append(f"{mask_name} length={len(mask)} does not match token_item_count={token_item_count}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GOFA trace-source audit output.")
    parser.add_argument("--input-dir", required=True, help="Audit output directory.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    metadata_path = input_dir / "audit_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"audit_metadata.json missing in {input_dir}")
    _load_json(metadata_path)

    query_paths = sorted(input_dir.glob("query_*.json"))
    if not query_paths:
        raise SystemExit(f"no query_*.json files found in {input_dir}")

    all_errors: list[str] = []
    for query_path in query_paths:
        for error in validate_query(query_path):
            all_errors.append(f"{query_path.name}: {error}")
    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Validated {len(query_paths)} query audit file(s) in {input_dir}")


if __name__ == "__main__":
    main()
