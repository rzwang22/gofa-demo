#!/usr/bin/env python3
"""Classify GOFA pre-cache diagnostics produced by a strict quant cache miss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _integer_values(value: Any) -> list[int]:
    return [
        int(item)
        for item in _flatten(value)
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _raw_field(snapshot: dict[str, Any], field_name: str) -> Any:
    value = snapshot.get(field_name)
    if isinstance(value, dict) and "raw" in value:
        return value.get("raw")
    return value


def classify_missing_item(item: dict[str, Any]) -> str:
    if item.get("is_question_or_nog_candidate"):
        return "NOG candidate"
    item_type = item.get("item_type")
    if item_type == "node":
        return "node text item"
    if item_type == "edge":
        return "edge text item"
    if item.get("node_map_inverse_local_indices"):
        return "node text item"
    if item.get("structural_edge_positions"):
        return "edge text item"
    return "unresolved"


def analyze_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    question_raw = snapshot.get("question_index_raw")
    if question_raw is None:
        question_raw = _raw_field(snapshot, "question_index")
    question_indices = _integer_values(question_raw)

    mapped_entries = snapshot.get("node_map_at_each_question_index") or []
    mapped_question_items = [
        int(entry["node_map_value"])
        for entry in mapped_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("node_map_value"), int)
        and not isinstance(entry.get("node_map_value"), bool)
    ]
    distinct_mapped_items = sorted(set(mapped_question_items))
    skipped_items = sorted(set(_integer_values(
        snapshot.get("resolved_skip_cache_indices", snapshot.get("skip_cache_indices", []))
    )))
    missing_items = snapshot.get("missing_items") or []
    classified_missing = [
        {
            "cache_key": item.get("cache_key"),
            "item_index": item.get("item_index"),
            "classification": classify_missing_item(item),
            "is_skip_cache_item": item.get("is_skip_cache_item"),
            "is_question_or_nog_candidate": item.get("is_question_or_nog_candidate"),
        }
        for item in missing_items
        if isinstance(item, dict)
    ]
    return {
        "batch_size_inferred": snapshot.get("batch_size_inferred"),
        "number_of_question_indices": len(question_indices),
        "number_of_distinct_mapped_question_items": len(distinct_mapped_items),
        "number_of_skipped_items": len(skipped_items),
        "question_indices": question_indices,
        "distinct_mapped_question_items": distinct_mapped_items,
        "skipped_items": skipped_items,
        "counts_consistent": (
            len(question_indices) == len(distinct_mapped_items) == len(skipped_items)
        ),
        "all_mapped_question_items_skipped": set(distinct_mapped_items).issubset(skipped_items),
        "missing_items": classified_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a GOFA batch-size-2 strict quant cache miss snapshot.")
    parser.add_argument("--input", required=True, help="Path to cache_miss_000000.json.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")
    with input_path.open("r") as handle:
        snapshot = json.load(handle)
    if not isinstance(snapshot, dict):
        raise SystemExit("snapshot root must be a JSON object")

    result = analyze_snapshot(snapshot)
    print(f"input={input_path}")
    print(f"batch_size_inferred={result['batch_size_inferred']}")
    print(f"number_of_question_indices={result['number_of_question_indices']}")
    print(
        "number_of_distinct_mapped_question_items="
        f"{result['number_of_distinct_mapped_question_items']}"
    )
    print(f"number_of_skipped_items={result['number_of_skipped_items']}")
    print(f"question_indices={result['question_indices']}")
    print(f"distinct_mapped_question_items={result['distinct_mapped_question_items']}")
    print(f"skipped_items={result['skipped_items']}")
    print(f"counts_consistent={result['counts_consistent']}")
    print(f"all_mapped_question_items_skipped={result['all_mapped_question_items_skipped']}")
    if not result["missing_items"]:
        print("missing_items=none")
    for item in result["missing_items"]:
        print(
            "missing_key="
            f"{item['cache_key']} item_index={item['item_index']} "
            f"classification={item['classification']} "
            f"is_skip_cache_item={item['is_skip_cache_item']} "
            f"is_question_or_nog_candidate={item['is_question_or_nog_candidate']}"
        )


if __name__ == "__main__":
    main()
