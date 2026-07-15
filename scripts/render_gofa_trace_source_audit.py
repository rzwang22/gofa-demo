#!/usr/bin/env python3
"""Render real GOFA trace-source audit snapshots into a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text[:240] + ("..." if len(text) > 240 else "")


def _first_query(input_dir: Path) -> dict[str, Any] | None:
    query_paths = sorted(input_dir.glob("query_*.json"))
    if not query_paths:
        return None
    return _load_json(query_paths[0])


def _summarize_dir(input_dir: Path) -> dict[str, Any]:
    metadata_path = input_dir / "audit_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    query_paths = sorted(input_dir.glob("query_*.json"))
    query = _first_query(input_dir)
    selective = query.get("selective_kv", {}) if isinstance(query, dict) else {}
    edge_map = query.get("edge_map", {}) if isinstance(query, dict) else {}
    node_map = query.get("node_map", {}) if isinstance(query, dict) else {}
    target = query.get("target_index", {}) if isinstance(query, dict) else {}
    return {
        "input_dir": str(input_dir),
        "query_count": len(query_paths),
        "task_names": metadata.get("task_names"),
        "cache_tag": metadata.get("encoder_cache_namespace") or (query or {}).get("cache_tag"),
        "cache_mode": metadata.get("encoder_cache_mode") or (query or {}).get("cache_mode"),
        "token_item_count": (query or {}).get("token_item_count"),
        "target_raw": target.get("raw") if isinstance(target, dict) else None,
        "node_map_status": node_map.get("status") if isinstance(node_map, dict) else None,
        "node_map_min": node_map.get("min") if isinstance(node_map, dict) else None,
        "node_map_max": node_map.get("max") if isinstance(node_map, dict) else None,
        "edge_map_length": edge_map.get("edge_map_length") if isinstance(edge_map, dict) else None,
        "edge_map_unique_count": edge_map.get("edge_map_unique_count") if isinstance(edge_map, dict) else None,
        "kv_policy": selective.get("policy_name") if isinstance(selective, dict) else None,
        "key_policy_count": _count(selective.get("policy_selected_key_item_indices")) if isinstance(selective, dict) else None,
        "value_policy_count": _count(selective.get("policy_selected_value_item_indices")) if isinstance(selective, dict) else None,
        "effective_key_count": _count(selective.get("effective_consumed_key_item_indices")) if isinstance(selective, dict) else None,
        "effective_value_count": _count(selective.get("effective_consumed_value_item_indices")) if isinstance(selective, dict) else None,
        "complete_kv_count": _count(selective.get("complete_kv_item_indices")) if isinstance(selective, dict) else None,
        "query": query,
        "metadata": metadata,
    }


def _write_report(summaries: list[dict[str, Any]], output: Path) -> None:
    lines: list[str] = []
    lines.append("# GOFA Trace Source Runtime Results")
    lines.append("")
    lines.append(
        "This report is generated only from real `gofa_trace_source_audit` JSON directories passed on the command line. "
        "It does not infer missing runtime semantics."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "input_dir",
        "query_count",
        "task_names",
        "cache_mode",
        "token_item_count",
        "target_raw",
        "kv_policy",
        "key_policy_count",
        "value_policy_count",
        "effective_key_count",
        "effective_value_count",
        "complete_kv_count",
        "edge_map_length",
        "edge_map_unique_count",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for summary in summaries:
        lines.append("| " + " | ".join(_cell(summary.get(header)) for header in headers) + " |")
    lines.append("")
    lines.append("## Per-Run Details")
    lines.append("")
    for summary in summaries:
        lines.append(f"### `{summary['input_dir']}`")
        lines.append("")
        query = summary.get("query") or {}
        if not query:
            lines.append("No query JSON files were present in this directory.")
            lines.append("")
            continue
        lines.append(f"- `query_id`: {_cell(query.get('query_id'))}")
        lines.append(f"- `cache_tag`: {_cell(summary.get('cache_tag'))}")
        lines.append(f"- `target_index.raw`: {_cell((query.get('target_index') or {}).get('raw'))}")
        lines.append(f"- `node_map.values_at_target_index`: {_cell((query.get('node_map') or {}).get('values_at_target_index'))}")
        lines.append(f"- `edge_map.multiplicity_histogram`: {_cell((query.get('edge_map') or {}).get('multiplicity_histogram'))}")
        lines.append(f"- `nog.skip_cache_indices`: {_cell((query.get('nog') or {}).get('skip_cache_indices'))}")
        lines.append(f"- `selective_kv.eligible_item_indices`: {_cell((query.get('selective_kv') or {}).get('eligible_item_indices'))}")
        lines.append(f"- `layer_selection.selection_is_shared_across_suffix_layers`: {_cell((query.get('layer_selection') or {}).get('selection_is_shared_across_suffix_layers'))}")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GOFA trace-source audit JSON into Markdown.")
    parser.add_argument("--input-dir", action="append", required=True, help="Audit output directory. Can be repeated.")
    parser.add_argument("--output", default="GOFA_TRACE_SOURCE_RUNTIME_RESULTS.md", help="Markdown output path.")
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.input_dir]
    if not input_dirs:
        raise SystemExit("at least one --input-dir is required")
    summaries = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise SystemExit(f"input directory does not exist: {input_dir}")
        summaries.append(_summarize_dir(input_dir))
    if not any(summary["query_count"] > 0 for summary in summaries):
        raise SystemExit("no query_*.json files found in any input directory; refusing to fabricate a report")
    output = Path(args.output)
    _write_report(summaries, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
