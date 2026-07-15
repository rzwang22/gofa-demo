#!/usr/bin/env python3
"""Summarize formal GOFA query traces by task."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _trace_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("query_*.json"))
    if input_path.suffix == ".jsonl":
        paths = []
        with input_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                path = Path(entry["trace_path"])
                paths.append(path if path.is_absolute() else input_path.parent / path)
        return paths
    return [input_path]


def _percentile(values: list[int | float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": mean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("task_name") or "unknown")].append(trace)
    result = {"trace_format": "gofa_query_trace_summary_v1", "tasks": {}}
    for task_name, task_traces in sorted(grouped.items()):
        graphs = [trace["query_graph_structure"] for trace in task_traces]
        summaries = [trace["summary"] for trace in task_traces]
        traffic = [trace["traffic_metadata"] for trace in task_traces]
        node_text_counts = [graph["num_node_text_items"] for graph in graphs]
        edge_text_counts = [graph["num_edge_text_items"] for graph in graphs]
        persistent = [item["persistent_cache_bytes"] for item in traffic]
        loaded = [item["runtime_loaded_cache_bytes"] for item in traffic]
        nog_counts = [item["nog_online_item_count"] for item in traffic]
        result["tasks"][task_name] = {
            "query_count": len(task_traces),
            "node_count": _distribution([graph["num_graph_nodes"] for graph in graphs]),
            "edge_count": _distribution([graph["num_structural_edges"] for graph in graphs]),
            "node_text_item_count": {
                "total": sum(node_text_counts),
                "mean": mean(node_text_counts),
            },
            "edge_text_item_count": {
                "total": sum(edge_text_counts),
                "mean": mean(edge_text_counts),
            },
            "selected_K_ratio": mean(summary["selected_K_ratio"] for summary in summaries),
            "selected_V_ratio": mean(summary["selected_V_ratio"] for summary in summaries),
            "selected_KV_ratio": mean(summary["selected_KV_ratio"] for summary in summaries),
            "persistent_cache_bytes": {
                "total": sum(persistent),
                "mean": mean(persistent),
            },
            "loaded_cache_bytes": {
                "total": sum(loaded),
                "mean": mean(loaded),
            },
            "NOG_online_count": {
                "total": sum(nog_counts),
                "mean": mean(nog_counts),
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Trace JSON, trace_index.jsonl, or trace directory")
    parser.add_argument("--output", type=Path, help="Optional JSON summary output path")
    args = parser.parse_args()
    paths = _trace_paths(args.input)
    if not paths:
        raise RuntimeError(f"No query trace files found under {args.input}")
    traces = []
    for path in paths:
        with path.open() as handle:
            trace = json.load(handle)
        if trace.get("trace_format") != "gofa_query_trace":
            raise RuntimeError(f"{path}: unsupported trace_format={trace.get('trace_format')}")
        traces.append(trace)
    summary = summarize_traces(traces)
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.tmp")
        temporary.write_text(rendered + "\n")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
