#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

from large_workload_common import PROFILE_MODES, add_common_arguments, suite_root


TIME_FIELDS = (
    "query_wall_ms",
    "query_gpu_ms",
    "encoder_wall_ms",
    "decoder_wall_ms",
    "prefix_transformer_gpu_ms",
    "suffix_gnn_gpu_ms",
    "suffix_transformer_gpu_ms",
    "dense_fc_gpu_ms",
    "attention_gpu_ms",
)


def _stats(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Summarize a GOFA large-workload suite."))
    args = parser.parse_args()
    root = suite_root(args)
    with (root / "suite_manifest.json").open() as handle:
        suite = json.load(handle)
    result = {"summary_format": "gofa_large_workload_summary_v1", "profile": suite["profile"], "tasks": {}}
    for task in suite["tasks"]:
        task_summary = {"latency": {}, "graph": {}}
        trace_dir = Path(suite["paths"]["traces"]) / f"{task}_formal_v1"
        nodes, edges = [], []
        for trace_path in sorted(trace_dir.glob("query_*.json")):
            with trace_path.open() as handle:
                trace = json.load(handle)
            nodes.append(int(trace["num_graph_nodes"]))
            edges.append(int(trace["num_structural_edges"]))
        task_summary["graph"] = {"nodes": _stats(nodes), "edges": _stats(edges)}
        for mode in PROFILE_MODES:
            csv_path = Path(suite["paths"]["latency"]) / mode / f"{task}.csv"
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            task_summary["latency"][mode] = {
                field: _stats([float(row[field]) for row in rows]) for field in TIME_FIELDS
            }
        result["tasks"][task] = task_summary
    output = Path(suite["paths"]["summary"]) / "large_workload_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"summary={output}")


if __name__ == "__main__":
    main()
