#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .large_workload_common import PROFILE_MODES, add_common_arguments, load_locked_suite
except ImportError:
    from large_workload_common import PROFILE_MODES, add_common_arguments, load_locked_suite


TIME_FIELDS = (
    "query_wall_ms",
    "query_gpu_ms",
    "encoder_wall_ms",
    "decoder_wall_ms",
    "cache_load_wall_ms",
    "online_nog_prefix_wall_ms",
    "cache_assembly_wall_ms",
    "suffix_wall_ms",
    "prefix_transformer_gpu_ms",
    "suffix_gnn_gpu_ms",
    "suffix_transformer_gpu_ms",
    "dense_fc_gpu_ms",
    "attention_gpu_ms",
    "norm_residual_other_gpu_ms",
    "quant_kv_attention_gpu_ms",
    "kv_prepare_gpu_ms",
    "int_qk_gpu_ms",
    "softmax_prob_quant_gpu_ms",
    "int_pv_gpu_ms",
    "gnn_score_gpu_ms",
    "gnn_message_gpu_ms",
    "gnn_update_gpu_ms",
    "gnn_other_gpu_ms",
)

MEDIAN_IDENTITY_FIELDS = (
    "task",
    "profile_mode",
    "split",
    "query_uid",
    "trace_order",
    "query_index",
    "workload_profile",
    "graph_signature",
)


def _percentile(values, probability):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def aggregate_per_query_medians(rows, expected_reps):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["profile_mode"], row["task"], row["split"], row["query_uid"])
        grouped[key].append(row)
    expected_rep_set = set(range(int(expected_reps)))
    medians = []
    for key, query_rows in grouped.items():
        reps = [int(row["rep"]) for row in query_rows]
        if len(reps) != len(set(reps)) or set(reps) != expected_rep_set:
            raise RuntimeError(
                f"Incomplete or duplicate repetitions for mode/task/split/query_uid={key}: "
                f"actual={sorted(reps)}, expected={sorted(expected_rep_set)}"
            )
        first = query_rows[0]
        for field in MEDIAN_IDENTITY_FIELDS:
            if any(row[field] != first[field] for row in query_rows[1:]):
                raise RuntimeError(f"Per-query identity field {field} differs across repetitions for {key}")
        output = {field: first[field] for field in MEDIAN_IDENTITY_FIELDS}
        output["rep_count"] = int(expected_reps)
        for field in TIME_FIELDS:
            values = [float(row[field]) for row in query_rows]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise RuntimeError(f"Invalid latency values for {key} field={field}: {values}")
            output[field] = statistics.median(values)
        medians.append(output)
    medians.sort(key=lambda row: (row["profile_mode"], row["task"], int(row["trace_order"])))
    return medians


def summarize_latency_medians(median_rows, raw_counts):
    grouped = defaultdict(list)
    for row in median_rows:
        grouped[(row["task"], row["profile_mode"])].append(row)
    result = defaultdict(dict)
    for (task, mode), rows in sorted(grouped.items()):
        result[task][mode] = {
            "raw_row_count": int(raw_counts[(task, mode)]),
            "per_query_count": len(rows),
            "latency_ms": {
                field: _stats([row[field] for row in rows])
                for field in TIME_FIELDS
            },
        }
    return {task: modes for task, modes in result.items()}


def _write_median_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MEDIAN_IDENTITY_FIELDS) + ["rep_count"] + list(TIME_FIELDS)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Summarize a GOFA large-workload suite."))
    args = parser.parse_args()
    _manifest_path, suite = load_locked_suite(args)
    all_rows = []
    raw_counts = {}
    graph_summary = {}
    for task in suite["tasks"]:
        trace_dir = Path(suite["paths"]["traces"]) / f"{task}_formal_v1"
        nodes, edges = [], []
        for trace_path in sorted(trace_dir.glob("query_*.json")):
            with trace_path.open() as handle:
                trace = json.load(handle)
            nodes.append(int(trace["num_graph_nodes"]))
            edges.append(int(trace["num_structural_edges"]))
        graph_summary[task] = {"nodes": _stats(nodes), "edges": _stats(edges)}
        for mode in PROFILE_MODES:
            csv_path = Path(suite["paths"]["latency"]) / mode / f"{task}.csv"
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected_raw_rows = 2 * int(suite["profile"]["samples_per_split"]) * int(suite["reps"])
            if len(rows) != expected_raw_rows:
                raise RuntimeError(
                    f"{task}/{mode}: expected {expected_raw_rows} raw rows, got {len(rows)} in {csv_path}"
                )
            raw_counts[(task, mode)] = len(rows)
            all_rows.extend(rows)

    medians = aggregate_per_query_medians(all_rows, suite["reps"])
    latency_summary = summarize_latency_medians(medians, raw_counts)
    result = {
        "summary_format": "gofa_large_workload_summary_v2",
        "profile": suite["profile"],
        "reps": suite["reps"],
        "aggregation": "median_across_reps_then_task_distribution_across_unique_queries",
        "raw_rows_preserved_in_source_csv": True,
        "tasks": {
            task: {
                "graph": graph_summary[task],
                "latency": latency_summary.get(task, {}),
            }
            for task in suite["tasks"]
        },
    }
    summary_dir = Path(suite["paths"]["summary"])
    median_path = summary_dir / "per_query_median.csv"
    _write_median_csv(median_path, medians)
    output = summary_dir / "large_workload_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"per_query_median={median_path} rows={len(medians)}")
    print(f"summary={output}")


if __name__ == "__main__":
    main()
