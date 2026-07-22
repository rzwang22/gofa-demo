#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from .large_workload_common import PROFILE_MODES, add_common_arguments, load_locked_suite
    from .validate_gofa_query_trace import validate_trace
    from .validate_h100_per_query_latency import load_csv_rows, validate_rows
except ImportError:
    from large_workload_common import PROFILE_MODES, add_common_arguments, load_locked_suite
    from validate_gofa_query_trace import validate_trace
    from validate_h100_per_query_latency import load_csv_rows, validate_rows


def _load_suite(args):
    return load_locked_suite(args)


def validate_suite(args):
    manifest_path, suite = _load_suite(args)
    expected_per_split = int(suite["profile"]["samples_per_split"])
    expected_total = 2 * expected_per_split
    trace_orders = {}
    for task in suite["tasks"]:
        trace_dir = Path(suite["paths"]["traces"]) / f"{task}_formal_v1"
        index_path = trace_dir / "trace_index.jsonl"
        if not index_path.is_file():
            raise RuntimeError(f"Missing formal trace index: {index_path}")
        with index_path.open() as handle:
            index = [json.loads(line) for line in handle if line.strip()]
        if len(index) != expected_total:
            raise RuntimeError(f"{task}: expected {expected_total} traces, got {len(index)}")
        order = []
        split_counts = defaultdict(int)
        for entry in index:
            trace_path = trace_dir / entry["trace_path"]
            with trace_path.open() as handle:
                trace = json.load(handle)
            errors = validate_trace(trace, str(trace_path))
            if errors:
                raise RuntimeError(f"{trace_path}: " + "; ".join(errors))
            if trace["workload_profile"] != suite["profile"]:
                raise RuntimeError(f"{trace_path}: workload_profile mismatch")
            split_counts[trace["split"]] += 1
            order.append((
                trace["split"],
                int(trace["runtime_query_index"]),
                trace["query_uid"],
                trace["graph_signature"],
            ))
        if split_counts != {"val": expected_per_split, "test": expected_per_split}:
            raise RuntimeError(f"{task}: invalid trace split counts {dict(split_counts)}")
        trace_orders[task] = order

    for mode in PROFILE_MODES:
        for task in suite["tasks"]:
            csv_path = Path(suite["paths"]["latency"]) / mode / f"{task}.csv"
            if not csv_path.is_file():
                raise RuntimeError(f"Missing GPU latency CSV: {csv_path}")
            rows = load_csv_rows([str(csv_path)])
            validate_rows(
                rows,
                str(Path(suite["paths"]["traces"]) / "${TASK}_formal_v1" / "trace_index.jsonl"),
                expected_per_split,
            )
            by_rep = defaultdict(list)
            for row in rows:
                if row["profile_mode"] != mode:
                    raise RuntimeError(f"{csv_path}: expected profile_mode={mode}, got {row['profile_mode']}")
                by_rep[int(row["rep"])].append((
                    row["split"],
                    int(row["query_index"]),
                    row["query_uid"],
                    row["graph_signature"],
                ))
            if sorted(by_rep) != list(range(int(suite["reps"]))):
                raise RuntimeError(f"{csv_path}: repetition set mismatch {sorted(by_rep)}")
            for rep, order in by_rep.items():
                if order != trace_orders[task]:
                    raise RuntimeError(f"{csv_path}: rep={rep} query order/signature differs from formal trace")
    return manifest_path, suite


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Validate a GOFA large-workload suite."))
    args = parser.parse_args()
    manifest_path, suite = validate_suite(args)
    print(
        f"validated suite={manifest_path}, tasks={len(suite['tasks'])}, "
        f"modes={len(PROFILE_MODES)}, reps={suite['reps']}"
    )


if __name__ == "__main__":
    main()
