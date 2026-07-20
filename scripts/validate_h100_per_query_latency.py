#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict


REQUIRED_FIELDS = {
    "task",
    "split",
    "trace_order",
    "query_index",
    "query_uid",
    "rep",
    "query_wall_ms",
    "query_gpu_ms",
    "encoder_wall_ms",
    "decoder_wall_ms",
    "cache_load_wall_ms",
    "online_nog_prefix_wall_ms",
    "cache_assembly_wall_ms",
    "suffix_wall_ms",
    "suffix_gnn_gpu_ms",
    "suffix_transformer_gpu_ms",
    "cache_hits",
    "cache_misses",
    "cache_skips",
    "quant_kv_attention_calls",
    "fallback_count",
}

TIME_FIELDS = {
    "query_wall_ms",
    "query_gpu_ms",
    "encoder_wall_ms",
    "decoder_wall_ms",
    "cache_load_wall_ms",
    "online_nog_prefix_wall_ms",
    "cache_assembly_wall_ms",
    "suffix_wall_ms",
    "suffix_gnn_gpu_ms",
    "suffix_transformer_gpu_ms",
}


def normalize_split(value):
    value = str(value or "").strip().lower()
    if value in {"val", "valid", "validation"}:
        return "val"
    if value in {"test", "testing"}:
        return "test"
    return value


def resolve_task_path(path_template, task):
    path = str(path_template)
    path = path.replace("${TASK}", task).replace("$TASK", task)
    path = path.replace("{TASK}", task).replace("{task}", task)
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def load_expected_traces(path, task):
    if not os.path.isfile(path):
        raise RuntimeError(f"Trace index does not exist for task {task}: {path}")
    entries = []
    with open(path) as handle:
        for trace_order, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            index_entry = json.loads(line)
            trace_path = index_entry.get("trace_path")
            if not trace_path:
                raise RuntimeError(f"trace_order={trace_order} has no trace_path in {path}")
            if not os.path.isabs(trace_path):
                trace_path = os.path.join(os.path.dirname(path), trace_path)
            with open(trace_path) as trace_handle:
                trace = json.load(trace_handle)
            entry_task = trace.get("task_name") or index_entry.get("task")
            if entry_task != task:
                raise RuntimeError(
                    f"trace_order={trace_order} task mismatch: expected={task}, actual={entry_task}"
                )
            split = normalize_split(trace.get("split", index_entry.get("split")))
            query_index = trace.get("runtime_query_index", index_entry.get("query_index"))
            if query_index is None:
                raise RuntimeError(f"trace_order={trace_order} has no runtime_query_index")
            index_query_index = index_entry.get("query_index")
            if index_query_index is not None and int(index_query_index) != int(query_index):
                raise RuntimeError(
                    f"trace_order={trace_order} query_index mismatch: "
                    f"index={index_query_index}, trace={query_index}"
                )
            index_query_uid = index_entry.get("query_uid") or index_entry.get("query_id")
            trace_query_uid = trace.get("query_uid") or trace.get("query_id")
            if index_query_uid and trace_query_uid and str(index_query_uid) != str(trace_query_uid):
                raise RuntimeError(
                    f"trace_order={trace_order} query UID mismatch: "
                    f"index={index_query_uid}, trace={trace_query_uid}"
                )
            query_uid = index_query_uid or trace_query_uid
            if not query_uid:
                raise RuntimeError(f"trace_order={trace_order} has no query UID")
            entries.append({
                "task": entry_task,
                "split": split,
                "trace_order": int(trace_order),
                "query_index": int(query_index),
                "query_uid": str(query_uid),
            })
    return entries


def load_csv_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"Latency CSV {path} is missing fields: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                row["_source"] = path
                row["_line"] = line_number
                rows.append(row)
    if not rows:
        raise RuntimeError("No latency rows were found.")
    return rows


def validate_rows(rows, trace_index_template, expected_per_split):
    grouped = defaultdict(list)
    task_input_order = defaultdict(list)
    for row in rows:
        source = f"{row['_source']}:{row['_line']}"
        task = row["task"]
        split = normalize_split(row["split"])
        if split not in {"val", "test"}:
            raise RuntimeError(f"Unexpected split {row['split']!r} at {source}")
        try:
            rep = int(row["rep"])
            row["trace_order"] = int(row["trace_order"])
            row["query_index"] = int(row["query_index"])
            cache_misses = int(row["cache_misses"])
            fallback_count = int(row["fallback_count"])
        except ValueError as exc:
            raise RuntimeError(f"Invalid integer field at {source}: {exc}") from exc
        if cache_misses != 0:
            raise RuntimeError(f"cache_misses must be zero at {source}, got {cache_misses}")
        if fallback_count != 0:
            raise RuntimeError(f"fallback_count must be zero at {source}, got {fallback_count}")
        for field in TIME_FIELDS:
            try:
                value = float(row[field])
            except ValueError as exc:
                raise RuntimeError(f"Invalid {field} at {source}: {row[field]!r}") from exc
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"{field} must be finite and nonnegative at {source}, got {value}")
        row["split"] = split
        row["rep"] = rep
        grouped[(task, rep)].append(row)
        task_input_order[(task, rep)].append(split)

    summaries = []
    tasks = sorted({task for task, _ in grouped})
    expected_total = 2 * int(expected_per_split)
    for task in tasks:
        expected_path = resolve_task_path(trace_index_template, task)
        expected = load_expected_traces(expected_path, task)
        if len(expected) != expected_total:
            raise RuntimeError(
                f"Task {task} trace index must contain {expected_total} entries, got {len(expected)}"
            )
        expected_by_order = {entry["trace_order"]: entry for entry in expected}
        for split in ("val", "test"):
            count = sum(1 for entry in expected if entry["split"] == split)
            if count != expected_per_split:
                raise RuntimeError(
                    f"Task {task} trace index split {split} must contain {expected_per_split} entries, got {count}"
                )

        reps = sorted(rep for grouped_task, rep in grouped if grouped_task == task)
        if reps != list(range(max(reps) + 1)):
            raise RuntimeError(f"Task {task} rep values are not contiguous: {reps}")
        for rep in reps:
            current = grouped[(task, rep)]
            if len(current) != expected_total:
                raise RuntimeError(
                    f"Task {task} rep {rep} must contain {expected_total} rows, got {len(current)}"
                )
            for split in ("val", "test"):
                split_rows = [row for row in current if row["split"] == split]
                if len(split_rows) != expected_per_split:
                    raise RuntimeError(
                        f"Task {task} rep {rep} split {split} must contain "
                        f"{expected_per_split} rows, got {len(split_rows)}"
                    )
                indices = sorted(row["query_index"] for row in split_rows)
                if indices != list(range(expected_per_split)):
                    raise RuntimeError(
                        f"Task {task} rep {rep} split {split} query indices are incomplete: {indices}"
                    )

            trace_orders = [row["trace_order"] for row in current]
            if len(set(trace_orders)) != len(trace_orders):
                raise RuntimeError(f"Task {task} rep {rep} has duplicate trace_order values")
            if set(trace_orders) != set(expected_by_order):
                raise RuntimeError(
                    f"Task {task} rep {rep} trace_order set does not match {expected_path}"
                )
            seen_test = False
            for split in task_input_order[(task, rep)]:
                seen_test = seen_test or split == "test"
                if seen_test and split == "val":
                    raise RuntimeError(
                        f"Task {task} rep {rep} has validation rows after test rows"
                    )
            for row in current:
                expected_entry = expected_by_order[row["trace_order"]]
                for key in ("task", "split", "query_index", "query_uid"):
                    if row[key] != expected_entry[key]:
                        raise RuntimeError(
                            f"Task {task} rep {rep} trace_order={row['trace_order']} {key} mismatch: "
                            f"csv={row[key]!r}, trace={expected_entry[key]!r}"
                        )
            summaries.append((task, rep, len(current)))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Validate canonical GOFA H100 per-query latency CSV files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more latency CSV files.")
    parser.add_argument(
        "--trace-index-path",
        required=True,
        help="Trace index path or template containing ${TASK}, $TASK, {TASK}, or {task}.",
    )
    parser.add_argument("--expected-per-split", type=int, default=100)
    args = parser.parse_args()

    summaries = validate_rows(
        load_csv_rows(args.input),
        args.trace_index_path,
        args.expected_per_split,
    )
    for task, rep, count in summaries:
        print(
            f"PASS task={task} rep={rep} rows={count} "
            f"validation={args.expected_per_split} test={args.expected_per_split}"
        )


if __name__ == "__main__":
    main()
