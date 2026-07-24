#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TASKS = [
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
]

TASK_ORDER = {
    task: index
    for index, task in enumerate(TASKS)
}

SPLIT_ORDER = {
    "val": 0,
    "test": 1,
}

ID_FIELDS = [
    "task",
    "split",
    "trace_order",
    "query_index",
    "query_uid",
]

TIME_FIELDS = [
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
]

DERIVED_TIME_FIELDS = [
    "query_other_wall_ms",
    "encoder_non_suffix_wall_ms",
]

COUNTER_FIELDS = [
    "cache_hits",
    "cache_misses",
    "cache_skips",
    "quant_kv_attention_calls",
    "fallback_count",
]


def normalize_split(value):
    value = str(value).strip().lower()

    if value in {"val", "valid", "validation"}:
        return "val"

    if value in {"test", "testing"}:
        return "test"

    return value


def percentile(values, probability):
    values = sorted(float(value) for value in values)

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list.")

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"

    return value


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: format_value(row.get(field, ""))
                for field in fields
            })


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="H100 per-query result root.",
    )

    parser.add_argument(
        "--trace-root",
        required=True,
        type=Path,
        help="Canonical formal query trace root.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Hardware handoff output directory.",
    )

    args = parser.parse_args()

    source_root = args.root.resolve()
    trace_root = args.trace_root.resolve()
    output_root = args.output.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_rows = []

    required_fields = set(
        ID_FIELDS
        + ["rep"]
        + TIME_FIELDS
        + COUNTER_FIELDS
    )

    for task in TASKS:
        input_path = (
            source_root
            / task
            / "per_query.csv"
        )

        if not input_path.is_file():
            raise SystemExit(
                f"Missing input CSV: {input_path}"
            )

        with input_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            missing = (
                required_fields
                - set(reader.fieldnames or [])
            )

            if missing:
                raise SystemExit(
                    f"{input_path} is missing fields: "
                    f"{sorted(missing)}"
                )

            for line_number, source_row in enumerate(
                reader,
                start=2,
            ):
                row = {
                    "task": source_row["task"],
                    "split": normalize_split(
                        source_row["split"]
                    ),
                    "trace_order": int(
                        source_row["trace_order"]
                    ),
                    "query_index": int(
                        source_row["query_index"]
                    ),
                    "query_uid": source_row["query_uid"],
                    "rep": int(source_row["rep"]),
                }

                if row["task"] != task:
                    raise SystemExit(
                        f"Task mismatch in {input_path}:"
                        f"{line_number}"
                    )

                if row["split"] not in {
                    "val",
                    "test",
                }:
                    raise SystemExit(
                        f"Invalid split in {input_path}:"
                        f"{line_number}"
                    )

                for field in TIME_FIELDS:
                    value = float(source_row[field])

                    if value < 0:
                        raise SystemExit(
                            f"Negative {field} in "
                            f"{input_path}:{line_number}"
                        )

                    row[field] = value

                for field in COUNTER_FIELDS:
                    row[field] = int(
                        source_row[field]
                    )

                if row["cache_misses"] != 0:
                    raise SystemExit(
                        f"Cache miss in "
                        f"{input_path}:{line_number}"
                    )

                if row["fallback_count"] != 0:
                    raise SystemExit(
                        f"Fallback in "
                        f"{input_path}:{line_number}"
                    )

                row["query_other_wall_ms"] = (
                    row["query_wall_ms"]
                    - row["encoder_wall_ms"]
                    - row["decoder_wall_ms"]
                )

                row["encoder_non_suffix_wall_ms"] = (
                    row["encoder_wall_ms"]
                    - row["suffix_wall_ms"]
                )

                row["source_file"] = str(
                    input_path
                )

                raw_rows.append(row)

    if len(raw_rows) != 3000:
        raise SystemExit(
            f"Expected 3000 raw rows, "
            f"found {len(raw_rows)}"
        )

    groups = defaultdict(list)

    for row in raw_rows:
        key = tuple(
            row[field]
            for field in ID_FIELDS
        )

        groups[key].append(row)

    if len(groups) != 1000:
        raise SystemExit(
            f"Expected 1000 unique queries, "
            f"found {len(groups)}"
        )

    median_rows = []

    all_time_fields = (
        TIME_FIELDS
        + DERIVED_TIME_FIELDS
    )

    for key, rows in groups.items():
        rows = sorted(
            rows,
            key=lambda row: row["rep"],
        )

        reps = [
            row["rep"]
            for row in rows
        ]

        if reps != [0, 1, 2]:
            raise SystemExit(
                f"Query {key} has invalid reps: {reps}"
            )

        median_row = {
            field: value
            for field, value in zip(
                ID_FIELDS,
                key,
            )
        }

        median_row["repetitions"] = 3

        for field in COUNTER_FIELDS:
            values = {
                row[field]
                for row in rows
            }

            if len(values) != 1:
                raise SystemExit(
                    f"Counter {field} differs across "
                    f"repetitions for query {key}: "
                    f"{sorted(values)}"
                )

            median_row[field] = rows[0][field]

        for field in all_time_fields:
            values = [
                row[field]
                for row in rows
            ]

            # The base field itself is the three-run median.
            median_row[field] = statistics.median(
                values
            )

            median_row[
                f"{field}_mean"
            ] = statistics.mean(values)

            median_row[
                f"{field}_std"
            ] = statistics.stdev(values)

            median_row[
                f"{field}_min"
            ] = min(values)

            median_row[
                f"{field}_max"
            ] = max(values)

        median_rows.append(median_row)

    median_rows.sort(
        key=lambda row: (
            TASK_ORDER[row["task"]],
            SPLIT_ORDER[row["split"]],
            row["trace_order"],
        )
    )

    validation_rows = [
        row
        for row in median_rows
        if row["split"] == "val"
    ]

    test_rows = [
        row
        for row in median_rows
        if row["split"] == "test"
    ]

    if len(validation_rows) != 500:
        raise SystemExit(
            f"Expected 500 validation rows, "
            f"found {len(validation_rows)}"
        )

    if len(test_rows) != 500:
        raise SystemExit(
            f"Expected 500 test rows, "
            f"found {len(test_rows)}"
        )

    raw_fields = (
        ID_FIELDS
        + ["rep"]
        + TIME_FIELDS
        + DERIVED_TIME_FIELDS
        + COUNTER_FIELDS
        + ["source_file"]
    )

    median_fields = (
        ID_FIELDS
        + ["repetitions"]
        + COUNTER_FIELDS
    )

    for field in all_time_fields:
        median_fields.extend([
            field,
            f"{field}_mean",
            f"{field}_std",
            f"{field}_min",
            f"{field}_max",
        ])

    raw_path = (
        output_root
        / "h100_per_query_raw.csv"
    )

    completed_path = (
        output_root
        / "h100_profile_completed.csv"
    )

    validation_path = (
        output_root
        / "h100_validation_median.csv"
    )

    test_path = (
        output_root
        / "h100_test_median.csv"
    )

    write_csv(
        raw_path,
        raw_rows,
        raw_fields,
    )

    write_csv(
        completed_path,
        median_rows,
        median_fields,
    )

    write_csv(
        validation_path,
        validation_rows,
        median_fields,
    )

    write_csv(
        test_path,
        test_rows,
        median_fields,
    )

    summary_rows = []

    summary_metrics = [
        "query_wall_ms",
        "query_gpu_ms",
        "encoder_wall_ms",
        "decoder_wall_ms",
        "query_other_wall_ms",
        "cache_load_wall_ms",
        "online_nog_prefix_wall_ms",
        "cache_assembly_wall_ms",
        "suffix_wall_ms",
        "encoder_non_suffix_wall_ms",
        "suffix_gnn_gpu_ms",
        "suffix_transformer_gpu_ms",
    ]

    for task in TASKS:
        for split in ("val", "test"):
            rows = [
                row
                for row in median_rows
                if (
                    row["task"] == task
                    and row["split"] == split
                )
            ]

            if len(rows) != 100:
                raise SystemExit(
                    f"Expected 100 rows for "
                    f"{task}/{split}, found {len(rows)}"
                )

            summary = {
                "task": task,
                "split": split,
                "queries": len(rows),
            }

            for metric in summary_metrics:
                values = [
                    row[metric]
                    for row in rows
                ]

                summary[
                    f"{metric}_mean"
                ] = statistics.mean(values)

                summary[
                    f"{metric}_p50"
                ] = percentile(values, 0.50)

                summary[
                    f"{metric}_p95"
                ] = percentile(values, 0.95)

            summary_rows.append(summary)

    summary_fields = [
        "task",
        "split",
        "queries",
    ]

    for metric in summary_metrics:
        summary_fields.extend([
            f"{metric}_mean",
            f"{metric}_p50",
            f"{metric}_p95",
        ])

    summary_path = (
        output_root
        / "h100_task_summary.csv"
    )

    write_csv(
        summary_path,
        summary_rows,
        summary_fields,
    )

    trace_indexes = {}

    for task in TASKS:
        trace_path = (
            trace_root
            / f"{task}_formal_v1"
            / "trace_index.jsonl"
        )

        if not trace_path.is_file():
            raise SystemExit(
                f"Missing trace index: {trace_path}"
            )

        trace_indexes[task] = str(
            trace_path.resolve()
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": git_commit(),
        "baseline": (
            "H100 Quantized Software Baseline "
            "(M4K2V2 + W4A8 torch_int_mm "
            "+ INT-QK/PV)"
        ),
        "source_root": str(source_root),
        "trace_root": str(trace_root),
        "tasks": TASKS,
        "repetitions": [0, 1, 2],
        "raw_rows": 3000,
        "unique_queries": 1000,
        "validation_queries": 500,
        "test_queries": 500,
        "aggregation": (
            "Median across three repetitions "
            "for every task/split/trace_order/query_uid"
        ),
        "trace_indexes": trace_indexes,
        "join_key": [
            "task",
            "split",
            "trace_order",
            "query_uid",
        ],
        "timing_semantics": {
            "query_wall_ms": (
                "Full measured query wall-clock latency."
            ),
            "encoder_wall_ms": (
                "Measured H100 encoder wall-clock latency."
            ),
            "decoder_wall_ms": (
                "Measured H100 decoder wall-clock latency."
            ),
            "query_other_wall_ms": (
                "query_wall_ms - encoder_wall_ms "
                "- decoder_wall_ms."
            ),
            "suffix_wall_ms": (
                "Diagnostic suffix wall-clock timing."
            ),
            "suffix_gnn_gpu_ms": (
                "CUDA-event time for suffix GNN."
            ),
            "suffix_transformer_gpu_ms": (
                "CUDA-event time for suffix Transformer."
            ),
            "warning": (
                "Do not directly add wall-clock fields "
                "and CUDA-event fields."
            ),
        },
        "recommended_usage": {
            "calibration": (
                "Use h100_validation_median.csv only."
            ),
            "final_evaluation": (
                "Use h100_test_median.csv only after "
                "hardware parameters are frozen."
            ),
            "whole_encoder_target": (
                "Use encoder_wall_ms."
            ),
            "end_to_end_baseline": (
                "Use query_wall_ms."
            ),
            "proposed_end_to_end": (
                "simulated_encoder_ms "
                "+ decoder_wall_ms "
                "+ query_other_wall_ms"
            ),
        },
        "files": [
            raw_path.name,
            completed_path.name,
            validation_path.name,
            test_path.name,
            summary_path.name,
        ],
    }

    manifest_path = (
        output_root
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    output_files = [
        raw_path,
        completed_path,
        validation_path,
        test_path,
        summary_path,
        manifest_path,
    ]

    checksum_path = (
        output_root
        / "SHA256SUMS"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in output_files:
            handle.write(
                f"{sha256(path)}  {path.name}\n"
            )

    print("H100 hardware handoff prepared")
    print(f"raw rows        = {len(raw_rows)}")
    print(f"unique queries  = {len(median_rows)}")
    print(f"validation      = {len(validation_rows)}")
    print(f"test            = {len(test_rows)}")
    print(f"output          = {output_root}")

    for path in output_files:
        print(path)

    print(checksum_path)


if __name__ == "__main__":
    main()
