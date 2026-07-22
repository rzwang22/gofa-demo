#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

try:
    from .large_workload_common import DEFAULT_TASKS, PROFILE_MODES, add_common_arguments
    from .validate_large_workload_suite import validate_suite
except ImportError:
    from large_workload_common import DEFAULT_TASKS, PROFILE_MODES, add_common_arguments
    from validate_large_workload_suite import validate_suite


INCLUDE_PATHS = ("suite_manifest.json", "configs", "plans", "manifests", "traces", "latency", "summary")


def _csv_row_count(path):
    with Path(path).open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def validate_handoff_artifacts(suite):
    root = Path(suite["paths"]["root"])
    missing = [str(root / relative) for relative in INCLUDE_PATHS if not (root / relative).exists()]
    samples = int(suite["profile"]["samples_per_split"])
    reps = int(suite["reps"])
    expected_queries_per_task = 2 * samples
    expected_rows_per_csv = expected_queries_per_task * reps
    trace_count = 0
    raw_row_count = 0
    csv_count = 0
    for task in suite["tasks"]:
        manifest_path = Path(suite["paths"]["manifests"]) / f"{task}.json"
        if not manifest_path.is_file():
            missing.append(str(manifest_path))
        trace_dir = Path(suite["paths"]["traces"]) / f"{task}_formal_v1"
        index_path = trace_dir / "trace_index.jsonl"
        if not index_path.is_file():
            missing.append(str(index_path))
        task_trace_count = len(list(trace_dir.glob("query_*.json"))) if trace_dir.is_dir() else 0
        if task_trace_count != expected_queries_per_task:
            raise RuntimeError(
                f"{task}: expected {expected_queries_per_task} trace JSON files, got {task_trace_count}"
            )
        trace_count += task_trace_count
        for mode in PROFILE_MODES:
            csv_path = Path(suite["paths"]["latency"]) / mode / f"{task}.csv"
            if not csv_path.is_file():
                missing.append(str(csv_path))
                continue
            rows = _csv_row_count(csv_path)
            if rows != expected_rows_per_csv:
                raise RuntimeError(
                    f"{task}/{mode}: expected {expected_rows_per_csv} raw rows, got {rows}"
                )
            raw_row_count += rows
            csv_count += 1

    summary_json = Path(suite["paths"]["summary"]) / "large_workload_summary.json"
    median_csv = Path(suite["paths"]["summary"]) / "per_query_median.csv"
    for path in (summary_json, median_csv):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise RuntimeError("Simulator handoff is incomplete; missing: " + ", ".join(missing))

    with summary_json.open() as handle:
        summary_payload = json.load(handle)
    if summary_payload.get("summary_format") != "gofa_large_workload_summary_v2":
        raise RuntimeError(
            f"Simulator handoff has an unsupported summary format: {summary_payload.get('summary_format')}"
        )
    with median_csv.open(newline="") as handle:
        median_reader = csv.DictReader(handle)
        median_fields = set(median_reader.fieldnames or [])
    required_median_fields = {
        "task",
        "profile_mode",
        "split",
        "query_uid",
        "logical_memory_loaded_bytes",
        "logical_key_loaded_bytes",
        "logical_value_loaded_bytes",
        "quant_kv_attention_calls",
        "int_gemm_call_count",
        "cache_hits",
        "cache_skips",
    }
    if not required_median_fields <= median_fields:
        raise RuntimeError(
            "Simulator handoff median CSV is missing fields: "
            f"{sorted(required_median_fields - median_fields)}"
        )

    expected_csv_count = len(PROFILE_MODES) * len(suite["tasks"])
    expected_trace_count = expected_queries_per_task * len(suite["tasks"])
    expected_raw_row_count = expected_rows_per_csv * expected_csv_count
    expected_median_count = expected_queries_per_task * expected_csv_count
    canonical_expectations = {
        "tasks": tuple(DEFAULT_TASKS),
        "samples_per_split": 100,
        "reps": 3,
        "latency_csv_count": 15,
        "trace_count": 1000,
        "raw_gpu_row_count": 9000,
        "per_query_median_row_count": 3000,
    }
    actual_identity = {
        "tasks": tuple(suite["tasks"]),
        "samples_per_split": samples,
        "reps": reps,
        "latency_csv_count": expected_csv_count,
        "trace_count": expected_trace_count,
        "raw_gpu_row_count": expected_raw_row_count,
        "per_query_median_row_count": expected_median_count,
    }
    identity_mismatches = [
        f"{key}={actual_identity[key]!r}, expected={expected!r}"
        for key, expected in canonical_expectations.items()
        if actual_identity[key] != expected
    ]
    if identity_mismatches:
        raise RuntimeError(
            "Simulator handoff requires the canonical five-task, 100-query-per-split, three-repetition suite: "
            + "; ".join(identity_mismatches)
        )
    median_count = _csv_row_count(median_csv)
    checks = {
        "latency_csv_count": (csv_count, expected_csv_count),
        "trace_count": (trace_count, expected_trace_count),
        "raw_gpu_row_count": (raw_row_count, expected_raw_row_count),
        "per_query_median_row_count": (median_count, expected_median_count),
    }
    mismatches = [f"{key}={actual}, expected={expected}" for key, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError("Simulator handoff count validation failed: " + "; ".join(mismatches))
    return {key: actual for key, (actual, _expected) in checks.items()}


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Package GOFA simulator handoff artifacts."))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    _manifest_path, suite = validate_suite(args)
    counts = validate_handoff_artifacts(suite)
    root = Path(suite["paths"]["root"])
    output = Path(args.output).expanduser().resolve() if args.output else root / f"{args.profile_name}_simulator_handoff.tar.gz"
    checksums = {}
    with tarfile.open(output, "w:gz") as archive:
        for relative in INCLUDE_PATHS:
            path = root / relative
            if not path.exists():
                raise RuntimeError(f"Simulator handoff required path disappeared during packaging: {path}")
            archive.add(path, arcname=f"{args.profile_name}/{relative}")
            if path.is_file():
                checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                    key = str(file_path.relative_to(root))
                    checksums[key] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256.json")
    with checksum_path.open("w") as handle:
        json.dump({"archive": str(output), "files": checksums}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"handoff={output}")
    print(f"checksums={checksum_path}")
    print("validated_counts=" + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
