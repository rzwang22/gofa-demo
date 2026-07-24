#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path


MODES = (
    "nocache_bf16",
    "cache_bf16",
    "cache_w8a8_m4k2v2",
)

INCLUDE_PATHS = (
    "suite_manifest.json",
    "configs",
    "plans",
    "manifests",
    "traces",
    "latency",
    "summary",
)


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a validated subset GOFA workload suite."
    )
    parser.add_argument(
        "--suite-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    root = args.suite_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    manifest_path = root / "suite_manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing suite manifest: {manifest_path}"
        )

    suite = json.load(manifest_path.open())

    tasks = list(suite["tasks"])
    samples = int(suite["profile"]["samples_per_split"])
    reps = int(suite["reps"])

    expected_queries_per_task = 2 * samples
    expected_rows_per_csv = expected_queries_per_task * reps
    expected_trace_count = expected_queries_per_task * len(tasks)
    expected_csv_count = len(MODES) * len(tasks)
    expected_raw_rows = expected_rows_per_csv * expected_csv_count
    expected_medians = expected_queries_per_task * expected_csv_count

    paths = suite["paths"]

    errors = []
    trace_count = 0
    csv_count = 0
    raw_rows = 0

    for relative in INCLUDE_PATHS:
        path = root / relative
        if not path.exists():
            errors.append(f"missing required path: {path}")

    for task in tasks:
        task_manifest = Path(paths["manifests"]) / f"{task}.json"

        if not task_manifest.is_file():
            errors.append(
                f"missing task manifest: {task_manifest}"
            )

        trace_dir = (
            Path(paths["traces"])
            / f"{task}_formal_v1"
        )
        index_path = trace_dir / "trace_index.jsonl"

        if not index_path.is_file():
            errors.append(
                f"missing trace index: {index_path}"
            )
        else:
            index_count = sum(
                1
                for line in index_path.open()
                if line.strip()
            )

            if index_count != expected_queries_per_task:
                errors.append(
                    f"{task}: trace index count "
                    f"{index_count}, expected "
                    f"{expected_queries_per_task}"
                )

        task_trace_count = len(
            list(trace_dir.glob("query_*.json"))
        )

        if task_trace_count != expected_queries_per_task:
            errors.append(
                f"{task}: trace JSON count "
                f"{task_trace_count}, expected "
                f"{expected_queries_per_task}"
            )

        trace_count += task_trace_count

        for mode in MODES:
            csv_path = (
                Path(paths["latency"])
                / mode
                / f"{task}.csv"
            )

            if not csv_path.is_file():
                errors.append(
                    f"missing latency CSV: {csv_path}"
                )
                continue

            rows = csv_row_count(csv_path)

            if rows != expected_rows_per_csv:
                errors.append(
                    f"{task}/{mode}: rows={rows}, "
                    f"expected={expected_rows_per_csv}"
                )

            csv_count += 1
            raw_rows += rows

    summary_json = (
        Path(paths["summary"])
        / "large_workload_summary.json"
    )
    median_csv = (
        Path(paths["summary"])
        / "per_query_median.csv"
    )

    if not summary_json.is_file():
        errors.append(
            f"missing summary: {summary_json}"
        )

    if not median_csv.is_file():
        errors.append(
            f"missing median CSV: {median_csv}"
        )
        median_rows = 0
    else:
        median_rows = csv_row_count(median_csv)

        if median_rows != expected_medians:
            errors.append(
                f"median rows={median_rows}, "
                f"expected={expected_medians}"
            )

    count_checks = {
        "trace_count": (
            trace_count,
            expected_trace_count,
        ),
        "latency_csv_count": (
            csv_count,
            expected_csv_count,
        ),
        "raw_gpu_row_count": (
            raw_rows,
            expected_raw_rows,
        ),
        "per_query_median_row_count": (
            median_rows,
            expected_medians,
        ),
    }

    for field, (actual, expected) in count_checks.items():
        if actual != expected:
            errors.append(
                f"{field}={actual}, expected={expected}"
            )

    if errors:
        raise RuntimeError(
            "Subset handoff validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checksums = {}

    with tarfile.open(output, "w:gz") as archive:
        for relative in INCLUDE_PATHS:
            path = root / relative

            archive.add(
                path,
                arcname=f"{root.name}/{relative}",
            )

            if path.is_file():
                checksums[relative] = sha256_file(path)
            else:
                for file_path in sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                ):
                    key = str(
                        file_path.relative_to(root)
                    )
                    checksums[key] = sha256_file(
                        file_path
                    )

    checksum_path = Path(
        str(output) + ".sha256.json"
    )

    payload = {
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "suite_root": str(root),
        "profile": suite["profile"],
        "tasks": tasks,
        "reps": reps,
        "validated_counts": {
            key: actual
            for key, (actual, _expected)
            in count_checks.items()
        },
        "files": checksums,
    }

    with checksum_path.open("w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"handoff={output}")
    print(f"checksums={checksum_path}")
    print(
        "validated_counts="
        + json.dumps(
            payload["validated_counts"],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
