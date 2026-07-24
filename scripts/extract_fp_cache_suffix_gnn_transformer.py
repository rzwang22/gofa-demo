#!/usr/bin/env python3

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


TASKS = [
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
]


def mean(values):
    return statistics.mean(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def extract_final_stage_block(text: str) -> str:
    # The shell runner appends grep output after this marker. Remove it to
    # avoid parsing duplicated profiler lines.
    text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    header_pattern = re.compile(
        r"GOFA stage timing summary: "
        r"report=(\d+), "
        r"model_total=([0-9.]+)s,[^\n]*"
    )

    matches = [
        match
        for match in header_pattern.finditer(text)
        if int(match.group(1)) == 200
    ]

    if not matches:
        raise ValueError(
            "Missing final stage timing summary: report=200"
        )

    final_match = matches[-1]
    block_start = final_match.end()

    next_header = header_pattern.search(
        text,
        block_start,
    )

    block_end = (
        next_header.start()
        if next_header
        else len(text)
    )

    return text[block_start:block_end]


def extract_layer_times(block: str, prefix: str, expected_layers):
    values = {}

    for layer in expected_layers:
        pattern = re.compile(
            rf"^\s*{re.escape(prefix)}{layer}:"
            rf"\s*([0-9.]+)s",
            re.MULTILINE,
        )

        match = pattern.search(block)

        if match is None:
            raise ValueError(
                f"Missing profiler field: {prefix}{layer}"
            )

        values[layer] = float(match.group(1))

    return values


def extract_suffix_total(text: str) -> float:
    text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    lines = re.findall(
        r"GOFA encoder cache timing:[^\n]+",
        text,
    )

    if not lines:
        raise ValueError(
            "Missing GOFA encoder cache timing"
        )

    # Select the final call_total=200 line.
    candidates = [
        line
        for line in lines
        if "call_total=200" in line
    ]

    if not candidates:
        raise ValueError(
            "Missing final cache timing call_total=200"
        )

    line = candidates[-1]

    match = re.search(
        r"\bcum_suffix=([0-9.]+)s",
        line,
    )

    if match is None:
        raise ValueError(
            "Missing cum_suffix in final cache timing"
        )

    return float(match.group(1))


def parse_log(log_path: Path, task: str, rep: str):
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    block = extract_final_stage_block(text)

    gnn_layers = extract_layer_times(
        block,
        "encoder_gnn_layer_",
        range(6),
    )

    transformer_layers = extract_layer_times(
        block,
        "encoder_transformer_layer_",
        range(26, 32),
    )

    suffix_total_s = extract_suffix_total(text)

    gnn_s = sum(gnn_layers.values())
    transformer_s = sum(transformer_layers.values())

    explicit_compute_s = (
        gnn_s + transformer_s
    )

    residual_s = (
        suffix_total_s - explicit_compute_s
    )

    if residual_s < -0.01:
        raise ValueError(
            f"Negative suffix residual for {task}/{rep}: "
            f"{residual_s:.6f}s"
        )

    row = {
        "task": task,
        "rep": rep,

        "suffix_total_ms_per_query":
            suffix_total_s * 1000.0 / 200.0,

        "gnn_ms_per_query":
            gnn_s * 1000.0 / 200.0,

        "transformer_ms_per_query":
            transformer_s * 1000.0 / 200.0,

        "other_suffix_ms_per_query":
            residual_s * 1000.0 / 200.0,

        "gnn_pct_of_explicit_compute":
            gnn_s / explicit_compute_s * 100.0,

        "transformer_pct_of_explicit_compute":
            transformer_s / explicit_compute_s * 100.0,

        "gnn_pct_of_suffix":
            gnn_s / suffix_total_s * 100.0,

        "transformer_pct_of_suffix":
            transformer_s / suffix_total_s * 100.0,

        "other_suffix_pct":
            residual_s / suffix_total_s * 100.0,
    }

    for layer, seconds in gnn_layers.items():
        row[
            f"gnn_layer_{layer}_ms_per_query"
        ] = seconds * 1000.0 / 200.0

    for layer, seconds in transformer_layers.items():
        row[
            f"transformer_layer_{layer}_ms_per_query"
        ] = seconds * 1000.0 / 200.0

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )

    args = parser.parse_args()
    root = args.root.resolve()

    run_rows = []

    for task in TASKS:
        for rep_index in [1, 2, 3]:
            rep = f"rep{rep_index}"

            log_path = (
                root
                / task
                / "fp_cache"
                / rep
                / "run_stdout.log"
            )

            if not log_path.is_file():
                raise SystemExit(
                    f"Missing log: {log_path}"
                )

            try:
                row = parse_log(
                    log_path,
                    task,
                    rep,
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to parse {log_path}: {exc}"
                ) from exc

            row["log"] = str(log_path)
            run_rows.append(row)

    run_fields = list(run_rows[0].keys())

    runs_csv = (
        root
        / "fp_cache_suffix_gnn_transformer_runs.csv"
    )

    with runs_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=run_fields,
        )

        writer.writeheader()
        writer.writerows(run_rows)

    groups = defaultdict(list)

    for row in run_rows:
        groups[row["task"]].append(row)

    summary_rows = []

    summary_metrics = [
        "suffix_total_ms_per_query",
        "gnn_ms_per_query",
        "transformer_ms_per_query",
        "other_suffix_ms_per_query",
        "gnn_pct_of_explicit_compute",
        "transformer_pct_of_explicit_compute",
        "gnn_pct_of_suffix",
        "transformer_pct_of_suffix",
        "other_suffix_pct",
    ]

    for task in TASKS:
        rows = groups[task]

        output = {
            "task": task,
            "repetitions": len(rows),
        }

        for metric in summary_metrics:
            values = [
                float(row[metric])
                for row in rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = std(values)

        summary_rows.append(output)

    summary_fields = list(
        summary_rows[0].keys()
    )

    summary_csv = (
        root
        / "fp_cache_suffix_gnn_transformer_summary.csv"
    )

    with summary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(
        f"{'task':<15}"
        f"{'suffix(ms)':>12}"
        f"{'gnn(ms)':>12}"
        f"{'trans(ms)':>12}"
        f"{'gnn %':>10}"
        f"{'trans %':>10}"
        f"{'other %':>10}"
    )

    for row in summary_rows:
        print(
            f"{row['task']:<15}"
            f"{row['suffix_total_ms_per_query_mean']:>12.2f}"
            f"{row['gnn_ms_per_query_mean']:>12.2f}"
            f"{row['transformer_ms_per_query_mean']:>12.2f}"
            f"{row['gnn_pct_of_suffix_mean']:>10.2f}"
            f"{row['transformer_pct_of_suffix_mean']:>10.2f}"
            f"{row['other_suffix_pct_mean']:>10.2f}"
        )

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()
