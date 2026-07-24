#!/usr/bin/env python3
"""
Extract GOFA baseline latency breakdown from profiler logs.

Expected directory structure:
    ROOT/
      cora_node/baseline/rep1/run_stdout.log
      cora_node/baseline/rep2/run_stdout.log
      cora_node/baseline/rep3/run_stdout.log
      ...

The script generates:
    gofa_baseline_breakdown_runs.csv
    gofa_baseline_breakdown_summary.csv

Key behavior:
1. Reads the last "GOFA stage timing summary" block for the requested report.
2. Correctly parses the aggregate "encoder_prefix_transformer" item.
3. Parses numbered suffix Transformer layers such as
   "encoder_transformer_layer_26".
4. Avoids double-counting when aggregate and per-layer timings coexist.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_TASKS = [
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
]


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]

    if not valid:
        return None

    if len(valid) == 1:
        return 0.0

    return statistics.stdev(valid)


def remove_appended_summary(text: str) -> str:
    """
    Some run scripts append grep output after the original log.
    Remove that appended duplicate section before parsing.
    """
    return text.split("\n===== TIMING LINES =====", 1)[0]


def extract_final_block(
    text: str,
    expected_report: int,
) -> tuple[float, str]:
    """
    Return:
        model_total_seconds, text_of_the_selected_summary_block
    """
    text = remove_appended_summary(text)

    header_pattern = re.compile(
        r"GOFA stage timing summary:\s*"
        r"report=(\d+),\s*"
        r"model_total=([0-9]+(?:\.[0-9]+)?)s,[^\n]*"
    )

    matches = [
        match
        for match in header_pattern.finditer(text)
        if int(match.group(1)) == expected_report
    ]

    if not matches:
        available_reports = sorted(
            {
                int(match.group(1))
                for match in header_pattern.finditer(text)
            }
        )
        raise ValueError(
            f"Missing stage timing summary report={expected_report}. "
            f"Available reports: {available_reports or 'none'}"
        )

    selected = matches[-1]
    block_start = selected.end()

    next_header = header_pattern.search(text, block_start)
    block_end = next_header.start() if next_header else len(text)

    return float(selected.group(2)), text[block_start:block_end]


def parse_single_value(
    block: str,
    field: str,
) -> Optional[float]:
    """
    Parse a line such as:
        encoder_prefix_transformer: 77.0075s (59.42%)
    """
    match = re.search(
        rf"^\s*{re.escape(field)}:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)s(?:\s|\(|$)",
        block,
        re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def parse_numbered_layers(
    block: str,
    prefix: str,
) -> dict[int, float]:
    """
    Parse lines such as:
        encoder_transformer_layer_26: 2.9934s (2.31%)
    """
    matches = re.findall(
        rf"^\s*{re.escape(prefix)}(\d+):\s*"
        rf"([0-9]+(?:\.[0-9]+)?)s(?:\s|\(|$)",
        block,
        re.MULTILINE,
    )

    return {
        int(layer): float(seconds)
        for layer, seconds in matches
    }


def sum_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) if values else None


def to_ms_per_query(
    seconds: Optional[float],
    report_count: int,
) -> Optional[float]:
    if seconds is None:
        return None

    return seconds * 1000.0 / report_count


def safe_percentage(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None

    return numerator / denominator * 100.0


def near_zero(value: float, tolerance_s: float = 0.02) -> float:
    """
    Clamp tiny residuals caused by rounded profiler output.
    """
    if abs(value) < tolerance_s:
        return 0.0
    return value


def validate_nonnegative(
    name: str,
    value: Optional[float],
    tolerance_s: float = 0.02,
) -> Optional[float]:
    if value is None:
        return None

    value = near_zero(value, tolerance_s)

    if value < 0:
        raise ValueError(
            f"{name} is negative ({value:.6f}s). "
            "The profiler fields may overlap or be incomplete."
        )

    return value


def parse_log(
    log_path: Path,
    task: str,
    rep: str,
    report_count: int,
    transformer_split_layer: int,
) -> dict[str, object]:
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    model_total_s, block = extract_final_block(
        text,
        expected_report=report_count,
    )

    decoder_s = parse_single_value(block, "decoder")
    if decoder_s is None:
        raise ValueError("Missing decoder timing")

    encoder_s = model_total_s - decoder_s
    encoder_s = validate_nonnegative("encoder", encoder_s)

    encoder_norm_s = (
        parse_single_value(block, "encoder_norm")
        or 0.0
    )

    gnn_layers = parse_numbered_layers(
        block,
        "encoder_gnn_layer_",
    )
    gnn_s = sum_or_none(gnn_layers.values())

    transformer_layers = parse_numbered_layers(
        block,
        "encoder_transformer_layer_",
    )

    per_layer_prefix_s = sum_or_none(
        seconds
        for layer, seconds in transformer_layers.items()
        if layer < transformer_split_layer
    )
    per_layer_suffix_s = sum_or_none(
        seconds
        for layer, seconds in transformer_layers.items()
        if layer >= transformer_split_layer
    )
    per_layer_full_s = sum_or_none(transformer_layers.values())

    aggregate_prefix_s = parse_single_value(
        block,
        "encoder_prefix_transformer",
    )
    aggregate_suffix_s = parse_single_value(
        block,
        "encoder_suffix_transformer",
    )
    aggregate_full_s = parse_single_value(
        block,
        "encoder_full_transformer",
    )

    # Prefer explicit aggregate values for each region. Fall back to
    # numbered-layer sums only when the aggregate value is unavailable.
    prefix_transformer_s = (
        aggregate_prefix_s
        if aggregate_prefix_s is not None
        else per_layer_prefix_s
    )

    suffix_transformer_s = (
        aggregate_suffix_s
        if aggregate_suffix_s is not None
        else per_layer_suffix_s
    )

    # Prefer an explicitly reported full-transformer total.
    if aggregate_full_s is not None:
        full_transformer_s = aggregate_full_s

    # Current GOFA logs typically report prefix as one aggregate item and
    # suffix as individual layers 26--31.
    elif (
        prefix_transformer_s is not None
        and suffix_transformer_s is not None
    ):
        full_transformer_s = (
            prefix_transformer_s
            + suffix_transformer_s
        )

    # If only numbered layers exist and cover the complete Transformer,
    # use their total.
    elif per_layer_full_s is not None:
        full_transformer_s = per_layer_full_s

    else:
        full_transformer_s = (
            prefix_transformer_s
            if prefix_transformer_s is not None
            else suffix_transformer_s
        )

    known_encoder_s = encoder_norm_s

    if full_transformer_s is not None:
        known_encoder_s += full_transformer_s

    if gnn_s is not None:
        known_encoder_s += gnn_s

    other_encoder_s = encoder_s - known_encoder_s
    other_encoder_s = validate_nonnegative(
        "other_encoder",
        other_encoder_s,
    )

    row: dict[str, object] = {
        "task": task,
        "rep": rep,
        "report_count": report_count,
        "model_ms_per_query": to_ms_per_query(
            model_total_s,
            report_count,
        ),
        "encoder_ms_per_query": to_ms_per_query(
            encoder_s,
            report_count,
        ),
        "decoder_ms_per_query": to_ms_per_query(
            decoder_s,
            report_count,
        ),
        "full_transformer_ms_per_query": to_ms_per_query(
            full_transformer_s,
            report_count,
        ),
        "prefix_transformer_ms_per_query": to_ms_per_query(
            prefix_transformer_s,
            report_count,
        ),
        "suffix_transformer_ms_per_query": to_ms_per_query(
            suffix_transformer_s,
            report_count,
        ),
        "gnn_ms_per_query": to_ms_per_query(
            gnn_s,
            report_count,
        ),
        "encoder_norm_ms_per_query": to_ms_per_query(
            encoder_norm_s,
            report_count,
        ),
        "other_encoder_ms_per_query": to_ms_per_query(
            other_encoder_s,
            report_count,
        ),
        "encoder_pct_of_model": safe_percentage(
            encoder_s,
            model_total_s,
        ),
        "decoder_pct_of_model": safe_percentage(
            decoder_s,
            model_total_s,
        ),
        "full_transformer_pct_of_encoder": safe_percentage(
            full_transformer_s,
            encoder_s,
        ),
        "prefix_transformer_pct_of_encoder": safe_percentage(
            prefix_transformer_s,
            encoder_s,
        ),
        "suffix_transformer_pct_of_encoder": safe_percentage(
            suffix_transformer_s,
            encoder_s,
        ),
        "gnn_pct_of_encoder": safe_percentage(
            gnn_s,
            encoder_s,
        ),
        "encoder_norm_pct_of_encoder": safe_percentage(
            encoder_norm_s,
            encoder_s,
        ),
        "other_encoder_pct_of_encoder": safe_percentage(
            other_encoder_s,
            encoder_s,
        ),
        "log": str(log_path),
    }

    for layer, seconds in sorted(transformer_layers.items()):
        row[
            f"transformer_layer_{layer}_ms_per_query"
        ] = to_ms_per_query(seconds, report_count)

    for layer, seconds in sorted(gnn_layers.items()):
        row[
            f"gnn_layer_{layer}_ms_per_query"
        ] = to_ms_per_query(seconds, report_count)

    return row


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    run_rows: list[dict[str, object]],
    tasks: list[str],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in run_rows:
        groups[str(row["task"])].append(row)

    base_metrics = sorted(
        {
            key
            for row in run_rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {"report_count"}
            and math.isfinite(float(value))
        }
    )

    summary_rows: list[dict[str, object]] = []

    for task in tasks:
        rows = groups[task]
        output: dict[str, object] = {
            "task": task,
            "repetitions": len(rows),
        }

        for metric in base_metrics:
            values = [
                float(row[metric])
                if metric in row
                and isinstance(row[metric], (int, float))
                else None
                for row in rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = std(values)

        summary_rows.append(output)

    return summary_rows


def format_value(
    row: dict[str, object],
    name: str,
    digits: int = 2,
) -> str:
    value = row.get(name)

    if value is None:
        return "NA"

    return f"{float(value):.{digits}f}"


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print()
    print(
        f"{'task':<15}"
        f"{'model':>11}"
        f"{'encoder':>11}"
        f"{'prefix':>11}"
        f"{'suffix-T':>11}"
        f"{'GNN':>11}"
        f"{'other':>11}"
        f"{'decoder':>11}"
        f"{'T/enc%':>10}"
    )

    for row in summary_rows:
        print(
            f"{str(row['task']):<15}"
            f"{format_value(row, 'model_ms_per_query_mean'):>11}"
            f"{format_value(row, 'encoder_ms_per_query_mean'):>11}"
            f"{format_value(row, 'prefix_transformer_ms_per_query_mean'):>11}"
            f"{format_value(row, 'suffix_transformer_ms_per_query_mean'):>11}"
            f"{format_value(row, 'gnn_ms_per_query_mean'):>11}"
            f"{format_value(row, 'other_encoder_ms_per_query_mean'):>11}"
            f"{format_value(row, 'decoder_ms_per_query_mean'):>11}"
            f"{format_value(row, 'full_transformer_pct_of_encoder_mean'):>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GOFA baseline latency breakdown from profiler logs."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing TASK/baseline/repN/run_stdout.log",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names to process",
    )
    parser.add_argument(
        "--reps",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Repetition indices, for example: --reps 1 2 3",
    )
    parser.add_argument(
        "--report",
        type=int,
        default=200,
        help="Profiler report count used for the final summary",
    )
    parser.add_argument(
        "--transformer-split-layer",
        type=int,
        default=26,
        help=(
            "First suffix Transformer layer. "
            "Default: 26, so layers 0--25 are prefix."
        ),
    )
    parser.add_argument(
        "--log-name",
        default="run_stdout.log",
        help="Log filename inside each repetition directory",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    tasks = list(args.tasks)

    run_rows: list[dict[str, object]] = []

    for task in tasks:
        for rep_index in args.reps:
            rep = f"rep{rep_index}"
            log_path = (
                root
                / task
                / "baseline"
                / rep
                / args.log_name
            )

            if not log_path.is_file():
                raise SystemExit(f"Missing log: {log_path}")

            try:
                row = parse_log(
                    log_path=log_path,
                    task=task,
                    rep=rep,
                    report_count=args.report,
                    transformer_split_layer=args.transformer_split_layer,
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to parse {log_path}: {exc}"
                ) from exc

            run_rows.append(row)

    runs_csv = root / "gofa_baseline_breakdown_runs.csv"
    write_csv(runs_csv, run_rows)

    summary_rows = build_summary_rows(
        run_rows=run_rows,
        tasks=tasks,
    )

    summary_csv = root / "gofa_baseline_breakdown_summary.csv"
    write_csv(summary_csv, summary_rows)

    print_summary(summary_rows)

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()