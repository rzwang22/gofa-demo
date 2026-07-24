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
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def std(values):
    values = [v for v in values if v is not None]

    if not values:
        return None

    if len(values) == 1:
        return 0.0

    return statistics.stdev(values)


def remove_appended_summary(text):
    # run script 最后可能又 grep 打印了一次 timing lines。
    return text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]


def extract_final_block(text):
    text = remove_appended_summary(text)

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
            "Missing final stage summary report=200"
        )

    match = matches[-1]
    start = match.end()

    next_match = header_pattern.search(text, start)
    end = next_match.start() if next_match else len(text)

    return (
        float(match.group(2)),
        text[start:end],
    )


def parse_single_value(block, field):
    match = re.search(
        rf"^\s*{re.escape(field)}:"
        rf"\s*([0-9.]+)s",
        block,
        re.MULTILINE,
    )

    return float(match.group(1)) if match else None


def parse_numbered_layers(block, prefix):
    matches = re.findall(
        rf"^\s*{re.escape(prefix)}(\d+):"
        rf"\s*([0-9.]+)s",
        block,
        re.MULTILINE,
    )

    return {
        int(layer): float(seconds)
        for layer, seconds in matches
    }


def to_ms_per_query(seconds):
    if seconds is None:
        return None

    return seconds * 1000.0 / 200.0


def parse_log(log_path, task, rep):
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    model_total_s, block = extract_final_block(text)

    decoder_s = parse_single_value(
        block,
        "decoder",
    )

    encoder_norm_s = parse_single_value(
        block,
        "encoder_norm",
    ) or 0.0

    transformer_layers = parse_numbered_layers(
        block,
        "encoder_transformer_layer_",
    )

    gnn_layers = parse_numbered_layers(
        block,
        "encoder_gnn_layer_",
    )

    encoder_s = (
        model_total_s - decoder_s
        if decoder_s is not None
        else None
    )

    prefix_transformer_s = None
    suffix_transformer_s = None
    full_transformer_s = None

    if transformer_layers:
        prefix_transformer_s = sum(
            seconds
            for layer, seconds
            in transformer_layers.items()
            if layer < 26
        )

        suffix_transformer_s = sum(
            seconds
            for layer, seconds
            in transformer_layers.items()
            if layer >= 26
        )

        full_transformer_s = sum(
            transformer_layers.values()
        )

    else:
        # 某些 profiler 可能只报告 aggregate full transformer。
        full_transformer_s = parse_single_value(
            block,
            "encoder_full_transformer",
        )

    gnn_s = (
        sum(gnn_layers.values())
        if gnn_layers
        else None
    )

    known_encoder_s = encoder_norm_s

    for value in [
        full_transformer_s,
        gnn_s,
    ]:
        if value is not None:
            known_encoder_s += value

    other_encoder_s = (
        encoder_s - known_encoder_s
        if encoder_s is not None
        else None
    )

    # 小的负值可能来自四舍五入。
    if (
        other_encoder_s is not None
        and other_encoder_s < 0
        and abs(other_encoder_s) < 0.02
    ):
        other_encoder_s = 0.0

    row = {
        "task": task,
        "rep": rep,

        "model_ms_per_query":
            to_ms_per_query(model_total_s),

        "encoder_ms_per_query":
            to_ms_per_query(encoder_s),

        "decoder_ms_per_query":
            to_ms_per_query(decoder_s),

        "full_transformer_ms_per_query":
            to_ms_per_query(full_transformer_s),

        "prefix_transformer_ms_per_query":
            to_ms_per_query(prefix_transformer_s),

        "suffix_transformer_ms_per_query":
            to_ms_per_query(suffix_transformer_s),

        "gnn_ms_per_query":
            to_ms_per_query(gnn_s),

        "encoder_norm_ms_per_query":
            to_ms_per_query(encoder_norm_s),

        "other_encoder_ms_per_query":
            to_ms_per_query(other_encoder_s),

        "log": str(log_path),
    }

    if encoder_s and full_transformer_s is not None:
        row["full_transformer_pct_of_encoder"] = (
            full_transformer_s / encoder_s * 100.0
        )

    if encoder_s and prefix_transformer_s is not None:
        row["prefix_transformer_pct_of_encoder"] = (
            prefix_transformer_s / encoder_s * 100.0
        )

    if encoder_s and suffix_transformer_s is not None:
        row["suffix_transformer_pct_of_encoder"] = (
            suffix_transformer_s / encoder_s * 100.0
        )

    if encoder_s and gnn_s is not None:
        row["gnn_pct_of_encoder"] = (
            gnn_s / encoder_s * 100.0
        )

    if encoder_s and other_encoder_s is not None:
        row["other_encoder_pct"] = (
            other_encoder_s / encoder_s * 100.0
        )

    for layer, seconds in transformer_layers.items():
        row[
            f"transformer_layer_{layer}_ms_per_query"
        ] = to_ms_per_query(seconds)

    for layer, seconds in gnn_layers.items():
        row[
            f"gnn_layer_{layer}_ms_per_query"
        ] = to_ms_per_query(seconds)

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
                / "baseline"
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

            run_rows.append(row)

    runs_csv = (
        root
        / "gofa_baseline_breakdown_runs.csv"
    )

    run_fields = sorted({
        key
        for row in run_rows
        for key in row
    })

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

    summary_metrics = [
        "model_ms_per_query",
        "encoder_ms_per_query",
        "decoder_ms_per_query",
        "full_transformer_ms_per_query",
        "prefix_transformer_ms_per_query",
        "suffix_transformer_ms_per_query",
        "gnn_ms_per_query",
        "encoder_norm_ms_per_query",
        "other_encoder_ms_per_query",
        "full_transformer_pct_of_encoder",
        "prefix_transformer_pct_of_encoder",
        "suffix_transformer_pct_of_encoder",
        "gnn_pct_of_encoder",
        "other_encoder_pct",
    ]

    summary_rows = []

    for task in TASKS:
        rows = groups[task]

        output = {
            "task": task,
            "repetitions": len(rows),
        }

        for metric in summary_metrics:
            values = [
                row.get(metric)
                for row in rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = std(values)

        summary_rows.append(output)

    summary_csv = (
        root
        / "gofa_baseline_breakdown_summary.csv"
    )

    summary_fields = list(summary_rows[0].keys())

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
        f"{'encoder':>11}"
        f"{'prefix':>11}"
        f"{'suffix-T':>11}"
        f"{'GNN':>11}"
        f"{'other':>11}"
        f"{'decoder':>11}"
    )

    for row in summary_rows:
        def fmt(name):
            value = row.get(name)
            return (
                f"{value:.2f}"
                if value is not None
                else "NA"
            )

        print(
            f"{row['task']:<15}"
            f"{fmt('encoder_ms_per_query_mean'):>11}"
            f"{fmt('prefix_transformer_ms_per_query_mean'):>11}"
            f"{fmt('suffix_transformer_ms_per_query_mean'):>11}"
            f"{fmt('gnn_ms_per_query_mean'):>11}"
            f"{fmt('other_encoder_ms_per_query_mean'):>11}"
            f"{fmt('decoder_ms_per_query_mean'):>11}"
        )

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()
