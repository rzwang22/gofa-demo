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

COMPONENTS = [
    "kv_to_device",
    "input_norm",
    "qkv_proj",
    "rope_cache_repeat",
    "attn_score_softmax_value",
    "o_proj",
    "post_attn_norm",
    "mlp",
]


def mean(values):
    return statistics.mean(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def remove_appended_shell_summary(text):
    # The runner appends selected grep output after the real model log.
    return text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]


def extract_final_stage_block(text):
    text = remove_appended_shell_summary(text)

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
            "Missing final stage timing summary report=200"
        )

    final_match = matches[-1]
    start = final_match.end()

    next_match = header_pattern.search(text, start)

    end = (
        next_match.start()
        if next_match
        else len(text)
    )

    return text[start:end]


def extract_transformer_layers(block):
    values = {}

    for layer in range(26, 32):
        match = re.search(
            rf"^\s*encoder_transformer_layer_{layer}:"
            rf"\s*([0-9.]+)s",
            block,
            re.MULTILINE,
        )

        if match is None:
            raise ValueError(
                f"Missing encoder_transformer_layer_{layer}"
            )

        values[layer] = float(match.group(1))

    return values


def extract_component_breakdown(block):
    matches = re.findall(
        r"^\s*memory_kv_transformer_breakdown_total:"
        r"[^\n]+",
        block,
        re.MULTILINE,
    )

    if not matches:
        raise ValueError(
            "Missing memory_kv_transformer_breakdown_total"
        )

    line = matches[-1]
    output = {}

    for component in COMPONENTS:
        match = re.search(
            rf"\b{re.escape(component)}=([0-9.]+)s",
            line,
        )

        if match is None:
            raise ValueError(
                f"Missing Transformer component: {component}"
            )

        output[component] = float(match.group(1))

    return output


def parse_log(log_path, task, rep):
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    block = extract_final_stage_block(text)

    layer_times = extract_transformer_layers(block)
    components = extract_component_breakdown(block)

    transformer_total_s = sum(layer_times.values())
    profiled_total_s = sum(components.values())
    other_s = transformer_total_s - profiled_total_s

    # Small negative differences can arise from profiler rounding.
    if other_s < -0.02:
        raise ValueError(
            f"Invalid negative residual: {other_s:.6f}s"
        )

    other_s = max(other_s, 0.0)

    row = {
        "task": task,
        "rep": rep,

        "transformer_total_ms_per_query":
            transformer_total_s * 1000.0 / 200.0,

        "profiled_components_ms_per_query":
            profiled_total_s * 1000.0 / 200.0,

        "other_transformer_ms_per_query":
            other_s * 1000.0 / 200.0,

        "other_transformer_pct":
            other_s / transformer_total_s * 100.0,

        "log": str(log_path),
    }

    for component, seconds in components.items():
        ms = seconds * 1000.0 / 200.0

        row[f"{component}_ms_per_query"] = ms

        row[f"{component}_pct_of_transformer"] = (
            seconds / transformer_total_s * 100.0
        )

        row[f"{component}_pct_of_profiled_components"] = (
            seconds / profiled_total_s * 100.0
        )

    # Paper-friendly grouped categories.
    grouped = {
        "kv_handling":
            components["kv_to_device"]
            + components["rope_cache_repeat"],

        "normalization":
            components["input_norm"]
            + components["post_attn_norm"],

        "linear_projection":
            components["qkv_proj"]
            + components["o_proj"],

        "attention_core":
            components["attn_score_softmax_value"],

        "mlp":
            components["mlp"],

        "other":
            other_s,
    }

    for name, seconds in grouped.items():
        row[f"group_{name}_ms_per_query"] = (
            seconds * 1000.0 / 200.0
        )

        row[f"group_{name}_pct_of_transformer"] = (
            seconds / transformer_total_s * 100.0
        )

    for layer, seconds in layer_times.items():
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

            run_rows.append(row)

    runs_csv = (
        root
        / "fp_cache_transformer_breakdown_runs.csv"
    )

    run_fields = list(run_rows[0].keys())

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
        key
        for key in run_rows[0]
        if (
            key.endswith("_ms_per_query")
            or key.endswith("_pct_of_transformer")
            or key.endswith(
                "_pct_of_profiled_components"
            )
            or key == "other_transformer_pct"
        )
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
                float(row[metric])
                for row in rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = std(values)

        summary_rows.append(output)

    summary_csv = (
        root
        / "fp_cache_transformer_breakdown_summary.csv"
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
        f"{'trans':>10}"
        f"{'KV':>10}"
        f"{'norm':>10}"
        f"{'proj':>10}"
        f"{'attn':>10}"
        f"{'MLP':>10}"
        f"{'other':>10}"
    )

    for row in summary_rows:
        print(
            f"{row['task']:<15}"
            f"{row['transformer_total_ms_per_query_mean']:>10.2f}"
            f"{row['group_kv_handling_ms_per_query_mean']:>10.2f}"
            f"{row['group_normalization_ms_per_query_mean']:>10.2f}"
            f"{row['group_linear_projection_ms_per_query_mean']:>10.2f}"
            f"{row['group_attention_core_ms_per_query_mean']:>10.2f}"
            f"{row['group_mlp_ms_per_query_mean']:>10.2f}"
            f"{row['group_other_ms_per_query_mean']:>10.2f}"
        )

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()
