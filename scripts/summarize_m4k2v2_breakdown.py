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

TRANSFORMER_COMPONENTS = [
    "kv_to_device",
    "input_norm",
    "qkv_proj",
    "rope_cache_repeat",
    "attn_score_softmax_value",
    "o_proj",
    "post_attn_norm",
    "mlp",
]


def to_ms_per_query(seconds, queries):
    if seconds is None or not queries:
        return None

    return seconds * 1000.0 / queries


def mean(values):
    values = [
        value for value in values
        if value is not None
    ]

    return (
        statistics.mean(values)
        if values else None
    )


def std(values):
    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None

    if len(values) == 1:
        return 0.0

    return statistics.stdev(values)


def extract_number(line, field, suffix=""):
    match = re.search(
        rf"\b{re.escape(field)}=([0-9.]+){re.escape(suffix)}",
        line,
    )

    return (
        float(match.group(1))
        if match else None
    )


def extract_integer(line, field):
    match = re.search(
        rf"\b{re.escape(field)}=([0-9]+)",
        line,
    )

    return (
        int(match.group(1))
        if match else None
    )


def extract_final_stage(text):
    pattern = re.compile(
        r"GOFA stage timing summary: "
        r"report=(\d+), "
        r"model_total=([0-9.]+)s,[^\n]*"
    )

    matches = [
        match
        for match in pattern.finditer(text)
        if int(match.group(1)) == 200
    ]

    if not matches:
        raise ValueError(
            "Missing final stage timing summary report=200"
        )

    match = matches[-1]

    start = match.end()

    next_match = pattern.search(text, start)

    end = (
        next_match.start()
        if next_match
        else len(text)
    )

    block = text[start:end]

    return {
        "queries": int(match.group(1)),
        "model_total_s": float(match.group(2)),
        "block": block,
    }


def extract_stage_field(block, field):
    match = re.search(
        rf"^\s*{re.escape(field)}:"
        rf"\s*([0-9.]+)s",
        block,
        re.MULTILINE,
    )

    return (
        float(match.group(1))
        if match else None
    )


def extract_numbered_layers(block, prefix):
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


def extract_final_cache_timing(text):
    lines = re.findall(
        r"GOFA encoder cache timing:[^\n]+",
        text,
    )

    candidates = [
        line for line in lines
        if "call_total=200" in line
    ]

    if not candidates:
        raise ValueError(
            "Missing final cache timing call_total=200"
        )

    line = candidates[-1]

    return {
        "cache_total_s":
            extract_number(line, "cum_total", "s"),

        "cache_load_s":
            extract_number(line, "cum_load", "s"),

        "online_prefix_s":
            extract_number(
                line,
                "cum_miss_compute",
                "s",
            ),

        "cache_assembly_s":
            extract_number(
                line,
                "cum_assemble",
                "s",
            ),

        "suffix_total_s":
            extract_number(
                line,
                "cum_suffix",
                "s",
            ),
    }


def extract_final_cache_status(text):
    lines = re.findall(
        r"GOFA encoder memory/text-KV cache:[^\n]+",
        text,
    )

    if not lines:
        return {}

    line = lines[-1]

    return {
        "total_hits":
            extract_integer(line, "total_hits"),

        "total_misses":
            extract_integer(line, "total_misses"),

        "total_skips":
            extract_integer(line, "total_skips"),
    }


def extract_transformer_components(text):
    lines = re.findall(
        r"memory_kv_transformer_breakdown_total:[^\n]+",
        text,
    )

    if not lines:
        raise ValueError(
            "Missing memory_kv_transformer_breakdown_total"
        )

    line = lines[-1]

    output = {}

    for component in TRANSFORMER_COMPONENTS:
        value = extract_number(
            line,
            component,
            "s",
        )

        if value is None:
            raise ValueError(
                f"Missing Transformer component: {component}"
            )

        output[f"{component}_s"] = value

    return output


def parse_log(log_path, task, rep):
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    stage = extract_final_stage(text)
    block = stage["block"]

    queries = stage["queries"]
    model_total_s = stage["model_total_s"]

    decoder_s = extract_stage_field(
        block,
        "decoder",
    )

    if decoder_s is None:
        raise ValueError(
            "Missing decoder timing"
        )

    encoder_s = model_total_s - decoder_s

    gnn_layers = extract_numbered_layers(
        block,
        "encoder_gnn_layer_",
    )

    transformer_layers = extract_numbered_layers(
        block,
        "encoder_transformer_layer_",
    )

    expected_gnn_layers = set(range(6))
    expected_transformer_layers = set(range(26, 32))

    if not expected_gnn_layers.issubset(gnn_layers):
        raise ValueError(
            f"Missing GNN layers; found {sorted(gnn_layers)}"
        )

    if not expected_transformer_layers.issubset(
        transformer_layers
    ):
        raise ValueError(
            "Missing suffix Transformer layers; "
            f"found {sorted(transformer_layers)}"
        )

    suffix_gnn_s = sum(
        gnn_layers[layer]
        for layer in range(6)
    )

    suffix_transformer_s = sum(
        transformer_layers[layer]
        for layer in range(26, 32)
    )

    cache = extract_final_cache_timing(text)
    status = extract_final_cache_status(text)
    components = extract_transformer_components(text)

    suffix_total_s = cache["suffix_total_s"]

    other_suffix_s = (
        suffix_total_s
        - suffix_gnn_s
        - suffix_transformer_s
    )

    profiled_transformer_s = sum(
        components.values()
    )

    other_transformer_s = (
        suffix_transformer_s
        - profiled_transformer_s
    )

    # Only suppress tiny negative residuals caused by printed-value rounding.
    if -0.02 < other_suffix_s < 0:
        other_suffix_s = 0.0

    if -0.02 < other_transformer_s < 0:
        other_transformer_s = 0.0

    row = {
        "task": task,
        "rep": rep,
        "queries": queries,
        "log": str(log_path),

        "model_ms_per_query":
            to_ms_per_query(
                model_total_s,
                queries,
            ),

        "encoder_ms_per_query":
            to_ms_per_query(
                encoder_s,
                queries,
            ),

        "decoder_ms_per_query":
            to_ms_per_query(
                decoder_s,
                queries,
            ),

        "cache_path_total_ms_per_query":
            to_ms_per_query(
                cache["cache_total_s"],
                queries,
            ),

        "cache_load_ms_per_query":
            to_ms_per_query(
                cache["cache_load_s"],
                queries,
            ),

        "online_nog_prefix_ms_per_query":
            to_ms_per_query(
                cache["online_prefix_s"],
                queries,
            ),

        "cache_assembly_ms_per_query":
            to_ms_per_query(
                cache["cache_assembly_s"],
                queries,
            ),

        "suffix_total_ms_per_query":
            to_ms_per_query(
                suffix_total_s,
                queries,
            ),

        "suffix_gnn_ms_per_query":
            to_ms_per_query(
                suffix_gnn_s,
                queries,
            ),

        "suffix_transformer_ms_per_query":
            to_ms_per_query(
                suffix_transformer_s,
                queries,
            ),

        "other_suffix_ms_per_query":
            to_ms_per_query(
                other_suffix_s,
                queries,
            ),

        "total_hits": status.get("total_hits"),
        "total_misses": status.get("total_misses"),
        "total_skips": status.get("total_skips"),
    }

    # Percentages of the diagnostic cache path.
    cache_total_s = cache["cache_total_s"]

    for name, seconds in [
        ("cache_load", cache["cache_load_s"]),
        ("online_nog_prefix", cache["online_prefix_s"]),
        ("cache_assembly", cache["cache_assembly_s"]),
        ("suffix_total", suffix_total_s),
    ]:
        row[f"{name}_pct_of_cache_path"] = (
            seconds / cache_total_s * 100.0
        )

    # Percentages inside suffix.
    for name, seconds in [
        ("suffix_gnn", suffix_gnn_s),
        ("suffix_transformer", suffix_transformer_s),
        ("other_suffix", other_suffix_s),
    ]:
        row[f"{name}_pct_of_suffix"] = (
            seconds / suffix_total_s * 100.0
        )

    # Fine-grained Transformer components.
    for component in TRANSFORMER_COMPONENTS:
        seconds = components[f"{component}_s"]

        row[f"{component}_ms_per_query"] = (
            to_ms_per_query(
                seconds,
                queries,
            )
        )

        row[f"{component}_pct_of_transformer"] = (
            seconds
            / suffix_transformer_s
            * 100.0
        )

    row["other_transformer_ms_per_query"] = (
        to_ms_per_query(
            other_transformer_s,
            queries,
        )
    )

    row["other_transformer_pct_of_transformer"] = (
        other_transformer_s
        / suffix_transformer_s
        * 100.0
    )

    # Paper-friendly grouping.
    grouped_components = {
        "kv_handling_s":
            components["kv_to_device_s"]
            + components["rope_cache_repeat_s"],

        "normalization_s":
            components["input_norm_s"]
            + components["post_attn_norm_s"],

        "linear_projection_s":
            components["qkv_proj_s"]
            + components["o_proj_s"],

        "attention_core_s":
            components[
                "attn_score_softmax_value_s"
            ],

        "mlp_group_s":
            components["mlp_s"],

        "other_transformer_group_s":
            other_transformer_s,
    }

    for name, seconds in grouped_components.items():
        clean_name = name[:-2]

        row[f"group_{clean_name}_ms_per_query"] = (
            to_ms_per_query(
                seconds,
                queries,
            )
        )

        row[f"group_{clean_name}_pct_of_transformer"] = (
            seconds
            / suffix_transformer_s
            * 100.0
        )

    return row


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, metric_names):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["task"]].append(row)

    outputs = []

    for task in TASKS:
        task_rows = grouped.get(task, [])

        if not task_rows:
            continue

        output = {
            "task": task,
            "repetitions": len(task_rows),
            "max_total_misses": max(
                row.get("total_misses") or 0
                for row in task_rows
            ),
        }

        for metric in metric_names:
            values = [
                row.get(metric)
                for row in task_rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = std(values)

        outputs.append(output)

    return outputs


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
        task_root = root / task / "breakdown"

        rep_dirs = sorted(
            task_root.glob("rep*")
        )

        if not rep_dirs:
            raise SystemExit(
                f"No breakdown runs found for {task}"
            )

        for rep_dir in rep_dirs:
            log_path = (
                rep_dir / "run_stdout.log"
            )

            if not log_path.is_file():
                continue

            try:
                run_rows.append(
                    parse_log(
                        log_path,
                        task,
                        rep_dir.name,
                    )
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to parse {log_path}: {exc}"
                ) from exc

    runs_csv = (
        root
        / "m4k2v2_breakdown_runs.csv"
    )

    write_csv(
        runs_csv,
        run_rows,
        sorted({
            key
            for row in run_rows
            for key in row
        }),
    )

    encoder_metrics = [
        "model_ms_per_query",
        "encoder_ms_per_query",
        "decoder_ms_per_query",
        "cache_path_total_ms_per_query",
        "cache_load_ms_per_query",
        "online_nog_prefix_ms_per_query",
        "cache_assembly_ms_per_query",
        "suffix_total_ms_per_query",
        "suffix_gnn_ms_per_query",
        "suffix_transformer_ms_per_query",
        "other_suffix_ms_per_query",
        "cache_load_pct_of_cache_path",
        "online_nog_prefix_pct_of_cache_path",
        "cache_assembly_pct_of_cache_path",
        "suffix_total_pct_of_cache_path",
        "suffix_gnn_pct_of_suffix",
        "suffix_transformer_pct_of_suffix",
        "other_suffix_pct_of_suffix",
    ]

    transformer_metrics = []

    for component in TRANSFORMER_COMPONENTS:
        transformer_metrics.extend([
            f"{component}_ms_per_query",
            f"{component}_pct_of_transformer",
        ])

    transformer_metrics.extend([
        "other_transformer_ms_per_query",
        "other_transformer_pct_of_transformer",

        "group_kv_handling_ms_per_query",
        "group_kv_handling_pct_of_transformer",

        "group_normalization_ms_per_query",
        "group_normalization_pct_of_transformer",

        "group_linear_projection_ms_per_query",
        "group_linear_projection_pct_of_transformer",

        "group_attention_core_ms_per_query",
        "group_attention_core_pct_of_transformer",

        "group_mlp_group_ms_per_query",
        "group_mlp_group_pct_of_transformer",

        "group_other_transformer_group_ms_per_query",
        "group_other_transformer_group_pct_of_transformer",
    ])

    encoder_summary = aggregate(
        run_rows,
        encoder_metrics,
    )

    transformer_summary = aggregate(
        run_rows,
        transformer_metrics,
    )

    encoder_csv = (
        root
        / "m4k2v2_encoder_breakdown_summary.csv"
    )

    transformer_csv = (
        root
        / "m4k2v2_transformer_breakdown_summary.csv"
    )

    write_csv(
        encoder_csv,
        encoder_summary,
    )

    write_csv(
        transformer_csv,
        transformer_summary,
    )

    print()
    print("Encoder/cache/suffix breakdown")
    print(
        f"{'task':<15}"
        f"{'cache':>10}"
        f"{'load':>10}"
        f"{'online':>10}"
        f"{'assembly':>10}"
        f"{'suffix':>10}"
        f"{'GNN':>10}"
        f"{'Trans':>10}"
    )

    for row in encoder_summary:
        print(
            f"{row['task']:<15}"
            f"{row['cache_path_total_ms_per_query_mean']:>10.2f}"
            f"{row['cache_load_ms_per_query_mean']:>10.2f}"
            f"{row['online_nog_prefix_ms_per_query_mean']:>10.2f}"
            f"{row['cache_assembly_ms_per_query_mean']:>10.2f}"
            f"{row['suffix_total_ms_per_query_mean']:>10.2f}"
            f"{row['suffix_gnn_ms_per_query_mean']:>10.2f}"
            f"{row['suffix_transformer_ms_per_query_mean']:>10.2f}"
        )

    print()
    print("Suffix Transformer grouped breakdown")
    print(
        f"{'task':<15}"
        f"{'KV':>10}"
        f"{'Norm':>10}"
        f"{'Proj':>10}"
        f"{'Attn':>10}"
        f"{'MLP':>10}"
        f"{'Other':>10}"
    )

    for row in transformer_summary:
        print(
            f"{row['task']:<15}"
            f"{row['group_kv_handling_ms_per_query_mean']:>10.2f}"
            f"{row['group_normalization_ms_per_query_mean']:>10.2f}"
            f"{row['group_linear_projection_ms_per_query_mean']:>10.2f}"
            f"{row['group_attention_core_ms_per_query_mean']:>10.2f}"
            f"{row['group_mlp_group_ms_per_query_mean']:>10.2f}"
            f"{row['group_other_transformer_group_ms_per_query_mean']:>10.2f}"
        )

    print()
    print("Generated:")
    print(runs_csv)
    print(encoder_csv)
    print(transformer_csv)


if __name__ == "__main__":
    main()
