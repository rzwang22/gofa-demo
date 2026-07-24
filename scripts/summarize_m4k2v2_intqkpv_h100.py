#!/usr/bin/env python3

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TASKS = [
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
]

# 终端显示时优先展示 timing，随后展示 breakdown。
MODES = [
    "timing",
    "breakdown",
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


def read_log(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean_values:
        return None

    return statistics.mean(clean_values)


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean_values:
        return None

    if len(clean_values) == 1:
        return 0.0

    return statistics.stdev(clean_values)


def to_ms_per_query(
    seconds: Optional[float],
    queries: int,
) -> Optional[float]:
    if seconds is None:
        return None

    if queries <= 0:
        raise ValueError(
            f"Invalid query count: {queries}"
        )

    return seconds * 1000.0 / queries


def parse_float_field(
    line: str,
    field: str,
    suffix: str = "",
) -> Optional[float]:
    match = re.search(
        rf"\b{re.escape(field)}="
        rf"([-+]?[0-9]+(?:\.[0-9]+)?)"
        rf"{re.escape(suffix)}",
        line,
    )

    if match is None:
        return None

    return float(match.group(1))


def parse_int_field(
    line: str,
    field: str,
) -> Optional[int]:
    match = re.search(
        rf"\b{re.escape(field)}=([0-9]+)",
        line,
    )

    if match is None:
        return None

    return int(match.group(1))


def remove_appended_shell_output(text: str) -> str:
    """
    某些运行脚本会在模型退出后再次 grep timing lines。
    这里删除这些重复打印，避免解析到重复的 profiler 行。
    """
    markers = [
        "\n===== TIMING LINES =====",
        "\n===== FINAL CACHE STATUS =====",
        "\n===== FINAL CACHE TIMING =====",
        "\n===== FINAL STAGE SUMMARY =====",
    ]

    cut_position = len(text)

    for marker in markers:
        position = text.find(marker)

        if position >= 0:
            cut_position = min(
                cut_position,
                position,
            )

    return text[:cut_position]


def extract_final_stage_summary(
    text: str,
) -> Dict[str, Any]:
    text = remove_appended_shell_output(text)

    pattern = re.compile(
        r"GOFA stage timing summary:\s*"
        r"report=(\d+),\s*"
        r"model_total=([0-9.]+)s,"
        r"[^\n]*"
    )

    matches = list(pattern.finditer(text))

    if not matches:
        raise ValueError(
            "Missing GOFA stage timing summary"
        )

    # 优先选择正式的 200-query 汇总。
    report_200_matches = [
        match
        for match in matches
        if int(match.group(1)) == 200
    ]

    if report_200_matches:
        final_match = report_200_matches[-1]
    else:
        final_match = matches[-1]

    queries = int(final_match.group(1))
    model_total_s = float(final_match.group(2))

    block_start = final_match.end()

    next_match = pattern.search(
        text,
        block_start,
    )

    block_end = (
        next_match.start()
        if next_match is not None
        else len(text)
    )

    return {
        "queries": queries,
        "model_total_s": model_total_s,
        "block": text[block_start:block_end],
    }


def extract_stage_value(
    block: str,
    field: str,
) -> Optional[float]:
    match = re.search(
        rf"^\s*{re.escape(field)}:"
        rf"\s*([0-9.]+)s",
        block,
        re.MULTILINE,
    )

    if match is None:
        return None

    return float(match.group(1))


def extract_numbered_layers(
    block: str,
    prefix: str,
) -> Dict[int, float]:
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


def extract_cache_timing(
    text: str,
    expected_queries: int,
) -> Dict[str, Optional[float]]:
    text = remove_appended_shell_output(text)

    lines = re.findall(
        r"GOFA encoder cache timing:[^\n]+",
        text,
    )

    if not lines:
        return {
            "cache_total_s": None,
            "cache_load_s": None,
            "online_prefix_s": None,
            "cache_assembly_s": None,
            "suffix_s": None,
        }

    expected_marker = (
        f"call_total={expected_queries}"
    )

    candidates = [
        line
        for line in lines
        if expected_marker in line
    ]

    line = (
        candidates[-1]
        if candidates
        else lines[-1]
    )

    return {
        "cache_total_s":
            parse_float_field(
                line,
                "cum_total",
                "s",
            ),

        "cache_load_s":
            parse_float_field(
                line,
                "cum_load",
                "s",
            ),

        "online_prefix_s":
            parse_float_field(
                line,
                "cum_miss_compute",
                "s",
            ),

        "cache_assembly_s":
            parse_float_field(
                line,
                "cum_assemble",
                "s",
            ),

        "suffix_s":
            parse_float_field(
                line,
                "cum_suffix",
                "s",
            ),
    }


def extract_cache_status(
    text: str,
) -> Dict[str, Optional[int]]:
    text = remove_appended_shell_output(text)

    lines = re.findall(
        r"GOFA encoder memory/text-KV cache:[^\n]+",
        text,
    )

    if not lines:
        return {
            "total_hits": None,
            "total_misses": None,
            "total_skips": None,
        }

    line = lines[-1]

    return {
        "total_hits":
            parse_int_field(
                line,
                "total_hits",
            ),

        "total_misses":
            parse_int_field(
                line,
                "total_misses",
            ),

        "total_skips":
            parse_int_field(
                line,
                "total_skips",
            ),
    }


def extract_transformer_breakdown(
    text: str,
) -> Dict[str, Optional[float]]:
    text = remove_appended_shell_output(text)

    lines = re.findall(
        r"memory_kv_transformer_breakdown_total:"
        r"[^\n]+",
        text,
    )

    output: Dict[str, Optional[float]] = {
        f"{component}_s": None
        for component in TRANSFORMER_COMPONENTS
    }

    if not lines:
        return output

    line = lines[-1]

    for component in TRANSFORMER_COMPONENTS:
        output[f"{component}_s"] = (
            parse_float_field(
                line,
                component,
                "s",
            )
        )

    return output


def sum_layers(
    layers: Dict[int, float],
    selected_layers: Iterable[int],
) -> Optional[float]:
    selected_layers = list(selected_layers)

    if not selected_layers:
        return None

    if not all(
        layer in layers
        for layer in selected_layers
    ):
        return None

    return sum(
        layers[layer]
        for layer in selected_layers
    )


def parse_run(
    log_path: Path,
    task: str,
    mode: str,
    rep: str,
) -> Dict[str, Any]:
    text = read_log(log_path)

    if re.search(
        r"Traceback|CUDA out of memory|"
        r"torch\.OutOfMemoryError|RuntimeError:",
        text,
    ):
        raise ValueError(
            "Runtime error found in log"
        )

    stage = extract_final_stage_summary(text)

    queries = int(stage["queries"])
    model_total_s = float(
        stage["model_total_s"]
    )
    block = str(stage["block"])

    if queries != 200:
        raise ValueError(
            f"Expected report=200, found report={queries}"
        )

    decoder_s = extract_stage_value(
        block,
        "decoder",
    )

    if decoder_s is None:
        raise ValueError(
            "Missing decoder stage time"
        )

    encoder_s = (
        model_total_s - decoder_s
    )

    transformer_layers = (
        extract_numbered_layers(
            block,
            "encoder_transformer_layer_",
        )
    )

    gnn_layers = extract_numbered_layers(
        block,
        "encoder_gnn_layer_",
    )

    suffix_transformer_s = sum_layers(
        transformer_layers,
        range(26, 32),
    )

    suffix_gnn_s = sum_layers(
        gnn_layers,
        range(6),
    )

    cache = extract_cache_timing(
        text,
        expected_queries=queries,
    )

    status = extract_cache_status(text)

    transformer_breakdown = (
        extract_transformer_breakdown(text)
    )

    row: Dict[str, Any] = {
        "task": task,
        "mode": mode,
        "rep": rep,
        "queries": queries,
        "log": str(log_path),

        "model_total_ms_per_query":
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

        "cache_total_ms_per_query":
            to_ms_per_query(
                cache["cache_total_s"],
                queries,
            ),

        "cache_load_ms_per_query":
            to_ms_per_query(
                cache["cache_load_s"],
                queries,
            ),

        "online_prefix_ms_per_query":
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
                cache["suffix_s"],
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

        "total_hits": status["total_hits"],
        "total_misses": status["total_misses"],
        "total_skips": status["total_skips"],
    }

    for component in TRANSFORMER_COMPONENTS:
        seconds = transformer_breakdown[
            f"{component}_s"
        ]

        row[
            f"{component}_ms_per_query"
        ] = to_ms_per_query(
            seconds,
            queries,
        )

    # 只有 breakdown 日志中同时存在层级 Transformer 时间
    # 和内部组件时间时，才计算 Other Transformer。
    profiled_values = [
        transformer_breakdown[
            f"{component}_s"
        ]
        for component in TRANSFORMER_COMPONENTS
    ]

    if (
        suffix_transformer_s is not None
        and all(
            value is not None
            for value in profiled_values
        )
    ):
        profiled_s = sum(
            float(value)
            for value in profiled_values
            if value is not None
        )

        other_transformer_s = (
            suffix_transformer_s
            - profiled_s
        )

        # 仅消除日志打印四舍五入造成的极小负值。
        if -0.02 < other_transformer_s < 0:
            other_transformer_s = 0.0

        row[
            "other_transformer_ms_per_query"
        ] = to_ms_per_query(
            other_transformer_s,
            queries,
        )
    else:
        row[
            "other_transformer_ms_per_query"
        ] = None

    return row


def format_csv_value(value: Any) -> Any:
    """
    CSV 中浮点数固定保留 6 位小数，
    避免出现 1892.3036666666667 这样的长尾。
    """
    if isinstance(value, float):
        return f"{value:.6f}"

    if value is None:
        return ""

    return value


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: format_csv_value(
                    row.get(field)
                )
                for field in fieldnames
            })


def format_terminal_float(
    value: Optional[float],
    width: int = 14,
) -> str:
    if value is None:
        return f"{'NA':>{width}}"

    return f"{value:>{width}.2f}"


def format_terminal_int(
    value: Optional[int],
    width: int = 10,
) -> str:
    if value is None:
        return f"{'NA':>{width}}"

    return f"{value:>{width}d}"


def collect_runs(
    root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for task in TASKS:
        for mode in MODES:
            mode_root = (
                root / task / mode
            )

            if not mode_root.is_dir():
                continue

            rep_dirs = sorted(
                (
                    path
                    for path in mode_root.glob("rep*")
                    if path.is_dir()
                ),
                key=lambda path: path.name,
            )

            for rep_dir in rep_dirs:
                log_path = (
                    rep_dir / "run_stdout.log"
                )

                if not log_path.is_file():
                    continue

                try:
                    row = parse_run(
                        log_path=log_path,
                        task=task,
                        mode=mode,
                        rep=rep_dir.name,
                    )
                except Exception as exception:
                    raise SystemExit(
                        f"Failed to parse {log_path}: "
                        f"{exception}"
                    ) from exception

                rows.append(row)

    if not rows:
        raise SystemExit(
            f"No valid logs found under {root}"
        )

    return rows


def aggregate_runs(
    run_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[
        tuple,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in run_rows:
        groups[
            (
                row["task"],
                row["mode"],
            )
        ].append(row)

    metrics = [
        "model_total_ms_per_query",
        "encoder_ms_per_query",
        "decoder_ms_per_query",

        "cache_total_ms_per_query",
        "cache_load_ms_per_query",
        "online_prefix_ms_per_query",
        "cache_assembly_ms_per_query",

        "suffix_total_ms_per_query",
        "suffix_gnn_ms_per_query",
        "suffix_transformer_ms_per_query",

        "kv_to_device_ms_per_query",
        "input_norm_ms_per_query",
        "qkv_proj_ms_per_query",
        "rope_cache_repeat_ms_per_query",
        "attn_score_softmax_value_ms_per_query",
        "o_proj_ms_per_query",
        "post_attn_norm_ms_per_query",
        "mlp_ms_per_query",
        "other_transformer_ms_per_query",
    ]

    summary_rows: List[
        Dict[str, Any]
    ] = []

    for task in TASKS:
        for mode in MODES:
            rows = groups.get(
                (task, mode),
                [],
            )

            if not rows:
                continue

            output: Dict[str, Any] = {
                "task": task,
                "mode": mode,
                "repetitions": len(rows),

                "max_total_misses": max(
                    (
                        int(
                            row["total_misses"]
                        )
                        for row in rows
                        if row.get(
                            "total_misses"
                        ) is not None
                    ),
                    default=0,
                ),
            }

            for metric in metrics:
                values = [
                    row.get(metric)
                    for row in rows
                ]

                output[
                    f"{metric}_mean"
                ] = mean(values)

                output[
                    f"{metric}_std"
                ] = std(values)

            summary_rows.append(output)

    return summary_rows


def print_main_summary(
    summary_rows: List[Dict[str, Any]],
) -> None:
    print()
    print(
        f"{'task':<15}"
        f"{'mode':<12}"
        f"{'reps':>6}"
        f"{'model ms':>14}"
        f"{'encoder ms':>14}"
        f"{'decoder ms':>14}"
        f"{'misses':>10}"
    )

    print(
        "-" * 85
    )

    for row in summary_rows:
        print(
            f"{row['task']:<15}"
            f"{row['mode']:<12}"
            f"{int(row['repetitions']):>6d}"
            + format_terminal_float(
                row.get(
                    "model_total_ms_per_query_mean"
                )
            )
            + format_terminal_float(
                row.get(
                    "encoder_ms_per_query_mean"
                )
            )
            + format_terminal_float(
                row.get(
                    "decoder_ms_per_query_mean"
                )
            )
            + format_terminal_int(
                row.get(
                    "max_total_misses"
                )
            )
        )


def print_timing_repetitions(
    run_rows: List[Dict[str, Any]],
) -> None:
    print()
    print("Timing per-repetition model latency")
    print(
        f"{'task':<15}"
        f"{'rep':<8}"
        f"{'model ms':>14}"
        f"{'encoder ms':>14}"
        f"{'decoder ms':>14}"
    )

    print(
        "-" * 65
    )

    for task in TASKS:
        rows = [
            row
            for row in run_rows
            if (
                row["task"] == task
                and row["mode"] == "timing"
            )
        ]

        for row in rows:
            print(
                f"{task:<15}"
                f"{row['rep']:<8}"
                + format_terminal_float(
                    row.get(
                        "model_total_ms_per_query"
                    )
                )
                + format_terminal_float(
                    row.get(
                        "encoder_ms_per_query"
                    )
                )
                + format_terminal_float(
                    row.get(
                        "decoder_ms_per_query"
                    )
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize canonical H100 "
            "M4K2V2 + W4A8 + INT-QK/PV runs."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=(
            "Root of latency_m4k2v2_"
            "w4a8_intqkpv"
        ),
    )

    arguments = parser.parse_args()

    root = arguments.root.resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Root directory does not exist: {root}"
        )

    run_rows = collect_runs(root)

    summary_rows = aggregate_runs(
        run_rows
    )

    run_fields = sorted({
        key
        for row in run_rows
        for key in row.keys()
    })

    summary_fields = sorted({
        key
        for row in summary_rows
        for key in row.keys()
    })

    runs_csv = (
        root
        / "m4k2v2_intqkpv_h100_runs.csv"
    )

    summary_csv = (
        root
        / "m4k2v2_intqkpv_h100_summary.csv"
    )

    write_csv(
        runs_csv,
        run_rows,
        run_fields,
    )

    write_csv(
        summary_csv,
        summary_rows,
        summary_fields,
    )

    print_main_summary(
        summary_rows
    )

    print_timing_repetitions(
        run_rows
    )

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()