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

CASES = [
    "baseline",
    "fp_cache",
]


def last_match(pattern, text, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def parse_float_field(text, name):
    match = re.search(
        rf"\b{re.escape(name)}=([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    return float(match.group(1)) if match else None


def parse_int_field(text, name):
    match = re.search(
        rf"\b{re.escape(name)}=([0-9]+)",
        text,
    )
    return int(match.group(1)) if match else None


def parse_stage_summary(text):
    # Exclude the script's grep-based summary printed after the real run.
    main_text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    pattern = re.compile(
        r"GOFA stage timing summary: "
        r"report=(\d+), "
        r"model_total=([0-9.]+)s, "
        r"encoder_full_calls=(\d+), "
        r"prefix_calls=(\d+), "
        r"suffix_calls=(\d+), "
        r"decoder_calls=(\d+)"
    )

    matches = list(pattern.finditer(main_text))

    if not matches:
        return {}

    # Prefer the largest report count; break ties using the latest line.
    match = max(
        matches,
        key=lambda item: (
            int(item.group(1)),
            item.start(),
        ),
    )

    block_start = match.end()
    next_summary = pattern.search(
        main_text,
        block_start,
    )

    block_end = (
        next_summary.start()
        if next_summary
        else len(main_text)
    )

    block = main_text[block_start:block_end]

    decoder_match = re.search(
        r"^\s+decoder:\s+([0-9.]+)s",
        block,
        re.MULTILINE,
    )

    return {
        "queries": int(match.group(1)),
        "model_total_s": float(match.group(2)),
        "encoder_full_calls": int(match.group(3)),
        "prefix_calls": int(match.group(4)),
        "suffix_calls": int(match.group(5)),
        "decoder_calls": int(match.group(6)),
        "decoder_s": (
            float(decoder_match.group(1))
            if decoder_match
            else None
        ),
    }


def parse_cache_timing(text):
    main_text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    matches = list(
        re.finditer(
            r"GOFA encoder cache timing:[^\n]+",
            main_text,
        )
    )

    if not matches:
        return {}

    line = matches[-1].group(0)

    return {
        "cache_calls": parse_int_field(
            line,
            "call_total",
        ),
        "cache_hit_rate_pct": parse_float_field(
            line,
            "hit_rate",
        ),
        "cache_skip_rate_pct": parse_float_field(
            line,
            "skip_rate",
        ),
        "cache_total_s": parse_float_field(
            line,
            "cum_total",
        ),
        "cache_load_s": parse_float_field(
            line,
            "cum_load",
        ),
        "cache_online_prefix_s": parse_float_field(
            line,
            "cum_miss_compute",
        ),
        "cache_save_s": parse_float_field(
            line,
            "cum_save",
        ),
        "cache_assemble_s": parse_float_field(
            line,
            "cum_assemble",
        ),
        "cache_suffix_s": parse_float_field(
            line,
            "cum_suffix",
        ),
    }


def parse_cache_status(text):
    main_text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    matches = list(
        re.finditer(
            r"GOFA encoder memory/text-KV cache:[^\n]+",
            main_text,
        )
    )

    if not matches:
        return {}

    line = matches[-1].group(0)

    return {
        "total_hits": parse_int_field(
            line,
            "total_hits",
        ),
        "total_misses": parse_int_field(
            line,
            "total_misses",
        ),
        "total_skips": parse_int_field(
            line,
            "total_skips",
        ),
    }


def parse_transformer_breakdown(text):
    main_text = text.split(
        "\n===== TIMING LINES =====",
        1,
    )[0]

    matches = list(
        re.finditer(
            r"memory_kv_transformer_breakdown_total:[^\n]+",
            main_text,
        )
    )

    if not matches:
        return {}

    line = matches[-1].group(0)

    names = [
        "kv_to_device",
        "input_norm",
        "qkv_proj",
        "rope_cache_repeat",
        "attn_score_softmax_value",
        "o_proj",
        "post_attn_norm",
        "mlp",
    ]

    return {
        f"suffix_{name}_s": parse_float_field(
            line,
            name,
        )
        for name in names
    }


def parse_process_elapsed(text):
    matches = re.findall(
        r"PROCESS_ELAPSED_S=([0-9.]+)",
        text,
    )

    return float(matches[-1]) if matches else None


def safe_ms_per_query(value, queries):
    if value is None or not queries:
        return None

    return value * 1000.0 / queries


def mean(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    return statistics.mean(values) if values else None


def stdev(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if len(values) < 2:
        return 0.0 if values else None

    return statistics.stdev(values)


def fmt(value, digits=2):
    if value is None:
        return "NA"

    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    rows = []

    for task in TASKS:
        for case in CASES:
            case_dir = root / task / case

            for rep_dir in sorted(
                case_dir.glob("rep*")
            ):
                log_path = rep_dir / "run_stdout.log"

                if not log_path.is_file():
                    continue

                text = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                stage = parse_stage_summary(text)
                cache = parse_cache_timing(text)
                status = parse_cache_status(text)
                suffix = parse_transformer_breakdown(text)

                queries = stage.get("queries")

                row = {
                    "task": task,
                    "case": case,
                    "rep": rep_dir.name,
                    "log": str(log_path),
                    **stage,
                    **cache,
                    **status,
                    **suffix,
                    "process_elapsed_s": parse_process_elapsed(
                        text
                    ),
                }

                model_total = row.get(
                    "model_total_s"
                )
                decoder = row.get(
                    "decoder_s"
                )

                encoder_stage = None

                if (
                    model_total is not None
                    and decoder is not None
                ):
                    encoder_stage = (
                        model_total - decoder
                    )

                row["encoder_stage_s"] = encoder_stage

                row["model_ms_per_query"] = (
                    safe_ms_per_query(
                        model_total,
                        queries,
                    )
                )

                row["encoder_stage_ms_per_query"] = (
                    safe_ms_per_query(
                        encoder_stage,
                        queries,
                    )
                )

                row["decoder_ms_per_query"] = (
                    safe_ms_per_query(
                        decoder,
                        queries,
                    )
                )

                row["process_ms_per_query"] = (
                    safe_ms_per_query(
                        row.get("process_elapsed_s"),
                        queries,
                    )
                )

                for key in [
                    "cache_total_s",
                    "cache_load_s",
                    "cache_online_prefix_s",
                    "cache_assemble_s",
                    "cache_suffix_s",
                    "suffix_kv_to_device_s",
                    "suffix_input_norm_s",
                    "suffix_qkv_proj_s",
                    "suffix_rope_cache_repeat_s",
                    "suffix_attn_score_softmax_value_s",
                    "suffix_o_proj_s",
                    "suffix_post_attn_norm_s",
                    "suffix_mlp_s",
                ]:
                    row[
                        key.replace(
                            "_s",
                            "_ms_per_query",
                        )
                    ] = safe_ms_per_query(
                        row.get(key),
                        queries,
                    )

                rows.append(row)

    if not rows:
        raise SystemExit(
            f"No latency logs found under {root}"
        )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    runs_csv = root / "fp_latency_runs.csv"

    with runs_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["task"], row["case"])
        ].append(row)

    summary_rows = []

    metrics = [
        "model_ms_per_query",
        "encoder_stage_ms_per_query",
        "decoder_ms_per_query",
        "process_ms_per_query",
        "cache_total_ms_per_query",
        "cache_load_ms_per_query",
        "cache_online_prefix_ms_per_query",
        "cache_assemble_ms_per_query",
        "cache_suffix_ms_per_query",
        "suffix_kv_to_device_ms_per_query",
        "suffix_qkv_proj_ms_per_query",
        "suffix_rope_cache_repeat_ms_per_query",
        "suffix_attn_score_softmax_value_ms_per_query",
        "suffix_mlp_ms_per_query",
    ]

    for (task, case), group_rows in sorted(
        groups.items()
    ):
        output = {
            "task": task,
            "case": case,
            "repetitions": len(group_rows),
        }

        for metric in metrics:
            values = [
                row.get(metric)
                for row in group_rows
            ]

            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std"] = stdev(values)

        output["total_misses_max"] = max(
            (
                row.get("total_misses", 0)
                or 0
                for row in group_rows
            ),
            default=0,
        )

        summary_rows.append(output)

    summary_fields = sorted(
        {
            key
            for row in summary_rows
            for key in row
        }
    )

    summary_csv = root / "fp_latency_summary.csv"

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
    print("Per-run latency")
    print(
        f"{'task':<15}"
        f"{'case':<12}"
        f"{'rep':<7}"
        f"{'model(ms)':>12}"
        f"{'encoder(ms)':>14}"
        f"{'decoder(ms)':>14}"
        f"{'misses':>10}"
    )

    for row in rows:
        print(
            f"{row['task']:<15}"
            f"{row['case']:<12}"
            f"{row['rep']:<7}"
            f"{fmt(row.get('model_ms_per_query')):>12}"
            f"{fmt(row.get('encoder_stage_ms_per_query')):>14}"
            f"{fmt(row.get('decoder_ms_per_query')):>14}"
            f"{str(row.get('total_misses', '-')):>10}"
        )

    summary_lookup = {
        (row["task"], row["case"]): row
        for row in summary_rows
    }

    print()
    print("Baseline vs FP-cache")
    print(
        f"{'task':<15}"
        f"{'baseline enc':>15}"
        f"{'cache enc':>15}"
        f"{'enc speedup':>14}"
        f"{'enc reduction':>16}"
        f"{'model speedup':>16}"
    )

    for task in TASKS:
        baseline = summary_lookup.get(
            (task, "baseline")
        )

        cache = summary_lookup.get(
            (task, "fp_cache")
        )

        if baseline is None or cache is None:
            continue

        baseline_encoder = baseline.get(
            "encoder_stage_ms_per_query_mean"
        )

        cache_encoder = cache.get(
            "encoder_stage_ms_per_query_mean"
        )

        baseline_model = baseline.get(
            "model_ms_per_query_mean"
        )

        cache_model = cache.get(
            "model_ms_per_query_mean"
        )

        encoder_speedup = (
            baseline_encoder / cache_encoder
            if baseline_encoder and cache_encoder
            else None
        )

        encoder_reduction = (
            (
                baseline_encoder
                - cache_encoder
            )
            / baseline_encoder
            * 100.0
            if baseline_encoder and cache_encoder
            else None
        )

        model_speedup = (
            baseline_model / cache_model
            if baseline_model and cache_model
            else None
        )

        print(
            f"{task:<15}"
            f"{fmt(baseline_encoder):>15}"
            f"{fmt(cache_encoder):>15}"
            f"{fmt(encoder_speedup, 3):>14}"
            f"{fmt(encoder_reduction):>15}%"
            f"{fmt(model_speedup, 3):>16}"
        )

    print()
    print("Generated:")
    print(runs_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()
