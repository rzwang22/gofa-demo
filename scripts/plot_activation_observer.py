#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


SUMMARY_FIELDS = (
    "quant_error_4bit_rel_l2",
    "quant_error_8bit_rel_l2",
    "max_over_p99",
    "token_max_over_p99_mean",
    "channel_max_over_p99_mean",
    "quant_error_4bit_zero_ratio",
    "quant_error_4bit_saturation_ratio",
    "quant_error_8bit_zero_ratio",
    "quant_error_8bit_saturation_ratio",
)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for scripts/plot_activation_observer.py. "
            "Install matplotlib on the server environment and rerun this script."
        ) from exc


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _downsample_2d(tensor: torch.Tensor, max_tokens: int, max_channels: int) -> torch.Tensor:
    if tensor.dim() != 2:
        tensor = tensor.reshape(-1, tensor.size(-1))
    if tensor.size(0) > max_tokens:
        token_idx = torch.linspace(0, tensor.size(0) - 1, steps=max_tokens).round().long()
        tensor = tensor.index_select(0, token_idx)
    if tensor.size(1) > max_channels:
        channel_idx = torch.linspace(0, tensor.size(1) - 1, steps=max_channels).round().long()
        tensor = tensor.index_select(1, channel_idx)
    return tensor


def _quantile(values: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    return torch.quantile(values.to(torch.float32), q, dim=dim)


def _plot_heatmap(plt, tensor: torch.Tensor, output_path: Path, title: str):
    array = tensor.to(torch.float32).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(array.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xlabel("token")
    ax.set_ylabel("channel")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="activation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_surface(plt, tensor: torch.Tensor, output_path: Path, title: str):
    sampled = _downsample_2d(tensor.abs().to(torch.float32), max_tokens=256, max_channels=128)
    z = sampled.T.numpy()
    tokens = np.arange(sampled.size(0))
    channels = np.arange(sampled.size(1))
    x, y = np.meshgrid(tokens, channels)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("token")
    ax.set_ylabel("channel")
    ax.set_zlabel("abs activation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_token_stats(plt, tensor: torch.Tensor, output_path: Path, title: str):
    abs_tensor = tensor.abs().to(torch.float32)
    token_abs_max = abs_tensor.amax(dim=1)
    token_abs_p99 = _quantile(abs_tensor, 0.99, dim=1)
    token_abs_p999 = _quantile(abs_tensor, 0.999, dim=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(token_abs_max.numpy(), label="token_abs_max")
    ax.plot(token_abs_p99.numpy(), label="token_abs_p99")
    ax.plot(token_abs_p999.numpy(), label="token_abs_p999")
    ax.set_xlabel("token")
    ax.set_ylabel("abs activation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_channel_stats(plt, tensor: torch.Tensor, output_path: Path, title: str):
    abs_tensor = tensor.abs().to(torch.float32)
    channel_abs_max = abs_tensor.amax(dim=0)
    channel_abs_p99 = _quantile(abs_tensor, 0.99, dim=0)
    channel_abs_p999 = _quantile(abs_tensor, 0.999, dim=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(channel_abs_max.numpy(), label="channel_abs_max")
    ax.plot(channel_abs_p99.numpy(), label="channel_abs_p99")
    ax.plot(channel_abs_p999.numpy(), label="channel_abs_p999")
    ax.set_xlabel("channel")
    ax.set_ylabel("abs activation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_hist(plt, tensor: torch.Tensor, output_path: Path, title: str):
    values = tensor.to(torch.float32).flatten().numpy()
    abs_values = tensor.abs().to(torch.float32).flatten().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(values, bins=120)
    axes[0].set_title("activation")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count")
    axes[1].hist(abs_values, bins=120)
    axes[1].set_title("abs activation")
    axes[1].set_xlabel("abs value")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _iter_tensor_files(input_dir: Path, max_files: int) -> Iterable[Path]:
    tensor_dir = input_dir / "tensors"
    search_dir = tensor_dir if tensor_dir.exists() else input_dir
    files = sorted(search_dir.glob("*.pt"))
    if max_files > 0:
        files = files[:max_files]
    return files


def _plot_tensor_file(plt, tensor_path: Path, output_dir: Path):
    payload = torch.load(tensor_path, map_location="cpu")
    tensor = payload["saved_tensor"].to(torch.float32)
    if tensor.dim() != 2:
        tensor = tensor.reshape(-1, tensor.size(-1))
    stem = _safe_stem(tensor_path)
    title = (
        f"layer={payload.get('layer_idx')} projection={payload.get('projection_type')} "
        f"module={payload.get('module_name')}"
    )
    _plot_heatmap(plt, tensor, output_dir / f"{stem}_heatmap.png", title)
    _plot_surface(plt, tensor, output_dir / f"{stem}_surface3d.png", title)
    _plot_token_stats(plt, tensor, output_dir / f"{stem}_token_stats.png", title)
    _plot_channel_stats(plt, tensor, output_dir / f"{stem}_channel_stats.png", title)
    _plot_hist(plt, tensor, output_dir / f"{stem}_hist.png", title)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_grouped_csv(records: List[Dict], output_path: Path, group_fields: List[str]):
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field, "") for field in group_fields)
        grouped[key].append(record)
    header = group_fields + ["record_count"] + [f"mean_{field}" for field in SUMMARY_FIELDS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for key, items in sorted(grouped.items()):
            row = {field: value for field, value in zip(group_fields, key)}
            row["record_count"] = len(items)
            for field in SUMMARY_FIELDS:
                values = [float(item[field]) for item in items if field in item and item[field] is not None]
                row[f"mean_{field}"] = _mean(values)
            writer.writerow(row)


def _write_summary_csvs(input_dir: Path, output_dir: Path):
    records = _read_jsonl(input_dir / "activation_stats.jsonl")
    if not records:
        print(f"No activation_stats.jsonl records found under {input_dir}; skipping CSV summaries.")
        return
    _write_grouped_csv(records, output_dir / "summary_by_projection.csv", ["projection_type"])
    _write_grouped_csv(records, output_dir / "summary_by_layer_projection.csv", ["layer_idx", "projection_type"])


def main():
    parser = argparse.ArgumentParser(description="Plot GOFA suffix activation observer outputs.")
    parser.add_argument("--input-dir", required=True, help="Observer output directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG and CSV outputs.")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum tensor .pt files to plot; <=0 means all.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()

    tensor_files = list(_iter_tensor_files(input_dir, args.max_files))
    for tensor_path in tensor_files:
        print(f"Plotting {tensor_path}")
        _plot_tensor_file(plt, tensor_path, output_dir)
    _write_summary_csvs(input_dir, output_dir)
    print(
        "Activation observer plotting complete: "
        f"input_dir={input_dir}, output_dir={output_dir}, tensor_files={len(tensor_files)}"
    )


if __name__ == "__main__":
    main()
