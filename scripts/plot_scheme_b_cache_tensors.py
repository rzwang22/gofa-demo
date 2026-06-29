#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


SUMMARY_FIELDS = (
    "abs_max",
    "abs_p99",
    "max_over_p99",
    "rel_l2_error_base",
    "rel_l2_error_base_delta",
    "cosine_similarity_base",
    "cosine_similarity_base_delta",
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
            "matplotlib is required for scripts/plot_scheme_b_cache_tensors.py. "
            "Install matplotlib on the server environment and rerun this script."
        ) from exc


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _downsample_2d(tensor: torch.Tensor, max_tokens: int = 256, max_channels: int = 128) -> torch.Tensor:
    if tensor.dim() != 2:
        tensor = tensor.reshape(-1, tensor.size(-1))
    if tensor.size(0) > max_tokens:
        token_idx = torch.linspace(0, tensor.size(0) - 1, steps=max_tokens).round().long()
        tensor = tensor.index_select(0, token_idx)
    if tensor.size(1) > max_channels:
        channel_idx = torch.linspace(0, tensor.size(1) - 1, steps=max_channels).round().long()
        tensor = tensor.index_select(1, channel_idx)
    return tensor.to(torch.float32).cpu()


def _quantile(values: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    return torch.quantile(values.to(torch.float32), q, dim=dim)


def _align(*tensors: torch.Tensor) -> List[torch.Tensor]:
    min_tokens = min(int(tensor.size(0)) for tensor in tensors)
    min_channels = min(int(tensor.size(1)) for tensor in tensors)
    return [tensor[:min_tokens, :min_channels] for tensor in tensors]


def _is_empty_2d(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 0 or tensor.size(0) == 0 or tensor.size(1) == 0


def _plot_heatmap(plt, tensor: torch.Tensor, output_path: Path, title: str):
    array = tensor.to(torch.float32).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(array.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xlabel("token")
    ax.set_ylabel("channel")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_surface(plt, tensor: torch.Tensor, output_path: Path, title: str):
    sampled = _downsample_2d(tensor.abs(), max_tokens=256, max_channels=128)
    z = sampled.T.numpy()
    tokens = np.arange(sampled.size(0))
    channels = np.arange(sampled.size(1))
    x, y = np.meshgrid(tokens, channels)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("token")
    ax.set_ylabel("channel")
    ax.set_zlabel("abs value")
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
    ax.set_ylabel("abs value")
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
    ax.set_ylabel("abs value")
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
    axes[0].set_title("value")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count")
    axes[1].hist(abs_values, bins=120)
    axes[1].set_title("abs value")
    axes[1].set_xlabel("abs value")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_compare_heatmap(plt, versions: Dict[str, torch.Tensor], output_path: Path, title: str):
    modes = [mode for mode in ("full", "base", "base_delta") if mode in versions]
    if any(_is_empty_2d(versions[mode]) for mode in modes):
        return
    aligned = _align(*[versions[mode] for mode in modes])
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), squeeze=False)
    for ax, mode, tensor in zip(axes[0], modes, aligned):
        image = ax.imshow(tensor.to(torch.float32).numpy().T, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_title(mode)
        ax.set_xlabel("token")
        ax.set_ylabel("channel")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_heatmap(plt, versions: Dict[str, torch.Tensor], output_path: Path, title: str):
    if "full" not in versions:
        return
    modes = [mode for mode in ("base", "base_delta") if mode in versions]
    if not modes:
        return
    if _is_empty_2d(versions["full"]) or any(_is_empty_2d(versions[mode]) for mode in modes):
        return
    full_and_recon = _align(versions["full"], *[versions[mode] for mode in modes])
    full = full_and_recon[0]
    recon = full_and_recon[1:]
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), squeeze=False)
    for ax, mode, tensor in zip(axes[0], modes, recon):
        err = (full - tensor).abs()
        image = ax.imshow(err.to(torch.float32).numpy().T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"abs(full - {mode})")
        ax.set_xlabel("token")
        ax.set_ylabel("channel")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _rel_l2_by_dim(full: torch.Tensor, reconstructed: torch.Tensor, dim: int) -> torch.Tensor:
    diff = (reconstructed - full).to(torch.float32)
    full_f = full.to(torch.float32)
    numerator = torch.linalg.vector_norm(diff, dim=dim)
    denominator = torch.linalg.vector_norm(full_f, dim=dim).clamp(min=1e-12)
    return numerator / denominator


def _plot_token_error(plt, versions: Dict[str, torch.Tensor], output_path: Path, title: str):
    if "full" not in versions:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for mode in ("base", "base_delta"):
        if mode not in versions:
            continue
        if _is_empty_2d(versions["full"]) or _is_empty_2d(versions[mode]):
            continue
        full, reconstructed = _align(versions["full"], versions[mode])
        ax.plot(_rel_l2_by_dim(full, reconstructed, dim=1).numpy(), label=f"{mode}_rel_l2")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("token")
    ax.set_ylabel("relative L2 error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_channel_error(plt, versions: Dict[str, torch.Tensor], output_path: Path, title: str):
    if "full" not in versions:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for mode in ("base", "base_delta"):
        if mode not in versions:
            continue
        if _is_empty_2d(versions["full"]) or _is_empty_2d(versions[mode]):
            continue
        full, reconstructed = _align(versions["full"], versions[mode])
        ax.plot(_rel_l2_by_dim(full, reconstructed, dim=0).numpy(), label=f"{mode}_rel_l2")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("channel")
    ax.set_ylabel("relative L2 error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _iter_tensor_files(input_dir: Path, max_files: int) -> Iterable[Path]:
    tensor_dir = input_dir / "tensors"
    files = sorted(tensor_dir.glob("*.pt")) if tensor_dir.exists() else sorted(input_dir.glob("*.pt"))
    if max_files > 0:
        files = files[:max_files]
    return files


def _payload_group_key(payload: Dict) -> Tuple:
    return (
        payload.get("item_idx"),
        payload.get("cache_key"),
        payload.get("layer_idx"),
        payload.get("tensor_kind"),
    )


def _group_stem(payload: Dict) -> str:
    layer = payload.get("layer_idx")
    layer_part = "memory" if layer is None else f"layer{layer}"
    return _safe_stem(
        f"item{payload.get('item_idx')}_cachekey_{str(payload.get('cache_key'))[:12]}_"
        f"{layer_part}_{payload.get('tensor_kind')}"
    )


def _plot_tensor_payload(plt, payload: Dict, output_dir: Path):
    tensor = payload["tensor"].to(torch.float32)
    if tensor.dim() != 2:
        tensor = tensor.reshape(-1, tensor.size(-1))
    if _is_empty_2d(tensor):
        print(
            "Skipping empty tensor plot: "
            f"cache_key={payload.get('cache_key')}, layer={payload.get('layer_idx')}, "
            f"kind={payload.get('tensor_kind')}, mode={payload.get('reconstruct_mode')}"
        )
        return
    stem = _safe_stem(
        f"item{payload.get('item_idx')}_cachekey_{str(payload.get('cache_key'))[:12]}_"
        f"layer{payload.get('layer_idx')}_{payload.get('tensor_kind')}_{payload.get('reconstruct_mode')}"
    )
    title = (
        f"cache_key={str(payload.get('cache_key'))[:12]} mode={payload.get('reconstruct_mode')} "
        f"layer={payload.get('layer_idx')} kind={payload.get('tensor_kind')}"
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


def _sort_key(key: tuple) -> tuple:
    return tuple("" if value is None else str(value) for value in key)


def _write_grouped_csv(records: List[Dict], output_path: Path, group_fields: List[str]):
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field, "") for field in group_fields)
        grouped[key].append(record)
    header = group_fields + ["record_count"] + [f"mean_{field}" for field in SUMMARY_FIELDS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for key, items in sorted(grouped.items(), key=lambda item: _sort_key(item[0])):
            row = {field: value for field, value in zip(group_fields, key)}
            row["record_count"] = len(items)
            for field in SUMMARY_FIELDS:
                values = [float(item[field]) for item in items if item.get(field) is not None]
                row[f"mean_{field}"] = _mean(values)
            writer.writerow(row)


def _write_summary_csvs(input_dir: Path, output_dir: Path):
    stats_path = input_dir / "stats" / "cache_tensor_stats.jsonl"
    if not stats_path.exists():
        stats_path = input_dir / "cache_tensor_stats.jsonl"
    records = _read_jsonl(stats_path)
    if not records:
        print(f"No cache_tensor_stats.jsonl records found under {input_dir}; skipping CSV summaries.")
        return
    _write_grouped_csv(records, output_dir / "summary_by_tensor_kind.csv", ["tensor_kind"])
    _write_grouped_csv(records, output_dir / "summary_by_layer_kind.csv", ["layer_idx", "tensor_kind"])
    _write_grouped_csv(records, output_dir / "summary_by_reconstruct_mode.csv", ["reconstruct_mode"])
    _write_grouped_csv(records, output_dir / "summary_by_layer_kind_mode.csv", ["layer_idx", "tensor_kind", "reconstruct_mode"])


def main():
    parser = argparse.ArgumentParser(description="Plot GOFA Scheme-B cache tensor observer outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()

    tensor_files = list(_iter_tensor_files(input_dir, args.max_files))
    groups: Dict[Tuple, Dict[str, torch.Tensor]] = defaultdict(dict)
    group_payloads: Dict[Tuple, Dict] = {}
    for tensor_path in tensor_files:
        payload = torch.load(tensor_path, map_location="cpu")
        tensor = payload["tensor"].to(torch.float32)
        if tensor.dim() != 2:
            tensor = tensor.reshape(-1, tensor.size(-1))
        payload["tensor"] = tensor
        print(f"Plotting {tensor_path}")
        _plot_tensor_payload(plt, payload, output_dir)
        key = _payload_group_key(payload)
        groups[key][payload.get("reconstruct_mode")] = tensor
        group_payloads[key] = payload

    for key, versions in groups.items():
        if "full" not in versions or not ({"base", "base_delta"} & set(versions)):
            continue
        payload = group_payloads[key]
        stem = _group_stem(payload)
        title = (
            f"cache_key={str(payload.get('cache_key'))[:12]} "
            f"layer={payload.get('layer_idx')} kind={payload.get('tensor_kind')}"
        )
        _plot_compare_heatmap(plt, versions, output_dir / f"{stem}_compare_heatmap.png", title)
        _plot_error_heatmap(plt, versions, output_dir / f"{stem}_error_heatmap.png", title)
        _plot_token_error(plt, versions, output_dir / f"{stem}_token_error.png", title)
        _plot_channel_error(plt, versions, output_dir / f"{stem}_channel_error.png", title)

    _write_summary_csvs(input_dir, output_dir)
    print(
        "Scheme-B cache tensor plotting complete: "
        f"input_dir={input_dir}, output_dir={output_dir}, tensor_files={len(tensor_files)}, "
        f"comparison_groups={sum(1 for versions in groups.values() if 'full' in versions and ({'base', 'base_delta'} & set(versions)))}"
    )


if __name__ == "__main__":
    main()
