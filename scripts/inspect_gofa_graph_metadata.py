#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _jsonify(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if hasattr(value, "shape"):
        try:
            return list(value.shape)
        except Exception:
            return None
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
        try:
            return [len(value)]
        except Exception:
            return None
    return None


def _object_field_summary(obj: Any, field_names: Iterable[str]) -> Dict[str, Any]:
    summary = {}
    for name in field_names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            summary[name] = {
                "exists": True,
                "type": type(value).__name__,
                "shape": _shape(value),
                "sample": _jsonify(value[:5] if hasattr(value, "__getitem__") and not isinstance(value, dict) else value)
                if name in {"node_map", "edge_map", "target_index", "question_index", "node_ids"} else None,
            }
        else:
            summary[name] = {"exists": False}
    return summary


def _edge_index_tensor(obj: Any) -> Optional[torch.Tensor]:
    edge_index = getattr(obj, "edge_index", None)
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2:
        return edge_index.detach().cpu().long()
    return None


def _num_nodes(obj: Any, edge_index: Optional[torch.Tensor] = None) -> Optional[int]:
    for attr in ("num_nodes", "num_node", "num_node_feat"):
        if hasattr(obj, attr):
            try:
                return int(getattr(obj, attr))
            except Exception:
                pass
    node_map = getattr(obj, "node_map", None)
    if isinstance(node_map, torch.Tensor):
        return int(node_map.numel())
    x = getattr(obj, "x", None)
    if x is not None and hasattr(x, "__len__"):
        try:
            return int(len(x))
        except Exception:
            pass
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return None


def _degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    degree = torch.zeros(int(num_nodes), dtype=torch.long)
    for src, dst in edge_index.t().tolist():
        src = int(src)
        dst = int(dst)
        if 0 <= src < num_nodes:
            degree[src] += 1
        if 0 <= dst < num_nodes and dst != src:
            degree[dst] += 1
    return degree


def _degree_summary(degree: torch.Tensor) -> Dict[str, Any]:
    if degree.numel() == 0:
        return {}
    degree_f = degree.to(torch.float32)
    top_k = min(20, int(degree.numel()))
    top_values, top_indices = torch.topk(degree, k=top_k)
    return {
        "min": int(degree.min().item()),
        "max": int(degree.max().item()),
        "mean": float(degree_f.mean().item()),
        "p50": float(torch.quantile(degree_f, 0.50).item()),
        "p90": float(torch.quantile(degree_f, 0.90).item()),
        "p99": float(torch.quantile(degree_f, 0.99).item()),
        "top_degree_nodes": [
            {"node_idx": int(idx.item()), "degree": int(value.item())}
            for idx, value in zip(top_indices, top_values)
        ],
    }


def _int_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_int_list(item))
        return result
    return [int(value)]


def _neighbors(edge_index: Optional[torch.Tensor], targets: Iterable[int], num_nodes: int) -> List[int]:
    if edge_index is None:
        return []
    target_set = {int(idx) for idx in targets if 0 <= int(idx) < num_nodes}
    neighbors = set()
    for src, dst in edge_index.t().tolist():
        src = int(src)
        dst = int(dst)
        if src in target_set and 0 <= dst < num_nodes:
            neighbors.add(dst)
        if dst in target_set and 0 <= src < num_nodes:
            neighbors.add(src)
    return sorted(neighbors)


def _sample_debug(sample: Any, sample_idx: int) -> Dict[str, Any]:
    edge_index = _edge_index_tensor(sample)
    num_nodes = _num_nodes(sample, edge_index=edge_index) or 0
    local_degree = _degree(edge_index, num_nodes) if edge_index is not None else torch.zeros(num_nodes, dtype=torch.long)
    target_index = _int_list(getattr(sample, "target_index", None))
    valid_targets = [idx for idx in target_index if 0 <= idx < num_nodes]
    target_neighbors = _neighbors(edge_index, valid_targets, num_nodes)
    edge_count = int(edge_index.size(1)) if edge_index is not None else 0
    target_local_degree = {
        str(idx): int(local_degree[idx].item()) for idx in valid_targets if idx < local_degree.numel()
    }
    node_map = getattr(sample, "node_map", None)
    edge_map = getattr(sample, "edge_map", None)
    edge_cache_mapping = []
    if edge_index is not None:
        edge_map_values = edge_map.detach().cpu().reshape(-1).tolist() if isinstance(edge_map, torch.Tensor) else None
        for edge_pos, (src, dst) in enumerate(edge_index.t().tolist()[:10]):
            edge_offset = int(edge_map_values[edge_pos]) if edge_map_values is not None and edge_pos < len(edge_map_values) else edge_pos
            edge_cache_mapping.append({
                "edge_pos": edge_pos,
                "src_local": int(src),
                "dst_local": int(dst),
                "edge_map_value": edge_offset,
                "cache_item_idx": int(num_nodes + edge_offset),
            })
    return {
        "sample_idx": sample_idx,
        "fields": _object_field_summary(
            sample,
            (
                "edge_index",
                "target_index",
                "question_index",
                "node_map",
                "edge_map",
                "node_ids",
                "node_attr",
                "edge_attr",
                "x",
            ),
        ),
        "target_index": target_index,
        "valid_target_index": valid_targets,
        "local_node_count": int(num_nodes),
        "edge_count": edge_count,
        "local_degree_summary": _degree_summary(local_degree),
        "target_local_degree": target_local_degree,
        "target_1hop_neighbors": target_neighbors,
        "node_map_example": _jsonify(node_map[:10]) if isinstance(node_map, torch.Tensor) else _jsonify(node_map[:10]) if hasattr(node_map, "__getitem__") else None,
        "edge_map_example": _jsonify(edge_map[:10]) if isinstance(edge_map, torch.Tensor) else _jsonify(edge_map[:10]) if hasattr(edge_map, "__getitem__") else None,
        "cache_item_order_equals_local_node_order": True,
        "node_cache_item_range": [0, int(num_nodes)],
        "edge_cache_item_range": [int(num_nodes), int(num_nodes + edge_count)],
        "edge_index_to_cache_item_example": edge_cache_mapping,
    }


def _candidate_objects(root_obj: Any) -> List[Tuple[str, Any]]:
    candidates = [("wrapper", root_obj)]
    for name in ("data", "dataset", "task", "task_list", "tasks"):
        if hasattr(root_obj, name):
            value = getattr(root_obj, name)
            candidates.append((name, value))
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value[:5]):
                    candidates.append((f"{name}[{idx}]", item))
                    for subname in ("data", "dataset"):
                        if hasattr(item, subname):
                            candidates.append((f"{name}[{idx}].{subname}", getattr(item, subname)))
    return candidates


def _find_full_graph(candidates: List[Tuple[str, Any]]) -> Tuple[Optional[str], Optional[Any]]:
    for name, obj in candidates:
        if _edge_index_tensor(obj) is not None:
            return name, obj
        data = getattr(obj, "data", None)
        if data is not None and _edge_index_tensor(data) is not None:
            return f"{name}.data", data
    return None, None


def _inspect_split(args, split: str) -> Dict[str, Any]:
    from tasks import GOFAFineTuneTaskWrapper

    split_summary: Dict[str, Any] = {"split": split}
    wrapper = GOFAFineTuneTaskWrapper(
        args.task_name,
        root=args.data_root_path,
        split=split,
        save_data=False,
        from_saved=True,
        sample_size=args.max_samples,
        hop=args.hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        way=args.ways,
        instruction=args.instructs,
        selection=args.selections,
        num_workers=args.num_workers,
    )
    split_summary["wrapper_type"] = type(wrapper).__name__
    split_summary["wrapper_len"] = len(wrapper) if hasattr(wrapper, "__len__") else None
    candidates = _candidate_objects(wrapper)
    split_summary["candidate_object_fields"] = {
        name: _object_field_summary(
            obj,
            ("edge_index", "num_nodes", "node_map", "edge_map", "degree", "adj_t", "node_ids"),
        )
        for name, obj in candidates[:20]
    }
    full_graph_name, full_graph = _find_full_graph(candidates)
    split_summary["full_graph_candidate"] = full_graph_name
    if full_graph is not None:
        full_edge_index = _edge_index_tensor(full_graph)
        full_num_nodes = _num_nodes(full_graph, edge_index=full_edge_index)
        if full_edge_index is not None and full_num_nodes is not None:
            global_degree = _degree(full_edge_index, full_num_nodes)
            split_summary["full_graph"] = {
                "found": True,
                "object": full_graph_name,
                "num_nodes": int(full_num_nodes),
                "edge_count": int(full_edge_index.size(1)),
                "degree_summary": _degree_summary(global_degree),
                "node_degree": {str(idx): int(value.item()) for idx, value in enumerate(global_degree)},
            }
    else:
        split_summary["full_graph"] = {
            "found": False,
            "message": "no full graph degree metadata found",
        }

    sample_count = min(int(args.max_samples), int(len(wrapper))) if hasattr(wrapper, "__len__") else int(args.max_samples)
    samples = []
    for sample_idx in range(sample_count):
        try:
            sample = wrapper[sample_idx]
            samples.append(_sample_debug(sample, sample_idx))
        except Exception as exc:
            samples.append({"sample_idx": sample_idx, "error": repr(exc)})
    split_summary["sample_count"] = sample_count
    split_summary["sample_debug"] = samples
    return split_summary


def main():
    parser = argparse.ArgumentParser(description="Inspect GOFA task graph metadata for Scheme-B cache policies.")
    parser.add_argument("--data-root-path", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--splits", default="val,test,train")
    parser.add_argument("--hop", type=int, default=3)
    parser.add_argument("--max-nodes-per-hop", type=int, default=10)
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--instructs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(
        args.output_path
        or f"/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/metadata/{args.task_name}_graph_inspect.json"
    )
    summary = {
        "data_root_path": args.data_root_path,
        "task_name": args.task_name,
        "max_samples": args.max_samples,
        "splits": [],
    }
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        try:
            split_summary = _inspect_split(args, split)
        except Exception as exc:
            split_summary = {"split": split, "error": repr(exc)}
        summary["splits"].append(split_summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    full_graph_found = any(split.get("full_graph", {}).get("found") for split in summary["splits"])
    print(
        "GOFA graph metadata inspection complete: "
        f"task={args.task_name}, full_graph_found={full_graph_found}, output_path={output_path}"
    )
    if not full_graph_found:
        print("no full graph degree metadata found")
    for split in summary["splits"]:
        if split.get("error"):
            print(f"split={split.get('split')} error={split['error']}")
            continue
        full_graph = split.get("full_graph", {})
        print(
            f"split={split.get('split')} samples={split.get('sample_count')} "
            f"full_graph_found={full_graph.get('found')} "
            f"full_graph_object={split.get('full_graph_candidate')}"
        )
        for sample in split.get("sample_debug", [])[:3]:
            print(
                "sample_debug: "
                f"split={split.get('split')}, sample_idx={sample.get('sample_idx')}, "
                f"target_index={sample.get('target_index')}, "
                f"local_node_count={sample.get('local_node_count')}, "
                f"edge_count={sample.get('edge_count')}, "
                f"target_local_degree={sample.get('target_local_degree')}, "
                f"target_1hop_neighbors={sample.get('target_1hop_neighbors')}"
            )


if __name__ == "__main__":
    main()
