from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Set

import torch


BASE_ONLY = "base_only"
BASE_DELTA = "base_delta"


def compute_static_tiers(
    degrees: Sequence[float],
    high_ratio: float = 0.10,
    mid_ratio: float = 0.40,
) -> List[str]:
    if len(degrees) == 0:
        return []
    high_ratio = max(float(high_ratio), 0.0)
    mid_ratio = max(float(mid_ratio), 0.0)
    n_items = len(degrees)
    n_high = min(int(math.ceil(n_items * high_ratio)) if high_ratio > 0 else 0, n_items)
    n_mid = min(int(math.ceil(n_items * mid_ratio)) if mid_ratio > 0 else 0, max(n_items - n_high, 0))
    tiers = ["low"] * n_items
    order = sorted(range(n_items), key=lambda idx: (float(degrees[idx]), -idx), reverse=True)
    for idx in order[:n_high]:
        tiers[idx] = "high"
    for idx in order[n_high:n_high + n_mid]:
        tiers[idx] = "mid"
    return tiers


def _tensor_to_int_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_tensor_to_int_list(item))
        return result
    return [int(value)]


def _local_to_cache_node_indices(graph, local_indices: Iterable[int]) -> Set[int]:
    mapped = set()
    node_map = getattr(graph, "node_map", None)
    node_map_cpu = node_map.detach().cpu() if isinstance(node_map, torch.Tensor) else None
    num_node_feat = int(getattr(graph, "num_node_feat", len(node_map_cpu) if node_map_cpu is not None else 0))
    for local_idx in local_indices:
        local_idx = int(local_idx)
        if local_idx < 0 or local_idx >= num_node_feat:
            continue
        if node_map_cpu is not None and local_idx < node_map_cpu.numel():
            cache_idx = int(node_map_cpu[local_idx].item())
        else:
            cache_idx = local_idx
        if 0 <= cache_idx < num_node_feat:
            mapped.add(cache_idx)
    return mapped


def _one_hop_neighbors(graph, local_indices: Iterable[int]) -> Set[int]:
    edge_index = getattr(graph, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
        return set()
    local_set = set(int(idx) for idx in local_indices)
    neighbors = set()
    edge_cpu = edge_index.detach().cpu()
    for src, dst in edge_cpu.t().tolist():
        src = int(src)
        dst = int(dst)
        if src in local_set:
            neighbors.add(dst)
        if dst in local_set:
            neighbors.add(src)
    num_node_feat = int(getattr(graph, "num_node_feat", 0))
    return {idx for idx in neighbors if 0 <= idx < num_node_feat}


def target_aware_cache_node_indices(graph, target_index=None, question_index=None) -> Set[int]:
    if graph is None:
        return set()
    if target_index is None and hasattr(graph, "target_index"):
        target_index = graph.target_index
    if question_index is None and hasattr(graph, "question_index"):
        question_index = graph.question_index

    target_local = set(_tensor_to_int_list(target_index))
    question_local = set(_tensor_to_int_list(question_index))
    important_local = set(target_local)
    important_local.update(_one_hop_neighbors(graph, target_local))

    if question_local:
        question_neighbors = _one_hop_neighbors(graph, question_local)
        important_local.update(idx for idx in question_neighbors if idx in target_local)

    return _local_to_cache_node_indices(graph, important_local)


def build_scheme_b_load_policy(
    graph,
    static_tiers: Optional[Sequence[Optional[str]]],
    num_items: int,
    target_index=None,
    question_index=None,
    target_aware_delta: bool = True,
) -> List[str]:
    policies = [BASE_ONLY] * int(num_items)
    num_node_feat = int(getattr(graph, "num_node_feat", num_items)) if graph is not None else int(num_items)
    static_tiers = list(static_tiers or [])

    for item_idx in range(min(num_node_feat, num_items)):
        tier = static_tiers[item_idx] if item_idx < len(static_tiers) else None
        if tier == "high":
            policies[item_idx] = BASE_DELTA

    if target_aware_delta:
        for cache_idx in target_aware_cache_node_indices(graph, target_index=target_index, question_index=question_index):
            if 0 <= cache_idx < min(num_node_feat, num_items):
                policies[cache_idx] = BASE_DELTA
    return policies


def summarize_load_policy(policies: Sequence[str]) -> dict:
    base_delta = sum(1 for policy in policies if policy == BASE_DELTA)
    total = len(policies)
    base_only = total - base_delta
    return {
        "total": total,
        "base_only": base_only,
        "base_delta": base_delta,
        "delta_load_ratio": (base_delta / total) if total else 0.0,
    }
