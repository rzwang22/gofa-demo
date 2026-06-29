from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    # Scheme-B cache item order follows the sampled subgraph local node order:
    # node items occupy [0, num_node_feat). graph.node_map may point to a
    # dataset/global id and must not be used as a cache-item index.
    mapped = set()
    num_node_feat = int(getattr(graph, "num_node_feat", 0))
    for local_idx in local_indices:
        local_idx = int(local_idx)
        if 0 <= local_idx < num_node_feat:
            mapped.add(local_idx)
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


def _edge_index_cpu(graph) -> Optional[torch.Tensor]:
    edge_index = getattr(graph, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
        return None
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return None
    return edge_index.detach().cpu()


def local_node_degrees(graph, num_nodes: Optional[int] = None) -> List[int]:
    if graph is None:
        return []
    num_node_feat = int(num_nodes if num_nodes is not None else getattr(graph, "num_node_feat", 0))
    degrees = [0] * max(num_node_feat, 0)
    edge_cpu = _edge_index_cpu(graph)
    if edge_cpu is None:
        return degrees
    for src, dst in edge_cpu.t().tolist():
        src = int(src)
        dst = int(dst)
        if 0 <= src < num_node_feat:
            degrees[src] += 1
        if 0 <= dst < num_node_feat and dst != src:
            degrees[dst] += 1
    return degrees


def local_neighbors_by_hop(graph, target_local_indices: Iterable[int], max_hops: int = 1) -> Dict[int, int]:
    if graph is None:
        return {}
    num_node_feat = int(getattr(graph, "num_node_feat", 0))
    targets = sorted({int(idx) for idx in target_local_indices if 0 <= int(idx) < num_node_feat})
    if not targets:
        return {}
    max_hops = max(int(max_hops), 0)
    adjacency = {idx: set() for idx in range(num_node_feat)}
    edge_cpu = _edge_index_cpu(graph)
    if edge_cpu is not None:
        for src, dst in edge_cpu.t().tolist():
            src = int(src)
            dst = int(dst)
            if 0 <= src < num_node_feat and 0 <= dst < num_node_feat:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
    distances = {idx: 0 for idx in targets}
    queue = deque(targets)
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        if current_distance >= max_hops:
            continue
        for neighbor in adjacency.get(current, ()):
            if neighbor in distances:
                continue
            distances[neighbor] = current_distance + 1
            queue.append(neighbor)
    return distances


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


def _target_local_indices(graph, target_index=None) -> Set[int]:
    if graph is None:
        return set()
    if target_index is None and hasattr(graph, "target_index"):
        target_index = graph.target_index
    num_node_feat = int(getattr(graph, "num_node_feat", 0))
    return {idx for idx in _tensor_to_int_list(target_index) if 0 <= idx < num_node_feat}


def _edge_item_index(graph, edge_pos: int, num_node_feat: int, num_items: int) -> Optional[int]:
    edge_map = getattr(graph, "edge_map", None)
    edge_map_cpu = edge_map.detach().cpu() if isinstance(edge_map, torch.Tensor) else None
    if edge_map_cpu is not None and edge_pos < edge_map_cpu.numel():
        edge_offset = int(edge_map_cpu[edge_pos].item())
    else:
        edge_offset = int(edge_pos)
    item_idx = int(num_node_feat) + edge_offset
    if num_node_feat <= item_idx < num_items:
        return item_idx
    return None


def edge_item_indices_touching_nodes(graph, local_node_indices: Iterable[int], num_items: int) -> Set[int]:
    if graph is None:
        return set()
    edge_cpu = _edge_index_cpu(graph)
    if edge_cpu is None:
        return set()
    num_node_feat = int(getattr(graph, "num_node_feat", 0))
    node_set = {int(idx) for idx in local_node_indices if 0 <= int(idx) < num_node_feat}
    if not node_set:
        return set()
    selected = set()
    for edge_pos, (src, dst) in enumerate(edge_cpu.t().tolist()):
        if int(src) not in node_set and int(dst) not in node_set:
            continue
        item_idx = _edge_item_index(graph, edge_pos, num_node_feat, int(num_items))
        if item_idx is not None:
            selected.add(item_idx)
    return selected


def _local_degree_top_nodes(
        graph,
        ratio: float = 0.0,
        threshold: Optional[float] = None,
        num_nodes: Optional[int] = None) -> Set[int]:
    num_node_feat = int(num_nodes if num_nodes is not None else getattr(graph, "num_node_feat", 0))
    degrees = local_node_degrees(graph, num_nodes=num_node_feat)
    selected = set()
    if threshold is not None:
        selected.update(idx for idx, degree in enumerate(degrees) if float(degree) >= float(threshold))
    ratio = max(float(ratio or 0.0), 0.0)
    if ratio > 0.0 and num_node_feat > 0:
        k = min(int(math.ceil(num_node_feat * ratio)), num_node_feat)
        order = sorted(range(num_node_feat), key=lambda idx: (degrees[idx], -idx), reverse=True)
        selected.update(order[:k])
    return selected


def build_scheme_b_load_policy(
    graph,
    static_tiers: Optional[Sequence[Optional[str]]],
    num_items: int,
    target_index=None,
    question_index=None,
    target_aware_delta: bool = True,
    target_aware_policy: str = "target_1hop",
    target_delta_hops: int = 1,
    keep_target_edges: bool = True,
    local_degree_top_ratio: float = 0.0,
    local_degree_threshold: Optional[float] = None,
    max_delta_items_per_batch: Optional[int] = None,
    return_details: bool = False,
):
    policies = [BASE_ONLY] * int(num_items)
    num_node_feat = int(getattr(graph, "num_node_feat", num_items)) if graph is not None else int(num_items)
    num_node_feat = min(num_node_feat, int(num_items))
    target_aware_policy = str(target_aware_policy or "target_1hop")
    selected_target_nodes = set()
    selected_hop_nodes = set()
    selected_local_degree_nodes = set()
    selected_edges = set()

    if target_aware_delta:
        if target_aware_policy not in {"target_only", "target_1hop", "local_degree_top", "target_1hop_local_degree", "all_delta"}:
            raise ValueError(
                "scheme_b_quant.target_aware_policy must be one of "
                "target_only, target_1hop, local_degree_top, target_1hop_local_degree, all_delta."
            )
        if target_aware_policy == "all_delta":
            for item_idx in range(int(num_items)):
                policies[item_idx] = BASE_DELTA
            selected_target_nodes = _target_local_indices(graph, target_index=target_index)
            selected_hop_nodes = set(range(num_node_feat))
            selected_edges = set(range(num_node_feat, int(num_items)))
        else:
            target_nodes = _target_local_indices(graph, target_index=target_index)
            selected_target_nodes = _local_to_cache_node_indices(graph, target_nodes)
            if target_aware_policy in {"target_only", "target_1hop", "target_1hop_local_degree"}:
                selected_hop_nodes.update(selected_target_nodes)
            if target_aware_policy in {"target_1hop", "target_1hop_local_degree"}:
                hop_distances = local_neighbors_by_hop(graph, selected_target_nodes, max_hops=target_delta_hops)
                selected_hop_nodes.update(
                    idx for idx, distance in hop_distances.items()
                    if 0 <= idx < num_node_feat and distance <= int(target_delta_hops)
                )
            if target_aware_policy in {"local_degree_top", "target_1hop_local_degree"}:
                selected_local_degree_nodes = _local_degree_top_nodes(
                    graph,
                    ratio=local_degree_top_ratio,
                    threshold=local_degree_threshold,
                    num_nodes=num_node_feat,
                )
            node_candidates = set(selected_hop_nodes) | set(selected_local_degree_nodes)
            if target_aware_policy == "target_only":
                node_candidates = set(selected_target_nodes)
            for cache_idx in node_candidates:
                if 0 <= cache_idx < num_node_feat:
                    policies[cache_idx] = BASE_DELTA
            if keep_target_edges:
                edge_anchor_nodes = node_candidates if target_aware_policy != "target_only" else selected_target_nodes
                selected_edges = edge_item_indices_touching_nodes(graph, edge_anchor_nodes, int(num_items))
                for item_idx in selected_edges:
                    policies[item_idx] = BASE_DELTA

    if max_delta_items_per_batch is not None:
        max_delta = max(int(max_delta_items_per_batch), 0)
        delta_items = [idx for idx, policy in enumerate(policies) if policy == BASE_DELTA]
        if len(delta_items) > max_delta:
            target_priority = set(selected_target_nodes)
            edge_priority = set(selected_edges)
            def _priority(idx: int) -> Tuple[int, int]:
                if idx in target_priority:
                    return (0, idx)
                if idx < num_node_feat:
                    return (1, idx)
                if idx in edge_priority:
                    return (2, idx)
                return (3, idx)
            keep = set(sorted(delta_items, key=_priority)[:max_delta])
            policies = [BASE_DELTA if idx in keep else BASE_ONLY for idx in range(len(policies))]
            selected_target_nodes &= keep
            selected_hop_nodes &= keep
            selected_local_degree_nodes &= keep
            selected_edges &= keep

    details = {
        "target_node_indices": sorted(selected_target_nodes),
        "one_hop_node_indices": sorted(idx for idx in selected_hop_nodes if idx not in selected_target_nodes),
        "target_or_hop_node_indices": sorted(selected_hop_nodes),
        "local_degree_node_indices": sorted(selected_local_degree_nodes),
        "edge_item_indices": sorted(selected_edges),
        "selected_edge_count": len(selected_edges),
        "local_degrees": local_node_degrees(graph, num_nodes=num_node_feat) if graph is not None else [],
        "target_aware_policy": target_aware_policy,
        "target_delta_hops": int(target_delta_hops),
        "keep_target_edges": bool(keep_target_edges),
        "local_degree_top_ratio": float(local_degree_top_ratio or 0.0),
        "local_degree_threshold": local_degree_threshold,
        "max_delta_items_per_batch": max_delta_items_per_batch,
    }
    if return_details:
        return policies, details
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
