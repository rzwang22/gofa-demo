"""Deterministic text serialization for sampled TAGLAS QA subgraphs.

This module deliberately knows nothing about GOFA's GNN, ICAE compressor, or
memory-token representation.  It consumes the ordinary ``TAGData`` fields
returned by a TAGLAS QA task and emits one causal-LM prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_TASKS = (
    "cora_node",
    "cora_link",
    "pubmed_node",
    "pubmed_link",
    "arxiv",
    "wikics",
    "wn18rr",
    "fb15k237",
    "products",
)

TARGET_EDGE_MASK_TASKS = frozenset({"cora_link", "pubmed_link", "wn18rr", "fb15k237"})

TASK_DESCRIPTIONS: Mapping[str, str] = {
    "cora_node": (
        "Predict the category of the target paper from its text and the sampled "
        "Cora co-citation subgraph. Generate the category name directly."
    ),
    "cora_link": (
        "Predict whether the two target papers are co-cited from their text and "
        "the sampled Cora subgraph. Generate Yes or No directly."
    ),
    "pubmed_node": (
        "Predict the category of the target paper from its text and the sampled "
        "PubMed citation subgraph. Generate the category name directly."
    ),
    "pubmed_link": (
        "Predict whether the two target papers are co-cited from their text and "
        "the sampled PubMed subgraph. Generate Yes or No directly."
    ),
    "arxiv": (
        "Predict the arXiv category of the target paper from its text and the "
        "sampled citation subgraph. Generate the category name directly."
    ),
    "wikics": (
        "Predict the category of the target Wikipedia term from its text and the "
        "sampled WikiCS subgraph. Generate the category name directly."
    ),
    "wn18rr": (
        "Predict the canonical relation between the two target WordNet entities "
        "from their text and the sampled relational subgraph. Generate the relation name directly."
    ),
    "fb15k237": (
        "Predict the canonical relation between the two target entities from "
        "their text and the sampled knowledge-graph subgraph. Generate the relation name directly."
    ),
    "products": (
        "Predict the category of the target product from its text and the sampled "
        "product co-purchasing subgraph. Generate the category name directly."
    ),
}


class LabelLeakageError(AssertionError):
    """Raised when a target link/relation is still present in model input."""


@dataclass(frozen=True)
class SerializedEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class SerializedGraphSample:
    """One serialized QA example plus metadata that is never shown to the LLM."""

    task_name: str
    sample_index: int
    prompt: str
    answer: str
    target_label: str
    target_node_ids: tuple[str, ...]
    local_node_ids: tuple[str, ...]
    edges: tuple[SerializedEdge, ...] = field(default_factory=tuple)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _flatten_indices(value: Any) -> list[int]:
    values = _as_list(value)
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = list(values[0])
    result: list[int] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            result.extend(int(v) for v in item)
        else:
            result.append(int(item))
    return result


def _edge_pairs(edge_index: Any) -> list[tuple[int, int]]:
    values = _as_list(edge_index)
    if not values:
        return []
    if len(values) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    rows, cols = _as_list(values[0]), _as_list(values[1])
    if len(rows) != len(cols):
        raise ValueError("edge_index rows and columns must have equal length")
    return [(int(src), int(dst)) for src, dst in zip(rows, cols)]


def _clean_inline_text(value: Any) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def _canonical_target(answer: str) -> str:
    answer = _clean_inline_text(answer)
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer.rstrip(" \t\r\n.!?")


def _stable_value_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def node_id_from_rank(rank: int) -> str:
    """Return Excel-like deterministic IDs: A..Z, AA..AZ, BA..."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    chars: list[str] = []
    value = rank + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return f"[Node {''.join(reversed(chars))}]"


def _feature_per_item(features: Any, feature_map: Any, count: int, default: str) -> list[str]:
    values = _as_list(features)
    mapping = _flatten_indices(feature_map)
    if len(values) == count:
        return [_clean_inline_text(value) for value in values]
    if mapping and len(mapping) == count and values and max(mapping, default=-1) < len(values):
        return [_clean_inline_text(values[index]) for index in mapping]
    if not values:
        return [default for _ in range(count)]
    raise ValueError(
        f"Cannot align {len(values)} text features to {count} graph items "
        f"with a mapping of length {len(mapping)}"
    )


def assert_target_edge_absent(data: Any, task_name: str) -> None:
    """Assert TAGLAS removed the held-out target pair from a link QA subgraph.

    TAGLAS ``LQATask`` removes both directions of the target pair after sampling.
    Keeping this independent assertion at the LLM serialization boundary catches
    stale task caches or upstream regressions before any target relation reaches
    the language model.
    """
    if task_name not in TARGET_EDGE_MASK_TASKS:
        return
    targets = _flatten_indices(getattr(data, "target_index", None))
    if len(targets) != 2:
        raise LabelLeakageError(
            f"{task_name} requires exactly two target nodes, found {targets!r}"
        )
    source_target = (targets[0], targets[1])
    target_source = (targets[1], targets[0])
    for edge_position, pair in enumerate(_edge_pairs(getattr(data, "edge_index", None))):
        if pair == source_target or pair == target_source:
            relation_texts = _feature_per_item(
                getattr(data, "edge_attr", None),
                getattr(data, "edge_map", None),
                len(_edge_pairs(getattr(data, "edge_index", None))),
                "connected",
            )
            relation = relation_texts[edge_position]
            raise LabelLeakageError(
                f"Target-edge leakage in {task_name}: local edge {pair} at position "
                f"{edge_position} has relation text {relation!r}. TAGLAS LQATask "
                "must remove the target pair before LLM-N serialization."
            )


def _replace_node_placeholders(text: str, local_ids: Mapping[int, str]) -> str:
    pattern = re.compile(r"\[NODE_INDEX\s+(\d+)\]")

    def replacement(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return local_ids.get(index, match.group(0))

    return pattern.sub(replacement, text)


def _node_order(data: Any, num_nodes: int, targets: Sequence[int]) -> list[int]:
    global_node_map = _as_list(getattr(data, "node_map", None))
    if len(global_node_map) != num_nodes:
        global_node_map = list(range(num_nodes))

    ordered_targets: list[int] = []
    for target in targets:
        if target < 0 or target >= num_nodes:
            raise ValueError(f"target_index {target} is outside a {num_nodes}-node subgraph")
        if target not in ordered_targets:
            ordered_targets.append(target)

    target_set = set(ordered_targets)
    remaining = [index for index in range(num_nodes) if index not in target_set]
    remaining.sort(key=lambda index: (_stable_value_key(global_node_map[index]), index))
    return ordered_targets + remaining


def _first_text(value: Any, field_name: str) -> str:
    values = _as_list(value)
    if not values:
        raise ValueError(f"TAGLAS sample has no {field_name} text")
    return _clean_inline_text(values[0])


def serialize_taglas_sample(
    data: Any,
    task_name: str,
    sample_index: int = 0,
    task_description: str | None = None,
) -> SerializedGraphSample:
    """Serialize one already-sampled TAGLAS QA subgraph deterministically."""
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported LLM-N task {task_name!r}; supported tasks: {', '.join(SUPPORTED_TASKS)}"
        )

    assert_target_edge_absent(data, task_name)

    node_map = _as_list(getattr(data, "node_map", None))
    raw_node_texts = _as_list(getattr(data, "x", None))
    num_nodes = len(node_map) if node_map else len(raw_node_texts)
    if num_nodes == 0:
        raise ValueError("TAGLAS sample contains no nodes")
    node_texts = _feature_per_item(
        getattr(data, "x", None), getattr(data, "node_map", None), num_nodes, "(no node text)"
    )

    targets = _flatten_indices(getattr(data, "target_index", None))
    order = _node_order(data, num_nodes, targets)
    local_ids = {original_index: node_id_from_rank(rank) for rank, original_index in enumerate(order)}

    node_lines = [f"- {local_ids[index]}: {node_texts[index]}" for index in order]

    pairs = _edge_pairs(getattr(data, "edge_index", None))
    relation_texts = _feature_per_item(
        getattr(data, "edge_attr", None),
        getattr(data, "edge_map", None),
        len(pairs),
        "connected",
    )
    edge_rows: list[tuple[int, int, str, int]] = []
    rank_by_index = {node_index: rank for rank, node_index in enumerate(order)}
    for original_position, ((source, target), relation) in enumerate(zip(pairs, relation_texts)):
        if source not in local_ids or target not in local_ids:
            raise ValueError(f"edge {(source, target)} references a node outside the sampled subgraph")
        edge_rows.append((rank_by_index[source], rank_by_index[target], relation, original_position))
    edge_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    serialized_edges = tuple(
        SerializedEdge(node_id_from_rank(source), node_id_from_rank(target), relation)
        for source, target, relation, _ in edge_rows
    )
    if task_name in TARGET_EDGE_MASK_TASKS:
        target_pair = frozenset(local_ids[index] for index in targets)
        for edge in serialized_edges:
            if frozenset((edge.source, edge.target)) == target_pair:
                raise LabelLeakageError(
                    f"Target-edge leakage remained after {task_name} serialization: {edge!r}"
                )
    edge_lines = [
        f"- {edge.source} -> {edge.target}: {edge.relation}" for edge in serialized_edges
    ] or ["- (none)"]

    question = _replace_node_placeholders(
        _first_text(getattr(data, "question", None), "question"), local_ids
    )
    answer = _replace_node_placeholders(
        _first_text(getattr(data, "answer", None), "answer"), local_ids
    )
    description = task_description or getattr(data, "llm_n_task_description", None)
    if description is None:
        description = TASK_DESCRIPTIONS[task_name]
    description = _clean_inline_text(description)

    target_node_ids = tuple(local_ids[index] for index in targets)
    target_text = " and ".join(target_node_ids)
    prompt = "\n".join(
        [
            "Task Description:",
            description,
            "",
            "Target:",
            target_text,
            "",
            "Nodes:",
            *node_lines,
            "",
            "Edges:",
            *edge_lines,
            "",
            "Question:",
            question,
            "",
            "Answer:",
        ]
    )

    return SerializedGraphSample(
        task_name=task_name,
        sample_index=int(sample_index),
        prompt=prompt,
        answer=answer,
        target_label=_canonical_target(answer),
        target_node_ids=target_node_ids,
        local_node_ids=tuple(local_ids[index] for index in order),
        edges=serialized_edges,
    )
