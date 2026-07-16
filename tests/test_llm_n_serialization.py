from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llm_n.serialization import (
    LabelLeakageError,
    serialize_taglas_sample,
)


TASK_LABELS = {
    "cora_node": ("Neural_Networks", "Rule_Learning"),
    "cora_link": ("Yes", "No"),
    "pubmed_node": ("Type 1", "Experimental"),
    "pubmed_link": ("Yes", "No"),
    "arxiv": ("cs.LG", "cs.AI"),
    "wikics": ("computer science", "artificial intelligence"),
    "wn18rr": ("_hypernym", "_also_see"),
}

LINK_TASKS = frozenset({"cora_link", "pubmed_link", "wn18rr"})


def synthetic_taglas_sample(
    task_name: str,
    sample_index: int = 0,
    *,
    edge_permutation: tuple[int, ...] | None = None,
    leaked_direction: str | None = None,
) -> SimpleNamespace:
    """Build the subset of TAGData fields consumed by the serializer."""

    is_link = task_name in LINK_TASKS
    targets = (2, 0) if is_link else (1,)
    if is_link:
        pairs = [(2, 1), (1, 0), (0, 3), (3, 1)]
    else:
        pairs = [(3, 2), (1, 3), (2, 1), (0, 2)]

    if leaked_direction is not None:
        if not is_link:
            raise ValueError("Only link tasks can contain a leaked target edge")
        if leaked_direction == "forward":
            pairs.insert(0, (targets[0], targets[1]))
        elif leaked_direction == "reverse":
            pairs.insert(0, (targets[1], targets[0]))
        else:
            raise ValueError(leaked_direction)

    relation_prefix = "relation" if task_name == "wn18rr" else "edge"
    relations = [f"{relation_prefix} {source}-{target}" for source, target in pairs]
    if edge_permutation is not None:
        if sorted(edge_permutation) != list(range(len(pairs))):
            raise ValueError("edge_permutation must permute every edge exactly once")
        pairs = [pairs[index] for index in edge_permutation]
        relations = [relations[index] for index in edge_permutation]

    rows = [source for source, _ in pairs]
    columns = [target for _, target in pairs]
    question_targets = " and ".join(f"[NODE_INDEX {index}]" for index in targets)
    label = TASK_LABELS[task_name][sample_index]
    return SimpleNamespace(
        # Deliberately non-monotonic global IDs exercise deterministic local ordering.
        node_map=torch.tensor([30, 10, 20, 40], dtype=torch.long),
        x=[
            f"{task_name} synthetic node zero for sample {sample_index}",
            f"{task_name} synthetic node one for sample {sample_index}",
            f"{task_name} synthetic node two for sample {sample_index}",
            f"{task_name} synthetic node three for sample {sample_index}",
        ],
        edge_index=torch.tensor([rows, columns], dtype=torch.long),
        edge_map=torch.arange(len(pairs), dtype=torch.long),
        edge_attr=relations,
        target_index=torch.tensor(targets, dtype=torch.long),
        question=[f"What is the answer for {question_targets}?"],
        answer=[f"{label}."],
        label=[label],
    )


@pytest.mark.parametrize("task_name", tuple(TASK_LABELS))
def test_each_supported_required_task_serializes_two_samples(task_name: str) -> None:
    for sample_index, expected_label in enumerate(TASK_LABELS[task_name]):
        serialized = serialize_taglas_sample(
            synthetic_taglas_sample(task_name, sample_index),
            task_name,
            sample_index=sample_index,
        )

        assert serialized.task_name == task_name
        assert serialized.sample_index == sample_index
        assert serialized.target_label == expected_label
        assert serialized.prompt.endswith("\nAnswer:")
        assert f"\n{serialized.answer}" not in serialized.prompt
        assert "Task Description:\n" in serialized.prompt
        assert "\nTarget:\n" in serialized.prompt
        assert "\nNodes:\n" in serialized.prompt
        assert "\nEdges:\n" in serialized.prompt
        assert "\nQuestion:\n" in serialized.prompt
        assert "[NODE_INDEX" not in serialized.prompt
        assert serialized.target_node_ids[0] == "[Node A]"
        if task_name in LINK_TASKS:
            assert serialized.target_node_ids == ("[Node A]", "[Node B]")


@pytest.mark.parametrize("task_name", ("cora_link", "pubmed_link", "wn18rr"))
@pytest.mark.parametrize("leaked_direction", ("forward", "reverse"))
def test_target_edge_leakage_raises_for_both_directions(
    task_name: str,
    leaked_direction: str,
) -> None:
    data = synthetic_taglas_sample(
        task_name,
        leaked_direction=leaked_direction,
    )

    with pytest.raises(LabelLeakageError, match="Target-edge leakage") as error:
        serialize_taglas_sample(data, task_name)

    assert isinstance(error.value, AssertionError)
    assert task_name in str(error.value)


@pytest.mark.parametrize("task_name", tuple(TASK_LABELS))
def test_serialization_is_independent_of_input_edge_order(task_name: str) -> None:
    original = synthetic_taglas_sample(task_name)
    permuted = synthetic_taglas_sample(task_name, edge_permutation=(3, 0, 2, 1))

    first = serialize_taglas_sample(original, task_name)
    second = serialize_taglas_sample(permuted, task_name)

    assert first.prompt == second.prompt
    assert first.edges == second.edges
    assert first.local_node_ids == second.local_node_ids


def test_target_first_then_global_id_node_order_is_stable() -> None:
    serialized = serialize_taglas_sample(
        synthetic_taglas_sample("cora_node"),
        "cora_node",
    )

    # Target local index 1 comes first. Remaining global IDs are 20, 30, 40.
    expected_node_fragments = (
        "- [Node A]: cora_node synthetic node one",
        "- [Node B]: cora_node synthetic node two",
        "- [Node C]: cora_node synthetic node zero",
        "- [Node D]: cora_node synthetic node three",
    )
    positions = [serialized.prompt.index(fragment) for fragment in expected_node_fragments]
    assert positions == sorted(positions)
