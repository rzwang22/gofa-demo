"""TAGLAS dataset adapter for the LLM-N baseline.

The adapter intentionally calls TAGLAS ``get_task(..., task_type="QA")`` with
the same split and sampling arguments as GOFA.  The only changed post-process is
the representation: GOFA's random node IDs and prompt graph are not added.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterator, Sequence

from .serialization import (
    TASK_DESCRIPTIONS,
    SerializedGraphSample,
    serialize_taglas_sample,
)


def gofa_data_size_filter(data: Any, **_: Any) -> Any | None:
    """The exact supervised-training sample filter used by ``run_gofa.py``."""
    import torch

    estimated_mem = (
        24.495
        + 0.4645 * len(data.node_map)
        + 0.0042 * len(torch.unique(data.node_map))
        + 0.1689 * len(data.edge_map)
        + 0.2846 * len(torch.unique(data.edge_map))
    )
    if len(data.node_map) + len(torch.unique(data.edge_map)) < 40 and estimated_mem < 65:
        return data
    return None


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _prepare_llm_n_prompt(data: Any, task_class: Any, task_name: str) -> Any:
    """Apply GOFA's supervised prompt without choices, then split graph context.

    ``build_finetune_task_prompt`` is the exact prompt hook used by
    ``GOFAFineTuneTaskWrapper``.  ``selection=False`` and ``instruction=False``
    match the supervised setting and ensure no candidate label list or
    label-description examples are exposed to the predictor.
    """
    from tasks.build_prompt import build_finetune_task_prompt

    data = build_finetune_task_prompt(
        data,
        task_class=task_class,
        task_name=task_name,
        way=-1,
        selection=False,
        instruction=False,
    )
    graph_description = str(getattr(task_class.dataset, "graph_description", "")).strip()
    questions: list[str] = []
    for question in _text_list(data.question):
        question = question.strip()
        if graph_description and question.startswith(graph_description):
            question = question[len(graph_description):].lstrip()
        if "choose from the following:" in question.lower():
            raise AssertionError("LLM-N supervised prompts must not contain candidate label lists")
        questions.append(question)
    data.question = questions
    description_parts = [part for part in (graph_description, TASK_DESCRIPTIONS[task_name]) if part]
    data.llm_n_task_description = " ".join(description_parts)
    return data


class LLMNTaskDataset:
    """A deterministic serialized view of one TAGLAS QA task."""

    def __init__(
        self,
        task_name: str,
        root: str = "TAGDataset",
        split: str = "train",
        hop: int = 3,
        max_nodes_per_hop: int = 5,
        sample_size: float | int | list[int] = 1.0,
        sample_mode: str = "random",
        num_workers: int = 0,
        save_data: bool = True,
        from_saved: bool = True,
        save_name: str | None = None,
        filter_func: Callable[[Any], Any] | None = None,
        **taglas_kwargs: Any,
    ) -> None:
        # Lazy import keeps serialization/unit tests independent from the
        # optional TAGLAS + PyG runtime.
        from TAGLAS import get_task

        if "post_funcs" in taglas_kwargs:
            raise ValueError("LLMNTaskDataset owns post_funcs so GOFA graph rewriting cannot be injected")
        self.task_name = task_name
        self.split = split
        self.hop = int(hop)
        self.max_nodes_per_hop = int(max_nodes_per_hop)
        prompt_func = partial(_prepare_llm_n_prompt, task_name=task_name)
        self.taglas_task = get_task(
            name=task_name,
            task_type="QA",
            root=root,
            split=split,
            save_data=save_data,
            from_saved=from_saved,
            save_name=save_name,
            post_funcs=[prompt_func],
            filter_func=filter_func,
            sample_size=sample_size,
            sample_mode=sample_mode,
            hop=self.hop,
            max_nodes_per_hop=self.max_nodes_per_hop,
            num_workers=num_workers,
            **taglas_kwargs,
        )
        labels = _text_list(getattr(self.taglas_task.dataset, "label", None))
        if task_name in {"cora_node", "pubmed_node", "arxiv"}:
            labels = labels[:-2]
        elif task_name in {"cora_link", "pubmed_link"}:
            labels = labels[-2:]
        self.label_space = tuple(labels)

    def __len__(self) -> int:
        return len(self.taglas_task)

    def __getitem__(self, index: int) -> SerializedGraphSample:
        data = self.taglas_task[index]
        return serialize_taglas_sample(data, self.task_name, sample_index=int(index))


class ConcatSerializedDataset:
    """Small dependency-free equivalent of torch ``ConcatDataset``."""

    def __init__(self, datasets: Sequence[LLMNTaskDataset]) -> None:
        if not datasets:
            raise ValueError("At least one LLM-N task dataset is required")
        self.datasets = list(datasets)
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> SerializedGraphSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        previous = 0
        for dataset, cumulative in zip(self.datasets, self.cumulative_sizes):
            if index < cumulative:
                return dataset[index - previous]
            previous = cumulative
        raise IndexError(index)

    def __iter__(self) -> Iterator[SerializedGraphSample]:
        for index in range(len(self)):
            yield self[index]
