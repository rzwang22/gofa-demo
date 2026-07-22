from __future__ import annotations

import functools
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import llm_n.runner as runner
from llm_n.data import LLMNTaskDataset, _prepare_llm_n_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TASKS = {
    "cora_node",
    "cora_link",
    "pubmed_node",
    "pubmed_link",
    "arxiv",
    "wikics",
    "wn18rr",
}

DATASET_LABELS = {
    "cora_node": (
        "Case_Based",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Reinforcement_Learning",
        "Rule_Learning",
        "Theory",
        "No",
        "Yes",
    ),
    "cora_link": (
        "Case_Based",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Reinforcement_Learning",
        "Rule_Learning",
        "Theory",
        "No",
        "Yes",
    ),
    "pubmed_node": ("Experimental", "Type 1", "Type 2", "No", "Yes"),
    "pubmed_link": ("Experimental", "Type 1", "Type 2", "No", "Yes"),
    "arxiv": ("cs.AI", "cs.LG", "cs.RO", "No", "Yes"),
    "wikics": ("Computational linguistics", "Databases"),
    "wn18rr": ("_hypernym", "_also_see"),
    "fb15k237": ("/people/person/nationality", "/location/contains"),
    "products": ("Books", "Electronics"),
}

EXPECTED_LABEL_SPACES = {
    "cora_node": DATASET_LABELS["cora_node"][:-2],
    "cora_link": ("No", "Yes"),
    "pubmed_node": ("Experimental", "Type 1", "Type 2"),
    "pubmed_link": ("No", "Yes"),
    "arxiv": ("cs.AI", "cs.LG", "cs.RO"),
    "wikics": DATASET_LABELS["wikics"],
    "wn18rr": DATASET_LABELS["wn18rr"],
    "fb15k237": DATASET_LABELS["fb15k237"],
    "products": DATASET_LABELS["products"],
}


class _FakeTask:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.dataset = SimpleNamespace(label=list(labels))

    def __len__(self) -> int:
        return 0


@pytest.fixture
def fake_taglas(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    module = types.ModuleType("TAGLAS")

    def get_task(**kwargs: Any) -> _FakeTask:
        calls.append(kwargs)
        return _FakeTask(DATASET_LABELS[kwargs["name"]])

    module.get_task = get_task  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "TAGLAS", module)
    return calls


@pytest.mark.parametrize(
    "task_name",
    (
        "cora_node",
        "cora_link",
        "pubmed_node",
        "pubmed_link",
        "arxiv",
        "wikics",
        "wn18rr",
        "fb15k237",
        "products",
    ),
)
def test_dataset_adapter_reuses_taglas_qa_and_task_specific_labels(
    task_name: str,
    fake_taglas: list[dict[str, Any]],
) -> None:
    dataset = LLMNTaskDataset(
        task_name=task_name,
        root="/tmp/fake-taglas",
        split="val",
    )

    assert dataset.label_space == EXPECTED_LABEL_SPACES[task_name]
    assert len(fake_taglas) == 1
    call = fake_taglas[0]
    assert call["name"] == task_name
    assert call["task_type"] == "QA"
    assert call["root"] == "/tmp/fake-taglas"
    assert call["split"] == "val"
    assert call["hop"] == 3
    assert call["max_nodes_per_hop"] == 5
    assert call["from_saved"] is True
    assert call["save_data"] is True
    assert not any("negative" in key.lower() for key in call)

    post_funcs = call["post_funcs"]
    assert len(post_funcs) == 1
    prompt_func = post_funcs[0]
    assert isinstance(prompt_func, functools.partial)
    assert prompt_func.func is _prepare_llm_n_prompt
    assert prompt_func.keywords == {"task_name": task_name}
    assert prompt_func.func.__name__ != "build_GOFA_task_graph"


def test_runner_passes_gofa_size_filter_only_to_sft_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class CapturingDataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(runner, "LLMNTaskDataset", CapturingDataset)
    config = {
        "hops": 3,
        "max_nodes_per_hop": 5,
        "sample_size_per_task": 2,
        "inf_sample_size_per_task": 2,
        "gofa_size_filter": True,
    }

    runner._build_datasets(config, ["arxiv"], split="train", inference=False)
    runner._build_datasets(config, ["arxiv"], split="test", inference=True)

    assert calls[0]["filter_func"] is runner.gofa_data_size_filter
    assert calls[1]["filter_func"] is None


def _load_yaml(relative_path: str) -> dict[str, Any]:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_train_and_inference_configs_keep_llm_n_baseline_defaults() -> None:
    train = _load_yaml("configs/llm_n_train_config.yaml")
    inference = _load_yaml("configs/llm_n_inference_config.yaml")

    for config in (train, inference):
        assert config["model_type"] == "llm_n"
        assert config["llm_n_enabled"] is True
        assert config["model_name_or_path"] == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config["batch_size"] == 1
        assert config["truncate_mode"] == "none"

    assert train["llm_n_mode"] == "llm_n_sft"
    assert train["run_mode"] == "train"
    assert set(train["train_task_names"]) == REQUIRED_TASKS
    assert train["hops"] == 3
    assert train["max_nodes_per_hop"] == 5
    assert train["gofa_size_filter"] is True
    assert train["resume_from_checkpoint"] is None
    assert train["lora_r"] > 0
    assert train["lora_alpha"] > 0
    assert 0.0 <= train["lora_dropout"] < 1.0
    assert set(train["lora_target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    assert inference["llm_n_mode"] == "llm_n_zero_shot"
    assert inference["run_mode"] == "inference"
    assert set(inference["eval_task_names"]) == REQUIRED_TASKS
    assert inference["inf_hops"] == 3
    assert inference["inf_max_nodes_per_hops"] == 5
    assert inference["warm_up_samples"] >= 10
