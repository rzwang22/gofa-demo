from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import llm_n.runner as runner
from llm_n.model import LLMNPredictor
from llm_n.resume import (
    CHECKPOINT_CONFIG_FILE,
    CHECKPOINT_MARKER,
    CHECKPOINT_STATE_FILE,
    ResumableRandomSampler,
    load_training_state,
    resolve_resume_checkpoint,
)
from llm_n.serialization import SerializedGraphSample


class SyntheticInterruption(RuntimeError):
    pass


class TinyResumeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, vocabulary: Mapping[str, int] | None = None) -> None:
        self.vocabulary = dict(vocabulary or {"<pad>": 0, "<s>": 1, "</s>": 2})

    @classmethod
    def from_checkpoint(cls, checkpoint: Path | None) -> "TinyResumeTokenizer":
        if checkpoint is None:
            return cls()
        with (checkpoint / "tiny_tokenizer.json").open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        output: list[int] = []
        for token in str(text).split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
            output.append(self.vocabulary[token])
        return output

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        return f"<s> [INST] {messages[0]['content']} [/INST]"

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "tiny_tokenizer.json").open("w", encoding="utf-8") as handle:
            json.dump(self.vocabulary, handle, sort_keys=True)
            handle.write("\n")
        (output_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class TinyResumeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.tensor(0.40))
        self.lora_B = torch.nn.Parameter(torch.tensor(-0.15))
        self.peft_config = {"default": SimpleNamespace(r=1)}

    @classmethod
    def from_checkpoint(cls, checkpoint: Path | None) -> "TinyResumeModel":
        model = cls()
        if checkpoint is not None:
            try:
                adapter = torch.load(
                    checkpoint / "adapter_model.bin",
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                adapter = torch.load(checkpoint / "adapter_model.bin", map_location="cpu")
            model.lora_A.data.copy_(adapter["lora_A"])
            model.lora_B.data.copy_(adapter["lora_B"])
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        active = attention_mask.float()
        token_signal = (input_ids.float() * active).sum() / active.sum().clamp_min(1.0)
        answer_signal = labels.ne(-100).float().mean()
        stochastic_signal = torch.rand((), device=input_ids.device) * 0.01
        prediction = self.base_weight + self.lora_A + self.lora_B * token_signal / 100.0
        target = answer_signal + stochastic_signal
        return SimpleNamespace(loss=(prediction - target).square())

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_A": self.lora_A.detach().cpu().clone(),
                "lora_B": self.lora_B.detach().cpu().clone(),
            },
            output_path / "adapter_model.bin",
        )
        (output_path / "adapter_config.json").write_text(
            '{"peft_type": "LORA"}\n',
            encoding="utf-8",
        )


class SyntheticResumeDataset:
    task_name = "cora_node"
    split = "train"
    label_space = ("Class_A", "Class_B")

    def __init__(self, size: int, visitation: list[int]) -> None:
        self.visitation = visitation
        self.samples = tuple(
            SerializedGraphSample(
                task_name=self.task_name,
                sample_index=index,
                prompt=(
                    "Task Description:\nSynthetic resume test.\n\n"
                    f"Target:\n[Node A]\n\nNodes:\n- [Node A]: sample-{index}\n\n"
                    "Edges:\n- (none)\n\nQuestion:\nWhich class?\n\nAnswer:"
                ),
                answer="Class_A" if index % 2 == 0 else "Class_B",
                target_label="Class_A" if index % 2 == 0 else "Class_B",
                target_node_ids=("[Node A]",),
                local_node_ids=("[Node A]",),
            )
            for index in range(size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SerializedGraphSample:
        self.visitation.append(index)
        return self.samples[index]


class TinyRuntime:
    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        self.visitation_logs: list[list[int]] = []
        self.models: list[TinyResumeModel] = []

    def build_datasets(
        self,
        config: Mapping[str, Any],
        tasks: list[str],
        split: str,
        inference: bool,
    ) -> list[SyntheticResumeDataset]:
        del config, tasks, split, inference
        visitation: list[int] = []
        self.visitation_logs.append(visitation)
        return [SyntheticResumeDataset(self.dataset_size, visitation)]

    def predictor_factory(
        self,
        cls: type[LLMNPredictor],
        config: Mapping[str, Any],
        for_training: bool = False,
    ) -> LLMNPredictor:
        del cls
        assert for_training
        checkpoint_value = config.get("_resolved_resume_checkpoint")
        checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
        model = TinyResumeModel.from_checkpoint(checkpoint)
        tokenizer = TinyResumeTokenizer.from_checkpoint(checkpoint)
        self.models.append(model)
        return LLMNPredictor(
            model,
            tokenizer,
            max_context_length=128,
            max_new_tokens=8,
            truncate_mode=str(config.get("truncate_mode", "none")),
        )


def base_config(output_dir: Path, *, dataset_size: int = 16) -> dict[str, Any]:
    return {
        "model_type": "llm_n",
        "llm_n_mode": "llm_n_sft",
        "run_mode": "train",
        "model_name_or_path": "tiny-resume-model",
        "train_task_names": ["cora_node"],
        "sample_size_per_task": [dataset_size],
        "data_root_path": ["synthetic"],
        "sample_mode": ["random"],
        "hops": [3],
        "max_nodes_per_hop": [5],
        "seed": 73,
        "batch_size": 2,
        "grad_acc_step": 2,
        "num_epochs": 1,
        "max_steps": 4,
        "lr": 0.04,
        "l2": 0.0,
        "warmup_ratio": 0.25,
        "lr_scheduler_type": "cosine",
        "training_precision": "fp32",
        "torch_dtype": "float32",
        "truncate_mode": "none",
        "max_context_length": 128,
        "gradient_checkpointing": False,
        "load_in_4bit": False,
        "lora_r": 1,
        "lora_alpha": 2,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj"],
        "gofa_size_filter": True,
        "dataloader_num_workers": 0,
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": 2,
        "save_total_limit": 4,
        "output_dir": str(output_dir),
        "run_name": "resume-run",
        "adapter_output_dir": None,
        "resume_from_checkpoint": None,
    }


def install_tiny_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: TinyRuntime,
) -> None:
    monkeypatch.setattr(runner, "_build_datasets", runtime.build_datasets)
    monkeypatch.setattr(
        runner.LLMNPredictor,
        "from_pretrained",
        classmethod(
            lambda cls, config, for_training=False: runtime.predictor_factory(
                cls,
                config,
                for_training,
            )
        ),
    )


def read_adapter(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path / "adapter_model.bin", map_location="cpu")


def assert_nested_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    elif isinstance(first, Mapping):
        assert isinstance(second, Mapping)
        assert set(first) == set(second)
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            assert_nested_equal(first_item, second_item)
    elif isinstance(first, float):
        assert second == pytest.approx(first, rel=0.0, abs=0.0)
    else:
        assert first == second


def run_with_interruption_at_step_two(
    config: dict[str, Any],
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_save = runner._atomic_save_checkpoint

    def interrupting_save(
        checkpoint_dir: Path,
        predictor: LLMNPredictor,
        training_state: Mapping[str, Any],
        training_config_record: Mapping[str, Any],
    ) -> None:
        original_save(checkpoint_dir, predictor, training_state, training_config_record)
        if int(training_state["global_step"]) == 2:
            raise SyntheticInterruption("simulated process interruption after checkpoint-2")

    monkeypatch.setattr(runner, "_atomic_save_checkpoint", interrupting_save)
    with pytest.raises(SyntheticInterruption):
        runner.train_sft(config, run_root)
    monkeypatch.setattr(runner, "_atomic_save_checkpoint", original_save)


def test_uninterrupted_and_resumed_four_step_training_are_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted_runtime = TinyRuntime(dataset_size=16)
    install_tiny_runtime(monkeypatch, uninterrupted_runtime)
    uninterrupted_root = tmp_path / "uninterrupted" / "resume-run"
    uninterrupted_root.mkdir(parents=True)
    config = base_config(tmp_path / "uninterrupted")
    uninterrupted_metrics = runner.train_sft(config, uninterrupted_root)

    resumed_runtime = TinyRuntime(dataset_size=16)
    install_tiny_runtime(monkeypatch, resumed_runtime)
    resumed_root = tmp_path / "resumed" / "resume-run"
    resumed_root.mkdir(parents=True)
    resumed_config = base_config(tmp_path / "resumed")
    run_with_interruption_at_step_two(resumed_config, resumed_root, monkeypatch)
    checkpoint_two = resumed_root / "trainer" / "checkpoint-2"
    resumed_config["resume_from_checkpoint"] = str(checkpoint_two)
    resumed_metrics = runner.train_sft(resumed_config, tmp_path / "ignored-new-run")

    uninterrupted_checkpoint = uninterrupted_root / "trainer" / "checkpoint-4"
    resumed_checkpoint = resumed_root / "trainer" / "checkpoint-4"
    uninterrupted_state = load_training_state(uninterrupted_checkpoint)
    resumed_state = load_training_state(resumed_checkpoint)
    uninterrupted_adapter = read_adapter(uninterrupted_checkpoint)
    resumed_adapter = read_adapter(resumed_checkpoint)

    required_state = {
        "optimizer",
        "scheduler",
        "scaler",
        "global_step",
        "epoch",
        "sample_cursor",
        "samples_seen",
        "loss_sum",
        "loss_count",
        "rng_state",
        "sampler",
        "total_steps",
        "warmup_steps",
        "context_events_offset",
    }
    assert required_state <= uninterrupted_state.keys()
    assert {"python", "numpy", "torch_cpu", "torch_cuda"} <= uninterrupted_state[
        "rng_state"
    ].keys()

    assert uninterrupted_metrics["train_metrics"]["global_steps"] == 4
    assert resumed_metrics["train_metrics"]["global_steps"] == 4
    assert uninterrupted_metrics["train_metrics"]["learning_rate"] == pytest.approx(
        resumed_metrics["train_metrics"]["learning_rate"], rel=0.0, abs=0.0
    )
    uninterrupted_visits = uninterrupted_runtime.visitation_logs[0]
    resumed_visits = resumed_runtime.visitation_logs[0] + resumed_runtime.visitation_logs[1]
    assert resumed_visits == uninterrupted_visits
    assert len(resumed_visits) == 16
    assert sorted(resumed_visits) == list(range(16))
    assert set(uninterrupted_adapter) == set(resumed_adapter) == {"lora_A", "lora_B"}
    for name in uninterrupted_adapter:
        torch.testing.assert_close(
            uninterrupted_adapter[name],
            resumed_adapter[name],
            rtol=1e-7,
            atol=1e-8,
        )
    assert_nested_equal(uninterrupted_state["optimizer"], resumed_state["optimizer"])
    assert_nested_equal(uninterrupted_state["scheduler"], resumed_state["scheduler"])
    assert uninterrupted_state["loss_sum"] == pytest.approx(
        resumed_state["loss_sum"], rel=0.0, abs=1e-12
    )
    assert uninterrupted_state["loss_count"] == resumed_state["loss_count"]
    assert uninterrupted_state["samples_seen"] == resumed_state["samples_seen"] == 16


def test_latest_ignores_temporary_and_incomplete_checkpoints(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    trainer = run_root / "trainer"
    for step in (2, 4):
        checkpoint = trainer / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / CHECKPOINT_MARKER).write_text("complete\n", encoding="utf-8")
        (checkpoint / CHECKPOINT_STATE_FILE).write_bytes(b"state")
        (checkpoint / CHECKPOINT_CONFIG_FILE).write_text("{}\n", encoding="utf-8")
    temporary = trainer / "checkpoint-999.tmp"
    temporary.mkdir()
    (temporary / CHECKPOINT_MARKER).write_text("complete\n", encoding="utf-8")
    incomplete = trainer / "checkpoint-1000"
    incomplete.mkdir()
    (incomplete / CHECKPOINT_STATE_FILE).write_bytes(b"state")

    selected = resolve_resume_checkpoint("latest", {"output_dir": str(run_root)})

    assert selected == (trainer / "checkpoint-4").resolve()


def test_resume_config_mismatch_lists_conflicting_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TinyRuntime(dataset_size=16)
    install_tiny_runtime(monkeypatch, runtime)
    run_root = tmp_path / "mismatch" / "resume-run"
    run_root.mkdir(parents=True)
    config = base_config(tmp_path / "mismatch")
    run_with_interruption_at_step_two(config, run_root, monkeypatch)
    config["resume_from_checkpoint"] = str(run_root / "trainer" / "checkpoint-2")
    config["batch_size"] = 4
    config["max_steps"] = 100

    with pytest.raises(runner.ResumeConfigMismatchError) as error:
        runner.train_sft(config, tmp_path / "ignored")

    message = str(error.value)
    assert "batch_size" in message
    assert "max_steps" in message
    assert "total_steps" in message


def test_save_total_limit_is_applied_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TinyRuntime(dataset_size=16)
    install_tiny_runtime(monkeypatch, runtime)
    run_root = tmp_path / "prune" / "resume-run"
    run_root.mkdir(parents=True)
    config = base_config(tmp_path / "prune")
    config["save_steps"] = 2
    config["save_total_limit"] = 2
    run_with_interruption_at_step_two(config, run_root, monkeypatch)
    config["resume_from_checkpoint"] = "latest"
    config["output_dir"] = str(run_root)
    config["run_name"] = None
    config["save_steps"] = 1
    runner.train_sft(config, tmp_path / "ignored")

    checkpoints = sorted(path.name for path in (run_root / "trainer").glob("checkpoint-*"))
    assert checkpoints == ["checkpoint-3", "checkpoint-4"]


def test_epoch_boundary_resume_preserves_next_epoch_permutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TinyRuntime(dataset_size=8)
    install_tiny_runtime(monkeypatch, runtime)
    run_root = tmp_path / "epoch" / "resume-run"
    run_root.mkdir(parents=True)
    config = base_config(tmp_path / "epoch", dataset_size=8)
    config["num_epochs"] = 2
    run_with_interruption_at_step_two(config, run_root, monkeypatch)
    checkpoint_two = run_root / "trainer" / "checkpoint-2"
    state = load_training_state(checkpoint_two)
    assert state["epoch"] == 1
    assert state["sample_cursor"] == 0

    config["resume_from_checkpoint"] = str(checkpoint_two)
    metrics = runner.train_sft(config, tmp_path / "ignored")
    visits = runtime.visitation_logs[0] + runtime.visitation_logs[1]
    sampler = ResumableRandomSampler(list(range(8)), seed=73)
    expected = sampler.permutation(0) + sampler.permutation(1)

    assert metrics["train_metrics"]["global_steps"] == 4
    assert visits == expected
    assert sorted(visits[:8]) == list(range(8))
    assert sorted(visits[8:]) == list(range(8))
