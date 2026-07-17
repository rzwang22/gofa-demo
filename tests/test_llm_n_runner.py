from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

import llm_n.runner as runner
from llm_n.model import (
    ContextOverflowError,
    GenerationResult,
    LLMNPredictor,
    TokenizedPrompt,
)
from llm_n.serialization import SerializedGraphSample


def _sample(index: int, label: str) -> SerializedGraphSample:
    return SerializedGraphSample(
        task_name="cora_link",
        sample_index=index,
        prompt=(
            "Task Description:\nSynthetic runner test.\n\nTarget:\n"
            "[Node A] and [Node B]\n\nNodes:\n- [Node A]: A\n- [Node B]: B\n\n"
            "Edges:\n- (none)\n\nQuestion:\nIs there a link?\n\nAnswer:"
        ),
        answer=f"{label}.",
        target_label=label,
        target_node_ids=("[Node A]", "[Node B]"),
        local_node_ids=("[Node A]", "[Node B]"),
    )


class FakeDataset:
    task_name = "cora_link"
    split = "test"
    label_space = ("Yes", "No")

    def __init__(self) -> None:
        self.samples = (_sample(0, "Yes"), _sample(1, "No"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SerializedGraphSample:
        return self.samples[index]


class CountingEvaluator:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.updates: list[tuple[str, str]] = []

    def update(self, prediction: str, target: str) -> None:
        self.updates.append((prediction, target))
        self.correct += int(prediction.strip().lower() == target.strip().lower())
        self.total += 1

    def compute(self) -> float:
        return self.correct / self.total if self.total else 0.0


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    evaluators: list[CountingEvaluator] = []
    oom_clears: list[bool] = []

    def build_evaluator(task_name: str) -> CountingEvaluator:
        assert task_name == "cora_link"
        evaluator = CountingEvaluator()
        evaluators.append(evaluator)
        return evaluator

    monkeypatch.setattr(runner, "build_taglas_text_accuracy", build_evaluator)
    monkeypatch.setattr(
        runner,
        "update_text_accuracy",
        lambda evaluator, prediction, target: evaluator.update(prediction, target),
    )
    monkeypatch.setattr(runner, "compute_text_accuracy", lambda evaluator: evaluator.compute())
    monkeypatch.setattr(runner, "_reset_peak_memory", lambda: None)
    monkeypatch.setattr(runner, "_read_peak_memory", lambda: (123, {"fake:0": 123}))
    monkeypatch.setattr(runner, "_clear_cuda_after_oom", lambda: oom_clears.append(True))
    return {"evaluators": evaluators, "oom_clears": oom_clears}


def _result(
    raw_text: str,
    *,
    input_tokens: int,
    original_tokens: int,
    output_tokens: int,
    prefill_ms: float,
    decode_ms: float,
    truncated: bool,
) -> GenerationResult:
    return GenerationResult(
        raw_text=raw_text,
        generated_token_ids=tuple(range(output_tokens)),
        input_token_count=input_tokens,
        original_input_token_count=original_tokens,
        output_token_count=output_tokens,
        truncated=truncated,
        prefill_latency_ms=prefill_ms,
        decode_latency_ms=decode_ms,
        total_latency_ms=prefill_ms + decode_ms,
    )


WARMUP_RESULT = _result(
    "warm-up output that must never be scored",
    input_tokens=900,
    original_tokens=950,
    output_tokens=90,
    prefill_ms=100.0,
    decode_ms=200.0,
    truncated=True,
)


class TenCallWarmupPredictor:
    """Return warm-up results for ten calls, then delegate measured samples."""

    def __init__(self, measured: Callable[[SerializedGraphSample], GenerationResult]) -> None:
        self.tokenization = SimpleNamespace(truncate_mode="none")
        self.measured = measured
        self.calls: list[SerializedGraphSample | str] = []

    def generate(self, sample: SerializedGraphSample | str) -> GenerationResult:
        self.calls.append(sample)
        if len(self.calls) <= 10:
            return WARMUP_RESULT
        assert isinstance(sample, SerializedGraphSample)
        return self.measured(sample)


class PretokenizingPolicy:
    truncate_mode = "right"

    def encode_prompt(self, prompt: str) -> TokenizedPrompt:
        del prompt
        return TokenizedPrompt(
            input_ids=tuple(range(64)),
            attention_mask=tuple(1 for _ in range(64)),
            original_token_count=70,
            token_count=64,
            truncated=True,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_infer_task_writes_two_predictions_and_only_aggregates_measured_samples(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    dataset = FakeDataset()

    def measured(sample: SerializedGraphSample) -> GenerationResult:
        if sample.sample_index == 0:
            return _result(
                "Yes",
                input_tokens=11,
                original_tokens=11,
                output_tokens=2,
                prefill_ms=1.0,
                decode_ms=2.0,
                truncated=False,
            )
        return _result(
            "No",
            input_tokens=17,
            original_tokens=22,
            output_tokens=3,
            prefill_ms=2.0,
            decode_ms=3.0,
            truncated=True,
        )

    predictor = TenCallWarmupPredictor(measured)
    output_dir = tmp_path / "normal"
    metrics = runner.infer_task(
        {"warm_up_samples": 1, "save_prompts": True},
        predictor,
        dataset,
        output_dir,
    )

    # The requested value is one, but the runner enforces at least ten warm-ups.
    assert metrics["warm_up_samples"] == 10
    assert len(predictor.calls) == 12
    assert sum(isinstance(call, str) for call in predictor.calls[:10]) == 8
    assert all(isinstance(call, SerializedGraphSample) for call in predictor.calls[10:])

    # None of WARMUP_RESULT's deliberately huge values may enter measured totals.
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["num_samples"] == 2
    assert metrics["num_latency_samples"] == 2
    assert metrics["total_latency_ms"] == pytest.approx(8.0)
    assert metrics["latency_per_sample_ms"] == pytest.approx(4.0)
    assert metrics["prefill_latency_ms"] == pytest.approx(3.0)
    assert metrics["prefill_latency_per_sample_ms"] == pytest.approx(1.5)
    assert metrics["decode_latency_ms"] == pytest.approx(5.0)
    assert metrics["decode_latency_per_sample_ms"] == pytest.approx(2.5)
    assert metrics["input_token_count"] == 28
    assert metrics["original_input_token_count"] == 33
    assert metrics["output_token_count"] == 5
    assert metrics["truncation_count"] == 1
    assert metrics["overflow_count"] == 0
    assert metrics["oom_count"] == 0
    assert metrics["overflow_oom_count"] == 0
    assert metrics["peak_gpu_memory_bytes"] == 123
    assert metrics["peak_gpu_memory_by_device"] == {"fake:0": 123}

    evaluator = fake_runtime["evaluators"][0]
    assert evaluator.total == 2
    assert evaluator.updates == [("Yes", "Yes"), ("No", "No")]

    prediction_path = output_dir / "predictions.jsonl"
    metric_path = output_dir / "metrics.json"
    assert prediction_path.is_file()
    assert metric_path.is_file()
    records = _read_jsonl(prediction_path)
    assert len(records) == 2
    assert [record["status"] for record in records] == ["ok", "ok"]
    assert [record["correct"] for record in records] == [True, True]
    assert records[0]["serialized_input"] == dataset[0].prompt
    assert records[0]["truncated_input_token_count"] is None
    assert records[1]["original_input_token_count"] == 22
    assert records[1]["truncated_input_token_count"] == 17
    with metric_path.open("r", encoding="utf-8") as handle:
        written_metrics = json.load(handle)
    assert written_metrics["accuracy"] == pytest.approx(1.0)
    assert written_metrics["predictions_file"] == str(prediction_path.resolve())


def test_overflow_and_oom_after_warmup_continue_and_use_all_samples_for_accuracy(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    dataset = FakeDataset()

    def measured(sample: SerializedGraphSample) -> GenerationResult:
        if sample.sample_index == 0:
            raise ContextOverflowError(original_tokens=123, capacity=64, context="inference prompt")
        raise RuntimeError("synthetic CUDA out of memory")

    predictor = TenCallWarmupPredictor(measured)
    predictor.tokenization = PretokenizingPolicy()
    output_dir = tmp_path / "errors"
    metrics = runner.infer_task(
        {"warm_up_samples": 2, "save_prompts": False},
        predictor,
        dataset,
        output_dir,
    )

    assert metrics["warm_up_samples"] == 10
    assert len(predictor.calls) == 12
    assert metrics["num_samples"] == 2
    assert metrics["num_latency_samples"] == 0
    assert metrics["total_latency_ms"] == 0.0
    assert metrics["latency_per_sample_ms"] == 0.0
    assert metrics["prefill_latency_ms"] == 0.0
    assert metrics["decode_latency_ms"] == 0.0
    assert metrics["input_token_count"] == 64
    assert metrics["original_input_token_count"] == 193
    assert metrics["output_token_count"] == 0
    assert metrics["truncation_count"] == 1
    assert metrics["overflow_count"] == 1
    assert metrics["oom_count"] == 1
    assert metrics["overflow_oom_count"] == 2
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert fake_runtime["oom_clears"] == [True]

    # Accuracy is over both requested samples, not successful-latency samples or warm-ups.
    evaluator = fake_runtime["evaluators"][0]
    assert evaluator.total == 2
    assert evaluator.updates == [("", "Yes"), ("", "No")]

    records = _read_jsonl(output_dir / "predictions.jsonl")
    assert [record["status"] for record in records] == ["overflow", "oom"]
    assert all(record["correct"] is False for record in records)
    assert records[0]["original_input_token_count"] == 123
    assert records[0]["truncated_input_token_count"] is None
    assert "only 64 are allowed" in records[0]["error"]
    assert "out of memory" in records[1]["error"]
    assert records[1]["original_input_token_count"] == 70
    assert records[1]["truncated_input_token_count"] == 64
    assert records[1]["input_token_count"] == 64
    assert records[1]["truncated"] is True
    assert all("serialized_input" not in record for record in records)


class AllRealSamplesOverflowPredictor:
    def __init__(self) -> None:
        self.tokenization = SimpleNamespace(truncate_mode="none")
        self.calls: list[SerializedGraphSample | str] = []

    def generate(self, sample: SerializedGraphSample | str) -> GenerationResult:
        self.calls.append(sample)
        if isinstance(sample, SerializedGraphSample):
            raise ContextOverflowError(
                original_tokens=101 + sample.sample_index,
                capacity=64,
                context="inference prompt",
            )
        return WARMUP_RESULT


def test_all_real_warmup_samples_overflow_uses_neutral_prompt_then_records_overflows(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    dataset = FakeDataset()
    predictor = AllRealSamplesOverflowPredictor()
    output_dir = tmp_path / "all-overflow"

    metrics = runner.infer_task(
        {"warm_up_samples": 1, "save_prompts": True},
        predictor,
        dataset,
        output_dir,
    )

    # Two failed real probes, ten successful neutral warm-ups, then two measured overflows.
    assert metrics["warm_up_samples"] == 10
    assert len(predictor.calls) == 14
    assert sum(isinstance(call, SerializedGraphSample) for call in predictor.calls[:12]) == 2
    assert sum(isinstance(call, str) for call in predictor.calls[:12]) == 10
    assert all(isinstance(call, SerializedGraphSample) for call in predictor.calls[12:])

    assert metrics["num_samples"] == 2
    assert metrics["num_latency_samples"] == 0
    assert metrics["overflow_count"] == 2
    assert metrics["oom_count"] == 0
    assert metrics["overflow_oom_count"] == 2
    assert metrics["original_input_token_count"] == 203
    assert metrics["input_token_count"] == 0
    assert metrics["output_token_count"] == 0
    assert metrics["total_latency_ms"] == 0.0
    assert metrics["truncation_count"] == 0
    assert metrics["accuracy"] == pytest.approx(0.0)

    evaluator = fake_runtime["evaluators"][0]
    assert evaluator.total == 2
    records = _read_jsonl(output_dir / "predictions.jsonl")
    assert len(records) == 2
    assert [record["status"] for record in records] == ["overflow", "overflow"]
    assert [record["original_input_token_count"] for record in records] == [101, 102]
    assert all(record["serialized_input"].endswith("Answer:") for record in records)
    assert (output_dir / "metrics.json").is_file()


class TinyTrainingTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2}

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
        (output_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        target = labels.ne(-100).float().mean()
        return SimpleNamespace(loss=(self.lora_weight - target).square())

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "adapter_model.bin")


class TinyTrainingDataset:
    task_name = "cora_node"
    split = "train"
    label_space = ("Neural_Networks",)

    def __init__(self, prompt_repetitions: int = 1) -> None:
        context = " ".join("context" for _ in range(prompt_repetitions))
        self.samples = tuple(
            SerializedGraphSample(
                task_name="cora_node",
                sample_index=index,
                prompt=f"{context} Question for sample {index}\nAnswer:",
                answer="Neural_Networks",
                target_label="Neural_Networks",
                target_node_ids=("[Node A]",),
                local_node_ids=("[Node A]",),
            )
            for index in range(3)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SerializedGraphSample:
        return self.samples[index]


def test_isolated_sft_loop_updates_parameters_and_saves_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TinyTrainingTokenizer()
    model = TinyLoRAModel()
    predictor = LLMNPredictor(
        model,
        tokenizer,
        max_context_length=64,
        max_new_tokens=8,
        truncate_mode="none",
    )
    initial_weight = model.lora_weight.detach().clone()
    monkeypatch.setattr(
        runner,
        "_build_datasets",
        lambda config, tasks, split, inference: [TinyTrainingDataset()],
    )
    monkeypatch.setattr(
        runner.LLMNPredictor,
        "from_pretrained",
        classmethod(lambda cls, config, for_training=False: predictor),
    )
    run_root = tmp_path / "run"
    adapter_dir = tmp_path / "adapter"
    run_root.mkdir()

    metrics = runner.train_sft(
        {
            "train_task_names": ["cora_node"],
            "batch_size": 1,
            "grad_acc_step": 2,
            "num_epochs": 1,
            "lr": 0.1,
            "training_precision": "fp32",
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 1,
            "adapter_output_dir": str(adapter_dir),
        },
        run_root,
    )

    assert metrics["train_metrics"]["global_steps"] == 2
    assert metrics["train_metrics"]["train_samples_seen"] == 3
    assert metrics["train_metrics"]["truncation_count"] == 0
    updated_weight = model.lora_weight.detach().to(initial_weight.device)
    assert not torch.equal(updated_weight, initial_weight)
    assert (adapter_dir / "adapter_model.bin").is_file()
    assert (adapter_dir / "tokenizer_config.json").is_file()
    assert [path.name for path in (run_root / "trainer").glob("checkpoint-*")] == [
        "checkpoint-2"
    ]
    assert (run_root / "training_metrics.json").is_file()


def test_sft_explicit_truncation_writes_per_sample_context_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = LLMNPredictor(
        TinyLoRAModel(),
        TinyTrainingTokenizer(),
        max_context_length=12,
        max_new_tokens=4,
        truncate_mode="right",
    )
    monkeypatch.setattr(
        runner,
        "_build_datasets",
        lambda config, tasks, split, inference: [TinyTrainingDataset(prompt_repetitions=30)],
    )
    monkeypatch.setattr(
        runner.LLMNPredictor,
        "from_pretrained",
        classmethod(lambda cls, config, for_training=False: predictor),
    )
    run_root = tmp_path / "truncate-run"
    run_root.mkdir()

    metrics = runner.train_sft(
        {
            "train_task_names": ["cora_node"],
            "batch_size": 1,
            "grad_acc_step": 1,
            "num_epochs": 1,
            "training_precision": "fp32",
            "save_strategy": "no",
        },
        run_root,
    )

    assert metrics["train_metrics"]["truncation_count"] == 3
    records = _read_jsonl(run_root / "training_context_events.jsonl")
    assert len(records) == 3
    assert all(record["status"] == "truncated" for record in records)
    assert all(record["original_token_count"] > record["token_count"] for record in records)


def test_sft_default_no_truncation_records_overflow_before_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = LLMNPredictor(
        TinyLoRAModel(),
        TinyTrainingTokenizer(),
        max_context_length=12,
        max_new_tokens=4,
        truncate_mode="none",
    )
    monkeypatch.setattr(
        runner,
        "_build_datasets",
        lambda config, tasks, split, inference: [TinyTrainingDataset(prompt_repetitions=30)],
    )
    monkeypatch.setattr(
        runner.LLMNPredictor,
        "from_pretrained",
        classmethod(lambda cls, config, for_training=False: predictor),
    )
    run_root = tmp_path / "overflow-run"
    run_root.mkdir()

    with pytest.raises(ContextOverflowError):
        runner.train_sft(
            {
                "train_task_names": ["cora_node"],
                "batch_size": 1,
                "grad_acc_step": 1,
                "num_epochs": 1,
                "training_precision": "fp32",
                "save_strategy": "no",
            },
            run_root,
        )

    records = _read_jsonl(run_root / "training_context_events.jsonl")
    assert len(records) == 1
    assert records[0]["status"] == "overflow"
    assert records[0]["task_name"] == "cora_node"
    assert records[0]["original_token_count"] > records[0]["max_context_length"]
    with (run_root / "training_failure.json").open("r", encoding="utf-8") as handle:
        failure = json.load(handle)
    assert failure["overflow_count"] == 1
