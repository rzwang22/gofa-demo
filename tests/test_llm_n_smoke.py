from __future__ import annotations

import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from llm_n.evaluation import (
    PredictionParser,
    build_taglas_text_accuracy,
    compute_text_accuracy,
    normalize_like_taglas,
    update_text_accuracy,
)
from llm_n.model import (
    AnswerOnlyDataCollator,
    ContextOverflowError,
    LLMNPredictor,
    LLMNTokenizer,
)
from llm_n.serialization import SerializedGraphSample, serialize_taglas_sample


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


def synthetic_taglas_sample(task_name: str, sample_index: int) -> SimpleNamespace:
    is_link = task_name in LINK_TASKS
    targets = (2, 0) if is_link else (1,)
    pairs = (
        [(2, 1), (1, 0), (0, 3), (3, 1)]
        if is_link
        else [(3, 2), (1, 3), (2, 1), (0, 2)]
    )
    rows = [source for source, _ in pairs]
    columns = [target for _, target in pairs]
    label = TASK_LABELS[task_name][sample_index]
    return SimpleNamespace(
        node_map=torch.tensor([30, 10, 20, 40], dtype=torch.long),
        x=[f"{task_name} node {index}, sample {sample_index}" for index in range(4)],
        edge_index=torch.tensor([rows, columns], dtype=torch.long),
        edge_map=torch.arange(len(pairs), dtype=torch.long),
        edge_attr=[f"observed relation {source}-{target}" for source, target in pairs],
        target_index=torch.tensor(targets, dtype=torch.long),
        question=[
            "Predict the label for "
            + " and ".join(f"[NODE_INDEX {target}]" for target in targets)
            + "."
        ],
        answer=[f"{label}."],
        label=[label],
    )


class FakeTokenizer:
    """Small reversible whitespace tokenizer with a Mistral-like chat hook."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self._token_to_id = {"<pad>": 0, "<s>": 1, "</s>": 2}
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for token in re.findall(r"\S+", str(text)):
            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return f"<s> [INST] {messages[0]['content']} [/INST]"

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del clean_up_tokenization_spaces
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = [
            self._id_to_token[int(token_id)]
            for token_id in token_ids
            if not (skip_special_tokens and int(token_id) in special)
        ]
        return " ".join(tokens)


class FakeCausalLM(torch.nn.Module):
    """CPU-only causal LM supporting loss-bearing forward and cached decode."""

    def __init__(self, vocab_size: int = 4096) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self._planned_ids = (2,)
        self._cursor = 0
        self.last_labels: torch.Tensor | None = None

    def plan(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("A generation plan must contain at least one token")
        if max(token_ids) >= self.vocab_size:
            raise ValueError("Generation plan exceeds fake vocabulary")
        self._planned_ids = tuple(int(token_id) for token_id in token_ids)
        self._cursor = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        assert return_dict is True
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )

        if labels is not None:
            self.last_labels = labels.detach().clone()
            supervised = labels.ne(-100)
            safe_labels = labels.masked_fill(~supervised, 0)
            logits.scatter_(2, safe_labels.unsqueeze(-1), 10.0)
            loss = self.anchor.square() + supervised.float().mean()
            return SimpleNamespace(logits=logits, loss=loss, past_key_values=None)

        if past_key_values is None:
            self._cursor = 0
        token_id = self._planned_ids[min(self._cursor, len(self._planned_ids) - 1)]
        self._cursor += 1
        logits[:, -1, token_id] = 10.0
        return SimpleNamespace(
            logits=logits,
            loss=None,
            past_key_values=("fake-cache", self._cursor),
        )


class FakeTextAccuracy:
    """TAGLAS-shaped stateful text metric used without importing TAGLAS."""

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.correct = 0
        self.total = 0

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for prediction, target in zip(predictions, targets):
            if self.task_name in {"cora_link", "pubmed_link"}:
                match = re.search(r"\b(yes|no)\b", prediction, flags=re.IGNORECASE)
                normalized_prediction = match.group(1) if match else ""
            else:
                normalized_prediction = prediction
            self.correct += int(
                normalize_like_taglas(normalized_prediction) == normalize_like_taglas(target)
            )
            self.total += 1

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.correct / self.total if self.total else 0.0)


@pytest.fixture
def fake_taglas(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("TAGLAS")

    def get_evaluators(
        names: list[str],
        task_types: str,
    ) -> tuple[list[str], list[FakeTextAccuracy]]:
        assert task_types == "QA"
        return ["text_accuracy"], [FakeTextAccuracy(names[0])]

    module.get_evaluators = get_evaluators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "TAGLAS", module)
    return module


@pytest.mark.parametrize("task_name", tuple(TASK_LABELS))
def test_two_sample_end_to_end_smoke_per_required_task(
    task_name: str,
    fake_taglas: types.ModuleType,
) -> None:
    del fake_taglas
    tokenizer = FakeTokenizer()
    model = FakeCausalLM()
    predictor = LLMNPredictor(
        model,
        tokenizer,
        max_context_length=512,
        max_new_tokens=8,
        truncate_mode="none",
    )
    samples = [
        serialize_taglas_sample(
            synthetic_taglas_sample(task_name, sample_index),
            task_name,
            sample_index=sample_index,
        )
        for sample_index in range(2)
    ]

    # One supervised forward covers both samples and validates answer-only labels.
    collator = AnswerOnlyDataCollator(predictor.tokenization)
    batch = collator(samples)
    output = predictor.forward(**batch)
    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(output.loss)
    assert model.last_labels is not None
    assert torch.equal(model.last_labels, batch["labels"])
    for row_labels, row_attention in zip(batch["labels"], batch["attention_mask"]):
        active_labels = row_labels[row_attention.bool()]
        answer_positions = active_labels.ne(-100).nonzero(as_tuple=False).flatten()
        assert answer_positions.numel() > 0
        first_answer = int(answer_positions[0])
        assert bool(active_labels[:first_answer].eq(-100).all())
        assert bool(active_labels[first_answer:].ne(-100).all())

    evaluator = build_taglas_text_accuracy(task_name)
    parser = PredictionParser(task_name, canonical_labels=TASK_LABELS[task_name])
    model.train()
    for sample, expected_label in zip(samples, TASK_LABELS[task_name]):
        planned_ids = tokenizer.encode(expected_label, add_special_tokens=False)
        model.plan(planned_ids + [tokenizer.eos_token_id])
        generation = predictor.generate(sample)
        parsed = parser.parse(generation.raw_text)

        assert model.training is True  # generate restores the caller's mode
        assert generation.raw_text == expected_label
        assert generation.input_token_count > 0
        assert generation.original_input_token_count == generation.input_token_count
        assert generation.output_token_count == len(planned_ids) + 1
        assert generation.truncated is False
        assert generation.prefill_latency_ms >= 0.0
        assert generation.decode_latency_ms >= 0.0
        assert generation.total_latency_ms >= 0.0
        assert parsed.matched is True
        assert parsed.normalized_label == expected_label
        if task_name in {"cora_link", "pubmed_link"}:
            assert parsed.parse_mode == "yes_no_regex"
        elif task_name == "wn18rr":
            assert parsed.parse_mode == "canonical_relation_exact"
        else:
            assert parsed.parse_mode == "canonical_exact"
        update_text_accuracy(evaluator, parsed.normalized_label, sample.target_label)

    assert compute_text_accuracy(evaluator) == pytest.approx(1.0)


def serialized_stub(prompt: str, answer: str = "label") -> SerializedGraphSample:
    return SerializedGraphSample(
        task_name="cora_node",
        sample_index=0,
        prompt=prompt,
        answer=answer,
        target_label=answer,
        target_node_ids=("[Node A]",),
        local_node_ids=("[Node A]",),
    )


def test_overflow_and_explicit_truncation_counts_preserve_token_metadata() -> None:
    tokenizer = FakeTokenizer()
    long_prompt = " ".join(f"context-{index}" for index in range(160)) + "\nAnswer:"
    short_prompt = "short context\nAnswer:"

    overflow_count = 0
    no_truncation = LLMNTokenizer(
        tokenizer,
        max_context_length=48,
        max_new_tokens=4,
        truncate_mode="none",
    )
    try:
        no_truncation.encode_prompt(long_prompt)
    except ContextOverflowError as error:
        overflow_count += 1
        assert error.context == "inference prompt"
        assert error.original_tokens > error.capacity
        assert error.capacity == 44
    assert overflow_count == 1

    model = FakeCausalLM()
    truncating = LLMNPredictor(
        model,
        tokenizer,
        max_context_length=48,
        max_new_tokens=4,
        truncate_mode="middle",
    )
    model.plan([tokenizer.eos_token_id])
    results = [truncating.generate(prompt) for prompt in (long_prompt, short_prompt)]
    truncation_count = sum(int(result.truncated) for result in results)

    assert truncation_count == 1
    assert results[0].original_input_token_count > results[0].input_token_count
    assert results[0].input_token_count == 44
    assert results[1].original_input_token_count == results[1].input_token_count
    assert results[1].truncated is False


@pytest.mark.parametrize(
    "generation",
    (
        "hypernym",
        "_HYPERNYM",
        "the _hypernym",
        "_hypernym because the first entity is broader",
    ),
)
def test_wn18rr_rejects_nonliteral_relation_near_misses(generation: str) -> None:
    parser = PredictionParser("wn18rr", ("_hypernym", "_also_see"))

    parsed = parser.parse(generation)

    assert parsed.parse_mode == "canonical_relation_exact"
    assert parsed.matched is False
    assert parsed.normalized_label == ""


def test_wn18rr_accepts_exact_relation_with_answer_wrapper() -> None:
    parser = PredictionParser("wn18rr", ("_hypernym", "_also_see"))

    parsed = parser.parse("Answer: _hypernym.</s>")

    assert parsed.matched is True
    assert parsed.normalized_label == "_hypernym"


@pytest.mark.parametrize("truncate_mode", ("left", "right", "middle"))
def test_training_truncation_keeps_answer_tokens_supervised(truncate_mode: str) -> None:
    tokenizer = FakeTokenizer()
    policy = LLMNTokenizer(
        tokenizer,
        max_context_length=40,
        max_new_tokens=4,
        truncate_mode=truncate_mode,
    )
    sample = serialized_stub(
        " ".join(f"long-node-text-{index}" for index in range(100)) + "\nAnswer:",
        answer="canonical label",
    )

    encoded = policy.encode_training_sample(sample)

    assert encoded.truncated is True
    assert encoded.original_token_count > encoded.token_count
    assert encoded.token_count == 40
    assert all(label == -100 for label in encoded.labels[: encoded.prompt_token_count])
    assert tuple(encoded.labels[encoded.prompt_token_count :]) == tuple(
        encoded.input_ids[encoded.prompt_token_count :]
    )
    assert encoded.answer_token_count == len(tokenizer.encode("canonical label")) + 1
