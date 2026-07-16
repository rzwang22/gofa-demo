"""Prediction parsing and TAGLAS text-accuracy integration."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


YES_NO_TASKS = frozenset({"cora_link", "pubmed_link"})
EXACT_RELATION_TASKS = frozenset({"wn18rr", "fb15k237"})


def normalize_like_taglas(text: str) -> str:
    """Mirror TAGLAS ``normalize_text`` for canonical-label lookup."""
    text = str(text).lower()
    text = re.sub(r"<pad>|</s>|<s>|<bos>|<eos>", "", text)
    punctuation = set(string.punctuation)
    text = "".join(character for character in text if character not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonicalize_label(text: str) -> str:
    text = str(text)
    text = re.sub(r"<pad>|</s>|<s>|<bos>|<eos>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:assistant\s*)?answer\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).rstrip(" \t\r\n.!?")


@dataclass(frozen=True)
class ParsedPrediction:
    raw_generation: str
    normalized_label: str
    matched: bool
    parse_mode: str


class PredictionParser:
    """Turn direct generations into evaluator-ready canonical labels."""

    def __init__(self, task_name: str, canonical_labels: Sequence[str] = ()) -> None:
        self.task_name = task_name
        self.canonical_labels = tuple(canonicalize_label(label) for label in canonical_labels)
        normalized: dict[str, list[str]] = {}
        for label in self.canonical_labels:
            normalized.setdefault(normalize_like_taglas(label), []).append(label)
        self._normalized_labels = normalized

    def parse(self, generation: str) -> ParsedPrediction:
        raw = str(generation)
        if self.task_name in YES_NO_TASKS:
            match = re.search(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
            label = match.group(1).capitalize() if match else ""
            return ParsedPrediction(raw, label, bool(match), "yes_no_regex")

        candidate = canonicalize_label(raw)
        if self.task_name in EXACT_RELATION_TASKS:
            exact_matches = [label for label in self.canonical_labels if candidate == label]
            label = exact_matches[0] if len(exact_matches) == 1 else ""
            return ParsedPrediction(raw, label, bool(label), "canonical_relation_exact")

        candidate_key = normalize_like_taglas(candidate)
        matches = self._normalized_labels.get(candidate_key, [])
        if len(matches) == 1:
            label = matches[0]
            matched = True
        elif len(matches) > 1:
            # Preserve exact spelling when TAGLAS normalization collapses two
            # labels (possible for relation names containing punctuation).
            exact = [label for label in matches if label == candidate]
            label = exact[0] if len(exact) == 1 else ""
            matched = bool(label)
        elif not self.canonical_labels and candidate:
            label = candidate
            matched = True
        else:
            label = ""
            matched = False
        return ParsedPrediction(raw, label, matched, "canonical_exact")


def build_taglas_text_accuracy(task_name: str) -> Any:
    """Return the evaluator configured by TAGLAS for this exact QA task."""
    from TAGLAS import get_evaluators

    metric_names, evaluators = get_evaluators([task_name], task_types="QA")
    if metric_names != ["text_accuracy"] or len(evaluators) != 1:
        raise RuntimeError(
            f"TAGLAS did not return one text_accuracy evaluator for {task_name}: {metric_names!r}"
        )
    return evaluators[0]


def update_text_accuracy(evaluator: Any, prediction: str, target: str) -> None:
    """Update a TAGLAS TextAccuracy instance with one canonicalized sample."""
    evaluator.update([prediction], [canonicalize_label(target)])


def compute_text_accuracy(evaluator: Any) -> float:
    value = evaluator.compute()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
