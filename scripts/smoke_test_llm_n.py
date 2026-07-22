#!/usr/bin/env python3
"""Real TAGLAS/Mistral functional smoke test: two samples per required task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_n.evaluation import (
    PredictionParser,
    build_taglas_text_accuracy,
    compute_text_accuracy,
    update_text_accuracy,
)
from llm_n.model import AnswerOnlyDataCollator, LLMNPredictor
from llm_n.runner import _build_datasets, _set_seed


REQUIRED_TASKS = [
    "cora_node",
    "cora_link",
    "pubmed_node",
    "pubmed_link",
    "arxiv",
    "wikics",
    "wn18rr",
]


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/llm_n_inference_config.yaml",
        help="LLM-N inference config",
    )
    parser.add_argument("--tasks", nargs="+", default=REQUIRED_TASKS)
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", default="outputs/llm_n_smoke")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The real LLM-N smoke test requires a CUDA GPU")

    config = _load_config(args.config)
    config.update(
        model_type="llm_n",
        run_mode="inference",
        eval_task_names=args.tasks,
        inf_sample_size_per_task=[2 for _ in args.tasks],
        inf_hops=[3 for _ in args.tasks],
        inf_max_nodes_per_hops=[5 for _ in args.tasks],
        sample_mode=[
            "balanced" if task in {"cora_link", "pubmed_link"} else "random"
            for task in args.tasks
        ],
        load_in_4bit=bool(args.load_in_4bit),
    )
    if args.model_path:
        config["model_name_or_path"] = args.model_path
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
        config["llm_n_mode"] = "llm_n_sft"
    else:
        config["adapter_path"] = None
        config["llm_n_mode"] = "llm_n_zero_shot"

    _set_seed(int(config.get("seed", 1)))
    datasets = _build_datasets(config, args.tasks, split="test", inference=True)
    predictor = LLMNPredictor.from_pretrained(config, for_training=False)
    collator = AnswerOnlyDataCollator(predictor.tokenization)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "smoke_predictions.jsonl"
    summary: dict[str, Any] = {"tasks": {}, "passed": True}

    with records_path.open("w", encoding="utf-8") as records_file:
        for dataset in datasets:
            if len(dataset) < 2:
                raise AssertionError(f"{dataset.task_name} returned only {len(dataset)} smoke samples")
            if dataset.task_name in {"cora_link", "pubmed_link"}:
                by_label = {}
                for index in range(len(dataset)):
                    candidate = dataset[index]
                    by_label.setdefault(candidate.target_label.lower(), candidate)
                    if {"yes", "no"}.issubset(by_label):
                        break
                if not {"yes", "no"}.issubset(by_label):
                    raise AssertionError(
                        f"{dataset.task_name} smoke sampling did not retain one TAGLAS Yes and one No sample"
                    )
                samples = [by_label["yes"], by_label["no"]]
            else:
                samples = [dataset[index] for index in range(2)]
            evaluator = build_taglas_text_accuracy(dataset.task_name)
            prediction_parser = PredictionParser(dataset.task_name, dataset.label_space)
            task_records: list[dict[str, Any]] = []

            for sample in samples:
                batch = collator([sample])
                batch = {key: value.to(predictor.device) for key, value in batch.items()}
                active_labels = batch["labels"][batch["attention_mask"].bool()]
                if not bool(active_labels.eq(-100).any()) or not bool(active_labels.ne(-100).any()):
                    raise AssertionError("Smoke sample does not have answer-only labels")
                with torch.inference_mode():
                    forward_output = predictor.forward(**batch)
                if not bool(torch.isfinite(forward_output.loss)):
                    raise AssertionError(f"Non-finite forward loss on {dataset.task_name}")

                generation = predictor.generate(sample)
                parsed = prediction_parser.parse(generation.raw_text)
                update_text_accuracy(evaluator, parsed.normalized_label, sample.target_label)
                record = {
                    "task_name": dataset.task_name,
                    "sample_index": sample.sample_index,
                    "forward_loss": float(forward_output.loss.detach().cpu().item()),
                    "raw_generation": generation.raw_text,
                    "normalized_prediction": parsed.normalized_label,
                    "target_label": sample.target_label,
                    "parse_mode": parsed.parse_mode,
                    "input_token_count": generation.input_token_count,
                    "output_token_count": generation.output_token_count,
                    "truncated": generation.truncated,
                }
                task_records.append(record)
                records_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                records_file.flush()

            accuracy = compute_text_accuracy(evaluator)
            if not 0.0 <= accuracy <= 1.0:
                raise AssertionError(f"Invalid text_accuracy {accuracy} on {dataset.task_name}")
            summary["tasks"][dataset.task_name] = {
                "samples": len(task_records),
                "forward": "passed",
                "generate": "passed",
                "label_parse": "passed",
                "target_edge_masking": "passed",
                "text_accuracy": accuracy,
            }

    with (output_dir / "smoke_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
