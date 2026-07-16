"""Training and inference runner for LLM-N-SFT."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import ConcatSerializedDataset, LLMNTaskDataset, gofa_data_size_filter
from .evaluation import (
    PredictionParser,
    build_taglas_text_accuracy,
    compute_text_accuracy,
    normalize_like_taglas,
    update_text_accuracy,
)
from .model import (
    AnswerOnlyDataCollator,
    ContextOverflowError,
    LLMNPredictor,
)
from .serialization import SUPPORTED_TASKS


LLM_N_MODES = frozenset({"llm_n_zero_shot", "llm_n_sft"})


def _get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_tasks(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        tasks = [value]
    elif isinstance(value, Sequence):
        tasks = [str(item) for item in value]
    else:
        raise ValueError(f"{field_name} must be a task name or list of task names")
    unsupported = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError(
            f"Unsupported tasks in {field_name}: {unsupported}; supported tasks: {list(SUPPORTED_TASKS)}"
        )
    return tasks


def _per_task(value: Any, tasks: Sequence[str], field_name: str) -> list[Any]:
    if isinstance(value, list):
        if len(value) != len(tasks):
            raise ValueError(
                f"{field_name} has {len(value)} values for {len(tasks)} tasks; use a scalar or one value per task"
            )
        return value
    return [value for _ in tasks]


def _build_datasets(config: Mapping[str, Any], tasks: Sequence[str], split: str, inference: bool) -> list[LLMNTaskDataset]:
    prefix = "inf_" if inference else ""
    hops = _per_task(_get(config, prefix + "hops", _get(config, "hops", 3)), tasks, prefix + "hops")
    max_nodes = _per_task(
        _get(
            config,
            prefix + "max_nodes_per_hops",
            _get(config, "max_nodes_per_hop", _get(config, "train_max_nodes_per_hops", 5)),
        ),
        tasks,
        prefix + "max_nodes_per_hops",
    )
    sample_key = "inf_sample_size_per_task" if inference else "sample_size_per_task"
    sample_sizes = _per_task(_get(config, sample_key, 1.0), tasks, sample_key)
    sample_modes = _per_task(_get(config, "sample_mode", "random"), tasks, "sample_mode")
    roots = _per_task(_get(config, "data_root_path", "TAGDataset"), tasks, "data_root_path")

    datasets: list[LLMNTaskDataset] = []
    filter_func = (
        gofa_data_size_filter
        if not inference and bool(_get(config, "gofa_size_filter", True))
        else None
    )
    for task, root, hop, max_per_hop, sample_size, sample_mode in zip(
        tasks, roots, hops, max_nodes, sample_sizes, sample_modes
    ):
        datasets.append(
            LLMNTaskDataset(
                task_name=task,
                root=str(root),
                split=split,
                hop=int(hop),
                max_nodes_per_hop=int(max_per_hop),
                sample_size=sample_size,
                sample_mode=str(sample_mode),
                num_workers=int(_get(config, "num_workers", 0)),
                save_data=bool(_get(config, "save_data", True)),
                from_saved=bool(_get(config, "from_saved", True)),
                filter_func=filter_func,
                to_sparse=bool(_get(config, "taglas_to_sparse", True)),
                fast_data_load=bool(_get(config, "fast_data_load", True)),
            )
        )
    return datasets


def _run_root(config: Mapping[str, Any]) -> Path:
    base = Path(str(_get(config, "output_dir", _get(config, "exp_dir", "outputs/llm_n"))))
    run_name = config.get("run_name")
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / str(run_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_sft(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    tasks = _normalize_tasks(_get(config, "train_task_names", _get(config, "task_names", [])), "train_task_names")
    datasets = _build_datasets(config, tasks, split="train", inference=False)
    train_dataset = ConcatSerializedDataset(datasets)
    if config.get("resume_from_checkpoint"):
        raise NotImplementedError(
            "resume_from_checkpoint is not implemented by the isolated LLM-N loop; "
            "start a new LoRA SFT run or load the completed adapter for inference"
        )
    predictor = LLMNPredictor.from_pretrained(config, for_training=True)
    collator = AnswerOnlyDataCollator(
        predictor.tokenization,
        include_metadata=True,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("LLM-N SFT received an empty TAGLAS training dataset")
    batch_size = int(_get(config, "batch_size", 1))
    grad_accumulation = int(_get(config, "grad_acc_step", 8))
    num_epochs = int(_get(config, "num_epochs", 1))
    if batch_size <= 0 or grad_accumulation <= 0 or num_epochs <= 0:
        raise ValueError("batch_size, grad_acc_step, and num_epochs must be positive")

    seed = int(_get(config, "seed", 1))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=int(_get(config, "dataloader_num_workers", 0)),
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not bool(_get(config, "load_in_4bit", False)):
        predictor.model.to(device)
    else:
        device = predictor.device

    trainable_parameters = [parameter for parameter in predictor.model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("PEFT produced no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(_get(config, "lr", 2e-4)),
        weight_decay=float(_get(config, "l2", 0.0)),
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
    configured_max_steps = int(_get(config, "max_steps", -1))
    total_steps = configured_max_steps if configured_max_steps > 0 else updates_per_epoch * num_epochs
    effective_epochs = (
        max(num_epochs, math.ceil(total_steps / max(1, updates_per_epoch)))
        if configured_max_steps > 0
        else num_epochs
    )
    warmup_steps = int(math.ceil(total_steps * float(_get(config, "warmup_ratio", 0.03))))
    scheduler_name = str(_get(config, "lr_scheduler_type", "cosine"))
    if scheduler_name != "cosine":
        raise ValueError("The isolated LLM-N loop currently supports lr_scheduler_type=cosine only")

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    precision = str(_get(config, "training_precision", "bf16")).lower()
    use_bf16 = bool(torch.cuda.is_available() and precision in {"bf16", "bf16-mixed", "bfloat16"})
    use_fp16 = bool(torch.cuda.is_available() and precision in {"fp16", "fp16-mixed", "float16"})
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None
    if use_fp16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            # PyTorch 1.13 exposes GradScaler only under torch.cuda.amp.
            scaler = torch.cuda.amp.GradScaler()
    trainer_output = run_root / "trainer"
    trainer_output.mkdir(parents=True, exist_ok=True)
    context_events_path = run_root / "training_context_events.jsonl"
    logging_steps = max(1, int(_get(config, "logging_steps", 10)))
    save_strategy = str(_get(config, "save_strategy", "steps"))
    if save_strategy not in {"steps", "no"}:
        raise ValueError("save_strategy must be steps or no")
    save_steps = max(1, int(_get(config, "save_steps", 500)))
    save_total_limit = max(0, int(_get(config, "save_total_limit", 2)))
    grad_clip = float(_get(config, "grad_clip", 1.0))

    def save_training_checkpoint(step: int) -> None:
        checkpoint_dir = trainer_output / f"checkpoint-{step}"
        predictor.model.save_pretrained(checkpoint_dir)
        predictor.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "global_step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / "training_state.pt",
        )
        if save_total_limit > 0:
            checkpoints = sorted(
                trainer_output.glob("checkpoint-*"),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            for stale_checkpoint in checkpoints[:-save_total_limit]:
                shutil.rmtree(stale_checkpoint)

    def append_context_event(event: Mapping[str, Any]) -> None:
        with context_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True) + "\n")

    predictor.model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_steps_since_update = 0
    loss_sum = 0.0
    loss_count = 0
    samples_seen = 0
    truncation_count = 0
    overflow_count = 0
    started_at = time.perf_counter()
    stop_training = False
    try:
        for epoch in range(effective_epochs):
            for batch_index, batch in enumerate(train_loader):
                metadata = batch.pop("_llm_n_metadata")
                samples_seen += len(metadata)
                for sample_metadata in metadata:
                    if sample_metadata["truncated"]:
                        truncation_count += 1
                        append_context_event(
                            {
                                "status": "truncated",
                                "epoch": epoch + 1,
                                **sample_metadata,
                            }
                        )
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=use_bf16 or use_fp16,
                ):
                    output = predictor.forward(**batch)
                    raw_loss = output.loss
                    loss = raw_loss / grad_accumulation
                if not bool(torch.isfinite(raw_loss.detach())):
                    raise RuntimeError(f"Non-finite LLM-N training loss at micro step {batch_index}")
                if use_fp16:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += float(raw_loss.detach().float().cpu().item())
                loss_count += 1
                micro_steps_since_update += 1
                is_last_batch = batch_index + 1 == len(train_loader)
                if micro_steps_since_update < grad_accumulation and not is_last_batch:
                    continue

                if use_fp16:
                    assert scaler is not None
                    scaler.unscale_(optimizer)
                if is_last_batch and micro_steps_since_update < grad_accumulation:
                    remainder_scale = grad_accumulation / micro_steps_since_update
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(remainder_scale)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
                if use_fp16:
                    assert scaler is not None
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_steps_since_update = 0
                global_step += 1

                if global_step % logging_steps == 0:
                    print(
                        json.dumps(
                            {
                                "llm_n_train_step": global_step,
                                "epoch": epoch + 1,
                                "mean_loss": loss_sum / max(1, loss_count),
                                "learning_rate": scheduler.get_last_lr()[0],
                            },
                            sort_keys=True,
                        )
                    )
                if save_strategy == "steps" and global_step % save_steps == 0:
                    save_training_checkpoint(global_step)
                if global_step >= total_steps:
                    stop_training = True
                    break
            if stop_training:
                break
    except ContextOverflowError as error:
        overflow_count += 1
        failure = {
            "status": "overflow",
            "task_name": getattr(error, "task_name", None),
            "sample_index": getattr(error, "sample_index", None),
            "context": error.context,
            "original_prompt_token_count": error.original_tokens,
            "original_token_count": getattr(
                error,
                "original_total_token_count",
                error.original_tokens,
            ),
            "answer_token_count": getattr(error, "answer_token_count", None),
            "max_context_length": getattr(error, "max_context_length", None),
            "capacity": error.capacity,
            "truncate_mode": predictor.tokenization.truncate_mode,
            "error": str(error),
        }
        append_context_event(failure)
        _write_json(
            run_root / "training_failure.json",
            {
                "mode": "llm_n_sft",
                "global_steps": global_step,
                "truncation_count": truncation_count,
                "overflow_count": overflow_count,
                "context_event": failure,
            },
        )
        raise

    adapter_dir = Path(str(_get(config, "adapter_output_dir", run_root / "adapter")))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    predictor.model.save_pretrained(adapter_dir)
    predictor.tokenizer.save_pretrained(adapter_dir)
    elapsed_seconds = time.perf_counter() - started_at
    metrics = {
        "mode": "llm_n_sft",
        "tasks": tasks,
        "num_train_samples": len(train_dataset),
        "adapter_dir": str(adapter_dir.resolve()),
        "train_metrics": {
            "global_steps": global_step,
            "mean_loss": loss_sum / max(1, loss_count),
            "train_runtime_seconds": elapsed_seconds,
            "train_samples_seen": samples_seen,
            "train_samples_per_second": samples_seen / max(elapsed_seconds, 1e-9),
            "truncation_count": truncation_count,
            "overflow_count": overflow_count,
            "context_events_file": (
                str(context_events_path.resolve()) if context_events_path.exists() else None
            ),
        },
    }
    _write_json(run_root / "training_metrics.json", metrics)
    return metrics


def _cuda_devices_for_peak() -> list[int]:
    import torch

    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def _reset_peak_memory() -> None:
    import torch

    for device_index in _cuda_devices_for_peak():
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)


def _read_peak_memory() -> tuple[int, dict[str, int]]:
    import torch

    by_device: dict[str, int] = {}
    for device_index in _cuda_devices_for_peak():
        torch.cuda.synchronize(device_index)
        by_device[f"cuda:{device_index}"] = int(torch.cuda.max_memory_allocated(device_index))
    return max(by_device.values(), default=0), by_device


def _clear_cuda_after_oom() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_index in _cuda_devices_for_peak():
            torch.cuda.synchronize(device_index)


def _warm_up(predictor: LLMNPredictor, dataset: LLMNTaskDataset, requested: int) -> int:
    warm_up_samples = max(10, int(requested))
    if len(dataset) == 0:
        raise RuntimeError(f"Cannot warm up on empty task {dataset.task_name}")
    completed = 0
    attempts = 0
    # Probe a bounded number of real samples first.  If every sampled subgraph
    # overflows under truncate_mode=none, use a short neutral prompt for the
    # hardware warm-up so the measured pass can still record every overflow.
    max_real_attempts = min(max(warm_up_samples * 2, warm_up_samples), len(dataset))
    while completed < warm_up_samples and attempts < max_real_attempts:
        sample = dataset[attempts % len(dataset)]
        attempts += 1
        try:
            predictor.generate(sample)
            completed += 1
        except ContextOverflowError:
            continue
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            _clear_cuda_after_oom()
    neutral_prompt = (
        "Task Description:\nWarm-up only.\n\nTarget:\n[Node A]\n\nNodes:\n"
        "- [Node A]: warm-up\n\nEdges:\n- (none)\n\nQuestion:\nWarm-up?\n\nAnswer:"
    )
    neutral_attempts = 0
    while completed < warm_up_samples and neutral_attempts < warm_up_samples * 2:
        neutral_attempts += 1
        try:
            predictor.generate(neutral_prompt)
            completed += 1
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            _clear_cuda_after_oom()
    if completed < warm_up_samples:
        raise RuntimeError(
            f"Only {completed}/{warm_up_samples} warm-up generations completed for {dataset.task_name}; "
            "the neutral warm-up prompt repeatedly ran out of memory"
        )
    return completed


def infer_task(
    config: Mapping[str, Any],
    predictor: LLMNPredictor,
    dataset: LLMNTaskDataset,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = build_taglas_text_accuracy(dataset.task_name)
    parser = PredictionParser(dataset.task_name, dataset.label_space)
    warm_up_count = _warm_up(predictor, dataset, int(_get(config, "warm_up_samples", 10)))
    _reset_peak_memory()

    total_latency_ms = 0.0
    prefill_latency_ms = 0.0
    decode_latency_ms = 0.0
    input_token_count = 0
    original_input_token_count = 0
    output_token_count = 0
    successful_latency_samples = 0
    truncation_count = 0
    overflow_count = 0
    oom_count = 0
    save_prompts = bool(_get(config, "save_prompts", True))
    prediction_path = output_dir / "predictions.jsonl"

    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for index in range(len(dataset)):
            sample = dataset[index]
            prepared_prompt = None
            record: dict[str, Any] = {
                "task_name": dataset.task_name,
                "split": dataset.split,
                "sample_index": sample.sample_index,
                "target_node_ids": list(sample.target_node_ids),
                "target_label": sample.target_label,
                "raw_generation": "",
                "normalized_prediction": "",
                "parse_mode": "",
                "parse_matched": False,
                "correct": False,
                "status": "ok",
                "truncated": False,
                "original_input_token_count": None,
                "truncated_input_token_count": None,
                "input_token_count": 0,
                "output_token_count": 0,
                "prefill_latency_ms": 0.0,
                "decode_latency_ms": 0.0,
                "total_latency_ms": 0.0,
            }
            if save_prompts:
                record["serialized_input"] = sample.prompt
            try:
                encode_prompt = getattr(predictor.tokenization, "encode_prompt", None)
                if callable(encode_prompt):
                    prepared_prompt = encode_prompt(sample.prompt)
                generation = predictor.generate(sample)
                parsed = parser.parse(generation.raw_text)
                record.update(
                    raw_generation=generation.raw_text,
                    normalized_prediction=parsed.normalized_label,
                    parse_mode=parsed.parse_mode,
                    parse_matched=parsed.matched,
                    truncated=generation.truncated,
                    original_input_token_count=generation.original_input_token_count,
                    truncated_input_token_count=(
                        generation.input_token_count if generation.truncated else None
                    ),
                    input_token_count=generation.input_token_count,
                    output_token_count=generation.output_token_count,
                    prefill_latency_ms=generation.prefill_latency_ms,
                    decode_latency_ms=generation.decode_latency_ms,
                    total_latency_ms=generation.total_latency_ms,
                )
                record["correct"] = (
                    normalize_like_taglas(parsed.normalized_label)
                    == normalize_like_taglas(sample.target_label)
                    and bool(parsed.normalized_label)
                )
                total_latency_ms += generation.total_latency_ms
                prefill_latency_ms += generation.prefill_latency_ms
                decode_latency_ms += generation.decode_latency_ms
                input_token_count += generation.input_token_count
                original_input_token_count += generation.original_input_token_count
                output_token_count += generation.output_token_count
                successful_latency_samples += 1
                truncation_count += int(generation.truncated)
            except ContextOverflowError as error:
                overflow_count += 1
                original_input_token_count += error.original_tokens
                record.update(
                    status="overflow",
                    original_input_token_count=error.original_tokens,
                    error=str(error),
                )
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                oom_count += 1
                record.update(status="oom", error=str(error))
                if prepared_prompt is not None:
                    record.update(
                        truncated=prepared_prompt.truncated,
                        original_input_token_count=prepared_prompt.original_token_count,
                        truncated_input_token_count=(
                            prepared_prompt.token_count if prepared_prompt.truncated else None
                        ),
                        input_token_count=prepared_prompt.token_count,
                    )
                    original_input_token_count += prepared_prompt.original_token_count
                    input_token_count += prepared_prompt.token_count
                    truncation_count += int(prepared_prompt.truncated)
                _clear_cuda_after_oom()

            update_text_accuracy(evaluator, str(record["normalized_prediction"]), sample.target_label)
            prediction_file.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
            prediction_file.flush()

    peak_gpu_memory, peak_by_device = _read_peak_memory()
    accuracy = compute_text_accuracy(evaluator)
    denominator = successful_latency_samples or 1
    metrics = {
        "task_name": dataset.task_name,
        "split": dataset.split,
        "accuracy": accuracy,
        "num_samples": len(dataset),
        "num_latency_samples": successful_latency_samples,
        "warm_up_samples": warm_up_count,
        "total_latency_ms": total_latency_ms,
        "latency_per_sample_ms": total_latency_ms / denominator,
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_latency_per_sample_ms": prefill_latency_ms / denominator,
        "decode_latency_ms": decode_latency_ms,
        "decode_latency_per_sample_ms": decode_latency_ms / denominator,
        "input_token_count": input_token_count,
        "original_input_token_count": original_input_token_count,
        "output_token_count": output_token_count,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "peak_gpu_memory_by_device": peak_by_device,
        "truncation_count": truncation_count,
        "overflow_count": overflow_count,
        "oom_count": oom_count,
        "overflow_oom_count": overflow_count + oom_count,
        "truncate_mode": predictor.tokenization.truncate_mode,
        "predictions_file": str(prediction_path.resolve()),
    }
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_inference(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    mode = str(_get(config, "llm_n_mode", "llm_n_zero_shot"))
    if mode == "llm_n_zero_shot" and config.get("adapter_path"):
        raise ValueError("llm_n_zero_shot must not load an adapter")
    if mode == "llm_n_sft" and not config.get("adapter_path"):
        raise ValueError("llm_n_sft inference requires adapter_path")

    tasks = _normalize_tasks(_get(config, "eval_task_names", _get(config, "task_names", [])), "eval_task_names")
    split = str(_get(config, "eval_split", "test"))
    datasets = _build_datasets(config, tasks, split=split, inference=True)
    # Model loading occurs after dataset construction and before warm-up; neither
    # download/model initialization is inside any inference CUDA event.
    predictor = LLMNPredictor.from_pretrained(config, for_training=False)
    task_metrics: dict[str, Any] = {}
    for dataset in datasets:
        task_dir = run_root / dataset.task_name / split
        task_metrics[dataset.task_name] = infer_task(config, predictor, dataset, task_dir)
    summary = {
        "mode": mode,
        "model_name_or_path": _get(config, "model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.2"),
        "adapter_path": config.get("adapter_path"),
        "tasks": task_metrics,
    }
    _write_json(run_root / "inference_summary.json", summary)
    return summary


def run(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        config = vars(config)
    config = dict(config)
    model_type = str(_get(config, "model_type", "gofa"))
    enabled = bool(_get(config, "llm_n_enabled", False))
    if model_type != "llm_n" and not enabled:
        raise ValueError("LLM-N runner requires model_type=llm_n or llm_n_enabled=true")
    mode = str(_get(config, "llm_n_mode", "llm_n_zero_shot"))
    if mode not in LLM_N_MODES:
        raise ValueError(f"llm_n_mode must be one of {sorted(LLM_N_MODES)}")
    run_mode = str(_get(config, "run_mode", "inference")).lower()
    run_mode = {"ft": "train", "inf": "inference"}.get(run_mode, run_mode)
    if run_mode == "train" and mode != "llm_n_sft":
        raise ValueError("Training is only valid with llm_n_mode=llm_n_sft")
    if run_mode not in {"train", "inference"}:
        raise ValueError("run_mode must be train/inference (or GOFA aliases ft/inf)")

    _set_seed(int(_get(config, "seed", 1)))
    run_root = _run_root(config)
    _write_json(run_root / "resolved_config.json", config)
    if run_mode == "train":
        return train_sft(config, run_root)
    return run_inference(config, run_root)
