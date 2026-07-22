#!/usr/bin/env python3
"""Interrupt and resume a real four-step Cora-node Mistral LoRA run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.executable,
        "taglas_available": importlib.util.find_spec("TAGLAS") is not None,
        "transformers_available": importlib.util.find_spec("transformers") is not None,
        "peft_available": importlib.util.find_spec("peft") is not None,
    }
    try:
        import torch

        report.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
    except Exception as error:  # pragma: no cover - diagnostic path
        report.update(cuda_available=False, torch_error=str(error))
    report["ready"] = bool(
        report.get("cuda_available")
        and report["taglas_available"]
        and report["transformers_available"]
        and report["peft_available"]
    )
    return report


def common_training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "run_llm_n.py"),
        "--config",
        str(Path(args.train_config).resolve()),
        "--tasks",
        "cora_node",
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "run_name",
        args.run_name,
        "sample_size_per_task",
        "[32]",
        "hops",
        "[3]",
        "max_nodes_per_hop",
        "[5]",
        "max_steps",
        "4",
        "num_epochs",
        "1",
        "save_strategy",
        "steps",
        "save_steps",
        "2",
        "save_total_limit",
        "2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="configs/llm_n_train_config.yaml")
    parser.add_argument("--inference-config", default="configs/llm_n_inference_config.yaml")
    parser.add_argument("--output-dir", default="outputs/llm_n_cora_resume_smoke")
    parser.add_argument("--run-name", default="cora_resume_smoke")
    parser.add_argument("--checkpoint-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()

    report = environment_report()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.check_environment:
        return
    if not report["ready"]:
        raise RuntimeError(
            "Real resume smoke requires CUDA plus TAGLAS, Transformers, and PEFT; "
            "run with --check-environment for details"
        )

    output_root = Path(args.output_dir).resolve()
    run_root = output_root / args.run_name
    if run_root.exists():
        raise FileExistsError(
            f"Refusing to reuse smoke directory {run_root}; remove or rename it explicitly"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    initial_log = output_root / "initial_interrupted.log"
    resume_log = output_root / "resume.log"
    inference_log = output_root / "inference.log"
    checkpoint_two = run_root / "trainer" / "checkpoint-2"
    checkpoint_marker = checkpoint_two / "_SUCCESS"

    training_command = common_training_command(args)
    with initial_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            training_command,
            cwd=REPOSITORY_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + args.checkpoint_timeout_seconds
        while not checkpoint_marker.is_file():
            if process.poll() is not None:
                raise RuntimeError(
                    f"Initial training exited with code {process.returncode} before checkpoint-2; "
                    f"see {initial_log}"
                )
            if time.monotonic() >= deadline:
                process.kill()
                raise TimeoutError(f"Timed out waiting for {checkpoint_marker}")
            time.sleep(0.1)
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    checkpoint_four = run_root / "trainer" / "checkpoint-4"
    if checkpoint_four.exists():
        raise RuntimeError(
            "Initial process reached checkpoint-4 before interruption; use a shorter polling interval"
        )
    resume_command = training_command[:8] + [
        "--resume-from-checkpoint",
        str(checkpoint_two),
    ] + training_command[8:]
    with resume_log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            resume_command,
            cwd=REPOSITORY_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    with (run_root / "training_metrics.json").open("r", encoding="utf-8") as handle:
        training_metrics = json.load(handle)
    if training_metrics["train_metrics"]["global_steps"] != 4:
        raise AssertionError(f"Resume stopped at unexpected step: {training_metrics}")

    adapter_dir = run_root / "adapter"
    inference_root = output_root / "inference"
    inference_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "run_llm_n.py"),
        "--config",
        str(Path(args.inference_config).resolve()),
        "--tasks",
        "cora_node",
        "--mode",
        "llm_n_sft",
        "--adapter-path",
        str(adapter_dir),
        "--output-dir",
        str(inference_root),
        "run_name",
        "cora_resume_inference",
        "inf_sample_size_per_task",
        "[20]",
    ]
    with inference_log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            inference_command,
            cwd=REPOSITORY_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    inference_summary_path = inference_root / "cora_resume_inference" / "inference_summary.json"
    with inference_summary_path.open("r", encoding="utf-8") as handle:
        inference_summary = json.load(handle)
    evaluated = inference_summary["tasks"]["cora_node"]["num_samples"]
    if evaluated != 20:
        raise AssertionError(f"Expected 20 inference samples, got {evaluated}")

    summary = {
        "passed": True,
        "checkpoint_2": str(checkpoint_two),
        "final_global_step": training_metrics["train_metrics"]["global_steps"],
        "final_adapter": str(adapter_dir),
        "inference_samples": evaluated,
        "logs": {
            "initial": str(initial_log),
            "resume": str(resume_log),
            "inference": str(inference_log),
        },
    }
    summary_path = output_root / "resume_smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
