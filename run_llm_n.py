"""Standalone command-line entry point for LLM-N-SFT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from llm_n.runner import run


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if len(overrides) % 2:
        raise ValueError("Config overrides must be KEY VALUE pairs")
    for key, raw_value in zip(overrides[0::2], overrides[1::2]):
        _set_nested(config, key, yaml.safe_load(raw_value))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure LLM-as-Predictor baseline for GOFA/TAGLAS")
    parser.add_argument("--config", required=True, help="Path to an LLM-N YAML config")
    parser.add_argument("--tasks", nargs="+", help="Override train/eval task names")
    parser.add_argument("--mode", choices=["llm_n_zero_shot", "llm_n_sft"])
    parser.add_argument("--run-mode", choices=["train", "inference"])
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional KEY VALUE overrides")
    args = parser.parse_args()

    config = _load_config(args.config, args.opts)
    if args.mode:
        config["llm_n_mode"] = args.mode
    if args.run_mode:
        config["run_mode"] = args.run_mode
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.tasks:
        field = "train_task_names" if config.get("run_mode") == "train" else "eval_task_names"
        config[field] = args.tasks
    run(config)


if __name__ == "__main__":
    main()
