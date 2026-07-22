import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modules.gofa.workload_profile import normalize_workload_profile, saved_workload_name


DEFAULT_TASKS = ("cora_node", "cora_link", "pubmed_node", "wikics", "arxiv")
TASK_WAYS = {
    "cora_node": 7,
    "cora_link": 2,
    "pubmed_node": 3,
    "wikics": 10,
    "arxiv": 40,
}
PROFILE_MODES = ("nocache_bf16", "cache_bf16", "cache_w8a8_m4k2v2")


def add_common_arguments(parser):
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--load-dir", required=True)
    parser.add_argument("--profile-name", default="large_h6_n32_s100")
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--max-nodes-per-hop", type=int, default=32)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--kv-policy", choices=("all", "target_1hop"), default="target_1hop")
    parser.add_argument("--kv-target-hops", type=int, default=1)
    parser.add_argument("--overwrite-suite", action="store_true")
    return parser


def normalize_args(args):
    tasks = []
    for value in args.tasks:
        tasks.extend(part for part in str(value).split(",") if part)
    if not tasks:
        raise ValueError("--tasks must not be empty")
    unsupported_tasks = sorted(set(tasks) - set(TASK_WAYS))
    if unsupported_tasks:
        raise ValueError(f"Unsupported task(s): {unsupported_tasks}; expected one of {sorted(TASK_WAYS)}")
    if int(args.reps) <= 0:
        raise ValueError("--reps must be positive")
    if str(args.kv_policy) not in {"all", "target_1hop"}:
        raise ValueError("--kv-policy must be one of all, target_1hop")
    if int(args.kv_target_hops) < 0:
        raise ValueError("--kv-target-hops must be non-negative")
    for field_name in ("data_root", "model_name_or_path", "checkpoint_dir", "load_dir"):
        raw_path = str(getattr(args, field_name, "") or "").strip()
        if not raw_path:
            raise ValueError(f"--{field_name.replace('_', '-')} must not be empty")
        path = Path(raw_path).expanduser().resolve()
        expects_file = field_name == "load_dir"
        if expects_file and not path.is_file():
            raise ValueError(f"--load-dir must be an existing file: {path}")
        if not expects_file and not path.is_dir():
            raise ValueError(f"--{field_name.replace('_', '-')} must be an existing directory: {path}")
        setattr(args, field_name, str(path))
    profile = normalize_workload_profile({
        "name": args.profile_name,
        "seed": args.seed,
        "samples_per_split": args.samples_per_split,
        "hops": args.hops,
        "max_nodes_per_hop": args.max_nodes_per_hop,
    })
    return profile, tasks


def repository_commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _suite_identity(args, profile, tasks):
    return {
        "repository_commit_sha": repository_commit_sha(),
        "profile": profile,
        "tasks": list(tasks),
        "reps": int(args.reps),
        "runtime": {
            "data_root_path": str(args.data_root),
            "model_name_or_path": str(args.model_name_or_path),
            "checkpoint_dir": str(args.checkpoint_dir),
            "load_dir": str(args.load_dir),
            "load_model": True,
        },
        "kv_policy": {
            "name": str(args.kv_policy),
            "target_hops": int(args.kv_target_hops),
        },
    }


def _validate_locked_suite(existing, expected_identity):
    mismatches = []
    for field, expected in expected_identity.items():
        actual = existing.get(field)
        if actual != expected:
            mismatches.append(f"{field}: locked={actual!r}, requested={expected!r}")
    if mismatches:
        raise RuntimeError(
            "Large-workload suite identity mismatch; use --overwrite-suite only to intentionally rebuild it: "
            + "; ".join(mismatches)
        )


def load_locked_suite(args):
    profile, tasks = normalize_args(args)
    manifest_path = suite_root(args) / "suite_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Large-workload suite manifest does not exist: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    _validate_locked_suite(manifest, _suite_identity(args, profile, tasks))
    return manifest_path, manifest


def suite_root(args):
    return Path(args.profile_root).expanduser().resolve() / args.profile_name


def suite_paths(args):
    root = suite_root(args)
    return {
        "root": root,
        "configs": root / "configs",
        "plans": root / "plans",
        "manifests": root / "manifests",
        "traces": root / "traces",
        "latency": root / "latency",
        "logs": root / "logs",
        "full_cache": root / "cache" / "full" / "shared",
        "quant_cache": root / "cache" / "quant" / "m4k2v2",
        "summary": root / "summary",
    }


def _base_config(args, profile, task, from_saved=True):
    samples = profile["samples_per_split"]
    ways = TASK_WAYS[task]
    return {
        "run_mode": "inf",
        "mode": "generate",
        "data_root_path": str(args.data_root),
        "model_name_or_path": str(args.model_name_or_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "load_dir": str(args.load_dir),
        "load_model": True,
        "seed": profile["seed"],
        "batch_size": 1,
        "eval_sample_size": samples,
        "sample_size_per_task": samples,
        "train_sample_size": -1,
        "skip_validation": False,
        "task_names": [task],
        "train_task_names": [task],
        "eval_task_names": [task],
        "ways": ways,
        "inf_sample_size_per_task": [samples],
        "inf_hops": [profile["hops"]],
        "inf_max_nodes_per_hops": [profile["max_nodes_per_hop"]],
        "inf_ways": [ways],
        "inf_instructs": [True],
        "inf_selections": [True],
        "inf_from_saved": bool(from_saved),
        "inf_save_names": {},
        "save_data": True,
        "workload_profile": profile,
        "gofa_query_trace": {"enabled": False},
        "gofa_per_query_latency": {"enabled": False},
        "encoder_cache_manifest": {"enabled": False},
        "scheme_b_quant": {"enabled": False},
        "scheme_b_weight_quant": {"enabled": False},
        "scheme_b_activation_quant": {"enabled": False},
        "scheme_b_int_gemm": {"enabled": False},
        "scheme_b_quant_kv_attention": {"enabled": False},
    }


def _quant_config(args, paths, task):
    return {
        "enabled": True,
        "base_bits": 4,
        "delta_bits": 4,
        "memory_base_bits": 4,
        "key_base_bits": 2,
        "value_base_bits": 2,
        "memory_delta_bits": 4,
        "key_delta_bits": 2,
        "value_delta_bits": 2,
        "cache_dir": str(paths["quant_cache"] / task),
        "fake_quant": True,
        "strict": True,
        "debug_zero_base": False,
        "target_aware_delta": False,
        "load_memory_delta": False,
        "load_key_delta": False,
        "load_value_delta": False,
        "load_key_base": True,
        "load_value_base": True,
        "kv_base_load_policy": str(args.kv_policy),
        "kv_base_target_hops": int(args.kv_target_hops),
    }


def _int_gemm_config():
    return {
        "enabled": True,
        "target": "suffix_transformer",
        "weight_bits": 8,
        "activation_bits": 8,
        "backend": "torch_int_mm",
        "quantize_attention": True,
        "quantize_mlp": True,
        "quantize_layernorm": False,
        "fallback_to_fake_quant": False,
    }


def _quant_attention_config():
    return {
        "enabled": True,
        "backend": "torch_int_mm_qscale_fold",
        "key_scale_fold_into_q": True,
        "quantize_query_bits": 8,
        "key_bits": 2,
        "value_bits": 2,
        "use_int_qk": True,
        "pv_compute_mode": "int_pv",
        "quantize_prob_bits": 8,
        "prob_quant_granularity": "per_query",
        "prob_quant_unsigned": False,
        "prob_quant_qmax": 127,
        "fallback_to_fp_attention": False,
        "fallback_to_scale_delayed_v": False,
        "compare_int_pv_with_fp_pv": False,
    }


def build_task_configs(args, profile, task):
    paths = suite_paths(args)
    common_cache = {
        "use_encoder_cache": True,
        "encoder_cache_dir": str(paths["full_cache"]),
        "encoder_cache_mode": "memory_kv",
        "encoder_cache_skip_nog": True,
    }
    configs = {}

    generation = _base_config(args, profile, task, from_saved=False)
    generation.update({"use_encoder_cache": False})
    configs["workload_generate"] = generation

    full_cache = _base_config(args, profile, task)
    full_cache.update(common_cache)
    full_cache["encoder_cache_manifest"] = {
        "enabled": True,
        "output_path": str(paths["manifests"] / f"{task}.json"),
        "append": False,
        "log_interval": 20,
    }
    configs["full_cache"] = full_cache

    formal_trace = _base_config(args, profile, task)
    formal_trace.update(common_cache)
    formal_trace["scheme_b_quant"] = _quant_config(args, paths, task)
    formal_trace["gofa_query_trace"] = {
        "enabled": True,
        "output_dir": str(paths["traces"] / f"{task}_formal_v1"),
        "max_queries": 2 * profile["samples_per_split"],
        "include_token_ids": False,
        "include_text_preview": True,
        "rank_zero_only": True,
        "strict": True,
        "resume": False,
    }
    configs["formal_trace"] = formal_trace

    for mode in PROFILE_MODES:
        latency = _base_config(args, profile, task)
        if mode != "nocache_bf16":
            latency.update(common_cache)
        else:
            latency["use_encoder_cache"] = False
        if mode == "cache_w8a8_m4k2v2":
            latency["scheme_b_quant"] = _quant_config(args, paths, task)
            latency["scheme_b_int_gemm"] = _int_gemm_config()
            latency["scheme_b_quant_kv_attention"] = _quant_attention_config()
        latency["gofa_per_query_latency"] = {
            "enabled": True,
            "profile_mode": mode,
            "output_csv": str(paths["latency"] / mode / f"{task}.csv"),
            "trace_index_path": str(paths["traces"] / f"{task}_formal_v1" / "trace_index.jsonl"),
            "strict_trace_match": True,
            "cuda_sync": True,
            "export_wall_time": True,
            "export_gpu_time": True,
            "export_detail_gpu_time": True,
            "append": False,
            "rank_zero_only": True,
        }
        configs[f"gpu_{mode}"] = latency
    return configs


def write_json_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_suite(args):
    profile, tasks = normalize_args(args)
    paths = suite_paths(args)
    manifest_path = paths["root"] / "suite_manifest.json"
    expected_identity = _suite_identity(args, profile, tasks)
    existing_manifest = None
    if manifest_path.is_file():
        with manifest_path.open() as handle:
            existing_manifest = json.load(handle)
        if not bool(getattr(args, "overwrite_suite", False)):
            _validate_locked_suite(existing_manifest, expected_identity)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    config_paths = {}
    for task in tasks:
        config_paths[task] = {}
        for stage, config in build_task_configs(args, profile, task).items():
            path = paths["configs"] / task / f"{stage}.yaml"
            write_json_yaml(path, config)
            config_paths[task][stage] = str(path)
    manifest = {
        "suite_format": "gofa_large_workload_suite_v1",
        **expected_identity,
        "saved_workloads": {
            task: {
                split: saved_workload_name(profile, task, split)
                for split in ("val", "test")
            }
            for task in tasks
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "configs": config_paths,
    }
    if existing_manifest is None or bool(getattr(args, "overwrite_suite", False)):
        write_json_yaml(manifest_path, manifest)
    else:
        manifest = existing_manifest
    return manifest_path, manifest


def stage_commands(args, stage):
    manifest_path, manifest = prepare_suite(args)
    commands = []
    configs = manifest["configs"]
    if stage in {"workload", "full-cache", "formal-trace"}:
        config_key = {
            "workload": "workload_generate",
            "full-cache": "full_cache",
            "formal-trace": "formal_trace",
        }[stage]
        commands = [
            ["python3", str(REPOSITORY_ROOT / "run_gofa.py"), "--override", configs[task][config_key]]
            for task in manifest["tasks"]
        ]
    elif stage == "gpu":
        for mode in PROFILE_MODES:
            for task in manifest["tasks"]:
                for rep in range(manifest["reps"]):
                    command = [
                        "python3",
                        str(REPOSITORY_ROOT / "run_gofa.py"),
                        "--override",
                        configs[task][f"gpu_{mode}"],
                    ]
                    if rep > 0:
                        command.extend(["gofa_per_query_latency_append", "True"])
                    commands.append(command)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    plan_path = Path(manifest["paths"]["plans"]) / f"{stage}.sh"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for command in commands:
            handle.write(" ".join(shlex.quote(part) for part in command) + "\n")
    os.chmod(plan_path, 0o755)
    return manifest_path, plan_path, commands
