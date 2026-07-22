import hashlib
import json
import re


DEFAULT_WORKLOAD_PROFILE = {
    "name": "canonical_h3_n10_s100",
    "seed": 1,
    "samples_per_split": 100,
    "hops": 3,
    "max_nodes_per_hop": 10,
}


def _first(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _mapping(value):
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def normalize_workload_profile(profile=None, *, fallback=None):
    values = dict(DEFAULT_WORKLOAD_PROFILE)
    values.update(_mapping(fallback))
    values.update(_mapping(profile))
    normalized = {
        "name": str(values.get("name") or "").strip(),
        "seed": int(values["seed"]),
        "samples_per_split": int(values["samples_per_split"]),
        "hops": int(values["hops"]),
        "max_nodes_per_hop": int(values["max_nodes_per_hop"]),
    }
    if not normalized["name"]:
        raise ValueError("workload_profile.name must not be empty")
    for key in ("samples_per_split", "hops", "max_nodes_per_hop"):
        if normalized[key] <= 0:
            raise ValueError(f"workload_profile.{key} must be positive")
    return normalized


def workload_profile_from_runtime(params):
    fallback = {
        "seed": getattr(params, "seed", DEFAULT_WORKLOAD_PROFILE["seed"]),
        "samples_per_split": getattr(
            params,
            "eval_sample_size",
            DEFAULT_WORKLOAD_PROFILE["samples_per_split"],
        ),
        "hops": _first(getattr(params, "inf_hops", None), DEFAULT_WORKLOAD_PROFILE["hops"]),
        "max_nodes_per_hop": _first(
            getattr(params, "inf_max_nodes_per_hops", None),
            DEFAULT_WORKLOAD_PROFILE["max_nodes_per_hop"],
        ),
    }
    return normalize_workload_profile(getattr(params, "workload_profile", None), fallback=fallback)


def validate_runtime_sampling(profile, *, seed, eval_sample_size, tasks, sample_sizes, hops, max_nodes):
    profile = normalize_workload_profile(profile)
    errors = []
    if int(seed) != profile["seed"]:
        errors.append(f"seed must match workload_profile.seed={profile['seed']}")
    if int(eval_sample_size) != profile["samples_per_split"]:
        errors.append(
            "eval_sample_size must match "
            f"workload_profile.samples_per_split={profile['samples_per_split']}"
        )
    expected_by_field = (
        ("inf_sample_size_per_task", sample_sizes, profile["samples_per_split"]),
        ("inf_hops", hops, profile["hops"]),
        ("inf_max_nodes_per_hops", max_nodes, profile["max_nodes_per_hop"]),
    )
    for field_name, raw_values, expected in expected_by_field:
        values = list(raw_values or [])
        if len(values) != len(tasks) or any(int(value) != expected for value in values):
            errors.append(f"{field_name} must contain {expected} once per eval task")
    return errors


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-")


def saved_workload_name(profile, task, split):
    profile = normalize_workload_profile(profile)
    return "__".join((
        _safe_name(profile["name"]),
        _safe_name(task),
        _safe_name(split),
        f"s{profile['seed']}",
        f"h{profile['hops']}",
        f"n{profile['max_nodes_per_hop']}",
    ))


def resolve_saved_workload_names(profile, tasks, split, configured=None):
    profile = normalize_workload_profile(profile)
    tasks = list(tasks)
    if configured is None:
        return [None for _ in tasks]
    if isinstance(configured, dict):
        configured = configured.get(split)
    if configured in (None, "", []):
        return [saved_workload_name(profile, task, split) for task in tasks]
    if isinstance(configured, str):
        configured = [configured for _ in tasks]
    configured = list(configured)
    if len(configured) != len(tasks):
        raise ValueError(f"inf_save_names for split={split} must contain one entry per eval task")
    names = []
    for task, template in zip(tasks, configured):
        name = str(template).format(
            profile=profile["name"],
            task=task,
            split=split,
            seed=profile["seed"],
            hops=profile["hops"],
            max_nodes=profile["max_nodes_per_hop"],
            max_nodes_per_hop=profile["max_nodes_per_hop"],
        )
        required = (
            _safe_name(profile["name"]),
            _safe_name(task),
            _safe_name(split),
            f"s{profile['seed']}",
            f"h{profile['hops']}",
            f"n{profile['max_nodes_per_hop']}",
        )
        safe = _safe_name(name)
        missing = [part for part in required if part not in safe]
        if missing:
            raise ValueError(
                f"inf_save_names entry {name!r} is missing workload identity components {missing}"
            )
        names.append(name)
    return names


def resolve_from_saved_flags(value, task_count):
    if isinstance(value, (list, tuple)):
        if len(value) != int(task_count):
            raise ValueError("inf_from_saved must contain one value per eval task")
        return [bool(item) for item in value]
    return [bool(value) for _ in range(int(task_count))]


def _json_value(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return int(value)
    return str(value)


def graph_signature_payload(task, split, query_index, *, node_map, edge_map, edge_index, target_index, question_index):
    return {
        "task": str(task),
        "split": str(split),
        "query_index": int(query_index),
        "node_map": _json_value(node_map),
        "edge_map": _json_value(edge_map),
        "edge_index": _json_value(edge_index),
        "target_index": _json_value(target_index),
        "question_index": _json_value(question_index),
    }


def graph_signature(task, split, query_index, graph=None, **components):
    if graph is not None:
        for name in ("node_map", "edge_map", "edge_index", "target_index", "question_index"):
            components.setdefault(name, getattr(graph, name, None))
    payload = graph_signature_payload(task, split, query_index, **components)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_uid(task, split, query_index, signature):
    return f"{task}:{split}:{int(query_index):06d}:{str(signature)[:16]}"
