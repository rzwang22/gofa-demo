import json
import shutil
from pathlib import Path


def _load_json(path):
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def validate_resume_trace_directory(trace_dir, task, samples_per_split):
    trace_dir = Path(trace_dir)
    index_path = trace_dir / "trace_index.jsonl"
    query_paths = sorted(trace_dir.glob("query_*.json"))
    if not index_path.is_file():
        raise RuntimeError(f"Formal trace resume requires an existing trace index: {index_path}")
    entries = []
    with index_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid trace index JSON at {index_path}:{line_number}: {exc}") from exc
            if not isinstance(entry, dict):
                raise RuntimeError(f"Trace index entry {line_number} is not an object: {index_path}")
            entries.append(entry)
    if not entries:
        raise RuntimeError(f"Formal trace resume index is empty: {index_path}")

    expected_names = [f"query_{index:06d}.json" for index in range(len(entries))]
    actual_names = [path.name for path in query_paths]
    if actual_names != expected_names:
        raise RuntimeError(
            "Formal trace resume requires a complete continuous query file sequence: "
            f"expected={expected_names}, actual={actual_names}"
        )
    maximum = 2 * int(samples_per_split)
    if len(entries) > maximum:
        raise RuntimeError(f"Formal trace index has {len(entries)} entries, expected at most {maximum}")

    for trace_order, (entry, filename) in enumerate(zip(entries, expected_names)):
        expected_split = "val" if trace_order < int(samples_per_split) else "test"
        expected_query_index = trace_order if expected_split == "val" else trace_order - int(samples_per_split)
        expected_query_id = f"query_{trace_order:06d}"
        trace_path = trace_dir / filename
        trace = _load_json(trace_path)
        checks = {
            "query_id": (entry.get("query_id"), expected_query_id),
            "trace_path": (entry.get("trace_path"), filename),
            "index task": (entry.get("task"), task),
            "trace task": (trace.get("task_name"), task),
            "index split": (entry.get("split"), expected_split),
            "trace split": (trace.get("split"), expected_split),
            "index query_index": (entry.get("query_index"), expected_query_index),
            "trace query_index": (trace.get("runtime_query_index"), expected_query_index),
            "trace query_id": (trace.get("query_id"), expected_query_id),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                raise RuntimeError(
                    f"Formal trace resume {field} mismatch at trace_order={trace_order}: "
                    f"actual={actual!r}, expected={expected!r}"
                )
        index_signature = entry.get("graph_signature")
        trace_signature = trace.get("graph_signature")
        if not index_signature or index_signature != trace_signature:
            raise RuntimeError(
                f"Formal trace resume graph_signature mismatch at trace_order={trace_order}: "
                f"index={index_signature!r}, trace={trace_signature!r}"
            )
    return entries


def prepare_formal_trace_outputs(manifest, fresh=False, resume=False, execute=False):
    if fresh and resume:
        raise ValueError("--fresh and --resume are mutually exclusive")
    states = {}
    samples = int(manifest["profile"]["samples_per_split"])
    for task in manifest["tasks"]:
        trace_dir = Path(manifest["paths"]["traces"]) / f"{task}_formal_v1"
        directory_is_empty = not trace_dir.exists() or not any(trace_dir.iterdir())
        index_exists = (trace_dir / "trace_index.jsonl").exists()
        query_exists = any(trace_dir.glob("query_*.json")) if trace_dir.exists() else False
        has_outputs = index_exists or query_exists
        if fresh:
            states[task] = {"action": "fresh", "existing_entries": 0, "trace_dir": str(trace_dir)}
            if execute and trace_dir.exists():
                shutil.rmtree(trace_dir)
            if execute:
                trace_dir.mkdir(parents=True, exist_ok=True)
        elif resume:
            if directory_is_empty:
                states[task] = {
                    "action": "new",
                    "existing_entries": 0,
                    "trace_dir": str(trace_dir),
                }
            else:
                entries = validate_resume_trace_directory(trace_dir, task, samples)
                states[task] = {
                    "action": "resume",
                    "existing_entries": len(entries),
                    "trace_dir": str(trace_dir),
                }
        else:
            if has_outputs:
                raise RuntimeError(
                    f"Formal trace output already exists for task={task}: {trace_dir}. "
                    "Use --fresh to rebuild or --resume to validate and continue."
                )
            states[task] = {"action": "new", "existing_entries": 0, "trace_dir": str(trace_dir)}
    return states
