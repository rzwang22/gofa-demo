"""Lightweight GOFA demo runner.

The default replay path is intentionally pure Python and does not import GOFA
model code, allocate GPU memory, or require a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEMO_DIR = Path(__file__).resolve().parent
CACHED_OUTPUTS_DIR = DEMO_DIR / "cached_outputs"
DEFAULT_GRAPH_PATH = CACHED_OUTPUTS_DIR / "sample_graph.json"
DEFAULT_RESULT_PATH = CACHED_OUTPUTS_DIR / "sample_result.json"
PIPELINE_STAGES = [
    "load graph",
    "validate graph",
    "convert input",
    "load model",
    "generate",
    "decode output",
]


class DemoError(RuntimeError):
    """Base class for expected demo failures."""


class LiveGOFANotAvailable(DemoError):
    """Raised when Live GOFA mode cannot be started safely."""


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    duration_ms: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_graph(graph_path: str | Path = DEFAULT_GRAPH_PATH) -> dict[str, Any]:
    """Load the demo graph fixture."""

    return _load_json(Path(graph_path))


def load_sample_result(result_path: str | Path = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    """Load cached generation output for replay mode."""

    return _load_json(Path(result_path))


def validate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Validate the tiny graph schema used by the demo."""

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise DemoError("sample_graph.json must contain a non-empty 'nodes' list.")
    if not isinstance(edges, list):
        raise DemoError("sample_graph.json must contain an 'edges' list.")

    node_ids: set[str] = set()
    question_nodes: list[str] = []
    complete_nodes: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise DemoError("Each graph node must be an object.")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise DemoError("Each graph node must include a non-empty string 'id'.")
        if node_id in node_ids:
            raise DemoError(f"Duplicate graph node id: {node_id}")
        node_ids.add(node_id)

        role = str(node.get("role") or node.get("type") or "").lower()
        if role == "question":
            question_nodes.append(node_id)
        if role == "complete":
            complete_nodes.append(node_id)

    for edge in edges:
        if not isinstance(edge, dict):
            raise DemoError("Each graph edge must be an object.")
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids or target not in node_ids:
            raise DemoError(f"Edge references missing node: {source!r} -> {target!r}")

    if not question_nodes:
        raise DemoError("Graph must contain a question node.")
    if not complete_nodes:
        raise DemoError("Graph must contain a complete node.")

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "question_nodes": question_nodes,
        "complete_nodes": complete_nodes,
    }


def convert_input(graph: dict[str, Any], task_type: str) -> dict[str, Any]:
    """Convert the graph into a compact model-input preview for the UI."""

    node_lines = []
    for node in graph["nodes"]:
        label = node.get("label", node["id"])
        text = node.get("text", "")
        node_lines.append(f"[{node['id']}] {label}: {text}")

    edge_lines = []
    for edge in graph.get("edges", []):
        label = edge.get("label", "relates_to")
        edge_lines.append(f"{edge['source']} -{label}-> {edge['target']}")

    return {
        "task_type": task_type,
        "prompt_preview": "\n".join(
            [
                f"Task: {task_type}",
                "Nodes:",
                *node_lines,
                "Edges:",
                *edge_lines,
            ]
        ),
    }


def _record_stage(name: str, detail: str, started_at: float) -> StageResult:
    return StageResult(
        name=name,
        status="ok",
        duration_ms=(time.perf_counter() - started_at) * 1000,
        detail=detail,
    )


def run_replay(
    task_type: str = "QA",
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    result_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, Any]:
    """Run the checkpoint-free replay path."""

    normalized_task_type = task_type if task_type in {"QA", "Completion"} else "QA"
    stages: list[StageResult] = []

    started = time.perf_counter()
    graph = load_sample_graph(graph_path)
    stages.append(_record_stage("load graph", str(Path(graph_path)), started))

    started = time.perf_counter()
    graph_stats = validate_graph(graph)
    stages.append(
        _record_stage(
            "validate graph",
            f"{graph_stats['node_count']} nodes, {graph_stats['edge_count']} edges",
            started,
        )
    )

    started = time.perf_counter()
    converted = convert_input(graph, normalized_task_type)
    stages.append(_record_stage("convert input", "graph converted to prompt preview", started))

    started = time.perf_counter()
    cached_result = load_sample_result(result_path)
    stages.append(_record_stage("load model", "replay stub, checkpoint_loaded=false", started))

    started = time.perf_counter()
    task_output = cached_result["outputs"][normalized_task_type]
    stages.append(_record_stage("generate", "cached output selected", started))

    started = time.perf_counter()
    decoded_output = {
        "text": task_output["generated_text"],
        "question_node_id": task_output["question_node_id"],
        "complete_node_id": task_output["complete_node_id"],
    }
    stages.append(_record_stage("decode output", "decoded cached generation", started))

    return {
        "mode": "replay",
        "task_type": normalized_task_type,
        "graph": graph,
        "graph_stats": graph_stats,
        "converted_input": converted,
        "output": decoded_output,
        "pipeline": [stage.as_dict() for stage in stages],
        "runtime": runtime_info(mode="replay"),
    }


def runtime_info(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "gpu_required": False,
        "checkpoint_loaded": False,
        "checkpoint_downloaded": False,
    }


def _repo_root() -> Path:
    return DEMO_DIR.parent


def _find_chat_gofa(repo_root: Path) -> Path | None:
    direct_candidates = [
        repo_root / "chat_gofa.py",
        repo_root / "scripts" / "chat_gofa.py",
        repo_root / "examples" / "chat_gofa.py",
        repo_root / "GOFA" / "chat_gofa.py",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    for candidate in repo_root.rglob("chat_gofa.py"):
        if ".git" not in candidate.parts:
            return candidate
    return None


def run_live(task_type: str = "QA", timeout_s: int = 20) -> dict[str, Any]:
    """Probe a Live GOFA entrypoint without downloading checkpoints.

    Set GOFA_DEMO_LIVE_COMMAND to a full command when the real environment is
    ready. Without that env var, this function only probes chat_gofa.py with
    ``--help`` so missing dependencies are surfaced cleanly in the UI.
    """

    repo_root = _repo_root()
    command_from_env = os.environ.get("GOFA_DEMO_LIVE_COMMAND")
    if command_from_env:
        command = command_from_env
        shell = True
    else:
        chat_gofa = _find_chat_gofa(repo_root)
        if chat_gofa is None:
            raise LiveGOFANotAvailable(
                "Live GOFA mode is unavailable: could not find chat_gofa.py in this repository."
            )
        command = [sys.executable, str(chat_gofa), "--help"]
        shell = False

    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            shell=shell,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise LiveGOFANotAvailable(f"Live GOFA command was not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LiveGOFANotAvailable(
            f"Live GOFA command timed out after {timeout_s}s. No checkpoint was downloaded."
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if not detail:
            detail = f"process exited with code {completed.returncode}"
        raise LiveGOFANotAvailable(f"Live GOFA probe failed gracefully: {detail}")

    return {
        "mode": "live",
        "task_type": task_type if task_type in {"QA", "Completion"} else "QA",
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "runtime": runtime_info(mode="live-probe"),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight GOFA demo runner.")
    parser.add_argument("--task-type", choices=["QA", "Completion"], default="QA")
    parser.add_argument("--live", action="store_true", help="Probe Live GOFA mode instead of replay.")
    args = parser.parse_args()

    try:
        result = run_live(args.task_type) if args.live else run_replay(args.task_type)
    except DemoError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()

