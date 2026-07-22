"""Deterministic sampling and checkpoint metadata for resumable LLM-N SFT."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Sampler


CHECKPOINT_VERSION = 1
CHECKPOINT_MARKER = "_SUCCESS"
CHECKPOINT_STATE_FILE = "training_state.pt"
CHECKPOINT_CONFIG_FILE = "training_config.json"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


class ResumeConfigMismatchError(ValueError):
    """Raised when a resumed run changes training-critical configuration."""


class IncompleteCheckpointError(RuntimeError):
    """Raised when a requested checkpoint was not committed atomically."""


class ResumableRandomSampler(Sampler[int]):
    """Random sampler whose permutation is a pure function of seed and epoch.

    ``__iter__`` never mutates the cursor.  The training loop advances it only
    after a batch was successfully processed.  This avoids DataLoader prefetch
    from moving persisted state past the last completed optimizer boundary.
    """

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        seed: int,
        epoch: int = 0,
        cursor: int = 0,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.cursor = int(cursor)
        self._validate_position()

    @property
    def dataset_size(self) -> int:
        return len(self.data_source)

    def _validate_position(self) -> None:
        if self.epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        if self.cursor < 0 or self.cursor > self.dataset_size:
            raise ValueError(
                f"sampler cursor {self.cursor} is outside [0, {self.dataset_size}]"
            )

    def permutation(self, epoch: int | None = None) -> list[int]:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        generator = torch.Generator()
        generator.manual_seed(self.seed + selected_epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation()[self.cursor :])

    def __len__(self) -> int:
        return self.dataset_size - self.cursor

    def advance(self, sample_count: int) -> None:
        sample_count = int(sample_count)
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        self.cursor += sample_count
        self._validate_position()

    def start_next_epoch(self) -> None:
        if self.cursor != self.dataset_size:
            raise RuntimeError(
                f"cannot finish epoch {self.epoch} at cursor {self.cursor}/{self.dataset_size}"
            )
        self.epoch += 1
        self.cursor = 0

    def next_position(self) -> tuple[int, int]:
        if self.cursor == self.dataset_size:
            return self.epoch + 1, 0
        return self.epoch, self.cursor

    def state_dict(self, *, normalize_completed_epoch: bool = False) -> dict[str, int]:
        epoch, cursor = (
            self.next_position()
            if normalize_completed_epoch
            else (self.epoch, self.cursor)
        )
        return {
            "seed": self.seed,
            "epoch": epoch,
            "cursor": cursor,
            "dataset_size": self.dataset_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_size = int(state["dataset_size"])
        expected_seed = int(state["seed"])
        if expected_size != self.dataset_size:
            raise ValueError(
                f"sampler dataset_size mismatch: checkpoint={expected_size}, "
                f"current={self.dataset_size}"
            )
        if expected_seed != self.seed:
            raise ValueError(
                f"sampler seed mismatch: checkpoint={expected_seed}, current={self.seed}"
            )
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._validate_position()


def checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def is_complete_checkpoint(path: Path) -> bool:
    return bool(
        path.is_dir()
        and checkpoint_step(path) is not None
        and (path / CHECKPOINT_MARKER).is_file()
        and (path / CHECKPOINT_STATE_FILE).is_file()
        and (path / CHECKPOINT_CONFIG_FILE).is_file()
    )


def complete_checkpoints(run_root: Path) -> list[Path]:
    trainer_dir = run_root / "trainer"
    if not trainer_dir.is_dir():
        return []
    checkpoints = [path for path in trainer_dir.iterdir() if is_complete_checkpoint(path)]
    return sorted(checkpoints, key=lambda path: checkpoint_step(path) or -1)


def _candidate_run_roots(config: Mapping[str, Any]) -> list[Path]:
    output_dir = Path(str(config.get("output_dir") or config.get("exp_dir") or "outputs/llm_n"))
    run_name = config.get("run_name")
    if run_name:
        return [output_dir / str(run_name)]
    if (output_dir / "trainer").is_dir():
        return [output_dir]
    if not output_dir.is_dir():
        return [output_dir]
    return sorted(
        [path for path in output_dir.iterdir() if path.is_dir() and (path / "trainer").is_dir()]
    )


def resolve_resume_checkpoint(
    value: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Resolve an explicit checkpoint or the greatest complete checkpoint."""
    raw_value = str(value)
    if raw_value != "latest":
        checkpoint = Path(raw_value).expanduser().resolve()
        if not is_complete_checkpoint(checkpoint):
            raise IncompleteCheckpointError(
                f"Resume checkpoint is missing {CHECKPOINT_MARKER}, {CHECKPOINT_STATE_FILE}, "
                f"or {CHECKPOINT_CONFIG_FILE}: {checkpoint}"
            )
        return checkpoint

    candidates: list[Path] = []
    for run_root in _candidate_run_roots(config):
        candidates.extend(complete_checkpoints(run_root))
    if not candidates:
        searched = ", ".join(str(path) for path in _candidate_run_roots(config))
        raise FileNotFoundError(
            f"No complete trainer/checkpoint-* directories found for latest under: {searched}"
        )
    candidates.sort(
        key=lambda path: (
            checkpoint_step(path) or -1,
            path.stat().st_mtime_ns,
        )
    )
    return candidates[-1].resolve()


def run_root_from_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "trainer":
        raise ValueError(
            f"Checkpoint must have layout <run>/trainer/checkpoint-<step>: {checkpoint}"
        )
    return checkpoint.parent.parent


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_config(
    checkpoint_record: Mapping[str, Any],
    current_critical_config: Mapping[str, Any],
) -> None:
    saved_config = checkpoint_record.get("critical_config")
    saved_fingerprint = checkpoint_record.get("fingerprint")
    if not isinstance(saved_config, Mapping) or not isinstance(saved_fingerprint, str):
        raise IncompleteCheckpointError(
            "Checkpoint training_config.json has no critical_config/fingerprint"
        )
    actual_saved_fingerprint = config_fingerprint(saved_config)
    if saved_fingerprint != actual_saved_fingerprint:
        raise IncompleteCheckpointError(
            "Checkpoint critical configuration fingerprint is corrupt or was edited"
        )

    conflicts: list[str] = []
    all_keys = sorted(set(saved_config) | set(current_critical_config))
    for key in all_keys:
        saved_value = saved_config.get(key, "<missing>")
        current_value = current_critical_config.get(key, "<missing>")
        if saved_value != current_value:
            conflicts.append(
                f"- {key}: checkpoint={saved_value!r}, current={current_value!r}"
            )
    if conflicts:
        raise ResumeConfigMismatchError(
            "Resume checkpoint configuration mismatch:\n" + "\n".join(conflicts)
        )


def read_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    with (checkpoint / CHECKPOINT_CONFIG_FILE).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise IncompleteCheckpointError("training_config.json must contain a JSON object")
    return value


def load_training_state(checkpoint: Path) -> dict[str, Any]:
    try:
        value = torch.load(
            checkpoint / CHECKPOINT_STATE_FILE,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        value = torch.load(checkpoint / CHECKPOINT_STATE_FILE, map_location="cpu")
    if not isinstance(value, dict):
        raise IncompleteCheckpointError("training_state.pt must contain a state dictionary")
    return value
