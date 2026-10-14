"""Versioned local manifests referencing durable Tinker checkpoints.

A manifest also records the learning-rate schedule its weights were trained
under and the optimizer steps that schedule has completed: the optimizer
state lives in the remote checkpoint, and the schedule's progress travels with
it, so a job branched from this checkpoint continues the schedule.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reef.runtime.interfaces import LearningRateScheduleState

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerCheckpoint:
    base_model: str
    lora_rank: int
    state_path: str
    sampler_path: str
    schema_version: int = 1
    #: ``None`` until a job selects a schedule: the configured learning rate applies.
    learning_rate_schedule_state: LearningRateScheduleState | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.base_model or self.lora_rank <= 0:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")

    def validate_model(self, base_model: str, lora_rank: int) -> None:
        if (self.base_model, self.lora_rank) != (base_model, lora_rank):
            raise ValueError("Tinker checkpoint model/rank does not match the runtime")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.learning_rate_schedule_state is None:
            # Manifests stay as they were until a schedule is selected.
            value.pop("learning_rate_schedule_state")
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TinkerCheckpoint:
        if not isinstance(value, Mapping):
            raise ValueError("Tinker checkpoint manifest must be an object")
        fields = dict(value)
        schedule_state = fields.pop("learning_rate_schedule_state", None)
        return cls(
            **fields,
            learning_rate_schedule_state=(
                None if schedule_state is None else LearningRateScheduleState.from_dict(schedule_state)
            ),
        )

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / MANIFEST, self.to_dict())

    @classmethod
    def read(cls, directory: Path) -> TinkerCheckpoint:
        return cls.from_dict(json.loads((directory / MANIFEST).read_text()))


def atomic_json(path: Path, value: Any) -> None:
    """Persist a complete manifest before exposing its directory to Reef."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tinker-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
