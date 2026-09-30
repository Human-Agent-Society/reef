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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.core.training_method import LearningRateScheduleState

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerCheckpoint:
    base_model: str
    lora_rank: int
    state_path: str
    sampler_path: str
    schema_version: int = 1
    #: ``None`` until a job selects a schedule: the configured learning rate applies.
    learning_rate_schedule: LearningRateScheduleState | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.base_model or self.lora_rank <= 0:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")
        if self.learning_rate_schedule is not None and not isinstance(
            self.learning_rate_schedule, LearningRateScheduleState
        ):
            raise TypeError("Tinker checkpoint learning_rate_schedule must be a LearningRateScheduleState")

    def validate_model(self, base_model: str, lora_rank: int) -> None:
        if (self.base_model, self.lora_rank) != (base_model, lora_rank):
            raise ValueError("Tinker checkpoint model/rank does not match the runtime")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "state_path": self.state_path,
            "sampler_path": self.sampler_path,
            "schema_version": self.schema_version,
        }
        if self.learning_rate_schedule is not None:
            value["learning_rate_schedule"] = self.learning_rate_schedule.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TinkerCheckpoint:
        if not isinstance(value, Mapping):
            raise ValueError("Tinker checkpoint manifest must be an object")
        fields = dict(value)
        schedule = fields.pop("learning_rate_schedule", None)
        return cls(
            **fields,
            learning_rate_schedule=None if schedule is None else LearningRateScheduleState.from_dict(schedule),
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
