"""Versioned local manifests referencing durable Tinker checkpoints.

A manifest also records the learning-rate schedule its weights were trained
under and the optimizer steps that schedule has completed, and the objective's
state after the job that saved it: the optimizer state lives in the remote
checkpoint, and the schedule's progress and the objective's state travel with
it, so a job that starts from this checkpoint continues both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reef.runtime.interfaces import LearningRateScheduleState
from reef.runtime.recovery import write_json

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
    #: The objective's state after the job that saved this checkpoint, a JSON object; empty for the base.
    algorithm_state: Mapping[str, Any] = field(default_factory=dict)
    #: The objective whose state ``algorithm_state`` is; ``None`` exactly when the state is empty. A job of
    #: another objective starts with an empty state.
    objective: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.base_model or self.lora_rank <= 0:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")
        if (self.objective is None) != (not self.algorithm_state):
            raise ValueError("a Tinker checkpoint names the objective of a non-empty state, and only of one")

    def validate_model(self, base_model: str, lora_rank: int) -> None:
        if (self.base_model, self.lora_rank) != (base_model, lora_rank):
            raise ValueError("Tinker checkpoint model/rank does not match the runtime")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.learning_rate_schedule_state is None:
            # Manifests stay as they were until a schedule is selected.
            value.pop("learning_rate_schedule_state")
        if not self.algorithm_state:
            value.pop("algorithm_state")
            value.pop("objective")
        return value

    def state_for(self, objective: str) -> Mapping[str, Any]:
        """The objective state a job of ``objective`` continues: this checkpoint's state for the same objective, else empty."""
        return self.algorithm_state if self.objective == objective else {}

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
        write_json(directory / MANIFEST, self.to_dict())

    @classmethod
    def read(cls, directory: Path) -> TinkerCheckpoint:
        return cls.from_dict(json.loads((directory / MANIFEST).read_text()))
