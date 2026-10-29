"""Versioned local manifests referencing durable Tinker checkpoints, which Tinker training and serving share.

A manifest also records the learning-rate schedule its weights were trained
under and the optimizer steps that schedule has completed, and the objective's
state: the optimizer state lives in the remote checkpoint, and the schedule's
progress and the objective's state travel with it, so a job that starts from
this checkpoint continues both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reef.runtime.durable_files import write_json
from reef.runtime.interfaces import LearningRateScheduleState

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerCheckpoint:
    """One durable Tinker checkpoint of ``base_model``: a LoRA of rank ``lora_rank``, or all its parameters.

    Values: ``base_model`` is non-empty. ``lora_rank`` is a positive int for a LoRA, or ``None`` for a checkpoint of
    all the parameters, which Spindle's full-parameter training saves. ``state_path`` (the weights and optimizer state
    that a later job restores) and ``sampler_path`` (the weights that the service's sampler serves; for a LoRA, the
    adapter that a serving runtime downloads to load into its own engines) are ``tinker://`` paths. ``service`` is the base URL of
    the Tinker-compatible service that holds every path (``TinkerConfig.base_url``), or ``None`` for Thinking
    Machines' hosted Tinker; a path resolves only on that service. The training component makes the values and
    writes them as the manifest ``MANIFEST`` of a checkpoint directory, which is the whole release; the component,
    the Tinker inference runtime, ``TinkerSamplerWeights`` and a serving runtime that resolves the weights itself
    read them.
    ``check_compatible`` is how a reader refuses a checkpoint of another model or service.
    """

    base_model: str
    lora_rank: int | None
    state_path: str
    sampler_path: str
    schema_version: int = 1
    #: ``None`` until a job selects a schedule: the configured learning rate applies.
    learning_rate_schedule_state: LearningRateScheduleState | None = None
    #: The objective's state, a JSON object: after the data of the job that saved this checkpoint when it is
    #: the job's last, else the state that job started with (the objective computes its state for the whole
    #: data); empty for the base.
    algorithm_state: Mapping[str, Any] = field(default_factory=dict)
    #: The objective whose state ``algorithm_state`` is; ``None`` exactly when the state is empty. A job of
    #: another objective starts with an empty state.
    objective: str | None = None
    service: str | None = None

    def __post_init__(self) -> None:
        rank = self.lora_rank
        if self.schema_version != 1 or not self.base_model:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
            raise ValueError("a Tinker checkpoint's lora_rank is a positive int, or null for all the parameters")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")
        if (self.objective is None) != (not self.algorithm_state):
            raise ValueError("a Tinker checkpoint names the objective of a non-empty state, and only of one")

    def check_compatible(self, base_model: str, *, service: str | None) -> None:
        """Raise ``ValueError`` unless the checkpoint is of ``base_model`` and held by ``service``.

        A sampler serves a checkpoint of any rank; a trainer also checks its parameterization (``lora_rank``).
        """
        if self.base_model != base_model:
            raise ValueError(f"the Tinker checkpoint is of base model {self.base_model}, not of {base_model}")
        if self.service != service:
            raise ValueError(
                f"the Tinker checkpoint is held by {self.service or 'hosted Tinker'}, "
                f"not by the configured service {service or 'hosted Tinker'}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.learning_rate_schedule_state is None:
            # Manifests stay as they were until a schedule is selected.
            value.pop("learning_rate_schedule_state")
        if not self.algorithm_state:
            value.pop("algorithm_state")
            value.pop("objective")
        if self.service is None:
            value.pop("service")
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


@dataclass(frozen=True)
class FreshStart:
    """The start of a new model of all the parameters of ``base_model``, with the base model's own weights.

    Values: ``base_model`` is non-empty. The start has no training state: no optimizer state, no learning-rate
    schedule progress, and an empty objective state, as a ``TinkerCheckpoint`` of the untrained base would have; it
    names no sampler, so its rollouts sample the base model. Purpose: the full-parameter training component starts a
    job from it when the job starts from the base; ``TinkerClient.train`` creates the model on the service.
    """

    base_model: str

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError("a fresh start names its base model")

    @property
    def learning_rate_schedule_state(self) -> LearningRateScheduleState | None:
        return None

    @property
    def algorithm_state(self) -> Mapping[str, Any]:
        return {}

    @property
    def objective(self) -> str | None:
        return None

    @property
    def sampler_path(self) -> str | None:
        return None

    def state_for(self, objective: str) -> Mapping[str, Any]:
        return {}
