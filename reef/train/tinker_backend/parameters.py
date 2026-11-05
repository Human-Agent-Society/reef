"""The parameters of a Tinker job: their shape, their validation and their JSON Schema."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.storage.job_data import JobDataKind
from reef.train.algos.schedule import StepScheduling
from reef.train.learning_rate import LEARNING_RATE_DECAY_STYLES, LearningRateSchedule
from reef.train.task_rollouts import RolloutSettings
from reef.train.tinker_backend.config import TinkerServiceConfig
from reef.train.tinker_backend.objectives import resolve_tinker_objective


def rollout_settings(parameters: Mapping[str, Any]) -> RolloutSettings:
    """The ``rollout`` parameter of a task job. ``ValueError``: it is missing, malformed or has an unknown key."""
    value = parameters.get("rollout")
    if not isinstance(value, Mapping):
        raise ValueError("a task job needs the rollout parameter as an object")
    unknown = set(value) - set(RolloutSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"the rollout parameter takes no {sorted(unknown)}")
    try:
        return RolloutSettings(**value)
    except TypeError as exc:
        raise ValueError(f"the rollout parameter is incomplete: {exc}") from exc


@dataclass(frozen=True)
class TinkerJobParameters:
    """The parameters of one Tinker job, with the component's defaults for those the job leaves out.

    Values: ``objective`` is a non-empty name or dotted reference that ``resolve_tinker_objective`` resolves.
    ``learning_rate_schedule`` is the schedule that the job selects, or None to keep the active one. ``scheduling``
    is the cut of the job's data into optimizer steps, which the objective accepts. ``batch_size`` is a positive int,
    the rollouts of one optimizer step; ``learning_rate`` is finite and positive; ``kl_coef`` is finite and
    non-negative; ``save_interval`` is a positive int, the optimizer steps between two saved checkpoints, or None for
    one checkpoint when the job ends; ``rollout`` is the rollouts of a task job, or None for a job of trajectories.
    Purpose: ``from_parameters`` makes it when a Tinker component checks and when it runs a job; the run reads it.
    """

    objective: str
    learning_rate_schedule: LearningRateSchedule | None
    scheduling: StepScheduling
    batch_size: int
    learning_rate: float
    kl_coef: float
    save_interval: int | None
    rollout: RolloutSettings | None

    def __post_init__(self) -> None:
        counts = {
            "batch_size": self.batch_size,
            "save_interval": 1 if self.save_interval is None else self.save_interval,
        }
        for name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} must be a positive int")
        for name, rate in (("learning_rate", self.learning_rate), ("kl_coef", self.kl_coef)):
            if isinstance(rate, bool) or not isinstance(rate, int | float) or not math.isfinite(rate):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.kl_coef < 0:
            raise ValueError("learning_rate must be positive and kl_coef non-negative")
        resolve_tinker_objective(self.objective).validate_scheduling(self.scheduling)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any], config: TinkerServiceConfig) -> TinkerJobParameters:
        """``parameters`` over the defaults of ``config``. ``ValueError``: a key is unknown, or a value is malformed."""
        unknown = set(parameters) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Tinker jobs accept no parameters {sorted(unknown)}")
        objective = parameters.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("a Tinker job needs the objective parameter: a name or a 'package.module:Objective' path")
        schedule = parameters.get("learning_rate_schedule")
        scheduling = parameters.get("scheduling", {})
        if not isinstance(scheduling, Mapping) or set(scheduling) - set(StepScheduling.__dataclass_fields__):
            raise ValueError(
                "the scheduling parameter must be an object of epochs, shuffle, remainder and packing_length"
            )
        return cls(
            objective,
            None if schedule is None else LearningRateSchedule.from_dict(schedule),
            StepScheduling(**scheduling),
            parameters.get("batch_size", config.batch_size),
            parameters.get("learning_rate", config.learning_rate),
            parameters.get("kl_coef", config.kl_coef),
            parameters.get("save_interval"),
            None if parameters.get("rollout") is None else rollout_settings(parameters),
        )


def tinker_parameters_schema(kind: JobDataKind) -> dict[str, Any]:
    """The JSON Schema of a Tinker job's parameters, for a job of ``kind`` samples: a task job requires ``rollout``."""
    rollout: dict[str, Any] = {
        "type": "object",
        "description": "Where a task job's samples come from: the Harbor rollouts of its tasks.",
        "additionalProperties": False,
        "required": ["rollouts_per_task"],
        "properties": {
            "rollouts_per_task": {"type": "integer", "minimum": 2},
            "environment": {"type": "string", "minLength": 1},
            "concurrency": {"type": "integer", "minimum": 1},
            "agent": {"type": ["object", "null"]},
            "agent_host": {"type": ["string", "null"], "minLength": 1},
            "verifier_timeout_reward": {"type": ["number", "null"]},
            "scaffold_tolerance": {"type": "integer", "minimum": 0},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["objective", "rollout"] if kind == "task" else ["objective"],
        "properties": {
            "objective": {
                "type": "string",
                "minLength": 1,
                "description": "A registered Tinker objective, or a 'package.module:Objective' reference.",
            },
            "learning_rate_schedule": {
                "type": ["object", "null"],
                "description": "The schedule that the job selects; null or absent keeps the active schedule.",
                "additionalProperties": False,
                "required": ["name", "peak_learning_rate"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "peak_learning_rate": {"type": "number", "exclusiveMinimum": 0},
                    "warmup_steps": {"type": "integer", "minimum": 0},
                    "decay_style": {"enum": list(LEARNING_RATE_DECAY_STYLES)},
                    "decay_steps": {"type": "integer", "minimum": 0},
                    "min_learning_rate": {"type": "number", "minimum": 0},
                    "initial_learning_rate": {"type": "number", "minimum": 0},
                },
            },
            "scheduling": {
                "type": "object",
                "description": "How the job's data becomes optimizer steps.",
                "additionalProperties": False,
                "properties": {
                    "epochs": {"type": "integer", "minimum": 1},
                    "shuffle": {"type": "boolean"},
                    "remainder": {"enum": ["partial", "drop", "error"]},
                    "packing_length": {"type": ["integer", "null"], "minimum": 1},
                },
            },
            "batch_size": {"type": "integer", "minimum": 1, "description": "The rollouts of one optimizer step."},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0},
            "kl_coef": {"type": "number", "minimum": 0},
            "save_interval": {"type": ["integer", "null"], "minimum": 1},
            "rollout": rollout if kind == "task" else {"type": "null"},
        },
    }
