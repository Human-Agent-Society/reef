"""The parameters of a Slime job: their shape, their validation and their JSON Schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.storage.job_data import JobDataKind
from reef.train.algos.schedule import StepScheduling
from reef.train.learning_rate import LEARNING_RATE_DECAY_STYLES, LearningRateSchedule
from reef.train.slime_backend.dataset import DatasetSettings
from reef.train.slime_backend.objectives import resolve_slime_objective
from reef.train.task_rollouts import HarborSettings


def harbor_settings(parameters: Mapping[str, Any]) -> HarborSettings:
    """The ``harbor`` parameter of a task job. ``ValueError``: it is missing, malformed or has an unknown key."""
    value = parameters.get("harbor")
    if not isinstance(value, Mapping):
        raise ValueError("a task job needs the harbor parameter as an object")
    unknown = set(value) - set(HarborSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"the harbor parameter takes no {sorted(unknown)}")
    return HarborSettings(**value)


@dataclass(frozen=True)
class SlimeJobParameters:
    """The parameters of one Slime job.

    Values: ``objective`` is a non-empty name or dotted reference that ``resolve_slime_objective`` resolves.
    ``learning_rate_schedule`` is the schedule that the job selects, or None to keep the active one. ``scheduling``
    is the cut of the job's data into optimizer steps, which the objective accepts. ``save_interval`` is a positive
    int: the optimizer steps between two saved checkpoints, in rollouts for a dataset job, or None for one checkpoint
    when the job ends. ``dataset`` is the prompt dataset of a dataset job, or None. ``harbor`` says how to run the
    Harbor rollouts of a task job's tasks, or is None. It reads nothing of ``objective``, which says only how the
    samples train. Purpose: ``from_parameters`` makes it from a job's parameters; ``SlimeJobs.check_job`` and
    ``SlimeJobRunner.run`` read it.
    """

    objective: str
    learning_rate_schedule: LearningRateSchedule | None
    scheduling: StepScheduling
    save_interval: int | None
    dataset: DatasetSettings | None
    harbor: HarborSettings | None

    def __post_init__(self) -> None:
        interval = self.save_interval
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0):
            raise ValueError("save_interval must be a positive int")
        resolve_slime_objective(self.objective).validate_scheduling(self.scheduling)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> SlimeJobParameters:
        """``parameters`` as this value. ``ValueError``: a key is unknown, or a value is missing or malformed."""
        unknown = set(parameters) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Slime jobs accept no parameters {sorted(unknown)}")
        objective = parameters.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("a Slime job needs the objective parameter: a name or a 'package.module:Objective' path")
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
            parameters.get("save_interval"),
            DatasetSettings.from_parameters(parameters) if "dataset" in parameters else None,
            harbor_settings(parameters) if "harbor" in parameters else None,
        )


LEARNING_RATE_SCHEDULE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "description": "The learning-rate schedule that the job selects; null or absent keeps the active schedule.",
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
}
SCHEDULING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "How the job's data becomes optimizer steps.",
    "additionalProperties": False,
    "properties": {
        "epochs": {"type": "integer", "minimum": 1},
        "shuffle": {"type": "boolean"},
        "remainder": {"enum": ["partial", "drop", "error"]},
        "packing_length": {"type": ["integer", "null"], "minimum": 1},
    },
}
DATASET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "The prompt dataset of a job without samples, which Slime rolls out and rewards.",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string", "minLength": 1, "description": "An absolute path to a JSONL file."},
        "epochs": {"type": "integer", "minimum": 1},
    },
}
HARBOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "How to run the Harbor rollouts of a task job's tasks.",
    "additionalProperties": False,
    "properties": {
        "environment": {"type": "string", "minLength": 1},
        "concurrency": {"type": "integer", "minimum": 1},
        "agent": {"type": ["object", "null"]},
        "agent_host": {"type": ["string", "null"], "minLength": 1},
        "verifier_timeout_reward": {"type": ["number", "null"]},
        "scaffold_tolerance": {"type": "integer", "minimum": 0},
    },
}


def slime_parameters_schema(kind: JobDataKind) -> dict[str, Any]:
    """The JSON Schema of a Slime job's parameters, for a job of ``kind`` samples.

    A job without samples, a dataset job, takes the recipe's ``trajectory`` parameters, so the ``trajectory`` schema
    holds ``dataset``. The ``task`` schema requires ``harbor``.
    """
    properties: dict[str, Any] = {
        "objective": {
            "type": "string",
            "minLength": 1,
            "description": "A registered Slime objective, or a 'package.module:Objective' reference.",
        },
        "learning_rate_schedule": LEARNING_RATE_SCHEDULE_SCHEMA,
        "scheduling": SCHEDULING_SCHEMA,
        "save_interval": {"type": ["integer", "null"], "minimum": 1},
    }
    if kind == "task":
        properties["harbor"] = HARBOR_SCHEMA
    else:
        properties["dataset"] = DATASET_SCHEMA
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["objective", "harbor"] if kind == "task" else ["objective"],
        "properties": properties,
    }
