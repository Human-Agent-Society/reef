"""Slime checkpoint records and the start directories that later jobs continue.

Checkpoint saving and recovery ordering stay with the job runner. This module
owns start.json, its validated values, and the snapshot layout of saved starts.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.core.weight_layout import ADAPTER_DIR
from reef.storage.durable_files import read_json, write_json
from reef.train.learning_rate import LearningRateScheduleState

#: A start directory's description of the weights a later job continues from.
START_STATE = "start.json"


def check_objective_state(objective: str | None, algorithm_state: Mapping[str, Any], owner: str) -> None:
    """Raise unless ``algorithm_state`` is a JSON object and ``objective`` names its objective exactly when it is non-empty."""
    if not isinstance(algorithm_state, Mapping):
        raise TypeError(f"{owner}.algorithm_state must be a mapping")
    try:
        json.dumps(dict(algorithm_state), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}.algorithm_state must encode as JSON: {exc}") from exc
    if objective is not None and (not isinstance(objective, str) or not objective):
        raise ValueError(f"{owner}.objective must be a non-empty string or None")
    if (objective is None) != (not algorithm_state):
        raise ValueError(f"{owner}.objective names the objective of a non-empty state, and only of one")


def check_schedule_state(value: object, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a ``LearningRateScheduleState`` or ``None``."""
    if value is not None and not isinstance(value, LearningRateScheduleState):
        raise TypeError(f"{name} must be a LearningRateScheduleState or None")


@dataclass(frozen=True)
class SlimeStart:
    """What a job continues from: the weights' rollout, the objective's state, and the schedule.

    ``rollout_id`` is the non-negative rollout whose checkpoint holds the
    weights, or ``None`` for the weights the workers loaded when they first
    started, before they saved any rollout. ``objective`` names the objective
    whose state ``algorithm_state`` (a mapping that encodes as a JSON object)
    is; it is ``None`` exactly when the state is empty. A job continues that
    state only when it trains the same objective; another objective starts
    with an empty state. ``schedule_state`` is the learning-rate schedule's
    progress, or ``None`` for Slime's configured scheduler. A start
    directory keeps the value in ``start.json``; ``record_rollout`` writes
    it, and ``check_job`` and ``run`` read it.
    """

    rollout_id: int | None
    objective: str | None
    algorithm_state: Mapping[str, Any]
    schedule_state: LearningRateScheduleState | None

    def __post_init__(self) -> None:
        if self.rollout_id is not None and (
            isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) or self.rollout_id < 0
        ):
            raise ValueError("SlimeStart.rollout_id must be a non-negative int or None")
        check_objective_state(self.objective, self.algorithm_state, "SlimeStart")
        check_schedule_state(self.schedule_state, "SlimeStart.schedule_state")

    def write(self, directory: Path) -> None:
        write_json(
            directory / START_STATE,
            {
                "rollout_id": self.rollout_id,
                "objective": self.objective,
                "algorithm_state": dict(self.algorithm_state),
                "schedule_state": None if self.schedule_state is None else self.schedule_state.to_dict(),
            },
        )

    @classmethod
    def read(cls, directory: Path) -> SlimeStart:
        """Read the start in ``directory``; raise ``ValueError`` for a directory that holds none."""
        value = read_json(directory / START_STATE)
        if value is None:
            raise ValueError(f"{directory} is not a Slime start: it has no {START_STATE}")
        schedule = value["schedule_state"]
        return cls(
            value["rollout_id"],
            value["objective"],
            value["algorithm_state"],
            None if schedule is None else LearningRateScheduleState.from_dict(schedule),
        )

    def state_for(self, objective: str) -> Mapping[str, Any]:
        """The objective state a job of ``objective`` continues: this start's state for the same objective, else empty."""
        return self.algorithm_state if self.objective == objective else {}


@dataclass(frozen=True)
class SlimeRound:
    """One rollout of a job: the run of optimizer steps that the workers train and save as one checkpoint.

    ``rollout_id`` is the non-negative id that names the rollout's Megatron
    iteration and Hugging Face export. ``step`` is a positive job progress count: cumulative optimizer steps for
    external-data jobs, or cumulative rollouts for native dataset jobs. ``schedule_state`` is the learning-rate
    schedule's progress before the rollout and ``next_schedule`` after it:
    both are ``None`` (Slime's configured scheduler), or both are set.
    ``objective`` and ``algorithm_state`` are the
    objective state that the rollout's start records, with the same
    condition as in ``SlimeStart``: the job's next state on its last rollout,
    and the state of the job's start on an earlier one. ``run`` makes the
    value; ``train_and_save`` and ``record_rollout`` read it.
    """

    rollout_id: int
    step: int
    schedule_state: LearningRateScheduleState | None
    next_schedule: LearningRateScheduleState | None
    objective: str | None
    algorithm_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) or self.rollout_id < 0:
            raise ValueError("SlimeRound.rollout_id must be a non-negative int")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ValueError("SlimeRound.step must be a positive int")
        check_schedule_state(self.schedule_state, "SlimeRound.schedule_state")
        check_schedule_state(self.next_schedule, "SlimeRound.next_schedule")
        if (self.schedule_state is None) != (self.next_schedule is None):
            raise ValueError("SlimeRound has a schedule before and after the rollout, or neither")
        check_objective_state(self.objective, self.algorithm_state, "SlimeRound")


def write_rollout_start(
    starts_root: Path,
    hf_directory: Path,
    rollout: SlimeRound,
    schedule_state: LearningRateScheduleState | None,
    *,
    lora: bool,
) -> tuple[Path, Path]:
    """Write the saved rollout's start and return (start directory, release directory).

    The runner supplies a validated, non-symlink Hugging Face directory.
    Full-weight artifacts use that directory directly. LoRA artifacts copy its
    files into the start's release/adapter directory. The start records the
    supplied schedule and the rollout's objective state before returning.
    """
    start_dir = starts_root / f"rollout_{rollout.rollout_id}"
    artifact_dir = hf_directory
    if lora:
        artifact_dir = start_dir / "release"
        shutil.rmtree(artifact_dir, ignore_errors=True)
        shutil.copytree(hf_directory, artifact_dir / ADAPTER_DIR)
    SlimeStart(
        rollout.rollout_id,
        rollout.objective,
        rollout.algorithm_state,
        schedule_state,
    ).write(start_dir)
    return start_dir, artifact_dir
