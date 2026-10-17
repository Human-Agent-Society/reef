"""The task generation contract: the request, the generated tasks, and their play results.

A deployment selects one :class:`TaskGenerator` in its
``pipeline.record2task.config.generator`` setting. A recipe that trains on
generated tasks asks the generator for tasks, validates each one, plays the
valid ones, and sends the results back to the generator. This module holds the values, their field validation, and the
generator interface. Consumers own file reads, transport limits, and execution.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from reef.core.plugins import Plugin
from reef.core.records_types import AgentRecord
from reef.core.tasks.harbor import HarborTask

ARM_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class TaskGenerationRequest:
    """Source records, requirements, and materials for one call to a task generator.

    Assets name files or directories on the caller's machine. Alternatively,
    asset_files maps relative file names to UTF-8 contents. Supply paths or
    contents, not both. Construction validates fields but performs no I/O.
    Consumers own file reads, transport limits, and generation settings.
    Description-only generation uses an empty source_records tuple.
    ``generation`` numbers the call; ``record_results`` names it again.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()
    asset_files: Mapping[str, str] = field(default_factory=dict)
    generation: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(self.source_records, tuple):
            raise ValueError("source_records must be a tuple of AgentRecord values")
        if any(not isinstance(record, AgentRecord) for record in self.source_records):
            raise TypeError("source_records must contain AgentRecord values")
        if len({record.scenario for record in self.source_records}) > 1:
            raise ValueError("source_records must belong to one scenario")
        record_ids = [record.agent_record_id for record in self.source_records]
        if any(not isinstance(record_id, str) or not record_id for record_id in record_ids) or len(
            set(record_ids)
        ) != len(record_ids):
            raise ValueError("source_records must have distinct non-empty record ids")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(self.assets, tuple) or any(not isinstance(path, Path) for path in self.assets):
            raise TypeError("assets must be a tuple of pathlib.Path values")
        if not isinstance(self.asset_files, Mapping) or any(
            not isinstance(name, str) or not isinstance(text, str) for name, text in self.asset_files.items()
        ):
            raise ValueError("asset_files must map relative file names to UTF-8 text")
        if self.assets and self.asset_files:
            raise ValueError("supply assets or asset_files, not both")
        for name, text in self.asset_files.items():
            path = PurePosixPath(name)
            if not name or name == "." or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
                raise ValueError("asset_files must use relative file names without parent traversal")
            if "\x00" in text:
                raise ValueError("asset_files must contain UTF-8 text without NUL bytes")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("asset_files must contain UTF-8 text") from exc


@dataclass(frozen=True)
class TaskValidationResult:
    """A completed validation: no errors means the task passed all required checks.

    Errors describe task defects. Failures to execute the checks (for example,
    an unavailable container runtime) raise exceptions instead of rejecting the
    task as invalid.
    """

    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple of non-empty strings")
        if any(not isinstance(error, str) or not error.strip() for error in self.errors):
            raise ValueError("errors must contain non-empty strings")

    @property
    def is_valid(self) -> bool:
        """Whether every required check passed."""
        return not self.errors


@dataclass(frozen=True)
class PlayArm:
    """Plays of a task that only the generator reads. The recipe never trains on them.

    Each play shows the agent the task's instruction and then the files that
    ``extra_instruction_files`` names, relative to the task directory (for
    example ``solution/hint.txt``).
    """

    name: str
    count: int
    extra_instruction_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not ARM_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"an arm name must match {ARM_NAME_PATTERN.pattern}")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("an arm's count must be a positive integer")
        if not isinstance(self.extra_instruction_files, tuple):
            raise TypeError("extra_instruction_files must be a tuple of relative file names")
        for name in self.extra_instruction_files:
            path = PurePosixPath(name) if isinstance(name, str) else None
            if path is None or not name or path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("extra_instruction_files must be relative file names inside the task")


@dataclass(frozen=True)
class GeneratedTask:
    """One task from a generator, with the measurement arms the generator wants played on it."""

    task: HarborTask
    measurement_arms: tuple[PlayArm, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        if not isinstance(self.measurement_arms, tuple) or any(
            not isinstance(arm, PlayArm) for arm in self.measurement_arms
        ):
            raise TypeError("measurement_arms must be a tuple of PlayArm values")
        names = [arm.name for arm in self.measurement_arms]
        if len(set(names)) != len(names):
            raise ValueError("measurement arm names must be unique")


@dataclass(frozen=True)
class TaskPlayResult:
    """What became of one generated task in a round.

    ``training_rewards`` holds the rewards of the plays the recipe trains on.
    ``measurement_rewards`` maps the name of each measurement arm to the
    rewards of its plays. Each tuple holds only the episodes that ran. An
    episode that ran and returned no reward counts as 0. ``error`` is not
    empty when the task was not played: it failed validation, the task player
    refused it, or the agent could not start on it.
    ``infrastructure_failure`` marks an error where the task could not run,
    not a failure of the solver.
    An eval task has no rewards and is not played, even with an empty error.
    """

    task: HarborTask
    training_rewards: tuple[float, ...] = ()
    measurement_rewards: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    error: str = ""
    split: Literal["train", "eval"] = "train"
    infrastructure_failure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        if not isinstance(self.measurement_rewards, Mapping) or any(
            not isinstance(arm, str) for arm in self.measurement_rewards
        ):
            raise TypeError("measurement_rewards must map arm names to tuples of rewards")
        for rewards in (self.training_rewards, *self.measurement_rewards.values()):
            if not isinstance(rewards, tuple):
                raise TypeError("rewards must be tuples of numbers")
            if any(
                isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)
                for reward in rewards
            ):
                raise ValueError("rewards must be finite numbers")
        if not isinstance(self.error, str):
            raise TypeError("error must be text")
        if self.error and (self.training_rewards or self.measurement_rewards):
            raise ValueError("a task that was not played has no rewards")
        if self.split not in ("train", "eval"):
            raise ValueError("task result split must be train or eval")
        if self.split == "eval" and (self.training_rewards or self.measurement_rewards):
            raise ValueError("an eval task result cannot contain rollout rewards")
        if self.infrastructure_failure and not self.error:
            raise ValueError("an infrastructure failure needs an error")

    @property
    def is_played(self) -> bool:
        return self.split == "train" and not self.error


class TaskGenerator(Plugin, ABC):
    """Makes tasks for one scenario, and adapts to how the agent played them.

    A deployment names the class in ``pipeline.record2task.config.generator``. Reef calls
    :meth:`from_config` once for each scenario. A generator gets what it needs
    from its config and from these calls only; it does not read the recipe's
    settings.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        """Build the generator of ``scenario`` from the section's ``config``.

        ``environ`` holds the environment the deployment was read with, for
        credentials that the config names by environment variable.
        """

    @abstractmethod
    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        """Return zero or more new tasks.

        Preserve the request's source record ids in each task's
        ``source_agent_record_ids``. An exception is a failure to run, and the
        recipe retries the generation within its limits.
        """

    @abstractmethod
    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """Check one generated task before the recipe plays it.

        Return task defects as validation errors. Raise when the checks cannot
        run; the recipe retries the generation within its limits. The task is a value: a validation
        that needs a directory writes its own copy.
        """

    async def record_results(self, results: tuple[TaskPlayResult, ...], *, generation: int) -> None:
        """Receive the result of each task that the :meth:`generate` call of ``generation`` returned.

        The recipe calls this once for each generation, after every train task
        of it has a result. Later generations may have started before, and the
        process may have restarted since the generation. A generator keeps what
        it needs for each open generation across a restart, and ignores a
        second call for a generation that it already closed. The default keeps
        nothing.
        """
