"""The task generator contract of the generated-task recipe: the generator, its validation results, and task results.

A deployment selects one :class:`TaskGenerator` in its
``pipeline.record2sample.config.generator`` setting. The recipe asks the
generator for tasks with a ``TaskGenerationRequest``
(:mod:`reef.record2sample.generator.inputs`), validates each one, runs the
valid ones, and sends the results back to the generator. This module holds the
values, their field validation, and the generator interface. Consumers own
file reads, transport limits, and execution.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.plugins import Plugin
from reef.core.tasks import HarborTask
from reef.record2sample.generator.inputs import TaskGenerationRequest


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


def check_rewards(rewards: object) -> None:
    """Raise unless ``rewards`` is a non-empty tuple of finite numbers."""
    if not isinstance(rewards, tuple) or not rewards:
        raise TypeError("rollout_rewards must be a non-empty tuple of numbers")
    if any(
        isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)
        for reward in rewards
    ):
        raise ValueError("rewards must be finite numbers")


@dataclass(frozen=True)
class RejectedTask:
    """A candidate that the recipe rejected, so it was never published.

    Values: ``(task, reason)`` with ``task`` a ``HarborTask`` and ``reason`` a
    non-empty text: the failed validation, or the conflict that kept the
    candidate out of the task store.
    Purpose: the recipe sends it to :meth:`TaskGenerator.record_results` for a
    candidate whose outcome in the task store is a rejection.
    """

    task: HarborTask
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("a rejected task needs its reason")


@dataclass(frozen=True)
class HeldOutTask:
    """A task published to the eval split: it runs no rollouts and gives no training feedback.

    Values: ``task``, a ``HarborTask``.
    Purpose: the recipe sends it for a candidate published to ``eval``. A
    generator must not use it as training feedback or as experience.
    """

    task: HarborTask

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")


@dataclass(frozen=True)
class RolledOutTask:
    """A train task whose first rollout ran: the verifier rewards of its training rollouts.

    Values: ``(task, rollout_rewards)`` with ``task`` a ``HarborTask`` and
    ``rollout_rewards`` a non-empty tuple of finite numbers, one for each
    training rollout of the first rollout that ran, in rollout order, as the
    rollout's policy scored them before it trained on them. A rollout that ran
    and returned no reward counts as 0.
    Purpose: the recipe sends it for a train task whose first rollout ran.
    """

    task: HarborTask
    rollout_rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        check_rewards(self.rollout_rewards)


@dataclass(frozen=True)
class FailedTask:
    """A train task whose first rollout ran no rollout.

    Values: ``(task, error, infrastructure_failure)`` with ``task`` a
    ``HarborTask``, ``error`` a non-empty text and ``infrastructure_failure`` a
    bool. ``infrastructure_failure`` is true when the task could not run at all,
    for example because the sandbox failed or its result was lost; it is not a
    failure of the solver. It is false when the generator service refused the
    task.
    Purpose: the recipe sends it for a train task whose first rollout ran no
    rollout.
    """

    task: HarborTask
    error: str
    infrastructure_failure: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        if not isinstance(self.error, str) or not self.error:
            raise ValueError("a failed task needs the error that stopped it")
        if not isinstance(self.infrastructure_failure, bool):
            raise TypeError("infrastructure_failure must be a bool")


#: What became of one candidate of a generation. Each candidate with an outcome is in exactly one case: a rejected
#: candidate is a ``RejectedTask``, an eval task a ``HeldOutTask``, and a train task a ``RolledOutTask`` or a
#: ``FailedTask`` by its first rollout.
GeneratedTaskResult = RejectedTask | HeldOutTask | RolledOutTask | FailedTask


class TaskGenerator(Plugin, ABC):
    """Makes tasks for one scenario, and adapts to how the agent did on them.

    A deployment names the class in ``pipeline.record2sample.config.generator``. Reef calls
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
    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        """Return zero or more new tasks.

        Preserve the request's source record ids in each task's
        ``source_agent_record_ids``. An exception stops the round.
        """

    @abstractmethod
    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """Check one generated task before the recipe runs its rollouts.

        Return task defects as validation errors. Raise when the checks cannot
        run; the exception stops the round. The task is a value: a validation
        that needs a directory writes its own copy.
        """

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        """Receive the results of the tasks that the :meth:`generate` call of ``generation`` returned.

        ``results`` holds one ``GeneratedTaskResult`` for each candidate of the
        generation, in candidate order. The recipe calls this once every train
        task of the generation has a first rollout: possibly after later
        ``generate`` calls, and after a restart. A crash before the recipe
        records the delivery sends the same results again, so a second call for
        the same generation must change nothing. The default keeps nothing.
        """
