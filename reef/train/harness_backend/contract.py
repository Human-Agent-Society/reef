"""The method contract of harness evolution: what a method implements, and what each job gives it.

A method is a :class:`Method`. Each job gives it one :class:`MethodJob`:
the job's :class:`~reef.train.harness_backend.workspace.Workspace`, where the
harness's components and the job's material are files, and the
:class:`Services`, the services that Reef runs for the method: model calls, the
tree check, the evaluation, and a run of the deployment's coding agent. The
method edits ``job.workspace.harness``; the tree it leaves there is the
job's candidate, and an unchanged tree skips the job. What it says about the
job it writes to ``output/`` (``Workspace.write_output``). A method never
subclasses the backend.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from reef.core.plugins import ConfiguredPlugin
from reef.evaluation.harness import RunRecord
from reef.harness.gateway import ModelCalls
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, ValidationResult
from reef.train.harness_backend.workspace import Workspace


@dataclass(frozen=True)
class AgentOutcome:
    """How one run of the deployment's coding agent ended.

    Values: ``exit_code`` is the agent's exit status, ``None`` when it did
    not exit by itself. ``timed_out`` is True when it ran past
    ``evolution.agent.timeout_s``. ``error`` is why it could not start,
    ``None`` when it started. Exactly one holds: an exit code, a time out or
    an error. ``seconds`` ≥ 0 is how long the run took.
    Purpose: ``Services.run_agent`` returns it; the method decides what the run
    is worth, and ``to_json`` goes into its notes.
    """

    exit_code: int | None
    timed_out: bool
    error: str | None
    seconds: float

    def __post_init__(self) -> None:
        ends = (self.exit_code is not None, self.timed_out, self.error is not None)
        if sum(ends) != 1:
            raise ValueError("an agent run ends in exactly one way: an exit code, a time out or an error")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("AgentOutcome.seconds must be a finite number of at least 0")

    @property
    def succeeded(self) -> bool:
        """The agent exited by itself with status 0."""
        return self.exit_code == 0

    def describe(self) -> str:
        """How the run ended, in a few words, for a summary."""
        if self.timed_out:
            return f"ran past its limit after {self.seconds:g} s"
        if self.error is not None:
            return f"could not start: {self.error}"
        return f"exited {self.exit_code} after {self.seconds:g} s"

    def to_json(self) -> dict[str, Any]:
        return {"exit_code": self.exit_code, "timed_out": self.timed_out, "error": self.error, "seconds": self.seconds}


class Services(ABC):
    """What a method may do in one job besides editing its workspace.

    Purpose: the backend makes one per job and passes it to ``Method.run``.
    """

    @property
    @abstractmethod
    def models(self) -> ModelBindings:
        """The model bindings: ``served`` (the model under test) and every ``evolution.models`` entry.

        Every call spends from the job's budget and lands in the job's call record.
        """

    @property
    @abstractmethod
    def calls(self) -> ModelCalls:
        """The job's model-call budget, record and activity."""

    @abstractmethod
    def check(self, tree: HarnessTree) -> ValidationResult:
        """Every problem ``tree`` has for the adapter; a candidate with problems is not evaluated."""

    @abstractmethod
    def evaluate(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int = 1) -> tuple[RunRecord, ...]:
        """The records of ``repeats`` runs of ``tree`` on each task, task by task.

        The tasks run like the job's evaluation: the same rendering, worker
        pool and scorer. A record's score is a ``RolloutScore``: failed when the
        environment could not run the run, 0 when the tree gave no answer.
        A run in the history that did not fail is reused, so asking again
        costs nothing; ask for more repeats to get new samples. Raise
        ``RuntimeError`` when the deployment configured no scorer, or when the
        call needs more runs than the job's budget has left; the whole
        call is refused. Raise ``ValueError`` when ``tree`` fails ``check``, and
        ``TypeError`` when ``tasks`` is one string. A refused call adds nothing
        to the history.
        """

    @abstractmethod
    def run_agent(self, prompt: str, *, tree: HarnessTree, model: str) -> AgentOutcome:
        """Run the deployment's coding agent (``evolution.agent``) once in the job's workspace, and wait for it.

        The agent works in the workspace's root: it may write ``harness/``,
        ``state/``, ``scratch/`` and ``output/``, and it reads ``material/``
        and the history. ``tree`` is the agent's own harness, such as its
        rules and skills, rendered for the agent's adapter after Reef's
        protocol rules, which describe the workspace and the tools. The
        agent's model calls go to the binding ``model`` names, through a
        loopback gateway that spends from the job's budget and records each
        call; the gateway also serves ``check`` and ``evaluate`` on the
        workspace's trees. Raise ``RuntimeError`` when no agent runs for the
        method (it needs ``runs_agent`` and ``evolution.agent``), ``KeyError``
        when ``model`` names no binding, and ``ValueError`` when ``tree`` does
        not render for the agent's adapter.
        """


@dataclass(frozen=True)
class MethodJob:
    """What one job gives a method.

    Values: ``workspace`` is the job's directories: the head tree in
    ``harness/``, the material, the method's state, scratch and output. Data
    and the harness's components are files there. ``services`` are the services
    that Reef runs for the method in this job: model calls, the call record,
    the tree check and the evaluation.
    Purpose: the backend makes one per job and passes it to ``Method.run``.
    """

    workspace: Workspace
    services: Services

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace) or not isinstance(self.services, Services):
            raise TypeError("MethodJob needs a Workspace and Services")


class Method(ConfiguredPlugin, ABC):
    """Base class of a harness evolution method.

    Values: ``runs_agent`` is True for a method that runs the deployment's
    coding agent (``Services.run_agent``); only such a method gets the agent,
    and the recipe refuses a deployment that configures none for it.
    Purpose: a deployment names the class in ``evolution.method``, the recipe
    builds it with ``from_config``, and the backend calls ``run`` once per
    job. A method keeps what it learns across jobs in the workspace's
    ``state/``; the backend commits that directory with the job.
    """

    runs_agent: ClassVar[bool] = False

    @abstractmethod
    def run(self, job: MethodJob) -> None:
        """Edit ``job.workspace.harness`` into this job's candidate, or leave it unchanged to skip the job.

        Say what the job did in ``output/``: ``job.workspace.write_output``
        writes the summary, the notes and the requires.
        """


__all__ = ["AgentOutcome", "Method", "MethodJob", "Services"]
