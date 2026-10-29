"""The method contract of harness evolution: what a method implements, and what each step gives it.

A method is a :class:`Method`. Each step gives it one :class:`StepContext`:
the step's :class:`~reef.train.harness_backend.workspace.Workspace`, where the
harness's components and the step's material are files, and the
:class:`Tools`, the services that Reef runs for the method. The method edits
``context.workspace.harness``; the tree it leaves there is the step's
candidate, and an unchanged tree skips the step. It returns a
:class:`StepResult` that describes what it did. A method never subclasses
the backend.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from reef.core.plugins import ConfiguredPlugin
from reef.core.requirements import parse_requires
from reef.evaluation.harness import RolloutRecord
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, ValidationResult
from reef.train.harness_backend.workspace import Workspace


class ModelCalls(ABC):
    """The step's model-call budget, its call record, and its live activity.

    Purpose: every binding in ``Tools.models`` spends and records through it.
    A method that sends model traffic another way, such as the gateway of an
    agent process, spends and records through ``Tools.calls`` itself.
    """

    @abstractmethod
    def spend(self) -> None:
        """Count one call; raise ``RuntimeError`` when the step's budget is exhausted."""

    @abstractmethod
    def record(self, entry: Mapping[str, Any]) -> None:
        """Append one call to the step's call record, which redacts and clips its texts."""

    @abstractmethod
    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        """Add one line to the step's live activity: ``kind`` names what acted (such as ``model``, ``agent`` or
        ``evaluate``), ``text`` what it did, and ``failed`` marks a line that went wrong."""


class Tools(ABC):
    """What a method may do in one step besides editing its workspace.

    Purpose: the backend makes one per step and passes it to ``Method.step``.
    """

    @property
    @abstractmethod
    def models(self) -> ModelBindings:
        """The model bindings: ``served`` (the model under test) and every ``evolution.models`` entry.

        Every call spends from the step's budget and lands in the step's call record.
        """

    @property
    @abstractmethod
    def calls(self) -> ModelCalls:
        """The step's model-call budget, record and activity."""

    @abstractmethod
    def check(self, tree: HarnessTree) -> ValidationResult:
        """Every problem ``tree`` has for the adapter; a candidate with problems is not evaluated."""

    @abstractmethod
    def evaluate(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int = 1) -> tuple[RolloutRecord, ...]:
        """The records of ``repeats`` rollouts of ``tree`` on each task, task by task.

        The rollouts run like the step's evaluation: the same rendering, worker
        pool and scorer. A record's score is a ``RolloutScore``: failed when the
        environment could not run the rollout, 0 when the tree gave no answer.
        A rollout in the history that did not fail is reused, so asking again
        costs nothing; ask for more repeats to get new samples. Raise
        ``RuntimeError`` when the deployment configured no scorer, or when the
        call needs more rollouts than the step's budget has left; the whole
        call is refused. Raise ``ValueError`` when ``tree`` fails ``check``, and
        ``TypeError`` when ``tasks`` is one string. A refused call adds nothing
        to the history.
        """


@dataclass(frozen=True)
class StepResult:
    """What a method says about its step.

    Values: ``summary`` is plain text for the step pages, possibly empty.
    ``notes`` is a JSON object: its keys are text, and ``json.dumps`` writes
    it without NaN or infinity. The commit metrics keep it as
    ``method_notes``. ``requires`` are what the change needs from the
    person's machine beyond the request's own, as ``reef.core.requirements``
    items.
    Purpose: the method makes it, as the result of ``Method.step``; the
    backend records it with the step and the history, whether the step
    keeps a candidate or not.
    """

    summary: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            raise TypeError("StepResult.summary must be text")
        notes = dict(self.notes)
        if any(not isinstance(key, str) for key in notes):
            raise ValueError("StepResult.notes must be a JSON object, whose keys are text")
        try:
            json.dumps(notes, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"StepResult.notes must be a JSON object: {exc}") from exc
        object.__setattr__(self, "notes", MappingProxyType(notes))
        object.__setattr__(self, "requires", tuple(parse_requires(list(self.requires))))


@dataclass(frozen=True)
class StepContext:
    """What one step gives a method.

    Values: ``workspace`` is the step's directories: the head tree in
    ``harness/``, the material, the method's state, scratch and output. Data
    and the harness's components are files there. ``tools`` are the services
    that Reef runs for the method in this step: model calls, the call record,
    the tree check and the evaluation.
    Purpose: the backend makes one per step and passes it to ``Method.step``.
    """

    workspace: Workspace
    tools: Tools

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace) or not isinstance(self.tools, Tools):
            raise TypeError("StepContext needs a Workspace and Tools")


class Method(ConfiguredPlugin, ABC):
    """Base class of a harness evolution method.

    Purpose: a deployment names the class in ``evolution.method``, the recipe
    builds it with ``from_config``, and the backend calls ``step`` once per
    step. A method keeps what it learns across steps in the workspace's
    ``state/``; the backend commits that directory with the step.
    """

    @abstractmethod
    def step(self, context: StepContext) -> StepResult:
        """Edit ``context.workspace.harness`` into this step's candidate, or leave it unchanged to skip the step."""


__all__ = ["Method", "ModelCalls", "StepContext", "StepResult", "Tools"]
