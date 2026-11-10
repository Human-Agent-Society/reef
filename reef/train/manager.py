"""The training manager: the plugin that decides what a scenario in ``auto`` mode trains (RFC #1324).

Reef runs explicit training operations: a training request, a new training start, and the cancel of a job. In
``manual`` mode a person sends them. In ``auto`` mode Reef starts a job by its own rule when ``job_samples``
untrained samples wait, unless the deployment names a manager (``training.manager``). Then the manager decides: control
shows it one trainer's state (``ManagerView``) at an interval, and runs the actions that it returns
(``ManagerAction``) with the code of the HTTP routes. A manager never changes what a scenario serves.

A manager can take minutes for one decision, for example when it calls a model. It runs on control's manager
thread, outside every scenario lock, so its view can be old when its actions run. Reef checks each action when it
runs, as it checks a person's request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.core.plugins import ConfiguredPlugin
from reef.core.values import require_text

#: The ``session`` of each training request that a manager's ``Train`` action files.
MANAGER_SESSION = "training-manager"
#: The start of the ID of each such request, by which the job listing tells a manager's jobs from a person's.
MANAGER_REQUEST_PREFIX = "manager-"


@dataclass(frozen=True)
class ManagedJob:
    """One training job of a scenario that has not ended, as a manager sees it.

    Values: ``job_id`` is the job's non-empty ID. ``state`` is ``"queued"`` or ``"running"``. ``train_from`` is the ID
    of the artifact that the job starts from, or None when Reef does not know it. ``managed`` is whether the job
    trains a request that a manager filed.
    Purpose: control makes it from the job listing for ``ManagerView.jobs``.
    """

    job_id: str
    state: str
    train_from: str | None
    managed: bool

    def __post_init__(self) -> None:
        require_text(self.job_id, "job_id")
        if self.state not in ("queued", "running"):
            raise ValueError("a managed job is queued or running")


@dataclass(frozen=True)
class ManagerView:
    """The state of one trainer of one scenario at one time, which a manager reads to decide.

    Values: ``scenario`` is the scenario's non-empty name, and ``component`` the trainer's component.
    ``training_start`` is the ID of the artifact that a job with no ``train_from`` starts from, or None when Reef
    does not know it. ``selection_serves`` is whether the artifact that the scenario selected answers its requests
    now; while it does not, new samples still come from the artifact before. ``untrained_samples`` are the IDs of the
    samples that an automatic job would train next, in store order, and ``job_samples`` is the number of them that
    the recipe's job size asks for, a positive int. ``waiting_requests`` is the number of queued training requests
    that no job trains yet. ``jobs`` are the scenario's jobs that have not ended, oldest first. ``artifacts`` are the
    rows of the scenario's artifact listing, newest first, each with its ``artifact_id``, ``component``,
    ``parent_artifact_id`` and ``metrics``.
    Purpose: control makes one for each trainer of a scenario in ``auto`` mode (``Dispatcher.manage_scenarios``),
    and ``TrainingManager.actions`` reads it.
    """

    scenario: str
    component: str
    training_start: str | None
    selection_serves: bool
    untrained_samples: tuple[str, ...]
    job_samples: int
    waiting_requests: int
    jobs: tuple[ManagedJob, ...] = ()
    artifacts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        require_text(self.scenario, "scenario")
        require_text(self.component, "component")
        if isinstance(self.job_samples, bool) or not isinstance(self.job_samples, int) or self.job_samples < 1:
            raise ValueError("job_samples is a positive int")
        if isinstance(self.waiting_requests, bool) or not isinstance(self.waiting_requests, int):
            raise TypeError("waiting_requests is an int")
        if self.waiting_requests < 0:
            raise ValueError("waiting_requests is not negative")


@dataclass(frozen=True)
class Train:
    """The action that files one training request.

    Values: ``samples`` are the distinct IDs of the samples that the job trains, one or more. ``reason`` is the
    non-empty text of the request, which the job's commits keep. ``parameters`` go over the recipe's job parameters.
    ``train_from`` is the ID of the artifact that the job starts from; the job then runs beside the other jobs and
    does not move the training start. With None, the job continues the training start, one such job at a time.
    Purpose: a manager returns it; control files it as ``POST /reef/train`` does.
    """

    samples: tuple[str, ...]
    reason: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    train_from: str | None = None

    def __post_init__(self) -> None:
        if not self.samples or len(set(self.samples)) != len(self.samples):
            raise ValueError("a Train action names one or more distinct samples")
        require_text(self.reason, "reason")


@dataclass(frozen=True)
class TrainFrom:
    """The action that sets the trainer's training start to the artifact ``artifact_id``, as the ``train-from``
    route does. A manager returns it; control runs it."""

    artifact_id: str

    def __post_init__(self) -> None:
        require_text(self.artifact_id, "artifact_id")


@dataclass(frozen=True)
class Cancel:
    """The action that cancels the job ``job_id`` of the scenario, as the job's cancel route does. A manager returns
    it; control runs it."""

    job_id: str

    def __post_init__(self) -> None:
        require_text(self.job_id, "job_id")


ManagerAction = Train | TrainFrom | Cancel


class TrainingManager(ConfiguredPlugin, ABC):
    """Decides what the scenarios in ``auto`` mode train.

    Operations: ``actions(view)`` gives the actions that Reef runs now for the trainer of ``view``, in order; an empty
    sequence waits. It decides from ``view`` and the manager's own configuration. ``job_limit`` is the most jobs of
    the manager's requests that one trainer has waiting or running; Reef drops a ``Train`` action above it.
    Purpose: a deployment names one as ``training.manager``; control calls it for each trainer of each loaded
    scenario in ``auto`` mode.
    """

    @property
    def job_limit(self) -> int:
        """The most jobs of this manager's requests that one trainer has waiting or running; 1 by default."""
        return 1

    @abstractmethod
    def actions(self, view: ManagerView) -> Sequence[ManagerAction]:
        """The actions that Reef runs now for the trainer of ``view``, in order."""


__all__ = [
    "MANAGER_REQUEST_PREFIX",
    "MANAGER_SESSION",
    "Cancel",
    "ManagedJob",
    "ManagerAction",
    "ManagerView",
    "Train",
    "TrainFrom",
    "TrainingManager",
]
