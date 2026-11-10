"""Control's worker of the training manager: it shows the manager each managed trainer and runs its actions.

The worker calls the manager outside every scenario lock (``reef.train.manager``). A failed read, decision or action
is logged and ends the round of that trainer; the next round reads its state again.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from reef.execution.workers import ServiceWorker
from reef.train.manager import ManagerAction, ManagerView, TrainingManager

logger = logging.getLogger(__name__)

#: The seconds between two rounds of the manager.
MANAGER_INTERVAL_SECONDS = 2.0


class ManagedTraining(ABC):
    """Gives the views of the managed trainers and runs a manager's actions.

    Purpose: the dispatcher implements it; ``TrainingManagerWorker`` calls it at each round.
    """

    @abstractmethod
    def manager_views(self) -> tuple[ManagerView, ...]:
        """One view for each trainer that trains, of each loaded scenario in ``auto`` mode."""

    @abstractmethod
    def run_manager_action(self, view: ManagerView, action: ManagerAction, *, job_limit: int) -> None:
        """Run ``action`` for the trainer of ``view``, as the HTTP route of that operation does. Drop a ``Train``
        action when the trainer has ``job_limit`` jobs of manager requests waiting or running. Raise the route's
        error for an action that Reef refuses."""


class TrainingManagerWorker(ServiceWorker):
    """Asks ``manager`` for the actions of each view of ``target`` and runs them, every ``interval_seconds``.

    Values: ``manager`` decides; ``target`` gives the views and runs the actions; ``interval_seconds`` is positive.
    Purpose: the dispatcher of a control instance runs one when the deployment names a manager.
    """

    def __init__(
        self,
        manager: TrainingManager,
        target: ManagedTraining,
        *,
        interval_seconds: float = MANAGER_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds is positive")
        self.manager = manager
        self.target = target
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.run, name="reef-training-manager", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            self.manage()
            self.stopped.wait(self.interval_seconds)

    def manage(self) -> None:
        """One round: read the views, and for each one ask the manager and run its actions in order."""
        try:
            views = self.target.manager_views()
        except Exception:
            logger.exception("could not read the trainers that the training manager manages")
            return
        for view in views:
            try:
                for action in self.manager.actions(view):
                    self.target.run_manager_action(view, action, job_limit=self.manager.job_limit)
            except Exception:
                logger.exception(
                    "the training manager's round failed for scenario %r component %r", view.scenario, view.component
                )


__all__ = ["MANAGER_INTERVAL_SECONDS", "ManagedTraining", "TrainingManagerWorker"]
