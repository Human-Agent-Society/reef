"""Control's watcher of the shared store: records, samples, job progress and instructions that other processes add.

In the split layout, the inference service appends records, the samples service stores samples and the training
service runs jobs in other processes. Control learns of all three from the store alone: each round,
``StoreChangesWatcher`` reads the newest record sequence of each loaded scenario and the changed items of the
samples, jobs and instructions queues. New records make control ask the samples service for a selection; a finished
selection, or a new training instruction, wakes the scenario's local workers, which build the next job from the
sample store; a job that saved a checkpoint or ended wakes them too, so they commit it without waiting for their
next poll. In the one-process layout the same events also arrive in process; a second request or wake changes
nothing.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from reef.execution.workers import ServiceWorker
from reef.record2sample.service import RECORD_SELECTION_KIND, SAMPLES_QUEUE
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.storage.work_queue import WorkQueue
from reef.train.instructions import INSTRUCTIONS_QUEUE
from reef.train.job_queue import JOBS_QUEUE

logger = logging.getLogger(__name__)


class StoreChanges(ABC):
    """What control does when the shared store changes.

    Purpose: the dispatcher implements it; ``StoreChangesWatcher`` calls it.
    """

    @abstractmethod
    def records_added(self, current: Scenario) -> None:
        """``current``'s record store holds records that this process did not accept."""

    @abstractmethod
    def samples_added(self, scenario: str) -> None:
        """The samples service finished a selection of ``scenario``."""

    @abstractmethod
    def job_changed(self, scenario: str) -> None:
        """A training job of ``scenario`` saved a checkpoint or ended."""

    @abstractmethod
    def instructions_added(self, scenario: str) -> None:
        """A process queued a training instruction of ``scenario``."""


class StoreChangesWatcher(ServiceWorker):
    """Reads the record store and the samples queue for changes that other processes make.

    Values: ``registry`` holds the loaded scenarios; ``queue`` is the work queue; ``changes`` is called for each
    change; ``interval_seconds`` is the time between rounds. ``record_sequences`` holds the newest record sequence
    seen for each loaded scenario, ``queue_sequence`` the newest ``changed_sequence`` read from the samples queue,
    ``jobs_sequence`` the newest one read from the jobs queue, and ``instruction_sequence`` the newest one read from
    the instructions queue.

    Purpose: the dispatcher of a control instance runs it.
    """

    def __init__(
        self, registry: ScenarioRegistry, queue: WorkQueue, changes: StoreChanges, *, interval_seconds: float = 1.0
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds is positive")
        self.registry = registry
        self.queue = queue
        self.changes = changes
        self.interval_seconds = interval_seconds
        self.record_sequences: dict[str, int] = {}
        self.queue_sequence = 0
        self.jobs_sequence = 0
        self.instruction_sequence = 0
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name="reef-store-changes", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("could not read the changes of the shared store")
            self.stopped.wait(self.interval_seconds)

    def run_once(self) -> None:
        """One round: report the scenarios with new records, then those with new samples, then those whose jobs
        saved a checkpoint or ended, then those with new training instructions.

        A sequence advances only after its report returns, so a report that raises comes again next round.
        """
        loaded = {current.name: current for current in self.registry.loaded_scenarios()}
        for name in set(self.record_sequences) - set(loaded):
            self.record_sequences.pop(name)
        for name, current in loaded.items():
            newest = current.records.last_sequence(name)
            seen = self.record_sequences.get(name)
            # A scenario seen for the first time asks for a selection when it loads.
            if seen is not None and newest > seen:
                self.changes.records_added(current)
            self.record_sequences[name] = newest
        while changed := self.queue.read_changes(SAMPLES_QUEUE, self.queue_sequence):
            # A selection that brought no new input gives the trainer's worker nothing to build.
            finished = sorted(
                {
                    item.scenario
                    for item in changed
                    if item.kind == RECORD_SELECTION_KIND
                    and item.state == "finished"
                    and item.result is not None
                    and item.result.get("changed") is True
                }
            )
            for scenario in finished:
                self.changes.samples_added(scenario)
            self.queue_sequence = changed[-1].changed_sequence
        while changed := self.queue.read_changes(JOBS_QUEUE, self.jobs_sequence):
            for scenario in sorted({item.scenario for item in changed if item.final or item.progress}):
                self.changes.job_changed(scenario)
            self.jobs_sequence = changed[-1].changed_sequence
        while changed := self.queue.read_changes(INSTRUCTIONS_QUEUE, self.instruction_sequence):
            for scenario in sorted({item.scenario for item in changed if item.state == "queued"}):
                self.changes.instructions_added(scenario)
            self.instruction_sequence = changed[-1].changed_sequence


__all__ = ["StoreChanges", "StoreChangesWatcher"]
