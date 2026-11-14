"""The delete of a scenario: a mark that the request writes, and the cleanup that control's sweep runs.

A delete writes two things, in this order: a cleanup item in the ``scenario_deletes`` queue, and the archive of the
scenario's registration in the artifact store (``CachedRepositoryBackendFactory.archive_registration``). The
archived registration is the mark: from it on, no path creates, loads or trains the scenario, and a training worker
that stores for it discards what it stored. The cleanup item tells control's sweep which scenarios to clean up, so
the sweep lists no archive.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Sequence
from threading import Lock

from reef.artifact.repository import RepositoryBackendFactory
from reef.core.errors import ReefError
from reef.dispatcher.served import ServedModelsPublisher
from reef.execution.workers import ServiceWorker
from reef.record2sample.service import SAMPLES_QUEUE, request_selection
from reef.scenario.registry import ScenarioRegistry
from reef.storage.work_queue import JsonValue, LeaseExtender, WorkLeaseLostError, WorkQueue
from reef.train.component import TrainingComponent
from reef.train.job_queue import CHECKS_QUEUE, archive_item_id, cancel_scenario_jobs

logger = logging.getLogger(__name__)

#: The queue of the cleanup items of deleted scenarios. Control's sweep takes them.
DELETES_QUEUE = "scenario_deletes"
#: The kind of a cleanup item.
DELETE_KIND = "scenario_delete"
#: The wait, in seconds, between two looks at the items of a deleted scenario that other services answer.
ANSWER_POLL_SECONDS = 0.05


class AnswerPendingError(ReefError):
    """A selection or an archive item of a deleted scenario has not ended within the cleanup's wait: the cleanup
    item waits for the next sweep."""


def request_cleanup(queue: WorkQueue, scenario: str) -> None:
    """Add the cleanup item of ``scenario``, unless a queued one covers it (``WorkQueue.request``). A delete calls
    it before it archives the registration, so a stop between the two leaves an item and no mark: the sweep then
    finds the scenario whole and ends the item."""
    queue.request(DELETES_QUEUE, scenario, DELETE_KIND, f"scenario:{scenario}", {})


class DeletedScenarioFollower(ServiceWorker):
    """Drops what a process without control keeps of each deleted scenario.

    Values: ``queue`` holds the cleanup items; ``registry`` holds the loaded scenarios and the backends of this
    process; ``interval_seconds`` is the positive time between two reads; ``after_sequence`` is the newest
    ``changed_sequence`` of a cleanup item that this follower handled, 0 before the first.

    Purpose: the dispatcher of a process that does not run control runs one, such as the inference, samples or
    harness service. Each cleanup item names a scenario that a delete marked, or tried to mark. The follower closes
    the scenario's loaded instance and drops its backend, its clone and its cached snapshots
    (``ScenarioRegistry.drop_deleted``), which does nothing for a scenario that is not marked. When it starts, it
    first drops every deleted scenario that this process keeps something of
    (``ScenarioRegistry.drop_deleted_local_scenarios``), so a process that was down longer than the work retention
    still drops it. A failed drop is tried again at the next read.
    """

    def __init__(self, queue: WorkQueue, registry: ScenarioRegistry, *, interval_seconds: float = 2.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds is positive")
        self.queue = queue
        self.registry = registry
        self.interval_seconds = interval_seconds
        self.after_sequence = 0
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.run, name="reef-deleted-scenarios", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        started = False
        while not self.stopped.is_set():
            try:
                if not started:
                    # A cleanup item goes after the work retention: what an earlier run left is found by the mark.
                    self.registry.drop_deleted_local_scenarios()
                    started = True
                self.run_once()
            except Exception:
                logger.exception("could not drop the deleted scenarios")
            self.stopped.wait(self.interval_seconds)

    def run_once(self) -> None:
        """Drop the scenario of each cleanup item that changed since the last read, in change order."""
        for item in self.queue.read_changes(DELETES_QUEUE, self.after_sequence):
            self.registry.drop_deleted(item.scenario)
            self.after_sequence = item.changed_sequence


class DeletedScenarioCleanup:
    """Removes what control owns of each deleted scenario, and asks the training service to remove what it owns.

    Values: ``queue`` holds the cleanup items and the scenario's jobs; ``registry`` holds the scenario's lock, its
    storage and its registration; ``backends`` opens the scenario's artifact store; ``publisher`` writes the served
    models when this process holds the control lease, else it is None; ``job_components`` are control's views of
    the training components (``JobQueueComponent``); ``record_retention_lock`` keeps record retention from running
    while the storage moves; ``worker_id`` is a non-empty name of this cleanup in the queue;
    ``answer_seconds`` is the longest wait, not negative, for the samples and training services to end their items
    of one deleted scenario.

    Purpose: control's sweep calls ``clean_up`` (``EndedWorkSweeper``). Every step of one scenario is idempotent,
    so a step that failed, and a stop at any point, leave work that the next ``clean_up`` finishes. A cleanup item
    ends ``finished`` only after every step of its scenario ran, and a failed item is queued again by the next
    call, so the cleanup of a scenario runs until nothing of it is left.
    """

    def __init__(
        self,
        queue: WorkQueue,
        registry: ScenarioRegistry,
        backends: RepositoryBackendFactory,
        *,
        publisher: ServedModelsPublisher | None,
        job_components: Sequence[TrainingComponent],
        record_retention_lock: Lock,
        worker_id: str,
        answer_seconds: float = 30.0,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id is a non-empty string")
        if answer_seconds < 0:
            raise ValueError("answer_seconds is not negative")
        self.queue = queue
        self.registry = registry
        self.backends = backends
        self.publisher = publisher
        self.job_components = tuple(job_components)
        self.record_retention_lock = record_retention_lock
        self.worker_id = worker_id
        self.answer_seconds = answer_seconds
        #: Held for one whole ``clean_up``, so two calls at once run one after the other.
        self.lock = threading.Lock()
        #: Set by ``stop``: the wait of the running ``clean_up`` for the samples service ends at once.
        self.stopping = threading.Event()

    def stop(self) -> None:
        """End the wait of the running ``clean_up`` for the samples service; its item waits for the next call. The
        sweep calls it when its process stops. The next ``clean_up`` waits again."""
        self.stopping.set()

    def clean_up(self) -> int:
        """Queue each failed cleanup item again, then clean up the scenario of each queued item; return how many
        scenarios were cleaned up in full. A scenario whose step failed keeps its item, failed, for the next call."""
        cleaned = 0
        with self.lock:
            self.stopping.clear()
            for ended in self.queue.items(DELETES_QUEUE):
                if ended.state == "failed":
                    self.queue.retry(DELETES_QUEUE, ended.id)
            while (item := self.queue.take(DELETES_QUEUE, self.worker_id, (DELETE_KIND,))) is not None:
                try:
                    with LeaseExtender(self.queue, item, self.worker_id):
                        result = self.clean_up_scenario(item.scenario)
                    self.queue.finish(DELETES_QUEUE, item.id, self.worker_id, result)
                except WorkLeaseLostError:
                    # Another control instance continues the item.
                    continue
                except Exception as exc:
                    logger.exception(
                        "could not clean up the deleted scenario %r; the next sweep tries again", item.scenario
                    )
                    with contextlib.suppress(WorkLeaseLostError):
                        self.queue.fail(DELETES_QUEUE, item.id, self.worker_id, repr(exc), retryable=False)
                    continue
                cleaned += 1 if result["deleted"] else 0
        return cleaned

    def clean_up_scenario(self, scenario: str) -> dict[str, JsonValue]:
        """Remove what control owns of ``scenario`` when its registration is archived; ``{"deleted": false}`` and
        nothing removed otherwise, because the delete stopped before its mark and the scenario is whole.

        The steps, each of which a repeat finds done:

        - The inference service stops serving the scenario (``ServedModelsPublisher.withdraw``). The delete did
          this already, unless it stopped after its mark or ran in a process without control.
        - Every job of the scenario that has not ended is cancelled (``cancel_scenario_jobs``).
        - Every stored content of the scenario leaves the artifact store (``RepositoryBackend.discard_all_content``).
          A training worker that stores after this finds the mark and discards what it stored. Then this process
          drops its backend of the scenario and its local state (``RepositoryBackendFactory.forget_scenario``).
        - Each training service is asked to move aside what its component keeps for the scenario
          (``JobQueueComponent.archive_scenario``, item ``archive_item_id``). An archive item that failed is queued
          again, once in each call.
        - The samples service is asked, by a ``delete`` selection of each trainer that had a selection, to drop its
          instance and its backend of the scenario (``LocalSampleProducers``). The selection runs after any
          selection of the trainer that runs now, so once every selection of the scenario has ended, no selection
          writes samples any more (``UndeletedSelection``). The cleanup waits for that.
        - The storage archives the scenario's records, commits, tasks, samples and settings
          (``ScenarioStorage.archive``).
        - The cleanup waits for every archive item to finish.

        Each wait lasts up to ``answer_seconds`` in all, without the scenario's lock. When a service has not
        answered by then, the call raises ``AnswerPendingError``, and the next sweep runs every step again. So the
        item ends only after the training services archived their directories.

        Return ``{"deleted": true, "archived": [...]}`` with the archive locations of the storage.
        """
        with self.registry.lock_for(scenario):
            if not self.registry.was_deleted(scenario):
                return {"deleted": False}
            if self.publisher is not None:
                self.publisher.withdraw(scenario)
            cancel_scenario_jobs(self.queue, scenario)
            self.backends(scenario).discard_all_content()
            self.backends.forget_scenario(scenario)
            archives = [archive_item_id(scenario, component.name) for component in self.job_components]
            for component in self.job_components:
                component.archive_scenario(scenario)
            for item_id in archives:
                archive = self.queue.read(CHECKS_QUEUE, item_id)
                if archive is not None and archive.state == "failed":
                    self.queue.retry(CHECKS_QUEUE, item_id)
            selections = self.queue.items(SAMPLES_QUEUE, scenario=scenario)
            for component_name in sorted({str(selection.payload["component"]) for selection in selections}):
                request_selection(self.queue, scenario, component_name, trigger="delete")
        deadline = time.monotonic() + self.answer_seconds
        while pending := [item.id for item in self.queue.items(SAMPLES_QUEUE, scenario=scenario) if not item.final]:
            self.wait_for_answer(scenario, pending, deadline)
        with self.record_retention_lock, self.registry.lock_for(scenario):
            archived: list[JsonValue] = list(self.registry.archive_store(scenario))
        while pending := [item_id for item_id in archives if not self.archive_finished(item_id)]:
            self.wait_for_answer(scenario, pending, deadline)
        self.registry.forget_lock(scenario)
        return {"deleted": True, "archived": archived}

    def archive_finished(self, item_id: str) -> bool:
        """Whether the archive item ``item_id`` of a training service finished; a failed one has not."""
        archive = self.queue.read(CHECKS_QUEUE, item_id)
        return archive is not None and archive.state == "finished"

    def wait_for_answer(self, scenario: str, pending: list[str], deadline: float) -> None:
        """Wait ``ANSWER_POLL_SECONDS`` for the items ``pending`` of other services; raise ``AnswerPendingError`` at
        ``deadline`` (``time.monotonic``) or when ``stop`` was called."""
        if time.monotonic() >= deadline or self.stopping.wait(ANSWER_POLL_SECONDS):
            raise AnswerPendingError(f"the services have not answered the delete of {scenario!r} yet: {pending}")


__all__ = [
    "DELETES_QUEUE",
    "DELETE_KIND",
    "AnswerPendingError",
    "DeletedScenarioCleanup",
    "DeletedScenarioFollower",
    "request_cleanup",
]
