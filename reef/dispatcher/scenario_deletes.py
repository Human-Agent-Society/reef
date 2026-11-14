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
from collections.abc import Sequence
from threading import Lock

from reef.artifact.repository import RepositoryBackendFactory
from reef.dispatcher.served import ServedModelsPublisher
from reef.record2sample.service import SAMPLES_QUEUE, request_selection
from reef.scenario.registry import ScenarioRegistry
from reef.storage.work_queue import JsonValue, LeaseExtender, WorkLeaseLostError, WorkQueue
from reef.train.component import TrainingComponent
from reef.train.job_queue import cancel_scenario_jobs

logger = logging.getLogger(__name__)

#: The queue of the cleanup items of deleted scenarios. Control's sweep takes them.
DELETES_QUEUE = "scenario_deletes"
#: The kind of a cleanup item.
DELETE_KIND = "scenario_delete"


def request_cleanup(queue: WorkQueue, scenario: str) -> None:
    """Add the cleanup item of ``scenario``, unless a queued one covers it (``WorkQueue.request``). A delete calls
    it before it archives the registration, so a stop between the two leaves an item and no mark: the sweep then
    finds the scenario whole and ends the item."""
    queue.request(DELETES_QUEUE, scenario, DELETE_KIND, f"scenario:{scenario}", {})


class DeletedScenarioCleanup:
    """Removes what control owns of each deleted scenario, and asks the training service to remove what it owns.

    Values: ``queue`` holds the cleanup items and the scenario's jobs; ``registry`` holds the scenario's lock, its
    storage and its registration; ``backends`` opens the scenario's artifact store; ``publisher`` writes the served
    models when this process holds the control lease, else it is None; ``job_components`` are control's views of
    the training components (``JobQueueComponent``); ``record_retention_lock`` keeps record retention from running
    while the storage moves; ``worker_id`` is a non-empty name of this cleanup in the queue.

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
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id is a non-empty string")
        self.queue = queue
        self.registry = registry
        self.backends = backends
        self.publisher = publisher
        self.job_components = tuple(job_components)
        self.record_retention_lock = record_retention_lock
        self.worker_id = worker_id
        #: Held for one whole ``clean_up``, so two calls at once run one after the other.
        self.lock = threading.Lock()

    def clean_up(self) -> int:
        """Queue each failed cleanup item again, then clean up the scenario of each queued item; return how many
        scenarios were cleaned up in full. A scenario whose step failed keeps its item, failed, for the next call."""
        cleaned = 0
        with self.lock:
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
          drops its backend of the scenario and its work state (``RepositoryBackendFactory.forget_scenario``).
        - The storage archives the scenario's records, commits, tasks and settings (``ScenarioStorage.archive``).
        - Each training service is asked to move aside what its component keeps for the scenario
          (``JobQueueComponent.archive_scenario``); the request waits while no training service runs.
        - The samples service is asked, by a selection of each trainer that had one, to drop its instance and its
          backend of the scenario (``LocalSampleProducers``).

        Return ``{"deleted": true, "archived": [...]}`` with the archive locations of the storage.
        """
        with self.record_retention_lock, self.registry.lock_for(scenario):
            if not self.registry.was_deleted(scenario):
                return {"deleted": False}
            if self.publisher is not None:
                self.publisher.withdraw(scenario)
            cancel_scenario_jobs(self.queue, scenario)
            self.backends(scenario).discard_all_content()
            self.backends.forget_scenario(scenario)
            archived: list[JsonValue] = list(self.registry.archive_store(scenario))
            for component in self.job_components:
                component.archive_scenario(scenario)
            selections = self.queue.items(SAMPLES_QUEUE, scenario=scenario)
            for component_name in sorted({str(selection.payload["component"]) for selection in selections}):
                request_selection(self.queue, scenario, component_name, trigger="delete")
        self.registry.forget_lock(scenario)
        return {"deleted": True, "archived": archived}


__all__ = ["DELETES_QUEUE", "DELETE_KIND", "DeletedScenarioCleanup", "request_cleanup"]
