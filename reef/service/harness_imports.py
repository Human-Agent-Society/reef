"""How ``POST /reef/harness/import`` commits a sync of a person's harness files (RFC #1048, harness supplement).

Control commits a sync itself (``RequestService.harness_import``). The harness service writes no release: it adds a
``harness_sync`` item to control's sync queue, and control's ``HarnessSyncWorker`` commits the sync and finishes the
item with its answer. The harness service waits up to ``SYNC_WAIT_SECONDS`` for the item.

An item's result is ``{"answer": ...}``, the answer of the sync, or ``{"refused": {"kind", "message"}}`` when control
refused the files: ``kind`` is ``conflict`` (HTTP 409), ``not_found`` (404) or ``invalid`` (400).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, cast

from reef.artifact.artifact import ArtifactConflict, ArtifactError, ArtifactNotFound
from reef.core.errors import ReefError, UnknownScenario
from reef.core.records_types import RequestType
from reef.execution.workers import QueueWorker
from reef.service.request_service import RequestService
from reef.service.wire import SCENARIO_HEADER, parse_request_headers
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkQueue

#: The queue of control's harness syncs, and the kind of its items.
HARNESS_SYNC_QUEUE = "harness_sync"
HARNESS_SYNC_KIND = "harness_sync"
#: How long the harness service waits for control to commit an import.
SYNC_WAIT_SECONDS = 30.0
SYNC_POLL_SECONDS = 0.1


class HarnessImportPending(ArtifactError):
    """Control did not commit an import within ``SYNC_WAIT_SECONDS``: no control instance runs, or it is busy.

    The import stays queued, and control commits it when it takes the item; a retry of the same files waits for
    the same item. Purpose: ``QueuedImports`` raises it; the service answers 503.
    """


class HarnessImports(ABC):
    """How a harness app commits ``POST /reef/harness/import``.

    Values: ``import_files(headers, payload)`` commits the sync of the person's files in ``payload`` for the
    scenario that ``headers`` name and returns its answer (``RequestService.harness_import``), or raises the error
    of a refused import: ``ArtifactConflict``, ``ArtifactNotFound`` or ``ValueError``.

    Purpose: control's app uses ``DirectImports``; the harness service's app uses ``QueuedImports``.
    """

    @abstractmethod
    def import_files(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]: ...


class DirectImports(HarnessImports):
    """Commits each import in this process, which runs control."""

    def __init__(self, request_service: RequestService) -> None:
        self.request_service = request_service

    def import_files(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.request_service.harness_import(headers, payload)


class QueuedImports(HarnessImports):
    """Adds each import to control's sync queue and waits for control's answer.

    Values: ``queue`` is the deployment's work queue. An item's ID is a hash of its scenario and payload, with
    ``:<n>`` after it for the ``n``-th item of the same import: a retry of an import whose item is not final waits
    for that item, and an import whose earlier items all ended gets a new one. ``HarnessImportPending`` when
    control does not answer within ``SYNC_WAIT_SECONDS``.
    """

    def __init__(self, queue: WorkQueue) -> None:
        self.queue = queue

    def import_files(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]:
        # Control checks the scenario and the files when it takes the item.
        scenario = parse_request_headers(headers, RequestType.INFERENCE).scenario
        work = dict(payload)
        encoded = json.dumps({"scenario": scenario, "payload": work}, sort_keys=True, separators=(",", ":"))
        base_id = f"{HARNESS_SYNC_KIND}:{hashlib.sha256(encoded.encode()).hexdigest()}"
        item_id, number = base_id, 1
        # An item that ended answered an earlier import: the active tree may have moved since, so it is not reused.
        while (earlier := self.queue.get(HARNESS_SYNC_QUEUE, item_id)) is not None and earlier.final:
            number += 1
            item_id = f"{base_id}:{number}"
        self.queue.add(NewWork(HARNESS_SYNC_QUEUE, item_id, scenario, HARNESS_SYNC_KIND, work))
        deadline = time.monotonic() + SYNC_WAIT_SECONDS
        while True:
            item = self.queue.get(HARNESS_SYNC_QUEUE, item_id)
            if item is not None and item.final:
                return import_answer(item)
            if time.monotonic() >= deadline:
                raise HarnessImportPending(
                    f"control did not commit the import of scenario {scenario!r} within {SYNC_WAIT_SECONDS:.0f} s; "
                    "send it again to wait for it"
                )
            time.sleep(SYNC_POLL_SECONDS)


def import_answer(item: WorkItem) -> dict[str, Any]:
    """The answer of the final ``harness_sync`` item ``item``, or the error of the import that control refused.

    ``ArtifactError`` (HTTP 503) for an item that failed: control could not run it.
    """
    if item.state == "failed" or item.result is None:
        raise ArtifactError(f"control could not commit the import: {item.error}")
    refused = item.result.get("refused")
    if isinstance(refused, Mapping):
        message = str(refused.get("message"))
        kind = refused.get("kind")
        if kind == "conflict":
            raise ArtifactConflict(message)
        if kind == "not_found":
            raise ArtifactNotFound(message)
        raise ValueError(message)
    answer = item.result.get("answer")
    if not isinstance(answer, Mapping):
        raise ArtifactError("control's import result holds no answer")
    return dict(answer)


class HarnessSyncWorker(QueueWorker):
    """Control's worker of the ``harness_sync`` items: it commits each import with ``RequestService``.

    Values: as ``QueueWorker``, on ``HARNESS_SYNC_QUEUE``; ``request_service`` runs control. A refused import
    finishes its item with ``{"refused": ...}``, so the harness service answers with its error; an artifact store
    failure fails the attempt, which a later attempt retries.

    Purpose: the assembly of a process that runs control starts one.
    """

    def __init__(
        self, queue: WorkQueue, request_service: RequestService, *, worker_id: str, poll_seconds: float = 0.2
    ) -> None:
        super().__init__(
            queue, HARNESS_SYNC_QUEUE, [HARNESS_SYNC_KIND], worker_id=worker_id, poll_seconds=poll_seconds
        )
        self.request_service = request_service

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        headers = {SCENARIO_HEADER: item.scenario}
        try:
            answer = self.request_service.harness_import(headers, item.payload)
        except ArtifactConflict as exc:
            return {"refused": {"kind": "conflict", "message": str(exc)}}
        except (ArtifactNotFound, UnknownScenario) as exc:
            return {"refused": {"kind": "not_found", "message": str(exc)}}
        except ArtifactError:
            # The artifact store failed: a retry of the same import can pass.
            raise
        except (ReefError, ValueError) as exc:
            # The files or the scenario refuse the import, such as a tree that the trainer cannot start from.
            return {"refused": {"kind": "invalid", "message": str(exc)}}
        return {"answer": cast(JsonValue, answer)}


__all__ = [
    "HARNESS_SYNC_KIND",
    "HARNESS_SYNC_QUEUE",
    "SYNC_WAIT_SECONDS",
    "DirectImports",
    "HarnessImportPending",
    "HarnessImports",
    "HarnessSyncWorker",
    "QueuedImports",
    "import_answer",
]
