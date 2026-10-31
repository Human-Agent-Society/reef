"""The data service: the work on a deployment's data store that runs apart from control (RFC #1028).

The data service needs no control lease: it writes the data store, the marks of its record copies and the items of
the ``data`` queue, and no table that control alone writes. A ``DataWorker`` takes the ``data`` queue items of the
data service and runs each with the ``DataJob`` of its kind; the ``data_sync`` items are control's
(``reef.dispatcher.data_sync_worker``).

- ``record_copy``: copy the new records of one scenario to record items (``copy_scenario_records``).
- ``artifact_copy``: make the artifact items of one scenario's artifacts (``copy_scenario_artifacts``).
- ``data_import`` and ``data_export``: the transfers of ``reef.service.data_transfer``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.execution.workers import QueueWorker
from reef.service.data import MAX_ITEMS_PER_CREATE, DataAccess, NoDataStore
from reef.service.data_transfer import ArchiveWriters, run_export, run_import
from reef.storage.artifact_items import artifact_item
from reef.storage.data import ItemConflict, ItemNotFound, NewItem
from reef.storage.data_queue import (
    ARTIFACT_COPY_KIND,
    DATA_EXPORT_KIND,
    DATA_IMPORT_KIND,
    DATA_QUEUE,
    RECORD_COPY_KIND,
)
from reef.storage.record_items import copy_scenario_records
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue


def copy_scenario_artifacts(dispatcher: Dispatcher, scenario: str) -> int:
    """Store the artifact item of each artifact that ``scenario`` lists and that its owner does not store yet
    (``reef.storage.artifact_items``); return how many items it stored.

    A scenario without an owner has no store to copy into, and copies nothing. An item that the owner stores already
    stays as it is, so a copy can run again at any time.
    """
    owner = dispatcher.record_storage.read_settings(scenario).owner
    if owner is None:
        return 0
    store = DataAccess.of(dispatcher).require_store()
    with dispatcher.locked_scenario(scenario) as current:
        surface = current.surface
        missing: list[NewItem] = []
        for row in current.artifacts():
            if store.read(owner, f"{scenario}/{row['artifact_id']}") is not None:
                continue
            found = current.release_queries.find_release(row["release_id"])
            uri = current.repository.release_uri(found[0]) if found is not None and found[1] else None
            if uri is not None and not surface.single:
                # A release of several components keeps each one in a directory named after it.
                uri = f"{uri}{row['component']}/"
            # A component that the surface does not declare, such as the records of a trainer that serves nothing,
            # has no kind.
            declared = surface.components.get(row["component"])
            missing.append(artifact_item(scenario, row, None if declared is None else declared.kind, uri))
    for start in range(0, len(missing), MAX_ITEMS_PER_CREATE):
        store.create(owner, missing[start : start + MAX_ITEMS_PER_CREATE])
    return len(missing)


class DataJob(ABC):
    """The work of one kind of ``data`` queue item, which the data service runs.

    Values: ``kind`` is the item kind that it runs. ``run(item)`` does the work of ``item``, an item of ``kind``,
    and returns its result.

    Purpose: ``data_jobs`` makes the data service's jobs; ``DataWorker`` runs each item with the job of its kind.
    """

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def run(self, item: WorkItem) -> Mapping[str, JsonValue]: ...


class RecordCopy(DataJob):
    """Copies the new records of the item's scenario to record items (``copy_scenario_records``): ``{pages}``."""

    kind = RECORD_COPY_KIND

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def run(self, item: WorkItem) -> Mapping[str, JsonValue]:
        dispatcher = self.dispatcher
        pages = copy_scenario_records(
            dispatcher.record_storage,
            DataAccess.of(dispatcher).require_store(),
            dispatcher.coordination.marks,
            item.scenario,
        )
        return {"pages": pages}


class ArtifactCopy(DataJob):
    """Makes the artifact items of the item's scenario (``copy_scenario_artifacts``): ``{items}``."""

    kind = ARTIFACT_COPY_KIND

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def run(self, item: WorkItem) -> Mapping[str, JsonValue]:
        return {"items": copy_scenario_artifacts(self.dispatcher, item.scenario)}


class DataImport(DataJob):
    """Imports the files at the item's URI, which ``sources`` reads (``run_import``)."""

    kind = DATA_IMPORT_KIND

    def __init__(self, dispatcher: Dispatcher, sources: ArtifactSources) -> None:
        self.dispatcher = dispatcher
        self.sources = sources

    def run(self, item: WorkItem) -> Mapping[str, JsonValue]:
        return run_import(DataAccess.of(self.dispatcher), self.sources, item.payload)


class DataExport(DataJob):
    """Writes the archive of the item's export with ``writers`` (``run_export``)."""

    kind = DATA_EXPORT_KIND

    def __init__(self, dispatcher: Dispatcher, writers: ArchiveWriters) -> None:
        self.dispatcher = dispatcher
        self.writers = writers

    def run(self, item: WorkItem) -> Mapping[str, JsonValue]:
        return run_export(DataAccess.of(self.dispatcher), self.writers, item.payload)


def data_jobs(dispatcher: Dispatcher, sources: ArtifactSources, writers: ArchiveWriters) -> tuple[DataJob, ...]:
    """The jobs of the data service over ``dispatcher``'s data store: record and artifact copies, imports that
    ``sources`` read, and exports that ``writers`` write."""
    return (
        RecordCopy(dispatcher),
        ArtifactCopy(dispatcher),
        DataImport(dispatcher, sources),
        DataExport(dispatcher, writers),
    )


class DataWorker(QueueWorker):
    """One worker of the data service.

    Values: as ``QueueWorker``, on the ``data`` queue of ``queue``; ``jobs`` maps each kind that it takes to the
    job that runs it, one job per kind. A missing item, a conflicting item and a deployment without a data store
    fail an item for good, as a bad payload does.

    Purpose: the process that runs the data service starts one or more.
    """

    permanent_errors = (ValueError, ItemNotFound, ItemConflict, NoDataStore)

    def __init__(
        self, queue: WorkQueue, jobs: Sequence[DataJob], *, worker_id: str, poll_seconds: float = 0.2
    ) -> None:
        self.jobs = {job.kind: job for job in jobs}
        if len(self.jobs) != len(jobs):
            raise ValueError("a data worker takes one job per kind")
        super().__init__(queue, DATA_QUEUE, list(self.jobs), worker_id=worker_id, poll_seconds=poll_seconds)

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        return self.jobs[item.kind].run(item)


__all__ = [
    "ArtifactCopy",
    "DataExport",
    "DataImport",
    "DataJob",
    "DataWorker",
    "RecordCopy",
    "copy_scenario_artifacts",
    "data_jobs",
]
