"""The data service: the work on a deployment's data store that runs apart from control (RFC #1028).

The data service needs no control lease: it writes the data store, the marks of its record copies and the items of
the ``data`` queue, and no table that control alone writes. A ``DataWorker`` takes the ``data`` queue items of the
data service; the ``data_sync`` items are control's (``reef.dispatcher.data_sync_worker``).

- ``record_copy``: copy the new records of one scenario to record items (``copy_scenario_records``).
- ``artifact_copy``: make the artifact items of one scenario's artifacts (``copy_scenario_artifacts``).
- ``data_import`` and ``data_export``: the transfers of ``reef.service.data_transfer``.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.execution.workers import QueueWorker
from reef.service.data import MAX_ITEMS_PER_CREATE, data_store
from reef.service.data_transfer import DATA_EXPORT_KIND, DATA_IMPORT_KIND, ArchiveWriters, run_export, run_import
from reef.storage.artifact_items import ARTIFACT_COPY_KIND, artifact_item
from reef.storage.data import DATA_QUEUE, NewItem
from reef.storage.record_items import RECORD_COPY_KIND, copy_scenario_records
from reef.storage.work_queue import JsonValue, WorkItem


def copy_scenario_artifacts(dispatcher: Dispatcher, scenario: str) -> int:
    """Store the artifact item of each artifact that ``scenario`` lists and that its owner does not store yet
    (``reef.storage.artifact_items``); return how many items it stored.

    A scenario without an owner has no store to copy into, and copies nothing. An item that the owner stores already
    stays as it is, so a copy can run again at any time.
    """
    owner = dispatcher.record_storage.read_settings(scenario).owner
    if owner is None:
        return 0
    store = data_store(dispatcher)
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


class DataWorker(QueueWorker):
    """One worker of the data service.

    Values: as ``QueueWorker``, on the queue of ``dispatcher``; ``dispatcher`` runs the data service and holds the
    data store, the record storage and the marks; ``sources`` reads the URI of an import, and ``writers`` writes the
    archive of an export.

    Purpose: the process that runs the data service starts one or more.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        sources: ArtifactSources,
        writers: ArchiveWriters,
        *,
        worker_id: str,
        poll_seconds: float = 0.2,
    ) -> None:
        super().__init__(
            dispatcher.queue,
            DATA_QUEUE,
            [RECORD_COPY_KIND, ARTIFACT_COPY_KIND, DATA_IMPORT_KIND, DATA_EXPORT_KIND],
            worker_id=worker_id,
            poll_seconds=poll_seconds,
        )
        self.dispatcher = dispatcher
        self.sources = sources
        self.writers = writers

    def run_item(self, item: WorkItem) -> Mapping[str, JsonValue]:
        if item.kind == DATA_IMPORT_KIND:
            return run_import(self.dispatcher, self.sources, item.payload)
        if item.kind == DATA_EXPORT_KIND:
            return run_export(self.dispatcher, self.writers, item.payload)
        if item.kind == ARTIFACT_COPY_KIND:
            return {"items": copy_scenario_artifacts(self.dispatcher, item.scenario)}
        pages = copy_scenario_records(
            self.dispatcher.record_storage,
            data_store(self.dispatcher),
            self.dispatcher.coordination.marks,
            item.scenario,
        )
        return {"pages": pages}


__all__ = ["DataWorker", "copy_scenario_artifacts"]
