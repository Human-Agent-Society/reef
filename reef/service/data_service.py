"""The data service: the work on a deployment's data store that runs apart from control (RFC #1028).

The data service needs no control lease: it writes the data store, the marks of its record copies and the items of
the ``data`` queue, and no table that control alone writes. A ``DataWorker`` takes the ``data`` queue items of the
data service; the ``data_sync`` items are control's (``reef.dispatcher.data_sync_worker``).

- ``record_copy``: copy the new records of one scenario to record items (``copy_scenario_records``).
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.execution.workers import QueueWorker
from reef.storage.data import DATA_QUEUE, DataStore
from reef.storage.marks import Marks
from reef.storage.record_items import RECORD_COPY_KIND, copy_scenario_records
from reef.storage.scenario import ScenarioStorage
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue


class DataWorker(QueueWorker):
    """One worker of the data service.

    Values: as ``QueueWorker``; ``storage`` reads the scenarios' records and settings without opening a scenario,
    ``data`` is the data store, and ``marks`` keeps the record copy positions.

    Purpose: the process that runs the data service starts one or more.
    """

    def __init__(
        self,
        queue: WorkQueue,
        storage: ScenarioStorage,
        data: DataStore,
        marks: Marks,
        *,
        worker_id: str,
        poll_seconds: float = 0.2,
    ) -> None:
        super().__init__(queue, DATA_QUEUE, [RECORD_COPY_KIND], worker_id=worker_id, poll_seconds=poll_seconds)
        self.storage = storage
        self.data = data
        self.marks = marks

    def run_item(self, item: WorkItem) -> Mapping[str, JsonValue]:
        return {"pages": copy_scenario_records(self.storage, self.data, self.marks, item.scenario)}


__all__ = ["DataWorker"]
