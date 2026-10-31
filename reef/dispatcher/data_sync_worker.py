"""Control's worker of the ``data_sync`` items: it syncs a loaded scenario with the items of its owner (RFC #1028).

The data service writes the data store and asks for a sync of each scenario that a write touched
(``reef.storage.data_sync.request_data_sync``). The sync runs in control, because the trainers of a loaded scenario
must forget the uploads that leave it, and its workers wake for new ones. A scenario that is not loaded syncs when
it opens, so the item does nothing for it.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from reef.core.values import require_text
from reef.execution.workers import QueueWorker
from reef.storage.data_queue import DATA_SYNC_KIND, DATA_SYNC_QUEUE
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue


class ItemSync(ABC):
    """Syncs loaded scenarios with the items of their owner.

    Purpose: the dispatcher implements it; ``DataSyncWorker`` calls it for each ``data_sync`` item.
    """

    @abstractmethod
    def sync_items(self, owner: str, scenarios: Iterable[str]) -> None:
        """Sync each loaded scenario of ``owner`` among ``scenarios`` with that owner's items."""


class DataSyncWorker(QueueWorker):
    """Takes the ``data_sync`` items of control's sync queue and runs each sync with ``target``.

    Values: as ``QueueWorker``; ``target`` runs the syncs. Purpose: control runs one.
    """

    def __init__(self, queue: WorkQueue, target: ItemSync, *, worker_id: str, poll_seconds: float = 0.2) -> None:
        super().__init__(queue, DATA_SYNC_QUEUE, [DATA_SYNC_KIND], worker_id=worker_id, poll_seconds=poll_seconds)
        self.target = target

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        owner = require_text(item.payload.get("owner"), "owner")
        self.target.sync_items(owner, [item.scenario])
        return {}


__all__ = ["DataSyncWorker", "ItemSync"]
