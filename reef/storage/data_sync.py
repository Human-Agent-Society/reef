"""The data sync: a data write tells each scenario that its items name (RFC #1028, RFC #1140).

A scenario's trainers read its training samples from the data store (``reef.storage.samples``), so a write needs no
copy of them. Control still runs a sync of each scenario that a write touched: it appends to the scenario's record
store the record items of another scenario that it now uses (``reef.storage.record_items.sync_scenario_records``),
and it wakes the scenario's trainers, which read the new and changed items. The data service writes the data store;
after each write it asks control for the sync with a ``data_sync`` item in control's sync queue
(``request_data_sync``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from reef.core.job_data import TrajectorySample
from reef.storage.data import DataStore
from reef.storage.data_queue import DATA_SYNC_KIND, DATA_SYNC_QUEUE
from reef.storage.work_queue import WorkItem, WorkQueue


class ScenarioUses(ABC):
    """What a data write asks of the scenarios that its items name.

    ``scenario_owner(scenario)`` is the owner of ``scenario`` in the data store, or None when the scenario does not
    exist. ``check_use(scenario, trajectories, train_tasks)`` raises ``ValueError`` unless ``scenario`` can train
    on the new ``trajectories`` (pairs of item ID and sample), and on train tasks when ``train_tasks``.

    Purpose: the dispatcher implements it; the data writes (``reef.service.data``) call it before they store.
    """

    @abstractmethod
    def scenario_owner(self, scenario: str) -> str | None: ...

    @abstractmethod
    def check_use(
        self, scenario: str, trajectories: Sequence[tuple[str, TrajectorySample]], train_tasks: bool
    ) -> None: ...


def release_items(data: DataStore, owner: str, scenario: str) -> None:
    """Remove ``scenario`` from the ``scenarios`` of each item of ``owner`` that it uses, page by page.

    The record items that ``scenario`` made keep it: it is their origin. The archive of a scenario calls this, so
    that its items stay and a later scenario with the same name does not use them.
    """
    after = 0
    while page := data.query(owner, scenario=scenario, after=after):
        after = page[-1].sequence
        ids = [
            item.item_id for item in page if item.kind != "record" or item.metadata["origin"]["scenario"] != scenario
        ]
        if ids:
            data.update(owner, ids, {"scenarios": {scenario: None}})


def request_data_sync(queue: WorkQueue, owner: str, scenario: str) -> WorkItem:
    """Ask control to sync ``scenario`` with the items of ``owner``; return the item that will run the sync.

    A sync reads the items when it runs, so a queued sync covers every write made while it waits.
    """
    return queue.request(DATA_SYNC_QUEUE, scenario, DATA_SYNC_KIND, data_sync_key(scenario), {"owner": owner})


def data_sync_key(scenario: str) -> str:
    """The serial key of the ``data_sync`` items of ``scenario``: its syncs run one at a time."""
    return f"{DATA_SYNC_KIND}:{scenario}"


__all__ = ["ScenarioUses", "data_sync_key", "release_items", "request_data_sync"]
