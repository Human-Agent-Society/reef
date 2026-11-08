"""The data sync: a data write tells each scenario that its items name (RFC #1028, RFC #1140).

A scenario's trainers read its training samples, and the record items of other scenarios that it uses, from the data
store (``reef.storage.samples``), so a write needs no copy of them. Control still runs a sync of each scenario that a
write touched: it wakes the scenario's trainers and its samples service, which read the new and changed items. The data service writes the data store;
after each write it asks control for the sync with a ``data_sync`` item in control's sync queue
(``request_data_sync``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from reef.storage.data_queue import DATA_SYNC_KIND, DATA_SYNC_QUEUE
from reef.storage.work_queue import WorkItem, WorkQueue


class ScenarioUses(ABC):
    """What a data write asks of the scenarios that its items name.

    ``scenario_owner(scenario)`` is the owner of ``scenario`` in the data store, or None when the scenario does not
    exist. ``check_use(scenario, trajectories, train_tasks)`` raises ``ValueError`` unless ``scenario`` can train
    on new trajectories when ``trajectories``, and on train tasks when ``train_tasks``.

    Purpose: the dispatcher implements it; the data writes (``reef.service.data``) call it before they store.
    """

    @abstractmethod
    def scenario_owner(self, scenario: str) -> str | None: ...

    @abstractmethod
    def check_use(self, scenario: str, trajectories: bool, train_tasks: bool) -> None: ...


def request_data_sync(queue: WorkQueue, owner: str, scenario: str) -> WorkItem:
    """Ask control to sync ``scenario`` with the items of ``owner``; return the item that will run the sync.

    A sync reads the items when it runs, so a queued sync covers every write made while it waits.
    """
    return queue.request(DATA_SYNC_QUEUE, scenario, DATA_SYNC_KIND, data_sync_key(scenario), {"owner": owner})


def data_sync_key(scenario: str) -> str:
    """The serial key of the ``data_sync`` items of ``scenario``: its syncs run one at a time."""
    return f"{DATA_SYNC_KIND}:{scenario}"


__all__ = ["ScenarioUses", "data_sync_key", "request_data_sync"]
