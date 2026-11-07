"""Record stores alone, for tests: the record items of one scenario in a data store."""

from __future__ import annotations

from reef.storage.data import DataStore, MemoryDataStore
from reef.storage.record_items import ItemRecordStore
from reef.storage.tasks import LocalTaskFileStore


def item_records(data: DataStore, scenario: str = "math", owner: str | None = None) -> ItemRecordStore:
    """The ``ItemRecordStore`` of ``scenario`` in ``data``; ``owner`` defaults to the scenario."""
    owner = owner or scenario
    return ItemRecordStore(data, owner, scenario, data.scenario_generation(owner, scenario))


def memory_records(scenario: str = "math", owner: str | None = None) -> ItemRecordStore:
    """The ``ItemRecordStore`` of ``scenario`` in a new ``MemoryDataStore``."""
    return item_records(MemoryDataStore(LocalTaskFileStore(None)), scenario, owner)
