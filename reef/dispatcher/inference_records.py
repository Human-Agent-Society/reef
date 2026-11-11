"""The inference records that a serving worker gives its runtime, read from the deployment's stores."""

from __future__ import annotations

from reef.core.records_types import AgentRecord, RequestType
from reef.inference.interfaces import InferenceRecords
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, StoredItem
from reef.storage.record_items import ItemRecordStore
from reef.storage.scenario import ScenarioStorage


class StoredInferenceRecords(InferenceRecords):
    """The inference records of the scenarios of ``storage``, which are record items in ``data`` (RFC #1219).

    Values: ``storage`` gives each scenario's owner, and ``data`` is the data store that holds the owner's items.
    The reader owns neither: the service that opened them closes them.
    """

    def __init__(self, storage: ScenarioStorage, data: DataStore) -> None:
        self.storage = storage
        self.data = data

    def newest(self, scenario: str, count: int) -> tuple[AgentRecord, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        owner = self.storage.read_settings(scenario).owner or scenario
        last = self.data.last_sequence(
            owner, kind="record", scenario=scenario, origin=scenario, request_type=RequestType.INFERENCE.value
        )
        # Other items of the owner lie between the scenario's inference records, so the window of sequences grows
        # until it holds ``count`` of them or reaches the first item. The first reads take no content.
        window = max(count, MAX_QUERY_LIMIT)
        while True:
            start = max(0, last - window)
            sequences: list[int] = []
            after = start
            while page := self.page(owner, scenario, after, MAX_QUERY_LIMIT, content=False):
                sequences.extend(item.sequence for item in page)
                after = page[-1].sequence
            if len(sequences) >= count or start == 0:
                break
            window *= 4
        newest = sequences[-count:]
        if not newest:
            return ()
        # The store of the scenario's own records gives each item as the record that the scenario stored.
        stored = ItemRecordStore(self.data, owner, scenario, self.data.scenario_generation(owner, scenario))
        records: list[AgentRecord] = []
        after = newest[0] - 1
        while len(records) < len(newest):
            page = self.page(owner, scenario, after, len(newest) - len(records), content=True)
            if not page:
                # A record that the retention removed between the two reads.
                break
            records.extend(stored.record(item) for item in page)
            after = page[-1].sequence
        return tuple(reversed(records))

    def page(self, owner: str, scenario: str, after: int, limit: int, *, content: bool) -> tuple[StoredItem, ...]:
        """The inference record items of ``scenario`` after the sequence ``after``, at most ``limit``, in order."""
        return self.data.query(
            owner,
            kind="record",
            scenario=scenario,
            origin=scenario,
            request_type=RequestType.INFERENCE.value,
            after=after,
            limit=min(limit, MAX_QUERY_LIMIT),
            content=content,
        )


__all__ = ["StoredInferenceRecords"]
