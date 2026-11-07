"""Record items: the inferences and reports of a scenario as items of its owner (RFC #1028).

Let ``s`` be a scenario, ``o`` its owner, and ``r`` an inference or report in the record store of ``s``. Its record
item ``record_item(r)`` is the item of ``o`` with:

* the ID ``<s>/<r.agent_record_id>``, unique within ``o`` because a scenario name has no ``/`` and a record ID is
  unique within its scenario;
* the content ``{agent_record_id, request_type, payload, created_at, references, artifact_id}`` of ``r``;
* the metadata ``kind: record``, ``scenarios: {s: train}`` and ``origin: {scenario: s, artifact_id}``,
  where ``artifact_id`` names the artifact that answered it, or null when ``r`` names none.

The records of ``s`` are its record items (RFC #1219). ``ItemRecordStorage`` gives each session of ``s`` an
``ItemRecordStore``, which appends each record as its record item and reads the record items that ``s`` made, in item
sequence order. A record store from before this keeps its records. The first open of ``s`` in a process copies them
(``copy_records``): the mark ``record_copy:<s>`` keeps how far the records of ``s`` are copied, and each record of ``s``
with a sequence at most the mark that ``record_item`` gives an item has that item in the store of ``o``.

Another scenario ``t`` of ``o`` uses a record item ``i`` whose ``scenarios`` names ``t``. A trainer of ``t`` reads it from
the data store as ``copied_record(i, t)``, with the record ID ``i.item_id``, and each reference ``x`` of ``i`` becomes
``<origin>/x`` (``used_records``). Nothing is stored in ``t``, so when ``i`` leaves ``t``, the read stops
giving it (RFC #1219). A record store from before this can still hold such a copy. It stays, it has no record item of
its own, and a read skips the item.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, Self

from reef.core.plugins import PluginConfigError
from reef.core.records_types import AgentRecord, RecordLoss, RequestType
from reef.storage.commits import CommitRecord, CommitSnapshots
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, ItemConflict, NewItem, StoredItem, item_digest_matches
from reef.storage.evaluations import EvaluationStore
from reef.storage.item_kinds import copied_record, record_digest
from reef.storage.marks import Marks
from reef.storage.records import AppendResult, RecordConflict, RecordStore, StoredRecord
from reef.storage.samples import SampleStore, TrainingState
from reef.storage.scenario import ScenarioSettings, ScenarioStorage, ScenarioStore, TaskStorage
from reef.storage.tasks import TaskFileStore

#: The most records that one ``copy_records`` call reads.
RECORD_PAGE_SIZE = 256


def record_content(record: AgentRecord) -> dict[str, Any]:
    """The content of the record item of ``record``."""
    return {
        "agent_record_id": record.agent_record_id,
        "request_type": record.request_type.value,
        "payload": dict(record.payload),
        "created_at": record.created_at,
        "references": list(record.references),
        "artifact_id": record.artifact_id,
    }


def record_item(record: AgentRecord) -> NewItem:
    """The record item of ``record``, an inference or a report, as the module docstring defines it."""
    artifact_id = record.artifact_id
    return NewItem(
        f"{record.scenario}/{record.agent_record_id}",
        record_content(record),
        {
            "kind": "record",
            "scenarios": {record.scenario: "train"},
            "origin": {
                "scenario": record.scenario,
                "artifact_id": artifact_id if isinstance(artifact_id, str) and artifact_id else None,
            },
        },
    )


def origin_of(item: StoredItem) -> str:
    """The scenario that made the record item ``item``."""
    return item.metadata["origin"]["scenario"]


def is_record_copy(data: DataStore, owner: str, scenario: str, record: AgentRecord) -> bool:
    """Whether ``record`` of ``scenario`` is the copy of a record item of ``owner`` that another scenario made."""
    item = data.read(owner, record.agent_record_id)
    return item is not None and item.kind == "record" and origin_of(item) != scenario


def copy_records(records: RecordStore, data: DataStore, owner: str, scenario: str, after_sequence: int) -> int:
    """Create the record items of the next page of records of ``scenario`` after ``after_sequence``.

    The page is at most ``RECORD_PAGE_SIZE`` records of ``records`` with a sequence greater than
    ``after_sequence``; its training requests and its copies of other scenarios' record items get no item.
    Return the sequence of the page's last record, or ``after_sequence`` for an empty page. A record item that
    ``owner`` stores already changes nothing, so a copy that runs twice is safe. Raise ``ItemConflict``, and
    create nothing, when ``owner`` stores one of the IDs with other content.
    """
    page = records.replay_page(scenario, after_sequence=after_sequence, limit=RECORD_PAGE_SIZE)
    if not page:
        return after_sequence
    data.create(
        owner,
        [record_item(record) for _, record in page if not is_record_copy(data, owner, scenario, record)],
    )
    return page[-1][0]


def record_copy_mark(scenario: str) -> str:
    """The name of the mark that keeps how far the records of ``scenario`` are copied."""
    return f"record_copy:{scenario}"


def copy_scenario_records(records: RecordStore, data: DataStore, marks: Marks, owner: str, scenario: str) -> int:
    """Copy the records of ``scenario`` in ``records`` after its mark to items of ``owner``, page by page, and advance
    the mark after each page. Return the number of pages copied. A crash between a page and its mark copies the page
    again, which changes nothing."""
    name = record_copy_mark(scenario)
    after = marks.read(name)
    pages = 0
    while (sequence := copy_records(records, data, owner, scenario, after)) != after:
        marks.advance(name, sequence)
        after = sequence
        pages += 1
    return pages


def used_records(
    data: DataStore, owner: str, scenario: str, *, after_sequence: int = 0, limit: int = RECORD_PAGE_SIZE
) -> tuple[int, tuple[AgentRecord, ...]]:
    """The next record items of ``owner`` that another scenario made and that ``scenario`` uses, read as records of
    ``scenario`` (``copied_record``), with the cursor of the next call. The cursor is a use sequence. It moves past the
    scenario's own record items too, so a later call does not read them again."""
    newest = data.last_use_sequence(owner, scenario)
    uses = data.scenario_items(
        owner, scenario, kinds=("record",), after=after_sequence, limit=limit, excluding_origin=scenario
    )
    records = tuple(read_as(item, scenario) for _, item in uses)
    if len(uses) == limit:
        return uses[-1][0], records
    return max(newest, uses[-1][0] if uses else after_sequence), records


def used_record(data: DataStore, owner: str, scenario: str, record_id: str) -> AgentRecord | None:
    """The record ``record_id`` when it names a record item of another scenario that ``scenario`` uses, else None."""
    item = data.read(owner, record_id)
    if item is None or item.kind != "record" or scenario not in item.scenarios or origin_of(item) == scenario:
        return None
    return read_as(item, scenario)


def read_as(item: StoredItem, scenario: str) -> AgentRecord:
    if item.content is None:
        raise ValueError(f"record item {item.item_id!r} was read without its content")
    return copied_record(item.item_id, item.content, origin_of(item), scenario)


class ItemRecordStore(RecordStore):
    """The records of one scenario as its record items in the data store (RFC #1219).

    Values: ``data`` is the data store, ``owner`` the scenario's owner, ``scenario`` the scenario and ``generation``
    its generation (``DataStore.scenario_generation``) when the session opened. A record of the scenario is its
    record item ``record_item(r)``. The store reads the record items that the scenario made and still uses, as the
    records that it stored. An append sequence is the item's sequence. A
    read of another scenario gives nothing.

    The store differs from a record store in two ways. A retry of a record that ``prune_records`` deleted stores it
    again, and ``trajectory`` gives the record alone.

    Purpose: ``ItemRecordStorage`` gives one to each session.
    """

    def __init__(self, data: DataStore, owner: str, scenario: str, generation: str) -> None:
        self.data = data
        self.owner = owner
        self.scenario = scenario
        self.generation = generation

    def record(self, item: StoredItem) -> AgentRecord:
        """The record of the record item ``item`` that this scenario made: the record that ``record_item`` stored,
        with its own ID and references. ``copied_record`` checks the content."""
        if item.content is None:
            raise ValueError(f"record item {item.item_id!r} was read without its content")
        checked = copied_record(item.item_id, item.content, self.scenario, self.scenario)
        content = item.content
        return replace(
            checked,
            agent_record_id=content["agent_record_id"],
            payload=dict(content["payload"]),
            references=tuple(content["references"]),
        )

    def stored(self, scenario: str, agent_record_id: str) -> StoredItem | None:
        """The record item of ``agent_record_id`` that ``scenario`` made and uses, or None."""
        if scenario != self.scenario:
            return None
        item = self.data.read(self.owner, f"{scenario}/{agent_record_id}")
        if item is None or item.kind != "record" or scenario not in item.scenarios:
            return None
        return item

    def page(
        self, scenario: str, after_sequence: int, limit: int, request_type: RequestType | None = None
    ) -> tuple[StoredItem, ...]:
        """The record items of ``scenario`` after ``after_sequence``, at most ``limit``, in sequence order."""
        if scenario != self.scenario:
            return ()
        return self.data.query(
            self.owner,
            kind="record",
            scenario=scenario,
            origin=scenario,
            request_type=None if request_type is None else request_type.value,
            after=after_sequence,
            limit=min(limit, MAX_QUERY_LIMIT),
        )

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def append_result(self, item: AgentRecord) -> AppendResult:
        return self.append_many([item])[0]

    def append_many(self, items: Sequence[AgentRecord]) -> tuple[AppendResult, ...]:
        if any(item.scenario != self.scenario for item in items):
            raise ValueError(f"the records of this store belong to scenario {self.scenario!r}")
        # A batch can give one record twice: the later one is a retry of the first.
        first: dict[str, AgentRecord] = {}
        for item in items:
            earlier = first.setdefault(item.agent_record_id, item)
            if record_digest(record_content(earlier)) != record_digest(record_content(item)):
                raise RecordConflict(f"record {item.agent_record_id!r} is given twice with other content")
        try:
            created = self.data.create(
                self.owner, [record_item(item) for item in first.values()], generation=self.generation
            )
        except ItemConflict as error:
            raise RecordConflict(str(error)) from error
        stored = dict(zip(first, created, strict=True))
        retried = {
            item.item_id: self.record(item)
            for item in self.data.read_many(self.owner, [result.item_id for result in created if not result.inserted])
        }
        results: list[AppendResult] = []
        for item in items:
            result = stored.pop(item.agent_record_id, None)
            if result is not None and result.inserted:
                results.append(AppendResult(item, True))
            else:
                receipt = retried.get(f"{self.scenario}/{item.agent_record_id}", first[item.agent_record_id])
                results.append(AppendResult(receipt, False))
        return tuple(results)

    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        stored = self.stored(item.scenario, item.agent_record_id)
        if stored is None:
            return None
        if not item_digest_matches(stored, record_digest(record_content(item))):
            raise RecordConflict(f"record {item.agent_record_id!r} is stored with other content")
        return self.record(stored)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        stored = self.stored(scenario, agent_record_id)
        return None if stored is None else self.record(stored)

    def replay(self, scenario: str, *, offset: int = 0, limit: int | None = None) -> tuple[AgentRecord, ...]:
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be non-negative")
        found: list[AgentRecord] = []
        after = 0
        while limit is None or len(found) < offset + limit:
            page = self.page(scenario, after, MAX_QUERY_LIMIT)
            if not page:
                break
            found.extend(self.record(item) for item in page)
            after = page[-1].sequence
        return tuple(found[offset:] if limit is None else found[offset : offset + limit])

    def replay_page(
        self, scenario: str, *, after_sequence: int = 0, limit: int = 256
    ) -> tuple[tuple[int, AgentRecord], ...]:
        return tuple((item.sequence, self.record(item)) for item in self.page(scenario, after_sequence, limit))

    def count(self, scenario: str, *, request_type: RequestType | None = None, after_sequence: int = 0) -> int:
        if scenario != self.scenario:
            return 0
        return self.data.count(
            self.owner,
            kind="record",
            scenario=scenario,
            origin=scenario,
            request_type=None if request_type is None else request_type.value,
            after=after_sequence,
        )

    def last_sequence(self, scenario: str) -> int:
        if scenario != self.scenario:
            return 0
        return self.data.last_sequence(self.owner, kind="record", scenario=scenario, origin=scenario)

    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        stored = self.stored(scenario, agent_record_id)
        return None if stored is None else StoredRecord(stored.sequence, self.record(stored))

    def audit_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        request_type: RequestType | None = None,
    ) -> tuple[StoredRecord, ...]:
        return tuple(
            StoredRecord(item.sequence, self.record(item))
            for item in self.page(scenario, after_sequence, limit, request_type)
        )

    def loss(self, scenario: str) -> RecordLoss:
        return RecordLoss() if scenario != self.scenario else self.data.record_loss(self.owner, scenario)

    def close(self) -> None:
        """The data store outlives the session, so closing changes nothing."""


class ItemRecordsScenarioStore(ScenarioStore):
    """A scenario session whose records are its record items (``ItemRecordStore``); the rest is ``inner``'s."""

    def __init__(self, inner: ScenarioStore, records: ItemRecordStore) -> None:
        self._inner = inner
        self._records = records

    @property
    def records(self) -> RecordStore:
        return self._records

    @property
    def durable(self) -> bool:
        return self._inner.durable

    @property
    def tasks(self) -> TaskStorage:
        return self._inner.tasks

    @property
    def training_state(self) -> TrainingState:
        return self._inner.training_state

    @property
    def evaluations(self) -> EvaluationStore:
        return self._inner.evaluations

    def use_snapshots(self, snapshots: CommitSnapshots) -> None:
        self._inner.use_snapshots(snapshots)

    def history(self) -> tuple[CommitRecord, ...]:
        return self._inner.history()

    def training_run_position(self) -> tuple[int, int]:
        return self._inner.training_run_position()

    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        return self._inner.commit_step(expected_step=expected_step, commit=commit)

    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        return self._inner.recover(checkpoint=checkpoint)

    def close(self) -> None:
        self._inner.close()


class ItemRecordStorage(ScenarioStorage):
    """Storage whose sessions keep the records of their scenario as record items (RFC #1219).

    Values: ``inner`` keeps everything else, and the records from before this storage. ``data`` is the data store,
    and ``marks`` keeps how far each scenario's records in ``inner`` are copied. The first ``open`` of a scenario in
    this process copies its records in ``inner`` after the mark (``copy_scenario_records``). A scenario without an
    owner has the owner with its own name, as ``DataScenarioStorage`` gives it.

    Purpose: deployment assembly wraps the scenario storage of each service in it, so control, the data service and
    the inference service append records to the data store.
    """

    def __init__(self, inner: ScenarioStorage, data: DataStore, marks: Marks) -> None:
        self.inner = inner
        self.data = data
        self.marks = marks
        self.copied: set[str] = set()
        self.copy_lock = Lock()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        raise PluginConfigError(f"{cls.__name__} wraps another storage; deployment assembly applies it")

    @property
    def durable(self) -> bool:
        return self.inner.durable

    def read_settings(self, scenario: str) -> ScenarioSettings:
        return self.inner.read_settings(scenario)

    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        self.inner.write_settings(scenario, settings)

    def open(self, scenario: str) -> ScenarioStore:
        session = self.inner.open(scenario)
        try:
            owner = self.inner.read_settings(scenario).owner or scenario
            with self.copy_lock:
                if scenario not in self.copied:
                    copy_scenario_records(session.records, self.data, self.marks, owner, scenario)
                    self.copied.add(scenario)
            records = ItemRecordStore(self.data, owner, scenario, self.data.scenario_generation(owner, scenario))
        except BaseException:
            session.close()
            raise
        return ItemRecordsScenarioStore(session, records)

    def sample_store(self, scenario: str, session: ScenarioStore) -> SampleStore | None:
        return self.inner.sample_store(scenario, session)

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self.copy_lock:
            self.copied.discard(scenario)
        return self.inner.archive(scenario)

    def prune(self, *, max_bytes: int) -> int:
        return self.inner.prune(max_bytes=max_bytes)

    def close(self) -> None:
        self.inner.close()


__all__ = [
    "RECORD_PAGE_SIZE",
    "ItemRecordStorage",
    "ItemRecordStore",
    "ItemRecordsScenarioStore",
    "copy_records",
    "copy_scenario_records",
    "is_record_copy",
    "record_copy_mark",
    "record_item",
    "used_record",
    "used_records",
]
