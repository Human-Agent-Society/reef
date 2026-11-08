"""Record items: the inferences and reports of a scenario as items of its owner (RFC #1028).

Let ``s`` be a scenario, ``o`` its owner, and ``r`` an inference or report of ``s``. Its record
item ``record_item(r)`` is the item of ``o`` with:

* the ID ``<s>/<r.agent_record_id>``, unique within ``o`` because a scenario name has no ``/`` and a record ID is
  unique within its scenario;
* the content ``{agent_record_id, request_type, payload, created_at, references, artifact_id}`` of ``r``;
* the metadata ``kind: record``, ``scenarios: {s: train}`` and ``origin: {scenario: s, artifact_id}``,
  where ``artifact_id`` names the artifact that answered it, or null when ``r`` names none.

The records of ``s`` are its record items (RFC #1219). Each session of ``s`` has an ``ItemRecordStore``, which
appends each record as its record item and reads the record items that ``s`` made, in item sequence order.

Another scenario ``t`` of ``o`` uses a record item ``i`` whose ``scenarios`` names ``t``. A trainer of ``t`` reads it from
the data store as ``copied_record(i, t)``, with the record ID ``i.item_id``, and each reference ``x`` of ``i`` becomes
``<origin>/x`` (``used_records``). Nothing is stored in ``t``, so when ``i`` leaves ``t``, the read stops
giving it (RFC #1219).

An evaluation of ``s`` keeps each model call of its rollouts as an evaluation call: a record item of ``o`` whose
origin source is ``evaluation`` (``evaluation_call_item``). It is the data of no scenario, so its ``scenarios`` stays
empty, and no record store, trainer or record route of ``s`` reads it. It links only to the calls of the same
evaluation attempt (``reef.storage.data.record_link_scope``), and ``prune_records`` keeps it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from typing import Any, cast

from reef.core.evaluation import EvaluationCall, EvaluationCallSummary
from reef.core.records_types import AgentRecord, RecordLoss, RequestType
from reef.core.trajectory_index import trajectory_calls
from reef.core.values import JsonValue
from reef.storage.data import (
    MAX_QUERY_LIMIT,
    DataStore,
    ItemConflict,
    NewItem,
    StoredItem,
    item_digest_matches,
    record_source,
)
from reef.storage.item_kinds import copied_record, evaluation_call_id, record_digest
from reef.storage.records import AppendResult, RecordConflict, RecordStore, StoredRecord

#: The most record items that one ``used_records`` call reads.
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


def evaluation_call_item(scenario: str, artifact_id: str, evaluation_id: str, call: EvaluationCall) -> NewItem:
    """The record item of ``call``, a call of evaluation ``evaluation_id`` of ``scenario`` to the candidate
    ``artifact_id``.

    Its ID is ``<scenario>/<evaluation_call_id>``. Its content is the record of an inference: the payload is the
    request body with ``response``, the response body or null, and ``created_at`` is the call's start. Its metadata
    is ``kind: record``, empty ``scenarios``, the ``evaluation_id``, and the origin ``{scenario, artifact_id, source:
    evaluation, evaluation}``, where ``evaluation`` holds the ``evaluation_id`` and every field of the call's summary.
    """
    summary = call.summary
    record_id = evaluation_call_id(evaluation_id, summary.task_index, summary.repeat, summary.attempt, summary.index)
    payload = {**(call.request or {}), "response": None if call.response is None else dict(call.response)}
    record = AgentRecord(
        record_id, scenario, RequestType.INFERENCE, payload, summary.started_at, artifact_id=artifact_id
    )
    return NewItem(
        f"{scenario}/{record_id}",
        record_content(record),
        {
            "kind": "record",
            "scenarios": {},
            "evaluation_id": evaluation_id,
            "origin": {
                "scenario": scenario,
                "artifact_id": artifact_id,
                "source": "evaluation",
                "evaluation": {"evaluation_id": evaluation_id, **asdict(summary)},
            },
        },
    )


def evaluation_call_summary(item: StoredItem) -> EvaluationCallSummary:
    """The summary of the evaluation call ``item``, from its metadata alone, so the item can be read without its
    content. ``ValueError`` for an item that is no evaluation call."""
    if item.kind != "record" or record_source(item.metadata) != "evaluation":
        raise ValueError(f"item {item.item_id!r} is no evaluation call")
    location = dict(item.metadata["origin"]["evaluation"])
    location.pop("evaluation_id")
    return EvaluationCallSummary(**location)


def stored_evaluation_call(item: StoredItem) -> EvaluationCall:
    """The ``EvaluationCall`` that ``evaluation_call_item`` stored as ``item``, read with its content."""
    if item.content is None:
        raise ValueError(f"item {item.item_id!r} was read without its content")
    payload = dict(item.content["payload"])
    response = payload.pop("response")
    # The item keeps an absent request as an empty payload, so an empty request reads as None too.
    return EvaluationCall(evaluation_call_summary(item), cast(dict[str, JsonValue], payload) or None, response)


def evaluation_call_items(
    data: DataStore, owner: str, scenario: str, evaluation_id: str, *, content: bool
) -> Iterator[StoredItem]:
    """The evaluation calls of evaluation ``evaluation_id`` of ``scenario``, in sequence order, a page at a time: one
    query of the record items of ``scenario`` with that ``evaluation_id`` for each page."""
    after = 0
    while page := data.query(
        owner,
        kind="record",
        origin=scenario,
        group=("evaluation_id", evaluation_id),
        after=after,
        content=content,
    ):
        yield from (item for item in page if record_source(item.metadata) == "evaluation")
        after = page[-1].sequence


def origin_of(item: StoredItem) -> str:
    """The scenario that made the record item ``item``."""
    return item.metadata["origin"]["scenario"]


def old_record_copied(data: DataStore, owner: str, scenario: str, record_id: str) -> bool:
    """Whether the record ``record_id`` of the old record store of ``scenario`` is in ``data``: as its own record item,
    or, for a copy of another scenario's record item from before RFC #1219, as that item."""
    for item_id in (f"{scenario}/{record_id}", record_id):
        item = data.read(owner, item_id)
        if item is not None and item.kind == "record":
            return True
    return False


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

    A retry of a record that ``prune_records`` deleted stores it again.

    Purpose: ``SQLiteScenarioStorage`` and ``PostgresScenarioStorage`` give one to each session.
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

    def trajectory(self, scenario: str, agent_record_id: str) -> tuple[str, ...]:
        """The record IDs of the trajectory that the record ends, oldest first, from the data store's links
        (``DataStore.record_links``): one read for each call on its path. A record without a message list is a
        trajectory of one call; ``()`` for a record that the scenario does not store."""
        if self.stored(scenario, agent_record_id) is None:
            return ()
        path: list[tuple[str, int]] = []
        item_id: str | None = f"{scenario}/{agent_record_id}"
        while item_id is not None:
            link = self.data.record_links(self.owner, [item_id]).get(item_id)
            if link is None:
                break
            path.append((item_id.removeprefix(f"{scenario}/"), link.entry_count))
            item_id = link.parent
        return trajectory_calls(path[::-1]) if path else (agent_record_id,)

    def loss(self, scenario: str) -> RecordLoss:
        return RecordLoss() if scenario != self.scenario else self.data.record_loss(self.owner, scenario)

    def close(self) -> None:
        """The data store outlives the session, so closing changes nothing."""


__all__ = [
    "RECORD_PAGE_SIZE",
    "ItemRecordStore",
    "evaluation_call_item",
    "evaluation_call_items",
    "evaluation_call_summary",
    "old_record_copied",
    "record_content",
    "record_item",
    "stored_evaluation_call",
    "used_record",
    "used_records",
]
