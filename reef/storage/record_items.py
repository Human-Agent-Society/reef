"""Record items: the inferences and reports of a scenario as items of its owner (RFC #1028).

Let ``s`` be a scenario, ``o`` its owner, and ``r`` an inference or report in the record store of ``s``. Its record
item ``record_item(r)`` is the item of ``o`` with:

* the ID ``<s>/<r.agent_record_id>``, unique within ``o`` because a scenario name has no ``/`` and a record ID is
  unique within its scenario;
* the content ``{agent_record_id, request_type, payload, created_at, references, artifact_ref}`` of ``r``;
* the metadata ``kind: record``, ``scenarios: {s: train}`` and ``origin: {scenario: s, release_id, artifact_id}``,
  where ``release_id`` is the release that ``r`` names and ``artifact_id`` the artifact that answered it, each
  null when ``r`` names none.

A training instruction is a record but not data, so it has no record item. ``copy_records`` makes the record items
of one page of a record store, in record order. The data service copies each scenario (``copy_scenario_records``)
when it takes a ``record_copy`` item from the ``data`` queue, which control adds when records arrive
(``request_record_copy``). The mark ``record_copy:<s>`` keeps how far the records of ``s`` are copied: each record
of ``s`` with a sequence at most the mark that ``record_item`` gives an item has that item in the store of ``o``.

Another scenario ``t`` of ``o`` uses a record item ``i`` whose ``scenarios`` names ``t`` through a copy in its own
record store: ``copied_record(i, t)``, with the record ID ``i.item_id``. Each reference ``x`` of ``i`` becomes
``<origin>/x``, the ID of the copy of that inference. ``sync_scenario_records`` appends the copies that ``t`` does not
hold yet. A record store keeps what it holds, so a copy stays in ``t`` when ``i`` leaves ``t``: the removal stops
copies that are not made yet, and not a copy that ``t`` holds. The copy of a record is not data that ``t`` made,
so it has no record item of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from reef.core.artifact_ref import decode_artifact_ref, encode_artifact_ref
from reef.core.records_types import AgentRecord, RequestType
from reef.storage.data import DATA_QUEUE, DataStore, NewItem, StoredItem
from reef.storage.marks import Marks
from reef.storage.records import RecordConflict, RecordStore
from reef.storage.scenario import ScenarioStorage
from reef.storage.work_queue import WorkItem, WorkQueue, request_waiting

logger = logging.getLogger(__name__)

#: The kind of the ``data`` queue item that copies the new records of one scenario.
RECORD_COPY_KIND = "record_copy"

#: The most records that one ``copy_records`` call reads.
RECORD_PAGE_SIZE = 256
#: The keys of a record item's content.
RECORD_CONTENT_KEYS = frozenset(
    {"agent_record_id", "request_type", "payload", "created_at", "references", "artifact_ref"}
)


def record_item(record: AgentRecord) -> NewItem:
    """The record item of ``record``, an inference or a report, as the module docstring defines it."""
    if record.request_type is RequestType.TRAIN:
        raise ValueError(f"record {record.agent_record_id!r} is a training instruction, which is not data")
    artifact_id = record.payload.get("artifact_id")
    return NewItem(
        f"{record.scenario}/{record.agent_record_id}",
        {
            "agent_record_id": record.agent_record_id,
            "request_type": record.request_type.value,
            "payload": dict(record.payload),
            "created_at": record.created_at,
            "references": list(record.references),
            "artifact_ref": None if record.artifact_ref is None else encode_artifact_ref(record.artifact_ref),
        },
        {
            "kind": "record",
            "scenarios": {record.scenario: "train"},
            "origin": {
                "scenario": record.scenario,
                "release_id": None if record.artifact_ref is None else record.artifact_ref.release_id,
                "artifact_id": artifact_id if isinstance(artifact_id, str) and artifact_id else None,
            },
        },
    )


def copied_record(item_id: str, content: Mapping[str, Any], origin: str, scenario: str) -> AgentRecord:
    """The copy in ``scenario`` of the record item ``item_id`` with ``content``, made in the scenario ``origin``.

    The copy has the record ID ``item_id``, and the request type, payload, time and release of ``content``. Each
    reference ``x`` becomes ``<origin>/x``, and so does each entry of a ``references`` list in the payload. Raise
    ``ValueError`` unless ``content`` has the keys ``RECORD_CONTENT_KEYS`` with the types that ``record_item``
    gives them, for an inference or a report, and ``item_id`` is ``<origin>/<agent_record_id>``.
    """
    if not isinstance(content, Mapping) or content.keys() != RECORD_CONTENT_KEYS:
        raise ValueError(f"a record item's content has exactly the keys {', '.join(sorted(RECORD_CONTENT_KEYS))}")
    record_id, request_type, payload = content["agent_record_id"], content["request_type"], content["payload"]
    created_at, references, artifact = content["created_at"], content["references"], content["artifact_ref"]
    if not isinstance(record_id, str) or item_id != f"{origin}/{record_id}":
        raise ValueError(f"a record item made in scenario {origin!r} has the ID {origin}/<agent_record_id>")
    if request_type not in (RequestType.INFERENCE.value, RequestType.REPORT.value):
        raise ValueError("a record item's request_type is 'inference' or 'report'")
    if not isinstance(payload, Mapping):
        raise ValueError("a record item's payload is an object")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
        raise ValueError("a record item's created_at is a number")
    if not isinstance(references, list) or not all(isinstance(reference, str) for reference in references):
        raise ValueError("a record item's references is a list of strings")
    if artifact is not None and not isinstance(artifact, Mapping):
        raise ValueError("a record item's artifact_ref is an object or null")
    copied = tuple(f"{origin}/{reference}" for reference in references)
    payload = dict(payload)
    if isinstance(payload.get("references"), list):
        payload["references"] = [f"{origin}/{reference}" for reference in payload["references"]]
    return AgentRecord(
        agent_record_id=item_id,
        scenario=scenario,
        request_type=RequestType(request_type),
        payload=payload,
        created_at=float(created_at),
        references=copied,
        artifact_ref=None if artifact is None else decode_artifact_ref(artifact),
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
    ``after_sequence``; its training instructions and its copies of other scenarios' record items get no item.
    Return the sequence of the page's last record, or ``after_sequence`` for an empty page. A record item that
    ``owner`` stores already changes nothing, so a copy that runs twice is safe. Raise ``ItemConflict``, and
    create nothing, when ``owner`` stores one of the IDs with other content.
    """
    page = records.replay_page(scenario, after_sequence=after_sequence, limit=RECORD_PAGE_SIZE)
    if not page:
        return after_sequence
    data.create(
        owner,
        [
            record_item(record)
            for _, record in page
            if record.request_type is not RequestType.TRAIN and not is_record_copy(data, owner, scenario, record)
        ],
    )
    return page[-1][0]


def record_copy_mark(scenario: str) -> str:
    """The name of the mark that keeps how far the records of ``scenario`` are copied."""
    return f"{RECORD_COPY_KIND}:{scenario}"


def request_record_copy(queue: WorkQueue, scenario: str) -> WorkItem:
    """Ask the data service to copy the new records of ``scenario``; return the item that will copy them."""
    return request_waiting(queue, DATA_QUEUE, scenario, RECORD_COPY_KIND, record_copy_mark(scenario), {})


def copy_scenario_records(storage: ScenarioStorage, data: DataStore, marks: Marks, scenario: str) -> int:
    """Copy the records of ``scenario`` after its mark, page by page, and advance the mark after each page.

    Return the number of pages copied. A scenario without an owner has no store to copy into, and copies nothing.
    A crash between a page and its mark copies the page again, which changes nothing.
    """
    owner = storage.read_settings(scenario).owner
    if owner is None:
        return 0
    name = record_copy_mark(scenario)
    store = storage.open(scenario)
    pages = 0
    try:
        after = marks.read(name)
        while True:
            sequence = copy_records(store.records, data, owner, scenario, after)
            if sequence == after:
                return pages
            marks.advance(name, sequence)
            after = sequence
            pages += 1
    finally:
        store.close()


def sync_scenario_records(records: RecordStore, data: DataStore, owner: str, scenario: str) -> tuple[str, ...]:
    """Append to ``records``, the record store of ``scenario``, the copy of each record item that it uses.

    The record items are those of ``owner`` whose ``scenarios`` names ``scenario`` and that another scenario made,
    in arrival order. A copy that ``records`` holds already changes nothing. A copy that ``records`` holds with
    other content stays, and the sync logs it: the item was deleted and made again. Return the IDs of the copies
    that the sync appended.
    """
    added: list[str] = []
    after = 0
    while page := data.query(owner, kind="record", scenario=scenario, after=after):
        after = page[-1].sequence
        for item in page:
            if item.content is None or origin_of(item) == scenario:
                continue
            copy = copied_record(item.item_id, item.content, origin_of(item), scenario)
            try:
                appended = records.append_result(copy)
            except RecordConflict:
                logger.warning(
                    "scenario %r keeps its copy of record item %r, which now has other content", scenario, item.item_id
                )
                continue
            if appended.inserted:
                added.append(item.item_id)
    return tuple(added)


__all__ = [
    "RECORD_CONTENT_KEYS",
    "RECORD_COPY_KIND",
    "RECORD_PAGE_SIZE",
    "copied_record",
    "copy_records",
    "copy_scenario_records",
    "is_record_copy",
    "record_copy_mark",
    "record_item",
    "request_record_copy",
    "sync_scenario_records",
]
