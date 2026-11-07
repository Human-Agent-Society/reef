"""Record items: the inferences and reports of a scenario as items of its owner (RFC #1028).

Let ``s`` be a scenario, ``o`` its owner, and ``r`` an inference or report in the record store of ``s``. Its record
item ``record_item(r)`` is the item of ``o`` with:

* the ID ``<s>/<r.agent_record_id>``, unique within ``o`` because a scenario name has no ``/`` and a record ID is
  unique within its scenario;
* the content ``{agent_record_id, request_type, payload, created_at, references, artifact_id}`` of ``r``;
* the metadata ``kind: record``, ``scenarios: {s: train}`` and ``origin: {scenario: s, artifact_id}``,
  where ``artifact_id`` names the artifact that answered it, or null when ``r`` names none.

``copy_records`` makes the record items of one page of a record store, in record order. The data service copies each scenario (``copy_scenario_records``)
when it takes a ``record_copy`` item from the ``data`` queue, which control adds when records arrive
(``request_record_copy``). The mark ``record_copy:<s>`` keeps how far the records of ``s`` are copied: each record
of ``s`` with a sequence at most the mark that ``record_item`` gives an item has that item in the store of ``o``.

Another scenario ``t`` of ``o`` uses a record item ``i`` whose ``scenarios`` names ``t``. A trainer of ``t`` reads it from
the data store as ``copied_record(i, t)``, with the record ID ``i.item_id``, and each reference ``x`` of ``i`` becomes
``<origin>/x`` (``used_records``). Nothing is stored in ``t``, so when ``i`` leaves ``t``, the read stops
giving it (RFC #1219). A record store from before this can still hold such a copy. It stays, it has no record item of
its own, and a read skips the item.
"""

from __future__ import annotations

from reef.core.records_types import AgentRecord
from reef.storage.data import DataStore, NewItem, StoredItem
from reef.storage.data_queue import DATA_QUEUE, RECORD_COPY_KIND
from reef.storage.item_kinds import copied_record
from reef.storage.marks import Marks
from reef.storage.records import RecordStore
from reef.storage.scenario import ScenarioStorage
from reef.storage.work_queue import WorkItem, WorkQueue

#: The most records that one ``copy_records`` call reads.
RECORD_PAGE_SIZE = 256


def record_item(record: AgentRecord) -> NewItem:
    """The record item of ``record``, an inference or a report, as the module docstring defines it."""
    artifact_id = record.artifact_id
    return NewItem(
        f"{record.scenario}/{record.agent_record_id}",
        {
            "agent_record_id": record.agent_record_id,
            "request_type": record.request_type.value,
            "payload": dict(record.payload),
            "created_at": record.created_at,
            "references": list(record.references),
            "artifact_id": record.artifact_id,
        },
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


def record_copy_key(scenario: str) -> str:
    """The serial key of the ``record_copy`` items of ``scenario``: its copies run one at a time."""
    return f"{RECORD_COPY_KIND}:{scenario}"


def request_record_copy(queue: WorkQueue, scenario: str) -> WorkItem:
    """Ask the data service to copy the new records of ``scenario``; return the item that will copy them."""
    return queue.request(DATA_QUEUE, scenario, RECORD_COPY_KIND, record_copy_key(scenario), {})


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


__all__ = [
    "RECORD_PAGE_SIZE",
    "copy_records",
    "copy_scenario_records",
    "is_record_copy",
    "record_copy_mark",
    "record_item",
    "request_record_copy",
    "used_record",
    "used_records",
]
