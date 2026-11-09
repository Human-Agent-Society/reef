"""Read sample processing work without running the samples pipeline."""

from __future__ import annotations

import time

from reef.dispatcher import Dispatcher
from reef.record2sample.service import SAMPLES_QUEUE, selection_triggers
from reef.storage.records import RecordStore
from reef.storage.work_queue import HELD_STATES, JsonValue, WorkItem

#: The most tasks that one page with ``changed=True`` reads to fill the page.
MAX_SCANNED_TASKS = 5000
#: The tasks that one read of the queue gives while a page with ``changed=True`` fills.
SCAN_BATCH = 200


def processing_item(item: WorkItem) -> dict[str, JsonValue]:
    """Return the public task fields. Queue payloads and worker identities stay internal.

    ``triggers`` lists why the task ran (``selection_triggers``). ``progress`` holds the counts so far of the attempt
    that runs now, and is null otherwise.
    """
    result = item.result or {}
    progress = item.metadata.get("progress")
    current = (
        item.state in HELD_STATES and isinstance(progress, dict) and progress.get("attempt") == item.attempt_count
    )
    return {
        "id": item.id,
        "component": item.payload.get("component"),
        "state": item.state,
        "attempt_count": item.attempt_count,
        "created_at": item.created_at,
        "started_at": item.started_at,
        "ended_at": item.ended_at,
        "available_at": item.available_at,
        "error": item.error,
        "changed": result.get("changed"),
        "background": result.get("background"),
        "records_read": result.get("records_read"),
        "samples_created": result.get("samples_created"),
        "record_sequences": result.get("record_sequences"),
        "triggers": selection_triggers(item),
        "progress": progress if current else None,
    }


def unchanged(item: WorkItem) -> bool:
    """Whether ``item`` finished and gave its trainer nothing new."""
    return item.state == "finished" and (item.result or {}).get("changed") is False


def get_processing(dispatcher: Dispatcher, scenario: str, task_id: str) -> dict[str, JsonValue] | None:
    """Return one task of the scenario, or None when the task is unknown or belongs to another scenario."""
    item = dispatcher.queue.get(SAMPLES_QUEUE, task_id)
    return None if item is None or item.scenario != scenario else processing_item(item)


def read_processing_log(
    dispatcher: Dispatcher,
    scenario: str,
    task_id: str,
    *,
    attempt: int | None = None,
    after: int = -1,
    limit: int = 500,
) -> dict[str, JsonValue] | None:
    """Return one page of the log of one attempt of a task, or None when the task is unknown or belongs to another
    scenario.

    ``attempt`` is 1 to the task's ``attempt_count``, and the last attempt when None. The page holds up to ``limit``
    lines (1 to 1000) with a number above ``after``. ``next`` is the ``after`` of the next page, or null when the page
    is the end of the lines kept so far. A task that no worker took has no attempt and no lines.
    """
    item = dispatcher.queue.get(SAMPLES_QUEUE, task_id)
    if item is None or item.scenario != scenario:
        return None
    chosen = item.attempt_count if attempt is None else attempt
    if item.attempt_count and not 1 <= chosen <= item.attempt_count:
        raise ValueError(f"attempt must be 1 to {item.attempt_count}")
    lines = (
        dispatcher.queue.read_log(SAMPLES_QUEUE, task_id, chosen, after=after, limit=limit)
        if item.attempt_count
        else ()
    )
    return {
        "task_id": task_id,
        "attempt": chosen if item.attempt_count else None,
        "attempt_count": item.attempt_count,
        "running": item.state in HELD_STATES and chosen == item.attempt_count,
        "lines": [
            {"number": line.number, "at": line.at, "level": line.level, "logger": line.logger, "message": line.message}
            for line in lines
        ],
        "next": lines[-1].number if len(lines) == limit else None,
    }


def list_processing(
    dispatcher: Dispatcher,
    scenario: str,
    *,
    state: str | None = None,
    before: int | None = None,
    limit: int = 50,
    changed: bool = False,
) -> dict[str, JsonValue]:
    """Return one task page and counts of all retained tasks of the scenario.

    ``before`` excludes newer creation sequences. ``limit`` is 1 to 100. Counts are independent of the filter.
    Each task's counts cover its completed attempt only. Missing results remain null, including legacy results.

    With ``changed``, the page omits the finished tasks that gave the trainer nothing new (``unchanged``), and reads
    at most ``MAX_SCANNED_TASKS`` tasks to fill it. ``hidden`` then lists each run of omitted tasks: ``after`` is the
    ID of the shown task just newer than the run, or null at the top of the page; ``count`` is the number of tasks
    in the run; ``first_created_at`` and ``last_created_at`` are the creation times of its oldest and newest task.
    Without ``changed``, ``hidden`` is empty.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with dispatcher.locked_scenario(scenario) as current:
        trains = frozenset(component.component for component in current.trainers if component.trainer.trains)
        records = current.store.records
    rows, hidden, next_cursor = (
        changed_page(dispatcher, scenario, state, before, limit)
        if changed
        else plain_page(dispatcher, scenario, state, before, limit)
    )
    return {
        "scenario": scenario,
        "enabled": bool(trains),
        "records_waiting": records_waiting(dispatcher, scenario, trains, records),
        "read_at": time.time(),
        "retention_seconds": dispatcher.work_retention_seconds,
        "counts": dict(dispatcher.queue.counts(SAMPLES_QUEUE, scenario=scenario)),
        "tasks": [processing_item(item) for item in rows],
        "hidden": hidden,
        "next": next_cursor,
    }


def records_waiting(dispatcher: Dispatcher, scenario: str, trains: frozenset[str], records: RecordStore) -> int | None:
    """The number of the scenario's records after the last record that a trainer read, for the trainer furthest
    behind; None when a trainer that trains has no retained task with ``record_sequences``.

    The newest such task of each trainer gives its cursor. A page reads at most ``MAX_SCANNED_TASKS`` tasks for them.
    """
    if not trains:
        return None
    cursors: dict[str, int] = {}
    before: int | None = None
    for _ in range(MAX_SCANNED_TASKS // SCAN_BATCH):
        batch = dispatcher.queue.page(
            SAMPLES_QUEUE, scenario=scenario, state="finished", before_sequence=before, limit=SCAN_BATCH
        )
        for item in batch:
            component = item.payload.get("component")
            sequences = (item.result or {}).get("record_sequences")
            if (
                isinstance(component, str)
                and component in trains
                and component not in cursors
                and isinstance(sequences, list)
                and isinstance(sequences[-1], int)
            ):
                cursors[component] = sequences[-1]
        if set(cursors) == trains:
            return records.count(scenario, after_sequence=min(cursors.values()))
        if len(batch) < SCAN_BATCH:
            return None
        before = batch[-1].created_sequence
    return None


def plain_page(
    dispatcher: Dispatcher, scenario: str, state: str | None, before: int | None, limit: int
) -> tuple[list[WorkItem], list[JsonValue], int | None]:
    """The newest ``limit`` tasks before ``before``, no hidden runs, and the cursor of the next page or None."""
    page = dispatcher.queue.page(
        SAMPLES_QUEUE, scenario=scenario, state=state, before_sequence=before, limit=limit + 1
    )
    rows = list(page[:limit])
    return rows, [], rows[-1].created_sequence if len(page) > limit else None


def changed_page(
    dispatcher: Dispatcher, scenario: str, state: str | None, before: int | None, limit: int
) -> tuple[list[WorkItem], list[JsonValue], int | None]:
    """The newest ``limit`` tasks before ``before`` that are not ``unchanged``, the runs of omitted tasks between
    them, and the cursor of the next page or None. A page that reads ``MAX_SCANNED_TASKS`` tasks ends there. A run
    after the last shown task belongs to the next page, so a page that is full does not list it."""
    rows: list[WorkItem] = []
    hidden: list[JsonValue] = []
    run: list[WorkItem] = []
    cursor, scanned = before, 0

    def close_run() -> None:
        if run:
            hidden.append(
                {
                    "after": rows[-1].id if rows else None,
                    "count": len(run),
                    "first_created_at": run[-1].created_at,
                    "last_created_at": run[0].created_at,
                }
            )
            run.clear()

    while scanned < MAX_SCANNED_TASKS:
        batch = dispatcher.queue.page(
            SAMPLES_QUEUE, scenario=scenario, state=state, before_sequence=cursor, limit=SCAN_BATCH
        )
        for item in batch:
            scanned += 1
            cursor = item.created_sequence
            if unchanged(item):
                run.append(item)
                continue
            if len(rows) == limit:
                return rows, hidden, rows[-1].created_sequence
            close_run()
            rows.append(item)
        if len(batch) < SCAN_BATCH:
            close_run()
            return rows, hidden, None
    close_run()
    return rows, hidden, cursor
