"""Read sample processing work without running the samples pipeline."""

from __future__ import annotations

import time

from reef.dispatcher import Dispatcher
from reef.record2sample.service import SAMPLES_QUEUE
from reef.storage.work_queue import JsonValue, WorkItem


def processing_item(item: WorkItem) -> dict[str, JsonValue]:
    """Return the public task fields. Queue payloads and worker identities stay internal."""
    result = item.result or {}
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
    }


def get_processing(dispatcher: Dispatcher, scenario: str, task_id: str) -> dict[str, JsonValue] | None:
    """Return one task of the scenario, or None when the task is unknown or belongs to another scenario."""
    item = dispatcher.queue.get(SAMPLES_QUEUE, task_id)
    return None if item is None or item.scenario != scenario else processing_item(item)


def list_processing(
    dispatcher: Dispatcher,
    scenario: str,
    *,
    state: str | None = None,
    before: int | None = None,
    limit: int = 50,
) -> dict[str, JsonValue]:
    """Return one task page and counts of all retained tasks of the scenario.

    ``before`` excludes newer creation sequences. ``limit`` is 1 to 100. Counts are independent of the filter.
    Each task's counts cover its completed attempt only. Missing results remain null, including legacy results.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with dispatcher.locked_scenario(scenario) as current:
        enabled = any(component.trainer.trains for component in current.trainers)
    page = dispatcher.queue.page(
        SAMPLES_QUEUE, scenario=scenario, state=state, before_sequence=before, limit=limit + 1
    )
    rows = page[:limit]
    return {
        "scenario": scenario,
        "enabled": enabled,
        "read_at": time.time(),
        "retention_seconds": dispatcher.work_retention_seconds,
        "counts": dict(dispatcher.queue.counts(SAMPLES_QUEUE, scenario=scenario)),
        "tasks": [processing_item(item) for item in rows],
        "next": rows[-1].created_sequence if len(page) > limit else None,
    }
