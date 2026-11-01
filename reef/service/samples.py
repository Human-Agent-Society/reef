"""Transport-independent handling of a scenario's training samples, of a trainer's buffer, and of the trainer's
requested task-generation rounds.

``list_samples`` reads the scenario's sample store, and ``read_buffer`` and ``change_buffer`` read and change the
buffer of one trainer: the samples that its next job takes. A client changes the buffer only while the trainer
trains in ``manual`` mode. In the same mode, ``request_task_generation`` asks for one round of a trainer that
generates tasks, from the source sessions that the client names, and ``read_task_generation`` reads the request
(``reef.record2sample.requested_generation``). No aiohttp types appear here. ``reef.service.routes.samples``
adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.dispatcher import Dispatcher
from reef.record2sample.service.task_generation import requested_reports
from reef.storage.work_queue import WorkItem

#: The most samples that one listing returns.
MAX_SAMPLES_PER_PAGE = 1000
#: The most sample IDs that one buffer change names.
MAX_SAMPLES_PER_CHANGE = 1000
#: The most report IDs that one task-generation request names; the stage's source_records_per_round bounds it too.
MAX_REPORTS_PER_GENERATION = 1000


def list_samples(dispatcher: Dispatcher, scenario: str, *, after_sequence: int, limit: int) -> dict[str, Any]:
    """The stored samples after ``after_sequence``, at most ``limit``, as ``{scenario, samples, next}``.

    Each sample is ``{sample_id, sequence, kind, split, component, group, train_counts}``. ``next`` is the
    ``after`` of the next page, or None after the last sample. Raise ``ValueError`` for a negative
    ``after_sequence`` or a ``limit`` outside 1 to ``MAX_SAMPLES_PER_PAGE``.
    """
    if after_sequence < 0 or not 1 <= limit <= MAX_SAMPLES_PER_PAGE:
        raise ValueError(f"after must be non-negative and limit between 1 and {MAX_SAMPLES_PER_PAGE}")
    page = dispatcher.stored_samples(scenario, after_sequence=after_sequence, limit=limit)
    return {
        "scenario": scenario,
        "samples": [
            {
                "sample_id": stored.sample_id,
                "sequence": stored.sequence,
                "kind": stored.kind,
                "split": stored.split,
                "component": stored.component,
                "group": stored.group,
                "train_counts": dict(stored.train_counts),
            }
            for stored in page
        ],
        "next": page[-1].sequence if len(page) == limit else None,
    }


def read_buffer(dispatcher: Dispatcher, scenario: str, component: str | None) -> dict[str, Any]:
    """The buffer of the trainer of ``component`` as ``{scenario, component, training_mode, samples}``."""
    return {"scenario": scenario, "component": component, **dispatcher.sample_buffer(scenario, component)}


def change_buffer(dispatcher: Dispatcher, scenario: str, body: Mapping[str, Any], *, insert: bool) -> dict[str, Any]:
    """Insert the samples of ``body`` into a trainer's manual buffer, or remove them, and return ``read_buffer``.

    ``body`` holds ``sample_ids``, 1 to ``MAX_SAMPLES_PER_CHANGE`` non-empty strings, and optionally
    ``component``, the trainer's component in a scenario of several. Raise ``ValueError`` for another body,
    and as ``Dispatcher.change_buffer`` does.
    """
    if not set(body) <= {"sample_ids", "component"} or "sample_ids" not in body:
        raise ValueError("the body holds sample_ids and, optionally, component")
    sample_ids, component = body["sample_ids"], body.get("component")
    if (
        not isinstance(sample_ids, list)
        or not 1 <= len(sample_ids) <= MAX_SAMPLES_PER_CHANGE
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise ValueError(f"sample_ids must be 1 to {MAX_SAMPLES_PER_CHANGE} non-empty strings")
    if component is not None and (not isinstance(component, str) or not component):
        raise ValueError("component must be a non-empty string")
    listing = dispatcher.change_buffer(scenario, tuple(dict.fromkeys(sample_ids)), insert=insert, component=component)
    return {"scenario": scenario, "component": component, **listing}


def task_generation_json(item: WorkItem) -> dict[str, Any]:
    """A requested task-generation round as ``{id, scenario, component, report_ids, state, result, error}``.

    ``state`` is the state of the request's work item. ``result`` is ``{generation, train_tasks}`` once the round's
    generation is complete, else None; ``error`` names why the round failed, else None.
    """
    component, report_ids = requested_reports(item)
    return {
        "id": item.id,
        "scenario": item.scenario,
        "component": component,
        "report_ids": list(report_ids),
        "state": item.state,
        "result": None if item.result is None else dict(item.result),
        "error": item.error if item.state == "failed" else None,
    }


def request_task_generation(dispatcher: Dispatcher, scenario: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Queue one task-generation round of a trainer from the source sessions of ``body``; return the request.

    ``body`` holds ``report_ids``, 0 to ``MAX_REPORTS_PER_GENERATION`` non-empty strings, each the ID of a report in
    the scenario's record store, and optionally ``component``, the trainer's component in a scenario of several.
    An empty list asks for a round of the configured description alone. Raise ``ValueError`` for another body,
    and as ``Dispatcher.request_task_generation`` does.
    """
    if not set(body) <= {"report_ids", "component"} or "report_ids" not in body:
        raise ValueError("the body holds report_ids and, optionally, component")
    report_ids, component = body["report_ids"], body.get("component")
    if (
        not isinstance(report_ids, list)
        or len(report_ids) > MAX_REPORTS_PER_GENERATION
        or any(not isinstance(report_id, str) or not report_id for report_id in report_ids)
    ):
        raise ValueError(f"report_ids must be at most {MAX_REPORTS_PER_GENERATION} non-empty strings")
    if component is not None and (not isinstance(component, str) or not component):
        raise ValueError("component must be a non-empty string")
    item = dispatcher.request_task_generation(scenario, tuple(dict.fromkeys(report_ids)), component=component)
    return task_generation_json(item)


def read_task_generation(dispatcher: Dispatcher, scenario: str, request_id: str) -> dict[str, Any]:
    """The requested task-generation round ``request_id`` of ``scenario`` (``task_generation_json``)."""
    return task_generation_json(dispatcher.task_generation(scenario, request_id))
