"""Transport-independent handling of a scenario's training samples and of a trainer's buffer.

``list_samples`` reads the scenario's sample store, and ``read_buffer`` and ``change_buffer`` read and change the
buffer of one trainer: the samples that its next job takes. A client changes the buffer only while the trainer
trains in ``manual`` mode. No aiohttp types appear here. ``reef.service.routes.samples`` adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.dispatcher import Dispatcher

#: The most samples that one listing returns.
MAX_SAMPLES_PER_PAGE = 1000
#: The most sample IDs that one buffer change names.
MAX_SAMPLES_PER_CHANGE = 1000


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
