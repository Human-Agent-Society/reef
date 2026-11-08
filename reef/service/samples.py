"""Transport-independent reads of a scenario's training samples.

``list_samples`` reads the data items that the scenario uses, and ``read_next_job`` reads the samples that one trainer's
next automatic job takes. No aiohttp types appear here. ``reef.service.routes.samples`` adapts it to HTTP.
"""

from __future__ import annotations

from typing import Any

from reef.dispatcher import Dispatcher

#: The most samples that one listing returns.
MAX_SAMPLES_PER_PAGE = 1000


def list_samples(dispatcher: Dispatcher, scenario: str, *, after_sequence: int, limit: int) -> dict[str, Any]:
    """The stored samples after ``after_sequence``, at most ``limit``, as ``{scenario, samples, next}``.

    Each sample is ``{sample_id, sequence, kind, split, component, group, train_counts}``. ``next`` is the
    ``after`` of the next page, or None after the last sample. Raise ``ValueError`` for a negative
    ``after_sequence`` or a ``limit`` outside 1 to ``MAX_SAMPLES_PER_PAGE``.
    """
    if after_sequence < 0 or not 1 <= limit <= MAX_SAMPLES_PER_PAGE:
        raise ValueError(f"after must be non-negative and limit between 1 and {MAX_SAMPLES_PER_PAGE}")
    page = dispatcher.listed_samples(scenario, after_sequence=after_sequence, limit=limit)
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


def read_next_job(dispatcher: Dispatcher, scenario: str, component: str | None) -> dict[str, Any]:
    """The next automatic job of the trainer of ``component`` as ``{scenario, component, training_mode, samples}``."""
    return {"scenario": scenario, "component": component, **dispatcher.next_job(scenario, component)}
