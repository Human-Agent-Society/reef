"""The samples service: record selections through the samples queue, and the worker that makes the samples."""

from __future__ import annotations

import pytest

from reef.record2sample.service import (
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    SampleProducers,
    SamplesWorker,
    request_selection,
    selection_key,
)
from reef.storage.work_queue import MemoryWorkQueue


class Producers(SampleProducers):
    """Counts each call; ``background`` says whether the stages still run background work; ``error`` fails it."""

    def __init__(self, *, background: bool = False, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.background = background
        self.error = error

    def produce(self, scenario: str, component: str) -> bool:
        self.calls.append((scenario, component))
        if self.error is not None:
            raise self.error
        return self.background


@pytest.mark.unit
def test_a_waiting_selection_reads_every_new_record_so_requests_add_one_item():
    queue = MemoryWorkQueue()
    first = request_selection(queue, "s", "weights")
    assert request_selection(queue, "s", "weights") == first
    assert (first.kind, first.serial_key, first.payload) == (
        RECORD_SELECTION_KIND,
        selection_key("s", "weights"),
        {"scenario": "s", "component": "weights"},
    )
    # Once a worker took it, records that arrive later need a selection of their own, which waits for it.
    queue.take(SAMPLES_QUEUE, "w", [RECORD_SELECTION_KIND])
    second = request_selection(queue, "s", "weights")
    assert second.id != first.id and second.state == "queued"
    assert queue.take(SAMPLES_QUEUE, "w2", [RECORD_SELECTION_KIND]) is None
    assert request_selection(queue, "s", "harness").id != second.id


@pytest.mark.unit
def test_the_worker_makes_the_samples_and_comes_back_while_background_work_runs():
    queue = MemoryWorkQueue()
    producers = Producers(background=True)
    worker = SamplesWorker(queue, producers, worker_id="samples-1")
    request_selection(queue, "s", "weights")

    assert worker.run_once() is True
    assert producers.calls == [("s", "weights")]
    items = queue.items(SAMPLES_QUEUE, scenario="s")
    assert [item.state for item in items] == ["finished", "queued"]
    assert items[1].available_at > items[0].created_at
    producers.background = False
    assert worker.run_once() is False  # the next look waits for its time


@pytest.mark.unit
def test_a_selection_that_fails_is_retried_then_fails():
    queue = MemoryWorkQueue(max_attempts={SAMPLES_QUEUE: 2})
    worker = SamplesWorker(queue, Producers(error=RuntimeError("judge down")), worker_id="samples-1")
    item = request_selection(queue, "s", "weights")
    assert worker.run_once() and queue.get(SAMPLES_QUEUE, item.id).state == "queued"
    assert worker.run_once()
    failed = queue.get(SAMPLES_QUEUE, item.id)
    assert failed.state == "failed" and "judge down" in failed.error
