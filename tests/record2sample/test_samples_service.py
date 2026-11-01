"""The samples service: record selections through the samples queue, the worker that makes the samples, and the
worker of requested task-generation rounds."""

from __future__ import annotations

import threading

import pytest

from reef.record2sample.requested_generation import RequestedRoundResult
from reef.record2sample.service import (
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    TASK_GENERATION_KIND,
    ProducedSamples,
    SampleProducers,
    SamplesWorker,
    TaskGenerations,
    TaskGenerationWorker,
    file_task_generation,
    request_selection,
    selection_key,
)
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination
from reef.storage.work_queue import WorkLeaseLost


class Producers(SampleProducers):
    """Counts each call; ``background`` says whether the stages still run background work; ``error`` fails it."""

    def __init__(self, *, background: bool = False, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.background = background
        self.error = error

    def produce(self, scenario: str, component: str) -> ProducedSamples:
        self.calls.append((scenario, component))
        if self.error is not None:
            raise self.error
        return ProducedSamples(changed=True, background=self.background)


@pytest.mark.unit
def test_a_waiting_selection_reads_every_new_record_so_requests_add_one_item():
    queue = memory_coordination().work_queue
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
    queue = memory_coordination().work_queue
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
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={SAMPLES_QUEUE: 2})
    worker = SamplesWorker(queue, Producers(error=RuntimeError("judge down")), worker_id="samples-1")
    item = request_selection(queue, "s", "weights")
    assert worker.run_once() and queue.get(SAMPLES_QUEUE, item.id).state == "queued"
    assert worker.run_once()
    failed = queue.get(SAMPLES_QUEUE, item.id)
    assert failed.state == "failed" and "judge down" in failed.error


class FlakyQueue(SQLWorkQueue):
    """A queue that fails once, as a database that drops a connection, and then stops its worker."""

    def __init__(self) -> None:
        super().__init__(SQLiteCoordinationDatabase(None))
        self.calls = 0
        self.worker: SamplesWorker | None = None

    def requeue_expired(self, *, now=None):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("the database dropped the connection")
        self.worker.stopped.set()
        return super().requeue_expired(now=now)


@pytest.mark.unit
def test_a_worker_keeps_running_after_the_store_fails_for_a_moment():
    queue = FlakyQueue()
    worker = SamplesWorker(queue, Producers(), worker_id="samples-1", poll_seconds=0.01)
    queue.worker = worker
    worker.run()  # returns only because the second look stops it
    assert queue.calls == 2


class Generations(TaskGenerations):
    """Ends each round after ``looks`` looks with ``result``; keeps every look."""

    def __init__(self, result: RequestedRoundResult, *, looks: int = 2) -> None:
        self.result = result
        self.looks = looks
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []

    def advance_request(self, scenario, component, request_id, report_ids):
        self.calls.append((scenario, component, request_id, report_ids))
        return self.result if len(self.calls) >= self.looks else None


@pytest.mark.unit
def test_a_requested_round_holds_its_item_until_the_round_ends_and_the_requests_of_one_trainer_run_in_order():
    queue = memory_coordination().work_queue
    generations = Generations(RequestedRoundResult(3, ("t-1", "t-2")))
    worker = TaskGenerationWorker(queue, generations, worker_id="samples-1", result_poll_seconds=0.01)
    first = file_task_generation(queue, "s", "weights", ("rep-a", "rep-b"))
    second = file_task_generation(queue, "s", "weights", ("rep-a", "rep-b"))
    assert first.id != second.id, "the same reports again are a second round"
    assert (first.kind, first.serial_key, first.payload) == (
        TASK_GENERATION_KIND,
        "task_generation:s:weights",
        {"scenario": "s", "component": "weights", "report_ids": ["rep-a", "rep-b"]},
    )
    assert queue.take(SAMPLES_QUEUE, "other", [RECORD_SELECTION_KIND]) is None, "a selection worker takes none"

    assert worker.run_once() is True
    assert generations.calls == [("s", "weights", first.id, ("rep-a", "rep-b"))] * 2
    assert queue.get(SAMPLES_QUEUE, first.id).result == {"generation": 3, "train_tasks": ["t-1", "t-2"]}
    assert queue.get(SAMPLES_QUEUE, second.id).state == "queued"


@pytest.mark.unit
def test_a_requested_round_that_failed_fails_its_item_for_good():
    queue = memory_coordination().work_queue
    worker = TaskGenerationWorker(
        queue, Generations(RequestedRoundResult(0, error="the generator is down"), looks=1), worker_id="samples-1"
    )
    item = file_task_generation(queue, "s", "weights", ())
    worker.run_once()
    failed = queue.get(SAMPLES_QUEUE, item.id)
    assert (failed.state, failed.error, failed.attempt_count) == ("failed", "the generator is down", 1)


@pytest.mark.unit
def test_a_worker_that_stops_or_loses_its_lease_leaves_the_round_to_the_next_worker():
    queue = memory_coordination().work_queue
    generations = Generations(RequestedRoundResult(0), looks=10)
    worker = TaskGenerationWorker(queue, generations, worker_id="samples-1", result_poll_seconds=0.01)
    item = file_task_generation(queue, "s", "weights", ())
    lost = threading.Event()
    lost.set()
    with pytest.raises(WorkLeaseLost):
        worker.run_item(item, lost)
    worker.stopped.set()
    worker.run_once()
    assert queue.get(SAMPLES_QUEUE, item.id).state == "queued", "the item goes back to the queue"
