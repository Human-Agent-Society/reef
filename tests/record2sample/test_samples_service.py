"""The samples service: record selections through the samples queue, and the worker that makes the samples."""

from __future__ import annotations

import time

import pytest

from reef.record2sample.service import (
    BACKGROUND_POLL_SECONDS,
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    ProducedSamples,
    SampleProducers,
    SamplesWorker,
    request_selection,
    selection_key,
    selection_triggers,
)
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination


class Producers(SampleProducers):
    """Counts each call; ``background`` says whether the stages still run background work; ``due_in_seconds`` when a
    set that waits for time is due; ``error`` fails it; ``changed`` is whether a call gives the trainer new input."""

    def __init__(
        self,
        *,
        background: bool = False,
        due_in_seconds: float | None = None,
        error: Exception | None = None,
        changed: bool = True,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.background = background
        self.due_in_seconds = due_in_seconds
        self.error = error
        self.changed = changed

    def produce(self, scenario: str, component: str, selection) -> ProducedSamples:
        self.calls.append((scenario, component))
        selection.progress(4, 0)
        if self.error is not None:
            raise self.error
        return ProducedSamples(
            changed=self.changed,
            background=self.background,
            records_read=4,
            samples_created=2 if self.changed else 0,
            due_in_seconds=self.due_in_seconds,
        )


@pytest.mark.unit
def test_a_waiting_selection_reads_every_new_record_so_requests_add_one_item():
    queue = memory_coordination().work_queue
    first = request_selection(queue, "s", "weights", trigger="records")
    assert request_selection(queue, "s", "weights", trigger="records") == first
    assert (first.kind, first.serial_key, first.payload) == (
        RECORD_SELECTION_KIND,
        selection_key("s", "weights"),
        {"scenario": "s", "component": "weights"},
    )
    # Once a worker took it, records that arrive later need a selection of their own, which waits for it.
    queue.take(SAMPLES_QUEUE, "w", [RECORD_SELECTION_KIND])
    second = request_selection(queue, "s", "weights", trigger="records")
    assert second.id != first.id and second.state == "queued"
    assert queue.take(SAMPLES_QUEUE, "w2", [RECORD_SELECTION_KIND]) is None
    assert request_selection(queue, "s", "harness", trigger="records").id != second.id


@pytest.mark.unit
def test_the_worker_makes_the_samples_and_comes_back_while_background_work_runs():
    queue = memory_coordination().work_queue
    producers = Producers(background=True)
    worker = SamplesWorker(queue, producers, worker_id="samples-1")
    request_selection(queue, "s", "weights", trigger="records")

    assert worker.run_once() is True
    assert producers.calls == [("s", "weights")]
    items = queue.items(SAMPLES_QUEUE, scenario="s")
    assert [item.state for item in items] == ["finished", "queued"]
    assert items[0].result == {
        "changed": True,
        "background": True,
        "records_read": 4,
        "samples_created": 2,
        "due_in_seconds": None,
        "record_sequences": None,
    }
    assert items[1].available_at > items[0].created_at
    producers.background = False
    assert worker.run_once() is False  # the next look waits for its time


@pytest.mark.unit
def test_the_worker_comes_back_when_a_set_that_waits_for_time_is_due_or_a_record_arrives():
    queue = memory_coordination().work_queue
    worker = SamplesWorker(queue, Producers(due_in_seconds=600.0), worker_id="samples-1")
    request_selection(queue, "s", "weights", trigger="records")

    assert worker.run_once() is True
    finished, due = queue.items(SAMPLES_QUEUE, scenario="s")
    assert finished.result["due_in_seconds"] == 600.0
    assert due.state == "queued" and due.available_at >= finished.created_at + 600.0
    assert worker.run_once() is False
    # A new record does not wait for the due time: its request makes the queued selection available now.
    assert request_selection(queue, "s", "weights", trigger="records").id == due.id
    assert worker.run_once() is True


@pytest.mark.unit
def test_a_selection_that_fails_is_retried_then_fails():
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={SAMPLES_QUEUE: 2})
    worker = SamplesWorker(queue, Producers(error=RuntimeError("judge down")), worker_id="samples-1")
    item = request_selection(queue, "s", "weights", trigger="records")
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


@pytest.mark.unit
def test_a_selection_names_each_trigger_once_with_the_time_of_its_first_request():
    queue = memory_coordination().work_queue
    first = request_selection(queue, "s", "weights", trigger="commit")
    request_selection(queue, "s", "weights", trigger="records")
    again = request_selection(queue, "s", "weights", trigger="commit")
    assert again.id == first.id
    triggers = selection_triggers(queue.get(SAMPLES_QUEUE, first.id))
    assert [trigger["name"] for trigger in triggers] == ["commit", "records"]
    assert again.metadata["trigger:commit"] == first.metadata["trigger:commit"]
    with pytest.raises(ValueError, match="trigger"):
        request_selection(queue, "s", "weights", trigger="poll")


@pytest.mark.unit
def test_background_looks_back_off_while_nothing_changes_and_reset_after_a_change():
    queue = memory_coordination().work_queue
    producers = Producers(background=True, changed=False)
    worker = SamplesWorker(queue, producers, worker_id="samples-1")
    request_selection(queue, "s", "weights", trigger="records")
    waits = []
    for _ in range(7):
        assert worker.run_once() is True
        queued = queue.items(SAMPLES_QUEUE, scenario="s")[-1]
        assert [trigger["name"] for trigger in selection_triggers(queued)] == ["background"]
        waits.append(queued.metadata["background_poll_seconds"])
        # The test does not wait: the queued look becomes available now.
        request_selection(queue, "s", "weights", trigger="records")
    assert waits == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    producers.changed = True
    assert worker.run_once() is True
    assert queue.items(SAMPLES_QUEUE, scenario="s")[-1].metadata["background_poll_seconds"] == BACKGROUND_POLL_SECONDS


@pytest.mark.unit
def test_a_running_selection_writes_its_counts_so_far_for_its_attempt():
    queue = memory_coordination().work_queue
    seen = []

    class Watching(Producers):
        def produce(self, scenario, component, selection):
            assert selection.task_id == item.id
            selection.progress(4, 0)
            seen.append(queue.get(SAMPLES_QUEUE, item.id).metadata["progress"])
            selection.progress(5, 1)  # within the interval: no write
            seen.append(queue.get(SAMPLES_QUEUE, item.id).metadata["progress"])
            return ProducedSamples(changed=True, background=False, records_read=5, samples_created=1)

    item = request_selection(queue, "s", "weights", trigger="records")
    assert SamplesWorker(queue, Watching(), worker_id="samples-1").run_once() is True
    assert seen[0] == seen[1]
    assert {key: seen[0][key] for key in ("attempt", "records_read", "samples_created")} == {
        "attempt": 1,
        "records_read": 4,
        "samples_created": 0,
    }


@pytest.mark.unit
def test_a_background_job_that_a_selection_submits_asks_for_the_next_selection_when_it_ends(monkeypatch):
    import threading

    from reef.record2sample.service import worker as samples_worker

    # A long fallback look, so only the job's end can make the next selection available soon.
    monkeypatch.setattr(samples_worker, "BACKGROUND_POLL_SECONDS", 600.0)

    from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, SupportsReceipt

    release = threading.Event()

    class Judge(BackgroundJobRunner):
        async def run(self, job):
            import asyncio

            await asyncio.to_thread(release.wait, 5)
            return job

    background = BackgroundJobWorker(Judge(), thread_name="judge", concurrency=1)

    class Submitting(Producers):
        def produce(self, scenario, component, selection):
            assert background.submit(SupportsReceipt("r1"))
            return ProducedSamples(changed=False, background=True, records_read=0, samples_created=0)

    queue = memory_coordination().work_queue
    worker = SamplesWorker(queue, Submitting(), worker_id="samples-1")
    request_selection(queue, "s", "weights", trigger="records")
    try:
        assert worker.run_once() is True
        fallback = queue.items(SAMPLES_QUEUE, scenario="s")[-1]
        assert fallback.available_at > fallback.created_at + 500, "the fallback look waits"
        release.set()
        # The job's end makes the queued selection available now.
        deadline = time.monotonic() + 5
        while queue.get(SAMPLES_QUEUE, fallback.id).available_at > fallback.created_at + 1:
            assert time.monotonic() < deadline, "the job's end asked for no selection"
            time.sleep(0.01)
    finally:
        background.close()
    assert worker.run_once() is True
