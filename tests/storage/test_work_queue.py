import dataclasses
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass

import pytest
import sqlalchemy
from tests.support.polling import wait_until

from reef.core.errors import RuntimeConfigError
from reef.core.plugins import PluginConfigError
from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.execution import workers
from reef.execution.workers import ItemDeferredError, ItemFailedError, QueueWorker, WorkerStoppedError
from reef.storage.coordination.control_lease import ControlLease
from reef.storage.coordination.served_models import (
    AUTOMATIC_RETRIES,
    INFERENCE_OWNER_QUEUE,
    SERVING_KIND,
    SERVING_QUEUE,
    STOP_SERVING_KIND,
    ControlLeaseConflictError,
    ServedModelsWriter,
)
from reef.storage.coordination.work_queue import (
    DEFERRED_ATTEMPTS,
    LEASE_RETRY_MAX_SECONDS,
    LeaseExtender,
    NewWork,
    WorkConflictError,
    WorkItem,
    WorkLeaseLostError,
    WorkQueue,
)
from reef.storage.sql.coordination import (
    ADDED_INDEXES,
    PostgresCoordinationDatabase,
    SQLControlLease,
    SQLiteCoordinationDatabase,
    SQLServedModels,
    SQLWorkQueue,
)


@dataclass
class Coordination:
    queue: WorkQueue
    served: ServedModelsWriter
    lease: ControlLease


@pytest.fixture(params=["sqlite-memory", "sqlite", "postgres"])
def open_coordination(request, tmp_path):
    """Open the coordination stores of one backend with the given lease and attempt settings."""
    databases = []

    def opened(*, lease_seconds: float = 10, max_attempts: dict[str, int] | None = None) -> Coordination:
        if request.param == "postgres":
            url, schema = request.getfixturevalue("postgres_config")
            database = PostgresCoordinationDatabase(url, schema=schema)
        else:
            database = SQLiteCoordinationDatabase(None if request.param == "sqlite-memory" else tmp_path / "c")
        databases.append(database)
        lease = SQLControlLease(database, lease_seconds=lease_seconds)
        return Coordination(
            SQLWorkQueue(database, lease_seconds=lease_seconds, max_attempts=max_attempts),
            SQLServedModels(database, lease),
            lease,
        )

    yield opened
    for database in databases:
        with closing(database):
            pass


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_a_work_queue_that_an_earlier_reef_made_gets_the_later_indexes(backend, request, tmp_path):
    def opened():
        if backend == "postgres":
            url, schema = request.getfixturevalue("postgres_config")
            return PostgresCoordinationDatabase(url, schema=schema)
        return SQLiteCoordinationDatabase(tmp_path / "c")

    def index_names(database) -> set[str]:
        with database.connection() as connection:
            schema = database.tables.work_item.schema
            return {index["name"] for index in sqlalchemy.inspect(connection).get_indexes("work_item", schema=schema)}

    with closing(opened()) as database:
        SQLWorkQueue(database).append(job("j1"))
        # The tables of an earlier Reef: the same work_item, without the later indexes. SQLite has no index of the IDs
        # in code point order: its primary key is in that order.
        added = {index.name for index in database.tables.work_item.indexes} & ADDED_INDEXES
        assert ("work_item_by_id" in added) == (backend == "postgres")
        prefix = "" if database.tables.work_item.schema is None else f"{database.tables.work_item.schema}."
        with database.connection(write=True) as connection:
            for name in added:
                connection.exec_driver_sql(f"DROP INDEX {prefix}{name}")
        assert not added & index_names(database)
    with closing(opened()) as database:
        assert index_names(database) >= added
        assert SQLWorkQueue(database).take("jobs", "w1", ["training_job:tinker"]).id == "j1"


@pytest.mark.unit
def test_prefix_state_counts_include_only_the_requested_items_and_top_level_boolean_marker(open_coordination):
    queue = open_coordination().queue
    for item_id, scenario, kind in (
        ("e:1", "s", "rollout"),
        ("e:2", "s", "rollout"),
        ("e:3", "s", "rollout"),
        ("e:4", "other", "rollout"),
        ("e:5", "s", "template"),
        ("other:1", "s", "rollout"),
    ):
        queue.append(NewWork("q", item_id, scenario, kind, {}))
    queue.take("q", "worker", ["rollout"])
    queue.finish("q", "e:1", "worker", {})
    queue.write_metadata("q", "e:1", "waits", True)
    queue.write_metadata("q", "e:2", "nested", {"waits": True})
    queue.write_metadata("q", "e:3", "waits", 1)

    assert queue.prefix_state_counts("q", scenario="s", kind="rollout", id_prefix="e:", marker="waits") == {
        "finished": (1, "e:1", True),
        "queued": (2, "e:3", False),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("queue", ""), ("scenario", None), ("kind", None), ("id_prefix", ""), ("id_prefix", None), ("marker", 1)],
)
def test_prefix_state_counts_rejects_invalid_arguments(open_coordination, field, value):
    queue = open_coordination().queue
    inputs = {"queue": "q", "scenario": "s", "kind": "rollout", "id_prefix": "e:", "marker": "waits"}
    inputs[field] = value

    with pytest.raises(ValueError, match=field):
        queue.prefix_state_counts(**inputs)
    with pytest.raises(ValueError, match=field):
        WorkQueue.prefix_state_counts(queue, **inputs)


def test_bulk_reads_preserve_id_order_and_duplicates_and_skip_missing_items(open_coordination):
    queue = open_coordination().queue
    first = queue.append(job("a"))
    second = queue.append(job("b"))
    queue.append(NewWork("other", "missing", "s", "task", {}))
    item_ids = ["b", "missing", "a", "b"]

    assert queue.read_items("jobs", item_ids) == (second, first, second)
    assert WorkQueue.read_items(queue, "jobs", item_ids) == (second, first, second)
    assert queue.read_items("jobs", []) == ()


def test_the_items_of_an_id_prefix_are_the_items_whose_id_starts_with_it(open_coordination):
    queue = open_coordination().queue
    for item_id in ("rollout:e1:0", "rollout:e1:1", "rollout:e10:0", "rollout:e1;0", "Rollout:e1:2"):
        queue.append(job(item_id, serial_key=None))
    assert [item.id for item in queue.items("jobs", id_prefix="rollout:e1:")] == ["rollout:e1:0", "rollout:e1:1"]
    assert [item.id for item in queue.items("jobs", scenario="other", id_prefix="rollout:e1:")] == []
    with pytest.raises(ValueError, match="id_prefix"):
        queue.items("jobs", id_prefix="")


@pytest.mark.unit
def test_a_read_without_payload_gives_items_with_an_empty_payload(open_coordination):
    queue = open_coordination().queue
    first = queue.append(NewWork("jobs", "rollout:e1:0", "s", "rollout", {"tree": {"files": {"a.md": "x" * 1000}}}))
    second = queue.append(NewWork("jobs", "rollout:e1:1", "s", "rollout", {"task_index": 4}))
    # Only the payload differs from the stored items.
    empty = (dataclasses.replace(first, payload={}), dataclasses.replace(second, payload={}))

    assert queue.items("jobs", id_prefix="rollout:e1:", payload=False) == empty
    assert queue.items("jobs", id_prefix="rollout:e1:") == (first, second)
    ids = ["rollout:e1:1", "missing", "rollout:e1:0"]
    assert queue.read_items("jobs", ids, payload=False) == empty[::-1]
    assert WorkQueue.read_items(queue, "jobs", ids, payload=False) == empty[::-1]
    assert queue.read_items("jobs", ids) == (second, first)


def job(
    item_id: str, *, scenario: str = "s", kind: str = "training_job:tinker", serial_key: str | None = "s"
) -> NewWork:
    return NewWork("jobs", item_id, scenario, kind, {"input": item_id}, serial_key=serial_key)


@pytest.mark.unit
def test_an_attempt_that_its_worker_gave_back_does_not_count(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.append(job("a"), now=0)
    # A service that restarts three times gives the item back three times.
    for restart in range(3):
        queue.take("jobs", f"w{restart}", ["training_job:tinker"], now=restart)
        given_back = queue.give_back("jobs", "a", f"w{restart}", now=restart)
        assert (given_back.state, given_back.error, given_back.ended_at) == ("queued", "the worker stopped", None)
        assert (given_back.attempt_count, given_back.stopped_attempt_count) == (restart + 1, restart + 1)
    with pytest.raises(WorkLeaseLostError):
        queue.give_back("jobs", "a", "w2", now=3)
    # The item still has its two attempts.
    queue.take("jobs", "w", ["training_job:tinker"], now=4)
    assert queue.fail("jobs", "a", "w", "transient", retryable=True, now=5).state == "queued"
    queue.take("jobs", "w", ["training_job:tinker"], now=6)
    assert [item.state for item in queue.requeue_expired(now=100)] == ["failed"]


@pytest.mark.unit
def test_an_item_given_back_with_a_time_and_a_reason_waits_until_that_time(open_coordination):
    queue = open_coordination().queue
    queue.append(job("a"), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=1)
    waiting = queue.give_back("jobs", "a", "w", now=1, available_at=60, error="waiting for capacity")
    assert (waiting.state, waiting.error, waiting.available_at) == ("queued", "waiting for capacity", 60)
    assert queue.take("jobs", "w", ["training_job:tinker"], now=59) is None
    assert queue.take("jobs", "w", ["training_job:tinker"], now=60).attempt_count == 2
    with pytest.raises(ValueError):
        queue.give_back("jobs", "a", "w", now=61, available_at=math.inf)


@pytest.mark.unit
def test_a_deferred_give_back_counts_apart_and_only_for_the_attempt_that_holds_the_item(
    open_coordination, monkeypatch
):
    queue = open_coordination().queue
    queue.append(job("a"), now=0)
    first = queue.take("jobs", "w", ["training_job:tinker"], now=1)
    deferred = queue.give_back(
        "jobs",
        "a",
        "w",
        now=1,
        available_at=31,
        error="waiting for capacity",
        deferred=True,
        attempt=first.attempt_count,
    )
    assert (deferred.stopped_attempt_count, deferred.deferred_attempt_count) == (1, 1)
    assert deferred.metadata == {DEFERRED_ATTEMPTS: 1}
    # A worker that stops gives the item back too; that is no refusal of capacity. An item without the key, such as
    # one that an earlier Reef gave back, has the count 0.
    queue.take("jobs", "w", ["training_job:tinker"], now=31)
    stopped = queue.give_back("jobs", "a", "w", now=32)
    assert (stopped.stopped_attempt_count, stopped.deferred_attempt_count) == (2, 1)
    assert WorkItem("jobs", "b", "s", "k", {}, attempt_count=3, stopped_attempt_count=3).deferred_attempt_count == 0
    # An attempt whose lease expired cannot give back the item that a newer attempt of the same worker holds.
    expired = queue.take("jobs", "w", ["training_job:tinker"], now=33)
    queue.requeue_expired(now=100)
    newer = queue.take("jobs", "w", ["training_job:tinker"], now=101)
    with pytest.raises(WorkLeaseLostError):
        queue.give_back("jobs", "a", "w", now=102, deferred=True, attempt=expired.attempt_count)
    # A late refusal at the retry cutoff must also leave the new lease with its worker.
    monkeypatch.setattr(workers.time, "time", lambda: 102.0)
    worker = FailingWorker(queue, ItemDeferredError("waiting for capacity"))
    worker.max_deferrals = 1
    worker.defer(expired, "waiting for capacity")
    held = queue.read("jobs", "a")
    assert (held.state, held.attempt_count) == ("assigned", newer.attempt_count)
    assert (held.stopped_attempt_count, held.deferred_attempt_count) == (2, 1)


@pytest.mark.unit
def test_a_worker_takes_the_oldest_item_it_accepts_and_finishes_it_under_its_lease(open_coordination):
    queue = open_coordination().queue
    queue.append(job("a"), now=0)
    queue.append(job("b", kind="training_job:slime", serial_key=None), now=0)

    assert queue.take("jobs", "w1", ["training_job:other"], now=1) is None
    taken = queue.take("jobs", "w1", ["training_job:tinker", "training_job:slime"], now=1)
    assert taken is not None and taken.id == "a" and taken.state == "assigned"
    assert taken.worker_id == "w1" and taken.lease_expires_at == 11 and taken.attempt_count == 1

    queue.start("jobs", "a", "w1", now=2)
    queue.append_progress("jobs", "a", "w1", {"step": 1}, number=0, now=2)
    queue.append_progress("jobs", "a", "w1", {"step": 1}, number=0, now=2)  # a continuing worker reports again safely
    with pytest.raises(ValueError, match="skips"):
        queue.append_progress("jobs", "a", "w1", {"step": 3}, number=2, now=2)
    with pytest.raises(WorkLeaseLostError):
        queue.finish("jobs", "a", "w2", {"ok": True})
    done = queue.finish("jobs", "a", "w1", {"ok": True}, now=3)
    assert done.state == "finished" and done.final and done.result == {"ok": True}
    assert done.worker_id is None and done.progress == ({"step": 1},)


@pytest.mark.unit
def test_adding_is_idempotent_by_id_and_rejects_a_different_payload(open_coordination):
    queue = open_coordination().queue
    first = queue.append(job("a"))
    assert queue.append(job("a")) == first
    with pytest.raises(WorkConflictError):
        queue.append(NewWork("jobs", "a", "s", "training_job:tinker", {"input": "other"}))


@pytest.mark.unit
def test_items_with_one_serial_key_run_one_at_a_time_in_order(open_coordination):
    queue = open_coordination().queue
    queue.append(job("first"), now=0)
    queue.append(job("second"), now=0)
    queue.append(job("other", scenario="t", serial_key="t"), now=0)

    taken = [queue.take("jobs", "w", ["training_job:tinker"], now=1) for _ in range(3)]
    assert [item.id if item else None for item in taken] == ["first", "other", None]
    queue.finish("jobs", "first", "w", {}, now=2)
    assert queue.take("jobs", "w", ["training_job:tinker"], now=2).id == "second"


@pytest.mark.unit
def test_an_expired_lease_returns_the_item_and_a_late_worker_loses_it(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.append(job("a"), now=0)
    queue.take("jobs", "w1", ["training_job:tinker"], now=0)
    assert queue.extend_lease("jobs", "a", "w1", now=5).lease_expires_at == 15

    assert [item.state for item in queue.requeue_expired(now=16)] == ["queued"]
    with pytest.raises(WorkLeaseLostError):
        queue.extend_lease("jobs", "a", "w1", now=16)
    again = queue.take("jobs", "w2", ["training_job:tinker"], now=17)
    assert again.worker_id == "w2" and again.attempt_count == 2
    # The last attempt expires too: no attempts are left, so the item fails.
    assert [item.state for item in queue.requeue_expired(now=100)] == ["failed"]


class FlakyLeaseWorkQueue(SQLWorkQueue):
    """A work queue whose ``extend_lease`` raises scripted errors before delegating to ``SQLWorkQueue``.

    Values: ``outcomes`` is a list of optional exceptions popped from the front on each ``extend_lease`` call;
    ``None`` or an exhausted list delegates to ``SQLWorkQueue.extend_lease``. ``call_count`` counts every
    ``extend_lease`` call and ``success_count`` counts completed delegations, both guarded by ``lock``.

    Purpose: unit tests of ``LeaseExtender`` inject transient errors and lease-loss errors cleanly.
    """

    def __init__(
        self,
        database: SQLiteCoordinationDatabase,
        *,
        lease_seconds: float,
        outcomes: list[Exception | None],
    ) -> None:
        super().__init__(database, lease_seconds=lease_seconds)
        self.outcomes: list[Exception | None] = list(outcomes)
        self.call_count = 0
        self.success_count = 0
        self.lock = threading.Lock()

    def extend_lease(
        self, queue: str, item_id: str, worker_id: str, *, claim: WorkItem | None = None, now: float | None = None
    ) -> WorkItem:
        with self.lock:
            self.call_count += 1
            outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome
        extended = super().extend_lease(queue, item_id, worker_id, claim=claim, now=now)
        with self.lock:
            self.success_count += 1
        return extended


class RecordingStopEvent(threading.Event):
    """A ``threading.Event`` that records each ``wait`` timeout and sets itself after ``max_waits`` calls.

    Values: ``max_waits`` is a positive integer; ``waits`` records each ``timeout`` passed to ``wait``. Calling
    ``wait`` appends ``timeout`` to ``waits``, calls ``set()`` once ``len(waits) > max_waits`` if the event is not yet
    set, and returns ``is_set()``.

    Purpose: deterministic unit tests inspect the exact wait intervals of ``LeaseExtender.run`` without sleeping.
    """

    def __init__(self, max_waits: int) -> None:
        if isinstance(max_waits, bool) or max_waits < 1:
            raise ValueError("max_waits must be at least 1")
        super().__init__()
        self.max_waits = max_waits
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        if not self.is_set() and len(self.waits) > self.max_waits:
            self.set()
        return self.is_set()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lease_seconds", "expected_waits"),
    [
        (60.0, [20.0, LEASE_RETRY_MAX_SECONDS, LEASE_RETRY_MAX_SECONDS, 20.0]),
        (6.0, [2.0, 0.6, 0.6, 2.0]),
    ],
)
def test_lease_extender_retries_with_bounded_short_wait_and_resets_after_recovery(
    lease_seconds: float, expected_waits: list[float]
):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = FlakyLeaseWorkQueue(
            database,
            lease_seconds=lease_seconds,
            outcomes=[RuntimeError("pool timeout"), RuntimeError("db restart"), None],
        )
        queue.append(job("a"))
        taken = queue.take("jobs", "w1", ["training_job:tinker"])
        extender = LeaseExtender(queue, taken, "w1")
        recorder = RecordingStopEvent(max_waits=3)
        extender.done = recorder
        extender.run()

        assert recorder.waits == expected_waits
        assert recorder.is_set()
        assert queue.call_count == 3
        assert not extender.lease_lost.is_set()
        assert queue.read("jobs", "a").lease_expires_at >= taken.lease_expires_at


@pytest.mark.unit
def test_lease_extender_recovers_before_lease_expiry_on_transient_error():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = FlakyLeaseWorkQueue(database, lease_seconds=1.5, outcomes=[RuntimeError("transient"), None])
        queue.append(job("a"))
        taken = queue.take("jobs", "w1", ["training_job:tinker"])
        with LeaseExtender(queue, taken, "w1") as extender:
            wait_until(lambda: queue.success_count >= 1, timeout_seconds=5)
            refreshed = queue.read("jobs", "a")
            assert not extender.lease_lost.is_set()
            assert queue.call_count >= 2
            assert refreshed.lease_expires_at > taken.lease_expires_at
            assert queue.requeue_expired() == ()


@pytest.mark.unit
def test_lease_extender_exits_immediately_on_lease_lost_without_retrying():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = FlakyLeaseWorkQueue(database, lease_seconds=0.06, outcomes=[WorkLeaseLostError("lost")])
        queue.append(job("a"))
        taken = queue.take("jobs", "w1", ["training_job:tinker"])
        with LeaseExtender(queue, taken, "w1") as extender:
            assert extender.lease_lost.wait(2.0)
            extender.thread.join(timeout=2.0)
            assert not extender.thread.is_alive()
            assert queue.call_count == 1


@pytest.mark.unit
def test_lease_extender_stop_wakes_promptly_during_wait():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = FlakyLeaseWorkQueue(database, lease_seconds=60.0, outcomes=[])
        queue.append(job("a"))
        taken = queue.take("jobs", "w1", ["training_job:tinker"])
        extender = LeaseExtender(queue, taken, "w1")
        extender.thread.start()
        started = time.monotonic()
        extender.stop()
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert not extender.thread.is_alive()
        assert not extender.lease_lost.is_set()


@pytest.mark.unit
def test_extending_a_lease_refreshes_expiry_without_taking_the_sequence_counter(open_coordination, monkeypatch):
    queue = open_coordination().queue
    queue.append(job("a"), now=0)
    taken = queue.take("jobs", "w1", ["training_job:tinker"], now=0)

    def refuse_counter(connection, sequence):
        raise AssertionError(f"extend_lease must not acquire sequence counter {sequence}")

    monkeypatch.setattr(queue.database, "counter", refuse_counter)
    extended = queue.extend_lease("jobs", "a", "w1", now=5)

    assert extended.lease_expires_at == 15
    assert extended.changed_sequence == taken.changed_sequence
    assert queue.read("jobs", "a") == extended
    assert queue.read_changes("jobs", taken.changed_sequence) == ()


@pytest.mark.unit
def test_extending_a_lease_rejects_reassigned_ownership_without_mutating_the_item(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.append(job("a"), now=0)
    queue.take("jobs", "w1", ["training_job:tinker"], now=0)
    assert queue.requeue_expired(now=16)[0].state == "queued"
    reassigned = queue.take("jobs", "w2", ["training_job:tinker"], now=17)

    with pytest.raises(WorkLeaseLostError):
        queue.extend_lease("jobs", "a", "w1", now=18)

    current = queue.read("jobs", "a")
    assert (current.worker_id, current.attempt_count, current.lease_expires_at, current.changed_sequence) == (
        reassigned.worker_id,
        reassigned.attempt_count,
        reassigned.lease_expires_at,
        reassigned.changed_sequence,
    )


@pytest.mark.unit
def test_a_new_attempt_with_the_same_worker_id_fences_checkpoint_writes(open_coordination):
    queue = open_coordination().queue
    queue.append(job("a"))
    first = queue.take("jobs", "worker", ["training_job:tinker"])
    queue.requeue_expired(now=time.time() + 60)
    replacement = queue.take("jobs", "worker", ["training_job:tinker"])

    with pytest.raises(WorkLeaseLostError):
        queue.write_metadata("jobs", "a", "content", "first", lease=("worker", first.attempt_count))
    with pytest.raises(WorkLeaseLostError):
        queue.append_progress("jobs", "a", "worker", {"content": "first"}, number=0, attempt=first.attempt_count)
    assert queue.read("jobs", "a").metadata == {}
    assert queue.read("jobs", "a").progress == ()

    saved = queue.write_metadata("jobs", "a", "content", "new", lease=("worker", replacement.attempt_count))
    assert saved.metadata["content"] == "new"
    saved = queue.append_progress(
        "jobs", "a", "worker", {"content": "new"}, number=0, attempt=replacement.attempt_count
    )
    assert saved.progress == ({"content": "new"},)


@pytest.mark.unit
def test_a_retryable_failure_requeues_until_the_attempts_run_out(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.append(job("a"), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=0)
    retried = queue.fail("jobs", "a", "w", "transient", retryable=True, now=1)
    assert retried.state == "queued" and retried.error == "transient"
    queue.take("jobs", "w", ["training_job:tinker"], now=2)
    assert queue.fail("jobs", "a", "w", "transient", retryable=True, now=3).state == "failed"


@pytest.mark.unit
def test_a_request_reuses_the_queued_item_of_its_key_and_ended_items_go_after_the_retention(open_coordination):
    queue = open_coordination().queue
    first = queue.request("data", "s", "record_copy", "record_copy:s", {}, now=0)
    assert first.serial_key == "record_copy:s" and first.id.startswith("record_copy:s:")
    # A queued item covers every request while it waits.
    assert queue.request("data", "s", "record_copy", "record_copy:s", {}, now=1).id == first.id
    queue.take("data", "w1", ["record_copy"], now=2)
    second = queue.request("data", "s", "record_copy", "record_copy:s", {}, now=3)
    assert second.id != first.id and second.state == "queued"
    queue.finish("data", first.id, "w1", {}, now=4)

    assert queue.delete_final(finished_before=4) == 0
    assert queue.delete_final(finished_before=5) == 1
    assert queue.read("data", first.id) is None and queue.read("data", second.id) is not None
    # The next request after a deletion gets a new ID: no ID is used twice.
    queue.take("data", "w1", ["record_copy"], now=6)
    third = queue.request("data", "s", "record_copy", "record_copy:s", {}, now=7)
    assert third.id not in (first.id, second.id)


@pytest.mark.unit
def test_a_request_makes_the_queued_item_of_its_key_available_no_later_than_it_asks(open_coordination):
    queue = open_coordination().queue
    later = queue.request("samples", "s", "record_selection", "record_selection:s", {}, available_at=60, now=0)
    assert later.available_at == 60
    # A request for a later time leaves it; a request for now moves it to now.
    assert queue.request("samples", "s", "record_selection", "record_selection:s", {}, available_at=90, now=1) == later
    moved = queue.request("samples", "s", "record_selection", "record_selection:s", {}, now=2)
    assert (moved.id, moved.available_at) == (later.id, 2)
    assert queue.take("samples", "w", ["record_selection"], now=2).id == later.id


class FailingWorker(QueueWorker):
    """A worker whose every item ends with ``error``."""

    def __init__(self, queue: WorkQueue, error: Exception, worker_id: str = "w") -> None:
        super().__init__(queue, "jobs", ["training_job:tinker"], worker_id=worker_id)
        self.error = error

    def run_item(self, item, lease_lost):
        raise self.error


class CapacityWorker(QueueWorker):
    """A worker whose items with an ID in ``refused`` find no capacity, and whose other items finish."""

    def __init__(self, queue: WorkQueue, refused: frozenset[str]) -> None:
        super().__init__(queue, "jobs", ["training_job:tinker"], worker_id="w")
        self.refused = refused

    def run_item(self, item, lease_lost):
        if item.id in self.refused:
            raise ItemDeferredError("waiting for sandbox capacity")
        return {}


def warnings_of(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "attempts", "state"),
    [
        (ItemFailedError("bad input"), 2, "failed"),
        (ValueError("bad input"), 2, "failed"),
        (PluginConfigError("bad plugin config"), 2, "failed"),
        (RuntimeConfigError("bad runtime config"), 2, "queued"),
        (RuntimeError("engine down"), 2, "queued"),
        (RuntimeError("engine down"), 1, "failed"),
        (WorkerStoppedError(), 1, "queued"),
    ],
)
def test_a_worker_logs_one_warning_for_each_item_that_fails_for_good(caplog, error, attempts, state):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": attempts})
        queue.append(job("a"))
        with caplog.at_level(logging.WARNING, logger="reef.execution.workers"):
            assert FailingWorker(queue, error).run_once()
        item = queue.read("jobs", "a")
        assert item.state == state
        assert warnings_of(caplog) == (
            [f"work item jobs/a of scenario s failed: {item.error}"] if state == "failed" else []
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "stored"),
    [
        (
            RuntimeError("modal run failed:\n  the export failed\n"),
            "RuntimeError: modal run failed:\n  the export failed",
        ),
        (RuntimeError(), "RuntimeError:"),
        (ItemFailedError(" "), "the work item failed"),
    ],
)
def test_a_worker_stores_an_error_message_with_surrounding_spaces(error, stored):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": 1})
        queue.append(job("a"))
        assert FailingWorker(queue, error).run_once()
        item = queue.read("jobs", "a")
        assert (item.state, item.error) == ("failed", stored)


@pytest.mark.unit
def test_a_worker_logs_an_expired_item_that_has_no_attempts_left(caplog):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10, max_attempts={"jobs": 1})
        queue.append(job("a"), now=0)
        queue.take("jobs", "stopped-worker", ["training_job:tinker"], now=0)
        with caplog.at_level(logging.WARNING, logger="reef.execution.workers"):
            assert not FailingWorker(queue, ItemFailedError("unused")).run_once()
        assert queue.read("jobs", "a").state == "failed"
        assert warnings_of(caplog) == ["work item jobs/a of scenario s failed: the worker's lease expired"]


class WaitingWorker(QueueWorker):
    """A worker whose items run until ``release`` is set, or until the worker stops."""

    def __init__(self, queue: WorkQueue, slots: int) -> None:
        super().__init__(queue, "jobs", ["training_job:tinker"], worker_id="w", poll_seconds=0.01, slots=slots)
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.running: set[str] = set()
        self.most_running = 0

    def run_item(self, item, lease_lost):
        with self.lock:
            self.running.add(item.id)
            self.most_running = max(self.most_running, len(self.running))
        try:
            while not self.release.wait(0.01):
                if self.stopped.is_set():
                    raise WorkerStoppedError
            return {}
        finally:
            with self.lock:
                self.running.discard(item.id)


@pytest.mark.unit
def test_a_worker_runs_at_most_its_slots_of_items_at_once(tmp_path):
    with closing(SQLiteCoordinationDatabase(tmp_path / "c")) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        for item_id in "abcde":
            queue.append(job(item_id, serial_key=None))
        worker = WaitingWorker(queue, slots=3)
        worker.start()
        try:
            wait_until(lambda: len(worker.running) == 3, timeout_seconds=5)
            assert [queue.read("jobs", item_id).state for item_id in "de"] == ["queued", "queued"]
            worker.release.set()
            wait_until(
                lambda: all(queue.read("jobs", item_id).state == "finished" for item_id in "abcde"), timeout_seconds=5
            )
        finally:
            worker.stop()
        assert worker.most_running == 3


@pytest.mark.unit
def test_a_stopped_worker_returns_every_item_in_progress_to_the_queue(tmp_path):
    with closing(SQLiteCoordinationDatabase(tmp_path / "c")) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        for item_id in "ab":
            queue.append(job(item_id, serial_key=None))
        worker = WaitingWorker(queue, slots=2)
        worker.start()
        wait_until(lambda: len(worker.running) == 2, timeout_seconds=5)
        worker.stop()
        assert worker.item_threads == []
        assert [queue.read("jobs", item_id).state for item_id in "ab"] == ["queued", "queued"]


@pytest.mark.unit
def test_a_worker_gives_an_item_without_capacity_back_and_waits_longer_each_time(monkeypatch):
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={"jobs": 3})
    # The test sets the clock that the queue reads too.
    queue.append(job("a"), now=0)
    worker = FailingWorker(queue, ItemDeferredError("waiting for sandbox capacity"))
    worker.max_deferrals = 2
    clock = {"wall": 1000.0}
    monkeypatch.setattr(workers.time, "time", lambda: clock["wall"])
    for delay_seconds in (30, 60):
        assert worker.run_once()
        item = queue.read("jobs", "a")
        assert (item.state, item.error) == ("queued", "waiting for sandbox capacity")
        assert item.available_at == clock["wall"] + delay_seconds
        # No worker takes the item while it waits.
        assert not worker.run_once()
        clock["wall"] += delay_seconds
    # The give-backs did not use the item's attempts; after ``max_deferrals`` of them the error counts as a failure.
    assert worker.run_once()
    item = queue.read("jobs", "a")
    assert (item.attempt_count, item.stopped_attempt_count, item.deferred_attempt_count) == (3, 2, 2)
    assert item.state == "queued" and item.error == "waiting for sandbox capacity (given back 2 times)"


@pytest.mark.unit
def test_a_worker_runs_other_items_while_an_item_without_capacity_waits():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database)
        # A large item finds no capacity, and a small one after it can run.
        queue.append(job("large", serial_key=None))
        queue.append(job("small", serial_key=None))
        worker = CapacityWorker(queue, frozenset({"large"}))
        assert worker.run_once()
        waiting = queue.read("jobs", "large")
        assert waiting.state == "queued"
        # The same worker takes the small item at once: only the large item waits.
        assert worker.run_once()
        assert queue.read("jobs", "small").state == "finished"
        assert not worker.run_once()
        queue.append(job("newer", serial_key=None))
        assert queue.take("jobs", "w", ["training_job:tinker"], now=waiting.available_at).id == "large"


@pytest.mark.unit
@pytest.mark.parametrize("restarts", [5, 24])
def test_give_backs_of_a_stopping_worker_do_not_lengthen_the_first_wait_for_capacity(monkeypatch, restarts):
    clock = {"wall": 1000.0}
    monkeypatch.setattr(workers.time, "time", lambda: clock["wall"])
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": 1})
        queue.append(job("a"))
        for restart in range(restarts):
            queue.take("jobs", f"stopped-{restart}", ["training_job:tinker"])
            queue.give_back("jobs", "a", f"stopped-{restart}")
        assert FailingWorker(queue, ItemDeferredError("waiting for sandbox capacity")).run_once()
        item = queue.read("jobs", "a")
        # The first refusal of capacity waits the first delay, and the item keeps its one attempt.
        assert (item.state, item.error) == ("queued", "waiting for sandbox capacity")
        assert item.available_at == clock["wall"] + 30
        assert (item.stopped_attempt_count, item.deferred_attempt_count) == (restarts + 1, 1)
        assert queue.has_attempts_left(item)


@pytest.mark.unit
def test_the_wait_for_capacity_of_an_item_grows_across_workers_and_restarts(monkeypatch):
    clock = {"wall": 1000.0}
    monkeypatch.setattr(workers.time, "time", lambda: clock["wall"])
    refused = ItemDeferredError("waiting for sandbox capacity")
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database)
        queue.append(job("a"))
        assert FailingWorker(queue, refused, worker_id="w1").run_once()
        clock["wall"] += 30
        # A worker that stops between the two refusals gives the item back, which counts as no refusal.
        queue.take("jobs", "stopping", ["training_job:tinker"])
        queue.give_back("jobs", "a", "stopping")
        assert FailingWorker(queue, refused, worker_id="w2").run_once()
        item = queue.read("jobs", "a")
        assert item.available_at == clock["wall"] + 60
        assert (item.attempt_count, item.stopped_attempt_count, item.deferred_attempt_count) == (3, 3, 2)


@pytest.mark.unit
@pytest.mark.parametrize("slots", [0, True, 1.5])
def test_a_worker_needs_a_positive_integer_of_slots(slots):
    with closing(SQLiteCoordinationDatabase(None)) as database, pytest.raises(ValueError, match="slots"):
        WaitingWorker(SQLWorkQueue(database, lease_seconds=10), slots=slots)


@pytest.mark.unit
def test_an_item_waits_for_its_available_time_and_readers_follow_changes_in_order(open_coordination):
    queue = open_coordination().queue
    queue.append(NewWork("samples", "pass-1", "s", "record_selection", {}, available_at=50), now=0)
    assert queue.take("samples", "w", ["record_selection"], now=10) is None
    assert queue.take("samples", "w", ["record_selection"], now=50).id == "pass-1"

    changes = queue.read_changes("samples", 0)
    assert [item.changed_sequence for item in changes] == sorted(item.changed_sequence for item in changes)
    assert queue.read_changes("samples", changes[-1].changed_sequence) == ()


@pytest.mark.unit
def test_an_item_records_when_a_worker_first_took_it_and_when_it_ended(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    added = queue.append(job("a"), now=1)
    assert (added.created_at, added.started_at, added.ended_at) == (1, None, None)
    queue.take("jobs", "w1", ["training_job:tinker"], now=2)
    retried = queue.fail("jobs", "a", "w1", "transient", retryable=True, now=3)
    assert (retried.state, retried.started_at, retried.ended_at) == ("queued", 2, None)
    # A second take keeps the time of the first one.
    assert queue.take("jobs", "w2", ["training_job:tinker"], now=4).started_at == 2
    done = queue.finish("jobs", "a", "w2", {}, now=5)
    assert (done.started_at, done.ended_at) == (2, 5)

    queue.append(job("b", serial_key=None), now=6)
    queue.take("jobs", "w1", ["training_job:tinker"], now=7)
    assert queue.requeue_expired(now=100)[0].ended_at is None
    queue.take("jobs", "w1", ["training_job:tinker"], now=101)
    assert queue.requeue_expired(now=200)[0].ended_at == 200
    queue.append(job("c", serial_key=None), now=8)
    cancelled = queue.cancel("jobs", "c", "cancelled")
    assert cancelled.started_at is None and cancelled.ended_at is not None


@pytest.mark.unit
def test_a_reader_sets_metadata_on_any_item_and_an_equal_value_changes_nothing(open_coordination):
    queue = open_coordination().queue
    assert queue.write_metadata("jobs", "missing", "experiment", {"run_id": "r"}) is None
    queue.append(job("a"), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=1)
    done = queue.finish("jobs", "a", "w", {"ok": True}, now=2)

    # A final item takes metadata too; its state, result and times stay.
    noted = queue.write_metadata("jobs", "a", "experiment", {"run_id": "r"})
    assert noted.metadata == {"experiment": {"run_id": "r"}}
    assert (noted.state, noted.result, noted.ended_at) == ("finished", {"ok": True}, 2)
    assert noted.changed_sequence > done.changed_sequence
    assert queue.write_metadata("jobs", "a", "experiment", {"run_id": "r"}) == noted
    assert queue.write_metadata("jobs", "a", "other", 1).metadata == {"experiment": {"run_id": "r"}, "other": 1}
    # Retention counts from the end, not from the last metadata change.
    assert queue.delete_final(finished_before=3) == 1


@pytest.mark.unit
def test_a_work_item_has_a_worker_exactly_while_it_is_held():
    with pytest.raises(ValueError, match="worker and a lease"):
        WorkItem("jobs", "a", "s", "k", {}, state="running")
    with pytest.raises(ValueError, match="result exactly"):
        WorkItem("jobs", "a", "s", "k", {}, state="finished")
    with pytest.raises(ValueError, match="ended_at exactly"):
        WorkItem("jobs", "a", "s", "k", {}, state="finished", result={})
    with pytest.raises(ValueError, match="no started_at"):
        WorkItem("jobs", "a", "s", "k", {}, started_at=1.0)


@pytest.mark.unit
def test_served_models_writes_need_the_current_control_lease_number(open_coordination):
    coordination = open_coordination(lease_seconds=10)
    lease, table = coordination.lease, coordination.served
    first = lease.acquire("control-a", now=0)
    assert first.number == 1 and lease.acquire("control-b", now=5) is None
    assert lease.extend("control-a", now=6).expires_at == 16
    taken_over = lease.acquire("control-b", now=17)
    assert taken_over.instance_id == "control-b" and taken_over.number == 2
    assert lease.extend("control-a", now=17) is None and lease.current() == taken_over
    # A stopped instance releases the lease; the next one takes it at once, with the next number.
    assert not lease.release("control-a", now=18)
    assert lease.release("control-b", now=18)
    assert lease.acquire("control-a", now=19).number == 3

    served = ServedArtifact("artifact-1", "http://engine:8000/", "artifact-1")
    assert served.url == "http://engine:8000"
    before = table.changed_sequence()
    with pytest.raises(ControlLeaseConflictError):
        table.write(ScenarioModel("s"), served, control_lease_number=taken_over.number)
    table.write(ScenarioModel("s"), served, control_lease_number=3)
    assert table.read(ScenarioModel("s")) == served and table.read_scenario("s") == {"active": served}
    assert table.read(ScenarioModel("s", "judge")) is None
    assert table.is_referenced(served) and table.changed_sequence() > before

    table.schedule_stop(ScheduledStop(served, 30), control_lease_number=3)
    assert table.due_stops(29) == () and table.due_stops(30) == (ScheduledStop(served, 30),)
    table.delete_stop(served, control_lease_number=3)
    assert table.due_stops(100) == ()
    assert table.delete(ScenarioModel("s"), control_lease_number=3) == served
    assert not table.is_referenced(served)


@pytest.mark.unit
def test_control_takeover_waits_for_the_serving_request_transaction(open_coordination, monkeypatch):
    coordination = open_coordination(lease_seconds=10)
    lease, table = coordination.lease, coordination.served
    first = lease.acquire("control-a", now=0)
    checked, release, taking, taken = (threading.Event() for _ in range(4))
    check = table.checked

    def pause_after_check(connection, number):
        check(connection, number)
        checked.set()
        assert release.wait(5)

    def take_over():
        taking.set()
        value = lease.acquire("control-b", now=11)
        taken.set()
        return value

    monkeypatch.setattr(table, "checked", pause_after_check)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writing = executor.submit(table.request, "s", {"artifact_id": "a"}, control_lease_number=first.number, now=1)
        try:
            assert checked.wait(2)
            takeover = executor.submit(take_over)
            assert taking.wait(2)
            blocked = not taken.wait(0.05)
        finally:
            release.set()
        assert writing.result(2).state == "queued"
        assert takeover.result(2).number == first.number + 1
        assert blocked
    with pytest.raises(ControlLeaseConflictError):
        table.request("s", {"artifact_id": "b"}, control_lease_number=first.number, now=12)


@pytest.mark.unit
def test_two_workers_never_hold_one_item(open_coordination):
    queue = open_coordination().queue
    for index in range(20):
        queue.append(NewWork("evaluations", f"rollout-{index}", "s", "evaluation_rollout", {"k": index}), now=0)
    taken: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        while (item := queue.take("evaluations", name, ["evaluation_rollout"], now=1)) is not None:
            with lock:
                taken.append(item.id)

    threads = [threading.Thread(target=worker, args=(f"w{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(taken) == sorted(f"rollout-{index}" for index in range(20))


@pytest.mark.unit
def test_a_served_artifact_is_reachable_and_names_one_model():
    with pytest.raises(ValueError, match="HTTP"):
        ServedArtifact("a", "ftp://engine", "a")
    with pytest.raises(ValueError, match="credentials"):
        ServedArtifact("a", "http://user:secret@engine", "a")
    assert ServedArtifact.from_json(ServedArtifact("a", "http://e", "m").to_json()) == ServedArtifact(
        "a", "http://e", "m"
    )


@pytest.mark.unit
def test_serving_routes_and_the_finished_request_commit_together_and_survive_retention(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    assert table.current_request("s") is None
    payload = {"artifact_id": "a"}
    requested = table.request("s", payload, control_lease_number=lease.number, now=0)
    assert requested.state == "queued" and requested.started_at is None
    assert table.request("s", payload, control_lease_number=lease.number, now=1).id == requested.id
    assert table.read_scenario("s") == {}
    item = queue.take(SERVING_QUEUE, "inference", [SERVING_KIND], now=1)
    served = ServedArtifact("a", "http://engine-a", "model-a")
    done = table.complete_request(
        item, "inference", {"active": served, "a": served}, serving_config={}, stop_delay_seconds=10, now=2
    )
    assert done.state == "finished"
    assert done.result == {"active": served.to_json(), "a": served.to_json()}
    assert table.current_request("s").id == done.id
    assert table.read_scenario("s") == {"active": served, "a": served}
    assert queue.delete_final(finished_before=100) == 0
    assert table.current_request("s").state == "finished"


@pytest.mark.unit
def test_new_selection_waits_for_held_engine_work_but_rejects_its_obsolete_routes(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    old = queue.take(SERVING_QUEUE, "first", [SERVING_KIND], now=1)
    waiting = table.request("s", {"artifact_id": "b"}, control_lease_number=lease.number, now=2)
    newest = table.request("s", {"artifact_id": "c"}, control_lease_number=lease.number, now=3)
    assert queue.read(SERVING_QUEUE, waiting.id).state == "failed"
    assert queue.read(SERVING_QUEUE, old.id).state == "assigned"
    assert queue.take(SERVING_QUEUE, "second", [SERVING_KIND], now=4) is None
    obsolete = table.complete_request(
        old,
        "first",
        {"active": ServedArtifact("a", "http://old", "a")},
        serving_config={},
        stop_delay_seconds=10,
        now=4,
    )
    assert obsolete.state == "finished" and obsolete.result == {}
    assert table.read_scenario("s") == {}
    assert table.due_stops(14) == (ScheduledStop(ServedArtifact("a", "http://old", "a"), 14),)
    assert queue.take(SERVING_QUEUE, "second", [SERVING_KIND], now=5).id == newest.id
    assert queue.delete_final(finished_before=100) == 2
    assert table.current_request("s").id == newest.id


@pytest.mark.unit
def test_withdrawal_removes_routes_and_cannot_be_undone_by_a_running_start(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    first = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    before = table.changed_sequence()
    table.complete_request(
        first, "worker", {"active": served}, serving_config={"temperature": 0.3}, stop_delay_seconds=10, now=2
    )
    assert table.read_serving_config("s") == {"temperature": 0.3}
    assert table.changed_sequence() > before
    # An artifact can have a weight before it has a route.
    before_weights = table.changed_sequence()
    table.write_traffic_weights("s", {"a": 50, "b": 25, "c": 25}, control_lease_number=lease.number)
    assert table.read_traffic_weights("s") == {"a": 50, "b": 25, "c": 25}
    assert table.changed_sequence() > before_weights
    table.write_traffic_weights("s", {"active": 100, "absent": 0}, control_lease_number=lease.number)
    assert table.read_traffic_weights("s") == {"active": 100, "absent": 0}
    table.request("s", {"artifact_id": "b"}, control_lease_number=lease.number, now=3)
    running = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=4)
    withdrawn = table.request("s", {}, control_lease_number=lease.number, stop_delay_seconds=10, now=5)
    assert withdrawn.state == "finished" and withdrawn.payload == {}
    assert table.read_scenario("s") == {} and table.read_serving_config("s") == {}
    assert table.read_traffic_weights("s") == {}
    assert table.due_stops(14) == ()
    assert table.due_stops(15) == (ScheduledStop(served, 15),)
    table.complete_request(
        running, "worker", {"active": served}, serving_config={"temperature": 0.9}, stop_delay_seconds=10, now=6
    )
    assert table.read_scenario("s") == {} and table.read_serving_config("s") == {}
    again = table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=7)
    assert again.id not in {first.id, running.id, withdrawn.id}
    assert table.current_request("s").id == again.id


@pytest.mark.unit
def test_serving_completion_rejects_old_attempt_and_rolls_back_bad_routes(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    old_attempt = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    queue.requeue_expired(now=12)
    current = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=13)
    with pytest.raises(WorkLeaseLostError, match="later attempt"):
        table.complete_request(
            old_attempt, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=14
        )
    before = table.changed_sequence()
    with pytest.raises(ValueError, match="model name"):
        table.complete_request(current, "worker", {"": served}, serving_config={}, stop_delay_seconds=10, now=14)
    assert table.changed_sequence() == before and table.read_scenario("s") == {}
    assert table.current_request("s").state == "assigned"
    table.complete_request(current, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=15)
    assert table.current_request("s").state == "finished"


@pytest.mark.unit
def test_failed_serving_retries_only_on_explicit_request_and_old_control_cannot_select(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    payload = {"artifact_id": "a"}
    table.request("s", payload, control_lease_number=lease.number, now=0)
    item = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    queue.fail(SERVING_QUEUE, item.id, "worker", "cannot start", retryable=False, now=2)
    assert table.request("s", payload, control_lease_number=lease.number, now=3).state == "failed"
    with pytest.raises(ValueError, match="automatic_retries"):
        table.request("s", payload, control_lease_number=lease.number, retry_failed=True, automatic_retries=-1)
    retried = table.request(
        "s", payload, control_lease_number=lease.number, retry_failed=True, automatic_retries=2, now=4
    )
    assert retried.id != item.id and retried.state == "queued"
    assert retried.metadata == {AUTOMATIC_RETRIES: 2}
    # Only a retry of the same payload records the count.
    other = table.request("s", {"artifact_id": "b"}, control_lease_number=lease.number, automatic_retries=3, now=5)
    assert other.metadata == {}
    stores.lease.acquire("new-control", now=11)
    with pytest.raises(ControlLeaseConflictError):
        table.request("s", {}, control_lease_number=lease.number, now=12)
    assert table.current_request("s").id == other.id


@pytest.mark.unit
def test_replaced_routes_stop_only_when_unreferenced_and_cleanup_keeps_a_newer_stop(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    for scenario in ("s", "t"):
        table.request(scenario, {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
        item = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
        table.complete_request(item, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=2)
    table.request("s", {}, control_lease_number=lease.number, stop_delay_seconds=10, now=3)
    assert table.due_stops(100) == ()
    table.request("t", {}, control_lease_number=lease.number, stop_delay_seconds=10, now=4)
    stop = table.due_stops(14)[0]
    queue.append(NewWork(SERVING_QUEUE, "stop-1", "t", STOP_SERVING_KIND, {}, serial_key=SERVING_QUEUE), now=14)
    item = queue.take(SERVING_QUEUE, "worker", [STOP_SERVING_KIND], now=14)
    table.schedule_stop(ScheduledStop(served, 30), control_lease_number=lease.number)
    table.complete_stop(item, "worker", stop, now=15)
    assert table.due_stops(30) == (ScheduledStop(served, 30),)
    assert table.current_request("t").payload == {}


@pytest.mark.unit
def test_a_new_inference_owner_recovers_only_completed_current_selections(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    ready = table.request("ready", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    running = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    table.complete_request(running, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=2)
    failed = table.request("failed", {"artifact_id": "b"}, control_lease_number=lease.number, now=0)
    queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    queue.fail(SERVING_QUEUE, failed.id, "worker", "cannot load", retryable=False, now=2)
    queued = table.request("queued", {"artifact_id": "c"}, control_lease_number=lease.number, now=3)
    withdrawn = table.request("withdrawn", {}, control_lease_number=lease.number, now=3)
    previous = ServedArtifact("previous-c", "http://engine", "previous-c")
    table.write(ScenarioModel("queued"), previous, control_lease_number=lease.number)
    table.write(ScenarioModel("failed"), previous, control_lease_number=lease.number)
    queue.append(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=3)
    owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=3)

    table.recover_completed(owner, "worker", stop_delay_seconds=10, kept=(), now=4)
    recovery = table.current_request("ready")
    assert recovery.id != ready.id and recovery.payload == ready.payload and recovery.state == "queued"
    assert recovery.metadata == {"recovery": True}
    assert table.read_scenario("ready") == table.read_scenario("queued") == table.read_scenario("failed") == {}
    assert set(table.due_stops(14)) == {ScheduledStop(served, 14), ScheduledStop(previous, 14)}
    assert table.current_request("queued") == queued
    assert table.current_request("withdrawn") == withdrawn
    assert table.current_request("failed").state == "failed"
    table.recover_completed(owner, "worker", stop_delay_seconds=10, kept=(), now=5)
    assert table.current_request("ready") == recovery
    with pytest.raises(WorkLeaseLostError):
        table.recover_completed(owner, "worker", stop_delay_seconds=10, kept=(), now=14)
    assert table.current_request("ready") == recovery


@pytest.mark.unit
def test_a_new_owner_keeps_the_routes_whose_engines_still_serve_until_it_serves_again(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    live, gone = ServedArtifact("a", "http://live", "a"), ServedArtifact("b", "http://gone", "b")
    for scenario, served in (("live", live), ("gone", gone)):
        table.request(scenario, {"artifact_id": served.artifact_id}, control_lease_number=lease.number, now=0)
        running = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
        table.complete_request(running, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=2)
    queue.append(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=3)
    owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=3)

    table.recover_completed(owner, "worker", stop_delay_seconds=10, kept=(live,), now=4)
    assert table.current_request("live").metadata == table.current_request("gone").metadata == {"recovery": True}
    assert table.read_scenario("live") == {"active": live}
    assert table.read_scenario("gone") == {}
    assert table.due_stops(14) == (ScheduledStop(gone, 14),)


@pytest.mark.unit
@pytest.mark.parametrize("recreate_owner", [False, True])
def test_startup_and_cleanup_completion_require_the_current_runtime_owner(open_coordination, recreate_owner):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    queue.append(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=0)
    owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=0)
    table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    startup = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    queue.extend_lease(SERVING_QUEUE, startup.id, "worker", now=9)
    with pytest.raises(WorkLeaseLostError):
        table.complete_request(
            startup, "worker", {"active": served}, owner=owner, serving_config={}, stop_delay_seconds=10, now=11
        )
    assert table.read_scenario("s") == {} and table.current_request("s").state == "assigned"
    if recreate_owner:
        queue.cancel(INFERENCE_OWNER_QUEUE, "owner", "shutdown")
        queue.delete_final(finished_before=time.time() + 1)
        queue.append(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=11)
    else:
        queue.requeue_expired(now=11)
    current_owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=12)
    with pytest.raises(WorkLeaseLostError, match="later attempt"):
        table.complete_request(
            startup, "worker", {"active": served}, owner=owner, serving_config={}, stop_delay_seconds=10, now=13
        )
    table.complete_request(
        startup, "worker", {"active": served}, owner=current_owner, serving_config={}, stop_delay_seconds=10, now=13
    )

    stop = ScheduledStop(ServedArtifact("old", "http://engine", "old"), 14)
    table.schedule_stop(stop, control_lease_number=lease.number)
    queue.append(NewWork(SERVING_QUEUE, "stop", "s", STOP_SERVING_KIND, serial_key=SERVING_QUEUE), now=14)
    cleanup = queue.take(SERVING_QUEUE, "worker", [STOP_SERVING_KIND], now=14)
    with pytest.raises(WorkLeaseLostError, match="later attempt"):
        table.complete_stop(cleanup, "worker", stop, owner=owner, now=15)
    assert table.due_stops(15) == (stop,)
    table.complete_stop(cleanup, "worker", stop, owner=current_owner, now=15)
    assert table.due_stops(15) == ()


@pytest.mark.unit
def test_replacing_an_artifact_at_the_same_endpoint_does_not_stop_that_endpoint(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    previous = ServedArtifact("old", "http://engine", "shared-model")
    current = ServedArtifact("new", previous.url, previous.model_name)
    for index, served in enumerate((previous, current)):
        table.request("s", {"artifact_id": served.artifact_id}, control_lease_number=lease.number, now=index)
        item = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=index)
        table.complete_request(item, "worker", {"active": served}, serving_config={}, stop_delay_seconds=10, now=index)
    assert table.read_scenario("s") == {"active": current}
    assert table.is_referenced(previous) and table.due_stops(100) == ()
    table.request("s", {}, control_lease_number=lease.number, stop_delay_seconds=10, now=2)
    assert not table.is_referenced(previous) and table.due_stops(12) == (ScheduledStop(current, 12),)


@pytest.mark.unit
def test_queue_pages_and_counts_are_scoped_and_newest_first(open_coordination):
    queue = open_coordination().queue
    first = queue.append(job("a", serial_key=None), now=0)
    second = queue.append(job("b", serial_key=None), now=0)
    queue.append(job("other", scenario="other", serial_key=None), now=0)
    queue.cancel("jobs", "a", "stopped")
    assert [item.id for item in queue.page("jobs", scenario="s", limit=1)] == ["b"]
    assert [item.id for item in queue.page("jobs", scenario="s", before_sequence=second.created_sequence)] == ["a"]
    assert queue.page("jobs", scenario="s", before_sequence=first.created_sequence) == ()
    assert [item.id for item in queue.page("jobs", scenario="s", state="failed")] == ["a"]
    assert queue.counts("jobs", scenario="s") == {"queued": 1, "assigned": 0, "running": 0, "finished": 0, "failed": 1}
    for kwargs in ({"state": "missing"}, {"limit": 0}, {"before_sequence": 0}):
        with pytest.raises(ValueError):
            queue.page("jobs", scenario="s", **kwargs)


@pytest.mark.unit
def test_the_log_of_each_attempt_reads_in_pages_and_goes_with_its_item(open_coordination):
    from reef.storage.coordination.work_queue import WorkLogLine

    queue = open_coordination().queue
    queue.append(job("a"), now=0)
    claim = queue.take("jobs", "w", ["training_job:tinker"], now=1)
    queue.append_log("jobs", "a", 1, [WorkLogLine(n, 1.0 + n, "INFO", "reef.test", f"line {n}") for n in range(3)])
    queue.append_log("jobs", "a", 2, [WorkLogLine(0, 9.0, "ERROR", "reef.test", "second attempt")])

    first = queue.read_log("jobs", "a", 1, limit=2)
    assert [line.message for line in first] == ["line 0", "line 1"]
    assert [line.number for line in queue.read_log("jobs", "a", 1, after=first[-1].number)] == [2]
    assert [line.level for line in queue.read_log("jobs", "a", 2)] == ["ERROR"]
    with pytest.raises(ValueError):
        queue.read_log("jobs", "a", 1, limit=0)

    queue.finish("jobs", "a", "w", {}, now=2)
    assert queue.delete_final(finished_before=3) == 1
    queue.append(job("a"), now=4)
    queue.append_log("jobs", "a", 1, first, created_sequence=claim.created_sequence)
    assert queue.read_log("jobs", "a", 1) == ()


def test_a_producer_queues_a_final_item_again_and_it_gets_one_more_attempt(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 1}).queue
    queue.append(job("a"), now=0)
    queue.append(job("b", serial_key=None), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=1)
    failed = queue.fail("jobs", "a", "w", "the sandbox stopped", retryable=False, now=2)

    again = queue.retry("jobs", "a", now=5)
    assert (again.state, again.error, again.ended_at, again.attempt_count) == (
        "queued",
        "the sandbox stopped",
        None,
        1,
    )
    assert (again.available_at, again.started_at, again.payload) == (5, 1, failed.payload)
    assert again.changed_sequence > failed.changed_sequence
    # The queue allows one attempt, and the item still gets the one that its producer asked for.
    assert queue.take("jobs", "w", ["training_job:tinker"], now=6).attempt_count == 2
    assert queue.fail("jobs", "a", "w", "transient", retryable=True, now=7).state == "failed"

    queue.take("jobs", "w", ["training_job:tinker"], now=8)
    queue.finish("jobs", "b", "w", {"score": 1}, now=9)
    rerun = queue.retry("jobs", "b", now=10)
    assert (rerun.state, rerun.result, rerun.error, rerun.ended_at) == ("queued", None, None, None)
    # An item that is not final stays as it is, and an unknown item is None.
    assert queue.retry("jobs", "b", now=11) == rerun
    assert queue.retry("jobs", "missing") is None


@pytest.mark.parametrize("recreate", [False, True])
def test_old_claim_cannot_mutate_an_item_retaken_by_the_same_worker(open_coordination, recreate, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 6)
    queue = open_coordination().queue
    work = NewWork("jobs", "j", "s", "train", {})
    queue.append(work, now=0)
    old = queue.take("jobs", "worker", ["train"], now=1)
    if recreate:
        queue.finish("jobs", "j", "worker", {}, claim=old, now=2)
        queue.delete_final(finished_before=3)
        queue.append(work, now=4)
    else:
        queue.give_back("jobs", "j", "worker", claim=old, now=2)
    current = queue.take("jobs", "worker", ["train"], now=5)
    for mutation in (queue.start, queue.extend_lease):
        with pytest.raises(WorkLeaseLostError):
            mutation("jobs", "j", "worker", claim=old, now=6)
    with pytest.raises(WorkLeaseLostError):
        queue.finish("jobs", "j", "worker", {}, claim=old, now=6)
    with pytest.raises(WorkLeaseLostError):
        queue.write_metadata("jobs", "j", "progress", 1, lease=old)
    with pytest.raises(WorkLeaseLostError):
        queue.give_back("jobs", "j", "worker", claim=current, attempt=current.attempt_count + 1, now=6)
    assert queue.read("jobs", "j") == current
    assert queue.finish("jobs", "j", "worker", {}, claim=current, now=6).state == "finished"
