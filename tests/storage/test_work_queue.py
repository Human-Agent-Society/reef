import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass

import pytest

from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.execution.workers import ItemFailed, QueueWorker, WorkerStopped
from reef.storage.control_lease import ControlLease
from reef.storage.served_models import (
    AUTOMATIC_RETRIES,
    INFERENCE_OWNER_QUEUE,
    SERVING_KIND,
    SERVING_QUEUE,
    STOP_SERVING_KIND,
    ControlLeaseConflict,
    ServedModelsWriter,
)
from reef.storage.sql_coordination import (
    PostgresCoordinationDatabase,
    SQLControlLease,
    SQLiteCoordinationDatabase,
    SQLServedModels,
    SQLWorkQueue,
)
from reef.storage.work_queue import NewWork, WorkConflict, WorkItem, WorkLeaseLost, WorkQueue


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


def job(
    item_id: str, *, scenario: str = "s", kind: str = "training_job:tinker", serial_key: str | None = "s"
) -> NewWork:
    return NewWork("jobs", item_id, scenario, kind, {"input": item_id}, serial_key=serial_key)


@pytest.mark.unit
def test_a_worker_takes_the_oldest_item_it_accepts_and_finishes_it_under_its_lease(open_coordination):
    queue = open_coordination().queue
    queue.add(job("a"), now=0)
    queue.add(job("b", kind="training_job:slime", serial_key=None), now=0)

    assert queue.take("jobs", "w1", ["training_job:other"], now=1) is None
    taken = queue.take("jobs", "w1", ["training_job:tinker", "training_job:slime"], now=1)
    assert taken is not None and taken.id == "a" and taken.state == "assigned"
    assert taken.worker_id == "w1" and taken.lease_expires_at == 11 and taken.attempt_count == 1

    queue.start("jobs", "a", "w1", now=2)
    queue.add_progress("jobs", "a", "w1", {"step": 1}, number=0, now=2)
    queue.add_progress("jobs", "a", "w1", {"step": 1}, number=0, now=2)  # a continuing worker reports again safely
    with pytest.raises(ValueError, match="skips"):
        queue.add_progress("jobs", "a", "w1", {"step": 3}, number=2, now=2)
    with pytest.raises(WorkLeaseLost):
        queue.finish("jobs", "a", "w2", {"ok": True})
    done = queue.finish("jobs", "a", "w1", {"ok": True}, now=3)
    assert done.state == "finished" and done.final and done.result == {"ok": True}
    assert done.worker_id is None and done.progress == ({"step": 1},)


@pytest.mark.unit
def test_adding_is_idempotent_by_id_and_rejects_a_different_payload(open_coordination):
    queue = open_coordination().queue
    first = queue.add(job("a"))
    assert queue.add(job("a")) == first
    with pytest.raises(WorkConflict):
        queue.add(NewWork("jobs", "a", "s", "training_job:tinker", {"input": "other"}))


@pytest.mark.unit
def test_items_with_one_serial_key_run_one_at_a_time_in_order(open_coordination):
    queue = open_coordination().queue
    queue.add(job("first"), now=0)
    queue.add(job("second"), now=0)
    queue.add(job("other", scenario="t", serial_key="t"), now=0)

    taken = [queue.take("jobs", "w", ["training_job:tinker"], now=1) for _ in range(3)]
    assert [item.id if item else None for item in taken] == ["first", "other", None]
    queue.finish("jobs", "first", "w", {}, now=2)
    assert queue.take("jobs", "w", ["training_job:tinker"], now=2).id == "second"


@pytest.mark.unit
def test_an_expired_lease_returns_the_item_and_a_late_worker_loses_it(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.add(job("a"), now=0)
    queue.take("jobs", "w1", ["training_job:tinker"], now=0)
    assert queue.extend_lease("jobs", "a", "w1", now=5).lease_expires_at == 15

    assert [item.state for item in queue.requeue_expired(now=16)] == ["queued"]
    with pytest.raises(WorkLeaseLost):
        queue.extend_lease("jobs", "a", "w1", now=16)
    again = queue.take("jobs", "w2", ["training_job:tinker"], now=17)
    assert again.worker_id == "w2" and again.attempt_count == 2
    # The last attempt expires too: no attempts are left, so the item fails.
    assert [item.state for item in queue.requeue_expired(now=100)] == ["failed"]


@pytest.mark.unit
def test_a_retryable_failure_requeues_until_the_attempts_run_out(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    queue.add(job("a"), now=0)
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
    assert queue.get("data", first.id) is None and queue.get("data", second.id) is not None
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

    def __init__(self, queue: WorkQueue, error: Exception) -> None:
        super().__init__(queue, "jobs", ["training_job:tinker"], worker_id="w")
        self.error = error

    def run_item(self, item, lease_lost):
        raise self.error


def warnings_of(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "attempts", "state"),
    [
        (ItemFailed("bad input"), 2, "failed"),
        (ValueError("bad input"), 2, "failed"),
        (RuntimeError("engine down"), 2, "queued"),
        (RuntimeError("engine down"), 1, "failed"),
        (WorkerStopped(), 1, "failed"),
    ],
)
def test_a_worker_logs_one_warning_for_each_item_that_fails_for_good(caplog, error, attempts, state):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": attempts})
        queue.add(job("a"))
        with caplog.at_level(logging.WARNING, logger="reef.execution.workers"):
            assert FailingWorker(queue, error).run_once()
        item = queue.get("jobs", "a")
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
        (ItemFailed(" "), "the work item failed"),
    ],
)
def test_a_worker_stores_an_error_message_with_surrounding_spaces(error, stored):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": 1})
        queue.add(job("a"))
        assert FailingWorker(queue, error).run_once()
        item = queue.get("jobs", "a")
        assert (item.state, item.error) == ("failed", stored)


@pytest.mark.unit
def test_a_worker_logs_an_expired_item_that_has_no_attempts_left(caplog):
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10, max_attempts={"jobs": 1})
        queue.add(job("a"), now=0)
        queue.take("jobs", "stopped-worker", ["training_job:tinker"], now=0)
        with caplog.at_level(logging.WARNING, logger="reef.execution.workers"):
            assert not FailingWorker(queue, ItemFailed("unused")).run_once()
        assert queue.get("jobs", "a").state == "failed"
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
                    raise WorkerStopped
            return {}
        finally:
            with self.lock:
                self.running.discard(item.id)


def wait_until(condition) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not become true in 5 s")
        time.sleep(0.01)


@pytest.mark.unit
def test_a_worker_runs_at_most_its_slots_of_items_at_once(tmp_path):
    with closing(SQLiteCoordinationDatabase(tmp_path / "c")) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        for item_id in "abcde":
            queue.add(job(item_id, serial_key=None))
        worker = WaitingWorker(queue, slots=3)
        worker.start()
        try:
            wait_until(lambda: len(worker.running) == 3)
            assert [queue.get("jobs", item_id).state for item_id in "de"] == ["queued", "queued"]
            worker.release.set()
            wait_until(lambda: all(queue.get("jobs", item_id).state == "finished" for item_id in "abcde"))
        finally:
            worker.stop()
        assert worker.most_running == 3


@pytest.mark.unit
def test_a_stopped_worker_returns_every_item_in_progress_to_the_queue(tmp_path):
    with closing(SQLiteCoordinationDatabase(tmp_path / "c")) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        for item_id in "ab":
            queue.add(job(item_id, serial_key=None))
        worker = WaitingWorker(queue, slots=2)
        worker.start()
        wait_until(lambda: len(worker.running) == 2)
        worker.stop()
        assert worker.item_threads == []
        assert [queue.get("jobs", item_id).state for item_id in "ab"] == ["queued", "queued"]


@pytest.mark.unit
@pytest.mark.parametrize("slots", [0, True, 1.5])
def test_a_worker_needs_a_positive_integer_of_slots(slots):
    with closing(SQLiteCoordinationDatabase(None)) as database, pytest.raises(ValueError, match="slots"):
        WaitingWorker(SQLWorkQueue(database, lease_seconds=10), slots=slots)


@pytest.mark.unit
def test_an_item_waits_for_its_available_time_and_readers_follow_changes_in_order(open_coordination):
    queue = open_coordination().queue
    queue.add(NewWork("samples", "pass-1", "s", "record_selection", {}, available_at=50), now=0)
    assert queue.take("samples", "w", ["record_selection"], now=10) is None
    assert queue.take("samples", "w", ["record_selection"], now=50).id == "pass-1"

    changes = queue.read_changes("samples", 0)
    assert [item.changed_sequence for item in changes] == sorted(item.changed_sequence for item in changes)
    assert queue.read_changes("samples", changes[-1].changed_sequence) == ()


@pytest.mark.unit
def test_an_item_records_when_a_worker_first_took_it_and_when_it_ended(open_coordination):
    queue = open_coordination(max_attempts={"jobs": 2}).queue
    added = queue.add(job("a"), now=1)
    assert (added.created_at, added.started_at, added.ended_at) == (1, None, None)
    queue.take("jobs", "w1", ["training_job:tinker"], now=2)
    retried = queue.fail("jobs", "a", "w1", "transient", retryable=True, now=3)
    assert (retried.state, retried.started_at, retried.ended_at) == ("queued", 2, None)
    # A second take keeps the time of the first one.
    assert queue.take("jobs", "w2", ["training_job:tinker"], now=4).started_at == 2
    done = queue.finish("jobs", "a", "w2", {}, now=5)
    assert (done.started_at, done.ended_at) == (2, 5)

    queue.add(job("b", serial_key=None), now=6)
    queue.take("jobs", "w1", ["training_job:tinker"], now=7)
    assert queue.requeue_expired(now=100)[0].ended_at is None
    queue.take("jobs", "w1", ["training_job:tinker"], now=101)
    assert queue.requeue_expired(now=200)[0].ended_at == 200
    queue.add(job("c", serial_key=None), now=8)
    cancelled = queue.cancel("jobs", "c", "cancelled")
    assert cancelled.started_at is None and cancelled.ended_at is not None


@pytest.mark.unit
def test_a_reader_sets_metadata_on_any_item_and_an_equal_value_changes_nothing(open_coordination):
    queue = open_coordination().queue
    assert queue.set_metadata("jobs", "missing", "experiment", {"run_id": "r"}) is None
    queue.add(job("a"), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=1)
    done = queue.finish("jobs", "a", "w", {"ok": True}, now=2)

    # A final item takes metadata too; its state, result and times stay.
    noted = queue.set_metadata("jobs", "a", "experiment", {"run_id": "r"})
    assert noted.metadata == {"experiment": {"run_id": "r"}}
    assert (noted.state, noted.result, noted.ended_at) == ("finished", {"ok": True}, 2)
    assert noted.changed_sequence > done.changed_sequence
    assert queue.set_metadata("jobs", "a", "experiment", {"run_id": "r"}) == noted
    assert queue.set_metadata("jobs", "a", "other", 1).metadata == {"experiment": {"run_id": "r"}, "other": 1}
    # Retention counts from the end, not from the last metadata change.
    assert queue.delete_final(finished_before=3) == 1


@pytest.mark.unit
def test_a_work_item_table_from_before_the_times_and_metadata_gets_them(tmp_path):
    directory = tmp_path / "coordination"
    with closing(SQLiteCoordinationDatabase(directory)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        queue.add(job("ended"), now=0)
        queue.add(job("waiting", serial_key=None), now=0)
        queue.take("jobs", "w", ["training_job:tinker"], now=1)
        queue.finish("jobs", "ended", "w", {}, now=2)
    with closing(sqlite3.connect(directory / "coordination.sqlite3")) as connection:
        for column in ("started_at", "ended_at", "metadata_json"):
            connection.execute(f"ALTER TABLE work_item DROP COLUMN {column}")
        connection.commit()

    with closing(SQLiteCoordinationDatabase(directory)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        ended, waiting = queue.get("jobs", "ended"), queue.get("jobs", "waiting")
        # An ended item ended at its last change; the start of an earlier item is not known.
        assert (ended.started_at, ended.ended_at, ended.metadata) == (None, 2, {})
        assert (waiting.started_at, waiting.ended_at, waiting.metadata) == (None, None, {})


def test_the_queued_training_requests_of_an_older_version_move_to_their_new_queue_and_kind(tmp_path):
    directory = tmp_path / "coordination"
    with closing(SQLiteCoordinationDatabase(directory)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        queue.add(NewWork("instructions", "ask-1", "my/agent", "train_instruction:my/agent", {"text": "t"}), now=0)
        queue.add(job("other"), now=0)

    with closing(SQLiteCoordinationDatabase(directory)) as database:
        queue = SQLWorkQueue(database, lease_seconds=10)
        moved = queue.get("training_requests", "ask-1")
        assert (moved.kind, moved.payload, queue.get("instructions", "ask-1")) == (
            "training_request:my/agent",
            {"text": "t"},
            None,
        )
        assert queue.get("jobs", "other").kind == job("other").kind


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
    with pytest.raises(ControlLeaseConflict):
        table.write(ScenarioModel("s"), served, control_lease_number=taken_over.number)
    table.write(ScenarioModel("s"), served, control_lease_number=3)
    assert table.read(ScenarioModel("s")) == served and table.read_scenario("s") == {"active": served}
    assert table.read(ScenarioModel("s", "judge")) is None
    assert table.is_referenced(served) and table.changed_sequence() > before

    table.schedule_stop(ScheduledStop(served, 30), control_lease_number=3)
    assert table.due_stops(29) == () and table.due_stops(30) == (ScheduledStop(served, 30),)
    table.delete_stop(served, control_lease_number=3)
    assert table.due_stops(100) == ()
    assert table.remove(ScenarioModel("s"), control_lease_number=3) == served
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
    with pytest.raises(ControlLeaseConflict):
        table.request("s", {"artifact_id": "b"}, control_lease_number=first.number, now=12)


@pytest.mark.unit
def test_two_workers_never_hold_one_item(open_coordination):
    queue = open_coordination().queue
    for index in range(20):
        queue.add(NewWork("evaluations", f"rollout-{index}", "s", "evaluation_rollout", {"k": index}), now=0)
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
def test_an_upgrade_deletes_the_record_copies_of_an_older_version(tmp_path):
    database = SQLiteCoordinationDatabase(tmp_path / "c")
    queue = SQLWorkQueue(database)
    queue.request("data", "s", "record_copy", "record_copy:s", {})
    kept = queue.request("data", "s", "artifact_copy", "artifact_copy:s", {})
    database.close()
    reopened = SQLiteCoordinationDatabase(tmp_path / "c")
    try:
        assert [item.id for item in SQLWorkQueue(reopened).items("data")] == [kept.id]
    finally:
        reopened.close()


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
    assert queue.get(SERVING_QUEUE, waiting.id).state == "failed"
    assert queue.get(SERVING_QUEUE, old.id).state == "assigned"
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
    with pytest.raises(ValueError, match="b, c"):
        table.write_traffic_weights("s", {"a": 50, "b": 25, "c": 25}, control_lease_number=lease.number)
    assert table.read_traffic_weights("s") == {}
    before_weights = table.changed_sequence()
    table.write_traffic_weights("s", {"active": 100, "absent": 0}, control_lease_number=lease.number)
    assert table.read_traffic_weights("s") == {"active": 100, "absent": 0}
    assert table.changed_sequence() > before_weights
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
    with pytest.raises(WorkLeaseLost, match="later attempt"):
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
    with pytest.raises(ControlLeaseConflict):
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
    queue.add(NewWork(SERVING_QUEUE, "stop-1", "t", STOP_SERVING_KIND, {}, serial_key=SERVING_QUEUE), now=14)
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
    queue.add(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=3)
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
    with pytest.raises(WorkLeaseLost):
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
    queue.add(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=3)
    owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=3)

    table.recover_completed(owner, "worker", stop_delay_seconds=10, kept=(live,), now=4)
    assert table.current_request("live").metadata == table.current_request("gone").metadata == {"recovery": True}
    assert table.read_scenario("live") == {"active": live}
    assert table.read_scenario("gone") == {}
    assert table.due_stops(14) == (ScheduledStop(gone, 14),)


@pytest.mark.unit
def test_startup_and_cleanup_completion_require_the_current_runtime_owner(open_coordination):
    stores = open_coordination()
    lease = stores.lease.acquire("control", now=0)
    table, queue = stores.served, stores.queue
    served = ServedArtifact("a", "http://engine", "a")
    queue.add(NewWork(INFERENCE_OWNER_QUEUE, "owner", "runtime", INFERENCE_OWNER_QUEUE), now=0)
    owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=0)
    table.request("s", {"artifact_id": "a"}, control_lease_number=lease.number, now=0)
    startup = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND], now=1)
    queue.extend_lease(SERVING_QUEUE, startup.id, "worker", now=9)
    with pytest.raises(WorkLeaseLost):
        table.complete_request(
            startup, "worker", {"active": served}, owner=owner, serving_config={}, stop_delay_seconds=10, now=11
        )
    assert table.read_scenario("s") == {} and table.current_request("s").state == "assigned"
    queue.requeue_expired(now=11)
    current_owner = queue.take(INFERENCE_OWNER_QUEUE, "worker", [INFERENCE_OWNER_QUEUE], now=12)
    with pytest.raises(WorkLeaseLost, match="later attempt"):
        table.complete_request(
            startup, "worker", {"active": served}, owner=owner, serving_config={}, stop_delay_seconds=10, now=13
        )
    table.complete_request(
        startup, "worker", {"active": served}, owner=current_owner, serving_config={}, stop_delay_seconds=10, now=13
    )

    stop = ScheduledStop(ServedArtifact("old", "http://engine", "old"), 14)
    table.schedule_stop(stop, control_lease_number=lease.number)
    queue.add(NewWork(SERVING_QUEUE, "stop", "s", STOP_SERVING_KIND, serial_key=SERVING_QUEUE), now=14)
    cleanup = queue.take(SERVING_QUEUE, "worker", [STOP_SERVING_KIND], now=14)
    with pytest.raises(WorkLeaseLost, match="later attempt"):
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
    first = queue.add(job("a", serial_key=None), now=0)
    second = queue.add(job("b", serial_key=None), now=0)
    queue.add(job("other", scenario="other", serial_key=None), now=0)
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
    from reef.storage.work_queue import WorkLogLine

    queue = open_coordination().queue
    queue.add(job("a"), now=0)
    queue.take("jobs", "w", ["training_job:tinker"], now=1)
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
    assert queue.read_log("jobs", "a", 1) == ()
