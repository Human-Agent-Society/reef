import sqlite3
from contextlib import closing
from dataclasses import dataclass

import pytest

from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.storage.control_lease import ControlLease
from reef.storage.marks import Marks
from reef.storage.served_models import ControlLeaseConflict, ServedModelsWriter
from reef.storage.sql_coordination import (
    PostgresCoordinationDatabase,
    SQLControlLease,
    SQLiteCoordinationDatabase,
    SQLMarks,
    SQLServedModels,
    SQLWorkQueue,
)
from reef.storage.work_queue import NewWork, WorkConflict, WorkItem, WorkLeaseLost, WorkQueue


@dataclass
class Coordination:
    queue: WorkQueue
    served: ServedModelsWriter
    lease: ControlLease
    marks: Marks


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
            SQLMarks(database),
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
def test_two_workers_never_hold_one_item(open_coordination):
    queue = open_coordination().queue
    for index in range(20):
        queue.add(NewWork("evaluations", f"rollout-{index}", "s", "evaluation_rollout", {"k": index}), now=0)
    import threading

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
def test_a_mark_reads_zero_until_it_advances_and_never_moves_back(open_coordination):
    marks = open_coordination().marks
    assert marks.read("record_copy:s") == 0
    marks.advance("record_copy:s", 5)
    marks.advance("record_copy:s", 3)
    marks.advance("record_copy:t", 1)
    assert (marks.read("record_copy:s"), marks.read("record_copy:t")) == (5, 1)
    with pytest.raises(ValueError, match="non-negative"):
        marks.advance("record_copy:s", -1)
    with pytest.raises(ValueError, match="non-empty"):
        marks.read("")
