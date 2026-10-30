from contextlib import closing
from dataclasses import dataclass

import pytest

from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.storage.control_lease import ControlLease
from reef.storage.served_models import ControlLeaseConflict, ServedModelsWriter
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
        return Coordination(
            SQLWorkQueue(database, lease_seconds=lease_seconds, max_attempts=max_attempts),
            SQLServedModels(database),
            SQLControlLease(database, lease_seconds=lease_seconds),
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
def test_an_item_waits_for_its_available_time_and_readers_follow_changes_in_order(open_coordination):
    queue = open_coordination().queue
    queue.add(NewWork("samples", "pass-1", "s", "record_selection", {}, available_at=50), now=0)
    assert queue.take("samples", "w", ["record_selection"], now=10) is None
    assert queue.take("samples", "w", ["record_selection"], now=50).id == "pass-1"

    changes = queue.read_changes("samples", 0)
    assert [item.changed_sequence for item in changes] == sorted(item.changed_sequence for item in changes)
    assert queue.read_changes("samples", changes[-1].changed_sequence) == ()


@pytest.mark.unit
def test_a_work_item_has_a_worker_exactly_while_it_is_held():
    with pytest.raises(ValueError, match="worker and a lease"):
        WorkItem("jobs", "a", "s", "k", {}, state="running")
    with pytest.raises(ValueError, match="result exactly"):
        WorkItem("jobs", "a", "s", "k", {}, state="finished")


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
