"""Publication/recovery contracts using CPU-only, framework-independent engines."""

from __future__ import annotations


import pytest

from reef.runtime.adapter_residency import AdapterCapacityExhausted, AdapterEvictionFailed
from reef.runtime.training_job import marker as markers
from reef.runtime.training_job.publication import TrainingPublication


class MemoryPublisher:
    def __init__(self, path):
        self.path = path
        self.events = []
        self.fail = None
        self.paused = False
        self.version = "engine:1"

    def _event(self, name):
        marker = markers.read_marker(self.path)
        self.events.append((name, None if marker is None else marker["status"]))
        if self.fail == name:
            raise RuntimeError(f"failed {name}")

    def recover(self, marker):
        self._event("recover")

    def pause(self):
        self._event("pause")
        self.paused = True

    def publish(self, marker, *, force_full):
        assert self.paused
        self._event("full-transfer" if force_full else "transfer")
        return self.version

    def republish(self, runtime_load_id, marker):
        assert self.paused
        self._event("republish")
        return self.version

    def resume(self):
        self._event("resume")
        self.paused = False

    def restore_incumbent(self):
        self._event("restore")
        self.paused = False

    def abort(self):
        self._event("abort")
        self.paused = True


@pytest.fixture
def publication(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    path = tmp_path / ".reef-latest-job.json"
    markers.write_marker(
        path,
        {
            "job_id": "job-1",
            "rollout_id": 0,
            "status": "CHECKPOINT",
            "checkpoint_path": str(checkpoint),
            "train_metrics": {"loss": 0.5},
        },
    )
    publisher = MemoryPublisher(path)
    return TrainingPublication(path, publisher), publisher, path


def test_publication_waits_for_durable_head_and_replays_without_transfer(publication):
    coordinator, publisher, path = publication
    first = coordinator.publish("job-1")
    assert first.published
    assert first.marker["runtime_load_id"] == "engine:1"
    assert coordinator.phase == "awaiting_commit"
    assert publisher.paused
    assert publisher.events == [("pause", "CHECKPOINT"), ("transfer", "UPDATING_WEIGHTS")]
    assert not coordinator.publish("job-1").published
    coordinator.acknowledge("job-1")
    assert publisher.events[-1] == ("resume", "HEAD_COMMITTED")
    assert not publisher.paused
    assert markers.read_marker(path)["status"] == "COMPLETE"
    assert coordinator.phase == "serving"
    count = len(publisher.events)
    coordinator.acknowledge("job-1")
    assert not coordinator.publish("job-1").published
    assert len(publisher.events) == count


@pytest.mark.parametrize("failure,status", [("pause", "CHECKPOINT"), ("transfer", "UPDATING_WEIGHTS")])
def test_failed_publication_retries_at_the_durable_stage(publication, failure, status):
    coordinator, publisher, path = publication
    publisher.fail = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        coordinator.publish("job-1")
    assert markers.read_marker(path)["status"] == status
    assert publisher.events[-1][0] == "abort"
    assert coordinator.phase == "weight_sync_failed"
    publisher.fail = None
    publisher.events.clear()
    # A new coordinator has no process-local transaction state to rely on.
    recovered = TrainingPublication(path, publisher)
    recovered.publish("job-1")
    names = [name for name, _ in publisher.events]
    assert names == (["recover", "pause", "full-transfer"] if status == "UPDATING_WEIGHTS" else ["pause", "transfer"])
    assert publisher.paused
    assert recovered.phase == "awaiting_commit"


def test_resume_failure_retries_without_republishing(publication):
    coordinator, publisher, path = publication
    coordinator.publish("job-1")
    publisher.fail = "resume"
    with pytest.raises(RuntimeError, match="failed resume"):
        coordinator.acknowledge("job-1")
    marker = markers.read_marker(path)
    assert marker["status"] == "HEAD_COMMITTED" and marker["commit_acknowledged"]
    publisher.fail = None
    publisher.events.clear()
    TrainingPublication(path, publisher).acknowledge("job-1")
    assert publisher.events == [("resume", "HEAD_COMMITTED")]


def test_failed_acknowledgement_write_never_resumes(publication, monkeypatch):
    coordinator, publisher, path = publication
    coordinator.publish("job-1")
    marker = markers.read_marker(path)

    def fail_write(path, value):
        raise OSError("disk full")

    monkeypatch.setattr(markers, "write_marker", fail_write)
    with pytest.raises(OSError, match="disk full"):
        coordinator.acknowledge("job-1")
    assert publisher.paused
    assert markers.read_marker(path)["status"] == "READY_TO_COMMIT"
    with pytest.raises(OSError):
        markers.transition_marker(path, marker, "HEAD_COMMITTED", commit_acknowledged=True)
    assert marker["status"] == "READY_TO_COMMIT"
    assert "commit_acknowledged" not in marker


def test_failed_publication_record_forces_complete_transfer_on_retry(publication, monkeypatch):
    coordinator, publisher, path = publication
    write = markers.write_marker

    def fail_ready(path, value):
        if value["status"] == "READY_TO_COMMIT":
            raise OSError("disk full")
        write(path, value)

    with monkeypatch.context() as context:
        context.setattr(markers, "write_marker", fail_ready)
        with pytest.raises(OSError, match="disk full"):
            coordinator.publish("job-1")
    assert markers.read_marker(path)["status"] == "UPDATING_WEIGHTS"
    assert publisher.paused
    coordinator.publish("job-1")
    assert publisher.events[-1] == ("full-transfer", "UPDATING_WEIGHTS")


def test_rejection_persists_before_restoration_and_retries(publication):
    coordinator, publisher, path = publication
    publisher.fail = "restore"
    with pytest.raises(RuntimeError, match="failed restore"):
        coordinator.reject("job-1")
    assert publisher.events == [("restore", "REJECTING")]
    assert markers.read_marker(path)["status"] == "REJECTING"
    publisher.fail = None
    TrainingPublication(path, publisher).reject("job-1")
    assert markers.read_marker(path)["status"] == "REJECTED"
    count = len(publisher.events)
    coordinator.reject("job-1")
    assert len(publisher.events) == count


@pytest.mark.parametrize("status", ["CHECKPOINT", "UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"])
def test_startup_recovery_resumes_only_a_committed_publication(publication, status):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker.update(
        status=status, runtime_load_id="engine:1", commit_acknowledged=status in {"HEAD_COMMITTED", "COMPLETE"}
    )
    markers.write_marker(path, marker)
    coordinator.prepare_recovery(marker)
    if status != "COMPLETE":
        assert publisher.paused
    coordinator.finish_recovery(marker, "engine:1")
    if status in {"HEAD_COMMITTED", "COMPLETE"}:
        assert not publisher.paused
        assert markers.read_marker(path)["status"] == "COMPLETE"
    else:
        assert publisher.paused
        assert markers.read_marker(path)["status"] == "READY_TO_COMMIT"
        assert not any(name == "resume" for name, _ in publisher.events)


def test_recovery_rejects_a_changed_serving_identity(publication):
    coordinator, publisher, path = publication
    coordinator.publish("job-1")
    marker = markers.read_marker(path)
    coordinator.prepare_recovery(marker)
    with pytest.raises(RuntimeError, match="changed runtime load ID"):
        coordinator.finish_recovery(marker, "engine:2")
    assert publisher.paused
    assert publisher.events[-1][0] == "abort"
    assert markers.read_marker(path)["runtime_load_id"] == "engine:1"


@pytest.mark.parametrize("error,abort", [(AdapterCapacityExhausted, False), (AdapterEvictionFailed, True)])
def test_capacity_refusal_does_not_terminate_unrelated_adapters(publication, error, abort):
    _, _, path = publication

    class CapacityPublisher(MemoryPublisher):
        def publish(self, marker, *, force_full):
            raise error("no capacity")

    publisher = CapacityPublisher(path)
    coordinator = TrainingPublication(path, publisher)
    with pytest.raises(error):
        coordinator.publish("job-1")
    assert any(name == "abort" for name, _ in publisher.events) == abort
    assert publisher.paused


@pytest.mark.parametrize("operation", ["publish", "reject", "acknowledge"])
def test_wrong_job_cannot_change_serving(publication, operation):
    coordinator, publisher, _ = publication
    with pytest.raises(RuntimeError, match="unknown training job"):
        getattr(coordinator, operation)("another-job")
    assert publisher.events == []


def test_candidate_cannot_be_acknowledged_before_publication(publication):
    coordinator, publisher, path = publication
    with pytest.raises(RuntimeError, match="cannot acknowledge"):
        coordinator.acknowledge("job-1")
    assert publisher.events == []
    assert markers.read_marker(path)["status"] == "CHECKPOINT"


def test_failed_completion_record_retries_only_resume(publication, monkeypatch):
    coordinator, publisher, path = publication
    coordinator.publish("job-1")
    write = markers.write_marker

    def fail_complete(path, value):
        if value["status"] == "COMPLETE":
            raise OSError("disk full")
        write(path, value)

    with monkeypatch.context() as context:
        context.setattr(markers, "write_marker", fail_complete)
        with pytest.raises(OSError, match="disk full"):
            coordinator.acknowledge("job-1")
    assert markers.read_marker(path)["status"] == "HEAD_COMMITTED"
    # The durable head already authorizes serving; repeating its resume is safe.
    assert not publisher.paused
    publisher.events.clear()
    TrainingPublication(path, publisher).acknowledge("job-1")
    assert publisher.events == [("resume", "HEAD_COMMITTED")]
    assert markers.read_marker(path)["status"] == "COMPLETE"


@pytest.mark.parametrize("version", ["", None])
def test_malformed_publication_identity_never_reaches_commit(publication, version):
    coordinator, publisher, path = publication
    publisher.version = version
    with pytest.raises(RuntimeError, match="empty runtime load ID"):
        coordinator.publish("job-1")
    assert markers.read_marker(path)["status"] == "UPDATING_WEIGHTS"
    assert publisher.paused
    marker = markers.read_marker(path)
    with pytest.raises(RuntimeError, match="empty runtime load ID"):
        coordinator.finish_recovery(marker, version)
    assert markers.read_marker(path)["status"] == "UPDATING_WEIGHTS"


@pytest.mark.parametrize("status", ["READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"])
def test_republication_preserves_commit_gate_and_identity(publication, status):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker.update(status=status, runtime_load_id=publisher.version, commit_acknowledged=status != "READY_TO_COMMIT")
    markers.write_marker(path, marker)
    assert coordinator.republish("engine:1") == "engine:1"
    assert publisher.events[:3] == [("pause", status), ("recover", status), ("republish", status)]
    result = markers.read_marker(path)
    assert result["runtime_load_id"] == "engine:1"
    if status == "READY_TO_COMMIT":
        assert publisher.paused
        assert coordinator.phase == "awaiting_commit"
        assert result == marker
        coordinator.acknowledge("job-1")
        assert publisher.events[-1] == ("resume", "HEAD_COMMITTED")
    else:
        assert not publisher.paused
        assert coordinator.phase == "serving"
        assert result["status"] == "COMPLETE"
        assert publisher.events[-1][0] == "resume"


@pytest.mark.parametrize("status", ["RUNNING", "CHECKPOINT", "UPDATING_WEIGHTS", "REJECTING", "REJECTED"])
def test_republication_refuses_trainer_state_that_may_contain_a_candidate(publication, status):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker["status"] = status
    markers.write_marker(path, marker)
    with pytest.raises(RuntimeError, match="use training job recovery"):
        coordinator.republish("engine:1")
    assert publisher.events == []
    assert markers.read_marker(path) == marker


@pytest.mark.parametrize("configured_path", [False, True])
def test_republication_without_a_job_resumes_verified_weights(publication, configured_path):
    _, publisher, path = publication
    path.unlink()
    coordinator = TrainingPublication(path if configured_path else None, publisher)
    assert coordinator.republish("engine:1") == "engine:1"
    assert [name for name, _ in publisher.events] == ["pause", "recover", "republish", "resume"]
    assert not path.exists()
    assert coordinator.phase == "serving"


@pytest.mark.parametrize("failure", ["pause", "recover", "republish", "resume"])
def test_failed_republication_stays_fenced_and_retries_same_committed_version(publication, failure):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker.update(status="COMPLETE", runtime_load_id="engine:1")
    markers.write_marker(path, marker)
    publisher.fail = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        coordinator.republish("engine:1")
    assert publisher.paused
    assert coordinator.phase == "weight_sync_failed"
    assert markers.read_marker(path) == marker
    publisher.fail = None
    publisher.events.clear()
    assert coordinator.republish("engine:1") == "engine:1"
    assert not publisher.paused


def test_mismatched_republication_aborts_before_resume(publication):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker.update(status="COMPLETE", runtime_load_id="engine:1")
    markers.write_marker(path, marker)
    with pytest.raises(RuntimeError, match="does not match"):
        coordinator.republish("wrong-seed")
    assert publisher.events == []
    publisher.version = "wrong-result"
    with pytest.raises(RuntimeError, match="changed runtime load ID"):
        coordinator.republish("engine:1")
    assert publisher.paused
    assert publisher.events[-1][0] == "abort"
    assert not any(name == "resume" for name, _ in publisher.events)
    assert markers.read_marker(path) == marker


def test_failed_republication_completion_write_remains_retryable(publication, monkeypatch):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    marker.update(status="HEAD_COMMITTED", runtime_load_id="engine:1", commit_acknowledged=True)
    markers.write_marker(path, marker)
    write = markers.write_marker

    def fail_complete(path, value):
        if value["status"] == "COMPLETE":
            raise OSError("disk full")
        write(path, value)

    with monkeypatch.context() as context:
        context.setattr(markers, "write_marker", fail_complete)
        with pytest.raises(OSError, match="disk full"):
            coordinator.republish("engine:1")
    assert publisher.paused
    assert markers.read_marker(path) == marker
    coordinator.republish("engine:1")
    assert markers.read_marker(path)["status"] == "COMPLETE"
    assert not publisher.paused


def test_stopped_coordinator_cannot_republish(publication):
    coordinator, publisher, _ = publication
    coordinator.phase = "stopped"
    with pytest.raises(RuntimeError, match="stopped"):
        coordinator.republish("engine:1")
    assert publisher.events == []


@pytest.mark.parametrize("status", [None, "COMPLETE", "READY_TO_COMMIT"])
def test_startup_backend_failure_aborts_even_a_committed_or_marker_free_restart(publication, status):
    coordinator, publisher, path = publication
    marker = markers.read_marker(path)
    if status is None:
        path.unlink()
        marker = None
    else:
        marker.update(status=status, runtime_load_id="engine:1")
        markers.write_marker(path, marker)
    with pytest.raises(RuntimeError, match="restore failed"), coordinator.recovery(marker):
        assert publisher.paused
        assert coordinator.phase == "recovering"
        raise RuntimeError("restore failed")
    assert publisher.paused
    assert publisher.events[-1][0] == "abort"
    assert coordinator.phase == "weight_sync_failed"
    assert markers.read_marker(path) == marker
