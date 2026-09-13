"""CPU contracts for the shared training-job coordinator."""

from contextlib import contextmanager

import pytest

from reef.runtime.base import TrainingJobResult
from reef.runtime.training_job import execution
from reef.runtime.training_job import marker as markers
from reef.runtime.training_job.execution import TrainingCheckpoint, TrainingExecution, TrainingMetrics, training_job_id
from reef.runtime.training_job.state import TrainingJobState

PAYLOAD = {"rollout_id": 0, "samples": [["sample-1"]], "expected_runtime_load_id": "engine:0"}


class MemoryTrainingBackend:
    def __init__(self, root):
        self.path = root / ".reef-latest-job.json"
        self.checkpoint = TrainingCheckpoint(0, root / "checkpoint")
        self.events = []
        self.fail = None
        self.early = None
        self.reserved = False
        self.state = TrainingJobState()
        self.prior = None

    def event(self, name):
        marker = markers.read_marker(self.path)
        self.events.append((name, None if marker is None else marker["status"]))
        if self.fail == name:
            raise RuntimeError(f"failed {name}")

    @contextmanager
    def prepare(self, payload, *, job_id, rollout_id, prior_marker):
        self.prior = prior_marker
        self.reserved = True
        try:
            self.event("prepare")
            yield self.early or self
        finally:
            self.reserved = False
            self.event("release")

    def train(self):
        assert self.reserved
        assert self.state.phase == "training"
        self.event("train")
        return TrainingMetrics(training={"loss": 0.25}, durable={"reward": 1.0})

    def save_checkpoint(self):
        assert self.reserved
        assert self.state.phase == "checkpointing"
        self.event("save")
        self.checkpoint.path.mkdir()


@pytest.fixture
def backend(tmp_path):
    return MemoryTrainingBackend(tmp_path)


def coordinator(backend):
    return TrainingExecution(backend.path, backend, backend.state)


def test_train_checkpoint_and_telemetry_are_recorded_before_releasing_reservation(backend):
    result = coordinator(backend).execute(PAYLOAD)
    assert result.outcome == "checkpoint"
    assert result.metrics == {"loss": 0.25, "reward": 1.0}
    assert backend.events == [("prepare", None), ("train", "RUNNING"), ("save", "RUNNING"), ("release", "CHECKPOINT")]
    marker = markers.read_marker(backend.path)
    assert marker["train_metrics"] == {"loss": 0.25}
    assert marker["metrics"] == {"reward": 1.0}
    assert result.training_job_id == training_job_id(PAYLOAD)
    assert not backend.reserved
    backend.events.clear()
    assert coordinator(backend).execute(PAYLOAD) == result
    assert backend.events == []


@pytest.mark.parametrize("outcome", ["stale", "storage_blocked"])
def test_admission_can_decline_without_a_running_marker(backend, outcome):
    backend.early = TrainingJobResult(outcome=outcome, runtime_load_id="engine:0")
    assert coordinator(backend).execute(PAYLOAD) is backend.early
    assert markers.read_marker(backend.path) is None
    assert backend.events == [("prepare", None), ("release", None)]


def test_preparation_failure_remains_retryable(backend):
    backend.fail = "prepare"
    with pytest.raises(RuntimeError, match="failed prepare"):
        coordinator(backend).execute(PAYLOAD)
    assert markers.read_marker(backend.path) is None
    assert not backend.reserved
    backend.fail = None
    assert coordinator(backend).execute(PAYLOAD).outcome == "checkpoint"


@pytest.mark.parametrize("failure,phase", [("train", "training_failed"), ("save", "checkpoint_failed")])
def test_failure_after_running_cannot_automatically_repeat_optimizer(backend, failure, phase, capsys):
    backend.fail = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        coordinator(backend).execute(PAYLOAD)
    assert backend.state.phase == phase
    assert f"failed {failure}" in capsys.readouterr().err
    assert markers.read_marker(backend.path)["status"] == "RUNNING"
    assert not backend.reserved
    backend.fail = None
    backend.events.clear()
    with pytest.raises(RuntimeError, match="operator recovery required"):
        coordinator(backend).execute(PAYLOAD)
    with pytest.raises(RuntimeError, match="ambiguous training job"):
        coordinator(backend).recover()
    assert backend.events == []


def test_running_write_failure_does_not_train(backend, monkeypatch):
    def fail_write(path, value):
        raise OSError("disk full")

    monkeypatch.setattr(execution, "write_marker", fail_write)
    with pytest.raises(OSError, match="disk full"):
        coordinator(backend).execute(PAYLOAD)
    assert backend.events == [("prepare", None), ("release", None)]
    assert not backend.reserved


def test_checkpoint_record_failure_leaves_ambiguous_state(backend, monkeypatch):
    def fail_write(path, value):
        raise OSError("disk full")

    monkeypatch.setattr(markers, "write_marker", fail_write)
    with pytest.raises(OSError, match="disk full"):
        coordinator(backend).execute(PAYLOAD)
    assert backend.checkpoint.path.is_dir()
    assert backend.state.phase == "checkpoint_failed"
    assert markers.read_marker(backend.path)["status"] == "RUNNING"
    with pytest.raises(RuntimeError, match="operator recovery required"):
        coordinator(backend).execute(PAYLOAD)


def test_cleanup_failure_after_record_replays_all_metrics(backend):
    backend.fail = "release"
    with pytest.raises(RuntimeError, match="failed release"):
        coordinator(backend).execute(PAYLOAD)
    backend.events.clear()
    result = coordinator(backend).execute(PAYLOAD)
    assert result.metrics == {"loss": 0.25, "reward": 1.0}
    assert backend.events == []


@pytest.mark.parametrize("status", ["CHECKPOINT", "READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE", "REJECTED"])
def test_replay_skips_backend_for_every_replayable_state(backend, status):
    coordinator(backend).execute(PAYLOAD)
    marker = markers.read_marker(backend.path)
    marker.update(status=status, runtime_load_id="engine:1", commit_acknowledged=True)
    markers.write_marker(backend.path, marker)
    backend.events.clear()
    result = coordinator(backend).execute(PAYLOAD)
    assert result.outcome == ("complete" if status == "COMPLETE" else "checkpoint")
    assert backend.events == []
    assert coordinator(backend).recover() == marker


@pytest.mark.parametrize(
    "status", ["RUNNING", "CHECKPOINT", "UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED", "REJECTING"]
)
def test_another_job_cannot_overwrite_inflight_state(backend, status):
    coordinator(backend).execute(PAYLOAD)
    marker = markers.read_marker(backend.path)
    marker.update(status=status, runtime_load_id="engine:1", commit_acknowledged=True)
    markers.write_marker(backend.path, marker)
    backend.events.clear()
    with pytest.raises(RuntimeError, match="operator recovery required"):
        coordinator(backend).execute({**PAYLOAD, "samples": [["another"]]})
    assert markers.read_marker(backend.path) == marker
    assert backend.events == []


def test_job_identity_preserves_scenarios_but_excludes_staleness_fences():
    payload = {**PAYLOAD, "max_staleness": 1, "scenario": "one"}
    assert training_job_id(payload) == training_job_id(
        {**payload, "max_staleness": 2, "expected_runtime_load_id": "new"}
    )
    assert training_job_id(payload) != training_job_id({**payload, "scenario": "two"})
    assert training_job_id(PAYLOAD) != training_job_id({**PAYLOAD, "expected_runtime_load_id": "new"})


def test_scenario_steps_can_use_a_separate_global_checkpoint_index(backend):
    backend.checkpoint = TrainingCheckpoint(12, backend.checkpoint.path, "scenario-a", 0)
    coordinator(backend).execute({**PAYLOAD, "scenario": "scenario-a"})
    marker = markers.read_marker(backend.path)
    assert (marker["rollout_id"], marker["scenario"], marker["scenario_step"]) == (12, "scenario-a", 0)


@pytest.mark.parametrize("invalid", [-1, True, "0", None])
def test_invalid_step_is_rejected_before_preparation(backend, invalid):
    with pytest.raises(ValueError, match="rollout_id"):
        coordinator(backend).execute({**PAYLOAD, "rollout_id": invalid})
    assert backend.events == []


def test_backend_cannot_claim_an_early_checkpoint(backend):
    backend.early = TrainingJobResult(outcome="checkpoint", checkpoint_path="fake", runtime_load_id="pending")
    with pytest.raises(RuntimeError, match="may only return stale or storage_blocked"):
        coordinator(backend).execute(PAYLOAD)
    assert markers.read_marker(backend.path) is None


def test_second_backend_uses_same_coordinator_and_checkpoint_contract(tmp_path):
    class FileTrainingBackend(MemoryTrainingBackend):
        def train(self):
            self.value = 7
            return TrainingMetrics(training={"value": self.value})

        def save_checkpoint(self):
            self.checkpoint.path.mkdir()
            (self.checkpoint.path / "weights").write_text(str(self.value))

    backend = FileTrainingBackend(tmp_path)
    result = coordinator(backend).execute(PAYLOAD)
    assert (backend.checkpoint.path / "weights").read_text() == "7"
    assert result.metrics == {"value": 7}


def test_missing_checkpoint_never_becomes_a_candidate(backend, monkeypatch):
    monkeypatch.setattr(backend, "save_checkpoint", lambda: None)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        coordinator(backend).execute(PAYLOAD)
    assert markers.read_marker(backend.path)["status"] == "RUNNING"
    assert backend.state.phase == "checkpoint_failed"


def test_training_and_publication_share_progress_and_keep_serving_unchanged_until_commit(backend):
    from reef.runtime.training_job.publication import TrainingPublication

    class Publisher:
        def __init__(self):
            self.version = "engine:0"
            self.paused = False

        def pause(self):
            self.paused = True

        def publish(self, marker, *, force_full):
            assert self.paused
            assert backend.checkpoint.path.is_dir()
            self.version = "engine:1"
            return self.version

        def resume(self):
            assert markers.read_marker(backend.path)["status"] == "HEAD_COMMITTED"
            self.paused = False

        def recover(self, marker):
            raise AssertionError("fresh publication must not recover engines")

        def restore_incumbent(self):
            raise AssertionError("candidate was not rejected")

        def abort(self):
            raise AssertionError("publication must not fail")

    publisher = Publisher()
    publication = TrainingPublication(backend.path, publisher)
    backend.state = publication.state
    job = coordinator(backend).execute(PAYLOAD)
    assert publisher.version == "engine:0"
    assert not publisher.paused
    assert publication.phase == "checkpointing"
    publication.publish(job.training_job_id)
    assert publisher.version == "engine:1"
    assert publisher.paused
    assert backend.state.phase == "awaiting_commit"
    publication.acknowledge(job.training_job_id)
    assert not publisher.paused
    assert backend.state.phase == "serving"
    assert coordinator(backend).execute(PAYLOAD).outcome == "complete"
