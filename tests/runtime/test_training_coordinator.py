"""Role-independent scheduling contracts, with no training or inference framework."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from reef.runtime.interfaces import (
    InferenceBackend,
    PreparedTrainingJob,
    TrainingBackend,
    TrainingCheckpoint,
    TrainingContext,
    TrainingCoordinationConfig,
    TrainingMetrics,
)
from reef.runtime.recovery import marker_path, read_marker
from reef.runtime.scheduler import TrainingCoordinator


class Receiver(InferenceBackend):
    def __init__(self, events):
        self.events = events
        self.versions = ["inc:0", "inc:0"]
        self.paused = False

    def initialize_version(self, runtime_load_id):
        self.events.append(("inference.initialize", runtime_load_id))
        self.versions = [runtime_load_id, runtime_load_id]

    def inference_url(self):
        return "http://inference.example"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("inference.pause")
        self.paused = True

    def resume(self):
        self.events.append("inference.resume")
        self.paused = False

    def recover(self):
        self.events.append("inference.recover")

    def abort(self):
        self.events.append("inference.abort")
        self.paused = True

    def offload(self, tags):
        self.events.append(("inference.offload", tags))

    def onload_weights(self):
        self.events.append("inference.weights")

    def onload_kv(self):
        self.events.append("inference.kv")

    def unload_adapter(self, name):
        self.events.append(("inference.unload", name))


class Trainer(TrainingBackend):
    def __init__(self, tmp_path, receiver, events, *, colocate=False):
        self._config = TrainingCoordinationConfig(str(tmp_path / "step-{rollout_id}"), colocate=colocate)
        self._context = TrainingContext(runtime_load_id="inc:0")
        # Fake transport endpoint: production adapters exchange tensor data
        # through native transport, never through coordinator control methods.
        self.receiver = receiver
        self.events = events
        self.sequence = 0
        self.incarnation = "native-trainer"
        self.partial = False
        self.targets = []

    def start(self):
        self.events.append("training.monitor")

    def check_health(self):
        pass

    def prepare_weights(self, runtime_load_id, *, force_full):
        self.events.append(("training.prepare", runtime_load_id))

    def send_weights(self, runtime_load_id, *, force_full):
        self.events.append(("training.send", force_full))
        self.targets.append(runtime_load_id)
        self.incarnation, sequence = runtime_load_id.split(":")
        self.sequence = int(sequence)
        version = self.current_runtime_load_id()
        self.receiver.versions[0] = version
        if not self.partial:
            self.receiver.versions[1] = version
        return version

    def current_runtime_load_id(self):
        return f"{self.incarnation}:{self.sequence}"

    def restore_runtime_load_id(self, version):
        self.events.append(("training.restore_version", version))
        self.sequence = int(version.split(":")[1]) - 1

    def initialize_version(self, runtime_load_id):
        self.incarnation, sequence = runtime_load_id.split(":")
        self.sequence = int(sequence)

    def close(self):
        self.events.append("training.close")

    @contextmanager
    def prepare(self, payload, *, job_id, scenario_step, prior_marker):
        rollout_id = self.context.next_rollout_id
        yield Job(
            TrainingCheckpoint(
                rollout_id,
                Path(self.config.save_hf_template.format(rollout_id=rollout_id)),
                scenario_step=scenario_step,
            ),
            self.events,
        )

    @property
    def config(self):
        return self._config

    @property
    def context(self):
        return self._context

    def prepare_training_step(self, batch, objective, algorithm_state, scheduling):
        raise AssertionError("unexpected prepare_training_step in this fixture")

    def activate_scenario(self, scenario):
        raise AssertionError("unexpected activate_scenario in this fixture")

    def send_adapter(self, scenario, name):
        raise AssertionError("unexpected send_adapter in this fixture")


class Job(PreparedTrainingJob):
    def __init__(self, checkpoint, events):
        self._checkpoint = checkpoint
        self.events = events

    @property
    def checkpoint(self):
        return self._checkpoint

    def train(self):
        self.events.append("training.train")
        return TrainingMetrics(training={"loss": 1.0})

    def save_checkpoint(self):
        self.events.append("training.checkpoint")
        self.checkpoint.path.mkdir()


def build(tmp_path, *, colocate=False, owns_training=True):
    events = []
    inference = Receiver(events)
    training = Trainer(tmp_path, inference, events, colocate=colocate)
    coordinator = TrainingCoordinator(training, inference, owns_training=owns_training)
    events.clear()
    return coordinator, training, inference, events


@pytest.mark.parametrize("colocate", [False, True])
def test_two_backend_operations_share_one_commit_gate(tmp_path, colocate):
    coordinator, training, inference, events = build(tmp_path, colocate=colocate)
    result = coordinator.execute_training_job(
        {"scenario_step": 0, "expected_runtime_load_id": coordinator.serving_runtime_load_id()}
    )
    assert result.outcome == "checkpoint"
    assert "training.send" not in [item[0] if isinstance(item, tuple) else item for item in events]
    if colocate:
        assert events[:3] == ["inference.pause", ("inference.offload", None), "training.train"]
    else:
        assert events[0] == "training.train"
    coordinator.update_serving_weights(result.training_job_id)
    assert inference.paused
    marker = read_marker(marker_path(training.config.save_hf_template))
    assert marker["status"] == "READY_TO_COMMIT"
    assert "inference.resume" not in events
    coordinator.acknowledge_training_commit(result.training_job_id)
    assert events[-1] == "inference.resume"
    assert not inference.paused
    assert read_marker(marker_path(training.config.save_hf_template))["status"] == "COMPLETE"


def test_partial_receive_stays_fenced_and_retries_full_transfer(tmp_path):
    coordinator, training, inference, events = build(tmp_path)
    result = coordinator.execute_training_job(
        {"scenario_step": 0, "expected_runtime_load_id": coordinator.serving_runtime_load_id()}
    )
    training.partial = True
    with pytest.raises(RuntimeError, match="engines disagree"):
        coordinator.update_serving_weights(result.training_job_id)
    assert inference.paused
    assert "inference.abort" in events
    assert read_marker(marker_path(training.config.save_hf_template))["status"] == "UPDATING_WEIGHTS"
    training.partial = False
    events.clear()
    coordinator.update_serving_weights(result.training_job_id)
    assert events[:4] == [
        "inference.recover",
        "inference.pause",
        ("training.prepare", "inc:2"),
        ("training.send", True),
    ]
    assert inference.paused
    coordinator.acknowledge_training_commit(result.training_job_id)
    assert not inference.paused


def test_restart_republishes_pending_version_without_resuming(tmp_path):
    coordinator, training, inference, events = build(tmp_path)
    result = coordinator.execute_training_job(
        {"scenario_step": 0, "expected_runtime_load_id": coordinator.serving_runtime_load_id()}
    )
    published = coordinator.update_serving_weights(result.training_job_id).runtime_load_id
    events.clear()
    replacement = TrainingCoordinator(training, inference)
    assert replacement.serving_runtime_load_id() == published
    assert ("training.send", True) in events
    assert "inference.resume" not in events
    assert inference.paused
    replacement.acknowledge_training_commit(result.training_job_id)
    assert not inference.paused


def test_deployment_retains_training_worker_ownership(tmp_path):
    coordinator, _, _, events = build(tmp_path, owns_training=False)
    coordinator.shutdown()
    coordinator.shutdown()
    assert events == []
    assert coordinator.health()["phase"] == "stopped"


def test_reef_selects_and_persists_target_before_partial_transfer(tmp_path):
    coordinator, training, inference, _ = build(tmp_path)
    result = coordinator.execute_training_job({"scenario_step": 0, "expected_runtime_load_id": "inc:1"})
    training.partial = True
    with pytest.raises(RuntimeError, match="engines disagree"):
        coordinator.update_serving_weights(result.training_job_id)
    pending = read_marker(marker_path(training.config.save_hf_template))
    assert pending["target_runtime_load_id"] == "inc:2"
    assert training.targets == ["inc:1", "inc:2"]
    # Cold trainer and replacement engines have forgotten their transport
    # counters. Reef must recover the persisted target, not allocate inc:1.
    training.sequence = 0
    training.partial = False
    inference.versions = ["default", "default"]
    replacement = TrainingCoordinator(training, inference)
    assert replacement.serving_runtime_load_id() == "inc:2"
    assert training.targets[-1] == "inc:2"
    assert inference.paused


def test_sender_cannot_choose_a_different_published_identity(tmp_path, monkeypatch):
    coordinator, training, inference, _ = build(tmp_path)
    result = coordinator.execute_training_job({"scenario_step": 0, "expected_runtime_load_id": "inc:1"})

    class WrongSender(Trainer):
        def send_weights(self, runtime_load_id, *, force_full):
            self.receiver.versions = ["inc:99", "inc:99"]
            return "inc:99"

    wrong = WrongSender(tmp_path, inference, training.events)
    monkeypatch.setattr(training, "send_weights", wrong.send_weights)
    with pytest.raises(RuntimeError, match="expected 'inc:2'"):
        coordinator.update_serving_weights(result.training_job_id)
    assert coordinator.serving_runtime_load_id() == "inc:1"
    assert inference.paused
    pending = read_marker(marker_path(training.config.save_hf_template))
    assert pending["status"] == "UPDATING_WEIGHTS"
    assert pending["target_runtime_load_id"] == "inc:2"


@pytest.mark.parametrize("lora", [False, True])
def test_fresh_full_and_lora_deployments_use_a_reef_owned_incarnation(tmp_path, lora):
    events = []
    inference = Receiver(events)
    training = Trainer(tmp_path, inference, events)
    training._context = TrainingContext()
    initial = training.context.runtime_load_id
    training._config = TrainingCoordinationConfig(str(tmp_path / "step-{rollout_id}"), lora=lora)
    coordinator = TrainingCoordinator(training, inference)
    expected = initial if lora else initial.rsplit(":", 1)[0] + ":1"
    assert coordinator.serving_runtime_load_id() == expected
    assert inference.versions == [expected, expected]
    assert training.incarnation == initial.rsplit(":", 1)[0]
    assert not inference.paused
    assert training.targets == ([] if lora else [expected])
    if not lora:
        assert ("training.send", True) in events


def test_backend_terminal_failure_is_observable_during_idle_serving(tmp_path):
    coordinator, training, _, _ = build(tmp_path)

    class FailedTrainer(Trainer):
        def check_health(self):
            raise RuntimeError("training worker exited")

    coordinator._training = FailedTrainer(tmp_path, training.receiver, training.events)
    with pytest.raises(RuntimeError, match="training worker exited"):
        coordinator.health()


def test_colocated_sender_prepares_before_receiver_memory_is_restored(tmp_path):
    coordinator, _, inference, events = build(tmp_path, colocate=True)
    candidate = coordinator.execute_training_job({"scenario_step": 0, "expected_runtime_load_id": "inc:1"})
    events.clear()
    coordinator.update_serving_weights(candidate.training_job_id)
    assert events == [
        "inference.pause",
        ("inference.offload", None),
        ("training.prepare", "inc:2"),
        "inference.weights",
        ("training.send", False),
        "inference.kv",
    ]
    assert inference.paused
