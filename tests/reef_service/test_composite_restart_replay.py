"""A weight job of a composite survives a restart after the other components moved the scenario step.

Real pieces: TrainingExecution, the file marker store and its state machine, the
coordinator's health derivation, ExecutorTrainingRuntime and RuntimeScheduler.
Faked: the RPC transport (in process) and the inference receiver.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service.test_runtime_scheduler import WeightReceiver
from reef_service.test_training_execution import PAYLOAD, MemoryTrainingBackend

from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import TrainingCheckpoint, TrainingCoordinationConfig
from reef.runtime.recovery import FileTrainingJobStore, read_marker
from reef.runtime.scheduler import RuntimeScheduler, TrainingCoordinator, TrainingExecution
from reef.train.runtime import ExecutorTrainingRuntime

RESERVED_AT = 0
MOVED_TO = 3  # three harness commits landed while the job was out


class _MarkerCoordinator(CoordinatorClient):
    """The coordinator client over the real execution, marker store and health derivation; no transport."""

    def __init__(self, backend: MemoryTrainingBackend) -> None:
        self.backend = backend
        self.store = FileTrainingJobStore(backend.path)
        self.execution = TrainingExecution(self.store, backend, backend.state)
        self.commits: list[str] = []
        self._config = TrainingCoordinationConfig(save_hf_template=str(backend.path.parent / "hf-{rollout_id}"))
        self._store = self.store

    def health(self):
        return {
            "ok": True,
            "phase": "serving",
            "colocate": False,
            "inference_url": "http://inference",
            "lora_adapter": None,
            "lora_mode": None,
            "lora_adapters": {},
            "adapter_residency": None,
            "training_job": TrainingCoordinator._training_job_health(self),
        }

    def prepare_training_step(self, batch, objective, algorithm_state, scheduling):
        raise NotImplementedError

    def execute_training_job(self, payload):
        return self.execution.execute(payload)

    def update_serving_weights(self, training_job_id):
        raise NotImplementedError

    def reject_training_candidate(self, training_job_id):
        raise NotImplementedError

    def acknowledge_training_commit(self, training_job_id):
        self.commits.append(training_job_id)


def _weight_job_before_the_restart(root: Path, *, published: bool) -> dict:
    """The job reserved at RESERVED_AT reaches CHECKPOINT, or READY_TO_COMMIT when published."""
    backend = MemoryTrainingBackend(root)
    backend.checkpoint = TrainingCheckpoint(0, root / "checkpoint", scenario_step=RESERVED_AT)
    result = TrainingExecution(FileTrainingJobStore(backend.path), backend, backend.state).execute(
        {**PAYLOAD, "scenario_step": RESERVED_AT}
    )
    marker = read_marker(backend.path)
    assert (result.outcome, marker["status"], marker["scenario_step"]) == ("checkpoint", "CHECKPOINT", RESERVED_AT)
    if published:
        store = FileTrainingJobStore(backend.path)
        store.transition(marker, "UPDATING_WEIGHTS", target_runtime_load_id="engine:1")
        store.transition(marker, "READY_TO_COMMIT", runtime_load_id="engine:1")
    return read_marker(backend.path)


def _restarted(root: Path) -> tuple[_MarkerCoordinator, RuntimeScheduler]:
    coordinator = _MarkerCoordinator(MemoryTrainingBackend(root))
    return coordinator, RuntimeScheduler(ExecutorTrainingRuntime(coordinator), WeightReceiver())


@pytest.mark.unit
@pytest.mark.parametrize("published", [False, True])
def test_the_rebuilt_batch_replays_the_job_at_the_moved_step(tmp_path: Path, published: bool) -> None:
    before = _weight_job_before_the_restart(tmp_path / "moved", published=published)
    reference = _weight_job_before_the_restart(tmp_path / "same", published=published)
    assert before["job_id"] == reference["job_id"]

    # After the restart the trainer rebuilds the same batch; the scenario step is what the harness left.
    coordinator, scheduler = _restarted(tmp_path / "moved")
    scheduler.recover_pending_step(MOVED_TO, committed_training_job_id=None, committed_training_without_job_id=False)
    moved = scheduler.train_candidate({**PAYLOAD, "scenario_step": MOVED_TO})
    same_coordinator, same_scheduler = _restarted(tmp_path / "same")
    same_scheduler.recover_pending_step(
        RESERVED_AT, committed_training_job_id=None, committed_training_without_job_id=False
    )
    same = same_scheduler.train_candidate({**PAYLOAD, "scenario_step": RESERVED_AT})

    # The moved step behaves exactly as the unmoved one: the marker's job resumes, nothing trains again.
    assert (type(moved), moved.training_job_id) == (type(same), same.training_job_id)
    assert moved.training_job_id == before["job_id"]
    assert coordinator.backend.events == same_coordinator.backend.events == []
    assert read_marker(coordinator.backend.path)["status"] == read_marker(same_coordinator.backend.path)["status"]
