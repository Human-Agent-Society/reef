"""A job the backend published before a restart is committed before any new batch."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.train.backend import TrainingBackend
from reef.train.types import RecoveredTrainingStep, TrainStepResult

from .test_ray_runtime import DeferredWeightUpdateTrainGroupHandle

pytestmark = pytest.mark.unit


def _dispatcher() -> Dispatcher:
    root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
    initial = root / "initial"
    initial.mkdir()
    return Dispatcher(Recipe(), InMemoryRepositoryBackend.factory(initial, root=root / "repository"))


class _RecoveringBackend(TrainingBackend):
    """A dispatched backend whose recovery hands back one published job."""

    def __init__(self, recovered: RecoveredTrainingStep | None) -> None:
        self.recovered = recovered
        self.calls: list[tuple] = []

    @property
    def dispatched(self) -> bool:
        return True

    def initial_state(self):
        return {}

    def recover_pending_step(self, scenario_step, *, committed_training_job_id, committed_training_without_job_id):
        self.calls.append(("recover", scenario_step, committed_training_job_id))
        return self.recovered

    def acknowledge_commit(self, scenario_step, training_job_id):
        self.calls.append(("acknowledge", scenario_step, training_job_id))

    def prepare_step(self, batch, state, scenario_step):
        raise AssertionError("no batch is prepared while a recovered job waits for its commit")

    def evaluate(self, candidate):
        raise AssertionError("not used")

    def settle_step(self, prepared, decision):
        raise AssertionError("not used")

    def abort_step(self, prepared):
        raise AssertionError("not used")


def _scenario(backend: TrainingBackend, reserved: list, *, batch_reserved: list) -> SimpleNamespace:
    handle = DeferredWeightUpdateTrainGroupHandle(status="READY_TO_COMMIT", rollout_id=1)
    runtime = ExecutorTrainingRuntime(train_group_handle=handle, inference_url="http://router")

    def reserve_training_batch():
        batch_reserved.append(True)

    return SimpleNamespace(
        name="s",
        runtime=runtime,
        scenario_step=1,
        committed_training_job_id="job-0",
        committed_training_without_job_id=False,
        trainer=SimpleNamespace(training_backend=backend),
        reserve_recovered_step=reserved.append,
        reserve_training_batch=reserve_training_batch,
    )


def test_dispatcher_commits_a_recovered_job_before_reserving_a_batch(monkeypatch) -> None:
    result = TrainStepResult(
        {"steps": 2},
        {"selected": True},
        runtime_load_id="engine:2",
        checkpoint_path="/checkpoint",
        training_job_id="job-1",
    )
    recovered = RecoveredTrainingStep(result, frozenset({"i1"}))
    backend = _RecoveringBackend(recovered)
    reserved: list = []
    batch_reserved: list = []
    committed: list = []
    dispatcher = _dispatcher()
    scenario = _scenario(backend, reserved, batch_reserved=batch_reserved)
    dispatcher._registry = SimpleNamespace(get_optional=lambda name: scenario if name == "s" else None)
    monkeypatch.setattr(dispatcher, "_commit_result", lambda name, value: committed.append((name, value)))
    monkeypatch.setattr(dispatcher, "_record_training_error", lambda name, value: None)

    progressed = dispatcher._process_training_scenario("s")

    # The published job is reserved, committed and acknowledged in that order,
    # and this turn ends there: no batch is reserved behind it.
    assert progressed is True
    assert reserved == [recovered]
    assert committed == [("s", result)]
    assert backend.calls == [("recover", 1, "job-0"), ("acknowledge", 1, "job-1")]
    assert batch_reserved == []


def test_dispatcher_reserves_a_batch_when_nothing_was_left_pending(monkeypatch) -> None:
    backend = _RecoveringBackend(None)
    reserved: list = []
    batch_reserved: list = []
    dispatcher = _dispatcher()
    scenario = _scenario(backend, reserved, batch_reserved=batch_reserved)
    dispatcher._registry = SimpleNamespace(get_optional=lambda name: scenario if name == "s" else None)
    monkeypatch.setattr(dispatcher, "_record_training_error", lambda name, value: None)

    assert dispatcher._process_training_scenario("s") is False
    assert reserved == []
    assert batch_reserved == [True]
    assert backend.calls == [("recover", 1, "job-0")]
