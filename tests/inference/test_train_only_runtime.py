"""A training-only deployment commits each step through the coordinator and refuses every request with 503."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path

import pytest
from aiohttp import web

from reef.core.errors import InferenceUnavailable
from reef.inference.train_only import TrainOnlyInferenceRuntime
from reef.runtime.deployment import RuntimeConfigError, RuntimeRegistry
from reef.runtime.executor.connection import ExecutorCoordinatorClient
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.interfaces import (
    PreparedTrainingJob,
    TrainingBackend,
    TrainingCheckpoint,
    TrainingContext,
    TrainingCoordinationConfig,
    TrainingMethod,
    TrainingMetrics,
)
from reef.runtime.recovery import marker_path, read_marker
from reef.runtime.scheduler import TrainingCoordinator
from reef.service.assembly import _connect_training_runtime
from reef.service.deploy.service_config import ServiceConfig
from reef.service.errors import translate_error
from reef.train.algos import StepScheduling
from reef.train.runtime import ExecutorTrainingRuntime
from reef.train.runtime_backend import FixedTrainingMethodSelector, RuntimeCandidateBackend


class Job(PreparedTrainingJob):
    def __init__(self, checkpoint: TrainingCheckpoint) -> None:
        self._checkpoint = checkpoint

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._checkpoint

    def train(self) -> TrainingMetrics:
        return TrainingMetrics(training={"loss": 1.0})

    def save_checkpoint(self) -> None:
        self.checkpoint.path.mkdir()


class ResidentTrainer(TrainingBackend):
    """A full-weight trainer that saves every second step and never sends weights."""

    def __init__(self, tmp_path: Path) -> None:
        self._config = TrainingCoordinationConfig(str(tmp_path / "step-{rollout_id}"))
        self._context = TrainingContext(runtime_load_id="inc:0")

    @property
    def config(self) -> TrainingCoordinationConfig:
        return self._config

    @property
    def context(self) -> TrainingContext:
        return self._context

    def start(self) -> None:
        pass

    def check_health(self) -> None:
        pass

    @contextmanager
    def prepare(self, payload, *, job_id, scenario_step, prior_marker):
        rollout_id = self.context.next_rollout_id
        path = Path(self.config.save_hf_template.format(rollout_id=rollout_id))
        yield Job(TrainingCheckpoint(rollout_id, path, scenario_step, writes_checkpoint=scenario_step % 2 == 1))

    def prepare_weights(self, runtime_load_id, *, force_full):
        raise AssertionError("a training-only deployment sends no weights")

    def send_weights(self, runtime_load_id, *, force_full):
        raise AssertionError("a training-only deployment sends no weights")

    def initialize_version(self, runtime_load_id):
        raise AssertionError("a full-weight trainer with a checkpoint path publishes at startup")

    def prepare_training_step(self, batch, objective, algorithm_state, scheduling):
        raise AssertionError("unexpected prepare_training_step in this fixture")

    def activate_scenario(self, scenario):
        raise AssertionError("unexpected activate_scenario in this fixture")

    def send_adapter(self, scenario, name):
        raise AssertionError("unexpected send_adapter in this fixture")

    def close(self) -> None:
        pass


def test_each_step_commits_without_engines_and_requests_fail_with_503(tmp_path):
    trainer = ResidentTrainer(tmp_path)
    client = ExecutorCoordinatorClient(UniProcExecutor.from_workers([TrainingCoordinator(trainer, None)]))
    inference = RuntimeRegistry().build({"type": "train_only", "control": client}, model_path="model", environ={})
    assert isinstance(inference, TrainOnlyInferenceRuntime)
    backend = RuntimeCandidateBackend(
        ExecutorTrainingRuntime(client),
        FixedTrainingMethodSelector(TrainingMethod("sft")),
        StepScheduling(),
        inference_runtime=inference,
    )
    checkpoints = []
    for scenario_step in (0, 1):
        candidate = backend.train_candidate(
            {
                "scenario_step": scenario_step,
                "parent_runtime_load_id": inference.current_runtime_load_id(),
                "samples": [scenario_step],
            }
        )
        activated = backend.activate_candidate(candidate)
        backend.acknowledge_commit(scenario_step + 1, candidate.training_job_id)
        assert read_marker(marker_path(trainer.config.save_hf_template))["status"] == "COMPLETE"
        assert inference.current_runtime_load_id() == activated.runtime_load_id
        checkpoints.append(candidate.checkpoint_path)
    # Only the save-interval step keeps a checkpoint; every step commits a new version.
    assert checkpoints == [None, str(tmp_path / "step-1")]
    assert inference.current_runtime_load_id() == "inc:3"

    with pytest.raises(InferenceUnavailable, match=r"inference\.num-gpus: 0") as refused:
        asyncio.run(inference.acquire_inference())
    response = translate_error(refused.value)
    assert isinstance(response, web.HTTPServiceUnavailable)
    assert "serves no inference requests" in response.text


def test_service_assembly_selects_the_training_only_runtime(monkeypatch):
    trainer_client = ExecutorCoordinatorClient(
        UniProcExecutor.from_workers([TrainingCoordinator(ResidentTrainer(Path("/unused")), None)])
    )
    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", lambda **kwargs: trainer_client)
    settings = ServiceConfig(recipe="recipe", training_backend="slime", inference_num_gpus=0, ray_address="auto")
    training, inference = _connect_training_runtime(settings, model_path="model")
    try:
        assert isinstance(training, ExecutorTrainingRuntime)
        assert isinstance(inference, TrainOnlyInferenceRuntime)
    finally:
        inference.shutdown()
        training.shutdown()


def test_training_only_runtime_refuses_request_handler_settings():
    client = ExecutorCoordinatorClient(UniProcExecutor.from_workers([object()]))
    with pytest.raises(RuntimeConfigError, match="no inference engines"):
        TrainOnlyInferenceRuntime(control=client, inference_handler_config={"sampling_defaults": {}})
