"""Exercise the training lifecycle through real executor control RPC."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_named_recipe
from reef.runtime import (
    Executor,
    ExecutorConfig,
    ExecutorTrainGroupHandle,
    ExecutorTrainingRuntime,
    PreparedTrainingStep,
    RayRuntime,
    RayRuntimeError,
    RayTrainGroupHandle,
    RuntimeConfigError,
    RuntimeRegistry,
    TrainingGroupHandle,
    TrainingRuntimeError,
    WorkerSpec,
)
from reef.runtime.adapters import ray_runtime
from reef.runtime.executor import ray as ray_executor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.service import assembly
from reef.service.deploy.settings import ServiceSettings
from reef.train.evaluation import EvaluationResult, SelectionDecision

from .test_ray_runtime import DeferredWeightUpdateTrainGroupHandle, policy_batch


class Coordinator(DeferredWeightUpdateTrainGroupHandle):
    def __init__(self, *, inference_url="http://router", shutdown_events=None, **kwargs):
        super().__init__(**kwargs)
        self.inference_url = inference_url
        self.shutdown_events = [] if shutdown_events is None else shutdown_events

    def health(self):
        return {**super().health(), "inference_url": self.inference_url}

    def prepare_training_step(self, batch, step_preparer, algorithm_state):
        self.calls.append("prepare")
        return PreparedTrainingStep(
            action="train", payload={"samples": ["prepared"]}, next_algorithm_state=dict(algorithm_state), metrics={}
        )

    def shutdown(self):
        self.shutdown_events.append("shutdown")


def test_ray_public_names_remain_compatible_aliases():
    assert RayRuntime is ExecutorTrainingRuntime
    assert RayRuntimeError is TrainingRuntimeError
    assert RayTrainGroupHandle is TrainingGroupHandle


@pytest.mark.parametrize("backend", ["uni", UniProcExecutor, "reef.runtime.executor.uniproc:UniProcExecutor"])
def test_local_executor_runs_candidate_activation_and_durable_commit(backend):
    runtime = RuntimeRegistry().build(
        {
            "type": "executor_training",
            "executor": ExecutorConfig(backend=backend, workers=(WorkerSpec(Coordinator),)),
        },
        model_path="model-path",
    )
    assert isinstance(runtime, ExecutorTrainingRuntime)
    assert isinstance(runtime.train_group_handle, ExecutorTrainGroupHandle)
    worker = runtime.train_group_handle.executor.workers[0]
    try:
        assert runtime.model_path == "model-path"
        assert runtime.base_url == "http://router"
        prepared = runtime.prepare_training_step(policy_batch(), "custom-preparer", {}, 0)
        assert prepared.payload["expected_runtime_load_id"] == "v0"
        candidate = runtime.train_candidate(prepared.payload)
        assert worker.calls == ["prepare", "execute"]
        assert runtime.serving_runtime_load_id() == "engine:0"

        activated = runtime.activate_candidate(candidate)
        assert activated.runtime_load_id == "engine:1"
        assert runtime.inference_admission_status["open"] is False
        runtime.reconcile_training_job(1, committed_training_job_id=candidate.training_job_id)
        assert worker.calls == ["prepare", "execute", "update_weights", "acknowledge"]
        assert runtime.current_runtime_load_id() == "engine:1"
        assert runtime.inference_admission_status["open"] is True
    finally:
        runtime.shutdown()
    assert worker.shutdown_events == ["shutdown"]
    assert runtime.inference_admission_status["open"] is False


def test_mapping_config_resolves_worker_import_and_constructor_arguments():
    runtime = RuntimeRegistry().build(
        {
            "type": "executor_training",
            "executor": {
                "backend": "uni",
                "workers": [
                    {
                        "worker_cls": "reef_service.test_executor_runtime:Coordinator",
                        "kwargs": {"inference_url": "http://custom-router", "colocate": True},
                    }
                ],
            },
        },
        model_path="model",
    )
    try:
        assert runtime.base_url == "http://custom-router"
        candidate = runtime.train_candidate({})
        assert runtime.inference_admission_status["open"] is False
        decision = SelectionDecision("reject", "test", "1", "test rejection", EvaluationResult("test", "1", {}))
        runtime.reject_candidate(candidate, decision)
        assert runtime.inference_admission_status["open"] is True
        assert runtime.serving_runtime_load_id() == "engine:0"
    finally:
        runtime.shutdown()


def test_training_operations_target_only_the_selected_coordinator():
    from executor_helpers import AttachedTestGroup

    worker0, worker1 = Coordinator(), Coordinator()
    executor = AttachedTestGroup.from_workers((worker0, worker1))
    runtime = RuntimeRegistry().build(
        {"type": "executor_training", "executor": executor, "coordinator_rank": 1},
        model_path="model",
    )
    try:
        result = runtime.execute_training_job({"job": "test"})
        assert result.outcome == "complete"
        assert worker0.calls == []
        assert worker1.calls == ["execute", "update_weights"]
    finally:
        runtime.shutdown()
    assert worker0.shutdown_events == []
    assert worker1.shutdown_events == []


def test_failed_runtime_construction_releases_created_executor_workers():
    shutdown_events = []
    with pytest.raises(TrainingRuntimeError, match="inference_url is unset"):
        RuntimeRegistry().build(
            {
                "type": "executor_training",
                "executor": {
                    "backend": "uni",
                    "workers": [
                        WorkerSpec(Coordinator, kwargs={"inference_url": None, "shutdown_events": shutdown_events})
                    ],
                },
            },
            model_path="model",
        )
    assert shutdown_events == ["shutdown"]


def test_failed_runtime_construction_preserves_an_injected_executor():
    worker = Coordinator(inference_url=None)
    executor = UniProcExecutor.from_workers((worker,), owned=True)
    try:
        with pytest.raises(TrainingRuntimeError, match="inference_url is unset"):
            RuntimeRegistry().build({"type": "executor_training", "executor": executor}, model_path="model")
        executor.check_health()
        assert worker.shutdown_events == []
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "value",
    [None, "uni", {"workers": "bad"}, {"workers": [None]}, {"workers": [{"unexpected": "argument"}]}],
)
def test_invalid_executor_runtime_configuration_is_rejected(value):
    with pytest.raises(RuntimeConfigError, match=r"runtime\.executor"):
        RuntimeRegistry().build({"type": "executor_training", "executor": value}, model_path="model")


@pytest.mark.parametrize("timeout", [0, -1, True, "300", float("inf"), float("nan")])
def test_training_handle_rejects_invalid_timeout(timeout):
    executor = UniProcExecutor.from_workers((Coordinator(),))
    try:
        with pytest.raises(TrainingRuntimeError, match="training timeout"):
            ExecutorTrainGroupHandle(executor, timeout_s=timeout)
    finally:
        executor.shutdown()


@pytest.mark.parametrize("rank", [-1, True, "0"])
def test_training_handle_rejects_invalid_coordinator_rank(rank):
    executor = UniProcExecutor.from_workers((Coordinator(),))
    try:
        with pytest.raises(TrainingRuntimeError, match="coordinator rank"):
            ExecutorTrainGroupHandle(executor, rank=rank)
    finally:
        executor.shutdown()


@pytest.mark.parametrize("initialized", [False, True])
def test_named_ray_connection_honors_namespace_even_in_an_existing_session(monkeypatch, initialized):
    worker = Coordinator()
    actor = SimpleNamespace(
        health=SimpleNamespace(remote=worker.health),
        serving_runtime_load_id=SimpleNamespace(remote=worker.serving_runtime_load_id),
    )
    calls = []

    class FakeRay:
        @staticmethod
        def is_initialized():
            return initialized

        @staticmethod
        def init(**kwargs):
            calls.append(("init", kwargs))

        @staticmethod
        def get_actor(name, *, namespace):
            calls.append(("get_actor", name, namespace))
            return actor

        @staticmethod
        def get(value, *, timeout):
            calls.append(("get", timeout))
            return value

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: FakeRay)
    monkeypatch.setattr(ray_executor, "_require_ray", lambda: FakeRay)
    runtime = ray_runtime.connect_ray_runtime(actor_name="trainer", namespace="reef-other", train_timeout_s=900)
    try:
        assert ("get_actor", "trainer", "reef-other") in calls
        assert runtime.train_group_handle._timeout_s == 900
        assert ("get", 0.1) in calls  # Short waits allow terminal failure to interrupt the overall budget.
        init_calls = [call for call in calls if call[0] == "init"]
        assert init_calls == ([] if initialized else [("init", {"address": "auto", "namespace": "reef-other"})])
        assert isinstance(runtime.train_group_handle.executor, Executor)
    finally:
        runtime.shutdown()


def test_named_recipe_yaml_launches_a_configured_executor(tmp_path):
    (tmp_path / "local.yaml").write_text(
        """implementation: recipe
model:
  path: test-model
runtime:
  type: executor_training
  executor:
    backend: uni
    workers:
      - worker_cls: reef_service.test_executor_runtime:Coordinator
        kwargs:
          inference_url: http://configured-router
"""
    )
    recipe = build_named_recipe("local", config_directory=tmp_path, environ={})
    try:
        assert isinstance(recipe.runtime, ExecutorTrainingRuntime)
        assert recipe.runtime.model_path == "test-model"
        assert recipe.runtime.base_url == "http://configured-router"
    finally:
        recipe.runtime.shutdown()


def test_invalid_named_recipe_releases_its_configured_runtime(tmp_path):
    (tmp_path / "invalid.yaml").write_text(
        """implementation: recipe
model:
  path: test-model
artifact:
  checkpoint_every_n_versions: 0
runtime:
  type: injected
"""
    )
    worker = Coordinator()
    runtime = ExecutorTrainingRuntime(
        train_group_handle=ExecutorTrainGroupHandle(UniProcExecutor.from_workers((worker,), owned=True))
    )
    registry = RuntimeRegistry({"injected": lambda *args: runtime})
    with pytest.raises(RecipeConfigError, match="checkpoint_every_n_versions"):
        build_named_recipe("invalid", config_directory=tmp_path, environ={}, runtime_registry=registry)
    assert worker.shutdown_events == ["shutdown"]


def test_dispatcher_closes_owned_runtime_after_all_scenarios_but_not_on_reload(tmp_path):
    class SharedCoordinator(Coordinator):
        def health(self):
            return {**super().health(), "lora_mode": "scenario"}

    events = []
    worker = SharedCoordinator(shutdown_events=events)
    runtime = ExecutorTrainingRuntime(
        train_group_handle=ExecutorTrainGroupHandle(UniProcExecutor.from_workers((worker,), owned=True))
    )
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(runtime=runtime),
        InMemoryRepositoryBackend.factory(initial),
        owns_runtime=True,
        agent_record_dir=tmp_path / "records",
    )
    dispatcher.get_or_create_scenario("math")
    dispatcher.get_or_create_scenario("code")
    dispatcher._registry.reload("math")
    assert events == []

    def track_close(scenario):
        original = scenario.close

        def close():
            events.append(scenario.name)
            original()

        scenario.close = close

    track_close(dispatcher.get_or_create_scenario("math"))
    track_close(dispatcher.get_or_create_scenario("code"))
    dispatcher.close()
    dispatcher.close()
    assert events == ["math", "code", "shutdown"]


def test_dispatcher_keeps_a_borrowed_runtime_alive(tmp_path):
    worker = Coordinator()
    executor = UniProcExecutor.from_workers((worker,), owned=True)
    runtime = ExecutorTrainingRuntime(train_group_handle=ExecutorTrainGroupHandle(executor))
    dispatcher = Dispatcher(Recipe(runtime=runtime), InMemoryRepositoryBackend.factory(tmp_path))
    try:
        dispatcher.close()
        executor.check_health()
        assert worker.shutdown_events == []
    finally:
        runtime.shutdown()


def test_dispatcher_releases_owned_runtime_when_scenario_teardown_fails(tmp_path, monkeypatch):
    # This ownership-only fake recipe has no training backend. Do not let a
    # background training failure reload the scenario before close is tested.
    monkeypatch.setattr(Dispatcher, "_start_training", lambda *args: None)
    worker = Coordinator()
    runtime = ExecutorTrainingRuntime(
        train_group_handle=ExecutorTrainGroupHandle(UniProcExecutor.from_workers((worker,), owned=True))
    )
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(runtime=runtime),
        InMemoryRepositoryBackend.factory(initial),
        owns_runtime=True,
        agent_record_dir=tmp_path / "records",
    )
    scenario = dispatcher.get_or_create_scenario("math")
    original_close = scenario.close

    def failed_close():
        original_close()
        raise RuntimeError("scenario teardown failed")

    scenario.close = failed_close
    with pytest.raises(RuntimeError, match="scenario teardown failed"):
        dispatcher.close()
    assert worker.shutdown_events == ["shutdown"]
    dispatcher.close()


def test_service_app_cleanup_shuts_down_its_owned_runtime_once(monkeypatch, tmp_path):
    worker = Coordinator()
    runtime = ExecutorTrainingRuntime(
        train_group_handle=ExecutorTrainGroupHandle(UniProcExecutor.from_workers((worker,), owned=True))
    )
    monkeypatch.setattr(assembly, "_serving_recipe", lambda *args: Recipe(runtime=runtime))
    monkeypatch.setattr(assembly.GitLFSRepositoryBackend, "factory", lambda *args, **kwargs: lambda name: object())

    async def run():
        app = assembly.build_app(ServiceSettings(recipe="recipe", agent_record_dir=str(tmp_path)))
        runner = web.AppRunner(app)
        await runner.setup()
        await runner.cleanup()
        await runner.cleanup()

    asyncio.run(run())
    assert worker.shutdown_events == ["shutdown"]


def test_failed_service_assembly_releases_its_runtime(monkeypatch, tmp_path):
    worker = Coordinator()
    runtime = ExecutorTrainingRuntime(
        train_group_handle=ExecutorTrainGroupHandle(UniProcExecutor.from_workers((worker,), owned=True))
    )
    monkeypatch.setattr(assembly, "_serving_recipe", lambda *args: Recipe(runtime=runtime))
    monkeypatch.setattr(assembly.GitLFSRepositoryBackend, "factory", lambda *args, **kwargs: lambda name: object())
    with pytest.raises(ValueError, match="training must be an object"):
        assembly.build_dispatcher(
            ServiceSettings(recipe="recipe", agent_record_dir=str(tmp_path), training_settings="invalid")
        )
    assert worker.shutdown_events == ["shutdown"]
