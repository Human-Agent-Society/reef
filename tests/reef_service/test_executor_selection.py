"""Role-aware selection must not confuse local processes with GPU launchers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSettings, executor_settings, select_executor
from reef.runtime.executor.local import LocalExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.service.deploy.execution import service_executor_config, service_executor_selection
from reef.train.slime_backend.reef_adapters.executors.config import (
    DEFAULT_EXECUTOR_BACKEND,
    DEFAULT_ROLLOUT_EXECUTOR,
    slime_executor_class,
)


@pytest.mark.parametrize(
    "context, expected",
    [
        ({}, "local"),
        ({"requires_resources": True}, "ray"),
        ({"in_ray_placement_group": True}, "ray"),
        ({"local_cuda": True}, "local"),
        ({"local_cuda": True, "in_ray_placement_group": True}, "local"),
    ],
)
def test_service_auto_policy(context, expected):
    decision = select_executor(ExecutorSettings(), role="services", **context)
    assert decision.settings.backend == expected
    assert decision.reason


@pytest.mark.parametrize("backend", ["local", "ray", "custom:Executor"])
def test_explicit_selection_is_never_replaced(backend):
    settings = ExecutorSettings(backend, {"custom_option": 1})
    assert select_executor(settings, role="services", local_cuda=True, requires_resources=True).settings == settings


def test_auto_profile_keeps_options_and_rejects_conflicting_cuda():
    settings = executor_settings({"executors": {"gpu": {"backend": "auto", "options": {"num_gpus": 2}}}}, "gpu")
    result = select_executor(settings, role="services")
    assert result.settings == ExecutorSettings("ray", {"num_gpus": 2})
    with pytest.raises(ValueError, match="cannot combine"):
        select_executor(settings, role="services", local_cuda=True)
    assert executor_settings({}, None).backend == "auto"


@pytest.mark.parametrize("placement_group", [None, object()])
def test_only_existing_ray_placement_context_changes_default(monkeypatch, placement_group):
    ray = ModuleType("ray")
    ray.is_initialized = lambda: True
    util = ModuleType("ray.util")
    util.get_current_placement_group = lambda: placement_group
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", util)
    assert service_executor_selection({}, {}).settings.backend == ("ray" if placement_group else "local")
    assert service_executor_selection({}, {"executor": "local"}).settings.backend == "local"
    assert service_executor_selection({}, {"cuda": 0}).settings.backend == "local"
    ray.is_initialized = lambda: False
    assert service_executor_selection({}, {}).settings.backend == "local"


def test_ray_address_alone_does_not_import_or_select_ray(monkeypatch):
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster:10001")
    assert service_executor_selection({}, {}).settings.backend == "local"
    assert "ray" not in sys.modules


def test_service_resources_auto_select_ray_without_starting_it(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    config = service_executor_config({}, {"resources": {"num_gpus": 2}}, tmp_path, 30, tmp_path / "config.yaml")
    assert config.backend is RayExecutor
    assert config.workers[0].options == {"num_gpus": 2}
    assert "ray" not in sys.modules
    config = service_executor_config({}, {"cuda": 0}, tmp_path, 30, tmp_path / "config.yaml")
    assert config.backend is LocalExecutor


@pytest.mark.parametrize(
    "role, expected", [("training", DEFAULT_EXECUTOR_BACKEND), ("rollout", DEFAULT_ROLLOUT_EXECUTOR)]
)
@pytest.mark.parametrize("backend", [None, "auto", "ray"])
def test_slime_auto_maps_to_specialized_backend_before_launch(monkeypatch, role, expected, backend):
    # Only test class resolution: it must not allocate any GPU resources.
    monkeypatch.setattr(Executor, "get_class", staticmethod(lambda value: value))
    assert slime_executor_class(backend, role=role) == expected


@pytest.mark.parametrize("role", ["training", "rollout"])
@pytest.mark.parametrize("backend", ["local", "mp", "uni"])
def test_slime_rejects_unsupported_builtin_gpu_launchers(role, backend):
    with pytest.raises(ValueError, match="not a GPU launcher"):
        slime_executor_class(backend, role=role)


def test_driver_auto_and_cli_auto_override_yaml(caplog):
    pytest.importorskip("ray")
    from reef.train.slime_backend.reef_adapters.driver import _configure_executors

    args = SimpleNamespace(reef_executor_backend="auto", reef_rollout_executor_backend="auto")
    with caplog.at_level("INFO"):
        _configure_executors(args, {}, [])
    assert args.reef_executor_backend == args.reef_rollout_executor_backend == "ray"
    assert "training: executor=ray" in caplog.text
    assert "Slime model workers" in caplog.text
    args.reef_executor_backend = "auto"
    _configure_executors(args, {"execution": {"training": "custom:Training"}}, ["--reef-executor-backend", "auto"])
    assert args.reef_executor_backend == "ray"


@pytest.mark.parametrize("gpus", [1, 8])
def test_train_group_auto_resolves_before_create(gpus):
    from reef.train.slime_backend.reef_adapters.executors.ray import SlimeRayExecutor
    from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup

    group = SlimeTrainGroup(SimpleNamespace(reef_executor_backend="auto"), 1, gpus, pg=None)
    assert group._executor_config.backend is SlimeRayExecutor
    assert group._executor is None


def test_rollout_entrypoint_accepts_auto():
    from reef.train.slime_backend.reef_adapters.executors.rollout import (
        SlimeRayRolloutExecutor,
        rollout_executor_class,
    )

    assert rollout_executor_class(SimpleNamespace(reef_rollout_executor_backend="auto")) is SlimeRayRolloutExecutor


def test_unknown_service_backend_does_not_fall_back(tmp_path):
    with pytest.raises(ValueError, match="invalid class import path"):
        service_executor_config({}, {"executor": "nonexistent"}, tmp_path, 30, tmp_path / "config.yaml")


@pytest.mark.parametrize(
    "requirements, expected",
    [
        (ExecutionRequirements(), "uni"),
        (ExecutionRequirements(workers=4), "mp"),
        (ExecutionRequirements(gpus_per_worker=1), "ray"),
        (ExecutionRequirements(cluster=True), "ray"),
    ],
)
def test_component_needs_drive_selection_not_installed_hardware(monkeypatch, requirements, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster:10001")
    selected = select_executor(ExecutorSettings(), role="evolution", requirements=requirements)
    assert selected.settings.backend == expected


def test_unsupported_component_launcher_is_rejected():
    requirements = ExecutionRequirements(workers=2, supported_backends=("uni",))
    with pytest.raises(ValueError, match="does not support"):
        select_executor(ExecutorSettings(), role="evolution", requirements=requirements)


def test_scorer_can_declare_gpu_needs_without_yaml():
    from reef.train.cordis_backend.execution import evaluation_selection
    from reef.train.cordis_backend.strategies import EpisodeScorer

    class GPUScorer(EpisodeScorer):
        def execution_requirements(self):
            return ExecutionRequirements(gpus_per_worker=1)

        def __call__(self, task, result):
            return 1.0

    selected, requirements = evaluation_selection(GPUScorer(), 2, ExecutorSettings())
    assert selected.settings.backend == "ray"
    assert requirements.workers == 2 and requirements.gpus_per_worker == 1
    with pytest.raises(ValueError, match="below"):
        evaluation_selection(GPUScorer(), 2, ExecutorSettings(), 0)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1"])
def test_invalid_gpu_requirement_is_rejected(value):
    with pytest.raises(ValueError, match="gpus_per_worker"):
        ExecutionRequirements(gpus_per_worker=value)
