"""Slime components borrow connections and share one Reef-owned allocation."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.runtime.deployment import ModelDeploymentPlan
from reef.service.training_driver import ModelDeployment
from reef.train.slime_backend import resources


@pytest.fixture
def resource_runtime(monkeypatch):
    events = []
    allocation = SimpleNamespace(id="shared")
    placements = {"actor": (allocation, [], []), "rollout": (allocation, [], [])}
    state = SimpleNamespace(events=events, placements=placements, failure=None, config=None, initialized=False)

    def event(name):
        events.append(name)
        if name == state.failure:
            raise RuntimeError(f"failed {name}")

    def connect(**kwargs):
        state.initialized = True
        state.ray_options = kwargs
        event("connect")

    def disconnect():
        state.initialized = False
        event("disconnect")

    def allocate(args):
        state.args = args
        event("allocate")
        return dict(placements)

    for name, values in (
        ("slime.ray.placement_group", {"create_placement_groups": allocate}),
        ("slime.ray.utils", {"add_default_ray_env_vars": lambda values: values}),
        ("reef.train.slime_backend.reef_adapters.worker_hooks", {"reef_rollout_env_vars": lambda: {"TEST": "1"}}),
    ):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(resources.ray, "is_initialized", lambda: state.initialized)
    monkeypatch.setattr(resources.ray, "init", connect)
    monkeypatch.setattr(resources.ray, "shutdown", disconnect)
    monkeypatch.setattr(resources.ray, "nodes", list)
    placement_module = importlib.import_module("ray.util.placement_group")
    monkeypatch.setattr(placement_module, "remove_placement_group", lambda pg: event("release-" + pg.id))

    class Executor:
        def __init__(self, config):
            state.config = config
            self.workers = ("inference-controller",)
            event("create-inference")

        def rpc(self, rank, method, **kwargs):
            event(method)

        def shutdown(self):
            event("kill-controller")

        @classmethod
        def from_workers(cls, workers):
            return SimpleNamespace(workers=workers, owned=False)

    monkeypatch.setattr(resources, "RayExecutor", Executor)
    return state


def plan_for(state):
    args = SimpleNamespace(rollout_num_gpus=4, rollout_num_gpus_per_engine=2)
    allocation = resources.SlimeDeploymentResources(
        args,
        ray_address="external",
        namespace="test",
        runtime_env={"env_vars": {"PYTHONPATH": "/repo"}},
    )

    class Training:
        inference_protocol = resources.INFERENCE_PROTOCOL

        def start(self, supplied, inference):
            assert supplied is allocation
            assert inference.control.owned is False
            state.events.append("training-start")

        def check_health(self):
            pass

        def close(self):
            state.events.append("training-close")

    return ModelDeploymentPlan(allocation, resources.SlimeInferenceService(args), Training())


def test_deployment_allocates_once_and_closes_training_inference_then_reservations(resource_runtime):
    plan = plan_for(resource_runtime)
    owner = ModelDeployment(plan)
    assert resource_runtime.events == []
    owner.start()
    config = resource_runtime.config
    assert config.options["num_gpus"] == 0 and config.options["num_cpus"] == 1
    assert config.workers[0].args[1] is resource_runtime.placements["rollout"]
    assert resource_runtime.args.rollout_num_gpus == 4
    assert resource_runtime.args.rollout_num_gpus_per_engine == 2
    assert resource_runtime.ray_options == {
        "address": "external",
        "namespace": "test",
        "runtime_env": {
            "env_vars": {"PYTHONPATH": "/repo", resources.DEPLOYMENT_ENV: plan.resources._process_lease},
            "worker_process_setup_hook": "reef.runtime.executor.process_guard.install",
        },
    }
    owner.close()
    owner.close()
    assert resource_runtime.events == [
        "connect",
        "allocate",
        "create-inference",
        "check_health",
        "training-start",
        "check_health",
        "training-close",
        "shutdown",
        "kill-controller",
        "release-shared",
        "disconnect",
    ]


@pytest.mark.parametrize("failure", ["allocate", "create-inference", "check_health", "release-shared"])
def test_partial_failure_always_disconnects_the_owned_ray_job(resource_runtime, failure):
    resource_runtime.failure = failure
    owner = ModelDeployment(plan_for(resource_runtime))
    with pytest.raises(RuntimeError, match=failure):
        owner.start()
        owner.close()
    owner.close()
    assert resource_runtime.events[-1] == "disconnect"
    if failure != "allocate":
        assert resource_runtime.events.count("release-shared") == 1


def test_existing_client_session_is_not_disconnected(resource_runtime):
    resource_runtime.initialized = True
    owner = ModelDeployment(plan_for(resource_runtime))
    with pytest.raises(RuntimeError, match="own Ray client session"):
        owner.start()
    assert resource_runtime.events == []
    assert resource_runtime.initialized is True


@pytest.mark.parametrize("confirmed", [True, False])
def test_failed_component_shutdown_requires_process_retirement(resource_runtime, monkeypatch, confirmed):
    plan = plan_for(resource_runtime)
    owner = ModelDeployment(plan)
    owner.start()
    resource_runtime.failure = "shutdown"
    plan.resources._nodes = ["node"]

    def retire():
        resource_runtime.events.append("retire-processes")
        if not confirmed:
            raise RuntimeError("process cleanup unconfirmed")

    monkeypatch.setattr(plan.resources, "_retire_processes", retire)
    if confirmed:
        owner.close()
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            owner.close()
    assert resource_runtime.events[-2:] == ["retire-processes", "disconnect"]


def test_training_worker_loss_fails_health_without_waiting_for_another_job():
    from reef.runtime.executor.failure import ExecutorFailedError, ExecutorFailure
    from reef.train.slime_backend.training import SlimeTrainingService

    service = SlimeTrainingService(
        SimpleNamespace(),
        preparation=SimpleNamespace(),
        loss_family_config=None,
        actor_name="bridge",
        namespace="test",
        separate_inference=True,
    )
    service.on_executor_failure(ExecutorFailure("test", "worker died", rank=1))
    with pytest.raises(ExecutorFailedError, match="worker died"):
        service.check_health()
    with pytest.raises(ExecutorFailedError, match="worker died"):
        service.poll()


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("failure", [None, "health", "shutdown"])
def test_training_adapter_attaches_without_allocating_or_closing_inference(
    resource_runtime, monkeypatch, separate, failure
):
    from reef.train.slime_backend import training

    allocation = plan_for(resource_runtime).resources
    allocation.allocate_models = separate
    allocation.start()
    borrowed = SimpleNamespace(owned=False)
    connection = resources.InferenceConnection(resources.INFERENCE_PROTOCOL, borrowed) if separate else None

    def shutdown():
        resource_runtime.events.append("training-close")
        if failure == "shutdown":
            raise RuntimeError("shutdown failed")

    bridge = SimpleNamespace(
        health=SimpleNamespace(remote=lambda: {"ok": failure != "health"}),
        shutdown=SimpleNamespace(remote=shutdown),
    )

    def start(args, **kwargs):
        assert kwargs["serving"] is (borrowed if separate else None)
        assert kwargs["placement_groups"] is (allocation.placement_groups if separate else None)
        return bridge

    monkeypatch.setattr(training, "start_bridge", start)

    def get(result, **kwargs):
        if isinstance(result, dict):
            assert "timeout" not in kwargs  # Startup health waits for checkpoint recovery.
        return result

    monkeypatch.setattr(training.ray, "get", get)
    monkeypatch.setattr(training.ray, "kill", lambda *args, **kwargs: resource_runtime.events.append("kill-training"))
    service = training.SlimeTrainingService(
        allocation.args,
        preparation=SimpleNamespace(),
        loss_family_config=None,
        actor_name="test",
        namespace="test",
        separate_inference=separate,
    )
    if separate:
        with pytest.raises(ValueError, match="existing inference connection"):
            service.start(allocation, None)
    service.start(allocation, connection)
    if failure == "health":
        with pytest.raises(RuntimeError, match="health check"):
            service.check_health()
    else:
        service.check_health()
    if failure == "shutdown" and not separate:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            service.close()
    else:
        service.close()
    service.close()
    assert resource_runtime.events == [
        "connect",
        *(["allocate"] if separate else []),
        "training-close",
        "kill-training",
    ]
    allocation.close()
