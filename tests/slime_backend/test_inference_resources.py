"""Slime components borrow connections and share one Reef-owned allocation."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.inference.sglang import service as inference_service
from reef.inference.sglang.config import SGLangConfig
from reef.runtime.deployment import ModelDeploymentPlan
from reef.runtime.executor.placement import ModelGpuLayout
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

    def allocate(layout):
        state.layout = layout
        event("allocate")
        return SimpleNamespace(
            training=placements["actor"],
            inference=placements["rollout"],
            release=lambda: event("release-" + allocation.id),
        )

    monkeypatch.setattr(resources, "reserve_model_gpus", allocate)
    for name, values in (
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
            return SimpleNamespace(workers=workers, owned=False, rpc=lambda rank, method, **kwargs: event(method))

    monkeypatch.setattr(inference_service, "RayExecutor", Executor)
    return state


def plan_for(state, *, colocate=False):
    args = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        colocate=colocate,
        rollout_external=False,
        use_critic=False,
    )
    allocation = resources.SlimeDeploymentResources(
        args,
        ray_address="external",
        namespace="test",
        runtime_env={"env_vars": {"PYTHONPATH": "/repo"}},
    )

    class Training:
        weight_transfer_protocol = inference_service.INFERENCE_PROTOCOL

        def start(self, supplied):
            assert supplied is allocation
            state.events.append("training-start")

        def attach_weight_transport(self, session):
            assert session.receiver.owned is False
            state.events.append("training-attach")

        def check_health(self):
            pass

        def close(self):
            state.events.append("training-close")

    return ModelDeploymentPlan(
        allocation,
        inference_service.SGLangInferenceService(SGLangConfig("model", 4, 2, 4, env_vars={"TEST": "1"})),
        Training(),
    )


@pytest.mark.parametrize("colocate", [False, True])
def test_deployment_allocates_once_and_closes_training_inference_then_reservations(resource_runtime, colocate):
    plan = plan_for(resource_runtime, colocate=colocate)
    owner = ModelDeployment(plan)
    assert resource_runtime.events == []
    owner.start()
    config = resource_runtime.config
    assert config.options["num_gpus"] == 0 and config.options["num_cpus"] == 1
    assert config.workers[0].args[1] is resource_runtime.placements["rollout"]
    assert resource_runtime.layout == ModelGpuLayout(training_gpus=2, inference_gpus=4, colocate=colocate)
    assert plan.resources.placement_groups["critic"] is None
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
        "prepare_training_connection",
        "training-start",
        "training-attach",
        "check_health",
        "training-close",
        "shutdown",
        "kill-controller",
        "release-shared",
        "disconnect",
    ]


@pytest.mark.parametrize(
    "failure", ["allocate", "create-inference", "check_health", "prepare_training_connection", "release-shared"]
)
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


def test_training_worker_loss_fails_operations_health_without_waiting_for_another_job():
    from reef.runtime.executor.failure import ExecutorFailedError
    from reef.runtime.executor.uniproc import UniProcExecutor
    from reef.train.slime_backend.reef_adapters.bridge import SlimeTrainingBackend

    class TrainingGroup:
        def __init__(self):
            self.executor = UniProcExecutor.from_workers([object()])

        def register_failure_listener(self, listener):
            self.executor.register_failure_listener(listener)

    group = TrainingGroup()
    operations = SlimeTrainingBackend(group, batch_processor=SimpleNamespace(), save_hf_template=None)
    try:
        operations.start()
        operations.check_health()
        group.executor._fail("worker died", rank=0)
        with pytest.raises(ExecutorFailedError, match="worker died"):
            operations.check_health()
    finally:
        group.executor.shutdown()
