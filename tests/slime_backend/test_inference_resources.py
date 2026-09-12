"""CPU ownership contracts for independently controlled Slime inference."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.train.slime_backend import resources


@pytest.fixture
def resource_runtime(monkeypatch):
    events = []
    allocation = SimpleNamespace(id="shared")
    placements = {"actor": (allocation, [], []), "rollout": (allocation, [], [])}

    def allocate(args):
        events.append("allocate")
        return dict(placements)

    for name, values in (
        ("slime.ray.placement_group", {"create_placement_groups": allocate}),
        ("slime.ray.utils", {"add_default_ray_env_vars": lambda values: values}),
        (
            "reef.train.slime_backend.reef_adapters.worker_hooks",
            {"reef_rollout_env_vars": lambda: {"INFERENCE_TEST": "1"}},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    placement_module = importlib.import_module("ray.util.placement_group")
    monkeypatch.setattr(placement_module, "remove_placement_group", lambda pg: events.append(("release", pg.id)))
    state = SimpleNamespace(events=events, placements=placements, failure=None, config=None)

    class Executor:
        def __init__(self, config):
            state.config = config
            self.workers = ("inference-controller",)
            events.append("create-inference")

        def rpc(self, rank, method, **kwargs):
            events.append(method)
            if method == state.failure:
                raise RuntimeError(f"failed {method}")

        def shutdown(self):
            events.append("kill-controller")

        @classmethod
        def from_workers(cls, workers):
            return SimpleNamespace(workers=workers, owned=False)

    monkeypatch.setattr(resources, "RayExecutor", Executor)
    return state


def test_deployment_owns_inference_and_releases_shared_reservations_once(resource_runtime):
    owner = resources.SlimeInferenceResources()
    owner.start(SimpleNamespace())
    config = resource_runtime.config
    assert config.options["num_gpus"] == 0
    assert config.options["num_cpus"] == 1
    assert config.options["runtime_env"]["env_vars"] == {"INFERENCE_TEST": "1"}
    assert config.workers[0].args[1] is resource_runtime.placements["rollout"]
    assert owner.serving.workers == ("inference-controller",)
    assert owner.serving.owned is False
    owner.close()
    owner.close()
    assert resource_runtime.events == [
        "allocate",
        "create-inference",
        "check_health",
        "shutdown",
        "kill-controller",
        ("release", "shared"),
    ]
    with pytest.raises(RuntimeError, match="not running"):
        _ = owner.serving
    with pytest.raises(RuntimeError, match="once"):
        owner.start(SimpleNamespace())


@pytest.mark.parametrize("failure", ["check_health", "shutdown"])
def test_inference_failure_still_releases_controller_and_allocation(resource_runtime, failure):
    resource_runtime.failure = failure
    owner = resources.SlimeInferenceResources()
    if failure == "check_health":
        with pytest.raises(RuntimeError, match="check_health"):
            owner.start(SimpleNamespace())
    else:
        owner.start(SimpleNamespace())
        with pytest.raises(RuntimeError, match="shutdown"):
            owner.close()
    owner.close()
    assert resource_runtime.events[-2:] == ["kill-controller", ("release", "shared")]
    assert resource_runtime.events.count(("release", "shared")) == 1


@pytest.mark.parametrize("field,value", [("colocate", True), ("rollout_external", True), ("megatron_lora_rank", 8)])
def test_first_stage_rejects_unsupported_modes_before_allocation(resource_runtime, field, value):
    owner = resources.SlimeInferenceResources()
    with pytest.raises(ValueError, match="non-colocated full-weight"):
        owner.start(SimpleNamespace(**{field: value}))
    assert resource_runtime.events == []
