"""The HTTP service builds Slime's training component, then the runtime that serves its checkpoints."""

from __future__ import annotations

import pytest
from tests.support.runtime_stubs import StubTrainingComponent

from reef.inference.null import NullInferenceRuntime
from reef.runtime.deployment import RuntimeConfigError, RuntimeFactory
from reef.service.assembly import connect_training_component
from reef.service.deploy.service_config import ServiceConfig

# The Slime component module imports Ray; the base installation has none.
slime_component = pytest.importorskip("reef.train.slime_backend.component", exc_type=ImportError)


class ClosingComponent(StubTrainingComponent):
    """A stub component that records whether the factory released it."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.fixture
def created(monkeypatch):
    """The settings each Slime component was created from, and the components, without Ray or GPUs."""
    calls: list[tuple[slime_component.SlimeSettings, ClosingComponent]] = []

    def create(settings):
        component = ClosingComponent()
        calls.append((settings, component))
        return component

    monkeypatch.setattr(slime_component, "create_slime_component", create)
    return calls


def training_settings(**changes) -> ServiceConfig:
    values = {
        "recipe": "recipes.sao.recipe:SAORecipe",
        "training_backend": "slime",
        "inference_backend": "none",
        "inference_num_gpus": 0,
        "ray_address": "ray://head:10001",
        "ray_namespace": "training-cluster",
        "train_timeout_s": 14400,
        "training_backend_options": {"global-batch-size": 8},
        "execution_config": {"execution": {"training": {"backend": "ray", "options": {"queue": "gpu"}}}},
    }
    return ServiceConfig(**{**values, **changes})


def test_slime_without_serving_builds_the_component_and_a_null_runtime(created):
    component, runtime = connect_training_component(training_settings(), model_path="model", loss_family="sao")
    try:
        [(settings, built)] = created
        assert component is built
        assert isinstance(runtime, NullInferenceRuntime)
        assert settings.arguments == ("--global-batch-size=8", "--rollout-num-gpus=0")
        assert (settings.loss_family, settings.recipe) == ("sao", "recipes.sao.recipe:SAORecipe")
        assert (settings.ray_address, settings.namespace) == ("ray://head:10001", "training-cluster")
        assert settings.train_timeout_s == 14400.0
        # The deployment's execution section selects the workers' executors.
        assert settings.execution == {"execution": {"training": {"backend": "ray", "options": {"queue": "gpu"}}}}
    finally:
        runtime.shutdown()
        component.close()


def test_slime_requires_a_ray_cluster_before_creating_the_component(created, monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    with pytest.raises(RuntimeConfigError, match=r"training\.ray-address or RAY_ADDRESS"):
        connect_training_component(training_settings(ray_address=None), model_path="model", loss_family="sao")
    assert created == []


def test_a_serving_factory_that_builds_no_inference_runtime_releases_both_components(created, monkeypatch):
    built: list[ClosingComponent] = []

    class WrongServingFactory(RuntimeFactory):
        kind = "none"

        def __call__(self, config, model_path, recipe_config, environ):
            built.append(ClosingComponent())
            return built[-1]

    monkeypatch.setattr("reef.runtime.deployment._runtime_kinds", {"none": WrongServingFactory()})
    with pytest.raises(TypeError, match="built a training component: a runtime serves"):
        connect_training_component(training_settings(), model_path="model", loss_family="sao")
    [(_, component)] = created
    assert component.closed and built[0].closed
