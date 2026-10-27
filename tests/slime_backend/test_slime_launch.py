"""Slime's deployment and runtime factory: the flags Slime receives and the training component the HTTP service builds."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubTrainingComponent

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import RuntimeConfigError
from reef.runtime.jobs import ReleaseFormat
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.serving import serving_runtime_config
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend.launch import SlimeDeployment, component_factory, slime_arguments

MODEL = "/models/demo"


def deployment_config(tmp_path: Path, *, inference: dict | None = None, training: dict | None = None) -> dict:
    """The resolved configuration of a Slime deployment of SAO."""
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {"model-path": MODEL, "num-gpus": 2, **(inference or {})},
        "training": {
            "backend": "slime",
            "ray-address": "ray://head:10001",
            "options": {"global-batch-size": 4, "lr": 1e-6},
            **(training or {}),
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    return config


def settings_of(config: dict) -> dict:
    return service_config_from_mapping(config).__dict__


def test_native_training_options_become_slime_flags_and_slime_starts_no_engine():
    settings = {
        "training_backend_options": {"lr": 1e-6, "use-critic": True},
        # The inference section configures the serving engines, never Slime.
        "inference_num_gpus": 4,
        "tensor_parallel_size": 2,
        "inference_options": {"context-length": 8192},
    }
    assert slime_arguments(settings, loss_family="sao", environ={}) == [
        "--lr=1e-06",
        "--use-critic",
        "--rollout-num-gpus=0",
    ]


def test_slime_trains_and_the_inference_section_serves_its_checkpoints_on_an_inference_component(tmp_path):
    config = deployment_config(tmp_path)
    assert [service["name"] for service in config["services"]] == ["inference", "reef"]
    settings = settings_of(config)
    assert SlimeDeployment().http_environment(settings) == {}
    runtime = SlimeDeployment().component_config(settings, loss_family="sao")
    assert (runtime["type"], runtime["loss_family"], runtime["ray_address"]) == (
        "reef.train.slime_backend.launch:component_factory",
        "sao",
        "ray://head:10001",
    )
    # Slime and the serving engines read the one resolved model path; Slime gets no serving settings.
    assert runtime["settings"]["training_backend_options"]["hf-checkpoint"] == MODEL
    assert set(runtime["settings"]) == {
        "model_path",
        "training_backend_options",
        "teacher_model_path",
        "teacher_timeout",
    }
    assert not {"serving", "inference_timeout_s"} & runtime.keys()
    serving = serving_runtime_config(service_config_from_mapping(config), SlimeDeployment(), ReleaseFormat("full"))
    assert serving["remote"] == config["reef"]["inference_remote"]
    assert (serving["type"], serving["engine"], serving["num_gpus"]) == ("remote_serving", "sglang", 2)

    none = deployment_config(tmp_path, inference={"backend": "none", "num-gpus": None})
    assert [service["name"] for service in none["services"]] == ["reef"]
    assert serving_runtime_config(service_config_from_mapping(none), SlimeDeployment(), None)["type"] == "none"


@pytest.mark.parametrize(
    ("inference", "training", "message"),
    [
        # Training and serving never share GPUs: there is no colocation option.
        ({}, {"colocate": True}, "unknown config fields: training.colocate"),
        ({}, {"options": {"global-batch-size": 4, "offload-rollout": True}}, "shares GPUs"),
        ({}, {"options": {"global-batch-size": 4, "sglang-mem-fraction-static": 0.5}}, "configures inference"),
        ({}, {"options": {"global-batch-size": 4, "rollout-num-gpus": 2}}, "configures inference"),
        ({"backend": "vllm", "num-gpus": 2, "tensor-parallel-size": 1}, {}, "vllm serves one engine"),
        ({}, {"options": {"global-batch-size": 4, "hf-checkpoint": "/models/other"}}, "must match"),
        ({}, {"options": {"lr": 1e-6}}, "global-batch-size"),
        ({}, {"options": {"global-batch-size": 4, "use-rollout-routing-replay": True}}, "routed experts"),
    ],
)
def test_the_deployment_refuses_what_slime_cannot_train_or_serve(tmp_path, inference, training, message):
    with pytest.raises(DeployConfigError, match=message):
        deployment_config(tmp_path, inference=inference, training=training)


@pytest.fixture
def created_components(monkeypatch):
    """Replace the Slime component; record the settings the factory gives it and the components it built."""
    calls: dict = {"settings": [], "components": []}

    def create(settings):
        calls["settings"].append(settings)
        calls["components"].append(StubTrainingComponent(name="slime"))
        return calls["components"][-1]

    monkeypatch.setattr(slime_component, "create_slime_component", create)
    return calls


def test_the_factory_builds_only_the_training_component(tmp_path, created_components):
    runtime = SlimeDeployment().component_config(settings_of(deployment_config(tmp_path)), loss_family="sao")
    component = component_factory({**runtime, "train_timeout_s": 600}, MODEL, {})
    assert component is created_components["components"][0]
    (settings,) = created_components["settings"]
    assert (settings.loss_family, settings.ray_address, settings.train_timeout_s) == (
        "sao",
        "ray://head:10001",
        600,
    )
    # The deployment's execution sections select the training workers' executor.
    assert settings.execution == runtime["execution"]
    assert "--global-batch-size=4" in settings.arguments and "--rollout-num-gpus=0" in settings.arguments
    # Without a configured timeout the component keeps its default worker timeout.
    component_factory(runtime, MODEL, {})
    assert created_components["settings"][1].train_timeout_s == slime_component.DEFAULT_TRAIN_TIMEOUT_S


@pytest.mark.parametrize("value", [0, -1])
def test_the_factory_refuses_a_train_timeout_that_is_not_positive(tmp_path, created_components, value):
    runtime = SlimeDeployment().component_config(settings_of(deployment_config(tmp_path)), loss_family="sao")
    with pytest.raises(ValueError, match="train_timeout_s must be positive"):
        component_factory({**runtime, "train_timeout_s": value}, MODEL, {})
    assert created_components["settings"] == []


def test_the_factory_needs_a_ray_cluster(tmp_path, created_components, monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    runtime = SlimeDeployment().component_config(settings_of(deployment_config(tmp_path)), loss_family="sao")
    with pytest.raises(RuntimeConfigError, match="Ray cluster"):
        component_factory({**runtime, "ray_address": ""}, MODEL, {})
    assert created_components["components"] == []
