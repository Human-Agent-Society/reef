"""Slime's deployment and runtime factory: the flags Slime receives and the runtimes the HTTP service builds."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.core.errors import DeployConfigError
from reef.inference.null import NullInferenceRuntime
from reef.inference.remote.runtime import RemoteServingRuntimeFactory
from reef.runtime.deployment import RuntimeConfigError
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend.launch import SlimeDeployment, runtime_factory, slime_arguments
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments

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


@pytest.mark.parametrize("managed", [False, True])
def test_native_training_options_and_the_serving_engines_size_become_slime_flags(managed):
    settings: dict = {"training_backend_options": {"lr": 1e-6, "use-critic": True}}
    if managed:
        settings.update(inference_num_gpus=4, tensor_parallel_size=2, inference_options={"context-length": 8192})
    assert slime_arguments(settings, loss_family="sao", environ={}) == [
        "--lr=1e-06",
        "--use-critic",
        *(
            ["--rollout-num-gpus=4", "--rollout-num-gpus-per-engine=2", "--sglang-context-length=8192"]
            if managed
            else []
        ),
    ]
    # Without serving engines Slime starts none; vLLM's options never reach Slime's SGLang parser.
    assert slime_arguments(
        {**settings, "inference_num_gpus": 2, "inference_backend": "none"}, loss_family="", environ={}
    )[-1] == ("--rollout-num-gpus=0")
    vllm = slime_arguments(
        {**settings, "inference_num_gpus": 2, "tensor_parallel_size": 2, "inference_backend": "vllm"},
        loss_family="",
        environ={},
    )
    assert not any(argument.startswith("--sglang-") for argument in vllm)


def test_slime_serves_its_checkpoints_on_an_inference_component(tmp_path):
    config = deployment_config(tmp_path)
    assert [service["name"] for service in config["services"]] == ["inference", "reef"]
    settings = settings_of(config)
    assert SlimeDeployment().serves_on_inference_component(settings)
    assert SlimeDeployment().http_environment(settings) == {}
    runtime = SlimeDeployment().runtime_config(settings, loss_family="sao")
    assert (runtime["type"], runtime["loss_family"], runtime["ray_address"]) == (
        "reef.train.slime_backend.launch:runtime_factory",
        "sao",
        "ray://head:10001",
    )
    # Slime and the serving engines read the one resolved model path.
    assert runtime["settings"]["training_backend_options"]["hf-checkpoint"] == MODEL
    assert runtime["serving"]["remote"] == config["reef"]["inference_remote"]
    assert (runtime["serving"]["engine"], runtime["serving"]["num_gpus"]) == ("sglang", 2)

    none = settings_of(deployment_config(tmp_path, inference={"backend": "none", "num-gpus": None}))
    assert not SlimeDeployment().serves_on_inference_component(none)
    assert "serving" not in SlimeDeployment().runtime_config(none, loss_family="sao")


@pytest.mark.parametrize(
    ("inference", "training", "message"),
    [
        # Training and serving never share GPUs: there is no colocation option.
        ({}, {"colocate": True}, "unknown config fields: training.colocate"),
        ({"backend": "vllm"}, {"options": {"global-batch-size": 4, "megatron-lora-rank": 8}}, "LoRA"),
        ({"backend": "vllm", "num-gpus": 2, "tensor-parallel-size": 1}, {}, "one engine per stack"),
        ({}, {"options": {"global-batch-size": 4, "hf-checkpoint": "/models/other"}}, "must match"),
        ({}, {"options": {"lr": 1e-6}}, "global-batch-size"),
    ],
)
def test_the_deployment_refuses_what_slime_cannot_serve(tmp_path, inference, training, message):
    with pytest.raises(DeployConfigError, match=message):
        deployment_config(tmp_path, inference=inference, training=training)


@pytest.mark.parametrize(
    ("name", "value"), [("inference_timeout_s", 0), ("train_timeout_s", 0), ("train_timeout_s", -1)]
)
def test_runtime_config_refuses_a_timeout_that_is_not_positive(tmp_path, name, value):
    settings = {**settings_of(deployment_config(tmp_path)), name: value}
    with pytest.raises(ValueError, match=name):
        SlimeDeployment().runtime_config(settings, loss_family="sao")


class BuiltComponent(StubTrainingComponent):
    """The Slime training component as the factory sees it: its parsed arguments and its close."""

    def __init__(self, args: SlimeArguments) -> None:
        super().__init__(name="slime")
        self.args = args


def worker_arguments(**overrides) -> SlimeArguments:
    """The parsed arguments that the factory reads to describe the serving engines."""
    values = {
        "hf_checkpoint": "/cache/snapshot",
        "seed": 1,
        "fp16": False,
        "use_rollout_routing_replay": False,
        "megatron_lora_rank": 0,
        "megatron_lora_target_modules": None,
        "rollout_num_gpus": 2,
        "rollout_num_gpus_per_engine": 1,
        "num_gpus_per_node": 8,
        "disjoint_prefix_sharing": False,
        "reef_rollout_executor_backend": "auto",
        "reef_rollout_executor_options": {},
    }
    values.update(overrides)
    return SlimeArguments(**values)


@pytest.fixture
def factory_calls(monkeypatch):
    """Replace the Slime component and the serving engines; record what the factory gives them."""
    calls: dict = {"settings": [], "serving": [], "components": [], "args": worker_arguments()}

    def create(settings):
        calls["settings"].append(settings)
        component = BuiltComponent(calls["args"])
        calls["components"].append(component)
        return component

    def serve(factory, config, model_path, recipe_config, environ):
        calls["serving"].append(config)
        if calls.get("serving_error"):
            raise RuntimeError("engines failed to start")
        return StubInferenceRuntime()

    monkeypatch.setattr(slime_component, "create_slime_component", create)
    monkeypatch.setattr(RemoteServingRuntimeFactory, "__call__", serve)
    return calls


def build(config: dict) -> tuple:
    return runtime_factory(config, MODEL, {}, {})


@pytest.mark.parametrize(("lora", "layout", "capacity"), [(False, "full", 1), (True, "adapter", 3)])
def test_the_factory_returns_the_component_and_the_runtime_that_serves_its_checkpoints(
    tmp_path, factory_calls, lora, layout, capacity
):
    if lora:
        factory_calls["args"] = worker_arguments(megatron_lora_rank=8, max_loaded_loras=3)
    runtime = SlimeDeployment().runtime_config(settings_of(deployment_config(tmp_path)), loss_family="sao")
    component, inference = build({**runtime, "train_timeout_s": 600})
    assert component is factory_calls["components"][0]
    assert isinstance(inference, StubInferenceRuntime)
    (settings,) = factory_calls["settings"]
    assert (settings.loss_family, settings.ray_address, settings.train_rpc_timeout_s) == (
        "sao",
        "ray://head:10001",
        600,
    )
    # The deployment's execution sections select the workers' executors.
    assert settings.execution == runtime["execution"]
    assert "--global-batch-size=4" in settings.arguments
    (serving,) = factory_calls["serving"]
    assert (serving["layout"], serving["adapter_capacity"], serving["num_gpus"]) == (layout, capacity, 2)
    # The engines load the model on the component's host, not this host's download.
    assert serving["engine_config"]["model_path"] == MODEL
    assert serving["engine_config"]["external_engines"] == ()


def test_the_factory_without_serving_engines_builds_the_null_runtime(tmp_path, factory_calls):
    settings = settings_of(deployment_config(tmp_path, inference={"backend": "none", "num-gpus": None}))
    _, inference = build(SlimeDeployment().runtime_config(settings, loss_family="sao"))
    assert isinstance(inference, NullInferenceRuntime)
    assert factory_calls["serving"] == []
    # Without a configured timeout the component keeps its default worker timeout.
    assert factory_calls["settings"][0].train_rpc_timeout_s == slime_component.DEFAULT_TRAIN_RPC_TIMEOUT_S


def test_the_factory_needs_a_ray_cluster_and_closes_the_component_when_serving_fails(
    tmp_path, factory_calls, monkeypatch
):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    runtime = SlimeDeployment().runtime_config(settings_of(deployment_config(tmp_path)), loss_family="sao")
    with pytest.raises(RuntimeConfigError, match="Ray cluster"):
        build({**runtime, "ray_address": ""})
    assert factory_calls["components"] == []
    factory_calls["serving_error"] = True
    with pytest.raises(RuntimeError, match="engines failed"):
        build(runtime)
    assert factory_calls["components"][0].closed
