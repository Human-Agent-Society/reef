"""How a weight-training deployment serves its releases: the serving mode and the serving runtime's config."""

from __future__ import annotations

import pytest

from reef.core.errors import DeployConfigError
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.runtime.jobs import ReleaseFormat
from reef.service.deploy.service_config import ServiceConfig
from reef.service.deploy.serving import ENGINES, NONE, TRAINER, serving_mode, serving_runtime_config
from reef.train.slime_backend.launch import SlimeDeployment
from reef.train.tinker_backend.launch import TinkerDeployment

RECIPE = "recipes.sao.recipe:SAORecipe"
ADAPTER = ReleaseFormat("adapter", 16, ("all",))


def tinker(**changes) -> ServiceConfig:
    return ServiceConfig(
        recipe=RECIPE, training_backend="tinker", training_backend_options={"state-dir": "state"}, **changes
    )


def slime(**changes) -> ServiceConfig:
    return ServiceConfig(recipe=RECIPE, training_backend="slime", **changes)


def test_without_settings_each_backend_serves_in_its_default_mode():
    assert serving_mode(tinker(), TinkerDeployment()) == TRAINER
    assert serving_mode(slime(), SlimeDeployment()) == ENGINES
    assert serving_mode(slime(inference_backend="none"), SlimeDeployment()) == NONE


@pytest.mark.parametrize(
    "setting",
    [
        {"inference_backend": "sglang"},
        {"inference_num_gpus": 2},
        {"tensor_parallel_size": 2},
        {"inference_options": {"context-length": 4096}},
        {"inference_remote": {"url": "https://gpu-host.example"}},
        {"inference_handler_factory": "reef.inference.sglang.chat.SGLangInferenceHandler"},
        {"inference_handler_config": {"tool_call_parser": "qwen25"}},
        {"inference_prefix_sharing": True},
        {"inference_max_loaded_adapters": 4},
    ],
)
def test_any_engine_setting_selects_reefs_engines(setting):
    assert serving_mode(tinker(**setting), TinkerDeployment()) == ENGINES


@pytest.mark.parametrize(
    ("settings", "backend", "message"),
    [
        (tinker(inference_serving=TRAINER, inference_num_gpus=2), TinkerDeployment(), "remove inference.num-gpus"),
        (
            tinker(inference_serving=NONE, inference_max_loaded_adapters=2),
            TinkerDeployment(),
            "remove inference.max-loaded-adapters",
        ),
        (slime(inference_serving=TRAINER), SlimeDeployment(), "serves its own releases"),
        (slime(inference_serving="sampler"), SlimeDeployment(), "engines, trainer, none or module:Factory"),
        (slime(inference_backend="none", inference_serving=ENGINES), SlimeDeployment(), "serves nothing"),
        (tinker(inference_serving_config={"replicas": 2}), TinkerDeployment(), "module:Factory serving runtime only"),
        (slime(inference_serving_config={"replicas": 2}), SlimeDeployment(), "module:Factory serving runtime only"),
    ],
)
def test_a_mode_that_the_settings_contradict_is_refused(settings, backend, message):
    with pytest.raises(DeployConfigError, match=message):
        serving_mode(settings, backend)


def test_engine_serving_runs_the_settings_engines_for_the_components_release_format():
    settings = slime(
        inference_backend="sglang",
        inference_num_gpus=4,
        tensor_parallel_size=2,
        inference_options={"context-length": 4096},
        inference_remote={"url": "https://gpu-host.example"},
        inference_max_loaded_adapters=3,
        inference_prefix_sharing=True,
        inference_handler_factory="reef.inference.sglang.chat.SGLangInferenceHandler",
        inference_handler_config={"tool_call_parser": "qwen25"},
        inference_timeout_s=60.0,
        execution_config={"execution": {"rollout": {"backend": "ray", "options": {"queue": "gpu"}}}},
    )
    assert serving_runtime_config(settings, SlimeDeployment(), ADAPTER) == {
        "type": "remote_serving",
        "remote": {"url": "https://gpu-host.example"},
        "release_format": ADAPTER.to_dict(),
        "engine": "sglang",
        "num_gpus": 4,
        "tensor_parallel_size": 2,
        "options": {"context-length": 4096},
        "adapter_capacity": 3,
        "prefix_sharing": True,
        # The rollout role selects the serving engines' executor.
        "executor": "ray",
        "executor_options": {"queue": "gpu"},
        "inference_timeout_s": 60.0,
        "inference_handler_config": {"tool_call_parser": "qwen25"},
        "inference_handler_factory": SGLangInferenceHandler,
    }


def test_engines_cannot_serve_releases_without_weight_files_or_unresolved_engine_sizes():
    resolved = tinker(inference_backend="sglang", inference_num_gpus=1, tensor_parallel_size=1)
    with pytest.raises(DeployConfigError, match=r"training\.options\.export-adapter: true"):
        serving_runtime_config(resolved, TinkerDeployment(), None)
    with pytest.raises(DeployConfigError, match=r"resolved inference\.backend, inference\.num-gpus"):
        serving_runtime_config(tinker(inference_backend="sglang"), TinkerDeployment(), ADAPTER)
    with pytest.raises(DeployConfigError, match=r"resolved inference\.backend"):
        serving_runtime_config(tinker(inference_num_gpus=1, tensor_parallel_size=1), TinkerDeployment(), ADAPTER)


def test_trainer_and_none_serving_build_the_backends_own_and_the_null_runtime():
    assert serving_runtime_config(tinker(inference_timeout_s=30.0), TinkerDeployment(), None) == {
        "type": "tinker",
        "api_key_env": "TINKER_API_KEY",
        "project_id": None,
        "base_url": None,
        "timeout_s": 300.0,
    }
    assert serving_runtime_config(
        slime(inference_backend="none", inference_timeout_s=30.0), SlimeDeployment(), ReleaseFormat("full")
    ) == {"type": "none", "inference_timeout_s": 30.0}


@pytest.mark.parametrize("release_format", [ADAPTER, None])
def test_a_module_factory_gets_its_serving_config_and_the_release_format(release_format):
    settings = tinker(inference_serving="my.serving:Factory", inference_serving_config={"replicas": 2})
    assert serving_mode(settings, TinkerDeployment()) == "my.serving:Factory"
    assert serving_runtime_config(settings, TinkerDeployment(), release_format) == {
        "replicas": 2,
        "type": "my.serving:Factory",
        "release_format": None if release_format is None else release_format.to_dict(),
    }
