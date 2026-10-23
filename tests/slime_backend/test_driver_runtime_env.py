"""The driver's job runtime_env: its PYTHONPATH must reach the actors it creates.

Cookbook loss families (``recipes.<method>.slime:...``) are resolved inside
Megatron workers. The deploy layer puts the recipe source root on the
*driver's* PYTHONPATH, but Ray actors fork from the raylet — started before
any service environment exists — so the driver must carry its PYTHONPATH
across the job boundary itself.
"""

from __future__ import annotations

import pytest
import ray

from reef.train.slime_backend.driver import _job_runtime_env


def test_driver_pythonpath_becomes_the_job_runtime_env():
    env = {"PYTHONPATH": "/repo:/opt/sglang/python", "OTHER": "x"}
    assert _job_runtime_env(env) == {"env_vars": {"PYTHONPATH": "/repo:/opt/sglang/python"}}


def test_empty_or_missing_pythonpath_means_no_runtime_env():
    assert _job_runtime_env({}) is None
    assert _job_runtime_env({"PYTHONPATH": "   "}) is None


@pytest.mark.parametrize("managed", [False, True])
def test_native_training_options_reach_slime(tmp_path, monkeypatch, managed):
    import pytest

    from reef.train.slime_backend import driver as slime_driver

    class StopBeforeRuntime(Exception):
        pass

    class Algorithm:
        def parse_driver_options(self, arguments):
            return None, arguments

    captured = []

    def parse(arguments):
        captured.extend(arguments)
        raise StopBeforeRuntime

    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setenv("REEF_CONFIG", "unused.yaml")
    reef = {"training_backend_options": {"lr": 1e-6, "use-critic": True}}
    if managed:
        reef.update(inference_num_gpus=4, tensor_parallel_size=2, inference_options={"context-length": 8192})
    monkeypatch.setattr(slime_driver, "resolve_loss_family", lambda name: Algorithm())
    monkeypatch.setattr(slime_driver, "_parse_slime_args", parse)
    with pytest.raises(StopBeforeRuntime):
        slime_driver.create_training_plan({"reef": reef}, loss_family="loss")
    assert captured == [
        "--lr=1e-06",
        "--use-critic",
        *(
            ["--rollout-num-gpus=4", "--rollout-num-gpus-per-engine=2", "--sglang-context-length=8192"]
            if managed
            else []
        ),
    ]


def plan_args(mode):
    from types import SimpleNamespace

    return SimpleNamespace(
        hf_checkpoint="model",
        seed=1,
        offload_rollout=False,
        fp16=False,
        use_rollout_routing_replay=False,
        megatron_lora_rank=0,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=4,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        colocate="colocate" in mode,
        rollout_external=mode == "external",
        keep_lora_base_resident=mode == "lora-colocate-keep-base",
        disjoint_prefix_sharing=False,
        save_interval=1,
    )


def patch_driver(monkeypatch, args, *, lora):
    """Plan construction on these ``args``, with Slime's parsing and Ray connection replaced."""
    from types import SimpleNamespace

    from reef.train.slime_backend import driver
    from reef.train.slime_backend.reef_adapters import bridge, slime_arguments

    class Algorithm:
        def parse_driver_options(self, arguments):
            return None, arguments

        def apply_driver_options(self, args, config):
            pass

        def validate_backend_args(self, args, recipe):
            pass

    monkeypatch.setenv("RAY_ADDRESS", "external")
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.setattr(driver, "resolve_loss_family", lambda name: Algorithm())
    monkeypatch.setattr(driver, "_parse_slime_args", lambda arguments: args)
    for name in (
        "_configure_executors",
        "_validate_tracking_args",
        "_apply_bridge_resume_fallback",
        "_stamp_loss_family_reference",
    ):
        monkeypatch.setattr(driver, name, lambda *args: None)
    monkeypatch.setattr(slime_arguments, "configure_reef_loss_args", lambda *args: None)
    monkeypatch.setattr(bridge, "prepare_bridge", lambda *args, **kwargs: SimpleNamespace(lora=lora))

    def unexpected(**kwargs):
        pytest.fail("plan construction must not connect or allocate resources")

    monkeypatch.setattr(ray, "init", unexpected)
    return driver


@pytest.mark.parametrize(
    "mode", ["managed", "colocate", "lora", "lora-colocate", "lora-colocate-keep-base", "external"]
)
@pytest.mark.parametrize(
    "timeout_config", [{}, {"train_timeout_s": None}, {"train_timeout_s": 43200}, {"train_timeout_s": 0.5}]
)
def test_plan_preflight_selects_components_without_allocating(monkeypatch, mode, timeout_config):
    from reef.service.training_driver import assemble_model_plan

    driver = patch_driver(monkeypatch, plan_args(mode), lora="lora" in mode)
    training = driver.create_training_plan({"reef": timeout_config}, loss_family="loss")
    configured_timeout = timeout_config.get("train_timeout_s")
    expected_timeout = 14400 if configured_timeout is None else configured_timeout
    assert training.training.train_rpc_timeout_s == expected_timeout
    assert not hasattr(training, "inference")
    plan = assemble_model_plan({}, training)
    plan.validate()
    assert plan.inference is not None
    assert plan.resources.allocate_models
    assert (plan.health is not None) == (mode != "external")
    assert plan.resources.placement_groups == {}
    assert plan.resources.runtime_env == {"env_vars": {"PYTHONPATH": "/repo"}}
    assert plan.training.weight_transfer_protocols


def test_plan_with_remote_engines_reserves_training_gpus_only_and_sends_the_files_model(monkeypatch):
    from reef.inference.remote.service import RemoteInferenceService
    from reef.runtime.deployment import WEIGHT_FILES_PROTOCOL
    from reef.runtime.executor.placement import ModelGpuLayout
    from reef.service.training_driver import assemble_model_plan
    from reef.train.slime_backend.resources import model_gpu_layout

    args = plan_args("managed")
    args.hf_checkpoint = "/cache/snapshot"
    driver = patch_driver(monkeypatch, args, lora=False)
    monkeypatch.setenv("REEF_REMOTE_CONTROL_TOKEN", "control-secret")
    remote = {
        "url": "https://gpu-host.example",
        "control_token_env": "REEF_REMOTE_CONTROL_TOKEN",
        "inference_token_env": "REEF_REMOTE_INFERENCE_TOKEN",
        "startup_timeout_s": 3600.0,
        "model_path": "Qwen/Qwen3-8B",
    }
    reef = {"inference_remote": remote, "inference_num_gpus": 4, "tensor_parallel_size": 2}
    training = driver.create_training_plan({"reef": reef}, loss_family="loss")
    assert training.resources.remote_inference
    assert model_gpu_layout(args, remote_inference=True) == ModelGpuLayout(4, 4, external_inference=True)
    assert model_gpu_layout(args, remote_inference=True).total == 4
    assert (training.inference_config["engine"], training.inference_config["num_gpus"]) == ("sglang", 4)
    assert training.inference_config["engine_config"]["model_path"] == "Qwen/Qwen3-8B"
    plan = assemble_model_plan({"reef": reef}, training)
    assert isinstance(plan.inference, RemoteInferenceService)
    assert plan.weight_transfer_protocol(plan.inference) == WEIGHT_FILES_PROTOCOL

    lora_driver = patch_driver(monkeypatch, plan_args("lora"), lora=True)
    with pytest.raises(RuntimeError, match="only full weights to remote engines"):
        lora_driver.create_training_plan({"reef": reef}, loss_family="loss")
