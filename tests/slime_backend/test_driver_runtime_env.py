"""The driver's job runtime_env: its PYTHONPATH must reach the actors it creates.

Cookbook loss families (``recipes.<method>.slime:...``) are resolved inside
Megatron workers. The deploy layer puts the recipe source root on the
*driver's* PYTHONPATH, but Ray actors fork from the raylet — started before
any service environment exists — so the driver must carry its PYTHONPATH
across the job boundary itself.
"""

from __future__ import annotations

import pytest

from reef.train.slime_backend.driver import _job_runtime_env


def test_driver_pythonpath_becomes_the_job_runtime_env():
    env = {"PYTHONPATH": "/repo:/opt/sglang/python", "OTHER": "x"}
    assert _job_runtime_env(env) == {"env_vars": {"PYTHONPATH": "/repo:/opt/sglang/python"}}


def test_empty_or_missing_pythonpath_means_no_runtime_env():
    assert _job_runtime_env({}) is None
    assert _job_runtime_env({"PYTHONPATH": "   "}) is None


@pytest.mark.parametrize("managed", [False, True])
def test_native_training_options_reach_slime_before_legacy_direct_flags(tmp_path, monkeypatch, managed):
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
    monkeypatch.delenv("SLIME_ARGS_FILE", raising=False)
    reef = {"training_backend_options": {"lr": 1e-6, "use-critic": True}}
    if managed:
        reef.update(inference_num_gpus=4, tensor_parallel_size=2, inference_options={"context-length": 8192})
    monkeypatch.setattr(slime_driver, "resolve_loss_family", lambda name: Algorithm())
    monkeypatch.setattr(slime_driver, "_parse_slime_args", parse)
    with pytest.raises(StopBeforeRuntime):
        slime_driver.create_model_plan({"reef": reef}, [] if managed else ["--lr=2e-6"], loss_family="loss")
    assert captured == [
        "--lr=1e-06",
        "--use-critic",
        *(
            ["--rollout-num-gpus=4", "--rollout-num-gpus-per-engine=2", "--sglang-context-length=8192"]
            if managed
            else ["--lr=2e-6"]
        ),
    ]


@pytest.mark.parametrize(
    "mode", ["managed", "colocate", "lora", "lora-colocate", "lora-colocate-keep-base", "external"]
)
def test_plan_preflight_selects_components_without_allocating(monkeypatch, mode):
    from types import SimpleNamespace
    from reef.train.slime_backend import driver
    from reef.train.slime_backend.reef_adapters import bridge, slime_arguments

    args = SimpleNamespace(
        hf_checkpoint="model",
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=4,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        colocate="colocate" in mode,
        rollout_external=mode == "external",
        keep_lora_base_resident=mode == "lora-colocate-keep-base",
    )

    class Algorithm:
        def parse_driver_options(self, arguments):
            return None, arguments

        def apply_driver_options(self, args, config):
            pass

        def validate_backend_args(self, args, recipe):
            pass

    monkeypatch.setenv("RAY_ADDRESS", "external")
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.delenv("SLIME_ARGS_FILE", raising=False)
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
    monkeypatch.setattr(bridge, "prepare_bridge", lambda *args, **kwargs: SimpleNamespace(lora="lora" in mode))

    def unexpected(**kwargs):
        pytest.fail("plan construction must not connect or allocate resources")

    monkeypatch.setattr(bridge.ray, "init", unexpected)
    plan = driver.create_model_plan({}, loss_family="loss")
    plan.validate()
    assert plan.inference is not None
    assert plan.resources.allocate_models
    assert (plan.health is not None) == (mode != "external")
    assert plan.resources.placement_groups == {}
    assert plan.resources.runtime_env == {"env_vars": {"PYTHONPATH": "/repo"}}
    assert plan.training.inference_protocol is not None
