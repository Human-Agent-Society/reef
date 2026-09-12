"""The driver's job runtime_env: its PYTHONPATH must reach the actors it creates.

Cookbook loss families (``recipes.<method>.slime:...``) are resolved inside
Megatron workers. The deploy layer puts the recipe source root on the
*driver's* PYTHONPATH, but Ray actors fork from the raylet — started before
any service environment exists — so the driver must carry its PYTHONPATH
across the job boundary itself.
"""

from __future__ import annotations

import inspect

import pytest

from reef.service.slime_driver import _job_runtime_env, _serve


def test_driver_pythonpath_becomes_the_job_runtime_env():
    env = {"PYTHONPATH": "/repo:/opt/sglang/python", "OTHER": "x"}
    assert _job_runtime_env(env) == {"env_vars": {"PYTHONPATH": "/repo:/opt/sglang/python"}}


def test_empty_or_missing_pythonpath_means_no_runtime_env():
    assert _job_runtime_env({}) is None
    assert _job_runtime_env({"PYTHONPATH": "   "}) is None


def test_serve_initializes_ray_with_the_job_runtime_env():
    # The wiring is textual by necessity — _serve needs a live Ray cluster to
    # run — but the contract it pins is real: the serve path must pass the
    # job runtime_env, or workers on a cluster the driver did not start
    # cannot import the cookbook loss family.
    source = inspect.getsource(_serve)
    assert "runtime_env=_job_runtime_env()" in source


def test_native_training_options_reach_slime_before_legacy_direct_flags(tmp_path, monkeypatch):
    import pytest

    from reef.service import slime_driver

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
    monkeypatch.setattr(
        slime_driver,
        "load_config",
        lambda path: {"reef": {"training_backend_options": {"lr": 1e-6, "use-critic": True}}},
    )
    monkeypatch.setattr(slime_driver, "_resolve_training_recipe", lambda config: ("loss", "recipe", Algorithm()))
    monkeypatch.setattr(slime_driver, "_parse_slime_args", parse)
    with pytest.raises(StopBeforeRuntime):
        slime_driver._serve(["--lr=2e-6"], tmp_path / "ready")
    assert captured == ["--lr=1e-06", "--use-critic", "--lr=2e-6"]


def _driver_lifecycle(monkeypatch, tmp_path, *, mode="managed", failure=None):
    from types import SimpleNamespace

    from reef.service import slime_driver
    from reef.train.slime_backend import resources
    from reef.train.slime_backend.reef_adapters import bridge, slime_arguments

    events = []
    args = SimpleNamespace(colocate=mode == "colocate", rollout_external=mode == "external")

    class Algorithm:
        def parse_driver_options(self, arguments):
            return None, arguments

        def apply_driver_options(self, args, config):
            pass

        def validate_backend_args(self, args, recipe):
            pass

    class Stopping:
        def set(self):
            pass

        def wait(self):
            events.append("wait")

    class Inference:
        serving = object()
        placement_groups = {"actor": "training-pg", "rollout": "inference-pg"}

        def start(self, args):
            events.append("inference-start")
            if failure == "inference-start":
                raise RuntimeError(failure)

        def close(self):
            events.append("inference-close")

    def health():
        events.append("bridge-health")
        return {"ok": failure != "health"}

    def shutdown():
        events.append("bridge-shutdown")
        if failure == "shutdown":
            raise RuntimeError(failure)

    actor = SimpleNamespace(health=SimpleNamespace(remote=health), shutdown=SimpleNamespace(remote=shutdown))

    def prepare(args, **kwargs):
        events.append("prepare")
        return SimpleNamespace(lora=mode == "lora")

    def start(args, **kwargs):
        events.append("bridge-start")
        if mode == "managed":
            assert kwargs["serving"] is Inference.serving
            assert kwargs["placement_groups"] is Inference.placement_groups
        else:
            assert kwargs["serving"] is None
            assert kwargs["placement_groups"] is None
        if failure == "bridge-start":
            raise RuntimeError(failure)
        return actor

    monkeypatch.setenv("RAY_ADDRESS", "external-cluster")
    monkeypatch.setenv("REEF_CONFIG", "unused.yaml")
    monkeypatch.delenv("SLIME_ARGS_FILE", raising=False)
    monkeypatch.setattr(slime_driver, "load_config", lambda path: {})
    monkeypatch.setattr(slime_driver, "_resolve_training_recipe", lambda config: ("loss", "recipe", Algorithm()))
    monkeypatch.setattr(slime_driver, "_parse_slime_args", lambda arguments: args)
    monkeypatch.setattr(slime_driver, "_configure_executors", lambda *args: None)
    monkeypatch.setattr(slime_driver, "_validate_tracking_args", lambda *args: None)
    monkeypatch.setattr(slime_driver, "_apply_bridge_resume_fallback", lambda *args: None)
    monkeypatch.setattr(slime_driver, "_stamp_loss_family_reference", lambda *args: None)
    monkeypatch.setattr(slime_arguments, "configure_reef_loss_args", lambda *args: None)
    monkeypatch.setattr(slime_driver.threading, "Event", Stopping)
    monkeypatch.setattr(slime_driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(slime_driver.ray, "init", lambda **kwargs: events.append("ray-connect"))
    monkeypatch.setattr(slime_driver.ray, "get", lambda result, **kwargs: result)
    monkeypatch.setattr(slime_driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(slime_driver.ray, "kill", lambda *args, **kwargs: events.append("bridge-kill"))
    monkeypatch.setattr(slime_driver.ray, "shutdown", lambda: events.append("ray-disconnect"))
    monkeypatch.setattr(bridge, "prepare_bridge", prepare)
    monkeypatch.setattr(bridge, "start_bridge", start)
    monkeypatch.setattr(resources, "SlimeInferenceResources", Inference)
    return events


def test_driver_owns_inference_outside_training_and_closes_it_last(monkeypatch, tmp_path):
    from reef.service import slime_driver

    events = _driver_lifecycle(monkeypatch, tmp_path)
    ready = tmp_path / "ready"
    assert slime_driver._serve([], ready) == 0
    assert events == [
        "ray-connect",
        "prepare",
        "inference-start",
        "bridge-start",
        "bridge-health",
        "wait",
        "bridge-shutdown",
        "bridge-kill",
        "inference-close",
        "ray-disconnect",
    ]
    assert not ready.exists()


@pytest.mark.parametrize("failure", ["inference-start", "bridge-start", "health", "shutdown"])
def test_driver_failure_releases_inference_after_training(monkeypatch, tmp_path, failure):
    from reef.service import slime_driver

    events = _driver_lifecycle(monkeypatch, tmp_path, failure=failure)
    ready = tmp_path / "ready"
    if failure == "shutdown":
        assert slime_driver._serve([], ready) == 0
    else:
        with pytest.raises(RuntimeError):
            slime_driver._serve([], ready)
    assert events[-2:] == ["inference-close", "ray-disconnect"]
    if "bridge-kill" in events:
        assert events.index("bridge-kill") < events.index("inference-close")
    assert events.count("inference-close") == 1
    assert not ready.exists()


@pytest.mark.parametrize("mode", ["colocate", "external", "lora"])
def test_driver_keeps_unmigrated_modes_on_existing_lifecycle(monkeypatch, tmp_path, mode):
    from reef.service import slime_driver

    events = _driver_lifecycle(monkeypatch, tmp_path, mode=mode)
    assert slime_driver._serve([], tmp_path / "ready") == 0
    assert "inference-start" not in events
    assert "inference-close" not in events
    assert events[-3:] == ["bridge-shutdown", "bridge-kill", "ray-disconnect"]
