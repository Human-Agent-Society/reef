"""The driver's job runtime_env: its PYTHONPATH must reach the actors it creates.

Cookbook loss families (``recipes.<method>.slime:...``) are resolved inside
Megatron workers. The deploy layer puts the recipe source root on the
*driver's* PYTHONPATH, but Ray actors fork from the raylet — started before
any service environment exists — so the driver must carry its PYTHONPATH
across the job boundary itself.
"""

from __future__ import annotations

import inspect

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
