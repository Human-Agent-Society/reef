from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime as StubRuntime

from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.service import deploy
from reef.service.assembly import _repository_location
from reef.service.deploy.config import interpolate_config
from reef.service.deploy.settings import ServiceSettings

OPENCLAWRL_RECIPE = "recipes.openclawrl.recipe:OpenClawRLRecipe"
SAO_RECIPE = "recipes.sao.recipe:SAORecipe"


class _Process:
    """Small Popen stand-in for orchestrator lifecycle tests."""

    def __init__(self, returncode=None, stop_on_poll=None, *, pid=123):
        self.pid = pid
        self.returncode = returncode
        self._stop_on_poll = stop_on_poll

    def poll(self):
        if self._stop_on_poll is not None:
            self._stop_on_poll.set()
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def wait(self, timeout=None):
        return self.returncode


def _example_owned(relative_path: str):
    """An example-owned stack, skipped where the examples do not ship.

    The package job runs this suite against the installed distribution, so
    PROJECT_ROOT is site-packages and nothing under ``recipes/`` ships with
    the wheel. Absent there is correct, not broken.
    """
    return pytest.param(
        relative_path,
        marks=pytest.mark.skipif(
            not (deploy.PROJECT_ROOT / relative_path).exists(),
            reason="example-owned stack: shipped with the repo, not with the package",
        ),
    )


def _settings(**overrides) -> ServiceSettings:
    recipe_settings = {
        "batch_size": 2,
        "checkpoint_every_n_versions": 3,
        **overrides.pop("recipe_settings", {}),
    }
    values = {
        "host": "0.0.0.0",
        "port": 8900,
        "tokens": ("token",),
        "recipe": OPENCLAWRL_RECIPE,
        "ray_address": "ray-head:6379",
        "ray_namespace": "reef",
        "ray_actor_name": "reef-train-bridge",
        "inference_url": "http://ray-head:30000",
        "model_path": "/models/demo",
        "inference_timeout_s": 30.0,
        "inference_backend_factory": None,
        "inference_backend_config": {},
        "inference_retry_initial_s": 0.05,
        "inference_retry_max_s": 1.0,
        "inference_retry_timeout_s": 30.0,
        "artifact_repository": "/state/artifacts.git",
        "artifact_work_dir": "/state/work",
        "artifact_cache_dir": "/state/cache",
        "agent_record_dir": "/state/agent-record",
        "recipe_settings": recipe_settings,
    }
    values.update(overrides)
    return ServiceSettings(**values)


@pytest.mark.unit
def test_service_config_exposes_shared_batch_controls() -> None:
    args = deploy.service_settings_from_config({"reef": {"recipe": OPENCLAWRL_RECIPE, "batch_size": 4}})

    assert args.recipe == OPENCLAWRL_RECIPE
    assert not hasattr(args, "default_recipe")
    # Recipe config fields are not service settings fields: they pass through raw so each
    # default lives with its recipe.
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "groups_per_step")
    assert args.recipe_settings["batch_size"] == "4"
    assert "groups_per_step" not in args.recipe_settings
    assert args.inference_backend_factory is None
    assert args.inference_backend_config == {}
    assert (args.inference_retry_initial_s, args.inference_retry_max_s, args.inference_retry_timeout_s) == (
        0.05,
        1.0,
        300.0,
    )


@pytest.mark.unit
def test_service_config_preserves_inference_backend_config() -> None:
    args = deploy.service_settings_from_config(
        {
            "reef": {
                "recipe": OPENCLAWRL_RECIPE,
                "inference_backend_factory": "example.factory",
                "inference_backend_config": {"tool_call_parser": "qwen25"},
            }
        }
    )

    assert args.inference_backend_config == {"tool_call_parser": "qwen25"}


@pytest.mark.unit
def test_service_config_preserves_candidate_evaluation_section() -> None:
    evaluation = {
        "module": "cookbook.evaluation:build_evaluator",
        "config": {"threshold": 0.8},
    }

    args = deploy.service_settings_from_config({"reef": {"recipe": SAO_RECIPE}, "evaluation": evaluation})

    assert args.evaluation_settings == evaluation


@pytest.mark.unit
def test_service_config_rejects_non_object_evaluation_section() -> None:
    with pytest.raises(ValueError, match="evaluation must be an object"):
        deploy.service_settings_from_config({"reef": {"recipe": SAO_RECIPE}, "evaluation": "disabled"})


@pytest.mark.unit
def test_service_config_selects_inference_recipe_and_interpolates_settings() -> None:
    args = deploy.service_settings_from_config(
        {
            "reef": {
                "recipe": "recipe",
                "port": 9000,
                "inference_url": "http://127.0.0.1:${provider.port}",
            },
            "provider": {"port": 30000},
        }
    )

    assert args.recipe == "recipe"
    assert not hasattr(args, "default_recipe")
    assert args.port == 9000
    assert args.inference_url == "http://127.0.0.1:30000"


@pytest.mark.unit
def test_service_config_requires_recipe() -> None:
    with pytest.raises(ValueError, match=r"reef\.recipe"):
        deploy.service_settings_from_config({"reef": {}})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_name", "recipe"),
    [
        ("recipes/basic/local-sglang.yaml", "recipe"),
        ("recipes/basic/external-provider.yaml", "recipe"),
        ("recipes/openclawrl/examples/openclawrl/serve.yaml", OPENCLAWRL_RECIPE),
    ],
)
def test_cookbook_configs_launch_internal_service_from_reef_settings(
    config_name: str,
    recipe: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = deploy.PROJECT_ROOT / config_name
    if not config_path.exists():
        pytest.skip("repo-owned stack: shipped with the repo, not with the package")
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    config = deploy.load_config(config_path)
    service = next(item for item in config["services"] if item["name"] == "reef")
    args = deploy.service_settings_from_config(config)

    assert service["command"] == [sys.executable, "-m", "reef.service"]
    assert "REEF_TOKEN" not in service.get("env", {})
    assert args.tokens == ()
    assert args.recipe == recipe


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_path",
    [
        # example-owned stacks (the openclawrl config lives with its example,
        # like recipes/tttd/examples/tttd/serve.yaml)
        _example_owned("recipes/openclawrl/examples/openclawrl/serve.yaml"),
    ],
)
def test_cookbook_training_configs_leave_max_staleness_unset(relative_path) -> None:
    config = deploy.load_config(deploy.PROJECT_ROOT / relative_path)

    assert "max_staleness" not in config["reef"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_path",
    (_example_owned("recipes/openclawrl/examples/openclawrl/serve.yaml"),),
)
def test_training_configs_make_checkpoint_budget_mandatory(relative_path, monkeypatch) -> None:
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    config = deploy.load_config(deploy.PROJECT_ROOT / relative_path)
    retention = config["training"]["checkpoint_retention"]
    command = interpolate_config(
        config,
        next(service["command"] for service in config["services"] if service["name"] == "slime-driver"),
    )

    assert retention["max_storage_fraction"] == 0.8
    assert retention["min_free_space_fraction"] == 0.1
    assert retention["policy"] == "latest"
    assert "--reef-checkpoint-max-storage-fraction=0.8" in command
    assert "--reef-checkpoint-min-free-space-fraction=0.1" in command


@pytest.mark.unit
def test_build_dispatcher_connects_runtime_and_injects_selected_recipe(monkeypatch, tmp_path) -> None:
    connected = {}
    backend = {}

    def connector(**kwargs):
        connected.update(kwargs)
        return StubRuntime()

    def factory(repository, **kwargs):
        backend["repository"] = repository
        backend.update(kwargs)
        return lambda scenario: object()

    monkeypatch.setattr(deploy.GitLFSRepositoryBackend, "factory", factory)

    dispatcher = deploy.build_dispatcher(
        _settings(agent_record_dir=str(tmp_path / "agent-record")),
        environ={},
        connector=connector,
    )

    assert dispatcher._recipe.name == "openclawrl"
    assert connected == {
        "inference_url": "http://ray-head:30000",
        "actor_name": "reef-train-bridge",
        "namespace": "reef",
        "ray_address": "ray-head:6379",
        "model_path": "/models/demo",
        "inference_timeout_s": 30.0,
        "train_timeout_s": None,
    }
    assert backend["repository"] == Path("/state/artifacts.git")


@pytest.mark.unit
def test_build_dispatcher_loads_recipe_for_inference_service(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe="recipe",
            agent_record_dir=str(tmp_path / "agent-record"),
        )
    )

    assert dispatcher._recipe.name == "recipe"


@pytest.mark.unit
def test_build_dispatcher_uses_the_recipe_name_for_a_dotted_reference(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )
    selected = "reef.recipe.base:Recipe"

    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe=selected,
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={},
    )

    assert dispatcher._recipe.name == "recipe"


@pytest.mark.unit
def test_build_dispatcher_passes_recipe_settings_to_a_dotted_reference(monkeypatch, tmp_path) -> None:
    module = tmp_path / "demo_evolution.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe="reef.train.cordis_backend.recipe:CordisRecipe",
            recipe_settings={
                "evolution": {
                    "propose": "demo_evolution:propose",
                    "evaluate": "demo_evolution:evaluate",
                    "tasks": ["task one"],
                    "adapter": "pi",
                }
            },
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={},
    )

    recipe = dispatcher._recipe
    assert recipe.adapter == "pi"
    assert recipe.tasks == ("task one",)


@pytest.mark.unit
def test_build_dispatcher_applies_common_recipe_controls(monkeypatch, tmp_path) -> None:
    captured = {}

    def factory(repository, **kwargs):
        return lambda scenario: object()

    monkeypatch.setattr(deploy.GitLFSRepositoryBackend, "factory", factory)
    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe=OPENCLAWRL_RECIPE,
            recipe_settings={"batch_size": 4, "checkpoint_every_n_versions": 5},
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={},
        connector=lambda **kwargs: StubRuntime(),
    )
    recipe = dispatcher._recipe
    captured["batch_size"] = recipe.batch_size
    captured["checkpoint_strategy"] = recipe.checkpoint_strategy

    assert captured == {"batch_size": 4, "checkpoint_strategy": EveryNVersions(5)}


@pytest.mark.unit
def test_build_dispatcher_injects_candidate_evaluation_into_weight_recipe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )
    evaluation = {
        "module": "reef_service._candidate_evaluation_plugin:build_evaluator",
        "config": {"score": 1.0, "threshold": 0.0},
    }

    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe=OPENCLAWRL_RECIPE,
            evaluation_settings=evaluation,
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={"EVALUATION_TOKEN": "secret"},
        connector=lambda **kwargs: StubRuntime(),
    )

    recipe = dispatcher._recipe
    assert recipe.candidate_evaluation is not None
    assert recipe.candidate_evaluation.module == evaluation["module"]


@pytest.mark.unit
def test_build_dispatcher_rejects_candidate_evaluation_for_non_weight_recipe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    with pytest.raises(ValueError, match="evaluation section requires a weight-training recipe"):
        deploy.build_dispatcher(
            _settings(
                recipe="recipe",
                evaluation_settings={"module": "cookbook.evaluation:build_evaluator"},
                agent_record_dir=str(tmp_path / "agent-record"),
            )
        )


@pytest.mark.unit
def test_build_dispatcher_rejects_recipe_settings_nothing_consumes(monkeypatch, tmp_path) -> None:
    """A recipe-shaped reef.* key no config field consumes must fail the boot loudly:
    the alternative is a deployment silently running on recipe defaults."""
    from reef.recipe import RecipeConfigError

    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    with pytest.raises(RecipeConfigError, match=r"OpenClawRLRecipe does not consume reef\.groups_per_step") as excinfo:
        deploy.build_dispatcher(
            _settings(
                recipe=OPENCLAWRL_RECIPE,
                recipe_settings={"groups_per_step": 4},
                agent_record_dir=str(tmp_path / "agent-record"),
            ),
            environ={},
            connector=lambda **kwargs: StubRuntime(),
        )
    # The error lists what the recipe would consume.
    assert "reef.batch_size" in str(excinfo.value)
    assert "reef.checkpoint_every_n_versions" in str(excinfo.value)


@pytest.mark.unit
def test_build_dispatcher_loads_configured_inference_backend(monkeypatch, tmp_path) -> None:
    connected = {}
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    def connector(**kwargs):
        connected.update(kwargs)
        return StubRuntime()

    dotted_path = "reef.train.slime_backend.reef_adapters.sglang.chat.SGLangChatTrainingInferenceBackend"
    deploy.build_dispatcher(
        _settings(
            inference_backend_factory=dotted_path,
            inference_backend_config={"tool_call_parser": "qwen25"},
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={},
        connector=connector,
    )

    factory = connected["inference_backend_factory"]
    assert factory.__name__ == "SGLangChatTrainingInferenceBackend"
    assert connected["inference_backend_config"] == {"tool_call_parser": "qwen25"}


@pytest.mark.unit
def test_build_dispatcher_treats_sao_as_training_recipe(monkeypatch, tmp_path) -> None:
    # A training recipe must route through the runtime-injection branch, not
    # the inference loader; SAORecipe requires an injected TrainingRuntime and
    # would otherwise raise RecipeConfigError at load time.
    connected = {}

    def connector(**kwargs):
        connected.update(kwargs)
        return StubRuntime(max_staleness=kwargs["max_staleness"])

    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    dispatcher = deploy.build_dispatcher(
        _settings(
            recipe=SAO_RECIPE,
            recipe_settings={"max_staleness": 2},
            agent_record_dir=str(tmp_path / "agent-record"),
        ),
        environ={},
        connector=connector,
    )

    assert dispatcher._recipe.name == "sao"
    # The runtime connector was invoked (inference branch never connects).
    assert connected["actor_name"] == "reef-train-bridge"
    assert connected["max_staleness"] == 2
    assert dispatcher._recipe.max_staleness == 2


@pytest.mark.unit
@pytest.mark.parametrize("max_staleness", [0, 1, 2])
def test_build_dispatcher_resolves_max_staleness_environment_for_runtime(
    monkeypatch,
    tmp_path,
    max_staleness,
) -> None:
    connected = {}
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    def connector(**kwargs):
        connected.update(kwargs)
        return StubRuntime(max_staleness=kwargs.get("max_staleness", 0))

    dispatcher = deploy.build_dispatcher(
        _settings(recipe=OPENCLAWRL_RECIPE, agent_record_dir=str(tmp_path / "agent-record")),
        environ={"REEF_MAX_STALENESS": str(max_staleness)},
        connector=connector,
    )

    assert connected.get("max_staleness", 0) == max_staleness
    assert dispatcher._recipe.max_staleness == max_staleness


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ["ray_address", "model_path"])
def test_build_dispatcher_requires_runtime_locations(attribute, monkeypatch) -> None:
    monkeypatch.setattr(
        deploy.GitLFSRepositoryBackend,
        "factory",
        lambda *args, **kwargs: lambda scenario: object(),
    )

    with pytest.raises(ValueError, match="required"):
        deploy.build_dispatcher(
            _settings(**{attribute: ""}),
            environ={},
            connector=lambda **kwargs: StubRuntime(),
        )


@pytest.mark.unit
def test_repository_location_preserves_remote_git_locations() -> None:
    assert _repository_location("https://git.example/repository") == "https://git.example/repository"
    assert _repository_location("git@example:repository") == "git@example:repository"


def test_service_tokens_merge_token_and_tokens_and_drop_empties() -> None:
    config = {
        "reef": {
            "recipe": "openclawrl",
            "token": "alice",
            "tokens": ["bob", "", "alice", "  carol  "],
        }
    }
    assert deploy.service_settings_from_config(config).tokens == ("alice", "bob", "carol")


def test_service_tokens_rejects_non_list() -> None:
    config = {"reef": {"recipe": "openclawrl", "tokens": "alice,bob"}}
    with pytest.raises(ValueError, match=r"reef\.tokens must be a list"):
        deploy.service_settings_from_config(config)


def test_reef_token_is_service_owned_and_never_reaches_the_recipe() -> None:
    """``reef.token`` feeds ``ServiceSettings.tokens`` under another name; the
    recipe-owned remainder of the section must still exclude it, or every
    training recipe would reject the cookbook configs as unconsumed settings."""
    from reef.service.assembly import _recipe_owned_settings

    config = {"reef": {"recipe": "openclawrl", "token": "secret", "tokens": ["next"], "batch_size": 2}}
    owned = _recipe_owned_settings(deploy.service_settings_from_config(config))
    assert set(owned) == {"batch_size"}


@pytest.mark.unit
def test_stack_places_the_bridge_ready_marker_under_run_dir(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy.orchestrator import _Stack

    monkeypatch.delenv("REEF_BRIDGE_READY_FILE", raising=False)
    config = {"reef": {"port": 8900}}
    stack = _Stack(config, [{"name": "slime-driver"}], tmp_path / "stack", 60, tmp_path / "serve.yaml")

    env = stack._service_env({"name": "slime-driver", "env": {"RAY_ADDRESS": "127.0.0.1:${reef.port}"}})

    assert env["REEF_BRIDGE_READY_FILE"] == str(tmp_path / "stack" / "bridge.ready")
    assert env["RAY_ADDRESS"] == "127.0.0.1:8900"
    assert env["REEF_CONFIG"] == str(tmp_path / "serve.yaml")


@pytest.mark.unit
def test_stack_graceful_shutdown_ignores_deliberate_child_termination(tmp_path: Path) -> None:
    from reef.service.deploy.orchestrator import _Stack

    stack = _Stack({}, [{"name": "service"}], tmp_path, 60, tmp_path / "serve.yaml")
    process = _Process()
    stack._procs["service"] = process

    stack.shutdown(grace=0)

    assert process.returncode == -signal.SIGTERM
    assert stack.exit_code == 0


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle")
def test_stack_shutdown_terminates_descendants_after_the_service_leader_exits(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy.orchestrator import _Stack

    run_dir = tmp_path / "stack"
    run_dir.mkdir()
    child_ready = tmp_path / "child.ready"
    child_stopped = tmp_path / "child.stopped"
    child_code = f"""
import os
import signal
import time
from pathlib import Path

def stop(signum, frame):
    Path({str(child_stopped)!r}).write_text(str(signum))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
Path({str(child_ready)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(1)
"""
    leader_code = f"""
import subprocess
import sys

subprocess.Popen([sys.executable, "-c", {child_code!r}])
"""
    service = {
        "name": "service",
        "command": [sys.executable, "-c", leader_code],
    }
    stack = _Stack({}, [service], run_dir, 60, tmp_path / "serve.yaml")
    # This test targets lifecycle, not readiness polling.
    monkeypatch.setattr(stack, "_wait_ready", lambda service, process: None)

    try:
        stack.start()
        leader = stack._procs["service"]
        deadline = time.monotonic() + 5
        while not child_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_ready.exists(), "descendant did not start"
        child_pid = int(child_ready.read_text())
        assert leader.wait(timeout=5) == 0
        assert os.getpgid(child_pid) == leader.pid

        stack.shutdown(grace=2)

        assert child_stopped.read_text() == str(int(signal.SIGTERM))
    finally:
        group = stack._process_groups.get("service")
        if group is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle")
def test_stack_shutdown_signals_descendants_after_the_leader_exits(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy import orchestrator

    stack = orchestrator._Stack({}, [{"name": "service"}], tmp_path, 60, tmp_path / "serve.yaml")
    stack._procs["service"] = _Process(returncode=0)
    stack._process_groups["service"] = 123
    group_alive = True
    signals = []

    def killpg(pgid, signum):
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append((pgid, signum))
        group_alive = False

    monkeypatch.setattr(orchestrator.os, "killpg", killpg)
    monkeypatch.setattr(
        orchestrator.os,
        "getpgid",
        lambda pid: (_ for _ in ()).throw(ProcessLookupError),
    )
    monkeypatch.setattr(orchestrator.os, "getpgrp", lambda: 999)

    stack.shutdown(grace=1)

    assert signals == [(123, signal.SIGTERM)]


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle")
def test_stack_shutdown_escalates_surviving_groups_in_reverse_dependency_order(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy import orchestrator

    stack = orchestrator._Stack(
        {},
        [{"name": "dependency"}, {"name": "dependent"}],
        tmp_path,
        60,
        tmp_path / "serve.yaml",
    )
    stack._procs = {
        "dependency": _Process(pid=101),
        "dependent": _Process(pid=202),
    }
    stack._process_groups = {"dependency": 101, "dependent": 202}
    alive_groups = {101, 202}
    signals = []

    def killpg(pgid, signum):
        if signum == 0:
            if pgid not in alive_groups:
                raise ProcessLookupError
            return
        signals.append((pgid, signum))
        if signum == signal.SIGKILL:
            alive_groups.remove(pgid)

    monkeypatch.setattr(orchestrator.os, "killpg", killpg)
    monkeypatch.setattr(orchestrator.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(orchestrator.os, "getpgrp", lambda: 999)

    stack.shutdown(grace=0)

    assert signals == [
        (202, signal.SIGTERM),
        (101, signal.SIGTERM),
        (202, signal.SIGKILL),
        (101, signal.SIGKILL),
    ]


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle")
def test_stack_shutdown_refuses_a_reused_process_group_id(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy import orchestrator

    stack = orchestrator._Stack({}, [{"name": "service"}], tmp_path, 60, tmp_path / "serve.yaml")
    stack._procs["service"] = _Process(returncode=0, pid=123)
    stack._process_groups["service"] = 123
    signals = []

    monkeypatch.setattr(orchestrator.os, "getpgid", lambda pid: 123)
    monkeypatch.setattr(orchestrator.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(orchestrator.os, "killpg", lambda pgid, signum: signals.append((pgid, signum)))

    stack.shutdown(grace=0)

    assert signals == []


@pytest.mark.unit
@pytest.mark.parametrize("returncode", [0, 1])
def test_stack_treats_any_unexpected_child_exit_as_failure(tmp_path: Path, returncode: int) -> None:
    from reef.service.deploy.orchestrator import _Stack

    stack = _Stack({}, [{"name": "service"}], tmp_path, 60, tmp_path / "serve.yaml")
    stack._procs["service"] = _Process(returncode)

    stack._watchdog()

    assert stack.exit_code == 1


@pytest.mark.unit
def test_watchdog_does_not_reclassify_a_signal_driven_child_exit(tmp_path: Path) -> None:
    from reef.service.deploy.orchestrator import _Stack

    stack = _Stack({}, [{"name": "service"}], tmp_path, 60, tmp_path / "serve.yaml")
    # Simulate a signal arriving after the watchdog began its polling pass but
    # before shutdown made the child exit.
    stack._procs["service"] = _Process(-signal.SIGTERM, stop_on_poll=stack._stopping)

    stack._watchdog()

    assert stack.exit_code == 0


@pytest.mark.unit
def test_stack_installs_signal_handlers_before_starting_watchdog(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy import orchestrator

    stack = orchestrator._Stack({}, [], tmp_path, 60, tmp_path / "serve.yaml")
    handlers = {}
    started = []

    monkeypatch.setattr(orchestrator.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))

    class Watcher:
        def start(self):
            assert signal.SIGINT in handlers and signal.SIGTERM in handlers
            started.append(True)
            stack._stopping.set()

    monkeypatch.setattr(orchestrator.threading, "Thread", lambda **kwargs: Watcher())

    stack.block()

    assert started == [True]
