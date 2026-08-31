"""Harbor-dependent tests: skipped cleanly when harbor isn't installed.

These pin the *mapping* between reef-eval and Harbor (config assembly, task
validation) without running containers. Full end-to-end runs live in
examples/ and require Docker.
"""

import os
from pathlib import Path
from unittest import mock

import pytest

harbor = pytest.importorskip("harbor")


def test_agent_dict_maps_onto_agent_config():
    from harbor.models.trial.config import AgentConfig

    cfg = AgentConfig.model_validate(
        {"name": "claude-code", "model_name": "anthropic/claude-opus-5"}
    )
    assert cfg.name == "claude-code"


def test_trial_config_assembly_matches_executor():
    from harbor.models.trial.config import TaskConfig, TrialConfig

    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"name": "org/some-task"}).model_dump(),
            "trials_dir": "trials",
            "agent": {"name": "nop"},
            "verifier": {"disable": True},
        }
    )
    assert config.agent.name == "nop"
    assert config.verifier.disable is True


def test_state_mount_maps_onto_trial_config():
    """A stream's state_dir override must assemble into valid Harbor config:
    a bind mount on the main service plus REEF_EVAL_STATE_DIR on both sides."""
    from harbor.models.trial.config import TaskConfig, TrialConfig

    from reef_eval.executors import STATE_TARGET, apply_state_mount

    agent, overrides = apply_state_mount({"name": "nop"}, {}, "/host/streams/s1/state")
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"name": "org/some-task"}).model_dump(),
            "trials_dir": "trials",
            "agent": agent,
            **overrides,
        }
    )
    assert config.environment.mounts == [
        {"type": "bind", "source": "/host/streams/s1/state", "target": STATE_TARGET}
    ]
    assert config.environment.env["REEF_EVAL_STATE_DIR"] == STATE_TARGET
    assert config.agent.env["REEF_EVAL_STATE_DIR"] == STATE_TARGET


def test_exemplar_task_is_valid_stock_harbor(request):
    """The shipped autoresearch exemplar must parse under Harbor's TaskConfig
    schema: the 'tasks stay 100% stock harbor' promise, enforced."""
    import tomllib
    from pathlib import Path

    from harbor.models.task.config import TaskConfig

    task_dir = (
        Path(request.config.rootpath)
        / "tasks"
        / "autoresearch"
        / "first-party"
        / "circle-packing"
    )
    raw = tomllib.loads((task_dir / "task.toml").read_text())
    cfg = TaskConfig.model_validate(raw)
    assert cfg.verifier is not None
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "solution" / "solve.sh").exists()


def test_unset_compose_variables_are_caught_before_the_run(tmp_path):
    """A task whose compose needs a variable nobody sets would run to a score
    of 0 that looks like the agent's fault: Compose reads an unset variable as
    an empty string, so the mount silently points elsewhere."""
    from reef_eval.executors import _unset_compose_vars

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes:\n      - ${DATA_PATH}/x:/x:ro\n"
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        assert _unset_compose_vars(tmp_path) == ["DATA_PATH"]

        # Compose reads a .env beside the compose file, so the check must too.
        (environment / ".env").write_text("# a comment\nDATA_PATH=/somewhere\n")
        assert _unset_compose_vars(tmp_path) == []

        (environment / ".env").unlink()
        assert _unset_compose_vars(tmp_path) == ["DATA_PATH"]

    # An exported variable counts just the same.
    with mock.patch.dict(os.environ, {"DATA_PATH": "/somewhere"}):
        assert _unset_compose_vars(tmp_path) == []


def test_self_contained_tasks_need_nothing_set():
    from reef_eval.executors import _unset_compose_vars

    root = Path(__file__).parent.parent / "tasks"
    assert _unset_compose_vars(root / "_template") == []
