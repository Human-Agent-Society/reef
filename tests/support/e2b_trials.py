"""One synthetic Harbor task for the tests of Reef's E2B trial class: its ``task.toml``, its E2B trial environments
as Harbor's factory makes them, and its prepared images."""

from __future__ import annotations

from pathlib import Path

from reef.execution.sandbox.prepared_images import PreparedImage
from reef.execution.sandbox.task_environment import environment_config

#: A task with time limits, resources and an environment variable of its own.
E2B_TASK_TOML = """\
[agent]
timeout_sec = 900

[verifier]
timeout_sec = 300

[environment]
build_timeout_sec = 120
cpus = 2
memory_mb = 4096

[environment.env]
TASK_SETTING = "from-the-task"
"""
#: Synthetic images of the task of ``harbor_e2b_environments`` and of its separate verifier, by digest.
TASK_IMAGE = "registry.example/tasks/task@sha256:" + "c" * 64
VERIFIER_IMAGE = "registry.example/tasks/task-tests@sha256:" + "d" * 64


def harbor_e2b_environments(
    tmp_path: Path,
    agent: dict[str, object] | None = None,
    dockerfile: str = "FROM python:3.12-slim\nWORKDIR /app\n",
) -> dict[str, object]:
    """Reef's E2B environment and Harbor's, made by Harbor's factory for one task as a trial makes them.

    ``reef`` has the config of ``environment_config`` for the Harbor agent config ``agent``; ``harbor`` has the same
    config without its ``import_path``, which is Harbor's own class. Both have the same session, trial environment and
    network allowlist. The task's environment has ``dockerfile`` and a file ``tool``.
    """
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.task.task import Task
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text(dockerfile)
    (task_path / "environment" / "tool").write_text("#!/bin/sh\n")
    (task_path / "tests").mkdir()
    (task_path / "tests" / "test.sh").write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
    (task_path / "instruction.md").write_text("Write nothing.\n")
    (task_path / "task.toml").write_text(E2B_TASK_TOML)
    task = Task(task_path)
    reef_config = environment_config("e2b", task_path, agent)
    configs = {
        "reef": reef_config,
        "harbor": {key: value for key, value in reef_config.items() if key != "import_path"},
    }
    return {
        name: EnvironmentFactory.create_environment_from_config(
            config=EnvironmentConfig.model_validate({**config, "env": {"TRIAL_SETTING": "from-the-trial"}}),
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id="t__env",
            trial_paths=TrialPaths(tmp_path / name),
            task_env_config=task.config.environment.model_copy(deep=True),
            network_policy=NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"]),
        )
        for name, config in configs.items()
    }


def prepared_image(
    directory: str = "environment",
    image: str = TASK_IMAGE,
    *,
    source_image: str | None = None,
    template_id: str | None = "template-1",
    start: str | None = None,
    ready: str | None = None,
) -> PreparedImage:
    """A prepared image of the task of ``harbor_e2b_environments``: by default, the image of its environment, with a
    template and no start command."""
    return PreparedImage(directory, source_image, image, template_id, start, ready)


__all__ = ["E2B_TASK_TOML", "TASK_IMAGE", "VERIFIER_IMAGE", "harbor_e2b_environments", "prepared_image"]
