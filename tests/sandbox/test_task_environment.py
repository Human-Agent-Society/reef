"""The Harbor task environments that ``reef.execution.sandbox.task_environment`` owns: their readiness probes, and
the environment config of a trial with Reef's E2B environment class."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
from pathlib import Path

import pytest

from reef.execution.sandbox.remote import REMOTE_TASK_ENVIRONMENTS
from reef.execution.sandbox.task_environment import (
    HARBOR_TASK_ENVIRONMENTS,
    PROBED_TASK_ENVIRONMENTS,
    TASK_ENVIRONMENT_PROBES,
    DockerProbe,
    EnvironmentVariableProbe,
    ModuleNamesProbe,
    ModuleProbe,
    checked_task_environment,
    environment_config,
    task_environment_probes,
)


@pytest.mark.unit
def test_every_probed_environment_is_a_harbor_task_environment() -> None:
    assert ("docker", *REMOTE_TASK_ENVIRONMENTS) == HARBOR_TASK_ENVIRONMENTS
    assert PROBED_TASK_ENVIRONMENTS == HARBOR_TASK_ENVIRONMENTS
    assert tuple(TASK_ENVIRONMENT_PROBES) == PROBED_TASK_ENVIRONMENTS


@pytest.mark.unit
def test_only_a_harbor_task_environment_has_probes() -> None:
    for name in ("docker", "e2b", "modal"):
        assert checked_task_environment(name) == name
    for value in ("podman", "Docker", "", None, 3):
        with pytest.raises(ValueError, match=r"^the Harbor environment must be docker or e2b or modal, not "):
            checked_task_environment(value)

    class EqualsDocker:
        """Not a str, but equal to one of the names."""

        def __eq__(self, other: object) -> bool:
            return other == "docker"

        def __hash__(self) -> int:
            return hash("docker")

    with pytest.raises(ValueError, match="must be docker or e2b or modal"):
        checked_task_environment(EqualsDocker())
    with pytest.raises(ValueError, match=r"^generator.environment must be docker or e2b or modal, not 'podman'$"):
        checked_task_environment("podman", "generator.environment")
    with pytest.raises(ValueError, match="must be docker or e2b or modal, not 'podman'"):
        task_environment_probes("podman")


@pytest.mark.unit
def test_each_environment_names_what_its_host_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [probe.name for probe in task_environment_probes("docker")] == ["docker"]
    e2b = task_environment_probes("e2b", credential_hint="export it before reef serve starts")
    assert [probe.name for probe in e2b] == [*REMOTE_TASK_ENVIRONMENTS["e2b"], "e2b", "dockerfile_parse"]
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    assert e2b[0].missing() == "E2B_API_KEY is not set; export it before reef serve starts"
    assert (
        task_environment_probes("e2b")[0].missing() == "E2B_API_KEY is not set; export it before this process starts"
    )
    assert EnvironmentVariableProbe("E2B_API_KEY").missing() == "E2B_API_KEY is not set"
    # Modal: its two credential variables, and the package that Harbor's modal environment imports.
    modal = task_environment_probes("modal", credential_hint="export it before reef serve starts")
    assert [probe.name for probe in modal] == [*REMOTE_TASK_ENVIRONMENTS["modal"], "modal"]
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert modal[1].missing() == "MODAL_TOKEN_SECRET is not set; export it before reef serve starts"


@pytest.mark.unit
def test_a_probe_refuses_values_it_cannot_check() -> None:
    with pytest.raises(ValueError, match="non-empty module name"):
        ModuleProbe("")
    with pytest.raises(ValueError, match="non-empty command"):
        DockerProbe(docker="")
    for timeout_s in (0, -1.0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive number of seconds"):
            DockerProbe(timeout_s=timeout_s)
    with pytest.raises(ValueError, match="non-empty environment variable name"):
        EnvironmentVariableProbe("")
    with pytest.raises(ValueError, match="non-empty module name"):
        ModuleNamesProbe("", ("A",))
    with pytest.raises(ValueError, match="tuple of non-empty attribute names"):
        ModuleNamesProbe("e2b", ["A"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuple of non-empty attribute names"):
        ModuleNamesProbe("e2b", ("A", ""))


@pytest.mark.unit
def test_a_probe_answers_with_a_reason_and_never_raises(tmp_path: Path) -> None:
    reason = ModuleProbe("no_such_package_anywhere.sub").missing()
    assert reason.startswith(f"no_such_package_anywhere.sub does not import under {sys.executable}")
    if os.name != "posix":
        return
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexec sleep 5\n")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    reason = DockerProbe(docker=str(docker), timeout_s=0.2).missing()
    assert reason.startswith(f"{docker} version did not answer:"), reason


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


def harbor_e2b_environments(tmp_path: Path, agent: dict[str, object] | None = None) -> dict[str, object]:
    """Reef's E2B environment and Harbor's, made by Harbor's factory for one task as a trial makes them.

    ``reef`` has the config of ``environment_config`` for the Harbor agent config ``agent``; ``harbor`` has the same
    config without its ``import_path``, which is Harbor's own class. Both have the same session, trial environment and
    network allowlist.
    """
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.task.task import Task
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
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


@pytest.mark.unit
def test_an_e2b_trial_gets_reefs_class_on_harbors_template_with_the_tasks_limits(tmp_path: Path) -> None:
    pytest.importorskip("e2b")
    from harbor.environments.e2b import E2BEnvironment
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.trial.config import EnvironmentConfig

    from reef.execution.sandbox.e2b import KillOnTimeoutE2BEnvironment

    environments = harbor_e2b_environments(tmp_path)
    reef, harbor = environments["reef"], environments["harbor"]
    assert type(reef) is KillOnTimeoutE2BEnvironment and type(harbor) is E2BEnvironment
    assert reef.type() == harbor.type() == "e2b"
    # Harbor's private names: the same template, so one that Harbor's class built is used and none is built again.
    assert reef.environment_id == harbor.environment_id and reef._template_name == harbor._template_name
    # The lifetime is the task's build, agent and verifier limits plus an hour. The task's resources stay its own.
    assert reef._sandbox_timeout == harbor._sandbox_timeout == 120 + 900 + 300 + 3600
    assert (reef.task_env_config.cpus, reef.task_env_config.memory_mb) == (2, 4096)
    # A time limit of the trial's own agent config replaces the task's 900 s in the lifetime that the class gets.
    longer = harbor_e2b_environments(tmp_path / "longer", {"override_timeout_sec": 7200})["reef"]
    assert type(longer) is KillOnTimeoutE2BEnvironment and longer._sandbox_timeout == 120 + 7200 + 300 + 3600
    # Harbor's job checks read the class's resource capabilities through the import path: E2B's.
    config = EnvironmentConfig.model_validate(environment_config("e2b", tmp_path / "task"))
    resources = EnvironmentFactory.resource_capabilities(config.type, config.import_path)
    assert resources == E2BEnvironment.resource_capabilities()
    EnvironmentFactory.validate_resource_policies(config)


@pytest.mark.unit
def test_reefs_e2b_sandbox_is_created_as_harbors_and_asks_to_be_deleted_at_its_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2b = pytest.importorskip("e2b")
    environments = harbor_e2b_environments(tmp_path)
    calls: dict[str, list[dict[str, object]]] = {name: [] for name in environments}
    created = object()

    def recording(made: list[dict[str, object]]) -> staticmethod:
        async def create(**options: object) -> object:
            made.append(options)
            # The first try fails, as a create can for a moment, and the retry policy tries once more.
            if len(made) == 1:
                raise RuntimeError("the service did not answer")
            return created

        return staticmethod(create)

    for name, environment in environments.items():
        monkeypatch.setattr(e2b.AsyncSandbox, "create", recording(calls[name]))
        # The private Harbor method that Reef's class overrides; ``start`` calls it.
        asyncio.run(environment._create_sandbox())
        assert environment._sandbox is created
    assert len(calls["reef"]) == len(calls["harbor"]) == 2
    harbor_call = calls["harbor"][-1]
    assert harbor_call["metadata"] == {"environment_name": "task", "session_id": "t__env"}
    assert harbor_call["envs"] == {"TASK_SETTING": "from-the-task", "TRIAL_SETTING": "from-the-trial"}
    assert harbor_call["network"] == {"allow_out": ["pypi.org"], "deny_out": [e2b.ALL_TRAFFIC]}
    assert harbor_call["timeout"] == 120 + 900 + 300 + 3600 and "lifecycle" not in harbor_call
    # Every argument of Harbor's call, and one more: delete the sandbox at its timeout, never pause it.
    assert calls["reef"][-1] == {**harbor_call, "lifecycle": {"on_timeout": "kill"}}
