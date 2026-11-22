"""The Harbor task environments that ``reef.execution.sandbox.task_environment`` owns: their readiness probes, and
the environment config of a trial with Reef's E2B environment class."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.unit
def test_reefs_e2b_start_waits_for_the_tmp_cleanup_of_systemd_after_harbors_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    from harbor.environments.base import ExecResult
    from harbor.environments.e2b import E2BEnvironment

    from reef.execution.sandbox import e2b as reef_e2b

    steps: list[tuple[object, ...]] = []

    async def harbor_start(environment: E2BEnvironment, force_build: bool) -> None:
        steps.append(("harbor start", force_build))

    async def run(environment: E2BEnvironment, command: str, **options: object) -> ExecResult:
        steps.append((command, options))
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(E2BEnvironment, "start", harbor_start)
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "exec", run)
    asyncio.run(harbor_e2b_environments(tmp_path)["reef"].start(force_build=False))
    assert steps == [
        ("harbor start", False),
        (
            reef_e2b.SYSTEMD_TMP_CLEANUP_WAIT,
            {"timeout_sec": reef_e2b.SYSTEMD_TMP_CLEANUP_WAIT_SECONDS, "user": "root"},
        ),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("first_process", "systemctl_answers", "questions"),
    [
        ("systemd", True, ["is-active", "is-system-running", "is-active"]),
        # systemctl hangs, so timeout stops it after the first question, and the wait ends.
        ("systemd", False, ["is-active"]),
        ("bash", True, []),
    ],
)
def test_the_systemd_tmp_cleanup_wait_waits_only_for_a_systemd_that_answers(
    tmp_path: Path, first_process: str, systemctl_answers: bool, questions: list[str]
) -> None:
    import subprocess

    from reef.execution.sandbox.e2b import SYSTEMD_TMP_CLEANUP_WAIT

    tools = tmp_path / "bin"
    tools.mkdir()
    asked = tmp_path / "systemctl-questions"
    hang = "" if systemctl_answers else f'[ "$2" = systemctl ] && echo "$3" >> {asked} && exit 124\n'
    stand_ins = {
        # The first process of the image.
        "cat": f'[ "$1" = /proc/1/comm ] && echo {first_process}\n',
        # The cleanup step is active at the second is-active; the boot does not end.
        "systemctl": f'echo "$1" >> {asked}\n[ "$1" = is-active ] && [ "$(grep -c is-active {asked})" -ge 2 ]\n',
        # coreutils timeout exits with 124 when it stops the command.
        "timeout": f'{hang}shift\nexec "$@"\n',
        "sleep": "true\n",
    }
    for name, body in stand_ins.items():
        (tools / name).write_text("#!/bin/sh\n" + body)
        (tools / name).chmod(0o755)
    path = f"{tools}:{os.environ['PATH']}"
    subprocess.run(["sh", "-c", SYSTEMD_TMP_CLEANUP_WAIT], check=True, env={**os.environ, "PATH": path}, timeout=30)
    assert (asked.read_text().split() if asked.exists() else []) == questions


@pytest.mark.unit
def test_a_dockerfile_for_e2b_sets_the_modes_of_copy_chmod_with_chmod(tmp_path: Path) -> None:
    pytest.importorskip("dockerfile_parse")
    from reef.execution.sandbox.e2b import dockerfile_with_copy_modes

    (tmp_path / "target" / "samples").mkdir(parents=True)
    for name in ("real", "a.bin", "b.bin", "samples/one"):
        (tmp_path / "target" / name).write_text("")
    dockerfile = (
        "# syntax=docker/dockerfile:1\n"
        "FROM ubuntu:24.04\n"
        "COPY --chmod=0500 target/real /opt/target/.real\n"
        "COPY --chmod=0644 target/samples/ /opt/target/samples/\n"
        "USER solver:solver\n"
        "COPY --chown=solver --chmod=4755 \\\n"
        "    target/*.bin /opt/bin/\n"
        "COPY target/real /app/real\n"
    )
    assert dockerfile_with_copy_modes(dockerfile, tmp_path) == (
        "# syntax=docker/dockerfile:1\n"
        "FROM ubuntu:24.04\n"
        "COPY target/real /opt/target/.real\n"
        "RUN chmod 0500 /opt/target/.real\n"
        # A copied directory gets the mode with everything in it, as Docker's BuildKit sets it.
        "COPY target/samples/ /opt/target/samples/\n"
        "RUN chmod -R 0644 /opt/target/samples/\n"
        "USER solver:solver\n"
        # Each file of the pattern is in the destination directory. Only root can set the setuid bit.
        "COPY --chown=solver target/*.bin /opt/bin/\n"
        "USER root\n"
        "RUN chmod 4755 /opt/bin/a.bin && chmod 4755 /opt/bin/b.bin\n"
        "USER solver:solver\n"
        "COPY target/real /app/real\n"
    )
    # A Dockerfile without --chmod is the same text.
    plain = "FROM ubuntu:24.04\n\nCOPY target/real /app/real\nUSER solver\n"
    assert dockerfile_with_copy_modes(plain, tmp_path) == plain


@pytest.mark.unit
def test_reefs_e2b_template_is_harbors_and_keeps_the_modes_of_copy_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox.e2b import COPY_MODES_TEMPLATE_SUFFIX

    def recording(builds: dict[str, dict[str, object]], name: str) -> staticmethod:
        async def build(**options: object) -> None:
            template = options.pop("template")
            builds[name] = {**options, "steps": json.loads(e2b.Template.to_json(template))["steps"]}

        return staticmethod(build)

    def built(environments: dict[str, object]) -> dict[str, dict[str, object]]:
        builds: dict[str, dict[str, object]] = {}
        for name, environment in environments.items():
            monkeypatch.setattr(e2b.AsyncTemplate, "build", recording(builds, name))
            # The private Harbor method that Reef's class overrides; ``start`` calls it for a missing template.
            asyncio.run(environment._create_template())
        return builds

    # Without --chmod, Reef's class builds Harbor's template, with the task's resources, under Harbor's alias.
    plain = built(harbor_e2b_environments(tmp_path / "plain"))
    assert plain["reef"] == plain["harbor"]
    assert (plain["reef"]["cpu_count"], plain["reef"]["memory_mb"]) == (2, 4096)

    dockerfile = "FROM python:3.12-slim\nUSER solver\nCOPY --chmod=0500 tool /opt/tool\n"
    environments = harbor_e2b_environments(tmp_path / "modes", dockerfile=dockerfile)
    modes = built(environments)
    # Harbor's class drops the mode. Reef's class sets it as root, under an alias of its own.
    assert modes["reef"]["alias"] == modes["harbor"]["alias"] + COPY_MODES_TEMPLATE_SUFFIX
    assert environments["reef"]._template_name == modes["reef"]["alias"]
    steps = [(step["type"], step["args"]) for step in modes["reef"]["steps"]]
    assert steps[steps.index(("USER", ["solver"])) + 1 :] == [
        ("COPY", ["tool", "/opt/tool", "", ""]),
        ("USER", ["root"]),
        ("RUN", ["chmod 0500 /opt/tool"]),
        ("USER", ["solver"]),
        # E2B's default directory for a Dockerfile without WORKDIR.
        ("WORKDIR", ["/home/user"]),
    ]
    assert not any(step["type"] == "RUN" for step in modes["harbor"]["steps"])


@pytest.mark.unit
def test_with_aenv_builds_reefs_class_builds_the_tasks_dockerfile_with_aenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_build import AgentEnvTemplates, DockerfileBuild

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example/")
    monkeypatch.setenv("E2B_API_KEY", "key")
    dockerfile = "FROM debian AS build\nRUN make\nFROM python:3.12-slim\nCOPY --chmod=0500 tool /opt/tool\n"
    environments = harbor_e2b_environments(tmp_path, dockerfile=dockerfile)
    reef, harbor = environments["reef"], environments["harbor"]
    calls: list[tuple[AgentEnvTemplates, DockerfileBuild]] = []

    async def ensure(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        calls.append((templates, build))

    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", ensure)
    # The alias is its own: an E2B template build of AgentENV, without the task's files, never answers for it.
    assert reef._template_name == harbor._template_name + reef_e2b.AENV_TEMPLATE_SUFFIX
    # Reef's class always reads the alias's state itself; AgentENV's alias check also answers for a failed build.
    assert asyncio.run(reef._does_template_exist()) is False
    asyncio.run(reef._create_template())
    templates, build = calls[0]
    assert templates == AgentEnvTemplates("https://api.agentenv.example", "key")
    environment_dir = tmp_path / "task" / "environment"
    # The Dockerfile as the task wrote it: BuildKit builds the stages and sets the mode. The task's resources and
    # build time limit; no start command, as the last stage has none.
    assert build == DockerfileBuild(
        context_dir=environment_dir,
        dockerfile=environment_dir / "Dockerfile",
        alias=reef._template_name,
        cpu_count=2,
        memory_mb=4096,
        timeout_seconds=120,
        start_command=False,
    )
    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "kaniko")
    with pytest.raises(ValueError, match="REEF_E2B_DOCKERFILE_BUILD must be one of e2b, aenv"):
        harbor_e2b_environments(tmp_path / "other", dockerfile=dockerfile)


@pytest.mark.unit
def test_a_task_template_built_before_the_trials_is_the_template_of_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_build import AgentEnvTemplates, DockerfileBuild

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example")
    monkeypatch.setenv("E2B_API_KEY", "key")
    trial = harbor_e2b_environments(tmp_path)["reef"]
    builds: list[DockerfileBuild] = []

    async def ensure(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        builds.append(build)

    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", ensure)
    alias = asyncio.run(reef_e2b.build_task_template(tmp_path / "task"))
    asyncio.run(trial._create_template())
    # The build before the trials is the trial's own: the same alias, resources and build time limit.
    assert alias == trial._template_name
    assert builds[0] == builds[1]


@pytest.mark.unit
def test_the_template_of_a_separate_verifier_environment_is_built_before_the_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
    from harbor.models.trial.paths import TrialPaths

    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_build import AgentEnvTemplates, DockerfileBuild

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example")
    monkeypatch.setenv("E2B_API_KEY", "key")
    trial = harbor_e2b_environments(tmp_path)["reef"]
    task_path = tmp_path / "task"
    (task_path / "task.toml").write_text(
        E2B_TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    )
    (task_path / "tests" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY test.sh /tests/test.sh\n")
    builds: list[DockerfileBuild] = []

    async def ensure(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        builds.append(build)

    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", ensure)
    assert asyncio.run(reef_e2b.build_task_template(task_path)) == trial._template_name
    # The verifier's template is the one that the trial's separate verifier environment builds after the agent.
    task = Task(task_path)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier is not None
    verifier_environment = reef_e2b.KillOnTimeoutE2BEnvironment(
        environment_dir=verifier.directory,
        environment_name=task.short_name,
        session_id="t__verifier",
        trial_paths=TrialPaths(tmp_path / "verifier"),
        task_env_config=verifier.config,
    )
    assert [build.alias for build in builds] == [trial._template_name, verifier_environment._template_name]
    assert builds[1].context_dir == task_path / "tests"


@pytest.mark.unit
def test_a_task_template_that_exists_is_not_built_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b

    monkeypatch.delenv("REEF_E2B_DOCKERFILE_BUILD", raising=False)
    trial = harbor_e2b_environments(tmp_path)["reef"]
    built: list[str] = []

    async def exists(environment: reef_e2b.KillOnTimeoutE2BEnvironment) -> bool:
        return True

    async def create(environment: reef_e2b.KillOnTimeoutE2BEnvironment) -> None:
        built.append(environment._template_name)

    monkeypatch.setattr(reef_e2b.E2BEnvironment, "_does_template_exist", exists)
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "_create_template", create)
    assert asyncio.run(reef_e2b.build_task_template(tmp_path / "task")) == trial._template_name
    assert built == []


@pytest.mark.unit
def test_only_an_e2b_trial_can_be_told_to_build_no_template(tmp_path: Path) -> None:
    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "task.toml").write_text(E2B_TASK_TOML)
    assert "builds_templates" not in environment_config("e2b", task_path).get("kwargs", {})
    assert environment_config("e2b", task_path, builds_templates=False)["kwargs"]["builds_templates"] is False
    with pytest.raises(ValueError, match="only an e2b trial can refuse to build its templates"):
        environment_config("docker", task_path, builds_templates=False)


class SandboxCreatedError(Exception):
    """A stand-in ``_create_sandbox`` raises it: the template part of ``start`` passed, and the sandbox starts."""


def trial_environments(tmp_path: Path, *, builds_templates: bool) -> list[object]:
    """The environment of a trial of the task under ``tmp_path`` and its separate verifier environment, made by
    Harbor's factory from one config of ``environment_config``, as Harbor's trial makes them."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    task_path = tmp_path / "task"
    task = Task(task_path)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier is not None
    config = EnvironmentConfig.model_validate(environment_config("e2b", task_path, builds_templates=builds_templates))
    return [
        EnvironmentFactory.create_environment_from_config(
            config=config,
            environment_dir=directory,
            environment_name=task.short_name,
            session_id=session,
            trial_paths=TrialPaths(tmp_path / session),
            task_env_config=task_env_config,
        )
        for directory, task_env_config, session in (
            (task.paths.environment_dir, task.config.environment, "t__env"),
            (verifier.directory, verifier.config, "t__verifier"),
        )
    ]


def write_separate_verifier_task(tmp_path: Path) -> None:
    """The task of ``harbor_e2b_environments`` under ``tmp_path``, with a separate verifier environment of its own."""
    harbor_e2b_environments(tmp_path)
    task_path = tmp_path / "task"
    (task_path / "task.toml").write_text(
        E2B_TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    )
    (task_path / "tests" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY test.sh /tests/test.sh\n")


@pytest.mark.unit
def test_a_trial_that_builds_no_template_starts_only_from_the_prepared_task_and_verifier_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b

    monkeypatch.delenv("REEF_E2B_DOCKERFILE_BUILD", raising=False)
    write_separate_verifier_task(tmp_path)
    aliases: set[str] = set()
    built: list[str] = []
    started: list[str] = []

    async def alias_exists(alias: str, **options: object) -> bool:
        return alias in aliases

    async def build(**options: object) -> None:
        built.append(str(options["alias"]))
        aliases.add(str(options["alias"]))

    async def create_sandbox(environment: reef_e2b.KillOnTimeoutE2BEnvironment) -> None:
        started.append(environment._template_name)
        raise SandboxCreatedError

    monkeypatch.setattr(e2b.AsyncTemplate, "alias_exists", staticmethod(alias_exists))
    monkeypatch.setattr(e2b.AsyncTemplate, "build", staticmethod(build))
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "_create_sandbox", create_sandbox)

    # The preparation builds both templates, as before; a trial that builds no template starts from each one.
    prepared = asyncio.run(reef_e2b.build_task_template(tmp_path / "task"))
    environment, verifier_environment = trial_environments(tmp_path, builds_templates=False)
    assert built == [prepared, verifier_environment._template_name]
    assert prepared == environment._template_name != verifier_environment._template_name
    for trial_environment in (environment, verifier_environment):
        with pytest.raises(SandboxCreatedError):
            asyncio.run(trial_environment.start(force_build=False))
    assert started == built

    # A template that went away after the preparation fails the start: nothing builds it and no sandbox starts.
    aliases.remove(verifier_environment._template_name)
    with pytest.raises(reef_e2b.TemplateNotPreparedError, match="this trial builds no template"):
        asyncio.run(trial_environments(tmp_path, builds_templates=False)[1].start(force_build=False))
    assert (len(built), len(started)) == (2, 2)
    # A trial that builds templates, the default, builds the missing one as before.
    with pytest.raises(SandboxCreatedError):
        asyncio.run(trial_environments(tmp_path, builds_templates=True)[1].start(force_build=False))
    assert built[2:] == [verifier_environment._template_name]


@pytest.mark.unit
@pytest.mark.parametrize(("status", "starts"), [("ready", True), ("building", False), ("error", False), (None, False)])
def test_on_agentenv_a_trial_that_builds_no_template_starts_only_from_a_ready_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str | None, starts: bool
) -> None:
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_build import AgentEnvTemplates, DockerfileBuild, TemplateState

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example")
    monkeypatch.setenv("E2B_API_KEY", "key")
    write_separate_verifier_task(tmp_path)
    read: list[str] = []

    async def state(templates: AgentEnvTemplates, alias: str) -> TemplateState | None:
        read.append(alias)
        return None if status is None else TemplateState("template-1", status)

    async def ensure(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        raise AssertionError("a trial that builds no template never runs aenv build")

    async def create_sandbox(environment: reef_e2b.KillOnTimeoutE2BEnvironment) -> None:
        raise SandboxCreatedError

    monkeypatch.setattr(AgentEnvTemplates, "state", state)
    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", ensure)
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "_create_sandbox", create_sandbox)
    for trial_environment in trial_environments(tmp_path, builds_templates=False):
        with pytest.raises(SandboxCreatedError if starts else reef_e2b.TemplateNotPreparedError):
            asyncio.run(trial_environment.start(force_build=False))
    assert len(read) == 2 and all(alias.endswith(reef_e2b.AENV_TEMPLATE_SUFFIX) for alias in read)


#: The Compose file of a task with a second service that the agent reaches by name.
TASK_COMPOSE = (
    "services:\n  main:\n    networks: [task]\n  server:\n    image: registry.example/server:1\n"
    "    networks:\n      task:\n        aliases: [challenge]\nnetworks:\n  task: {}\n"
)


class ComposeGuest:
    """A stand-in for the E2B SDK's sandbox of a Compose project's guest. It records each command that the
    environment runs in the guest, and answers each with ``results``, in order; then with success."""

    def __init__(self) -> None:
        self.sandbox_id = "sandbox-1"
        self.commands = self
        self.ran: list[dict[str, object]] = []
        self.results: list[SimpleNamespace] = []
        self.killed = False

    async def run(self, **options: object) -> ComposeGuest:
        self.ran.append(options)
        return self

    async def wait(self) -> SimpleNamespace:
        return self.results.pop(0) if self.results else SimpleNamespace(stdout="", stderr="", exit_code=0)

    async def kill(self) -> None:
        self.killed = True


def compose_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    builds_templates: bool = True,
    build: str = "aenv",
    compose: str | None = "aenv",
) -> object:
    """Reef's E2B environment of a trial of the task of ``harbor_e2b_environments`` with ``TASK_COMPOSE``, with
    ``REEF_E2B_DOCKERFILE_BUILD`` ``build`` and ``REEF_E2B_COMPOSE`` ``compose`` (unset for None). Harbor's factory
    makes it from ``environment_config``."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.task import Task
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", build)
    if compose is None:
        monkeypatch.delenv("REEF_E2B_COMPOSE", raising=False)
    else:
        monkeypatch.setenv("REEF_E2B_COMPOSE", compose)
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example")
    monkeypatch.setenv("E2B_API_KEY", "key")
    task_path = tmp_path / "task"
    if not task_path.exists():
        harbor_e2b_environments(tmp_path)
    (task_path / "environment" / "docker-compose.yaml").write_text(TASK_COMPOSE)
    task = Task(task_path)
    config = environment_config("e2b", task_path, builds_templates=builds_templates)
    return EnvironmentFactory.create_environment_from_config(
        config=EnvironmentConfig.model_validate(config),
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="t__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task.config.environment.model_copy(deep=True),
    )


def compose_stand_ins(monkeypatch: pytest.MonkeyPatch, *, image_exists: bool) -> tuple[list[object], ComposeGuest]:
    """Stand-ins for the AgentENV deployment of a Compose start. Return the calls, in order, and the guest that the
    E2B SDK connects to. The first ``aenv compose up`` finds no image of ``main`` unless ``image_exists``."""
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_build import AgentEnvTemplates, DockerfileBuild
    from reef.execution.sandbox.agentenv_compose import ComposeSandbox, ImageNotFoundError

    calls: list[object] = []
    guest = ComposeGuest()
    missing = [] if image_exists else [ImageNotFoundError("image tag 't:latest' was not found")]

    async def start(sandbox: ComposeSandbox, templates: AgentEnvTemplates) -> str:
        calls.append(sandbox)
        if missing:
            raise missing.pop()
        return guest.sandbox_id

    async def build_image(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        calls.append(build)

    async def prepare_guest(templates: AgentEnvTemplates, sandbox_id: str) -> None:
        calls.append(("prepare", sandbox_id))

    async def connect(sandbox_id: str, timeout: int) -> ComposeGuest:
        calls.append(("connect", sandbox_id, timeout))
        return guest

    async def delete_sandbox(templates: AgentEnvTemplates, sandbox_id: str) -> None:
        calls.append(("delete", sandbox_id))

    monkeypatch.setattr(ComposeSandbox, "start", start)
    monkeypatch.setattr(reef_e2b, "build_image", build_image)
    monkeypatch.setattr(reef_e2b, "prepare_guest", prepare_guest)
    monkeypatch.setattr(reef_e2b, "delete_sandbox", delete_sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "connect", staticmethod(connect))
    return calls, guest


@pytest.mark.unit
def test_on_agentenv_a_task_with_a_compose_file_runs_its_project_and_the_trial_works_in_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.execution.sandbox.agentenv_build import DockerfileBuild
    from reef.execution.sandbox.agentenv_compose import ComposeSandbox, main_image

    environment = compose_environment(tmp_path, monkeypatch)
    calls, guest = compose_stand_ins(monkeypatch, image_exists=False)
    name, reference = main_image(environment._template_name)
    # AgentENV does not tell whether it has the image, so the template part of a start always builds.
    assert asyncio.run(environment._does_template_exist()) is False
    asyncio.run(environment.start(force_build=False))

    # The start builds the image only after AgentENV refused the project for it, then starts the project again.
    first, build, second, *rest = calls
    assert first == second and isinstance(first, ComposeSandbox)
    assert (first.cpu_count, first.memory_mb, first.startup_seconds) == (2, 4096, 120)
    assert first.lifetime_seconds == environment._sandbox_timeout
    services = first.project["services"]
    assert services["main"]["image"] == reference and services["main"]["networks"] == ["task"]
    assert services["main"]["environment"] == {"TASK_SETTING": "from-the-task"}
    assert services["server"]["networks"] == {"task": {"aliases": ["challenge"]}}
    environment_dir = tmp_path / "task" / "environment"
    assert build == DockerfileBuild(
        context_dir=environment_dir,
        dockerfile=environment_dir / "Dockerfile",
        alias=name,
        cpu_count=None,
        memory_mb=None,
        timeout_seconds=120,
        start_command=True,
        image=True,
    )
    assert build.command()[-1] == "--image"
    assert rest == [("prepare", "sandbox-1"), ("connect", "sandbox-1", environment._sandbox_timeout)]

    # A command of the trial runs in main, in the Dockerfile's directory, with the trial's variables and user.
    guest.ran.clear()
    guest.results.append(SimpleNamespace(stdout="out", stderr="err", exit_code=3))
    result = asyncio.run(environment.exec("make test", env={"MODE": "a b"}, timeout_sec=30, user="solver"))
    assert (result.stdout, result.stderr, result.return_code) == ("out", "err", 3)
    (ran,) = guest.ran
    assert ran["cmd"] == (
        "docker compose --project-name aenv exec -T -w /app -e TASK_SETTING=from-the-task -e 'MODE=a b' -u solver "
        "main bash -c 'make test'"
    )
    # The guest runs it as root, outside the task's directory, which only main has.
    assert (ran["user"], ran["cwd"], ran["envs"], ran["timeout"]) == ("root", None, None, 30)

    asyncio.run(environment.stop(delete=True))
    assert guest.killed


@pytest.mark.unit
def test_a_compose_trial_that_builds_no_template_needs_the_image_of_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_compose import ComposeSandbox

    environment = compose_environment(tmp_path, monkeypatch, builds_templates=False)
    calls, _ = compose_stand_ins(monkeypatch, image_exists=False)
    with pytest.raises(reef_e2b.TemplateNotPreparedError):
        asyncio.run(environment.start(force_build=False))
    assert len(calls) == 1 and isinstance(calls[0], ComposeSandbox)

    # With the image, it starts.
    calls, _ = compose_stand_ins(monkeypatch, image_exists=True)
    asyncio.run(environment.start(force_build=False))
    assert [call for call in calls if not isinstance(call, ComposeSandbox)] == [
        ("prepare", "sandbox-1"),
        ("connect", "sandbox-1", environment._sandbox_timeout),
    ]


@pytest.mark.unit
def test_a_compose_start_that_fails_after_the_sandbox_exists_deletes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv_compose import ComposeStartError

    environment = compose_environment(tmp_path, monkeypatch)
    calls, _ = compose_stand_ins(monkeypatch, image_exists=True)

    async def prepare_guest(templates: object, sandbox_id: str) -> None:
        raise ComposeStartError("could not prepare the guest")

    monkeypatch.setattr(reef_e2b, "prepare_guest", prepare_guest)
    with pytest.raises(ComposeStartError):
        asyncio.run(environment.start(force_build=False))
    assert calls[-1] == ("delete", "sandbox-1") and environment._sandbox is None


@pytest.mark.unit
def test_a_compose_file_changes_nothing_unless_the_deployment_selects_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    # The default is off, also on AgentENV: a Compose start writes image records to the deployment's repository.
    for build in ("aenv", "e2b"):
        for compose in (None, "off"):
            assert compose_environment(tmp_path, monkeypatch, build=build, compose=compose).compose_ops is None
    with pytest.raises(ValueError, match="REEF_E2B_COMPOSE=aenv needs REEF_E2B_DOCKERFILE_BUILD=aenv"):
        compose_environment(tmp_path, monkeypatch, build="e2b")
    with pytest.raises(ValueError, match="REEF_E2B_COMPOSE must be one of off, aenv, not 'docker'"):
        compose_environment(tmp_path, monkeypatch, compose="docker")


@pytest.mark.unit
def test_the_image_of_main_is_built_before_the_trials_of_a_compose_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.execution.sandbox.agentenv_build import DockerfileBuild
    from reef.execution.sandbox.agentenv_compose import main_image
    from reef.execution.sandbox.e2b import build_task_main_image, build_task_template

    environment = compose_environment(tmp_path, monkeypatch)
    calls, _ = compose_stand_ins(monkeypatch, image_exists=False)
    name, reference = main_image(environment._template_name)
    assert asyncio.run(build_task_main_image(tmp_path / "task")) == reference
    # build_task_template builds it too, and returns the alias as for a task with a template.
    assert asyncio.run(build_task_template(tmp_path / "task")) == environment._template_name
    assert [call.alias for call in calls if isinstance(call, DockerfileBuild)] == [name, name]

    # A task whose trials run no Compose project has no such image.
    (tmp_path / "task" / "environment" / "docker-compose.yaml").unlink()
    assert asyncio.run(build_task_main_image(tmp_path / "task")) is None
    assert len(calls) == 2


#: A fake ``docker`` for the guest: ``compose --project-name aenv cp`` copies between the guest and the directory
#: FAKE_MAIN_ROOT, which stands in for the filesystem of ``main``.
FAKE_DOCKER = """#!/bin/sh
[ "$1 $2 $3 $4" = "compose --project-name aenv cp" ] || exit 64
main() { case "$1" in main:*) printf '%s' "$FAKE_MAIN_ROOT${1#main:}" ;; *) printf '%s' "$1" ;; esac; }
source=$(main "$5"); target=$(main "$6")
mkdir -p "$(dirname "$target")" && cp -R "$source" "$target"
"""


@pytest.mark.unit
def test_a_compose_trial_moves_files_through_the_guest_to_and_from_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guest is a directory of this test: its commands run here, with its paths under that directory."""
    import subprocess

    from harbor.environments.base import ExecResult

    environment = compose_environment(tmp_path, monkeypatch)
    guest_root, main_root, bin_dir = tmp_path / "guest", tmp_path / "main", tmp_path / "bin"
    for directory in (guest_root / "tmp", main_root, bin_dir):
        directory.mkdir(parents=True)
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"

    async def host_exec(command: str, timeout_sec: int | None = None) -> ExecResult:
        done = subprocess.run(
            ["sh", "-c", command.replace("/tmp/harbor_", f"{guest_root}/tmp/harbor_")],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": path, "FAKE_MAIN_ROOT": str(main_root)},
        )
        return ExecResult(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode)

    async def host_write(guest_path: str, data: bytes) -> None:
        (guest_root / guest_path.lstrip("/")).write_bytes(data)

    async def host_read(guest_path: str) -> bytes:
        return (guest_root / guest_path.lstrip("/")).read_bytes()

    monkeypatch.setattr(environment, "host_exec", host_exec)
    monkeypatch.setattr(environment, "host_write", host_write)
    monkeypatch.setattr(environment, "host_read", host_read)

    source = tmp_path / "source"
    (source / "empty").mkdir(parents=True)
    (source / "bin").mkdir()
    (source / "bin" / "run.sh").write_text("#!/bin/sh\n")
    (source / "bin" / "run.sh").chmod(0o755)
    asyncio.run(environment.upload_dir(source, "/logs/agent/files"))
    asyncio.run(environment.upload_file(source / "bin" / "run.sh", "/logs/agent/one.sh"))
    uploaded = main_root / "logs" / "agent"
    # An empty directory and a file's mode arrive in main.
    assert (uploaded / "files" / "empty").is_dir() and (uploaded / "one.sh").read_text() == "#!/bin/sh\n"
    assert stat.S_IMODE((uploaded / "files" / "bin" / "run.sh").stat().st_mode) == 0o755

    asyncio.run(environment.download_dir("/logs/agent/files", tmp_path / "back"))
    asyncio.run(environment.download_file("/logs/agent/one.sh", tmp_path / "one.sh"))
    assert (tmp_path / "back" / "empty").is_dir() and (tmp_path / "one.sh").read_text() == "#!/bin/sh\n"
    assert stat.S_IMODE((tmp_path / "back" / "bin" / "run.sh").stat().st_mode) == 0o755
    # Nothing stays in the guest.
    assert list((guest_root / "tmp").iterdir()) == []
