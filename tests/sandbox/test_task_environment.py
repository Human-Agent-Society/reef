"""The Harbor task environments that ``reef.execution.sandbox.task_environment`` owns: their readiness probes, and
the environment config of a trial with Reef's E2B environment class."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from collections.abc import Sequence
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import httpx
import pytest
from tests.support.agentenv import ColdAnswer, FakeAgentEnvApi, serving

from reef.execution.sandbox.agentenv.compose import TransientStartError
from reef.execution.sandbox.agentenv.images import NETWORK_FIELDS, ColdSandboxError
from reef.execution.sandbox.prepared_images import PreparedImage
from reef.execution.sandbox.process import (
    SANDBOX_CAPACITY_ERRORS,
    SANDBOX_REQUEST_ERRORS,
    SandboxCapacityError,
    SandboxRequestError,
)
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
def test_saved_e2b_trial_path_and_executor_imports() -> None:
    from harbor.utils.import_path import import_class

    from reef.execution.sandbox import e2b
    from reef.execution.sandbox.e2b import executor, trial
    from reef.execution.sandbox.task_environment import HARBOR_E2B_ENVIRONMENT

    assert HARBOR_E2B_ENVIRONMENT == "reef.execution.sandbox.e2b:KillOnTimeoutE2BEnvironment"
    assert import_class(HARBOR_E2B_ENVIRONMENT) is trial.KillOnTimeoutE2BEnvironment
    assert (e2b.E2BExecutor, e2b.SANDBOX_MAX_TIMEOUT_S) == (executor.E2BExecutor, 3600)
    assert callable(e2b.build_image) and callable(e2b.prepare_guest)


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
@pytest.mark.parametrize("refusal", ["busy", "rate", "gateway", "client"])
def test_reefs_e2b_sandbox_that_the_service_refuses_for_want_of_capacity_raises_sandbox_capacity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    e2b = pytest.importorskip("e2b")
    environment = harbor_e2b_environments(tmp_path)["reef"]
    errors = {
        # A generic "no nodes available" says nothing of the sandbox's size: it is a refusal for want of capacity.
        "busy": e2b.ServiceBusyException("503: Service temporarily unavailable, please retry. - no nodes available"),
        "rate": e2b.RateLimitException("429: too many requests"),
        # A gateway's 504 answer has an HTML body, which the SDK cannot read as JSON.
        "gateway": json.JSONDecodeError("Expecting value", "<html>504 Gateway Time-out</html>", 0),
        "client": httpx.ReadTimeout("timed out"),
    }
    calls: list[dict[str, object]] = []

    async def create(**options: object) -> object:
        calls.append(options)
        raise errors[refusal]

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    with pytest.raises(SandboxCapacityError) as raised:
        asyncio.run(environment._create_sandbox())
    assert raised.value.__cause__ is errors[refusal] and len(calls) == 1
    # Harbor keeps the class name of the error that ended a trial, and an evaluation reads it.
    assert type(raised.value).__name__ in SANDBOX_CAPACITY_ERRORS
    assert TransientStartError.__name__ in SANDBOX_CAPACITY_ERRORS and issubclass(
        TransientStartError, SandboxCapacityError
    )


@pytest.mark.unit
def test_reefs_e2b_create_skips_retry_for_capacity_errors_and_retries_other_transient_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2b = pytest.importorskip("e2b")
    capacity_environment = harbor_e2b_environments(tmp_path)["reef"]
    capacity_calls: list[dict[str, object]] = []

    async def refuse_capacity(**options: object) -> object:
        capacity_calls.append(options)
        raise e2b.ServiceBusyException("503: Service temporarily unavailable - no nodes available")

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(refuse_capacity))
    with pytest.raises(SandboxCapacityError):
        asyncio.run(capacity_environment._create_sandbox())
    assert len(capacity_calls) == 1

    transient_environment = harbor_e2b_environments(tmp_path / "other")["reef"]
    transient_calls: list[dict[str, object]] = []
    created = object()

    async def fail_once(**options: object) -> object:
        transient_calls.append(options)
        if len(transient_calls) == 1:
            raise RuntimeError("transient connection drop")
        return created

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(fail_once))
    asyncio.run(transient_environment._create_sandbox())
    assert transient_environment._sandbox is created and len(transient_calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "raised", "creates"),
    [
        # A bad request fails at once, with the service's reason, and is not sent again.
        (400, SandboxRequestError, 1),
        # A credential error stays the SDK's, with Harbor's one more try.
        (401, None, 2),
    ],
    ids=["bad-request", "credentials"],
)
def test_reefs_e2b_create_reads_the_status_of_the_services_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, raised: type[Exception] | None, creates: int
) -> None:
    e2b = pytest.importorskip("e2b")
    from e2b.api import handle_api_exception
    from e2b.api.client.types import Response

    environment = harbor_e2b_environments(tmp_path)["reef"]
    # The SDK's own error of the create call's answer.
    body = json.dumps({"code": status, "message": "the template needs 64 vCPUs"}).encode()
    error = handle_api_exception(Response(status_code=HTTPStatus(status), content=body, headers={}, parsed=None))
    calls: list[dict[str, object]] = []

    async def create(**options: object) -> object:
        calls.append(options)
        raise error

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    with pytest.raises(type(error) if raised is None else raised) as created:
        asyncio.run(environment._create_sandbox())
    assert len(calls) == creates
    if raised is None:
        assert created.value is error
    else:
        assert isinstance(error, e2b.SandboxException) and error.status_code == 400
        assert created.value.__cause__ is error and "the template needs 64 vCPUs" in str(created.value)
        assert type(created.value).__name__ in SANDBOX_REQUEST_ERRORS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("session", "refusals", "pauses", "created"),
    [
        # A separate verifier waits for capacity, so the agent's work is kept.
        ("t__verifier__tests", 2, [15, 30], True),
        # Its wait ends before 80% of its start time limit (96 of 120 s), without repeating the wait loop.
        ("t__verifier__tests", 99, [15, 30], False),
        # The agent's environment does not wait: its rollout goes back to the queue (EvaluationRolloutWorker).
        ("t__env", 99, [], False),
    ],
    ids=["verifier-gets-capacity", "verifier-gives-up", "agent-environment"],
)
def test_reefs_e2b_verifier_sandbox_waits_for_capacity_within_its_start_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session: str, refusals: int, pauses: list[int], created: bool
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox.e2b import trial as reef_e2b

    environment = harbor_e2b_environments(tmp_path)["reef"]
    environment.session_id = session
    clock = {"seconds": 1000.0}
    environment.started_at = clock["seconds"]
    slept: list[float] = []
    calls: list[int] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["seconds"] += seconds

    async def create(**options: object) -> object:
        calls.append(1)
        if len(calls) <= refusals:
            raise e2b.ServiceBusyException("503: Service temporarily unavailable, please retry. - no nodes available")
        return "sandbox"

    monkeypatch.setattr(reef_e2b, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(reef_e2b, "time", SimpleNamespace(monotonic=lambda: clock["seconds"]))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    if created:
        asyncio.run(environment._create_sandbox())
        assert environment._sandbox == "sandbox"
    else:
        with pytest.raises(SandboxCapacityError):
            asyncio.run(environment._create_sandbox())
    assert slept == pauses


def harbor_modal_environment(tmp_path: Path) -> object:
    """Reef's Modal environment, made by Harbor's factory as a trial makes it, from the config of
    ``environment_config`` for one task with a Dockerfile."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.task.task import Task
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    task_path = tmp_path / "task"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_path / "tests").mkdir()
    (task_path / "tests" / "test.sh").write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
    (task_path / "instruction.md").write_text("Write nothing.\n")
    (task_path / "task.toml").write_text(E2B_TASK_TOML)
    task = Task(task_path)
    return EnvironmentFactory.create_environment_from_config(
        config=EnvironmentConfig.model_validate(environment_config("modal", task_path)),
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="t__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task.config.environment.model_copy(deep=True),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
    )


@pytest.mark.unit
def test_a_modal_trial_gets_reefs_class_on_harbors_modal_environment(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    from harbor.environments.modal import ModalEnvironment
    from harbor.utils.import_path import import_class

    from reef.execution.sandbox.modal.trial import ModalTrialEnvironment
    from reef.execution.sandbox.task_environment import HARBOR_MODAL_ENVIRONMENT

    environment = harbor_modal_environment(tmp_path)
    assert type(environment) is ModalTrialEnvironment and isinstance(environment, ModalEnvironment)
    assert environment.type() == ModalEnvironment.type() == "modal"
    # A trial config names the class by its module, and Harbor resolves that path.
    assert HARBOR_MODAL_ENVIRONMENT == "reef.execution.sandbox.modal.trial:ModalTrialEnvironment"
    assert import_class(HARBOR_MODAL_ENVIRONMENT) is ModalTrialEnvironment


def modal_errors() -> dict[str, Exception]:
    """One Modal error of each kind that the tests of Reef's Modal class raise, by name."""
    modal = pytest.importorskip("modal")
    return {
        "no-resources": modal.exception.ResourceExhaustedError("no machine has 64 CPUs free"),
        "invalid-request": modal.exception.InvalidError("a sandbox has at most 64 CPUs"),
        "credentials": modal.exception.AuthError("the token was refused"),
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("refusal", "raised", "names"),
    [
        ("no-resources", SandboxCapacityError, SANDBOX_CAPACITY_ERRORS),
        ("invalid-request", SandboxRequestError, SANDBOX_REQUEST_ERRORS),
        # Any other error of the creation stays Modal's, and an evaluation counts it as an ordinary failed attempt.
        ("credentials", None, frozenset()),
    ],
    ids=["no-resources", "invalid-request", "credentials"],
)
def test_reefs_modal_sandbox_creation_raises_reefs_error_for_modals_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refusal: str,
    raised: type[Exception] | None,
    names: frozenset[str],
) -> None:
    modal = pytest.importorskip("modal")

    error = modal_errors()[refusal]
    created = object()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def create_sandbox(*entrypoint: str, **options: object) -> object:
        calls.append((entrypoint, options))
        if len(calls) > 1:
            raise error
        return created

    monkeypatch.setattr(modal.Sandbox, "create", SimpleNamespace(aio=create_sandbox))
    environment = harbor_modal_environment(tmp_path)
    creation = environment._create_sandbox(
        entrypoint=["sleep", "60"], block_network=True, experimental_options={"probe": 1}
    )
    assert asyncio.run(creation) is created
    entrypoint, options = calls[0]
    assert entrypoint == ("sleep", "60")
    assert options["block_network"] is True and options["experimental_options"] == {"probe": 1}
    with pytest.raises(type(error) if raised is None else raised) as refused:
        asyncio.run(environment._create_sandbox())
    assert len(calls) == 3  # Harbor retries the failed create once.
    if raised is None:
        assert refused.value is error
        assert type(error).__name__ not in SANDBOX_CAPACITY_ERRORS | SANDBOX_REQUEST_ERRORS
    else:
        # Harbor keeps the class name of the error that ended a trial, and an evaluation reads it; the message keeps
        # Modal's reason.
        assert type(refused.value).__name__ in names
        assert refused.value.__cause__ is error and str(error) in str(refused.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("step", "refusal", "raised"),
    [
        # Modal refuses the sandbox itself.
        ("creation", "no-resources", SandboxCapacityError),
        ("creation", "invalid-request", SandboxRequestError),
        # A step after the creation, such as the image's stream or the setup, fails with Modal's error: it stays
        # Modal's, so a rollout whose sandbox exists does not fail for good or wait for capacity.
        ("after-creation", "invalid-request", None),
        ("after-creation", "no-resources", None),
    ],
    ids=["refused-creation", "rejected-creation", "invalid-after-creation", "exhausted-after-creation"],
)
def test_reefs_modal_start_names_only_the_refusals_of_the_sandbox_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str, refusal: str, raised: type[Exception] | None
) -> None:
    pytest.importorskip("modal")
    from harbor.environments.modal import ModalEnvironment

    error = modal_errors()[refusal]

    async def create_sandbox(environment: ModalEnvironment, **options: object) -> object:
        if step == "creation":
            raise error
        return object()

    environment = harbor_modal_environment(tmp_path)

    async def start_strategy(*arguments: object, **options: object) -> None:
        # Harbor's start strategy creates the sandbox, then runs its later steps.
        await environment._create_sandbox()
        raise error

    monkeypatch.setattr(ModalEnvironment, "_create_sandbox", create_sandbox)
    # Harbor's own start runs, and calls its strategy.
    monkeypatch.setattr(environment._strategy, "start", start_strategy)
    with pytest.raises(type(error) if raised is None else raised) as started:
        asyncio.run(environment.start(force_build=False))
    if raised is None:
        assert started.value is error
    else:
        assert started.value.__cause__ is error


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
def test_a_cancelled_template_build_ends_and_is_not_built_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that stops, such as a process at its shutdown, cancels the build. The build's retry must not take the
    cancellation as a failed attempt: the build would run again, and its caller would never end."""
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setenv("E2B_API_URL", "https://api.agentenv.example")
    monkeypatch.setenv("E2B_API_KEY", "key")
    reef = harbor_e2b_environments(tmp_path)["reef"]
    builds: list[DockerfileBuild] = []

    async def ensure(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
        builds.append(build)
        await asyncio.sleep(3600)

    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", ensure)

    async def cancelled() -> bool:
        task = asyncio.create_task(reef._create_template())
        while not builds:
            await asyncio.sleep(0)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=5)
        return bool(done) and task.cancelled()

    assert asyncio.run(cancelled()) and len(builds) == 1


@pytest.mark.unit
def test_with_aenv_builds_reefs_class_builds_the_tasks_dockerfile_with_aenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild

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
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild

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
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild

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
    from reef.execution.sandbox.e2b import trial as reef_e2b

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


def trial_environments(
    tmp_path: Path, *, builds_templates: bool, prepared_images: Sequence[PreparedImage] = ()
) -> list[object]:
    """The environment of a trial of the task under ``tmp_path`` and its separate verifier environment, made by
    Harbor's factory from one config of ``environment_config`` with ``prepared_images``, as Harbor's trial makes
    them."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    task_path = tmp_path / "task"
    task = Task(task_path)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier is not None
    config = EnvironmentConfig.model_validate(
        environment_config("e2b", task_path, builds_templates=builds_templates, prepared_images=prepared_images)
    )
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
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild, TemplateState

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
    prepared_images: Sequence[PreparedImage] = (),
) -> object:
    """Reef's E2B environment of a trial of the task of ``harbor_e2b_environments`` with ``TASK_COMPOSE``, with
    ``REEF_E2B_DOCKERFILE_BUILD`` ``build`` and ``REEF_E2B_COMPOSE`` ``compose`` (unset for None). Harbor's factory
    makes it from ``environment_config`` with ``prepared_images``."""
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
    config = environment_config("e2b", task_path, builds_templates=builds_templates, prepared_images=prepared_images)
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
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates, DockerfileBuild
    from reef.execution.sandbox.agentenv.compose import ComposeSandbox, ImageNotFoundError

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
    from reef.execution.sandbox.agentenv.build import DockerfileBuild
    from reef.execution.sandbox.agentenv.compose import ComposeSandbox, main_image

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
    from reef.execution.sandbox.agentenv.compose import ComposeSandbox

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
    from reef.execution.sandbox.agentenv.compose import ComposeStartError

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
    from reef.execution.sandbox.agentenv.build import DockerfileBuild
    from reef.execution.sandbox.agentenv.compose import main_image
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


@pytest.mark.unit
def test_a_compose_command_as_root_names_root_by_its_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent can overwrite /etc/passwd in main. Docker then finds no user named root, but it still runs UID 0."""
    import shlex

    from harbor.environments.base import ExecResult

    environment = compose_environment(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    async def host_exec(command: str, timeout_sec: int | None = None) -> ExecResult:
        commands.append(shlex.split(command))
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(environment, "host_exec", host_exec)
    for user in ("root", "agent", 1000):
        asyncio.run(environment.exec("true", user=user))
    users = [command[command.index("-u") + 1] for command in commands]
    assert users == ["0", "agent", "1000"]


#: Synthetic images of the task of ``harbor_e2b_environments`` and of its separate verifier, by digest.
TASK_IMAGE = "registry.example/tasks/task@sha256:" + "c" * 64
VERIFIER_IMAGE = "registry.example/tasks/task-tests@sha256:" + "d" * 64


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


def prepared_environment(tmp_path: Path, images: Sequence[PreparedImage], *, network: str = "allowlist") -> object:
    """Reef's E2B environment of a trial of the task of ``harbor_e2b_environments``, made by Harbor's factory from
    the config of ``environment_config`` with ``images``. Its network is an allowlist of pypi.org, or no network for
    ``network`` ``none``."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.task.task import Task
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    harbor_e2b_environments(tmp_path)
    task_path = tmp_path / "task"
    task = Task(task_path)
    policy = (
        NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"])
        if network == "allowlist"
        else NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    config = {
        **environment_config("e2b", task_path, prepared_images=images),
        "env": {"TRIAL_SETTING": "from-the-trial"},
    }
    return EnvironmentFactory.create_environment_from_config(
        config=EnvironmentConfig.model_validate(config),
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="t__env",
        trial_paths=TrialPaths(tmp_path / "prepared"),
        task_env_config=task.config.environment.model_copy(deep=True),
        network_policy=policy,
    )


class ColdGuest:
    """A stand-in for the E2B SDK's sandbox ``sandbox-1`` of a cold start, which is also the handle of its start
    command.

    Each command that runs in it is added to ``steps`` with its options. A command in the background is the start
    command: it runs until the test ends, or ends at once with ``start_code`` when that is not None. Any other
    command is the ready command. It exits with each code of ``ready_codes`` in order, then with 0, and adds
    ``ready_seconds`` to ``clock``; with ``ready_hangs``, it sets ``ready_started`` and never ends. ``disconnected``
    counts the disconnects from the start command.
    """

    def __init__(self, steps: list[tuple[object, ...]], ready_codes: list[int]) -> None:
        self.sandbox_id = "sandbox-1"
        self.commands = self
        self.steps = steps
        self.ready_codes = ready_codes
        self.start_code: int | None = None
        self.clock = 1000.0
        self.ready_seconds = 0.0
        self.ready_hangs = False
        self.ready_started = asyncio.Event()
        self.disconnected = 0

    async def run(self, **options: object) -> object:
        from e2b.sandbox.commands.command_handle import CommandExitException

        if options.get("background"):
            self.steps.append(("start", options))
            return self
        self.steps.append(("ready", options))
        self.ready_started.set()
        if self.ready_hangs:
            await asyncio.Event().wait()
        self.clock += self.ready_seconds
        code = self.ready_codes.pop(0) if self.ready_codes else 0
        if code:
            raise CommandExitException(stderr="not ready", stdout="", exit_code=code, error=None)
        return SimpleNamespace(stdout="", stderr="", exit_code=0, error=None)

    async def wait(self) -> object:
        from e2b.sandbox.commands.command_handle import CommandExitException

        if self.start_code is None:
            await asyncio.Event().wait()
        if self.start_code:
            raise CommandExitException(stderr="no config", stdout="", exit_code=self.start_code, error=None)
        return SimpleNamespace(stdout="", stderr="", exit_code=0, error=None)

    async def disconnect(self) -> None:
        self.disconnected += 1


def created_sandbox(environment: object, api: FakeAgentEnvApi, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``environment._create_sandbox()``, the method that Harbor's ``start`` calls, against ``api``, which
    ``E2B_API_URL`` names on a free local port."""

    async def run() -> None:
        async with serving(api) as templates:
            monkeypatch.setenv("E2B_API_URL", templates.api_url)
            monkeypatch.setenv("E2B_API_KEY", templates.api_key)
            await environment._create_sandbox()

    asyncio.run(run())


def cold_stand_ins(
    monkeypatch: pytest.MonkeyPatch, ready_codes: list[int]
) -> tuple[list[tuple[object, ...]], ColdGuest]:
    """Stand-ins for a cold start's connect, which gives a ``ColdGuest`` with ``ready_codes``, and for every build,
    which fails the test. Return the steps of the start, in order, and the guest. The ready command runs again at
    once after a failure."""
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.e2b import trial

    steps: list[tuple[object, ...]] = []
    guest = ColdGuest(steps, ready_codes)

    async def connect(sandbox_id: str, timeout: int) -> ColdGuest:
        steps.append(("connect", sandbox_id, timeout))
        return guest

    async def no_build(*arguments: object, **options: object) -> None:
        raise AssertionError("a prepared image is never built")

    monkeypatch.setattr(e2b.AsyncSandbox, "connect", staticmethod(connect))
    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", no_build)
    monkeypatch.setattr(e2b.AsyncTemplate, "build", staticmethod(no_build))
    monkeypatch.setattr(trial, "READY_POLL_SECONDS", 0.0)
    return steps, guest


def sdk_error(status: int, message: str) -> Exception:
    """The E2B SDK's own error for an answer of the create call with ``status`` and ``message``."""
    pytest.importorskip("e2b")
    from e2b.api import handle_api_exception
    from e2b.api.client.types import Response

    body = json.dumps({"code": status, "message": message}).encode()
    return handle_api_exception(Response(status_code=HTTPStatus(status), content=body, headers={}, parsed=None))


@pytest.mark.unit
def test_only_an_e2b_config_carries_prepared_images(tmp_path: Path) -> None:
    pytest.importorskip("e2b")
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    from reef.execution.sandbox.e2b import KillOnTimeoutE2BEnvironment, TemplateNotPreparedError

    harbor_e2b_environments(tmp_path)
    task_path = tmp_path / "task"
    image = prepared_image()
    kwargs = environment_config("e2b", task_path, prepared_images=[image])["kwargs"]
    assert (kwargs["prepared_images"], kwargs["task_directory"]) == ([image.to_json()], str(task_path))
    assert "prepared_images" not in environment_config("e2b", task_path).get("kwargs", {})
    for other in ("docker", "modal"):
        with pytest.raises(ValueError, match="only an e2b trial starts from prepared images"):
            environment_config(other, task_path, prepared_images=[image])
    # A forged list, or a list without its task, fails the environment before it starts.
    task = Task(task_path)
    options = {
        "environment_dir": task.paths.environment_dir,
        "environment_name": task.short_name,
        "session_id": "t__env",
        "trial_paths": TrialPaths(tmp_path / "forged"),
        "task_env_config": task.config.environment,
    }
    for forged_image in ("registry.example/tasks/task:latest", "aenv/task@sha256:" + "c" * 64):
        forged = {**image.to_json(), "image": forged_image}
        with pytest.raises(ValueError, match="host/repository@sha256"):
            KillOnTimeoutE2BEnvironment(**options, prepared_images=[forged], task_directory=str(task_path))
    with pytest.raises(ValueError, match="given together"):
        KillOnTimeoutE2BEnvironment(**options, prepared_images=[image.to_json()])
    # A list without the image of the task's environment fails it, and the environment builds nothing instead.
    other = prepared_image("tests", VERIFIER_IMAGE).to_json()
    with pytest.raises(TemplateNotPreparedError, match=r"have no image of .*environment \(source image None\)"):
        KillOnTimeoutE2BEnvironment(**options, prepared_images=[other], task_directory=str(task_path))


@pytest.mark.unit
@pytest.mark.parametrize("force_build", [False, True])
def test_each_environment_of_a_trial_starts_its_own_prepared_image_and_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_build: bool
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox import e2b as reef_e2b
    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    write_separate_verifier_task(tmp_path)
    started: list[object] = []

    async def never(*arguments: object, **options: object) -> None:
        raise AssertionError("a prepared image builds no template and reads no alias")

    async def create_sandbox(environment: reef_e2b.KillOnTimeoutE2BEnvironment) -> None:
        started.append(environment.prepared_image)
        raise SandboxCreatedError

    monkeypatch.setattr(e2b.AsyncTemplate, "alias_exists", staticmethod(never))
    monkeypatch.setattr(e2b.AsyncTemplate, "build", staticmethod(never))
    monkeypatch.setattr(AgentEnvTemplates, "state", never)
    monkeypatch.setattr(reef_e2b, "ensure_aenv_template", never)
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "_create_sandbox", create_sandbox)
    images = (prepared_image(), prepared_image("tests", VERIFIER_IMAGE, template_id=None))
    # Also a trial that builds no template, and a forced build: the deployment prepared the images.
    for builds_templates in (True, False):
        for environment in trial_environments(tmp_path, builds_templates=builds_templates, prepared_images=images):
            with pytest.raises(SandboxCreatedError):
                asyncio.run(environment.start(force_build=force_build))
    assert started == [*images, *images]
    # A list without an image of the task's environment, or of its verifier, or with another source image, fails the
    # trial before its agent runs: the environment does not go back to its template or its Dockerfile.
    pinned = prepared_image(source_image="registry.example/tasks/task:1")
    for partial in (images[1:], images[:1], [pinned, images[1]]):
        with pytest.raises(reef_e2b.TemplateNotPreparedError, match="a task with prepared images builds none"):
            trial_environments(tmp_path, builds_templates=True, prepared_images=partial)


@pytest.mark.unit
def test_a_prepared_image_starts_from_its_ready_template_as_a_template_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2b = pytest.importorskip("e2b")
    environment = prepared_environment(tmp_path, [prepared_image(start="/usr/local/bin/serve", ready="true")])
    plain = harbor_e2b_environments(tmp_path / "plain")["reef"]
    steps, _ = cold_stand_ins(monkeypatch, [])
    created: list[dict[str, object]] = []

    async def create(**options: object) -> object:
        created.append(options)
        return "sandbox"

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    api = FakeAgentEnvApi(statuses={"template-1": "ready"})
    created_sandbox(environment, api, monkeypatch)
    asyncio.run(plain._create_sandbox())
    # The create call of the trial's own template, from the verified template and not from the task's alias. Its
    # snapshot already ran the start command, so nothing runs again, and no cold start follows.
    assert environment._sandbox == "sandbox" and created[0] == {**created[1], "template": "template-1"}
    assert created[1]["template"] == plain._template_name
    assert (steps, api.cold_bodies) == ([], [])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("template_id", "status", "answer"),
    [
        (None, None, None),
        ("template-1", None, None),
        ("template-1", "building", None),
        ("template-1", "error", None),
        # The read of the template's state allocates nothing, so a read without an answer starts the image cold.
        ("template-1", "slow", None),
        ("template-1", 429, None),
        ("template-1", 500, None),
        ("template-1", 503, None),
        ("template-1", "invalid-json", None),
        ("template-1", "missing-status", None),
        # AgentENV's answers that the worker has no snapshot of the template, which it gives before it allocates.
        ("template-1", "ready", (400, "template template-1 not found")),
        ("template-1", "ready", (400, "worker error: template template-1 not found")),
        ("template-1", "ready", (500, "worker error: snapshot 'template-1' is not ready")),
    ],
    ids=[
        "no-template",
        "missing",
        "building",
        "failed",
        "stalled",
        "status-rate-limited",
        "status-error",
        "cache-unavailable",
        "unreadable-status",
        "missing-status-field",
        "not-found",
        "worker-not-found",
        "not-ready",
    ],
)
def test_a_prepared_image_starts_cold_when_its_template_cannot_start_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template_id: str | None,
    status: str | int | None,
    answer: tuple[int, str] | None,
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox.e2b import trial

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setattr(trial, "TEMPLATE_STATUS_WAIT_SECONDS", 0.2)
    image = prepared_image(template_id=template_id, start="/usr/local/bin/serve --port 8080", ready="test -f /ready")
    environment = prepared_environment(tmp_path, [image])
    # The ready command fails once, as a server that is still starting, and then passes.
    steps, guest = cold_stand_ins(monkeypatch, [1])

    async def create(**options: object) -> object:
        steps.append(("create", options["template"]))
        assert answer is not None, "a template that is not ready is not started"
        raise sdk_error(*answer)

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    api = FakeAgentEnvApi(statuses={} if status is None else {"template-1": status})
    created_sandbox(environment, api, monkeypatch)

    # The cold start has the values of the template's create call, and the task's resources of a template build.
    assert api.cold_bodies == [
        {
            "image": TASK_IMAGE,
            "timeout": environment._sandbox_timeout,
            "autoPause": False,
            "secure": True,
            "allowInternetAccess": True,
            "network": {"allowOut": ["pypi.org"], "denyOut": [e2b.ALL_TRAFFIC]},
            "metadata": {"environment_name": "task", "session_id": "t__env"},
            "envVars": {"TASK_SETTING": "from-the-task", "TRIAL_SETTING": "from-the-trial"},
            "cpuCount": 2,
            "memoryMB": 4096,
        }
    ]
    # Then the image's start runs as AgentENV's template runner ran it: the start command keeps running, and the
    # start ends when the ready command passes. Neither command gets a user, a directory or a variable of the trial:
    # they run as the image's own user, in its directory, with its variables.
    warm = [("create", "template-1")] if answer is not None else []
    start = {"cmd": "/usr/local/bin/serve --port 8080", "background": True, "timeout": 0}
    assert steps[: len(warm) + 2] == [*warm, ("connect", "sandbox-1", environment._sandbox_timeout), ("start", start)]
    ready_runs = steps[len(warm) + 2 :]
    assert [options["cmd"] for _, options in ready_runs] == ["test -f /ready"] * 2
    for _, options in ready_runs:
        # Each run ends within the task's start time limit of 120 s.
        assert set(options) == {"cmd", "timeout"} and 1 <= options["timeout"] <= 120
    assert guest.disconnected == 1 and environment._sandbox is guest and api.deleted == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "raised", "creates"),
    [
        # No worker has room: the rollout waits for capacity, and the image does not start cold elsewhere.
        ("busy", SandboxCapacityError, 1),
        # An answer that did not come can still start the template's sandbox, so no second sandbox starts.
        ("timeout", SandboxCapacityError, 1),
        ("gateway", SandboxCapacityError, 1),
        ("bad-request", SandboxRequestError, 1),
        # The answer of another template, or an answer that AgentENV does not give for a missing snapshot, is no
        # sign that this template's snapshot is missing.
        ("other-template", SandboxRequestError, 1),
        ("not-found-404", None, 2),
        ("other-500", None, 2),
        # A credential error keeps Harbor's one more try.
        ("unauthorized", None, 2),
        ("status-unauthorized", aiohttp.ClientResponseError, 0),
        # A redirect of the read is not followed with the API key.
        ("status-redirect", aiohttp.ClientResponseError, 0),
    ],
)
def test_a_template_start_that_fails_otherwise_never_starts_the_image_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, raised: type[Exception] | None, creates: int
) -> None:
    e2b = pytest.importorskip("e2b")
    environment = prepared_environment(tmp_path, [prepared_image(start="/usr/local/bin/serve")])
    steps, _ = cold_stand_ins(monkeypatch, [])
    errors = {
        "busy": e2b.ServiceBusyException("503: Service temporarily unavailable, please retry. - no nodes available"),
        "timeout": httpx.ReadTimeout("timed out"),
        "gateway": json.JSONDecodeError("Expecting value", "<html>504 Gateway Time-out</html>", 0),
        "bad-request": sdk_error(400, "the template needs 64 vCPUs"),
        "other-template": sdk_error(400, "template template-2 not found"),
        "not-found-404": sdk_error(404, "template template-1 not found"),
        "other-500": sdk_error(500, "worker error: disk full"),
        "unauthorized": sdk_error(401, "invalid API key"),
    }
    calls: list[dict[str, object]] = []

    async def create(**options: object) -> object:
        calls.append(options)
        raise errors[failure]

    monkeypatch.setattr(e2b.AsyncSandbox, "create", staticmethod(create))
    status = {"status-unauthorized": 401, "status-redirect": "redirect"}.get(failure, "ready")
    api = FakeAgentEnvApi(statuses={"template-1": status})
    with pytest.raises(type(errors[failure]) if raised is None else raised):
        created_sandbox(environment, api, monkeypatch)
    assert len(calls) == creates
    assert (steps, api.cold_bodies, api.deleted, api.followed) == ([], [], [], [])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("answer", "raised"),
    [
        ((503, {"message": "no nodes available"}), SandboxCapacityError),
        ((504, "<html>Gateway Time-out</html>"), ColdSandboxError),
        ((401, {"message": "invalid API key"}), ColdSandboxError),
        ((500, {"message": "image pull failed"}), ColdSandboxError),
        ("drop", ColdSandboxError),
    ],
    ids=["busy", "gateway", "unauthorized", "service", "dropped"],
)
def test_a_cold_start_that_fails_is_sent_once_and_keeps_its_meaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: ColdAnswer, raised: type[Exception]
) -> None:
    from reef.execution.sandbox.process import SANDBOX_CAPACITY_ERRORS

    environment = prepared_environment(tmp_path, [prepared_image(template_id=None, start="/usr/local/bin/serve")])
    if raised is not SandboxCapacityError:
        # A separate verifier waits for capacity, but it does not ask again after another failure either.
        environment.session_id = "t__verifier__tests"
    steps, _ = cold_stand_ins(monkeypatch, [])
    api = FakeAgentEnvApi([answer])
    with pytest.raises(raised) as failed:
        created_sandbox(environment, api, monkeypatch)
    # Harbor's retry does not send it again; a full worker waits in the queue as for a template's start.
    assert (len(api.cold_bodies), steps) == (1, [])
    assert (type(failed.value).__name__ in SANDBOX_CAPACITY_ERRORS) == (raised is SandboxCapacityError)


@pytest.mark.unit
def test_a_cold_start_without_network_turns_the_internet_off_when_it_creates_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e2b")
    environment = prepared_environment(tmp_path, [prepared_image(template_id=None)], network="none")
    cold_stand_ins(monkeypatch, [])
    api = FakeAgentEnvApi()
    created_sandbox(environment, api, monkeypatch)
    [body] = api.cold_bodies
    options = environment._sandbox_create_network_options()
    assert body["allowInternetAccess"] is False
    # Each network option of the template's create call, with AgentENV's names.
    network = None if options is None else {NETWORK_FIELDS[name]: value for name, value in options.items()}
    assert body.get("network") == network


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("connect", "did not start: the sandbox's agent did not answer"),
        # A ready command that runs for 50 s and fails: each run gets only what is left of the task's start time
        # limit of 120 s, and no run starts after it.
        ("start-time-limit", "was not ready in 120 s"),
        # With a start time limit of an hour, the wait ends at the template runner's 10 minutes.
        ("ready-limit", "was not ready in 600 s"),
        # The start command fails before the ready command passes, as the template runner refuses such a start.
        ("start-failed", "ended before the start was ready"),
        ("start-failed-with-ready", "ended before the start was ready"),
        ("cancelled", None),
    ],
)
def test_a_cold_start_that_fails_after_its_sandbox_exists_deletes_it_and_starts_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str | None
) -> None:
    e2b = pytest.importorskip("e2b")
    from reef.execution.sandbox.e2b import trial

    image = prepared_image(template_id=None, start="/usr/local/bin/serve", ready="test -f /ready")
    environment = prepared_environment(tmp_path, [image])
    # A separate verifier waits for capacity, but no other failure makes it start a second sandbox.
    environment.session_id = "t__verifier__tests"
    steps, guest = cold_stand_ins(monkeypatch, [] if failure == "start-failed-with-ready" else [1] * 20)
    timeouts = {"start-time-limit": ([120, 70, 20], 50.0), "ready-limit": ([600, 350, 100], 250.0)}
    if failure in timeouts:
        guest.ready_seconds = timeouts[failure][1]
        monkeypatch.setattr(trial, "time", SimpleNamespace(monotonic=lambda: guest.clock))
        environment.started_at = guest.clock
        if failure == "ready-limit":
            environment.task_env_config.build_timeout_sec = 3600
    guest.start_code = 3 if failure.startswith("start-failed") else None
    guest.ready_hangs = failure == "cancelled"

    async def connect(sandbox_id: str, timeout: int) -> object:
        raise RuntimeError("the sandbox's agent did not answer")

    if failure == "connect":
        monkeypatch.setattr(e2b.AsyncSandbox, "connect", staticmethod(connect))
    api = FakeAgentEnvApi()

    async def cancelled() -> bool:
        async with serving(api) as templates:
            monkeypatch.setenv("E2B_API_URL", templates.api_url)
            monkeypatch.setenv("E2B_API_KEY", templates.api_key)
            start = asyncio.ensure_future(environment._create_sandbox())
            await guest.ready_started.wait()
            start.cancel()
            await asyncio.wait([start])
            return start.cancelled()

    if message is None:
        assert asyncio.run(cancelled())
    else:
        with pytest.raises(ColdSandboxError, match=message):
            created_sandbox(environment, api, monkeypatch)
    # Harbor's stop deletes only a sandbox that the environment holds, so the start deletes its own, once.
    assert api.deleted == ["sandbox-1"] and environment._sandbox is None and len(api.cold_bodies) == 1
    if failure in timeouts:
        assert [step[1]["timeout"] for step in steps if step[0] == "ready"] == timeouts[failure][0]
    if failure != "connect":
        assert guest.disconnected == 1


@pytest.mark.unit
@pytest.mark.parametrize("start_code", [None, 0])
def test_a_start_command_without_a_ready_command_is_ready_after_the_template_runners_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start_code: int | None
) -> None:
    import time

    from reef.execution.sandbox.e2b import trial

    environment = prepared_environment(tmp_path, [prepared_image(template_id=None, start="/usr/local/bin/serve")])
    steps, guest = cold_stand_ins(monkeypatch, [])
    # A start command that ends with 0 leaves no process of its own, and the start still counts as ready.
    guest.start_code = start_code
    monkeypatch.setattr(trial, "DEFAULT_READY_SECONDS", 0.2)
    monkeypatch.setattr(trial, "READY_POLL_SECONDS", 0.02)
    began = time.monotonic()
    created_sandbox(environment, FakeAgentEnvApi(), monkeypatch)
    # No ready command runs: the template runner waits a fixed time for such a start.
    assert time.monotonic() - began >= 0.2
    assert [name for name, *_ in steps] == ["connect", "start"] and guest.disconnected == 1


@pytest.mark.unit
@pytest.mark.parametrize("step", ["harbor", "systemd", "cancelled"])
def test_a_start_from_a_prepared_image_whose_later_step_fails_deletes_its_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    pytest.importorskip("e2b")
    from harbor.environments.base import ExecResult
    from harbor.environments.e2b import E2BEnvironment

    from reef.execution.sandbox import e2b as reef_e2b

    environment = prepared_environment(tmp_path, [prepared_image(template_id=None)])
    cold_stand_ins(monkeypatch, [])
    waiting = asyncio.Event()

    async def harbor_start(started: E2BEnvironment, force_build: bool) -> None:
        # Harbor's start creates the sandbox, then makes the trial's directories and uploads its files.
        await started._create_sandbox()
        if step == "harbor":
            raise RuntimeError("could not make the trial's directories")

    async def systemd_wait(started: E2BEnvironment, command: str, **options: object) -> ExecResult:
        waiting.set()
        if step == "cancelled":
            await asyncio.Event().wait()
        raise RuntimeError("the sandbox's agent did not answer")

    monkeypatch.setattr(E2BEnvironment, "start", harbor_start)
    monkeypatch.setattr(reef_e2b.KillOnTimeoutE2BEnvironment, "exec", systemd_wait)
    api = FakeAgentEnvApi()

    async def run() -> asyncio.Future[None]:
        async with serving(api) as templates:
            monkeypatch.setenv("E2B_API_URL", templates.api_url)
            monkeypatch.setenv("E2B_API_KEY", templates.api_key)
            start = asyncio.ensure_future(environment.start(force_build=False))
            if step == "cancelled":
                await waiting.wait()
                start.cancel()
            await asyncio.wait([start])
            return start

    start = asyncio.run(run())
    assert start.cancelled() if step == "cancelled" else isinstance(start.exception(), RuntimeError)
    assert api.deleted == ["sandbox-1"] and environment._sandbox is None


@pytest.mark.unit
def test_a_compose_task_starts_its_project_and_its_plain_verifier_starts_its_prepared_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.execution.sandbox.agentenv.compose import ComposeSandbox, ImageNotFoundError
    from reef.execution.sandbox.e2b import TemplateNotPreparedError

    images = [prepared_image(), prepared_image("tests", VERIFIER_IMAGE)]
    write_separate_verifier_task(tmp_path)
    environment = compose_environment(tmp_path, monkeypatch, prepared_images=images)
    # The project of a Compose task starts as before; its separate verifier has no Compose file.
    assert environment.compose_ops is not None and environment.prepared_image == images[0]
    calls, _ = compose_stand_ins(monkeypatch, image_exists=True)
    asyncio.run(environment.start(force_build=False))
    assert calls[0].project["services"]["main"]["image"] == TASK_IMAGE
    assert [call for call in calls if not isinstance(call, ComposeSandbox)] == [
        ("prepare", "sandbox-1"),
        ("connect", "sandbox-1", environment._sandbox_timeout),
    ]
    verifier = trial_environments(tmp_path, builds_templates=False, prepared_images=images)[1]
    assert verifier.compose_ops is None and verifier.prepared_image == images[1]
    for forced in (False, True):
        missing = compose_environment(tmp_path, monkeypatch, prepared_images=images)
        calls, _ = compose_stand_ins(monkeypatch, image_exists=False)
        with pytest.raises(ImageNotFoundError):
            asyncio.run(missing.start(force_build=forced))
        assert len(calls) == 1 and calls[0].project["services"]["main"]["image"] == TASK_IMAGE
    # Without Compose the main service runs as an ordinary environment and needs its own verified image.
    single = compose_environment(tmp_path, monkeypatch, compose=None, prepared_images=images)
    assert single.compose_ops is None and single.prepared_image == images[0]
    rewritten = [prepared_image(source_image="registry.example/tasks/task@sha256:" + "e" * 64), images[1]]
    with pytest.raises(TemplateNotPreparedError, match="have no image of"):
        compose_environment(tmp_path, monkeypatch, compose=None, prepared_images=rewritten)
