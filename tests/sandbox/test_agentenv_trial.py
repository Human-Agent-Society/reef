"""Reef's E2B trial class starting its environments from prepared images on AgentENV
(``reef.execution.sandbox.agentenv.images``): a ready template or a cold start, the cold start's request and failures,
the image's start and readiness, and the deletion of a sandbox whose start failed. A local stand-in serves the
AgentENV API, and stand-ins take the place of the E2B SDK's sandbox."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import httpx
import pytest
from tests.support.agentenv import ColdAnswer, FakeAgentEnvApi, serving
from tests.support.e2b_trials import TASK_IMAGE, harbor_e2b_environments, prepared_image

from reef.execution.sandbox.agentenv.images import NETWORK_FIELDS, ColdSandboxError
from reef.execution.sandbox.prepared_images import PreparedImage
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError
from reef.execution.sandbox.task_environment import environment_config


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
    from reef.execution.sandbox.agentenv import images as agentenv_images

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
    monkeypatch.setattr(agentenv_images, "READY_POLL_SECONDS", 0.0)
    return steps, guest


def sdk_error(status: int, message: str) -> Exception:
    """The E2B SDK's own error for an answer of the create call with ``status`` and ``message``."""
    pytest.importorskip("e2b")
    from e2b.api import handle_api_exception
    from e2b.api.client.types import Response

    body = json.dumps({"code": status, "message": message}).encode()
    return handle_api_exception(Response(status_code=HTTPStatus(status), content=body, headers={}, parsed=None))


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
    from reef.execution.sandbox.agentenv import images as agentenv_images

    monkeypatch.setenv("REEF_E2B_DOCKERFILE_BUILD", "aenv")
    monkeypatch.setattr(agentenv_images, "TEMPLATE_STATUS_WAIT_SECONDS", 0.2)
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
    from reef.execution.sandbox.agentenv import images as agentenv_images

    image = prepared_image(template_id=None, start="/usr/local/bin/serve", ready="test -f /ready")
    environment = prepared_environment(tmp_path, [image])
    # A separate verifier waits for capacity, but no other failure makes it start a second sandbox.
    environment.session_id = "t__verifier__tests"
    steps, guest = cold_stand_ins(monkeypatch, [] if failure == "start-failed-with-ready" else [1] * 20)
    timeouts = {"start-time-limit": ([120, 70, 20], 50.0), "ready-limit": ([600, 350, 100], 250.0)}
    if failure in timeouts:
        guest.ready_seconds = timeouts[failure][1]
        monkeypatch.setattr(agentenv_images, "time", SimpleNamespace(monotonic=lambda: guest.clock))
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

    from reef.execution.sandbox.agentenv import images as agentenv_images

    environment = prepared_environment(tmp_path, [prepared_image(template_id=None, start="/usr/local/bin/serve")])
    steps, guest = cold_stand_ins(monkeypatch, [])
    # A start command that ends with 0 leaves no process of its own, and the start still counts as ready.
    guest.start_code = start_code
    monkeypatch.setattr(agentenv_images, "DEFAULT_READY_SECONDS", 0.2)
    monkeypatch.setattr(agentenv_images, "READY_POLL_SECONDS", 0.02)
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
