"""``reef.execution.sandbox.agentenv_compose``: the Compose project that an AgentENV sandbox runs for a task, the
``aenv`` commands that start it, and the trial's operations on its ``main`` service."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tomllib
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.execution.sandbox.agentenv_build import AgentEnvTemplates
from reef.execution.sandbox.agentenv_compose import (
    GUEST_BASH_COMMAND,
    ComposeSandbox,
    ComposeStartError,
    ImageNotFoundError,
    TransientStartError,
    compose_project,
    delete_sandbox,
    main_image,
    prepare_guest,
)

#: The Compose file of a CTF task: the agent's container and the challenge's server on one network.
TASK_COMPOSE = """services:
  main:
    networks:
      agent_network: null
    depends_on:
      target:
        condition: service_started
  target:
    image: registry.example/challenge:1
    environment:
      FLAG: flag{secret}
    networks:
      agent_network:
        aliases:
        - challenge
networks:
  agent_network: null
"""

#: A fake ``aenv``: it records its arguments, its standard input and the credentials that it reads, then acts as
#: FAKE_AENV_MODE says.
FAKE_AENV = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_AENV_LOG.args"
cat > "$FAKE_AENV_LOG.input"
cp "$XDG_CONFIG_HOME/aenv/credentials" "$FAKE_AENV_LOG.credentials"
echo 'Starting Compose sandbox; waiting for services to become ready...' >&2
case "$FAKE_AENV_MODE" in
  ok) echo sandbox-1 ;;
  missing)
    echo "Error: HTTP 400: resolve attached drive 'compose_0' image 'aenv/t:latest': image tag 't:latest' was not found" >&2
    exit 1 ;;
  busy) echo 'Error: HTTP 503: no node can take the sandbox' >&2; exit 1 ;;
  *) echo 'Error: start Compose services: container aenv-target-1 is unhealthy' >&2; exit 1 ;;
esac
"""

TEMPLATES = AgentEnvTemplates("https://api.agentenv.example", "key")


@pytest.fixture
def fake_aenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fake ``aenv`` first on PATH. Yields the prefix of the files that it writes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "aenv"
    script.write_text(FAKE_AENV)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_AENV_LOG", str(tmp_path / "aenv"))
    yield tmp_path / "aenv"


@pytest.mark.unit
def test_the_image_of_main_has_a_valid_name_of_its_own_for_each_alias() -> None:
    name, reference = main_image("alibabactf-2023-pwn-EscapeFromTheEarth__2e5b8d019070__aenv")
    assert name.startswith("alibabactf-2023-pwn-escapefromtheearth-2e5b8d019070-aenv-")
    assert name == name.lower() and reference == f"aenv/{name}:latest"
    # Two aliases that differ only in what an image name cannot hold still have two names.
    assert main_image("Task__1")[0] != main_image("task__1")[0] != main_image("task-1")[0]
    with pytest.raises(ValueError, match="needs a template alias"):
        main_image("")


@pytest.mark.unit
def test_the_project_runs_the_tasks_services_with_the_built_image_as_main() -> None:
    project = compose_project(TASK_COMPOSE, "aenv/task:latest", {"TRIAL_SETTING": "from-the-trial"})
    services = project["services"]
    assert isinstance(services, dict)
    assert services["main"] == {
        "networks": {"agent_network": None},
        "depends_on": {"target": {"condition": "service_started"}},
        "image": "aenv/task:latest",
        "command": ["sh", "-c", "sleep infinity"],
        "environment": {"TRIAL_SETTING": "from-the-trial"},
    }
    # The other services and the networks are the task's.
    assert services["target"]["environment"] == {"FLAG": "flag{secret}"}
    assert services["target"]["networks"] == {"agent_network": {"aliases": ["challenge"]}}
    assert project["networks"] == {"agent_network": None}


@pytest.mark.unit
def test_the_project_keeps_what_the_task_sets_for_main_except_its_build() -> None:
    compose = (
        "services:\n  main:\n    build: {context: .}\n    command: [run]\n"
        "    environment: [KEPT=task, REPLACED=task, FROM_COMPOSE]\n"
    )
    main = compose_project(compose, "aenv/task:latest", {"REPLACED": "trial"})["services"]["main"]  # type: ignore[index]
    assert main == {
        "command": ["run"],
        "image": "aenv/task:latest",
        "environment": {"KEPT": "task", "REPLACED": "trial", "FROM_COMPOSE": None},
    }
    # A project that names only other services gets main; without variables, main has no environment.
    services = compose_project("services:\n  db:\n    image: postgres:17\n", "aenv/task:latest", {})["services"]
    assert services == {  # type: ignore[comparison-overlap]
        "db": {"image": "postgres:17"},
        "main": {"image": "aenv/task:latest", "command": ["sh", "-c", "sleep infinity"]},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose", "message"),
    [
        ("- not a mapping\n", "not a Compose project with a services mapping"),
        ("services: [main]\n", "not a Compose project with a services mapping"),
        ("services:\n  main: {}\n  db:\n    build: ./db\n", "the service db of docker-compose.yaml has a build"),
        ("services:\n  main: image\n", "the service main of docker-compose.yaml is not a mapping"),
        ("services:\n  main:\n    environment: text\n", "the environment of the service main is not a list"),
    ],
)
def test_a_project_that_a_sandbox_cannot_run_is_refused(compose: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compose_project(compose, "aenv/task:latest", {})


@pytest.mark.unit
def test_a_start_gives_aenv_the_project_and_the_deployment_and_returns_the_sandbox(
    fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    project = compose_project(TASK_COMPOSE, "aenv/task:latest", {})
    sandbox = ComposeSandbox(project, cpu_count=2, memory_mb=4096, lifetime_seconds=7200, startup_seconds=600)
    assert asyncio.run(sandbox.start(TEMPLATES)) == "sandbox-1"
    arguments = Path(f"{fake_aenv}.args").read_text().splitlines()
    assert arguments == [
        *("compose", "up", "--file", "-", "--timeout", "7200", "--startup-timeout", "600"),
        *("--cpu", "2", "--memory", "4096"),
    ]
    assert json.loads(Path(f"{fake_aenv}.input").read_text()) == project
    credentials = tomllib.loads(Path(f"{fake_aenv}.credentials").read_text())
    assert credentials == {"url": "https://api.agentenv.example", "api_key": "key"}
    # Without resources, the sandbox has AgentENV's default.
    asyncio.run(ComposeSandbox(project, None, None, lifetime_seconds=60, startup_seconds=60).start(TEMPLATES))
    assert "--cpu" not in Path(f"{fake_aenv}.args").read_text()
    with pytest.raises(ValueError, match="at least 1 s"):
        ComposeSandbox(project, None, None, lifetime_seconds=0, startup_seconds=60)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "error", "text"),
    [
        ("missing", ImageNotFoundError, "image tag 't:latest' was not found"),
        ("busy", TransientStartError, "HTTP 503"),
        ("unhealthy", ComposeStartError, "aenv-target-1 is unhealthy"),
    ],
)
def test_a_failed_start_raises_by_its_cause_with_the_end_of_aenvs_output(
    fake_aenv: Path, monkeypatch: pytest.MonkeyPatch, mode: str, error: type[ComposeStartError], text: str
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", mode)
    sandbox = ComposeSandbox({"services": {}}, None, None, lifetime_seconds=60, startup_seconds=60)
    with pytest.raises(ComposeStartError, match=text) as raised:
        asyncio.run(sandbox.start(TEMPLATES))
    assert type(raised.value) is error


@pytest.mark.unit
def test_the_guest_gets_a_bash_through_aenv_exec(fake_aenv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    asyncio.run(prepare_guest(TEMPLATES, "sandbox-1"))
    arguments = Path(f"{fake_aenv}.args").read_text().split("\n", 5)
    assert arguments[:5] == ["exec", "sandbox-1", "--", "sh", "-c"] and arguments[5].strip() == GUEST_BASH_COMMAND
    monkeypatch.setenv("FAKE_AENV_MODE", "unhealthy")
    with pytest.raises(ComposeStartError, match="could not prepare the guest of the sandbox sandbox-1"):
        asyncio.run(prepare_guest(TEMPLATES, "sandbox-1"))


@pytest.mark.unit
def test_the_guests_bash_runs_sh_with_the_arguments_of_the_sdk(tmp_path: Path) -> None:
    """The command writes a wrapper that passes ``-l -c <command>`` on to ``sh``. Here it writes to a directory of
    the test in place of ``/bin``."""
    import subprocess

    wrapper = tmp_path / "bash"
    subprocess.run(["sh", "-c", GUEST_BASH_COMMAND.replace("/bin/bash", str(wrapper))], check=True)
    result = subprocess.run([str(wrapper), "-c", 'echo "$0 says $1"', "bash", "hello"], capture_output=True, text=True)
    assert result.stdout == "bash says hello\n"
    # A guest that has a bash keeps it.
    wrapper.write_text("#!/bin/sh\necho kept\n")
    subprocess.run(["sh", "-c", GUEST_BASH_COMMAND.replace("/bin/bash", str(wrapper))], check=True)
    assert subprocess.run([str(wrapper)], capture_output=True, text=True).stdout == "kept\n"


@pytest.mark.unit
@pytest.mark.parametrize(("status", "deleted"), [(204, True), (404, True), (500, False)])
def test_a_sandbox_is_deleted_with_the_deployments_key(status: int, deleted: bool) -> None:
    requests: list[tuple[str, str | None]] = []

    async def delete(request: web.Request) -> web.Response:
        requests.append((request.match_info["sandbox"], request.headers.get("X-API-Key")))
        return web.Response(status=status)

    async def run() -> None:
        app = web.Application()
        app.router.add_delete("/sandboxes/{sandbox}", delete)
        server = TestServer(app)
        await server.start_server()
        try:
            await delete_sandbox(AgentEnvTemplates(str(server.make_url("")).rstrip("/"), "key"), "sandbox-1")
        finally:
            await server.close()

    if deleted:
        asyncio.run(run())
    else:
        with pytest.raises(aiohttp.ClientResponseError):
            asyncio.run(run())
    assert requests == [("sandbox-1", "key")]
