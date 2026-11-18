"""``reef.execution.sandbox.agentenv_build``: an ``aenv build`` of a task's Dockerfile into an AgentENV template, and
the state of a template alias that decides whether to build, wait or delete first."""

from __future__ import annotations

import asyncio
import os
import stat
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.execution.sandbox import agentenv_build
from reef.execution.sandbox.agentenv_build import (
    AgentEnvBuildError,
    AgentEnvTemplates,
    DockerfileBuild,
    dockerfile_build,
    dockerfile_sets_start_command,
    ensure_aenv_template,
)

#: A fake ``aenv``: it records its arguments and the credentials that it reads, then acts as FAKE_AENV_MODE says.
FAKE_AENV = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_AENV_LOG.args"
cp "$XDG_CONFIG_HOME/aenv/credentials" "$FAKE_AENV_LOG.credentials"
case "$FAKE_AENV_MODE" in
  ok) echo "Template ready." ;;
  conflict) echo "Error: HTTP 400: alias 'a' already points to 'x', cannot rebind to 'y'"; exit 1 ;;
  *) echo "#5 ERROR: process apt-get did not complete"; exit 1 ;;
esac
"""


class FakeAgentEnv:
    """A fake AgentENV template API. ``states`` holds the answers to the reads of the alias, in order; the last answer
    repeats. An answer is None (no template) or ``(template_id, status)``. ``deleted`` records each deletion."""

    def __init__(self, states: list[tuple[str, str] | None]) -> None:
        self.states = states
        self.deleted: list[str] = []
        self.current: tuple[str, str] | None = None

    def app(self) -> web.Application:
        async def alias(request: web.Request) -> web.Response:
            assert request.headers["X-API-Key"] == "key"
            self.current = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if self.current is None:
                return web.json_response({"message": "template alias not found"}, status=404)
            return web.json_response({"templateID": self.current[0], "public": False})

        async def status(request: web.Request) -> web.Response:
            assert self.current is not None and request.match_info["build"] == self.current[0]
            reason = {"message": "BuildKit build failed"} if self.current[1] == "error" else None
            body = {"templateID": self.current[0], "buildID": self.current[0], "status": self.current[1]}
            return web.json_response({**body, "reason": reason})

        async def delete(request: web.Request) -> web.Response:
            self.deleted.append(request.match_info["template"])
            return web.Response(status=204)

        app = web.Application()
        app.router.add_get("/templates/aliases/{alias}", alias)
        app.router.add_get("/templates/{template}/builds/{build}/status", status)
        app.router.add_delete("/templates/{template}", delete)
        return app


@pytest.fixture
def fake_aenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fake ``aenv`` first on PATH; it writes ``<log>.args`` and ``<log>.credentials``. Yields the log prefix."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "aenv"
    script.write_text(FAKE_AENV)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_AENV_LOG", str(tmp_path / "aenv"))
    monkeypatch.setattr(agentenv_build, "BUILD_POLL_INTERVAL_S", 0.01)
    yield tmp_path / "aenv"


def ensured(fake: FakeAgentEnv, build: DockerfileBuild) -> None:
    """Run ``ensure_aenv_template`` against ``fake``, served on a free local port."""

    async def run() -> None:
        server = TestServer(fake.app())
        await server.start_server()
        try:
            await ensure_aenv_template(AgentEnvTemplates(str(server.make_url("")).rstrip("/"), "key"), build)
        finally:
            await server.close()

    asyncio.run(run())


def task_build(tmp_path: Path, timeout_seconds: int = 60) -> DockerfileBuild:
    return DockerfileBuild(
        context_dir=tmp_path / "environment",
        dockerfile=tmp_path / "environment" / "Dockerfile",
        alias="task--x__1234__aenv",
        cpu_count=2,
        memory_mb=4096,
        timeout_seconds=timeout_seconds,
        start_command=False,
    )


@pytest.mark.unit
def test_the_variable_selects_the_dockerfile_build() -> None:
    assert dockerfile_build({}) == dockerfile_build({"REEF_E2B_DOCKERFILE_BUILD": " "}) == "e2b"
    assert dockerfile_build({"REEF_E2B_DOCKERFILE_BUILD": "aenv"}) == "aenv"
    with pytest.raises(ValueError, match="must be one of e2b, aenv"):
        dockerfile_build({"REEF_E2B_DOCKERFILE_BUILD": "buildkit"})


@pytest.mark.unit
def test_only_a_command_of_the_last_stage_starts_the_template() -> None:
    pytest.importorskip("dockerfile_parse")
    assert not dockerfile_sets_start_command("FROM ubuntu:24.04\nRUN true\nUSER solver\n")
    assert dockerfile_sets_start_command('FROM ubuntu:24.04\nCMD ["sleep", "infinity"]\n')
    assert not dockerfile_sets_start_command(
        "FROM debian AS build\nCMD make\nFROM ubuntu:24.04\nCOPY --from=build /a /a\n"
    )


@pytest.mark.unit
def test_a_missing_template_is_built_with_aenv_and_the_deployments_credentials(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("dockerfile_parse")
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    fake = FakeAgentEnv([None])
    build = task_build(tmp_path)
    ensured(fake, build)
    assert Path(f"{fake_aenv}.args").read_text().splitlines() == [
        "build",
        str(tmp_path / "environment"),
        "--file",
        str(tmp_path / "environment" / "Dockerfile"),
        "--name",
        "task--x__1234__aenv",
        "--progress",
        "plain",
        "--timeout",
        "60",
        "--cpu",
        "2",
        "--memory",
        "4096",
        "--start-cmd",
        "",
    ]
    credentials = tomllib.loads(Path(f"{fake_aenv}.credentials").read_text())
    assert credentials["api_key"] == "key" and credentials["url"].startswith("http://127.0.0.1:")
    assert fake.deleted == []


@pytest.mark.unit
def test_a_ready_template_is_used_and_a_building_one_is_waited_for(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "fail")
    ensured(FakeAgentEnv([("t1", "ready")]), task_build(tmp_path))
    ensured(FakeAgentEnv([("t1", "waiting"), ("t1", "building"), ("t1", "ready")]), task_build(tmp_path))
    # No build ran: the fake aenv would have failed.
    assert not Path(f"{fake_aenv}.args").exists()
    with pytest.raises(TimeoutError, match="was building after 1 s"):
        ensured(FakeAgentEnv([("t1", "building")]), task_build(tmp_path, timeout_seconds=1))


@pytest.mark.unit
def test_a_failed_template_is_deleted_and_built_again(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    fake = FakeAgentEnv([("t1", "error"), None])
    ensured(fake, task_build(tmp_path))
    assert fake.deleted == ["t1"] and Path(f"{fake_aenv}.args").exists()


@pytest.mark.unit
def test_a_build_that_another_trial_started_first_is_waited_for(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "conflict")
    fake = FakeAgentEnv([None, ("t2", "building"), ("t2", "ready")])
    ensured(fake, task_build(tmp_path))
    assert fake.deleted == []


@pytest.mark.unit
def test_a_failed_build_raises_with_the_end_of_its_output(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "fail")
    with pytest.raises(AgentEnvBuildError, match="exit code 1: #5 ERROR: process apt-get did not complete"):
        ensured(FakeAgentEnv([None]), task_build(tmp_path))
