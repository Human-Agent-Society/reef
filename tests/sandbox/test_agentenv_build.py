"""``reef.execution.sandbox.agentenv_build``: an ``aenv build`` of a task's Dockerfile into an AgentENV template, and
the state of a template alias that decides whether to build, wait or delete first."""

from __future__ import annotations

import asyncio
import os
import stat
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.execution.sandbox import agentenv_build
from reef.execution.sandbox.agentenv_build import (
    AgentEnvBuildError,
    AgentEnvTemplates,
    DockerfileBuild,
    TransientBuildError,
    dockerfile_build,
    dockerfile_sets_start_command,
    ensure_aenv_template,
)

#: A fake ``aenv``: it records its arguments, the credentials that it reads and each run, then acts as FAKE_AENV_MODE
#: says. ``flaky`` fails its first run on an overloaded API and passes the next. A run of ``flaky`` or of a failing
#: build first reports the template FAKE_AENV_CREATED as made, when the variable is set.
FAKE_AENV = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_AENV_LOG.args"
cp "$XDG_CONFIG_HOME/aenv/credentials" "$FAKE_AENV_LOG.credentials"
echo run >> "$FAKE_AENV_LOG.runs"
created() { if [ -n "$FAKE_AENV_CREATED" ]; then echo "Created template $FAKE_AENV_CREATED (build $FAKE_AENV_CREATED)"; fi; }
case "$FAKE_AENV_MODE" in
  ok) echo "Template ready." ;;
  conflict) echo "Error: HTTP 400: alias 'a' already points to 'x', cannot rebind to 'y'"; exit 1 ;;
  flaky)
    if [ "$(wc -l < "$FAKE_AENV_LOG.runs")" -eq 1 ]; then
      created
      printf '%s\\n' "$FAKE_AENV_ERROR"
      exit 1
    fi
    echo "Template ready." ;;
  *)
    created
    echo '#5 ERROR: process apt-get did not complete'
    echo '#5 dependency returned HTTP 503'
    echo 'Error: BuildKit build failed (exit status: 1)'
    exit 1 ;;
esac
"""


class FakeAgentEnv:
    """A fake AgentENV template API. ``states`` holds the answers to the reads of the alias, in order; the last answer
    repeats. An answer is None (no template), ``(template_id, status)``, an HTTP error status, or ``"proxy"`` (the
    HTML page of a proxy's 500). A template
    that ``deleted`` records, each deletion in order, reads as no template. Each build was made ``build_age`` ago."""

    def __init__(self, states: list[tuple[str, str] | int | str | None], build_age: timedelta = timedelta()) -> None:
        self.states = states
        self.build_age = build_age
        self.deleted: list[str] = []
        self.current: tuple[str, str] | None = None

    def app(self) -> web.Application:
        async def alias(request: web.Request) -> web.Response:
            assert request.headers["X-API-Key"] == "key"
            answer = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if answer == "proxy":
                return web.Response(status=500, text="<html>Internal Server Error</html>", content_type="text/html")
            if isinstance(answer, int):
                return web.json_response({"code": answer, "message": "backend error"}, status=answer)
            assert not isinstance(answer, str)
            self.current = None if answer is not None and answer[0] in self.deleted else answer
            if self.current is None:
                return web.json_response({"message": "template alias not found"}, status=404)
            return web.json_response({"templateID": self.current[0], "public": False})

        async def status(request: web.Request) -> web.Response:
            assert self.current is not None and request.match_info["build"] == self.current[0]
            reason = {"message": "BuildKit build failed"} if self.current[1] == "error" else None
            body = {"templateID": self.current[0], "buildID": self.current[0], "status": self.current[1]}
            return web.json_response({**body, "reason": reason})

        async def template(request: web.Request) -> web.Response:
            assert self.current is not None and request.match_info["template"] == self.current[0]
            created_at = (datetime.now(UTC) - self.build_age).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            builds = [{"buildID": self.current[0], "status": self.current[1], "createdAt": created_at}]
            return web.json_response({"templateID": self.current[0], "builds": builds})

        async def delete(request: web.Request) -> web.Response:
            self.deleted.append(request.match_info["template"])
            return web.Response(status=204)

        app = web.Application()
        app.router.add_get("/templates/aliases/{alias}", alias)
        app.router.add_get("/templates/{template}/builds/{build}/status", status)
        app.router.add_get("/templates/{template}", template)
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
    monkeypatch.setattr(agentenv_build, "ORPHAN_WAITING_S", 0.1)
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
        ensured(FakeAgentEnv([None, ("peer", "ready")]), task_build(tmp_path))


@pytest.mark.unit
def test_a_template_left_waiting_is_an_orphan_that_is_deleted_and_built_again(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An aenv build that was killed or refused left the template waiting, with no build behind it.
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    fake = FakeAgentEnv([("t1", "waiting")])
    ensured(fake, task_build(tmp_path))
    assert fake.deleted == ["t1"] and Path(f"{fake_aenv}.args").exists()


@pytest.mark.unit
@pytest.mark.parametrize(("age_seconds", "orphan"), [(4400, False), (4600, True)])
def test_a_template_building_long_past_its_deadline_is_an_orphan_that_is_deleted_and_built_again(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch, age_seconds: int, orphan: bool
) -> None:
    # A worker that AgentENV lost during the build left the template building, with no build behind it. A live
    # worker ends a build of a 1 s deadline within 1 s plus ORPHAN_BUILDING_GRACE_S (4500 s).
    monkeypatch.setenv("FAKE_AENV_MODE", "ok")
    fake = FakeAgentEnv([("t1", "building")], build_age=timedelta(seconds=age_seconds))
    if orphan:
        ensured(fake, task_build(tmp_path, timeout_seconds=1))
        assert fake.deleted == ["t1"] and Path(f"{fake_aenv}.args").exists()
    else:
        with pytest.raises(TimeoutError, match="was building after 1 s"):
            ensured(fake, task_build(tmp_path, timeout_seconds=1))
        assert fake.deleted == [] and not Path(f"{fake_aenv}.args").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("read_status", "build_error"),
    [
        pytest.param(503, "Error: HTTP 429: node concurrent build limit reached", id="worker-busy"),
        pytest.param("proxy", "Error: HTTP 503: try later", id="alias-read-proxy-500"),
        pytest.param(None, "Error: HTTP 500: <html>\n<center>nginx</center>\n</html>", id="proxy-500"),
        pytest.param(
            None,
            "Error: error sending request for url (https://agentenv.example/v3/templates)\n\n"
            "Caused by:\n    0: client error (Connect)\n    1: operation timed out",
            id="connect-timeout",
        ),
    ],
)
def test_a_build_and_a_read_that_an_overloaded_api_refuses_are_tried_again(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch, read_status: int | str | None, build_error: str
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "flaky")
    monkeypatch.setenv("FAKE_AENV_ERROR", build_error)
    ensured(FakeAgentEnv([read_status, None]), task_build(tmp_path))
    assert Path(f"{fake_aenv}.runs").read_text().splitlines() == ["run", "run"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "error", ["Error: HTTP 500: <html>Internal Server Error</html>", "Error: HTTP 503: try later"]
)
def test_a_proxy_failure_waits_for_the_build_already_under_the_alias(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "flaky")
    monkeypatch.setenv("FAKE_AENV_ERROR", error)
    fake = FakeAgentEnv([None, ("t1", "waiting"), ("t1", "building"), ("t1", "ready")])
    ensured(fake, task_build(tmp_path))
    assert fake.deleted == []
    assert Path(f"{fake_aenv}.runs").read_text().splitlines() == ["run"]


@pytest.mark.unit
def test_a_refused_build_deletes_the_waiting_template_that_it_made_and_builds_again_at_once(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AgentENV answers 429 to the request for a builder, after aenv made the template: the template stays waiting.
    monkeypatch.setenv("FAKE_AENV_MODE", "flaky")
    monkeypatch.setenv("FAKE_AENV_CREATED", "t1")
    monkeypatch.setenv("FAKE_AENV_ERROR", "Error: HTTP 429: node concurrent build limit reached; retry later")
    # The second build must not wait until the template counts as an orphan.
    monkeypatch.setattr(agentenv_build, "ORPHAN_WAITING_S", 3600.0)
    fake = FakeAgentEnv([None, ("t1", "waiting")])
    ensured(fake, task_build(tmp_path))
    assert fake.deleted == ["t1"]
    assert Path(f"{fake_aenv}.runs").read_text().splitlines() == ["run", "run"]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["flaky", "fail"])
def test_a_failed_build_does_not_delete_the_waiting_template_of_another_trial(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # The alias names another trial's template by the time this trial's build has failed.
    monkeypatch.setenv("FAKE_AENV_MODE", mode)
    monkeypatch.setenv("FAKE_AENV_CREATED", "t1")
    monkeypatch.setenv("FAKE_AENV_ERROR", "Error: HTTP 503: try later")
    fake = FakeAgentEnv([None, ("peer", "waiting"), ("peer", "building"), ("peer", "ready")])
    if mode == "flaky":
        ensured(fake, task_build(tmp_path))
    else:
        with pytest.raises(AgentEnvBuildError, match="BuildKit build failed"):
            ensured(fake, task_build(tmp_path))
    assert fake.deleted == []
    assert Path(f"{fake_aenv}.runs").read_text().splitlines() == ["run"]


@pytest.mark.unit
def test_a_failed_build_deletes_the_waiting_template_that_it_made(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AENV_MODE", "fail")
    monkeypatch.setenv("FAKE_AENV_CREATED", "t1")
    fake = FakeAgentEnv([None, ("t1", "waiting")])
    with pytest.raises(AgentEnvBuildError, match="BuildKit build failed"):
        ensured(fake, task_build(tmp_path))
    assert fake.deleted == ["t1"]


@pytest.mark.unit
def test_a_500_of_the_agentenv_api_itself_is_not_tried_again(
    tmp_path: Path, fake_aenv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The API's own 500 names a failure that the next request meets again, such as a full catalog disk.
    full = "backend error: sync temp json '/srv/catalog/.tmp7': No space left on device (os error 28)"
    monkeypatch.setenv("FAKE_AENV_MODE", "flaky")
    monkeypatch.setenv("FAKE_AENV_ERROR", f"Error: HTTP 500: {full}")
    with pytest.raises(AgentEnvBuildError, match="No space left on device") as raised:
        ensured(FakeAgentEnv([None]), task_build(tmp_path))
    assert not isinstance(raised.value, TransientBuildError)
    assert Path(f"{fake_aenv}.runs").read_text().splitlines() == ["run"]
    with pytest.raises(aiohttp.ClientResponseError, match="500"):
        ensured(FakeAgentEnv([500]), task_build(tmp_path))
