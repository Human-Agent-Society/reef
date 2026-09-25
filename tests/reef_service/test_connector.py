"""Connector contracts: bounded access, private credentials and durable command outcomes."""

import asyncio
import json
import os
import signal
import socket
import sys
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.cli import main
from reef.service.connector import Connector, _running, authorize
from reef.service.connector.runtime import HTTPFailure, JSONClient, ReefRuntime, endpoint_url, release_summary
from reef.service.connector.service import ReefService, serve_address
from reef.service.connector.state import ConnectorState


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_connection_state_survives_restart_without_replay(tmp_path):
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    state = ConnectorState(tmp_path)
    state.save({"connector_token": "private-test-token", "instance_id": first})
    assert state.start(first)
    assert state.start(second)
    state.finish(second, {"state": "succeeded", "value": {"scenario": "original"}})
    state.close()
    state = ConnectorState(tmp_path)
    try:
        assert state.load()["instance_id"] == first
        state.recover()
        results = dict(state.pending())
        assert results[first]["state"] == "unknown"
        assert results[second]["state"] == "succeeded"
        assert not state.start(first)
        state.acknowledge(first)
        assert first not in dict(state.pending())
        with state.lock(), pytest.raises(RuntimeError, match="already running"), state.lock():
            pass
        if os.name != "nt":
            assert tmp_path.stat().st_mode & 0o777 == 0o700
            assert state.config_path.stat().st_mode & 0o777 == 0o600
            assert (tmp_path / "commands.sqlite3").stat().st_mode & 0o777 == 0o600
    finally:
        state.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "file:///tmp/a",
        "http://127.0.0.1:bad",
        "https://example.com/#secret",
    ],
)
def test_endpoint_rejects_insecure_or_credential_urls(url):
    with pytest.raises(ValueError):
        endpoint_url(url)


def test_snapshots_and_releases_strip_private_data():
    async def run():
        client = AsyncMock()
        client.base_url = "http://127.0.0.1:8900"
        client.request.side_effect = [
            {"scenarios": [{"scenario": "original", "release_id": "seed", "token": "private"}]},
            {"scenarios": {"original": {"training_mode": "manual", "provider_key": "private"}}, "config": "private"},
        ]
        snapshot = await ReefRuntime(client).snapshot()
        assert snapshot == {
            "reachable": True,
            "scenarios": [{"scenario": "original", "release_id": "seed", "training_mode": "manual"}],
            "reef_url": "http://127.0.0.1:8900",
        }
        client.request.side_effect = HTTPFailure(401, "private")
        failed = await ReefRuntime(client).snapshot()
        assert failed["reachable"] is False and "private" not in json.dumps(failed)

    asyncio.run(run())
    row = release_summary(
        {
            "release_id": "seed",
            "current": True,
            "artifact": "private",
            "metrics": {
                "wins": 3,
                "selected": True,
                "training_request": {"text": "private"},
                "mutation": "private",
                "skipped": "private reason",
            },
        }
    )
    assert row == {"release_id": "seed", "current": True, "metrics": {"wins": 3, "selected": True, "skipped": True}}


@pytest.mark.parametrize(
    ("failure", "error_code"),
    [
        (HTTPFailure(401, "private"), "unauthorized"),
        (HTTPFailure(404, "private"), "not_reef"),
        (HTTPFailure(500, "HTTP 500"), "invalid_response"),
        (TimeoutError(), "timeout"),
        (ValueError("private"), "invalid_response"),
    ],
)
def test_unreachable_snapshot_names_the_setting_to_check(failure, error_code):
    client = AsyncMock()
    client.base_url = "http://127.0.0.1:8901"
    client.request.side_effect = failure
    snapshot = asyncio.run(ReefRuntime(client).snapshot())
    assert snapshot["reachable"] is False and snapshot["error_code"] == error_code
    assert "http://127.0.0.1:8901" in snapshot["error"] and "private" not in json.dumps(snapshot)


def test_snapshot_reports_nothing_listening_at_the_url():
    async def run():
        url = f"http://127.0.0.1:{unused_port()}"
        async with aiohttp.ClientSession() as session:
            return url, await ReefRuntime(JSONClient(session, url)).snapshot()

    url, snapshot = asyncio.run(run())
    assert snapshot["error_code"] == "connection_failed" and snapshot["reef_url"] == url


def test_commands_use_original_scenarios_and_stable_training_receipts():
    async def run():
        client = AsyncMock()
        client.request.return_value = {"secret": "private"}
        runtime = ReefRuntime(client)
        command = {"id": str(uuid.uuid4()), "action": "train", "scenario": "existing a", "text": "improve tests"}
        result = await runtime.execute(command)
        client.request.assert_awaited_once_with(
            "/reef/train",
            scenario="existing a",
            body={"text": "improve tests", "agent_record_id": command["id"]},
            timeout=60,
        )
        assert result == {"scenario": "existing a", "agent_record_id": command["id"]}
        await runtime.execute({"action": "promote", "scenario": "existing a", "release_id": "candidate"})
        assert client.request.call_args.args == ("/reef/scenarios/existing%20a/promote",)
        count = client.request.await_count
        for command in [
            {"action": "http", "scenario": "original", "url": "http://localhost/secret"},
            {"action": "train", "scenario": "../other", "text": "x"},
            {"action": "set_training_mode", "scenario": "original", "training_mode": "invalid"},
        ]:
            with pytest.raises(ValueError):
                await runtime.execute(command)
        assert client.request.await_count == count

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure,expected",
    [
        (HTTPFailure(400, "private"), "failed"),
        (HTTPFailure(500, "private"), "unknown"),
        (TimeoutError("private"), "unknown"),
    ],
)
def test_execution_outcomes_are_not_repeated(tmp_path, failure, expected):
    async def run():
        state = ConnectorState(tmp_path)
        runtime, platform = AsyncMock(), AsyncMock()
        runtime.execute.side_effect = failure
        connector = Connector(platform, runtime, state)
        command = {"id": str(uuid.uuid4()), "action": "train"}
        try:
            await connector.execute(command)
            await connector.execute(command)
            assert runtime.execute.await_count == 1
            result = dict(state.pending())[command["id"]]
            assert result["state"] == expected and "private" not in json.dumps(result)
            platform.request.side_effect = aiohttp.ClientConnectionError()
            with pytest.raises(aiohttp.ClientConnectionError):
                await connector.flush_results()
            assert state.pending()
            platform.request.side_effect = None
            await connector.flush_results()
            assert not state.pending()
        finally:
            state.close()

    asyncio.run(run())


def test_success_reports_snapshot_atomically_and_cancel_is_unknown(tmp_path):
    async def run():
        state = ConnectorState(tmp_path)
        runtime = AsyncMock()
        runtime.execute.return_value = {"scenario": "new"}
        runtime.snapshot.return_value = {"reachable": True, "scenarios": [{"scenario": "new"}]}
        connector = Connector(AsyncMock(), runtime, state)
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        try:
            await connector.execute({"id": first, "action": "create_scenario"})
            assert dict(state.pending())[first]["snapshot"]["scenarios"] == [{"scenario": "new"}]
            runtime.execute.side_effect = asyncio.CancelledError
            with pytest.raises(asyncio.CancelledError):
                await connector.execute({"id": second, "action": "train"})
            assert dict(state.pending())[second]["state"] == "unknown"
        finally:
            state.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 409])
def test_delete_scenario_uses_native_endpoint_and_reports_outcome(tmp_path, status):
    async def run():
        seen = []

        async def handler(request):
            seen.append((request.method, request.path, request.headers.get("Authorization")))
            if request.method == "DELETE":
                return web.json_response({"archived": ["/private/records"]}, status=status)
            if request.path == "/reef/scenarios":
                return web.json_response({"scenarios": [{"scenario": "keep"}]})
            return web.json_response({"scenarios": {"keep": {"training_mode": "manual"}}})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            runtime = ReefRuntime(JSONClient(session, str(server.make_url("")).rstrip("/"), "local-test-token"))
            state = ConnectorState(tmp_path)
            connector = Connector(AsyncMock(), runtime, state)
            command = {"id": str(uuid.uuid4()), "action": "delete_scenario", "scenario": "existing a"}
            try:
                await connector.execute(command)
                await connector.execute(command)
                result = dict(state.pending())[command["id"]]
                assert result["state"] == ("succeeded" if status == 200 else "failed")
                assert "/private/records" not in json.dumps(result)
                assert [entry for entry in seen if entry[0] == "DELETE"] == [
                    ("DELETE", "/reef/scenarios/existing a", "Bearer local-test-token")
                ]
                if status == 200:
                    assert result["value"] == {"scenario": "existing a"}
                    assert result["snapshot"]["scenarios"] == [{"scenario": "keep", "training_mode": "manual"}]
                else:
                    assert "snapshot" not in result
                with pytest.raises(ValueError, match="Invalid scenario name"):
                    await runtime.execute({"action": "delete_scenario", "scenario": "../other"})
            finally:
                state.close()

    asyncio.run(run())


def test_http_boundary_keeps_tokens_separate_and_blocks_redirects():
    async def run():
        seen = []

        async def handler(request):
            seen.append((request.path, request.headers.get("Authorization"), request.headers.get("Cookie")))
            if request.path == "/redirect":
                raise web.HTTPFound("/secret")
            if request.path == "/invalid":
                return web.Response(text="accepted, but not JSON")
            if request.path == "/large":
                return web.Response(body=b"x" * (2 * 1024 * 1024 + 1))
            return web.json_response({"ok": True}, headers={"Set-Cookie": "private=value"})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server, aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as session:
            platform = JSONClient(session, str(server.make_url("")).rstrip("/"), "platform-test-token")
            local = JSONClient(session, platform.base_url, "local-test-token")
            await platform.request("/cloud", body={"ready": True})
            await local.request("/reef/status")
            for path in ("/redirect", "/invalid", "/large"):
                with pytest.raises(HTTPFailure):
                    await local.request(path)
            assert seen[:2] == [
                ("/cloud", "Bearer platform-test-token", None),
                ("/reef/status", "Bearer local-test-token", None),
            ]
            assert not any(path == "/secret" for path, _, _ in seen)

    asyncio.run(run())


def test_pairing_saves_only_after_approval_and_reuses_identity(tmp_path, monkeypatch, capsys):
    async def run():
        state = ConnectorState(tmp_path)
        calls = []

        async def handler(request):
            calls.append((request.method, request.headers.get("Authorization")))
            if request.method == "POST":
                body = await request.json()
                assert body == {"protocol": 1, "instance_id": "stable-id", "name": "my-machine"}
                return web.json_response(
                    {
                        "verification_uri": str(server.make_url("/local/authorize")),
                        "device_token": "platform-test-token",
                        "user_code": "ABCD-EF12-3456",
                        "expires_in": 30,
                    }
                )
            return web.json_response({"status": "approved", "runtime_id": "runtime-id"})

        app = web.Application()
        app.router.add_route("*", "/api/connector/pair", handler)
        try:
            async with TestServer(app) as server:
                config = {
                    "platform_url": str(server.make_url("")).rstrip("/"),
                    "instance_id": "stable-id",
                    "name": "my-machine",
                    "reef_token": "local-test-token",
                }
                await authorize(config, state, no_browser=True)
            assert state.load()["connector_token"] == "platform-test-token"
            assert calls == [("POST", None), ("GET", "Bearer platform-test-token")]
            output = capsys.readouterr().out
            assert "test-token" not in output
            assert "Device code: ABCD-EF12-3456" in output
            assert "Paste this code into the page" in output
            assert "/local/authorize" in output
            assert "/local/connect/ABCD" not in output
        finally:
            state.close()

    asyncio.run(run())


def test_connect_cli_is_dispatched_without_starting_a_deployment(capsys):
    with pytest.raises(SystemExit) as result:
        main(["connect", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--foreground" in output and "--reef-token-env" in output and "--stop" in output


def test_release_summary_is_valid_json_with_nonfinite_scores():
    summary = release_summary(
        {
            "release_id": "seed",
            "recorded_at": float("inf"),
            "metrics": {"candidate_score": float("nan"), "current_score": 0.5},
        }
    )
    assert summary == {"release_id": "seed", "metrics": {"current_score": 0.5}}
    json.dumps(summary, allow_nan=False)


@pytest.mark.skipif(os.name == "nt", reason="POSIX background process lifecycle")
def test_background_cli_stops_restarts_without_pairing_and_exits_on_revocation(tmp_path):
    async def run():
        counts = {"pair": 0, "poll": 0}
        revoked = False

        async def handler(request):
            if request.path == "/api/connector/pair":
                if request.method == "POST":
                    counts["pair"] += 1
                    return web.json_response(
                        {
                            "device_token": "background-test-token",
                            "user_code": "ABCD-EF12-3456",
                            "verification_uri": str(server.make_url("/local/authorize")),
                            "expires_in": 60,
                        }
                    )
                return web.json_response({"status": "approved", "runtime_id": "runtime-id"})
            if request.path == "/api/connector/poll":
                counts["poll"] += 1
                assert request.headers["Authorization"] == "Bearer background-test-token"
                if revoked:
                    return web.json_response({"error": "revoked"}, status=401)
                return web.json_response({"protocol": 1, "command": None})
            if request.path == "/reef/scenarios":
                return web.json_response({"scenarios": [{"scenario": "original"}]})
            return web.json_response({"scenarios": {"original": {"training_mode": "manual"}}})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        state = ConnectorState(tmp_path)

        async def wait_for_running(expected):
            for _ in range(100):
                if _running(state) is expected:
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("Connector process did not reach expected state")

        try:
            async with TestServer(app) as server:

                async def cli(*options):
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "reef.cli",
                        "connect",
                        "--no-browser",
                        "--platform",
                        str(server.make_url("")).rstrip("/"),
                        "--url",
                        str(server.make_url("")).rstrip("/"),
                        "--state-dir",
                        str(tmp_path),
                        *options,
                        cwd=Path(__file__).resolve().parents[2],
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                    assert process.returncode == 0, stderr.decode()
                    return stdout.decode()

                assert "Connected" in await cli()
                await wait_for_running(True)
                identity = state.load()["instance_id"]
                assert "running" in await cli("--status")
                await cli("--stop")
                await wait_for_running(False)
                await cli()
                await wait_for_running(True)
                assert state.load()["instance_id"] == identity and counts["pair"] == 1
                revoked = True
                await wait_for_running(False)
                assert counts["poll"] >= 2
                assert "background-test-token" not in (tmp_path / "connector.log").read_text()
        finally:
            if _running(state):
                os.kill(int((tmp_path / "connector.pid").read_text()), signal.SIGTERM)
                await wait_for_running(False)
            state.close()

    asyncio.run(run())


def foreground_platform(polled: asyncio.Event, on_approval: Callable[[], object] = lambda: None) -> web.Application:
    """A platform that approves the pairing at once and answers polls; it also answers as the Reef the connector checks."""

    async def handler(request):
        if request.path == "/api/connector/pair":
            if request.method == "POST":
                return web.json_response(
                    {
                        "device_token": "foreground-test-token",
                        "user_code": "ABCD-EF12-3456",
                        "verification_uri": str(request.url.with_path("/local/authorize")),
                        "expires_in": 60,
                    }
                )
            on_approval()
            return web.json_response({"status": "approved", "runtime_id": "runtime-id"})
        if request.path == "/api/connector/poll":
            polled.set()
            return web.json_response({"protocol": 1, "command": None})
        if request.path == "/reef/scenarios":
            return web.json_response({"scenarios": [{"scenario": "original"}]})
        return web.json_response({"scenarios": {"original": {"training_mode": "manual"}}})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    return app


async def foreground_connect(tmp_path: Path, platform_url: str) -> asyncio.subprocess.Process:
    """``reef connect --foreground`` against ``platform_url``, which also stands in for the Reef."""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "reef.cli",
        "connect",
        "--no-browser",
        "--foreground",
        "--name",
        "workstation",
        "--platform",
        platform_url,
        "--url",
        platform_url,
        "--state-dir",
        str(tmp_path),
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX process lifecycle")
def test_foreground_cli_says_it_connected_once_the_pairing_is_approved(tmp_path):
    """The foreground connector prints the line the background one prints, while it keeps running."""

    async def run():
        polled = asyncio.Event()
        async with TestServer(foreground_platform(polled)) as server:
            platform_url = str(server.make_url("")).rstrip("/")
            process = await foreground_connect(tmp_path, platform_url)
            try:
                lines = []
                while not lines or not lines[-1].startswith("Connected"):
                    line = await asyncio.wait_for(process.stdout.readline(), 15)
                    assert line, "the connector exited before it said it connected"
                    lines.append(line.decode().rstrip("\n"))
                assert lines[-1] == f"Connected workstation. Open {platform_url}/local"
                # The connector is running: it polls after it said so, and stops cleanly on SIGTERM.
                await asyncio.wait_for(polled.wait(), 15)
                assert process.returncode is None
            finally:
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)
                _, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stderr.decode()

    asyncio.run(run())


@pytest.mark.skipif(os.name == "nt", reason="POSIX process lifecycle")
def test_foreground_cli_never_says_connected_when_another_connector_took_the_lock(tmp_path):
    """A connector that started while this one waited for approval holds the lock: this one says so and exits
    without the Connected line, since it never runs."""

    async def run():
        polled = asyncio.Event()
        other = ConnectorState(tmp_path)
        held = ExitStack()
        try:
            # The other connector takes the lock after this one's start check, while the pairing waits.
            platform = foreground_platform(polled, on_approval=lambda: held.enter_context(other.lock()))
            async with TestServer(platform) as server:
                process = await foreground_connect(tmp_path, str(server.make_url("")).rstrip("/"))
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        finally:
            held.close()
            other.close()
        assert process.returncode == 1
        assert "Device code: ABCD-EF12-3456" in stdout.decode()
        assert "Connected" not in stdout.decode()
        assert "reef connect: A connector is already running for this instance" in stderr.decode()
        assert not polled.is_set()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("url", "arguments"),
    [
        ("https://127.0.0.1:8901", []),
        ("http://127.0.0.1", []),
        ("http://127.0.0.1:8901/reef", []),
        ("http://127.0.0.1:8901", ["--reef.port", "9000"]),
        ("http://127.0.0.1:8901", ["--reef.host=0.0.0.0"]),
    ],
)
def test_serve_address_comes_only_from_the_connector_url(url, arguments):
    with pytest.raises(ValueError):
        serve_address(url, arguments)


def test_serve_address_accepts_a_loopback_port():
    assert serve_address("http://localhost:8901", ["--recipe", "reefine"]) == ("localhost", 8901)


def test_connect_warns_before_pairing_when_reef_does_not_answer(tmp_path, monkeypatch, capsys):
    async def stop_before_pairing(*_args, **_kwargs):
        raise RuntimeError("pairing skipped")

    monkeypatch.setattr("reef.service.connector.authorize", stop_before_pairing)
    url = f"http://127.0.0.1:{unused_port()}"
    with pytest.raises(SystemExit):
        main(["connect", "--url", url, "--platform", "https://platform.example", "--state-dir", str(tmp_path)])
    error = capsys.readouterr().err
    assert f"Warning: Nothing answered at {url}" in error and "--serve" in error


def test_connect_serve_refuses_an_address_already_in_use(tmp_path, capsys):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        with pytest.raises(SystemExit):
            main(["connect", "--url", url, "--state-dir", str(tmp_path), "--serve", "--model", "ollama/test"])
    assert f"A service already answers at {url}" in capsys.readouterr().err


def test_serve_reports_its_exit_code(tmp_path):
    async def run():
        service = ReefService(
            ["--recipe", "missing-profile"], "http://127.0.0.1:8901", "", tmp_path, tmp_path / "serve.log"
        )
        service.start()
        await asyncio.to_thread(service.process.wait, 30)
        return service.status(reachable=False)

    assert asyncio.run(run()) == {"state": "exited", "exit_code": 2}
    assert "missing-profile" in (tmp_path / "serve.log").read_text()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process lifecycle")
def test_serve_starts_on_the_connector_url_and_stops_with_it(tmp_path):
    async def run():
        url = f"http://127.0.0.1:{unused_port()}"
        service = ReefService(["--model", "ollama/test"], url, "", tmp_path, tmp_path / "serve.log")
        service.start()
        try:
            async with aiohttp.ClientSession() as session:
                runtime = ReefRuntime(JSONClient(session, url))
                assert service.status(reachable=False) == {"state": "starting"}
                for _ in range(300):
                    if (await runtime.snapshot())["reachable"]:
                        break
                    await asyncio.sleep(0.1)
                assert service.status(reachable=True) == {"state": "running"}
                # Once Reef has answered, a missed check no longer means it is starting.
                assert service.status(reachable=False) == {"state": "running"}
        finally:
            await service.stop()
        assert service.process.returncode == 0
        async with aiohttp.ClientSession() as session:
            assert (await ReefRuntime(JSONClient(session, url)).snapshot())["error_code"] == "connection_failed"

    asyncio.run(run())


@pytest.mark.skipif(os.name == "nt", reason="POSIX background process lifecycle")
def test_background_connect_serve_reports_the_service_and_stops_it(tmp_path):
    async def run():
        snapshots = []

        async def handler(request):
            if request.path == "/api/connector/pair":
                if request.method == "POST":
                    return web.json_response(
                        {
                            "device_token": "serve-test-token",
                            "user_code": "ABCD-EF12-3456",
                            "verification_uri": str(server.make_url("/local/authorize")),
                            "expires_in": 60,
                        }
                    )
                return web.json_response({"status": "approved", "runtime_id": "runtime-id"})
            body = await request.json()
            if "snapshot" in body:
                snapshots.append(body["snapshot"])
            return web.json_response({"protocol": 1, "command": None})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        state = ConnectorState(tmp_path / "state")
        url = f"http://127.0.0.1:{unused_port()}"
        try:
            async with TestServer(app) as server:

                async def cli(*options):
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "reef.cli",
                        "connect",
                        "--no-browser",
                        "--platform",
                        str(server.make_url("")).rstrip("/"),
                        "--url",
                        url,
                        "--state-dir",
                        str(tmp_path / "state"),
                        *options,
                        cwd=tmp_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                    assert process.returncode == 0, stderr.decode()
                    return stdout.decode()

                assert f"Starting Reef at {url}" in await cli("--serve", "--model", "ollama/test")
                for _ in range(300):
                    if snapshots and snapshots[-1].get("service") == {"state": "running"}:
                        break
                    await asyncio.sleep(0.1)
                assert snapshots[-1]["reachable"] is True and snapshots[-1]["reef_url"] == url
                assert "Reef logs:" in await cli("--status")
                assert "together with the Reef service" in await cli("--stop")
                for _ in range(100):
                    if not _running(state):
                        break
                    await asyncio.sleep(0.1)
                async with aiohttp.ClientSession() as session:
                    assert (await ReefRuntime(JSONClient(session, url)).snapshot())[
                        "error_code"
                    ] == "connection_failed"
        finally:
            if _running(state):
                os.kill(int((tmp_path / "state" / "connector.pid").read_text()), signal.SIGTERM)
            state.close()

    asyncio.run(run())


@pytest.mark.parametrize("action", ["releases", "create_scenario"])
def test_completed_commands_report_and_drain_without_heartbeat_delay(tmp_path, action):
    async def run():
        state = ConnectorState(tmp_path)
        stop = asyncio.Event()
        commands = [{"id": str(uuid.uuid4()), "action": action, "scenario": "fixture"} for _ in range(3)]
        expected_ids = [command["id"] for command in commands]
        reported = []
        runtime, platform = AsyncMock(), AsyncMock()
        runtime.execute.return_value = {"scenario": "fixture"}
        runtime.snapshot.return_value = {"reachable": True, "scenarios": []}

        async def request(path, *, body):
            if path.endswith("/result"):
                reported.append(path.split("/")[-2])
                assert body["state"] == "succeeded"
                if action == "create_scenario":
                    assert body["snapshot"] == runtime.snapshot.return_value
                if len(reported) == 3:
                    stop.set()
                return {"accepted": True}
            assert body["ready"] is True
            assert len(reported) == 3 - len(commands), "Report before dispatching another operation"
            return {"protocol": 1, "command": commands.pop(0) if commands else None}

        platform.request.side_effect = request
        try:
            # Previously each result waited three seconds, so this batch took nine.
            await asyncio.wait_for(Connector(platform, runtime, state).run(stop), 1)
            assert reported == expected_ids
            assert runtime.execute.await_count == 3
            assert runtime.snapshot.await_count == (4 if action == "create_scenario" else 1)
            assert state.pending() == []
        finally:
            state.close()

    asyncio.run(run())


def test_report_failure_keeps_backoff_and_saved_result(tmp_path):
    async def run():
        state = ConnectorState(tmp_path)
        stop, attempted = asyncio.Event(), asyncio.Event()
        runtime, platform = AsyncMock(), AsyncMock()
        runtime.snapshot.return_value = {"reachable": True, "scenarios": []}
        runtime.execute.return_value = {"releases": []}
        command = {"id": str(uuid.uuid4()), "action": "releases", "scenario": "fixture"}
        attempts = 0

        async def request(path, *, body):
            nonlocal attempts
            if path.endswith("/result"):
                attempts += 1
                attempted.set()
                raise aiohttp.ClientConnectionError
            return {"protocol": 1, "command": command}

        platform.request.side_effect = request
        task = asyncio.create_task(Connector(platform, runtime, state).run(stop))
        try:
            await asyncio.wait_for(attempted.wait(), 1)
            await asyncio.sleep(0.1)
            assert attempts == 1, "A completed operation must not wake network backoff repeatedly"
            assert runtime.execute.await_count == 1
            assert dict(state.pending())[command["id"]]["state"] == "succeeded"
        finally:
            stop.set()
            await asyncio.wait_for(task, 1)
            state.close()

    asyncio.run(run())


def test_slow_operation_keeps_heartbeats_and_stops_without_replay(tmp_path):
    async def run():
        state = ConnectorState(tmp_path)
        stop = asyncio.Event()
        runtime, platform = AsyncMock(), AsyncMock()
        runtime.snapshot.return_value = {"reachable": True, "scenarios": []}
        command = {"id": str(uuid.uuid4()), "action": "train", "scenario": "fixture"}
        polls = []

        async def execute(_command):
            await asyncio.Event().wait()

        async def request(path, *, body):
            polls.append(body["ready"])
            if len(polls) == 1:
                return {"protocol": 1, "command": command}
            stop.set()
            return {"protocol": 1, "command": None}

        runtime.execute.side_effect = execute
        platform.request.side_effect = request
        try:
            await asyncio.wait_for(Connector(platform, runtime, state).run(stop), 5)
            assert polls == [True, False]
            assert runtime.execute.await_count == 1
            assert dict(state.pending())[command["id"]]["state"] == "unknown"
        finally:
            state.close()

    asyncio.run(run())
