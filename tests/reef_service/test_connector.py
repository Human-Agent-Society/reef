"""Connector contracts: bounded access, private credentials and durable command outcomes."""

import asyncio
import json
import os
import signal
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.cli import main
from reef.service.connector import Connector, _running, authorize
from reef.service.connector.runtime import HTTPFailure, JSONClient, ReefRuntime, endpoint_url, release_summary
from reef.service.connector.state import ConnectorState


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
        client.request.side_effect = [
            {"scenarios": [{"scenario": "original", "release_id": "seed", "token": "private"}]},
            {"scenarios": {"original": {"training_mode": "manual", "provider_key": "private"}}, "config": "private"},
        ]
        snapshot = await ReefRuntime(client).snapshot()
        assert snapshot == {
            "reachable": True,
            "scenarios": [{"scenario": "original", "release_id": "seed", "training_mode": "manual"}],
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
