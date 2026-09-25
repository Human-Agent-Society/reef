"""Opt-in outbound connection from a Reef service to its owner's console, optionally starting that service."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from reef.service.connector.runtime import HTTPFailure, JSONClient, ReefRuntime, endpoint_url
from reef.service.connector.service import ReefService, address_in_use, serve_address
from reef.service.connector.state import ConnectorState

logger = logging.getLogger(__name__)

BODY_LIMIT = 192 * 1024


class Connector:
    """Maintain heartbeats independently of one in-flight local operation."""

    def __init__(
        self, platform: JSONClient, runtime: ReefRuntime, state: ConnectorState, service: ReefService | None = None
    ):
        self.platform = platform
        self.runtime = runtime
        self.state = state
        self.service = service

    async def snapshot(self) -> dict[str, Any]:
        snapshot = await self.runtime.snapshot()
        if self.service is not None:
            snapshot["service"] = self.service.status(snapshot["reachable"])
        return snapshot

    async def execute(self, command: dict[str, Any]) -> None:
        command_id = str(uuid.UUID(command["id"]))
        if not self.state.start(command_id):
            return
        try:
            refresh = command.get("action") == "refresh"
            value = await self.snapshot() if refresh else await self.runtime.execute(command)
            result: dict[str, Any] = {"state": "succeeded", "value": value}
            if command.get("action") != "releases":
                result["snapshot"] = value if refresh else await self.snapshot()
            if len(json.dumps(result).encode()) > BODY_LIMIT:
                result = {"state": "failed", "error": "The result exceeds the connector size limit"}
        except asyncio.CancelledError:
            self.state.finish(
                command_id,
                {"state": "unknown", "error": "Connector stopped during execution. Check Reef before retrying."},
            )
            raise
        except HTTPFailure as exc:
            result = {
                "state": "failed" if 400 <= exc.status < 500 else "unknown",
                "error": f"Reef returned HTTP {exc.status}. Check the runtime before retrying.",
            }
        except (aiohttp.ClientError, TimeoutError):
            result = {
                "state": "unknown",
                "error": "Lost contact with Reef after dispatch. Check the runtime before retrying.",
            }
        except (ValueError, KeyError, TypeError):
            result = {"state": "failed", "error": "Reef returned an unsupported response or command"}
        self.state.finish(command_id, result)

    async def flush_results(self) -> None:
        for command_id, result in self.state.pending():
            try:
                await self.platform.request(f"/api/connector/commands/{command_id}/result", body=result)
            except HTTPFailure as exc:
                if exc.status not in (404, 409):
                    raise
                logger.warning("Result %s no longer accepts updates (HTTP %s)", command_id, exc.status)
            self.state.acknowledge(command_id)

    async def run(self, stop: asyncio.Event) -> None:
        self.state.recover()
        if self.service is not None:
            self.service.start()
        operation: asyncio.Task[None] | None = None
        next_snapshot = 0.0
        snapshot: dict[str, Any] | None = None
        retry_seconds = 3.0
        active_until = 0.0
        stop_wait = asyncio.create_task(stop.wait())
        try:
            while not stop.is_set():
                backing_off = False
                try:
                    if operation is not None and operation.done():
                        await operation
                        operation = None
                        active_until = time.monotonic() + 30
                    await self.flush_results()
                    if time.monotonic() >= next_snapshot:
                        snapshot = await self.snapshot()
                        # Check sooner while Reef is not answering, so the console sees it come up quickly.
                        next_snapshot = time.monotonic() + (15 if snapshot["reachable"] else 5)
                    body: dict[str, Any] = {"protocol": 1, "ready": operation is None}
                    if snapshot is not None:
                        body["snapshot"] = snapshot
                    response = await self.platform.request("/api/connector/poll", body=body)
                    snapshot = None
                    if response.get("protocol") != 1:
                        raise RuntimeError("Platform requires a newer connector")
                    command = response.get("command")
                    if command:
                        operation = asyncio.create_task(self.execute(command))
                        active_until = time.monotonic() + 30
                    retry_seconds = 3.0
                except HTTPFailure as exc:
                    if exc.status in (401, 403):
                        raise RuntimeError("Connection was revoked. Run reef connect to authorize it again.") from exc
                    logger.warning("Platform unavailable (HTTP %s); reconnecting", exc.status)
                    retry_seconds = min(30, retry_seconds * 2)
                    backing_off = True
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    logger.warning("Platform connection interrupted; reconnecting")
                    retry_seconds = min(30, retry_seconds * 2)
                    backing_off = True
                # Report completed work immediately and drain queued commands. Keep
                # slow-operation heartbeats and idle polling at three seconds, but
                # check every second for follow-up commands during active use.
                wait_seconds = retry_seconds
                if not backing_off and operation is None and time.monotonic() < active_until:
                    wait_seconds = 1.0
                waiters: set[asyncio.Task[bool] | asyncio.Task[None]] = {stop_wait}
                if operation is not None and not backing_off:
                    waiters.add(operation)
                await asyncio.wait(waiters, timeout=wait_seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_wait.cancel()
            with suppress(asyncio.CancelledError):
                await stop_wait
            if operation is not None:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
            if self.service is not None:
                await self.service.stop()


async def authorize(config: dict[str, Any], state: ConnectorState, *, no_browser: bool) -> dict[str, Any]:
    async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=True) as session:
        platform = JSONClient(session, config["platform_url"], config.get("connector_token", ""))
        if platform.token:
            try:
                await platform.request("/api/connector/poll", body={"protocol": 1, "ready": False})
                return config
            except HTTPFailure as exc:
                if exc.status != 401:
                    raise
        platform.token = ""
        pairing = await platform.request(
            "/api/connector/pair",
            body={
                "protocol": 1,
                "instance_id": config["instance_id"],
                "name": config["name"],
            },
        )
        verification = pairing["verification_uri"]
        login_url, platform_url = urlsplit(endpoint_url(verification)), urlsplit(config["platform_url"])
        if (login_url.scheme, login_url.netloc) != (platform_url.scheme, platform_url.netloc):
            raise ValueError("Platform returned a login URL on a different origin")
        platform.token = pairing["device_token"]
        print(
            f"Open {verification}\nDevice code: {pairing['user_code']}\nPaste this code into the page to connect {config['name']}.",
            flush=True,
        )
        print(
            "The platform can read scenario/version summaries and run training and version commands. Service credentials stay on this machine.",
            flush=True,
        )
        if not no_browser:
            webbrowser.open(verification)
        deadline = time.monotonic() + min(float(pairing.get("expires_in", 600)), 600)
        while time.monotonic() < deadline:
            status = await platform.request("/api/connector/pair")
            if status.get("status") == "approved":
                config["connector_token"] = platform.token
                config["runtime_id"] = status["runtime_id"]
                state.save(config)
                return config
            await asyncio.sleep(3)
        raise RuntimeError("Connection code expired; run reef connect again")


async def run_connector(config: dict[str, Any], state: ConnectorState) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop.set)
    async with (
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=True) as platform_session,
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as local_session,
    ):
        serve = config.get("serve")
        service = (
            ReefService(
                serve["arguments"],
                config["reef_url"],
                config.get("reef_token", ""),
                Path(serve["directory"]),
                state.directory / "serve.log",
            )
            if serve
            else None
        )
        connector = Connector(
            JSONClient(platform_session, config["platform_url"], config["connector_token"]),
            ReefRuntime(JSONClient(local_session, config["reef_url"], config.get("reef_token", ""))),
            state,
            service,
        )
        await connector.run(stop)


async def check_reef(config: dict[str, Any]) -> dict[str, Any]:
    async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as session:
        return await ReefRuntime(JSONClient(session, config["reef_url"], config.get("reef_token", ""))).snapshot()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reef connect",
        description="Connect a Reef runtime to your API platform account, optionally starting it with --serve.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8900", help="Reef service URL")
    parser.add_argument("--platform", default="https://api.reefinfra.ai", help="API platform URL")
    parser.add_argument("--name", help="Name shown in the console")
    parser.add_argument(
        "--reef-token-env", default="REEF_TOKEN", help="Environment variable holding the local service token"
    )
    parser.add_argument("--state-dir", type=Path, help="Private state directory for this connection")
    parser.add_argument("--foreground", action="store_true", help="Stay in the foreground (for service supervisors)")
    parser.add_argument("--no-browser", action="store_true", help="Print the login link without opening a browser")
    parser.add_argument(
        "--stop", action="store_true", help="Stop this connector, keeping its credentials for a later restart"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show this connector's local process and state directory"
    )
    parser.add_argument(
        "--serve",
        nargs=argparse.REMAINDER,
        metavar="SERVE_OPTION",
        help="Start `reef serve` with the options that follow on the --url address, and stop it with the connector."
        " Put it last.",
    )
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    return parser


def _running(state: ConnectorState) -> bool:
    try:
        with state.lock():
            return False
    except RuntimeError:
        return True


def main(argv: list[str] | None = None) -> None:
    """Authorize once, then maintain the connection independently of the browser."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        platform_url, reef_url = endpoint_url(args.platform), endpoint_url(args.url)
        connection_key = hashlib.sha256(f"{platform_url}\n{reef_url}".encode()).hexdigest()[:16]
        directory = args.state_dir or Path.home() / ".reef" / "connections" / connection_key
        state = ConnectorState(directory)
        try:
            running = _running(state)
            config = state.load()
            serving = bool(config.get("serve"))
            if args.status:
                print(f"Connector {'running' if running else 'stopped'}\nState: {directory}")
                if serving:
                    print(f"Reef logs: {directory / 'serve.log'}")
                return
            if args.stop:
                if running:
                    os.kill(int((directory / "connector.pid").read_text()), signal.SIGTERM)
                if serving:
                    print("Connector stopping, together with the Reef service it started.")
                else:
                    print(
                        "Connector stopping. Reef continues serving; revoke access in the console to remove"
                        " authorization."
                    )
                return
            if running:
                raise RuntimeError("This connector is already running; use --status or --stop")
            if args.run:
                if not config.get("connector_token"):
                    raise RuntimeError("Run reef connect to authorize this instance first")
                with state.lock():
                    (directory / "connector.pid").write_text(str(os.getpid()))
                    asyncio.run(run_connector(config, state))
                return
            if config and (config.get("platform_url") != platform_url or config.get("reef_url") != reef_url):
                raise ValueError(
                    "This state directory belongs to another endpoint; use its original --platform and --url"
                )
            if args.serve is not None and address_in_use(*serve_address(reef_url, args.serve)):
                raise RuntimeError(f"A service already answers at {reef_url}. Stop it, or connect without --serve.")
            config.update(
                {
                    "platform_url": platform_url,
                    "reef_url": reef_url,
                    "instance_id": config.get("instance_id") or str(uuid.uuid4()),
                    "name": args.name or config.get("name") or socket.gethostname(),
                    "reef_token": os.environ.get(args.reef_token_env, config.get("reef_token", "")),
                    # Each invocation decides whether the connector starts Reef; serve runs where connect ran.
                    "serve": (
                        {"arguments": args.serve, "directory": str(Path.cwd())} if args.serve is not None else None
                    ),
                }
            )
            # Save identity before authorization so interrupted logins cannot create new identities.
            state.save(config)
            if args.serve is None:
                reef = asyncio.run(check_reef(config))
                if not reef["reachable"]:
                    print(
                        f"Warning: {reef['error']}\nThe connection still completes; the console shows the runtime"
                        " once Reef answers. To start Reef with the connector, add --serve and its options.",
                        file=sys.stderr,
                    )
            config = asyncio.run(authorize(config, state, no_browser=args.no_browser))
            if args.serve is not None:
                print(f"Starting Reef at {reef_url}. Logs: {directory / 'serve.log'}", flush=True)
            if args.foreground:
                with state.lock():
                    (directory / "connector.pid").write_text(str(os.getpid()))
                    # Only a connector that holds the lock runs, so the line waits for it.
                    print(f"Connected {config['name']}. Open {platform_url}/local", flush=True)
                    asyncio.run(run_connector(config, state))
            else:
                log_path = directory / "connector.log"
                descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, "a") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "reef.service.connector", "--run", "--state-dir", str(directory)],
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                    )
                time.sleep(0.5)
                if process.poll() is not None:
                    raise RuntimeError(f"Connector did not start. See {log_path}")
                print(f"Connected {config['name']}. Open {platform_url}/local\nLogs: {log_path}")
        finally:
            state.close()
    except (RuntimeError, ValueError, OSError, aiohttp.ClientError) as exc:
        print(f"reef connect: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
