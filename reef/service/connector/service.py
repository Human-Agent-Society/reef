"""Run `reef serve` for `reef connect --serve`, so the console can tell starting, serving and exited apart."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# `reef serve` stops its own services within 30 seconds of SIGTERM.
STOP_TIMEOUT_SECONDS = 40


def serve_address(reef_url: str, arguments: list[str]) -> tuple[str, int]:
    """The host and port `reef serve` listens on: always the connector's own URL, so the two cannot disagree."""
    parsed = urlsplit(reef_url)
    hostname = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost"
    if parsed.scheme != "http" or not loopback or parsed.port is None or parsed.path not in ("", "/"):
        raise ValueError("--serve starts Reef on this machine; use --url http://127.0.0.1:PORT")
    for argument in arguments:
        if argument.split("=", 1)[0] in ("--reef.host", "--reef.port"):
            raise ValueError("Set Reef's address with --url; --serve passes it to reef serve")
    return hostname, parsed.port


def address_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class ReefService:
    """A `reef serve` process started and stopped by the connector.

    It runs in its own session, so a terminal interrupt reaches only the connector, which then stops it.
    """

    def __init__(self, arguments: list[str], reef_url: str, token: str, directory: Path, log_path: Path):
        self.host, self.port = serve_address(reef_url, arguments)
        self.arguments = arguments
        self.token = token
        self.directory = directory
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.answered = False
        self.exit_logged = False

    def start(self) -> None:
        environment = dict(os.environ)
        if self.token:
            # Profiles read the service token from REEF_TOKEN; keep it equal to the token the connector sends.
            environment["REEF_TOKEN"] = self.token
        command = [
            sys.executable,
            "-m",
            "reef.cli",
            "serve",
            *self.arguments,
            "--reef.host",
            self.host,
            "--reef.port",
            str(self.port),
        ]
        descriptor = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as output:
            self.process = subprocess.Popen(
                command,
                cwd=self.directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        logger.info("Started reef serve (PID %s); logs: %s", self.process.pid, self.log_path)

    def status(self, reachable: bool) -> dict[str, Any]:
        """`starting` until Reef first answers, then `running`, or `exited` with the exit code."""
        if self.process is None:
            raise RuntimeError("reef serve has not been started")
        exit_code = self.process.poll()
        if exit_code is not None:
            if not self.exit_logged:
                logger.warning("reef serve exited with code %s; see %s", exit_code, self.log_path)
                self.exit_logged = True
            return {"state": "exited", "exit_code": exit_code}
        self.answered = self.answered or reachable
        return {"state": "running" if self.answered else "starting"}

    async def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            await asyncio.to_thread(self.process.wait, STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("reef serve did not stop within %s seconds; killing it", STOP_TIMEOUT_SECONDS)
            self.process.kill()
            await asyncio.to_thread(self.process.wait)
