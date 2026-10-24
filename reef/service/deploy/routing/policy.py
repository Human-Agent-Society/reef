"""Policy-only updates of a managed routing deployment through the router's management API.

The owner changes Scenario rules, defaults and target membership in place.
Target catalog, provider, revision, endpoint, automatic selector, Scenario
names and image changes require a restart. The committed router/Reef pair and
any pending update are stored privately, so a lost reply or an owner restart is
inspected instead of replayed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import (
    ClientError,
    ClientHandlerType,
    ClientRequest,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
)

from reef.runtime.recovery import fsync_dir, write_json

#: At most this many bytes of one management reply are read and decoded; any remainder is not read.
MANAGEMENT_REPLY_LIMIT_BYTES = 16 * 1024 * 1024


class RouterPolicyError(RuntimeError):
    """The router's configuration cannot be confirmed as this owner's committed or candidate document.

    A failed mutation may still have reached the router, so its caller inspects the result and never replays it.
    The controller turns a failure before any mutation into a rejected update, and a failed read while it
    observes the router into a failed readiness check.
    """


async def refuse_replay(request: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
    """aiohttp client middleware: turn a transport failure into ``RouterPolicyError`` before aiohttp retries.

    aiohttp sends some requests again after a lost connection, a PUT included. A PUT that reached the router
    may have been applied, so it must be inspected, never sent twice.
    """
    try:
        return await handler(request)
    except ClientError as exc:
        raise RouterPolicyError(f"router management {request.method} failed") from exc


@dataclass
class RouterManagementClient:
    """Authenticated loopback calls to the router's configuration API; no redirects, proxies or replays.

    Values: ``port`` is an int from 1 to 65535 on 127.0.0.1; ``token`` is a non-empty bearer token of
    printable ASCII other than space; ``timeout_seconds`` is finite and positive.
    The synchronous deployment controller calls this client from a thread without a running event loop.
    Operations: ``call`` makes one exchange and returns ``(status, body, etag)``: the HTTP status, the reply as
    a JSON object or ``{}`` when it is not one, and the ETag or ``""``. Each call has one end on the
    ``time.monotonic`` clock: ``timeout_seconds`` after the call starts, or ``deadline`` if that is earlier.
    ``deadline`` is normally the end of the caller's phase. ``call`` sends no request once the end has
    passed, stops waiting for the reply at the end, and treats a reply decoded after the end as lost. It
    then raises ``RouterPolicyError``. A mutation may have been applied even then. The end bounds network
    waiting; local encoding and decoding are not interrupted. ``config_etag``, ``plan`` and
    ``runtime_hashes`` raise ``RouterPolicyError`` for any status but 200; ``replace`` returns the PUT's
    status and body for the caller to inspect; ``ready`` is true exactly for a 200 ``/ready``. Each passes
    its ``deadline`` to ``call``. The controller sends each PUT once.
    """

    port: int
    token: str = field(repr=False)
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("router management port must be an integer from 1 to 65535")
        if not isinstance(self.token, str) or not self.token or any(not 33 <= ord(char) <= 126 for char in self.token):
            raise ValueError("router management token must be nonempty printable ASCII without spaces")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("router management timeout must be finite positive seconds")

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        if_match: str | None = None,
        deadline: float | None = None,
    ) -> tuple[int, dict[str, object], str]:
        end = time.monotonic() + self.timeout_seconds
        if deadline is not None:
            end = min(end, deadline)
        # ``identity`` keeps the reply uncompressed, so the byte limit applies to what is decoded.
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        if if_match is not None:
            headers["If-Match"] = if_match
        data = None if body is None else json.dumps(body).encode()
        try:
            # A fresh event loop per exchange: the controller is synchronous, and nothing outlives the call.
            status, raw, etag = asyncio.run(self.exchange(method, path, data, headers, end))
        except (ClientError, TimeoutError) as exc:
            # A mutation may have been applied even though its reply was lost.
            raise RouterPolicyError(f"router management {method} failed") from exc
        try:
            decoded = json.loads(raw) if raw else {}
        except ValueError:
            decoded = {}
        if time.monotonic() > end:
            raise RouterPolicyError(f"router management {method} replied after its deadline")
        return status, decoded if isinstance(decoded, dict) else {}, etag

    async def exchange(
        self, method: str, path: str, data: bytes | None, headers: dict[str, str], end: float
    ) -> tuple[int, bytes, str]:
        """One request and at most ``MANAGEMENT_REPLY_LIMIT_BYTES`` of its reply, ending at ``end``.

        ``end`` is a ``time.monotonic`` value. No request is sent once it has passed.
        """
        async with ClientSession(
            middlewares=(refuse_replay,), cookie_jar=DummyCookieJar(), trust_env=False, auto_decompress=False
        ) as session:
            # Measured just before the request, so time spent encoding it is not given back.
            remaining_seconds = end - time.monotonic()
            if remaining_seconds <= 0:
                raise RouterPolicyError(f"router management {method} was not sent: its deadline has passed")
            # aiohttp rounds a timeout of five seconds or more up to a whole second unless ``ceil_threshold`` is
            # infinite.
            timeout = ClientTimeout(total=remaining_seconds, ceil_threshold=math.inf)
            async with session.request(
                method,
                f"http://127.0.0.1:{self.port}{path}",
                data=data,
                headers=headers,
                allow_redirects=False,
                timeout=timeout,
            ) as response:
                raw = bytearray()
                while len(raw) < MANAGEMENT_REPLY_LIMIT_BYTES and (
                    chunk := await response.content.read(MANAGEMENT_REPLY_LIMIT_BYTES - len(raw))
                ):
                    raw += chunk
                return response.status, bytes(raw), response.headers.get("ETag", "")

    def config_etag(self, *, deadline: float | None = None) -> str:
        status, _, etag = self.call("GET", "/api/v1/config", deadline=deadline)
        if status != 200 or not etag:
            raise RouterPolicyError("router configuration identity is unavailable")
        return etag

    def plan(self, document: str, *, deadline: float | None = None) -> tuple[str, str]:
        """The current and candidate source ETags for replacing the configuration with ``document``."""
        status, body, _ = self.call(
            "POST", "/api/v1/config/plan", {"yaml": document, "mode": "replace"}, deadline=deadline
        )
        current, candidate = body.get("current_etag"), body.get("candidate_etag")
        if status != 200 or not isinstance(current, str) or not isinstance(candidate, str) or not candidate:
            raise RouterPolicyError("router rejected the configuration plan")
        return current, candidate

    def replace(self, document: str, if_match: str, *, deadline: float | None = None) -> tuple[int, dict[str, object]]:
        status, body, _ = self.call("PUT", "/api/v1/config", {"yaml": document}, if_match=if_match, deadline=deadline)
        return status, body

    def runtime_hashes(self, *, deadline: float | None = None) -> dict[str, object]:
        status, body, _ = self.call("GET", "/api/v1/config/hash", deadline=deadline)
        if status != 200:
            raise RouterPolicyError("router runtime hash is unavailable")
        return body

    def ready(self, *, deadline: float | None = None) -> bool:
        return self.call("GET", "/ready", deadline=deadline)[0] == 200


def document_etag(data: bytes) -> str:
    """The router's ETag for a configuration document: the quoted hex SHA-256 of its bytes."""
    return '"' + hashlib.sha256(data).hexdigest() + '"'


def write_private_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with owner-only ``text``; the file and its directory entry are synced first."""
    directory = path.parent
    descriptor, temporary = tempfile.mkstemp(prefix=".policy-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    fsync_dir(directory)


class PolicyStateStore:
    """Read and replace the deployment's private policy journal: one JSON object in one file.

    The store holds only ``path`` and checks only that the file holds a JSON object; the controller defines
    and checks its keys. By its convention ``committed`` names the router document, its ETag and runtime
    hash, and a copy of the Reef manifest kept outside generation directories. It changes only after the
    router reports that exact runtime active, and before the Reef child publishes it. ``pending`` remains
    until the outcome is verified.
    Operations: ``load`` returns ``{}`` when the file does not exist, the object it holds otherwise, and
    raises ``ValueError`` for any other JSON value. ``write`` replaces the whole file atomically.
    ``keep_source`` copies a manifest to ``committed/<digest>.yaml`` beside the journal, privately and
    atomically, and returns that path.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("routing policy state must be an object")
        return value

    def write(self, state: dict[str, object]) -> None:
        write_json(self.path, state)

    def keep_source(self, manifest: Path, snapshot_version: str) -> Path:
        """Copy a committed Reef manifest where restarts can find it after its generation is gone."""
        directory = self.path.parent / "committed"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (snapshot_version.removeprefix("sha256:") + ".yaml")
        write_private_atomically(path, manifest.read_text(encoding="utf-8"))
        return path
