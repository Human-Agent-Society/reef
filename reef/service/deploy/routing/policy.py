"""Policy-only updates of a managed routing deployment through the router's management API.

The owner changes Scenario rules, defaults and target membership in place.
Target catalog, provider, revision, endpoint, automatic selector, Scenario
names and image changes require a restart. The committed router/Reef pair and
any pending update are stored privately, so a lost reply or an owner restart is
inspected instead of replayed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
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

from reef.runtime.durable_files import fsync_dir, write_json

#: At most this many bytes of one management reply are read and decoded; any remainder is not read.
MANAGEMENT_REPLY_LIMIT_BYTES = 16 * 1024 * 1024


class RouterPolicyError(RuntimeError):
    """The router's configuration cannot be confirmed as this owner's committed or candidate document.

    A failed mutation may still have reached the router, so its caller inspects the result and never replays it.
    The controller turns a failure before any mutation into a rejected update, and a failed read while it
    observes the router into a failed readiness check.
    """


@dataclass(frozen=True)
class RouterRuntime:
    """What the router's ``/api/v1/config/hash`` reports about the runtime it generated and the one it serves.

    Values: each field is None or a non-empty string. A reported field is kept only when it is a non-empty string,
    so a missing field, a null and any other value read as None. ``activation_status`` keeps whatever status
    the router reports; the pinned router reports ``active``, ``pending``, ``failed`` or ``unknown``.
    Purpose: ``RouterManagementClient.runtime_hashes`` makes it from one reply. The deployment controller treats
    the router as serving its generated runtime only when the status is ``active`` and both hashes are present
    and equal, and compares the active hash with its committed pair.
    """

    activation_status: str | None
    generated_runtime_hash: str | None
    active_runtime_hash: str | None

    def __post_init__(self) -> None:
        for value in (self.activation_status, self.generated_runtime_hash, self.active_runtime_hash):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("router runtime fields must be None or non-empty strings")

    @classmethod
    def from_reply(cls, body: Mapping[str, object]) -> RouterRuntime:
        def reported(name: str) -> str | None:
            value = body.get(name)
            return value if isinstance(value, str) and value else None

        return cls(reported("activation_status"), reported("generated_runtime_hash"), reported("active_runtime_hash"))


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
    ``runtime_hashes`` raise ``RouterPolicyError`` for any status but 200, and ``runtime_hashes`` also for a
    reply that is not a non-empty JSON object; ``replace`` returns the PUT's
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

    def runtime_hashes(self, *, deadline: float | None = None) -> RouterRuntime:
        status, body, _ = self.call("GET", "/api/v1/config/hash", deadline=deadline)
        # ``call`` reads a reply that is not a JSON object as ``{}``, which reports nothing.
        if status != 200 or not body:
            raise RouterPolicyError("router runtime hash is unavailable")
        return RouterRuntime.from_reply(body)

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


@dataclass(frozen=True)
class CommittedPolicy:
    """The router document and Reef manifest that the deployment serves, as the policy journal records them.

    Values: every field is a non-empty string; paths are not checked to exist. ``generation`` is the routing
    generation that recorded the pair and ``snapshot_version`` its Reef snapshot. ``source`` is the manifest
    copy that ``PolicyStateStore.keep_source`` keeps outside generation directories. ``manifest`` and
    ``router_document`` are the files rendered for the pair, and ``router_etag`` and ``runtime_hash`` are the
    router's identities of that document.
    Purpose: the deployment controller makes it only after the router reports exactly this runtime active, and
    before the Reef child publishes it. A start reads ``source`` to recover from an interrupted update, and every
    opening checks that the router still runs ``router_etag`` and ``runtime_hash``.
    """

    generation: str
    snapshot_version: str
    source: str
    manifest: str
    router_document: str
    router_etag: str
    runtime_hash: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in dataclasses.astuple(self)):
            raise ValueError("a committed routing policy needs non-empty strings in every field")


@dataclass(frozen=True)
class PendingPolicy:
    """A router mutation the journal records before it starts, kept until its outcome is verified.

    Values: each field is None or a string. This owner records ``phase`` (``replacing``, ``publishing``,
    ``restoring`` or ``restored``), the replacement's ``from_etag`` and ``to_etag``, the candidate's
    ``snapshot_version``, ``manifest`` and ``router_document``, and ``restore_etag`` while restoring. A record
    from an older owner may lack any field: only that a pending update exists changes what a start or a check
    does, and the fields are kept for diagnosis.
    Purpose: the deployment controller makes it before each router mutation and drops it once the router is
    verified to run the committed pair.
    """

    phase: str | None = None
    from_etag: str | None = None
    to_etag: str | None = None
    snapshot_version: str | None = None
    manifest: str | None = None
    router_document: str | None = None
    restore_etag: str | None = None

    def __post_init__(self) -> None:
        if any(value is not None and not isinstance(value, str) for value in dataclasses.astuple(self)):
            raise ValueError("pending routing policy fields must be strings")


@dataclass(frozen=True)
class PolicyState:
    """The deployment's policy journal: the committed pair and the pending update, each when one exists.

    Values: ``committed`` is None or a ``CommittedPolicy``, and ``pending`` is None or a ``PendingPolicy``. Both
    are None before a generation first records its pair. A pending update without a committed pair is kept as
    read, so a start reports it as an interrupted update that needs explicit recovery.
    Wire form: one JSON object with only the optional keys ``committed`` and ``pending``. A missing key and null
    both read as None. ``committed`` holds every field of ``CommittedPolicy``; ``pending`` holds any of the fields
    of ``PendingPolicy``, and ``{}`` is a pending update without details. ``to_json`` omits None, also inside
    ``pending``, so each value it writes reads back equal.
    Purpose: ``PolicyStateStore`` reads and replaces it; the deployment controller makes each new value.
    """

    committed: CommittedPolicy | None = None
    pending: PendingPolicy | None = None

    def __post_init__(self) -> None:
        if self.committed is not None and not isinstance(self.committed, CommittedPolicy):
            raise ValueError("a routing policy state's committed pair must be a CommittedPolicy or None")
        if self.pending is not None and not isinstance(self.pending, PendingPolicy):
            raise ValueError("a routing policy state's pending update must be a PendingPolicy or None")

    def to_json(self) -> dict[str, object]:
        state: dict[str, object] = {}
        if self.committed is not None:
            state["committed"] = dataclasses.asdict(self.committed)
        if self.pending is not None:
            pending = dataclasses.asdict(self.pending)
            state["pending"] = {name: value for name, value in pending.items() if value is not None}
        return state

    @classmethod
    def from_json(cls, value: object) -> PolicyState:
        """Read the wire form.

        Raises:
            ValueError: ``value`` is not that form, or a field is outside its domain.
        """
        if not isinstance(value, dict) or value.keys() - {"committed", "pending"}:
            raise ValueError("routing policy state must be an object with only committed and pending")
        committed, pending = value.get("committed"), value.get("pending")
        committed_fields = {item.name for item in dataclasses.fields(CommittedPolicy)}
        if committed is not None and (not isinstance(committed, dict) or committed.keys() != committed_fields):
            raise ValueError("a committed routing policy must be an object with exactly its fields")
        pending_fields = {item.name for item in dataclasses.fields(PendingPolicy)}
        if pending is not None and (not isinstance(pending, dict) or pending.keys() - pending_fields):
            raise ValueError("a pending routing policy must be an object with only its fields")
        return cls(
            None if committed is None else CommittedPolicy(**committed),
            None if pending is None else PendingPolicy(**pending),
        )


class PolicyStateStore:
    """Read and replace the deployment's private policy journal, one ``PolicyState`` in one file.

    The store holds only ``path``. The committed pair changes only after the router reports that exact runtime
    active, and before the Reef child publishes it. A pending update remains until its outcome is verified.
    Operations: ``load`` returns an empty ``PolicyState`` when the file does not exist, the state it holds
    otherwise, and raises ``ValueError`` for any other content. ``write`` replaces the whole file atomically.
    ``keep_source`` copies a manifest to ``committed/<digest>.yaml`` beside the journal, privately and
    atomically, and returns that path.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> PolicyState:
        if not self.path.exists():
            return PolicyState()
        return PolicyState.from_json(json.loads(self.path.read_text(encoding="utf-8")))

    def write(self, state: PolicyState) -> None:
        write_json(self.path, state.to_json())

    def keep_source(self, manifest: Path, snapshot_version: str) -> Path:
        """Copy a committed Reef manifest where restarts can find it after its generation is gone."""
        directory = self.path.parent / "committed"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (snapshot_version.removeprefix("sha256:") + ".yaml")
        write_private_atomically(path, manifest.read_text(encoding="utf-8"))
        return path
