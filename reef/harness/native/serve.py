"""``reef-native serve``: one resident process per installed tree, holding the tree as a live composition.

The process boots a ``Loader`` over ``NATIVE_PLUGINS`` from ``native/tree.json``,
starts the wrapper's capture proxy in process, and listens on a Unix socket
for turns and control requests. A session holds one ``Run`` across turns. A
new head Reef serves reaches the process through the release poll or the
``x-reef-release-id`` header of an inference answer and is mounted between
two steps of the open turn, or at once when no turn is open; a mount that
leaves an entry FAILED is rolled back whole. Every mount, unmount and failed
mount is an event in the open turn's session, else in ``sessions/serve.jsonl``.
``reef-native -p`` stays the episode form: one process, one turn.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import socketserver
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from reef.harness.adapters import get_adapter
from reef.harness.harness_wrapper import HARNESS_RELEASE_SIDECAR, CaptureProxy, WrapperError
from reef.harness.model_binding import ModelBinding
from reef.harness.native import SESSION_VERSION, SPILL_DIR, Session, _Loop, binding_from
from reef.harness.native.enforce import Enforcer, InProcessEnforcer, Tool, select_enforcer
from reef.harness.native.graph import Run, run_graph
from reef.harness.native.host import NativeHost
from reef.harness.native.plugins import NATIVE_PLUGINS
from reef.harness.native.release_client import HeadWatch, ReleaseClient, ReleaseClientError
from reef.harness.native.selftools import RESERVED_NAMES, self_tools
from reef.train.cordis_backend.backend import admit_mutations
from reef.train.cordis_backend.compose import Context, FiberState
from reef.train.cordis_backend.compose.loader import Loader
from reef.train.cordis_backend.strategies import Mutation

SERVE_LOG = "serve.jsonl"
SOCKET_NAME = "serve.sock"
#: The one mount directory of a serve process; unchanged entries keep their modules there across mounts.
MOUNT_DIR = "live"
TREE_FILE = "tree.json"
FOLLOW_MODES = ("head", "pinned")
DEFAULT_POLL_INTERVAL_S = 60.0
DEFAULT_TURN_TIMEOUT_S = 600.0
#: Longer than this and the socket path does not fit ``sun_path``; a short one under /tmp stands in.
MAX_SOCKET_PATH_BYTES = 100
_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ServeError(Exception):
    """The serve process cannot start, or a request cannot be served; the message says why."""


class EventSink(Protocol):
    """Where a turn's events are echoed line by line: the socket connection that asked for the turn."""

    def send(self, line: str) -> None: ...


# -- the tree on disk --------------------------------------------------------------------------------------


def tree_layout(tree: Path) -> tuple[Path, Path]:
    """(the pulled tree root, where the sidecar sits; its ``native/`` directory); the native directory itself is accepted."""
    tree = tree.resolve()
    if (tree / "native" / "models.json").is_file():
        return tree, tree / "native"
    if tree.name == "native" and (tree / "models.json").is_file():
        return tree.parent, tree
    raise ServeError(
        f"{tree} holds no native/models.json; --tree names the pulled tree, the directory that holds native/"
    )


def socket_path(tree: Path, override: Path | None = None) -> Path:
    """``native/serve.sock`` under the tree, or a short path under /tmp when that one does not fit a socket address."""
    if override is not None:
        return override
    dest, native = tree_layout(tree)
    path = native / SOCKET_NAME
    if len(str(path).encode("utf-8")) > MAX_SOCKET_PATH_BYTES:
        return Path("/tmp") / f"reef-native-{hashlib.sha256(str(dest).encode('utf-8')).hexdigest()[:12]}.sock"
    return path


def _parse_tree(text: str, where: str) -> list[dict[str, Any]]:
    try:
        entries = json.loads(text)
    except ValueError as exc:
        raise ServeError(f"{where} is not JSON: {exc}") from exc
    if not isinstance(entries, list) or not all(
        isinstance(entry, Mapping) and entry.get("id") and entry.get("name") for entry in entries
    ):
        raise ServeError(f"{where} must be a JSON array of entries with id, name and config")
    return [dict(entry) for entry in entries]


def read_tree(native: Path) -> list[dict[str, Any]]:
    """The installed ``native/tree.json``: the entries the serve form boots from."""
    path = native / TREE_FILE
    if not path.is_file():
        raise ServeError(
            f"{path} is missing: the serve form boots from the tree file; pull the tree from a reef that renders it"
        )
    return _parse_tree(path.read_text(encoding="utf-8"), str(path))


def _entries_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files") or {}
    text = files.get(f"native/{TREE_FILE}") if isinstance(files, Mapping) else None
    if not isinstance(text, str):
        raise ServeError(f"release {manifest.get('release_id')!r} carries no native/{TREE_FILE}")
    return _parse_tree(text, f"release {manifest.get('release_id')!r} native/{TREE_FILE}")


def read_sidecar(dest: Path) -> dict[str, Any]:
    """The release sidecar beside the tree, or an empty record when there is none or it is unreadable."""
    try:
        record = json.loads((dest / HARNESS_RELEASE_SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(record) if isinstance(record, Mapping) else {}


def _without_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _last_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


# -- the live composition ----------------------------------------------------------------------------------


def _entries_of(loader: Loader) -> list[dict[str, Any]]:
    """The live entry options in tree order."""
    result = []
    for options in loader.root.data:
        entry = loader.store.get(str(options.get("id")))
        if entry is not None:
            result.append(copy.deepcopy(dict(entry.options)))
    return result


def _failures(loader: Loader) -> list[tuple[str, str, str]]:
    """Every entry that is not ACTIVE: (id, kind, error); a kind with no plugin is one of them."""
    failures = []
    for options in loader.root.data:
        id_, kind = str(options.get("id")), str(options.get("name"))
        entry = loader.store.get(id_)
        if entry is None or entry.disabled:
            continue
        fiber = entry.fiber
        if fiber is None:
            failures.append((id_, kind, f"no plugin for kind {kind!r}"))
        elif fiber.error is not None:
            failures.append((id_, kind, str(fiber.error)))
        elif fiber.state is not FiberState.ACTIVE:
            failures.append((id_, kind, f"node fiber is {fiber.state.name}"))
    return failures


# -- sessions and the log ----------------------------------------------------------------------------------


class ServeSession(Session):
    """A session file opened for append with ``seq`` continuing, safe across threads, echoed line by line to a sink."""

    def __init__(self, path: Path, sink: EventSink | None = None) -> None:
        # seq continues where the file ends: a reader wants one contiguous sequence per file across turns.
        written = 0
        if path.is_file():
            written = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        super().__init__(path)
        self._seq = written
        self._sink = sink
        self._lock = threading.Lock()
        self.path = path

    def write(self, type_: str, data: Mapping[str, Any]) -> None:
        with self._lock:
            event = {"type": type_, "seq": self._seq, "time": int(time.time() * 1000), "data": dict(data)}
            self._seq += 1
            line = json.dumps(event, ensure_ascii=False, default=str)
            self._handle.write(line + "\n")
            self._handle.flush()
            if self._sink is not None:
                self._sink.send(line)


class EventLog:
    """Where the process's own events land: the open turn's session while one is bound, else ``serve.jsonl``."""

    def __init__(self, path: Path) -> None:
        self._own = ServeSession(path)
        self._bound: Session | None = None
        self._lock = threading.Lock()

    def bind(self, session: Session | None) -> None:
        with self._lock:
            self._bound = session

    def write(self, type_: str, data: Mapping[str, Any]) -> None:
        with self._lock:
            (self._bound or self._own).write(type_, data)

    def close(self) -> None:
        self._own.close()


class HostPlaneEnforcer:
    """A host plane tool runs in process whatever ``REEF_NATIVE_ENFORCE`` says: it is reef's code, not the tree's."""

    def __init__(self, inner: Enforcer) -> None:
        self._inner = inner
        self._local = InProcessEnforcer()
        self.mode = inner.mode

    @staticmethod
    def _host_plane(tool: Tool | None) -> bool:
        return bool(getattr(tool, "host_plane", False))

    def describe(self, tool: Tool | None) -> dict[str, Any]:
        return (self._local if self._host_plane(tool) else self._inner).describe(tool)

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any:
        return (self._local if self._host_plane(tool) else self._inner).run(tool, arguments, workdir)


class _ServeLoop(_Loop):
    """One session's loop in the serve form: a queued mount lands and the wall clock is checked before each step."""

    def __init__(
        self,
        server: Server,
        session: Session,
        root: Path,
        session_dir: Path,
        header: Mapping[str, Any],
        enforcer: Enforcer,
    ) -> None:
        super().__init__(session, root, session_dir, header, enforcer=enforcer)
        self._server = server

    def before_step(self, run: Any) -> None:
        self._server.between_steps(run)


class _Conversation:
    """One session held across turns: its loop and its run, built at the first turn."""

    def __init__(self, id_: str, directory: Path) -> None:
        self.id = id_
        self.dir = directory
        self.loop: _ServeLoop | None = None
        self.run: Run | None = None


class _Pending:
    """One queued release mount and the answer whoever queued it waits for."""

    def __init__(self, manifest: Mapping[str, Any], source: str) -> None:
        self.manifest = dict(manifest)
        self.source = source
        self.release_id = str(manifest.get("release_id") or "")
        self.result: dict[str, Any] = {}
        self._done = threading.Event()

    def settle(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        self._done.set()

    def wait(self, timeout_s: float) -> dict[str, Any]:
        if not self._done.wait(timeout_s):
            return {"mounted": False, "release_id": self.release_id, "error": f"no mount within {timeout_s:.0f} s"}
        return self.result


# -- the process -------------------------------------------------------------------------------------------


class Server:
    """The resident process: the live composition, the sessions, the proxy, the release watch and the socket."""

    def __init__(
        self,
        tree: Path,
        *,
        scenario: str,
        follow: str = "head",
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        self_tools: bool = False,
        socket_override: Path | None = None,
        reef_url: str | None = None,
        turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
        token: str | None = None,
    ) -> None:
        if follow not in FOLLOW_MODES:
            raise ServeError(f"--follow must be one of {', '.join(FOLLOW_MODES)}, not {follow!r}")
        if not scenario:
            raise ServeError("the scenario is required: --scenario NAME or REEF_HARNESS_SCENARIO")
        self.dest, self.native = tree_layout(tree)
        self.scenario = scenario
        self.follow = follow
        self.poll_interval_s = float(poll_interval_s)
        self.self_tools = self_tools
        self.turn_timeout_s = float(turn_timeout_s)
        self.socket_path = socket_path(tree, socket_override)
        self.sessions_dir = self.native / "sessions"
        self.mount_dir = self.native / "mounts" / MOUNT_DIR
        self.descriptor = get_adapter("native")
        try:
            self._installed_binding = binding_from(self.native / "models.json")
        except (OSError, ValueError, KeyError) as exc:
            raise ServeError(f"{self.native / 'models.json'} does not bind a model: {exc}") from exc
        self.reef_url = _without_v1(reef_url or self._installed_binding.base_url)
        #: The token the wrapper convention names wins; the binding's key is what an installed tree carries.
        self.token = (
            token if token is not None else (os.environ.get("REEF_TOKEN") or self._installed_binding.api_key or None)
        )
        self.binding: ModelBinding = self._installed_binding
        self.client = ReleaseClient(self.reef_url, self.token, scenario)
        self._served: list[dict[str, Any]] = []
        self._release_id: str | None = None
        self._parent_release_id: str | None = None
        self._pending: _Pending | None = None
        self._trial: str | None = None
        self._conversations: dict[str, _Conversation] = {}
        self._current: _Conversation | None = None
        self._deadline = 0.0
        self._turn_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopped = threading.Event()
        self._started = False
        self._log: EventLog
        self._host: NativeHost
        self._loader: Loader
        self._proxy: CaptureProxy
        self._watch: HeadWatch
        self._enforcer: HostPlaneEnforcer
        self._socket_server: _SocketServer | None = None

    # -- what the self tools and the status read ---------------------------------------------------------------

    @property
    def release_id(self) -> str | None:
        return self._release_id

    @property
    def session_id(self) -> str | None:
        return None if self._current is None else self._current.id

    @property
    def host(self) -> NativeHost:
        return self._host

    def live_entries(self) -> list[dict[str, Any]]:
        """The entries the loader holds now: the served ones, or the trial's for the rest of a turn."""
        return _entries_of(self._loader)

    def served_entries(self) -> list[dict[str, Any]]:
        with self._state_lock:
            return [copy.deepcopy(entry) for entry in self._served]

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "release_id": self._release_id,
                "parent_release_id": self._parent_release_id,
                "follow": self.follow,
                "entries": len(self._served),
                "pending_mount": None if self._pending is None else self._pending.release_id,
                "sessions": len(self._conversations),
                "socket": str(self.socket_path),
                "self_tools": self.self_tools,
            }

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Boot: the proxy, the host, the tree as a mount with source ``boot``, the release watch, the socket."""
        self._refuse_second_process()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._log = EventLog(self.sessions_dir / SERVE_LOG)
        self._started = True
        try:
            self._enforcer = HostPlaneEnforcer(select_enforcer(os.environ))
        except ValueError as exc:
            raise ServeError(str(exc)) from exc
        sidecar = read_sidecar(self.dest)
        release = str(sidecar["release_id"]) if sidecar.get("release_id") else None
        parent = str(sidecar["parent_release_id"]) if sidecar.get("parent_release_id") else None
        self._watch = HeadWatch(self.client, self, self._log, self.poll_interval_s, mounted=release)
        self._proxy = CaptureProxy(
            self.reef_url,
            self.scenario,
            self.token,
            tags={"release": release} if release else {},
            observer=self._watch,
        )
        try:
            self._proxy.start()
        except WrapperError as exc:
            raise ServeError(str(exc)) from exc
        self.binding = replace(self._installed_binding, base_url=f"http://127.0.0.1:{self._proxy.port}")
        # A crashed process leaves modules behind; every one would be a twin of the entry that owns its name.
        shutil.rmtree(self.mount_dir, ignore_errors=True)
        ctx = Context()
        self._host = NativeHost(mount_dir=self.mount_dir)
        ctx.provide("native", self._host)
        self._loader = Loader(ctx, NATIVE_PLUGINS.get)
        if self.self_tools:
            for tool in self_tools(self):
                self._host.add_tool(tool)
        failure = self._mount(read_tree(self.native), release, parent, source="boot")
        if failure is not None:
            self.stop()
            raise ServeError(f"the installed tree does not mount: {failure['entry']}: {failure['error']}")
        self._watch.start()
        self._socket_server = _SocketServer(self.socket_path, self)
        threading.Thread(target=self._socket_server.serve_forever, name="reef-native-socket", daemon=True).start()

    def _refuse_second_process(self) -> None:
        path = self.socket_path
        if not path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(path))
        except OSError:
            path.unlink()  # a stale socket file: the process that owned it is gone
        else:
            raise ServeError(f"a serve process already listens at {path}")
        finally:
            probe.close()

    def wait(self) -> None:
        """Block until SIGINT or SIGTERM; the caller stops the process after."""

        def on_signal(signum: int, frame: Any) -> None:
            self._stopped.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, on_signal)
        while not self._stopped.wait(1.0):
            pass

    def stop(self) -> None:
        """Take the process down: the socket, the watch, every module's inverse, the proxy, the log."""
        self._stopped.set()
        if self._socket_server is not None:
            self._socket_server.shutdown()
            self._socket_server.server_close()
            self._socket_server = None
            self.socket_path.unlink(missing_ok=True)
        if hasattr(self, "_watch"):
            self._watch.stop()
        if hasattr(self, "_loader"):
            with self._turn_lock:
                self._loader.root.update([])
        shutil.rmtree(self.mount_dir, ignore_errors=True)
        if self.mount_dir.parent.is_dir() and not any(self.mount_dir.parent.iterdir()):
            self.mount_dir.parent.rmdir()
        if hasattr(self, "_proxy"):
            self._proxy.stop()
        if self._started:
            self._log.close()
            self._started = False

    # -- requests over the socket ------------------------------------------------------------------------------

    def handle(self, request: Any, sink: EventSink) -> dict[str, Any]:
        """One request's final answer; a turn's events went to ``sink`` on the way."""
        if not isinstance(request, Mapping):
            return _error("request must be a JSON object")
        if "turn" in request:
            turn = request["turn"]
            if not isinstance(turn, Mapping) or not isinstance(turn.get("prompt"), str):
                return _error("turn must be an object with a string prompt")
            session = turn.get("session")
            if session is not None and not isinstance(session, str):
                return _error("turn session must be a string or null")
            workdir = turn.get("workdir")
            if workdir is not None and not isinstance(workdir, str):
                return _error("turn workdir must be a string")
            try:
                result = self.turn(turn["prompt"], session, Path(workdir or os.getcwd()), sink)
            except ServeError as exc:
                return _error(str(exc))
            return {"type": "turn/result", "data": result}
        if "control" in request:
            control = request["control"]
            if control == "status":
                return {"type": "control/result", "data": self.status()}
            if control == "mount":
                release_id = request.get("release_id")
                if not isinstance(release_id, str) or not release_id:
                    return _error("mount needs a release_id")
                return {"type": "control/result", "data": self.mount(release_id)}
            return _error(f"unknown control {control!r}; status or mount")
        return _error("request must carry turn or control")

    def mount(self, release_id: str) -> dict[str, Any]:
        """A manual mount: fetch the release, queue it, and answer once it landed or failed."""
        try:
            manifest = self.client.fetch(release_id)
        except ReleaseClientError as exc:
            return {"mounted": False, "release_id": release_id, "error": str(exc)}
        pending = self._enqueue(manifest, "release")
        self._drain_if_idle()
        return pending.wait(self.turn_timeout_s + 5.0)

    # -- the head ----------------------------------------------------------------------------------------------

    def new_head(self, release_id: str, source: str) -> None:
        """A head the watch learned of: queued for a mount under ``head``, announced and left alone under ``pinned``."""
        if self.follow == "pinned":
            self._log.write("release/available", {"release_id": release_id})
            return
        self._enqueue(self.client.fetch(release_id), "release")
        self._drain_if_idle()

    def _enqueue(self, manifest: Mapping[str, Any], source: str) -> _Pending:
        pending = _Pending(manifest, source)
        with self._state_lock:
            previous, self._pending = self._pending, pending
        if previous is not None:
            previous.settle(
                {"mounted": False, "release_id": previous.release_id, "error": f"superseded by {pending.release_id}"}
            )
        return pending

    def _drain_if_idle(self) -> None:
        if not self._turn_lock.acquire(blocking=False):
            return
        try:
            self._drain_pending()
        finally:
            self._turn_lock.release()

    def _drain_pending(self) -> None:
        """Land the newest queued mount; the caller holds the turn lock. A trial keeps the served entries until the turn ends."""
        while self._trial is None:
            with self._state_lock:
                pending, self._pending = self._pending, None
            if pending is None:
                return
            pending.settle(self._mount_manifest(pending))

    def _mount_manifest(self, pending: _Pending) -> dict[str, Any]:
        manifest = pending.manifest
        try:
            entries = _entries_from_manifest(manifest)
        except ServeError as exc:
            self._log.write(
                "harness/mount-failed",
                {"release_id": pending.release_id, "source": pending.source, "entry": None, "error": str(exc)},
            )
            return {"mounted": False, "release_id": pending.release_id, "error": str(exc)}
        parent = manifest.get("parent_release_id")
        failure = self._mount(
            entries,
            pending.release_id,
            None if parent is None else str(parent),
            source=pending.source,
            manifest=manifest,
        )
        if failure is not None:
            return {
                "mounted": False,
                "release_id": pending.release_id,
                "error": f"{failure['entry']}: {failure['error']}",
            }
        return {"mounted": True, "release_id": pending.release_id, "error": None}

    # -- mounts ------------------------------------------------------------------------------------------------

    def _mount(
        self,
        entries: Sequence[Mapping[str, Any]],
        release_id: str | None,
        parent_release_id: str | None,
        *,
        source: str,
        manifest: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """``root.update(entries)`` checked entry by entry; None on success, else the first failure after the rollback."""
        previous = self.served_entries()
        failures = self._reserved(entries)
        if not failures:
            self._loader.root.update([copy.deepcopy(dict(entry)) for entry in entries])
            failures = _failures(self._loader)
        if failures:
            self._restore(previous)
            entry, kind, error = failures[0]
            self._log.write(
                "harness/mount-failed",
                {"release_id": release_id, "source": source, "entry": entry, "kind": kind, "error": error},
            )
            return {"entry": entry, "error": error}
        if source in ("boot", "release"):
            with self._state_lock:
                self._served = [copy.deepcopy(dict(entry)) for entry in entries]
                self._release_id = release_id
                self._parent_release_id = parent_release_id
            if release_id:
                self._proxy.tags["release"] = release_id
            else:
                self._proxy.tags.pop("release", None)
            self._watch.mounted(release_id)
            if source == "release" and manifest is not None:
                self._install(manifest)
        self._log.write(
            "harness/mount",
            {
                "release_id": release_id,
                "parent_release_id": parent_release_id,
                "source": source,
                "entries": len(entries),
                **(extra or {}),
            },
        )
        return None

    @staticmethod
    def _reserved(entries: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
        """A tree entry that takes a self tool's name: the host plane owns those names."""
        failures = []
        for entry in entries:
            config = entry.get("config")
            name = str(config.get("name")) if isinstance(config, Mapping) else ""
            if entry.get("name") == "native_tool" and name in RESERVED_NAMES:
                failures.append(
                    (
                        str(entry.get("id")),
                        "native_tool",
                        f"reserved name {name!r}: the host plane's self tools own it",
                    )
                )
        return failures

    def _restore(self, previous: Sequence[Mapping[str, Any]]) -> None:
        self._loader.root.update([copy.deepcopy(dict(entry)) for entry in previous])
        if _failures(self._loader):
            # An in place rollback can trip on a name two entries swapped; a clean reload restores the composition exactly.
            self._loader.root.update([])
            self._loader.root.update([copy.deepcopy(dict(entry)) for entry in previous])

    def _install(self, manifest: Mapping[str, Any]) -> None:
        """What a restart boots from: the mounted release's tree file and the sidecar naming it."""
        files = manifest.get("files") or {}
        text = files.get(f"native/{TREE_FILE}") if isinstance(files, Mapping) else None
        if isinstance(text, str):
            (self.native / TREE_FILE).write_text(text, encoding="utf-8")
        record = {
            "release_id": manifest.get("release_id"),
            "parent_release_id": manifest.get("parent_release_id"),
            "content_id": manifest.get("content_id"),
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (self.dest / HARNESS_RELEASE_SIDECAR).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def try_mount(self, mutations: Sequence[Mutation], try_id: str) -> dict[str, Any]:
        """Mount the served entries plus ``mutations`` for the rest of the open turn; the sidecar stays as it is."""
        if self._current is None:
            return {"error": "no turn is open"}
        if self._trial is not None:
            self._unmount_trial()  # one trial at a time: a second try replaces the first
        records = [
            {"op": m.op, "id": m.id, "options": None if m.options is None else dict(m.options)} for m in mutations
        ]
        entries, reason = admit_mutations(self.served_entries(), mutations, self.descriptor)
        if reason is not None:
            return {"error": reason}
        failure = self._mount(
            entries,
            self._release_id,
            self._parent_release_id,
            source="try",
            extra={"try_id": try_id, "mutations": records},
        )
        if failure is not None:
            return failure
        self._trial = try_id
        self._proxy.tags["trial"] = try_id
        return {"try_id": try_id, "mounted": True}

    def _unmount_trial(self) -> None:
        if self._trial is None:
            return
        try_id, self._trial = self._trial, None
        self._restore(self._served)
        self._proxy.tags.pop("trial", None)
        self._log.write(
            "harness/unmount",
            {"try_id": try_id, "release_id": self._release_id, "source": "rollback", "entries": len(self._served)},
        )

    # -- turns -------------------------------------------------------------------------------------------------

    def between_steps(self, run: Run) -> None:
        """At the top of every model stage: the wall clock ends the turn, then a queued mount lands."""
        if time.monotonic() > self._deadline:
            run.end_turn({"kind": "turn-timeout", "seconds": self.turn_timeout_s}, "budget")
        self._drain_pending()

    def _conversation(self, session_id: str | None) -> _Conversation:
        with self._state_lock:
            if session_id is None:
                session_id = secrets.token_hex(6)
                while session_id in self._conversations:
                    session_id = secrets.token_hex(6)
            conversation = self._conversations.get(session_id)
            if conversation is None:
                conversation = self._conversations[session_id] = _Conversation(
                    session_id, self.sessions_dir / session_id
                )
            return conversation

    def turn(
        self, prompt: str, session_id: str | None, workdir: Path, sink: EventSink | None = None
    ) -> dict[str, Any]:
        """One turn on one session; the events go to ``sink`` as written and the result comes back."""
        if session_id is not None and not _SESSION_NAME.fullmatch(session_id):
            raise ServeError(f"session id {session_id!r} must match {_SESSION_NAME.pattern}")
        with self._turn_lock:
            conversation = self._conversation(session_id)
            session = ServeSession(conversation.dir / "session.jsonl", sink)
            self._current = conversation
            self._log.bind(session)
            self._deadline = time.monotonic() + self.turn_timeout_s
            try:
                run = self._begin_turn(conversation, prompt, workdir, session)
                exit_code = run_graph(run, self._host.graph("main"))
                result = {
                    "exit": exit_code,
                    "session": conversation.id,
                    "turn": run.turn,
                    "text": _last_text(run.messages),
                }
            finally:
                self._unmount_trial()
                self._proxy.publish_turn()
                if conversation.loop is not None:
                    conversation.loop.open.clear()
                self._log.bind(None)
                self._current = None
                session.close()
                self._drain_pending()
        self._drain_if_idle()
        return result

    def _begin_turn(self, conversation: _Conversation, prompt: str, workdir: Path, session: ServeSession) -> Run:
        loop, run = conversation.loop, conversation.run
        if loop is None or run is None:
            header = {
                "version": SESSION_VERSION,
                "model": self.binding.model,
                "base_url": self.binding.base_url,
                "cwd": str(workdir),
                "enforcement": self._enforcer.mode,
                "mode": "serve",
                "session": conversation.id,
                "release_id": self._release_id,
            }
            loop = _ServeLoop(self, session, self.native, conversation.dir, header, self._enforcer)
            run = Run(loop, prompt, self.binding, self._host, workdir, session=session)
            tools, hooks = run.tools, run.hooks
            session.write(
                "session",
                {
                    **header,
                    "task": prompt,
                    "agent": "root",
                    "turn": 1,
                    "tools": sorted(tools),
                    "capabilities": {name: list(tools[name].capabilities) for name in sorted(tools)},
                    "hooks": {hook.name: event for event, listeners in hooks.items() for hook in listeners},
                    "graph": self._host.graph("main").source,
                    "agents": sorted(self._host.agents),
                    "tree": TREE_FILE,
                },
            )
            conversation.loop, conversation.run = loop, run
        else:
            loop.session = session
            run.session = session
            run.turn += 1
            run.prompt = prompt
            run.workdir = workdir
            run.messages.append({"role": "user", "content": prompt})
            run.step, run.tool_calls, run.tool_errors, run.step_open, run.last = 0, 0, 0, False, {}
        # Steps restart at 1 each turn, so the spill files of two turns get directories of their own.
        loop.SPILL_DIR = f"{SPILL_DIR}/t{run.turn}"
        session.write("turn/start", {"turn": run.turn, "prompt": prompt, "cwd": str(workdir)})
        return run


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "data": {"message": message}}


# -- the socket --------------------------------------------------------------------------------------------


class _ConnectionSink:
    """Lines to one connection; a client that went away stops the echo and never the turn."""

    def __init__(self, wfile: Any) -> None:
        self._wfile = wfile
        self.alive = True

    def send(self, line: str) -> None:
        if not self.alive:
            return
        try:
            self._wfile.write((line + "\n").encode("utf-8"))
            self._wfile.flush()
        except OSError:
            self.alive = False


class _Handler(socketserver.StreamRequestHandler):
    """One request per connection: one JSON line in, the events and the final answer out."""

    def handle(self) -> None:
        server: _SocketServer = self.server  # type: ignore[assignment]
        sink = _ConnectionSink(self.wfile)
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
        except ValueError:
            sink.send(json.dumps(_error("request is not one JSON line")))
            return
        try:
            reply = server.reef.handle(request, sink)
        except Exception as exc:
            traceback.print_exc()
            reply = _error(f"{type(exc).__name__}: {exc}"[:600])
        sink.send(json.dumps(reply, ensure_ascii=False, default=str))


class _SocketServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, reef: Server) -> None:
        self.reef = reef
        super().__init__(str(path), _Handler)


def request(path: Path, payload: Mapping[str, Any], sink: EventSink | None = None) -> dict[str, Any]:
    """One request to a serve process: every line but the last goes to ``sink``; the last is the answer."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        try:
            conn.connect(str(path))
        except OSError as exc:
            raise ServeError(f"no serve process at {path}: {exc}") from exc
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        last: str | None = None
        with conn.makefile("r", encoding="utf-8") as lines:
            for line in lines:
                if last is not None and sink is not None:
                    sink.send(last)
                last = line.rstrip("\n")
    if last is None:
        raise ServeError("the serve process closed the connection without an answer")
    return json.loads(last)


# -- the subcommands ---------------------------------------------------------------------------------------


class _Stdout:
    def __init__(self, quiet: bool) -> None:
        self._quiet = quiet

    def send(self, line: str) -> None:
        if not self._quiet:
            print(line, flush=True)


def serve_command(args: Any) -> int:
    try:
        server = Server(
            args.tree,
            scenario=args.scenario or "",
            follow=args.follow,
            poll_interval_s=args.poll_interval,
            self_tools=args.self_tools,
            socket_override=args.socket,
            reef_url=args.reef_url,
            turn_timeout_s=args.turn_timeout,
        )
        server.start()
    except ServeError as exc:
        print(f"reef-native serve: {exc}", file=sys.stderr)
        return 2
    print(f"reef-native serve: release {server.release_id} on {server.socket_path}", file=sys.stderr, flush=True)
    try:
        server.wait()
    finally:
        server.stop()
    return 0


def turn_command(args: Any) -> int:
    payload = {
        "turn": {"prompt": args.prompt, "session": args.session, "workdir": str(Path(args.workdir or ".").resolve())}
    }
    try:
        reply = request(socket_path(args.tree, args.socket), payload, _Stdout(args.quiet))
    except ServeError as exc:
        print(f"reef-native turn: {exc}", file=sys.stderr)
        return 2
    if reply.get("type") != "turn/result":
        print(f"reef-native turn: {(reply.get('data') or {}).get('message', reply)}", file=sys.stderr)
        return 2
    data = reply["data"]
    print(data.get("text", "") if args.quiet else json.dumps(reply, ensure_ascii=False), flush=True)
    return int(data.get("exit", 1))


def control_command(args: Any) -> int:
    payload: dict[str, Any] = {"control": args.command}
    if args.command == "mount":
        payload["release_id"] = args.release_id
    try:
        reply = request(socket_path(args.tree, args.socket), payload)
    except ServeError as exc:
        print(f"reef-native {args.command}: {exc}", file=sys.stderr)
        return 2
    if reply.get("type") != "control/result":
        print(f"reef-native {args.command}: {(reply.get('data') or {}).get('message', reply)}", file=sys.stderr)
        return 2
    print(json.dumps(reply["data"], indent=2, sort_keys=True), flush=True)
    return 0 if args.command == "status" or reply["data"].get("mounted") else 1


def run_command(args: Any) -> int:
    if args.command == "serve":
        return serve_command(args)
    if args.command == "turn":
        return turn_command(args)
    return control_command(args)


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_TURN_TIMEOUT_S",
    "FOLLOW_MODES",
    "MOUNT_DIR",
    "SERVE_LOG",
    "SOCKET_NAME",
    "EventLog",
    "EventSink",
    "HostPlaneEnforcer",
    "ServeError",
    "ServeSession",
    "Server",
    "admit_mutations",
    "read_sidecar",
    "read_tree",
    "request",
    "run_command",
    "socket_path",
    "tree_layout",
]
