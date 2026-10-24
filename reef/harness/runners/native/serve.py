"""``reef-native serve``: one resident process that runs a scenario's harness release and follows its head.

A mount renders a source tree for the native adapter into a fresh directory
under ``native/mounts/`` and builds a ``NativeHost`` from the rendered files.
The process boots on the source tree it mounted last (``source/`` beside the
release file), else on the head's, read through
``GET /reef/harness?form=source``. It starts the capture proxy in process
and listens on a Unix socket for turns and control requests. A session holds
one ``Run`` across turns. A new head reaches the process through the release
poll or the ``x-reef-release-id`` header of an inference answer, and its host
replaces the current one between two steps of the open turn, or at once when
no turn is open. A tree that does not render or load leaves the current host
in place. Every mount, unmount and failed mount is an event in the open
turn's session, else in ``sessions/serve.jsonl``. ``reef-native -p`` stays
the episode form: one process, one turn.
"""

from __future__ import annotations

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
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reef.harness.adapters import get_adapter
from reef.harness.capture import CaptureError, CaptureProxy
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.runners.native import SESSION_VERSION, TOOL_OUTPUT_DIR, LoadError, Session, _Loop, binding_from
from reef.harness.runners.native.enforce import Enforcer, InProcessEnforcer, Tool, select_enforcer
from reef.harness.runners.native.graph import GraphError, Run, run_graph, run_loop_module
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.release_client import (
    EventWriter,
    HeadWatch,
    ReleaseClient,
    ReleaseClientError,
    ReleaseUpdateListener,
)
from reef.harness.runners.native.selftools import HostTool, ServeState, self_tools
from reef.harness.tree import HarnessTree, RenderError, TreePatch, render_tree, validate_patch

SERVE_LOG = "serve.jsonl"
SOCKET_NAME = "serve.sock"
#: The directory under ``native/`` that holds one rendered directory per mount; only the current one stays.
MOUNTS_DIR = "mounts"
#: The release file the serve process writes at the tree root: the release it mounted, so a restart resumes it.
HARNESS_RELEASE_FILE = ".reef-harness-release"
#: The directory beside the release file that holds the mounted release's source tree.
SOURCE_DIR = "source"
FOLLOW_MODES = ("head", "pinned")
DEFAULT_POLL_INTERVAL_S = 60.0
DEFAULT_TURN_TIMEOUT_S = 600.0
#: Longer than this and the socket path does not fit ``sun_path``; a short one under /tmp stands in.
MAX_SOCKET_PATH_BYTES = 100
_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ServeError(Exception):
    """The serve process cannot start, or a request cannot be served; the message says why."""


class EventSink(ABC):
    """Where a turn's events are echoed line by line: the socket connection that asked for the turn."""

    @abstractmethod
    def send(self, line: str) -> None: ...


# -- the tree on disk --------------------------------------------------------------------------------------


def tree_layout(tree: Path) -> tuple[Path, Path]:
    """(the pulled tree root, where the release file sits; its ``native/`` directory); the native directory itself is accepted."""
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


def read_source(dest: Path) -> HarnessTree | None:
    """The source tree of the release the process mounted last; ``None`` before the first mount."""
    directory = dest / SOURCE_DIR
    if not directory.is_dir():
        return None
    try:
        return HarnessTree.read(directory)
    except ValueError as exc:
        raise ServeError(f"{directory} is not a harness tree: {exc}") from exc


def tree_of(manifest: Mapping[str, Any]) -> HarnessTree:
    """The source tree a ``form=source`` manifest carries as ``files``."""
    try:
        return HarnessTree(dict(manifest.get("files") or {}))
    except (TypeError, ValueError) as exc:
        raise ServeError(f"release {manifest.get('release_id')!r} carries no source tree: {exc}") from exc


def read_release_info(dest: Path) -> dict[str, Any]:
    """The release file beside the tree, or an empty record when there is none or it is unreadable."""
    try:
        record = json.loads((dest / HARNESS_RELEASE_FILE).read_text(encoding="utf-8"))
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


class EventLog(EventWriter):
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


class BuiltinToolEnforcer(Enforcer):
    """A built-in tool runs in process whatever ``REEF_NATIVE_ENFORCE`` says: it is reef's code, not the tree's."""

    def __init__(self, inner: Enforcer) -> None:
        self._inner = inner
        self._local = InProcessEnforcer()

    @property
    def mode(self) -> str:
        return self._inner.mode

    @staticmethod
    def _is_builtin_tool(tool: Tool | None) -> bool:
        return bool(getattr(tool, "builtin_tool", False))

    def describe(self, tool: Tool | None) -> dict[str, Any]:
        return (self._local if self._is_builtin_tool(tool) else self._inner).describe(tool)

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any:
        return (self._local if self._is_builtin_tool(tool) else self._inner).run(tool, arguments, workdir)


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
    """One queued release mount and the answer whoever queued it waits for; the manifest is fetched when it lands."""

    def __init__(self, release_id: str, source: str, manifest: Mapping[str, Any] | None = None) -> None:
        self.manifest: dict[str, Any] | None = None if manifest is None else dict(manifest)
        self.source = source
        self.release_id = release_id
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


class Server(ServeState, ReleaseUpdateListener):
    """The resident process: the mounted host, the sessions, the proxy, the release watch and the socket."""

    @property
    def client(self) -> ReleaseClient:
        return self._client

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
        release_timeout_s: float = 30.0,
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
        self.mounts_dir = self.native / MOUNTS_DIR
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
        self._client = ReleaseClient(self.reef_url, self.token, scenario, timeout_s=release_timeout_s)
        #: The served release's source tree, and the tree the current host runs: the served one, or a trial's.
        self._tree = HarnessTree({})
        self._live_tree = self._tree
        self._release_id: str | None = None
        self._parent_release_id: str | None = None
        self._pending: _Pending | None = None
        self._trial: str | None = None
        self._builtin_tools: list[HostTool] = []
        self._mount_count = 0
        self._mount_dir: Path | None = None
        self._conversations: dict[str, _Conversation] = {}
        self._current: _Conversation | None = None
        self._deadline = 0.0
        self._turn_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopped = threading.Event()
        self._started = False
        self._log: EventLog
        self._host: NativeHost
        self._proxy: CaptureProxy
        self._watch: HeadWatch
        self._enforcer: BuiltinToolEnforcer
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

    def live_tree(self) -> HarnessTree:
        """The tree the current host runs: the served one, or the trial's for the rest of a turn."""
        return self._live_tree

    def served_tree(self) -> HarnessTree:
        with self._state_lock:
            return self._tree

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "release_id": self._release_id,
                "parent_release_id": self._parent_release_id,
                "follow": self.follow,
                "files": len(self._tree.files),
                "pending_mount": None if self._pending is None else self._pending.release_id,
                "sessions": len(self._conversations),
                "socket": str(self.socket_path),
                "self_tools": self.self_tools,
            }

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Boot: the proxy, the boot tree as a mount with source ``boot``, the release watch, the socket."""
        self._refuse_second_process()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._log = EventLog(self.sessions_dir / SERVE_LOG)
        self._started = True
        try:
            self._enforcer = BuiltinToolEnforcer(select_enforcer(os.environ))
        except ValueError as exc:
            raise ServeError(str(exc)) from exc
        release_info = read_release_info(self.dest)
        release = str(release_info["release_id"]) if release_info.get("release_id") else None
        parent = str(release_info["parent_release_id"]) if release_info.get("parent_release_id") else None
        stored = read_source(self.dest) if release is not None else None
        manifest: Mapping[str, Any] | None = None
        if stored is None:
            # Nothing mounted here yet: the head's source tree, which the boot mount stores for the next start.
            try:
                manifest = self.client.fetch(release)
            except ReleaseClientError as exc:
                raise ServeError(
                    f"no source tree under {self.dest / SOURCE_DIR} and Reef answered none: {exc}"
                ) from exc
            stored = tree_of(manifest)
            release = str(manifest["release_id"])
            parent = None if manifest.get("parent_release_id") is None else str(manifest["parent_release_id"])
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
        except CaptureError as exc:
            raise ServeError(str(exc)) from exc
        self.binding = replace(self._installed_binding, base_url=f"http://127.0.0.1:{self._proxy.port}")
        # A crashed process leaves its mounts behind; none of them is current.
        shutil.rmtree(self.mounts_dir, ignore_errors=True)
        if self.self_tools:
            self._builtin_tools = self_tools(self)
        failure = self._mount(stored, release, parent, source="boot", manifest=manifest)
        if failure is not None:
            self.stop()
            raise ServeError(f"release {release} does not mount: {failure['error']}")
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
        with self._turn_lock:
            shutil.rmtree(self.mounts_dir, ignore_errors=True)
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
        """A manual mount: queue the release and answer once it landed or failed."""
        pending = self._enqueue(release_id, "release")
        self._drain_if_idle()
        return pending.wait(self.turn_timeout_s + 5.0)

    # -- the head ----------------------------------------------------------------------------------------------

    def new_head(self, release_id: str, source: str) -> None:
        """A head the watch learned of: queued for a mount under ``head``, announced and left alone under ``pinned``."""
        if self.follow == "pinned":
            self._log.write("release/available", {"release_id": release_id})
            return
        # Only the id is queued here: the caller may be the proxy's relay thread, and the manifest fetch belongs
        # to the thread that mounts, so an inference answer is never held behind a round trip to reef.
        self._enqueue(release_id, "release")
        self._drain_if_idle()

    def _enqueue(self, release_id: str, source: str) -> _Pending:
        pending = _Pending(release_id, source)
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
        """Land the newest queued mount; the caller holds the turn lock. A trial keeps its tree until the turn ends."""
        while self._trial is None:
            with self._state_lock:
                pending, self._pending = self._pending, None
            if pending is None:
                return
            pending.settle(self._mount_manifest(pending))

    def _mount_manifest(self, pending: _Pending) -> dict[str, Any]:
        try:
            manifest = pending.manifest if pending.manifest is not None else self.client.fetch(pending.release_id)
            tree = tree_of(manifest)
        except (ServeError, ReleaseClientError) as exc:
            self._log.write(
                "harness/mount-failed", {"release_id": pending.release_id, "source": pending.source, "error": str(exc)}
            )
            if isinstance(exc, ReleaseClientError) and exc.timed_out:
                # Reef is busy (an evolve step holds the manifest as it holds the catalog), not gone: the next poll
                # that names this head announces it again, so the mount is retried at the interval, not forgotten.
                self._watch.forget(pending.release_id)
            return {"mounted": False, "release_id": pending.release_id, "error": str(exc)}
        parent = manifest.get("parent_release_id")
        failure = self._mount(
            tree, pending.release_id, None if parent is None else str(parent), source=pending.source, manifest=manifest
        )
        if failure is not None:
            return {"mounted": False, "release_id": pending.release_id, "error": failure["error"]}
        return {"mounted": True, "release_id": pending.release_id, "error": None}

    # -- mounts ------------------------------------------------------------------------------------------------

    def _mount(
        self,
        tree: HarnessTree,
        release_id: str | None,
        parent_release_id: str | None,
        *,
        source: str,
        manifest: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make ``tree`` the current host's tree; None on success, else the failure, with the current host kept.

        ``source`` is ``boot`` or ``release`` for a served release, ``try`` for
        a trial and ``rollback`` for the served tree back after one. A served
        release with its ``manifest`` is stored for the next start.
        """
        try:
            host, directory = self._build_host(tree)
        except (RenderError, LoadError, GraphError) as exc:
            self._log.write("harness/mount-failed", {"release_id": release_id, "source": source, "error": str(exc)})
            return {"error": str(exc)}
        previous, self._host, self._mount_dir, self._live_tree = self._mount_dir, host, directory, tree
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        if source in ("boot", "release"):
            with self._state_lock:
                self._tree = tree
                self._release_id = release_id
                self._parent_release_id = parent_release_id
            if release_id:
                self._proxy.tags["release"] = release_id
            else:
                self._proxy.tags.pop("release", None)
            self._watch.mounted(release_id)
            if manifest is not None:
                self._install(tree, manifest)
        self._log.write(
            "harness/mount",
            {
                "release_id": release_id,
                "parent_release_id": parent_release_id,
                "source": source,
                "files": len(tree.files),
                **(extra or {}),
            },
        )
        return None

    def _build_host(self, tree: HarnessTree) -> tuple[NativeHost, Path]:
        """``tree`` rendered for the native adapter into a fresh mount directory, and the host its files make.

        The built-in tools join every host; a tree file that takes one of their
        names fails the load. A failure removes the directory it wrote.
        """
        files = render_tree(tree, self.descriptor)
        self._mount_count += 1
        directory = self.mounts_dir / str(self._mount_count)
        shutil.rmtree(directory, ignore_errors=True)
        try:
            for relative, text in files.items():
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            host = NativeHost.from_root(directory / "native")
            for tool in self._builtin_tools:
                host.add_tool(tool)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return host, directory

    def _install(self, tree: HarnessTree, manifest: Mapping[str, Any]) -> None:
        """What a restart boots from: the release's source tree under ``source/`` and the release file naming it."""
        staging = self.dest / f".{SOURCE_DIR}.part"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        tree.write(staging)
        shutil.rmtree(self.dest / SOURCE_DIR, ignore_errors=True)
        staging.rename(self.dest / SOURCE_DIR)
        record = {
            "release_id": manifest.get("release_id"),
            "content_id": manifest.get("content_id"),
            "parent_release_id": manifest.get("parent_release_id"),
            "installed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (self.dest / HARNESS_RELEASE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def try_patch(self, patch: TreePatch, try_id: str) -> dict[str, Any]:
        """Mount the served tree with ``patch`` applied for the rest of the open turn; the release file stays."""
        if self._current is None:
            return {"error": "no turn is open"}
        if self._trial is not None:
            self._unmount_trial()  # one trial at a time: a second try replaces the first
        served = self.served_tree()
        validation = validate_patch(served, patch, self.descriptor)
        if not validation.passed:
            return {"error": validation.message()}
        failure = self._mount(
            served.apply(patch),
            self._release_id,
            self._parent_release_id,
            source="try",
            extra={"try_id": try_id, "paths": sorted(patch.changes)},
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
        self._mount(self.served_tree(), self._release_id, self._parent_release_id, source="rollback")
        self._proxy.tags.pop("trial", None)
        self._log.write("harness/unmount", {"try_id": try_id, "release_id": self._release_id})

    # -- turns -------------------------------------------------------------------------------------------------

    def between_steps(self, run: Run) -> None:
        """At the top of every model stage: the wall clock ends the turn, then a queued mount lands on ``run``."""
        if time.monotonic() > self._deadline:
            run.end_turn({"kind": "turn-timeout", "seconds": self.turn_timeout_s}, "budget")
        self._drain_pending()
        run.host = self._host

    def _conversation(self, session_id: str | None) -> _Conversation:
        with self._state_lock:
            if session_id is None:
                session_id = secrets.token_hex(6)
                while session_id in self._conversations:
                    session_id = secrets.token_hex(6)
            conversation = self._conversations.get(session_id)
            if conversation is None:
                if (self.sessions_dir / session_id / "session.jsonl").exists():
                    # The history is on disk but not in this process; resuming as turn 1 would hide that.
                    raise ServeError(
                        f"session {session_id!r} predates this process and cannot be continued; start a new session"
                    )
                conversation = self._conversations[session_id] = _Conversation(
                    session_id, self.sessions_dir / session_id
                )
            return conversation

    def turn(
        self, prompt: str, session_id: str | None, workdir: Path, sink: EventSink | None = None
    ) -> dict[str, Any]:
        """One turn on one session; the events go to ``sink`` as written and the result comes back.

        The result holds the exit status, the session, the turn number, the
        final text, and ``receipts``: the receipts of the model calls the turn
        made, in order, which a report references.
        """
        if session_id is not None and not _SESSION_NAME.fullmatch(session_id):
            raise ServeError(f"session id {session_id!r} must match {_SESSION_NAME.pattern}")
        with self._turn_lock:
            conversation = self._conversation(session_id)
            session = ServeSession(conversation.dir / "session.jsonl", sink)
            self._current = conversation
            self._log.bind(session)
            self._deadline = time.monotonic() + self.turn_timeout_s
            run: Run | None = None
            try:
                run = self._begin_turn(conversation, prompt, workdir, session)
                # Read at each turn start: a mount between turns lands on the next one, and a mount inside the turn
                # (landed by before_step) changes the host under a loop that keeps this module object until the end.
                module = self._host.loop
                exit_code = (
                    run_graph(run, self._host.graph("main")) if module is None else run_loop_module(run, module)
                )
                result = {
                    "exit": exit_code,
                    "session": conversation.id,
                    "turn": run.turn,
                    "text": _last_text(run.messages),
                }
            except Exception as exc:
                # The host changes under a graph pinned at turn start (a mount that dropped an agent the graph
                # names), so the turn ends in the log with the failure instead of the connection dying on it.
                failure = {"code": "TURN_ERROR", "message": f"{type(exc).__name__}: {exc}"[:600]}
                turn_number = run.turn if run is not None else 1
                session.write("turn/end", {"turn": turn_number, "reason": {"kind": "error", "error": failure}})
                result = {"exit": 1, "session": conversation.id, "turn": turn_number, "text": ""}
            finally:
                self._unmount_trial()
                if conversation.loop is not None:
                    conversation.loop.open.clear()
                self._log.bind(None)
                self._current = None
                session.close()
                self._drain_pending()
            # The receipts of the model calls this turn made, so its caller reports feedback against them.
            result["receipts"] = [capture["receipt"] for capture in self._proxy.drain() if capture.get("receipt")]
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
            tools, hooks, module = run.tools, run.hooks, self._host.loop
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
                    "loop": None if module is None else module.name,
                    "agents": sorted(self._host.agents),
                },
            )
            conversation.loop, conversation.run = loop, run
        else:
            loop.session = session
            run.session = session
            # A mount between turns built a new host; the held run reads it from this turn on.
            run.host = self._host
            run.turn += 1
            run.prompt = prompt
            run.workdir = workdir
            run.messages.append({"role": "user", "content": prompt})
            run.step, run.tool_calls, run.tool_errors, run.step_open, run.last = 0, 0, 0, False, {}
        # Steps restart at 1 each turn, so each turn gets its own directory for complete tool outputs.
        loop.TOOL_OUTPUT_DIR = f"{TOOL_OUTPUT_DIR}/t{run.turn}"
        session.write("turn/start", {"turn": run.turn, "prompt": prompt, "cwd": str(workdir)})
        return run


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "data": {"message": message}}


# -- the socket --------------------------------------------------------------------------------------------


class _ConnectionSink(EventSink):
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


class _Stdout(EventSink):
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
    "MOUNTS_DIR",
    "SERVE_LOG",
    "SOCKET_NAME",
    "SOURCE_DIR",
    "BuiltinToolEnforcer",
    "EventLog",
    "EventSink",
    "ServeError",
    "ServeSession",
    "Server",
    "read_release_info",
    "read_source",
    "request",
    "run_command",
    "socket_path",
    "tree_layout",
    "tree_of",
]
