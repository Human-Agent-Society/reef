"""The serve process's view of Reef's harness artifact channel: the poll, the manifest read, and the head watch.

``ArtifactClient`` reads the artifact catalog and one artifact's source tree.
``HeadWatch`` learns the head from the periodic poll, and notifies its listener once per new head; a failure is
logged and retried with backoff, and the process keeps serving.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

#: The poll backoff never waits longer than this between attempts.
MAX_BACKOFF_S = 600.0


class ArtifactClientError(Exception):
    """Reef did not answer a snapshot request; ``status`` carries the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None, timed_out: bool = False) -> None:
        super().__init__(message)
        self.status = status
        #: Reef was reached but answered nothing within the client timeout: busy, not down.
        self.timed_out = timed_out


def _timed_out(exc: BaseException) -> bool:
    """Whether a request failure is a timeout: the socket's own, or the one urllib wraps as a URLError reason."""
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)


class ArtifactUpdateListener(ABC):
    """What learns of a new head: the serve process, which mounts it or announces it."""

    @abstractmethod
    def new_head(self, artifact_id: str, source: str) -> None: ...


class EventWriter(ABC):
    """Where the watch logs: the serve process's event log."""

    @abstractmethod
    def write(self, type_: str, data: Mapping[str, Any]) -> None: ...


class ArtifactClient:
    """The snapshot routes of one scenario: the catalog and one manifest."""

    def __init__(self, reef_url: str, token: str | None, scenario: str, *, timeout_s: float = 30.0) -> None:
        self.reef_url = reef_url.rstrip("/")
        self.scenario = scenario
        self._token = token
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        headers = {"x-reef-scenario": self.scenario, "accept": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        request = urllib.request.Request(f"{self.reef_url}{path}", data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return int(response.status), json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:600]
            raise ArtifactClientError(f"{method} {path} answered {exc.code}: {detail}", status=exc.code) from exc
        except (OSError, ValueError) as exc:
            raise ArtifactClientError(
                f"{method} {path} failed: {type(exc).__name__}: {exc}", timed_out=_timed_out(exc)
            ) from exc

    def artifacts(self) -> list[dict[str, Any]]:
        """The catalog rows, oldest first, as ``GET /reef/harness/artifacts`` lists them."""
        _, catalog = self._request("GET", "/reef/harness/artifacts")
        rows = catalog.get("artifacts") if isinstance(catalog, Mapping) else None
        if not isinstance(rows, list):
            raise ArtifactClientError("the artifact catalog carries no artifacts list")
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def poll(self) -> str | None:
        """The active selection: the ``artifact_id`` of the catalog row marked ``active``, or None while no row is
        active."""
        artifact_id = next((row.get("artifact_id") for row in self.artifacts() if row.get("active")), None)
        return str(artifact_id) if artifact_id else None

    def fetch(self, artifact_id: str | None = None) -> dict[str, Any]:
        """Read the source manifest for one artifact, or the active harness when no ID is given."""
        query = {"form": "source"}
        if artifact_id is not None:
            query["artifact_id"] = artifact_id
        return self.manifest(query)

    def manifest(self, query: Mapping[str, str]) -> dict[str, Any]:
        _, manifest = self._request("GET", f"/reef/harness?{urllib.parse.urlencode(query)}")
        if not isinstance(manifest, Mapping) or "files" not in manifest or not manifest.get("artifact_id"):
            raise ArtifactClientError("the harness answered no artifact manifest")
        return {key: value for key, value in manifest.items() if key != "content_id"}


class HeadWatch:
    """Compares the active artifact of each poll with the mounted one and notifies the listener of a new head once.

    The poll runs on its own thread every ``interval_s``."""

    def __init__(
        self,
        client: ArtifactClient,
        listener: ArtifactUpdateListener,
        log: EventWriter,
        interval_s: float,
        mounted: str | None,
    ) -> None:
        self._client = client
        self._listener = listener
        self._log = log
        self._interval_s = interval_s
        self._mounted = mounted
        self._announced: str | None = None
        self._failure = ""
        self._failure_timed_out = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def mounted(self, artifact_id: str | None) -> None:
        with self._lock:
            self._mounted = artifact_id

    def forget(self, artifact_id: str) -> None:
        """A head whose mount could not even fetch its manifest: the next poll that names it announces it again."""
        with self._lock:
            if self._announced == artifact_id:
                self._announced = None

    def _consider(self, artifact_id: str | None, source: str) -> None:
        with self._lock:
            if not artifact_id:
                return
            # Announced once per head: a mount that failed, or one the operator moved off, is not redone every
            # poll; a head that runs here is announced too, so moving off it later is not news either.
            if artifact_id == self._announced:
                return
            self._announced = artifact_id
            if artifact_id == self._mounted:
                return
        try:
            # Outside the lock: the listener may mount at once, and a mount reports back through ``mounted``.
            self._listener.new_head(artifact_id, source)
        except BaseException:
            with self._lock:
                if self._announced == artifact_id:
                    self._announced = None
            raise

    def poll_once(self) -> bool:
        """One poll and its consequence; False when Reef did not answer."""
        try:
            self._consider(self._client.poll(), "poll")
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"[:600]
            self._failure_timed_out = isinstance(exc, ArtifactClientError) and exc.timed_out
            return False
        return True

    def _run(self) -> None:
        delay = 0.0
        backoff = self._interval_s
        while not self._stop.wait(delay):
            if self.poll_once():
                backoff = delay = self._interval_s
                continue
            if self._failure_timed_out:
                # Reef answered nothing in time but is there: a catalog read waits behind a running evolve
                # step. The next poll goes at the interval, so the head lands as soon as the step ends.
                backoff = delay = self._interval_s
            else:
                delay = backoff
                backoff = min(backoff * 2, MAX_BACKOFF_S)
            self._log.write("artifact/poll-failed", {"error": self._failure, "retry_in_s": delay})

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="reef-native-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = [
    "MAX_BACKOFF_S",
    "ArtifactClient",
    "ArtifactClientError",
    "ArtifactUpdateListener",
    "EventWriter",
    "HeadWatch",
]
