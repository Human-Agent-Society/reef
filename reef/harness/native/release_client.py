"""The serve process's view of Reef's release channel: the poll, the manifest read, and the head watch.

``ReleaseClient`` is the Python form of ``version_check.ts``: it reads the
release catalog and one manifest by id. ``HeadWatch`` learns the head from
the periodic poll and from the ``x-reef-release-id`` header the capture proxy
reads on every inference answer, and tells its sink once per new head; a
failure is logged and retried with backoff, and the process keeps serving.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol

#: The poll backoff never waits longer than this between attempts.
MAX_BACKOFF_S = 600.0


class ReleaseClientError(Exception):
    """Reef did not answer a release request; ``status`` carries the HTTP status when there was one."""

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


class HeadSink(Protocol):
    """What learns of a new head: the serve process, which mounts it or announces it."""

    def new_head(self, release_id: str, source: str) -> None: ...


class EventWriter(Protocol):
    """Where the watch logs: the serve process's event log."""

    def write(self, type_: str, data: Mapping[str, Any]) -> None: ...


class ReleaseClient:
    """The release routes of one scenario: the catalog, one manifest, and the proposals route."""

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
            raise ReleaseClientError(f"{method} {path} answered {exc.code}: {detail}", status=exc.code) from exc
        except (OSError, ValueError) as exc:
            raise ReleaseClientError(
                f"{method} {path} failed: {type(exc).__name__}: {exc}", timed_out=_timed_out(exc)
            ) from exc

    def releases(self) -> list[dict[str, Any]]:
        """The catalog rows, oldest first, as ``GET /reef/harness/releases`` lists them."""
        _, catalog = self._request("GET", "/reef/harness/releases")
        rows = catalog.get("releases") if isinstance(catalog, Mapping) else None
        if not isinstance(rows, list):
            raise ReleaseClientError("the release catalog carries no releases list")
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def poll(self) -> str | None:
        """The head: the last catalog row that is not pending review, or None while the scenario has no release."""
        for row in reversed(self.releases()):
            if row.get("pending"):
                continue  # a release held for review is not served yet, so it is not the head
            release = row.get("release_id")
            return str(release) if release else None
        return None

    def fetch(self, release_id: str) -> dict[str, Any]:
        """One release's manifest; its ``files`` carry ``native/tree.json`` and ``gate`` the verdict."""
        _, manifest = self._request("GET", f"/reef/harness?release_id={urllib.parse.quote(release_id, safe='')}")
        if not isinstance(manifest, Mapping) or "files" not in manifest:
            raise ReleaseClientError(f"release {release_id!r} answered no manifest")
        return dict(manifest)

    def propose(self, body: Mapping[str, Any]) -> tuple[int, Any]:
        """``POST /reef/harness/proposals``; the status and the JSON answer, a 404 included."""
        try:
            return self._request("POST", "/reef/harness/proposals", body)
        except ReleaseClientError as exc:
            if exc.status == 404:
                return 404, None
            raise


class HeadWatch:
    """Compares every release id it hears of with the mounted one and tells the sink of a new head once.

    The poll runs on its own thread every ``interval_s``; a response header
    is considered on the proxy's thread at once, before the answer reaches
    the agent, so the mount is queued by the time the next step starts."""

    def __init__(
        self, client: ReleaseClient, sink: HeadSink, log: EventWriter, interval_s: float, mounted: str | None
    ) -> None:
        self._client = client
        self._sink = sink
        self._log = log
        self._interval_s = interval_s
        self._mounted = mounted
        self._announced: str | None = None
        self._failure = ""
        self._failure_timed_out = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def mounted(self, release_id: str | None) -> None:
        with self._lock:
            self._mounted = release_id

    def _consider(self, release_id: str | None, source: str) -> None:
        with self._lock:
            if not release_id:
                return
            # Announced once per head: a mount that failed, or one the operator moved off, is not redone every
            # poll; a head that runs here is announced too, so moving off it later is not news either.
            if release_id == self._announced:
                return
            self._announced = release_id
            if release_id == self._mounted:
                return
        try:
            # Outside the lock: the sink may mount at once, and a mount reports back through ``mounted``.
            self._sink.new_head(release_id, source)
        except BaseException:
            with self._lock:
                if self._announced == release_id:
                    self._announced = None
            raise

    def observe(self, release_id: str) -> None:
        """The head an inference answer named; a failure here is logged like a poll failure."""
        try:
            self._consider(release_id, "response")
        except Exception as exc:
            self._log.write("release/poll-failed", {"error": f"{type(exc).__name__}: {exc}"[:600], "retry_in_s": 0})

    def poll_once(self) -> bool:
        """One poll and its consequence; False when Reef did not answer."""
        try:
            self._consider(self._client.poll(), "poll")
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"[:600]
            self._failure_timed_out = isinstance(exc, ReleaseClientError) and exc.timed_out
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
            self._log.write("release/poll-failed", {"error": self._failure, "retry_in_s": delay})

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="reef-native-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = ["MAX_BACKOFF_S", "EventWriter", "HeadSink", "HeadWatch", "ReleaseClient", "ReleaseClientError"]
