"""The capture proxy between a harness and Reef: forward model calls, tag them, keep their receipts.

A harness that runs outside Reef's evaluation, such as a Harbor task run
or the native serve process, reaches Reef through this proxy. It forwards to
Reef with ``x-reef-scenario`` and the token, tags every call as
``x-reef-tag-<name>``, and keeps the receipt each answer carries. The harness watch polls its own catalog for
updates.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from typing import Any

from reef_client.serve import CapturedTurn, CaptureStore, ServeConfig, build_handler

from reef.core.records_types import ROLLOUT_HEADER


class CaptureError(Exception):
    """The capture proxy could not start or is not running."""


def wait_for_proxy(port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_captures", timeout=0.5)
            return True
        except OSError:
            time.sleep(0.05)
    return False


#: The paths Reef serves inference on; a receipt rides on each. The proxy matches the path with its query, and
#: the Anthropic SDK posts /v1/messages?beta=true under beta headers, so that form is listed.
CAPTURE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages?beta=true")


class ProxyTags(MutableMapping[str, str]):
    """The proxy's tag channel: each name rides every forwarded call as ``x-reef-tag-<name>``."""

    def __init__(self, config: ServeConfig, fixed: Mapping[str, str]) -> None:
        self.config = config
        self.fixed = dict(fixed)
        self.tag_values: dict[str, str] = {}
        self.publish()

    def publish(self) -> None:
        # A fresh dict per change: a handler reads config.override_headers whole per request, never a torn one.
        tagged = {f"x-reef-tag-{name}": value for name, value in self.tag_values.items()}
        self.config.override_headers = {**self.fixed, **tagged}

    def __getitem__(self, name: str) -> str:
        return self.tag_values[name]

    def __setitem__(self, name: str, value: str) -> None:
        self.tag_values[name] = value
        self.publish()

    def __delitem__(self, name: str) -> None:
        self.tag_values.pop(name)
        self.publish()

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self.tag_values))

    def __len__(self) -> int:
        return len(self.tag_values)


class CaptureObserver(ABC):
    """Receives each call that a capture proxy captures, as it lands.

    Purpose: a caller that keeps the calls while the rollout runs gives one to ``CaptureProxy``. The proxy calls
    ``call_captured`` once for each captured call, on the thread that answered the call, after the call is in the
    proxy's store. Calls that end at the same time reach it from several threads at once.
    """

    @abstractmethod
    def call_captured(self, turn: CapturedTurn) -> None:
        """Take ``turn``, one captured call. An error that this raises reaches the proxy's request handler."""


class TaggedStore(CaptureStore):
    """The capture store plus, per capture, the tags in force when it landed, so a trial's receipts are told apart.
    ``observer``, when set, gets each capture after it landed."""

    def __init__(self, tags: Mapping[str, str], observer: CaptureObserver | None = None) -> None:
        super().__init__()
        self.tag_values = tags
        self.observer = observer
        self.tagged: list[dict[str, Any]] = []
        self.guard = threading.Lock()

    def add(self, turn: CapturedTurn) -> None:
        with self.guard:
            super().add(turn)
            self.tagged.append({**asdict(turn), "tags": dict(self.tag_values)})
        if self.observer is not None:
            self.observer.call_captured(turn)

    def drain(self) -> list[dict[str, Any]]:
        """The captures since the last drain, with their tags; what one publish carries."""
        with self.guard:
            turns, self.tagged = self.tagged, []
            return turns

    def clear(self) -> int:
        # An agent that reported its receipts clears the proxy; the list a publish drains must go with them.
        with self.guard:
            self.tagged = []
            return super().clear()


class CaptureProxyServer(ThreadingHTTPServer):
    """The proxy's HTTP server, quiet when a client resets a connection.

    An agent that drops a kept alive connection with a reset, as hermes does
    on every run, ends that connection normally, so no traceback reaches the
    person's terminal; every other error still prints one."""

    def handle_error(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: tuple[str, int]
    ) -> None:
        if isinstance(sys.exc_info()[1], ConnectionResetError):
            return
        super().handle_error(request, client_address)


class CaptureProxy:
    """The capture proxy between an agent and Reef, in process.

    Forwards to ``upstream`` with ``x-reef-scenario`` and the token, tags every
    call with ``tags`` as ``x-reef-tag-<name>`` headers, and keeps the receipts.

    With ``is_unrecorded``, every call carries ``ROLLOUT_HEADER`` and is sent without streaming: Reef keeps no
    record of it, and the captured response holds the whole training block. A streaming agent still gets a
    stream, which the proxy makes from the complete answer.

    ``observer``, when set, gets each captured call as it lands (``CaptureObserver``)."""

    def __init__(
        self,
        upstream: str,
        scenario: str,
        token: str | None = None,
        *,
        tags: Mapping[str, str] | None = None,
        listen_host: str = "127.0.0.1",
        is_unrecorded: bool = False,
        observer: CaptureObserver | None = None,
    ) -> None:
        self.upstream = upstream
        self.scenario = scenario
        self.listen_host = listen_host
        fixed: dict[str, str] = {"x-reef-scenario": scenario}
        if token:
            fixed["authorization"] = f"Bearer {token}"
        if is_unrecorded:
            fixed[ROLLOUT_HEADER] = "1"
        self.config = ServeConfig(
            upstream=upstream, listen_port=0, capture_paths=CAPTURE_PATHS, force_non_stream=is_unrecorded
        )
        #: The tag channel: the record keeps which snapshot (and which trial) answered, under ``metadata.tags``.
        self.tags: MutableMapping[str, str] = ProxyTags(self.config, fixed)
        for name, value in (tags or {}).items():
            self.tags[name] = value
        self.store = TaggedStore(self.tags, observer)
        self.handler = build_handler(self.config, self.store)
        self.server: CaptureProxyServer | None = None

    @property
    def port(self) -> int:
        if self.server is None:
            raise CaptureError("the capture proxy is not running")
        return int(self.server.server_address[1])

    def start(self) -> None:
        server = CaptureProxyServer((self.listen_host, 0), self.handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.server = server
        if not wait_for_proxy(self.port):
            self.stop()
            raise CaptureError("capture proxy failed to start")

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def drain(self) -> list[dict[str, Any]]:
        """The captures since the last drain or publish, each with the tags in force when it landed."""
        return self.store.drain()


__all__ = ["CAPTURE_PATHS", "CaptureError", "CaptureObserver", "CaptureProxy"]
