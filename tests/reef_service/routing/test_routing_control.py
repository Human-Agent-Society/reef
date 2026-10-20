"""Private lifecycle calls reach the REEF child only over its Unix socket."""

import contextlib
import json
import math
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingUnixStreamServer

import pytest

from reef.service.deploy.routing.routing_control import (
    ReefLifecycleClient,
    RoutingControlError,
    RoutingControlTimeout,
    UnixHTTPConnection,
    json_object,
)

pytestmark = pytest.mark.unit


class UnixHTTPServer(ThreadingUnixStreamServer):
    """A threaded HTTP server on a Unix socket whose handler threads never block process exit."""

    daemon_threads = True


class SlowChild:
    """A lifecycle endpoint that applies each revision once and answers request ``n`` after ``delays[n]`` seconds.

    A request after the last delay is answered at once. ``bodies`` lists every request body in arrival
    order, repeats included, and ``applied`` each revision when it was first seen. ``reply`` is False to close
    each connection without any answer; ``reply_body`` replaces the JSON status, so malformed replies stay
    possible. With ``headers_first`` the status line and headers go out at once and only the body waits;
    ``closed`` then records, for each waiting request, whether the client closed its connection meanwhile.
    ``protocol`` is the HTTP version it answers with. ``handler`` makes a request handler class bound to
    this child; ``serving`` serves it on a temporary Unix socket and shuts it down.
    """

    def __init__(
        self,
        delays: list[float],
        *,
        reply: bool = True,
        reply_body: bytes | None = None,
        protocol: str = "HTTP/1.0",
        headers_first: bool = False,
    ) -> None:
        self.delays = delays
        self.reply = reply
        self.reply_body = reply_body
        self.protocol = protocol
        self.headers_first = headers_first
        self.bodies: list[bytes] = []
        self.applied: list[int] = []
        self.closed: list[bool] = []
        self.revision = -1
        self.lock = threading.Lock()

    def handler(self) -> type[BaseHTTPRequestHandler]:
        child = self

        class Handler(BaseHTTPRequestHandler):
            """One connection to this child: it records the request in ``child`` and answers as configured."""

            protocol_version = child.protocol

            def do_GET(self) -> None:
                self.answer(b"")

            def do_POST(self) -> None:
                self.answer(self.rfile.read(int(self.headers["Content-Length"])))

            def answer(self, body: bytes) -> None:
                with child.lock:
                    index = len(child.bodies)
                    child.bodies.append(body)
                    revision = json.loads(body)["control_revision"] if body else child.revision
                    if revision > child.revision:
                        child.revision = revision
                        child.applied.append(revision)
                    reply = child.reply_body or json.dumps({"control_revision": child.revision}).encode()
                delay = child.delays[index] if index < len(child.delays) else 0
                if not child.headers_first:
                    time.sleep(delay)
                if not child.reply:
                    return
                # The client may have stopped waiting and closed the connection.
                with contextlib.suppress(OSError):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    if child.headers_first and delay:
                        self.connection.settimeout(delay)
                        try:
                            closed = self.connection.recv(1) == b""
                        except TimeoutError:
                            closed = False
                        child.closed.append(closed)
                        if closed:
                            return
                    self.wfile.write(reply)

            def finish(self) -> None:
                with contextlib.suppress(OSError):
                    super().finish()

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Handler


@contextlib.contextmanager
def serving(child: SlowChild) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="reef-ctl-") as directory:
        path = Path(directory) / "control.sock"
        server = UnixHTTPServer(str(path), child.handler())
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield path
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


def client_for(path: Path, *, open_timeout_seconds: float = 30.0) -> ReefLifecycleClient:
    return ReefLifecycleClient(
        path,
        "test",
        "generation",
        "private-test-token",
        timeout_seconds=0.2,
        open_timeout_seconds=open_timeout_seconds,
    )


@pytest.mark.parametrize("action", [None, "close"])
def test_a_reply_that_times_out_is_requested_again_with_the_same_bytes_and_applied_once(action: str | None) -> None:
    child = SlowChild([0.6])
    with serving(child) as path:
        client = client_for(path)
        status = client.request(action)
    # The first attempt's reply came too late; the second, on a new connection, read the same revision.
    expected_revision = -1 if action is None else 1
    assert status == {"control_revision": expected_revision}
    assert len(child.bodies) == 2 and child.bodies[0] == child.bodies[1]
    assert child.applied == ([] if action is None else [1])


def test_persistent_timeouts_end_after_three_attempts_of_the_ordinary_allowance() -> None:
    child = SlowChild([1.0, 1.0, 1.0])
    with serving(child) as path:
        client = client_for(path)
        started = time.monotonic()
        with pytest.raises(RoutingControlError, match="timed out") as error:
            client.request("close")
        elapsed = time.monotonic() - started
    assert error.value.retryable
    assert len(child.bodies) == 3 and len(set(child.bodies)) == 1 and child.applied == [1]
    assert 0.6 <= elapsed < 0.9


@pytest.mark.parametrize(
    ("reply", "failure"),
    [({"reply": False}, "request failed"), ({"reply_body": b"not json"}, "not valid JSON")],
    ids=["no_reply", "invalid_json"],
)
def test_only_a_timeout_is_requested_again(reply: dict[str, bool | bytes], failure: str) -> None:
    """HTTP statuses other than 200 are checked, also with a single request, in the socket test below."""
    child = SlowChild([], **reply)
    with serving(child) as path, pytest.raises(RoutingControlError, match=failure) as error:
        client_for(path).request("close")
    assert not isinstance(error.value, RoutingControlTimeout)
    assert len(child.bodies) == 1


@pytest.mark.parametrize("protocol", ["HTTP/1.0", "HTTP/1.1"])
def test_each_attempt_whose_body_times_out_closes_its_reply_and_the_retries_end_at_the_callers_deadline(
    protocol: str,
) -> None:
    """Headers arrive at once but no body does. An HTTP/1.0 reply holds the connection's socket by itself."""
    child = SlowChild([3.0, 3.0, 3.0], protocol=protocol, headers_first=True)
    with serving(child) as path:
        started = time.monotonic()
        with pytest.raises(RoutingControlTimeout) as error:
            client_for(path).request("close", deadline=started + 0.5)
        elapsed = time.monotonic() - started
        # The error held here keeps the client's frames alive; no reply socket may stay open with it.
        waited_until = time.monotonic() + 3
        while len(child.closed) < 3 and time.monotonic() < waited_until:
            time.sleep(0.01)
        assert error.value.retryable
    assert child.closed == [True, True, True]
    assert len(set(child.bodies)) == 1 and child.applied == [1]
    # Two ordinary 0.2-second attempts, then the 0.1 seconds left in the phase.
    assert 0.5 <= elapsed < 0.65


def test_a_callers_deadline_ends_the_attempts_and_is_the_opens_deadline() -> None:
    child = SlowChild([1.0, 1.0, 1.0])
    with serving(child) as path:
        client = client_for(path, open_timeout_seconds=5)
        started = time.monotonic()
        with pytest.raises(RoutingControlError, match="timed out"):
            client.request("open", deadline=started + 0.3)
        elapsed = time.monotonic() - started
        # Without time left in the phase, nothing is sent.
        with pytest.raises(RoutingControlError, match="timed out"):
            client.request(deadline=time.monotonic())
    assert 0.3 <= elapsed < 0.45
    # The open's deadline is the caller's deadline, which ended its attempt.
    assert child.bodies and len(set(child.bodies)) == 1
    assert json.loads(child.bodies[0])["open_deadline_monotonic"] == started + 0.3


def test_a_reply_within_the_callers_deadline_is_accepted() -> None:
    child = SlowChild([0.05])
    with serving(child) as path:
        assert client_for(path).request("close", deadline=time.monotonic() + 1) == {"control_revision": 1}


def test_an_open_carries_the_end_of_its_first_attempt_and_every_retry_reuses_it() -> None:
    child = SlowChild([0.6])
    with serving(child) as path:
        client = client_for(path, open_timeout_seconds=0.3)
        started = time.monotonic()
        assert client.request("open") == {"control_revision": 1}
    first, retry = child.bodies
    assert first == retry
    deadline = json.loads(first)["open_deadline_monotonic"]
    # A retry waits at most the ordinary allowance and never moves the open's deadline.
    assert started + 0.3 <= deadline < started + 0.35
    assert child.applied == [1]


@pytest.mark.parametrize(("status", "retryable"), [(200, None), (302, False), (409, True), (503, True)])
def test_control_uses_its_socket_ignores_proxies_and_never_follows_redirects(
    monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool | None
) -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        """Records each POST in ``requests`` and answers with the parametrized ``status``, maybe a redirect."""

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, self.headers["Authorization"], body))
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/credential-sink")
            self.send_header("Content-Length", "19")
            self.end_headers()
            self.wfile.write(b'{"accepting":false}')

        def address_string(self) -> str:
            return "control-socket"

        def log_message(self, format: str, *args: object) -> None:
            pass

    with tempfile.TemporaryDirectory(prefix="reef-ctl-") as directory:
        path = Path(directory) / "control.sock"
        server = UnixHTTPServer(str(path), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        client = ReefLifecycleClient(path, "test", "generation", "private-test-token")
        try:
            if status == 200:
                assert client.request("close") == {"accepting": False}
            else:
                with pytest.raises(RoutingControlError, match=str(status)) as error:
                    client.request("close")
                assert error.value.retryable is retryable and error.value.status_code == status
            [(path_seen, authorization, body)] = requests
            assert path_seen == "/reef/lifecycle" and authorization == "Bearer private-test-token"
            assert b'"control_revision": 1' in body and b'"action": "close"' in body
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


def test_a_missing_socket_is_a_retryable_control_failure(tmp_path: Path) -> None:
    client = ReefLifecycleClient(tmp_path / "absent.sock", "test", "generation", "private-test-token")
    with pytest.raises(RoutingControlError, match="request failed") as error:
        client.request()
    assert error.value.retryable


def test_the_client_never_connects_over_tcp(tmp_path: Path) -> None:
    """A socket path that names a TCP listener's address is never dialed as TCP."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        port = listener.getsockname()[1]
        client = ReefLifecycleClient(Path(f"127.0.0.1:{port}"), "test", "generation", "private-test-token")
        with pytest.raises(RoutingControlError):
            client.request()
        listener.settimeout(0.1)
        with pytest.raises(TimeoutError):
            listener.accept()
    finally:
        listener.close()


@pytest.mark.parametrize("body", [b"invalid", b"\xff", b"[]"])
def test_invalid_control_json_is_a_typed_error(body: bytes) -> None:
    with pytest.raises(RoutingControlError):
        json_object(body)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ReefLifecycleClient(Path("control.sock"), "", "generation", "token"),
        lambda: ReefLifecycleClient(Path("control.sock"), "test", "generation", "token", revision=-1),
        lambda: ReefLifecycleClient(Path("control.sock"), "test", "generation", "token", timeout_seconds=math.nan),
        lambda: ReefLifecycleClient(Path("control.sock"), "test", "generation", "token", open_timeout_seconds=0),
        lambda: RoutingControlError("no status", status_code=42),
        lambda: RoutingControlError("no flag", retryable="yes"),
        lambda: UnixHTTPConnection(Path("control.sock"), math.inf),
    ],
)
def test_control_values_refuse_what_their_definitions_exclude(make: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        make()


def test_a_deadline_that_already_passed_is_a_timeout_not_an_invalid_connection(tmp_path: Path) -> None:
    connection = UnixHTTPConnection(tmp_path / "control.sock", time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        connection.connect()
