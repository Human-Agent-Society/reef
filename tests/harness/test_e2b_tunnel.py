"""The E2B executor's tunnel and file copies, run locally: the relay as the sandbox runs it, a pump as Reef does."""

from __future__ import annotations

import http.client
import io
import json
import socket
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reef.execution.sandbox.e2b import TRAFFIC_ACCESS_TOKEN_HEADER, E2BExecutor, E2BSession, template_alias
from reef.execution.sandbox.process import (
    ProcessOutcome,
    ProgramLaunchError,
    ProgramTimeoutError,
    SandboxUnavailableError,
)
from reef.execution.sandbox.relay import Relay
from reef.execution.sandbox.remote import RELAY_SCRIPT, TunnelPump, pack, remote_command, remote_root, unpack


@pytest.mark.unit
@pytest.mark.parametrize("disconnects", [1, 3])
def test_command_disconnect_reconnects_without_starting_the_command_again(tmp_path, monkeypatch, disconnects):
    pytest.importorskip("e2b")
    rpc = pytest.importorskip("connectrpc.errors")
    from connectrpc.code import Code

    sandbox = Mock()
    command = sandbox.commands.run.return_value
    command.pid = 42
    failure = rpc.ConnectError(Code.INTERNAL, "peer closed connection without sending TLS close_notify")
    command.wait.side_effect = failure
    resumed = sandbox.commands.connect.return_value
    resumed.pid = 42
    if disconnects == 1:
        resumed.wait.return_value = SimpleNamespace(exit_code=0, stdout="complete output", stderr="")
    else:
        resumed.wait.side_effect = failure
    session = E2BSession(sandbox)
    pushed, pulled = [], []
    monkeypatch.setattr(session, "push", pushed.append)
    monkeypatch.setattr(session, "pull", pulled.append)
    if disconnects == 1:
        outcome = session.launch(["pi"], root=tmp_path, workspace=tmp_path, env={}, timeout=60)
        assert outcome == ProcessOutcome(0, "complete output", "")
    else:
        with pytest.raises(ProgramLaunchError, match="connection"):
            session.launch(["pi"], root=tmp_path, workspace=tmp_path, env={}, timeout=60)
        sandbox.commands.kill.assert_called_once_with(42)
    sandbox.commands.run.assert_called_once()
    assert sandbox.commands.run.call_args.kwargs["background"] is True
    assert sandbox.commands.connect.call_count == min(disconnects, 2)
    for call in sandbox.commands.connect.call_args_list:
        assert call.args == (42,)
        assert 0 < call.kwargs["timeout"] < 60
    assert pushed == pulled == [tmp_path]


@pytest.mark.unit
def test_command_timeout_kills_the_process_before_copying_its_files(tmp_path, monkeypatch):
    e2b = pytest.importorskip("e2b")
    sandbox = Mock()
    command = sandbox.commands.run.return_value
    command.pid = 42
    command.wait.side_effect = e2b.TimeoutException("deadline exceeded")
    session = E2BSession(sandbox)
    monkeypatch.setattr(session, "push", lambda root: None)

    def pull(root):
        sandbox.commands.kill.assert_called_once_with(42)

    monkeypatch.setattr(session, "pull", pull)
    with pytest.raises(ProgramTimeoutError):
        session.launch(["pi"], root=tmp_path, workspace=tmp_path, env={}, timeout=60)
    sandbox.commands.connect.assert_not_called()


@pytest.mark.unit
def test_reconnecting_does_not_extend_the_command_deadline(tmp_path, monkeypatch):
    pytest.importorskip("e2b")
    rpc = pytest.importorskip("connectrpc.errors")
    from connectrpc.code import Code

    import reef.execution.sandbox.e2b as executor

    monkeypatch.setattr(executor, "time", SimpleNamespace(monotonic=Mock(side_effect=[0.0, 61.0])))
    sandbox = Mock()
    process = sandbox.commands.run.return_value
    process.pid = 42
    process.wait.side_effect = rpc.ConnectError(Code.INTERNAL, "connection interrupted")
    session = E2BSession(sandbox)
    monkeypatch.setattr(session, "push", lambda root: None)
    monkeypatch.setattr(session, "pull", lambda root: None)
    with pytest.raises(ProgramTimeoutError):
        session.launch(["pi"], root=tmp_path, workspace=tmp_path, env={}, timeout=60)
    sandbox.commands.connect.assert_not_called()
    sandbox.commands.kill.assert_called_once_with(42)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Gateway:
    """The Reef-side port the tunnel reaches: a JSON answer, a stream held open between pieces, and a request that
    waits for another to arrive, as a trial waits on the model calls it makes. ``received`` holds the headers of each
    request, with lowercase names."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.waiting = threading.Event()
        self.arrived = threading.Event()
        self.received: list[dict[str, str]] = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                gateway.received.append({name.lower(): value for name, value in self.headers.items()})
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/echo":
                    data = json.dumps({"body": json.loads(body), "auth": self.headers.get("Authorization")}).encode()
                    self.send_response(201)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for piece in (b"data: one\n\n", b"data: two\n\n"):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                        self.wfile.flush()
                        gateway.release.wait(10)  # the client reads the first piece before the second exists
                    self.wfile.write(b"0\r\n\r\n")
                elif self.path == "/wait":
                    gateway.waiting.set()
                    ok = gateway.arrived.wait(10)
                    data = b"waited" if ok else b"alone"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/arrive":
                    gateway.arrived.set()
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/broken":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"b\r\ndata: one\n\n\r\n")
                    self.wfile.flush()
                    self.close_connection = True

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def tunnel(tmp_path: Path):
    """A gateway, the relay in front of it on another port, and a pump between them."""
    gateway = Gateway()
    local_port, tunnel_port = free_port(), free_port()
    secret_file = tmp_path / "secret"
    secret_file.write_text("s3cret")
    relay = subprocess.Popen(
        [
            sys.executable,
            str(RELAY_SCRIPT),
            "--port",
            str(local_port),
            "--tunnel-port",
            str(tunnel_port),
            "--tunnel-host",
            "127.0.0.1",
            "--secret-file",
            str(secret_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert relay.stdout is not None and relay.stdout.readline().strip() == "relay ready"
    pump = TunnelPump(f"http://127.0.0.1:{tunnel_port}", "s3cret", gateway.port, pollers=3)
    pump.start(timeout=10)
    try:
        yield gateway, local_port, tunnel_port
    finally:
        gateway.release.set()
        pump.stop()
        relay.kill()
        relay.wait()
        gateway.close()


def post(url: str, body: dict, headers: dict | None = None) -> urllib.request.Request:
    return urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )


def serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """``handler`` on a free loopback port, served on a daemon thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


#: The traffic access token of a private sandbox in these tests.
TRAFFIC_TOKEN = "traffic-token"


@pytest.mark.unit
def test_a_request_inside_the_sandbox_reaches_the_reef_port_and_its_answer_streams_back(tunnel) -> None:
    gateway, local_port, _ = tunnel
    request = post(f"http://127.0.0.1:{local_port}/echo", {"x": 1}, {"Authorization": "Bearer inside"})
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 201 and response.headers["Content-Type"] == "application/json"
        assert json.loads(response.read()) == {"body": {"x": 1}, "auth": "Bearer inside"}

    # A stream arrives piece by piece: the first is read while the gateway still holds the second back.
    with urllib.request.urlopen(post(f"http://127.0.0.1:{local_port}/stream", {}), timeout=10) as response:
        assert response.readline() == b"data: one\n"
        gateway.release.set()
        assert response.read() == b"\ndata: two\n\n"

    # A request that waits on another does not hold up the tunnel: a trial waits on the model calls it makes.
    waited: list[bytes] = []
    waiter = threading.Thread(
        target=lambda: waited.append(
            urllib.request.urlopen(post(f"http://127.0.0.1:{local_port}/wait", {}), timeout=15).read()
        )
    )
    waiter.start()
    assert gateway.waiting.wait(10)
    urllib.request.urlopen(post(f"http://127.0.0.1:{local_port}/arrive", {}), timeout=10).read()
    waiter.join(15)
    assert waited == [b"waited"]


@pytest.mark.unit
def test_the_tunnel_answers_no_one_without_the_secret(tunnel) -> None:
    _, _, tunnel_port = tunnel
    for secret in (None, "wrong"):
        headers = {} if secret is None else {"x-relay-secret": secret}
        request = urllib.request.Request(f"http://127.0.0.1:{tunnel_port}/next", headers=headers)
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=5)
        assert refused.value.code == 403


@pytest.mark.unit
def test_an_interrupted_upstream_stream_is_not_reported_as_complete(tunnel) -> None:
    _, local_port, _ = tunnel
    with (
        urllib.request.urlopen(post(f"http://127.0.0.1:{local_port}/broken", {}), timeout=10) as response,
        pytest.raises(http.client.IncompleteRead),
    ):
        response.read()


@pytest.mark.unit
@pytest.mark.parametrize("lost_acknowledgements", [1, 3])
def test_a_lost_reply_acknowledgement_does_not_duplicate_stream_content(lost_acknowledgements: int) -> None:
    gateway = Gateway()
    relay = Relay("test-secret")
    handler = relay.tunnel_handler()
    lost: list[int] = []

    class LoseAcknowledgement(handler):
        def answer(self, status: int, body: bytes = b"") -> None:
            if self.path.startswith("/reply/") and status == 200 and len(lost) < lost_acknowledgements:
                lost.append(status)
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            super().answer(status, body)

    local_server = ThreadingHTTPServer(("127.0.0.1", 0), relay.local_handler())
    tunnel_server = ThreadingHTTPServer(("127.0.0.1", 0), LoseAcknowledgement)
    for server in (local_server, tunnel_server):
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
    pump = TunnelPump(f"http://127.0.0.1:{tunnel_server.server_port}", "test-secret", gateway.port, pollers=1)
    try:
        pump.start(timeout=10)
        gateway.release.set()
        request = post(f"http://127.0.0.1:{local_server.server_port}/stream", {})
        with urllib.request.urlopen(request, timeout=10) as response:
            if lost_acknowledgements == 1:
                assert response.read() == b"data: one\n\ndata: two\n\n"
            else:
                with pytest.raises(http.client.IncompleteRead):
                    response.read()
        assert len(lost) == lost_acknowledgements
    finally:
        pump.stop()
        for server in (local_server, tunnel_server):
            server.shutdown()
            server.server_close()
        gateway.close()


@pytest.mark.unit
def test_a_relay_that_answers_after_a_refused_poll_starts_the_pump() -> None:
    # The sandbox is reachable a moment before its relay listens, and E2B answers that first poll with 502.
    gateway = Gateway()
    relay = Relay("test-secret")
    refused: list[str] = []

    class RefuseFirstReady(relay.tunnel_handler()):
        def answer(self, status: int, body: bytes = b"") -> None:
            if self.path == "/ready" and not refused:
                refused.append(self.path)
                status = 502
            super().answer(status, body)

    tunnel_server = ThreadingHTTPServer(("127.0.0.1", 0), RefuseFirstReady)
    tunnel_server.daemon_threads = True
    threading.Thread(target=tunnel_server.serve_forever, daemon=True).start()
    pump = TunnelPump(f"http://127.0.0.1:{tunnel_server.server_port}", "test-secret", gateway.port, pollers=1)
    try:
        pump.start(timeout=10)
        assert refused == ["/ready"]
        assert pump.failure is None
    finally:
        pump.stop()
        tunnel_server.shutdown()
        tunnel_server.server_close()
        gateway.close()


@pytest.mark.unit
def test_a_private_port_gets_the_token_on_each_relay_request_and_the_reef_port_never_does() -> None:
    # The relay behind a private E2B port: the provider refuses a request without the sandbox's token.
    gateway = Gateway()
    relay = Relay("test-secret")
    seen: list[tuple[str, str | None, str | None]] = []

    class PrivatePort(relay.tunnel_handler()):
        def allowed(self) -> bool:
            token = self.headers.get(TRAFFIC_ACCESS_TOKEN_HEADER)
            seen.append((self.path.split("/")[1], token, self.headers.get("x-relay-secret")))
            if token != TRAFFIC_TOKEN:
                self.answer(403)
                return False
            return super().allowed()

    local_server, tunnel_server = serve(relay.local_handler()), serve(PrivatePort)
    pump = TunnelPump(
        f"http://127.0.0.1:{tunnel_server.server_port}",
        "test-secret",
        gateway.port,
        pollers=1,
        tunnel_headers={TRAFFIC_ACCESS_TOKEN_HEADER: TRAFFIC_TOKEN},
    )
    try:
        pump.start(timeout=10)
        request = post(f"http://127.0.0.1:{local_server.server_port}/echo", {"x": 1}, {"Authorization": "Bearer in"})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert json.loads(response.read()) == {"body": {"x": 1}, "auth": "Bearer in"}
    finally:
        pump.stop()
        for server in (local_server, tunnel_server):
            server.shutdown()
            server.server_close()
        gateway.close()
    assert {path for path, _, _ in seen} == {"ready", "next", "reply"}
    assert all(token == TRAFFIC_TOKEN and secret == "test-secret" for _, token, secret in seen)
    # The Reef port gets the process's request only: neither the provider's token nor the relay's secret.
    (received,) = gateway.received
    assert received["authorization"] == "Bearer in"
    assert TRAFFIC_ACCESS_TOKEN_HEADER not in received and "x-relay-secret" not in received


@pytest.mark.unit
@pytest.mark.parametrize("redirected_path", ["ready", "next", "reply"])
def test_a_relay_request_never_follows_a_redirect(redirected_path: str) -> None:
    """A redirect from the relay's address fails the one request it answers. Neither the provider's token nor the
    relay's secret reaches the redirect's target, and an answer the relay did not take never counts as delivered."""
    gateway = Gateway()
    relay = Relay("test-secret")
    redirected: list[str] = []
    stolen: list[dict[str, str]] = []

    class Target(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def collect(self) -> None:
            stolen.append(dict(self.headers.items()))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            self.collect()

        def do_POST(self) -> None:
            self.collect()

        def log_message(self, *args) -> None:
            pass

    target = serve(Target)

    class RedirectOnce(relay.tunnel_handler()):
        def allowed(self) -> bool:
            if self.path.split("/")[1] != redirected_path or redirected:
                return super().allowed()
            redirected.append(self.path)
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/collect")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

    local_server, tunnel_server = serve(relay.local_handler()), serve(RedirectOnce)
    pump = TunnelPump(
        f"http://127.0.0.1:{tunnel_server.server_port}",
        "test-secret",
        gateway.port,
        pollers=1,
        tunnel_headers={TRAFFIC_ACCESS_TOKEN_HEADER: TRAFFIC_TOKEN},
    )
    request = post(f"http://127.0.0.1:{local_server.server_port}/echo", {"x": 1})
    try:
        pump.start(timeout=10)
        if redirected_path == "reply":
            # The relay did not take the answer's first piece, so the process gets an error and not the answer.
            with pytest.raises(urllib.error.HTTPError) as failed:
                urllib.request.urlopen(request, timeout=10)
            assert failed.value.code == 502
            failed.value.close()
        else:
            # A refused /ready or /next is tried again, and the tunnel still answers.
            with urllib.request.urlopen(request, timeout=10) as response:
                assert json.loads(response.read())["body"] == {"x": 1}
    finally:
        pump.stop()
        for server in (local_server, tunnel_server, target):
            server.shutdown()
            server.server_close()
        gateway.close()
    assert len(redirected) == 1 and redirected[0].startswith(f"/{redirected_path}")
    assert stolen == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tunnel_headers", "message"),
    [
        ({"X-Relay-Secret": "replacement-secret"}, "cannot replace x-relay-secret"),
        ({"x-relay-secret": "replacement-secret"}, "cannot replace x-relay-secret"),
        (
            {TRAFFIC_ACCESS_TOKEN_HEADER: "first-token", "E2B-Traffic-Access-Token": "second-token"},
            "occurs twice, ignoring case",
        ),
        ({"bad name": "header-value"}, "is not an HTTP token"),
        ({"": "header-value"}, "is not an HTTP token"),
        ({TRAFFIC_ACCESS_TOKEN_HEADER: "token-value\r\nx-relay-secret: injected"}, "is not visible ASCII text"),
        ({TRAFFIC_ACCESS_TOKEN_HEADER: ""}, "is not visible ASCII text"),
        ({TRAFFIC_ACCESS_TOKEN_HEADER: None}, "is not visible ASCII text"),
    ],
)
def test_a_tunnel_header_cannot_replace_the_relay_secret_or_break_a_request(
    tunnel_headers: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message) as refused:
        TunnelPump("https://relay.example", "test-secret", 9000, tunnel_headers=tunnel_headers)
    # The message names the header and never its value.
    assert not any(value and value in str(refused.value) for value in tunnel_headers.values())


@pytest.mark.unit
def test_a_root_crosses_as_it_is_and_the_command_names_its_sandbox_path(tmp_path: Path) -> None:
    root = tmp_path / "reef-agent-abc"
    (root / "workspace" / "harness" / "skills").mkdir(parents=True)
    (root / "workspace" / "harness" / "skills" / "a.md").write_text("A")
    (root / "pi-agent").mkdir()
    (root / "pi-agent" / "models.json").write_text("{}")
    copy = tmp_path / "copy"
    (copy / "stale").mkdir(parents=True)
    unpack(pack(root), copy)
    assert sorted(p.relative_to(copy).as_posix() for p in copy.rglob("*")) == [
        "pi-agent",
        "pi-agent/models.json",
        "workspace",
        "workspace/harness",
        "workspace/harness/skills",
        "workspace/harness/skills/a.md",
    ]
    # A member that would land outside the directory, or link there, is left out and the rest still comes back:
    # pulseaudio leaves such a link in a trial's home when an extension plays sound.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.addfile(tarfile.TarInfo("../escape"), io.BytesIO(b""))
        link = tarfile.TarInfo(".config/pulse/abc-runtime")
        link.type, link.linkname = tarfile.SYMTYPE, "/tmp/pulse-xyz"
        archive.addfile(link)
        session = tarfile.TarInfo("pi-agent/sessions/s.jsonl")
        session.size = 5
        archive.addfile(session, io.BytesIO(b"hello"))
    target = tmp_path / "target"
    unpack(buffer.getvalue(), target)
    assert [p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() or p.is_symlink()] == [
        "pi-agent/sessions/s.jsonl"
    ]
    assert not (tmp_path / "escape").exists()

    command, env = remote_command(
        ["/opt/agents/pi", "--mode", "json", "-p", "it's here"],
        {"HOME": str(root), "PI_CODING_AGENT_DIR": f"{root}/pi-agent", "PATH": "/host/bin", "REEF_TOKEN": "t"},
        root,
    )
    assert remote_root(root) == "/home/user/reef/reef-agent-abc"
    assert command == "/opt/agents/pi --mode json -p 'it'\"'\"'s here'"
    assert env == {
        "HOME": "/home/user/reef/reef-agent-abc",
        "PI_CODING_AGENT_DIR": "/home/user/reef/reef-agent-abc/pi-agent",
        "REEF_TOKEN": "t",
    }
    assert template_alias("pi", "0.84.2") == "reef-pi-0-84-2-runs-v1"


@pytest.mark.unit
def test_an_e2b_executor_needs_its_key(monkeypatch) -> None:
    with pytest.raises(SandboxUnavailableError, match="E2B_API_KEY"):
        E2BExecutor(api_key="").preflight()
    assert "sk" not in repr(E2BExecutor(api_key="sk-e2b"))


@pytest.mark.unit
def test_the_first_sandbox_stops_only_the_ones_its_own_deployment_left(tmp_path: Path, monkeypatch) -> None:
    e2b = pytest.importorskip("e2b")  # the e2b extra; the reap is the SDK's list and kill

    import reef.execution.sandbox.remote as remote_executor
    from reef.execution.sandbox.remote import deployment_owner

    mine, theirs = deployment_owner(tmp_path / "a"), deployment_owner(tmp_path / "b")
    assert mine != theirs and mine == deployment_owner(tmp_path / "a")
    running = [
        SimpleNamespace(sandbox_id="left-1", metadata={"reef": "sandbox", "reef_owner": mine}),
        SimpleNamespace(sandbox_id="left-2", metadata={"reef": "sandbox", "reef_owner": mine}),
        # A filter the API does not honour must not cost another deployment its sandbox.
        SimpleNamespace(sandbox_id="other", metadata={"reef": "sandbox", "reef_owner": theirs}),
    ]
    queries, killed = [], []

    class Pages:
        has_next = True

        def next_items(self):
            self.has_next = False
            return running

    def listing(query=None, **opts):
        queries.append((query.metadata, opts["api_key"]))
        return Pages()

    monkeypatch.setattr(e2b.Sandbox, "list", staticmethod(listing))
    monkeypatch.setattr(
        e2b.Sandbox, "kill", staticmethod(lambda sandbox_id, **opts: killed.append(sandbox_id) or True)
    )

    executor = E2BExecutor(api_key="k", owner=mine)
    assert executor.labels() == {"reef": "sandbox", "reef_owner": mine}
    monkeypatch.setattr(remote_executor, "REAPED_OWNERS", set())
    executor.reap_once()
    assert killed == ["left-1", "left-2"]
    # Once per process: the sandboxes it starts itself are never taken for leftovers.
    executor.reap_once()
    assert len(queries) == 1
    assert queries == [({"reef": "sandbox", "reef_owner": mine}, "k")]
    # No owner, no deployment to speak for: nothing is listed or stopped.
    E2BExecutor(api_key="k").reap_once()
    assert E2BExecutor(api_key="k").reap() == 0 and len(queries) == 1


def test_a_run_longer_than_e2bs_ceiling_opens_within_it_and_keeps_its_sandbox_alive(monkeypatch) -> None:
    """E2B refuses a sandbox asked to live past its ceiling, so a long agent run opens within it and the session
    extends the deadline while it works, instead of losing the sandbox mid-run."""
    e2b = pytest.importorskip("e2b")

    import reef.execution.sandbox.remote as remote_executor
    from reef.execution.sandbox.e2b import SANDBOX_MAX_TIMEOUT_S, E2BExecutor, E2BSession

    asked: list[int] = []
    extended: list[int] = []

    class Sandbox:
        def set_timeout(self, seconds):
            extended.append(seconds)

        def kill(self):
            return True

    def create(**options):
        asked.append(options["timeout"])
        return Sandbox()

    monkeypatch.setattr(e2b.Sandbox, "create", staticmethod(create))
    monkeypatch.setattr(remote_executor, "REAPED_OWNERS", set())

    # Four hours of agent run: the sandbox is still created within what E2B grants.
    session = E2BExecutor(api_key="k", template="reef-pi", timeout_s=14400).open()
    try:
        assert asked == [SANDBOX_MAX_TIMEOUT_S]
        # The keeper asks for the ceiling again rather than waiting out its own period.
        session.keeper.sandbox.set_timeout(SANDBOX_MAX_TIMEOUT_S)
        assert extended == [SANDBOX_MAX_TIMEOUT_S]
        assert session.keeper.is_alive() and not session.keeper.stopped.is_set()
    finally:
        session.close()
    assert session.keeper.stopped.is_set()

    # A short run asks for what it needs and no more.
    E2BExecutor(api_key="k", template="reef-pi", timeout_s=60).open().close()
    assert asked[-1] == 660

    # A session closed twice stops its keeper once and stays closed.
    plain = E2BSession(Sandbox())
    plain.close()
    assert plain.keeper.stopped.is_set()


class CreatedSandbox:
    """An E2B sandbox as ``Sandbox.create`` returns it: with the traffic access token of a private sandbox, or None
    for a public one."""

    def __init__(self, traffic_access_token: str | None) -> None:
        self.traffic_access_token = traffic_access_token
        self.files = Mock()
        self.commands = Mock()

    def get_host(self, port: int) -> str:
        # E2B's wildcard host of a sandbox port; no other header routes the request.
        return f"{port}-sandbox-id.agentenv.example"

    def set_timeout(self, seconds: int) -> None:
        pass

    def kill(self) -> bool:
        return True


@pytest.mark.unit
@pytest.mark.parametrize("traffic_access_token", [TRAFFIC_TOKEN, None])
def test_an_e2b_relay_gets_the_sandboxs_traffic_token_when_the_service_returns_one(
    monkeypatch, traffic_access_token: str | None
) -> None:
    e2b = pytest.importorskip("e2b")

    import reef.execution.sandbox.remote as remote_executor

    created: list[dict] = []
    started: list[tuple[str, dict[str, str]]] = []

    def create(**options):
        created.append(options)
        return CreatedSandbox(traffic_access_token)

    class RecordingPump:
        """Stands in for ``TunnelPump`` in ``RemoteSession.forward``: records the relay's address and the session's
        tunnel headers, and starts no thread. The relay tests above run the real pump."""

        def __init__(self, url: str, secret: str, port: int, *, tunnel_headers: dict[str, str]) -> None:
            started.append((url, tunnel_headers))

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(e2b.Sandbox, "create", staticmethod(create))
    monkeypatch.setattr(remote_executor, "TunnelPump", RecordingPump)
    E2BExecutor(api_key="fleet-api-key", template="reef-pi", forward_ports=(9000,)).open().close()
    # Reef creates an ordinary sandbox: the service decides whether its ports are private.
    (options,) = created
    assert "network" not in options
    # The relay at the sandbox's wildcard host gets the token when there is one, and nothing else: never the API key.
    tunnel_headers = {TRAFFIC_ACCESS_TOKEN_HEADER: TRAFFIC_TOKEN} if traffic_access_token else {}
    assert started == [("https://21080-sandbox-id.agentenv.example", tunnel_headers)]
