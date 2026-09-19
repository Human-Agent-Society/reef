"""The E2B executor's tunnel and file copies, run locally: the relay as the sandbox runs it, a pump as Reef does."""

from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.harness.episodes.e2b import (
    RELAY_SCRIPT,
    E2BExecutor,
    TunnelPump,
    pack,
    remote_command,
    remote_root,
    template_alias,
    unpack,
)
from reef.harness.episodes.executor import SandboxUnavailable


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Gateway:
    """The Reef-side port the tunnel reaches: a JSON answer, a stream held open between pieces, and a request that
    waits for another to arrive, as a trial waits on the model calls it makes."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.arrived = threading.Event()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
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
    time.sleep(0.3)
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
def test_a_root_crosses_as_it_is_and_the_command_names_its_sandbox_path(tmp_path: Path) -> None:
    root = tmp_path / "reef-proposer-abc"
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
        ["/opt/reef/pi/node_modules/.bin/pi", "--mode", "json", "-p", "it's here"],
        {"HOME": str(root), "PI_CODING_AGENT_DIR": f"{root}/pi-agent", "PATH": "/host/bin", "REEF_TOKEN": "t"},
        root,
    )
    assert remote_root(root) == "/home/user/reef/reef-proposer-abc"
    assert command == "pi --mode json -p 'it'\"'\"'s here'"
    assert env == {
        "HOME": "/home/user/reef/reef-proposer-abc",
        "PI_CODING_AGENT_DIR": "/home/user/reef/reef-proposer-abc/pi-agent",
        "REEF_TOKEN": "t",
    }
    assert template_alias("pi", "0.84.2") == "reef-pi-0-84-2"


@pytest.mark.unit
def test_an_e2b_executor_needs_its_key(monkeypatch) -> None:
    with pytest.raises(SandboxUnavailable, match="E2B_API_KEY"):
        E2BExecutor(api_key="").preflight()
    assert "sk" not in repr(E2BExecutor(api_key="sk-e2b"))


@pytest.mark.unit
def test_the_first_sandbox_stops_only_the_ones_its_own_deployment_left(tmp_path: Path, monkeypatch) -> None:
    e2b = pytest.importorskip("e2b")  # the e2b extra; the reap is the SDK's list and kill

    import reef.harness.episodes.e2b as e2b_executor
    from reef.harness.episodes.e2b import deployment_owner

    mine, theirs = deployment_owner(tmp_path / "a"), deployment_owner(tmp_path / "b")
    assert mine != theirs and mine == deployment_owner(tmp_path / "a")
    running = [
        SimpleNamespace(sandbox_id="left-1", metadata={"reef": "episode", "reef_owner": mine}),
        SimpleNamespace(sandbox_id="left-2", metadata={"reef": "episode", "reef_owner": mine}),
        # A filter the API does not honour must not cost another deployment its sandbox.
        SimpleNamespace(sandbox_id="other", metadata={"reef": "episode", "reef_owner": theirs}),
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
    assert executor.labels() == {"reef": "episode", "reef_owner": mine}
    monkeypatch.setattr(e2b_executor, "REAPED_OWNERS", set())
    executor.reap_once()
    assert killed == ["left-1", "left-2"]
    # Once per process: the sandboxes it starts itself are never taken for leftovers.
    executor.reap_once()
    assert len(queries) == 1
    assert queries == [({"reef": "episode", "reef_owner": mine}, "k")]
    # No owner, no deployment to speak for: nothing is listed or stopped.
    E2BExecutor(api_key="k").reap_once()
    assert E2BExecutor(api_key="k").reap() == 0 and len(queries) == 1
