"""Opt-in cloud acceptance of the Modal sandbox: REEF_TEST_MODAL=1 and Modal credentials.

The test builds an image for a small npm program, starts one sandbox, and checks a process in it: it runs the
program, reaches a server on this host through the relay tunnel, writes its workspace (copied back here), cannot
write outside it, and is stopped at its limit. It costs a few cents of Modal CPU time.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.sandbox.modal import ModalExecutor, ModalSession
from reef.sandbox.process import RolloutTimeout, SandboxProgram
from reef.sandbox.remote import deployment_owner

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_MODAL") != "1", reason="opt-in paid Modal test")

ANSWER = "REEF_MODAL_OK"


class Cowsay(SandboxProgram):
    def __init__(self) -> None:
        self.name = "cowsay"
        self.binary = "cowsay"
        self.self_isolating = False
        self.npm_package = "cowsay@1.6.0"
        self.version = "1.6.0"

    def host_binary(self) -> str:
        raise AssertionError("the program runs in the sandbox")

    def host_install_dir(self) -> Path | None:
        return None


@pytest.fixture
def host_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = ANSWER.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()


def test_a_process_runs_jailed_in_a_modal_sandbox_and_reaches_the_host_through_the_tunnel(tmp_path, host_server):
    executor, binary = ModalExecutor(app_name="reef-tests", owner=deployment_owner(tmp_path)).for_program(Cowsay())
    assert isinstance(executor, ModalExecutor) and binary == "cowsay"
    executor.preflight()
    root = tmp_path / "rollout"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "task.md").write_text("say hi")
    script = (
        "cowsay hi > answer.txt"
        f" && python3 -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{host_server}/')"
        '.read().decode())" > fetched.txt'
        " && (touch /etc/reef-escape 2>/dev/null && echo writable || echo read-only) > jail.txt"
        " && cat task.md"
    )
    session = executor.open_session(forward_ports=(host_server,), timeout_s=300)
    assert isinstance(session, ModalSession)
    try:
        outcome = session.launch(["/bin/sh", "-c", script], root=root, workspace=workspace, env={}, timeout=120)
        assert outcome.exit_code == 0, outcome.stderr
        assert outcome.stdout.strip() == "say hi"
        assert "hi" in (workspace / "answer.txt").read_text()
        assert (workspace / "fetched.txt").read_text().strip() == ANSWER
        assert (workspace / "jail.txt").read_text().strip() == "read-only"
        with pytest.raises(RolloutTimeout):
            session.launch(["sleep", "60"], root=root, workspace=workspace, env={}, timeout=3)
    finally:
        session.close()
