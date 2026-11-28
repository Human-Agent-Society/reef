"""Opt-in cloud acceptance of the boxd sandbox: REEF_TEST_BOXD=1 and BOXD_API_KEY.

The test checks the startup probe, then starts one isolated machine and checks a process in it: it runs a small npm
program, reaches a server on this host through the relay tunnel, writes its workspace (copied back here), cannot
write outside it or become root, finds no boxd tools, cannot reach Docker or the machine user's files, and is stopped
at its limit. Closing the session deletes the machine.

Optional: BOXD_ORG, BOXD_REGION, REEF_TEST_BOXD_PREFIX, REEF_TEST_BOXD_LABELS (name=value,name=value) and
REEF_TEST_BOXD_AUTO_DESTROY=0.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.execution.sandbox.boxd import MACHINE_ENV_FILE, BoxdExecutor, BoxdSession
from reef.execution.sandbox.process import ProgramTimeoutError, SandboxProgram

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_BOXD") != "1", reason="opt-in paid boxd test")

ANSWER = "REEF_BOXD_OK"


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


def test_a_process_runs_jailed_on_an_isolated_boxd_machine_and_reaches_the_host(tmp_path, host_server):
    labels = dict(pair.split("=", 1) for pair in os.environ.get("REEF_TEST_BOXD_LABELS", "").split(",") if "=" in pair)
    configured = BoxdExecutor.from_config(
        {
            "boxd_region": os.environ.get("BOXD_REGION", ""),
            "boxd_name_prefix": os.environ.get("REEF_TEST_BOXD_PREFIX", "reef-test-"),
            "boxd_labels": labels,
            "boxd_auto_destroy": os.environ.get("REEF_TEST_BOXD_AUTO_DESTROY") != "0",
        },
        os.environ,
    )
    executor, binary = configured.for_program(Cowsay())
    assert isinstance(executor, BoxdExecutor) and binary == "cowsay"
    executor.preflight()
    root = tmp_path / "run"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "task.md").write_text("say hi")
    # The first command reads the standard input to its end, as pi does: it must be empty, not open.
    script = (
        "cat > stdin.txt"
        " && cowsay hi > answer.txt"
        f" && python3 -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{host_server}/')"
        '.read().decode())" > fetched.txt'
        " && (touch /etc/reef-escape 2>/dev/null && echo writable || echo read-only) > jail.txt"
        " && (sudo -n true 2>/dev/null && echo root || echo no-root) > root.txt"
        " && (command -v boxd run || echo no-tools) > tools.txt"
        " && (docker version >/dev/null 2>&1 && echo docker || echo no-docker) > docker.txt"
        " && (ls /home/boxd >/dev/null 2>&1 && echo readable || echo denied) > home.txt"
        " && (ls /run/user/1000 >/dev/null 2>&1 && echo readable || echo denied) > sockets.txt"
        f" && (cat {MACHINE_ENV_FILE} >/dev/null 2>&1 && echo readable || echo denied) > env-file.txt"
        " && cat task.md"
    )
    session = executor.open_session(forward_ports=(host_server,), timeout_s=300)
    assert isinstance(session, BoxdSession)
    try:
        outcome = session.launch(["/bin/sh", "-c", script], root=root, workspace=workspace, env={}, timeout=120)
        assert outcome.exit_code == 0, outcome.stderr
        assert outcome.stdout.strip() == "say hi"
        assert (workspace / "stdin.txt").read_text() == ""
        assert "hi" in (workspace / "answer.txt").read_text()
        assert (workspace / "fetched.txt").read_text().strip() == ANSWER
        assert (workspace / "jail.txt").read_text().strip() == "read-only"
        assert (workspace / "root.txt").read_text().strip() == "no-root"
        assert (workspace / "tools.txt").read_text().strip() == "no-tools"
        assert (workspace / "docker.txt").read_text().strip() == "no-docker"
        assert (workspace / "home.txt").read_text().strip() == "denied"
        assert (workspace / "sockets.txt").read_text().strip() == "denied"
        assert (workspace / "env-file.txt").read_text().strip() == "denied"
        with pytest.raises(ProgramTimeoutError):
            session.launch(["sleep", "60"], root=root, workspace=workspace, env={}, timeout=3)
    finally:
        session.close()
    from boxd.errors import NotFoundError

    with pytest.raises(NotFoundError):
        executor.client().machines.get(session.machine_id)
