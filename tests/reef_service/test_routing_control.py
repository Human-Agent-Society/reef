"""Private lifecycle calls reach the REEF child only over its Unix socket."""

import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingUnixStreamServer

import pytest

from reef.service.deploy.routing_control import ReefLifecycleClient, RoutingControlError, json_object

pytestmark = pytest.mark.unit


class UnixHTTPServer(ThreadingUnixStreamServer):
    daemon_threads = True


@pytest.mark.parametrize(("status", "retryable"), [(200, None), (302, False), (409, True), (503, True)])
def test_control_uses_its_socket_ignores_proxies_and_never_follows_redirects(
    monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool | None
) -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
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
