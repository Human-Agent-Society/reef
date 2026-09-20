"""The relay an E2B sandbox runs so its processes reach one loopback port of the Reef host.

Uploaded into the sandbox and started as root with the standard library only
(the sandbox's ``python3``); nothing in Reef imports it. It listens on
``127.0.0.1:<port>`` inside the sandbox, the port the processes were told
(the Reef gateway's own, so no URL is rewritten), and on ``0.0.0.0:<tunnel>``,
which E2B exposes at a public address. The sandbox cannot reach the Reef host,
so Reef reaches in: it polls ``GET /next`` for the next request a process
made and answers it with ``POST /reply/<id>`` pieces, the first carrying the
status and headers, the last ``end``. The tunnel refuses a request without
the secret, so the internet can neither take nor answer one. The sandbox's
own root could, but the tunnel carries nothing that holds a key: Reef adds
every credential on its side of the gateway.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: How long one ``GET /next`` waits for a request before answering 204.
POLL_SECONDS = 20.0
#: How long a process waits for Reef to answer before the relay gives up with 504.
ANSWER_SECONDS = 3600.0


class Pending:
    """One request a process made, waiting for Reef's answer in pieces."""

    def __init__(self, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        self.id = uuid.uuid4().hex
        self.request = {
            "id": self.id,
            "method": method,
            "path": path,
            "headers": headers,
            "body": base64.b64encode(body).decode(),
        }
        self.pieces: queue.Queue[dict] = queue.Queue()


class Relay:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.waiting: queue.Queue[Pending] = queue.Queue()
        self.open: dict[str, Pending] = {}
        self.lock = threading.Lock()

    def local_handler(self) -> type[BaseHTTPRequestHandler]:
        relay = self

        class Local(BaseHTTPRequestHandler):
            # A close-delimited answer, so a stream passes piece by piece without a length or chunking.
            protocol_version = "HTTP/1.0"

            def relay_request(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                headers = {name: value for name, value in self.headers.items() if name.lower() != "host"}
                pending = Pending(self.command, self.path, headers, body)
                with relay.lock:
                    relay.open[pending.id] = pending
                relay.waiting.put(pending)
                try:
                    first = pending.pieces.get(timeout=ANSWER_SECONDS)
                except queue.Empty:
                    self.send_error(504, "Reef did not answer through the tunnel")
                    return
                try:
                    self.send_response(int(first.get("status") or 502))
                    for name, value in (first.get("headers") or {}).items():
                        if name.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(name, value)
                    self.end_headers()
                    piece = first
                    while True:
                        data = base64.b64decode(piece.get("data") or "")
                        if data:
                            self.wfile.write(data)
                            self.wfile.flush()
                        if piece.get("end"):
                            return
                        piece = pending.pieces.get(timeout=ANSWER_SECONDS)
                except (BrokenPipeError, ConnectionResetError, queue.Empty):
                    return
                finally:
                    with relay.lock:
                        relay.open.pop(pending.id, None)

            def do_GET(self) -> None:
                self.relay_request()

            def do_POST(self) -> None:
                self.relay_request()

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Local

    def tunnel_handler(self) -> type[BaseHTTPRequestHandler]:
        relay = self

        class Tunnel(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def answer(self, status: int, body: bytes = b"") -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def allowed(self) -> bool:
                if self.headers.get("x-relay-secret") != relay.secret:
                    self.answer(403)
                    return False
                return True

            def do_GET(self) -> None:
                if not self.allowed():
                    return
                if self.path == "/ready":
                    self.answer(200)
                    return
                if self.path != "/next":
                    self.answer(404)
                    return
                try:
                    pending = relay.waiting.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    self.answer(204)
                    return
                self.answer(200, json.dumps(pending.request).encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not self.allowed():
                    return
                if not self.path.startswith("/reply/"):
                    self.answer(404)
                    return
                with relay.lock:
                    pending = relay.open.get(self.path[len("/reply/") :])
                if pending is None:
                    self.answer(410)
                    return
                pending.pieces.put(json.loads(raw or b"{}"))
                self.answer(200)

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Tunnel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="the loopback port the sandbox's processes call")
    parser.add_argument("--tunnel-port", type=int, required=True, help="the port Reef polls from outside")
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--tunnel-host", default="0.0.0.0")
    args = parser.parse_args()
    relay = Relay(Path(args.secret_file).read_text(encoding="utf-8").strip())
    local = ThreadingHTTPServer(("127.0.0.1", args.port), relay.local_handler())
    tunnel = ThreadingHTTPServer((args.tunnel_host, args.tunnel_port), relay.tunnel_handler())
    local.daemon_threads = tunnel.daemon_threads = True
    threading.Thread(target=local.serve_forever, daemon=True).start()
    print("relay ready", flush=True)
    tunnel.serve_forever()


if __name__ == "__main__":
    main()
