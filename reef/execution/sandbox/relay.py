"""The relay a remote sandbox (E2B or Modal) runs so its processes reach one loopback port of the Reef host.

Uploaded into the sandbox and started as root with the standard library only
(the sandbox's ``python3``); nothing in Reef imports it. It listens on
``127.0.0.1:<port>`` inside the sandbox, the port the processes were told
(the Reef gateway's own, so no URL is rewritten), and on ``0.0.0.0:<tunnel>``,
which the provider exposes at a public address. The sandbox cannot reach the Reef host,
so Reef reaches in: it polls ``GET /next`` for the next request a process
made and answers it with ``POST /reply/<id>`` pieces, the first carrying the
status and headers, the last ``end``. A request that Reef took but did not
answer within ``REOFFER_SECONDS`` goes back to ``/next``: the exchange that took
it can fail before Reef reads it, and Reef ignores a request that it already
answers. The tunnel refuses a request without
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
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: How long one ``GET /next`` waits for a request before answering 204.
POLL_SECONDS = 20.0
#: How long a process waits for Reef to answer before the relay gives up with 504.
ANSWER_SECONDS = 3600.0
#: How long a request that ``/next`` gave out can go without a reply piece before ``/next`` gives it out again.
REOFFER_SECONDS = 60.0


class Pending:
    """The ID and ordered answer pieces of one request waiting for Reef."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.pieces: queue.Queue[dict] = queue.Queue()
        self.sequence = 0


class Relay:
    """The requests of the sandbox's processes, from their arrival until their answer ends.

    Values: ``open`` maps the ID of each request that a process still waits on to its ``Pending``. ``waiting`` holds
    the open requests that ``/next`` can give out, oldest first. ``given`` maps each open request that ``/next`` gave
    out, and that has no reply piece yet, to the monotonic time it went out and the request. No ID is in both
    ``waiting`` and ``given``. ``lock`` guards all three, and ``request_ready`` signals a new waiting request.

    Purpose: ``local_handler`` adds a request and removes it when its answer ends; ``tunnel_handler`` gives requests
    to Reef (``take``) and takes their reply pieces.
    """

    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.waiting: OrderedDict[str, dict[str, str | dict[str, str]]] = OrderedDict()
        self.open: dict[str, Pending] = {}
        self.given: dict[str, tuple[float, dict[str, str | dict[str, str]]]] = {}
        self.lock = threading.Lock()
        self.request_ready = threading.Condition(self.lock)

    def enqueue(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Pending:
        """Queue a request and return the state that receives its answer."""
        pending = Pending()
        request: dict[str, str | dict[str, str]] = {
            "id": pending.id,
            "method": method,
            "path": path,
            "headers": headers,
            "body": base64.b64encode(body).decode(),
        }
        with self.request_ready:
            self.open[pending.id] = pending
            self.waiting[pending.id] = request
            self.request_ready.notify()
        return pending

    def take(self) -> dict[str, str | dict[str, str]] | None:
        """The oldest waiting request, or None when none waits within ``POLL_SECONDS``. A given request without a
        reply piece for ``REOFFER_SECONDS`` waits again first: the exchange that took it may have lost it."""
        deadline = time.monotonic() + POLL_SECONDS
        with self.request_ready:
            while True:
                now = time.monotonic()
                for request_id, (given_at, request) in list(self.given.items()):
                    if now - given_at >= REOFFER_SECONDS:
                        self.given.pop(request_id)
                        self.waiting[request_id] = request
                if self.waiting:
                    request_id, request = self.waiting.popitem(last=False)
                    self.given[request_id] = (now, request)
                    return request
                if now >= deadline:
                    return None
                # Wake at least each second, so a given request goes out again on time.
                self.request_ready.wait(min(deadline - now, 1.0))

    def local_handler(self) -> type[BaseHTTPRequestHandler]:
        relay = self

        class Local(BaseHTTPRequestHandler):
            # Chunk framing makes an interrupted answer distinguishable from a complete one.
            protocol_version = "HTTP/1.1"

            def relay_request(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                headers = {name: value for name, value in self.headers.items() if name.lower() != "host"}
                pending = relay.enqueue(self.command, self.path, headers, self.rfile.read(length) if length else b"")
                first = None
                try:
                    first = pending.pieces.get(timeout=ANSWER_SECONDS)
                    status = int(first.get("status") or 502)
                    self.send_response(status)
                    for name, value in (first.get("headers") or {}).items():
                        if name.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(name, value)
                    chunked = status not in (204, 304)
                    if chunked:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    piece = first
                    while True:
                        data = base64.b64decode(piece.get("data") or "")
                        if data and chunked:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                            self.wfile.flush()
                        if piece.get("end"):
                            if piece.get("error"):
                                self.close_connection = True
                            elif chunked:
                                self.wfile.write(b"0\r\n\r\n")
                                self.wfile.flush()
                            return
                        piece = pending.pieces.get(timeout=ANSWER_SECONDS)
                except (BrokenPipeError, ConnectionResetError, queue.Empty):
                    self.close_connection = True
                finally:
                    with relay.lock:
                        relay.waiting.pop(pending.id, None)
                        relay.given.pop(pending.id, None)
                        relay.open.pop(pending.id, None)
                if first is None:
                    self.send_error(504, "Reef did not answer through the tunnel")

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
                request = relay.take()
                if request is None:
                    self.answer(204)
                    return
                self.answer(200, json.dumps(request).encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not self.allowed():
                    return
                if not self.path.startswith("/reply/"):
                    self.answer(404)
                    return
                piece = json.loads(raw or b"{}")
                sequence = piece.get("sequence")
                if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                    self.answer(400)
                    return
                with relay.lock:
                    request_id = self.path[len("/reply/") :]
                    pending = relay.open.get(request_id)
                    if pending is None:
                        status = 410
                    elif piece.get("error"):
                        # An abort must arrive even if the last data piece's acknowledgements were all lost.
                        pending.pieces.put(piece)
                        relay.given.pop(request_id, None)
                        relay.waiting.pop(request_id, None)
                        status = 200
                    elif sequence > pending.sequence:
                        status = 409
                    else:
                        if sequence == pending.sequence:
                            pending.pieces.put(piece)
                            pending.sequence += 1
                        # Reef answers the request, so it does not go out again.
                        relay.given.pop(request_id, None)
                        relay.waiting.pop(request_id, None)
                        status = 200
                self.answer(status)

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
