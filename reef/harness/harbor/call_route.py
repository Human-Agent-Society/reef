"""The public route of the model calls of evaluation rollouts whose agent runs in a remote sandbox.

An agent that runs in a task's container in a remote environment calls its rollout's capture proxy at a signed URL
under ``evaluation.call_url``: ``<call_url>/<ticket>/v1/...``. The ticket names the evaluation process that runs the
rollout (its address and the port of its route), the port of the rollout's capture proxy in that process and a key of
the rollout. Its signature is keyed by the deployment's service tokens, so every process of the deployment checks it,
and no one without a token can make one.

Each evaluation process serves ``ROLLOUT_CALL_PATH`` on a port of its own, ``evaluation.call_port``, from the start
of its rollout worker until the worker has stopped. So a process that stops still answers the calls of the rollouts
that it finishes (``evaluation.on_stop``), after its service port has closed. The deployment's ingress sends
``<call_url>/`` to ``ROLLOUT_CALL_PATH`` of any of its evaluation processes. That process sends the call to the
process that the ticket names, which sends it to the capture proxy on its loopback while the rollout runs. The sandbox
holds no credential: the proxy adds Reef's token.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import secrets
import socket
import threading
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass

import aiohttp
from aiohttp import web

from reef.execution.sandbox.remote import HOP_BY_HOP

#: The path under which an evaluation process takes the model calls of its deployment's rollouts.
ROLLOUT_CALL_PATH = "/reef/rollout-calls"
#: What the signature of a ticket covers before its payload, so it signs nothing else.
TICKET_SIGNATURE_CONTEXT = b"reef-rollout-call\n"
#: The header of a call that one evaluation process sends to another: the receiver must hold the rollout itself.
FORWARDED_HEADER = "x-reef-rollout-call-forwarded"
#: The largest call body: an agent's request holds its whole conversation, as for the service (``MAX_REQUEST_BYTES``).
MAX_CALL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CallTarget:
    """Where one rollout's model calls go.

    Values: ``address`` and ``route_port`` are the address and the route's port of the evaluation process that runs
    the rollout; ``proxy_port`` is the port of the rollout's capture proxy on that process's loopback, from 1 to 65535;
    ``key`` is the rollout's random key, which ``RolloutCallRoute.open`` made.
    Purpose: ``RolloutCallRoute`` writes one into each ticket and reads it back from a call.
    """

    address: str
    route_port: int
    proxy_port: int
    key: str


class RolloutCallRoute:
    """The rollout calls of one evaluation process: the tickets of its rollouts, and its server of the route.

    Values: ``call_url`` is the public URL under which the deployment's ingress reaches ``ROLLOUT_CALL_PATH`` of its
    evaluation processes: an ``http`` or ``https`` URL with a host, no query and no fragment, without a final ``/``.
    ``address`` is where the other evaluation processes reach this one, and ``port``, from 1 to 65535, the port on which
    this process serves the route, on every interface. ``tokens`` are the service tokens, at least one non-empty: the
    keys are their SHA-256 digests. ``signing_key`` signs the tickets that this process makes, and ``verifying_keys``
    are the keys of the tickets that it accepts, one for each token, so a token rotation keeps every ticket good.
    ``open_calls`` maps the port of each running rollout's capture proxy to its key, under ``lock``. ``loop`` and
    ``thread`` run the server between ``start`` and ``stop``, and are None otherwise.
    Purpose: the evaluation service makes one from its settings (``from_settings``), and its rollout worker starts and
    stops it. The worker opens a ticket for each rollout whose agent runs in a remote sandbox (``open``) and closes it
    when the rollout ends (``close``).
    """

    def __init__(self, call_url: str, tokens: Iterable[str], *, address: str, port: int) -> None:
        parsed = urllib.parse.urlsplit(call_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("evaluation.call_url must be an http or https URL with a host, without query or fragment")
        accepted = sorted(token for token in tokens if token)
        if not accepted:
            raise ValueError("evaluation.call_url needs the service tokens, which sign the URLs of rollout calls")
        if not address:
            raise ValueError("the address of an evaluation process must be non-empty")
        if not 1 <= port <= 65535:
            raise ValueError("evaluation.call_port must be 1 to 65535")
        self.call_url = call_url.rstrip("/")
        self.address = address
        self.port = port
        self.verifying_keys = tuple(hashlib.sha256(token.encode("utf-8")).digest() for token in accepted)
        self.signing_key = self.verifying_keys[0]
        self.open_calls: dict[int, str] = {}
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    @classmethod
    def from_settings(
        cls, call_url: str | None, tokens: Iterable[str], *, address: str | None, port: int
    ) -> RolloutCallRoute | None:
        """The route of ``evaluation.call_url``, or None when it is unset or empty, as an unset ``${VAR}`` makes it.
        ``tokens`` are the service tokens (``normalize_tokens``) and ``port`` is ``evaluation.call_port``. ``address``
        is ``evaluation.call_address``; None takes the address of this host's name, the Pod's address on Kubernetes."""
        if not call_url:
            return None
        return cls(call_url, tokens, address=address or socket.gethostbyname(socket.gethostname()), port=port)

    @property
    def host(self) -> str:
        """The host name of ``call_url``: the one host that a sandbox's network policy must allow."""
        return str(urllib.parse.urlsplit(self.call_url).hostname)

    def start(self) -> None:
        """Serve the route on ``port`` of every interface, on a thread of its own, until ``stop``; a started route
        stays as it is. ``OSError`` when the port cannot be bound."""
        if self.thread is not None:
            return
        app = web.Application(client_max_size=MAX_CALL_BYTES)
        app.router.add_get("/healthz", self.health)
        app.router.add_route("*", ROLLOUT_CALL_PATH + "/{ticket}/{path:.*}", self.call)
        runner = web.AppRunner(app, access_log=None)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", self.port).start())
        except BaseException:
            loop.run_until_complete(runner.cleanup())
            loop.close()
            raise

        def serve() -> None:
            loop.run_forever()
            loop.run_until_complete(runner.cleanup())
            loop.close()

        self.loop = loop
        self.thread = threading.Thread(target=serve, name="reef-rollout-calls", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Close the port and wait for the server's thread; a route that does not serve stays as it is."""
        loop, thread, self.loop, self.thread = self.loop, self.thread, None, None
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join()

    def open(self, proxy_port: int) -> str:
        """The base URL of the model calls of a rollout whose capture proxy listens on ``proxy_port`` of this
        process's loopback, until ``close(proxy_port)``. Raise ``ValueError`` for a port that a running rollout
        holds."""
        key = secrets.token_hex(16)
        with self.lock:
            if proxy_port in self.open_calls:
                raise ValueError(f"the capture proxy port {proxy_port} has an open rollout call route")
            self.open_calls[proxy_port] = key
        payload = json.dumps({"a": self.address, "r": self.port, "p": proxy_port, "k": key}, separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        return f"{self.call_url}/{encoded}.{self.signature(self.signing_key, encoded)}"

    def close(self, proxy_port: int) -> None:
        """End the calls of the rollout whose capture proxy listens on ``proxy_port``; a closed port stays closed."""
        with self.lock:
            self.open_calls.pop(proxy_port, None)

    def is_open(self, target: CallTarget) -> bool:
        """Whether ``target`` is a rollout of this process that runs."""
        with self.lock:
            return self.open_calls.get(target.proxy_port) == target.key

    def is_local(self, target: CallTarget) -> bool:
        """Whether ``target`` names this process."""
        return target.address == self.address and target.route_port == self.port

    def read(self, ticket: str) -> CallTarget | None:
        """The target of ``ticket``, or None for a ticket that no service token of the deployment signed."""
        encoded, separator, signature = ticket.partition(".")
        if not separator:
            return None
        signed = False
        for key in self.verifying_keys:
            signed |= hmac.compare_digest(signature, self.signature(key, encoded))
        if not signed:
            return None
        try:
            fields = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
            return CallTarget(str(fields["a"]), int(fields["r"]), int(fields["p"]), str(fields["k"]))
        except (binascii.Error, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def signature(key: bytes, encoded: str) -> str:
        return hmac.new(key, TICKET_SIGNATURE_CONTEXT + encoded.encode(), hashlib.sha256).hexdigest()

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def call(self, request: web.Request) -> web.StreamResponse:
        """Send a call whose ticket a service token signed to its rollout's capture proxy, unchanged, and stream its
        answer back.

        A ticket of this process sends the call to the proxy while the rollout runs, and answers 410 after. A ticket of
        another process sends the call to that process's route, once: a call that another process sent here for a
        rollout of a third answers 421. A ticket that no service token signed answers 404, and a process that cannot be
        reached 502.
        """
        ticket, path = request.match_info["ticket"], request.match_info["path"]
        target = self.read(ticket)
        if target is None:
            raise web.HTTPNotFound(text="no rollout call has this URL")
        headers = {name: value for name, value in request.headers.items() if name.lower() not in HOP_BY_HOP}
        if self.is_local(target):
            if not self.is_open(target):
                raise web.HTTPGone(text="the rollout of this call ended")
            headers.pop(FORWARDED_HEADER, None)
            url = f"http://127.0.0.1:{target.proxy_port}/{path}"
        elif FORWARDED_HEADER in request.headers:
            raise web.HTTPMisdirectedRequest(text="the rollout of this call runs in another evaluation process")
        else:
            host = f"[{target.address}]" if ":" in target.address else target.address
            url = f"http://{host}:{target.route_port}{ROLLOUT_CALL_PATH}/{ticket}/{path}"
            headers[FORWARDED_HEADER] = "1"
        if request.query_string:
            url = f"{url}?{request.query_string}"
        body = await request.read()
        # The capture proxy and the endpoint behind it bound the call; a stream may be silent while the model thinks.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        try:
            async with (
                aiohttp.ClientSession(auto_decompress=False, timeout=timeout) as session,
                session.request(request.method, url, headers=headers, data=body, allow_redirects=False) as upstream,
            ):
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={n: v for n, v in upstream.headers.items() if n.lower() not in HOP_BY_HOP},
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except aiohttp.ClientConnectionError as error:
            raise web.HTTPBadGateway(text=f"the evaluation process of this call did not answer: {error}") from error


__all__ = ["FORWARDED_HEADER", "ROLLOUT_CALL_PATH", "CallTarget", "RolloutCallRoute"]
