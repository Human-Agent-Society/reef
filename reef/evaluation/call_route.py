"""The public route of the model calls of evaluation rollouts whose agent runs in a remote sandbox.

An agent that runs in a task's container in a remote environment calls its rollout's capture proxy at a signed URL
under ``evaluation.call_url``: ``<call_url>/<ticket>/v1/...``. The ticket names the evaluation process that runs the
rollout (its address and port), the port of the rollout's capture proxy in that process and a key of the rollout. Its
signature is keyed by the deployment's service tokens, so every process of the deployment checks it, and no one
without a token can make one. The deployment's ingress sends ``<call_url>/`` to ``ROLLOUT_CALL_PATH`` of any of its
evaluation processes. That process sends the call to the process that the ticket names, which sends it to the capture
proxy on its loopback while the rollout runs. The sandbox holds no credential: the proxy adds Reef's token.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class CallTarget:
    """Where one rollout's model calls go.

    Values: ``address`` and ``server_port`` are the address and HTTP port of the evaluation process that runs the
    rollout; ``proxy_port`` is the port of the rollout's capture proxy on that process's loopback, from 1 to 65535;
    ``key`` is the rollout's random key, which ``RolloutCallRoute.open`` made.
    Purpose: ``RolloutCallRoute`` writes one into each ticket and reads it back from a call.
    """

    address: str
    server_port: int
    proxy_port: int
    key: str


class RolloutCallRoute:
    """The tickets of the rollouts that this evaluation process runs, and the check of a ticket.

    Values: ``call_url`` is the public URL under which the deployment's ingress reaches ``ROLLOUT_CALL_PATH`` of its
    evaluation processes: an ``http`` or ``https`` URL with a host, no query and no fragment, without a final ``/``.
    ``address`` and ``server_port`` are where the other evaluation processes reach this one. ``tokens`` are the
    service tokens, at least one non-empty: the keys are their SHA-256 digests. ``signing_key`` is the
    key of the tickets that this process makes, and ``verifying_keys`` the keys of the tickets it accepts: one for
    each service token, so a token rotation keeps the tickets of every process good. ``open_calls`` maps the port of
    each running rollout's capture proxy to its key; ``lock`` guards it.
    Purpose: the evaluation service makes one from its settings (``from_settings``). Its rollout worker opens a ticket
    for each rollout whose agent runs in a remote sandbox (``open``) and closes it when the rollout ends (``close``);
    the route (``register_rollout_call_route``) sends each call to the rollout's capture proxy.
    """

    def __init__(self, call_url: str, tokens: Iterable[str], *, address: str, server_port: int) -> None:
        parsed = urllib.parse.urlsplit(call_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("evaluation.call_url must be an http or https URL with a host, without query or fragment")
        accepted = sorted(token for token in tokens if token)
        if not accepted:
            raise ValueError("evaluation.call_url needs the service tokens, which sign the URLs of rollout calls")
        if not address:
            raise ValueError("the address of an evaluation process must be non-empty")
        if not 1 <= server_port <= 65535:
            raise ValueError("the port of an evaluation process must be 1 to 65535")
        self.call_url = call_url.rstrip("/")
        self.address = address
        self.server_port = server_port
        self.verifying_keys = tuple(hashlib.sha256(token.encode("utf-8")).digest() for token in accepted)
        self.signing_key = self.verifying_keys[0]
        self.open_calls: dict[int, str] = {}
        self.lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, call_url: str | None, tokens: Iterable[str], *, address: str | None, server_port: int
    ) -> RolloutCallRoute | None:
        """The route of ``evaluation.call_url``, or None without one. ``tokens`` are the service tokens
        (``normalize_tokens``). ``address`` is ``evaluation.call_address``; None takes the address of this host's
        name, which is the Pod's address on Kubernetes."""
        if call_url is None:
            return None
        return cls(
            call_url, tokens, address=address or socket.gethostbyname(socket.gethostname()), server_port=server_port
        )

    @property
    def host(self) -> str:
        """The host name of ``call_url``: the one host that a sandbox's network policy must allow."""
        return str(urllib.parse.urlsplit(self.call_url).hostname)

    def open(self, proxy_port: int) -> str:
        """The base URL of the model calls of a rollout whose capture proxy listens on ``proxy_port`` of this
        process's loopback, until ``close(proxy_port)``. Raise ``ValueError`` for a port that a running rollout
        holds."""
        key = secrets.token_hex(16)
        with self.lock:
            if proxy_port in self.open_calls:
                raise ValueError(f"the capture proxy port {proxy_port} has an open rollout call route")
            self.open_calls[proxy_port] = key
        payload = json.dumps(
            {"a": self.address, "s": self.server_port, "p": proxy_port, "k": key}, separators=(",", ":")
        ).encode()
        encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
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
        return target.address == self.address and target.server_port == self.server_port

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
            return CallTarget(str(fields["a"]), int(fields["s"]), int(fields["p"]), str(fields["k"]))
        except (binascii.Error, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def signature(key: bytes, encoded: str) -> str:
        return hmac.new(key, TICKET_SIGNATURE_CONTEXT + encoded.encode(), hashlib.sha256).hexdigest()


def register_rollout_call_route(app: web.Application, route: RolloutCallRoute) -> None:
    """Add ``ROLLOUT_CALL_PATH/{ticket}/{path}`` to ``app``: a call whose ticket a service token signed goes to its
    rollout's capture proxy, unchanged, and its answer streams back.

    A ticket of this process sends the call to the proxy while the rollout runs, and answers 410 after. A ticket of
    another process sends the call to that process, once: a call that another process sent here for a rollout of a
    third answers 421. A ticket that no service token signed answers 404, and a process that cannot be reached 502.
    The authentication middleware lets these paths through without a token (``is_rollout_call``).
    """

    async def call(request: web.Request) -> web.StreamResponse:
        ticket, path = request.match_info["ticket"], request.match_info["path"]
        target = route.read(ticket)
        if target is None:
            raise web.HTTPNotFound(text="no rollout call has this URL")
        headers = {name: value for name, value in request.headers.items() if name.lower() not in HOP_BY_HOP}
        if route.is_local(target):
            if not route.is_open(target):
                raise web.HTTPGone(text="the rollout of this call ended")
            headers.pop(FORWARDED_HEADER, None)
            url = f"http://127.0.0.1:{target.proxy_port}/{path}"
        elif FORWARDED_HEADER in request.headers:
            raise web.HTTPMisdirectedRequest(text="the rollout of this call runs in another evaluation process")
        else:
            host = f"[{target.address}]" if ":" in target.address else target.address
            url = f"http://{host}:{target.server_port}{ROLLOUT_CALL_PATH}/{ticket}/{path}"
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

    app.router.add_route("*", ROLLOUT_CALL_PATH + "/{ticket}/{path:.*}", call)


def is_rollout_call(path: str) -> bool:
    """Whether ``path`` is under ``ROLLOUT_CALL_PATH``: its route checks the ticket of the call, not a token."""
    return path.startswith(ROLLOUT_CALL_PATH + "/")


__all__ = [
    "FORWARDED_HEADER",
    "ROLLOUT_CALL_PATH",
    "CallTarget",
    "RolloutCallRoute",
    "is_rollout_call",
    "register_rollout_call_route",
]
