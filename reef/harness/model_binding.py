"""The model bindings Reef hands to harness-evolution methods and episodes.

A :class:`ModelBinding` is one model endpoint as a value: where it is, which
model to request, how to authenticate, and which API dialect it speaks. The
recipe derives the served model's binding from its inference runtime, so the
URL and credential come from the deployment config, never from the method's
own environment reads; a method's auxiliary models (a stronger proposer, a
judge) are declared in the recipe config and resolved the same way.

:class:`ModelBindings` is the set a method receives: ``served`` is the model
under test - what the HTTP service proxies to and what evaluation episodes
run against - and the named ones are the method's own. Methods call
:meth:`ModelBinding.chat`; the evolution backend renders
:meth:`ModelBinding.compose_nodes` into each evaluation episode, through
:func:`episode_model_binding` when the adapter must keep upstream credentials
out of the harness process. Neither path goes through the HTTP service, so
none of this traffic becomes a scenario record.
"""

from __future__ import annotations

import contextlib
import hmac
import http.client
import json
import secrets
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from reef.core.errors import ReefError
from reef.harness.descriptor import AdapterDescriptor
from reef.runtime.base import InferenceRuntime

#: The API dialects a binding can speak. ``openai`` is Chat Completions,
#: ``responses`` is OpenAI Responses, and ``anthropic`` is Messages.
MODEL_APIS = ("openai", "responses", "anthropic")
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_MAX_PROXY_REQUEST_BYTES = 64 * 1024 * 1024


class ModelBindingError(ReefError):
    """A model call failed; ``status`` carries the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class ModelBinding:
    """One model endpoint plus the model name to request from it.

    ``base_url`` carries no ``/v1`` suffix; the request paths add it.
    """

    base_url: str
    model: str
    api_key: str | None = None
    api: str = "openai"
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("model binding requires a base_url")
        if not self.model:
            raise ValueError("model binding requires a model name")
        if self.api not in MODEL_APIS:
            raise ValueError(f"model binding api must be one of {MODEL_APIS}, got {self.api!r}")
        if self.timeout_s <= 0:
            raise ValueError("model binding timeout_s must be positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_runtime(cls, runtime: InferenceRuntime, *, model: str | None = None) -> ModelBinding:
        """Bind to ``runtime``'s endpoint; ``model`` overrides its model path."""

        name = model or getattr(runtime, "model_path", "") or ""
        if not name:
            raise ValueError("model binding requires a model name: set reef.upstream_model or the recipe's model.path")
        return cls(
            base_url=runtime.base_url,
            model=name,
            api_key=getattr(runtime, "api_key", None),
            api=getattr(runtime, "api", "openai") or "openai",
            timeout_s=float(runtime.inference_timeout_s),
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        environ: Mapping[str, str],
        *,
        where: str = "model",
    ) -> ModelBinding:
        """A binding from a config section: ``url``, ``model``, optional
        ``api`` (default ``openai``), ``timeout_s``, and the credential as a
        literal ``api_key`` or the name of an environment variable in
        ``api_key_env`` (so the key itself stays out of the file)."""

        def text(key: str, *, required: bool = True) -> str | None:
            value = config.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{where}.{key} must be a non-empty string")
            return value.strip()

        api_key = text("api_key", required=False)
        env_name = text("api_key_env", required=False)
        if api_key is None and env_name is not None:
            api_key = environ.get(env_name, "").strip() or None
        timeout_raw = config.get("timeout_s", 600.0)
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.timeout_s must be a number") from exc
        try:
            return cls(
                base_url=text("url") or "",
                model=text("model") or "",
                api_key=api_key,
                api=text("api", required=False) or "openai",
                timeout_s=timeout_s,
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

    # -- Method-side calls ---------------------------------------------------

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        """One chat completion; returns the assistant text.

        ``messages`` use the OpenAI role shape (``system`` / ``user`` /
        ``assistant``) whatever the binding's API; the request is translated
        for the dialect. ``params`` are request fields (``temperature``,
        ``max_tokens``, ``stream``...) passed through. ``timeout_s`` overrides
        the binding's timeout for this call.
        """

        if self.api == "responses":
            responses_params = dict(params)
            if "max_tokens" in responses_params:
                if "max_output_tokens" in responses_params:
                    raise ValueError("responses chat cannot set both max_tokens and max_output_tokens")
                responses_params["max_output_tokens"] = responses_params.pop("max_tokens")
            response = self.complete(
                {"input": [dict(message) for message in messages], **responses_params}, timeout_s=timeout_s
            )
            pieces = [
                content.get("text", "")
                for item in response.get("output", ())
                if isinstance(item, Mapping) and item.get("type") == "message" and item.get("role") == "assistant"
                for content in item.get("content", ())
                if isinstance(content, Mapping) and content.get("type") == "output_text"
            ]
            if not all(isinstance(piece, str) for piece in pieces) or not pieces:
                raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600])
            return "".join(pieces)

        if self.api == "anthropic":
            system = "\n\n".join(
                str(message.get("content", "")) for message in messages if message.get("role") == "system"
            )
            body: dict[str, Any] = {
                "max_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS,
                "messages": [dict(message) for message in messages if message.get("role") != "system"],
                **params,
            }
            if system:
                body["system"] = system
            response = self.complete(body, timeout_s=timeout_s)
            try:
                return "".join(
                    str(block.get("text", "")) for block in response["content"] if block.get("type") == "text"
                )
            except (KeyError, TypeError, AttributeError) as exc:
                raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc

        response = self.complete({"messages": [dict(message) for message in messages], **params}, timeout_s=timeout_s)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc
        if not isinstance(content, str):
            raise ModelBindingError("model endpoint returned non-text content")
        return content

    def complete(self, body: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """POST one request in the binding's native dialect and return the
        response object: Chat Completions for ``openai``, Responses for
        ``responses``, and Messages for ``anthropic``. ``model`` defaults to this binding's. A streaming
        request is read to the end and folded into the non-streaming response
        shape, so callers see one contract either way. Prefer :meth:`chat`
        unless the method needs the raw response.
        """

        request_body = {"model": self.model, **body}
        headers = {"content-type": "application/json"}
        if self.api == "anthropic":
            path = "/v1/messages"
            headers["anthropic-version"] = ANTHROPIC_VERSION
            if self.api_key:
                headers["x-api-key"] = self.api_key
        else:
            path = "/v1/responses" if self.api == "responses" else "/v1/chat/completions"
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                if request_body.get("stream"):
                    if self.api == "anthropic":
                        return _fold_anthropic_stream(response)
                    if self.api == "responses":
                        return _fold_responses_stream(response)
                    return _fold_stream(response)
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")
            except Exception:
                detail = ""
            raise ModelBindingError(
                f"model endpoint answered {exc.code}: {detail[:400]}",
                status=exc.code,
                detail=detail,
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelBindingError(f"model endpoint unreachable: {exc}") from exc

    # -- Episode-side rendering ----------------------------------------------

    def compose_nodes(self, descriptor: AdapterDescriptor) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        """``config`` nodes that point ``descriptor``'s harness at this binding."""

        templates = descriptor.model_binding.get(self.api)
        if not templates:
            known = ", ".join(sorted(descriptor.model_binding)) or "none"
            raise ModelBindingError(
                f"adapter {descriptor.name!r} declares no model_binding for the {self.api!r} api "
                f"(declared: {known}); episodes cannot reach a model"
            )
        if descriptor.model_binding_proxy and not isinstance(self, _ProxiedModelBinding):
            raise ModelBindingError(
                f"adapter {descriptor.name!r} requires a temporary model proxy; "
                "the upstream credential may not be rendered directly"
            )
        values = {"base_url": self.base_url, "api_key": self.api_key or "", "model": self.model}
        return tuple(("config", _substitute(node, values)) for node in templates)


class _ProxiedModelBinding(ModelBinding):
    """A loopback binding carrying only its proxy's short-lived capability."""


@dataclass(frozen=True)
class ModelBindings(Mapping[str, ModelBinding]):
    """The models a method may call: ``served`` plus the method's named ones.

    ``served`` is the model under test - the one the HTTP service proxies to
    and evaluation episodes run against. The named bindings come from the
    recipe's ``evolution.models`` section; ``models["teacher"]`` looks one up
    and raises :class:`KeyError` naming what is declared when it is missing.
    """

    served: ModelBinding
    named: Mapping[str, ModelBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "served" in self.named:
            raise ValueError("'served' is reserved for the model under test; name the extra model differently")

    def __getitem__(self, name: str) -> ModelBinding:
        if name == "served":
            return self.served
        try:
            return self.named[name]
        except KeyError:
            declared = ", ".join(sorted(self.named)) or "none"
            raise KeyError(f"no model named {name!r}; declared under evolution.models: {declared}") from None

    def __iter__(self) -> Iterator[str]:
        yield "served"
        yield from self.named

    def __len__(self) -> int:
        return 1 + len(self.named)


def _api_path(api: str) -> str:
    if api == "anthropic":
        return "/v1/messages"
    if api == "responses":
        return "/v1/responses"
    return "/v1/chat/completions"


def _upstream_headers(binding: ModelBinding, accept: str = "") -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if accept:
        headers["accept"] = accept
    if binding.api == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if binding.api_key:
            headers["x-api-key"] = binding.api_key
    elif binding.api_key:
        headers["authorization"] = f"Bearer {binding.api_key}"
    return headers


class _ModelProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, binding: ModelBinding, capability: str) -> None:
        super().__init__(("127.0.0.1", 0), _ModelProxyHandler)
        self.binding = binding
        self.capability = capability
        self.closing = threading.Event()
        self._connections: set[http.client.HTTPConnection] = set()
        self._connections_lock = threading.Lock()

    def track(self, connection: http.client.HTTPConnection) -> bool:
        with self._connections_lock:
            if self.closing.is_set():
                return False
            self._connections.add(connection)
            return True

    def release(self, connection: http.client.HTTPConnection) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def may_request(self) -> bool:
        """Check teardown after connect and before the one allowed request."""
        with self._connections_lock:
            return not self.closing.is_set()

    def close_upstreams(self) -> None:
        """Stop active paid calls before the proxy context returns."""

        with self._connections_lock:
            self.closing.set()
            connections = tuple(self._connections)
        for connection in connections:
            sock = connection.sock
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)


class _ModelProxyHandler(BaseHTTPRequestHandler):
    server: _ModelProxyServer

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def do_POST(self) -> None:
        binding = self.server.binding
        expected = f"Bearer {self.server.capability}"
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, expected):
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            return
        path = _api_path(binding.api)
        if self.path != path:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(400)
            return
        if size < 0 or size > _MAX_PROXY_REQUEST_BYTES:
            self.send_error(413)
            return
        try:
            body = json.loads(self.rfile.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400)
            return
        if not isinstance(body, dict):
            self.send_error(400)
            return
        # The transient binding owns the served model as well as the endpoint.
        # Never let a harness request another upstream model through the proxy.
        body["model"] = binding.model
        url = urllib.parse.urlsplit(binding.base_url)
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        if url.scheme not in ("http", "https") or url.hostname is None:
            self.send_error(502)
            return
        connection = connection_type(url.hostname, url.port, timeout=binding.timeout_s)
        if not self.server.track(connection):
            connection.close()
            return
        try:
            # Connect may block before ``connection.sock`` exists, so teardown
            # cannot interrupt that phase. Re-enter the server's lock after it
            # completes: a context that already closed then refuses the HTTP
            # request instead of starting a paid call late.
            connection.connect()
            target = f"{url.path.rstrip('/')}{path}"
            if not self.server.may_request():
                return
            # Teardown shuts down, but does not clear, this already-connected
            # socket. A race here therefore fails the send instead of letting
            # HTTPConnection reconnect after the proxy context has closed.
            connection.request(
                "POST",
                target,
                body=json.dumps(body).encode("utf-8"),
                headers=_upstream_headers(binding, self.headers.get("Accept", "")),
            )
            upstream = connection.getresponse()
            self.send_response(upstream.status)
            content_type = upstream.getheader("Content-Type")
            if content_type:
                self.send_header("Content-Type", content_type)
            content_encoding = upstream.getheader("Content-Encoding")
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while chunk := upstream.read(64 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (http.client.HTTPException, OSError, ValueError):
            if not self.server.closing.is_set():
                with contextlib.suppress(OSError):
                    self.send_error(502)
        finally:
            self.server.release(connection)
            connection.close()
        self.close_connection = True


@contextlib.contextmanager
def temporary_model_proxy(binding: ModelBinding) -> Iterator[ModelBinding]:
    """Expose ``binding`` on loopback without exposing its credential.

    The proxy lives only for the surrounding evaluation, injects the real
    upstream authentication from Reef's process, and pins every request to
    the served model. Harness config receives only the returned short-lived
    capability, so reading it cannot disclose the upstream credential.
    """

    try:
        capability = secrets.token_urlsafe(32)
        server = _ModelProxyServer(binding, capability)
    except OSError as exc:
        raise ModelBindingError(f"cannot start the temporary model proxy: {exc}") from exc
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        name="reef-model-proxy",
        daemon=True,
    )
    thread.start()
    address = server.server_address
    host, port = str(address[0]), int(address[1])
    try:
        yield _ProxiedModelBinding(
            base_url=f"http://{host}:{port}",
            model=binding.model,
            api_key=capability,
            api=binding.api,
            timeout_s=binding.timeout_s,
        )
    finally:
        server.close_upstreams()
        server.shutdown()
        server.server_close()
        thread.join()


@contextlib.contextmanager
def episode_model_binding(binding: ModelBinding, descriptor: AdapterDescriptor) -> Iterator[ModelBinding]:
    """Return the binding an evaluation episode may safely render."""

    if descriptor.model_binding_proxy:
        with temporary_model_proxy(binding) as proxied:
            yield proxied
        return
    yield binding


def _substitute(value: Any, values: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, Mapping):
        return {_substitute(key, values): _substitute(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, values) for item in value]
    return value


def _sse_payloads(response: Any) -> Iterator[Any]:
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _fold_stream(response: Any) -> dict[str, Any]:
    """Fold an SSE chat-completions stream into one response object."""

    pieces: list[str] = []
    role = "assistant"
    finish_reason = None
    model = None
    for chunk in _sse_payloads(response):
        model = chunk.get("model", model)
        for choice in chunk.get("choices", ()):
            delta = choice.get("delta", {})
            role = delta.get("role", role)
            content = delta.get("content")
            if isinstance(content, str):
                pieces.append(content)
            finish_reason = choice.get("finish_reason", finish_reason)
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": "".join(pieces)},
                "finish_reason": finish_reason,
            }
        ],
    }


def _fold_anthropic_stream(response: Any) -> dict[str, Any]:
    """Fold an SSE messages stream into one response object."""

    pieces: list[str] = []
    model = None
    stop_reason = None
    for event in _sse_payloads(response):
        kind = event.get("type")
        if kind == "message_start":
            model = event.get("message", {}).get("model", model)
        elif kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                pieces.append(delta["text"])
        elif kind == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
    return {
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "".join(pieces)}],
        "stop_reason": stop_reason,
    }


def _fold_responses_stream(response: Any) -> dict[str, Any]:
    """Fold an SSE Responses stream into one response object."""

    pieces: list[str] = []
    completed: dict[str, Any] | None = None
    for event in _sse_payloads(response):
        if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str):
            pieces.append(event["delta"])
        elif event.get("type") == "response.completed" and isinstance(event.get("response"), dict):
            completed = event["response"]
    if completed is not None:
        return completed
    return {
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(pieces)}],
            }
        ],
    }


__all__ = ["MODEL_APIS", "ModelBinding", "ModelBindingError", "ModelBindings"]
