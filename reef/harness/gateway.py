"""The loopback gateway an agent process reaches its model and its tools through.

The agent holds one URL and nothing else: no provider key and no Reef token.
The URL's path carries a random token, so the agent and every program it
starts reach the gateway with no credential of their own. Behind it:

- the model routes (``MODEL_PATHS``) go to one model binding with its key;
  each call spends one call of a ``ModelCalls`` budget and lands in its
  record, and the request's ``model`` is always the binding's;
- ``GET /tools`` lists the tools (``AgentTool``), each with its input
  schema, and ``POST /<tool>`` calls one with a JSON object of arguments;
- everything else is 404.

A subclass serves more routes by overriding ``post`` or ``get``.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

from reef.harness.rollouts.model_binding import ModelBinding
from reef.inference.provider_protocol import (
    ANTHROPIC_CLIENT_HEADERS,
    ANTHROPIC_PATHS,
    provider_headers,
    response_events,
    usage_of,
)

logger = logging.getLogger(__name__)

MODEL_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
#: The most of one reply the gateway keeps to record; the client always gets all of it.
MAX_RECORDED_BYTES = 2 * 1024 * 1024


class ModelCalls(ABC):
    """A budget of model calls, their record, and the live activity of what calls them.

    Purpose: a harness evolution step's model calls spend and record through
    one (``Tools.calls``): its budgeted bindings, and a gateway that relays an
    agent process's calls.
    """

    @abstractmethod
    def spend(self) -> None:
        """Count one call; raise ``RuntimeError`` when the budget is exhausted."""

    @abstractmethod
    def record(self, entry: Mapping[str, Any]) -> None:
        """Append one call to the call record, which redacts and clips its texts."""

    @abstractmethod
    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        """Add one line to the live activity: ``kind`` names what acted (such as ``model``, ``agent`` or
        ``evaluate``), ``text`` what it did, and ``failed`` marks a line that went wrong."""


class AgentTool(ABC):
    """One tool the gateway serves: ``POST /<name>`` with a JSON object of arguments.

    Values: ``name`` is one route segment that is not ``v1`` or ``tools``.
    ``description`` says what the tool does and what it answers, and
    ``input_schema`` is the JSON schema of its arguments.
    Purpose: an agent run gives the gateway its tools; ``GET /tools`` lists
    them. The gateway serves each request on its own thread, so a tool whose
    calls must take turns holds a lock of its own.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[Mapping[str, Any]]

    @abstractmethod
    def call(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """The tool's answer as a JSON object; ``ValueError`` or ``RuntimeError`` refuses the call with its message."""


class AgentGateway:
    """A loopback HTTP server for one agent run; ``base_url`` is the only thing the agent is given.

    Values: ``binding`` is the model every model route reaches; ``calls`` is
    the step's budget, call record and activity; ``tools`` are the tools
    the gateway serves by name. ``address`` is the loopback address it
    listens on and ``token`` the random path segment every route needs.
    ``server`` is the running server, ``None`` before ``start`` and after
    ``stop``.
    Purpose: an agent run starts one and stops it when the agent ends.
    """

    def __init__(
        self,
        binding: ModelBinding,
        calls: ModelCalls,
        tools: Sequence[AgentTool] = (),
        *,
        host: str = "127.0.0.1",
    ) -> None:
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names) or {"v1", "tools"} & set(names):
            raise ValueError("gateway tools need distinct names other than v1 and tools")
        self.binding = binding
        self.calls = calls
        self.tools = {tool.name: tool for tool in tools}
        self.address = host
        self.token = secrets.token_urlsafe(24)
        self.server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        if self.server is None:
            raise RuntimeError("the agent gateway is not running")
        return int(self.server.server_address[1])

    @property
    def base_url(self) -> str:
        """The address the agent uses in place of Reef's: no ``/v1`` suffix."""
        return f"http://{self.address}:{self.port}/t/{self.token}"

    def start(self) -> None:
        server = ThreadingHTTPServer((self.address, 0), self.handler_class())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="reef-agent-gateway", daemon=True).start()
        self.server = server

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        gateway = self
        prefix = f"/t/{self.token}"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if not secrets.compare_digest(path[: len(prefix)], prefix):
                    respond(self, 404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    respond(self, 400, {"error": "the body must be JSON"})
                    return
                if not isinstance(payload, dict):
                    respond(self, 400, {"error": "the body must be a JSON object"})
                    return
                gateway.post(self, path[len(prefix) :], payload)

            def do_GET(self) -> None:
                path, _, query = self.path.partition("?")
                if not secrets.compare_digest(path[: len(prefix)], prefix):
                    respond(self, 404, {"error": "not found"})
                    return
                gateway.get(self, path[len(prefix) :], query)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("agent gateway: " + format, *args)

        return Handler

    def post(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        """Answer ``POST <route>``: a model call or a tool call; 404 for anything else."""
        if route in MODEL_PATHS:
            self.relay_model_call(client, route, payload)
        elif route.removeprefix("/") in self.tools:
            self.call_tool(client, self.tools[route.removeprefix("/")], payload)
        else:
            respond(client, 404, {"error": f"the gateway serves no {route}"})

    def get(self, client: BaseHTTPRequestHandler, route: str, query: str) -> None:
        """Answer ``GET <route>``: the tool list; 404 for anything else."""
        if route != "/tools":
            respond(client, 404, {"error": "not found"})
            return
        listed = [
            {"name": tool.name, "description": tool.description, "input_schema": dict(tool.input_schema)}
            for tool in self.tools.values()
        ]
        respond(client, 200, {"tools": listed})

    def call_tool(self, client: BaseHTTPRequestHandler, tool: AgentTool, arguments: Mapping[str, Any]) -> None:
        try:
            answer = tool.call(arguments)
        except (ValueError, RuntimeError) as exc:
            respond(client, 400, {"error": str(exc)})
            return
        respond(client, 200, answer)

    def relay_model_call(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        try:
            self.calls.spend()
        except RuntimeError as exc:
            refuse(client, 429, str(exc))
            return
        # The binding's model is the only one the agent reaches: a request naming another is sent to it anyway.
        payload = {**payload, "model": self.binding.model}
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            **provider_headers(route, self.binding.api_key),
        }
        if route in ANTHROPIC_PATHS:
            # The agent's own API version and beta features win over the default version.
            for name in ANTHROPIC_CLIENT_HEADERS:
                value = client.headers.get(name)
                if value:
                    headers[name] = value
        url = f"{self.binding.base_url.rstrip('/')}{route}"
        started = time.monotonic()
        status, body, _ = relay(client, url, headers, payload, timeout_s=self.binding.timeout_s)
        text = body.decode("utf-8", errors="replace")
        entry: dict[str, Any] = {
            "source": "agent",
            "model": self.binding.model,
            "path": route,
            "status": status,
            "messages": len(payload.get("messages") or payload.get("input") or []),
            "last_message": (payload.get("messages") or [None])[-1],
            "reply": reply_text(text),
            "seconds": round(time.monotonic() - started, 3),
        }
        usage = sse_usage(text)
        if usage is not None:
            entry["usage"] = usage
        self.calls.record(entry)
        # What the agent decided this turn, for the step page: each tool it called, else what it said.
        if status >= 400:
            self.calls.note(
                "agent", f"{self.binding.model} answered {status}: {entry['reply'] or text[:200]}", failed=True
            )
            return
        called = reply_tool_calls(text)
        for name, arguments in called:
            self.calls.note("agent", tool_summary(name, arguments))
        said = entry["reply"].strip()
        if said and not called:
            self.calls.note("agent", said.splitlines()[0])


def respond(client: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
    """Answer ``client`` with ``body`` as JSON."""
    data = json.dumps(body, default=str).encode()
    client.send_response(status)
    client.send_header("Content-Type", "application/json")
    client.send_header("Content-Length", str(len(data)))
    client.end_headers()
    client.wfile.write(data)


def refuse(client: BaseHTTPRequestHandler, status: int, message: str) -> None:
    """Refuse a model call as a provider would: ``{"error": {"message": ...}}``."""
    respond(client, status, {"error": {"message": message}})


def relay(
    client: BaseHTTPRequestHandler,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    timeout_s: float,
) -> tuple[int, bytes, dict[str, str]]:
    """Forward ``payload`` and stream the answer to ``client`` as it arrives: the status, what was kept of the
    body for the record, and the upstream headers."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=dict(headers), method="POST")
    try:
        upstream = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as error:
        # A refusal is short and whole: relay it as the provider wrote it.
        body = error.read()
        upstream_headers = dict(error.headers.items())
        client.send_response(error.code)
        client.send_header("Content-Type", error.headers.get("Content-Type") or "application/json")
        client.send_header("Content-Length", str(len(body)))
        client.end_headers()
        client.wfile.write(body)
        return error.code, body[:MAX_RECORDED_BYTES], upstream_headers
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        refuse(client, 502, f"the upstream did not answer: {error}")
        return 502, b"", {}
    kept = bytearray()
    with upstream:
        upstream_headers = dict(upstream.headers.items())
        client.send_response(upstream.status)
        for name in ("Content-Type", "Content-Disposition"):
            if upstream.headers.get(name):
                client.send_header(name, upstream.headers[name])
        client.end_headers()
        while True:
            # read1 returns what has arrived, so a streamed reply reaches the agent as it is written.
            chunk = upstream.read1(65536)
            if not chunk:
                break
            client.wfile.write(chunk)
            client.wfile.flush()
            if len(kept) < MAX_RECORDED_BYTES:
                kept.extend(chunk)
    return upstream.status, bytes(kept), upstream_headers


def reply_text(text: str) -> str:
    """The text a model reply carries, whole or streamed, in the chat, Anthropic or Responses dialect."""
    parts: list[str] = []
    for event in response_events(text):
        for choice in event.get("choices") or ():
            if isinstance(choice, dict):
                piece = (choice.get("delta") or choice.get("message") or {}).get("content")
                if isinstance(piece, str):
                    parts.append(piece)
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            parts.append(delta["text"])
        elif event.get("type") == "response.output_text.delta" and isinstance(delta, str):
            parts.append(delta)
        parts.extend(
            block["text"]
            for block in event.get("content") or ()
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return "".join(parts)[:20_000]


def reply_tool_calls(text: str) -> list[tuple[str, str]]:
    """The tools a model reply called, whole or streamed, in the chat, Anthropic or Responses dialect: each
    ``(name, arguments as JSON text)``, in order."""
    calls: dict[tuple[str, Any], list[str]] = {}
    for event in response_events(text):
        # Chat Completions: tool_calls on the message, or their pieces on each delta, keyed by index.
        for choice in event.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            holder = choice.get("delta") or choice.get("message") or {}
            for position, call in enumerate(holder.get("tool_calls") or ()):
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                slot = calls.setdefault(("chat", call.get("index", position)), ["", ""])
                slot[0] += function.get("name") or ""
                arguments = function.get("arguments")
                slot[1] += arguments if isinstance(arguments, str) else json.dumps(arguments or {})
        # Anthropic: tool_use blocks whole, or started and filled by input_json_delta.
        block = event.get("content_block")
        if event.get("type") == "content_block_start" and isinstance(block, dict) and block.get("type") == "tool_use":
            calls[("anthropic", event.get("index"))] = [str(block.get("name") or ""), ""]
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            started = calls.get(("anthropic", event.get("index")))
            if started is not None:
                started[1] += str(delta.get("partial_json") or "")
        for position, part in enumerate(event.get("content") or ()):
            if isinstance(part, dict) and part.get("type") == "tool_use":
                calls[("anthropic-whole", position)] = [
                    str(part.get("name") or ""),
                    json.dumps(part.get("input") or {}),
                ]
        # Responses: a finished function_call item carries its whole arguments.
        items = [event.get("item")] if event.get("type") == "response.output_item.done" else []
        items += list(event.get("output") or ()) if event.get("object") == "response" else []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = ("responses", item.get("call_id") or item.get("id") or len(calls))
                calls[key] = [str(item.get("name") or ""), str(item.get("arguments") or "")]
    return [(name, arguments) for name, arguments in calls.values() if name]


#: The argument that says what a tool call did, in the order a summary looks for it.
SUMMARY_ARGUMENTS = ("path", "file_path", "command", "task", "url", "pattern", "query")


def tool_summary(name: str, arguments: str) -> str:
    """One line for a tool call: its name and the argument that says what it did (a path, a command, a task)."""
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        parsed = arguments
    detail = ""
    if isinstance(parsed, dict):
        named = [parsed[key] for key in SUMMARY_ARGUMENTS if isinstance(parsed.get(key), str)]
        strings = named or [value for value in parsed.values() if isinstance(value, str)]
        detail = strings[0] if strings else ""
    elif isinstance(parsed, str):
        detail = parsed
    detail = " ".join(detail.split())
    return f"{name} {detail[:160]}".strip()


def sse_usage(text: str) -> dict[str, int] | None:
    """Input and output tokens a reply reported, whole or streamed; ``None`` when it reported none."""
    input_tokens = output_tokens = 0
    seen = False
    for event in response_events(text):
        for holder in (event, event.get("message") if isinstance(event.get("message"), dict) else None):
            if holder is None or not isinstance(holder.get("usage"), dict):
                continue
            usage = usage_of(holder)
            if usage is None:
                continue
            seen = True
            input_tokens = max(input_tokens, usage.get("input_tokens", 0))
            output_tokens = max(output_tokens, usage.get("output_tokens", 0))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens} if seen else None


__all__ = [
    "MODEL_PATHS",
    "AgentGateway",
    "AgentTool",
    "ModelCalls",
    "refuse",
    "relay",
    "reply_text",
    "reply_tool_calls",
    "respond",
    "sse_usage",
    "tool_summary",
]
