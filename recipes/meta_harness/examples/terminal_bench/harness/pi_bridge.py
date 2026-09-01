"""Run Pi locally while delegating its only terminal tool to Harbor."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import secrets
import signal
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from reef.harness import AdapterDescriptor
from reef.harness.trajectory import reader_for

from .config import TARGET_MODEL_PRICING, TokenPricing

BRIDGE_EXTENSION_PATH = "pi-agent/extensions/zzzz-reef-harbor-bridge.ts"
MAX_REQUEST_BYTES = 1_048_576
BRIDGE_EXTENSION = r"""import {
  createBashToolDefinition,
  type BashOperations,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

const endpoint = process.env.REEF_HARBOR_BRIDGE_URL;
const token = process.env.REEF_HARBOR_BRIDGE_TOKEN;
if (!endpoint || !token) throw new Error("Reef Harbor bridge environment is missing");

const operations: BashOperations = {
  async exec(command, _cwd, { onData, signal, timeout }) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timer = timeout ? setTimeout(abort, timeout * 1000) : undefined;
    try {
      const response = await fetch(`${endpoint}/exec`, {
        method: "POST",
        headers: {
          "authorization": `Bearer ${token}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ command, timeout_sec: timeout }),
        signal: controller.signal,
      });
      const body = await response.json() as {
        stdout?: string;
        stderr?: string;
        return_code?: number;
        error?: string;
      };
      if (!response.ok) throw new Error(body.error || `bridge HTTP ${response.status}`);
      if (body.stdout) onData(Buffer.from(body.stdout));
      if (body.stderr) onData(Buffer.from(body.stderr));
      return { exitCode: Number.isInteger(body.return_code) ? body.return_code! : 1 };
    } finally {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  },
};

export default function (pi: ExtensionAPI) {
  pi.registerTool(createBashToolDefinition("/", {
    operations,
    exposeSessionEnvironment: false,
  }));
  pi.on("before_agent_start", async (event) => ({
    systemPrompt: `${event.systemPrompt}\n\nThe bash tool executes inside the isolated benchmark task environment. ` +
      "Use bash for every task file operation; the host-side working directory is not the task filesystem.",
  }));
}
"""


class PiBridgeError(RuntimeError):
    """The Pi process or its Harbor execution bridge failed."""


@dataclass(frozen=True)
class PiProcessResult:
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float


@dataclass(frozen=True)
class PiEpisodeResult:
    exit_code: int
    stdout: str
    stderr: str
    trajectory: tuple[dict[str, Any], ...]
    usage: Mapping[str, int]
    estimated_cost_usd: float
    provider_reported_cost_usd: float
    wall_time_s: float


class PiProcessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        workspace: Path,
        environment: Mapping[str, str],
        timeout_s: float,
    ) -> PiProcessResult: ...


class ExecCoroutine(Protocol):
    def __call__(self, *, command: str, timeout_sec: int) -> Awaitable[Any]: ...


class HarborExecBridge:
    """Authenticated loopback HTTP bridge into one Harbor environment."""

    def __init__(
        self,
        executor: ExecCoroutine,
        loop: asyncio.AbstractEventLoop,
        *,
        request_timeout_s: float = 3600.0,
    ) -> None:
        self._executor = executor
        self._loop = loop
        self._request_timeout_s = request_timeout_s
        self.token = secrets.token_urlsafe(32)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> HarborExecBridge:
        if self._thread is not None:
            raise PiBridgeError("Harbor execution bridge is already running")
        import threading

        self._thread = threading.Thread(target=self._server.serve_forever, name="reef-harbor-bridge", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._thread = None

    def __enter__(self) -> HarborExecBridge:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/exec":
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown bridge path"})
                    return
                supplied = self.headers.get("authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {bridge.token}"):
                    self._reply(HTTPStatus.UNAUTHORIZED, {"error": "invalid bridge token"})
                    return
                try:
                    length = int(self.headers.get("content-length", "0"))
                except ValueError:
                    length = 0
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid request size"})
                    return
                try:
                    body = json.loads(self.rfile.read(length))
                    command = body.get("command") if isinstance(body, Mapping) else None
                    if not isinstance(command, str) or not command.strip():
                        raise ValueError("command must be a non-empty string")
                    timeout_raw = body.get("timeout_sec")
                    timeout_sec = min(max(int(timeout_raw or 600), 1), int(bridge._request_timeout_s))
                    future = asyncio.run_coroutine_threadsafe(
                        bridge._executor(command=command, timeout_sec=timeout_sec),
                        bridge._loop,
                    )
                    result = future.result(timeout=timeout_sec + 5)
                    self._reply(
                        HTTPStatus.OK,
                        {
                            "stdout": getattr(result, "stdout", None) or "",
                            "stderr": getattr(result, "stderr", None) or "",
                            "return_code": int(result.return_code),
                        },
                    )
                except Exception as exc:
                    self._reply(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"[:1000]},
                    )

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _reply(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
                data = json.dumps(dict(payload)).encode("utf-8")
                self.send_response(status.value)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


class PiEpisodeRunner:
    """Materialize one transient Pi composition and collect its native session."""

    def __init__(
        self,
        descriptor: AdapterDescriptor,
        *,
        binary: Path,
        timeout_s: float,
        process_runner: PiProcessRunner | None = None,
        pricing: TokenPricing = TARGET_MODEL_PRICING,
    ) -> None:
        if descriptor.name != "pi":
            raise ValueError("PiEpisodeRunner requires the Pi adapter")
        if timeout_s <= 0:
            raise ValueError("Pi timeout must be positive")
        self.descriptor = descriptor
        self.binary = Path(binary).resolve()
        self.timeout_s = timeout_s
        self._process_runner = process_runner or _run_pi_process
        self.pricing = pricing

    def run(
        self,
        files: Mapping[str, str],
        instruction: str,
        *,
        bridge_url: str,
        bridge_token: str,
    ) -> PiEpisodeResult:
        root = Path(tempfile.mkdtemp(prefix="reef-meta-pi-"))
        try:
            rendered = dict(files)
            if BRIDGE_EXTENSION_PATH in rendered:
                raise PiBridgeError(f"candidate uses reserved runtime path {BRIDGE_EXTENSION_PATH!r}")
            rendered[BRIDGE_EXTENSION_PATH] = BRIDGE_EXTENSION
            for relative, text in rendered.items():
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise PiBridgeError(f"render path escapes the episode root: {relative!r}")
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            workspace = root / "workspace"
            workspace.mkdir()
            env = _pi_environment(self.descriptor, root, bridge_url, bridge_token)
            extension_paths = sorted(
                path
                for path in (root / "pi-agent" / "extensions").glob("*.ts")
                if path.name != Path(BRIDGE_EXTENSION_PATH).name
            )
            bridge_path = root / BRIDGE_EXTENSION_PATH
            command = [str(self.binary), "--mode", "json", "--no-extensions"]
            for extension_path in extension_paths:
                command.extend(("--extension", str(extension_path)))
            command.extend(
                (
                    "--extension",
                    str(bridge_path),
                    "--tools",
                    "bash",
                    "--print",
                    instruction,
                )
            )
            result = self._process_runner(command, workspace, env, self.timeout_s)
            trajectory = reader_for(self.descriptor.trajectory_format)(root / self.descriptor.trajectory_path)
            usage, provider_reported_cost = _trajectory_usage(trajectory)
            return PiEpisodeResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                trajectory=trajectory,
                usage=usage,
                estimated_cost_usd=self.pricing.estimate_usd(usage),
                provider_reported_cost_usd=provider_reported_cost,
                wall_time_s=result.wall_time_s,
            )
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


def _pi_environment(
    descriptor: AdapterDescriptor,
    root: Path,
    bridge_url: str,
    bridge_token: str,
) -> dict[str, str]:
    inherited = {key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "TMPDIR"}}
    configured = {key: value.replace("{root}", str(root)) for key, value in descriptor.env.items()}
    return {
        **inherited,
        **configured,
        "HOME": str(root),
        "REEF_HARBOR_BRIDGE_URL": bridge_url,
        "REEF_HARBOR_BRIDGE_TOKEN": bridge_token,
    }


def _trajectory_usage(trajectory: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], float]:
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    cost = 0.0
    for event in trajectory:
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        raw = message.get("usage")
        if not isinstance(raw, Mapping):
            continue
        usage["input_tokens"] += int(raw.get("input") or 0)
        usage["cached_input_tokens"] += int(raw.get("cacheRead") or 0)
        usage["output_tokens"] += int(raw.get("output") or 0)
        raw_cost = raw.get("cost")
        if isinstance(raw_cost, Mapping):
            cost += float(raw_cost.get("total") or 0.0)
    return usage, cost


def _run_pi_process(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_s: float,
) -> PiProcessResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise PiBridgeError(f"could not start Pi: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise PiBridgeError(f"Pi timed out after {timeout_s:g} seconds") from exc
    return PiProcessResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_time_s=time.monotonic() - started,
    )
