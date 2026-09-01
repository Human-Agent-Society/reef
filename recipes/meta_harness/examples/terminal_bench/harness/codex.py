"""Audited non-interactive Codex proposer for Meta-Harness search."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .search import ProposerSession

_CODEX_FROM_PATH = shutil.which("codex")
DEFAULT_CODEX_PATH = Path(
    os.environ.get("REEF_CODEX_BINARY") or _CODEX_FROM_PATH or "/Applications/ChatGPT.app/Contents/Resources/codex"
)
_SAFE_ENVIRONMENT_KEYS = {
    "CODEX_HOME",
    "HOME",
    "LANG",
    "LOGNAME",
    "NO_COLOR",
    "PATH",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
}
_PERMISSION_PROFILE = "meta_harness"
_FILESYSTEM_POLICY = 'permissions.meta_harness.filesystem={":minimal"="read",":workspace_roots"={"."="write"}}'
_NETWORK_POLICY = "permissions.meta_harness.network={enabled=false}"
_SHELL_HISTORY_READER = re.compile(
    r"(?:^|[;&|]\s*)(?:awk|bat|cat|grep|head|jq|less|more|rg|sed|tail)\b[^;&|\n]*\bhistory/"
)
_PYTHON_HISTORY_READER = re.compile(
    r"(?:open\(\s*['\"]history/|path\(\s*['\"]history/[^'\"]*['\"]\s*\)\.(?:read|read_text)\()",
    re.IGNORECASE,
)


class CodexProposerError(RuntimeError):
    """Codex did not produce a complete, auditable proposer turn."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        environment: Mapping[str, str],
        timeout_s: float,
    ) -> ProcessResult: ...


class VersionReader(Protocol):
    def __call__(self, path: Path) -> str: ...


class CodexProposer:
    """Run one isolated Codex turn and retain its machine-readable evidence."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        codex_path: Path = DEFAULT_CODEX_PATH,
        timeout_s: float = 1800.0,
        process_runner: ProcessRunner | None = None,
        version_reader: VersionReader | None = None,
    ) -> None:
        if not model:
            raise ValueError("Codex proposer model must not be empty")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported Codex reasoning effort")
        if timeout_s <= 0:
            raise ValueError("Codex proposer timeout must be positive")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_path = Path(codex_path).resolve()
        self.timeout_s = float(timeout_s)
        self._process_runner = process_runner or _run_process
        self._version_reader = version_reader or _read_version

    def propose(
        self,
        *,
        surface: Path,
        prompt: str,
        session_dir: Path,
        round_index: int,
    ) -> ProposerSession:
        surface = Path(surface).resolve()
        if not surface.is_dir():
            raise ValueError(f"proposal surface does not exist: {surface}")
        session_dir = Path(session_dir).resolve()
        artifact_dir = session_dir / f"invocation-{uuid.uuid4().hex}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        last_message = artifact_dir / "last-message.txt"
        version = self._version_reader(self.codex_path)
        command = self._command(surface, prompt, last_message)
        try:
            result = self._process_runner(command, _codex_environment(), self.timeout_s)
        except Exception as exc:
            failed = ProcessResult(returncode=-1, stdout="", stderr=str(exc), wall_time_s=0.0)
            (artifact_dir / "events.jsonl").write_text("", encoding="utf-8")
            (artifact_dir / "stderr.log").write_text(str(exc), encoding="utf-8")
            self._write_metadata(
                artifact_dir,
                version=version,
                command=command,
                round_index=round_index,
                result=failed,
                status="process_exception",
                error=str(exc),
            )
            if isinstance(exc, CodexProposerError):
                raise
            raise CodexProposerError(f"Codex execution failed: {exc}") from exc
        (artifact_dir / "events.jsonl").write_text(result.stdout, encoding="utf-8")
        (artifact_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

        try:
            events = _parse_events(result.stdout)
            evidence = _summarize_events(events)
        except Exception as exc:
            self._write_metadata(
                artifact_dir,
                version=version,
                command=command,
                round_index=round_index,
                result=result,
                status="invalid_jsonl",
                error=str(exc),
            )
            raise CodexProposerError(f"Codex returned invalid JSONL: {exc}") from exc

        error_events = [event for event in events if event.get("type") == "error"]
        total_tokens = evidence["usage"]["input_tokens"] + evidence["usage"]["output_tokens"]
        status = "complete"
        error: str | None = None
        if result.returncode != 0:
            status = "process_failed"
            error = f"Codex exited with status {result.returncode}"
        elif error_events:
            status = "event_failed"
            error = str(error_events[-1].get("message") or "Codex emitted an error event")
        elif not evidence["session_id"]:
            status = "missing_session"
            error = "Codex emitted no thread.started session id"
        elif evidence["completed_turns"] < 1:
            status = "incomplete_turn"
            error = "Codex emitted no turn.completed event"
        elif total_tokens <= 0:
            status = "missing_usage"
            error = "Codex completed without non-zero token usage"

        self._write_metadata(
            artifact_dir,
            version=version,
            command=command,
            round_index=round_index,
            result=result,
            status=status,
            error=error,
            evidence=evidence,
        )
        if error is not None:
            raise CodexProposerError(error)

        return ProposerSession(
            session_id=evidence["session_id"],
            input_tokens=evidence["usage"]["input_tokens"],
            output_tokens=evidence["usage"]["output_tokens"],
            wall_time_s=result.wall_time_s,
            artifact_dir=artifact_dir.relative_to(session_dir.parent.parent).as_posix(),
        )

    def _command(self, surface: Path, prompt: str, last_message: Path) -> list[str]:
        return [
            str(self.codex_path),
            "exec",
            "--json",
            "--color",
            "never",
            "--cd",
            str(surface),
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--config",
            f'default_permissions="{_PERMISSION_PROFILE}"',
            "--config",
            _FILESYSTEM_POLICY,
            "--config",
            _NETWORK_POLICY,
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--config",
            "sandbox_workspace_write.network_access=false",
            "--output-last-message",
            str(last_message),
            prompt,
        ]

    @staticmethod
    def _write_metadata(
        artifact_dir: Path,
        *,
        version: str,
        command: Sequence[str],
        round_index: int,
        result: ProcessResult,
        status: str,
        error: str | None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "round": round_index,
            "status": status,
            "error": error,
            "codex_version": version,
            "command": list(command),
            "returncode": result.returncode,
            "wall_time_s": result.wall_time_s,
            "evidence": dict(evidence or {}),
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _codex_environment() -> dict[str, str]:
    """Pass only process essentials; target credentials never reach the proposer."""
    return {key: value for key, value in os.environ.items() if key in _SAFE_ENVIRONMENT_KEYS or key.startswith("LC_")}


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not JSON") from exc
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            raise ValueError(f"line {line_number} is not a Codex event object")
        events.append(value)
    if not events:
        raise ValueError("event stream is empty")
    return events


def _summarize_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    session_id = ""
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    item_types: Counter[str] = Counter()
    tool_evidence: list[dict[str, Any]] = []
    history_accesses: list[dict[str, Any]] = []
    completed_turns = 0
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "thread.started":
            session_id = str(event.get("thread_id") or "")
        if event_type == "turn.completed":
            completed_turns += 1
            raw_usage = event.get("usage")
            if isinstance(raw_usage, Mapping):
                for key in usage:
                    usage[key] += int(raw_usage.get(key) or 0)
        item = event.get("item")
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "unknown")
        item_types[item_type] += 1
        if event_type != "item.completed" or item_type not in {"command_execution", "file_change"}:
            continue
        row = {"type": item_type, "status": item.get("status")}
        if item_type == "command_execution":
            row["command"] = item.get("command")
            row["exit_code"] = item.get("exit_code")
            if _is_completed_history_content_read(item):
                history_accesses.append(
                    {
                        "event_type": event_type,
                        "item_type": item_type,
                        "command": item.get("command"),
                    }
                )
        else:
            changes = item.get("changes")
            row["paths"] = (
                [change.get("path") for change in changes if isinstance(change, Mapping)]
                if isinstance(changes, list)
                else []
            )
        tool_evidence.append(row)
    return {
        "session_id": session_id,
        "completed_turns": completed_turns,
        "usage": usage,
        "item_types": dict(sorted(item_types.items())),
        "tool_evidence": tool_evidence,
        "history_accesses": history_accesses,
    }


def _is_completed_history_content_read(item: Mapping[str, Any]) -> bool:
    """Recognize logged commands that read candidate or trajectory content."""
    if item.get("status") != "completed" or item.get("exit_code") not in (None, 0):
        return False
    command = item.get("command")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if not isinstance(command, str):
        return False
    normalized = command.lower()
    return bool(_SHELL_HISTORY_READER.search(normalized) or _PYTHON_HISTORY_READER.search(normalized))


def verify_permission_profile(codex_path: Path) -> None:
    """Prove this Codex build denies sibling reads while allowing surface writes."""
    with tempfile.TemporaryDirectory(prefix="reef-meta-harness-sandbox-") as temporary:
        root = Path(temporary)
        surface = root / "surface"
        surface.mkdir()
        (surface / "inside.txt").write_text("inside\n", encoding="utf-8")
        (root / "sealed.txt").write_text("sealed\n", encoding="utf-8")
        command = [
            str(Path(codex_path).resolve()),
            "sandbox",
            "--cd",
            str(surface),
            "--permission-profile",
            _PERMISSION_PROFILE,
            "--config",
            _FILESYSTEM_POLICY,
            "--config",
            _NETWORK_POLICY,
            "--",
            "sh",
            "-c",
            (
                'test "$(cat inside.txt)" = inside && ! cat ../sealed.txt >/dev/null 2>&1 '
                "&& printf writable > writable.txt"
            ),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_codex_environment(),
        )
        if completed.returncode != 0 or not (surface / "writable.txt").is_file():
            raise CodexProposerError("Codex permission profile did not enforce the sealed proposal surface")


def _read_version(codex_path: Path) -> str:
    try:
        result = subprocess.run(
            [str(codex_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=_codex_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodexProposerError(f"could not read Codex version: {exc}") from exc
    version = result.stdout.strip()
    if not version:
        raise CodexProposerError("Codex version output was empty")
    return version


def _run_process(command: Sequence[str], environment: Mapping[str, str], timeout_s: float) -> ProcessResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(environment),
            start_new_session=True,
        )
    except OSError as exc:
        raise CodexProposerError(f"could not start Codex: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise CodexProposerError(f"Codex timed out after {timeout_s:g} seconds") from exc
    return ProcessResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_time_s=time.monotonic() - started,
    )
