"""Codex CLI proposer, interface-compatible with ``claude_wrapper``.

Upstream hardcodes the Claude CLI: ``propose_claude`` calls
``claude_wrapper.run(model="opus")`` and there is no provider seam. Running
the reference arm on the same proposer as the Reef arm therefore needs this
substitution, and it is the only change to upstream's loop — the search,
the frontier, and the evaluation are untouched.

Only ``exit_code`` and ``stderr`` are read by the caller, plus the side
effect that matters: the session writes ``pending_eval.json`` itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class SessionResult:
    prompt: str
    text: str = ""
    exit_code: int = 0
    stderr: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    model: str = ""
    command: list = field(default_factory=list)
    raw_events: list = field(default_factory=list)

    def show(self) -> None:
        print(f"codex exit={self.exit_code} {self.duration_seconds:.1f}s model={self.model}")


def _binary() -> str:
    return os.environ.get("REEF_CODEX_BINARY") or shutil.which("codex") or "codex"


def run(
    prompt,
    model="gpt-5.6-sol",
    allowed_tools=None,
    tools=None,
    disallowed_tools=None,
    cwd=None,
    log_dir=None,
    name=None,
    system_prompt=None,
    skill_path=None,
    skills=None,
    skill_dir=None,
    timeout_seconds=None,
    disable_skills=True,
    disable_mcp=True,
    progress=True,
    effort=None,
) -> SessionResult:
    """Run one non-interactive Codex session in ``cwd``.

    The proposer must write files, so the session gets workspace-write. The
    unused keyword arguments exist to match ``claude_wrapper.run`` — dropping
    them would make this a different function and force edits at the call
    site, which is exactly what this substitution is trying to avoid.
    """
    del allowed_tools, tools, disallowed_tools, system_prompt
    del skill_path, skills, skill_dir, disable_skills, disable_mcp, progress

    command = [
        _binary(),
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--model",
        model,
    ]
    if effort:
        command += ["-c", f"model_reasoning_effort={effort!r}".replace("'", '"')]
    if cwd:
        command += ["-C", str(cwd)]
    command.append(prompt)

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            # Without this codex finishes the work and then blocks on
            # "Reading additional input from stdin...", so the session never
            # returns and the iteration times out with its files already written.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, stdout, stderr = 124, exc.stdout or "", f"codex timed out after {timeout_seconds}s"
    except FileNotFoundError as exc:
        exit_code, stdout, stderr = 127, "", str(exc)

    def _event(line):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    events = [event for event in map(_event, (stdout or "").splitlines()) if event is not None]

    if log_dir:
        directory = os.path.join(log_dir, name or "session")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "stdout.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(stdout or "")
        with open(os.path.join(directory, "stderr.txt"), "w", encoding="utf-8") as handle:
            handle.write(stderr or "")

    return SessionResult(
        prompt=prompt,
        text=stdout or "",
        exit_code=exit_code,
        stderr=stderr or "",
        duration_seconds=time.time() - started,
        model=model,
        command=command,
        raw_events=events,
    )
