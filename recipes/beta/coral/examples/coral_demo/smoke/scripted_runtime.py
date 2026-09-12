"""A scriptable CORAL runtime for the no-GPU smoke run.

Implements CORAL's ``AgentRuntime`` protocol for real: the manager creates
worktrees, registers gateway keys, and spawns this runtime exactly as it
would spawn ``opencode`` or ``claude``. The spawned process is
``scripted_agent.py`` — a minimal agent that asks the gateway for a
solution, writes it, and submits ``coral eval`` in a loop. Everything
between the manager and the model (gateway splice, journaling, receipts,
grading, reporting) is exercised unmodified; only the agent's
"intelligence" is scripted.

Referenced from ``run.py --runtime smoke.scripted_runtime:ScriptedRuntime``
(CORAL custom-entrypoint form).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from coral.agent.runtime import AgentHandle
from coral.sandbox.protocol import AgentSandboxSpec


class ScriptedRuntime:
    """AgentRuntime spawning the scripted smoke agent."""

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    @property
    def shared_dir_name(self) -> str:
        return ".scripted"

    def extract_session_id(self, log_path: Path) -> str | None:
        del log_path
        return None

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "reef-policy",
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 0,
        log_dir: Path | None = None,
        verbose: bool = False,
        resume_session_id: str | None = None,
        prompt: str | None = None,
        prompt_source: str | None = None,
        task_name: str | None = None,
        task_description: str | None = None,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
        run_as_user: dict[str, Any] | None = None,
        sandbox: AgentSandboxSpec | None = None,
    ) -> AgentHandle:
        del coral_md_path, runtime_options, max_turns, verbose, resume_session_id
        del prompt, prompt_source, task_name, task_description, run_as_user, sandbox
        agent_id_file = worktree_path / ".coral_agent_id"
        agent_id = agent_id_file.read_text(encoding="utf-8").strip() if agent_id_file.exists() else "unknown"
        if log_dir is None:
            log_dir = worktree_path / ".scripted" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{agent_id}.log"

        env = dict(os.environ)
        # `coral eval` must resolve to the same environment the manager runs in.
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        if gateway_url:
            env["OPENAI_BASE_URL"] = gateway_url
        if gateway_api_key:
            env["OPENAI_API_KEY"] = gateway_api_key
        env["SMOKE_MODEL"] = model.split("/", 1)[-1]  # provider prefix is runtime-internal

        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — lives on the handle
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("scripted_agent.py"))],
            cwd=worktree_path,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return AgentHandle(
            agent_id=agent_id,
            process=process,
            worktree_path=worktree_path,
            log_path=log_path,
            _log_file=log_file,
        )
