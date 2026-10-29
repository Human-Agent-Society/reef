"""``evolution.agent``: which coding agent a method runs, where, and for how long.

``agent_settings`` reads the section once per deployment; ``AgentSettings.bind``
binds it to the agent's program once per scenario, as the evaluation's
executor is bound to the harness's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.execution.sandbox.config import REMOTE_EXECUTORS, remote_executor
from reef.execution.sandbox.process import LocalExecutor, ProgramExecutor, SandboxExecutor, SandboxUnavailable
from reef.harness.adapters import get_adapter
from reef.harness.tasks.vendor_install import AdapterProgram
from reef.train.harness_backend.agent.run import AgentExecution

logger = logging.getLogger(__name__)

#: The adapter a method's agent runs on when ``evolution.agent.adapter`` names none.
DEFAULT_ADAPTER = "pi"
#: How long one agent run may take when ``evolution.agent.timeout_s`` says nothing.
DEFAULT_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class AgentSettings:
    """``evolution.agent``, read: which coding agent a method runs, where, and for how long.

    Values: ``adapter`` names the agent's harness adapter. ``executor`` is
    the agent's isolation, not yet bound to the adapter's program.
    ``timeout_s`` > 0 bounds one run. ``binary`` is the agent's binary,
    ``None`` for the adapter's pinned install.
    Purpose: ``agent_settings`` reads it from the deployment, and the recipe
    binds it for each scenario.
    """

    adapter: str
    executor: ProgramExecutor
    timeout_s: float
    binary: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("evolution.agent.timeout_s must be positive")

    def bind(self) -> AgentExecution:
        """The agent's execution: the executor bound to the adapter's program and preflighted. The binary is
        resolved, and installed when it is missing, on the Reef host, or named in a cloud sandbox's image."""
        descriptor = get_adapter(self.adapter)
        executor, binary = self.executor.for_program(AdapterProgram(descriptor), binary=self.binary)
        executor.preflight()
        return AgentExecution(descriptor, executor, binary, self.timeout_s)


def agent_settings(section: Any, environ: Mapping[str, str], *, state_dir: Path) -> AgentSettings | None:
    """``evolution.agent`` as the agent's adapter, binary, isolation and time limit; ``None`` runs no agent.

    ``adapter`` (default ``pi``) is any adapter that takes a text prompt, and
    ``binary`` names its binary in place of the pinned install.
    ``sandbox: bwrap`` jails the agent and gives it the internet but no host
    port but the gateway's, and refuses to start where bwrap or pasta is
    missing. A cloud sandbox (``sandbox: e2b`` or ``sandbox: modal``, with
    that provider's options in the section; its image is the adapter's
    pinned binary, built on first use, unless an option names one) runs it
    where it reaches the gateway through a tunnel and nothing else of the
    host, and carries the deployment that ``state_dir`` names, so a later
    start stops the sandboxes a stopped run left. ``sandbox: none`` runs it
    unisolated with the service's privileges, and must be chosen. Left empty
    (and ``REEF_AGENT_SANDBOX`` unset), the agent is jailed where the host
    can, and off where it cannot. No section runs no agent. The variable's
    old name, ``REEF_PROPOSER_SANDBOX``, is refused, so a deployment that
    still sets it does not start without the isolation it names. Raises
    ValueError for a bad value.
    """
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("evolution.agent must be a mapping")
    if "REEF_PROPOSER_SANDBOX" in environ:
        raise ValueError("REEF_PROPOSER_SANDBOX is no longer read: set REEF_AGENT_SANDBOX instead")
    adapter = section.get("adapter", DEFAULT_ADAPTER)
    binary = section.get("binary")
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError("evolution.agent.adapter must name an adapter")
    if binary is not None and (not isinstance(binary, str) or not binary.strip()):
        raise ValueError("evolution.agent.binary must be the path of the agent's binary when it is set")
    timeout_s = section.get("timeout_s", DEFAULT_TIMEOUT_S)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("evolution.agent.timeout_s must be a positive number")
    timeout_s = float(timeout_s)
    sandbox = str(section.get("sandbox") or environ.get("REEF_AGENT_SANDBOX") or "").strip()
    if sandbox == "none":
        logger.warning(
            "evolution.agent.sandbox is none: the agent runs with the service's own privileges and full "
            "network access, fed text from clients; use it only where you trust every client"
        )
        return AgentSettings(adapter, LocalExecutor(), timeout_s, binary)
    if sandbox in REMOTE_EXECUTORS:
        try:
            remote = remote_executor(sandbox, section, environ, timeout_s=timeout_s, owner_anchor=state_dir)
            remote.preflight()
        except SandboxUnavailable as exc:
            raise ValueError(f"evolution.agent.sandbox is {sandbox}, but {exc}") from exc
        return AgentSettings(adapter, remote, timeout_s, binary)
    if sandbox not in ("", "bwrap"):
        choices = ", ".join(f"'{name}'" for name in ("bwrap", *sorted(REMOTE_EXECUTORS), "none"))
        raise ValueError(f"evolution.agent.sandbox must be one of {choices}")
    executor = SandboxExecutor(network="isolated")
    try:
        executor.preflight()
    except SandboxUnavailable as exc:
        if sandbox == "bwrap":
            raise ValueError(f"evolution.agent.sandbox is bwrap, but {exc}") from exc
        logger.warning(
            "the agent is off: %s. Set evolution.agent.sandbox: none to run the agent without isolation", exc
        )
        return None
    return AgentSettings(adapter, executor, timeout_s, binary)


__all__ = ["DEFAULT_ADAPTER", "AgentSettings", "agent_settings"]
