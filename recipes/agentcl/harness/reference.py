"""Privileged reference verification agent: no model calls and no student role."""

from __future__ import annotations

from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .agent import HarborSandbox


class ReferenceAgent(BaseAgent):
    """Submit a host-only gold module to a fresh verifier qualification trial."""

    def __init__(self, *args, reference_path: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reference_path = Path(reference_path)

    @staticmethod
    def name() -> str:
        return "reef-agentcl-reference-verification"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        """No model or environment setup is needed for reference submission."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await HarborSandbox(environment).submit(self.reference_path.read_text(encoding="utf-8"))
        context.metadata = {"agentcl_role": "reference-verification"}
        context.n_input_tokens = 0
        context.n_output_tokens = 0
