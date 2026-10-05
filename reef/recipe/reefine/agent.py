"""Compatibility wiring for the Reefine agent proposer."""

from pathlib import Path

from reef.recipe.reefine.prompts import AGENT_PROMPT, INSTRUCTIONS
from reef.train.reefine import proposer as evolution
from reef.train.reefine.agent import AGENT_TOOLS, TOOLS_ENTRY_ID, TRIAL_DRIVER
from reef.train.reefine.agent import AgentProposer as Implementation
from reef.train.reefine.agent import AgentRun, WorkspaceTools
from reef.train.reefine.agent import agent_rules as render_agent_rules
from reef.train.reefine.agent import launch_pi, workspace_mutations, write_workspace
from reef.train.reefine.multimodal import MultimodalProvider

AGENT_RULES = Path(__file__).with_name("agent_rules.md")


class AgentProposer(Implementation):
    def __init__(self, provider: MultimodalProvider | None = None, adapter: str = "pi") -> None:
        super().__init__(INSTRUCTIONS, AGENT_RULES.read_text(encoding="utf-8"), AGENT_PROMPT, provider, adapter)


def agent_rules(provider: MultimodalProvider | None) -> str:
    return render_agent_rules(provider, AGENT_RULES.read_text(encoding="utf-8"))


propose = AgentProposer()

__all__ = [
    "AGENT_PROMPT",
    "AGENT_TOOLS",
    "INSTRUCTIONS",
    "TOOLS_ENTRY_ID",
    "TRIAL_DRIVER",
    "AgentProposer",
    "AgentRun",
    "Implementation",
    "WorkspaceTools",
    "agent_rules",
    "evolution",
    "launch_pi",
    "propose",
    "render_agent_rules",
    "workspace_mutations",
    "write_workspace",
]
