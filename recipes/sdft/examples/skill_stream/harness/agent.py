"""One Harbor trial, one stage of the skill stream: the stage runner in the task container.

The agent makes no model calls of its own. ``stage.py`` in the task's image
samples, reports and waits for training against the Reef service on the
host, and asks the task's judge to score the served model on both skills.
This class runs it, hands it the stage settings from the host environment
(every ``SKILLS_*`` variable), and keeps its output in the trial's agent log.
Harbor runs the verifier afterwards, which asks the judge for the final
result; reef-eval stores that and the judge's per-submission scores.
"""

from __future__ import annotations

import os

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: The runner the task image ships; its output streams into the trial's agent log.
STAGE_COMMAND = (
    "bash -c 'set -o pipefail; mkdir -p /logs/agent; python3 /opt/skills/stage.py 2>&1 | tee -a /logs/agent/stage.log'"
)
#: Host environment forwarded into the stage: the runner's settings (epochs, prompts per step, ...).
FORWARDED_PREFIX = "SKILLS_"
#: A stage is up to 252 steps, each a checkpoint save; the ceiling matches the task's agent timeout.
DEFAULT_STAGE_TIMEOUT_S = 172_800.0


class HarborAgent(BaseAgent):
    """The stage runner, executed in the task container with the host's stage settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        environ = os.environ
        self.stage_environment = {key: value for key, value in environ.items() if key.startswith(FORWARDED_PREFIX)}
        self.timeout_s = float(environ.get("SKILLS_STAGE_TIMEOUT_S", "") or DEFAULT_STAGE_TIMEOUT_S)

    @staticmethod
    def name() -> str:
        return "reef-sdft-skill-stream"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the runner and the reference datasets."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.logger.info("running the stage with %s", sorted(self.stage_environment) or "the runner's defaults")
        result = await environment.exec(STAGE_COMMAND, env=self.stage_environment, timeout_sec=int(self.timeout_s))
        if result.return_code != 0:
            tail = (result.stderr or result.stdout or "")[-2000:]
            raise RuntimeError(f"the stage runner exited {result.return_code}: {tail}")
        self.logger.info("the stage finished; the judge holds its scores")
