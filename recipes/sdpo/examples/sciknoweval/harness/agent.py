"""One Harbor trial, one SDPO run on Chemistry: the stage runner in the task container.

The agent makes no model calls of its own. ``stage.py`` in the task's image
samples, reports and waits for training against the Reef service on the host,
evaluates avg@16 on the test split every few steps, and writes the curve where
the verifier reads it. This class runs it, hands it the run's settings from the
host environment (every ``SDPO_*`` and ``REEF_*`` variable), and keeps its
output in the trial's agent log.

The runner already reports every rollout as training feedback, so this harness
posts no report of its own: the Harbor reward is the run's last avg@16 and
stays in the trial result as an evaluation.
"""

from __future__ import annotations

import os

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: The runner the task image ships; its output streams into the trial's agent log.
STAGE_COMMAND = (
    "bash -c 'set -o pipefail; mkdir -p /logs/agent; "
    "python3 /opt/chemistry/stage.py 2>&1 | tee -a /logs/agent/stage.log'"
)
#: Host variables forwarded into the container: the run's settings and how to reach Reef.
FORWARDED_PREFIXES = ("SDPO_", "REEF_")
#: A run is many steps, each a grid, an optimizer step and a weight release; the
#: ceiling matches the task's agent timeout.
DEFAULT_STAGE_TIMEOUT_S = 172_800.0


class HarborAgent(BaseAgent):
    """The stage runner, executed in the task container with the host's run settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        environ = os.environ
        self.stage_environment = {key: value for key, value in environ.items() if key.startswith(FORWARDED_PREFIXES)}
        self.timeout_s = float(environ.get("SDPO_STAGE_TIMEOUT_S", "") or DEFAULT_STAGE_TIMEOUT_S)

    @staticmethod
    def name() -> str:
        return "reef-sdpo-sciknoweval"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the runner and the reference's split."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.logger.info("running the stage with %s", sorted(self.stage_environment) or "the runner's defaults")
        result = await environment.exec(STAGE_COMMAND, env=self.stage_environment, timeout_sec=int(self.timeout_s))
        if result.return_code != 0:
            tail = (result.stderr or result.stdout or "")[-2000:]
            raise RuntimeError(f"the stage runner exited {result.return_code}: {tail}")
        self.logger.info("the stage finished; the verifier reads its curve")
