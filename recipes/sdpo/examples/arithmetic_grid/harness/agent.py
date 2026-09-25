"""One Harbor trial, one SDPO run: the grid runner in the task container.

The agent makes no model calls of its own. ``grid.py`` in the task's image
samples, reports and waits for training against the Reef service on the host,
and writes the run's record where the verifier reads it. This class runs it,
hands it the run's settings from the host environment (every ``SDPO_*``
variable and the service credentials), and keeps its output in the trial's
agent log.

The runner already reports every rollout as training feedback, so this harness
posts no report of its own: the Harbor reward is the run's final accuracy and
stays in the trial result.
"""

from __future__ import annotations

import os

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: The runner the task image ships; its output streams into the trial's agent log.
RUN_COMMAND = (
    "bash -c 'set -o pipefail; mkdir -p /logs/agent; python3 /opt/grid/grid.py 2>&1 | tee -a /logs/agent/grid.log'"
)
#: Host variables forwarded into the container: the run's settings and how to reach Reef.
FORWARDED_PREFIXES = ("SDPO_", "REEF_")
#: Every step samples a grid and waits for its training release; the ceiling matches the task's agent timeout.
DEFAULT_RUN_TIMEOUT_S = 14_400.0


class HarborAgent(BaseAgent):
    """The grid runner, executed in the task container with the host's run settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        environ = os.environ
        self.run_environment = {key: value for key, value in environ.items() if key.startswith(FORWARDED_PREFIXES)}
        self.timeout_s = float(environ.get("SDPO_RUN_TIMEOUT_S", "") or DEFAULT_RUN_TIMEOUT_S)

    @staticmethod
    def name() -> str:
        return "reef-sdpo-arithmetic-grid"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the runner."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.logger.info("running the grid with %s", sorted(self.run_environment) or "the runner's defaults")
        result = await environment.exec(RUN_COMMAND, env=self.run_environment, timeout_sec=int(self.timeout_s))
        if result.return_code != 0:
            tail = (result.stderr or result.stdout or "")[-2000:]
            raise RuntimeError(f"the grid runner exited {result.return_code}: {tail}")
        self.logger.info("the run finished; the verifier reads its record")
