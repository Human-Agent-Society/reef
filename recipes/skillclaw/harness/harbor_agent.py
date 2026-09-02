"""The minimal Harbor agent behind ``run.py solve``.

One recorded model call per trial, the sibling examples' smoke shape
(``recipes/basic/harness/agent.py``): the task instruction goes to the
embedded Reef service's ``/v1/chat/completions`` (the service injects the
served pool's catalog and records the exchange), and the completion is
written to the results file the task's grader falls back to. The campaign
driver owns reporting (``run.py``'s day loop), so this agent posts no
report: the smoke proves the task contract - environment up, instruction
in, exchange recorded, verifier reward out of ``lab.run``.

The connection is passed as agent kwargs by ``run.py solve`` (Harbor's
``AgentConfig.kwargs``): ``service_url`` and ``scenario``.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

#: Where the benchmark task's grader looks when the mock services are down,
#: and where its skill card tells the agent to leave the final report.
RESULTS_PATH = "/tmp_workspace/results/results.md"


class HarborAgent(BaseAgent):
    """A one-shot Harbor agent whose single model call is recorded by Reef."""

    def __init__(
        self,
        *args: Any,
        service_url: str,
        scenario: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._scenario = scenario
        self._client = ReefClient(
            service_url,
            timeout_s=float(os.environ.get("REEF_TIMEOUT_S", "300")),
        )

    @staticmethod
    def name() -> str:
        return "reef-skillclaw-solve"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the agent logic runs on the host."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        response, agent_record_id = await asyncio.to_thread(self._ask_reef, instruction)
        answer = response["choices"][0]["message"]["content"]
        if not isinstance(answer, str):
            raise RuntimeError("Reef inference response content must be a string")

        result = await environment.exec(
            f"printf %s {shlex.quote(answer)} > {RESULTS_PATH}",
        )
        if result.return_code != 0:
            raise RuntimeError(f"writing {RESULTS_PATH} failed: {result.stderr}")

        context.metadata = {**(context.metadata or {}), "reef": {"agent_record_ids": [agent_record_id]}}
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
                context.n_input_tokens = prompt_tokens
            if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
                context.n_output_tokens = completion_tokens

    def _ask_reef(self, instruction: str) -> tuple[dict[str, Any], str]:
        return self._client.inference_with_record(
            self._scenario,
            "/v1/chat/completions",
            {
                "model": self.model_name or "reef",
                "messages": [{"role": "user", "content": instruction}],
            },
        )
