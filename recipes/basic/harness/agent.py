"""The smallest Harbor agent that talks to a Reef service.

One model call per trial: the task instruction goes to Reef's
``/v1/chat/completions`` (Reef records it and returns a receipt), the
completion is written to the path the instruction names, and the receipt
lands in the Harbor agent context. Harbor runs the verifier after ``run()``
returns and ends the trial by writing ``result.json`` next to the agent's log
directory; a watcher thread waits for that file and posts the verifier reward
back to Reef (``harness.report``), closing the loop without a runner-side step.
"""

import asyncio
import atexit
import json
import shlex
import threading
import time
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .report import post_report

#: The Reef ``run.sh`` starts from ``external-provider.yaml``.
SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
#: This workload's isolated lane.
SCENARIO = "basic-arithmetic"
#: Where the basic task's instruction says to put the answer
#: (see ``harbor/instruction.md``).
ANSWER_PATH = "/workspace/answer.txt"
#: The seconds that ``setup`` waits for inference to serve the scenario's model.
INFERENCE_TIMEOUT_S = 300.0


class HarborAgent(BaseAgent):
    """A one-shot Harbor agent whose single model call is recorded by Reef."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=300)
        # Harbor runs the verifier after run() returns, so the reward arrives
        # in the trial's result.json. Watch from construction: a run() that
        # fails before recording a receipt still gets its reward reported.
        self._started = time.time()
        self._reporter = threading.Thread(target=self._report_trial_result, daemon=True)
        self._reporter.start()
        # Don't drop an in-flight report when the runner exits after the trial.
        atexit.register(self._reporter.join, 10)

    @staticmethod
    def name() -> str:
        return "reef-basic"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Create the scenario if it does not exist yet, then wait until inference serves its model.

        The inference routes serve only a scenario that exists, and scenario creation does not wait for the model.
        """
        await asyncio.to_thread(self._client.post, "/reef/scenarios", SCENARIO, {"name": SCENARIO})
        deadline = time.monotonic() + INFERENCE_TIMEOUT_S
        while time.monotonic() < deadline:
            startup = await asyncio.to_thread(self._client.get, f"/reef/scenarios/{SCENARIO}/inference/status")
            if startup["state"] == "ready":
                return
            if startup["state"] == "failed":
                raise RuntimeError(f"inference did not start: {startup['error']}")
            await asyncio.sleep(0.5)
        raise RuntimeError(f"inference did not start within {INFERENCE_TIMEOUT_S:.0f}s")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response, receipt = await asyncio.to_thread(self._ask_reef, instruction)
        answer = response["choices"][0]["message"]["content"]

        result = await environment.exec(f"printf %s {shlex.quote(answer)} > {ANSWER_PATH}")
        if result.return_code != 0:
            raise RuntimeError(f"writing {ANSWER_PATH} failed: {result.stderr}")

        # The receipt: report.py grades the verifier reward against it. Its record names the artifact that answered.
        context.metadata = {**(context.metadata or {}), "reef": {"agent_record_ids": [receipt]}}
        context.n_input_tokens = response["usage"]["prompt_tokens"]
        context.n_output_tokens = response["usage"]["completion_tokens"]

    def _ask_reef(self, instruction: str) -> tuple[dict[str, Any], str]:
        return self._client.inference_with_record(
            SCENARIO,
            "/v1/chat/completions",
            {"model": self.model_name, "messages": [{"role": "user", "content": instruction}]},
        )

    def _report_trial_result(self) -> None:
        """Post the verifier reward once Harbor writes result.json in the trial directory."""
        result_path = self.logs_dir.parent / "result.json"
        while not (result_path.exists() and result_path.stat().st_mtime >= self._started):
            time.sleep(1)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        post_report(result, client=self._client, scenario=SCENARIO)
        self.logger.info("reported verifier reward %s to reef", result["verifier_result"]["rewards"]["reward"])
