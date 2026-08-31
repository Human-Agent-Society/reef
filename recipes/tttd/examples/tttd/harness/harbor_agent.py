"""Harbor ``BaseAgent`` subclass for TTT-Discover.

This module depends on the external ``harbor`` package and is only imported
when Harbor/reef-eval loads the agent at runtime. The remaining harness modules
have no Harbor dependency and can be imported standalone (e.g. in tests).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .agent import ReefTTTDiscoverHarness
from .run_controller import ReefTrainingStatusClient, TTTDRunController, TTTDRunIdentity, TTTDRunStateStore
from .scorer import JudgeScorer, _codeblock_body
from .search import TTTDChatRequestBuilder

SERVICE_URL = "http://127.0.0.1:8900"  # the Reef run.sh started
RECIPE = "tttd"  # matches serve.yaml
TOKEN = "reef-local"  # matches serve.yaml
JUDGE_URL = "http://127.0.0.1:8082"  # the task's judge, published by its compose file
SOLUTION_PATH = "/workspace/solution.py"

# Everything that follows the chosen task, derived from run.sh's one variable.
TASK = os.environ.get("TTTD_TASK", "erdos_min_overlap")
SCENARIO = "tttd-" + TASK.replace("_", "-")
SEARCH_STATE_PATH = Path(__file__).resolve().parents[1] / "work" / TASK / "tttd-search-state.json"

# The step grid, read from the stack config rather than repeated here: Reef
# trains only after exactly groups_per_step x rollouts_per_group reports
# arrive, so a harness that disagreed would fail on the training timeout.
_STACK = yaml.safe_load((Path(__file__).resolve().parents[1] / "serve.yaml").read_text())
GROUPS_PER_STEP = _STACK["reef"]["groups_per_step"]
ROLLOUTS_PER_GROUP = _STACK["reef"]["rollouts_per_group"]
STEPS = _STACK["training"]["steps"]
MAX_NEW_TOKENS = _STACK["training"]["max_new_tokens"]
MAX_WORKERS = 256 if TASK.startswith("circle_packing") else 512  # packing needs the headroom


class HarborAgent(BaseAgent):
    """A Harbor agent that runs TTT-Discover PUCT search through Reef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=7200)

    @staticmethod
    def name() -> str:
        return "reef-tttd"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the search runs on the host."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        inference_path = "/v1/chat/completions"
        temperature = 1.0
        top_p = 1.0
        top_k = -1
        enable_thinking = True
        exploration = 1.0
        invalid_reward = 0.0

        harness = ReefTTTDiscoverHarness(
            self._client,
            JudgeScorer(JUDGE_URL),
            instruction,
            scenario=SCENARIO,
            model=self.model_name,
            recipe=RECIPE,
            inference_path=inference_path,
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
            exploration=exploration,
            invalid_reward=invalid_reward,
            max_workers=MAX_WORKERS,
            request_builder=TTTDChatRequestBuilder(
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            ),
        )
        identity = TTTDRunIdentity(
            scenario=SCENARIO,
            model=self.model_name,
            recipe=RECIPE,
            inference_path=inference_path,
            instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            exploration=exploration,
            invalid_reward=invalid_reward,
        )
        controller = TTTDRunController(
            harness,
            ReefTrainingStatusClient(SERVICE_URL, token=TOKEN),
            TTTDRunStateStore(SEARCH_STATE_PATH, identity),
            emit=lambda event: self.logger.info("TTTD event: %s", event),
        )
        outcome = await asyncio.to_thread(controller.run, STEPS)

        best = harness.archive.best()
        program_code = _codeblock_body(best.solution)

        result = await environment.exec(
            f"cat > {SOLUTION_PATH} <<'REEF_TTTD_EOF'\n{program_code}\nREEF_TTTD_EOF",
        )
        if result.return_code != 0:
            raise RuntimeError(f"writing {SOLUTION_PATH} failed: {result.stderr}")

        valid_receipts = [r.agent_record_id for r in outcome.results if r.agent_record_id and r.solution]
        metadata = dict(context.metadata or {})
        metadata["reef"] = {
            "agent_record_ids": valid_receipts,
            "start_step": outcome.start_step,
            "next_step": outcome.next_step,
            "runtime_load_id": outcome.runtime_load_id,
        }
        context.metadata = metadata
