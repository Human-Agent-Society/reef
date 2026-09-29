"""Harbor ``BaseAgent`` subclass for summary-only Guidance-TTT.

One Harbor trial owns the complete Guidance-TTT trajectory: every step samples
the trainable guidance policy through Reef, executes the guidance with a frozen
external model, verifies the candidate with the external judge, and waits for
Reef's durable LoRA training transaction before the next step starts. It ends
by writing the archive's best candidate into the task environment, where the
Harbor verifier scores it independently.

This is the only module that imports the external ``harbor`` package; the rest
of the harness can be imported standalone (e.g. in tests).
"""

from __future__ import annotations

import asyncio
import base64

import yaml
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .agent import ReefGuidanceTTTHarness, prepare_library
from .bootstrap import prepare_seed
from .config import RunConfig
from .contract import TaskContract
from .execution import OpenAICompatibleExecutionClient
from .run_controller import GuidanceRunController, GuidanceRunIdentity, GuidanceRunStateStore, RayTrainingBridge
from .scorer import SOURCE_FILES, JudgeScorer

CONFIG = RunConfig.load()
SERVICE_URL = CONFIG.service_url
SCENARIO = CONFIG.scenario
TOKEN = CONFIG.token
CLIENT_TIMEOUT_S = 28_800.0
VERIFIER_TIMEOUT_S = CONFIG.verifier_timeout_s
TASK_DIR = CONFIG.task_dir
STATE_DIR = CONFIG.state_dir
RUN_DIR = STATE_DIR / "guidance-run"
GROUPS_PER_STEP = CONFIG.groups
ROLLOUTS_PER_GROUP = CONFIG.rollouts
STEPS = CONFIG.steps
MAX_TOKENS = CONFIG.max_tokens
SEQ_LENGTH = CONFIG.sequence_length
LORA_RANK = CONFIG.lora_rank
TENSOR_PARALLEL_SIZE = CONFIG.tensor_parallel_size
MAX_WORKERS = CONFIG.max_workers
TRAIN_TIMEOUT_S = 14_400.0
TRAIN_POLL_S = 2.0
TASK_CONTRACT = TASK_DIR / "contract.json"
WORKSPACE = "/workspace"


class HarborAgent(BaseAgent):
    """A Harbor agent that runs Guidance-TTT search through Reef."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=CLIENT_TIMEOUT_S)

    @staticmethod
    def name() -> str:
        return "reef-guidance-ttt"

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
        CONFIG.validate_state()
        model = self.model_name or "reef"
        backend = CONFIG.backend
        contract = TaskContract.load(TASK_CONTRACT, problem_prompt=instruction)
        scorer = JudgeScorer(
            CONFIG.judge_url,
            problem_id=contract.judge_problem_id,
            language=contract.solution_language,
            timeout_s=VERIFIER_TIMEOUT_S,
        )

        RUN_DIR.mkdir(parents=True, exist_ok=True)
        state_store = GuidanceRunStateStore(
            RUN_DIR,
            GuidanceRunIdentity(
                model=model,
                executor=backend.name,
                gpt_oss_reasoning_effort="high",
                groups_per_step=GROUPS_PER_STEP,
                rollouts_per_group=ROLLOUTS_PER_GROUP,
                guidance_max_tokens=MAX_TOKENS,
                sequence_length=SEQ_LENGTH,
                lora_rank=LORA_RANK,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            ),
        )
        if state_store.resume_path.is_file() or state_store.committed_library_path.is_file():
            state_store.restore_working_library()
        seed_path = (
            prepare_seed(CONFIG, scorer)
            if not state_store.working_library_path.exists()
            else state_store.working_library_path
        )
        library = prepare_library(
            seed_path=seed_path,
            run_path=state_store.working_library_path,
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
            score_direction=contract.score_direction,
        )
        if not state_store.committed_library_path.exists():
            state_store.commit_library()

        harness = ReefGuidanceTTTHarness(
            self._client,
            OpenAICompatibleExecutionClient(backend),
            library,
            scenario=SCENARIO,
            model=model,
            contract=contract,
            scorer=scorer,
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
            guidance_max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )
        # reef serve publishes the shared runtime's address before starting
        # the driver. Read that snapshot, not a fixed port or another cluster.
        runtime = yaml.safe_load((STATE_DIR / "stack" / "slime-driver" / "runtime.yaml").read_text())["reef"]
        controller = GuidanceRunController(
            harness=harness,
            library=library,
            bridge=RayTrainingBridge(
                SERVICE_URL,
                SCENARIO,
                token=TOKEN,
                ray_address=runtime["ray_address"],
                ray_namespace=runtime["ray_namespace"],
                ray_actor_name=runtime["ray_actor_name"],
                timeout_s=TRAIN_TIMEOUT_S,
                poll_interval_s=TRAIN_POLL_S,
            ),
            state_store=state_store,
            checkpoint_root=STATE_DIR / "checkpoints" / "megatron",
            resume_extra={"executor_backend": backend.safe_dict(), "prompt_mode": "summary_only"},
            emit=lambda event: self.logger.info("Guidance-TTT event: %s", event),
        )
        outcome = await asyncio.to_thread(controller.run, STEPS)

        snapshot = library.snapshot()
        best_node = snapshot["nodes"].get(snapshot.get("best_node_id")) or {}
        entry = snapshot["entries"].get(best_node.get("entry_id")) or {}
        solution = str(entry.get("solution") or "").strip()
        if not solution:
            raise RuntimeError("the Guidance archive holds no executable candidate to submit")

        # The file the task's verifier reads, named for the candidate's language.
        solution_path = f"{WORKSPACE}/{SOURCE_FILES[contract.solution_language.lower()][0]}"
        encoded = base64.b64encode(solution.encode()).decode()
        result = await environment.exec(f"printf %s {encoded} | base64 -d > {solution_path}")
        if result.return_code != 0:
            raise RuntimeError(f"writing {solution_path} failed: {result.stderr}")

        metadata = dict(context.metadata or {})
        metadata["reef"] = {
            "agent_record_ids": [result.agent_record_id for result in outcome.results if result.guidance_format_ok],
            "start_step": outcome.start_step,
            "next_step": outcome.next_step,
            "runtime_load_id": outcome.runtime_load_id,
            "step_summaries": list(outcome.step_summaries),
        }
        context.metadata = metadata
