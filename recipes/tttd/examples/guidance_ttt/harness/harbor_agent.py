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
from pathlib import Path

import yaml

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .agent import ReefGuidanceTTTHarness, prepare_library
from .contract import TaskContract
from .execution import OpenAICompatibleExecutionClient, gpt_oss_120b_backend
from .run_controller import GuidanceRunController, GuidanceRunIdentity, GuidanceRunStateStore, RayTrainingBridge
from .scorer import SOURCE_FILES, JudgeScorer

# The Reef service run.sh starts, and this workload's isolated training lane.
SERVICE_URL = "http://127.0.0.1:8900"
SCENARIO = "guidance-ttt-polyomino-packing"
TOKEN = "reef-local"  # matches serve.yaml
CLIENT_TIMEOUT_S = 28_800.0

# The frozen execution model and the authoritative verifier. Neither is
# trained: only the guidance response mask reaches the optimizer. Swap
# gpt_oss_120b_backend for openrouter_glm_5_2_backend to use the API executor.
EXECUTOR_BASE_URL = "http://127.0.0.1:8000/v1"
REASONING_EFFORT = "high"
JUDGE_URL = "http://127.0.0.1:8081"
VERIFIER_TIMEOUT_S = 340

# The qualification grid and model limits, read from the stack config rather
# than repeated here: Reef trains only after exactly groups_per_step x
# rollouts_per_group reports arrive, so a harness that disagreed with the stack
# would fail on the training timeout rather than at the edit.
EXAMPLE_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = EXAMPLE_DIR / "harbor" / "polyomino_packing"
STATE_DIR = EXAMPLE_DIR / "work" / "polyomino_packing"  # checkpoints, artifacts, records, logs
RUN_DIR = STATE_DIR / "guidance-run"

_STACK = yaml.safe_load((EXAMPLE_DIR / "serve.yaml").read_text())
GROUPS_PER_STEP = _STACK["reef"]["groups_per_step"]
ROLLOUTS_PER_GROUP = _STACK["reef"]["rollouts_per_group"]
STEPS = _STACK["training"]["steps"]
MAX_TOKENS = _STACK["training"]["max_tokens"]
SEQ_LENGTH = _STACK["training"]["seq_length"]
LORA_RANK = _STACK["training"]["lora_rank"]
TENSOR_PARALLEL_SIZE = _STACK["training"]["tensor_parallel_size"]
MAX_WORKERS = 8  # host-side concurrency; the stack has no matching knob

# Reef's Ray train bridge, as serve.yaml names it.
RAY_ADDRESS = "127.0.0.1:6379"
RAY_NAMESPACE = "reef"
RAY_ACTOR_NAME = "reef-train-bridge"
TRAIN_TIMEOUT_S = 14_400.0
TRAIN_POLL_S = 2.0

SEED_LIBRARY = TASK_DIR / "solution" / "gpt_oss_120b_bootstrap_library.json"
TASK_CONTRACT = TASK_DIR / "contract.json"  # the task's prompt vocabulary
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
        model = self.model_name or "reef"
        backend = gpt_oss_120b_backend(
            base_url=EXECUTOR_BASE_URL,
            concurrency=MAX_WORKERS,
            reasoning_effort=REASONING_EFFORT,
        )

        RUN_DIR.mkdir(parents=True, exist_ok=True)
        state_store = GuidanceRunStateStore(
            RUN_DIR,
            GuidanceRunIdentity(
                model=model,
                executor=backend.name,
                gpt_oss_reasoning_effort=REASONING_EFFORT,
                groups_per_step=GROUPS_PER_STEP,
                rollouts_per_group=ROLLOUTS_PER_GROUP,
                guidance_max_tokens=MAX_TOKENS,
                sequence_length=SEQ_LENGTH,
                lora_rank=LORA_RANK,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            ),
        )
        if state_store.resume_path.is_file():
            state_store.restore_working_library()
        library = prepare_library(
            seed_path=SEED_LIBRARY,
            run_path=state_store.working_library_path,
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
        )
        if not state_store.committed_library_path.exists():
            state_store.commit_library()

        contract = TaskContract.load(TASK_CONTRACT, problem_prompt=instruction)
        harness = ReefGuidanceTTTHarness(
            self._client,
            OpenAICompatibleExecutionClient(backend),
            library,
            scenario=SCENARIO,
            model=model,
            contract=contract,
            scorer=JudgeScorer(
                JUDGE_URL,
                problem_id=contract.judge_problem_id,
                language=contract.solution_language,
                timeout_s=VERIFIER_TIMEOUT_S,
            ),
            groups_per_step=GROUPS_PER_STEP,
            rollouts_per_group=ROLLOUTS_PER_GROUP,
            guidance_max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )
        controller = GuidanceRunController(
            harness=harness,
            library=library,
            bridge=RayTrainingBridge(
                SERVICE_URL,
                SCENARIO,
                token=TOKEN,
                ray_address=RAY_ADDRESS,
                ray_namespace=RAY_NAMESPACE,
                ray_actor_name=RAY_ACTOR_NAME,
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
        result = await environment.exec(
            f"cat > {solution_path} <<'REEF_GUIDANCE_EOF'\n{solution}\nREEF_GUIDANCE_EOF",
        )
        if result.return_code != 0:
            raise RuntimeError(f"writing {solution_path} failed: {result.stderr}")

        metadata = dict(context.metadata or {})
        metadata["reef"] = {
            "agent_record_ids": [result.agent_record_id for result in outcome.results if result.guidance_format_ok],
            "start_step": outcome.start_step,
            "next_step": outcome.next_step,
            "weight_version": outcome.weight_version,
            "step_summaries": list(outcome.step_summaries),
        }
        context.metadata = metadata
