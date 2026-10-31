"""Select one benchmark and keep its service, archive, and verifier paths together."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from .contract import TaskContract
from .execution import ExecutionBackend, gpt_oss_120b_backend, openai_compatible_backend, openrouter_glm_5_2_backend

TASKS = ("polyomino_packing", "lasso_path", "ahc058", "trimul")
# CPU-only FrontierCS 2 tasks scored by judges/frontiercs2_server.py; a task's name is its FrontierCS 2 id.
FRONTIERCS2_TASKS = ("join_tree_rewriting", "approximate_mips_query_stopping", "lejepa_world_model", "polar_express")
# Seconds for one judge call: the task's evaluation timeout plus a margin for queueing behind other candidates.
FRONTIERCS2_TIMEOUTS = {
    "join_tree_rewriting": 2_700,
    "approximate_mips_query_stopping": 2_700,
    "lejepa_world_model": 1_200,
    "polar_express": 5_400,
}
EXAMPLE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    task: str
    state_dir: Path
    config_path: Path
    model: str
    service_url: str
    token: str
    judge_url: str
    verifier_timeout_s: float
    groups: int
    rollouts: int
    steps: int
    max_tokens: int
    sequence_length: int
    lora_rank: int
    tensor_parallel_size: int
    max_workers: int
    backend: ExecutionBackend

    @property
    def task_dir(self) -> Path:
        return EXAMPLE_DIR / "harbor" / self.task

    @property
    def scenario(self) -> str:
        return f"guidance-ttt-{self.task.replace('_', '-')}"

    def execution_context(self) -> str:
        """The repository code that prepare.py wrote for a FrontierCS 2 task; empty for the other tasks."""
        path = self.state_dir / "execution_context.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def contract(self) -> TaskContract:
        return TaskContract.load(
            self.task_dir / "contract.json",
            problem_prompt=(self.task_dir / "instruction.md").read_text(),
            execution_context=self.execution_context(),
        )

    def validate_state(self) -> None:
        """Reject reuse of a task directory with a different task or executor."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "task": self.task,
            "model": self.model,
            "contract_sha256": hashlib.sha256(
                json.dumps(asdict(self.contract()), sort_keys=True).encode()
            ).hexdigest(),
            "executor_sha256": hashlib.sha256(
                json.dumps(self.backend.safe_dict(), sort_keys=True).encode()
            ).hexdigest(),
        }
        path = self.state_dir / "task-identity.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("task, model, contract, or executor changed; use a new GUIDANCE_STATE_DIR")
        else:
            path.write_text(json.dumps(identity, indent=2) + "\n")

    @classmethod
    def load(cls) -> RunConfig:
        task = os.environ.get("GUIDANCE_TASK", TASKS[0])
        if task not in TASKS + FRONTIERCS2_TASKS:
            raise ValueError(f"unknown GUIDANCE_TASK {task!r}; choose {', '.join(TASKS + FRONTIERCS2_TASKS)}")
        config_path = Path(os.environ.get("GUIDANCE_CONFIG", EXAMPLE_DIR / "serve.yaml")).resolve()
        stack = yaml.safe_load(config_path.read_text())
        grid = stack["pipeline"]["record2record"]["config"]
        training = stack["training"]["config"]
        groups, rollouts = int(grid["groups_per_step"]), int(grid["rollouts_per_group"])
        if groups < 1 or rollouts < 2:
            raise ValueError("Guidance-TTT needs at least one group and two rollouts per group")
        if int(stack["training"]["options"]["global-batch-size"]) != groups * rollouts:
            raise ValueError("global-batch-size must equal groups_per_step * rollouts_per_group")
        workers = int(os.environ.get("GUIDANCE_MAX_WORKERS", "8"))
        executor = os.environ.get("GUIDANCE_EXECUTOR", "local")
        if executor == "local":
            backend = gpt_oss_120b_backend(
                base_url=os.environ.get("GUIDANCE_EXECUTOR_URL", "http://127.0.0.1:8000/v1"),
                concurrency=workers,
            )
        elif executor == "openrouter":
            backend = openrouter_glm_5_2_backend(concurrency=workers)
        elif executor == "openai":
            backend = openai_compatible_backend(
                model=os.environ["GUIDANCE_EXECUTOR_MODEL"],
                base_url=os.environ["GUIDANCE_EXECUTOR_URL"],
                api_key_env=os.environ.get("GUIDANCE_EXECUTOR_API_KEY_ENV", "OPENAI_API_KEY"),
                api_key_file=os.environ.get("GUIDANCE_EXECUTOR_API_KEY_FILE"),
                concurrency=workers,
            )
        else:
            raise ValueError("GUIDANCE_EXECUTOR must be local, openrouter, or openai")
        if "GUIDANCE_EXECUTOR_MAX_TOKENS" in os.environ:
            backend = replace(backend, max_tokens=int(os.environ["GUIDANCE_EXECUTOR_MAX_TOKENS"]))
        default_timeout = {
            "polyomino_packing": 340,
            "lasso_path": 660,
            "ahc058": 530,
            "trimul": 1160,
            **FRONTIERCS2_TIMEOUTS,
        }[task]
        state_dir = Path(os.environ.get("GUIDANCE_STATE_DIR", EXAMPLE_DIR / "work" / task)).resolve()
        port = int(stack["reef"]["port"])
        return cls(
            task=task,
            state_dir=state_dir,
            config_path=config_path,
            model=os.environ.get("GUIDANCE_MODEL", "Qwen/Qwen3-8B"),
            service_url=f"http://127.0.0.1:{port}",
            token=str(stack["reef"]["token"]),
            judge_url=os.environ.get("GUIDANCE_JUDGE_URL", f"http://127.0.0.1:{8081 if task == TASKS[0] else 8082}"),
            verifier_timeout_s=float(os.environ.get("GUIDANCE_VERIFIER_TIMEOUT", default_timeout)),
            groups=groups,
            rollouts=rollouts,
            steps=int(training["steps"]),
            max_tokens=int(training["max_tokens"]),
            sequence_length=int(training["seq_length"]),
            lora_rank=int(training["lora_rank"]),
            tensor_parallel_size=int(training["tensor_parallel_size"]),
            max_workers=workers,
            backend=backend,
        )
