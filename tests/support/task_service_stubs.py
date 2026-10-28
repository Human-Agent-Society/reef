"""Test doubles for the generator service: a Harbor task document and a service in memory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.core.tasks import HarborTask, HarborTaskConflict, HarborTaskError, read_harbor_task, write_harbor_task
from reef.harness.client.tasks import RolloutTurn, TaskRollout
from reef.record2sample.generator import (
    DesignerRequest,
    DuplicateTask,
    Generator,
    GeneratorError,
    OracleResult,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
    content_hash,
)

DOCUMENT = {
    "instruction": (
        "A service on this machine writes the port it listens on under /var/run. Find that file and write the "
        "port number, and nothing else, to /workspace/port.txt."
    ),
    "environment": {
        "Dockerfile": "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y tmux && echo 8471 > /var/run/app.port\nWORKDIR /workspace\n"
    },
    "tests": {
        "test.sh": '#!/bin/sh\nmkdir -p /logs/verifier\ntest "$(cat /workspace/port.txt)" = 8471 && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n'
    },
    "solution": {"solve.sh": "#!/bin/sh\ncat /var/run/app.port > /workspace/port.txt\n"},
    "hint": "Look under /var/run for what the service left behind.",
}


def task_for(name: str, record_id: str, port: int, *, skill: str | None = None, hint: bool = True) -> HarborTask:
    """A valid Harbor task; ``port`` makes its content, and so its digest, distinct."""
    document = json.loads(json.dumps(DOCUMENT).replace("8471", str(port)))
    solution = dict(document["solution"])
    if hint:
        solution["hint.txt"] = document["hint"] + "\n"
    parts = name.split("-")
    metadata: dict[str, object] = (
        {"generation": int(parts[1]), "index": int(parts[2]), "step": 0}
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
        else {}
    )
    if skill is not None:
        metadata["skill"] = skill
    return HarborTask(
        name=name,
        instruction=document["instruction"] + "\n",
        tests=document["tests"],
        environment=document["environment"],
        solution=solution,
        metadata=metadata,
        source_agent_record_ids=(record_id,),
    )


def staged_digest(path: Path) -> str | None:
    """The digest of a staged task, None for a copy that cannot be read, as the service treats one."""
    try:
        return read_harbor_task(path).digest
    except HarborTaskError:
        return None


def rollout_reward(rewards: float | Sequence[float], index: int) -> float:
    """The reward of a task's rollout ``index``: one reward for every rollout, or the rewards in turn."""
    if isinstance(rewards, (int, float)):
        return float(rewards)
    return float(rewards[index % len(rewards)])


def rollout_turn(model: str | None, rollout: int, runtime_load_id: str | None = None) -> RolloutTurn:
    """One answered model call of a rollout, with the training block the service returns."""
    training: dict[str, object] = {
        "tokens": [1, 2, 3, rollout],
        "loss_mask": [1, 1],
        "rollout_log_probs": [-0.1, -0.2],
    }
    if runtime_load_id is not None:
        training["runtime_load_id"] = runtime_load_id
    return RolloutTurn(
        request={"model": model or "served", "messages": [{"role": "user", "content": f"solve {rollout}"}]},
        response={"choices": [{"message": {"role": "assistant", "content": "done"}}], "training": training},
    )


class StandInGenerator(Generator):
    """A generator service in memory: scripted refusals, checks and rewards; tasks written under a root."""

    def __init__(
        self,
        root: Path,
        *,
        refusals: Mapping[int, str] | None = None,
        same_port: bool = False,
        is_solvable: bool = True,
        rewards: Mapping[str, float | Sequence[float]] | None = None,
        rollout_error: str = "",
        fail_after: int | None = None,
        taken: Sequence[str] = (),
        check_fail_after: int | None = None,
    ) -> None:
        self.root = root
        self.refusals = dict(refusals or {})
        self.same_port = same_port
        # Names no write ever takes, the way a directory that is no task of reef's holds one.
        self.taken = set(taken)
        self.is_solvable = is_solvable
        self.rewards = dict(rewards or {"plain": 0.25, "hint": 0.75})
        self.rollout_error = rollout_error
        #: The version that answers each rollout call; None names none.
        self.runtime_load_id: str | None = None
        self.fail_after = fail_after
        self.check_fail_after = check_fail_after
        self.proposals: list[dict[str, object]] = []
        self.writes: list[str] = []
        self.checks: list[Path] = []
        self.rollout_requests: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.rollouts = 0
        self.task_rollouts: dict[str, int] = {}

    async def propose(
        self, request: DesignerRequest, *, scenario, generation, index, tags, model=None
    ) -> ProposedTask:
        self.proposals.append(
            {
                "request": request,
                "scenario": scenario,
                "generation": generation,
                "index": index,
                "tags": dict(tags),
                "model": model,
            }
        )
        if self.fail_after is not None and len(self.proposals) > self.fail_after:
            raise GeneratorError("the generator went away")
        record_id = f"designer-{len(self.proposals)}"
        if index in self.refusals:
            return ProposedTask(record_id=record_id, task=None, refusal=self.refusals[index])
        name = f"harbor-{generation:05d}-{index:03d}" + (f"-{request.skill}" if request.skill else "")
        port = 8471 if self.same_port else 8471 + len(self.proposals)
        return ProposedTask(record_id=record_id, task=task_for(name, record_id, port, skill=request.skill), refusal="")

    async def report_proposal(self, record_id, *, scenario, score, metadata) -> str:
        self.reports.append({"record_id": record_id, "scenario": scenario, "score": score, "metadata": dict(metadata)})
        return f"report-{len(self.reports)}"

    async def write_task(self, task: HarborTask) -> WrittenTask:
        self.writes.append(task.name)
        staged = self.root / task.name
        if (staged / "task.toml").is_file() and staged_digest(staged) == task.digest:
            return WrittenTask(path=staged, name=task.name, digest=task.digest, created=False)
        known = (
            {
                content_hash(read_harbor_task(entry))
                for entry in self.root.iterdir()
                if entry.is_dir() and (entry / "task.toml").is_file()
            }
            if self.root.is_dir()
            else set()
        )
        if content_hash(task) in known:
            raise DuplicateTask("duplicate of a task already under the root")
        if task.name in self.taken:
            raise TaskNameConflict(f"a different task holds the name {task.name}: it is not a task reef wrote")
        try:
            path = write_harbor_task(task, self.root)
        except HarborTaskConflict as exc:
            raise TaskNameConflict(f"a different task holds the name {task.name}: {exc}") from exc
        return WrittenTask(path=path, name=task.name, digest=task.digest)

    async def delete_task(self, name: str) -> None:
        self.deleted.append(name)
        shutil.rmtree(self.root / name, ignore_errors=True)

    async def check(self, task_path: Path) -> OracleResult:
        self.checks.append(task_path)
        if self.check_fail_after is not None and len(self.checks) > self.check_fail_after:
            raise GeneratorError("job 7 failed: OracleUnavailable: the harbor command line is not installed")
        if self.is_solvable:
            return OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0)
        return OracleResult(is_solvable=False, reason="the oracle scored 0", oracle_reward=0.0)

    async def run_rollouts(self, task_path, *, scenario, arm, count, extra_instruction_files, tags, model=None):
        self.rollout_requests.append(
            {
                "task": task_path.name,
                "arm": arm,
                "count": count,
                "extra": tuple(extra_instruction_files),
                "tags": dict(tags),
                "model": model,
            }
        )
        rows = []
        for _ in range(count):
            self.rollouts += 1
            index = self.task_rollouts.get(task_path.name, 0)
            self.task_rollouts[task_path.name] = index + 1
            reward = None if self.rollout_error else rollout_reward(self.rewards[arm], index)
            rows.append(
                TaskRollout(
                    task_path=task_path,
                    name=task_path.name,
                    rollout_id=f"rollout-{self.rollouts}",
                    reward=reward,
                    rewards={} if reward is None else {"reward": reward},
                    error=self.rollout_error,
                    receipts=(),
                    failed_calls=0,
                    report_agent_record_ids=(),
                    trial_uri=None,
                    turns=() if self.rollout_error else (rollout_turn(model, self.rollouts, self.runtime_load_id),),
                )
            )
        return tuple(rows)
