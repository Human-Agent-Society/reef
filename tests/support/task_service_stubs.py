"""Test doubles for the generator service: a Harbor task document and a service in memory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.core.tasks import HarborTask, HarborTaskConflict, read_harbor_task, write_harbor_task
from reef.harness.client.tasks import TaskPlay
from reef.record2task.generator import (
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


class StandInGenerator(Generator):
    """A generator service in memory: scripted refusals, checks and rewards; tasks written under a root."""

    def __init__(
        self,
        root: Path,
        *,
        refusals: Mapping[int, str] | None = None,
        same_port: bool = False,
        is_solvable: bool = True,
        rewards: Mapping[str, float] | None = None,
        play_error: str = "",
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
        self.play_error = play_error
        self.fail_after = fail_after
        self.check_fail_after = check_fail_after
        self.proposals: list[dict[str, object]] = []
        self.writes: list[str] = []
        self.checks: list[Path] = []
        self.plays: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.episodes = 0

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

    async def play(self, task_path, *, scenario, arm, plays, is_reporting, extra_instruction_files, tags, model=None):
        self.plays.append(
            {
                "task": task_path.name,
                "arm": arm,
                "plays": plays,
                "is_reporting": is_reporting,
                "extra": tuple(extra_instruction_files),
                "tags": dict(tags),
                "model": model,
            }
        )
        reward = None if self.play_error else self.rewards[arm]
        rows = []
        for _ in range(plays):
            self.episodes += 1
            rows.append(
                TaskPlay(
                    task_path=task_path,
                    name=task_path.name,
                    episode_id=f"episode-{self.episodes}",
                    reward=reward,
                    rewards={} if reward is None else {"reward": reward},
                    error=self.play_error,
                    receipts=() if self.play_error else (f"rec-{self.episodes}",),
                    failed_calls=0,
                    report_agent_record_ids=(f"rep-{self.episodes}",) if is_reporting and not self.play_error else (),
                    trial_uri=None,
                )
            )
        return tuple(rows)
