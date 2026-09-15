"""Stand ins for SPADE's Designer, checks and Reasoning Agent, shared by the generation and rounds tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from recipes.beta.spade import OracleResult
from recipes.beta.spade.generation import Checks, Designer, DesignerAnswer, ReasoningAgent
from reef.harness.client.tasks import TaskPlay

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
REPLY = "```json\n" + json.dumps(DOCUMENT) + "\n```\n"


def reply_for(index: int) -> str:
    """A distinct task per proposal: the port differs, so the content hash differs."""
    document = json.loads(json.dumps(DOCUMENT))
    port = 8471 + index
    document["environment"]["Dockerfile"] = document["environment"]["Dockerfile"].replace("8471", str(port))
    document["tests"]["test.sh"] = document["tests"]["test.sh"].replace("8471", str(port))
    return "```json\n" + json.dumps(document) + "\n```\n"


class StandInDesigner(Designer):
    """Answers with a distinct task per call (or a scripted reply) and keeps the reports it gets."""

    def __init__(self, scripted: Sequence[str] = ()) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []

    def answer(self, messages, *, tags) -> DesignerAnswer:
        self.calls.append({"messages": list(messages), "tags": dict(tags)})
        text = self.scripted.pop(0) if self.scripted else reply_for(len(self.calls))
        return DesignerAnswer(text=text, record_id=f"designer-{len(self.calls)}")

    def report(self, record_id, *, score, metadata, feedback="SPADE Designer regret") -> str:
        self.reports.append({"record_id": record_id, "score": score, "metadata": dict(metadata), "feedback": feedback})
        return f"report-{len(self.reports)}"


class StandInChecks(Checks):
    def __init__(self, *, is_solvable: bool = True) -> None:
        self.is_solvable = is_solvable
        self.calls: list[Path] = []

    def oracle(self, task_path) -> OracleResult:
        self.calls.append(task_path)
        if not self.is_solvable:
            return OracleResult(is_solvable=False, reason="the oracle scored 0", oracle_reward=0.0, nop_reward=0.0)
        return OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0)


class StandInReasoningAgent(ReasoningAgent):
    """Scripted rewards per arm; records how each arm was asked for."""

    def __init__(self, *, plain: float = 0.25, hint: float = 0.75, error: str = "") -> None:
        self.rewards = {"plain": plain, "hint": hint}
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []
        self.episodes = 0
        self.reported_episodes = 0

    def report_plays(self, plays, *, reward, metadata) -> tuple[TaskPlay, ...]:
        self.reports.append(
            {
                "plays": [play.name for play in plays],
                "scores": [reward.score(play) for play in plays],
                "metadata": dict(metadata),
            }
        )
        reported = []
        for play in plays:
            score = reward.score(play)
            if score is None or not play.receipts:
                reported.append(play)
                continue
            self.reported_episodes += 1
            reported.append(replace(play, report_agent_record_ids=(f"rep-{self.reported_episodes}",)))
        return tuple(reported)

    def play(self, task_path, *, arm, plays, is_reporting, extra_instruction_paths, tags) -> tuple[TaskPlay, ...]:
        self.calls.append(
            {
                "task": task_path.name,
                "arm": arm,
                "plays": plays,
                "is_reporting": is_reporting,
                "extra": [Path(path) for path in extra_instruction_paths],
                "tags": dict(tags),
            }
        )
        reward = None if self.error else self.rewards[arm]
        played = []
        for _ in range(plays):
            self.episodes += 1
            played.append(
                TaskPlay(
                    task_path=task_path,
                    name=task_path.name,
                    episode_id=f"episode-{self.episodes}",
                    reward=reward,
                    rewards={} if reward is None else {"reward": reward},
                    error=self.error,
                    receipts=() if self.error else (f"rec-{self.episodes}",),
                    failed_calls=0,
                    report_agent_record_ids=(f"rep-{self.episodes}",) if is_reporting and not self.error else (),
                    trial_uri=None,
                    labels={**tags, "arm": arm},
                )
            )
        return tuple(played)
