"""Records of live traffic and a generation service in memory, for the tests of the continual-environments recipe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from tests.support.task_service_stubs import task_for

from recipes.beta.continual_envs.generation_service import GenerationRun, GenerationService
from reef.core import AgentRecord, RequestType
from reef.core.tasks import write_harbor_task
from reef.core.values import JsonValue

SCENARIO = "traffic"


def call(record_id: str, messages: list[dict[str, str]]) -> AgentRecord:
    """An inference record of a chat call with ``messages``, as Reef stores it."""
    return AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        payload={
            "model": "policy",
            "messages": messages,
            "metadata": {"tags": {"source": "traffic"}},
            "artifact_id": "artifact-1",
            "response": {"choices": [{"message": {"role": "assistant", "content": f"reply of {record_id}"}}]},
        },
    )


def report(record_id: str, references: tuple[str, ...], score: float | None) -> AgentRecord:
    """A report of a session: the judge's score and feedback, and metadata that must not reach the generation run."""
    payload: dict[str, object] = {"feedback": "the agent stopped early", "metadata": {"verifier": "hidden"}}
    if score is not None:
        payload["score"] = score
    return AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.REPORT,
        agent_record_id=record_id,
        payload=payload,
        references=references,
    )


def session(name: str, score: float | None, turns: int = 2) -> tuple[AgentRecord, ...]:
    """The records of one session as a round carries them: its calls, then its report on the last call."""
    calls = tuple(
        call(f"{name}-call-{turn}", [{"role": "user", "content": f"task {name}"}] * (turn + 1))
        for turn in range(turns)
    )
    return (*calls, report(f"{name}-report", (calls[-1].agent_record_id,), score))


class MemoryGenerationService(GenerationService):
    """A generation service that keeps its runs in memory.

    Each started run ends after ``polls_to_end`` reads and publishes one task for each record folder of
    its source, whose source record is that folder. ``end_state`` is the state that a run ends in; a run that ended
    keeps its end state.
    """

    def __init__(self, *, polls_to_end: int = 1, end_state: str = "finished") -> None:
        self.polls_to_end = polls_to_end
        self.end_state = end_state
        self.started: list[dict[str, JsonValue]] = []
        self.cancelled: list[str] = []
        self.listed: list[GenerationRun] = []
        self.tasks: dict[str, tuple[str, ...]] = {}
        self.sources: dict[str, str] = {}
        self.reads: dict[str, int] = {}
        self.ended: dict[str, str] = {}

    async def start(
        self,
        *,
        spec: str,
        generator: str,
        sources: Sequence[Mapping[str, JsonValue]],
        params: Mapping[str, JsonValue],
        targets: Mapping[str, JsonValue],
    ) -> GenerationRun:
        run_id = f"run-{len(self.started) + 1}"
        self.started.append(
            {"spec": spec, "generator": generator, "sources": [dict(s) for s in sources], "params": dict(params)}
        )
        (source,) = sources
        files = source["files"]
        folders = sorted({path.split("/", 1)[0] for path in files}) if isinstance(files, dict) else []
        self.tasks[run_id] = tuple(f"fail-{len(self.sources) + index:012x}__task" for index in range(len(folders)))
        self.sources.update(zip(self.tasks[run_id], folders, strict=True))
        self.reads[run_id] = 0
        return GenerationRun(run_id, "queued", "", 100.0)

    async def run(self, run_id: str) -> GenerationRun:
        self.reads[run_id] += 1
        if run_id not in self.ended and self.reads[run_id] >= self.polls_to_end:
            self.ended[run_id] = self.end_state
        state = self.ended.get(run_id, "running")
        return GenerationRun(run_id, state, "a stage failed" if state == "failed" else "", 100.0)

    async def runs(self, generator: str) -> tuple[GenerationRun, ...]:
        return tuple(self.listed)

    async def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    async def published_tasks(self, run_id: str, stage: str) -> tuple[str, ...]:
        return self.tasks[run_id]

    async def download_task(self, name: str, directory: Path) -> Path:
        task = task_for(name, self.sources[name], 9000 + len(name))
        return write_harbor_task(task, directory)
