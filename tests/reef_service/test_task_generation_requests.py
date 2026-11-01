"""Requested task-generation rounds: in manual mode a client names the source sessions of each round."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from aiohttp.test_utils import TestClient, TestServer
from tests.support.runtime_stubs import RecordingComponent
from tests.support.task_service_stubs import StandInGenerator, task_for
from tests.support.trajectory_uploads import HEADERS, TOKEN, UploadRecipe, dispatcher_for

from reef.core import AgentRecord, RequestType
from reef.core.tasks import HarborTask
from reef.recipe.generated_tasks import GeneratedTaskJobData, GeneratedTaskSamples, GeneratedTaskSettings
from reef.recipe.generated_tasks.generator import GeneratedTaskResult, TaskGenerator, TaskValidationResult
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.service.app import create_app
from reef.train import Trainer
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline

AUTH = {"Authorization": HEADERS["Authorization"]}
ROUTE = "/reef/scenarios/s/task-generations"


class OneTaskGenerator(TaskGenerator):
    """Makes one valid task in each generation and keeps each request."""

    def __init__(self) -> None:
        self.requests: list[TaskGenerationRequest] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        return cls()

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        self.requests.append(request)
        return (task_for(f"harbor-{len(self.requests) - 1}", "designer-0", 9000 + len(self.requests)),)

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        return TaskValidationResult(())

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class GeneratingRecipe(UploadRecipe):
    """The generated-task stages, with at most one tagged source session in each round."""

    tasks_root: Path
    task_generator: OneTaskGenerator = field(default_factory=OneTaskGenerator)

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        settings = GeneratedTaskSettings(
            rounds=3, description="shell tasks", source_tag="source=session", source_records_per_round=1
        )
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(
                context,
                record2sample=GeneratedTaskSamples(
                    context,
                    settings,
                    task_generator=self.task_generator,
                    generator_service=StandInGenerator(self.tasks_root),
                ),
                sample2job=GeneratedTaskJobData(context),
            ),
            job_loop=JobLoop(RecordingComponent(self.batches), scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            samples=samples,
        )


def session(name: str, tags: Mapping[str, str]) -> tuple[AgentRecord, AgentRecord]:
    """A recorded session of scenario ``s``: one model call with ``tags`` and the report on it."""
    call = AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": f"fix {name}"}],
            "response": {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            "metadata": {"tags": dict(tags)},
        },
        agent_record_id=f"rec-{name}",
    )
    report = AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": [call.agent_record_id]},
        agent_record_id=f"rep-{name}",
    )
    return call, report


def test_a_client_requests_a_round_of_the_sessions_it_names_and_reads_its_result(tmp_path):
    recipe = GeneratingRecipe(training_mode="manual", tasks_root=tmp_path / "tasks")
    dispatcher = dispatcher_for(tmp_path, recipe)

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/scenarios", headers=HEADERS, json={"name": "s"})
            with dispatcher.locked_scenario("s") as current:
                for name, tags in (("a", {"source": "session"}), ("b", {"source": "session"}), ("plain", {})):
                    current.store.records.append_many(session(name, tags))
            for body, status in (
                ({"report_ids": ["rep-missing"]}, 400),
                ({"report_ids": ["rec-a"]}, 400),
                ({"report_ids": ["rep-plain"]}, 400),
                ({"report_ids": ["rep-a", "rep-b"]}, 400),
                ({"ids": []}, 400),
            ):
                response = await client.post(ROUTE, headers=AUTH, json=body)
                assert response.status == status, await response.text()

            queued = await client.post(ROUTE, headers=AUTH, json={"report_ids": ["rep-b"]})
            assert queued.status == 202, await queued.text()
            request = await queued.json()
            assert (request["component"], request["report_ids"], request["result"]) == ("records", ["rep-b"], None)
            async with asyncio.timeout(10):
                while (found := await (await client.get(f"{ROUTE}/{request['id']}", headers=AUTH)).json())[
                    "state"
                ] not in ("finished", "failed"):
                    await asyncio.sleep(0.02)
            assert (found["state"], found["result"], found["error"]) == (
                "finished",
                {"generation": 0, "train_tasks": ["harbor-0"]},
                None,
            )
            (generation,) = recipe.task_generator.requests
            assert [record.agent_record_id for record in generation.source_records] == ["rec-b", "rep-b"]
            async with asyncio.timeout(5):
                while "generated-tasks/harbor-0" not in [
                    sample["sample_id"]
                    for sample in (await (await client.get("/reef/scenarios/s/samples", headers=AUTH)).json())[
                        "samples"
                    ]
                ]:
                    await asyncio.sleep(0.02)
            buffer = await (await client.get("/reef/scenarios/s/buffer", headers=AUTH)).json()
            assert buffer["samples"] == [], "the client chooses the generated tasks that train"
            assert len(recipe.task_generator.requests) == 1, "manual starts no round of its own"

            missing = await client.get(f"{ROUTE}/task_generation:s:missing", headers=AUTH)
            assert missing.status == 404
            switched = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"training_mode": "auto"})
            assert switched.status == 200, await switched.text()
            refused = await client.post(ROUTE, headers=AUTH, json={"report_ids": ["rep-a"]})
            assert refused.status == 409, await refused.text()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
