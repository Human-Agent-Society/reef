"""Imported examples and subsequent chat traffic use the same ingestion path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.core.reports import ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import PreparedStep, Trainer, TrainingBatch
from reef.train.processors.reported import ReportedFeedbackProcessor
from reef.train.types import TaskItem

from .test_reef_trainer_contracts import ExampleBackend


HEADERS = {"x-reef-scenario": "s", "Authorization": "Bearer import-test-token"}


def inference_payload(text):
    return {
        "model": "example-model",
        "messages": [{"role": "user", "content": text}],
        "response": {"choices": [{"message": {"role": "assistant", "content": f"answer:{text}"}}]},
        "metadata": {"source": "fixture"},
    }


def imported(record_id, payload=None, request_type="inference"):
    return {
        "agent_record_id": record_id,
        "request_type": request_type,
        "payload": inference_payload(record_id) if payload is None else payload,
    }


class FeedbackProcessor(ReportedFeedbackProcessor):
    def make_sample(self, context):
        return TaskItem(Path(context.report.agent_record_id), metadata=context.inferences[0].payload)

    def make_batch(self, items, batch_number):
        return TrainingBatch(f"batch:{batch_number}", items)


class LearningBackend(ExampleBackend):
    def __init__(self, scenario, batches):
        super().__init__(scenario, [])
        self.batches = batches

    def prepare_step(self, batch, state, scenario_step):
        self.batches.append(batch)
        return PreparedStep.skipped(state={"trained": state.get("trained", 0) + len(batch.items)})


@dataclass(frozen=True, kw_only=True)
class LearningRecipe(Recipe):
    batches: list[TrainingBatch] = field(default_factory=list)

    @property
    def report_type(self):
        return ScoredRolloutReport

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: FeedbackProcessor(context.with_config({"batch_size": 2})),
            candidate_backend=LearningBackend(scenario, self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            report_type=self.report_type,
            training_mode=self.training_mode,
        )


class EchoHandler(InferenceHandler):
    def __init__(self):
        self.calls = 0

    async def inference(self, artifact, path, payload):
        self.calls += 1
        return inference_payload(payload["messages"][0]["content"])["response"]


def dispatcher_for(tmp_path, recipe):
    initial = tmp_path / "initial"
    initial.mkdir()
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


async def wait_for_training(scenario, count):
    async with asyncio.timeout(5):
        while scenario.trainer.state.get("trained", 0) < count:
            await asyncio.sleep(0.01)
    assert scenario.trainer.state["trained"] == count


def test_import_then_chat_continue_with_the_same_reported_processor(tmp_path):
    recipe = LearningRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)
    handler = EchoHandler()

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            # Interleave records and their feedback just as an online producer
            # does, so the existing processor can consume each ready batch.
            for index in range(6):
                name = f"offline-{index}"
                response = await client.post("/reef/records", headers=HEADERS, json=imported(name))
                assert response.status == 200, await response.text()
                response = await client.post(
                    "/reef/records",
                    headers=HEADERS,
                    json=imported(f"score-{name}", {"score": 1, "references": [name]}, "report"),
                )
                assert response.status == 200, await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            processor = scenario.trainer.processor
            await wait_for_training(scenario, 6)
            assert handler.calls == 0
            assert len(recipe.batches) == 3

            # Retrying an imported record after training/compaction must not
            # reactivate it. Changed content still conflicts.
            retry = await client.post("/reef/records", headers=HEADERS, json=imported("offline-0"))
            assert retry.status == 200
            conflict = await client.post(
                "/reef/records", headers=HEADERS, json=imported("offline-0", inference_payload("changed"))
            )
            assert conflict.status == 409
            for index in range(2):
                text = f"online-{index}"
                response = await client.post(
                    "/v1/chat/completions",
                    headers=HEADERS,
                    json={"model": "example-model", "messages": [{"role": "user", "content": text}]},
                )
                assert response.status == 200, await response.text()
                receipt = response.headers["x-reef-agent-record-id"]
                feedback = await client.post(
                    "/reef/report", headers=HEADERS, json={"score": 1, "references": [receipt]}
                )
                assert feedback.status == 200, await feedback.text()
            await wait_for_training(scenario, 8)
            assert handler.calls == 2
            assert scenario.trainer.processor is processor
            assert len(scenario.store.history()) == len(recipe.batches) == 4
            samples = [sample for batch in recipe.batches for sample in batch.items]
            assert [sample.metadata["messages"][0]["content"] for sample in samples] == [
                *(f"offline-{index}" for index in range(6)),
                "online-0",
                "online-1",
            ]
            assert all(sample.metadata["response"]["choices"] for sample in samples)
            for index in range(6):
                stored = scenario.records.get_for_audit("s", f"offline-{index}")
                assert stored.item.payload == inference_payload(f"offline-{index}")

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_import_admission_authentication_and_scenario_isolation(tmp_path):
    dispatcher = dispatcher_for(tmp_path, Recipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            body = imported("first")
            response = await client.post("/reef/records", json=body, headers={"x-reef-scenario": "s"})
            assert response.status == 401
            response = await client.post(
                "/reef/records", json=body, headers={"Authorization": "Bearer import-test-token"}
            )
            assert response.status == 400
            invalid = [
                [],
                {},
                {**body, "agent_record_id": " "},
                {**body, "agent_record_id": 1},
                {**body, "request_type": "train"},
                {**body, "request_type": "other"},
                {**body, "payload": []},
                {**body, "scenario": "other"},
            ]
            for envelope in invalid:
                response = await client.post("/reef/records", headers=HEADERS, json=envelope)
                assert response.status == 400, await response.text()
            for _ in range(2):
                response = await client.post("/reef/records", headers=HEADERS, json=body)
                assert response.status == 200
                assert await response.json() == {
                    "agent_record_id": "first",
                    "scenario": "s",
                    "request_type": "inference",
                }
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.records.count("s") == 1
            assert scenario.records.get("s", "first").payload == body["payload"]
            other_headers = {**HEADERS, "x-reef-scenario": "other"}
            response = await client.post("/reef/records", headers=other_headers, json=imported("second"))
            assert response.status == 200
            other = dispatcher.get_or_create_scenario("other")
            assert other.records.get("other", "first") is None
            assert scenario.records.get("s", "second") is None
            response = await client.post(
                "/reef/records",
                headers=other_headers,
                json=imported("foreign-report", {"score": 1, "references": ["first"]}, "report"),
            )
            assert response.status == 400
            response = await client.post(
                "/reef/records",
                headers=HEADERS,
                json=imported("bad-report", {"references": ["missing"]}, "report"),
            )
            assert response.status == 400
            assert scenario.records.count("s") == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_imported_reports_follow_the_recipes_report_schema(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            response = await client.post("/reef/records", headers=HEADERS, json=imported("source"))
            assert response.status == 200
            response = await client.post(
                "/reef/records",
                headers=HEADERS,
                json=imported("missing-score", {"references": ["source"]}, "report"),
            )
            assert response.status == 400
            assert dispatcher.get_or_create_scenario("s").records.get("s", "missing-score") is None

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
