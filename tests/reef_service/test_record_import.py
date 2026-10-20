"""Imported examples and subsequent chat traffic use the same ingestion path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.runtime_stubs import StubTrainingRuntime, runtime_bindings
from tests.train.test_reef_trainer_contracts import ExampleBackend

from recipes.sao.recipe import SAORecipe
from reef.artifact import InMemoryRepositoryBackend
from reef.core.reports import ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2record import ReportGrouping, ReportRecord2Record
from reef.record2task import ReportTasks
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.service.routes.records import MAX_RECORD_IMPORT_BYTES
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.task2batch import Task2Batch
from reef.train import PreparedStep, Trainer, TrainingBatch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem

HEADERS = {"x-reef-scenario": "s", "Authorization": "Bearer import-test-token"}
IMAGE_URL_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
ANTHROPIC_IMAGE_BLOCK = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}


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


class FeedbackTasks(ReportTasks):
    def make_sample(self, context):
        return TaskItem(Path(context.report.agent_record_id), metadata=context.inferences[0].payload)


class FeedbackBatches(Task2Batch):
    def make_batch(self, tasks, batch_id):
        return TrainingBatch(batch_id, tasks)


def feedback_pipeline(context, batch_size):
    return TrainingPipeline(
        context,
        record2record=ReportRecord2Record(context, ReportGrouping()),
        record2task=FeedbackTasks(),
        task2batch=FeedbackBatches(context, batch_size),
    )


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

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: feedback_pipeline(context, batch_size=2),
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


@pytest.mark.parametrize("endpoint", ("/v1/chat/completions", "/v1/messages"))
def test_large_inference_context_reaches_the_handler(tmp_path, endpoint):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    handler = EchoHandler()
    text = "x" * (3 * 1024 * 1024)

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            response = await client.post(
                endpoint, headers=HEADERS, json={"messages": [{"role": "user", "content": text}]}
            )
            assert response.status == 200, await response.text()
            assert (await response.json())["choices"][0]["message"]["content"] == f"answer:{text}"
            assert handler.calls == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("bulk", (False, True))
def test_record_import_takes_a_long_trajectory_and_keeps_its_body_limit(tmp_path, bulk):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    long_record = imported("long", {"text": "x" * (3 * 1024 * 1024)})
    too_large = imported("too-large", {"text": "x" * MAX_RECORD_IMPORT_BYTES})

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            endpoint = "/reef/records/batch" if bulk else "/reef/records"
            for record, status in ((long_record, 200), (too_large, 413)):
                body = {"records": [record]} if bulk else record
                response = await client.post(endpoint, headers=HEADERS, json=body)
                assert response.status == status, await response.text()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("bulk", (False, True))
def test_import_then_chat_continue_with_the_same_reported_processor(tmp_path, bulk):
    recipe = LearningRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)
    handler = EchoHandler()

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            # Interleave records and their feedback just as an online producer
            # does, so the existing processor can consume each ready batch.
            records = []
            for index in range(6):
                name = f"offline-{index}"
                records.extend(
                    [
                        imported(name),
                        imported(f"score-{name}", {"score": 1, "references": [name]}, "report"),
                    ]
                )
            if bulk:
                response = await client.post("/reef/records/batch", headers=HEADERS, json={"records": records})
                assert response.status == 200, await response.text()
                receipts = (await response.json())["records"]
                assert [item["agent_record_id"] for item in receipts] == [item["agent_record_id"] for item in records]
            else:
                for record in records:
                    response = await client.post("/reef/records", headers=HEADERS, json=record)
                    assert response.status == 200, await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            processor = scenario.trainer.pipeline
            await wait_for_training(scenario, 6)
            if bulk:
                retry = await client.post("/reef/records/batch", headers=HEADERS, json={"records": records})
                assert retry.status == 200
                assert len(recipe.batches) == 3
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
            assert scenario.trainer.pipeline is processor
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


def test_batch_admission_is_atomic_and_ordered(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            endpoint = "/reef/records/batch"
            response = await client.post(endpoint, json={"records": [imported("first")]})
            assert response.status == 401
            invalid = [
                [],
                {},
                {"records": []},
                {"records": [1]},
                {"records": [{}]},
                {"records": [imported("first")], "scenario": "other"},
                {"records": [imported(str(index)) for index in range(1001)]},
                {"records": [imported("first"), imported("bad", {"references": ["first"]}, "report")]},
                {"records": [imported("first"), imported("bad", {"score": 1, "references": ["missing"]}, "report")]},
                {"records": [imported("bad", {"score": 1, "references": ["first"]}, "report"), imported("first")]},
                {"records": [imported("first"), imported("instruction", {}, "train")]},
            ]
            for body in invalid:
                response = await client.post(endpoint, headers=HEADERS, json=body)
                assert response.status == 400, await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.records.count("s") == 0

            body = {"records": [imported("first"), imported("first", inference_payload("conflict"))]}
            response = await client.post(endpoint, headers=HEADERS, json=body)
            assert response.status == 409
            assert scenario.records.count("s") == 0
            response = await client.post(endpoint, headers=HEADERS, json={"records": [imported("first")] * 2})
            assert response.status == 200
            assert scenario.records.count("s") == 1
            response = await client.post(
                endpoint,
                headers=HEADERS,
                json={
                    "records": [
                        imported("new"),
                        imported("first", inference_payload("conflict")),
                    ]
                },
            )
            assert response.status == 409
            assert scenario.records.get("s", "new") is None
            response = await client.post(
                endpoint,
                headers={**HEADERS, "x-reef-scenario": "other"},
                json={
                    "records": [imported("bad", {"score": 1, "references": ["first"]}, "report")],
                },
            )
            assert response.status == 400
            response = await client.post(
                endpoint,
                headers=HEADERS,
                json={"records": [imported("large", {"text": "x" * MAX_RECORD_IMPORT_BYTES})]},
            )
            assert response.status == 413
            assert scenario.records.get("s", "large") is None
            assert not dispatcher.recipe.batches

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@dataclass(frozen=True, kw_only=True)
class ImageRefusingRecipe(LearningRecipe):
    """Test double that overrides the refusal rule: it refuses image content without training weights.

    So it breaks the base definition of ``Recipe.refuses_image_content`` on purpose. It is only for tests of
    the refusal path: it isolates the service and ingress checks from the ``WeightTrainingRecipe`` rule.
    ``REFUSING_RECIPES`` also runs the real ``SAORecipe``, so the rule itself is covered too.
    """

    @property
    def refuses_image_content(self):
        return True


#: Recipes that refuse image content: one that overrides the property, and a real weight-training recipe, whose
#: refusal comes from the WeightTrainingRecipe rule through the scenario factory and binding.
REFUSING_RECIPES = {
    "image-refusing": ImageRefusingRecipe,
    "sao": lambda: SAORecipe(**runtime_bindings(StubTrainingRuntime())),
}


def deeply_nested_image(depth):
    """An image part under ``depth`` levels of nested lists."""
    content = [IMAGE_URL_PART]
    for _ in range(depth):
        content = [content]
    return content


@pytest.mark.parametrize(
    ("endpoint", "body"),
    (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [IMAGE_URL_PART]}]}),
        ("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": [IMAGE_URL_PART]}]}),
        ("/v1/messages", {"max_tokens": 8, "messages": [{"role": "user", "content": [ANTHROPIC_IMAGE_BLOCK]}]}),
        ("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": [ANTHROPIC_IMAGE_BLOCK]}]}),
        (
            "/v1/messages",
            {
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call-1", "content": [ANTHROPIC_IMAGE_BLOCK]}
                        ],
                    }
                ],
            },
        ),
        (
            "/v1/responses",
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
                ]
            },
        ),
        (
            "/v1/responses",
            {
                "stream": True,
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
                ],
            },
        ),
        (
            "/v1/responses",
            {
                "input": [
                    {
                        "type": "computer_call_output",
                        "call_id": "call-1",
                        "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
                    }
                ]
            },
        ),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": deeply_nested_image(1000)}]}),
    ),
    ids=(
        "openai",
        "openai-stream",
        "anthropic",
        "anthropic-count-tokens",
        "anthropic-tool-result",
        "responses",
        "responses-stream",
        "responses-computer-screenshot",
        "openai-deeply-nested",
    ),
)
@pytest.mark.parametrize("recipe", sorted(REFUSING_RECIPES))
def test_a_recorded_image_request_is_refused_when_the_recipe_refuses_image_content(tmp_path, recipe, endpoint, body):
    dispatcher = dispatcher_for(tmp_path, REFUSING_RECIPES[recipe]())
    handler = EchoHandler()

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            response = await client.post(endpoint, headers=HEADERS, json={"model": "example-model", **body})
            assert response.status == 400
            assert "does not support images" in await response.text()
            assert handler.calls == 0
            assert dispatcher.get_or_create_scenario("s").records.replay("s") == ()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_image_evaluation_call_is_served_because_it_stores_nothing(tmp_path):
    dispatcher = dispatcher_for(tmp_path, ImageRefusingRecipe())
    handler = EchoHandler()
    messages = [{"role": "user", "content": [IMAGE_URL_PART]}]

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            scenario = dispatcher.get_or_create_scenario("s")
            response = await client.post(
                "/reef/scenarios/s/evaluation/v1/chat/completions",
                headers=HEADERS,
                json={"model": "example-model", "messages": messages},
            )
            assert response.status == 200, await response.text()
            assert handler.calls == 1
            assert scenario.records.replay("s") == ()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_recipe_that_does_not_train_weights_keeps_image_requests_unchanged(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    handler = EchoHandler()
    messages = [{"role": "user", "content": [{"type": "text", "text": "look"}, IMAGE_URL_PART]}]

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            response = await client.post(
                "/v1/chat/completions", headers=HEADERS, json={"model": "example-model", "messages": messages}
            )
            assert response.status == 200, await response.text()
            assert handler.calls == 1
            [record] = dispatcher.get_or_create_scenario("s").records.replay("s")
            assert record.payload["messages"] == messages

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def image_record_payloads():
    in_messages = inference_payload("look")
    in_messages["messages"] = [{"role": "user", "content": [{"type": "text", "text": "look"}, IMAGE_URL_PART]}]
    in_training = inference_payload("look")
    in_training["response"] = {
        **in_training["response"],
        "training": {"request_messages": [{"role": "user", "content": [IMAGE_URL_PART]}]},
    }
    deeply_nested = inference_payload("look")
    deeply_nested["messages"] = [{"role": "user", "content": deeply_nested_image(1000)}]
    in_document = inference_payload("look")
    document = {"type": "document", "source": {"type": "content", "content": [ANTHROPIC_IMAGE_BLOCK]}}
    in_document["messages"] = [{"role": "user", "content": [document]}]
    return {
        "messages": in_messages,
        "training-request-messages": in_training,
        "deeply-nested": deeply_nested,
        "anthropic-document": in_document,
    }


@pytest.mark.parametrize("recipe", sorted(REFUSING_RECIPES))
@pytest.mark.parametrize("shape", ("messages", "training-request-messages", "deeply-nested", "anthropic-document"))
@pytest.mark.parametrize("bulk", (False, True))
def test_an_image_record_is_refused_and_not_stored_when_the_recipe_refuses_image_content(
    tmp_path, bulk, shape, recipe
):
    dispatcher = dispatcher_for(tmp_path, REFUSING_RECIPES[recipe]())
    payload = image_record_payloads()[shape]

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            if bulk:
                # A batch is atomic: the text-only record before the refused one is not stored either.
                records = [imported("text-only"), imported("with-image", payload)]
                response = await client.post("/reef/records/batch", headers=HEADERS, json={"records": records})
            else:
                response = await client.post("/reef/records", headers=HEADERS, json=imported("with-image", payload))
            assert response.status == 400
            assert "record 'with-image' carries image content" in await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.records.replay("s") == ()
            assert scenario.operations.snapshot()["ingest/rejected_request_total"] == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_record_whose_part_type_is_not_a_string_is_checked_without_a_server_error(tmp_path):
    dispatcher = dispatcher_for(tmp_path, ImageRefusingRecipe())
    payload = inference_payload("look")
    payload["messages"] = [{"role": "user", "content": [{"type": ["image"]}, {"type": {"name": "image"}}]}]

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            response = await client.post("/reef/records", headers=HEADERS, json=imported("odd-type", payload))
            # Not an image part type, so the image check does not refuse it.
            assert response.status == 200, await response.text()
            assert len(dispatcher.get_or_create_scenario("s").records.replay("s")) == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("bulk", (False, True))
def test_an_identical_retry_of_a_stored_image_record_keeps_its_receipt(tmp_path, bulk):
    from reef.core import AgentRecord, RequestType

    dispatcher = dispatcher_for(tmp_path, ImageRefusingRecipe())
    payload = image_record_payloads()["messages"]

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            # A record stored before the check, written to the store directly.
            scenario = dispatcher.get_or_create_scenario("s")
            scenario.records.append(
                AgentRecord.create(
                    scenario="s", request_type=RequestType.INFERENCE, agent_record_id="with-image", payload=payload
                )
            )
            if bulk:
                response = await client.post(
                    "/reef/records/batch", headers=HEADERS, json={"records": [imported("with-image", payload)]}
                )
            else:
                response = await client.post("/reef/records", headers=HEADERS, json=imported("with-image", payload))
            assert response.status == 200, await response.text()
            assert len(scenario.records.replay("s")) == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
