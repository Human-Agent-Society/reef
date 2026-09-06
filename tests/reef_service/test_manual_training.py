"""Native manual scheduling: instructions authorize one durable step without a data batch."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.recipe import RecipeConfigError
from reef.records import RecordStore
from reef.service.app import create_app
from reef.train.backend import PreparedStep
from reef.train.cordis_backend.processor import CordisProcessor, RecordDrivenTraceProcessor
from reef.train.processors.base import DataProcessor
from reef.train.processors.manual import ManualTrainingProcessor
from reef.train.processors.modes import ModeDataProcessor
from reef.train.trainer import Trainer
from reef.train.types import ProcessorContext, TraceBatch

from .test_harness_proposals import _dispatcher, _recipe
from .test_reef_trainer_contracts import ExampleBackend, ExampleBatch


class CaptureBackend(ExampleBackend):
    def __init__(self, *, dispatched=False):
        super().__init__("s", [])
        self.batches = []
        self._dispatched = dispatched

    @property
    def dispatched(self):
        return self._dispatched

    def prepare_step(self, batch, state, scenario_step):
        self.batches.append(batch)
        return PreparedStep.skipped(state={"steps": state.get("steps", 0) + 1})


def inference(receipt, scenario="s"):
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": receipt}]},
        agent_record_id=receipt,
    )


def instruction(receipt):
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.TRAIN,
        payload={"text": receipt, "session": "session-1", "release_id": "release-1"},
        agent_record_id=receipt,
    )


def build(records, backend, processor=RecordDrivenTraceProcessor, mode="manual", batch_size=1):
    return Trainer.build(
        "s",
        records,
        processor_factory=lambda ctx: processor(ctx.with_config({"batch_size": batch_size})),
        training_backend=backend,
        training_mode=mode,
    )


@pytest.mark.parametrize("processor", [CordisProcessor, RecordDrivenTraceProcessor])
@pytest.mark.parametrize("batch_size", [1, 100])
def test_manual_waits_for_instruction_without_automatic_batch_gates(processor, batch_size):
    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, batch_size=batch_size)
    for receipt in ("other-session", "turn-1", "turn-2"):
        records.append(inference(receipt))
    records.append(
        AgentRecord.create(
            scenario="s", request_type=RequestType.REPORT, payload={"score": 0, "references": ["other-session"]}
        )
    )
    assert trainer.run_once() is None
    assert not trainer.batch_ready()
    records.append(instruction("request-1"))
    result = trainer.run_once()
    assert result is not None
    batch = backend.batches[0]
    assert batch.request.text == "request-1"
    assert batch.samples == ()
    prepared = trainer.prepare_commit(result)
    assert prepared.consumed_ids == frozenset({"request-1"})
    assert prepared.metrics["training_request"]["text"] == "request-1"
    trainer.commit(prepared)
    trainer.apply_compaction(prepared.compacted_ids)
    assert records.get("s", "turn-1") is not None
    assert trainer.run_once() is None
    assert not records.append_result(instruction("request-1")).inserted
    assert trainer.run_once() is None
    trainer.close()
    records.close()


def test_auto_keeps_recipe_batching():
    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, mode="auto", batch_size=2)
    records.append(inference("a"))
    assert trainer.run_once() is None
    records.append(inference("b"))
    assert trainer.run_once() is not None
    assert [sample.source_agent_record_id for sample in backend.batches[0].samples] == ["a", "b"]
    assert backend.batches[0].request is None
    trainer.close()
    records.close()


def test_dispatched_manual_reserves_one_instruction_and_leaves_the_next_pending():
    records, backend = RecordStore(), CaptureBackend(dispatched=True)
    trainer = build(records, backend)
    records.append(inference("a"))
    assert trainer.reserve_training_batch() is None
    records.append(instruction("one"))
    first = trainer.reserve_training_batch()
    records.append(instruction("two"))
    assert trainer.reserve_training_batch() is first
    result = trainer.execute_reserved_step(0).result
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    trainer.apply_compaction(prepared.compacted_ids)
    second = trainer.reserve_training_batch()
    assert second.request.text == "two"
    assert second.samples == ()
    trainer.close()
    records.close()


def test_manual_recovery_replays_pending_requests_but_not_committed_ones(tmp_path):
    path = tmp_path / "records.sqlite"
    records, backend = RecordStore(path), CaptureBackend()
    first = build(records, backend)
    records.append(inference("a"))
    records.append(instruction("committed"))
    result = first.run_once()
    prepared = first.prepare_commit(result)
    first.commit(prepared)
    # Simulate a crash after the commit log landed but before compaction.
    records.append(instruction("pending"))
    first.close()
    records.close()
    records = RecordStore(path)
    recovered = build(records, backend)
    recovered.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    recovered.reingest(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    assert recovered.run_once() is not None
    assert [batch.request.text for batch in backend.batches] == ["committed", "pending"]
    recovered.close()
    records.close()


def test_manual_requires_an_explicit_recipe_assembler():
    records = RecordStore()
    with pytest.raises(NotImplementedError, match="does not implement training_mode='manual'"):
        build(records, CaptureBackend(), DataProcessor)
    with pytest.raises(ValueError, match="training_mode"):
        build(records, CaptureBackend(), mode="typo")
    records.close()


def test_train_route_needs_no_inference_and_retries_do_not_train_twice(tmp_path):
    called = Event()
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append((requests[0], samples))
        called.set()

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", batch_size=100)
    dispatcher = _dispatcher(tmp_path, recipe)

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            for body in (
                {},
                {"text": ""},
                {"text": "x", "session": "s"},
                {"text": "x", "session": 1, "release_id": "r"},
                {"text": "x" * 4001, "session": "s", "release_id": "r"},
            ):
                response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
                assert response.status == 400, await response.text()
            assert not called.is_set()
            body = {
                "agent_record_id": "request-1",
                "text": "Prefer tests first",
                "session": "session-1",
                "release_id": "release-1",
            }
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert (await response.json())["request_type"] == "train"
            assert await asyncio.to_thread(called.wait, 5)
            assert seen[0][0]["text"] == "Prefer tests first"
            assert seen[0][1] == ()
            assert seen[0][0]["id"] == "request-1"
            retry = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert retry.status == 200
            conflict = await client.post(
                "/reef/train", headers={"x-reef-scenario": "s"}, json={**body, "text": "Different"}
            )
            assert conflict.status == 409
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
    assert len(seen) == 1


def test_auto_rejects_manual_requests(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))
    try:
        with pytest.raises(ValueError, match="training_mode='manual'"):
            dispatcher.accept_record(instruction("one"))
    finally:
        dispatcher.close()


def test_manual_is_a_native_contract_for_arbitrary_batch_schemas():
    class InstructionProcessor(ManualTrainingProcessor):
        output_schema = ExampleBatch

        def make_request_batch(self, request):
            return ExampleBatch(request.agent_record_id, (request.payload["text"],))

    with pytest.raises(NotImplementedError, match="training_mode='auto'"):
        InstructionProcessor(ProcessorContext("s"))
    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, InstructionProcessor)
    records.append(instruction("change"))
    assert trainer.run_once() is not None
    assert isinstance(backend.batches[0], ExampleBatch)
    assert backend.batches[0].values == ("change",)
    trainer.close()
    records.close()


def test_processor_context_keeps_mode_when_recipe_applies_config():
    context = ProcessorContext("s", training_mode="manual")
    configured = context.with_config({"batch_size": 100})
    assert configured.training_mode == "manual"
    assert configured.config == {"batch_size": 100}
    processor = CordisProcessor(configured)
    assert processor.training_mode == "manual"
    assert RequestType.TRAIN in processor.required_request_types
    processor.close()


def test_missing_mode_implementation_fails_at_processor_initialization():
    class AutoOnlyProcessor(ModeDataProcessor):
        mode_processors = {"auto": DataProcessor}

    with pytest.raises(NotImplementedError, match=r"AutoOnlyProcessor.*manual"):
        AutoOnlyProcessor(ProcessorContext("s", training_mode="manual"))
    with pytest.raises(ValueError, match="training_mode"):
        ProcessorContext("s", training_mode="invalid")


def test_mode_implementation_cannot_silently_fall_back_to_auto():
    class DropsModeProcessor(DataProcessor):
        def __init__(self, context):
            super().__init__(ProcessorContext(context.scenario))

    class ConfiguredProcessor(ModeDataProcessor):
        mode_processors = {"manual": DropsModeProcessor}

    with pytest.raises(ValueError, match="mode implementation must preserve"):
        ConfiguredProcessor(ProcessorContext("s", training_mode="manual"))


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_processor_owns_mode_specific_ingestion_and_readiness(mode):
    class TrajectoryProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})
        required_request_types = frozenset({RequestType.INFERENCE, RequestType.TRAIN})
        output_schema = ExampleBatch

        def __init__(self, context):
            super().__init__(context)
            self.exchanges = []
            self.request = None

        def ingest(self, item):
            super().ingest(item)
            if item.request_type is RequestType.INFERENCE:
                self.exchanges.append(item)
            elif item.request_type is RequestType.TRAIN:
                self.request = item

        def _ready_count(self):
            authorized = self.training_mode == "auto" or self.request is not None
            return int(authorized and len(self.exchanges) >= 2)

        def _make_pending(self, batch_number):
            request = (
                None
                if self.request is None
                else replace(TrainingRequest.from_dict(self.request.payload), id=self.request.agent_record_id)
            )
            return ExampleBatch(
                "custom-batch", tuple(record.agent_record_id for record in self.exchanges), request=request
            )

        def _consume_pending(self):
            consumed = {record.agent_record_id for record in self.exchanges}
            if self.request is not None:
                consumed.add(self.request.agent_record_id)
            self.exchanges.clear()
            self.request = None
            return frozenset(consumed)

    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, TrajectoryProcessor, mode=mode)
    assert type(trainer.processor) is TrajectoryProcessor
    records.append(inference("first"))
    assert trainer.run_once() is None
    if mode == "manual":
        records.append(instruction("use-the-trajectory"))
        # This processor requires trajectory data as well as the instruction;
        # the trainer must honor its readiness rather than applying a manual queue.
        assert trainer.run_once() is None
    records.append(inference("second"))
    result = trainer.run_once()
    assert result is not None
    assert backend.batches[0].values == ("first", "second")
    assert (backend.batches[0].request is None) == (mode == "auto")
    prepared = trainer.prepare_commit(result)
    expected = {"first", "second"} | ({"use-the-trajectory"} if mode == "manual" else set())
    assert prepared.consumed_ids == frozenset(expected)
    trainer.commit(prepared)
    assert trainer.run_once() is None
    trainer.close()
    records.close()


def test_factory_cannot_silently_change_the_processor_mode():
    records = RecordStore()
    with pytest.raises(ValueError, match="preserve the requested training_mode"):
        Trainer.build(
            "s",
            records,
            processor_factory=lambda ctx: DataProcessor(replace(ctx, training_mode="auto")),
            training_backend=CaptureBackend(),
            training_mode="manual",
        )
    records.close()


def test_manual_instruction_cannot_be_consumed_by_recheck_or_inbox_proposal(tmp_path):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(requests)

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", recheck_every=1)
    records = RecordStore()
    trainer = recipe.build("s", records)
    backend = trainer.training_backend
    state = dict(backend.initial_state())
    state["rollback_entries"] = state["entries"]
    backend.proposals.submit(
        "pending-proposal",
        {
            "mutations": [{"op": "create", "id": "rules", "options": {"name": "rules", "config": {"text": "marker"}}}],
            "session": "unrelated",
            "release_id": "r",
            "reason": "unrelated",
        },
    )
    batch = TraceBatch(
        "manual-request", (), request=TrainingRequest("Follow the request", "session", "r", "request-id")
    )
    prepared = backend.prepare_step(batch, state, 0)
    assert prepared.outcome == "skip"
    assert seen[0][0]["id"] == "request-id"
    assert (recipe.proposals_path("s") / "pending-proposal.json").is_file()
    assert "recheck" not in prepared.metrics
    trainer.close()
    records.close()


def test_manual_harness_requires_explicit_requests_keyword(tmp_path):
    recipe = replace(_recipe(tmp_path, lambda n, s, m, **kwargs: None), training_mode="manual")
    records = RecordStore()
    with pytest.raises(RecipeConfigError, match="requests"):
        recipe.build("s", records)
    records.close()
