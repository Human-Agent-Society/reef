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
    class AutoOnlyProcessor(DataProcessor):
        pass

    with pytest.raises(NotImplementedError, match=r"AutoOnlyProcessor.*manual"):
        AutoOnlyProcessor(ProcessorContext("s", training_mode="manual"))
    with pytest.raises(ValueError, match="training_mode"):
        ProcessorContext("s", training_mode="invalid")


@pytest.mark.parametrize(
    "operation",
    ["ingest", "ready", "build_batch", "acknowledge", "retention_decision", "compaction_applied"],
)
def test_unimplemented_manual_hooks_never_fall_back_to_auto(operation):
    class IncompleteProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})

    processor = IncompleteProcessor(ProcessorContext("s", training_mode="manual"))
    with pytest.raises(NotImplementedError, match=f"{operation}_manual"):
        if operation == "ingest":
            processor.ingest(instruction("one"))
        elif operation == "ready":
            processor.ready()
        elif operation == "build_batch":
            processor.build_batch_manual(1)
        elif operation == "acknowledge":
            processor.acknowledge_manual()
        elif operation == "retention_decision":
            processor.retention_decision()
        else:
            processor.compaction_applied(frozenset())


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

        def ingest_auto(self, item):
            super().ingest_auto(item)
            if item.request_type is RequestType.INFERENCE:
                self.exchanges.append(item)
            elif item.request_type is RequestType.TRAIN:
                self.request = item

        def ingest_manual(self, item):
            self.ingest_auto(item)

        def ready_auto(self):
            return len(self.exchanges) >= 2

        def ready_manual(self):
            return self.request is not None and len(self.exchanges) >= 2

        def build_batch_auto(self, batch_number):
            request = (
                None
                if self.request is None
                else replace(TrainingRequest.from_dict(self.request.payload), id=self.request.agent_record_id)
            )
            return ExampleBatch(
                "custom-batch", tuple(record.agent_record_id for record in self.exchanges), request=request
            )

        def build_batch_manual(self, batch_number):
            return self.build_batch_auto(batch_number)

        def acknowledge_manual(self):
            return self.acknowledge_auto()

        def retention_decision_manual(self):
            return self.retention_decision_auto()

        def compaction_applied_manual(self, agent_record_ids):
            self.compaction_applied_auto(agent_record_ids)

        def acknowledge_auto(self):
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


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_mode_methods_share_one_reservation_until_acknowledgement(mode):
    class MethodProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})
        output_schema = ExampleBatch

        def __init__(self, context):
            super().__init__(context)
            self.calls = []
            self.available = True

        def ready_auto(self):
            self.calls.append("ready_auto")
            return self.available

        def ready_manual(self):
            self.calls.append("ready_manual")
            return self.available

        def build_batch_auto(self, batch_number):
            self.calls.append("build_auto")
            return ExampleBatch(f"auto:{batch_number}", ("auto",))

        def build_batch_manual(self, batch_number):
            self.calls.append("build_manual")
            return ExampleBatch(
                f"manual:{batch_number}",
                ("manual",),
                request=TrainingRequest(text="change", session="session", release_id="release"),
            )

        def acknowledge_auto(self):
            self.calls.append("acknowledge_auto")
            return frozenset({"auto-record"})

        def acknowledge_manual(self):
            self.calls.append("acknowledge_manual")
            return frozenset({"manual-record"})

    processor = MethodProcessor(ProcessorContext("s", training_mode=mode))
    first = processor.build_batch()
    assert first.values == (mode,)
    # New arrivals/readiness changes cannot replace the reserved batch.
    processor.available = False
    assert processor.ready()
    assert processor.build_batch() is first
    with pytest.raises(ValueError, match="unknown batch_id"):
        processor.acknowledge("wrong")
    assert processor.build_batch() is first
    assert processor.calls == [f"ready_{mode}", f"build_{mode}"]
    assert processor.acknowledge(first.batch_id) == frozenset({f"{mode}-record"})
    assert not processor.ready()
    processor.available = True
    second = processor.build_batch()
    assert second is not first
    assert second.batch_id == f"{mode}:2"
    assert processor.calls.count(f"build_{mode}") == 2


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


@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_switch_during_reserved_batch_keeps_original_acknowledgement(mode, dispatched):
    records, backend = RecordStore(), CaptureBackend(dispatched=dispatched)
    trainer = build(records, backend, mode=mode)
    try:
        records.append(inference("first") if mode == "auto" else instruction("first"))
        if dispatched:
            original = trainer.reserve_training_batch()
        else:
            result = trainer.run_once()
            original = trainer.pending_batch
        target = "manual" if mode == "auto" else "auto"
        processor = trainer.processor
        trainer.set_training_mode(target)
        assert trainer.processor is processor
        assert trainer.training_mode == target
        assert trainer.pending_batch is original
        if dispatched:
            assert trainer.reserve_training_batch() is original
            result = trainer.execute_reserved_step(0).result
        prepared = trainer.prepare_commit(result)
        assert prepared.consumed_ids == frozenset({"first"})
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        records.append(instruction("next") if target == "manual" else inference("next"))
        if dispatched:
            following = trainer.reserve_training_batch()
        else:
            assert trainer.run_once(1) is not None
            following = trainer.pending_batch
        assert (following.request is not None) == (target == "manual")
    finally:
        trainer.close()
        records.close()


def test_switch_preserves_incomplete_auto_batch_and_unread_manual_instructions():
    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, mode="auto", batch_size=2)
    try:
        records.append(inference("a"))
        assert trainer.run_once() is None
        trainer.set_training_mode("manual")
        records.append(instruction("change"))
        # Change back before the accepted instruction has been read.
        trainer.set_training_mode("auto")
        assert trainer.run_once() is None
        records.append(inference("b"))
        result = trainer.run_once()
        assert [item.source_agent_record_id for item in backend.batches[-1].samples] == ["a", "b"]
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        assert records.get("s", "change") is not None
        trainer.set_training_mode("manual")
        result = trainer.run_once(1)
        assert backend.batches[-1].request.id == "change"
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        trainer.set_training_mode("auto")
        assert trainer.run_once(2) is None
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("reads_requests", [False, True])
def test_http_training_mode_updates_only_existing_supported_processors(tmp_path, reads_requests):
    def propose(nodes, samples, models, *, requests=()):
        return None

    recipe = _recipe(tmp_path, propose if reads_requests else lambda n, s, m: None)
    dispatcher = _dispatcher(tmp_path, recipe)
    scenario = dispatcher.get_or_create_scenario("s")
    processor = scenario.trainer.processor

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            url = "/reef/scenarios/s/training-mode"
            response = await client.post(url, json={"training_mode": "manual"})
            assert response.status == (200 if reads_requests else 501)
            if reads_requests:
                assert await response.json() == {"scenario": "s", "training_mode": "manual"}
            assert scenario.trainer.training_mode == ("manual" if reads_requests else "auto")
            assert scenario.trainer.processor is processor
            for invalid in (
                {},
                {"training_mode": "bad"},
                {"training_mode": []},
                {"training_mode": "auto", "batch_size": 2},
            ):
                response = await client.post(url, json=invalid)
                assert response.status == 400
            response = await client.post("/reef/scenarios/missing/training-mode", json={"training_mode": "manual"})
            assert response.status == 404
            assert not dispatcher.has_scenario("missing")
            response = await client.get("/reef/scenarios/s/config")
            assert response.status == 404
            response = await client.post(url, json={"training_mode": "auto"})
            assert response.status == 200
            assert scenario.trainer.training_mode == "auto"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_http_mode_change_does_not_wait_for_running_proposer(tmp_path):
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        entered.set()
        assert release.wait(5)

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="manual"))

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await client.post(
                "/reef/train", headers={"x-reef-scenario": "s"}, json=instruction("one").payload
            )
            assert response.status == 200
            assert await asyncio.to_thread(entered.wait, 3)
            response = await asyncio.wait_for(
                client.post("/reef/scenarios/s/training-mode", json={"training_mode": "auto"}), timeout=2
            )
            assert response.status == 200
            assert not release.is_set()
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.trainer.training_mode == "auto"
            assert scenario.trainer.pending_batch.request.text == "one"
        finally:
            release.set()
            await client.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        dispatcher.close()


def test_mode_selection_resets_to_recipe_default_on_reload(tmp_path):
    def propose(nodes, samples, models, *, requests=()):
        return None

    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        scenario.set_training_mode("manual")
        assert scenario.trainer.training_mode == "manual"
        dispatcher._registry.reload("s")
        assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "auto"
    finally:
        dispatcher.close()
