"""Training instructions: manual runs them alone, hybrid runs them ahead of automatic batches, auto refuses them."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe, RecipeConfigError
from reef.runtime.base import TrainingRuntime
from reef.service.app import create_app
from reef.storage.factory import SQLiteScenarioStoreFactory
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.backend import PreparedStep
from reef.train.cordis_backend.processor import CordisProcessor, RecordDrivenTraceProcessor
from reef.train.processors.base import DataProcessor
from reef.train.trainer import Trainer
from reef.train.types import ProcessorContext, TraceBatch

from .runtime_stubs import StubTrainingRuntime
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


def report(receipt):
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload={"score": 0, "references": [receipt]},
        agent_record_id=f"report-{receipt}",
    )


def failure(records, receipt):
    """One failing exchange: the inference and its score 0 report."""
    records.append(inference(receipt))
    records.append(report(receipt))


def build(records, backend, processor=RecordDrivenTraceProcessor, mode="manual", batch_size=1):
    return Trainer.build(
        "s",
        records,
        processor_factory=lambda ctx: processor(ctx.with_config({"batch_size": batch_size})),
        training_backend=backend,
        training_mode=mode,
    )


def _wait(predicate, seconds=10.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _release_texts(dispatcher, scenario="s"):
    rows = dispatcher.get_or_create_scenario(scenario).releases()
    return [row.get("metrics", {}).get("training_request", {}).get("text") for row in rows]


@pytest.mark.parametrize("processor", [CordisProcessor, RecordDrivenTraceProcessor])
@pytest.mark.parametrize("batch_size", [1, 100])
def test_manual_waits_for_instruction_without_automatic_batch_gates(processor, batch_size):
    records, backend = SQLiteRecordStore(), CaptureBackend()
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
    records, backend = SQLiteRecordStore(), CaptureBackend()
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
    records, backend = SQLiteRecordStore(), CaptureBackend(dispatched=True)
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
    records, backend = SQLiteRecordStore(path), CaptureBackend()
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
    records = SQLiteRecordStore(path)
    recovered = build(records, backend)
    recovered.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    recovered.reingest(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    assert recovered.run_once() is not None
    assert [batch.request.text for batch in backend.batches] == ["committed", "pending"]
    recovered.close()
    records.close()


def test_manual_requires_an_explicit_recipe_assembler():
    records = SQLiteRecordStore()
    with pytest.raises(NotImplementedError, match="does not implement training_mode='manual'"):
        build(records, CaptureBackend(), DataProcessor)
    with pytest.raises(NotImplementedError, match="does not implement training_mode='hybrid'"):
        build(records, CaptureBackend(), DataProcessor, mode="hybrid")
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
    dispatcher.get_or_create_scenario("s")

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
        dispatcher.get_or_create_scenario("s")
        with pytest.raises(ValueError, match="training_mode='manual'"):
            dispatcher.accept_record(instruction("one"))
    finally:
        dispatcher.close()


def test_manual_is_a_native_contract_for_arbitrary_batch_schemas():
    class InstructionProcessor(DataProcessor):
        supported_training_modes = frozenset({"manual"})
        required_request_types = frozenset(RequestType)
        output_schema = ExampleBatch

        def make_training_batch(self, batch_number, request):
            return ExampleBatch(request.id, (request.text,))

    with pytest.raises(NotImplementedError, match="training_mode='auto'"):
        InstructionProcessor(ProcessorContext("s"))
    records, backend = SQLiteRecordStore(), CaptureBackend()
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
    with pytest.raises(NotImplementedError, match=r"AutoOnlyProcessor.*hybrid"):
        AutoOnlyProcessor(ProcessorContext("s", training_mode="hybrid"))
    with pytest.raises(ValueError, match="training_mode"):
        ProcessorContext("s", training_mode="invalid")


def test_unimplemented_manual_assembly_never_falls_back_to_auto():
    class IncompleteProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})

    processor = IncompleteProcessor(ProcessorContext("s", training_mode="manual"))
    processor.ingest(instruction("one"))
    with pytest.raises(NotImplementedError, match="instruction batch assembly"):
        processor.build_batch()


def test_status_reports_buffered_requests_when_a_processor_takes_instructions_in_hybrid_only():
    class HybridOnlyProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "hybrid"})
        required_request_types = frozenset(RequestType)

    processor = HybridOnlyProcessor(ProcessorContext("s", training_mode="hybrid"))
    processor.ingest(instruction("one"))
    assert processor.status() == {"buffered_requests": 1}
    processor.close()
    auto_only = DataProcessor(ProcessorContext("s"))
    auto_only.ingest(instruction("one"))
    assert auto_only.status() == {}
    auto_only.close()


@pytest.mark.parametrize("mode", ["auto", "manual", "hybrid"])
def test_processor_uses_shared_data_and_one_batch_assembly_hook(mode):
    class TrajectoryProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual", "hybrid"})
        required_request_types = frozenset({RequestType.INFERENCE, RequestType.TRAIN})
        output_schema = ExampleBatch

        def __init__(self, context):
            super().__init__(context)
            self.exchanges = []

        def ingest(self, item):
            super().ingest(item)
            if item.request_type is RequestType.INFERENCE:
                self.exchanges.append(item)

        def _ready_count(self):
            return len(self.exchanges)

        def ready(self):
            return self._pending is not None or (len(self.exchanges) >= 2 and super().ready())

        def make_training_batch(self, batch_number, request):
            return ExampleBatch("custom-batch", tuple(record.agent_record_id for record in self.exchanges))

        def _consume_pending(self):
            consumed = frozenset(record.agent_record_id for record in self.exchanges)
            self.exchanges.clear()
            return consumed

    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, TrajectoryProcessor, mode=mode, batch_size=2)
    try:
        records.append(inference("first"))
        assert trainer.run_once() is None
        if mode != "auto":
            records.append(instruction("use-the-trajectory"))
            assert trainer.run_once() is None
        records.append(inference("second"))
        result = trainer.run_once()
        assert backend.batches[0].values == ("first", "second")
        assert (backend.batches[0].request is None) == (mode == "auto")
        prepared = trainer.prepare_commit(result)
        expected = {"first", "second"} | ({"use-the-trajectory"} if mode != "auto" else set())
        assert prepared.consumed_ids == frozenset(expected)
        trainer.commit(prepared)
        assert trainer.run_once() is None
    finally:
        trainer.close()
        records.close()


def test_factory_cannot_silently_change_the_processor_mode():
    records = SQLiteRecordStore()
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
    records = SQLiteRecordStore()
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


@pytest.mark.parametrize("mode", ["manual", "hybrid"])
def test_harness_instruction_modes_require_explicit_requests_keyword(tmp_path, mode):
    recipe = replace(_recipe(tmp_path, lambda n, s, m, **kwargs: None), training_mode=mode)
    records = SQLiteRecordStore()
    with pytest.raises(RecipeConfigError, match="requests"):
        recipe.build("s", records)
    records.close()


@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_switch_during_reserved_batch_keeps_original_acknowledgement(mode, dispatched):
    records, backend = SQLiteRecordStore(), CaptureBackend(dispatched=dispatched)
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
    records, backend = SQLiteRecordStore(), CaptureBackend()
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
            url = "/reef/scenarios/s/update"
            for mode in ("manual", "hybrid"):
                response = await client.post(url, json={"training_mode": mode})
                assert response.status == (200 if reads_requests else 501)
                if reads_requests:
                    assert await response.json() == {"scenario": "s", "training_mode": mode}
                assert scenario.trainer.training_mode == (mode if reads_requests else "auto")
                assert scenario.trainer.processor is processor
                status = await (await client.get("/reef/status")).json()
                assert status["scenarios"]["s"]["training_mode"] == scenario.trainer.training_mode
            for invalid in (
                {},
                {"training_mode": "bad"},
                {"training_mode": "either"},
                {"training_mode": []},
                {"training_mode": "auto", "batch_size": 2},
            ):
                response = await client.post(url, json=invalid)
                assert response.status == 400
            response = await client.post("/reef/scenarios/missing/update", json={"training_mode": "manual"})
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
    dispatcher.get_or_create_scenario("s")

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
                client.post("/reef/scenarios/s/update", json={"training_mode": "auto"}), timeout=2
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


def test_mode_selection_survives_a_reload_and_resets_on_restart(tmp_path):
    def propose(nodes, samples, models, *, requests=()):
        return None

    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert dispatcher.set_training_mode("s", "manual") == {"scenario": "s", "training_mode": "manual"}
        assert scenario.trainer.training_mode == "manual"
        reloaded = dispatcher._registry.reload("s")
        assert reloaded is not scenario and reloaded.trainer.training_mode == "manual"
        assert dispatcher.set_training_mode("s", "auto")["training_mode"] == "auto"
        assert dispatcher._registry.reload("s").trainer.training_mode == "auto"
        dispatcher.set_training_mode("s", "manual")
        dispatcher.accept_record(instruction("one"))
        assert _wait(lambda: _committed_skip(dispatcher, "one") == "no proposal")
        assert dispatcher.get_or_create_scenario("s").scenario_step == 1
    finally:
        dispatcher.close()
    # A new process loads the committed state and starts from the recipe's configured mode.
    restarted = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    try:
        loaded = restarted.get_or_create_scenario("s")
        assert loaded.scenario_step == 1 and loaded.trainer.training_mode == "auto"
    finally:
        restarted.close()


def _committed_row(dispatcher, text):
    for row in dispatcher.get_or_create_scenario("s").releases():
        metrics = row.get("metrics") or {}
        if metrics.get("training_request", {}).get("text") == text:
            return metrics
    return None


def _committed_skip(dispatcher, text):
    row = _committed_row(dispatcher, text)
    return None if row is None else row.get("skipped")


def _automatic_traces(dispatcher):
    """The trace count of every committed step that ran without an instruction."""
    return [
        metrics["traces"]
        for row in dispatcher.get_or_create_scenario("s").releases()
        if (metrics := row.get("metrics")) and "training_request" not in metrics
    ]


def _raising_proposer(calls, *, poison="poison", error="poison proposer"):
    """A proposer that raises on the poison instruction and holds its first attempt open until released."""
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        text = requests[0]["text"] if requests else None
        calls.append(text)
        if text == poison or poison is None:
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError(error)
        return

    return propose, entered, release


def test_a_failed_step_keeps_the_selected_mode_and_the_next_instruction_runs(tmp_path):
    calls = []
    propose, entered, release = _raising_proposer(calls, poison=None, error="poison proposer")
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert scenario.store.durable
        dispatcher.set_training_mode("s", "manual")
        dispatcher.accept_record(instruction("one"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("two"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "one") == "instruction failed")
        assert _wait(lambda: _committed_skip(dispatcher, "two") == "instruction failed")
        # A failed instruction is not run again; the reload after each failure kept the selected mode.
        assert calls == ["one", "two"]
        assert _committed_row(dispatcher, "one")["error"] == "RuntimeError: poison proposer"
        current = dispatcher.get_or_create_scenario("s")
        assert current is not scenario and current.trainer.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
    finally:
        release.set()
        dispatcher.close()


def test_a_failed_instruction_is_skipped_with_its_error_and_the_queue_moves_on(tmp_path):
    calls = []
    propose, entered, release = _raising_proposer(calls)
    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="manual"))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("fine"))
        dispatcher.accept_record(instruction("fine again"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "fine again") == "no proposal")
        # The skip row consumed the failed instruction without another proposer call; the fresh ones ran after it.
        assert calls == ["poison", "fine", "fine again"]
        assert _committed_skip(dispatcher, "poison") == "instruction failed"
        assert _committed_row(dispatcher, "poison")["error"] == "RuntimeError: poison proposer"
        assert _committed_skip(dispatcher, "fine") == "no proposal"
        assert "error" not in _committed_row(dispatcher, "fine")
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.processor_status() == {"buffered_requests": 0}
        assert current.trainer.instruction_failures() == {}
        assert current.records.get("s", "poison") is None
        assert current.trainer.training_mode == "manual"
    finally:
        release.set()
        dispatcher.close()


def test_hybrid_skips_a_failed_instruction_before_the_next_and_keeps_the_failure_path(tmp_path):
    calls = []
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        text = requests[0]["text"] if requests else None
        calls.append((text, tuple(sample.source_agent_record_id for sample in samples)))
        if text == "poison":
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError("poison proposer")
        return

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="hybrid"))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("fine"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "fine") == "no proposal")
        # The wake after the failure consumed the failed instruction with its skip row, no second call, then ran the fresh one.
        assert calls == [("poison", ()), ("fine", ())]
        assert _committed_skip(dispatcher, "poison") == "instruction failed"
        assert _committed_row(dispatcher, "poison")["error"] == "RuntimeError: poison proposer"
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.training_mode == "hybrid"
        assert current.trainer.instruction_failures() == {}
        # With the queue empty, one failing exchange at batch_size 1 is an automatic step with no instruction.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        assert _wait(lambda: _automatic_traces(dispatcher) == [1])
        assert calls[-1] == (None, ("a",))
        # The next instruction runs as an instruction step again, with nothing held beside it.
        dispatcher.accept_record(instruction("after"))
        assert _wait(lambda: _committed_skip(dispatcher, "after") == "no proposal")
        assert calls[2:] == [(None, ("a",)), ("after", ())]
        assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "hybrid"
    finally:
        release.set()
        dispatcher.close()


def test_hybrid_skips_a_failed_instruction_alone_and_keeps_the_units_it_carried(tmp_path):
    calls = []
    entered, release, automatic, outage = Event(), Event(), Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        text = requests[0]["text"] if requests else None
        calls.append((text, tuple(sample.source_agent_record_id for sample in samples)))
        if text == "poison":
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError("poison proposer")
        # The automatic step holds so the skip row can be read, then fails so the unit stays stored for the restart.
        automatic.set()
        outage.wait(10)
        raise RuntimeError("proposer outage")

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="hybrid"))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        # The failing exchange arrives while the attempt holds, so the skip row's batch carries it as its sample.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        release.set()
        assert automatic.wait(5)
        # The skip row took no proposer call; the automatic step that followed read the unit the skip row gave back.
        assert calls == [("poison", ()), (None, ("a",))]
        assert _committed_skip(dispatcher, "poison") == "instruction failed"
        current = dispatcher.get_or_create_scenario("s")
        skip = next(
            record
            for record in current.store.history()
            if record.metrics is not None and "training_request" in record.metrics
        )
        # The skip row consumed the instruction alone: the unit it carried is held for the automatic step.
        assert skip.consumed_ids == frozenset({"poison"})
        assert skip.compacted_ids == frozenset({"poison"})
        assert current.records.get("s", "poison") is None
        assert current.records.get("s", "report-a") is not None
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.instruction_failures() == {}
        outage.set()
    finally:
        release.set()
        outage.set()
        dispatcher.close()

    seen = []

    def propose_again(nodes, samples, models, *, requests=()):
        seen.append((requests[0]["text"] if requests else None, tuple(s.source_agent_record_id for s in samples)))
        return

    restarted = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose_again), training_mode="hybrid"))
    try:
        loaded = restarted.get_or_create_scenario("s")
        assert loaded.scenario_step == 1
        assert loaded.records.get("s", "poison") is None
        # Recovery replays the unit as unconsumed and never the instruction the skip row named.
        restarted.set_training_mode("s", "hybrid")
        assert _wait(lambda: _automatic_traces(restarted) == [1])
        assert seen == [(None, ("a",))]
        assert _wait(lambda: loaded.records.get("s", "report-a") is None)
        assert loaded.trainer.pending_instructions() == 0
        assert restarted.get_or_create_scenario("s").scenario_step == 2
    finally:
        restarted.close()


def test_a_queue_of_failing_instructions_drains_without_another_record(tmp_path):
    calls = []
    propose, entered, release = _raising_proposer(calls, poison=None, error="proposer outage")
    recipe = replace(_recipe(tmp_path, propose), training_mode="manual")
    dispatcher = _dispatcher(tmp_path, recipe)
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("p1"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("p2"))
        release.set()
        # No further records are submitted, so the failures wake the worker until both are consumed.
        assert _wait(lambda: _committed_skip(dispatcher, "p1") == "instruction failed")
        assert _wait(lambda: _committed_skip(dispatcher, "p2") == "instruction failed")
        assert calls == ["p1", "p2"]
        assert _committed_row(dispatcher, "p2")["error"] == "RuntimeError: proposer outage"
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pending_instructions() == 0
        assert dispatcher.accept_record(instruction("p3")).agent_record_id == "p3"
        assert _wait(lambda: _committed_skip(dispatcher, "p3") == "instruction failed")
    finally:
        release.set()
        dispatcher.close()


def test_a_logless_scenario_keeps_the_failed_batch_and_skips_it_on_its_next_wake(tmp_path):
    calls = []
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        text = requests[0]["text"]
        calls.append(text)
        if text == "poison":
            raise RuntimeError("poison proposer")
        # The step after the skip row holds, so the skip row is observable as the last commit.
        entered.set()
        release.wait(10)
        return

    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        replace(_recipe(tmp_path, propose), training_mode="manual"),
        factory,
        scenario_store_factory=SQLiteScenarioStoreFactory(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert not scenario.store.durable
        dispatcher.accept_record(instruction("poison"))
        dispatcher.accept_record(instruction("fine"))
        assert entered.wait(5)
        # No reload without a log: the same scenario kept the batch and committed its skip row on the next wake.
        assert dispatcher.get_or_create_scenario("s") is scenario
        skipped = _last_committed(scenario)
        assert skipped["skipped"] == "instruction failed"
        assert skipped["training_request"]["id"] == "poison"
        assert skipped["error"] == "RuntimeError: poison proposer"
        assert calls == ["poison", "fine"]
        release.set()
        assert _wait(lambda: scenario.scenario_step == 2)
        assert _last_committed(scenario).get("skipped") == "no proposal"
        assert scenario.trainer.pending_instructions() == 0
        assert scenario.trainer.processor_status() == {"buffered_requests": 0}
        assert scenario.trainer.instruction_failures() == {}
        assert scenario.records.get("s", "poison") is None
    finally:
        release.set()
        dispatcher.close()


def _last_committed(scenario):
    committed = scenario.commit_status.get("last_committed_step")
    return {} if committed is None else committed.get("metrics") or {}


def test_the_dispatched_training_thread_skips_a_failed_instruction(tmp_path):
    calls = []

    class RaisingBackend(CaptureBackend):
        def prepare_step(self, batch, state, scenario_step):
            calls.append(None if batch.request is None else batch.request.text)
            if batch.request is not None:
                raise RuntimeError("dispatched step failed")
            return super().prepare_step(batch, state, scenario_step)

    class DispatchedRecipe(Recipe):
        def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
            return Trainer.build(
                scenario,
                records,
                processor_factory=lambda ctx: RecordDrivenTraceProcessor(ctx.with_config({"batch_size": 1})),
                training_backend=RaisingBackend(dispatched=True),
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
                training_mode=self.training_mode,
            )

    dispatcher = _dispatcher(tmp_path, DispatchedRecipe(runtime=StubTrainingRuntime(), training_mode="manual"))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert isinstance(scenario.runtime, TrainingRuntime)
        dispatcher.accept_record(instruction("one"))
        assert _wait(lambda: _committed_skip(dispatcher, "one") == "instruction failed")
        assert calls == ["one"]
        assert _committed_row(dispatcher, "one")["error"] == "RuntimeError: dispatched step failed"
        current = dispatcher.get_or_create_scenario("s")
        assert current is not scenario and current.trainer.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
    finally:
        dispatcher.close()


def test_an_instruction_runs_past_the_step_budget_and_the_failure_streak(tmp_path):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(requests[0]["text"] if requests else None)
        return

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", max_steps=1)
    records = SQLiteRecordStore()
    trainer = recipe.build("s", records)
    try:
        records.append(instruction("one"))
        records.append(instruction("two"))
        rows = []
        for step in range(2):
            result = trainer.run_once(step)
            assert result is not None
            prepared = trainer.prepare_commit(result)
            trainer.commit(prepared)
            trainer.apply_compaction(prepared.compacted_ids)
            rows.append(prepared.metrics)
        assert seen == ["one", "two"]
        assert [row["training_request"]["text"] for row in rows] == ["one", "two"]
        assert [row["skipped"] for row in rows] == ["no proposal", "no proposal"]
        assert [row["steps"] for row in rows] == [1, 2]
        assert trainer.run_once(2) is None

        backend = trainer.training_backend
        state = {**backend.initial_state(), "steps": 5}
        automatic = backend.prepare_step(TraceBatch("auto", ()), state, 5)
        assert automatic.metrics["skipped"] == "step budget of 1 exhausted"
        assert seen == ["one", "two"]
    finally:
        trainer.close()
        records.close()

    streak = replace(_recipe(tmp_path, propose), training_mode="manual", max_failure_streak=1)
    records = SQLiteRecordStore()
    trainer = streak.build("s", records)
    try:
        backend = trainer.training_backend
        state = {**backend.initial_state(), "failure_streak": 1}
        automatic = backend.prepare_step(TraceBatch("auto", ()), state, 0)
        assert automatic.metrics["skipped"] == "failure streak breaker open after 1 consecutive rejections"
        request = TrainingRequest("Follow the request", "session", "r", "request-id")
        asked = backend.prepare_step(TraceBatch("request-id", (), request=request), state, 0)
        assert asked.metrics["skipped"] == "no proposal" and seen[-1] == "Follow the request"
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("processor", [CordisProcessor, RecordDrivenTraceProcessor])
def test_manual_traffic_is_available_to_auto_without_reingestion(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="manual", batch_size=2)
    try:
        for receipt in ("a", "b"):
            records.append(inference(receipt))
            if processor is CordisProcessor:
                records.append(
                    AgentRecord.create(
                        scenario="s",
                        request_type=RequestType.REPORT,
                        payload={"score": 0, "references": [receipt]},
                    )
                )
        assert trainer.run_once() is None
        original = trainer.processor
        offset = trainer.data_offset
        trainer.set_training_mode("auto")
        assert trainer.run_once() is not None
        assert trainer.processor is original
        assert trainer.data_offset == offset
        assert [sample.source_agent_record_id for sample in backend.batches[-1].samples] == ["a", "b"]
    finally:
        trainer.close()
        records.close()


# -- training_mode: hybrid --------------------------------------------------


def test_hybrid_is_a_recipe_and_processor_mode_and_a_fourth_value_is_refused(tmp_path):
    def propose(nodes, samples, models, *, requests=()):
        return None

    recipe = replace(_recipe(tmp_path, propose), training_mode="hybrid")
    records = SQLiteRecordStore()
    try:
        trainer = recipe.build("s", records)
        assert trainer.training_mode == "hybrid"
        assert trainer.processor.training_mode == "hybrid"
        assert trainer.processor.status() == {"buffered_requests": 0}
        trainer.close()
        with pytest.raises(ValueError, match="training_mode"):
            replace(recipe, training_mode="either")
        with pytest.raises(ValueError, match="training_mode"):
            ProcessorContext("s", training_mode="either")
        with pytest.raises(ValueError, match="training_mode"):
            CordisProcessor(ProcessorContext("s")).set_training_mode("either")
    finally:
        records.close()


@pytest.mark.parametrize("processor", [CordisProcessor, RecordDrivenTraceProcessor])
def test_hybrid_runs_a_queued_instruction_alone_when_no_units_are_held(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="hybrid", batch_size=2)
    try:
        assert trainer.run_once() is None
        records.append(instruction("alone"))
        result = trainer.run_once()
        assert result is not None
        batch = backend.batches[0]
        assert batch.batch_id == "s:instruction:alone"
        assert batch.request.text == "alone"
        assert batch.samples == ()
        prepared = trainer.prepare_commit(result)
        assert prepared.consumed_ids == frozenset({"alone"})
        assert prepared.metrics["training_request"]["id"] == "alone"
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        assert records.get("s", "alone") is None
        assert trainer.run_once() is None
        assert trainer.training_mode == "hybrid"
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("processor", [CordisProcessor, RecordDrivenTraceProcessor])
def test_hybrid_runs_a_queued_instruction_with_the_held_units_as_samples(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="hybrid", batch_size=2)
    try:
        failure(records, "a")
        assert trainer.run_once() is None
        records.append(instruction("with-context"))
        result = trainer.run_once()
        batch = backend.batches[0]
        assert batch.request.id == "with-context"
        assert [sample.source_agent_record_id for sample in batch.samples] == ["a"]
        prepared = trainer.prepare_commit(result)
        assert {"a", "with-context"} <= prepared.consumed_ids
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        assert records.get("s", "a") is None
        records.append(instruction("after"))
        assert trainer.run_once() is not None
        assert backend.batches[1].request.id == "after"
        assert backend.batches[1].samples == ()
    finally:
        trainer.close()
        records.close()


def test_hybrid_runs_two_queued_instructions_oldest_first_one_per_step():
    processor = CordisProcessor(ProcessorContext("s", {"batch_size": 1}, training_mode="hybrid"))
    processor.ingest(instruction("first"))
    processor.ingest(instruction("second"))
    assert processor.status() == {"buffered_requests": 2}
    assert processor.ready()
    batch = processor.build_batch()
    assert batch.batch_id == "s:instruction:first"
    assert batch.request.id == "first"
    assert processor.build_batch() is batch
    assert processor.acknowledge(batch.batch_id) == frozenset({"first"})
    assert processor.build_batch().request.id == "second"
    assert processor.acknowledge("s:instruction:second") == frozenset({"second"})
    assert not processor.ready()
    processor.close()


def test_hybrid_batches_as_auto_does_without_an_instruction():
    processor = CordisProcessor(ProcessorContext("s", {"batch_size": 1, "max_score": 0.0}, training_mode="hybrid"))
    processor.ingest(inference("a"))
    processor.ingest(report("a"))
    batch = processor.build_batch()
    assert batch.batch_id == "s:harness_evolve:1"
    assert batch.request is None
    assert [sample.source_agent_record_id for sample in batch.samples] == ["a"]
    assert processor.acknowledge(batch.batch_id) == frozenset({"a", "report-a"})
    processor.close()


def test_hybrid_alternates_the_instruction_path_and_the_failure_path_without_a_mode_change():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, CordisProcessor, mode="hybrid", batch_size=2)

    def step():
        result = trainer.run_once()
        if result is None:
            return None
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        batch = backend.batches[-1]
        request = None if batch.request is None else batch.request.id
        return request, [sample.source_agent_record_id for sample in batch.samples]

    try:
        failure(records, "a")
        failure(records, "b")
        assert step() == (None, ["a", "b"])
        failure(records, "c")
        assert step() is None
        # An instruction goes first, and the failure held below the batch size rides beside it.
        records.append(instruction("one"))
        assert step() == ("one", ["c"])
        failure(records, "d")
        failure(records, "e")
        assert step() == (None, ["d", "e"])
        records.append(instruction("two"))
        assert step() == ("two", [])
        failure(records, "f")
        records.append(instruction("three"))
        failure(records, "g")
        assert step() == ("three", ["f"])
        assert step() is None
        failure(records, "h")
        assert step() == (None, ["g", "h"])
        assert trainer.training_mode == "hybrid"
    finally:
        trainer.close()
        records.close()


def test_switching_hybrid_to_auto_holds_the_unread_instruction_for_a_mode_that_takes_it():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, CordisProcessor, mode="hybrid", batch_size=2)
    try:
        records.append(instruction("later"))
        trainer.set_training_mode("auto")
        assert trainer.run_once() is None
        assert trainer.processor.status() == {"buffered_requests": 1}
        failure(records, "a")
        failure(records, "b")
        result = trainer.run_once()
        assert backend.batches[-1].request is None
        assert [sample.source_agent_record_id for sample in backend.batches[-1].samples] == ["a", "b"]
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        trainer.apply_compaction(prepared.compacted_ids)
        assert records.get("s", "later") is not None
        trainer.set_training_mode("hybrid")
        result = trainer.run_once(1)
        assert backend.batches[-1].request.id == "later"
        assert backend.batches[-1].samples == ()
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        assert trainer.run_once(2) is None
    finally:
        trainer.close()
        records.close()


def test_hybrid_runs_an_instruction_from_the_route_without_an_update_call(tmp_path):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(
            (
                tuple(request["text"] for request in requests),
                tuple(sample.source_agent_record_id for sample in samples),
            )
        )

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="hybrid"))

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "hybrid"
            body = {
                "agent_record_id": "ask-1",
                "text": "Prefer tests first",
                "session": "session-1",
                "release_id": "release-1",
            }
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: "Prefer tests first" in _release_texts(dispatcher))
            assert seen == [(("Prefer tests first",), ())]
            # The failure path stays on: a failing exchange batches by itself with no instruction queued.
            dispatcher.accept_record(inference("a"))
            dispatcher.accept_record(report("a"))
            assert await asyncio.to_thread(_wait, lambda: len(seen) == 2)
            assert seen[1] == ((), ("a",))
            assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "hybrid"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("target", ["auto", "hybrid"])
def test_the_proposal_route_refuses_in_manual_mode_and_admits_again_in_a_batching_mode(tmp_path, target):
    def propose(nodes, samples, models, *, requests=()):
        return None

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="manual"))
    proposal = {
        "mutations": [{"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}],
        "reason": "because",
        "session": "3f1c2a9d0b7e",
        "release_id": "rel-0",
    }
    inbox = tmp_path / "inbox" / "s"

    async def run():
        assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "manual"
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": "s"}
            response = await client.post("/reef/harness/proposals", headers=headers, json=proposal)
            assert response.status == 200
            answer = await response.json()
            assert answer["admitted"] is False
            assert answer["reason"] == "manual mode takes instructions only"
            assert not inbox.exists() or not list(inbox.glob("*.json"))
            response = await client.post("/reef/scenarios/s/update", json={"training_mode": target})
            assert response.status == 200
            response = await client.post("/reef/harness/proposals", headers=headers, json=proposal)
            answer = await response.json()
            assert answer["admitted"] is True and answer["reason"] is None
            assert [path.name for path in inbox.glob("*.json")] == [f"{answer['proposal_id']}.json"]
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_manual_mode_caps_held_units_at_four_batches_and_the_batching_modes_hold_them_all(caplog):
    def fill(mode):
        processor = CordisProcessor(ProcessorContext("s", {"batch_size": 1, "max_score": 0.0}, training_mode=mode))
        for i in range(20):
            processor.ingest(inference(f"inf-{i}"))
            processor.ingest(
                AgentRecord.create(
                    scenario="s",
                    request_type=RequestType.REPORT,
                    payload={"score": 0, "references": [f"inf-{i}"]},
                    agent_record_id=f"rep-{i}",
                )
            )
        return processor

    with caplog.at_level(logging.WARNING, logger="reef.train.processors.reported"):
        manual = fill("manual")
    assert manual._ready_count() == 4
    assert not manual.ready()
    retention = manual.retention_decision()
    shed = {f"inf-{i}" for i in range(16)} | {f"rep-{i}" for i in range(16)}
    assert shed <= retention.releasable_agent_record_ids
    kept = {f"inf-{i}" for i in range(16, 20)} | {f"rep-{i}" for i in range(16, 20)}
    assert kept <= retention.protected_agent_record_ids
    assert manual.never_reasons == {"more than 4 units held in manual mode": 16}
    assert sum("released report rep-0" in record.message for record in caplog.records) == 1
    assert len(caplog.records) == 1
    manual.ingest(instruction("do-it"))
    batch = manual.build_batch()
    assert batch.samples == ()
    assert manual.acknowledge(batch.batch_id) == frozenset({"do-it"})
    assert manual._ready_count() == 4
    manual.close()

    for mode in ("auto", "hybrid"):
        uncapped = fill(mode)
        assert uncapped._ready_count() == 20
        assert uncapped.never_reasons == {}
        # The switch to manual trims the pile at once; the batch already handed out keeps its unit.
        reserved = uncapped.build_batch()
        uncapped.set_training_mode("manual")
        assert uncapped._ready_count() == 4
        assert uncapped.never_reasons == {"more than 4 units held in manual mode": 16}
        assert "inf-0" in uncapped.retention_decision().protected_agent_record_ids
        assert uncapped.acknowledge(reserved.batch_id) == frozenset({"inf-0", "rep-0"})
        assert uncapped._ready_count() == 3
        uncapped.close()


def test_hybrid_promotes_the_failures_an_instruction_step_carries(tmp_path):
    calls = []

    def propose(nodes, samples, models, *, requests=()):
        calls.append(
            (
                tuple(request["text"] for request in requests),
                tuple(sample.source_agent_record_id for sample in samples),
            )
        )

    recipe = replace(_recipe(tmp_path, propose), training_mode="hybrid", promote_failures=True, batch_size=2)
    dispatcher = _dispatcher(tmp_path, recipe)

    def gate_rows():
        rows = dispatcher.get_or_create_scenario("s").releases()
        return [
            (metrics["traces"], metrics["promoted_tasks"], metrics["gate_tasks"], "training_request" in metrics)
            for row in rows
            if (metrics := row.get("metrics"))
        ]

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            # One failure held below the batch size; the instruction step takes it and promotes it.
            dispatcher.accept_record(inference("fails-1"))
            dispatcher.accept_record(report("fails-1"))
            body = {"agent_record_id": "ask-1", "text": "do it", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(gate_rows()) == 1)
            assert calls == [(("do it",), ("fails-1",))]
            assert gate_rows() == [(1, 1, 2, True)]
            # The automatic step after grows the gate on top of the promoted one.
            for receipt in ("fails-2", "fails-3"):
                dispatcher.accept_record(inference(receipt))
                dispatcher.accept_record(report(receipt))
            assert await asyncio.to_thread(_wait, lambda: len(gate_rows()) == 2)
            # Newest first: the automatic step's row, then the instruction step's.
            assert gate_rows() == [(2, 3, 4, False), (1, 1, 2, True)]
            assert calls[1] == ((), ("fails-2", "fails-3"))
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
