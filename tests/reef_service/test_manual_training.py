"""Native manual scheduling: instructions authorize one durable step without a data batch."""

from __future__ import annotations

import asyncio
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
from reef.records import RecordStore
from reef.runtime.base import TrainingRuntime
from reef.service.app import create_app
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


def test_unimplemented_manual_assembly_never_falls_back_to_auto():
    class IncompleteProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})

    processor = IncompleteProcessor(ProcessorContext("s", training_mode="manual"))
    processor.ingest(instruction("one"))
    with pytest.raises(NotImplementedError, match="manual batch assembly"):
        processor.build_batch()


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_processor_uses_shared_data_and_one_batch_assembly_hook(mode):
    class TrajectoryProcessor(DataProcessor):
        supported_training_modes = frozenset({"auto", "manual"})
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

    records, backend = RecordStore(), CaptureBackend()
    trainer = build(records, backend, TrajectoryProcessor, mode=mode, batch_size=2)
    try:
        records.append(inference("first"))
        assert trainer.run_once() is None
        if mode == "manual":
            records.append(instruction("use-the-trajectory"))
            assert trainer.run_once() is None
        records.append(inference("second"))
        result = trainer.run_once()
        assert backend.batches[0].values == ("first", "second")
        assert (backend.batches[0].request is None) == (mode == "auto")
        prepared = trainer.prepare_commit(result)
        expected = {"first", "second"} | ({"use-the-trajectory"} if mode == "manual" else set())
        assert prepared.consumed_ids == frozenset(expected)
        trainer.commit(prepared)
        assert trainer.run_once() is None
    finally:
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
            url = "/reef/scenarios/s/update"
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


def _wait(predicate, seconds=10.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _committed_row(dispatcher, text):
    for row in dispatcher.get_or_create_scenario("s").releases():
        metrics = row.get("metrics") or {}
        if metrics.get("training_request", {}).get("text") == text:
            return metrics
    return None


def _committed_skip(dispatcher, text):
    row = _committed_row(dispatcher, text)
    return None if row is None else row.get("skipped")


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
        assert scenario.commit_log is not None
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


def test_a_full_queue_of_failing_instructions_drains_without_another_record(tmp_path):
    calls = []
    propose, entered, release = _raising_proposer(calls, poison=None, error="proposer outage")
    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", max_pending_requests=2)
    dispatcher = _dispatcher(tmp_path, recipe)
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("p1"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("p2"))
        with pytest.raises(ValueError, match="requests full"):
            dispatcher.accept_record(instruction("p3"))
        release.set()
        # Nothing else is admitted, so the failures themselves wake the worker until both are consumed.
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
    dispatcher = Dispatcher(replace(_recipe(tmp_path, propose), training_mode="manual"), factory)
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert scenario.commit_log is None
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
                max_pending_requests=self.max_pending_requests,
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
    records = RecordStore()
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
    records = RecordStore()
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
    records, backend = RecordStore(), CaptureBackend()
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
