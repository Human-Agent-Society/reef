"""Training instructions: manual runs them alone, hybrid runs them ahead of automatic batches, auto refuses them."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.harness_service import harness_dispatcher
from tests.support.harness_steps import COMPONENT, UNCHANGED, TreeReleases, changed_rules, harness_recipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent
from tests.train.test_harness_evolve_processor import harness_stages, records_pipeline, reports_pipeline
from tests.train.test_reef_trainer_contracts import ExampleBackend, ExampleJobData

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.harness.tree import TreePatch
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2record import InferenceRecord2Record
from reef.record2sample.traces import InferenceSamples
from reef.sample2job import Sample2Job
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import ComponentReleases, PreparedStep
from reef.train.harness_backend.state import HarnessState
from reef.train.jobs import FixedJobPlan, JobLoop, JobRequest
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.types import JobData, Sample


class CaptureBackend(ExampleBackend):
    def __init__(self):
        super().__init__("s", [])
        self.batches = []

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


def build(records, backend, processor=records_pipeline, mode="manual", job_samples=1):
    return Trainer.build(
        "s",
        records,
        pipeline_factory=processor,
        candidate_backend=backend,
        training_mode=mode,
        default_job_samples=job_samples,
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


def recipe_in(mode, tmp_path, method, **fields) -> HarnessEvolutionRecipe:
    """A pi harness recipe over ``method`` in training mode ``mode``; ``fields`` go to ``harness_recipe``."""
    return replace(harness_recipe(tmp_path, method, **fields), training_mode=mode)


def _trainer(recipe: HarnessEvolutionRecipe, records) -> Trainer:
    """The recipe's trainer over an in-memory commit history whose base is the recipe's seed."""
    return recipe.build("s", records, releases=ComponentReleases(TreeReleases(recipe.seed), COMPONENT))


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
@pytest.mark.parametrize("batch_size", [1, 100])
def test_manual_waits_for_instruction_without_automatic_batch_gates(processor, batch_size):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, job_samples=batch_size)
    for receipt in ("other-session", "turn-1", "turn-2"):
        records.append(inference(receipt))
    records.append(
        AgentRecord.create(
            scenario="s", request_type=RequestType.REPORT, payload={"score": 0, "references": ["other-session"]}
        )
    )
    assert trainer.run_once() is None
    assert not trainer.job_data_ready()
    records.append(instruction("request-1"))
    result = trainer.run_once()
    assert result is not None
    batch = backend.batches[0]
    assert batch.request.text == "request-1"
    assert batch.samples == ()
    prepared = trainer.prepare_commit(result)
    assert "request-1" in prepared.consumed_ids
    assert not {"other-session", "turn-1", "turn-2"} & prepared.consumed_ids
    assert prepared.metrics["training_request"]["text"] == "request-1"
    trainer.commit(prepared)
    assert records.get("s", "turn-1") is not None
    assert trainer.run_once() is None
    assert not records.append_result(instruction("request-1")).inserted
    assert trainer.run_once() is None
    trainer.close()
    records.close()


def test_auto_keeps_recipe_batching():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, mode="auto", job_samples=2)
    records.append(inference("a"))
    assert trainer.run_once() is None
    records.append(inference("b"))
    assert trainer.run_once() is not None
    assert [source_record_id(sample) for sample in backend.batches[0].samples] == ["a", "b"]
    assert backend.batches[0].request is None
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
    # Simulate a crash after the commit history landed but before compaction.
    records.append(instruction("pending"))
    first.close()
    records.close()
    records = SQLiteRecordStore(path)
    recovered = build(records, backend)
    recovered.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    recovered.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    assert recovered.run_once() is not None
    assert [batch.request.text for batch in backend.batches] == ["committed", "pending"]
    recovered.close()
    records.close()


def test_manual_requires_an_explicit_recipe_assembler():
    def default_pipeline(context):
        return TrainingPipeline(context)

    records = SQLiteRecordStore()
    with pytest.raises(NotImplementedError, match="does not implement training_mode='manual'"):
        build(records, CaptureBackend(), default_pipeline)
    with pytest.raises(NotImplementedError, match="does not implement training_mode='hybrid'"):
        build(records, CaptureBackend(), default_pipeline, mode="hybrid")
    with pytest.raises(ValueError, match="training_mode"):
        build(records, CaptureBackend(), mode="typo")
    records.close()


def test_train_route_needs_no_inference_and_retries_do_not_train_twice(tmp_path):
    called = Event()
    seen = []

    def method(workspace, services):
        seen.append((workspace.request(), workspace.traces()))
        called.set()

    recipe = replace(recipe_in("manual", tmp_path, method), pipeline_stages=harness_stages())
    dispatcher = harness_dispatcher(tmp_path, recipe)
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
            assert seen[0][0].text == "Prefer tests first"
            assert seen[0][1] == ()
            assert seen[0][0].id == "request-1"
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
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda workspace, services: None))
    try:
        dispatcher.get_or_create_scenario("s")
        with pytest.raises(ValueError, match="training_mode='manual'"):
            dispatcher.accept_record(instruction("one"))
    finally:
        dispatcher.close()


def test_manual_is_a_native_contract_for_arbitrary_batch_schemas():
    class InstructionJobData(Sample2Job):
        supported_training_modes = frozenset({"manual"})
        required_request_types = frozenset(RequestType)
        output_schema = ExampleJobData

        def make_job_data(self, samples, data_id):
            return ExampleJobData(data_id, values=tuple(source_record_id(task) for task in samples))

    def instruction_pipeline(context):
        return TrainingPipeline(context, sample2job=InstructionJobData(context, None, None))

    with pytest.raises(NotImplementedError, match="training_mode='auto'"):
        instruction_pipeline(PipelineContext("s"))
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, instruction_pipeline)
    records.append(instruction("change"))
    assert trainer.run_once() is not None
    assert isinstance(backend.batches[0], ExampleJobData)
    # The pipeline attaches the instruction to the recipe's own batch type; a manual batch has no tasks.
    assert backend.batches[0].data_id == "s:instruction:change"
    assert backend.batches[0].request.text == "change"
    assert backend.batches[0].values == ()
    trainer.close()
    records.close()


def test_pipeline_context_keeps_mode_when_recipe_takes_instructions():
    context = PipelineContext("s", training_mode="manual")
    processor = reports_pipeline(context)
    assert processor.training_mode == "manual"
    assert RequestType.TRAIN in processor.required_request_types
    processor.close()


def test_missing_mode_implementation_fails_at_processor_initialization():
    class AutoOnlyJobData(Sample2Job):
        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    for mode in ("manual", "hybrid"):
        context = PipelineContext("s", training_mode=mode)
        with pytest.raises(NotImplementedError, match=rf"AutoOnlyJobData.*{mode}"):
            TrainingPipeline(context, sample2job=AutoOnlyJobData(context, None, None))
    with pytest.raises(ValueError, match="training_mode"):
        PipelineContext("s", training_mode="invalid")


def test_manual_instruction_batch_never_falls_back_to_auto():
    class TraceOnlyJobData(Sample2Job):
        supported_training_modes = frozenset({"auto", "manual"})

        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="manual")
    pipeline = TrainingPipeline(
        context,
        record2record=InferenceRecord2Record(),
        record2sample=InferenceSamples(),
        sample2job=TraceOnlyJobData(context, None, None),
    )
    pipeline.push(inference("held"))
    assert not pipeline.ready(1)
    pipeline.push(instruction("one"))
    batch = pipeline.build_job_data(1)
    # The instruction runs alone: the held set an automatic batch would take stays held.
    assert batch.request.id == "one"
    assert batch.samples == ()
    assert pipeline.acknowledge(batch.data_id) == frozenset({"one"})
    assert len(pipeline.record2record.pool) == 1
    pipeline.close()


def test_status_shows_how_far_a_partly_filled_batch_still_has_to_go():
    """A batch that will never fill has to look different from one that is still filling.

    An on-policy producer waits on the release its batch makes. If the record stream picks up rows the
    producer did not send, the batch boundary shifts and the scenario settles below the batch size with the
    producer waiting for a release that cannot come. Nothing else in the status distinguishes that from
    training that is merely slow.
    """

    class CountedJobData(Sample2Job):
        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="auto")
    pipeline = TrainingPipeline(
        context,
        record2record=InferenceRecord2Record(),
        record2sample=InferenceSamples(),
        sample2job=CountedJobData(context, 6, None),
    )
    assert pipeline.status() == {"ready_samples": 0, "buffer_size": 6}
    for index in range(3):
        pipeline.push(inference(f"inf-{index}"))
    assert pipeline.status() == {"ready_samples": 3, "buffer_size": 6}
    assert not pipeline.ready(4)
    pipeline.push(inference("inf-3"))
    assert pipeline.status()["ready_samples"] == 4
    assert pipeline.ready(4)
    pipeline.close()


def test_status_reports_buffered_requests_when_a_processor_takes_instructions_in_hybrid_only():
    class HybridOnlyJobData(Sample2Job):
        supported_training_modes = frozenset({"auto", "hybrid"})
        required_request_types = frozenset(RequestType)

        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="hybrid")
    pipeline = TrainingPipeline(context, sample2job=HybridOnlyJobData(context, None, None))
    pipeline.push(instruction("one"))
    assert pipeline.status()["buffered_requests"] == 1
    pipeline.close()
    auto_only = TrainingPipeline(PipelineContext("s"))
    auto_only.push(instruction("one"))
    assert "buffered_requests" not in auto_only.status()
    auto_only.close()


@pytest.mark.parametrize("mode", ["auto", "manual", "hybrid"])
def test_processor_uses_shared_data_and_one_batch_assembly_hook(mode):
    class TrajectoryJobData(Sample2Job):
        supported_training_modes = frozenset({"auto", "manual", "hybrid"})
        required_request_types = frozenset({RequestType.INFERENCE, RequestType.TRAIN})
        output_schema = ExampleJobData

        def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> ExampleJobData:
            return ExampleJobData(data_id, values=tuple(source_record_id(task) for task in samples))

    def trajectory_pipeline(context):
        return TrainingPipeline(
            context,
            record2record=InferenceRecord2Record(),
            record2sample=InferenceSamples(),
            sample2job=TrajectoryJobData(context, None, None),
        )

    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, trajectory_pipeline, mode=mode, job_samples=2)
    try:
        records.append(inference("first"))
        assert trainer.run_once() is None
        if mode == "auto":
            records.append(inference("second"))
        else:
            records.append(instruction("use-the-trajectory"))
        result = trainer.run_once()
        # The same hook shapes every mode's batch: manual runs the instruction alone, hybrid with the held sets.
        expected_values = {"auto": ("first", "second"), "manual": (), "hybrid": ("first",)}[mode]
        assert isinstance(backend.batches[0], ExampleJobData)
        assert backend.batches[0].values == expected_values
        assert (backend.batches[0].request is None) == (mode == "auto")
        # The pipeline assigns the ID of each batch, and the hook uses it.
        expected_id = "s:data:1" if mode == "auto" else "s:instruction:use-the-trajectory"
        assert backend.batches[0].data_id == expected_id
        prepared = trainer.prepare_commit(result)
        expected = set(expected_values) | ({"use-the-trajectory"} if mode != "auto" else set())
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
            pipeline_factory=lambda ctx: TrainingPipeline(replace(ctx, training_mode="auto")),
            candidate_backend=CaptureBackend(),
            training_mode="manual",
            default_job_samples=1,
        )
    records.close()


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_switch_during_reserved_batch_keeps_original_acknowledgement(mode):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, mode=mode)
    try:
        records.append(inference("first") if mode == "auto" else instruction("first"))
        result = trainer.run_once()
        original = trainer.pending_job_data
        target = "manual" if mode == "auto" else "auto"
        processor = trainer.pipeline
        trainer.set_training_mode(target)
        assert trainer.pipeline is processor
        assert trainer.pipeline.training_mode == target
        assert trainer.pending_job_data is original
        prepared = trainer.prepare_commit(result)
        assert prepared.consumed_ids == frozenset({"first"})
        trainer.commit(prepared)
        records.append(instruction("next") if target == "manual" else inference("next"))
        assert trainer.run_once(1) is not None
        following = trainer.pending_job_data
        assert (following.request is not None) == (target == "manual")
    finally:
        trainer.close()
        records.close()


def test_switch_preserves_incomplete_auto_batch_and_unread_manual_instructions():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, mode="auto", job_samples=2)
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
        assert [source_record_id(item) for item in backend.batches[-1].samples] == ["a", "b"]
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        assert records.get("s", "change") is not None
        trainer.set_training_mode("manual")
        result = trainer.run_once(1)
        assert backend.batches[-1].request.id == "change"
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        trainer.set_training_mode("auto")
        assert trainer.run_once(2) is None
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("instructions", [False, True], ids=["record-only", "harness"])
def test_http_training_mode_updates_only_existing_supported_processors(tmp_path, instructions):
    # A record-only recipe makes no job data, so it trains in auto alone; a harness recipe takes instructions.
    recipe = harness_recipe(tmp_path, lambda workspace, services: None) if instructions else Recipe()
    dispatcher = harness_dispatcher(tmp_path, recipe)
    scenario = dispatcher.get_or_create_scenario("s")
    processor = scenario.trainer.pipeline

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            url = "/reef/scenarios/s/update"
            for mode in ("manual", "hybrid"):
                response = await client.post(url, json={"training_mode": mode})
                assert response.status == (200 if instructions else 501)
                assert scenario.trainer.pipeline.training_mode == (mode if instructions else "auto")
                assert scenario.trainer.pipeline is processor
                if instructions:
                    # The status lists the scenarios that train; a record-only one does not.
                    assert await response.json() == {"scenario": "s", "training_mode": mode}
                    status = await (await client.get("/reef/status")).json()
                    assert status["scenarios"]["s"]["training_mode"] == mode
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
            assert scenario.trainer.pipeline.training_mode == "auto"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_http_mode_change_does_not_wait_for_running_method(tmp_path):
    entered, release = Event(), Event()

    def method(workspace, services):
        entered.set()
        assert release.wait(5)

    dispatcher = harness_dispatcher(tmp_path, recipe_in("manual", tmp_path, method))
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
            assert scenario.trainer.pipeline.training_mode == "auto"
            assert scenario.trainer.pending_job_data.request.text == "one"
        finally:
            release.set()
            await client.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        dispatcher.close()


def test_mode_selection_survives_a_reload_and_a_restart(tmp_path, caplog):
    def method(workspace, services):
        return None

    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, method))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert dispatcher.set_training_mode("s", "manual") == {"scenario": "s", "training_mode": "manual"}
        assert scenario.trainer.pipeline.training_mode == "manual"
        reloaded = dispatcher.registry.reload("s")
        assert reloaded is not scenario and reloaded.trainer.pipeline.training_mode == "manual"
        assert dispatcher.set_training_mode("s", "auto")["training_mode"] == "auto"
        assert dispatcher.registry.reload("s").trainer.pipeline.training_mode == "auto"
        dispatcher.set_training_mode("s", "manual")
        dispatcher.accept_record(instruction("one"))
        assert _wait(lambda: _committed_skip(dispatcher, "one") == UNCHANGED)
        assert dispatcher.get_or_create_scenario("s").scenario_step == 1
    finally:
        dispatcher.close()
    # A new process loads the committed state and the stored mode.
    restarted = harness_dispatcher(tmp_path, harness_recipe(tmp_path, method))
    try:
        loaded = restarted.get_or_create_scenario("s")
        assert loaded.scenario_step == 1 and loaded.trainer.pipeline.training_mode == "manual"
    finally:
        restarted.close()
    # A recipe that takes no instructions keeps its default, so a person can still select a mode.
    without_requests = harness_dispatcher(tmp_path, Recipe())
    try:
        with caplog.at_level(logging.WARNING, logger="reef.scenario.registry"):
            assert without_requests.get_or_create_scenario("s").trainer.pipeline.training_mode == "auto"
        assert "keeps its default training mode" in caplog.text
    finally:
        without_requests.close()


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


def raising_method(calls, *, poison="poison", error="poison method"):
    """A method that raises on the poison instruction and holds its first attempt open until released."""
    entered, release = Event(), Event()

    def method(workspace, services):
        request = workspace.request()
        text = request.text if request else None
        calls.append(text)
        if text == poison or poison is None:
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError(error)

    return method, entered, release


def test_a_step_whose_evaluation_raised_keeps_its_candidate_and_says_where_it_failed(tmp_path):
    """The candidate reached its evaluation and the scorer raised: the skip row says the step failed during its
    evaluation, and keeps the candidate's change and the method's notes, so the pages show what was tried."""
    from reef.service.release_page import failed_words

    def method(workspace, services):
        workspace.write_tree(workspace.tree().apply(TreePatch(changed_rules("marker rules\n"))))
        workspace.write_output("one rules entry", {"entries": 1})

    def score(task, result):
        raise ValueError("the scorer could not read the episode")

    dispatcher = harness_dispatcher(tmp_path, recipe_in("manual", tmp_path, method, score=score))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("broken scorer"))
        assert _wait(lambda: _committed_skip(dispatcher, "broken scorer") == "instruction failed")
        row = _committed_row(dispatcher, "broken scorer")
        assert row["failed_stage"] == "evaluating"
        assert row["error"] == "ValueError: the scorer could not read the episode"
        assert row["method_notes"] == {"entries": 1}
        assert row["patch"] == changed_rules("marker rules\n")
        assert failed_words(row).startswith("The step failed during its evaluation")
    finally:
        dispatcher.close()


def test_a_failed_step_keeps_the_selected_mode_and_the_next_instruction_runs(tmp_path):
    calls = []
    method, entered, release = raising_method(calls, poison=None)
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, method))
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
        assert _committed_row(dispatcher, "one")["error"] == "RuntimeError: poison method"
        current = dispatcher.get_or_create_scenario("s")
        assert current is not scenario and current.trainer.pipeline.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
    finally:
        release.set()
        dispatcher.close()


def test_a_failed_instruction_is_skipped_with_its_error_and_the_queue_moves_on(tmp_path):
    calls = []
    method, entered, release = raising_method(calls)
    dispatcher = harness_dispatcher(tmp_path, recipe_in("manual", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("fine"))
        dispatcher.accept_record(instruction("fine again"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "fine again") == UNCHANGED)
        # The skip row consumed the failed instruction without another method call; the fresh ones ran after it.
        assert calls == ["poison", "fine", "fine again"]
        assert _committed_skip(dispatcher, "poison") == "instruction failed"
        assert _committed_row(dispatcher, "poison")["error"] == "RuntimeError: poison method"
        assert _committed_skip(dispatcher, "fine") == UNCHANGED
        assert "error" not in _committed_row(dispatcher, "fine")
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.pipeline_status()["buffered_requests"] == 0
        assert current.trainer.instruction_failures() == {}
        assert current.records.get("s", "poison") is not None
        assert current.trainer.pipeline.training_mode == "manual"
    finally:
        release.set()
        dispatcher.close()


def asked(workspace):
    """The instruction a step answers, ``None`` for an automatic step, and the records of the step's traces."""
    request = workspace.request()
    return (None if request is None else request.text, tuple(trace.record for trace in workspace.traces()))


def test_hybrid_skips_a_failed_instruction_before_the_next_and_keeps_the_failure_path(tmp_path):
    calls = []
    entered, release = Event(), Event()

    def method(workspace, services):
        calls.append(asked(workspace))
        if calls[-1][0] == "poison":
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError("poison method")

    dispatcher = harness_dispatcher(tmp_path, recipe_in("hybrid", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("fine"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "fine") == UNCHANGED)
        # The wake after the failure consumed the failed instruction with its skip row, no second call, then ran the fresh one.
        assert calls == [("poison", ()), ("fine", ())]
        assert _committed_skip(dispatcher, "poison") == "instruction failed"
        assert _committed_row(dispatcher, "poison")["error"] == "RuntimeError: poison method"
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.pipeline.training_mode == "hybrid"
        assert current.trainer.instruction_failures() == {}
        # With the queue empty, one failing exchange at batch_size 1 is an automatic step with no instruction.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        assert _wait(lambda: _automatic_traces(dispatcher) == [1])
        assert calls[-1] == (None, ("a",))
        # The next instruction runs as an instruction step again, with nothing held beside it.
        dispatcher.accept_record(instruction("after"))
        assert _wait(lambda: _committed_skip(dispatcher, "after") == UNCHANGED)
        assert calls[2:] == [(None, ("a",)), ("after", ())]
        assert dispatcher.get_or_create_scenario("s").trainer.pipeline.training_mode == "hybrid"
    finally:
        release.set()
        dispatcher.close()


def test_hybrid_skips_a_failed_instruction_alone_and_keeps_the_units_it_carried(tmp_path):
    calls = []
    entered, release, automatic, outage = Event(), Event(), Event(), Event()

    def method(workspace, services):
        calls.append(asked(workspace))
        if calls[-1][0] == "poison":
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError("poison method")
        # The automatic step holds so the skip row can be read, then fails so the unit stays stored for the restart.
        automatic.set()
        outage.wait(10)
        raise RuntimeError("method outage")

    dispatcher = harness_dispatcher(tmp_path, recipe_in("hybrid", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        # The failing exchange arrives while the attempt holds, so the skip row's batch carries it as its sample.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        release.set()
        assert automatic.wait(5)
        # The skip row took no method call; the automatic step that followed read the unit the skip row gave back.
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
        assert skip.consumed_ids == frozenset({"poison"})
        assert current.records.get("s", "poison") is not None
        assert current.records.get("s", "report-a") is not None
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.instruction_failures() == {}
        outage.set()
    finally:
        release.set()
        outage.set()
        dispatcher.close()

    seen = []

    def method_again(workspace, services):
        seen.append(asked(workspace))

    restarted = harness_dispatcher(tmp_path, recipe_in("hybrid", tmp_path, method_again))
    try:
        loaded = restarted.get_or_create_scenario("s")
        assert loaded.scenario_step == 1
        assert loaded.records.get("s", "poison") is not None
        # Recovery replays the unit as unconsumed and never the instruction the skip row named.
        restarted.set_training_mode("s", "hybrid")
        assert _wait(lambda: _automatic_traces(restarted) == [1])
        assert seen == [(None, ("a",))]
        assert _wait(lambda: loaded.scenario_step == 2)
        assert loaded.records.get("s", "report-a") is not None
        assert loaded.trainer.pending_instructions() == 0
        assert restarted.get_or_create_scenario("s").scenario_step == 2
    finally:
        restarted.close()


def test_a_queue_of_failing_instructions_drains_without_another_record(tmp_path):
    calls = []
    method, entered, release = raising_method(calls, poison=None, error="method outage")
    dispatcher = harness_dispatcher(tmp_path, recipe_in("manual", tmp_path, method))
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
        assert _committed_row(dispatcher, "p2")["error"] == "RuntimeError: method outage"
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

    def method(workspace, services):
        text = workspace.request().text
        calls.append(text)
        if text == "poison":
            raise RuntimeError("poison method")
        # The step after the skip row holds, so the skip row is observable as the last commit.
        entered.set()
        release.wait(10)

    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        recipe_in("manual", tmp_path, method),
        factory,
        scenario_storage=SQLiteScenarioStorage(),
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
        assert skipped["error"] == "RuntimeError: poison method"
        assert calls == ["poison", "fine"]
        release.set()
        assert _wait(lambda: scenario.scenario_step == 2)
        assert _last_committed(scenario).get("skipped") == UNCHANGED
        assert scenario.trainer.pending_instructions() == 0
        assert scenario.trainer.pipeline_status()["buffered_requests"] == 0
        assert scenario.trainer.instruction_failures() == {}
        assert scenario.records.get("s", "poison") is not None
    finally:
        release.set()
        dispatcher.close()


def _last_committed(scenario):
    committed = scenario.commit_status.get("last_committed_step")
    return {} if committed is None else committed.get("metrics") or {}


class _FirstJobFails(StubTrainingComponent):
    """Fails the job of the first instruction and saves one checkpoint for every later job."""

    def train(self, job):
        if job.data.request is not None and job.data.request.text == "one":
            raise RuntimeError("job failed")
        return super().train(job)


class _InstructionThresholdJobData(ThresholdJobData):
    """The threshold batches, reading instructions in every mode."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)


@dataclass(frozen=True)
class _InstructionJobRecipe(WeightTrainingRecipe):
    """A weight recipe whose pipeline takes instructions, so each instruction is one training job."""

    def build(
        self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None, releases=None
    ):
        component = self.training_component
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: threshold_pipeline(
                context, sample2job=_InstructionThresholdJobData(context, None, None)
            ),
            job_loop=JobLoop(
                component,
                FixedJobPlan(JobRequest()),
                AlwaysSelectCheckpointPlugin(),
                scenario=scenario,
            ),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            default_job_samples=1,
        )


def _job_events(dispatcher):
    """The (instruction text, job event) of every committed row, oldest first."""
    rows = reversed(dispatcher.get_or_create_scenario("s").releases())
    return [
        (metrics.get("training_request", {}).get("text"), metrics["training_job"]["event"])
        for row in rows
        if "training_job" in (metrics := row.get("metrics") or {})
    ]


def test_a_job_trainer_runs_each_instruction_as_a_job_and_a_failed_job_does_not_stop_the_queue(tmp_path):
    component, runtime = _FirstJobFails(tmp_path / "checkpoints"), StubInferenceRuntime()
    recipe = _InstructionJobRecipe(component, runtime=runtime, training_mode="manual")
    dispatcher = harness_dispatcher(tmp_path, recipe)
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(instruction("one"))
        dispatcher.accept_record(instruction("two"))
        expected = [
            ("one", "submitted"),
            (None, "failed"),
            ("two", "submitted"),
            (None, "checkpoint"),
            (None, "finished"),
        ]
        assert _wait(lambda: _job_events(dispatcher) == expected), _job_events(dispatcher)
        # Each job trains one instruction alone; the held inference is not part of a manual batch.
        assert [job.data.request.text for job in component.submitted] == ["one", "two"]
        assert all(job.data.samples == () for job in component.submitted)
        releases = dispatcher.get_or_create_scenario("s").releases()
        failed = next(
            row["metrics"]
            for row in releases
            if row.get("metrics", {}).get("training_job", {}).get("event") == "failed"
        )
        assert failed["training_job"]["error"] == "RuntimeError: job failed"
        # Only the second job saved a checkpoint, and its release now serves.
        assert runtime.served == "w2"
        scenario = dispatcher.get_or_create_scenario("s")
        assert scenario.trainer.pipeline.training_mode == "manual"
        assert scenario.trainer.pending_instructions() == 0
    finally:
        dispatcher.close()


def test_an_instruction_runs_past_the_step_budget_and_the_failure_streak(tmp_path):
    seen = []

    def method(workspace, services):
        seen.append(asked(workspace)[0])

    recipe = recipe_in("manual", tmp_path, method, max_steps=1)
    records = SQLiteRecordStore()
    trainer = _trainer(recipe, records)
    try:
        records.append(instruction("one"))
        records.append(instruction("two"))
        rows = []
        for step in range(2):
            result = trainer.run_once(step)
            assert result is not None
            prepared = trainer.prepare_commit(result)
            trainer.commit(prepared)
            rows.append(prepared.metrics)
        assert seen == ["one", "two"]
        assert [row["training_request"]["text"] for row in rows] == ["one", "two"]
        assert [row["skipped"] for row in rows] == [UNCHANGED, UNCHANGED]
        assert trainer.run_once(2) is None

        # An automatic step past the budget skips; the two instructions above ran past it.
        backend = trainer.candidate_backend
        automatic = backend.prepare_step(JobData("auto", ()), HarnessState().to_json(), 5)
        assert automatic.metrics["skipped"] == "step budget of 1 exhausted"
        assert seen == ["one", "two"]
    finally:
        trainer.close()
        records.close()

    (tmp_path / "streak").mkdir()
    streak = recipe_in("manual", tmp_path / "streak", method, max_failure_streak=1)
    records = SQLiteRecordStore()
    trainer = _trainer(streak, records)
    try:
        backend = trainer.candidate_backend
        state = HarnessState(failure_streak=1).to_json()
        automatic = backend.prepare_step(JobData("auto", ()), state, 0)
        assert automatic.metrics["skipped"] == "failure streak of 1 rejections"
        request = TrainingRequest("Follow the request", "session", "r", "request-id")
        answered = backend.prepare_step(JobData("request-id", (), request=request), state, 0)
        assert answered.metrics["skipped"] == UNCHANGED and seen[-1] == "Follow the request"
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_manual_traffic_is_available_to_auto_without_a_replay(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="manual", job_samples=2)
    try:
        for receipt in ("a", "b"):
            records.append(inference(receipt))
            if processor is reports_pipeline:
                records.append(
                    AgentRecord.create(
                        scenario="s",
                        request_type=RequestType.REPORT,
                        payload={"score": 0, "references": [receipt]},
                    )
                )
        assert trainer.run_once() is None
        original = trainer.pipeline
        offset = trainer.data_offset
        trainer.set_training_mode("auto")
        assert trainer.run_once() is not None
        assert trainer.pipeline is original
        assert trainer.data_offset == offset
        assert [source_record_id(sample) for sample in backend.batches[-1].samples] == ["a", "b"]
    finally:
        trainer.close()
        records.close()


# -- training_mode: hybrid --------------------------------------------------


def test_hybrid_is_a_recipe_and_processor_mode_and_a_fourth_value_is_refused(tmp_path):
    recipe = recipe_in("hybrid", tmp_path, lambda workspace, services: None)
    records = SQLiteRecordStore()
    try:
        trainer = _trainer(recipe, records)
        assert trainer.pipeline.training_mode == "hybrid"
        assert trainer.pipeline.training_mode == "hybrid"
        assert trainer.pipeline.status()["buffered_requests"] == 0
        trainer.close()
        with pytest.raises(ValueError, match="training_mode"):
            replace(recipe, training_mode="either")
        with pytest.raises(ValueError, match="training_mode"):
            PipelineContext("s", training_mode="either")
        with pytest.raises(ValueError, match="training_mode"):
            reports_pipeline(PipelineContext("s")).set_training_mode("either")
    finally:
        records.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_hybrid_runs_a_queued_instruction_alone_when_no_units_are_held(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="hybrid", job_samples=2)
    try:
        assert trainer.run_once() is None
        records.append(instruction("alone"))
        result = trainer.run_once()
        assert result is not None
        batch = backend.batches[0]
        assert batch.data_id == "s:instruction:alone"
        assert batch.request.text == "alone"
        assert batch.samples == ()
        prepared = trainer.prepare_commit(result)
        assert prepared.consumed_ids == frozenset({"alone"})
        assert prepared.metrics["training_request"]["id"] == "alone"
        trainer.commit(prepared)
        assert records.get("s", "alone") is not None
        assert trainer.run_once() is None
        assert trainer.pipeline.training_mode == "hybrid"
    finally:
        trainer.close()
        records.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_hybrid_runs_a_queued_instruction_with_the_held_units_as_samples(processor):
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, processor, mode="hybrid", job_samples=2)
    try:
        failure(records, "a")
        assert trainer.run_once() is None
        records.append(instruction("with-context"))
        result = trainer.run_once()
        batch = backend.batches[0]
        assert batch.request.id == "with-context"
        assert [source_record_id(sample) for sample in batch.samples] == ["a"]
        prepared = trainer.prepare_commit(result)
        assert {"a", "with-context"} <= prepared.consumed_ids
        trainer.commit(prepared)
        assert records.get("s", "a") is not None
        records.append(instruction("after"))
        assert trainer.run_once() is not None
        assert backend.batches[1].request.id == "after"
        assert backend.batches[1].samples == ()
    finally:
        trainer.close()
        records.close()


def test_hybrid_runs_two_queued_instructions_oldest_first_one_per_step():
    processor = reports_pipeline(PipelineContext("s", training_mode="hybrid"))
    processor.push(instruction("first"))
    processor.push(instruction("second"))
    assert processor.status()["buffered_requests"] == 2
    assert processor.ready(1)
    batch = processor.build_job_data(1)
    assert batch.data_id == "s:instruction:first"
    assert batch.request.id == "first"
    assert processor.build_job_data(1) is batch
    assert processor.acknowledge(batch.data_id) == frozenset({"first"})
    assert processor.build_job_data(1).request.id == "second"
    assert processor.acknowledge("s:instruction:second") == frozenset({"second"})
    assert not processor.ready(1)
    processor.close()


def test_hybrid_batches_as_auto_does_without_an_instruction():
    processor = reports_pipeline(PipelineContext("s", training_mode="hybrid"))
    processor.push(inference("a"))
    processor.push(report("a"))
    batch = processor.build_job_data(1)
    assert batch.data_id == "s:data:1"
    assert batch.request is None
    assert [source_record_id(sample) for sample in batch.samples] == ["a"]
    assert processor.acknowledge(batch.data_id) == frozenset({"a", "report-a"})
    processor.close()


def test_hybrid_alternates_the_instruction_path_and_the_failure_path_without_a_mode_change():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, reports_pipeline, mode="hybrid", job_samples=2)

    def step():
        result = trainer.run_once()
        if result is None:
            return None
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
        batch = backend.batches[-1]
        request = None if batch.request is None else batch.request.id
        return request, [source_record_id(sample) for sample in batch.samples]

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
        assert trainer.pipeline.training_mode == "hybrid"
    finally:
        trainer.close()
        records.close()


def test_switching_hybrid_to_auto_holds_the_unread_instruction_for_a_mode_that_takes_it():
    records, backend = SQLiteRecordStore(), CaptureBackend()
    trainer = build(records, backend, reports_pipeline, mode="hybrid", job_samples=2)
    try:
        records.append(instruction("later"))
        trainer.set_training_mode("auto")
        assert trainer.run_once() is None
        assert trainer.pipeline.status()["buffered_requests"] == 1
        failure(records, "a")
        failure(records, "b")
        result = trainer.run_once()
        assert backend.batches[-1].request is None
        assert [source_record_id(sample) for sample in backend.batches[-1].samples] == ["a", "b"]
        prepared = trainer.prepare_commit(result)
        trainer.commit(prepared)
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

    def method(workspace, services):
        seen.append(asked(workspace))

    dispatcher = harness_dispatcher(tmp_path, recipe_in("hybrid", tmp_path, method))

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            assert dispatcher.get_or_create_scenario("s").trainer.pipeline.training_mode == "hybrid"
            body = {
                "agent_record_id": "ask-1",
                "text": "Prefer tests first",
                "session": "session-1",
                "release_id": "release-1",
            }
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: "Prefer tests first" in _release_texts(dispatcher))
            assert seen == [("Prefer tests first", ())]
            # The failure path stays on: a failing exchange batches by itself with no instruction queued.
            dispatcher.accept_record(inference("a"))
            dispatcher.accept_record(report("a"))
            assert await asyncio.to_thread(_wait, lambda: len(seen) == 2)
            assert seen[1] == (None, ("a",))
            assert dispatcher.get_or_create_scenario("s").trainer.pipeline.training_mode == "hybrid"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("mode", ["auto", "manual", "hybrid"])
def test_every_mode_releases_the_oldest_held_units_beyond_the_buffer(caplog, mode):
    # The default buffer holds the data of four jobs of one sample each.
    processor = reports_pipeline(PipelineContext("s", training_mode=mode, default_job_samples=1))
    assert processor.buffer_size == 4

    def fail(index):
        processor.push(inference(f"inf-{index}"))
        processor.push(
            AgentRecord.create(
                scenario="s",
                request_type=RequestType.REPORT,
                payload={"score": 0, "references": [f"inf-{index}"]},
                agent_record_id=f"rep-{index}",
            )
        )

    with caplog.at_level(logging.WARNING, logger="reef.train.pipeline"):
        for index in range(20):
            fail(index)
    assert len(processor.record2record.pool) == 4
    retention = processor.releasable_record_ids()
    assert {f"inf-{i}" for i in range(16)} | {f"rep-{i}" for i in range(16)} <= retention
    assert retention.isdisjoint({f"inf-{i}" for i in range(16, 20)} | {f"rep-{i}" for i in range(16, 20)})
    # One warning names the buffer size; later releases are not logged.
    assert len(caplog.records) == 1
    assert "buffer size 4" in caplog.records[0].message
    if mode == "manual":
        processor.push(instruction("do-it"))
        batch = processor.build_job_data(1)
        assert batch.samples == ()
        assert processor.acknowledge(batch.data_id) == frozenset({"do-it"})
        assert len(processor.record2record.pool) == 4
    else:
        reserved = processor.build_job_data(1)
        fail(20)
        fail(21)
        # The handed-out batch keeps its unit and does not count; the buffer releases the oldest unreserved one.
        assert "inf-16" not in processor.releasable_record_ids()
        assert "inf-17" in processor.releasable_record_ids()
        assert processor.acknowledge(reserved.data_id) == frozenset({"inf-16", "rep-16"})
        assert len(processor.record2record.pool) == 4
    processor.close()


def test_hybrid_collects_the_failures_an_instruction_step_carries(tmp_path):
    calls = []

    def method(workspace, services):
        calls.append(asked(workspace))

    recipe = recipe_in("hybrid", tmp_path, method, batch_size=2, evaluation={"collect_failures": True})
    dispatcher = harness_dispatcher(tmp_path, recipe)

    def evaluation_rows():
        """Each step's traces, its evaluation's task count (the seed task and the collected ones) and whether it
        answered an instruction, newest first."""
        rows = dispatcher.get_or_create_scenario("s").releases()
        return [
            (metrics["traces"], metrics["evaluation_task_count"], "training_request" in metrics)
            for row in rows
            if (metrics := row.get("metrics"))
        ]

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            # One failure held below the batch size; the instruction step takes it and collects it.
            dispatcher.accept_record(inference("fails-1"))
            dispatcher.accept_record(report("fails-1"))
            body = {"agent_record_id": "ask-1", "text": "do it", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 1)
            assert calls == [("do it", ("fails-1",))]
            assert evaluation_rows() == [(1, 2, True)]
            # The automatic step after grows the evaluation on top of the collected one.
            for receipt in ("fails-2", "fails-3"):
                dispatcher.accept_record(inference(receipt))
                dispatcher.accept_record(report(receipt))
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 2)
            # Newest first: the automatic step's row, then the instruction step's.
            assert evaluation_rows() == [(2, 4, False), (1, 2, True)]
            assert calls[1] == (None, ("fails-2", "fails-3"))
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
