"""Training instructions: manual runs each with the samples automatic job data would take next, auto refuses them."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import closing
from dataclasses import dataclass, replace
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._threshold_processor import threshold_pipeline
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
from reef.harness.tree import TreePatch
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2record import InferenceRecord2Record
from reef.record2sample.traces import InferenceSamples
from reef.sample2job import Sample2Job
from reef.service.app import create_app
from reef.storage.scenario import ScenarioSettings
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


@pytest.fixture
def records():
    store = SQLiteRecordStore()
    yield store
    store.close()


@pytest.fixture
def build(records):
    """Build trainers of scenario "s" over ``records`` (or ``store``); each closes at teardown."""
    trainers = []

    def build_trainer(backend, processor=records_pipeline, mode="manual", job_samples=1, store=None):
        trainer = Trainer.build(
            "s",
            records if store is None else store,
            pipeline_factory=processor,
            candidate_backend=backend,
            training_mode=mode,
            default_job_samples=job_samples,
        )
        trainers.append(trainer)
        return trainer

    yield build_trainer
    for trainer in trainers:
        trainer.close()


@pytest.fixture
def open_dispatcher(tmp_path):
    """Open ``harness_dispatcher`` over the test's directory; each dispatcher closes at teardown."""
    opened = []

    def open_one(recipe):
        dispatcher = harness_dispatcher(tmp_path, recipe)
        opened.append(dispatcher)
        return dispatcher

    yield open_one
    for dispatcher in reversed(opened):
        dispatcher.close()


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
def test_manual_waits_for_an_instruction_and_gives_it_the_oldest_held_units(records, build, processor, batch_size):
    backend = CaptureBackend()
    trainer = build(backend, processor, job_samples=batch_size)
    for receipt in ("other-session", "turn-1", "turn-2"):
        records.append(inference(receipt))
    records.append(
        AgentRecord.create(
            scenario="s", request_type=RequestType.REPORT, payload={"score": 0, "references": ["other-session"]}
        )
    )
    # Held units start no job in manual, however many there are.
    assert trainer.run_once() is None
    assert not trainer.job_data_ready()
    records.append(instruction("request-1"))
    result = trainer.run_once()
    assert result is not None
    batch = backend.batches[0]
    assert batch.request.text == "request-1"
    # The harness stage reads the instruction, so it runs with the held units it has, up to job_samples.
    held = ["other-session", "turn-1", "turn-2"] if processor is records_pipeline else ["other-session"]
    taken = held[:batch_size]
    assert [source_record_id(sample) for sample in batch.samples] == taken
    prepared = trainer.prepare_commit(result)
    assert {"request-1", *taken} <= prepared.consumed_ids
    assert not ({"other-session", "turn-1", "turn-2"} - set(taken)) & prepared.consumed_ids
    assert prepared.metrics["training_request"]["text"] == "request-1"
    trainer.commit(prepared)
    assert records.get("s", "turn-1") is not None
    assert trainer.run_once() is None
    assert not records.append_result(instruction("request-1")).inserted
    assert trainer.run_once() is None


def test_auto_keeps_recipe_batching(records, build):
    backend = CaptureBackend()
    trainer = build(backend, mode="auto", job_samples=2)
    records.append(inference("a"))
    assert trainer.run_once() is None
    records.append(inference("b"))
    assert trainer.run_once() is not None
    assert [source_record_id(sample) for sample in backend.batches[0].samples] == ["a", "b"]
    assert backend.batches[0].request is None


def test_manual_recovery_replays_pending_requests_but_not_committed_ones(tmp_path, build):
    path = tmp_path / "records.sqlite"
    records, backend = SQLiteRecordStore(path), CaptureBackend()
    first = build(backend, store=records)
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
    recovered = build(backend, store=records)
    recovered.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    recovered.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    assert recovered.run_once() is not None
    assert [batch.request.text for batch in backend.batches] == ["committed", "pending"]
    recovered.close()
    records.close()


def test_train_route_needs_no_inference_and_retries_do_not_train_twice(tmp_path, open_dispatcher):
    called = Event()
    seen = []

    def method(workspace, services):
        seen.append((workspace.request(), workspace.traces()))
        called.set()

    dispatcher = open_dispatcher(replace(recipe_in("manual", tmp_path, method), pipeline_stages=harness_stages()))
    dispatcher.get_or_create_scenario("s")

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
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

    asyncio.run(run())
    dispatcher.close()
    assert len(seen) == 1


def test_auto_rejects_manual_requests(tmp_path, open_dispatcher):
    dispatcher = open_dispatcher(harness_recipe(tmp_path, lambda workspace, services: None))
    dispatcher.get_or_create_scenario("s")
    with pytest.raises(ValueError, match="training_mode='manual'"):
        dispatcher.accept_record(instruction("one"))


def test_manual_is_a_native_contract_for_arbitrary_batch_schemas(records, build):
    class InstructionJobData(Sample2Job):
        supported_training_modes = frozenset({"manual"})
        reads_instructions = True
        required_request_types = frozenset(RequestType)
        output_schema = ExampleJobData

        def make_job_data(self, samples, data_id):
            return ExampleJobData(data_id, values=tuple(source_record_id(task) for task in samples))

    def instruction_pipeline(context):
        return TrainingPipeline(context, sample2job=InstructionJobData(context, None, None))

    with pytest.raises(NotImplementedError, match="training_mode='auto'"):
        instruction_pipeline(PipelineContext("s"))
    backend = CaptureBackend()
    trainer = build(backend, instruction_pipeline)
    records.append(instruction("change"))
    assert trainer.run_once() is not None
    assert isinstance(backend.batches[0], ExampleJobData)
    # The pipeline attaches the instruction to the recipe's own batch type; with nothing held it has no tasks.
    assert backend.batches[0].data_id == "s:instruction:change"
    assert backend.batches[0].request.text == "change"
    assert backend.batches[0].values == ()


def test_pipeline_context_keeps_mode_when_recipe_takes_instructions():
    context = PipelineContext("s", training_mode="manual")
    processor = reports_pipeline(context)
    assert processor.training_mode == "manual"
    assert RequestType.TRAIN in processor.required_request_types
    processor.close()


def test_missing_mode_implementation_fails_at_processor_initialization():
    class AutoOnlyJobData(Sample2Job):
        supported_training_modes = frozenset({"auto"})

        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="manual")
    with pytest.raises(NotImplementedError, match="AutoOnlyJobData does not implement training_mode='manual'"):
        TrainingPipeline(context, sample2job=AutoOnlyJobData(context, None, None))
    # The default job data trains nothing, so it is automatic only and a manual recipe has to name its own.
    with pytest.raises(NotImplementedError, match="does not implement training_mode='manual'"):
        TrainingPipeline(context)


def test_a_weight_stage_in_manual_trains_its_job_samples_only_for_an_instruction():
    """A stage that does not read instructions takes one as a request to train: the job waits for job_samples."""

    class WeightJobData(Sample2Job):
        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="manual")
    pipeline = TrainingPipeline(
        context,
        record2record=InferenceRecord2Record(),
        record2sample=InferenceSamples(),
        sample2job=WeightJobData(context, None, None),
    )
    pipeline.push(inference("a"))
    pipeline.push(inference("b"))
    # The job samples are held, but no instruction is queued: manual starts no job.
    assert not pipeline.ready(2)
    pipeline.push(instruction("one"))
    batch = pipeline.build_job_data(2)
    assert batch.data_id == "s:instruction:one"
    assert batch.request.id == "one"
    assert [source_record_id(sample) for sample in batch.samples] == ["a", "b"]
    assert pipeline.acknowledge(batch.data_id) == frozenset({"a", "b", "one"})
    # An instruction with fewer held samples than job_samples waits for the rest.
    pipeline.push(inference("c"))
    pipeline.push(instruction("two"))
    assert not pipeline.ready(2)
    pipeline.push(inference("d"))
    batch = pipeline.build_job_data(2)
    assert batch.request.id == "two"
    assert [source_record_id(sample) for sample in batch.samples] == ["c", "d"]
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
    assert pipeline.status() == {"ready_samples": 0, "buffer_size": 6, "buffered_requests": 0}
    for index in range(3):
        pipeline.push(inference(f"inf-{index}"))
    assert pipeline.status() == {"ready_samples": 3, "buffer_size": 6, "buffered_requests": 0}
    assert not pipeline.ready(4)
    pipeline.push(inference("inf-3"))
    assert pipeline.status()["ready_samples"] == 4
    assert pipeline.ready(4)
    pipeline.close()


def test_status_and_request_types_name_instructions_only_for_a_stage_that_supports_manual():
    class WeightJobData(Sample2Job):
        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    class AutoOnlyJobData(WeightJobData):
        supported_training_modes = frozenset({"auto"})

    for job_data, takes_instructions in ((WeightJobData, True), (AutoOnlyJobData, False)):
        context = PipelineContext("s")
        pipeline = TrainingPipeline(context, sample2job=job_data(context, None, None))
        pipeline.push(instruction("one"))
        assert pipeline.status().get("buffered_requests") == (1 if takes_instructions else None)
        assert (RequestType.TRAIN in pipeline.required_request_types) == takes_instructions
        pipeline.close()


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_processor_uses_shared_data_and_one_batch_assembly_hook(records, build, mode):
    class TrajectoryJobData(Sample2Job):
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

    backend = CaptureBackend()
    trainer = build(backend, trajectory_pipeline, mode=mode, job_samples=2)
    records.append(inference("first"))
    assert trainer.run_once() is None
    records.append(inference("second"))
    if mode == "manual":
        # The job samples are held, and manual still waits for an instruction.
        assert trainer.run_once() is None
        records.append(instruction("use-the-trajectory"))
    result = trainer.run_once()
    # The same hook shapes the job data of both modes.
    assert isinstance(backend.batches[0], ExampleJobData)
    assert backend.batches[0].values == ("first", "second")
    assert (backend.batches[0].request is None) == (mode == "auto")
    # The pipeline assigns the ID of each batch, and the hook uses it.
    expected_id = "s:data:1" if mode == "auto" else "s:instruction:use-the-trajectory"
    assert backend.batches[0].data_id == expected_id
    prepared = trainer.prepare_commit(result)
    expected = {"first", "second"} | ({"use-the-trajectory"} if mode == "manual" else set())
    assert prepared.consumed_ids == frozenset(expected)
    trainer.commit(prepared)
    assert trainer.run_once() is None


def test_factory_cannot_silently_change_the_processor_mode(build):
    with pytest.raises(ValueError, match="preserve the requested training_mode"):
        build(CaptureBackend(), lambda ctx: TrainingPipeline(replace(ctx, training_mode="auto")))


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_switch_during_reserved_batch_keeps_original_acknowledgement(records, build, mode):
    trainer = build(CaptureBackend(), mode=mode)
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


def test_switch_preserves_incomplete_auto_batch_and_unread_manual_instructions(records, build):
    backend = CaptureBackend()
    trainer = build(backend, mode="auto", job_samples=2)
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
    trainer.commit(trainer.prepare_commit(result))
    assert records.get("s", "change") is not None
    trainer.set_training_mode("manual")
    result = trainer.run_once(1)
    assert backend.batches[-1].request.id == "change"
    trainer.commit(trainer.prepare_commit(result))
    trainer.set_training_mode("auto")
    assert trainer.run_once(2) is None


@pytest.mark.parametrize("instructions", [False, True], ids=["record-only", "harness"])
def test_http_training_mode_updates_only_existing_supported_processors(tmp_path, open_dispatcher, instructions):
    # A record-only recipe makes no job data, so it trains in auto alone; a harness recipe takes instructions.
    recipe = harness_recipe(tmp_path, lambda workspace, services: None) if instructions else Recipe()
    dispatcher = open_dispatcher(recipe)
    scenario = dispatcher.get_or_create_scenario("s")
    processor = scenario.trainer.pipeline

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            url = "/reef/scenarios/s/update"
            response = await client.post(url, json={"training_mode": "manual"})
            assert response.status == (200 if instructions else 501)
            assert scenario.trainer.pipeline.training_mode == ("manual" if instructions else "auto")
            assert scenario.trainer.pipeline is processor
            if instructions:
                # The status lists the scenarios that train; a record-only one does not.
                assert await response.json() == {"scenario": "s", "training_mode": "manual"}
                status = await (await client.get("/reef/status")).json()
                assert status["scenarios"]["s"]["training_mode"] == "manual"
            for invalid in (
                {},
                {"training_mode": "hybrid"},
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

    asyncio.run(run())


def test_http_mode_change_does_not_wait_for_running_method(tmp_path, open_dispatcher):
    entered, release = Event(), Event()

    def method(workspace, services):
        entered.set()
        assert release.wait(5)

    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method))
    dispatcher.get_or_create_scenario("s")

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
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

    asyncio.run(run())


def test_mode_selection_survives_a_reload_and_a_restart(tmp_path, open_dispatcher, caplog):
    def method(workspace, services):
        return None

    dispatcher = open_dispatcher(harness_recipe(tmp_path, method))
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
    dispatcher.close()
    # A new process loads the committed state and the stored mode.
    restarted = open_dispatcher(harness_recipe(tmp_path, method))
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 1 and loaded.trainer.pipeline.training_mode == "manual"
    restarted.close()
    # A recipe that takes no instructions keeps its default, so a person can still select a mode.
    without_requests = open_dispatcher(Recipe())
    with caplog.at_level(logging.WARNING, logger="reef.scenario.registry"):
        assert without_requests.get_or_create_scenario("s").trainer.pipeline.training_mode == "auto"
    assert "keeps its default training mode" in caplog.text


def test_a_stored_hybrid_mode_loads_the_scenario_with_the_recipe_default(tmp_path, open_dispatcher, caplog):
    def method(workspace, services):
        return None

    dispatcher = open_dispatcher(harness_recipe(tmp_path, method))
    dispatcher.get_or_create_scenario("s")
    dispatcher.close()
    # The setting of a scenario that selected hybrid before Reef removed the mode.
    with closing(SQLiteScenarioStorage(tmp_path / "agent-record")) as storage:
        storage.write_settings("s", ScenarioSettings(training_mode="hybrid"))
    restarted = open_dispatcher(recipe_in("manual", tmp_path, method))
    with caplog.at_level(logging.WARNING, logger="reef.scenario.registry"):
        assert restarted.get_or_create_scenario("s").trainer.pipeline.training_mode == "manual"
    assert "keeps its default training mode: the stored mode 'hybrid' is not supported" in caplog.text


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


def asked(workspace):
    """The instruction a step answers, ``None`` for an automatic step, and the records of the step's traces."""
    request = workspace.request()
    return (None if request is None else request.text, tuple(trace.record for trace in workspace.traces()))


def raising_method(calls, *, poison="poison", error="poison method", with_traces=False):
    """A method that raises on the poison instruction (on every call when ``poison`` is None) and holds its first
    attempt open until released. It records each instruction text, with the trace records if ``with_traces``."""
    entered, release = Event(), Event()

    def method(workspace, services):
        text, traces = asked(workspace)
        calls.append((text, traces) if with_traces else text)
        if text == poison or poison is None:
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError(error)

    return method, entered, release


def test_a_step_whose_evaluation_raised_keeps_its_candidate_and_says_where_it_failed(tmp_path, open_dispatcher):
    """The candidate reached its evaluation and the scorer raised: the skip row says the step failed during its
    evaluation, and keeps the candidate's change and the method's notes, so the pages show what was tried."""
    from reef.service.release_page import failed_words

    def method(workspace, services):
        workspace.write_tree(workspace.tree().apply(TreePatch(changed_rules("marker rules\n"))))
        workspace.write_output("one rules entry", {"entries": 1})

    def score(task, result):
        raise ValueError("the scorer could not read the rollout")

    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method, score=score))
    dispatcher.get_or_create_scenario("s")
    dispatcher.accept_record(instruction("broken scorer"))
    assert _wait(lambda: _committed_skip(dispatcher, "broken scorer") == "instruction failed")
    row = _committed_row(dispatcher, "broken scorer")
    assert row["failed_stage"] == "evaluating"
    assert row["error"] == "ValueError: the scorer could not read the rollout"
    assert row["method_notes"] == {"entries": 1}
    assert row["patch"] == changed_rules("marker rules\n")
    assert failed_words(row).startswith("The step failed during its evaluation")


def test_a_queue_of_failing_instructions_drains_and_keeps_the_selected_mode(tmp_path, open_dispatcher):
    """No further record arrives after the two instructions, so the failures wake the worker until both are consumed.
    A failed instruction is not run again, and the reload after each failure keeps the selected mode."""
    calls = []
    method, entered, release = raising_method(calls, poison=None, error="method outage")
    dispatcher = open_dispatcher(harness_recipe(tmp_path, method))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert scenario.store.durable
        dispatcher.set_training_mode("s", "manual")
        dispatcher.accept_record(instruction("p1"))
        assert entered.wait(5)
        dispatcher.accept_record(instruction("p2"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "p1") == "instruction failed")
        assert _wait(lambda: _committed_skip(dispatcher, "p2") == "instruction failed")
        assert calls == ["p1", "p2"]
        assert _committed_row(dispatcher, "p2")["error"] == "RuntimeError: method outage"
        current = dispatcher.get_or_create_scenario("s")
        assert current is not scenario and current.trainer.pipeline.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
        assert dispatcher.accept_record(instruction("p3")).agent_record_id == "p3"
        assert _wait(lambda: _committed_skip(dispatcher, "p3") == "instruction failed")
    finally:
        release.set()


def test_a_failed_instruction_is_skipped_with_its_error_and_the_queue_moves_on(tmp_path, open_dispatcher):
    calls = []
    method, entered, release = raising_method(calls)
    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method))
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


def test_a_failed_instruction_is_skipped_alone_and_its_units_wait_for_the_next_instruction(tmp_path, open_dispatcher):
    calls = []
    method, entered, release = raising_method(calls, with_traces=True)
    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        # The failing exchange arrives while the attempt holds, so the skip row's job data carries it as its sample.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "poison") == "instruction failed")
        # The skip row took no method call.
        assert calls == [("poison", ())]
        current = dispatcher.get_or_create_scenario("s")
        skip = next(
            record
            for record in current.store.history()
            if record.metrics is not None and "training_request" in record.metrics
        )
        # The skip row consumed the instruction alone: the unit it carried stays held.
        assert skip.consumed_ids == frozenset({"poison"})
        assert current.records.get("s", "report-a") is not None
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.instruction_failures() == {}
    finally:
        release.set()
    dispatcher.close()

    seen = []

    def method_again(workspace, services):
        seen.append(asked(workspace))

    restarted = open_dispatcher(recipe_in("manual", tmp_path, method_again))
    assert restarted.get_or_create_scenario("s").scenario_step == 1
    # Recovery replays the unit as unconsumed and never the instruction that the skip row named.
    restarted.accept_record(instruction("after"))
    assert _wait(lambda: _committed_skip(restarted, "after") == UNCHANGED)
    assert seen == [("after", ("a",))]
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 2
    assert loaded.records.get("s", "poison") is not None
    assert loaded.trainer.pending_instructions() == 0


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


class SubmitRaisesForInstructionOne(StubTrainingComponent):
    """Raises from ``submit`` for the job of the instruction ``one``; ``submit_texts`` lists each submitted
    instruction's text, the failed one too."""

    def __init__(self, checkpoint_root):
        super().__init__(checkpoint_root)
        self.submit_texts = []

    def submit(self, job):
        self.submit_texts.append(job.data.request.text)
        if job.data.request.text == "one":
            raise RuntimeError("submit failed")
        super().submit(job)


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
            pipeline_factory=threshold_pipeline,
            job_loop=JobLoop(
                component,
                FixedJobPlan(JobRequest()),
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


def test_a_job_trainer_runs_each_instruction_as_a_job_and_a_failed_job_does_not_stop_the_queue(
    tmp_path, open_dispatcher
):
    component, runtime = _FirstJobFails(tmp_path / "checkpoints"), StubInferenceRuntime()
    dispatcher = open_dispatcher(_InstructionJobRecipe(component, runtime=runtime, training_mode="manual"))
    served = dispatcher.get_or_create_scenario("s").serving_runtime_load_id()
    for receipt in ("a", "b"):
        # The exchange names the served version, so the staleness check keeps it for the job.
        dispatcher.accept_record(
            AgentRecord.create(
                scenario="s",
                request_type=RequestType.INFERENCE,
                payload={"response": {"training": {"runtime_load_id": served}}},
                agent_record_id=receipt,
            )
        )
        dispatcher.accept_record(report(receipt))
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
    # Each job trains one instruction with the job's one sample, the oldest held one.
    assert [
        (job.data.request.text, [source_record_id(sample) for sample in job.data.samples])
        for job in component.submitted
    ] == [("one", ["a"]), ("two", ["b"])]
    releases = dispatcher.get_or_create_scenario("s").releases()
    failed = next(
        row["metrics"] for row in releases if row.get("metrics", {}).get("training_job", {}).get("event") == "failed"
    )
    assert failed["training_job"]["error"] == "RuntimeError: job failed"
    # Only the second job saved a checkpoint, and its release now serves.
    assert runtime.served == "w2"
    scenario = dispatcher.get_or_create_scenario("s")
    assert scenario.trainer.pipeline.training_mode == "manual"
    assert scenario.trainer.pending_instructions() == 0


def test_a_job_trainer_skips_a_failed_instruction_instead_of_submitting_it_again(tmp_path, records):
    component = SubmitRaisesForInstructionOne(tmp_path / "checkpoints")
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=threshold_pipeline,
        job_loop=JobLoop(component, FixedJobPlan(JobRequest()), scenario="s"),
        training_mode="manual",
        default_job_samples=1,
    )
    failure(records, "a")
    records.append(instruction("one"))
    with pytest.raises(RuntimeError, match="submit failed"):
        trainer.run_once()
    # What the dispatcher does when an instruction's step raises.
    assert trainer.fail_pending_instruction("RuntimeError: submit failed")
    result = trainer.run_once()
    assert result.metrics["skipped"] == "instruction failed"
    assert result.metrics["error"] == "RuntimeError: submit failed"
    prepared = trainer.prepare_commit(result)
    assert prepared.consumed_ids == frozenset({"one"})
    trainer.commit(prepared)
    # The next instruction's job takes the unit that the skipped instruction left held.
    records.append(instruction("two"))
    assert trainer.run_once() is not None
    assert component.submit_texts == ["one", "two"]
    assert [source_record_id(sample) for sample in component.submitted[-1].data.samples] == ["a"]
    trainer.close()


def test_an_instruction_runs_past_the_step_budget_and_the_failure_streak(tmp_path, records):
    seen = []

    def method(workspace, services):
        seen.append(asked(workspace)[0])

    trainer = _trainer(recipe_in("manual", tmp_path, method, max_steps=1), records)
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
    trainer.close()

    (tmp_path / "streak").mkdir()
    trainer = _trainer(recipe_in("manual", tmp_path / "streak", method, max_failure_streak=1), records)
    backend = trainer.candidate_backend
    state = HarnessState(failure_streak=1).to_json()
    automatic = backend.prepare_step(JobData("auto", ()), state, 0)
    assert automatic.metrics["skipped"] == "failure streak of 1 rejections"
    request = TrainingRequest("Follow the request", "session", "r", "request-id")
    answered = backend.prepare_step(JobData("request-id", (), request=request), state, 0)
    assert answered.metrics["skipped"] == UNCHANGED and seen[-1] == "Follow the request"
    trainer.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_manual_traffic_is_available_to_auto_without_a_replay(records, build, processor):
    backend = CaptureBackend()
    trainer = build(backend, processor, mode="manual", job_samples=2)
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


def test_hybrid_and_other_unknown_modes_are_refused(tmp_path):
    recipe = harness_recipe(tmp_path, lambda workspace, services: None)
    pipeline = reports_pipeline(PipelineContext("s"))
    for mode in ("hybrid", "either"):
        with pytest.raises(ValueError, match="training_mode must be 'auto' or 'manual'"):
            replace(recipe, training_mode=mode)
        with pytest.raises(ValueError, match="training_mode must be 'auto' or 'manual'"):
            PipelineContext("s", training_mode=mode)
        with pytest.raises(ValueError, match="training_mode must be 'auto' or 'manual'"):
            pipeline.set_training_mode(mode)
    pipeline.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_manual_runs_a_harness_instruction_alone_when_no_units_are_held(records, build, processor):
    backend = CaptureBackend()
    trainer = build(backend, processor, job_samples=2)
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
    assert trainer.pipeline.training_mode == "manual"


def test_manual_runs_two_queued_instructions_oldest_first_one_per_step():
    processor = reports_pipeline(PipelineContext("s", training_mode="manual"))
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


def test_manual_gives_each_instruction_at_most_job_samples_of_the_oldest_held_failures(records, build):
    backend = CaptureBackend()
    trainer = build(backend, reports_pipeline, job_samples=2)

    def step():
        result = trainer.run_once()
        if result is None:
            return None
        trainer.commit(trainer.prepare_commit(result))
        batch = backend.batches[-1]
        return batch.request.id, [source_record_id(sample) for sample in batch.samples]

    for receipt in ("a", "b", "c"):
        failure(records, receipt)
    # The failures start no job: they wait for an instruction.
    assert step() is None
    records.append(instruction("one"))
    assert step() == ("one", ["a", "b"])
    failure(records, "d")
    records.append(instruction("two"))
    assert step() == ("two", ["c", "d"])
    failure(records, "e")
    assert step() is None


@pytest.mark.parametrize("mode", ["auto", "manual"])
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
    reserved = processor.build_job_data(1)
    fail(20)
    fail(21)
    # The handed-out job data keeps its unit and does not count; the buffer releases the oldest unreserved one.
    assert "inf-16" not in processor.releasable_record_ids()
    assert "inf-17" in processor.releasable_record_ids()
    instruction_ids = {"do-it"} if mode == "manual" else set()
    assert processor.acknowledge(reserved.data_id) == frozenset({"inf-16", "rep-16", *instruction_ids})
    assert len(processor.record2record.pool) == 4
    processor.close()


def test_manual_collects_the_failures_each_instruction_step_carries(tmp_path, open_dispatcher):
    calls = []

    def method(workspace, services):
        calls.append(asked(workspace))

    recipe = recipe_in("manual", tmp_path, method, batch_size=2, evaluation={"collect_failures": True})
    dispatcher = open_dispatcher(recipe)

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
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            # One failure held below the batch size; the instruction step takes it and collects it.
            dispatcher.accept_record(inference("fails-1"))
            dispatcher.accept_record(report("fails-1"))
            body = {"agent_record_id": "ask-1", "text": "do it", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 1)
            assert calls == [("do it", ("fails-1",))]
            assert evaluation_rows() == [(1, 2, True)]
            # The next instruction step takes two failures and grows the evaluation on top of the collected one.
            for receipt in ("fails-2", "fails-3"):
                dispatcher.accept_record(inference(receipt))
                dispatcher.accept_record(report(receipt))
            body = {"agent_record_id": "ask-2", "text": "again", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 2)
            # Newest first: the second instruction step's row, then the first one's.
            assert evaluation_rows() == [(2, 4, True), (1, 2, True)]
            assert calls[1] == ("again", ("fails-2", "fails-3"))

    asyncio.run(run())
