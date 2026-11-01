"""Training instructions: manual runs each with the samples of the manual buffer, which a client fills; auto refuses
them."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import closing
from dataclasses import dataclass, replace
from functools import partial
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._threshold_processor import threshold_pipeline
from tests.support.harness_service import catalog, drive_trainer_job, harness_dispatcher, job_backend
from tests.support.harness_steps import COMPONENT, UNCHANGED, TreeReleases, changed_rules, harness_recipe
from tests.support.runtime_stubs import RecordingComponent, StubInferenceRuntime, StubTrainingComponent
from tests.support.samples_service import make_samples as produce_samples
from tests.train.test_harness_evolve_processor import harness_stages, records_pipeline, reports_pipeline
from tests.train.test_reef_trainer_contracts import ExampleJobData

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest, normalize_training_request_payload
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.harness.tree import TreePatch
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2sample.traces import InferenceSamples
from reef.sample2job import Sample2Job
from reef.service.app import create_app
from reef.storage.samples import CONSUMED_SAMPLE_PREFIX
from reef.storage.scenario import ScenarioSettings
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import ComponentReleases, ReleaseFiles
from reef.train.harness_backend.state import HarnessState
from reef.train.instructions import ScenarioInstructions
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.types import JobData, Sample


def commit_running_job(trainer):
    """Commit the end of the job whose submission ``trainer`` committed, when one runs: a ``RecordingComponent`` job
    ends in ``submit``, so its end is the trainer's next result."""
    while trainer.job_running:
        ended = trainer.run_once()
        assert ended is not None, "the running job has its end to commit"
        trainer.commit(trainer.prepare_commit(ended))


def inference(receipt, scenario="s"):
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": receipt}]},
        agent_record_id=receipt,
    )


def instruction(receipt):
    """The ``POST /reef/train`` body of the instruction ``receipt``."""
    return {"text": receipt, "session": "session-1", "release_id": "release-1"}


def request(receipt):
    """The instruction ``receipt`` as the scenario gives it to a pipeline."""
    return TrainingRequest.from_dict(normalize_training_request_payload(instruction(receipt)), request_id=receipt)


#: The instructions queue of the running test (the ``instructions`` fixture); ``make_samples`` gives it to a trainer,
#: as a scenario does before each step.
TEST_INSTRUCTIONS: list[ScenarioInstructions] = []


def make_samples(trainer):
    """Give ``trainer`` the test's queued instructions, then make the samples of its new records."""
    for instructions in TEST_INSTRUCTIONS:
        instructions.give([trainer])
    return produce_samples(trainer)


def file_instruction(dispatcher, receipt):
    """Queue the instruction ``receipt`` of scenario "s" through the dispatcher; return its ID."""
    return dispatcher.accept_instruction("s", instruction(receipt), request_id=receipt).id


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


def sample_ids_of(pipeline, *receipts):
    """The IDs of the samples in the store of ``pipeline`` that the records ``receipts`` made, in store order."""
    return tuple(stored.sample_id for stored in pipeline.store.page() if source_record_id(stored.sample) in receipts)


def insert_samples_of(dispatcher, *receipts):
    """Wait until scenario "s" stores one sample of each record of ``receipts``, then insert them into its manual
    buffer, as a client does."""

    def stored_ids():
        stored = dispatcher.stored_samples("s", after_sequence=0, limit=1000)
        return tuple(item.sample_id for item in stored if source_record_id(item.sample) in receipts)

    assert _wait(lambda: len(stored_ids()) == len(receipts)), stored_ids()
    dispatcher.change_buffer("s", stored_ids(), insert=True)


@pytest.fixture
def records():
    store = SQLiteRecordStore()
    yield store
    store.close()


@pytest.fixture
def instructions():
    """The instructions queue of scenario "s", in memory; ``make_samples`` gives it to the trainer that it steps."""
    queue = ScenarioInstructions(memory_coordination().work_queue, "s")
    TEST_INSTRUCTIONS.append(queue)
    yield queue
    TEST_INSTRUCTIONS.remove(queue)


def file(instructions, receipt):
    """Queue the instruction ``receipt``, as ``POST /reef/train`` does."""
    return instructions.file(receipt, normalize_training_request_payload(instruction(receipt)))


@pytest.fixture
def build(records, instructions):
    """Build trainers of scenario "s" over ``records`` (or ``store``) and ``instructions``; each closes at teardown.

    ``processor`` takes the context and the stage's ``job_samples``.
    """
    trainers = []

    def build_trainer(component, processor=records_pipeline, mode="manual", job_samples=1, store=None):
        trainer = Trainer.build(
            "s",
            records if store is None else store,
            pipeline_factory=partial(processor, job_samples=job_samples),
            job_loop=JobLoop(component, scenario="s"),
            training_mode=mode,
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
    rows = catalog(dispatcher.get_or_create_scenario(scenario))
    return [row.get("metrics", {}).get("training_request", {}).get("text") for row in rows]


def recipe_in(mode, tmp_path, method, **fields) -> HarnessEvolutionRecipe:
    """A pi harness recipe over ``method`` in training mode ``mode``; ``fields`` go to ``harness_recipe``."""
    return replace(harness_recipe(tmp_path, method, **fields), training_mode=mode)


def _trainer(recipe: HarnessEvolutionRecipe, records) -> Trainer:
    """The recipe's trainer over an in-memory commit history whose base is the recipe's seed."""
    return recipe.build("s", records, releases=ComponentReleases(TreeReleases(recipe.seed), COMPONENT))


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
@pytest.mark.parametrize("job_samples", [1, 100])
def test_manual_waits_for_an_instruction_and_trains_the_inserted_samples_whatever_job_samples(
    records, build, processor, job_samples, instructions
):
    backend = RecordingComponent()
    trainer = build(backend, processor, job_samples=job_samples)
    for receipt in ("other-session", "turn-1", "turn-2"):
        records.append(inference(receipt))
    records.append(
        AgentRecord.create(
            scenario="s", request_type=RequestType.REPORT, payload={"score": 0, "references": ["other-session"]}
        )
    )
    # Held samples start no job in manual, however many there are, and none joins the manual buffer by itself.
    make_samples(trainer)
    assert trainer.run_once() is None
    assert not trainer.job_data_ready()
    assert trainer.pipeline.buffer() == ()
    held = [source_record_id(sample) for _, sample in trainer.pipeline.held.values()]
    assert held == (["other-session", "turn-1", "turn-2"] if processor is records_pipeline else ["other-session"])
    # The client chooses the newest held samples; the instruction trains each of them, whatever job_samples.
    chosen = held[-2:]
    trainer.change_buffer(sample_ids_of(trainer.pipeline, *chosen), insert=True)
    file(instructions, "request-1")
    make_samples(trainer)
    result = trainer.run_once()
    assert result is not None
    batch = backend.batches[0]
    assert batch.request.text == "request-1"
    assert [source_record_id(sample) for sample in batch.samples] == chosen
    prepared = trainer.prepare_commit(result)
    assert {"request-1", *chosen} <= prepared.consumed_ids
    # The job data consumes the samples it takes; the samples that the client did not choose wait.
    assert len({item for item in prepared.consumed_ids if item.startswith(CONSUMED_SAMPLE_PREFIX)}) == len(chosen)
    assert prepared.metrics["training_request"]["text"] == "request-1"
    trainer.commit(prepared)
    commit_running_job(trainer)
    # The commit empties the manual buffer.
    assert trainer.pipeline.buffer() == ()
    assert len(trainer.pipeline.held) == len(held) - len(chosen)
    assert records.get("s", "turn-1") is not None
    make_samples(trainer)
    assert trainer.run_once() is None
    # A retry of the instruction returns its queue item and does not train again.
    assert file(instructions, "request-1").final
    make_samples(trainer)
    assert trainer.run_once() is None


def test_auto_keeps_recipe_batching(records, build):
    backend = RecordingComponent()
    trainer = build(backend, mode="auto", job_samples=2)
    records.append(inference("a"))
    make_samples(trainer)
    assert trainer.run_once() is None
    records.append(inference("b"))
    make_samples(trainer)
    assert trainer.run_once() is not None
    assert [source_record_id(sample) for sample in backend.batches[0].samples] == ["a", "b"]
    assert backend.batches[0].request is None


def test_manual_recovery_replays_pending_requests_but_not_committed_ones(tmp_path, build, instructions):
    path = tmp_path / "records.sqlite"
    records, backend = SQLiteRecordStore(path), RecordingComponent()
    first = build(backend, store=records)
    records.append(inference("a"))
    file(instructions, "committed")
    make_samples(first)
    result = first.run_once()
    prepared = first.prepare_commit(result)
    first.commit(prepared)
    # Simulate a crash after the commit history landed but before compaction.
    file(instructions, "pending")
    first.close()
    records.close()
    records = SQLiteRecordStore(path)
    recovered = build(backend, store=records)
    recovered.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    recovered.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    make_samples(recovered)
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
        file_instruction(dispatcher, "one")


def test_manual_is_a_native_contract_for_arbitrary_batch_schemas(records, build, instructions):
    class InstructionJobData(Sample2Job):
        supported_training_modes = frozenset({"manual"})
        reads_instructions = True
        required_request_types = frozenset(RequestType)
        output_schema = ExampleJobData

        def make_job_data(self, samples, data_id):
            return ExampleJobData(data_id, values=tuple(source_record_id(task) for task in samples))

    def instruction_pipeline(context, job_samples=None):
        return TrainingPipeline(context, sample2job=InstructionJobData(context, None, job_samples))

    with pytest.raises(NotImplementedError, match="training_mode='auto'"):
        instruction_pipeline(PipelineContext("s"))
    backend = RecordingComponent()
    trainer = build(backend, instruction_pipeline)
    file(instructions, "change")
    make_samples(trainer)
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


def test_a_weight_stage_in_manual_trains_the_inserted_samples_only_for_an_instruction():
    """A stage that does not read instructions takes one as a request to train the manual buffer: the job waits for
    one or more inserted samples, and then trains all of them, whatever job_samples."""

    class WeightJobData(Sample2Job):
        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    context = PipelineContext("s", training_mode="manual")
    pipeline = TrainingPipeline(
        context,
        record2sample=InferenceSamples(),
        sample2job=WeightJobData(context, None, None),
    )
    pipeline.push(inference("a"))
    pipeline.push(inference("b"))
    pipeline.insert_into_buffer(sample_ids_of(pipeline, "a", "b"))
    # The job samples are in the buffer, but no instruction is queued: manual starts no job.
    assert not pipeline.ready(2)
    pipeline.queue_instructions([request("one")])
    batch = pipeline.build_job_data(2)
    assert batch.data_id == "s:instruction:one"
    assert batch.request.id == "one"
    assert [source_record_id(sample) for sample in batch.samples] == ["a", "b"]
    assert pipeline.acknowledge(batch.data_id) == frozenset({"sample:a:0", "sample:b:0", "one"})
    assert pipeline.buffer() == ()
    # An instruction waits while the buffer is empty, although samples are held.
    pipeline.push(inference("c"))
    pipeline.push(inference("d"))
    pipeline.queue_instructions([request("one"), request("two")])
    assert not pipeline.ready(2)
    # One inserted sample is enough, fewer than job_samples; the other held sample waits for the client.
    pipeline.insert_into_buffer(sample_ids_of(pipeline, "d"))
    batch = pipeline.build_job_data(2)
    assert batch.request.id == "two"
    assert [source_record_id(sample) for sample in batch.samples] == ["d"]
    pipeline.acknowledge(batch.data_id)
    assert [source_record_id(sample) for _, sample in pipeline.held.values()] == ["c"]
    # A trained sample can go into the buffer again.
    pipeline.insert_into_buffer(sample_ids_of(pipeline, "a"))
    assert pipeline.buffer() == sample_ids_of(pipeline, "a")
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
        record2sample=InferenceSamples(),
        sample2job=CountedJobData(context, 6, None),
    )
    assert pipeline.status() == {"ready_samples": 0, "buffer_size": 6, "buffered_requests": 0}
    for index in range(3):
        pipeline.push(inference(f"inf-{index}"))
    # The status counts the stored samples that the pipeline read; advancing it reads them.
    pipeline.advance()
    assert pipeline.status() == {"ready_samples": 3, "buffer_size": 6, "buffered_requests": 0}
    assert not pipeline.ready(4)
    pipeline.push(inference("inf-3"))
    assert pipeline.ready(4)
    assert pipeline.status()["ready_samples"] == 4
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
        pipeline.queue_instructions([request("one")])
        assert pipeline.status().get("buffered_requests") == (1 if takes_instructions else None)
        pipeline.close()


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_processor_uses_shared_data_and_one_batch_assembly_hook(records, build, mode, instructions):
    class TrajectoryJobData(Sample2Job):
        required_request_types = frozenset({RequestType.INFERENCE})
        output_schema = ExampleJobData

        def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> ExampleJobData:
            return ExampleJobData(data_id, values=tuple(source_record_id(task) for task in samples))

    def trajectory_pipeline(context, job_samples=None):
        return TrainingPipeline(
            context,
            record2sample=InferenceSamples(),
            sample2job=TrajectoryJobData(context, None, job_samples),
        )

    backend = RecordingComponent()
    trainer = build(backend, trajectory_pipeline, mode=mode, job_samples=2)
    records.append(inference("first"))
    make_samples(trainer)
    assert trainer.run_once() is None
    records.append(inference("second"))
    if mode == "manual":
        # The job samples are in the manual buffer, and manual still waits for an instruction.
        make_samples(trainer)
        assert trainer.run_once() is None
        trainer.change_buffer(sample_ids_of(trainer.pipeline, "first", "second"), insert=True)
        make_samples(trainer)
        assert trainer.run_once() is None
        file(instructions, "use-the-trajectory")
    make_samples(trainer)
    result = trainer.run_once()
    # The same hook shapes the job data of both modes.
    assert isinstance(backend.batches[0], ExampleJobData)
    assert backend.batches[0].values == ("first", "second")
    assert (backend.batches[0].request is None) == (mode == "auto")
    # The pipeline assigns the ID of each batch, and the hook uses it.
    expected_id = "s:data:1" if mode == "auto" else "s:instruction:use-the-trajectory"
    assert backend.batches[0].data_id == expected_id
    prepared = trainer.prepare_commit(result)
    expected = {"first", "second", "sample:first:0", "sample:second:0"} | (
        {"use-the-trajectory"} if mode == "manual" else set()
    )
    assert prepared.consumed_ids == frozenset(expected)
    trainer.commit(prepared)
    commit_running_job(trainer)
    make_samples(trainer)
    assert trainer.run_once() is None


def test_factory_cannot_silently_change_the_processor_mode(build):
    with pytest.raises(ValueError, match="preserve the requested training_mode"):
        build(RecordingComponent(), lambda ctx, job_samples: TrainingPipeline(replace(ctx, training_mode="auto")))


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_switch_during_reserved_batch_keeps_original_acknowledgement(records, build, mode, instructions):
    trainer = build(RecordingComponent(), mode=mode)
    if mode == "auto":
        records.append(inference("first"))
    else:
        file(instructions, "first")
    make_samples(trainer)
    result = trainer.run_once()
    original = trainer.pending_job_data
    target = "manual" if mode == "auto" else "auto"
    processor = trainer.pipeline
    trainer.set_training_mode(target)
    assert trainer.pipeline is processor
    assert trainer.pipeline.training_mode == target
    assert trainer.pending_job_data is original
    prepared = trainer.prepare_commit(result)
    assert prepared.consumed_ids == ({"first", "sample:first:0"} if mode == "auto" else {"first"})
    trainer.commit(prepared)
    commit_running_job(trainer)
    if target == "manual":
        file(instructions, "next")
    else:
        records.append(inference("next"))
    make_samples(trainer)
    assert trainer.run_once() is not None
    following = trainer.pending_job_data
    assert (following.request is not None) == (target == "manual")


def test_switch_preserves_incomplete_auto_batch_and_unread_manual_instructions(records, build, instructions):
    backend = RecordingComponent()
    trainer = build(backend, mode="auto", job_samples=2)
    records.append(inference("a"))
    make_samples(trainer)
    assert trainer.run_once() is None
    trainer.set_training_mode("manual")
    file(instructions, "change")
    # Change back before the accepted instruction has been read.
    trainer.set_training_mode("auto")
    make_samples(trainer)
    assert trainer.run_once() is None
    records.append(inference("b"))
    make_samples(trainer)
    result = trainer.run_once()
    assert [source_record_id(item) for item in backend.batches[-1].samples] == ["a", "b"]
    trainer.commit(trainer.prepare_commit(result))
    commit_running_job(trainer)
    # The automatic job did not consume the instruction: it stays queued for manual.
    make_samples(trainer)
    assert not instructions.get("change").final
    trainer.set_training_mode("manual")
    make_samples(trainer)
    result = trainer.run_once()
    assert backend.batches[-1].request.id == "change"
    trainer.commit(trainer.prepare_commit(result))
    commit_running_job(trainer)
    trainer.set_training_mode("auto")
    make_samples(trainer)
    assert trainer.run_once() is None


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
                response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=instruction("one"))
                assert response.status == 200
                assert await asyncio.to_thread(entered.wait, 3)
                response = await asyncio.wait_for(
                    client.post("/reef/scenarios/s/update", json={"training_mode": "auto"}), timeout=2
                )
                assert response.status == 200
                assert not release.is_set()
                scenario = dispatcher.get_or_create_scenario("s")
                assert scenario.trainer.pipeline.training_mode == "auto"
                # The instruction's job still runs: the mode change neither waits for it nor stops it.
                assert scenario.trainer.job_running
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
    file_instruction(dispatcher, "one")
    assert _wait(lambda: _committed_skip(dispatcher, "one") == UNCHANGED)
    # The step's job committed twice: its submission and its end.
    assert _wait(lambda: dispatcher.get_or_create_scenario("s").scenario_step == 2)
    dispatcher.close()
    # A new process loads the committed state and the stored mode.
    restarted = open_dispatcher(harness_recipe(tmp_path, method))
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 2 and loaded.trainer.pipeline.training_mode == "manual"
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
    """The metrics of the step that answered the instruction ``text``, newest first; a job's submission names the
    instruction too, and is passed over."""
    for row in dispatcher.get_or_create_scenario("s").releases():
        metrics = row.get("metrics") or {}
        if metrics.get("training_job", {}).get("event") == "submitted":
            continue
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
        for row in catalog(dispatcher.get_or_create_scenario("s"))
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
    file_instruction(dispatcher, "broken scorer")
    assert _wait(lambda: _committed_skip(dispatcher, "broken scorer") == "instruction failed")
    row = _committed_row(dispatcher, "broken scorer")
    assert row["failed_stage"] == "evaluating"
    assert row["error"] == "ValueError: the scorer could not read the rollout"
    assert row["method_notes"] == {"entries": 1}
    assert row["patch"] == changed_rules("marker rules\n")
    assert failed_words(row).startswith("The step failed during its evaluation")


def test_a_queue_of_failing_instructions_drains_and_keeps_the_selected_mode(tmp_path, open_dispatcher):
    """No further record arrives after the two instructions, so the failed jobs wake the worker until both are
    consumed. A failed instruction is not run again, and the scenario keeps the selected mode."""
    calls = []
    method, entered, release = raising_method(calls, poison=None, error="method outage")
    dispatcher = open_dispatcher(harness_recipe(tmp_path, method))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert scenario.store.durable
        dispatcher.set_training_mode("s", "manual")
        file_instruction(dispatcher, "p1")
        assert entered.wait(5)
        file_instruction(dispatcher, "p2")
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "p1") == "instruction failed")
        assert _wait(lambda: _committed_skip(dispatcher, "p2") == "instruction failed")
        assert calls == ["p1", "p2"]
        assert _committed_row(dispatcher, "p2")["error"] == "RuntimeError: method outage"
        current = dispatcher.get_or_create_scenario("s")
        assert current.trainer.pipeline.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
        assert file_instruction(dispatcher, "p3") == "p3"
        assert _wait(lambda: _committed_skip(dispatcher, "p3") == "instruction failed")
    finally:
        release.set()


def test_a_failed_instruction_is_skipped_with_its_error_and_the_queue_moves_on(tmp_path, open_dispatcher):
    calls = []
    method, entered, release = raising_method(calls)
    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        file_instruction(dispatcher, "poison")
        assert entered.wait(5)
        file_instruction(dispatcher, "fine")
        file_instruction(dispatcher, "fine again")
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
        assert _wait(lambda: current.instructions.get("poison").final)
        assert current.trainer.pipeline.training_mode == "manual"
    finally:
        release.set()


def test_a_failed_instruction_is_skipped_alone_and_its_units_wait_for_the_next_instruction(tmp_path, open_dispatcher):
    calls = []
    method, entered, release = raising_method(calls, with_traces=True)
    dispatcher = open_dispatcher(recipe_in("manual", tmp_path, method))
    try:
        dispatcher.get_or_create_scenario("s")
        file_instruction(dispatcher, "poison")
        assert entered.wait(5)
        # The failing exchange arrives while the attempt holds.
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
        # The skip row consumed the instruction and no sample. The samples service reads the exchange's records
        # on its own; when it stored their sample before the skip, the skip consumes those records too.
        assert "poison" in skip.consumed_ids
        assert skip.consumed_ids <= {"poison", "a", "report-a"}
        assert not any(item.startswith(CONSUMED_SAMPLE_PREFIX) for item in skip.consumed_ids)
        assert current.records.get("s", "report-a") is not None
        assert current.trainer.pending_instructions() == 0
        assert current.trainer.instruction_failures() == {}
        # The client puts the exchange's sample into the manual buffer, where it waits for the next instruction.
        insert_samples_of(dispatcher, "a")
    finally:
        release.set()
    dispatcher.close()

    seen = []

    def method_again(workspace, services):
        seen.append(asked(workspace))

    restarted = open_dispatcher(recipe_in("manual", tmp_path, method_again))
    # The failed instruction's job committed twice: its submission and its end.
    assert restarted.get_or_create_scenario("s").scenario_step == 2
    # Recovery keeps the manual buffer and never replays the instruction that the skip row named.
    file_instruction(restarted, "after")
    assert _wait(lambda: _committed_skip(restarted, "after") == UNCHANGED)
    assert seen == [("after", ("a",))]
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 4
    assert "poison" not in {request.id for request in loaded.instructions.queued()}
    assert loaded.trainer.pending_instructions() == 0


def test_a_logless_scenario_ends_a_failed_instructions_job_and_runs_the_next_instruction(tmp_path):
    calls = []
    entered, release = Event(), Event()

    def method(workspace, services):
        text = workspace.request().text
        calls.append(text)
        if text == "poison":
            raise RuntimeError("poison method")
        entered.set()
        release.wait(10)

    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        recipe_in("manual", tmp_path, method),
        factory,
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        assert not scenario.store.durable
        file_instruction(dispatcher, "poison")
        # The failed job's end is the last commit: its submission was the first.
        assert _wait(lambda: scenario.scenario_step == 2)
        assert dispatcher.get_or_create_scenario("s") is scenario
        skipped = _last_committed(scenario)
        assert skipped["skipped"] == "instruction failed"
        assert skipped["training_request"]["id"] == "poison"
        assert skipped["error"] == "RuntimeError: poison method"
        file_instruction(dispatcher, "fine")
        assert entered.wait(5)
        assert calls == ["poison", "fine"]
        release.set()
        assert _wait(lambda: scenario.scenario_step == 4)
        assert _last_committed(scenario).get("skipped") == UNCHANGED
        assert scenario.trainer.pending_instructions() == 0
        assert scenario.trainer.pipeline_status()["buffered_requests"] == 0
        assert scenario.trainer.instruction_failures() == {}
        assert _wait(lambda: scenario.instructions.get("poison").final)
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
        component = self.training_component
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=partial(threshold_pipeline, job_samples=1),
            job_loop=JobLoop(component, scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            samples=samples,
            training_mode=self.training_mode,
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
    for receipt in ("a", "b"):
        dispatcher.accept_record(
            AgentRecord.create(
                scenario="s",
                request_type=RequestType.INFERENCE,
                payload={"response": {"training": {}}},
                agent_record_id=receipt,
            )
        )
        dispatcher.accept_record(report(receipt))
    insert_samples_of(dispatcher, "a")
    file_instruction(dispatcher, "one")
    assert _wait(lambda: _job_events(dispatcher)[:1] == [("one", "submitted")]), _job_events(dispatcher)
    # The commit that submitted the first job emptied the manual buffer.
    assert dispatcher.sample_buffer("s")["samples"] == []
    insert_samples_of(dispatcher, "b")
    file_instruction(dispatcher, "two")
    expected = [
        ("one", "submitted"),
        (None, "failed"),
        ("two", "submitted"),
        (None, "checkpoint"),
        (None, "finished"),
    ]
    assert _wait(lambda: _job_events(dispatcher) == expected), _job_events(dispatcher)
    # Each job trains one instruction with the samples that the client inserted for it.
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


def test_the_job_of_an_instruction_lists_the_instructions_parameters_and_its_times(tmp_path, open_dispatcher):
    component = StubTrainingComponent(tmp_path / "checkpoints")
    dispatcher = open_dispatcher(
        _InstructionJobRecipe(component, runtime=StubInferenceRuntime(), training_mode="manual")
    )
    dispatcher.accept_record(
        AgentRecord.create(
            scenario="s",
            request_type=RequestType.INFERENCE,
            payload={"response": {"training": {}}},
            agent_record_id="a",
        )
    )
    dispatcher.accept_record(report("a"))
    insert_samples_of(dispatcher, "a")
    body = {
        "agent_record_id": "train-1",
        "text": "Prefer tests first",
        "session": "session-1",
        "release_id": "release-1",
        "parameters": {"learning_rate": 0.5},
    }

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            async with asyncio.timeout(10):
                while True:
                    listing = await (await client.get("/reef/scenarios/s/jobs")).json()
                    if listing["jobs"] and listing["jobs"][0]["state"] == "finished":
                        return listing["jobs"][0]
                    await asyncio.sleep(0.05)

    job = asyncio.run(run())
    # The recipe gives no job parameters, so the job's are the instruction's.
    assert job["parameters"] == {"learning_rate": 0.5}
    assert job["instruction"] == {"id": "train-1", "text": "Prefer tests first", "parameters": {"learning_rate": 0.5}}
    assert (job["sample_count"], job["sample_kind"]) == (1, "trajectory")
    assert job["created_at"] <= job["started_at"] <= job["ended_at"]
    assert job["experiment"] is None


def test_a_job_trainer_skips_a_failed_instruction_instead_of_submitting_it_again(tmp_path, records, instructions):
    component = SubmitRaisesForInstructionOne(tmp_path / "checkpoints")
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=partial(threshold_pipeline, job_samples=1),
        job_loop=JobLoop(component, scenario="s"),
        training_mode="manual",
    )
    failure(records, "a")
    make_samples(trainer)
    assert trainer.run_once() is None
    trainer.change_buffer(sample_ids_of(trainer.pipeline, "a"), insert=True)
    file(instructions, "one")
    with pytest.raises(RuntimeError, match="submit failed"):
        make_samples(trainer)
        trainer.run_once()
    # What the dispatcher does when an instruction's step raises.
    assert trainer.fail_pending_instruction("RuntimeError: submit failed")
    make_samples(trainer)
    result = trainer.run_once()
    assert result.metrics["skipped"] == "instruction failed"
    assert result.metrics["error"] == "RuntimeError: submit failed"
    prepared = trainer.prepare_commit(result)
    # The skip consumes the instruction and the records whose sample is stored; it trains no sample.
    assert prepared.consumed_ids == frozenset({"one", "a", "report-a"})
    trainer.commit(prepared)
    # The next instruction's job takes the sample that the skipped instruction left in the manual buffer.
    file(instructions, "two")
    make_samples(trainer)
    assert trainer.run_once() is not None
    assert component.submit_texts == ["one", "two"]
    assert [source_record_id(sample) for sample in component.submitted[-1].data.samples] == ["a"]
    trainer.close()


def test_an_instruction_runs_past_the_step_budget_and_the_failure_streak(tmp_path, records, instructions):
    seen = []

    def method(workspace, services):
        seen.append(asked(workspace)[0])

    recipe = recipe_in("manual", tmp_path, method, max_steps=1)
    head = ReleaseFiles("base", dict(recipe.seed.files))
    trainer = _trainer(recipe, records)
    file(instructions, "one")
    file(instructions, "two")
    rows = []
    for _ in range(2):
        make_samples(trainer)
        result = drive_trainer_job(trainer)
        assert result is not None
        rows.append(result.metrics)
    assert seen == ["one", "two"]
    assert [row["training_request"]["text"] for row in rows] == ["one", "two"]
    assert [row["skipped"] for row in rows] == [UNCHANGED, UNCHANGED]
    make_samples(trainer)
    assert trainer.run_once() is None

    # An automatic step past the budget skips; the two instructions above ran past it.
    backend = job_backend(trainer)
    automatic = backend.prepare_step(JobData("auto", ()), HarnessState().to_json(), 5, head=head)
    assert automatic.metrics["skipped"] == "step budget of 1 exhausted"
    assert seen == ["one", "two"]
    trainer.close()

    (tmp_path / "streak").mkdir()
    trainer = _trainer(recipe_in("manual", tmp_path / "streak", method, max_failure_streak=1), records)
    backend = job_backend(trainer)
    state = HarnessState(failure_streak=1).to_json()
    automatic = backend.prepare_step(JobData("auto", ()), state, 0, head=head)
    assert automatic.metrics["skipped"] == "failure streak of 1 rejections"
    request = TrainingRequest("Follow the request", "session", "r", "request-id")
    answered = backend.prepare_step(JobData("request-id", (), request=request), state, 0, head=head)
    assert answered.metrics["skipped"] == UNCHANGED and seen[-1] == "Follow the request"
    trainer.close()


@pytest.mark.parametrize("processor", [reports_pipeline, records_pipeline])
def test_manual_traffic_is_available_to_auto_without_a_replay(records, build, processor):
    backend = RecordingComponent()
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
    make_samples(trainer)
    assert trainer.run_once() is None
    original = trainer.pipeline
    offset = trainer.data_offset
    trainer.set_training_mode("auto")
    make_samples(trainer)
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
def test_manual_runs_a_harness_instruction_alone_when_no_units_are_held(records, build, processor, instructions):
    backend = RecordingComponent()
    trainer = build(backend, processor, job_samples=2)
    make_samples(trainer)
    assert trainer.run_once() is None
    file(instructions, "alone")
    make_samples(trainer)
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
    commit_running_job(trainer)
    make_samples(trainer)
    assert instructions.get("alone").final
    make_samples(trainer)
    assert trainer.run_once() is None
    assert trainer.pipeline.training_mode == "manual"


def test_manual_runs_two_queued_instructions_oldest_first_one_per_step():
    processor = reports_pipeline(PipelineContext("s", training_mode="manual"))
    processor.queue_instructions([request("first"), request("second")])
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


def test_manual_gives_each_instruction_the_failures_that_the_client_inserted(records, build, instructions):
    backend = RecordingComponent()
    trainer = build(backend, reports_pipeline, job_samples=2)

    def step():
        make_samples(trainer)
        result = trainer.run_once()
        if result is None:
            return None
        trainer.commit(trainer.prepare_commit(result))
        commit_running_job(trainer)
        batch = backend.batches[-1]
        return batch.request.id, [source_record_id(sample) for sample in batch.samples]

    def insert(*receipts):
        trainer.change_buffer(sample_ids_of(trainer.pipeline, *receipts), insert=True)

    for receipt in ("a", "b", "c"):
        failure(records, receipt)
    # The failures start no job: they wait for an instruction.
    assert step() is None
    insert("a", "c")
    file(instructions, "one")
    assert step() == ("one", ["a", "c"])
    failure(records, "d")
    assert step() is None
    # The client inserts a trained failure again; the instruction trains more than job_samples, in store order.
    insert("d", "b", "a")
    file(instructions, "two")
    assert step() == ("two", ["a", "b", "d"])
    failure(records, "e")
    assert step() is None


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_every_mode_keeps_the_samples_beyond_the_buffer_for_later_jobs(caplog, mode):
    # A trace job takes one sample by default, and the default buffer holds the data of one job.
    processor = reports_pipeline(PipelineContext("s", training_mode=mode))
    assert processor.buffer_size == 1

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
    # Every ready unit goes to the sample store, which consumes its records; nothing is released untrained.
    assert processor.record2sample.ready_units == {}
    retention = processor.releasable_record_ids()
    assert {f"inf-{i}" for i in range(20)} | {f"rep-{i}" for i in range(20)} <= retention
    assert not caplog.records
    if mode == "manual":
        # The client chooses the oldest sample for each instruction.
        processor.insert_into_buffer(sample_ids_of(processor, "inf-0"))
        processor.queue_instructions([request("do-it")])
    reserved = processor.build_job_data(1)
    fail(20)
    fail(21)
    # The handed-out job data keeps the oldest sample; the samples beyond the buffer wait for later jobs.
    assert [source_record_id(sample) for sample in reserved.samples] == ["inf-0"]
    instruction_ids = {"do-it"} if mode == "manual" else set()
    assert processor.acknowledge(reserved.data_id) == frozenset({"sample:rep-0:0", *instruction_ids})
    if mode == "manual":
        processor.insert_into_buffer(sample_ids_of(processor, "inf-1"))
        processor.queue_instructions([request("do-it"), request("again")])
    following = processor.build_job_data(1)
    assert [source_record_id(sample) for sample in following.samples] == ["inf-1"]
    assert len(processor.held) == 21
    processor.close()


def test_manual_collects_the_failures_each_instruction_step_carries(tmp_path, open_dispatcher):
    calls = []

    def method(workspace, services):
        calls.append(asked(workspace))

    recipe = recipe_in("manual", tmp_path, method, job_samples=2, evaluation={"collect_failures": True})
    dispatcher = open_dispatcher(recipe)

    def evaluation_rows():
        """Each step's traces, its evaluation's task count (the seed task and the collected ones) and whether it
        answered an instruction, newest first."""
        rows = catalog(dispatcher.get_or_create_scenario("s"))
        return [
            (metrics["traces"], metrics["evaluation_task_count"], "training_request" in metrics)
            for row in rows
            if (metrics := row.get("metrics"))
        ]

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            # The client inserts one failure, fewer than the batch size; the instruction step takes it and collects it.
            dispatcher.accept_record(inference("fails-1"))
            dispatcher.accept_record(report("fails-1"))
            await asyncio.to_thread(insert_samples_of, dispatcher, "fails-1")
            body = {"agent_record_id": "ask-1", "text": "do it", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 1)
            assert calls == [("do it", ("fails-1",))]
            assert evaluation_rows() == [(1, 2, True)]
            # The next instruction step takes the two inserted failures and grows the evaluation on top of the
            # collected one.
            for receipt in ("fails-2", "fails-3"):
                dispatcher.accept_record(inference(receipt))
                dispatcher.accept_record(report(receipt))
            await asyncio.to_thread(insert_samples_of, dispatcher, "fails-2", "fails-3")
            body = {"agent_record_id": "ask-2", "text": "again", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(evaluation_rows()) == 2)
            # Newest first: the second instruction step's row, then the first one's.
            assert evaluation_rows() == [(2, 4, True), (1, 2, True)]
            assert calls[1] == ("again", ("fails-2", "fails-3"))

    asyncio.run(run())
