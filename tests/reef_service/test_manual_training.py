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
from tests.reef_service.test_harness_proposals import _dispatcher, _recipe
from tests.support._threshold_processor import threshold_pipeline
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent
from tests.train.test_harness_evolve_processor import harness_stages, records_pipeline, reports_pipeline
from tests.train.test_reef_trainer_contracts import ExampleBackend, ExampleJobData

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.pipeline_context import PipelineContext
from reef.recipe import RecipeConfigError
from reef.recipe.base import WeightTrainingRecipe
from reef.record2record import InferenceRecord2Record
from reef.sample2job import Sample2Job
from reef.service.app import create_app
from reef.storage.samples import CONSUMED_SAMPLE_PREFIX
from reef.storage.scenario import ScenarioSettings
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import PreparedStep
from reef.train.cordis_backend.processor import InferenceSamples
from reef.train.jobs import JobLoop
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


def propose_nothing(nodes, samples, models, *, requests=()):
    """A proposer that reads instructions and proposes nothing."""
    return


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
    """Open ``_dispatcher`` over the test's directory; each dispatcher closes at teardown."""
    opened = []

    def open_one(recipe):
        dispatcher = _dispatcher(tmp_path, recipe)
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
    # The job data consumes the samples it takes; the samples of the other held units wait for an instruction.
    assert len({item for item in prepared.consumed_ids if item.startswith(CONSUMED_SAMPLE_PREFIX)}) == len(taken)
    assert prepared.metrics["training_request"]["text"] == "request-1"
    trainer.commit(prepared)
    assert len(trainer.pipeline.held) == len(held) - len(taken)
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

    def propose(nodes, samples, models, *, requests=()):
        seen.append((requests[0], samples))
        called.set()

    dispatcher = open_dispatcher(
        replace(_recipe(tmp_path, propose), training_mode="manual", pipeline_stages=harness_stages())
    )
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
            assert seen[0][0]["text"] == "Prefer tests first"
            assert seen[0][1] == ()
            assert seen[0][0]["id"] == "request-1"
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
    dispatcher = open_dispatcher(_recipe(tmp_path, lambda n, s, m: None))
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
    configured = replace(context, manual_enabled=True)
    assert configured.training_mode == "manual"
    processor = reports_pipeline(configured)
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
    # A context that takes no instructions removes manual from a stage that supports it.
    with pytest.raises(NotImplementedError, match="does not implement training_mode='manual'"):
        TrainingPipeline(replace(context, manual_enabled=False))


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
    assert pipeline.acknowledge(batch.data_id) == frozenset({"sample:a:0", "sample:b:0", "one"})
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
    expected = {"first", "second", "sample:first:0", "sample:second:0"} | (
        {"use-the-trajectory"} if mode == "manual" else set()
    )
    assert prepared.consumed_ids == frozenset(expected)
    trainer.commit(prepared)
    assert trainer.run_once() is None


def test_factory_cannot_silently_change_the_processor_mode(build):
    with pytest.raises(ValueError, match="preserve the requested training_mode"):
        build(CaptureBackend(), lambda ctx: TrainingPipeline(replace(ctx, training_mode="auto")))


def test_manual_instruction_cannot_be_consumed_by_recheck_or_inbox_proposal(tmp_path, records):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(requests)

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", recheck_every=1)
    trainer = recipe.build("s", records)
    backend = trainer.candidate_backend
    state = dict(backend.initial_state())
    state["recheck_entries"] = state["entries"]
    backend.proposals.submit(
        "pending-proposal",
        {
            "mutations": [{"op": "create", "id": "rules", "options": {"name": "rules", "config": {"text": "marker"}}}],
            "session": "unrelated",
            "release_id": "r",
            "reason": "unrelated",
        },
    )
    batch = JobData(
        "manual-request",
        (),
        request=TrainingRequest("Follow the request", "session", "r", "request-id"),
    )
    prepared = backend.prepare_step(batch, state, 0)
    assert prepared.outcome == "skip"
    assert seen[0][0]["id"] == "request-id"
    assert (recipe.proposals_path("s") / "pending-proposal.json").is_file()
    assert "recheck" not in prepared.metrics
    trainer.close()


def test_harness_manual_mode_requires_explicit_requests_keyword(tmp_path, records):
    recipe = replace(_recipe(tmp_path, lambda n, s, m, **kwargs: None), training_mode="manual")
    with pytest.raises(RecipeConfigError, match="requests"):
        recipe.build("s", records)


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
    assert prepared.consumed_ids == ({"first", "sample:first:0"} if mode == "auto" else {"first"})
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


@pytest.mark.parametrize("reads_requests", [False, True])
def test_http_training_mode_updates_only_existing_supported_processors(tmp_path, open_dispatcher, reads_requests):
    dispatcher = open_dispatcher(_recipe(tmp_path, propose_nothing if reads_requests else lambda n, s, m: None))
    scenario = dispatcher.get_or_create_scenario("s")
    processor = scenario.trainer.pipeline

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            url = "/reef/scenarios/s/update"
            response = await client.post(url, json={"training_mode": "manual"})
            assert response.status == (200 if reads_requests else 501)
            if reads_requests:
                assert await response.json() == {"scenario": "s", "training_mode": "manual"}
            assert scenario.trainer.pipeline.training_mode == ("manual" if reads_requests else "auto")
            assert scenario.trainer.pipeline is processor
            status = await (await client.get("/reef/status")).json()
            assert status["scenarios"]["s"]["training_mode"] == scenario.trainer.pipeline.training_mode
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


def test_http_mode_change_does_not_wait_for_running_proposer(tmp_path, open_dispatcher):
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        entered.set()
        assert release.wait(5)

    dispatcher = open_dispatcher(replace(_recipe(tmp_path, propose), training_mode="manual"))
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
    dispatcher = open_dispatcher(_recipe(tmp_path, propose_nothing))
    scenario = dispatcher.get_or_create_scenario("s")
    assert dispatcher.set_training_mode("s", "manual") == {"scenario": "s", "training_mode": "manual"}
    assert scenario.trainer.pipeline.training_mode == "manual"
    reloaded = dispatcher.registry.reload("s")
    assert reloaded is not scenario and reloaded.trainer.pipeline.training_mode == "manual"
    assert dispatcher.set_training_mode("s", "auto")["training_mode"] == "auto"
    assert dispatcher.registry.reload("s").trainer.pipeline.training_mode == "auto"
    dispatcher.set_training_mode("s", "manual")
    dispatcher.accept_record(instruction("one"))
    assert _wait(lambda: _committed_skip(dispatcher, "one") == "no proposal")
    assert dispatcher.get_or_create_scenario("s").scenario_step == 1
    dispatcher.close()
    # A new process loads the committed state and the stored mode.
    restarted = open_dispatcher(_recipe(tmp_path, propose_nothing))
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 1 and loaded.trainer.pipeline.training_mode == "manual"
    restarted.close()
    # A recipe that no longer takes instructions keeps its default, so a person can still select a mode.
    without_requests = open_dispatcher(_recipe(tmp_path, lambda n, s, m: None))
    with caplog.at_level(logging.WARNING, logger="reef.scenario.registry"):
        assert without_requests.get_or_create_scenario("s").trainer.pipeline.training_mode == "auto"
    assert "keeps its default training mode" in caplog.text


def test_a_stored_hybrid_mode_loads_the_scenario_with_the_recipe_default(tmp_path, open_dispatcher, caplog):
    dispatcher = open_dispatcher(_recipe(tmp_path, propose_nothing))
    dispatcher.get_or_create_scenario("s")
    dispatcher.close()
    # The setting of a scenario that selected hybrid before Reef removed the mode.
    with closing(SQLiteScenarioStorage(tmp_path / "agent-record")) as storage:
        storage.write_settings("s", ScenarioSettings(training_mode="hybrid"))
    restarted = open_dispatcher(replace(_recipe(tmp_path, propose_nothing), training_mode="manual"))
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


def _raising_proposer(calls, *, poison="poison", error="poison proposer", with_samples=False):
    """A proposer that raises on the poison instruction (on every call when ``poison`` is None) and holds its first
    attempt open until released. It records each instruction text, with the sample sources if ``with_samples``."""
    entered, release = Event(), Event()

    def propose(nodes, samples, models, *, requests=()):
        text = requests[0]["text"] if requests else None
        calls.append((text, tuple(source_record_id(sample) for sample in samples)) if with_samples else text)
        if text == poison or poison is None:
            if len(calls) == 1:
                entered.set()
                release.wait(10)
            raise RuntimeError(error)
        return

    return propose, entered, release


def test_a_step_whose_evaluation_raised_keeps_its_proposal_and_says_where_it_failed(tmp_path, open_dispatcher):
    """The candidate reached its evaluation and the scorer raised: the skip row says the step failed during its
    evaluation, and keeps the proposed change and the method's notes, so the pages show what was tried."""
    from tests.support._strategies import resolve_rollout_scorer

    from reef.harness.tree.mutations import Mutation
    from reef.service.release_page import failed_words
    from reef.train.cordis_backend.strategies import StepProposal

    marker = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})

    def propose(nodes, samples, models, *, requests=()):
        return StepProposal((marker,), {"design": "one rules entry"})

    def score(task, result):
        raise ValueError("the scorer could not read the rollout")

    recipe = replace(_recipe(tmp_path, propose), score_rollout=resolve_rollout_scorer(score), training_mode="manual")
    dispatcher = open_dispatcher(recipe)
    dispatcher.get_or_create_scenario("s")
    dispatcher.accept_record(instruction("broken scorer"))
    assert _wait(lambda: _committed_skip(dispatcher, "broken scorer") == "instruction failed")
    row = _committed_row(dispatcher, "broken scorer")
    assert row["failed_stage"] == "evaluating"
    assert row["error"] == "ValueError: the scorer could not read the rollout"
    assert row["proposal_notes"]["design"] == "one rules entry"
    assert [mutation["id"] for mutation in row["mutations"]] == ["r1"]
    assert failed_words(row).startswith("The step failed during its evaluation")


def test_a_queue_of_failing_instructions_drains_and_keeps_the_selected_mode(tmp_path, open_dispatcher):
    """No further record arrives after the two instructions, so the failures wake the worker until both are consumed.
    A failed instruction is not run again, and the reload after each failure keeps the selected mode."""
    calls = []
    propose, entered, release = _raising_proposer(calls, poison=None, error="proposer outage")
    dispatcher = open_dispatcher(_recipe(tmp_path, propose))
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
        assert _committed_row(dispatcher, "p2")["error"] == "RuntimeError: proposer outage"
        current = dispatcher.get_or_create_scenario("s")
        assert current is not scenario and current.trainer.pipeline.training_mode == "manual"
        assert current.trainer.pending_instructions() == 0
        assert dispatcher.accept_record(instruction("p3")).agent_record_id == "p3"
        assert _wait(lambda: _committed_skip(dispatcher, "p3") == "instruction failed")
    finally:
        release.set()


def test_a_failed_instruction_is_skipped_with_its_error_and_the_queue_moves_on(tmp_path, open_dispatcher):
    calls = []
    propose, entered, release = _raising_proposer(calls)
    dispatcher = open_dispatcher(replace(_recipe(tmp_path, propose), training_mode="manual"))
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
        assert current.trainer.pipeline_status()["buffered_requests"] == 0
        assert current.trainer.instruction_failures() == {}
        assert current.records.get("s", "poison") is not None
        assert current.trainer.pipeline.training_mode == "manual"
    finally:
        release.set()


def test_a_failed_instruction_is_skipped_alone_and_its_units_wait_for_the_next_instruction(tmp_path, open_dispatcher):
    calls = []
    propose, entered, release = _raising_proposer(calls, with_samples=True)
    dispatcher = open_dispatcher(replace(_recipe(tmp_path, propose), training_mode="manual"))
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(instruction("poison"))
        assert entered.wait(5)
        # The failing exchange arrives while the attempt holds, so the skip row's job data carries it as its sample.
        dispatcher.accept_record(inference("a"))
        dispatcher.accept_record(report("a"))
        release.set()
        assert _wait(lambda: _committed_skip(dispatcher, "poison") == "instruction failed")
        # The skip row took no proposer call.
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

    def propose_again(nodes, samples, models, *, requests=()):
        seen.append((requests[0]["text"] if requests else None, tuple(source_record_id(s) for s in samples)))

    restarted = open_dispatcher(replace(_recipe(tmp_path, propose_again), training_mode="manual"))
    assert restarted.get_or_create_scenario("s").scenario_step == 1
    # Recovery replays the unit as unconsumed and never the instruction that the skip row named.
    restarted.accept_record(instruction("after"))
    assert _wait(lambda: _committed_skip(restarted, "after") == "no proposal")
    assert seen == [("after", ("a",))]
    loaded = restarted.get_or_create_scenario("s")
    assert loaded.scenario_step == 2
    assert loaded.records.get("s", "poison") is not None
    assert loaded.trainer.pending_instructions() == 0


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
        assert skipped["error"] == "RuntimeError: poison proposer"
        assert calls == ["poison", "fine"]
        release.set()
        assert _wait(lambda: scenario.scenario_step == 2)
        assert _last_committed(scenario).get("skipped") == "no proposal"
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
        self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None, samples=None
    ):
        component = self.training_component
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=threshold_pipeline,
            job_loop=JobLoop(component, scenario=scenario),
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
        job_loop=JobLoop(component, scenario="s"),
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
    # The skip consumes the instruction and the records whose sample is stored; it trains no sample.
    assert prepared.consumed_ids == frozenset({"one", "a", "report-a"})
    trainer.commit(prepared)
    # The next instruction's job takes the sample that the skipped instruction left held.
    records.append(instruction("two"))
    assert trainer.run_once() is not None
    assert component.submit_texts == ["one", "two"]
    assert [source_record_id(sample) for sample in component.submitted[-1].data.samples] == ["a"]
    trainer.close()


def test_an_instruction_runs_past_the_step_budget_and_the_failure_streak(tmp_path, records):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(requests[0]["text"] if requests else None)

    trainer = replace(_recipe(tmp_path, propose), training_mode="manual", max_steps=1).build("s", records)
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
    assert [row["skipped"] for row in rows] == ["no proposal", "no proposal"]
    assert [row["steps"] for row in rows] == [1, 2]
    assert trainer.run_once(2) is None

    backend = trainer.candidate_backend
    state = {**backend.initial_state(), "steps": 5}
    automatic = backend.prepare_step(JobData("auto", ()), state, 5)
    assert automatic.metrics["skipped"] == "step budget of 1 exhausted"
    assert seen == ["one", "two"]
    trainer.close()

    streak = replace(_recipe(tmp_path, propose), training_mode="manual", max_failure_streak=1)
    trainer = streak.build("s", records)
    backend = trainer.candidate_backend
    state = {**backend.initial_state(), "failure_streak": 1}
    automatic = backend.prepare_step(JobData("auto", ()), state, 0)
    assert automatic.metrics["skipped"] == "failure streak breaker open after 1 consecutive rejections"
    request = TrainingRequest("Follow the request", "session", "r", "request-id")
    asked = backend.prepare_step(JobData("request-id", (), request=request), state, 0)
    assert asked.metrics["skipped"] == "no proposal" and seen[-1] == "Follow the request"
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
    recipe = _recipe(tmp_path, propose_nothing)
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


def test_the_proposal_route_refuses_in_manual_mode_and_admits_again_in_auto(tmp_path, open_dispatcher):
    dispatcher = open_dispatcher(replace(_recipe(tmp_path, propose_nothing), training_mode="manual"))
    proposal = {
        "mutations": [{"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}],
        "reason": "because",
        "session": "3f1c2a9d0b7e",
        "release_id": "rel-0",
    }
    inbox = tmp_path / "inbox" / "s"

    async def run():
        assert dispatcher.get_or_create_scenario("s").trainer.pipeline.training_mode == "manual"
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            headers = {"x-reef-scenario": "s"}
            response = await client.post("/reef/harness/proposals", headers=headers, json=proposal)
            assert response.status == 200
            answer = await response.json()
            assert answer["admitted"] is False
            assert answer["reason"] == "manual mode takes instructions only"
            assert not inbox.exists() or not list(inbox.glob("*.json"))
            response = await client.post("/reef/scenarios/s/update", json={"training_mode": "auto"})
            assert response.status == 200
            response = await client.post("/reef/harness/proposals", headers=headers, json=proposal)
            answer = await response.json()
            assert answer["admitted"] is True and answer["reason"] is None
            assert [path.name for path in inbox.glob("*.json")] == [f"{answer['proposal_id']}.json"]

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_every_mode_keeps_the_samples_beyond_the_buffer_for_later_jobs(caplog, mode):
    # The default buffer holds the data of one job of one sample.
    processor = reports_pipeline(PipelineContext("s", training_mode=mode, default_job_samples=1))
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
    assert len(processor.record2record.pool) == 0
    retention = processor.releasable_record_ids()
    assert {f"inf-{i}" for i in range(20)} | {f"rep-{i}" for i in range(20)} <= retention
    assert not caplog.records
    if mode == "manual":
        processor.push(instruction("do-it"))
    reserved = processor.build_job_data(1)
    fail(20)
    fail(21)
    # The handed-out job data keeps the oldest sample; the samples beyond the buffer wait for later jobs.
    assert [source_record_id(sample) for sample in reserved.samples] == ["inf-0"]
    instruction_ids = {"do-it"} if mode == "manual" else set()
    assert processor.acknowledge(reserved.data_id) == frozenset({"sample:rep-0:0", *instruction_ids})
    if mode == "manual":
        processor.push(instruction("again"))
    following = processor.build_job_data(1)
    assert [source_record_id(sample) for sample in following.samples] == ["inf-1"]
    assert len(processor.held) == 21
    processor.close()


def test_manual_promotes_the_failures_each_instruction_step_carries(tmp_path, open_dispatcher):
    calls = []

    def propose(nodes, samples, models, *, requests=()):
        calls.append(
            (
                tuple(request["text"] for request in requests),
                tuple(source_record_id(sample) for sample in samples),
            )
        )

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", promote_failures=True, batch_size=2)
    dispatcher = open_dispatcher(recipe)

    def gate_rows():
        rows = dispatcher.get_or_create_scenario("s").releases()
        return [
            (
                metrics["traces"],
                metrics["promoted_tasks"],
                metrics["evaluation_task_count"],
                "training_request" in metrics,
            )
            for row in rows
            if (metrics := row.get("metrics"))
        ]

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            # One failure held below the batch size; the instruction step takes it and promotes it.
            dispatcher.accept_record(inference("fails-1"))
            dispatcher.accept_record(report("fails-1"))
            body = {"agent_record_id": "ask-1", "text": "do it", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(gate_rows()) == 1)
            assert calls == [(("do it",), ("fails-1",))]
            assert gate_rows() == [(1, 1, 2, True)]
            # The next instruction step takes two failures and grows the evaluation tasks on top of the promoted one.
            for receipt in ("fails-2", "fails-3"):
                dispatcher.accept_record(inference(receipt))
                dispatcher.accept_record(report(receipt))
            body = {"agent_record_id": "ask-2", "text": "again", "session": "session-1", "release_id": "release-1"}
            response = await client.post("/reef/train", headers={"x-reef-scenario": "s"}, json=body)
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(_wait, lambda: len(gate_rows()) == 2)
            # Newest first: the second instruction step's row, then the first one's.
            assert gate_rows() == [(2, 3, 4, True), (1, 1, 2, True)]
            assert calls[1] == (("again",), ("fails-2", "fails-3"))

    asyncio.run(run())
