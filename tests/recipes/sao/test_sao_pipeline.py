"""Reef-side SAO pipeline: recipe, processor, backend preparation, and a full
accept -> train -> commit loop against a stub training component.

Everything here is torch/ray free so it runs in the minimal CI gate. The Slime
backend transcription is covered separately by the container-only bridge and
parity tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import yaml
from tests.support._trajectories import policy_trajectory
from tests.support.chat_templates import CharTemplate
from tests.support.conversations import conversation
from tests.support.runtime_stubs import (
    STUB_ROOT,
    StubInferenceRuntime,
    StubTrainingComponent,
    job_sized_stages,
    runtime_bindings,
)
from tests.support.samples_service import make_samples

from recipes.sao import SAORecipe
from recipes.sao.processor import SAOJobData, SAOSamples
from reef.artifact import CachedRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.artifact.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import rendered_job_data
from reef.core.reports import ReportValidationError
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.dispatcher import Dispatcher
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.record2sample.assembly import SampleAssemblySettings
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.service.deploy.config_utils import interpolate_config_values
from reef.storage.commits import CommitRecord
from reef.storage.samples import TrainerSamples
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import PipelineContext, Trainer
from reef.train.algos.schedule import StepScheduling
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData

EXAMPLES = Path(__file__).resolve().parents[3] / "recipes" / "sao" / "examples"


def sao_processor(context: PipelineContext, job_samples: int | None = None) -> TrainingPipeline:
    return TrainingPipeline(
        context,
        record2sample=SAOSamples(context, SampleAssemblySettings()),
        sample2job=SAOJobData(context, None, job_samples),
    )


def single_rollout_steps() -> PipelineConfig:
    """SAO's default stages with jobs of one sample; with a runtime batch size of 1 each rollout is one optimizer step."""
    return job_sized_stages(SAORecipe.pipeline_spec())


def _sao_inference(agent_record_id: str, *, artifact_id: str | None = "artifact-3") -> AgentRecord:
    """A served call of one turn, without training tensors, as the inference service records it."""
    record = conversation(agent_record_id, 1)
    payload: dict[str, Any] = dict(record.payload)
    # The inference service records the artifact that answered.
    if artifact_id:
        payload["artifact_id"] = artifact_id
    artifact_ref = ArtifactRef(content_id="math", release_id="release-1", parent_release_id=None)
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=artifact_ref,
        created_at=record.created_at,
    )


def _sao_job(**training: Any) -> JobData:
    """Job data of one SAO sample with the tensors that the training component trains; ``training`` replaces some."""
    sample = policy_trajectory("i1", (5, 1, 2, 3), (1, 1, 1), (-0.1, -0.2, -0.3), 1.0, action_mask=(1, 1, 1))
    return JobData("math:data:1", (sample.with_training(**training),))


def _sao_report(agent_record_id: str, reference: str, score: float) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": [reference]},
        agent_record_id=agent_record_id,
        references=(reference,),
    )


# --- recipe ---------------------------------------------------------------


@pytest.mark.unit
def test_sao_recipe_resolves_by_dotted_reference() -> None:
    reference = "recipes.sao.recipe:SAORecipe"
    assert recipe_class_for(reference) is SAORecipe
    assert recipe_class_for("sao") is None

    recipe = build_recipe(reference, {}, **runtime_bindings())

    assert isinstance(recipe, SAORecipe)
    assert recipe.name == "sao"


@pytest.mark.unit
def test_sao_recipe_defaults_are_reef_side_only() -> None:
    recipe = SAORecipe(**runtime_bindings())

    # Objective defaults live with the Slime implementation. The Reef recipe
    # owns only its stages.
    stages = recipe.stages()
    assert (stages.record2sample.plugin_class, stages.sample2job.plugin_class) == (SAOSamples, SAOJobData)
    # Without a configured buffer_size the buffer holds the data of one job.
    sized = SAORecipe(
        **runtime_bindings(), pipeline_stages=job_sized_stages(SAORecipe.pipeline_spec(), job_samples=128)
    )
    assert sized.build_pipeline(PipelineContext("math")).buffer_size == 128


@pytest.mark.unit
def test_sao_recipe_reads_reef_side_config() -> None:
    recipe = SAORecipe.from_environment(
        {},
        config={
            "pipeline": {"sample2job": {"config": {"job_samples": 2, "buffer_size": 16}}},
        },
        **runtime_bindings(),
    )

    sample2job = recipe.build_pipeline(PipelineContext("math")).sample2job
    assert (sample2job.job_samples, sample2job.buffer_size) == (2, 16)


@pytest.mark.unit
@pytest.mark.parametrize("example", ["imo_answerbench/serve.yaml", "ceobench/serve.yaml"])
def test_example_pipeline_section_builds_the_sao_stages(example: str) -> None:
    deployment = yaml.safe_load((EXAMPLES / example).read_text(encoding="utf-8"))
    pipeline_section = interpolate_config_values(deployment, deployment["pipeline"])
    recipe = SAORecipe.from_environment({}, config={"pipeline": pipeline_section}, **runtime_bindings())
    # One job is one optimizer step: job_samples names Slime's global batch size.
    global_batch_size = deployment["training"]["config"]["global_batch_size"]
    assert deployment["training"]["options"]["global-batch-size"] == "${training.config.global_batch_size}"
    assert deployment["pipeline"]["sample2job"]["config"]["job_samples"] == "${training.config.global_batch_size}"

    pipeline = recipe.build_pipeline(PipelineContext("math"))
    assert pipeline.sample2job.job_samples == global_batch_size

    assert isinstance(pipeline.record2sample, SAOSamples)
    assert pipeline.record2sample.exclusive_sources, "every rollout owns its inference records"
    assert isinstance(pipeline.sample2job, SAOJobData)
    assert pipeline.buffer_size is not None and pipeline.buffer_size >= global_batch_size


@pytest.mark.unit
@pytest.mark.parametrize("key", ["batch_size", "batch-size"])
def test_sao_recipe_refuses_batch_size_in_the_recipe_config(key: str) -> None:
    with pytest.raises(RecipeConfigError, match="does not consume"):
        SAORecipe.from_environment({}, config={"data": {key: 2}}, **runtime_bindings())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"sample2job": {"config": {"batch_size": 8}}}, "SAOJobData: unknown config fields: batch_size"),
        ({"sample2job": {"config": {"buffer_size": 0}}}, "buffer_size must be None or a positive int"),
        ({"sample2job": {"config": {"groups_per_step": 2}}}, "groups_per_step"),
        ({"record2sample": {"config": {"batch_size": 2}}}, "SAOSamples: unknown config fields: batch_size"),
        ({"record2sample": {"config": {"max_staleness": -1}}}, "max_staleness must be a non-negative int"),
    ],
)
def test_sao_recipe_refuses_a_bad_stage_config_when_the_pipeline_builds(
    pipeline: dict[str, Any], message: str
) -> None:
    recipe = SAORecipe.from_environment({}, config={"pipeline": pipeline}, **runtime_bindings())
    with pytest.raises(RecipeConfigError, match=message):
        recipe.build_pipeline(PipelineContext("math"))


@pytest.mark.unit
def test_sao_recipe_rejects_backend_objective_config() -> None:
    from reef.recipe.errors import RecipeConfigError

    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        SAORecipe.from_environment(
            {},
            config={"optimization": {"eps_low": 0.8}},
            **runtime_bindings(),
        )


# --- processor -------------------------------------------------------------


@pytest.mark.unit
def test_processor_emits_one_independently_scheduled_sample_per_rollout() -> None:
    processor = sao_processor(PipelineContext("math"))
    processor.push(_sao_inference("i1"))
    processor.push(_sao_report("r1", "i1", 0.75))

    batch = processor.build_job_data(1)

    assert isinstance(batch, JobData)
    assert len(batch.samples) == 1
    sample = batch.samples[0]
    assert source_record_id(sample) == "i1"
    assert trajectory_reward(sample) == pytest.approx(0.75)
    # The sample is the rollout's ATIF trajectory: the training component renders its tokens and masks.
    assert not {"tokens", "loss_mask", "action_mask"} & set(sample.training)
    # The producing artifact and timestamp support the staleness check and queue-age reporting.
    assert sample.artifact_id == "artifact-3"
    assert sample.training["rollout_created_at"] == 123.0


@pytest.mark.unit
def test_processor_reads_the_producing_artifact_from_the_payload_of_a_record_without_a_release() -> None:
    # The producing artifact lives in payload["artifact_id"], set by the
    # inference service. The sample must still record it, or the staleness
    # check finds the sample stale and no SAO step ever runs.
    inference = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={**conversation("i1", 1).payload, "artifact_id": "artifact-7"},
        agent_record_id="i1",
        artifact_ref=None,
    )
    processor = sao_processor(PipelineContext("math"))
    processor.push(inference)
    processor.push(_sao_report("r1", "i1", 0.5))

    batch = processor.build_job_data(1)

    assert batch.samples[0].artifact_id == "artifact-7"


@pytest.mark.unit
def test_backend_keeps_an_explicit_observation_boundary_and_defaults_the_action_mask_to_the_loss_mask() -> None:
    # Response tokens: action, observation, action. loss_mask trains only the action tokens.
    explicit = prepare_slime_job(
        _sao_job(loss_mask=[1, 0, 1], action_mask=[1, 0, 1]), TrainingMethod("sao"), {}, StepScheduling(), step_size=1
    )
    assert explicit.payload is not None
    (row,) = explicit.payload["samples"]
    assert (row[2], row[5]) == ([1, 0, 1], [1, 0, 1])
    # A rendered sample has no action mask: its actions are the tokens that it trains.
    rendered = prepare_slime_job(
        _sao_job(loss_mask=[1, 0, 1], action_mask=[]), TrainingMethod("sao"), {}, StepScheduling(), step_size=1
    )
    assert rendered.payload is not None
    (row,) = rendered.payload["samples"]
    assert row[5] == [1, 0, 1]
    to_slime_rollout_data(rendered.payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    "training",
    [
        # loss_mask trains index 1, but the action mask marks it an observation. Skip-observation GAE would give it
        # zero advantage: reject the rollout rather than silently waste gradient on it.
        pytest.param({"action_mask": [1, 0, 1]}, id="trains-a-non-action-token"),
        pytest.param({"rollout_log_probs": [-0.1, -0.2]}, id="log-prob-length-mismatch"),
        pytest.param({"loss_mask": [0, 0, 0]}, id="trains-nothing"),
    ],
)
def test_backend_rejects_malformed_rollout_tensors(training: dict[str, Any]) -> None:
    prepared = prepare_slime_job(_sao_job(**training), TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    with pytest.raises(ValueError):
        to_slime_rollout_data(prepared.payload)


@pytest.mark.unit
def test_processor_rejects_non_finite_report_scores() -> None:
    processor = sao_processor(PipelineContext("math"))
    processor.push(_sao_inference("i1"))
    with pytest.raises(ReportValidationError, match="finite"):
        processor.push(_sao_report("r1", "i1", float("nan")))
    assert not processor.ready(1)
    assert processor.releasable_record_ids().isdisjoint({"i1"})


@pytest.mark.unit
@pytest.mark.parametrize("dead_report_first", [True, False])
def test_invalid_eligibility_report_does_not_block_valid_feedback(dead_report_first: bool) -> None:
    # Invalid eligibility flags never claim or consume the source record;
    # a valid report is processed independently of arrival order.
    processor = sao_processor(PipelineContext("math"))
    processor.push(_sao_inference("i1"))
    dead = AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": ["i1"], "metadata": {"training": {"eligible": False}}},
        agent_record_id="dead",
        references=("i1",),
    )
    retry = _sao_report("retry", "i1", 0.5)
    for item in (dead, retry) if dead_report_first else (retry, dead):
        if item is dead:
            with pytest.raises(ReportValidationError, match="eligible"):
                processor.push(item)
        else:
            processor.push(item)
        processor.releasable_record_ids()  # a read between arrivals must not latch the release

    batch = processor.build_job_data(1)
    assert [source_record_id(sample) for sample in batch.samples] == ["i1"]
    processor.acknowledge(batch.data_id)
    assert processor.releasable_record_ids() == frozenset({"retry", "i1"})


# --- backend preparation ---------------------------------------------------


@pytest.mark.unit
def test_backend_declares_sao_and_defers_model_dependent_advantages() -> None:
    batch = JobData(
        "math:data:1",
        (policy_trajectory("i1", (5, 1), (1,), (-0.1,), 0.5, action_mask=(1,)),),
    )

    result = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)

    assert result.payload is not None and result.payload["loss"] == "sao"
    assert "advantages" not in result.payload
    assert result.metrics == {"steps": 1, "rollouts": 1, "optimizer_steps": 1}


@pytest.mark.unit
def test_backend_preparation_advances_step_state() -> None:
    batch = JobData(
        "math:data:1",
        (policy_trajectory("i1", (5, 1), (1,), (-0.1,), 0.5, action_mask=(1,)),),
    )

    first = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    second = prepare_slime_job(batch, TrainingMethod("sao"), first.next_algorithm_state, StepScheduling(), step_size=1)

    assert first.next_algorithm_state == {"steps": 1}
    assert second.next_algorithm_state == {"steps": 2}


# --- e2e: accept -> train -> commit ----------------------------------------


class _SAOComponent(StubTrainingComponent):
    """A training component that renders and prepares each job's data as the Slime component does, one sample per
    step.

    ``payloads`` lists the prepared Slime payload of each job. Each job saves
    one checkpoint whose metrics carry SAO's async telemetry, as a real SAO
    backend reports it (the shapes are asserted in test_sao_slime_component).
    """

    def __init__(self, checkpoint_root: Path) -> None:
        super().__init__(checkpoint_root)
        self.payloads: list[dict[str, Any]] = []

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        data, left_out = rendered_job_data(job.data, CharTemplate())
        assert left_out == ()
        prepared = prepare_slime_job(
            data,
            TrainingMethod.from_dict(job.parameters["method"]),
            {},
            StepScheduling.from_parameters(job.parameters),
            step_size=1,
        )
        assert prepared.payload is not None
        self.payloads.append(dict(prepared.payload))
        [checkpoint] = super().train(job)
        return (
            SavedCheckpoint(
                0, checkpoint.path, checkpoint.release_path, {"sao/critic_updates": 2, "sao/actor_trained": 1}
            ),
        )


def _repository(tmp_path: Path) -> CachedRepositoryBackendFactory:
    return InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")


def _sao_dispatcher(
    tmp_path: Path,
    recipe: SAORecipe,
    *,
    repository: CachedRepositoryBackendFactory | None = None,
    agent_dir: Path | None = None,
) -> Dispatcher:
    return Dispatcher(
        recipe,
        _repository(tmp_path) if repository is None else repository,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_dir),
        coordination=memory_coordination(),
    )


def _rollout(dispatcher: Dispatcher, index: int, score: float) -> None:
    """Accept rollout ``i<index>`` as served by the scenario's served artifact, and its report."""
    served = dispatcher.get_or_create_scenario("math").serving_artifact_id()
    dispatcher.accept_record(_sao_inference(f"i{index}", artifact_id=served))
    dispatcher.accept_record(_sao_report(f"r{index}", f"i{index}", score))


def _wait_for_finished_jobs(dispatcher: Dispatcher, count: int) -> None:
    for _ in range(500):
        state = JobLoopState.from_dict(dispatcher.get_or_create_scenario("math").trainer.state)
        if state.finished_jobs == count:
            return
        Event().wait(0.01)
    pytest.fail(f"scenario did not finish {count} jobs: {dispatcher.build_training_status()}")


def _checkpoint_commits(dispatcher: Dispatcher) -> list[CommitRecord]:
    history = dispatcher.get_or_create_scenario("math").store.history()
    return [
        record for record in history if (record.metrics or {}).get("training_job", {}).get("event") == "checkpoint"
    ]


@pytest.mark.unit
def test_sao_submits_every_batch_with_the_sao_objective() -> None:
    parameters = SAORecipe(**runtime_bindings()).parameters_by_kind()["trajectory"]

    assert TrainingMethod.from_dict(parameters["method"]) == TrainingMethod("sao")
    assert set(parameters) == {"method"}, "the training component owns the step schedule"


@pytest.mark.integration
def test_dispatcher_runs_a_full_sao_train_step_per_rollout(tmp_path) -> None:
    component = _SAOComponent(tmp_path / "checkpoints")
    dispatcher = _sao_dispatcher(
        tmp_path, SAORecipe(**runtime_bindings(component), pipeline_stages=single_rollout_steps())
    )
    try:
        _rollout(dispatcher, 1, 0.9)

        # A component batch size of 1 => exactly one accepted rollout is one training job,
        # submitted asynchronously by the dispatcher's training worker.
        _wait_for_finished_jobs(dispatcher, 1)

        [job] = component.submitted
        assert [source_record_id(item) for item in job.data.samples] == ["i1"]
        [payload] = component.payloads
        assert payload["loss"] == "sao"
        # SAO defers advantages to the critic in the backend; reef ships none.
        assert "advantages" not in payload
        assert trajectory_reward(job.data.samples[0]) == pytest.approx(0.9)
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_an_activation_rule_that_activates_nothing_keeps_the_checkpoint_from_serving(tmp_path) -> None:
    component = _SAOComponent(tmp_path / "checkpoints")
    inference = StubInferenceRuntime()
    recipe = SAORecipe.from_environment(
        {},
        config={"pipeline": {"sample2job": {"config": {"job_samples": 1}}}, "activation": {"module": "never"}},
        **runtime_bindings(component, inference),
    )
    dispatcher = _sao_dispatcher(tmp_path, recipe)
    try:
        scenario = dispatcher.get_or_create_scenario("math")
        base = inference.activated[:]
        served = scenario.current_artifact_ref()
        _rollout(dispatcher, 1, 0.9)

        _wait_for_finished_jobs(dispatcher, 1)

        # The checkpoint is published as the chain tip, but never serves: only the root release was activated.
        assert inference.activated == base
        [job] = component.submitted
        [record] = _checkpoint_commits(dispatcher)
        assert record.checkpoint and not record.activated
        assert record.metrics["training_job"] == {
            "job_id": job.job_id,
            "backend": component.name,
            "event": "checkpoint",
            "step": 0,
        }
        assert scenario.current_artifact_ref() == served
        assert scenario.repository.checkpoint_artifact == record.artifact_ref
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_sao_train_step_swaps_the_served_artifact(tmp_path) -> None:
    # The paper's async claim: after a step, the artifact that answers an
    # inference IS the freshly-trained one, not the pre-train head.
    component = _SAOComponent(tmp_path / "checkpoints")
    inference = StubInferenceRuntime()
    dispatcher = _sao_dispatcher(
        tmp_path,
        SAORecipe(**runtime_bindings(component, inference), pipeline_stages=single_rollout_steps()),
        agent_dir=tmp_path / "records",
    )
    try:
        scenario = dispatcher.get_or_create_scenario("math")
        pre_head = scenario.repository.require_current_artifact()
        pre_served = scenario.serving_artifact_id()

        _rollout(dispatcher, 1, 0.9)

        _wait_for_finished_jobs(dispatcher, 1)
        head = scenario.repository.require_current_artifact()
        # The new head is the saved release of the trained checkpoint, and it
        # actually moved off the pre-train head (guards against a constant stub).
        assert head.release_id != pre_head.release_id
        # The runtime loaded the checkpoint once, and the scenario serves its artifact.
        assert inference.activated[1:] == ["w1"]
        assert pre_served != scenario.serving_artifact_id() == head.release_id
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_sao_recovers_its_job_state_from_the_commit_log_after_restart(tmp_path) -> None:
    # Kill/restart recovery at the pipeline level: drop the in-memory dispatcher
    # and rebuild the scenario from durable state ALONE (commit history + snapshot),
    # then prove the job state was reconstructed, not remembered, and that
    # training resumes at the next rollout from the last checkpoint.
    # The component keeps its jobs across the restart; the inference runtime starts anew.
    component = _SAOComponent(tmp_path / "checkpoints")
    agent_dir = tmp_path / "records"
    repository = _repository(tmp_path)

    def _make_dispatcher() -> Dispatcher:
        recipe = SAORecipe(**runtime_bindings(component), pipeline_stages=single_rollout_steps())
        return _sao_dispatcher(tmp_path, recipe, repository=repository, agent_dir=agent_dir)

    first = _make_dispatcher()
    try:
        _rollout(first, 1, 0.9)
        _wait_for_finished_jobs(first, 1)
    finally:
        first.close()

    # Simulated restart: discard the dispatcher, rebuild over the SAME durable
    # commit history + repository backend. Nothing carries the state in memory.
    del first
    second = _make_dispatcher()
    try:
        recovered = second.get_or_create_scenario("math")

        state = JobLoopState.from_dict(recovered.trainer.state)
        assert (state.finished_jobs, state.job) == (1, None)  # from the commit history
        assert state.start == str(tmp_path / "checkpoints" / "1")
        assert recovered.store.durable
        # SAO's async telemetry is carried opaquely on the durable record, so a
        # checkpoint's schedule cadence survives the restart that produced it.
        [record] = _checkpoint_commits(second)
        assert record.metrics["sao/critic_updates"] == 2
        assert record.metrics["sao/actor_trained"] == 1

        # Training resumes at the next rollout from the first job's checkpoint.
        _rollout(second, 2, 0.5)
        _wait_for_finished_jobs(second, 2)
        assert [job.start for job in component.submitted] == [
            component.base_start,
            str(tmp_path / "checkpoints" / "1"),
        ]
        assert source_record_id(component.submitted[-1].data.samples[0]) == "i2"
    finally:
        second.close()


@pytest.mark.integration
def test_sao_train_step_recovers_across_a_restart(tmp_path) -> None:
    # Restore the job state and record progress as scenario recovery does;
    # retained records from the first rollout must not train again. The
    # samples service stores the samples of both rollouts before the first
    # job, so the scenario's durable sample store carries the second one
    # across the restart.
    component = _SAOComponent(tmp_path / "checkpoints")

    def _trainer(session, algorithm_state: Mapping[str, Any] | None = None) -> Trainer:
        return Trainer.build(
            "math",
            session.records,
            pipeline_factory=lambda context: sao_processor(context, job_samples=1),
            job_loop=JobLoop(component, scenario="math"),
            job_parameters=SAORecipe(**runtime_bindings(component)).parameters_by_kind(),
            root=STUB_ROOT,
            algorithm_state=algorithm_state,
            samples=TrainerSamples(session.samples),
        )

    first_storage = SQLiteScenarioStorage(tmp_path / "records")
    first_session = first_storage.open("math")
    for item in (
        _sao_inference("i1"),
        _sao_report("r1", "i1", 1.0),
        _sao_inference("i2"),
        _sao_report("r2", "i2", 0.5),
    ):
        first_session.records.append(item)
    first = _trainer(first_session)
    make_samples(first)
    consumed: set[str] = set()
    # Submission, the checkpoint decision and the job's end are one commit each.
    for step in range(1, 4):
        result = first.run_once()
        assert result is not None
        prepared = first.prepare_commit(result)
        first_session.commit_step(
            expected_step=step - 1,
            commit=CommitRecord(
                scenario="math",
                step=step,
                artifact_ref=ArtifactRef("content:base", "release:base", "base"),
                checkpoint=False,
                algorithm_state=prepared.algorithm_state,
                high_water_sequence=prepared.high_water_sequence,
                high_water_offset=prepared.high_water_offset,
                consumed_ids=prepared.consumed_ids,
            ),
        )
        first.commit(prepared)
        consumed |= prepared.consumed_ids
    assert JobLoopState.from_dict(first.state).finished_jobs == 1
    assert first_session.records.count("math") == 4
    first_session.close()
    first_storage.close()

    second_storage = SQLiteScenarioStorage(tmp_path / "records")
    second_session = second_storage.open("math")
    try:
        second = _trainer(second_session, prepared.algorithm_state)
        second.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=frozenset(consumed))
        second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        make_samples(second)
        assert JobLoopState.from_dict(second.state).finished_jobs == 1
        assert second.run_once() is not None
        assert [source_record_id(job.data.samples[0]) for job in component.submitted] == ["i1", "i2"]
    finally:
        second_session.close()
        second_storage.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
