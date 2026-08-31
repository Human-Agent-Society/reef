"""Reef-side SAO pipeline: recipe, processor, backend preparation, and a full
accept -> train -> commit loop against a stub training runtime.

Everything here is torch/ray free so it runs in the minimal CI gate. The Slime
backend transcription is covered separately by the container-only bridge and
parity tests.
"""

from __future__ import annotations

from threading import Event
from typing import Any

import pytest

from recipes.sao import SAORecipe
from recipes.sao.processor import SAOProcessor
from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.recipe import RecipeRegistry
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.records import RecordStore
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingRuntime
from reef.runtime.candidates import StaleCandidate
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.train import ProcessorContext, Trainer
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicyBatch, PolicySample


class _StateOnlySaoBackend(TrainingBackend):
    @property
    def dispatched(self) -> bool:
        return True

    def initial_state(self):
        return {}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        prepared = prepare_slime_step(batch, "sao", state)
        return PreparedStep.skipped(state=prepared.next_algorithm_state, metrics=prepared.metrics)

    def evaluate(self, candidate):
        raise AssertionError("state-only backend does not produce candidates")

    def settle_step(self, prepared, decision):
        raise AssertionError("state-only backend does not settle candidates")

    def abort_step(self, prepared):
        del prepared


def _sao_inference(
    agent_record_id: str,
    *,
    tokens: tuple[int, ...] = (5, 1, 2, 3),
    loss_mask: tuple[int, ...] = (1, 1, 1),
    action_mask: tuple[int, ...] | None = None,
    rollout_log_probs: tuple[float, ...] = (-0.1, -0.2, -0.3),
    runtime_load_id: str | None = "slime-v3",
) -> AgentRecord:
    payload: dict[str, Any] = {
        "tokens": list(tokens),
        "loss_mask": list(loss_mask),
        "rollout_log_probs": list(rollout_log_probs),
    }
    if action_mask is not None:
        payload["action_mask"] = list(action_mask)
    artifact_ref = (
        LiveWeightArtifactRef(
            content_id="math",
            release_id=runtime_load_id,
            parent_release_id=None,
            runtime_load_id=runtime_load_id,
        )
        if runtime_load_id
        else None
    )
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=artifact_ref,
    )


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

    recipe = build_recipe(reference, {}, runtime=_StubTrainingRuntime())

    assert isinstance(recipe, SAORecipe)
    assert recipe.name == "sao"


@pytest.mark.unit
def test_sao_recipe_defaults_are_reef_side_only() -> None:
    recipe = SAORecipe(_StubTrainingRuntime())

    # Objective defaults live with the Slime implementation. The Reef recipe
    # owns only batching and checkpoint cadence.
    assert recipe.batch_size == 1
    assert recipe.checkpoint_strategy == EveryNVersions(1)


@pytest.mark.unit
def test_sao_recipe_reads_reef_side_config() -> None:
    recipe = SAORecipe.from_environment(
        {},
        config={
            "data": {"batch_size": 2},
            "artifact": {"checkpoint_every_n_versions": 4},
        },
        runtime=_StubTrainingRuntime(),
    )

    assert recipe.batch_size == 2
    assert recipe.checkpoint_strategy == EveryNVersions(4)


@pytest.mark.unit
def test_sao_recipe_rejects_backend_objective_config() -> None:
    from reef.recipe.errors import RecipeConfigError

    with pytest.raises(RecipeConfigError, match=r"training\.slime_flags"):
        SAORecipe.from_environment(
            {},
            config={"optimization": {"eps_low": 0.8}},
            runtime=_StubTrainingRuntime(),
        )


# --- processor -------------------------------------------------------------


@pytest.mark.unit
def test_processor_emits_one_independently_scheduled_sample_per_rollout() -> None:
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(_sao_inference("i1"))
    processor.ingest(_sao_report("r1", "i1", 0.75))

    batch = processor.build_batch()

    assert isinstance(batch, PolicyBatch)
    assert len(batch.samples) == 1
    sample = batch.samples[0]
    assert sample.source_agent_record_id == "i1"
    assert sample.reward == pytest.approx(0.75)
    # No explicit action mask -> the whole response is one action.
    assert sample.action_mask == sample.loss_mask
    # Provenance rides through for policy-lag / queue-age reporting.
    assert sample.runtime_load_id == "slime-v3"
    assert sample.rollout_created_at is not None


@pytest.mark.unit
def test_processor_reads_runtime_load_id_from_the_payload_when_ref_is_not_live() -> None:
    # Rollouts served before the first training commit carry a plain checkpoint
    # ArtifactRef (kind="artifact"), not a LiveWeightArtifactRef; their producing
    # version lives only in payload["runtime_load_id"], set by the durable serve
    # path. The sample must still record its producing version, or prepare_training_step rejects
    # the whole batch ("requires one recorded producing runtime load ID") and no
    # SAO step ever runs.
    inference = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={
            "tokens": [5, 1, 2, 3],
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
            "runtime_load_id": "slime-v7",
        },
        agent_record_id="i1",
        artifact_ref=None,
    )
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference)
    processor.ingest(_sao_report("r1", "i1", 0.5))

    batch = processor.build_batch()

    assert batch.samples[0].runtime_load_id == "slime-v7"


@pytest.mark.unit
def test_processor_preserves_explicit_observation_boundaries() -> None:
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    # Response tokens: action, observation, action. loss_mask trains only the
    # action tokens.
    processor.ingest(
        _sao_inference(
            "i1",
            tokens=(5, 1, 2, 3),
            loss_mask=(1, 0, 1),
            action_mask=(1, 0, 1),
            rollout_log_probs=(-0.1, -0.2, -0.3),
        )
    )
    processor.ingest(_sao_report("r1", "i1", 1.0))

    batch = processor.build_batch()

    assert batch.samples[0].action_mask == (1, 0, 1)
    assert batch.samples[0].loss_mask == (1, 0, 1)


@pytest.mark.unit
def test_processor_drops_rollout_that_trains_a_non_action_token() -> None:
    # loss_mask trains index 1, but the action mask marks it an observation.
    # Skip-observation GAE would give it zero advantage: reject the rollout
    # rather than silently waste gradient on it.
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(
        _sao_inference(
            "i1",
            tokens=(5, 1, 2, 3),
            loss_mask=(1, 1, 1),
            action_mask=(1, 0, 1),
        )
    )
    processor.ingest(_sao_report("r1", "i1", 1.0))

    assert not processor.ready()


@pytest.mark.unit
def test_processor_drops_rollout_with_logprob_length_mismatch() -> None:
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(_sao_inference("i1", loss_mask=(1, 1, 1), rollout_log_probs=(-0.1, -0.2)))
    processor.ingest(_sao_report("r1", "i1", 1.0))

    assert not processor.ready()


@pytest.mark.unit
def test_processor_drops_rollout_with_non_finite_score() -> None:
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(_sao_inference("i1"))
    processor.ingest(_sao_report("r1", "i1", float("nan")))

    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"r1", "i1"})


@pytest.mark.unit
def test_processor_releases_inference_of_a_malformed_rollout() -> None:
    # A rollout the bridge would reject is terminal: exactly one report exists
    # per rollout, so its inference can never train and must be released rather
    # than protected forever.
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(_sao_inference("i1", loss_mask=(0, 0, 0)))
    processor.ingest(_sao_report("r1", "i1", 1.0))

    decision = processor.retention_decision()

    assert "i1" not in decision.protected_agent_record_ids


@pytest.mark.unit
@pytest.mark.parametrize("dead_report_first", [True, False])
def test_processor_trains_a_corrected_retry_after_a_dead_report(dead_report_first: bool) -> None:
    # A dead report marks its rollout releasable, but release is derived at
    # read time: while the rollout is not compacted, a corrected retry
    # re-claims it and trains — in either arrival order.
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(_sao_inference("i1"))
    dead = AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": ["i1"], "metadata": {"training": {"eligible": False}}},
        agent_record_id="dead",
        references=("i1",),
    )
    retry = _sao_report("retry", "i1", 0.5)
    for item in (dead, retry) if dead_report_first else (retry, dead):
        processor.ingest(item)
        processor.retention_decision()  # a read between arrivals must not latch the release

    batch = processor.build_batch()
    assert [sample.source_agent_record_id for sample in batch.samples] == ["i1"]
    processor.acknowledge(batch.batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"dead", "retry", "i1"})


# --- backend preparation ---------------------------------------------------


@pytest.mark.unit
def test_backend_declares_sao_and_defers_model_dependent_advantages() -> None:
    batch = PolicyBatch(
        "math:sao:1",
        (PolicySample("i1", (5, 1), (1,), (-0.1,), 0.5, action_mask=(1,)),),
    )

    result = prepare_slime_step(batch, "sao", {})

    assert result.payload is not None and result.payload["loss"] == "sao"
    assert "advantages" not in result.payload
    assert result.metrics == {"steps": 1, "rollouts": 1}


@pytest.mark.unit
def test_backend_preparation_advances_step_state() -> None:
    batch = PolicyBatch("math:sao:1", (PolicySample("i1", (5, 1), (1,), (-0.1,), 0.5, action_mask=(1,)),))

    first = prepare_slime_step(batch, "sao", {})
    second = prepare_slime_step(batch, "sao", first.next_algorithm_state)

    assert first.next_algorithm_state == {"steps": 1}
    assert second.next_algorithm_state == {"steps": 2}


# --- e2e: accept -> train -> commit ----------------------------------------


class _StubTrainingRuntime(TrainingRuntime):
    """A durable SAO runtime driven by the background training worker.

    Mirrors the runtime's prepare/train/activate contract: training exports a
    candidate without touching serving, then activation applies only a
    selected candidate. Candidate production is idempotent per rollout so a
    re-driven job returns the same candidate instead of training twice.
    """

    def __init__(self, checkpoint_root: Any = None) -> None:
        super().__init__(base_url="http://trainer")
        self.checkpoint_root = checkpoint_root
        self.jobs: list[dict[str, Any]] = []
        self.completed: dict[int, ModelCandidate] = {}
        self.activations: list[ModelCandidate] = []
        self.rejections: list[tuple[ModelCandidate, Any]] = []
        # The version the serving engine answers from. A train step hot-swaps
        # new weights in, so this moves from v3 -> v4 the first time a job
        # completes; starting below the synced value keeps the serving-swap
        # test from passing on a constant.
        self._served_version = "slime-v3"

    @property
    def inference_backend(self):
        return None

    def serving_runtime_load_id(self):
        return self._served_version

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
        assert isinstance(batch, PolicyBatch)
        sample = batch.samples[0]
        prepared = prepare_slime_step(batch, step_preparer, algorithm_state)
        assert prepared.payload is not None
        payload = {
            **prepared.payload,
            "rollout_id": scenario_step,
            "reward": sample.reward,
            "expected_runtime_load_id": sample.runtime_load_id,
        }
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state=prepared.next_algorithm_state,
            metrics=prepared.metrics,
        )

    def train_candidate(self, payload):
        rollout_id = payload["rollout_id"]
        existing = self.completed.get(rollout_id)
        if existing is not None:
            return existing
        if payload["expected_runtime_load_id"] != self._served_version:
            raise StaleCandidate
        self.jobs.append(dict(payload))
        job_id = f"job-{rollout_id}"
        checkpoint = self.checkpoint_root / job_id
        checkpoint.mkdir(parents=True)
        result = ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=str(checkpoint),
            current_runtime_load_id=self._served_version,
            # A real SAO backend reports its schedule cadence and async
            # provenance here; the shapes are asserted in test_sao_bridge.
            training_metrics={"sao/critic_updates": 2, "sao/actor_trained": 1},
        )
        self.completed[rollout_id] = result
        return result

    def activate_candidate(self, candidate):
        self.activations.append(candidate)
        self._served_version = "slime-v4"
        return ActivatedModel(candidate.candidate_id, self._served_version)

    def reject_candidate(self, candidate, decision):
        self.rejections.append((candidate, decision))


def _wait_for_step(dispatcher: Dispatcher, step: int) -> None:
    for _ in range(500):
        if dispatcher.get_or_create_scenario("math").scenario_step == step:
            return
        Event().wait(0.01)
    pytest.fail(f"scenario did not reach step {step}: {dispatcher.training_status}")


@pytest.mark.integration
def test_dispatcher_runs_a_full_sao_train_step_per_rollout(tmp_path) -> None:
    runtime = _StubTrainingRuntime(tmp_path / "checkpoints")
    initial = tmp_path / "initial"
    initial.mkdir()

    dispatcher = Dispatcher(
        RecipeRegistry(recipes={"sao": SAORecipe(runtime)}),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
    )
    try:
        dispatcher.accept_record(_sao_inference("i1"), recipe="sao")
        dispatcher.accept_record(_sao_report("r1", "i1", 0.9), recipe="sao")

        # batch_size 1 => exactly one accepted rollout drives one training step,
        # executed asynchronously on the dispatcher's training worker.
        _wait_for_step(dispatcher, 1)
        scenario = dispatcher.get_or_create_scenario("math")
        assert scenario.trainer.state == {"steps": 1}

        assert len(runtime.jobs) == 1
        job = runtime.jobs[0]
        assert job["rollout_id"] == 0
        assert job["loss"] == "sao"
        # SAO defers advantages to the critic in the backend; reef ships none.
        assert "advantages" not in job
        assert job["reward"] == pytest.approx(0.9)
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_external_checkpoint_evaluation_rejects_before_serving_activation(tmp_path) -> None:
    runtime = _StubTrainingRuntime(tmp_path / "checkpoints")
    initial = tmp_path / "initial"
    initial.mkdir()
    recipe = SAORecipe.from_environment(
        {"EVALUATION_TOKEN": "secret"},
        config={
            "evaluation": {
                "module": "reef_service._candidate_evaluation_plugin:build_evaluator",
                "config": {
                    "score": 0.25,
                    "threshold": 0.8,
                    "token_env": "EVALUATION_TOKEN",
                },
            }
        },
        runtime=runtime,
    )
    dispatcher = Dispatcher(
        RecipeRegistry(recipes={"sao": recipe}),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
    )
    try:
        dispatcher.accept_record(_sao_inference("i1"), recipe="sao")
        dispatcher.accept_record(_sao_report("r1", "i1", 0.9), recipe="sao")

        _wait_for_step(dispatcher, 1)

        assert runtime.serving_runtime_load_id() == "slime-v3"
        assert runtime.activations == []
        assert len(runtime.rejections) == 1
        candidate, decision = runtime.rejections[0]
        assert decision.outcome == "reject"
        assert decision.evaluation.evaluator == "external_checkpoint"
        assert decision.evaluation.metrics == {
            "score": 0.25,
            "checkpoint_path": candidate.checkpoint_path,
        }
        assert decision.evaluation.metadata == {"scenario": "math", "token_present": True}
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_sao_train_step_swaps_the_served_runtime_load_id(tmp_path) -> None:
    # The paper's async claim: after a step, the version an inference would
    # freeze IS the freshly-synced one, not the pre-train head. Force a live
    # step (checkpoint_every_n high) so the head stays a LiveWeightArtifactRef
    # whose runtime_load_id is observable.
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = _StubTrainingRuntime(tmp_path / "checkpoints")

    dispatcher = Dispatcher(
        RecipeRegistry(recipes={"sao": SAORecipe(runtime, checkpoint_strategy=EveryNVersions(99))}),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
    )
    try:
        pre_head = dispatcher.get_or_create_scenario("math", recipe="sao").repository.require_current_artifact()
        assert runtime.serving_runtime_load_id() == "slime-v3"

        dispatcher.accept_record(_sao_inference("i1"), recipe="sao")
        dispatcher.accept_record(_sao_report("r1", "i1", 0.9), recipe="sao")

        _wait_for_step(dispatcher, 1)
        head = dispatcher.get_or_create_scenario("math").repository.require_current_artifact()
        # The new head is a live-weight ref carrying the freshly-synced version,
        # and it actually moved off the pre-train head (guards against a
        # constant stub).
        assert isinstance(head, LiveWeightArtifactRef)
        assert head.runtime_load_id == "slime-v4"
        assert head.release_id != pre_head.release_id
        assert runtime.serving_runtime_load_id() == "slime-v4"
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_sao_recovers_step_from_the_commit_log_after_restart(tmp_path) -> None:
    # Kill/restart recovery at the pipeline level: drop the in-memory dispatcher
    # and rebuild the scenario from durable state ALONE (commit log + snapshot),
    # then prove the step counter and algorithm state were reconstructed, not
    # remembered, and that training resumes at the next rollout.
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = _StubTrainingRuntime(tmp_path / "checkpoints")
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_dir = tmp_path / "records"

    def _make_dispatcher() -> Dispatcher:
        return Dispatcher(
            RecipeRegistry(recipes={"sao": SAORecipe(runtime)}),
            backend,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=agent_dir,
        )

    first = _make_dispatcher()
    try:
        first.accept_record(_sao_inference("i1"), recipe="sao")
        first.accept_record(_sao_report("r1", "i1", 0.9), recipe="sao")
        _wait_for_step(first, 1)
    finally:
        first.close()

    # Simulated restart: discard the dispatcher, rebuild over the SAME durable
    # commit log + repository backend. Nothing carries the step in memory.
    del first
    second = _make_dispatcher()
    try:
        recovered = second.get_or_create_scenario("math")

        assert recovered.scenario_step == 1  # step rebuilt from durable state
        assert recovered.trainer.state == {"steps": 1}  # algorithm_state from the commit log
        assert recovered.commit_log is not None
        record = recovered.commit_log.records()[-1]
        assert record.step == 1  # the journal drove recovery
        # SAO's async telemetry is carried opaquely on the durable record, so a
        # step's schedule cadence survives the restart that produced it.
        assert record.metrics is not None
        assert record.metrics["sao/critic_updates"] == 2
        assert record.metrics["sao/actor_trained"] == 1

        # Training resumes at the next rollout and advances past the recovered
        # step. The serving version already moved to slime-v4 on the first step.
        second.accept_record(_sao_inference("i2", runtime_load_id="slime-v4"), recipe="sao")
        second.accept_record(_sao_report("r2", "i2", 0.5), recipe="sao")
        _wait_for_step(second, 2)
        assert second.get_or_create_scenario("math").trainer.state == {"steps": 2}
    finally:
        second.close()
    assert second.get_or_create_scenario("math").trainer.state == {"steps": 2}


@pytest.mark.integration
def test_sao_train_step_recovers_across_a_restart(tmp_path) -> None:
    # A second rollout after a fresh Trainer built from the persisted algorithm
    # state must resume the step counter, mirroring recovery from a checkpoint.
    database = tmp_path / "records.sqlite3"
    first_inference = _sao_inference("i1")
    first_report = _sao_report("r1", "i1", 1.0)
    second_inference = _sao_inference("i2")
    second_report = _sao_report("r2", "i2", 0.5)

    with RecordStore(database) as first_store:
        for item in (first_inference, first_report, second_inference, second_report):
            first_store.append(item)
        first = Trainer.build(
            "math",
            first_store,
            processor_factory=lambda context: SAOProcessor(context.with_config({"batch_size": 1})),
            training_backend=_StateOnlySaoBackend(),
        )
        batch = first.reserve_training_batch()
        assert batch is not None
        assert first.execute_reserved_step(0).result is not None
        prepared = first.commit()
        first.apply_compaction(prepared.compacted_ids)

    with RecordStore(database) as second_store:
        second = Trainer.build(
            "math",
            second_store,
            processor_factory=lambda context: SAOProcessor(context.with_config({"batch_size": 1})),
            training_backend=_StateOnlySaoBackend(),
            algorithm_state={"steps": 1},
        )
        assert second.state == {"steps": 1}
        assert second.reserve_training_batch() is not None
        assert second.pending_batch.samples[0].source_agent_record_id == "i2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
