"""The Reasoning Agent's training data: reports grouped by task, complete groups batched, group relative advantages."""

from __future__ import annotations

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.beta.spade import SpadePreparer, SpadeProcessor, SpadeRecipe
from recipes.beta.spade.processor import reported_task_name
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.core.trajectories import make_trajectory
from reef.inference.http import InferenceProxyRuntime
from reef.train.algos.registry import resolve_preparer
from reef.train.processors.reported import GroupDecision
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem, trajectory_groups


def inference(record_id: str) -> AgentRecord:
    payload = {
        "messages": [{"role": "user", "content": "ls"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "ls -la"}}]},
        "training": {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "rollout_log_probs": [0.0, -0.5, -0.7],
            "runtime_load_id": "r1",
        },
    }
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def report(record_id: str, reference: str, task: str | None, score: float) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "feedback": f"verifier reward {score}", "references": [reference]}
    if task is not None:
        payload["metadata"] = {"task": {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}}
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def processor(**config: object) -> SpadeProcessor:
    return SpadeProcessor(ProcessorContext("spade", {"tasks_per_step": 2, "rollouts_per_task": 2, **config}))


def played(processor: SpadeProcessor, task: str, index: int, score: float) -> None:
    processor.ingest(inference(f"rec-{task}-{index}"))
    processor.ingest(report(f"rep-{task}-{index}", f"rec-{task}-{index}", task, score))


def test_reported_task_name_reads_the_task_players_metadata() -> None:
    assert reported_task_name(report("r", "i", "harbor-00000-000", 1.0)) == "harbor-00000-000"
    assert reported_task_name(report("r", "i", None, 1.0)) is None


def test_episodes_of_one_task_form_a_group_and_a_batch_holds_complete_groups_only() -> None:
    p = processor()
    played(p, "harbor-00000-000", 0, 1.0)
    assert not p.ready(), "one episode of a task is not a group"
    played(p, "harbor-00000-001", 0, 0.0)
    played(p, "harbor-00000-000", 1, 0.0)
    assert not p.ready(), "one complete group is not a batch of two"
    played(p, "harbor-00000-001", 1, 1.0)
    assert p.ready()
    batch = p.build_batch()
    assert isinstance(batch, TrainingBatch) and batch.batch_id == "spade:spade:1"
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["harbor-00000-000", "harbor-00000-000"],
        ["harbor-00000-001", "harbor-00000-001"],
    ]
    assert [sample.metadata["reward"] for group in groups for sample in group] == [1.0, 0.0, 0.0, 1.0]


def test_a_report_without_a_task_is_refused_and_keeps_its_report_retryable() -> None:
    p = processor()
    p.ingest(inference("rec-x"))
    with pytest.raises(ValueError, match=r"requires metadata\.task\.name"):
        p.ingest(report("rep-x", "rec-x", None, 1.0))


def test_decide_group_waits_for_rollouts_per_task_episodes() -> None:
    p = processor(rollouts_per_task=3)
    one: tuple[TrajectoryItem, ...] = (make_trajectory((inference("a"),), 1.0),)
    assert p.decide_group("t", one) is GroupDecision.INCOMPLETE
    assert p.decide_group("t", one * 3) is GroupDecision.READY


@pytest.mark.parametrize(
    ("config", "message"),
    [({"tasks_per_step": 0}, "tasks_per_step must be positive"), ({"rollouts_per_task": 1}, "at least two")],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_the_preparer_centers_and_scales_rewards_within_each_task_group() -> None:
    p = processor()
    for index, score in enumerate((1.0, 0.0)):
        played(p, "harbor-00000-000", index, score)
    for index, score in enumerate((0.5, 0.5)):
        played(p, "harbor-00000-001", index, score)
    batch = p.build_batch()
    signal = resolve_preparer("spade")(batch, {})
    assert signal.action == "train" and signal.loss_family == "importance_sampling"
    assert signal.advantages == (1.0, -1.0, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1
    assert isinstance(SpadePreparer(), SpadePreparer)


def test_the_recipe_binds_the_processor_the_preparer_and_tinkers_loss() -> None:
    spec = SpadeRecipe.training_spec()
    assert spec.processor is SpadeProcessor and spec.step_preparer == "spade"
    assert spec.loss_family == "importance_sampling"
    assert SpadeRecipe.report_type.fget(SpadeRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")
    training_runtime = StubTrainingRuntime()
    recipe = SpadeRecipe(training_runtime=training_runtime, runtime=runtime)
    assert recipe.processor_config() == {"tasks_per_step": 4, "rollouts_per_task": 4, "scaffold_tolerance": 8}
    with pytest.raises(ValueError, match="scaffold_tolerance"):
        SpadeRecipe(training_runtime=training_runtime, runtime=runtime, scaffold_tolerance=-1)


def test_the_assembly_spans_an_episodes_turns_and_realigns_the_think_scaffold() -> None:
    assembly = processor()._assembly
    assert assembly.accept_multi_turn and assembly.scaffold_tolerance == 8
    assert processor(scaffold_tolerance=2)._assembly.scaffold_tolerance == 2
