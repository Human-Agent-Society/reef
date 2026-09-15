"""SPADE training data: the Reasoning Agent's reports grouped by task, the Designer's by generation, both batched by complete groups."""

from __future__ import annotations

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.beta.spade import SpadePreparer, SpadeProcessor, SpadeRecipe
from recipes.beta.spade.designer_processor import SpadeDesignerProcessor, generation_label, reported_generation
from recipes.beta.spade.processor import reported_task_name
from recipes.beta.spade.recipe import SpadeDesignerRecipe
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


def designer_report(record_id: str, reference: str, score: float, **metadata: object) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "feedback": "SPADE Designer regret", "references": [reference]}
    if metadata:
        payload["metadata"] = dict(metadata)
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def designer_processor(**config: object) -> SpadeDesignerProcessor:
    return SpadeDesignerProcessor(ProcessorContext("spade", dict(config)))


def proposed(
    processor: SpadeDesignerProcessor, generation: int, index: int, score: float, proposals: int, **extra: object
) -> None:
    processor.ingest(inference(f"designer-{generation}-{index}"))
    processor.ingest(
        designer_report(
            f"regret-{generation}-{index}",
            f"designer-{generation}-{index}",
            score,
            generation=generation,
            proposals=proposals,
            **extra,
        )
    )


def test_reported_generation_reads_the_generations_report_metadata() -> None:
    assert reported_generation(designer_report("r", "i", 0.5, generation=3, proposals=8)) == (3, 8)
    assert generation_label(3) == "generation-00003"


def test_a_generations_proposals_form_a_group_and_a_refusal_is_a_member() -> None:
    p = designer_processor()
    proposed(p, 0, 0, 0.5, 3, designer_version={"kind": "release", "id": "r1"}, opponent={"model": "m"})
    proposed(p, 0, 1, 0.0, 3, refusal="reply refused: no json block")
    assert not p.ready(), "two of three proposals are not a generation"
    proposed(p, 0, 2, -0.25, 3)
    assert p.ready()
    batch = p.build_batch()
    assert batch.batch_id == "spade:spade-designer:1"
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [["generation-00000"] * 3]
    first = groups[0][0]
    assert first.metadata["generation"] == 0 and first.metadata["proposals"] == 3
    assert first.metadata["designer_version"] == {"kind": "release", "id": "r1"}
    assert first.metadata["opponent"] == {"model": "m"}
    assert [sample.metadata["reward"] for sample in groups[0]] == [0.5, 0.0, -0.25]


def test_generations_per_step_batches_that_many_complete_generations() -> None:
    p = designer_processor(generations_per_step=2)
    for index, score in enumerate((1.0, 0.0)):
        proposed(p, 4, index, score, 2)
    assert not p.ready(), "one complete generation is not a batch of two"
    proposed(p, 5, 0, 0.5, 2)
    assert not p.ready()
    proposed(p, 5, 1, 0.5, 2)
    assert p.ready()
    groups = trajectory_groups(p.build_batch())
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["generation-00004"] * 2,
        ["generation-00005"] * 2,
    ]


@pytest.mark.parametrize(
    ("metadata", "key"),
    [
        ({"proposals": 2}, "generation"),
        ({"generation": 1}, "proposals"),
        ({"generation": 1, "proposals": 0}, "proposals"),
    ],
)
def test_a_designer_report_without_its_generation_or_size_is_refused(metadata: dict[str, object], key: str) -> None:
    p = designer_processor()
    p.ingest(inference("designer-x"))
    with pytest.raises(ValueError, match=rf"requires metadata\.{key}"):
        p.ingest(designer_report("regret-x", "designer-x", 0.5, **metadata))


def test_a_generations_per_step_that_cannot_train_is_refused() -> None:
    with pytest.raises(ValueError, match="generations_per_step must be positive"):
        designer_processor(generations_per_step=0)


def test_the_preparer_centers_regret_within_one_generation() -> None:
    p = designer_processor()
    for index, score in enumerate((0.5, 0.5, -0.5, -0.5)):
        proposed(p, 7, index, score, 4)
    signal = resolve_preparer("spade")(p.build_batch(), {})
    assert signal.loss_family == "importance_sampling"
    assert signal.advantages == (1.0, 1.0, -1.0, -1.0)
    assert signal.metrics["constant_groups"] == 0


def test_the_designer_recipe_binds_its_processor_the_shared_preparer_and_tinkers_loss() -> None:
    spec = SpadeDesignerRecipe.training_spec()
    assert spec.processor is SpadeDesignerProcessor and spec.step_preparer == "spade"
    assert spec.loss_family == "importance_sampling"
    assert SpadeDesignerRecipe.report_type.fget(SpadeDesignerRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8001")
    training_runtime = StubTrainingRuntime()
    recipe = SpadeDesignerRecipe(training_runtime=training_runtime, runtime=runtime)
    assert recipe.name == "spade_designer" and recipe.processor_config() == {"generations_per_step": 1}
    with pytest.raises(ValueError, match="generations_per_step"):
        SpadeDesignerRecipe(training_runtime=training_runtime, runtime=runtime, generations_per_step=0)


def hint_report(record_id: str, reference: str, task: str) -> AgentRecord:
    payload: dict[str, object] = {
        "score": 1.0,
        "feedback": "verifier reward 1.0",
        "references": [reference],
        "metadata": {"task": {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}, "arm": "hint"},
    }
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def test_a_hint_arm_report_is_discarded_and_releases_its_episode() -> None:
    p = processor()
    p.ingest(inference("rec-hint"))
    p.ingest(hint_report("rep-hint", "rec-hint", "harbor-00000-000"))
    assert not p.ready() and p.group_status() == {"ready_groups": 0, "groups": {}}
    assert "rec-hint" in p.retention_decision().releasable_agent_record_ids, "the measured episode holds nothing"
    played(p, "harbor-00000-000", 0, 1.0)
    assert p.status()["groups"] == {"harbor-00000-000": 1}, "a plain episode still waits for its group"


def test_a_batch_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    p = processor()
    for task in ("harbor-00000-000", "harbor-00000-001"):
        for index in range(2):
            played(p, task, index, 1.0)
    signal = resolve_preparer("spade")(p.build_batch(), {"steps": 3})
    assert signal.action == "skip" and signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 3}
    assert signal.metrics == {"constant_groups": 2, "skipped": "every group is constant"}


def test_designer_proposals_compare_within_one_generation_and_one_skill() -> None:
    p = designer_processor()
    proposed(p, 2, 0, 0.5, 3, skill="inspection")
    proposed(p, 2, 1, 0.0, 3, skill="repair")
    assert p.status() == {"ready_groups": 0, "groups": {"2": 2}}, "a generation is one unit, ready at proposals"
    proposed(p, 2, 2, 1.0, 3, skill="repair")
    groups = trajectory_groups(p.build_batch())
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["generation-00002-inspection"],
        ["generation-00002-repair", "generation-00002-repair"],
    ]


def round_report(record_id: str, reference: str, task: str, score: float, plays: int) -> AgentRecord:
    payload: dict[str, object] = {
        "score": score,
        "feedback": f"verifier reward {score}",
        "references": [reference],
        "metadata": {
            "task": {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32},
            "arm": "plain",
            "round": "generation-00003",
            "round_plays": plays,
        },
    }
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def test_a_rounds_plays_are_one_batch_ordered_by_task_whatever_tasks_per_step_says() -> None:
    p = processor(tasks_per_step=4, rollouts_per_task=3)
    plays = [
        ("harbor-00003-001", 1.0),
        ("harbor-00003-000", 0.0),
        ("harbor-00003-001", 0.0),
        ("harbor-00003-000", 1.0),
    ]
    for index, (task, score) in enumerate(plays):
        p.ingest(inference(f"rec-{index}"))
        p.ingest(round_report(f"rep-{index}", f"rec-{index}", task, score, len(plays)))
        assert p.ready() == (index == len(plays) - 1)
    batch = p.build_batch()
    assert [sample.group_id for sample in batch.items] == ["harbor-00003-000"] * 2 + ["harbor-00003-001"] * 2
    assert [sample.metadata["round"] for sample in batch.items] == ["generation-00003"] * 4
    assert resolve_preparer("spade")(batch, {}).advantages == (-1.0, 1.0, 1.0, -1.0)


def test_a_round_report_without_its_size_is_refused() -> None:
    p = processor()
    p.ingest(inference("rec-r"))
    broken = round_report("rep-r", "rec-r", "harbor-00003-000", 1.0, 0)
    with pytest.raises(ValueError, match="round_plays"):
        p.ingest(broken)
