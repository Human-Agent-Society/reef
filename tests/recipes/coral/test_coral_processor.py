"""CoralBatches: CORAL sibling groups -> grouped policy batches.

Requires the reef package (and its train stack) importable; skipped
otherwise so the adapter suite stays standalone.
"""

from __future__ import annotations

import pytest

from reef.core.trajectories import trajectory_reward
from reef.train.types import trajectory_groups

reef_types = pytest.importorskip("reef.train.types", reason="requires a reef checkout")

from recipes.beta.coral.processor import ROOT_GROUP, CoralAssemblySettings, CoralBatches, CoralTasks
from recipes.beta.coral.recipe import CoralGroupSettings, CoralRecord2Record
from reef.core import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainingBatch

SCENARIO = "coral-demo"


def _processor(group_size=2):
    context = PipelineContext(SCENARIO)
    return TrainingPipeline(
        context,
        record2record=CoralRecord2Record(context, CoralGroupSettings(group_size=group_size)),
        record2task=CoralTasks(CoralAssemblySettings()),
        task2batch=CoralBatches(context, None),
    )


def _inference(record_id, tokens, loss_mask, log_probs):
    return AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        payload={
            "response": {
                "training": {
                    "tokens": tokens,
                    "loss_mask": loss_mask,
                    "rollout_log_probs": log_probs,
                    "runtime_load_id": "wv-1",
                }
            }
        },
    )


def _attempt_report(record_id, references, score, *, commit, parent=None, agent="agent-1"):
    refs = (references,) if isinstance(references, str) else tuple(references)
    return AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.REPORT,
        agent_record_id=record_id,
        references=refs,
        payload={
            "score": score,
            "references": list(refs),
            "metadata": {
                "coral": {
                    "agent_id": agent,
                    "commit_hash": commit,
                    "parent_hash": parent,
                    "status": "improved",
                    "run_id": "run-1",
                }
            },
        },
    )


def test_sibling_group_trains_as_one_grouped_batch():
    processor = _processor(group_size=2)
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    processor.push(_inference("i2", [3, 4], [1], [-0.2]))
    processor.push(_attempt_report("r1", "i1", 0.3, commit="c-a", parent="p0"))
    assert not processor.ready(2)  # one sibling is not a comparison group

    processor.push(_attempt_report("r2", "i2", 0.9, commit="c-b", parent="p0"))
    assert processor.ready(2)

    batch = processor.build_batch(2)
    assert isinstance(batch, TrainingBatch)
    (group,) = trajectory_groups(batch)
    assert sorted(trajectory_reward(sample) for sample in group) == [0.3, 0.9]


def test_root_attempts_group_together():
    processor = _processor(group_size=2)
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    processor.push(_inference("i2", [3, 4], [1], [-0.2]))
    processor.push(_attempt_report("r1", "i1", 0.1, commit="c-a", parent=None))
    processor.push(_attempt_report("r2", "i2", 0.2, commit="c-b", parent=None))
    assert processor.ready(2)
    batch = processor.build_batch(2)
    assert {item.group_id for item in batch.items} == {ROOT_GROUP}


def test_groups_do_not_mix_across_parents():
    processor = _processor(group_size=2)
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    processor.push(_inference("i2", [3, 4], [1], [-0.2]))
    processor.push(_attempt_report("r1", "i1", 0.5, commit="c-a", parent="p0"))
    processor.push(_attempt_report("r2", "i2", 0.6, commit="c-b", parent="p1"))
    # two half-full groups, no cross-parent comparison
    assert not processor.ready(2)


def test_a_batch_holds_complete_sibling_groups_each_grouped_by_its_own_parent():
    processor = _processor(group_size=2)
    for index in range(4):
        processor.push(_inference(f"i{index}", [1, 2], [1], [-0.1]))
    processor.push(_attempt_report("r0", "i0", 0.1, commit="c-0", parent="p0"))
    processor.push(_attempt_report("r1", "i1", 0.2, commit="c-1", parent="p0"))
    assert not processor.ready(4), "one sibling group is half a batch of four"
    processor.push(_attempt_report("r2", "i2", 0.3, commit="c-2", parent="p1"))
    processor.push(_attempt_report("r3", "i3", 0.4, commit="c-3", parent="p1"))
    assert processor.ready(4)

    groups = trajectory_groups(processor.build_batch(4))
    assert [[(sample.group_id, trajectory_reward(sample)) for sample in group] for group in groups] == [
        [("p0", 0.1), ("p0", 0.2)],
        [("p1", 0.3), ("p1", 0.4)],
    ]


def test_regrade_retry_at_same_commit_is_terminal_not_double_counted():
    processor = _processor(group_size=2)
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    processor.push(_inference("i2", [3, 4], [1], [-0.2]))
    processor.push(_attempt_report("r1", "i1", 0.5, commit="c-a", parent="p0"))
    # duplicate grader run for the same attempt commit, different record id
    processor.push(_attempt_report("r1b", "i1", 0.5, commit="c-a", parent="p0"))
    assert not processor.ready(2)  # still one distinct sibling

    processor.push(_attempt_report("r2", "i2", 0.7, commit="c-b", parent="p0"))
    assert processor.ready(2)
    (group,) = trajectory_groups(processor.build_batch(2))
    assert len(group) == 2


def test_multi_call_attempt_assembles_multi_turn():
    processor = _processor(group_size=2)
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    processor.push(_inference("i2", [1, 2, 3, 4], [1], [-0.2]))  # extends i1
    processor.push(_inference("i3", [5, 6], [1], [-0.3]))
    processor.push(_attempt_report("r1", ("i1", "i2"), 0.5, commit="c-a", parent="p0"))
    processor.push(_attempt_report("r2", "i3", 0.8, commit="c-b", parent="p0"))
    assert processor.ready(2)
    (group,) = trajectory_groups(processor.build_batch(2))
    multi = next(s for s in group if trajectory_reward(s) == 0.5)
    assert multi.training["turn_count"] == 2


def test_report_without_coral_metadata_fails_explicitly():
    processor = _processor()
    processor.push(_inference("i1", [1, 2], [1], [-0.1]))
    with pytest.raises(ValueError, match=r"metadata\.coral"):
        processor.push(
            AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.REPORT,
                agent_record_id="r-bare",
                references=("i1",),
                payload={"score": 1.0, "references": ["i1"]},
            )
        )
    assert processor.releasable_record_ids().isdisjoint({"i1", "r-bare"})


def test_group_size_floor():
    with pytest.raises(ValueError, match="at least two"):
        _processor(group_size=1)


def test_status_is_a_mapping_even_before_any_discard():
    """Regression: a private-name collision with the base class made
    status() call .items() on the base's discard set."""
    processor = _processor()
    status = processor.status()
    assert status["discarded_groups"] == []
