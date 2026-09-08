"""CoralProcessor: CORAL sibling groups -> grouped policy batches.

Requires the reef package (and its train stack) importable; skipped
otherwise so the adapter suite stays standalone.
"""

from __future__ import annotations

import pytest

reef_types = pytest.importorskip("reef.train.types", reason="requires a reef checkout")

from recipes.coral.processor import ROOT_GROUP, CoralProcessor

from reef.core import AgentRecord, RequestType
from reef.train.types import GroupedPolicyBatch, ProcessorContext

SCENARIO = "coral-demo"


def _processor(group_size=2):
    return CoralProcessor(ProcessorContext(SCENARIO, {"group_size": group_size}))


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
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [3, 4], [1], [-0.2]))
    processor.ingest(_attempt_report("r1", "i1", 0.3, commit="c-a", parent="p0"))
    assert not processor.ready()  # one sibling is not a comparison group

    processor.ingest(_attempt_report("r2", "i2", 0.9, commit="c-b", parent="p0"))
    assert processor.ready()

    batch = processor.build_batch()
    assert isinstance(batch, GroupedPolicyBatch)
    (group,) = batch.comparison_sets
    assert sorted(sample.reward for sample in group) == [0.3, 0.9]


def test_root_attempts_group_together():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [3, 4], [1], [-0.2]))
    processor.ingest(_attempt_report("r1", "i1", 0.1, commit="c-a", parent=None))
    processor.ingest(_attempt_report("r2", "i2", 0.2, commit="c-b", parent=None))
    assert processor.ready()
    batch = processor.build_batch()
    assert ROOT_GROUP in batch.batch_id


def test_groups_do_not_mix_across_parents():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [3, 4], [1], [-0.2]))
    processor.ingest(_attempt_report("r1", "i1", 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i2", 0.6, commit="c-b", parent="p1"))
    # two half-full groups, no cross-parent comparison
    assert not processor.ready()


def test_regrade_retry_at_same_commit_is_terminal_not_double_counted():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [3, 4], [1], [-0.2]))
    processor.ingest(_attempt_report("r1", "i1", 0.5, commit="c-a", parent="p0"))
    # duplicate grader run for the same attempt commit, different record id
    processor.ingest(_attempt_report("r1b", "i1", 0.5, commit="c-a", parent="p0"))
    assert not processor.ready()  # still one distinct sibling

    processor.ingest(_attempt_report("r2", "i2", 0.7, commit="c-b", parent="p0"))
    assert processor.ready()
    (group,) = processor.build_batch().comparison_sets
    assert len(group) == 2


def test_multi_call_attempt_assembles_multi_turn():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [1, 2, 3, 4], [1], [-0.2]))  # extends i1
    processor.ingest(_inference("i3", [5, 6], [1], [-0.3]))
    processor.ingest(_attempt_report("r1", ("i1", "i2"), 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i3", 0.8, commit="c-b", parent="p0"))
    assert processor.ready()
    (group,) = processor.build_batch().comparison_sets
    multi = next(s for s in group if s.reward == 0.5)
    assert multi.is_multi_turn and multi.turn_count == 2


def test_report_without_coral_metadata_is_named_never():
    processor = _processor()
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(
        AgentRecord.create(
            scenario=SCENARIO,
            request_type=RequestType.REPORT,
            agent_record_id="r-bare",
            references=("i1",),
            payload={"score": 1.0, "references": ["i1"]},
        )
    )
    assert any("metadata.coral" in reason for reason in processor.never_reasons)


def test_group_size_floor():
    with pytest.raises(ValueError, match="at least two"):
        _processor(group_size=1)


def test_status_is_a_mapping_even_before_any_discard():
    """Regression: a private-name collision with the base class made
    status() call .items() on the base's discard set."""
    processor = _processor()
    status = processor.status()
    assert status == {"discarded_groups": [], "never_reasons": {}}
