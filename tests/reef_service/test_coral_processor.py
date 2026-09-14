"""CoralProcessor: CORAL sibling groups -> grouped policy batches.

Requires the reef package (and its train stack) importable; skipped
otherwise so the adapter suite stays standalone.
"""

from __future__ import annotations

import pytest

from reef.core.trajectories import trajectory_reward
from reef.train.types import trajectory_groups

reef_types = pytest.importorskip("reef.train.types", reason="requires a reef checkout")

from recipes.beta.coral.processor import ROOT_GROUP, CoralProcessor
from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import ArtifactRef
from reef.train.types import ProcessorContext, TrainingBatch

SCENARIO = "coral-demo"


def _processor(group_size=2, **config):
    return CoralProcessor(ProcessorContext(SCENARIO, {"group_size": group_size, **config}))


def _inference(record_id, tokens, loss_mask, log_probs, release="wv-1"):
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
                    "runtime_load_id": release,
                }
            }
        },
        artifact_ref=ArtifactRef(content_id="c", release_id=release, parent_release_id=None),
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
    assert isinstance(batch, TrainingBatch)
    (group,) = trajectory_groups(batch)
    assert sorted(trajectory_reward(sample) for sample in group) == [0.3, 0.9]


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
    (group,) = trajectory_groups(processor.build_batch())
    assert len(group) == 2


def test_multi_call_attempt_assembles_multi_turn():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [1, 2, 3, 4], [1], [-0.2]))  # extends i1
    processor.ingest(_inference("i3", [5, 6], [1], [-0.3]))
    processor.ingest(_attempt_report("r1", ("i1", "i2"), 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i3", 0.8, commit="c-b", parent="p0"))
    assert processor.ready()
    (group,) = trajectory_groups(processor.build_batch())
    multi = next(s for s in group if trajectory_reward(s) == 0.5)
    assert multi.training["turn_count"] == 2


def test_report_without_coral_metadata_fails_explicitly():
    processor = _processor()
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    with pytest.raises(ValueError, match=r"metadata\.coral"):
        processor.ingest(
            AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.REPORT,
                agent_record_id="r-bare",
                references=("i1",),
                payload={"score": 1.0, "references": ["i1"]},
            )
        )
    assert {"i1", "r-bare"} <= processor.retention_decision().protected_agent_record_ids


def test_group_size_floor():
    with pytest.raises(ValueError, match="at least two"):
        _processor(group_size=1)


def test_status_is_a_mapping_even_before_any_discard():
    """Regression: a private-name collision with the base class made
    status() call .items() on the base's discard set."""
    processor = _processor()
    status = processor.status()
    assert status == {
        "discarded_groups": [],
        "terminal_call_fallbacks": 0,
        "fallback_calls_kept": 0,
        "fallback_calls_total": 0,
        "release_truncations": 0,
    }


def test_forked_attempt_trains_on_its_longest_linear_suffix():
    """A fork early in the call sequence drops only the calls before it."""
    processor = _processor(group_size=2)
    # A fork is a divergence larger than the realign window (1024 tokens), the
    # way a compacted history or a re-rendered system prompt shows up.
    a, b = [1] * 1100, [9] * 1100
    processor.ingest(_inference("i1", a + [3], [1], [-0.1]))
    processor.ingest(_inference("i2", b + [4], [1], [-0.2]))  # diverges from i1: a fork
    processor.ingest(_inference("i3", b + [4, 7, 5], [1], [-0.3]))  # extends i2
    processor.ingest(_inference("i4", b + [4, 7, 5, 8, 6], [1], [-0.4]))  # extends i3
    processor.ingest(_inference("i5", [5, 6], [1], [-0.5]))
    processor.ingest(_attempt_report("r1", ("i1", "i2", "i3", "i4"), 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i5", 0.8, commit="c-b", parent="p0"))
    assert processor.ready()
    (group,) = trajectory_groups(processor.build_batch())
    kept = next(s for s in group if trajectory_reward(s) == 0.5)
    assert kept.training["turn_count"] == 3  # i2, i3, i4; i1 dropped
    status = processor.status()
    assert status["terminal_call_fallbacks"] == 1
    assert (status["fallback_calls_kept"], status["fallback_calls_total"]) == (3, 4)


def test_fork_at_the_last_call_keeps_only_that_call():
    processor = _processor(group_size=2)
    a, b = [1] * 1100, [9] * 1100
    processor.ingest(_inference("i1", a + [3], [1], [-0.1]))
    processor.ingest(_inference("i2", b + [4], [1], [-0.2]))  # fork right before the graded call
    processor.ingest(_inference("i3", [5, 6], [1], [-0.3]))
    processor.ingest(_attempt_report("r1", ("i1", "i2"), 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i3", 0.8, commit="c-b", parent="p0"))
    (group,) = trajectory_groups(processor.build_batch())
    kept = next(s for s in group if trajectory_reward(s) == 0.5)
    assert list(kept.training["tokens"]) == b + [4]
    assert processor.status()["fallback_calls_kept"] == 1


def test_group_by_release_groups_a_linear_lineage():
    """Single-agent evolution is a chain: with group_by=parent no group ever fills."""
    processor = _processor(group_size=2, group_by="release")
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1]))
    processor.ingest(_inference("i2", [3, 4], [1], [-0.2]))
    processor.ingest(_attempt_report("r1", "i1", 0.3, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i2", 0.9, commit="c-b", parent="c-a"))  # child of the first
    assert processor.ready()
    (group,) = trajectory_groups(processor.build_batch())
    assert sorted(trajectory_reward(sample) for sample in group) == [0.3, 0.9]


def test_group_by_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _processor(group_by="commit")


def test_group_by_agent_keeps_agents_apart():
    processor = _processor(group_size=2, group_by="agent")
    for i in range(1, 5):
        processor.ingest(_inference(f"i{i}", [i, i + 1], [1], [-0.1]))
    processor.ingest(_attempt_report("r1", "i1", 0.3, commit="c-a", parent="p0", agent="alpha"))
    processor.ingest(_attempt_report("r2", "i2", 0.9, commit="c-b", parent="p1", agent="beta"))
    assert not processor.ready()  # different agents never share a group
    processor.ingest(_attempt_report("r3", "i3", 0.5, commit="c-c", parent="c-a", agent="alpha"))
    assert processor.ready()
    (group,) = trajectory_groups(processor.build_batch())
    assert sorted(trajectory_reward(sample) for sample in group) == [0.3, 0.5]


def test_attempt_spanning_a_weight_update_trains_on_the_new_release_suffix():
    processor = _processor(group_size=2)
    processor.ingest(_inference("i1", [1, 2], [1], [-0.1], release="wv-1"))
    processor.ingest(_inference("i2", [1, 2, 3, 4], [1], [-0.2], release="wv-2"))  # after the update
    processor.ingest(_inference("i3", [5, 6], [1], [-0.3], release="wv-2"))
    processor.ingest(_attempt_report("r1", ("i1", "i2"), 0.5, commit="c-a", parent="p0"))
    processor.ingest(_attempt_report("r2", "i3", 0.8, commit="c-b", parent="p0"))
    assert processor.ready()
    (group,) = trajectory_groups(processor.build_batch())
    cut = next(s for s in group if trajectory_reward(s) == 0.5)
    assert list(cut.training["tokens"]) == [1, 2, 3, 4]
    assert processor.status()["release_truncations"] == 1
