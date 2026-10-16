"""SampleAssembly: episodes whose calls stop extending each other (#825).

A context compaction rebuilds the transcript from a summary, so the next
call's prompt no longer extends the previous prompt and reply. Assembly
refuses the episode; the training pipeline decides what the report is worth.
"""

from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.record2record import ReportContext
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings


def inference(record_id: str, tokens: list[int], response_length: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        payload={
            "response": {
                "training": {
                    "tokens": tokens,
                    "loss_mask": [1] * response_length,
                    "rollout_log_probs": [-0.1] * response_length,
                    "runtime_load_id": "wv-1",
                }
            }
        },
    )


def report(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        agent_record_id=record_id,
        references=references,
        payload={"score": 1.0, "references": list(references)},
    )


def assembly() -> SampleAssembly:
    return SampleAssembly(SampleAssemblySettings(accept_multi_turn_policy_samples=True))


@pytest.mark.unit
def test_an_episode_whose_calls_extend_each_other_assembles() -> None:
    context = ReportContext(
        report("r1", "i1", "i2"),
        1.0,
        (inference("i1", [1, 2, 3, 4, 5], 2), inference("i2", [1, 2, 3, 4, 5, 6, 7], 2)),
    )
    sample = assembly().build(context, 1.0)
    assert sample.training["turn_count"] == 2


@pytest.mark.unit
def test_a_compacted_episode_cannot_assemble() -> None:
    # The second call's prompt restarts from a compaction summary: it shares no
    # prefix with the first call's transcript, so the episode forks and the
    # assembly refuses it with a training data error.
    context = ReportContext(
        report("r1", "i1", "i2"),
        1.0,
        (inference("i1", [1, 2, 3, 4, 5], 2), inference("i2", [9, 9, 9, 7], 1)),
    )
    with pytest.raises(ValueError, match="cannot assemble"):
        assembly().build(context, 1.0)
