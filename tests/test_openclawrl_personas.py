"""Unit tests for the student persona preference checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.openclawrl.examples.openclawrl.user_sim.personas import (
    STUDENT,
    student_style_violations,
    student_violations,
)

CLEAN_REPLY = """Sure, here is the working.

First we figure out how many apples are left after lunch, that's 15 minus 6
which gives 9. Then the 3 rotten ones go away too, leaving 9 minus 3 equals 6.
So the answer is 6 apples.
"""


@pytest.mark.unit
def test_student_accepts_plain_prose_with_steps() -> None:
    assert STUDENT.prefers(CLEAN_REPLY)
    assert student_style_violations(CLEAN_REPLY) == []
    assert student_violations(CLEAN_REPLY) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "42",
        "The answer is 42.",
        "Reading homework. Ready to help.",
        "Sure, give me a second and I will work through it.",
    ],
)
def test_student_flags_contentless_replies_as_no_shown_work(reply: str) -> None:
    # Style-clean but workless replies must not satisfy the criterion.
    assert student_style_violations(reply) == []
    assert student_violations(reply) == ["no-shown-work"]
    assert not STUDENT.prefers(reply)


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "The answer is **42** because of the following.",
        "# Solution\nwe add the numbers",
        "Steps:\n1. add\n2. subtract",
        "Here you go:\n- first item\n- second item",
        "So we get \\boxed{42}.",
        "Final answer: 42",
        "| a | b |\n|---|---|",
    ],
)
def test_student_flags_ai_style_markers(reply: str) -> None:
    assert not STUDENT.prefers(reply)
    assert student_style_violations(reply)


@pytest.mark.unit
def test_student_does_not_flag_inline_math_or_asterisk_multiplication() -> None:
    reply = "we do 3 * 4 which is 12, then 12 + 5 makes 17, so 17 is the answer"
    assert STUDENT.prefers(reply)


class _FakeWorker:
    def __init__(self) -> None:
        self._verdicts = []

    def submit(self, job) -> bool:
        return True

    def poll(self):
        judgments, self._verdicts = self._verdicts, []
        return judgments

    def close(self) -> None:
        pass

    def push(self, judgment) -> None:
        self._verdicts.append(judgment)


def _judged_turns(processor, worker, versions: list[str], score: float = 1.0) -> None:
    """One single-turn session per version entry, judged with ``score``."""
    from recipes.openclawrl.turns import TurnJudgment
    from reef.core.records_types import AgentRecord, RequestType

    for index, version in enumerate(versions):
        first = [{"role": "user", "content": f"q-{index}"}]
        for receipt, messages in (
            (f"i{index}", first),
            (f"i{index}-next", [*first, {"role": "assistant", "content": "a"}, {"role": "user", "content": "ok"}]),
        ):
            processor.ingest(
                AgentRecord.create(
                    scenario="s",
                    request_type=RequestType.INFERENCE,
                    payload={
                        "messages": messages,
                        "tools": [{"type": "function", "function": {"name": "noop"}}],
                        "response": {
                            "choices": [{"message": {"role": "assistant", "content": "a"}}],
                            "training": {
                                "tokens": [1, 2, 3],
                                "loss_mask": [0, 1],
                                "rollout_log_probs": [-0.3, -0.2],
                                "topk_indices": [[1, 2], [3, 4]],
                                "topk_log_probs": [[-0.1, -0.2], [-0.3, -0.4]],
                                "runtime_load_id": version,
                            },
                        },
                    },
                    agent_record_id=receipt,
                )
            )
        worker.push(
            TurnJudgment(
                f"i{index}",
                score=score,
                # The un-enhanced anchor: native ids verbatim.
                teacher_cands=({"hint": "", "teacher_tokens": [1, 2, 3]},),
            )
        )


@pytest.mark.unit
def test_openclawrl_processor_drops_unfillable_stale_version_candidates() -> None:
    from recipes.openclawrl import OpenClawRLProcessor
    from reef.train.types import ProcessorContext

    worker = _FakeWorker()
    processor = OpenClawRLProcessor(ProcessorContext("s", {"batch_size": 3}), worker=worker)
    # Two stale leftovers from before a weight swap, then a fresh generation.
    _judged_turns(processor, worker, ["v1", "v1", "v2", "v2", "v2"])

    batch = processor.build_batch()
    versions = {sample.runtime_load_id for sample in batch.samples}
    assert versions == {"v2"}
    assert len(batch.samples) == 3


@pytest.mark.unit
def test_openclawrl_processor_keeps_single_version_queue_intact() -> None:
    from recipes.openclawrl import OpenClawRLProcessor
    from reef.train.types import ProcessorContext

    worker = _FakeWorker()
    processor = OpenClawRLProcessor(ProcessorContext("s", {"batch_size": 3}), worker=worker)
    _judged_turns(processor, worker, ["v1", "v1"])
    # All-one-version queue below batch size: nothing is dropped, not ready yet.
    assert not processor.ready()
    assert len(processor.retention_decision().releasable_agent_record_ids) == 0


@pytest.mark.unit
def test_student_counts_money_and_unit_arithmetic_as_shown_work() -> None:
    """GSM8K work is full of $ and units; the shown-work rule must see through
    them (a money-laden solution once counted as no work at all, putting
    reward 1.0 out of reach for a whole class of tasks)."""
    reply = (
        "The original price per balloon is $900 divided by 20 which is $45. "
        "After the increase each one costs $45 plus $20, so $65. "
        "Filling 170 balloons means 170 times $65, and that comes to $11,050."
    )
    assert STUDENT.prefers(reply)
    assert student_violations(reply) == []
