"""The shared distillation processor: the student's rollout plus the teacher's prompt, one sample per report.

Torch/ray free. The tokenizer is a fake that counts tokens deterministically,
so no model files are needed; a test subclass stands in for a recipe's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.train import ProcessorContext
from reef.train.processors import DistillProcessor
from reef.train.processors.common import recorded_response
from reef.train.processors.distill import TeacherPromptTokenizer
from reef.train.types import TrainingBatch

STUDENT_TOKENS = (5, 6, 7, 1, 2, 3)  # three prompt ids, three response ids
STUDENT_LOSS_MASK = (1, 1, 1)
STUDENT_LOG_PROBS = (-0.1, -0.2, -0.3)
QUESTION = "What is the boiling point of water?"


class CountingTokenizer(TeacherPromptTokenizer):
    """One token per message plus one per ten characters of text; records what it rendered."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None]] = []

    def prompt_token_ids(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None) -> list[int]:
        self.calls.append((list(messages), tools))
        text = "".join(str(message.get("content") or "") for message in messages)
        return [100 + index for index in range(len(messages) + len(text) // 10)]


class FeedbackProcessor(DistillProcessor):
    """A recipe's composition: the student's own answer and the teacher context as a system message, no tools."""

    batch_label = "feedback"

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        system = {"role": "system", "content": f"You answered: {response}\nVerifier: {teacher_context}"}
        return [system, *messages], None


def _inference(agent_record_id: str, *, messages: list[dict[str, Any]] | None = None) -> AgentRecord:
    payload: dict[str, Any] = {
        "messages": messages or [{"role": "user", "content": QUESTION}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "About 90 degrees."}}]},
        "tokens": list(STUDENT_TOKENS),
        "loss_mask": list(STUDENT_LOSS_MASK),
        "rollout_log_probs": list(STUDENT_LOG_PROBS),
    }
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=LiveWeightArtifactRef(
            content_id="science", release_id="slime-v3", parent_release_id=None, runtime_load_id="slime-v3"
        ),
    )


def _report(
    agent_record_id: str, references: tuple[str, ...], teacher_context: str = "100 degrees Celsius."
) -> AgentRecord:
    body = TeacherContextReport(teacher_context=teacher_context).to_dict(references=references)
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=body,
        agent_record_id=agent_record_id,
        references=references,
    )


def _processor(tokenizer: CountingTokenizer | None = None, **config: Any) -> DistillProcessor:
    return DistillProcessor(ProcessorContext("science", {"batch_size": 1, **config}, TeacherContextReport), tokenizer)


@pytest.mark.unit
def test_by_default_the_teacher_reads_the_request_as_recorded() -> None:
    tokenizer = CountingTokenizer()
    processor = _processor(tokenizer)
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",), teacher_context=""))

    batch = processor.build_batch()

    assert isinstance(batch, TrainingBatch)
    (sample,) = batch.items
    assert source_record_id(sample) == "i1"
    rendered, tools = tokenizer.calls[0]
    assert rendered == [{"role": "user", "content": QUESTION}]
    assert tools == [{"type": "function", "function": {"name": "lookup"}}]
    # The teacher sequence is the rendered prompt plus the response ids verbatim.
    prompt_ids = tokenizer.prompt_token_ids(rendered, tools)
    assert list(sample.training["teacher_tokens"]) == [*prompt_ids, 1, 2, 3]
    assert list(sample.training["tokens"]) == list(STUDENT_TOKENS)
    assert batch.batch_id == "science:teacher:1"
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0


@pytest.mark.unit
def test_a_recipe_composes_the_teacher_request_from_the_response_and_the_context() -> None:
    tokenizer = CountingTokenizer()
    processor = FeedbackProcessor(ProcessorContext("science", {"batch_size": 1}, TeacherContextReport), tokenizer)
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",), teacher_context="Too low."))

    batch = processor.build_batch()

    rendered, tools = tokenizer.calls[0]
    assert rendered[0] == {"role": "system", "content": "You answered: About 90 degrees.\nVerifier: Too low."}
    assert rendered[1:] == [{"role": "user", "content": QUESTION}]
    assert tools is None
    assert list(batch.items[0].training["teacher_tokens"]) == [*tokenizer.prompt_token_ids(rendered, tools), 1, 2, 3]
    assert batch.batch_id == "science:feedback:1"


@pytest.mark.unit
def test_the_processor_skips_and_counts_a_teacher_sequence_over_the_window() -> None:
    tokenizer = CountingTokenizer()
    long_request = _inference("i1")
    short_request = _inference("i2", messages=[{"role": "user", "content": "q"}])
    # The window admits the short request's teacher sequence and not the long one's.
    window = len(tokenizer.prompt_token_ids(short_request.payload["messages"], None)) + 3
    processor = _processor(tokenizer, max_teacher_tokens=window)
    processor.ingest(long_request)
    processor.ingest(_report("r1", ("i1",)))

    assert not processor.ready()
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1
    # The report and its inference are released for compaction.
    assert {"r1", "i1"} <= processor.retention_decision().releasable_agent_record_ids

    # A later report that fits still trains.
    processor.ingest(short_request)
    processor.ingest(_report("r2", ("i2",), teacher_context="ok"))
    assert len(processor.build_batch().items) == 1
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1


@pytest.mark.unit
def test_the_processor_requires_one_recorded_request_per_report() -> None:
    processor = _processor(CountingTokenizer(), accept_multi_turn_policy_samples=True)
    processor.ingest(_inference("i1"))
    processor.ingest(_inference("i2"))

    with pytest.raises(ValueError, match="one recorded request per report"):
        processor.ingest(_report("r1", ("i1", "i2")))


@pytest.mark.unit
def test_the_processor_requires_the_tokenizer_path_and_a_valid_window() -> None:
    with pytest.raises(ValueError, match="tokenizer_path"):
        _processor()
    with pytest.raises(ValueError, match="max_teacher_tokens"):
        _processor(CountingTokenizer(), max_teacher_tokens=-1)


@pytest.mark.unit
def test_recorded_response_reads_the_final_assistant_message() -> None:
    assert recorded_response(_inference("i1").payload) == "About 90 degrees."
    assert recorded_response({"response": {"choices": [{"text": "plain"}]}}) == "plain"
    assert (
        recorded_response(
            {"response": {"training": {"response_message": {"content": [{"type": "text", "text": "x"}]}}}}
        )
        == "x"
    )
    assert recorded_response({"messages": []}) == ""
