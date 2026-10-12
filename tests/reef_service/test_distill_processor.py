"""The shared distillation processor: the student's rollout plus the teacher's prompt, one sample per report.

Torch/ray free. The tokenizer is a fake installed as ``transformers.AutoTokenizer``
that counts tokens deterministically, so neither transformers nor model files
are needed; a test subclass stands in for a recipe's.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.train import ProcessorContext
from reef.train.processors import DistillProcessor
from reef.train.processors.common import recorded_response
from reef.train.types import TrainingBatch

STUDENT_TOKENS = (5, 6, 7, 1, 2, 3)  # three prompt ids, three response ids
STUDENT_LOSS_MASK = (1, 1, 1)
STUDENT_LOG_PROBS = (-0.1, -0.2, -0.3)
QUESTION = "What is the boiling point of water?"


class CountingTokenizer:
    """The served model's tokenizer: one token per message plus one per ten characters of text.

    It records where it was loaded from and what it rendered, and like
    transformers 5 returns a mapping unless ``return_dict=False``.
    """

    def __init__(self) -> None:
        self.loaded: list[tuple[str, dict[str, Any]]] = []
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None]] = []

    def from_pretrained(self, path: str, **options: Any) -> CountingTokenizer:
        self.loaded.append((path, options))
        return self

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None = None,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
    ) -> list[int] | dict[str, list[int]]:
        self.calls.append((list(conversation), tools))
        ids = self.count_ids(conversation)
        return ids if not return_dict else {"input_ids": ids}

    @staticmethod
    def count_ids(messages: Sequence[Mapping[str, Any]]) -> list[int]:
        text = "".join(str(message.get("content") or "") for message in messages)
        return [100 + index for index in range(len(messages) + len(text) // 10)]


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> CountingTokenizer:
    fake = CountingTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake))
    return fake


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


def _processor(**config: Any) -> DistillProcessor:
    return DistillProcessor(
        ProcessorContext(
            "science", {"batch_size": 1, "tokenizer_path": "/models/science", **config}, TeacherContextReport
        )
    )


@pytest.mark.unit
def test_by_default_the_teacher_reads_the_request_as_recorded(tokenizer: CountingTokenizer) -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",), teacher_context=""))

    batch = processor.build_batch()

    assert isinstance(batch, TrainingBatch)
    (sample,) = batch.items
    assert source_record_id(sample) == "i1"
    assert tokenizer.loaded == [("/models/science", {"trust_remote_code": True})]
    rendered, tools = tokenizer.calls[0]
    assert rendered == [{"role": "user", "content": QUESTION}]
    assert tools == [{"type": "function", "function": {"name": "lookup"}}]
    # The teacher sequence is the rendered prompt plus the response ids verbatim.
    prompt_ids = tokenizer.count_ids(rendered)
    assert list(sample.training["teacher_tokens"]) == [*prompt_ids, 1, 2, 3]
    assert list(sample.training["tokens"]) == list(STUDENT_TOKENS)
    assert batch.batch_id == "science:teacher:1"
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0


@pytest.mark.unit
def test_a_recipe_composes_the_teacher_request_from_the_response_and_the_context(tokenizer: CountingTokenizer) -> None:
    processor = FeedbackProcessor(
        ProcessorContext("science", {"batch_size": 1, "tokenizer_path": "/models/science"}, TeacherContextReport)
    )
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",), teacher_context="Too low."))

    batch = processor.build_batch()

    rendered, tools = tokenizer.calls[0]
    assert rendered[0] == {"role": "system", "content": "You answered: About 90 degrees.\nVerifier: Too low."}
    assert rendered[1:] == [{"role": "user", "content": QUESTION}]
    assert tools is None
    assert list(batch.items[0].training["teacher_tokens"]) == [*tokenizer.count_ids(rendered), 1, 2, 3]
    assert batch.batch_id == "science:feedback:1"


@pytest.mark.unit
def test_the_processor_skips_and_counts_a_teacher_sequence_over_the_window(tokenizer: CountingTokenizer) -> None:
    long_request = _inference("i1")
    short_request = _inference("i2", messages=[{"role": "user", "content": "q"}])
    # The window admits the short request's teacher sequence and not the long one's.
    window = len(tokenizer.count_ids(short_request.payload["messages"])) + 3
    processor = _processor(max_teacher_tokens=window)
    processor.ingest(long_request)
    processor.ingest(_report("r1", ("i1",)))

    assert not processor.ready()
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1
    # The report and its inference are released for compaction.
    assert {"r1", "i1"} <= processor.releasable_record_ids()

    # A later report that fits still trains.
    processor.ingest(short_request)
    processor.ingest(_report("r2", ("i2",), teacher_context="ok"))
    assert len(processor.build_batch().items) == 1
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1


@pytest.mark.unit
def test_teacher_tokens_render_the_request_cut_the_prompt_and_append_the_response(
    tokenizer: CountingTokenizer,
) -> None:
    processor = _processor()
    messages = [{"role": "user", "content": QUESTION}]
    prompt_ids = tokenizer.count_ids(messages)

    assert processor.teacher_tokens(messages, None, (1, 2, 3)) == [*prompt_ids, 1, 2, 3]
    # A prompt window cuts the rendered prompt on the right and leaves the response whole.
    assert processor.teacher_tokens(messages, None, (1, 2, 3), max_prompt_tokens=2) == [*prompt_ids[:2], 1, 2, 3]


@pytest.mark.unit
def test_the_processor_requires_one_recorded_request_per_report(tokenizer: CountingTokenizer) -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_inference("i2"))

    with pytest.raises(ValueError, match="one recorded request per report"):
        processor.ingest(_report("r1", ("i1", "i2")))


@pytest.mark.unit
def test_the_processor_requires_the_tokenizer_path_and_a_valid_window() -> None:
    with pytest.raises(ValueError, match="tokenizer_path"):
        _processor(tokenizer_path="")
    with pytest.raises(ValueError, match="max_teacher_tokens"):
        _processor(max_teacher_tokens=-1)


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


def episode_inferences(prefix: str = "i") -> tuple[AgentRecord, ...]:
    """Three recorded assistant turns with exact tool observations between them."""
    messages = [{"role": "user", "content": QUESTION}]
    tokens = [5, 6, 7]
    records = []
    for index, (response_ids, context_ids) in enumerate((([1, 2, 3], [20, 21]), ([4, 5], [22]), ([6], []))):
        response: dict[str, Any] = {"role": "assistant", "content": f"<think>private</think>Action {prefix}-{index}"}
        if index < 2:
            response["tool_calls"] = [
                {
                    "id": f"call-{prefix}-{index}",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code":"1+1"}'},
                }
            ]
        tokens = [*tokens, *response_ids]
        base = _inference(f"{prefix}{index + 1}", messages=list(messages))
        payload = {
            "messages": [{"role": "user", "content": "not the authoritative request"}],
            "response": {
                "training": {
                    "request_messages": list(messages),
                    "request_tools": base.payload["tools"],
                    "response_message": response,
                    "tokens": list(tokens),
                    "loss_mask": [1] * len(response_ids),
                    "rollout_log_probs": [-value / 10 for value in response_ids],
                    "runtime_load_id": "slime-v3",
                }
            },
        }
        records.append(replace(base, payload=payload))
        messages = [*messages, response]
        if context_ids:
            messages.append(
                {"role": "tool", "tool_call_id": f"call-{prefix}-{index}", "content": f"Observation {prefix}-{index}"}
            )
            tokens.extend(context_ids)
    return tuple(records)


@pytest.mark.unit
def test_episode_keeps_all_selected_positions_and_reserved_sources(tokenizer: CountingTokenizer) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = episode_inferences()
    references = tuple(record.agent_record_id for record in records)
    report = _report("episode-report", references)
    for record in records:
        target.ingest(record)
    target.ingest(report)
    target.ingest(report)
    batch = target.build_batch()
    assert target.build_batch() is batch
    (sample,) = batch.items
    assert source_record_id(sample) == "episode-report"
    assert sample.source_agent_record_ids == (*references, "episode-report")
    assert sample.training["turn_count"] == 3
    assert sample.training["tokens"] == [5, 6, 7, 1, 2, 3, 20, 21, 4, 5, 22, 6]
    assert sample.training["loss_mask"] == [1, 1, 1, 0, 0, 1, 1, 0, 1]
    assert sample.training["rollout_log_probs"] == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, 0.0, -0.6]
    assert sum(sample.training["loss_mask"]) == 6
    assert tokenizer.calls[0][0] == [{"role": "user", "content": QUESTION}]
    assert sample.training["teacher_tokens"][-9:] == sample.training["tokens"][-9:]
    assert target.releasable_record_ids() == frozenset()
    target.release_batch(batch.batch_id)
    retry = target.build_batch()
    assert retry.items == batch.items
    assert target.acknowledge(retry.batch_id) == frozenset((*references, "episode-report"))
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset((*references, "episode-report"))
    target.ingest(_report("duplicate-source", references))
    assert not target.ready()
    target.release_records(target.releasable_record_ids())
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
@pytest.mark.parametrize("field", ["realign_threshold", "scaffold_tolerance"])
def test_episode_rejects_alignment_tolerances(tokenizer: CountingTokenizer, field: str) -> None:
    with pytest.raises(ValueError, match="requires realign_threshold=0 and scaffold_tolerance=0"):
        _processor(accept_multi_turn_policy_samples=True, **{field: 1})


@pytest.mark.unit
@pytest.mark.parametrize("turn_index", [0, 1, 2])
@pytest.mark.parametrize(
    "fault", ["mask", "log_probs", "nonfinite", "release", "runtime", "missing_release", "mixed_spans"]
)
def test_episode_rejects_invalid_turns_without_releasing_sources(
    tokenizer: CountingTokenizer, turn_index: int, fault: str
) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = list(episode_inferences())
    record = records[turn_index]
    training = dict(record.payload["response"]["training"])
    artifact_ref = record.artifact_ref
    if fault == "mask":
        training["loss_mask"] = [0, *training["loss_mask"][1:]]
    elif fault == "log_probs":
        training["rollout_log_probs"] = []
    elif fault == "nonfinite":
        training["rollout_log_probs"] = [float("nan"), *training["rollout_log_probs"][1:]]
    elif fault == "release":
        artifact_ref = replace(artifact_ref, release_id="slime-v4")
    elif fault == "runtime":
        training["runtime_load_id"] = "slime-v4"
    elif fault == "missing_release":
        artifact_ref = None
    else:
        training["runtime_load_id"] = None
        response_length = len(training["loss_mask"])
        training["runtime_load_spans"] = [{"start": 0, "end": response_length, "runtime_load_id": "slime-v4"}]
    records[turn_index] = replace(record, artifact_ref=artifact_ref, payload={"response": {"training": training}})
    for record in records:
        target.ingest(record)
    report = _report("invalid-episode", tuple(record.agent_record_id for record in records))
    with pytest.raises(ValueError, match="episode"):
        target.ingest(report)
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
@pytest.mark.parametrize("position", [0, 6, 8])
def test_episode_rejects_fork_response_drift_and_scaffold_drift(tokenizer: CountingTokenizer, position: int) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = list(episode_inferences())
    training = dict(records[2].payload["response"]["training"])
    tokens = list(training["tokens"])
    tokens[position] = 999
    training["tokens"] = tokens
    records[2] = replace(records[2], payload={"response": {"training": training}})
    for record in records:
        target.ingest(record)
    with pytest.raises(ValueError, match="cannot assemble"):
        target.ingest(_report("fork", tuple(record.agent_record_id for record in records)))
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
def test_episode_rejects_out_of_order_references(tokenizer: CountingTokenizer) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = episode_inferences()
    for record in records:
        target.ingest(record)
    with pytest.raises(ValueError, match="first assistant turn"):
        target.ingest(_report("reversed", tuple(record.agent_record_id for record in reversed(records))))
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
def test_episode_overflow_raises_and_keeps_all_sources(tokenizer: CountingTokenizer) -> None:
    target = _processor(accept_multi_turn_policy_samples=True, max_teacher_tokens=5)
    records = episode_inferences()
    for record in records:
        target.ingest(record)
    with pytest.raises(ValueError, match="episode teacher sequence exceeds"):
        target.ingest(_report("overflow", tuple(record.agent_record_id for record in records)))
    assert target.operational_metrics()["teacher_overflow_reports"] == 0
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
def test_episode_single_turn_matches_default_exactly(tokenizer: CountingTokenizer) -> None:
    samples = []
    inference = _inference("single")
    report = _report("single-report", ("single",))
    for enabled in (False, True):
        target = _processor(accept_multi_turn_policy_samples=enabled)
        target.ingest(inference)
        target.ingest(report)
        samples.append(target.build_batch().items[0])
    assert samples[0] == samples[1]


@pytest.mark.unit
def test_episode_rejects_missing_assistant_turn_in_references(tokenizer: CountingTokenizer) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = episode_inferences()
    for record in records:
        target.ingest(record)
    with pytest.raises(ValueError, match="include every assistant turn"):
        target.ingest(_report("incomplete", (records[0].agent_record_id, records[2].agent_record_id)))
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
def test_episode_rejects_duplicate_receipts(tokenizer: CountingTokenizer) -> None:
    from reef.core.reports import ReportValidationError

    target = _processor(accept_multi_turn_policy_samples=True)
    record = episode_inferences()[0]
    target.ingest(record)
    with pytest.raises(ReportValidationError, match="unique"):
        target.ingest(_report("duplicate", (record.agent_record_id, record.agent_record_id)))
    assert not target.ready()


@pytest.mark.unit
def test_episode_rejects_missing_initial_receipt(tokenizer: CountingTokenizer) -> None:
    target = _processor(accept_multi_turn_policy_samples=True)
    records = episode_inferences()
    for record in records:
        target.ingest(record)
    with pytest.raises(ValueError, match="start before the first assistant turn"):
        target.ingest(_report("missing-initial", tuple(record.agent_record_id for record in records[1:])))
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset()
