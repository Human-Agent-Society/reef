"""The shared distillation processor: the student's call plus the teacher's request, one sample per report.

Torch/ray free. The record-to-sample stage renders nothing: a sample is the student's call as an ATIF trajectory
with the teacher's request in its metadata. ``CharTemplate`` stands in for the training component's chat template
to check the teacher sequence that the component renders; a test subclass stands in for a recipe's composition.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.support.chat_templates import CharTemplate

from reef.artifact.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import TeacherRequest, rendered_job_data
from reef.core.plugins import PluginConfigError
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.record2sample.distill import DistillSamples, DistillSettings
from reef.record2sample.samples import recorded_response
from reef.sample2job import DistillJobData
from reef.train import PipelineContext
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData

QUESTION = "What is the boiling point of water?"
REPLY = "About 90 degrees."
TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "description": "Look up a fact.", "parameters": {"type": "object"}},
    }
]


class FeedbackSamples(DistillSamples):
    """A recipe's composition: the student's own answer and the teacher context as a system message, no tools."""

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        system = {"role": "system", "content": f"You answered: {response}\nVerifier: {teacher_context}"}
        return [system, *messages], None


def inference_record(agent_record_id: str, *, question: str = QUESTION) -> AgentRecord:
    """A served call: the request and the reply, with no training tensors."""
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": question}],
        "tools": TOOLS,
        "response": {"choices": [{"message": {"role": "assistant", "content": REPLY}}]},
    }
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=ArtifactRef(content_id="science", release_id="slime-v3", parent_release_id=None),
    )


def report_record(
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


def distill_pipeline(record2sample_type: type[DistillSamples] = DistillSamples, **config: Any) -> TrainingPipeline:
    context = PipelineContext("science", TeacherContextReport)
    return TrainingPipeline(
        context,
        record2sample=record2sample_type(context, DistillSettings(**config)),
        sample2job=DistillJobData(context, None, None),
    )


@pytest.mark.unit
def test_by_default_the_teacher_reads_the_request_as_recorded() -> None:
    processor = distill_pipeline()
    processor.push(inference_record("i1"))
    processor.push(report_record("r1", ("i1",), teacher_context=""))

    batch = processor.build_job_data(1)

    assert isinstance(batch, JobData)
    assert batch.data_id == "science:data:1"
    (sample,) = batch.samples
    assert source_record_id(sample) == "i1"
    # The stage renders nothing: the sample carries no tokens, only the teacher's request.
    assert "tokens" not in sample.training
    assert sample.metadata["teacher"] == TeacherRequest([{"role": "user", "content": QUESTION}], TOOLS).as_json()
    # The teacher reads what the student read, so the component renders the same sequence for both.
    rendered_data, left_out = rendered_job_data(batch, CharTemplate())
    assert left_out == ()
    (rendered,) = rendered_data.samples
    assert rendered.training["teacher_tokens"] == rendered.training["tokens"]


@pytest.mark.unit
def test_a_recipe_composes_the_teacher_request_from_the_response_and_the_context() -> None:
    processor = distill_pipeline(FeedbackSamples)
    processor.push(inference_record("i1"))
    processor.push(report_record("r1", ("i1",), teacher_context="Too low."))

    batch = processor.build_job_data(1)

    teacher = TeacherRequest.of_sample(batch.samples[0])
    assert teacher is not None
    assert teacher.messages == [
        {"role": "system", "content": f"You answered: {REPLY}\nVerifier: Too low."},
        {"role": "user", "content": QUESTION},
    ]
    assert teacher.tools is None
    # The teacher sequence is the rendered teacher request plus the student's reply tokens verbatim.
    template = CharTemplate()
    (rendered,) = rendered_job_data(batch, template)[0].samples
    reply_ids = rendered.training["tokens"][-len(rendered.training["loss_mask"]) :]
    assert rendered.training["teacher_tokens"] == [*template.render(teacher.messages, None, prompt=True), *reply_ids]


@pytest.mark.unit
def test_a_teacher_sequence_over_the_window_is_left_out_when_the_component_renders() -> None:
    template = CharTemplate()
    short_question = "q"
    # The window admits the short request's teacher sequence and not the long one's.
    short_prompt = template.render([{"role": "user", "content": short_question}], TOOLS, prompt=True)
    window = len(short_prompt) + 1 + len(REPLY)
    processor = distill_pipeline(max_teacher_tokens=window)
    processor.push(inference_record("i1"))
    processor.push(report_record("r1", ("i1",)))
    processor.push(inference_record("i2", question=short_question))
    processor.push(report_record("r2", ("i2",), teacher_context="ok"))

    batch = processor.build_job_data(2)

    assert [sample.metadata["teacher"]["max_tokens"] for sample in batch.samples] == [window, window]
    rendered, left_out = rendered_job_data(batch, template)
    assert left_out == (0,)
    assert [source_record_id(sample) for sample in rendered.samples] == ["i2"]


@pytest.mark.unit
def test_the_processor_requires_one_recorded_request_per_report() -> None:
    processor = distill_pipeline(accept_multi_turn_policy_samples=True)
    processor.push(inference_record("i1"))
    processor.push(inference_record("i2"))

    with pytest.raises(ValueError, match="one recorded request per report"):
        processor.push(report_record("r1", ("i1", "i2")))


@pytest.mark.unit
def test_the_processor_refuses_a_tokenizer_path_and_an_invalid_window() -> None:
    # The training component renders with its own chat template; the stage has no tokenizer to configure.
    with pytest.raises(PluginConfigError, match="unknown config fields: tokenizer_path"):
        DistillSamples.from_config(
            {"tokenizer_path": "/models/science"},
            context=PipelineContext("science", TeacherContextReport),
            environ={},
        )
    with pytest.raises(ValueError, match="max_teacher_tokens"):
        distill_pipeline(max_teacher_tokens=-1)


@pytest.mark.unit
def test_recorded_response_reads_the_final_assistant_message() -> None:
    assert recorded_response(inference_record("i1").payload) == REPLY
    assert recorded_response({"response": {"choices": [{"text": "plain"}]}}) == "plain"
    assert (
        recorded_response(
            {"response": {"training": {"response_message": {"content": [{"type": "text", "text": "x"}]}}}}
        )
        == "x"
    )
    assert recorded_response({"messages": []}) == ""
