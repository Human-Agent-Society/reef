"""Reef-side SDFT pipeline: the demonstration composition, the recipe and the Slime wire payload.

Everything here is torch/ray free so it runs in the minimal CI gate; the
shared processor is covered in ``test_distill_processor.py`` and the tensor
kernels in ``test_distill_parity.py``. The record-to-sample stage renders
nothing; ``CharTemplate`` stands in for the training component's chat template
to check the teacher sequence the component renders from a sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from tests.support.chat_templates import CharTemplate
from tests.support.runtime_stubs import runtime_bindings

from recipes.sdft import SdftObjective, SDFTRecipe
from recipes.sdft.processor import DEFAULT_CONTEXT_TEMPLATE, SDFTSamples, SDFTSettings
from recipes.sdft.slime import SdftSettings
from reef.artifact.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import TeacherRequest, rendered_job_data
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.record2sample.distill import student_call
from reef.runtime.interfaces import TrainingMethod
from reef.sample2job import DistillJobData
from reef.train import PipelineContext
from reef.train.algos.schedule import StepScheduling
from reef.train.pipeline import TrainingPipeline
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData

QUESTION = "What is the boiling point of water?"
REPLY = "About 90 degrees."
TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "description": "Look up a fact.", "parameters": {"type": "object"}},
    }
]
#: A Slime wire row's student columns: three prompt ids and three response ids.
STUDENT_TOKENS = (5, 6, 7, 1, 2, 3)
STUDENT_LOSS_MASK = (1, 1, 1)
STUDENT_LOG_PROBS = (-0.1, -0.2, -0.3)


def inference_record(
    agent_record_id: str, *, messages: list[dict[str, Any]] | None = None, tools: list[dict[str, Any]] | None = None
) -> AgentRecord:
    """A served call: the request and the reply, with no training tensors."""
    payload: dict[str, Any] = {
        "messages": messages or [{"role": "user", "content": QUESTION}],
        "tools": TOOLS if tools is None else tools,
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


def sdft_pipeline(**config: Any) -> TrainingPipeline:
    context = PipelineContext("science", TeacherContextReport)
    return TrainingPipeline(
        context,
        record2sample=SDFTSamples(context, SDFTSettings(**config)),
        sample2job=DistillJobData(context, 1, None),
    )


def one_job_data(processor: TrainingPipeline, inference: AgentRecord, teacher_context: str) -> JobData:
    processor.push(inference)
    processor.push(report_record("r1", (inference.agent_record_id,), teacher_context))
    return processor.build_job_data(1)


# --- the teacher's request --------------------------------------------------


@pytest.mark.unit
def test_the_demonstration_block_is_added_to_the_final_user_message() -> None:
    request = [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Question?"}]},
    ]

    batch = one_job_data(sdft_pipeline(), inference_record("i1", messages=request), "The answer.")

    (sample,) = batch.samples
    assert sample.metadata["teacher"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Question?\n\n" + DEFAULT_CONTEXT_TEMPLATE.replace("{context}", "The answer.")},
    ]
    assert sample.metadata["teacher"]["tools"] == TOOLS
    # The student's own request is left as recorded.
    assert student_call(sample).messages[-1] == {"role": "user", "content": "Question?"}


@pytest.mark.unit
def test_the_block_becomes_a_user_message_after_a_tool_result() -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
    request = [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    processor = sdft_pipeline(context_template="Demonstration: {context}")

    batch = one_job_data(processor, inference_record("i1", messages=request, tools=[]), "call bash with cat a.txt")

    teacher = batch.samples[0].metadata["teacher"]
    assert teacher["messages"][-1] == {"role": "user", "content": "Demonstration: call bash with cat a.txt"}
    assert teacher["messages"][:3] == [
        {"role": "user", "content": "Run it."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{**call, "function": {"name": "bash", "arguments": {"cmd": "ls"}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    assert teacher["tools"] is None


@pytest.mark.unit
def test_the_template_needs_the_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{context\}"):
        sdft_pipeline(context_template="no placeholder")


@pytest.mark.unit
def test_processor_emits_one_sample_with_the_teacher_request() -> None:
    batch = one_job_data(sdft_pipeline(), inference_record("i1"), "100 degrees Celsius.")

    assert isinstance(batch, JobData)
    assert batch.data_id == "science:data:1"
    (sample,) = batch.samples
    assert source_record_id(sample) == "i1"
    assert "tokens" not in sample.training
    teacher = TeacherRequest.of_sample(sample)
    assert teacher is not None
    assert (teacher.tools, teacher.max_prompt_tokens, teacher.max_tokens) == (TOOLS, 0, 0)
    assert teacher.messages is not None
    (last,) = teacher.messages
    assert last["role"] == "user"
    assert last["content"].startswith(f"{QUESTION}\n\n")
    assert "100 degrees Celsius." in last["content"]
    # The component renders the teacher request with the generation prompt, then the student's reply tokens verbatim.
    template = CharTemplate()
    rendered, left_out = rendered_job_data(batch, template)
    assert left_out == ()
    (row,) = rendered.samples
    reply_ids = row.training["tokens"][-len(row.training["loss_mask"]) :]
    assert row.training["teacher_tokens"] == [*template.render(teacher.messages, TOOLS, prompt=True), *reply_ids]


# --- recipe -----------------------------------------------------------------


def sdft_recipe(config: Mapping[str, Any]) -> SDFTRecipe:
    recipe = SDFTRecipe.from_environment({}, config=config, **runtime_bindings())
    assert isinstance(recipe, SDFTRecipe)
    return recipe


@pytest.mark.unit
def test_sdft_recipe_resolves_by_dotted_reference() -> None:
    reference = "recipes.sdft.recipe:SDFTRecipe"
    assert recipe_class_for(reference) is SDFTRecipe

    recipe = build_recipe(reference, {}, {}, **runtime_bindings())

    assert isinstance(recipe, SDFTRecipe)
    assert recipe.name == "sdft"
    assert recipe.report_type is TeacherContextReport
    spec = SDFTRecipe.training_spec()
    assert (spec.objective, spec.loss_family) == ("sdft", "sdft")
    assert SdftObjective.rollout_unit == "sample"
    pipeline = recipe.build_pipeline(PipelineContext("science", TeacherContextReport))
    assert isinstance(pipeline.record2sample, SDFTSamples)
    assert pipeline.record2sample.exclusive_sources, "each report owns its inference records"
    assert isinstance(pipeline.sample2job, DistillJobData)
    assert pipeline.sample2job.buffer_size is None, "without a trainer default the buffer holds every sample"
    assert pipeline.record2sample.max_teacher_tokens == 0
    assert pipeline.record2sample.template == DEFAULT_CONTEXT_TEMPLATE


@pytest.mark.unit
def test_sdft_recipe_hands_each_stage_its_pipeline_config() -> None:
    recipe = sdft_recipe(
        {
            "pipeline": {
                "record2sample": {"config": {"max_teacher_tokens": 24000, "context_template": "Example: {context}"}},
                "sample2job": {"config": {"job_samples": 4, "buffer_size": 16}},
            },
        }
    )

    pipeline = recipe.build_pipeline(PipelineContext("science", TeacherContextReport))
    assert isinstance(pipeline.record2sample, SDFTSamples)
    assert (pipeline.sample2job.job_samples, pipeline.sample2job.buffer_size) == (4, 16)
    assert pipeline.record2sample.max_teacher_tokens == 24000
    assert pipeline.record2sample.template == "Example: {context}"


@pytest.mark.unit
def test_sdft_recipe_validates_the_demonstration_template_when_the_pipeline_builds() -> None:
    recipe = sdft_recipe({"pipeline": {"record2sample": {"config": {"context_template": "bare"}}}})
    with pytest.raises(RecipeConfigError, match=r"context_template must contain \{context\}"):
        recipe.build_pipeline(PipelineContext("science", TeacherContextReport))


@pytest.mark.unit
def test_sdft_recipe_refuses_a_tokenizer_path_and_backend_objective_config() -> None:
    # The training component renders with its own chat template; the stage has no tokenizer to configure.
    recipe = sdft_recipe({"pipeline": {"record2sample": {"config": {"tokenizer_path": "/models/served"}}}})
    with pytest.raises(RecipeConfigError, match="unknown config fields: tokenizer_path"):
        recipe.build_pipeline(PipelineContext("science", TeacherContextReport))
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        sdft_recipe({"optimization": {"kl_direction": "reverse"}})


@pytest.mark.unit
def test_sdft_recipe_refuses_stage_settings_in_the_recipe_config() -> None:
    # A recipe config key that a stage reads is refused: the stage reads it from the pipeline section.
    with pytest.raises(RecipeConfigError, match="does not consume"):
        sdft_recipe({"data": {"batch_size": 4}})
    with pytest.raises(RecipeConfigError, match="does not consume"):
        SDFTRecipe.service_config({"context-template": "Example: {context}"}, model_path="/models/served")
    # A key in the wrong stage fails when the pipeline builds.
    recipe = sdft_recipe({"pipeline": {"record2sample": {"config": {"batch_size": 4}}}})
    with pytest.raises(RecipeConfigError, match="batch_size"):
        recipe.build_pipeline(PipelineContext("science", TeacherContextReport))


# --- objective and Slime wire payload ----------------------------------------


@pytest.mark.unit
def test_objective_trains_every_batch_and_refuses_multiple_epochs() -> None:
    objective = SdftObjective()
    batch = one_job_data(sdft_pipeline(), inference_record("i1"), "100 degrees Celsius.")

    signal = objective.prepare(batch, {"steps": 2})

    assert (signal.action, signal.next_algorithm_state, signal.advantages) == ("train", {"steps": 3}, None)
    assert signal.metrics == {"steps": 3, "samples": 1}
    with pytest.raises(ValueError, match="epochs"):
        objective.validate_scheduling(StepScheduling(epochs=2))


@pytest.mark.unit
def test_slime_payload_carries_the_teacher_sequence_beside_the_policy_row() -> None:
    batch = one_job_data(sdft_pipeline(), inference_record("i1"), "100 degrees Celsius.")
    rendered, left_out = rendered_job_data(batch, CharTemplate())
    assert left_out == ()
    (sample,) = rendered.samples
    tokens, loss_mask, teacher_tokens = (sample.training[key] for key in ("tokens", "loss_mask", "teacher_tokens"))

    prepared = prepare_slime_job(rendered, TrainingMethod("sdft"), {}, StepScheduling(), step_size=1)
    payload = prepared.payload
    assert payload is not None
    assert payload["loss"] == "sdft"
    assert "advantages" not in payload
    # No rollout log-probabilities: Slime recomputes them with the training weights.
    assert payload["samples"] == [["i1", tokens, loss_mask, [], 0.0, teacher_tokens, 1.0]]

    data = to_slime_rollout_data(payload)
    assert data["loss"] == "sdft"
    assert data["tokens"] == [tokens]
    assert data["response_lengths"] == [len(REPLY)]
    assert "rollout_log_probs" not in data
    assert data["teacher_tokens"] == [teacher_tokens]
    # Every SDFT sample weighs the same in the step's mean.
    assert data["distill_sample_weights"] == [1.0]


def wire_payload(teacher_tokens: list[Any], **overrides: Any) -> dict[str, Any]:
    row = ["i1", list(STUDENT_TOKENS), list(STUDENT_LOSS_MASK), list(STUDENT_LOG_PROBS), 0.0, teacher_tokens, 1.0]
    return {"samples": [row], "rollout_ids": [0], "loss": "sdft", "external_step_sizes": [1], **overrides}


@pytest.mark.unit
def test_slime_payload_rejects_a_teacher_sequence_that_does_not_end_with_the_response() -> None:
    with pytest.raises(ValueError, match="end with the student's response ids"):
        to_slime_rollout_data(wire_payload([9, 9, 1, 2, 4]))
    with pytest.raises(ValueError, match="carry a prompt"):
        to_slime_rollout_data(wire_payload([1, 2, 3]))
    with pytest.raises(ValueError, match="sequence of integers"):
        to_slime_rollout_data(wire_payload([9, "1", 2, 3]))
    with pytest.raises(ValueError, match="must be \\[source_id"):
        to_slime_rollout_data(
            {
                "samples": [["i1", [5, 1], [1], [-0.1], 0.0]],
                "rollout_ids": [0],
                "loss": "sdft",
                "external_step_sizes": [1],
            }
        )
    with pytest.raises(ValueError, match="must omit advantages"):
        to_slime_rollout_data(wire_payload([9, 1, 2, 3], advantages=[1.0]))


@pytest.mark.unit
def test_sdft_driver_options_travel_from_argv_onto_args() -> None:
    family = resolve_loss_family("sdft")
    settings, remaining = family.parse_driver_options(
        [
            "--sdft-divergence=reverse",
            "--sdft-importance-sampling-cap=0",
            "--sdft-skip-response-tokens=3",
            "--sdft-teacher-update-rate=1",
            "--lr=1",
        ]
    )

    assert settings == SdftSettings(
        divergence="reverse", importance_sampling_cap=0.0, skip_response_tokens=3, teacher_update_rate=1.0
    )
    assert remaining == ["--lr=1"]
    args = SimpleNamespace()
    family.apply_driver_options(args, settings)
    assert args.loss_family == "sdft"
    # The base stamps the settings under distill_* names, whatever the family's prefix.
    assert (args.distill_divergence, args.distill_importance_sampling_cap, args.distill_skip_response_tokens) == (
        "reverse",
        0.0,
        3,
    )
    assert (args.distill_teacher, args.distill_teacher_update_rate, args.distill_top_k) == ("self", 1.0, 0)

    defaults = SimpleNamespace()
    family.apply_driver_options(defaults, None)
    assert (
        defaults.distill_divergence,
        defaults.distill_importance_sampling_cap,
        defaults.distill_skip_response_tokens,
    ) == (
        "forward",
        2.0,
        0,
    )
    # The reference's ref_model_mixup_alpha: the teacher copy moves toward the policy by 1% per step.
    assert (defaults.distill_teacher, defaults.distill_teacher_update_rate, defaults.distill_top_k) == (
        "self",
        0.01,
        0,
    )
    assert family.bind(settings) is family
    with pytest.raises(TypeError, match="SdftSettings"):
        family.bind(SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("divergence", "hellinger"),
        ("teacher", "oracle"),
        ("importance_sampling_cap", -1.0),
        ("importance_sampling_cap", float("nan")),
        ("skip_response_tokens", -1),
        ("teacher_update_rate", 1.5),
        ("teacher_update_rate", -0.1),
    ],
)
def test_sdft_settings_reject_invalid_values(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        SdftSettings(**{name: value})


@pytest.mark.unit
def test_sdft_backend_validation_pins_the_loss_type_and_one_step_per_rollout() -> None:
    family = resolve_loss_family("sdft")
    accepted = {"loss_type": "custom_loss", "use_rollout_logprobs": False, "num_steps_per_rollout": 1}

    def _args(**changes: Any) -> SimpleNamespace:
        # The driver stamps the family's settings on args before it validates them.
        args = SimpleNamespace(**{**accepted, **changes})
        family.apply_driver_options(args, None)
        return args

    family.validate_backend_args(_args())

    with pytest.raises(RuntimeError, match="loss-type custom_loss"):
        family.validate_backend_args(_args(loss_type="policy_loss"))
    with pytest.raises(RuntimeError, match="remove --use-rollout-logprobs"):
        family.validate_backend_args(_args(use_rollout_logprobs=True))
    with pytest.raises(RuntimeError, match="num-steps-per-rollout"):
        family.validate_backend_args(_args(num_steps_per_rollout=2))
