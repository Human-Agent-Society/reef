"""SDPO's full-step barrier, feedback selection and backend wire contract.

Torch/ray free like ``test_sdft_pipeline.py``: the stages render nothing, and
``CharTemplate`` stands in for the training component's chat template; the
kernels are pinned in ``test_sdpo_parity.py``.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import pytest
from tests.support.chat_templates import CharTemplate
from tests.support.runtime_stubs import STUB_ROOT, runtime_bindings

from recipes.sdpo import SDPOJobData, SDPORecipe, SDPOReport, SDPOSamples
from recipes.sdpo.processor import SDPODemonstrationSettings, SDPOSettings
from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
from reef.artifact.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import rendered_job_data
from reef.core.plugins import PluginConfigError
from reef.recipe.errors import RecipeConfigError
from reef.storage.sqlite import SQLiteRecordStore
from reef.train import PipelineContext, Trainer
from reef.train.algos.registry import resolve_objective
from reef.train.pipeline import TrainingPipeline
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill.algorithm import settings_from_args
from reef.train.types import JobData

QUESTION = [{"role": "user", "content": "What is the boiling point of water?"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "description": "Look up a fact.", "parameters": {"type": "object"}},
    }
]


def processor(**config: Any) -> TrainingPipeline:
    """An SDPO pipeline; a test asks it for job data of one grid, so one job is one optimizer step."""
    config = {"groups_per_step": 1, "rollouts_per_group": 2, "max_teacher_tokens": 18432, **config}
    context = PipelineContext("science", SDPOReport)

    def _settings(settings_type: type):
        # Each stage takes its own keys of the shared test config.
        names = {field.name for field in fields(settings_type)}
        return settings_type(**{key: value for key, value in config.items() if key in names})

    record2sample = SDPOSamples(context, _settings(SDPOSettings))
    return TrainingPipeline(
        context,
        record2sample=record2sample,
        sample2job=SDPOJobData(context, _settings(SDPODemonstrationSettings), record2sample),
    )


def reply_text(name: str) -> str:
    """Each rollout's reply names it, so a demonstration is traceable to its sibling."""
    return f"<think>hmm</think>Answer from {name}"


def push(
    target: TrainingPipeline,
    rollout: int,
    *,
    group: int = 0,
    step: int = 0,
    score: float = 0.0,
    feedback: str = "",
    release: str | None = "slime-v3",
    question: list[dict[str, Any]] | None = None,
    suffix: str = "",
) -> None:
    name = f"s{step}g{group}r{rollout}{suffix}"
    target.push(
        AgentRecord.create(
            scenario="science",
            request_type=RequestType.INFERENCE,
            agent_record_id=name,
            artifact_ref=(
                None
                if release is None
                else ArtifactRef(content_id="science", release_id=release, parent_release_id=None)
            ),
            payload={
                "messages": question or QUESTION,
                "tools": TOOLS,
                "response": {"choices": [{"message": {"role": "assistant", "content": reply_text(name)}}]},
            },
        )
    )
    body = SDPOReport(step=step, group=group, rollout=rollout, score=score, teacher_context=feedback).to_dict(
        references=(name,)
    )
    target.push(
        AgentRecord.create(
            scenario="science",
            request_type=RequestType.REPORT,
            agent_record_id=f"report-{name}",
            references=(name,),
            payload=body,
        )
    )


def teacher_prompt(sample_index: int, batch: JobData) -> str:
    """The final user message that the teacher of the batch's sample reads."""
    return batch.samples[sample_index].metadata["teacher"]["messages"][-1]["content"]


@pytest.mark.unit
def test_the_trainer_refuses_job_samples_above_one_grid() -> None:
    def build(job_samples: int) -> Trainer:
        return Trainer.build(
            "science",
            SQLiteRecordStore(),
            pipeline_factory=lambda context: processor(groups_per_step=2, job_samples=job_samples),
            root=STUB_ROOT,
        )

    build(4).close()
    with pytest.raises(
        PluginConfigError, match="SDPOJobData holds at most 4 samples in one job's data, not job_samples 5"
    ):
        build(5)


@pytest.mark.unit
def test_waits_for_entire_grid_and_uses_successful_sibling() -> None:
    target = processor(groups_per_step=2)
    push(target, 1, score=0.0)
    push(target, 0, group=1, score=0.0)
    push(target, 1, group=1, score=0.0)
    with pytest.raises(RuntimeError, match="no job data ready"):
        target.build_job_data(4)
    push(target, 0, score=1.0)
    batch = target.build_job_data(4)
    assert isinstance(batch, JobData)
    assert batch.data_id == "science:data:1"
    assert [s.training["distill_sample_weight"] for s in batch.samples] == [0.0, 1.0, 0.0, 0.0]
    # The demonstration is the sibling's recorded response with its thinking block removed.
    assert teacher_prompt(1, batch).endswith(
        "Correct solution:\n\nAnswer from s0g0r0\n\nCorrectly solve the original question."
    )
    assert "Correct solution" not in teacher_prompt(0, batch)
    for sample in batch.samples:
        teacher = sample.metadata["teacher"]
        assert (teacher["tools"], teacher["max_prompt_tokens"], teacher["max_tokens"]) == (TOOLS, 10240, 18432)
    # The component renders each teacher sequence to end with the student's own reply tokens.
    rendered, left_out = rendered_job_data(batch, CharTemplate())
    assert left_out == ()
    for sample in rendered.samples:
        reply_length = len(sample.training["loss_mask"])
        assert sample.training["teacher_tokens"][-reply_length:] == sample.training["tokens"][-reply_length:]


@pytest.mark.unit
def test_a_rollout_reads_its_own_success_only_when_allowed() -> None:
    target = processor(allow_own_success_as_demonstration=True)
    push(target, 0, score=1.0)
    push(target, 1, score=0.0)
    batch = target.build_job_data(2)
    assert [s.training["distill_sample_weight"] for s in batch.samples] == [1.0, 1.0]
    assert "Answer from s0g0r0" in teacher_prompt(0, batch)


@pytest.mark.unit
def test_first_report_wins_a_retry_slot() -> None:
    target = processor()
    push(target, 0, score=0.0)
    push(target, 0, score=1.0, suffix="retry")
    with pytest.raises(RuntimeError, match="no job data ready"):
        target.build_job_data(2)
    push(target, 1, score=0.0)
    batch = target.build_job_data(2)
    assert len(batch.samples) == 2
    assert all(sample.training["distill_sample_weight"] == 0.0 for sample in batch.samples)


@pytest.mark.unit
@pytest.mark.parametrize("release", [None, "slime-v4"])
def test_rejects_steps_without_one_known_policy_release(release: str | None) -> None:
    target = processor()
    push(target, 0)
    push(target, 1, release=release)
    with pytest.raises(RuntimeError, match="no job data ready"):
        target.build_job_data(2)
    assert target.status()["failed_steps"] == [{"step": 0, "reason": "missing_or_mixed_release_ids"}]


@pytest.mark.unit
def test_rejects_different_questions_in_one_group() -> None:
    target = processor()
    push(target, 0)
    push(target, 1, question=[{"role": "user", "content": "A different question"}])
    with pytest.raises(RuntimeError, match="no job data ready"):
        target.build_job_data(2)
    assert target.status()["failed_steps"][0]["reason"] == "different_requests_in_group"


@pytest.mark.unit
@pytest.mark.parametrize("include_feedback", [False, True])
def test_environment_feedback_is_opt_in(include_feedback: bool) -> None:
    target = processor(include_environment_feedback=include_feedback)
    push(target, 0, feedback="The tool call is missing a required argument.")
    push(target, 1)
    batch = target.build_job_data(2)
    assert [s.training["distill_sample_weight"] for s in batch.samples] == [float(include_feedback), 0.0]
    assert ("missing a required argument" in teacher_prompt(0, batch)) == include_feedback


@pytest.mark.unit
def test_solution_can_take_priority_over_feedback_and_prompt_truncates_right() -> None:
    target = processor(
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
        max_teacher_prompt_tokens=2,
    )
    push(target, 0, score=1.0)
    push(target, 1, feedback="Do not include this feedback when a solution exists.")
    batch = target.build_job_data(2)
    assert "Do not include" not in teacher_prompt(1, batch)
    # The component cuts the rendered teacher prompt to two tokens on the right and keeps the reply whole.
    template = CharTemplate()
    teacher = batch.samples[1].metadata["teacher"]
    prompt = template.render(teacher["messages"], TOOLS, prompt=True)
    rendered, _ = rendered_job_data(batch, template)
    reply_ids = rendered.samples[1].training["tokens"][-len(reply_text("s0g0r1")) :]
    assert len(prompt) > 2
    assert rendered.samples[1].training["teacher_tokens"] == [*prompt[:2], *reply_ids]


@pytest.mark.unit
def test_a_teacher_sequence_over_the_window_leaves_out_its_sample_and_not_the_step() -> None:
    # The window holds the two prompt tokens and the first rollout's reply; the second's reply is longer.
    window = 2 + len(reply_text("s0g0r0"))
    target = processor(max_teacher_prompt_tokens=2, max_teacher_tokens=window)
    push(target, 0)
    push(target, 1, suffix="-longer")
    batch = target.build_job_data(2)
    assert len(batch.samples) == 2
    rendered, left_out = rendered_job_data(batch, CharTemplate())
    assert left_out == (1,)
    assert len(rendered.samples) == 1


@pytest.mark.unit
def test_the_prompt_window_sits_inside_the_teacher_window() -> None:
    with pytest.raises(ValueError, match="max_teacher_prompt_tokens"):
        processor(max_teacher_prompt_tokens=0)
    # The shared processor's 0 disables its window; SDPO's teacher window has no unlimited value.
    with pytest.raises(ValueError, match="max_teacher_prompt_tokens"):
        processor(max_teacher_tokens=0)


def recipe_from(config: Mapping[str, Any]) -> SDPORecipe:
    recipe = SDPORecipe.from_environment({}, config=config, **runtime_bindings())
    assert isinstance(recipe, SDPORecipe)
    return recipe


@pytest.mark.unit
def test_sdpo_recipe_trains_samples_from_the_paper_grid() -> None:
    recipe = recipe_from({})
    assert recipe.report_type is SDPOReport
    assert resolve_objective(recipe.training_spec().objective).rollout_unit == "sample"
    pipeline = recipe.build_pipeline(PipelineContext("science", SDPOReport))
    record2sample, sample2job = pipeline.record2sample, pipeline.sample2job
    assert isinstance(record2sample, SDPOSamples)
    assert isinstance(sample2job, SDPOJobData)
    assert record2sample.step_grouping.groups_per_step == 32 and record2sample.step_grouping.rollouts_per_group == 8
    assert (record2sample.max_teacher_prompt_tokens, record2sample.max_teacher_tokens) == (10240, 18432)
    assert sample2job.allow_own_success_as_demonstration is False
    assert sample2job.max_job_samples == 256, "a batch is at most one grid of 32 groups of 8 rollouts"
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        recipe_from({"optimization": {"divergence": "forward"}})


@pytest.mark.unit
def test_the_pipeline_section_configures_each_stage_and_the_job_stage_reads_the_sample_stage() -> None:
    recipe = recipe_from(
        {
            "pipeline": {
                "record2sample": {
                    "config": {
                        "groups_per_step": 4,
                        "rollouts_per_group": 2,
                        "max_teacher_prompt_tokens": 2048,
                        "max_teacher_tokens": 4096,
                    }
                },
                "sample2job": {"config": {"success_reward_threshold": 1.0, "include_environment_feedback": True}},
            }
        }
    )
    pipeline = recipe.build_pipeline(PipelineContext("science", SDPOReport))
    record2sample, sample2job = pipeline.record2sample, pipeline.sample2job
    assert isinstance(record2sample, SDPOSamples) and isinstance(sample2job, SDPOJobData)
    # The job stage reads the grid and the teacher's windows from the sample stage.
    assert (record2sample.groups_per_step, record2sample.rollouts_per_group) == (4, 2)
    assert sample2job.max_job_samples == 8
    assert (sample2job.max_teacher_prompt_tokens, sample2job.max_teacher_tokens) == (2048, 4096)
    assert (sample2job.success_reward_threshold, sample2job.include_environment_feedback) == (1.0, True)


@pytest.mark.unit
@pytest.mark.parametrize(("key", "value"), [("tokenizer_path", "/models/test"), ("enable_thinking", True)])
def test_the_sample_stage_refuses_rendering_settings(key: str, value: Any) -> None:
    # The training component renders with its own chat template and template arguments.
    recipe = recipe_from({"pipeline": {"record2sample": {"config": {key: value}}}})
    with pytest.raises(RecipeConfigError, match=f"unknown config fields: {key}"):
        recipe.build_pipeline(PipelineContext("science", SDPOReport))


@pytest.mark.unit
def test_the_recipe_refuses_stage_settings_in_the_wrong_place() -> None:
    # A recipe config key that a stage reads is refused: the stage reads it from the pipeline section.
    with pytest.raises(RecipeConfigError, match="does not consume"):
        recipe_from({"data": {"groups_per_step": 4}})
    with pytest.raises(RecipeConfigError, match="does not consume"):
        SDPORecipe.service_config({"tokenizer-path": "/models/test"}, model_path="/models/test")
    # A key in the wrong stage, and an invalid teacher window, fail when the pipeline builds.
    context = PipelineContext("science", SDPOReport)
    with pytest.raises(RecipeConfigError, match="success_reward_threshold"):
        recipe_from({"pipeline": {"record2sample": {"config": {"success_reward_threshold": 1}}}}).build_pipeline(
            context
        )
    with pytest.raises(RecipeConfigError, match="max_teacher_prompt_tokens"):
        recipe_from({"pipeline": {"record2sample": {"config": {"max_teacher_tokens": 0}}}}).build_pipeline(context)


def wire(weights: list[float]) -> dict[str, Any]:
    rows = [
        ["one", [9, 1, 2], [1, 1], [-0.1, -0.2], 1.0, [7, 1, 2], weights[0]],
        ["two", [8, 3], [1], [-0.3], 0.0, [7, 3], weights[1]],
    ]
    return {"loss": "sdpo", "samples": rows, "rollout_ids": [0, 1], "external_step_sizes": [2]}


@pytest.mark.unit
@pytest.mark.parametrize("weights", [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
def test_the_sample_weight_rides_the_shared_wire_row(weights: list[float]) -> None:
    data = to_slime_rollout_data(wire(weights))
    assert data["distill_sample_weights"] == weights
    assert data["loss_masks"] == [[1, 1], [1]]
    assert data["teacher_tokens"] == [[7, 1, 2], [7, 3]]


@pytest.mark.unit
def test_sdpo_settings_roundtrip_and_training_requirements() -> None:
    family = SdpoAlgorithm()
    settings, remaining = family.parse_driver_options(["--sdpo-top-k=20", "--lr=1e-5"])
    args = Namespace(
        num_steps_per_rollout=1, calculate_per_token_loss=False, attention_dropout=0.0, hidden_dropout=0.0
    )
    family.apply_driver_options(args, settings)
    assert remaining == ["--lr=1e-5"]
    actual = settings_from_args(args)
    assert (actual.top_k, actual.top_k_source, actual.top_k_distribution) == (20, "student", "tail")
    assert actual.importance_sampling_level == "token"
    family.validate_specific_args(args, "test")
    # The base refuses dropout while the student selects the support; the family refuses Slime's per-token mean.
    with pytest.raises(RuntimeError, match="dropout"):
        family.validate_specific_args(Namespace(**{**vars(args), "hidden_dropout": 0.1}), "test")
    args.calculate_per_token_loss = True
    with pytest.raises(RuntimeError, match="calculate-per-token-loss"):
        family.validate_specific_args(args, "test")
    assert SdpoSettings().teacher_update_rate == 0.05
    assert family.rollout_data_keys == (
        "teacher_tokens",
        "distill_sample_weights",
        "distill_teacher_topk_ids",
        "distill_teacher_topk_log_probs",
        "distill_teacher_sampled_log_probs",
    )
