"""SDPO's full-step barrier, feedback selection and backend wire contract."""

import sys
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sdpo import SDPOProcessor, SDPORecipe, SDPOReport
from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
from reef.core import AgentRecord, RequestType
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill.algorithm import settings_from_args
from reef.train.types import TrainingBatch

from .test_sdft_pipeline import CountingTokenizer, _inference


class SDPOTokenizer(CountingTokenizer):
    def decode(self, tokens, *, skip_special_tokens):
        return f"Answer {tokens[-1]}"

    def apply_chat_template(self, *args, enable_thinking=False, **kwargs):
        return super().apply_chat_template(*args, **kwargs)


@pytest.fixture
def tokenizer(monkeypatch):
    tokenizer = SDPOTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    return tokenizer


def processor(**config):
    return SDPOProcessor(
        ProcessorContext(
            "science",
            {
                "groups_per_step": 1,
                "rollouts_per_group": 2,
                "tokenizer_path": "/models/test",
                **config,
            },
            SDPOReport,
        )
    )


def ingest(target, rollout, *, group=0, step=0, score=0.0, feedback="", release="slime-v3", question=None, suffix=""):
    name = f"s{step}g{group}r{rollout}{suffix}"
    inference = _inference(name, messages=question)
    if release is None:
        inference = replace(inference, artifact_ref=None)
    elif release != "slime-v3":
        inference = replace(inference, artifact_ref=replace(inference.artifact_ref, release_id=release))
    target.ingest(inference)
    body = SDPOReport(step=step, group=group, rollout=rollout, score=score, teacher_context=feedback).to_dict(
        references=(name,)
    )
    target.ingest(
        AgentRecord.create(
            scenario="science",
            request_type=RequestType.REPORT,
            agent_record_id=f"report-{name}",
            references=(name,),
            payload=body,
        )
    )


@pytest.mark.unit
def test_waits_for_entire_grid_and_uses_successful_sibling(tokenizer):
    target = processor(groups_per_step=2)
    ingest(target, 1, score=0.0)
    ingest(target, 0, group=1, score=0.0)
    ingest(target, 1, group=1, score=0.0)
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    ingest(target, 0, score=1.0)
    batch = target.build_batch()
    assert isinstance(batch, TrainingBatch)
    assert [s.training["distill_sample_mask"] for s in batch.items] == [0, 1, 0, 0]
    assert "Correct solution:\n\nAnswer 3" in tokenizer.calls[1][0][-1]["content"]
    assert "Correct solution" not in tokenizer.calls[0][0][-1]["content"]
    for sample in batch.items:
        assert sample.training["teacher_tokens"][-3:] == sample.training["tokens"][-3:]
        assert sample.training["loss_mask"] == [1, 1, 1]


@pytest.mark.unit
def test_first_report_wins_a_retry_slot(tokenizer):
    target = processor()
    ingest(target, 0, score=0.0)
    ingest(target, 0, score=1.0, suffix="retry")
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    ingest(target, 1, score=0.0)
    batch = target.build_batch()
    assert len(batch.items) == 2
    assert all(sample.training["distill_sample_mask"] == 0 for sample in batch.items)


@pytest.mark.unit
@pytest.mark.parametrize("release", [None, "slime-v4"])
def test_rejects_steps_without_one_known_policy_release(tokenizer, release):
    target = processor()
    ingest(target, 0)
    ingest(target, 1, release=release)
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    assert target.status()["failed_steps"] == [{"step": 0, "reason": "missing_or_mixed_release_ids"}]


@pytest.mark.unit
def test_rejects_different_questions_in_one_group(tokenizer):
    target = processor()
    ingest(target, 0)
    ingest(target, 1, question=[{"role": "user", "content": "A different question"}])
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    assert target.status()["failed_steps"][0]["reason"] == "different_requests_in_group"


@pytest.mark.unit
@pytest.mark.parametrize("include_feedback", [False, True])
def test_environment_feedback_is_opt_in(tokenizer, include_feedback):
    target = processor(include_environment_feedback=include_feedback)
    ingest(target, 0, feedback="The tool call is missing a required argument.")
    ingest(target, 1)
    batch = target.build_batch()
    assert [s.training["distill_sample_mask"] for s in batch.items] == [int(include_feedback), 0]
    assert ("missing a required argument" in tokenizer.calls[0][0][-1]["content"]) == include_feedback


@pytest.mark.unit
def test_solution_can_take_priority_over_feedback_and_prompt_truncates_right(tokenizer):
    target = processor(
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
        max_teacher_prompt_tokens=2,
        max_teacher_tokens=5,
    )
    ingest(target, 0, score=1.0)
    ingest(target, 1, feedback="Do not include this feedback when a solution exists.")
    batch = target.build_batch()
    assert "Do not include" not in tokenizer.calls[1][0][-1]["content"]
    assert list(batch.items[1].training["teacher_tokens"]) == [100, 101, 1, 2, 3]


@pytest.mark.unit
def test_teacher_overflow_fails_without_silently_dropping_part_of_a_step(tokenizer):
    target = processor(max_teacher_prompt_tokens=2, max_teacher_tokens=3)
    ingest(target, 0)
    ingest(target, 1)
    with pytest.raises(ValueError, match="exceeds max_teacher_tokens"):
        target.build_batch()


@pytest.mark.unit
def test_sdpo_recipe_uses_one_actual_sized_step():
    recipe = SDPORecipe(tokenizer_path="/models/test", **runtime_bindings(StubTrainingRuntime()))
    assert recipe.report_type is SDPOReport
    assert recipe.groups_per_step == 32 and recipe.rollouts_per_group == 8
    assert recipe.training_spec().scheduling == StepScheduling(unit="sample", batch_size="actual")


def wire(masks):
    rows = [
        ["one", [9, 1, 2], [1, 1], [-0.1, -0.2], 1.0, [7, 1, 2], masks[0]],
        ["two", [8, 3], [1], [-0.3], 0.0, [7, 3], masks[1]],
    ]
    return {"loss": "sdpo", "samples": rows, "rollout_ids": [0, 1]}


@pytest.mark.unit
@pytest.mark.parametrize(("mask", "weights"), [([1, 1], [1, 1]), ([0, 1], [0, 1]), ([1, 0], [1, 0]), ([0, 0], [0, 0])])
def test_reference_microbatch_normalization_keeps_inactive_rows(mask, weights):
    data = to_slime_rollout_data(wire(mask))
    assert data["distill_sample_weights"] == weights
    assert data["loss_masks"] == [[1, 1], [1]]
    # The reference token-means each one-sample microbatch, then averages all
    # samples. Lengths are unequal and inactive samples still count in N.
    got = sum(value * weight for value, weight in zip([0.6 / 2, 0.5], weights, strict=True)) / 2
    expected = (0.3 * mask[0] + 0.5 * mask[1]) / 2
    assert got == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("mask", [2, -1, True, 0.5, None])
def test_invalid_distillation_masks_fail(mask):
    with pytest.raises(ValueError, match="distill_sample_mask"):
        to_slime_rollout_data(wire([mask, 1]))


@pytest.mark.unit
def test_sdpo_settings_roundtrip_and_training_requirements():
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
    args.calculate_per_token_loss = True
    with pytest.raises(RuntimeError, match="calculate-per-token-loss"):
        family.validate_specific_args(args, "test")
    assert SdpoSettings().teacher_update_rate == 0.05
