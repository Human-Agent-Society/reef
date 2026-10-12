"""SDPO's full-step barrier, feedback selection and backend wire contract.

Torch/ray free like ``test_sdft_pipeline.py``: the tokenizer is a fake
installed as ``transformers.AutoTokenizer``; the kernels are pinned in
``test_sdpo_parity.py``.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sdpo import SDPOProcessor, SDPORecipe, SDPOReport
from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
from reef.core import AgentRecord, RequestType
from reef.recipe.errors import RecipeConfigError
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill.algorithm import settings_from_args
from reef.train.types import TrainingBatch

from .test_sdft_pipeline import CountingTokenizer, _inference


class SDPOTokenizer(CountingTokenizer):
    """The counting tokenizer, accepting the thinking switch SDPO hands to the chat template."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None = None,
        *,
        enable_thinking: bool = False,
        **options: Any,
    ) -> list[int] | dict[str, list[int]]:
        return super().apply_chat_template(conversation, tools, **options)


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> SDPOTokenizer:
    fake = SDPOTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake))
    return fake


def processor(**config: Any) -> SDPOProcessor:
    return SDPOProcessor(
        ProcessorContext(
            "science",
            {
                "groups_per_step": 1,
                "rollouts_per_group": 2,
                "tokenizer_path": "/models/test",
                "max_teacher_tokens": 18432,
                **config,
            },
            SDPOReport,
        )
    )


def ingest(
    target: SDPOProcessor,
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
    inference = _inference(name, messages=question)
    # Each rollout's recorded response names it, so a demonstration is traceable to its sibling.
    response = {"choices": [{"message": {"role": "assistant", "content": f"<think>hmm</think>Answer from {name}"}}]}
    inference = replace(inference, payload={**inference.payload, "response": response})
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
def test_waits_for_entire_grid_and_uses_successful_sibling(tokenizer: SDPOTokenizer) -> None:
    target = processor(groups_per_step=2)
    ingest(target, 1, score=0.0)
    ingest(target, 0, group=1, score=0.0)
    ingest(target, 1, group=1, score=0.0)
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    ingest(target, 0, score=1.0)
    batch = target.build_batch()
    assert isinstance(batch, TrainingBatch)
    assert batch.batch_id == "science:sdpo:1"
    assert [s.training["distill_sample_weight"] for s in batch.items] == [0.0, 1.0, 0.0, 0.0]
    # The demonstration is the sibling's recorded response with its thinking block removed.
    assert tokenizer.calls[1][0][-1]["content"].endswith(
        "Correct solution:\n\nAnswer from s0g0r0\n\nCorrectly solve the original question."
    )
    assert "Correct solution" not in tokenizer.calls[0][0][-1]["content"]
    for sample in batch.items:
        assert list(sample.training["teacher_tokens"][-3:]) == list(sample.training["tokens"][-3:])
        assert list(sample.training["loss_mask"]) == [1, 1, 1]


@pytest.mark.unit
def test_a_rollout_reads_its_own_success_only_when_allowed(tokenizer: SDPOTokenizer) -> None:
    target = processor(allow_own_success_as_demonstration=True)
    ingest(target, 0, score=1.0)
    ingest(target, 1, score=0.0)
    batch = target.build_batch()
    assert [s.training["distill_sample_weight"] for s in batch.items] == [1.0, 1.0]
    assert "Answer from s0g0r0" in tokenizer.calls[0][0][-1]["content"]


@pytest.mark.unit
def test_first_report_wins_a_retry_slot(tokenizer: SDPOTokenizer) -> None:
    target = processor()
    ingest(target, 0, score=0.0)
    ingest(target, 0, score=1.0, suffix="retry")
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    ingest(target, 1, score=0.0)
    batch = target.build_batch()
    assert len(batch.items) == 2
    assert all(sample.training["distill_sample_weight"] == 0.0 for sample in batch.items)


@pytest.mark.unit
@pytest.mark.parametrize("release", [None, "slime-v4"])
def test_rejects_steps_without_one_known_policy_release(tokenizer: SDPOTokenizer, release: str | None) -> None:
    target = processor()
    ingest(target, 0)
    ingest(target, 1, release=release)
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    assert target.status()["failed_steps"] == [{"step": 0, "reason": "missing_or_mixed_release_ids"}]


@pytest.mark.unit
def test_rejects_different_questions_in_one_group(tokenizer: SDPOTokenizer) -> None:
    target = processor()
    ingest(target, 0)
    ingest(target, 1, question=[{"role": "user", "content": "A different question"}])
    with pytest.raises(RuntimeError, match="batch is not ready"):
        target.build_batch()
    assert target.status()["failed_steps"][0]["reason"] == "different_requests_in_group"


@pytest.mark.unit
@pytest.mark.parametrize("include_feedback", [False, True])
def test_environment_feedback_is_opt_in(tokenizer: SDPOTokenizer, include_feedback: bool) -> None:
    target = processor(include_environment_feedback=include_feedback)
    ingest(target, 0, feedback="The tool call is missing a required argument.")
    ingest(target, 1)
    batch = target.build_batch()
    assert [s.training["distill_sample_weight"] for s in batch.items] == [float(include_feedback), 0.0]
    assert ("missing a required argument" in tokenizer.calls[0][0][-1]["content"]) == include_feedback


@pytest.mark.unit
def test_solution_can_take_priority_over_feedback_and_prompt_truncates_right(tokenizer: SDPOTokenizer) -> None:
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
def test_teacher_overflow_fails_without_silently_dropping_part_of_a_step(tokenizer: SDPOTokenizer) -> None:
    target = processor(max_teacher_prompt_tokens=2, max_teacher_tokens=3)
    ingest(target, 0)
    ingest(target, 1)
    with pytest.raises(ValueError, match="exceeds max_teacher_tokens"):
        target.build_batch()


@pytest.mark.unit
def test_the_prompt_window_sits_inside_the_teacher_window(tokenizer: SDPOTokenizer) -> None:
    with pytest.raises(ValueError, match="max_teacher_prompt_tokens"):
        processor(max_teacher_prompt_tokens=0)
    # The shared processor's 0 disables its window; SDPO fails a step on overflow, so it needs one.
    with pytest.raises(ValueError, match="max_teacher_prompt_tokens"):
        processor(max_teacher_tokens=0)


@pytest.mark.unit
def test_sdpo_recipe_uses_one_actual_sized_step() -> None:
    recipe = SDPORecipe(tokenizer_path="/models/test", **runtime_bindings(StubTrainingRuntime()))
    assert recipe.report_type is SDPOReport
    assert recipe.groups_per_step == 32 and recipe.rollouts_per_group == 8
    assert recipe.allow_own_success_as_demonstration is False
    assert recipe.processor_config()["max_teacher_tokens"] == 18432
    assert recipe.training_spec().scheduling == StepScheduling(unit="sample", batch_size="actual")
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        SDPORecipe.from_environment(
            {},
            config={"data": {"tokenizer_path": "/models/test"}, "optimization": {"divergence": "forward"}},
            **runtime_bindings(StubTrainingRuntime()),
        )


def wire(weights: list[float]) -> dict[str, Any]:
    rows = [
        ["one", [9, 1, 2], [1, 1], [-0.1, -0.2], 1.0, [7, 1, 2], weights[0]],
        ["two", [8, 3], [1], [-0.3], 0.0, [7, 3], weights[1]],
    ]
    return {"loss": "sdpo", "samples": rows, "rollout_ids": [0, 1]}


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
    assert family.rollout_data_keys == ("teacher_tokens", "distill_sample_weights")


@pytest.mark.unit
@pytest.mark.parametrize("remove_thinking", [False, True])
def test_sdpo_episode_demonstrates_full_sibling_and_preserves_every_suffix_position(
    tokenizer: SDPOTokenizer, remove_thinking: bool
) -> None:
    from .test_distill_processor import episode_inferences

    target = processor(accept_multi_turn_policy_samples=True, remove_thinking_from_demonstration=remove_thinking)
    all_references = []
    for rollout in (1, 0):
        records = episode_inferences(f"sibling-{rollout}")
        references = tuple(record.agent_record_id for record in records)
        all_references.extend(references)
        for record in records:
            target.ingest(record)
        report_id = f"report-{rollout}"
        all_references.append(report_id)
        target.ingest(
            AgentRecord.create(
                scenario="science",
                request_type=RequestType.REPORT,
                agent_record_id=report_id,
                payload=SDPOReport(step=0, group=0, rollout=rollout, score=float(rollout == 0)).to_dict(
                    references=references
                ),
                references=references,
            )
        )
        if rollout == 1:
            assert not target.ready()
    batch = target.build_batch()
    assert len(batch.items) == 2
    assert [sample.training["distill_sample_weight"] for sample in batch.items] == [0.0, 1.0]
    demonstration_prompt = tokenizer.calls[1][0][-1]["content"]
    assert "Action sibling-0-0" in demonstration_prompt
    assert "Action sibling-0-1" in demonstration_prompt
    assert "Action sibling-0-2" in demonstration_prompt
    assert "Observation sibling-0-0" in demonstration_prompt
    assert "Observation sibling-0-1" in demonstration_prompt
    assert "python" in demonstration_prompt and "1+1" in demonstration_prompt
    assert ("private" in demonstration_prompt) is (not remove_thinking)
    assert "sibling-1" not in demonstration_prompt
    for sample in batch.items:
        assert sample.training["turn_count"] == 3
        assert sample.training["teacher_tokens"][-9:] == sample.training["tokens"][-9:]
        assert sample.training["loss_mask"] == [1, 1, 1, 0, 0, 1, 1, 0, 1]
    assert target.build_batch() is batch
    assert target.acknowledge(batch.batch_id) == frozenset(all_references)
    assert not target.ready()


@pytest.mark.unit
def test_sdpo_episode_rejects_prompt_truncation(tokenizer: SDPOTokenizer) -> None:
    target = processor(accept_multi_turn_policy_samples=True, max_teacher_prompt_tokens=2)
    ingest(target, 0, score=1.0)
    ingest(target, 1)
    with pytest.raises(ValueError, match="truncation is not allowed"):
        target.build_batch()
    assert target.releasable_record_ids() == frozenset()


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [False, True])
def test_sdpo_recipe_exposes_existing_episode_opt_in(enabled: bool) -> None:
    recipe = SDPORecipe.from_environment(
        {},
        config={"data": {"tokenizer_path": "/models/test", "accept_multi_turn_policy_samples": enabled}},
        **runtime_bindings(StubTrainingRuntime()),
    )
    assert recipe.processor_config()["accept_multi_turn_policy_samples"] is enabled


@pytest.mark.unit
def test_sdpo_opt_in_keeps_single_turn_results_identical(tokenizer: SDPOTokenizer) -> None:
    batches = []
    calls = []
    for enabled in (False, True):
        target = processor(accept_multi_turn_policy_samples=enabled)
        ingest(target, 0, score=1.0)
        ingest(target, 1)
        batches.append(target.build_batch())
        calls.append(tokenizer.calls[-2:])
    assert calls[0] == calls[1]
    assert [sample.training for sample in batches[0].items] == [sample.training for sample in batches[1].items]


@pytest.mark.unit
@pytest.mark.parametrize("turn_index", [0, 1, 2])
def test_sdpo_checks_release_of_every_episode_turn(tokenizer: SDPOTokenizer, turn_index: int) -> None:
    from .test_distill_processor import episode_inferences

    target = processor(accept_multi_turn_policy_samples=True)
    records = list(episode_inferences())
    records[turn_index] = replace(
        records[turn_index], artifact_ref=replace(records[turn_index].artifact_ref, release_id="slime-v4")
    )
    for record in records:
        target.ingest(record)
    references = tuple(record.agent_record_id for record in records)
    with pytest.raises(ValueError, match="one policy release"):
        target.ingest(
            AgentRecord.create(
                scenario="science",
                request_type=RequestType.REPORT,
                payload=SDPOReport(step=0, group=0, rollout=0, score=1.0).to_dict(references=references),
                references=references,
            )
        )
    assert not target.ready()
    assert target.releasable_record_ids() == frozenset()
