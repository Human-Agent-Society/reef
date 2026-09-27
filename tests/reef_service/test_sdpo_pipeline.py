"""SDPO's graded rollout group, teacher prompt and Slime family contract."""

from __future__ import annotations

import sys
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from recipes.sdpo import SDPOAttempt, SDPOProcessor, SDPORecipe, prepare_group
from recipes.sdpo.slime import SdpoSettings
from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.recipe.registry import recipe_class_for
from reef.train import ProcessorContext
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family


def attempt(index: int, score: float, *, feedback: str | None = None, response: str | None = None) -> SDPOAttempt:
    return SDPOAttempt("q1", f"i{index}", "v1", response or f"answer {index}", score, feedback)


@pytest.mark.unit
def test_scalar_group_uses_a_successful_sibling_and_masks_uninformed_attempts() -> None:
    prepared = prepare_group([attempt(1, 0), attempt(2, 1), attempt(3, 0)])

    assert [item.inference_id for item in prepared] == ["i1", "i2", "i3"]
    assert [item.demonstration_id for item in prepared] == ["i2", None, "i2"]
    assert all("Correct solution:\n\nanswer 2" in item.teacher_context for item in (prepared[0], prepared[2]))
    assert prepared[1].teacher_context == ""
    (inactive,) = prepare_group([attempt(1, 0)])
    assert inactive.teacher_context == ""
    assert inactive.payload()["metadata"]["sdpo"]["active"] is False


@pytest.mark.unit
def test_rich_feedback_trains_even_without_a_successful_sibling() -> None:
    prepared = prepare_group([attempt(1, 0, feedback="Test 4 failed: expected 8")])

    assert len(prepared) == 1
    assert prepared[0].used_feedback
    assert prepared[0].demonstration_id is None
    assert prepared[0].payload()["references"] == ["i1"]
    assert prepared[0].payload()["metadata"]["sdpo"]["artifact_version"] == "v1"
    assert TeacherContextReport.from_dict(prepared[0].payload()).teacher_context == prepared[0].teacher_context


@pytest.mark.unit
def test_group_validation_rejects_mixed_versions_questions_and_duplicate_receipts() -> None:
    with pytest.raises(ValueError, match="one question and one artifact version"):
        prepare_group([attempt(1, 0), SDPOAttempt("q2", "i2", "v1", "x", 1)])
    with pytest.raises(ValueError, match="one question and one artifact version"):
        prepare_group([attempt(1, 0), SDPOAttempt("q1", "i2", "v2", "x", 1)])
    with pytest.raises(ValueError, match="duplicate inference receipts"):
        prepare_group([attempt(1, 0), attempt(1, 1)])


@pytest.mark.unit
def test_feedback_policy_and_self_success_match_the_reference_configuration() -> None:
    own = attempt(1, 1, feedback="passed", response="<think>trace</think>final")
    other = attempt(2, 0, feedback="failed")
    prepared = prepare_group(
        [own, other], feedback_only_without_solution=True, remove_thinking_from_demonstration=True
    )

    assert [item.inference_id for item in prepared] == ["i1", "i2"]
    assert prepared[0].demonstration_id is None and prepared[0].used_feedback
    assert prepared[1].demonstration_id == "i1" and not prepared[1].used_feedback
    assert "<think>" not in prepared[1].teacher_context
    assert "final" in prepared[1].teacher_context


class Tokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.options: dict[str, Any] = {}

    def from_pretrained(self, path: str, **kwargs: Any) -> Tokenizer:
        return self

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.messages = messages
        self.options = kwargs
        return [10 + index for index, _ in enumerate(messages)]


@pytest.mark.unit
def test_processor_preserves_student_response_ids_and_renders_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    processor = SDPOProcessor(
        ProcessorContext(
            "science", {"batch_size": 1, "tokenizer_path": "/model", "enable_thinking": False}, TeacherContextReport
        )
    )
    inference = AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": "Solve q1"}],
            "response": {"choices": [{"message": {"role": "assistant", "content": "wrong"}}]},
            "tokens": [5, 6, 1, 2],
            "loss_mask": [1, 1],
            "rollout_log_probs": [-0.2, -0.3],
        },
        agent_record_id="i1",
        artifact_ref=LiveWeightArtifactRef(
            content_id="science", release_id="v1", parent_release_id=None, runtime_load_id="v1"
        ),
    )
    prepared = prepare_group([attempt(1, 0, feedback="failed")])[0]
    report = AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=prepared.payload(),
        agent_record_id="r1",
        references=("i1",),
    )
    processor.ingest(inference)
    processor.ingest(report)
    (sample,) = processor.build_batch().items

    assert tokenizer.messages[-1]["content"] == (
        "Solve q1\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
        "failed\n\nCorrectly solve the original question."
    )
    assert tokenizer.options["enable_thinking"] is False
    assert list(sample.training["teacher_tokens"]) == [10, 1, 2]
    assert list(sample.training["tokens"]) == [5, 6, 1, 2]
    family = resolve_loss_family("sdpo")
    row = family.shape_sample_row(sample)
    data = to_slime_rollout_data({"samples": [row], "rollout_ids": [0], "loss": "sdpo"})
    assert data["teacher_tokens"] == [[10, 1, 2]]
    inactive = ["i2", *row[1:6], 0.0]
    mixed = to_slime_rollout_data({"samples": [row, inactive], "rollout_ids": [0, 1], "loss": "sdpo"})
    assert mixed["loss_masks"] == [[1, 1], [1, 1]]
    assert mixed["distill_sample_weights"] == [1.0, 0.0]


@pytest.mark.unit
def test_recipe_and_loss_family_defaults() -> None:
    assert recipe_class_for("recipes.sdpo.recipe:SDPORecipe") is SDPORecipe
    assert SDPORecipe.training_spec().objective == "sdpo"
    family = resolve_loss_family("sdpo")
    settings, remaining = family.parse_driver_options(["--sdpo-top-k=20", "--sdpo-divergence=reverse", "--lr=1e-6"])
    assert settings == SdpoSettings(top_k=20, divergence="reverse")
    assert remaining == ["--lr=1e-6"]
    args = SimpleNamespace()
    family.apply_driver_options(args, settings)
    assert (args.distill_top_k, args.distill_top_k_tail, args.distill_divergence) == (20, True, "reverse")
    assert args.distill_importance_sampling_mode == "token"
    backend = SimpleNamespace(
        loss_type="custom_loss",
        use_rollout_logprobs=True,
        num_steps_per_rollout=1,
        calculate_per_token_loss=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    family.apply_driver_options(backend, settings)
    family.validate_backend_args(backend)
    backend.calculate_per_token_loss = True
    with pytest.raises(RuntimeError, match="calculate-per-token-loss"):
        family.validate_backend_args(backend)
    with pytest.raises(ValueError, match="top_k_tail"):
        SdpoSettings(top_k=0)


@pytest.mark.unit
def test_teacher_overflow_fails_the_fixed_size_group_instead_of_dropping_a_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer()))
    processor = SDPOProcessor(
        ProcessorContext(
            "science", {"batch_size": 8, "tokenizer_path": "/model", "max_teacher_tokens": 2}, TeacherContextReport
        )
    )
    inference = AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": "Question?"}],
            "tokens": [5, 6, 1, 2],
            "loss_mask": [1, 1],
            "rollout_log_probs": [-0.2, -0.3],
        },
        agent_record_id="i1",
    )
    report = AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=TeacherContextReport(teacher_context="feedback").to_dict(references=("i1",)),
        agent_record_id="r1",
        references=("i1",),
    )
    processor.ingest(inference)
    with pytest.raises(ValueError, match="retrying the complete group"):
        processor.ingest(report)
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0


@pytest.mark.unit
@pytest.mark.parametrize("dropout", ["attention_dropout", "hidden_dropout"])
def test_student_topk_requires_deterministic_selection_and_training(dropout: str) -> None:
    family = resolve_loss_family("sdpo")
    settings = {
        "loss_type": "custom_loss",
        "use_rollout_logprobs": True,
        "num_steps_per_rollout": 1,
        "calculate_per_token_loss": False,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
    }
    settings[dropout] = 0.1
    args = Namespace(**settings)
    family.apply_driver_options(args, SdpoSettings())
    with pytest.raises(RuntimeError, match="student top-K selection"):
        family.validate_backend_args(args)


@pytest.mark.unit
def test_teacher_solution_template_matches_reference_without_replacing_prompt_braces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer()))
    processor = SDPOProcessor(ProcessorContext("science", {"tokenizer_path": "/model"}, TeacherContextReport))
    prepared = prepare_group([attempt(1, 0), attempt(2, 1, response="correct")])[0]
    messages, _ = processor.teacher_request(
        [{"role": "user", "content": "Explain {context}."}], None, "wrong", prepared.teacher_context
    )
    # Literal values obtained from actor.yaml's |-, not from our template constants.
    assert messages == [
        {
            "role": "user",
            "content": "Explain {context}.\nCorrect solution:\n\ncorrect\n\nCorrectly solve the original question.",
        }
    ]
