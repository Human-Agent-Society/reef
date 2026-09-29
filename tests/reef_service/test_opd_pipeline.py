"""Same-prefix OPD contracts from captured inference through Slime settings."""

import sys
from types import SimpleNamespace

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.opd import OpdObjective, OPDProcessor, OPDRecipe
from recipes.opd.slime import OpdSettings
from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.recipe.registry import build_recipe
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TrainingBatch

TOKENS = [5, 6, 7, 1, 2, 3]


class UnusedTokenizer:
    """OPD must not re-render a prompt whose exact token IDs are recorded."""

    @classmethod
    def from_pretrained(cls, path: str, **options: object) -> "UnusedTokenizer":
        return cls()

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("OPD must use the recorded prefix, including assistant prefill")


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> OPDProcessor:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=UnusedTokenizer))
    return OPDProcessor(ProcessorContext("math", {"batch_size": 1, "tokenizer_path": "/model"}, TeacherContextReport))


def inference() -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        agent_record_id="i1",
        artifact_ref=LiveWeightArtifactRef(
            content_id="math", release_id="slime-v3", parent_release_id=None, runtime_load_id="slime-v3"
        ),
        payload={
            "messages": [{"role": "user", "content": "Compute 1 + 1."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "tokens": list(TOKENS),
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
        },
    )


def report(context: str = "", score: float = 0.0) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        agent_record_id="r1",
        references=("i1",),
        payload=TeacherContextReport(teacher_context=context, score=score).to_dict(references=("i1",)),
    )


@pytest.mark.unit
@pytest.mark.parametrize("score", [0.0, 1.0])
def test_exact_student_tokens_and_no_scalar_reward_reach_slime(processor: OPDProcessor, score: float) -> None:
    processor.ingest(inference())
    processor.ingest(report(score=score))
    batch = processor.build_batch()
    assert isinstance(batch, TrainingBatch)
    assert batch.batch_id == "math:opd:1"
    prepared = prepare_slime_step(batch, "opd", {}, StepScheduling(unit="sample"))
    payload = prepared.payload
    assert payload is not None
    assert "advantages" not in payload
    assert payload["samples"] == [["i1", TOKENS, [1, 1, 1], [-0.1, -0.2, -0.3], 0.0, TOKENS, 1.0]]
    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    assert data["teacher_tokens"] == data["tokens"] == [TOKENS]
    assert data["response_lengths"] == [3]
    assert data["distill_sample_weights"] == [1.0]


@pytest.mark.unit
def test_privileged_teacher_context_is_rejected(processor: OPDProcessor) -> None:
    processor.ingest(inference())
    with pytest.raises(ValueError, match="teacher_context must be empty"):
        processor.ingest(report(context="The answer is 2."))


@pytest.mark.unit
def test_recipe_resolves_and_rejects_multiple_epochs() -> None:
    recipe = build_recipe(
        "recipes.opd.recipe:OPDRecipe",
        {},
        {"data": {"tokenizer_path": "/model"}},
        **runtime_bindings(StubTrainingRuntime()),
    )
    assert isinstance(recipe, OPDRecipe)
    assert recipe.report_type is TeacherContextReport
    assert recipe.max_staleness == 0
    spec = recipe.training_spec()
    assert (spec.objective, spec.processor, spec.loss_family) == ("opd", OPDProcessor, "opd")
    with pytest.raises(ValueError, match="epochs"):
        OpdObjective().validate_scheduling(StepScheduling(unit="sample", epochs=2))


@pytest.mark.unit
def test_driver_parses_required_checkpoint_before_validating_defaults() -> None:
    family = resolve_loss_family("opd")
    settings, remaining = family.parse_driver_options(["--opd-teacher-checkpoint=/teacher", "--lr=0.00005"])
    assert isinstance(settings, OpdSettings)
    assert remaining == ["--lr=0.00005"]
    assert (settings.teacher, settings.teacher_checkpoint, settings.teacher_update_rate) == (
        "separate",
        "/teacher",
        0.0,
    )
    assert (settings.divergence, settings.top_k, settings.importance_sampling_cap) == ("reverse", 1, 0.0)
    args = SimpleNamespace()
    family.apply_driver_options(args, settings)
    assert args.distill_teacher_checkpoint == "/teacher"
    assert args.distill_divergence == "reverse"
    assert args.distill_top_k == 1
    assert args.distill_importance_sampling_cap == 0.0
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        family.parse_driver_options([])


@pytest.mark.unit
def test_overflow_report_does_not_fill_the_batch(processor: OPDProcessor) -> None:
    limited = OPDProcessor(
        ProcessorContext(
            "math", {"batch_size": 1, "tokenizer_path": "/model", "max_teacher_tokens": 5}, TeacherContextReport
        )
    )
    limited.ingest(inference())
    limited.ingest(report())
    assert not limited.ready()
    assert limited.operational_metrics()["teacher_overflow_reports"] == 1
    limited.ingest(report())
    assert limited.operational_metrics()["teacher_overflow_reports"] == 1
