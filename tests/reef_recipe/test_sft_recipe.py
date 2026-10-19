"""The built-in SFT recipe: the supervised sample, the stages, the recipe, and the Slime wire payload.

Everything here is torch/ray free. The tokenizer is a fake installed as
``transformers.AutoTokenizer``, so neither transformers nor model files are needed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from tests.support.runtime_stubs import StubTrainingRuntime, runtime_bindings

from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.recipe.sft import SftObjective, SFTRecipe, SFTRecord2Record, SFTSettings, SFTTasks
from reef.runtime.interfaces import TrainingMethod
from reef.task2batch import DistillBatches
from reef.train import PipelineContext
from reef.train.algos import StepScheduling, registry
from reef.train.algos.registry import ObjectiveRegistry, resolve_objective
from reef.train.pipeline import TrainingPipeline
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.slime_backend.sft import SftAlgorithm

TOKENIZER_PATH = "/models/served"
GENERATION_PROMPT = 900
END_OF_TURN = 901
QUESTION = [{"role": "user", "content": "What is the boiling point of water?"}]


class TurnTokenizer:
    """A chat template with one ID per character, a generation prompt, and an end-of-turn ID.

    An assistant turn renders as the generation prompt, its content, its tool
    call names, and the end-of-turn ID, so the generation prompt is a prefix
    of the conversation with the assistant turn.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None, bool]] = []

    def from_pretrained(self, path: str, **options: Any) -> TurnTokenizer:
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
        self.calls.append((list(conversation), tools, add_generation_prompt))
        ids: list[int] = []
        for message in conversation:
            if message["role"] == "assistant":
                ids.append(GENERATION_PROMPT)
            ids.extend(self.text_ids(message))
            if message["role"] == "assistant":
                ids.append(END_OF_TURN)
        if add_generation_prompt:
            ids.append(GENERATION_PROMPT)
        return ids if not return_dict else {"input_ids": ids}

    @staticmethod
    def text_ids(message: Mapping[str, Any]) -> list[int]:
        text = str(message.get("content") or "")
        text += "".join(call["function"]["name"] for call in message.get("tool_calls") or ())
        return [ord(character) for character in text]


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> TurnTokenizer:
    fake = TurnTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake))
    return fake


def _inference(agent_record_id: str, response: Mapping[str, Any] | None = None) -> AgentRecord:
    payload: dict[str, Any] = {
        "messages": QUESTION,
        "response": response or {"choices": [{"message": {"role": "assistant", "content": "Cold."}}]},
        # The served model's own sample: SFT with a demonstration ignores it.
        "tokens": [5, 6, 7],
        "loss_mask": [1],
        "rollout_log_probs": [-0.5],
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


def _report(agent_record_id: str, references: tuple[str, ...], teacher_context: str = "100 C") -> AgentRecord:
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=TeacherContextReport(teacher_context=teacher_context).to_dict(references=references),
        agent_record_id=agent_record_id,
        references=references,
    )


def _pipeline(**config: Any) -> TrainingPipeline:
    context = PipelineContext("science", TeacherContextReport)
    return TrainingPipeline(
        context,
        record2record=SFTRecord2Record(context),
        record2task=SFTTasks(SFTSettings(tokenizer_path=TOKENIZER_PATH, **config)),
        task2batch=DistillBatches(context, 1),
    )


def _prompt_ids() -> list[int]:
    return [*TurnTokenizer.text_ids(QUESTION[0]), GENERATION_PROMPT]


# --- the supervised sample ----------------------------------------------------


@pytest.mark.unit
def test_the_demonstration_is_the_trained_assistant_turn(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))

    (sample,) = pipeline.build_batch().items

    assert source_record_id(sample) == "i1"
    assert sample.source_agent_record_ids == ("i1", "r1")
    target_ids = [ord(character) for character in "100 C"] + [END_OF_TURN]
    # The recorded sample is replaced: the prompt, then the demonstration's tokens, which alone carry loss.
    assert sample.training["tokens"] == [*_prompt_ids(), *target_ids]
    assert sample.training["loss_mask"] == [1] * len(target_ids)
    assert sample.training["rollout_log_probs"] == []
    assert tokenizer.calls[-1][0][-1] == {"role": "assistant", "content": "100 C"}


@pytest.mark.unit
def test_the_recorded_response_is_the_target_of_an_imported_example(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline(target="response")
    tool_call = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "water"}'}}
    response = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}]}
    pipeline.ingest(_inference("i1", response))
    pipeline.ingest(_report("r1", ("i1",), teacher_context=""))

    (sample,) = pipeline.build_batch().items

    target_ids = [ord(character) for character in "lookup"] + [END_OF_TURN]
    assert sample.training["tokens"] == [*_prompt_ids(), *target_ids]
    assert sample.training["loss_mask"] == [1] * len(target_ids)
    # The tool call renders with its arguments as an object, as chat templates expect.
    rendered_call = tokenizer.calls[-1][0][-1]["tool_calls"][0]
    assert rendered_call["function"]["arguments"] == {"q": "water"}


@pytest.mark.unit
def test_a_sample_over_max_tokens_is_skipped_and_counted(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline(max_tokens=len(_prompt_ids()) + 2)
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))

    assert not pipeline.ready()
    assert pipeline.record2task.operational_metrics() == {"sft_overflow_reports": 1}
    assert pipeline.record2record.releasable_record_ids() == frozenset({"i1", "r1"})


@pytest.mark.unit
def test_a_broken_training_contract_raises(tokenizer: TurnTokenizer) -> None:
    demonstration = _pipeline()
    demonstration.ingest(_inference("i1"))
    with pytest.raises(ValueError, match="teacher_context"):
        demonstration.ingest(_report("r1", ("i1",), teacher_context=""))
    response = _pipeline(target="response")
    response.ingest(_inference("i2", {"output_text": "no choices"}))
    with pytest.raises(ValueError, match="recorded response message"):
        response.ingest(_report("r2", ("i2",)))


@pytest.mark.unit
def test_a_template_that_rewrites_the_generation_prompt_is_refused(
    tokenizer: TurnTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = SFTTasks(SFTSettings(tokenizer_path=TOKENIZER_PATH))
    rendered = {True: [1, 2, 3], False: [1, 4, 3, 5]}
    monkeypatch.setattr(
        tokenizer, "apply_chat_template", lambda conversation, **options: rendered[options["add_generation_prompt"]]
    )

    with pytest.raises(ValueError, match="prefix of the assistant turn"):
        tasks.sequence_ids([*QUESTION, {"role": "assistant", "content": "x"}], None)


@pytest.mark.unit
def test_a_held_sample_never_becomes_stale(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))
    # Two later versions: with max_staleness 0 a sampled policy row would be dropped.
    pipeline.observe_serving_runtime_load_id("slime-v4")
    pipeline.observe_serving_runtime_load_id("slime-v5")

    assert pipeline.ready()
    assert len(pipeline.build_batch().items) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tokenizer_path": ""}, "tokenizer_path"),
        ({"tokenizer_path": TOKENIZER_PATH, "target": "teacher"}, "target"),
        ({"tokenizer_path": TOKENIZER_PATH, "max_tokens": -1}, "max_tokens"),
    ],
)
def test_settings_reject_invalid_values(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SFTSettings(**config)


# --- recipe -------------------------------------------------------------------


PIPELINE = {"record2task": {"config": {"tokenizer_path": TOKENIZER_PATH}}}


@pytest.mark.unit
def test_sft_recipe_resolves_by_dotted_reference_and_builds_its_stages(tokenizer: TurnTokenizer) -> None:
    reference = "reef.recipe.sft:SFTRecipe"
    assert recipe_class_for(reference) is SFTRecipe

    recipe = build_recipe(
        reference,
        {},
        {
            "pipeline": {
                "record2task": {"config": {"tokenizer_path": TOKENIZER_PATH, "target": "response", "max_tokens": 8}},
                "task2batch": {"config": {"batch_size": 64}},
            }
        },
        **runtime_bindings(StubTrainingRuntime()),
    )

    assert isinstance(recipe, SFTRecipe)
    assert recipe.name == "sft"
    assert recipe.report_type is TeacherContextReport
    spec = SFTRecipe.training_spec()
    assert (spec.objective, spec.loss_family) == ("reef.recipe.sft:SftObjective", "sft")
    assert spec.scheduling == StepScheduling(unit="sample")
    pipeline = recipe.build_pipeline(PipelineContext("science", TeacherContextReport))
    assert isinstance(pipeline.record2record, SFTRecord2Record)
    assert isinstance(pipeline.record2task, SFTTasks)
    assert (pipeline.record2task.target, pipeline.record2task.max_tokens) == ("response", 8)
    assert isinstance(pipeline.task2batch, DistillBatches)
    assert pipeline.task2batch.batch_size == 64


@pytest.mark.unit
def test_sft_recipe_rejects_a_missing_tokenizer_and_objective_options() -> None:
    def sft_recipe(config: Mapping[str, Any]) -> SFTRecipe:
        recipe = SFTRecipe.from_environment({}, config=config, **runtime_bindings(StubTrainingRuntime()))
        assert isinstance(recipe, SFTRecipe)
        return recipe

    with pytest.raises(RecipeConfigError, match="tokenizer_path"):
        sft_recipe({}).build_pipeline(PipelineContext("science", TeacherContextReport))
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        sft_recipe({"pipeline": PIPELINE, "optimization": {"label_smoothing": 0.1}})


# --- objective and Slime wire payload ----------------------------------------


@pytest.mark.unit
def test_objective_trains_every_batch_without_advantages_and_allows_several_epochs(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))

    signal = SftObjective().prepare(pipeline.build_batch(), {"steps": 2})

    assert (signal.action, signal.next_algorithm_state, signal.advantages) == ("train", {"steps": 3}, None)
    assert signal.metrics == {"steps": 3, "samples": 1}
    SftObjective().validate_scheduling(StepScheduling(unit="sample", epochs=3))
    # Another recipe selects the objective for a job by its dotted reference.
    assert isinstance(resolve_objective("reef.recipe.sft:SftObjective"), SftObjective)


@pytest.mark.unit
def test_the_recipe_objective_resolves_where_the_recipe_was_never_imported(
    tokenizer: TurnTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Slime bridge never imports the recipe module, so no objective is registered there.
    monkeypatch.setattr(registry, "OBJECTIVES", ObjectiveRegistry())
    pipeline = _pipeline()
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))

    step = prepare_slime_step(
        pipeline.build_batch(), TrainingMethod(SFTRecipe.training_spec().objective), {}, StepScheduling(unit="sample")
    )

    assert step.payload is not None
    assert step.payload["loss"] == "sft"


@pytest.mark.unit
def test_slime_payload_trains_the_target_tokens_without_log_probs(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.ingest(_inference("i1"))
    pipeline.ingest(_report("r1", ("i1",)))

    payload = prepare_slime_step(
        pipeline.build_batch(), TrainingMethod("sft"), {}, StepScheduling(unit="sample")
    ).payload

    assert payload is not None
    assert payload["loss"] == "sft"
    assert "advantages" not in payload
    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    target_length = len("100 C") + 1
    assert data["response_lengths"] == [target_length]
    assert data["loss_masks"] == [[1] * target_length]
    assert "rollout_log_probs" not in data
    family = resolve_loss_family("sft")
    assert isinstance(family, SftAlgorithm)
    assert (family.loss_type, family.requires_rollout_logprobs) == ("sft_loss", False)
