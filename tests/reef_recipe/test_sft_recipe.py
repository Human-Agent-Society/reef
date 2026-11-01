"""The built-in SFT recipe: the supervised sample, the stages, the recipe, and the Slime wire payload.

Everything here is torch/ray free. The stages make ATIF samples and load no tokenizer. The training component renders
them with its model's chat template (``rendered_job_data``); here that template is a fake Hugging Face tokenizer in a
``TokenizerChatTemplate``, so neither transformers nor model files are needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from tests.support.runtime_stubs import runtime_bindings

from reef.artifact.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import TokenizerChatTemplate, render_trajectory, rendered_job_data
from reef.core.job_data import JobData, TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.recipe.sft import SFTJobData, SftObjective, SFTRecipe, SFTSamples, SFTSettings
from reef.runtime.interfaces import TrainingMethod
from reef.sample2job import DistillJobData
from reef.train import PipelineContext
from reef.train.algos import registry
from reef.train.algos.registry import ObjectiveRegistry, resolve_objective
from reef.train.algos.schedule import StepScheduling
from reef.train.pipeline import TrainingPipeline
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.slime_backend.sft import SftAlgorithm

GENERATION_PROMPT = 900
END_OF_TURN = 901
QUESTION = [{"role": "user", "content": "What is the boiling point of water?"}]
LOOKUP_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}


class TurnTokenizer:
    """A Hugging Face tokenizer fake with one ID per character, a generation prompt, and an end-of-turn ID.

    An assistant turn renders as the generation prompt, its content, its tool
    call names, and the end-of-turn ID, so the generation prompt is a prefix
    of the conversation with the assistant turn.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None, bool]] = []
        self.template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None = None,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
        **template_kwargs: Any,
    ) -> list[int] | dict[str, list[int]]:
        self.calls.append((list(conversation), tools, add_generation_prompt))
        self.template_kwargs.append(template_kwargs)
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
def tokenizer() -> TurnTokenizer:
    return TurnTokenizer()


def _rendered(data: JobData, tokenizer: TurnTokenizer) -> JobData:
    """``data`` as the training component trains it: every sample rendered, none left out."""
    rendered, left_out = rendered_job_data(data, TokenizerChatTemplate(tokenizer))
    assert left_out == ()
    return rendered


def _rendered_sample(sample: TrajectorySample, tokenizer: TurnTokenizer) -> Mapping[str, Any]:
    (rendered,) = _rendered(JobData("science:data:1", (sample,)), tokenizer).samples
    assert isinstance(rendered, TrajectorySample)
    return rendered.training


def _left_out(sample: TrajectorySample, tokenizer: TurnTokenizer) -> tuple[int, ...]:
    return rendered_job_data(JobData("science:data:1", (sample,)), TokenizerChatTemplate(tokenizer))[1]


def _inference(
    agent_record_id: str,
    response: Mapping[str, Any] | None = None,
    messages: Sequence[Mapping[str, Any]] = QUESTION,
    tools: Sequence[Mapping[str, Any]] = (),
) -> AgentRecord:
    payload: dict[str, Any] = {
        "messages": list(messages),
        "response": response or {"choices": [{"message": {"role": "assistant", "content": "Cold."}}]},
    }
    if tools:
        payload["tools"] = list(tools)
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=ArtifactRef(content_id="science", release_id="slime-v3", parent_release_id=None),
    )


def _report(agent_record_id: str, references: tuple[str, ...], teacher_context: str = "100 C") -> AgentRecord:
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=TeacherContextReport(teacher_context=teacher_context).to_dict(references=references),
        agent_record_id=agent_record_id,
        references=references,
    )


def _samples(**config: Any) -> SFTSamples:
    return SFTSamples(PipelineContext("science", TeacherContextReport), SFTSettings(**config))


def _pipeline(**config: Any) -> TrainingPipeline:
    context = PipelineContext("science", TeacherContextReport)
    return TrainingPipeline(context, record2sample=_samples(**config), sample2job=DistillJobData(context, 1, None))


def _pipeline_with_report() -> TrainingPipeline:
    pipeline = _pipeline()
    pipeline.push(_inference("i1"))
    pipeline.push(_report("r1", ("i1",)))
    return pipeline


def _sample(pipeline: TrainingPipeline) -> TrajectorySample:
    (sample,) = pipeline.build_job_data(1).samples
    assert isinstance(sample, TrajectorySample)
    return sample


def _prompt_ids() -> list[int]:
    return [*TurnTokenizer.text_ids(QUESTION[0]), GENERATION_PROMPT]


# --- the supervised sample ----------------------------------------------------


@pytest.mark.unit
def test_the_demonstration_is_the_trained_assistant_turn(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.push(_inference("i1"))
    pipeline.push(_report("r1", ("i1",)))

    sample = _sample(pipeline)

    assert source_record_id(sample) == "i1"
    assert sample.source_agent_record_ids == ("i1", "r1")
    # The recorded reply is replaced: the request, then the demonstration as the final agent step.
    assert sample.trajectory["steps"] == [
        {"step_id": 1, "source": "user", "message": QUESTION[0]["content"]},
        {"step_id": 2, "source": "agent", "message": "100 C"},
    ]
    assert sample.training == {}, "the stage renders nothing"
    training = _rendered_sample(sample, tokenizer)
    target_ids = [ord(character) for character in "100 C"] + [END_OF_TURN]
    # The prompt, then the demonstration's tokens, which alone carry loss.
    assert training["tokens"] == [*_prompt_ids(), *target_ids]
    assert training["loss_mask"] == [1] * len(target_ids)
    assert "rollout_log_probs" not in training
    assert tokenizer.calls[-1][0][-1] == {"role": "assistant", "content": "100 C"}


@pytest.mark.unit
def test_earlier_agent_turns_of_the_request_are_copied_context(tokenizer: TurnTokenizer) -> None:
    earlier = [*QUESTION, {"role": "assistant", "content": "Hot."}, {"role": "user", "content": "In Celsius?"}]
    pipeline = _pipeline()
    pipeline.push(_inference("i1", messages=earlier))
    pipeline.push(_report("r1", ("i1",)))

    sample = _sample(pipeline)

    *context, target = sample.trajectory["steps"]
    assert [step.get("is_copied_context") for step in context if step["source"] == "agent"] == [True]
    assert target == {"step_id": target["step_id"], "source": "agent", "message": "100 C"}
    # The earlier reply renders as context; only the target carries loss.
    training = _rendered_sample(sample, tokenizer)
    target_ids = [ord(character) for character in "100 C"] + [END_OF_TURN]
    assert training["tokens"][-len(target_ids) :] == target_ids
    assert training["loss_mask"] == [1] * len(target_ids)


@pytest.mark.unit
def test_chat_template_kwargs_reach_every_render(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline()
    pipeline.push(_inference("i1"))
    pipeline.push(_report("r1", ("i1",)))

    data, left_out = rendered_job_data(
        pipeline.build_job_data(1), TokenizerChatTemplate(tokenizer, {"enable_thinking": False})
    )

    assert (len(data.samples), left_out) == (1, ())
    # The prompt and the sequence render alike, as the inference handler renders the served prompt.
    assert tokenizer.template_kwargs and all(
        kwargs == {"enable_thinking": False} for kwargs in tokenizer.template_kwargs
    )


@pytest.mark.unit
def test_the_recorded_response_is_the_target_of_an_imported_example(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline(target="response")
    tool_call = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "water"}'}}
    response = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}]}
    pipeline.push(_inference("i1", response, tools=[LOOKUP_TOOL]))
    pipeline.push(_report("r1", ("i1",), teacher_context=""))

    sample = _sample(pipeline)

    final = sample.trajectory["steps"][-1]
    assert (final["source"], final.get("is_copied_context")) == ("agent", None)
    assert final["tool_calls"] == [{"tool_call_id": "c1", "function_name": "lookup", "arguments": {"q": "water"}}]
    training = _rendered_sample(sample, tokenizer)
    target_ids = [ord(character) for character in "lookup"] + [END_OF_TURN]
    assert training["tokens"] == [*_prompt_ids(), *target_ids]
    assert training["loss_mask"] == [1] * len(target_ids)
    conversation, tools, _ = tokenizer.calls[-1]
    # The tool call renders with its arguments as an object, as chat templates expect, and with the request's tools.
    assert conversation[-1]["tool_calls"][0]["function"]["arguments"] == {"q": "water"}
    assert [tool["function"]["name"] for tool in tools] == ["lookup"]


@pytest.mark.unit
def test_a_sample_over_max_tokens_is_left_out_by_the_component(tokenizer: TurnTokenizer) -> None:
    pipeline = _pipeline(max_tokens=len(_prompt_ids()) + 2)
    pipeline.push(_inference("i1"))
    pipeline.push(_report("r1", ("i1",)))

    sample = _sample(pipeline)

    # The stage only names the limit; the component renders the sample and leaves out a longer one.
    assert sample.metadata["max_tokens"] == len(_prompt_ids()) + 2
    assert _left_out(sample, tokenizer) == (0,)
    assert "max_tokens" not in _sample(_pipeline_with_report()).metadata


@pytest.mark.unit
def test_a_broken_training_contract_raises() -> None:
    demonstration = _pipeline()
    demonstration.push(_inference("i1"))
    with pytest.raises(ValueError, match="teacher_context"):
        demonstration.push(_report("r1", ("i1",), teacher_context=""))
    response = _pipeline(target="response")
    # A recorded call without a response message ends with the request, not an agent step.
    response.push(_inference("i2", {"output": []}))
    with pytest.raises(ValueError, match="recorded response message"):
        response.push(_report("r2", ("i2",)))


@pytest.mark.unit
def test_a_template_that_rewrites_the_generation_prompt_is_left_out(
    tokenizer: TurnTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample(_pipeline_with_report())
    rendered = {True: [1, 2, 3], False: [1, 4, 3, 5]}
    monkeypatch.setattr(
        tokenizer, "apply_chat_template", lambda conversation, **options: rendered[options["add_generation_prompt"]]
    )

    with pytest.raises(ValueError, match="does not render each turn as a prefix"):
        render_trajectory(TokenizerChatTemplate(tokenizer), sample.trajectory)
    assert _left_out(sample, tokenizer) == (0,)


@pytest.mark.unit
def test_a_held_sample_never_becomes_stale() -> None:
    pipeline = _pipeline_with_report()
    # Two later artifacts: with max_staleness 0 a sampled policy row would be dropped.
    pipeline.observe_serving_artifact_id("slime-v4")
    pipeline.observe_serving_artifact_id("slime-v5")

    assert pipeline.ready(1)
    assert len(pipeline.build_job_data(1).samples) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"), [({"target": "teacher"}, "target"), ({"max_tokens": -1}, "max_tokens")]
)
def test_settings_reject_invalid_values(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SFTSettings(**config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config", [{"tokenizer_path": "/models/served"}, {"chat_template_kwargs": {"enable_thinking": False}}]
)
def test_settings_refuse_the_removed_rendering_keys(config: dict[str, Any]) -> None:
    # The training component renders with its own model's chat template, so the stage takes no tokenizer.
    with pytest.raises(PluginConfigError, match=f"unknown config fields: {next(iter(config))}"):
        SFTSettings.parse(config, plugin_class=SFTSamples)


# --- recipe -------------------------------------------------------------------


@pytest.mark.unit
def test_sft_recipe_resolves_by_dotted_reference_and_builds_its_stages() -> None:
    reference = "reef.recipe.sft:SFTRecipe"
    assert recipe_class_for(reference) is SFTRecipe

    recipe = build_recipe(
        reference,
        {},
        {
            "pipeline": {
                "record2sample": {"config": {"target": "response", "max_tokens": 8}},
                "sample2job": {"config": {"buffer_size": 256}},
            }
        },
        **runtime_bindings(),
    )

    assert isinstance(recipe, SFTRecipe)
    assert recipe.name == "sft"
    assert recipe.report_type is TeacherContextReport
    spec = SFTRecipe.training_spec()
    assert (spec.objective, spec.loss_family) == ("reef.recipe.sft:SftObjective", "sft")
    assert SftObjective.rollout_unit == "sample"
    pipeline = recipe.build_pipeline(PipelineContext("science", TeacherContextReport))
    assert isinstance(pipeline.record2sample, SFTSamples)
    assert (pipeline.record2sample.target, pipeline.record2sample.max_tokens) == ("response", 8)
    assert isinstance(pipeline.sample2job, DistillJobData)
    assert pipeline.sample2job.buffer_size == 256


@pytest.mark.unit
def test_sft_recipe_builds_without_stage_config_and_rejects_removed_keys_and_objective_options() -> None:
    def sft_recipe(config: Mapping[str, Any]) -> SFTRecipe:
        recipe = SFTRecipe.from_environment({}, config=config, **runtime_bindings())
        assert isinstance(recipe, SFTRecipe)
        return recipe

    context = PipelineContext("science", TeacherContextReport)
    assert sft_recipe({}).build_pipeline(context).record2sample.target == "demonstration"
    removed = {"pipeline": {"record2sample": {"config": {"tokenizer_path": "/models/served"}}}}
    with pytest.raises(RecipeConfigError, match="unknown config fields: tokenizer_path"):
        sft_recipe(removed).build_pipeline(context)
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        sft_recipe({"optimization": {"label_smoothing": 0.1}})


# --- objective and Slime wire payload ----------------------------------------


@pytest.mark.unit
def test_objective_trains_every_batch_without_advantages_and_allows_several_epochs(tokenizer: TurnTokenizer) -> None:
    signal = SftObjective().prepare(_rendered(_pipeline_with_report().build_job_data(1), tokenizer), {"steps": 2})

    assert (signal.action, signal.next_algorithm_state, signal.advantages) == ("train", {"steps": 3}, None)
    assert signal.metrics == {"steps": 3, "samples": 1}
    SftObjective().validate_scheduling(StepScheduling(epochs=3))
    # Another recipe selects the objective for a job by its dotted reference.
    assert isinstance(resolve_objective("reef.recipe.sft:SftObjective"), SftObjective)


@pytest.mark.unit
def test_the_recipe_objective_resolves_where_the_recipe_was_never_imported(
    tokenizer: TurnTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Slime bridge never imports the recipe module, so no objective is registered there.
    monkeypatch.setattr(registry, "OBJECTIVES", ObjectiveRegistry())

    step = prepare_slime_job(
        _rendered(_pipeline_with_report().build_job_data(1), tokenizer),
        TrainingMethod(SFTRecipe.training_spec().objective),
        {},
        StepScheduling(),
        step_size=1,
    )

    assert step.payload is not None
    assert step.payload["loss"] == "sft"


@pytest.mark.unit
def test_slime_payload_trains_the_target_tokens_without_log_probs(tokenizer: TurnTokenizer) -> None:
    payload = prepare_slime_job(
        _rendered(_pipeline_with_report().build_job_data(1), tokenizer),
        TrainingMethod("sft"),
        {},
        StepScheduling(),
        step_size=1,
    ).payload

    assert payload is not None
    assert payload["loss"] == "sft"
    assert "advantages" not in payload
    data = to_slime_rollout_data(payload)
    target_length = len("100 C") + 1
    assert data["response_lengths"] == [target_length]
    assert data["loss_masks"] == [[1] * target_length]
    assert "rollout_log_probs" not in data
    family = resolve_loss_family("sft")
    assert isinstance(family, SftAlgorithm)
    assert family.loss_type == "sft_loss"


# --- uploaded trajectories ----------------------------------------------------


def _uploaded(*steps: Mapping[str, Any]) -> TrajectorySample:
    return TrajectorySample(
        {"schema_version": "ATIF-v1.4", "session_id": "u", "agent": {"name": "a"}, "steps": list(steps)}
    )


ASK = {"step_id": 1, "source": "user", "message": "Boil?"}
CALL = {
    "step_id": 2,
    "source": "agent",
    "message": "Look",
    "tool_calls": [{"tool_call_id": "c1", "function_name": "search", "arguments": {"q": "boil"}}],
    "observation": {"results": [{"source_call_id": "c1", "content": "100"}]},
}
ANSWER = {"step_id": 3, "source": "agent", "message": "100 C"}
THANKS = {"step_id": 4, "source": "user", "message": "Thanks"}
ASK_IDS = TurnTokenizer.text_ids({"content": "Boil?"})
LOOK_IDS = TurnTokenizer.text_ids({"content": "Look", "tool_calls": [{"function": {"name": "search"}}]})
OBSERVATION_IDS = TurnTokenizer.text_ids({"content": "100"})
ANSWER_IDS = TurnTokenizer.text_ids({"content": "100 C"})


@pytest.mark.unit
def test_an_uploaded_trajectory_trains_every_agent_turn_and_masks_the_rest(tokenizer: TurnTokenizer) -> None:
    uploaded = _uploaded(ASK, CALL, ANSWER, THANKS)

    sample = _samples().uploaded_sample(uploaded)

    assert sample == uploaded, "the stage keeps the uploaded trajectory as it is"
    training = _rendered_sample(sample, tokenizer)
    # The step after the last agent turn is left out; the generation prompt of the first agent turn ends the prompt.
    assert training["tokens"] == [
        *ASK_IDS,
        GENERATION_PROMPT,
        *LOOK_IDS,
        END_OF_TURN,
        *OBSERVATION_IDS,
        GENERATION_PROMPT,
        *ANSWER_IDS,
        END_OF_TURN,
    ]
    assert training["loss_mask"] == [
        *[1] * (len(LOOK_IDS) + 1),
        *[0] * (len(OBSERVATION_IDS) + 1),
        *[1] * (len(ANSWER_IDS) + 1),
    ]
    assert "rollout_log_probs" not in training


@pytest.mark.unit
def test_copied_context_agent_steps_render_and_train_nothing(tokenizer: TurnTokenizer) -> None:
    template = TokenizerChatTemplate(tokenizer)

    # The copied turn is context: in the tokens, before the prompt that the loss mask starts after.
    rendered = render_trajectory(template, _uploaded(ASK, {**CALL, "is_copied_context": True}, ANSWER).trajectory)
    assert rendered.tokens == [
        *ASK_IDS,
        GENERATION_PROMPT,
        *LOOK_IDS,
        END_OF_TURN,
        *OBSERVATION_IDS,
        GENERATION_PROMPT,
        *ANSWER_IDS,
        END_OF_TURN,
    ]
    assert rendered.loss_mask == [1] * (len(ANSWER_IDS) + 1)

    # A copied turn after the last trained one trains nothing either, so the sample ends before it.
    rendered = render_trajectory(template, _uploaded(ASK, CALL, {**ANSWER, "is_copied_context": True}).trajectory)
    assert rendered.tokens == [*ASK_IDS, GENERATION_PROMPT, *LOOK_IDS, END_OF_TURN]
    assert rendered.loss_mask == [1] * (len(LOOK_IDS) + 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        ((ASK, THANKS), "needs one or more agent steps"),
        ((ASK, {**ANSWER, "is_copied_context": True}), "copied context steps train nothing"),
        ((ASK, {**ANSWER, "is_copied_context": "yes"}), "is_copied_context must be a boolean"),
        ((ANSWER, THANKS), "needs a system or user step before its first trained agent step"),
        (({"step_id": 1, "message": "no source"},), "string source"),
        ((ASK, {**CALL, "tool_calls": [{"tool_call_id": "c1"}]}), "tool_calls must be objects"),
        (({**ASK, "message": [{"type": "image"}]}, ANSWER), "image part needs source.path"),
        (({**ASK, "message": [{"type": "image", "source": {}}]}, ANSWER), "image part needs source.path"),
        (({**ASK, "message": [{"type": "image", "source": "a"}]}, ANSWER), "image part needs source.path"),
        (({**ASK, "message": ["a.png"]}, ANSWER), "content parts must be objects"),
        (({**ASK, "message": [{"type": "text", "text": None}]}, ANSWER), "text part needs a string text"),
        ((ASK, {**ANSWER, "observation": "ok"}), "observation must be an object"),
    ],
)
def test_an_uploaded_trajectory_that_cannot_train_is_refused(steps, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        _samples().uploaded_sample(_uploaded(*steps))


@pytest.mark.unit
def test_a_template_that_changes_earlier_turns_or_a_long_sample_is_left_out_by_the_component(
    tokenizer: TurnTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The upload names the limit; the component leaves out a sample that renders longer.
    long = _samples(max_tokens=5).uploaded_sample(_uploaded(ASK, ANSWER))
    assert long.metadata["max_tokens"] == 5
    assert _left_out(long, tokenizer) == (0,)
    render = tokenizer.apply_chat_template

    def drops_earlier_turns(conversation, **options):
        # Like a template that strips earlier turns' reasoning: the whole conversation renders its first turn shorter.
        ids = render(conversation, **options)
        return ids[1:] if len(conversation) > 2 else ids

    monkeypatch.setattr(tokenizer, "apply_chat_template", drops_earlier_turns)
    # The upload checks only the structure, so it accepts the trajectory; the component cannot render it.
    sample = _samples().uploaded_sample(_uploaded(ASK, CALL, ANSWER))
    with pytest.raises(ValueError, match="does not render each turn as a prefix"):
        render_trajectory(TokenizerChatTemplate(tokenizer), sample.trajectory)
    assert _left_out(sample, tokenizer) == (0,)


@pytest.mark.unit
def test_the_sft_recipe_takes_uploaded_trajectories_as_rendered_samples(tokenizer: TurnTokenizer) -> None:
    assert SFTRecipe.pipeline_spec().sample2job is SFTJobData and SFTJobData.accepts_trajectories
    context = PipelineContext("science", TeacherContextReport)
    pipeline = TrainingPipeline(
        context,
        record2sample=_samples(),
        sample2job=SFTJobData(context, None, 1),
    )
    pipeline.store.store.add_trajectories((("t1", _uploaded(ASK, ANSWER)),))
    data = pipeline.build_job_data(1)
    (item,) = _rendered(data, tokenizer).samples
    assert item.training["loss_mask"] == [1] * (len(ANSWER_IDS) + 1)
    assert pipeline.acknowledge(data.data_id) == {"sample:t1"}


def test_sft_gives_parameters_to_trajectory_jobs_and_trains_imported_tasks_only_by_instruction() -> None:
    recipe = SFTRecipe(**runtime_bindings())
    assert set(recipe.parameters_by_kind()) == {"trajectory"}
    assert (
        recipe.parameters_by_kind()["trajectory"]["method"] == TrainingMethod("reef.recipe.sft:SftObjective").to_dict()
    )
    assert SFTJobData.accepts_trajectories and SFTJobData.accepts_imported_tasks
