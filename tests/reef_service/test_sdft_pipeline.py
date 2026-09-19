"""Reef-side SDFT pipeline: the demonstration composition, the recipe and the Slime wire payload.

Everything here is torch/ray free so it runs in the minimal CI gate; the
shared processor is covered in ``test_distill_processor.py`` and the tensor
kernels in ``test_distill_parity.py``. The tokenizer is a fake installed as
``transformers.AutoTokenizer`` that counts tokens deterministically, so neither
transformers nor model files are needed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sdft import SdftObjective, SDFTProcessor, SDFTRecipe
from recipes.sdft.processor import DEFAULT_CONTEXT_TEMPLATE
from recipes.sdft.slime import SdftSettings
from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import source_record_id
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TrainingBatch

TOKENIZER_PATH = "/models/served"
STUDENT_TOKENS = (5, 6, 7, 1, 2, 3)  # three prompt ids, three response ids
STUDENT_LOSS_MASK = (1, 1, 1)
STUDENT_LOG_PROBS = (-0.1, -0.2, -0.3)


class CountingTokenizer:
    """The served model's tokenizer: one token per message plus one per ten characters of text.

    It records what it rendered, and like transformers 5 returns a mapping
    unless ``return_dict=False``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None]] = []

    def from_pretrained(self, path: str, **options: Any) -> CountingTokenizer:
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


def _inference(agent_record_id: str, *, messages: list[dict[str, Any]] | None = None) -> AgentRecord:
    payload: dict[str, Any] = {
        "messages": messages or [{"role": "user", "content": "What is the boiling point of water?"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
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


def _processor(**config: Any) -> SDFTProcessor:
    return SDFTProcessor(
        ProcessorContext(
            "science", {"batch_size": 1, "tokenizer_path": TOKENIZER_PATH, **config}, TeacherContextReport
        )
    )


# --- the teacher's request --------------------------------------------------


@pytest.mark.unit
def test_the_demonstration_block_is_added_to_the_final_user_message(tokenizer: CountingTokenizer) -> None:
    request = [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Question?"}]},
    ]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    messages, teacher_tools = _processor().teacher_request(request, tools, "", "The answer.")

    assert messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Question?\n\n" + DEFAULT_CONTEXT_TEMPLATE.replace("{context}", "The answer.")},
    ]
    assert teacher_tools == tools
    # The request is left as recorded.
    assert request[1]["content"] == [{"type": "text", "text": "Question?"}]


@pytest.mark.unit
def test_the_block_becomes_a_user_message_after_a_tool_result(tokenizer: CountingTokenizer) -> None:
    request = [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
        {"role": "tool", "content": "a.txt"},
    ]
    processor = _processor(context_template="Demonstration: {context}")

    messages, teacher_tools = processor.teacher_request(request, None, "", "call bash with cat a.txt")

    assert messages[-1] == {"role": "user", "content": "Demonstration: call bash with cat a.txt"}
    assert messages[:3] == [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
        {"role": "tool", "content": "a.txt"},
    ]
    assert teacher_tools is None


@pytest.mark.unit
def test_the_template_needs_the_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{context\}"):
        _processor(context_template="no placeholder")


@pytest.mark.unit
def test_processor_emits_one_sample_with_the_teacher_sequence(tokenizer: CountingTokenizer) -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))

    batch = processor.build_batch()

    assert isinstance(batch, TrainingBatch)
    assert batch.batch_id == "science:sdft:1"
    (sample,) = batch.items
    assert source_record_id(sample) == "i1"
    prompt_ids = tokenizer.count_ids(tokenizer.calls[0][0])
    # The teacher sequence is the rendered teacher prompt plus the response ids verbatim.
    assert list(sample.training["teacher_tokens"]) == [*prompt_ids, 1, 2, 3]
    assert list(sample.training["tokens"]) == list(STUDENT_TOKENS)
    rendered, tools = tokenizer.calls[0]
    assert tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert rendered[-1]["role"] == "user"
    assert rendered[-1]["content"].startswith("What is the boiling point of water?\n\n")
    assert "100 degrees Celsius." in rendered[-1]["content"]
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0


# --- recipe -----------------------------------------------------------------


@pytest.mark.unit
def test_sdft_recipe_resolves_by_dotted_reference() -> None:
    reference = "recipes.sdft.recipe:SDFTRecipe"
    assert recipe_class_for(reference) is SDFTRecipe

    recipe = build_recipe(
        reference, {}, {"data": {"tokenizer_path": TOKENIZER_PATH}}, **runtime_bindings(StubTrainingRuntime())
    )

    assert isinstance(recipe, SDFTRecipe)
    assert recipe.name == "sdft"
    assert recipe.report_type is TeacherContextReport
    assert recipe.batch_size == 1
    assert recipe.max_teacher_tokens == 0
    assert recipe.context_template == DEFAULT_CONTEXT_TEMPLATE
    assert recipe.checkpoint_strategy == EveryNVersions(1)
    spec = SDFTRecipe.training_spec()
    assert (spec.objective, spec.processor, spec.loss_family) == ("sdft", SDFTProcessor, "sdft")
    assert spec.scheduling == StepScheduling(unit="sample")


@pytest.mark.unit
def test_sdft_recipe_reads_reef_side_config_and_hands_it_to_the_processor() -> None:
    recipe = SDFTRecipe.from_environment(
        {},
        config={
            "data": {
                "batch_size": 4,
                "tokenizer_path": TOKENIZER_PATH,
                "max_teacher_tokens": 24000,
                "context_template": "Example: {context}",
            },
            "artifact": {"checkpoint_every_n_versions": 4},
        },
        **runtime_bindings(StubTrainingRuntime()),
    )

    assert recipe.batch_size == 4
    assert recipe.checkpoint_strategy == EveryNVersions(4)
    assert recipe.processor_config() == {
        "batch_size": 4,
        "tokenizer_path": TOKENIZER_PATH,
        "max_teacher_tokens": 24000,
        "context_template": "Example: {context}",
    }


@pytest.mark.unit
def test_sdft_recipe_validates_the_demonstration_template_at_construction() -> None:
    with pytest.raises(RecipeConfigError, match=r"context_template must contain \{context\}"):
        SDFTRecipe.from_environment(
            {},
            config={"data": {"tokenizer_path": TOKENIZER_PATH, "context_template": "bare"}},
            **runtime_bindings(StubTrainingRuntime()),
        )


@pytest.mark.unit
def test_sdft_recipe_rejects_a_missing_tokenizer_and_backend_objective_config() -> None:
    with pytest.raises(RecipeConfigError, match="tokenizer_path"):
        SDFTRecipe.from_environment({}, config={}, **runtime_bindings(StubTrainingRuntime()))
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        SDFTRecipe.from_environment(
            {},
            config={"data": {"tokenizer_path": TOKENIZER_PATH}, "optimization": {"kl_direction": "reverse"}},
            **runtime_bindings(StubTrainingRuntime()),
        )


# --- objective and Slime wire payload ----------------------------------------


@pytest.mark.unit
def test_objective_trains_every_batch_and_refuses_multiple_epochs(tokenizer: CountingTokenizer) -> None:
    objective = SdftObjective()
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))
    batch = processor.build_batch()

    signal = objective.prepare(batch, {"steps": 2})

    assert (signal.action, signal.next_algorithm_state, signal.advantages) == ("train", {"steps": 3}, None)
    assert signal.metrics == {"steps": 3, "samples": 1}
    with pytest.raises(ValueError, match="epochs"):
        objective.validate_scheduling(StepScheduling(unit="sample", epochs=2))


@pytest.mark.unit
def test_slime_payload_carries_the_teacher_sequence_beside_the_policy_row(tokenizer: CountingTokenizer) -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))
    batch = processor.build_batch()

    prepared = prepare_slime_step(batch, "sdft", {}, StepScheduling(unit="sample"))
    payload = prepared.payload
    assert payload is not None
    assert payload["loss"] == "sdft"
    assert "advantages" not in payload
    (row,) = payload["samples"]
    prompt_ids = tokenizer.count_ids(tokenizer.calls[0][0])
    assert row == [
        "i1",
        list(STUDENT_TOKENS),
        list(STUDENT_LOSS_MASK),
        list(STUDENT_LOG_PROBS),
        0.0,
        [*prompt_ids, 1, 2, 3],
    ]

    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    assert data["loss"] == "sdft"
    assert data["tokens"] == [list(STUDENT_TOKENS)]
    assert data["response_lengths"] == [3]
    assert data["rollout_log_probs"] == [list(STUDENT_LOG_PROBS)]
    assert data["teacher_tokens"] == [[*prompt_ids, 1, 2, 3]]


def _payload(teacher_tokens: list[Any], **overrides: Any) -> dict[str, Any]:
    row = ["i1", list(STUDENT_TOKENS), list(STUDENT_LOSS_MASK), list(STUDENT_LOG_PROBS), 0.0, teacher_tokens]
    return {"samples": [row], "rollout_ids": [0], "loss": "sdft", **overrides}


@pytest.mark.unit
def test_slime_payload_rejects_a_teacher_sequence_that_does_not_end_with_the_response() -> None:
    with pytest.raises(ValueError, match="end with the student's response ids"):
        to_slime_rollout_data(_payload([9, 9, 1, 2, 4]))
    with pytest.raises(ValueError, match="carry a prompt"):
        to_slime_rollout_data(_payload([1, 2, 3]))
    with pytest.raises(ValueError, match="sequence of integers"):
        to_slime_rollout_data(_payload([9, "1", 2, 3]))
    with pytest.raises(ValueError, match="must be \\[source_id"):
        to_slime_rollout_data({"samples": [["i1", [5, 1], [1], [-0.1], 0.0]], "rollout_ids": [0], "loss": "sdft"})
    with pytest.raises(ValueError, match="must omit advantages"):
        to_slime_rollout_data(_payload([9, 1, 2, 3], advantages=[1.0]))


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
    accepted = {"loss_type": "custom_loss", "use_rollout_logprobs": True, "num_steps_per_rollout": 1}
    family.validate_backend_args(SimpleNamespace(**accepted))

    with pytest.raises(RuntimeError, match="loss-type custom_loss"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "loss_type": "policy_loss"}))
    with pytest.raises(RuntimeError, match="use-rollout-logprobs"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "use_rollout_logprobs": False}))
    with pytest.raises(RuntimeError, match="num-steps-per-rollout"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "num_steps_per_rollout": 2}))
