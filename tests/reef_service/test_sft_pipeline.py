"""Reef-side SFT pipeline: the demonstration as the assistant turn, the recipe and the Slime wire payload.

Torch/ray free, like ``test_sdft_pipeline.py``: the demonstration tokenizer is
a fake that counts tokens deterministically, so no model files are needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sft import SftObjective, SFTProcessor, SFTRecipe
from recipes.sft.processor import DemonstrationTokenizer
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TrainingBatch


class CountingDemonstrationTokenizer(DemonstrationTokenizer):
    """Prompt ids one per message; response ids one per ten characters of the demonstration, then an end token."""

    def sequence_ids(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, demonstration: str
    ) -> tuple[list[int], list[int]]:
        return [100 + index for index in range(len(messages))], [*range(200, 200 + len(demonstration) // 10), 999]


MESSAGES = [{"role": "system", "content": "Answer."}, {"role": "user", "content": "Which acid?"}]
DEMONSTRATION = "<reasoning>\nthirty characters of reasoning\n</reasoning>\n<answer>\nB\n</answer>"


def _inference(agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": MESSAGES,
            "tokens": [5, 6, 7, 1, 2, 3],
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
        },
        agent_record_id=agent_record_id,
    )


def _report(agent_record_id: str, reference: str, context: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=TeacherContextReport(context=context).to_dict(references=(reference,)),
        agent_record_id=agent_record_id,
        references=(reference,),
    )


def _processor(**config: Any) -> SFTProcessor:
    return SFTProcessor(
        ProcessorContext("science", {"batch_size": 1, **config}, TeacherContextReport),
        CountingDemonstrationTokenizer(),
    )


@pytest.mark.unit
def test_processor_trains_the_demonstration_as_the_assistant_turn() -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", "i1", DEMONSTRATION))
    batch = processor.build_batch()

    (sample,) = batch.items
    prompt_ids, response_ids = CountingDemonstrationTokenizer().sequence_ids(MESSAGES, None, DEMONSTRATION)
    assert prompt_ids == [100, 101]
    assert list(sample.training["tokens"]) == [*prompt_ids, *response_ids]
    assert list(sample.training["loss_mask"]) == [1] * len(response_ids)
    # The student's own response is ignored: no engine log-probs, no load spans.
    assert list(sample.training["rollout_log_probs"]) == []
    assert list(sample.training["runtime_load_spans"]) == []


@pytest.mark.unit
def test_processor_requires_one_recorded_request_and_a_rendered_demonstration() -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", "i1", "short"))  # renders to the end token only: still a sample
    (sample,) = processor.build_batch().items
    assert list(sample.training["tokens"]) == [100, 101, 999]

    with pytest.raises(ValueError, match="tokenizer_path"):
        SFTProcessor(ProcessorContext("science", {"batch_size": 1}, TeacherContextReport))


@pytest.mark.unit
def test_slime_payload_carries_the_demonstration_row_and_refuses_advantages() -> None:
    processor = _processor()
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", "i1", DEMONSTRATION))
    spec = SFTRecipe.training_spec()
    prepared = prepare_slime_step(processor.build_batch(), spec.objective, {}, StepScheduling(unit="sample"))
    payload = prepared.payload
    assert payload is not None
    assert payload["loss"] == spec.loss_family == "sft"

    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    prompt_ids, response_ids = CountingDemonstrationTokenizer().sequence_ids(MESSAGES, None, DEMONSTRATION)
    assert data["loss"] == "sft"
    assert data["tokens"] == [[*prompt_ids, *response_ids]]
    assert data["loss_masks"] == [[1] * len(response_ids)]
    assert "rollout_log_probs" not in data
    with pytest.raises(ValueError, match="sft ignores advantages"):
        to_slime_rollout_data({**payload, "advantages": [1.0]})


@pytest.mark.unit
def test_sft_recipe_binds_the_shared_report_and_its_family() -> None:
    recipe = build_recipe(
        "recipes.sft.recipe:SFTRecipe",
        {},
        {"data": {"tokenizer_path": "/models/served", "batch_size": 32}},
        **runtime_bindings(StubTrainingRuntime()),
    )
    assert isinstance(recipe, SFTRecipe)
    assert recipe.report_type is TeacherContextReport
    assert recipe.batch_size == 32
    assert recipe.processor_config()["tokenizer_path"] == "/models/served"

    family = resolve_loss_family(SFTRecipe.training_spec().loss_family)
    assert (family.loss_family, family.loss_type, family.advantages) == ("sft", "sft_loss", "forbidden")
    family.validate_backend_args(SimpleNamespace(loss_type="sft_loss", use_rollout_logprobs=False))

    with pytest.raises(RecipeConfigError, match="tokenizer_path is required"):
        build_recipe("recipes.sft.recipe:SFTRecipe", {}, {}, **runtime_bindings(StubTrainingRuntime()))
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        build_recipe(
            "recipes.sft.recipe:SFTRecipe",
            {},
            {"data": {"tokenizer_path": "/models/served"}, "optimization": {"lr": 1e-5}},
            **runtime_bindings(StubTrainingRuntime()),
        )


@pytest.mark.unit
def test_objective_trains_every_batch() -> None:
    objective = SftObjective()
    assert (objective.name, objective.loss_family, objective.supports_multiple_epochs) == ("sft", "sft", True)
    signal = objective.prepare(TrainingBatch("science:sft:1", ()), {"steps": 3})
    assert signal.action == "train"
    assert signal.next_algorithm_state == {"steps": 4}
    assert signal.metrics == {"steps": 4, "samples": 0}
