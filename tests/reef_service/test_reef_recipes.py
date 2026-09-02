from __future__ import annotations

import subprocess
import sys
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.openclawrl import OpenClawRLProcessor, OpenClawRLRecipe
from recipes.sao import SAOProcessor, SAORecipe
from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor, TTTDRecipe
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.recipe import (
    Recipe,
    RecipeConfigError,
    WeightTrainingRecipe,
    WeightTrainingSpec,
    load_recipe_config,
)
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for
from reef.records import RecordStore
from reef.runtime import InferenceProxyRuntime
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.train.processors.base import DataProcessor
from reef.train.slime_backend.backend import SlimeTrainingBackend

from ._threshold_processor import ThresholdProcessor


@pytest.mark.unit
@pytest.mark.parametrize("recipe_type", [OpenClawRLRecipe, SAORecipe, TTTDRecipe])
def test_training_recipes_share_max_staleness(recipe_type) -> None:
    assert recipe_type(StubTrainingRuntime()).max_staleness == 0
    assert recipe_type(StubTrainingRuntime(max_staleness=2), max_staleness=2).max_staleness == 2
    with pytest.raises(ValueError, match="max_staleness must be a non-negative integer"):
        recipe_type(StubTrainingRuntime(), max_staleness=-1)
    with pytest.raises(ValueError, match="must match the training runtime"):
        recipe_type(StubTrainingRuntime(), max_staleness=2)


def test_recipe_class_resolver_has_no_method_short_names() -> None:
    assert recipe_class_for("recipe") is Recipe
    assert recipe_class_for("sao") is None
    assert recipe_class_for("tttd") is None
    assert recipe_class_for("openclawrl") is None
    assert recipe_class_for("harness_evolve") is None


def test_cookbook_weight_training_recipe_loss_family_mappings() -> None:
    mappings = {
        recipe_type.__name__: recipe_type.training_spec().loss_family
        for recipe_type in (OpenClawRLRecipe, SAORecipe, TTTDRecipe)
    }

    assert mappings == {
        "OpenClawRLRecipe": "openclawrl",
        "SAORecipe": "sao",
        "TTTDRecipe": "tttd",
    }


def test_dotted_reference_resolves_an_external_recipe_class() -> None:
    assert recipe_class_for("reef.recipe.base:Recipe") is Recipe
    assert recipe_class_for("recipes.sao.recipe:SAORecipe") is SAORecipe


@pytest.mark.parametrize(
    ("reference", "match"),
    [
        ("nowhere.to.be:Found", "cannot import recipe reference"),
        ("reef.recipe.base:missing", "cannot import recipe reference"),
        ("reef.recipe.base:config_positive_int", "is not a Recipe class"),
        (":Recipe", "must be 'package.module:ClassName'"),
    ],
)
def test_dotted_recipe_rejects_bad_references(reference: str, match: str) -> None:
    with pytest.raises(RecipeConfigError, match=match):
        recipe_class_for(reference)


@pytest.mark.parametrize(
    ("recipe", "name", "processor_type", "step_preparer", "report_type"),
    [
        (Recipe(), "recipe", DataProcessor, None, None),
        (
            OpenClawRLRecipe(StubTrainingRuntime(), batch_size=3),
            "openclawrl",
            OpenClawRLProcessor,
            "openclawrl",
            None,
        ),
        (SAORecipe(StubTrainingRuntime(), batch_size=1), "sao", SAOProcessor, "sao", ScoredRolloutReport),
        (
            TTTDRecipe(StubTrainingRuntime(), groups_per_step=2, rollouts_per_group=3),
            "tttd",
            TTTDProcessor,
            "tttd",
            TTTDGroupedRolloutReport,
        ),
    ],
)
def test_concrete_recipe_builds_its_processor_step_preparer_and_report_type(
    recipe, name, processor_type, step_preparer, report_type
) -> None:
    trainer = recipe.build("math", RecordStore())

    assert recipe.name == name
    assert recipe.report_type is report_type
    assert isinstance(trainer.processor, processor_type)
    if step_preparer is None:
        assert trainer.training_backend is None
    else:
        assert isinstance(trainer.training_backend, SlimeTrainingBackend)
        assert trainer.training_backend.step_preparer == step_preparer
    assert trainer.report_type is report_type


def test_build_rejects_an_unknown_step_preparer_before_any_training_step() -> None:
    # Regression: the preparer name used to be resolved only at the first
    # training step — after GPUs were already up. A recipe naming a preparer
    # that no longer exists (the deleted online_grpo arm, a typo) must fail at
    # recipe build, with the registry's available-preparers message.
    @dataclass(frozen=True)
    class StalePreparerRecipe(WeightTrainingRecipe):
        _: KW_ONLY
        name: str = "stale"

        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(
                step_preparer="online_grpo",
                loss_family="pg",
                processor=ThresholdProcessor,
            )

    with pytest.raises(ValueError, match=r"unknown step preparer 'online_grpo'.*available preparers"):
        StalePreparerRecipe(StubTrainingRuntime()).build("math", RecordStore())


def test_tttd_build_resolves_its_backend_registered_preparer_in_a_fresh_process() -> None:
    # A service process never imports the Slime driver, so the cookbook
    # package must register its preparer when the dotted recipe is imported for eager preparer
    # resolution to accept "tttd" in the process the recipe is served from.
    # In-process tests cannot pin this (a sibling test may have imported the
    # driver for the whole session), hence the subprocess.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
                "from recipes.tttd import TTTDRecipe\n"
                "from reef.records import RecordStore\n"
                "from reef_service.runtime_stubs import StubTrainingRuntime\n"
                "trainer = TTTDRecipe(StubTrainingRuntime(), groups_per_step=1, rollouts_per_group=2)"
                ".build('math', RecordStore())\n"
                "assert trainer.training_backend.step_preparer == 'tttd'\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _turn_inference(agent_record_id: str, tokens: list[int], log_prob: float) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        payload={
            "response": {
                "training": {
                    "tokens": tokens,
                    "loss_mask": [1],
                    "rollout_log_probs": [log_prob],
                    "runtime_load_id": "wv-1",
                }
            }
        },
    )


@pytest.mark.parametrize(
    ("recipe", "metadata"),
    [
        (SAORecipe(StubTrainingRuntime()), {}),
        (
            TTTDRecipe(StubTrainingRuntime(), groups_per_step=1, rollouts_per_group=2),
            {
                "algorithm": "tttd",
                "step": 0,
                "group": 0,
                "rollout": 0,
                "groups_per_step": 1,
                "rollouts_per_group": 2,
                "comparison_set": "tttd-step-0-group-0",
            },
        ),
    ],
)
def test_cookbook_recipes_reject_multi_turn_policy_samples(recipe, metadata) -> None:
    # Observable pin: a valid, assemblable two-turn episode is refused by the
    # cookbook recipes' processors — the report is terminal and released, not
    # accepted as a candidate. (openclawrl is absent: it consumes no reports
    # at all — see test_openclawrl_recipe_ignores_reports.)
    trainer = recipe.build("math", RecordStore())
    processor = trainer.processor
    processor.ingest(_turn_inference("i1", [10, 20], -0.1))
    processor.ingest(_turn_inference("i2", [10, 20, 11, 21], -0.2))
    processor.ingest(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            agent_record_id="r1",
            payload={"score": 1.0, "references": ["i1", "i2"], "metadata": metadata},
            references=("i1", "i2"),
        )
    )

    assert not processor.ready()
    assert "r1" in processor.retention_decision().releasable_agent_record_ids


def test_openclawrl_recipe_never_trains_on_reports() -> None:
    # The judging half lives inside the processor now: a report is consumed
    # held for retention only — terminal on sight, releasable, never a candidate.
    trainer = OpenClawRLRecipe(StubTrainingRuntime()).build("math", RecordStore())
    processor = trainer.processor
    processor.ingest(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            agent_record_id="r1",
            payload={"score": 1.0, "references": ["i1"]},
            references=("i1",),
        )
    )
    assert not processor.ready()
    decision = processor.retention_decision()
    assert "r1" in decision.releasable_agent_record_ids
    assert "r1" not in decision.protected_agent_record_ids
    trainer.close()


def test_recipe_factory_keeps_method_configuration_inside_recipe() -> None:
    runtime = StubTrainingRuntime()
    recipe = build_recipe(
        "recipes.tttd.recipe:TTTDRecipe",
        {
            "REEF_TRAINER_URL": "http://trainer:8901",
            "REEF_TRAINER_TOKEN": "secret",
            "REEF_TTTD_GROUPS_PER_STEP": "2",
            "REEF_TTTD_ROLLOUTS_PER_GROUP": "3",
        },
        runtime=runtime,
    )

    assert isinstance(recipe, TTTDRecipe)
    assert recipe.groups_per_step == 2
    assert recipe.rollouts_per_group == 3
    assert recipe.runtime is runtime


def test_training_recipe_requires_its_runtime_environment() -> None:
    with pytest.raises(RecipeConfigError, match="requires a training runtime"):
        build_recipe("recipes.tttd.recipe:TTTDRecipe", {})


def test_recipe_config_routes_sections_to_their_consumers(tmp_path) -> None:
    path = tmp_path / "openclawrl.yaml"
    path.write_text(
        """implementation: recipes.openclawrl.recipe:OpenClawRLRecipe
data:
  batch_size: 4
artifact:
  checkpoint_every_n_versions: 3
model:
  path: /models/qwen
"""
    )

    settings = load_recipe_config(path)
    recipe = build_recipe(settings["implementation"], {}, config=settings, runtime=StubTrainingRuntime())

    assert recipe.batch_size == 4
    assert recipe.checkpoint_strategy == EveryNVersions(3)


def test_recipe_config_requires_implementation(tmp_path) -> None:
    path = tmp_path / "recipes.yaml"
    path.write_text("data:\n  batch_size: 1\n")

    with pytest.raises(RecipeConfigError, match="non-empty 'implementation'"):
        load_recipe_config(path)


def test_named_recipe_resolves_preset_from_filename(tmp_path) -> None:
    path = tmp_path / "thorough.yaml"
    path.write_text(
        """implementation: recipe
runtime:
  type: inference_proxy
  base_url: http://provider
  api_key: secret
model:
  path: Qwen/Qwen3.5-27B
"""
    )

    recipe = build_named_recipe("thorough", config_directory=tmp_path)

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe.runtime, InferenceProxyRuntime)


def test_config_backed_training_recipe_requires_an_injected_runtime(tmp_path) -> None:
    path = tmp_path / "sao-qwen.yaml"
    path.write_text(
        """implementation: recipes.sao.recipe:SAORecipe
model:
  path: Qwen/Qwen3.5-27B
"""
    )
    with pytest.raises(RecipeConfigError, match="requires a training runtime"):
        build_named_recipe("sao-qwen", config_directory=tmp_path)


def test_named_recipe_without_config_directory_resolves_core_recipe_only() -> None:
    assert isinstance(build_named_recipe("recipe", {}), Recipe)
    with pytest.raises(RecipeConfigError, match="unknown deployment recipe 'thorough'"):
        build_named_recipe("thorough", {})


def test_named_recipe_reads_the_config_directory_from_the_environment(tmp_path) -> None:
    (tmp_path / "thorough.yaml").write_text("implementation: recipe\nmodel:\n  path: Qwen/Qwen3.5-27B\n")

    recipe = build_named_recipe("thorough", {"REEF_RECIPE_CONFIG_DIR": str(tmp_path)})

    assert isinstance(recipe, Recipe)


def test_named_recipes_never_import_implementation_references() -> None:
    with pytest.raises(RecipeConfigError, match="invalid recipe name"):
        build_named_recipe("reef.recipe.base:Recipe", {})


def test_named_recipe_configs_resolve_by_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-minimax-key")
    # A local proxy preset: the core recipe pinned to a deployment's
    # own endpoint and model.
    (tmp_path / "local-proxy.yaml").write_text(
        "implementation: recipe\n"
        "runtime:\n"
        "  type: inference_proxy\n"
        "  base_url: http://127.0.0.1:30000\n"
        "  timeout_s: 300\n"
        "model:\n"
        "  path: Qwen/Qwen3.6-27B\n"
    )
    local = build_named_recipe("local-proxy", config_directory=tmp_path)

    settings = load_recipe_config(tmp_path / "local-proxy.yaml")
    assert settings["model"]["path"] == "Qwen/Qwen3.6-27B"
    assert isinstance(local.runtime, InferenceProxyRuntime)
    assert local.runtime.model_path == "Qwen/Qwen3.6-27B"

    # The hosted-provider shape (api_key_env) needs no built-in example: any
    # directory of named YAMLs resolves the same way.
    (tmp_path / "hosted-proxy.yaml").write_text(
        "implementation: recipe\n"
        "runtime:\n"
        "  type: inference_proxy\n"
        "  base_url: https://api.minimax.io/anthropic\n"
        "  api_key_env: ANTHROPIC_AUTH_TOKEN\n"
        "  timeout_s: 300\n"
        "model:\n"
        "  path: MiniMax-M3\n"
    )
    runtime = build_named_recipe("hosted-proxy", config_directory=tmp_path).runtime
    assert isinstance(runtime, InferenceProxyRuntime)
    assert runtime.model_path == "MiniMax-M3"
    assert runtime.base_url == "https://api.minimax.io/anthropic"
    assert runtime.api_key == "test-minimax-key"


def test_cookbook_preparers_signal_their_recipes_loss_family() -> None:
    # The recipe's loss_family and the preparer's StepSignal.loss_family are
    # two strings; the bridge only compares them at the first training step.
    # Pin them here so a drift fails before any GPU spins up.
    from dataclasses import replace

    from reef.train.algos.registry import resolve_preparer
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.types import GroupedPolicyBatch, PolicyBatch, PolicySample

    first = PolicySample("i1", (5, 1), (1,), (-0.1,), 0.5)
    second = replace(first, source_agent_record_id="i2", reward=1.5)
    checked = set()
    for recipe_type in (OpenClawRLRecipe, SAORecipe, TTTDRecipe):
        spec = recipe_type.training_spec()
        assert spec.processor is not None
        if spec.processor.output_schema is GroupedPolicyBatch:
            batch = GroupedPolicyBatch("b", ((first, second),))
        else:
            batch = PolicyBatch("b", (first, second))
        signal = resolve_preparer(spec.step_preparer)(batch, {})
        assert signal.loss_family == spec.loss_family, recipe_type.__name__
        assert resolve_loss_family(signal.loss_family).loss_family == spec.loss_family
        checked.add(recipe_type.__name__)
    assert checked == {"OpenClawRLRecipe", "SAORecipe", "TTTDRecipe"}
