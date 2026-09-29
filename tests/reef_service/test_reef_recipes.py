from __future__ import annotations

import subprocess
import sys
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import pytest
from reef_service._trajectories import policy_trajectory
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.openclawrl import OpenClawRLProcessor, OpenClawRLRecipe
from recipes.sao import SAOProcessor, SAORecipe
from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor, TTTDRecipe
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.inference.http import InferenceProxyRuntime
from reef.recipe import Recipe, RecipeConfigError, WeightTrainingRecipe, WeightTrainingSpec, load_recipe_config
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.processors.base import DataProcessor
from reef.train.runtime_backend import RuntimeCandidateBackend

from ._threshold_processor import ThresholdProcessor


@pytest.mark.unit
@pytest.mark.parametrize("recipe_type", [OpenClawRLRecipe, SAORecipe, TTTDRecipe])
def test_training_recipes_share_max_staleness(recipe_type) -> None:
    assert recipe_type(**runtime_bindings(StubTrainingRuntime())).max_staleness == 0
    assert recipe_type(**runtime_bindings(StubTrainingRuntime(max_staleness=2)), max_staleness=2).max_staleness == 2
    with pytest.raises(ValueError, match="max_staleness must be a non-negative integer"):
        recipe_type(**runtime_bindings(StubTrainingRuntime()), max_staleness=-1)
    with pytest.raises(ValueError, match="must match the training runtime"):
        recipe_type(**runtime_bindings(StubTrainingRuntime()), max_staleness=2)


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
    ("recipe", "name", "processor_type", "objective", "report_type"),
    [
        (Recipe(), "recipe", DataProcessor, None, None),
        (
            OpenClawRLRecipe(**runtime_bindings(StubTrainingRuntime()), batch_size=3),
            "openclawrl",
            OpenClawRLProcessor,
            "openclawrl",
            None,
        ),
        (
            SAORecipe(**runtime_bindings(StubTrainingRuntime()), batch_size=1),
            "sao",
            SAOProcessor,
            "sao",
            ScoredRolloutReport,
        ),
        (
            TTTDRecipe(**runtime_bindings(StubTrainingRuntime()), groups_per_step=2, rollouts_per_group=3),
            "tttd",
            TTTDProcessor,
            "tttd",
            TTTDGroupedRolloutReport,
        ),
    ],
)
def test_concrete_recipe_builds_its_processor_objective_and_report_type(
    recipe, name, processor_type, objective, report_type
) -> None:
    trainer = recipe.build("math", SQLiteRecordStore())

    assert recipe.name == name
    assert recipe.report_type is report_type
    assert isinstance(trainer.processor, processor_type)
    if objective is None:
        assert trainer.candidate_backend is None
    else:
        assert isinstance(trainer.candidate_backend, RuntimeCandidateBackend)
        assert trainer.candidate_backend.objective == objective
    assert trainer.report_type is report_type


def test_build_rejects_an_unknown_objective_before_any_training_step() -> None:
    # Regression: the objective name used to be resolved only at the first
    # training step — after GPUs were already up. A recipe naming an objective
    # that no longer exists (the deleted online_grpo arm, a typo) must fail at
    # recipe build, with the registry's available-objectives message.
    @dataclass(frozen=True)
    class StaleObjectiveRecipe(WeightTrainingRecipe):
        _: KW_ONLY
        name: str = "stale"

        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(objective="online_grpo", processor=ThresholdProcessor)

    with pytest.raises(ValueError, match=r"unknown objective 'online_grpo'.*available objectives"):
        StaleObjectiveRecipe(**runtime_bindings(StubTrainingRuntime())).build("math", SQLiteRecordStore())


def test_tttd_build_resolves_its_backend_registered_objective_in_a_fresh_process() -> None:
    # A service process never imports the Slime driver, so the cookbook
    # package must register its objective when the dotted recipe is imported for eager objective
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
                "from reef.storage.sqlite import SQLiteRecordStore\n"
                "from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings\n"
                "trainer = TTTDRecipe(**runtime_bindings(StubTrainingRuntime()), groups_per_step=1, rollouts_per_group=2)"
                ".build('math', SQLiteRecordStore())\n"
                "assert trainer.candidate_backend.objective == 'tttd'\n"
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
        (SAORecipe(**runtime_bindings(StubTrainingRuntime())), {}),
        (
            TTTDRecipe(**runtime_bindings(StubTrainingRuntime()), groups_per_step=1, rollouts_per_group=2),
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
    # Unsupported multi-turn training fails explicitly and preserves its inputs.
    trainer = recipe.build("math", SQLiteRecordStore())
    processor = trainer.processor
    processor.ingest(_turn_inference("i1", [10, 20], -0.1))
    processor.ingest(_turn_inference("i2", [10, 20, 11, 21], -0.2))
    with pytest.raises(ValueError, match="accept_multi_turn"):
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
    assert "r1" not in processor.releasable_record_ids()


def test_openclawrl_recipe_never_trains_on_reports() -> None:
    # The judging half lives inside the processor now: a report is consumed
    # held for retention only — terminal on sight, releasable, never a candidate.
    trainer = OpenClawRLRecipe(**runtime_bindings(StubTrainingRuntime())).build("math", SQLiteRecordStore())
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
    decision = processor.releasable_record_ids()
    assert "r1" in decision
    assert "r1" in decision
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
        **runtime_bindings(runtime),
    )

    assert isinstance(recipe, TTTDRecipe)
    assert recipe.groups_per_step == 2
    assert recipe.rollouts_per_group == 3
    assert recipe.training_runtime is runtime
    assert recipe.runtime is runtime.inference


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
    recipe = build_recipe(settings["implementation"], {}, config=settings, **runtime_bindings(StubTrainingRuntime()))

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


def test_cookbook_objectives_signal_their_recipes_loss_family() -> None:
    # Both driver selection and batch preparation use the same method binding.
    from dataclasses import replace

    from reef.train.algos.registry import resolve_objective
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.types import TrainingBatch

    first = policy_trajectory("i1", (5, 1), (1,), (-0.1,), 0.5)
    second = first.with_metadata(source_agent_record_id="i2", reward=1.5)
    checked = set()
    for recipe_type in (OpenClawRLRecipe, SAORecipe, TTTDRecipe):
        spec = recipe_type.training_spec()
        assert spec.processor is not None
        if recipe_type is TTTDRecipe:
            batch = TrainingBatch(
                "b",
                tuple(
                    replace(sample, group_id=str(index))
                    for index, group in enumerate(((first, second),))
                    for sample in group
                ),
            )
        else:
            batch = TrainingBatch(
                "b",
                (
                    first,
                    second,
                ),
            )
        objective = resolve_objective(spec.objective)
        signal = objective.prepare(batch, {})
        assert signal.action == "train"
        assert objective.loss_family == spec.loss_family, recipe_type.__name__
        assert resolve_loss_family(objective.loss_family).loss_family == spec.loss_family
        checked.add(recipe_type.__name__)
    assert checked == {"OpenClawRLRecipe", "SAORecipe", "TTTDRecipe"}
