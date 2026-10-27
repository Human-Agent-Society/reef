from __future__ import annotations

import subprocess
import sys
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import pytest
import tests
from tests.support._threshold_processor import ThresholdJobData, ThresholdSamples
from tests.support._trajectories import policy_trajectory
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, job_sized_stages, runtime_bindings

from recipes.openclawrl import OpenClawRLJobData, OpenClawRLRecipe
from recipes.sao import SAOJobData, SAORecipe
from recipes.tttd import TTTDGroupedRolloutReport, TTTDJobData, TTTDRecipe
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe, RecipeConfigError, WeightTrainingRecipe, WeightTrainingSpec, load_recipe_config
from reef.recipe.config import recipe_config_from_mapping
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for
from reef.sample2job import NoJobData
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.jobs import FixedJobPlan, JobPlanContext
from reef.train.pipeline import PipelineConfig, PipelineSpec, TrainingPipeline
from reef.train.types import JobData


def pipeline_stages(recipe_type: type[Recipe], section: dict) -> PipelineConfig:
    """The recipe's stages configured by a deployment ``pipeline`` section, with jobs of one sample."""
    return job_sized_stages(recipe_type.pipeline_spec(), section)


def tttd_grid(groups_per_step: int, rollouts_per_group: int) -> PipelineConfig:
    return pipeline_stages(
        TTTDRecipe,
        {"record2record": {"config": {"groups_per_step": groups_per_step, "rollouts_per_group": rollouts_per_group}}},
    )


@pytest.mark.unit
@pytest.mark.parametrize("recipe_type", [OpenClawRLRecipe, SAORecipe, TTTDRecipe])
def test_training_recipes_take_max_staleness_in_their_record_stage_config(recipe_type) -> None:
    default = recipe_type(**runtime_bindings(), pipeline_stages=pipeline_stages(recipe_type, {}))
    assert default.build("scenario", SQLiteRecordStore()).pipeline.record2record.max_staleness == 0
    section = {"record2record": {"config": {"max_staleness": 2}}}
    recipe = recipe_type(**runtime_bindings(), pipeline_stages=pipeline_stages(recipe_type, section))
    assert recipe.build("scenario", SQLiteRecordStore()).pipeline.record2record.max_staleness == 2
    negative = {"record2record": {"config": {"max_staleness": -1}}}
    refused = recipe_type(**runtime_bindings(), pipeline_stages=pipeline_stages(recipe_type, negative))
    with pytest.raises(RecipeConfigError, match="max_staleness must be a non-negative int"):
        refused.build("scenario", SQLiteRecordStore())


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
        ("reef.recipe.base:config_field", "is not a Recipe class"),
        (":Recipe", "must be 'package.module:ClassName'"),
    ],
)
def test_dotted_recipe_rejects_bad_references(reference: str, match: str) -> None:
    with pytest.raises(RecipeConfigError, match=match):
        recipe_class_for(reference)


@pytest.mark.parametrize(
    ("recipe", "name", "sample2job_type", "objective", "report_type"),
    [
        (Recipe(), "recipe", NoJobData, None, None),
        (
            OpenClawRLRecipe(
                **runtime_bindings(),
                pipeline_stages=pipeline_stages(OpenClawRLRecipe, {"sample2job": {"config": {"buffer_size": 3}}}),
            ),
            "openclawrl",
            OpenClawRLJobData,
            "openclawrl",
            None,
        ),
        (
            SAORecipe(
                **runtime_bindings(),
                pipeline_stages=pipeline_stages(SAORecipe, {"sample2job": {"config": {"buffer_size": 4}}}),
            ),
            "sao",
            SAOJobData,
            "sao",
            ScoredRolloutReport,
        ),
        (
            TTTDRecipe(**runtime_bindings(), pipeline_stages=tttd_grid(2, 3)),
            "tttd",
            TTTDJobData,
            "tttd",
            TTTDGroupedRolloutReport,
        ),
    ],
)
def test_concrete_recipe_builds_its_processor_objective_and_report_type(
    recipe, name, sample2job_type, objective, report_type
) -> None:
    trainer = recipe.build("math", SQLiteRecordStore())

    assert recipe.name == name
    assert recipe.report_type is report_type
    assert type(trainer.pipeline) is TrainingPipeline
    assert isinstance(trainer.pipeline.sample2job, sample2job_type)
    if objective is None:
        assert trainer.job_loop is None
    else:
        assert trainer.job_loop is not None
        plan = trainer.job_loop.plan
        assert isinstance(plan, FixedJobPlan)
        request = plan.next_job(JobPlanContext(JobData("math:1"), 0, None))
        assert request.parameters["method"]["objective"] == objective
    assert trainer.pipeline.context.report_type is report_type


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
            return WeightTrainingSpec(objective="online_grpo")

        @classmethod
        def pipeline_spec(cls) -> PipelineSpec:
            return PipelineSpec(record2sample=ThresholdSamples, sample2job=ThresholdJobData)

    with pytest.raises(ValueError, match=r"unknown objective 'online_grpo'.*available objectives"):
        StaleObjectiveRecipe(**runtime_bindings()).build("math", SQLiteRecordStore())


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
                f"import sys; sys.path.insert(0, {str(Path(tests.__file__).parent.parent)!r})\n"
                "from recipes.tttd import TTTDRecipe\n"
                "from reef.storage.sqlite import SQLiteRecordStore\n"
                "from reef.train.pipeline import PipelineConfig\n"
                "from tests.support.runtime_stubs import runtime_bindings\n"
                "grid = {'record2record': {'config': {'groups_per_step': 1, 'rollouts_per_group': 2}},"
                " 'sample2job': {'config': {'job_samples': 2}}}\n"
                "stages = PipelineConfig.from_setting(grid, spec=TTTDRecipe.pipeline_spec(), environ={})\n"
                "trainer = TTTDRecipe(**runtime_bindings(), pipeline_stages=stages)"
                ".build('math', SQLiteRecordStore())\n"
                "assert trainer.job_loop.plan.request.parameters['method']['objective'] == 'tttd'\n"
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
        (SAORecipe(**runtime_bindings(), pipeline_stages=pipeline_stages(SAORecipe, {})), {}),
        (
            TTTDRecipe(**runtime_bindings(), pipeline_stages=tttd_grid(1, 2)),
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
    pipeline = trainer.pipeline
    pipeline.push(_turn_inference("i1", [10, 20], -0.1))
    pipeline.push(_turn_inference("i2", [10, 20, 11, 21], -0.2))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        pipeline.push(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.REPORT,
                agent_record_id="r1",
                payload={"score": 1.0, "references": ["i1", "i2"], "metadata": metadata},
                references=("i1", "i2"),
            )
        )

    assert not pipeline.ready(1)
    assert "r1" not in pipeline.releasable_record_ids()


def test_openclawrl_recipe_never_trains_on_reports() -> None:
    # The judging half lives inside the pipeline's record stage: a report is consumed
    # held for retention only — terminal on sight, releasable, never a candidate.
    trainer = OpenClawRLRecipe(**runtime_bindings(), pipeline_stages=pipeline_stages(OpenClawRLRecipe, {})).build(
        "math", SQLiteRecordStore()
    )
    pipeline = trainer.pipeline
    pipeline.push(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            agent_record_id="r1",
            payload={"score": 1.0, "references": ["i1"]},
            references=("i1",),
        )
    )
    assert not pipeline.ready(1)
    decision = pipeline.releasable_record_ids()
    assert "r1" in decision
    assert "r1" in decision
    trainer.close()


def test_recipe_factory_keeps_method_configuration_inside_recipe() -> None:
    component = StubTrainingComponent()
    runtime = StubInferenceRuntime()
    recipe = build_recipe(
        "recipes.tttd.recipe:TTTDRecipe",
        {"REEF_TRAINER_URL": "http://trainer:8901", "REEF_TRAINER_TOKEN": "secret"},
        config={"pipeline": {"record2record": {"config": {"groups_per_step": 2, "rollouts_per_group": 3}}}},
        **runtime_bindings(component, runtime),
    )

    assert isinstance(recipe, TTTDRecipe)
    record2record = recipe.build_pipeline(PipelineContext("math")).record2record
    assert (record2record.groups_per_step, record2record.rollouts_per_group) == (2, 3)
    assert recipe.training_component is component
    assert recipe.runtime is runtime


def test_recipe_factory_refuses_a_stage_setting_in_the_recipe_config() -> None:
    # The step grid is the record stage's config, not the recipe's.
    with pytest.raises(RecipeConfigError, match="does not consume"):
        build_recipe(
            "recipes.tttd.recipe:TTTDRecipe",
            {},
            config={"data": {"groups_per_step": 2}},
            **runtime_bindings(),
        )


def test_training_recipe_requires_its_training_component() -> None:
    with pytest.raises(RecipeConfigError, match="requires a training component"):
        build_recipe("recipes.tttd.recipe:TTTDRecipe", {})


def test_recipe_config_routes_sections_to_their_consumers() -> None:
    settings = recipe_config_from_mapping(
        {
            "implementation": "recipes.openclawrl.recipe:OpenClawRLRecipe",
            "pipeline": {"sample2job": {"config": {"buffer_size": 4}}},
            "model": {"path": "/models/qwen"},
        }
    )
    recipe = build_recipe(settings["implementation"], {}, config=settings, **runtime_bindings())

    assert recipe.stages().sample2job.config == {"buffer_size": 4}


def test_recipe_config_requires_implementation(tmp_path) -> None:
    path = tmp_path / "recipes.yaml"
    path.write_text("recipe:\n  config:\n    training_mode: auto\n")

    with pytest.raises(RecipeConfigError, match=r"recipe\.implementation must name a recipe class"):
        load_recipe_config(path)


def test_named_recipe_resolves_preset_from_filename(tmp_path) -> None:
    path = tmp_path / "thorough.yaml"
    path.write_text(
        """recipe:
  implementation: recipe
  runtime:
    type: inference_proxy
    base_url: http://provider
    api_key: secret
inference:
  upstream-model: Qwen/Qwen3.5-27B
"""
    )

    recipe = build_named_recipe("thorough", config_directory=tmp_path)

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe.runtime, InferenceProxyRuntime)


def test_config_backed_training_recipe_requires_an_injected_training_component(tmp_path) -> None:
    path = tmp_path / "sao-qwen.yaml"
    path.write_text(
        """recipe:
  implementation: recipes.sao.recipe:SAORecipe
inference:
  model-path: Qwen/Qwen3.5-27B
"""
    )
    with pytest.raises(RecipeConfigError, match="requires a training component"):
        build_named_recipe("sao-qwen", config_directory=tmp_path)


def test_named_recipe_without_config_directory_resolves_core_recipe_only() -> None:
    assert isinstance(build_named_recipe("recipe", {}), Recipe)
    with pytest.raises(RecipeConfigError, match="unknown deployment recipe 'thorough'"):
        build_named_recipe("thorough", {})


def test_named_recipe_reads_the_config_directory_from_the_environment(tmp_path) -> None:
    (tmp_path / "thorough.yaml").write_text(
        "recipe:\n  implementation: recipe\ninference:\n  upstream-model: Qwen/Qwen3.5-27B\n"
    )

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
        "recipe:\n"
        "  implementation: recipe\n"
        "  runtime:\n"
        "    type: inference_proxy\n"
        "    base_url: http://127.0.0.1:30000\n"
        "    timeout_s: 300\n"
        "inference:\n"
        "  upstream-model: Qwen/Qwen3.6-27B\n"
    )
    local = build_named_recipe("local-proxy", config_directory=tmp_path)

    settings = load_recipe_config(tmp_path / "local-proxy.yaml")
    assert settings["model"]["path"] == "Qwen/Qwen3.6-27B"
    assert isinstance(local.runtime, InferenceProxyRuntime)
    assert local.runtime.model_path == "Qwen/Qwen3.6-27B"

    # The hosted-provider shape (api_key_env) needs no built-in example: any
    # directory of named YAMLs resolves the same way.
    (tmp_path / "hosted-proxy.yaml").write_text(
        "recipe:\n"
        "  implementation: recipe\n"
        "  runtime:\n"
        "    type: inference_proxy\n"
        "    base_url: https://api.minimax.io/anthropic\n"
        "    api_key_env: ANTHROPIC_AUTH_TOKEN\n"
        "    timeout_s: 300\n"
        "inference:\n"
        "  upstream-model: MiniMax-M3\n"
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
    from reef.train.types import JobData

    first = policy_trajectory("i1", (5, 1), (1,), (-0.1,), 0.5)
    second = first.with_metadata(source_agent_record_id="i2", reward=1.5)
    checked = set()
    for recipe_type in (OpenClawRLRecipe, SAORecipe, TTTDRecipe):
        spec = recipe_type.training_spec()
        assert recipe_type.pipeline_spec().sample2job is not NoJobData
        if recipe_type is TTTDRecipe:
            batch = JobData(
                "b",
                tuple(
                    replace(sample, group_id=str(index))
                    for index, group in enumerate(((first, second),))
                    for sample in group
                ),
            )
        else:
            batch = JobData(
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


def test_opd_recipe_uses_configured_runtime() -> None:
    from recipes.opd import OPDRecipe

    component = StubTrainingComponent()
    recipe = build_recipe(
        "recipes.opd.recipe:OPDRecipe",
        {},
        config={
            "pipeline": {"sample2job": {"config": {"job_samples": 4, "buffer_size": 16}}},
        },
        **runtime_bindings(component),
    )
    assert isinstance(recipe, OPDRecipe)
    assert recipe.training_component is component
    sample2job = recipe.build_pipeline(PipelineContext("math")).sample2job
    assert (sample2job.job_samples, sample2job.buffer_size) == (4, 16)


def test_cordis_recipe_reads_the_evolution_batch_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.recipe.cordis import CordisRecipe

    (tmp_path / "batch_size_strategies.py").write_text(
        "from reef.evaluation.harness import EpisodeScorer\n"
        "from reef.train.cordis_backend.strategies import Proposer\n\n"
        "class Propose(Proposer):\n"
        "    def propose(self, nodes, samples, models, **keywords):\n        return None\n\n"
        "class Evaluate(EpisodeScorer):\n"
        "    def score(self, task, result):\n        return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def evolution(**settings: object) -> dict[str, object]:
        return {
            "evolution": {
                "propose": "batch_size_strategies:Propose",
                "evaluate": "batch_size_strategies:Evaluate",
                "tasks": ["task one"],
                **settings,
            }
        }

    assert CordisRecipe.from_environment({}, config=evolution()).batch_size == 1
    assert CordisRecipe.from_environment({}, config=evolution(batch_size=3)).batch_size == 3
    for invalid in (0, -2, 2.5, "3", True):
        with pytest.raises(RecipeConfigError, match=r"evolution\.batch_size must be an integer of at least 1"):
            CordisRecipe.from_environment({}, config=evolution(batch_size=invalid))
