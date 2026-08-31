from __future__ import annotations

import importlib.util

import pytest

import reef
from reef.train import processors


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "reef.session_index",
        "reef.groups",
        "reef.data_source",
        "reef.consumer",
        "reef.trainer_gateway",
        "reef.resolver",
        "reef.client",
        "reef.run",
    ],
)
def test_lifecycle_modules_are_removed(module: str) -> None:
    assert importlib.util.find_spec(module) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "symbol",
    [
        "SessionHandle",
        "AttemptHandle",
        "TraceKey",
        "RolloutGroupSpec",
        "SessionIndex",
        "ReefClient",
        "ReefClientError",
    ],
)
def test_lifecycle_symbols_are_not_exported(symbol: str) -> None:
    assert not hasattr(reef, symbol)


@pytest.mark.unit
def test_recipe_symbols_are_removed_in_favor_of_processors_and_trainer() -> None:
    assert not hasattr(reef, "TrainerRecipe")
    assert not hasattr(reef, "TrainerRecipeRuntime")
    assert hasattr(reef, "Trainer")


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "reef.routing",
        "reef.core.records",
        "reef.scenarios.records",
        "reef.trainer.managers",
        "reef.trainer.pipelines",
        "reef.trainer.recipe",
        "reef.trainer.recipes",
        "reef.trainer.algorithms",
        "reef.trainer.processors.noop",
        "reef.trainer",
        "reef.trainer.trainer",
        "reef.trainer.processors",
        "reef.backends",
        "reef.backends.base",
        "reef.backends.harness",
        "reef.agent_record_store",
        # Alternate module spellings stay absent: the entity is AgentRecord and
        # its store is RecordStore, both exported from their canonical modules.
        "reef.core.agent_record",
        "reef.service.routes.agent_record",
    ],
)
def test_old_naming_modules_are_removed(module: str) -> None:
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    assert spec is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "symbol",
    [
        "ScenarioRouter",
        "TypedRecord",
        "ScenarioRecordStore",
        "ScenarioRuntime",
        "TrainingDataManager",
        "WeightTrainingRecipe",
        "BackendRecipeLoader",
        "TypedRequestService",
        "DuplicateRecordError",
        "WeightUpdatePipeline",
        "Pipeline",
        # Renamed in the records vocabulary unification.
        "AgentData",
        "AgentDataStore",
        "AgentDataCodec",
        "AgentDataConflict",
        # Inlined into RecordStore as private methods; never a public class again.
        "RecordCodec",
    ],
)
def test_old_naming_symbols_are_not_exported(symbol: str) -> None:
    assert not hasattr(reef, symbol)


@pytest.mark.unit
def test_readme_names_are_exported() -> None:
    assert reef.Dispatcher is not None
    assert reef.AgentRecord is not None
    assert reef.RecordStore is not None
    assert reef.Scenario is not None
    assert reef.DataProcessor is not None
    assert reef.Trainer is not None
    assert reef.Recipe is not None


@pytest.mark.unit
def test_dead_accessors_and_symbols_are_removed() -> None:
    import reef.core
    import reef.core.errors
    from reef.dispatcher import Dispatcher
    from reef.scenario.scenario import Scenario
    from reef.service.request_service import RequestService
    from reef.surface.base import Surface

    # Removed dead code (no production callers).
    assert not hasattr(reef.core, "ReferenceStatus")
    assert not hasattr(reef.core.errors, "ConfigurationError")
    assert not hasattr(Scenario, "advance_to")
    assert not hasattr(RequestService, "current_files")
    # Test-only accessors migrated to canonical paths.
    assert not hasattr(Dispatcher, "recipes")
    assert not hasattr(Scenario, "checkpoint_strategy")
    # Capabilities are explicit fields rather than inferred method overrides.
    assert not hasattr(Surface, "kind")
    assert not hasattr(Surface, "validator")
    assert Surface().files is None


@pytest.mark.unit
def test_naming_unification_renames_are_complete() -> None:
    """One concept, one name: the renamed-away spellings must stay gone."""
    import reef.records
    import reef.surface.base
    from reef.recipe.base import WeightTrainingRecipe
    from reef.scenario.scenario import Scenario
    from reef.train.trainer import Trainer

    # The step-preparer id is no longer called "algorithm".
    assert not hasattr(WeightTrainingRecipe, "algorithm")
    assert WeightTrainingRecipe.training_spec().step_preparer == ""
    assert not hasattr(Trainer, "algorithm")
    assert not hasattr(Trainer, "step_preparer")
    assert not hasattr(Scenario, "algorithm")
    assert not hasattr(Scenario, "step_preparer")
    # Surface-side protocols use serving vocabulary without duplicating the
    # concrete runtime package's InferenceRuntime / TrainingRuntime names.
    for name in ("SurfaceRuntime", "TrainingSurfaceRuntime", "ServingHost", "TrainingHost"):
        assert not hasattr(reef.surface.base, name)
    for name in ("ServingRuntime", "WeightRuntime"):
        assert hasattr(reef.surface.base, name)
    # Route registration follows the records vocabulary.
    import reef.service.routes as routes

    assert not hasattr(routes, "register_agent_record_routes")
    assert hasattr(routes, "register_record_routes")
    # The Ray runtime speaks of one train group handle.
    from reef.runtime.adapters.ray_runtime import RayRuntime

    assert not hasattr(RayRuntime, "train_group")
    assert hasattr(RayRuntime, "train_group_handle")


@pytest.mark.unit
def test_surface_hook_bag_and_dead_adapter_residency_are_removed() -> None:
    import reef.surface as surfaces

    for name in (
        "EvolutionSurface",
        "LayerModule",
        "AdapterResidency",
        "AdapterRuntime",
        "AdapterSurface",
        "HarnessSurface",
        "ResidencyDecision",
        "SkillSurface",
        "WeightSurface",
    ):
        assert not hasattr(surfaces, name)


@pytest.mark.unit
def test_recipe_config_field_declarations_replaced_the_config_spelling_quartet() -> None:
    """One config field, one spelling: the field declaration
    (reef.recipe.config_fields.config_field)
    is the YAML key, env name, and parser; the parallel spellings stay gone."""
    import reef.recipe.config as recipe_config
    from reef.recipe.base import WeightTrainingRecipe

    # The per-recipe key list and the per-recipe field hook are replaced by
    # config_field() metadata.
    assert not hasattr(WeightTrainingRecipe, "data_settings")
    assert not hasattr(WeightTrainingRecipe, "_config_fields")
    # The hand-rolled env/config parser helpers the config field parsers replaced.
    for name in ("env_positive_int", "env_float", "config_float"):
        assert not hasattr(recipe_config, name)


@pytest.mark.unit
def test_online_grpo_preparer_and_recipe_are_removed() -> None:
    """Neither the online_grpo step preparer nor its recipe class may come back.

    Grouped preparers are cookbook-owned now; the grouped machinery (group
    keys, slots, the decide_group barrier in the reported-feedback processor) stays. The
    ``online_rft`` kind that replaced this arm is gone too — a filtered-SFT
    data recipe, with nothing in it addressing drift on a continually updated
    policy."""
    from reef.recipe.registry import recipe_class_for
    from reef.train.algos.registry import resolve_preparer

    with pytest.raises(ValueError, match="unknown step preparer"):
        resolve_preparer("online_grpo")
    assert recipe_class_for("online_grpo") is None
    assert recipe_class_for("online_rft") is None


@pytest.mark.unit
def test_noop_naming_is_removed_in_favor_of_recipe() -> None:
    assert not hasattr(reef, "NoopRecipe")
    assert not hasattr(reef.Dispatcher, "noop")
    assert not hasattr(processors, "NoopProcessor")
    assert importlib.util.find_spec("reef.noop") is None


@pytest.mark.unit
def test_pairing_interlock_surface_is_replaced_by_the_engine_spec() -> None:
    """The abstract subclassing base gave way to one spec-configured processor.

    Recipe processors keep their public class names as thin adapters over
    ``ReportedFeedbackProcessor``; the interlock vocabulary (``PairedCandidate``, the
    ``_collect_candidates``/``_terminal_report_ids`` dual bookkeeping) must
    stay gone.
    """
    from reef.train.processors import ReportedFeedbackProcessor, reported

    assert not hasattr(reported, "PairedCandidate")
    for hook in ("_collect_candidates", "_terminal_report_ids", "_terminal_source_ids", "_create_batch"):
        assert not hasattr(ReportedFeedbackProcessor, hook)
