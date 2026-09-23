"""A composite recipe binds one recipe per component; the config component supplies request defaults."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.artifact.composite import compose_release
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.core.reports import ReportValidationError, ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import CompositeRecipe, Recipe, RecipeConfigError, build_recipe
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.config import recipe_config_from_mapping
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import Surface, TextFileTree, create_config_surface, create_harness_surface
from reef.surface.config import CONFIG_FILE, ConfigInferenceHooks, ConfigValidator
from reef.train import CandidateBackend, PreparedStep, Trainer, TrainStepResult
from reef.train.evaluation import EvaluationResult, UpdateCandidate

from ._threshold_processor import ThresholdProcessor


class _FileBackend(CandidateBackend):
    """Publish one text file per step for whatever component this recipe evolves."""

    def __init__(self, label: str, artifact_dir: Path) -> None:
        self.label = label
        self.artifact_dir = artifact_dir
        self.result: TrainStepResult | None = None

    def initial_state(self) -> Mapping[str, Any]:
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        step = int(state["steps"]) + 1
        path = self.artifact_dir / self.label / str(step)
        path.mkdir(parents=True)
        (path / f"{self.label}.txt").write_text(f"{self.label} step {step}", encoding="utf-8")
        self.result = TrainStepResult({"steps": step}, artifact=Artifact.local(path))
        return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})

    def evaluate(self, candidate):
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass


@dataclass(frozen=True)
class _TreeRecipe(Recipe):
    """A flat harness-like recipe: one pulled file tree, evolved by a local backend."""

    label: str = "tree"
    artifact_dir: Path = Path(".")
    seed: Mapping[str, str] | None = None

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface(served_model="served-model")

    def base_artifact_files(self) -> Mapping[str, str] | None:
        return self.seed

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_FileBackend(self.label, self.artifact_dir),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
        )


class _HybridThresholdProcessor(ThresholdProcessor):
    """The same processor, taking instructions too."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})


@dataclass(frozen=True)
class _HybridTreeRecipe(_TreeRecipe):
    """The tree recipe with a processor that runs in every mode."""

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: _HybridThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_FileBackend(self.label, self.artifact_dir),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
        )


@dataclass(frozen=True)
class _ScoredTreeRecipe(_TreeRecipe):
    """The same tree recipe, consuming scored rollout reports."""

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport


@dataclass(frozen=True)
class _ConfigRecipe(Recipe):
    """A flat configuration recipe with no trainer of its own."""

    def build_surface(self, scenario: str) -> Surface:
        return create_config_surface()

    def base_artifact_files(self) -> Mapping[str, str] | None:
        return {CONFIG_FILE: json.dumps({"request_defaults": {"temperature": 0.2}})}


def _composite(tmp_path: Path) -> CompositeRecipe:
    return CompositeRecipe(
        components={
            "harness": _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
            "config": _ConfigRecipe(),
        }
    )


@pytest.mark.unit
def test_composite_recipe_merges_surfaces_seeds_and_trainers(tmp_path: Path) -> None:
    recipe = _composite(tmp_path)
    surface = recipe.build_surface("agent")
    assert surface.names == ("harness", "config")
    assert surface.files_component == "harness"
    assert surface.harness is not None and surface.harness.served_model == "served-model"
    assert isinstance(surface.components["config"].validator, ConfigValidator)
    assert recipe.base_artifact_files() == {
        "harness/AGENTS.md": "seed",
        f"config/{CONFIG_FILE}": json.dumps({"request_defaults": {"temperature": 0.2}}),
    }
    assert recipe.checkpoint_strategy == EveryNVersions(1)
    assert recipe.report_type is None
    # No weight-training component: a bootstrap model snapshot would sit at the release root.
    assert recipe.bootstrap_artifact_component() is None
    trainers = recipe.build_trainers(
        "agent", SQLiteScenarioStorage().open("agent").records, surface=surface, algorithm_states={}
    )
    assert [bound.component for bound in trainers] == ["harness", "config"]
    assert trainers[0].trainer.candidate_backend is not None
    assert trainers[1].trainer.candidate_backend is None


@pytest.mark.unit
def test_composite_recipe_rejects_incoherent_components(tmp_path: Path) -> None:
    with pytest.raises(RecipeConfigError, match="at least two"):
        CompositeRecipe(components={"harness": _TreeRecipe()})
    with pytest.raises(RecipeConfigError, match="shares one mode"):
        CompositeRecipe(components={"harness": _TreeRecipe(training_mode="manual"), "config": _ConfigRecipe()})
    with pytest.raises(RecipeConfigError, match="not itself"):
        CompositeRecipe(components={"outer": _composite(tmp_path), "config": _ConfigRecipe()})
    with pytest.raises(RecipeConfigError, match="reserves"):
        CompositeRecipe(components={"step": _TreeRecipe(), "config": _ConfigRecipe()})
    two_trees = CompositeRecipe(components={"a": _TreeRecipe(label="a"), "b": _TreeRecipe(label="b")})
    with pytest.raises(RecipeConfigError, match="harness information"):
        two_trees.build_surface("agent")


@pytest.mark.unit
def test_composite_recipe_builds_from_config() -> None:
    recipe = build_recipe(
        "reef.recipe.composite:CompositeRecipe",
        {},
        config={
            "implementation": "reef.recipe.composite:CompositeRecipe",
            "model": {"path": "served-model"},
            "components": {
                "harness": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
                "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
            },
        },
    )
    assert isinstance(recipe, CompositeRecipe)
    assert sorted(recipe.components) == ["config", "harness"]
    assert isinstance(recipe.components["harness"], _TreeRecipe)
    with pytest.raises(RecipeConfigError, match=r"components\.step is a metric prefix .*rename the component"):
        build_recipe(
            "reef.recipe.composite:CompositeRecipe",
            {},
            config={
                "implementation": "x",
                "model": {"path": "served-model"},
                "components": {
                    "step": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
                    "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
                },
            },
        )
    with pytest.raises(RecipeConfigError, match="components"):
        build_recipe("reef.recipe.composite:CompositeRecipe", {}, config={"implementation": "x", "model": {}})
    with pytest.raises(RecipeConfigError, match="different training modes"):
        build_recipe(
            "reef.recipe.composite:CompositeRecipe",
            {},
            config={
                "implementation": "x",
                "model": {"path": "m"},
                "components": {
                    "harness": {
                        "implementation": "reef_service.test_composite_recipe:_TreeRecipe",
                        "data": {"training_mode": "manual"},
                    },
                    "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
                },
            },
        )


_WEIGHTS_AND_HARNESS = {
    "weights": {"implementation": "recipes.sao.recipe:SAORecipe"},
    "harness": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
}
_LOCAL_TRAINING = "reef_service._training_deployment:LocalDeployment"


@pytest.mark.unit
def test_composite_recipe_names_its_weight_training_component() -> None:
    from recipes.sao.recipe import SAORecipe

    selected = CompositeRecipe.select_weight_training({"components": _WEIGHTS_AND_HARNESS})
    assert selected is not None
    assert selected[0] is SAORecipe and selected[1] is _WEIGHTS_AND_HARNESS["weights"]
    assert CompositeRecipe.select_weight_training({"components": {"harness": _WEIGHTS_AND_HARNESS["harness"]}}) is None
    assert _TreeRecipe.select_weight_training({}) is None
    assert SAORecipe.select_weight_training({"data": {}}) == (SAORecipe, {"data": {}})
    # A release loads one component, so two weight components are refused before any runtime is connected.
    with pytest.raises(RecipeConfigError, match=r"\['weights', 'weights2'\] all train weights"):
        CompositeRecipe.select_weight_training(
            {"components": {**_WEIGHTS_AND_HARNESS, "weights2": _WEIGHTS_AND_HARNESS["weights"]}}
        )


@pytest.mark.unit
def test_service_connects_the_training_runtime_of_a_composite_weight_component() -> None:
    from reef_service._training_deployment import LocalRuntime

    from reef.service.assembly import _serving_recipe
    from reef.service.deploy.service_config import service_config_from_mapping

    settings = service_config_from_mapping(
        {
            "reef": {
                "recipe": "reef.recipe.composite:CompositeRecipe",
                "model_path": "demo-model",
                "training_backend": _LOCAL_TRAINING,
                "components": _WEIGHTS_AND_HARNESS,
            }
        }
    )
    recipe = _serving_recipe("reef.recipe.composite:CompositeRecipe", settings, {}, None)
    assert isinstance(recipe, CompositeRecipe)
    assert isinstance(recipe.training_runtime, LocalRuntime)
    assert recipe.training_runtime.received_model_path == "demo-model"
    assert all(component.training_runtime is recipe.training_runtime for component in recipe.components.values())
    assert all(component.runtime is recipe.runtime for component in recipe.components.values())

    # A flat weight recipe setting has no place at the top of a composite.
    stray = service_config_from_mapping(
        {
            "reef": {
                "recipe": "reef.recipe.composite:CompositeRecipe",
                "model_path": "demo-model",
                "training_backend": _LOCAL_TRAINING,
                "max_staleness": 2,
                "components": _WEIGHTS_AND_HARNESS,
            }
        }
    )
    with pytest.raises(ValueError, match=r"reef\.max_staleness is not a setting of CompositeRecipe"):
        _serving_recipe("reef.recipe.composite:CompositeRecipe", stray, {}, None)

    # A top level evaluation section reaches the component that trains: it is that component which parses it.
    evaluated = service_config_from_mapping(
        {
            "reef": {
                "recipe": "reef.recipe.composite:CompositeRecipe",
                "model_path": "demo-model",
                "training_backend": _LOCAL_TRAINING,
                "components": _WEIGHTS_AND_HARNESS,
            },
            "evaluation": {"module": "no.such.module:Factory"},
        }
    )
    with pytest.raises(RecipeConfigError, match=r"candidate evaluation plugin factory 'no\.such\.module:Factory'"):
        _serving_recipe("reef.recipe.composite:CompositeRecipe", evaluated, {}, None)


@pytest.mark.unit
def test_deployment_treats_a_composite_with_a_weight_component_as_training(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy.execution import validate_services
    from reef.service.deploy.orchestrator import resolve_deployment_config
    from reef.service.training_driver import _resolve_training_recipe

    raw = {
        "schema-version": 2,
        "recipe": {
            "implementation": "reef.recipe.composite:CompositeRecipe",
            "config": {"components": _WEIGHTS_AND_HARNESS},
        },
        "inference": {"model-path": "demo-model"},
        "training": {"backend": _LOCAL_TRAINING},
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend"] == _LOCAL_TRAINING
    assert [process["name"] for process in validate_services(config, "test")] == ["reef"]
    loss_family, name = _resolve_training_recipe(config)
    assert name == "reef.recipe.composite:CompositeRecipe"
    from recipes.sao.recipe import SAORecipe

    assert loss_family == SAORecipe.training_spec().loss_family

    # A component named through the environment is classified once the environment is applied.
    monkeypatch.setenv("PROBE_WEIGHTS", "recipes.sao.recipe:SAORecipe")
    interpolated = {
        **raw,
        "recipe": {
            "implementation": "reef.recipe.composite:CompositeRecipe",
            "config": {"components": {**_WEIGHTS_AND_HARNESS, "weights": {"implementation": "${PROBE_WEIGHTS}"}}},
        },
    }
    config, _ = resolve_deployment_config(interpolated, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend"] == _LOCAL_TRAINING

    # The driver reports an unimportable component the way it reports an unimportable recipe.
    broken = {
        "reef": {
            "recipe": "reef.recipe.composite:CompositeRecipe",
            "components": {"weights": {"implementation": "no.such.module:Recipe"}},
        }
    }
    with pytest.raises(RuntimeError, match=r"cannot load reef\.recipe"):
        _resolve_training_recipe(broken)


@pytest.mark.unit
def test_composite_recipe_reads_components_from_the_versioned_layout() -> None:
    settings = recipe_config_from_mapping(
        {
            "schema-version": 2,
            "recipe": {
                "implementation": "reef.recipe.composite:CompositeRecipe",
                "config": {
                    "components": {
                        "harness": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
                        "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
                    }
                },
            },
            "inference": {"upstream-model": "served-model"},
        }
    )
    assert sorted(settings["components"]) == ["config", "harness"]
    recipe = build_recipe(settings["implementation"], {}, config=settings)
    assert isinstance(recipe, CompositeRecipe)


@pytest.mark.unit
def test_composite_recipe_refuses_a_checkpoint_cadence() -> None:
    components = {
        "harness": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
        "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
    }
    base = {"implementation": "reef.recipe.composite:CompositeRecipe", "model": {"path": "served-model"}}
    with pytest.raises(RecipeConfigError, match=r"artifact\.checkpoint_every_n_versions has no effect"):
        build_recipe(
            base["implementation"],
            {},
            config={**base, "artifact": {"checkpoint_every_n_versions": 5}, "components": components},
        )
    with pytest.raises(RecipeConfigError, match=r"components\.harness\.artifact\.checkpoint_every_n_versions"):
        build_recipe(
            base["implementation"],
            {},
            config={
                **base,
                "components": {
                    **components,
                    "harness": {**components["harness"], "artifact": {"checkpoint_every_n_versions": 5}},
                },
            },
        )
    # The flat deployment spelling and the hyphenated one are refused too.
    with pytest.raises(RecipeConfigError, match=r"recipe\.checkpoint_every_n_versions has no effect"):
        build_recipe(
            base["implementation"], {}, config={**base, "checkpoint_every_n_versions": 5, "components": components}
        )
    with pytest.raises(RecipeConfigError, match=r"components\.harness\.artifact\.checkpoint-every-n-versions"):
        build_recipe(
            base["implementation"],
            {},
            config={
                **base,
                "components": {
                    **components,
                    "harness": {**components["harness"], "artifact": {"checkpoint-every-n-versions": 5}},
                },
            },
        )


@pytest.mark.unit
def test_composite_recipe_shares_one_runtime_resolved_from_the_environment() -> None:
    # No runtime is injected: the deployment relies on REEF_UPSTREAM_URL, as a flat recipe may.
    recipe = build_recipe(
        "reef.recipe.composite:CompositeRecipe",
        {"REEF_UPSTREAM_URL": "http://upstream.test"},
        config={
            "implementation": "reef.recipe.composite:CompositeRecipe",
            "model": {"path": "served-model"},
            "components": {
                "harness": {"implementation": "reef_service.test_composite_recipe:_TreeRecipe"},
                "config": {"implementation": "reef_service.test_composite_recipe:_ConfigRecipe"},
            },
        },
    )
    assert isinstance(recipe, CompositeRecipe)
    assert recipe.runtime is not None
    assert recipe.inference_handler is not None
    assert all(component.runtime is recipe.runtime for component in recipe.components.values())


@pytest.mark.unit
def test_composite_scenario_serves_config_defaults_and_reports_components(tmp_path: Path) -> None:
    recipe = _composite(tmp_path)
    initial = tmp_path / "initial"
    for relative, text in (recipe.base_artifact_files() or {}).items():
        (initial / relative).parent.mkdir(parents=True, exist_ok=True)
        (initial / relative).write_text(text, encoding="utf-8")
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        hooks = scenario.surface.inference
        assert hooks is not None
        artifact = Artifact(scenario.current_artifact_ref(), scenario.repository)
        prepared = hooks.prepare_request(artifact, "/v1/chat/completions", {"messages": [], "max_tokens": 5})
        assert prepared == {"temperature": 0.2, "messages": [], "max_tokens": 5}
        assert hooks.prepare_request(artifact, "/v1/chat/completions", {"temperature": 1.0}) == {"temperature": 1.0}

        scenario.records.append(
            AgentRecord.create(
                scenario="agent",
                request_type=RequestType.INFERENCE,
                payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
                agent_record_id="i1",
            )
        )
        scenario.records.append(
            AgentRecord.create(
                scenario="agent",
                request_type=RequestType.REPORT,
                payload={"score": 1.0, "references": ["i1"]},
                agent_record_id="r1",
                references=("i1",),
            )
        )
        result = scenario.prepare_training_step("harness")
        assert result is not None
        scenario.commit(result, component="harness")

        rows = scenario.releases()
        assert rows[0]["component"] == "harness"
        assert rows[0]["base_release_id"] == rows[1]["release_id"]
        assert "component" not in rows[1]
        # The config component runs no step, so it has no say over which rows are retired.
        assert scenario.store.history()[-1].compacted_ids == frozenset({"i1", "r1"})
        status = dispatcher.build_training_status()["scenarios"]["agent"]
        assert status["components"]["harness"]["last_committed_step"]["step"] == 1
        assert status["components"]["config"]["last_committed_step"] is None
        manifest = RequestService(dispatcher).harness_manifest({"x-reef-scenario": "agent"})
        assert manifest["files"] == {"harness.txt": "harness step 1"}
        assert set(manifest["components"]) == {"harness", "config"}
        assert manifest["components"]["harness"] == result.artifact.ref.content_id
    finally:
        dispatcher.close()


def _serve(recipe: CompositeRecipe, tmp_path: Path) -> Dispatcher:
    initial = tmp_path / "initial"
    for relative, text in (recipe.base_artifact_files() or {}).items():
        (initial / relative).parent.mkdir(parents=True, exist_ok=True)
        (initial / relative).write_text(text, encoding="utf-8")
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


@dataclass(frozen=True)
class _GridTreeRecipe(_TreeRecipe):
    """The same tree recipe, consuming rollouts addressed into a TTT-Discover grid."""

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        from recipes.tttd.report import TTTDGroupedRolloutReport

        return TTTDGroupedRolloutReport


@dataclass(frozen=True)
class _GridConfigRecipe(_ConfigRecipe):
    """The config recipe, declaring the TTT-Discover grid report."""

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        from recipes.tttd.report import TTTDGroupedRolloutReport

        return TTTDGroupedRolloutReport


@pytest.mark.unit
def test_components_with_different_report_contracts_admit_what_any_of_them_accepts(tmp_path: Path) -> None:
    """A weights recipe's grid report and a harness recipe's scored report share one scenario's ingress."""
    from recipes.tttd.report import TTTDGroupedRolloutReport

    recipe = CompositeRecipe(
        components={
            "weights": _GridTreeRecipe(label="weights", artifact_dir=tmp_path / "w"),
            "harness": _ScoredTreeRecipe(label="harness", artifact_dir=tmp_path / "h", seed={"AGENTS.md": "seed"}),
        }
    )
    report_type = recipe.report_type
    assert report_type is not None
    plain = {"score": 1.0, "references": ["i1"]}
    grid = {
        "score": 1.0,
        "references": ["i1"],
        "metadata": {
            "algorithm": "tttd",
            "step": 0,
            "group": 0,
            "rollout": 1,
            "groups_per_step": 1,
            "rollouts_per_group": 4,
        },
    }
    assert isinstance(report_type.from_dict(plain), ScoredRolloutReport)
    assert isinstance(report_type.from_dict(grid), TTTDGroupedRolloutReport)
    with pytest.raises(ReportValidationError, match=r"TTTDGroupedRolloutReport.*ScoredRolloutReport"):
        report_type.from_dict({"references": ["i1"]})
    # Two components with the same contract keep it as is.
    same = CompositeRecipe(
        components={
            "a": _ScoredTreeRecipe(label="a", artifact_dir=tmp_path / "a", seed={"AGENTS.md": "seed"}),
            "b": _ConfigRecipe(),
        }
    )
    assert same.report_type is ScoredRolloutReport
    # Every trainer of a composite is told what its scenario's ingress admits.
    mixed = CompositeRecipe(
        components={
            "harness": _ScoredTreeRecipe(label="harness", artifact_dir=tmp_path / "h", seed={"AGENTS.md": "seed"}),
            "config": _GridConfigRecipe(),
        }
    )
    assert mixed.report_type is not None and mixed.report_type.__name__.startswith("AnyOf")
    trainers = mixed.build_trainers(
        "agent",
        SQLiteScenarioStorage().open("agent").records,
        surface=mixed.build_surface("agent"),
        algorithm_states={},
    )
    assert [bound.trainer.processor.context.admitted_report_type is mixed.report_type for bound in trainers] == [
        True,
        True,
    ]
    assert trainers[0].trainer.report_type is ScoredRolloutReport


@pytest.mark.unit
def test_composite_scenario_enforces_the_agreed_report_type_whatever_the_component_order(tmp_path: Path) -> None:
    # The component with no report contract is listed first; the harness contract still guards ingress.
    recipe = CompositeRecipe(
        components={
            "config": _ConfigRecipe(),
            "harness": _ScoredTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
        }
    )
    assert recipe.report_type is ScoredRolloutReport
    dispatcher = _serve(recipe, tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert scenario.report_type is ScoredRolloutReport
        scoreless = AgentRecord.create(
            scenario="agent",
            request_type=RequestType.REPORT,
            payload={"references": ["i1"]},
            agent_record_id="r1",
            references=("i1",),
        )
        with pytest.raises(ReportValidationError):
            dispatcher.accept_record(scoreless)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_config_validator_requires_a_json_object(tmp_path: Path) -> None:
    root = tmp_path / "config"
    root.mkdir()
    with pytest.raises(ReefError, match=r"requires config\.json"):
        ConfigValidator().validate(Artifact.local(root))
    (root / CONFIG_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(ReefError, match="JSON object"):
        ConfigValidator().validate(Artifact.local(root))
    (root / CONFIG_FILE).write_text(json.dumps({"request_defaults": 3}), encoding="utf-8")
    with pytest.raises(ReefError, match="request_defaults must be an object"):
        ConfigValidator().validate(Artifact.local(root))
    (root / CONFIG_FILE).write_text(json.dumps({"request_defaults": {"top_p": 0.9}}), encoding="utf-8")
    ConfigValidator().validate(Artifact.local(root))
    (root / CONFIG_FILE).write_text(json.dumps({"request_defaults": {"/v1/responses": 3}}), encoding="utf-8")
    with pytest.raises(ReefError, match="/v1/responses must be an object"):
        ConfigValidator().validate(Artifact.local(root))


class _EchoHandler(InferenceHandler):
    """Answer every chat completion with one word, so a test can tell served from recorded."""

    async def inference(self, artifact, path, payload):
        return {
            "id": "chat-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    async def inference_stream(self, artifact, path, payload):
        raise NotImplementedError


@pytest.mark.unit
def test_an_evaluation_call_is_served_and_kept_by_nobody(tmp_path: Path) -> None:
    dispatcher = _serve(_composite(tmp_path), tmp_path)
    try:
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}

        async def run() -> None:
            response, item = await service.infer_with_data(
                headers, {"messages": []}, "/v1/chat/completions", _EchoHandler(), record=False
            )
            assert item is None and response["choices"][0]["message"]["content"] == "ok"
            _, item = await service.infer_with_data(headers, {"messages": []}, "/v1/chat/completions", _EchoHandler())
            assert item is not None

        asyncio.run(run())
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.records.count("agent") == 1
        # An evaluation call is measured apart from served traffic.
        snapshot = scenario.operations.snapshot()
        assert snapshot["evaluate/request/completed_total"] == 1
        assert snapshot["serve/request/completed_total"] == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_flat_scenario_without_a_durable_log_still_serves_its_head(tmp_path: Path) -> None:
    """The harness head of a flat scenario is the served release, found on the chain when the log keeps no rows."""
    tree = tmp_path / "seed"
    tree.mkdir()
    (tree / "AGENTS.md").write_text("seed")
    dispatcher = Dispatcher(
        _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
        InMemoryRepositoryBackend.factory(tree, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(None),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and not scenario.store.durable
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "AGENTS.md").write_text("evolved")
        scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(evolved)))
        head = scenario.current_artifact_ref().release_id
        assert service.harness_head(headers) == head
        manifest = service.harness_manifest(headers)
        assert manifest["release_id"] == head and manifest["files"] == {"AGENTS.md": "evolved"}
        assert f'"release_id": "{head}"' in service.harness_install_script(headers, adapter="pi")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_evaluation_route_names_the_scenario_in_its_path(tmp_path: Path) -> None:
    dispatcher = _serve(_composite(tmp_path), tmp_path)
    try:

        async def run() -> None:
            client = TestClient(TestServer(create_app(dispatcher, inference_handler=_EchoHandler())))
            await client.start_server()
            try:
                response = await client.post(
                    "/reef/scenarios/agent/evaluation/v1/chat/completions", json={"messages": []}
                )
                assert response.status == 200
                assert "x-reef-agent-record-id" not in response.headers
                assert (await response.json())["choices"][0]["message"]["content"] == "ok"
                missing = await client.post("/reef/scenarios/agent/evaluation/v1/nothing", json={})
                assert missing.status == 404
            finally:
                await client.close()

        asyncio.run(run())
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.records.count("agent") == 0
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_component_without_a_step_does_not_hold_the_training_mode(tmp_path: Path) -> None:
    """The config component builds no batch, so the harness alone decides whether instructions are taken."""
    recipe = CompositeRecipe(
        components={
            "harness": _HybridTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
            "config": _ConfigRecipe(),
        }
    )
    dispatcher = _serve(recipe, tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert dispatcher.set_training_mode("agent", "hybrid") == {"scenario": "agent", "training_mode": "hybrid"}
        assert scenario.training_mode == "hybrid"
        assert scenario.trainer_for("harness").training_mode == "hybrid"
        assert scenario.trainer_for("config").training_mode == "auto"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_config_defaults_follow_the_route(tmp_path: Path) -> None:
    """Shared fields reach every generation route; a route's own entry wins there; a token count takes none."""
    root = tmp_path / "config"
    root.mkdir()
    defaults = {
        "temperature": 0.2,
        "/v1/chat/completions": {"max_tokens": 256},
        "/v1/responses": {"max_output_tokens": 256, "temperature": 0.5},
    }
    (root / CONFIG_FILE).write_text(json.dumps({"request_defaults": defaults}), encoding="utf-8")
    artifact = Artifact.local(root)
    hooks = ConfigInferenceHooks()
    assert hooks.prepare_request(artifact, "/v1/chat/completions", {"messages": []}) == {
        "temperature": 0.2,
        "max_tokens": 256,
        "messages": [],
    }
    assert hooks.prepare_request(artifact, "/v1/responses", {"input": "x"}) == {
        "temperature": 0.5,
        "max_output_tokens": 256,
        "input": "x",
    }
    assert hooks.prepare_request(artifact, "/v1/messages", {"messages": []}) == {"temperature": 0.2, "messages": []}
    count = {"messages": [], "model": "m"}
    assert hooks.prepare_request(artifact, "/v1/messages/count_tokens", count) == count


@pytest.mark.unit
def test_composed_release_links_carried_files_and_tolerates_an_empty_component(tmp_path: Path) -> None:
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter.bin").write_bytes(b"\x00" * 16)
    harness = Artifact.local(tmp_path / "missing")
    composed = compose_release({"weights": Artifact.local(weights), "harness": harness}, directory=tmp_path / "r")
    linked = tmp_path / "r" / "weights" / "adapter.bin"
    assert linked.read_bytes() == b"\x00" * 16
    assert linked.stat().st_nlink == 2
    assert (tmp_path / "r" / "harness").is_dir()
    assert TextFileTree().read_files(composed.component("harness")) is None


@pytest.mark.unit
def test_status_and_contract_speak_of_the_trainer_that_steps(tmp_path: Path) -> None:
    """A component listed first that runs no step does not lend its mode and processor to the whole scenario."""
    recipe = CompositeRecipe(
        components={
            "config": _ConfigRecipe(),
            "harness": _HybridTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
        }
    )
    dispatcher = _serve(recipe, tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        dispatcher.set_training_mode("agent", "hybrid")
        harness = scenario.trainer_for("harness")
        assert scenario.trainer is harness
        block = dispatcher.build_training_status()["scenarios"]["agent"]
        assert block["training_mode"] == "hybrid"
        assert block["processor"] == {
            **harness.processor_status(),
            "pending_instructions": harness.pending_instructions(),
        }
        contract = dispatcher.scenario_contract("agent")
        assert contract["training_mode"] == "hybrid"
        assert contract["processor"] == type(harness.processor).__name__
    finally:
        dispatcher.close()
