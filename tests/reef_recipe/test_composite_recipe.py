"""A composite recipe binds one recipe per component; the config component holds request defaults."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_jobs import harness_recipe
from tests.support.runtime_stubs import StubFacts, StubTrainingComponent, job_sized_stages
from tests.support.samples import data_storage
from tests.support.samples_service import make_samples
from tests.support.serving import wait_for_serving
from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples

from reef.artifact import Artifact, DefaultStart, GitLFSRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.artifact.artifact import ArtifactValidator
from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
from reef.artifact.composite import compose_snapshot
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginReference
from reef.core.reports import ReportValidationError, ScoredRolloutReport
from reef.core.snapshot import RootSource
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate, NeverActivate
from reef.evaluation.harness import EvaluationSettings
from reef.execution.workers import ServiceSet
from reef.harness.adapters import get_adapter
from reef.harness.sync import HARNESS_COMPONENT, HarnessFiles
from reef.inference.http import InferenceProxyRuntime, resolve_proxy_runtime
from reef.inference.interfaces import InferenceHandler
from reef.pipeline_context import PipelineContext
from reef.recipe import CompositeRecipe, Recipe, RecipeConfigError, build_recipe
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.recipe.loading import weight_training_selection
from reef.scenario.factory import read_trainer_specs, scenario_report_type, with_record2sample_configs
from reef.scenario.live_objects import (
    LiveObjects,
    ServedEndpoint,
    check_root_start,
    declared_components,
    default_start,
    harness_served_model,
    scenario_components,
)
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.storage.samples import consumed_sample_id, stage_item_id
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import TrainingCommit
from reef.train.component import SavedCheckpoint, ScenarioRoot, TrainingJob
from reef.train.harness_backend.settings import EvolutionSettings
from reef.train.pipeline import PipelineSettings, PipelineSpec
from reef.train.trainer_builder import ComponentTrainerSpecs, SpecTrainerBuilder, TrainerSpec


class _FileJobs(StubTrainingComponent):
    """A training component whose ``n``-th job saves one text file, ``<label>.txt`` = ``<label> step <n>``, for
    whatever component this recipe evolves."""

    def __init__(self, label: str, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name=label, shares_scenarios=True, artifact_format=None)
        self.label = label

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        step = len(self.submitted)
        path = self.checkpoint_root / self.label / str(step)
        path.mkdir(parents=True)
        (path / f"{self.label}.txt").write_text(f"{self.label} step {step}", encoding="utf-8")
        return (SavedCheckpoint(0, str(path), str(path)),)


@dataclass(frozen=True)
class _TreeRecipe(Recipe):
    """A flat harness-like recipe: one pulled file tree, evolved by a local backend."""

    label: str = "tree"
    artifact_dir: Path = Path(".")
    seed: Mapping[str, str] | None = None

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(
            {HARNESS_COMPONENT: ComponentSpec(files=HarnessFiles(get_adapter("pi")), kind="harness_plugin")}
        )

    def default_start(self) -> DefaultStart:
        return DefaultStart(files=self.seed or {})

    def step_stages(self) -> PipelineSettings:
        """The stages of the recipe's trainer: one report is one job."""
        return job_sized_stages(PipelineSpec(ThresholdSamples, ThresholdJobData))

    def step_jobs(self) -> _FileJobs:
        return _FileJobs(self.label, self.artifact_dir)


def tree_spec(recipe: Recipe, spec: TrainerSpec) -> TrainerSpec:
    """``spec`` for a ``_TreeRecipe``: its trainer runs the recipe's jobs on its own thread, with the recipe's
    stages and mode. Any other recipe keeps ``spec``."""
    if not isinstance(recipe, _TreeRecipe):
        return spec
    return replace(
        spec, stages=recipe.step_stages(), job_component=recipe.step_jobs(), training_mode=recipe.training_mode
    )


class _TreeTrainers(SpecTrainerBuilder):
    """Builds the trainers of ``recipe``, a ``_TreeRecipe`` or a composite with ``_TreeRecipe`` components, from
    each recipe's ``tree_spec``."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

    def build_trainers(self, specs, scenario, records, **arguments):
        if isinstance(specs, ComponentTrainerSpecs):
            recipes = self.recipe.components
            specs = replace(
                specs,
                specs={component: tree_spec(recipes[component], spec) for component, spec in specs.specs.items()},
            )
        else:
            specs = tree_spec(self.recipe, specs)
        return super().build_trainers(specs, scenario, records, **arguments)


def build_trainers(recipe: Recipe, scenario: str, records, **arguments):
    """The trainers that a dispatcher of ``_serve`` builds for ``recipe``."""
    return _TreeTrainers(recipe).build_trainers(
        read_trainer_specs(recipe, LiveObjects()), scenario, records, **arguments
    )


@dataclass(frozen=True)
class _ScoredTreeRecipe(_TreeRecipe):
    """The same tree recipe, consuming scored rollout reports."""

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport


@dataclass(frozen=True)
class _ConfigRecipe(Recipe):
    """A second component with one settings file and no trainer of its own."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(components={"config": ComponentSpec(kind="harness_plugin")})

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={CONFIG_FILE: json.dumps({"request_defaults": {"temperature": 0.2}})})


#: The settings file of the second component.
CONFIG_FILE = "config.json"

COMPOSITE = "reef.recipe.composite:CompositeRecipe"
TREE = "tests.reef_recipe.test_composite_recipe:_TreeRecipe"
CONFIG = "tests.reef_recipe.test_composite_recipe:_ConfigRecipe"


def composite_config(harness: Mapping[str, Any], *, name: str = "harness", **sections: Any) -> dict[str, Any]:
    """The config of a composite of ``harness`` (named ``name``) and the config recipe, with ``sections`` added."""
    return {
        "implementation": COMPOSITE,
        "model": {"path": "served-model"},
        **sections,
        "components": {name: harness, "config": {"implementation": CONFIG}},
    }


def _composite(tmp_path: Path) -> CompositeRecipe:
    return CompositeRecipe(
        components={
            "harness": _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
            "config": _ConfigRecipe(),
        }
    )


def engine_runtime() -> InferenceProxyRuntime:
    """A runtime whose engine has an HTTP address, so a loaded scenario of the composite has a served model."""
    return InferenceProxyRuntime(base_url="http://engine.test", model_path="m")


def _root_of(recipe: CompositeRecipe, directory: Path) -> ScenarioRoot:
    """The root snapshot of a scenario created from ``recipe``'s default start, composed under ``directory``."""
    start = default_start(recipe, LiveObjects())
    (directory / "start").mkdir(parents=True)
    start.write(directory / "start")
    components = {name: Artifact.local(directory / "start" / name) for name in recipe.components}
    return ScenarioRoot(
        compose_snapshot(components, directory=directory / "root"), RootSource("default", inputs=start.inputs)
    )


def _serve(
    recipe: CompositeRecipe,
    tmp_path: Path,
    *,
    hold_local_cycles: bool = False,
    runtime: InferenceProxyRuntime | None = None,
) -> Dispatcher:
    storage = data_storage(tmp_path / "records")
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        live=LiveObjects(runtime=runtime, data_store=storage.data),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=storage,
        record_storage=storage.inner,
        coordination=memory_coordination(),
        hold_local_cycles=hold_local_cycles,
        trainer_builder=_TreeTrainers(recipe),
    )


@pytest.mark.unit
def test_composite_recipe_merges_components_seeds_and_trainers(tmp_path: Path) -> None:
    recipe = _composite(tmp_path)
    components = declared_components(recipe, "agent")
    assert components.names == ("harness", "config")
    assert components.files_component == "harness"
    base = default_start(recipe, LiveObjects())
    assert base.files == {
        "harness/AGENTS.md": "seed",
        f"config/{CONFIG_FILE}": json.dumps({"request_defaults": {"temperature": 0.2}}),
    }
    assert scenario_report_type(recipe) is None
    trainers = build_trainers(
        recipe,
        "agent",
        SQLiteScenarioStorage().open("agent").records,
        components=components,
        algorithm_states={},
        root=_root_of(recipe, tmp_path / "root"),
    )
    assert [component_trainer.component for component_trainer in trainers] == ["harness", "config"]
    assert trainers[0].trainer.trains
    assert not trainers[1].trainer.trains


@pytest.mark.unit
def test_each_component_trainer_takes_only_its_own_record2sample_keys(tmp_path: Path) -> None:
    trajectories = {"record2sample": {"module": "reef.record2sample.trajectories:TrajectorySamples"}}
    stages = PipelineSettings.from_setting(trajectories, spec=PipelineSpec(), environ={})
    harness = _composite(tmp_path).components["harness"]
    recipe = CompositeRecipe(components={"harness": harness, "config": _ConfigRecipe(pipeline_stages=stages)})
    components = declared_components(recipe, "agent")
    configured = with_record2sample_configs(recipe, components, {"config": {"trajectory_idle_seconds": 7}})
    trainers = build_trainers(
        configured,
        "agent",
        SQLiteScenarioStorage().open("agent").records,
        components=components,
        algorithm_states={},
        root=_root_of(recipe, tmp_path / "root"),
    )
    assert trainers[1].trainer.pipeline.record2sample.trajectory_idle_seconds == 7
    # The other component's stage is its recipe's own.
    assert not isinstance(trainers[0].trainer.pipeline.record2sample, type(trainers[1].trainer.pipeline.record2sample))


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
    # A process has one harness training component, so the mapping holds at most one harness recipe.
    harness = HarnessEvolutionRecipe(evolution=EvolutionSettings(), evaluation=EvaluationSettings(policy="none"))
    with pytest.raises(RecipeConfigError, match=r"\['first', 'second'\] each evolve a harness"):
        CompositeRecipe(components={"first": harness, "second": harness})
    two_trees = CompositeRecipe(components={"a": _TreeRecipe(label="a"), "b": _TreeRecipe(label="b")})
    with pytest.raises(ValueError, match="at most one component exposes a client-pulled file tree"):
        declared_components(two_trees, "agent")


class _RootFiles(StubTrainingComponent):
    """A training component that starts only from a root that holds the file ``required``, as Tinker's needs its
    manifest; ``checked`` lists the file names of each root that it was asked about."""

    def __init__(self, name: str, required: str) -> None:
        super().__init__(name=name)
        self.required = required
        self.checked: list[list[str]] = []

    def root_start(self, root: ScenarioRoot) -> str:
        names = sorted(path.name for path in root.local_directory().iterdir())
        self.checked.append(names)
        if self.required not in names:
            raise ValueError(f"artifact is missing {self.required}")
        return self.base_start


@pytest.mark.unit
def test_each_component_of_an_item_root_is_checked_by_its_own_backend(tmp_path: Path) -> None:
    from recipes.sao.recipe import SAORecipe

    recipe = CompositeRecipe(
        components={
            "weights": SAORecipe(),
            "harness": HarnessEvolutionRecipe(
                evolution=EvolutionSettings(), evaluation=EvaluationSettings(policy="none")
            ),
        }
    )

    def root(name: str, weights_file: str, harness_file: str) -> ScenarioRoot:
        for component, file_name in (("weights", weights_file), ("harness", harness_file)):
            (tmp_path / name / component).mkdir(parents=True)
            (tmp_path / name / component / file_name).write_text("x", encoding="utf-8")
        components = {component: Artifact.local(tmp_path / name / component) for component in recipe.components}
        return ScenarioRoot(
            compose_snapshot(components, directory=tmp_path / name / "root"),
            RootSource("item", item="item-1", uri="file:///item", version=""),
        )

    weights, harness = _RootFiles("stub", "manifest.json"), _RootFiles("harness", "rules.md")
    live = LiveObjects(
        runtime=engine_runtime(),
        training_component=weights,
        training_facts=StubFacts(weights),
        harness_component=harness,
        harness_facts=StubFacts(harness),
    )
    # The facts of the weight backend check the weights alone: the harness tree holds no manifest of theirs.
    check_root_start(recipe, live, root("good", "manifest.json", "rules.md"))
    assert weights.checked == [["manifest.json"]]
    assert harness.checked == [["rules.md"]]
    with pytest.raises(ValueError, match=r"artifact is missing manifest\.json"):
        check_root_start(recipe, live, root("no-weights", "notes.txt", "rules.md"))
    with pytest.raises(ValueError, match=r"artifact is missing rules\.md"):
        check_root_start(recipe, live, root("no-tree", "manifest.json", "notes.txt"))
    # A process that holds no facts of the harness checks the weights alone.
    weights_only = replace(live, harness_component=None, harness_facts=None)
    check_root_start(recipe, weights_only, root("control", "manifest.json", "notes.txt"))


@pytest.mark.unit
def test_composite_recipe_builds_from_config() -> None:
    # No runtime is injected: the deployment relies on REEF_UPSTREAM_URL, as a flat recipe may.
    environ = {"REEF_UPSTREAM_URL": "http://upstream.test"}
    recipe = build_recipe(COMPOSITE, environ, config=composite_config({"implementation": TREE}))
    assert isinstance(recipe, CompositeRecipe)
    assert sorted(recipe.components) == ["config", "harness"]
    assert isinstance(recipe.components["harness"], _TreeRecipe)
    # The components share the one runtime of the deployment, resolved from the environment.
    assert resolve_proxy_runtime(environ, None) is not None
    with pytest.raises(RecipeConfigError, match=r"components\.step is a metric prefix .*rename the component"):
        build_recipe(COMPOSITE, {}, config=composite_config({"implementation": TREE}, name="step"))
    with pytest.raises(RecipeConfigError, match="components"):
        build_recipe(COMPOSITE, {}, config={"implementation": "x", "model": {}})
    with pytest.raises(RecipeConfigError, match="different training modes"):
        build_recipe(
            COMPOSITE, {}, config=composite_config({"implementation": TREE, "config": {"training_mode": "manual"}})
        )


_WEIGHTS_AND_HARNESS = {
    "weights": {"implementation": "recipes.sao.recipe:SAORecipe"},
    "harness": {"implementation": TREE},
}
_LOCAL_TRAINING = "tests.support.training_deployment:LocalDeployment"


@pytest.mark.unit
def test_composite_recipe_names_its_weight_training_component() -> None:
    from recipes.sao.recipe import SAORecipe

    selected = weight_training_selection(CompositeRecipe, {"components": _WEIGHTS_AND_HARNESS})
    assert selected is not None
    # The selected component's config comes back flat, as its recipe reads it.
    assert selected == (SAORecipe, {"implementation": "recipes.sao.recipe:SAORecipe", "data": {}})
    assert (
        weight_training_selection(CompositeRecipe, {"components": {"harness": _WEIGHTS_AND_HARNESS["harness"]}})
        is None
    )
    assert weight_training_selection(_TreeRecipe, {}) is None
    assert weight_training_selection(SAORecipe, {"data": {}}) == (SAORecipe, {"data": {}})
    # A snapshot loads one component, so two weight components are refused before any runtime is connected.
    with pytest.raises(RecipeConfigError, match=r"\['weights', 'weights2'\] all train weights"):
        weight_training_selection(
            CompositeRecipe, {"components": {**_WEIGHTS_AND_HARNESS, "weights2": _WEIGHTS_AND_HARNESS["weights"]}}
        )


@pytest.mark.unit
def test_service_connects_the_training_component_of_a_composite_weight_component(tmp_path) -> None:
    from tests.support.training_deployment import LocalRuntime

    from reef.service.assembly import _load_deployment
    from reef.service.deploy.service_config import ServiceSettings, service_config_from_mapping

    def composite_service(activation: Mapping[str, Any] | None = None, **reef: Any) -> ServiceSettings:
        settings = {
            "reef": {
                "recipe": COMPOSITE,
                "model_path": "demo-model",
                "training_backend": _LOCAL_TRAINING,
                "components": _WEIGHTS_AND_HARNESS,
                **reef,
            }
        }
        return service_config_from_mapping(settings if activation is None else {**settings, "activation": activation})

    recipe, live = _load_deployment(COMPOSITE, composite_service(), {}, runs_training=True)
    assert isinstance(recipe, CompositeRecipe)
    assert isinstance(live.training_component, LocalRuntime)
    assert live.training_component.received_model_path == "demo-model"

    # The components share one job component; a replacement reaches every component that submits its jobs.
    shared = live.training_component
    assert live.job_components() == (shared,)
    queued = StubTrainingComponent(name=shared.name)
    moved = replace(live, training_component=queued, training_facts=StubFacts(queued))
    assert moved.job_components() == (queued,)

    # A flat weight recipe setting has no place at the top of a composite.
    with pytest.raises(ValueError, match=r"reef\.training_mode is not a setting of CompositeRecipe"):
        _load_deployment(COMPOSITE, composite_service(training_mode="manual"), {}, runs_training=True)

    # A top level activation section is the composite's own: one rule activates the artifacts of every component.
    never, _ = _load_deployment(COMPOSITE, composite_service({"module": "never"}), {}, runs_training=True)
    assert isinstance(never.activation_rule(), NeverActivate)
    with pytest.raises(RecipeConfigError, match=r"cannot import activation 'no\.such\.module:Rule'"):
        _load_deployment(COMPOSITE, composite_service({"module": "no.such.module:Rule"}), {}, runs_training=True)


@pytest.mark.unit
def test_deployment_treats_a_composite_with_a_weight_component_as_training(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy.deployment_config import selected_weight_training
    from reef.service.deploy.execution import validate_services
    from reef.service.deploy.orchestrator import resolve_deployment_config

    raw = {
        "recipe": {
            "implementation": COMPOSITE,
            "config": {"components": _WEIGHTS_AND_HARNESS},
        },
        "inference": {"model-path": "demo-model"},
        "training": {"backend": _LOCAL_TRAINING},
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend"] == _LOCAL_TRAINING
    assert [process["name"] for process in validate_services(config, "test")] == ["reef"]
    assert config["reef"]["recipe"] == COMPOSITE
    from recipes.sao.recipe import SAORecipe

    # The service builds the training component for the weight component's class and loss family.
    assert selected_weight_training(CompositeRecipe, config) is SAORecipe

    # A component named through the environment is classified once the environment is applied.
    monkeypatch.setenv("PROBE_WEIGHTS", "recipes.sao.recipe:SAORecipe")
    interpolated = {
        **raw,
        "recipe": {
            "implementation": COMPOSITE,
            "config": {"components": {**_WEIGHTS_AND_HARNESS, "weights": {"implementation": "${PROBE_WEIGHTS}"}}},
        },
    }
    config, _ = resolve_deployment_config(interpolated, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend"] == _LOCAL_TRAINING

    assert selected_weight_training(CompositeRecipe, config) is SAORecipe

    # An unimportable component is refused the way an unimportable recipe is.
    broken = {
        "reef": {
            "recipe": COMPOSITE,
            "components": {"weights": {"implementation": "no.such.module:Recipe"}},
        }
    }
    with pytest.raises(RecipeConfigError, match=r"cannot import recipe reference 'no\.such\.module:Recipe'"):
        selected_weight_training(CompositeRecipe, broken)


@pytest.mark.unit
def test_composite_recipe_reads_components_from_the_file_layout() -> None:
    harness = {
        "implementation": TREE,
        "config": {"training-mode": "auto"},
    }

    def file_layout(harness_section: Any) -> dict[str, Any]:
        return {
            "recipe": {
                "implementation": COMPOSITE,
                "config": {
                    "components": {
                        "harness": harness_section,
                        "config": {"implementation": CONFIG},
                    }
                },
            },
            "inference": {"upstream-model": "served-model"},
        }

    settings = recipe_config_from_file_layout(file_layout(harness))
    assert sorted(settings["components"]) == ["config", "harness"]
    recipe = build_recipe(settings["implementation"], {}, config=settings)
    assert isinstance(recipe, CompositeRecipe)
    assert recipe.components["harness"].training_mode == "auto"

    # The flat recipe config a component used to take is refused with the place its fields now go.
    flat = recipe_config_from_file_layout(
        file_layout({"implementation": harness["implementation"], "data": {"training_mode": "auto"}})
    )
    with pytest.raises(
        RecipeConfigError, match=r"components\.harness\.data is not accepted: .*components\.harness\.config"
    ):
        build_recipe(flat["implementation"], {}, config=flat)
    # Every component shares the deployment's runtime.
    own_runtime = recipe_config_from_file_layout(file_layout({**harness, "runtime": {"type": "upstream"}}))
    with pytest.raises(RecipeConfigError, match=r"components\.harness\.runtime is not accepted"):
        build_recipe(own_runtime["implementation"], {}, config=own_runtime)


@pytest.mark.unit
def test_composite_scenario_serves_its_config_and_reports_components(tmp_path: Path) -> None:
    with closing(_serve(_composite(tmp_path), tmp_path, hold_local_cycles=True)) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        artifact = Artifact(scenario.current_snapshot(), scenario.repository)
        local = artifact.component("config").materialize().local_path
        assert json.loads((local / CONFIG_FILE).read_text()) == {"request_defaults": {"temperature": 0.2}}

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
        make_samples(scenario)
        result = scenario.next_commit("harness")
        assert result is not None
        scenario.commit(result, component="harness")

        rows = scenario.snapshot_queries.snapshot_rows(scenario.commit_number)
        assert rows[0]["component"] == "harness"
        assert "component" not in rows[1]
        # The config component runs no step: the one commit is the harness's job submission, naming the rows it
        # consumed and the harness's sample of them.
        assert scenario.store.history()[-1].consumed_ids == {
            "i1",
            "r1",
            consumed_sample_id(stage_item_id("agent", "harness:r1:0")),
        }
        # The job ended in submit: its checkpoint and its end follow, and consume nothing.
        checkpoint = None
        while scenario.trainer_for("harness").job_running:
            polled = scenario.next_commit("harness")
            checkpoint = polled if polled.artifact is not None else checkpoint
            scenario.commit(polled, component="harness")
        assert checkpoint is not None
        assert all(record.consumed_ids == frozenset() for record in scenario.store.history()[1:])
        status = dispatcher.build_training_status()["scenarios"]["agent"]
        assert status["components"]["harness"]["last_training_commit"]["commit_number"] == 3
        assert status["components"]["config"]["last_training_commit"] is None
        manifest = RequestService(dispatcher).harness_manifest({"x-reef-scenario": "agent"}, form="source")
        assert manifest["files"] == {"harness.txt": "harness step 1"}
        assert set(manifest["components"]) == {"harness", "config"}
        assert manifest["components"]["harness"] == scenario.artifact_status()["harness"]["active"]


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
    report_type = scenario_report_type(recipe)
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
    assert scenario_report_type(same) is ScoredRolloutReport
    # Every trainer of a composite is told what its scenario's ingress admits.
    mixed = CompositeRecipe(
        components={
            "harness": _ScoredTreeRecipe(label="harness", artifact_dir=tmp_path / "h", seed={"AGENTS.md": "seed"}),
            "config": _GridConfigRecipe(),
        }
    )
    mixed_report_type = scenario_report_type(mixed)
    assert mixed_report_type is not None and mixed_report_type.__name__.startswith("AnyOf")
    trainers = build_trainers(
        mixed,
        "agent",
        SQLiteScenarioStorage().open("agent").records,
        components=declared_components(mixed, "agent"),
        algorithm_states={},
        root=_root_of(mixed, tmp_path / "root"),
    )
    assert [
        component_trainer.trainer.pipeline.context.admitted_report_type is mixed_report_type
        for component_trainer in trainers
    ] == [
        True,
        True,
    ]
    assert trainers[0].trainer.pipeline.context.report_type is ScoredRolloutReport


@pytest.mark.unit
def test_composite_scenario_enforces_the_agreed_report_type_whatever_the_component_order(tmp_path: Path) -> None:
    # The component with no report contract is listed first; the harness contract still guards ingress.
    recipe = CompositeRecipe(
        components={
            "config": _ConfigRecipe(),
            "harness": _ScoredTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
        }
    )
    assert scenario_report_type(recipe) is ScoredRolloutReport
    with closing(_serve(recipe, tmp_path)) as dispatcher:
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
    with closing(_serve(_composite(tmp_path), tmp_path, runtime=engine_runtime())) as dispatcher:
        headers = {"x-reef-scenario": "agent"}
        # A rollout runs for a loaded scenario's step, so its calls find the scenario loaded.
        assert dispatcher.get_or_create_scenario("agent") is not None
        wait_for_serving(dispatcher, "agent")

        async def run() -> None:
            async with TestClient(TestServer(create_app(dispatcher, inference_handler=_EchoHandler()))) as client:
                evaluation = await client.post(
                    "/reef/scenarios/agent/evaluation/v1/chat/completions", json={"messages": []}
                )
                assert evaluation.status == 200, await evaluation.text()
                assert "x-reef-agent-record-id" not in evaluation.headers
                assert (await evaluation.json())["choices"][0]["message"]["content"] == "ok"
                served = await client.post("/v1/chat/completions", headers=headers, json={"messages": []})
                assert served.status == 200, await served.text()
                assert "x-reef-agent-record-id" in served.headers

        asyncio.run(run())
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.records.count("agent") == 1


@pytest.mark.unit
def test_a_flat_scenario_without_a_durable_log_still_serves_its_head(tmp_path: Path) -> None:
    """The harness head of a flat scenario is the served artifact, found on the chain when the log keeps no rows."""
    tree = tmp_path / "seed"
    tree.mkdir()
    (tree / "rules.md").write_text("seed")
    recipe = _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"rules.md": "seed"})
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(None),
        coordination=memory_coordination(),
        trainer_builder=_TreeTrainers(recipe),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and not scenario.store.durable
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "rules.md").write_text("evolved")
        scenario.commit(TrainingCommit(state={}, artifact=Artifact.local(evolved)))
        head = scenario.current_snapshot().revision
        source = service.harness_manifest(headers, form="source")
        assert source["files"] == {"rules.md": "evolved"}
        artifact_id = source["artifact_id"]
        assert artifact_id != head
        assert scenario.artifact_selection().components["harness"] == artifact_id
        assert any(row["artifact_id"] == artifact_id and row["active"] for row in scenario.artifacts())
        assert service.harness_manifest(headers, artifact_id=artifact_id, form="source")["files"] == source["files"]
        # The default form renders the source tree for the adapter.
        assert service.harness_manifest(headers)["files"]["plugin/rules.md"].startswith("evolved")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_evaluation_route_names_the_scenario_in_its_path(tmp_path: Path) -> None:
    with closing(_serve(_composite(tmp_path), tmp_path, runtime=engine_runtime())) as dispatcher:
        assert dispatcher.get_or_create_scenario("agent") is not None
        wait_for_serving(dispatcher, "agent")

        async def run() -> None:
            async with TestClient(TestServer(create_app(dispatcher, inference_handler=_EchoHandler()))) as client:
                response = await client.post(
                    "/reef/scenarios/agent/evaluation/v1/chat/completions", json={"messages": []}
                )
                assert response.status == 200, await response.text()
                assert "x-reef-agent-record-id" not in response.headers
                assert (await response.json())["choices"][0]["message"]["content"] == "ok"
                missing = await client.post("/reef/scenarios/agent/evaluation/v1/nothing", json={})
                assert missing.status == 404
                # An evaluation call never creates a scenario: only a loaded one has rollouts.
                unknown = await client.post(
                    "/reef/scenarios/other/evaluation/v1/chat/completions", json={"messages": []}
                )
                assert unknown.status == 404

        asyncio.run(run())
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.records.count("agent") == 0
        assert dispatcher.get_or_create_scenario("other", allow_implicit_creation=False) is None


class _ClosingJobData(ThresholdJobData):
    """A sample-to-job stage whose close waits for the rollout call in flight, as a worker pool's close waits for its
    batch: the trainer closes its pipeline when the scenario is deleted."""

    def __init__(self, context: PipelineContext, closing: threading.Event, rollout_done: threading.Event) -> None:
        super().__init__(context, None, 1)
        self.closing = closing
        self.rollout_done = rollout_done

    def close(self) -> None:
        self.closing.set()
        self.rollout_done.wait(10)
        super().close()


@dataclass(frozen=True)
class _RolloutTreeRecipe(_TreeRecipe):
    """The tree recipe whose pipeline closes only once its evaluation rollout has made its last call."""

    closing: threading.Event = field(default_factory=threading.Event)
    rollout_done: threading.Event = field(default_factory=threading.Event)

    def step_stages(self) -> PipelineSettings:
        closing, rollout_done = self.closing, self.rollout_done

        class JobData(_ClosingJobData):
            @classmethod
            def from_config(cls, config, *, context, record2sample, environ):
                return cls(context, closing, rollout_done)

        return PipelineSettings(
            record2sample=PluginReference(ThresholdSamples, {}), sample2job=PluginReference(JobData, {})
        )


@pytest.mark.unit
def test_an_evaluation_call_during_a_delete_neither_holds_it_up_nor_brings_the_scenario_back(tmp_path: Path) -> None:
    """A delete closes the scenario's pipeline, which waits for the evaluation rollout in flight; the rollout's call
    must not wait for the scenario the delete holds, and must not create the scenario again once the delete ends."""
    harness = _RolloutTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"})
    runtime = InferenceProxyRuntime(base_url="http://engine.test", model_path="m")
    dispatcher = _serve(
        CompositeRecipe(components={"harness": harness, "config": _ConfigRecipe()}), tmp_path, runtime=runtime
    )
    path = "/reef/scenarios/agent/evaluation/v1/chat/completions"

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, inference_handler=_EchoHandler()))) as client:
            assert dispatcher.get_or_create_scenario("agent") is not None
            deleting = asyncio.create_task(asyncio.to_thread(dispatcher.delete_scenario, "agent"))
            assert await asyncio.to_thread(harness.closing.wait, 10)
            try:
                # The rollout's call answers while the delete still waits for it: the scenario is gone.
                during = await asyncio.wait_for(client.post(path, json={"messages": []}), timeout=5)
                assert during.status == 404, await during.text()
            finally:
                harness.rollout_done.set()
            deleted = await asyncio.wait_for(deleting, timeout=10)
            assert deleted["scenario"] == "agent"
            after = await client.post(path, json={"messages": []})
            assert after.status == 404, await after.text()

    try:
        asyncio.run(run())
        assert dispatcher.get_or_create_scenario("agent", allow_implicit_creation=False) is None
    finally:
        harness.rollout_done.set()
        dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("composite", [True, False])
def test_only_a_composite_evaluates_through_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, composite: bool
) -> None:
    """A composite's evaluation calls come back to the service and hold its local cycles until it answers. A recipe of
    one component keeps calling its runtime's endpoint directly: it gets no served endpoint and runs at once."""
    from reef.service import assembly
    from reef.service.deploy.service_config import ServiceSettings

    flat = _TreeRecipe(label="harness")
    recipe = CompositeRecipe(components={"harness": flat, "tools": _TreeRecipe(label="tools")}) if composite else flat
    built: dict[str, Any] = {}

    def dispatcher(recipe: Recipe, *args: Any, **kwargs: Any) -> SimpleNamespace:
        built.update(live=kwargs["live"], hold=kwargs["hold_local_cycles"])
        return SimpleNamespace(services=ServiceSet.of(kwargs["components"]))

    monkeypatch.setattr(assembly, "_load_deployment", lambda *args, **kwargs: (recipe, LiveObjects()))
    monkeypatch.setattr(assembly, "Dispatcher", dispatcher)
    monkeypatch.setattr(GitLFSRepositoryBackendFactory, "from_config", lambda *args, **kwargs: lambda name: object())
    settings = ServiceSettings(recipe="recipe", agent_record_dir=str(tmp_path), port=8900)
    assembly.build_dispatcher(
        settings, hold_local_cycles=True, evaluation_token="rollout", components=frozenset({"control"})
    )
    if composite:
        assert built["hold"] is True
        assert built["live"].served_endpoint == ServedEndpoint("http://127.0.0.1:8900", token="rollout")
    else:
        assert built["hold"] is False and built["live"].served_endpoint is None


@pytest.mark.unit
def test_a_composite_names_each_component_in_its_served_endpoint(tmp_path: Path) -> None:
    harness, _ = harness_recipe(tmp_path, lambda workspace, services: None)
    live = LiveObjects(runtime=engine_runtime(), served_endpoint=ServedEndpoint("http://127.0.0.1:8900", token="t"))
    # A composite holds one harness recipe, so each name is the harness of a composite of its own.
    served = {}
    for name in ("harness", "tools"):
        composite = CompositeRecipe(components={name: harness, "config": _ConfigRecipe()})
        served[name] = harness_served_model(composite, live, "agent", name)
        assert harness_served_model(composite, live, "agent", "config") is None
    # The evaluation calls of each component go to the service's route that names the component.
    assert {name: (model.base_url, model.api_key) for name, model in served.items()} == {
        "harness": ("http://127.0.0.1:8900/reef/scenarios/agent/components/harness/evaluation", "t"),
        "tools": ("http://127.0.0.1:8900/reef/scenarios/agent/components/tools/evaluation", "t"),
    }


class _RefuseAll(ArtifactValidator):
    def validate(self, artifact: Artifact) -> None:
        raise ValueError("validator refused the artifact")


@dataclass(frozen=True)
class _CheckedRecordOnlyRecipe(Recipe):
    """A record only recipe (no component) that checks its whole snapshot."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(validator=_RefuseAll())


@pytest.mark.unit
def test_a_recipe_serving_no_component_admits_the_whole_snapshot_through_its_validator(tmp_path: Path) -> None:
    """With no component, the check admits the snapshot as a whole: the scenario builds, and a snapshot the check
    refuses is not published."""
    dispatcher = _serve(_CheckedRecordOnlyRecipe(), tmp_path)  # type: ignore[arg-type]
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.components.names == ()
        head = scenario.current_snapshot()
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "notes.txt").write_text("unvetted")
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.components.validate(Artifact.local(evolved))
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainingCommit(state={}, artifact=Artifact.local(evolved)))
        assert scenario.current_snapshot() == head
    finally:
        dispatcher.close()


class _RefuseUnvetted(ArtifactValidator):
    """Refuse a snapshot any file of which says unvetted."""

    def validate(self, artifact: Artifact) -> None:
        root = artifact.materialize().local_path
        files = [] if root is None else [path for path in Path(root).rglob("*") if path.is_file()]
        if any("unvetted" in path.read_text(encoding="utf-8", errors="ignore") for path in files):
            raise ValueError("validator refused the artifact")


@dataclass(frozen=True)
class _SnapshotCheckedTreeRecipe(_TreeRecipe):
    """A recipe that declares no component, publishes a tree, and checks its whole snapshot."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(validator=_RefuseUnvetted())


def _tree(tmp_path: Path, name: str, text: str) -> Artifact:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "notes.txt").write_text(text, encoding="utf-8")
    return Artifact.local(directory)


@pytest.mark.unit
def test_inside_a_composite_a_recipe_serving_no_component_admits_its_component_through_its_validator(
    tmp_path: Path,
) -> None:
    """Flat, such a recipe's snapshot is the whole artifact; inside a composite its snapshot is its component, so its
    check admits that component's steps as it admitted its whole snapshot flat."""
    composite = CompositeRecipe(
        components={
            "notes": _SnapshotCheckedTreeRecipe(label="notes", artifact_dir=tmp_path / "steps"),
            "config": _ConfigRecipe(),
        }
    )
    with pytest.raises(ValueError, match="validator refused the artifact"):
        declared_components(composite, "agent").components["notes"].validator.validate(
            _tree(tmp_path, "u", "unvetted")
        )
    with closing(_serve(composite, tmp_path / "svc")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        seed = scenario.current_snapshot()
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "c", "unvetted")), component="notes")
        assert scenario.current_snapshot() == seed
        scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "v", "vetted")), component="notes")
        assert scenario.current_snapshot() != seed


def check_whole_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every scenario that the factory builds the snapshot's own check ``_RefuseUnvetted``, beside each
    component's. A composite declares no such check; a recipe that declares several components itself can."""

    def checked_components(recipe: Recipe, live: LiveObjects, scenario: str) -> ScenarioComponents:
        return replace(scenario_components(recipe, live, scenario), validator=_RefuseUnvetted())

    monkeypatch.setattr("reef.scenario.factory.scenario_components", checked_components)


@pytest.mark.unit
def test_a_commit_runs_the_snapshot_check_an_activation_of_that_snapshot_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot's own check runs on the composed snapshot a step publishes, as an activation of it runs it: a
    snapshot the commit admits can be activated again, and one the activation would refuse is never published."""
    check_whole_snapshots(monkeypatch)
    composite = CompositeRecipe(
        components={
            "harness": _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
            "config": _ConfigRecipe(),
        }
    )
    with closing(_serve(composite, tmp_path / "svc")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        seed = scenario.current_snapshot()
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "c", "unvetted")), component="harness")
        assert scenario.current_snapshot() == seed
        scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "v", "vetted")), component="harness")
        published = scenario.current_snapshot()
        assert published != seed
        # The root snapshot made the creation's artifacts, under its own revision.
        seed_harness = scenario.repository.materialize(seed).components.artifact_ids(seed.revision)["harness"]
        assert (
            seed_harness
            == scenario.repository.resolve(scenario.repository.start_artifact).component("harness").artifact_id
        )
        scenario.activate(seed_harness, "harness")
        scenario.activate(scenario.repository.resolve(published).component("harness").artifact_id, "harness")
        served = scenario.repository.materialize(scenario.current_snapshot()).components
        made = scenario.repository.materialize(published).components
        assert served is not None and made is not None
        assert served.artifact_ids("served") == made.artifact_ids("made")


class _CountedChecks(_RefuseUnvetted):
    """Refuse what the parent refuses, and count the trees it judged."""

    def __init__(self) -> None:
        self.count = 0

    def validate(self, artifact: Artifact) -> None:
        self.count += 1
        super().validate(artifact)


@dataclass(frozen=True)
class _CountedTreeRecipe(_SnapshotCheckedTreeRecipe):
    """The tree recipe whose snapshot check counts what it judged."""

    checks: _CountedChecks | None = None

    def build_components(self, scenario: str) -> ScenarioComponents:
        assert self.checks is not None
        return ScenarioComponents(validator=self.checks)


@dataclass(frozen=True)
class _UnseededConfigRecipe(_ConfigRecipe):
    """A configuration recipe that seeds nothing: its component starts empty, which its own check would refuse."""

    def default_start(self) -> DefaultStart:
        return DefaultStart()


@pytest.mark.unit
def test_an_activation_never_judges_a_component_again(tmp_path: Path) -> None:
    """A component whose recipe seeds nothing starts empty, and the other component's steps carry it as it is. An
    activation of one of those steps, or of a held one, runs only the snapshot's own check: each component was
    admitted when its step published it, and judging the carried one would refuse snapshots their commits admitted."""
    checks = _CountedChecks()
    composite = CompositeRecipe(
        components={
            "harness": _CountedTreeRecipe(
                label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}, checks=checks
            ),
            "config": _UnseededConfigRecipe(),
        },
        # A step that changes the "held" kind waits for a person to activate it.
        activation=PluginReference(AlwaysActivate, {"review_kinds": ["held"]}),
    )
    with closing(_serve(composite, tmp_path / "svc")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "a", "one")), component="harness")
        first = scenario.current_snapshot()
        scenario.commit(TrainingCommit(state={}, artifact=_tree(tmp_path, "b", "two")), component="harness")
        checks.count = 0
        dispatcher.activate("agent", scenario.repository.resolve(first).component("harness").artifact_id, "harness")
        served = scenario.repository.materialize(scenario.current_snapshot()).components
        assert served is not None
        current = scenario.current_snapshot()
        assert served.artifact_ids(current.revision) == scenario.store.history()[0].components
        assert checks.count == 0
        head = scenario.current_snapshot()
        scenario.commit(
            TrainingCommit(state={}, artifact=_tree(tmp_path, "c", "three"), changed_kinds=frozenset({"held"})),
            component="harness",
        )
        assert scenario.current_snapshot() == head
        held = scenario.repository.resolve(scenario.store.history()[-1].snapshot).component("harness").artifact_id
        assert dispatcher.activate("agent", held, "harness").artifact_id == held
        assert checks.count == 1


@pytest.mark.unit
@pytest.mark.parametrize("component_config", [[["training_mode", "manual"]], "manual", [], "", None])
def test_a_component_config_that_is_not_an_object_is_refused_under_a_composite_mode(component_config: Any) -> None:
    harness = {"implementation": TREE, "config": component_config}
    with pytest.raises(RecipeConfigError, match=r"components\.harness\.config must be an object"):
        build_recipe(COMPOSITE, {}, config=composite_config(harness, data={"training_mode": "manual"}))


@pytest.mark.unit
def test_composed_snapshot_links_carried_files_and_tolerates_an_empty_component(tmp_path: Path) -> None:
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter.bin").write_bytes(b"\x00" * 16)
    harness = Artifact.local(tmp_path / "missing")
    composed = compose_snapshot({"weights": Artifact.local(weights), "harness": harness}, directory=tmp_path / "r")
    linked = tmp_path / "r" / "weights" / "adapter.bin"
    assert linked.read_bytes() == b"\x00" * 16
    assert linked.stat().st_nlink == 2
    assert (tmp_path / "r" / "harness").is_dir()
    assert TextFileTree().read_files(composed.component("harness")) is None


@pytest.mark.unit
def test_status_and_contract_speak_of_the_trainer_that_steps(tmp_path: Path) -> None:
    """A component listed first that runs no step does not lend its mode and processor to the whole scenario: the
    config component builds no batch, so the harness alone decides whether training requests are taken."""
    recipe = CompositeRecipe(
        components={
            "config": _ConfigRecipe(),
            "harness": _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
        }
    )
    with closing(_serve(recipe, tmp_path)) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert dispatcher.set_training_mode("agent", "manual") == {"scenario": "agent", "training_mode": "manual"}
        assert scenario.training_mode == "manual"
        assert scenario.trainer_for("config").pipeline.training_mode == "auto"
        harness = scenario.trainer_for("harness")
        assert harness.pipeline.training_mode == "manual"
        assert scenario.trainer is harness
        block = dispatcher.build_training_status()["scenarios"]["agent"]
        assert block["training_mode"] == "manual"
        assert block["processor"] == {
            **harness.pipeline_status(),
            "pending_requests": scenario.pending_requests(),
        }
        contract = dispatcher.scenario_contract("agent")
        assert contract["training_mode"] == "manual"
        assert contract["processor"] == type(harness.pipeline.sample2job).__name__


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "harness_config"),
    [
        ({"training_mode": "manual"}, None),
        # ``data.training-mode`` is the spelling every flat recipe accepts; a component naming the same mode in it
        # is not refused for naming the field twice.
        ({"training-mode": "manual"}, {"training-mode": "manual"}),
    ],
)
def test_a_composite_mode_is_every_components_mode_and_a_stepless_component_runs_auto(
    tmp_path: Path, data: dict[str, str], harness_config: dict[str, str] | None
) -> None:
    """The documented composite takes training requests from its config: the mode reaches every component, and the
    config component, which runs no step, keeps its bare processor in auto."""
    harness: dict[str, Any] = {"implementation": TREE}
    if harness_config is not None:
        harness["config"] = harness_config
    recipe = build_recipe(COMPOSITE, {}, config=composite_config(harness, data=data))
    assert isinstance(recipe, CompositeRecipe)
    assert recipe.training_mode == "manual"
    assert [component.training_mode for component in recipe.components.values()] == ["manual", "manual"]
    with closing(_serve(recipe, tmp_path)) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert scenario.training_mode == "manual"
        assert scenario.trainer_for("harness").pipeline.training_mode == "manual"
        assert scenario.trainer_for("config").pipeline.training_mode == "auto"
