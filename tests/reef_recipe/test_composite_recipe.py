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
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.runtime_stubs import StubTrainingComponent
from tests.support.samples_service import make_samples

from reef.artifact import Artifact, BaseArtifactSource, GitLFSRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.artifact.artifact import ArtifactValidator
from reef.artifact.composite import compose_release
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginReference
from reef.core.reports import ReportValidationError, ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate, NeverActivate
from reef.execution.workers import ServiceSet
from reef.harness.adapters import get_adapter
from reef.harness.sync import create_harness_surface
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe import CompositeRecipe, Recipe, RecipeConfigError, build_recipe
from reef.recipe.base import ServedEndpoint
from reef.recipe.config import recipe_config_from_file_layout
from reef.runtime.interfaces import InferenceHandler
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.storage.samples import consumed_sample_id
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.train import Trainer, TrainStepResult
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline


class _FileJobs(StubTrainingComponent):
    """A training component whose ``n``-th job saves one text file, ``<label>.txt`` = ``<label> step <n>``, for
    whatever component this recipe evolves."""

    def __init__(self, label: str, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name=label, shares_scenarios=True, release_format=None)
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

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface(get_adapter("pi"))

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource(files=self.seed or {})

    def step_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        return threshold_pipeline(context)

    def step_jobs(self) -> _FileJobs:
        return _FileJobs(self.label, self.artifact_dir)

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=self.step_pipeline,
            job_loop=JobLoop(self.step_jobs(), scenario=scenario),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            default_job_samples=1,
            samples=samples,
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

    def build_surface(self, scenario: str) -> Surface:
        return Surface(components={"config": ComponentSurface(kind="harness_plugin")})

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource(files={CONFIG_FILE: json.dumps({"request_defaults": {"temperature": 0.2}})})


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


def _served_composite(tmp_path: Path) -> CompositeRecipe:
    """The composite with a runtime whose engine has an HTTP address, so a loaded scenario has a served model."""
    return replace(_composite(tmp_path), runtime=InferenceProxyRuntime(base_url="http://engine.test", model_path="m"))


def _serve(recipe: CompositeRecipe, tmp_path: Path) -> Dispatcher:
    initial = tmp_path / "initial"
    for relative, text in recipe.base_artifact_source().files.items():
        (initial / relative).parent.mkdir(parents=True, exist_ok=True)
        (initial / relative).write_text(text, encoding="utf-8")
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )


@pytest.mark.unit
def test_composite_recipe_merges_surfaces_seeds_and_trainers(tmp_path: Path) -> None:
    recipe = _composite(tmp_path)
    surface = recipe.build_surface("agent")
    assert surface.names == ("harness", "config")
    assert surface.files_component == "harness"
    base = recipe.base_artifact_source()
    assert base.files == {
        "harness/AGENTS.md": "seed",
        f"config/{CONFIG_FILE}": json.dumps({"request_defaults": {"temperature": 0.2}}),
    }
    # No weight-training component: the base holds no model.
    assert base.model_files is None and base.model_component is None
    assert recipe.report_type is None
    trainers = recipe.build_trainers(
        "agent", SQLiteScenarioStorage().open("agent").records, surface=surface, algorithm_states={}
    )
    assert [component_trainer.component for component_trainer in trainers] == ["harness", "config"]
    assert trainers[0].trainer.trains
    assert not trainers[1].trainer.trains


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
    with pytest.raises(ValueError, match="at most one component exposes a client-pulled file tree"):
        two_trees.build_surface("agent")


@pytest.mark.unit
def test_composite_recipe_builds_from_config() -> None:
    # No runtime is injected: the deployment relies on REEF_UPSTREAM_URL, as a flat recipe may.
    recipe = build_recipe(
        COMPOSITE, {"REEF_UPSTREAM_URL": "http://upstream.test"}, config=composite_config({"implementation": TREE})
    )
    assert isinstance(recipe, CompositeRecipe)
    assert sorted(recipe.components) == ["config", "harness"]
    assert isinstance(recipe.components["harness"], _TreeRecipe)
    # The components share one runtime, resolved from the environment.
    assert recipe.runtime is not None
    assert all(component.runtime is recipe.runtime for component in recipe.components.values())
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
_LOCAL_TRAINING = "tests.support._training_deployment:LocalDeployment"


@pytest.mark.unit
def test_composite_recipe_names_its_weight_training_component() -> None:
    from recipes.sao.recipe import SAORecipe

    selected = CompositeRecipe.select_weight_training({"components": _WEIGHTS_AND_HARNESS})
    assert selected is not None
    # The selected component's config comes back flat, as its recipe reads it.
    assert selected == (SAORecipe, {"implementation": "recipes.sao.recipe:SAORecipe", "data": {}})
    assert CompositeRecipe.select_weight_training({"components": {"harness": _WEIGHTS_AND_HARNESS["harness"]}}) is None
    assert _TreeRecipe.select_weight_training({}) is None
    assert SAORecipe.select_weight_training({"data": {}}) == (SAORecipe, {"data": {}})
    # A release loads one component, so two weight components are refused before any runtime is connected.
    with pytest.raises(RecipeConfigError, match=r"\['weights', 'weights2'\] all train weights"):
        CompositeRecipe.select_weight_training(
            {"components": {**_WEIGHTS_AND_HARNESS, "weights2": _WEIGHTS_AND_HARNESS["weights"]}}
        )


@pytest.mark.unit
def test_service_connects_the_training_component_of_a_composite_weight_component() -> None:
    from tests.support._training_deployment import LocalRuntime

    from reef.service.assembly import _serving_recipe
    from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping

    def composite_service(activation: Mapping[str, Any] | None = None, **reef: Any) -> ServiceConfig:
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

    recipe = _serving_recipe(COMPOSITE, composite_service(), {})
    assert isinstance(recipe, CompositeRecipe)
    assert isinstance(recipe.training_component, LocalRuntime)
    assert recipe.training_component.received_model_path == "demo-model"
    assert all(component.training_component is recipe.training_component for component in recipe.components.values())
    assert all(component.runtime is recipe.runtime for component in recipe.components.values())

    # The components share one job component; a replacement reaches every component that submits its jobs.
    from tests.support.runtime_stubs import StubTrainingComponent

    shared = recipe.training_component
    assert recipe.job_components() == (shared,)
    queued = StubTrainingComponent(name=shared.name)
    moved = recipe.with_job_components({shared.name: queued})
    assert moved.job_components() == (queued,)
    assert all(
        component.training_component is (None if original.training_component is None else queued)
        for component, original in zip(moved.components.values(), recipe.components.values(), strict=True)
    )
    with pytest.raises(ValueError, match=r"the job components of CompositeRecipe are \['stub'\], not \[\]"):
        recipe.with_job_components({})

    # A flat weight recipe setting has no place at the top of a composite.
    with pytest.raises(ValueError, match=r"reef\.training_mode is not a setting of CompositeRecipe"):
        _serving_recipe(COMPOSITE, composite_service(training_mode="manual"), {})

    # A top level activation section is the composite's own: one rule activates the artifacts of every component.
    never = _serving_recipe(COMPOSITE, composite_service({"module": "never"}), {})
    assert isinstance(never.activation_rule(), NeverActivate)
    with pytest.raises(RecipeConfigError, match=r"cannot import activation 'no\.such\.module:Rule'"):
        _serving_recipe(COMPOSITE, composite_service({"module": "no.such.module:Rule"}), {})


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
    with closing(_serve(_composite(tmp_path), tmp_path)) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        artifact = Artifact(scenario.current_artifact_ref(), scenario.repository)
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
        result = scenario.prepare_training_step("harness")
        assert result is not None
        scenario.commit(result, component="harness")

        rows = scenario.releases()
        assert rows[0]["component"] == "harness"
        assert "component" not in rows[1]
        # The config component runs no step: the one commit is the harness's job submission, naming the rows it
        # consumed and the harness's sample of them.
        assert scenario.store.history()[-1].consumed_ids == {"i1", "r1", consumed_sample_id("harness:r1:0")}
        # The job ended in submit: its checkpoint and its end follow, and consume nothing.
        checkpoint = None
        while scenario.trainer_for("harness").job_running:
            polled = scenario.prepare_training_step("harness")
            checkpoint = polled if polled.artifact is not None else checkpoint
            scenario.commit(polled, component="harness")
        assert checkpoint is not None
        assert all(record.consumed_ids == frozenset() for record in scenario.store.history()[1:])
        status = dispatcher.build_training_status()["scenarios"]["agent"]
        assert status["components"]["harness"]["last_committed_step"]["step"] == 3
        assert status["components"]["config"]["last_committed_step"] is None
        manifest = RequestService(dispatcher).harness_manifest({"x-reef-scenario": "agent"}, form="source")
        assert manifest["files"] == {"harness.txt": "harness step 1"}
        assert set(manifest["components"]) == {"harness", "config"}
        assert manifest["components"]["harness"] == checkpoint.artifact.ref.content_id


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
    assert [
        component_trainer.trainer.pipeline.context.admitted_report_type is mixed.report_type
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
    assert recipe.report_type is ScoredRolloutReport
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
    with closing(_serve(_served_composite(tmp_path), tmp_path)) as dispatcher:
        headers = {"x-reef-scenario": "agent"}
        # A rollout runs for a loaded scenario's step, so its calls find the scenario loaded.
        assert dispatcher.get_or_create_scenario("agent") is not None

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
    """The harness head of a flat scenario is the served release, found on the chain when the log keeps no rows."""
    tree = tmp_path / "seed"
    tree.mkdir()
    (tree / "rules.md").write_text("seed")
    dispatcher = Dispatcher(
        _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"rules.md": "seed"}),
        InMemoryRepositoryBackend.factory(tree, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(None),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and not scenario.store.durable
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "rules.md").write_text("evolved")
        scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(evolved)))
        head = scenario.current_artifact_ref().release_id
        assert service.harness_head(headers) == head
        source = service.harness_manifest(headers, form="source")
        assert source["release_id"] == head and source["files"] == {"rules.md": "evolved"}
        # The default form renders the source tree for the adapter.
        assert service.harness_manifest(headers)["files"]["plugin/rules.md"].startswith("evolved")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_evaluation_route_names_the_scenario_in_its_path(tmp_path: Path) -> None:
    with closing(_serve(_served_composite(tmp_path), tmp_path)) as dispatcher:
        assert dispatcher.get_or_create_scenario("agent") is not None

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
        super().__init__(context, None, None)
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

    def step_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        return threshold_pipeline(context, sample2job=_ClosingJobData(context, self.closing, self.rollout_done))


@pytest.mark.unit
def test_an_evaluation_call_during_a_delete_neither_holds_it_up_nor_brings_the_scenario_back(tmp_path: Path) -> None:
    """A delete closes the scenario's pipeline, which waits for the evaluation rollout in flight; the rollout's call
    must not wait for the scenario the delete holds, and must not create the scenario again once the delete ends."""
    harness = _RolloutTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"})
    runtime = InferenceProxyRuntime(base_url="http://engine.test", model_path="m")
    dispatcher = _serve(
        CompositeRecipe(components={"harness": harness, "config": _ConfigRecipe()}, runtime=runtime), tmp_path
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


@dataclass(frozen=True)
class _EndpointTreeRecipe(_TreeRecipe):
    """The tree recipe, keeping the endpoint the service hands it."""

    endpoint: ServedEndpoint | None = None

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> _EndpointTreeRecipe:
        return replace(self, endpoint=endpoint)


@pytest.mark.unit
@pytest.mark.parametrize("composite", [True, False])
def test_only_a_composite_evaluates_through_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, composite: bool
) -> None:
    """A composite's evaluation calls come back to the service and hold its local cycles until it answers. A recipe of
    one component keeps calling its runtime's endpoint directly: it gets no served endpoint and runs at once."""
    from reef.service import assembly
    from reef.service.deploy.service_config import ServiceConfig

    flat = _EndpointTreeRecipe(label="harness")
    recipe = (
        CompositeRecipe(components={"harness": flat, "tools": _EndpointTreeRecipe(label="tools")})
        if composite
        else flat
    )
    built: dict[str, Any] = {}

    def dispatcher(recipe: Recipe, *args: Any, **kwargs: Any) -> SimpleNamespace:
        built.update(recipe=recipe, hold=kwargs["hold_local_cycles"])
        return SimpleNamespace(services=ServiceSet.of(kwargs["components"]))

    monkeypatch.setattr(assembly, "_serving_recipe", lambda *args: recipe)
    monkeypatch.setattr(assembly, "Dispatcher", dispatcher)
    monkeypatch.setattr(GitLFSRepositoryBackendFactory, "from_config", lambda *args, **kwargs: lambda name: object())
    settings = ServiceConfig(recipe="recipe", agent_record_dir=str(tmp_path), port=8900)
    assembly.build_dispatcher(
        settings, hold_local_cycles=True, evaluation_token="rollout", components=frozenset({"control"})
    )
    if composite:
        assert built["hold"] is True
        assert built["recipe"].components["harness"].endpoint == ServedEndpoint(
            "http://127.0.0.1:8900", token="rollout", component="harness"
        )
    else:
        assert built["hold"] is False and built["recipe"].endpoint is None


@pytest.mark.unit
def test_a_composite_names_each_component_in_its_served_endpoint(tmp_path: Path) -> None:
    composite = CompositeRecipe(
        components={"harness": _EndpointTreeRecipe(label="harness"), "tools": _EndpointTreeRecipe(label="tools")}
    )
    served = composite.with_served_endpoint(ServedEndpoint("http://127.0.0.1:8900", token="t"))
    assert {name: recipe.endpoint for name, recipe in served.components.items()} == {  # type: ignore[attr-defined]
        "harness": ServedEndpoint("http://127.0.0.1:8900", token="t", component="harness"),
        "tools": ServedEndpoint("http://127.0.0.1:8900", token="t", component="tools"),
    }


class _RefuseAll(ArtifactValidator):
    def validate(self, artifact: Artifact) -> None:
        raise ValueError("validator refused the artifact")


@dataclass(frozen=True)
class _GuardedTreeRecipe(_TreeRecipe):
    """A recipe written before admission moved onto the component surface: it overrides build_artifact_validator."""

    def build_artifact_validator(self) -> ArtifactValidator:
        return _RefuseAll()


@pytest.mark.unit
def test_a_recipe_that_overrides_build_artifact_validator_still_admits_through_it(tmp_path: Path) -> None:
    """The check joins the served component's own, alone or as a component of a composite; a recipe serving
    several components cannot say which one it admits, so it is refused at build."""
    flat = _GuardedTreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"})
    with closing(_serve(flat, tmp_path / "flat")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        head = scenario.current_artifact_ref()
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "AGENTS.md").write_text("unvetted")
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(evolved)))
        assert scenario.current_artifact_ref() == head
    composite = CompositeRecipe(components={"harness": flat, "config": _ConfigRecipe()})
    with pytest.raises(ValueError, match="validator refused the artifact"):
        composite.build_surface("agent").components["harness"].validator.validate(Artifact.local(evolved))

    @dataclass(frozen=True)
    class GuardedComposite(CompositeRecipe):
        def build_artifact_validator(self) -> ArtifactValidator:
            return _RefuseAll()

    guarded = GuardedComposite(components={"harness": _TreeRecipe(label="harness"), "config": _ConfigRecipe()})
    with pytest.raises(RecipeConfigError, match="overrides build_artifact_validator but serves components"):
        guarded.serving_surface("agent")


@dataclass(frozen=True)
class _GuardedRecordOnlyRecipe(Recipe):
    """A record only recipe (the default surface, no component) that keeps main's admission hook."""

    def build_artifact_validator(self) -> ArtifactValidator:
        return _RefuseAll()


@pytest.mark.unit
def test_a_recipe_serving_no_component_admits_the_whole_release_through_its_validator(tmp_path: Path) -> None:
    """With no component to join, the check admits the release as a whole, as every recipe's did before components
    existed: the scenario builds, and a release the check refuses is not published."""
    (tmp_path / "initial").mkdir()
    dispatcher = _serve(_GuardedRecordOnlyRecipe(), tmp_path)  # type: ignore[arg-type]
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None and scenario.surface.names == ()
        head = scenario.current_artifact_ref()
        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "notes.txt").write_text("unvetted")
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.surface.validate(Artifact.local(evolved))
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(evolved)))
        assert scenario.current_artifact_ref() == head
    finally:
        dispatcher.close()


class _RefuseUnvetted(ArtifactValidator):
    """Refuse a release any file of which says unvetted."""

    def validate(self, artifact: Artifact) -> None:
        root = artifact.materialize().local_path
        files = [] if root is None else [path for path in Path(root).rglob("*") if path.is_file()]
        if any("unvetted" in path.read_text(encoding="utf-8", errors="ignore") for path in files):
            raise ValueError("validator refused the artifact")


@dataclass(frozen=True)
class _LegacyTreeRecipe(_TreeRecipe):
    """A recipe written before components: the default surface, a tree it publishes, and main's admission hook."""

    def build_surface(self, scenario: str) -> Surface:
        return Surface()

    def build_artifact_validator(self) -> ArtifactValidator:
        return _RefuseUnvetted()


def _tree(tmp_path: Path, name: str, text: str) -> Artifact:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "notes.txt").write_text(text, encoding="utf-8")
    return Artifact.local(directory)


@pytest.mark.unit
def test_inside_a_composite_a_recipe_serving_no_component_admits_its_component_through_its_validator(
    tmp_path: Path,
) -> None:
    """Flat, such a recipe's release is the whole artifact; inside a composite its release is its component, so its
    check admits that component's steps as it admitted its whole release flat."""
    composite = CompositeRecipe(
        components={
            "notes": _LegacyTreeRecipe(label="notes", artifact_dir=tmp_path / "steps"),
            "config": _ConfigRecipe(),
        }
    )
    with pytest.raises(ValueError, match="validator refused the artifact"):
        composite.serving_surface("agent").components["notes"].validator.validate(_tree(tmp_path, "u", "unvetted"))
    with closing(_serve(composite, tmp_path / "svc")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        seed = scenario.current_artifact_ref()
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "c", "unvetted")), component="notes")
        assert scenario.current_artifact_ref() == seed
        scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "v", "vetted")), component="notes")
        assert scenario.current_artifact_ref() != seed


@dataclass(frozen=True)
class _ReleaseCheckedComposite(CompositeRecipe):
    """A composite that sets the release's own check itself, beside each component's."""

    def build_surface(self, scenario: str) -> Surface:
        return replace(super().build_surface(scenario), validator=_RefuseUnvetted())


@pytest.mark.unit
def test_a_commit_runs_the_release_check_an_activation_of_that_release_runs(tmp_path: Path) -> None:
    """The release's own check runs on the composed release a step publishes, as an activation of it runs it: a
    release the commit admits can be activated again, and one the activation would refuse is never published."""
    composite = _ReleaseCheckedComposite(
        components={
            "harness": _TreeRecipe(label="harness", artifact_dir=tmp_path / "steps", seed={"AGENTS.md": "seed"}),
            "config": _ConfigRecipe(),
        }
    )
    with closing(_serve(composite, tmp_path / "svc")) as dispatcher:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        seed = scenario.current_artifact_ref()
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "c", "unvetted")), component="harness")
        assert scenario.current_artifact_ref() == seed
        scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "v", "vetted")), component="harness")
        published = scenario.current_artifact_ref()
        assert published != seed
        # The creation made no artifact: it carries the base's, under the base release's ID.
        seed_harness = scenario.repository.materialize(seed).components.artifact_ids(seed.release_id)["harness"]
        assert seed_harness == f"{scenario.repository.backend.resolve_release('base').release_id}:harness"
        scenario.activate(seed_harness, "harness")
        scenario.activate(f"{published.release_id}:harness", "harness")
        served = scenario.repository.materialize(scenario.current_artifact_ref()).components
        made = scenario.repository.materialize(published).components
        assert served is not None and made is not None
        assert served.content_ids == made.content_ids


class _CountedChecks(_RefuseUnvetted):
    """Refuse what the parent refuses, and count the trees it judged."""

    def __init__(self) -> None:
        self.count = 0

    def validate(self, artifact: Artifact) -> None:
        self.count += 1
        super().validate(artifact)


@dataclass(frozen=True)
class _CountedTreeRecipe(_LegacyTreeRecipe):
    """The legacy tree recipe whose admission hook counts what it judged."""

    checks: _CountedChecks | None = None

    def build_artifact_validator(self) -> ArtifactValidator:
        assert self.checks is not None
        return self.checks


@dataclass(frozen=True)
class _UnseededConfigRecipe(_ConfigRecipe):
    """A configuration recipe that seeds nothing: its component starts empty, which its own check would refuse."""

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource()


@pytest.mark.unit
def test_an_activation_never_judges_a_component_again(tmp_path: Path) -> None:
    """A component whose recipe seeds nothing starts empty, and the other component's steps carry it as it is. An
    activation of one of those steps, or of a held one, runs only the release's own check: each component was
    admitted when its step published it, and judging the carried one would refuse releases their commits admitted."""
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
        scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "a", "one")), component="harness")
        first = scenario.current_artifact_ref()
        scenario.commit(TrainStepResult(state={}, artifact=_tree(tmp_path, "b", "two")), component="harness")
        checks.count = 0
        dispatcher.activate("agent", f"{first.release_id}:harness", "harness")
        served = scenario.repository.materialize(scenario.current_artifact_ref()).components
        assert served is not None
        current = scenario.current_artifact_ref()
        assert served.artifact_ids(current.release_id) == scenario.store.history()[0].components
        assert checks.count == 0
        head = scenario.current_artifact_ref()
        scenario.commit(
            TrainStepResult(state={}, artifact=_tree(tmp_path, "c", "three"), changed_kinds=frozenset({"held"})),
            component="harness",
        )
        assert scenario.current_artifact_ref() == head
        held = scenario.store.history()[-1].artifact_ref.release_id
        assert dispatcher.activate("agent", f"{held}:harness", "harness").artifact_id == f"{held}:harness"
        assert checks.count == 1


@pytest.mark.unit
@pytest.mark.parametrize("component_config", [[["training_mode", "manual"]], "manual", [], "", None])
def test_a_component_config_that_is_not_an_object_is_refused_under_a_composite_mode(component_config: Any) -> None:
    harness = {"implementation": TREE, "config": component_config}
    with pytest.raises(RecipeConfigError, match=r"components\.harness\.config must be an object"):
        build_recipe(COMPOSITE, {}, config=composite_config(harness, data={"training_mode": "manual"}))


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
    """A component listed first that runs no step does not lend its mode and processor to the whole scenario: the
    config component builds no batch, so the harness alone decides whether instructions are taken."""
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
            "pending_instructions": harness.pending_instructions(),
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
    """The documented composite takes instructions from its config: the mode reaches every component, and the
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
