"""A composite recipe binds one recipe per component; the config component supplies request defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.artifact.composite import compose_release
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.core.reports import ReportValidationError, ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import CompositeRecipe, Recipe, RecipeConfigError, build_recipe
from reef.recipe.checkpoint_strategy import EveryNVersions
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
