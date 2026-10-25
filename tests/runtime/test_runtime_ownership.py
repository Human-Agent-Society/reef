"""The dispatcher, the recipe registry and the service close the training component and runtime they own."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.artifact import GitLFSRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_named_recipe
from reef.runtime.deployment import RuntimeRegistry
from reef.service import assembly
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.sqlite import SQLiteScenarioStorage


class RecordingComponent(StubTrainingComponent):
    """A component that several training scenarios share, which records its close."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(shares_scenarios=True)
        self.events = events

    def close(self) -> None:
        self.events.append("close component")


class RecordingRuntime(StubInferenceRuntime):
    """A runtime that several training scenarios share, each served its own weights."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True

    def shutdown(self) -> None:
        self.events.append("shutdown runtime")


def owned_pair(events: list[str]) -> tuple[RecordingComponent, RecordingRuntime]:
    return RecordingComponent(events), RecordingRuntime(events)


def dispatcher_for(recipe: Recipe, tmp_path) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir()
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


def test_dispatcher_closes_the_component_and_runtime_after_all_scenarios_but_not_on_reload(tmp_path):
    events: list[str] = []
    component, runtime = owned_pair(events)
    dispatcher = dispatcher_for(Recipe(runtime=runtime, training_component=component), tmp_path)
    dispatcher.get_or_create_scenario("math")
    dispatcher.get_or_create_scenario("code")
    dispatcher.registry.reload("math")
    assert events == []

    def track_close(scenario):
        original = scenario.close

        def close():
            events.append(scenario.name)
            original()

        scenario.close = close

    track_close(dispatcher.get_or_create_scenario("math"))
    track_close(dispatcher.get_or_create_scenario("code"))
    dispatcher.close()
    dispatcher.close()
    assert events == ["math", "code", "close component", "shutdown runtime"]
    assert runtime.inference_admission_status["open"] is False


def test_dispatcher_releases_the_runtime_when_scenario_teardown_fails(tmp_path):
    events: list[str] = []
    component, runtime = owned_pair(events)
    dispatcher = dispatcher_for(Recipe(runtime=runtime, training_component=component), tmp_path)
    scenario = dispatcher.get_or_create_scenario("math")
    original_close = scenario.close

    def failed_close():
        original_close()
        raise RuntimeError("scenario teardown failed")

    scenario.close = failed_close
    with pytest.raises(RuntimeError, match="scenario teardown failed"):
        dispatcher.close()
    assert events == ["close component", "shutdown runtime"]
    dispatcher.close()
    assert events == ["close component", "shutdown runtime"]


def test_a_named_recipe_takes_its_configured_runtime_and_an_invalid_one_releases_it(tmp_path):
    events: list[str] = []
    runtimes: list[RecordingRuntime] = []

    def build_runtime(*args):
        runtimes.append(RecordingRuntime(events))
        return runtimes[-1]

    registry = RuntimeRegistry({"injected": build_runtime, "component": lambda *args: RecordingComponent(events)})
    (tmp_path / "valid.yaml").write_text(
        "recipe:\n  implementation: recipe\n  runtime:\n    type: injected\ninference:\n  upstream-model: test-model\n"
    )
    recipe = build_named_recipe("valid", config_directory=tmp_path, environ={}, runtime_registry=registry)
    assert (recipe.training_component, recipe.runtime) == (None, runtimes[0])
    assert events == []
    (tmp_path / "invalid.yaml").write_text(
        """recipe:
  implementation: recipe
  config:
    training_mode: sideways
  runtime:
    type: injected
inference:
  upstream-model: test-model
"""
    )
    with pytest.raises(RecipeConfigError, match="training_mode must be"):
        build_named_recipe("invalid", config_directory=tmp_path, environ={}, runtime_registry=registry)
    assert events == ["shutdown runtime"]
    # A recipe's runtime never trains; training.backend selects the training component.
    (tmp_path / "training.yaml").write_text(
        "recipe:\n  implementation: recipe\n  runtime:\n    type: component\ninference:\n  upstream-model: test-model\n"
    )
    with pytest.raises(TypeError, match=r"training\.backend selects training"):
        build_named_recipe("training", config_directory=tmp_path, environ={}, runtime_registry=registry)
    assert events == ["shutdown runtime", "close component"]


def test_service_app_cleanup_closes_its_owned_runtime_once(monkeypatch, tmp_path):
    events: list[str] = []
    component, runtime = owned_pair(events)
    monkeypatch.setattr(
        assembly, "_serving_recipe", lambda *args: Recipe(runtime=runtime, training_component=component)
    )
    monkeypatch.setattr(GitLFSRepositoryBackendFactory, "from_config", lambda *args, **kwargs: lambda name: object())

    async def run():
        app = assembly.build_app(ServiceConfig(recipe="recipe", agent_record_dir=str(tmp_path)))
        runner = web.AppRunner(app)
        await runner.setup()
        await runner.cleanup()
        await runner.cleanup()

    asyncio.run(run())
    assert events == ["close component", "shutdown runtime"]


def test_failed_service_assembly_releases_its_runtime(monkeypatch, tmp_path):
    events: list[str] = []
    component, runtime = owned_pair(events)
    monkeypatch.setattr(
        assembly, "_serving_recipe", lambda *args: Recipe(runtime=runtime, training_component=component)
    )
    monkeypatch.setattr(GitLFSRepositoryBackendFactory, "from_config", lambda *args, **kwargs: lambda name: object())
    with pytest.raises(ValueError, match="training must be an object"):
        assembly.build_dispatcher(
            ServiceConfig(recipe="recipe", agent_record_dir=str(tmp_path), training_settings="invalid")
        )
    assert events == ["close component", "shutdown runtime"]
