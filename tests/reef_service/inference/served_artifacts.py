"""Shared setup for the served-artifact tests: a runtime that serves under a model name, and two trained artifacts."""

from __future__ import annotations

from tests.support.artifact_versions import ImportingComponent, ImportingInference, train
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import live_objects

from reef.artifact import InMemoryRepositoryBackend, RepositoryBackendFactory
from reef.dispatcher import Dispatcher
from reef.inference.interfaces import InferenceHandler
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage


class ArtifactEcho(InferenceHandler):
    """Answer each call with the snapshot it was given, so a test sees which snapshot answered."""

    def __init__(self) -> None:
        self.artifact_ids: list[str] = []

    async def inference(self, artifact, path, payload):
        del path, payload
        self.artifact_ids.append(artifact.artifact_id)
        return {"choices": [{"message": {"content": "ok"}}]}


class NamedInference(ImportingInference):
    """An ``ImportingInference`` whose engine serves every scenario under one model name, so inference writes the
    scenario's ``served_models`` rows and the inference routes answer."""

    def served_model_name(self, scenario: str) -> str:
        return "stub-model"


class TwoSlotInference(NamedInference):
    """A ``NamedInference`` that keeps two artifacts of a scenario loaded at once, as engines with two checkpoint
    slots do. It answers only from a loaded artifact, so it evaluates no candidate."""

    @property
    def served_artifact_capacity(self) -> int:
        return 2


class NamedComponent(ImportingComponent):
    """An ``ImportingComponent`` whose checkpoints a ``NamedInference`` of ``inference_type`` serves."""

    def __init__(self, checkpoint_dir, inference_type: type[NamedInference] = NamedInference) -> None:
        super().__init__(checkpoint_dir)
        self.inference = inference_type(self.restore_order)


def serving_dispatcher(
    tmp_path, inference_type: type[NamedInference] = NamedInference
) -> tuple[Dispatcher, NamedComponent, RepositoryBackendFactory]:
    """The dispatcher of ``tests.support.artifact_versions.dispatcher`` whose runtime serves under a model name."""
    component = NamedComponent(tmp_path / "exported", inference_type)
    backend_factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    value = Dispatcher(
        TestPolicyRecipe(),
        backend_factory,
        live=live_objects(component, component.inference),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
    )
    return value, component, backend_factory


def answering_each_artifact(runtime) -> None:
    """Make the stub runtime answer each call from the snapshot that the call names, as Tinker does."""
    runtime.inference.answers_each_artifact = True


def two_artifacts(tmp_path, *, checkpoint_slots: bool = False):
    """A scenario that trained twice: the newest artifact is active, and ``older`` is the one before it.

    Its runtime answers each call from the artifact that the call names. With ``checkpoint_slots``, it is a
    ``TwoSlotInference`` instead.
    """
    value, runtime, backend_factory = serving_dispatcher(
        tmp_path, TwoSlotInference if checkpoint_slots else NamedInference
    )
    if not checkpoint_slots:
        answering_each_artifact(runtime)
    train(value, 1)
    train(value, 2)
    newest, older, _ = value.get_or_create_scenario("math").artifacts()
    return value, runtime, backend_factory, newest, older
