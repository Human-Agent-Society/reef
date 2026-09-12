"""Commit-log capability is checked before scenario registration or recovery."""

from collections.abc import Mapping

import pytest

from reef.artifact import (
    Artifact,
    ArtifactPublicationError,
    ArtifactRef,
    InMemoryRepositoryBackend,
    RepositoryBackend,
    StagedReleaseRepositoryBackend,
)
from reef.dispatcher import build_default_dispatcher
from reef.storage.factory import SQLiteScenarioStoreFactory


class BasicBackend(RepositoryBackend):
    """A custom backend exposing only the base storage contract."""

    def __init__(self, delegate: InMemoryRepositoryBackend) -> None:
        self.delegate = delegate

    def resolve_release(self, release_id: str | None = None) -> ArtifactRef:
        return self.delegate.resolve_release(release_id)

    def fork(self, release_id: str | None = None, *, metadata: Mapping[str, object] | None = None) -> ArtifactRef:
        return self.delegate.fork(release_id, metadata=metadata)

    def metadata(self) -> Mapping[str, object] | None:
        return self.delegate.metadata()

    def current(self) -> ArtifactRef:
        return self.delegate.current()

    def materialize(self, ref: ArtifactRef) -> Artifact:
        return self.delegate.materialize(ref)

    def publish(self, artifact: Artifact, *, expected_parent: ArtifactRef, advance_head: bool = True) -> ArtifactRef:
        return self.delegate.publish(artifact, expected_parent=expected_parent, advance_head=advance_head)


class IncompleteStagedBackend(BasicBackend, StagedReleaseRepositoryBackend):
    pass


class UndeclaredStagedBackend(BasicBackend):
    def commit_release(self, ref: ArtifactRef, *, expected_parent: ArtifactRef) -> None:
        self.delegate.commit_release(ref, expected_parent=expected_parent)


class StagedBackend(UndeclaredStagedBackend, StagedReleaseRepositoryBackend):
    pass


@pytest.mark.parametrize("backend_type", [BasicBackend, UndeclaredStagedBackend])
@pytest.mark.parametrize("registered", [False, True])
def test_commit_log_rejects_basic_backend_before_reading_or_changing_registration(
    tmp_path, monkeypatch, registered, backend_type
):
    backend = backend_type(InMemoryRepositoryBackend("math", tmp_path))
    if registered:
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend, scenario_store_factory=SQLiteScenarioStoreFactory()
        )
        try:
            dispatcher.get_or_create_scenario("math")
        finally:
            dispatcher.close()
    before = backend.metadata()

    def unexpected_metadata():
        pytest.fail("unsupported backend must be rejected before registration or recovery")

    dispatcher = build_default_dispatcher(
        backend_factory=lambda name: backend,
        agent_record_dir=tmp_path / "records",
        scenario_store_factory=SQLiteScenarioStoreFactory(tmp_path / "records"),
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "metadata", unexpected_metadata)
            with pytest.raises(ArtifactPublicationError, match="StagedReleaseRepositoryBackend"):
                dispatcher.get_or_create_scenario("math")
        assert backend.metadata() == before
    finally:
        dispatcher.close()


def test_basic_backend_remains_usable_without_commit_log(tmp_path):
    backend = BasicBackend(InMemoryRepositoryBackend("math", tmp_path))
    for _ in range(2):
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend, scenario_store_factory=SQLiteScenarioStoreFactory()
        )
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            assert not scenario.store.durable
            assert scenario.current_artifact_ref() == backend.current()
        finally:
            dispatcher.close()


def test_staged_backend_requires_a_commit_implementation(tmp_path):
    with pytest.raises(TypeError, match="commit_release"):
        IncompleteStagedBackend(InMemoryRepositoryBackend("math", tmp_path))


def test_custom_staged_backend_can_create_and_recover_scenario_with_commit_log(tmp_path):
    backend = StagedBackend(InMemoryRepositoryBackend("math", tmp_path))
    for _ in range(2):
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend,
            agent_record_dir=tmp_path / "records",
            scenario_store_factory=SQLiteScenarioStoreFactory(tmp_path / "records"),
        )
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            assert scenario.store.durable
            assert scenario.current_artifact_ref() == backend.current()
        finally:
            dispatcher.close()
