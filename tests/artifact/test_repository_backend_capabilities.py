"""Commit-history capability is checked before scenario registration or recovery."""

from collections.abc import Mapping

import pytest

from reef.artifact import (
    Artifact,
    ArtifactPublicationError,
    EnumerableRepositoryBackendFactory,
    InMemoryRepositoryBackend,
    RegistrationAwareRepositoryBackendFactory,
    RepositoryBackend,
    RepositoryBackendFactory,
    SnapshotRef,
    StagedSnapshotRepositoryBackend,
)
from reef.dispatcher import build_default_dispatcher
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage


class BasicBackend(RepositoryBackend):
    """A custom backend exposing only the base storage contract."""

    def __init__(self, delegate: InMemoryRepositoryBackend) -> None:
        self.delegate = delegate

    def resolve_snapshot(self, revision: str) -> SnapshotRef:
        return self.delegate.resolve_snapshot(revision)

    def artifact_snapshot(self, artifact_id: str) -> SnapshotRef:
        return self.delegate.artifact_snapshot(artifact_id)

    def bind_artifact_id(self, artifact_id: str, ref: SnapshotRef) -> SnapshotRef | None:
        return self.delegate.bind_artifact_id(artifact_id, ref)

    def artifact_content(self, ref: SnapshotRef, artifact_id: str) -> tuple[object, dict[str, object]]:
        return self.delegate.artifact_content(ref, artifact_id)

    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> SnapshotRef:
        return self.delegate.register(root, metadata=metadata)

    def metadata(self) -> Mapping[str, object] | None:
        return self.delegate.metadata()

    def current(self) -> SnapshotRef:
        return self.delegate.current()

    def materialize(self, ref: SnapshotRef) -> Artifact:
        return self.delegate.materialize(ref)

    def publish(self, artifact: Artifact, *, expected_parent: SnapshotRef, advance_head: bool = True) -> SnapshotRef:
        return self.delegate.publish(artifact, expected_parent=expected_parent, advance_head=advance_head)


class IncompleteStagedBackend(BasicBackend, StagedSnapshotRepositoryBackend):
    pass


class UndeclaredStagedBackend(BasicBackend):
    def commit_snapshot(self, ref: SnapshotRef, *, expected_parent: SnapshotRef) -> None:
        self.delegate.commit_snapshot(ref, expected_parent=expected_parent)


class StagedBackend(UndeclaredStagedBackend, StagedSnapshotRepositoryBackend):
    pass


def test_factory_capabilities_are_declared_independently(tmp_path):
    backend = InMemoryRepositoryBackend("math")

    class BasicFactory(RepositoryBackendFactory):
        def __call__(self, scenario):
            return backend

    class RegistrationFactory(BasicFactory, RegistrationAwareRepositoryBackendFactory):
        def has_registration(self, scenario):
            return scenario == "math"

    class ListingFactory(BasicFactory, EnumerableRepositoryBackendFactory):
        def list_registrations(self):
            return ("math",)

    basic = BasicFactory()
    assert basic("math") is backend
    assert not isinstance(basic, RegistrationAwareRepositoryBackendFactory)
    assert not isinstance(basic, EnumerableRepositoryBackendFactory)
    assert isinstance(RegistrationFactory(), RegistrationAwareRepositoryBackendFactory)
    assert not isinstance(RegistrationFactory(), EnumerableRepositoryBackendFactory)
    assert isinstance(ListingFactory(), EnumerableRepositoryBackendFactory)
    assert not isinstance(ListingFactory(), RegistrationAwareRepositoryBackendFactory)
    cached = InMemoryRepositoryBackend.factory()
    assert isinstance(cached, RegistrationAwareRepositoryBackendFactory)
    assert isinstance(cached, EnumerableRepositoryBackendFactory)


@pytest.mark.parametrize("backend_type", [BasicBackend, UndeclaredStagedBackend])
@pytest.mark.parametrize("registered", [False, True])
def test_commit_history_rejects_basic_backend_before_reading_or_changing_registration(
    tmp_path, monkeypatch, registered, backend_type
):
    backend = backend_type(InMemoryRepositoryBackend("math"))
    if registered:
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend,
            scenario_storage=SQLiteScenarioStorage(),
            coordination=memory_coordination(),
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
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "metadata", unexpected_metadata)
            with pytest.raises(ArtifactPublicationError, match="StagedSnapshotRepositoryBackend"):
                dispatcher.get_or_create_scenario("math")
        assert backend.metadata() == before
    finally:
        dispatcher.close()


def test_basic_backend_remains_usable_without_commit_log(tmp_path):
    backend = BasicBackend(InMemoryRepositoryBackend("math"))
    for _ in range(2):
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend,
            scenario_storage=SQLiteScenarioStorage(),
            coordination=memory_coordination(),
        )
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            assert not scenario.store.durable
            assert scenario.current_snapshot() == backend.current()
        finally:
            dispatcher.close()


def test_staged_backend_requires_a_commit_implementation(tmp_path):
    with pytest.raises(TypeError, match="commit_snapshot"):
        IncompleteStagedBackend(InMemoryRepositoryBackend("math"))


def test_custom_staged_backend_can_create_and_recover_scenario_with_commit_log(tmp_path):
    backend = StagedBackend(InMemoryRepositoryBackend("math"))
    for _ in range(2):
        dispatcher = build_default_dispatcher(
            backend_factory=lambda name: backend,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            coordination=memory_coordination(),
        )
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            assert scenario.store.durable
            assert scenario.current_snapshot() == backend.current()
        finally:
            dispatcher.close()
