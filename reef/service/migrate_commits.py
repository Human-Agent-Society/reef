"""``reef migrate-commits``: rewrite each scenario's older commit records in the current format.

Run it after you stop the writers and take the backup. Without ``--apply``, it changes nothing and reports what it
would rewrite. With ``--apply``, it binds the artifact IDs of the older records and rewrites the records of each
scenario in one transaction. It can run any number of times: a record in the current format stays as it is. Git
history does not change, so a snapshot without an explicit artifact ID keeps its revision-based ID.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace

from reef.artifact.artifact import ArtifactError, SnapshotRef
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackend
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import snapshot_components
from reef.core.errors import DeployConfigError
from reef.scenario.factory import RepositoryCommitSnapshots
from reef.service.assembly import build_artifact_backend, open_scenario_storage
from reef.service.check_artifacts import check_scenario
from reef.service.deploy.orchestrator import load_deployment
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitHistoryError,
    CommitRecord,
    encode_commit,
    parse_scenario_metadata,
)
from reef.storage.scenario import ScenarioStore


def snapshot_selection(backend: RepositoryBackend, ref: SnapshotRef, flat_component: str | None) -> ArtifactSelection:
    """The artifact ID of each component of the stored snapshot ``ref``.

    A flat snapshot holds one artifact, which ``flat_component`` names. Raise ``ValueError`` when it is None.
    """
    snapshot = backend.materialize(ref)
    manifest = snapshot.components
    if manifest is not None:
        return ArtifactSelection(manifest.artifact_ids(ref.revision))
    if flat_component is None:
        raise ValueError(f"the flat snapshot {ref.revision} has no component name in the scenario's manifest")
    return ArtifactSelection({flat_component: snapshot.artifact_id})


def migrated_record(record: CommitRecord, backend: RepositoryBackend, flat_component: str | None) -> CommitRecord:
    """``record`` in the current format: with its selection, and without a storage location.

    A record of a local snapshot without a selection has no artifact ID to name, and stays as it is.
    """
    if record.selection is not None:
        return replace(record, legacy_locators={})
    if not record.snapshot.durable:
        return record
    return replace(record, selection=snapshot_selection(backend, record.snapshot, flat_component))


def check_located(
    records: Sequence[CommitRecord], backend: RepositoryBackend, start: SnapshotRef
) -> list[CommitRecord]:
    """``records`` with the snapshots that the backend's bindings give them, as a read of the rewritten rows does.

    Raise ``CommitHistoryError`` when a binding is missing, and ``ValueError`` when a snapshot does not hold the
    artifact that its record names.
    """
    unlocated = [CommitRecord.from_dict(record.to_dict()) for record in records]
    located = list(RepositoryCommitSnapshots(backend, start).locate(unlocated))
    for record in located:
        if record.selection is None or not record.snapshot.durable or not record.made_artifact:
            continue
        held = snapshot_selection(backend, record.snapshot, next(iter(record.selection.components), None))
        if not set(record.selection.components.items()) & set(held.components.items()):
            raise ValueError(
                f"step {record.step}: snapshot {record.snapshot.revision} holds no artifact of the record"
            )
    return located


def migrate_scenario(
    name: str, backend: RepositoryBackend, store: ScenarioStore, *, apply: bool
) -> tuple[int, list[str]]:
    """The number of records of the scenario ``name`` not in the current format, and each problem.

    With ``apply`` and no problem, the records are in the current format afterwards. With a problem, no record changes.
    """
    if not isinstance(store, CommitHistoryScenarioStore):
        return 0, [f"{name}: this scenario storage keeps no commit history that Reef can rewrite"]
    problems = check_scenario(name, backend, store)
    if problems:
        return 0, problems
    metadata = backend.metadata() or {}
    registration = metadata[SCENARIO_METADATA_KEY]
    if not isinstance(registration, Mapping):
        return 0, [f"{name}: the head snapshot has no scenario metadata"]
    _, start, _, _ = parse_scenario_metadata(registration, checkpoint_head=backend.current())
    # The head of an older flat scenario can hold no manifest, while its root does.
    manifest = snapshot_components(metadata) or backend.materialize(start).components
    flat_component = manifest.names[0] if manifest is not None and manifest.single else None
    history = store.commit_history
    stored = history.stored_commits()
    try:
        records = [migrated_record(record, backend, flat_component) for record in store.history()]
        changed = [record for record in records if encode_commit(record) != stored[record.step]]
        if not apply or not changed:
            return len(changed), []
        for previous in store.history():
            for ref in (
                *previous.legacy_locators.values(),
                *([previous.snapshot] if previous.snapshot.durable else []),
            ):
                backend.index_snapshot(ref)
        check_located(records, backend, start)
    except (ArtifactError, CommitHistoryError, ValueError) as exc:
        return 0, [f"{name}: {exc}"]
    history.rewrite(changed)
    store.use_snapshots(RepositoryCommitSnapshots(backend, start))
    return len(changed), []


def migrate_deployment(settings: ServiceConfig, *, apply: bool) -> list[str]:
    """Migrate each scenario that the deployment's artifact backend lists; print its count, return each problem."""
    backend_factory = build_artifact_backend(settings)
    if not isinstance(backend_factory, EnumerableRepositoryBackendFactory):
        raise DeployConfigError(f"{type(backend_factory).__name__} cannot list its scenarios")
    storage, _ = open_scenario_storage(settings)
    problems: list[str] = []
    try:
        for name in backend_factory.list_registrations():
            store = storage.open(name)
            try:
                count, found = migrate_scenario(name, backend_factory(name), store, apply=apply)
            finally:
                store.close()
            problems.extend(found)
            if not found:
                print(f"{name}: {count} record(s) {'rewritten' if apply else 'to rewrite'}")
    finally:
        storage.close()
    return problems


def main(argv: list[str]) -> int:
    """Print each count and each problem; the exit status is 1 when there is a problem, else 0."""
    parser = argparse.ArgumentParser(prog="reef migrate-commits", description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="the deployment's config file")
    parser.add_argument("--apply", action="store_true", help="rewrite the records; without it, only report them")
    arguments = parser.parse_args(argv)
    settings = service_config_from_mapping(load_deployment(arguments.config, None).resolved)
    problems = migrate_deployment(settings, apply=arguments.apply)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0
