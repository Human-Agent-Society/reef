"""``reef check-artifacts``: check that every scenario's commit history resolves to stored files.

Run it after you stop the writers and take the backup, before the first new record. The check can run any number of
times. It fills the artifact cache and appends no commit. Opening a scenario's storage can upgrade its schema, as a
service start does.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from reef.artifact.artifact import ArtifactError, ArtifactNotFound, SnapshotRef, artifact_components
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackend
from reef.core.errors import DeployConfigError
from reef.scenario.factory import RepositoryCommitSnapshots
from reef.service.assembly import build_artifact_backend, open_scenario_storage
from reef.service.deploy.orchestrator import load_deployment
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitHistoryError, parse_scenario_metadata
from reef.storage.scenario import ScenarioStore


def check_scenario(name: str, backend: RepositoryBackend, store: ScenarioStore) -> list[str]:
    """Each problem in the scenario ``name``, as one line; empty when each record resolves to stored files.

    A problem is a history that cannot be read or located, a release that cannot be materialized, or an artifact ID
    of a record that no release and no binding names.
    """
    registration = (backend.metadata() or {}).get(SCENARIO_METADATA_KEY)
    if not isinstance(registration, Mapping):
        return [f"{name}: the head release has no scenario metadata"]
    try:
        _, start, _, _ = parse_scenario_metadata(registration, checkpoint_head=backend.current())
        store.use_snapshots(RepositoryCommitSnapshots(backend, start))
        records = store.history()
    except (ArtifactError, CommitHistoryError, ValueError) as exc:
        return [f"{name}: cannot read the history: {exc}"]
    releases: dict[str, SnapshotRef] = {start.revision: start, backend.current().revision: backend.current()}
    for record in records:
        for ref in (record.snapshot, *record.legacy_locators.values()):
            if ref.durable:
                releases[ref.revision] = ref
    problems = []
    stored_ids: set[str] = set()
    for revision, ref in releases.items():
        try:
            stored_ids.update(artifact_components(backend.materialize(ref)))
        except ArtifactError as exc:
            problems.append(f"{name}: release {revision} has no stored files: {exc}")
    for record in records:
        for artifact_id in (record.artifact_ids or {}).values():
            if artifact_id in stored_ids:
                continue
            try:
                backend.artifact_snapshot(artifact_id)
            except ArtifactNotFound:
                problems.append(f"{name}: step {record.step} names the missing artifact {artifact_id}")
    return problems


def check_deployment(settings: ServiceConfig) -> list[str]:
    """The problems of each scenario that the deployment's artifact backend lists."""
    backend_factory = build_artifact_backend(settings)
    if not isinstance(backend_factory, EnumerableRepositoryBackendFactory):
        raise DeployConfigError(f"{type(backend_factory).__name__} cannot list its scenarios")
    storage, _ = open_scenario_storage(settings)
    problems = []
    try:
        for name in backend_factory.list_registrations():
            store = storage.open(name)
            try:
                problems.extend(check_scenario(name, backend_factory(name), store))
            finally:
                store.close()
    finally:
        storage.close()
    return problems


def main(argv: list[str]) -> int:
    """Print each problem; the exit status is 1 when there is one, else 0."""
    parser = argparse.ArgumentParser(prog="reef check-artifacts", description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="the deployment's config file")
    arguments = parser.parse_args(argv)
    settings = service_config_from_mapping(load_deployment(arguments.config, None).resolved)
    problems = check_deployment(settings)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0
