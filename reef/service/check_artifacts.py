"""``reef check-artifacts``: check that every scenario's commit history resolves to stored files.

Run it after you stop the writers and take the backup, before the first new record. The check can run any number of
times. It fills the artifact cache and appends no commit. Opening a scenario's storage can upgrade its schema, as a
service start does.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping

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
        if record.snapshot.durable:
            releases[record.snapshot.revision] = record.snapshot
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


def repeated_artifacts(backend: RepositoryBackend, artifact_ids: Iterable[str]) -> int:
    """How many of ``artifact_ids`` hold the same files and metadata as an earlier one.

    Reef reuses a loaded artifact only by its ID, so each of these artifacts can cost one more load when it serves.
    The result is the most extra loads: an artifact that never serves costs none. An ID without a binding is left out.
    """
    contents: set[str] = set()
    repeated = 0
    for artifact_id in artifact_ids:
        try:
            ref = backend.artifact_snapshot(artifact_id)
        except ArtifactNotFound:
            continue
        content = json.dumps(backend.artifact_content(ref, artifact_id), sort_keys=True, default=str)
        repeated += content in contents
        contents.add(content)
    return repeated


def check_deployment(settings: ServiceConfig) -> list[str]:
    """The problems of each scenario that the deployment's artifact backend lists."""
    backend_factory = build_artifact_backend(settings)
    if not isinstance(backend_factory, EnumerableRepositoryBackendFactory):
        raise DeployConfigError(f"{type(backend_factory).__name__} cannot list its scenarios")
    storage, data = open_scenario_storage(settings)
    problems = []
    try:
        for name in backend_factory.list_registrations():
            store = storage.open(name)
            try:
                backend = backend_factory(name)
                found = check_scenario(name, backend, store)
                problems.extend(found)
                if not found:
                    artifact_ids = {
                        artifact_id
                        for record in store.history()
                        for artifact_id in (record.artifact_ids or {}).values()
                    }
                    repeated = repeated_artifacts(backend, sorted(artifact_ids))
                    print(f"{name}: {len(artifact_ids)} artifacts, {repeated} can cost one more model load each")
            finally:
                store.close()
    finally:
        # The data store shares the storage's task file store, so it closes first.
        data.close()
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
