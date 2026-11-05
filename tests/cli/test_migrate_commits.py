"""``reef migrate-commits``: older commit records in the current format, with the same artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.artifact.artifact import Artifact
from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.core.components import COMPONENTS_METADATA_KEY
from reef.core.snapshot import RootSource, encode_snapshot_ref
from reef.service.migrate_commits import migrate_scenario
from reef.storage.commits import (
    RECORD_KIND,
    SCENARIO_METADATA_KEY,
    RecordProgress,
    parse_scenario_metadata,
    scenario_metadata_for,
)
from reef.storage.sqlite import SQLiteScenarioStorage


def stored_tree(directory: Path, text: str, metadata: dict) -> Artifact:
    directory.mkdir()
    (directory / "rules.md").write_text(text)
    return Artifact.local(directory, metadata=metadata)


@pytest.fixture
def published(tmp_path: Path, fake_git_lfs: None):
    """A scenario whose commit history holds one ``reef-commit/6`` row, which names its snapshot."""
    backend = GitLFSRepositoryBackend(
        "math", tmp_path / "artifacts.git", work_dir=tmp_path / "work", cache_dir=tmp_path / "cache"
    )
    root_tree = stored_tree(tmp_path / "root", "root", {})
    metadata = {
        SCENARIO_METADATA_KEY: scenario_metadata_for(name="math", source=RootSource("default")),
        COMPONENTS_METADATA_KEY: {"model": {"artifact_id": root_tree.artifact_id, "metadata": {}}},
    }
    root = backend.register(root_tree, metadata=metadata)
    step_metadata = scenario_metadata_for(
        name="math",
        source=RootSource("default"),
        start_artifact=root,
        scenario_step=1,
        record_progress=RecordProgress(0, 0),
    )
    trained = stored_tree(tmp_path / "trained", "trained", {SCENARIO_METADATA_KEY: step_metadata})
    published_ref = backend.publish(trained, expected_parent=root)
    store = SQLiteScenarioStorage(tmp_path / "records").open("math")
    history = store.commit_history
    row = {
        "record": "reef-commit/6",
        "scenario": "math",
        "step": 1,
        "artifact_ref": encode_snapshot_ref(published_ref),
        "checkpoint": True,
        "algorithm_state": None,
        "record_progress": {"high_water_sequence": 0, "high_water_offset": 0, "consumed_ids": []},
        "recorded_at": 1000.0,
    }
    with history._transaction(write=True) as connection:
        connection.execute(
            history.table.table.insert().values(**history.table.scope, step=1, commit_json=json.dumps(row))
        )
    yield backend, store
    store.close()


def store_root(backend: GitLFSRepositoryBackend) -> str:
    registration = (backend.metadata() or {})[SCENARIO_METADATA_KEY]
    return parse_scenario_metadata(registration, checkpoint_head=backend.current())[1].revision


def test_a_report_changes_nothing_and_an_apply_rewrites_each_older_record_once(published) -> None:
    backend, store = published
    before = dict(store.commit_history.stored_commits())
    assert migrate_scenario("math", backend, store, apply=False) == (1, [])
    assert store.commit_history.stored_commits() == before

    assert migrate_scenario("math", backend, store, apply=True) == (1, [])
    row = json.loads(store.commit_history.stored_commits()[1])
    assert row["record"] == RECORD_KIND and "artifact_ref" not in row and "snapshot" not in row
    [record] = store.history()
    assert record.selection is not None and set(record.selection.components) == {"model"}
    assert backend.artifact_snapshot(record.selection.components["model"]).durable

    assert migrate_scenario("math", backend, store, apply=True) == (0, [])


def test_a_flat_snapshot_without_a_component_name_is_a_problem_and_changes_nothing(published, monkeypatch) -> None:
    backend, store = published
    root = backend.materialize(backend.resolve_snapshot(store_root(backend)))
    monkeypatch.setattr(type(root), "components", property(lambda self: None))
    before = dict(store.commit_history.stored_commits())
    count, problems = migrate_scenario("math", backend, store, apply=True)
    assert count == 0 and "no component name" in problems[0]
    assert store.commit_history.stored_commits() == before
