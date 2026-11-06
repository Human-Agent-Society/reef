"""``reef check-artifacts`` finds the records that do not resolve to stored files."""

from pathlib import Path

import pytest

from reef.artifact import Artifact, ArtifactError
from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.cli import main
from reef.core.artifact_selection import ArtifactSelection
from reef.core.snapshot import RootSource
from reef.service.check_artifacts import check_scenario, repeated_artifacts
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    UNLOCATED_SNAPSHOT,
    CommitRecord,
    RecordProgress,
    scenario_metadata_for,
)
from reef.storage.sqlite import SQLiteScenarioStorage


def stored_tree(directory: Path, text: str, metadata: dict | None = None) -> Artifact:
    directory.mkdir()
    (directory / "rules.md").write_text(text)
    return Artifact.local(directory, metadata=metadata or {})


def training_record(step: int, artifact_id: str, *, checkpoint: bool = True) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        snapshot=UNLOCATED_SNAPSHOT,
        checkpoint=checkpoint,
        algorithm_state=None,
        high_water_sequence=0,
        high_water_offset=0,
        selection=ArtifactSelection({"model": artifact_id}),
    )


@pytest.fixture
def published(tmp_path: Path, fake_git_lfs: None):
    backend = GitLFSRepositoryBackend(
        "math", tmp_path / "artifacts.git", work_dir=tmp_path / "work", cache_dir=tmp_path / "cache"
    )
    metadata = {SCENARIO_METADATA_KEY: scenario_metadata_for(name="math", source=RootSource("default"))}
    root = backend.register(stored_tree(tmp_path / "root", "root"), metadata=metadata)
    step_metadata = scenario_metadata_for(
        name="math",
        source=RootSource("default"),
        start_artifact=root,
        scenario_step=1,
        record_progress=RecordProgress(0, 0),
    )
    trained = stored_tree(tmp_path / "trained", "trained", {SCENARIO_METADATA_KEY: step_metadata})
    backend.publish(trained, expected_parent=root)
    store = SQLiteScenarioStorage(tmp_path / "records").open("math")
    yield backend, store, trained.artifact_id
    store.close()


def test_a_history_of_stored_artifacts_has_no_problem(published) -> None:
    backend, store, artifact_id = published
    store.commit_step(expected_step=0, commit=training_record(1, artifact_id))
    store.commit_step(expected_step=1, commit=training_record(2, artifact_id, checkpoint=False))
    assert check_scenario("math", backend, store) == []


def test_a_record_of_an_artifact_without_a_binding_is_a_problem(published) -> None:
    backend, store, _ = published
    store.commit_step(expected_step=0, commit=training_record(1, "artifact:missing"))
    problems = check_scenario("math", backend, store)
    assert len(problems) == 1 and "artifact:missing" in problems[0]


def test_a_release_without_stored_files_is_a_problem(published, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, store, artifact_id = published
    store.commit_step(expected_step=0, commit=training_record(1, artifact_id))
    head = backend.current()

    def materialize(ref):
        raise ArtifactError(f"no files at {ref.revision}")

    monkeypatch.setattr(backend, "materialize", materialize)
    problems = check_scenario("math", backend, store)
    assert any(head.revision in problem and "no stored files" in problem for problem in problems)


def test_artifacts_with_the_same_files_count_as_extra_model_loads(published, tmp_path: Path) -> None:
    backend, _, trained_id = published
    same = stored_tree(tmp_path / "same", "trained")
    other = stored_tree(tmp_path / "other", "other")
    head = backend.current()
    same_ref = backend.publish(same, expected_parent=head)
    backend.publish(other, expected_parent=same_ref)
    assert repeated_artifacts(backend, [trained_id, same.artifact_id, other.artifact_id, "artifact:missing"]) == 1


def test_the_command_exits_with_the_check_result(tmp_path: Path, fake_git_lfs: None, capsys) -> None:
    config = tmp_path / "reef.yaml"
    config.write_text(
        "inference:\n"
        "  upstream_url: http://127.0.0.1:9\n"
        "  upstream_model: model\n"
        "storage:\n"
        f"  artifact_repository: {tmp_path / 'artifacts.git'}\n"
        f"  agent_record_dir: {tmp_path / 'records'}\n"
        f"  artifact_work_dir: {tmp_path / 'work'}\n"
        f"  artifact_cache_dir: {tmp_path / 'cache'}\n"
    )
    with pytest.raises(SystemExit) as stopped:
        main(["check-artifacts", "-c", str(config)])
    assert stopped.value.code == 0 and "0 problem(s)" in capsys.readouterr().out
