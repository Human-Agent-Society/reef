"""Control commits a checkpoint by the name of its stored content: it reads no directory of the training service,
and it holds none of the checkpoint's bytes."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.component_trainers import (
    HARNESS,
    WEIGHTS,
    TextHarnessRecipe,
    WeightsRecipe,
    _dispatcher,
    harness_recipe,
    two_trainer_recipe,
)

from reef.artifact import (
    Artifact,
    ArtifactNotFoundError,
    GitLFSRepositoryBackend,
    InMemoryRepositoryBackend,
    RepositoryBackendFactory,
)
from reef.artifact.artifact import ArtifactValidator
from reef.artifact.components import ArtifactLayout, ComponentSpec, ScenarioComponents, TextFileTree
from reef.artifact.files import ArtifactFiles
from reef.recipe import Recipe
from reef.scenario import Scenario
from reef.train.types import TrainingCommit

MODEL = "model.safetensors"


class RefuseUnvetted(ArtifactValidator):
    """Refuse an artifact any file of which says unvetted. It reads the files, so the artifact is made local."""

    def validate(self, artifact: Artifact) -> None:
        root = artifact.materialize().local_path
        assert root is not None
        if any("unvetted" in path.read_text(errors="ignore") for path in root.rglob("*") if path.is_file()):
            raise ValueError("validator refused the artifact")


class AnyTree(ArtifactLayout):
    """A layout that accepts every directory as an artifact of its component."""

    def check(self, files: ArtifactFiles) -> None:
        return


@dataclass(frozen=True)
class CheckedHarnessRecipe(TextHarnessRecipe):
    """The harness recipe of the two trainer recipe whose component takes an import, and whose every artifact
    ``RefuseUnvetted`` checks."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        spec = ComponentSpec(files=TextFileTree(), validator=RefuseUnvetted(), layout=AnyTree())
        return ScenarioComponents({HARNESS: spec})


def checked_harness() -> CheckedHarnessRecipe:
    seeded = harness_recipe()
    return CheckedHarnessRecipe(
        evolution=seeded.evolution, evaluation=seeded.evaluation, start=seeded.start, training_mode="auto"
    )


@contextmanager
def control(
    tmp_path: Path, recipe: Recipe | None = None, backends: RepositoryBackendFactory | None = None
) -> Iterator[Scenario]:
    """The scenario ``agent`` in a control process with no training service: a dispatcher over ``recipe`` (the two
    trainer recipe by default) that runs no local cycle, so the test commits each result itself."""
    dispatcher, _ = _dispatcher(
        tmp_path / "control-state",
        recipe=recipe,
        backend_factory=backends,
        hold_local_cycles=True,
        components=frozenset({"control"}),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        yield scenario
    finally:
        dispatcher.close()


def stored_commit(scenario: Scenario, content: str, component: str) -> TrainingCommit:
    """The commit of a checkpoint of ``component``'s trainer whose files are the stored content ``content``."""
    state = dict(scenario.trainer_for(component).state)
    return TrainingCommit(state=state, artifact=Artifact.stored(content), component=component)


def checkpoint(directory: Path, files: dict[str, bytes]) -> Path:
    directory.mkdir(parents=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return directory


def git_backends(tmp_path: Path, name: str) -> RepositoryBackendFactory:
    """The artifact store of one service: its own work and cache directories over the shared repository."""
    return GitLFSRepositoryBackend.factory(
        tmp_path / "artifacts.git", work_dir=tmp_path / name / "work", cache_dir=tmp_path / name / "cache"
    )


@pytest.mark.integration
def test_control_commits_a_checkpoint_that_the_training_service_stored_and_holds_none_of_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    training = git_backends(tmp_path, "training")("agent")
    with control(tmp_path, backends=git_backends(tmp_path, "control")) as scenario:
        seed = scenario.current_snapshot()
        weights = os.urandom(4096)
        trained = checkpoint(tmp_path / "trained", {MODEL: weights, "adapter_config.json": b'{"r": 8}'})
        training.store_content("job-1:1", trained)
        evolved = checkpoint(tmp_path / "evolved", {"AGENTS.md": b"evolved"})
        training.store_content("job-2:0", evolved)
        # Control has no volume of the training service: the directories are gone before it commits.
        shutil.rmtree(trained)
        shutil.rmtree(evolved)

        result = stored_commit(scenario, "job-1:1", WEIGHTS)
        scenario.commit(result, component=WEIGHTS)
        # The harness step carries the weights by their artifact ID: control composes the snapshot by name.
        scenario.commit(stored_commit(scenario, "job-2:0", HARNESS), component=HARNESS)
        first, second = scenario.store.history()[-2:]
        assert (first.snapshot.parent_revision, second.snapshot.parent_revision) == (
            seed.revision,
            first.snapshot.revision,
        )
        assert second.components is not None and first.components is not None
        assert second.components[WEIGHTS] == first.components[WEIGHTS] == result.artifact.artifact_id

        # The control process never held the weights: no file of them, and no LFS object.
        assert not list((tmp_path / "control").rglob(MODEL))
        assert not [path for path in (tmp_path / "control").rglob("lfs/objects/*/*/*") if path.is_file()]
        # A commit discards the content that it took.
        for name in ("job-1:1", "job-2:0"):
            with pytest.raises(ArtifactNotFoundError):
                training.materialize_content(name)

        # Another service reads both components of the committed snapshot from the store.
        served = git_backends(tmp_path, "inference")("agent").materialize(second.snapshot)
        assert served.local_path is not None
        assert (served.local_path / WEIGHTS / MODEL).read_bytes() == weights
        assert (served.local_path / WEIGHTS / "adapter_config.json").read_text() == '{"r": 8}'
        assert (served.local_path / HARNESS / "AGENTS.md").read_text() == "evolved"


@pytest.mark.unit
def test_a_commit_that_fails_after_the_snapshot_is_stored_commits_once_also_after_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts")
    backend = backends("agent")
    backend.store_content("job-1:1", checkpoint(tmp_path / "one", {"weights.txt": b"w1"}))
    backend.store_content("job-2:1", checkpoint(tmp_path / "two", {"weights.txt": b"w2"}))

    def crash(**arguments):
        raise OSError("control stopped before the record")

    with control(tmp_path, WeightsRecipe(), backends) as scenario:
        commits = scenario.commit_number
        append_commit = scenario.store.append_commit
        # The backend stored the snapshot, and the record did not land.
        result = stored_commit(scenario, "job-1:1", WEIGHTS)
        monkeypatch.setattr(scenario.store, "append_commit", crash)
        with pytest.raises(OSError, match="before the record"):
            scenario.commit(result)
        assert scenario.commit_number == commits
        stored = backend.artifact_snapshot(result.artifact.artifact_id)
        # The content stays until a record names its snapshot, so the same result commits that snapshot.
        assert backend.materialize_content("job-1:1").is_dir()
        monkeypatch.setattr(scenario.store, "append_commit", append_commit)
        scenario.commit(result)
        assert scenario.store.history()[-1].snapshot == stored
        with pytest.raises(ArtifactNotFoundError):
            backend.materialize_content("job-1:1")

        # A restart loses the pending result: the trainer polls the job again and commits the content anew.
        monkeypatch.setattr(scenario.store, "append_commit", crash)
        with pytest.raises(OSError, match="before the record"):
            scenario.commit(stored_commit(scenario, "job-2:1", WEIGHTS))
        monkeypatch.undo()
    with control(tmp_path, WeightsRecipe(), backends) as restarted:
        assert restarted.commit_number == commits + 1
        restarted.commit(stored_commit(restarted, "job-2:1", WEIGHTS))
        assert restarted.commit_number == commits + 2
        committed = restarted.repository.materialize(restarted.store.history()[-1].snapshot)
        assert (committed.local_path / "weights.txt").read_text() == "w2"
        with pytest.raises(ArtifactNotFoundError):
            backend.materialize_content("job-2:1")


@pytest.mark.unit
def test_an_admission_check_reads_stored_content_and_a_refused_checkpoint_changes_nothing(tmp_path: Path) -> None:
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts")
    backend = backends("agent")
    backend.store_content("bad:0", checkpoint(tmp_path / "bad", {"AGENTS.md": b"unvetted"}))
    backend.store_content("good:0", checkpoint(tmp_path / "good", {"AGENTS.md": b"vetted"}))
    with control(tmp_path, two_trainer_recipe(harness=checked_harness()), backends) as scenario:
        seed, commits = scenario.current_snapshot(), scenario.commit_number
        # The component's check reads the content, which is made local only for it.
        with pytest.raises(ValueError, match="validator refused the artifact"):
            scenario.commit(stored_commit(scenario, "bad:0", HARNESS), component=HARNESS)
        assert (scenario.current_snapshot(), scenario.commit_number) == (seed, commits)
        assert not scenario.repository.stored_trees
        scenario.commit(stored_commit(scenario, "good:0", HARNESS), component=HARNESS)
        committed = scenario.repository.materialize(scenario.store.history()[-1].snapshot)
        assert (committed.local_path / HARNESS / "AGENTS.md").read_text() == "vetted"
        assert (committed.local_path / WEIGHTS / "weights.txt").read_text() == "weights seed"
        # A commit of content that the store does not hold fails, and the scenario stays.
        with pytest.raises(ArtifactNotFoundError):
            scenario.commit(stored_commit(scenario, "missing:0", HARNESS), component=HARNESS)
        assert scenario.commit_number == commits + 1
