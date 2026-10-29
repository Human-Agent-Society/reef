"""The scenario's committed releases as files, for a candidate backend that builds on its head.

``ScenarioReleaseFiles`` reads the commit history and the artifact
repository of one scenario. It is what ``reef.train.backend.CommittedReleases``
names: the harness backend reads its head tree through it, and the scenario
factory hands it to the recipe when it builds the trainers.
"""

from __future__ import annotations

from reef.artifact.artifact import Artifact, ArtifactNotFound, ArtifactRef
from reef.artifact.repository import Repository
from reef.storage.commits import CommitRecord
from reef.storage.scenario import ScenarioStore
from reef.surface.base import Surface
from reef.surface.files import TextFileTree
from reef.train.backend import CommittedReleases, ReleaseFiles


class ScenarioReleaseFiles(CommittedReleases):
    """One scenario's releases, read through its commit history and artifact repository.

    A component's head is its training start, as the scenario committer keeps
    it: the release of the newest ``"train_from"`` record for the component or
    of the newest training commit that made an artifact for it, whichever is
    later. An activation moves what serves, not the head, so a step builds on
    the latest tree it made even when the activation rule held that tree.
    Before any such record, and without a durable history, the head is the
    repository's current release: the release the scenario was created on, as
    the catalog lists it.
    """

    def __init__(self, *, name: str, store: ScenarioStore, repository: Repository, surface: Surface) -> None:
        self.name = name
        self.store = store
        self.repository = repository
        self.surface = surface

    def head(self, component: str) -> ReleaseFiles:
        history = self.store.history() if self.store.durable else ()
        records = [record for record in history if moves_head(record, component)]
        ref = records[-1].artifact_ref if records else self.repository.require_current_artifact()
        return self.files(ref, component)

    def release(self, release_id: str, component: str) -> ReleaseFiles:
        if release_id == self.repository.base_artifact.release_id:
            return self.files(self.repository.base_artifact, component)
        current = self.repository.require_current_artifact()
        if release_id == current.release_id:
            return self.files(current, component)
        for record in self.store.history():
            if record.artifact_ref.release_id == release_id:
                return self.files(record.artifact_ref, component)
        raise ArtifactNotFound(f"scenario {self.name!r} has no release {release_id!r}")

    def files(self, ref: ArtifactRef, component: str) -> ReleaseFiles:
        """``component``'s text files in the release ``ref`` names."""
        return ReleaseFiles(ref.release_id, component_files(self.surface, self.repository.resolve(ref), component))


def component_files(surface: Surface, artifact: Artifact, component: str) -> dict[str, str]:
    """``component``'s text files in ``artifact``, as the release stores them: the harness source tree."""
    return dict(TextFileTree().read_files(surface.component_artifact(artifact, component)) or {})


def moves_head(record: CommitRecord, component: str) -> bool:
    """Whether ``record`` makes its release ``component``'s head: a ``"train_from"`` record for it, or a training
    commit that made an artifact for it."""
    if record.component not in (None, component):
        return False
    return record.operation == "train_from" or record.trained_artifact


__all__ = ["ScenarioReleaseFiles", "component_files", "moves_head"]
