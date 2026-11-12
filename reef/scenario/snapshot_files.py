"""The scenario's committed snapshots as files: the head that a harness step builds on.

``ScenarioArtifactFiles`` reads the commit history and the artifact
repository of one scenario. It is what ``reef.train.backend.CommittedArtifacts``
names: the harness backend reads its head tree through it, and the scenario
factory hands it to the recipe when it builds the trainers.
"""

from __future__ import annotations

from reef.artifact.artifact import Artifact, ArtifactNotFound, SnapshotRef
from reef.artifact.components import ScenarioComponents, TextFileTree
from reef.artifact.repository import Repository
from reef.storage.commits import CommitRecord
from reef.storage.scenario import ScenarioStore
from reef.train.backend import ArtifactTextFiles, CommittedArtifacts


class ScenarioArtifactFiles(CommittedArtifacts):
    """One scenario's snapshots, read through its commit history and artifact repository.

    A component's head is its training start, as the scenario committer keeps
    it: the snapshot of the newest ``"train_from"`` or ``"sync"`` record for
    the component or of the newest training commit that made an artifact for
    it, whichever is later. An activation moves what serves, not the head, so a step builds on
    the latest tree it made even when the activation rule held that tree.
    Before any such record, and without a durable history, the head is the
    repository's current snapshot: the snapshot the scenario was created on, as
    the catalog lists it.
    """

    def __init__(
        self, *, name: str, store: ScenarioStore, repository: Repository, components: ScenarioComponents
    ) -> None:
        self.name = name
        self.store = store
        self.repository = repository
        self.components = components

    def head(self, component: str) -> ArtifactTextFiles:
        history = self.store.history() if self.store.durable else ()
        records = [record for record in history if moves_head(record, component)]
        ref = records[-1].snapshot if records else self.repository.require_current_artifact()
        return self.files(ref, component)

    def artifact(self, artifact_id: str, component: str) -> ArtifactTextFiles:
        refs = [self.repository.start_artifact, self.repository.require_current_artifact()]
        refs.extend(record.snapshot for record in self.store.history())
        for ref in dict.fromkeys(refs):
            artifact = self.repository.resolve(ref)
            selected = self.components.component_artifact(artifact, component)
            if selected.artifact_id == artifact_id:
                return ArtifactTextFiles(artifact_id, dict(TextFileTree().read_files(selected) or {}))
        raise ArtifactNotFound(f"scenario {self.name!r} has no artifact {artifact_id!r} of component {component!r}")

    def files(self, ref: SnapshotRef, component: str) -> ArtifactTextFiles:
        """Return the selected component's ID and text files from a stored snapshot."""
        artifact = self.repository.resolve(ref)
        selected = self.components.component_artifact(artifact, component)
        return ArtifactTextFiles(selected.artifact_id, dict(TextFileTree().read_files(selected) or {}))


def component_files(components: ScenarioComponents, artifact: Artifact, component: str) -> dict[str, str]:
    """``component``'s text files in ``artifact``, as the snapshot stores them: the harness source tree."""
    return dict(TextFileTree().read_files(components.component_artifact(artifact, component)) or {})


def moves_head(record: CommitRecord, component: str) -> bool:
    """Whether ``record`` makes its snapshot ``component``'s head: a ``"train_from"`` or ``"sync"`` record for it, or
    a training commit that made an artifact for it (``CommitRecord.sets_training_start``)."""
    if record.component not in (None, component):
        return False
    return record.sets_training_start


__all__ = ["ScenarioArtifactFiles", "component_files", "moves_head"]
