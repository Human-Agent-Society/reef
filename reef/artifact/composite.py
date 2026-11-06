"""Lay out several component artifacts as one local multi-component release."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from reef.artifact.artifact import LOCAL_SNAPSHOT_PREFIX, Artifact, ArtifactPublicationError, SnapshotRef
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, SnapshotComponents


def link_or_copy(source: str, destination: str) -> None:
    """Hard-link an immutable released file into another tree; copy when the filesystem refuses.

    A symlink in the source resolves first: a plain link on Linux would link
    the symlink itself, and the tree would then carry a link into the cache.
    """
    try:
        os.link(os.path.realpath(source), destination)
    except OSError:
        shutil.copy2(source, destination)


def compose_snapshot(components: Mapping[str, Artifact], *, directory: Path) -> Artifact:
    """Place each component's files under ``directory/<name>`` and describe them in a manifest.

    Stored files are immutable, so a carried-forward component is
    hard-linked rather than copied where the filesystem allows; a component
    whose source has no directory yet (a default start that seeded nothing for
    it) composes as an empty directory. The result is a process-local
    view of those components, with no content identity of its own.
    Every component keeps its independently allocated ``artifact_id``, including
    an unpublished component. Each
    component keeps its own metadata in the manifest; a component's
    release-level manifest key, if it was itself a flat snapshot, is not nested.
    """
    if len(components) < 2:
        raise ArtifactPublicationError("a composed snapshot binds at least two components")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, ComponentEntry] = {}
    for name, component in components.items():
        source = component.materialize()
        if source.local_path is None:
            raise ArtifactPublicationError(f"component {name!r} has no local content to compose")
        metadata = {key: value for key, value in source.metadata.items() if key != COMPONENTS_METADATA_KEY}
        # Composition changes the layout, never the identity of its components.
        carried = source.artifact_id
        entries[name] = ComponentEntry(metadata=metadata, artifact_id=carried)
        try:
            if source.local_path.is_dir():
                shutil.copytree(source.local_path, directory / name, copy_function=link_or_copy)
            else:
                (directory / name).mkdir()
        except OSError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ArtifactPublicationError(
                f"failed to compose component {name!r} from {source.local_path}: {exc}"
            ) from exc
    manifest = SnapshotComponents(entries)
    return Artifact(
        SnapshotRef(revision=f"{LOCAL_SNAPSHOT_PREFIX}{uuid.uuid4().hex}", parent_revision=None),
        None,
        local_path=directory,
        metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()},
    )


__all__ = ["compose_snapshot", "link_or_copy"]
