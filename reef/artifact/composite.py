"""Lay out several component artifacts as one local multi-component release."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from reef.artifact.artifact import LOCAL_RELEASE_PREFIX, Artifact, ArtifactPublicationError, ArtifactRef
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents


def _link_or_copy(source: str, destination: str) -> None:
    """Hard-link an immutable released file into the composed tree; copy when the filesystem refuses."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def compose_release(components: Mapping[str, Artifact], *, directory: Path) -> Artifact:
    """Place each component's files under ``directory/<name>`` and describe them in a manifest.

    Released files are immutable, so a carried-forward component is
    hard-linked rather than copied where the filesystem allows; a component
    whose source has no directory yet (a base release that seeded nothing for
    it) composes as an empty directory. The result is a process-local
    artifact whose ``content_id`` derives from the component content ids, so
    republishing the same combination keeps one content identity. Each
    component keeps its own metadata in the manifest; a component's
    release-level manifest key, if it was itself a flat release, is not nested.
    """
    if len(components) < 2:
        raise ArtifactPublicationError("a composed release binds at least two components")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, ComponentEntry] = {}
    for name, component in components.items():
        source = component.materialize()
        if source.local_path is None:
            raise ArtifactPublicationError(f"component {name!r} has no local content to compose")
        metadata = {key: value for key, value in source.metadata.items() if key != COMPONENTS_METADATA_KEY}
        entries[name] = ComponentEntry(content_id=source.ref.content_id, metadata=metadata)
        try:
            if source.local_path.is_dir():
                shutil.copytree(source.local_path, directory / name, copy_function=_link_or_copy)
            else:
                (directory / name).mkdir()
        except OSError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ArtifactPublicationError(
                f"failed to compose component {name!r} from {source.local_path}: {exc}"
            ) from exc
    manifest = ReleaseComponents(entries)
    return Artifact(
        ArtifactRef(
            content_id=manifest.content_id,
            release_id=f"{LOCAL_RELEASE_PREFIX}{uuid.uuid4().hex}",
            parent_release_id=None,
        ),
        None,
        local_path=directory,
        metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()},
    )


__all__ = ["compose_release"]
