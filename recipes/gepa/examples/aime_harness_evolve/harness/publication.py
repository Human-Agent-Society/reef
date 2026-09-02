"""Publish a selected provider-free composition as a durable Reef release."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact import Artifact, GitLFSRepositoryBackend

from .adapter import ReefAdapter
from .files import read_json, write_json


def publish_candidate(
    *,
    adapter: ReefAdapter,
    candidate: Mapping[str, str],
    output_dir: Path,
    scenario: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the candidate without its model binding and publish it through Reef's Git-LFS backend.

    A cell that already holds ``publication.json`` returns it. A run that was
    interrupted after the fork resumes from the scenario ref the backend kept.
    """
    manifest_path = output_dir / "publication.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    files = adapter.render_candidate(candidate)
    tree = output_dir / "published-composition"
    if tree.exists():
        shutil.rmtree(tree)
    for relative, text in files.items():
        path = tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    repository = output_dir / "artifacts.git"
    backend = GitLFSRepositoryBackend(
        scenario,
        repository,
        work_dir=output_dir / "artifact-work",
        cache_dir=output_dir / "artifact-cache",
    )
    identity = {"method": "gepa", "scenario": scenario}
    parent = backend.fork(metadata=identity)
    published = backend.publish(Artifact.local(tree, metadata={**metadata, **identity}), expected_parent=parent)
    manifest = {
        "content_id": published.content_id,
        "release_id": published.release_id,
        "parent_release_id": published.parent_release_id,
        "repository": str(repository),
        "files": sorted(files),
        "scenario": scenario,
    }
    write_json(manifest_path, manifest)
    return manifest
