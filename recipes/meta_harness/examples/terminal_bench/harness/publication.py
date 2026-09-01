"""Publish one selected provider-free composition through Reef artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from reef.artifact import Artifact, GitLFSRepositoryBackend
from reef.harness import AdapterDescriptor

from .composition import CompositionCandidate


def publish_candidate(
    candidate: CompositionCandidate,
    *,
    descriptor: AdapterDescriptor,
    output_dir: Path,
    scenario: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "publication.json"
    files = candidate.render(descriptor)
    render_sha256 = _mapping_sha256(files)
    _ensure_tree(output_dir / "published-composition", files)
    if manifest_path.is_file():
        manifest = _read_manifest(manifest_path)
        if (
            manifest.get("candidate_hash") != candidate.content_hash
            or manifest.get("render_sha256") != render_sha256
            or manifest.get("scenario") != scenario
            or manifest.get("files") != sorted(files)
        ):
            raise RuntimeError("published candidate does not match resumed selection")
        return manifest

    tree_dir = output_dir / "published-composition"
    repository_path = output_dir / "artifacts.git"
    backend = GitLFSRepositoryBackend(
        scenario,
        repository_path,
        work_dir=output_dir / "artifact-work",
        cache_dir=output_dir / "artifact-cache",
    )
    publication_identity = {
        "method": "meta-harness",
        "scenario": scenario,
        "candidate_hash": candidate.content_hash,
        "render_sha256": render_sha256,
    }
    existing_metadata = backend.metadata()
    if existing_metadata is None:
        parent = backend.fork(metadata={**publication_identity, "publication_state": "pending"})
    else:
        if any(existing_metadata.get(key) != value for key, value in publication_identity.items()):
            raise RuntimeError("existing artifact publication does not match the selected candidate")
        state = existing_metadata.get("publication_state")
        if state == "published":
            published = backend.current()
            manifest = _manifest_payload(
                candidate.content_hash,
                render_sha256,
                scenario,
                published.content_id,
                published.release_id,
                published.parent_release_id,
                repository_path,
                files,
            )
            _write_json_atomic(manifest_path, manifest)
            return manifest
        if state != "pending":
            raise RuntimeError(f"unknown artifact publication state {state!r}")
        parent = backend.current()
    published = backend.publish(
        Artifact.local(
            tree_dir,
            metadata={**dict(metadata), **publication_identity, "publication_state": "published"},
        ),
        expected_parent=parent,
    )
    manifest = _manifest_payload(
        candidate.content_hash,
        render_sha256,
        scenario,
        published.content_id,
        published.release_id,
        published.parent_release_id,
        repository_path,
        files,
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _ensure_tree(root: Path, files: Mapping[str, str]) -> None:
    if root.exists():
        observed = {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in root.rglob("*")
            if path.is_file()
        }
        if observed != dict(files):
            raise RuntimeError("existing published composition does not match the selected candidate")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        _write_tree(temporary, files)
        try:
            temporary.rename(root)
        except OSError:
            if not root.is_dir():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    _ensure_tree(root, files)


def _mapping_sha256(value: Mapping[str, str]) -> str:
    serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _manifest_payload(
    candidate_hash: str,
    render_sha256: str,
    scenario: str,
    content_id: str,
    release_id: str,
    parent_release_id: str | None,
    repository_path: Path,
    files: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "candidate_hash": candidate_hash,
        "render_sha256": render_sha256,
        "scenario": scenario,
        "content_id": content_id,
        "release_id": release_id,
        "parent_release_id": parent_release_id,
        "repository": repository_path.name,
        "files": sorted(files),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid publication manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid publication manifest: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for relative, text in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"published render path {relative!r} escapes its root")
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
