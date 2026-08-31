from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from reef.artifact.artifact import ArtifactSourceError


@dataclass(frozen=True)
class HuggingFaceSource:
    model_name: str
    version: str | None = None


@dataclass(frozen=True)
class GitVersionSource:
    version: str


ArtifactSource = HuggingFaceSource | GitVersionSource


@dataclass(frozen=True)
class DownloadedSnapshot:
    local_path: Path
    version: str


def parse_artifact_source(value: str) -> ArtifactSource:
    source = value.strip()
    if not source:
        raise ArtifactSourceError("artifact source must be non-empty")
    if source.startswith("git+lfs://"):
        lfs_version = source.removeprefix("git+lfs://")
        if not lfs_version:
            raise ArtifactSourceError("Git LFS source version must be non-empty")
        return GitVersionSource(lfs_version)
    model = source.removeprefix("hf://")
    version: str | None = None
    if "@" in model:
        model, version = model.rsplit("@", 1)
        if not version:
            raise ArtifactSourceError("Hugging Face version must be non-empty")
    if not model or "/" not in model:
        raise ArtifactSourceError("Hugging Face source must use org/model syntax")
    return HuggingFaceSource(model, version)


def download_huggingface_snapshot(
    source: HuggingFaceSource,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> DownloadedSnapshot:
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download

        snapshot_download = cast(Callable[..., str], huggingface_snapshot_download)
    try:
        if source.version is None:
            downloaded_path = snapshot_download(repo_id=source.model_name)
        else:
            # "revision" is Hugging Face's keyword for what Reef calls a version.
            downloaded_path = snapshot_download(repo_id=source.model_name, revision=source.version)
        local_path = Path(downloaded_path).resolve()
    except Exception as exc:
        raise ArtifactSourceError(f"failed to download Hugging Face artifact {source.model_name}: {exc}") from exc
    if not local_path.is_dir():
        raise ArtifactSourceError(f"Hugging Face snapshot is not a directory: {local_path}")
    try:
        version = local_path.name if local_path.parent.name == "snapshots" else ""
    except OSError:
        version = ""
    if not version:
        raise ArtifactSourceError(f"cannot determine immutable Hugging Face version from snapshot path: {local_path}")
    return DownloadedSnapshot(local_path=local_path, version=version)
