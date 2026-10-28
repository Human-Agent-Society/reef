from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self, cast

from reef.artifact.artifact import ArtifactSourceError
from reef.artifact.composite import link_or_copy
from reef.core.config import config_option
from reef.core.plugins import ConfiguredPlugin, PluginSettings


@dataclass(frozen=True)
class HuggingFaceSource:
    model_name: str
    version: str | None = None


@dataclass(frozen=True)
class DownloadedSnapshot:
    local_path: Path
    version: str


def parse_huggingface_source(value: str) -> HuggingFaceSource:
    """The Hugging Face model ``value`` names: ``org/model``, optionally after ``hf://`` and before ``@<version>``.

    Raise ``ValueError`` for a value of another form.
    """
    model = value.strip().removeprefix("hf://")
    version: str | None = None
    if "@" in model:
        model, version = model.rsplit("@", 1)
        if not version:
            raise ValueError("Hugging Face version must be non-empty")
    if not model or "/" not in model:
        raise ValueError("Hugging Face source must use org/model syntax")
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


class ArtifactSourceReader(ConfiguredPlugin, ABC):
    """Reads the directory that a URI names, for an artifact import (RFC #980).

    Values: ``schemes`` is a non-empty tuple of URI schemes, such as ``s3``, that the reader fetches.
    ``fetch(uri, directory)`` writes the files that ``uri`` names into the empty directory ``directory`` and
    returns the version it read: a non-empty name of what the source served, or an empty string when the source
    has none. It raises ``ValueError`` for a URI that names nothing it can read, such as a missing directory, and
    ``ArtifactSourceError`` when the source fails. The version is recorded for people to trace an import; the
    content is what ``directory`` holds afterwards.

    Purpose: a deployment names extra readers in ``storage.artifact_sources``; ``ArtifactSources`` dispatches a
    URI to the reader of its scheme.
    """

    @property
    @abstractmethod
    def schemes(self) -> tuple[str, ...]: ...

    @abstractmethod
    def fetch(self, uri: str, directory: Path) -> str: ...


class HuggingFaceReader(ArtifactSourceReader):
    """Reads ``hf://org/model[@revision]``: the snapshot of that revision, whose commit is the version."""

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("hf",)

    def fetch(self, uri: str, directory: Path) -> str:
        snapshot = download_huggingface_snapshot(parse_huggingface_source(uri))
        copy_tree(snapshot.local_path, directory)
        return snapshot.version


class FileReader(ArtifactSourceReader):
    """Reads ``file:///path``: a directory on the service's file system, which has no version."""

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("file",)

    def fetch(self, uri: str, directory: Path) -> str:
        source = Path(uri.removeprefix("file://"))
        if not source.is_absolute() or not source.is_dir():
            raise ValueError(f"{uri} does not name an absolute directory path")
        copy_tree(source, directory)
        return ""


@dataclass(frozen=True)
class S3ReaderSettings(PluginSettings):
    """The config of ``S3Reader``."""

    endpoint_url: str | None = config_option(None, help="An S3-compatible endpoint; unset is AWS S3.")


class S3Reader(ArtifactSourceReader):
    """Reads ``s3://bucket/prefix/``: every object under the prefix, at its path below it.

    The credentials are boto3's standard chain: the environment, a profile or an instance role. The version is
    the SHA-256 of the objects' keys and ETags, so the same objects read again give the same version.
    """

    def __init__(self, endpoint_url: str | None = None) -> None:
        self.endpoint_url = endpoint_url

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(S3ReaderSettings.parse(config, plugin_class=cls).endpoint_url)

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("s3",)

    def fetch(self, uri: str, directory: Path) -> str:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise ArtifactSourceError("reading s3:// sources needs boto3: uv pip install 'reef-infra[s3]'") from exc
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        if not bucket:
            raise ValueError(f"{uri} names no bucket")
        prefix = prefix if not prefix or prefix.endswith("/") else f"{prefix}/"
        client = boto3.client("s3", endpoint_url=self.endpoint_url)
        listed: list[tuple[str, str]] = []
        try:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", ()):
                    relative = PurePosixPath(item["Key"][len(prefix) :])
                    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                        raise ArtifactSourceError(f"{uri} holds an object outside its prefix: {item['Key']}")
                    target = directory / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, item["Key"], str(target))
                    listed.append((str(relative), str(item["ETag"])))
        except (BotoCoreError, ClientError) as exc:
            raise ArtifactSourceError(f"failed to read {uri}: {exc}") from exc
        if not listed:
            raise ValueError(f"{uri} holds no object")
        return hashlib.sha256(json.dumps(sorted(listed)).encode()).hexdigest()


@dataclass(frozen=True)
class ArtifactSources:
    """The source readers of a deployment, by URI scheme.

    Values: ``readers`` is a tuple of ``ArtifactSourceReader`` values, no two of which read one scheme.
    ``fetch(uri, directory)`` is the fetch of the reader of ``uri``'s scheme; ``ValueError`` for a URI with no
    scheme or a scheme no reader reads. Purpose: the service builds it from ``storage.artifact_sources``
    (``from_setting``) and reads an import's URI with it.
    """

    readers: tuple[ArtifactSourceReader, ...]

    def __post_init__(self) -> None:
        schemes = [scheme for reader in self.readers for scheme in reader.schemes]
        if len(schemes) != len(set(schemes)):
            raise ValueError(f"two artifact source readers read one scheme: {sorted(schemes)}")

    def fetch(self, uri: str, directory: Path) -> str:
        scheme, separator, _ = uri.partition("://")
        reader = next((reader for reader in self.readers if scheme in reader.schemes), None) if separator else None
        if reader is None:
            known = sorted(scheme for reader in self.readers for scheme in reader.schemes)
            raise ValueError(f"{uri!r} is no URI of a known scheme; the readers read {known}")
        return reader.fetch(uri, directory)

    @classmethod
    def from_setting(cls, value: Sequence[object]) -> ArtifactSources:
        """The built-in readers (``hf``, ``file``, ``s3``) and the readers ``value`` names.

        A configured reader replaces the built-in reader of a scheme it reads.
        """
        configured = tuple(
            ArtifactSourceReader.from_setting(item, setting="storage.artifact_sources") for item in value
        )
        taken = {scheme for reader in configured for scheme in reader.schemes}
        built_in = tuple(
            reader
            for reader in (HuggingFaceReader(), FileReader(), S3Reader())
            if not taken.intersection(reader.schemes)
        )
        return cls((*built_in, *configured))


def copy_tree(source: Path, directory: Path) -> None:
    """Link or copy every file under ``source`` into ``directory`` at the same path; a source file is never changed."""
    for path in sorted(source.rglob("*")):
        if path.is_file():
            target = directory / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(str(path), str(target))


def directory_content_id(directory: Path) -> str:
    """The content ID of the files under ``directory``: the SHA-256 of each file's relative path and SHA-256.

    Equal files at equal paths give equal IDs, wherever they were read from. Purpose: an artifact import names its
    content with it (RFC #980).
    """
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as file:
                files.append((path.relative_to(directory).as_posix(), hashlib.file_digest(file, "sha256").hexdigest()))
    return f"content:{hashlib.sha256(json.dumps(files).encode()).hexdigest()}"
