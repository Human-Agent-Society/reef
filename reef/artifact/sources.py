from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self, cast

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

from reef.artifact.artifact import REPOSITORY_FILES, ArtifactSourceError
from reef.artifact.composite import link_or_copy
from reef.artifact.files import ArtifactFiles, path_order
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


#: The one file of a reference artifact: the Hugging Face revision whose files are the artifact's content.
REFERENCE_FILE = "hf-reference.json"
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class HuggingFaceRevision:
    """One commit of a Hugging Face model repo (RFC #995).

    Values: ``repo`` is ``org/name`` with both parts non-empty and no other ``/``; ``commit`` is a commit SHA of
    40 lowercase hex digits. A commit names fixed bytes, so the files of a value never change while the repo exists.
    Purpose: a reference import writes one in ``REFERENCE_FILE`` as ``{"uri": "hf://<repo>", "version": "<commit>"}``
    (``to_dict``); a consumer that needs the files reads it back with ``read_reference``.
    """

    repo: str
    commit: str

    def __post_init__(self) -> None:
        owner, separator, name = self.repo.partition("/")
        if not separator or not owner or not name or "/" in name:
            raise ValueError(f"a Hugging Face repo is org/name, got {self.repo!r}")
        if not COMMIT_SHA.fullmatch(self.commit):
            raise ValueError(f"a Hugging Face revision is a 40-digit commit SHA, got {self.commit!r}")

    @property
    def uri(self) -> str:
        return f"hf://{self.repo}@{self.commit}"

    def to_dict(self) -> dict[str, str]:
        return {"uri": f"hf://{self.repo}", "version": self.commit}

    @classmethod
    def from_dict(cls, value: object) -> HuggingFaceRevision:
        """Read a ``to_dict`` value; ``ValueError`` for another shape."""
        if not isinstance(value, Mapping) or set(value) != {"uri", "version"}:
            raise ValueError(f"{REFERENCE_FILE} is the object {{uri, version}}")
        uri, version = value["uri"], value["version"]
        if not isinstance(uri, str) or not uri.startswith("hf://") or not isinstance(version, str):
            raise ValueError(f"{REFERENCE_FILE} names an hf:// repo and its commit")
        return cls(uri.removeprefix("hf://"), version)


def read_reference(directory: Path) -> HuggingFaceRevision | None:
    """The revision that the artifact in ``directory`` refers to; ``None`` for an artifact that holds its files.

    An artifact is a reference when ``REFERENCE_FILE`` is its only entry besides the files that a repository adds to
    a materialized release (``REPOSITORY_FILES``). Raise ``ValueError`` when that file is not a
    ``HuggingFaceRevision.to_dict`` value.
    """
    if not directory.is_dir() or {entry.name for entry in directory.iterdir()} - REPOSITORY_FILES != {REFERENCE_FILE}:
        return None
    try:
        value = json.loads((directory / REFERENCE_FILE).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{directory / REFERENCE_FILE} is no JSON object: {exc}") from exc
    return HuggingFaceRevision.from_dict(value)


class HuggingFaceRevisionFiles(ArtifactFiles):
    """The files of ``revision`` on the Hugging Face Hub, read without downloading its weights.

    One listing gives ``paths`` and the SHA-256 that the Hub keeps for each LFS file. The SHA-256 of another file,
    which is below the Hub's LFS size, and ``read_bytes`` of any file download that one file into Hugging Face's
    cache. The token is Hugging Face's own: ``HF_TOKEN`` or the cached login. Raise ``ValueError`` when the revision
    does not resolve for that token, and ``ArtifactSourceError`` when the Hub fails.
    """

    def __init__(self, revision: HuggingFaceRevision, api: HfApi | None = None) -> None:
        self.revision = revision
        self.api = HfApi() if api is None else api
        try:
            info = self.api.model_info(revision.repo, revision=revision.commit, files_metadata=True)
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise ValueError(f"{revision.uri} names no revision that the service can read: {exc}") from exc
        except (HfHubHTTPError, OSError) as exc:
            raise ArtifactSourceError(f"failed to list {revision.uri}: {exc}") from exc
        self.lfs_sha256 = {
            sibling.rfilename: sibling.lfs.sha256 for sibling in info.siblings or () if sibling.lfs is not None
        }
        self.listed = tuple(sorted((sibling.rfilename for sibling in info.siblings or ()), key=path_order))

    def paths(self) -> tuple[str, ...]:
        return self.listed

    def sha256(self, path: str) -> str:
        if path in self.lfs_sha256:
            return self.lfs_sha256[path]
        with self.download(path).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def read_bytes(self, path: str) -> bytes:
        return self.download(path).read_bytes()

    def download(self, path: str) -> Path:
        """The cached copy of the file at ``path``; ``FileNotFoundError`` unless ``path`` is one of ``paths()``."""
        if path not in self.listed:
            raise FileNotFoundError(f"{self.revision.uri} holds no file {path}")
        try:
            # Without dry_run, the download gives the cached file's path.
            return Path(cast(str, self.api.hf_hub_download(self.revision.repo, path, revision=self.revision.commit)))
        except (HfHubHTTPError, OSError) as exc:
            raise ArtifactSourceError(f"failed to read {path} of {self.revision.uri}: {exc}") from exc


@dataclass(frozen=True)
class ReferencedContent:
    """What a reader keeps by reference: the content's ``files`` where they are stored, and the ``version`` read."""

    files: ArtifactFiles
    version: str


class ArtifactSourceReader(ConfiguredPlugin, ABC):
    """Reads the directory that a URI names, for an artifact import (RFC #980).

    Values: ``schemes`` is a non-empty tuple of URI schemes, such as ``s3``, that the reader fetches.
    ``fetch(uri, directory)`` writes the files that ``uri`` names into the empty directory ``directory`` and
    returns the version it read: a non-empty name of what the source served, or an empty string when the source
    has none. It raises ``ValueError`` for a URI that names nothing it can read, such as a missing directory, and
    ``ArtifactSourceError`` when the source fails. The version is recorded for people to trace an import; the
    content is what ``directory`` holds afterwards.

    ``keep_by_reference(uri, directory)`` is ``None`` when the reader copies ``uri``, the default. A reader that keeps
    ``uri`` by reference (RFC #995) writes the files that Reef stores for it into the empty ``directory`` and
    returns the referenced content, which stays where it is; it raises as ``fetch`` does.

    Purpose: a deployment names extra readers in ``storage.artifact_sources``; ``ArtifactSources`` dispatches a
    URI to the reader of its scheme.
    """

    @property
    @abstractmethod
    def schemes(self) -> tuple[str, ...]: ...

    @abstractmethod
    def fetch(self, uri: str, directory: Path) -> str: ...

    def keep_by_reference(self, uri: str, directory: Path) -> ReferencedContent | None:
        return None


class HuggingFaceReader(ArtifactSourceReader):
    """Reads ``hf://org/model[@revision]``: the snapshot of that revision, whose commit is the version.

    A URI whose revision is a full commit SHA is kept by reference: Reef stores ``REFERENCE_FILE``, and the files
    stay on the Hub (RFC #995). Another revision, a branch or a tag, is copied.
    """

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("hf",)

    def fetch(self, uri: str, directory: Path) -> str:
        snapshot = download_huggingface_snapshot(parse_huggingface_source(uri))
        copy_tree(snapshot.local_path, directory)
        return snapshot.version

    def keep_by_reference(self, uri: str, directory: Path) -> ReferencedContent | None:
        source = parse_huggingface_source(uri)
        if source.version is None or not COMMIT_SHA.fullmatch(source.version):
            return None
        revision = HuggingFaceRevision(source.model_name, source.version)
        files = HuggingFaceRevisionFiles(revision)
        (directory / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()) + "\n", encoding="utf-8")
        return ReferencedContent(files, revision.commit)


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
    ``reader(uri)`` is the reader of ``uri``'s scheme. Purpose: the service builds it from
    ``storage.artifact_sources`` (``from_setting``) and reads an import's URI with the reader it gives.
    """

    readers: tuple[ArtifactSourceReader, ...]

    def __post_init__(self) -> None:
        schemes = [scheme for reader in self.readers for scheme in reader.schemes]
        if len(schemes) != len(set(schemes)):
            raise ValueError(f"two artifact source readers read one scheme: {sorted(schemes)}")

    def reader(self, uri: str) -> ArtifactSourceReader:
        """The reader of ``uri``'s scheme; ``ValueError`` for a URI with no scheme or a scheme no reader reads."""
        scheme, separator, _ = uri.partition("://")
        reader = next((reader for reader in self.readers if scheme in reader.schemes), None) if separator else None
        if reader is None:
            known = sorted(scheme for reader in self.readers for scheme in reader.schemes)
            raise ValueError(f"{uri!r} is no URI of a known scheme; the readers read {known}")
        return reader

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
