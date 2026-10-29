"""The sources of artifact imports: how a URI becomes files that Reef checks, copies or refers to (RFC #980, #995).

A source reader opens a URI as ``SourceFiles``, the files where the source keeps them: a directory on the service
(``DirectorySource``), an S3 prefix (``S3PrefixFiles``) or a Hugging Face revision (``HuggingFaceRevisionFiles``).
An import checks the component's layout on them before it copies anything, then stores a copy, or a reference to a
revision that cannot change (``REFERENCE_FILE``, which ``read_reference`` reads back).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self, cast
from urllib.parse import unquote

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

from reef.artifact.artifact import REPOSITORY_FILES, ArtifactSourceError
from reef.artifact.composite import link_or_copy
from reef.artifact.files import ArtifactFiles, DirectoryFiles, path_order
from reef.core.config import config_option
from reef.core.plugins import ConfiguredPlugin, PluginSettings

#: The one file of a reference artifact: the Hugging Face revision whose files are the artifact's content.
REFERENCE_FILE = "hf-reference.json"
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class SourceFiles(ArtifactFiles):
    """The files that an import's URI names, where the source keeps them.

    Values: ``ArtifactFiles`` of the source's directory ``D``, and three more members. ``version`` is a non-empty
    name of what the source served, or ``""`` when the source has none; it is recorded for people to trace an
    import. ``reference`` is ``None`` when Reef stores a copy of ``D``; otherwise it maps each path of the files that
    Reef stores instead to their bytes, for a source that keeps ``D`` fixed. ``copy_into(directory)`` writes ``D``
    into the empty ``directory``, so that ``directory_content_id(directory) == content_id()``; it raises
    ``ArtifactSourceError`` when the source fails.

    Purpose: an ``ArtifactSourceReader`` opens one for a URI. The import checks the component's layout on it, which
    reads only the files the check needs, and then stores ``reference`` or a copy.
    """

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    def reference(self) -> Mapping[str, bytes] | None:
        return None

    @abstractmethod
    def copy_into(self, directory: Path) -> None: ...


class DirectorySource(DirectoryFiles, SourceFiles):
    """A directory that an import copies, which ``source_version`` names.

    ``link`` hard links each file where the file system allows, for a directory whose files are never rewritten,
    such as Hugging Face's cache. Otherwise each file is copied, so a later edit of the directory changes nothing in
    the copy, and the copy drops the setuid and setgid bits, since the service owns it.
    """

    def __init__(self, directory: Path, *, source_version: str, link: bool) -> None:
        super().__init__(directory)
        self.source_version = source_version
        self.link = link

    @property
    def version(self) -> str:
        return self.source_version

    def copy_into(self, directory: Path) -> None:
        for relative in self.paths():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.link:
                link_or_copy(str(self.directory / relative), str(target))
            else:
                shutil.copy2(self.directory / relative, target)
                os.chmod(target, stat.S_IMODE(target.stat().st_mode) & ~(stat.S_ISUID | stat.S_ISGID))


class S3PrefixFiles(SourceFiles):
    """The objects under ``prefix`` in ``bucket``, at their keys below the prefix.

    One listing gives ``paths``. An empty object whose key ends in ``/`` is a folder marker, such as the S3
    console's "Create folder" writes, and is skipped. A key that ends in ``/`` and holds data, a key that is both a
    file and a folder, a key outside the prefix, and an empty prefix are refused with ``ValueError``. ``version``
    is the SHA-256 of the paths and ETags, so the same objects read again give the same version. ``sha256``,
    ``read_bytes`` and ``copy_into`` download the objects they read.
    """

    def __init__(self, uri: str, client: Any, client_errors: tuple[type[Exception], ...]) -> None:
        self.uri, self.client, self.client_errors = uri, client, client_errors
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        if not bucket:
            raise ValueError(f"{uri} names no bucket")
        self.bucket = bucket
        prefix = prefix if not prefix or prefix.endswith("/") else f"{prefix}/"
        self.keys: dict[str, str] = {}
        etags: list[tuple[str, str]] = []
        try:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", ()):
                    if item["Key"].endswith("/"):
                        if item.get("Size", 0):
                            raise ValueError(f"{uri} holds data in {item['Key']}, a key that names a folder")
                        continue
                    relative = PurePosixPath(item["Key"][len(prefix) :])
                    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                        raise ArtifactSourceError(f"{uri} holds an object outside its prefix: {item['Key']}")
                    self.keys[relative.as_posix()] = item["Key"]
                    etags.append((relative.as_posix(), str(item["ETag"])))
        except client_errors as exc:
            raise ArtifactSourceError(f"failed to read {uri}: {exc}") from exc
        if not self.keys:
            raise ValueError(f"{uri} holds no object")
        folders = {parent.as_posix() for path in self.keys for parent in PurePosixPath(path).parents}
        if both := sorted(folders & set(self.keys)):
            raise ValueError(f"{uri} holds a key that is both a file and a folder: {prefix}{both[0]}")
        self.listed = tuple(sorted(self.keys, key=path_order))
        self.etag_version = hashlib.sha256(json.dumps(sorted(etags)).encode()).hexdigest()

    @property
    def version(self) -> str:
        return self.etag_version

    def paths(self) -> tuple[str, ...]:
        return self.listed

    def sha256(self, path: str) -> str:
        return hashlib.sha256(self.read_bytes(path)).hexdigest()

    def read_bytes(self, path: str) -> bytes:
        key = self.key(path)
        try:
            return cast(bytes, self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except self.client_errors as exc:
            raise ArtifactSourceError(f"failed to read {key} of {self.uri}: {exc}") from exc

    def copy_into(self, directory: Path) -> None:
        for path in self.listed:
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.client.download_file(self.bucket, self.keys[path], str(target))
            except self.client_errors as exc:
                raise ArtifactSourceError(f"failed to read {self.keys[path]} of {self.uri}: {exc}") from exc

    def key(self, path: str) -> str:
        """The object key of ``path``; ``FileNotFoundError`` unless ``path`` is one of ``paths()``."""
        if path not in self.keys:
            raise FileNotFoundError(f"{self.uri} holds no object {path}")
        return self.keys[path]


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


@dataclass(frozen=True)
class HuggingFaceRevision:
    """One commit of a Hugging Face model repo (RFC #995).

    Values: ``repo`` is ``org/name`` with both parts non-empty and no other ``/``; ``commit`` is a commit SHA of
    40 lowercase hex digits. A commit names fixed bytes, so the files of a value never change while the repo exists.
    Purpose: a reference import stores one in ``REFERENCE_FILE`` as ``{"uri": "hf://<repo>", "version": "<commit>"}``
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

    @classmethod
    def parse(cls, uri: str) -> HuggingFaceRevision:
        """The revision ``hf://<repo>@<commit>`` names (``uri``); ``ValueError`` for another form."""
        repo, separator, commit = uri.removeprefix("hf://").rpartition("@")
        if not uri.startswith("hf://") or not separator:
            raise ValueError(f"a Hugging Face revision is hf://<repo>@<commit>, got {uri!r}")
        return cls(repo, commit)

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


class HuggingFaceRevisionFiles(SourceFiles):
    """The files of one commit of ``repo`` on the Hugging Face Hub, read without downloading its weights.

    ``revision`` names the commit: a commit SHA, a branch or a tag, or ``None`` for the default branch. One
    listing resolves it to a commit (``self.revision``), the ``version``, and gives ``paths`` and the SHA-256 that
    the Hub keeps for each LFS file. The SHA-256 of another file, which is below the Hub's LFS size, and
    ``read_bytes`` of any file download that one file into Hugging Face's cache; ``copy_into`` downloads the
    commit's snapshot there and links it. ``reference`` is ``REFERENCE_FILE`` when ``revision`` is the commit's
    full SHA, which names fixed files, and ``None`` for a branch or a tag, which can move. The token is Hugging
    Face's own: ``HF_TOKEN`` or the cached login. Raise ``ValueError`` when the revision does not resolve for that
    token, and ``ArtifactSourceError`` when the Hub fails.
    """

    def __init__(self, repo: str, revision: str | None, api: HfApi | None = None) -> None:
        self.api = HfApi() if api is None else api
        named = f"hf://{repo}" + ("" if revision is None else f"@{revision}")
        try:
            info = self.api.model_info(repo, revision=revision, files_metadata=True)
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise ValueError(f"{named} names no revision that the service can read: {exc}") from exc
        except (HfHubHTTPError, OSError) as exc:
            raise ArtifactSourceError(f"failed to list {named}: {exc}") from exc
        self.revision = HuggingFaceRevision(repo, str(info.sha))
        self.pinned = revision == self.revision.commit
        self.lfs_sha256 = {
            sibling.rfilename: sibling.lfs.sha256 for sibling in info.siblings or () if sibling.lfs is not None
        }
        self.listed = tuple(sorted((sibling.rfilename for sibling in info.siblings or ()), key=path_order))

    @property
    def version(self) -> str:
        return self.revision.commit

    @property
    def reference(self) -> Mapping[str, bytes] | None:
        if not self.pinned:
            return None
        return {REFERENCE_FILE: (json.dumps(self.revision.to_dict()) + "\n").encode()}

    def paths(self) -> tuple[str, ...]:
        return self.listed

    def sha256(self, path: str) -> str:
        if path in self.lfs_sha256:
            return self.lfs_sha256[path]
        with self.download(path).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def read_bytes(self, path: str) -> bytes:
        return self.download(path).read_bytes()

    def copy_into(self, directory: Path) -> None:
        snapshot = download_huggingface_snapshot(HuggingFaceSource(self.revision.repo, self.revision.commit))
        # huggingface_hub writes each file once under its hash and replaces it only by a rename, so a link keeps the
        # bytes that were imported.
        DirectorySource(snapshot.local_path, source_version=self.version, link=True).copy_into(directory)

    def download(self, path: str) -> Path:
        """The cached copy of the file at ``path``; ``FileNotFoundError`` unless ``path`` is one of ``paths()``."""
        if path not in self.listed:
            raise FileNotFoundError(f"{self.revision.uri} holds no file {path}")
        try:
            # Without dry_run, the download gives the cached file's path.
            return Path(cast(str, self.api.hf_hub_download(self.revision.repo, path, revision=self.revision.commit)))
        except (HfHubHTTPError, OSError) as exc:
            raise ArtifactSourceError(f"failed to read {path} of {self.revision.uri}: {exc}") from exc


class ArtifactSourceReader(ConfiguredPlugin, ABC):
    """Opens the files that a URI names, for an artifact import (RFC #980).

    Values: ``schemes`` is a non-empty tuple of URI schemes, such as ``s3``, that the reader opens. ``open(uri)``
    returns the ``SourceFiles`` that ``uri`` names, where the source keeps them; opening reads at most a listing, so
    an import checks the component's layout before it copies anything. It raises ``ValueError`` for a URI that
    names nothing the reader can read, such as a missing directory, and ``ArtifactSourceError`` when the source
    fails.

    Purpose: a deployment names extra readers in ``storage.artifact_sources``; ``ArtifactSources`` dispatches a
    URI to the reader of its scheme.
    """

    @property
    @abstractmethod
    def schemes(self) -> tuple[str, ...]: ...

    @abstractmethod
    def open(self, uri: str) -> SourceFiles: ...


class HuggingFaceReader(ArtifactSourceReader):
    """Opens ``hf://org/model[@revision]``: the files of the commit that the revision names, which is the version.

    A full commit SHA is kept by reference: Reef stores ``REFERENCE_FILE``, and the files stay on the Hub (RFC #995).
    A branch or a tag, which can move, is copied at the commit it names when it is opened.
    """

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("hf",)

    def open(self, uri: str) -> SourceFiles:
        source = parse_huggingface_source(uri)
        return HuggingFaceRevisionFiles(source.model_name, source.version)


class FileReader(ArtifactSourceReader):
    """Opens ``file:///path``: a directory on the service's file system, which has no version.

    The path is percent decoded, as ``Path.as_uri`` encodes it. The files are copied, not linked: files on the
    service can change after the import, and a hard link would carry the change into the store.
    """

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("file",)

    def open(self, uri: str) -> SourceFiles:
        # surrogateescape keeps a non UTF-8 name.
        source = Path(unquote(uri.removeprefix("file://"), errors="surrogateescape"))
        if not source.is_absolute() or not source.is_dir():
            raise ValueError(f"{uri} does not name an absolute directory path")
        return DirectorySource(source, source_version="", link=False)


@dataclass(frozen=True)
class S3ReaderSettings(PluginSettings):
    """The config of ``S3Reader``."""

    endpoint_url: str | None = config_option(None, help="An S3-compatible endpoint; unset is AWS S3.")


class S3Reader(ArtifactSourceReader):
    """Opens ``s3://bucket/prefix/``: every object under the prefix, at its path below it (``S3PrefixFiles``).

    The credentials are boto3's standard chain: the environment, a profile or an instance role.
    """

    def __init__(self, endpoint_url: str | None = None) -> None:
        self.endpoint_url = endpoint_url

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(S3ReaderSettings.parse(config, plugin_class=cls).endpoint_url)

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("s3",)

    def open(self, uri: str) -> SourceFiles:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise ArtifactSourceError("reading s3:// sources needs boto3: uv pip install 'reef-infra[s3]'") from exc
        return S3PrefixFiles(uri, boto3.client("s3", endpoint_url=self.endpoint_url), (BotoCoreError, ClientError))


@dataclass(frozen=True)
class ArtifactSources:
    """The source readers of a deployment, by URI scheme.

    Values: ``readers`` is a tuple of ``ArtifactSourceReader`` values, no two of which read one scheme.
    ``open(uri)`` is the open of the reader of ``uri``'s scheme; ``ValueError`` for a URI with no scheme or a
    scheme no reader reads. Purpose: the service builds it from ``storage.artifact_sources`` (``from_setting``) and
    opens an import's URI with it.
    """

    readers: tuple[ArtifactSourceReader, ...]

    def __post_init__(self) -> None:
        schemes = [scheme for reader in self.readers for scheme in reader.schemes]
        if len(schemes) != len(set(schemes)):
            raise ValueError(f"two artifact source readers read one scheme: {sorted(schemes)}")

    def open(self, uri: str) -> SourceFiles:
        scheme, separator, _ = uri.partition("://")
        reader = next((reader for reader in self.readers if scheme in reader.schemes), None) if separator else None
        if reader is None:
            known = sorted(scheme for reader in self.readers for scheme in reader.schemes)
            raise ValueError(f"{uri!r} is no URI of a known scheme; the readers read {known}")
        return reader.open(uri)

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
