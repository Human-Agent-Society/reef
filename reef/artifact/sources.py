"""The source readers of artifact imports: each opens the files that a URI names (RFC #980, RFC #995)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import unquote

from reef.artifact.artifact import ArtifactSourceError
from reef.artifact.files import (
    DirectorySource,
    HuggingFaceRevisionFiles,
    S3PrefixFiles,
    SourceFiles,
    parse_huggingface_source,
)
from reef.core.config import config_option
from reef.core.plugins import ConfiguredPlugin, PluginSettings


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
