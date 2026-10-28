"""Transport-independent handling of artifact imports (RFC #980).

A client imports one artifact into one component of a scenario. It sends the artifact's directory as a
gzip-compressed tar archive, or names it by a URI that a source reader fetches (``ArtifactSources``). Either way the
directory lands in a temporary directory under ``ArtifactImports.work_dir``, and the dispatcher checks and publishes
it (``Dispatcher.import_artifact``). No aiohttp types appear here. ``reef.service.routes.artifact_imports`` adapts
it to HTTP.
"""

from __future__ import annotations

import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.artifact.sources import ArtifactSources
from reef.core.errors import UnknownScenario
from reef.dispatcher import Dispatcher
from reef.scenario.committer import ComponentArtifact

#: The most archive entries, and the most bytes they unpack to. The route also refuses a body larger than
#: ``MAX_ARCHIVE_BYTES``. Full model weights are larger: they come by URI.
MAX_ARCHIVE_ENTRIES = 100_000
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024


def unpack_artifact_archive(archive: Path, target: Path) -> None:
    """Unpack the tar.gz file ``archive`` into the empty directory ``target``: the archive's root is the artifact's.

    Every entry is a regular file or a directory at a relative path. A link, a device, an absolute path or a ``..``
    component is refused with ``ValueError``, as is an archive of no file, of more than ``MAX_ARCHIVE_ENTRIES``
    entries, or of more than ``MAX_ARCHIVE_BYTES`` bytes.
    """
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"the archive holds more than {MAX_ARCHIVE_ENTRIES} entries")
            if sum(member.size for member in members) > MAX_ARCHIVE_BYTES:
                raise ValueError(f"the archive unpacks to more than {MAX_ARCHIVE_BYTES} bytes")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"the archive entry {member.name!r} is not a relative path")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"the archive entry {member.name!r} is not a file or a directory")
            if not any(member.isfile() for member in members):
                raise ValueError("the archive holds no file")
            tar.extractall(target, filter="data")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError(f"the body is not a gzip-compressed tar archive: {exc}") from exc


@dataclass(frozen=True)
class ArtifactImports:
    """How the service imports artifacts: the source readers, and the directory that holds each import while it runs.

    Values: ``sources`` reads the URIs of URI imports. ``work_dir`` is a directory on the service's file system,
    made when the first import runs; each import writes into its own temporary directory under it and removes that
    directory when it ends. A URI import of a full model needs its size there, and again in the artifact store.
    Purpose: the service builds it from ``storage.artifact_sources`` and ``storage.artifact_cache_dir``, and the
    import route calls it.
    """

    sources: ArtifactSources
    work_dir: Path

    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        """A new temporary directory under ``work_dir``, removed when its context ends."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="import-", dir=self.work_dir)

    def import_archive(
        self, dispatcher: Dispatcher, scenario: str, component: str | None, archive: Path
    ) -> dict[str, Any]:
        """Unpack ``archive`` (``unpack_artifact_archive``) and import it; return the receipt (``receipt``)."""
        with self.temporary_directory() as work:
            directory = Path(work) / "artifact"
            unpack_artifact_archive(archive, directory)
            return receipt(scenario, dispatcher.import_artifact(scenario, directory, component))

    def import_source(self, dispatcher: Dispatcher, scenario: str, component: str | None, uri: str) -> dict[str, Any]:
        """Fetch ``uri`` with ``sources`` and import it with its source; return the receipt (``receipt``).

        Raise ``UnknownScenario`` before the fetch, which can download many gigabytes, for an unknown scenario.
        """
        if not dispatcher.has_scenario(scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        with self.temporary_directory() as work:
            directory = Path(work) / "artifact"
            directory.mkdir()
            version = self.sources.fetch(uri, directory)
            imported = dispatcher.import_artifact(
                scenario, directory, component, source={"uri": uri, "version": version}
            )
            return receipt(scenario, imported)


def receipt(scenario: str, imported: ComponentArtifact) -> dict[str, Any]:
    """The answer to an import: ``{scenario, component, artifact_id, release_id, content_id}``."""
    return {
        "scenario": scenario,
        "component": imported.component,
        "artifact_id": imported.artifact_id,
        "release_id": imported.release_id,
        "content_id": imported.content_id,
    }


__all__ = ["MAX_ARCHIVE_BYTES", "ArtifactImports", "receipt", "unpack_artifact_archive"]
