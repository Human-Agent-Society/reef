"""Transport-independent handling of artifact imports (RFC #980).

A client imports one artifact into one component of a scenario by naming its directory with a URI. A source reader
(``ArtifactSources``) copies the directory into a temporary directory under ``ArtifactImports.work_dir``, or writes
the reference that Reef keeps for it there (RFC #995), and the dispatcher checks and publishes it
(``Dispatcher.import_artifact``). No aiohttp types appear here.
``reef.service.routes.artifact_imports`` adapts it to HTTP.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.files import ArtifactFiles, DirectoryFiles
from reef.artifact.sources import ArtifactSources
from reef.core.errors import UnknownScenario
from reef.dispatcher import Dispatcher


@dataclass(frozen=True)
class ArtifactImports:
    """How the service imports artifacts: the source readers, and the directory that holds each import while it runs.

    Values: ``sources`` reads the URIs of imports. ``work_dir`` is a directory on the service's file system, made
    when the first import runs; each import copies into its own temporary directory under it and removes that
    directory when it ends. A copied import of a full model needs its size there, and again in the artifact store;
    an import by reference needs neither.
    Purpose: the service builds it from ``storage.artifact_sources`` and ``storage.artifact_cache_dir``, and the
    import route calls ``import_source``.
    """

    sources: ArtifactSources
    work_dir: Path

    def import_source(self, dispatcher: Dispatcher, scenario: str, component: str | None, uri: str) -> dict[str, Any]:
        """Import the directory ``uri`` names into ``component`` of ``scenario``, by reference or as a copy.

        The reader of ``uri`` keeps it by reference when it can (``ArtifactSourceReader.keep_by_reference``); otherwise it
        copies the directory. Return ``{scenario, component, artifact_id, release_id, content_id}``. Raise
        ``UnknownScenario`` before the source is read, which can download many gigabytes, for an unknown scenario;
        otherwise raise as the reader and ``Dispatcher.import_artifact`` do.
        """
        if not dispatcher.has_scenario(scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        reader = self.sources.reader(uri)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="import-", dir=self.work_dir) as work:
            directory = Path(work)
            referenced = reader.keep_by_reference(uri, directory)
            files: ArtifactFiles
            if referenced is None:
                version = reader.fetch(uri, directory)
                files = DirectoryFiles(directory)
            else:
                version, files = referenced.version, referenced.files
            imported = dispatcher.import_artifact(
                scenario, directory, files, component, source={"uri": uri, "version": version}
            )
        return {
            "scenario": scenario,
            "component": imported.component,
            "artifact_id": imported.artifact_id,
            "release_id": imported.release_id,
            "content_id": imported.content_id,
        }


__all__ = ["ArtifactImports"]
