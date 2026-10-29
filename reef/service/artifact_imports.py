"""Transport-independent handling of artifact imports (RFC #980, RFC #995).

A client imports one artifact into one component of a scenario by naming its directory with a URI. A source reader
(``ArtifactSources``) opens the files there, and the dispatcher checks them against the component's layout and
stores a copy, or a reference to a revision that cannot change, in a temporary directory under
``ArtifactImports.work_dir`` before it publishes it (``Dispatcher.import_artifact``). No aiohttp types appear here.
``reef.service.routes.artifact_imports`` adapts it to HTTP.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher


@dataclass(frozen=True)
class ArtifactImports:
    """How the service imports artifacts: the source readers, and the directory that holds each import while it runs.

    Values: ``sources`` opens the URIs of imports. ``work_dir`` is a directory on the service's file system, made
    when the first import runs; each import stores into its own temporary directory under it and removes that
    directory when it ends. A copied import of a full model needs its size there, and again in the artifact store;
    an import by reference needs neither.
    Purpose: the service builds it from ``storage.artifact_sources`` and ``storage.artifact_cache_dir``, and the
    import route calls ``import_source``.
    """

    sources: ArtifactSources
    work_dir: Path

    def import_source(self, dispatcher: Dispatcher, scenario: str, component: str | None, uri: str) -> dict[str, Any]:
        """Import the files ``uri`` names into ``component`` of ``scenario``, by reference or as a copy.

        Return ``{scenario, component, artifact_id, release_id, content_id}``. Before any file is read, raise
        ``UnknownScenario`` for an unknown scenario and ``ValueError`` for a component that cannot take the import
        (``Dispatcher.import_target``). The layout check then runs on the source, before anything is copied;
        otherwise raise as ``ArtifactSources.open`` and ``Dispatcher.import_artifact`` do.
        """
        component = dispatcher.import_target(scenario, component)
        files = self.sources.open(uri)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="import-", dir=self.work_dir) as work:
            imported = dispatcher.import_artifact(
                scenario, files, component, source={"uri": uri, "version": files.version}, directory=Path(work)
            )
        return {
            "scenario": scenario,
            "component": imported.component,
            "artifact_id": imported.artifact_id,
            "release_id": imported.release_id,
            "content_id": imported.content_id,
        }


__all__ = ["ArtifactImports"]
