"""Transport-independent handling of artifact imports (RFC #980, RFC #995).

A client imports one artifact into one component of a scenario by naming its directory with a URI, or by naming an
artifact item of the scenario's owner, whose ``source_uri`` names the directory (``reef.storage.artifact_items``). A source reader
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
from reef.core.errors import UnknownScenario
from reef.dispatcher import Dispatcher
from reef.service.data import data_store
from reef.storage.data import ItemNotFound


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

    def import_source(
        self, dispatcher: Dispatcher, scenario: str, component: str | None, uri: str, *, item_id: str | None = None
    ) -> dict[str, Any]:
        """Import the files ``uri`` names into ``component`` of ``scenario``, by reference or as a copy.

        Return ``{scenario, component, artifact_id, release_id, content_id}``. The import's ``source`` is ``{uri,
        version}``, and also ``item``, ``item_id``, when the URI came from that artifact item. Before any file is
        read, raise ``UnknownScenario`` for an unknown scenario and ``ValueError`` for a component that cannot take
        the import (``Dispatcher.import_target``). The layout check then runs on the source, before anything is
        copied; otherwise raise as ``ArtifactSources.open`` and ``Dispatcher.import_artifact`` do.
        """
        component = dispatcher.import_target(scenario, component)
        files = self.sources.open(uri)
        source = {"uri": uri, "version": files.version}
        if item_id is not None:
            source["item"] = item_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="import-", dir=self.work_dir) as work:
            imported = dispatcher.import_artifact(scenario, files, component, source=source, directory=Path(work))
        return {
            "scenario": scenario,
            "component": imported.component,
            "artifact_id": imported.artifact_id,
            "release_id": imported.release_id,
            "content_id": imported.content_id,
        }

    def import_item(
        self, dispatcher: Dispatcher, scenario: str, component: str | None, item_id: str
    ) -> dict[str, Any]:
        """Import the artifact item ``item_id`` of the owner of ``scenario`` into ``component``: the files that its
        ``source_uri`` names, as ``import_source`` imports them.

        Raise ``UnknownScenario`` for an unknown scenario, ``NoDataStore`` without a data store, ``ItemNotFound``
        for an item that the owner does not store, and ``ValueError`` for an item that is not an artifact or that
        has no ``source_uri``; otherwise raise as ``import_source``.
        """
        owner = dispatcher.scenario_owner(scenario)
        if owner is None:
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        item = data_store(dispatcher).read(owner, item_id)
        if item is None:
            raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
        if item.kind != "artifact" or item.content is None:
            raise ValueError(f"item {item_id!r} is a {item.kind}, not an artifact")
        uri = item.content["source_uri"]
        if uri is None:
            raise ValueError(
                f"artifact item {item_id!r} has no source_uri: its repository gives no URI to import from"
            )
        return self.import_source(dispatcher, scenario, component, uri, item_id=item_id)


__all__ = ["ArtifactImports"]
