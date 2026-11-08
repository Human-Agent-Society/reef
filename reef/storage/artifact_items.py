"""Artifact items: the artifacts of a workspace in its data store, the workspace's artifact library (RFC #1028).

An artifact item is a reference: its content says what the artifact is and where its files are, and the files stay
where they are. There are two kinds of artifact item of an owner ``o``.

* The item of an artifact that a scenario ``s`` of ``o`` lists (``Scenario.artifacts``). Let ``a`` be that row. Its
  item has the ID ``<s>/<a.artifact_id>``, unique within ``o`` because a scenario name has no ``/`` and an artifact
  ID is unique within its scenario. Its metadata is ``kind: artifact`` and ``origin: {scenario: s, artifact_id}``
  of ``a``. Its content is ``{source_uri, kind, component}``: ``component`` of ``a``, ``kind`` the ``ArtifactKind`` of the component (null when the scenario declares none), and
  ``source_uri`` the URI of the component's files in the scenario's repository (``Repository.snapshot_uri``), or null
  when the repository gives none. An item of the kind ``harness_plugin`` also has ``adapter`` in its metadata: the
  adapter that its tree renders for.
* An item that a client creates for files that it stored, such as an upload. It has no ``origin``. Its content
  holds a ``source_uri``, a non-empty string, and may hold ``kind``; ``component`` is null. The client gives
  ``adapter`` in its metadata for a harness tree.

``scenarios`` is empty for every artifact item: a scenario uses one by importing its ``source_uri``
(``ArtifactImports.import_item``). The artifact keeps its ID in that scenario, and it gets no second item, so the
library holds it once (RFC #1218). An upload gets its ID, ``metadata.artifact_id``, at its first use. An evaluation
of a model artifact
can also read a harness tree from an item, and makes no artifact (``ArtifactEvaluations.tree_harness``).

The data service makes the items of a scenario's artifacts when it takes an ``artifact_copy`` item from the ``data``
queue, which control adds after each commit (``request_artifact_copy``). A copy makes the item of every artifact
that the scenario lists, and an item that the owner already stores with the same content is unchanged, so a copy
can run again at any time. An item that a scenario made stays while the scenario exists: the next copy would make
it again, so the service refuses to delete it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.storage.data import NewItem
from reef.storage.data_queue import ARTIFACT_COPY_KIND, DATA_QUEUE
from reef.storage.item_kinds import check_artifact_content
from reef.storage.work_queue import WorkItem, WorkQueue


def artifact_item(
    scenario: str,
    row: Mapping[str, Any],
    kind: str | None,
    source_uri: str | None,
    adapter: str | None = None,
) -> NewItem:
    """The artifact item of ``row``, one artifact that ``scenario`` lists, as the module docstring defines it.

    ``adapter`` is the adapter that a harness tree renders for; only an item of the kind ``harness_plugin`` keeps it.
    """
    content = {
        "source_uri": source_uri,
        "kind": kind,
        "component": row["component"],
    }
    check_artifact_content(content, made_by_scenario=True)
    return NewItem(
        f"{scenario}/{row['artifact_id']}",
        content,
        {
            "kind": "artifact",
            "origin": {"scenario": scenario, "artifact_id": row["artifact_id"]},
            **({"adapter": adapter} if kind == "harness_plugin" and adapter is not None else {}),
        },
    )


def artifact_copy_key(scenario: str) -> str:
    """The serial key of the ``artifact_copy`` items of ``scenario``."""
    return f"{ARTIFACT_COPY_KIND}:{scenario}"


def request_artifact_copy(queue: WorkQueue, scenario: str) -> WorkItem:
    """Ask the data service to copy the artifacts of ``scenario``; return the item that will copy them."""
    return queue.request(DATA_QUEUE, scenario, ARTIFACT_COPY_KIND, artifact_copy_key(scenario), {})


__all__ = [
    "artifact_copy_key",
    "artifact_item",
    "request_artifact_copy",
]
