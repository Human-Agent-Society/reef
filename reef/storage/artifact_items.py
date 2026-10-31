"""Artifact items: the artifacts of a workspace in its data store, the workspace's artifact library (RFC #1028).

An artifact item is a reference: its content says what the artifact is and where its files are, and the files stay
where they are. There are two kinds of artifact item of an owner ``o``.

* The item of an artifact that a scenario ``s`` of ``o`` lists (``Scenario.artifacts``). Let ``a`` be that row. Its
  item has the ID ``<s>/<a.artifact_id>``, unique within ``o`` because a scenario name has no ``/`` and an artifact
  ID is unique within its scenario. Its metadata is ``kind: artifact`` and ``origin: {scenario: s, release_id,
  artifact_id}`` of ``a``. Its content is ``{source_uri, kind, component, content_id}``: ``component`` and
  ``content_id`` of ``a``, ``kind`` the ``ArtifactKind`` of the component (null when the surface declares none), and
  ``source_uri`` the URI of the component's files in the scenario's repository (``Repository.release_uri``), or null
  when the repository gives none.
* An item that a client creates for files that it stored, such as an upload. It has no ``origin``. Its content
  holds a ``source_uri``, a non-empty string, and may hold ``kind`` and ``content_id``; ``component`` is null.

``scenarios`` is empty for every artifact item: a scenario uses one by importing its ``source_uri``
(``ArtifactImports.import_item``), which makes a new artifact of that scenario.

The data service makes the items of a scenario's artifacts when it takes an ``artifact_copy`` item from the ``data``
queue, which control adds after each commit (``request_artifact_copy``). A copy makes the item of every artifact
that the scenario lists, and an item that the owner already stores with the same content is unchanged, so a copy
can run again at any time. An item that a scenario made stays while the scenario exists: the next copy would make
it again, so the service refuses to delete it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.evaluation import ARTIFACT_KINDS
from reef.storage.data import DATA_QUEUE, NewItem
from reef.storage.work_queue import WorkItem, WorkQueue, request_waiting

#: The kind of the ``data`` queue item that copies the artifacts of one scenario to artifact items.
ARTIFACT_COPY_KIND = "artifact_copy"
#: The keys of an artifact item's content.
ARTIFACT_CONTENT_KEYS = frozenset({"source_uri", "kind", "component", "content_id"})


def check_artifact_content(content: Mapping[str, Any], *, made_by_scenario: bool) -> None:
    """Raise ``ValueError`` unless ``content`` is the content of an artifact item, as the module docstring defines it.

    ``made_by_scenario`` is whether the item has an ``origin``. Every value is a non-empty string or null;
    ``kind`` is an ``ArtifactKind``; a client's item has a ``source_uri`` and no ``component``, and an item that a
    scenario made has a ``component``.
    """
    if set(content) != ARTIFACT_CONTENT_KEYS:
        raise ValueError(f"an artifact item's content holds exactly {sorted(ARTIFACT_CONTENT_KEYS)}")
    for key in sorted(ARTIFACT_CONTENT_KEYS):
        value = content[key]
        if value is not None and not (isinstance(value, str) and value):
            raise ValueError(f"an artifact item's content.{key} is a non-empty string or null")
    if content["kind"] is not None and content["kind"] not in ARTIFACT_KINDS:
        raise ValueError(f"an artifact item's content.kind is one of {ARTIFACT_KINDS} or null")
    if made_by_scenario and content["component"] is None:
        raise ValueError("the artifact item of a scenario's artifact names its component")
    if not made_by_scenario and (content["source_uri"] is None or content["component"] is not None):
        raise ValueError("a client's artifact item names its source_uri, and no component")


def artifact_item(scenario: str, row: Mapping[str, Any], kind: str | None, source_uri: str | None) -> NewItem:
    """The artifact item of ``row``, one artifact that ``scenario`` lists, as the module docstring defines it."""
    content = {
        "source_uri": source_uri,
        "kind": kind,
        "component": row["component"],
        "content_id": row["content_id"],
    }
    check_artifact_content(content, made_by_scenario=True)
    return NewItem(
        f"{scenario}/{row['artifact_id']}",
        content,
        {
            "kind": "artifact",
            "origin": {"scenario": scenario, "release_id": row["release_id"], "artifact_id": row["artifact_id"]},
        },
    )


def artifact_copy_key(scenario: str) -> str:
    """The serial key of the ``artifact_copy`` items of ``scenario``."""
    return f"{ARTIFACT_COPY_KIND}:{scenario}"


def request_artifact_copy(queue: WorkQueue, scenario: str) -> WorkItem:
    """Ask the data service to copy the artifacts of ``scenario``; return the item that will copy them."""
    return request_waiting(queue, DATA_QUEUE, scenario, ARTIFACT_COPY_KIND, artifact_copy_key(scenario), {})


__all__ = [
    "ARTIFACT_CONTENT_KEYS",
    "ARTIFACT_COPY_KIND",
    "artifact_copy_key",
    "artifact_item",
    "check_artifact_content",
    "request_artifact_copy",
]
