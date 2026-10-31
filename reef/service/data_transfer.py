"""The data service's transfers: imports from a URI, exports to a URI, and the archive of one item (RFC #1028).

An import or an export is a work item of the ``data`` queue, so it runs on a data service worker and outlives the
request that started it. Its ``scenario`` field holds the owner: the transfer belongs to the owner's items, not to
a scenario.

- An import (``data_import``) reads the files at a URI with the deployment's source readers (``ArtifactSources``)
  and stores them as items of the owner with the request's metadata. For ``kind: task`` each Harbor task directory
  becomes a task item: the URI's root when it holds ``task.toml``, else each directory under the root that holds
  one. For ``kind: trajectory`` each ``*.jsonl`` file at the root holds one ATIF document a line, and each line
  becomes a trajectory item named after its ``session_id``. The items are stored in transactions of at most
  ``MAX_ITEMS_PER_CREATE``, so an import that fails can leave the items of its earlier transactions. It checks and stores the items as an upload does, and asks control to sync the scenarios
  without waiting.
- An export (``data_export``) writes the items that the request names, or that match its ``kind`` and ``scenario``,
  to a tar.gz at a URI (``ArchiveWriters``). The archive holds ``manifest.jsonl``, one line ``{id, kind, metadata,
  path}`` for each item, and each item at its ``path``: ``items/<n>.json`` for the content of a trajectory or a
  record, ``items/<n>/`` for the files of a task.
- The archive of one item (``item_archive``) is the tar.gz of a task's files, or the JSON content of another item.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from reef.artifact.sources import ArtifactSources
from reef.service.data import (
    KINDS,
    MAX_ITEMS_PER_CREATE,
    DataAccess,
    create_task_directories,
    create_trajectory_items,
    jsonl_trajectories,
)
from reef.storage.data import MAX_QUERY_LIMIT, ItemNotFound, StoredItem, check_owner
from reef.storage.data_queue import DATA_EXPORT_KIND, DATA_IMPORT_KIND, DATA_QUEUE
from reef.storage.work_queue import JsonValue, NewWork, WorkItem

#: The transfer kinds by the route segment that names them.
TRANSFER_KINDS = {"imports": DATA_IMPORT_KIND, "exports": DATA_EXPORT_KIND}
#: The item kinds that an import creates.
IMPORT_ITEM_KINDS = ("task", "trajectory")


def start_transfer(data: DataAccess, owner: str, segment: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Check the request ``body`` of an import or an export of ``owner`` and queue it; return its status.

    An import body is ``{uri, metadata}``, whose ``metadata.kind`` is one of ``IMPORT_ITEM_KINDS``. An export body
    is ``{uri}`` with either ``ids``, 1 to ``MAX_QUERY_LIMIT`` item IDs, or the query fields ``kind`` and
    ``scenario``. Raise ``ValueError`` for another body.
    """
    check_owner(owner)
    kind = TRANSFER_KINDS[segment]
    uri = body.get("uri")
    if not isinstance(uri, str) or "://" not in uri:
        raise ValueError("a transfer names a uri such as s3://bucket/prefix or file:///path")
    if kind == DATA_IMPORT_KIND:
        if body.keys() != {"uri", "metadata"} or not isinstance(body["metadata"], Mapping):
            raise ValueError("an import holds only 'uri' and 'metadata', an object")
        if body["metadata"].get("kind") not in IMPORT_ITEM_KINDS:
            raise ValueError(f"an import's metadata.kind is one of {', '.join(IMPORT_ITEM_KINDS)}")
    else:
        if not body.keys() <= {"uri", "ids", "kind", "scenario"}:
            raise ValueError("an export holds 'uri', and 'ids' or the query fields 'kind' and 'scenario'")
        ids = body.get("ids")
        if ids is not None and (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= MAX_QUERY_LIMIT
            or not all(isinstance(item_id, str) and item_id for item_id in ids)
            or body.keys() & {"kind", "scenario"}
        ):
            raise ValueError(f"an export's ids are 1 to {MAX_QUERY_LIMIT} item IDs, without query fields")
        if body.get("kind") is not None and body["kind"] not in KINDS:
            raise ValueError(f"an export's kind is one of {', '.join(KINDS)}")
    payload = cast(dict[str, JsonValue], {"owner": owner, **body})
    item = data.queue.add(NewWork(DATA_QUEUE, f"{kind}:{uuid.uuid4().hex}", owner, kind, payload))
    return transfer_json(item)


def transfer_status(data: DataAccess, owner: str, segment: str, transfer_id: str) -> dict[str, Any]:
    """The status of the import or export ``transfer_id`` of ``owner``; ``ItemNotFound`` for another ID."""
    item = data.queue.get(DATA_QUEUE, transfer_id)
    if item is None or item.kind != TRANSFER_KINDS[segment] or item.payload.get("owner") != owner:
        raise ItemNotFound(f"{owner!r} has no {segment[:-1]} {transfer_id!r}")
    return transfer_json(item)


def transfer_json(item: WorkItem) -> dict[str, Any]:
    """A transfer as JSON: ``{id, kind, state, request, result, error}``."""
    request = {key: value for key, value in item.payload.items() if key != "owner"}
    return {
        "id": item.id,
        "kind": item.kind,
        "state": item.state,
        "request": request,
        "result": None if item.result is None else dict(item.result),
        "error": item.error,
    }


def run_import(data: DataAccess, sources: ArtifactSources, payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Import the files at ``payload["uri"]`` as items of ``payload["owner"]``; return ``{items}``, the count."""
    owner, uri, metadata = payload["owner"], payload["uri"], payload["metadata"]
    with tempfile.TemporaryDirectory(prefix="reef-data-import-") as work:
        root = Path(work) / "files"
        root.mkdir()
        sources.open(uri).copy_into(root)
        if metadata["kind"] == "task":
            directories = (
                [root]
                if (root / "task.toml").is_file()
                else sorted(path for path in root.iterdir() if (path / "task.toml").is_file())
            )
            if not directories:
                raise ValueError(f"{uri} holds no Harbor task directory")
            if directories == [root]:
                # A single task is named after the last segment of its URI.
                name = Path(unquote(urlsplit(uri).path).rstrip("/")).name
                if not name:
                    raise ValueError(f"{uri} holds one task, and its path names no directory to name it after")
                directories = [root.rename(Path(work) / name)]
            create_task_directories(data, owner, directories, metadata, wait=False)
            return {"items": len(directories)}
        files = sorted(root.glob("*.jsonl"))
        if not files:
            raise ValueError(f"{uri} holds no *.jsonl file of trajectories at its root")
        documents = []
        for path in files:
            with path.open(encoding="utf-8") as lines:
                documents.extend(jsonl_trajectories(lines, path.name))
        repeated = sorted(
            item_id for item_id, count in Counter(item_id for item_id, _ in documents).items() if count > 1
        )
        if repeated:
            raise ValueError(f"{uri} holds the session_ids {repeated} in more than one file")
        for start in range(0, len(documents), MAX_ITEMS_PER_CREATE):
            create_trajectory_items(data, owner, documents[start : start + MAX_ITEMS_PER_CREATE], metadata, wait=False)
        return {"items": len(documents)}


def run_export(data: DataAccess, writers: ArchiveWriters, payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Write the items that ``payload`` selects to a tar.gz at ``payload["uri"]``; return ``{items, bytes}``."""
    store = data.require_store()
    owner = payload["owner"]
    ids = payload.get("ids")
    items: list[StoredItem] = []
    if ids is not None:
        for item_id in ids:
            item = store.read(owner, item_id)
            if item is None:
                raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
            items.append(item)
    else:
        after = 0
        kind = None if payload.get("kind") is None else KINDS[payload["kind"]]
        while page := store.query(
            owner, kind=kind, scenario=payload.get("scenario"), after=after, limit=MAX_QUERY_LIMIT
        ):
            items.extend(page)
            after = page[-1].sequence
    with tempfile.TemporaryDirectory(prefix="reef-data-export-") as work:
        archive = Path(work) / "export.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            lines = []
            for index, item in enumerate(items):
                if item.kind == "task":
                    path = f"items/{index}"
                    tar.add(store.task_path(item), arcname=path)
                else:
                    path = f"items/{index}.json"
                    add_bytes(tar, path, json.dumps(dict(item.content or {}), ensure_ascii=False).encode())
                lines.append(
                    json.dumps(
                        {"id": item.item_id, "kind": item.kind, "metadata": dict(item.metadata), "path": path},
                        ensure_ascii=False,
                    )
                )
            add_bytes(tar, "manifest.jsonl", ("\n".join(lines) + "\n" if lines else "").encode())
        size = archive.stat().st_size
        writers.write(payload["uri"], archive)
    return {"items": len(items), "bytes": size}


def item_archive(data: DataAccess, owner: str, item_id: str, directory: Path) -> tuple[Path, str]:
    """Write the archive of the item ``item_id`` of ``owner`` into ``directory``; return its file and media type.

    A task's archive is the tar.gz of its files under its name; another item's is its content as JSON. Raise
    ``ItemNotFound`` for an unknown ID.
    """
    store = data.require_store()
    item = store.read(owner, item_id)
    if item is None:
        raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
    name = item_id.rsplit("/", 1)[-1]
    if item.kind == "task":
        archive = directory / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(store.task_path(item), arcname=name)
        return archive, "application/gzip"
    document = directory / f"{name}.json"
    document.write_text(json.dumps(dict(item.content or {}), ensure_ascii=False), encoding="utf-8")
    return document, "application/json"


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    with tempfile.SpooledTemporaryFile() as buffer:
        buffer.write(data)
        buffer.seek(0)
        tar.addfile(info, buffer)


class ArchiveWriter(ABC):
    """Writes one export archive to a URI of its schemes.

    Purpose: ``ArchiveWriters`` dispatches an export's URI to the writer of its scheme.
    """

    @property
    @abstractmethod
    def schemes(self) -> tuple[str, ...]: ...

    @abstractmethod
    def write(self, uri: str, archive: Path) -> None:
        """Write the file ``archive`` at ``uri``; ``ValueError`` for a URI that it cannot write."""


class FileWriter(ArchiveWriter):
    """Writes ``file:///path``: a new file on the service's file system. It never replaces an existing file."""

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("file",)

    def write(self, uri: str, archive: Path) -> None:
        target = Path(unquote(uri.removeprefix("file://"), errors="surrogateescape"))
        if not target.is_absolute() or not target.parent.is_dir():
            raise ValueError(f"{uri} does not name a file in an existing absolute directory")
        with archive.open("rb") as source, target.open("xb") as written:
            while chunk := source.read(1024 * 1024):
                written.write(chunk)


class S3Writer(ArchiveWriter):
    """Writes ``s3://bucket/key`` with boto3's standard credential chain, as ``S3Reader`` reads."""

    def __init__(self, endpoint_url: str | None = None) -> None:
        self.endpoint_url = endpoint_url

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("s3",)

    def write(self, uri: str, archive: Path) -> None:
        parts = urlsplit(uri)
        key = parts.path.lstrip("/")
        if not parts.netloc or not key or key.endswith("/"):
            raise ValueError(f"{uri} does not name an object s3://bucket/key")
        try:
            import boto3
        except ImportError as exc:
            raise ValueError("writing s3:// exports needs boto3: uv pip install 'reef-infra[s3]'") from exc
        boto3.client("s3", endpoint_url=self.endpoint_url).upload_file(str(archive), parts.netloc, key)


@dataclass(frozen=True)
class ArchiveWriters:
    """The export writers of a deployment, by URI scheme.

    Values: ``writers`` is a tuple of ``ArchiveWriter`` values, no two of which write one scheme. Purpose: the data
    service writes each export with it.
    """

    writers: tuple[ArchiveWriter, ...] = (FileWriter(), S3Writer())

    def __post_init__(self) -> None:
        schemes = [scheme for writer in self.writers for scheme in writer.schemes]
        if len(schemes) != len(set(schemes)):
            raise ValueError(f"two archive writers write one scheme: {sorted(schemes)}")

    def write(self, uri: str, archive: Path) -> None:
        scheme = uri.partition("://")[0]
        writer = next((writer for writer in self.writers if scheme in writer.schemes), None)
        if writer is None:
            known = sorted(scheme for writer in self.writers for scheme in writer.schemes)
            raise ValueError(f"{uri!r} is no URI that an export writes; the writers write {known}")
        writer.write(uri, archive)


__all__ = [
    "TRANSFER_KINDS",
    "ArchiveWriter",
    "ArchiveWriters",
    "FileWriter",
    "S3Writer",
    "item_archive",
    "run_export",
    "run_import",
    "start_transfer",
    "transfer_json",
    "transfer_status",
]
