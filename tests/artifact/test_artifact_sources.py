"""The source readers of artifact imports and the files they open (RFC #980, RFC #995)."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support.hugging_face_hub import COMMIT, MemoryHub

from reef.artifact import (
    REFERENCE_FILE,
    ArtifactSourceError,
    ArtifactSourceReader,
    ArtifactSources,
    DirectoryFiles,
    DownloadedSnapshot,
    FileReader,
    HuggingFaceReader,
    HuggingFaceRevision,
    HuggingFaceRevisionFiles,
    S3Reader,
    SourceFiles,
    directory_content_id,
    read_reference,
)


class MemoryFiles(SourceFiles):
    """One file, ``uri.txt``, that holds the URI it was opened for."""

    def __init__(self, uri: str) -> None:
        self.files = {"uri.txt": uri.encode()}

    @property
    def version(self) -> str:
        return "v1"

    def paths(self) -> tuple[str, ...]:
        return tuple(self.files)

    def sha256(self, path: str) -> str:
        return hashlib.sha256(self.read_bytes(path)).hexdigest()

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    def copy_into(self, directory: Path) -> None:
        for path, data in self.files.items():
            (directory / path).write_bytes(data)


class MemoryReader(ArtifactSourceReader):
    """Opens ``mem://`` and ``s3://`` URIs as ``MemoryFiles``."""

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("mem", "s3")

    def open(self, uri: str) -> SourceFiles:
        return MemoryFiles(uri)


def write_tree(directory: Path, files: Mapping[str, bytes]) -> Path:
    for relative, content in files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(content)
    return directory


def copied(files: SourceFiles, directory: Path) -> Path:
    directory.mkdir()
    files.copy_into(directory)
    return directory


def tree(directory: Path) -> list[str]:
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())


@pytest.mark.unit
def test_the_content_id_names_the_files_and_their_paths_only(tmp_path: Path) -> None:
    files = {"config.json": b"{}", "adapter/weights.bin": b"w"}
    first = directory_content_id(write_tree(tmp_path / "a", files))
    assert first.startswith("content:")
    assert directory_content_id(write_tree(tmp_path / "b", files)) == first
    assert directory_content_id(write_tree(tmp_path / "c", {**files, "config.json": b"{ }"})) != first
    assert directory_content_id(write_tree(tmp_path / "d", {"config.json": b"{}", "weights.bin": b"w"})) != first
    # Paths sort by their components, as pathlib sorts them, so an existing artifact keeps its content ID.
    root = write_tree(tmp_path / "e", {"a-b": b"1", "a/b": b"2"})
    listed = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    assert directory_content_id(root) == f"content:{hashlib.sha256(json.dumps(listed).encode()).hexdigest()}"


@pytest.mark.unit
def test_a_file_uri_opens_an_absolute_directory_without_a_version(tmp_path: Path) -> None:
    source = write_tree(tmp_path / "source", {"config.json": b"{}", "sub/weights.bin": b"w"})
    files = ArtifactSources.from_setting(()).open(source.as_uri())
    assert (files.version, files.reference, files.paths()) == ("", None, ("config.json", "sub/weights.bin"))
    assert directory_content_id(copied(files, tmp_path / "target")) == files.content_id()
    for uri in ("file://relative/path", (tmp_path / "missing").as_uri()):
        with pytest.raises(ValueError, match="absolute directory"):
            FileReader().open(uri)


@pytest.mark.unit
def test_a_file_uri_copies_so_a_later_edit_of_the_source_changes_nothing(tmp_path: Path) -> None:
    source = write_tree(tmp_path / "my source", {"harness.txt": b"harness v1\n"})
    assert "%20" in source.as_uri()
    target = copied(FileReader().open(source.as_uri()), tmp_path / "target")
    with (source / "harness.txt").open("r+b") as file:
        file.write(b"changed v2\n")
    assert (target / "harness.txt").read_bytes() == b"harness v1\n"


@pytest.mark.unit
def test_a_file_uri_copy_keeps_the_mode_but_drops_setuid_and_setgid(tmp_path: Path) -> None:
    source = write_tree(tmp_path / "source", {"run.sh": b"#!/bin/sh\n"})
    (source / "run.sh").chmod(0o755 | stat.S_ISUID | stat.S_ISGID)
    target = copied(FileReader().open(source.as_uri()), tmp_path / "target")
    assert stat.S_IMODE((target / "run.sh").stat().st_mode) == 0o755


@pytest.mark.unit
def test_a_uri_goes_to_the_reader_of_its_scheme() -> None:
    sources = ArtifactSources.from_setting([f"{__name__}:MemoryReader"])
    # A configured reader replaces the built-in reader of a scheme it reads.
    assert sorted(type(reader).__name__ for reader in sources.readers) == [
        "FileReader",
        "HuggingFaceReader",
        "MemoryReader",
    ]
    files = sources.open("s3://bucket/model")
    assert (files.version, files.read_bytes("uri.txt")) == ("v1", b"s3://bucket/model")
    for uri in ("/no/scheme", "ftp://host/model"):
        with pytest.raises(ValueError, match="no URI of a known scheme"):
            sources.open(uri)
    with pytest.raises(ValueError, match="two artifact source readers read one scheme"):
        ArtifactSources((MemoryReader(), S3Reader()))


class FakeS3Client:
    """An S3 client over ``objects``, keys to bytes; ``fail`` makes every call raise ``ClientError``."""

    def __init__(self, objects: dict[str, bytes], error: type[Exception]) -> None:
        self.objects = objects
        self.error = error
        self.fail = False
        self.downloads: list[str] = []

    def get_paginator(self, name: str) -> Any:
        client = self

        class Paginator:
            def paginate(self, **query: str) -> list[dict[str, Any]]:
                if client.fail:
                    raise client.error("access denied")
                keys = sorted(key for key in client.objects if key.startswith(query["Prefix"]))
                contents = [{"Key": key, "ETag": f'"{key}-etag"', "Size": len(client.objects[key])} for key in keys]
                return [{"Contents": contents}]

        return Paginator()

    def get_object(self, **request: str) -> dict[str, Any]:
        # boto3 names the request's fields Bucket and Key.
        self.downloads.append(request["Key"])
        return {"Body": types.SimpleNamespace(read=lambda: self.objects[request["Key"]])}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        if self.fail:
            raise self.error("access denied")
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeS3Client, list[str | None]]:
    class ClientError(Exception):
        pass

    client = FakeS3Client({}, ClientError)
    endpoints: list[str | None] = []
    boto3 = types.SimpleNamespace(client=lambda service, endpoint_url: endpoints.append(endpoint_url) or client)
    exceptions = types.SimpleNamespace(BotoCoreError=OSError, ClientError=ClientError)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", types.SimpleNamespace(exceptions=exceptions))
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    return client, endpoints


@pytest.mark.unit
def test_an_s3_uri_opens_every_object_under_its_prefix(tmp_path: Path, s3) -> None:
    client, endpoints = s3
    client.objects = {"models/a/config.json": b"{}", "models/a/sub/w.bin": b"w", "models/ab/other": b"x"}
    reader = S3Reader.from_config({"endpoint_url": "https://s3.example"})

    files = reader.open("s3://bucket/models/a")
    # Opening lists the prefix and downloads nothing; a check reads only the objects it needs.
    assert files.paths() == ("config.json", "sub/w.bin") and client.downloads == []
    assert files.read_bytes("config.json") == b"{}" and client.downloads == ["models/a/config.json"]
    assert tree(copied(files, tmp_path / "copy")) == ["config.json", "sub/w.bin"]
    assert endpoints == ["https://s3.example"]
    # The same objects give the same version.
    assert reader.open("s3://bucket/models/a/").version == files.version

    with pytest.raises(ValueError, match="holds no object"):
        reader.open("s3://bucket/empty")
    with pytest.raises(ValueError, match="names no bucket"):
        reader.open("s3:///models")
    client.fail = True
    with pytest.raises(ArtifactSourceError, match="access denied"):
        reader.open("s3://bucket/models/a")
    with pytest.raises(ArtifactSourceError, match="access denied"):
        copied(files, tmp_path / "failed")


@pytest.mark.unit
def test_an_s3_uri_skips_folder_markers(tmp_path: Path, s3) -> None:
    client, _ = s3
    # The S3 console's "Create folder" writes an empty object whose key ends in "/".
    client.objects = {
        "models/a/": b"",
        "models/a/config.json": b"{}",
        "models/a/sub/": b"",
        "models/a/sub/w.bin": b"w",
    }
    reader = S3Reader.from_config({})
    assert tree(copied(reader.open("s3://bucket/models/a"), tmp_path / "target")) == ["config.json", "sub/w.bin"]
    # A key that ends in "/" and holds data, or a key that is a file and a folder, names no tree of files.
    for extra, message in (
        ({"models/b/": b"data"}, "a key that names a folder"),
        ({"models/b/w": b"w", "models/b/w/z": b"z"}, "both a file and a folder"),
    ):
        client.objects = {"models/b/config.json": b"{}", **extra}
        with pytest.raises(ValueError, match=message):
            reader.open("s3://bucket/models/b")


@pytest.mark.unit
def test_a_hugging_face_revision_is_a_repo_and_a_full_commit() -> None:
    revision = HuggingFaceRevision("org/model", COMMIT)
    assert revision.uri == f"hf://org/model@{COMMIT}"
    assert HuggingFaceRevision.from_dict(revision.to_dict()) == revision
    for repo, commit in (
        ("model", COMMIT),
        ("org/model/x", COMMIT),
        ("org/model", "main"),
        ("org/model", COMMIT.upper()),
    ):
        with pytest.raises(ValueError):
            HuggingFaceRevision(repo, commit)
    for value in ({"uri": "s3://org/model", "version": COMMIT}, {"uri": "hf://org/model"}, []):
        with pytest.raises(ValueError, match=REFERENCE_FILE):
            HuggingFaceRevision.from_dict(value)


@pytest.mark.unit
def test_revision_files_are_the_hub_listing_and_download_only_what_is_read(tmp_path: Path) -> None:
    files = {"config.json": b"{}", "model.safetensors": b"weights-bytes", "sub/tokenizer.json": b"tokenizer-bytes"}
    hub = MemoryHub(tmp_path / "cache", "org/model", files)
    remote = HuggingFaceRevisionFiles("org/model", COMMIT, hub)
    local = write_tree(tmp_path / "local", files)
    assert remote.paths() == DirectoryFiles(local).paths()
    # The same bytes get the content ID of a copy; only the file below the LFS size is downloaded, to hash it.
    assert remote.content_id() == directory_content_id(local)
    assert hub.downloads == ["config.json"]
    assert remote.read_bytes("sub/tokenizer.json") == b"tokenizer-bytes"
    with pytest.raises(FileNotFoundError):
        remote.read_bytes("missing.json")
    with pytest.raises(ValueError, match="names no revision"):
        HuggingFaceRevisionFiles("org/other", COMMIT, hub)


@pytest.mark.unit
def test_a_commit_is_kept_by_reference_and_a_branch_is_copied_at_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = MemoryHub(tmp_path / "cache", "org/model", {"config.json": b"{}", "model.safetensors": b"weights-bytes"})
    monkeypatch.setattr("reef.artifact.sources.HfApi", lambda: hub)
    snapshot = write_tree(tmp_path / "snapshots" / COMMIT, hub.files)
    monkeypatch.setattr(
        "reef.artifact.sources.download_huggingface_snapshot", lambda source: DownloadedSnapshot(snapshot, COMMIT)
    )
    reader = HuggingFaceReader()

    pinned = reader.open(f"hf://org/model@{COMMIT}")
    assert pinned.version == COMMIT and pinned.reference is not None
    stored = tmp_path / "stored"
    stored.mkdir()
    for path, data in pinned.reference.items():
        (stored / path).write_bytes(data)
    assert read_reference(stored) == HuggingFaceRevision("org/model", COMMIT)

    # A branch can move, so it is copied, at the commit it named when it was opened.
    branch = reader.open("hf://org/model@main")
    assert branch.version == COMMIT and branch.reference is None
    assert directory_content_id(copied(branch, tmp_path / "copy")) == pinned.content_id()


@pytest.mark.unit
def test_a_reference_is_the_only_file_of_its_artifact(tmp_path: Path) -> None:
    stored = tmp_path / "stored"
    stored.mkdir()
    (stored / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision("org/model", COMMIT).to_dict()))
    # The files a repository adds to a materialized release do not hide the reference.
    (stored / "reef-artifact.json").write_text("{}")
    assert read_reference(stored) == HuggingFaceRevision("org/model", COMMIT)
    (stored / "config.json").write_text("{}")
    assert read_reference(stored) is None
    (stored / "config.json").unlink()
    (stored / REFERENCE_FILE).write_text("not json")
    with pytest.raises(ValueError, match="no JSON object"):
        read_reference(stored)
