"""The source readers of artifact imports and the content ID of what they import (RFC #980, RFC #995)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
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
    FileReader,
    HuggingFaceReader,
    HuggingFaceRevision,
    HuggingFaceRevisionFiles,
    S3Reader,
    directory_content_id,
    read_reference,
)


class MemoryReader(ArtifactSourceReader):
    """Reads ``mem://`` and ``s3://`` URIs into one file named by the URI."""

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("mem", "s3")

    def fetch(self, uri: str, directory: Path) -> str:
        (directory / "uri.txt").write_text(uri)
        return "v1"


def write_tree(directory: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(content)
    return directory


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
def test_a_file_uri_copies_an_absolute_directory_without_a_version(tmp_path: Path) -> None:
    source = write_tree(tmp_path / "source", {"config.json": b"{}", "sub/weights.bin": b"w"})
    target = tmp_path / "target"
    target.mkdir()
    assert ArtifactSources.from_setting(()).reader(source.as_uri()).fetch(source.as_uri(), target) == ""
    assert directory_content_id(target) == directory_content_id(source)
    for uri in ("file://relative/path", (tmp_path / "missing").as_uri()):
        with pytest.raises(ValueError, match="absolute directory"):
            FileReader().fetch(uri, target)


@pytest.mark.unit
def test_a_uri_goes_to_the_reader_of_its_scheme(tmp_path: Path) -> None:
    sources = ArtifactSources.from_setting([f"{__name__}:MemoryReader"])
    # A configured reader replaces the built-in reader of a scheme it reads.
    assert sorted(type(reader).__name__ for reader in sources.readers) == [
        "FileReader",
        "HuggingFaceReader",
        "MemoryReader",
    ]
    assert isinstance(sources.reader("s3://bucket/model"), MemoryReader)
    assert sources.reader("s3://bucket/model").fetch("s3://bucket/model", tmp_path) == "v1"
    assert (tmp_path / "uri.txt").read_text() == "s3://bucket/model"
    for uri in ("/no/scheme", "ftp://host/model"):
        with pytest.raises(ValueError, match="no URI of a known scheme"):
            sources.reader(uri)
    with pytest.raises(ValueError, match="two artifact source readers read one scheme"):
        ArtifactSources((MemoryReader(), S3Reader()))


class FakeS3Client:
    """An S3 client over ``objects``, keys to bytes; ``fail`` makes every listing raise ``ClientError``."""

    def __init__(self, objects: dict[str, bytes], error: type[Exception]) -> None:
        self.objects = objects
        self.error = error
        self.fail = False

    def get_paginator(self, name: str) -> Any:
        client = self

        class Paginator:
            def paginate(self, **query: str) -> list[dict[str, Any]]:
                if client.fail:
                    raise client.error("access denied")
                keys = sorted(key for key in client.objects if key.startswith(query["Prefix"]))
                return [{"Contents": [{"Key": key, "ETag": f'"{key}-etag"'} for key in keys]}]

        return Paginator()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])


@pytest.mark.unit
def test_an_s3_uri_reads_every_object_under_its_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class ClientError(Exception):
        pass

    client = FakeS3Client(
        {"models/a/config.json": b"{}", "models/a/sub/w.bin": b"w", "models/ab/other": b"x"}, ClientError
    )
    endpoints = []
    boto3 = types.SimpleNamespace(client=lambda service, endpoint_url: endpoints.append(endpoint_url) or client)
    exceptions = types.SimpleNamespace(BotoCoreError=OSError, ClientError=ClientError)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", types.SimpleNamespace(exceptions=exceptions))
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    reader = S3Reader.from_config({"endpoint_url": "https://s3.example"})

    first = tmp_path / "first"
    first.mkdir()
    version = reader.fetch("s3://bucket/models/a", first)
    assert sorted(path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()) == [
        "config.json",
        "sub/w.bin",
    ]
    assert endpoints == ["https://s3.example"]
    # The same objects give the same version.
    second = tmp_path / "second"
    second.mkdir()
    assert reader.fetch("s3://bucket/models/a/", second) == version

    with pytest.raises(ValueError, match="holds no object"):
        reader.fetch("s3://bucket/empty", tmp_path)
    with pytest.raises(ValueError, match="names no bucket"):
        reader.fetch("s3:///models", tmp_path)
    client.fail = True
    with pytest.raises(ArtifactSourceError, match="access denied"):
        reader.fetch("s3://bucket/models/a", tmp_path)


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
    remote = HuggingFaceRevisionFiles(HuggingFaceRevision("org/model", COMMIT), hub)
    local = write_tree(tmp_path / "local", files)
    assert remote.paths() == DirectoryFiles(local).paths()
    # The same bytes get the content ID of a copy; only the file below the LFS size is downloaded, to hash it.
    assert remote.content_id() == directory_content_id(local)
    assert hub.downloads == ["config.json"]
    assert remote.read_bytes("sub/tokenizer.json") == b"tokenizer-bytes"
    with pytest.raises(FileNotFoundError):
        remote.read_bytes("missing.json")
    with pytest.raises(ValueError, match="names no revision"):
        HuggingFaceRevisionFiles(HuggingFaceRevision("org/other", COMMIT), hub)


@pytest.mark.unit
def test_the_hugging_face_reader_keeps_a_commit_by_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = MemoryHub(tmp_path / "cache", "org/model", {"config.json": b"{}", "model.safetensors": b"weights-bytes"})
    monkeypatch.setattr("reef.artifact.sources.HfApi", lambda: hub)
    reader = HuggingFaceReader()
    branch = tmp_path / "branch"
    branch.mkdir()
    # A branch or a tag can move, so it is copied.
    assert reader.keep_by_reference("hf://org/model@main", branch) is None
    assert list(branch.iterdir()) == []

    stored = tmp_path / "stored"
    stored.mkdir()
    kept = reader.keep_by_reference(f"hf://org/model@{COMMIT}", stored)
    assert kept is not None and kept.version == COMMIT
    assert kept.files.content_id() == directory_content_id(write_tree(tmp_path / "copy", hub.files))
    assert [path.name for path in stored.iterdir()] == [REFERENCE_FILE]
    assert read_reference(stored) == HuggingFaceRevision("org/model", COMMIT)
    # The files a repository adds to a materialized release do not hide the reference.
    (stored / "reef-artifact.json").write_text("{}")
    assert read_reference(stored) == HuggingFaceRevision("org/model", COMMIT)
    (stored / "config.json").write_text("{}")
    assert read_reference(stored) is None
    (stored / "config.json").unlink()
    (stored / REFERENCE_FILE).write_text("not json")
    with pytest.raises(ValueError, match="no JSON object"):
        read_reference(stored)
