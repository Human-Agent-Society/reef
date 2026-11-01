"""Artifact import tests: an import publishes beside the chain and changes nothing else (RFC #980)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import ImportingComponent, dispatcher, train
from tests.support.hugging_face_hub import COMMIT, MemoryHub
from tests.support.runtime_stubs import runtime_bindings

from reef.artifact import (
    REFERENCE_FILE,
    ArtifactSourceReader,
    ArtifactSources,
    DirectorySource,
    SourceFiles,
    directory_content_id,
    read_reference,
)
from reef.core.errors import CommitPending, UnknownScenario
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.service.artifact_imports import ArtifactImports
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.jobs import JobLoopState
from reef.train.trainer import Trainer


def weights(directory: Path, content: bytes = b"w9") -> Path:
    """A full-weight release: the layout of ``StubTrainingComponent``'s releases."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(content)
    return directory


def source(directory: Path) -> dict[str, str]:
    """The source of an import read from ``directory`` by the ``file`` reader."""
    return {"uri": directory.as_uri(), "version": ""}


def opened(directory: Path) -> SourceFiles:
    """The files of ``directory`` as the ``file`` reader opens them."""
    return DirectorySource(directory, source_version="", link=False)


def store(tmp_path: Path) -> Path:
    """A new empty directory for an import to store into."""
    stores = tmp_path / "stores"
    stores.mkdir(exist_ok=True)
    directory = stores / str(len(list(stores.iterdir())))
    directory.mkdir()
    return directory


@pytest.mark.unit
def test_an_import_publishes_beside_the_chain_and_changes_nothing_else(tmp_path) -> None:
    value, runtime, backend_factory = dispatcher(tmp_path)
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    served = scenario.current_artifact_ref()
    checkpoint = scenario.repository.checkpoint_artifact
    trainer_state = dict(scenario.trainer.state)
    status = value.build_training_status()["scenarios"]["math"]["artifacts"]
    served_before = len(runtime.activated)
    runtime.restore_order.clear()
    imported = value.import_artifact(
        "math", opened(weights(tmp_path / "w9")), None, source=source(tmp_path / "w9"), directory=store(tmp_path)
    )

    # Nothing serves, trains from or moves to the artifact: the active release, the chain tip and the start stay.
    assert scenario.current_artifact_ref() == served
    assert scenario.repository.checkpoint_artifact == checkpoint
    assert scenario.trainer.state == trainer_state
    assert value.build_training_status()["scenarios"]["math"]["artifacts"] == status
    assert runtime.activated[served_before:] == [] and runtime.restore_order == []
    record = scenario.store.history()[-1]
    assert (record.operation, record.checkpoint, record.component, record.metrics) == (
        "import",
        True,
        None,
        {"source": source(tmp_path / "w9")},
    )
    assert record.artifact_ref.release_id == imported.release_id
    assert record.artifact_ref.parent_release_id == checkpoint.release_id
    row = value.list_artifacts("math")[0]
    assert (row["operation"], row["artifact_id"], row["release_id"], row["content_id"]) == (
        "import",
        imported.artifact_id,
        imported.release_id,
        imported.content_id,
    )
    assert imported.artifact_id == imported.release_id
    assert imported.content_id == directory_content_id(tmp_path / "w9")
    assert (row["source"], row["active"], row["activated"]) == (source(tmp_path / "w9"), False, False)

    # The same bytes again are another artifact with the same content id.
    again = value.import_artifact(
        "math", opened(weights(tmp_path / "again")), None, source=source(tmp_path / "again"), directory=store(tmp_path)
    )
    assert again.artifact_id != imported.artifact_id and again.content_id == imported.content_id

    # The artifact is then an ordinary row: it serves and trains as any other.
    activated = value.activate("math", imported.artifact_id)
    assert scenario.current_artifact_ref().release_id == imported.release_id == activated.release_id
    value.train_from("math", imported.artifact_id)
    assert JobLoopState.from_dict(scenario.trainer.state).start == str(runtime.imported[-1])
    value.close()

    # A restart keeps the chain tip at the last training step and lists the imports.
    component = ImportingComponent(tmp_path / "restarted-export")
    restarted = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, component.inference)),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
    )
    try:
        scenario = restarted.get_or_create_scenario("math")
        assert scenario.repository.checkpoint_artifact == checkpoint
        assert scenario.current_artifact_ref().release_id == imported.release_id
        assert [row["operation"] for row in restarted.list_artifacts("math")] == [
            "import",
            "import",
            "training",
            "creation",
        ]
        # It serves the activated import and resumes training from the imported start.
        assert sorted(component.restore_order) == ["inference", "training"]
    finally:
        restarted.close()


@pytest.mark.unit
def test_a_refused_import_changes_nothing(tmp_path, monkeypatch) -> None:
    value, _, _ = dispatcher(tmp_path)
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    steps = len(scenario.store.history())
    rows = value.list_artifacts("math")
    missing_config = weights(tmp_path / "bad")
    (missing_config / "config.json").unlink()
    try:
        with pytest.raises(ValueError, match=r"config\.json and weight files"):
            value.import_artifact(
                "math", opened(missing_config), None, source=source(missing_config), directory=store(tmp_path)
            )
        with pytest.raises(ValueError, match="has no component 'other'"):
            value.import_artifact(
                "math",
                opened(weights(tmp_path / "good")),
                "other",
                source=source(tmp_path / "good"),
                directory=store(tmp_path),
            )
        monkeypatch.setattr(Trainer, "has_pending_commit", property(lambda self: True))
        with pytest.raises(CommitPending, match="cannot import an artifact"):
            value.import_artifact(
                "math", opened(tmp_path / "good"), None, source=source(tmp_path / "good"), directory=store(tmp_path)
            )
        assert len(scenario.store.history()) == steps
        assert value.list_artifacts("math") == rows
    finally:
        value.close()


class CountingReader(ArtifactSourceReader):
    """Opens ``mem://`` URIs as the full-weight release in ``directory`` and keeps each URI it opened."""

    def __init__(self, directory: Path) -> None:
        self.directory = weights(directory)
        self.read: list[str] = []

    @property
    def schemes(self) -> tuple[str, ...]:
        return ("mem",)

    def open(self, uri: str) -> SourceFiles:
        self.read.append(uri)
        return DirectorySource(self.directory, source_version="v1", link=False)


@pytest.mark.unit
def test_an_import_the_scenario_cannot_take_reads_nothing(tmp_path) -> None:
    value, _, _ = dispatcher(tmp_path)
    value.get_or_create_scenario("math")
    reader = CountingReader(tmp_path / "mem")
    imports = ArtifactImports(ArtifactSources((reader,)), tmp_path / "imports")
    try:
        with pytest.raises(ValueError, match="has no component 'wieghts'"):
            imports.import_source(value, "math", "wieghts", "mem://model")
        with pytest.raises(UnknownScenario):
            imports.import_source(value, "other", None, "mem://model")
        assert reader.read == []
        assert imports.import_source(value, "math", None, "mem://model")["scenario"] == "math"
        assert reader.read == ["mem://model"]
    finally:
        value.close()


@pytest.mark.unit
def test_the_import_http_api_takes_a_uri(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        value.get_or_create_scenario("math")
        imports = ArtifactImports(ArtifactSources.from_setting(()), tmp_path / "imports")
        client = TestClient(TestServer(create_app(value, artifact_imports=imports)))
        await client.start_server()
        try:
            uri = weights(tmp_path / "source").as_uri()
            imported = await client.post("/reef/scenarios/math/artifacts", json={"source": uri})
            assert imported.status == 201
            receipt = await imported.json()
            assert receipt == {
                "scenario": "math",
                "component": receipt["component"],
                "artifact_id": receipt["release_id"],
                "release_id": receipt["release_id"],
                "content_id": directory_content_id(tmp_path / "source"),
            }
            listed = (await (await client.get("/reef/scenarios/math/artifacts")).json())["artifacts"]
            assert (listed[0]["operation"], listed[0]["source"]) == ("import", {"uri": uri, "version": ""})
            named = await client.post(
                f"/reef/scenarios/math/artifacts?component={receipt['component']}", json={"source": uri}
            )
            assert named.status == 201
            # Each import's temporary directory is gone once it answers.
            assert list((tmp_path / "imports").iterdir()) == []

            missing_config = weights(tmp_path / "bad")
            (missing_config / "config.json").unlink()
            for body, message in (
                (b"not json", "Expecting value"),
                (b'{"uri": "file:///x"}', "an import is the object"),
                (b'{"source": "ftp://host/w"}', "no URI of a known scheme"),
                (b'{"source": "hf://model"}', "org/model syntax"),
                (b'{"source": "file:///missing"}', "absolute directory"),
                (f'{{"source": "{missing_config.as_uri()}"}}'.encode(), "config.json and weight files"),
            ):
                refused = await client.post("/reef/scenarios/math/artifacts", data=body)
                assert refused.status == 400 and message in await refused.text()
            # An unknown scenario is refused before the source is read.
            unknown = await client.post("/reef/scenarios/other/artifacts", json={"source": "file:///missing"})
            assert unknown.status == 404
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_import_of_a_commit_keeps_a_reference_and_downloads_no_weights(tmp_path, monkeypatch) -> None:
    hub = MemoryHub(tmp_path / "cache", "org/model", {"config.json": b"{}", "model.safetensors": b"w9-weights"})
    monkeypatch.setattr("reef.artifact.sources.HfApi", lambda: hub)

    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        value.get_or_create_scenario("math")
        imports = ArtifactImports(ArtifactSources.from_setting(()), tmp_path / "imports")
        client = TestClient(TestServer(create_app(value, artifact_imports=imports)))
        await client.start_server()
        try:
            uri = f"hf://org/model@{COMMIT}"
            imported = await client.post("/reef/scenarios/math/artifacts", json={"source": uri})
            assert imported.status == 201
            receipt = await imported.json()
            # The content ID is the copy's; the layout check read no weights.
            copy = tmp_path / "copy"
            copy.mkdir()
            for path, data in hub.files.items():
                (copy / path).write_bytes(data)
            assert receipt["content_id"] == directory_content_id(copy)
            assert "model.safetensors" not in hub.downloads
            listed = (await (await client.get("/reef/scenarios/math/artifacts")).json())["artifacts"]
            assert listed[0]["source"] == {"uri": uri, "version": COMMIT}
            # Reef stores the reference only.
            _, artifact = value.get_or_create_scenario("math").resolve_artifact(receipt["artifact_id"])
            stored = artifact.materialize().local_path
            assert read_reference(stored) is not None and (stored / REFERENCE_FILE).is_file()
            assert not (stored / "model.safetensors").exists()

            # The revision must have the layout of the component's releases, and resolve.
            del hub.files["config.json"]
            refused = await client.post("/reef/scenarios/math/artifacts", json={"source": uri})
            assert refused.status == 400 and "config.json and weight files" in await refused.text()
            missing = await client.post("/reef/scenarios/math/artifacts", json={"source": f"hf://org/other@{COMMIT}"})
            assert missing.status == 400 and "names no revision" in await missing.text()
        finally:
            await client.close()
            value.close()

    asyncio.run(run())
