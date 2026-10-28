"""Artifact import tests: an import publishes beside the chain and changes nothing else (RFC #980)."""

from __future__ import annotations

import asyncio
import io
import tarfile
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import ImportingComponent, dispatcher, train
from tests.support.runtime_stubs import runtime_bindings

from reef.artifact import ArtifactSources, directory_content_id
from reef.core.errors import CommitPending
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.service.artifact_imports import ArtifactImports
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.jobs import JobLoopState
from reef.train.trainer import Trainer


def weights(directory: Path, content: bytes = b"w9") -> Path:
    """A full-weight release: the layout of ``StubTrainingComponent``'s releases."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(content)
    return directory


def archive(directory: Path) -> bytes:
    """``directory`` as a tar.gz archive whose root is the directory's."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            tar.add(path, arcname=path.relative_to(directory).as_posix(), recursive=False)
    return buffer.getvalue()


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
    source = {"uri": "file:///srv/weights", "version": ""}

    imported = value.import_artifact("math", weights(tmp_path / "w9"), source=source)

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
        {"source": source},
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
    assert (row["source"], row["active"], row["activated"]) == (source, False, False)

    # The same bytes again are another artifact with the same content id.
    again = value.import_artifact("math", weights(tmp_path / "again"))
    assert again.artifact_id != imported.artifact_id and again.content_id == imported.content_id
    assert value.list_artifacts("math")[0]["source"] is None

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
            value.import_artifact("math", missing_config)
        with pytest.raises(ValueError, match="has no component 'other'"):
            value.import_artifact("math", weights(tmp_path / "good"), "other")
        monkeypatch.setattr(Trainer, "has_pending_commit", property(lambda self: True))
        with pytest.raises(CommitPending, match="cannot import an artifact"):
            value.import_artifact("math", tmp_path / "good")
        assert len(scenario.store.history()) == steps
        assert value.list_artifacts("math") == rows
    finally:
        value.close()


@pytest.mark.unit
def test_the_import_http_api_takes_an_archive_or_a_uri(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        value.get_or_create_scenario("math")
        imports = ArtifactImports(ArtifactSources.from_setting(()), tmp_path / "imports")
        client = TestClient(TestServer(create_app(value, artifact_imports=imports)))
        await client.start_server()
        try:
            uploaded = await client.post(
                "/reef/scenarios/math/artifacts",
                data=archive(weights(tmp_path / "uploaded")),
                headers={"Content-Type": "application/gzip"},
            )
            assert uploaded.status == 201
            receipt = await uploaded.json()
            component = receipt["component"]
            assert receipt == {
                "scenario": "math",
                "component": component,
                "artifact_id": receipt["release_id"],
                "release_id": receipt["release_id"],
                "content_id": directory_content_id(tmp_path / "uploaded"),
            }
            source = weights(tmp_path / "source", b"w10").as_uri()
            fetched = await client.post(
                f"/reef/scenarios/math/artifacts?component={component}", json={"source": source}
            )
            assert fetched.status == 201
            listed = (await (await client.get("/reef/scenarios/math/artifacts")).json())["artifacts"]
            assert [(row["operation"], row["source"]) for row in listed[:2]] == [
                ("import", {"uri": source, "version": ""}),
                ("import", None),
            ]
            # Each import's temporary directory is gone once it answers.
            assert list((tmp_path / "imports").iterdir()) == []

            for body, headers, status in (
                (b"plain", {"Content-Type": "text/plain"}, 415),
                (b"not gzip", {"Content-Type": "application/gzip"}, 400),
                (b'{"uri": "file:///x"}', {"Content-Type": "application/json"}, 400),
                (b'{"source": "ftp://host/w"}', {"Content-Type": "application/json"}, 400),
            ):
                refused = await client.post("/reef/scenarios/math/artifacts", data=body, headers=headers)
                assert refused.status == status, await refused.text()
            escaping = io.BytesIO()
            with tarfile.open(fileobj=escaping, mode="w:gz") as tar:
                tar.add(tmp_path / "uploaded" / "config.json", arcname="../config.json")
            refused = await client.post(
                "/reef/scenarios/math/artifacts",
                data=escaping.getvalue(),
                headers={"Content-Type": "application/gzip"},
            )
            assert refused.status == 400 and "not a relative path" in await refused.text()
            missing = await client.post("/reef/scenarios/other/artifacts", json={"source": source})
            assert missing.status == 404
        finally:
            await client.close()
            value.close()

    asyncio.run(run())
