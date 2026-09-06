"""Configuration boundaries, durable queues, and immutable operation snapshots."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.core.configuration import ConfigConflict, ConfigManager, PreparedConfigChange
from reef.dispatcher import build_default_dispatcher
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app
from reef.service.request_service import InferenceRetryPolicy, InferenceRetryTimeout, RequestService

from .test_harness_proposals import _dispatcher, _recipe


def validate(values):
    data = values["data"]
    if data["training_mode"] not in ("auto", "manual") or data["batch_size"] <= 0:
        raise ValueError("invalid configuration")


def test_manager_fifo_immutable_snapshots_conflicts_and_restart(tmp_path):
    path = tmp_path / "configuration.sqlite3"
    manager = ConfigManager(path)
    old = manager.register("s", {"data": {"training_mode": "auto", "batch_size": 2}}, validate)
    with pytest.raises(TypeError):
        old.values["data"]["batch_size"] = 99
    manager.submit("s", {"data": {"batch_size": 4}}, expected_revision=0)
    manager.submit("s", {"data": {"training_mode": "manual"}}, expected_revision=1)
    with pytest.raises(ConfigConflict):
        manager.submit("s", {"data": {"batch_size": 5}}, expected_revision=0)
    assert manager.snapshot("s").revision == 0
    manager.close()
    manager = ConfigManager(path)
    manager.register("s", {"data": {"training_mode": "auto", "batch_size": 100}}, validate)
    seen = []
    while manager.apply_next("s", lambda snapshot: PreparedConfigChange(lambda: seen.append(snapshot))):
        pass
    assert [snapshot.revision for snapshot in seen] == [1, 2]
    assert seen[0].values["data"] == {"training_mode": "auto", "batch_size": 4}
    assert seen[1].values["data"] == {"training_mode": "manual", "batch_size": 4}
    assert old.values["data"]["batch_size"] == 2
    manager.close()
    manager = ConfigManager(path)
    assert manager.snapshot("s").revision == 2
    assert [update["status"] for update in manager.status("s")["updates"]] == ["applied", "applied"]
    manager.close()


def test_failed_preparation_preserves_active_and_revalidates_next_patch():
    manager = ConfigManager()
    manager.register("s", {"data": {"training_mode": "auto", "batch_size": 2}}, validate)
    manager.submit("s", {"data": {"batch_size": 4}})
    manager.submit("s", {"data": {"training_mode": "manual"}})

    def fail(snapshot):
        raise RuntimeError("component could not prepare")

    assert manager.apply_next("s", fail)
    assert manager.snapshot("s").revision == 0
    assert manager.status("s")["updates"][0]["status"] == "failed"
    assert manager.apply_next("s", lambda snapshot: PreparedConfigChange(lambda: None))
    assert manager.snapshot("s").values["data"] == {"training_mode": "manual", "batch_size": 2}
    manager.close()


def test_persistence_failure_discards_candidate_without_activating(monkeypatch):
    manager = ConfigManager()
    manager.register("s", {"data": {"training_mode": "auto", "batch_size": 2}}, validate)
    manager.submit("s", {"data": {"batch_size": 4}})
    events = []

    def fail(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", fail)
        with pytest.raises(OSError, match="disk full"):
            manager.apply_next(
                "s",
                lambda snapshot: PreparedConfigChange(
                    lambda: events.append("activate"), lambda: events.append("discard")
                ),
            )
    assert events == ["discard"]
    assert manager.snapshot("s").revision == 0
    assert manager.status("s")["updates"][0]["status"] == "pending"
    manager.close()


def test_restart_changes_static_settings_without_overwriting_managed_values(tmp_path):
    path = tmp_path / "configuration.sqlite3"
    manager = ConfigManager(path)
    initial = {"data": {"training_mode": "auto", "batch_size": 2, "max_score": 0.0}}
    manager.register("s", initial, validate)
    manager.submit("s", {"data": {"batch_size": 4}})
    manager.apply_next("s", lambda snapshot: PreparedConfigChange(lambda: None))
    manager.close()
    manager = ConfigManager(path)
    snapshot = manager.register("s", initial, validate, restart_values={"data": {"max_score": 1.0}})
    assert snapshot.revision == 2
    assert snapshot.values["data"]["batch_size"] == 4
    assert snapshot.values["data"]["max_score"] == 1.0
    manager.submit("s", {"data": {"batch_size": 8}})
    manager.close()
    manager = ConfigManager(path)
    with pytest.raises(ConfigConflict, match="pending configuration updates"):
        manager.register("s", initial, validate, restart_values={"data": {"max_score": 2.0}})
    assert manager.snapshot("s").revision == 2
    assert manager.status("s")["updates"][-1]["status"] == "pending"
    manager.close()


def test_deployment_snapshot_updates_and_http_validation(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))
    service = RequestService(dispatcher, retry_policy=InferenceRetryPolicy(timeout_s=30))
    old = service._retry_snapshot()

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await client.post("/reef/config/updates", json={"reef": {"inference_retry_timeout_s": 60}})
            assert response.status == 202, await response.text()
            current = await (await client.get("/reef/config")).json()
            assert current["active_revision"] == 1
            assert current["active"]["reef"]["inference_retry_timeout_s"] == 60
            assert old.timeout_s == 30
            assert service._retry_snapshot().timeout_s == 60
            for patch in (
                {"reef": {"port": 9000}},
                {"reef": {"inference_retry_initial_s": 100}},
                {"reef": {"inference_retry_timeout_s": True}},
            ):
                response = await client.post("/reef/config/updates", json=patch)
                assert response.status == 400, await response.text()
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_inflight_inference_retries_keep_the_original_configuration():
    dispatcher = build_default_dispatcher()
    service = RequestService(dispatcher, retry_policy=InferenceRetryPolicy(initial_s=0.001, timeout_s=5))

    async def run():
        started, release = asyncio.Event(), asyncio.Event()

        class Backend(InferenceBackend):
            calls = 0

            async def inference(self, artifact, path, payload):
                self.calls += 1
                if self.calls == 1:
                    started.set()
                    await release.wait()
                    return {"choices": [{"finish_reason": "abort"}]}
                await asyncio.sleep(0.03)
                return {"choices": [{"message": {"content": "ok"}}]}

        backend = Backend()
        request = asyncio.create_task(
            service.infer({"x-reef-scenario": "s"}, {"messages": []}, "/v1/chat/completions", backend)
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            service.update_deployment_configuration({"reef": {"inference_retry_timeout_s": 0.001}})
            release.set()
            assert (await request)["choices"][0]["message"]["content"] == "ok"
            assert backend.calls == 2
            with pytest.raises(InferenceRetryTimeout):
                await service.infer({"x-reef-scenario": "s"}, {"messages": []}, "/v1/chat/completions", backend)
        finally:
            release.set()
            if not request.done():
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
