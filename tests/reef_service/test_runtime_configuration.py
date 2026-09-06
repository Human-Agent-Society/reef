"""Configuration boundaries, durable queues, and immutable operation snapshots."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.core.configuration import ConfigConflict, ConfigManager, PreparedConfigChange
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe.errors import RecipeConfigError
from reef.records import RecordConflict, RecordStore
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app
from reef.service.request_service import InferenceRetryPolicy, InferenceRetryTimeout, RequestService
from reef.train.cordis_backend.processor import RecordDrivenTraceProcessor

from .test_harness_proposals import _dispatcher, _recipe
from .test_manual_training import CaptureBackend, build, inference, instruction


def validate(values):
    data = values["data"]
    if data["training_mode"] not in ("auto", "manual") or data["batch_size"] <= 0:
        raise ValueError("invalid configuration")


def bind(manager, trainer, *, mode="auto", batch_size=2):
    trainer.bind_configuration(
        manager.register("scenario:s", {"data": {"training_mode": mode, "batch_size": batch_size}}, validate)
    )


def commit(trainer, result, *, compact=True):
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    if compact:
        trainer.apply_compaction(prepared.compacted_ids)
    return prepared


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


@pytest.mark.parametrize("dispatched", [False, True])
def test_update_waits_for_reserved_step_then_precedes_ready_batch(dispatched):
    records, backend, manager = RecordStore(), CaptureBackend(dispatched=dispatched), ConfigManager()
    trainer = build(records, backend, mode="auto", batch_size=2)
    bind(manager, trainer)
    try:
        for receipt in ("a", "b", "c", "d"):
            records.append(inference(receipt))
        if dispatched:
            trainer.reserve_training_batch()
            result = trainer.execute_reserved_step(0).result
        else:
            result = trainer.run_once()
        manager.submit("scenario:s", {"data": {"training_mode": "manual"}})
        trainer.apply_configuration(manager)
        assert manager.snapshot("scenario:s").revision == 0
        prepared = commit(trainer, result)
        assert prepared.metrics["config_revision"] == 0
        trainer.apply_configuration(manager)
        assert trainer.training_mode == "manual"
        assert manager.snapshot("scenario:s").revision == 1
        next_batch = trainer.reserve_training_batch() if dispatched else trainer.run_once()
        assert next_batch is None
        assert len(backend.batches) == 1
    finally:
        trainer.close()
        records.close()
        manager.close()


def test_batch_size_replay_and_consumed_audit_rows_do_not_train_again():
    records, backend, manager = RecordStore(), CaptureBackend(), ConfigManager()
    trainer = build(records, backend, mode="auto", batch_size=2)
    bind(manager, trainer)
    try:
        records.append(inference("a"))
        assert trainer.run_once() is None
        manager.submit("scenario:s", {"data": {"batch_size": 1}})
        trainer.apply_configuration(manager)
        result = trainer.run_once()
        prepared = commit(trainer, result, compact=False)
        assert prepared.metrics["config_revision"] == 1
        assert backend.batches[0].samples[0].source_agent_record_id == "a"
        manager.submit("scenario:s", {"data": {"batch_size": 3}})
        trainer.apply_configuration(manager)
        records.append(inference("b"))
        records.append(inference("c"))
        assert trainer.run_once() is None
        records.append(inference("d"))
        assert trainer.run_once() is not None
        assert [sample.source_agent_record_id for sample in backend.batches[1].samples] == ["b", "c", "d"]
    finally:
        trainer.close()
        records.close()
        manager.close()


def test_manual_transition_drains_instructions_then_restores_buffered_inferences():
    records, backend, manager = RecordStore(), CaptureBackend(), ConfigManager()
    trainer = build(records, backend, mode="manual", batch_size=2)
    bind(manager, trainer, mode="manual")
    try:
        records.append(inference("a"))
        records.append(instruction("one"))
        records.append(instruction("two"))
        manager.submit("scenario:s", {"data": {"training_mode": "auto", "batch_size": 1}})
        for request_id in ("one", "two"):
            trainer.apply_configuration(manager)
            assert trainer.training_mode == "manual"
            result = trainer.run_once()
            assert backend.batches[-1].request.id == request_id
            commit(trainer, result)
        trainer.apply_configuration(manager)
        assert trainer.training_mode == "auto"
        result = trainer.run_once()
        assert [sample.source_agent_record_id for sample in backend.batches[-1].samples] == ["a"]
        assert commit(trainer, result).metrics["config_revision"] == 1
    finally:
        trainer.close()
        records.close()
        manager.close()


def test_component_failure_keeps_previous_processor_and_snapshot(monkeypatch):
    records, backend, manager = RecordStore(), CaptureBackend(), ConfigManager()
    trainer = build(records, backend, mode="auto", batch_size=2)
    bind(manager, trainer)
    original = trainer.processor

    def fail(self, context):
        raise RuntimeError("cannot allocate processor")

    try:
        monkeypatch.setattr(RecordDrivenTraceProcessor, "prepare_reconfiguration", fail)
        manager.submit("scenario:s", {"data": {"batch_size": 1}})
        trainer.apply_configuration(manager)
        assert trainer.processor is original
        assert manager.snapshot("scenario:s").revision == 0
        assert manager.status("scenario:s")["updates"][0]["status"] == "failed"
    finally:
        trainer.close()
        records.close()
        manager.close()


def test_http_queues_during_training_and_applies_before_next_batch(tmp_path):
    started, release = Event(), Event()
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append((samples, requests))
        started.set()
        assert release.wait(10)

    recipe = replace(_recipe(tmp_path, propose), batch_policy="records", batch_size=1)
    dispatcher = _dispatcher(tmp_path, recipe)

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            dispatcher.accept_record(inference("a"))
            assert await asyncio.to_thread(started.wait, 5)
            dispatcher.accept_record(inference("b"))
            response = await client.post(
                "/reef/scenarios/s/config/updates",
                headers={"If-Match": '"0"'},
                json={"data": {"training_mode": "manual", "batch_size": 4}},
            )
            assert response.status == 202, await response.text()
            assert (await response.json())["status"] == "pending"
            current = await (await client.get("/reef/scenarios/s/config")).json()
            assert current["active_revision"] == 0
            assert current["revision"] == 1
            conflict = await client.post(
                "/reef/scenarios/s/config/updates", headers={"If-Match": '"0"'}, json={"data": {"batch_size": 3}}
            )
            assert conflict.status == 409
            release.set()
            for _ in range(500):
                current = await (await client.get("/reef/scenarios/s/config")).json()
                if current["active_revision"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert current["active"]["data"]["training_mode"] == "manual"
            assert current["active_revision"] == 1
            assert dispatcher.get_or_create_scenario("s").trainer.training_mode == "manual"
            assert len(seen) == 1
            for patch in ({"data": {"batch_size": 0}}, {"data": {"max_score": 0.2}}, {"port": 9000}):
                response = await client.post("/reef/scenarios/s/config/updates", json=patch)
                assert response.status == 400, await response.text()
            unknown = await client.get("/reef/scenarios/missing/config")
            assert unknown.status == 404
        finally:
            release.set()
            await client.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        dispatcher.close()


def test_applied_and_queued_configuration_recover_with_scenario(tmp_path):
    recipe = replace(_recipe(tmp_path, lambda n, s, m, requests=(): None), batch_policy="records", batch_size=2)
    first = _dispatcher(tmp_path, recipe)
    backend_factory = first._registry._backend_factory
    scenario = first.get_or_create_scenario("s")
    first.config_manager.submit("scenario:s", {"data": {"training_mode": "manual"}})
    scenario.trainer.apply_configuration(first.config_manager)
    first.config_manager.submit("scenario:s", {"data": {"batch_size": 5}})
    first.close()
    recovered = Dispatcher(recipe, backend_factory, agent_record_dir=tmp_path / "agent-record")
    try:
        scenario = recovered.get_or_create_scenario("s")
        assert scenario.trainer.training_mode == "manual"

        # The normal worker applies recovered pending updates without a fresh record.
        async def wait():
            for _ in range(500):
                if recovered.config_manager.snapshot("scenario:s").revision == 2:
                    return
                await asyncio.sleep(0.01)
            pytest.fail("recovered configuration queue was not drained")

        asyncio.run(wait())
        assert recovered.config_manager.snapshot("scenario:s").values["data"]["batch_size"] == 5
    finally:
        recovered.close()


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


def test_switch_away_from_manual_rejects_new_instructions_without_losing_accepted_ones(tmp_path):
    started, release = Event(), Event()
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append(requests[0]["id"])
        started.set()
        assert release.wait(10)

    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), training_mode="manual"))
    try:
        dispatcher.accept_record(instruction("one"))
        assert started.wait(5)
        dispatcher.accept_record(instruction("two"))
        dispatcher.update_scenario_configuration("s", {"data": {"training_mode": "auto"}})
        # Even if another update requests manual again, crossing the queued
        # auto boundary must not be starved by an endless stream of requests.
        dispatcher.update_scenario_configuration("s", {"data": {"training_mode": "manual"}})
        assert dispatcher.accept_record(instruction("one")).agent_record_id == "one"
        with pytest.raises(ConfigConflict, match="mode change"):
            dispatcher.accept_record(instruction("three"))
        release.set()

        async def wait():
            for _ in range(500):
                if dispatcher.config_manager.snapshot("scenario:s").revision == 2:
                    return
                await asyncio.sleep(0.01)
            pytest.fail("mode transition did not finish")

        asyncio.run(wait())
        assert seen == ["one", "two"]
        assert dispatcher.get_or_create_scenario("s").records.get("s", "three") is None
        dispatcher.update_scenario_configuration("s", {"data": {"training_mode": "auto"}})

        async def wait_auto():
            for _ in range(500):
                if dispatcher.config_manager.snapshot("scenario:s").revision == 3:
                    return
                await asyncio.sleep(0.01)
            pytest.fail("auto transition did not finish")

        asyncio.run(wait_auto())
        assert dispatcher.accept_record(instruction("one")).agent_record_id == "one"
        with pytest.raises(RecordConflict):
            dispatcher.accept_record(
                replace(instruction("one"), payload={"text": "changed", "session": "s", "release_id": "r"})
            )
        assert seen == ["one", "two"]
    finally:
        release.set()
        dispatcher.close()


def test_proposer_capability_is_validated_before_queueing(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))
    try:
        dispatcher.get_or_create_scenario("s")
        with pytest.raises(RecipeConfigError, match="requests"):
            dispatcher.update_scenario_configuration("s", {"data": {"training_mode": "manual"}})
        assert dispatcher.scenario_configuration("s")["updates"] == []
    finally:
        dispatcher.close()
