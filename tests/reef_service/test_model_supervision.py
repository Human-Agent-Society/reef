"""Recovery budget, readiness, cleanup and admission contracts without GPUs."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.ray_runtime import NamedRayTrainGroupHandle
from reef.runtime.sglang.service import RayHealthProbe
from reef.service.training_driver import ModelDeployment, supervise_deployment

from .test_model_deployment import plan_for
from .test_ray_runtime import FakeTrainGroupHandle


class FailedHealth:
    def poll(self):
        raise RuntimeError("component died")


class Stopping:
    def __init__(self, *, stop_at=100):
        self.waits = []
        self.stop_at = stop_at

    def wait(self, timeout):
        self.waits.append(timeout)
        return len(self.waits) >= self.stop_at

    def is_set(self):
        return len(self.waits) >= self.stop_at


class Source:
    def __init__(self, ready, *, error=None):
        self.ready = ready
        self.events = []
        self.error = error

    def create(self):
        assert not self.ready.exists()
        if self.error:
            raise self.error
        plan, events = plan_for()
        self.events.append(events)
        return replace(plan, health=FailedHealth())


def test_supervisor_rebuilds_fresh_plans_and_exhausts_restart_budget(tmp_path):
    ready = tmp_path / "ready"
    source = Source(ready)
    initial = ModelDeployment(source.create())
    initial.start()
    ready.write_text("ready")
    stopping = Stopping()
    with pytest.raises(RuntimeError, match="three restarts"):
        supervise_deployment(initial, source, ready, stopping)
    assert len(source.events) == 4
    assert stopping.waits == [1, 1, 1, 2, 1, 4, 1]
    assert not ready.exists()
    for events in source.events:
        assert events[-3:] == ["training-close", "inference-close", "release"]


@pytest.mark.parametrize("failure", ["release", "preflight", "stop"])
def test_recovery_requires_cleanup_and_preflight_and_obeys_stop(tmp_path, failure):
    ready = tmp_path / "ready"
    plan, events = plan_for(*(["release"] if failure == "release" else []))
    initial = ModelDeployment(replace(plan, health=FailedHealth()))
    initial.start()
    ready.write_text("ready")
    source = Source(ready, error=ValueError("ambiguous RUNNING optimizer step"))
    stopping = Stopping(stop_at=2 if failure == "stop" else 100)
    if failure == "stop":
        assert supervise_deployment(initial, source, ready, stopping) is initial
    else:
        with pytest.raises((RuntimeError, ValueError), match=r"release|ambiguous RUNNING"):
            supervise_deployment(initial, source, ready, stopping)
    assert source.events == []
    assert events[-1] == "release"
    assert not ready.exists()


def test_pending_health_probe_is_not_failure_and_does_not_queue_more_work(monkeypatch):
    from reef.train.slime_backend import resources

    submitted = []
    actor = SimpleNamespace(health=SimpleNamespace(remote=lambda: submitted.append("probe") or "reference"))
    monkeypatch.setattr(resources.ray, "wait", lambda refs, timeout: ([], refs))
    probe = RayHealthProbe()
    for _ in range(100):
        probe.poll(actor, "health")
    assert submitted == ["probe"]
    monkeypatch.setattr(resources.ray, "wait", lambda refs, timeout: (refs, []))
    monkeypatch.setattr(resources.ray, "get", lambda ref: {"ok": False, "recoverable": True})
    probe.poll(actor, "health")
    monkeypatch.setattr(resources.ray, "get", lambda ref: {"ok": False})
    with pytest.raises(RuntimeError, match="health check"):
        probe.poll(actor, "health")
    assert submitted == ["probe", "probe"]


class ReconnectingHandle(FakeTrainGroupHandle):
    reconnects = True
    endpoint = "http://first"
    status = "IDLE"

    def health(self):
        return {**super().health(), "ok": True, "inference_url": self.endpoint}


def test_managed_endpoint_refresh_preserves_backend_and_never_reopens_commit_gate():
    async def run():
        handle = ReconnectingHandle()
        runtime = ExecutorTrainingRuntime(train_group_handle=handle)
        backend = runtime.inference_backend
        handle.endpoint = "http://replacement/"
        admission = await runtime.acquire_inference()
        admission.release()
        assert runtime.base_url == "http://replacement"
        assert runtime.inference_backend is backend
        assert backend._upstream_url == "http://replacement"
        runtime._inference_admission.close()
        request = asyncio.create_task(runtime.acquire_inference())
        await asyncio.sleep(0.05)
        assert not request.done()
        runtime._inference_admission.open()
        (await request).release()
        handle.status = "READY_TO_COMMIT"
        request = asyncio.create_task(runtime.acquire_inference())
        await asyncio.sleep(0.05)
        assert not request.done()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

    asyncio.run(run())


def test_explicit_endpoint_is_not_replaced():
    handle = ReconnectingHandle()
    runtime = ExecutorTrainingRuntime(train_group_handle=handle, inference_url="http://gateway")
    handle.endpoint = "http://replacement"
    runtime._training_job_status()
    assert runtime.base_url == "http://gateway"


def test_named_handle_rediscovers_and_never_replays_a_submitted_write(monkeypatch):
    from reef.runtime.adapters import ray_runtime

    class ActorDied(Exception):
        pass

    state = SimpleNamespace(actor="old", calls=[], timeouts=[], fail=False)
    ray = SimpleNamespace(
        get_actor=lambda *args, **kwargs: state.actor,
        exceptions=SimpleNamespace(RayActorError=ActorDied),
    )

    class Executor:
        @classmethod
        def from_workers(cls, workers, **kwargs):
            return cls(workers[0])

        def __init__(self, actor):
            self.actor = actor

        def rpc(self, rank, method, **kwargs):
            state.calls.append((self.actor, method))
            state.timeouts.append(kwargs["timeout"])
            if state.fail:
                state.actor = "replacement"
                raise ActorDied("write may have completed")
            return {"training_job": {}}

        def shutdown(self):
            pass

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: ray)
    monkeypatch.setattr(ray_runtime, "RayExecutor", Executor)
    handle = NamedRayTrainGroupHandle("bridge", "test", timeout_s=7200, health_timeout_s=30)
    handle.health()
    state.actor = "new"
    handle.health()
    state.fail = True
    with pytest.raises(ActorDied):
        handle.execute_training_job({"job_id": "one"})
    assert state.calls == [("old", "health"), ("new", "health"), ("new", "execute_training_job")]
    assert 0 < state.timeouts[0] <= 30 and 0 < state.timeouts[1] <= 30
    assert state.timeouts[2] > 7000
    handle.shutdown()
