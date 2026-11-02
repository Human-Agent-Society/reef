"""The inference worker loads selected artifacts and publishes routes after startup completes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from reef.artifact.artifact import Artifact
from reef.artifact.repository import Repository, RepositoryBackendFactory
from reef.core.served_models import ACTIVE, CACHE_SECONDS, ScenarioModel, ScheduledStop, ServedArtifact
from reef.core.values import require_text
from reef.execution.workers import QueueWorker, WorkerStopped
from reef.recipe.base import Recipe
from reef.runtime.interfaces import InferenceRuntime
from reef.storage.coordination import Coordination
from reef.storage.served_models import INFERENCE_OWNER_QUEUE, SERVING_KIND, SERVING_QUEUE, STOP_SERVING_KIND
from reef.storage.work_queue import JsonValue, LeaseExtender, NewWork, WorkItem, WorkLeaseLost, WorkQueue


class ServingLease(LeaseExtender):
    """Renew one inference owner's work lease and stop its runtime after lease loss.

    ``runtime`` is the owner's runtime. ``stopped`` stops its serving worker.
    Only an inference-owner item is accepted. A lost lease ends ownership permanently for this runtime instance.
    Engine calls already in progress must still return before their thread can stop.
    """

    def __init__(
        self, queue: WorkQueue, item: WorkItem, worker_id: str, runtime: InferenceRuntime, stopped: threading.Event
    ) -> None:
        if item.queue != INFERENCE_OWNER_QUEUE or item.kind != INFERENCE_OWNER_QUEUE:
            raise ValueError("a serving lease needs inference ownership work")
        super().__init__(queue, item, worker_id)
        self.runtime = runtime
        self.stopped = stopped

    def run(self) -> None:
        super().run()
        if self.lease_lost.is_set():
            self.stopped.set()
            self.runtime.shutdown()


class ServingWorker(QueueWorker):
    """Start queued releases and retire unused endpoints in the inference service.

    ``recipe`` supplies the runtime and each scenario's surface. ``backends`` resolves durable releases.
    ``coordination`` owns requests and routes. ``local_artifact_dir`` holds materialized releases, or uses the repository default.
    A long-lived work lease gives one worker runtime ownership, including health checks and idle periods.
    Other replicas only read routes. The owner recovers completed selections before it takes new work.
    ``owner`` is None before ownership. ``recovery_routes`` is None before recovery, then holds routes to adopt.
    ``opened`` becomes true after the runtime opens and adopts these routes.
    """

    def __init__(
        self,
        recipe: Recipe,
        backends: RepositoryBackendFactory,
        coordination: Coordination,
        *,
        worker_id: str,
        local_artifact_dir: Path | None = None,
    ) -> None:
        super().__init__(
            coordination.work_queue, SERVING_QUEUE, (SERVING_KIND, STOP_SERVING_KIND), worker_id=worker_id
        )
        self.recipe = recipe
        self.backends = backends
        self.routes = coordination.served_models
        self.local_artifact_dir = local_artifact_dir
        runtime = recipe.runtime
        self.stop_delay_seconds = CACHE_SECONDS + (300.0 if runtime is None else runtime.inference_timeout_s)
        self.opened = False
        self.owner: ServingLease | None = None
        self.recovery_routes: tuple[ServedArtifact, ...] | None = None

    def stop(self) -> None:
        super().stop()
        if self.owner is not None:
            try:
                if not self.owner.lease_lost.is_set() and self.opened:
                    self.owner.runtime.shutdown()
            finally:
                self.owner.stop()
            with contextlib.suppress(WorkLeaseLost):
                self.queue.finish(INFERENCE_OWNER_QUEUE, self.owner.item.id, self.worker_id, {})
            self.owner = None

    def run_once(self) -> bool:
        if self.stopped.is_set():
            return False
        if self.owner is None:
            runtime = self.recipe.runtime
            if runtime is None or not runtime.serves_requests:
                return False
            self.queue.requeue_expired()
            self.queue.request(INFERENCE_OWNER_QUEUE, "inference", INFERENCE_OWNER_QUEUE, INFERENCE_OWNER_QUEUE, {})
            item = self.queue.take(INFERENCE_OWNER_QUEUE, self.worker_id, [INFERENCE_OWNER_QUEUE])
            if item is None:
                return False
            self.queue.start(INFERENCE_OWNER_QUEUE, item.id, self.worker_id)
            self.owner = ServingLease(self.queue, item, self.worker_id, runtime, self.stopped)
            self.owner.__enter__()
        if self.owner.lease_lost.is_set():
            return False
        if self.recovery_routes is None:
            previous = tuple(self.routes.all_served())
            self.routes.recover_completed(self.owner.item, self.worker_id, stop_delay_seconds=self.stop_delay_seconds)
            self.recovery_routes = previous
        for stop in self.routes.due_stops(time.time()):
            payload: dict[str, JsonValue] = {"served": dict(stop.served.to_json()), "stop_after": stop.stop_after}
            identifier = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            self.queue.add(
                NewWork(
                    SERVING_QUEUE,
                    f"stop:{identifier}",
                    "serving",
                    STOP_SERVING_KIND,
                    payload=payload,
                    serial_key=SERVING_QUEUE,
                )
            )
        return super().run_once()

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        if self.stopped.is_set():
            raise WorkerStopped
        runtime = self.recipe.runtime
        if runtime is None:
            raise ValueError("the recipe has no inference runtime")
        if self.owner is None or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost runtime ownership")
        if not self.opened:
            runtime.open()
            for served in (*(self.recovery_routes or ()), *self.routes.all_served()):
                runtime.adopt(served)
            self.opened = True
            self.recovery_routes = ()
        if lease_lost.is_set() or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost its lease")
        if item.kind == STOP_SERVING_KIND:
            served = ServedArtifact.from_json(item.payload["served"])
            due = any(
                stop.served == served and stop.stop_after == item.payload["stop_after"]
                for stop in self.routes.due_stops(time.time())
            )
            if due and not self.routes.is_referenced(served):
                runtime.stop_serving(served)
            return {}
        selected = self.routes.current_request(item.scenario)
        if selected is None or selected.id != item.id or not item.payload:
            return {}
        artifact_id = require_text(item.payload.get("artifact_id"), "artifact_id")
        release_id = require_text(item.payload.get("release_id"), "release_id")
        backend = self.backends(item.scenario)
        reference = backend.resolve_release(release_id)
        repository = Repository(backend, reference, local_dir=self.local_artifact_dir)
        artifact = Artifact(reference, repository)
        surface = self.recipe.serving_surface(item.scenario)
        current_route = self.routes.read(ScenarioModel(item.scenario))
        if current_route is None or current_route.artifact_id != artifact_id:
            surface.activate(artifact, runtime)
        if lease_lost.is_set() or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost its lease")
        active = runtime.served_artifact(artifact_id, item.scenario)
        if active is None:
            raise ValueError("the inference runtime provides no route for the selected artifact")
        models = {ACTIVE: active}
        aliases = item.payload.get("aliases")
        beside = item.payload.get("beside")
        if not isinstance(aliases, list) or not isinstance(beside, dict):
            raise ValueError("a serving request needs aliases and beside artifacts")
        for name in aliases:
            models[require_text(name, "model name")] = active
        for name, value in beside.items():
            if not isinstance(value, dict):
                raise ValueError("a beside artifact needs its release and component")
            component = require_text(value.get("component"), "component")
            if component == surface.loader_component:
                ref = backend.resolve_release(require_text(value.get("release_id"), "release_id"))
                beside_model = runtime.served_artifact_on_request(
                    surface.component_artifact(Artifact(ref, repository), component)
                )
                if beside_model is None:
                    raise ValueError("the inference runtime provides no route for a beside artifact")
                models[name] = beside_model
            else:
                models[name] = active
        return {name: dict(served.to_json()) for name, served in models.items()}

    def finish_item(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        if self.owner is None or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost runtime ownership")
        if item.kind == STOP_SERVING_KIND:
            moment = item.payload["stop_after"]
            if not isinstance(moment, (int, float)):
                raise ValueError("stop_after must be a time in seconds")
            stop = ScheduledStop(ServedArtifact.from_json(item.payload["served"]), moment)
            self.routes.complete_stop(item, self.worker_id, stop, owner=self.owner.item)
        else:
            self.routes.complete_request(
                item,
                self.worker_id,
                {name: ServedArtifact.from_json(value) for name, value in result.items()},
                stop_delay_seconds=self.stop_delay_seconds,
                owner=self.owner.item,
            )


__all__ = ["ServingWorker"]
