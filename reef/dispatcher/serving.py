"""The inference worker loads selected artifacts and publishes routes after startup completes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

from reef.artifact.artifact import Artifact
from reef.artifact.repository import Repository, RepositoryBackendFactory
from reef.core.artifact_selection import ArtifactSelection
from reef.core.served_models import ACTIVE, CACHE_SECONDS, ScheduledStop, ServedArtifact
from reef.core.values import require_text
from reef.execution.workers import ItemFailed, QueueWorker, WorkerStopped
from reef.inference.interfaces import InferenceRuntime
from reef.inference.startup import (
    PROGRESS_KEY,
    STARTUP_FAILED_PREFIX,
    StartupFailed,
    StartupReporter,
    StartupStopped,
    redeploying,
    reporting_startup,
    serving_work,
)
from reef.recipe.base import Recipe
from reef.scenario.binding import activate_snapshot
from reef.storage.coordination import Coordination
from reef.storage.served_models import (
    INFERENCE_OWNER_QUEUE,
    REDEPLOY,
    SERVE_ARTIFACT_KIND,
    SERVING_KIND,
    SERVING_QUEUE,
    STOP_SERVING_KIND,
)
from reef.storage.work_queue import JsonValue, LeaseExtender, NewWork, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)
#: The error of serving work that a newer request replaced; the queue gives queued work the same error.
REPLACED = "a newer serving request replaced this work"
#: How often held serving work looks whether a newer request replaced it.
REPLACED_CHECK_SECONDS = 2.0


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


def serving_config_of(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """The override of the served component that a serving request carries (RFC #1097); ``{}`` when it carries
    none, so the scenario serves with the deployment's values."""
    value = payload.get("serving_config", {})
    if not isinstance(value, dict):
        raise ValueError("a serving request's serving_config must be a JSON object")
    return dict(value)


class QueueStartupReporter(StartupReporter):
    """Adds the startup reports of a serving work item to its progress, one progress entry each (RFC #1283).

    Values: ``queue`` holds ``item``, which ``worker_id`` holds; ``number`` is the next progress number. A failed write
    is logged and lost.

    Purpose: ``ServingWorker.run_item`` makes one for each serving request, and binds the runtime's reports to it.
    """

    def __init__(self, queue: WorkQueue, item: WorkItem, worker_id: str) -> None:
        self.queue, self.item, self.worker_id = queue, item, worker_id
        self.number = len(item.progress)
        self.lock = threading.Lock()

    def add(self, entry: Mapping[str, JsonValue]) -> None:
        with self.lock:
            try:
                self.queue.add_progress(
                    self.item.queue, self.item.id, self.worker_id, {PROGRESS_KEY: dict(entry)}, number=self.number
                )
                self.number += 1
            except Exception:
                logger.warning("could not report the startup of %s", self.item.id, exc_info=True)


class ServingWorker(QueueWorker):
    """Start queued releases and retire unused endpoints in the inference service.

    ``recipe`` supplies the runtime and each scenario's components. ``backends`` resolves durable snapshots.
    ``coordination`` owns requests and routes. ``local_artifact_dir`` holds materialized snapshots, or uses the repository default.
    A long-lived work lease gives one worker runtime ownership, including health checks and idle periods.
    Other replicas only read routes. The owner recovers completed selections before it takes new work: it opens the
    runtime, offers it the routes from before (``InferenceRuntime.adopt``), keeps the routes that it took over, and
    queues every completed selection again. ``owner`` is None before ownership. ``recovered`` becomes true after the
    recovery. ``opened`` becomes true after the runtime opens. ``activated`` maps each scenario to the artifact that
    this worker activated for it last, until the worker stops the endpoint that serves that artifact for the scenario:
    a serving request for the artifact in ``activated`` loads nothing. A route that it kept from before is activated
    again once, so that the runtime takes over what it adopted.
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
        runtime = recipe.runtime
        super().__init__(
            coordination.work_queue,
            SERVING_QUEUE,
            (SERVING_KIND, STOP_SERVING_KIND, SERVE_ARTIFACT_KIND),
            worker_id=worker_id,
            slots=1 if runtime is None else runtime.serving_slots,
        )
        self.recipe = recipe
        self.backends = backends
        self.routes = coordination.served_models
        self.local_artifact_dir = local_artifact_dir
        self.stop_delay_seconds = CACHE_SECONDS + (300.0 if runtime is None else runtime.inference_timeout_s)
        self.opened = False
        # Slots open the runtime once.
        self.open_lock = threading.Lock()
        self.owner: ServingLease | None = None
        self.recovered = False
        self.activated: dict[str, str] = {}
        # Engines can read selected files after startup. Keep local views alive until this worker stops.
        self.repositories: dict[str, Repository] = {}
        self.repositories_lock = threading.Lock()

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
            logger.info("inference worker %s released runtime ownership", self.worker_id)
            self.owner = None
        self.repositories.clear()

    def run_once(self) -> bool:
        if self.stopped.is_set():
            return False
        if self.owner is None:
            runtime = self.recipe.runtime
            if runtime is None or not runtime.serves_requests:
                return False
            self.requeue_expired()
            self.queue.request(INFERENCE_OWNER_QUEUE, "inference", INFERENCE_OWNER_QUEUE, INFERENCE_OWNER_QUEUE, {})
            item = self.queue.take(INFERENCE_OWNER_QUEUE, self.worker_id, [INFERENCE_OWNER_QUEUE])
            if item is None:
                return False
            self.queue.start(INFERENCE_OWNER_QUEUE, item.id, self.worker_id)
            self.owner = ServingLease(self.queue, item, self.worker_id, runtime, self.stopped)
            self.owner.__enter__()
            logger.info("inference worker %s holds runtime ownership", self.worker_id)
        if self.owner.lease_lost.is_set():
            return False
        if not self.recovered:
            previous = tuple(self.routes.all_served())
            kept = self.adopted(previous) if previous else ()
            self.routes.recover_completed(
                self.owner.item, self.worker_id, stop_delay_seconds=self.stop_delay_seconds, kept=kept
            )
            self.recovered = True
            logger.info(
                "requeued the serving of every scenario; kept %d of %d routes from before, whose engines still serve",
                len(kept),
                len(previous),
            )
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

    def open_runtime(self, runtime: InferenceRuntime) -> None:
        started = time.monotonic()
        runtime.open()
        self.opened = True
        logger.info("opened the inference runtime in %.1f s", time.monotonic() - started)

    def adopted(self, previous: tuple[ServedArtifact, ...]) -> tuple[ServedArtifact, ...]:
        """Open the owner's runtime and return the routes of ``previous`` whose engines it took over. A runtime that
        does not open takes over none: the routes go, and the first serving opens it again."""
        if self.owner is None:
            raise WorkLeaseLost("the serving worker has no runtime ownership")
        runtime = self.owner.runtime
        try:
            self.open_runtime(runtime)
        except Exception:
            logger.exception("could not open the inference runtime; the %d routes from before go", len(previous))
            return ()
        return tuple(served for served in previous if runtime.adopt(served))

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        if self.stopped.is_set():
            raise WorkerStopped
        runtime = self.recipe.runtime
        if runtime is None:
            raise ValueError("the recipe has no inference runtime")
        if self.owner is None or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost runtime ownership")
        with self.open_lock:
            if not self.opened:
                self.open_runtime(runtime)
        if lease_lost.is_set() or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost its lease")
        if item.kind == STOP_SERVING_KIND:
            served = ServedArtifact.from_json(item.payload["served"])
            due = any(
                stop.served == served and stop.stop_after == item.payload["stop_after"]
                for stop in self.routes.due_stops(time.time())
            )
            if due and not self.routes.is_referenced(served):
                # A scenario that this endpoint served must activate the same artifact again to serve it again.
                stopped_scenarios = [
                    scenario
                    for scenario, artifact_id in list(self.activated.items())
                    if artifact_id == served.artifact_id and runtime.served_artifact(artifact_id, scenario) == served
                ]
                runtime.stop_serving(served)
                for scenario in stopped_scenarios:
                    self.activated.pop(scenario, None)
                logger.info("stopped serving artifact %s at %s", served.artifact_id, served.url)
            return {}
        if item.kind == SERVE_ARTIFACT_KIND:
            return self.serve_artifact(item, runtime)
        selected = self.routes.current_request(item.scenario)
        if selected is None or selected.id != item.id or not item.payload:
            return {}
        stop, done = threading.Event(), threading.Event()
        watcher = threading.Thread(
            target=self.watch_replaced,
            args=(item, lease_lost, stop, done),
            name=f"reef-replaced-{item.id}",
            daemon=True,
        )
        watcher.start()
        try:
            with serving_work(stop, redeploy=REDEPLOY in item.payload):
                return self.start_request(item, runtime, lambda: self.check_stop(stop, lease_lost))
        except StartupStopped as error:
            self.check_stop(stop, lease_lost)
            raise ItemFailed(REPLACED) from error
        except StartupFailed as error:
            raise ItemFailed(f"{STARTUP_FAILED_PREFIX}{error}") from error
        finally:
            done.set()

    def watch_replaced(
        self, item: WorkItem, lease_lost: threading.Event, stop: threading.Event, done: threading.Event
    ) -> None:
        """Set ``stop`` when a newer request replaces ``item`` or the worker loses it, until ``done`` (RFC #1299). A
        failed read leaves ``stop`` as it is."""
        while not done.wait(REPLACED_CHECK_SECONDS):
            if lease_lost.is_set():
                stop.set()
                return
            try:
                current = self.routes.current_request(item.scenario)
            except Exception:
                logger.warning("could not read the current serving request of %s", item.scenario, exc_info=True)
                continue
            if current is None or current.id != item.id:
                logger.info("scenario %s: a newer serving request replaced %s, which stops", item.scenario, item.id)
                stop.set()
                return

    def check_stop(self, stop: threading.Event, lease_lost: threading.Event) -> None:
        """``WorkLeaseLost`` when the worker lost the item or its runtime ownership, else ``ItemFailed`` when ``stop``
        is set: a newer request replaced the item."""
        if lease_lost.is_set() or self.owner is None or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost its lease")
        if stop.is_set():
            raise ItemFailed(REPLACED)

    def start_request(
        self, item: WorkItem, runtime: InferenceRuntime, check_stop: Callable[[], None]
    ) -> dict[str, JsonValue]:
        """Activate the artifact of the serving request ``item`` and serve each artifact beside it, all at once when
        the runtime has more than one slot; return the routes. ``check_stop`` raises when the work must stop."""
        artifact_id = require_text(item.payload.get("artifact_id"), "artifact_id")
        repository = self.repository(item.scenario)

        def selected_artifact(value: Mapping[str, object]) -> Artifact:
            return repository.select(ArtifactSelection.from_dict(value["selection"]))

        artifact = selected_artifact(item.payload)
        components = self.recipe.scenario_components(item.scenario)
        # The runtime reports each artifact's startup phases to the item's progress (RFC #1283).
        reporter = QueueStartupReporter(self.queue, item, self.worker_id)
        # Before the activation, so that an engine that starts for the artifact starts with the override.
        runtime.apply_serving_config(item.scenario, serving_config_of(item.payload))
        aliases = item.payload.get("aliases")
        beside = item.payload.get("beside")
        if not isinstance(aliases, list) or not isinstance(beside, dict):
            raise ValueError("a serving request needs aliases and beside artifacts")
        # The artifacts are resolved here, so that the threads only call the runtime.
        beside_artifacts: dict[str, Artifact] = {}
        for name, value in beside.items():
            if not isinstance(value, dict):
                raise ValueError("a beside artifact needs its snapshot and component")
            component = require_text(value.get("component"), "component")
            if component == components.loaded_component:
                beside_artifacts[name] = components.component_artifact(selected_artifact(value), component)

        def activate() -> None:
            if self.activated.get(item.scenario) == artifact_id and not redeploying():
                logger.info("scenario %s: artifact %s serves already", item.scenario, artifact_id)
                return
            started = time.monotonic()
            with reporting_startup(reporter, artifact_id):
                activate_snapshot(components, artifact, runtime, item.scenario)
            self.activated[item.scenario] = artifact_id
            logger.info(
                "scenario %s: activated artifact %s in %.1f s", item.scenario, artifact_id, time.monotonic() - started
            )

        def serve_beside(name: str) -> ServedArtifact:
            with reporting_startup(reporter, name):
                served = runtime.serve_artifact(beside_artifacts[name], scenario=item.scenario)
            if served is None:
                raise ValueError("the inference runtime provides no route for a beside artifact")
            return served

        if runtime.serving_slots > 1 and beside_artifacts:
            # Each call runs in a copy of this context, which binds the item's stop event and redeploy flag.
            with ThreadPoolExecutor(min(runtime.serving_slots, 1 + len(beside_artifacts))) as pool:
                active_start = pool.submit(copy_context().run, activate)
                served_beside = {
                    name: pool.submit(copy_context().run, serve_beside, name) for name in beside_artifacts
                }
                active_start.result()
                beside_models = {name: future.result() for name, future in served_beside.items()}
        else:
            activate()
            beside_models = {}
            for name in beside_artifacts:
                check_stop()
                beside_models[name] = serve_beside(name)
        check_stop()
        active = runtime.served_artifact(artifact_id, item.scenario)
        if active is None:
            raise ValueError("the inference runtime provides no route for the selected artifact")
        models = {ACTIVE: active}
        for alias in aliases:
            models[require_text(alias, "model name")] = active
        for beside_name in beside:
            models[beside_name] = beside_models.get(beside_name, active)
        return {name: dict(served.to_json()) for name, served in models.items()}

    def repository(self, scenario: str) -> Repository:
        """The repository of ``scenario``, which this worker keeps while its backend stays. Slots share it."""
        with self.repositories_lock:
            return self.repository_of(scenario)

    def repository_of(self, scenario: str) -> Repository:
        backend = self.backends(scenario)
        repository = self.repositories.get(scenario)
        if repository is None or repository.backend is not backend:
            repository = Repository(backend, backend.current(), local_dir=self.local_artifact_dir)
            self.repositories[scenario] = repository
        return repository

    def serve_artifact(self, item: WorkItem, runtime: InferenceRuntime) -> Mapping[str, JsonValue]:
        """Serve the artifact that ``item`` names, with no route, and give its ``ServedArtifact``, or null for a runtime
        with no address that another process can call. This worker is the only one that starts engines, one item at a
        time, so a second item for a served artifact starts nothing."""
        selection = ArtifactSelection.from_dict(item.payload["selection"])
        component = require_text(item.payload.get("component"), "component")
        components = self.recipe.scenario_components(item.scenario)
        artifact = components.component_artifact(self.repository(item.scenario).select(selection), component)
        runtime.apply_serving_config(item.scenario, self.routes.read_serving_config(item.scenario))
        served = runtime.serve_artifact(artifact, scenario=item.scenario)
        return {"served": None if served is None else dict(served.to_json())}

    def finish_item(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        if self.owner is None or self.owner.lease_lost.is_set():
            raise WorkLeaseLost("the serving worker lost runtime ownership")
        if item.kind == SERVE_ARTIFACT_KIND:
            super().finish_item(item, result)
            return
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
                serving_config=serving_config_of(item.payload),
                stop_delay_seconds=self.stop_delay_seconds,
                owner=self.owner.item,
            )


__all__ = ["ServingWorker"]
