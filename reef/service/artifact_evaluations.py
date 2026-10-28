"""Transport-independent handling of the artifact evaluations that clients ask a scenario for.

A client asks Reef to evaluate one artifact: a component of one of the scenario's releases, with the other
side of its policy configured (RFC #959). ``ArtifactEvaluations`` checks the request against the scenario,
stores it in the scenario's evaluation store, and runs its rollouts on a thread of its own. When the rollouts
end, it stores their ``PolicyEvaluation``, or the error that stopped them. No aiohttp types appear here.
``reef.service.routes.evaluations`` adapts it to HTTP.

This version evaluates a model artifact (a ``model`` or ``model_adapter`` component) on a runtime that serves
a release on request (``InferenceRuntime.release_model``), with a Harbor agent, on the Harbor tasks of the
eval split of the scenario's ``imported-tasks`` store. It refuses a harness artifact, and a model artifact
on any other runtime.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    PolicyEvaluation,
    PolicyModel,
    RolloutRequest,
    RolloutScores,
)
from reef.dispatcher import Dispatcher
from reef.evaluation.release_endpoint import ReleaseEndpoint
from reef.harness.client.tasks import DEFAULT_AGENT, TaskLab, TaskRolloutRunner
from reef.storage.evaluations import EvaluationNotRunning, EvaluationStore, StoredEvaluation
from reef.storage.tasks import IMPORTED_TASK_STORE

logger = logging.getLogger(__name__)

#: The most evaluations that one service runs at a time.
MAX_RUNNING_EVALUATIONS = 2
#: The most rollouts of one evaluation that run at a time.
ROLLOUT_CONCURRENCY = 4
#: The error of an evaluation that was running when the service that ran it stopped.
STOPPED_ERROR = "the service that ran this evaluation stopped before it ended"
#: The scenario name that an evaluation's model calls carry; the release endpoint serves no scenario.
EVALUATION_SCENARIO = "artifact-evaluation"


class EvaluationLimitReached(ReefError):
    """The service runs ``MAX_RUNNING_EVALUATIONS`` evaluations already."""


@dataclass(frozen=True)
class EvaluationRun:
    """An accepted evaluation, ready to run.

    Values: ``scenario`` and ``evaluation_id`` name it; ``request`` configures a Harbor agent (``harness`` is
    set); ``release`` is the materialized release that ``request`` names, and ``model`` the model that the
    runtime serves for it; ``tasks`` holds each task's name and Harbor task directory, in the store's
    publication order, with one or more tasks. Purpose:
    ``ArtifactEvaluations.start`` makes it after it checks the request, and ``ArtifactEvaluations.run`` runs it.
    """

    scenario: str
    evaluation_id: str
    request: ArtifactEvaluationRequest
    release: Artifact
    model: PolicyModel
    tasks: tuple[tuple[str, Path], ...]


def parsed_request(body: object) -> dict[str, Any]:
    """The fields of an evaluation request body, with the defaults of the fields that it leaves out.

    ``component`` stays None when the body leaves it out; the scenario's surface names it. Raise ``ValueError``
    for a body that is not an object, has an unknown field, or has ``task_names`` that is not a list.
    """
    if not isinstance(body, Mapping):
        raise ValueError("an evaluation request is a JSON object")
    known = {"release_id", "component", "harness", "model", "task_names", "rollouts_per_task", "environment"}
    unknown = sorted(set(body) - known)
    if unknown:
        raise ValueError(f"an evaluation request takes no {unknown}; it takes {sorted(known)}")
    task_names = body.get("task_names", [])
    if not isinstance(task_names, list):
        raise ValueError("task_names is a list of task names")
    return {
        "release_id": body.get("release_id"),
        "component": body.get("component"),
        "harness": body.get("harness"),
        "model": body.get("model"),
        "task_names": tuple(task_names),
        "rollouts_per_task": body.get("rollouts_per_task", 1),
        "environment": body.get("environment", "docker"),
    }


def harbor_agent(harness: Mapping[str, Any] | None) -> Mapping[str, object]:
    """The Harbor agent configuration that runs a request's ``harness``.

    A harness that names ``terminus-2`` alone runs Reef's default terminus-2 configuration (``DEFAULT_AGENT``),
    which points the agent at the rollout's endpoint. Any other harness is the agent's whole configuration, with
    ``{model}``, ``{base_url}`` and ``{api_key}`` where the runner fills them in, as Tinker's ``rollout.agent``.
    Raise ``ValueError`` for a request without a harness.
    """
    if harness is None:
        raise ValueError("a model artifact's evaluation needs a harness")
    if dict(harness) == {"name": DEFAULT_AGENT["name"]}:
        return DEFAULT_AGENT
    return harness


class ArtifactEvaluations:
    """Accepts, runs and reads the artifact evaluations of a service's scenarios.

    Values: ``dispatcher`` holds the scenarios. ``owner`` is a new ID of this instance, which each evaluation
    that it runs stores; an evaluation that another instance left ``running`` reads as failed. ``lab`` runs each
    rollout's Harbor trial, None for reef-eval's Harbor lab. ``running`` holds the IDs of the evaluations that
    run now, at most ``MAX_RUNNING_EVALUATIONS``. Purpose: the evaluation routes call ``start``, ``get`` and
    ``newest``.
    """

    def __init__(self, dispatcher: Dispatcher, *, lab: TaskLab | None = None) -> None:
        self.dispatcher = dispatcher
        self.lab = lab
        self.owner = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.running: set[str] = set()

    def start(self, scenario: str, body: object) -> StoredEvaluation:
        """Check the request ``body`` against ``scenario``, store it, and start its rollouts.

        Raise ``ValueError`` or ``ReefError`` for a request that the scenario cannot run, ``UnknownScenario``
        for an unknown scenario, ``ArtifactNotFound`` for an unknown release, and ``EvaluationLimitReached``
        when ``MAX_RUNNING_EVALUATIONS`` evaluations run.
        """
        fields = parsed_request(body)
        with self.dispatcher.locked_scenario(scenario) as current:
            kinds = {name: component.kind for name, component in current.surface.components.items()}
            component = fields["component"]
            if component is None:
                named = [name for name, kind in kinds.items() if kind is not None]
                if len(named) != 1:
                    raise ValueError(f"name the component to evaluate: the releases hold {sorted(named)}")
                component = named[0]
            if component not in kinds:
                raise ValueError(f"the releases of scenario {scenario!r} hold no component {component!r}")
            request = ArtifactEvaluationRequest(**{**fields, "component": component})
            kind = kinds[component]
            if kind == "harness_plugin":
                raise ReefError("Reef does not evaluate a harness artifact yet (RFC #959, step 2)")
            if kind is None:
                raise ValueError(f"component {component!r} is no side of a policy, so Reef cannot evaluate it")
            if request.harness is None:
                raise ValueError(f"component {component!r} is a {kind}: configure the harness that runs it")
            if any(other == "harness_plugin" for other in kinds.values()):
                raise ReefError("Reef evaluates a model artifact only in a scenario whose releases hold no harness")
            release = current.release_queries.artifact_for_version(request.release_id).materialize()
            runtime = current.runtime
            model = None if runtime is None else runtime.release_model(release)
            if model is None:
                raise ReefError(
                    f"scenario {scenario!r}'s runtime serves only its current release, so it cannot evaluate a "
                    "release on request"
                )
            store = current.store.tasks.open(IMPORTED_TASK_STORE)
            held = {task.name: task for task in store.tasks(split="eval")}
            unknown = sorted(set(request.task_names) - set(held))
            if unknown:
                raise ValueError(f"the eval split of {IMPORTED_TASK_STORE!r} holds no tasks {unknown}")
            names = request.task_names or tuple(held)
            if not names:
                raise ValueError(f"the eval split of {IMPORTED_TASK_STORE!r} holds no task to evaluate on")
            tasks = tuple((name, store.path(held[name].files_digest)) for name in names)
            with self.lock:
                if len(self.running) >= MAX_RUNNING_EVALUATIONS:
                    raise EvaluationLimitReached(
                        f"this service runs {MAX_RUNNING_EVALUATIONS} evaluations already; ask again when one ends"
                    )
                evaluation_id = uuid.uuid4().hex
                stored = current.store.evaluations.create(
                    evaluation_id=evaluation_id, owner=self.owner, request=request, created_at=time.time()
                )
                self.running.add(evaluation_id)
        run = EvaluationRun(scenario, evaluation_id, request, release, model, tasks)
        # The thread is a daemon: a stopping service does not wait for rollouts, and a later read reports the
        # evaluation failed.
        threading.Thread(target=self.run, args=(run,), name=f"reef-evaluation-{evaluation_id}", daemon=True).start()
        return stored

    def run(self, run: EvaluationRun) -> None:
        """Run the rollouts of ``run``, and store their evaluation, or the error that stopped them."""
        try:
            with self.dispatcher.locked_scenario(run.scenario) as current:
                runtime = current.runtime
            if runtime is None:
                raise ReefError(f"scenario {run.scenario!r} has no runtime")
            with (
                tempfile.TemporaryDirectory(prefix="reef-evaluation-") as work,
                ReleaseEndpoint(runtime, run.release) as endpoint,
            ):
                runner = TaskRolloutRunner(
                    reef_url=endpoint.url,
                    scenario=EVALUATION_SCENARIO,
                    model=run.model.name,
                    model_adapter=run.model.adapter,
                    work_dir=Path(work),
                    agent=harbor_agent(run.request.harness),
                    environment=run.request.environment,
                    is_reporting=False,
                    is_unrecorded=True,
                    concurrency=ROLLOUT_CONCURRENCY,
                    lab=self.lab,
                )
                rollouts = runner.run_rollouts(
                    [
                        RolloutRequest(runner.harness, runner.served_model, str(path))
                        for _, path in run.tasks
                        for _ in range(run.request.rollouts_per_task)
                    ]
                )
            scores = RolloutScores.by_task(
                [name for name, _ in run.tasks], [runner.rollout_score(rollout) for rollout in rollouts]
            )
            self.end(run, result=PolicyEvaluation(runner.harness, run.model, scores), error=None)
        except Exception as exc:
            logger.exception("artifact evaluation %s of scenario %s failed", run.evaluation_id, run.scenario)
            self.end(run, result=None, error=f"{type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                self.running.discard(run.evaluation_id)

    def end(self, run: EvaluationRun, *, result: PolicyEvaluation | None, error: str | None) -> None:
        """Store the end of ``run``: its ``result``, or its ``error`` when ``result`` is None."""
        try:
            with self.dispatcher.locked_scenario(run.scenario) as current:
                evaluations = current.store.evaluations
                if result is not None:
                    evaluations.finish(run.evaluation_id, result=result, finished_at=time.time())
                else:
                    evaluations.fail(
                        run.evaluation_id, error=error or "the evaluation failed", finished_at=time.time()
                    )
        except (ReefError, ValueError) as exc:
            # The scenario was deleted, or the evaluation ended already: nothing is left to record it in.
            logger.warning("could not store the end of evaluation %s: %s", run.evaluation_id, exc)

    def get(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id`` of ``scenario``, or None; one that another service left running is failed."""
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.get(evaluation_id)
            return None if stored is None else self.settled(current.store.evaluations, stored)

    def newest(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario``, newest first, as ``get`` reads each."""
        with self.dispatcher.locked_scenario(scenario) as current:
            evaluations = current.store.evaluations
            return tuple(self.settled(evaluations, stored) for stored in evaluations.newest(limit=limit))

    def settled(self, evaluations: EvaluationStore, stored: StoredEvaluation) -> StoredEvaluation:
        """``stored``, failed first when it is running for a service instance other than this one."""
        if stored.state != "running" or stored.owner == self.owner:
            return stored
        try:
            return evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
        except EvaluationNotRunning:
            # Another reader failed it first.
            return evaluations.get(stored.evaluation_id) or stored


__all__ = ["MAX_RUNNING_EVALUATIONS", "ArtifactEvaluations", "EvaluationLimitReached", "EvaluationRun"]
