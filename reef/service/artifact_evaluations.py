"""Transport-independent handling of the artifact evaluations of a scenario.

A client asks Reef to evaluate one artifact: a component of one of the scenario's artifacts, with the other side
of its policy configured (RFC #959). A scenario can also hold settings that Reef evaluates each new artifact with:
when a commit publishes a checkpoint as a new artifact, Reef queues its evaluation. ``ArtifactEvaluations``
checks each request against the scenario, stores it in the scenario's evaluation store, and runs its rollouts
on a thread of its own, at most ``MAX_RUNNING_EVALUATIONS`` at a time. When the rollouts end, it stores their
``PolicyEvaluation``, or the error that stopped them. No aiohttp types appear here.
``reef.service.routes.evaluations`` adapts it to HTTP.

This version evaluates a model artifact (a ``model`` or ``model_adapter`` component) on a runtime that serves an
artifact on request (``InferenceRuntime.artifact_model``), with a Harbor agent, on the Harbor tasks of the eval
split of the scenario's ``imported-tasks`` store. It refuses a harness artifact, and a model artifact on any
other runtime.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.components import made_artifact_id
from reef.core.errors import CommitPending, ReefError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    ArtifactEvaluationSettings,
    PolicyEvaluation,
    PolicyModel,
    RolloutRequest,
    RolloutScores,
)
from reef.dispatcher import Dispatcher
from reef.evaluation.artifact_endpoint import ArtifactEndpoint
from reef.harness.client.tasks import DEFAULT_AGENT, TaskLab, TaskRolloutRunner
from reef.scenario.scenario import Scenario
from reef.storage.commits import CommitRecord
from reef.storage.evaluations import EvaluationStateConflict, EvaluationStore, StoredEvaluation
from reef.storage.observer import RecordObserver
from reef.storage.tasks import IMPORTED_TASK_STORE

logger = logging.getLogger(__name__)

#: The most evaluations that one service runs at a time.
MAX_RUNNING_EVALUATIONS = 2
#: The most rollouts of one evaluation that run at a time.
ROLLOUT_CONCURRENCY = 4
#: The error of an evaluation that was queued or running when the service that held it stopped.
STOPPED_ERROR = "the service that held this evaluation stopped before it ended"
#: How often Reef tries to activate an artifact that passed its evaluation while training results wait to commit,
#: and the seconds it waits after the first attempt, growing with each.
ACTIVATION_ATTEMPTS = 5
ACTIVATION_RETRY_SECONDS = 1.0
#: The scenario name that an evaluation's model calls carry; the artifact endpoint serves no scenario.
EVALUATION_SCENARIO = "artifact-evaluation"


class EvaluationLimitReached(ReefError):
    """The service runs ``MAX_RUNNING_EVALUATIONS`` evaluations already."""


@dataclass(frozen=True)
class EvaluationRun:
    """An accepted evaluation, ready to run.

    Values: ``scenario`` and ``evaluation_id`` name it; ``request`` configures a Harbor agent (``harness`` is
    set); ``artifact`` is the materialized artifact that ``request`` names, and ``model`` the model that the
    runtime serves for it; ``tasks`` holds each task's name and Harbor task directory, in the store's
    publication order, with one or more tasks. Purpose: ``ArtifactEvaluations.accept`` makes it after it checks
    the request, and ``ArtifactEvaluations.run`` runs it.
    """

    scenario: str
    evaluation_id: str
    request: ArtifactEvaluationRequest
    artifact: Artifact
    model: PolicyModel
    tasks: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class NewArtifact:
    """An artifact that a commit made: the scenario, the artifact's ID, and the component whose artifact it is, or
    None for a scenario with one component. ``NewArtifacts`` makes it; ``ArtifactEvaluations``
    evaluates it with the scenario's settings."""

    scenario: str
    artifact_id: str
    component: str | None


class NewArtifacts(RecordObserver):
    """Reports each artifact that a committed training step published, for the automatic evaluations.

    Values: ``queue`` holds a ``NewArtifact`` for each committed training step that published a checkpoint, in
    commit order, and None once the observer closes. Purpose: the service wraps its scenario storage with
    ``ObservedScenarioStorage(storage, new_artifacts)``, so a commit reports here without the committer knowing,
    and ``ArtifactEvaluations`` reads ``queue`` on its own thread.
    """

    def __init__(self) -> None:
        self.queue: queue.SimpleQueue[NewArtifact | None] = queue.SimpleQueue()

    def record_committed(self, commit: CommitRecord) -> None:
        # An activation or a new training start names an artifact that exists already: it reports none.
        if not commit.made_artifact:
            return
        if commit.components is None:
            self.queue.put(NewArtifact(commit.scenario, commit.artifact_ref.release_id, commit.component))
            return
        # With several components, the release made the artifact of each component that it gave a new artifact_id.
        release_id = commit.artifact_ref.release_id
        for name, artifact_id in commit.components.items():
            if artifact_id == made_artifact_id(release_id, name, single=False):
                self.queue.put(NewArtifact(commit.scenario, artifact_id, name))

    def close(self) -> None:
        self.queue.put(None)


def request_settings(body: object) -> tuple[str, ArtifactEvaluationSettings]:
    """The artifact ID and the settings of a client's evaluation request ``body``.

    Raise ``ValueError`` for a body that is not an object, has no non-empty ``artifact_id``, or has settings
    that ``ArtifactEvaluationSettings.from_json`` refuses.
    """
    if not isinstance(body, Mapping):
        raise ValueError("an evaluation request is a JSON object")
    artifact_id = body.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("an evaluation request names a non-empty artifact_id")
    return artifact_id, ArtifactEvaluationSettings.from_json(
        {key: value for key, value in body.items() if key != "artifact_id"}
    )


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
    """Accepts, queues, runs and reads the artifact evaluations of a service's scenarios.

    Values: ``dispatcher`` holds the scenarios. ``owner`` is a new ID of this instance, which each evaluation
    that it holds stores; an evaluation that another instance left queued or running reads as failed. ``lab``
    runs each rollout's Harbor trial, None for reef-eval's Harbor lab. ``new_artifacts`` reports the new
    artifacts that Reef evaluates with each scenario's settings, or is None for no automatic evaluation.
    ``running`` holds the IDs of the evaluations that run now, at most ``MAX_RUNNING_EVALUATIONS``, and
    ``waiting`` the queued ones of this instance, oldest first. Purpose: the evaluation routes call ``start``,
    ``get`` and ``newest``, and the scenario update route calls ``configure``. ``open`` starts the thread that
    evaluates new artifacts, and ``close`` stops it.
    """

    def __init__(
        self, dispatcher: Dispatcher, *, lab: TaskLab | None = None, new_artifacts: NewArtifacts | None = None
    ) -> None:
        self.dispatcher = dispatcher
        self.lab = lab
        self.new_artifacts = new_artifacts
        self.owner = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.running: set[str] = set()
        self.waiting: deque[EvaluationRun] = deque()
        self.reader: threading.Thread | None = None

    def open(self) -> None:
        """Start evaluating the new artifacts that ``new_artifacts`` reports; nothing without it."""
        if self.new_artifacts is not None and self.reader is None:
            self.reader = threading.Thread(target=self.read_new_artifacts, name="reef-new-artifacts", daemon=True)
            self.reader.start()

    def close(self) -> None:
        """Stop evaluating new artifacts. Evaluations that run go on; queued ones read as failed after a restart."""
        if self.new_artifacts is not None and self.reader is not None:
            self.new_artifacts.queue.put(None)
            self.reader.join(timeout=5)
            self.reader = None

    def read_new_artifacts(self) -> None:
        """Queue the evaluation of each new artifact that ``new_artifacts`` reports, until it reports None."""
        if self.new_artifacts is None:
            return
        while (new := self.new_artifacts.queue.get()) is not None:
            try:
                self.evaluate_new(new)
            except Exception:
                logger.exception("could not queue the evaluation of artifact %s of %s", new.artifact_id, new.scenario)

    def evaluate_new(self, new: NewArtifact) -> StoredEvaluation | None:
        """Queue the evaluation of ``new`` with its scenario's settings; None when the scenario has none."""
        stored = self.dispatcher.stored_settings(new.scenario).artifact_evaluation
        if stored is None:
            return None
        settings = ArtifactEvaluationSettings.from_json(stored)
        if settings.component is None and new.component is not None:
            settings = replace(settings, component=new.component)
        elif settings.component is not None and new.component not in (None, settings.component):
            # Another component's trainer published the artifact: these settings evaluate their own component.
            return None
        return self.accept(new.scenario, new.artifact_id, settings, state="queued")

    def start(self, scenario: str, body: object) -> StoredEvaluation:
        """Check a client's request ``body`` against ``scenario``, store it, and start its rollouts.

        Raise ``ValueError`` or ``ReefError`` for a request that the scenario cannot run, ``UnknownScenario``
        for an unknown scenario, ``ArtifactNotFound`` for an unknown artifact, and ``EvaluationLimitReached``
        when ``MAX_RUNNING_EVALUATIONS`` evaluations run.
        """
        artifact_id, settings = request_settings(body)
        return self.accept(scenario, artifact_id, settings, state="running")

    def configure(self, scenario: str, changes: Mapping[str, object]) -> dict[str, object]:
        """Store how Reef evaluates and activates each new artifact of ``scenario``, and return each changed
        setting as stored.

        ``changes`` holds ``artifact_evaluation``, ``activation``, or both:

        - ``artifact_evaluation`` is the settings that Reef evaluates each new artifact with, or None to evaluate
          none. Reef checks them against the scenario's active artifact, as it checks a client's request. The eval
          tasks can still be missing.
        - ``activation`` is the scenario's own built-in activation rule, or None to follow the deployment's rule.

        A setting that ``changes`` leaves out keeps its stored value. Reef checks the two settings together, and
        stores both or neither (``Dispatcher.set_evaluation_and_activation``). Raise ``ValueError`` or
        ``ReefError`` for settings that the scenario cannot run.
        """
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = self.dispatcher.stored_settings(scenario)
            artifact_evaluation = stored.artifact_evaluation
            value = changes.get("artifact_evaluation")
            if value is not None:
                settings = ArtifactEvaluationSettings.from_json(value)
                self.checked(current, None, settings)
                artifact_evaluation = settings.to_json()
            elif "artifact_evaluation" in changes:
                artifact_evaluation = None
            activation = self.dispatcher.set_evaluation_and_activation(
                scenario,
                artifact_evaluation=artifact_evaluation,
                activation=changes.get("activation", stored.activation),
            )
        changed = {"artifact_evaluation": artifact_evaluation, "activation": activation}
        return {key: changed[key] for key in changed if key in changes}

    def checked(
        self, current: Scenario, artifact_id: str | None, settings: ArtifactEvaluationSettings
    ) -> tuple[ArtifactEvaluationRequest, Artifact, PolicyModel]:
        """The request of ``settings`` for ``artifact_id`` (None: the component's active artifact), the materialized
        artifact, and the model that the runtime serves for it; ``ValueError`` or ``ReefError`` when ``current``
        cannot run it."""
        kinds = {name: component.kind for name, component in current.surface.components.items()}
        component = settings.component
        if component is None:
            named = [name for name, kind in kinds.items() if kind is not None]
            if len(named) != 1:
                raise ValueError(f"name the component to evaluate: the artifacts hold {sorted(named)}")
            component = named[0]
        if component not in kinds:
            raise ValueError(f"the artifacts of scenario {current.name!r} hold no component {component!r}")
        if artifact_id is None:
            active = Artifact(current.current_artifact_ref(), current.repository)
            artifact_id = current.surface.component_artifact(active, component).artifact_id
        request = settings.request(artifact_id, component)
        kind = kinds[component]
        if kind == "harness_plugin":
            raise ReefError("Reef does not evaluate a harness artifact yet (RFC #959, step 2)")
        if kind is None:
            raise ValueError(f"component {component!r} is no side of a policy, so Reef cannot evaluate it")
        if request.harness is None:
            raise ValueError(f"component {component!r} is a {kind}: configure the harness that runs it")
        if any(other == "harness_plugin" for other in kinds.values()):
            raise ReefError("Reef evaluates a model artifact only in a scenario whose artifacts hold no harness")
        _, release = current.resolve_artifact(artifact_id, component)
        artifact = release.materialize()
        runtime = current.runtime
        model = None if runtime is None else runtime.artifact_model(artifact)
        if model is None:
            raise ReefError(
                f"scenario {current.name!r}'s runtime serves only its current artifact, so it cannot evaluate an "
                "artifact on request"
            )
        return request, artifact, model

    def accept(
        self,
        scenario: str,
        artifact_id: str,
        settings: ArtifactEvaluationSettings,
        *,
        state: Literal["queued", "running"],
    ) -> StoredEvaluation:
        """Check and store the evaluation of ``artifact_id`` with ``settings``, and run it, or queue it.

        A ``running`` evaluation starts at once, or raises ``EvaluationLimitReached`` when
        ``MAX_RUNNING_EVALUATIONS`` run. A ``queued`` one waits for a free run.
        """
        with self.dispatcher.locked_scenario(scenario) as current:
            request, artifact, model = self.checked(current, artifact_id, settings)
            store = current.store.tasks.open(IMPORTED_TASK_STORE)
            held = {task.name: task for task in store.tasks(split="eval")}
            unknown = sorted(set(request.task_names) - set(held))
            if unknown:
                raise ValueError(f"the eval split of {IMPORTED_TASK_STORE!r} holds no tasks {unknown}")
            names = request.task_names or tuple(held)
            if not names:
                raise ValueError(f"the eval split of {IMPORTED_TASK_STORE!r} holds no task to evaluate on")
            tasks = tuple((name, store.path(held[name].files_digest)) for name in names)
            evaluation_id = uuid.uuid4().hex
            run = EvaluationRun(scenario, evaluation_id, request, artifact, model, tasks)
            with self.lock:
                if state == "running" and len(self.running) >= MAX_RUNNING_EVALUATIONS:
                    raise EvaluationLimitReached(
                        f"this service runs {MAX_RUNNING_EVALUATIONS} evaluations already; ask again when one ends"
                    )
                stored = current.store.evaluations.create(
                    evaluation_id=evaluation_id, owner=self.owner, request=request, created_at=time.time(), state=state
                )
                if state == "running":
                    self.running.add(evaluation_id)
                else:
                    self.waiting.append(run)
        if state == "running":
            self.launch(run)
        else:
            self.launch_waiting()
        return stored

    def launch(self, run: EvaluationRun) -> None:
        """Run ``run`` on a thread of its own, which ``running`` already counts."""
        # The thread is a daemon: a stopping service does not wait for rollouts, and a later read reports the
        # evaluation failed.
        threading.Thread(
            target=self.run, args=(run,), name=f"reef-evaluation-{run.evaluation_id}", daemon=True
        ).start()

    def launch_waiting(self) -> None:
        """Start the oldest queued evaluations while fewer than ``MAX_RUNNING_EVALUATIONS`` run."""
        while True:
            with self.lock:
                if not self.waiting or len(self.running) >= MAX_RUNNING_EVALUATIONS:
                    return
                run = self.waiting.popleft()
                self.running.add(run.evaluation_id)
            try:
                with self.dispatcher.locked_scenario(run.scenario) as current:
                    current.store.evaluations.begin(run.evaluation_id)
            except (ReefError, ValueError) as exc:
                # The scenario was deleted, or a reader failed the evaluation: it does not run.
                logger.warning("could not start queued evaluation %s: %s", run.evaluation_id, exc)
                with self.lock:
                    self.running.discard(run.evaluation_id)
                continue
            self.launch(run)

    def run(self, run: EvaluationRun) -> None:
        """Run the rollouts of ``run``, and store their evaluation, or the error that stopped them."""
        try:
            with self.dispatcher.locked_scenario(run.scenario) as current:
                runtime = current.runtime
            if runtime is None:
                raise ReefError(f"scenario {run.scenario!r} has no runtime")
            with (
                tempfile.TemporaryDirectory(prefix="reef-evaluation-") as work,
                ArtifactEndpoint(runtime, run.artifact) as endpoint,
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
            self.launch_waiting()

    def end(self, run: EvaluationRun, *, result: PolicyEvaluation | None, error: str | None) -> None:
        """Store the end of ``run``: its ``result``, or its ``error`` when ``result`` is None. A finished run
        activates its artifact when the scenario's activation rule says so (``activates_evaluated``)."""
        activates = False
        try:
            with self.dispatcher.locked_scenario(run.scenario) as current:
                evaluations = current.store.evaluations
                if result is not None:
                    evaluations.finish(run.evaluation_id, result=result, finished_at=time.time())
                    activates = current.activation_rule.activates_evaluated(run.request.artifact_id, result)
                else:
                    evaluations.fail(
                        run.evaluation_id, error=error or "the evaluation failed", finished_at=time.time()
                    )
        except (ReefError, ValueError) as exc:
            # The scenario was deleted, or the evaluation ended already: nothing is left to record it in.
            logger.warning("could not store the end of evaluation %s: %s", run.evaluation_id, exc)
            return
        if activates:
            self.activate(run)

    def activate(self, run: EvaluationRun) -> None:
        """Activate the artifact of ``run``, waiting while a training result waits for its commit."""
        for attempt in range(ACTIVATION_ATTEMPTS):
            try:
                self.dispatcher.activate(run.scenario, run.request.artifact_id, run.request.component)
                return
            except CommitPending:
                time.sleep(ACTIVATION_RETRY_SECONDS * (attempt + 1))
            except (ReefError, ValueError) as exc:
                logger.warning("could not activate artifact %s of %s: %s", run.request.artifact_id, run.scenario, exc)
                return
        logger.warning(
            "artifact %s of %s passed its evaluation but stayed inactive: training kept committing",
            run.request.artifact_id,
            run.scenario,
        )

    def get(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id`` of ``scenario``, or None; one that another service left is failed."""
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.get(evaluation_id)
            return None if stored is None else self.settled(current.store.evaluations, stored)

    def newest(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario``, newest first, as ``get`` reads each."""
        with self.dispatcher.locked_scenario(scenario) as current:
            evaluations = current.store.evaluations
            return tuple(self.settled(evaluations, stored) for stored in evaluations.newest(limit=limit))

    def settled(self, evaluations: EvaluationStore, stored: StoredEvaluation) -> StoredEvaluation:
        """``stored``, failed first when it is queued or running for a service instance other than this one."""
        if stored.state not in ("queued", "running") or stored.owner == self.owner:
            return stored
        try:
            return evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
        except EvaluationStateConflict:
            # Another reader failed it first.
            return evaluations.get(stored.evaluation_id) or stored


__all__ = [
    "MAX_RUNNING_EVALUATIONS",
    "ArtifactEvaluations",
    "EvaluationLimitReached",
    "EvaluationRun",
    "NewArtifact",
    "NewArtifacts",
]
