"""Transport-independent handling of the artifact evaluations of a scenario.

A client asks Reef to evaluate one artifact: a component of one of the scenario's artifacts, with the other side
of its policy configured (RFC #959). A scenario can also hold settings that Reef evaluates each new artifact with:
when a commit publishes a checkpoint as a new artifact, Reef queues its evaluation. ``ArtifactEvaluations``
checks each request against the scenario and stores it in the scenario's evaluation store. It adds one
``evaluation_rollout`` item for each rollout to the ``evaluations`` work queue, where the evaluation service's
workers run them (``reef.evaluation.rollout_worker``). When every rollout of an evaluation is final, it reads the
scores, stores their ``PolicyEvaluation``, and gives it to the scenario's activation rule. No aiohttp types appear
here.
``reef.service.routes.evaluations`` adapts it to HTTP.

This version evaluates a model artifact (a ``model`` or ``model_adapter`` component) on a runtime that serves an
artifact on request (``InferenceRuntime.artifact_model``), with a Harbor agent, on the Harbor tasks of the eval
split of the scenario's imported tasks (its sample store). It refuses a harness artifact, and a model artifact on any
other runtime.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal, cast

from reef.artifact.artifact import Artifact
from reef.core.components import made_artifact_id
from reef.core.errors import CommitPending, ReefError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    ArtifactEvaluationSettings,
    PolicyEvaluation,
    PolicyModel,
    RolloutScore,
    RolloutScores,
)
from reef.core.job_data import TaskSample
from reef.dispatcher import Dispatcher
from reef.evaluation.artifact_endpoint import ArtifactEndpoint
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    CandidateEndpoint,
    CandidateEndpoints,
    EvaluationRollout,
    EvaluationRolloutWorker,
    agent_harness,
    harbor_agent,
    rollout_item_id,
)
from reef.harness.client.tasks import TaskLab
from reef.scenario.scenario import Scenario
from reef.storage.commits import CommitRecord
from reef.storage.coordination import instance_id
from reef.storage.evaluations import EvaluationStateConflict, EvaluationStore, StoredEvaluation
from reef.storage.observer import RecordObserver
from reef.storage.work_queue import NewWork, WorkItem

logger = logging.getLogger(__name__)

#: The evaluation workers that a service runs when it runs its own (the one-process layout).
LOCAL_ROLLOUT_WORKERS = 4
#: The error of an evaluation whose rollouts are no longer in the queue, so they never end.
STOPPED_ERROR = "the service that held this evaluation stopped before it ended"
#: The owner that every evaluation of the work queue stores: any service instance reads its rollouts.
QUEUE_OWNER = "evaluations-queue"
#: The seconds between two reads of the queue's changes when nothing changed.
RESULT_POLL_SECONDS = 0.2
#: How often Reef tries to activate an artifact that passed its evaluation while training results wait to commit,
#: and the seconds it waits after the first attempt, growing with each.
ACTIVATION_ATTEMPTS = 5
ACTIVATION_RETRY_SECONDS = 1.0
#: The scenario name that an evaluation's model calls carry; the artifact endpoint serves no scenario.
EVALUATION_SCENARIO = "artifact-evaluation"


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
        # Only training makes the artifacts that Reef evaluates on its own: an activation or a new training start
        # names one that exists, and a person who imports an artifact decides what to do with it.
        if not commit.trained_artifact:
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


class LocalCandidateEndpoints(CandidateEndpoints):
    """The candidate's model, served in the process of the evaluation workers: a loopback endpoint that answers from
    the candidate artifact with the scenario's runtime.

    Values: ``dispatcher`` loads the scenario of each rollout (on demand, in a process that runs no control).
    Purpose: the evaluation workers of this process open it around each rollout. The candidate is resolved from the
    scenario's repository, so neither control nor the inference service serves it.
    """

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    @contextmanager
    def open(self, rollout: EvaluationRollout) -> Iterator[CandidateEndpoint]:
        with self.dispatcher.locked_scenario(rollout.scenario) as current:
            runtime = current.runtime
            if runtime is None:
                raise ReefError(f"scenario {rollout.scenario!r} has no runtime")
            _, release = current.resolve_artifact(rollout.artifact_id, rollout.component)
            artifact = release.materialize()
        with ArtifactEndpoint(runtime, artifact) as endpoint:
            yield CandidateEndpoint(endpoint.url, None)


class ArtifactEvaluations:
    """Accepts, queues, reads and ends the artifact evaluations of a service's scenarios.

    Values: ``dispatcher`` holds the scenarios. ``queue`` is the dispatcher's work queue, which carries the rollouts.
    ``endpoints`` opens a candidate's model for the workers that this object runs itself, or is None when the
    evaluation service runs elsewhere. ``new_artifacts`` reports the new artifacts that Reef evaluates with each
    scenario's settings, or is None for no automatic evaluation. ``workers`` are ``LOCAL_ROLLOUT_WORKERS``
    evaluation workers with ``endpoints``, none without them.

    Purpose: the evaluation routes call ``start``, ``get`` and ``newest``, and the scenario update route calls
    ``configure``. ``open`` starts the threads that evaluate new artifacts, that run this object's workers, and
    that end each evaluation when its rollouts end; ``close`` stops them.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        lab: TaskLab | None = None,
        new_artifacts: NewArtifacts | None = None,
        endpoints: CandidateEndpoints | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.new_artifacts = new_artifacts
        self.queue = dispatcher.queue
        self.workers = (
            []
            if endpoints is None
            else [
                EvaluationRolloutWorker(self.queue, endpoints, worker_id=instance_id("evaluation"), lab=lab)
                for _ in range(LOCAL_ROLLOUT_WORKERS)
            ]
        )
        self.reader: threading.Thread | None = None
        self.results: threading.Thread | None = None
        self.stopped = threading.Event()

    def open(self) -> None:
        """Start the workers, the thread that ends evaluations, and the reader of ``new_artifacts``."""
        self.stopped.clear()
        for worker in self.workers:
            worker.start()
        if self.results is None:
            self.results = threading.Thread(target=self.read_results, name="reef-evaluation-results", daemon=True)
            self.results.start()
        if self.new_artifacts is not None and self.reader is None:
            self.reader = threading.Thread(target=self.read_new_artifacts, name="reef-new-artifacts", daemon=True)
            self.reader.start()

    def close(self) -> None:
        """Stop the threads. Queued rollouts stay in the queue; another instance of the service ends them."""
        self.stopped.set()
        if self.new_artifacts is not None and self.reader is not None:
            self.new_artifacts.queue.put(None)
            self.reader.join(timeout=5)
            self.reader = None
        for worker in self.workers:
            worker.stop()
        if self.results is not None:
            self.results.join(timeout=5)
            self.results = None

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
        """Check a client's request ``body`` against ``scenario``, store it, and queue its rollouts.

        Raise ``ValueError`` or ``ReefError`` for a request that the scenario cannot run, ``UnknownScenario``
        for an unknown scenario, and ``ArtifactNotFound`` for an unknown artifact.
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
        """Check and store the evaluation of ``artifact_id`` with ``settings``, and add its rollouts to the queue.

        A client's evaluation is stored ``running``; an automatic one ``queued``, until a worker takes one of its
        rollouts.
        """
        with self.dispatcher.locked_scenario(scenario) as current:
            request, _artifact, model = self.checked(current, artifact_id, settings)
            held = {
                stored.sample_id: stored.sample.task_path
                for stored in current.store.samples.imported_tasks(split="eval")
                if isinstance(stored.sample, TaskSample)
            }
            unknown = sorted(set(request.task_names) - set(held))
            if unknown:
                raise ValueError(f"the eval split of the imported tasks holds no tasks {unknown}")
            names = request.task_names or tuple(held)
            if not names:
                raise ValueError("the eval split of the imported tasks holds no task to evaluate on")
            evaluation_id = uuid.uuid4().hex
            stored = current.store.evaluations.create(
                evaluation_id=evaluation_id, owner=QUEUE_OWNER, request=request, created_at=time.time(), state=state
            )
        agent = harbor_agent(request.harness)
        added: list[str] = []
        try:
            for index, name in enumerate(names):
                for repeat in range(request.rollouts_per_task):
                    rollout = EvaluationRollout(
                        scenario,
                        evaluation_id,
                        request.artifact_id,
                        request.component,
                        index,
                        name,
                        str(held[name]),
                        repeat,
                        model,
                        agent,
                        request.environment,
                    )
                    self.queue.add(
                        NewWork(
                            EVALUATIONS_QUEUE,
                            rollout.item_id(),
                            scenario,
                            EVALUATION_ROLLOUT_KIND,
                            rollout.to_payload(),
                        )
                    )
                    added.append(rollout.item_id())
        except BaseException as exc:
            # An evaluation runs on all of its tasks or on none: the added rollouts are cancelled and it fails.
            error = f"could not queue the rollouts: {type(exc).__name__}: {exc}"
            for item_id in added:
                self.queue.cancel(EVALUATIONS_QUEUE, item_id, error)
            self.end(scenario, evaluation_id, request, result=None, error=error)
            raise
        return stored

    def rollouts(self, evaluation_id: str, scenario: str) -> tuple[WorkItem, ...]:
        """The rollout items of one evaluation, in task and repeat order."""
        prefix = rollout_item_id(evaluation_id, 0, 0).rsplit(":", 2)[0] + ":"
        return tuple(
            sorted(
                (
                    item
                    for item in self.queue.items(EVALUATIONS_QUEUE, scenario=scenario, kind=EVALUATION_ROLLOUT_KIND)
                    if item.id.startswith(prefix)
                ),
                key=lambda item: item.id,
            )
        )

    def read_results(self) -> None:
        """End each evaluation whose rollouts are all final, until ``close``."""
        sequence = 0
        while not self.stopped.is_set():
            try:
                changes = self.queue.read_changes(EVALUATIONS_QUEUE, sequence)
            except Exception:
                # The store can fail for a moment; the thread reads again after its poll time.
                logger.exception("could not read the evaluations queue")
                self.stopped.wait(RESULT_POLL_SECONDS)
                continue
            touched: dict[str, str] = {}
            for item in changes:
                sequence = item.changed_sequence
                if item.kind == EVALUATION_ROLLOUT_KIND:
                    # The payload names the evaluation; ``EvaluationRollout`` wrote it (``to_payload``).
                    touched[str(item.payload["evaluation_id"])] = item.scenario
            for evaluation_id, scenario in touched.items():
                try:
                    self.settle(scenario, evaluation_id)
                except Exception:
                    logger.exception("could not end evaluation %s of %s", evaluation_id, scenario)
            if not changes:
                self.stopped.wait(RESULT_POLL_SECONDS)

    def settle(self, scenario: str, evaluation_id: str) -> None:
        """Start the evaluation once a worker took one of its rollouts; end it once all of them are final."""
        items = self.rollouts(evaluation_id, scenario)
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.get(evaluation_id)
            if stored is None or stored.state in ("finished", "failed"):
                return
            if not all(item.final for item in items):
                if stored.state == "queued" and any(item.state in ("assigned", "running") for item in items):
                    current.store.evaluations.begin(evaluation_id)
                return
            request = stored.request
        rollouts = [EvaluationRollout.from_item(item) for item in items]
        names = list(dict.fromkeys(rollout.task_name for rollout in rollouts))
        scores = [
            (
                RolloutScore.from_json(cast(Mapping[str, object], item.result)["score"])
                if item.state == "finished"
                else RolloutScore(None)
            )
            for item in items
        ]
        result = PolicyEvaluation(
            agent_harness(harbor_agent(request.harness)), rollouts[0].model, RolloutScores.by_task(names, scores)
        )
        self.end(scenario, evaluation_id, request, result=result, error=None)

    def end(
        self,
        scenario: str,
        evaluation_id: str,
        request: ArtifactEvaluationRequest,
        *,
        result: PolicyEvaluation | None,
        error: str | None,
    ) -> None:
        """Store the end of an evaluation: its ``result``, or its ``error`` when ``result`` is None. A finished one
        activates its artifact when the scenario's activation rule says so (``activates_evaluated``)."""
        activates = False
        try:
            with self.dispatcher.locked_scenario(scenario) as current:
                evaluations = current.store.evaluations
                current_state = evaluations.get(evaluation_id)
                if current_state is not None and current_state.state == "queued":
                    evaluations.begin(evaluation_id)
                if result is not None:
                    evaluations.finish(evaluation_id, result=result, finished_at=time.time())
                    activates = current.activation_rule.activates_evaluated(request.artifact_id, result)
                else:
                    evaluations.fail(evaluation_id, error=error or "the evaluation failed", finished_at=time.time())
        except (ReefError, ValueError) as exc:
            # The scenario was deleted, or the evaluation ended already: nothing is left to record it in.
            logger.warning("could not store the end of evaluation %s: %s", evaluation_id, exc)
            return
        if activates:
            self.activate(scenario, request)

    def activate(self, scenario: str, request: ArtifactEvaluationRequest) -> None:
        """Activate the evaluated artifact, waiting while a training result waits for its commit."""
        for attempt in range(ACTIVATION_ATTEMPTS):
            try:
                self.dispatcher.activate(scenario, request.artifact_id, request.component)
                return
            except CommitPending:
                time.sleep(ACTIVATION_RETRY_SECONDS * (attempt + 1))
            except (ReefError, ValueError) as exc:
                logger.warning("could not activate artifact %s of %s: %s", request.artifact_id, scenario, exc)
                return
        logger.warning(
            "artifact %s of %s passed its evaluation but stayed inactive: training kept committing",
            request.artifact_id,
            scenario,
        )

    def get(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id`` of ``scenario``, or None; one whose rollouts left the queue is failed."""
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.get(evaluation_id)
            return None if stored is None else self.settled(scenario, current.store.evaluations, stored)

    def newest(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario``, newest first, as ``get`` reads each."""
        with self.dispatcher.locked_scenario(scenario) as current:
            evaluations = current.store.evaluations
            return tuple(self.settled(scenario, evaluations, stored) for stored in evaluations.newest(limit=limit))

    def settled(self, scenario: str, evaluations: EvaluationStore, stored: StoredEvaluation) -> StoredEvaluation:
        """``stored``, failed first when it is queued or running but its rollouts are not in the queue."""
        if stored.state not in ("queued", "running"):
            return stored
        if stored.owner == QUEUE_OWNER and self.rollouts(stored.evaluation_id, scenario):
            return stored
        try:
            return evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
        except EvaluationStateConflict:
            # Another reader failed it first.
            return evaluations.get(stored.evaluation_id) or stored


__all__ = [
    "LOCAL_ROLLOUT_WORKERS",
    "QUEUE_OWNER",
    "STOPPED_ERROR",
    "ArtifactEvaluations",
    "LocalCandidateEndpoints",
    "NewArtifact",
    "NewArtifacts",
]
