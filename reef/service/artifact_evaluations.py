"""Transport-independent handling of the artifact evaluations of a scenario.

A client asks Reef to evaluate one artifact: a component of one of the scenario's artifacts, with the other side
of its policy configured (RFC #959). A scenario can also hold settings that Reef evaluates each new artifact with:
when a commit publishes a checkpoint as a new artifact, Reef queues its evaluation. ``ArtifactEvaluations``
checks each request against the scenario and stores it in the scenario's evaluation store. It adds one
``evaluation_rollout`` item for each rollout to the ``evaluations`` work queue, where the evaluation service's
workers run them (``reef.evaluation.rollout_worker``). When every rollout of an evaluation is final,
``EvaluationResults`` reads the scores, stores their ``PolicyEvaluation``, and gives it to the scenario's activation
rule (``Scenario.evaluation_refusal`` says which components Reef evaluates). No aiohttp types appear here.
``reef.service.routes.evaluations`` adapts it to HTTP.

This version evaluates a model artifact (a ``model`` or ``model_adapter`` component) on a runtime that serves an
artifact on request (``InferenceRuntime.serves_artifacts_on_request``), with a Harbor agent, on the Harbor tasks of the eval
split of the scenario's imported tasks (its sample store). It refuses a harness artifact, and a model artifact on any
other runtime.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

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
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    EvaluationRollout,
    agent_harness,
    harbor_agent,
    rollout_item_id,
)
from reef.scenario.scenario import Scenario
from reef.storage.commits import CommitRecord
from reef.storage.evaluations import StoredEvaluation
from reef.storage.observer import RecordObserver
from reef.storage.work_queue import NewWork, WorkItem

logger = logging.getLogger(__name__)

#: The error of an evaluation whose rollouts are no longer in the queue, so they never end.
STOPPED_ERROR = "the service that held this evaluation stopped before it ended"
#: The owner that every evaluation of the work queue stores: any service instance reads its rollouts.
QUEUE_OWNER = "evaluations-queue"
#: How many of a scenario's newest evaluations ``fail_stopped`` reads.
MAX_STOPPED_EVALUATIONS = 1000
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


class EvaluationResults:
    """Ends the artifact evaluations of a service's scenarios as their rollouts end.

    Values: ``dispatcher`` holds the scenarios and the activation of an evaluated artifact; ``queue`` is its work
    queue, which carries the rollouts. ``thread`` runs ``read_results`` between ``open`` and ``close``.

    Purpose: ``ArtifactEvaluations`` holds one: it lists an evaluation's rollouts (``rollouts``) and ends an
    evaluation whose rollouts ended before it was stored (``settle``). At ``open``, it fails the evaluations whose
    rollouts the queue lost (``fail_stopped``).
    """

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher
        self.queue = dispatcher.queue
        self.thread: threading.Thread | None = None
        self.stopped = threading.Event()

    def open(self) -> None:
        """Start the thread that ends evaluations; a second call does nothing."""
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.read_results, name="reef-evaluation-results", daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop the thread. Queued rollouts stay in the queue; another instance of the service ends them."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

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
        """Fail the evaluations whose rollouts the queue lost (``fail_stopped``), then end each evaluation whose
        rollouts are all final, until ``close``."""
        self.fail_stopped()
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

    def fail_stopped(self) -> None:
        """Fail each queued or running evaluation whose rollouts are not in the queue, with ``STOPPED_ERROR``.

        ``accept`` queues the rollouts before it stores the evaluation, so this happens only when the queue lost
        them: an in-memory queue, and a service that stopped. It loads each known scenario once.
        """
        for row in self.dispatcher.list_scenarios():
            scenario = row["scenario"]
            try:
                with self.dispatcher.locked_scenario(scenario) as current:
                    evaluations = current.store.evaluations
                    for stored in evaluations.newest(limit=MAX_STOPPED_EVALUATIONS):
                        if stored.state in ("queued", "running") and not self.rollouts(stored.evaluation_id, scenario):
                            evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
            except Exception:
                logger.exception("could not fail the stopped evaluations of %s", scenario)


class ArtifactEvaluations:
    """Accepts, queues and reads the artifact evaluations of a service's scenarios.

    Values: ``dispatcher`` holds the scenarios. ``queue`` is the dispatcher's work queue, which carries the rollouts
    to the evaluation service's workers. ``new_artifacts`` reports the new artifacts that Reef evaluates with each
    scenario's settings, or is None for no automatic evaluation. ``results`` ends each evaluation as its rollouts
    end.

    Purpose: the evaluation routes call ``start``, ``get``, ``newest`` and ``rollouts_json``, and the scenario
    update route calls ``configure``. ``open`` starts ``results`` and the reader of ``new_artifacts``; ``close``
    stops them. ``get``, ``newest`` and ``rollouts_json`` only read.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        new_artifacts: NewArtifacts | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.new_artifacts = new_artifacts
        self.queue = dispatcher.queue
        self.results = EvaluationResults(dispatcher)
        self.reader: threading.Thread | None = None

    def open(self) -> None:
        """Start ``results`` and the reader of ``new_artifacts``."""
        self.results.open()
        if self.new_artifacts is not None and self.reader is None:
            self.reader = threading.Thread(target=self.read_new_artifacts, name="reef-new-artifacts", daemon=True)
            self.reader.start()

    def close(self) -> None:
        """Stop the reader and ``results``."""
        if self.new_artifacts is not None and self.reader is not None:
            self.new_artifacts.queue.put(None)
            self.reader.join(timeout=5)
            self.reader = None
        self.results.close()

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
        if kind is None:
            raise ValueError(f"component {component!r} is no side of a policy, so Reef cannot evaluate it")
        runtime = current.runtime
        refusal = current.evaluation_refusal(component)
        if refusal is not None or runtime is None:
            # A scenario without a runtime always has a refusal, which names the reason.
            raise ReefError(refusal or f"scenario {current.name!r} has no runtime")
        if request.harness is None:
            raise ValueError(f"component {component!r} is a {kind}: configure the harness that runs it")
        _, release = current.resolve_artifact(artifact_id, component)
        artifact = release.materialize()
        return request, artifact, runtime.artifact_model(artifact)

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
        agent = harbor_agent(request.harness)
        # The rollouts go to the queue before the evaluation exists, so that no reader sees an evaluation whose
        # rollouts are missing. An evaluation runs on all of its tasks or on none: when a step fails, the added
        # rollouts are cancelled and no evaluation is stored.
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
            with self.dispatcher.locked_scenario(scenario) as current:
                stored = current.store.evaluations.create(
                    evaluation_id=evaluation_id,
                    owner=QUEUE_OWNER,
                    request=request,
                    created_at=time.time(),
                    state=state,
                )
        except BaseException as exc:
            error = f"the evaluation was not stored: {type(exc).__name__}: {exc}"
            for item_id in added:
                self.queue.cancel(EVALUATIONS_QUEUE, item_id, error)
            raise
        # The workers can end every rollout before the evaluation exists; the results thread then found nothing to
        # end, so end it here.
        self.results.settle(scenario, evaluation_id)
        return stored

    def rollouts_json(self, scenario: str, evaluation_id: str) -> list[dict[str, Any]] | None:
        """The rollouts of one evaluation, in task and repeat order, each as ``{task, task_index, repeat, state,
        attempt_count, score, error}``; None when ``scenario`` has no evaluation ``evaluation_id``.

        ``state`` is the rollout's queue state. ``score`` is the rollout's ``RolloutScore`` as JSON once it finished,
        else null; ``error`` is the reason of a failed rollout, else null.
        """
        if self.get(scenario, evaluation_id) is None:
            return None
        views = []
        for item in self.results.rollouts(evaluation_id, scenario):
            rollout = EvaluationRollout.from_item(item)
            finished = item.state == "finished" and item.result is not None
            views.append(
                {
                    "task": rollout.task_name,
                    "task_index": rollout.task_index,
                    "repeat": rollout.repeat,
                    "state": item.state,
                    "attempt_count": item.attempt_count,
                    "score": cast(Mapping[str, object], item.result)["score"] if finished else None,
                    "error": item.error,
                }
            )
        return views

    def get(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id`` of ``scenario``, or None."""
        with self.dispatcher.locked_scenario(scenario) as current:
            return current.store.evaluations.get(evaluation_id)

    def newest(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario``, newest first."""
        with self.dispatcher.locked_scenario(scenario) as current:
            return current.store.evaluations.newest(limit=limit)


__all__ = [
    "QUEUE_OWNER",
    "STOPPED_ERROR",
    "ArtifactEvaluations",
    "EvaluationResults",
    "NewArtifact",
    "NewArtifacts",
]
