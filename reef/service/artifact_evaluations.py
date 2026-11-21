"""Transport-independent handling of the artifact evaluations of a scenario.

A client asks Reef to evaluate one artifact: a component of one of the scenario's artifacts, with the other side
of its policy configured (RFC #959). A scenario can also hold settings that Reef evaluates each new artifact with:
when a commit publishes a checkpoint as a new artifact, Reef queues its evaluation. ``ArtifactEvaluations``
checks each request against the scenario and stores it in the scenario's evaluation store. It adds one
``evaluation_rollout`` item for each rollout to the ``evaluations`` work queue, where the evaluation service's
workers run them (``reef.evaluation.rollout_worker``). With ``max_concurrent_rollouts``, it keeps at most that many
rollouts that are not final in the queue, and adds the next ones as rollouts end (RFC #1276). A retry runs the
rollouts that failed again, and a resume runs the rollouts that a cancel or a stop ended or that never started; both
keep the same limit. With
``max_failed_fraction``, it stops an evaluation whose ended rollouts fail too often (RFC #1287). When every rollout of an
evaluation is final, ``EvaluationResults`` reads the scores, stores their ``PolicyEvaluation``, and gives it to the
scenario's activation rule (``Scenario.evaluation_refusal`` says which components Reef evaluates). No aiohttp types
appear here.
``reef.service.routes.evaluations`` adapts it to HTTP.

This version evaluates a model artifact (a ``model`` or ``model_adapter`` component) on a runtime that serves an
artifact on request (``InferenceRuntime.serves_any_artifact``), on the Harbor tasks of the eval split of the
scenario's imported tasks (its samples). Its harness is a Harbor agent, or a harness tree that an artifact item of
the scenario's owner names, also from another scenario. It refuses a harness artifact, and a model artifact on any
other runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast, overload

from reef.artifact.artifact import Artifact
from reef.artifact.sources import ArtifactSources
from reef.core.errors import CommitPendingError, InternalError, ReefError, ScenarioNotFoundError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    ArtifactEvaluationSettings,
    Caller,
    EvaluationCall,
    EvaluationCallTotals,
    EvaluationCollection,
    PolicyEvaluation,
    PolicyModel,
    RolloutScore,
    RolloutScores,
    harness_item,
    harness_item_config,
    harness_item_env,
    mean_benchmark_score,
    mean_score,
)
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import TrajectoryBuilder
from reef.core.trajectory_index import TrajectoryIndex
from reef.core.values import deep_merge
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import EvaluatedActivation
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    TASK_TEMPLATE_KIND,
    EvaluationRollout,
    TreeHarness,
    harbor_agent,
    rollout_item_id,
    template_item_id,
)
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.adapters.parts import PLACEHOLDER
from reef.harness.harbor.agent import can_run_in_task_container
from reef.harness.tree import HarnessTree, validate_tree
from reef.scenario.scenario import Scenario
from reef.service.collections import collection_tasks
from reef.service.data import DataAccess, DataStoreUnavailableError
from reef.storage.commits import CommitRecord
from reef.storage.data import DataStore, StoredItem, record_source
from reef.storage.evaluations import (
    EvaluationStart,
    EvaluationStartConflictError,
    EvaluationStateConflictError,
    EvaluationStore,
    StoredEvaluation,
    check_evaluation_start_id,
)
from reef.storage.item_kinds import evaluation_call_id
from reef.storage.observer import RecordObserver
from reef.storage.record_items import (
    evaluation_call_items,
    evaluation_call_summary,
    evaluation_trajectory_items,
    stored_evaluation_call,
)
from reef.storage.task_collections import CollectionNotFoundError
from reef.storage.work_queue import JsonValue, NewWork, WorkItem

logger = logging.getLogger(__name__)

#: The error of an evaluation whose rollouts are no longer in the queue, so they never end.
STOPPED_ERROR = "the service that held this evaluation stopped before it ended"
#: The error of an evaluation that a client cancelled.
CANCELLED_ERROR = "cancelled by request"
#: The error of each rollout that was not final when its evaluation stopped for failed rollouts.
FAILURES_ERROR = "stopped: too many rollouts failed"
#: The rollouts that must end before Reef checks ``max_failed_fraction``; a smaller evaluation checks at its end.
FAILURE_CHECK_ROLLOUTS = 10
#: The last seconds whose ended rollouts give the rate of an evaluation's time left.
ETA_WINDOW_SECONDS = 1200.0
#: The owner that every evaluation of the work queue stores: any service instance reads its rollouts.
QUEUE_OWNER = "evaluations-queue"
#: How many of a scenario's newest evaluations ``fail_stopped`` reads.
MAX_STOPPED_EVALUATIONS = 1000
#: The metadata of a rollout that a retry or a resume runs again once ``max_concurrent_rollouts`` lets it
#: (``rollout_waits``).
RETRY_WAIT_METADATA = "waits_for_retry"
#: The seconds between two reads of the queue's changes when nothing changed.
RESULT_POLL_SECONDS = 0.2
#: How many tasks of an evaluation on e2b have template items before the evaluation adds their rollouts: from the task
#: of the next rollout to add. The templates of these tasks build while the evaluation's rollouts run.
TEMPLATE_TASKS_AHEAD = 32
#: How often Reef tries to activate an artifact that passed its evaluation while training results wait to commit,
#: and the seconds it waits after the first attempt, growing with each.
ACTIVATION_ATTEMPTS = 5
ACTIVATION_RETRY_SECONDS = 1.0


@dataclass(frozen=True)
class NewArtifact:
    """An artifact that a training commit published: the scenario, the artifact's ID, the component whose artifact
    it is (None for a scenario with one component), and ``revision``, the storage revision of the commit's snapshot.

    ``NewArtifacts`` makes it. ``ArtifactEvaluations`` evaluates it with the scenario's settings when the commit
    made the artifact, and skips a component that the snapshot carried from an earlier commit.
    """

    scenario: str
    artifact_id: str
    component: str | None
    revision: str


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
        revision = commit.snapshot.revision
        selected = commit.artifact_ids
        if selected is None:
            raise ValueError(f"commit {commit.commit_number} made an artifact and names no artifact ID")
        names = tuple(selected) if commit.component is None else (commit.component,)
        for name in names:
            component = None if len(selected) == 1 and commit.component is None else name
            self.queue.put(NewArtifact(commit.scenario, selected[name], component, revision))

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


def rollout_failed(item: WorkItem) -> bool:
    """Whether ``item``, a final rollout, failed: its work item failed, or it finished with no score."""
    if item.state != "finished":
        return True
    return RolloutScore.from_json(cast(Mapping[str, object], item.result)["score"]).is_failed


def rollout_cancelled(item: WorkItem) -> bool:
    """Whether a cancel or a stop of its evaluation ended ``item``, and not the rollout's own work."""
    return rollout_ended(item) and item.state == "failed" and item.error in (CANCELLED_ERROR, FAILURES_ERROR)


def rollout_waits(item: WorkItem) -> bool:
    """Whether ``item`` is a final rollout that a retry or a resume chose to run again, and that waits until fewer
    than ``max_concurrent_rollouts`` rollouts of its evaluation are not final (``RETRY_WAIT_METADATA``). Its evaluation
    runs, and the rollout counts as not ended."""
    return item.final and item.metadata.get(RETRY_WAIT_METADATA) is True


def settled_state(item: WorkItem) -> tuple[str, int, bool]:
    """The parts of the rollout item ``item`` that ``settle`` reads: its state, its attempt count and whether it waits
    for a retry (``rollout_waits``). A result or an error comes with a new state."""
    return item.state, item.attempt_count, rollout_waits(item)


def rollout_ended(item: WorkItem) -> bool:
    """Whether ``item`` is a final rollout that does not wait for a retry (``rollout_waits``)."""
    return item.final and not rollout_waits(item)


def evaluation_progress(stored: StoredEvaluation, items: Sequence[WorkItem], now: float) -> dict[str, JsonValue]:
    """The progress of ``stored`` from ``items``, its rollouts in the queue (RFC #1287): ``{total, ended, scored,
    failed, cancelled, running, mean_score, eta_seconds, benchmarks}``.

    ``ended`` counts the final rollouts that no cancel or stop ended. ``mean_score`` follows the request's weighting
    and includes only scores from those rollouts. ``eta_seconds`` divides the rollouts left by the rate in the last
    ``ETA_WINDOW_SECONDS``; it is None when none ended in that time, or when the evaluation ended.
    ``benchmarks`` has one entry for each benchmark of the request's collection, in its order: ``benchmark`` and
    the same counts and ``mean_score`` for the rollouts of the benchmark's tasks. A task in two benchmarks counts
    in both. It is empty without a collection.
    """
    request = stored.request

    def _counts(rollouts: Sequence[WorkItem], task_count: int) -> dict[str, JsonValue]:
        ended = [item for item in rollouts if rollout_ended(item) and not rollout_cancelled(item)]
        scores = [
            RolloutScore.from_json(cast(Mapping[str, object], item.result)["score"])
            for item in ended
            if item.state == "finished"
        ]
        numbers = [score.value for score in scores if score.value is not None]
        scored = len(numbers)
        full = sum(score >= 1 for score in numbers)
        part = sum(0 < score < 1 for score in numbers)
        cancelled = sum(rollout_cancelled(item) for item in rollouts)
        total = task_count * request.rollouts_per_task
        return {
            "score_levels": {
                "full": full,
                "part": part,
                "zero": scored - full - part,
                "failed": len(ended) - scored + cancelled,
                "pending": total - len(ended) - cancelled,
            },
            "total": total,
            "ended": len(ended),
            "scored": scored,
            "failed": len(ended) - scored,
            "cancelled": cancelled,
            "running": sum(item.state in ("assigned", "running") for item in rollouts),
            "mean_score": mean_score(scores),
        }

    window = min(ETA_WINDOW_SECONDS, now - stored.created_at)
    recent = sum(
        rollout_ended(item)
        and not rollout_cancelled(item)
        and item.ended_at is not None
        and item.ended_at >= now - window
        for item in items
    )
    counts = _counts(items, len(request.task_names))
    eta = None
    if stored.state in ("queued", "running") and recent and window > 0:
        eta = (cast(int, counts["total"]) - cast(int, counts["ended"])) * window / recent
    benchmarks: list[JsonValue] = []
    if request.collection is not None:
        # The payload names the rollout's task; ``EvaluationRollout`` wrote it (``to_payload``).
        by_task: dict[str, list[WorkItem]] = {}
        for item in items:
            by_task.setdefault(str(item.payload["task_name"]), []).append(item)
        benchmark_means: list[float | None] = []
        for benchmark, task_names in request.collection.benchmarks:
            rollouts = [item for name in task_names for item in by_task.get(name, ())]
            benchmark_counts = _counts(rollouts, len(task_names))
            benchmarks.append({"benchmark": benchmark, **benchmark_counts})
            benchmark_means.append(cast(float | None, benchmark_counts["mean_score"]))
        if request.collection.average == "benchmark":
            counts["mean_score"] = mean_benchmark_score(benchmark_means)
    return {**counts, "eta_seconds": eta, "benchmarks": benchmarks}


#: The most ended evaluations whose progress one ``EndedProgress`` keeps.
ENDED_PROGRESS_ENTRIES = 10_000


class EndedProgress:
    """The progress of ended evaluations, so that a list of evaluations does not read their rollouts again.

    Values: ``entries`` maps the key ``(scenario, evaluation_id, finished_at)`` of an ended evaluation to its
    ``evaluation_progress``, computed from rollouts that had all ended (``rollout_ended``). It holds at most
    ``ENDED_PROGRESS_ENTRIES`` keys, in the order of their last use. ``lock`` guards ``entries`` for the threads of
    concurrent routes.

    The rollouts of an ended evaluation stay ended until a retry or a resume reopens the evaluation, and its next end
    gives it a new ``finished_at``. So a key describes one run of the evaluation, and its progress does not change.
    An entry outlives the queue's deletion of the rollouts after their retention: the list then shows the progress
    that they gave, not null.

    Operations: ``read(scenario, stored)`` is the entry of ``stored``'s key, or None when ``stored`` runs or has no
    entry. ``add(scenario, stored, items, progress)`` keeps ``progress`` under ``stored``'s key when ``stored`` ended
    and every one of ``items``, its rollouts, ended; else it keeps nothing. Above ``ENDED_PROGRESS_ENTRIES``, the
    least recently used key goes.

    Purpose: ``ArtifactEvaluations.status_json`` reads an entry before it reads an evaluation's rollouts, and adds
    one after it computed the progress of an ended evaluation.
    """

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[str, str, float], dict[str, JsonValue]] = OrderedDict()
        self.lock = threading.Lock()

    def read(self, scenario: str, stored: StoredEvaluation) -> dict[str, JsonValue] | None:
        if stored.finished_at is None:
            return None
        key = (scenario, stored.evaluation_id, stored.finished_at)
        with self.lock:
            progress = self.entries.get(key)
            if progress is not None:
                self.entries.move_to_end(key)
            return progress

    def add(
        self, scenario: str, stored: StoredEvaluation, items: Sequence[WorkItem], progress: dict[str, JsonValue]
    ) -> None:
        # A cancel stores the ended evaluation before it ends the rollouts: a reader in between keeps nothing.
        if stored.finished_at is None or not all(rollout_ended(item) for item in items):
            return
        with self.lock:
            self.entries[(scenario, stored.evaluation_id, stored.finished_at)] = progress
            if len(self.entries) > ENDED_PROGRESS_ENTRIES:
                self.entries.popitem(last=False)


def planned_rollout(
    seed: EvaluationRollout, request: ArtifactEvaluationRequest, task_index: int, repeat: int
) -> EvaluationRollout:
    """The rollout at ``task_index`` and ``repeat`` with the accepted task and the seed's shared inputs."""
    return replace(
        seed,
        task_index=task_index,
        task_name=request.task_names[task_index],
        repeat=repeat,
        task_digest=request.task_digest(task_index),
    )


def rollout_work(rollout: EvaluationRollout) -> NewWork:
    """The work item of ``rollout`` in the evaluations queue."""
    return NewWork(
        EVALUATIONS_QUEUE, rollout.item_id(), rollout.scenario, EVALUATION_ROLLOUT_KIND, rollout.to_payload()
    )


def template_work(rollout: EvaluationRollout) -> NewWork:
    """The template item of the task of ``rollout`` in the evaluations queue. Its payload is the task's first
    rollout, so the item of a task is the same whichever rollout of the task makes it."""
    first = replace(rollout, repeat=0)
    return NewWork(
        EVALUATIONS_QUEUE,
        template_item_id(rollout.evaluation_id, rollout.task_index),
        rollout.scenario,
        TASK_TEMPLATE_KIND,
        first.to_payload(),
    )


#: The calls that one read of the data store gives to ``StoredCalls`` when it is read in order. A batch holds the
#: bodies of these calls at once, and each body holds its whole conversation.
READ_CALLS = 16


@dataclass(frozen=True)
class StoredCalls(Sequence[EvaluationCall]):
    """The kept calls of one rollout, which the data store reads one at a time.

    Values: ``data`` is the data store and ``owner`` the owner of the calls; ``items`` are the items of kept evaluation
    calls of ``owner``, without their content, in attempt and then index order. Element ``i`` is the call of
    ``items[i]`` with its bodies, read from ``data`` when it is asked for; a call that the store no longer keeps
    raises ``LookupError``.

    Purpose: ``ArtifactEvaluations.rollout_calls`` gives it, so that ``attempt_trajectories`` holds the bodies of one
    call at a time: each body holds the whole conversation, and a rollout can have thousands of calls.
    """

    data: DataStore
    owner: str
    items: tuple[StoredItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> EvaluationCall: ...

    @overload
    def __getitem__(self, index: slice) -> StoredCalls: ...

    def __getitem__(self, index: int | slice) -> EvaluationCall | StoredCalls:
        if isinstance(index, slice):
            return replace(self, items=self.items[index])
        item_id = self.items[index].item_id
        stored = self.data.read(self.owner, item_id)
        if stored is None:
            raise LookupError(f"the data store no longer keeps call {item_id!r}")
        return stored_evaluation_call(stored)

    def __iter__(self) -> Iterator[EvaluationCall]:
        """The calls in order, read ``READ_CALLS`` at a time: one read of the data store for each batch."""
        for start in range(0, len(self.items), READ_CALLS):
            batch = [item.item_id for item in self.items[start : start + READ_CALLS]]
            read = {stored.item_id: stored for stored in self.data.read_many(self.owner, batch)}
            for item_id in batch:
                if item_id not in read:
                    raise LookupError(f"the data store no longer keeps call {item_id!r}")
                yield stored_evaluation_call(read.pop(item_id))


def call_record(scenario: str, evaluation_id: str, call: EvaluationCall) -> AgentRecord:
    """The ``AgentRecord`` of the evaluation call ``call`` of ``scenario``: its ID is the call's
    ``evaluation_call_id``, its payload the request body plus ``response``, the response body, and its time the call's
    ``started_at``."""
    summary = call.summary
    record_id = evaluation_call_id(
        evaluation_id, summary.task_index, summary.repeat, summary.attempt, summary.index, summary.caller
    )
    payload = {**(call.request or {}), "response": dict(call.response or {})}
    return AgentRecord(record_id, scenario, RequestType.INFERENCE, payload, summary.started_at)


def attempt_trajectories(
    scenario: str,
    evaluation_id: str,
    item: WorkItem,
    calls: Sequence[EvaluationCall],
    caller: Caller = "agent",
    *,
    include_records: bool = True,
) -> list[dict[str, Any]]:
    """The ATIF trajectories of each attempt of the rollout of ``item`` that has a 200 call among ``calls``, in
    attempt order. ``calls`` are kept calls of ``caller`` of that rollout, in attempt and then index order.

    Each 200 call of an attempt is one ``AgentRecord`` (``call_record``). ``TrajectoryIndex`` links the calls of one
    attempt into conversations, as it links the records of training samples: an agent that plays several roles, each
    with its own messages, has one conversation for each. ``TrajectoryBuilder`` stitches each conversation, without the
    calls that a retry replaced; the conversations are in the order of their first calls. With ``include_records``
    false, a trajectory has no ``extra.reef.records``. The attempt that the rollout's score came from is the last
    attempt of a finished rollout: each of its agent trajectories has the score's value as its reward, null for a
    failed score, and every other trajectory has no reward. Each trajectory carries ``extra.reef.evaluation``:
    ``{evaluation_id, task, task_index, repeat, attempt, caller, conversation, scored}``, where ``conversation``
    counts from 0 in its attempt.

    It reads ``calls`` twice, in order: once to link the calls, and once to add each call to its conversation
    (``TrajectoryBuilder``). It keeps no call between two reads. So with ``StoredCalls`` it holds the bodies of one
    batch of calls at a time, and the trajectories that it returns.
    """
    rollout = EvaluationRollout.from_item(item)
    finished = item.state == "finished" and item.result is not None
    score = RolloutScore.from_json(cast(Mapping[str, object], item.result)["score"]).value if finished else None
    indexes: dict[int, TrajectoryIndex] = {}
    record_ids: dict[int, list[str]] = {}
    for call in calls:
        if call.summary.status != 200:
            continue
        record = call_record(scenario, evaluation_id, call)
        indexes.setdefault(call.summary.attempt, TrajectoryIndex()).add(record)
        record_ids.setdefault(call.summary.attempt, []).append(record.agent_record_id)
    # Each conversation of each attempt, by the ID of each of its calls; a call that a retry replaced is in none.
    builders: dict[tuple[int, int], TrajectoryBuilder] = {}
    conversation_of: dict[str, tuple[int, int]] = {}
    for attempt in sorted(record_ids):
        for conversation, linked in enumerate(indexes[attempt].conversations(record_ids[attempt])):
            builders[attempt, conversation] = TrajectoryBuilder(include_records)
            conversation_of.update(dict.fromkeys(linked, (attempt, conversation)))
    for call in calls:
        if call.summary.status != 200:
            continue
        record = call_record(scenario, evaluation_id, call)
        place = conversation_of.get(record.agent_record_id)
        if place is not None:
            builders[place].add(record)
    trajectories = []
    for (attempt, conversation), builder in builders.items():
        scored = caller == "agent" and finished and attempt == item.attempt_count
        trajectory = builder.sample(reward=score if scored else None).trajectory
        evaluation = {
            "evaluation_id": evaluation_id,
            "task": rollout.task_name,
            "task_index": rollout.task_index,
            "repeat": rollout.repeat,
            "attempt": attempt,
            "caller": caller,
            "conversation": conversation,
            "scored": scored,
        }
        extra = trajectory["extra"]
        trajectories.append({**trajectory, "extra": {**extra, "reef": {**extra["reef"], "evaluation": evaluation}}})
    return trajectories


def verifier_trajectory(item: StoredItem, rollout: EvaluationRollout) -> dict[str, Any]:
    """The ATIF document of the trajectory item ``item``, which the verifier of ``rollout`` wrote
    (``evaluation_trajectory_item``), with ``extra.reef.evaluation``: ``{evaluation_id, task, task_index, repeat,
    attempt, caller, file, scored}``, where ``caller`` is ``verifier``, ``file`` the name of its file and ``scored``
    false. It adds an ``extra`` or ``extra.reef`` object that the document has not."""
    location = item.metadata["evaluation"]
    document = dict(cast(Mapping[str, Any], item.content))
    extra = document.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    reef = extra.get("reef")
    reef = reef if isinstance(reef, Mapping) else {}
    evaluation = {
        "evaluation_id": location["evaluation_id"],
        "task": rollout.task_name,
        "task_index": rollout.task_index,
        "repeat": rollout.repeat,
        "attempt": location["attempt"],
        "caller": "verifier",
        "file": location["name"],
        "scored": False,
    }
    return {**document, "extra": {**extra, "reef": {**reef, "evaluation": evaluation}}}


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
        """The rollout items of one evaluation, in task and repeat order. The queue reads only the items of this
        evaluation: their IDs start with one prefix."""
        prefix = rollout_item_id(evaluation_id, 0, 0).rsplit(":", 2)[0] + ":"
        found = self.queue.items(EVALUATIONS_QUEUE, scenario=scenario, kind=EVALUATION_ROLLOUT_KIND, id_prefix=prefix)
        return tuple(
            sorted(found, key=lambda item: (cast(int, item.payload["task_index"]), cast(int, item.payload["repeat"])))
        )

    def templates(self, evaluation_id: str, scenario: str) -> dict[int, WorkItem]:
        """The template items of one evaluation (``template_work``), by task index. Only an evaluation on e2b has
        them."""
        prefix = template_item_id(evaluation_id, 0).rsplit(":", 1)[0] + ":"
        found = self.queue.items(EVALUATIONS_QUEUE, scenario=scenario, kind=TASK_TEMPLATE_KIND, id_prefix=prefix)
        return {cast(int, item.payload["task_index"]): item for item in found}

    def read_results(self) -> None:
        """Fail the evaluations whose rollouts the queue lost (``fail_stopped``), then end each evaluation whose
        rollouts are all final, until ``close``."""
        self.fail_stopped()
        sequence = 0
        # The state of each rollout that its evaluation was settled with (``settled_state``). A change that leaves it,
        # such as a lease extension or a progress line, settles nothing: a settle would find what the last one found.
        settled: dict[str, tuple[str, int, bool]] = {}
        while not self.stopped.is_set():
            try:
                changes = self.queue.read_changes(EVALUATIONS_QUEUE, sequence)
            except Exception:
                # The store can fail for a moment; the thread reads again after its poll time.
                logger.exception("could not read the evaluations queue")
                self.stopped.wait(RESULT_POLL_SECONDS)
                continue
            touched: dict[str, str] = {}
            states: dict[str, dict[str, tuple[str, int, bool]]] = {}
            for item in changes:
                sequence = item.changed_sequence
                if item.kind not in (EVALUATION_ROLLOUT_KIND, TASK_TEMPLATE_KIND):
                    continue
                if settled.get(item.id) != settled_state(item):
                    # The payload names the evaluation; ``EvaluationRollout`` wrote it (``to_payload``).
                    evaluation_id = str(item.payload["evaluation_id"])
                    touched[evaluation_id] = item.scenario
                    states.setdefault(evaluation_id, {})[item.id] = settled_state(item)
            for evaluation_id, scenario in touched.items():
                # A failed settle keeps the earlier states, so the next change of a rollout settles again.
                if self.settle_changed(scenario, evaluation_id):
                    settled.update(states[evaluation_id])
            if not changes:
                self.stopped.wait(RESULT_POLL_SECONDS)

    def settle_changed(self, scenario: str, evaluation_id: str) -> bool:
        """Settle the evaluation ``evaluation_id`` of ``scenario``, whose rollout items changed (``settle``), and log
        a failure; return whether the settle ended without one. The queue keeps the rollout items of a deleted
        scenario until their retention ends, and the evaluation went with the scenario, so an unknown scenario is no
        failure."""
        try:
            self.settle(scenario, evaluation_id)
        except ScenarioNotFoundError:
            logger.debug("evaluation %s: its scenario %s no longer exists", evaluation_id, scenario)
        except Exception:
            logger.exception("could not end evaluation %s of %s", evaluation_id, scenario)
            return False
        return True

    def settle_rollout_prefix(self, scenario: str, stored: StoredEvaluation) -> bool | None:
        """Admit rollouts from a contiguous queue prefix and return whether a worker holds a rollout.

        Return None if settlement needs complete rows.
        """
        request = stored.request
        per_task = request.rollouts_per_task
        if (
            request.environment == "e2b"
            or request.max_failed_fraction is not None
            or len(request.task_names) > 100_000
            or per_task > 100_000
        ):
            return None
        evaluation_id = stored.evaluation_id
        first_id = rollout_item_id(evaluation_id, 0, 0)
        counts = self.queue.prefix_state_counts(
            EVALUATIONS_QUEUE,
            scenario=scenario,
            kind=EVALUATION_ROLLOUT_KIND,
            id_prefix=first_id.rsplit(":", 2)[0] + ":",
            marker=RETRY_WAIT_METADATA,
        )
        if not counts or "failed" in counts or any(marked for _, _, marked in counts.values()):
            return None
        held = sum(count for count, _, _ in counts.values())
        active = held - counts.get("finished", (0, "", False))[0]
        total = len(request.task_names) * per_task
        started = bool(counts.keys() & {"assigned", "running"})
        if held == total:
            return started if active else None
        # A retained prefix has exactly one item per position through the largest ID.
        if held > total or max(largest for _, largest, _ in counts.values()) != rollout_item_id(
            evaluation_id, *divmod(held - 1, per_task)
        ):
            return None
        free = total - held if request.max_concurrent_rollouts is None else request.max_concurrent_rollouts - active
        positions = range(held, min(total, held + free))
        if positions:
            item = self.queue.read(EVALUATIONS_QUEUE, first_id)
            if item is None:
                return None
            first = EvaluationRollout.from_item(item)
            for position in positions:
                index, repeat = divmod(position, per_task)
                self.queue.append(rollout_work(planned_rollout(first, request, index, repeat)))
        return started

    def settle(self, scenario: str, evaluation_id: str) -> None:
        """Start the evaluation once a worker took one of its rollouts or template items; stop it when too many of
        its ended rollouts failed (``max_failed_fraction``); while fewer than ``max_concurrent_rollouts`` rollouts are
        not final, queue again the rollouts that wait to run again (``rollout_waits``), then add the rollouts that wait
        for their first run; end the evaluation once all of them are in the queue and ended.

        On e2b, the rollouts of a task wait for their first run until the task's template item is final. The
        evaluation keeps template items for the first ``TEMPLATE_TASKS_AHEAD`` tasks that have rollouts to add, and
        queues again each one that a cancel or a stop ended. A template item counts against no limit: its build holds
        no rollout slot. The rollouts of later tasks whose templates are ready start before those of a task whose
        template still builds, so a slow template holds back its own task only. A rollout that requires prepared
        templates and waits to run again also waits until its task's template item is final; until then it keeps no
        place of ``max_concurrent_rollouts`` from the rollouts of other tasks.

        A rollout that a cancel or a stop ended (``rollout_cancelled``) stays ended in a running evaluation only after
        a retry of a cancelled or stopped evaluation. While one does, no rollout starts its first run, the failure
        check does not count it, and the evaluation fails with its error again once the other rollouts end. A resume
        runs those rollouts again."""
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.read(evaluation_id)
            if stored is None or stored.state in ("finished", "failed"):
                return
            started = self.settle_rollout_prefix(scenario, stored)
            if started is not None:
                if stored.state == "queued" and started:
                    current.store.evaluations.begin(evaluation_id)
                return
            # Read under the lock: a retry changes the evaluation and its rollouts under it, and rollouts read
            # before it would count as failed again.
            items = self.rollouts(evaluation_id, scenario)
            request = stored.request
            templates = self.templates(evaluation_id, scenario) if request.environment == "e2b" else {}
            cancelled = [item for item in items if rollout_cancelled(item)]
            ended = [item for item in items if rollout_ended(item) and not rollout_cancelled(item)]
            failed = sum(rollout_failed(item) for item in ended)
            total = len(request.task_names) * request.rollouts_per_task
            limit = request.max_failed_fraction
            if limit is not None and len(ended) >= min(FAILURE_CHECK_ROLLOUTS, total) and failed > limit * len(ended):
                error = (
                    f"too many rollouts failed: {failed} of {len(ended)} ended ({failed / len(ended):.0%}), "
                    f"over the limit of {limit:.0%}"
                )
                logger.warning("evaluation %s of %s stopped: %s", evaluation_id, scenario, error)
                # The rollouts end first, so that a reader that sees the failed evaluation counts them. Under this
                # lock, no settle adds a rollout to the evaluation meanwhile.
                for item in items:
                    self.queue.cancel(EVALUATIONS_QUEUE, item.id, FAILURES_ERROR)
                    if rollout_waits(item):
                        self.queue.write_metadata(EVALUATIONS_QUEUE, item.id, RETRY_WAIT_METADATA, False)
                for item in templates.values():
                    self.queue.cancel(EVALUATIONS_QUEUE, item.id, FAILURES_ERROR)
                current.store.evaluations.fail(evaluation_id, error=error, finished_at=time.time())
                return
            # Each rollout has a fixed ID, so two settles at once add it once. The rollouts to add are those that the
            # queue does not hold, in task and repeat order.
            per_task = request.rollouts_per_task
            held = {(cast(int, item.payload["task_index"]), cast(int, item.payload["repeat"])) for item in items}
            missing = [position for position in range(total) if divmod(position, per_task) not in held]
            waiting = len(missing)
            retry_waiting = [item for item in items if rollout_waits(item)]
            if waiting > 0 or retry_waiting:
                # An evaluation on e2b can have template items and no rollout yet; the task's first rollout is the
                # payload of its template item.
                first = EvaluationRollout.from_item(items[0] if items else next(iter(templates.values())))
                # A rollout that requires prepared templates also waits for its task's template item, which a retry
                # or a resume queued again. A task whose item the queue no longer holds lets its rollouts go: the
                # worker then fails them, with that reason.
                building: set[int] = (
                    {index for index, template in templates.items() if not template.final}
                    if first.requires_prepared_templates
                    else set()
                )
                ready = [item for item in retry_waiting if cast(int, item.payload["task_index"]) not in building]
                # A retry or a resume can remove the limit while rollouts wait: then each one goes to the queue.
                limit = request.max_concurrent_rollouts
                free = waiting + len(retry_waiting) if limit is None else limit - sum(not item.final for item in items)
                retried = ready[: max(free, 0)]
                for item in retried:
                    self.queue.retry(EVALUATIONS_QUEUE, item.id)
                    self.queue.write_metadata(EVALUATIONS_QUEUE, item.id, RETRY_WAIT_METADATA, False)
                # A ready rollout that the limit holds back leaves no place, so the next settle queues it first. A
                # rollout whose template builds again keeps no place, so the ready tasks' rollouts can start.
                free -= len(retried)
                if request.environment == "e2b" and not cancelled:
                    # A template that builds slowly holds one of these places only: the window moves on as the
                    # rollouts of the other tasks go to the queue.
                    tasks = list(dict.fromkeys(position // per_task for position in missing))
                    for index in tasks[:TEMPLATE_TASKS_AHEAD]:
                        template = templates.get(index)
                        if template is None:
                            task = planned_rollout(first, request, index, 0)
                            templates[index] = self.queue.append(template_work(task))
                        elif rollout_cancelled(template):
                            templates[index] = self.queue.retry(EVALUATIONS_QUEUE, template.id) or template
                # Only a resume starts the rollouts that a cancel or a stop held back.
                added = 0
                for position in [] if cancelled else missing:
                    if added >= free:
                        break
                    index, repeat = divmod(position, per_task)
                    if request.environment == "e2b" and not (index in templates and templates[index].final):
                        continue
                    added += 1
                    self.queue.append(rollout_work(planned_rollout(first, request, index, repeat)))
            if (waiting > 0 and not cancelled) or retry_waiting or not all(item.final for item in items):
                started = any(item.state in ("assigned", "running") for item in (*items, *templates.values()))
                if stored.state == "queued" and started:
                    current.store.evaluations.begin(evaluation_id)
                return
        if cancelled:
            # The error says again what ended the evaluation: a cancel, or the failure check with its usual prefix.
            if any(item.error == CANCELLED_ERROR for item in cancelled):
                error = CANCELLED_ERROR
            else:
                error = f"too many rollouts failed: the stop ended {len(cancelled)} rollouts, which wait for a resume"
            self.end(scenario, evaluation_id, request, result=None, error=error)
            return
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
            rollouts[0].policy_harness(), rollouts[0].model, RolloutScores.by_task(names, scores)
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
                current_state = evaluations.read(evaluation_id)
                if current_state is not None and current_state.state == "queued":
                    evaluations.begin(evaluation_id)
                if result is not None:
                    finished = evaluations.finish(evaluation_id, result=result, finished_at=time.time())
                    # A score with a harness from an item says nothing of the harness that the deployment serves.
                    if harness_item(request.harness) is None:
                        rule = current.activation_rule
                        # Plugin subclasses keep the same evaluation and its rollout mean.
                        if type(rule) is EvaluatedActivation:
                            activates = rule.meets_threshold(finished.mean_score)
                        else:
                            activates = rule.activates_evaluated(request.artifact_id, result)
                else:
                    evaluations.fail(evaluation_id, error=error or "the evaluation failed", finished_at=time.time())
        except InternalError:
            raise
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
            except CommitPendingError:
                time.sleep(ACTIVATION_RETRY_SECONDS * (attempt + 1))
            except InternalError:
                raise
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

        Complete each pending keyed start first. An evaluation without initial work after this step lost its queue.
        Load each known scenario once.
        """
        for row in self.dispatcher.list_scenarios():
            scenario = row["scenario"]
            try:
                with self.dispatcher.locked_scenario(scenario) as current:
                    evaluations = current.store.evaluations
                    for evaluation_id, first_work_json in evaluations.pending_starts():
                        started = evaluations.read(evaluation_id)
                        if started is not None and started.state in ("queued", "running"):
                            self.publish_start(evaluations, started, first_work_json)
                        # A cancelled start can still need its first work item when the client resumes it.
                        elif started is None or started.error != CANCELLED_ERROR:
                            evaluations.complete_start(evaluation_id)
                    for stored in evaluations.newest(limit=MAX_STOPPED_EVALUATIONS):
                        if (
                            stored.state in ("queued", "running")
                            and not self.rollouts(stored.evaluation_id, scenario)
                            and not self.templates(stored.evaluation_id, scenario)
                        ):
                            evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
            except Exception:
                logger.exception("could not fail the stopped evaluations of %s", scenario)

    def start_work_expired(self, stored: StoredEvaluation) -> bool:
        """Return whether the queue could already have removed the initial work for this start."""
        return time.time() - stored.created_at >= self.dispatcher.work_retention_seconds

    def publish_start(
        self, evaluations: EvaluationStore, stored: StoredEvaluation, first_work_json: str
    ) -> StoredEvaluation:
        """Publish the pending work under the scenario lock, or fail if the queue could have removed it."""
        work = NewWork(**json.loads(first_work_json))
        if self.queue.read(work.queue, work.id) is None:
            if self.start_work_expired(stored):
                stored = evaluations.fail(stored.evaluation_id, error=STOPPED_ERROR, finished_at=time.time())
            else:
                self.queue.append(work)
        evaluations.complete_start(stored.evaluation_id)
        return stored


class ArtifactEvaluations:
    """Accepts, queues and reads the artifact evaluations of a service's scenarios.

    Values: ``dispatcher`` holds the scenarios. ``queue`` is the dispatcher's work queue, which carries the rollouts
    to the evaluation service's workers. ``sources`` reads the harness tree of an artifact item. ``new_artifacts``
    reports the new artifacts that Reef evaluates with each scenario's settings, or is None for no automatic evaluation.
    ``requires_prepared_templates`` is the deployment's ``evaluation.require_prepared_templates``: each new evaluation
    on e2b runs a trial only from its task's prepared templates (``EvaluationRollout.requires_prepared_templates``).
    If ``pin_task_files`` is true, Reef stores the task files digests when it accepts an evaluation.
    ``results`` ends each evaluation as its rollouts end. ``ended_progress`` keeps the progress of ended evaluations
    for ``status_json``.

    Purpose: the evaluation routes call ``start``, ``cancel``, ``retry``, ``resume``, ``get``, ``newest``,
    ``rollouts_json``, ``calls_json``, ``call_json``, ``trajectories_json``, ``exported_rollouts`` and
    ``rollout_trajectories``, and the scenario update route calls ``configure``. ``open`` starts ``results`` and the
    reader of ``new_artifacts``; ``close`` stops them.
    Every method but ``start``, ``cancel``, ``retry``, ``resume``, ``run_again``, ``configure``, ``open`` and
    ``close`` only reads.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        new_artifacts: NewArtifacts | None = None,
        sources: ArtifactSources | None = None,
        requires_prepared_templates: bool = False,
        pin_task_files: bool = False,
    ) -> None:
        if not isinstance(requires_prepared_templates, bool):
            raise TypeError("requires_prepared_templates is a bool")
        if not isinstance(pin_task_files, bool):
            raise TypeError("pin_task_files is a bool")
        self.dispatcher = dispatcher
        self.sources = ArtifactSources.from_setting(()) if sources is None else sources
        self.new_artifacts = new_artifacts
        self.requires_prepared_templates = requires_prepared_templates
        self.pin_task_files = pin_task_files
        self.queue = dispatcher.queue
        self.results = EvaluationResults(dispatcher)
        self.ended_progress = EndedProgress()
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
        """Queue the evaluation of ``new`` with its scenario's settings.

        Return None when the scenario has no settings, or when the artifact catalog lists ``new`` at an earlier
        snapshot: the commit carried that component, and it was evaluated when it was made.
        """
        stored = self.dispatcher.stored_settings(new.scenario).artifact_evaluation
        if stored is None:
            return None
        with self.dispatcher.locked_scenario(new.scenario) as current:
            made = any(
                row["artifact_id"] == new.artifact_id and row["snapshot"].revision == new.revision
                for row in current.artifact_rows()
            )
        if not made:
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

        Raise ``ValueError`` or ``ReefError`` for a request that the scenario cannot run, ``ScenarioNotFoundError``
        for an unknown scenario, and ``ArtifactNotFoundError`` for an unknown artifact.
        """
        if not isinstance(body, Mapping) or "start_id" not in body:
            artifact_id, settings = request_settings(body)
            return self.accept(scenario, artifact_id, settings, state="running")

        start_id = check_evaluation_start_id(body["start_id"])
        payload = {key: value for key, value in body.items() if key != "start_id"}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        with self.dispatcher.locked_scenario(scenario) as current:
            evaluations = current.store.evaluations
            previous = evaluations.read_started(start_id, fingerprint)
            if previous is None:
                artifact_id, settings = request_settings(payload)
                try:
                    stored = self.accept(
                        scenario, artifact_id, settings, state="running", start=(start_id, fingerprint)
                    )
                except EvaluationStartConflictError:
                    previous = evaluations.read_started(start_id, fingerprint)
                    if previous is None:
                        raise
            if previous is not None:
                stored, first_work_json = previous
                if first_work_json is None or stored.state not in ("queued", "running"):
                    return stored
                stored = self.results.publish_start(evaluations, stored, first_work_json)
        self.results.settle(scenario, stored.evaluation_id)
        return stored

    def cancel(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """Cancel the evaluation ``evaluation_id`` of ``scenario`` and return it after the cancel; None when the
        scenario has no such evaluation.

        A queued or running evaluation fails with ``CANCELLED_ERROR``, and each of its rollouts that is not final
        fails with the same error. A worker that runs one of them stops its trial at its next lease extension. No
        rollout that waits for ``max_concurrent_rollouts`` starts. An evaluation that ended stays as it is.
        """
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.read(evaluation_id)
            if stored is None or stored.state in ("finished", "failed"):
                return stored
            # ``settle`` adds rollouts under this lock and adds none to a failed evaluation, so the listed rollouts
            # are all that it will ever have.
            cancelled = current.store.evaluations.fail(evaluation_id, error=CANCELLED_ERROR, finished_at=time.time())
            for item in self.results.rollouts(evaluation_id, scenario):
                self.queue.cancel(EVALUATIONS_QUEUE, item.id, CANCELLED_ERROR)
                if rollout_waits(item):
                    # The rollout keeps the error of its last attempt.
                    self.queue.write_metadata(EVALUATIONS_QUEUE, item.id, RETRY_WAIT_METADATA, False)
            for item in self.results.templates(evaluation_id, scenario).values():
                self.queue.cancel(EVALUATIONS_QUEUE, item.id, CANCELLED_ERROR)
        return cancelled

    def retry(
        self, scenario: str, evaluation_id: str, changes: Mapping[str, object] | None = None
    ) -> StoredEvaluation | None:
        """Run the failed rollouts of the evaluation ``evaluation_id`` of ``scenario`` once more, and return the
        evaluation, which runs (RFC #1340); None when the scenario has no such evaluation.

        A rollout failed when its own run ended with no score: its work item failed, or it finished with a failed
        score (``rollout_failed``). A rollout that a cancel or a stop of the evaluation ended did not fail
        (``rollout_cancelled``): a retry leaves it, and the rollouts that never started, to ``resume``. After a retry
        of a cancelled or stopped evaluation, the evaluation fails with that error again once the failed rollouts end.

        ``EvaluationStateConflictError`` for an evaluation with no failed rollout. ``run_again`` gives the rest,
        and what ``changes`` can hold.
        """
        return self.run_again(scenario, evaluation_id, resumes=False, changes=changes)

    def resume(
        self, scenario: str, evaluation_id: str, changes: Mapping[str, object] | None = None
    ) -> StoredEvaluation | None:
        """Run the rollouts of the evaluation ``evaluation_id`` of ``scenario`` that a cancel or a stop ended
        (``rollout_cancelled``), and start the rollouts that never started, and return the evaluation, which runs;
        None when the scenario has no such evaluation.

        A rollout that a cancel stopped in the middle of its run starts again from the start of its task. The failed
        rollouts keep their failures: ``retry`` runs them again.

        ``EvaluationStateConflictError`` for an evaluation with no rollout that a cancel or a stop ended and no
        rollout that never started. ``run_again`` gives the rest, and what ``changes`` can hold.
        """
        return self.run_again(scenario, evaluation_id, resumes=True, changes=changes)

    def run_again(
        self,
        scenario: str,
        evaluation_id: str,
        *,
        resumes: bool,
        changes: Mapping[str, object] | None = None,
    ) -> StoredEvaluation | None:
        """Run once more the rollouts of the evaluation ``evaluation_id`` of ``scenario`` that ``resume`` (when
        ``resumes``) or ``retry`` chooses, and return the evaluation, which runs; None when the scenario has no such
        evaluation.

        Each chosen rollout goes back to the queue with its earlier attempts kept, while fewer than
        ``max_concurrent_rollouts`` rollouts are not final; the others wait (``rollout_waits``) and go back as
        rollouts end (``EvaluationResults.settle``). In an evaluation that requires prepared templates, a chosen
        rollout whose task's template item did not finish also waits: the template item goes to the queue again, and
        the rollout goes back once that item is final. The other rollouts stay as they are, and a scored rollout keeps
        its score. An evaluation that runs goes on with those rollouts in the queue again. An evaluation that ended
        runs again, without its result or its error. When the rollouts are final, ``EvaluationResults`` ends the
        evaluation as it ends a first run, and gives a finished one to the activation rule.

        ``changes`` is None or empty to keep the evaluation's request, or holds only ``max_concurrent_rollouts``: a
        positive int, or None for no limit. The evaluation then keeps that limit from now on, for these rollouts and
        for each rollout that waits. A lower limit stops no rollout that runs: fewer rollouts start until the
        rollouts that are not final are fewer than the limit.

        ``ValueError`` for any other ``changes``. ``EvaluationStateConflictError`` for an evaluation with no rollout
        to run again, and for an ended one whose
        rollouts the queue no longer keeps (``WorkQueue.delete_final``).
        """
        changes = {} if changes is None else changes
        if set(changes) - {"max_concurrent_rollouts"}:
            raise ValueError("a retry or a resume of an evaluation can change only max_concurrent_rollouts")
        with self.dispatcher.locked_scenario(scenario) as current:
            stored = current.store.evaluations.read(evaluation_id)
            if stored is None:
                return None
            request = stored.request
            if "max_concurrent_rollouts" in changes:
                # ``replace`` checks the new limit before any change, so that a refused one changes nothing.
                limit = cast(int | None, changes["max_concurrent_rollouts"])
                request = replace(request, max_concurrent_rollouts=limit)
            total = len(request.task_names) * request.rollouts_per_task
            items = self.results.rollouts(evaluation_id, scenario)
            # An evaluation on e2b can end before it added a rollout: its template items are then all it has.
            templates = self.results.templates(evaluation_id, scenario) if request.environment == "e2b" else {}
            ended = stored.state in ("finished", "failed")
            first_work_json = (
                current.store.evaluations.pending_start(evaluation_id)
                if resumes
                and not items
                and not templates
                and (not ended or stored.error == CANCELLED_ERROR)
                and not self.results.start_work_expired(stored)
                else None
            )
            # Off e2b, the queue holds a prefix of the rollouts in task and repeat order (``settle``). On e2b it holds
            # template items for a range of tasks with no gap, and rollouts only of those tasks and of the tasks before
            # the range. Each task before the range has all its rollouts. The range starts at the first task, or at a
            # later one: an evaluation that started before Reef had template items ran its first tasks without them,
            # and the retention deletes a task's template item before its rollouts. Anything else means that the
            # retention deleted some rollouts.
            per_task = request.rollouts_per_task
            start = min(templates, default=len(request.task_names))
            before = [item.id for item in items if cast(int, item.payload["task_index"]) < start]
            kept = before == [
                rollout_item_id(evaluation_id, *divmod(position, per_task)) for position in range(len(before))
            ]
            if templates:
                tasks = {cast(int, item.payload["task_index"]) for item in items}
                kept = (
                    kept
                    and len(before) == start * per_task
                    and sorted(templates) == list(range(start, start + len(templates)))
                    and tasks <= set(range(start)) | set(templates)
                )
            if ended and (
                (not items and not templates and first_work_json is None)
                or not kept
                or (stored.state == "finished" and len(items) < total)
            ):
                raise EvaluationStateConflictError(f"evaluation {evaluation_id!r}: Reef no longer keeps its rollouts")
            if resumes:
                again = [item for item in items if rollout_cancelled(item)]
                # The rollouts that never started have no item yet: the settle below adds them.
                if not again and not (ended and len(items) < total) and first_work_json is None:
                    raise EvaluationStateConflictError(
                        f"evaluation {evaluation_id!r} has no rollout that a cancel or a stop ended, "
                        "and no rollout that never started"
                    )
            else:
                again = [
                    item
                    for item in items
                    if rollout_ended(item) and rollout_failed(item) and not rollout_cancelled(item)
                ]
                if not again:
                    raise EvaluationStateConflictError(f"evaluation {evaluation_id!r} has no failed rollout")
            if request.max_concurrent_rollouts != stored.request.max_concurrent_rollouts:
                # The settles that follow read the new limit from the store.
                stored = current.store.evaluations.limit_rollouts(evaluation_id, request.max_concurrent_rollouts)
                logger.info(
                    "evaluation %s of %s runs at most %s rollouts at once",
                    evaluation_id,
                    scenario,
                    request.max_concurrent_rollouts,
                )
            # An ended evaluation runs before its rollouts change, so the settle that follows each change sees it
            # running.
            reopened = current.store.evaluations.reopen(evaluation_id) if ended else stored
            if first_work_json is not None:
                self.queue.append(NewWork(**json.loads(first_work_json)))
                current.store.evaluations.complete_start(evaluation_id)
            # A rollout that requires prepared templates runs again only once its task's template item finished.
            # Each template item of these tasks that did not finish goes to the queue again, or back to it when the
            # queue no longer holds it, and the task's rollouts wait until it is final (``settle``).
            unprepared: set[int] = set()
            if again and EvaluationRollout.from_item(again[0]).requires_prepared_templates:
                for item in again:
                    index = cast(int, item.payload["task_index"])
                    template = templates.get(index)
                    if index in unprepared or (template is not None and template.state == "finished"):
                        continue
                    if template is None:
                        self.queue.append(template_work(EvaluationRollout.from_item(item)))
                    else:
                        self.queue.retry(EVALUATIONS_QUEUE, template.id)
                    unprepared.add(index)
            waits = [item for item in again if cast(int, item.payload["task_index"]) in unprepared]
            runs = [item for item in again if cast(int, item.payload["task_index"]) not in unprepared]
            limit = request.max_concurrent_rollouts
            free = len(runs) if limit is None else max(limit - sum(not item.final for item in items), 0)
            # Under the scenario lock, a settle sees each of these rollouts either queued or marked to wait.
            for item in (*waits, *runs[free:]):
                self.queue.write_metadata(EVALUATIONS_QUEUE, item.id, RETRY_WAIT_METADATA, True)
            for item in runs[:free]:
                self.queue.retry(EVALUATIONS_QUEUE, item.id)
        operation = "resumes" if resumes else "retries"
        logger.info("evaluation %s of %s %s %d rollouts", evaluation_id, scenario, operation, len(again))
        # The settle starts the rollouts that never started, and ends an evaluation that has only waiting rollouts.
        self.results.settle_changed(scenario, evaluation_id)
        return reopened

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
                if harness_item(settings.harness) is not None:
                    raise ValueError("an automatic evaluation's harness is a Harbor agent, not an artifact item")
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
    ) -> tuple[ArtifactEvaluationRequest, Artifact, PolicyModel, str]:
        """The request of ``settings`` for ``artifact_id`` (None: the component's active artifact), the materialized
        artifact, the model that the runtime serves for it, and the runtime's API dialect; ``ValueError`` or
        ``ReefError`` when ``current`` cannot run it."""
        kinds = {name: component.kind for name, component in current.components.components.items()}
        component = settings.component
        if component is None:
            named = [name for name, kind in kinds.items() if kind is not None]
            if len(named) != 1:
                raise ValueError(f"name the component to evaluate: the artifacts hold {sorted(named)}")
            component = named[0]
        if component not in kinds:
            raise ValueError(f"the artifacts of scenario {current.name!r} hold no component {component!r}")
        if artifact_id is None:
            artifact_id = current.artifact_selection().components[component]
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
        # The evaluated component alone is made local: the snapshot's other components stay in the store.
        selected, snapshot = current.resolve_artifact(artifact_id, component)
        artifact = current.components.component_artifact(snapshot, selected.component).materialize()
        return request, artifact, runtime.artifact_model(artifact), runtime.api

    def held_collection(self, scenario: str, name: str, held: frozenset[str]) -> EvaluationCollection:
        """The collection ``name`` of the owner of ``scenario``, with the tasks of each benchmark that are in
        ``held``: the item IDs of the tasks of the scenario's eval split. A benchmark with no such task is left out.

        Raise ``ValueError`` when the owner stores no such collection, or when no task of it is in ``held``.
        """
        owner = self.dispatcher.scenario_owner(scenario)
        if owner is None:
            raise ValueError(f"scenario {scenario!r} has no owner, so it can use no collection")
        try:
            stored, tasks = collection_tasks(DataAccess.of(self.dispatcher).require_store(), owner, name)
        except (DataStoreUnavailableError, CollectionNotFoundError) as exc:
            raise ValueError(str(exc)) from exc
        benchmarks = tuple(
            (benchmark, in_split)
            for benchmark, item_ids in tasks.items()
            if (in_split := tuple(item_id for item_id in item_ids if item_id in held))
        )
        if not benchmarks:
            raise ValueError(f"the eval split of the imported tasks holds no task of the collection {name!r}")
        return EvaluationCollection(name, stored.average, benchmarks)

    def tree_harness(
        self,
        scenario: str,
        item_id: str,
        api: str,
        env: Mapping[str, str],
        config: Mapping[str, Mapping[str, Any]],
    ) -> TreeHarness:
        """The harness tree of the artifact item ``item_id`` of the owner of ``scenario``, for model calls of the
        dialect ``api``, whose agent's commands get the variables ``env``. Each object of ``config`` merges into the
        tree's ``config/<target>.json`` of its config target, as a tree's own config would.

        Raise ``ValueError`` when the item is not a tree that a task's container can run, when the tree with
        ``config`` fails the adapter's validation, such as for a key that Reef's model binding writes, or when
        ``env`` sets a token limit that is not a positive integer. The tasks are not read
        here: each rollout checks its own task against the tree's agent (``EvaluationRolloutWorker.score``).
        """
        owner = self.dispatcher.scenario_owner(scenario)
        if owner is None:
            raise ValueError(f"scenario {scenario!r} has no owner, so it can use no artifact item")
        try:
            item = DataAccess.of(self.dispatcher).require_store().read(owner, item_id)
        except DataStoreUnavailableError as exc:
            raise ValueError(str(exc)) from exc
        if item is None:
            raise ValueError(f"{owner!r} stores no item {item_id!r}")
        if item.kind != "artifact" or item.content is None or item.content["kind"] != "harness_plugin":
            raise ValueError(f"item {item_id!r} is not an artifact of the kind harness_plugin")
        uri, adapter = item.content["source_uri"], item.metadata.get("adapter")
        if uri is None:
            raise ValueError(
                f"artifact item {item_id!r} has no source_uri: only a published artifact can be evaluated"
            )
        if not isinstance(adapter, str):
            raise ValueError(f"artifact item {item_id!r} names no adapter in its metadata")
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise ValueError(str(exc)) from exc
        if not can_run_in_task_container(descriptor):
            raise ValueError(f"adapter {adapter!r} cannot run in a task's container")
        if api not in descriptor.model_binding:
            raise ValueError(f"adapter {adapter!r} cannot call this runtime's API dialect {api!r}")
        files = self.sources.open(uri)
        tree = HarnessTree({path: files.read_bytes(path).decode("utf-8") for path in files.paths()})
        validation = validate_tree(tree, descriptor)
        if not validation.passed:
            raise ValueError(f"the harness tree of item {item_id!r} fails validation: {validation.message()}")
        if config:
            configured = dict(tree.files)
            for target, data in config.items():
                path = descriptor.layout.parts["config"].path.replace(PLACEHOLDER, target)
                configured[path] = (
                    json.dumps(deep_merge(json.loads(configured.get(path, "{}")), data), indent=2) + "\n"
                )
            tree = HarnessTree(configured)
            validation = validate_tree(tree, descriptor)
            if not validation.passed:
                raise ValueError(f"the harness config fails validation: {validation.message()}")
        harness = TreeHarness(adapter, api, tree.files, env)
        # Rendering now refuses a bad tree here, and not at the end of the evaluation.
        harness.policy_harness()
        return harness

    def accept(
        self,
        scenario: str,
        artifact_id: str,
        settings: ArtifactEvaluationSettings,
        *,
        state: Literal["queued", "running"],
        start: tuple[str, str] | None = None,
    ) -> StoredEvaluation:
        """Check and store the evaluation of ``artifact_id`` with ``settings``, and add its rollouts to the queue.

        A client's evaluation is stored ``running``; an automatic one ``queued``, until a worker takes one of its
        rollouts. On e2b, the template worker gets the owner that the scenario's record stores.
        """
        with self.dispatcher.locked_scenario(scenario) as current:
            request, _artifact, model, api = self.checked(current, artifact_id, settings)
            owner = self.dispatcher.task_owner(scenario) if request.environment == "e2b" else None
            selection = {**current.artifact_selection().to_dict(), request.component: request.artifact_id}
            samples = self.dispatcher.storage.sample_store(scenario, current.store)
            # The listing reads no task files: the eval split can hold many more tasks than the evaluation runs.
            # A scenario can use a task more than once (RFC #1420): an evaluation runs it once.
            held: dict[str, str] = {}
            for listed in () if samples is None else samples.listed_imported_tasks():
                if listed.split == "eval":
                    held.setdefault(listed.sample_id, listed.digest)
        held_names = frozenset(held)
        unknown = sorted(set(request.task_names) - held_names)
        if unknown:
            raise ValueError(f"the eval split of the imported tasks holds no tasks {unknown}")
        collection = (
            None if settings.collection is None else self.held_collection(scenario, settings.collection, held_names)
        )
        if collection is not None:
            collection_names = collection.task_names()
            names = tuple(name for name in held if name in collection_names)
        else:
            names = request.task_names or tuple(held)
        if not names:
            raise ValueError("the eval split of the imported tasks holds no task to evaluate on")
        sampled = settings.sampled_task_names(names, collection)
        if collection is not None and len(sampled) < len(names):
            # A stored benchmark contains only the tasks that run.
            sampled_names = frozenset(sampled)
            collection = replace(
                collection,
                benchmarks=tuple(
                    (benchmark, kept)
                    for benchmark, task_names in collection.benchmarks
                    if (kept := tuple(name for name in task_names if name in sampled_names))
                ),
            )
        names = sampled
        evaluation_id = uuid.uuid4().hex
        item = harness_item(request.harness)
        tree = (
            None
            if item is None
            else self.tree_harness(
                scenario, item, api, harness_item_env(request.harness), harness_item_config(request.harness)
            )
        )
        agent = None if tree is not None else harbor_agent(request.harness)
        # Deferred rollouts use the task names and, when pinned, the digests recorded here.
        request = replace(
            request,
            task_names=names,
            collection=collection,
            task_digests=tuple(held[name] for name in names) if self.pin_task_files else None,
        )
        planned_rollouts = (
            EvaluationRollout(
                scenario,
                evaluation_id,
                request.artifact_id,
                request.component,
                index,
                name,
                repeat,
                model,
                agent,
                request.environment,
                cast(dict[str, JsonValue], selection),
                tree,
                request.log_probs,
                owner=owner,
                requires_prepared_templates=self.requires_prepared_templates and request.environment == "e2b",
                task_digest=request.task_digest(index),
            )
            for index, name in enumerate(names)
            for repeat in range(request.rollouts_per_task)
        )
        if start is not None:
            # Store the first work item before workers can take it. A replay or restart can publish it again.
            first = next(planned_rollouts)
            first_work = template_work(first) if request.environment == "e2b" else rollout_work(first)
            with self.dispatcher.locked_scenario(scenario) as current:
                evaluations = current.store.evaluations
                stored = evaluations.create(
                    evaluation_id=evaluation_id,
                    owner=QUEUE_OWNER,
                    request=request,
                    created_at=time.time(),
                    state=state,
                    start=EvaluationStart(*start, json.dumps(asdict(first_work), ensure_ascii=True)),
                )
                self.queue.append(first_work)
                evaluations.complete_start(evaluation_id)
            return stored
        # The rollouts go to the queue before the evaluation exists, so that no reader sees an evaluation whose
        # rollouts are missing. An evaluation runs on all of its tasks or on none: when a step fails, the added
        # rollouts are cancelled and no evaluation is stored.
        # On e2b, the first tasks' template items go first; ``settle`` adds each task's rollouts once its template
        # item is final.
        rollouts = list(planned_rollouts)
        initial_work = (
            [template_work(rollout) for rollout in rollouts[:: request.rollouts_per_task][:TEMPLATE_TASKS_AHEAD]]
            if request.environment == "e2b"
            else [rollout_work(rollout) for rollout in rollouts[: request.max_concurrent_rollouts]]
        )
        added: list[str] = []
        try:
            for work in initial_work:
                self.queue.append(work)
                added.append(work.id)
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

    def rollout_page_json(
        self,
        scenario: str,
        evaluation_id: str,
        *,
        offset: int,
        limit: int,
        calls: bool,
        state: str = "all",
        benchmark: str | None = None,
        lowest_first: bool = False,
    ) -> dict[str, object] | None:
        """The requested rollout page. The total counts rollouts in the current filter."""
        stored = self.get(scenario, evaluation_id)
        if stored is None:
            return None
        request = stored.request
        per_task = request.rollouts_per_task
        task_indexes: Sequence[int] = range(len(request.task_names))
        if benchmark is not None:
            benchmarks = request.collection.benchmarks if request.collection is not None else ()
            names = next((set(names) for key, names in benchmarks if key == benchmark), set())
            task_indexes = [index for index, name in enumerate(request.task_names) if name in names]
        items: tuple[WorkItem, ...] | None = None
        means: dict[int, float | None] = {}
        needs_all_items = state != "all" or lowest_first
        include_task_means = needs_all_items or benchmark is not None
        if needs_all_items:
            items = self.results.rollouts(evaluation_id, scenario)
        elif include_task_means:
            page_tasks = task_indexes[offset // per_task : (offset + limit - 1) // per_task + 1]
            item_ids = [
                rollout_item_id(evaluation_id, index, repeat) for index in page_tasks for repeat in range(per_task)
            ]
            items = self.queue.read_items(EVALUATIONS_QUEUE, item_ids) if item_ids else ()
        if include_task_means:
            scores: dict[int, list[RolloutScore]] = {}
            for item in items or ():
                if item.scenario == scenario and item.state == "finished" and not rollout_waits(item):
                    score = RolloutScore.from_json(cast(Mapping[str, object], item.result)["score"])
                    scores.setdefault(cast(int, item.payload["task_index"]), []).append(score)
            means = {index: mean_score(values) for index, values in scores.items()}
        if lowest_first:
            task_indexes = sorted(task_indexes, key=lambda index: (means.get(index) is None, means.get(index) or 0))
        if state == "all":
            total = len(task_indexes) * per_task
            positions = [
                (task_indexes[position // per_task], position % per_task)
                for position in range(offset, min(offset + limit, total))
            ]
            if items is None:
                item_ids = [rollout_item_id(evaluation_id, index, repeat) for index, repeat in positions]
                items = self.queue.read_items(EVALUATIONS_QUEUE, item_ids) if item_ids else ()
        else:
            selected_tasks = set(task_indexes)
            matching: dict[int, list[int]] = {}
            for item in items or ():
                index = cast(int, item.payload["task_index"])
                if index not in selected_tasks or rollout_waits(item):
                    continue
                if (state == "running" and item.state in ("assigned", "running")) or (
                    state == "failed" and item.final and rollout_failed(item)
                ):
                    matching.setdefault(index, []).append(cast(int, item.payload["repeat"]))
            matching_positions = [(index, repeat) for index in task_indexes for repeat in matching.get(index, ())]
            total = len(matching_positions)
            positions = matching_positions[offset : offset + limit]
        page_ids = {rollout_item_id(evaluation_id, index, repeat) for index, repeat in positions}
        views = self.rollout_views(
            scenario,
            stored,
            [item for item in items or () if item.id in page_ids and item.scenario == scenario],
            positions=positions,
            calls=calls,
        )
        page: dict[str, object] = {"rollouts": views, "total": total, "offset": offset}
        if include_task_means:
            page["task_means"] = {index: means.get(index) for index, _ in positions}
        return page

    def rollouts_json(self, scenario: str, evaluation_id: str, *, calls: bool = True) -> list[dict[str, Any]] | None:
        """The rollouts of one evaluation, in task and repeat order, each as ``{task, task_index, repeat, state,
        attempt_count, started_at, ended_at, score, rewards, error, calls, verifier_calls}``; None when ``scenario`` has no
        evaluation ``evaluation_id``.

        ``state`` is the rollout's queue state, or ``waiting`` for a rollout that ``max_concurrent_rollouts`` keeps
        out of the queue yet, also one that a retry or a resume runs again (``rollout_waits``). ``started_at`` and
        ``ended_at`` are its work item's times; a rollout that waits to run again has no ``ended_at``, ``score`` or
        ``error``.
        ``score`` is the rollout's ``RolloutScore`` as JSON once it finished, else null; ``rewards`` is every reward
        of its verifier (``verifier_rewards``, RFC #1297), and null when it has none, as for a rollout that finished
        before Reef kept them; ``error`` is the reason of a failed rollout, else null. ``calls`` is the ``EvaluationCallTotals`` of the rollout's kept agent calls as JSON,
        or null when the data store keeps no agent call of it; ``verifier_calls`` is the same for its verifier calls.
        It reads the metadata of the evaluation's calls, not their content. With ``calls`` false, the views have no
        ``calls`` and no ``verifier_calls``, and it reads the queue only, so the read does not grow with the calls.
        """
        stored = self.get(scenario, evaluation_id)
        if stored is None:
            return None
        return self.rollout_views(scenario, stored, self.results.rollouts(evaluation_id, scenario), calls=calls)

    def rollout_views(
        self,
        scenario: str,
        stored: StoredEvaluation,
        items: Sequence[WorkItem],
        *,
        calls: bool,
        positions: Sequence[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Render all rollouts, or the specified positions in their requested order."""
        views: dict[tuple[int, int], dict[str, Any]] = {}
        for item in items:
            rollout = EvaluationRollout.from_item(item)
            waits = rollout_waits(item)
            finished = item.state == "finished" and item.result is not None and not waits
            views[rollout.task_index, rollout.repeat] = {
                "task": rollout.task_name,
                "task_index": rollout.task_index,
                "repeat": rollout.repeat,
                "state": "waiting" if waits else item.state,
                "attempt_count": item.attempt_count,
                "started_at": item.started_at,
                "ended_at": None if waits else item.ended_at,
                "score": cast(Mapping[str, object], item.result)["score"] if finished else None,
                "rewards": cast(Mapping[str, object], item.result).get("rewards") if finished else None,
                # A rollout that finished with a failed score keeps its reason in its result. A rollout that
                # waits for a retry shows no result of its last attempt.
                "error": (
                    None
                    if waits
                    else (cast(Mapping[str, object], item.result).get("error") if finished else item.error)
                ),
            }
        request = stored.request
        ordered: Iterator[tuple[int, int]]
        if positions is None:
            ordered = (
                divmod(position, request.rollouts_per_task)
                for position in range(len(request.task_names) * request.rollouts_per_task)
            )
        else:
            ordered = iter(positions)
        listed = [
            views.get((index, repeat))
            or {
                "task": request.task_names[index],
                "task_index": index,
                "repeat": repeat,
                "state": "waiting",
                "attempt_count": 0,
                "started_at": None,
                "ended_at": None,
                "score": None,
                "rewards": None,
                "error": None,
            }
            for index, repeat in ordered
        ]
        if not calls:
            return listed
        totals: dict[tuple[int, int, Caller], EvaluationCallTotals] = {}
        evaluation_id = stored.evaluation_id
        place = self.call_store(scenario)
        kept: Iterator[StoredItem]
        if place is None:
            kept = iter(())
        elif positions is None:
            kept = evaluation_call_items(*place, scenario, evaluation_id, content=False)
        else:
            kept = (
                item
                for position in positions
                for item in evaluation_call_items(*place, scenario, evaluation_id, content=False, rollout=position)
            )
        for call_item in kept:
            summary = evaluation_call_summary(call_item)
            key = (summary.task_index, summary.repeat, summary.caller)
            found = totals.get(key)
            totals[key] = EvaluationCallTotals(
                (0 if found is None else found.count) + 1,
                (0 if found is None else found.failed) + (summary.status != 200),
                (0 if found is None else found.prompt_tokens) + (summary.prompt_tokens or 0),
                (0 if found is None else found.completion_tokens) + (summary.completion_tokens or 0),
            )
        fields: tuple[tuple[Caller, str], ...] = (("agent", "calls"), ("verifier", "verifier_calls"))
        for view in listed:
            for caller, field in fields:
                found = totals.get((view["task_index"], view["repeat"], caller))
                view[field] = None if found is None else found.to_json()
        return listed

    def call_store(self, scenario: str) -> tuple[DataStore, str] | None:
        """The data store that keeps the evaluation calls of ``scenario``, and the scenario's owner; None for a
        deployment without a data store, or for an unknown scenario."""
        owner = self.dispatcher.scenario_owner(scenario)
        data = self.dispatcher.data_store
        return None if data is None or owner is None else (data, owner)

    def rollout_items(
        self, scenario: str, evaluation_id: str, task_index: int, repeat: int, caller: Caller
    ) -> list[StoredItem]:
        """The items of the kept calls of ``caller`` of rollout ``(task_index, repeat)`` of one evaluation, without
        their content, in attempt and then index order. It reads the calls of this rollout only."""
        place = self.call_store(scenario)
        kept = (
            ()
            if place is None
            else evaluation_call_items(*place, scenario, evaluation_id, content=False, rollout=(task_index, repeat))
        )
        found = [item for item in kept if evaluation_call_summary(item).caller == caller]
        return sorted(
            found,
            key=lambda item: (
                item.metadata["origin"]["evaluation"]["attempt"],
                item.metadata["origin"]["evaluation"]["index"],
            ),
        )

    def rollout_calls(
        self,
        scenario: str,
        evaluation_id: str,
        task_index: int,
        repeat: int,
        caller: Caller = "agent",
        *,
        attempt: int | None = None,
    ) -> Sequence[EvaluationCall]:
        """The kept calls of ``caller`` of rollout ``(task_index, repeat)`` of one evaluation, of ``attempt`` only when
        it is given, in attempt and then index order. The content of a call is read when the sequence gives it
        (``StoredCalls``)."""
        place = self.call_store(scenario)
        items = self.rollout_items(scenario, evaluation_id, task_index, repeat, caller)
        if attempt is not None:
            items = [item for item in items if evaluation_call_summary(item).attempt == attempt]
        if place is None or not items:
            return ()
        data, owner = place
        return StoredCalls(data, owner, tuple(items))

    def calls_json(
        self, scenario: str, evaluation_id: str, task_index: int, repeat: int, caller: Caller = "agent"
    ) -> list[dict[str, JsonValue]] | None:
        """The summaries of the kept calls of ``caller`` of rollout ``(task_index, repeat)`` of one evaluation as JSON,
        in attempt and then index order; None when ``scenario`` has no evaluation ``evaluation_id``. It reads no
        content."""
        if self.get(scenario, evaluation_id) is None:
            return None
        items = self.rollout_items(scenario, evaluation_id, task_index, repeat, caller)
        return [evaluation_call_summary(item).to_json() for item in items]

    def call_json(
        self,
        scenario: str,
        evaluation_id: str,
        task_index: int,
        repeat: int,
        attempt: int,
        index: int,
        caller: Caller = "agent",
    ) -> dict[str, JsonValue] | None:
        """The kept call with the key ``(task_index, repeat, attempt, caller, index)`` of one evaluation as JSON, with
        its bodies; None when ``scenario`` has no such evaluation or call. It reads the call's item by its ID."""
        place = self.call_store(scenario)
        if place is None or self.get(scenario, evaluation_id) is None:
            return None
        data, owner = place
        record_id = evaluation_call_id(evaluation_id, task_index, repeat, attempt, index, caller)
        item = data.read(owner, f"{scenario}/{record_id}")
        if item is None or item.kind != "record" or record_source(item.metadata) != "evaluation":
            return None
        return stored_evaluation_call(item).to_json()

    def trajectories_json(
        self, scenario: str, evaluation_id: str, task_index: int, repeat: int, caller: Caller = "agent"
    ) -> list[dict[str, Any]] | None:
        """The trajectories of ``caller`` of each attempt of rollout ``(task_index, repeat)`` of one evaluation, in
        attempt order; None when ``scenario`` has no evaluation ``evaluation_id``. A rollout that the queue does not
        hold has none. The agent's trajectories are stitched from its calls (``attempt_trajectories``). In each
        attempt, the verifier's trajectories are the files that it wrote (``verifier_trajectory``), in name order, and
        then the trajectories stitched from its calls."""
        if self.get(scenario, evaluation_id) is None:
            return None
        item = self.queue.read(EVALUATIONS_QUEUE, rollout_item_id(evaluation_id, task_index, repeat))
        if item is None:
            return []
        if caller == "agent":
            return self.rollout_trajectories(scenario, evaluation_id, item, every_attempt=True)
        rollout = EvaluationRollout.from_item(item)
        place = self.call_store(scenario)
        written = sorted(
            (
                found
                for found in (
                    () if place is None else evaluation_trajectory_items(*place, scenario, evaluation_id, content=True)
                )
                if (found.metadata["evaluation"]["task_index"], found.metadata["evaluation"]["repeat"])
                == (task_index, repeat)
            ),
            key=lambda found: (found.metadata["evaluation"]["attempt"], found.metadata["evaluation"]["index"]),
        )
        calls = self.rollout_calls(scenario, evaluation_id, task_index, repeat, "verifier")
        stitched = attempt_trajectories(scenario, evaluation_id, item, calls, "verifier")
        views = [verifier_trajectory(found, rollout) for found in written] + stitched
        # A stable sort keeps the files of an attempt before its stitched trajectories.
        return sorted(views, key=lambda view: view["extra"]["reef"]["evaluation"]["attempt"])

    def exported_rollouts(self, scenario: str, evaluation_id: str) -> tuple[WorkItem, ...] | None:
        """The rollout items of one evaluation, in task and repeat order, for an export of its trajectories; None when
        ``scenario`` has no evaluation ``evaluation_id``."""
        if self.get(scenario, evaluation_id) is None:
            return None
        return self.results.rollouts(evaluation_id, scenario)

    def rollout_trajectories(
        self,
        scenario: str,
        evaluation_id: str,
        item: WorkItem,
        *,
        every_attempt: bool,
        include_records: bool = True,
    ) -> list[dict[str, Any]]:
        """The trajectories of the rollout of ``item`` (``attempt_trajectories``): of each attempt when
        ``every_attempt``, else of its last attempt only, without ``extra.reef.records`` when ``include_records`` is
        false. It reads the bodies of this one rollout's calls, one call at a time."""
        rollout = EvaluationRollout.from_item(item)
        attempt = None if every_attempt else item.attempt_count
        calls = self.rollout_calls(scenario, evaluation_id, rollout.task_index, rollout.repeat, attempt=attempt)
        return attempt_trajectories(scenario, evaluation_id, item, calls, include_records=include_records)

    def get(self, scenario: str, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id`` of ``scenario``, or None."""
        with self.dispatcher.locked_scenario(scenario) as current:
            return current.store.evaluations.read(evaluation_id)

    def status_json(
        self, scenario: str, evaluations: Sequence[StoredEvaluation], *, score_levels: bool = False
    ) -> list[dict[str, object]]:
        """Each of ``evaluations`` as the status routes answer it: ``StoredEvaluation.to_json`` and its
        ``progress`` (``evaluation_progress``), null when the queue holds none of its rollouts. It reads the rollouts
        of these evaluations only, not those of the scenario's other evaluations, and not those of an ended
        evaluation whose progress ``ended_progress`` keeps."""
        now = time.time()
        views: list[dict[str, object]] = []
        for stored in evaluations:
            progress = self.ended_progress.read(scenario, stored)
            if progress is None and (items := self.results.rollouts(stored.evaluation_id, scenario)):
                progress = evaluation_progress(stored, items, now)
                self.ended_progress.add(scenario, stored, items, progress)
            if progress is not None and not score_levels:
                benchmarks = cast(list[dict[str, JsonValue]], progress["benchmarks"])
                progress = {key: value for key, value in progress.items() if key != "score_levels"}
                progress["benchmarks"] = [
                    {key: value for key, value in benchmark.items() if key != "score_levels"}
                    for benchmark in benchmarks
                ]
            views.append({**stored.to_json(), "progress": progress})
        return views

    def newest(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario``, newest first."""
        with self.dispatcher.locked_scenario(scenario) as current:
            return current.store.evaluations.newest(limit=limit)

    def newest_for_list(self, scenario: str, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations of ``scenario`` for an HTTP list, newest first."""
        with self.dispatcher.locked_scenario(scenario) as current:
            return current.store.evaluations.newest_for_list(limit=limit)


__all__ = [
    "CANCELLED_ERROR",
    "FAILURES_ERROR",
    "QUEUE_OWNER",
    "STOPPED_ERROR",
    "ArtifactEvaluations",
    "EvaluationResults",
    "NewArtifact",
    "NewArtifacts",
]
