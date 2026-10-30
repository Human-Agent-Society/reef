"""The evaluation service's worker: it runs one rollout of an evaluation and writes its score.

Control adds an evaluation to the ``evaluations`` queue as one ``evaluation`` item and one ``evaluation_rollout``
item for each rollout (``reef.service.artifact_evaluations``). An ``EvaluationRolloutWorker`` takes a rollout item,
runs the task once with a Harbor agent against the candidate's model, and finishes the item with the rollout's
``RolloutScore``. Control reads the scores back when every rollout of the evaluation is final.

Control serves the candidate beside the active artifact while its rollouts run, so ``served_models`` holds a row
for it. A worker reaches the candidate at the inference service's unrecorded URL of that artifact
(``CandidateEndpoints``), in a split deployment and in the one-process layout alike. These calls are not recorded,
so an evaluation never trains the scenario.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from reef.core.evaluation import PolicyHarness, PolicyModel, RolloutRequest, RolloutScore
from reef.core.served_models import require_text
from reef.execution.workers import ServiceWorker
from reef.harness.client.tasks import DEFAULT_AGENT, TaskLab, TaskRolloutRunner
from reef.storage.work_queue import JsonValue, LeaseExtender, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of evaluations and their rollouts.
EVALUATIONS_QUEUE = "evaluations"
#: The kind of the item that holds an evaluation's request; no worker takes it.
EVALUATION_KIND = "evaluation"
#: The kind of the item of one rollout.
EVALUATION_ROLLOUT_KIND = "evaluation_rollout"


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


def agent_harness(agent: Mapping[str, object]) -> PolicyHarness:
    """The harness side of the policy that ``agent`` runs: the Harbor agent with no plugin. Raise ``ValueError``
    for an agent with neither a ``name`` nor an ``import_path``."""
    name = agent.get("name") or agent.get("import_path")
    if not isinstance(name, str) or not name:
        raise ValueError("a Harbor agent has a name or an import_path")
    return PolicyHarness(name)


@dataclass(frozen=True)
class EvaluationRollout:
    """The input of one ``evaluation_rollout`` item.

    Values: ``scenario`` and ``evaluation_id`` name the evaluation; ``artifact_id`` is the candidate artifact and
    ``component`` the component of the scenario's artifacts that it is;
    ``task_index`` is the task's position in the evaluation and ``task_name`` its name; ``task_path`` is the
    Harbor task directory; ``repeat`` numbers the rollouts of one task from 0; ``model`` is the model that the
    service serves for the candidate; ``agent`` is the Harbor agent configuration; ``environment`` is the Harbor
    environment type. The names are non-empty strings and the two numbers are non-negative integers.

    Purpose: control makes it when it adds an evaluation (``to_payload``); a worker reads it (``from_item``).
    """

    scenario: str
    evaluation_id: str
    artifact_id: str
    component: str
    task_index: int
    task_name: str
    task_path: str
    repeat: int
    model: PolicyModel
    agent: Mapping[str, object]
    environment: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.scenario, "scenario"),
            (self.evaluation_id, "evaluation_id"),
            (self.artifact_id, "artifact_id"),
            (self.component, "component"),
            (self.task_name, "task_name"),
            (self.task_path, "task_path"),
            (self.environment, "environment"),
        ):
            require_text(value, name)
        for number, name in ((self.task_index, "task_index"), (self.repeat, "repeat")):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} is a non-negative integer")
        if not isinstance(self.model, PolicyModel):
            raise TypeError("model is a PolicyModel")
        if not isinstance(self.agent, Mapping):
            raise TypeError("agent is a JSON object")

    def item_id(self) -> str:
        return rollout_item_id(self.evaluation_id, self.task_index, self.repeat)

    def to_payload(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "scenario": self.scenario,
                "evaluation_id": self.evaluation_id,
                "artifact_id": self.artifact_id,
                "component": self.component,
                "task_index": self.task_index,
                "task_name": self.task_name,
                "task_path": self.task_path,
                "repeat": self.repeat,
                "model": self.model.to_json(),
                "agent": dict(self.agent),
                "environment": self.environment,
            },
        )

    @classmethod
    def from_item(cls, item: WorkItem) -> EvaluationRollout:
        payload = cast(Mapping[str, Any], item.payload)
        return cls(
            payload["scenario"],
            payload["evaluation_id"],
            payload["artifact_id"],
            payload["component"],
            payload["task_index"],
            payload["task_name"],
            payload["task_path"],
            payload["repeat"],
            PolicyModel.from_json(payload["model"]),
            payload["agent"],
            payload["environment"],
        )


def rollout_item_id(evaluation_id: str, task_index: int, repeat: int) -> str:
    """The ID of the item of rollout ``repeat`` of task ``task_index`` of an evaluation."""
    return f"{EVALUATION_ROLLOUT_KIND}:{evaluation_id}:{task_index:05d}:{repeat:05d}"


def evaluation_item_id(evaluation_id: str) -> str:
    """The ID of the item that holds an evaluation's request."""
    return f"{EVALUATION_KIND}:{evaluation_id}"


def candidate_url(service_url: str, scenario: str, artifact_id: str) -> str:
    """The unrecorded URL of one artifact of a scenario on the Reef service at ``service_url``, for rollouts."""
    return f"{service_url.rstrip('/')}/reef/scenarios/{scenario}/artifacts/{artifact_id}/evaluation"


@dataclass(frozen=True)
class CandidateEndpoint:
    """Where one rollout reaches the candidate's model.

    Values: ``url`` is a non-empty base URL that answers Reef's inference routes for the candidate without
    recording; ``token`` is the bearer token that opens it, or None without authentication.

    Purpose: ``CandidateEndpoints.endpoint`` makes it; the worker's rollout runner calls it.
    """

    url: str
    token: str | None


class CandidateEndpoints:
    """The candidate's unrecorded URL on the inference service at ``inference_url``, opened with ``token``.

    Values: ``inference_url`` is the non-empty base URL of a service that answers Reef's inference routes from
    ``served_models``; ``token`` is a token that it accepts on the candidate route, or None for a service without
    tokens. Purpose: an ``EvaluationRolloutWorker`` reads the endpoint of each rollout's candidate (``endpoint``).
    Raise ``ValueError`` at construction for an empty ``inference_url``.
    """

    def __init__(self, inference_url: str, token: str | None) -> None:
        if not inference_url:
            raise ValueError("the evaluation workers need the inference service's URL")
        self.inference_url = inference_url
        self.token = token

    def endpoint(self, rollout: EvaluationRollout) -> CandidateEndpoint:
        """The endpoint of ``rollout``'s candidate."""
        return CandidateEndpoint(candidate_url(self.inference_url, rollout.scenario, rollout.artifact_id), self.token)


class EvaluationRolloutWorker(ServiceWorker):
    """One worker of the evaluation service.

    Values: ``queue`` is the work queue; ``endpoints`` names the candidate's URL for each rollout; ``lab`` runs
    each Harbor trial, None for reef-eval's lab; ``worker_id`` names this worker in leases; ``poll_seconds`` is the
    wait between two looks at the queue.

    Purpose: the evaluation service starts one or more (``start``) and stops them (``stop``).
    """

    def __init__(
        self,
        queue: WorkQueue,
        endpoints: CandidateEndpoints,
        *,
        worker_id: str,
        lab: TaskLab | None = None,
        poll_seconds: float = 0.5,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds is positive")
        self.queue = queue
        self.endpoints = endpoints
        self.lab = lab
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name=f"reef-evaluation-{self.worker_id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                took = self.run_once()
            except Exception:
                # The store can fail for a moment; the worker tries again after its poll time.
                logger.exception("evaluation worker %s could not read the evaluations queue", self.worker_id)
                took = False
            if not took:
                self.stopped.wait(self.poll_seconds)

    def run_once(self) -> bool:
        """Take one rollout and run it; return whether there was one."""
        self.queue.requeue_expired()
        item = self.queue.take(EVALUATIONS_QUEUE, self.worker_id, [EVALUATION_ROLLOUT_KIND])
        if item is None:
            return False
        try:
            with LeaseExtender(self.queue, item, self.worker_id):
                self.queue.start(EVALUATIONS_QUEUE, item.id, self.worker_id)
                score = self.score(EvaluationRollout.from_item(item))
            self.queue.finish(EVALUATIONS_QUEUE, item.id, self.worker_id, {"score": score.to_json()})
        except WorkLeaseLost:
            logger.warning("evaluation worker %s lost rollout %s", self.worker_id, item.id)
        except Exception as exc:
            logger.exception("evaluation worker %s could not run rollout %s", self.worker_id, item.id)
            try:
                self.queue.fail(
                    EVALUATIONS_QUEUE, item.id, self.worker_id, f"{type(exc).__name__}: {exc}", retryable=True
                )
            except WorkLeaseLost:
                logger.warning("evaluation worker %s lost rollout %s", self.worker_id, item.id)
        return True

    def score(self, rollout: EvaluationRollout) -> RolloutScore:
        """Run ``rollout`` once and return its score."""
        endpoint = self.endpoints.endpoint(rollout)
        with tempfile.TemporaryDirectory(prefix="reef-evaluation-") as work:
            runner = TaskRolloutRunner(
                reef_url=endpoint.url,
                scenario=rollout.scenario,
                model=rollout.model.name,
                model_adapter=rollout.model.adapter,
                token=endpoint.token,
                work_dir=Path(work),
                agent=rollout.agent,
                environment=rollout.environment,
                is_reporting=False,
                is_unrecorded=True,
                concurrency=1,
                lab=self.lab,
            )
            [result] = runner.run_rollouts([RolloutRequest(runner.harness, runner.served_model, rollout.task_path)])
            return runner.rollout_score(result)


__all__ = [
    "EVALUATIONS_QUEUE",
    "EVALUATION_KIND",
    "EVALUATION_ROLLOUT_KIND",
    "CandidateEndpoint",
    "CandidateEndpoints",
    "EvaluationRollout",
    "EvaluationRolloutWorker",
    "agent_harness",
    "candidate_url",
    "evaluation_item_id",
    "harbor_agent",
    "rollout_item_id",
]
