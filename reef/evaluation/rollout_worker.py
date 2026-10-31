"""The evaluation service's worker: it runs one rollout of an evaluation and writes its score.

Control adds an evaluation to the ``evaluations`` queue as one ``evaluation`` item and one ``evaluation_rollout``
item for each rollout (``reef.service.artifact_evaluations``). An ``EvaluationRolloutWorker`` takes a rollout item,
runs the task once with a Harbor agent against the candidate's model, and finishes the item with the rollout's
``RolloutScore``. Control reads the scores back when every rollout of the evaluation is final.

A worker serves the candidate itself: it resolves the candidate artifact from the scenario's repository and answers
the agent's model calls from it with the scenario's runtime, on a loopback endpoint (``CandidateEndpoints``). The
inference service and ``served_models`` take no part, and no call is a record, so an evaluation never trains the
scenario.
"""

from __future__ import annotations

import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from reef.core.evaluation import PolicyHarness, PolicyModel, RolloutRequest, RolloutScore
from reef.core.values import require_text
from reef.evaluation.artifact_endpoint import ArtifactEndpoint, CandidateArtifacts
from reef.execution.workers import QueueWorker
from reef.harness.client.tasks import DEFAULT_AGENT, TaskLab, TaskRolloutRunner
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue

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


@dataclass(frozen=True)
class CandidateEndpoint:
    """Where one rollout reaches the candidate's model.

    Values: ``url`` is a non-empty base URL that answers the model calls of a rollout from the candidate; ``token`` is the bearer token that opens it, or None without authentication.

    Purpose: ``CandidateEndpoints.open`` makes it; the worker's rollout runner calls it.
    """

    url: str
    token: str | None


class CandidateEndpoints(ABC):
    """Opens the endpoint of a candidate artifact for the time of one rollout.

    Purpose: an ``EvaluationRolloutWorker`` opens one around each rollout. The evaluation service serves the
    candidate itself (``reef.service.artifact_evaluations.LocalCandidateEndpoints``): no other service takes part.
    """

    @abstractmethod
    @contextmanager
    def open(self, rollout: EvaluationRollout) -> Iterator[CandidateEndpoint]:
        """The endpoint of ``rollout``'s candidate, until the block ends."""


class LocalCandidateEndpoints(CandidateEndpoints):
    """The candidate's model, served in the process of the evaluation workers: an ``ArtifactEndpoint`` that
    answers from the candidate artifact with its scenario's runtime.

    Values: ``artifacts`` finds the candidate of each rollout. Purpose: the evaluation workers open it around each
    rollout, so neither control nor the inference service serves the candidate.
    """

    def __init__(self, artifacts: CandidateArtifacts) -> None:
        self.artifacts = artifacts

    @contextmanager
    def open(self, rollout: EvaluationRollout) -> Iterator[CandidateEndpoint]:
        runtime, artifact = self.artifacts.candidate(rollout.scenario, rollout.artifact_id, rollout.component)
        with ArtifactEndpoint(runtime, artifact) as endpoint:
            yield CandidateEndpoint(endpoint.url, None)


class EvaluationRolloutWorker(QueueWorker):
    """One worker of the evaluation service.

    Values: as ``QueueWorker``, on the rollouts of the evaluations queue; ``endpoints`` opens the candidate's model
    for each rollout; ``lab`` runs each Harbor trial, None for reef-eval's lab. Every failed rollout is retried:
    a trial fails for reasons outside its input, such as a sandbox that did not start.

    Purpose: the evaluation service starts one or more (``start``) and stops them (``stop``).
    """

    permanent_errors = ()

    def __init__(
        self,
        queue: WorkQueue,
        endpoints: CandidateEndpoints,
        *,
        worker_id: str,
        lab: TaskLab | None = None,
        poll_seconds: float = 0.5,
    ) -> None:
        super().__init__(
            queue, EVALUATIONS_QUEUE, [EVALUATION_ROLLOUT_KIND], worker_id=worker_id, poll_seconds=poll_seconds
        )
        self.endpoints = endpoints
        self.lab = lab

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        return {"score": self.score(EvaluationRollout.from_item(item)).to_json()}

    def score(self, rollout: EvaluationRollout) -> RolloutScore:
        """Run ``rollout`` once and return its score."""
        with (
            tempfile.TemporaryDirectory(prefix="reef-evaluation-") as work,
            self.endpoints.open(rollout) as endpoint,
        ):
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
    "LocalCandidateEndpoints",
    "agent_harness",
    "evaluation_item_id",
    "harbor_agent",
    "rollout_item_id",
]
