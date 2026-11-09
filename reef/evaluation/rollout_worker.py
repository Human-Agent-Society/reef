"""The evaluation service's worker: it runs one rollout of an evaluation and writes its score.

Control adds an evaluation to the ``evaluations`` queue as one ``evaluation`` item and one ``evaluation_rollout``
item for each rollout (``reef.service.artifact_evaluations``). An ``EvaluationRolloutWorker`` takes a rollout item,
runs the task once with a Harbor agent against the candidate's model, and finishes the item with the rollout's
``RolloutScore``. Control reads the scores back when every rollout of the evaluation is final.

A worker asks inference to serve the candidate before each trial (a ``serve_artifact`` item of the ``serving`` queue),
and waits until the engine is ready. It answers the agent's model calls from that engine on a loopback endpoint
(``CandidateEndpoints``). The item writes no route, so no client reaches the candidate through ``served_models``, and
no call is a record, so an evaluation never trains the scenario. The worker keeps each call of the rollout, failed
calls too, as a record item of the scenario's owner in the data store as the call ends (``RolloutCalls``), so a client can read the calls
of a rollout while it runs. After each attempt, it also keeps the model calls and trajectories that the task's verifier
wrote to ``/logs/verifier/judge/`` (``verifier_output``, RFC #1293).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from reef_client.serve import CapturedTurn

from reef.core.errors import ReefError
from reef.core.evaluation import (
    EvaluationCall,
    EvaluationCallSummary,
    EvaluationTrajectory,
    PolicyHarness,
    PolicyModel,
    RolloutRequest,
    RolloutScore,
)
from reef.core.served_models import ServedArtifact
from reef.core.values import require_text
from reef.evaluation.artifact_endpoint import ArtifactEndpoint, CandidateArtifacts
from reef.execution.workers import QueueWorker, WorkerStopped
from reef.harness.adapters import get_adapter
from reef.harness.capture import CaptureObserver
from reef.harness.harbor.agent import tree_agent, trial_directory
from reef.harness.harbor.rollouts import DEFAULT_AGENT, TaskLab, TaskRollout, TaskRolloutRunner
from reef.harness.tree import HarnessTree, render_tree
from reef.storage.served_models import SERVE_ARTIFACT_KIND, SERVING_QUEUE
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of evaluations and their rollouts.
EVALUATIONS_QUEUE = "evaluations"
#: The kind of the item of one rollout.
EVALUATION_ROLLOUT_KIND = "evaluation_rollout"
#: The address at which a task's container reaches this host, by default.
DEFAULT_AGENT_HOST = "host.docker.internal"
#: The longest wait for inference to serve a candidate, in seconds: a first start can quantize its weights.
SERVE_WAIT_S = 12 * 3600.0
#: The wait between two reads of a serve item, in seconds.
SERVE_POLL_S = 1.0
#: The seconds between two checks of a running trial's lease.
LEASE_CHECK_S = 1.0
#: The limit of a tree agent's command, in seconds. Harbor's agent timeout of each task ends the run first.
TREE_AGENT_TIMEOUT_S = 24 * 3600.0
#: The directory of a trial's verifier directory where the verifier writes its own model calls and trajectories: the
#: task's ``/logs/verifier/judge/``, which Harbor downloads with the verifier directory (RFC #1293).
VERIFIER_JUDGE_DIR = Path("verifier") / "judge"
#: The file of the verifier's calls in ``VERIFIER_JUDGE_DIR``: one JSON object for each call, in call order.
VERIFIER_CALLS_FILE = "calls.jsonl"
#: The pattern of the names of the verifier's ATIF trajectory files in ``VERIFIER_JUDGE_DIR``.
VERIFIER_TRAJECTORY_FILES = "trajectory*.json"
#: The most rewards of its verifier that a rollout keeps.
MAX_VERIFIER_REWARDS = 64
#: The most calls that Reef keeps of the verifier of one attempt.
MAX_VERIFIER_CALLS = 2000
#: The most bytes of verifier files that Reef reads for one attempt.
MAX_VERIFIER_BYTES = 32 * 1024 * 1024


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
class TreeHarness:
    """A harness tree that a rollout runs, read from an artifact item.

    Values: ``adapter`` is the name of an adapter, ``api`` the API dialect of the model calls, and ``files`` maps
    each path of the tree to its text. Purpose: control reads it once when it accepts an evaluation, so that each
    rollout runs the same tree; a worker runs it.
    """

    adapter: str
    api: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        require_text(self.adapter, "adapter")
        require_text(self.api, "api")
        if not isinstance(self.files, Mapping):
            raise TypeError("files is a mapping of paths to text")

    def policy_harness(self) -> PolicyHarness:
        """The harness side of the policy: the adapter, with the tree rendered with no model binding."""
        return PolicyHarness(self.adapter, render_tree(HarnessTree(self.files), get_adapter(self.adapter)))

    def agent(self, directory: Path, *, bridge: bool = False) -> Mapping[str, object]:
        """The Harbor agent configuration that runs the tree, with its files under ``directory``. ``bridge``
        bridges the task container's loopback to the rollout's capture proxy (``tree_agent``)."""
        tree_dir = directory / "tree"
        HarnessTree(self.files).write(tree_dir)
        return tree_agent(
            get_adapter(self.adapter),
            api=self.api,
            tree_dir=tree_dir,
            files_dir=directory / "rendered",
            timeout=TREE_AGENT_TIMEOUT_S,
            bridge=bridge,
        )

    def to_json(self) -> dict[str, JsonValue]:
        return {"adapter": self.adapter, "api": self.api, "files": dict(self.files)}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> TreeHarness:
        return cls(value["adapter"], value["api"], value["files"])


@dataclass(frozen=True)
class EvaluationRollout:
    """The input of one ``evaluation_rollout`` item.

    Values: ``scenario`` and ``evaluation_id`` name the evaluation; ``artifact_id`` is the candidate artifact and
    ``component`` the component of the scenario's artifacts that it is;
    ``task_index`` is the task's position in the evaluation and ``task_name`` its name; ``repeat`` numbers the rollouts of one task from 0; ``model`` is the model that the
    service serves for the candidate; ``agent`` is the Harbor agent configuration, or None when ``tree`` is the
    harness; ``environment`` is the Harbor environment type; ``selection`` is the artifact selection that serves the
    candidate, as ``ArtifactSelection.to_dict`` gives it; ``tree`` is the harness tree, or None when ``agent``
    is the harness. The names are non-empty strings and the two numbers are non-negative integers.

    Purpose: control makes it when it adds an evaluation (``to_payload``); a worker reads it (``from_item``).
    """

    scenario: str
    evaluation_id: str
    artifact_id: str
    component: str
    task_index: int
    task_name: str
    repeat: int
    model: PolicyModel
    agent: Mapping[str, object] | None
    environment: str
    selection: Mapping[str, JsonValue]
    tree: TreeHarness | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.scenario, "scenario"),
            (self.evaluation_id, "evaluation_id"),
            (self.artifact_id, "artifact_id"),
            (self.component, "component"),
            (self.task_name, "task_name"),
            (self.environment, "environment"),
        ):
            require_text(value, name)
        for number, name in ((self.task_index, "task_index"), (self.repeat, "repeat")):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} is a non-negative integer")
        if not isinstance(self.model, PolicyModel):
            raise TypeError("model is a PolicyModel")
        if (self.agent is None) == (self.tree is None):
            raise ValueError("a rollout's harness is exactly one of agent and tree")
        if self.agent is not None and not isinstance(self.agent, Mapping):
            raise TypeError("agent is a JSON object")

    def item_id(self) -> str:
        return rollout_item_id(self.evaluation_id, self.task_index, self.repeat)

    def policy_harness(self) -> PolicyHarness:
        """The harness side of the policy that the rollout runs."""
        return agent_harness(self.agent) if self.agent is not None else cast(TreeHarness, self.tree).policy_harness()

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
                "repeat": self.repeat,
                "model": self.model.to_json(),
                "agent": None if self.agent is None else dict(self.agent),
                "environment": self.environment,
                "selection": dict(self.selection),
                "tree": None if self.tree is None else self.tree.to_json(),
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
            payload["repeat"],
            PolicyModel.from_json(payload["model"]),
            payload["agent"],
            payload["environment"],
            payload["selection"],
            None if payload["tree"] is None else TreeHarness.from_json(payload["tree"]),
        )


def rollout_item_id(evaluation_id: str, task_index: int, repeat: int) -> str:
    """The ID of the item of rollout ``repeat`` of task ``task_index`` of an evaluation."""
    return f"{EVALUATION_ROLLOUT_KIND}:{evaluation_id}:{task_index:05d}:{repeat:05d}"


@dataclass(frozen=True)
class CandidateEndpoint:
    """Where one rollout reaches the candidate's model.

    Values: ``url`` is a non-empty base URL that answers the model calls of a rollout from the candidate; ``token`` is the bearer token that opens it, or None without authentication.

    Purpose: ``CandidateEndpoints.open`` makes it; the worker's rollout runner calls it.
    """

    url: str
    token: str | None


class CandidateEndpoints(ABC):
    """Opens the endpoint of a candidate artifact for the time of one rollout, and keeps the calls of the rollout.

    Purpose: an ``EvaluationRolloutWorker`` opens one around each rollout, with the value that reaches the engine
    that inference started for the candidate (``LocalCandidateEndpoints``). It keeps each call of the rollout as the
    call ends (``add_call``).
    """

    @abstractmethod
    def task_directory(self, rollout: EvaluationRollout) -> Path:
        """The Harbor task directory of ``rollout``'s task, on this process's file system."""

    @abstractmethod
    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        """The endpoint of ``rollout``'s candidate, until the block ends."""

    @abstractmethod
    def add_call(self, rollout: EvaluationRollout, call: EvaluationCall) -> None:
        """Keep ``call``, a call of ``rollout``, as a record item in the data store
        (``reef.storage.record_items.evaluation_call_item``)."""

    @abstractmethod
    def add_trajectory(self, rollout: EvaluationRollout, trajectory: EvaluationTrajectory) -> None:
        """Keep ``trajectory``, which the verifier of ``rollout`` wrote, as a trajectory item in the data store
        (``reef.storage.record_items.evaluation_trajectory_item``)."""


class LocalCandidateEndpoints(CandidateEndpoints):
    """The candidate's model, on loopback in the process of the evaluation worker: an ``ArtifactEndpoint`` that
    sends each call to the engine that inference started, or to its scenario's runtime for an engine with no address.

    Values: ``artifacts`` finds the candidate of each rollout. Purpose: the evaluation worker opens it around each
    rollout.
    """

    def __init__(self, artifacts: CandidateArtifacts) -> None:
        self.artifacts = artifacts

    def task_directory(self, rollout: EvaluationRollout) -> Path:
        return self.artifacts.candidate_task(rollout.scenario, rollout.task_name)

    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        runtime, artifact = self.artifacts.candidate(rollout.scenario, rollout.artifact_id, rollout.component)
        with ArtifactEndpoint(runtime, artifact, served) as endpoint:
            yield CandidateEndpoint(endpoint.url, None)

    def add_call(self, rollout: EvaluationRollout, call: EvaluationCall) -> None:
        self.artifacts.add_evaluation_call(rollout.scenario, rollout.evaluation_id, rollout.artifact_id, call)

    def add_trajectory(self, rollout: EvaluationRollout, trajectory: EvaluationTrajectory) -> None:
        self.artifacts.add_evaluation_trajectory(rollout.scenario, rollout.evaluation_id, trajectory)


def reply_summary(path: str, reply: Mapping[str, Any]) -> tuple[int | None, int | None, str | None, int]:
    """The prompt tokens, the completion tokens, the finish reason and the number of tool calls of ``reply``, the
    200 answer to a call to ``path``.

    The fields come from an OpenAI chat completion, an OpenAI response or an Anthropic message, by ``path``. The
    prompt tokens of an Anthropic message add its cache tokens. The finish reason of an OpenAI response is the reason
    in its ``incomplete_details``, else its ``status``. A field that the reply does not hold is None, and its tool
    calls are 0; so is each field of a reply to another path.
    """
    usage = reply.get("usage")

    def tokens(*names: str) -> int | None:
        counts = [usage.get(name) for name in names] if isinstance(usage, Mapping) else []
        known = [count for count in counts if isinstance(count, int) and not isinstance(count, bool) and count >= 0]
        return sum(known) if known else None

    def reason(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    def objects(value: object) -> list[Mapping[str, Any]]:
        return [entry for entry in value if isinstance(entry, Mapping)] if isinstance(value, list) else []

    route = path.partition("?")[0]
    if route == "/v1/chat/completions":
        choices = objects(reply.get("choices"))
        choice: Mapping[str, Any] = choices[0] if choices else {}
        message = choice.get("message")
        tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        return (
            tokens("prompt_tokens"),
            tokens("completion_tokens"),
            reason(choice.get("finish_reason")),
            tool_call_count,
        )
    if route == "/v1/responses":
        details = reply.get("incomplete_details")
        finish = details.get("reason") if isinstance(details, Mapping) else reply.get("status")
        tool_call_count = sum(entry.get("type") == "function_call" for entry in objects(reply.get("output")))
        return tokens("input_tokens"), tokens("output_tokens"), reason(finish), tool_call_count
    if route == "/v1/messages":
        tool_call_count = sum(entry.get("type") == "tool_use" for entry in objects(reply.get("content")))
        prompt = tokens("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        return prompt, tokens("output_tokens"), reason(reply.get("stop_reason")), tool_call_count
    return None, None, None, 0


def evaluation_call(turn: CapturedTurn, rollout: EvaluationRollout, attempt: int, index: int) -> EvaluationCall:
    """The call ``index`` of attempt ``attempt`` of ``rollout`` that the capture proxy captured as ``turn``.

    The call started ``turn.duration_s`` seconds before the proxy captured it at ``turn.at``. Its summary fields come
    from the reply of a 200 call (``reply_summary``); a failed call has none, and 0 tool calls. The kept response has
    no ``training`` block: it holds token IDs for training, which nobody reads in a call log.
    """
    prompt, completion, finish, tool_calls = (
        reply_summary(turn.path, turn.response)
        if turn.status == 200 and turn.response is not None
        else (None, None, None, 0)
    )
    summary = EvaluationCallSummary(
        task_index=rollout.task_index,
        repeat=rollout.repeat,
        attempt=attempt,
        index=index,
        path=turn.path,
        status=turn.status,
        started_at=turn.at - turn.duration_s,
        duration_s=turn.duration_s,
        prompt_tokens=prompt,
        completion_tokens=completion,
        finish_reason=finish,
        tool_calls=tool_calls,
    )
    response = turn.response
    if response is not None and "training" in response:
        response = {key: value for key, value in response.items() if key != "training"}
    return EvaluationCall(summary, turn.request, response)


def verifier_call(line: str, rollout: EvaluationRollout, attempt: int, index: int) -> EvaluationCall:
    """The verifier call ``index`` of attempt ``attempt`` of ``rollout`` that ``line`` of ``VERIFIER_CALLS_FILE``
    holds: ``{request, response, status, started_at, duration_s, path, label}``.

    ``request`` is a JSON object, ``response`` a JSON object or null, ``status`` an HTTP status (default 200),
    ``started_at`` and ``duration_s`` numbers of seconds, ``path`` the called path (default ``/v1/chat/completions``)
    and ``label`` a short string or absent. Other keys are ignored. The summary fields come from the response of a
    200 call (``reply_summary``). Raise ``ValueError`` or ``TypeError`` for a line that is no such call.
    """
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("a verifier call is a JSON object")
    request, response = value.get("request"), value.get("response")
    if not isinstance(request, dict) or not (response is None or isinstance(response, dict)):
        raise ValueError("a verifier call has a request object and a response object or null")
    started_at, duration_s = value.get("started_at"), value.get("duration_s")
    if not all(
        isinstance(time_s, int | float) and not isinstance(time_s, bool) for time_s in (started_at, duration_s)
    ):
        raise ValueError("a verifier call has started_at and duration_s in seconds")
    path, status = value.get("path", "/v1/chat/completions"), value.get("status", 200)
    if not isinstance(path, str):
        raise ValueError("a verifier call's path is a string")
    prompt, completion, finish, tool_calls = (
        reply_summary(path, response) if status == 200 and response is not None else (None, None, None, 0)
    )
    summary = EvaluationCallSummary(
        task_index=rollout.task_index,
        repeat=rollout.repeat,
        attempt=attempt,
        index=index,
        path=path,
        status=status,
        started_at=float(cast(float, started_at)),
        duration_s=float(cast(float, duration_s)),
        prompt_tokens=prompt,
        completion_tokens=completion,
        finish_reason=finish,
        tool_calls=tool_calls,
        caller="verifier",
        label=value.get("label"),
    )
    return EvaluationCall(summary, request, response)


def verifier_output(
    trial: Path | None, rollout: EvaluationRollout, attempt: int
) -> tuple[list[EvaluationCall], list[EvaluationTrajectory], int]:
    """The calls and trajectories that the verifier of attempt ``attempt`` of ``rollout`` wrote to
    ``VERIFIER_JUDGE_DIR`` of the trial directory ``trial``, and the number of lines and files that it skipped.

    The calls come from ``VERIFIER_CALLS_FILE``, in line order, and the trajectories from the files that match
    ``VERIFIER_TRAJECTORY_FILES``, in name order. It reads at most ``MAX_VERIFIER_BYTES`` of files and keeps at most
    ``MAX_VERIFIER_CALLS`` calls. It skips a line that is no call (``verifier_call``), a call over the limit, a file
    that is no ATIF document, a file over the byte limit and a symbolic link, which the task could point at this
    host's files. A trial without the directory, or None, has no output.
    """
    judge = None if trial is None else trial / VERIFIER_JUDGE_DIR
    if judge is None or not judge.is_dir() or judge.is_symlink():
        return [], [], 0
    calls: list[EvaluationCall] = []
    trajectories: list[EvaluationTrajectory] = []
    skipped, budget = 0, MAX_VERIFIER_BYTES
    files = [judge / VERIFIER_CALLS_FILE, *sorted(judge.glob(VERIFIER_TRAJECTORY_FILES))]
    for path in files:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > budget:
            skipped += 1
            continue
        budget -= path.stat().st_size
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name != VERIFIER_CALLS_FILE:
            try:
                document = json.loads(text)
                trajectories.append(
                    EvaluationTrajectory(
                        rollout.task_index, rollout.repeat, attempt, len(trajectories), path.name, document
                    )
                )
            except (ValueError, TypeError):
                skipped += 1
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            if len(calls) == MAX_VERIFIER_CALLS:
                skipped += 1
                continue
            try:
                calls.append(verifier_call(line, rollout, attempt, len(calls)))
            except (ValueError, TypeError):
                skipped += 1
    return calls, trajectories, skipped


def verifier_rewards(rewards: Mapping[str, float]) -> dict[str, JsonValue] | None:
    """The rewards that a rollout keeps beside its score (RFC #1297): the first ``MAX_VERIFIER_REWARDS`` finite
    rewards of its verifier, by name, or None when the verifier wrote none. Nothing reads them to score."""
    kept = {name: value for name, value in rewards.items() if math.isfinite(value)}
    return dict(list(kept.items())[:MAX_VERIFIER_REWARDS]) or None


class RolloutCalls(CaptureObserver):
    """Keeps each call of one attempt of an evaluation rollout in the data store, as the call ends.

    Values: ``endpoints`` keeps the calls (``CandidateEndpoints.add_call``). ``rollout`` is the rollout, and
    ``attempt`` the 1-based attempt of its work item. ``next_index`` is the number of calls captured so far, which is
    the index of the next call.

    Purpose: ``EvaluationRolloutWorker.score`` gives one to the rollout's runner, and closes it when the rollout ends.
    One writer thread keeps the calls in call order, so the agent does not wait for the store. A call that the store
    does not keep is logged, and the rollout goes on.
    """

    def __init__(self, endpoints: CandidateEndpoints, rollout: EvaluationRollout, attempt: int) -> None:
        self.endpoints = endpoints
        self.rollout = rollout
        self.attempt = attempt
        self.next_index = 0
        self.lock = threading.Lock()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-evaluation-calls")

    def call_captured(self, turn: CapturedTurn) -> None:
        with self.lock:
            index = self.next_index
            self.next_index += 1
            self.writer.submit(self.keep, evaluation_call(turn, self.rollout, self.attempt, index))

    def keep(self, call: EvaluationCall) -> None:
        try:
            self.endpoints.add_call(self.rollout, call)
        except Exception:
            # The call log is for people to read: a call that it misses must not change the rollout or its score.
            logger.exception("could not keep call %d of rollout %s", call.summary.index, self.rollout.item_id())

    def close(self) -> None:
        """Wait until every captured call is kept or logged."""
        self.writer.shutdown(wait=True)


async def run_while_held(
    runner: TaskRolloutRunner, request: RolloutRequest, lease_lost: threading.Event
) -> TaskRollout:
    """Run the rollout of ``request`` with ``runner`` and return it. When ``lease_lost`` is set first, cancel the run,
    so that Harbor stops the trial and its sandbox, and raise ``WorkLeaseLost``: a client cancelled the rollout, or
    another worker took it."""
    run = asyncio.ensure_future(runner.run_rollouts_async([request], asyncio.Semaphore(1)))
    while not lease_lost.is_set():
        done, _ = await asyncio.wait({run}, timeout=LEASE_CHECK_S)
        if done:
            [result] = run.result()
            return result
    run.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run
    raise WorkLeaseLost("the evaluation worker lost its lease")


class EvaluationRolloutWorker(QueueWorker):
    """The worker of the evaluation service.

    Values: as ``QueueWorker``, on the rollouts of the evaluations queue, with one rollout in each slot;
    ``endpoints`` opens the candidate's model for each rollout; ``lab`` runs each Harbor trial, None for reef-eval's
    lab; ``agent_host`` is the address at which a task's container reaches this host. Every failed rollout is
    retried: a trial fails for reasons outside its input, such as a sandbox that did not start.

    Purpose: the evaluation service starts one (``start``) and stops it (``stop``).
    """

    permanent_errors = ()

    def __init__(
        self,
        queue: WorkQueue,
        endpoints: CandidateEndpoints,
        *,
        worker_id: str,
        lab: TaskLab | None = None,
        agent_host: str = DEFAULT_AGENT_HOST,
        poll_seconds: float = 0.5,
        slots: int = 1,
    ) -> None:
        super().__init__(
            queue,
            EVALUATIONS_QUEUE,
            [EVALUATION_ROLLOUT_KIND],
            worker_id=worker_id,
            poll_seconds=poll_seconds,
            slots=slots,
        )
        self.endpoints = endpoints
        self.lab = lab
        self.agent_host = agent_host

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        score, rewards = self.score(EvaluationRollout.from_item(item), lease_lost, item.attempt_count)
        return {"score": score.to_json(), "rewards": rewards}

    def served_candidate(self, rollout: EvaluationRollout, lease_lost: threading.Event) -> ServedArtifact | None:
        """Ask inference to serve the candidate of ``rollout``, wait until it is ready, and return the value that
        reaches it: None for a runtime with no address. Raise ``ReefError`` when inference fails or takes too long, so
        that the rollout fails and does not score 0."""
        item_id = f"{SERVE_ARTIFACT_KIND}:{uuid.uuid4().hex}"
        payload: dict[str, JsonValue] = {"selection": dict(rollout.selection), "component": rollout.component}
        self.queue.add(NewWork(SERVING_QUEUE, item_id, rollout.scenario, SERVE_ARTIFACT_KIND, payload))
        deadline = time.monotonic() + SERVE_WAIT_S
        while time.monotonic() < deadline:
            if self.stopped.is_set():
                raise WorkerStopped
            if lease_lost.is_set():
                raise WorkLeaseLost("the evaluation worker lost its lease")
            item = self.queue.get(SERVING_QUEUE, item_id)
            if item is not None and item.final:
                if item.state != "finished":
                    raise ReefError(f"inference could not serve artifact {rollout.artifact_id}: {item.error}")
                served = cast(Mapping[str, Any], item.result)["served"]
                return None if served is None else ServedArtifact.from_json(served)
            time.sleep(SERVE_POLL_S)
        raise ReefError(f"inference did not serve artifact {rollout.artifact_id} in {SERVE_WAIT_S:.0f} s")

    def score(
        self, rollout: EvaluationRollout, lease_lost: threading.Event, attempt: int
    ) -> tuple[RolloutScore, dict[str, JsonValue] | None]:
        """Run ``rollout`` once, when its candidate is ready, and return its score and the rewards of its verifier
        (``verifier_rewards``). ``attempt`` is the 1-based attempt of the rollout's work item; the rollout's calls are
        kept under it (``RolloutCalls``)."""
        task_directory = self.endpoints.task_directory(rollout)
        served = self.served_candidate(rollout, lease_lost)
        # A tree in a remote environment reaches the capture proxy through the bridge, on loopback (RFC #986).
        bridged = rollout.tree is not None and rollout.environment != "docker"
        calls = RolloutCalls(self.endpoints, rollout, attempt)
        with (
            tempfile.TemporaryDirectory(prefix="reef-evaluation-") as work,
            self.endpoints.open(rollout, served) as endpoint,
        ):
            runner = TaskRolloutRunner(
                reef_url=endpoint.url,
                scenario=rollout.scenario,
                model=rollout.model.name,
                model_adapter=rollout.model.adapter,
                token=endpoint.token,
                work_dir=Path(work),
                agent=rollout.tree.agent(Path(work), bridge=bridged) if rollout.tree is not None else rollout.agent,
                # A tree's agent on Docker runs in the task's container, which reaches the capture proxy at this address.
                agent_host=None if rollout.tree is None or bridged else self.agent_host,
                environment=rollout.environment,
                is_reporting=False,
                is_unrecorded=True,
                concurrency=1,
                lab=self.lab,
                capture_observer=calls,
            )
            request = RolloutRequest(runner.harness, runner.policy_model, str(task_directory))
            try:
                result = asyncio.run(run_while_held(runner, request, lease_lost))
            finally:
                calls.close()
            # The trial directory is under ``work``, so read the verifier's files before the block removes it.
            self.keep_verifier_output(rollout, attempt, trial_directory(result.trial_uri))
            if not result.turns:
                # A verifier can write a reward with no call, so that reward says nothing of the candidate's model. A
                # trial whose sandbox did not start makes no call either, so the attempt fails with the reason, and
                # the rollout is retried as any other failed attempt.
                raise ReefError(f"the rollout reached no model: {result.error or 'the agent made no model call'}")
            score = runner.rollout_score(result)
            return score, None if score.is_failed else verifier_rewards(result.rewards)

    def keep_verifier_output(self, rollout: EvaluationRollout, attempt: int, trial: Path | None) -> None:
        """Keep the calls and trajectories that the verifier of attempt ``attempt`` of ``rollout`` wrote to the trial
        directory ``trial`` (``verifier_output``). They are for people to read, so a file that Reef cannot read, or a
        call or trajectory that it cannot keep, is logged, and the rollout and its score do not change."""
        try:
            calls, trajectories, skipped = verifier_output(trial, rollout, attempt)
        except OSError:
            logger.exception("could not read the verifier output of rollout %s", rollout.item_id())
            return
        if skipped:
            logger.warning("skipped %d verifier lines or files of rollout %s", skipped, rollout.item_id())
        for call in calls:
            try:
                self.endpoints.add_call(rollout, call)
            except Exception:
                logger.exception(
                    "could not keep verifier call %d of rollout %s", call.summary.index, rollout.item_id()
                )
        for trajectory in trajectories:
            try:
                self.endpoints.add_trajectory(rollout, trajectory)
            except Exception:
                logger.exception(
                    "could not keep verifier trajectory %s of rollout %s", trajectory.name, rollout.item_id()
                )


__all__ = [
    "EVALUATIONS_QUEUE",
    "EVALUATION_ROLLOUT_KIND",
    "MAX_VERIFIER_BYTES",
    "MAX_VERIFIER_CALLS",
    "VERIFIER_CALLS_FILE",
    "VERIFIER_JUDGE_DIR",
    "VERIFIER_TRAJECTORY_FILES",
    "CandidateEndpoint",
    "CandidateEndpoints",
    "EvaluationRollout",
    "EvaluationRolloutWorker",
    "LocalCandidateEndpoints",
    "RolloutCalls",
    "TreeHarness",
    "agent_harness",
    "evaluation_call",
    "harbor_agent",
    "reply_summary",
    "rollout_item_id",
    "verifier_call",
    "verifier_output",
]
