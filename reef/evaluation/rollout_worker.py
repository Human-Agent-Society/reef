"""The evaluation service's worker: it runs one rollout of an evaluation and writes its score.

Control adds an evaluation to the ``evaluations`` queue as one ``evaluation`` item and one ``evaluation_rollout``
item for each rollout (``reef.control.operations.artifact_evaluations``). An ``EvaluationRolloutWorker`` takes a rollout item,
runs the task once with a Harbor agent against the candidate's model, and finishes the item with the rollout's
``RolloutScore``. Control reads the scores back when every rollout of the evaluation is final.

On ``e2b``, control first adds one ``evaluation_task_template`` item for each task, and adds the task's rollouts once
that item is final. A ``TaskTemplateWorker`` takes a template item and builds the task's sandbox templates for the
owner of the scenario's items. So a rollout that waits for a template builder holds no rollout slot of its evaluation
and no sandbox slot of a worker. A rollout of an evaluation that requires prepared templates runs no trial when its
task's template item failed, and its trial builds no template.

A worker asks inference to serve the candidate before each trial (a ``serve_artifact`` item of the ``serving`` queue),
and waits until the engine is ready. It answers the agent's model calls from that engine on a loopback endpoint
(``CandidateEndpoints``). The item writes no route, so no client reaches the candidate through ``served_models``, and
no call is a record, so an evaluation never trains the scenario. The worker keeps each call of the rollout, failed
calls too, as a record item of the scenario's owner in the data store as the call ends (``RolloutCalls``), so a client can read the calls
of a rollout while it runs. After each attempt, it also keeps the model calls and trajectories that the task's verifier
wrote to ``/logs/verifier/judge/`` (``verifier_output``, RFC #1293), and gives the trial to the deployment's
``TrialObserver``, when it names one.
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
from collections.abc import Awaitable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from reef_client.serve import CapturedTurn, has_tools

from reef.core.errors import ReefError, UpstreamLimit
from reef.core.evaluation import (
    EvaluationCall,
    EvaluationCallSummary,
    EvaluationTrajectory,
    PolicyHarness,
    PolicyModel,
    RolloutRequest,
    RolloutScore,
)
from reef.core.job_data import FILES_DIGEST_PATTERN
from reef.core.served_models import ServedArtifact
from reef.core.values import require_text
from reef.evaluation.artifact_endpoint import ArtifactEndpoint, CandidateArtifacts, ProviderCooldowns, refusal_limit
from reef.evaluation.task_templates import TaskTemplateBuilder
from reef.evaluation.trials import EvaluationTrial, TrialObserver
from reef.execution.sandbox.process import SANDBOX_CAPACITY_ERRORS, SANDBOX_REQUEST_ERRORS
from reef.execution.workers import ItemDeferredError, ItemFailedError, QueueWorker, WorkerStoppedError
from reef.harness.adapters import get_adapter
from reef.harness.capture import CaptureObserver
from reef.harness.harbor.agent import task_agent_tools, tree_agent, trial_directory
from reef.harness.harbor.rollouts import DEFAULT_AGENT, TaskLab, TaskRolloutRunner, trial_exception_type
from reef.harness.tasks.model_binding import token_limits
from reef.harness.tree import HarnessTree, render_tree
from reef.storage.coordination.served_models import SERVE_ARTIFACT_KIND, SERVING_QUEUE
from reef.storage.coordination.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLostError, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of evaluations and their rollouts.
EVALUATIONS_QUEUE = "evaluations"
#: The kind of the item of one rollout.
EVALUATION_ROLLOUT_KIND = "evaluation_rollout"
#: The kind of the item that builds the sandbox templates of one task of an evaluation on e2b.
TASK_TEMPLATE_KIND = "evaluation_task_template"
#: The address at which a task's container reaches this host, by default.
DEFAULT_AGENT_HOST = "host.docker.internal"
#: The longest wait for inference to serve a candidate, in seconds: a first start can quantize its weights.
SERVE_WAIT_S = 12 * 3600.0
#: The wait between two reads of a serve item, in seconds.
SERVE_POLL_S = 1.0
#: The longest time that one attempt of a template item waits for its task's templates. An AgentENV worker builds a
#: few templates at a time, so a template can wait hours for a free builder when many tasks are new.
TEMPLATE_WAIT_S = 6 * 3600.0
#: The seconds between two checks of a running trial's lease.
LEASE_CHECK_S = 1.0
#: What a stopping evaluation worker does with the rollouts that it runs: ``requeue`` cancels each one and gives it
#: back to the queue, where another worker starts it again; ``finish`` takes no new rollout and finishes each one,
#: and requeues those that still run at the end of the worker's ``finish_timeout_s``.
ON_STOP_CHOICES = ("requeue", "finish")
#: The limit of a tree agent's command, in seconds. Harbor's agent timeout of each task ends the run first.
TREE_AGENT_TIMEOUT_S = 24 * 3600.0
#: The variable of a tree harness's ``env`` that caps the agent's wall-clock time, in seconds.
AGENT_TIMEOUT_ENV = "REEF_AGENT_TIMEOUT_SECONDS"
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
#: The share of a rollout's model-call time that failed calls of the inference service may take before the rollout
#: has no score: below it, the agent lost little time to the failures, and its score still judges the model.
SERVICE_FAILURE_SHARE = 0.05


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


def agent_timeout_seconds(env: Mapping[str, str]) -> float | None:
    """Return the positive finite wall-clock timeout in seconds from ``AGENT_TIMEOUT_ENV`` in ``env``, or None when
    unset. Raise ``ValueError`` for any other value."""
    raw = env.get(AGENT_TIMEOUT_ENV)
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{AGENT_TIMEOUT_ENV} must be a positive number of seconds, not {raw!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{AGENT_TIMEOUT_ENV} must be a positive number of seconds, not {raw!r}")
    return seconds


def task_agent_timeout_seconds(task_directory: Path) -> float | None:
    """Return the positive finite ``[agent].timeout_sec`` parsed by Harbor from ``task_directory/task.toml``, or None
    when the file or timeout is absent."""
    task_file = task_directory / "task.toml"
    if not task_file.is_file():
        return None
    from harbor.models.task.config import TaskConfig

    timeout = TaskConfig.model_validate_toml(task_file.read_text(encoding="utf-8")).agent.timeout_sec
    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        return None
    return float(timeout)


@dataclass(frozen=True)
class TreeHarness:
    """A harness tree that a rollout runs, read from an artifact item.

    Values: ``adapter`` is the name of an adapter, ``api`` the API dialect of the model calls, and ``files`` maps
    each path of the tree to its text. ``env`` maps the name of each variable that the agent's commands get to its
    value; its token limits, when it sets them, are positive integers (``token_limits``), and
    ``REEF_AGENT_TIMEOUT_SECONDS``, when set, is a positive finite number of seconds (``agent_timeout_seconds``).
    Purpose: control reads it once when it accepts an evaluation, so that each rollout runs the same tree; a worker
    runs it.
    """

    adapter: str
    api: str
    files: Mapping[str, str]
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.adapter, "adapter")
        require_text(self.api, "api")
        if not isinstance(self.files, Mapping):
            raise TypeError("files is a mapping of paths to text")
        if not isinstance(self.env, Mapping) or not all(
            isinstance(name, str) and name and isinstance(value, str) for name, value in self.env.items()
        ):
            raise TypeError("env maps non-empty variable names to text")
        token_limits(self.env)
        agent_timeout_seconds(self.env)

    def policy_harness(self) -> PolicyHarness:
        """The harness side of the policy: the adapter, with the tree rendered with no model binding."""
        return PolicyHarness(self.adapter, render_tree(HarnessTree(self.files), get_adapter(self.adapter)))

    def agent(
        self, directory: Path, *, bridge: bool = False, task_directory: Path | None = None
    ) -> Mapping[str, object]:
        """The Harbor agent configuration that runs the tree, with its files under ``directory``. ``bridge``
        bridges the task container's loopback to the rollout's capture proxy (``tree_agent``). When ``env`` sets
        ``REEF_AGENT_TIMEOUT_SECONDS``, the cap overrides an unbounded task timeout or caps ``[agent].timeout_sec``
        in ``task_directory/task.toml``, propagating to Harbor's ``override_timeout_sec``, the command timeout and
        cloud sandbox lifetime."""
        tree_dir = directory / "tree"
        HarnessTree(self.files).write(tree_dir)
        cap = agent_timeout_seconds(self.env)
        if cap is None:
            command_timeout = TREE_AGENT_TIMEOUT_S
            override_timeout: float | None = None
        else:
            task_timeout = None if task_directory is None else task_agent_timeout_seconds(task_directory)
            override_timeout = cap if task_timeout is None else min(task_timeout, cap)
            command_timeout = float(math.ceil(override_timeout))
        config = tree_agent(
            get_adapter(self.adapter),
            api=self.api,
            tree_dir=tree_dir,
            files_dir=directory / "rendered",
            timeout=command_timeout,
            bridge=bridge,
            env=self.env,
        )
        if override_timeout is not None:
            return {**config, "override_timeout_sec": override_timeout}
        return config

    def to_json(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {"adapter": self.adapter, "api": self.api, "files": dict(self.files)}
        if self.env:
            value["env"] = dict(self.env)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> TreeHarness:
        # A rollout stored before a tree harness had an env has none.
        return cls(value["adapter"], value["api"], value["files"], value.get("env", {}))


@dataclass(frozen=True)
class EvaluationRollout:
    """The input of one ``evaluation_rollout`` item.

    Values: ``scenario`` and ``evaluation_id`` name the evaluation; ``artifact_id`` is the candidate artifact and
    ``component`` the component of the scenario's artifacts that it is;
    ``task_index`` is the task's position in the evaluation and ``task_name`` its name; ``repeat`` numbers the rollouts of one task from 0; ``model`` is the model that the
    service serves for the candidate; ``agent`` is the Harbor agent configuration, or None when ``tree`` is the
    harness; ``environment`` is the Harbor environment type; ``selection`` is the artifact selection that serves the
    candidate, as ``ArtifactSelection.to_dict`` gives it; ``tree`` is the harness tree, or None when ``agent``
    is the harness; ``log_probs`` is the evaluation's ``log_probs``. ``owner`` is, on ``e2b``, the owner that the
    scenario's record stored when control accepted the evaluation, a non-empty string; it is None when the record
    stored none, and on any other environment. With
    ``requires_prepared_templates``, the rollout runs a trial only from its task's prepared templates; only a rollout
    on ``e2b`` has it. The names are non-empty strings and the two numbers are non-negative integers. An item that an
    older Reef queued has neither field: its owner is None, and it does not require prepared templates.
    ``task_digest`` is the lowercase hex SHA-256 of the accepted task's files, or None for work without a pin.

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
    log_probs: bool = False
    owner: str | None = None
    requires_prepared_templates: bool = False
    task_digest: str | None = None

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
        if not isinstance(self.log_probs, bool):
            raise TypeError("log_probs is a bool")
        if self.owner is not None and (not isinstance(self.owner, str) or not self.owner):
            raise ValueError("owner is a non-empty string or None")
        if not isinstance(self.requires_prepared_templates, bool):
            raise TypeError("requires_prepared_templates is a bool")
        if self.requires_prepared_templates and self.environment != "e2b":
            raise ValueError("only a rollout on e2b requires prepared templates")
        if self.task_digest is not None and (
            not isinstance(self.task_digest, str) or FILES_DIGEST_PATTERN.fullmatch(self.task_digest) is None
        ):
            raise ValueError("task_digest must be a files digest")

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
                "log_probs": self.log_probs,
                "owner": self.owner,
                "requires_prepared_templates": self.requires_prepared_templates,
                **({"task_digest": self.task_digest} if self.task_digest is not None else {}),
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
            # A rollout that an older Reef queued asks for no log probabilities, names no owner, and does not require
            # prepared templates.
            payload.get("log_probs", False),
            payload.get("owner"),
            payload.get("requires_prepared_templates", False),
            payload.get("task_digest"),
        )


def rollout_item_id(evaluation_id: str, task_index: int, repeat: int) -> str:
    """The ID of the item of rollout ``repeat`` of task ``task_index`` of an evaluation."""
    return f"{EVALUATION_ROLLOUT_KIND}:{evaluation_id}:{task_index:05d}:{repeat:05d}"


def template_item_id(evaluation_id: str, task_index: int) -> str:
    """The ID of the template item of task ``task_index`` of an evaluation."""
    return f"{TASK_TEMPLATE_KIND}:{evaluation_id}:{task_index:05d}"


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
    call ends (``add_call``). A ``TaskTemplateWorker`` reads each task's directory and owner from it.
    """

    @abstractmethod
    def task_directory(self, rollout: EvaluationRollout) -> Path:
        """The Harbor task directory of ``rollout``'s task, on this process's file system."""

    def task_version(self, rollout: EvaluationRollout) -> tuple[Path, str | None]:
        """The resolved task directory and its stored digest, or None if unavailable for unpinned work.

        Template preparation and scoring both use this lookup. A lookup without a pin may select the
        current version on each attempt. A returned digest must identify the directory it resolved.
        An endpoint that resolves pinned work overrides this method.
        """
        if rollout.task_digest is not None:
            raise ItemFailedError("the task resolver does not support pinned task files")
        return self.task_directory(rollout), None

    def task_owner(self, rollout: EvaluationRollout) -> str | None:
        """The owner that the record of ``rollout``'s scenario stores now, whose data holds the task's item; None when
        the record stores none. These endpoints read no scenario record, so the owner is None; endpoints that read
        one return its owner."""
        return None

    @abstractmethod
    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        """The endpoint of ``rollout``'s candidate, until the block ends."""

    @abstractmethod
    def add_call(self, rollout: EvaluationRollout, call: EvaluationCall) -> None:
        """Keep ``call``, a call of ``rollout``, as a record item in the data store
        (``reef.storage.scenario.record_items.evaluation_call_item``)."""

    @abstractmethod
    def add_trajectory(self, rollout: EvaluationRollout, trajectory: EvaluationTrajectory) -> None:
        """Keep ``trajectory``, which the verifier of ``rollout`` wrote, as a trajectory item in the data store
        (``reef.storage.scenario.record_items.evaluation_trajectory_item``)."""


class LocalCandidateEndpoints(CandidateEndpoints):
    """The candidate's model, on loopback in the process of the evaluation worker: an ``ArtifactEndpoint`` that
    sends each call to the engine that inference started, or to its scenario's runtime for an engine with no address.

    Values: ``artifacts`` finds the candidate of each rollout. ``cooldowns`` holds the rate-limit waits that the
    endpoints of all its rollouts share. Purpose: the evaluation worker opens it around each rollout.
    """

    def __init__(self, artifacts: CandidateArtifacts) -> None:
        self.artifacts = artifacts
        self.cooldowns = ProviderCooldowns()

    def task_directory(self, rollout: EvaluationRollout) -> Path:
        if rollout.task_digest is None:
            return self.artifacts.candidate_task(rollout.scenario, rollout.task_name)
        return self.artifacts.candidate_task_for_digest(rollout.scenario, rollout.task_name, rollout.task_digest)

    def task_version(self, rollout: EvaluationRollout) -> tuple[Path, str | None]:
        if rollout.task_digest is None:
            if type(self).task_directory is not LocalCandidateEndpoints.task_directory:
                return self.task_directory(rollout), None
            return self.artifacts.candidate_task_version(rollout.scenario, rollout.task_name)
        for base in type(self).__mro__:
            if "task_version" in base.__dict__:
                break
            if "task_directory" in base.__dict__:
                raise ItemFailedError(
                    "a custom task directory resolver must override task_version for pinned task files"
                )
        task_directory = self.artifacts.candidate_task_for_digest(
            rollout.scenario, rollout.task_name, rollout.task_digest
        )
        return task_directory, rollout.task_digest

    def task_owner(self, rollout: EvaluationRollout) -> str | None:
        return self.artifacts.task_owner(rollout.scenario)

    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        runtime, artifact = self.artifacts.candidate(rollout.scenario, rollout.artifact_id, rollout.component)
        # Each rollout is its own conversation, also when the tasks of an evaluation start with the same messages.
        session_key = f"evaluation:{rollout.evaluation_id}:{rollout.task_index}:{rollout.repeat}"
        with ArtifactEndpoint(
            runtime, artifact, served, log_probs=rollout.log_probs, cooldowns=self.cooldowns, session_key=session_key
        ) as endpoint:
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
    no ``training`` block: it holds token IDs for training, which nobody reads in a call log. A body that is not a
    JSON object is kept as None.
    """
    request = turn.request if isinstance(turn.request, Mapping) else None
    response = turn.response if isinstance(turn.response, Mapping) else None
    prompt, completion, finish, tool_calls = (
        reply_summary(turn.path, response) if turn.status == 200 and response is not None else (None, None, None, 0)
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
    if response is not None and "training" in response:
        response = {key: value for key, value in response.items() if key != "training"}
    return EvaluationCall(summary, request, response)


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


@dataclass(frozen=True)
class ModelCallRefusal:
    """A model call of a rollout that a provider's limit refused, as its capture shows it.

    Values: ``limit`` is the ``UpstreamLimit`` of the refusal (``refusal_limit``), ``status`` the answer's HTTP status,
    an int from 400 to 599, and ``retry_after_seconds`` the answer's ``Retry-After``, an int of at least 0, or None.
    It holds no text of the answer. Purpose: ``RolloutCalls`` makes it for the last captured call, and
    ``EvaluationRolloutWorker.score`` fails an unscored rollout with its ``reason``.
    """

    limit: UpstreamLimit
    status: int
    retry_after_seconds: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.limit, UpstreamLimit):
            raise TypeError("limit is an UpstreamLimit")
        if isinstance(self.status, bool) or not isinstance(self.status, int) or not 400 <= self.status < 600:
            raise ValueError("status is an HTTP error status from 400 to 599")
        wait = self.retry_after_seconds
        if wait is not None and (isinstance(wait, bool) or not isinstance(wait, int) or wait < 0):
            raise ValueError("retry_after_seconds is a number of seconds of at least 0, or None")

    def reason(self) -> str:
        """The error of a rollout that this refusal left with no score, made from the fields alone."""
        text = f"the model provider refused the last model call with HTTP {self.status} ({self.limit.value})."
        if self.retry_after_seconds is not None:
            text += f" Retry after {self.retry_after_seconds} s."
        if self.limit is UpstreamLimit.PAYMENT and self.status == 403:
            text += " Check the provider's spending cap, or wait until it resets, before retrying."
        elif self.limit is UpstreamLimit.PAYMENT:
            text += " Check the provider balance or key limit, or reduce the request size before retrying."
        return text


class RolloutCalls(CaptureObserver):
    """Keeps each call of one attempt of an evaluation rollout in the data store, as the call ends.

    Values: ``endpoints`` keeps the calls (``CandidateEndpoints.add_call``). ``rollout`` is the rollout, and
    ``attempt`` the 1-based attempt of its work item. ``next_index`` is the number of calls captured so far, which is
    the index of the next call. ``all_answered`` counts calls that answered 200 and ``all_last_refusal`` holds the
    ``ModelCallRefusal`` of the call captured last, or None. ``tool_calls_seen`` is True once a call carried a
    non-empty ``tools`` field (``has_tools``); ``tool_answered`` and ``tool_last_refusal`` track those tool-bearing
    calls alone. ``answered`` and ``last_refusal`` read the tool-bearing counters when ``tool_calls_seen`` is True,
    else the all-call counters, so auxiliary calls without tools (such as title generation) neither mask a main-loop
    refusal nor fail a completed main loop. ``service_failures`` is the number of calls that the inference service
    failed (an HTTP status of 500 or more), ``service_failure_seconds`` their total duration, and ``last_failed``
    whether the call captured last was one. ``first_started_at`` and ``last_ended_at`` bound the calls in wall-clock
    seconds; both are None before the first.

    Purpose: ``EvaluationRolloutWorker.score`` gives one to the rollout's runner, and closes it when the rollout ends.
    One writer thread keeps the calls in call order, so the agent does not wait for the store. A call that the store
    does not keep is logged, and the rollout goes on.
    """

    def __init__(self, endpoints: CandidateEndpoints, rollout: EvaluationRollout, attempt: int) -> None:
        self.endpoints = endpoints
        self.rollout = rollout
        self.attempt = attempt
        self.next_index = 0
        self.all_answered = 0
        self.all_last_refusal: ModelCallRefusal | None = None
        self.tool_calls_seen = False
        self.tool_answered = 0
        self.tool_last_refusal: ModelCallRefusal | None = None
        self.service_failures = 0
        self.service_failure_seconds = 0.0
        self.last_failed = False
        self.first_started_at: float | None = None
        self.last_ended_at: float | None = None
        self.lock = threading.Lock()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-evaluation-calls")

    @property
    def answered(self) -> int:
        with self.lock:
            return self.tool_answered if self.tool_calls_seen else self.all_answered

    @property
    def last_refusal(self) -> ModelCallRefusal | None:
        with self.lock:
            return self.tool_last_refusal if self.tool_calls_seen else self.all_last_refusal

    def call_captured(self, turn: CapturedTurn) -> None:
        # The proxy keeps whatever JSON the endpoint answered; only an object can name a refusal.
        response = turn.response if isinstance(turn.response, Mapping) else None
        limit = refusal_limit(turn.status, response)
        error = response.get("error") if response is not None else None
        wait = error.get("retry_after_seconds") if isinstance(error, Mapping) else None
        if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0:
            wait = None
        refusal = None if limit is None else ModelCallRefusal(limit, turn.status, wait)
        is_tool_call = isinstance(turn.request, dict) and has_tools(turn.request) and bool(turn.request["tools"])
        with self.lock:
            index = self.next_index
            self.next_index += 1
            self.all_answered += turn.status == 200
            self.all_last_refusal = refusal
            if is_tool_call:
                self.tool_calls_seen = True
                self.tool_answered += turn.status == 200
                self.tool_last_refusal = refusal
            started = turn.at - turn.duration_s
            self.first_started_at = started if self.first_started_at is None else min(self.first_started_at, started)
            self.last_ended_at = turn.at if self.last_ended_at is None else max(self.last_ended_at, turn.at)
            self.last_failed = turn.status >= 500
            if self.last_failed:
                self.service_failures += 1
                self.service_failure_seconds += turn.duration_s
            self.writer.submit(self.keep, evaluation_call(turn, self.rollout, self.attempt, index))

    def service_failure(self) -> str | None:
        """The error of a rollout whose score would judge the inference service rather than the model, or None.

        That is a rollout whose last model call the service failed, or whose failed calls took at least
        ``SERVICE_FAILURE_SHARE`` of the time from its first call to the end of its last: the agent stopped on the
        failure, or it waited out much of its time limit for answers that did not come."""
        with self.lock:
            if not self.service_failures or self.first_started_at is None or self.last_ended_at is None:
                return None
            span = max(self.last_ended_at - self.first_started_at, 0.0)
            if not self.last_failed and self.service_failure_seconds < SERVICE_FAILURE_SHARE * span:
                return None
            last = ", the last one among them" if self.last_failed else ""
            return (
                f"the inference service failed {self.service_failures} of {self.next_index} model calls{last}, which "
                f"took {self.service_failure_seconds:.0f} s of {span:.0f} s of model calls; the rollout has no score"
            )

    def keep(self, call: EvaluationCall) -> None:
        try:
            self.endpoints.add_call(self.rollout, call)
        except Exception:
            # The call log is for people to read: a call that it misses must not change the rollout or its score.
            logger.exception("could not keep call %d of rollout %s", call.summary.index, self.rollout.item_id())

    def close(self) -> None:
        """Wait until every captured call is kept or logged."""
        self.writer.shutdown(wait=True)


def check_finish_timeout(on_stop: str, finish_timeout_s: object, *, prefix: str = "") -> None:
    """Raise ``ValueError`` unless ``finish_timeout_s`` is None, or ``on_stop`` is ``finish`` and ``finish_timeout_s``
    is an int or float, not a bool, above 0 and at most ``threading.TIMEOUT_MAX``: the longest wait of the timer that
    ends it. ``prefix`` starts the name of each setting in the message, such as ``evaluation.``."""
    if finish_timeout_s is None:
        return
    if on_stop != "finish":
        raise ValueError(f"{prefix}finish_timeout_s needs {prefix}on_stop: finish")
    if (
        isinstance(finish_timeout_s, bool)
        or not isinstance(finish_timeout_s, (int, float))
        # A comparison, not math.isfinite: it refuses NaN and infinity, and does not overflow on a large int.
        or not 0 < finish_timeout_s <= threading.TIMEOUT_MAX
    ):
        raise ValueError(
            f"{prefix}finish_timeout_s is a number of seconds above 0 and at most {threading.TIMEOUT_MAX:.0f}"
        )


Result = TypeVar("Result")


async def run_while_held(work: Awaitable[Result], lease_lost: threading.Event, stopped: threading.Event) -> Result:
    """Run ``work`` of a rollout and return its result. When ``lease_lost`` or ``stopped`` is set first, cancel
    ``work``, so that Harbor stops a trial and its sandbox, or a template build stops. Then raise
    ``WorkLeaseLostError`` for ``lease_lost``: a client cancelled the rollout, or another worker took it. Raise
    ``WorkerStoppedError`` for ``stopped`` alone: the worker stops, and the rollout goes back to the queue. Work that
    had already ended when ``stopped`` alone is seen keeps its result. A process that ends with a trial still running
    leaves the trial's sandbox running, so a stopping worker does not wait for the trial."""
    run = asyncio.ensure_future(work)
    while not lease_lost.is_set() and not stopped.is_set():
        done, _ = await asyncio.wait({run}, timeout=LEASE_CHECK_S)
        if done:
            return run.result()
    if run.done() and not run.cancelled() and not lease_lost.is_set():
        # The work ended before the stop could cancel it; the queue still fences the result to the lease holder.
        return run.result()
    run.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run
    if lease_lost.is_set():
        raise WorkLeaseLostError("the evaluation worker lost its lease")
    raise WorkerStoppedError


class EvaluationRolloutWorker(QueueWorker):
    """The worker of the evaluation service.

    Values: as ``QueueWorker``, on the rollouts of the evaluations queue, with one rollout in each slot;
    ``endpoints`` opens the candidate's model for each rollout; ``lab`` runs each Harbor trial, None for reef-eval's
    lab; ``trial_observer`` reads each trial before its directory is removed, None for none; ``agent_host`` is the
    address at which a task's container reaches this host. ``on_stop`` is one of ``ON_STOP_CHOICES``: what the worker
    does with the rollouts that run when it begins to stop (``begin_stop``). With ``requeue``, it cancels each one and
    gives it back to the queue. With ``finish``, it takes no new rollout, and ``stop`` returns when each running one
    has finished, so a process that a scale-down or a deploy stops loses no work in its grace period.
    ``finish_timeout_s`` is None, or, with ``finish`` only, a positive number of seconds: the time from the first
    ``begin_stop`` to the cancellation of the rollouts that still run (``cancel_rollouts``). Each of those goes back
    to the queue, as with ``requeue``; a rollout whose trial has ended keeps its result. A failed rollout is retried: a
    trial fails for reasons outside its input, such as a sandbox that did not start. A rollout of a harness tree whose
    task needs what the tree's agent does not give it (``task_agent_tools``) fails for good, and so does a rollout
    that requires prepared templates when its task's template item did not finish, one whose sandbox request the
    sandbox service refused as invalid, and one that a provider's limit left with no score (``score``). A rollout
    whose sandbox found no capacity goes back to the queue and waits (``ItemDeferredError``), and the worker goes on
    with other rollouts.

    Purpose: the evaluation service starts one (``start``), lets it begin to stop as soon as the process begins to
    stop (``begin_stop``), and stops it (``stop``).
    """

    permanent_errors = ()

    def __init__(
        self,
        queue: WorkQueue,
        endpoints: CandidateEndpoints,
        *,
        worker_id: str,
        lab: TaskLab | None = None,
        trial_observer: TrialObserver | None = None,
        agent_host: str = DEFAULT_AGENT_HOST,
        poll_seconds: float = 0.5,
        slots: int = 1,
        on_stop: str = "requeue",
        finish_timeout_s: float | None = None,
    ) -> None:
        if on_stop not in ON_STOP_CHOICES:
            raise ValueError(f"on_stop must be one of {', '.join(ON_STOP_CHOICES)}, not {on_stop!r}")
        check_finish_timeout(on_stop, finish_timeout_s)
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
        self.trial_observer = trial_observer
        self.agent_host = agent_host
        self.on_stop = on_stop
        self.finish_timeout_s = finish_timeout_s
        #: The event that cancels the running rollouts: the worker's stop with ``requeue``; with ``finish``, the end of
        #: ``finish_timeout_s``, and never without it.
        self.rollouts_cancelled = self.stopped if on_stop == "requeue" else threading.Event()
        #: The timer that ends ``finish_timeout_s``, from the first ``begin_stop`` until ``stop`` returns; else None.
        self.cancel_timer: threading.Timer | None = None
        #: Makes ``begin_stop`` one step, so that two calls start one timer.
        self.stop_lock = threading.Lock()

    def start(self) -> None:
        if self.thread is None:
            # A worker that starts again after a stop runs its rollouts to the end again.
            self.rollouts_cancelled.clear()
        super().start()

    def begin_stop(self) -> None:
        """Take no new rollout. With ``requeue``, cancel the running rollouts now. With ``finish_timeout_s``, start a
        timer that cancels the rollouts that still run that many seconds after the first call (``cancel_rollouts``).
        A later call does not move that time."""
        with self.stop_lock:
            if self.stopped.is_set():
                return
            self.stopped.set()
            if self.finish_timeout_s is None:
                return
            timer = threading.Timer(self.finish_timeout_s, self.cancel_rollouts)
            timer.daemon = True
            timer.start()
            # Only a started timer is kept, so that ``stop`` joins a thread that runs.
            self.cancel_timer = timer
        logger.info(
            "evaluation worker %s takes no new rollout; it cancels the rollouts that still run in %.0f s",
            self.worker_id,
            self.finish_timeout_s,
        )

    def cancel_rollouts(self) -> None:
        """Cancel the rollouts that still run at the end of ``finish_timeout_s``. Each one stops its trial and goes
        back to the queue (``run_while_held``), and another worker starts it again."""
        logger.warning(
            "evaluation worker %s cancels the rollouts that still run %.0f s after its stop began",
            self.worker_id,
            self.finish_timeout_s,
        )
        self.rollouts_cancelled.set()

    def stop(self) -> None:
        """Begin to stop (``begin_stop``), wait until each running rollout has finished or gone back to the queue, and
        end the timer of ``finish_timeout_s``."""
        self.begin_stop()
        super().stop()
        with self.stop_lock:
            timer, self.cancel_timer = self.cancel_timer, None
        if timer is not None:
            timer.cancel()
            timer.join()

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        score, rewards, error = self.score(EvaluationRollout.from_item(item), lease_lost, item.attempt_count)
        return {"score": score.to_json(), "rewards": rewards, "error": error}

    def served_candidate(self, rollout: EvaluationRollout, lease_lost: threading.Event) -> ServedArtifact | None:
        """Ask inference to serve the candidate of ``rollout``, wait until it is ready, and return the value that
        reaches it: None for a runtime with no address. Raise ``ReefError`` when inference fails or takes too long, so
        that the rollout fails and does not score 0.

        The serve items of one candidate run one at a time, in order: the rollouts of an evaluation wait for one start
        and do not fill each serving slot of inference with the same start. A rollout that stops waiting cancels its
        item, so that inference does not start the candidate for a rollout that is gone."""
        item_id = f"{SERVE_ARTIFACT_KIND}:{uuid.uuid4().hex}"
        payload: dict[str, JsonValue] = {"selection": dict(rollout.selection), "component": rollout.component}
        serial_key = f"{SERVE_ARTIFACT_KIND}:{rollout.scenario}:{rollout.artifact_id}"
        self.queue.append(
            NewWork(SERVING_QUEUE, item_id, rollout.scenario, SERVE_ARTIFACT_KIND, payload, serial_key=serial_key)
        )
        deadline = time.monotonic() + SERVE_WAIT_S
        try:
            while time.monotonic() < deadline:
                if self.rollouts_cancelled.is_set():
                    raise WorkerStoppedError
                if lease_lost.is_set():
                    raise WorkLeaseLostError("the evaluation worker lost its lease")
                item = self.queue.read(SERVING_QUEUE, item_id)
                if item is not None and item.final:
                    if item.state != "finished":
                        raise ReefError(f"inference could not serve artifact {rollout.artifact_id}: {item.error}")
                    served = cast(Mapping[str, Any], item.result)["served"]
                    return None if served is None else ServedArtifact.from_json(served)
                time.sleep(SERVE_POLL_S)
            raise ReefError(f"inference did not serve artifact {rollout.artifact_id} in {SERVE_WAIT_S:.0f} s")
        except (WorkerStoppedError, WorkLeaseLostError, ReefError):
            # A final item stays as it is.
            self.queue.cancel(SERVING_QUEUE, item_id, "the evaluation rollout stopped waiting for the candidate")
            raise

    def score(
        self, rollout: EvaluationRollout, lease_lost: threading.Event, attempt: int
    ) -> tuple[RolloutScore, dict[str, JsonValue] | None, str | None]:
        """Run ``rollout`` once, when its candidate is ready, and return its score, the rewards of its verifier
        (``verifier_rewards``), and the reason of a failed score (None for a scored rollout). ``attempt`` is the
        1-based attempt of the rollout's work item; the rollout's calls are kept under it (``RolloutCalls``).

        A rollout that requires prepared templates runs only when its task's template item finished, and its trial
        builds no template. Else it fails for good with the template item's error, before its candidate is served.

        A sandbox capacity refusal raises ``ItemDeferredError``. An invalid sandbox request fails the rollout.
        Both leave the score unset, even if the agent already called the model.

        When a provider's limit refused the last model call (``RolloutCalls.last_refusal``), or refused a trailing
        no-tools call on a trial that failed with an error (``RolloutCalls.all_last_refusal``), the rollout fails for
        good with no score (``ModelCallRefusal.reason``). When a rollout makes tool-bearing calls, auxiliary calls
        without tools do not mask a main-loop refusal and do not fail a trial that finished with no error. When the
        inference service failed its model calls (``RolloutCalls.service_failure``), the attempt fails with no score,
        and the rollout is retried as any other failed attempt."""
        if rollout.requires_prepared_templates:
            template = self.queue.read(EVALUATIONS_QUEUE, template_item_id(rollout.evaluation_id, rollout.task_index))
            unprepared = f"the templates of task {rollout.task_name!r} were not prepared"
            if template is None:
                raise ItemFailedError(f"{unprepared}: the queue no longer holds its template item")
            if template.state != "finished":
                raise ItemFailedError(f"{unprepared}: {template.error or f'its template item is {template.state}'}")
        task_directory, task_digest = self.endpoints.task_version(rollout)
        if rollout.task_digest is not None and task_digest != rollout.task_digest:
            raise ItemFailedError("the task resolver returned different files from the pinned version")
        if rollout.tree is not None:
            # A retry runs the same task with the same tree, so a task that asks for what the tree's agent cannot
            # give fails for good, before its candidate is served.
            tools = task_agent_tools(get_adapter(rollout.tree.adapter), task_directory)
            if tools:
                raise ItemFailedError(
                    f"task {rollout.task_name!r} needs {', '.join(tools)}, which adapter {rollout.tree.adapter!r} "
                    "does not give its agent"
                )
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
                agent=(
                    rollout.tree.agent(Path(work), bridge=bridged, task_directory=task_directory)
                    if rollout.tree is not None
                    else rollout.agent
                ),
                # A tree's agent on Docker runs in the task's container, which reaches the capture proxy at this address.
                agent_host=None if rollout.tree is None or bridged else self.agent_host,
                environment=rollout.environment,
                is_reporting=False,
                is_unrecorded=True,
                concurrency=1,
                lab=self.lab,
                capture_observer=calls,
                # ``calls`` keeps each call as it lands, so the runner keeps no call body: an agent's request holds
                # the whole conversation so far, and a long rollout's bodies reached hundreds of MiB.
                keeps_turns=False,
                # A template that was removed after its preparation fails the trial; the trial does not build it.
                builds_templates=not rollout.requires_prepared_templates,
                # The agent gets the engine's own stream, as it would from a provider, so nothing that the engine
                # answers is lost on the way, such as a reasoning model's reasoning. A call that asks for log
                # probabilities stays whole: a stream gives its token IDs in pieces.
                streams_calls=not rollout.log_probs,
            )
            request = RolloutRequest(runner.harness, runner.policy_model, str(task_directory))
            try:
                [result] = asyncio.run(
                    run_while_held(
                        runner.run_rollouts_async([request], asyncio.Semaphore(1)),
                        lease_lost,
                        self.rollouts_cancelled,
                    )
                )
            finally:
                calls.close()
            # The trial directory is under ``work``, so read it before the block removes it.
            trial = trial_directory(result.trial_uri)
            self.keep_verifier_output(rollout, attempt, trial)
            # A trial that the sandbox service refused for want of capacity waits in the queue for capacity and runs
            # again, and the attempt does not count. That holds also after the agent called the model: a separate
            # verifier that found no sandbox gave no score of the candidate. A sandbox request that the service
            # refused as invalid fails the same way again, so the rollout fails for good, with the service's reason
            # and no score.
            exception_type = trial_exception_type(result.trial_uri)
            self.observe_trial(
                rollout,
                EvaluationTrial(
                    rollout.scenario,
                    rollout.evaluation_id,
                    rollout.task_index,
                    rollout.task_name,
                    rollout.repeat,
                    attempt,
                    trial,
                    result.error,
                    deferred=exception_type in SANDBOX_CAPACITY_ERRORS,
                ),
            )
            if exception_type in SANDBOX_CAPACITY_ERRORS:
                raise ItemDeferredError(f"waiting for sandbox capacity: {result.error}")
            if exception_type in SANDBOX_REQUEST_ERRORS:
                raise ItemFailedError(f"the sandbox service refused the sandbox: {result.error}")
            last_refusal = calls.last_refusal
            if last_refusal is None and result.error:
                # A required tool-less agent step (such as a native sub-agent with ``tools: []``) still fails the
                # trial with ``result.error``; keep its terminal refusal rather than scoring 0.
                last_refusal = calls.all_last_refusal
            if last_refusal is not None:
                # The agent stopped on a provider refusal, so a score, 0 above all, would judge the provider, and a
                # new sandbox at once would meet the same refusal.
                raise ItemFailedError(last_refusal.reason())
            service_failure = calls.service_failure()
            if service_failure is not None:
                # The agent stopped on a failed call or lost much of its time to them, so a score, 0 above all, would
                # judge the service. Another attempt can find the service well again.
                raise ReefError(service_failure)
            if not calls.answered:
                # A verifier can write a reward with no call, so that reward says nothing of the candidate's model. A
                # trial whose sandbox did not start makes no call either, so the attempt fails with the reason, and
                # the rollout is retried as any other failed attempt.
                raise ReefError(f"the rollout reached no model: {result.error or 'the agent made no model call'}")
            score = runner.rollout_score(result)
            if score.is_failed:
                # The rollout's item finishes with this score, so the item has no error; keep the reason in the result.
                return score, None, result.error
            return score, verifier_rewards(result.rewards), None

    def observe_trial(self, rollout: EvaluationRollout, trial: EvaluationTrial) -> None:
        """Give ``trial`` of ``rollout`` to the trial observer, when there is one. Its failure is logged, and the
        rollout and its score do not change."""
        if self.trial_observer is None:
            return
        try:
            self.trial_observer.observe(trial)
        except Exception:
            logger.exception("the trial observer failed on rollout %s", rollout.item_id())

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


class TaskTemplateWorker(QueueWorker):
    """The worker that builds the sandbox templates of the tasks of evaluations on e2b.

    Values: as ``QueueWorker``, on the template items of the evaluations queue (``TASK_TEMPLATE_KIND``), with one
    item in each slot. An item's payload is the payload of the task's first rollout (``EvaluationRollout``).
    ``endpoints`` gives each task's directory and the owner that the scenario's record stores now, and ``builder``
    makes the task's templates ready for that owner (``TaskTemplateBuilder.build_for_version``). An item whose payload
    names an owner fails for good, with no build, when the record stores another owner or none. An item that an older
    Reef queued names no owner, so the record's owner alone counts. Every failure of a build is retryable, until the
    item uses up its attempts. Control adds a task's rollouts once its item is final, also when the build failed. The
    trial then builds the template itself, and its failure is the rollout's failure; but a rollout that requires
    prepared templates fails with the item's error, and runs no trial.

    Purpose: the evaluation service starts one beside its ``EvaluationRolloutWorker``. A wait for a template builder
    then holds a slot of this worker, not a rollout slot of the evaluation or a sandbox slot.
    """

    permanent_errors = ()

    def __init__(
        self,
        queue: WorkQueue,
        endpoints: CandidateEndpoints,
        builder: TaskTemplateBuilder,
        *,
        worker_id: str,
        poll_seconds: float = 0.5,
        slots: int = 1,
    ) -> None:
        super().__init__(
            queue,
            EVALUATIONS_QUEUE,
            [TASK_TEMPLATE_KIND],
            worker_id=worker_id,
            poll_seconds=poll_seconds,
            slots=slots,
        )
        self.endpoints = endpoints
        self.builder = builder

    def begin_stop(self) -> None:
        """Take no new template item, and cancel the running builds now: each one goes back to the queue."""
        self.stopped.set()

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        rollout = EvaluationRollout.from_item(item)
        owner = self.endpoints.task_owner(rollout)
        if rollout.owner is not None and owner != rollout.owner:
            # The scenario was replaced by one of another owner, or removed: its templates are not this item's.
            found = "no owner" if owner is None else f"the owner {owner!r}"
            raise ItemFailedError(
                f"scenario {rollout.scenario!r} has {found} now, not {rollout.owner!r}, the owner of the evaluation's "
                f"tasks: the templates of task {rollout.task_name!r} are not built for another owner"
            )
        if rollout.task_digest is not None and not self.builder.supports_pinned_versions():
            raise ItemFailedError("the template builder does not support pinned task files")
        task_directory, task_digest = self.endpoints.task_version(rollout)
        if rollout.task_digest is not None and task_digest != rollout.task_digest:
            raise ItemFailedError("the task resolver returned different files from the pinned version")
        asyncio.run(
            run_while_held(
                self.build_task_template(owner, rollout, task_directory, task_digest),
                lease_lost,
                self.stopped,
            )
        )
        return {}

    async def build_task_template(
        self, owner: str | None, rollout: EvaluationRollout, task_directory: Path, task_digest: str | None
    ) -> None:
        """Make the e2b templates of the task of ``rollout``, at ``task_directory``, ready for ``owner`` with
        ``builder``. A build that found no free builder in the task's build time limit (``TimeoutError``) waits again,
        at most ``TEMPLATE_WAIT_S`` in all. Then raise ``ReefError``, so the attempt fails and the item is retried.
        Any other failure of the build fails the attempt in the same way."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TEMPLATE_WAIT_S
        while True:
            try:
                await self.builder.build_for_version(
                    owner, rollout.scenario, rollout.task_name, task_directory, task_digest=task_digest
                )
                return
            except TimeoutError as error:
                if loop.time() >= deadline:
                    raise ReefError(
                        f"the template of task {rollout.task_name!r} was not ready after {TEMPLATE_WAIT_S:.0f} s: "
                        f"{error}"
                    ) from error
                logger.info("the template of task %r is not ready; waiting again: %s", rollout.task_name, error)


__all__ = [
    "AGENT_TIMEOUT_ENV",
    "EVALUATIONS_QUEUE",
    "EVALUATION_ROLLOUT_KIND",
    "MAX_VERIFIER_BYTES",
    "MAX_VERIFIER_CALLS",
    "TASK_TEMPLATE_KIND",
    "VERIFIER_CALLS_FILE",
    "VERIFIER_JUDGE_DIR",
    "VERIFIER_TRAJECTORY_FILES",
    "CandidateEndpoint",
    "CandidateEndpoints",
    "EvaluationRollout",
    "EvaluationRolloutWorker",
    "LocalCandidateEndpoints",
    "ModelCallRefusal",
    "RolloutCalls",
    "TaskTemplateWorker",
    "TreeHarness",
    "agent_harness",
    "agent_timeout_seconds",
    "check_finish_timeout",
    "evaluation_call",
    "harbor_agent",
    "reply_summary",
    "rollout_item_id",
    "template_item_id",
    "verifier_call",
    "verifier_output",
]
