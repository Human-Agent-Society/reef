"""A training component's own rollouts of a task job: Harbor rollouts sampled at the job's start weights.

A task job holds ``TaskSample`` values. Before it trains, the component runs ``rollouts_per_task`` rollouts of
each task with ``TaskRolloutRunner``: the agent works in the task's sandbox, the task's verifier scores the
rollout, and each model call goes through the runner's capture proxy to a ``ChatEndpoint`` on this host. The
endpoint answers each call with the component's ``RolloutChat``, which samples the weights that the job trains
and returns the training capture. Each rollout that ran assembles into one sample; the samples of one task form
its group, which trains only with at least two samples. Reef's serving takes no part.

Tinker answers on its sampler (``reef.train.tinker_backend.rollout``) and Slime on its own rollout engines
(``reef.train.slime_backend.rollout_group``).
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reef.core import AgentRecord, RequestType
from reef.core.evaluation import RolloutRequest, RolloutScores
from reef.core.job_data import TaskSample, TrajectorySample
from reef.harness.client.tasks import TaskLab, TaskRollout, TaskRolloutRunner
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.runtime.interfaces import inference_aborted

logger = logging.getLogger(__name__)

#: The scenario name that rollout calls carry; the endpoint serves no scenario and ignores it.
ROLLOUT_SCENARIO = "task-rollout"
#: Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response.
REALIGN_THRESHOLD = 32768
#: The OpenAI chat route that an agent's model calls take; the rollout chats answer it.
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


@dataclass(frozen=True)
class RolloutSettings:
    """A task job's ``rollout`` parameter: how the component samples the rollouts of the job's tasks.

    Values: ``rollouts_per_task`` is an int of at least 2, the group of one task. ``agent`` is a Harbor
    agent config with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders, or None for
    ``TaskRolloutRunner``'s default agent. ``environment`` is a non-empty Harbor environment type, such as
    ``docker``, ``e2b`` or ``modal``. ``agent_host`` is the host at which an agent in the sandbox reaches this host, or
    None for an agent on this host. ``concurrency`` is a positive int, the rollouts in flight.
    ``verifier_timeout_reward`` is the reward of a rollout whose verifier timed out, or None to leave it
    unscored. ``scaffold_tolerance`` is a non-negative int, the masked tokens that the assembly may realign
    before a response (for a thinking template's scaffold). Purpose: ``from_parameters`` reads it from a
    job's parameters; ``roll_out`` runs it.
    """

    rollouts_per_task: int
    agent: Mapping[str, Any] | None = None
    environment: str = "docker"
    agent_host: str | None = None
    concurrency: int = 2
    verifier_timeout_reward: float | None = 0.0
    scaffold_tolerance: int = 8

    def __post_init__(self) -> None:
        for name, value, least in (
            ("rollouts_per_task", self.rollouts_per_task, 2),
            ("concurrency", self.concurrency, 1),
            ("scaffold_tolerance", self.scaffold_tolerance, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"rollout.{name} must be an int of at least {least}, got {value!r}")
        if self.agent is not None and not isinstance(self.agent, Mapping):
            raise ValueError("rollout.agent must be a Harbor agent config object")
        if not isinstance(self.environment, str) or not self.environment:
            raise ValueError("rollout.environment must be a non-empty Harbor environment type")
        if self.agent_host is not None and (not isinstance(self.agent_host, str) or not self.agent_host):
            raise ValueError("rollout.agent_host must be a non-empty host or None")
        reward = self.verifier_timeout_reward
        if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
            raise ValueError("rollout.verifier_timeout_reward must be a number or None")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> RolloutSettings:
        """The ``rollout`` parameter of a task job; raise ``ValueError`` when it is missing or malformed."""
        value = parameters.get("rollout")
        if not isinstance(value, Mapping):
            raise ValueError("a task job needs the rollout parameter as an object")
        names = {name.replace("-", "_"): item for name, item in value.items()}
        unknown = set(names) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"the rollout parameter takes no {sorted(unknown)}")
        try:
            return cls(**names)
        except TypeError as exc:
            raise ValueError(f"the rollout parameter is incomplete: {exc}") from exc


@dataclass(frozen=True)
class TaskJobRollouts:
    """The rollouts of one task job.

    Values: ``samples`` holds the samples of the groups that train, each group contiguous, every sample with
    the ``group_id`` of its task. ``scores`` holds the score of each rollout, by task name, in run order; each
    group names a task of ``scores``. Purpose: ``roll_out`` makes it; the component trains ``samples`` and
    reports the scores in the job's metrics.
    """

    samples: tuple[TrajectorySample, ...]
    scores: RolloutScores

    def __post_init__(self) -> None:
        if any(not isinstance(sample, TrajectorySample) for sample in self.samples):
            raise TypeError("task rollouts train trajectories")
        if not isinstance(self.scores, RolloutScores):
            raise TypeError("TaskJobRollouts.scores must be RolloutScores")
        names = {task for task, _ in self.scores.tasks}
        groups = [sample.group_id for sample in self.samples]
        seen: list[str | None] = []
        for group in groups:
            if not seen or seen[-1] != group:
                if group in seen or group not in names:
                    raise ValueError("each task's samples form one contiguous group named by a task of the scores")
                seen.append(group)
        if any(groups.count(group) < 2 for group in seen):
            raise ValueError("a task's group trains only with two or more samples")

    def rewards(self) -> dict[str, tuple[float, ...]]:
        """Each task's rewards of the rollouts that did not fail, in run order."""
        return {
            task: tuple(score.value for score in scores if score.value is not None)
            for task, scores in self.scores.tasks
        }


class RolloutChat(ABC):
    """What answers the model calls of one task job's rollouts, at the weights that the job trains.

    ``answer`` maps one request, its ``path`` (such as ``/v1/chat/completions``) and its decoded JSON body
    ``payload``, to a status and a JSON body. A body of status 200 is the provider's response with the private
    ``training`` capture that the sample assembly reads; any other status is an error body. ``answer`` may run
    on several threads at once. Purpose: a training component makes one per job; ``ChatEndpoint`` serves it.
    """

    @abstractmethod
    def answer(self, path: str, payload: Any) -> tuple[int, Mapping[str, Any]]: ...


class ChatEndpoint:
    """An HTTP endpoint on loopback that answers every POST with one ``RolloutChat``, for the rollouts of one job.

    A request that is not JSON gets 400. The server runs on its own threads, so an agent that blocks the
    rollout's event loop cannot block it. Use it as a context manager: it binds its loopback port and sets
    ``url`` when it is entered, and closes the port when it exits.
    """

    def __init__(self, chat: RolloutChat) -> None:
        self.chat = chat
        self.server: ThreadingHTTPServer | None = None
        self.url = ""

    def __enter__(self) -> ChatEndpoint:
        chat = self.chat

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except json.JSONDecodeError as exc:
                    self.reply(400, {"error": {"message": f"the request is not JSON: {exc}"}})
                    return
                self.reply(*chat.answer(self.path, payload))

            def reply(self, status: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="task-rollout-chat", daemon=True).start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def task_name(task: TaskSample) -> str:
    """The name of a task: its generated name, else its directory's name."""
    identity = task.metadata.get("task")
    if isinstance(identity, Mapping) and isinstance(identity.get("name"), str) and identity["name"]:
        return identity["name"]
    return task.task_path.name


def rollout_sample(rollout: TaskRollout, group: str, assembly: SampleAssembly) -> TrajectorySample | None:
    """The sample of one rollout in ``group``, or None when it has no calls, an aborted call, or does not assemble.

    A rollout that ran without a reward scores 0.
    """
    if not rollout.turns or any(inference_aborted(turn.response) for turn in rollout.turns):
        return None
    inferences = tuple(
        AgentRecord.create(
            scenario=ROLLOUT_SCENARIO,
            request_type=RequestType.INFERENCE,
            payload={**turn.request, "response": dict(turn.response)},
        )
        for turn in rollout.turns
    )
    try:
        sample = assembly.assemble(inferences, rollout.reward or 0.0, source_id=rollout.rollout_id)
    except ValueError:
        logger.warning("rollout %s of task %s does not assemble into a sample", rollout.rollout_id, group)
        return None
    return replace(sample, group_id=group, source_agent_record_ids=())


def roll_out(
    tasks: Sequence[TaskSample],
    settings: RolloutSettings,
    chat: RolloutChat,
    *,
    model: str,
    work_dir: Path,
    model_adapter: str | None = None,
    lab: TaskLab | None = None,
) -> TaskJobRollouts:
    """Run ``settings.rollouts_per_task`` rollouts of each task, answering their model calls with ``chat``.

    ``model`` is the model name the agent sends, and ``model_adapter`` the adapter it names, or None.
    ``lab`` runs each rollout; None is reef-eval's Harbor lab under ``work_dir``.
    """
    assembly = SampleAssembly(
        SampleAssemblySettings(
            accept_multi_turn_policy_samples=True,
            realign_threshold=REALIGN_THRESHOLD,
            scaffold_tolerance=settings.scaffold_tolerance,
        )
    )
    with ChatEndpoint(chat) as endpoint:
        runner = TaskRolloutRunner(
            reef_url=endpoint.url,
            scenario=ROLLOUT_SCENARIO,
            model=model,
            work_dir=work_dir,
            agent=settings.agent,
            agent_host=settings.agent_host,
            environment=settings.environment,
            is_reporting=False,
            is_unrecorded=True,
            verifier_timeout_reward=settings.verifier_timeout_reward,
            lab=lab,
            concurrency=settings.concurrency,
            model_adapter=model_adapter,
        )
        rollouts = runner.run_rollouts(
            [
                RolloutRequest(runner.harness, runner.served_model, str(task.task_path))
                for task in tasks
                for _ in range(settings.rollouts_per_task)
            ]
        )
    samples: list[TrajectorySample] = []
    for index, task in enumerate(tasks):
        name = task_name(task)
        ran = [
            rollout
            for rollout in rollouts[index * settings.rollouts_per_task : (index + 1) * settings.rollouts_per_task]
            if not rollout.is_infrastructure_failure
        ]
        group = [sample for sample in (rollout_sample(rollout, name, assembly) for rollout in ran) if sample]
        # A group of fewer than two samples has no relative reward, so it does not train.
        if len(group) >= 2:
            samples.extend(group)
    scores = RolloutScores.by_task(
        [task_name(task) for task in tasks], [runner.rollout_score(rollout) for rollout in rollouts]
    )
    return TaskJobRollouts(tuple(samples), scores)
