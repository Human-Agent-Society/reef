"""A training component's own Harbor rollouts of a task job's tasks.

A task job holds ``TaskSample`` values. For each rollout, ``TaskRolloutRunner`` runs the agent in the task's
sandbox, and the task's verifier scores the rollout. Each model call goes through the runner's capture proxy to a
``ChatEndpoint`` on this host. The endpoint answers each call with the component's ``RolloutChat``, which samples
the weights that the job trains and returns the training capture. Each rollout that ran assembles into one sample
(``rollout_sample``). Reef's serving takes no part.

Tinker runs ``rollouts_per_task`` rollouts of each task before it trains (``run_train_tasks``), and answers on its
sampler (``reef.train.tinker_backend.rollout``). On Slime, each rollout is one generate step of Slime's own training
loop (``reef.train.slime_backend.harbor_rollouts``).
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reef.core import AgentRecord, RequestType
from reef.core.errors import inference_aborted
from reef.core.evaluation import RolloutRequest, RolloutScores
from reef.core.job_data import TaskSample, TrajectorySample
from reef.harness.harbor.rollouts import TaskLab, TaskRollout, TaskRolloutRunner
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.samples import recorded_training

logger = logging.getLogger(__name__)

#: The scenario name that rollout calls carry; the endpoint serves no scenario and ignores it.
ROLLOUT_SCENARIO = "task-rollout"
#: Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response.
REALIGN_THRESHOLD = 32768
#: The OpenAI chat route that an agent's model calls take; the rollout chats answer it.
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


@dataclass(frozen=True)
class HarborSettings:
    """How a training component runs Harbor rollouts of a task job's tasks.

    Values: ``agent`` is a Harbor agent config with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders, or None
    for ``TaskRolloutRunner``'s default agent. ``environment`` is a non-empty Harbor environment type, such as
    ``docker``, ``e2b`` or ``modal``. ``agent_host`` is the host at which an agent in the sandbox reaches the host that
    runs the rollouts, or None for an agent on that host. ``concurrency`` is a positive int, the rollouts in flight.
    ``verifier_timeout_reward`` is the reward of a rollout whose verifier timed out, or None to leave it unscored.
    ``scaffold_tolerance`` is a non-negative int, the most tokens at the end of a prompt that the next prompt's history
    may drop (a thinking template's scaffold) while the response after them still trains. Purpose: a training
    component makes it from its own job parameters, and reads it where it runs the rollouts.
    """

    agent: Mapping[str, Any] | None = None
    environment: str = "docker"
    agent_host: str | None = None
    concurrency: int = 2
    verifier_timeout_reward: float | None = 0.0
    scaffold_tolerance: int = 8

    def __post_init__(self) -> None:
        for name, value, least in (
            ("concurrency", self.concurrency, 1),
            ("scaffold_tolerance", self.scaffold_tolerance, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{name} must be an int of at least {least}, got {value!r}")
        if self.agent is not None and not isinstance(self.agent, Mapping):
            raise ValueError("agent must be a Harbor agent config object")
        if not isinstance(self.environment, str) or not self.environment:
            raise ValueError("environment must be a non-empty Harbor environment type")
        if self.agent_host is not None and (not isinstance(self.agent_host, str) or not self.agent_host):
            raise ValueError("agent_host must be a non-empty host or None")
        reward = self.verifier_timeout_reward
        if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
            raise ValueError("verifier_timeout_reward must be a number or None")


@dataclass(frozen=True)
class RolloutTokens:
    """The response tokens of one task rollout that ran, and how many of them train.

    Values: ``task`` is the name of the rollout's task. ``response_tokens`` is a non-negative int, the tokens that the
    rollout's calls sampled (the length of each call's loss mask). ``trained_tokens`` is an int from 0 to
    ``response_tokens``, the tokens of them that the job trains (the sum of the loss mask of the rollout's sample):
    fewer when the assembly masks a response that a later prompt rewrote, and 0 when the rollout has no sample or its
    task's group does not train. ``assembled`` holds exactly when the rollout's calls assemble into a sample
    (``rollout_sample``); ``trained_tokens`` is 0 when it does not. Purpose: ``run_train_tasks`` makes one for each
    rollout that ran; ``TaskJobRollouts.metrics`` reports them in the job's metrics, and a training component keeps
    them with the job's rollouts.
    """

    task: str
    response_tokens: int
    trained_tokens: int
    assembled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("RolloutTokens.task must be a non-empty task name")
        counts = (self.response_tokens, self.trained_tokens)
        if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
            raise TypeError("RolloutTokens counts tokens as ints")
        if not isinstance(self.assembled, bool):
            raise TypeError("RolloutTokens.assembled is a bool")
        if not 0 <= self.trained_tokens <= self.response_tokens:
            raise ValueError("RolloutTokens.trained_tokens must be from 0 to response_tokens")
        if not self.assembled and self.trained_tokens:
            raise ValueError("a rollout that does not assemble trains no tokens")


@dataclass(frozen=True)
class RolloutSettings(HarborSettings):
    """``HarborSettings`` of a component that runs a fixed number of rollouts of each task before it trains.

    Values: every value of ``HarborSettings`` with ``rollouts_per_task``, an int of at least 2, the group of one task.
    Purpose: Tinker makes it from its ``rollout`` job parameter; ``run_train_tasks`` runs it.
    """

    rollouts_per_task: int = field(kw_only=True)

    def __post_init__(self) -> None:
        count = self.rollouts_per_task
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(f"rollouts_per_task must be an int of at least 2, got {count!r}")
        super().__post_init__()


@dataclass(frozen=True)
class TaskJobRollouts:
    """The rollouts of one task job.

    Values: ``samples`` holds the samples of the groups that train, each group contiguous, every sample with the
    ``group_id`` of its task. ``scores`` holds the score of each rollout, by task name, in run order; each group names
    a task of ``scores``. ``tokens`` holds the ``RolloutTokens`` of each rollout that ran, in the order of ``scores``.
    ``left_out`` is a non-negative int, the samples left out because they are longer than the job's ``max_tokens``.
    Purpose: ``run_train_tasks`` makes it; the component trains ``samples`` and reports ``metrics`` in the job's
    metrics.
    """

    samples: tuple[TrajectorySample, ...]
    scores: RolloutScores
    tokens: tuple[RolloutTokens, ...]
    left_out: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.left_out, bool) or not isinstance(self.left_out, int) or self.left_out < 0:
            raise ValueError("TaskJobRollouts.left_out must be a non-negative int")
        if any(not isinstance(sample, TrajectorySample) for sample in self.samples):
            raise TypeError("task rollouts train trajectories")
        if not isinstance(self.scores, RolloutScores):
            raise TypeError("TaskJobRollouts.scores must be RolloutScores")
        if any(not isinstance(item, RolloutTokens) for item in self.tokens):
            raise TypeError("TaskJobRollouts.tokens must hold RolloutTokens")
        ran = [task for task, scores in self.scores.tasks for score in scores if not score.is_failed]
        if [item.task for item in self.tokens] != ran:
            raise ValueError("TaskJobRollouts.tokens holds the tokens of each rollout that ran, in run order")
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

    def metrics(self) -> dict[str, object]:
        """The job metrics of these rollouts.

        ``rollout_rewards`` maps each task to its rewards (``rewards``), ``rollout_samples`` counts the samples that
        train, and ``removed_rollouts`` the rollouts that give no sample: those that failed in the infrastructure and
        those whose calls do not assemble into a sample, as on Slime. ``rollout_tokens`` maps each task to the
        ``response_tokens`` and ``trained_tokens`` of each of its rollouts that ran, in run order, and
        ``rollout_response_tokens`` and ``rollout_trained_tokens`` are their sums over the job.
        ``unrenderable_samples`` counts the samples left out for their length (``left_out``), when there are any.
        """
        rollouts = sum(len(scores) for _, scores in self.scores.tasks)
        return {
            "rollout_rewards": {name: list(values) for name, values in self.rewards().items()},
            "rollout_samples": len(self.samples),
            "removed_rollouts": rollouts - sum(item.assembled for item in self.tokens),
            "rollout_tokens": {
                name: [
                    {"response_tokens": item.response_tokens, "trained_tokens": item.trained_tokens}
                    for item in self.tokens
                    if item.task == name
                ]
                for name, _ in self.scores.tasks
            },
            "rollout_response_tokens": sum(item.response_tokens for item in self.tokens),
            "rollout_trained_tokens": sum(item.trained_tokens for item in self.tokens),
            **({"unrenderable_samples": self.left_out} if self.left_out else {}),
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


def rollout_sample(rollout: TaskRollout, group: str, assembly: SampleAssembly, start: str) -> TrajectorySample | None:
    """The sample of one rollout in ``group``, or None when it has no calls, an aborted call, or does not assemble.

    Each call sampled the weights ``start``, so each turn's record names ``start`` as its ``artifact_id``: the
    turns of one rollout are one policy. A rollout that ran without a reward scores 0.
    """
    if not rollout.turns or any(inference_aborted(turn.response) for turn in rollout.turns):
        return None
    inferences = tuple(
        AgentRecord.create(
            scenario=ROLLOUT_SCENARIO,
            request_type=RequestType.INFERENCE,
            payload={**turn.request, "response": dict(turn.response), "artifact_id": start},
            artifact_id=start,
        )
        for turn in rollout.turns
    )
    try:
        sample = assembly.assemble(inferences, rollout.reward or 0.0, source_id=rollout.rollout_id)
    except ValueError:
        logger.warning("rollout %s of task %s does not assemble into a sample", rollout.rollout_id, group)
        return None
    return replace(sample, group_id=group, source_agent_record_ids=())


def rollout_assembly(settings: HarborSettings) -> SampleAssembly:
    """The assembly of a task rollout's model calls into one sample, as ``rollout_sample`` reads it."""
    return SampleAssembly(
        SampleAssemblySettings(
            accept_multi_turn_policy_samples=True,
            realign_threshold=REALIGN_THRESHOLD,
            scaffold_tolerance=settings.scaffold_tolerance,
        )
    )


def run_train_tasks(
    tasks: Sequence[TaskSample],
    settings: RolloutSettings,
    chat: RolloutChat,
    *,
    model: str,
    work_dir: Path,
    start: str,
    model_adapter: str | None = None,
    lab: TaskLab | None = None,
    max_tokens: int | None = None,
) -> TaskJobRollouts:
    """Run ``settings.rollouts_per_task`` rollouts of each task, answering their model calls with ``chat``.

    ``model`` is the model name the agent sends, and ``model_adapter`` the adapter it names, or None. ``start`` is
    the job's ``start``, the weights that ``chat`` samples; each sample names it as its ``artifact_id``.
    ``lab`` runs each rollout; None is reef-eval's Harbor lab under ``work_dir``. ``max_tokens`` is the most tokens a
    sample's targets, every token but the first, may have, or None for no bound: a longer sample is left out before
    its task's group forms. The result counts the tokens of each rollout that ran (``RolloutTokens``).
    """
    assembly = rollout_assembly(settings)
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
                RolloutRequest(runner.harness, runner.policy_model, str(task.task_path))
                for task in tasks
                for _ in range(settings.rollouts_per_task)
            ]
        )
    samples: list[TrajectorySample] = []
    tokens: list[RolloutTokens] = []
    left_out = 0
    for index, task in enumerate(tasks):
        name = task_name(task)
        ran = [
            rollout
            for rollout in rollouts[index * settings.rollouts_per_task : (index + 1) * settings.rollouts_per_task]
            if not rollout.is_infrastructure_failure
        ]
        assembled = [(rollout, rollout_sample(rollout, name, assembly, start)) for rollout in ran]
        # A sample whose targets are longer than max_tokens is left out before the group forms: Tinker refuses a whole
        # optimizer step for one such row.
        rows = [
            (
                rollout,
                sample,
                sample is not None and (max_tokens is None or len(sample.training["tokens"]) - 1 <= max_tokens),
            )
            for rollout, sample in assembled
        ]
        group = [sample for _, sample, is_fitting in rows if sample is not None and is_fitting]
        left_out += sum(sample is not None and not is_fitting for _, sample, is_fitting in rows)
        # A group of fewer than two samples has no relative reward, so it does not train.
        trains = len(group) >= 2
        if trains:
            samples.extend(group)
        for rollout, sample, is_fitting in rows:
            sampled = sum(
                len(recorded_training({"response": turn.response}).get("loss_mask", ())) for turn in rollout.turns
            )
            trained = sum(sample.training.get("loss_mask", [])) if sample is not None and is_fitting and trains else 0
            tokens.append(RolloutTokens(name, sampled, trained, assembled=sample is not None))
    scores = RolloutScores.by_task(
        [task_name(task) for task in tasks], [runner.rollout_score(rollout) for rollout in rollouts]
    )
    return TaskJobRollouts(tuple(samples), scores, tuple(tokens), left_out)
