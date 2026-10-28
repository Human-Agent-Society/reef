"""Tinker's own rollouts: a task job's Harbor rollouts, sampled on Tinker at the job's start weights.

A task job holds ``TaskSample`` values. Before it trains, the component runs ``rollouts_per_task`` rollouts of
each task with ``TaskRolloutRunner``: the agent works in the task's sandbox, the task's verifier scores the
rollout, and each model call goes through the runner's capture proxy to a ``SamplerEndpoint`` on this host. The
endpoint samples Tinker's sampler of the job's start, so the rollouts sample the weights that the job trains.
Each rollout that ran assembles into one sample; the samples of one task form its group, which trains only with
at least two samples. Reef's serving takes no part.
"""

from __future__ import annotations

import json
import logging
import threading
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
from reef.runtime.interfaces import UpstreamStatusError, inference_aborted
from reef.tinker.sampling import TinkerSampler, sample_chat

logger = logging.getLogger(__name__)

#: The scenario name that rollout calls carry; the endpoint serves no scenario and ignores it.
ROLLOUT_SCENARIO = "tinker-rollout"
#: Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response.
REALIGN_THRESHOLD = 32768


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


class SamplerEndpoint:
    """An OpenAI text chat endpoint on loopback that samples one Tinker sampler, for the rollouts of one job.

    ``sampler_path`` is the job's start sampler, or None for the base model. Each answer carries the training
    capture with ``runtime_load_id``. A request that is not JSON gets 400, a sampler refusal its own status, and a
    sampler failure (a malformed sample, a timeout) 502 with the reason. The server runs on its own threads, so
    an agent that blocks the rollout's event loop cannot block it. Use it as a context manager: it binds its
    loopback port and sets ``url`` when it is entered, and closes the port when it exits.
    """

    def __init__(self, sampler: TinkerSampler, sampler_path: str | None, *, model: str, runtime_load_id: str) -> None:
        self.sampler = sampler
        self.sampler_path = sampler_path
        self.model = model
        self.runtime_load_id = runtime_load_id
        self.server: ThreadingHTTPServer | None = None
        self.url = ""

    def answer(self, payload: Any) -> tuple[int, Mapping[str, Any]]:
        """The status and the body that answer the chat request ``payload``."""
        try:
            return 200, sample_chat(
                self.sampler, self.sampler_path, payload, model=self.model, runtime_load_id=self.runtime_load_id
            )
        except UpstreamStatusError as exc:
            return exc.status, {"error": {"message": str(exc)}}
        except (ValueError, TimeoutError) as exc:
            return 502, {"error": {"message": f"the Tinker sampler failed: {exc}"}}

    def __enter__(self) -> SamplerEndpoint:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.reply(404, {"error": {"message": f"no route {self.path}"}})
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except json.JSONDecodeError as exc:
                    self.reply(400, {"error": {"message": f"the request is not JSON: {exc}"}})
                    return
                self.reply(*endpoint.answer(payload))

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
        threading.Thread(target=self.server.serve_forever, name="tinker-rollout-sampler", daemon=True).start()
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
    sampler: TinkerSampler,
    sampler_path: str | None,
    *,
    model: str,
    runtime_load_id: str,
    work_dir: Path,
    lab: TaskLab | None = None,
) -> TaskJobRollouts:
    """Run ``settings.rollouts_per_task`` rollouts of each task on ``sampler_path``, and group their samples.

    ``lab`` runs each rollout; None is reef-eval's Harbor lab under ``work_dir``.
    """
    assembly = SampleAssembly(
        SampleAssemblySettings(
            accept_multi_turn_policy_samples=True,
            realign_threshold=REALIGN_THRESHOLD,
            scaffold_tolerance=settings.scaffold_tolerance,
        )
    )
    with SamplerEndpoint(sampler, sampler_path, model=model, runtime_load_id=runtime_load_id) as endpoint:
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
            model_adapter=sampler_path,
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
