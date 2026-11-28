"""TrafficTaskGenerator: Harbor tasks built from the failed sessions of live traffic by a generation service.

The flow of one generation, in the order it runs:

    generate        The round's records become traffic sessions (``sessions.traffic_sessions``). The sessions that
                    scored at most ``max_score`` go to one generation run of ``spec`` as a ``records`` source. The
                    generator waits until the run ends, then downloads each task that the run published. Each task
                    gets the record IDs of the session that it came from as its source records.
    validate        Harbor's oracle and nop agents on this host: the reference solution scores 1 and doing nothing
                    scores below 1. The training rollouts run on this host, so a task whose environment does not
                    build here does not train.
    record_results  The first rollout rewards of each task of the generation go to the state document
                    ``results/<generation>``, which an analysis reads.

The generation service owns the method: its spec mines environments from the failures, builds and reviews tasks,
and keeps what the reviews learned about building tasks (the stage memory of its generator). The generator here only
moves sessions in and tasks out, so a deployment changes the method by changing the spec.

The state keeps the run of each generation, so a restart waits for that run and starts no second one. A run that
ended ``failed`` or ``cancelled`` is the end of one attempt: the stage's next attempt starts a new run.

    starting/<generation>   {requested_at}: written before the start of the run
    run/<generation>        {run_id}: written when the service queued the run

A start whose answer was lost leaves ``starting`` only. After a restart, the newest run of ``generator`` queued after
``requested_at`` that no other generation names is the run of the generation; with none, the generation starts a run
again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Self

from recipes.beta.continual_envs.generation_service import GenerationRun, GenerationService, PlatformGenerationService
from recipes.beta.continual_envs.sessions import TrafficSession, record_files, traffic_sessions
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.core.values import JsonValue
from reef.recipe.generated_tasks.generator import (
    FailedTask,
    GeneratedTaskResult,
    GeneratorState,
    HeldOutTask,
    RejectedTask,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.samples.generator.harbor import oracle_check
from reef.samples.generator.inputs import TaskGenerationRequest

logger = logging.getLogger(__name__)

#: The name of the run source that carries the sessions; the spec's mine stage reads it.
SOURCE_NAME = "traffic"
#: The generation service's limit of the files of one source is 16 MiB; the rest is room for the JSON around them.
DEFAULT_MAX_SOURCE_BYTES = 15 * 1024**2
#: How far the service's clock may be behind this host's when the generator looks for a run that a lost answer queued.
CLOCK_SKEW_SECONDS = 120.0


@dataclass(frozen=True)
class TrafficGeneratorSettings:
    """The config of ``TrafficTaskGenerator``.

    Values: ``service_url`` is the Reef Platform's http or https URL, and ``token_env`` the variable of the
    deployment's environment that holds a workspace access token with the scope ``generators:write``. ``spec`` names
    the generation spec (``<name>`` or ``<name>@<version>``), and ``generator`` the service's generator, whose item
    tables and stage memory every generation of the scenario shares. ``stage`` is the spec's build stage that
    publishes the tasks. ``params`` and ``targets`` go to each run as they are. A session is a failure when its
    report's score is at most ``max_score``, in [0, 1]; 1 sends every scored session. ``max_source_bytes`` > 0 bounds
    the files of one run's source. A run that has not ended after ``run_timeout_seconds`` > 0 is cancelled, and its
    state is read every ``poll_seconds`` > 0. ``oracle_check`` runs Harbor's oracle and nop agents on each task, in the
    Harbor environment ``oracle_environment``.
    Purpose: ``TrafficTaskGenerator.from_config`` reads it from the section's config.
    """

    service_url: str
    spec: str
    generator: str
    token_env: str = "REEF_ACCESS_TOKEN"
    stage: str = "tasks"
    params: Mapping[str, JsonValue] = field(default_factory=dict)
    targets: Mapping[str, JsonValue] = field(default_factory=dict)
    max_score: float = 0.5
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    run_timeout_seconds: float = 6 * 3600.0
    poll_seconds: float = 30.0
    oracle_check: bool = True
    oracle_environment: str = "docker"

    def __post_init__(self) -> None:
        texts = {
            "service_url": self.service_url,
            "spec": self.spec,
            "generator": self.generator,
            "token_env": self.token_env,
            "stage": self.stage,
            "oracle_environment": self.oracle_environment,
        }
        for name, text in texts.items():
            if not isinstance(text, str) or not text.strip():
                raise PluginConfigError(f"TrafficTaskGenerator {name} must be a non-empty string")
        if not isinstance(self.params, Mapping) or not isinstance(self.targets, Mapping):
            raise PluginConfigError("TrafficTaskGenerator params and targets must be mappings")
        numbers = {
            "max_score": self.max_score,
            "max_source_bytes": self.max_source_bytes,
            "run_timeout_seconds": self.run_timeout_seconds,
            "poll_seconds": self.poll_seconds,
        }
        for name, number in numbers.items():
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise PluginConfigError(f"TrafficTaskGenerator {name} must be a number")
        if not 0 <= self.max_score <= 1:
            raise PluginConfigError("TrafficTaskGenerator max_score must be from 0 to 1")
        if not isinstance(self.max_source_bytes, int) or self.max_source_bytes <= 0:
            raise PluginConfigError("TrafficTaskGenerator max_source_bytes must be a whole number above 0")
        if self.run_timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise PluginConfigError("TrafficTaskGenerator run_timeout_seconds and poll_seconds must be above 0")
        if not isinstance(self.oracle_check, bool):
            raise PluginConfigError("TrafficTaskGenerator oracle_check must be true or false")


class TrafficTaskGenerator(TaskGenerator):
    """The generator of one scenario: the flow at the top of the module.

    ``service`` is the generation service, ``settings`` the config, and ``state`` the scenario's generator state.
    """

    def __init__(self, service: GenerationService, settings: TrafficGeneratorSettings, state: GeneratorState) -> None:
        self.service = service
        self.settings = settings
        self.state = state

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str], state: GeneratorState
    ) -> Self:
        known = {setting.name for setting in fields(TrafficGeneratorSettings)}
        unknown = sorted(str(key) for key in config if key not in known)
        if unknown:
            raise PluginConfigError(f"TrafficTaskGenerator does not take {', '.join(unknown)}")
        try:
            settings = TrafficGeneratorSettings(**config)
        except TypeError as error:
            raise PluginConfigError(f"TrafficTaskGenerator needs service_url, spec and generator: {error}") from error
        token = environ.get(settings.token_env, "")
        if not token:
            raise PluginConfigError(f"TrafficTaskGenerator needs a workspace access token in {settings.token_env}")
        return cls(PlatformGenerationService(settings.service_url, token), settings, state)

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        sessions = [
            session
            for session in traffic_sessions(request.source_records)
            if session.score is not None and session.score <= self.settings.max_score
        ]
        if not sessions:
            logger.info("generation %d has no failed session", request.generation)
            return ()
        files, left_out = record_files(sessions, self.settings.max_source_bytes)
        if left_out:
            logger.warning(
                "generation %d leaves out %d sessions over the source limit", request.generation, len(left_out)
            )
        run = await self.started_run(request.generation, files)
        run = await self.ended_run(run)
        if run.state != "finished":
            raise RuntimeError(f"the generation run {run.run_id} ended {run.state}: {run.error or 'no error text'}")
        names = await self.service.published_tasks(run.run_id, self.settings.stage)
        logger.info("generation %d: run %s published %d tasks", request.generation, run.run_id, len(names))
        return await self.downloaded_tasks(names, {session.report.agent_record_id: session for session in sessions})

    async def started_run(self, generation: int, files: Mapping[str, str]) -> GenerationRun:
        """The run of ``generation``: the one that the state names, or a new one. A run that ended ``failed`` or
        ``cancelled`` is not taken again: the stage's next attempt of the generation starts a new run."""
        if (document := self.state.read(f"run/{generation:05d}")) is not None:
            run = await self.service.run(json.loads(document)["run_id"])
            if run.state in ("queued", "running", "finished"):
                return run
        elif (document := self.state.read(f"starting/{generation:05d}")) is not None:
            # The clocks of this host and of the service differ a little; a run of another generation is known.
            earliest = json.loads(document)["requested_at"] - CLOCK_SKEW_SECONDS
            known = {json.loads(self.state.read(key) or "{}").get("run_id") for key in self.state.keys("run/")}
            listed = await self.service.runs(self.settings.generator)
            queued = [run for run in listed if run.created_at >= earliest and run.run_id not in known]
            if queued:
                self.state.write(f"run/{generation:05d}", json.dumps({"run_id": queued[0].run_id}))
                return queued[0]
        self.state.write(f"starting/{generation:05d}", json.dumps({"requested_at": time.time()}))
        source: dict[str, JsonValue] = {
            "name": SOURCE_NAME,
            "kind": "records",
            "id": f"generation-{generation:05d}",
            "files": dict(files),
        }
        run = await self.service.start(
            spec=self.settings.spec,
            generator=self.settings.generator,
            sources=[source],
            params=self.settings.params,
            targets=self.settings.targets,
        )
        self.state.write(f"run/{generation:05d}", json.dumps({"run_id": run.run_id}))
        logger.info(
            "generation %d: run %s of %s with %d sessions", generation, run.run_id, self.settings.spec, len(files)
        )
        return run

    async def ended_run(self, run: GenerationRun) -> GenerationRun:
        """``run`` once it ended. A run that has not ended within ``run_timeout_seconds`` is cancelled."""
        deadline = time.monotonic() + self.settings.run_timeout_seconds
        while not run.ended:
            if time.monotonic() >= deadline:
                await self.service.cancel(run.run_id)
                raise TimeoutError(f"the generation run {run.run_id} ran past {self.settings.run_timeout_seconds} s")
            await asyncio.sleep(self.settings.poll_seconds)
            run = await self.service.run(run.run_id)
        return run

    async def downloaded_tasks(
        self, names: tuple[str, ...], sessions: Mapping[str, TrafficSession]
    ) -> tuple[HarborTask, ...]:
        """The published tasks ``names``. Each task's source records are those of the sessions that it names among
        its own source records: the run names a task's session by its record folder, the report's ID."""
        tasks = []
        with tempfile.TemporaryDirectory(prefix="reef-traffic-tasks-") as scratch:
            for name in names:
                task = read_harbor_task(await self.service.download_task(name, Path(scratch)))
                named = [sessions[folder] for folder in task.source_agent_record_ids if folder in sessions]
                ids = tuple(dict.fromkeys(record_id for session in named for record_id in session.record_ids))
                tasks.append(replace(task, source_agent_record_ids=ids))
        return tuple(tasks)

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        if not self.settings.oracle_check:
            return TaskValidationResult()
        with tempfile.TemporaryDirectory(prefix="reef-traffic-check-") as scratch:
            path = write_harbor_task(task, Path(scratch))
            result = await asyncio.to_thread(oracle_check, path, environment=self.settings.oracle_environment)
        return TaskValidationResult() if result.is_solvable else TaskValidationResult((result.reason,))

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        rows: list[JsonValue] = []
        for result in results:
            match result:
                case RolledOutTask(task=task, rollout_rewards=rewards):
                    rows.append({"task": task.name, "result": "rolled_out", "rewards": list(rewards)})
                case RejectedTask(task=task, reason=reason):
                    rows.append({"task": task.name, "result": "rejected", "reason": reason})
                case HeldOutTask(task=task):
                    rows.append({"task": task.name, "result": "held_out"})
                case FailedTask(task=task, error=error):
                    rows.append({"task": task.name, "result": "failed", "reason": error})
        self.state.write(f"results/{generation:05d}", json.dumps(rows))
