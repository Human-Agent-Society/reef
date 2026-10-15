"""Benchmark real Cordis/Pi evolution with a deterministic local model endpoint.

The reference uses the same local admission/start/reclaim implementation and a
FIFO client pool with the same tenant quota and idle TTL, without a Scheduler.
Production scheduling code is unchanged. Pi is a real installed executable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import statistics
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock
from urllib.request import Request, urlopen

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, unused_port

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import final_assistant_text
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.cordis import CordisRecipe
from reef.scenario.scheduling import MultiTenantScheduler, SchedulerConfig
from reef.service.app import create_app
from reef.service.scheduler import create_scheduler_app
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import ServiceSchedulingConfig, object_value, text_value
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.scheduling import ScenarioAdmission
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.strategies import EpisodeScorer, Proposer
from reef.train.types import TrajectoryItem

REPLY_MARKER = "REEF-OVERHEAD-EVOLVED"
SERVICE_TOKEN = "benchmark-service-token"
CONTROL_TOKEN = "benchmark-control-token"
MODEL_NAME = "reef-controlled-model"


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
        "maximum": ordered[-1],
    }


@dataclass(frozen=True)
class Transition:
    seconds: float
    scenario_id: int
    tenant_id: int
    previous: str
    state: str
    generation: int
    idle_since: float | None
    queue_sequence: int


@dataclass(frozen=True)
class ControlCall:
    started_seconds: float
    duration_seconds: float
    operation: str
    scenario_id: int | None
    success: bool


class EvolutionTrace:
    """Record overlapping proposal intervals, real episode results and lifecycle transitions."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.release_proposals = Event()
        self.entered: set[str] = set()
        self.proposals: dict[str, tuple[float, float]] = {}
        self.episodes: list[dict[str, object]] = []
        self.transitions: list[Transition] = []
        self.eligible: dict[tuple[int, int], float] = {}
        self.queue_maximum: dict[int, int] = {}
        self.used_maximum: dict[int, int] = {}
        self.reference_used: dict[int, int] = {}
        self.enqueues: list[dict[str, float | int]] = []
        self.controls: list[ControlCall] = []

    def observe(self, scheduler: MultiTenantScheduler) -> None:
        now = time.perf_counter()
        with self.lock:
            for tenant_id, used in scheduler.used_by_tenant.items():
                self.used_maximum[tenant_id] = max(self.used_maximum.get(tenant_id, 0), used)
            for tenant_id, waiting in scheduler.waiting_by_tenant.items():
                self.queue_maximum[tenant_id] = max(self.queue_maximum.get(tenant_id, 0), len(waiting))
                if waiting:
                    head = scheduler.admissions_by_id[waiting[0]]
                    if scheduler.used_by_tenant.get(
                        tenant_id, 0
                    ) + head.execution_units <= scheduler.config.tenant_quotas[
                        tenant_id
                    ] and scheduler.availability.get(
                        head.service_id, False
                    ):
                        self.eligible.setdefault((head.scenario_id, head.generation), now)

    def record(self, previous: ScenarioAdmission, current: ScenarioAdmission) -> None:
        if (previous.state, previous.generation) == (current.state, current.generation):
            return
        with self.lock:
            self.transitions.append(
                Transition(
                    time.perf_counter(),
                    current.scenario_id,
                    current.tenant_id,
                    previous.state,
                    current.state,
                    current.generation,
                    current.idle_since,
                    current.queue_sequence,
                )
            )


class MeasuredScheduler(MultiTenantScheduler):
    def __init__(self, store: SQLiteSchedulingStore, config: SchedulerConfig, trace: EvolutionTrace) -> None:
        self.trace = trace
        super().__init__(store, config)

    def request_scenario_admission(
        self,
        tenant_id: int,
        service_name: str,
        recipe: str,
        scenario_name: str,
        *,
        release_id: str | None = None,
        model_config_ref: str | None = None,
        now_seconds: float | None = None,
    ) -> ScenarioAdmission:
        admission = super().request_scenario_admission(
            tenant_id,
            service_name,
            recipe,
            scenario_name,
            release_id=release_id,
            model_config_ref=model_config_ref,
            now_seconds=now_seconds,
        )
        with self.lock:
            self.trace.observe(self)
            with self.trace.lock:
                self.trace.enqueues.append(
                    {
                        "scenario_id": admission.scenario_id,
                        "tenant_id": admission.tenant_id,
                        "generation": admission.generation,
                        "queue_sequence": admission.queue_sequence,
                        "seconds": time.perf_counter(),
                    }
                )
        return admission

    def save_scenario_admission(
        self, admission: ScenarioAdmission, *, next_queue_sequence: int | None = None
    ) -> ScenarioAdmission:
        previous = self.find_scenario_admission(admission.scenario_id)
        self.trace.observe(self)
        current = super().save_scenario_admission(admission, next_queue_sequence=next_queue_sequence)
        self.trace.record(previous, current)
        self.trace.observe(self)
        return current


class MeasuredController(SchedulerController):
    def __init__(self, scheduler: MultiTenantScheduler, trace: EvolutionTrace) -> None:
        super().__init__(scheduler, {})
        self.trace = trace

    def control(self, service_id: int, payload: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        success = False
        try:
            response = super().control(service_id, payload)
            success = True
            return response
        finally:
            identifier = payload.get("scenario_id")
            if not isinstance(identifier, int):
                identifier = None
            with self.trace.lock:
                self.trace.controls.append(
                    ControlCall(
                        started, time.perf_counter() - started, text_value(payload, "operation"), identifier, success
                    )
                )


class ModelProposer(Proposer):
    """Call the controlled model through the real model-binding interface."""

    def __init__(self, trace: EvolutionTrace) -> None:
        self.trace = trace

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TrajectoryItem, ...],
        models: ModelBindings,
        *,
        requests: Sequence[Mapping[str, object]] = (),
    ) -> Mutation:
        if len(requests) != 1 or samples:
            raise ValueError("benchmark requires exactly one manual evolve request")
        request_id = text_value(requests[0], "id")
        with self.trace.lock:
            if request_id in self.trace.entered:
                raise ValueError("an evolve instruction was proposed more than once")
            self.trace.entered.add(request_id)
        if not self.trace.release_proposals.wait(120):
            raise TimeoutError("benchmark did not release its initial proposals")
        started = time.perf_counter()
        rule = models.served.chat(
            [
                {"role": "system", "content": "REEF_BENCHMARK_PROPOSAL: return only the requested AGENTS.md rule."},
                {"role": "user", "content": text_value(requests[0], "text")},
            ],
            temperature=0,
            max_tokens=256,
            timeout_s=600,
        )
        with self.trace.lock:
            self.trace.proposals[request_id] = (started, time.perf_counter())
        return Mutation("create", f"rules-{request_id}", {"name": "rules", "config": {"text": rule}})


class ReplyScorer(EpisodeScorer):
    def __init__(self, trace: EvolutionTrace) -> None:
        self.trace = trace

    def __call__(self, task: str, result: EpisodeResult) -> float:
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0 or not reply or result.residue:
            raise ValueError(
                f"Pi episode failed: exit={result.exit_code}, stderr={result.stderr[-500:]}, residue={result.residue}"
            )
        score = float(REPLY_MARKER in reply)
        with self.trace.lock:
            self.trace.episodes.append({"seconds": time.perf_counter(), "exit_code": result.exit_code, "score": score})
        return score


class ControlledModel:
    """OpenAI-compatible local model for actual proposal calls and actual Pi processes."""

    def __init__(self, episode_seconds: float, proposal_seconds: float) -> None:
        self.episode_seconds = episode_seconds
        self.proposal_seconds = proposal_seconds
        self.active = 0
        self.maximum_active = 0
        self.requests: list[dict[str, object]] = []

    async def catalog(self, request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": MODEL_NAME}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        messages = payload["messages"]
        serialized = json.dumps(messages)
        proposal = "REEF_BENCHMARK_PROPOSAL" in serialized
        if proposal:
            response_text = f"Always end your final answer with {REPLY_MARKER}."
            delay = self.proposal_seconds
        else:
            response_text = "Hello."
            if REPLY_MARKER in serialized:
                response_text += f" {REPLY_MARKER}"
            delay = self.episode_seconds
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        started = time.perf_counter()
        try:
            await asyncio.sleep(delay)
            common = {"id": "controlled-completion", "created": int(time.time()), "model": MODEL_NAME}
            if payload.get("stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                for delta, finish in (({"role": "assistant", "content": response_text}, None), ({}, "stop")):
                    chunk = {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            return web.json_response(
                {
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        finally:
            self.active -= 1
            self.requests.append({"started": started, "finished": time.perf_counter(), "proposal": proposal})


@dataclass
class EvolveJob:
    scenario_id: int
    tenant_id: int
    name: str
    round_number: int
    enqueued_seconds: float = 0
    admitted_seconds: float = 0
    submitted_seconds: float = 0
    settled_seconds: float = 0
    response_ms: float = 0
    selected: bool = False

    @property
    def request_id(self) -> str:
        return f"evolve-{self.name}-r{self.round_number}"


@dataclass
class HarnessCluster:
    trace: EvolutionTrace
    model: ControlledModel
    dispatcher: Dispatcher
    storage: SQLiteScenarioStorage
    service: TestClient
    scheduler_client: TestClient | None
    controller: SchedulerController | None
    idle_ttl_seconds: float
    quota: int
    live_model: bool
    cold_submissions: asyncio.Semaphore


@asynccontextmanager
async def harness_cluster(
    directory: Path,
    *,
    scheduled: bool,
    tenant_count: int,
    quota: int,
    idle_ttl_seconds: float,
    binary: str,
    episode_seconds: float,
    proposal_seconds: float,
    live_url: str | None = None,
    live_model_name: str | None = None,
    api_key_env: str | None = None,
) -> AsyncIterator[HarnessCluster]:
    directory.mkdir(parents=True)
    trace = EvolutionTrace()
    model = ControlledModel(episode_seconds, proposal_seconds)
    model_app = web.Application()
    model_app.router.add_get("/v1/models", model.catalog)
    model_app.router.add_post("/v1/chat/completions", model.chat)
    async with TestClient(TestServer(model_app)) as model_client:
        if live_url is not None:
            if api_key_env is None:
                raise ValueError("live model requires --api-key-env")
            api_key = os.environ.get(api_key_env, "")
            if not api_key or not live_model_name:
                raise ValueError("live model requires a credential and a model name")
            endpoint = live_url
            model_name = live_model_name
        else:
            api_key = "local-benchmark"
            endpoint = str(model_client.make_url("/")).rstrip("/")
            model_name = MODEL_NAME
        recipe = CordisRecipe(
            propose=ModelProposer(trace),
            score_episode=ReplyScorer(trace),
            tasks=("Say hello in one sentence. Do not use tools.",),
            binary=binary,
            executor=LocalExecutor(),
            episode_timeout_s=600,
            seed=(
                {"id": "models", "name": "config", "config": {"target": "models", "data": {"providers": {}}}},
                {"id": "settings", "name": "config", "config": {"data": {"defaultThinkingLevel": "off"}}},
            ),
            runtime=InferenceProxyRuntime(
                model_path=model_name, base_url=endpoint, api_key=api_key, inference_timeout_s=600
            ),
            training_mode="manual",
            episode_workers=1,
            proposals_dir=str(directory / "proposals"),
        )
        if recipe.scenario_execution_units() != 1:
            raise ValueError("benchmark requires one real evaluation worker per scenario")
        bootstrap = directory / "bootstrap"
        bootstrap.mkdir()
        for relative, content in (recipe.base_artifact_files() or {}).items():
            target = bootstrap / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        storage = SQLiteScenarioStorage(directory / "records")
        dispatcher = Dispatcher(
            recipe,
            InMemoryRepositoryBackend.factory(bootstrap, root=directory / "repositories"),
            scenario_storage=storage,
            agent_record_dir=directory / "agent-records",
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(directory / "assignments"),
        )
        controller = None
        scheduler_client = None
        service = None
        try:
            if scheduled:
                scheduler = MeasuredScheduler(
                    SQLiteSchedulingStore(directory / "scheduler.sqlite3"),
                    SchedulerConfig(
                        quota * tenant_count,
                        dict.fromkeys(range(1, tenant_count + 1), quota),
                        idle_ttl_seconds=idle_ttl_seconds,
                    ),
                    trace,
                )
                controller = MeasuredController(scheduler, trace)
                scheduler_client = TestClient(TestServer(create_scheduler_app(controller)))
                await scheduler_client.start_server()
                port = unused_port()
                scheduling = ServiceSchedulingConfig(
                    scheduler_url=str(scheduler_client.make_url("/")),
                    service_id=1,
                    service_name="harness",
                    recipe=recipe.name,
                    standard_execution_units=recipe.scenario_execution_units(),
                    control_token=CONTROL_TOKEN,
                    service_url=f"http://127.0.0.1:{port}",
                )
            else:
                port = unused_port()
                scheduling = None
            service = TestClient(
                TestServer(
                    create_app(
                        dispatcher,
                        tokens=SERVICE_TOKEN,
                        scheduling=scheduling,
                        staged_model_config_directory=directory / "models",
                    ),
                    port=port,
                )
            )
            await service.start_server()
            if controller is not None:
                async with asyncio.timeout(15):
                    while not controller.scheduler.availability.get(1, False):
                        await asyncio.sleep(0.02)
            yield HarnessCluster(
                trace,
                model,
                dispatcher,
                storage,
                service,
                scheduler_client,
                controller,
                idle_ttl_seconds,
                quota,
                live_url is not None,
                asyncio.Semaphore(3),
            )
        finally:
            trace.release_proposals.set()
            if scheduler_client is not None:
                await scheduler_client.close()
            if service is not None:
                await service.close()
            await asyncio.to_thread(dispatcher.close)


def job_headers(job: EvolveJob) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {SERVICE_TOKEN}",
        "x-reef-tenant": str(job.tenant_id),
        "x-reef-scenario": job.name,
    }


async def evolve(cluster: HarnessCluster, job: EvolveJob) -> None:
    job.submitted_seconds = time.perf_counter()
    # Cold POSTs synchronously call back through the Scheduler; bound their
    # fanout in this fixture, whose HTTP applications share one thread pool.
    cold = job.round_number > 1
    if cold:
        await cluster.cold_submissions.acquire()
    try:
        response = await cluster.service.post(
            "/reef/train",
            headers=job_headers(job),
            json={
                "agent_record_id": job.request_id,
                "text": f"End every answer with {REPLY_MARKER}.",
                "session": job.name,
                "release_id": "",
            },
        )
        payload = await response.json()
    finally:
        if cold:
            cluster.cold_submissions.release()
    job.response_ms = (time.perf_counter() - job.submitted_seconds) * 1000
    if response.status not in (200, 202) or payload.get("agent_record_id") != job.request_id:
        raise ValueError(f"evolve submission failed: {response.status}, {payload}")
    async with asyncio.timeout(900):
        while True:
            progress = await cluster.service.get(
                f"/reef/harness/requests/{job.request_id}/progress", headers=job_headers(job)
            )
            status = await progress.json()
            if progress.status != 200:
                raise ValueError(f"evolve progress failed: {progress.status}, {status}")
            if status.get("settled"):
                job.settled_seconds = time.perf_counter()
                return
            if status.get("error") or status.get("state") in ("failed", "deleted"):
                raise ValueError(f"evolve did not settle: {status}")
            await asyncio.sleep(0.05)


async def reference_job(cluster: HarnessCluster, job: EvolveJob, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        job.admitted_seconds = time.perf_counter()
        started = await asyncio.to_thread(
            cluster.dispatcher.start_scheduled_scenario,
            job.scenario_id,
            job.tenant_id,
            job.name,
            generation=2 * job.round_number - 1,
        )
        if started.state != "admitted":
            raise ValueError(f"reference could not start scenario: {started}")
        job.admitted_seconds = time.perf_counter()
        with cluster.trace.lock:
            used = cluster.trace.reference_used.get(job.tenant_id, 0) + 1
            cluster.trace.reference_used[job.tenant_id] = used
            cluster.trace.used_maximum[job.tenant_id] = max(cluster.trace.used_maximum.get(job.tenant_id, 0), used)
        await evolve(cluster, job)
        # Match the scheduler's resource policy before admitting the next queued job.
        async with asyncio.timeout(30):
            while True:
                status = cluster.dispatcher.scheduled_scenario_status(job.name)
                if status is not None and status.idle_since is not None:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(max(0, status.idle_since + cluster.idle_ttl_seconds - time.time()))
            reclaimed = await asyncio.to_thread(
                cluster.dispatcher.reclaim_scheduled_scenario,
                job.scenario_id,
                job.name,
                tenant_id=job.tenant_id,
                generation=2 * job.round_number,
                inactive_before=time.time() - cluster.idle_ttl_seconds,
            )
            if not reclaimed.reclaim_complete:
                raise ValueError("reference failed to release its occupied execution slot")
            with cluster.trace.lock:
                cluster.trace.reference_used[job.tenant_id] -= 1


async def scheduled_job(cluster: HarnessCluster, job: EvolveJob) -> None:
    controller = cluster.controller
    if controller is None:
        raise ValueError("scheduled job requires a Scheduler")
    async with asyncio.timeout(600):
        while controller.scheduler.get_scenario_admission(job.scenario_id, job.tenant_id).state != "admitted":
            if controller.maintenance_error is not None:
                raise ValueError(controller.maintenance_error)
            await asyncio.sleep(0.02)
    job.admitted_seconds = time.perf_counter()
    await evolve(cluster, job)


async def check_settlements(cluster: HarnessCluster, jobs: list[EvolveJob]) -> None:
    for job in jobs:
        # Read durable commits after all workers release resources; this also checks cold history.
        with closing(cluster.storage.open(job.name)) as store:
            matching = [commit for commit in store.history() if job.request_id in commit.consumed_ids]
        if len(matching) != 1:
            raise ValueError(f"instruction did not settle exactly once: {job.request_id}")
        metrics = object_value(matching[0].metrics)
        if object_value(metrics.get("training_request"))["id"] != job.request_id:
            raise ValueError("commit lost its evolve request linkage")
        job.selected = metrics.get("selected") is True


async def run_round(cluster: HarnessCluster, jobs: list[EvolveJob], *, scheduled: bool) -> dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    tasks = []
    if scheduled:
        client = cluster.scheduler_client
        controller = cluster.controller
        if client is None or controller is None:
            raise ValueError("scheduled benchmark requires a scheduler client")

        async def submit_tenant(tenant_jobs: list[EvolveJob]) -> None:
            for job in tenant_jobs:
                job.enqueued_seconds = time.perf_counter()
                if job.round_number == 1:
                    response = await client.post(
                        "/reef/scenarios",
                        headers={"x-reef-tenant": str(job.tenant_id)},
                        json={
                            "tenant_id": job.tenant_id,
                            "service_name": "harness",
                            "recipe": cluster.dispatcher.recipe.name,
                            "scenario_name": job.name,
                        },
                    )
                    payload = await response.json()
                    if response.status not in (201, 202):
                        raise ValueError(f"scenario admission failed: {response.status}, {payload}")
                    identifier = payload.get("scenario_id")
                    if not isinstance(identifier, int):
                        raise ValueError("admission response requires scenario_id")
                    job.scenario_id = identifier
                    tasks.append(asyncio.create_task(scheduled_job(cluster, job)))
                else:
                    # Sleeping scenarios retain the instruction through the production cold-wake path.
                    tasks.append(asyncio.create_task(evolve(cluster, job)))

        await asyncio.gather(
            *(
                submit_tenant([job for job in jobs if job.tenant_id == tenant_id])
                for tenant_id in sorted({job.tenant_id for job in jobs})
            )
        )
    else:
        semaphores = {tenant: asyncio.Semaphore(cluster.quota) for tenant in {job.tenant_id for job in jobs}}
        for tenant in semaphores:
            cluster.trace.queue_maximum[tenant] = max(0, sum(job.tenant_id == tenant for job in jobs) - cluster.quota)
        for job in jobs:
            job.enqueued_seconds = time.perf_counter()
            tasks.append(asyncio.create_task(reference_job(cluster, job, semaphores[job.tenant_id])))
    if jobs[0].round_number == 1:
        expected = sum(
            min(cluster.quota, sum(job.tenant_id == tenant for job in jobs))
            for tenant in {job.tenant_id for job in jobs}
        )
        async with asyncio.timeout(120):
            while len(cluster.trace.entered) < expected:
                for task in tasks:
                    if task.done():
                        task.result()
                await asyncio.sleep(0.02)
        released = time.perf_counter()
        cluster.trace.release_proposals.set()
    else:
        released = started
    try:
        await asyncio.gather(*tasks)
        finished = max(job.settled_seconds for job in jobs)
        if scheduled:
            async with asyncio.timeout(60):
                while any(
                    controller.scheduler.get_scenario_admission(job.scenario_id, job.tenant_id).state != "sleep"
                    for job in jobs
                ):
                    await asyncio.sleep(0.05)
        await check_settlements(cluster, jobs)
        return {
            "round": jobs[0].round_number,
            "scenario_count": len(jobs),
            "end_to_end_seconds": finished - started,
            "execution_seconds_after_initial_release": finished - released,
            "prefill_seconds": released - started,
            "completion_throughput_per_second": len(jobs) / (finished - started),
            "combined_parent_cpu_seconds_until_all_scenarios_sleep": time.process_time() - cpu_started,
            "jobs": [asdict(job) for job in jobs],
            "evolve_post_ms": distribution([job.response_ms for job in jobs]),
            "evolve_submit_to_settle_ms": distribution(
                [(job.settled_seconds - job.submitted_seconds) * 1000 for job in jobs]
            ),
            "per_tenant_completion_seconds": {
                tenant: max(job.settled_seconds for job in jobs if job.tenant_id == tenant) - started
                for tenant in sorted({job.tenant_id for job in jobs})
            },
        }
    finally:
        cluster.trace.release_proposals.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def trial(
    directory: Path,
    args: argparse.Namespace,
    *,
    scheduled: bool,
    tenant_count: int,
    scenarios_per_tenant: int,
    rounds: int,
) -> dict[str, object]:
    async with harness_cluster(
        directory,
        scheduled=scheduled,
        tenant_count=tenant_count,
        quota=args.quota,
        idle_ttl_seconds=args.idle_ttl,
        binary=args.pi_binary,
        episode_seconds=args.episode_seconds,
        proposal_seconds=args.proposal_seconds,
        live_url=args.live_url,
        live_model_name=args.model_name,
        api_key_env=args.api_key_env,
    ) as cluster:
        histories = []
        identifiers = {}
        for round_number in range(1, rounds + 1):
            jobs = [
                EvolveJob(
                    identifiers.get((tenant, index), (tenant - 1) * scenarios_per_tenant + index + 1),
                    tenant,
                    f"t{tenant}-s{index}",
                    round_number,
                )
                for tenant in range(1, tenant_count + 1)
                for index in range(scenarios_per_tenant)
            ]
            histories.append(await run_round(cluster, jobs, scheduled=scheduled))
            identifiers = {
                (job.tenant_id, index): job.scenario_id
                for tenant in range(1, tenant_count + 1)
                for index, job in enumerate(job for job in jobs if job.tenant_id == tenant)
            }
        expected = tenant_count * scenarios_per_tenant * rounds
        if len(cluster.trace.proposals) != expected or len(cluster.trace.episodes) != expected * 2:
            raise ValueError("each evolve must perform exactly one proposal and two real Pi episodes")
        if not cluster.live_model and len(cluster.model.requests) != expected * 3:
            raise ValueError("each evolve must make one proposal model call and two Pi model calls")
        if any(maximum > args.quota for maximum in cluster.trace.used_maximum.values()):
            raise ValueError("scheduler exceeded the configured per-tenant execution quota")
        if scheduled:
            for tenant in range(1, tenant_count + 1):
                expected_order = [
                    entry["scenario_id"]
                    for entry in sorted(
                        (entry for entry in cluster.trace.enqueues if entry["tenant_id"] == tenant),
                        key=lambda entry: entry["queue_sequence"],
                    )
                ]
                actual_order = [
                    entry.scenario_id
                    for entry in cluster.trace.transitions
                    if entry.tenant_id == tenant and entry.state == "starting" and entry.generation == 1
                ]
                if actual_order != expected_order:
                    raise ValueError("first-start claims did not preserve each tenant's FIFO")
            if cluster.controller is None or cluster.controller.errors or cluster.controller.maintenance_error:
                raise ValueError("controller reported an error during the benchmark")
            if any(not call.success for call in cluster.trace.controls):
                raise ValueError("a scheduler control request failed during the benchmark")
        result = {
            "scheduled": scheduled,
            "live_model": cluster.live_model,
            "cold_post_concurrency_limit": 3,
            "tenant_count": tenant_count,
            "scenarios_per_tenant": scenarios_per_tenant,
            "quota_per_tenant": args.quota,
            "idle_ttl_seconds": args.idle_ttl,
            "episode_delay_seconds": args.episode_seconds,
            "proposal_delay_seconds": args.proposal_seconds,
            "rounds": histories,
            "real_pi_episode_count": len(cluster.trace.episodes),
            "model_request_maximum_overlap": cluster.model.maximum_active,
            "model_requests": cluster.model.requests,
            "proposal_intervals": cluster.trace.proposals,
            "transitions": [asdict(transition) for transition in cluster.trace.transitions],
            "enqueues": cluster.trace.enqueues,
            "control_calls": [asdict(call) for call in cluster.trace.controls],
            "unix_minus_monotonic_seconds": time.time() - time.perf_counter(),
            "eligible": [
                {"scenario_id": identifier, "generation": generation, "seconds": seconds}
                for (identifier, generation), seconds in cluster.trace.eligible.items()
            ],
            "maximum_queue_by_tenant": cluster.trace.queue_maximum,
            "maximum_used_by_tenant": cluster.trace.used_maximum,
        }
        (directory / "trial.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True)
    scenarios = [("single-tenant-queue", 1, count, 1) for count in args.scenario_counts]
    scenarios += [("concurrent-tenants", count, args.scenarios_per_tenant, 1) for count in args.tenant_counts]
    if args.wake:
        scenarios.append(("sleep-wake", 3, args.scenarios_per_tenant, 2))
    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pi_binary": args.pi_binary,
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "settings": {key: value for key, value in vars(args).items() if key not in ("output", "pi_binary")},
        "trials": [],
    }
    trials = []
    result["trials"] = trials
    for kind, tenants, count, rounds in scenarios:
        for repetition in range(args.repetitions):
            for scheduled in (False, True) if repetition % 2 == 0 else (True, False):
                label = f"{kind}-tenants{tenants}-scenarios{count}-repeat{repetition}-scheduler{scheduled}"
                measured = await trial(
                    args.output / label,
                    args,
                    scheduled=scheduled,
                    tenant_count=tenants,
                    scenarios_per_tenant=count,
                    rounds=rounds,
                )
                measured.update(kind=kind, repetition=repetition)
                trials.append(measured)
                (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"{label}: "
                    + ", ".join(
                        f"round {entry['round']}={entry['end_to_end_seconds']:.3f}s" for entry in measured["rounds"]
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pi-binary", default="/private/tmp/reef-live-pi-20261013/node_modules/.bin/pi")
    parser.add_argument("--scenario-counts", type=int, nargs="*", default=[3, 12, 36])
    parser.add_argument("--tenant-counts", type=int, nargs="*", default=[1, 3, 6])
    parser.add_argument("--scenarios-per-tenant", type=int, default=6)
    parser.add_argument("--quota", type=int, default=3)
    parser.add_argument("--idle-ttl", type=float, default=1)
    parser.add_argument("--episode-seconds", type=float, default=0.2)
    parser.add_argument("--proposal-seconds", type=float, default=0.05)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--wake", action="store_true")
    parser.add_argument("--live-url")
    parser.add_argument("--model-name")
    parser.add_argument("--api-key-env")
    args = parser.parse_args()
    binary = shutil.which(args.pi_binary)
    if binary is None:
        parser.error("install the real Pi executable and provide --pi-binary")
    args.pi_binary = binary
    if args.live_url:
        args.live_url = args.live_url.rstrip("/")
        if not args.api_key_env or not os.environ.get(args.api_key_env):
            parser.error("live model requires a populated --api-key-env")
        if not args.model_name:
            catalog_request = Request(
                f"{args.live_url}/v1/models", headers={"Authorization": f"Bearer {os.environ[args.api_key_env]}"}
            )
            with urlopen(catalog_request, timeout=30) as response:
                catalog = object_value(json.load(response))
            entries = catalog.get("data")
            if not isinstance(entries, list) or len(entries) != 1:
                parser.error("provide --model-name when the endpoint has multiple models")
            args.model_name = text_value(object_value(entries[0]), "id")
    if any(
        value < 1
        for value in [
            *args.scenario_counts,
            *args.tenant_counts,
            args.scenarios_per_tenant,
            args.quota,
            args.repetitions,
        ]
    ):
        parser.error("counts, quota and repetitions must be positive")
    if args.idle_ttl <= 0 or min(args.episode_seconds, args.proposal_seconds) < 0:
        parser.error("idle TTL must be positive and model delays nonnegative")
    logging.basicConfig(level=logging.WARNING)
    log_directory = args.output.parent
    log_directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_directory / f"{args.output.name}-queue.log")
    scheduler_logger = logging.getLogger("reef.scenario.scheduling")
    scheduler_logger.setLevel(logging.INFO)
    scheduler_logger.addHandler(handler)
    scheduler_logger.propagate = False
    try:
        asyncio.run(run(args))
    finally:
        scheduler_logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
