"""Opt-in concurrent tenant queues with real model proposals and real Pi episodes.

Use the same REEF_RUN_LIVE_HARNESS, REEF_LIVE_MODEL_API_KEY,
REEF_LIVE_MODEL_URL, REEF_LIVE_MODEL_NAME and REEF_REAL_PI_BINARY settings
as test_scheduler_harness_evolution.py. Each tenant gets its own trace file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event, Lock
from typing import TextIO
from urllib.request import Request, urlopen

import pytest
from aiohttp.test_utils import TestClient, TestServer, unused_port
from reef_service.test_harness_recipe import SEED_MODELS, SEED_SETTINGS

from recipes.gepa.recipe import GEPARecipe
from recipes.meta_harness.recipe import MetaHarnessRecipe
from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.episodes.model_binding import ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import final_assistant_text
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.cordis import CordisRecipe
from reef.recipe.reefine import ReefineRecipe
from reef.scenario.admission import ScenarioAdmissionError
from reef.scenario.scheduling import MultiTenantScheduler, SchedulerConfig
from reef.service.app import create_app
from reef.service.scheduler import create_scheduler_app
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import ServiceSchedulingConfig, integer_value, object_value, text_value
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.scheduling import ScenarioAdmission
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.strategies import EpisodeScorer, Proposer
from reef.train.types import TrajectoryItem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("REEF_RUN_LIVE_HARNESS") != "1", reason="live model test requires explicit opt-in"
    ),
]

TENANT_IDS = (1, 2, 3)
IDLE_TTL_SECONDS = 10
STEP_TIMEOUT_SECONDS = 1800
REPLY_MARKER = "REEF-MULTI-TENANT-EVOLVED"


@dataclass(frozen=True)
class TenantQueueEvent:
    monotonic_seconds: float
    tenant_id: int
    queue_before: tuple[int, ...]
    queue: tuple[int, ...]
    allocated: tuple[int, ...]
    used_execution_units: int


@dataclass(frozen=True)
class ScenarioTransition:
    monotonic_seconds: float
    previous: ScenarioAdmission
    current: ScenarioAdmission
    used_execution_units: int


class TenantQueueTrace(logging.Handler):
    """Retain weighted FIFO changes and durable lifecycle transitions per tenant."""

    def __init__(self, directory: Path, service_units: Mapping[int, int]) -> None:
        super().__init__(logging.INFO)
        self.service_units = service_units
        self.paths = {tenant_id: directory / f"tenant-{tenant_id}-queue.log" for tenant_id in TENANT_IDS}
        self.outputs: dict[int, TextIO] = {
            tenant_id: path.open("w", encoding="utf-8") for tenant_id, path in self.paths.items()
        }
        self.lifecycle_paths = {tenant_id: directory / f"tenant-{tenant_id}-lifecycle.log" for tenant_id in TENANT_IDS}
        self.lifecycle_outputs: dict[int, TextIO] = {
            tenant_id: path.open("w", encoding="utf-8") for tenant_id, path in self.lifecycle_paths.items()
        }
        self.events: list[TenantQueueEvent] = []
        self.transitions: list[ScenarioTransition] = []
        self.event_lock = Lock()
        for tenant_id, path in self.paths.items():
            print(f"Tenant {tenant_id} queue trace: {path}", flush=True)
            print(f"Tenant {tenant_id} lifecycle trace: {self.lifecycle_paths[tenant_id]}", flush=True)

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not message.startswith("tenant_queue_changed "):
            return
        payload = object_value(json.loads(message.split(" ", 1)[1]))
        tenant_id = integer_value(payload, "tenant_id")
        if tenant_id not in self.outputs:
            return
        queue = payload["queue"]
        before = payload["queue_before"]
        allocated = payload["allocated_scenario_ids"]
        used = payload["used_execution_units"]
        assert isinstance(queue, list) and isinstance(before, list) and isinstance(allocated, list)
        assert isinstance(used, int) and not isinstance(used, bool)
        entries = [
            {
                **object_value(entry),
                "execution_units": self.service_units[integer_value(object_value(entry), "service_id")],
            }
            for entry in queue
        ]
        payload["queue"] = entries
        assert all(isinstance(identifier, int) for identifier in before + allocated)
        event = TenantQueueEvent(
            time.monotonic(),
            tenant_id,
            tuple(before),
            tuple(integer_value(entry, "scenario_id") for entry in entries),
            tuple(allocated),
            used,
        )
        line = f"{record.created:.3f} tenant_queue_changed {json.dumps(payload, separators=(',', ':'))}\n"
        with self.event_lock:
            self.events.append(event)
            self.outputs[tenant_id].write(line)
            self.outputs[tenant_id].flush()
        print(
            f"tenant={tenant_id} queue={[(entry['scenario_name'], entry['execution_units']) for entry in entries]} "
            f"used={used}/{payload['allocated_execution_units']}",
            flush=True,
        )

    def record_transition(self, previous: ScenarioAdmission, current: ScenarioAdmission, used: int) -> None:
        event = ScenarioTransition(time.monotonic(), previous, current, used)
        payload = {
            "tenant_id": current.tenant_id,
            "scenario_id": current.scenario_id,
            "scenario_name": current.scenario_name,
            "service_id": current.service_id,
            "execution_units": current.execution_units,
            "generation": current.generation,
            "previous_state": previous.state,
            "state": current.state,
            "execution_allocated": current.execution_allocated,
            "used_execution_units": used,
        }
        with self.event_lock:
            self.transitions.append(event)
            output = self.lifecycle_outputs[current.tenant_id]
            output.write(f"{time.time():.3f} scenario_transition {json.dumps(payload, separators=(',', ':'))}\n")
            output.flush()

    def snapshot(self) -> tuple[TenantQueueEvent, ...]:
        with self.event_lock:
            return tuple(self.events)

    def transition_snapshot(self) -> tuple[ScenarioTransition, ...]:
        with self.event_lock:
            return tuple(self.transitions)

    def close(self) -> None:
        for output in (*self.outputs.values(), *self.lifecycle_outputs.values()):
            output.close()
        super().close()


class TracedScheduler(MultiTenantScheduler):
    """Observe committed transitions without changing admission, reclaim or queue policy."""

    def __init__(self, store: SQLiteSchedulingStore, config: SchedulerConfig, trace: TenantQueueTrace) -> None:
        self.trace = trace
        super().__init__(store, config)

    def save_scenario_admission(
        self, admission: ScenarioAdmission, *, next_queue_sequence: int | None = None
    ) -> ScenarioAdmission:
        previous = self.find_scenario_admission(admission.scenario_id)
        current = super().save_scenario_admission(admission, next_queue_sequence=next_queue_sequence)
        if (previous.state, previous.generation) != (current.state, current.generation):
            self.trace.record_transition(previous, current, self.used_by_tenant.get(current.tenant_id, 0))
        return current


class LiveTenantProposer(Proposer):
    """Hold the initial work concurrently, then ask the configured model for actual rules."""

    def __init__(self, initial_sessions: set[str]) -> None:
        self.initial_sessions = initial_sessions
        self.started = Event()
        self.release = Event()
        self.lock = Lock()
        self.sessions: set[str] = set()
        self.request_counts: Counter[str] = Counter()
        self.failures: list[str] = []

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TrajectoryItem, ...],
        models: ModelBindings,
        *,
        requests: Sequence[Mapping[str, object]] = (),
    ) -> Mutation:
        assert samples == () and len(requests) == 1
        request = requests[0]
        request_id = text_value(request, "id")
        session = text_value(request, "session")
        with self.lock:
            self.sessions.add(session)
            self.request_counts[request_id] += 1
            if self.initial_sessions <= self.sessions:
                self.started.set()
        if session in self.initial_sessions:
            assert self.release.wait(120), "initial concurrent proposals were not released"
        try:
            rule = models.served.chat(
                [
                    {
                        "role": "system",
                        "content": "Write a short AGENTS.md rule implementing the requested behavior. "
                        "Return only the rule text, without markdown fences. Do not execute tasks or use tools.",
                    },
                    {"role": "user", "content": text_value(request, "text")},
                ],
                temperature=0,
                max_tokens=512,
                timeout_s=900,
            ).strip()
        except ModelBindingError as exc:
            with self.lock:
                self.failures.append(f"{session}: model call failed ({type(exc).__name__})")
            raise
        assert rule, "the model returned no proposal"
        return Mutation("create", "requested-rules", {"name": "rules", "config": {"text": rule}})

    def snapshot(self) -> tuple[Counter[str], tuple[str, ...]]:
        with self.lock:
            return self.request_counts.copy(), tuple(self.failures)


class LiveTenantScorer(EpisodeScorer):
    """Score real replies and synchronize initial episodes so all tenants exercise waiting queues."""

    def __init__(self, initial_scenario_count: int, *, synchronize: bool) -> None:
        self.barrier = Barrier(initial_scenario_count, timeout=STEP_TIMEOUT_SECONDS)
        self.synchronized_episode_count = 2 * initial_scenario_count if synchronize else 0
        self.lock = Lock()
        self.episode_count = 0
        self.failures: list[str] = []

    def __call__(self, task: str, result: EpisodeResult) -> float:
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0 or not reply:
            with self.lock:
                self.failures.append(f"Pi episode failed: exit={result.exit_code}, reply_present={bool(reply)}")
            self.barrier.abort()
            raise ValueError("real Pi must exit successfully and produce an assistant reply")
        with self.lock:
            self.episode_count += 1
            synchronize = self.episode_count <= self.synchronized_episode_count
        if synchronize:
            try:
                self.barrier.wait()
            except BrokenBarrierError:
                with self.lock:
                    self.failures.append("initial episode synchronization failed")
                raise
        return float(REPLY_MARKER in reply)

    def snapshot(self) -> tuple[int, tuple[str, ...]]:
        with self.lock:
            return self.episode_count, tuple(self.failures)


@dataclass(frozen=True)
class RecordedLiveScorer(EpisodeScorer):
    """Share actual episode outcomes across spawned evaluation workers through atomic files."""

    directory: Path

    def __call__(self, task: str, result: EpisodeResult) -> float:
        reply = final_assistant_text(result.trajectory)
        score = float(REPLY_MARKER in reply)
        path = self.directory / f"{uuid.uuid4().hex}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"exit_code": result.exit_code, "reply_present": bool(reply), "score": score}),
            encoding="utf-8",
        )
        temporary.replace(path)
        if result.exit_code != 0 or not reply:
            raise ValueError("real Pi must exit successfully and produce an assistant reply")
        return score

    def snapshot(self) -> tuple[int, tuple[str, ...]]:
        outcomes = [object_value(json.loads(path.read_text())) for path in self.directory.glob("*.json")]
        failures = tuple(
            f"Pi episode failed: exit={outcome['exit_code']}, reply_present={outcome['reply_present']}"
            for outcome in outcomes
            if outcome["exit_code"] != 0 or not outcome["reply_present"]
        )
        return len(outcomes), failures


@dataclass
class HarnessService:
    service_id: int
    directory: Path
    recipe: CordisRecipe
    dispatcher: Dispatcher
    client: TestClient


@dataclass
class TenantScenario:
    tenant_id: int
    index: int
    scenario_id: int
    name: str
    service: HarnessService
    request_id: str | None = None
    committed: bool = False


def live_recipe(
    kind: str,
    directory: Path,
    proposer: LiveTenantProposer,
    scorer: EpisodeScorer,
    binary: str,
    model_name: str,
    base_url: str,
    api_key: str,
    execution_units: int,
) -> CordisRecipe:
    """Use genuine recipe classes with the same model-driven proposal seam and paired Pi task."""
    runtime = InferenceProxyRuntime(model_path=model_name, base_url=base_url, api_key=api_key, inference_timeout_s=900)
    options = {
        "propose": proposer,
        "score_episode": scorer,
        "tasks": ("Say hello in one short sentence. Do not use tools or inspect files.",),
        "binary": binary,
        "seed": (SEED_MODELS, SEED_SETTINGS),
        "runtime": runtime,
        "executor": LocalExecutor(),
        "episode_timeout_s": 900,
        "training_mode": "manual",
        "episode_workers": execution_units,
        "proposals_dir": str(directory / "proposals"),
    }
    if kind == "cordis":
        return CordisRecipe(**options)
    if kind == "reefine":
        return ReefineRecipe(**options)
    if kind == "gepa":
        return GEPARecipe(**options, archive_dir=directory / "gepa")
    if kind == "meta_harness":
        return MetaHarnessRecipe(**options, archive_dir=directory / "meta-harness")
    raise ValueError(f"unknown live recipe {kind!r}")


def run_concurrent_tenants(
    tmp_path: Path,
    service_specs: tuple[tuple[str, int], ...],
    *,
    tenant_quota: int,
    submissions: Mapping[int, tuple[int, ...]],
) -> None:
    """Exercise ordered weighted queues with one evolve per scenario and real idle recycling."""
    api_key = os.environ.get("REEF_LIVE_MODEL_API_KEY", "").strip()
    assert api_key, "set REEF_LIVE_MODEL_API_KEY before enabling live tests"
    binary = shutil.which(os.environ.get("REEF_REAL_PI_BINARY", "pi"))
    assert binary is not None, "install Pi and set REEF_REAL_PI_BINARY"
    base_url = os.environ.get("REEF_LIVE_MODEL_URL", "http://192.168.1.119:30000").rstrip("/")
    model_name = os.environ.get("REEF_LIVE_MODEL_NAME", "").strip()
    if not model_name:
        request = Request(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        with urlopen(request, timeout=30) as response:
            catalog = object_value(json.load(response))
        entries = catalog["data"]
        assert isinstance(entries, list) and len(entries) == 1, "set REEF_LIVE_MODEL_NAME for multi-model endpoints"
        model_name = text_value(object_value(entries[0]), "id")
    initial_sessions = set()
    initial_used: dict[int, int] = {}
    for tenant_id, execution_units in submissions.items():
        used = 0
        for index, units in enumerate(execution_units, start=1):
            if used + units > tenant_quota:
                break
            initial_sessions.add(f"t{tenant_id}-s{index}")
            used += units
        initial_used[tenant_id] = used
    proposer = LiveTenantProposer(initial_sessions)
    scorer: LiveTenantScorer | RecordedLiveScorer
    if all(units == 1 for _, units in service_specs):
        scorer = LiveTenantScorer(len(initial_sessions), synchronize=True)
    else:
        episode_directory = tmp_path / "episode-results"
        episode_directory.mkdir()
        scorer = RecordedLiveScorer(episode_directory)
    trace = TenantQueueTrace(tmp_path, {index: units for index, (_, units) in enumerate(service_specs, start=1)})
    logger = logging.getLogger("reef.scenario.scheduling")
    previous_level = logger.level
    logger.addHandler(trace)
    logger.setLevel(logging.INFO)
    scheduler = TracedScheduler(
        SQLiteSchedulingStore(tmp_path / "scheduler.sqlite3"),
        SchedulerConfig(
            tenant_quota * len(TENANT_IDS), dict.fromkeys(TENANT_IDS, tenant_quota), idle_ttl_seconds=IDLE_TTL_SECONDS
        ),
        trace,
    )
    controller = SchedulerController(scheduler, {})
    services: list[HarnessService] = []
    scenarios: dict[str, TenantScenario] = {}

    async def run() -> None:
        async with AsyncExitStack() as stack:
            scheduler_client = await stack.enter_async_context(
                TestClient(TestServer(create_scheduler_app(controller)))
            )
            for service_id, (kind, execution_units) in enumerate(service_specs, start=1):
                directory = tmp_path / kind
                directory.mkdir()
                recipe = live_recipe(
                    kind, directory, proposer, scorer, binary, model_name, base_url, api_key, execution_units
                )
                assert recipe.scenario_execution_units() == execution_units
                initial = directory / "initial"
                initial.mkdir()
                for relative, content in (recipe.base_artifact_files() or {}).items():
                    target = initial / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                dispatcher = Dispatcher(
                    recipe,
                    InMemoryRepositoryBackend.factory(initial, root=directory / "repository"),
                    agent_record_dir=directory / "agent-records",
                    scenario_storage=SQLiteScenarioStorage(directory / "records"),
                    require_scenario_admission=True,
                    scenario_admission_store=JSONScenarioAdmissionStore(directory / "assignments"),
                )
                port = unused_port()
                client = TestClient(
                    TestServer(
                        create_app(
                            dispatcher,
                            tokens="service-test-token",
                            scheduling=ServiceSchedulingConfig(
                                scheduler_url=str(scheduler_client.make_url("/")),
                                service_id=service_id,
                                service_name=kind,
                                recipe=recipe.name,
                                standard_execution_units=execution_units,
                                control_token="control-test-token",
                                service_url=f"http://127.0.0.1:{port}",
                                registration_interval_seconds=0.05,
                            ),
                            staged_model_config_directory=directory / "staged-models",
                        ),
                        port=port,
                    )
                )
                services.append(HarnessService(service_id, directory, recipe, dispatcher, client))
                await stack.enter_async_context(client)
            try:
                async with asyncio.timeout(30):
                    while not all(scheduler.availability.get(service.service_id, False) for service in services):
                        await asyncio.sleep(0.05)

                async def submit_evolve(scenario: TenantScenario) -> bool:
                    headers = {
                        "Authorization": "Bearer service-test-token",
                        "x-reef-tenant": str(scenario.tenant_id),
                        "x-reef-scenario": scenario.name,
                    }
                    try:
                        instance = scenario.service.dispatcher.loaded_scenario(scenario.name)
                    except ScenarioAdmissionError:
                        return False
                    if instance is None:
                        return False
                    assert instance.training_mode == "manual"
                    response = await scenario.service.client.post(
                        "/reef/train",
                        headers=headers,
                        json={
                            "agent_record_id": f"evolve-{scenario.name}",
                            "text": f"Always end every final answer with the exact text {REPLY_MARKER}. Keep greetings short and do not use tools.",
                            "session": scenario.name,
                            "release_id": instance.repository.require_current_artifact().release_id,
                        },
                    )
                    if response.status == 409:
                        return False
                    assert response.status == 200, await response.text()
                    scenario.request_id = text_value(object_value(await response.json()), "agent_record_id")
                    return True

                async def submit_tenant(tenant_id: int) -> None:
                    services_by_units = {service.recipe.scenario_execution_units(): service for service in services}
                    for index, execution_units in enumerate(submissions[tenant_id], start=1):
                        service = services_by_units[execution_units]
                        name = f"t{tenant_id}-s{index}"
                        response = await service.client.post(
                            "/reef/scenarios",
                            headers={"Authorization": "Bearer service-test-token", "x-reef-tenant": str(tenant_id)},
                            json={"name": name},
                        )
                        assert response.status in (201, 202), await response.text()
                        admission = object_value(await response.json())
                        scenario = TenantScenario(
                            tenant_id, index, integer_value(admission, "scenario_id"), name, service
                        )
                        scenarios[name] = scenario
                        assert admission["service_id"] == service.service_id
                        assert admission["recipe"] == service.recipe.name
                        assert admission["execution_units"] == execution_units
                        if name in initial_sessions:
                            async with asyncio.timeout(30):
                                while (
                                    scheduler.get_scenario_admission(scenario.scenario_id, tenant_id).state
                                    != "admitted"
                                ):
                                    await asyncio.sleep(0.05)
                            assert await submit_evolve(scenario)
                        else:
                            assert response.status == 202 and admission["state"] == "queued"
                            assert not service.dispatcher.has_scenario(name)

                # Tenant submissions overlap; each tenant retains its own FIFO order.
                await asyncio.gather(*(submit_tenant(tenant_id) for tenant_id in TENANT_IDS))
                assert await asyncio.to_thread(proposer.started.wait, 60), "all initial proposals must overlap"
                assert len(scenarios) == 15
                for tenant_id in TENANT_IDS:
                    waiting = [
                        entry for entry in scheduler.list_scenario_admissions(tenant_id) if entry.state == "queued"
                    ]
                    assert [entry.scenario_name for entry in waiting] == [
                        f"t{tenant_id}-s{index}"
                        for index in range(1, 6)
                        if f"t{tenant_id}-s{index}" not in initial_sessions
                    ]
                    assert controller.status(tenant_id)["used"] == initial_used[tenant_id]
                    if initial_used[tenant_id] < tenant_quota:
                        assert waiting[0].execution_units > tenant_quota - initial_used[tenant_id]
                        assert any(
                            entry.execution_units <= tenant_quota - initial_used[tenant_id] for entry in waiting
                        )
                        print(
                            f"Tenant {tenant_id}: FIFO head needs {waiting[0].execution_units}, "
                            f"available={tenant_quota - initial_used[tenant_id]}; smaller followers stay queued",
                            flush=True,
                        )
                    for scenario in scenarios.values():
                        if scenario.tenant_id != tenant_id or scenario.name not in initial_sessions:
                            continue
                        status = scenario.service.dispatcher.scheduled_scenario_status(scenario.name)
                        assert status is not None and status.idle_since is None
                    other_tenant = TENANT_IDS[tenant_id % len(TENANT_IDS)]
                    denied = await scheduler_client.get(
                        f"/reef/scenarios/{scenarios[f't{tenant_id}-s1'].scenario_id}",
                        headers={"x-reef-tenant": str(other_tenant)},
                    )
                    assert denied.status == 403
                proposer.release.set()

                async with asyncio.timeout(STEP_TIMEOUT_SECONDS):
                    while not all(scenario.committed for scenario in scenarios.values()):
                        _, proposal_failures = proposer.snapshot()
                        _, episode_failures = scorer.snapshot()
                        assert not proposal_failures and not episode_failures, (proposal_failures, episode_failures)
                        for scenario in scenarios.values():
                            admission = scheduler.get_scenario_admission(scenario.scenario_id, scenario.tenant_id)
                            if admission.state != "admitted":
                                continue
                            if scenario.request_id is None and not await submit_evolve(scenario):
                                continue
                            if scenario.committed:
                                continue
                            response = await scenario.service.client.get(
                                f"/reef/scenarios/{scenario.name}/releases",
                                headers={
                                    "Authorization": "Bearer service-test-token",
                                    "x-reef-tenant": str(scenario.tenant_id),
                                },
                            )
                            if response.status == 409:
                                continue
                            assert response.status == 200, await response.text()
                            rows = object_value(await response.json())["releases"]
                            assert isinstance(rows, list)
                            matched = []
                            for row in rows:
                                metrics = object_value(object_value(row).get("metrics", {}))
                                request = metrics.get("training_request")
                                if isinstance(request, dict) and request.get("id") == scenario.request_id:
                                    matched.append(metrics)
                            if matched:
                                assert len(matched) == 1 and matched[0]["steps"] == 1
                                scenario.committed = True
                                print(
                                    f"tenant={scenario.tenant_id} scenario={scenario.name} recipe={scenario.service.recipe.name} request={scenario.request_id} selected={matched[0]['selected']}",
                                    flush=True,
                                )
                        assert all(controller.status(tenant_id)["used"] <= tenant_quota for tenant_id in TENANT_IDS)
                        assert sum(
                            value.allocated_execution_units for value in scheduler.tenant_allocations()
                        ) == tenant_quota * len(TENANT_IDS)
                        await asyncio.sleep(0.05)

                counts, failures = proposer.snapshot()
                assert not failures and counts == Counter(scenario.request_id for scenario in scenarios.values())
                episode_count, failures = scorer.snapshot()
                assert not failures and episode_count == 30
                assert all(scenario.committed and scenario.request_id is not None for scenario in scenarios.values())
                events = trace.snapshot()
                costs = {
                    scenario.scenario_id: scenario.service.recipe.scenario_execution_units()
                    for scenario in scenarios.values()
                }
                names = {scenario.scenario_id: scenario.name for scenario in scenarios.values()}
                claim_counts: Counter[int] = Counter()
                transitions = trace.transition_snapshot()
                reclaim_counts = Counter(
                    event.current.scenario_id
                    for event in transitions
                    if event.previous.state == "reclaim_pending" and event.current.state == "sleep"
                )
                dequeue_delays: list[float] = []
                for tenant_id in TENANT_IDS:
                    owned = {
                        scenario.scenario_id for scenario in scenarios.values() if scenario.tenant_id == tenant_id
                    }
                    tenant_events = [event for event in events if event.tenant_id == tenant_id]
                    assert tenant_events and trace.paths[tenant_id].stat().st_size > 0
                    assert all(0 <= event.used_execution_units <= tenant_quota for event in tenant_events)
                    assert all(
                        set(event.queue_before + event.queue + event.allocated) <= owned for event in tenant_events
                    )
                    queue: tuple[int, ...] = ()
                    head_since: dict[int, float] = {}
                    for event in tenant_events:
                        assert event.queue_before == queue
                        current_allocated = set(event.allocated)
                        claimed = set(queue) - set(event.queue)
                        assert event.used_execution_units == sum(costs[identifier] for identifier in current_allocated)
                        if claimed:
                            assert len(claimed) == 1
                            identifier = next(iter(claimed))
                            assert queue[0] == identifier and event.queue == queue[1:]
                            assert identifier in current_allocated
                            claim_counts[identifier] += 1
                            dequeue_delays.append(event.monotonic_seconds - head_since.pop(identifier))
                        else:
                            assert event.queue[:-1] == queue
                        queue = event.queue
                        if queue:
                            head_since.setdefault(queue[0], event.monotonic_seconds)
                    assert all(claim_counts[identifier] >= 1 for identifier in owned)
                    assert any(reclaim_counts[identifier] >= 1 for identifier in owned)
                    print(
                        f"Tenant {tenant_id}: 5 evolves settled once, quota={tenant_quota}, weighted FIFO verified; "
                        f"starts/reclaims={[(names[identifier], claim_counts[identifier], reclaim_counts[identifier]) for identifier in sorted(owned)]}; "
                        f"trace={trace.paths[tenant_id]}",
                        flush=True,
                    )

                transitions = trace.transition_snapshot()
                starting: dict[tuple[int, int], float] = {}
                reclaiming: dict[tuple[int, int], float] = {}
                start_delays: list[float] = []
                reclaim_delays: list[float] = []
                for event in transitions:
                    current = event.current
                    assert 0 <= event.used_execution_units <= tenant_quota
                    key = (current.scenario_id, current.generation)
                    if current.state == "starting":
                        assert current.execution_allocated
                        starting[key] = event.monotonic_seconds
                    elif current.state == "admitted" and event.previous.state == "starting":
                        assert current.execution_allocated
                        start_delays.append(event.monotonic_seconds - starting.pop(key))
                    elif current.state == "reclaim_pending":
                        assert event.previous.execution_allocated and current.execution_allocated
                        assert event.used_execution_units >= current.execution_units
                        reclaiming[key] = event.monotonic_seconds
                    elif current.state == "sleep" and event.previous.state == "reclaim_pending":
                        assert event.previous.execution_allocated and not current.execution_allocated
                        reclaim_delays.append(event.monotonic_seconds - reclaiming.pop(key))

                for service in services:
                    records = list((service.directory / "records").glob("*.commits.jsonl"))
                    expected = [scenario for scenario in scenarios.values() if scenario.service is service]
                    assert len(records) == len(expected)
                    for path in records:
                        rows = path.read_text().splitlines()
                        assert len(rows) == 1
                        record = object_value(json.loads(rows[0]))
                        scenario = scenarios[text_value(record, "scenario")]
                        metrics = object_value(record["metrics"])
                        assert text_value(object_value(metrics["training_request"]), "id") == scenario.request_id
                        assert object_value(record["record_progress"])["consumed_ids"] == [scenario.request_id]
                        assert metrics["steps"] == object_value(record["algorithm_state"])["steps"] == 1

                latencies: dict[str, object] = {}
                for operation, delays in (
                    ("fifo_head_to_dequeue", dequeue_delays),
                    ("starting_to_admitted", start_delays),
                    ("reclaim_pending_to_sleep", reclaim_delays),
                ):
                    assert delays and all(delay >= 0 for delay in delays)
                    ordered = sorted(delay * 1000 for delay in delays)
                    latencies[operation] = {
                        "count": len(ordered),
                        "p50_ms": statistics.median(ordered),
                        "p95_ms": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
                        "max_ms": ordered[-1],
                    }
                summary = {
                    "tenant_quota": tenant_quota,
                    "global_budget": tenant_quota * len(TENANT_IDS),
                    "service_units": {
                        service.recipe.name: service.recipe.scenario_execution_units() for service in services
                    },
                    "submissions": submissions,
                    "evolve_count": len(scenarios),
                    "episode_count": episode_count,
                    "latencies": latencies,
                    "scenarios": {
                        scenario.name: {
                            "execution_units": costs[scenario.scenario_id],
                            "starts": claim_counts[scenario.scenario_id],
                            "reclaims": reclaim_counts[scenario.scenario_id],
                            "request_id": scenario.request_id,
                        }
                        for scenario in scenarios.values()
                    },
                }
                (tmp_path / "scheduling-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"Scheduling latency observations (milliseconds): {latencies}", flush=True)
                assert controller.maintenance_error is None and not controller.errors
            finally:
                proposer.release.set()
                if isinstance(scorer, LiveTenantScorer):
                    scorer.barrier.abort()

    try:
        asyncio.run(run())
    finally:
        proposer.release.set()
        if isinstance(scorer, LiveTenantScorer):
            scorer.barrier.abort()
        for service in services:
            service.dispatcher.close()
        if not controller.closed:
            controller.close()
        logger.removeHandler(trace)
        logger.setLevel(previous_level)
        trace.close()


def test_concurrent_tenants_share_one_cordis_service(tmp_path: Path) -> None:
    run_concurrent_tenants(
        tmp_path, (("cordis", 1),), tenant_quota=3, submissions=dict.fromkeys(TENANT_IDS, (1, 1, 1, 1, 1))
    )


def test_concurrent_tenants_share_three_harness_services(tmp_path: Path) -> None:
    run_concurrent_tenants(
        tmp_path,
        (("reefine", 2), ("gepa", 4), ("meta_harness", 6)),
        tenant_quota=6,
        submissions={1: (4, 4, 2, 2, 6), 2: (2, 4, 4, 2, 6), 3: (6, 4, 2, 2, 4)},
    )
