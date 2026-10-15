"""Measure scheduler overhead with paired real HTTP workloads and durable storage.

Run with the conda reef Python using ``-m tests.reef_service.benchmark_scheduler_overhead``.
No model endpoint or credentials are needed. Timing results are measurements,
not pytest assertions; the same workload runs with scheduling disabled/enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import platform
import statistics
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, unused_port
from aiohttp.typedefs import Handler
from tests.reef_service.test_multi_tenant_service import (
    CONTROL_TOKEN,
    SCHEDULER_TOKEN,
    SERVICE_TOKEN,
    service_dispatcher,
    tenant_headers,
)

from reef.artifact import Artifact
from reef.dispatcher import Dispatcher
from reef.runtime.interfaces import InferenceHandler
from reef.scenario.scheduling import MultiTenantScheduler, SchedulerConfig, ServiceRegistration
from reef.service.app import create_app
from reef.service.auth import create_authentication_middleware
from reef.service.errors import translate_errors
from reef.service.routes.scheduling import register_scheduler_routes
from reef.service.scheduler import create_scheduler_app
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import ServiceSchedulingConfig
from reef.storage.scheduling import ScenarioAdmission, SchedulingSnapshot, TenantAllocation
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore


@dataclass
class Measurements:
    """Raw request latencies and combined client/service/scheduler process cost."""

    wall_seconds: float
    cpu_seconds: float
    latency_ms: list[float]

    def summary(self) -> dict[str, float | int]:
        ordered = sorted(self.latency_ms)
        return {
            "count": len(ordered),
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "cpu_percent_of_one_core": 100 * self.cpu_seconds / self.wall_seconds,
            "operations_per_second": len(ordered) / self.wall_seconds,
            "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "p99_ms": ordered[math.ceil(0.99 * len(ordered)) - 1],
        }


class FixedDurationInference(InferenceHandler):
    """Keep provider time and output identical in each paired run."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
        if self.duration_seconds:
            await asyncio.sleep(self.duration_seconds)
        return {"choices": [{"message": {"role": "assistant", "content": "benchmark response"}}]}


@dataclass
class BenchmarkCluster:
    controller: SchedulerController | None = None
    scheduler_client: TestClient | None = None
    services: list[TestClient] = field(default_factory=list)
    dispatchers: list[Dispatcher] = field(default_factory=list)
    control_requests: dict[str, int] = field(default_factory=dict)
    background_maintenance: bool = True
    setup_reconciliations: int = 0

    @web.middleware
    async def count_control(self, request: web.Request, handler: Handler) -> web.StreamResponse:
        if request.path == "/reef/scheduler/control":
            payload = await request.json()
            operation = payload["operation"]
            self.control_requests[operation] = self.control_requests.get(operation, 0) + 1
        return await handler(request)


@asynccontextmanager
async def benchmark_cluster(
    directory: Path,
    *,
    scheduled: bool,
    scenario_count: int,
    service_count: int,
    duration_seconds: float,
    background_maintenance: bool = True,
) -> AsyncIterator[BenchmarkCluster]:
    directory.mkdir(parents=True)
    stack = BenchmarkCluster(background_maintenance=background_maintenance)
    try:
        if scheduled:
            quota = math.ceil(scenario_count / 3) * 2
            config = SchedulerConfig(
                quota * 3,
                dict.fromkeys((1, 2, 3), quota),
                {"recipe": 2},
                # Scan idle state normally, but prevent benchmark setup from triggering eviction.
                idle_ttl_seconds=3600,
                database=directory / "scheduler.sqlite3",
            )
            stack.controller = SchedulerController(
                MultiTenantScheduler(SQLiteSchedulingStore(config.database), config), {}
            )
            if background_maintenance:
                scheduler_app = create_scheduler_app(stack.controller, token=SCHEDULER_TOKEN)
            else:
                # Time complete scans without the periodic task racing the explicit scan.
                scheduler_app = web.Application(
                    middlewares=[create_authentication_middleware(SCHEDULER_TOKEN), translate_errors]
                )
                register_scheduler_routes(scheduler_app, stack.controller)
            stack.scheduler_client = TestClient(TestServer(scheduler_app))
            await stack.scheduler_client.start_server()
        for index in range(service_count):
            name = f"service-{index}"
            port = unused_port()
            dispatcher = service_dispatcher(directory / name, scheduled=scheduled)
            stack.dispatchers.append(dispatcher)
            if stack.scheduler_client is not None:
                scheduling = ServiceSchedulingConfig(
                    scheduler_url=str(stack.scheduler_client.make_url("/")),
                    service_id=index + 1,
                    service_name=name,
                    recipe="recipe",
                    standard_execution_units=2,
                    control_token=CONTROL_TOKEN,
                    scheduler_token=SCHEDULER_TOKEN,
                    service_url=f"http://127.0.0.1:{port}",
                    registration_interval_seconds=5 if background_maintenance else 3600,
                )
            else:
                scheduling = None
            app = create_app(
                dispatcher,
                tokens=SERVICE_TOKEN,
                inference_handler=FixedDurationInference(duration_seconds),
                scheduling=scheduling,
                staged_model_config_directory=directory / name / "model-settings",
            )
            app.middlewares.append(stack.count_control)
            service = TestClient(TestServer(app, port=port))
            stack.services.append(service)
            await service.start_server()
        if stack.controller is not None:
            async with asyncio.timeout(10):
                while len(stack.controller.scheduler.services) != service_count or not all(
                    stack.controller.scheduler.availability.values()
                ):
                    await asyncio.sleep(0.01)
        yield stack
    finally:
        if stack.scheduler_client is not None:
            await stack.scheduler_client.close()
        if not background_maintenance and stack.controller is not None:
            await asyncio.to_thread(stack.controller.close, save_snapshot=False)
        for service in stack.services:
            await service.close()
        for dispatcher in stack.dispatchers:
            await asyncio.to_thread(dispatcher.close)


async def scenario_operations(stack: BenchmarkCluster, scenario_count: int, *, delete: bool) -> Measurements:
    latencies = []
    started = time.perf_counter()
    cpu_started = time.process_time()
    for index in range(scenario_count):
        name = f"scenario-{index}"
        service_id = (index // 3 + index % 3) % len(stack.services) + 1
        service = stack.services[service_id - 1]
        request_started = time.perf_counter()
        async with asyncio.timeout(30):
            if delete:
                response = await service.delete(f"/reef/scenarios/{name}", headers=tenant_headers(index % 3 + 1))
                allowed_statuses = (200, 202)
            else:
                response = await service.post(
                    "/reef/scenarios", json={"name": name}, headers=tenant_headers(index % 3 + 1)
                )
                allowed_statuses = (201, 202)
            payload = await response.json()
            if response.status not in allowed_statuses:
                raise RuntimeError(f"scenario operation failed: {response.status}, {payload}")
            if stack.controller is not None:
                expected = "deleted" if delete else "admitted"
                while stack.controller.scheduler.find_latest_scenario_admission(service_id, name).state != expected:
                    if not stack.background_maintenance:
                        # The isolated scan fixture still needs lifecycle recovery during setup.
                        stack.setup_reconciliations += 1
                        await asyncio.to_thread(stack.controller.reconcile)
                    await asyncio.sleep(0.001)
                    if stack.controller.maintenance_error is not None:
                        raise RuntimeError(stack.controller.maintenance_error)
        latencies.append((time.perf_counter() - request_started) * 1000)
    return Measurements(time.perf_counter() - started, time.process_time() - cpu_started, latencies)


async def inference_requests(
    stack: BenchmarkCluster, scenario_count: int, request_count: int, concurrency: int
) -> Measurements:
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def request(index: int) -> None:
        async with semaphore:
            scenario_index = index % scenario_count
            started = time.perf_counter()
            service_index = (scenario_index // 3 + scenario_index % 3) % len(stack.services)
            response = await stack.services[service_index].post(
                "/v1/chat/completions",
                json={"model": "benchmark", "messages": [{"role": "user", "content": "fixed workload"}]},
                headers={**tenant_headers(scenario_index % 3 + 1), "x-reef-scenario": f"scenario-{scenario_index}"},
            )
            payload = await response.json()
            if response.status != 200 or payload["choices"][0]["message"]["content"] != "benchmark response":
                raise RuntimeError(f"inference failed: {response.status}, {payload}")
            latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    cpu_started = time.process_time()
    async with asyncio.timeout(max(60, request_count * 2)):
        await asyncio.gather(*(request(index) for index in range(request_count)))
    return Measurements(time.perf_counter() - started, time.process_time() - cpu_started, latencies)


async def http_trial(
    directory: Path,
    *,
    scheduled: bool,
    duration_ms: int,
    concurrency: int,
    scenario_count: int,
    request_count: int,
    service_count: int,
) -> dict[str, object]:
    async with benchmark_cluster(
        directory,
        scheduled=scheduled,
        scenario_count=scenario_count,
        service_count=service_count,
        duration_seconds=duration_ms / 1000,
    ) as stack:
        created = await scenario_operations(stack, scenario_count, delete=False)
        # Establish connections and exercise the complete record path before timed serving.
        await inference_requests(stack, scenario_count, scenario_count, concurrency)
        served = await inference_requests(stack, scenario_count, request_count, concurrency)
        if stack.controller is not None:
            if stack.controller.maintenance_error is not None or stack.controller.errors:
                raise RuntimeError("scheduler maintenance or control reported an error")
            if len(stack.controller.scheduler.list_scenario_admissions()) != scenario_count:
                raise RuntimeError("benchmark did not keep the expected scenario count")
        deleted = await scenario_operations(stack, scenario_count, delete=True)
        if any(
            dispatcher.has_loaded(f"scenario-{index}")
            for dispatcher in stack.dispatchers
            for index in range(scenario_count)
        ):
            raise RuntimeError("cleanup left a loaded scenario")
        return {
            "scheduled": scheduled,
            "duration_ms": duration_ms,
            "concurrency": concurrency,
            "scenario_count": scenario_count,
            "request_count": request_count,
            "service_count": service_count,
            "create": asdict(created),
            "serve": asdict(served),
            "delete": asdict(deleted),
            "lifecycle_seconds": created.wall_seconds + served.wall_seconds + deleted.wall_seconds,
            "control_requests": dict(stack.control_requests),
        }


async def reconcile_trial(directory: Path, scenario_count: int, service_count: int) -> dict[str, object]:
    async with benchmark_cluster(
        directory,
        scheduled=True,
        scenario_count=scenario_count,
        service_count=service_count,
        duration_seconds=0,
        background_maintenance=False,
    ) as stack:
        created = await scenario_operations(stack, scenario_count, delete=False)
        controller = stack.controller
        if controller is None:
            raise RuntimeError("reconcile measurement requires scheduling")
        # Warm the idle observations so stable scans do not include first observation writes.
        await asyncio.to_thread(controller.reconcile)
        latencies = []
        call_counts = []
        cpu_started = time.process_time()
        started = time.perf_counter()
        for _ in range(10):
            before = dict(stack.control_requests)
            scan_started = time.perf_counter()
            await asyncio.to_thread(controller.reconcile)
            latencies.append((time.perf_counter() - scan_started) * 1000)
            status_calls = stack.control_requests.get("status", 0) - before.get("status", 0)
            describe_calls = stack.control_requests.get("describe", 0) - before.get("describe", 0)
            if status_calls != scenario_count or describe_calls != service_count:
                raise RuntimeError("measurement requires one complete uncontended scan")
            call_counts.append(status_calls + describe_calls)
        measured = Measurements(time.perf_counter() - started, time.process_time() - cpu_started, latencies)
        return {
            "scenario_count": scenario_count,
            "service_count": service_count,
            "create": asdict(created),
            "reconcile": asdict(measured),
            "control_calls_per_scan": call_counts,
            "setup_reconciliations": stack.setup_reconciliations,
            "control_requests": dict(stack.control_requests),
            "errors": dict(controller.errors),
        }


def core_trial(directory: Path, scenario_count: int, *, durable: bool) -> dict[str, object]:
    """Measure changed and unchanged idle observations at fixed active-set sizes."""
    directory.mkdir(parents=True)
    database = directory / "scheduler.sqlite3" if durable else ":memory:"
    store = SQLiteSchedulingStore(database)
    registration = ServiceRegistration(1, "benchmark", "recipe", "http://127.0.0.1", 1)
    # Seeding is outside timing and avoids confusing population cost with operation cost.
    scenarios = tuple(
        ScenarioAdmission(
            scenario_id=index + 1,
            tenant_id=1,
            service_id=1,
            recipe="recipe",
            scenario_name=f"scenario-{index}",
            release_id=None,
            model_config_ref=None,
            execution_units=1,
            state="admitted",
            generation=1,
            created_at=1,
            service_config_ref=registration.registration_fingerprint,
            queue_sequence=index + 1,
            has_started=True,
            execution_allocated=True,
        )
        for index in range(scenario_count)
    )
    store.save(
        SchedulingSnapshot(
            0, scenario_count + 1, (TenantAllocation(1, scenario_count),), scenarios, scenario_count + 1
        )
    )
    scheduler = MultiTenantScheduler(store, SchedulerConfig(scenario_count, {1: scenario_count}, {"recipe": 1}))
    try:
        measured = {}
        for changed in (False, True):
            latencies = []
            started = time.perf_counter()
            cpu_started = time.process_time()
            for index in range(200):
                if changed:
                    idle_since = float(index + 1)
                else:
                    idle_since = None
                operation_started = time.perf_counter()
                scheduler.report_scenario_idle(1, 1, idle_since=idle_since)
                latencies.append((time.perf_counter() - operation_started) * 1000)
            measured["changed" if changed else "unchanged"] = asdict(
                Measurements(time.perf_counter() - started, time.process_time() - cpu_started, latencies)
            )
        return {"scenario_count": scenario_count, "durable": durable, "measurements": measured}
    finally:
        scheduler.close()


async def run_benchmark(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "settings": {"repetitions": args.repetitions, "request_count": args.requests, "target_overhead_percent": 5},
        "http": [],
        "reconcile": [],
        "core": [],
    }
    http_results = []
    reconcile_results = []
    core_results = []
    results.update(http=http_results, reconcile=reconcile_results, core=core_results)
    workloads = [(0, 1), (0, 8), (10, 8), (100, 8), (1000, 8)]
    if args.smoke:
        workloads = [(0, 2)]
    for service_count in (1, 3):
        for duration_ms, concurrency in workloads:
            for repetition in range(args.repetitions):
                # Alternate order to reduce warm-cache and host-drift bias.
                for scheduled in (False, True) if repetition % 2 == 0 else (True, False):
                    label = f"http-services{service_count}-delay{duration_ms}-workers{concurrency}-repeat{repetition}-scheduled{scheduled}"
                    trial = await http_trial(
                        args.output / label,
                        scheduled=scheduled,
                        duration_ms=duration_ms,
                        concurrency=concurrency,
                        scenario_count=3 if args.smoke else 15,
                        request_count=args.requests,
                        service_count=service_count,
                    )
                    trial["repetition"] = repetition
                    http_results.append(trial)
                    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                    print(f"{label}: lifecycle={trial['lifecycle_seconds']:.3f}s", flush=True)
    for service_count in (1, 3):
        for scenario_count in (3,) if args.smoke else (15, 60, 150):
            label = f"reconcile-services{service_count}-scenarios{scenario_count}"
            trial = await reconcile_trial(args.output / label, scenario_count, service_count)
            reconcile_results.append(trial)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{label}: complete", flush=True)
    for durable in (False, True):
        for scenario_count in (3,) if args.smoke else (15, 100, 1000, 10000):
            label = f"core-durable{durable}-scenarios{scenario_count}"
            core_results.append(core_trial(args.output / label, scenario_count, durable=durable))
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{label}: complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Use a new directory for each benchmark run")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.requests < 1:
        parser.error("repetitions and requests must be positive")
    logging.basicConfig(level=logging.WARNING)
    # Include production INFO queue serialization and disk output in the benchmark.
    args.output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.output / "queue.log")
    scheduler_logger = logging.getLogger("reef.scenario.scheduling")
    scheduler_logger.setLevel(logging.INFO)
    scheduler_logger.addHandler(handler)
    scheduler_logger.propagate = False
    try:
        asyncio.run(run_benchmark(args))
    finally:
        scheduler_logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
