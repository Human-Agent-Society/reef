"""Opt-in tenant recycling with real Pi episodes and a remote OpenAI-compatible model.

Set REEF_RUN_LIVE_HARNESS=1 and REEF_LIVE_MODEL_API_KEY before running.
REEF_REAL_PI_BINARY selects the installed Pi executable (default: pi).
REEF_LIVE_MODEL_NAME overrides single-model discovery from /v1/models.
Every committed queue change is recorded in the test directory's tenant-queue.log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from threading import Barrier, Event, Lock
from urllib.request import Request, urlopen

import pytest
from aiohttp.test_utils import TestClient, TestServer, unused_port
from reef_service.test_harness_recipe import SEED_MODELS, SEED_SETTINGS

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
from reef.service.scheduling import ServiceSchedulingConfig
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.scheduling import ScenarioAdmission
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.strategies import EpisodeScorer, resolve_proposer
from reef.train.types import TrajectoryItem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("REEF_RUN_LIVE_HARNESS") != "1", reason="live model test requires explicit opt-in"
    ),
]

REPLY_MARKER = "REEF-SCHEDULER-EVOLVED"
STEP_TIMEOUT_SECONDS = 3600


@pytest.fixture
def queue_log(tmp_path: Path) -> Iterator[Path]:
    """Keep timestamped scheduler queue events even when pytest live logging is disabled."""
    path = tmp_path / "tenant-queue.log"
    logger = logging.getLogger("reef.scenario.scheduling")
    previous_level = logger.level
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        print(f"Tenant queue trace: {path}")
        yield path
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)


class ControlledReclaimScheduler(MultiTenantScheduler):
    """Let the ordered-rotation test request deadlines explicitly while real inference runs."""

    def reclaim_idle_scenarios(self, *, now_seconds: float | None = None) -> tuple[ScenarioAdmission, ...]:
        return ()


class LiveReplyScorer(EpisodeScorer):
    """Score actual replies and synchronize the first three scenarios' evaluation rounds."""

    def __init__(self) -> None:
        self.round_finished = Barrier(3, timeout=STEP_TIMEOUT_SECONDS)
        self.lock = Lock()
        self.episode_count = 0

    def __call__(self, task: str, result: EpisodeResult) -> float:
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0 or not reply:
            self.round_finished.abort()
            raise ValueError("real Pi evaluation must exit successfully and produce an assistant reply")
        with self.lock:
            self.episode_count += 1
            synchronize = self.episode_count <= 6
        # Two rounds of three episodes keep slow inference from making one scenario idle much earlier.
        if synchronize:
            self.round_finished.wait()
        return float(REPLY_MARKER in reply)


@pytest.mark.parametrize("real_time_ttl", [False, True], ids=["sleep-and-wake", "wall-clock-10s"])
def test_three_harness_scenarios_evolve_then_recycle_for_fourth_at_ten_second_idle_ttl(
    tmp_path: Path, real_time_ttl: bool, queue_log: Path
) -> None:
    api_key = os.environ.get("REEF_LIVE_MODEL_API_KEY", "").strip()
    if not api_key:
        pytest.fail("set REEF_LIVE_MODEL_API_KEY before enabling the live model test")
    binary = shutil.which(os.environ.get("REEF_REAL_PI_BINARY", "pi"))
    if binary is None:
        pytest.fail("install real Pi and set REEF_REAL_PI_BINARY to its executable")
    base_url = os.environ.get("REEF_LIVE_MODEL_URL", "http://192.168.1.119:30000").rstrip("/")
    model_name = os.environ.get("REEF_LIVE_MODEL_NAME", "").strip()
    if not model_name:
        request = Request(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        with urlopen(request, timeout=30) as response:
            catalog = json.load(response)
        if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
            pytest.fail("model endpoint must return a data array from /v1/models")
        names = [entry.get("id") for entry in catalog["data"] if isinstance(entry, dict)]
        if len(names) != 1 or not isinstance(names[0], str) or not names[0].strip():
            pytest.fail("set REEF_LIVE_MODEL_NAME when /v1/models does not identify exactly one model")
        model_name = names[0]
    first_three_running = Event()
    finish_proposals = Event()
    proposal_lock = Lock()
    proposed_requests: list[str] = []
    running_sessions: set[str] = set()

    def propose(
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TrajectoryItem, ...],
        models: ModelBindings,
        *,
        requests: Sequence[Mapping[str, object]] = (),
    ) -> Mutation:
        assert samples == () and len(requests) == 1
        request = requests[0]
        assert isinstance(request["id"], str) and isinstance(request["session"], str)
        with proposal_lock:
            proposed_requests.append(request["id"])
            running_sessions.add(request["session"])
            if {"s1", "s2", "s3"} <= running_sessions:
                first_three_running.set()
        if request["session"] in {"s1", "s2", "s3"}:
            assert finish_proposals.wait(120), "test did not release the concurrent proposals"
        rules = models.served.chat(
            [
                {
                    "role": "system",
                    "content": "Write a short AGENTS.md rule implementing the requested behavior. "
                    "Return only the rule text, without markdown fences. Do not execute the task or use tools.",
                },
                {"role": "user", "content": str(request["text"])},
            ],
            temperature=0,
            max_tokens=512,
            timeout_s=900,
        ).strip()
        assert rules, "the live proposer returned no rule text"
        return Mutation("create", "requested-rules", {"name": "rules", "config": {"text": rules}})

    scorer = LiveReplyScorer()
    recipe = CordisRecipe(
        resolve_proposer(propose),
        scorer,
        ("Say hello in one short sentence. Do not use tools or inspect files.",),
        binary=binary,
        seed=(SEED_MODELS, SEED_SETTINGS),
        runtime=InferenceProxyRuntime(
            model_path=model_name, base_url=base_url, api_key=api_key, inference_timeout_s=900
        ),
        executor=LocalExecutor(),
        episode_timeout_s=900,
        training_mode="manual",
        proposals_dir=str(tmp_path / "proposals"),
        episode_workers=1,
    )
    assert recipe.scenario_execution_units() == 1
    initial = tmp_path / "initial"
    initial.mkdir()
    for relative, text in (recipe.base_artifact_files() or {}).items():
        target = initial / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        agent_record_dir=tmp_path / "agent-records",
        require_scenario_admission=True,
        scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
    )
    config = SchedulerConfig(3, {1: 3}, idle_ttl_seconds=10)
    if real_time_ttl:
        scheduler = MultiTenantScheduler(SQLiteSchedulingStore(tmp_path / "scheduler.sqlite3"), config)
    else:
        scheduler = ControlledReclaimScheduler(SQLiteSchedulingStore(tmp_path / "scheduler.sqlite3"), config)
    controller = SchedulerController(scheduler, {})

    async def run() -> None:
        scheduler_client = TestClient(TestServer(create_scheduler_app(controller, token="scheduler-test-token")))
        await scheduler_client.start_server()
        service_port = unused_port()
        service = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    tokens="service-test-token",
                    scheduling=ServiceSchedulingConfig(
                        scheduler_url=str(scheduler_client.make_url("/")),
                        service_id=1,
                        service_name="local-harness",
                        recipe=recipe.name,
                        standard_execution_units=recipe.scenario_execution_units(),
                        control_token="control-test-token",
                        scheduler_token="scheduler-test-token",
                        service_url=f"http://127.0.0.1:{service_port}",
                        registration_interval_seconds=0.05,
                    ),
                    staged_model_config_directory=tmp_path / "staged-models",
                ),
                port=service_port,
            )
        )
        headers = {"Authorization": "Bearer service-test-token", "x-reef-tenant": "1"}
        admissions = {}
        request_ids = {}
        initial_releases = {}
        committed_releases = {}
        original_instances = {}

        async def wait_for_idle_commit(name: str) -> None:
            async with asyncio.timeout(STEP_TIMEOUT_SECONDS):
                while True:
                    response = await service.get(f"/reef/scenarios/{name}/releases", headers=headers)
                    assert response.status == 200, await response.text()
                    rows = (await response.json())["releases"]
                    steps = [row for row in rows if row.get("metrics", {}).get("training_request")]
                    status = dispatcher.scheduled_scenario_status(name)
                    if steps and status is not None and status.idle_since is not None:
                        assert len(steps) == 1
                        metrics = steps[0]["metrics"]
                        assert metrics["training_request"]["id"] == request_ids[name]
                        assert metrics["published"] is True and metrics["selected"] is True
                        assert metrics["steps"] == 1 and metrics["wins"] == 1
                        scenario = dispatcher.loaded_scenario(name)
                        assert scenario is not None
                        committed_releases[name] = scenario.repository.require_current_artifact().release_id
                        assert committed_releases[name] != initial_releases[name]
                        return
                    await asyncio.sleep(0.01)

        async def submit_step(name: str) -> None:
            scenario = dispatcher.loaded_scenario(name)
            assert scenario is not None and scenario.training_mode == "manual"
            initial_releases[name] = scenario.repository.require_current_artifact().release_id
            response = await service.post(
                "/reef/train",
                headers={**headers, "x-reef-scenario": name},
                json={
                    "text": f"Always end every final answer with the exact text {REPLY_MARKER}. "
                    "Keep replies short and do not use tools for simple greetings.",
                    "session": name,
                    "release_id": initial_releases[name],
                },
            )
            assert response.status == 200, await response.text()
            request_ids[name] = (await response.json())["agent_record_id"]

        try:
            await service.start_server()
            async with asyncio.timeout(5):
                while not scheduler.availability.get(1, False):
                    await asyncio.sleep(0.01)
            for name in ("s1", "s2", "s3", "s4"):
                response = await service.post("/reef/scenarios", json={"name": name}, headers=headers)
                assert response.status in (201, 202), await response.text()
                admissions[name] = await response.json()
                if name != "s4":
                    # A concurrent maintenance pass can leave creation pending even when quota is available.
                    async with asyncio.timeout(10):
                        while admissions[name]["state"] != "admitted":
                            response = await scheduler_client.get(
                                f"/reef/scenarios/{admissions[name]['scenario_id']}",
                                headers={"Authorization": "Bearer scheduler-test-token", "x-reef-tenant": "1"},
                            )
                            assert response.status == 200, await response.text()
                            admissions[name] = await response.json()
                            assert admissions[name]["state"] in {"queued", "starting", "admitted"}
                            await asyncio.sleep(0.05)
                    original_instances[name] = dispatcher.loaded_scenario(name)
                else:
                    assert response.status == 202
            assert controller.status(1)["used"] == 3
            assert admissions["s4"]["state"] == "queued"
            with scheduler.lock:
                fourth_id = admissions["s4"]["scenario_id"]
                assert scheduler.waiting_by_tenant[1] == (fourth_id,)
                fourth_admission = scheduler.get_scenario_admission(fourth_id, 1)
                assert fourth_admission.scenario_name == "s4" and not fourth_admission.execution_allocated
                persisted_queue = [
                    value.scenario_id
                    for value in scheduler.store.load().scenarios
                    if value.tenant_id == 1 and value.state == "queued"
                ]
                assert persisted_queue == [fourth_id]
            queue_events = [
                json.loads(line.split("tenant_queue_changed ", 1)[1])
                for line in queue_log.read_text().splitlines()
                if "tenant_queue_changed " in line
            ]
            assert any(
                event["tenant_id"] == 1
                and event["queue"] == [{"scenario_id": fourth_id, "service_id": 1, "scenario_name": "s4"}]
                for event in queue_events
            ), "s4 must appear in the persisted tenant queue trace"
            assert not dispatcher.has_scenario("s4")
            assert dispatcher.scheduled_scenario_status("s4") is None

            await asyncio.gather(*(submit_step(name) for name in ("s1", "s2", "s3")))
            assert await asyncio.to_thread(first_three_running.wait, 60), "all three proposals must run concurrently"
            for name in ("s1", "s2", "s3"):
                status = dispatcher.scheduled_scenario_status(name)
                assert status is not None and status.idle_since is None
                if not real_time_ttl:
                    scheduler.report_scenario_idle(status.scenario_id, status.generation, idle_since=None)
                    assert (
                        scheduler.request_reclaim(status.scenario_id, now_seconds=time.time() + 11).state == "admitted"
                    )
            assert controller.status(1)["used"] == 3
            assert not dispatcher.has_loaded("s4")
            finish_proposals.set()
            await asyncio.gather(*(wait_for_idle_commit(name) for name in ("s1", "s2", "s3")))

            if real_time_ttl:
                idle_starts = []
                for name in ("s1", "s2", "s3"):
                    status = dispatcher.scheduled_scenario_status(name)
                    assert status is not None and status.idle_since is not None
                    idle_starts.append(status.idle_since)
                async with asyncio.timeout(15):
                    while True:
                        response = await scheduler_client.get(
                            "/reef/scheduler/status",
                            headers={"Authorization": "Bearer scheduler-test-token", "x-reef-tenant": "1"},
                        )
                        assert response.status == 200, await response.text()
                        usage = await response.json()
                        assert usage["allocated"] == 3 and usage["used"] <= 3
                        fourth = next(value for value in usage["scenarios"] if value["scenario_name"] == "s4")
                        if fourth["state"] == "admitted":
                            break
                        await asyncio.sleep(0.05)
                elapsed = time.time() - min(idle_starts)
                assert elapsed >= 10
                assert dispatcher.has_loaded("s4")
                final = scheduler.list_scenario_admissions(1)
                recycled = [value for value in final if value.scenario_name != "s4" and value.generation > 1]
                assert recycled
                for value in final:
                    if value.scenario_name != "s4" and value.state == "admitted":
                        restored = dispatcher.loaded_scenario(value.scenario_name)
                        assert restored is not None
                        assert (
                            restored.repository.require_current_artifact().release_id
                            == committed_releases[value.scenario_name]
                        )
                        await wait_for_idle_commit(value.scenario_name)
                assert Counter(proposed_requests) == Counter(request_ids.values())
                queued = [value.scenario_name for value in final if value.state == "queued"]
                print(f"Real idle TTL: s4 admitted after {elapsed:.2f}s; used={usage['used']}/3; queued={queued}")
                return

            for reclaimed_name in ("s1", "s2", "s3", "s4"):
                waiting = [
                    value.scenario_name for value in scheduler.list_scenario_admissions(1) if value.state == "queued"
                ]
                assert waiting == (["s4"] if reclaimed_name == "s1" else [])
                status = dispatcher.scheduled_scenario_status(reclaimed_name)
                assert status is not None and status.idle_since is not None
                scheduler.report_scenario_idle(status.scenario_id, status.generation, idle_since=status.idle_since)
                before_deadline = scheduler.request_reclaim(status.scenario_id, now_seconds=status.idle_since + 9.999)
                assert before_deadline.state == "admitted"
                pending = scheduler.request_reclaim(status.scenario_id, now_seconds=status.idle_since + 10)
                assert pending.state == "reclaim_pending" and pending.execution_allocated
                async with asyncio.timeout(5):
                    while True:
                        await asyncio.to_thread(controller.reconcile)
                        reclaimed = scheduler.get_scenario_admission(status.scenario_id, 1)
                        if reclaimed.state == "sleep":
                            break
                        await asyncio.sleep(0.01)
                assert not reclaimed.execution_allocated
                assert reclaimed.scenario_id == admissions[reclaimed_name]["scenario_id"]
                assert scheduler.cold_scenarios_by_id[reclaimed.scenario_id] == reclaimed
                assert not dispatcher.has_loaded(reclaimed_name) and dispatcher.has_scenario(reclaimed_name)
                if reclaimed_name == "s1":
                    await submit_step("s4")
                    await wait_for_idle_commit("s4")

            assert controller.status(1)["used"] == 0
            assert len(scheduler.cold_scenarios_by_id) == 4
            assert Counter(proposed_requests) == Counter(request_ids.values())
            response = await service.post(
                "/reef/train",
                headers={**headers, "x-reef-scenario": "s1"},
                json={
                    "text": f"Keep ending final answers with {REPLY_MARKER}. Keep this rule concise.",
                    "session": "s1-wake",
                    "release_id": committed_releases["s1"],
                },
            )
            assert response.status in (200, 202), await response.text()
            wake_id = (await response.json())["agent_record_id"]
            async with asyncio.timeout(STEP_TIMEOUT_SECONDS):
                while True:
                    response = await service.get(
                        f"/reef/harness/requests/{wake_id}/progress", headers={**headers, "x-reef-scenario": "s1"}
                    )
                    assert response.status == 200, await response.text()
                    progress = await response.json()
                    if progress["settled"]:
                        assert progress["state"] != "failed"
                        break
                    await asyncio.sleep(0.05)
            restored = dispatcher.loaded_scenario("s1")
            assert restored is not None and restored is not original_instances["s1"]
            assert restored.training_mode == "manual"
            assert sum(wake_id in record.consumed_ids for record in restored.store.history()) == 1
            assert sum(request_ids["s1"] in record.consumed_ids for record in restored.store.history()) == 1
            request_ids["s1-wake"] = wake_id
            final = scheduler.list_scenario_admissions(1)
            assert {value.scenario_name for value in final if value.execution_allocated} == {"s1"}
            assert {value.scenario_name for value in final if value.state == "sleep"} == {"s2", "s3", "s4"}
            assert Counter(proposed_requests) == Counter(request_ids.values())
            assert controller.status(1)["allocated"] == 3
        finally:
            finish_proposals.set()
            scorer.round_finished.abort()
            await scheduler_client.close()
            await service.close()

    try:
        asyncio.run(run())
    finally:
        finish_proposals.set()
        scorer.round_finished.abort()
        dispatcher.close()
