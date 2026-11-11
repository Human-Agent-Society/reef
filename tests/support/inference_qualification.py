"""A reusable two-artifact cutover check through the real inference service.

The caller supplies an isolated runtime and artifacts with distinguishable outputs. Nothing here starts a
runtime on import or imports an engine library. Fault injection remains in the receiver contract tests; this
run checks actual outputs, route/header/record agreement, and a load-to-publication gap on a real integration.
``python -m tests.smoke.inference_cutover`` runs it against an integration that a factory provides.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, web

from reef.artifact import Artifact
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.inference.interfaces import EngineClients, InferenceRuntime, RuntimeContractError
from reef.service.inference_service import ARTIFACT_HEADER, RECORD_HEADER, build_inference_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage


class OutputOracle(ABC):
    """Identify weights from generated content, independently of metadata, labels and runtime load IDs."""

    @abstractmethod
    def artifact_id(self, response: Mapping[str, Any]) -> str:
        """Return the artifact identified by a distinguishable output, or raise for ambiguous output."""


@dataclass(frozen=True)
class InferenceQualification:
    """An isolated integration and two distinguishable artifacts for an explicit qualification run.

    ``runtime`` and ``engines`` are existing production contracts, not alternative provider interfaces.
    ``artifacts`` has two different IDs. ``oracle`` identifies output from ``payload`` without trusting a label.
    ``engine_version`` identifies the exact integration build; ``capabilities`` declares its supported features.
    ``owner_sequence`` is a fresh durable owner sequence for the isolated component, never a production takeover.
    The caller owns external service cleanup; the run opens and shuts down this runtime.
    """

    runtime: InferenceRuntime
    engines: EngineClients
    artifacts: tuple[Artifact, Artifact]
    oracle: OutputOracle
    payload: Mapping[str, Any]
    engine_version: str
    capabilities: tuple[str, ...]
    owner_sequence: int = 1

    def __post_init__(self) -> None:
        if len(self.artifacts) != 2 or self.artifacts[0].artifact_id == self.artifacts[1].artifact_id:
            raise ValueError("qualification needs two different artifacts")
        if not self.engine_version or self.owner_sequence <= 0:
            raise ValueError("qualification needs an engine version and a positive isolated owner sequence")


async def qualify_inference(
    case: InferenceQualification, directory: Path, *, samples_per_artifact: int = 5
) -> dict[str, Any]:
    """Verify completed outputs and stale routes; report what the run exercised.

    Every success must agree with its header and durable record. The gap deliberately retains A's route after
    B's activation: a response must still be A or fail without a receipt. Probes run during the cutover, and each
    of their answers must pass the same checks or fail without a receipt.
    """
    if isinstance(samples_per_artifact, bool) or samples_per_artifact < 1:
        raise ValueError("samples_per_artifact must be positive")
    scenario = "qualification"
    token = uuid4().hex
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(directory / "records")
    table = coordination.served_models
    app = build_inference_app(
        table, storage, case.engines, tokens=token, timeout_seconds=case.runtime.inference_timeout_s
    )
    runner = web.AppRunner(app)
    completed = 0
    stop = asyncio.Event()
    probe_task: asyncio.Task[None] | None = None

    def publish(served: ServedArtifact) -> None:
        lease = coordination.control_lease.acquire("qualification")
        if lease is None:
            raise RuntimeContractError("qualification could not acquire its isolated route lease")
        for name in (ACTIVE, served.artifact_id):
            table.write(ScenarioModel(scenario, name), served, control_lease_number=lease.number)

    def check_record(record_id: str, actual: str) -> None:
        store = storage.open(scenario)
        try:
            records = [record for record in store.records.replay(scenario) if record.agent_record_id == record_id]
        finally:
            store.close()
        if len(records) != 1 or records[0].artifact_id != actual:
            raise RuntimeContractError("generated output and durable record name different artifacts")
        response = records[0].payload.get("response")
        if not isinstance(response, Mapping) or case.oracle.artifact_id(response) != actual:
            raise RuntimeContractError("the stored response does not contain the selected artifact's output")

    async def activate(artifact: Artifact) -> ServedArtifact:
        await asyncio.to_thread(case.runtime.activate_checkpoint, artifact, scenario=scenario)
        served = case.runtime.served_artifact(artifact.artifact_id, scenario)
        if served is None:
            raise RuntimeContractError("activation did not establish a route")
        return served

    try:
        case.runtime.set_owner_sequence(case.owner_sequence)
        await asyncio.to_thread(case.runtime.open)
        first = await activate(case.artifacts[0])
        publish(first)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        headers = {"Authorization": f"Bearer {token}", "x-reef-scenario": scenario}
        async with ClientSession(timeout=ClientTimeout(total=case.runtime.inference_timeout_s)) as client:

            async def sample(
                *, expected: str | None = None, allow_failure: bool = False, route: str = "/v1/chat/completions"
            ) -> None:
                nonlocal completed
                async with client.post(url + route, headers=headers, json=dict(case.payload)) as response:
                    if response.status != 200:
                        await response.read()
                        if not allow_failure or response.status < 400 or RECORD_HEADER in response.headers:
                            raise RuntimeContractError(f"qualification request failed with HTTP {response.status}")
                        return
                    body = await response.json()
                    actual = case.oracle.artifact_id(body)
                    if response.headers.get(ARTIFACT_HEADER) != actual or (
                        expected is not None and expected != actual
                    ):
                        raise RuntimeContractError("generated output and selected artifact disagree")
                    record_id = response.headers.get(RECORD_HEADER)
                    if record_id is None:
                        raise RuntimeContractError("a completed request has no record receipt")
                    await asyncio.to_thread(check_record, record_id, actual)
                    completed += 1

            async def probe() -> None:
                while not stop.is_set():
                    await sample(allow_failure=True)
                    await asyncio.sleep(0.02)

            for _ in range(samples_per_artifact):
                await sample(expected=first.artifact_id)
            probe_task = asyncio.create_task(probe())
            cutover_started = time.monotonic()
            second = await activate(case.artifacts[1])
            # Deliberately exercise the old row before publishing B, including the cached-row path.
            await sample(expected=first.artifact_id, allow_failure=True)
            publish(second)
            cutover_seconds = time.monotonic() - cutover_started
            # A pinned refusal refreshes the row; adapters may still validly serve A until the cache refreshes.
            await asyncio.sleep(0)
            stop.set()
            await probe_task
            for _ in range(samples_per_artifact):
                # Verify B through its named route, independent of the active-row cache interval.
                await sample(
                    expected=second.artifact_id,
                    route=f"/reef/scenarios/{scenario}/artifacts/{quote(second.artifact_id, safe='')}/v1/chat/completions",
                )
        return {
            "engine_version": case.engine_version,
            "capabilities": list(case.capabilities),
            "artifacts": [artifact.artifact_id for artifact in case.artifacts],
            "completed_requests": completed,
            "cutover_seconds": cutover_seconds,
            "measurement": "small qualification workload; not a capacity benchmark",
            "exercised": ["buffered outputs", "recorded outputs", "load-to-publication gap", "promotion probes"],
            "not_exercised": [
                "normal streaming",
                "stale-owner operations",
                "mixed or wrong load IDs",
                "adapter retirement",
                "partial replica failure",
                "component restart",
                "stream interruption",
                "prefix cache reuse",
                "traffic splits",
            ],
        }
    finally:
        stop.set()
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
        if probe_task is not None:
            await asyncio.gather(probe_task, return_exceptions=True)
        try:
            await runner.cleanup()
        finally:
            try:
                await asyncio.to_thread(case.runtime.shutdown)
            finally:
                coordination.close()
