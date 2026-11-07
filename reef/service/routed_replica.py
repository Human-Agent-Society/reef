"""Routed chat completions in the inference service, beside a Semantic Router that another system runs.

``reef.routing_deployment.lifecycle: external`` selects this mode. Each inference replica calls one router on its own
loopback. The replica serves a request of a manifest Scenario under the routing rules, and every other request on the
ordinary inference routes. A policy change is a new replica: the replica reads its manifest once.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace

from aiohttp import web
from aiohttp.typedefs import Handler

from reef.core.records_types import AgentRecord
from reef.observability.operations import OperationMetrics
from reef.recipe import Recipe
from reef.service.deploy.routing.policy import RouterManagementClient, RouterPolicyError
from reef.service.deploy.routing.renderer import MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV
from reef.service.deploy.service_config import ServiceConfig
from reef.service.inference_service import ServedModelsCache
from reef.service.lifecycle import PendingWrites
from reef.service.routed_inference import InferenceRetryPolicy, RoutedInference, RoutedRecords, RoutedScenario
from reef.service.routes.routed_chat import serve_routed_chat
from reef.service.routing import RoutingAdapter, RoutingRequestError, is_router_config_hash
from reef.service.routing_preview import PreviewSettings, RoutingPreviewClient
from reef.service.wire import SCENARIO_HEADER
from reef.storage.scenario import ScenarioStorage

logger = logging.getLogger(__name__)

#: How often a replica reads its router's health and runtime hash.
ROUTER_CHECK_INTERVAL_SECONDS = 5.0


class ReplicaRecords(RoutedRecords):
    """The Scenarios and records of an inference replica, which reads ``cache`` and appends to ``storage``.

    The replica creates no Scenario, as for its ordinary records. ``operations`` measures the replica's routed requests
    over all Scenarios.
    """

    def __init__(self, cache: ServedModelsCache, storage: ScenarioStorage) -> None:
        self.cache = cache
        self.storage = storage
        self.operations = OperationMetrics(("serve/request",))

    def open_scenario(self, name: str) -> RoutedScenario:
        return RoutedScenario(self.operations, self.cache.serving_config(name))

    def accept_record(self, item: AgentRecord) -> AgentRecord:
        store = self.storage.open(item.scenario)
        try:
            return store.records.append_result(item).item
        finally:
            store.close()


class ReplicaRouter:
    """One replica's view of its router: the policy that requests use, and whether the replica can take traffic.

    Values: ``routing`` is the adapter of the replica's manifest. Its ``router_config_hash`` is the runtime that
    ``check`` last verified, or None before the first success. ``ready`` is true exactly when the last ``check``
    succeeded. Purpose: ``add_replica_routing`` makes it; its middleware and ``/readyz`` read it.

    ``check`` accepts a new runtime hash only after every enabled target and one preview per Scenario pass, so a router
    restart does not need a new replica.
    """

    def __init__(
        self,
        routing: RoutingAdapter,
        routed: RoutedInference,
        preview: RoutingPreviewClient,
        management: RouterManagementClient,
    ) -> None:
        self.routing = routing
        self.routed = routed
        self.preview = preview
        self.management = management
        self.ready = False

    async def verified_runtime_hash(self) -> str | None:
        """The hash of the runtime that the router serves, or None when it is not ready or not active."""
        runtime = await asyncio.to_thread(self.management.runtime_hashes)
        is_ready = await asyncio.to_thread(self.management.ready)
        if (
            not is_ready
            or runtime.activation_status != "active"
            or runtime.active_runtime_hash != runtime.generated_runtime_hash
            or runtime.active_runtime_hash is None
            or not is_router_config_hash(runtime.active_runtime_hash)
        ):
            return None
        return runtime.active_runtime_hash

    async def check(self) -> None:
        """Set ``ready`` from one reading of the router; publish a new runtime only after it passes."""
        try:
            runtime_hash = await self.verified_runtime_hash()
            if runtime_hash is not None and runtime_hash != self.routing.router_config_hash:
                candidate = replace(self.routing, router_config_hash=runtime_hash)
                await candidate.check_ready()
                await self.preview.check_ready(candidate)
                self.routing = candidate
                logger.info("routing uses router runtime %s", runtime_hash)
        except (RouterPolicyError, RoutingRequestError) as exc:
            logger.warning("routing check failed: %s", exc)
            runtime_hash = None
        ready = runtime_hash is not None
        if self.ready and not ready:
            logger.error("the router is not ready; this replica stops taking traffic")
        self.ready = ready

    async def watch(self) -> None:
        while True:
            await self.check()
            await asyncio.sleep(ROUTER_CHECK_INTERVAL_SECONDS)


def add_replica_routing(
    app: web.Application,
    settings: ServiceConfig,
    recipe: Recipe,
    records: ReplicaRecords,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Serve the Scenarios of ``settings.routing_manifest`` through their router on ``app``, and add ``/readyz``.

    Raise ``ValueError`` when the deployment cannot route: no service token, a recipe other than the base record-only
    Recipe, an invalid manifest, or a missing router token or provider credential.
    """
    if settings.routing_manifest is None:
        raise ValueError("replica routing requires reef.routing_manifest")
    environment = os.environ if environ is None else environ
    if not settings.tokens:
        raise ValueError("routing requires a service token")
    if type(recipe) is not Recipe or recipe.training_component is not None:
        raise ValueError("routing requires the base record-only Recipe")
    for name in (MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV):
        if not environment.get(name):
            raise ValueError(f"external routing requires environment variable {name}")
    policy = settings.routing_deployment
    if settings.port == policy.sr_management_port:
        raise ValueError("reef.port must differ from routing sr_management_port")
    routing = RoutingAdapter.from_manifest(
        settings.routing_manifest, router_config_hash=None, environ=environment, timeout_s=settings.inference_timeout_s
    )
    preview = RoutingPreviewClient(
        PreviewSettings(
            f"http://127.0.0.1:{policy.sr_management_port}",
            environment[SELECTION_TOKEN_ENV],
            policy.selection_timeout_seconds,
        )
    )
    writes = PendingWrites()
    retry = InferenceRetryPolicy(
        initial_s=settings.inference_retry_initial_s,
        max_s=settings.inference_retry_max_s,
        timeout_s=settings.inference_retry_timeout_s,
    )
    router = ReplicaRouter(
        routing,
        RoutedInference(records, preview, retry, writes),
        preview,
        RouterManagementClient(policy.sr_management_port, environment[MANAGEMENT_TOKEN_ENV]),
    )
    for_codex = recipe.harness_adapter == "codex"

    @web.middleware
    async def route_manifest_scenarios(request: web.Request, handler: Handler) -> web.StreamResponse:
        names = (name.strip() for name in request.headers.getall(SCENARIO_HEADER, ()))
        if (
            request.method == "POST"
            and request.path == "/v1/chat/completions"
            and any(name in routing.snapshot.scenarios for name in names)
        ):
            return await serve_routed_chat(request, router.routing, router.routed, for_codex=for_codex)
        return await handler(request)

    async def readiness(request: web.Request) -> web.Response:
        return web.json_response({"ready": router.ready}, status=200 if router.ready else 503)

    async def run_router_checks(app: web.Application) -> AsyncIterator[None]:
        await preview.start()
        watch = asyncio.create_task(router.watch())
        try:
            yield
        finally:
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
            # All requests have ended, so a write that has not finished is the last write of a cancelled request.
            if not await writes.wait(policy.finalization_timeout_seconds):
                logger.error("routed record writes did not finish; %d are still pending", len(writes.tasks))
            await preview.close()

    # After authentication and error translation, so a routed request gets both.
    app.middlewares.append(route_manifest_scenarios)
    app.router.add_get("/readyz", readiness)
    app.cleanup_ctx.append(run_router_checks)


__all__ = ["ROUTER_CHECK_INTERVAL_SECONDS", "ReplicaRecords", "ReplicaRouter", "add_replica_routing"]
