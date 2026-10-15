"""Run the standalone tenant scheduler and its in-process service directory.

Usage: python -m reef.service.scheduler --config scheduler.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import yaml
from aiohttp import web

from reef.scenario.scheduling import MultiTenantScheduler, SchedulingError
from reef.service.auth import create_authentication_middleware
from reef.service.errors import translate_errors
from reef.service.routes.scheduling import register_scheduler_routes
from reef.service.scheduler_config import configured_token, parse_scheduler_config
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import SchedulerHTTPClient, object_value
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore

logger = logging.getLogger(__name__)


def create_scheduler_app(controller: SchedulerController, *, token: str | None = None) -> web.Application:
    app = web.Application(middlewares=[create_authentication_middleware(token), translate_errors])

    async def health(request: web.Request) -> web.Response:
        unavailable = controller.stopping or controller.maintenance_error is not None
        return web.json_response(
            {"status": "unavailable" if unavailable else "ok", "reason": controller.maintenance_error},
            status=503 if unavailable else 200,
        )

    app.router.add_get("/healthz", health)
    register_scheduler_routes(app, controller)

    async def stop_admission(app: web.Application) -> None:
        controller.stop_accepting_requests()

    app.on_shutdown.append(stop_admission)

    async def lifecycle(app: web.Application) -> AsyncIterator[None]:
        stopped = asyncio.Event()

        async def maintain() -> None:
            while not stopped.is_set():
                try:
                    await asyncio.to_thread(controller.reconcile)
                except Exception as exc:
                    controller.maintenance_error = f"scheduler maintenance failed ({type(exc).__name__})"
                    logger.error("%s; new admission is stopped", controller.maintenance_error)
                    controller.stop_accepting_requests()
                    return
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=1.0)
                except TimeoutError:
                    continue

        task: asyncio.Task[None] | None = None
        initialized = False
        try:
            await asyncio.to_thread(controller.reconcile)
            task = asyncio.create_task(maintain())
            initialized = True
            yield
        finally:
            stopped.set()
            # Finish the in-flight control request before closing persistent state.
            try:
                if task is not None:
                    await task
            finally:
                await asyncio.to_thread(controller.close, save_snapshot=initialized)

    app.cleanup_ctx.append(lifecycle)
    return app


def build_scheduler(settings: dict[str, object], *, config_path: Path | None = None) -> SchedulerController:
    config = parse_scheduler_config(settings, path=config_path)
    controls: dict[int, SchedulerHTTPClient] = {}
    registrations = {registration.service_id: registration for registration in config.services}
    for service_id, variable in config.control_token_envs.items():
        token = configured_token({"control_token_env": variable}, "control_token_env")
        controls[service_id] = SchedulerHTTPClient(registrations[service_id].service_url, token=token)
    fingerprints = {
        registrations[new_id].legacy_registration_fingerprint(old_id): registrations[new_id].registration_fingerprint
        for old_id, new_id in config.service_id_map.items()
        if new_id in registrations
    }
    store = SQLiteSchedulingStore(
        config.database,
        tenant_id_map=config.tenant_id_map,
        service_id_map=config.service_id_map,
        registration_fingerprint_map=fingerprints,
    )
    try:
        scheduler = MultiTenantScheduler(store, config)
        for registration in registrations.values():
            scheduler.register_service(registration)
    except (ValueError, TypeError, SchedulingError):
        store.close()
        raise
    return SchedulerController(scheduler, controls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reef's tenant admission scheduler and service directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = object_value(yaml.safe_load(args.config.read_text()))
    token = configured_token(settings, "token_env")
    controller = build_scheduler(settings, config_path=args.config)
    web.run_app(create_scheduler_app(controller, token=token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
