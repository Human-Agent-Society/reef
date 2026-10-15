"""Parse scheduler settings and save its service directory on normal shutdown."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml

from reef.scenario.scheduling import MultiTenantScheduler, SchedulerConfig, ServiceRegistration
from reef.service.scheduling import integer_value, object_value, optional_text, text_value, validate_endpoint

logger = logging.getLogger(__name__)


def configured_token(settings: dict[str, object], name: str) -> str | None:
    variable = optional_text(settings, name)
    if variable is None:
        return None
    token = os.environ.get(variable, "")
    if not token:
        raise ValueError(f"the environment variable named by {name} is empty")
    return token


def configured_id_map(settings: dict[str, object], name: str) -> dict[str, int]:
    values = object_value(settings.get(name, {}))
    mapping = {key: integer_value(values, key) for key in values}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{name} cannot map multiple old IDs to the same numeric ID")
    return mapping


def parse_scheduler_config(settings: dict[str, object], *, path: Path | None = None) -> SchedulerConfig:
    """Validate deployment settings into the scheduler's single configuration value."""
    if "queue_ttl_seconds" in settings:  # This should be removed in the future
        raise ValueError("queue_ttl_seconds is no longer supported; only idle_ttl_seconds is configurable")
    percentage = settings.get("max_logical_thread_percentage", 70.0)
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, int | float)
        or not math.isfinite(percentage)
        or not 0 < percentage <= 100
    ):
        raise ValueError("max_logical_thread_percentage must be a finite number greater than 0 and at most 100")
    logical_thread_count = os.cpu_count()
    if logical_thread_count is None:
        raise ValueError("cannot determine the host logical thread count")
    global_execution_units = math.floor(logical_thread_count * percentage / 100)
    if global_execution_units < 1:
        raise ValueError("max_logical_thread_percentage allows less than one execution unit on this host")
    if "global_execution_units" in settings:
        logger.warning(
            "global_execution_units in scheduler YAML is ignored; "
            "the budget is computed from logical threads and max_logical_thread_percentage"
        )
    tenant_values = settings.get("tenant_quotas")
    if not isinstance(tenant_values, dict) or not tenant_values:
        raise ValueError("tenant_quotas must be a nonempty mapping of numeric tenant IDs")
    quotas: dict[int, int] = {}
    for key, value in tenant_values.items():
        tenant_id = int(key) if isinstance(key, str) and key.isascii() and key.isdecimal() else key
        tenant_id = integer_value({"tenant_id": tenant_id}, "tenant_id")
        if tenant_id in quotas:
            raise ValueError("tenant_quotas contains duplicate numeric IDs")
        quotas[tenant_id] = integer_value({"quota": value}, "quota")
    recipe_values = object_value(settings.get("recipe_execution_units", {}))
    recipe_costs = {name: integer_value(recipe_values, name) for name in recipe_values}
    idle_ttl = settings.get("idle_ttl_seconds")
    if idle_ttl is not None and (isinstance(idle_ttl, bool) or not isinstance(idle_ttl, int | float)):
        raise ValueError("idle_ttl_seconds must be a positive duration in seconds")
    services = settings.get("services", [])
    if not isinstance(services, list):
        raise ValueError("services must be a list")
    registrations: dict[int, ServiceRegistration] = {}
    control_token_envs: dict[int, str] = {}
    for raw in services:
        service = object_value(raw)
        if "control_token" in service:
            raise ValueError("service snapshots accept control_token_env references, not control_token values")
        recipe = text_value(service, "recipe")
        revision = service.get("config_revision", "")
        if not isinstance(revision, str):
            raise ValueError("config_revision must be a string")
        # Older static directories obtained the service cost from the recipe mapping.
        cost = service.get("standard_execution_units", recipe_costs.get(recipe))
        registration = ServiceRegistration(
            service_id=integer_value(service, "service_id"),
            service_name=text_value(service, "service_name"),
            recipe=recipe,
            service_url=text_value(service, "service_url"),
            standard_execution_units=integer_value({"standard_execution_units": cost}, "standard_execution_units"),
            config_revision=revision,
        )
        validate_endpoint(registration.service_url)
        if registration.service_id in registrations:
            raise ValueError("services contains a duplicate service ID")
        registrations[registration.service_id] = registration
        variable = optional_text(service, "control_token_env")
        if variable is not None:
            control_token_envs[registration.service_id] = variable
    config = SchedulerConfig(
        global_execution_units=global_execution_units,
        tenant_quotas=quotas,
        recipe_execution_units=recipe_costs,
        idle_ttl_seconds=None if idle_ttl is None else float(idle_ttl),
        database=Path(text_value(settings, "database")),
        services=tuple(registrations.values()),
        control_token_envs=control_token_envs,
        token_env=optional_text(settings, "token_env"),
        tenant_id_map=configured_id_map(settings, "tenant_id_map"),
        service_id_map=configured_id_map(settings, "service_id_map"),
        path=None if path is None else path.resolve(),
        max_logical_thread_percentage=float(percentage),
    )
    logger.info(
        "scheduler execution budget: logical_threads=%s max_logical_thread_percentage=%s global_execution_units=%s",
        logical_thread_count,
        percentage,
        global_execution_units,
    )
    return config


def save_scheduler_config(scheduler: MultiTenantScheduler) -> None:
    """Atomically save the current directory and configuration without raw credentials."""
    config = scheduler.config
    if config.path is None:
        return
    with scheduler.lock:
        services: list[dict[str, object]] = []
        for registration in scheduler.service_registrations():
            service: dict[str, object] = asdict(registration)
            variable = config.control_token_envs.get(registration.service_id)
            if variable is not None:
                service["control_token_env"] = variable
            services.append(service)
        settings: dict[str, object] = {
            "database": str(config.database),
            "max_logical_thread_percentage": config.max_logical_thread_percentage,
            "tenant_quotas": dict(config.tenant_quotas),
            "recipe_execution_units": dict(scheduler.recipe_execution_units),
            "idle_ttl_seconds": config.idle_ttl_seconds,
            "services": services,
        }
    if config.token_env is not None:
        settings["token_env"] = config.token_env
    if config.tenant_id_map:
        settings["tenant_id_map"] = config.tenant_id_map
    if config.service_id_map:
        settings["service_id_map"] = config.service_id_map
    descriptor, temporary_name = tempfile.mkstemp(
        dir=config.path.parent, prefix=f".{config.path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config.path)
    finally:
        temporary.unlink(missing_ok=True)
