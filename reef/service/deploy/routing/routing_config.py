"""Validated settings for the Semantic Router container that ``reef serve`` owns when routing is enabled."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.config import config_arguments, config_option, parse_config_values


@dataclass(frozen=True)
class RoutingDeploymentSettings:
    """The owned router container: identity, resources, preview limits and recovery policy.

    Values: ``deployment_id`` is None or 1 to 48 lowercase letters, digits or hyphens, starting with a letter
    or digit. ``restart_policy`` is ``fail_stop`` or ``bounded``; ``max_restarts`` is an int ≥ 0, and ≥ 1
    when bounded. ``restart_backoff_seconds`` is a non-empty tuple of finite numbers ≥ 0, in order, with
    repeats allowed. The ready, finalization and container stop timeouts and ``sr_cpus`` are finite and
    positive; ``drain_timeout_seconds`` is finite and ≥ 0. ``sr_management_port`` is an int from 1 to 65535,
    ``selection_timeout_seconds`` an int from 1 to 300, ``preview_max_concurrency`` an int from 1 to 4096,
    and ``sr_memory_bytes`` a positive int.
    Purpose: ``routing_deployment_settings`` makes it from ``reef.routing_deployment``; the controller and
    the renderer read it. Recovery attempt ``n`` (from 1) waits ``restart_backoff_seconds[min(n - 1,
    len - 1)]``, so the last delay repeats. These settings apply only when ``reef.routing_manifest`` is set;
    ``require_identity`` then returns ``deployment_id`` or refuses a deployment without one.
    """

    deployment_id: str | None = config_option(None, help="Stable local deployment identifier.")
    restart_policy: str = config_option("fail_stop", help="Failure policy: fail_stop or bounded.")
    max_restarts: int = config_option(2, help="Recovery attempts allowed during this explicit deployment.")
    restart_backoff_seconds: tuple[float, ...] = (1.0, 2.0)
    ready_timeout_seconds: float = config_option(120.0, help="Complete routing readiness deadline in seconds.")
    drain_timeout_seconds: float = config_option(30.0, help="Seconds allowed for accepted requests to drain.")
    finalization_timeout_seconds: float = config_option(5.0, help="Seconds allowed for pending record writes.")
    container_stop_timeout_seconds: float = config_option(10.0, help="Container stop command deadline in seconds.")
    sr_management_port: int = config_option(
        50053, help="Host loopback port of the router's authenticated HTTP API (preview and configuration)."
    )
    selection_timeout_seconds: int = config_option(
        10, help="Whole seconds the router may spend selecting for one request (its preview worker deadline)."
    )
    preview_max_concurrency: int = config_option(
        32, help="Router preview workers; more concurrent selections are refused with 429. Changes need a restart."
    )
    sr_cpus: float = config_option(4.0, help="Semantic Router container CPU limit.")
    sr_memory_bytes: int = config_option(4 * 1024**3, help="Semantic Router container memory limit in bytes.")

    def __post_init__(self) -> None:
        if self.deployment_id is not None and (
            not isinstance(self.deployment_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", self.deployment_id) is None
        ):
            raise ValueError("routing deployment_id must contain 1-48 lowercase letters, digits or hyphens")
        if self.restart_policy not in {"fail_stop", "bounded"}:
            raise ValueError("routing restart_policy must be fail_stop or bounded")
        if type(self.max_restarts) is not int or self.max_restarts < 0:
            raise ValueError("routing max_restarts must be a nonnegative integer")
        if self.restart_policy == "bounded" and self.max_restarts == 0:
            raise ValueError("bounded routing requires at least one recovery attempt")
        if type(self.restart_backoff_seconds) is not tuple:
            # A frozen value must not keep a list the caller can still change.
            raise ValueError("routing restart_backoff_seconds must be a tuple of numbers")
        if not self.restart_backoff_seconds or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in self.restart_backoff_seconds
        ):
            raise ValueError("routing restart_backoff_seconds must be a nonempty list of finite nonnegative numbers")
        for name, value in (
            ("ready_timeout_seconds", self.ready_timeout_seconds),
            ("finalization_timeout_seconds", self.finalization_timeout_seconds),
            ("container_stop_timeout_seconds", self.container_stop_timeout_seconds),
            ("sr_cpus", self.sr_cpus),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"routing {name} must be finite and positive")
        if (
            type(self.drain_timeout_seconds) not in (int, float)
            or not math.isfinite(self.drain_timeout_seconds)
            or self.drain_timeout_seconds < 0
        ):
            raise ValueError("routing drain_timeout_seconds must be finite and nonnegative")
        if type(self.sr_management_port) is not int or not 1 <= self.sr_management_port <= 65535:
            raise ValueError("routing sr_management_port must be an integer between 1 and 65535")
        if type(self.selection_timeout_seconds) is not int or not 1 <= self.selection_timeout_seconds <= 300:
            raise ValueError("routing selection_timeout_seconds must be a whole number of seconds from 1 to 300")
        if type(self.preview_max_concurrency) is not int or not 1 <= self.preview_max_concurrency <= 4096:
            raise ValueError("routing preview_max_concurrency must be an integer from 1 to 4096")
        if type(self.sr_memory_bytes) is not int or self.sr_memory_bytes <= 0:
            raise ValueError("routing sr_memory_bytes must be a positive integer")

    def require_identity(self) -> str:
        """The deployment identifier, which an enabled routing deployment must name."""
        if self.deployment_id is None:
            raise ValueError("reef.routing_manifest requires reef.routing_deployment.deployment_id")
        return self.deployment_id


def routing_deployment_settings(section: Mapping[str, object] | None) -> RoutingDeploymentSettings:
    """Parse the public object and reject unknown or misspelled settings before any process starts."""
    if section is None:
        return RoutingDeploymentSettings()
    if not isinstance(section, Mapping):
        raise ValueError("reef.routing_deployment must be an object")
    folded = {str(key).replace("-", "_"): value for key, value in section.items()}
    if len(folded) != len(section):
        raise ValueError("reef.routing_deployment contains duplicate field spellings")
    backoff = folded.pop("restart_backoff_seconds", (1.0, 2.0))
    if not isinstance(backoff, (list, tuple)):
        raise ValueError("routing restart_backoff_seconds must be a list of numbers")
    if any(type(value) not in (int, float) for value in backoff):
        raise ValueError("routing restart_backoff_seconds must be a list of numbers")
    values = parse_config_values(
        config_arguments(RoutingDeploymentSettings, prefix=("reef", "routing_deployment")), folded
    )
    return RoutingDeploymentSettings(**values, restart_backoff_seconds=tuple(float(value) for value in backoff))
